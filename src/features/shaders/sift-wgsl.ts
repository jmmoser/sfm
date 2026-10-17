/**
 * WGSL kernels of the WebGPU SIFT (src/features/sift-gpu.ts). They mirror the CPU
 * reference (sift-cpu.ts) operation by operation; see sift-common.ts for the
 * scale-space and descriptor conventions.
 *
 * Portability (iOS Safari / Apple GPUs): only core WebGPU, f32 math, no f32 atomics
 * (descriptor histograms use fixed-point u32 workgroup atomics), no vec3 in buffers,
 * workgroups <= 256 invocations and <= 10 KB workgroup memory, <= 4 storage buffers
 * per kernel. All parameter structs consist of u32 fields followed by f32 fields
 * (see ParamBlock in gpu-common.ts).
 */

/** Max blur radius of the horizontal pass (shared row tile of 256 + 2 * MAXR). */
export const BLUR_H_MAX_RADIUS = 128;
/** Max blur radius of the tiled vertical pass; larger radii use the direct kernel. */
export const BLUR_V_TILED_MAX_RADIUS = 24;
/** Words per detection candidate record. */
export const CANDIDATE_WORDS = 8;
/**
 * Fixed-point scale of the descriptor histogram accumulation (u32 workgroup atomics):
 * the largest power of two such that one padded bin cannot overflow 2^32 for 0..1
 * images. A bin collects samples from a 2x2-bin area (<= 4 (SBP + 1)^2 pixels, SBP =
 * 3 sigma_max) with gradient magnitude <= sqrt(2) (one-sided borders) and weights <= 1.
 * Default options (S = 3, sigma 1.6): 2^22 (resolution 2.4e-7).
 */
export function descriptorFixedPoint(sigma: number, S: number): number {
  const sbp = 3 * sigma * 2 ** ((S + 0.6) / S);
  const bound = 4 * (sbp + 1) ** 2 * Math.SQRT2;
  return 2 ** Math.min(24, Math.max(12, Math.floor(Math.log2(2 ** 32 / bound))));
}

const TWO_PI = '6.283185307179586';

/** Area-averaging / bilinear resample (separable tables, same as image.ts) from the input. */
export function resizeShader(kind: 'u8' | 'f32' | 'rgba'): string {
  const srcDecl =
    kind === 'rgba'
      ? '@group(0) @binding(3) var src: texture_2d<f32>;'
      : kind === 'u8'
        ? '@group(0) @binding(3) var<storage, read> src: array<u32>;'
        : '@group(0) @binding(3) var<storage, read> src: array<f32>;';
  const srcAt =
    kind === 'rgba'
      ? `fn srcAt(x: u32, y: u32) -> f32 {
  let c = textureLoad(src, vec2<i32>(i32(x), i32(y)), 0);
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}`
      : kind === 'u8'
        ? `fn srcAt(x: u32, y: u32) -> f32 {
  let i = y * p.sw + x;
  return f32((src[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu);
}`
        : `fn srcAt(x: u32, y: u32) -> f32 {
  return src[y * p.sw + x];
}`;
  return /* wgsl */ `
struct Params {
  sw: u32, sh: u32, dw: u32, dh: u32,
  tx: u32, ty: u32, oxi: u32, oxw: u32,
  oyi: u32, oyw: u32, dstOff: u32, scale: f32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> filt: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
${srcDecl}

${srcAt}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.dw || g.y >= p.dh) { return; }
  var acc = 0.0;
  for (var t = 0u; t < p.ty; t++) {
    let wy = bitcast<f32>(filt[p.oyw + g.y * p.ty + t]);
    if (wy == 0.0) { continue; }
    let iy = filt[p.oyi + g.y * p.ty + t];
    var row = 0.0;
    for (var s = 0u; s < p.tx; s++) {
      let wx = bitcast<f32>(filt[p.oxw + g.x * p.tx + s]);
      row += wx * srcAt(filt[p.oxi + g.x * p.tx + s], iy);
    }
    acc += wy * row;
  }
  dst[p.dstOff + g.y * p.dw + g.x] = acc * p.scale;
}
`;
}

/** 2x upsample: up(2i) = in(i), up(2i+1) = (in(i) + in(i+1)) / 2 (edge clamped), within one buffer. */
export const UPSAMPLE_WGSL = /* wgsl */ `
struct Params { w: u32, h: u32, srcOff: u32, dstOff: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

fn hval(y: u32, X: u32) -> f32 {
  let x = X >> 1u;
  let o = p.srcOff + y * p.w;
  let a = buf[o + x];
  if ((X & 1u) == 0u) { return a; }
  return 0.5 * (a + buf[o + min(x + 1u, p.w - 1u)]);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let W2 = 2u * p.w;
  if (g.x >= W2 || g.y >= 2u * p.h) { return; }
  let y = g.y >> 1u;
  var v = hval(y, g.x);
  if ((g.y & 1u) == 1u) { v = 0.5 * (v + hval(min(y + 1u, p.h - 1u), g.x)); }
  buf[p.dstOff + g.y * W2 + g.x] = v;
}
`;

/** Horizontal Gaussian pass (replicated borders) with a shared-memory row tile. */
export const BLUR_H_WGSL = /* wgsl */ `
const WG: u32 = 256u;
const MAXR: u32 = ${BLUR_H_MAX_RADIUS}u;
struct Params { w: u32, h: u32, r: u32, kOff: u32, srcOff: u32, dstOff: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> kern: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
var<workgroup> tile: array<f32, ${256 + 2 * BLUR_H_MAX_RADIUS}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let r = min(p.r, MAXR);
  let y = wid.y;
  let x0 = wid.x * WG;
  let n = WG + 2u * r;
  let row = p.srcOff + y * p.w;
  let wm1 = i32(p.w) - 1;
  for (var i = li; i < n; i += WG) {
    let xs = clamp(i32(x0 + i) - i32(r), 0, wm1);
    tile[i] = src[row + u32(xs)];
  }
  workgroupBarrier();
  let x = x0 + li;
  if (x < p.w) {
    let c = li + r;
    var acc = kern[p.kOff] * tile[c];
    for (var j = 1u; j <= r; j++) {
      acc += kern[p.kOff + j] * (tile[c - j] + tile[c + j]);
    }
    dst[p.dstOff + y * p.w + x] = acc;
  }
}
`;

/** Horizontal Gaussian pass, direct global loads (any radius; used on software adapters). */
export const BLUR_H_DIRECT_WGSL = /* wgsl */ `
struct Params { w: u32, h: u32, r: u32, kOff: u32, srcOff: u32, dstOff: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> kern: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64, 4)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.w || g.y >= p.h) { return; }
  let x = i32(g.x);
  let row = p.srcOff + g.y * p.w;
  let wm1 = i32(p.w) - 1;
  var acc = kern[p.kOff] * src[row + g.x];
  for (var j = 1u; j <= p.r; j++) {
    let xa = u32(max(x - i32(j), 0));
    let xb = u32(min(x + i32(j), wm1));
    acc += kern[p.kOff + j] * (src[row + xa] + src[row + xb]);
  }
  dst[p.dstOff + g.y * p.w + g.x] = acc;
}
`;

/** Vertical Gaussian pass with a shared-memory column tile (radius <= BLUR_V_TILED_MAX_RADIUS). */
export const BLUR_V_TILED_WGSL = /* wgsl */ `
const MAXR: u32 = ${BLUR_V_TILED_MAX_RADIUS}u;
struct Params { w: u32, h: u32, r: u32, kOff: u32, srcOff: u32, dstOff: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> kern: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
var<workgroup> vt: array<f32, ${(32 + 2 * BLUR_V_TILED_MAX_RADIUS) * 32}>;

@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = min(p.r, MAXR);
  let x = wid.x * 32u + lid.x;
  let y0 = wid.y * 32u;
  let rows = 32u + 2u * r;
  let xc = min(x, p.w - 1u);
  let hm1 = i32(p.h) - 1;
  for (var j = lid.y; j < rows; j += 8u) {
    let ys = clamp(i32(y0 + j) - i32(r), 0, hm1);
    vt[j * 32u + lid.x] = src[p.srcOff + u32(ys) * p.w + xc];
  }
  workgroupBarrier();
  if (x < p.w) {
    for (var k = 0u; k < 4u; k++) {
      let ly = lid.y + 8u * k;
      let y = y0 + ly;
      if (y < p.h) {
        let c = (ly + r) * 32u + lid.x;
        var acc = kern[p.kOff] * vt[c];
        for (var j = 1u; j <= r; j++) {
          acc += kern[p.kOff + j] * (vt[c - j * 32u] + vt[c + j * 32u]);
        }
        dst[p.dstOff + y * p.w + x] = acc;
      }
    }
  }
}
`;

/** Vertical Gaussian pass, direct (any radius). */
export const BLUR_V_DIRECT_WGSL = /* wgsl */ `
struct Params { w: u32, h: u32, r: u32, kOff: u32, srcOff: u32, dstOff: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> kern: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64, 4)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.w || g.y >= p.h) { return; }
  let x = g.x;
  let y = i32(g.y);
  let hm1 = i32(p.h) - 1;
  var acc = kern[p.kOff] * src[p.srcOff + g.y * p.w + x];
  for (var j = 1u; j <= p.r; j++) {
    let ya = u32(max(y - i32(j), 0));
    let yb = u32(min(y + i32(j), hm1));
    acc += kern[p.kOff + j] * (src[p.srcOff + ya * p.w + x] + src[p.srcOff + yb * p.w + x]);
  }
  dst[p.dstOff + g.y * p.w + x] = acc;
}
`;

/** Next octave base: next(i, j) = cur(2i, 2j). */
export const DECIMATE_WGSL = /* wgsl */ `
struct Params { dw: u32, dh: u32, sw: u32, srcOff: u32, dstOff: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.dw || g.y >= p.dh) { return; }
  dst[p.dstOff + g.y * p.dw + g.x] = src[p.srcOff + 2u * g.y * p.sw + 2u * g.x];
}
`;

/**
 * DoG extrema (3x3x3, strict), Taylor refinement with position moves (<= 5 iterations,
 * offsets <= 0.6), contrast + edge rejection; survivors are appended to `cand` through
 * an atomic counter (the counter keeps counting past `capacity` so the host can grow
 * the buffer and re-run). DoG values are formed on the fly: D_s = L_{s+1} - L_s.
 * Record (8 words): xr, yr, sr, value (f32 bits), xi | yi << 16, si | s0 << 8 | octIdx << 16,
 * x0 | y0 << 16, 0 — final integer location (dedupe key) and start location (CPU order).
 */
export const DETECT_WGSL = /* wgsl */ `
struct Params {
  w: u32, h: u32, S: u32, N: u32,
  octIdx: u32, capacity: u32,
  pre: f32, peak: f32, edge: f32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> cand: array<u32>;
@group(0) @binding(3) var<storage, read_write> counter: array<atomic<u32>>;

fn Dat(s: i32, x: i32, y: i32) -> f32 {
  let i = u32(y) * p.w + u32(x);
  return G[u32(s + 1) * p.N + i] - G[u32(s) * p.N + i];
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let w = i32(p.w);
  let h = i32(p.h);
  let S = i32(p.S);
  let x0 = i32(gid.x) + 1;
  let y0 = i32(gid.y) + 1;
  let s0 = i32(gid.z) + 1;
  if (x0 > w - 2 || y0 > h - 2 || s0 > S) { return; }
  let v = Dat(s0, x0, y0);
  let isMax = v > p.pre;
  if (!isMax && !(v < -p.pre)) { return; }
  for (var ds = -1; ds <= 1; ds++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        if (ds == 0 && dy == 0 && dx == 0) { continue; }
        let n = Dat(s0 + ds, x0 + dx, y0 + dy);
        if (isMax) {
          if (!(v > n)) { return; }
        } else {
          if (!(v < n)) { return; }
        }
      }
    }
  }

  var x = x0;
  var y = y0;
  var s = s0;
  var converged = false;
  var vv = 0.0;
  var dX = 0.0; var dY = 0.0; var dS = 0.0;
  var dxx = 0.0; var dyy = 0.0; var dxy = 0.0;
  var bx = 0.0; var by = 0.0; var bs = 0.0;
  for (var iter = 0; iter < 5; iter++) {
    vv = Dat(s, x, y);
    let xp = Dat(s, x + 1, y);
    let xm = Dat(s, x - 1, y);
    let yp = Dat(s, x, y + 1);
    let ym = Dat(s, x, y - 1);
    let sp = Dat(s + 1, x, y);
    let sm = Dat(s - 1, x, y);
    dX = 0.5 * (xp - xm);
    dY = 0.5 * (yp - ym);
    dS = 0.5 * (sp - sm);
    dxx = xp + xm - 2.0 * vv;
    dyy = yp + ym - 2.0 * vv;
    let dss = sp + sm - 2.0 * vv;
    dxy = 0.25 * (Dat(s, x + 1, y + 1) - Dat(s, x - 1, y + 1) - Dat(s, x + 1, y - 1) + Dat(s, x - 1, y - 1));
    let dxs = 0.25 * (Dat(s + 1, x + 1, y) - Dat(s + 1, x - 1, y) - Dat(s - 1, x + 1, y) + Dat(s - 1, x - 1, y));
    let dys = 0.25 * (Dat(s + 1, x, y + 1) - Dat(s + 1, x, y - 1) - Dat(s - 1, x, y + 1) + Dat(s - 1, x, y - 1));
    let c00 = dyy * dss - dys * dys;
    let c01 = dxs * dys - dxy * dss;
    let c02 = dxy * dys - dxs * dyy;
    let det = dxx * c00 + dxy * c01 + dxs * c02;
    if (!(abs(det) > 1e-30)) { return; }
    let c11 = dxx * dss - dxs * dxs;
    let c12 = dxs * dxy - dxx * dys;
    let c22 = dxx * dyy - dxy * dxy;
    let id = -1.0 / det;
    bx = (c00 * dX + c01 * dY + c02 * dS) * id;
    by = (c01 * dX + c11 * dY + c12 * dS) * id;
    bs = (c02 * dX + c12 * dY + c22 * dS) * id;
    if (abs(bx) <= 0.6 && abs(by) <= 0.6 && abs(bs) <= 0.6) {
      converged = true;
      break;
    }
    if (!(abs(bx) < 3.0e38 && abs(by) < 3.0e38 && abs(bs) < 3.0e38)) { return; }
    var mx = 0;
    if (bx > 0.6 && x < w - 2) { mx = 1; } else if (bx < -0.6 && x > 1) { mx = -1; }
    var my = 0;
    if (by > 0.6 && y < h - 2) { my = 1; } else if (by < -0.6 && y > 1) { my = -1; }
    var ms = 0;
    if (bs > 0.6 && s < S) { ms = 1; } else if (bs < -0.6 && s > 1) { ms = -1; }
    if (mx == 0 && my == 0 && ms == 0) { break; }
    x += mx;
    y += my;
    s += ms;
  }
  if (!converged) { return; }
  let value = vv + 0.5 * (dX * bx + dY * by + dS * bs);
  if (!(abs(value) >= p.peak)) { return; }
  let det2 = dxx * dyy - dxy * dxy;
  let tr = dxx + dyy;
  let r1 = p.edge + 1.0;
  if (!(det2 > 0.0) || tr * tr * p.edge >= r1 * r1 * det2) { return; }
  let xr = f32(x) + bx;
  let yr = f32(y) + by;
  let sr = f32(s) + bs;
  if (xr < 0.0 || xr > f32(w - 1) || yr < 0.0 || yr > f32(h - 1)) { return; }
  let k = atomicAdd(&counter[0], 1u);
  if (k >= p.capacity) { return; }
  let o = k * ${CANDIDATE_WORDS}u;
  cand[o] = bitcast<u32>(xr);
  cand[o + 1u] = bitcast<u32>(yr);
  cand[o + 2u] = bitcast<u32>(sr);
  cand[o + 3u] = bitcast<u32>(value);
  cand[o + 4u] = u32(x) | (u32(y) << 16u);
  cand[o + 5u] = u32(s) | (u32(s0) << 8u) | (p.octIdx << 16u);
  cand[o + 6u] = u32(x0) | (u32(y0) << 16u);
  cand[o + 7u] = 0u;
}
`;

/** Gradient (magnitude, angle in [0, 2π)) of one Gaussian level, CPU border rules. */
const GRAD_FN = /* wgsl */ `
fn gradAt(x: i32, y: i32) -> vec2f {
  let w = i32(p.w);
  let h = i32(p.h);
  let row = p.levelOff + u32(y) * p.w;
  var gx = 0.0;
  if (x > 0 && x < w - 1) {
    gx = 0.5 * (G[row + u32(x + 1)] - G[row + u32(x - 1)]);
  } else if (w > 1) {
    if (x == 0) { gx = G[row + 1u] - G[row]; } else { gx = G[row + u32(x)] - G[row + u32(x - 1)]; }
  }
  var gy = 0.0;
  if (y > 0 && y < h - 1) {
    gy = 0.5 * (G[row + p.w + u32(x)] - G[row - p.w + u32(x)]);
  } else {
    var up = row;
    var dn = row;
    if (y > 0) { up = row - p.w; }
    if (y < h - 1) { dn = row + p.w; }
    gy = G[dn + u32(x)] - G[up + u32(x)];
  }
  let mag = sqrt(gx * gx + gy * gy);
  var ang = 0.0;
  if (mag > 0.0) {
    ang = atan2(gy, gx);
    if (ang < 0.0) { ang += ${TWO_PI}; }
    if (ang >= ${TWO_PI}) { ang = 0.0; }
  }
  return vec2f(mag, ang);
}
`;

/**
 * Orientation assignment, one workgroup (64) per keypoint: per-thread partial
 * 36-bin histograms in workgroup memory (no float atomics), reduction, six circular
 * box smoothings, peaks >= 0.8 max with parabolic interpolation, sorted by value.
 * Output: ori[q] = count, ori[angOff + q * maxOri + a] = angle bits.
 */
export const ORIENT_WGSL = /* wgsl */ `
struct Params { w: u32, h: u32, levelOff: u32, base: u32, count: u32, maxOri: u32, angOff: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read> kps: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> ori: array<u32>;

const STRIDE: u32 = 37u;
var<workgroup> part: array<f32, ${64 * 37}>;
var<workgroup> hs: array<f32, 72>;
${GRAD_FN}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
  let k = wid.x + wid.y * nwg.x;
  if (k >= p.count) { return; }
  let q = p.base + k;
  let kp = kps[q];
  let x = kp.x;
  let y = kp.y;
  let sigma = kp.z;
  let w = i32(p.w);
  let h = i32(p.h);
  let sigmaw = 1.5 * sigma;
  let W = max(1, i32(floor(3.0 * sigmaw)));
  let xi = i32(floor(x + 0.5));
  let yi = i32(floor(y + 0.5));
  let pb = li * STRIDE;
  for (var b = 0u; b < 36u; b++) { part[pb + b] = 0.0; }
  if (xi >= 0 && xi < w && yi >= 0 && yi < h) {
    let dx0 = max(-W, -xi);
    let dx1 = min(W, w - 1 - xi);
    let dy0 = max(-W, -yi);
    let dy1 = min(W, h - 1 - yi);
    let nx = dx1 - dx0 + 1;
    let total = nx * (dy1 - dy0 + 1);
    let g = -1.0 / (2.0 * sigmaw * sigmaw);
    let r2max = f32(W * W) + 0.6;
    for (var t = i32(li); t < total; t += 64) {
      let dy = dy0 + t / nx;
      let dx = dx0 + t % nx;
      let tx = f32(xi + dx) - x;
      let ty = f32(yi + dy) - y;
      if (tx * tx + ty * ty >= r2max) { continue; }
      let gr = gradAt(xi + dx, yi + dy);
      if (!(gr.x > 0.0)) { continue; }
      let wgt = gr.x * exp(tx * tx * g) * exp(ty * ty * g);
      let fbin = gr.y * (36.0 / ${TWO_PI}) - 0.5;
      let bf = floor(fbin);
      let rb = fbin - bf;
      var b0 = i32(bf);
      if (b0 < 0) { b0 += 36; }
      b0 = min(b0, 35);
      var b1 = b0 + 1;
      if (b1 == 36) { b1 = 0; }
      part[pb + u32(b0)] += (1.0 - rb) * wgt;
      part[pb + u32(b1)] += rb * wgt;
    }
  }
  workgroupBarrier();
  if (li < 36u) {
    var s = 0.0;
    for (var t = 0u; t < 64u; t++) { s += part[t * STRIDE + li]; }
    hs[li] = s;
  }
  workgroupBarrier();
  for (var it = 0u; it < 6u; it++) {
    let src = (it & 1u) * 36u;
    let dst = 36u - src;
    if (li < 36u) {
      let im = select(li - 1u, 35u, li == 0u);
      let ip = select(li + 1u, 0u, li == 35u);
      hs[dst + li] = (hs[src + im] + hs[src + li] + hs[src + ip]) / 3.0;
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    var maxh = 0.0;
    for (var i = 0u; i < 36u; i++) { maxh = max(maxh, hs[i]); }
    var n = 0u;
    if (maxh > 0.0) {
      let thr = 0.8 * maxh;
      var pv: array<f32, 18>;
      var pa: array<f32, 18>;
      var np = 0u;
      for (var i = 0u; i < 36u; i++) {
        let h0 = hs[i];
        if (h0 < thr) { continue; }
        let hm = hs[select(i - 1u, 35u, i == 0u)];
        let hp = hs[select(i + 1u, 0u, i == 35u)];
        if (!(h0 > hm && h0 > hp)) { continue; }
        let den = hp + hm - 2.0 * h0;
        var di = 0.0;
        if (den < 0.0) { di = (-0.5 * (hp - hm)) / den; }
        var th = (${TWO_PI} * (f32(i) + di + 0.5)) / 36.0;
        if (th < 0.0) { th += ${TWO_PI}; } else if (th >= ${TWO_PI}) { th -= ${TWO_PI}; }
        if (np >= 18u) { continue; }
        var kk = np;
        np++;
        loop {
          if (kk == 0u) { break; }
          if (!(pv[kk - 1u] < h0)) { break; }
          pv[kk] = pv[kk - 1u];
          pa[kk] = pa[kk - 1u];
          kk--;
        }
        pv[kk] = h0;
        pa[kk] = th;
      }
      n = min(np, p.maxOri);
      for (var a = 0u; a < n; a++) { ori[p.angOff + q * p.maxOri + a] = bitcast<u32>(pa[a]); }
    }
    ori[q] = n;
  }
}
`;

/**
 * 4x4x8 SIFT descriptor (VLFeat style: magnification 3, Gaussian window sigma = 2
 * bins, trilinear interpolation into a padded 6x6x9 histogram), one workgroup (64)
 * per oriented keypoint slot (q, a); slots with a >= count[q] exit immediately.
 * Accumulation in fixed point with u32 workgroup atomics (scale p.fix, see
 * descriptorFixedPoint), then
 * L2 normalise, clamp 0.2, renormalise, RootSIFT (L1 + sqrt), quantise
 * min(255, round(512 v)) and pack 4 bytes per u32 (byte k of word i = element 4i + k).
 */
export const DESCRIBE_WGSL = /* wgsl */ `
struct Params { w: u32, h: u32, levelOff: u32, base: u32, count: u32, maxOri: u32, angOff: u32, fix: f32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read> kps: array<vec4f>;
@group(0) @binding(3) var<storage, read> ori: array<u32>;
@group(0) @binding(4) var<storage, read_write> desc: array<u32>;
var<workgroup> Ph: array<atomic<u32>, 324>;
var<workgroup> hist: array<f32, 128>;
var<workgroup> red: array<f32, 64>;
var<workgroup> qb: array<u32, 128>;
var<workgroup> wgN: u32;
${GRAD_FN}

fn fx32(v: f32) -> u32 { return u32(v * p.fix + 0.5); }

fn reduce64(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (li < s) { red[li] = red[li] + red[li + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
  let k = wid.x + wid.y * nwg.x;
  if (k >= p.count * p.maxOri) { return; }
  let q = p.base + k / p.maxOri;
  let a = k % p.maxOri;
  if (li == 0u) { wgN = ori[q]; }
  let nOri = workgroupUniformLoad(&wgN);
  if (a >= nOri) { return; }
  let kp = kps[q];
  let x = kp.x;
  let y = kp.y;
  let sigma = kp.z;
  let theta = bitcast<f32>(ori[p.angOff + q * p.maxOri + a]);
  for (var i = li; i < 324u; i += 64u) { atomicStore(&Ph[i], 0u); }
  workgroupBarrier();

  let w = i32(p.w);
  let h = i32(p.h);
  let SBP = 3.0 * sigma;
  let W = i32(floor(1.4142135623730951 * SBP * 2.5 + 0.5));
  let xi = i32(floor(x + 0.5));
  let yi = i32(floor(y + 0.5));
  let ct = cos(theta);
  let st = sin(theta);
  let isbp = 1.0 / SBP;
  let dx0 = max(-W, -xi);
  let dx1 = min(W, w - 1 - xi);
  let dy0 = max(-W, -yi);
  let dy1 = min(W, h - 1 - yi);
  if (dx0 <= dx1 && dy0 <= dy1) {
    let nxs = dx1 - dx0 + 1;
    let total = nxs * (dy1 - dy0 + 1);
    // Contiguous runs per thread: neighbouring threads work ~total/64 samples apart,
    // which spreads their atomics over different histogram bins.
    let per = (total + 63) / 64;
    let t0 = i32(li) * per;
    let t1 = min(t0 + per, total);
    let sw = 2.0 * SBP;
    let g = -1.0 / (2.0 * sw * sw);
    for (var t = t0; t < t1; t++) {
      let dy = dy0 + t / nxs;
      let dx = dx0 + t % nxs;
      let tx = f32(xi + dx) - x;
      let ty = f32(yi + dy) - y;
      let nx = (ct * tx + st * ty) * isbp;
      let ny = (ct * ty - st * tx) * isbp;
      if (nx <= -2.5 || nx >= 2.5 || ny <= -2.5 || ny >= 2.5) { continue; }
      let gr = gradAt(xi + dx, yi + dy);
      if (!(gr.x > 0.0)) { continue; }
      let m = gr.x * exp(tx * tx * g) * exp(ty * ty * g);
      var an = gr.y - theta;
      if (an < 0.0) { an += ${TWO_PI}; }
      let nt = an * (8.0 / ${TWO_PI});
      let fx = nx + 1.5;
      let fy = ny + 1.5;
      let bxf = clamp(floor(fx), -1.0, 3.0);
      let byf = clamp(floor(fy), -1.0, 3.0);
      let btf = clamp(floor(nt), 0.0, 7.0);
      let rx = fx - bxf;
      let ry = fy - byf;
      let rt = nt - btf;
      let base = u32((i32(byf) + 1) * 54 + (i32(bxf) + 1) * 9 + i32(btf));
      let m0 = m * (1.0 - ry);
      let m1 = m * ry;
      let m00 = m0 * (1.0 - rx);
      let m01 = m0 * rx;
      let m10 = m1 * (1.0 - rx);
      let m11 = m1 * rx;
      let rt0 = 1.0 - rt;
      atomicAdd(&Ph[base], fx32(m00 * rt0));
      atomicAdd(&Ph[base + 1u], fx32(m00 * rt));
      atomicAdd(&Ph[base + 9u], fx32(m01 * rt0));
      atomicAdd(&Ph[base + 10u], fx32(m01 * rt));
      atomicAdd(&Ph[base + 54u], fx32(m10 * rt0));
      atomicAdd(&Ph[base + 55u], fx32(m10 * rt));
      atomicAdd(&Ph[base + 63u], fx32(m11 * rt0));
      atomicAdd(&Ph[base + 64u], fx32(m11 * rt));
    }
  }
  workgroupBarrier();
  for (var i = li; i < 128u; i += 64u) {
    let by = i >> 5u;
    let bx = (i >> 3u) & 3u;
    let t = i & 7u;
    let pb = (by + 1u) * 54u + (bx + 1u) * 9u;
    let ifix = 1.0 / p.fix;
    var v = f32(atomicLoad(&Ph[pb + t])) * ifix;
    if (t == 0u) { v += f32(atomicLoad(&Ph[pb + 8u])) * ifix; }
    hist[i] = v;
  }
  workgroupBarrier();
  let h0 = hist[2u * li];
  let h1 = hist[2u * li + 1u];
  let ss = reduce64(li, h0 * h0 + h1 * h1);
  let ok = ss > 0.0;
  let inv = select(0.0, 1.0 / sqrt(ss), ok);
  var v0 = min(h0 * inv, 0.2);
  var v1 = min(h1 * inv, 0.2);
  let ss2 = reduce64(li, v0 * v0 + v1 * v1);
  let inv2 = select(0.0, 1.0 / sqrt(ss2), ss2 > 0.0);
  v0 = v0 * inv2;
  v1 = v1 * inv2;
  let l1 = reduce64(li, v0 + v1);
  let il1 = select(0.0, 1.0 / l1, l1 > 0.0);
  qb[2u * li] = select(0u, u32(min(255.0, floor(512.0 * sqrt(max(v0 * il1, 0.0)) + 0.5))), ok);
  qb[2u * li + 1u] = select(0u, u32(min(255.0, floor(512.0 * sqrt(max(v1 * il1, 0.0)) + 0.5))), ok);
  workgroupBarrier();
  if (li < 32u) {
    let o = (q * p.maxOri + a) * 32u + li;
    desc[o] = qb[4u * li] | (qb[4u * li + 1u] << 8u) | (qb[4u * li + 2u] << 16u) | (qb[4u * li + 3u] << 24u);
  }
}
`;
