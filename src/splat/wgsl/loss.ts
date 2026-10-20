/**
 * WGSL for the training loss: target preparation (box downscale of the RGBA8 training image to
 * the current resolution level), the per-view affine colour correction x = M·[C, 1] (learned
 * 3×4 per training view), loss = (1 − λ)·L1 + λ·(1 − SSIM) with the separable 11-tap Gaussian
 * SSIM (zero padding, as in ../cpu-reference.ts `lossCPU`), its gradient ∂loss/∂C, the colour
 * matrices' gradient and their Adam step.
 *
 * SSIM backward: with per-pixel partials P = ∂S/∂μx − 2μx ∂S/∂σxx − μy ∂S/∂σxy, Q = ∂S/∂σxx,
 * R = ∂S/∂σxy, ∂ΣS/∂x = G⋆P + 2x·G⋆Q + y·G⋆R (G symmetric, zero padding: the blur is its own
 * adjoint). Both SSIM kernels work on 16×16 tiles with a 5-pixel halo staged in workgroup memory
 * (≤ 13.7 KB), one channel at a time.
 */
import { SSIM_C1, SSIM_C2, SSIM_WEIGHTS } from '../cpu-reference';
import { FRAME_WGSL } from './common';

/** Byte size of the LossCfg uniform. */
export const LOSS_CFG_BYTES = 32;
/** Floats per view in the colour buffer: M (12), Adam m (12), v (12), step count, pad. */
export const COLOUR_FLOATS = 48;

const WEIGHTS = `const SW = array<f32, 11>(${Array.from(SSIM_WEIGHTS, (w) => w.toPrecision(9)).join(', ')});`;

const LOSS_CFG = /* wgsl */ `
struct LossCfg {
  view: u32,
  srcW: u32,
  srcH: u32,
  factor: u32,
  lambda: f32,
  invN: f32,
  pad0: f32,
  pad1: f32,
}
`;

/** Box-downscales the view's RGBA8 image by `factor` into `target` (alpha = min over the box). */
export const TARGET_WGSL = /* wgsl */ `
${FRAME_WGSL}
${LOSS_CFG}
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> lc: LossCfg;
@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read_write> tgt: array<u32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let W = frame.dims.x;
  let H = frame.dims.y;
  if (gid.x >= W || gid.y >= H) { return; }
  let f = lc.factor;
  var acc = vec3f(0.0);
  var a = 1.0;
  for (var dy = 0u; dy < f; dy++) {
    for (var dx = 0u; dx < f; dx++) {
      let v = unpack4x8unorm(src[(gid.y * f + dy) * lc.srcW + gid.x * f + dx]);
      acc += v.rgb;
      a = min(a, v.a);
    }
  }
  tgt[gid.y * W + gid.x] = pack4x8unorm(vec4f(acc / f32(f * f), a));
}
`;

const COMMON = /* wgsl */ `
${FRAME_WGSL}
${LOSS_CFG}
${WEIGHTS}
const C1: f32 = ${SSIM_C1};
const C2: f32 = ${SSIM_C2};
fn colourRow(c: u32) -> vec4f {
  let b = lc.view * 48u + 4u * c;
  return vec4f(colour[b], colour[b + 1u], colour[b + 2u], colour[b + 3u]);
}
`;

/**
 * SSIM forward + per-pixel partials. Writes pqr (3 vec4 per pixel: P, Q, R per channel) and
 * per-workgroup loss sums (L1, SSIM, valid pixels).
 */
export const SSIM_FWD_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> lc: LossCfg;
@group(0) @binding(2) var<storage, read> render: array<vec4f>;
@group(0) @binding(3) var<storage, read> tgt: array<u32>;
@group(0) @binding(4) var<storage, read> colour: array<f32>;
@group(0) @binding(5) var<storage, read_write> pqr: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> lossPartial: array<vec4f>;
${COMMON}
var<workgroup> xs: array<f32, 676>;
var<workgroup> ys: array<f32, 676>;
var<workgroup> hs: array<f32, 2080>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let W = frame.dims.x;
  let H = frame.dims.y;
  let lx = li & 15u;
  let ly = li >> 4u;
  let px = wg.x * 16u + lx;
  let py = wg.y * 16u + ly;
  let inside = px < W && py < H;
  var valid = false;
  if (inside) { valid = unpack4x8unorm(tgt[py * W + px]).a >= 0.5; }
  var P = vec3f(0.0);
  var Q = vec3f(0.0);
  var R = vec3f(0.0);
  var l1 = 0.0;
  var ssim = 0.0;
  for (var c = 0u; c < 3u; c++) {
    let row = colourRow(c);
    for (var e = li; e < 676u; e += 256u) {
      let gx = i32(wg.x * 16u) - 5 + i32(e % 26u);
      let gy = i32(wg.y * 16u) - 5 + i32(e / 26u);
      var xv = 0.0;
      var yv = 0.0;
      if (gx >= 0 && gy >= 0 && gx < i32(W) && gy < i32(H)) {
        let o = u32(gy) * W + u32(gx);
        let t = unpack4x8unorm(tgt[o]);
        if (t.a >= 0.5) {
          xv = dot(row, vec4f(render[o].xyz, 1.0));
          yv = t[c];
        }
      }
      xs[e] = xv;
      ys[e] = yv;
    }
    workgroupBarrier();
    for (var e = li; e < 416u; e += 256u) {
      let r = e / 16u;
      let cc = e % 16u;
      var m = vec4f(0.0);
      var xy = 0.0;
      for (var k = 0u; k < 11u; k++) {
        let x = xs[r * 26u + cc + k];
        let y = ys[r * 26u + cc + k];
        m += SW[k] * vec4f(x, y, x * x, y * y);
        xy += SW[k] * x * y;
      }
      hs[e] = m.x;
      hs[416u + e] = m.y;
      hs[832u + e] = m.z;
      hs[1248u + e] = m.w;
      hs[1664u + e] = xy;
    }
    workgroupBarrier();
    var m = vec4f(0.0);
    var exy = 0.0;
    for (var k = 0u; k < 11u; k++) {
      let e = (ly + k) * 16u + lx;
      m += SW[k] * vec4f(hs[e], hs[416u + e], hs[832u + e], hs[1248u + e]);
      exy += SW[k] * hs[1664u + e];
    }
    if (valid) {
      let mx = m.x;
      let my = m.y;
      let sxx = m.z - mx * mx;
      let syy = m.w - my * my;
      let sxy = exy - mx * my;
      let A1 = 2.0 * mx * my + C1;
      let A2 = 2.0 * sxy + C2;
      let B1 = mx * mx + my * my + C1;
      let B2 = sxx + syy + C2;
      let S = (A1 * A2) / (B1 * B2);
      ssim += S;
      let dmu = (2.0 * my * A2) / (B1 * B2) - (S * 2.0 * mx) / B1;
      let dsxx = -S / B2;
      let dsxy = (2.0 * A1) / (B1 * B2);
      P[c] = dmu - 2.0 * mx * dsxx - my * dsxy;
      Q[c] = dsxx;
      R[c] = dsxy;
      let ce = (ly + 5u) * 26u + lx + 5u;
      l1 += abs(xs[ce] - ys[ce]);
    }
    workgroupBarrier();
  }
  if (inside) {
    let o = py * W + px;
    pqr[3u * o] = vec4f(P, 0.0);
    pqr[3u * o + 1u] = vec4f(Q, 0.0);
    pqr[3u * o + 2u] = vec4f(R, 0.0);
  }
  // workgroup sums (hs reused)
  hs[li] = l1;
  hs[256u + li] = ssim;
  hs[512u + li] = select(0.0, 1.0, valid);
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (li < st) {
      hs[li] += hs[li + st];
      hs[256u + li] += hs[256u + li + st];
      hs[512u + li] += hs[512u + li + st];
    }
    workgroupBarrier();
  }
  if (li == 0u) { lossPartial[wg.y * frame.dims.z + wg.x] = vec4f(hs[0], hs[256], hs[512], 0.0); }
}
`;

/**
 * Loss gradient: ∂loss/∂x from the L1 term and the blurred SSIM partials, then
 * ∂loss/∂C = M₃ᵀ ∂loss/∂x (M₃ = the 3×3 part of the view's colour matrix) and the
 * per-workgroup sums of ∂loss/∂M (3 vec4: row c = ∂/∂x_c · [C, 1]).
 */
export const SSIM_BWD_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> lc: LossCfg;
@group(0) @binding(2) var<storage, read> render: array<vec4f>;
@group(0) @binding(3) var<storage, read> tgt: array<u32>;
@group(0) @binding(4) var<storage, read> colour: array<f32>;
@group(0) @binding(5) var<storage, read> pqr: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> dLdC: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> colourPartial: array<vec4f>;
${COMMON}
var<workgroup> ps: array<f32, 2028>;
var<workgroup> hs: array<f32, 1248>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let W = frame.dims.x;
  let H = frame.dims.y;
  let lx = li & 15u;
  let ly = li >> 4u;
  let px = wg.x * 16u + lx;
  let py = wg.y * 16u + ly;
  let inside = px < W && py < H;
  var valid = false;
  var Craw = vec3f(0.0);
  var y = vec3f(0.0);
  if (inside) {
    let t = unpack4x8unorm(tgt[py * W + px]);
    valid = t.a >= 0.5;
    y = t.rgb;
    Craw = render[py * W + px].xyz;
  }
  var x = vec3f(0.0);
  for (var c = 0u; c < 3u; c++) { x[c] = dot(colourRow(c), vec4f(Craw, 1.0)); }
  var dx = vec3f(0.0);
  for (var c = 0u; c < 3u; c++) {
    for (var e = li; e < 676u; e += 256u) {
      let gx = i32(wg.x * 16u) - 5 + i32(e % 26u);
      let gy = i32(wg.y * 16u) - 5 + i32(e / 26u);
      var v = vec3f(0.0);
      if (gx >= 0 && gy >= 0 && gx < i32(W) && gy < i32(H)) {
        let o = u32(gy) * W + u32(gx);
        v = vec3f(pqr[3u * o][c], pqr[3u * o + 1u][c], pqr[3u * o + 2u][c]);
      }
      ps[e] = v.x;
      ps[676u + e] = v.y;
      ps[1352u + e] = v.z;
    }
    workgroupBarrier();
    for (var e = li; e < 416u; e += 256u) {
      let r = e / 16u;
      let cc = e % 16u;
      var m = vec3f(0.0);
      for (var k = 0u; k < 11u; k++) {
        let s = r * 26u + cc + k;
        m += SW[k] * vec3f(ps[s], ps[676u + s], ps[1352u + s]);
      }
      hs[e] = m.x;
      hs[416u + e] = m.y;
      hs[832u + e] = m.z;
    }
    workgroupBarrier();
    var b = vec3f(0.0);
    for (var k = 0u; k < 11u; k++) {
      let e = (ly + k) * 16u + lx;
      b += SW[k] * vec3f(hs[e], hs[416u + e], hs[832u + e]);
    }
    if (valid) {
      let d = x[c] - y[c];
      let dS = b.x + 2.0 * x[c] * b.y + y[c] * b.z;
      dx[c] = lc.invN * ((1.0 - lc.lambda) * sign(d) - lc.lambda * dS);
    }
    workgroupBarrier();
  }
  // ∂/∂C = M₃ᵀ ∂/∂x
  let r0 = colourRow(0u);
  let r1 = colourRow(1u);
  let r2 = colourRow(2u);
  let dC = dx.x * r0.xyz + dx.y * r1.xyz + dx.z * r2.xyz;
  if (inside) { dLdC[py * W + px] = vec4f(dC, 0.0); }
  // colour-matrix gradient sums, one row at a time (ps reused as 256 × vec4)
  let cin = vec4f(Craw, 1.0);
  for (var c = 0u; c < 3u; c++) {
    let v = dx[c] * cin;
    ps[4u * li] = v.x;
    ps[4u * li + 1u] = v.y;
    ps[4u * li + 2u] = v.z;
    ps[4u * li + 3u] = v.w;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
      if (li < st) {
        for (var k = 0u; k < 4u; k++) { ps[4u * li + k] += ps[4u * (li + st) + k]; }
      }
      workgroupBarrier();
    }
    if (li == 0u) { colourPartial[3u * (wg.y * frame.dims.z + wg.x) + c] = vec4f(ps[0], ps[1], ps[2], ps[3]); }
    workgroupBarrier();
  }
}
`;

/** Byte size of the ColourCfg uniform. */
export const COLOUR_CFG_BYTES = 48;

/**
 * One workgroup: sums the per-workgroup colour gradients and loss sums, takes an Adam step on
 * the view's colour matrix (with an L2 pull towards [I | 0]) and stores the loss sums.
 */
export const COLOUR_ADAM_WGSL = /* wgsl */ `
struct ColourCfg {
  numWG: u32,
  view: u32,
  enabled: u32,
  statSlot: u32,
  lr: f32,
  beta1: f32,
  beta2: f32,
  eps: f32,
  reg: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
}
@group(0) @binding(0) var<uniform> cc: ColourCfg;
@group(0) @binding(1) var<storage, read> colourPartial: array<vec4f>;
@group(0) @binding(2) var<storage, read> lossPartial: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> colour: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<vec4f>;
var<workgroup> red: array<vec4f, 1024>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var a0 = vec4f(0.0);
  var a1 = vec4f(0.0);
  var a2 = vec4f(0.0);
  var l = vec4f(0.0);
  for (var w = li; w < cc.numWG; w += 256u) {
    a0 += colourPartial[3u * w];
    a1 += colourPartial[3u * w + 1u];
    a2 += colourPartial[3u * w + 2u];
    l += lossPartial[w];
  }
  red[li] = a0;
  red[256u + li] = a1;
  red[512u + li] = a2;
  red[768u + li] = l;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (li < st) {
      red[li] += red[li + st];
      red[256u + li] += red[256u + li + st];
      red[512u + li] += red[512u + li + st];
      red[768u + li] += red[768u + li + st];
    }
    workgroupBarrier();
  }
  if (li == 0u) { stats[cc.statSlot] = red[768]; }
  if (cc.enabled != 0u && li < 12u) {
    let r = li / 4u;
    let k = li % 4u;
    let base = cc.view * 48u;
    let p = colour[base + li];
    let ident = select(0.0, 1.0, r == k);
    let g = red[256u * r][k] + 2.0 * cc.reg * (p - ident);
    let t = colour[base + 36u] + 1.0;
    let m = cc.beta1 * colour[base + 12u + li] + (1.0 - cc.beta1) * g;
    let v = cc.beta2 * colour[base + 24u + li] + (1.0 - cc.beta2) * g * g;
    colour[base + 12u + li] = m;
    colour[base + 24u + li] = v;
    let mh = m / (1.0 - pow(cc.beta1, t));
    let vh = v / (1.0 - pow(cc.beta2, t));
    colour[base + li] = p - cc.lr * mh / (sqrt(vh) + cc.eps);
  }
  workgroupBarrier();
  if (cc.enabled != 0u && li == 0u) { colour[cc.view * 48u + 36u] = colour[cc.view * 48u + 36u] + 1.0; }
}
`;
