/**
 * WGSL for PatchMatch MVS (see patchmatch-cpu.ts for the algorithm; this shader
 * mirrors it line by line, including the PCG random streams).
 *
 * Bindings (group 0, explicit layout in patchmatch-gpu.ts):
 *   0 uniform Params (dynamic offset: one 256-byte block per pass/band)
 *   1 uniform Cams   (16 × SrcCam)
 *   2 texture_2d<f32>       refTex   rgba8unorm, rgb = colour, a = Rec.709 grey
 *   3 texture_2d_array<f32> srcTex   r8unorm grey, one layer per source (linear filtering)
 *   4 sampler (linear, clamp-to-edge)
 *   5 texture_2d_array<f32> srcDepth r32float source depth maps (geometric mode; textureLoad only)
 *   6 storage rw planes : array<vec4f>  (nx, ny, nz, z) per pixel
 *   7 storage rw costs  : array<f32>    total cost (photometric + geometric) used for propagation
 *   8 storage rw photo  : array<f32>    photometric cost (output)
 *   9 storage rw views  : array<u32>    selected-view bit mask
 *
 * Portability: core WebGPU only, workgroup 8×8, no workgroup memory, 4 storage
 * buffers, no f32 atomics.
 */

export interface PatchMatchShaderConfig {
  /** Compile-time capacity for sources (array sizes); 4, 8, 12 or 16. */
  maxSources: number;
  patchRadius: number;
  patchStep: number;
}

/** Byte size of the Params uniform block (must match the struct below). */
export const PARAMS_BYTES = 128;
/** Stride between per-pass Params blocks (minUniformBufferOffsetAlignment upper bound). */
export const PARAMS_STRIDE = 256;
export const WORKGROUP_X = 8;
export const WORKGROUP_Y = 8;

/**
 * Params layout (32 × 4 bytes), written by patchmatch-gpu.ts:
 *  u32: width, height, numSrc, colour, rowStart, rowEnd, iter, seed
 *  f32: f, cx, cy, invF, zMin, zMax, invZMin, invZMax,
 *       tauGood, badCost, pert, npert, lambda, tau, colorCoef, minVar,
 *  u32: topK, numProposals, usePrior, geo,
 *  f32: spatialCoef, pad, pad, pad
 */
export function patchMatchWGSL(cfg: PatchMatchShaderConfig): string {
  const np = Math.floor((2 * cfg.patchRadius) / cfg.patchStep) + 1;
  return /* wgsl */ `
const NMAX: u32 = ${cfg.maxSources}u;
const NP: u32 = ${np}u;
const NS: u32 = ${np * np}u;
const PR: i32 = ${cfg.patchRadius};
const PSTEP: i32 = ${cfg.patchStep};
const COST_MAX: f32 = 2.0;
const PI: f32 = 3.14159265358979;

struct Params {
  width: u32, height: u32, numSrc: u32, colour: u32,
  rowStart: u32, rowEnd: u32, iter: u32, seed: u32,
  f: f32, cx: f32, cy: f32, invF: f32,
  zMin: f32, zMax: f32, invZMin: f32, invZMax: f32,
  tauGood: f32, badCost: f32, pert: f32, npert: f32,
  lambda: f32, tau: f32, colorCoef: f32, minVar: f32,
  topK: u32, numProposals: u32, usePrior: u32, geo: u32,
  spatialCoef: f32, pad0: f32, pad1: f32, pad2: f32,
};

struct SrcCam {
  A0: vec4f, A1: vec4f, A2: vec4f, b: vec4f,
  M0: vec4f, M1: vec4f, M2: vec4f, v: vec4f,
  size: vec4f,   // imgW, imgH, 1/texW, 1/texH
  dsize: vec4f,  // depthScaleX, depthScaleY, depthW, depthH
};

struct Cams { c: array<SrcCam, 16> };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> cams: Cams;
@group(0) @binding(2) var refTex: texture_2d<f32>;
@group(0) @binding(3) var srcTex: texture_2d_array<f32>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var srcDepth: texture_2d_array<f32>;
@group(0) @binding(6) var<storage, read_write> planes: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> costs: array<f32>;
@group(0) @binding(8) var<storage, read_write> photo: array<f32>;
@group(0) @binding(9) var<storage, read_write> views: array<u32>;

var<private> wk: array<f32, NS>;
var<private> wrk: array<f32, NS>;
var<private> refMean: f32;
var<private> refVar: f32;
var<private> invSumW: f32;
var<private> rng: u32;

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand() -> f32 {
  rng = pcg(rng);
  return f32(rng >> 8u) * (1.0 / 16777216.0);
}

fn randomInvDepth() -> f32 {
  let u = rand();
  return 1.0 / (P.invZMax + u * (P.invZMin - P.invZMax));
}

fn randomNormal(ray: vec3f) -> vec3f {
  let zc = 2.0 * rand() - 1.0;
  let phi = 2.0 * PI * rand();
  let rr = sqrt(max(0.0, 1.0 - zc * zc));
  var n = vec3f(rr * cos(phi), rr * sin(phi), zc);
  if (dot(n, ray) > 0.0) { n = -n; }
  return n;
}

fn loadRef(x: i32, y: i32) -> vec4f {
  let c = vec2i(clamp(x, 0, i32(P.width) - 1), clamp(y, 0, i32(P.height) - 1));
  return textureLoad(refTex, c, 0);
}

// Bilateral weights + weighted reference statistics of the window at (x, y).
fn prepareRef(x: i32, y: i32) -> bool {
  let c = loadRef(x, y);
  let crgb = round(c.rgb * 255.0);
  var sw = 0.0;
  var swr = 0.0;
  var swrr = 0.0;
  var k = 0u;
  for (var a = 0u; a < NP; a++) {
    let dy = i32(a) * PSTEP - PR;
    for (var b = 0u; b < NP; b++) {
      let dx = i32(b) * PSTEP - PR;
      let t = loadRef(x + dx, y + dy);
      let r = t.a;
      let dc = dot(abs(round(t.rgb * 255.0) - crgb), vec3f(1.0));
      let w = exp(-sqrt(f32(dx * dx + dy * dy)) * P.spatialCoef - dc * P.colorCoef);
      wk[k] = w;
      wrk[k] = w * r;
      sw += w;
      swr += w * r;
      swrr += w * r * r;
      k++;
    }
  }
  invSumW = 1.0 / sw;
  refMean = swr * invSumW;
  refVar = swrr * invSumW - refMean * refMean;
  return refVar >= P.minVar;
}

// (m = K_r^-T n, 1/delta); w <= 0 => invalid plane.
fn hypParams(ray: vec3f, n: vec3f, z: f32) -> vec4f {
  let delta = -z * dot(n, ray);
  if (!(delta > 0.0)) { return vec4f(0.0); }
  let m = vec3f(n.x * P.invF, n.y * P.invF, n.z - (n.x * P.cx + n.y * P.cy) * P.invF);
  return vec4f(m, 1.0 / delta);
}

// 1 - bilateral weighted NCC of the window warped into source s by H = A - b m^T / delta.
fn viewCost(s: u32, px: vec2f, hp: vec4f) -> f32 {
  let cam = cams.c[s];
  let bs = cam.b.xyz * hp.w;
  let H0 = cam.A0.xyz - bs.x * hp.xyz;
  let H1 = cam.A1.xyz - bs.y * hp.xyz;
  let H2 = cam.A2.xyz - bs.z * hp.xyz;
  let p = vec3f(px, 1.0);
  let h = vec3f(dot(H0, p), dot(H1, p), dot(H2, p));
  if (!(h.z > 1e-9)) { return COST_MAX; }
  let sz = cam.size.xy - 1.0;
  let c = h.xy / h.z;
  if (!(c.x >= 0.0 && c.x <= sz.x && c.y >= 0.0 && c.y <= sz.y)) { return COST_MAX; }
  let hu = vec3f(H0.x, H1.x, H2.x);
  let hv = vec3f(H0.y, H1.y, H2.y);
  let layer = i32(s);
  var sws = 0.0;
  var swss = 0.0;
  var swrs = 0.0;
  var k = 0u;
  for (var a = 0u; a < NP; a++) {
    let dy = f32(i32(a) * PSTEP - PR);
    let hrow = h + dy * hv;
    for (var b = 0u; b < NP; b++) {
      let dx = f32(i32(b) * PSTEP - PR);
      let q = hrow + dx * hu;
      if (!(q.z > 1e-9)) { return COST_MAX; }
      let uv = clamp(q.xy / q.z, vec2f(0.0), sz);
      let val = textureSampleLevel(srcTex, samp, (uv + 0.5) * cam.size.zw, layer, 0.0).r;
      let w = wk[k];
      sws += w * val;
      swss += w * val * val;
      swrs += wrk[k] * val;
      k++;
    }
  }
  let ms = sws * invSumW;
  let vs = swss * invSumW - ms * ms;
  if (vs < P.minVar) { return COST_MAX; }
  let cov = swrs * invSumW - refMean * ms;
  let ncc = cov / sqrt(refVar * vs);
  return clamp(1.0 - ncc, 0.0, COST_MAX);
}

// Truncated forward-backward reprojection error against source depth map s.
fn geoCost(s: u32, px: vec2f, z: f32) -> f32 {
  let cam = cams.c[s];
  let p = vec3f(px, 1.0);
  let q = z * vec3f(dot(cam.A0.xyz, p), dot(cam.A1.xyz, p), dot(cam.A2.xyz, p)) + cam.b.xyz;
  if (!(q.z > 1e-9)) { return P.tau; }
  let ps = q.xy / q.z;
  let sz = cam.size.xy - 1.0;
  if (!(ps.x >= 0.0 && ps.x <= sz.x && ps.y >= 0.0 && ps.y <= sz.y)) { return P.tau; }
  if (cam.dsize.z < 1.0 || cam.dsize.w < 1.0) { return P.tau; }
  let dmax = vec2i(cam.dsize.zw) - vec2i(1);
  let di = clamp(vec2i(floor((ps + 0.5) * cam.dsize.xy)), vec2i(0), dmax);
  let zs = textureLoad(srcDepth, di, i32(s), 0).r;
  if (!(zs > 0.0)) { return P.tau; }
  let ph = vec3f(ps, 1.0);
  let r = zs * vec3f(dot(cam.M0.xyz, ph), dot(cam.M1.xyz, ph), dot(cam.M2.xyz, ph)) + cam.v.xyz;
  if (!(r.z > 1e-9)) { return P.tau; }
  return min(length(r.xy / r.z - px), P.tau);
}

fn checkerOffset(region: u32, k: u32) -> vec2i {
  var d: vec2i;
  if ((region & 1u) == 0u) {
    if (k == 0u) {
      d = vec2i(0, -1);
    } else {
      let j = i32((k + 1u) >> 1u);
      d = vec2i(select(j, -j, (k & 1u) == 1u), -(1 + j));
    }
  } else {
    d = vec2i(0, -(3 + 2 * i32(k)));
  }
  let dir = region >> 1u;
  if (dir == 1u) { return vec2i(d.x, -d.y); }
  if (dir == 2u) { return vec2i(d.y, d.x); }
  if (dir == 3u) { return vec2i(-d.y, d.x); }
  return d;
}

@compute @workgroup_size(${WORKGROUP_X}, ${WORKGROUP_Y})
fn init(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;
  let y = P.rowStart + gid.y;
  if (x >= P.width || y >= P.rowEnd) { return; }
  let idx = y * P.width + x;
  rng = pcg(idx ^ P.seed);
  let px = vec2f(f32(x), f32(y));
  let ray = vec3f((px - vec2f(P.cx, P.cy)) * P.invF, 1.0);
  var pl = planes[idx];
  let ok = P.usePrior != 0u && pl.w >= P.zMin && pl.w <= P.zMax && dot(pl.xyz, ray) < 0.0;
  if (!ok) {
    let n = randomNormal(ray);
    let z = randomInvDepth();
    pl = vec4f(n, z);
    planes[idx] = pl;
  }
  if (!prepareRef(i32(x), i32(y))) {
    costs[idx] = COST_MAX;
    photo[idx] = COST_MAX;
    views[idx] = 0u;
    return;
  }
  let hp = hypParams(ray, pl.xyz, pl.w);
  var sc: array<f32, NMAX>;
  for (var s = 0u; s < P.numSrc; s++) {
    sc[s] = select(COST_MAX, viewCost(s, px, hp), hp.w > 0.0);
  }
  let K = min(P.topK, P.numSrc);
  var sel = 0u;
  var sum = 0.0;
  var g = 0.0;
  for (var k = 0u; k < K; k++) {
    var bi = 0u;
    var bc = 1e30;
    for (var s = 0u; s < P.numSrc; s++) {
      if ((sel & (1u << s)) == 0u && sc[s] < bc) {
        bc = sc[s];
        bi = s;
      }
    }
    sel = sel | (1u << bi);
    sum += bc;
    if (P.geo != 0u) { g += geoCost(bi, px, pl.w); }
  }
  let fk = f32(K);
  photo[idx] = sum / fk;
  costs[idx] = sum / fk + select(0.0, P.lambda * g / fk, P.geo != 0u);
  views[idx] = sel;
}

@compute @workgroup_size(${WORKGROUP_X}, ${WORKGROUP_Y})
fn propagate(@builtin(global_invocation_id) gid: vec3u) {
  let y = P.rowStart + gid.y;
  if (y >= P.rowEnd) { return; }
  let x = 2u * gid.x + ((y + P.colour) & 1u);
  if (x >= P.width) { return; }
  let W = i32(P.width);
  let H = i32(P.height);
  let xi = i32(x);
  let yi = i32(y);
  let idx = y * P.width + x;
  if (!prepareRef(xi, yi)) {
    costs[idx] = COST_MAX;
    photo[idx] = COST_MAX;
    views[idx] = 0u;
    return;
  }
  rng = pcg(idx ^ P.seed);
  let px = vec2f(f32(x), f32(y));
  let cxy = vec2f(P.cx, P.cy);
  let ray = vec3f((px - cxy) * P.invF, 1.0);
  let N = P.numSrc;
  let geo = P.geo != 0u;

  // ---- candidates: slot 0 = current, 1..8 = best of each checkerboard region
  var candN: array<vec3f, 9>;
  var candZ: array<f32, 9>;
  var candOk: array<u32, 9>;
  let cur = planes[idx];
  candN[0] = cur.xyz;
  candZ[0] = cur.w;
  candOk[0] = 1u;
  for (var r = 0u; r < 8u; r++) {
    let cnt = select(11u, 7u, (r & 1u) == 0u);
    var best = -1;
    var bestC = 1e30;
    for (var k = 0u; k < cnt; k++) {
      let q = vec2i(xi, yi) + checkerOffset(r, k);
      if (q.x < 0 || q.y < 0 || q.x >= W || q.y >= H) { continue; }
      let qi = q.y * W + q.x;
      let c = costs[qi];
      if (c < bestC) {
        bestC = c;
        best = qi;
      }
    }
    let slot = r + 1u;
    candOk[slot] = 0u;
    if (best < 0) { continue; }
    let qx = best % W;
    let qy = best / W;
    let pl = planes[best];
    let rq = vec3f((vec2f(f32(qx), f32(qy)) - cxy) * P.invF, 1.0);
    let nrq = dot(pl.xyz, rq);
    let nrp = dot(pl.xyz, ray);
    if (!(nrp < 0.0)) { continue; }
    let zp = pl.w * nrq / nrp;
    if (!(zp >= P.zMin && zp <= P.zMax)) { continue; }
    candN[slot] = pl.xyz;
    candZ[slot] = zp;
    candOk[slot] = 1u;
  }

  // ---- per-view costs of the 9 candidates
  var C: array<f32, ${9 * cfg.maxSources}>;
  for (var j = 0u; j < 9u; j++) {
    var hp = vec4f(0.0);
    if (candOk[j] != 0u) { hp = hypParams(ray, candN[j], candZ[j]); }
    if (hp.w > 0.0) {
      for (var s = 0u; s < N; s++) { C[j * NMAX + s] = viewCost(s, px, hp); }
    } else {
      candOk[j] = 0u;
      for (var s = 0u; s < N; s++) { C[j * NMAX + s] = COST_MAX; }
    }
  }

  // ---- ACMH view selection with a soft prior from the 4-neighbours
  var nNb = 0u;
  var m0 = 0u;
  var m1 = 0u;
  var m2 = 0u;
  var m3 = 0u;
  if (yi > 0) { m0 = views[idx - P.width]; nNb++; }
  if (yi < H - 1) { m1 = views[idx + P.width]; nNb++; }
  if (xi > 0) { m2 = views[idx - 1u]; nNb++; }
  if (xi < W - 1) { m3 = views[idx + 1u]; nNb++; }
  var wts: array<f32, NMAX>;
  var score: array<f32, NMAX>;
  var sumW = 0.0;
  for (var s = 0u; s < N; s++) {
    var nGood = 0u;
    var nBad = 0u;
    var conf = 0.0;
    var minC = COST_MAX;
    for (var j = 0u; j < 9u; j++) {
      if (candOk[j] == 0u) { continue; }
      let c = C[j * NMAX + s];
      minC = min(minC, c);
      if (c < P.tauGood) {
        nGood++;
        conf += exp(-c * c / 0.18);
      } else if (c > P.badCost) {
        nBad++;
      }
    }
    score[s] = minC;
    var w = 0.0;
    if (nGood > 2u && nBad < 3u) {
      let bit = 1u << s;
      let cnt = select(0u, 1u, (m0 & bit) != 0u) + select(0u, 1u, (m1 & bit) != 0u)
              + select(0u, 1u, (m2 & bit) != 0u) + select(0u, 1u, (m3 & bit) != 0u);
      let prior = 0.3 + 0.7 * f32(cnt) / f32(max(1u, nNb));
      w = conf / f32(nGood) * prior;
    }
    wts[s] = w;
    sumW += w;
  }
  if (!(sumW > 0.0)) {
    sumW = 0.0;
    let K = min(P.topK, N);
    for (var k = 0u; k < K; k++) {
      var bi = 0u;
      var bc = 1e30;
      for (var s = 0u; s < N; s++) {
        if (wts[s] == 0.0 && score[s] < bc) {
          bc = score[s];
          bi = s;
        }
      }
      wts[bi] = 1.0;
      sumW += 1.0;
    }
  }
  var mask = 0u;
  for (var s = 0u; s < N; s++) {
    if (wts[s] > 0.0) { mask = mask | (1u << s); }
  }
  let invSum = 1.0 / sumW;

  // ---- best candidate
  var bestJ = -1;
  var bestTotal = 1e30;
  var bestPhoto = COST_MAX;
  for (var j = 0u; j < 9u; j++) {
    if (candOk[j] == 0u) { continue; }
    var pc = 0.0;
    var g = 0.0;
    for (var s = 0u; s < N; s++) {
      let w = wts[s];
      if (w <= 0.0) { continue; }
      pc += w * C[j * NMAX + s];
      if (geo) { g += w * geoCost(s, px, candZ[j]); }
    }
    pc *= invSum;
    let tot = pc + select(0.0, P.lambda * g * invSum, geo);
    if (tot < bestTotal) {
      bestTotal = tot;
      bestPhoto = pc;
      bestJ = i32(j);
    }
  }
  var bn: vec3f;
  var bz: f32;
  if (bestJ < 0) {
    bn = candN[0];
    bz = candZ[0];
    bestTotal = COST_MAX + select(0.0, P.lambda * P.tau, geo);
    bestPhoto = COST_MAX;
  } else {
    bn = candN[u32(bestJ)];
    bz = candZ[u32(bestJ)];
  }

  // ---- refinement proposals
  if (P.numProposals > 0u) {
    let zPert = clamp(bz * (1.0 + P.pert * (2.0 * rand() - 1.0)), P.zMin, P.zMax);
    let r0 = rand();
    let r1 = rand();
    let r2 = rand();
    var pn = bn + P.npert * (2.0 * vec3f(r0, r1, r2) - 1.0);
    pn = pn / length(pn);
    if (!(dot(pn, ray) < 0.0)) { pn = bn; }
    let zRand = randomInvDepth();
    let qn = randomNormal(ray);
    let cn0 = bn;
    let cz0 = bz;
    for (var pi = 0u; pi < P.numProposals; pi++) {
      var n: vec3f;
      var z: f32;
      switch (pi) {
        case 0u: { n = cn0; z = zPert; }
        case 1u: { n = pn; z = cz0; }
        case 2u: { n = qn; z = zRand; }
        case 3u: { n = pn; z = zPert; }
        case 4u: { n = cn0; z = zRand; }
        default: { n = qn; z = cz0; }
      }
      if (!(z >= P.zMin && z <= P.zMax)) { continue; }
      let hp = hypParams(ray, n, z);
      if (!(hp.w > 0.0)) { continue; }
      var pc = 0.0;
      var g = 0.0;
      for (var s = 0u; s < N; s++) {
        let w = wts[s];
        if (w <= 0.0) { continue; }
        pc += w * viewCost(s, px, hp);
        if (geo) { g += w * geoCost(s, px, z); }
      }
      pc *= invSum;
      let tot = pc + select(0.0, P.lambda * g * invSum, geo);
      if (tot < bestTotal) {
        bestTotal = tot;
        bestPhoto = pc;
        bn = n;
        bz = z;
      }
    }
  }
  planes[idx] = vec4f(bn, bz);
  costs[idx] = bestTotal;
  photo[idx] = bestPhoto;
  views[idx] = mask;
}
`;
}
