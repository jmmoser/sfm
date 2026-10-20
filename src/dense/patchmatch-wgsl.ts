/**
 * WGSL for PatchMatch MVS (see patchmatch-cpu.ts for the algorithm; this shader
 * mirrors it, including the PCG random streams).
 *
 * Bindings (group 0, explicit layout in patchmatch-gpu.ts):
 *   0  uniform Params (dynamic offset: ring of 256-byte blocks, one per submission)
 *   1  uniform Cams   (16 × SrcCam)
 *   2  texture_2d<f32>       refTex   rgba8unorm reference image (rgb; the Rec.709 grey is derived in `loadRef`)
 *   3  texture_2d_array<f32> srcTex   r8unorm grey images (linear filtering); source s reads layer `cams.c[s].b.w`
 *   4  sampler (linear, clamp-to-edge)
 *   5  texture_2d_array<f32> srcDepth r32float source depth maps (geometric mode; textureLoad only); layer `cams.c[s].M0.w`
 *   6  storage rw planes : array<vec4f>  (nx, ny, nz, z) per pixel
 *   7  storage rw costs  : array<f32>    total cost (photometric + geometric) used for propagation;
 *                                       after `finalize`: output depth (0 = invalid); before
 *                                       `priorPlanes`: the prior's depth (priorW × priorH)
 *   8  storage rw photo  : array<f32>    photometric cost (output); before `priorPlanes`: the
 *                                       prior normal's octahedral x
 *   9  storage rw views  : array<u32>    selected-view bit mask; after `finalize`: octahedral
 *                                       normal (pack2x16snorm); before `priorPlanes`: octahedral y (f32 bits)
 *   10 storage rw status : array<u32>    [0] = number of executed dispatches (dropped-work sentinel)
 *
 * Portability / Safari: core WebGPU only, workgroup 8×8, no workgroup memory,
 * 5 storage buffers, no f32 atomics, no vec3 in buffers, no structs passed by
 * value. The fixed-size patch-window and checkerboard loops are unrolled at
 * generation time (Safari 26 adds a volatile forward-progress counter to every
 * WGSL loop and bounds-checks non-constant array indices).
 */

export interface PatchMatchShaderConfig {
  /** Compile-time capacity for sources (array sizes); 4, 8, 12 or 16. */
  maxSources: number;
  patchRadius: number;
  patchStep: number;
  /** Affine window warp (`PatchMatchOptions.affineWindow`): Jacobian at the window centre, no divide per sample. */
  affine?: boolean;
}

/** Byte size of the Params uniform block (must match the struct below). */
export const PARAMS_BYTES = 128;
/** Stride between Params blocks (minUniformBufferOffsetAlignment upper bound). */
export const PARAMS_STRIDE = 256;
export const WORKGROUP_X = 8;
export const WORKGROUP_Y = 8;
/** Windows with more samples than this are generated as loops instead of unrolled. */
export const MAX_UNROLLED_SAMPLES = 81;

/** ACMH checkerboard offset k of region r (identical to patchmatch-common checkerOffset). */
function checker(region: number, k: number): [number, number] {
  let dx: number;
  let dy: number;
  if ((region & 1) === 0) {
    if (k === 0) {
      dx = 0;
      dy = -1;
    } else {
      const j = (k + 1) >> 1;
      dx = (k & 1) === 1 ? -j : j;
      dy = -(1 + j);
    }
  } else {
    dx = 0;
    dy = -(3 + 2 * k);
  }
  switch (region >> 1) {
    case 1:
      return [dx, -dy];
    case 2:
      return [dy, dx];
    case 3:
      return [-dy, dx];
    default:
      return [dx, dy];
  }
}

const f = (v: number): string => {
  const s = String(v);
  return /[.eE]/.test(s) ? s : `${s}.0`;
};

/**
 * Params layout (32 × 4 bytes), written by patchmatch-gpu.ts:
 *  u32: width, height, numSrc, colour, rowStart, rowEnd, iter, seed
 *  f32: f, cx, cy, invF, zMin, zMax, invZMin, invZMax,
 *       tauGood, badCost, pert, npert, lambda, tau, colorCoef, minVar,
 *  u32: topK, numProposals, usePrior, geo,
 *  f32: spatialCoef, maxCost; u32: priorW, priorH
 */
export function patchMatchWGSL(cfg: PatchMatchShaderConfig): string {
  const offs: number[] = [];
  for (let i = -cfg.patchRadius; i <= cfg.patchRadius; i += cfg.patchStep) offs.push(i);
  const np = offs.length;
  const ns = np * np;
  const unroll = ns <= MAX_UNROLLED_SAMPLES;
  const lo = offs[0];
  const hi = offs[np - 1];

  // ---- reference window (bilateral weights) ----
  let prepareBody: string;
  if (unroll) {
    const lines: string[] = [];
    let k = 0;
    for (const dy of offs) {
      for (const dx of offs) {
        const dist = Math.sqrt(dx * dx + dy * dy);
        lines.push(
          `  t = loadRef(x + (${dx}), y + (${dy})); w = exp(${f(-dist)} * P.spatialCoef - dot(abs(round(t.rgb * 255.0) - crgb), vec3f(1.0)) * P.colorCoef); ` +
            `wk[${k}] = w; wrk[${k}] = w * t.a; sw += w; swr += w * t.a; swrr += w * t.a * t.a;`,
        );
        k++;
      }
    }
    prepareBody = `  var t: vec4f;\n  var w: f32;\n${lines.join('\n')}`;
  } else {
    prepareBody = `  var k = 0u;
  for (var a = 0u; a < ${np}u; a++) {
    let dy = i32(a) * ${cfg.patchStep} - ${cfg.patchRadius};
    for (var b = 0u; b < ${np}u; b++) {
      let dx = i32(b) * ${cfg.patchStep} - ${cfg.patchRadius};
      let t = loadRef(x + dx, y + dy);
      let w = exp(-sqrt(f32(dx * dx + dy * dy)) * P.spatialCoef - dot(abs(round(t.rgb * 255.0) - crgb), vec3f(1.0)) * P.colorCoef);
      wk[k] = w; wrk[k] = w * t.a; sw += w; swr += w * t.a; swrr += w * t.a * t.a;
      k++;
    }
  }`;
  }

  // ---- warped source window ----
  let sampleBody: string;
  // Affine: the window centre's Jacobian ju / jv (texture units per pixel offset) replaces the
  // projective divide per sample; clamping to [0, size − 1] px is the same as the projective path.
  const affineSetup = `  let ic = 1.0 / h.z;
  let ju = (hu.xy - c * hu.z) * ic * isz;
  let jv = (hv.xy - c * hv.z) * ic * isz;
  let b0 = (c + 0.5) * isz;
  let lo2 = 0.5 * isz;
  let hi2 = (sz + 0.5) * isz;`;
  if (unroll && cfg.affine) {
    const lines: string[] = [];
    let k = 0;
    for (const dy of offs) {
      lines.push(`  br = b0 + ${f(dy)} * jv;`);
      for (const dx of offs) {
        lines.push(
          `  v = textureSampleLevel(srcTex, samp, clamp(br + ${f(dx)} * ju, lo2, hi2), layer, 0.0).r; ` +
            `sws += wk[${k}] * v; swss += wk[${k}] * v * v; swrs += wrk[${k}] * v;`,
        );
        k++;
      }
    }
    sampleBody = `${affineSetup}\n  var br: vec2f;\n  var v: f32;\n${lines.join('\n')}`;
  } else if (unroll) {
    const lines: string[] = [];
    let k = 0;
    for (const dy of offs) {
      lines.push(`  hr = h + ${f(dy)} * hv;`);
      for (const dx of offs) {
        lines.push(
          `  q = hr + ${f(dx)} * hu; v = textureSampleLevel(srcTex, samp, (clamp(q.xy / q.z, vec2f(0.0), sz) + 0.5) * isz, layer, 0.0).r; ` +
            `sws += wk[${k}] * v; swss += wk[${k}] * v * v; swrs += wrk[${k}] * v;`,
        );
        k++;
      }
    }
    sampleBody = `  var hr: vec3f;\n  var q: vec3f;\n  var v: f32;\n${lines.join('\n')}`;
  } else if (cfg.affine) {
    sampleBody = `${affineSetup}
  var k = 0u;
  for (var a = 0u; a < ${np}u; a++) {
    let br = b0 + f32(i32(a) * ${cfg.patchStep} - ${cfg.patchRadius}) * jv;
    for (var b = 0u; b < ${np}u; b++) {
      let v = textureSampleLevel(srcTex, samp, clamp(br + f32(i32(b) * ${cfg.patchStep} - ${cfg.patchRadius}) * ju, lo2, hi2), layer, 0.0).r;
      sws += wk[k] * v; swss += wk[k] * v * v; swrs += wrk[k] * v;
      k++;
    }
  }`;
  } else {
    sampleBody = `  var k = 0u;
  for (var a = 0u; a < ${np}u; a++) {
    let hr = h + f32(i32(a) * ${cfg.patchStep} - ${cfg.patchRadius}) * hv;
    for (var b = 0u; b < ${np}u; b++) {
      let q = hr + f32(i32(b) * ${cfg.patchStep} - ${cfg.patchRadius}) * hu;
      let v = textureSampleLevel(srcTex, samp, (clamp(q.xy / q.z, vec2f(0.0), sz) + 0.5) * isz, layer, 0.0).r;
      sws += wk[k] * v; swss += wk[k] * v * v; swrs += wrk[k] * v;
      k++;
    }
  }`;
  }

  // ---- ACMH adaptive checkerboard sampling (unrolled) ----
  const regionLines: string[] = [];
  for (let r = 0; r < 8; r++) {
    const cnt = (r & 1) === 0 ? 7 : 11;
    regionLines.push(`  best = -1; bestC = 1e30;`);
    for (let k = 0; k < cnt; k++) {
      const [dx, dy] = checker(r, k);
      const conds: string[] = [];
      if (dx < 0) conds.push(`xi >= ${-dx}`);
      if (dx > 0) conds.push(`xi < W - ${dx}`);
      if (dy < 0) conds.push(`yi >= ${-dy}`);
      if (dy > 0) conds.push(`yi < H - ${dy}`);
      const qi = `idxi + (${dy * 1}) * W + (${dx})`;
      regionLines.push(`  if (${conds.join(' && ') || 'true'}) { let qi = ${qi}; let c = costs[qi]; if (c < bestC) { bestC = c; best = qi; } }`);
    }
    regionLines.push(`  cr = regionCandidate(best, ray, cxy); candN[${r + 1}] = cr.xyz; candZ[${r + 1}] = cr.w; candOk[${r + 1}] = select(0u, 1u, cr.w > 0.0);`);
  }

  return /* wgsl */ `
const NMAX: u32 = ${cfg.maxSources}u;
const NS: u32 = ${ns}u;
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
  spatialCoef: f32, maxCost: f32, priorW: u32, priorH: u32,
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
@group(0) @binding(10) var<storage, read_write> status: array<u32>;

var<private> wk: array<f32, NS>;
var<private> wrk: array<f32, NS>;
var<private> refMean: f32;
var<private> refVar: f32;
var<private> invSumW: f32;
var<private> rng: u32;
var<private> candN: array<vec3f, 9>;
var<private> candZ: array<f32, 9>;
var<private> candOk: array<u32, 9>;

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

// Reference texel: rgb + the Rec.709 grey of the 8-bit colour, rounded to 8 bit (= rgbaToGray8) in a.
fn loadRef(x: i32, y: i32) -> vec4f {
  let c = vec2i(clamp(x, 0, i32(P.width) - 1), clamp(y, 0, i32(P.height) - 1));
  let t = textureLoad(refTex, c, 0).rgb;
  return vec4f(t, floor(dot(round(t * 255.0), vec3f(0.2126, 0.7152, 0.0722)) + 0.5) * (1.0 / 255.0));
}

// Bilateral weights + weighted reference statistics of the window at (x, y).
fn prepareRef(x: i32, y: i32) -> bool {
  let crgb = round(loadRef(x, y).rgb * 255.0);
  var sw = 0.0;
  var swr = 0.0;
  var swrr = 0.0;
${prepareBody}
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
  let bs = cams.c[s].b.xyz * hp.w;
  let H0 = cams.c[s].A0.xyz - bs.x * hp.xyz;
  let H1 = cams.c[s].A1.xyz - bs.y * hp.xyz;
  let H2 = cams.c[s].A2.xyz - bs.z * hp.xyz;
  let p = vec3f(px, 1.0);
  let h = vec3f(dot(H0, p), dot(H1, p), dot(H2, p));
  let hu = vec3f(H0.x, H1.x, H2.x);
  let hv = vec3f(H0.y, H1.y, H2.y);
  // projective depth is affine in the window offset: its minimum is at a corner
  let zmin = min(min(h.z + ${f(lo)} * hu.z + ${f(lo)} * hv.z, h.z + ${f(hi)} * hu.z + ${f(lo)} * hv.z),
                 min(h.z + ${f(lo)} * hu.z + ${f(hi)} * hv.z, h.z + ${f(hi)} * hu.z + ${f(hi)} * hv.z));
  if (!(h.z > 1e-9) || !(zmin > 1e-9)) { return COST_MAX; }
  let sz = cams.c[s].size.xy - 1.0;
  let isz = cams.c[s].size.zw;
  let c = h.xy / h.z;
  if (!(c.x >= 0.0 && c.x <= sz.x && c.y >= 0.0 && c.y <= sz.y)) { return COST_MAX; }
  let layer = i32(cams.c[s].b.w);
  var sws = 0.0;
  var swss = 0.0;
  var swrs = 0.0;
${sampleBody}
  let ms = sws * invSumW;
  let vs = swss * invSumW - ms * ms;
  if (vs < P.minVar) { return COST_MAX; }
  let cov = swrs * invSumW - refMean * ms;
  let ncc = cov / sqrt(refVar * vs);
  return clamp(1.0 - ncc, 0.0, COST_MAX);
}

// Truncated forward-backward reprojection error against source depth map s.
fn geoCost(s: u32, px: vec2f, z: f32) -> f32 {
  let p = vec3f(px, 1.0);
  let q = z * vec3f(dot(cams.c[s].A0.xyz, p), dot(cams.c[s].A1.xyz, p), dot(cams.c[s].A2.xyz, p)) + cams.c[s].b.xyz;
  if (!(q.z > 1e-9)) { return P.tau; }
  let ps = q.xy / q.z;
  let sz = cams.c[s].size.xy - 1.0;
  if (!(ps.x >= 0.0 && ps.x <= sz.x && ps.y >= 0.0 && ps.y <= sz.y)) { return P.tau; }
  let ds = cams.c[s].dsize;
  if (ds.z < 1.0 || ds.w < 1.0) { return P.tau; }
  let di = clamp(vec2i(floor((ps + 0.5) * ds.xy)), vec2i(0), vec2i(ds.zw) - vec2i(1));
  let zs = textureLoad(srcDepth, di, i32(cams.c[s].M0.w), 0).r;
  if (!(zs > 0.0)) { return P.tau; }
  let ph = vec3f(ps, 1.0);
  let r = zs * vec3f(dot(cams.c[s].M0.xyz, ph), dot(cams.c[s].M1.xyz, ph), dot(cams.c[s].M2.xyz, ph)) + cams.c[s].v.xyz;
  if (!(r.z > 1e-9)) { return P.tau; }
  return min(length(r.xy / r.z - px), P.tau);
}

// Plane of neighbour pixel 'best' re-evaluated at this pixel: (n, z) or z = 0 if invalid.
fn regionCandidate(best: i32, ray: vec3f, cxy: vec2f) -> vec4f {
  if (best < 0) { return vec4f(0.0); }
  let W = i32(P.width);
  let pl = planes[best];
  let rq = vec3f((vec2f(f32(best % W), f32(best / W)) - cxy) * P.invF, 1.0);
  let nrp = dot(pl.xyz, ray);
  if (!(nrp < 0.0)) { return vec4f(0.0); }
  let zp = pl.w * dot(pl.xyz, rq) / nrp;
  if (!(zp >= P.zMin && zp <= P.zMax)) { return vec4f(0.0); }
  return vec4f(pl.xyz, zp);
}

@compute @workgroup_size(${WORKGROUP_X}, ${WORKGROUP_Y})
fn init(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x == 0u && gid.y == 0u) { status[0] = status[0] + 1u; }
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
    sc[s] = COST_MAX;
    if (hp.w > 0.0) { sc[s] = viewCost(s, px, hp); }
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
  if (gid.x == 0u && gid.y == 0u) { status[0] = status[0] + 1u; }
  let y = P.rowStart + gid.y;
  if (y >= P.rowEnd) { return; }
  let x = 2u * gid.x + ((y + P.colour) & 1u);
  if (x >= P.width) { return; }
  let W = i32(P.width);
  let H = i32(P.height);
  let xi = i32(x);
  let yi = i32(y);
  let idx = y * P.width + x;
  let idxi = i32(idx);
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
  let cur = planes[idx];
  candN[0] = cur.xyz;
  candZ[0] = cur.w;
  candOk[0] = 1u;
  var best: i32;
  var bestC: f32;
  var cr: vec4f;
${regionLines.join('\n')}

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

// Hypotheses from a prior map (priorToPlanes on the GPU): depth in costs, the normal's
// octahedral coordinates (f32) in photo and views (bits), priorW × priorH (any resolution),
// resampled nearest (§2 rule); a plane is re-intersected with this pixel's ray when the sizes
// differ (kept when nearly edge-on).
@compute @workgroup_size(${WORKGROUP_X}, ${WORKGROUP_Y})
fn priorPlanes(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x == 0u && gid.y == 0u) { status[0] = status[0] + 1u; }
  let x = gid.x;
  let y = P.rowStart + gid.y;
  if (x >= P.width || y >= P.rowEnd) { return; }
  let sx = f32(P.priorW) / f32(P.width);
  let sy = f32(P.priorH) / f32(P.height);
  let qx = clamp(floor((f32(x) + 0.5) * sx), 0.0, f32(P.priorW - 1u));
  let qy = clamp(floor((f32(y) + 0.5) * sy), 0.0, f32(P.priorH - 1u));
  let j = u32(qy) * P.priorW + u32(qx);
  var z = costs[j];
  var pl = vec4f(0.0, 0.0, -1.0, 0.0);
  if (z > 0.0 && z < 3.0e38) {
    var o = vec2f(photo[j], bitcast<f32>(views[j]));
    let oz = 1.0 - abs(o.x) - abs(o.y);
    if (oz < 0.0) { o = (1.0 - abs(o.yx)) * select(vec2f(-1.0), vec2f(1.0), o >= vec2f(0.0)); }
    let n = normalize(vec3f(o, oz));
    if (P.priorW != P.width || P.priorH != P.height) {
      let cxy = vec2f(P.cx, P.cy);
      let nrq = dot(n, vec3f(((vec2f(qx, qy) + 0.5) / vec2f(sx, sy) - 0.5 - cxy) * P.invF, 1.0));
      let nrp = dot(n, vec3f((vec2f(f32(x), f32(y)) - cxy) * P.invF, 1.0));
      let zp = z * nrq / nrp;
      if (nrp < 0.0 && nrq < 0.0 && zp > 0.5 * z && zp < 2.0 * z) { z = zp; }
    }
    pl = vec4f(n, z);
  }
  planes[y * P.width + x] = pl;
}

// Output of a finished run (planesToDepthMap on the GPU, for a compact readback): costs = depth
// (0 = invalid: cost > maxCost, z outside the range, no normal), views = octahedral unit normal,
// photo = cost (non-finite → COST_MAX).
@compute @workgroup_size(${WORKGROUP_X}, ${WORKGROUP_Y})
fn finalize(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x == 0u && gid.y == 0u) { status[0] = status[0] + 1u; }
  let x = gid.x;
  let y = P.rowStart + gid.y;
  if (x >= P.width || y >= P.rowEnd) { return; }
  let idx = y * P.width + x;
  let pl = planes[idx];
  let c = photo[idx];
  var depth = 0.0;
  var nrm = 0u;
  let len = length(pl.xyz);
  if (c <= P.maxCost && pl.w >= P.zMin && pl.w <= P.zMax && len > 0.0) {
    depth = pl.w;
    let n = pl.xyz / len;
    var p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) { p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0)); }
    nrm = pack2x16snorm(p);
  }
  costs[idx] = depth;
  views[idx] = nrm;
  photo[idx] = select(COST_MAX, c, abs(c) <= 3.0e38);
}
`;
}
