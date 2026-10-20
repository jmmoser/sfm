/**
 * WGSL for the per-Gaussian backward pass: projection backward (mirrors
 * `projectBackwardGaussian` in ../cpu-reference.ts) fused with the Adam update, the MCMC
 * regularisers and the MCMC position noise, so the parameter gradients never touch memory.
 * Only Gaussians that were projected into the frame (tile count > 0) are updated
 * ("visible Adam"); the bias correction uses the global step.
 *
 * mode 'adam'  — bindings: frame, opt, params (rw), adamM, adamV, grad2d, tileCount
 * mode 'grads' — bindings: frame, params (read), grad2d, tileCount, gradsOut
 *                (writes the raw gradients in the parameter layout; tests)
 * Both modes read the 9 (13 with the geometry channels) accumulated 2-D gradients of a Gaussian
 * and reset them to 0.
 *
 * Shape regularisers (Adam mode, `opt.shape`): flatten — L1 on the smallest scale (÷ scene extent,
 * mean over the Gaussians: surfel-like discs); anisotropy — max(0, ln s_max − ln s_mid − ln r)
 * (mean over the Gaussians: no needles longer than r × their width).
 */
import { FRAME_WGSL, PARAMS_WGSL, ROT_WGSL, constantsWGSL, type SplatShaderConfig } from './common';

/** Byte size of the Opt uniform. */
export const OPT_BYTES = 112;

const OPT_WGSL = /* wgsl */ `
struct Opt {
  lr0: vec4f,   // learning rates: mean, log-scale, quaternion, opacity logit
  lr1: vec4f,   // SH band 0, SH band 1, position noise (noise lr × mean lr), 0
  adam: vec4f,  // beta1, beta2, epsilon, 0
  bc: vec4f,    // 1 − beta1^t, 1 − beta2^t, 0, 0
  reg: vec4f,   // opacity regulariser / N, scale regulariser / (3N), 0, 0
  seed: vec4u,  // random seed of this step
  shape: vec4f, // flatten / (N · extent), anisotropy / N, ln(max ratio), 0
}
`;

const gradMath = (geo: boolean): string => /* wgsl */ `
struct PGrad {
  mean: vec3f,
  scale: vec3f,
  quat: vec4f,
  opac: f32,
  sh0: vec3f,
  sh1: vec3f,
  sh2: vec3f,
  sh3: vec3f,
  sig: f32,
  s: vec3f,
}

fn finite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }

// Parameter gradients of Gaussian i from its 2-D gradients g2 = [du, dv, dA, dB, dC, do, dr, dg, db]
// (+ [d(1/z), dnx, dny, dnz] with the geometry channels).
fn projBackward(i: u32, g2: array<f32, G2D>) -> PGrad {
  var o: PGrad;
  let mean = loadMean(i);
  let Wr = camRot();
  let p = Wr * mean + vec3f(frame.r0.w, frame.r1.w, frame.r2.w);
  let q = loadQuat(i);
  let qn2 = dot(q, q);
  let qinv = inverseSqrt(qn2);
  let qn = q * qinv;
  let Rq = rotFromQuat(qn);
  let s = exp(loadScale(i));
  let M = mat3x3f(Rq[0] * s.x, Rq[1] * s.y, Rq[2] * s.z);
  let Sig = M * transpose(M);
  let rz = 1.0 / p.z;
  let fx = frame.intr.x;
  let fy = frame.intr.y;
  let xr = p.x * rz;
  let yr = p.y * rz;
  let clampX = xr < -frame.lim.y || xr > frame.lim.x;
  let clampY = yr < -frame.lim.w || yr > frame.lim.z;
  let tx = p.z * clamp(xr, -frame.lim.y, frame.lim.x);
  let ty = p.z * clamp(yr, -frame.lim.w, frame.lim.z);
  let J = mat3x2f(vec2f(fx * rz, 0.0), vec2f(0.0, fy * rz), vec2f(-fx * tx * rz * rz, -fy * ty * rz * rz));
  let TM = J * Wr;
  let S2 = TM * Sig * transpose(TM);
  let a = S2[0][0];
  let b = S2[1][0];
  let c = S2[1][1];
  let ap = a + BLUR_EPS;
  let cp = c + BLUR_EPS;
  let det1 = ap * cp - b * b;
  var comp = 1.0;
  if (ANTIALIASED) {
    let det0 = a * c - b * b;
    comp = select(0.0, sqrt(det0 / det1), det0 > 0.0);
  }
  let sig = 1.0 / (1.0 + exp(-loadOpacity(i)));
  let conA = cp / det1;
  let conB = -b / det1;
  let conC = ap / det1;
  o.sig = sig;
  o.s = s;
  // opacity
  o.opac = g2[5] * comp * sig * (1.0 - sig);
  let dcomp = g2[5] * sig;
  // conic = (Σ₂ + εI)⁻¹ : ∂L/∂Σ' = −Q G_Q Q
  let Q = mat2x2f(vec2f(conA, conB), vec2f(conB, conC));
  let GQ = mat2x2f(vec2f(g2[2], 0.5 * g2[3]), vec2f(0.5 * g2[3], g2[4]));
  let dSp = (Q * GQ * Q) * -1.0;
  var da = dSp[0][0];
  var dbb = 2.0 * dSp[1][0];
  var dc = dSp[1][1];
  if (ANTIALIASED && comp > 0.0) {
    let c2 = comp * comp;
    let f = dcomp / (2.0 * comp * det1);
    da += f * (c - c2 * cp);
    dc += f * (a - c2 * ap);
    dbb += f * (-2.0 * b * (1.0 - c2));
  }
  let G2 = mat2x2f(vec2f(da, 0.5 * dbb), vec2f(0.5 * dbb, dc));
  let dTM = 2.0 * (G2 * TM * Sig);
  let dSig = transpose(TM) * G2 * TM;
  let dJ = dTM * transpose(Wr);
  let rz2 = rz * rz;
  let rz3 = rz2 * rz;
  var dp = vec3f(fx * rz * g2[0], fy * rz * g2[1], -fx * p.x * rz2 * g2[0] - fy * p.y * rz2 * g2[1]);
  dp.z += dJ[0][0] * (-fx * rz2) + dJ[2][0] * (2.0 * fx * tx * rz3) + dJ[1][1] * (-fy * rz2) + dJ[2][1] * (2.0 * fy * ty * rz3);
  ${geo ? 'dp.z -= g2[9] * rz2;' : ''}
  let dtx = dJ[2][0] * (-fx * rz2);
  let dty = dJ[2][1] * (-fy * rz2);
  if (clampX) { dp.z += dtx * tx * rz; } else { dp.x += dtx; }
  if (clampY) { dp.z += dty * ty * rz; } else { dp.y += dty; }
  var dmean = transpose(Wr) * dp;
  // colour
  let dv = mean - frame.campos.xyz;
  let dl = length(dv);
  let dir = select(vec3f(0.0, 0.0, 1.0), dv / dl, dl > 0.0);
  let colRaw = shColour(i, dir);
  let dc3 = select(vec3f(0.0), vec3f(g2[6], g2[7], g2[8]), colRaw > vec3f(0.0));
  o.sh0 = SH_C0 * dc3;
  if (SH_K > 1u) {
    o.sh1 = -SH_C1 * dir.y * dc3;
    o.sh2 = SH_C1 * dir.z * dc3;
    o.sh3 = -SH_C1 * dir.x * dc3;
    let ddir = vec3f(-SH_C1 * dot(loadSh(i, 3u), dc3), -SH_C1 * dot(loadSh(i, 1u), dc3), SH_C1 * dot(loadSh(i, 2u), dc3));
    if (dl > 0.0) { dmean += (ddir - dir * dot(dir, ddir)) / dl; }
  }
  o.mean = dmean;
  // Σ = M Mᵀ, M = Rq diag(s)
  let dM = 2.0 * (dSig * M);
  let ds = vec3f(dot(dM[0], Rq[0]), dot(dM[1], Rq[1]), dot(dM[2], Rq[2]));
  o.scale = ds * s;
  var G = mat3x3f(dM[0] * s.x, dM[1] * s.y, dM[2] * s.z); // ∂L/∂Rq, G[c][r]
  ${
    geo
      ? `{
    // n = ±Wr · Rq[k] (smallest-scale axis, facing the camera): ∂L/∂Rq[k] += ±Wrᵀ ∂L/∂n
    let k = minAxis(loadScale(i));
    let nc = Wr * Rq[k];
    let sg = select(1.0, -1.0, dot(nc, p) > 0.0);
    G[k] += sg * (transpose(Wr) * vec3f(g2[10], g2[11], g2[12]));
  }`
      : ''
  }
  let w = qn.x;
  let x = qn.y;
  let y = qn.z;
  let z = qn.w;
  let dw = 2.0 * (-z * G[1][0] + y * G[2][0] + z * G[0][1] - x * G[2][1] - y * G[0][2] + x * G[1][2]);
  let dx = 2.0 * (y * G[1][0] + z * G[2][0] + y * G[0][1] - 2.0 * x * G[1][1] - w * G[2][1] + z * G[0][2] + w * G[1][2] - 2.0 * x * G[2][2]);
  let dy = 2.0 * (-2.0 * y * G[0][0] + x * G[1][0] + w * G[2][0] + x * G[0][1] + z * G[2][1] - w * G[0][2] + z * G[1][2] - 2.0 * y * G[2][2]);
  let dz = 2.0 * (-2.0 * z * G[0][0] - w * G[1][0] + x * G[2][0] + w * G[0][1] - 2.0 * z * G[1][1] + y * G[2][1] + x * G[0][2] + y * G[1][2]);
  let dq = vec4f(dw, dx, dy, dz);
  o.quat = (dq - qn * dot(qn, dq)) * qinv;
  return o;
}

fn minAxis(ls: vec3f) -> u32 {
  var k = 0u;
  if (ls.y < ls[k]) { k = 1u; }
  if (ls.z < ls[k]) { k = 2u; }
  return k;
}

fn shColour(i: u32, dir: vec3f) -> vec3f {
  var c = SH_C0 * loadSh(i, 0u);
  if (SH_K > 1u) {
    c += SH_C1 * (-dir.y * loadSh(i, 1u) + dir.z * loadSh(i, 2u) - dir.x * loadSh(i, 3u));
  }
  return c + vec3f(0.5);
}
`;

export function backwardWGSL(cfg: SplatShaderConfig, mode: 'adam' | 'grads'): string {
  const adam = mode === 'adam';
  return /* wgsl */ `
${constantsWGSL(cfg)}
${FRAME_WGSL}
${OPT_WGSL}
@group(0) @binding(0) var<uniform> frame: Frame;
${
  adam
    ? `@group(0) @binding(1) var<uniform> opt: Opt;
@group(0) @binding(2) var<storage, read_write> params: array<f32>;
@group(0) @binding(3) var<storage, read_write> adamM: array<f32>;
@group(0) @binding(4) var<storage, read_write> adamV: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad2d: array<u32>;
@group(0) @binding(6) var<storage, read> tileCount: array<u32>;`
    : `@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad2d: array<u32>;
@group(0) @binding(3) var<storage, read> tileCount: array<u32>;
@group(0) @binding(4) var<storage, read_write> gradsOut: array<f32>;`
}
${PARAMS_WGSL}
${ROT_WGSL}
${gradMath(!!cfg.geometry)}
${adam ? ADAM_FNS : ''}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= frame.info.x) { return; }
  if (tileCount[i] == 0u) { return; }
  var g2: array<f32, G2D>;
  for (var c = 0u; c < G2D; c++) {
    g2[c] = bitcast<f32>(grad2d[G2D * i + c]);
    grad2d[G2D * i + c] = 0u;
  }
  var g = projBackward(i, g2);
${adam ? ADAM_BODY : GRADS_BODY}
}
`;
}

const GRADS_BODY = /* wgsl */ `
  let om = 4u * i;
  let os = offScale() + 4u * i;
  let oq = offQuat() + 4u * i;
  for (var k = 0u; k < 3u; k++) {
    gradsOut[om + k] = g.mean[k];
    gradsOut[os + k] = g.scale[k];
  }
  for (var k = 0u; k < 4u; k++) { gradsOut[oq + k] = g.quat[k]; }
  gradsOut[offOpacity() + i] = g.opac;
  let sh = offSh() + SH_N * i;
  for (var k = 0u; k < 3u; k++) { gradsOut[sh + k] = g.sh0[k]; }
  if (SH_K > 1u) {
    for (var k = 0u; k < 3u; k++) {
      gradsOut[sh + 3u + k] = g.sh1[k];
      gradsOut[sh + 6u + k] = g.sh2[k];
      gradsOut[sh + 9u + k] = g.sh3[k];
    }
  }
`;

const ADAM_FNS = /* wgsl */ `
fn adamStep(idx: u32, g: f32, lr: f32) -> f32 {
  let m = opt.adam.x * adamM[idx] + (1.0 - opt.adam.x) * g;
  let v = opt.adam.y * adamV[idx] + (1.0 - opt.adam.y) * g * g;
  adamM[idx] = m;
  adamV[idx] = v;
  let p = params[idx] - lr * (m / opt.bc.x) / (sqrt(v / opt.bc.y) + opt.adam.z);
  params[idx] = p;
  return p;
}

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn rand01(h: u32) -> f32 { return (f32(pcg(h) >> 8u) + 0.5) * (1.0 / 16777216.0); }

fn gauss2(h: u32) -> vec2f {
  let u1 = rand01(h);
  let u2 = rand01(h ^ 0x9e3779b9u);
  let r = sqrt(-2.0 * log(u1));
  return vec2f(r * cos(6.283185307 * u2), r * sin(6.283185307 * u2));
}
`;

const ADAM_BODY = /* wgsl */ `
  // MCMC regularisers (opacity and scale L1, means over all Gaussians)
  g.opac += opt.reg.x * g.sig * (1.0 - g.sig);
  g.scale += opt.reg.y * g.s;
  // shape regularisers: flatten (smallest scale → 0), anisotropy cap (s_max ≤ r · s_mid)
  if (opt.shape.x > 0.0 || opt.shape.y > 0.0) {
    let ls = loadScale(i);
    var kmin = 0u;
    var kmax = 0u;
    for (var k = 1u; k < 3u; k++) {
      if (ls[k] < ls[kmin]) { kmin = k; }
      if (ls[k] > ls[kmax]) { kmax = k; }
    }
    let kmid = select(3u - kmin - kmax, 1u, kmin == kmax);
    g.scale[kmin] += opt.shape.x * g.s[kmin];
    if (kmin != kmax && ls[kmax] - ls[kmid] > opt.shape.z) {
      g.scale[kmax] += opt.shape.y;
      g.scale[kmid] -= opt.shape.y;
    }
  }
  let tot = dot(g.mean, vec3f(1.0)) + dot(g.scale, vec3f(1.0)) + dot(g.quat, vec4f(1.0)) + g.opac
    + dot(g.sh0, vec3f(1.0)) + dot(g.sh1, vec3f(1.0)) + dot(g.sh2, vec3f(1.0)) + dot(g.sh3, vec3f(1.0));
  if (!finite(tot)) { return; }
  let om = 4u * i;
  let os = offScale() + 4u * i;
  let oq = offQuat() + 4u * i;
  var mean: vec3f;
  var ls: vec3f;
  var q: vec4f;
  for (var k = 0u; k < 3u; k++) {
    mean[k] = adamStep(om + k, g.mean[k], opt.lr0.x);
    ls[k] = adamStep(os + k, g.scale[k], opt.lr0.y);
  }
  for (var k = 0u; k < 4u; k++) { q[k] = adamStep(oq + k, g.quat[k], opt.lr0.z); }
  let op = adamStep(offOpacity() + i, g.opac, opt.lr0.w);
  let sh = offSh() + SH_N * i;
  for (var k = 0u; k < 3u; k++) { _ = adamStep(sh + k, g.sh0[k], opt.lr1.x); }
  if (SH_K > 1u) {
    for (var k = 0u; k < 3u; k++) {
      _ = adamStep(sh + 3u + k, g.sh1[k], opt.lr1.y);
      _ = adamStep(sh + 6u + k, g.sh2[k], opt.lr1.y);
      _ = adamStep(sh + 9u + k, g.sh3[k], opt.lr1.y);
    }
  }
  // MCMC noise: dμ = noise · sigmoid(−100 (o − 0.005)) · Σ ε  (large for nearly transparent Gaussians)
  if (opt.lr1.z > 0.0) {
    let o = 1.0 / (1.0 + exp(-op));
    let gate = 1.0 / (1.0 + exp(100.0 * (o - 0.005)));
    if (gate > 1e-4) {
      let qq = dot(q, q);
      if (qq > 1e-24) {
        let R = rotFromQuat(q * inverseSqrt(qq));
        let s = exp(ls);
        let Mn = mat3x3f(R[0] * s.x, R[1] * s.y, R[2] * s.z);
        let h = pcg(i * 3u + opt.seed.x);
        let e01 = gauss2(h);
        let e2 = gauss2(pcg(h + 1u)).x;
        let d = (Mn * transpose(Mn)) * vec3f(e01, e2) * (opt.lr1.z * gate);
        if (finite(d.x + d.y + d.z)) {
          for (var k = 0u; k < 3u; k++) { params[om + k] = mean[k] + d[k]; }
        }
      }
    }
  }
`;

/** MCMC relocation: copy (dst ← src) entries, then source updates (new opacity, scaled size). */
export function relocateWGSL(cfg: SplatShaderConfig, stage: 'copy' | 'update'): string {
  return /* wgsl */ `
${constantsWGSL(cfg)}
struct RCfg { nCopies: u32, nUpdates: u32, capacity: u32, pad: u32 }
@group(0) @binding(0) var<uniform> rc: RCfg;
@group(0) @binding(1) var<storage, read> list: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> params: array<f32>;
@group(0) @binding(3) var<storage, read_write> adamM: array<f32>;
@group(0) @binding(4) var<storage, read_write> adamV: array<f32>;
fn zeroMoments(i: u32) {
  let cap = rc.capacity;
  for (var k = 0u; k < 4u; k++) {
    adamM[4u * i + k] = 0.0; adamV[4u * i + k] = 0.0;
    adamM[4u * cap + 4u * i + k] = 0.0; adamV[4u * cap + 4u * i + k] = 0.0;
    adamM[8u * cap + 4u * i + k] = 0.0; adamV[8u * cap + 4u * i + k] = 0.0;
  }
  adamM[12u * cap + i] = 0.0; adamV[12u * cap + i] = 0.0;
  for (var k = 0u; k < SH_N; k++) { adamM[13u * cap + SH_N * i + k] = 0.0; adamV[13u * cap + SH_N * i + k] = 0.0; }
}
fn setOpacityScale(i: u32, logitBits: u32, deltaBits: u32) {
  let cap = rc.capacity;
  params[12u * cap + i] = bitcast<f32>(logitBits);
  let d = bitcast<f32>(deltaBits);
  for (var k = 0u; k < 3u; k++) { params[4u * cap + 4u * i + k] = params[4u * cap + 4u * i + k] + d; }
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let e = gid.x;
  let cap = rc.capacity;
  ${
    stage === 'copy'
      ? `if (e >= rc.nCopies) { return; }
  let it = list[e];
  let dst = it.x;
  let src = it.y;
  for (var k = 0u; k < 4u; k++) {
    params[4u * dst + k] = params[4u * src + k];
    params[4u * cap + 4u * dst + k] = params[4u * cap + 4u * src + k];
    params[8u * cap + 4u * dst + k] = params[8u * cap + 4u * src + k];
  }
  for (var k = 0u; k < SH_N; k++) { params[13u * cap + SH_N * dst + k] = params[13u * cap + SH_N * src + k]; }
  setOpacityScale(dst, it.z, it.w);
  zeroMoments(dst);`
      : `if (e >= rc.nUpdates) { return; }
  let it = list[rc.nCopies + e];
  setOpacityScale(it.x, it.z, it.w);
  zeroMoments(it.x);`
  }
}
`;
}
