/**
 * WGSL shared by the splat kernels: constants, the per-view `Frame` uniform, parameter
 * access into the SoA parameter buffer, and the forward projection (which mirrors
 * `projectGaussian` in ../cpu-reference.ts operation by operation).
 *
 * Parameter buffer (array<f32>, capacity `cap` Gaussians, SoA sections):
 *   [0, 4cap)       mean xyz + pad
 *   [4cap, 8cap)    log-scale xyz + pad
 *   [8cap, 12cap)   quaternion wxyz (unnormalised)
 *   [12cap, 13cap)  opacity logit
 *   [13cap, …)      SH: 3·K floats per Gaussian, coefficient-major (K = 1 or 4)
 * The gradient / Adam-moment buffers use the same layout, so one index addresses a parameter
 * in all of them.
 */
import { ALPHA_MAX, ALPHA_MIN, BLUR_EPS, SH_C0, SH_C1, TILE_SIZE, T_MIN, shCoeffCount, type ShDegree } from '../types';

/** Compile-time options shared by every splat shader. */
export interface SplatShaderConfig {
  shDegree: ShDegree;
  /** Mip-Splatting opacity compensation of the screen-space low-pass filter. */
  antialiased: boolean;
  /**
   * Geometry channels (the trainer's depth / normal supervision, ../cpu-reference.ts): the
   * projection also stores the camera-frame normal (proj stride 4 instead of 3), the rasteriser
   * renders Σ w·(1/z, n) next to the colour, and the backward carries 13 gradients per Gaussian
   * instead of 9. Default false.
   */
  geometry?: boolean;
  /**
   * Proj stride 4 without the geometry channels: the trainer's kernels after its geometry losses
   * end (`SplatGeometryOptions.until`) keep the geometry run's buffer layout. Default false.
   */
  wideProj?: boolean;
}

/** vec4 rows of `proj` per Gaussian. */
export function projStride(cfg: Pick<SplatShaderConfig, 'geometry' | 'wideProj'>): 3 | 4 {
  return cfg.geometry || cfg.wideProj ? 4 : 3;
}

/** 2-D gradient accumulators per Gaussian (wgsl/raster.ts backward → wgsl/backward.ts). */
export function grad2dStride(cfg: Pick<SplatShaderConfig, 'geometry'>): 9 | 13 {
  return cfg.geometry ? 13 : 9;
}

/** Pipeline-cache key of a shader configuration. */
export function shaderKey(cfg: SplatShaderConfig): string {
  return `${cfg.shDegree}-${cfg.antialiased}${cfg.geometry ? '-geo' : cfg.wideProj ? '-wide' : ''}`;
}

/** Byte size of the Frame uniform (must match the struct below). */
export const FRAME_BYTES = 160;

function f(x: number): string {
  const s = String(x);
  return /[.eE]/.test(s) ? s : `${s}.0`;
}

export function constantsWGSL(cfg: SplatShaderConfig): string {
  const K = shCoeffCount(cfg.shDegree);
  return /* wgsl */ `
const TILE: u32 = ${TILE_SIZE}u;
const SH_C0: f32 = ${f(SH_C0)};
const SH_C1: f32 = ${f(SH_C1)};
const ALPHA_MIN: f32 = ${f(ALPHA_MIN)};
const ALPHA_MAX: f32 = ${f(ALPHA_MAX)};
const T_MIN: f32 = ${f(T_MIN)};
const BLUR_EPS: f32 = ${f(BLUR_EPS)};
const SH_K: u32 = ${K}u;
const SH_N: u32 = ${3 * K}u;
const ANTIALIASED: bool = ${cfg.antialiased};
const GEOMETRY: bool = ${!!cfg.geometry};
const PS: u32 = ${projStride(cfg)}u;
const G2D: u32 = ${grad2dStride(cfg)}u;
`;
}

/**
 * Per-view uniform. r0..r2 = rows of the world-to-camera R with t in .w; intr = fx, fy, cx, cy;
 * campos = camera centre + near plane; bg = background rgb; lim = frustum-guard tangents
 * (x+, x−, y+, y−); dims = W, H, tilesX, tilesY; info = count, capacity (parameter layout),
 * intersection capacity, flags.
 */
export const FRAME_WGSL = /* wgsl */ `
struct Frame {
  r0: vec4f,
  r1: vec4f,
  r2: vec4f,
  intr: vec4f,
  campos: vec4f,
  bg: vec4f,
  lim: vec4f,
  dims: vec4u,
  info: vec4u,
  extra: vec4f,
}
`;

/** Parameter loads from a global `params: array<f32>` (declared by the including shader). */
export const PARAMS_WGSL = /* wgsl */ `
fn capacity() -> u32 { return frame.info.y; }
fn offScale() -> u32 { return 4u * capacity(); }
fn offQuat() -> u32 { return 8u * capacity(); }
fn offOpacity() -> u32 { return 12u * capacity(); }
fn offSh() -> u32 { return 13u * capacity(); }
fn loadMean(i: u32) -> vec3f { let o = 4u * i; return vec3f(params[o], params[o + 1u], params[o + 2u]); }
fn loadScale(i: u32) -> vec3f { let o = offScale() + 4u * i; return vec3f(params[o], params[o + 1u], params[o + 2u]); }
fn loadQuat(i: u32) -> vec4f { let o = offQuat() + 4u * i; return vec4f(params[o], params[o + 1u], params[o + 2u], params[o + 3u]); }
fn loadOpacity(i: u32) -> f32 { return params[offOpacity() + i]; }
fn loadSh(i: u32, k: u32) -> vec3f { let o = offSh() + SH_N * i + 3u * k; return vec3f(params[o], params[o + 1u], params[o + 2u]); }
`;

/** Rotation (column-major mat3x3f) of a unit quaternion (w, x, y, z). */
export const ROT_WGSL = /* wgsl */ `
fn rotFromQuat(q: vec4f) -> mat3x3f {
  let w = q.x;
  let x = q.y;
  let y = q.z;
  let z = q.w;
  return mat3x3f(
    vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
    vec3f(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
    vec3f(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)));
}
fn camRot() -> mat3x3f {
  return transpose(mat3x3f(frame.r0.xyz, frame.r1.xyz, frame.r2.xyz));
}
`;

/**
 * Forward projection of Gaussian i (cpu-reference.ts `projectGaussian`). Needs `frame`,
 * `params` and the PARAMS / ROT snippets.
 */
export const PROJECT_WGSL = /* wgsl */ `
struct Proj {
  visible: bool,
  mean2d: vec2f,
  conic: vec3f,
  opacity: f32,
  colour: vec3f,
  rect: vec4u,
  depth: f32,
  normal: vec3f,
}

// Axis of the smallest scale (first on ties, log-scales compared like ../cpu-reference.ts).
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

fn projectGaussian(i: u32) -> Proj {
  var o: Proj;
  o.visible = false;
  let mean = loadMean(i);
  let Wr = camRot();
  let p = Wr * mean + vec3f(frame.r0.w, frame.r1.w, frame.r2.w);
  if (!(p.z >= frame.campos.w)) { return o; }
  let q = loadQuat(i);
  let qn2 = dot(q, q);
  if (!(qn2 > 1e-24)) { return o; }
  let Rq = rotFromQuat(q * inverseSqrt(qn2));
  let s = exp(loadScale(i));
  let M = mat3x3f(Rq[0] * s.x, Rq[1] * s.y, Rq[2] * s.z);
  let Sig = M * transpose(M);
  let rz = 1.0 / p.z;
  let fx = frame.intr.x;
  let fy = frame.intr.y;
  let tx = p.z * clamp(p.x * rz, -frame.lim.y, frame.lim.x);
  let ty = p.z * clamp(p.y * rz, -frame.lim.w, frame.lim.z);
  let J = mat3x2f(vec2f(fx * rz, 0.0), vec2f(0.0, fy * rz), vec2f(-fx * tx * rz * rz, -fy * ty * rz * rz));
  let TM = J * Wr;
  let S2 = TM * Sig * transpose(TM);
  let a = S2[0][0];
  let b = S2[1][0];
  let c = S2[1][1];
  let ap = a + BLUR_EPS;
  let cp = c + BLUR_EPS;
  let det1 = ap * cp - b * b;
  if (!(det1 > 0.0)) { return o; }
  var comp = 1.0;
  if (ANTIALIASED) {
    let det0 = a * c - b * b;
    comp = select(0.0, sqrt(det0 / det1), det0 > 0.0);
  }
  let opac = comp / (1.0 + exp(-loadOpacity(i)));
  if (!(opac >= ALPHA_MIN)) { return o; }
  let k = sqrt(2.0 * log(opac / ALPHA_MIN));
  let rad = k * sqrt(vec2f(ap, cp));
  let m2 = vec2f(fx * p.x * rz + frame.intr.z, fy * p.y * rz + frame.intr.w);
  let lo = max(vec2f(0.0), ceil(m2 - rad));
  let hi = min(vec2f(f32(frame.dims.x) - 1.0, f32(frame.dims.y) - 1.0), floor(m2 + rad));
  if (!(lo.x <= hi.x && lo.y <= hi.y)) { return o; }
  let dv = mean - frame.campos.xyz;
  let dl = length(dv);
  let dir = select(vec3f(0.0, 0.0, 1.0), dv / dl, dl > 0.0);
  o.visible = true;
  o.mean2d = m2;
  o.conic = vec3f(cp, -b, ap) / det1;
  o.opacity = opac;
  o.colour = max(shColour(i, dir), vec3f(0.0));
  o.rect = vec4u(u32(lo.x) / TILE, u32(lo.y) / TILE, u32(hi.x) / TILE, u32(hi.y) / TILE);
  o.depth = p.z;
  if (GEOMETRY) {
    // camera-frame normal: the smallest-scale axis, facing the camera
    let nc = Wr * Rq[minAxis(loadScale(i))];
    o.normal = select(nc, -nc, dot(nc, p) > 0.0);
  }
  return o;
}
`;

/**
 * Exact tile test (cpu-reference.ts `ellipseRectMin` / `tileTouched`): does the ellipse
 * dᵀ·conic·d ≤ tau reach the pixel-centre rectangle of tile (tx, ty)? Needs `frame`.
 */
export const TILE_TEST_WGSL = /* wgsl */ `
fn quadForm(c: vec3f, dx: f32, dy: f32) -> f32 { return c.x * dx * dx + 2.0 * c.y * dx * dy + c.z * dy * dy; }
fn tileTouched(m: vec2f, c: vec3f, tau: f32, tx: u32, ty: u32) -> bool {
  let x0 = f32(tx * TILE);
  let y0 = f32(ty * TILE);
  let x1 = min(x0 + f32(TILE - 1u), f32(frame.dims.x - 1u));
  let y1 = min(y0 + f32(TILE - 1u), f32(frame.dims.y - 1u));
  if (m.x >= x0 && m.x <= x1 && m.y >= y0 && m.y <= y1) { return true; }
  var best = 1e30;
  for (var e = 0u; e < 2u; e++) {
    let dx = select(x0, x1, e == 1u) - m.x;
    let dy = clamp(-c.y * dx / c.z, y0 - m.y, y1 - m.y);
    best = min(best, quadForm(c, dx, dy));
    let dy2 = select(y0, y1, e == 1u) - m.y;
    let dx2 = clamp(-c.y * dy2 / c.x, x0 - m.x, x1 - m.x);
    best = min(best, quadForm(c, dx2, dy2));
  }
  return best <= tau;
}
`;

/** f32 add on a u32 atomic (compare-exchange loop; WGSL has no float atomics). */
export const ATOMIC_ADD_F32_WGSL = (buffer: string): string => /* wgsl */ `
fn atomicAddF32(idx: u32, v: f32) {
  var old = atomicLoad(&${buffer}[idx]);
  loop {
    let r = atomicCompareExchangeWeak(&${buffer}[idx], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}
`;
