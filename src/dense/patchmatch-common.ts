/**
 * Shared pieces of the CPU and GPU PatchMatch implementations: option
 * resolution, camera / homography set-up, PCG random numbers, the ACMH adaptive
 * checkerboard sampling pattern, prior resampling and output finalisation.
 *
 * ## Plane-induced homography (sign convention)
 *
 * A hypothesis at reference pixel p = (u, v) is (z, n): z-depth and unit normal n
 * in the reference camera frame, facing the camera (n · X < 0). With the ray
 * r(p) = K_r⁻¹ (u, v, 1)ᵀ, the 3D point is X = z r(p) and the plane is
 *     { X : nᵀX + δ = 0 },  δ = −nᵀ(z r(p)) = −z (n · r(p)) > 0
 * (δ = distance of the plane from the reference camera centre). With the relative
 * pose ref → source computed from world-to-camera poses,
 *     R_sr = R_s R_rᵀ,  t_sr = t_s − R_sr t_r   (X_s = R_sr X_r + t_sr),
 * every plane point satisfies −nᵀX_r / δ = 1, so
 *     X_s = R_sr X_r + t_sr (−nᵀX_r / δ) = (R_sr − t_sr nᵀ / δ) X_r
 * and the homography mapping reference pixels to source pixels is
 *     H = K_s (R_sr − t_sr nᵀ / δ) K_r⁻¹ = A − b mᵀ / δ,
 *     A = K_s R_sr K_r⁻¹,  b = K_s t_sr,  m = K_r⁻ᵀ n   (so m · (u, v, 1) = n · r(p)).
 * (Equivalently, with the plane written nᵀX = d, d = −δ < 0 and H = K_s(R_sr + t_sr nᵀ/d)K_r⁻¹.)
 *
 * ## Geometric consistency (ACMM)
 *
 * X_r = z r(p) projects into source s at p_s ≅ z A (u, v, 1)ᵀ + b. The source depth
 * map is read (nearest) at p_s, giving z_s; X_s' = z_s K_s⁻¹ p̃_s maps back with
 *     K_r X_r' = z_s M p̃_s + v,  M = K_r R_rs K_s⁻¹,  v = K_r t_rs  (R_rs = R_srᵀ, t_rs = −R_srᵀ t_sr)
 * and the error is |π(K_r X_r') − p| truncated at τ.
 */
import type { CameraIntrinsics, Pose, RGBAImage } from '../core/types';
import {
  DEFAULT_PATCHMATCH_OPTIONS,
  MAX_PATCHMATCH_SOURCES,
  type DenseView,
  type DepthMap,
  type DepthRange,
  type GeometricInput,
  type PatchMatchOptions,
} from './types';

export const COST_MAX = 2;

/** Fully resolved options (every field defined). */
export type ResolvedPatchMatchOptions = Required<PatchMatchOptions>;

export function resolvePatchMatchOptions(opts?: Partial<PatchMatchOptions>): ResolvedPatchMatchOptions {
  const o = { ...DEFAULT_PATCHMATCH_OPTIONS, ...(opts ?? {}) } as ResolvedPatchMatchOptions;
  for (const k of Object.keys(DEFAULT_PATCHMATCH_OPTIONS) as (keyof PatchMatchOptions)[]) {
    if (o[k] === undefined) (o as unknown as Record<string, unknown>)[k] = DEFAULT_PATCHMATCH_OPTIONS[k];
  }
  // non-positive / NaN sigmas would make every bilateral weight 0 or NaN
  if (!(o.sigmaSpatial > 0)) o.sigmaSpatial = DEFAULT_PATCHMATCH_OPTIONS.sigmaSpatial;
  if (!(o.sigmaColor > 0)) o.sigmaColor = DEFAULT_PATCHMATCH_OPTIONS.sigmaColor;
  o.iterations = Math.max(0, Math.floor(o.iterations));
  o.geometricIterations = Math.max(0, Math.floor(o.geometricIterations));
  o.patchRadius = Math.max(1, Math.floor(o.patchRadius));
  o.patchStep = Math.max(1, Math.floor(o.patchStep));
  o.numSamplesPerIter = Math.min(6, Math.max(0, Math.floor(o.numSamplesPerIter)));
  o.topK = Math.max(1, Math.floor(o.topK));
  return o;
}

/** Window offsets −r, −r+step, … ≤ r (same as ACMM). */
export function patchOffsets(radius: number, step: number): number[] {
  const out: number[] = [];
  for (let i = -radius; i <= radius; i += step) out.push(i);
  return out;
}

/** Whether geometric mode is active for the given arguments. */
export function isGeometricActive(opts: Partial<PatchMatchOptions> | undefined, geometric: GeometricInput | undefined): boolean {
  if (opts?.geometric === true && !geometric) {
    throw new Error('PatchMatch: options.geometric = true requires the `geometric` argument (prior + sourceDepths)');
  }
  return !!geometric && opts?.geometric !== false;
}

// ---------------------------------------------------------------------------
// Images

/** Rec.709 luma of sRGB values, rounded to 8 bit (identical to an r8unorm upload). */
export function rgbaToGray8(img: RGBAImage, out?: Uint8Array): Uint8Array {
  const n = img.width * img.height;
  const g = out && out.length >= n ? out : new Uint8Array(n);
  const d = img.data;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    g[i] = Math.round(0.2126 * d[j] + 0.7152 * d[j + 1] + 0.0722 * d[j + 2]);
  }
  return g;
}

// ---------------------------------------------------------------------------
// Cameras

/** Per-source camera data (see file header). */
export interface SourceCamera {
  A: Float64Array; // 9, K_s R_sr K_r⁻¹
  b: Float64Array; // 3, K_s t_sr
  M: Float64Array; // 9, K_r R_rs K_s⁻¹
  v: Float64Array; // 3, K_r t_rs
  width: number;
  height: number;
  /** Geometric mode: source depth-map size and image → depth-map scale. */
  depthWidth: number;
  depthHeight: number;
  depthScaleX: number;
  depthScaleY: number;
}

function kMat(c: CameraIntrinsics): Float64Array {
  return new Float64Array([c.f, 0, c.cx, 0, c.f, c.cy, 0, 0, 1]);
}

function kInv(c: CameraIntrinsics): Float64Array {
  const fi = 1 / c.f;
  return new Float64Array([fi, 0, -c.cx * fi, 0, fi, -c.cy * fi, 0, 0, 1]);
}

function mul3(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const o = new Float64Array(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) o[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c];
  }
  return o;
}

function mulT3(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  // a bᵀ
  const o = new Float64Array(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) o[3 * r + c] = a[3 * r] * b[3 * c] + a[3 * r + 1] * b[3 * c + 1] + a[3 * r + 2] * b[3 * c + 2];
  }
  return o;
}

function transpose3(a: ArrayLike<number>): Float64Array {
  return new Float64Array([a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]);
}

function mulVec3(a: ArrayLike<number>, v: ArrayLike<number>): Float64Array {
  return new Float64Array([
    a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
    a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
    a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
  ]);
}

/** Relative pose ref → src: X_s = R_sr X_r + t_sr. */
export function relativePose(ref: Pose, src: Pose): { R: Float64Array; t: Float64Array } {
  const R = mulT3(src.R, ref.R); // R_s R_rᵀ
  const Rt = mulVec3(R, ref.t);
  return { R, t: new Float64Array([src.t[0] - Rt[0], src.t[1] - Rt[1], src.t[2] - Rt[2]]) };
}

export function buildSourceCamera(ref: DenseView, src: DenseView, depthSize?: { width: number; height: number }): SourceCamera {
  const { R, t } = relativePose(ref.pose, src.pose);
  const Ks = kMat(src.cam);
  const Kr = kMat(ref.cam);
  const A = mul3(mul3(Ks, R), kInv(ref.cam));
  const b = mulVec3(Ks, t);
  const Rrs = transpose3(R);
  const trs = mulVec3(Rrs, t);
  trs[0] = -trs[0];
  trs[1] = -trs[1];
  trs[2] = -trs[2];
  const M = mul3(mul3(Kr, Rrs), kInv(src.cam));
  const v = mulVec3(Kr, trs);
  const dw = depthSize?.width ?? src.cam.width;
  const dh = depthSize?.height ?? src.cam.height;
  return {
    A,
    b,
    M,
    v,
    width: src.cam.width,
    height: src.cam.height,
    depthWidth: dw,
    depthHeight: dh,
    depthScaleX: dw / src.cam.width,
    depthScaleY: dh / src.cam.height,
  };
}

/**
 * Homography (row-major 3×3) mapping reference pixels to source pixels for the
 * plane through depth z at pixel (u, v) with normal n (reference camera frame).
 * Returns null if the plane is seen edge-on / from behind.
 */
export function planeHomography(ref: CameraIntrinsics, sc: SourceCamera, u: number, v: number, z: number, n: ArrayLike<number>): Float64Array | null {
  const rx = (u - ref.cx) / ref.f;
  const ry = (v - ref.cy) / ref.f;
  const nr = n[0] * rx + n[1] * ry + n[2];
  const delta = -z * nr;
  if (!(delta > 0)) return null;
  const m0 = n[0] / ref.f;
  const m1 = n[1] / ref.f;
  const m2 = n[2] - (n[0] * ref.cx + n[1] * ref.cy) / ref.f;
  const H = new Float64Array(9);
  for (let r = 0; r < 3; r++) {
    const s = sc.b[r] / delta;
    H[3 * r] = sc.A[3 * r] - s * m0;
    H[3 * r + 1] = sc.A[3 * r + 1] - s * m1;
    H[3 * r + 2] = sc.A[3 * r + 2] - s * m2;
  }
  return H;
}

/**
 * 40 floats (10 vec4) per source, std140/std430 compatible layout used by the WGSL. The w of `b`
 * holds the source's layer in the grey-image texture array, the w of `M0` its layer in the
 * depth-map array (defaults: the source index).
 */
export const GPU_SOURCE_CAMERA_FLOATS = 40;

export function packSourceCameras(
  cams: SourceCamera[],
  texWidth: number,
  texHeight: number,
  out?: Float32Array,
  layers?: { image?: ArrayLike<number>; depth?: ArrayLike<number> },
): Float32Array {
  const o = out ?? new Float32Array(MAX_PATCHMATCH_SOURCES * GPU_SOURCE_CAMERA_FLOATS);
  o.fill(0);
  cams.forEach((c, i) => {
    const base = i * GPU_SOURCE_CAMERA_FLOATS;
    for (let r = 0; r < 3; r++) {
      o[base + 4 * r] = c.A[3 * r];
      o[base + 4 * r + 1] = c.A[3 * r + 1];
      o[base + 4 * r + 2] = c.A[3 * r + 2];
    }
    o[base + 12] = c.b[0];
    o[base + 13] = c.b[1];
    o[base + 14] = c.b[2];
    o[base + 15] = layers?.image?.[i] ?? i;
    o[base + 19] = layers?.depth?.[i] ?? i;
    for (let r = 0; r < 3; r++) {
      o[base + 16 + 4 * r] = c.M[3 * r];
      o[base + 16 + 4 * r + 1] = c.M[3 * r + 1];
      o[base + 16 + 4 * r + 2] = c.M[3 * r + 2];
    }
    o[base + 28] = c.v[0];
    o[base + 29] = c.v[1];
    o[base + 30] = c.v[2];
    // size: imgW, imgH, 1/texW, 1/texH
    o[base + 32] = c.width;
    o[base + 33] = c.height;
    o[base + 34] = 1 / texWidth;
    o[base + 35] = 1 / texHeight;
    // dsize: depthScaleX, depthScaleY, depthW, depthH
    o[base + 36] = c.depthScaleX;
    o[base + 37] = c.depthScaleY;
    o[base + 38] = c.depthWidth;
    o[base + 39] = c.depthHeight;
  });
  return o;
}

// ---------------------------------------------------------------------------
// Random numbers (identical on CPU and GPU)

/** PCG hash (Jarzynski & Olano 2020) on u32. */
export function pcgHash(v: number): number {
  const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  const word = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

/** Seed of one pass (init = pass 0xffff). */
export function passSeed(seed: number, pass: number): number {
  return pcgHash((seed >>> 0) ^ pcgHash((pass + 0x9e3779b9) >>> 0));
}

// ---------------------------------------------------------------------------
// ACMH adaptive checkerboard sampling

/**
 * 8 regions: 0 up-near (V, 7 px), 1 up-far (strip, 11 px), 2 down-near, 3 down-far,
 * 4 left-near, 5 left-far, 6 right-near, 7 right-far. All offsets have odd
 * |dx|+|dy|, i.e. they lie on the other checkerboard colour.
 */
export const CHECKER_REGION_SIZES = [7, 11, 7, 11, 7, 11, 7, 11] as const;

function buildCheckerOffsets(): Int32Array {
  const upNear: [number, number][] = [
    [0, -1],
    [-1, -2],
    [1, -2],
    [-2, -3],
    [2, -3],
    [-3, -4],
    [3, -4],
  ];
  const upFar: [number, number][] = [];
  for (let k = 0; k < 11; k++) upFar.push([0, -(3 + 2 * k)]);
  const tf: ((d: [number, number]) => [number, number])[] = [
    (d) => d, // up
    (d) => [d[0], -d[1]], // down
    (d) => [d[1], d[0]], // left
    (d) => [-d[1], d[0]], // right
  ];
  const out: number[] = [];
  for (const f of tf) {
    for (const d of upNear) out.push(...f(d));
    for (const d of upFar) out.push(...f(d));
  }
  return Int32Array.from(out);
}

/** Flat (dx, dy) pairs, regions concatenated in CHECKER_REGION_SIZES order (72 pairs). */
export const CHECKER_OFFSETS: Int32Array = buildCheckerOffsets();

/**
 * Procedural form of CHECKER_OFFSETS (the WGSL `checkerOffset` mirrors this exactly):
 * offset k of region r.
 */
export function checkerOffset(region: number, k: number): [number, number] {
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

// ---------------------------------------------------------------------------
// Prior / output

/**
 * Hypothesis buffer (4 floats per pixel: nx, ny, nz, depth) from a prior depth
 * map at any resolution (nearest; §2 resampling rule). Invalid pixels have depth 0.
 *
 * With `cam` (the intrinsics at `width × height`), a resampled plane is
 * re-intersected with the target pixel's ray (z_p = z_q (n·r_q)/(n·r_p), r_q the
 * ray through the prior pixel centre), so a slanted plane from a ½-resolution
 * prior does not carry a depth offset of up to half a coarse pixel along its slope.
 */
export function priorToPlanes(prior: DepthMap, width: number, height: number, out?: Float32Array, cam?: CameraIntrinsics): Float32Array {
  const planes = out ?? new Float32Array(width * height * 4);
  const sx = prior.width / width;
  const sy = prior.height / height;
  const reproject = !!cam && (prior.width !== width || prior.height !== height);
  const invF = cam ? 1 / cam.f : 0;
  for (let y = 0; y < height; y++) {
    const py = Math.min(prior.height - 1, Math.max(0, Math.round((y + 0.5) * sy - 0.5)));
    for (let x = 0; x < width; x++) {
      const px = Math.min(prior.width - 1, Math.max(0, Math.round((x + 0.5) * sx - 0.5)));
      const j = py * prior.width + px;
      const i = 4 * (y * width + x);
      let z = prior.depth[j];
      if (z > 0 && Number.isFinite(z)) {
        const nx = prior.normal[3 * j];
        const ny = prior.normal[3 * j + 1];
        const nz = prior.normal[3 * j + 2];
        if (reproject) {
          // prior pixel centre in target pixel coordinates (§2 rule, inverse direction)
          const qx = (px + 0.5) / sx - 0.5;
          const qy = (py + 0.5) / sy - 0.5;
          const nrq = nx * (qx - cam!.cx) * invF + ny * (qy - cam!.cy) * invF + nz;
          const nrp = nx * (x - cam!.cx) * invF + ny * (y - cam!.cy) * invF + nz;
          const zp = (z * nrq) / nrp;
          // keep the prior depth when the plane is (nearly) edge-on to this ray
          if (nrp < 0 && nrq < 0 && zp > 0.5 * z && zp < 2 * z) z = zp;
        }
        planes[i] = nx;
        planes[i + 1] = ny;
        planes[i + 2] = nz;
        planes[i + 3] = z;
      } else {
        planes[i] = 0;
        planes[i + 1] = 0;
        planes[i + 2] = -1;
        planes[i + 3] = 0;
      }
    }
  }
  return planes;
}

/**
 * Unit normal → octahedral snorm16 × 2 (WGSL `pack2x16snorm` layout: x in the low 16 bits). The
 * GPU finalisation writes this; `octDecode` inverts it to ≤ 6.5e-5 (3·√2·0.5/32767).
 */
export function octEncode(nx: number, ny: number, nz: number): number {
  const s = Math.abs(nx) + Math.abs(ny) + Math.abs(nz);
  if (!(s > 0)) return 0;
  let px = nx / s;
  let py = ny / s;
  if (nz < 0) {
    const qx = (1 - Math.abs(py)) * (px >= 0 ? 1 : -1);
    const qy = (1 - Math.abs(px)) * (py >= 0 ? 1 : -1);
    px = qx;
    py = qy;
  }
  const ix = Math.round(Math.max(-1, Math.min(1, px)) * 32767) & 0xffff;
  const iy = Math.round(Math.max(-1, Math.min(1, py)) * 32767) & 0xffff;
  return (ix | (iy << 16)) >>> 0;
}

/** Unit normal → octahedral coordinates in [−1, 1]² (unquantised `octEncode`), written to `out[o..o+1]`. */
export function octCoords(nx: number, ny: number, nz: number, out: Float32Array, o: number): void {
  const s = Math.abs(nx) + Math.abs(ny) + Math.abs(nz);
  let px = s > 0 ? nx / s : 0;
  let py = s > 0 ? ny / s : 0;
  if (nz < 0) {
    const qx = (1 - Math.abs(py)) * (px >= 0 ? 1 : -1);
    const qy = (1 - Math.abs(px)) * (py >= 0 ? 1 : -1);
    px = qx;
    py = qy;
  }
  out[o] = px;
  out[o + 1] = py;
}

/** Octahedral snorm16 × 2 → unit normal written to `out[o..o+2]`. */
export function octDecode(v: number, out: Float32Array, o: number): void {
  let x = Math.max(-1, ((v << 16) >> 16) / 32767);
  let y = Math.max(-1, (v >> 16) / 32767);
  const z = 1 - Math.abs(x) - Math.abs(y);
  if (z < 0) {
    const qx = (1 - Math.abs(y)) * (x >= 0 ? 1 : -1);
    const qy = (1 - Math.abs(x)) * (y >= 0 ? 1 : -1);
    x = qx;
    y = qy;
  }
  const l = 1 / Math.sqrt(x * x + y * y + z * z);
  out[o] = x * l;
  out[o + 1] = y * l;
  out[o + 2] = z * l;
}

/** Final DepthMap from the hypothesis buffer + photometric costs (invalidation rules applied). */
export function planesToDepthMap(planes: Float32Array, photoCost: Float32Array, width: number, height: number, range: DepthRange, maxCost: number): DepthMap {
  const n = width * height;
  const depth = new Float32Array(n);
  const normal = new Float32Array(3 * n);
  const cost = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const z = planes[4 * i + 3];
    const c = photoCost[i];
    cost[i] = Number.isFinite(c) ? c : COST_MAX;
    if (!(c <= maxCost) || !(z >= range.min && z <= range.max)) continue;
    let nx = planes[4 * i];
    let ny = planes[4 * i + 1];
    let nz = planes[4 * i + 2];
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (!(len > 0)) continue;
    nx /= len;
    ny /= len;
    nz /= len;
    depth[i] = z;
    normal[3 * i] = nx;
    normal[3 * i + 1] = ny;
    normal[3 * i + 2] = nz;
  }
  return { width, height, depth, normal, cost };
}

/** All-invalid depth map. */
export function emptyDepthMap(width: number, height: number): DepthMap {
  const n = width * height;
  return { width, height, depth: new Float32Array(n), normal: new Float32Array(3 * n), cost: new Float32Array(n).fill(COST_MAX) };
}

/** Clamp the source list to MAX_PATCHMATCH_SOURCES (warns once per call when truncating). */
export function clampSources<T>(sources: T[], label: string): T[] {
  if (sources.length <= MAX_PATCHMATCH_SOURCES) return sources;
  console.warn(`[${label}] ${sources.length} source views given; using the first ${MAX_PATCHMATCH_SOURCES}`);
  return sources.slice(0, MAX_PATCHMATCH_SOURCES);
}

/** Validates the depth range. */
export function checkRange(range: DepthRange): void {
  if (!(range.min > 0) || !(range.max > range.min) || !Number.isFinite(range.max)) {
    throw new Error(`PatchMatch: invalid depth range [${range.min}, ${range.max}]`);
  }
}

/**
 * Relative depth perturbation of iteration `it`. `fromPrior`: the hypotheses were
 * initialised from a prior depth map (geometric pass or prior-initialised
 * photometric pass), which starts from the smaller `geometricPerturbation`.
 */
export function depthPerturbation(o: ResolvedPatchMatchOptions, it: number, fromPrior: boolean): number {
  const p0 = fromPrior ? o.geometricPerturbation : o.perturbation;
  return Math.max(o.minPerturbation, p0 * Math.pow(o.perturbationDecay, it));
}

/** Normal perturbation (radians) of iteration `it` (scaled like the depth perturbation when `fromPrior`). */
export function normalPerturbation(o: ResolvedPatchMatchOptions, it: number, fromPrior: boolean): number {
  // ratio guarded against perturbation = 0 (was a division by zero ⇒ Infinity ⇒ NaN normals)
  const ratio = o.perturbation > 0 ? Math.min(1, o.geometricPerturbation / o.perturbation) : 1;
  const p0 = fromPrior ? o.normalPerturbation * ratio : o.normalPerturbation;
  return Math.max(0.01, p0 * Math.pow(o.perturbationDecay, it));
}

/** ACMH good-cost threshold at iteration `it`. */
export function goodCostThreshold(o: ResolvedPatchMatchOptions, it: number): number {
  return o.goodCost * Math.exp(-(it * it) / o.goodCostDecay);
}

/**
 * Adaptive band height for GPU submissions: rows ≈ target / (measured ms per row,
 * exponentially averaged), clamped to [minRows, maxRows]. With a non-finite
 * target it always returns `initialRows`.
 */
export class BandController {
  private msPerRow = 0;
  constructor(
    private readonly targetMs: number,
    private readonly initialRows: number,
    private readonly maxRows: number,
    private readonly minRows: number,
  ) {}

  next(): number {
    if (!(this.msPerRow > 0) || !Number.isFinite(this.targetMs)) return this.initialRows;
    return Math.max(this.minRows, Math.min(this.maxRows, Math.floor(this.targetMs / this.msPerRow)));
  }

  update(rows: number, ms: number): void {
    if (!Number.isFinite(this.targetMs) || rows <= 0) return;
    const r = Math.max(ms, 0.05) / rows;
    this.msPerRow = this.msPerRow > 0 ? 0.5 * this.msPerRow + 0.5 * r : r;
  }
}
