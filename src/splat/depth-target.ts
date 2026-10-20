/**
 * Depth / normal targets of the trainer's geometry supervision (`SplatTrainerOptions.geometry`):
 * the dense stage's fusion-filtered depth maps resampled to a training view's pinhole camera at
 * 1/factor of its size and packed 8 B per pixel for the GPU (wgsl/loss.ts `GEO_LOSS_WGSL`), and
 * the CPU reference of that loss.
 *
 * Packed pixel (two u32, WGSL `vec2u`): x = f16 inverse depth (low half) | f16 weight (high half)
 * — `unpack2x16float` order —; y = the camera-frame unit normal, octahedral, 2 × snorm16
 * (`unpack2x16snorm`). Weight 0 = no target. Resampling: nearest source pixel along each target
 * pixel's ray (both cameras are undistorted pinholes of the same pose, so z and the camera-frame
 * normal carry over unchanged).
 */
import type { CameraIntrinsics } from '../core/types';
import { float16BitsToFloat32, float32ToFloat16Bits } from '../io/serialize';
import { GEO_MIN_ALPHA } from './wgsl/loss';

/** Depth map as the dense stage stores it (z-depth 0 = invalid, camera-frame normals, cost 0..2). */
export interface DepthMapLike {
  width: number;
  height: number;
  depth: ArrayLike<number>;
  normal: ArrayLike<number>;
  cost: ArrayLike<number>;
}

export interface DepthTarget {
  /** Target size = ⌊view size / factor⌋. */
  width: number;
  height: number;
  factor: number;
  /** 2 u32 per pixel (see the module comment). */
  data: Uint32Array;
  /** Pixels with a target. */
  valid: number;
}

export interface DepthTargetOptions {
  /** Downscale of the target vs the training view (default 2). */
  factor?: number;
  /**
   * Weight of a pixel from its photometric cost (1 − NCC, 0..2): default
   * clamp((costMax − cost) / costMax, 0, 1) with `costMax`.
   */
  weight?: (cost: number) => number;
  /** Cost at which the default weight reaches 0 (default 1). */
  costMax?: number;
  /**
   * Viewing-angle weighting: the weight is multiplied by |cos θ|^grazing, θ = angle between the
   * target normal and the pixel's ray (surfaces seen at grazing angles count less). Default 0 (off).
   */
  grazing?: number;
}

/** Octahedral encoding of a unit vector → [−1, 1]². */
export function octEncode(x: number, y: number, z: number): [number, number] {
  const l1 = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1;
  let u = x / l1;
  let v = y / l1;
  if (z < 0) {
    const ou = (1 - Math.abs(v)) * (u >= 0 ? 1 : -1);
    const ov = (1 - Math.abs(u)) * (v >= 0 ? 1 : -1);
    u = ou;
    v = ov;
  }
  return [u, v];
}

/** Inverse of `octEncode` (normalised), as the WGSL `octDecode`. */
export function octDecode(u: number, v: number): [number, number, number] {
  let x = u;
  let y = v;
  const z = 1 - Math.abs(u) - Math.abs(v);
  if (z < 0) {
    x = (1 - Math.abs(v)) * (u >= 0 ? 1 : -1);
    y = (1 - Math.abs(u)) * (v >= 0 ? 1 : -1);
  }
  const l = Math.hypot(x, y, z) || 1;
  return [x / l, y / l, z / l];
}

const snorm16 = (v: number): number => Math.round(Math.max(-1, Math.min(1, v)) * 32767) & 0xffff;
const unsnorm16 = (b: number): number => Math.max(-1, ((b << 16) >> 16) / 32767);

/** Resamples and packs a depth map for a view camera (`mapCam` = the depth map's pinhole camera). */
export function packDepthTarget(map: DepthMapLike, mapCam: Pick<CameraIntrinsics, 'f' | 'cx' | 'cy'>, viewCam: Pick<CameraIntrinsics, 'width' | 'height' | 'f' | 'cx' | 'cy'>, o: DepthTargetOptions = {}): DepthTarget {
  const factor = Math.max(1, Math.round(o.factor ?? 2));
  const costMax = o.costMax ?? 1;
  const weightOf = o.weight ?? ((c: number) => Math.max(0, Math.min(1, (costMax - c) / costMax)));
  const grazing = Math.max(0, o.grazing ?? 0);
  const W = Math.max(1, Math.floor(viewCam.width / factor));
  const H = Math.max(1, Math.floor(viewCam.height / factor));
  // target pixel (i, j) = full-resolution block [factor·i, factor·i + factor): its centre
  const s = 1 / factor;
  const f = viewCam.f * s;
  const cx = (viewCam.cx + 0.5) * s - 0.5;
  const cy = (viewCam.cy + 0.5) * s - 0.5;
  const data = new Uint32Array(2 * W * H);
  let valid = 0;
  for (let j = 0; j < H; j++) {
    const v = Math.round((mapCam.f * (j - cy)) / f + mapCam.cy);
    if (v < 0 || v >= map.height) continue;
    for (let i = 0; i < W; i++) {
      const u = Math.round((mapCam.f * (i - cx)) / f + mapCam.cx);
      if (u < 0 || u >= map.width) continue;
      const p = v * map.width + u;
      const z = map.depth[p];
      if (!(z > 0) || !Number.isFinite(z)) continue;
      let w = weightOf(map.cost[p]);
      if (!(w > 0)) continue;
      const nx = map.normal[3 * p];
      const ny = map.normal[3 * p + 1];
      const nz = map.normal[3 * p + 2];
      if (!(nx * nx + ny * ny + nz * nz > 0.25)) continue;
      if (grazing > 0) {
        const rx = (i - cx) / f;
        const ry = (j - cy) / f;
        const cos = Math.abs(nx * rx + ny * ry + nz) / (Math.hypot(rx, ry, 1) * Math.hypot(nx, ny, nz));
        w *= Math.pow(cos, grazing);
        if (!(w >= 1e-3)) continue;
      }
      const [ou, ov] = octEncode(nx, ny, nz);
      const t = j * W + i;
      data[2 * t] = (float32ToFloat16Bits(1 / z) | (float32ToFloat16Bits(Math.min(1, w)) << 16)) >>> 0;
      data[2 * t + 1] = (snorm16(ou) | (snorm16(ov) << 16)) >>> 0;
      valid++;
    }
  }
  return { width: W, height: H, factor, data, valid };
}

/** Decoded target pixel t: inverse depth, weight, normal (zero weight = none). */
export function unpackTargetPixel(d: Uint32Array, t: number): { invZ: number; w: number; n: [number, number, number] } {
  const x = d[2 * t];
  const y = d[2 * t + 1];
  return { invZ: float16BitsToFloat32(x & 0xffff), w: float16BitsToFloat32(x >>> 16), n: octDecode(unsnorm16(y & 0xffff), unsnorm16(y >>> 16)) };
}

export interface GeometryLossResult {
  /** Σ w·|D̂·z_t − 1|, Σ w·‖N̂ − n_t‖₁, pixels with a target, Σ w (the kernel's partial sums). */
  depth: number;
  normal: number;
  valid: number;
  weight: number;
  /** ∂L/∂(D, N) per pixel (4 floats). */
  grad: Float64Array;
  /** ∂L/∂A per pixel (accumulated alpha 1 − T). */
  gradA: Float64Array;
}

/**
 * CPU reference of the geometry loss kernel for a render `geo` (4 floats per pixel: D = Σ w/z,
 * N = Σ w·n) with accumulated alpha `alpha` (A = 1 − T per pixel) at level `factor` (W × H):
 * L = (1 / (W·H)) Σ w (λ_d |D̂ z_t − 1| + λ_n ‖N̂ − n_t‖₁), D̂ = D / A, N̂ = N / A, over the pixels
 * with a target and A ≥ GEO_MIN_ALPHA.
 */
export function geometryLossCPU(geo: ArrayLike<number>, alpha: ArrayLike<number>, W: number, H: number, factor: number, target: DepthTarget, wDepth: number, wNormal: number): GeometryLossResult {
  const grad = new Float64Array(4 * W * H);
  const gradA = new Float64Array(W * H);
  const invPx = 1 / (W * H);
  const r = { depth: 0, normal: 0, valid: 0, weight: 0, grad, gradA };
  const sgn = (x: number): number => (x > 0 ? 1 : x < 0 ? -1 : 0);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const j = Math.floor((factor * x + Math.floor(factor / 2)) / target.factor);
      const k = Math.floor((factor * y + Math.floor(factor / 2)) / target.factor);
      if (j >= target.width || k >= target.height) continue;
      const o = y * W + x;
      const A = alpha[o];
      if (!(A >= GEO_MIN_ALPHA)) continue;
      const t = unpackTargetPixel(target.data, k * target.width + j);
      if (!(t.invZ > 0 && t.w > 0)) continue;
      const zt = 1 / t.invZ;
      const D = geo[4 * o] / A;
      const e = D * zt - 1;
      const c = invPx * t.w;
      const gD = wDepth * c * sgn(e) * zt;
      grad[4 * o] = gD / A;
      let gA = -gD * D;
      let nl = 0;
      for (let q = 0; q < 3; q++) {
        const N = geo[4 * o + 1 + q] / A;
        const dn = N - t.n[q];
        const gN = wNormal * c * sgn(dn);
        grad[4 * o + 1 + q] = gN / A;
        gA -= gN * N;
        nl += Math.abs(dn);
      }
      gradA[o] = gA / A;
      r.depth += t.w * Math.abs(e);
      r.normal += t.w * nl;
      r.valid++;
      r.weight += t.w;
    }
  }
  return r;
}
