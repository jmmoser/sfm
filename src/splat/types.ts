/**
 * 3D Gaussian Splatting (src/splat): shared types, constants and the memory plan.
 *
 * Conventions (ARCHITECTURE §2): OpenCV camera frame (+x right, +y down, +z forward),
 * world-to-camera poses `x_cam = R X + t` (R row-major), pixel centres at integer
 * coordinates, quaternions `[w, x, y, z]`. Colours are sRGB values in [0, 1] (the training
 * images are sRGB; like 3DGS the model is fitted in that space).
 *
 * A Gaussian has a mean (3), log-scales (3), a rotation quaternion (4, unnormalised), an
 * opacity logit (1) and spherical-harmonics colour coefficients of degree 0 or 1
 * (1 or 4 RGB triplets). The rendered colour is `max(0, SH_C0·f0 + SH_C1·(−y f1 + z f2 − x f3) + 0.5)`
 * with (x, y, z) the unit direction from the camera centre to the mean (gsplat / INRIA
 * convention, so exported PLY files render the same in other viewers).
 */
import type { CameraIntrinsics, Pose } from '../core/types';

/** Screen tile edge (pixels); one 256-thread workgroup per tile in the rasterisers. */
export const TILE_SIZE = 16;
/** Real spherical-harmonics constants of bands 0 and 1. */
export const SH_C0 = 0.28209479177387814;
export const SH_C1 = 0.4886025119029199;
/** Splats with alpha below this are skipped (and the screen extent is derived from it). */
export const ALPHA_MIN = 1 / 255;
/** Alpha clamp (keeps 1 − alpha away from 0 in the backward pass). */
export const ALPHA_MAX = 0.99;
/** Compositing stops before transmittance would drop below this. */
export const T_MIN = 1e-4;
/** Isotropic screen-space low-pass filter added to every projected covariance (px²). */
export const BLUR_EPS = 0.3;
/** Projection Jacobian guard: tan-angles are clamped this fraction of the image beyond the frustum. */
export const FRUSTUM_GUARD = 0.3;
/** Default near plane (camera z, world units). */
export const DEFAULT_NEAR = 0.01;

export type ShDegree = 0 | 1;

/** RGB coefficient triplets per Gaussian for an SH degree. */
export function shCoeffCount(deg: ShDegree): 1 | 4 {
  return deg === 0 ? 1 : 4;
}

/**
 * Host-side Gaussian model (tight arrays). `sh` holds `count × K × 3` floats, coefficient-major
 * per Gaussian: `[f0.r, f0.g, f0.b, f1.r, …]` with K = shCoeffCount(shDegree).
 */
export interface SplatModel {
  count: number;
  shDegree: ShDegree;
  means: Float32Array; // 3N
  logScales: Float32Array; // 3N
  quats: Float32Array; // 4N, [w, x, y, z], not necessarily normalised
  opacities: Float32Array; // N, logits
  sh: Float32Array; // N·K·3
}

/** Pinhole camera for rendering (no distortion). */
export interface SplatCamera {
  width: number;
  height: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  /** World-to-camera rotation, row-major 3×3. */
  R: ArrayLike<number>;
  /** World-to-camera translation. */
  t: ArrayLike<number>;
  /** Near plane (camera z); default DEFAULT_NEAR. */
  near?: number;
}

/** Pinhole SplatCamera from undistorted intrinsics (k1 = k2 = 0 expected) and a pose. */
export function splatCamera(cam: CameraIntrinsics, pose: Pose, near = DEFAULT_NEAR): SplatCamera {
  return { width: cam.width, height: cam.height, fx: cam.f, fy: cam.f, cx: cam.cx, cy: cam.cy, R: pose.R, t: pose.t, near };
}

/**
 * The camera of an image downscaled by an integer `factor` (floor size, §2 resampling rule with
 * s = 1 / factor exactly, so pixel (x, y) of the small image is the mean of the factor × factor
 * block starting at (factor·x, factor·y)).
 */
export function downscaleSplatCamera(c: SplatCamera, factor: number): SplatCamera {
  if (factor === 1) return c;
  const s = 1 / factor;
  return {
    ...c,
    width: Math.max(1, Math.floor(c.width / factor)),
    height: Math.max(1, Math.floor(c.height / factor)),
    fx: c.fx * s,
    fy: c.fy * s,
    cx: (c.cx + 0.5) * s - 0.5,
    cy: (c.cy + 0.5) * s - 0.5,
  };
}

/** Camera centre C = −Rᵀ t. */
export function cameraCentre(c: SplatCamera): [number, number, number] {
  const R = c.R;
  const t = c.t;
  return [-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])];
}

/** Empty model with `count` Gaussians (identity rotations, everything else zero). */
export function allocSplatModel(count: number, shDegree: ShDegree): SplatModel {
  const quats = new Float32Array(4 * count);
  for (let i = 0; i < count; i++) quats[4 * i] = 1;
  return {
    count,
    shDegree,
    means: new Float32Array(3 * count),
    logScales: new Float32Array(3 * count),
    quats,
    opacities: new Float32Array(count),
    sh: new Float32Array(count * shCoeffCount(shDegree) * 3),
  };
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function logit(p: number): number {
  return Math.log(p / (1 - p));
}

// ------------------------------------------------------------------ memory plan

/** GPU memory of a training run, bytes per item (see `estimateSplatMemory`). */
export interface SplatMemoryInput {
  /** Hard cap on the number of Gaussians. */
  maxGaussians: number;
  shDegree: ShDegree;
  /** Largest training image (pixels, full resolution of the schedule). */
  width: number;
  height: number;
  /** Number of training views kept on the GPU (RGBA8 each). */
  views: number;
  /** Tile intersections reserved per Gaussian (default 12; grows on demand). */
  intersectionsPerGaussian?: number;
  /**
   * Geometry supervision (trainer `geometry`): views with a depth target and the target
   * downscale (8 B per target pixel); also the 4th projection row, 4 more 2-D gradients per
   * Gaussian and 32 B per pixel of geometry channels.
   */
  geometry?: { views: number; factor: number; normals?: boolean };
}

export interface SplatMemoryEstimate {
  total: number;
  breakdown: Record<string, number>;
  /** Largest single buffer (must stay ≤ maxStorageBufferBindingSize, 256 MiB on iOS 26.0–26.1). */
  largestBuffer: number;
}

/** Floats per Gaussian in the parameter layout (mean 4 + log-scale 4 + quaternion 4 + opacity 1 + SH). */
export function paramFloatsPerGaussian(deg: ShDegree): number {
  return 13 + 3 * shCoeffCount(deg);
}

/**
 * GPU memory plan of a `SplatTrainer` (the buffers it allocates, ±1 %). Parameters + Adam
 * moments (3 × params), 2-D gradient accumulators (9 floats), projection outputs (3 vec4 +
 * rect + tile count + depth sort ping-pong), intersection keys / values (ping-pong) at
 * `intersectionsPerGaussian`, per-pixel buffers (render, contributor count, loss
 * gradient, SSIM partial maps, target) and the training images.
 */
export function estimateSplatMemory(m: SplatMemoryInput): SplatMemoryEstimate {
  const N = m.maxGaussians;
  const P = paramFloatsPerGaussian(m.shDegree) * 4;
  const px = m.width * m.height;
  const isect = Math.ceil(N * (m.intersectionsPerGaussian ?? 12));
  const tiles = Math.ceil(m.width / TILE_SIZE) * Math.ceil(m.height / TILE_SIZE);
  const g = m.geometry;
  const breakdown: Record<string, number> = {
    params: N * P,
    adam: 2 * N * P,
    grad2d: N * (g ? (g.normals === false ? 40 : 52) : 36),
    projection: N * ((g && g.normals !== false ? 64 : 48) + 8 + 4 + 4 * 4 + 4),
    intersections: isect * 16,
    sortScratch: Math.ceil(isect / 1024) * 16 * 4 + 4096 * 4,
    pixels: px * (16 + 4 + 16 + 48 + 4 + (g ? 32 : 0)) + tiles * (g ? 24 : 8),
    images: m.views * px * 4,
    colour: m.views * 36 * 4,
  };
  if (g) breakdown.depthTargets = g.views * Math.floor(m.width / g.factor) * Math.floor(m.height / g.factor) * 8;
  let total = 0;
  for (const k of Object.keys(breakdown)) total += breakdown[k];
  const largestBuffer = Math.max(N * P, isect * 4, px * 48, px * 4);
  return { total, breakdown, largestBuffer };
}
