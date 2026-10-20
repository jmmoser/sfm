/**
 * Pure (DOM-free, three-free) helpers for the viewer: robust bounds, up-vector
 * estimation, formatting, depth (un)packing and a CPU nearest-point picker.
 *
 * Everything here works in the WORLD frame of the reconstruction (Z-up when
 * gravity aligned, see docs/ARCHITECTURE.md §2) and is unit-tested in Node.
 */
import type { Mat3, Pose, Vec3 } from '../core/types';

/** Axis-aligned box. */
export interface Bounds3 {
  min: Float64Array; // 3
  max: Float64Array; // 3
}

/** Percentile box plus the full (min/max) box of the sampled points. */
export interface RobustBounds extends Bounds3 {
  /** Centre of the robust box. */
  center: Float64Array;
  /** Half the diagonal of the robust box (≥ a tiny epsilon). */
  radius: number;
  /** Full extent of the sampled points (outliers included). */
  full: Bounds3;
  /** Number of points the bounds describe (not the number of samples). */
  count: number;
}

export interface RobustBoundsOptions {
  /** Lower percentile (0..100), default 2. */
  lowerPercentile?: number;
  /** Upper percentile (0..100), default 98. */
  upperPercentile?: number;
  /** Maximum number of points sampled (deterministic stride), default 65536. */
  maxSamples?: number;
}

const MIN_RADIUS = 1e-9;

/**
 * Deterministic stride subsample of a packed xyz array. Returns a fresh
 * Float32Array of at most `maxSamples` points (never aliases the input).
 * Non-finite points are skipped.
 */
export function subsamplePositions(
  positions: ArrayLike<number>,
  count: number,
  maxSamples: number,
): Float32Array {
  const n = Math.max(0, Math.min(count, Math.floor(positions.length / 3)));
  if (n === 0 || maxSamples <= 0) return new Float32Array(0);
  const stride = n <= maxSamples ? 1 : n / maxSamples;
  const m = Math.min(n, maxSamples);
  const out = new Float32Array(3 * m);
  let k = 0;
  for (let s = 0; s < m; s++) {
    const i = stride === 1 ? s : Math.min(n - 1, Math.floor(s * stride));
    const x = positions[3 * i];
    const y = positions[3 * i + 1];
    const z = positions[3 * i + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    out[3 * k] = x;
    out[3 * k + 1] = y;
    out[3 * k + 2] = z;
    k++;
  }
  return k === m ? out : out.slice(0, 3 * k);
}

/** Percentile (0..100) of a SORTED array with linear interpolation. */
function sortedPercentile(sorted: Float32Array | Float64Array, p: number): number {
  const n = sorted.length;
  if (n === 0) return NaN;
  if (n === 1) return sorted[0];
  const pos = (Math.min(100, Math.max(0, p)) / 100) * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(n - 1, lo + 1);
  const f = pos - lo;
  return sorted[lo] * (1 - f) + sorted[hi] * f;
}

/**
 * Robust axis-aligned bounds: per-axis [lower, upper] percentiles (default
 * 2–98 %) over a deterministic subsample. Ignores far outliers (typical for
 * SfM / MVS clouds) when framing the view. Returns null for an empty input.
 */
export function robustBounds(
  positions: ArrayLike<number>,
  count?: number,
  opts: RobustBoundsOptions = {},
): RobustBounds | null {
  const n = count ?? Math.floor(positions.length / 3);
  const lo = opts.lowerPercentile ?? 2;
  const hi = opts.upperPercentile ?? 98;
  const samples = subsamplePositions(positions, n, opts.maxSamples ?? 65536);
  return robustBoundsOfSamples(samples, n, lo, hi);
}

/** Same as robustBounds() but on an already-subsampled xyz array. */
export function robustBoundsOfSamples(
  samples: Float32Array,
  count: number,
  lowerPercentile = 2,
  upperPercentile = 98,
): RobustBounds | null {
  const m = Math.floor(samples.length / 3);
  if (m === 0) return null;
  const axis = new Float32Array(m);
  const min = new Float64Array(3);
  const max = new Float64Array(3);
  const fmin = new Float64Array(3);
  const fmax = new Float64Array(3);
  for (let a = 0; a < 3; a++) {
    for (let i = 0; i < m; i++) axis[i] = samples[3 * i + a];
    axis.sort();
    fmin[a] = axis[0];
    fmax[a] = axis[m - 1];
    min[a] = sortedPercentile(axis, lowerPercentile);
    max[a] = sortedPercentile(axis, upperPercentile);
  }
  const center = new Float64Array([(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2]);
  const dx = max[0] - min[0];
  const dy = max[1] - min[1];
  const dz = max[2] - min[2];
  const radius = Math.max(MIN_RADIUS, 0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz));
  return { min, max, center, radius, full: { min: fmin, max: fmax }, count };
}

/** Bounds of a list of points given as Vec3-like arrays (no percentiles). */
export function boundsOfPoints(points: ArrayLike<ArrayLike<number>>): Bounds3 | null {
  if (points.length === 0) return null;
  const min = new Float64Array([Infinity, Infinity, Infinity]);
  const max = new Float64Array([-Infinity, -Infinity, -Infinity]);
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    for (let a = 0; a < 3; a++) {
      if (p[a] < min[a]) min[a] = p[a];
      if (p[a] > max[a]) max[a] = p[a];
    }
  }
  return { min, max };
}

/** Union of two boxes (either may be null). */
export function unionBounds(a: Bounds3 | null, b: Bounds3 | null): Bounds3 | null {
  if (!a) return b ? { min: b.min.slice(), max: b.max.slice() } : null;
  if (!b) return { min: a.min.slice(), max: a.max.slice() };
  const min = new Float64Array(3);
  const max = new Float64Array(3);
  for (let i = 0; i < 3; i++) {
    min[i] = Math.min(a.min[i], b.min[i]);
    max[i] = Math.max(a.max[i], b.max[i]);
  }
  return { min, max };
}

/** Bounding sphere (centre + half diagonal) of a box. */
export function boundsSphere(b: Bounds3): { center: Float64Array; radius: number } {
  const center = new Float64Array(3);
  let d2 = 0;
  for (let i = 0; i < 3; i++) {
    center[i] = 0.5 * (b.min[i] + b.max[i]);
    const d = b.max[i] - b.min[i];
    d2 += d * d;
  }
  return { center, radius: Math.max(MIN_RADIUS, 0.5 * Math.sqrt(d2)) };
}

/**
 * The full box clamped to the robust box grown by `factor` around its centre.
 * Used for near/far planes: keeps (almost) everything visible without letting a
 * single outlier at 1e6 destroy depth precision.
 */
export function clampedFullBounds(rb: RobustBounds, factor = 6): Bounds3 {
  const min = new Float64Array(3);
  const max = new Float64Array(3);
  for (let a = 0; a < 3; a++) {
    const half = Math.max(0.5 * (rb.max[a] - rb.min[a]), rb.radius * 0.25);
    min[a] = Math.max(rb.full.min[a], rb.center[a] - factor * half);
    max[a] = Math.min(rb.full.max[a], rb.center[a] + factor * half);
  }
  return { min, max };
}

/**
 * p-th percentile (0..100) of the projection of the sampled points onto `dir`
 * (e.g. height along "up" for the ground grid).
 */
export function percentileAlong(samples: Float32Array, dir: ArrayLike<number>, p: number): number {
  const m = Math.floor(samples.length / 3);
  if (m === 0) return NaN;
  const proj = new Float32Array(m);
  const dx = dir[0];
  const dy = dir[1];
  const dz = dir[2];
  for (let i = 0; i < m; i++) proj[i] = samples[3 * i] * dx + samples[3 * i + 1] * dy + samples[3 * i + 2] * dz;
  proj.sort();
  return sortedPercentile(proj, p);
}

/** Camera centre C = -Rᵀ t (world frame). */
export function cameraCenter(pose: Pose, out: Float64Array = new Float64Array(3)): Vec3 {
  const R = pose.R;
  const t = pose.t;
  out[0] = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]);
  out[1] = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]);
  out[2] = -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);
  return out;
}

/** Camera "up" (−y camera axis) in the world frame: −Rᵀ e_y = −(row 1 of R). */
export function cameraUp(R: Mat3, out: Float64Array = new Float64Array(3)): Vec3 {
  out[0] = -R[3];
  out[1] = -R[4];
  out[2] = -R[5];
  return out;
}

/**
 * Jacobi eigen-decomposition of a symmetric 3×3 (row-major). Returns eigenvalues
 * ascending and eigenvectors as columns of a row-major 3×3.
 */
export function symEig3(A: ArrayLike<number>): { values: Float64Array; vectors: Float64Array } {
  const a = Float64Array.from(A as ArrayLike<number>);
  const v = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
    if (off < 1e-30) break;
    for (let p = 0; p < 2; p++) {
      for (let q = p + 1; q < 3; q++) {
        const apq = a[3 * p + q];
        if (Math.abs(apq) < 1e-300) continue;
        const app = a[3 * p + p];
        const aqq = a[3 * q + q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        // A' = Jᵀ A J with rotation in the (p, q) plane
        for (let k = 0; k < 3; k++) {
          const akp = a[3 * k + p];
          const akq = a[3 * k + q];
          a[3 * k + p] = c * akp - s * akq;
          a[3 * k + q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = a[3 * p + k];
          const aqk = a[3 * q + k];
          a[3 * p + k] = c * apk - s * aqk;
          a[3 * q + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = v[3 * k + p];
          const vkq = v[3 * k + q];
          v[3 * k + p] = c * vkp - s * vkq;
          v[3 * k + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[4 * i] - a[4 * j]);
  const values = new Float64Array(3);
  const vectors = new Float64Array(9);
  for (let c = 0; c < 3; c++) {
    const src = order[c];
    values[c] = a[4 * src];
    for (let r = 0; r < 3; r++) vectors[3 * r + c] = v[3 * r + src];
  }
  return { values, vectors };
}

function normalize3(v: Float64Array): boolean {
  const n = Math.hypot(v[0], v[1], v[2]);
  if (!(n > 1e-12)) return false;
  v[0] /= n;
  v[1] /= n;
  v[2] /= n;
  return true;
}

/**
 * Estimate the world "up" direction from camera orientations (for models that
 * are not gravity aligned).
 *
 * Camera up: `upCam` (unit, camera frame: −gravity, or up for the UI rotation the photo
 * was taken in) when given, else the −y axis (image up, which is not up when the images
 * are upright for another rotation, e.g. raw sensor buffers in a portrait UI); camera
 * right: e_z × up (the x axis for −y).
 *
 * Base estimate: the mean of the camera up vectors in the world (Rᵀ up).
 * People hold the phone level, so the camera right axes (in the world) are
 * nearly horizontal: when they span two directions (the user turned around /
 * orbited), up is the direction most orthogonal to all of them (smallest
 * eigenvector of Σ w x xᵀ, IRLS with Cauchy weights to ignore rolled frames),
 * which — unlike the mean up vector — is unbiased by the camera pitch on partial
 * orbits. If the right axes are (nearly) parallel (no yaw change: the tilt about that
 * axis is unobservable), the mean up vector made orthogonal to them is used; if
 * they are not near-horizontal at all (phone rolled arbitrarily), the mean up
 * vector. The sign always agrees with the mean up vector. Null if no cameras.
 */
export function estimateUpVector(rotations: ArrayLike<{ R: Mat3; upCam?: ArrayLike<number> | null }>): Vec3 | null {
  const n = rotations.length;
  if (n === 0) return null;
  // Per camera in the world: up (Rᵀ u) and right (Rᵀ (e_z × u)); without upCam −R row 1 and R row 0.
  const ups = new Float64Array(3 * n);
  const rights = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    const { R, upCam: u } = rotations[i];
    if (!u) {
      ups.set([-R[3], -R[4], -R[5]], 3 * i);
      rights.set([R[0], R[1], R[2]], 3 * i);
      continue;
    }
    let rx = -u[1];
    let ry = u[0];
    const rn = Math.hypot(rx, ry);
    // Looking along up: every image axis is horizontal.
    if (rn > 1e-6) {
      rx /= rn;
      ry /= rn;
    } else {
      rx = 1;
      ry = 0;
    }
    for (let k = 0; k < 3; k++) {
      ups[3 * i + k] = R[k] * u[0] + R[3 + k] * u[1] + R[6 + k] * u[2];
      rights[3 * i + k] = R[k] * rx + R[3 + k] * ry;
    }
  }
  const meanUp = new Float64Array(3);
  for (let i = 0; i < n; i++) {
    meanUp[0] += ups[3 * i];
    meanUp[1] += ups[3 * i + 1];
    meanUp[2] += ups[3 * i + 2];
  }
  if (!normalize3(meanUp)) {
    // Up vectors cancel (pathological). Fall back to the first camera.
    meanUp.set(ups.subarray(0, 3));
    normalize3(meanUp);
  }
  if (n < 2) return meanUp;

  const weights = new Float64Array(n).fill(1);
  let up: Float64Array = meanUp.slice();
  const M = new Float64Array(9);
  for (let iter = 0; iter < 4; iter++) {
    M.fill(0);
    let wsum = 0;
    for (let i = 0; i < n; i++) {
      const w = weights[i];
      const x0 = rights[3 * i];
      const x1 = rights[3 * i + 1];
      const x2 = rights[3 * i + 2];
      M[0] += w * x0 * x0;
      M[1] += w * x0 * x1;
      M[2] += w * x0 * x2;
      M[4] += w * x1 * x1;
      M[5] += w * x1 * x2;
      M[8] += w * x2 * x2;
      wsum += w;
    }
    M[3] = M[1];
    M[6] = M[2];
    M[7] = M[5];
    for (let k = 0; k < 9; k++) M[k] /= wsum;
    const { values, vectors } = symEig3(M);
    const l0 = values[0];
    const l1 = values[1];
    const e0 = new Float64Array([vectors[0], vectors[3], vectors[6]]);
    const e2 = new Float64Array([vectors[2], vectors[5], vectors[8]]);
    let cand: Float64Array;
    if (l1 > 0.01 && l0 < 0.3 * l1) {
      // Right axes span a plane: up is its normal.
      cand = e0;
    } else if (l1 <= 0.01) {
      // Right axes (nearly) parallel: remove that direction from the mean up.
      const d = meanUp[0] * e2[0] + meanUp[1] * e2[1] + meanUp[2] * e2[2];
      cand = new Float64Array([meanUp[0] - d * e2[0], meanUp[1] - d * e2[1], meanUp[2] - d * e2[2]]);
      if (!normalize3(cand)) cand = meanUp.slice();
    } else {
      // Phone not held level (random roll): right axes say nothing about up.
      cand = meanUp.slice();
    }
    if (cand[0] * meanUp[0] + cand[1] * meanUp[1] + cand[2] * meanUp[2] < 0) {
      cand[0] = -cand[0];
      cand[1] = -cand[1];
      cand[2] = -cand[2];
    }
    up = cand;
    // Cauchy re-weighting on |x · up| (scale ≈ 0.1 rad of roll/tilt).
    for (let i = 0; i < n; i++) {
      const r = (rights[3 * i] * up[0] + rights[3 * i + 1] * up[1] + rights[3 * i + 2] * up[2]) / 0.1;
      weights[i] = 1 / (1 + r * r);
    }
  }
  return up;
}

/** "Nice" step (1, 2 or 5 × 10^k) closest to but ≥ x/1.5 — for grid cells. */
export function niceStep(x: number): number {
  if (!(x > 0) || !Number.isFinite(x)) return 1;
  const e = Math.floor(Math.log10(x));
  const base = Math.pow(10, e);
  const f = x / base;
  const m = f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10;
  return m * base;
}

/**
 * Human-readable distance. Metric: metres (≥ 1 m) or centimetres; otherwise
 * three significant digits followed by "units".
 */
export function formatDistance(d: number, metric: boolean): string {
  if (!Number.isFinite(d)) return '–';
  if (metric) {
    if (d >= 1) return `${d.toFixed(d >= 100 ? 1 : 2)} m`;
    const cm = d * 100;
    return `${cm.toFixed(cm >= 10 ? 1 : 2)} cm`;
  }
  const a = Math.abs(d);
  let s: string;
  if (a === 0) s = '0';
  else if (a >= 1e5 || a < 1e-3) s = d.toExponential(2);
  else s = d.toPrecision(3);
  return `${s} units`;
}

/**
 * Inverse of three.js `packDepthToRGBA` (packing.glsl) for bytes read back
 * from an RGBA8 target. Returns a value in [0, 1].
 */
export function unpackRGBAToDepth(r: number, g: number, b: number, a: number): number {
  const down = 255 / 256;
  return (
    (r / 255) * down +
    (g / 255) * (down / 256) +
    (b / 255) * (down / 65536) +
    (a / 255) * (1 / 16777216)
  );
}

/** Window depth d ∈ [0,1] (perspective, glDepthRange [0,1]) → linear view depth. */
export function linearizeDepth(d: number, near: number, far: number): number {
  const z = 2 * d - 1;
  return (2 * near * far) / (far + near - z * (far - near));
}

/** Fitted inverse-depth plane 1/z = a + b·u + c·v over screen offsets (u, v). */
export interface InverseDepthPlane {
  a: number;
  b: number;
  c: number;
}

/**
 * Least-squares fit of 1/z = a + b·u + c·v (exact for a planar surface under
 * perspective projection) from samples of view depth z at screen offsets (u, v).
 * Collinear samples: slope along their direction only. < 3 samples (or a
 * degenerate fit): constant inverse-distance-weighted mean (b = c = 0).
 * Null when n = 0.
 */
export function fitInverseDepthPlane(
  u: ArrayLike<number>,
  v: ArrayLike<number>,
  z: ArrayLike<number>,
  n: number,
): InverseDepthPlane | null {
  if (n <= 0) return null;
  let su = 0;
  let sv = 0;
  let suu = 0;
  let suv = 0;
  let svv = 0;
  let r1 = 0;
  let ru = 0;
  let rv = 0;
  let wm = 0;
  let wmy = 0;
  for (let i = 0; i < n; i++) {
    const y = 1 / z[i];
    const a = u[i];
    const b = v[i];
    su += a;
    sv += b;
    suu += a * a;
    suv += a * b;
    svv += b * b;
    r1 += y;
    ru += a * y;
    rv += b * y;
    const w = 1 / (1 + a * a + b * b);
    wm += w;
    wmy += w * y;
  }
  const constant: InverseDepthPlane = { a: wmy / wm, b: 0, c: 0 };
  if (n < 3) return constant;
  // Centred moments for conditioning.
  const ub = su / n;
  const vb = sv / n;
  const yb = r1 / n;
  const cuu = suu / n - ub * ub;
  const cuv = suv / n - ub * vb;
  const cvv = svv / n - vb * vb;
  const cuy = ru / n - ub * yb;
  const cvy = rv / n - vb * yb;
  const det = cuu * cvv - cuv * cuv;
  const tr = cuu + cvv;
  if (tr <= 1e-12) return constant;
  let b: number;
  let c: number;
  if (det > 1e-6 * tr * tr) {
    b = (cuy * cvv - cvy * cuv) / det;
    c = (cvy * cuu - cuy * cuv) / det;
  } else {
    // Collinear samples: direction of the (rank-1) covariance, slope along it.
    let ex = cuu >= cvv ? cuu : cuv;
    let ey = cuu >= cvv ? cuv : cvv;
    const en = Math.hypot(ex, ey);
    ex /= en;
    ey /= en;
    const k = (cuy * ex + cvy * ey) / tr;
    b = k * ex;
    c = k * ey;
  }
  const a = yb - b * ub - c * vb;
  if (!(a > 0) || !Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) return constant;
  return { a, b, c };
}

/** Surface view depth on the ray through screen offset (0, 0); NaN when n = 0. See fitInverseDepthPlane. */
export function surfaceDepthAt(u: ArrayLike<number>, v: ArrayLike<number>, z: ArrayLike<number>, n: number): number {
  const p = fitInverseDepthPlane(u, v, z, n);
  return p ? 1 / p.a : NaN;
}

let _cand = new Float64Array(5 * 4096);

/** How points are sized on screen, for the CPU picker's coverage test. */
export interface PickSplatModel {
  /** Fixed point diameter in the same pixel units as the viewport (default 1). */
  sizePx?: number;
  /** Per-point world radius; diameter = clamp(2·r·projScale / w, minPx, maxPx). */
  radii?: ArrayLike<number>;
  /** viewportHeight / (2·tan(fovY/2)), same pixel units. */
  projScale?: number;
  minPx?: number;
  maxPx?: number;
  /**
   * Per-point confidence; points with confidence < `minConfidence` are skipped
   * (they are not drawn either, see PointCloudMaterial).
   */
  confidence?: ArrayLike<number>;
  minConfidence?: number;
}

/**
 * CPU picking fallback: index of the point under a screen position, or -1.
 *
 * Projects every point with the column-major 4×4 matrix `mvp` (three.js
 * `Matrix4.elements` of projection · view · model) and collects the points whose
 * projection lies within `radiusPx` of (px, py) (continuous coordinates, origin
 * at the top-left corner of a `width × height` viewport, like clientX/Y). Then it emulates the depth buffer: among the
 * points whose splat (see `splat`) covers the tapped pixel, those of the FRONT
 * surface (what the depth buffer shows) are kept and the one closest to the tap
 * is returned. If no splat covers it (tap in a gap), the point nearest on screen
 * wins (ties → front). `out` (optional) receives whether a splat covered the
 * pixel and the surface depth on the tap ray (planar fit of 1/w over the front
 * surface points around the tap, see surfaceDepthAt; the nearest point's w when
 * not covered) so callers can place the hit on the ray, like the GPU path.
 * One O(n) pass of ~20 flops per point (≈ 10–30 ms per 2 M points on one core)
 * plus a pass over the few candidates; allocation-free in steady state.
 */
export function pickNearestProjected(
  positions: ArrayLike<number>,
  count: number,
  mvp: ArrayLike<number>,
  width: number,
  height: number,
  px: number,
  py: number,
  radiusPx: number,
  splat: PickSplatModel = {},
  out?: { covered: boolean; w: number },
): number {
  const m0 = mvp[0];
  const m1 = mvp[1];
  const m3 = mvp[3];
  const m4 = mvp[4];
  const m5 = mvp[5];
  const m7 = mvp[7];
  const m8 = mvp[8];
  const m9 = mvp[9];
  const m11 = mvp[11];
  const m12 = mvp[12];
  const m13 = mvp[13];
  const m15 = mvp[15];
  // NDC target and tolerance (anisotropic in NDC when width != height).
  const tx = (px / width) * 2 - 1;
  const ty = 1 - (py / height) * 2;
  const sx = width / 2;
  const sy = height / 2;
  const r2 = radiusPx * radiusPx;
  let nc = 0;
  const n = Math.min(count, Math.floor(positions.length / 3));
  const conf = splat.confidence && (splat.minConfidence ?? 0) > 0 ? splat.confidence : null;
  const minConf = splat.minConfidence ?? 0;
  for (let i = 0, j = 0; i < n; i++, j += 3) {
    const x = positions[j];
    const y = positions[j + 1];
    const z = positions[j + 2];
    const w = m3 * x + m7 * y + m11 * z + m15;
    if (!(w > 0)) continue;
    // Division-free test: |c/w − t|·s ≤ r  ⇔  |c − t·w|·s ≤ r·w (x first: most points fail it)
    const rw2 = r2 * w * w;
    const dx = (m0 * x + m4 * y + m8 * z + m12 - tx * w) * sx;
    const dx2 = dx * dx;
    if (dx2 > rw2) continue;
    const dy = (m1 * x + m5 * y + m9 * z + m13 - ty * w) * sy;
    const dw2 = dx2 + dy * dy;
    if (dw2 > rw2) continue;
    if (conf && conf[i] < minConf) continue; // hidden by the confidence filter (same test as the shader)
    if (5 * nc + 5 > _cand.length) {
      const grown = new Float64Array(_cand.length * 2);
      grown.set(_cand);
      _cand = grown;
    }
    const o = 5 * nc;
    _cand[o] = i;
    _cand[o + 1] = dx / w;
    _cand[o + 2] = dy / w;
    _cand[o + 3] = dw2 / (w * w);
    _cand[o + 4] = w;
    nc++;
  }
  const fixedHalf = 0.5 * Math.max(1, splat.sizePx ?? 1);
  const radii = splat.radii;
  const proj = splat.projScale ?? 0;
  const minPx = splat.minPx ?? 1;
  const maxPx = splat.maxPx ?? 64;
  // Pass 1: front-most covering splat (what the depth buffer shows) and nearest point.
  let coverW = Infinity;
  let near = -1;
  let nearD2 = Infinity;
  let nearW = Infinity;
  let ncov = 0;
  for (let k = 0; k < nc; k++) {
    const o = 5 * k;
    const i = _cand[o];
    const d2 = _cand[o + 3];
    const w = _cand[o + 4];
    let half = fixedHalf;
    if (radii && proj > 0) half = 0.5 * Math.min(maxPx, Math.max(minPx, (2 * radii[i] * proj) / w));
    const h = Math.max(0.5, half);
    if (d2 <= h * h) {
      if (w < coverW) coverW = w;
      _cand[o] = -1 - i; // mark as covering
      ncov++;
    }
    if (d2 < nearD2 - 1 || (Math.abs(d2 - nearD2) <= 1 && w < nearW)) {
      near = i;
      nearD2 = d2;
      nearW = w;
    }
  }
  if (ncov === 0) {
    if (out) {
      out.covered = false;
      out.w = nearW;
    }
    return near;
  }
  // Pass 2: the front surface. Seed = covering splats within 3 % of the
  // front-most depth; fit 1/w = a + b·dx + c·dy (planar patch) and re-select all
  // candidates within ±2 % of that patch (catches the far side of oblique
  // surfaces, excludes occluded ones); the depth at the tap is 1/a. The returned
  // index is the front covering point closest to the tap.
  if (_su.length < nc) {
    _su = new Float64Array(nc * 2);
    _sv = new Float64Array(nc * 2);
    _sz = new Float64Array(nc * 2);
  }
  let m = 0;
  let cover = -1;
  let coverD2 = Infinity;
  const wMax = coverW * 1.03;
  for (let k = 0; k < nc; k++) {
    const o = 5 * k;
    if (_cand[o] >= 0) continue;
    const w = _cand[o + 4];
    if (w > wMax) continue;
    _su[m] = _cand[o + 1];
    _sv[m] = _cand[o + 2];
    _sz[m] = w;
    m++;
    if (_cand[o + 3] < coverD2) {
      coverD2 = _cand[o + 3];
      cover = -1 - _cand[o];
    }
  }
  let wTap = surfaceDepthAt(_su, _sv, _sz, m);
  const plane = m >= 3 ? fitInverseDepthPlane(_su, _sv, _sz, m) : null;
  if (plane) {
    // Re-select every candidate consistent with the seed patch (±2 % in 1/w).
    let m2 = 0;
    for (let k = 0; k < nc; k++) {
      const o = 5 * k;
      const du = _cand[o + 1];
      const dv = _cand[o + 2];
      const pred = plane.a + plane.b * du + plane.c * dv;
      const yi = 1 / _cand[o + 4];
      if (!(pred > 0) || Math.abs(yi - pred) > 0.02 * pred) continue;
      _su[m2] = du;
      _sv[m2] = dv;
      _sz[m2] = _cand[o + 4];
      m2++;
    }
    if (m2 >= 3) {
      const w2 = surfaceDepthAt(_su, _sv, _sz, m2);
      if (Number.isFinite(w2)) wTap = w2;
    }
  }
  if (out) {
    out.covered = true;
    out.w = wTap;
  }
  return cover;
}

/**
 * Closest ray–triangle intersection (Möller–Trumbore, double-sided, float64).
 * Ray `o + t·d` in the same frame as `positions`; returns the smallest t > tMin
 * (in units of |d|), or Infinity when nothing is hit. Indices out of range are
 * skipped. One pass of ~40 flops per triangle (≈ 10–20 ms per 1 M triangles on
 * one core) — used by the CPU picking fallback for meshes.
 */
export function raycastTriangles(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  triangleCount: number,
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  dz: number,
  tMin = 0,
): number {
  const nt = Math.min(triangleCount, Math.floor(indices.length / 3));
  const nv = Math.floor(positions.length / 3);
  let best = Infinity;
  for (let k = 0; k < nt; k++) {
    const i0 = indices[3 * k];
    const i1 = indices[3 * k + 1];
    const i2 = indices[3 * k + 2];
    if (i0 >= nv || i1 >= nv || i2 >= nv) continue;
    const ax = positions[3 * i0];
    const ay = positions[3 * i0 + 1];
    const az = positions[3 * i0 + 2];
    const e1x = positions[3 * i1] - ax;
    const e1y = positions[3 * i1 + 1] - ay;
    const e1z = positions[3 * i1 + 2] - az;
    const e2x = positions[3 * i2] - ax;
    const e2y = positions[3 * i2 + 1] - ay;
    const e2z = positions[3 * i2 + 2] - az;
    // p = d × e2, det = e1 · p
    const px = dy * e2z - dz * e2y;
    const py = dz * e2x - dx * e2z;
    const pz = dx * e2y - dy * e2x;
    const det = e1x * px + e1y * py + e1z * pz;
    if (det === 0 || !Number.isFinite(det)) continue;
    const inv = 1 / det;
    const sx = ox - ax;
    const sy = oy - ay;
    const sz = oz - az;
    const u = (sx * px + sy * py + sz * pz) * inv;
    if (u < 0 || u > 1) continue;
    // q = s × e1
    const qx = sy * e1z - sz * e1y;
    const qy = sz * e1x - sx * e1z;
    const qz = sx * e1y - sy * e1x;
    const v = (dx * qx + dy * qy + dz * qz) * inv;
    if (v < 0 || u + v > 1) continue;
    const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
    if (t > tMin && t < best) best = t;
  }
  return best;
}

let _su = new Float64Array(256);
let _sv = new Float64Array(256);
let _sz = new Float64Array(256);

/**
 * First `k` entries of the golden-ratio stride permutation of [0, n): i ↦ (i·P)
 * mod n with P ≈ 0.618 n coprime to n. Any prefix is an evenly spread subset of
 * the index range (used for the interactive level of detail of huge clouds,
 * whose point order is spatially coherent per depth map).
 */
export function goldenStridePermutation(n: number, k: number): Uint32Array {
  const m = Math.max(0, Math.min(n, k));
  const out = new Uint32Array(m);
  if (n <= 0) return out;
  let P = Math.max(1, Math.round(n * 0.6180339887498949));
  const gcd = (a: number, b: number): number => {
    while (b) [a, b] = [b, a % b];
    return a;
  };
  while (gcd(P, n) !== 1) P++;
  let acc = 0;
  for (let i = 0; i < m; i++) {
    out[i] = acc;
    acc += P;
    if (acc >= n) acc -= n;
  }
  return out;
}
