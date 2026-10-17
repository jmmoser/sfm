/**
 * RADIAL camera model (see docs/ARCHITECTURE.md §2 and `CameraIntrinsics`):
 *   x = X/Z, y = Y/Z, r² = x² + y², d = 1 + k1 r² + k2 r⁴,
 *   u = f d x + cx, v = f d y + cy   (pixel centres at integer coordinates).
 *
 * All functions are allocation-free when `out` is supplied.
 */
import type { CameraIntrinsics, Pose } from '../core/types';

/**
 * Intrinsics from an image size and a horizontal field of view:
 * f = (W/2) / tan(hfov/2), cx = (W−1)/2, cy = (H−1)/2.
 */
export function createIntrinsics(width: number, height: number, hfovRad: number, k1 = 0, k2 = 0): CameraIntrinsics {
  const f = (0.5 * width) / Math.tan(0.5 * hfovRad);
  return { width, height, f, cx: 0.5 * (width - 1), cy: 0.5 * (height - 1), k1, k2 };
}

/**
 * Intrinsics of the same camera for an image resampled to newWidth × newHeight
 * (§2 resampling rule: c' = (c + 0.5)·s − 0.5 per axis; f' = f·(sx + sy)/2, which
 * is f·s for aspect-preserving resampling). Distortion is unchanged (it acts on
 * normalised coordinates).
 */
export function scaleIntrinsics(cam: CameraIntrinsics, newWidth: number, newHeight: number): CameraIntrinsics {
  const sx = newWidth / cam.width;
  const sy = newHeight / cam.height;
  return {
    width: newWidth,
    height: newHeight,
    f: cam.f * 0.5 * (sx + sy),
    cx: (cam.cx + 0.5) * sx - 0.5,
    cy: (cam.cy + 0.5) * sy - 0.5,
    k1: cam.k1,
    k2: cam.k2,
  };
}

/** Applies radial distortion to normalised coordinates: out = (1 + k1 r² + k2 r⁴)·(x, y). */
export function distortNormalized(cam: CameraIntrinsics, x: number, y: number, out: Float64Array): void {
  const r2 = x * x + y * y;
  const d = 1 + r2 * (cam.k1 + cam.k2 * r2);
  out[0] = d * x;
  out[1] = d * y;
}

/**
 * Radius r ≥ 0 with g(r) = r (1 + k1 r² + k2 r⁴) = rd, by bracketed Newton
 * (safeguarded with bisection). The search is restricted to the monotone branch
 * [0, rMax] of g (rMax = first positive root of g′ if any); if rd exceeds the
 * maximum of g on that branch (no inverse exists — a point outside the valid image
 * circle of a strongly barrel-distorted lens), rMax is returned.
 */
function undistortRadius(k1: number, k2: number, rd: number): number {
  // g'(r) = 1 + 3 k1 s + 5 k2 s², s = r². First positive root s* (if any).
  let sMax = Infinity;
  if (k2 === 0) {
    if (k1 < 0) sMax = -1 / (3 * k1);
  } else {
    const a = 5 * k2,
      b = 3 * k1;
    const disc = b * b - 4 * a; // c = 1
    if (disc >= 0) {
      const sq = Math.sqrt(disc);
      const q = -0.5 * (b + (b >= 0 ? sq : -sq));
      const s1 = q / a,
        s2 = q !== 0 ? 1 / q : Infinity; // c / q
      if (s1 > 0) sMax = Math.min(sMax, s1);
      if (s2 > 0) sMax = Math.min(sMax, s2);
    }
  }
  let lo = 0;
  let hi: number;
  if (sMax < Infinity) {
    hi = Math.sqrt(sMax);
    const gHi = hi * (1 + sMax * (k1 + k2 * sMax));
    if (rd >= gHi) return hi;
  } else {
    // g is increasing and unbounded: grow hi until g(hi) ≥ rd.
    hi = Math.max(rd, 1e-300);
    for (let it = 0; it < 2000; it++) {
      const s = hi * hi;
      if (hi * (1 + s * (k1 + k2 * s)) >= rd) break;
      hi *= 2;
    }
  }
  let r = Math.min(Math.max(rd, lo), hi);
  for (let it = 0; it < 100; it++) {
    const s = r * r;
    const g = r * (1 + s * (k1 + k2 * s)) - rd;
    if (g === 0) return r;
    if (g > 0) hi = r;
    else lo = r;
    const dg = 1 + s * (3 * k1 + 5 * k2 * s);
    let rn = r - g / dg;
    if (!(rn > lo && rn < hi) || !(dg > 0)) rn = 0.5 * (lo + hi);
    if (Math.abs(rn - r) <= 1e-16 * Math.max(r, 1e-300)) return rn;
    r = rn;
    if (hi - lo <= 2e-16 * hi) return r;
  }
  return r;
}

/**
 * Inverse of `distortNormalized` (undistorted normalised coordinates from distorted
 * ones). Solves the 1-D radial equation exactly (Newton, ~3–5 iterations; agrees
 * with `distortNormalized` to ≤ 1e-12 relative for |k| < 0.5 inside the valid
 * radius). Outside the invertible radius of a barrel lens the point is clamped to
 * the maximum valid radius.
 */
export function undistortNormalized(cam: CameraIntrinsics, xd: number, yd: number, out: Float64Array): void {
  const k1 = cam.k1,
    k2 = cam.k2;
  if (k1 === 0 && k2 === 0) {
    out[0] = xd;
    out[1] = yd;
    return;
  }
  const rd = Math.sqrt(xd * xd + yd * yd);
  if (rd === 0) {
    out[0] = 0;
    out[1] = 0;
    return;
  }
  const r = undistortRadius(k1, k2, rd);
  const s = r / rd;
  out[0] = xd * s;
  out[1] = yd * s;
}

/** Camera-frame point → pixel. Returns false (out untouched) if Z ≤ 1e-12. */
export function projectCam(cam: CameraIntrinsics, X: ArrayLike<number>, out: Float64Array): boolean {
  const Z = X[2];
  if (!(Z > 1e-12)) return false;
  const x = X[0] / Z,
    y = X[1] / Z;
  const r2 = x * x + y * y;
  const d = 1 + r2 * (cam.k1 + cam.k2 * r2);
  out[0] = cam.f * d * x + cam.cx;
  out[1] = cam.f * d * y + cam.cy;
  return true;
}

/** World point → pixel through a world-to-camera pose. False if behind the camera. */
export function projectWorld(cam: CameraIntrinsics, pose: Pose, X: ArrayLike<number>, out: Float64Array): boolean {
  const R = pose.R,
    t = pose.t;
  const x0 = X[0],
    x1 = X[1],
    x2 = X[2];
  const Z = R[6] * x0 + R[7] * x1 + R[8] * x2 + t[2];
  if (!(Z > 1e-12)) return false;
  const x = (R[0] * x0 + R[1] * x1 + R[2] * x2 + t[0]) / Z;
  const y = (R[3] * x0 + R[4] * x1 + R[5] * x2 + t[1]) / Z;
  const r2 = x * x + y * y;
  const d = 1 + r2 * (cam.k1 + cam.k2 * r2);
  out[0] = cam.f * d * x + cam.cx;
  out[1] = cam.f * d * y + cam.cy;
  return true;
}

/** Undistorted normalised coordinates → pixel (applies distortion and K). */
export function normalizedToPixel(cam: CameraIntrinsics, x: number, y: number, out: Float64Array): void {
  const r2 = x * x + y * y;
  const d = 1 + r2 * (cam.k1 + cam.k2 * r2);
  out[0] = cam.f * d * x + cam.cx;
  out[1] = cam.f * d * y + cam.cy;
}

/** Pixel → undistorted normalised image coordinates (x, y) with the ray (x, y, 1). */
export function pixelToNormalized(cam: CameraIntrinsics, u: number, v: number, out: Float64Array): void {
  const inv = 1 / cam.f;
  undistortNormalized(cam, (u - cam.cx) * inv, (v - cam.cy) * inv, out);
}

const SCR2 = new Float64Array(2);

/** Pixel → unit bearing vector in the camera frame. */
export function pixelToBearing(cam: CameraIntrinsics, u: number, v: number, out: Float64Array): void {
  pixelToNormalized(cam, u, v, SCR2);
  const x = SCR2[0],
    y = SCR2[1];
  const inv = 1 / Math.sqrt(x * x + y * y + 1);
  out[0] = x * inv;
  out[1] = y * inv;
  out[2] = inv;
}

/**
 * Pixel → undistorted pixel of the equivalent pinhole camera with the same f, cx, cy
 * (u' = f x + cx with (x, y) = pixelToNormalized(u, v)).
 */
export function undistortPixel(cam: CameraIntrinsics, u: number, v: number, out: Float64Array): void {
  pixelToNormalized(cam, u, v, out);
  out[0] = cam.f * out[0] + cam.cx;
  out[1] = cam.f * out[1] + cam.cy;
}

/**
 * Keypoints (stride 4: x, y, scale, orientation; pixels) → undistorted normalised
 * coordinates, flat [x0, y0, x1, y1, …] (length 2n).
 */
export function keypointsToNormalized(cam: CameraIntrinsics, kp: Float32Array, out?: Float64Array): Float64Array {
  const n = kp.length >> 2;
  const res = out && out.length >= 2 * n ? out : new Float64Array(2 * n);
  const inv = 1 / cam.f,
    cx = cam.cx,
    cy = cam.cy;
  const distorted = cam.k1 !== 0 || cam.k2 !== 0;
  for (let i = 0; i < n; i++) {
    const xd = (kp[4 * i] - cx) * inv,
      yd = (kp[4 * i + 1] - cy) * inv;
    if (distorted) {
      undistortNormalized(cam, xd, yd, SCR2);
      res[2 * i] = SCR2[0];
      res[2 * i + 1] = SCR2[1];
    } else {
      res[2 * i] = xd;
      res[2 * i + 1] = yd;
    }
  }
  return res;
}

/** 3×3 calibration matrix K (row-major) of the pinhole part. */
export function intrinsicsMatrix(cam: CameraIntrinsics, out: Float64Array = new Float64Array(9)): Float64Array {
  out[0] = cam.f;
  out[1] = 0;
  out[2] = cam.cx;
  out[3] = 0;
  out[4] = cam.f;
  out[5] = cam.cy;
  out[6] = 0;
  out[7] = 0;
  out[8] = 1;
  return out;
}

/** Inverse calibration matrix K⁻¹ (row-major). */
export function intrinsicsMatrixInverse(cam: CameraIntrinsics, out: Float64Array = new Float64Array(9)): Float64Array {
  const inv = 1 / cam.f;
  out[0] = inv;
  out[1] = 0;
  out[2] = -cam.cx * inv;
  out[3] = 0;
  out[4] = inv;
  out[5] = -cam.cy * inv;
  out[6] = 0;
  out[7] = 0;
  out[8] = 1;
  return out;
}
