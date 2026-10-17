/**
 * Triangulation: normalised linear DLT (Hartley & Zisserman §12.2), Gauss–Newton/LM
 * refinement of the pixel reprojection error with RADIAL distortion, and helpers.
 */
import type { CameraIntrinsics, Pose, Vec3 } from '../core/types';
import { Matrix } from '../math/matrix';
import { nullVector, solveSPD } from '../math/decomp';

let dltMat = new Matrix(4, 4);

/**
 * Linear triangulation from n ≥ 2 views: poses (world-to-camera) and undistorted
 * normalised observations xn (2n). The world frame is first re-centred on the mean
 * camera centre and scaled by the mean centre distance (so large ENU offsets and any
 * scene scale are well conditioned), then the smallest right singular vector of the
 * 2n×4 DLT system is taken. Null if the point is (numerically) at infinity, all
 * camera centres coincide, or the input is degenerate.
 */
export function triangulateDLT(poses: Pose[], xn: ArrayLike<number>, n: number, out?: Vec3): Vec3 | null {
  if (n < 2 || poses.length < n) return null;
  // Normalising similarity: X = s X' + c
  let cx = 0,
    cy = 0,
    cz = 0;
  for (let k = 0; k < n; k++) {
    const R = poses[k].R,
      t = poses[k].t;
    cx -= R[0] * t[0] + R[3] * t[1] + R[6] * t[2];
    cy -= R[1] * t[0] + R[4] * t[1] + R[7] * t[2];
    cz -= R[2] * t[0] + R[5] * t[1] + R[8] * t[2];
  }
  cx /= n;
  cy /= n;
  cz /= n;
  let s = 0;
  for (let k = 0; k < n; k++) {
    const R = poses[k].R,
      t = poses[k].t;
    const Cx = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]) - cx;
    const Cy = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]) - cy;
    const Cz = -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]) - cz;
    s += Math.sqrt(Cx * Cx + Cy * Cy + Cz * Cz);
  }
  s /= n;
  // Coincident centres (no baseline): depth is unobservable.
  if (!(s > 1e-12 * (Math.abs(cx) + Math.abs(cy) + Math.abs(cz) + 1e-300)) || !Number.isFinite(s)) return null;
  const is = 1 / s;
  if (dltMat.rows !== 2 * n) dltMat = new Matrix(2 * n, 4);
  const A = dltMat.data;
  for (let k = 0; k < n; k++) {
    const R = poses[k].R,
      t = poses[k].t;
    // t' = (R c + t) / s
    const t0 = (R[0] * cx + R[1] * cy + R[2] * cz + t[0]) * is;
    const t1 = (R[3] * cx + R[4] * cy + R[5] * cz + t[1]) * is;
    const t2 = (R[6] * cx + R[7] * cy + R[8] * cz + t[2]) * is;
    const x = xn[2 * k],
      y = xn[2 * k + 1];
    const o = 8 * k;
    A[o] = x * R[6] - R[0];
    A[o + 1] = x * R[7] - R[1];
    A[o + 2] = x * R[8] - R[2];
    A[o + 3] = x * t2 - t0;
    A[o + 4] = y * R[6] - R[3];
    A[o + 5] = y * R[7] - R[4];
    A[o + 6] = y * R[8] - R[5];
    A[o + 7] = y * t2 - t1;
    // row normalisation (equal weight per observation)
    const n0 = Math.hypot(A[o], A[o + 1], A[o + 2], A[o + 3]),
      n1 = Math.hypot(A[o + 4], A[o + 5], A[o + 6], A[o + 7]);
    if (n0 > 0) for (let j = 0; j < 4; j++) A[o + j] /= n0;
    if (n1 > 0) for (let j = 4; j < 8; j++) A[o + j] /= n1;
  }
  const v = nullVector(dltMat);
  const w = v[3];
  const vn = Math.hypot(v[0], v[1], v[2]);
  if (!(Math.abs(w) > 1e-12 * vn) || !Number.isFinite(w)) return null;
  const res = out ?? new Float64Array(3);
  res[0] = (v[0] / w) * s + cx;
  res[1] = (v[1] / w) * s + cy;
  res[2] = (v[2] / w) * s + cz;
  if (!Number.isFinite(res[0] + res[1] + res[2])) return null;
  return res;
}

const TV_POSES: Pose[] = [];
const TV_X = new Float64Array(4);

/** Two-view linear triangulation from normalised observations x1, x2 (length 2 each). */
export function triangulateTwoView(
  p1: Pose,
  p2: Pose,
  x1: ArrayLike<number>,
  x2: ArrayLike<number>,
  out?: Vec3,
): Vec3 | null {
  TV_POSES[0] = p1;
  TV_POSES[1] = p2;
  TV_X[0] = x1[0];
  TV_X[1] = x1[1];
  TV_X[2] = x2[0];
  TV_X[3] = x2[1];
  return triangulateDLT(TV_POSES, TV_X, 2, out);
}

/** Sum of squared pixel reprojection errors (Infinity if behind a camera). */
function pointCost(
  poses: Pose[],
  cams: CameraIntrinsics[],
  uv: ArrayLike<number>,
  n: number,
  X0: number,
  X1: number,
  X2: number,
): number {
  let c = 0;
  for (let k = 0; k < n; k++) {
    const R = poses[k].R,
      t = poses[k].t,
      cam = cams[Math.min(k, cams.length - 1)];
    const pz = R[6] * X0 + R[7] * X1 + R[8] * X2 + t[2];
    if (!(pz > 0)) return Infinity;
    const x = (R[0] * X0 + R[1] * X1 + R[2] * X2 + t[0]) / pz;
    const y = (R[3] * X0 + R[4] * X1 + R[5] * X2 + t[1]) / pz;
    const r2 = x * x + y * y;
    const d = 1 + r2 * (cam.k1 + cam.k2 * r2);
    const du = cam.f * d * x + cam.cx - uv[2 * k],
      dv = cam.f * d * y + cam.cy - uv[2 * k + 1];
    c += du * du + dv * dv;
  }
  return c;
}

const RP_H = new Float64Array(9);
const RP_A = new Float64Array(9);
const RP_G = new Float64Array(3);

/**
 * Levenberg–Marquardt refinement of a 3-D point on the pixel reprojection error over
 * n observations (uv: 2n pixels, cams[k] for view k, or a single shared camera when
 * cams.length === 1), including RADIAL distortion. Returns a new Vec3 (the input
 * point if it cannot be improved, e.g. behind a camera).
 */
export function refinePoint(
  poses: Pose[],
  cams: CameraIntrinsics[],
  uv: ArrayLike<number>,
  n: number,
  X: ArrayLike<number>,
  iterations = 10,
): Vec3 {
  let X0 = X[0],
    X1 = X[1],
    X2 = X[2];
  if (cams.length === 0 || n <= 0 || poses.length < n) return Float64Array.of(X0, X1, X2);
  let cost = pointCost(poses, cams, uv, n, X0, X1, X2);
  if (!Number.isFinite(cost)) return Float64Array.of(X0, X1, X2);
  let lambda = 1e-3;
  for (let it = 0; it < iterations && cost > 0; it++) {
    RP_H.fill(0);
    RP_G.fill(0);
    for (let k = 0; k < n; k++) {
      const R = poses[k].R,
        t = poses[k].t,
        cam = cams[Math.min(k, cams.length - 1)];
      const p0 = R[0] * X0 + R[1] * X1 + R[2] * X2 + t[0];
      const p1 = R[3] * X0 + R[4] * X1 + R[5] * X2 + t[1];
      const p2 = R[6] * X0 + R[7] * X1 + R[8] * X2 + t[2];
      const iz = 1 / p2;
      const x = p0 * iz,
        y = p1 * iz;
      const r2 = x * x + y * y;
      const d = 1 + r2 * (cam.k1 + cam.k2 * r2);
      const dd = cam.k1 + 2 * cam.k2 * r2;
      const f = cam.f;
      const ru = f * d * x + cam.cx - uv[2 * k],
        rv = f * d * y + cam.cy - uv[2 * k + 1];
      // ∂(u,v)/∂(x,y)
      const a00 = f * (d + 2 * x * x * dd),
        a01 = f * 2 * x * y * dd,
        a11 = f * (d + 2 * y * y * dd);
      // ∂(x,y)/∂P = [[iz, 0, −x iz], [0, iz, −y iz]]; B = ∂(u,v)/∂P
      const b00 = a00 * iz,
        b01 = a01 * iz,
        b02 = -(a00 * x + a01 * y) * iz;
      const b10 = a01 * iz,
        b11 = a11 * iz,
        b12 = -(a01 * x + a11 * y) * iz;
      // J = B R
      const ju0 = b00 * R[0] + b01 * R[3] + b02 * R[6],
        ju1 = b00 * R[1] + b01 * R[4] + b02 * R[7],
        ju2 = b00 * R[2] + b01 * R[5] + b02 * R[8];
      const jv0 = b10 * R[0] + b11 * R[3] + b12 * R[6],
        jv1 = b10 * R[1] + b11 * R[4] + b12 * R[7],
        jv2 = b10 * R[2] + b11 * R[5] + b12 * R[8];
      RP_H[0] += ju0 * ju0 + jv0 * jv0;
      RP_H[1] += ju0 * ju1 + jv0 * jv1;
      RP_H[2] += ju0 * ju2 + jv0 * jv2;
      RP_H[4] += ju1 * ju1 + jv1 * jv1;
      RP_H[5] += ju1 * ju2 + jv1 * jv2;
      RP_H[8] += ju2 * ju2 + jv2 * jv2;
      RP_G[0] -= ju0 * ru + jv0 * rv;
      RP_G[1] -= ju1 * ru + jv1 * rv;
      RP_G[2] -= ju2 * ru + jv2 * rv;
    }
    RP_H[3] = RP_H[1];
    RP_H[6] = RP_H[2];
    RP_H[7] = RP_H[5];
    let improved = false;
    let converged = false;
    for (let tries = 0; tries < 10; tries++) {
      RP_A.set(RP_H);
      RP_A[0] += lambda * Math.max(RP_H[0], 1e-300);
      RP_A[4] += lambda * Math.max(RP_H[4], 1e-300);
      RP_A[8] += lambda * Math.max(RP_H[8], 1e-300);
      const dX = solveSPD(RP_A, 3, RP_G);
      if (!dX) {
        lambda *= 10;
        continue;
      }
      const n0 = X0 + dX[0],
        n1 = X1 + dX[1],
        n2 = X2 + dX[2];
      const c2 = pointCost(poses, cams, uv, n, n0, n1, n2);
      if (c2 < cost) {
        converged = (cost - c2) <= 1e-12 * cost;
        X0 = n0;
        X1 = n1;
        X2 = n2;
        cost = c2;
        lambda = Math.max(lambda * 0.1, 1e-12);
        improved = true;
        break;
      }
      lambda *= 10;
    }
    if (!improved || converged) break;
  }
  return Float64Array.of(X0, X1, X2);
}

/** Angle (radians) between the rays C1 → X and C2 → X. */
export function triangulationAngle(C1: ArrayLike<number>, C2: ArrayLike<number>, X: ArrayLike<number>): number {
  const ax = X[0] - C1[0],
    ay = X[1] - C1[1],
    az = X[2] - C1[2];
  const bx = X[0] - C2[0],
    by = X[1] - C2[1],
    bz = X[2] - C2[2];
  const cx = ay * bz - az * by,
    cy = az * bx - ax * bz,
    cz = ax * by - ay * bx;
  return Math.atan2(Math.sqrt(cx * cx + cy * cy + cz * cz), ax * bx + ay * by + az * bz);
}

/** Depth z of a world point in the camera frame of `pose`. */
export function pointDepth(pose: Pose, X: ArrayLike<number>): number {
  const R = pose.R;
  return R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + pose.t[2];
}
