/**
 * Frame / camera conventions of docs/ARCHITECTURE.md §2, implemented locally for the
 * synthetic generator (ground truth must not depend on the code under test).
 *
 *  - World: Z-up ENU (x east, y north, z up), metres.
 *  - Pose: world-to-camera, x_cam = R X + t, OpenCV camera frame.
 *  - R_cam_device(θ) = diag(1,−1,−1) · Rz(θ); camera rotation R = R_cam_device · R_wdᵀ.
 *  - RADIAL camera model with pixel centres at integer coordinates.
 */
import type { CameraIntrinsics, Mat3, Pose, Vec3 } from '../../core/types';
import { m3Mul, m3T, m3TVec, m3Vec, rotZ } from './mat';

export type ScreenAngle = 0 | 90 | 180 | 270;

/** R_cam_device(θ) = diag(1, −1, −1) · Rz(θ)  (v_cam = R_cam_device · v_device). */
export function rCamDevice(thetaDeg: number): Mat3 {
  const th = (thetaDeg * Math.PI) / 180;
  const Rz = rotZ(th);
  // Snap to exact 0/±1 for the multiples of 90° we use.
  for (let i = 0; i < 9; i++) if (Math.abs(Rz[i]) < 1e-12) Rz[i] = 0;
  const D = Float64Array.of(1, 0, 0, 0, -1, 0, 0, 0, -1);
  return m3Mul(D, Rz);
}

/** Camera pose (world-to-camera) from the device attitude R_wd and the camera centre. */
export function poseFromDevice(R_wd: ArrayLike<number>, center: ArrayLike<number>, thetaDeg: number): Pose {
  const R = m3Mul(rCamDevice(thetaDeg), m3T(R_wd));
  const t = m3Vec(R, center);
  t[0] = -t[0];
  t[1] = -t[1];
  t[2] = -t[2];
  return { R, t };
}

/** Inverse of poseFromDevice: device attitude R_wd = Rᵀ R_cam_device and the centre. */
export function deviceFromPose(pose: Pose, thetaDeg: number): { R_wd: Mat3; center: Vec3 } {
  const R_wd = m3Mul(m3T(pose.R), rCamDevice(thetaDeg));
  return { R_wd, center: cameraCenter(pose) };
}

/** C = −Rᵀ t */
export function cameraCenter(pose: Pose): Vec3 {
  const c = m3TVec(pose.R, pose.t);
  c[0] = -c[0];
  c[1] = -c[1];
  c[2] = -c[2];
  return c;
}

/** Optical axis (+z_cam) in world coordinates = third row of R. */
export function cameraForward(pose: Pose): Vec3 {
  return Float64Array.of(pose.R[6], pose.R[7], pose.R[8]);
}

/**
 * Compass heading (degrees clockwise from true north, [0, 360)) of a world direction
 * projected onto the horizontal plane; NaN when the direction is (nearly) vertical.
 */
export function headingDegOfDirection(d: ArrayLike<number>, minHorizontal = 1e-6): number {
  const h = Math.hypot(d[0], d[1]);
  if (h < minHorizontal) return NaN;
  let deg = (Math.atan2(d[0], d[1]) * 180) / Math.PI; // east = +90
  if (deg < 0) deg += 360;
  if (deg >= 360) deg -= 360;
  return deg;
}

// ---------------------------------------------------------------------------
// RADIAL camera model
// ---------------------------------------------------------------------------

/** Distortion of normalized coordinates: (x, y) ↦ d·(x, y), d = 1 + k1 r² + k2 r⁴. */
export function distortNormalized(k1: number, k2: number, x: number, y: number, out: Float64Array): void {
  const r2 = x * x + y * y;
  const d = 1 + k1 * r2 + k2 * r2 * r2;
  out[0] = d * x;
  out[1] = d * y;
}

/** Inverse of distortNormalized by Newton iteration on the radius. */
export function undistortNormalized(k1: number, k2: number, xd: number, yd: number, out: Float64Array): void {
  const rd = Math.hypot(xd, yd);
  if (rd < 1e-14 || (k1 === 0 && k2 === 0)) {
    out[0] = xd;
    out[1] = yd;
    return;
  }
  let r = rd;
  for (let i = 0; i < 30; i++) {
    const r2 = r * r;
    const f = r * (1 + k1 * r2 + k2 * r2 * r2) - rd;
    const fp = 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2;
    const step = f / fp;
    r -= step;
    if (Math.abs(step) < 1e-15 * Math.max(1, r)) break;
  }
  const s = r / rd;
  out[0] = xd * s;
  out[1] = yd * s;
}

const _xy = new Float64Array(2);

/**
 * Project a world point with the RADIAL model. Writes [u, v, z] into out and returns
 * true when the point is in front of the camera (z > 1e-9). Bounds are not checked.
 */
export function projectRadial(cam: CameraIntrinsics, pose: Pose, X: ArrayLike<number>, out: Float64Array): boolean {
  const R = pose.R,
    t = pose.t;
  const xc = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
  const yc = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
  const zc = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
  out[2] = zc;
  if (!(zc > 1e-9)) return false;
  distortNormalized(cam.k1, cam.k2, xc / zc, yc / zc, _xy);
  out[0] = cam.f * _xy[0] + cam.cx;
  out[1] = cam.f * _xy[1] + cam.cy;
  return true;
}

/** Rectangle of UNDISTORTED normalized coordinates (x right, y down). */
export interface NormalizedWindow {
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
}

/**
 * Smallest axis-aligned window of undistorted normalized coordinates that contains
 * the undistorted image of the whole distorted image area [-0.5, W-0.5] × [-0.5, H-0.5],
 * grown by `marginPx` (in pixels of focal f).
 */
export function undistortedWindow(cam: CameraIntrinsics, marginPx = 2): NormalizedWindow {
  let xmin = Infinity,
    xmax = -Infinity,
    ymin = Infinity,
    ymax = -Infinity;
  const out = new Float64Array(2);
  const visit = (u: number, v: number) => {
    undistortNormalized(cam.k1, cam.k2, (u - cam.cx) / cam.f, (v - cam.cy) / cam.f, out);
    if (out[0] < xmin) xmin = out[0];
    if (out[0] > xmax) xmax = out[0];
    if (out[1] < ymin) ymin = out[1];
    if (out[1] > ymax) ymax = out[1];
  };
  const W = cam.width,
    H = cam.height;
  const n = 256;
  for (let i = 0; i <= n; i++) {
    const u = -0.5 + (W * i) / n;
    const v = -0.5 + (H * i) / n;
    visit(u, -0.5);
    visit(u, H - 0.5);
    visit(-0.5, v);
    visit(W - 0.5, v);
  }
  // For pincushion distortion the extremum can lie inside the edges; also sample a grid.
  for (let j = 0; j <= 16; j++) for (let i = 0; i <= 16; i++) visit(-0.5 + (W * i) / 16, -0.5 + (H * j) / 16);
  const m = marginPx / cam.f;
  return { xmin: xmin - m, xmax: xmax + m, ymin: ymin - m, ymax: ymax + m };
}

/** Pinhole window covering exactly the pixel area of an undistorted camera. */
export function pinholeWindow(cam: CameraIntrinsics): NormalizedWindow {
  return {
    xmin: (-0.5 - cam.cx) / cam.f,
    xmax: (cam.width - 0.5 - cam.cx) / cam.f,
    ymin: (-0.5 - cam.cy) / cam.f,
    ymax: (cam.height - 0.5 - cam.cy) / cam.f,
  };
}

/** Focal length (px) for a horizontal field of view spanning `size` pixels. */
export function focalFromFov(sizePx: number, fovDeg: number): number {
  return sizePx / 2 / Math.tan((fovDeg * Math.PI) / 360);
}
