/**
 * Gyro-based rolling-shutter (RS) correction of keypoints, and its estimation from a reconstruction.
 *
 * Model. A CMOS sensor exposes its lines one after the other: line s ∈ [0, 1] (in readout order)
 * of a frame stamped t is captured at t + (s − ½)·|readout| (the stamp is mid-readout; any
 * constant difference — iOS may stamp the start of readout or exposure — is absorbed by the
 * camera–IMU offset). In the canonical image the lines are rows (`axis: 'rows'`, s = v/(H−1)) or
 * columns (`'cols'`, s = u/(W−1)); a NEGATIVE `readoutMs` means the sweep runs towards −v / −u.
 * Hence a keypoint at (u, v) was seen at τ = (s(u, v) − ½)·readoutMs relative to the stamp.
 *
 * Correction. During the readout the camera rotates with the gyro angular velocity ω_c (camera
 * frame; a static direction moves as ẋ = −ω_c × x, `GyroReference.angularVelocityCam`); the
 * translation in ≤ 40 ms is negligible against the scene depth. The bearing observed at τ is
 * mapped to the stamp time by the rotation homography x₀ = exp([ω_c τ]×) x_τ, applied to
 * UNDISTORTED normalised coordinates (RADIAL model), then distorted back to pixels — so
 * corrected keypoints are ordinary keypoints of a global-shutter camera at the frame's pose.
 *
 * Estimation (`estimateRollingShutter`). The readout direction / time of a real device are not
 * documented (axis and sign depend on the sensor and the UI orientation): hypotheses axis ∈
 * {rows, cols} × readout ∈ [−40, 40] ms are scored on the registered images that rotated fastest
 * (where RS matters): their observations are corrected, each pose is re-fitted to the fixed 3-D
 * points (motion-only BA, Huber) and the robust (Cauchy, 1 px) reprojection cost is summed. A
 * coarse grid is refined by golden-section search on the best axis. RS is only accepted when it
 * lowers the cost significantly (default ≥ 3 %) — otherwise 0 (global shutter) is kept.
 */
import type { CameraIntrinsics, Pose, Vec3 } from '../core/types';
import { distortNormalized, undistortNormalized } from '../geometry/camera';
import { refinePose } from '../geometry/pnp';

/** Rolling-shutter readout in the canonical image (see the module comment). */
export interface RollingShutterModel {
  axis: 'rows' | 'cols';
  /** Signed readout time (ms) of the whole frame along +v (rows) / +u (cols); 0 = global shutter. */
  readoutMs: number;
}

/** Device-frame description (independent of the UI orientation): sweep direction + readout time. */
export interface RollingShutterDevice {
  /** Unit vector (device frame) in which the readout sweeps (from the first to the last line). */
  sweepDevice: [number, number, number];
  /** Readout time (ms, ≥ 0). */
  readoutMs: number;
}

/** Readout time offset (ms, relative to the frame stamp) of pixel (u, v) in a W×H image. */
export function rsLineTimeMs(model: RollingShutterModel, u: number, v: number, width: number, height: number): number {
  const s = model.axis === 'rows' ? v / Math.max(1, height - 1) : u / Math.max(1, width - 1);
  return (s - 0.5) * model.readoutMs;
}

/** Device-frame form of a canonical-image model (R_cam_device of the canonical images). */
export function rsModelToDevice(model: RollingShutterModel, R_cam_device: ArrayLike<number>): RollingShutterDevice {
  const sg = model.readoutMs >= 0 ? 1 : -1;
  const dc = model.axis === 'rows' ? [0, sg, 0] : [sg, 0, 0];
  const R = R_cam_device;
  // d_dev = R_cdᵀ d_cam
  const d: [number, number, number] = [R[0] * dc[0] + R[3] * dc[1] + R[6] * dc[2], R[1] * dc[0] + R[4] * dc[1] + R[7] * dc[2], R[2] * dc[0] + R[5] * dc[1] + R[8] * dc[2]];
  return { sweepDevice: d, readoutMs: Math.abs(model.readoutMs) };
}

/** Canonical-image model of a device-frame description; null when the sweep is not along an image axis. */
export function rsModelFromDevice(dev: RollingShutterDevice, R_cam_device: ArrayLike<number>): RollingShutterModel | null {
  const R = R_cam_device;
  const d = dev.sweepDevice;
  const x = R[0] * d[0] + R[1] * d[1] + R[2] * d[2];
  const y = R[3] * d[0] + R[4] * d[1] + R[5] * d[2];
  if (!(dev.readoutMs >= 0)) return null;
  if (Math.abs(y) > 0.9) return { axis: 'rows', readoutMs: Math.sign(y) * dev.readoutMs };
  if (Math.abs(x) > 0.9) return { axis: 'cols', readoutMs: Math.sign(x) * dev.readoutMs };
  return null;
}

const tmp = new Float64Array(2);

/**
 * Bearing (undistorted normalised x, y) of pixel (u, v) mapped from its readout time to the frame
 * stamp: x₀ = exp([ω τ]×) x_τ (written to out[0..1]); false if it rotates behind the camera.
 */
function correctNormalized(cam: CameraIntrinsics, omega: ArrayLike<number>, model: RollingShutterModel, u: number, v: number, out: Float64Array): boolean {
  undistortNormalized(cam, (u - cam.cx) / cam.f, (v - cam.cy) / cam.f, tmp);
  const x = tmp[0],
    y = tmp[1];
  const tau = rsLineTimeMs(model, u, v, cam.width, cam.height) / 1000;
  const px = omega[0] * tau,
    py = omega[1] * tau,
    pz = omega[2] * tau;
  const th = Math.sqrt(px * px + py * py + pz * pz);
  let X = x,
    Y = y,
    Z = 1;
  if (th > 1e-12) {
    // Rodrigues: v cosθ + (k × v) sinθ + k (k·v)(1 − cosθ)
    const kx = px / th,
      ky = py / th,
      kz = pz / th;
    const c = Math.cos(th),
      s = Math.sin(th);
    const kv = kx * x + ky * y + kz;
    X = x * c + (ky * 1 - kz * y) * s + kx * kv * (1 - c);
    Y = y * c + (kz * x - kx * 1) * s + ky * kv * (1 - c);
    Z = 1 * c + (kx * y - ky * x) * s + kz * kv * (1 - c);
  }
  if (!(Z > 1e-9)) return false;
  out[0] = X / Z;
  out[1] = Y / Z;
  return true;
}

/**
 * Rolling-shutter-corrected copy of keypoints (stride 4: x, y, scale, orientation; full-resolution
 * pixels of a cam.width × cam.height image): positions mapped to the frame stamp with the camera
 * angular velocity ω (rad/s, camera frame); scale / orientation copied. `out` may alias `kp`.
 */
export function correctKeypointsRS(
  kp: Float32Array,
  cam: CameraIntrinsics,
  omega: ArrayLike<number>,
  model: RollingShutterModel,
  out: Float32Array = new Float32Array(kp.length),
): Float32Array {
  const n = kp.length >> 2;
  const nrm = new Float64Array(2);
  const d = new Float64Array(2);
  for (let i = 0; i < n; i++) {
    const u = kp[4 * i],
      v = kp[4 * i + 1];
    let ou = u,
      ov = v;
    if (model.readoutMs !== 0 && Number.isFinite(u) && Number.isFinite(v) && correctNormalized(cam, omega, model, u, v, nrm)) {
      distortNormalized(cam, nrm[0], nrm[1], d);
      ou = cam.f * d[0] + cam.cx;
      ov = cam.f * d[1] + cam.cy;
    }
    out[4 * i] = ou;
    out[4 * i + 1] = ov;
    out[4 * i + 2] = kp[4 * i + 2];
    out[4 * i + 3] = kp[4 * i + 3];
  }
  return out;
}

/** Observations of one registered image for the RS estimation. */
export interface RSImageObservations {
  id: number;
  cam: CameraIntrinsics;
  pose: Pose;
  /** Camera angular velocity at the frame time (rad/s, camera frame). */
  omega: Vec3;
  /** Raw (uncorrected) pixel observations, 2 per observation. */
  uv: Float64Array;
  /** The observed 3-D points (world), 3 per observation. */
  X: Float64Array;
}

export interface RollingShutterEstimateOptions {
  /** Readout range searched (ms, symmetric; default 40). */
  maxReadoutMs: number;
  /** Coarse grid step (ms; default 5). */
  stepMs: number;
  /** Cauchy scale of the reprojection cost (px; default 1). */
  lossScalePx: number;
  /** Relative cost decrease required to accept RS ≠ 0 (default 0.03). */
  minImprovement: number;
  /** |readout| below which RS is not worth applying (ms; default 3). */
  minReadoutMs: number;
  /** Pose-refinement iterations per image and hypothesis (default 8). */
  iterations: number;
}

export const DEFAULT_RS_ESTIMATE: Readonly<RollingShutterEstimateOptions> = Object.freeze({
  maxReadoutMs: 40,
  stepMs: 5,
  lossScalePx: 1,
  minImprovement: 0.03,
  minReadoutMs: 3,
  iterations: 8,
});

export interface RollingShutterEstimate {
  /** Best hypothesis (readoutMs 0 when not accepted). */
  model: RollingShutterModel;
  accepted: boolean;
  reason: string;
  /** Robust cost of global shutter and of the best hypothesis. */
  cost0: number;
  costBest: number;
  /** (cost0 − costBest) / cost0. */
  improvement: number;
  /** The best hypothesis even when not accepted. */
  best: RollingShutterModel;
  /** Every evaluated hypothesis (diagnostics). */
  curve: { axis: 'rows' | 'cols'; readoutMs: number; cost: number }[];
  numImages: number;
  numObservations: number;
}

/** Robust cost of one hypothesis: per image, correct, re-fit the pose to the fixed points, sum Cauchy(r²). */
export function rsHypothesisCost(images: readonly RSImageObservations[], model: RollingShutterModel, o: Partial<RollingShutterEstimateOptions> = {}): number {
  const c2 = (o.lossScalePx ?? DEFAULT_RS_ESTIMATE.lossScalePx) ** 2;
  const iters = o.iterations ?? DEFAULT_RS_ESTIMATE.iterations;
  const nrm = new Float64Array(2);
  let cost = 0;
  for (const im of images) {
    const n = im.uv.length >> 1;
    const xn = new Float64Array(2 * n);
    for (let k = 0; k < n; k++) {
      const u = im.uv[2 * k],
        v = im.uv[2 * k + 1];
      if (model.readoutMs !== 0 && correctNormalized(im.cam, im.omega, model, u, v, nrm)) {
        xn[2 * k] = nrm[0];
        xn[2 * k + 1] = nrm[1];
      } else {
        undistortNormalized(im.cam, (u - im.cam.cx) / im.cam.f, (v - im.cam.cy) / im.cam.f, tmp);
        xn[2 * k] = tmp[0];
        xn[2 * k + 1] = tmp[1];
      }
    }
    const pose = refinePose(im.pose, im.X, xn, n, { iterations: iters, huberNormalized: 2 / im.cam.f });
    const R = pose.R,
      t = pose.t,
      f = im.cam.f;
    for (let k = 0; k < n; k++) {
      const x = im.X[3 * k],
        y = im.X[3 * k + 1],
        z = im.X[3 * k + 2];
      const p2 = R[6] * x + R[7] * y + R[8] * z + t[2];
      if (!(p2 > 0)) {
        cost += c2 * Math.log1p(1e4 / c2);
        continue;
      }
      const du = ((R[0] * x + R[1] * y + R[2] * z + t[0]) / p2 - xn[2 * k]) * f;
      const dv = ((R[3] * x + R[4] * y + R[5] * z + t[1]) / p2 - xn[2 * k + 1]) * f;
      cost += c2 * Math.log1p((du * du + dv * dv) / c2);
    }
  }
  return cost;
}

/**
 * Estimate the rolling-shutter readout (axis, signed time) from registered images with their
 * observations (pick the ones that rotated fastest; see the module comment).
 */
export function estimateRollingShutter(images: readonly RSImageObservations[], opts: Partial<RollingShutterEstimateOptions> = {}): RollingShutterEstimate {
  const o = { ...DEFAULT_RS_ESTIMATE, ...opts };
  let nObs = 0;
  for (const im of images) nObs += im.uv.length >> 1;
  const curve: RollingShutterEstimate['curve'] = [];
  const evalH = (axis: 'rows' | 'cols', readoutMs: number): number => {
    const c = rsHypothesisCost(images, { axis, readoutMs }, o);
    curve.push({ axis, readoutMs, cost: c });
    return c;
  };
  const zero: RollingShutterModel = { axis: 'rows', readoutMs: 0 };
  if (images.length === 0 || nObs === 0) {
    return { model: zero, accepted: false, reason: 'no observations', cost0: NaN, costBest: NaN, improvement: 0, best: zero, curve, numImages: 0, numObservations: 0 };
  }
  const cost0 = evalH('rows', 0);
  let best: RollingShutterModel = zero,
    bestCost = cost0;
  const nGrid = Math.max(1, Math.round(o.maxReadoutMs / o.stepMs));
  for (const axis of ['rows', 'cols'] as const) {
    for (let k = -nGrid; k <= nGrid; k++) {
      if (k === 0) continue;
      const r = (k * o.maxReadoutMs) / nGrid;
      const c = evalH(axis, r);
      if (c < bestCost) {
        bestCost = c;
        best = { axis, readoutMs: r };
      }
    }
  }
  if (best.readoutMs !== 0) {
    // golden-section refinement on the best axis within ± one grid step
    const step = o.maxReadoutMs / nGrid;
    let a = best.readoutMs - step,
      b = best.readoutMs + step;
    const g = (Math.sqrt(5) - 1) / 2;
    const axis = best.axis;
    let x1 = b - g * (b - a),
      x2 = a + g * (b - a);
    let f1 = evalH(axis, x1),
      f2 = evalH(axis, x2);
    for (let it = 0; it < 8; it++) {
      if (f1 < f2) {
        b = x2;
        x2 = x1;
        f2 = f1;
        x1 = b - g * (b - a);
        f1 = evalH(axis, x1);
      } else {
        a = x1;
        x1 = x2;
        f1 = f2;
        x2 = a + g * (b - a);
        f2 = evalH(axis, x2);
      }
    }
    const xm = f1 < f2 ? x1 : x2;
    const fm = Math.min(f1, f2);
    if (fm < bestCost) {
      bestCost = fm;
      best = { axis, readoutMs: xm };
    }
  }
  const improvement = cost0 > 0 ? (cost0 - bestCost) / cost0 : 0;
  let accepted = true,
    reason = `${best.axis} ${best.readoutMs.toFixed(1)} ms lowers the robust reprojection cost by ${(100 * improvement).toFixed(1)} %`;
  if (!(Math.abs(best.readoutMs) >= o.minReadoutMs)) {
    accepted = false;
    reason = `best readout ${best.readoutMs.toFixed(1)} ms is below ${o.minReadoutMs} ms`;
  } else if (!(improvement >= o.minImprovement)) {
    accepted = false;
    reason = `${best.axis} ${best.readoutMs.toFixed(1)} ms lowers the cost by only ${(100 * improvement).toFixed(1)} % (< ${(100 * o.minImprovement).toFixed(0)} %)`;
  }
  return {
    model: accepted ? best : zero,
    accepted,
    reason,
    cost0,
    costBest: bestCost,
    improvement,
    best,
    curve,
    numImages: images.length,
    numObservations: nObs,
  };
}
