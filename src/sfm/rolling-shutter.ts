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
 * frame; a static direction moves as ẋ = −ω_c × x, `GyroReference.angularVelocityCam`) and its
 * centre moves with the velocity v_c (camera frame). The bearing observed at τ is mapped to the
 * stamp time by the rotation homography x₀ = exp([ω_c τ]×) x_τ, applied to UNDISTORTED normalised
 * coordinates (RADIAL model), plus — when the velocity and the point's inverse depth ρ = 1/Z are
 * known (`RSTranslation`, from a reconstruction) — the first-order translational term
 * x₀ += τ ρ (v_xy − x v_z), then distorted back to pixels: corrected keypoints are ordinary
 * keypoints of a global-shutter camera at the frame's pose. The translational term matters on
 * close-range orbits: there the camera moves sideways at ≈ |ω| × the object distance, its image
 * motion cancels the yaw term at the orbit centre, and a rotation-only model is left with an
 * effective (too short: e2e 16.5 of 22 ms) readout that under-corrects the roll term — a
 * horizontal readout with roll bends the image vertically ∝ x², i.e. a common pitch of every
 * camera, which moves the object ≈ 3 mm against the camera ring (object-real).
 *
 * Estimation (`estimateRollingShutter`). The readout direction / time of a real device are not
 * documented (axis and sign depend on the sensor and the UI orientation): hypotheses axis ∈
 * {rows, cols} × readout ∈ [−40, 40] ms are scored by a robust (Cauchy, 1 px) reprojection cost
 * after a SHORT BUNDLE ADJUSTMENT on a subset (`RSBundleCost`): the given images (the engine: every
 * registered image, with a gauge) and a subsample of their points are free, any other image observing
 * those points is held constant, all observations are corrected for the hypothesis (with the
 * translational term when the images' velocities are given: the observation's inverse depth comes
 * from the input reconstruction, the velocity from its neighbouring camera centres). The points must be
 * free: they were triangulated from RS-distorted observations, mostly of the same fast pans, and
 * a motion-only test against them (`rsHypothesisCost`, kept for tests) finds the axis and
 * direction but only ≈ 15–20 % of the readout time (e2e: 3–5 ms of 22 ms). Search: 0, then
 * ±10 / ±25 ms on both axes pick the axis and direction, then parabola fits along it. RS is only accepted when it lowers the cost significantly (default ≥ 3 %) —
 * otherwise 0 (global shutter) is kept.
 */
import type { BACameraRefine, BAProblem } from '../ba/problem';
import { bundleAdjust } from '../ba/solver';
import type { CameraIntrinsics, Pose, RGBAImage, Vec3 } from '../core/types';
import { distortNormalized, undistortNormalized } from '../geometry/camera';
import { refinePose } from '../geometry/pnp';
import type { Reconstruction } from './reconstruction';

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

/**
 * Translational part of an image's rolling-shutter correction (see the module comment). Scene units
 * cancel (velocity × inverse depth), so any reconstruction scale works.
 */
export interface RSTranslation {
  /** Camera-frame velocity of the camera centre (scene units / s). */
  velocity: ArrayLike<number>;
  /** Inverse depth (1 / scene units) per keypoint; NaN or ≤ 0 = unknown ⇒ `defaultInvDepth`. */
  invDepth?: ArrayLike<number> | null;
  /** Inverse depth of keypoints without one (e.g. the image's median); 0 = no translational term for them. */
  defaultInvDepth: number;
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
 * stamp: x₀ = exp([ω τ]×) x_τ (+ τ ρ (v_xy − x v_z) when `vel` and the inverse depth ρ > 0 are given;
 * written to out[0..1]); false if it rotates behind the camera.
 */
function correctNormalized(
  cam: CameraIntrinsics,
  omega: ArrayLike<number>,
  model: RollingShutterModel,
  u: number,
  v: number,
  out: Float64Array,
  vel: ArrayLike<number> | null = null,
  invDepth = 0,
): boolean {
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
  if (vel && invDepth > 0) {
    const k = tau * invDepth;
    out[0] += k * (vel[0] - out[0] * vel[2]);
    out[1] += k * (vel[1] - out[1] * vel[2]);
  }
  return true;
}

/**
 * Rolling-shutter-corrected copy of keypoints (stride 4: x, y, scale, orientation; full-resolution
 * pixels of a cam.width × cam.height image): positions mapped to the frame stamp with the camera
 * angular velocity ω (rad/s, camera frame) and, when given, the translational term (`trans`);
 * scale / orientation copied. `out` may alias `kp`.
 */
export function correctKeypointsRS(
  kp: Float32Array,
  cam: CameraIntrinsics,
  omega: ArrayLike<number>,
  model: RollingShutterModel,
  out: Float32Array = new Float32Array(kp.length),
  trans: RSTranslation | null = null,
): Float32Array {
  const n = kp.length >> 2;
  const nrm = new Float64Array(2);
  const d = new Float64Array(2);
  const vel = trans ? trans.velocity : null;
  const inv = trans?.invDepth ?? null;
  const inv0 = trans ? trans.defaultInvDepth : 0;
  for (let i = 0; i < n; i++) {
    const u = kp[4 * i],
      v = kp[4 * i + 1];
    let ou = u,
      ov = v;
    const rho = inv && inv[i] > 0 ? inv[i] : inv0;
    if (model.readoutMs !== 0 && Number.isFinite(u) && Number.isFinite(v) && correctNormalized(cam, omega, model, u, v, nrm, vel, rho)) {
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
  /**
   * Refine an earlier estimate instead of searching every axis / direction: hypotheses 0.85 / 1 /
   * 1.15 × its readout on its axis, extended by 1.3× steps until bracketed, then the parabola search.
   * For a second pass on the reconstruction re-adjusted with the first estimate (the short BA
   * starts from the input reconstruction, which under-estimates when that was built on distorted
   * keypoints: synthetic orbit −18.5 → −21.7 ms of −22).
   */
  around?: RollingShutterModel | null;
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

/** Cost of a rolling-shutter hypothesis (lower is better). */
export type RSCostFunction = (model: RollingShutterModel) => number;

/**
 * Search the readout (axis, signed time) minimising `cost`: 0; ±10 and ±25 ms (for the default
 * 40 ms range) on both axes choose the axis and direction — ±25 ms only along the direction that
 * is clearly best at ±10 ms, when one is (3 evaluations fewer, same result); along it, successive
 * parabola fits (≤ 3 more evaluations). Accepted when the relative decrease against 0 reaches
 * `minImprovement` and |readout| ≥ `minReadoutMs`. ≈ 10–18 cost evaluations in total.
 */
export function searchRollingShutter(cost: RSCostFunction, opts: Partial<RollingShutterEstimateOptions> = {}, info: { numImages?: number; numObservations?: number } = {}): RollingShutterEstimate {
  const o = { ...DEFAULT_RS_ESTIMATE, ...opts };
  const curve: RollingShutterEstimate['curve'] = [];
  const evalH = (axis: 'rows' | 'cols', readoutMs: number): number => {
    const c = cost({ axis, readoutMs });
    curve.push({ axis, readoutMs, cost: c });
    return c;
  };
  const zero: RollingShutterModel = { axis: 'rows', readoutMs: 0 };
  const cost0 = evalH('rows', 0);
  let best: RollingShutterModel = zero,
    bestCost = cost0;
  let dir: { axis: 'rows' | 'cols'; sign: number } | null = null;
  let dirCost = Infinity;
  const around = o.around && o.around.readoutMs !== 0 ? o.around : null;
  // axis / direction from probes (or the earlier estimate's)
  const probes = around ? [0.85, 1, 1.15].map((f) => f * Math.abs(around.readoutMs)) : [0.25, 0.625].map((f) => f * o.maxReadoutMs);
  const dirs: { axis: 'rows' | 'cols'; sign: number }[] = [];
  for (const axis of around ? [around.axis] : (['rows', 'cols'] as const)) for (const sign of around ? [Math.sign(around.readoutMs)] : [1, -1]) dirs.push({ axis, sign });
  // probe costs per direction, in probe order (NaN = not evaluated)
  const probeCost = dirs.map(() => probes.map(() => NaN));
  for (let d = 0; d < dirs.length; d++) probeCost[d][0] = evalH(dirs[d].axis, dirs[d].sign * probes[0]);
  // the outer probes of a direction are only needed when the first probes leave the direction open:
  // a direction that is clearly best at the first probe (≥ 10 % below every other one and below 0)
  // keeps being best further out (the other directions move away from the readout)
  const order = dirs.map((_, d) => d).sort((x, y) => probeCost[x][0] - probeCost[y][0] || x - y);
  const clear = dirs.length > 1 && probeCost[order[0]][0] < 0.9 * Math.min(cost0, probeCost[order[1]][0]);
  for (let d = 0; d < dirs.length; d++) {
    if (clear && d !== order[0]) continue;
    for (let k = 1; k < probes.length; k++) probeCost[d][k] = evalH(dirs[d].axis, dirs[d].sign * probes[k]);
  }
  for (let d = 0; d < dirs.length; d++)
    for (let k = 0; k < probes.length; k++) {
      const c = probeCost[d][k];
      if (!(c === c)) continue;
      if (c < dirCost) {
        dirCost = c;
        dir = dirs[d];
      }
      if (c < bestCost) {
        bestCost = c;
        best = { axis: dirs[d].axis, readoutMs: dirs[d].sign * probes[k] };
      }
    }
  if (dir && Number.isFinite(cost0)) {
    // 1-D minimisation of the magnitude along the chosen direction: bracket from the probes, then
    // safeguarded successive parabolic interpolation (Brent-like; bisection when the parabola's
    // vertex is outside the bracket or on an evaluated point), to `tolMs`
    const d = dir;
    const f = (x: number): number => {
      const c = evalH(d.axis, d.sign * x);
      if (c < bestCost) {
        bestCost = c;
        best = { axis: d.axis, readoutMs: d.sign * x };
      }
      return c;
    };
    const known = new Map<number, number>([[0, cost0]]);
    for (const c of curve) if (c.axis === d.axis && Math.sign(c.readoutMs) === d.sign) known.set(Math.abs(c.readoutMs), c.cost);
    let xs = [...known.keys()].sort((a, b) => a - b);
    let bi = xs.reduce((k, x, i) => (known.get(x)! < known.get(xs[k])! ? i : k), 0);
    for (let e = 0; e < 4 && bi === xs.length - 1 && xs[bi] < o.maxReadoutMs; e++) {
      const nx = around ? Math.min(o.maxReadoutMs, 1.3 * xs[bi] + 1) : o.maxReadoutMs;
      known.set(nx, f(nx));
      xs = [...known.keys()].sort((a, b) => a - b);
      bi = xs.reduce((k, x, i) => (known.get(x)! < known.get(xs[k])! ? i : k), 0);
    }
    let a = xs[Math.max(0, bi - 1)],
      b = xs[bi],
      c = xs[Math.min(xs.length - 1, bi + 1)];
    let fa = known.get(a)!,
      fb = known.get(b)!,
      fc = known.get(c)!;
    const tol = Math.max(0.5, 0.2 * o.stepMs);
    for (let it = 0; it < 8 && c - a > tol && a < b && b < c; it++) {
      // vertex of the parabola through (a, fa), (b, fb), (c, fc)
      const num = (b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa);
      const den = (b - a) * (fb - fc) - (b - c) * (fb - fa);
      let v = den !== 0 ? b - (0.5 * num) / den : NaN;
      if (!(v > a && v < c) || Math.abs(v - b) < 0.5 * tol) v = b - a > c - b ? 0.5 * (a + b) : 0.5 * (b + c);
      const fv = f(v);
      if (fv < fb) {
        if (v < b) {
          c = b;
          fc = fb;
        } else {
          a = b;
          fa = fb;
        }
        b = v;
        fb = fv;
      } else if (v < b) {
        a = v;
        fa = fv;
      } else {
        c = v;
        fc = fv;
      }
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
    numImages: info.numImages ?? 0,
    numObservations: info.numObservations ?? 0,
  };
}

/**
 * Estimate the rolling-shutter readout from registered images with their observations against
 * FIXED 3-D points (motion-only cost, `rsHypothesisCost`) — exact when the points are (as in tests),
 * a screen of axis / direction otherwise; the pipeline uses `RSBundleCost` (points free).
 */
export function estimateRollingShutter(images: readonly RSImageObservations[], opts: Partial<RollingShutterEstimateOptions> = {}): RollingShutterEstimate {
  let nObs = 0;
  for (const im of images) nObs += im.uv.length >> 1;
  if (images.length === 0 || nObs === 0) {
    const zero: RollingShutterModel = { axis: 'rows', readoutMs: 0 };
    return { model: zero, accepted: false, reason: 'no observations', cost0: NaN, costBest: NaN, improvement: 0, best: zero, curve: [], numImages: 0, numObservations: 0 };
  }
  return searchRollingShutter((m) => rsHypothesisCost(images, m, opts), opts, { numImages: images.length, numObservations: nObs });
}

/** Subset bundle adjustment scoring rolling-shutter hypotheses on a reconstruction (see the module comment). */
export class RSBundleCost {
  /** Free images (fastest rotation) and every image observing their points. */
  readonly freeIds: number[];
  readonly numObservations: number;
  private readonly poseIds: number[];
  private readonly obsPose: Uint32Array;
  private readonly obsPoint: Uint32Array;
  /** Original pixel observation and its image's camera / ω per observation. */
  private readonly uv0: Float64Array;
  private readonly obsImage: Int32Array;
  private readonly points0: Float64Array;
  private readonly cams: CameraIntrinsics[];
  private readonly imageCamera: Uint32Array;
  private readonly omega: (Vec3 | null)[];
  /** Camera-frame velocity per pose (translational term; null = rotation only). */
  private readonly velocity: (Float64Array | null)[];
  /** Inverse depth of each observation in the input reconstruction (0 = unknown). */
  private readonly obsInvDepth: Float64Array;
  private readonly poseFixed: Uint8Array;

  /**
   * @param rec reconstruction (poses / points read once; not modified)
   * @param freeIds registered images whose poses are free (e.g. the fastest-rotating ones)
   * @param original original (uncorrected) keypoints of an image
   * @param omegaOf camera angular velocity (rad/s) of an image at its frame time (null = unknown ⇒ uncorrected)
   * @param maxObsPerImage observations kept per free image (subsampled)
   * @param velocityOf camera-frame velocity (scene units / s) of an image's centre: adds the
   *   translational term with each observation's inverse depth in `rec` (null = rotation only)
   */
  constructor(
    rec: Reconstruction,
    freeIds: readonly number[],
    original: (id: number) => Float32Array | null,
    omegaOf: (id: number) => Vec3 | null,
    readonly opts: {
      maxObsPerImage?: number;
      maxTrack?: number;
      iterations?: number;
      lossScalePx?: number;
      refineFocal?: boolean;
      refinePrincipalPoint?: boolean;
      velocityOf?: (id: number) => ArrayLike<number> | null;
    } = {},
  ) {
    const maxObs = opts.maxObsPerImage ?? 300;
    const maxTrack = Math.max(2, opts.maxTrack ?? Infinity);
    const free = new Set(freeIds.filter((id) => rec.images.get(id)?.registered));
    this.freeIds = [...free];
    const pts: number[] = [];
    const ptIndex = new Map<number, number>();
    for (const id of this.freeIds) {
      const img = rec.images.get(id)!;
      const feats: number[] = [];
      for (let f = 0; f < img.point3D.length; f++) if (img.point3D[f] >= 0 && rec.points.has(img.point3D[f])) feats.push(img.point3D[f]);
      const stride = Math.max(1, Math.ceil(feats.length / maxObs));
      for (let k = 0; k < feats.length; k += stride)
        if (!ptIndex.has(feats[k])) {
          ptIndex.set(feats[k], pts.length);
          pts.push(feats[k]);
        }
    }
    const poseIndex = new Map<number, number>();
    const poseIds: number[] = [];
    const op: number[] = [],
      ox: number[] = [],
      uv: number[] = [],
      oi: number[] = [];
    const camIndex = new Map<number, number>();
    const cams: CameraIntrinsics[] = [];
    const imageCamera: number[] = [];
    const omega: (Vec3 | null)[] = [];
    const velocity: (Float64Array | null)[] = [];
    const fixed: number[] = [];
    const pointsArr = new Float64Array(3 * pts.length);
    pts.forEach((pid, j) => {
      const p = rec.points.get(pid)!;
      pointsArr.set(p.xyz, 3 * j);
      // long tracks: free images first, then every ⌈L/cap⌉-th of the others
      let track = p.track;
      if (track.length > maxTrack) {
        const inFree = track.filter((el) => free.has(el.image));
        const others = track.filter((el) => !free.has(el.image));
        const room = Math.max(0, maxTrack - inFree.length);
        const stride = others.length > room && room > 0 ? Math.ceil(others.length / room) : 1;
        track = inFree.slice(0, maxTrack).concat(room > 0 ? others.filter((_, k) => k % stride === 0).slice(0, room) : []);
      }
      for (const el of track) {
        const img = rec.images.get(el.image);
        if (!img || !img.registered) continue;
        const kp = original(el.image);
        if (!kp) continue;
        let pi = poseIndex.get(el.image);
        if (pi === undefined) {
          pi = poseIds.length;
          poseIndex.set(el.image, pi);
          poseIds.push(el.image);
          let ci = camIndex.get(img.cameraId);
          if (ci === undefined) {
            ci = cams.length;
            camIndex.set(img.cameraId, ci);
            cams.push({ ...rec.cameras.get(img.cameraId)! });
          }
          imageCamera.push(ci);
          const w = omegaOf(el.image);
          omega.push(w ? Float64Array.from(w) : null);
          const vel = opts.velocityOf?.(el.image);
          velocity.push(vel ? Float64Array.from(vel) : null);
          fixed.push(free.has(el.image) ? 0 : 1);
        }
        op.push(pi);
        ox.push(j);
        uv.push(kp[4 * el.feature], kp[4 * el.feature + 1]);
        oi.push(pi);
      }
    });
    this.poseIds = poseIds;
    this.obsPose = Uint32Array.from(op);
    this.obsPoint = Uint32Array.from(ox);
    this.uv0 = Float64Array.from(uv);
    this.obsImage = Int32Array.from(oi);
    this.points0 = pointsArr;
    this.cams = cams;
    this.imageCamera = Uint32Array.from(imageCamera);
    this.omega = omega;
    this.velocity = velocity;
    this.poseFixed = Uint8Array.from(fixed);
    this.numObservations = op.length;
    this.posesRec = poseIds.map((id) => {
      const p = rec.images.get(id)!.pose;
      return { R: Float64Array.from(p.R), t: Float64Array.from(p.t) };
    });
    this.obsInvDepth = new Float64Array(op.length);
    for (let o = 0; o < op.length; o++) {
      const P = this.posesRec[op[o]];
      const j = 3 * ox[o];
      const z = P.R[6] * pointsArr[j] + P.R[7] * pointsArr[j + 1] + P.R[8] * pointsArr[j + 2] + P.t[2];
      this.obsInvDepth[o] = z > 0 ? 1 / z : 0;
    }
    // gauge when every pose is free: first pose fixed, largest relative translation component of
    // the second pinned (scale)
    this.tMask = null;
    if (poseIds.length >= 2 && !fixed.some((x) => x === 1)) {
      this.poseFixed[0] = 1;
      const a = this.posesRec[0],
        b = this.posesRec[1];
      const w = [0, 1, 2].map((c) => a.R[c] * a.t[0] + a.R[3 + c] * a.t[1] + a.R[6 + c] * a.t[2]);
      const rel = [0, 1, 2].map((r) => Math.abs(b.t[r] - (b.R[3 * r] * w[0] + b.R[3 * r + 1] * w[1] + b.R[3 * r + 2] * w[2])));
      const ax = rel[0] >= rel[1] && rel[0] >= rel[2] ? 0 : rel[1] >= rel[2] ? 1 : 2;
      this.tMask = new Uint8Array(poseIds.length);
      this.tMask[1] = 1 << ax;
    }
  }

  private tMask: Uint8Array | null;

  private readonly posesRec: Pose[];

  /** Robust (Cauchy) cost after a short BA with all observations corrected for `model`. */
  cost(model: RollingShutterModel): number {
    const n = this.numObservations;
    if (n === 0) return NaN;
    const uv = new Float64Array(2 * n);
    const nrm = new Float64Array(2),
      d = new Float64Array(2);
    for (let o = 0; o < n; o++) {
      const u = this.uv0[2 * o],
        v = this.uv0[2 * o + 1];
      const pi = this.obsImage[o];
      const w = this.omega[pi];
      const cam = this.cams[this.imageCamera[pi]];
      if (model.readoutMs !== 0 && w && correctNormalized(cam, w, model, u, v, nrm, this.velocity[pi], this.obsInvDepth[o])) {
        distortNormalized(cam, nrm[0], nrm[1], d);
        uv[2 * o] = cam.f * d[0] + cam.cx;
        uv[2 * o + 1] = cam.f * d[1] + cam.cy;
      } else {
        uv[2 * o] = u;
        uv[2 * o + 1] = v;
      }
    }
    const refine: BACameraRefine = { focal: !!this.opts.refineFocal, principalPoint: !!this.opts.refinePrincipalPoint, distortion: false };
    const problem: BAProblem = {
      cameras: this.cams.map((c) => ({ ...c })),
      cameraRefine: this.cams.map(() => refine),
      poses: this.posesRec.map((p) => ({ R: Float64Array.from(p.R), t: Float64Array.from(p.t) })),
      imageCamera: this.imageCamera,
      poseFixed: this.poseFixed,
      translationFixedMask: this.tMask ?? undefined,
      points: Float64Array.from(this.points0),
      obsPose: this.obsPose,
      obsPoint: this.obsPoint,
      obsUV: uv,
    };
    const r = bundleAdjust(problem, {
      loss: 'cauchy',
      lossScalePx: this.opts.lossScalePx ?? 1,
      maxIterations: this.opts.iterations ?? 10,
      functionTolerance: 1e-5,
      parameterTolerance: 0,
    });
    return r.finalCost;
  }
}

/**
 * Rolling-shutter correction of a PINHOLE image (lens distortion already removed, e.g. a dense
 * view from `undistortImage`): the output is the global-shutter image at the frame stamp. Output
 * pixel x₀ samples the input where the camera, rotated to the readout time τ of that input line,
 * saw it: x_τ = exp(−[ω τ]×) x₀ (τ from the input position, 2 fixed-point iterations), bilinear;
 * pixels mapped outside the input are transparent black. The line coordinate is taken in the
 * image's own (pinhole) frame — the lens distortion moves the lines by a few pixels, i.e. < 0.1 ms
 * of readout. Images are the canonical orientation of `model` (same axis / sign), any scale.
 * `nu` (camera-frame velocity / the image's median depth, 1/s) adds the translational term for a
 * fronto-parallel plane at that depth: x_τ −= τ (ν_xy − x₀ ν_z) (exact only at that depth; the
 * pixels' own depths are what the dense stage computes).
 */
export function rsCorrectPinholeImage(img: RGBAImage, cam: CameraIntrinsics, omega: ArrayLike<number>, model: RollingShutterModel, nu: ArrayLike<number> | null = null): RGBAImage {
  const W = img.width,
    H = img.height;
  const out = new Uint8ClampedArray(W * H * 4);
  const src = img.data;
  if (model.readoutMs === 0) {
    out.set(src.subarray(0, W * H * 4));
    return { width: W, height: H, data: out };
  }
  const f = cam.f,
    cx = cam.cx,
    cy = cam.cy;
  const rows = model.axis === 'rows';
  const L = Math.max(1, (rows ? H : W) - 1);
  const ro = model.readoutMs / 1000;
  const wx = omega[0],
    wy = omega[1],
    wz = omega[2];
  for (let v = 0; v < H; v++) {
    for (let u = 0; u < W; u++) {
      const x0 = (u - cx) / f,
        y0 = (v - cy) / f;
      let su = u,
        sv = v;
      let ok = true;
      for (let it = 0; it < 3; it++) {
        const tau = (((rows ? sv : su) / L) - 0.5) * ro;
        // exp(−[ω τ]×) x₀ to first order in the small angle (|ω τ| ≲ 0.03 rad) + second-order term
        const ax = -wx * tau,
          ay = -wy * tau,
          az = -wz * tau;
        // v' = v + a × v + ½ a × (a × v)
        const cxv = ay * 1 - az * y0,
          cyv = az * x0 - ax * 1,
          czv = ax * y0 - ay * x0;
        const X = x0 + cxv + 0.5 * (ay * czv - az * cyv),
          Y = y0 + cyv + 0.5 * (az * cxv - ax * czv),
          Z = 1 + czv + 0.5 * (ax * cyv - ay * cxv);
        if (!(Z > 1e-6)) {
          ok = false;
          break;
        }
        let xs = X / Z,
          ys = Y / Z;
        if (nu) {
          // translational term at the image's median depth (ν = v_c / Z_med)
          xs -= tau * (nu[0] - x0 * nu[2]);
          ys -= tau * (nu[1] - y0 * nu[2]);
        }
        su = f * xs + cx;
        sv = f * ys + cy;
      }
      if (!ok || !(su >= 0 && sv >= 0 && su <= W - 1 && sv <= H - 1)) continue;
      const iu = Math.min(W - 2, Math.floor(su)),
        iv = Math.min(H - 2, Math.floor(sv));
      const fu = su - iu,
        fv = sv - iv;
      const o = 4 * (v * W + u);
      const a = 4 * (iv * W + iu),
        b = a + 4,
        c = a + 4 * W,
        d = c + 4;
      const w00 = (1 - fu) * (1 - fv),
        w10 = fu * (1 - fv),
        w01 = (1 - fu) * fv,
        w11 = fu * fv;
      for (let k = 0; k < 4; k++) out[o + k] = w00 * src[a + k] + w10 * src[b + k] + w01 * src[c + k] + w11 * src[d + k];
    }
  }
  return { width: W, height: H, data: out };
}
