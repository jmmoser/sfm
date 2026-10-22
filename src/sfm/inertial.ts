/**
 * Visual-inertial alignment of a finished reconstruction (ARCHITECTURE §5 "Refine"):
 *  (a) `estimateTimeOffset` — camera ↔ IMU clock offset by matching relative rotations of
 *      consecutive keyframes (SfM) with gyro-integrated rotations, jointly with a gyro bias;
 *  (a') `estimateTimeOffsetFromAcceleration` — the same offset from linear acceleration, for
 *      captures with too little angular acceleration (an offset error biases the scale);
 *  (b) `estimateGravity` — gravity direction in the reconstruction frame from the specific
 *      force rotated into the world by the vision-anchored attitude;
 *  (c) `estimateMetricScale` — metres per reconstruction unit by regressing band-limited
 *      second derivatives of the visual camera-centre trajectory on the same operator applied
 *      to double-integrated world-frame IMU acceleration;
 *  (d) `composeGravityAlignment` — rotation mapping the gravity direction to −Z.
 *
 * Conventions (ARCHITECTURE §2): keyframe rotations are camera-to-world `R_wc` (= pose Rᵀ),
 * device attitude R_wd = R_wc · R_cam_device, gyro dR_wd/dt = R_wd [ω]×, accel = R_wdᵀ a,
 * accelG = R_wdᵀ (a − g_world) with g_world = 9.81 · down. Time offset: an image stamped
 * t_cam was exposed at IMU time t_cam + offsetMs.
 *
 * Assumptions / limits: one constant gyro bias and constant accelerometer biases per capture;
 * the scale needs band-limited (0.3–3 Hz) acceleration ≥ 0.08 m/s² RMS (`excitationRms`), so
 * tripod-smooth motion comes out `reliable: false` (as does the gyro time offset of a smooth
 * constant-rate pan); segments are
 * split at keyframe gaps > 3 s and IMU gaps > 250 ms, and the gravity skips keyframe intervals
 * with an IMU gap (a capture continued later: one session, several segments). `estimateTimeOffset` costs ∝ keyframes ×
 * grid points (±250 ms at 2 ms, evaluated twice: 2000 keyframes ≈ 1 s per grid pass). When the
 * device hardly rotates, an uncompensated horizontal accelerometer bias b tilts gravity by ≈ b/g.
 *
 * Synthetic accuracy (extras-testutil: gyro noise 0.005 rad/s, bias ≈ 0.01; accel noise / bias
 * ≈ 0.05 m/s²; 2–4 Hz keyframes; 30 seeds): hand-held 0.4 m orbit scale rms 0.5 % (max 1.0 %),
 * 40 m walk rms 1.0 % (max 2.4 %); time offset max error < 1 ms; gravity rms 0.2° (orbit) /
 * 0.4° (walk); scale z-scores rms 0.8–0.95. `estimateInertialAlignment` ≈ 20–30 ms (Node).
 */
import type { ImuSample, Mat3, Vec3 } from '../core/types';
import { solveSPD } from '../math/decomp';
import { mat3ToRotvec, rotationBetweenVectors, rotvecToMat3 } from '../math/rotation';
import { median } from '../math/stats';

/** Keyframe input of the inertial estimators. */
export interface InertialKeyframe {
  /** Camera timestamp (ms). */
  t: number;
  /** Camera-to-world rotation in the reconstruction frame (= pose Rᵀ). */
  R_wc: Mat3;
  /** Camera centre in the reconstruction frame. */
  C: ArrayLike<number>;
  /** v_cam = R_cam_device · v_device. */
  R_cam_device: Mat3;
  /** Optional gravity (unit, DOWN) in the camera frame — fallback for `estimateGravity`. */
  gravityCam?: ArrayLike<number> | null;
}

const G0 = 9.81;

// ---------------------------------------------------------------------------------------
// Small quaternion / matrix helpers (allocation-free)
// ---------------------------------------------------------------------------------------

/** out[o..o+3] = exp(φ/2) as a unit quaternion [w, x, y, z]. */
function qexp(px: number, py: number, pz: number, out: Float64Array, o: number): void {
  const th = Math.sqrt(px * px + py * py + pz * pz);
  let s: number, c: number;
  if (th < 1e-8) {
    c = 1 - (th * th) / 8;
    s = 0.5 - (th * th) / 48;
  } else {
    c = Math.cos(0.5 * th);
    s = Math.sin(0.5 * th) / th;
  }
  out[o] = c;
  out[o + 1] = s * px;
  out[o + 2] = s * py;
  out[o + 3] = s * pz;
}

/** out[oo..] = a[ao..] ⊗ b[bo..], normalised. `out` may alias neither a nor b at the same offset. */
function qmul(a: Float64Array, ao: number, b: Float64Array, bo: number, out: Float64Array, oo: number): void {
  const aw = a[ao],
    ax = a[ao + 1],
    ay = a[ao + 2],
    az = a[ao + 3];
  const bw = b[bo],
    bx = b[bo + 1],
    by = b[bo + 2],
    bz = b[bo + 3];
  const w = aw * bw - ax * bx - ay * by - az * bz;
  const x = aw * bx + ax * bw + ay * bz - az * by;
  const y = aw * by - ax * bz + ay * bw + az * bx;
  const z = aw * bz + ax * by - ay * bx + az * bw;
  const inv = 1 / Math.sqrt(w * w + x * x + y * y + z * z);
  out[oo] = w * inv;
  out[oo + 1] = x * inv;
  out[oo + 2] = y * inv;
  out[oo + 3] = z * inv;
}

/** Rotation matrix of quaternion q[o..o+3] into out[oo..oo+8]. */
function q2m(q: Float64Array, o: number, out: Float64Array, oo: number): void {
  const w = q[o],
    x = q[o + 1],
    y = q[o + 2],
    z = q[o + 3];
  const xx = x * x,
    yy = y * y,
    zz = z * z,
    xy = x * y,
    xz = x * z,
    yz = y * z,
    wx = w * x,
    wy = w * y,
    wz = w * z;
  out[oo] = 1 - 2 * (yy + zz);
  out[oo + 1] = 2 * (xy - wz);
  out[oo + 2] = 2 * (xz + wy);
  out[oo + 3] = 2 * (xy + wz);
  out[oo + 4] = 1 - 2 * (xx + zz);
  out[oo + 5] = 2 * (yz - wx);
  out[oo + 6] = 2 * (xz - wy);
  out[oo + 7] = 2 * (yz + wx);
  out[oo + 8] = 1 - 2 * (xx + yy);
}

/** out = a · b (3×3 at offsets). */
function mm(a: Float64Array, ao: number, b: Float64Array, bo: number, out: Float64Array, oo: number): void {
  for (let i = 0; i < 3; i++) {
    const a0 = a[ao + 3 * i],
      a1 = a[ao + 3 * i + 1],
      a2 = a[ao + 3 * i + 2];
    out[oo + 3 * i] = a0 * b[bo] + a1 * b[bo + 3] + a2 * b[bo + 6];
    out[oo + 3 * i + 1] = a0 * b[bo + 1] + a1 * b[bo + 4] + a2 * b[bo + 7];
    out[oo + 3 * i + 2] = a0 * b[bo + 2] + a1 * b[bo + 5] + a2 * b[bo + 8];
  }
}

/** out = aᵀ · b (3×3 at offsets). */
function mtm(a: Float64Array, ao: number, b: Float64Array, bo: number, out: Float64Array, oo: number): void {
  for (let i = 0; i < 3; i++) {
    const a0 = a[ao + i],
      a1 = a[ao + 3 + i],
      a2 = a[ao + 6 + i];
    out[oo + 3 * i] = a0 * b[bo] + a1 * b[bo + 3] + a2 * b[bo + 6];
    out[oo + 3 * i + 1] = a0 * b[bo + 1] + a1 * b[bo + 4] + a2 * b[bo + 7];
    out[oo + 3 * i + 2] = a0 * b[bo + 2] + a1 * b[bo + 5] + a2 * b[bo + 8];
  }
}

/** Whether a 3-vector is present and all its components are finite. */
function finite3(v: ArrayLike<number> | null | undefined): boolean {
  return !!v && Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2]);
}

function sortKeyframes(kfs: readonly InertialKeyframe[]): InertialKeyframe[] {
  return kfs.filter((k) => Number.isFinite(k.t)).sort((a, b) => a.t - b.t);
}

// ---------------------------------------------------------------------------------------
// Gyro integration
// ---------------------------------------------------------------------------------------

/**
 * Cumulative gyro integration over a sample sequence. Q(t) = R_wd(t₀)ᵀ R_wd(t) (device
 * frame at the first sample), ω linear between samples and integrated with the exact
 * 2-term Magnus step φ = h(ωa+ωb)/2 + (h²/12) ωa×ωb (4th order for linear ω; e.g. Iserles
 * & Nørsett 1999). Also W(t) = ∫ R(Q(τ)) dτ (seconds) for the gyro-bias Jacobian
 * ∂ log(Q(a)ᵀQ(b)) / ∂b ≈ R(Q(b))ᵀ (W(b) − W(a)) (Forster et al. 2017, "On-manifold
 * preintegration", eq. 44 with Jr ≈ I). Outside the sample range ω is held constant.
 */
export class GyroIntegrator {
  /** Sample times (ms), strictly increasing. */
  readonly t: Float64Array;
  /** Angular velocity (rad/s), 3 per sample. */
  readonly w: Float64Array;
  /** Cumulative quaternions, 4 per sample. */
  readonly q: Float64Array;
  /** Cumulative ∫R dτ (s), 9 per sample. */
  readonly W: Float64Array;
  readonly n: number;
  private cursor = 0;
  private tq = new Float64Array(4);
  private tm = new Float64Array(9);
  private ti = new Float64Array(9);

  /**
   * @param samples IMU samples (any order; duplicates in time and samples with a non-finite
   *   time or gyro value are dropped — one NaN would otherwise poison every later attitude)
   * @param gyroBias optional constant gyro bias (rad/s, device frame) subtracted from ω
   */
  constructor(samples: readonly ImuSample[], gyroBias?: ArrayLike<number> | null) {
    const s = samples.filter((x) => Number.isFinite(x.t) && finite3(x.gyro)).sort((a, b) => a.t - b.t);
    const keep: ImuSample[] = [];
    for (const x of s) if (keep.length === 0 || x.t > keep[keep.length - 1].t) keep.push(x);
    const n = keep.length;
    this.n = n;
    this.t = new Float64Array(n);
    this.w = new Float64Array(3 * n);
    this.q = new Float64Array(4 * n);
    this.W = new Float64Array(9 * n);
    const bx = gyroBias ? gyroBias[0] : 0,
      by = gyroBias ? gyroBias[1] : 0,
      bz = gyroBias ? gyroBias[2] : 0;
    for (let i = 0; i < n; i++) {
      this.t[i] = keep[i].t;
      this.w[3 * i] = keep[i].gyro[0] - bx;
      this.w[3 * i + 1] = keep[i].gyro[1] - by;
      this.w[3 * i + 2] = keep[i].gyro[2] - bz;
    }
    if (n === 0) return;
    this.q[0] = 1;
    const Ra = new Float64Array(9),
      Rb = new Float64Array(9);
    q2m(this.q, 0, Ra, 0);
    for (let i = 0; i + 1 < n; i++) {
      const h = (this.t[i + 1] - this.t[i]) / 1000;
      this.step(i, h, this.w[3 * i + 3], this.w[3 * i + 4], this.w[3 * i + 5], this.q, 4 * i + 4);
      q2m(this.q, 4 * i + 4, Rb, 0);
      for (let k = 0; k < 9; k++) {
        this.W[9 * i + 9 + k] = this.W[9 * i + k] + 0.5 * h * (Ra[k] + Rb[k]);
        Ra[k] = Rb[k];
      }
    }
  }

  /** out[oo..] = q_i ⊗ exp(φ) with φ the Magnus step from sample i over h seconds to ω_end. */
  private step(i: number, h: number, bx: number, by: number, bz: number, out: Float64Array, oo: number): void {
    const ax = this.w[3 * i],
      ay = this.w[3 * i + 1],
      az = this.w[3 * i + 2];
    const c = (h * h) / 12;
    const px = 0.5 * h * (ax + bx) + c * (ay * bz - az * by);
    const py = 0.5 * h * (ay + by) + c * (az * bx - ax * bz);
    const pz = 0.5 * h * (az + bz) + c * (ax * by - ay * bx);
    qexp(px, py, pz, this.tq, 0);
    qmul(this.q, 4 * i, this.tq, 0, out, oo);
  }

  /** Index i with t[i] ≤ tm < t[i+1] (clamped to [0, n−2]); −1 if n < 2. */
  private interval(tm: number): number {
    const n = this.n;
    if (n < 2) return -1;
    let i = this.cursor;
    if (i > n - 2) i = n - 2;
    if (this.t[i] <= tm && (tm < this.t[i + 1] || i === n - 2)) return i;
    if (i + 1 <= n - 2 && this.t[i + 1] <= tm && tm < this.t[i + 2]) return (this.cursor = i + 1);
    let lo = 0,
      hi = n - 1;
    if (tm < this.t[0]) return (this.cursor = 0);
    if (tm >= this.t[n - 1]) return (this.cursor = n - 2);
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (this.t[mid] <= tm) lo = mid;
      else hi = mid;
    }
    return (this.cursor = lo);
  }

  /** Q(tm) (unit quaternion) into out[o..o+3]. */
  quatAt(tm: number, out: Float64Array, o = 0): void {
    const n = this.n;
    if (n === 0) {
      out[o] = 1;
      out[o + 1] = out[o + 2] = out[o + 3] = 0;
      return;
    }
    if (n === 1 || tm <= this.t[0] || tm >= this.t[n - 1]) {
      const i = n === 1 || tm <= this.t[0] ? 0 : n - 1;
      const h = (tm - this.t[i]) / 1000;
      qexp(h * this.w[3 * i], h * this.w[3 * i + 1], h * this.w[3 * i + 2], this.tq, 0);
      qmul(this.q, 4 * i, this.tq, 0, out, o);
      return;
    }
    const i = this.interval(tm);
    const h = (this.t[i + 1] - this.t[i]) / 1000;
    const u = (tm - this.t[i]) / 1000;
    const a = h > 0 ? u / h : 0;
    const bx = this.w[3 * i] + a * (this.w[3 * i + 3] - this.w[3 * i]);
    const by = this.w[3 * i + 1] + a * (this.w[3 * i + 4] - this.w[3 * i + 1]);
    const bz = this.w[3 * i + 2] + a * (this.w[3 * i + 5] - this.w[3 * i + 2]);
    this.step(i, u, bx, by, bz, out, o);
  }

  /** W(tm) = ∫ R(Q) dτ (s) into out[o..o+8] (trapezoid within the interval). */
  integralAt(tm: number, out: Float64Array, o = 0): void {
    const n = this.n;
    if (n === 0) {
      for (let k = 0; k < 9; k++) out[o + k] = 0;
      return;
    }
    this.quatAt(tm, this.tq, 0);
    q2m(this.tq, 0, this.tm, 0);
    let i: number;
    if (n === 1 || tm <= this.t[0]) i = 0;
    else if (tm >= this.t[n - 1]) i = n - 1;
    else i = this.interval(tm);
    const u = (tm - this.t[i]) / 1000;
    const Ri = this.ti;
    q2m(this.q, 4 * i, Ri, 0);
    for (let k = 0; k < 9; k++) out[o + k] = this.W[9 * i + k] + 0.5 * u * (Ri[k] + this.tm[k]);
  }

  /** Device rotation between two times: R_wd(ta)ᵀ R_wd(tb). */
  relativeRotation(ta: number, tb: number, out: Mat3 = new Float64Array(9)): Mat3 {
    const qa = new Float64Array(4),
      qb = new Float64Array(4);
    this.quatAt(ta, qa, 0);
    this.quatAt(tb, qb, 0);
    const A = new Float64Array(9),
      B = new Float64Array(9);
    q2m(qa, 0, A, 0);
    q2m(qb, 0, B, 0);
    mtm(A, 0, B, 0, out, 0);
    return out;
  }
}

// ---------------------------------------------------------------------------------------
// Vision-anchored attitude
// ---------------------------------------------------------------------------------------

/**
 * Device attitude R_wd(t) (IMU clock) in the reconstruction frame: exact at keyframes
 * (R_wc·R_cam_device), gyro-integrated in between with the end-point mismatch distributed
 * linearly in time, R_wd(t) = R_wd,k · G(τ_k, t) · Exp(α φ_k), α = (t − τ_k)/(τ_{k+1} − τ_k),
 * φ_k = log((R_wd,k G(τ_k, τ_{k+1}))ᵀ R_wd,k+1). Before/after the keyframes: gyro only.
 */
export class AttitudeInterpolator {
  readonly n: number;
  /** Keyframe IMU times τ_k = t_k + offset (ms). */
  readonly tau: Float64Array;
  /** R_wd at keyframes, 9 per keyframe. */
  readonly Rwd: Float64Array;
  private Rq: Float64Array;
  private phi: Float64Array;
  private cursor = 0;
  private q = new Float64Array(4);
  private m1 = new Float64Array(9);
  private m2 = new Float64Array(9);
  private m3 = new Float64Array(9);
  private v = new Float64Array(3);

  constructor(
    keyframes: readonly InertialKeyframe[],
    private gyro: GyroIntegrator,
    offsetMs: number,
  ) {
    const kf = sortKeyframes(keyframes);
    const n = kf.length;
    this.n = n;
    this.tau = new Float64Array(n);
    this.Rwd = new Float64Array(9 * n);
    this.Rq = new Float64Array(9 * n);
    this.phi = new Float64Array(3 * Math.max(0, n - 1));
    const R1 = new Float64Array(9);
    for (let k = 0; k < n; k++) {
      this.tau[k] = kf[k].t + offsetMs;
      R1.set(kf[k].R_wc);
      const Rcd = new Float64Array(9);
      Rcd.set(kf[k].R_cam_device);
      mm(R1, 0, Rcd, 0, this.Rwd, 9 * k);
      gyro.quatAt(this.tau[k], this.q, 0);
      q2m(this.q, 0, this.Rq, 9 * k);
    }
    for (let k = 0; k + 1 < n; k++) {
      // pred = Rwd_k · Rq_kᵀ Rq_{k+1};  φ = log(predᵀ Rwd_{k+1})
      mtm(this.Rq, 9 * k, this.Rq, 9 * k + 9, this.m1, 0);
      mm(this.Rwd, 9 * k, this.m1, 0, this.m2, 0);
      mtm(this.m2, 0, this.Rwd, 9 * k + 9, this.m3, 0);
      mat3ToRotvec(this.m3, this.v);
      this.phi[3 * k] = this.v[0];
      this.phi[3 * k + 1] = this.v[1];
      this.phi[3 * k + 2] = this.v[2];
    }
  }

  /** Angle (rad) of the vision–gyro mismatch over interval k (diagnostic). */
  mismatch(k: number): number {
    return Math.hypot(this.phi[3 * k], this.phi[3 * k + 1], this.phi[3 * k + 2]);
  }

  /** R_wd at IMU time tm (ms). */
  rotationAt(tm: number, out: Mat3 = new Float64Array(9)): Mat3 {
    const n = this.n;
    if (n === 0) {
      out.fill(0);
      out[0] = out[4] = out[8] = 1;
      return out;
    }
    let k: number;
    let alpha = 0;
    if (n === 1 || tm <= this.tau[0]) k = 0;
    else if (tm >= this.tau[n - 1]) k = n - 1;
    else {
      k = this.cursor < n - 1 && this.tau[this.cursor] <= tm && tm < this.tau[this.cursor + 1] ? this.cursor : -1;
      if (k < 0) {
        let lo = 0,
          hi = n - 1;
        while (hi - lo > 1) {
          const mid = (lo + hi) >> 1;
          if (this.tau[mid] <= tm) lo = mid;
          else hi = mid;
        }
        k = lo;
      }
      this.cursor = k;
      alpha = (tm - this.tau[k]) / (this.tau[k + 1] - this.tau[k]);
    }
    this.gyro.quatAt(tm, this.q, 0);
    q2m(this.q, 0, this.m1, 0);
    mtm(this.Rq, 9 * k, this.m1, 0, this.m2, 0); // G(τ_k, t)
    mm(this.Rwd, 9 * k, this.m2, 0, this.m3, 0);
    if (alpha > 0) {
      this.v[0] = alpha * this.phi[3 * k];
      this.v[1] = alpha * this.phi[3 * k + 1];
      this.v[2] = alpha * this.phi[3 * k + 2];
      rotvecToMat3(this.v, this.m1);
      mm(this.m3, 0, this.m1, 0, out, 0);
    } else out.set(this.m3);
    return out;
  }
}

// ---------------------------------------------------------------------------------------
// (a) Time offset
// ---------------------------------------------------------------------------------------

/** Options of `estimateTimeOffset`. */
export interface TimeOffsetOptions {
  /** Search half-range around `initialOffsetMs` (default 250). */
  searchRangeMs: number;
  /** Coarse grid step (default 2 ms). */
  coarseStepMs: number;
  /** Centre of the search (default 0). */
  initialOffsetMs: number;
  /** Keyframe pairs further apart are skipped (default 2000 ms). */
  maxPairGapMs: number;
  /** Minimum number of usable pairs (default 6). */
  minPairs: number;
  /** Estimate a constant gyro bias jointly (default true). */
  estimateBias: boolean;
  /** `reliable` requires stdMs below this (default 5). */
  maxStdMs: number;
}

/** Result of `estimateTimeOffset`. */
export interface TimeOffsetResult {
  /** IMU time = camera time + offsetMs. */
  offsetMs: number;
  /** 1-σ (ms) from the curvature of the residual sum of squares. */
  stdMs: number;
  /** 1 − min cost / median cost over the search grid, in [0, 1] (peak prominence). */
  quality: number;
  reliable: boolean;
  /** Gyro bias estimate (rad/s, device frame). */
  gyroBias: Vec3;
  /** RMS rotation residual per pair at the optimum (rad). */
  rmsResidualRad: number;
  numPairs: number;
  numInliers: number;
  /** Robust cost on the coarse grid. */
  curve: { offsetMs: Float64Array; cost: Float64Array };
}

const DEFAULT_TIME_OFFSET: TimeOffsetOptions = {
  searchRangeMs: 250,
  coarseStepMs: 2,
  initialOffsetMs: 0,
  maxPairGapMs: 2000,
  minPairs: 6,
  estimateBias: true,
  maxStdMs: 5,
};

class OffsetCost {
  readonly np: number;
  private a: Int32Array;
  private b: Int32Array;
  private V: Float64Array; // vision relative device rotations, 9 per pair
  private tk: Float64Array;
  private Rq: Float64Array;
  private Wk: Float64Array;
  private r: Float64Array;
  private J: Float64Array;
  private e: Float64Array;
  private q = new Float64Array(4);
  private m1 = new Float64Array(9);
  private m2 = new Float64Array(9);
  private v = new Float64Array(3);
  readonly bias = new Float64Array(3);
  private wts: Float64Array;

  constructor(
    kf: readonly InertialKeyframe[],
    private gyro: GyroIntegrator,
    maxGapMs: number,
    private estimateBias: boolean,
  ) {
    const n = kf.length;
    this.tk = new Float64Array(n);
    const Rwd = new Float64Array(9 * n);
    const R1 = new Float64Array(9),
      R2 = new Float64Array(9);
    for (let k = 0; k < n; k++) {
      this.tk[k] = kf[k].t;
      R1.set(kf[k].R_wc);
      R2.set(kf[k].R_cam_device);
      mm(R1, 0, R2, 0, Rwd, 9 * k);
    }
    const pa: number[] = [],
      pb: number[] = [];
    for (let k = 0; k + 1 < n; k++) {
      const dt = kf[k + 1].t - kf[k].t;
      if (dt > 0 && dt <= maxGapMs) {
        pa.push(k);
        pb.push(k + 1);
      }
    }
    this.np = pa.length;
    this.a = Int32Array.from(pa);
    this.b = Int32Array.from(pb);
    this.V = new Float64Array(9 * this.np);
    for (let p = 0; p < this.np; p++) mtm(Rwd, 9 * pa[p], Rwd, 9 * pb[p], this.V, 9 * p);
    this.Rq = new Float64Array(9 * n);
    this.Wk = new Float64Array(9 * n);
    this.r = new Float64Array(3 * this.np);
    this.J = new Float64Array(9 * this.np);
    this.e = new Float64Array(this.np);
    this.wts = new Float64Array(this.np);
  }

  /** Residuals r_p = log(V_pᵀ G_p(δ)) and Jacobians J_p for offset δ. */
  private prepare(delta: number): void {
    const n = this.tk.length;
    for (let k = 0; k < n; k++) {
      const tm = this.tk[k] + delta;
      this.gyro.quatAt(tm, this.q, 0);
      q2m(this.q, 0, this.Rq, 9 * k);
      this.gyro.integralAt(tm, this.Wk, 9 * k);
    }
    for (let p = 0; p < this.np; p++) {
      const ka = this.a[p],
        kb = this.b[p];
      mtm(this.Rq, 9 * ka, this.Rq, 9 * kb, this.m1, 0); // G
      mtm(this.V, 9 * p, this.m1, 0, this.m2, 0); // Vᵀ G
      mat3ToRotvec(this.m2, this.v);
      this.r[3 * p] = this.v[0];
      this.r[3 * p + 1] = this.v[1];
      this.r[3 * p + 2] = this.v[2];
      for (let k = 0; k < 9; k++) this.m1[k] = this.Wk[9 * kb + k] - this.Wk[9 * ka + k];
      mtm(this.Rq, 9 * kb, this.m1, 0, this.J, 9 * p);
    }
  }

  private residualNorms(): void {
    const b = this.bias;
    for (let p = 0; p < this.np; p++) {
      const J = this.J,
        o = 9 * p;
      const ex = this.r[3 * p] - (J[o] * b[0] + J[o + 1] * b[1] + J[o + 2] * b[2]);
      const ey = this.r[3 * p + 1] - (J[o + 3] * b[0] + J[o + 4] * b[1] + J[o + 5] * b[2]);
      const ez = this.r[3 * p + 2] - (J[o + 6] * b[0] + J[o + 7] * b[1] + J[o + 8] * b[2]);
      this.e[p] = Math.sqrt(ex * ex + ey * ey + ez * ez);
    }
  }

  private solveBias(): void {
    const A = new Float64Array(9),
      g = new Float64Array(3);
    for (let p = 0; p < this.np; p++) {
      const w = this.wts[p];
      if (!(w > 0)) continue;
      const o = 9 * p;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          let s = 0;
          for (let k = 0; k < 3; k++) s += this.J[o + 3 * k + i] * this.J[o + 3 * k + j];
          A[3 * i + j] += w * s;
        }
        let s = 0;
        for (let k = 0; k < 3; k++) s += this.J[o + 3 * k + i] * this.r[3 * p + k];
        g[i] += w * s;
      }
    }
    const tr = A[0] + A[4] + A[8];
    for (let i = 0; i < 3; i++) A[4 * i] += 1e-9 * tr + 1e-15;
    const b = solveSPD(A, 3, g);
    if (b) this.bias.set(b);
    else this.bias.fill(0);
  }

  /** Robust (Cauchy, scale c) cost at offset δ, bias re-estimated by 2 IRLS passes. */
  cost(delta: number, c: number): number {
    this.prepare(delta);
    this.bias.fill(0);
    if (this.estimateBias) {
      this.residualNorms();
      for (let pass = 0; pass < 2; pass++) {
        for (let p = 0; p < this.np; p++) this.wts[p] = 1 / (1 + (this.e[p] * this.e[p]) / (c * c));
        this.solveBias();
        this.residualNorms();
      }
    } else this.residualNorms();
    let s = 0;
    const c2 = c * c;
    for (let p = 0; p < this.np; p++) s += Number.isFinite(c2) ? c2 * Math.log1p((this.e[p] * this.e[p]) / c2) : this.e[p] * this.e[p];
    return s;
  }

  /** Least-squares sum of squared residual norms over pairs with mask, bias re-estimated on them. */
  sse(delta: number, mask: Uint8Array): number {
    this.prepare(delta);
    this.bias.fill(0);
    if (this.estimateBias) {
      for (let p = 0; p < this.np; p++) this.wts[p] = mask[p];
      this.solveBias();
    }
    this.residualNorms();
    let s = 0;
    for (let p = 0; p < this.np; p++) if (mask[p]) s += this.e[p] * this.e[p];
    return s;
  }

  residuals(): Float64Array {
    return this.e;
  }
}

/** Golden-section minimisation of f on [a, b] to tolerance tol. */
function goldenSection(f: (x: number) => number, a: number, b: number, tol: number): number {
  const gr = (Math.sqrt(5) - 1) / 2;
  let c = b - gr * (b - a),
    d = a + gr * (b - a);
  let fc = f(c),
    fd = f(d);
  while (b - a > tol) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - gr * (b - a);
      fc = f(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + gr * (b - a);
      fd = f(d);
    }
  }
  return 0.5 * (a + b);
}

/**
 * Camera–IMU time offset: minimises the robust (Cauchy) misfit between visual relative
 * device rotations of consecutive keyframes V = R_wd,kᵀ R_wd,k+1 and gyro-integrated
 * rotations G(t_k + δ, t_{k+1} + δ), with a constant gyro bias re-estimated for every δ
 * (residual log(VᵀG) ≈ J b). Coarse grid over ±`searchRangeMs`, then golden-section
 * refinement within ±1 grid step. Only angular ACCELERATION makes δ observable (constant-rate
 * rotation is invariant to a shift), which `stdMs` reflects. Returns null with fewer than
 * `minPairs` pairs or no IMU samples.
 */
export function estimateTimeOffset(
  keyframes: readonly InertialKeyframe[],
  imu: readonly ImuSample[],
  options: Partial<TimeOffsetOptions> = {},
): TimeOffsetResult | null {
  const o = { ...DEFAULT_TIME_OFFSET, ...options };
  const kf = sortKeyframes(keyframes);
  const gyro = new GyroIntegrator(imu);
  if (gyro.n < 2 || kf.length < 2) return null;
  // Keep keyframes whose search window lies inside the IMU coverage.
  const lo = gyro.t[0],
    hi = gyro.t[gyro.n - 1];
  const inRange = kf.filter((k) => k.t + o.initialOffsetMs - o.searchRangeMs >= lo && k.t + o.initialOffsetMs + o.searchRangeMs <= hi);
  const cf = new OffsetCost(inRange, gyro, o.maxPairGapMs, o.estimateBias);
  if (cf.np < o.minPairs) return null;

  // Robust scale from the residuals at the initial offset (LS).
  cf.cost(o.initialOffsetMs, Infinity);
  let c = Math.max(2 * median(cf.residuals()), 2e-4);

  const step = Math.max(o.coarseStepMs, 0.05);
  const nGrid = Math.floor((2 * o.searchRangeMs) / step) + 1;
  const offs = new Float64Array(nGrid),
    costs = new Float64Array(nGrid);
  let bi = 0;
  for (let i = 0; i < nGrid; i++) {
    offs[i] = o.initialOffsetMs - o.searchRangeMs + i * step;
    costs[i] = cf.cost(offs[i], c);
    if (costs[i] < costs[bi]) bi = i;
  }
  // Re-scale the robust kernel at the coarse optimum, refine locally.
  cf.cost(offs[bi], c);
  c = Math.max(2 * median(cf.residuals()), 2e-4);
  let a = offs[Math.max(0, bi - 2)],
    b = offs[Math.min(nGrid - 1, bi + 2)];
  let best = offs[bi],
    bestCost = cf.cost(best, c);
  for (let x = a; x <= b + 1e-9; x += step / 2) {
    const v = cf.cost(x, c);
    if (v < bestCost) {
      bestCost = v;
      best = x;
    }
  }
  a = Math.max(offs[0], best - step / 2);
  b = Math.min(offs[nGrid - 1], best + step / 2);
  const ref = goldenSection((x) => cf.cost(x, c), a, b, 0.005);
  if (cf.cost(ref, c) < bestCost) best = ref;
  bestCost = cf.cost(best, c);
  const gyroBias = Float64Array.from(cf.bias);

  // Inliers and curvature of the LS cost.
  const e = cf.residuals();
  const mask = new Uint8Array(cf.np);
  let nIn = 0;
  for (let p = 0; p < cf.np; p++) {
    mask[p] = e[p] < 3 * c ? 1 : 0;
    nIn += mask[p];
  }
  const h = 1;
  const s0 = cf.sse(best, mask);
  const sp = cf.sse(best + h, mask);
  const sm = cf.sse(best - h, mask);
  const curv = (sp - 2 * s0 + sm) / (h * h);
  const dof = Math.max(1, 3 * nIn - (o.estimateBias ? 4 : 1));
  const sigma2 = s0 / dof;
  const stdMs = curv > 0 ? Math.sqrt((2 * sigma2) / curv) : Infinity;
  // prominence under ONE kernel: the grid `costs` used the scale at the initial offset, bestCost the final one
  const costsC = new Float64Array(nGrid);
  for (let i = 0; i < nGrid; i++) costsC[i] = cf.cost(offs[i], c);
  const medCost = median(costsC);
  const quality = medCost > 0 ? Math.min(1, Math.max(0, 1 - bestCost / medCost)) : 0;
  const atEdge = Math.abs(best - o.initialOffsetMs) > o.searchRangeMs - 2 * step;
  cf.sse(best, mask); // leave state at the optimum
  return {
    offsetMs: best,
    stdMs,
    quality,
    reliable: !atEdge && stdMs <= o.maxStdMs && nIn >= o.minPairs && quality > 0.2,
    gyroBias,
    rmsResidualRad: Math.sqrt(s0 / Math.max(1, nIn)),
    numPairs: cf.np,
    numInliers: nIn,
    curve: { offsetMs: offs, cost: costs },
  };
}

// ---------------------------------------------------------------------------------------
// (b) Gravity
// ---------------------------------------------------------------------------------------

/** Options of `estimateGravity`. */
export interface GravityOptions {
  /** Camera–IMU offset (ms). */
  offsetMs: number;
  /** Nominal gravity (default 9.81). */
  g0: number;
  /** Width of the |f| ≈ g weighting (m/s², default 0.5). */
  magnitudeSigma: number;
  /** Minimum IMU samples (default 20). */
  minSamples: number;
  /** Correlation time of hand motion used for the uncertainty (s, default 0.5). */
  correlationS: number;
  /** Gyro bias (rad/s) to remove before integration, e.g. `TimeOffsetResult.gyroBias`. */
  gyroBias: ArrayLike<number> | null;
  /**
   * IMU gaps (ms) that break the attitude between two keyframes (default 250, as the scale's
   * segments): the samples inside such an interval are not used, except within 100 ms of its
   * keyframes.
   */
  maxImuGapMs: number;
}

/** Gravity estimate in the reconstruction frame. */
export interface GravityResult {
  /** Unit vector pointing DOWN (reconstruction frame). */
  down: Vec3;
  /** Estimated |g| (m/s²) (from the projected specific force). */
  magnitude: number;
  /** 1-σ direction uncertainty (rad). */
  stdRad: number;
  numSamples: number;
  reliable: boolean;
  source: 'accelG' | 'gravityCam';
}

const DEFAULT_GRAVITY: GravityOptions = { offsetMs: 0, g0: G0, magnitudeSigma: 0.5, minSamples: 20, correlationS: 0.5, gyroBias: null, maxImuGapMs: 250 };

/**
 * Gravity direction: weighted mean of the world-frame specific force f = R_wd(t)·accelG over
 * all IMU samples in the keyframe span (the motion acceleration integrates to Δv/T ≈ 0 over a
 * capture), weights 1/(1 + ((|f| − g)/σ)²) favour quasi-static samples. down = −mean/|mean|.
 * Samples between two keyframes whose IMU has a gap (> `maxImuGapMs`: a capture continued later,
 * EngineOptions.extend) are skipped except within 100 ms of the keyframes: the gyro does not
 * carry the attitude across the gap, and interpolating the unobserved rotation over the interval
 * tilted gravity by degrees (docs/research/extend-scan.md: 5.4° → 0.07° on a synthetic e2e
 * run). Falls back to the mean of R_wc·gravityCam over keyframes when no accelG is available.
 */
export function estimateGravity(
  keyframes: readonly InertialKeyframe[],
  imu: readonly ImuSample[],
  options: Partial<GravityOptions> = {},
): GravityResult | null {
  const o = { ...DEFAULT_GRAVITY, ...options };
  const kf = sortKeyframes(keyframes);
  if (kf.length === 0) return null;
  const gyro = new GyroIntegrator(imu, o.gyroBias);
  const t0 = kf[0].t + o.offsetMs - 100,
    t1 = kf[kf.length - 1].t + o.offsetMs + 100;
  let cnt = 0;
  if (gyro.n >= 2) {
    const att = new AttitudeInterpolator(kf, gyro, o.offsetMs);
    // keyframe intervals the IMU does not cover continuously
    const tau = att.tau;
    const st = gyro.t;
    const broken = new Uint8Array(Math.max(0, kf.length - 1));
    for (let k = 0, i = 0; k + 1 < kf.length; k++) {
      while (i + 1 < st.length && st[i + 1] <= tau[k]) i++;
      for (let j = i; j + 1 < st.length && st[j] < tau[k + 1]; j++) {
        if (st[j + 1] - st[j] > o.maxImuGapMs) {
          broken[k] = 1;
          break;
        }
      }
    }
    const anyBroken = broken.includes(1);
    const usable = (tm: number): boolean => {
      if (!anyBroken || !(tm > tau[0] && tm < tau[tau.length - 1])) return true;
      let lo = 0,
        hi = tau.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (tau[mid] <= tm) lo = mid;
        else hi = mid;
      }
      return !(broken[lo] && tm > tau[lo] + 100 && tm < tau[lo + 1] - 100);
    };
    const R = new Float64Array(9);
    let sx = 0,
      sy = 0,
      sz = 0,
      sw = 0;
    const fx: number[] = [],
      fy: number[] = [],
      fz: number[] = [],
      fw: number[] = [];
    for (const s of imu) {
      if (!s.accelG || !(s.t >= t0 && s.t <= t1) || !usable(s.t)) continue;
      att.rotationAt(s.t, R);
      const a = s.accelG;
      const x = R[0] * a[0] + R[1] * a[1] + R[2] * a[2];
      const y = R[3] * a[0] + R[4] * a[1] + R[5] * a[2];
      const z = R[6] * a[0] + R[7] * a[1] + R[8] * a[2];
      const m = Math.sqrt(x * x + y * y + z * z);
      if (!Number.isFinite(m)) continue;
      const d = (m - o.g0) / o.magnitudeSigma;
      const w = 1 / (1 + d * d);
      sx += w * x;
      sy += w * y;
      sz += w * z;
      sw += w;
      fx.push(x);
      fy.push(y);
      fz.push(z);
      fw.push(w);
      cnt++;
    }
    if (cnt >= o.minSamples && sw > 0) {
      const mx = sx / sw,
        my = sy / sw,
        mz = sz / sw;
      const mn = Math.sqrt(mx * mx + my * my + mz * mz);
      if (mn > 0) {
        const ux = mx / mn,
          uy = my / mn,
          uz = mz / mn; // up
        let sp = 0,
          sperp = 0,
          sw2 = 0;
        for (let i = 0; i < cnt; i++) {
          const pr = fx[i] * ux + fy[i] * uy + fz[i] * uz;
          const px = fx[i] - pr * ux,
            py = fy[i] - pr * uy,
            pz = fz[i] - pr * uz;
          sp += fw[i] * pr;
          sperp += fw[i] * (px * px + py * py + pz * pz);
          sw2 += fw[i];
        }
        const mag = sp / sw2;
        const durS = Math.max(1e-3, (t1 - t0) / 1000);
        const neff = Math.max(1, durS / o.correlationS);
        const stdRad = Math.max(Math.sqrt(sperp / sw2 / 2 / neff) / Math.max(mag, 1e-6), 1e-3);
        return {
          down: Float64Array.of(-ux, -uy, -uz),
          magnitude: mag,
          stdRad,
          numSamples: cnt,
          reliable: stdRad < (2 * Math.PI) / 180 && Math.abs(mag - o.g0) < 1,
          source: 'accelG',
        };
      }
    }
  }
  // Fallback: per-keyframe gravity in the camera frame. Its σ treats the keyframes as independent
  // (the DeviceOrientation attitude errors are time-correlated), so it is optimistic.
  let gx = 0,
    gy = 0,
    gz = 0,
    ng = 0;
  const dirs: number[] = [];
  for (const k of kf) {
    const g = k.gravityCam;
    if (!g) continue;
    const R = k.R_wc;
    const x = R[0] * g[0] + R[1] * g[1] + R[2] * g[2];
    const y = R[3] * g[0] + R[4] * g[1] + R[5] * g[2];
    const z = R[6] * g[0] + R[7] * g[1] + R[8] * g[2];
    const m = Math.hypot(x, y, z);
    if (!(m > 0)) continue;
    gx += x / m;
    gy += y / m;
    gz += z / m;
    dirs.push(x / m, y / m, z / m);
    ng++;
  }
  if (ng === 0) return null;
  const m = Math.hypot(gx, gy, gz);
  if (!(m > 0)) return null;
  const down = Float64Array.of(gx / m, gy / m, gz / m);
  let s2 = 0;
  for (let i = 0; i < ng; i++) {
    const c = dirs[3 * i] * down[0] + dirs[3 * i + 1] * down[1] + dirs[3 * i + 2] * down[2];
    s2 += 1 - c * c;
  }
  const stdRad = Math.max(Math.sqrt(s2 / ng / Math.max(1, ng)), 1e-3);
  return { down, magnitude: o.g0, stdRad, numSamples: ng, reliable: ng >= 3 && stdRad < (2 * Math.PI) / 180, source: 'gravityCam' };
}

/** (d) Rotation R with R·down = (0, 0, −1) (minimal rotation; yaw left free). */
export function composeGravityAlignment(down: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  return rotationBetweenVectors(down, [0, 0, -1], out);
}

// ---------------------------------------------------------------------------------------
// Signal processing helpers for (c)
// ---------------------------------------------------------------------------------------

/**
 * Second derivative of the cubic smoothing spline (Reinsch 1967) through (tᵢ, yᵢ), evaluated
 * on a grid. Minimises Σ (yᵢ − f(tᵢ))² + λ ∫ f''², natural end conditions; the second
 * derivatives γ at the interior knots solve (R + λ QᵀQ) γ = Qᵀ y (pentadiagonal, banded
 * Cholesky factored once, so many signals can share the operator). f'' is piecewise linear
 * between knots. λ = 0 gives the interpolating natural spline. Equivalent low-pass for
 * uniform spacing h: 1/(1 + λ h ω⁴) ⇒ λ = 1/(h ω_c⁴) for a cutoff ω_c.
 */
export class SplineSecondDerivative {
  readonly n: number;
  readonly m: number;
  private h: Float64Array;
  private L0: Float64Array;
  private L1: Float64Array;
  private L2: Float64Array;
  private gi: Int32Array;
  private ga: Float64Array;
  private rhs: Float64Array;
  private gam: Float64Array;

  /**
   * @param t knot times (strictly increasing, seconds)
   * @param lambda smoothing weight (≥ 0)
   * @param grid evaluation times (within [t₀, t_{n−1}], clamped otherwise)
   */
  constructor(t: Float64Array, lambda: number, grid: Float64Array) {
    const n = t.length;
    this.n = n;
    this.m = grid.length;
    this.h = new Float64Array(Math.max(0, n - 1));
    for (let i = 0; i + 1 < n; i++) this.h[i] = t[i + 1] - t[i];
    const ni = Math.max(0, n - 2);
    this.L0 = new Float64Array(ni);
    this.L1 = new Float64Array(ni);
    this.L2 = new Float64Array(ni);
    this.rhs = new Float64Array(ni);
    this.gam = new Float64Array(n);
    const h = this.h;
    // A = R + λ QᵀQ, bands a0 (diag), a1 (j, j+1), a2 (j, j+2), interior index j ↔ knot j+1.
    const a0 = new Float64Array(ni),
      a1 = new Float64Array(ni),
      a2 = new Float64Array(ni);
    for (let j = 0; j < ni; j++) {
      const i0 = 1 / h[j],
        i1 = 1 / h[j + 1];
      a0[j] = (h[j] + h[j + 1]) / 3 + lambda * (i0 * i0 + (i0 + i1) * (i0 + i1) + i1 * i1);
      if (j + 1 < ni) {
        const i2 = 1 / h[j + 2];
        a1[j] = h[j + 1] / 6 + lambda * (-(i0 + i1) * i1 - i1 * (i1 + i2));
      }
      if (j + 2 < ni) a2[j] = lambda * i1 * (1 / h[j + 2]);
    }
    // Banded Cholesky: L[j][j] = L0[j], L[j][j-1] = L1[j], L[j][j-2] = L2[j].
    for (let j = 0; j < ni; j++) {
      const l2 = j >= 2 ? a2[j - 2] / this.L0[j - 2] : 0;
      const l1 = j >= 1 ? (a1[j - 1] - (j >= 2 ? l2 * this.L1[j - 1] : 0)) / this.L0[j - 1] : 0;
      const d = a0[j] - l1 * l1 - l2 * l2;
      this.L2[j] = l2;
      this.L1[j] = l1;
      this.L0[j] = Math.sqrt(Math.max(d, 1e-300));
    }
    // Grid → interval index and weight.
    this.gi = new Int32Array(this.m);
    this.ga = new Float64Array(this.m);
    let k = 0;
    for (let g = 0; g < this.m; g++) {
      const x = grid[g];
      if (n < 2) {
        this.gi[g] = 0;
        continue;
      }
      if (x <= t[0]) {
        this.gi[g] = 0;
        this.ga[g] = 0;
        continue;
      }
      if (x >= t[n - 1]) {
        this.gi[g] = n - 2;
        this.ga[g] = 1;
        continue;
      }
      if (k > n - 2 || t[k] > x) k = 0;
      while (k < n - 2 && t[k + 1] <= x) k++;
      this.gi[g] = k;
      this.ga[g] = (x - t[k]) / h[k];
    }
  }

  /** f'' on the grid for data y (length n) into out (length m). */
  apply(y: ArrayLike<number>, out: Float64Array = new Float64Array(this.m)): Float64Array {
    const n = this.n,
      ni = Math.max(0, n - 2);
    const h = this.h,
      b = this.rhs,
      g = this.gam;
    if (n < 3) {
      out.fill(0);
      return out;
    }
    for (let j = 0; j < ni; j++) b[j] = (y[j + 2] - y[j + 1]) / h[j + 1] - (y[j + 1] - y[j]) / h[j];
    // forward
    for (let j = 0; j < ni; j++) {
      let s = b[j];
      if (j >= 1) s -= this.L1[j] * b[j - 1];
      if (j >= 2) s -= this.L2[j] * b[j - 2];
      b[j] = s / this.L0[j];
    }
    // backward
    for (let j = ni - 1; j >= 0; j--) {
      let s = b[j];
      if (j + 1 < ni) s -= this.L1[j + 1] * b[j + 1];
      if (j + 2 < ni) s -= this.L2[j + 2] * b[j + 2];
      b[j] = s / this.L0[j];
    }
    g[0] = 0;
    g[n - 1] = 0;
    for (let j = 0; j < ni; j++) g[j + 1] = b[j];
    for (let q = 0; q < this.m; q++) {
      const k = this.gi[q],
        a = this.ga[q];
      out[q] = g[k] + a * (g[k + 1] - g[k]);
    }
    return out;
  }
}

/** Second-order Butterworth biquad (RBJ cookbook, Q = 1/√2), normalised coefficients. */
interface Biquad {
  b0: number;
  b1: number;
  b2: number;
  a1: number;
  a2: number;
}

function butter2(kind: 'low' | 'high', f0: number, fs: number): Biquad {
  const w0 = (2 * Math.PI * f0) / fs;
  const cw = Math.cos(w0),
    alpha = Math.sin(w0) / Math.SQRT2;
  const a0 = 1 + alpha;
  if (kind === 'low') return { b0: (1 - cw) / 2 / a0, b1: (1 - cw) / a0, b2: (1 - cw) / 2 / a0, a1: (-2 * cw) / a0, a2: (1 - alpha) / a0 };
  return { b0: (1 + cw) / 2 / a0, b1: -(1 + cw) / a0, b2: (1 + cw) / 2 / a0, a1: (-2 * cw) / a0, a2: (1 - alpha) / a0 };
}

/** In-place biquad (DF2T) with steady-state initial conditions for x[0]. */
function biquadRun(f: Biquad, x: Float64Array, n: number): void {
  const G = (f.b0 + f.b1 + f.b2) / (1 + f.a1 + f.a2);
  const x0 = x[0];
  let z1 = G * x0 - f.b0 * x0;
  let z2 = f.b2 * x0 - f.a2 * G * x0;
  for (let i = 0; i < n; i++) {
    const xi = x[i];
    const yi = f.b0 * xi + z1;
    z1 = f.b1 * xi - f.a1 * yi + z2;
    z2 = f.b2 * xi - f.a2 * yi;
    x[i] = yi;
  }
}

/**
 * Zero-phase Butterworth band-pass (2nd-order high-pass at `lowHz` and 2nd-order low-pass at
 * `highHz`, applied forward and backward ⇒ 4th-order magnitude, zero phase), with odd-
 * reflection padding as in scipy's filtfilt. Disabled edges: lowHz ≤ 0, highHz ≥ 0.45·fs.
 */
export function filtfiltBandpass(x: Float64Array, fs: number, lowHz: number, highHz: number, out: Float64Array = new Float64Array(x.length)): Float64Array {
  const n = x.length;
  const filters: Biquad[] = [];
  if (lowHz > 0) filters.push(butter2('high', lowHz, fs));
  if (highHz > 0 && highHz < 0.45 * fs) filters.push(butter2('low', highHz, fs));
  if (filters.length === 0 || n < 2) {
    out.set(x);
    return out;
  }
  const pad = Math.min(n - 1, Math.ceil((3 * fs) / Math.max(lowHz > 0 ? lowHz : highHz, 1e-3)));
  const N = n + 2 * pad;
  const buf = new Float64Array(N);
  for (let i = 0; i < pad; i++) buf[i] = 2 * x[0] - x[pad - i];
  buf.set(x, pad);
  for (let i = 0; i < pad; i++) buf[pad + n + i] = 2 * x[n - 1] - x[n - 2 - i];
  for (const f of filters) biquadRun(f, buf, N);
  buf.reverse();
  for (const f of filters) biquadRun(f, buf, N);
  buf.reverse();
  for (let i = 0; i < n; i++) out[i] = buf[pad + i];
  return out;
}

// ---------------------------------------------------------------------------------------
// (c) Metric scale
// ---------------------------------------------------------------------------------------

/** Options of `estimateMetricScale`. */
export interface MetricScaleOptions {
  /** Camera–IMU offset (ms). */
  offsetMs: number;
  /** Gravity (DOWN, reconstruction frame); required when only accelG is available. */
  gravity: { down: ArrayLike<number>; magnitude?: number } | null;
  /** Use 'accel' (gravity-free) or 'accelG' − gravity; 'auto' prefers accel (default). */
  source: 'auto' | 'accel' | 'accelG';
  /** Band-pass (Hz) applied to both acceleration signals (defaults 0.3 / 3). */
  bandLowHz: number;
  bandHighHz: number;
  /** Resampling rate of the band-limited signals (default 20 Hz). */
  gridHz: number;
  /** Seconds trimmed at both ends of each segment (default 0.5). */
  edgeTrimS: number;
  /**
   * Also estimate a constant device-frame accelerometer bias with a Gaussian prior (default
   * true). Important for orbits: a device-frame bias along the optical axis rotates with the
   * centripetal acceleration and would otherwise masquerade as a scale change.
   */
  estimateDeviceBias: boolean;
  /** Prior σ (m/s²) of the device-frame bias (default 0.05). */
  deviceBiasSigma: number;
  /** Max lag (s) of the Newey–West uncertainty estimate (default 3). */
  hacLagS: number;
  /** Gyro bias (rad/s) to remove before integration, e.g. `TimeOffsetResult.gyroBias`. */
  gyroBias: ArrayLike<number> | null;
  /** Keyframe gaps (ms) above which the trajectory is split into segments (default 3000). */
  maxGapMs: number;
  /** IMU gaps (ms) above which segments are split (default 250). */
  maxImuGapMs: number;
  /** Minimum segment length (s) and keyframes (defaults 3 s, 6). */
  minSegmentS: number;
  minSegmentKeyframes: number;
  /** Reliability thresholds: band-limited IMU acceleration RMS (m/s², 0.08), rel. σ (0.1), R² (0.3). */
  minExcitation: number;
  maxRelUncertainty: number;
  minR2: number;
}

/** Result of `estimateMetricScale`. */
export interface MetricScaleResult {
  /** Metres per reconstruction unit (multiply reconstruction coordinates by this). */
  scale: number;
  /** 1-σ relative uncertainty of `scale`. */
  relUncertainty: number;
  reliable: boolean;
  /** Why it is unreliable ('' when reliable). */
  reason: string;
  /** RMS of the band-limited IMU acceleration used (m/s²) — the excitation. */
  excitationRms: number;
  /** Explained variance of the visual signal. */
  r2: number;
  /** Constant world-frame acceleration bias (m/s², reconstruction axes). */
  biasWorld: Vec3;
  /** Device-frame accelerometer bias (m/s²) when estimated. */
  biasDevice: Vec3 | null;
  numKeyframes: number;
  numSegments: number;
  /** Rows of the regression and their effective (decorrelated) count. */
  numRows: number;
  numEffective: number;
  durationS: number;
  source: 'accel' | 'accelG';
}

const DEFAULT_SCALE: MetricScaleOptions = {
  offsetMs: 0,
  gravity: null,
  source: 'auto',
  bandLowHz: 0.3,
  bandHighHz: 3,
  gridHz: 20,
  edgeTrimS: 0.5,
  estimateDeviceBias: true,
  deviceBiasSigma: 0.05,
  hacLagS: 3,
  gyroBias: null,
  maxGapMs: 3000,
  maxImuGapMs: 250,
  minSegmentS: 3,
  minSegmentKeyframes: 6,
  minExcitation: 0.08,
  maxRelUncertainty: 0.1,
  minR2: 0.3,
};

/**
 * Empirical factor on the HAC σ: on synthetic hand-held orbits and walks (30 seeds each) the
 * plain Newey–West σ was 1.2–1.35× too small (rms z-score), as usual for HAC with strongly
 * coloured residuals and a short bandwidth.
 */
const HAC_CALIBRATION = 1.3;

/** Number of filtered channels per row block: C(3), P(3), Q(1), Dm(9). */
const NCH = 16;

/**
 * Double integral of piecewise-linear samples a(s) (dim channels, zero initial position and
 * velocity at s₀), evaluated at query times (seconds, sorted, within [s₀, s_last]).
 */
function doubleIntegrateAt(s: Float64Array, a: Float64Array, dim: number, q: Float64Array, out: Float64Array): void {
  const n = s.length;
  const v = new Float64Array(dim),
    p = new Float64Array(dim);
  let i = 0;
  for (let j = 0; j < q.length; j++) {
    const tq = q[j];
    while (i < n - 2 && s[i + 1] <= tq) {
      const h = s[i + 1] - s[i];
      for (let c = 0; c < dim; c++) {
        const a0 = a[i * dim + c],
          a1 = a[(i + 1) * dim + c];
        p[c] += h * v[c] + (h * h * (2 * a0 + a1)) / 6;
        v[c] += (h * (a0 + a1)) / 2;
      }
      i++;
    }
    const h = s[i + 1] - s[i];
    const u = Math.min(Math.max(tq - s[i], 0), h);
    for (let c = 0; c < dim; c++) {
      const a0 = a[i * dim + c],
        a1 = a[(i + 1) * dim + c];
      out[j * dim + c] = h > 0 ? p[c] + v[c] * u + (a0 * u * u) / 2 + ((a1 - a0) * u * u * u) / (6 * h) : p[c];
    }
  }
}

/**
 * Metric scale from IMU acceleration. With the same linear operator F (smoothing-spline
 * second derivative through the keyframe samples → uniform grid → zero-phase band-pass)
 * applied to the visual camera centres C_k and to the double-integrated world-frame IMU
 * acceleration P(τ_k) (R_wd from `AttitudeInterpolator`), the model
 *   F[C] = s₀ F[P] + c_w F[½t²] (+ F[∫∫R_wd] c_d)
 * holds exactly up to noise — aliasing of fast motion by the keyframe sampling affects both
 * sides identically, and drift from unknown initial velocity vanishes under F. Vision is the
 * regressand (its noise is larger after differentiation; IMU noise in the band is small, so
 * the errors-in-variables attenuation ≈ σ²_noise / var(F[P]) is small — ≈ 2 % only at the
 * `minExcitation` floor with 0.05 m/s² noise — and is not corrected). IRLS (Huber) least
 * squares; scale = 1/s₀.
 * The uncertainty is a Newey–West (HAC) sandwich estimate, since the band-limited, oversampled
 * residuals are strongly autocorrelated.
 * Unknowns c_w absorb a constant world-frame acceleration bias (e.g. gravity leakage when
 * accelG − g is used); c_d (optional) a device-frame accelerometer bias with a Gaussian prior.
 */
export function estimateMetricScale(
  keyframes: readonly InertialKeyframe[],
  imu: readonly ImuSample[],
  options: Partial<MetricScaleOptions> = {},
): MetricScaleResult | null {
  const o = { ...DEFAULT_SCALE, ...options };
  const kf = sortKeyframes(keyframes);
  const gyro = new GyroIntegrator(imu, o.gyroBias);
  if (gyro.n < 2 || kf.length < o.minSegmentKeyframes) return null;
  let nAcc = 0,
    nAccG = 0;
  for (const s of imu) {
    if (finite3(s.accel)) nAcc++;
    if (finite3(s.accelG)) nAccG++;
  }
  let source: 'accel' | 'accelG';
  if (o.source === 'accel' || (o.source === 'auto' && nAcc >= 0.5 * imu.length && nAcc > 0)) source = 'accel';
  else source = 'accelG';
  if (source === 'accel' && nAcc === 0) return null;
  if (source === 'accelG' && (nAccG === 0 || !o.gravity)) return null;
  const gdown = o.gravity ? o.gravity.down : [0, 0, -1];
  const gn = Math.hypot(gdown[0], gdown[1], gdown[2]) || 1;
  const gmag = o.gravity?.magnitude ?? G0;
  const gwx = (gmag * gdown[0]) / gn,
    gwy = (gmag * gdown[1]) / gn,
    gwz = (gmag * gdown[2]) / gn;

  // IMU samples with the chosen acceleration (sorted, unique times).
  const samp = imu
    .filter((s) => Number.isFinite(s.t) && finite3(source === 'accel' ? s.accel : s.accelG))
    .sort((a, b) => a.t - b.t)
    .filter((s, i, arr) => i === 0 || s.t > arr[i - 1].t);
  const ns = samp.length;
  if (ns < 4) return null;
  const st = new Float64Array(ns);
  for (let i = 0; i < ns; i++) st[i] = samp[i].t;

  const att = new AttitudeInterpolator(kf, gyro, o.offsetMs);
  // Segment the keyframes.
  const tau = kf.map((k) => k.t + o.offsetMs);
  const segments: [number, number][] = [];
  let segStart = -1;
  const covered = (x: number): boolean => x >= st[0] && x <= st[ns - 1];
  const imuGapBetween = (ta: number, tb: number): boolean => {
    // max IMU sample gap within [ta, tb]
    let lo = 0,
      hi = ns - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (st[mid] <= ta) lo = mid;
      else hi = mid;
    }
    for (let i = lo; i + 1 < ns && st[i] < tb; i++) if (st[i + 1] - st[i] > o.maxImuGapMs) return true;
    return false;
  };
  for (let k = 0; k < kf.length; k++) {
    if (!covered(tau[k])) {
      if (segStart >= 0) segments.push([segStart, k - 1]);
      segStart = -1;
      continue;
    }
    if (segStart < 0) {
      segStart = k;
      continue;
    }
    if (tau[k] - tau[k - 1] > o.maxGapMs || tau[k] - tau[k - 1] <= 1e-6 || imuGapBetween(tau[k - 1], tau[k])) {
      segments.push([segStart, k - 1]);
      segStart = k;
    }
  }
  if (segStart >= 0) segments.push([segStart, kf.length - 1]);

  const rowsY: number[] = [];
  const rowsX: number[][] = [[], [], [], [], [], [], []];
  const rowSeg: number[] = [];
  let usedKf = 0,
    usedSeg = 0,
    dur = 0;
  const R = new Float64Array(9);
  for (const [s0i, s1i] of segments) {
    const nk = s1i - s0i + 1;
    if (nk < o.minSegmentKeyframes) continue;
    const T0 = tau[s0i];
    const durS = (tau[s1i] - T0) / 1000;
    if (durS < o.minSegmentS) continue;
    // IMU samples spanning the segment (one beyond each end when available).
    let i0 = 0;
    while (i0 + 1 < ns && st[i0 + 1] <= tau[s0i]) i0++;
    let i1 = i0;
    while (i1 < ns - 1 && st[i1] < tau[s1i]) i1++;
    const m = i1 - i0 + 1;
    if (m < 4) continue;
    const ss = new Float64Array(m);
    const aw = new Float64Array(3 * m);
    const Rw = new Float64Array(9 * m);
    for (let j = 0; j < m; j++) {
      const smp = samp[i0 + j];
      ss[j] = (smp.t - T0) / 1000;
      att.rotationAt(smp.t, R);
      const a = (source === 'accel' ? smp.accel : smp.accelG)!;
      let x = R[0] * a[0] + R[1] * a[1] + R[2] * a[2];
      let y = R[3] * a[0] + R[4] * a[1] + R[5] * a[2];
      let z = R[6] * a[0] + R[7] * a[1] + R[8] * a[2];
      if (source === 'accelG') {
        x += gwx;
        y += gwy;
        z += gwz;
      }
      aw[3 * j] = x;
      aw[3 * j + 1] = y;
      aw[3 * j + 2] = z;
      Rw.set(R, 9 * j);
    }
    // Keyframe times (s) — strictly increasing by construction of the segments.
    const tk = new Float64Array(nk);
    for (let j = 0; j < nk; j++) tk[j] = (tau[s0i + j] - T0) / 1000;
    const P = new Float64Array(3 * nk);
    const Dm = new Float64Array(9 * nk);
    doubleIntegrateAt(ss, aw, 3, tk, P);
    if (o.estimateDeviceBias) doubleIntegrateAt(ss, Rw, 9, tk, Dm);
    // Channels per keyframe.
    const ch: Float64Array[] = [];
    for (let c = 0; c < NCH; c++) ch.push(new Float64Array(nk));
    for (let j = 0; j < nk; j++) {
      const C = kf[s0i + j].C;
      ch[0][j] = C[0];
      ch[1][j] = C[1];
      ch[2][j] = C[2];
      ch[3][j] = P[3 * j];
      ch[4][j] = P[3 * j + 1];
      ch[5][j] = P[3 * j + 2];
      ch[6][j] = 0.5 * tk[j] * tk[j];
      for (let c = 0; c < 9; c++) ch[7 + c][j] = Dm[9 * j + c];
    }
    // Operator.
    const hs = new Float64Array(nk - 1);
    for (let j = 0; j + 1 < nk; j++) hs[j] = tk[j + 1] - tk[j];
    const hMed = median(hs);
    const fc = Math.min(o.bandHighHz, 0.4 / hMed);
    const wc = 2 * Math.PI * fc;
    const lambda = 1 / (hMed * wc * wc * wc * wc);
    const ng = Math.floor(tk[nk - 1] * o.gridHz) + 1;
    const grid = new Float64Array(ng);
    for (let g = 0; g < ng; g++) grid[g] = g / o.gridHz;
    const op = new SplineSecondDerivative(tk, lambda, grid);
    const F: Float64Array[] = [];
    const tmp = new Float64Array(ng);
    const nch = o.estimateDeviceBias ? NCH : 7;
    for (let c = 0; c < nch; c++) {
      op.apply(ch[c], tmp);
      F.push(filtfiltBandpass(tmp, o.gridHz, o.bandLowHz, o.bandHighHz));
    }
    const gA = Math.ceil(o.edgeTrimS * o.gridHz),
      gB = ng - 1 - gA;
    if (gB - gA < 4) continue;
    for (let ax = 0; ax < 3; ax++) {
      for (let g = gA; g <= gB; g++) {
        rowsY.push(F[ax][g]);
        rowsX[0].push(F[3 + ax][g]);
        for (let b = 0; b < 3; b++) rowsX[1 + b].push(ax === b ? F[6][g] : 0);
        for (let b = 0; b < 3; b++) rowsX[4 + b].push(o.estimateDeviceBias ? F[7 + 3 * ax + b][g] : 0);
        rowSeg.push(usedSeg * 3 + ax);
      }
    }
    usedKf += nk;
    usedSeg++;
    dur += durS;
  }
  const N = rowsY.length;
  if (N < 20 || usedSeg === 0) return null;
  const p = o.estimateDeviceBias ? 7 : 4;
  const y = Float64Array.from(rowsY);
  const X: Float64Array[] = [];
  for (let c = 0; c < p; c++) X.push(Float64Array.from(rowsX[c]));
  const w = new Float64Array(N).fill(1);
  const r = new Float64Array(N);
  let beta: Float64Array = new Float64Array(p);
  let excit = 0;
  let ysq = 0;
  for (let i = 0; i < N; i++) {
    excit += X[0][i] * X[0][i];
    ysq += y[i] * y[i];
  }
  excit = Math.sqrt(excit / N);

  const solve = (prior: number): { beta: Float64Array; A: Float64Array } | null => {
    const A = new Float64Array(p * p),
      bv = new Float64Array(p);
    for (let i = 0; i < N; i++) {
      const wi = w[i];
      if (wi === 0) continue;
      for (let a = 0; a < p; a++) {
        const xa = X[a][i] * wi;
        if (xa === 0) continue;
        bv[a] += xa * y[i];
        for (let b = a; b < p; b++) A[a * p + b] += xa * X[b][i];
      }
    }
    for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) A[a * p + b] = A[b * p + a];
    // prior < 0 ⇒ first pass: device-bias columns effectively excluded (prior unknown yet)
    for (let a = 1; a < p; a++) A[a * p + a] += 1e-9 * N + (a >= 4 ? (prior < 0 ? 1e6 * A[a * p + a] + 1e-6 * N : prior) : 0);
    const sol = solveSPD(A, p, bv);
    return sol ? { beta: sol, A } : null;
  };
  let A: Float64Array = new Float64Array(p * p);
  let prior = -1;
  for (let it = 0; it < 6; it++) {
    const res = solve(prior);
    if (!res) return null;
    beta = res.beta;
    A = res.A;
    for (let i = 0; i < N; i++) {
      let f = 0;
      for (let a = 0; a < p; a++) f += beta[a] * X[a][i];
      r[i] = y[i] - f;
    }
    const absr = new Float64Array(N);
    for (let i = 0; i < N; i++) absr[i] = Math.abs(r[i]);
    const sig = Math.max(1.4826 * median(absr), 1e-12);
    const kH = 2 * sig;
    for (let i = 0; i < N; i++) w[i] = absr[i] <= kH ? 1 : kH / absr[i];
    if (o.estimateDeviceBias) {
      // prior on c_d = −s₀ b_d: variance (s₀ σ_b)² in units of y; ridge = σ_r² / var.
      const s0 = Math.abs(beta[0]) || 1;
      prior = (sig * sig) / Math.max((s0 * o.deviceBiasSigma) ** 2, 1e-300);
    }
  }
  // Uncertainty.
  let rss = 0,
    sw = 0;
  for (let i = 0; i < N; i++) {
    rss += w[i] * r[i] * r[i];
    sw += w[i];
  }
  const sigma2 = rss / Math.max(1, sw - p);
  const e0 = new Float64Array(p);
  e0[0] = 1;
  const Ainv0 = solveSPD(A, p, e0);
  // Newey–West (Bartlett) HAC variance of ŝ₀: influence gᵢ = wᵢ rᵢ (A⁻¹)₀· xᵢ, autocovariances
  // summed within each contiguous block (segment × axis) up to `hacLagS` (Newey & West 1987).
  let varS0 = Infinity,
    nEff = 0;
  if (Ainv0) {
    const g = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let ax = 0;
      for (let a = 0; a < p; a++) ax += Ainv0[a] * X[a][i];
      g[i] = w[i] * r[i] * ax;
    }
    const L = Math.max(1, Math.ceil(o.hacLagS * o.gridHz));
    let v = 0;
    for (let i = 0; i < N; i++) v += g[i] * g[i];
    for (let l = 1; l <= L; l++) {
      const kw = 1 - l / (L + 1);
      let c = 0;
      for (let i = 0; i + l < N; i++) if (rowSeg[i] === rowSeg[i + l]) c += g[i] * g[i + l];
      v += 2 * kw * c;
    }
    const iid = sigma2 * Ainv0[0];
    varS0 = Math.max(v * (N / Math.max(1, N - p)), iid) * HAC_CALIBRATION * HAC_CALIBRATION;
    nEff = (N * iid) / varS0;
  }
  const s0 = beta[0];
  const relUnc = Math.sqrt(Math.max(varS0, 0)) / Math.abs(s0);
  // R² with the robust (Huber) weights of the fit: one misregistered keyframe (a few rows with
  // huge residuals) must not turn an excellent fit into R² ≈ 0 (e2e yard-real: the IMU scale was
  // rejected for "R² 0.01" and the far less precise GPS scale used instead).
  let rr = 0,
    wy = 0;
  for (let i = 0; i < N; i++) {
    rr += w[i] * r[i] * r[i];
    wy += w[i] * y[i] * y[i];
  }
  const r2 = ysq > 0 && wy > 0 ? 1 - rr / wy : 0;
  const reasons: string[] = [];
  if (!(s0 > 0)) reasons.push('non-positive scale');
  if (excit < o.minExcitation) reasons.push(`insufficient excitation (${excit.toFixed(3)} m/s² < ${o.minExcitation})`);
  if (!(relUnc <= o.maxRelUncertainty)) reasons.push(`uncertainty ${(100 * relUnc).toFixed(1)} % > ${(100 * o.maxRelUncertainty).toFixed(0)} %`);
  if (!(r2 >= o.minR2)) reasons.push(`poor fit (R² ${r2.toFixed(2)})`);
  const invS0 = s0 !== 0 ? 1 / s0 : NaN;
  return {
    scale: invS0,
    relUncertainty: relUnc,
    reliable: reasons.length === 0,
    reason: reasons.join('; '),
    excitationRms: excit,
    r2,
    biasWorld: Float64Array.of(-beta[1] * invS0, -beta[2] * invS0, -beta[3] * invS0),
    biasDevice: o.estimateDeviceBias ? Float64Array.of(-beta[4] * invS0, -beta[5] * invS0, -beta[6] * invS0) : null,
    numKeyframes: usedKf,
    numSegments: usedSeg,
    numRows: N,
    numEffective: nEff,
    durationS: dur,
    source,
  };
}

// ---------------------------------------------------------------------------------------
// (a') Time offset from linear acceleration (fallback when rotation cannot observe it)
// ---------------------------------------------------------------------------------------

/** Options of `estimateTimeOffsetFromAcceleration`. */
export interface AccelTimeOffsetOptions {
  /** Search half-range around `initialOffsetMs` (default 250). */
  searchRangeMs: number;
  /** Coarse grid step (default 25 ms; the misfit is smooth on the scale of the motion period). */
  coarseStepMs: number;
  /** Centre of the search (default 0). */
  initialOffsetMs: number;
  /** `reliable` requires stdMs below this (default 5). */
  maxStdMs: number;
  /** Options forwarded to `estimateMetricScale` (`gravity` is required for the accelG source). */
  scale: Partial<MetricScaleOptions>;
}

/** Result of `estimateTimeOffsetFromAcceleration`. */
export interface AccelTimeOffsetResult {
  /** IMU time = camera time + offsetMs. */
  offsetMs: number;
  /** Approximate 1-σ (ms) from the misfit curvature, corrected for residual autocorrelation. */
  stdMs: number;
  /** 1 − min misfit / median misfit over the grid, in [0, 1]. */
  quality: number;
  reliable: boolean;
  /** Metric-scale fit at the optimum. */
  scale: MetricScaleResult;
  /** Misfit 1 − R² of the scale regression on the coarse grid (Infinity where it failed). */
  curve: { offsetMs: Float64Array; cost: Float64Array };
}

const DEFAULT_ACCEL_OFFSET: Omit<AccelTimeOffsetOptions, 'scale'> = { searchRangeMs: 250, coarseStepMs: 25, initialOffsetMs: 0, maxStdMs: 5 };

/**
 * Camera–IMU time offset from LINEAR acceleration: minimises the misfit 1 − R² of the
 * metric-scale regression (`estimateMetricScale`, vision ≈ s₀ · IMU after the same
 * band-limited second-derivative operator) over the offset (coarse grid, then golden section;
 * offsets giving a non-positive scale are rejected). Use it when `estimateTimeOffset` is
 * unreliable because the device hardly rotates (e.g. walking with a steady camera): a wrong
 * offset Δ attenuates the in-band regression by ≈ cos(2π f Δ) and biases the metric scale
 * (40 ms at ~1 Hz motion ⇒ ≈ 5 %). Cost ≈ 35 scale fits. Returns null without any valid fit.
 */
export function estimateTimeOffsetFromAcceleration(
  keyframes: readonly InertialKeyframe[],
  imu: readonly ImuSample[],
  options: Partial<AccelTimeOffsetOptions> = {},
): AccelTimeOffsetResult | null {
  const o = { ...DEFAULT_ACCEL_OFFSET, ...options };
  const sopts = options.scale ?? {};
  const kf = sortKeyframes(keyframes);
  if (kf.length < 2 || imu.length < 4) return null;
  const f = (d: number): number => {
    const r = estimateMetricScale(kf, imu, { ...sopts, offsetMs: d });
    if (!r || !(r.scale > 0) || !Number.isFinite(r.r2)) return Infinity;
    return 1 - r.r2;
  };
  const step = Math.max(o.coarseStepMs, 0.5);
  const range = Math.max(o.searchRangeMs, step);
  const nGrid = Math.floor((2 * range) / step) + 1;
  const offs = new Float64Array(nGrid),
    costs = new Float64Array(nGrid);
  let bi = -1;
  for (let i = 0; i < nGrid; i++) {
    offs[i] = o.initialOffsetMs - range + i * step;
    costs[i] = f(offs[i]);
    if (Number.isFinite(costs[i]) && (bi < 0 || costs[i] < costs[bi])) bi = i;
  }
  if (bi < 0) return null;
  const lo = offs[Math.max(0, bi - 1)],
    hi = offs[Math.min(nGrid - 1, bi + 1)];
  let best = offs[bi],
    bestCost = costs[bi];
  const ref = goldenSection(f, lo, hi, 0.25);
  const refCost = f(ref);
  if (refCost < bestCost) {
    best = ref;
    bestCost = refCost;
  }
  // Curvature of the misfit; residuals are strongly autocorrelated (band-limited, oversampled),
  // so the information is scaled by the effective row count of the fit at the optimum.
  const h = 2;
  const cp = f(best + h),
    cm = f(best - h);
  const res = estimateMetricScale(kf, imu, { ...sopts, offsetMs: best });
  if (!res) return null;
  const curv = (cp - 2 * bestCost + cm) / (h * h);
  const nEff = Math.max(1, Number.isFinite(res.numEffective) ? res.numEffective : 1);
  const stdMs = curv > 0 && Number.isFinite(curv) ? Math.sqrt((2 * Math.max(bestCost, 1e-12)) / (curv * nEff)) : Infinity;
  const finite: number[] = [];
  for (let i = 0; i < nGrid; i++) if (Number.isFinite(costs[i])) finite.push(Math.min(costs[i], 1));
  const med = median(finite);
  const quality = med > 0 ? Math.min(1, Math.max(0, 1 - bestCost / med)) : 0;
  const atEdge = Math.abs(best - o.initialOffsetMs) > range - step;
  return {
    offsetMs: best,
    stdMs,
    quality,
    reliable: res.reliable && !atEdge && stdMs <= o.maxStdMs && quality > 0.3,
    scale: res,
    curve: { offsetMs: offs, cost: costs },
  };
}

// ---------------------------------------------------------------------------------------
// Convenience
// ---------------------------------------------------------------------------------------

/** All inertial estimates in sequence: time offset → gravity → metric scale. */
export interface InertialAlignment {
  timeOffset: TimeOffsetResult | null;
  /** Acceleration-based offset (only computed when the gyro-based one is unreliable). */
  accelTimeOffset: AccelTimeOffsetResult | null;
  /** Offset actually used (estimated when reliable, else the given/initial one). */
  offsetMs: number;
  /** Where `offsetMs` comes from. */
  offsetSource: 'gyro' | 'accel' | 'default';
  gravity: GravityResult | null;
  scale: MetricScaleResult | null;
  /** composeGravityAlignment(gravity.down) or null. */
  R_gravity: Mat3 | null;
}

/**
 * Runs (a)–(d) with sensible fallbacks. When the rotation-based time offset is unreliable
 * (too little angular acceleration), the offset is estimated from the linear acceleration
 * instead (`estimateTimeOffsetFromAcceleration`), since an offset error biases the scale.
 */
export function estimateInertialAlignment(
  keyframes: readonly InertialKeyframe[],
  imu: readonly ImuSample[],
  opts: {
    timeOffset?: Partial<TimeOffsetOptions>;
    accelTimeOffset?: Partial<Omit<AccelTimeOffsetOptions, 'scale'>>;
    gravity?: Partial<GravityOptions>;
    scale?: Partial<MetricScaleOptions>;
    offsetMs?: number;
  } = {},
): InertialAlignment {
  const timeOffset = imu.length >= 2 ? estimateTimeOffset(keyframes, imu, { initialOffsetMs: opts.offsetMs ?? 0, ...opts.timeOffset }) : null;
  let offsetMs = opts.offsetMs ?? 0;
  let offsetSource: InertialAlignment['offsetSource'] = 'default';
  const gyroBias = timeOffset && timeOffset.reliable ? timeOffset.gyroBias : null;
  let gravity: GravityResult | null;
  let accelTimeOffset: AccelTimeOffsetResult | null = null;
  const gravityFor = (g: GravityResult | null): MetricScaleOptions['gravity'] =>
    g ? { down: g.down, magnitude: g.source === 'accelG' ? g.magnitude : undefined } : (opts.scale?.gravity ?? null);
  if (timeOffset && timeOffset.reliable) {
    offsetMs = timeOffset.offsetMs;
    offsetSource = 'gyro';
    gravity = estimateGravity(keyframes, imu, { gyroBias, ...opts.gravity, offsetMs });
  } else {
    // Gravity is insensitive to the offset when the device rotates slowly (the case here).
    gravity = estimateGravity(keyframes, imu, { gyroBias, ...opts.gravity, offsetMs });
    if (imu.length >= 4) {
      accelTimeOffset = estimateTimeOffsetFromAcceleration(keyframes, imu, {
        initialOffsetMs: offsetMs,
        ...opts.accelTimeOffset,
        scale: { gyroBias, ...opts.scale, gravity: gravityFor(gravity) },
      });
      if (accelTimeOffset && accelTimeOffset.reliable) {
        offsetMs = accelTimeOffset.offsetMs;
        offsetSource = 'accel';
        gravity = estimateGravity(keyframes, imu, { gyroBias, ...opts.gravity, offsetMs });
      }
    }
  }
  const scale = estimateMetricScale(keyframes, imu, { gyroBias, ...opts.scale, offsetMs, gravity: gravityFor(gravity) });
  return {
    timeOffset,
    accelTimeOffset,
    offsetMs,
    offsetSource,
    gravity,
    scale,
    R_gravity: gravity ? composeGravityAlignment(gravity.down) : null,
  };
}
