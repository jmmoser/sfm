/**
 * Gyro consistency of vision rotations (pairs, registrations, whole models) and an incremental,
 * bias-aware gyro integrator in the CAMERA frame.
 *
 * Why: hand-held captures have fast pans. A pan can leave a large rotation gap between
 * keyframes, and the mapper then reconnects the walk through FALSE two-view pairs (repeated
 * structure such as fence boards: 60–80 inliers, a relative rotation 100°+ off) — the result is
 * a model made of rigid fragments joined by a wrong rotation (e2e `yard-real`, README "Yard
 * breakdown"). The gyro measures the true relative rotation of any two frames to within a few
 * degrees (integration noise, residual bias × Δt, camera–IMU offset), so every vision rotation
 * can be checked against it:
 *
 *  - `checkPairRotation`: a verified two-view pair (x_b = R x_a) vs the gyro rotation;
 *  - `checkPoseRotation`: a new absolute pose vs gyro-propagated rotations of registered images
 *    close in time (majority vote — a neighbour may itself be wrong);
 *  - `gyroConsistency`: consecutive registered images of a model — breaks (fragments joined by a
 *    wrong rotation) and isolated outlier images.
 *
 * Tolerance = baseDeg + driftDegPerS·|Δt| + rotationFraction·|gyro rotation| (`gyroToleranceRad`):
 * the constant covers integration / time-offset / R_cam_device errors and the vision rotation's
 * own error, the drift an uncompensated (or residual) gyro bias, the last the gyro scale error and
 * time-offset errors during fast rotation. Wrong pairs measured on the e2e datasets are 50–140°
 * off; correct ones ≤ 2–3°.
 *
 * Conventions (ARCHITECTURE §2): gyro ω_dev with dR_wd/dt = R_wd [ω]×; R_cam_device maps device
 * to camera vectors; "camera(ta) → camera(tb)" = R with x_b = R x_a for a static direction
 * (= R_b R_aᵀ for world-to-camera poses); IMU time = camera time + offsetMs.
 */
import type { ImuSample, Mat3, Pose, Vec3 } from '../core/types';
import { mat3Mul, mat3MulTranspose } from '../math/mat3';
import { mat3ToRotvec, rotationAngle } from '../math/rotation';
import { median } from '../math/stats';

const DEG = Math.PI / 180;

/** Tolerance model of the gyro checks. */
export interface GyroToleranceOptions {
  /** Constant part (deg; default 5). */
  baseDeg: number;
  /** Growth with |Δt| between the two views (deg/s; default 0.3 ≈ an uncompensated 0.005 rad/s bias). */
  driftDegPerS: number;
  /** Growth with the gyro rotation angle itself (fraction; default 0.05). */
  rotationFraction: number;
}

export const DEFAULT_GYRO_TOLERANCE: Readonly<GyroToleranceOptions> = Object.freeze({ baseDeg: 5, driftDegPerS: 0.3, rotationFraction: 0.05 });

/** Tolerance (rad) of a gyro-vs-vision rotation comparison over |Δt| ms and a gyro rotation of `angleRad`. */
export function gyroToleranceRad(dtMs: number, angleRad: number, o: Partial<GyroToleranceOptions> = {}): number {
  const base = o.baseDeg ?? DEFAULT_GYRO_TOLERANCE.baseDeg;
  const drift = o.driftDegPerS ?? DEFAULT_GYRO_TOLERANCE.driftDegPerS;
  const frac = o.rotationFraction ?? DEFAULT_GYRO_TOLERANCE.rotationFraction;
  return (base + drift * Math.abs(dtMs) * 1e-3) * DEG + frac * Math.abs(angleRad);
}

/** Angle (rad) between two rotations of the same kind (e.g. a vision and a gyro relative rotation). */
export function rotationDisagreement(Ra: ArrayLike<number>, Rb: ArrayLike<number>): number {
  return rotationAngle(mat3MulTranspose(Ra, Rb));
}

// ------------------------------------------------------------------------------------------
// quaternion helpers (allocation-free; [w, x, y, z], Hamilton)

function qmulInto(a: Float64Array, ao: number, b: Float64Array, bo: number, out: Float64Array, oo: number): void {
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
  const inv = 1 / Math.hypot(w, x, y, z);
  out[oo] = w * inv;
  out[oo + 1] = x * inv;
  out[oo + 2] = y * inv;
  out[oo + 3] = z * inv;
}

function qexpInto(px: number, py: number, pz: number, out: Float64Array, o: number): void {
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

/** Rotation matrix of quaternion q (row-major). */
function q2mat(q: Float64Array, o: number, out: Float64Array): void {
  const w = q[o],
    x = q[o + 1],
    y = q[o + 2],
    z = q[o + 3];
  out[0] = 1 - 2 * (y * y + z * z);
  out[1] = 2 * (x * y - w * z);
  out[2] = 2 * (x * z + w * y);
  out[3] = 2 * (x * y + w * z);
  out[4] = 1 - 2 * (x * x + z * z);
  out[5] = 2 * (y * z - w * x);
  out[6] = 2 * (x * z - w * y);
  out[7] = 2 * (y * z + w * x);
  out[8] = 1 - 2 * (x * x + y * y);
}

function finite3(v: ArrayLike<number> | null | undefined): boolean {
  return !!v && Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2]);
}

/**
 * Longest IMU sample gap (ms) a rotation is integrated across. Longer gaps (page hidden, worker
 * respawn, ring-buffer overflow) have an unknown rotation: an interval spanning one is not covered.
 */
export const IMU_MAX_GAP_MS = 500;

/**
 * Incremental gyro integrator: cumulative device attitude Q(t) = R_wd(t₀)ᵀ R_wd(t) at every sample
 * (2nd-order Magnus steps, linear ω within a sample interval), bias-corrected, grown by `sync` as
 * samples arrive (amortised O(new samples); a rebuild only when the bias changes or a sample was
 * inserted out of order). Queries take CAMERA times (IMU time = camera time + offsetMs); an
 * interval across a sample gap > IMU_MAX_GAP_MS is not covered (no rotation, so no gyro check).
 */
export class GyroReference {
  private t = new Float64Array(256);
  private w = new Float64Array(3 * 256);
  private q = new Float64Array(4 * 256);
  private n = 0;
  /** Indices i with t[i] − t[i−1] > IMU_MAX_GAP_MS (ascending). */
  private gapEnd: number[] = [];
  private last: ImuSample | null = null;
  private srcLen = 0;
  private readonly tq = new Float64Array(8);
  private readonly ta = new Float64Array(9);
  private readonly tb = new Float64Array(9);
  private biasV: Vec3 | null = null;
  /** Camera–IMU offset (ms): IMU time = camera time + offsetMs. */
  offsetMs = 0;
  /** R_cam_device used to express rotations in the camera frame. */
  R_cam_device: Mat3 = Float64Array.of(1, 0, 0, 0, -1, 0, 0, 0, -1);

  constructor(samples?: readonly ImuSample[], opts: { R_cam_device?: Mat3; offsetMs?: number; bias?: ArrayLike<number> | null } = {}) {
    if (opts.R_cam_device) this.R_cam_device = Float64Array.from(opts.R_cam_device);
    if (opts.offsetMs !== undefined && Number.isFinite(opts.offsetMs)) this.offsetMs = opts.offsetMs;
    if (opts.bias && finite3(opts.bias)) this.biasV = Float64Array.from(opts.bias);
    if (samples) this.sync(samples);
  }

  /** Gyro bias (rad/s, device frame) subtracted from ω; null = none. */
  get bias(): Vec3 | null {
    return this.biasV;
  }

  /** Number of integrated samples. */
  get numSamples(): number {
    return this.n;
  }

  /** Change the bias (re-integrates on the next `sync`). */
  setBias(b: ArrayLike<number> | null, samples: readonly ImuSample[]): void {
    const nb = b && finite3(b) ? Float64Array.from(b) : null;
    const same = (nb === null && this.biasV === null) || (nb !== null && this.biasV !== null && nb[0] === this.biasV[0] && nb[1] === this.biasV[1] && nb[2] === this.biasV[2]);
    this.biasV = nb;
    if (!same) {
      this.n = 0;
      this.last = null;
      this.srcLen = 0;
    }
    this.sync(samples);
  }

  /** A copy (same samples, bias, offset and R_cam_device). */
  clone(): GyroReference {
    const g = new GyroReference(undefined, { R_cam_device: this.R_cam_device, offsetMs: this.offsetMs, bias: this.biasV });
    g.t = this.t.slice(0, Math.max(1, this.n));
    g.w = this.w.slice(0, Math.max(3, 3 * this.n));
    g.q = this.q.slice(0, Math.max(4, 4 * this.n));
    g.n = this.n;
    g.gapEnd = this.gapEnd.slice();
    g.last = this.last;
    g.srcLen = this.srcLen;
    return g;
  }

  /**
   * Bring the integration up to date with `samples` (sorted by time, as `ImuLog.samples`; samples
   * with a non-finite time / gyro or a time not after the previous one are skipped).
   */
  sync(samples: readonly ImuSample[]): void {
    const L = samples.length;
    // prefix intact ⇔ the last source sample seen by the previous sync is still at its position
    let start = 0;
    if (this.srcLen > 0 && L >= this.srcLen && samples[this.srcLen - 1] === this.last) start = this.srcLen;
    else {
      this.n = 0;
      this.gapEnd = [];
    }
    const bx = this.biasV ? this.biasV[0] : 0,
      by = this.biasV ? this.biasV[1] : 0,
      bz = this.biasV ? this.biasV[2] : 0;
    for (let k = start; k < L; k++) {
      const s = samples[k];
      if (!s || !Number.isFinite(s.t) || !finite3(s.gyro)) continue;
      const i = this.n;
      if (i > 0 && !(s.t > this.t[i - 1])) continue;
      this.ensure(i + 1);
      this.t[i] = s.t;
      this.w[3 * i] = s.gyro[0] - bx;
      this.w[3 * i + 1] = s.gyro[1] - by;
      this.w[3 * i + 2] = s.gyro[2] - bz;
      if (i === 0) {
        this.q[0] = 1;
        this.q[1] = this.q[2] = this.q[3] = 0;
      } else {
        if (this.t[i] - this.t[i - 1] > IMU_MAX_GAP_MS) this.gapEnd.push(i);
        const h = (this.t[i] - this.t[i - 1]) / 1000;
        this.step(i - 1, h, this.w[3 * i], this.w[3 * i + 1], this.w[3 * i + 2], this.q, 4 * i);
      }
      this.n = i + 1;
    }
    this.srcLen = L;
    this.last = L > 0 ? samples[L - 1] : null;
  }

  private ensure(m: number): void {
    if (m <= this.t.length) return;
    const cap = Math.max(m, 2 * this.t.length);
    const t = new Float64Array(cap);
    t.set(this.t.subarray(0, this.n));
    const w = new Float64Array(3 * cap);
    w.set(this.w.subarray(0, 3 * this.n));
    const q = new Float64Array(4 * cap);
    q.set(this.q.subarray(0, 4 * this.n));
    this.t = t;
    this.w = w;
    this.q = q;
  }

  /** out[oo..] = q_i ⊗ exp(φ), φ = Magnus step from sample i over h s to the end rate (ex, ey, ez). */
  private step(i: number, h: number, ex: number, ey: number, ez: number, out: Float64Array, oo: number): void {
    const ax = this.w[3 * i],
      ay = this.w[3 * i + 1],
      az = this.w[3 * i + 2];
    const c = (h * h) / 12;
    const px = 0.5 * h * (ax + ex) + c * (ay * ez - az * ey);
    const py = 0.5 * h * (ay + ey) + c * (az * ex - ax * ez);
    const pz = 0.5 * h * (az + ez) + c * (ax * ey - ay * ex);
    qexpInto(px, py, pz, this.tq, 4);
    qmulInto(this.q, 4 * i, this.tq, 4, out, oo);
  }

  /** IMU-clock coverage [first, last] sample time; null when < 2 samples. */
  get range(): [number, number] | null {
    return this.n >= 2 ? [this.t[0], this.t[this.n - 1]] : null;
  }

  /** True when the samples cover the CAMERA times [ta, tb] (within slackMs) without a gap > IMU_MAX_GAP_MS. */
  covers(ta: number, tb: number, slackMs = 60): boolean {
    if (this.n < 2 || !Number.isFinite(ta) || !Number.isFinite(tb)) return false;
    const lo = Math.min(ta, tb) + this.offsetMs + slackMs,
      hi = Math.max(ta, tb) + this.offsetMs - slackMs;
    if (!(this.t[0] <= lo && this.t[this.n - 1] >= hi)) return false;
    // the first gap ending after lo is the only one that can start before hi
    const ge = this.gapEnd;
    let a = 0,
      b = ge.length;
    while (a < b) {
      const mid = (a + b) >> 1;
      if (this.t[ge[mid]] <= lo) a = mid + 1;
      else b = mid;
    }
    return a === ge.length || this.t[ge[a] - 1] >= hi;
  }

  /** Q(t) at IMU time t into out[o..o+3] (ω held constant outside the samples). */
  private quatAt(tm: number, out: Float64Array, o: number): void {
    const n = this.n;
    if (n === 0) {
      out[o] = 1;
      out[o + 1] = out[o + 2] = out[o + 3] = 0;
      return;
    }
    if (n === 1 || tm <= this.t[0] || tm >= this.t[n - 1]) {
      const i = n === 1 || tm <= this.t[0] ? 0 : n - 1;
      const h = (tm - this.t[i]) / 1000;
      qexpInto(h * this.w[3 * i], h * this.w[3 * i + 1], h * this.w[3 * i + 2], this.tq, 4);
      qmulInto(this.q, 4 * i, this.tq, 4, out, o);
      return;
    }
    let lo = 0,
      hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (this.t[mid] <= tm) lo = mid;
      else hi = mid;
    }
    const i = lo;
    const span = this.t[i + 1] - this.t[i];
    const a = span > 0 ? (tm - this.t[i]) / span : 0;
    const ex = this.w[3 * i] + a * (this.w[3 * i + 3] - this.w[3 * i]);
    const ey = this.w[3 * i + 1] + a * (this.w[3 * i + 4] - this.w[3 * i + 1]);
    const ez = this.w[3 * i + 2] + a * (this.w[3 * i + 5] - this.w[3 * i + 2]);
    this.step(i, (tm - this.t[i]) / 1000, ex, ey, ez, out, o);
  }

  /**
   * Device rotation G with v_d(tb) = G · v_d(ta) for a static direction (= R_wd(tb)ᵀ R_wd(ta),
   * the convention of `sensors/motion.integrateGyro`), CAMERA times.
   */
  relativeDevice(ta: number, tb: number, out: Mat3 = new Float64Array(9)): Mat3 {
    const qa = new Float64Array(4),
      qb = new Float64Array(4);
    this.quatAt(ta + this.offsetMs, qa, 0);
    this.quatAt(tb + this.offsetMs, qb, 0);
    q2mat(qa, 0, this.ta);
    q2mat(qb, 0, this.tb);
    // Q(tb)ᵀ Q(ta)
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++) out[3 * r + c] = this.tb[r] * this.ta[c] + this.tb[3 + r] * this.ta[3 + c] + this.tb[6 + r] * this.ta[6 + c];
    return out;
  }

  /** Camera rotation camera(ta) → camera(tb) (R_cd · G · R_cdᵀ); null when the IMU does not cover [ta, tb]. */
  relative(ta: number, tb: number, out: Mat3 = new Float64Array(9)): Mat3 | null {
    if (!this.covers(ta, tb)) return null;
    const G = this.relativeDevice(ta, tb);
    const Rcd = this.R_cam_device;
    const out2 = mat3MulTranspose(mat3Mul(Rcd, G), Rcd);
    out.set(out2);
    return out;
  }

  /**
   * Mean angular velocity (rad/s) in the CAMERA frame over [t − h, t + h] (camera time):
   * ω_c with R_c(t + τ ← t) ≈ exp(−[ω_c τ]×), i.e. a static direction x moves as ẋ = −ω_c × x.
   * Null when not covered.
   */
  angularVelocityCam(t: number, halfWindowMs = 12, out: Vec3 = new Float64Array(3)): Vec3 | null {
    if (!this.covers(t - halfWindowMs, t + halfWindowMs, 20)) return null;
    // device: G(t−h → t+h) = R_wd(t+h)ᵀ R_wd(t−h) ≈ exp(−[ω_d 2h]×)
    const G = this.relativeDevice(t - halfWindowMs, t + halfWindowMs);
    const r = mat3ToRotvec(G);
    const s = -1000 / (2 * halfWindowMs);
    const wd0 = r[0] * s,
      wd1 = r[1] * s,
      wd2 = r[2] * s;
    const R = this.R_cam_device;
    out[0] = R[0] * wd0 + R[1] * wd1 + R[2] * wd2;
    out[1] = R[3] * wd0 + R[4] * wd1 + R[5] * wd2;
    out[2] = R[6] * wd0 + R[7] * wd1 + R[8] * wd2;
    return out;
  }
}

/** Result of one gyro check. */
export interface GyroCheck {
  ok: boolean;
  /** Disagreement (rad; median over the neighbours for `checkPoseRotation`). */
  errRad: number;
  /** Tolerance (rad) it was compared with (of the nearest neighbour for `checkPoseRotation`). */
  tolRad: number;
  /** Neighbours that agreed / disagreed (`checkPoseRotation`; 1/0 or 0/1 for pairs). */
  agree: number;
  disagree: number;
}

/**
 * Gyro check of a two-view relative rotation R_ba (x_b = R_ba x_a) of frames at camera times ta, tb.
 * Null when the IMU does not cover the interval or the pair is too far apart in time (maxGapMs).
 */
export function checkPairRotation(
  gyro: GyroReference,
  ta: number,
  tb: number,
  R_ba: ArrayLike<number>,
  tol: Partial<GyroToleranceOptions> = {},
  maxGapMs = Infinity,
): GyroCheck | null {
  if (!(Math.abs(tb - ta) <= maxGapMs)) return null;
  const G = gyro.relative(ta, tb);
  if (!G) return null;
  const err = rotationDisagreement(R_ba, G);
  const t = gyroToleranceRad(tb - ta, rotationAngle(G), tol);
  const ok = err <= t;
  return { ok, errRad: err, tolRad: t, agree: ok ? 1 : 0, disagree: ok ? 0 : 1 };
}

/** A registered image for the pose / model checks: capture time (camera clock) and world-to-camera rotation. */
export interface TimedRotation {
  id: number;
  t: number;
  R: ArrayLike<number>;
}

/**
 * Gyro check of the rotation R (world-to-camera) of a frame at time t against up to `maxNeighbours`
 * registered images nearest in time (within maxGapMs): predicted R̂ = R_c(t_n → t) · R_n. Majority
 * vote (ties pass): a neighbour may itself be misregistered. Null when no neighbour can be checked.
 */
export function checkPoseRotation(
  gyro: GyroReference,
  t: number,
  R: ArrayLike<number>,
  neighbours: readonly TimedRotation[],
  opts: { maxNeighbours?: number; maxGapMs?: number; tolerance?: Partial<GyroToleranceOptions> } = {},
): GyroCheck | null {
  const k = opts.maxNeighbours ?? 3;
  const maxGap = opts.maxGapMs ?? 30_000;
  const near = neighbours
    .filter((n) => Number.isFinite(n.t) && Math.abs(n.t - t) <= maxGap)
    .sort((a, b) => Math.abs(a.t - t) - Math.abs(b.t - t) || a.id - b.id);
  const errs: number[] = [];
  let agree = 0,
    disagree = 0,
    tol0 = NaN;
  for (const nb of near) {
    if (agree + disagree >= k) break;
    const G = gyro.relative(nb.t, t);
    if (!G) continue;
    const Rhat = mat3Mul(G, nb.R);
    const e = rotationDisagreement(R, Rhat);
    const tl = gyroToleranceRad(t - nb.t, rotationAngle(G), opts.tolerance);
    if (Number.isNaN(tol0)) tol0 = tl;
    errs.push(e);
    if (e <= tl) agree++;
    else disagree++;
  }
  if (agree + disagree === 0) return null;
  return { ok: agree >= disagree, errRad: median(errs), tolRad: tol0, agree, disagree };
}

/** Gyro consistency of a model (registered images sorted by time). */
export interface GyroConsistency {
  /** Consecutive (time-ordered) image pairs that could be checked. */
  pairs: number;
  /** Consecutive pairs whose relative rotation disagrees with the gyro. */
  breaks: { a: number; b: number; errDeg: number; tolDeg: number }[];
  /**
   * Isolated outliers: images that disagree with every checked neighbour among the 2 before and
   * the 2 after (≥ 2 checks) while those neighbours agree with each other.
   */
  outliers: number[];
  /** Breaks left after removing the outliers (fragments joined by a wrong rotation). */
  breaksWithoutOutliers: number;
  /** Median disagreement (deg) of the consecutive pairs. */
  medianErrDeg: number;
}

/**
 * Consistency of the registered images' rotations with the gyro: consecutive pairs (within
 * maxGapMs), isolated outlier images, and the breaks that remain without them.
 */
export function gyroConsistency(
  images: readonly TimedRotation[],
  gyro: GyroReference,
  opts: { maxGapMs?: number; tolerance?: Partial<GyroToleranceOptions> } = {},
): GyroConsistency {
  const maxGap = opts.maxGapMs ?? 5000;
  const imgs = images.filter((x) => Number.isFinite(x.t)).sort((a, b) => a.t - b.t || a.id - b.id);
  const n = imgs.length;
  const check = (a: TimedRotation, b: TimedRotation): GyroCheck | null =>
    checkPairRotation(gyro, a.t, b.t, mat3MulTranspose(b.R, a.R), opts.tolerance, maxGap);
  const breaks: GyroConsistency['breaks'] = [];
  const errs: number[] = [];
  let pairs = 0;
  for (let i = 0; i + 1 < n; i++) {
    const c = check(imgs[i], imgs[i + 1]);
    if (!c) continue;
    pairs++;
    errs.push(c.errRad / DEG);
    if (!c.ok) breaks.push({ a: imgs[i].id, b: imgs[i + 1].id, errDeg: c.errRad / DEG, tolDeg: c.tolRad / DEG });
  }
  const outliers: number[] = [];
  if (breaks.length) {
    const bad = new Set<number>();
    for (const b of breaks) {
      bad.add(b.a);
      bad.add(b.b);
    }
    for (let i = 0; i < n; i++) {
      if (!bad.has(imgs[i].id)) continue;
      let ok = 0,
        nok = 0;
      const nbr: TimedRotation[] = [];
      for (const j of [i - 2, i - 1, i + 1, i + 2]) {
        if (j < 0 || j >= n) continue;
        const c = check(imgs[Math.min(i, j)], imgs[Math.max(i, j)]);
        if (!c) continue;
        nbr.push(imgs[j]);
        if (c.ok) ok++;
        else nok++;
      }
      if (ok > 0 || nok < 2) continue;
      // the neighbours must agree with each other (else this is a fragment boundary)
      let consistent = true;
      for (let a = 0; a + 1 < nbr.length && consistent; a++) {
        const c = check(nbr[a], nbr[a + 1]);
        if (c && !c.ok) consistent = false;
      }
      if (consistent) outliers.push(imgs[i].id);
    }
  }
  let rest = 0;
  if (outliers.length) {
    const out = new Set(outliers);
    const kept = imgs.filter((x) => !out.has(x.id));
    for (let i = 0; i + 1 < kept.length; i++) {
      const c = check(kept[i], kept[i + 1]);
      if (c && !c.ok) rest++;
    }
  } else rest = breaks.length;
  return { pairs, breaks, outliers, breaksWithoutOutliers: rest, medianErrDeg: errs.length ? median(errs) : NaN };
}

/** Rotations of the registered images of a pose map, for the checks above. */
export function timedRotations(ids: readonly number[], pose: (id: number) => Pose | undefined, time: (id: number) => number | undefined): TimedRotation[] {
  const out: TimedRotation[] = [];
  for (const id of ids) {
    const p = pose(id);
    const t = time(id);
    if (p && t !== undefined && Number.isFinite(t)) out.push({ id, t, R: p.R });
  }
  return out;
}
