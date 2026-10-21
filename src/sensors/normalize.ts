/**
 * Pure, DOM-free normalisation of raw W3C DeviceMotion / DeviceOrientation events into
 * the project's `ImuSample` convention (ARCHITECTURE §2), plus the run-time self-checks
 * of docs/research/ios-sensors.md §7 that detect platform deviations from the spec.
 *
 * Verified platform facts (docs/research/ios-sensors.md, WebKit source):
 *  - rotationRate: deg/s, alpha = ω about device x, beta = ω about y, gamma = ω about z
 *    (current spec + WebKit `rad2deg(rotationRate.x, .y, .z)`; older docs that call alpha the
 *    z-rate are wrong). Right-hand rule. → rad/s: × π/180, identity axis map.
 *  - interval: iOS reports SECONDS (constant 1/60), the spec says ms → `normalizeIntervalMs`.
 *  - accelerationIncludingGravity AND acceleration: iOS reports the NEGATED spec values
 *    (face-up z ≈ −9.81; spec / Chrome: +9.81). Normalised output uses the SPEC sign
 *    (`ImuSample.accelG` face-up ≈ (0, 0, +9.81), i.e. specific force pointing UP at rest).
 *  - event.timeStamp is on the performance.now() timeline (1 ms resolution, 20 µs when
 *    cross-origin isolated); WebKit replays the last cached sample when a listener is added
 *    and can poll the same CoreMotion sample twice (bit-identical duplicate).
 *
 * The accelerometer sign is never taken from the user agent: the prior (−1 on iOS WebKit,
 * detected by `DeviceMotionEvent.requestPermission`) is replaced as soon as check B decides:
 *  B  (primary) at near-stillness, c = (raw · up_dev(β, γ)) / |raw| with up_dev from the
 *     DeviceOrientation event of the same tick (CoreMotion's fused attitude, independent of
 *     the accelerometer sign convention): c > 0.85 ⇒ +1 (spec), c < −0.85 ⇒ −1 (iOS).
 *  B' (fallback, no orientation events) the UI is upright, so the specific force has a
 *     positive component along the UI "up" axis (sin θ, cos θ, 0) whenever the phone is held
 *     roughly upright; a weak vote that needs ≥ 2 s of consistent evidence.
 *  B'' sign of `acceleration` relative to `accelerationIncludingGravity`: correlation of the
 *     gravity-free field with the high-passed gravity-including field.
 *  D  gyro axes / sign / units: body rates derived from consecutive orientation events
 *     (log(R_kᵀ R_{k+1}) / dt) are compared with rotationRate; the best signed permutation
 *     and scale (1 ⇒ deg/s, 57.3 ⇒ the platform reported rad/s) are applied when conclusive.
 */
import type { ImuSample, Mat3, Vec3 } from '../core/types';
import { mat3ToRotvec } from '../math/rotation';

export const STANDARD_GRAVITY = 9.80665;
export const DEG_TO_RAD = Math.PI / 180;

/** Structural subset of `DeviceMotionEventAcceleration`. */
export interface RawXYZ {
  readonly x: number | null;
  readonly y: number | null;
  readonly z: number | null;
}
/** Structural subset of `DeviceMotionEventRotationRate`. */
export interface RawRotationRate {
  readonly alpha: number | null;
  readonly beta: number | null;
  readonly gamma: number | null;
}
/** Structural subset of `DeviceMotionEvent` (a real event can be passed directly). */
export interface RawMotionEvent {
  readonly timeStamp: number;
  readonly interval?: number | null;
  readonly rotationRate?: RawRotationRate | null;
  readonly acceleration?: RawXYZ | null;
  readonly accelerationIncludingGravity?: RawXYZ | null;
}
/** Structural subset of `DeviceOrientationEvent` incl. the iOS-only compass fields. */
export interface RawOrientationEvent {
  readonly timeStamp: number;
  readonly alpha: number | null;
  readonly beta: number | null;
  readonly gamma: number | null;
  readonly absolute?: boolean | null;
  readonly webkitCompassHeading?: number | null;
  readonly webkitCompassAccuracy?: number | null;
}

/**
 * Raw record layout used by the ring buffer (values exactly as delivered by the platform,
 * so a later calibration change re-interprets everything still buffered):
 * [alpha, beta, gamma, ax, ay, az, gx, gy, gz, flags]; flags bit0 = `acceleration` present,
 * bit1 = `accelerationIncludingGravity` present.
 */
export const RAW_STRIDE = 10;
export const RAW_HAS_ACCEL = 1;
export const RAW_HAS_ACCELG = 2;

/** Platform `interval` → milliseconds (iOS reports seconds). NaN if missing/invalid. */
export function normalizeIntervalMs(interval: number | null | undefined): number {
  if (typeof interval !== 'number' || !Number.isFinite(interval) || interval <= 0) return NaN;
  return interval < 1 ? interval * 1000 : interval;
}

function finite3(v: RawXYZ | null | undefined): boolean {
  return !!v && Number.isFinite(v.x as number) && Number.isFinite(v.y as number) && Number.isFinite(v.z as number);
}

// ---------------------------------------------------------------------------------------------
// DeviceOrientation Euler angles (spec: intrinsic Z-X'-Y'', R = Rz(α)·Rx(β)·Ry(γ), device→earth)
// ---------------------------------------------------------------------------------------------

/**
 * Device → earth rotation from DeviceOrientation angles (degrees): R = Rz(α)·Rx(β)·Ry(γ)
 * (W3C intrinsic Z-X'-Y'', α ∈ [0,360) about the vertical, β ∈ [−180,180) about device x,
 * γ ∈ [−90,90) about device y). Earth frame: Z up; X east / Y north for absolute events, an
 * arbitrary yaw reference otherwise (iOS: CoreMotion xArbitraryZVertical).
 */
export function deviceOrientationMatrix(alphaDeg: number, betaDeg: number, gammaDeg: number, out: Mat3 = new Float64Array(9)): Mat3 {
  const a = alphaDeg * DEG_TO_RAD;
  const b = betaDeg * DEG_TO_RAD;
  const g = gammaDeg * DEG_TO_RAD;
  const ca = Math.cos(a),
    sa = Math.sin(a),
    cb = Math.cos(b),
    sb = Math.sin(b),
    cg = Math.cos(g),
    sg = Math.sin(g);
  out[0] = ca * cg - sa * sb * sg;
  out[1] = -sa * cb;
  out[2] = ca * sg + sa * sb * cg;
  out[3] = sa * cg + ca * sb * sg;
  out[4] = ca * cb;
  out[5] = sa * sg - ca * sb * cg;
  out[6] = -cb * sg;
  out[7] = sb;
  out[8] = cb * cg;
  return out;
}

/** Inverse of {@link deviceOrientationMatrix} with the spec ranges (degrees). */
export function deviceOrientationFromMatrix(R: ArrayLike<number>): { alpha: number; beta: number; gamma: number } {
  // row 2 = (−cβ sγ, sβ, cβ cγ); with γ ∈ [−90, 90) cγ ≥ 0 ⇒ sign(cβ) = sign(R22).
  const sgn = R[8] >= 0 ? 1 : -1;
  const cbAbs = Math.hypot(R[6], R[8]);
  const beta = Math.atan2(R[7], sgn * cbAbs) / DEG_TO_RAD;
  const gamma = Math.atan2(-R[6] * sgn, R[8] * sgn) / DEG_TO_RAD;
  let alpha = Math.atan2(-R[1] * sgn, R[4] * sgn) / DEG_TO_RAD;
  if (alpha < 0) alpha += 360;
  if (alpha >= 360) alpha -= 360;
  let b = beta;
  if (b >= 180) b -= 360;
  let g = gamma;
  if (g >= 90) g -= 180; // boundary case (cβ sγ = ±|cβ|): equivalent representation
  return { alpha, beta: b, gamma: g };
}

/**
 * Unit "up" (opposite gravity) in the device frame from β, γ (degrees):
 * up_dev = Rᵀ·(0,0,1) = (−cos β sin γ, sin β, cos β cos γ). Flat face-up ⇒ (0,0,1);
 * upright portrait ⇒ (0,1,0).
 */
export function deviceUpFromOrientation(betaDeg: number, gammaDeg: number, out: Vec3 = new Float64Array(3)): Vec3 {
  const b = betaDeg * DEG_TO_RAD;
  const g = gammaDeg * DEG_TO_RAD;
  const cb = Math.cos(b);
  out[0] = -cb * Math.sin(g);
  out[1] = Math.sin(b);
  out[2] = cb * Math.cos(g);
  return out;
}

/** Compass-style azimuth (degrees clockwise from +Y "north", towards +X "east"), [0, 360). */
export function azimuthDeg(x: number, y: number): number {
  let a = Math.atan2(x, y) / DEG_TO_RAD;
  if (a < 0) a += 360;
  if (a >= 360) a -= 360;
  return a;
}

// ---------------------------------------------------------------------------------------------
// Calibration (how raw values map to the ImuSample convention)
// ---------------------------------------------------------------------------------------------

export interface MotionCalibration {
  /** Multiply raw `accelerationIncludingGravity` by this to get the spec sign (+z face-up). */
  accelSign: 1 | -1;
  accelSignSource: 'prior' | 'orientation' | 'heuristic';
  /** Multiply raw `acceleration` (gravity-free) by this to get the spec sign. */
  linearAccelSign: 1 | -1;
  linearAccelSignSource: 'prior' | 'measured';
  /** rad/s per raw rotationRate unit (π/180 for the standard deg/s). */
  gyroScale: number;
  /** Signed permutation: gyro = gyroScale · gyroMap · (alpha, beta, gamma). Identity per spec. */
  gyroMap: Mat3;
  gyroSource: 'default' | 'verified' | 'corrected';
  /** Incremented on every change (consumers that cached normalised samples can detect it). */
  version: number;
}

export function defaultCalibration(accelSignPrior: 1 | -1 = 1): MotionCalibration {
  return {
    accelSign: accelSignPrior,
    accelSignSource: 'prior',
    linearAccelSign: accelSignPrior,
    linearAccelSignSource: 'prior',
    gyroScale: DEG_TO_RAD,
    gyroMap: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
    gyroSource: 'default',
    version: 0,
  };
}

/** Normalised gyro (rad/s, device frame) of one raw record. */
export function rawGyro(raw: ArrayLike<number>, off: number, cal: MotionCalibration, out: Vec3 = new Float64Array(3)): Vec3 {
  const m = cal.gyroMap;
  const s = cal.gyroScale;
  const a = raw[off],
    b = raw[off + 1],
    c = raw[off + 2];
  out[0] = s * (m[0] * a + m[1] * b + m[2] * c);
  out[1] = s * (m[3] * a + m[4] * b + m[5] * c);
  out[2] = s * (m[6] * a + m[7] * b + m[8] * c);
  return out;
}

/** Build an `ImuSample` from a raw record (layout {@link RAW_STRIDE}) and a calibration. */
export function rawToImuSample(t: number, raw: ArrayLike<number>, off: number, cal: MotionCalibration): ImuSample {
  const m = cal.gyroMap;
  const s = cal.gyroScale;
  const a = raw[off],
    b = raw[off + 1],
    c = raw[off + 2];
  const flags = raw[off + 9];
  const la = cal.linearAccelSign;
  const ga = cal.accelSign;
  return {
    t,
    gyro: [s * (m[0] * a + m[1] * b + m[2] * c), s * (m[3] * a + m[4] * b + m[5] * c), s * (m[6] * a + m[7] * b + m[8] * c)],
    accel: flags & RAW_HAS_ACCEL ? [la * raw[off + 3], la * raw[off + 4], la * raw[off + 5]] : null,
    accelG: flags & RAW_HAS_ACCELG ? [ga * raw[off + 6], ga * raw[off + 7], ga * raw[off + 8]] : null,
  };
}

/** One-shot normalisation of a raw event with a given calibration (no self-checks, no dedupe). */
export function normalizeMotionEvent(e: RawMotionEvent, cal: MotionCalibration = defaultCalibration()): ImuSample | null {
  const raw = new Float64Array(RAW_STRIDE);
  if (!writeRaw(e, raw, 0)) return null;
  return rawToImuSample(e.timeStamp, raw, 0, cal);
}

/** Copy the raw values of an event into a record; false if rotationRate is unusable. */
export function writeRaw(e: RawMotionEvent, out: Float64Array, off: number): boolean {
  const r = e.rotationRate;
  if (!r) return false;
  const ra = r.alpha,
    rb = r.beta,
    rg = r.gamma;
  if (!Number.isFinite(ra as number) || !Number.isFinite(rb as number) || !Number.isFinite(rg as number)) return false;
  out[off] = ra as number;
  out[off + 1] = rb as number;
  out[off + 2] = rg as number;
  let flags = 0;
  const al = e.acceleration;
  if (finite3(al)) {
    out[off + 3] = al!.x as number;
    out[off + 4] = al!.y as number;
    out[off + 5] = al!.z as number;
    flags |= RAW_HAS_ACCEL;
  } else {
    out[off + 3] = out[off + 4] = out[off + 5] = 0;
  }
  const ag = e.accelerationIncludingGravity;
  if (finite3(ag)) {
    out[off + 6] = ag!.x as number;
    out[off + 7] = ag!.y as number;
    out[off + 8] = ag!.z as number;
    flags |= RAW_HAS_ACCELG;
  } else {
    out[off + 6] = out[off + 7] = out[off + 8] = 0;
  }
  out[off + 9] = flags;
  return true;
}

// ---------------------------------------------------------------------------------------------
// MotionNormalizer: acceptance (first-event replay, duplicates) + self-checks
// ---------------------------------------------------------------------------------------------

export interface MotionNormalizerOptions {
  /** Prior for the accelerometer sign until check B decides (default +1 = spec). */
  accelSignPrior?: 1 | -1;
  /** Drop bit-identical repeats: 'auto' (default) = only when values do not look quantised. */
  dedupe?: 'auto' | boolean;
  /** Drop the first event after reset() (WebKit replays the last cached sample). Default true. */
  dropFirst?: boolean;
  /** Run the accelerometer sign checks (B, B', B''). Default true. */
  signCheck?: boolean;
  /** Run the gyro axis/unit check (D). Default true. */
  gyroCheck?: boolean;
  /** Max |t_motion − t_orientation| (ms) for pairing. Default 50. */
  pairToleranceMs?: number;
}

export interface MotionNormalizerStats {
  accepted: number;
  firstDropped: number;
  duplicates: number;
  noGyro: number;
  /** Check B votes (orientation-based): raw specific force along +up / −up. */
  signVotesUp: number;
  signVotesDown: number;
  /** Check B' votes (UI-up heuristic). */
  heuristicVotesUp: number;
  heuristicVotesDown: number;
  /** Check B'' votes (acceleration vs accelerationIncludingGravity). */
  linearVotesSame: number;
  linearVotesOpposite: number;
  /** Check D accumulated pairs and the last evaluation. */
  gyroPairs: number;
  gyroCheck: {
    scale: number;
    correlation: number;
    map: number[];
    /** Smallest per-axis correlation of the matched axes (NaN while not every axis is excited). */
    minAxisCorrelation: number;
    /** Every device axis carried ≥ 2 % of the rotation energy (needed to decide the mapping). */
    excited: boolean;
  } | null;
}

const IDENTITY: readonly number[] = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const PERMS: readonly (readonly [number, number, number])[] = [
  [0, 1, 2],
  [0, 2, 1],
  [1, 0, 2],
  [1, 2, 0],
  [2, 0, 1],
  [2, 1, 0],
];

/**
 * Stateful acceptance + self-calibration of DeviceMotion events. DOM-free; the browser
 * glue lives in motion.ts. Usage: call `reset()` whenever the listener is (re)attached, feed
 * every 'deviceorientation' event to `pushOrientation` and every 'devicemotion' event to
 * `acceptMotion` (which writes the raw record when the event is accepted).
 */
export class MotionNormalizer {
  private cal: MotionCalibration;
  private readonly opts: Required<MotionNormalizerOptions>;
  private firstPending = true;
  private readonly last = new Float64Array(RAW_STRIDE);
  private hasLast = false;
  private quantised = 0; // running evidence that values are quantised (spec 0.1 precision limit)
  private readonly listeners = new Set<(c: MotionCalibration) => void>();
  readonly stats: MotionNormalizerStats = {
    accepted: 0,
    firstDropped: 0,
    duplicates: 0,
    noGyro: 0,
    signVotesUp: 0,
    signVotesDown: 0,
    heuristicVotesUp: 0,
    heuristicVotesDown: 0,
    linearVotesSame: 0,
    linearVotesOpposite: 0,
    gyroPairs: 0,
    gyroCheck: null,
  };

  // Orientation state (latest event) for checks B and D.
  private oriT = -Infinity;
  private oriBeta = 0;
  private oriGamma = 0;
  private oriAlpha = 0;
  private readonly up = new Float64Array(3);
  private readonly Rprev = new Float64Array(9);
  private readonly Rcur = new Float64Array(9);
  private readonly dR = new Float64Array(9);
  private readonly rv = new Float64Array(3);

  // Latest accepted motion for check D pairing.
  private lastGyroT = -Infinity;
  private readonly lastGyroRaw = new Float64Array(3);

  // Check B'' state: low-pass of raw accelG.
  private readonly lp = new Float64Array(3);
  private lpT = -Infinity;
  private linearRel: 1 | -1 = 1;

  // Check D accumulators (M = Σ w_est ⊗ gyro_raw, per-axis energies Sw / Sg).
  private readonly M = new Float64Array(9);
  private readonly Sw = new Float64Array(3);
  private readonly Sg = new Float64Array(3);
  private sgg = 0;
  private sww = 0;
  private gyroAttempts = 0;
  private gyroDecided = false;

  constructor(opts: MotionNormalizerOptions = {}) {
    this.opts = {
      accelSignPrior: opts.accelSignPrior ?? 1,
      dedupe: opts.dedupe ?? 'auto',
      dropFirst: opts.dropFirst ?? true,
      signCheck: opts.signCheck ?? true,
      gyroCheck: opts.gyroCheck ?? true,
      pairToleranceMs: opts.pairToleranceMs ?? 50,
    };
    this.cal = defaultCalibration(this.opts.accelSignPrior);
    this.firstPending = this.opts.dropFirst;
  }

  get calibration(): MotionCalibration {
    return this.cal;
  }

  /** Subscribe to calibration changes. Returns an unsubscribe function. */
  onChange(cb: (c: MotionCalibration) => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }

  /** Call when the listener is (re)attached: re-arms the first-event drop and duplicate state. */
  reset(): void {
    this.firstPending = this.opts.dropFirst;
    this.hasLast = false;
    this.oriT = -Infinity;
    this.lastGyroT = -Infinity;
    this.lpT = -Infinity;
  }

  /** Replace the calibration (e.g. restore a verified one). */
  setCalibration(c: Partial<Omit<MotionCalibration, 'version'>>): void {
    this.cal = { ...this.cal, ...c, gyroMap: c.gyroMap ? new Float64Array(c.gyroMap) : this.cal.gyroMap, version: this.cal.version + 1 };
    this.emit();
  }

  private emit(): void {
    for (const cb of this.listeners) {
      try {
        cb(this.cal);
      } catch (e) {
        console.error(e);
      }
    }
  }

  private update(c: Partial<MotionCalibration>): void {
    this.cal = { ...this.cal, ...c, version: this.cal.version + 1 };
    this.emit();
  }

  /** Feed a DeviceOrientation event (used for checks B and D). */
  pushOrientation(o: RawOrientationEvent): void {
    const a = o.alpha,
      b = o.beta,
      g = o.gamma;
    if (!Number.isFinite(a as number) || !Number.isFinite(b as number) || !Number.isFinite(g as number)) return;
    const t = o.timeStamp;
    // Check D: body rate from consecutive orientations, paired with the latest gyro sample.
    if (this.opts.gyroCheck && !this.gyroDecided && Number.isFinite(this.oriT)) {
      const dt = t - this.oriT;
      if (dt >= 5 && dt <= 60 && Math.abs(t - this.lastGyroT) <= 30) {
        deviceOrientationMatrix(this.oriAlpha, this.oriBeta, this.oriGamma, this.Rprev);
        deviceOrientationMatrix(a as number, b as number, g as number, this.Rcur);
        // dR = Rprevᵀ · Rcur
        const P = this.Rprev,
          C = this.Rcur,
          D = this.dR;
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) D[3 * i + j] = P[i] * C[j] + P[3 + i] * C[3 + j] + P[6 + i] * C[6 + j];
        }
        mat3ToRotvec(D, this.rv);
        const inv = 1000 / dt;
        const wx = this.rv[0] * inv,
          wy = this.rv[1] * inv,
          wz = this.rv[2] * inv;
        const gx = this.lastGyroRaw[0] * DEG_TO_RAD,
          gy = this.lastGyroRaw[1] * DEG_TO_RAD,
          gz = this.lastGyroRaw[2] * DEG_TO_RAD;
        const ww = wx * wx + wy * wy + wz * wz;
        if (ww > 0.16 && ww < 400) this.accumulateGyro(wx, wy, wz, gx, gy, gz, ww);
      }
    }
    this.oriT = t;
    this.oriAlpha = a as number;
    this.oriBeta = b as number;
    this.oriGamma = g as number;
  }

  private accumulateGyro(wx: number, wy: number, wz: number, gx: number, gy: number, gz: number, ww: number): void {
    const M = this.M;
    M[0] += wx * gx;
    M[1] += wx * gy;
    M[2] += wx * gz;
    M[3] += wy * gx;
    M[4] += wy * gy;
    M[5] += wy * gz;
    M[6] += wz * gx;
    M[7] += wz * gy;
    M[8] += wz * gz;
    this.Sw[0] += wx * wx;
    this.Sw[1] += wy * wy;
    this.Sw[2] += wz * wz;
    this.Sg[0] += gx * gx;
    this.Sg[1] += gy * gy;
    this.Sg[2] += gz * gz;
    this.sgg += gx * gx + gy * gy + gz * gz;
    this.sww += ww;
    this.stats.gyroPairs++;
    if (this.stats.gyroPairs % 60 === 0) this.evaluateGyro();
  }

  private resetGyroAccumulators(): void {
    this.M.fill(0);
    this.Sw.fill(0);
    this.Sg.fill(0);
    this.sgg = this.sww = 0;
  }

  private evaluateGyro(): void {
    const M = this.M;
    let best = -Infinity;
    let bestPerm = PERMS[0];
    const bestSign = [1, 1, 1];
    for (const p of PERMS) {
      const score = Math.abs(M[p[0]]) + Math.abs(M[3 + p[1]]) + Math.abs(M[6 + p[2]]);
      if (score > best) {
        best = score;
        bestPerm = p;
        bestSign[0] = M[p[0]] >= 0 ? 1 : -1;
        bestSign[1] = M[3 + p[1]] >= 0 ? 1 : -1;
        bestSign[2] = M[6 + p[2]] >= 0 ? 1 : -1;
      }
    }
    const k = best / Math.max(this.sgg, 1e-30);
    const corr = (best * best) / Math.max(this.sgg * this.sww, 1e-30);
    const map = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let i = 0; i < 3; i++) map[3 * i + bestPerm[i]] = bestSign[i];
    // Every device axis must have been excited: otherwise the permutation and the signs of the
    // quiet axes are decided by noise. Panning with the phone upright (THE common capture motion)
    // rotates about device y only — without this gate it "corrected" x/z into a random signed
    // permutation in most runs.
    let excited = true;
    let minAxisCorr = Infinity;
    for (let i = 0; i < 3; i++) {
      const j = bestPerm[i];
      if (!(this.Sw[i] >= 0.02 * this.sww && this.Sg[j] >= 0.02 * this.sgg)) excited = false;
      else minAxisCorr = Math.min(minAxisCorr, Math.abs(M[3 * i + j]) / Math.sqrt(this.Sw[i] * this.Sg[j]));
    }
    this.stats.gyroCheck = { scale: k, correlation: Math.sqrt(corr), map, minAxisCorrelation: excited ? minAxisCorr : NaN, excited };
    if (corr < 0.7) {
      // Inconclusive (e.g. lagging/independent orientation source): restart, a few attempts.
      if (++this.gyroAttempts >= 5) this.gyroDecided = true;
      this.resetGyroAccumulators();
      return;
    }
    if (!excited) return; // keep accumulating until the motion has covered every axis
    let scale: number | null = null;
    if (k > 0.6 && k < 1.6) scale = DEG_TO_RAD;
    else if (k > 40 && k < 80) scale = 1; // platform reported rad/s
    if (scale === null || minAxisCorr < 0.8) {
      if (++this.gyroAttempts >= 5) this.gyroDecided = true;
      this.resetGyroAccumulators();
      return;
    }
    this.gyroDecided = true;
    const isIdentity = map.every((v, i) => v === IDENTITY[i]);
    if (isIdentity && scale === DEG_TO_RAD) {
      this.update({ gyroSource: 'verified' });
    } else {
      console.warn('[sensors] rotationRate deviates from the spec; corrected', { map, scale: k });
      this.update({ gyroSource: 'corrected', gyroMap: new Float64Array(map), gyroScale: scale });
    }
  }

  /**
   * Validate an event and write its raw record into `out[off .. off+RAW_STRIDE)`. Returns
   * false for the replayed first event, exact duplicates and events without rotationRate.
   * `screenAngleDeg` (UI rotation θ) is only used by the fallback sign heuristic B'.
   */
  acceptMotion(e: RawMotionEvent, screenAngleDeg: number, out: Float64Array, off: number): boolean {
    if (!writeRaw(e, out, off)) {
      this.stats.noGyro++;
      return false;
    }
    if (this.firstPending) {
      this.firstPending = false;
      this.stats.firstDropped++;
      this.copyLast(out, off);
      return false;
    }
    // Duplicate detection.
    const q = isQuantised(out[off]) && isQuantised(out[off + 1]) && isQuantised(out[off + 2]);
    this.quantised = q ? Math.min(this.quantised + 1, 50) : Math.max(this.quantised - 5, 0);
    const dedupe = this.opts.dedupe === 'auto' ? this.quantised < 10 : this.opts.dedupe;
    if (dedupe && this.hasLast) {
      let same = true;
      for (let i = 0; i < 9; i++) {
        if (out[off + i] !== this.last[i]) {
          same = false;
          break;
        }
      }
      if (same) {
        this.stats.duplicates++;
        return false;
      }
    }
    this.copyLast(out, off);
    this.stats.accepted++;
    const t = e.timeStamp;
    this.lastGyroT = t;
    this.lastGyroRaw[0] = out[off];
    this.lastGyroRaw[1] = out[off + 1];
    this.lastGyroRaw[2] = out[off + 2];
    if (this.opts.signCheck) this.signChecks(t, out, off, screenAngleDeg);
    return true;
  }

  private copyLast(out: Float64Array, off: number): void {
    for (let i = 0; i < RAW_STRIDE; i++) this.last[i] = out[off + i];
    this.hasLast = true;
  }

  private signChecks(t: number, raw: Float64Array, off: number, screenAngleDeg: number): void {
    const flags = raw[off + 9];
    if (!(flags & RAW_HAS_ACCELG)) return;
    const gx = raw[off + 6],
      gy = raw[off + 7],
      gz = raw[off + 8];
    const gn = Math.hypot(gx, gy, gz);
    if (!(gn > 1e-6)) return;
    const s = this.cal.gyroScale;
    const wn = s * Math.hypot(raw[off], raw[off + 1], raw[off + 2]);
    const st = this.stats;

    // ---- B'': sign of `acceleration` relative to `accelerationIncludingGravity` ----
    if (flags & RAW_HAS_ACCEL) {
      if (!Number.isFinite(this.lpT)) {
        this.lp[0] = gx;
        this.lp[1] = gy;
        this.lp[2] = gz;
        this.lpT = t;
      } else if (t >= this.lpT) {
        // t === lpT: a burst after a main-thread stall (1 ms timeStamp); votes, no low-pass step
        const dt = Math.min(t - this.lpT, 200);
        const hx = gx - this.lp[0],
          hy = gy - this.lp[1],
          hz = gz - this.lp[2];
        const hn = Math.hypot(hx, hy, hz);
        const ax = raw[off + 3],
          ay = raw[off + 4],
          az = raw[off + 5];
        const an = Math.hypot(ax, ay, az);
        if (wn < 0.3 && hn > 0.8 && an > 0.5) {
          const d = (ax * hx + ay * hy + az * hz) / (an * hn);
          if (d > 0.5) st.linearVotesSame++;
          else if (d < -0.5) st.linearVotesOpposite++;
          if (this.cal.linearAccelSignSource === 'prior') {
            const tot = st.linearVotesSame + st.linearVotesOpposite;
            if (tot >= 40) {
              let rel: 1 | -1 | 0 = 0;
              if (st.linearVotesSame >= 0.85 * tot) rel = 1;
              else if (st.linearVotesOpposite >= 0.85 * tot) rel = -1;
              if (rel !== 0) {
                this.linearRel = rel;
                this.update({ linearAccelSign: (this.cal.accelSign * rel) as 1 | -1, linearAccelSignSource: 'measured' });
              }
            }
          }
        }
        const k = dt / (1000 + dt);
        this.lp[0] += k * (gx - this.lp[0]);
        this.lp[1] += k * (gy - this.lp[1]);
        this.lp[2] += k * (gz - this.lp[2]);
        this.lpT = t;
      } // t < lpT (out-of-order timestamp): skipped, the low-pass state is kept
    }

    // Near-stillness gate for the gravity-direction checks.
    if (wn > 0.5 || Math.abs(gn - STANDARD_GRAVITY) > 1.0) return;

    // ---- B: orientation-based (primary) ----
    if (Math.abs(t - this.oriT) <= this.opts.pairToleranceMs) {
      deviceUpFromOrientation(this.oriBeta, this.oriGamma, this.up);
      const c = (gx * this.up[0] + gy * this.up[1] + gz * this.up[2]) / gn;
      if (c > 0.85) st.signVotesUp++;
      else if (c < -0.85) st.signVotesDown++;
      const tot = st.signVotesUp + st.signVotesDown;
      const decided = this.cal.accelSignSource === 'orientation';
      // First decision after 30 consistent votes (0.5 s at 60 Hz); a decided sign is only
      // overturned by overwhelming evidence (≥ 300 votes, ≥ 90 % opposite).
      const minVotes = decided ? 300 : 30;
      if (tot >= minVotes) {
        let sign: 1 | -1 | 0 = 0;
        if (st.signVotesUp >= 0.9 * tot) sign = 1;
        else if (st.signVotesDown >= 0.9 * tot) sign = -1;
        if (sign !== 0 && (!decided || sign !== this.cal.accelSign)) this.setAccelSign(sign, 'orientation');
      }
      return;
    }

    // ---- B': UI-up heuristic (no orientation events) ----
    if (this.cal.accelSignSource !== 'prior') return;
    const r = (screenAngleDeg * Math.PI) / 180;
    const ux = Math.sin(r),
      uy = Math.cos(r);
    const c = (gx * ux + gy * uy) / gn;
    if (c > 0.7) st.heuristicVotesUp++;
    else if (c < -0.7) st.heuristicVotesDown++;
    const tot = st.heuristicVotesUp + st.heuristicVotesDown;
    if (tot >= 120) {
      if (st.heuristicVotesUp >= 0.9 * tot) this.setAccelSign(1, 'heuristic');
      else if (st.heuristicVotesDown >= 0.9 * tot) this.setAccelSign(-1, 'heuristic');
    }
  }

  private setAccelSign(sign: 1 | -1, source: 'orientation' | 'heuristic'): void {
    if (sign !== this.cal.accelSign && this.cal.accelSignSource !== 'prior') {
      console.warn('[sensors] accelerometer sign re-decided', { from: this.cal.accelSign, to: sign, source });
    }
    const linear = this.cal.linearAccelSignSource === 'measured' ? ((sign * this.linearRel) as 1 | -1) : sign;
    this.update({ accelSign: sign, accelSignSource: source, linearAccelSign: linear });
  }
}

function isQuantised(v: number): boolean {
  const x = v * 10;
  return Math.abs(x - Math.round(x)) < 1e-9;
}
