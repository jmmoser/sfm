/**
 * DeviceMotion capture: permission flow, normalisation to `ImuSample` (via normalize.ts),
 * a 60 s ring buffer with burst re-timing, gyro integration helpers.
 *
 * Timing (docs/research/ios-sensors.md §1.3, §1.8): iOS polls CoreMotion at a fixed ~60 Hz
 * and `event.timeStamp` is the event creation time in the web process (performance.now
 * timeline, after IPC). When the main thread stalls, the queued events are dispatched
 * back-to-back with almost identical timestamps; the ring buffer re-times such bursts by
 * spacing them at the nominal interval, ending at the newest (least delayed) arrival.
 * The remaining constant latency (~10–20 ms) is left to the camera–IMU time-offset estimation.
 */
import type { ImuSample, Mat3, Vec3 } from '../core/types';
import { screenAngle } from './frame-orientation';
import {
  MotionNormalizer,
  RAW_HAS_ACCELG,
  RAW_STRIDE,
  deviceUpFromOrientation,
  normalizeIntervalMs,
  rawGyro,
  rawToImuSample,
  type MotionCalibration,
  type MotionNormalizerOptions,
  type MotionNormalizerStats,
  type RawMotionEvent,
  type RawOrientationEvent,
} from './normalize';

export type MotionPermission = 'granted' | 'denied' | 'unsupported';

// ---------------------------------------------------------------------------------------------
// Ring buffer
// ---------------------------------------------------------------------------------------------

export interface ImuBufferStats {
  /** Samples whose time was moved earlier by burst re-timing. */
  retimed: number;
  /** Arrivals closer than 0.35 × nominal interval to the previous one. */
  bursts: number;
  /** Arrivals with a timestamp before the previous arrival (clamped). */
  nonMonotonic: number;
  /** Largest gap between consecutive arrivals (ms). */
  maxGapMs: number;
}

/**
 * Fixed-capacity ring buffer of raw IMU records (layout `RAW_STRIDE`, see normalize.ts) with
 * estimated sample times. Absolute indices grow monotonically; index i is valid while
 * `first <= i < end`. DOM-free.
 */
export class ImuRingBuffer {
  readonly capacity: number;
  readonly raw: Float64Array;
  private readonly times: Float64Array;
  private readonly arrivals: Float64Array;
  private head = 0;
  private frozen = 0;
  private readonly dts = new Float64Array(64);
  private dtCount = 0;
  private nominalSet = false;
  /** Nominal sample interval (ms): median of recent arrival intervals (or the event `interval`). */
  nominalMs = 1000 / 60;
  readonly stats: ImuBufferStats = { retimed: 0, bursts: 0, nonMonotonic: 0, maxGapMs: 0 };

  constructor(capacity = 6000) {
    this.capacity = Math.max(16, Math.floor(capacity));
    this.raw = new Float64Array(this.capacity * RAW_STRIDE);
    this.times = new Float64Array(this.capacity);
    this.arrivals = new Float64Array(this.capacity);
  }

  /** Absolute index of the oldest retained sample. */
  get first(): number {
    return Math.max(0, this.head - this.capacity);
  }
  /** Absolute index one past the newest sample (= total number pushed). */
  get end(): number {
    return this.head;
  }
  get size(): number {
    return this.head - this.first;
  }
  timeAt(i: number): number {
    return this.times[i % this.capacity];
  }
  arrivalAt(i: number): number {
    return this.arrivals[i % this.capacity];
  }
  rawOffset(i: number): number {
    return (i % this.capacity) * RAW_STRIDE;
  }

  /** Seed the nominal interval (e.g. from DeviceMotionEvent.interval) until enough data exist. */
  setNominal(ms: number): void {
    if (!this.nominalSet && ms >= 2 && ms <= 200) {
      this.nominalMs = ms;
      this.nominalSet = true;
    }
  }

  /** Times of samples with absolute index < i are never changed again (they were handed out). */
  freeze(i: number): void {
    if (i > this.frozen) this.frozen = i;
  }

  clear(): void {
    this.head = 0;
    this.frozen = 0;
    this.dtCount = 0;
  }

  /** Append one raw record (RAW_STRIDE values starting at `src[srcOff]`) that arrived at `arrivalT`. */
  push(arrivalT: number, src: ArrayLike<number>, srcOff = 0): void {
    const cap = this.capacity;
    const n = this.head;
    let ta = arrivalT;
    const hasPrev = n > 0;
    if (hasPrev) {
      const pa = this.arrivals[(n - 1) % cap];
      if (ta < pa) {
        this.stats.nonMonotonic++;
        ta = pa;
      }
      const dt = ta - pa;
      if (dt > this.stats.maxGapMs) this.stats.maxGapMs = dt;
      if (dt > 3 && dt < 200) {
        this.dts[this.dtCount % this.dts.length] = dt;
        this.dtCount++;
        if (this.dtCount >= 16 && this.dtCount % 16 === 0) this.updateNominal();
      }
    }
    const slot = n % cap;
    const ro = slot * RAW_STRIDE;
    for (let k = 0; k < RAW_STRIDE; k++) this.raw[ro + k] = src[srcOff + k];
    this.arrivals[slot] = ta;
    this.times[slot] = ta;
    this.head = n + 1;
    if (hasPrev && ta - this.arrivals[(n - 1) % cap] < 0.35 * this.nominalMs) {
      this.stats.bursts++;
      this.retimeBack(n);
    }
  }

  private updateNominal(): void {
    const m = Math.min(this.dtCount, this.dts.length);
    const tmp = Array.from(this.dts.subarray(0, m)).sort((a, b) => a - b);
    const med = tmp[m >> 1];
    if (med >= 2 && med <= 200) this.nominalMs = med;
  }

  /**
   * Re-time the burst run ending at absolute index `last`: the samples that arrived back-to-back
   * (arrival gaps < 0.35 × nominal) are spread at the nominal interval ending at the newest
   * arrival (the least delayed one); evenly compressed if that would overlap the sample before
   * the run. Times only ever move earlier; frozen (drained) samples are never touched.
   */
  private retimeBack(last: number): void {
    const cap = this.capacity;
    const nom = this.nominalMs;
    const lo = Math.max(this.first, this.frozen);
    let s = last;
    while (s - 1 >= lo && last - s < 64 && this.arrivals[s % cap] - this.arrivals[(s - 1) % cap] < 0.35 * nom) s--;
    if (s === last) return;
    const tEnd = this.arrivals[last % cap];
    const tBefore = s - 1 >= this.first ? this.times[(s - 1) % cap] : -Infinity;
    const n = last - s;
    let step = nom;
    if (tEnd - n * nom <= tBefore + 0.5 * nom) step = (tEnd - tBefore) / (n + 1);
    for (let k = s; k < last; k++) {
      const nt = tEnd - (last - k) * step;
      if (nt < this.times[k % cap]) {
        this.times[k % cap] = nt;
        this.stats.retimed++;
      }
    }
  }

  /** Smallest absolute index i in [first, end] with timeAt(i) >= t. */
  lowerBound(t: number): number {
    let lo = this.first;
    let hi = this.head;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.times[mid % this.capacity] < t) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
}

// ---------------------------------------------------------------------------------------------
// Gyro integration (pure)
// ---------------------------------------------------------------------------------------------

function upperBoundSamples(s: readonly ImuSample[], t: number): number {
  let lo = 0;
  let hi = s.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (s[mid].t <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

const MAX_CUBIC_SPAN_MS = 100;

/**
 * ω at time t inside sample interval (i−1, i) (i = first index with s[i].t > t): cubic Hermite
 * with node tangents from the 3-point parabola (2nd-order on non-uniform grids), which reduces the interpolation
 * error of smooth hand motion by ~2 orders of magnitude vs. linear interpolation. Falls back to
 * the secant (linear) at the ends of the data and across gaps (span > 100 ms or a neighbouring
 * span > 2.5× the current one). Holds the first/last sample outside the data.
 */
function rateIn(s: readonly ImuSample[], i: number, t: number, out: number[]): void {
  const n = s.length;
  if (i <= 0 || i >= n) {
    const g = s[i <= 0 ? 0 : n - 1].gyro;
    out[0] = g[0];
    out[1] = g[1];
    out[2] = g[2];
    return;
  }
  const a = s[i - 1];
  const b = s[i];
  const h = b.t - a.t;
  if (!(h > 0)) {
    out[0] = b.gyro[0];
    out[1] = b.gyro[1];
    out[2] = b.gyro[2];
    return;
  }
  const u = Math.min(1, Math.max(0, (t - a.t) / h));
  const wa = a.gyro;
  const wb = b.gyro;
  const cubic = h <= MAX_CUBIC_SPAN_MS;
  const p = i >= 2 ? s[i - 2] : null;
  const q = i + 1 < n ? s[i + 1] : null;
  const useP = cubic && p !== null && a.t - p.t > 0 && a.t - p.t <= 2.5 * h;
  const useQ = cubic && q !== null && q.t - b.t > 0 && q.t - b.t <= 2.5 * h;
  const u2 = u * u;
  const u3 = u2 * u;
  const h00 = 2 * u3 - 3 * u2 + 1;
  const h10 = u3 - 2 * u2 + u;
  const h01 = -2 * u3 + 3 * u2;
  const h11 = u3 - u2;
  const h0 = useP ? a.t - p!.t : 1;
  const h2 = useQ ? q!.t - b.t : 1;
  for (let k = 0; k < 3; k++) {
    const s1 = (wb[k] - wa[k]) / h;
    // Node derivatives of the 3-point parabola (2nd order on non-uniform grids), scaled by h.
    const ma = useP ? ((h * (wa[k] - p!.gyro[k])) / h0 + h0 * s1) / (h0 + h) * h : s1 * h;
    const mb = useQ ? ((h2 * s1 + (h * (q!.gyro[k] - wb[k])) / h2) / (h + h2)) * h : s1 * h;
    out[k] = h00 * wa[k] + h10 * ma + h01 * wb[k] + h11 * mb;
  }
}

const GAUSS_A = 0.5 - Math.sqrt(3) / 6;
const GAUSS_B = 0.5 + Math.sqrt(3) / 6;
const MAGNUS_C = Math.sqrt(3) / 12;

/**
 * Device rotation R_{d(t1)←d(t0)} (v_{d(t1)} = R · v_{d(t0)} for a world-fixed direction)
 * from gyro samples (rad/s, device frame, sorted by t). With dR_wd/dt = R_wd [ω]× this is
 * R_wd(t1)ᵀ · R_wd(t0).
 *
 * ω(t) between samples is a cubic Hermite interpolant (linear at data ends / gaps, held
 * constant outside the data); each sub-interval [ta, tb] (split at sample times and at t0, t1)
 * is advanced with the 4th-order Gauss–Legendre Magnus step for right-multiplied body rates:
 * φ = h/2 (ω₁ + ω₂) + (√3/12) h² (ω₁ × ω₂), ω₁,₂ = ω(ta + (½ ∓ √3/6) h)  (for linear ω this
 * reduces to the exact h(ωa+ωb)/2 + (h²/12) ωa×ωb). Measured error on smooth hand-like motion
 * at 60 Hz: ~5e-6 rad over 2.5 s (linear interpolation + trapezoid: ~1.4e-3 rad).
 * t1 < t0 returns the inverse rotation. No samples ⇒ identity.
 */
export function integrateGyro(samples: readonly ImuSample[], t0: number, t1: number, out: Mat3 = new Float64Array(9)): Mat3 {
  out.fill(0);
  out[0] = out[4] = out[8] = 1;
  const n = samples.length;
  if (n === 0 || !(t0 !== t1) || !Number.isFinite(t0) || !Number.isFinite(t1)) return out;
  if (t1 < t0) {
    integrateGyro(samples, t1, t0, out);
    let tmp = out[1];
    out[1] = out[3];
    out[3] = tmp;
    tmp = out[2];
    out[2] = out[6];
    out[6] = tmp;
    tmp = out[5];
    out[5] = out[7];
    out[7] = tmp;
    return out;
  }
  // q accumulates R_wd(t0)ᵀ R_wd(t) (right-multiplied steps).
  let qw = 1,
    qx = 0,
    qy = 0,
    qz = 0;
  const g1 = [0, 0, 0];
  const g2 = [0, 0, 0];
  let i = upperBoundSamples(samples, t0); // first sample with t > ta
  let ta = t0;
  while (ta < t1) {
    const tb = i < n && samples[i].t < t1 ? samples[i].t : t1;
    if (tb > ta) {
      const hm = tb - ta;
      rateIn(samples, i, ta + GAUSS_A * hm, g1);
      rateIn(samples, i, ta + GAUSS_B * hm, g2);
      const h = hm / 1000;
      const c = MAGNUS_C * h * h;
      const px = 0.5 * h * (g1[0] + g2[0]) + c * (g1[1] * g2[2] - g1[2] * g2[1]);
      const py = 0.5 * h * (g1[1] + g2[1]) + c * (g1[2] * g2[0] - g1[0] * g2[2]);
      const pz = 0.5 * h * (g1[2] + g2[2]) + c * (g1[0] * g2[1] - g1[1] * g2[0]);
      const th = Math.hypot(px, py, pz);
      let sw: number, sk: number;
      if (th < 1e-8) {
        sw = 1 - (th * th) / 8;
        sk = 0.5;
      } else {
        sw = Math.cos(0.5 * th);
        sk = Math.sin(0.5 * th) / th;
      }
      const sx = sk * px,
        sy = sk * py,
        sz = sk * pz;
      // q ← q ⊗ s
      const nw = qw * sw - qx * sx - qy * sy - qz * sz;
      const nx = qw * sx + qx * sw + qy * sz - qz * sy;
      const ny = qw * sy - qx * sz + qy * sw + qz * sx;
      const nz = qw * sz + qx * sy - qy * sx + qz * sw;
      const inv = 1 / Math.hypot(nw, nx, ny, nz);
      qw = nw * inv;
      qx = nx * inv;
      qy = ny * inv;
      qz = nz * inv;
    }
    ta = tb;
    if (tb >= t1) break;
    i++;
  }
  // out = R(q)ᵀ
  const xx = qx * qx,
    yy = qy * qy,
    zz = qz * qz;
  const xy = qx * qy,
    xz = qx * qz,
    yz = qy * qz;
  const wx = qw * qx,
    wy = qw * qy,
    wz = qw * qz;
  out[0] = 1 - 2 * (yy + zz);
  out[3] = 2 * (xy - wz);
  out[6] = 2 * (xz + wy);
  out[1] = 2 * (xy + wz);
  out[4] = 1 - 2 * (xx + zz);
  out[7] = 2 * (yz - wx);
  out[2] = 2 * (xz - wy);
  out[5] = 2 * (yz + wx);
  out[8] = 1 - 2 * (xx + yy);
  return out;
}

/**
 * |ω| (rad/s) at time t from sorted samples (linear interpolation of the magnitude). NaN when
 * no sample lies within `maxGapMs` (default 100) of t.
 */
export function angularSpeedAt(samples: readonly ImuSample[], t: number, maxGapMs = 100): number {
  const n = samples.length;
  if (n === 0) return NaN;
  const i = upperBoundSamples(samples, t);
  const mag = (s: ImuSample): number => Math.hypot(s.gyro[0], s.gyro[1], s.gyro[2]);
  if (i === 0) return samples[0].t - t <= maxGapMs ? mag(samples[0]) : NaN;
  if (i === n) return t - samples[n - 1].t <= maxGapMs ? mag(samples[n - 1]) : NaN;
  const a = samples[i - 1];
  const b = samples[i];
  const span = b.t - a.t;
  if (span > 2 * maxGapMs && t - a.t > maxGapMs && b.t - t > maxGapMs) return NaN;
  const u = span > 0 ? (t - a.t) / span : 1;
  return mag(a) + u * (mag(b) - mag(a));
}

// ---------------------------------------------------------------------------------------------
// MotionSensor
// ---------------------------------------------------------------------------------------------

export interface MotionSensorOptions extends MotionNormalizerOptions {
  /** Ring-buffer capacity in samples (default 6000 = 60 s at 100 Hz). */
  capacity?: number;
  /** Event source (default: `window` / globalThis). */
  target?: EventTarget | null;
  /** UI rotation θ provider for the fallback sign heuristic (default `screenAngle`). */
  screenAngle?: () => number;
  /** Also listen to 'deviceorientation' (self-checks + gravity). Default true. */
  listenOrientation?: boolean;
}

export interface MotionSensorStats extends MotionNormalizerStats, ImuBufferStats {
  rateHz: number;
  nominalIntervalMs: number;
  buffered: number;
  orientationEvents: number;
}

type AnyCtor = { requestPermission?: () => Promise<string> } | undefined;

function errorName(e: unknown): string {
  return (e as { name?: string })?.name ?? String(e);
}

/**
 * Wraps 'devicemotion' (+ 'deviceorientation' for the self-checks). One permanent listener per
 * session is recommended: removing the last listener restarts CoreMotion on iOS, which resets
 * the DeviceOrientation yaw reference.
 */
export class MotionSensor {
  /** Name of the last permission failure ('NotAllowedError' = called outside a user gesture). */
  static lastPermissionError: string | null = null;

  /** DeviceMotionEvent is exposed (iOS 26.4+: secure contexts only). */
  static isSupported(): boolean {
    return typeof (globalThis as { DeviceMotionEvent?: unknown }).DeviceMotionEvent !== 'undefined';
  }

  /** True on iOS/iPadOS WebKit, where `requestPermission()` must be called from a user gesture. */
  static needsPermission(): boolean {
    const DME = (globalThis as { DeviceMotionEvent?: AnyCtor }).DeviceMotionEvent;
    return typeof DME?.requestPermission === 'function';
  }

  /**
   * Request motion + orientation access. MUST be called synchronously at the top of a
   * `click`/`touchend` handler (before any await): the platform call happens synchronously
   * inside this function. Motion and orientation share one grant per origin on iOS; the
   * orientation request is made after the motion grant (it then resolves without a prompt).
   * 'granted' is also returned on platforms without a permission model (events may still
   * never fire, e.g. desktop — check `rateHz`).
   */
  static requestPermission(): Promise<MotionPermission> {
    const g = globalThis as { DeviceMotionEvent?: AnyCtor; DeviceOrientationEvent?: AnyCtor };
    const DME = g.DeviceMotionEvent;
    if (typeof DME === 'undefined') {
      MotionSensor.lastPermissionError = 'unsupported';
      return Promise.resolve('unsupported');
    }
    const req = DME.requestPermission;
    if (typeof req !== 'function') {
      MotionSensor.lastPermissionError = null;
      return Promise.resolve('granted');
    }
    let p: Promise<string>;
    try {
      p = req.call(DME); // synchronous, inside the user gesture
    } catch (e) {
      MotionSensor.lastPermissionError = errorName(e);
      return Promise.resolve('denied');
    }
    return Promise.resolve(p).then(
      async (state): Promise<MotionPermission> => {
        if (state !== 'granted') {
          MotionSensor.lastPermissionError = 'denied';
          return 'denied';
        }
        const DOE = g.DeviceOrientationEvent;
        if (typeof DOE?.requestPermission === 'function') {
          try {
            await DOE.requestPermission.call(DOE);
          } catch {
            /* shared grant: motion is what matters */
          }
        }
        MotionSensor.lastPermissionError = null;
        return 'granted';
      },
      (e): MotionPermission => {
        MotionSensor.lastPermissionError = errorName(e);
        return 'denied';
      },
    );
  }

  readonly buffer: ImuRingBuffer;
  readonly normalizer: MotionNormalizer;
  private readonly target: EventTarget | null;
  private readonly angleFn: () => number;
  private readonly listenOrientation: boolean;
  private isRunning = false;
  private drainCursor = 0;
  private readonly scratch = new Float64Array(RAW_STRIDE);
  private readonly g3 = new Float64Array(3);
  // Orientation (β, γ) history for gravityAt().
  private readonly oriCap = 512;
  private readonly oriT = new Float64Array(512);
  private readonly oriB = new Float64Array(512);
  private readonly oriG = new Float64Array(512);
  private oriHead = 0;

  private readonly onMotionEvent = (e: Event): void => {
    this.handleMotion(e as unknown as RawMotionEvent);
  };
  private readonly onOrientationEvent = (e: Event): void => {
    this.handleOrientation(e as unknown as RawOrientationEvent);
  };

  constructor(opts: MotionSensorOptions = {}) {
    const prior = opts.accelSignPrior ?? (MotionSensor.needsPermission() ? -1 : 1);
    this.normalizer = new MotionNormalizer({ ...opts, accelSignPrior: prior });
    this.buffer = new ImuRingBuffer(opts.capacity ?? 6000);
    const g = globalThis as unknown as EventTarget & { addEventListener?: unknown };
    this.target = opts.target !== undefined ? opts.target : typeof g.addEventListener === 'function' ? g : null;
    this.angleFn = opts.screenAngle ?? screenAngle;
    this.listenOrientation = opts.listenOrientation ?? true;
  }

  get running(): boolean {
    return this.isRunning;
  }

  /** Attach the listeners (call after permission was granted). */
  start(): void {
    if (this.isRunning) return;
    if (!this.target) {
      console.warn('[sensors] MotionSensor: no event target (not in a browser)');
      return;
    }
    this.isRunning = true;
    this.normalizer.reset();
    this.target.addEventListener('devicemotion', this.onMotionEvent);
    if (this.listenOrientation) this.target.addEventListener('deviceorientation', this.onOrientationEvent);
  }

  stop(): void {
    if (!this.isRunning) return;
    this.isRunning = false;
    this.target?.removeEventListener('devicemotion', this.onMotionEvent);
    if (this.listenOrientation) this.target?.removeEventListener('deviceorientation', this.onOrientationEvent);
  }

  /** Feed one DeviceMotion event (the listener calls this; public for replay/tests). */
  handleMotion(e: RawMotionEvent): boolean {
    this.buffer.setNominal(normalizeIntervalMs(e.interval));
    if (!this.normalizer.acceptMotion(e, this.angleFn(), this.scratch, 0)) return false;
    this.buffer.push(e.timeStamp, this.scratch, 0);
    return true;
  }

  /** Feed one DeviceOrientation event (the listener calls this; public for replay/tests). */
  handleOrientation(e: RawOrientationEvent): void {
    this.normalizer.pushOrientation(e);
    if (!Number.isFinite(e.beta as number) || !Number.isFinite(e.gamma as number)) return;
    const k = this.oriHead % this.oriCap;
    this.oriT[k] = e.timeStamp;
    this.oriB[k] = e.beta as number;
    this.oriG[k] = e.gamma as number;
    this.oriHead++;
  }

  get calibration(): MotionCalibration {
    return this.normalizer.calibration;
  }

  onCalibrationChange(cb: (c: MotionCalibration) => void): () => void {
    return this.normalizer.onChange(cb);
  }

  /** Measured event rate over the last 2 s (0 if fewer than 2 samples). */
  get rateHz(): number {
    const b = this.buffer;
    if (b.size < 2) return 0;
    const tl = b.timeAt(b.end - 1);
    const i0 = Math.max(b.first, Math.min(b.end - 2, b.lowerBound(tl - 2000)));
    const span = tl - b.timeAt(i0);
    return span > 0 ? ((b.end - 1 - i0) * 1000) / span : 0;
  }

  /** Time of the newest buffered sample (NaN if none). */
  get lastSampleTime(): number {
    return this.buffer.size ? this.buffer.timeAt(this.buffer.end - 1) : NaN;
  }

  get sampleCount(): number {
    return this.buffer.size;
  }

  get stats(): MotionSensorStats {
    return {
      ...this.normalizer.stats,
      ...this.buffer.stats,
      rateHz: this.rateHz,
      nominalIntervalMs: this.buffer.nominalMs,
      buffered: this.buffer.size,
      orientationEvents: this.oriHead,
    };
  }

  private sample(i: number): ImuSample {
    return rawToImuSample(this.buffer.timeAt(i), this.buffer.raw, this.buffer.rawOffset(i), this.normalizer.calibration);
  }

  /**
   * Normalised samples with t0 ≤ t ≤ t1 (current calibration). With `pad`, one extra sample
   * before t0 and after t1 is included when available (what `integrateGyro` needs to
   * interpolate at the interval ends).
   */
  samplesBetween(t0: number, t1: number, opts: { pad?: boolean } = {}): ImuSample[] {
    const b = this.buffer;
    if (!(t1 >= t0) || b.size === 0) return [];
    let i0 = b.lowerBound(t0);
    let i1 = b.lowerBound(t1);
    while (i1 < b.end && b.timeAt(i1) <= t1) i1++;
    if (opts.pad) {
      if (i0 > b.first) i0--;
      if (i1 < b.end) i1++;
    }
    const out: ImuSample[] = new Array(Math.max(0, i1 - i0));
    for (let i = i0; i < i1; i++) out[i - i0] = this.sample(i);
    return out;
  }

  /**
   * Samples added since the previous drain (for streaming to the worker); oldest first.
   * The buffer keeps RAW values, so a calibration change (`calibration.version`++) re-signs every
   * sample still buffered, but samples already drained keep the calibration they were drained
   * with: consumers must expect a change mid-stream (PumpFrame.imuCalibrationVersion; the world
   * alignment re-checks the accelerometer sign, src/sfm/world.ts `accelSignCorrected`).
   */
  drain(): ImuSample[] {
    const b = this.buffer;
    const i0 = Math.max(this.drainCursor, b.first);
    const out: ImuSample[] = new Array(Math.max(0, b.end - i0));
    for (let i = i0; i < b.end; i++) out[i - i0] = this.sample(i);
    this.drainCursor = b.end;
    b.freeze(b.end);
    return out;
  }

  /** |ω| (rad/s) at time t, linearly interpolated; NaN if no sample within 100 ms. */
  angularSpeed(t: number): number {
    const b = this.buffer;
    if (b.size === 0) return NaN;
    const cal = this.normalizer.calibration;
    const mag = (i: number): number => {
      rawGyro(b.raw, b.rawOffset(i), cal, this.g3);
      return Math.hypot(this.g3[0], this.g3[1], this.g3[2]);
    };
    const i = b.lowerBound(t);
    if (i <= b.first) return b.timeAt(b.first) - t <= 100 ? mag(b.first) : NaN;
    if (i >= b.end) return t - b.timeAt(b.end - 1) <= 100 ? mag(b.end - 1) : NaN;
    const ta = b.timeAt(i - 1);
    const tb = b.timeAt(i);
    if (t - ta > 100 && tb - t > 100) return NaN;
    const u = tb > ta ? (t - ta) / (tb - ta) : 1;
    const ma = mag(i - 1);
    return ma + u * (mag(i) - ma);
  }

  /** R_{d(t1)←d(t0)} from the buffered gyro (see {@link integrateGyro}). */
  rotationBetween(t0: number, t1: number, out?: Mat3): Mat3 {
    const lo = Math.min(t0, t1);
    const hi = Math.max(t0, t1);
    return integrateGyro(this.samplesBetween(lo, hi, { pad: true }), t0, t1, out);
  }

  /**
   * Unit gravity direction (pointing DOWN) in the device frame at time t: from the
   * DeviceOrientation attitude when an event lies within 100 ms, else from the mean
   * spec-sign specific force over ±250 ms. Null when neither is available.
   */
  gravityAt(t: number, out: Vec3 = new Float64Array(3)): Vec3 | null {
    // Orientation history (nearest event).
    const n = Math.min(this.oriHead, this.oriCap);
    if (n > 0) {
      let best = -1;
      let bestDt = Infinity;
      for (let k = 0; k < n; k++) {
        const idx = (this.oriHead - 1 - k) % this.oriCap;
        const dt = Math.abs(this.oriT[idx] - t);
        if (dt < bestDt) {
          bestDt = dt;
          best = idx;
        }
        if (this.oriT[idx] < t - 200) break;
      }
      if (best >= 0 && bestDt <= 100) {
        deviceUpFromOrientation(this.oriB[best], this.oriG[best], out);
        out[0] = -out[0];
        out[1] = -out[1];
        out[2] = -out[2];
        return out;
      }
    }
    const b = this.buffer;
    if (b.size === 0) return null;
    const i0 = b.lowerBound(t - 250);
    let sx = 0,
      sy = 0,
      sz = 0,
      cnt = 0;
    for (let i = i0; i < b.end && b.timeAt(i) <= t + 250; i++) {
      const o = b.rawOffset(i);
      if (!(b.raw[o + 9] & RAW_HAS_ACCELG)) continue;
      sx += b.raw[o + 6];
      sy += b.raw[o + 7];
      sz += b.raw[o + 8];
      cnt++;
    }
    if (cnt < 3) return null;
    const s = this.normalizer.calibration.accelSign;
    const nrm = Math.hypot(sx, sy, sz);
    if (!(nrm > 0)) return null;
    // Spec-sign specific force points up at rest ⇒ down = −s·mean/|mean|.
    out[0] = (-s * sx) / nrm;
    out[1] = (-s * sy) / nrm;
    out[2] = (-s * sz) / nrm;
    return out;
  }
}
