/**
 * Compass heading and full device attitude (DeviceOrientation).
 *
 * Euler convention (W3C, verified in WebKit WebCoreMotionManager.mm): intrinsic Z-X'-Y'',
 * R_earth_device = Rz(α)·Rx(β)·Ry(γ), α ∈ [0,360) about the vertical (counter-clockwise seen
 * from above), β ∈ [−180,180) about device x, γ ∈ [−90,90) about device y. Earth frame: Z up.
 *  - `deviceorientationabsolute` (Chrome/Android): X = east, Y = north ⇒ R is ENU directly.
 *    Its α is relative to MAGNETIC north (apply the declination for true north).
 *  - iOS has no absolute event: α has an ARBITRARY yaw reference (CoreMotion
 *    xArbitraryZVertical, reset when motion updates restart) and drifts slowly; the absolute
 *    reference comes from `webkitCompassHeading` (CLHeading.magneticHeading, clockwise from
 *    magnetic north, referenced to the device top in portrait; 0 with accuracy −1 when invalid).
 *
 * Yaw offset ψ (iOS): heading_ENU(d) = azimuth_arb(d) + ψ for every direction d, hence
 * R_world_device(ENU) = Rz(−ψ)·R_arb. ψ is estimated continuously (circular mean, weighted by
 * compass accuracy and pose quality, exponential forgetting τ = 30 s to follow α drift) from
 * ψ_k = webkitCompassHeading_k − azimuth_arb(ref_k), where ref is the direction CoreLocation
 * reports: the device top (+y) when it is within 45° of horizontal; otherwise (phone upright)
 * the camera direction (−z) with a lower weight — the upright behaviour is UNVERIFIED.
 * A second model (“ui-top”: heading of the UI top, the iOS 26 landscape compensation reported
 * for native apps, UNVERIFIED for Safari) is tracked in parallel and selected automatically
 * when it is clearly more consistent across UI rotations.
 */
import type { Mat3 } from '../core/types';
import { mat3ToRotvec, rotvecToMat3 } from '../math/rotation';
import { screenAngle } from './frame-orientation';
import { DEG_TO_RAD, azimuthDeg, deviceOrientationMatrix, type RawOrientationEvent } from './normalize';

export interface HeadingReading {
  /** Degrees clockwise from north (magnetic unless a declination was set), [0, 360). */
  heading: number;
  /** Degrees (platform-reported, or a nominal value for absolute-orientation sources). */
  accuracy: number;
  source?: 'compass' | 'absolute' | 'attitude';
}

export interface AttitudeReading {
  t: number;
  /** Device → world rotation. World = ENU (X east, Y north, Z up) when `absolute`, else Z-up with arbitrary yaw. */
  R_world_device: Mat3;
  absolute: boolean;
  /** 1-σ-ish yaw uncertainty (degrees) when absolute, null otherwise. */
  yawAccuracyDeg: number | null;
}

export interface CompassSensorOptions {
  target?: EventTarget | null;
  /** Orientation samples kept per source (default 4096 ≈ 68 s at 60 Hz). */
  capacity?: number;
  screenAngle?: () => number;
  /** Magnetic declination (east positive, degrees) added to every heading. Default 0 (magnetic). */
  declinationDeg?: number;
  /** Ignore compass readings with a worse accuracy (degrees). Default 25. */
  maxAccuracyDeg?: number;
  /** Forgetting time constant of the yaw-offset estimate (ms). Default 30000. */
  yawTimeConstantMs?: number;
  /**
   * A gap of this length (ms) in the relative event stream means CoreMotion was restarted (page
   * suspended / listener re-attached), which resets iOS's arbitrary yaw reference ⇒ ψ is
   * re-estimated from scratch. Default 1000.
   */
  restartGapMs?: number;
  /** Nominal accuracy reported for `deviceorientationabsolute` (no accuracy field). Default 20°. */
  absoluteAccuracyDeg?: number;
}

class OrientationRing {
  readonly cap: number;
  readonly t: Float64Array;
  readonly a: Float64Array;
  readonly b: Float64Array;
  readonly g: Float64Array;
  readonly h: Float64Array;
  readonly acc: Float64Array;
  head = 0;
  constructor(cap: number) {
    this.cap = cap;
    this.t = new Float64Array(cap);
    this.a = new Float64Array(cap);
    this.b = new Float64Array(cap);
    this.g = new Float64Array(cap);
    this.h = new Float64Array(cap);
    this.acc = new Float64Array(cap);
  }
  get size(): number {
    return Math.min(this.head, this.cap);
  }
  get first(): number {
    return Math.max(0, this.head - this.cap);
  }
  push(t: number, a: number, b: number, g: number, h: number, acc: number): void {
    if (this.head > 0 && t < this.t[(this.head - 1) % this.cap]) t = this.t[(this.head - 1) % this.cap];
    const k = this.head % this.cap;
    this.t[k] = t;
    this.a[k] = a;
    this.b[k] = b;
    this.g[k] = g;
    this.h[k] = h;
    this.acc[k] = acc;
    this.head++;
  }
  /** Smallest absolute index with time >= t. */
  lowerBound(t: number): number {
    let lo = this.first;
    let hi = this.head;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.t[mid % this.cap] < t) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
}

/** Per-model, per-UI-rotation circular accumulators of ψ samples. */
class YawModel {
  readonly C = new Float64Array(4);
  readonly S = new Float64Array(4);
  readonly W = new Float64Array(4);
  readonly A = new Float64Array(4);
  reset(): void {
    this.C.fill(0);
    this.S.fill(0);
    this.W.fill(0);
    this.A.fill(0);
  }
  decay(f: number): void {
    for (let i = 0; i < 4; i++) {
      this.C[i] *= f;
      this.S[i] *= f;
      this.W[i] *= f;
      this.A[i] *= f;
    }
  }
  add(bucket: number, psiDeg: number, w: number, acc: number): void {
    const r = psiDeg * DEG_TO_RAD;
    this.C[bucket] += w * Math.cos(r);
    this.S[bucket] += w * Math.sin(r);
    this.W[bucket] += w;
    this.A[bucket] += w * acc;
  }
  total(): { psi: number; R: number; W: number; meanAcc: number; buckets: number } {
    let c = 0,
      s = 0,
      w = 0,
      a = 0,
      buckets = 0;
    for (let i = 0; i < 4; i++) {
      c += this.C[i];
      s += this.S[i];
      w += this.W[i];
      a += this.A[i];
      if (this.W[i] >= 3) buckets++;
    }
    const psi = azimuthDeg(s, c); // atan2(s, c) in [0, 360)
    return { psi, R: w > 0 ? Math.hypot(c, s) / w : 0, W: w, meanAcc: w > 0 ? a / w : NaN, buckets };
  }
}

function wrap360(d: number): number {
  const r = d % 360;
  return r < 0 ? r + 360 : r;
}

/** Interpolate two angles (degrees) along the shorter arc. */
function lerpAngle(a: number, b: number, u: number): number {
  let d = (b - a) % 360;
  if (d > 180) d -= 360;
  if (d < -180) d += 360;
  return wrap360(a + u * d);
}

/** Rz(φ) · R (rotate the world frame about +Z by φ radians). */
function rotateYaw(phiRad: number, R: Mat3): Mat3 {
  const c = Math.cos(phiRad);
  const s = Math.sin(phiRad);
  for (let j = 0; j < 3; j++) {
    const x = R[j];
    const y = R[3 + j];
    R[j] = c * x - s * y;
    R[3 + j] = s * x + c * y;
  }
  return R;
}

/**
 * Azimuth (degrees) of the compass reference direction of a device→world rotation: device top
 * (+y, column 1) when it is within 45° of horizontal, else the camera direction (−z).
 * `uiTop` (sin θ, cos θ) replaces the device top for the "ui-top" model.
 */
export function referenceAzimuth(R: ArrayLike<number>, uiTop: readonly [number, number] = [0, 1]): { azimuth: number; weight: number } {
  const ux = uiTop[0];
  const uy = uiTop[1];
  const tx = ux * R[0] + uy * R[1];
  const ty = ux * R[3] + uy * R[4];
  const m2 = tx * tx + ty * ty;
  if (m2 >= 0.5) return { azimuth: azimuthDeg(tx, ty), weight: m2 };
  const bx = -R[2];
  const by = -R[5];
  return { azimuth: azimuthDeg(bx, by), weight: 0.3 * (bx * bx + by * by) };
}

/**
 * Compass + attitude from DeviceOrientation events. `start()` needs the motion/orientation
 * permission on iOS (MotionSensor.requestPermission covers both).
 */
export class CompassSensor {
  private readonly target: EventTarget | null;
  private readonly angleFn: () => number;
  private readonly rel: OrientationRing;
  private readonly abs: OrientationRing;
  private readonly maxAcc: number;
  private readonly tau: number;
  private readonly restartGap: number;
  private readonly absAcc: number;
  private declination: number;
  private isRunning = false;
  private readonly h0 = new YawModel();
  private readonly h1 = new YawModel();
  private lastYawT = -Infinity;
  private readonly Rtmp = new Float64Array(9);
  private readonly Rprev = new Float64Array(9);
  private prevT = -Infinity;
  private model: 'device-top' | 'ui-top' = 'device-top';
  readonly stats = { events: 0, absoluteEvents: 0, compassSamples: 0, rejectedAccuracy: 0, rejectedMotion: 0, yawResets: 0 };

  private readonly onRel = (e: Event): void => this.handleOrientation(e as unknown as RawOrientationEvent, false);
  private readonly onAbs = (e: Event): void => this.handleOrientation(e as unknown as RawOrientationEvent, true);

  constructor(opts: CompassSensorOptions = {}) {
    const g = globalThis as unknown as EventTarget & { addEventListener?: unknown };
    this.target = opts.target !== undefined ? opts.target : typeof g.addEventListener === 'function' ? g : null;
    this.angleFn = opts.screenAngle ?? screenAngle;
    const cap = opts.capacity ?? 4096;
    this.rel = new OrientationRing(cap);
    this.abs = new OrientationRing(cap);
    this.maxAcc = opts.maxAccuracyDeg ?? 25;
    this.tau = opts.yawTimeConstantMs ?? 30000;
    this.restartGap = opts.restartGapMs ?? 1000;
    this.absAcc = opts.absoluteAccuracyDeg ?? 20;
    this.declination = opts.declinationDeg ?? 0;
  }

  get running(): boolean {
    return this.isRunning;
  }

  start(): void {
    if (this.isRunning || !this.target) return;
    this.isRunning = true;
    // Re-attaching may restart CoreMotion (new arbitrary yaw reference on iOS): forget ψ.
    this.resetYaw();
    this.target.addEventListener('deviceorientation', this.onRel);
    if ('ondeviceorientationabsolute' in (this.target as object)) {
      this.target.addEventListener('deviceorientationabsolute', this.onAbs);
    }
  }

  stop(): void {
    if (!this.isRunning) return;
    this.isRunning = false;
    this.target?.removeEventListener('deviceorientation', this.onRel);
    this.target?.removeEventListener('deviceorientationabsolute', this.onAbs);
  }

  private resetYaw(): void {
    this.h0.reset();
    this.h1.reset();
    this.lastYawT = -Infinity;
  }

  /** Set the magnetic declination (east positive) to report true-north headings / ENU. */
  setDeclination(deg: number): void {
    this.declination = Number.isFinite(deg) ? deg : 0;
  }

  get declinationDeg(): number {
    return this.declination;
  }

  /** Which compass reference model is in use (see file header). */
  get headingModel(): 'device-top' | 'ui-top' {
    return this.model;
  }

  /** Feed one event (listeners call this; public for replay/tests). */
  handleOrientation(e: RawOrientationEvent, absoluteEvent = false): void {
    const a = e.alpha,
      b = e.beta,
      g = e.gamma;
    if (!Number.isFinite(a as number) || !Number.isFinite(b as number) || !Number.isFinite(g as number)) return;
    const isAbs = absoluteEvent || e.absolute === true;
    const t = e.timeStamp;
    this.stats.events++;
    if (isAbs) {
      this.stats.absoluteEvents++;
      this.abs.push(t, a as number, b as number, g as number, NaN, NaN);
      return;
    }
    const acc = e.webkitCompassAccuracy;
    const hdg = e.webkitCompassHeading;
    const valid = typeof acc === 'number' && acc >= 0 && typeof hdg === 'number' && Number.isFinite(hdg);
    this.rel.push(t, a as number, b as number, g as number, valid ? (hdg as number) : NaN, valid ? (acc as number) : NaN);

    // Yaw offset update.
    const R = deviceOrientationMatrix(a as number, b as number, g as number, this.Rtmp);
    if (Number.isFinite(this.prevT) && t - this.prevT > this.restartGap) {
      // Event stream interrupted ⇒ CoreMotion restarted with a NEW arbitrary yaw reference:
      // the old ψ is meaningless now (it would otherwise bias ψ for ~τ·ln2 ≈ 20 s).
      this.resetYaw();
      this.stats.yawResets++;
    }
    let fast = false;
    if (t - this.prevT > 0 && t - this.prevT < 200) {
      // rotation angle between consecutive attitudes
      let tr = 0;
      for (let k = 0; k < 9; k++) tr += this.Rprev[k] * R[k];
      const ang = Math.acos(Math.min(1, Math.max(-1, 0.5 * (tr - 1))));
      fast = ang / ((t - this.prevT) / 1000) > 1.0; // > ~57°/s: compass lags
    }
    this.Rprev.set(R);
    this.prevT = t;
    if (!valid) return;
    if ((acc as number) > this.maxAcc) {
      this.stats.rejectedAccuracy++;
      return;
    }
    if (fast) {
      this.stats.rejectedMotion++;
      return;
    }
    this.stats.compassSamples++;
    if (Number.isFinite(this.lastYawT) && t > this.lastYawT) {
      const f = Math.exp(-(t - this.lastYawT) / this.tau);
      this.h0.decay(f);
      this.h1.decay(f);
    }
    this.lastYawT = t;
    const accW = (5 / Math.max(acc as number, 5)) ** 2;
    const theta = this.angleFn();
    const bucket = ((Math.round(theta / 90) % 4) + 4) % 4;
    const r0 = referenceAzimuth(R);
    this.h0.add(bucket, wrap360((hdg as number) - r0.azimuth), r0.weight * accW, acc as number);
    const th = bucket * 90 * DEG_TO_RAD;
    const r1 = referenceAzimuth(R, [Math.round(Math.sin(th)), Math.round(Math.cos(th))]);
    this.h1.add(bucket, wrap360((hdg as number) - r1.azimuth), r1.weight * accW, acc as number);
    this.selectModel();
  }

  private selectModel(): void {
    const a = this.h0.total();
    const b = this.h1.total();
    if (a.buckets >= 2 && b.buckets >= 2) {
      const next = b.R > 0.9 && a.R < 0.75 ? 'ui-top' : a.R > 0.9 && b.R < 0.75 ? 'device-top' : this.model;
      if (next !== this.model) {
        console.warn('[sensors] compass heading model switched to', next, { R0: a.R, R1: b.R });
        this.model = next;
      }
    }
  }

  private yawModel(): YawModel {
    return this.model === 'ui-top' ? this.h1 : this.h0;
  }

  /**
   * Yaw offset ψ (degrees, heading_ENU = azimuth_arb + ψ, magnetic) of the iOS arbitrary
   * DeviceOrientation frame, or null until enough compass evidence was collected.
   */
  get yawOffsetDeg(): number | null {
    const s = this.yawModel().total();
    return s.W >= 5 && s.R > 0.5 ? s.psi : null;
  }

  /** Accuracy of ψ (degrees): hypot(mean reported accuracy, circular spread). */
  get yawAccuracyDeg(): number | null {
    const s = this.yawModel().total();
    if (!(s.W >= 5 && s.R > 0.5)) return null;
    const spread = Math.sqrt(-2 * Math.log(Math.min(1, s.R))) / DEG_TO_RAD;
    return Math.hypot(s.meanAcc, spread);
  }

  private attitudeFrom(ring: OrientationRing, t: number, maxGapMs: number, out: Mat3): boolean {
    if (ring.size === 0) return false;
    const i = ring.lowerBound(t);
    const cap = ring.cap;
    const idx = (k: number): number => k % cap;
    let i0: number, i1: number;
    if (i <= ring.first) i0 = i1 = ring.first;
    else if (i >= ring.head) i0 = i1 = ring.head - 1;
    else {
      i0 = i - 1;
      i1 = i;
    }
    const t0 = ring.t[idx(i0)];
    const t1 = ring.t[idx(i1)];
    if (Math.min(Math.abs(t - t0), Math.abs(t - t1)) > maxGapMs) return false;
    deviceOrientationMatrix(ring.a[idx(i0)], ring.b[idx(i0)], ring.g[idx(i0)], out);
    if (i0 !== i1 && t1 > t0) {
      const u = (t - t0) / (t1 - t0);
      const R1 = deviceOrientationMatrix(ring.a[idx(i1)], ring.b[idx(i1)], ring.g[idx(i1)], new Float64Array(9));
      // Δ = R0ᵀ R1, R = R0 · exp(u · log Δ)
      const D = new Float64Array(9);
      for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++) D[3 * r + c] = out[r] * R1[c] + out[3 + r] * R1[3 + c] + out[6 + r] * R1[6 + c];
      const w = mat3ToRotvec(D);
      w[0] *= u;
      w[1] *= u;
      w[2] *= u;
      const E = rotvecToMat3(w);
      const R0 = new Float64Array(out);
      for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++) out[3 * r + c] = R0[3 * r] * E[c] + R0[3 * r + 1] * E[3 + c] + R0[3 * r + 2] * E[6 + c];
    }
    return true;
  }

  /**
   * Device attitude at time t (interpolated). Prefers `deviceorientationabsolute`; on iOS the
   * arbitrary-yaw attitude is rotated into ENU once the yaw offset ψ is known. The declination
   * is applied in both cases (default 0 ⇒ magnetic north).
   */
  attitudeAt(t: number, maxGapMs = 200): AttitudeReading | null {
    const R = new Float64Array(9);
    if (this.attitudeFrom(this.abs, t, maxGapMs, R)) {
      rotateYaw(-this.declination * DEG_TO_RAD, R);
      return { t, R_world_device: R, absolute: true, yawAccuracyDeg: this.absAcc };
    }
    if (!this.attitudeFrom(this.rel, t, maxGapMs, R)) return null;
    const psi = this.yawOffsetDeg;
    if (psi === null) return { t, R_world_device: R, absolute: false, yawAccuracyDeg: null };
    rotateYaw(-(psi + this.declination) * DEG_TO_RAD, R);
    return { t, R_world_device: R, absolute: true, yawAccuracyDeg: this.yawAccuracyDeg };
  }

  /**
   * Compass heading at time t: the platform value (iOS `webkitCompassHeading`, interpolated
   * along the shorter arc, within 500 ms) or, for absolute-orientation sources, the azimuth of
   * the same reference direction (device top, camera direction when upright).
   */
  headingAt(t: number): HeadingReading | null {
    const ring = this.rel;
    if (ring.size > 0) {
      // nearest valid compass readings around t
      const i = ring.lowerBound(t);
      let before = -1;
      let after = -1;
      for (let k = i - 1; k >= ring.first && k >= i - 64; k--) {
        if (!Number.isNaN(ring.h[k % ring.cap])) {
          before = k;
          break;
        }
      }
      for (let k = i; k < ring.head && k < i + 64; k++) {
        if (!Number.isNaN(ring.h[k % ring.cap])) {
          after = k;
          break;
        }
      }
      const tb = before >= 0 ? ring.t[before % ring.cap] : -Infinity;
      const ta = after >= 0 ? ring.t[after % ring.cap] : Infinity;
      const okB = t - tb <= 500;
      const okA = ta - t <= 500;
      if (okB || okA) {
        let h: number;
        let acc: number;
        if (okB && okA && ta > tb) {
          const u = (t - tb) / (ta - tb);
          h = lerpAngle(ring.h[before % ring.cap], ring.h[after % ring.cap], u);
          acc = Math.max(ring.acc[before % ring.cap], ring.acc[after % ring.cap]);
        } else {
          const k = okB && (!okA || t - tb <= ta - t) ? before : after;
          h = ring.h[k % ring.cap];
          acc = ring.acc[k % ring.cap];
        }
        return { heading: wrap360(h + this.declination), accuracy: acc, source: 'compass' };
      }
    }
    const att = this.attitudeAt(t);
    if (!att || !att.absolute) return null;
    const r = referenceAzimuth(att.R_world_device);
    return {
      heading: r.azimuth,
      accuracy: att.yawAccuracyDeg ?? this.absAcc,
      source: this.abs.size > 0 ? 'absolute' : 'attitude',
    };
  }

  /**
   * Azimuth (degrees from north, clockwise) and elevation (degrees, + = up) of the rear-camera
   * optical axis (−z_device) at time t; null without an absolute attitude.
   */
  cameraHeadingAt(t: number): { heading: number; elevation: number; accuracy: number } | null {
    const att = this.attitudeAt(t);
    if (!att || !att.absolute) return null;
    const R = att.R_world_device;
    const x = -R[2],
      y = -R[5],
      z = -R[8];
    return {
      heading: azimuthDeg(x, y),
      elevation: Math.asin(Math.max(-1, Math.min(1, z))) / DEG_TO_RAD,
      accuracy: att.yawAccuracyDeg ?? this.absAcc,
    };
  }

  /** Newest raw sample (arbitrary-yaw source first), for UI/diagnostics. */
  latest(): { t: number; alpha: number; beta: number; gamma: number; heading: number | null; accuracy: number | null; absolute: boolean } | null {
    const pick = this.abs.head > 0 && (this.rel.head === 0 || this.abs.t[(this.abs.head - 1) % this.abs.cap] >= this.rel.t[(this.rel.head - 1) % this.rel.cap]) ? this.abs : this.rel;
    if (pick.head === 0) return null;
    const k = (pick.head - 1) % pick.cap;
    return {
      t: pick.t[k],
      alpha: pick.a[k],
      beta: pick.b[k],
      gamma: pick.g[k],
      heading: Number.isNaN(pick.h[k]) ? null : pick.h[k],
      accuracy: Number.isNaN(pick.acc[k]) ? null : pick.acc[k],
      absolute: pick === this.abs,
    };
  }
}
