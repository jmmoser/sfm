/**
 * GNSS fixes via `navigator.geolocation.watchPosition` (main thread only on iOS).
 *
 * iOS facts (docs/research/ios-sensors.md §4): `enableHighAccuracy: true` ⇒
 * kCLLocationAccuracyBest, ~1 Hz while moving; `coords.accuracy` = horizontalAccuracy
 * (treat as ~1 σ); `altitude` is orthometric (MSL, EGM2008), NOT WGS84-ellipsoidal;
 * `timestamp` = fix time in epoch ms on the wall clock. With "Precise Location" off the
 * accuracy degrades to 1–20 km (`approximate` flag).
 *
 * Time base: GeoFix.t = position.timestamp − (Date.now() − performance.now()), i.e. the wall
 * clock → performance.now() offset measured when the fix is delivered (robust against wall
 * clock adjustments since page load, unlike performance.timeOrigin).
 */
import type { GeoFix } from '../core/types';

export type GeoStatus = 'idle' | 'watching' | 'denied' | 'unavailable' | 'error';

export interface GeoPositionLike {
  timestamp: number;
  coords: {
    latitude: number;
    longitude: number;
    altitude?: number | null;
    accuracy: number;
    altitudeAccuracy?: number | null;
  };
}

export interface GeoSensorOptions {
  /** Default true (kCLLocationAccuracyBest on iOS). */
  highAccuracy?: boolean;
  /** Keep at most this many fixes (default 7200 ≈ 2 h at 1 Hz). */
  maxFixes?: number;
  /** `interpolate` only bridges gaps up to this length (ms, default 5000). */
  maxInterpolationGapMs?: number;
  /** Accuracy above which the location is considered approximate (m, default 1000). */
  approximateThresholdM?: number;
  /** Injected geolocation implementation (tests). */
  geolocation?: Geolocation | null;
}

/** Convert an epoch-ms wall-clock timestamp to the performance.now() domain. */
export function epochToPerfMs(epochMs: number, nowEpochMs: number = Date.now(), nowPerfMs: number = performance.now()): number {
  return epochMs - (nowEpochMs - nowPerfMs);
}

function lowerBoundFix(fixes: readonly GeoFix[], t: number): number {
  let lo = 0;
  let hi = fixes.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (fixes[mid].t < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export class GeoSensor {
  /** Chronological fixes (performance.now() time base). Do not mutate. */
  readonly fixes: GeoFix[] = [];
  private watchId: number | null = null;
  private st: GeoStatus = 'idle';
  private err: string | null = null;
  private approx = false;
  private readonly opts: Required<Omit<GeoSensorOptions, 'geolocation'>>;
  private readonly geo: Geolocation | null;
  private readonly listeners = new Set<(f: GeoFix) => void>();

  static isSupported(): boolean {
    return typeof navigator !== 'undefined' && !!navigator.geolocation;
  }

  constructor(opts: GeoSensorOptions = {}) {
    this.opts = {
      highAccuracy: opts.highAccuracy ?? true,
      maxFixes: opts.maxFixes ?? 7200,
      maxInterpolationGapMs: opts.maxInterpolationGapMs ?? 5000,
      approximateThresholdM: opts.approximateThresholdM ?? 1000,
    };
    this.geo =
      opts.geolocation !== undefined ? opts.geolocation : typeof navigator !== 'undefined' ? navigator.geolocation ?? null : null;
  }

  get status(): GeoStatus {
    return this.st;
  }
  get lastError(): string | null {
    return this.err;
  }
  /**
   * True when the NEWEST fix is coarser than the threshold (iOS "Precise Location" off gives
   * 1–20 km on every fix). Not sticky: the first fix of a session is often a coarse cell/Wi-Fi
   * fix before GNSS converges.
   */
  get approximate(): boolean {
    return this.approx;
  }

  start(): void {
    if (this.watchId !== null) return;
    if (!this.geo) {
      this.st = 'unavailable';
      this.err = 'Geolocation API not available';
      return;
    }
    this.st = 'watching';
    this.watchId = this.geo.watchPosition(
      (p) => this.handlePosition(p),
      (e) => this.handleError(e),
      { enableHighAccuracy: this.opts.highAccuracy, maximumAge: 0, timeout: Infinity },
    );
  }

  stop(): void {
    if (this.watchId !== null && this.geo) this.geo.clearWatch(this.watchId);
    this.watchId = null;
    if (this.st === 'watching') this.st = 'idle';
  }

  onFix(cb: (f: GeoFix) => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }

  /** Feed a position (watchPosition calls this; public for replay/tests). Returns the stored fix or null. */
  handlePosition(p: GeoPositionLike, nowEpochMs: number = Date.now(), nowPerfMs: number = performance.now()): GeoFix | null {
    const c = p.coords;
    if (!Number.isFinite(c.latitude) || !Number.isFinite(c.longitude)) return null;
    let t = epochToPerfMs(p.timestamp, nowEpochMs, nowPerfMs);
    if (!Number.isFinite(t) || t > nowPerfMs + 1000) t = nowPerfMs; // clock oddities: fix cannot be from the future
    const alt = typeof c.altitude === 'number' && Number.isFinite(c.altitude) ? c.altitude : null;
    const altAcc = typeof c.altitudeAccuracy === 'number' && Number.isFinite(c.altitudeAccuracy) && c.altitudeAccuracy >= 0 ? c.altitudeAccuracy : null;
    const fix: GeoFix = {
      t,
      lat: c.latitude,
      lon: c.longitude,
      alt,
      accuracy: Number.isFinite(c.accuracy) ? c.accuracy : Infinity,
      altitudeAccuracy: alt === null ? null : altAcc,
    };
    const n = this.fixes.length;
    if (n > 0) {
      const last = this.fixes[n - 1];
      // Duplicate delivery of the same fix.
      if (Math.abs(last.t - fix.t) < 1 && last.lat === fix.lat && last.lon === fix.lon) return null;
    }
    if (n > 0 && fix.t < this.fixes[n - 1].t) {
      const i = lowerBoundFix(this.fixes, fix.t);
      this.fixes.splice(i, 0, fix);
    } else {
      this.fixes.push(fix);
      this.approx = fix.accuracy > this.opts.approximateThresholdM;
    }
    if (this.fixes.length > this.opts.maxFixes) this.fixes.splice(0, this.fixes.length - this.opts.maxFixes);
    if (this.st !== 'watching' && this.watchId !== null) this.st = 'watching';
    this.err = null;
    for (const cb of this.listeners) {
      try {
        cb(fix);
      } catch (e) {
        console.error(e);
      }
    }
    return fix;
  }

  /** Watch error (public for tests). PERMISSION_DENIED stops the watch. */
  handleError(e: { code: number; message?: string }): void {
    this.err = e.message || `GeolocationPositionError ${e.code}`;
    if (e.code === 1) {
      this.st = 'denied';
      if (this.watchId !== null && this.geo) this.geo.clearWatch(this.watchId);
      this.watchId = null;
    } else if (e.code === 2) {
      // POSITION_UNAVAILABLE: keep watching, fixes may come later.
      this.st = this.watchId !== null ? 'watching' : 'unavailable';
    } else if (e.code !== 3) {
      this.st = 'error';
    }
  }

  /** Most recent fix (by time) or null. */
  latest(): GeoFix | null {
    return this.fixes.length ? this.fixes[this.fixes.length - 1] : null;
  }

  /** Fix nearest in time to t (any distance in time), or null if there is none. */
  nearest(t: number): GeoFix | null {
    const f = this.fixes;
    if (!f.length) return null;
    const i = lowerBoundFix(f, t);
    if (i === 0) return f[0];
    if (i === f.length) return f[f.length - 1];
    return t - f[i - 1].t <= f[i].t - t ? f[i - 1] : f[i];
  }

  /**
   * Position at time t: linear interpolation between the bracketing fixes (lat/lon/alt and
   * accuracies) when they are at most `maxInterpolationGapMs` apart; outside the covered
   * range the nearest fix if within that gap; otherwise null.
   */
  interpolate(t: number): GeoFix | null {
    const f = this.fixes;
    if (!f.length) return null;
    const gap = this.opts.maxInterpolationGapMs;
    const i = lowerBoundFix(f, t);
    if (i < f.length && f[i].t === t) return { ...f[i], t };
    if (i === 0) return f[0].t - t <= gap ? { ...f[0], t } : null;
    if (i === f.length) return t - f[i - 1].t <= gap ? { ...f[i - 1], t } : null;
    const a = f[i - 1];
    const b = f[i];
    if (b.t - a.t > gap) {
      const n = t - a.t <= b.t - t ? a : b;
      return Math.abs(n.t - t) <= gap ? { ...n, t } : null;
    }
    const u = (t - a.t) / (b.t - a.t);
    let dlon = b.lon - a.lon;
    if (dlon > 180) dlon -= 360;
    if (dlon < -180) dlon += 360;
    let lon = a.lon + u * dlon;
    if (lon > 180) lon -= 360;
    if (lon < -180) lon += 360;
    const alt = a.alt !== null && b.alt !== null ? a.alt + u * (b.alt - a.alt) : u < 0.5 ? a.alt : b.alt;
    const altAcc =
      a.altitudeAccuracy !== null && b.altitudeAccuracy !== null
        ? a.altitudeAccuracy + u * (b.altitudeAccuracy - a.altitudeAccuracy)
        : u < 0.5
          ? a.altitudeAccuracy
          : b.altitudeAccuracy;
    return {
      t,
      lat: a.lat + u * (b.lat - a.lat),
      lon,
      alt,
      accuracy: a.accuracy + u * (b.accuracy - a.accuracy),
      altitudeAccuracy: alt === null ? null : altAcc,
    };
  }
}
