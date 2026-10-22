/**
 * Pure helpers that turn a synthetic dataset (e2e/data/<name>/) into the FrameInput stream a
 * FramePump would deliver: frame selection (stride / simulated pump rate / limit), IMU slices
 * since the previous delivered frame, the latest GPS fix and compass heading, and the gravity
 * direction metadata. For a raw capture recording (frames.jsonl, src/io/recording.ts)
 * `verbatimInput` gives back the recorded FrameInput fields unchanged instead. No DOM,
 * unit-tested in feed.test.ts.
 */
import type { GeoFix, ImuSample } from '../../core/types';
import type { FrameMeta, RecordedFrame } from '../../io/recording';
import { CompassSensor } from '../../sensors/orientation';
import { decimalYearFromDate, magneticField, trueToMagneticHeading } from '../../sensors/wmm';

export interface TimedFrame {
  index: number;
  t: number;
}

export interface PlanOptions {
  /** Position (in time order) of the first frame, default 0. */
  startFrame?: number;
  /** Keep every Nth frame, default 1. */
  frameStride?: number;
  /** Simulated pump rate (Hz); null / undefined = no rate limit. */
  fps?: number | null;
  /** Dataset frame rate (Hz): half a frame interval of tolerance for the rate limit. */
  datasetFps?: number;
  /** Maximum number of frames, null / undefined = all. */
  maxFrames?: number | null;
  /** Keep the given order instead of sorting by time (a recording: the order frames were sent in). */
  keepOrder?: boolean;
}

/**
 * Frames to replay, in time order (or as given with `keepOrder`): from `startFrame`, every
 * `frameStride`-th, then rate-limited like the FramePump (a frame is delivered when at least
 * 1000/fps ms minus half a dataset frame interval passed since the last delivered one), then at
 * most `maxFrames`.
 */
export function planFrames<T extends TimedFrame>(frames: readonly T[], o: PlanOptions = {}): T[] {
  const sorted = o.keepOrder ? [...frames] : [...frames].sort((a, b) => a.t - b.t || a.index - b.index);
  const start = Math.max(0, Math.floor(o.startFrame ?? 0));
  const stride = Math.max(1, Math.floor(o.frameStride ?? 1));
  const out: T[] = [];
  const minDt = o.fps && o.fps > 0 ? 1000 / o.fps - (o.datasetFps && o.datasetFps > 0 ? 500 / o.datasetFps : 0) : 0;
  let lastT = -Infinity;
  for (let k = start; k < sorted.length; k += stride) {
    const f = sorted[k];
    if (minDt > 0 && f.t - lastT < minDt) continue;
    out.push(f);
    lastT = f.t;
    if (o.maxFrames && out.length >= o.maxFrames) break;
  }
  return out;
}

/** Index of the first element with `t > x` (arr sorted by t). */
export function upperBound(arr: readonly { t: number }[], x: number): number {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (arr[mid].t <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Samples with t0 < t ≤ t1 (arr sorted by t). */
export function samplesBetween<T extends { t: number }>(arr: readonly T[], t0: number, t1: number): T[] {
  if (!(t1 > t0)) return [];
  return arr.slice(upperBound(arr, t0), upperBound(arr, t1));
}

/** Latest element with t ≤ x, or null. */
export function latestAtOrBefore<T extends { t: number }>(arr: readonly T[] | null | undefined, x: number): T | null {
  if (!arr || arr.length === 0) return null;
  const i = upperBound(arr, x) - 1;
  return i >= 0 ? arr[i] : null;
}

/**
 * Unit gravity direction (DOWN) in the device frame at time t, from the newest IMU sample at or
 * before t: −(accelG − accel)/|…| (the specific force of gravity alone points UP), or −accelG
 * normalised when the linear acceleration is missing. Null without a usable sample within 200 ms.
 * Returned as a `Vec3` (Float64Array) like FrameInput.gravityDevice.
 */
export function gravityDownAt(imu: readonly ImuSample[], t: number): Float64Array | null {
  const s = latestAtOrBefore(imu, t);
  if (!s || t - s.t > 200 || !s.accelG) return null;
  const a = s.accel ?? [0, 0, 0];
  const up = [s.accelG[0] - a[0], s.accelG[1] - a[1], s.accelG[2] - a[2]];
  const n = Math.hypot(up[0], up[1], up[2]);
  if (!(n > 1e-6)) return null;
  return Float64Array.of(-up[0] / n, -up[1] / n, -up[2] / n);
}

/** A heading.json row (e2e/README.md): `heading` = optical-axis azimuth, `headingDeviceTop` = device +y azimuth. */
export interface HeadingRow {
  t: number;
  heading: number;
  accuracy: number;
  headingDeviceTop?: number | null;
}

/**
 * The heading a CompassSensor would report for a dataset row (src/sensors: the azimuth of the
 * device TOP while it is within 45° of horizontal, else of the camera axis), i.e. the reference
 * that computeWorldAlignment's `compassReference: 'auto'` assumes. The dataset's `heading` field
 * is always the optical axis: passing it unchanged in a landscape capture (device top horizontal,
 * pointing left) makes the engine read it as a device-top azimuth ⇒ a 90° yaw error.
 * `gravityDownDevice` = unit DOWN vector in the device frame (null: unknown ⇒ optical axis).
 */
export function compassHeading(row: HeadingRow | null, gravityDownDevice: ArrayLike<number> | null): { heading: number; accuracy: number } | null {
  if (!row || !Number.isFinite(row.heading)) return null;
  const topNearHorizontal = !!gravityDownDevice && Math.abs(gravityDownDevice[1]) < Math.SQRT1_2;
  const top = row.headingDeviceTop;
  const heading = topNearHorizontal && typeof top === 'number' && Number.isFinite(top) ? top : row.heading;
  return { heading, accuracy: row.accuracy };
}

/**
 * North reference of the replayed compass (the datasets store TRUE-north headings):
 *  - 'magnetic': converted to MAGNETIC headings with WMM2025 at the dataset location
 *    (`meta.originLatLonAlt`) and the capture date, flagged `north: 'magnetic'` — a phone whose
 *    compass never learned the declination; the engine converts back with the declination at its
 *    first GPS fix (the full magnetic → true path);
 *  - 'app': what the app sends — magnetic until a usable GPS fix was delivered, then true north
 *    via `CompassSensor.setLocationFromFix` (the app's policy, run on the delivered fixes);
 *  - 'true': the dataset headings unchanged, flagged `north: 'true'` (the declination is known
 *    from elsewhere). Datasets without a location always replay 'true' (magnetic north is
 *    undefined without a place).
 */
export type HeadingNorthMode = 'magnetic' | 'app' | 'true';

/** A FrameInput heading with its north reference (protocol `HeadingSample` extension fields). */
export interface NorthHeading {
  heading: number;
  accuracy: number;
  north: 'true' | 'magnetic';
  declinationDeg: number | null;
}

/** Turns dataset (true-north) compass headings into what the phone would report (see `HeadingNorthMode`). */
export class HeadingNorthEmulator {
  readonly mode: HeadingNorthMode;
  /** WMM declination (deg, east positive) at the dataset location and date (0 without a location). */
  readonly datasetDeclinationDeg: number;
  private readonly compass = new CompassSensor({ target: null });
  private readonly dateMs: number;

  constructor(mode: HeadingNorthMode, location: readonly number[] | null | undefined, dateMs: number) {
    const ok = !!location && location.length >= 2 && Number.isFinite(location[0]) && Number.isFinite(location[1]);
    this.mode = ok ? mode : 'true';
    this.dateMs = dateMs;
    const alt = ok && Number.isFinite(location![2]) ? location![2] : 0;
    this.datasetDeclinationDeg = ok ? magneticField(location![0], location![1], alt / 1000, decimalYearFromDate(dateMs)).declination : 0;
  }

  /** The heading for one frame; `fix` = the GPS fix delivered with it (app mode learns the declination from it). */
  convert(h: { heading: number; accuracy: number } | null, fix: GeoFix | null): NorthHeading | null {
    if (!h || !Number.isFinite(h.heading)) return null;
    if (this.mode === 'true') return { heading: h.heading, accuracy: h.accuracy, north: 'true', declinationDeg: null };
    const magnetic = trueToMagneticHeading(h.heading, this.datasetDeclinationDeg);
    if (this.mode === 'magnetic') return { heading: magnetic, accuracy: h.accuracy, north: 'magnetic', declinationDeg: 0 };
    this.compass.setLocationFromFix(fix, this.dateMs);
    const d = this.compass.declinationDeg;
    const north = this.compass.northReference;
    return { heading: north === 'true' ? (((magnetic + d) % 360) + 360) % 360 : magnetic, accuracy: h.accuracy, north, declinationDeg: d };
  }
}

/** Field of view (deg) across the long image side of a pinhole with focal f (px). */
export function hfovLongSideDeg(f: number, width: number, height: number): number {
  return (2 * Math.atan(Math.max(width, height) / 2 / f) * 180) / Math.PI;
}

/** A GeoFix without dataset-only fields (e.g. `trueEnu`). */
export function cleanGeoFix(g: GeoFix | null): GeoFix | null {
  if (!g) return null;
  return { t: g.t, lat: g.lat, lon: g.lon, alt: g.alt, accuracy: g.accuracy, altitudeAccuracy: g.altitudeAccuracy };
}

/** n, mean, median, p90, max of a list (nulls when empty). */
export function numStats(values: readonly number[]): { n: number; mean: number | null; median: number | null; p90: number | null; max: number | null } {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (v.length === 0) return { n: 0, mean: null, median: null, p90: null, max: null };
  const q = (p: number): number => v[Math.min(v.length - 1, Math.floor(p * v.length))];
  return { n: v.length, mean: v.reduce((a, b) => a + b, 0) / v.length, median: q(0.5), p90: q(0.9), max: v[v.length - 1] };
}

/** Sensor streams a replay passes (HarnessConfig useImu / useGps / useHeading / useGravity). */
export interface VerbatimSensors {
  imu: boolean;
  gps: boolean;
  heading: boolean;
  gravity: boolean;
}

/**
 * The FrameInput (without pixels) of recorded frame `rows[pos]` as the app sent it: every field
 * verbatim, except that the IMU samples of the rows skipped since the previously replayed row
 * `prevPos` (stride / rate limit / start) are prepended, like the FramePump drains everything
 * since its last delivered frame — with every row replayed, each frame gets exactly its own
 * samples. A sensor stream turned off is withheld the way the app would without that sensor
 * (`imu`: no samples, `imuCalibrationVersion` −1; `heading`: no heading and no attitude).
 */
export function verbatimInput(rows: readonly RecordedFrame[], pos: number, prevPos: number, use: VerbatimSensors): FrameMeta {
  const src = rows[pos].input;
  const out: FrameMeta = { ...src };
  if (!use.imu) {
    out.imu = [];
    if (src.imuCalibrationVersion !== undefined) out.imuCalibrationVersion = -1;
  } else if (pos - prevPos > 1) {
    const imu: ImuSample[] = [];
    for (let k = Math.max(0, prevPos + 1); k <= pos; k++) imu.push(...(rows[k].input.imu ?? []));
    out.imu = imu;
  }
  if (!use.gps && src.geo !== undefined) out.geo = null;
  if (!use.heading) {
    if (src.heading !== undefined) out.heading = null;
    if (src.attitude !== undefined) out.attitude = null;
  }
  if ((!use.gravity || !use.imu) && src.gravityDevice !== undefined) out.gravityDevice = null;
  return out;
}
