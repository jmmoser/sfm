/**
 * World-frame alignment of a finished reconstruction (ARCHITECTURE §2 "World frame after
 * alignment", §5 "Refine"): combines the inertial estimates (time offset, gravity, metric
 * scale), GPS (scale, yaw, geo-reference) and the compass (yaw) into one similarity
 * X' = s R X + t to apply with `Reconstruction.transform`, so that +Z is up, +Y is (true)
 * north when known, and units are metres when known.
 *
 * Decision logic:
 *  - time offset: gyro rotations (`estimateTimeOffset`); if unobservable (little angular
 *    acceleration), linear acceleration (`estimateTimeOffsetFromAcceleration`); else default;
 *  - gravity: IMU specific force (`estimateGravity`), else per-keyframe `gravityCam`. When the
 *    two point to opposite hemispheres the accelerometer samples carry the wrong sign (the
 *    platform sign self-check decided late or wrongly; `gravityCam` comes from the
 *    DeviceOrientation attitude, which does not depend on that sign): the specific force is
 *    negated for everything that uses it (gravity, acceleration time offset, metric scale);
 *  - north: compass headings flagged MAGNETIC (`WorldKeyframe.headingNorth`) are turned into
 *    true-north headings with the WMM2025 declination (`sensors/wmm.ts`) at the first GPS fix
 *    whose accuracy is ≤ 50 km (an "approximate" iOS fix is plenty), else at the optional
 *    `coarseLocation`, at `captureDate` (default now). Without any location the compass yaw is
 *    MAGNETIC: it is still applied (+Y ≈ magnetic north, better than an arbitrary yaw) but
 *    `northAligned` stays false and `details.north` = 'magnetic';
 *  - yaw: GPS 4-DOF alignment (needs gravity) when the GPS track is long enough and/or the
 *    compass; with both, inverse-variance fusion on the circle (GPS σ from `alignToGps`,
 *    compass σ from the reported accuracies, the sample count and a systematic floor); when they
 *    disagree by > 3 combined σ the compass is dropped against a confident GPS yaw (σ ≤ 10°),
 *    otherwise the less certain one (`details.yawFusion`, `yawSource`, warning);
 *  - scale: IMU (`estimateMetricScale`; when that is unreliable — e.g. too little band-limited
 *    excitation at phone keyframe rates of 1–2 /s — the keyframe-interval preintegration estimate
 *    `estimateMetricScaleFromIntervals`) and GPS, combined by inverse-variance weighting in
 *    log-scale when both are reliable and consistent (else the more certain one);
 *  - translation: GPS ⇒ local ENU at `originLatLonAlt`; otherwise the first camera (or the
 *    centroid) is put at the origin.
 */
import type { FrameInfo, GeoFix, ImuSample, Mat3, MetricInfo, Pose, Vec3 } from '../core/types';
import { mat3Mul } from '../math/mat3';
import { decimalYearFromDate, magneticField } from '../sensors/wmm';
import {
  alignToGps,
  fuseYaw,
  yawFromCompass,
  type CompassObservation,
  type CompassYaw,
  type CompassYawOptions,
  type GpsAlignment,
  type GpsAlignOptions,
  type YawFusion,
} from './geo';
import {
  composeGravityAlignment,
  estimateGravity,
  estimateMetricScale,
  estimateMetricScaleFromIntervals,
  estimateTimeOffset,
  estimateTimeOffsetFromAcceleration,
  type AccelTimeOffsetResult,
  type GravityOptions,
  type GravityResult,
  type InertialKeyframe,
  type IntervalScaleOptions,
  type MetricScaleOptions,
  type MetricScaleResult,
  type TimeOffsetOptions,
  type TimeOffsetResult,
} from './inertial';

/** Keyframe input of `computeWorldAlignment`. */
export interface WorldKeyframe {
  id: number;
  /** Capture time (ms, camera clock). */
  t: number;
  /** World-to-camera pose in the reconstruction frame. */
  pose: Pose;
  R_cam_device: Mat3;
  /** Gravity DOWN in the camera frame (e.g. FrameInfo.gravityCam). */
  gravityCam?: ArrayLike<number> | null;
  /** Compass heading (deg clockwise from north, see `headingNorth`) of the reference axis (FrameInfo.heading). */
  heading?: number | null;
  /** 1-σ heading accuracy (deg). */
  headingAccuracy?: number | null;
  /**
   * North reference of `heading`: 'magnetic' headings get the WMM declination at the capture
   * location (see the module comment); 'true' / absent = already true north.
   */
  headingNorth?: 'true' | 'magnetic' | null;
  /** GPS fix associated with the keyframe (used when no `gpsFixes` are given). */
  geo?: GeoFix | null;
}

/** `WorldKeyframe` from the recorded frame metadata and the reconstructed pose. */
export function worldKeyframeFromFrame(
  info: FrameInfo,
  pose: Pose,
  headingAccuracyDeg: number | null = null,
  headingNorth: 'true' | 'magnetic' | null = null,
): WorldKeyframe {
  return {
    id: info.id,
    t: info.t,
    pose,
    R_cam_device: info.R_cam_device,
    gravityCam: info.gravityCam,
    heading: info.heading,
    headingAccuracy: headingAccuracyDeg,
    headingNorth,
    geo: info.geo,
  };
}

/** Inputs of `computeWorldAlignment`. */
export interface WorldAlignmentInput {
  keyframes: readonly WorldKeyframe[];
  imu?: readonly ImuSample[] | null;
  /** All GPS fixes of the session (preferred over per-keyframe `geo`). */
  gpsFixes?: readonly GeoFix[] | null;
}

/** Options of `computeWorldAlignment`. */
export interface WorldAlignmentOptions {
  /** Camera–IMU offset used when it cannot be estimated (ms; also the search centre). Default 0. */
  timeOffsetMs: number;
  timeOffset: Partial<TimeOffsetOptions>;
  gravity: Partial<GravityOptions>;
  scale: Partial<MetricScaleOptions>;
  /** Options of the keyframe-interval scale estimate (used when the band-pass one is unreliable). */
  intervalScale: Partial<IntervalScaleOptions>;
  gps: Partial<GpsAlignOptions>;
  /**
   * Axis the compass heading refers to: 'device-top' (+y of the device), 'optical-axis', or
   * 'auto' = device top while it is within 45° of horizontal, else the optical axis (the
   * convention of `sensors/CompassSensor`). Default 'auto'.
   */
  compassReference: 'auto' | 'device-top' | 'optical-axis';
  /** Minimum compass observations (default 3). */
  minCompassObservations: number;
  /** Options of `yawFromCompass` (outlier gates, systematic σ). */
  compass: Partial<CompassYawOptions>;
  /** Capture date for the magnetic declination (Date or epoch ms); default: now. */
  captureDate: Date | number | null;
  /**
   * Location [lat°, lon°, alt m] for the declination when no usable GPS fix exists (e.g. a coarse
   * location from elsewhere); default null.
   */
  coarseLocation: readonly [number, number, number?] | null;
  /** GPS fixes less accurate than this (m) are not used for the declination (default 50 000). */
  maxDeclinationFixAccuracyM: number;
  /** Explicit declination (deg, east positive) instead of the model; default null (WMM). */
  declinationDeg: number | null;
  /** Fuse the compass yaw with the GPS 4-DOF yaw (default true; false: GPS decides alone). */
  fuseCompassYaw: boolean;
  /** GPS and compass yaws disagreeing by more than this many combined σ are not fused (default 3). */
  maxYawDisagreementSigma: number;
  /** On such a disagreement a GPS yaw with σ ≤ this (deg, default 10) wins even against a more certain compass. */
  confidentGpsYawStdDeg: number;
  /** Apply a MAGNETIC compass yaw when no declination is available (default true; northAligned stays false). */
  useMagneticYaw: boolean;
  /** GPS scale is used only below this relative σ (default 0.25). */
  maxGpsScaleRelUncertainty: number;
  /** IMU and GPS scales disagreeing by more than this many σ are not averaged (default 3). */
  maxScaleDisagreementSigma: number;
  /** Translation without GPS: put the first camera, the camera centroid, or nothing at the origin. */
  origin: 'first-camera' | 'centroid' | 'keep';
}

const DEFAULT_WORLD: WorldAlignmentOptions = {
  timeOffsetMs: 0,
  timeOffset: {},
  gravity: {},
  scale: {},
  intervalScale: {},
  gps: {},
  compassReference: 'auto',
  minCompassObservations: 3,
  compass: {},
  captureDate: null,
  coarseLocation: null,
  maxDeclinationFixAccuracyM: 50_000,
  declinationDeg: null,
  fuseCompassYaw: true,
  maxYawDisagreementSigma: 3,
  confidentGpsYawStdDeg: 10,
  useMagneticYaw: true,
  maxGpsScaleRelUncertainty: 0.25,
  maxScaleDisagreementSigma: 3,
  origin: 'first-camera',
};

/** Result of `computeWorldAlignment`. Apply as X' = s R X + t. */
export interface WorldAlignment {
  s: number;
  R: Mat3;
  t: Vec3;
  metric: MetricInfo;
  details: {
    timeOffset: TimeOffsetResult | null;
    /** Acceleration-based offset (computed only when the gyro-based one is unreliable). */
    accelTimeOffset: AccelTimeOffsetResult | null;
    /** Camera–IMU offset used (ms). */
    offsetMs: number;
    /** Where `offsetMs` comes from: gyro rotation, linear acceleration, or the default. */
    offsetSource: 'gyro' | 'accel' | 'default';
    gravity: GravityResult | null;
    /** The accelerometer samples were negated: their gravity contradicted the keyframes' `gravityCam`. */
    accelSignCorrected: boolean;
    /** Band-pass IMU scale (`estimateMetricScale`). */
    imuScale: MetricScaleResult | null;
    /** Keyframe-interval IMU scale, computed only when `imuScale` is unreliable (else null). */
    imuScaleIntervals: MetricScaleResult | null;
    gps: GpsAlignment | null;
    /** Compass yaw (true north when `north` is 'true' or a GPS yaw exists, else magnetic). */
    compass: CompassYaw | null;
    /** Where the applied yaw comes from ('gps+compass' = inverse-variance fusion). */
    yawSource: 'gps' | 'compass' | 'gps+compass' | 'none';
    /** 1-σ of the applied yaw (rad), null without one. */
    yawStdRad: number | null;
    /** GPS ⊕ compass fusion (null unless both yaws were available). */
    yawFusion: YawFusion | null;
    /** What +Y points to: true north, magnetic north (compass without a declination), or nothing known. */
    north: 'true' | 'magnetic' | 'none';
    /** Magnetic declination used / available at the capture location (null without a location). */
    declination: DeclinationEstimate | null;
    /** Flat summary for engine stats: north reference, declination, yaw σ and fusion. */
    headingStats: Record<string, number | string>;
    /** Scale candidates (metres per reconstruction unit) with their relative σ. */
    scaleCandidates: { source: 'imu' | 'gps'; scale: number; relUncertainty: number }[];
    warnings: string[];
  };
}

/** Declination at the capture location (WMM2025, `sensors/wmm.ts`). */
export interface DeclinationEstimate {
  /** Degrees, east positive (true heading = magnetic heading + declination). */
  deg: number;
  /** Location source: first usable GPS fix, `coarseLocation`, or an explicit `declinationDeg`. */
  source: 'gps' | 'coarse' | 'explicit';
  latDeg: number | null;
  lonDeg: number | null;
  decimalYear: number | null;
  /** The date is inside the model's validity period (2025.0–2030.0). */
  inValidityPeriod: boolean;
  /** Horizontal field < 2000 nT: compass unreliable (compass yaw not used). */
  blackoutZone: boolean;
  /** Keyframe headings converted from magnetic to true north. */
  numConverted: number;
}

/** Samples with the specific force (`accelG`) and linear acceleration (`accel`) negated. */
function negateAccelerometer(imu: readonly ImuSample[]): ImuSample[] {
  return imu.map((s) => ({
    ...s,
    accel: s.accel ? [-s.accel[0], -s.accel[1], -s.accel[2]] : s.accel,
    accelG: s.accelG ? [-s.accelG[0], -s.accelG[1], -s.accelG[2]] : s.accelG,
  }));
}

/**
 * Pose after the similarity X' = s R X + t (camera coordinates scaled by s as well):
 * R' = R_p Rᵀ, t' = s t_p − R' t.
 */
export function transformPose(pose: Pose, s: number, R: Mat3, t: ArrayLike<number>): Pose {
  const Rp = pose.R;
  const Rn = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) Rn[3 * i + j] = Rp[3 * i] * R[3 * j] + Rp[3 * i + 1] * R[3 * j + 1] + Rp[3 * i + 2] * R[3 * j + 2];
  const tn = new Float64Array(3);
  for (let i = 0; i < 3; i++) tn[i] = s * pose.t[i] - (Rn[3 * i] * t[0] + Rn[3 * i + 1] * t[1] + Rn[3 * i + 2] * t[2]);
  return { R: Rn, t: tn };
}

function rotZ(phi: number): Mat3 {
  const c = Math.cos(phi),
    s = Math.sin(phi);
  return Float64Array.of(c, -s, 0, s, c, 0, 0, 0, 1);
}

function mulVec(R: ArrayLike<number>, x: number, y: number, z: number, out: Float64Array, o: number): void {
  out[o] = R[0] * x + R[1] * y + R[2] * z;
  out[o + 1] = R[3] * x + R[4] * y + R[5] * z;
  out[o + 2] = R[6] * x + R[7] * y + R[8] * z;
}

/**
 * Combine IMU (time offset, gravity, metric scale), GPS and compass into the world
 * similarity. Every source is optional; with none, the result is the identity (plus the
 * origin choice) and `metric` reports nothing aligned. See the module comment for the logic.
 */
export function computeWorldAlignment(input: WorldAlignmentInput, options: Partial<WorldAlignmentOptions> = {}): WorldAlignment {
  const o: WorldAlignmentOptions = { ...DEFAULT_WORLD, ...options };
  const warnings: string[] = [];
  const kfs = [...input.keyframes].filter((k) => Number.isFinite(k.t)).sort((a, b) => a.t - b.t);
  const n = kfs.length;
  const C = new Float64Array(3 * n);
  const Rwc: Mat3[] = [];
  const times = new Float64Array(n);
  const ikf: InertialKeyframe[] = [];
  for (let i = 0; i < n; i++) {
    const { R, t } = kfs[i].pose;
    const rwc = Float64Array.of(R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]);
    Rwc.push(rwc);
    C[3 * i] = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]);
    C[3 * i + 1] = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]);
    C[3 * i + 2] = -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);
    times[i] = kfs[i].t;
    ikf.push({ t: kfs[i].t, R_wc: rwc, C: C.subarray(3 * i, 3 * i + 3), R_cam_device: kfs[i].R_cam_device, gravityCam: kfs[i].gravityCam ?? null });
  }

  // ---- inertial
  let imu = input.imu ?? [];
  let timeOffset: TimeOffsetResult | null = null;
  let accelTimeOffset: AccelTimeOffsetResult | null = null;
  let offsetMs = o.timeOffsetMs;
  let offsetSource: WorldAlignment['details']['offsetSource'] = 'default';
  let gyroBias: Float64Array | null = null;
  if (imu.length >= 2 && n >= 2) {
    timeOffset = estimateTimeOffset(ikf, imu, { initialOffsetMs: o.timeOffsetMs, ...o.timeOffset });
    if (timeOffset && timeOffset.reliable) {
      offsetMs = timeOffset.offsetMs;
      offsetSource = 'gyro';
      gyroBias = timeOffset.gyroBias;
    }
  }
  let gravity: GravityResult | null = n > 0 ? estimateGravity(ikf, imu, { gyroBias, ...o.gravity, offsetMs }) : null;
  // An accelerometer with the wrong sign yields gravity pointing UP (a model upside down) that
  // still looks reliable; the orientation-based gravityCam is independent of that sign and goes
  // through the same R_cam_device, so a disagreement beyond 90° can only be the sign. Re-checked
  // here because samples streamed before MotionSensor's runtime sign check decided keep the prior
  // sign (its ring buffer re-signs only what it still holds).
  let accelSignCorrected = false;
  if (gravity && gravity.source === 'accelG') {
    const gc = estimateGravity(ikf, [], o.gravity);
    if (gc && gc.numSamples >= 3 && gc.down[0] * gravity.down[0] + gc.down[1] * gravity.down[1] + gc.down[2] * gravity.down[2] < 0) {
      imu = negateAccelerometer(imu);
      gravity = estimateGravity(ikf, imu, { gyroBias, ...o.gravity, offsetMs });
      accelSignCorrected = true;
      warnings.push('accelerometer gravity opposite to the orientation sensor: accelerometer sign corrected');
    }
  }
  if (gravity && !gravity.reliable) {
    warnings.push(`gravity estimate unreliable (σ ${((gravity.stdRad * 180) / Math.PI).toFixed(1)}°)`);
    // try the per-keyframe fallback if the IMU path was used
    if (gravity.source === 'accelG') {
      const g2 = estimateGravity(ikf, [], o.gravity);
      if (g2 && g2.reliable) gravity = g2;
    }
  }
  const gravityOk = !!gravity && gravity.reliable;
  const scaleGravity = (): MetricScaleOptions['gravity'] =>
    gravity ? { down: gravity.down, magnitude: gravity.source === 'accelG' ? gravity.magnitude : undefined } : null;
  // Without enough angular acceleration the rotation-based offset is unobservable; an offset
  // error would bias the metric scale (≈ 5 % for 40 ms), so estimate it from linear acceleration.
  if (offsetSource !== 'gyro' && imu.length >= 4 && n >= 6) {
    accelTimeOffset = estimateTimeOffsetFromAcceleration(ikf, imu, {
      initialOffsetMs: o.timeOffsetMs,
      searchRangeMs: o.timeOffset.searchRangeMs ?? 250,
      scale: { gyroBias, gravity: scaleGravity(), ...o.scale },
    });
    if (accelTimeOffset && accelTimeOffset.reliable) {
      offsetMs = accelTimeOffset.offsetMs;
      offsetSource = 'accel';
      if (gravity && gravity.source === 'accelG') {
        const g2 = estimateGravity(ikf, imu, { gyroBias, ...o.gravity, offsetMs });
        if (g2 && g2.source === 'accelG' && g2.reliable) gravity = g2;
      }
    }
  }
  if (imu.length >= 2 && n >= 2 && offsetSource === 'default') warnings.push('camera–IMU time offset not observable; using the default');
  let imuScale: MetricScaleResult | null = null;
  if (imu.length >= 2 && n >= 6) {
    imuScale = estimateMetricScale(ikf, imu, {
      gyroBias,
      gravity: scaleGravity(),
      ...o.scale,
      offsetMs,
    });
    if (imuScale && !imuScale.reliable) warnings.push(`IMU scale unreliable: ${imuScale.reason}`);
  }
  // The band-pass estimate collapses at low keyframe rates (its excitation is the 0.3–3 Hz
  // acceleration as sampled at the keyframes); the interval estimate integrates the full-rate IMU.
  let imuScaleIntervals: MetricScaleResult | null = null;
  if (imu.length >= 2 && n >= 6 && !(imuScale && imuScale.reliable)) {
    imuScaleIntervals = estimateMetricScaleFromIntervals(ikf, imu, { gyroBias, gravity: scaleGravity(), ...o.intervalScale, offsetMs });
    if (imuScaleIntervals && imuScaleIntervals.reliable)
      warnings.push(
        `IMU scale from keyframe intervals (σ ${(100 * imuScaleIntervals.relUncertainty).toFixed(1)} %, span ${imuScaleIntervals.spanS!.toFixed(2)} s, R² ${imuScaleIntervals.r2.toFixed(3)})`,
      );
    else if (imuScaleIntervals) warnings.push(`IMU interval scale unreliable: ${imuScaleIntervals.reason}`);
  }
  const imuUsed = imuScale && imuScale.reliable ? imuScale : imuScaleIntervals && imuScaleIntervals.reliable ? imuScaleIntervals : null;
  const Rg = gravityOk ? composeGravityAlignment(gravity!.down) : Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);

  // ---- GPS
  let fixes: GeoFix[] = [];
  if (input.gpsFixes && input.gpsFixes.length) fixes = [...input.gpsFixes];
  else {
    const seen = new Set<number>();
    for (const k of kfs) {
      if (k.geo && !seen.has(k.geo.t)) {
        seen.add(k.geo.t);
        fixes.push(k.geo);
      }
    }
  }
  let gps: GpsAlignment | null = null;
  const Cg = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) mulVec(Rg, C[3 * i], C[3 * i + 1], C[3 * i + 2], Cg, 3 * i);
  if (fixes.length > 0 && n >= 2) {
    gps = alignToGps(gravityOk ? Cg : C, times, fixes, { ...o.gps, gravityAligned: gravityOk });
    if (!gps) warnings.push('GPS track too short / too few fixes for alignment');
  }

  // ---- north reference: magnetic declination at the capture location
  const anyHeading = kfs.some((k) => k.heading !== null && k.heading !== undefined && Number.isFinite(k.heading));
  const anyMagnetic = kfs.some((k) => k.headingNorth === 'magnetic' && k.heading !== null && k.heading !== undefined && Number.isFinite(k.heading));
  const declination = anyHeading ? resolveDeclination(fixes, o) : null;
  if (declination && !declination.inValidityPeriod) warnings.push(`capture date outside the WMM2025 validity period (${declination.decimalYear?.toFixed(2)}): declination extrapolated`);
  if (anyMagnetic && !declination) warnings.push('compass headings are magnetic and no location is known for the declination: compass yaw refers to magnetic north');

  // ---- compass yaw (true north when every used heading is true north)
  let compass: CompassYaw | null = null;
  let compassNorth: 'true' | 'magnetic' = 'true';
  if (gravityOk && anyHeading && declination?.blackoutZone) warnings.push('compass unusable here (WMM blackout zone, horizontal field < 2000 nT)');
  else if (gravityOk && anyHeading) {
    const trueObs: CompassObservation[] = [];
    const magObs: CompassObservation[] = [];
    let converted = 0;
    for (let i = 0; i < n; i++) {
      const k = kfs[i];
      if (k.heading === null || k.heading === undefined || !Number.isFinite(k.heading)) continue;
      // device top (+y_device) and optical axis (+z_cam) in the gravity-aligned frame
      const Rwd = mat3Mul(Rwc[i], k.R_cam_device);
      const top = new Float64Array(3);
      mulVec(Rg, Rwd[1], Rwd[4], Rwd[7], top, 0);
      const axis = new Float64Array(3);
      mulVec(Rg, Rwc[i][2], Rwc[i][5], Rwc[i][8], axis, 0);
      let ref: Float64Array;
      if (o.compassReference === 'device-top') ref = top;
      else if (o.compassReference === 'optical-axis') ref = axis;
      else ref = Math.abs(top[2]) / Math.hypot(top[0], top[1], top[2]) < Math.SQRT1_2 ? top : axis;
      const ob: CompassObservation = { headingDeg: k.heading, accuracyDeg: k.headingAccuracy ?? null, direction: ref };
      if (k.headingNorth !== 'magnetic') trueObs.push(ob);
      else if (declination) {
        trueObs.push({ ...ob, headingDeg: k.heading + declination.deg });
        converted++;
      } else magObs.push(ob);
    }
    if (declination) declination.numConverted = converted;
    // never mix references: true north when enough true headings exist, else magnetic ones
    let obs = trueObs;
    if (trueObs.length < o.minCompassObservations && magObs.length >= o.minCompassObservations && o.useMagneticYaw) {
      obs = magObs;
      compassNorth = 'magnetic';
    }
    if (obs.length >= o.minCompassObservations) {
      compass = yawFromCompass(obs, o.compass);
      if (compass && compass.numInliers < o.minCompassObservations) compass = null;
    }
  }

  // ---- rotation / yaw
  let R: Mat3 = Rg;
  let yawSource: WorldAlignment['details']['yawSource'] = 'none';
  let yawStdRad: number | null = null;
  let yawFusion: YawFusion | null = null;
  let north: WorldAlignment['details']['north'] = 'none';
  let gravityAligned = gravityOk;
  if (gps && gps.mode === '4dof') {
    const gpsYaw = Math.atan2(gps.R[3], gps.R[0]);
    let yaw = gpsYaw;
    yawSource = 'gps';
    yawStdRad = gps.yawStdRad;
    if (compass && compassNorth === 'true' && o.fuseCompassYaw) {
      yawFusion = fuseYaw(
        { yaw: gpsYaw, stdRad: gps.yawStdRad },
        { yaw: compass.yaw, stdRad: compass.stdRad },
        { maxDisagreementSigma: o.maxYawDisagreementSigma, confidentGpsStdRad: (o.confidentGpsYawStdDeg * Math.PI) / 180 },
      );
      yaw = yawFusion.yaw;
      yawStdRad = yawFusion.stdRad;
      yawSource = yawFusion.used === 'both' ? 'gps+compass' : yawFusion.used;
      if (yawFusion.rejected) {
        warnings.push(
          `compass and GPS yaws disagree by ${((yawFusion.differenceRad * 180) / Math.PI).toFixed(1)}° (${yawFusion.disagreementSigma.toFixed(1)}σ); ` +
            `using the ${yawFusion.used === 'gps' ? 'GPS' : 'compass'} yaw`,
        );
      }
    } else if (compass && compassNorth === 'magnetic') warnings.push('magnetic compass yaw not fused with the GPS yaw');
    R = mat3Mul(rotZ(yaw), Rg);
    north = 'true';
  } else if (gps && gps.mode === 'sim3') {
    R = gps.R;
    yawSource = 'gps';
    yawStdRad = gps.yawStdRad;
    north = 'true';
    gravityAligned = true;
    warnings.push('up direction from the GPS track only (no gravity)');
  } else if (compass) {
    R = mat3Mul(rotZ(compass.yaw), Rg);
    yawSource = 'compass';
    yawStdRad = compass.stdRad;
    north = compassNorth;
  }
  const northAligned = north === 'true';

  // ---- scale
  const cands: { source: 'imu' | 'gps'; scale: number; relUncertainty: number }[] = [];
  if (imuUsed) cands.push({ source: 'imu', scale: imuUsed.scale, relUncertainty: imuUsed.relUncertainty });
  if (gps && gps.scaleRelUncertainty <= o.maxGpsScaleRelUncertainty && gps.s > 0)
    cands.push({ source: 'gps', scale: gps.s, relUncertainty: gps.scaleRelUncertainty });
  else if (gps) warnings.push(`GPS scale too uncertain (${(100 * gps.scaleRelUncertainty).toFixed(0)} %)`);
  let s = 1;
  let source: MetricInfo['source'] = 'none';
  let rel: number | null = null;
  if (cands.length === 1) {
    s = cands[0].scale;
    source = cands[0].source;
    rel = cands[0].relUncertainty;
  } else if (cands.length === 2) {
    const [a, b] = cands;
    const va = a.relUncertainty ** 2,
      vb = b.relUncertainty ** 2;
    const d = Math.abs(Math.log(a.scale / b.scale));
    if (d > o.maxScaleDisagreementSigma * Math.sqrt(va + vb)) {
      const best = va <= vb ? a : b;
      warnings.push(`IMU and GPS scales disagree (${a.scale.toPrecision(4)} vs ${b.scale.toPrecision(4)}); using ${best.source}`);
      s = best.scale;
      source = best.source;
      rel = best.relUncertainty;
    } else {
      const wa = 1 / Math.max(va, 1e-12),
        wb = 1 / Math.max(vb, 1e-12);
      s = Math.exp((wa * Math.log(a.scale) + wb * Math.log(b.scale)) / (wa + wb));
      rel = Math.sqrt(1 / (wa + wb));
      source = 'imu+gps';
    }
  }
  const metric = source !== 'none';

  // ---- translation
  const t = new Float64Array(3);
  let originLatLonAlt: [number, number, number] | null = null;
  if (gps) {
    // weighted mean offset with the final s and R (pairs are in the frame given to alignToGps)
    const Rpair = gps.mode === '4dof' ? mat3Mul(R, transposeOf(Rg)) : R;
    const p = new Float64Array(3);
    let sw = 0,
      sx = 0,
      sy = 0,
      swz = 0,
      sz = 0,
      mz = 0,
      cnt = 0;
    for (let i = 0; i < gps.numFixes; i++) {
      if (!gps.inlierMask[i]) continue;
      mulVec(Rpair, gps.src[3 * i], gps.src[3 * i + 1], gps.src[3 * i + 2], p, 0);
      const w = gps.weights[i];
      sw += w;
      sx += w * (gps.dst[3 * i] - s * p[0]);
      sy += w * (gps.dst[3 * i + 1] - s * p[1]);
      if (gps.hasAlt[i]) {
        swz += w;
        sz += w * (gps.dst[3 * i + 2] - s * p[2]);
      }
      mz += p[2];
      cnt++;
    }
    t[0] = sx / sw;
    t[1] = sy / sw;
    t[2] = swz > 0 ? sz / swz : -s * (mz / Math.max(cnt, 1));
    originLatLonAlt = gps.origin;
  } else if (o.origin !== 'keep' && n > 0) {
    let cx = 0,
      cy = 0,
      cz = 0;
    if (o.origin === 'first-camera') {
      cx = C[0];
      cy = C[1];
      cz = C[2];
    } else {
      for (let i = 0; i < n; i++) {
        cx += C[3 * i];
        cy += C[3 * i + 1];
        cz += C[3 * i + 2];
      }
      cx /= n;
      cy /= n;
      cz /= n;
    }
    const p = new Float64Array(3);
    mulVec(R, cx, cy, cz, p, 0);
    t[0] = -s * p[0];
    t[1] = -s * p[1];
    t[2] = -s * p[2];
  }

  return {
    s,
    R,
    t,
    metric: {
      metric,
      source,
      relativeUncertainty: metric ? rel : null,
      gravityAligned,
      northAligned,
      originLatLonAlt,
    },
    details: {
      timeOffset,
      accelTimeOffset,
      offsetMs,
      offsetSource,
      gravity,
      accelSignCorrected,
      imuScale,
      imuScaleIntervals,
      gps,
      compass,
      yawSource,
      yawStdRad,
      yawFusion,
      north,
      declination,
      headingStats: headingStats(north, yawStdRad, declination, compass, compassNorth, gps, yawFusion),
      scaleCandidates: cands,
      warnings,
    },
  };
}

/**
 * Declination for the magnetic → true conversion: an explicit value, else WMM2025 at the first
 * (in time) GPS fix with a finite position and an accuracy ≤ `maxDeclinationFixAccuracyM`, else
 * at `coarseLocation`; null without a location. Heights: GeoFix altitudes are MSL on iOS — fine
 * for the declination.
 */
function resolveDeclination(fixes: readonly GeoFix[], o: WorldAlignmentOptions): DeclinationEstimate | null {
  const date = o.captureDate instanceof Date ? o.captureDate.getTime() : typeof o.captureDate === 'number' ? o.captureDate : Date.now();
  const year = decimalYearFromDate(date);
  if (o.declinationDeg !== null && Number.isFinite(o.declinationDeg)) {
    return { deg: o.declinationDeg, source: 'explicit', latDeg: null, lonDeg: null, decimalYear: null, inValidityPeriod: true, blackoutZone: false, numConverted: 0 };
  }
  let loc: { lat: number; lon: number; alt: number; source: 'gps' | 'coarse' } | null = null;
  let firstT = Infinity;
  for (const f of fixes) {
    if (!Number.isFinite(f.lat) || !Number.isFinite(f.lon) || Math.abs(f.lat) > 90) continue;
    if (!(f.accuracy <= o.maxDeclinationFixAccuracyM)) continue;
    const t = Number.isFinite(f.t) ? f.t : Infinity;
    if (loc && !(t < firstT)) continue;
    firstT = t;
    loc = { lat: f.lat, lon: f.lon, alt: f.alt !== null && Number.isFinite(f.alt) ? f.alt : 0, source: 'gps' };
  }
  const c = o.coarseLocation;
  if (!loc && c && Number.isFinite(c[0]) && Number.isFinite(c[1]) && Math.abs(c[0]) <= 90) {
    const alt = c[2];
    loc = { lat: c[0], lon: c[1], alt: typeof alt === 'number' && Number.isFinite(alt) ? alt : 0, source: 'coarse' };
  }
  if (!loc) return null;
  const e = magneticField(loc.lat, loc.lon, loc.alt / 1000, year);
  if (!Number.isFinite(e.declination)) return null;
  return {
    deg: e.declination,
    source: loc.source,
    latDeg: loc.lat,
    lonDeg: loc.lon,
    decimalYear: e.decimalYear,
    inValidityPeriod: e.inValidityPeriod,
    blackoutZone: e.blackoutZone,
    numConverted: 0,
  };
}

const RAD2DEG = 180 / Math.PI;
const r3 = (x: number): number => Math.round(x * 1000) / 1000;

/** Flat, JSON-friendly heading summary (merged into the engine's refine stats). */
function headingStats(
  north: WorldAlignment['details']['north'],
  yawStdRad: number | null,
  decl: DeclinationEstimate | null,
  compass: CompassYaw | null,
  compassNorth: 'true' | 'magnetic',
  gps: GpsAlignment | null,
  fusion: YawFusion | null,
): Record<string, number | string> {
  const st: Record<string, number | string> = { north };
  if (yawStdRad !== null && Number.isFinite(yawStdRad)) st.yawStdDeg = r3(yawStdRad * RAD2DEG);
  if (decl) {
    st.declinationDeg = r3(decl.deg);
    st.declinationSource = decl.source;
    if (decl.numConverted > 0) st.headingsConverted = decl.numConverted;
  }
  if (compass) {
    st.compassNorth = compassNorth;
    st.compassYawStdDeg = r3(compass.stdRad * RAD2DEG);
    st.compassInliers = compass.numInliers;
  }
  if (gps && gps.mode === '4dof') st.gpsYawStdDeg = r3(gps.yawStdRad * RAD2DEG);
  if (fusion) {
    st.compassMinusGpsYawDeg = r3(fusion.differenceRad * RAD2DEG);
    st.yawDisagreementSigma = Math.round(fusion.disagreementSigma * 100) / 100;
    st.yawFusion = fusion.rejected ? `${fusion.rejected} rejected` : `compass weight ${fusion.compassWeight.toFixed(2)}`;
  }
  return st;
}

function transposeOf(R: Mat3): Mat3 {
  return Float64Array.of(R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]);
}
