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
 *  - gravity: IMU specific force (`estimateGravity`), else per-keyframe `gravityCam`;
 *  - yaw: GPS 4-DOF alignment (needs gravity) when the GPS track is long enough, else compass;
 *  - scale: IMU and GPS combined by inverse-variance weighting in log-scale when both are
 *    reliable and consistent (else the more certain one);
 *  - translation: GPS ⇒ local ENU at `originLatLonAlt`; otherwise the first camera (or the
 *    centroid) is put at the origin.
 */
import type { FrameInfo, GeoFix, ImuSample, Mat3, MetricInfo, Pose, Vec3 } from '../core/types';
import { mat3Mul } from '../math/mat3';
import { alignToGps, yawFromCompass, type CompassObservation, type CompassYaw, type GpsAlignment, type GpsAlignOptions } from './geo';
import {
  composeGravityAlignment,
  estimateGravity,
  estimateMetricScale,
  estimateTimeOffset,
  estimateTimeOffsetFromAcceleration,
  type AccelTimeOffsetResult,
  type GravityOptions,
  type GravityResult,
  type InertialKeyframe,
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
  /** Compass heading (deg clockwise from true north) of the reference axis (FrameInfo.heading). */
  heading?: number | null;
  /** 1-σ heading accuracy (deg). */
  headingAccuracy?: number | null;
  /** GPS fix associated with the keyframe (used when no `gpsFixes` are given). */
  geo?: GeoFix | null;
}

/** `WorldKeyframe` from the recorded frame metadata and the reconstructed pose. */
export function worldKeyframeFromFrame(info: FrameInfo, pose: Pose, headingAccuracyDeg: number | null = null): WorldKeyframe {
  return {
    id: info.id,
    t: info.t,
    pose,
    R_cam_device: info.R_cam_device,
    gravityCam: info.gravityCam,
    heading: info.heading,
    headingAccuracy: headingAccuracyDeg,
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
  gps: Partial<GpsAlignOptions>;
  /**
   * Axis the compass heading refers to: 'device-top' (+y of the device), 'optical-axis', or
   * 'auto' = device top while it is within 45° of horizontal, else the optical axis (the
   * convention of `sensors/CompassSensor`). Default 'auto'.
   */
  compassReference: 'auto' | 'device-top' | 'optical-axis';
  /** Minimum compass observations (default 3). */
  minCompassObservations: number;
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
  gps: {},
  compassReference: 'auto',
  minCompassObservations: 3,
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
    imuScale: MetricScaleResult | null;
    gps: GpsAlignment | null;
    compass: CompassYaw | null;
    yawSource: 'gps' | 'compass' | 'none';
    /** Scale candidates (metres per reconstruction unit) with their relative σ. */
    scaleCandidates: { source: 'imu' | 'gps'; scale: number; relUncertainty: number }[];
    warnings: string[];
  };
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
  const imu = input.imu ?? [];
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

  // ---- rotation / yaw
  let R: Mat3 = Rg;
  let yawSource: 'gps' | 'compass' | 'none' = 'none';
  let compass: CompassYaw | null = null;
  let northAligned = false;
  let gravityAligned = gravityOk;
  if (gps && gps.mode === '4dof') {
    R = mat3Mul(gps.R, Rg);
    yawSource = 'gps';
    northAligned = true;
  } else if (gps && gps.mode === 'sim3') {
    R = gps.R;
    yawSource = 'gps';
    northAligned = true;
    gravityAligned = true;
    warnings.push('up direction from the GPS track only (no gravity)');
  } else if (gravityOk) {
    const obs: CompassObservation[] = [];
    const dir = new Float64Array(3);
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
      dir.set(ref);
      obs.push({ headingDeg: k.heading, accuracyDeg: k.headingAccuracy ?? null, direction: Float64Array.from(dir) });
    }
    if (obs.length >= o.minCompassObservations) {
      compass = yawFromCompass(obs);
      if (compass && compass.numInliers >= o.minCompassObservations) {
        R = mat3Mul(rotZ(compass.yaw), Rg);
        yawSource = 'compass';
        northAligned = true;
      } else compass = null;
    }
  }

  // ---- scale
  const cands: { source: 'imu' | 'gps'; scale: number; relUncertainty: number }[] = [];
  if (imuScale && imuScale.reliable) cands.push({ source: 'imu', scale: imuScale.scale, relUncertainty: imuScale.relUncertainty });
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
    details: { timeOffset, accelTimeOffset, offsetMs, offsetSource, gravity, imuScale, gps, compass, yawSource, scaleCandidates: cands, warnings },
  };
}

function transposeOf(R: Mat3): Mat3 {
  return Float64Array.of(R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]);
}
