/**
 * Keyframe selection policy for continuous capture (ARCHITECTURE §5 "Live", step 3).
 *
 * Every analysed frame is summarised as a `KeyframeCandidate` (sharpness, gyro angular
 * speed, feature / match / inlier counts against the last keyframe, rotation-compensated
 * parallax and gyro rotation since the last keyframe). `KeyframeSelector.evaluate` rejects
 * blurry frames (relative to a running median of recent sharpness values) and frames taken
 * while rotating fast, then makes a keyframe when the view has changed enough:
 *   - overlap: inliers to the last keyframe < `overlapThreshold` × the best inlier count
 *     observed since that keyframe (the achievable match rate, so the criterion does not
 *     depend on the absolute matching ratio of the feature type);
 *   - parallax: rotation-compensated median parallax > fraction × image width
 *     (object 3 %, auto 4 %, scene 5 %);
 *   - rotation: gyro rotation since the last keyframe > `maxRotationDeg`;
 *   - tracking lost: inliers < `minInliers`;
 *   - `maxIntervalMs` elapsed while the camera is moving;
 * never earlier than `minIntervalMs` after the previous keyframe. Soft triggers wait up to
 * `maxDeferMs` for a sharper frame. The first keyframe is taken once `sharpnessWarmup` frames
 * have been seen (so it is not blurry). Guidance for the UI is attached to every decision.
 * Pure rotation (rotation without rotation-compensated parallax) ⇒ 'translate_more'.
 */
import type { CameraIntrinsics, Mat3, MatchPairs } from '../core/types';
import { pixelToNormalized } from '../geometry/camera';
import { mat3Inv } from '../math/mat3';
import { quickselect } from '../math/stats';

/** Capture mode: close-range object, large scene, or automatic. */
export type CaptureMode = 'object' | 'scene' | 'auto';

/** Outcome of `KeyframeSelector.evaluate`. */
export type KeyframeDecisionKind = 'keyframe' | 'skip' | 'reject_blur' | 'reject_motion';

/** User guidance shown during capture. */
export type CaptureGuidance = 'move_slower' | 'more_texture' | 'translate_more' | 'tracking_lost' | 'good';

/** What made a frame a keyframe. */
export type KeyframeTrigger = 'first' | 'overlap' | 'parallax' | 'rotation' | 'tracking_lost' | 'max_interval';

/** Per-frame measurements for the keyframe decision. */
export interface KeyframeCandidate {
  /** Frame time (ms). */
  t: number;
  /** Sharpness (variance of Laplacian, `features/image.sharpness`). */
  sharpness: number;
  /** Gyro angular speed (rad/s) at the frame time; NaN when unknown. */
  angularSpeed: number;
  /** Features detected in this frame. */
  numFeatures: number;
  /** Putative matches to the last keyframe (0 if there is none). */
  numMatchesToLastKF: number;
  /** Verified (geometric) inliers to the last keyframe; NaN = not measured for this frame. */
  numInliersToLastKF: number;
  /** Median rotation-compensated parallax to the last keyframe (px, full resolution). NaN = unknown. */
  medianParallaxPx: number;
  /** Rotation since the last keyframe (rad), e.g. from the gyro. NaN = unknown. */
  rotationSinceLastKFRad: number;
  /** Image width (px) the parallax refers to. */
  imageWidth: number;
}

/** Decision for one candidate frame. */
export interface KeyframeDecision {
  decision: KeyframeDecisionKind;
  /** Human-readable reason (for logs / debugging UI). */
  reason: string;
  guidance?: CaptureGuidance;
  /** Set when decision = 'keyframe'. */
  trigger?: KeyframeTrigger;
}

/** Options of `KeyframeSelector`. */
export interface KeyframeSelectorOptions {
  mode: CaptureMode;
  /** Frames rotating faster than this (rad/s) are rejected (default 1.2). */
  maxAngularSpeed: number;
  /** Frames with sharpness < blurRatio × running median are rejected (default 0.6). */
  blurRatio: number;
  /** Absolute sharpness floor (default 0 = off). */
  minSharpness: number;
  /** Running sharpness window (frames, default 30) and warm-up length (default 5). */
  sharpnessWindow: number;
  sharpnessWarmup: number;
  /** Minimum / maximum keyframe interval (ms; defaults 150 / 2000). */
  minIntervalMs: number;
  maxIntervalMs: number;
  /** Keyframe when inliers / best-inliers-since-KF falls below this (default 0.7). */
  overlapThreshold: number;
  /** Parallax trigger as a fraction of the image width, per mode. */
  parallaxFraction: { object: number; scene: number; auto: number };
  /** Rotation trigger (degrees, default 12). */
  maxRotationDeg: number;
  /** Tracking is lost below this many inliers (default 30). */
  minInliers: number;
  /** Frames with fewer features are skipped with 'more_texture' (default 150). */
  minFeatures: number;
  /** Pure-rotation guidance: rotation ≥ this (deg, default 4) with parallax < pureRotationParallaxRatio × trigger. */
  pureRotationMinDeg: number;
  pureRotationParallaxRatio: number;
  /** Max-interval keyframes only while moving: parallax ≥ ratio × trigger or rotation ≥ movingRotationDeg. */
  movingParallaxRatio: number;
  movingRotationDeg: number;
  /** Soft triggers wait up to this long (ms) for a frame with sharpness ≥ deferSharpnessRatio × median. */
  maxDeferMs: number;
  deferSharpnessRatio: number;
  /** Overlap below which a keyframe is never deferred (default 0.5). */
  hardOverlap: number;
  /** 'move_slower' warning above this fraction of maxAngularSpeed (default 0.75). */
  warnAngularRatio: number;
}

/** Defaults of `KeyframeSelectorOptions`. */
export const DEFAULT_KEYFRAME_OPTIONS: KeyframeSelectorOptions = {
  mode: 'auto',
  maxAngularSpeed: 1.2,
  blurRatio: 0.6,
  minSharpness: 0,
  sharpnessWindow: 30,
  sharpnessWarmup: 5,
  minIntervalMs: 150,
  maxIntervalMs: 2000,
  overlapThreshold: 0.7,
  parallaxFraction: { object: 0.03, scene: 0.05, auto: 0.04 },
  maxRotationDeg: 12,
  minInliers: 30,
  minFeatures: 150,
  pureRotationMinDeg: 4,
  pureRotationParallaxRatio: 0.25,
  movingParallaxRatio: 0.25,
  movingRotationDeg: 2,
  maxDeferMs: 250,
  deferSharpnessRatio: 0.85,
  hardOverlap: 0.5,
  warnAngularRatio: 0.75,
};

const DEG = Math.PI / 180;
/** Number of earlier keyframes remembered for `revertLastKeyframe`. */
const HISTORY = 16;

/**
 * Stateful keyframe policy. `evaluate` both decides and, for 'keyframe' decisions, commits
 * the frame as the new reference. Use `commitKeyframe` to force a keyframe chosen elsewhere
 * and `revertLastKeyframe` when the engine could not use the keyframe it was given.
 */
export class KeyframeSelector {
  readonly options: KeyframeSelectorOptions;
  private sharpRing: Float64Array;
  private sharpCount = 0;
  private sharpHead = 0;
  private scratch: Float64Array;
  private lastKfT: number | null = null;
  private bestInliers = 0;
  /** Previously committed keyframes (times / best-inlier counts), most recent last, for revert. */
  private histT = new Float64Array(HISTORY);
  private histBest = new Float64Array(HISTORY);
  private histLen = 0;
  private deferStart: number | null = null;
  private kfCount = 0;

  constructor(opts: Partial<KeyframeSelectorOptions> = {}) {
    this.options = {
      ...DEFAULT_KEYFRAME_OPTIONS,
      ...opts,
      parallaxFraction: { ...DEFAULT_KEYFRAME_OPTIONS.parallaxFraction, ...(opts.parallaxFraction ?? {}) },
    };
    const w = Math.max(1, Math.floor(this.options.sharpnessWindow));
    this.sharpRing = new Float64Array(w);
    this.scratch = new Float64Array(w);
  }

  /** Time of the last keyframe (ms) or null. */
  get lastKeyframeTime(): number | null {
    return this.lastKfT;
  }

  /** Number of keyframes committed so far. */
  get numKeyframes(): number {
    return this.kfCount;
  }

  /** Running median sharpness (NaN before any frame). */
  get sharpnessReference(): number {
    const n = this.sharpCount;
    if (n === 0) return NaN;
    const s = this.scratch.subarray(0, n);
    s.set(this.sharpRing.subarray(0, n));
    return quickselect(s, n >> 1);
  }

  /** Parallax trigger (px) for the given image width in the current mode. */
  parallaxThresholdPx(imageWidth: number): number {
    return this.options.parallaxFraction[this.options.mode] * imageWidth;
  }

  /** Change the capture mode (thresholds apply from the next frame). */
  setMode(mode: CaptureMode): void {
    this.options.mode = mode;
  }

  /** Forget everything (new capture / new segment). */
  reset(): void {
    this.sharpCount = 0;
    this.sharpHead = 0;
    this.resetTracking();
    this.kfCount = 0;
  }

  /** Start a new tracking segment (keeps the sharpness statistics). */
  resetTracking(): void {
    this.lastKfT = null;
    this.bestInliers = 0;
    this.histLen = 0;
    this.deferStart = null;
  }

  /** Make the frame at time t the reference keyframe. */
  commitKeyframe(t: number): void {
    if (this.lastKfT !== null) {
      if (this.histLen === HISTORY) {
        this.histT.copyWithin(0, 1);
        this.histBest.copyWithin(0, 1);
        this.histLen--;
      }
      this.histT[this.histLen] = this.lastKfT;
      this.histBest[this.histLen] = this.bestInliers;
      this.histLen++;
    }
    this.lastKfT = t;
    this.bestInliers = 0;
    this.deferStart = null;
    this.kfCount++;
  }

  /**
   * Undo the last committed keyframe (e.g. the engine failed to register it): the previous
   * keyframe becomes the reference again. Repeated calls walk back through the last 16
   * keyframes of the current tracking segment; beyond that (or after reverting the segment's
   * first keyframe) there is no reference and the next good frame becomes a 'first' keyframe.
   */
  revertLastKeyframe(): void {
    if (this.lastKfT === null) return;
    if (this.histLen > 0) {
      this.histLen--;
      this.lastKfT = this.histT[this.histLen];
      this.bestInliers = this.histBest[this.histLen];
    } else {
      this.lastKfT = null;
      this.bestInliers = 0;
    }
    this.deferStart = null;
    this.kfCount = Math.max(0, this.kfCount - 1);
  }

  private pushSharpness(v: number): void {
    if (!Number.isFinite(v)) return;
    const w = this.sharpRing.length;
    this.sharpRing[this.sharpHead] = v;
    this.sharpHead = (this.sharpHead + 1) % w;
    if (this.sharpCount < w) this.sharpCount++;
  }

  /** Decide for one frame (frames must be presented in time order; a non-finite `t` is skipped). */
  evaluate(c: KeyframeCandidate): KeyframeDecision {
    const o = this.options;
    if (!Number.isFinite(c.t)) return { decision: 'skip', reason: `invalid timestamp ${c.t}` };
    // 1) Blur relative to the running median of PREVIOUS frames (then include this one so the
    //    reference adapts when the whole scene becomes less textured or the light drops).
    //    The warm-up cannot exceed the ring size (else the first keyframe would never come).
    const warmup = Math.min(Math.max(0, o.sharpnessWarmup), this.sharpRing.length);
    const ref = this.sharpCount >= warmup ? this.sharpnessReference : NaN;
    this.pushSharpness(c.sharpness);
    const fastRot = Number.isFinite(c.angularSpeed) && c.angularSpeed > o.warnAngularRatio * o.maxAngularSpeed;
    if (!(c.sharpness >= o.minSharpness) || (Number.isFinite(ref) && c.sharpness < o.blurRatio * ref)) {
      return {
        decision: 'reject_blur',
        reason: `sharpness ${fmt(c.sharpness)} < ${fmt(o.blurRatio)} × median ${fmt(ref)}`,
        guidance: fastRot ? 'move_slower' : 'good',
      };
    }
    // 2) Fast rotation (motion blur / rolling shutter).
    if (Number.isFinite(c.angularSpeed) && c.angularSpeed > o.maxAngularSpeed) {
      return {
        decision: 'reject_motion',
        reason: `angular speed ${fmt(c.angularSpeed)} rad/s > ${fmt(o.maxAngularSpeed)}`,
        guidance: 'move_slower',
      };
    }
    // 3) Too little texture to be useful.
    if (c.numFeatures < o.minFeatures) {
      return { decision: 'skip', reason: `only ${c.numFeatures} features`, guidance: 'more_texture' };
    }
    // 4) First keyframe — only once the blur statistic is warmed up, so that a blurry first
    //    frame cannot become the initialisation keyframe.
    if (this.lastKfT === null) {
      if (!Number.isFinite(ref)) return { decision: 'skip', reason: 'warming up the sharpness statistic', guidance: fastRot ? 'move_slower' : 'good' };
      this.commitKeyframe(c.t);
      return { decision: 'keyframe', reason: 'first keyframe', guidance: fastRot ? 'move_slower' : 'good', trigger: 'first' };
    }

    const dt = c.t - this.lastKfT;
    // Non-finite inlier count = not measured for this frame: neither "lost" nor an overlap drop.
    const inlKnown = Number.isFinite(c.numInliersToLastKF);
    const inl = inlKnown ? Math.max(0, c.numInliersToLastKF) : 0;
    const lost = inlKnown && inl < o.minInliers;
    if (inlKnown && !lost && inl > this.bestInliers) this.bestInliers = inl;
    const overlap = !inlKnown ? 1 : this.bestInliers > 0 ? inl / this.bestInliers : 0;
    const parThr = this.parallaxThresholdPx(c.imageWidth);
    const parallax = Number.isFinite(c.medianParallaxPx) ? c.medianParallaxPx : 0;
    const rot = Number.isFinite(c.rotationSinceLastKFRad) ? c.rotationSinceLastKFRad : 0;
    const pureRotation =
      !lost && rot >= o.pureRotationMinDeg * DEG && Number.isFinite(c.medianParallaxPx) && parallax < o.pureRotationParallaxRatio * parThr;
    const guidance: CaptureGuidance = lost ? 'tracking_lost' : fastRot ? 'move_slower' : pureRotation ? 'translate_more' : 'good';

    if (dt < o.minIntervalMs) {
      return { decision: 'skip', reason: `${fmt(dt)} ms since last keyframe < ${o.minIntervalMs}`, guidance };
    }

    let trigger: KeyframeTrigger | null = null;
    let reason = '';
    if (lost) {
      trigger = 'tracking_lost';
      reason = `${inl} inliers < ${o.minInliers}`;
    } else if (overlap < o.overlapThreshold) {
      trigger = 'overlap';
      reason = `overlap ${fmt(overlap)} < ${fmt(o.overlapThreshold)}`;
    } else if (parallax > parThr) {
      trigger = 'parallax';
      reason = `parallax ${fmt(parallax)} px > ${fmt(parThr)} px`;
    } else if (rot > o.maxRotationDeg * DEG) {
      trigger = 'rotation';
      reason = `rotation ${fmt(rot / DEG)}° > ${o.maxRotationDeg}°`;
    } else if (dt > o.maxIntervalMs && (parallax >= o.movingParallaxRatio * parThr || rot >= o.movingRotationDeg * DEG)) {
      trigger = 'max_interval';
      reason = `${fmt(dt)} ms since last keyframe while moving`;
    }
    if (trigger === null) {
      this.deferStart = null;
      return { decision: 'skip', reason: 'view has not changed enough', guidance };
    }
    // Soft triggers: wait briefly for a sharper frame.
    if (trigger !== 'tracking_lost' && overlap >= o.hardOverlap && Number.isFinite(ref) && c.sharpness < o.deferSharpnessRatio * ref) {
      if (this.deferStart === null) this.deferStart = c.t;
      if (c.t - this.deferStart < o.maxDeferMs) {
        return { decision: 'skip', reason: `${reason}; waiting for a sharper frame`, guidance };
      }
    }
    this.commitKeyframe(c.t);
    return { decision: 'keyframe', reason, guidance, trigger };
  }
}

function fmt(v: number): string {
  return Number.isFinite(v) ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3)) : String(v);
}

let parScratch = new Float64Array(1024);
const nrmA = new Float64Array(2);
const nrmB = new Float64Array(2);

/**
 * Median rotation-compensated parallax (px) between matched keypoints of image a and b:
 * |x_b − π(K_b R_ba K_a⁻¹ x_a)| — the displacement left after removing the image motion that
 * a pure rotation R_ba (camera-a → camera-b frame, e.g. from the gyro) would cause.
 * With `CameraIntrinsics`, keypoints are undistorted first (RADIAL model) and the parallax is
 * measured in undistorted pixels (f × normalized distance); with a 3×3 K matrix (row-major)
 * the homography is applied to raw pixels. Points rotating behind the camera are skipped.
 * Returns NaN when no match is usable.
 *
 * @param kpA keypoints of image a (stride 4: x, y, scale, orientation)
 * @param kpB keypoints of image b
 * @param matches flat pairs [ia0, ib0, ia1, ib1, …]
 * @param K intrinsics of image a (and of b unless `KB` is given)
 * @param R_ba rotation taking camera-a coordinates to camera-b coordinates
 */
export function rotationCompensatedParallax(
  kpA: Float32Array,
  kpB: Float32Array,
  matches: MatchPairs,
  K: CameraIntrinsics | Mat3,
  R_ba: Mat3,
  KB?: CameraIntrinsics | Mat3,
): number {
  const m = matches.length >> 1;
  if (m === 0) return NaN;
  if (parScratch.length < m) parScratch = new Float64Array(Math.max(m, 2 * parScratch.length));
  const out = parScratch;
  const Kb = KB ?? K;
  let cnt = 0;
  const r = R_ba;
  if (isIntrinsics(K) && isIntrinsics(Kb)) {
    const fb = Kb.f;
    for (let k = 0; k < m; k++) {
      const ia = matches[2 * k],
        ib = matches[2 * k + 1];
      pixelToNormalized(K, kpA[4 * ia], kpA[4 * ia + 1], nrmA);
      pixelToNormalized(Kb, kpB[4 * ib], kpB[4 * ib + 1], nrmB);
      const x = nrmA[0],
        y = nrmA[1];
      const X = r[0] * x + r[1] * y + r[2];
      const Y = r[3] * x + r[4] * y + r[5];
      const Z = r[6] * x + r[7] * y + r[8];
      if (!(Z > 1e-9)) continue;
      const d = fb * Math.hypot(nrmB[0] - X / Z, nrmB[1] - Y / Z);
      if (d === d) out[cnt++] = d; // skip NaN (non-finite keypoints / intrinsics)
    }
  } else {
    const Ka = toK(K),
      KbM = toK(Kb);
    const Kai = mat3Inv(Ka);
    if (!Kai) return NaN;
    // H = Kb R Ka⁻¹
    const H = new Float64Array(9);
    const T = new Float64Array(9);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) T[3 * i + j] = r[3 * i] * Kai[j] + r[3 * i + 1] * Kai[3 + j] + r[3 * i + 2] * Kai[6 + j];
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) H[3 * i + j] = KbM[3 * i] * T[j] + KbM[3 * i + 1] * T[3 + j] + KbM[3 * i + 2] * T[6 + j];
    // Sign: points in front of camera a stay in front iff the third row of R·Ka⁻¹x is > 0.
    for (let k = 0; k < m; k++) {
      const ia = matches[2 * k],
        ib = matches[2 * k + 1];
      const u = kpA[4 * ia],
        v = kpA[4 * ia + 1];
      const zr = T[6] * u + T[7] * v + T[8];
      if (!(zr > 1e-12)) continue;
      const w = H[6] * u + H[7] * v + H[8];
      const pu = (H[0] * u + H[1] * v + H[2]) / w;
      const pv = (H[3] * u + H[4] * v + H[5]) / w;
      const d = Math.hypot(kpB[4 * ib] - pu, kpB[4 * ib + 1] - pv);
      if (d === d) out[cnt++] = d; // skip NaN
    }
  }
  if (cnt === 0) return NaN;
  const sub = out.subarray(0, cnt);
  const hi = quickselect(sub, cnt >> 1);
  if (cnt & 1) return hi;
  // even count: mean of the two middle elements (everything before index cnt/2 is ≤ hi)
  let lo = -Infinity;
  for (let i = 0; i < cnt >> 1; i++) if (sub[i] > lo) lo = sub[i];
  return 0.5 * (lo + hi);
}

function isIntrinsics(K: CameraIntrinsics | Mat3): K is CameraIntrinsics {
  return !(K instanceof Float64Array) && typeof (K as CameraIntrinsics).f === 'number';
}

function toK(K: CameraIntrinsics | Mat3): Float64Array {
  if (K instanceof Float64Array) return K;
  return Float64Array.of(K.f, 0, K.cx, 0, K.f, K.cy, 0, 0, 1);
}
