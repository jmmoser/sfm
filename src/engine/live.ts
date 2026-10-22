/**
 * Live capture (ENGINE.md "Frame flow (live)"): per-frame analysis, keyframe selection and the
 * keyframe mapping queue.
 *
 * Per frame: orientation (canonical θ₀, settle, raw buffers) → ≤ 640 px copy (sharpness,
 * colours) → blur / motion pre-gate → features at the preset's working size → guided matching
 * to the last keyframe with the gyro rotation homography H = K R_c K⁻¹,
 * R_c = R_cam_device · R_{d(t)←d(t_kf)} · R_cam_deviceᵀ (generous radius; unguided fallback)
 * → two-view verification → rotation-compensated parallax → KeyframeSelector → LiveUpdate
 * (overlay in the frame's UI orientation).
 *
 * Per keyframe (serial queue, `background` by default — the frame response does not wait):
 * matching against the previous keyframes, vocabulary-retrieval candidates and spatial
 * neighbours (bounded per preset), verification, FeatureDatabase / CorrespondenceGraph update,
 * `LiveMapper.addKeyframe`, vocabulary training / indexing, the R_cam_device hand-eye check,
 * 'auto' mode resolution, persistence and a `live` event with the decimated model.
 *
 * Gyro-aided registration (`predictRotation`, once R_cam_device is verified, also with the gyro
 * gates off): a keyframe P3P cannot register even after LiveMapper re-triangulated its neighbours
 * is registered with the rotation the gyro propagates from the registered keyframe nearest in time
 * (its baseline checked against its best verified pair). At a phone's 2–4 analysed frames per
 * second a fast pan rotates 30°+ between frames and the first keyframe past the map sees a few
 * dozen triangulated points; P3P's 30-inlier floor used to lose tracking there for the rest of the
 * capture (e2e yard-real at stride 3: 19/48 keyframes registered live before;
 * docs/research/real-capture-robustness.md "Phone-rate capture"). Every registration that was not a
 * plain P3P one is logged at debug level (visible points before / after re-triangulation, P3P and
 * known-rotation inliers): `logRegistrations`.
 *
 * Limits: there is one live model — when tracking is lost for good, later keyframes stay pending
 * (refine's offline SfM handles disconnected captures); the 'auto' decision is a heuristic. The
 * worker is single-threaded, so synchronous steps (LiveMapper global BA, vocabulary training at
 * `vocabularyTrainAfter` keyframes) delay frames and cancel by up to ≈ 1 s.
 */
import type { CameraIntrinsics, CancelToken, FeatureSet, FrameInfo, GeoFix, ImuSample, Mat3, MatchPairs, Pose, TwoViewGeometry } from '../core/types';
import type { FeaturePipeline } from '../features/extractor';
import { deserializeFeatureSet, serializeFeatureSet } from '../io/serialize';
import { opticalAxisConvergence } from '../geometry/convergence';
import { solvePnPRansac } from '../geometry/pnp';
import { estimateTwoViewGeometry, type TwoViewOptions } from '../geometry/twoview';
import { mat3Mul, mat3MulTranspose } from '../math/mat3';
import { rotationAngle } from '../math/rotation';
import { median } from '../math/stats';
import { cameraFromDevice, snapScreenAngle, type ScreenAngle } from '../sensors/frame-orientation';
import { angularSpeedAt, integrateGyro } from '../sensors/motion';
import { CorrespondenceGraph } from '../sfm/correspondence-graph';
import { FeatureDatabase, pairKey } from '../sfm/database';
import { checkPairRotation, checkPoseRotation, GyroReference, IMU_MAX_GAP_MS, type GyroToleranceOptions, type TimedRotation } from '../sfm/gyro';
import { estimateTimeOffset, type InertialKeyframe } from '../sfm/inertial';
import { KeyframeSelector, rotationCompensatedParallax, type KeyframeCandidate } from '../sfm/keyframes';
import { LiveMapper, type LiveKeyframeResult } from '../sfm/live';
import { rsModelFromDevice, type RollingShutterModel } from '../sfm/rolling-shutter';
import { intrinsicsFromFov, type MapperOptions } from '../sfm/mapper';
import type { Reconstruction } from '../sfm/reconstruction';
import { VocabularyIndex, type BowVector } from '../sfm/vocab';
import type { CaptureMode, DecodedFrameInput as FrameInput, Guidance, LiveCamera, LiveDebug, LiveUpdate, QualityPreset, TrackingState } from '../workers/protocol';
import {
  OrientationTracker,
  closeSource,
  gravityDown,
  overlayPoints,
  lumaStats,
  prepareFrame,
  rotateVec,
  sampleFeatureColors,
  type PreparedFrame,
} from './frame-ingest';
import {
  ENGINE_STATE_VERSION,
  mergeGeoFixes,
  packImu,
  pairToStored,
  storedToGeometry,
  unpackImu,
  type EngineStateRecord,
  type EngineStore,
  type StoredPair,
  type StoredPairs,
} from './persistence';
import { MB, MEM, captureMemory, formatBreakdown, sumBreakdown, type MemoryBreakdown } from './memory';
import { correctKeyframeRS } from './refine-camera';
import { ENGINE_DEFAULTS, modeConfig, type QualityPresetConfig } from './presets';
import type { EngineConfig, KeyframeRecord } from './types';

const DEG = Math.PI / 180;

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

/** Services the live session needs from the engine. */
export interface LiveDeps {
  mode: CaptureMode;
  quality: QualityPreset;
  hfovPriorDeg: number | null;
  config: EngineConfig;
  preset: QualityPresetConfig;
  store: EngineStore;
  pipeline(): FeaturePipeline;
  /** Run GPU work under the engine's one-command-buffer lock (and only while visible). */
  gpu<T>(fn: () => Promise<T>): Promise<T>;
  hidden(): boolean;
  log(level: 'debug' | 'info' | 'warn' | 'error', message: string, data?: unknown): void;
  emitLive(u: LiveUpdate): void;
  storageError(e: unknown): void;
  /** Tab memory budget in MB (device-derived; default ENGINE_DEFAULTS.memoryBudgetMB). */
  memoryBudgetMB?(): number;
  /** The capture reached its memory cap (no more keyframes); called once. */
  memoryLimit?(message: string): void;
}

/** Sorted, de-duplicated IMU history of the capture (+ chunked persistence). */
export class ImuLog {
  samples: ImuSample[] = [];
  private persisted = 0;

  append(list: readonly ImuSample[] | null | undefined): void {
    if (!list) return;
    for (const s of list) {
      if (!s || !Number.isFinite(s.t) || !s.gyro) continue;
      const n = this.samples.length;
      if (n === 0 || s.t > this.samples[n - 1].t) {
        this.samples.push(s);
        continue;
      }
      // out of order (rare): insert unless a sample with the same t exists
      let lo = 0,
        hi = n;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (this.samples[mid].t < s.t) lo = mid + 1;
        else hi = mid;
      }
      if (lo < n && this.samples[lo].t === s.t) continue;
      this.samples.splice(lo, 0, s);
      if (lo < this.persisted) this.persisted++;
    }
  }

  /** Samples not yet persisted (marks them persisted). */
  takeChunk(): ImuSample[] {
    const c = this.samples.slice(this.persisted);
    this.persisted = this.samples.length;
    return c;
  }

  load(samples: ImuSample[]): void {
    this.samples = [];
    this.append(samples.sort((a, b) => a.t - b.t));
    this.persisted = this.samples.length;
  }

  /** True when the samples cover [t0, t1] (within `slackMs`) without a gap > IMU_MAX_GAP_MS (as GyroReference.covers). */
  covers(t0: number, t1: number, slackMs = 60): boolean {
    const n = this.samples.length;
    if (n < 2) return false;
    const lo = Math.min(t0, t1) + slackMs,
      hi = Math.max(t0, t1) - slackMs;
    if (!(this.samples[0].t <= lo && this.samples[n - 1].t >= hi)) return false;
    // first sample after lo, then every sample interval that starts before hi
    let a = 1,
      b = n;
    while (a < b) {
      const mid = (a + b) >> 1;
      if (this.samples[mid].t <= lo) a = mid + 1;
      else b = mid;
    }
    for (let i = a; i < n && this.samples[i - 1].t < hi; i++) if (this.samples[i].t - this.samples[i - 1].t > IMU_MAX_GAP_MS) return false;
    return true;
  }
}

/** Everything recorded during capture (shared with refine / export). */
export class CaptureState {
  readonly db = new FeatureDatabase();
  readonly graph = new CorrespondenceGraph();
  readonly records = new Map<number, KeyframeRecord>();
  /** Keyframe ids in capture order. */
  readonly order: number[] = [];
  readonly imu = new ImuLog();
  geo: GeoFix[] = [];
  /** pairKey of every pair tried (verified or not). */
  readonly attempted = new Set<number>();
  /** Stored geometry per pair (for persistence of refine pairs). */
  vocab: VocabularyIndex | null = null;
  vocabTrainedOn: number[] | null = null;
  thetaSession: ScreenAngle | null = null;
  /** θ whose R_cam_device describes the canonical images. */
  cameraTheta: ScreenAngle | null = null;
  cameraThetaVerified = false;
  /** Frame delivery path (OrientationTracker.rawDelivery): raw sensor buffers, upright frames, unknown. */
  rawDelivery: boolean | null = null;
  resolvedMode: 'object' | 'scene' | null = null;
  nextKeyframeId = 0;
  frameCounter = 0;
  imuChunks = 0;
  captureFinished = false;
  timeOffsetMs: number | null = null;
  /** Gyro bias (rad/s, device frame) estimated from the live keyframes; null until estimated. */
  gyroBias: [number, number, number] | null = null;
  /**
   * Rolling-shutter model applied to the keyframe keypoints (in place; null = none) and the
   * original keypoints of every corrected keyframe (src/engine/refine-camera.ts).
   */
  rollingShutter: RollingShutterModel | null = null;
  readonly rsOriginal = new Map<number, Float32Array>();
  /**
   * Camera angular velocity (rad/s, camera frame) each corrected keyframe's keypoints were corrected
   * with: the dense stage / exports must correct the images with exactly these (not a later gyro
   * query, whose offset / bias / R_cam_device may have changed since).
   */
  readonly rsOmega = new Map<number, Float64Array>();
  /** Corrected keyframes whose correction included the translational term. */
  readonly rsTranslated = new Set<number>();

  /** R_cam_device of the canonical images. */
  cameraFromDevice(): Mat3 {
    return cameraFromDevice(this.cameraTheta ?? this.thetaSession ?? 0);
  }

  /** Keyframe ids in capture order that the store / DB knows. */
  keyframeIds(): number[] {
    return this.order.slice();
  }
}

interface TrackingRef {
  id: number;
  fs: FeatureSet;
  t: number;
  width: number;
  height: number;
}

/** A frame kept by best-of-window keyframing (its prepared image, features and tracking geometry). */
interface HeldFrame {
  f: FrameInput;
  frameIndex: number;
  prep: PreparedFrame;
  fs: FeatureSet;
  k: number;
  trackGeom: TwoViewGeometry | null;
  refId: number | null;
  width: number;
  height: number;
}

interface MappingJob {
  id: number;
  /** Geometry of (reference keyframe → this keyframe) from the per-frame tracking. */
  tracking: { other: number; geom: TwoViewGeometry } | null;
}

/** Copy of a geometry whose arrays are not shared with mutable scratch. */
function usable(g: TwoViewGeometry | null | undefined, minInliers: number): g is TwoViewGeometry {
  return !!g && g.config !== 'degenerate' && g.inliers.length >> 1 >= minInliers;
}

function kMatrix(cam: CameraIntrinsics): Float64Array {
  return Float64Array.of(cam.f, 0, cam.cx, 0, cam.f, cam.cy, 0, 0, 1);
}

function kInverse(cam: CameraIntrinsics): Float64Array {
  const fi = 1 / cam.f;
  return Float64Array.of(fi, 0, -cam.cx * fi, 0, fi, -cam.cy * fi, 0, 0, 1);
}

function mat3TransposeOf(R: ArrayLike<number>): Float64Array {
  return Float64Array.of(R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]);
}

function mat3MulVecOf(R: ArrayLike<number>, v: ArrayLike<number>): Float64Array {
  return Float64Array.of(R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]);
}

function negate(v: ArrayLike<number>): Float64Array {
  return Float64Array.of(-v[0], -v[1], -v[2]);
}

function poseCentre(p: Pose): Float64Array {
  const R = p.R,
    t = p.t;
  return Float64Array.of(-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]));
}

/** Result of the R_cam_device verification. */
export interface HandEyeResult {
  /** Median rotation error (deg) per hypothesis θ = 0, 90, 180, 270. */
  medianErrDeg: number[];
  pairs: number;
  best: ScreenAngle;
}

/**
 * Compare visual relative rotations R_j R_iᵀ of consecutive registered keyframes with the gyro
 * rotation R_cd(θ) · R_{d(tj)←d(ti)} · R_cd(θ)ᵀ for the 4 quarter-turn hypotheses.
 */
export function handEyeCheck(
  kfs: readonly { t: number; pose: Pose }[],
  imu: ImuLog,
  opts: { maxGapMs?: number; minRotationDeg?: number; offsetMs?: number } = {},
): HandEyeResult | null {
  const maxGap = opts.maxGapMs ?? 2500;
  const minRot = (opts.minRotationDeg ?? 2) * DEG;
  const off = opts.offsetMs ?? 0;
  const sorted = [...kfs].sort((a, b) => a.t - b.t);
  const errs: number[][] = [[], [], [], []];
  const hyp = [0, 90, 180, 270].map((th) => cameraFromDevice(th));
  const G = new Float64Array(9);
  for (let i = 0; i + 1 < sorted.length; i++) {
    const a = sorted[i],
      b = sorted[i + 1];
    if (b.t - a.t > maxGap || !imu.covers(a.t + off, b.t + off)) continue;
    integrateGyro(imu.samples, a.t + off, b.t + off, G);
    if (rotationAngle(G) < minRot) continue;
    const V = mat3MulTranspose(b.pose.R, a.pose.R); // R_b R_aᵀ: cam_a → cam_b
    for (let h = 0; h < 4; h++) {
      const P = mat3MulTranspose(mat3Mul(hyp[h], G), hyp[h]);
      errs[h].push(rotationAngle(mat3MulTranspose(V, P)));
    }
  }
  const n = errs[0].length;
  if (n === 0) return null;
  const med = errs.map((e) => median(e) / DEG);
  let best = 0;
  for (let h = 1; h < 4; h++) if (med[h] < med[best]) best = h;
  return { medianErrDeg: med, pairs: n, best: (best * 90) as ScreenAngle };
}

/** Direction pointing up (device frame) in a phone held in UI rotation θ: (sin θ, cos θ, 0). */
const DEVICE_UP: Record<ScreenAngle, [number, number, number]> = { 0: [0, 1, 0], 90: [1, 0, 0], 180: [0, -1, 0], 270: [-1, 0, 0] };

/**
 * Physical up in the camera frame of a keyframe (unit; LiveCamera.upCam): −gravityCam when gravity
 * is known, else up for the UI rotation the frame was captured in, R_cam_device · DEVICE_UP[θ_ui].
 * The canonical image is upright for θ₀, not for θ_ui (raw sensor buffers: θ₀ = 90 in a portrait UI).
 */
export function keyframeUpCam(r: KeyframeRecord): [number, number, number] | undefined {
  const g = r.info.gravityCam;
  const gn = g ? Math.hypot(g[0], g[1], g[2]) : 0;
  if (g && gn > 1e-9) return [-g[0] / gn, -g[1] / gn, -g[2] / gn];
  if (!Number.isFinite(r.screenAngle)) return undefined;
  const u = rotateVec(r.info.R_cam_device, DEVICE_UP[snapScreenAngle(r.screenAngle)]);
  return [u[0], u[1], u[2]];
}

/** Decide 'object' vs 'scene' from the registered cameras (optical-axis convergence). */
export function classifyScene(poses: readonly Pose[], medianDepth: number): 'object' | 'scene' | null {
  if (poses.length < 6) return null;
  const c = opticalAxisConvergence(poses);
  if (!c.point) return 'scene';
  const dMed = c.distMedian;
  const depthOk = !(medianDepth > 0) || (dMed < 2.5 * medianDepth && dMed > 0.3 * medianDepth);
  return c.converging && depthOk ? 'object' : 'scene';
}

/** Live capture session. */
export class LiveSession {
  readonly state = new CaptureState();
  readonly selector: KeyframeSelector;
  readonly orientation: OrientationTracker;
  readonly mapper: LiveMapper;
  private ref: TrackingRef | null = null;
  private segment: number | null = null;
  private queue: Promise<void> = Promise.resolve();
  private queued = 0;
  private readonly writes = new Set<Promise<unknown>>();
  /** Bytes held by writes not yet completed (JPEG canvases / blobs, features, colours). */
  private pendingWriteBytes = 0;
  /** Capture memory estimate after the last mapped keyframe (bytes) and when it was last logged. */
  private memEstimate = 0;
  private memLoggedAt = 0;
  private memWarned = false;
  private consecutiveUnregistered = 0;
  private lastFrameInliers = NaN;
  private lastGuidance: Guidance = 'initializing';
  private lastDecision: LiveUpdate['decision'] = 'skip';
  private lastFrameIndex = -1;
  private modelVersion = 0;
  private sentVersion = -1;
  private disposed = false;
  private readonly minPairInliers = ENGINE_DEFAULTS.minPairInliers;
  /** Accumulated diagnostics. */
  readonly stats = {
    frames: 0,
    keyframes: 0,
    mappingMs: 0,
    featureMs: 0,
    trackingMs: 0,
    guidedFallbacks: 0,
    handEye: null as HandEyeResult | null,
    /** Keyframes forced by the hard rotation limit (fast pans). */
    forcedKeyframes: 0,
    /** Verified pairs / registrations / initial pairs rejected by the gyro check. */
    gyroRejectedPairs: 0,
    gyroRejectedPoses: 0,
    /** Planar pairs whose two-fold ambiguity the gyro resolved (alternative branch taken). */
    gyroPlanarSwaps: 0,
    /** Live gyro calibrations (bias / time offset from the registered keyframes). */
    gyroCalibrations: 0,
    /** Keyframes taken from an earlier, better frame of a deferral window (best-of-window). */
    heldKeyframes: 0,
  };
  /** Frame held by best-of-window keyframing (null = none). */
  private held: HeldFrame | null = null;
  /** Low-feature frames re-extracted from gray input to check the GPU bitmap upload. */
  private bitmapProbes = 0;
  private static readonly MAX_BITMAP_PROBES = 8;
  /** The GPU bitmap upload gave far fewer features than gray input: use gray input. */
  private bitmapInputBroken = false;
  /** Bias-aware gyro integration (camera frame) shared by tracking, gates and refine. */
  private readonly gyroRef = new GyroReference();
  private gyroCalibratedAt = 0;

  constructor(private readonly deps: LiveDeps) {
    const mc = modeConfig(deps.mode);
    this.selector = new KeyframeSelector({ mode: mc.keyframeMode, ...mc.keyframe, ...(deps.config.keyframe ?? {}) });
    // RS-aware motion gate: with the device's rolling shutter corrected live, fast rotation no
    // longer skews the keypoints — only blur limits the angular speed (the sharpness gate sees it)
    if (deps.config.rollingShutter && deps.config.rollingShutter.readoutMs > 0 && deps.config.keyframe?.maxAngularSpeed === undefined) {
      this.selector.options.maxAngularSpeed = Math.max(this.selector.options.maxAngularSpeed, deps.config.rsMaxAngularSpeed ?? 2.0);
    }
    this.orientation = new OrientationTracker(deps.config.settleMs ?? ENGINE_DEFAULTS.settleMs);
    this.mapper = new LiveMapper(this.state.db, this.state.graph, {
      hfovPriorDeg: deps.hfovPriorDeg ?? 69,
      rotationCheck: this.rotationCheck(),
      ...(deps.config.gyroAidedRegistration === false ? {} : { predictRotation: (id: number, rec: Reconstruction) => this.predictRotation(id, rec) }),
      reportReprojError: false, // never read here; a full reprojection per keyframe
      ...(deps.config.live ?? {}),
    });
  }

  private get inline(): boolean {
    return this.deps.config.mapping === 'inline';
  }

  // ------------------------------------------------------------------------------------------
  // geometry helpers

  /** Current intrinsics guess for a canonical image size (live-refined when available). */
  intrinsics(width: number, height: number): CameraIntrinsics {
    const rec = this.mapper.reconstruction;
    if (this.mapper.initialized) {
      for (const cam of rec.cameras.values()) if (cam.width === width && cam.height === height) return { ...cam };
    }
    return intrinsicsFromFov(width, height, this.deps.hfovPriorDeg ?? 69);
  }

  /** Gyro rotation of the camera between t0 and t1 (camera-t0 → camera-t1 frame), or null. */
  gyroRotation(t0: number, t1: number): Mat3 | null {
    return this.gyro().relative(t0, t1);
  }

  /**
   * The capture's gyro (bias-compensated once estimated, camera–IMU offset, current R_cam_device),
   * synchronised with the IMU log.
   */
  gyro(): GyroReference {
    const st = this.state;
    const g = this.gyroRef;
    g.offsetMs = st.timeOffsetMs ?? 0;
    g.R_cam_device = st.cameraFromDevice();
    const b = st.gyroBias;
    const cur = g.bias;
    if ((b === null) !== (cur === null) || (b && cur && (b[0] !== cur[0] || b[1] !== cur[1] || b[2] !== cur[2]))) g.setBias(b, st.imu.samples);
    else g.sync(st.imu.samples);
    return g;
  }

  /**
   * The gyro once the camera←device mapping is verified (hand-eye check) and ≥ 2 IMU samples exist;
   * null otherwise — a wrong R_cam_device would reject every correct pair and predict wrong rotations.
   */
  verifiedGyro(): GyroReference | null {
    if (!this.state.cameraThetaVerified) return null;
    const g = this.gyro();
    return g.numSamples >= 2 ? g : null;
  }

  /**
   * The gyro for consistency checks (pairs, registrations, models): `verifiedGyro` unless the
   * checks are disabled (`EngineConfig.gyroGate: false`). Null otherwise.
   */
  gyroForChecks(): GyroReference | null {
    return this.deps.config.gyroGate === false ? null : this.verifiedGyro();
  }

  /**
   * Tolerance of the gyro checks. The drift allowance (0.3°/s) is NOT reduced once a bias is
   * estimated: the estimate comes from vision rotations that rolling shutter biases (e2e yard-real:
   * x component 0.5 instead of 3.0 mrad/s ⇒ 5° over a 32 s loop closure), and correct loop-closure
   * pairs were rejected with 0.1°/s.
   */
  gyroTolerance(): Partial<GyroToleranceOptions> {
    return { ...(this.deps.config.gyroTolerance ?? {}) };
  }

  /**
   * Gyro check of a verified pair (a → b): null when its rotation disagrees with the gyro (false
   * matches on repeated structure); a planar pair whose ambiguous alternative branch agrees is
   * switched to it. Pairs without a rotation or without gyro coverage pass unchanged.
   */
  gatePair(a: number, b: number, g: TwoViewGeometry | null): TwoViewGeometry | null {
    if (!g || g.config === 'degenerate' || !g.R) return g;
    const gyro = this.gyroForChecks();
    const ra = this.state.records.get(a),
      rb = this.state.records.get(b);
    if (!gyro || !ra || !rb) return g;
    const tol = this.gyroTolerance();
    const c = checkPairRotation(gyro, ra.t, rb.t, g.R, tol);
    if (!c || c.ok) return g;
    const alt = (g as TwoViewGeometry & { planarAlternative?: { R: Mat3; t: Float64Array } }).planarAlternative;
    if (alt) {
      const c2 = checkPairRotation(gyro, ra.t, rb.t, alt.R, tol);
      if (c2 && c2.ok) {
        this.stats.gyroPlanarSwaps++;
        return { ...g, R: alt.R, t: alt.t };
      }
    }
    this.stats.gyroRejectedPairs++;
    this.deps.log('debug', `pair ${a}-${b} rejected by the gyro: rotation ${(c.errRad / DEG).toFixed(1)}° off (tolerance ${(c.tolRad / DEG).toFixed(1)}°)`);
    return null;
  }

  /** Gyro check of a new absolute pose against the registered keyframes nearest in time (majority vote). */
  checkPoseGyro(id: number, pose: Pose, rec: Reconstruction): boolean {
    const gyro = this.gyroForChecks();
    const r = this.state.records.get(id);
    if (!gyro || !r) return true;
    const nbs: TimedRotation[] = [];
    for (const rid of rec.registeredImageIds()) {
      if (rid === id) continue;
      const rr = this.state.records.get(rid);
      const im = rec.images.get(rid);
      if (rr && im) nbs.push({ id: rid, t: rr.t, R: im.pose.R });
    }
    const c = checkPoseRotation(gyro, r.t, pose.R, nbs, { tolerance: this.gyroTolerance() });
    if (!c || c.ok) return true;
    this.stats.gyroRejectedPoses++;
    this.deps.log('debug', `registration of keyframe ${id} rejected by the gyro: rotation ${(c.errRad / DEG).toFixed(1)}° off (${c.disagree}/${c.agree + c.disagree} neighbours disagree)`);
    return false;
  }

  /** `MapperOptions.rotationCheck` backed by the capture's gyro (live and offline mapping). */
  rotationCheck(): NonNullable<MapperOptions['rotationCheck']> {
    return {
      pair: (a, b, R) => {
        const g = this.gatePair(a, b, { config: 'calibrated', inliers: new Uint32Array(0), R });
        return g !== null;
      },
      pose: (id, pose, rec) => this.checkPoseGyro(id, pose, rec),
    };
  }

  /**
   * Debug log of the registration attempts of an `addKeyframe` call that were not a plain P3P
   * registration (re-triangulated neighbours, the gyro-predicted rotation, or still pending), e.g.
   * "keyframe 19 registered (P3P): 36 map points → 96 re-triangulated; P3P no pose (36
   * correspondences), then P3P 84/99".
   */
  private logRegistrations(res: LiveKeyframeResult): void {
    for (const a of res.registrations) {
      if (a.registered && a.calls.length === 1 && a.calls[0].via === 'p3p' && Number.isNaN(a.visibleRetriangulated)) continue;
      const calls = a.calls.map((c) => {
        const parts: string[] = [];
        const count = (k: number): string => (k > 0 ? `${k}/${c.correspondences}` : `no pose (${c.correspondences} correspondences)`);
        if (!Number.isNaN(c.p3pInliers)) parts.push(`P3P ${count(c.p3pInliers)}`);
        if (!Number.isNaN(c.knownRotationInliers)) {
          const notes: string[] = [];
          if (Number.isFinite(c.rotationChangeDeg)) notes.push(`moved ${c.rotationChangeDeg.toFixed(2)}°`);
          if (Number.isFinite(c.pairBaselineDeg)) notes.push(`baseline ${c.pairBaselineDeg.toFixed(1)}° off its best pair's`);
          parts.push(`known rotation ${count(c.knownRotationInliers)}${notes.length ? ` (${notes.join(', ')})` : ''}`);
        }
        return parts.length ? parts.join(', ') : `${c.correspondences} correspondences`;
      });
      const via = a.calls[a.calls.length - 1]?.via;
      const state = a.registered ? `registered (${via === 'knownRotation' ? 'gyro rotation' : 'P3P'})` : 'pending';
      const retri = Number.isNaN(a.visibleRetriangulated) ? '' : ` → ${a.visibleRetriangulated} re-triangulated`;
      this.deps.log('debug', `keyframe ${a.id} ${state}: ${a.visible} map points${retri}; ${calls.join(', then ')}`);
    }
  }

  /** Longest |Δt| (ms) over which `predictRotation` propagates a registered keyframe's rotation. */
  private static readonly PREDICT_MAX_GAP_MS = 3000;

  /**
   * `MapperOptions.predictRotation` (gyro-aided registration; `EngineConfig.gyroAidedRegistration`):
   * the rotation of keyframe `id` in `rec`'s frame, R̂ = R_c(t_n → t) · R_n from the registered
   * keyframe n nearest in time (|Δt| ≤ 3 s, IMU-covered), with σ = 1° + 0.3°/s·|Δt| + 3 % of the
   * gyro rotation (integration noise, the uncompensated bias and camera–IMU offset during capture,
   * rolling shutter; the bias-compensated gyro is within 0.2–1.6° of the truth on the e2e captures).
   * Null until R_cam_device is verified (`verifiedGyro`; independent of `EngineConfig.gyroGate`).
   */
  predictRotation(id: number, rec: Reconstruction): { R: Mat3; sigmaRad: number } | null {
    const gyro = this.verifiedGyro();
    const r = this.state.records.get(id);
    if (!gyro || !r) return null;
    const near: { t: number; R: Mat3; dt: number }[] = [];
    for (const rid of rec.registeredImageIds()) {
      const rr = rid === id ? undefined : this.state.records.get(rid);
      const im = rec.images.get(rid);
      if (!rr || !im) continue;
      const dt = Math.abs(rr.t - r.t);
      if (dt <= LiveSession.PREDICT_MAX_GAP_MS) near.push({ t: rr.t, R: im.pose.R, dt });
    }
    near.sort((a, b) => a.dt - b.dt);
    for (const nb of near) {
      const G = gyro.relative(nb.t, r.t);
      if (!G) continue;
      return { R: mat3Mul(G, nb.R), sigmaRad: (1 + 0.3e-3 * nb.dt) * DEG + 0.03 * rotationAngle(G) };
    }
    return null;
  }

  /**
   * Opt-in (`EngineConfig.liveGyroCalibration: true`): gyro bias and camera–IMU offset from the
   * registered live keyframes every 10 registrations once 12 are registered and R_cam_device is
   * verified (`calibrateGyro`). Off by default: on the e2e captures it changed the keyframe
   * selection (rotation since the last keyframe) without a measurable benefit and the final
   * accuracy was equal or worse (yard ATE 2.7e-4 vs 2.1e-4, yard-real 7.0e-4 vs 3.3e-4); refine
   * calibrates once on the refined live model instead.
   */
  private maybeCalibrateGyro(): void {
    const st = this.state;
    if (this.deps.config.liveGyroCalibration !== true || !st.cameraThetaVerified) return;
    const n = this.mapper.reconstruction.numRegisteredImages;
    if (n < 12 || n - this.gyroCalibratedAt < 10) return;
    this.gyroCalibratedAt = n;
    this.calibrateGyro('live');
  }

  /**
   * Gyro bias + camera–IMU offset from the registered keyframes of the live model
   * (`estimateTimeOffset`: vision vs gyro relative rotations, robust; the bias is adopted with ≥ 8
   * consistent pairs, the offset when reliable). Refine calls it on the refined live model before
   * the rolling-shutter estimation (which needs ω at the right time); `maybeCalibrateGyro` during
   * capture when enabled. Returns true when the bias (and, if reliable, the offset) was updated.
   */
  calibrateGyro(label: 'live' | 'refine' = 'live'): boolean {
    const st = this.state;
    if (!st.cameraThetaVerified || st.imu.samples.length < 10) return false;
    const kfs = this.registeredKeyframes();
    if (kfs.length < 12) return false;
    const Rcd = st.cameraFromDevice();
    const ik: InertialKeyframe[] = kfs.map((k) => ({ t: k.t, R_wc: mat3TransposeOf(k.pose.R), C: poseCentre(k.pose), R_cam_device: Rcd }));
    const cur = st.timeOffsetMs ?? 0;
    const t0 = now();
    const r = estimateTimeOffset(ik, st.imu.samples, { initialOffsetMs: cur, searchRangeMs: st.gyroBias ? 60 : 150, maxPairGapMs: 2000 });
    if (!r || r.numInliers < 8 || ![...r.gyroBias].every(Number.isFinite)) return false;
    st.gyroBias = [r.gyroBias[0], r.gyroBias[1], r.gyroBias[2]];
    if (r.reliable) st.timeOffsetMs = r.offsetMs;
    this.stats.gyroCalibrations++;
    this.deps.log(
      'info',
      `${label} gyro calibration (${kfs.length} keyframes, ${Math.round(now() - t0)} ms): bias ${st.gyroBias.map((x) => (x * 1000).toFixed(1)).join(', ')} mrad/s, ` +
        `time offset ${r.reliable ? `${r.offsetMs.toFixed(1)} ms` : `kept ${cur.toFixed(1)} ms (σ ${r.stdMs.toFixed(1)} ms)`}`,
    );
    this.persistState();
    return true;
  }

  // ------------------------------------------------------------------------------------------
  // frames

  private trackingState(): TrackingState {
    if (!this.mapper.initialized) return 'initializing';
    const lostFrame = Number.isFinite(this.lastFrameInliers) && this.lastFrameInliers < this.selector.options.minInliers;
    return lostFrame || this.consecutiveUnregistered >= 3 ? 'lost' : 'tracking';
  }

  private baseUpdate(frameIndex: number, decision: LiveUpdate['decision'], guidance: Guidance, timings: Record<string, number>): LiveUpdate {
    const rec = this.mapper.reconstruction;
    return {
      frameIndex,
      decision,
      guidance,
      tracking: this.trackingState(),
      numKeyframes: this.state.order.length,
      numRegistered: rec.numRegisteredImages,
      numPoints: rec.numPoints,
      timings,
    };
  }

  private mapGuidance(g: string | undefined): Guidance {
    const tr = this.trackingState();
    if (tr === 'lost') return 'tracking_lost';
    const base: Guidance = g === 'move_slower' || g === 'more_texture' || g === 'translate_more' || g === 'tracking_lost' ? g : 'good';
    if (base === 'good' && tr === 'initializing') return 'initializing';
    return base;
  }

  /** Process one frame (the engine closes / owns the bitmap). */
  async processFrame(f: FrameInput): Promise<LiveUpdate> {
    const t0 = now();
    const st = this.state;
    const frameIndex = typeof f.index === 'number' && Number.isFinite(f.index) ? f.index : st.frameCounter;
    st.frameCounter = Math.max(st.frameCounter + 1, frameIndex + 1);
    this.lastFrameIndex = frameIndex;
    this.stats.frames++;
    st.imu.append(f.imu);
    if (f.geo && Number.isFinite(f.geo.lat) && Number.isFinite(f.geo.lon)) st.geo = mergeGeoFixes(st.geo, [f.geo]);
    const timings: Record<string, number> = {};
    const dbg = this.newDebug();
    const finish = (u: LiveUpdate): LiveUpdate => {
      timings.total = now() - t0;
      u.debug = dbg;
      this.lastDecision = u.decision;
      this.lastGuidance = u.guidance;
      return u;
    };
    if (this.disposed || this.deps.hidden() || st.captureFinished) {
      closeSource(f.bitmap);
      return finish(this.baseUpdate(frameIndex, 'skip', this.lastGuidance, timings));
    }
    // new camera segment (interruption): new tracking reference
    const seg = typeof f.segment === 'number' ? f.segment : 0;
    if (this.segment !== null && seg !== this.segment) {
      this.selector.resetTracking();
      this.releaseHeld();
      this.ref = null;
    }
    this.segment = seg;
    const verdict = this.orientation.classify({
      width: f.bitmap.width,
      height: f.bitmap.height,
      screenAngle: f.screenAngle,
      t: f.t,
      raw: f.rawBuffer === true,
      screenAngleSince: f.screenAngleSince,
    });
    dbg.srcWidth = f.bitmap.width;
    dbg.srcHeight = f.bitmap.height;
    if (!verdict.ok) {
      dbg.reason = `dropped: ${verdict.reason}`;
      closeSource(f.bitmap);
      this.deps.log('debug', `frame ${frameIndex} dropped: ${verdict.reason}`);
      return finish(this.baseUpdate(frameIndex, 'skip', this.lastGuidance, timings));
    }
    const ts = this.orientation.thetaSession;
    if (st.thetaSession === null) st.thetaSession = ts;
    else if (ts !== null && ts !== st.thetaSession) this.reinterpretSessionTheta(ts);
    if (st.cameraTheta === null) st.cameraTheta = st.thetaSession;
    if (st.rawDelivery !== this.orientation.rawDelivery) {
      st.rawDelivery = this.orientation.rawDelivery;
      this.persistState();
    }
    if (verdict.raw) this.deps.log('debug', `frame ${frameIndex}: raw sensor-orientation buffer (treated as θ=90)`);
    let prep: PreparedFrame;
    try {
      prep = prepareFrame(f.bitmap, verdict.k, ENGINE_DEFAULTS.smallMaxSide);
    } catch (e) {
      closeSource(f.bitmap);
      throw e;
    }
    let handedOff = false;
    try {
      timings.prep = now() - t0;
      // overlay orientation: canonical → the frame as submitted (inverse of the canonical turn)
      const kOut = (4 - verdict.k) % 4;
      const W = prep.width,
        H = prep.height;
      dbg.width = W;
      dbg.height = H;
      dbg.sharpness = prep.sharpness;
      const lm = lumaStats(prep.small);
      dbg.lumaMean = lm.mean;
      dbg.lumaStd = lm.std;
      const angSpeed = angularSpeedAt(st.imu.samples, f.t + (st.timeOffsetMs ?? 0));
      // gyro rotation since the last keyframe (hard rotation limit: fast pans force keyframes)
      const Rk = this.ref ? this.gyroRotation(this.ref.t, f.t) : null;
      const cand: KeyframeCandidate = {
        t: f.t,
        sharpness: prep.sharpness,
        angularSpeed: angSpeed,
        numFeatures: 0,
        numMatchesToLastKF: 0,
        numInliersToLastKF: NaN,
        medianParallaxPx: NaN,
        rotationSinceLastKFRad: Rk ? rotationAngle(Rk) : NaN,
        imageWidth: W,
      };
      // pre-gate: blurry / fast-rotating frames are rejected without feature extraction (the
      // selector's own gate, relaxed for frames the hard rotation limit forces)
      const gate = this.selector.gate(cand);
      dbg.sharpnessRef = gate.reference;
      dbg.angularSpeed = angSpeed;
      if (gate.reject) {
        const d = this.selector.evaluate(cand);
        dbg.reason = `${d.decision}: ${d.reason}`;
        return finish(this.baseUpdate(frameIndex, d.decision, this.mapGuidance(d.guidance), timings));
      }
      // features (the pipeline is read when the GPU job runs: a GPU-loss recovery queued ahead of
      // it has replaced it by then; either input kind works on either backend)
      const tf = now();
      const fi = prep.featureInput(this.deps.pipeline().gpu && !this.bitmapInputBroken, this.featureMaxSide());
      let fs = await this.deps.gpu(() => {
        const pipeline = this.deps.pipeline();
        return fi.kind === 'bitmap' ? pipeline.extractFromBitmap(fi.source, W, H) : pipeline.extract(fi.gray, W, H);
      });
      if (fi.kind === 'bitmap' && fs.count < this.selector.options.minFeatures && this.bitmapProbes < LiveSession.MAX_BITMAP_PROBES) {
        // The GPU bitmap upload (copyExternalImageToTexture) is not covered by the start-up
        // self-test, and on some WebKit builds it yields a blank texture for camera frames
        // (every frame "not enough detail"). Re-extract from the 2D-canvas gray image; if that
        // finds clearly more, use gray input for the rest of the session.
        this.bitmapProbes++;
        const gi = prep.featureInput(false, this.featureMaxSide());
        if (gi.kind === 'gray') {
          const gfs = await this.deps.gpu(() => this.deps.pipeline().extract(gi.gray, W, H));
          if (gfs.count >= this.selector.options.minFeatures && gfs.count > 4 * fs.count) {
            this.bitmapInputBroken = true;
            this.deps.log('warn', `GPU bitmap input gave ${fs.count} features, gray input ${gfs.count}: using gray input for this session`);
            fs = gfs;
          }
        }
      }
      timings.features = now() - tf;
      this.stats.featureMs += timings.features;
      cand.numFeatures = fs.count;
      dbg.numFeatures = fs.count;
      dbg.featureInput = fi.kind === 'bitmap' && !this.bitmapInputBroken ? 'bitmap' : 'gray';
      // tracking against the last keyframe
      let trackGeom: TwoViewGeometry | null = null;
      if (this.ref && (this.ref.width !== W || this.ref.height !== H)) {
        // frame size changed (camera format switch): start a new tracking segment
        this.deps.log('warn', `frame size changed ${this.ref.width}x${this.ref.height} → ${W}x${H}: new tracking segment`);
        this.selector.resetTracking();
        this.releaseHeld();
        this.ref = null;
      }
      const ref = this.ref;
      let framePose: Pose | null = null;
      if (ref && fs.count > 0 && ref.fs.count > 0) {
        const tm = now();
        const tr = await this.trackAgainst(ref, fs, f.t, timings);
        trackGeom = tr.geom;
        if (tr.geom) {
          const tp = now();
          framePose = this.framePose(ref.id, tr.geom.inliers, fs);
          timings.framePose = now() - tp;
        }
        cand.numMatchesToLastKF = tr.numMatches;
        cand.numInliersToLastKF = tr.numInliers;
        cand.medianParallaxPx = tr.parallax;
        cand.rotationSinceLastKFRad = tr.rotation;
        this.lastFrameInliers = tr.numInliers;
        this.stats.trackingMs += now() - tm;
      } else {
        this.lastFrameInliers = NaN;
      }
      let decision = this.selector.evaluate(cand);
      dbg.inliers = cand.numInliersToLastKF;
      dbg.reason = decision.reason;
      if (decision.decision === 'keyframe' && decision.forced) this.stats.forcedKeyframes++;
      if (decision.decision === 'keyframe' && decision.useHeld && !this.held) {
        // the engine keeps every frame the selector holds — defensive: keyframe = this frame
        this.selector.revertLastKeyframe();
        this.selector.commitKeyframe(f.t);
        decision = { ...decision, useHeld: false };
      }
      if (decision.decision === 'keyframe' && (st.captureFinished || this.disposed)) {
        // finishCapture / refine started while this frame was being analysed: no new keyframe
        // (its mapping job would change the live model under refine)
        this.selector.revertLastKeyframe();
        decision = { ...decision, decision: 'skip' };
      }
      if (decision.decision === 'keyframe' && this.captureMemoryFull()) {
        // memory cap: refine peaks at ≈ 1.9× the capture state — keep tracking, add no
        // more keyframes (the app should offer to finish the scan)
        this.selector.revertLastKeyframe();
        decision = { ...decision, decision: 'skip' };
        timings.memoryLimited = 1;
      }
      const overlay = overlayPoints(fs, W, H, kOut, this.deps.config.overlayMaxPoints ?? ENGINE_DEFAULTS.overlayMaxPoints);
      // best-of-window keyframing: the selector holds this frame as the best of a deferral window,
      // or makes the held frame the keyframe (one frame kept at most)
      let src: HeldFrame = { f, frameIndex, prep, fs, k: verdict.k, trackGeom, refId: ref ? ref.id : null, width: W, height: H };
      if (decision.decision === 'keyframe' && decision.useHeld && this.held) {
        src = this.held;
        this.held = null;
        this.stats.heldKeyframes++;
        // the keyframe is an EARLIER frame than the one this update answers (for consumers that map
        // keyframes to frames by the 'keyframe' decision, e.g. the e2e live-trajectory metric)
        timings.keyframeFrameIndex = src.frameIndex;
      } else if (decision.hold && !st.captureFinished && !this.disposed) {
        this.releaseHeld();
        this.held = src;
        handedOff = true;
      } else if (this.selector.heldTime === null || st.captureFinished) this.releaseHeld();
      let mapped: Promise<void> | null = null;
      if (decision.decision === 'keyframe') {
        const tk = now();
        const id = this.createKeyframe(src.f, src.frameIndex, src.prep, src.fs, src.k);
        if (src.prep === prep) handedOff = true;
        const job: MappingJob = { id, tracking: src.refId !== null && src.trackGeom ? { other: src.refId, geom: src.trackGeom } : null };
        this.ref = { id, fs: src.fs, t: src.f.t, width: src.width, height: src.height };
        mapped = this.enqueue(job);
        timings.keyframe = now() - tk;
      }
      if (mapped && this.inline) {
        await mapped;
        timings.mapping = this.lastJobMs;
      } else if (mapped) {
        // back-pressure: keep the live model (and the guidance derived from it) close to the capture
        const maxBacklog = this.deps.config.maxMappingBacklog ?? ENGINE_DEFAULTS.maxMappingBacklog;
        if (maxBacklog >= 0 && this.queued > maxBacklog) {
          const tw = now();
          while (this.queued > maxBacklog && !this.disposed && this.jobs.length) await this.jobs[0];
          timings.mappingWait = now() - tw;
        }
      }
      if (mapped) {
        // back-pressure of the store: a slow / stalled IndexedDB must not pile up full-size
        // canvases, JPEG blobs and feature buffers in memory (≈ 12 MB per 1920×1440 keyframe)
        const cap = this.writeBacklogBytes();
        if (this.pendingWriteBytes > cap) {
          const tw = now();
          while (this.pendingWriteBytes > cap && this.writes.size && !this.disposed) await Promise.race([...this.writes]);
          timings.writeWait = now() - tw;
        }
      }
      const u = this.baseUpdate(frameIndex, decision.decision, this.mapGuidance(decision.guidance), timings);
      u.overlay = overlay;
      if (framePose) {
        u.framePose = { pose: framePose, t: f.t + (st.timeOffsetMs ?? 0) };
        u.R_cam_device = Float64Array.from(st.cameraFromDevice());
      }
      if (this.inline && this.modelVersion !== this.sentVersion) this.attachModel(u);
      return finish(u);
    } finally {
      if (!handedOff) prep.release();
    }
  }

  /**
   * Pose of a tracked frame in the live model (the "camera view" of the capture screen): PnP on its
   * tracking inliers whose reference-keyframe feature has a triangulated point. Null while the
   * reference keyframe is not registered (mapping behind) or the fit is weak.
   */
  private framePose(refId: number, inliers: MatchPairs, fs: FeatureSet): Pose | null {
    const rec = this.mapper.reconstruction;
    const img = rec.images.get(refId);
    const cam = img && img.registered ? rec.cameras.get(img.cameraId) : undefined;
    if (!img || !cam) return null;
    const m = inliers.length >> 1;
    const X = new Float64Array(3 * m),
      uv = new Float64Array(2 * m);
    const kp = fs.keypoints;
    let n = 0;
    for (let i = 0; i < m; i++) {
      const pid = img.point3D[inliers[2 * i]];
      const p = pid >= 0 ? rec.points.get(pid) : undefined;
      if (!p) continue;
      const j = inliers[2 * i + 1];
      X[3 * n] = p.xyz[0];
      X[3 * n + 1] = p.xyz[1];
      X[3 * n + 2] = p.xyz[2];
      uv[2 * n] = kp[4 * j];
      uv[2 * n + 1] = kp[4 * j + 1];
      n++;
    }
    if (n < LiveSession.FRAME_POSE_MIN_POINTS) return null;
    const res = solvePnPRansac(X, uv, n, cam, { maxErrorPx: 6, minInliers: LiveSession.FRAME_POSE_MIN_POINTS, maxIterations: 300 });
    if (!res || res.numInliers < 0.5 * n) return null;
    return res.pose;
  }

  /** Minimum 2D–3D correspondences (and PnP inliers) for a frame pose. */
  private static readonly FRAME_POSE_MIN_POINTS = 15;

  /** Diagnostics of a frame before it is analysed (filled in as processing goes). */
  private newDebug(): LiveDebug {
    let backend: LiveDebug['backend'] = 'cpu';
    let fallbackReason: string | null = null;
    try {
      const p = this.deps.pipeline();
      backend = p.gpu ? 'webgpu' : 'cpu';
      fallbackReason = p.fallbackReason;
    } catch (e) {
      fallbackReason = `no pipeline: ${(e as Error)?.message ?? e}`;
    }
    return {
      backend,
      fallbackReason,
      featureInput: null,
      srcWidth: 0,
      srcHeight: 0,
      width: 0,
      height: 0,
      numFeatures: null,
      minFeatures: this.selector.options.minFeatures,
      sharpness: NaN,
      sharpnessRef: NaN,
      angularSpeed: NaN,
      lumaMean: NaN,
      lumaStd: NaN,
      inliers: NaN,
      reason: 'skipped (paused / hidden / finished)',
    };
  }

  private featureMaxSide(): number {
    return this.deps.config.sift?.maxImageSize ?? this.deps.preset.featureMaxSide;
  }

  /** Guided match + verification of a frame against the reference keyframe. */
  private async trackAgainst(
    ref: TrackingRef,
    fs: FeatureSet,
    t: number,
    timings: Record<string, number>,
  ): Promise<{ geom: TwoViewGeometry | null; numMatches: number; numInliers: number; parallax: number; rotation: number }> {
    // the pipeline is read when each GPU job runs (a GPU-loss recovery may replace it in between)
    const cam = this.intrinsics(ref.width, ref.height);
    const Rc = this.gyroRotation(ref.t, t);
    const minInl = this.selector.options.minInliers;
    const verify = (m: MatchPairs): TwoViewGeometry =>
      estimateTwoViewGeometry(ref.fs.keypoints, fs.keypoints, m, cam, cam, {
        maxIterations: this.deps.config.trackingMaxIterations ?? ENGINE_DEFAULTS.trackingMaxIterations,
        rotationPrior: Rc ?? undefined,
      });
    const tm = now();
    let matches: MatchPairs;
    let guided = false;
    if (Rc) {
      const Hm = mat3Mul(mat3Mul(kMatrix(cam), Rc), kInverse(cam));
      const radius = Math.max(24, (this.deps.config.guidedRadiusFrac ?? ENGINE_DEFAULTS.guidedRadiusFrac) * Math.max(ref.width, ref.height));
      matches = await this.deps.gpu(() => this.deps.pipeline().matchGuided(ref.fs, fs, { kind: 'H', H: Hm, maxErrorPx: radius }));
      guided = true;
    } else {
      matches = await this.deps.gpu(() => this.deps.pipeline().match(ref.fs, fs));
    }
    timings.match = now() - tm;
    const tv = now();
    let geom = matches.length >= 16 ? verify(matches) : null;
    let nInl = usable(geom, 8) ? geom!.inliers.length >> 1 : 0;
    if (guided && nInl < minInl) {
      // gyro prediction wrong (time offset, fast motion, wrong R_cam_device?) → unguided
      this.stats.guidedFallbacks++;
      const m2 = await this.deps.gpu(() => this.deps.pipeline().match(ref.fs, fs));
      const g2 = m2.length >= 16 ? verify(m2) : null;
      const n2 = usable(g2, 8) ? g2!.inliers.length >> 1 : 0;
      if (n2 > nInl) {
        geom = g2;
        nInl = n2;
        matches = m2;
      }
    }
    timings.verify = now() - tv;
    const Rba = Rc ?? (geom && geom.R ? geom.R : null);
    const inl = nInl > 0 && geom ? geom.inliers : matches;
    const parallax = Rba && inl.length > 0 ? rotationCompensatedParallax(ref.fs.keypoints, fs.keypoints, inl, cam, Rba) : NaN;
    const rotation = Rc ? rotationAngle(Rc) : geom && geom.R && nInl > 0 ? rotationAngle(geom.R) : NaN;
    return { geom: nInl > 0 ? geom : null, numMatches: matches.length >> 1, numInliers: nInl, parallax, rotation };
  }

  // ------------------------------------------------------------------------------------------
  // keyframes

  private createKeyframe(f: FrameInput, frameIndex: number, prep: PreparedFrame, fs: FeatureSet, k: number): number {
    const st = this.state;
    const id = st.nextKeyframeId++;
    const Rcd = st.cameraFromDevice();
    const gDev = gravityDown(st.imu.samples, f.t + (st.timeOffsetMs ?? 0), f.gravityDevice ?? null);
    const info: FrameInfo = {
      id,
      t: f.t,
      width: prep.width,
      height: prep.height,
      sharpness: prep.sharpness,
      R_cam_device: Rcd,
      gravityCam: gDev ? rotateVec(Rcd, gDev) : null,
      heading: f.heading && Number.isFinite(f.heading.heading) ? f.heading.heading : null,
      geo: f.geo ?? null,
    };
    const rec: KeyframeRecord = {
      id,
      frameIndex,
      t: f.t,
      width: prep.width,
      height: prep.height,
      info,
      gravityDevice: gDev,
      headingAccuracy: f.heading && Number.isFinite(f.heading.accuracy) ? f.heading.accuracy : null,
      headingNorth: f.heading?.north === 'magnetic' ? 'magnetic' : 'true',
      segment: typeof f.segment === 'number' ? f.segment : 0,
      screenAngle: f.screenAngle,
      quarterTurns: k,
    };
    const colors = sampleFeatureColors(fs, prep.small, prep.smallScaleX, prep.smallScaleY);
    st.db.addFrame(info, fs, colors);
    st.records.set(id, rec);
    st.order.push(id);
    this.stats.keyframes++;
    // persistence (async): JPEG of the full-resolution canonical image, features, colours, IMU
    const store = this.deps.store;
    const img = prep.fullImage(0.92);
    prep.release();
    // the full-size canvas is alive while the JPEG is encoded, then the blob until it is stored
    this.track(img.then((im) => store.putFrame(id, im, rec)), 4 * prep.width * prep.height);
    const fsBytes = serializeFeatureSet(fs);
    this.track(store.putBinary(`kf/${id}/features`, fsBytes), 2 * fsBytes.byteLength);
    // a rolling shutter known in advance is corrected live (the store keeps the original keypoints)
    this.applyLiveRollingShutter(id);
    this.track(store.putBinary(`kf/${id}/colors`, colors), colors.byteLength);
    const chunk = st.imu.takeChunk();
    if (chunk.length) this.track(store.putBinary(`imu/${st.imuChunks++}`, packImu(chunk)));
    if (st.geo.length) this.track(store.putValue('geo', st.geo));
    this.persistState();
    return id;
  }

  /** Canonical-image rolling-shutter model known in advance (`EngineConfig.rollingShutter`), or null. */
  liveRollingShutter(): RollingShutterModel | null {
    const dev = this.deps.config.rollingShutter;
    if (!dev || !(dev.readoutMs > 0)) return null;
    return rsModelFromDevice(dev, this.state.cameraFromDevice());
  }

  /** Correct keyframe `id` for the rolling shutter known in advance (no-op without one). */
  private applyLiveRollingShutter(id: number): void {
    const model = this.liveRollingShutter();
    const r = this.state.records.get(id);
    if (!model || !r) return;
    if (correctKeyframeRS(this.state, id, this.intrinsics(r.width, r.height), this.gyro(), model)) this.state.rollingShutter = model;
  }

  /** Track a background write holding `bytes` until it completes; failures are reported as storage errors. */
  private track(p: Promise<unknown>, bytes = 0): void {
    const q = p.catch((e: unknown) => this.deps.storageError(e));
    this.writes.add(q);
    this.pendingWriteBytes += bytes;
    void q.finally(() => {
      this.writes.delete(q);
      this.pendingWriteBytes -= bytes;
    });
  }

  private get memoryBudgetMB(): number {
    return this.deps.memoryBudgetMB?.() ?? this.deps.config.memoryBudgetMB ?? ENGINE_DEFAULTS.memoryBudgetMB;
  }

  /** Pending-write bytes above which a keyframe frame waits for the store (3 % of the budget, ≥ 24 MB). */
  private writeBacklogBytes(): number {
    const cfg = this.deps.config.maxPendingWriteMB;
    if (cfg !== undefined && cfg >= 0) return cfg * MB;
    return Math.max(24 * MB, 0.03 * this.memoryBudgetMB * MB);
  }

  /** Estimated resident memory of the capture (features, pairs, graph, live model, vocabulary, IMU, pending writes). */
  memoryBreakdown(): MemoryBreakdown {
    const st = this.state;
    return captureMemory({
      db: st.db,
      graph: st.graph,
      rec: this.mapper.initialized ? this.mapper.reconstruction : null,
      vocab: st.vocab,
      imuSamples: st.imu.samples.length,
      pendingWriteBytes: this.pendingWriteBytes,
      // a readout known in advance keeps every keyframe's original keypoints (+ ≈ 11 % of the features)
      rsOriginal: st.rsOriginal,
      // the best-of-window frame: full-size source (+ rotated canvas) and the small copy
      heldFrameBytes: this.held ? 2 * 4 * this.held.width * this.held.height + this.held.prep.small.data.byteLength : 0,
    });
  }

  /** Last capture memory estimate (bytes; updated per mapped keyframe). */
  get memoryEstimate(): number {
    return this.memEstimate;
  }

  /**
   * Capture-state bytes above which no keyframes are added: EngineConfig.captureMemoryFraction of
   * the budget left after the page / JS reserve and the feature pipeline's GPU caches.
   */
  private captureMemoryCap(): number {
    const f = this.deps.config.captureMemoryFraction ?? ENGINE_DEFAULTS.captureMemoryFraction;
    return f * Math.max(0, this.memoryBudgetMB * MB - MEM.reserve - 64 * MB);
  }

  /** True once the capture state reached its cap (reported once). */
  private captureMemoryFull(): boolean {
    if (!(this.memEstimate > this.captureMemoryCap())) return false;
    if (!this.memWarned) {
      this.memWarned = true;
      const msg = `capture memory ≈ ${Math.round(this.memEstimate / MB)} MB at ${this.state.order.length} keyframes reached ${Math.round(this.captureMemoryCap() / MB)} MB (of a ${this.memoryBudgetMB} MB budget): no more keyframes — finish the scan`;
      this.deps.log('warn', msg);
      this.deps.memoryLimit?.(msg);
    }
    return true;
  }

  /** Update the capture memory estimate (per mapped keyframe); log it every 25 keyframes. */
  private updateMemoryEstimate(): void {
    const b = this.memoryBreakdown();
    const total = sumBreakdown(b);
    this.memEstimate = total;
    const n = this.state.order.length;
    if (n - this.memLoggedAt >= 25) {
      this.memLoggedAt = n;
      this.deps.log('info', `capture memory ≈ ${formatBreakdown(b, total)} at ${n} keyframes`, b);
    }
  }

  /** Snapshot of the persisted engine state (extended by the engine). */
  stateRecord(extra: Partial<EngineStateRecord> = {}): EngineStateRecord {
    const st = this.state;
    return {
      version: ENGINE_STATE_VERSION,
      mode: this.deps.mode,
      quality: this.deps.quality,
      hfovPriorDeg: this.deps.hfovPriorDeg,
      thetaSession: st.thetaSession,
      cameraTheta: st.cameraTheta,
      cameraThetaVerified: st.cameraThetaVerified,
      rawDelivery: st.rawDelivery,
      resolvedMode: st.resolvedMode,
      nextKeyframeId: st.nextKeyframeId,
      frameCounter: st.frameCounter,
      imuChunks: st.imuChunks,
      captureFinished: st.captureFinished,
      vocabTrainedOn: st.vocabTrainedOn,
      hasSparse: false,
      hasDense: false,
      hasMesh: false,
      timeOffsetMs: st.timeOffsetMs,
      gyroBias: st.gyroBias,
      rollingShutter: st.rollingShutter,
      updatedAt: Date.now(),
      ...extra,
    };
  }

  /** Hook set by the engine to persist the full state record. */
  persistState: () => void = () => {
    this.track(this.deps.store.putValue('engine/state', this.stateRecord()));
  };

  private lastJobMs = 0;

  /** Promises of the queued / running mapping jobs (oldest first). */
  private readonly jobs: Promise<void>[] = [];

  private enqueue(job: MappingJob): Promise<void> {
    this.queued++;
    const run: Promise<void> = this.queue.then(async () => {
      const t0 = now();
      try {
        if (this.disposed) return;
        await this.mapKeyframe(job);
      } catch (e) {
        this.deps.log('error', `mapping keyframe ${job.id} failed: ${(e as Error)?.message ?? e}`, e);
      } finally {
        this.lastJobMs = now() - t0;
        this.stats.mappingMs += this.lastJobMs;
        this.queued--;
        const i = this.jobs.indexOf(run);
        if (i >= 0) this.jobs.splice(i, 1);
      }
    });
    this.jobs.push(run);
    this.queue = run;
    return run;
  }

  /** Number of keyframes waiting for mapping. */
  get pendingMapping(): number {
    return this.queued;
  }

  /**
   * End of capture: the frame held by best-of-window keyframing would have become a keyframe once its
   * deferral window ended (`KeyframeSelector.commitHeld`), but no later frame will end it — make it
   * the keyframe now (persisted and mapped like any other) instead of dropping it. Call before
   * `flush` (which releases a held frame once the capture is finished). Returns the new keyframe id,
   * or null (nothing held, disposed, capture memory cap reached).
   */
  commitHeldKeyframe(): number | null {
    const src = this.held;
    if (!src) return null;
    this.held = null;
    if (this.disposed || this.selector.heldTime !== src.f.t || this.captureMemoryFull()) {
      src.prep.release();
      return null;
    }
    this.selector.commitHeld();
    this.stats.heldKeyframes++;
    const id = this.createKeyframe(src.f, src.frameIndex, src.prep, src.fs, src.k);
    this.ref = { id, fs: src.fs, t: src.f.t, width: src.width, height: src.height };
    void this.enqueue({ id, tracking: src.refId !== null && src.trackGeom ? { other: src.refId, geom: src.trackGeom } : null });
    this.deps.log('info', `the held best-of-window frame ${src.frameIndex} became keyframe ${id} at the end of the capture`);
    return id;
  }

  /** Resolves when the mapping queue is empty and all writes are done. */
  async flush(): Promise<void> {
    if (this.state.captureFinished) this.releaseHeld();
    for (;;) {
      const q = this.queue;
      await q;
      if (q === this.queue) break;
    }
    while (this.writes.size) await Promise.all([...this.writes]);
  }

  /** Matching + verification of a keyframe pair (a < b in capture order). */
  async matchPair(a: number, b: number, opts: { maxIterations?: number } = {}): Promise<TwoViewGeometry | null> {
    const m = await this.putativeMatches(a, b);
    return m ? this.verifyPair(a, b, m, opts) : null;
  }

  /**
   * GPU half of `matchPair`: putative matches of a keyframe pair, null when a set is missing or
   * too small. Split from `verifyPair` so a caller can match the next pair while it verifies this
   * one (refine): still one GPU job at a time, queued on the engine's GPU chain. A job that
   * reaches the GPU after `cancel` is set does no work and returns null.
   */
  async putativeMatches(a: number, b: number, cancel?: CancelToken): Promise<MatchPairs | null> {
    const st = this.state;
    const fa = st.db.features.get(a);
    const fb = st.db.features.get(b);
    if (!fa || !fb || !st.records.has(a) || !st.records.has(b) || fa.count < 8 || fb.count < 8) return null;
    // the pipeline is read when the job runs: a GPU-loss recovery queued ahead of a prefetched
    // job has replaced it by then
    return this.deps.gpu(async () => (cancel?.cancelled ? null : this.deps.pipeline().match(fa, fb)));
  }

  /** CPU half of `matchPair`: two-view verification of putative matches and the gyro gate. */
  verifyPair(a: number, b: number, m: MatchPairs, opts: { maxIterations?: number } = {}): TwoViewGeometry | null {
    const v = this.verifyInputs(a, b, m, opts);
    return v ? this.gatePair(a, b, estimateTwoViewGeometry(v.kp1, v.kp2, m, v.cam1, v.cam2, v.opts)) : null;
  }

  /**
   * Inputs of `verifyPair`'s two-view estimation (null: the pair cannot verify), for running it
   * elsewhere (refine's CPU pool) and gating the result with `gatePair`.
   */
  verifyInputs(
    a: number,
    b: number,
    m: MatchPairs,
    opts: { maxIterations?: number } = {},
  ): { kp1: Float32Array; kp2: Float32Array; cam1: CameraIntrinsics; cam2: CameraIntrinsics; opts: TwoViewOptions } | null {
    const st = this.state;
    const fa = st.db.features.get(a);
    const fb = st.db.features.get(b);
    const ra = st.records.get(a);
    const rb = st.records.get(b);
    if (!fa || !fb || !ra || !rb) return null;
    if (m.length >> 1 < this.minPairInliers) return null;
    const Rp = this.gyroRotation(ra.t, rb.t);
    return {
      kp1: fa.keypoints,
      kp2: fb.keypoints,
      cam1: this.intrinsics(ra.width, ra.height),
      cam2: this.intrinsics(rb.width, rb.height),
      opts: { maxIterations: opts.maxIterations ?? 2000, rotationPrior: Rp ?? undefined },
    };
  }

  /** Record a verified pair in the DB and graph. */
  addPair(a: number, b: number, g: TwoViewGeometry): void {
    this.state.db.setTwoView(a, b, g);
    if (this.state.graph.hasImage(a) && this.state.graph.hasImage(b)) this.state.graph.addTwoView(a, b, g.inliers);
  }

  private async mapKeyframe(job: MappingJob): Promise<void> {
    const st = this.state;
    const id = job.id;
    const fs = st.db.features.get(id);
    if (!fs) return;
    if (!st.graph.hasImage(id)) st.graph.addImage(id, fs.count);
    const pos = st.order.indexOf(id);
    const preset = this.deps.preset;
    const lp = this.deps.config.livePairs ?? {};
    // previous 6 keyframes (4 on the CPU tier: unguided CPU matching costs ~0.1–0.2 s per pair)
    const nSeq = lp.sequential ?? (this.deps.pipeline().gpu ? preset.liveSequentialPairs : Math.min(4, preset.liveSequentialPairs));
    const nRet = lp.retrieval ?? preset.liveRetrievalPairs;
    const nSpa = lp.spatial ?? preset.liveSpatialPairs;
    const cands: { other: number; speculative: boolean }[] = [];
    const chosen = new Set<number>([id]);
    for (let i = pos - 1; i >= 0 && cands.length < nSeq; i--) {
      const o = st.order[i];
      if (!st.graph.hasImage(o)) continue;
      cands.push({ other: o, speculative: false });
      chosen.add(o);
    }
    // vocabulary retrieval (the BoW is kept for indexing this keyframe below: quantised once)
    let bow: { vocab: VocabularyIndex; bow: BowVector } | null = null;
    if (st.vocab && st.vocab.size > 0 && nRet > 0) {
      bow = { vocab: st.vocab, bow: st.vocab.computeBow(fs) };
      const res = st.vocab.query(bow.bow, nRet + 4, (x: number) => chosen.has(x));
      let k = 0;
      for (const r of res) {
        if (k >= nRet || !(r.score > 0)) break;
        if (!st.graph.hasImage(r.id) || chosen.has(r.id)) continue;
        cands.push({ other: r.id, speculative: true });
        chosen.add(r.id);
        k++;
      }
    }
    // spatial neighbours of the most recent registered keyframe
    if (this.mapper.initialized && nSpa > 0) {
      for (const o of this.spatialCandidates(pos, chosen, nSpa)) {
        cands.push({ other: o, speculative: true });
        chosen.add(o);
      }
    }
    const stored: StoredPair[] = [];
    const attempted: number[] = [];
    for (const c of cands) {
      const key = pairKey(c.other, id);
      if (st.attempted.has(key)) continue;
      st.attempted.add(key);
      attempted.push(key);
      let g: TwoViewGeometry | null = null;
      if (job.tracking && job.tracking.other === c.other && usable(job.tracking.geom, this.minPairInliers)) g = this.gatePair(c.other, id, job.tracking.geom);
      else {
        try {
          g = await this.matchPair(c.other, id, { maxIterations: c.speculative ? 1000 : 2000 });
        } catch (e) {
          this.deps.log('warn', `pair ${c.other}-${id} failed: ${(e as Error)?.message ?? e}`);
        }
        if (!this.inline) await tick();
      }
      if (usable(g, this.minPairInliers)) {
        this.addPair(c.other, id, g);
        stored.push(pairToStored(c.other, id, g));
      }
    }
    // disposed while matching (e.g. a discarded scan, deleted next): nothing more is stored
    if (this.disposed) return;
    const res = this.mapper.addKeyframe(id);
    this.logRegistrations(res);
    this.consecutiveUnregistered = res.registered ? 0 : this.consecutiveUnregistered + 1;
    this.modelVersion++;
    // vocabulary: train once enough keyframes exist, then index incrementally
    this.updateVocabulary(id, bow);
    this.maybeVerifyCameraFromDevice();
    this.maybeCalibrateGyro();
    this.maybeResolveMode();
    const rec: StoredPairs = { pairs: stored, attempted };
    this.track(this.deps.store.putValue(`kf/${id}/pairs`, rec));
    this.persistState();
    this.updateMemoryEstimate();
    if (!this.inline) this.emitModelUpdate();
  }

  private spatialCandidates(pos: number, chosen: Set<number>, n: number): number[] {
    const st = this.state;
    const rec = this.mapper.reconstruction;
    let anchor: number | null = null;
    for (let i = pos - 1; i >= 0; i--) {
      const o = st.order[i];
      if (rec.images.get(o)?.registered) {
        anchor = o;
        break;
      }
    }
    if (anchor === null) return [];
    const depth = this.medianSceneDepth();
    const look = (p: Pose): Float64Array => {
      const C = poseCentre(p);
      return Float64Array.of(C[0] + depth * p.R[6], C[1] + depth * p.R[7], C[2] + depth * p.R[8]);
    };
    // The new keyframe looks where the gyro says, not where the anchor looked: after a fast pan
    // (or while tracking is lost) the last registered keyframe can be tens of degrees away, and
    // its neighbours are then wrong candidates (false matches on repeated structure). Centre:
    // the anchor's (translation is not observed by the gyro).
    const anchorPose = rec.images.get(anchor)!.pose;
    const ra = st.records.get(anchor),
      rn = st.records.get(st.order[pos]);
    const G = ra && rn ? this.gyroRotation(ra.t, rn.t) : null;
    const ap: Pose = G ? { R: mat3Mul(G, anchorPose.R), t: mat3MulVecOf(mat3Mul(G, anchorPose.R), negate(poseCentre(anchorPose))) } : anchorPose;
    const aL = look(ap);
    const out: { id: number; d: number }[] = [];
    for (const rid of rec.registeredImageIds()) {
      if (chosen.has(rid) || rid === anchor) continue;
      const p = rec.images.get(rid)!.pose;
      const cos = p.R[6] * ap.R[6] + p.R[7] * ap.R[7] + p.R[8] * ap.R[8];
      if (cos < Math.cos(45 * DEG)) continue;
      const L = look(p);
      out.push({ id: rid, d: Math.hypot(L[0] - aL[0], L[1] - aL[1], L[2] - aL[2]) });
    }
    out.sort((a, b) => a.d - b.d);
    return out.slice(0, n).map((x) => x.id);
  }

  private depthCache: { version: number; depth: number } = { version: -1, depth: 1 };

  /** Median depth of the live model's points in their observing cameras (sampled). */
  medianSceneDepth(): number {
    if (this.depthCache.version === this.modelVersion) return this.depthCache.depth;
    const rec = this.mapper.reconstruction;
    const ds: number[] = [];
    let k = 0;
    const stride = Math.max(1, Math.floor(rec.numPoints / 2000));
    for (const p of rec.points.values()) {
      if (k++ % stride) continue;
      const el = p.track[0];
      const img = rec.images.get(el.image);
      if (!img || !img.registered) continue;
      const R = img.pose.R,
        t = img.pose.t;
      const z = R[6] * p.xyz[0] + R[7] * p.xyz[1] + R[8] * p.xyz[2] + t[2];
      if (z > 0) ds.push(z);
    }
    const d = ds.length ? median(ds) : 1;
    this.depthCache = { version: this.modelVersion, depth: d };
    return d;
  }

  /** `bow`: the keyframe's BoW from retrieval, reused when computed with the current vocabulary. */
  private updateVocabulary(id: number, bow: { vocab: VocabularyIndex; bow: BowVector } | null = null): void {
    const st = this.state;
    const vc = this.deps.config.vocabulary ?? {};
    const trainAfter = vc.trainAfter ?? ENGINE_DEFAULTS.vocabularyTrainAfter;
    try {
      if (!st.vocab) {
        const ids = st.order.filter((x) => st.db.features.has(x) && st.graph.hasImage(x));
        if (ids.length < trainAfter) return;
        const t0 = now();
        const sources = ids.map((x) => st.db.features.get(x)!);
        st.vocab = VocabularyIndex.train(sources, { numWords: vc.words ?? ENGINE_DEFAULTS.vocabularyWords });
        for (const x of ids) st.vocab.add(x, st.db.features.get(x)!);
        st.vocabTrainedOn = ids;
        this.track(this.deps.store.putBinary('vocab', st.vocab.exportCodebook()));
        this.deps.log('info', `vocabulary trained on ${ids.length} keyframes (${st.vocab.numWords} words, ${Math.round(now() - t0)} ms)`);
      } else if (!st.vocab.has(id)) {
        if (bow && bow.vocab === st.vocab) st.vocab.addBow(id, bow.bow);
        else st.vocab.add(id, st.db.features.get(id)!);
      }
    } catch (e) {
      this.deps.log('warn', `vocabulary update failed: ${(e as Error)?.message ?? e}`);
    }
  }

  /** Registered keyframes with their poses (capture order). */
  registeredKeyframes(): { id: number; t: number; pose: Pose }[] {
    const rec = this.mapper.reconstruction;
    const out: { id: number; t: number; pose: Pose }[] = [];
    for (const id of this.state.order) {
      const im = rec.images.get(id);
      const r = this.state.records.get(id);
      if (im && im.registered && r) out.push({ id, t: r.t, pose: im.pose });
    }
    return out;
  }

  /**
   * R_cam_device hand-eye check (≥ 6 registered keyframes): if another quarter-turn hypothesis
   * explains the gyro clearly better, switch to it (logged); the inertial alignment then uses
   * the verified mapping.
   */
  maybeVerifyCameraFromDevice(force = false): HandEyeResult | null {
    const st = this.state;
    if (this.deps.config.verifyCameraFromDevice === false) return null;
    if (st.cameraThetaVerified && !force) return null;
    const kfs = this.registeredKeyframes();
    if (kfs.length < ENGINE_DEFAULTS.handEyeMinKeyframes || st.imu.samples.length < 10) return null;
    const r = handEyeCheck(kfs, st.imu, { offsetMs: st.timeOffsetMs ?? 0 });
    if (!r) return null;
    this.stats.handEye = r;
    if (r.pairs < 4) return r;
    const cur = (st.cameraTheta ?? 0) / 90;
    const best = r.best / 90;
    const sorted = [...r.medianErrDeg].sort((a, b) => a - b);
    const clearly = r.medianErrDeg[best] < 4 && r.medianErrDeg[best] < 0.5 * sorted[1];
    if (!clearly) return r;
    if (best !== cur) {
      this.deps.log('warn', `R_cam_device check: θ=${r.best} explains the gyro better than θ=${st.cameraTheta} (median ${r.medianErrDeg[best].toFixed(2)}° vs ${r.medianErrDeg[cur].toFixed(2)}°); switching`, r);
      this.setCameraTheta(r.best);
    } else if (r.pairs >= 6) {
      this.deps.log('info', `R_cam_device verified (θ=${r.best}, median ${r.medianErrDeg[best].toFixed(2)}° over ${r.pairs} keyframe pairs)`);
    }
    st.cameraThetaVerified = r.pairs >= 6;
    return r;
  }

  /**
   * The orientation tracker re-interpreted θ₀ (a session started in a landscape-270 UI turned out
   * to deliver raw sensor buffers: the unrotated first frames are upright for θ = 90). The
   * canonical images did not change, only what they are upright for: unless the hand-eye check
   * already verified the mapping from the gyro, R_cam_device of every keyframe follows θ₀.
   */
  private reinterpretSessionTheta(theta: ScreenAngle): void {
    const st = this.state;
    const old = st.thetaSession;
    st.thetaSession = theta;
    this.deps.log('warn', `raw sensor buffers detected: the session's first frames (θ=${old}) were raw buffers, canonical θ is ${theta}`);
    if (!st.cameraThetaVerified && st.cameraTheta !== theta) this.setCameraTheta(theta);
    this.persistState();
  }

  /** Change the camera←device mapping of the canonical images (updates every keyframe's FrameInfo). */
  setCameraTheta(theta: ScreenAngle): void {
    const st = this.state;
    st.cameraTheta = theta;
    const Rcd = st.cameraFromDevice();
    for (const r of st.records.values()) {
      r.info.R_cam_device = Float64Array.from(Rcd);
      r.info.gravityCam = r.gravityDevice ? rotateVec(Rcd, r.gravityDevice) : null;
      const fi = st.db.frames.get(r.id);
      if (fi && fi !== r.info) {
        fi.R_cam_device = r.info.R_cam_device;
        fi.gravityCam = r.info.gravityCam;
      }
    }
  }

  private maybeResolveMode(): void {
    const st = this.state;
    if (this.deps.mode !== 'auto' || st.resolvedMode) return;
    const kfs = this.registeredKeyframes();
    if (kfs.length < 8) return;
    const m = classifyScene(
      kfs.map((k) => k.pose),
      this.medianSceneDepth(),
    );
    if (!m) return;
    st.resolvedMode = m;
    this.selector.setMode(m);
    this.deps.log('info', `capture mode resolved: ${m}`);
  }

  /** Mode in effect ('auto' until resolved). */
  effectiveMode(): 'object' | 'scene' | 'auto' {
    return this.deps.mode === 'auto' ? (this.state.resolvedMode ?? 'auto') : this.deps.mode;
  }

  // ------------------------------------------------------------------------------------------
  // live model output

  private attachModel(u: LiveUpdate): void {
    const snap = this.mapper.snapshot();
    const n = snap.points.length / 3;
    const max = this.deps.config.liveMaxPoints ?? ENGINE_DEFAULTS.liveMaxPoints;
    if (n > max) {
      const stride = Math.ceil(n / max);
      const m = Math.ceil(n / stride);
      const pos = new Float32Array(3 * m);
      const col = new Uint8Array(3 * m);
      for (let i = 0, j = 0; i < n && j < m; i += stride, j++) {
        pos.set(snap.points.subarray(3 * i, 3 * i + 3), 3 * j);
        col.set(snap.colors.subarray(3 * i, 3 * i + 3), 3 * j);
      }
      u.sparse = { positions: pos, colors: col };
    } else u.sparse = { positions: snap.points, colors: snap.colors };
    const rec = this.mapper.reconstruction;
    const st = this.state;
    const offset = st.timeOffsetMs ?? 0;
    const cams: LiveCamera[] = [];
    for (const { id, pose } of snap.poses.sort((a, b) => a.id - b.id)) {
      const cam = rec.cameras.get(rec.images.get(id)!.cameraId)!;
      const r = st.records.get(id);
      const upCam = r ? keyframeUpCam(r) : undefined;
      cams.push({ pose, intrinsics: { ...cam }, ...(r ? { t: r.t + offset } : {}), ...(upCam ? { upCam } : {}) });
    }
    u.cameras = cams;
    u.R_cam_device = Float64Array.from(st.cameraFromDevice());
    u.gyroBias = st.gyroBias ? [st.gyroBias[0], st.gyroBias[1], st.gyroBias[2]] : null;
    this.sentVersion = this.modelVersion;
  }

  private emitModelUpdate(): void {
    if (this.modelVersion === this.sentVersion) return;
    const u = this.baseUpdate(this.lastFrameIndex, this.lastDecision, this.lastGuidance, {});
    u.tracking = this.trackingState();
    u.guidance = this.mapGuidance(this.lastGuidance === 'initializing' ? 'good' : this.lastGuidance);
    this.attachModel(u);
    this.deps.emitLive(u);
  }

  // ------------------------------------------------------------------------------------------
  // resume

  /**
   * Restore a capture from the store (worker respawn): keyframes, features, colours, pairs,
   * IMU, GPS, vocabulary and counters; the live model is rebuilt by re-adding the keyframes
   * (in the mapping queue). Returns the number of keyframes restored.
   */
  async restore(rec: EngineStateRecord): Promise<number> {
    const st = this.state;
    const store = this.deps.store;
    st.thetaSession = rec.thetaSession as ScreenAngle | null;
    st.cameraTheta = (rec.cameraTheta ?? rec.thetaSession) as ScreenAngle | null;
    st.cameraThetaVerified = rec.cameraThetaVerified;
    st.resolvedMode = rec.resolvedMode;
    st.nextKeyframeId = rec.nextKeyframeId;
    st.frameCounter = rec.frameCounter;
    st.captureFinished = rec.captureFinished;
    st.timeOffsetMs = rec.timeOffsetMs ?? null;
    st.gyroBias = rec.gyroBias && rec.gyroBias.length === 3 && rec.gyroBias.every(Number.isFinite) ? [rec.gyroBias[0], rec.gyroBias[1], rec.gyroBias[2]] : null;
    this.orientation.thetaSession = st.thetaSession;
    st.rawDelivery = rec.rawDelivery ?? null;
    this.orientation.rawDelivery = st.rawDelivery;
    if (st.resolvedMode) this.selector.setMode(st.resolvedMode);
    // IMU + GPS
    const imuKeys = await store.listBinaryKeys('imu/');
    const samples: ImuSample[] = [];
    for (const k of imuKeys) {
      const b = await store.getBinary(k);
      if (b) samples.push(...unpackImu(b));
    }
    st.imu.load(samples);
    st.imuChunks = Math.max(rec.imuChunks, imuKeys.length);
    st.geo = (await store.getValue<GeoFix[]>('geo')) ?? [];
    // keyframes
    const records = await store.getFrameRecords();
    for (const r of records) {
      const fb = await store.getBinary(`kf/${r.id}/features`);
      if (!fb) {
        this.deps.log('warn', `keyframe ${r.id}: features missing, skipped`);
        continue;
      }
      const fs = deserializeFeatureSet(fb);
      const cb = await store.getBinary(`kf/${r.id}/colors`);
      const colors = cb ? new Uint8Array(cb) : undefined;
      st.db.addFrame(r.info, fs, colors && colors.length >= 3 * fs.count ? colors : undefined);
      st.records.set(r.id, r);
      st.order.push(r.id);
      st.nextKeyframeId = Math.max(st.nextKeyframeId, r.id + 1);
    }
    // a keyframe record is written once, when the keyframe is created: R_cam_device / gravityCam
    // of keyframes stored before a hand-eye switch (or a θ₀ re-interpretation) are stale
    if (st.cameraTheta !== null) this.setCameraTheta(st.cameraTheta);
    // the store holds the ORIGINAL keypoints: a refine-estimated rolling shutter is estimated again
    // by the next refine; one known in advance is applied again
    st.rollingShutter = null;
    for (const id of st.order) this.applyLiveRollingShutter(id);
    // pairs
    const pairsById = new Map<number, StoredPair[]>();
    for (const id of st.order) {
      const p = await store.getValue<StoredPairs>(`kf/${id}/pairs`);
      if (!p) continue;
      for (const k of p.attempted) st.attempted.add(k);
      pairsById.set(id, p.pairs);
      for (const sp of p.pairs) if (st.db.features.has(sp.a) && st.db.features.has(sp.b)) st.db.setTwoView(sp.a, sp.b, storedToGeometry(sp));
    }
    // vocabulary
    const cb = await store.getBinary('vocab');
    if (cb && rec.vocabTrainedOn) {
      st.vocab = new VocabularyIndex(new Float32Array(cb));
      st.vocabTrainedOn = rec.vocabTrainedOn;
      for (const id of st.order) st.vocab.add(id, st.db.features.get(id)!);
    }
    // tracking reference: the last keyframe
    const last = st.order.length ? st.records.get(st.order[st.order.length - 1])! : null;
    if (last) {
      this.ref = { id: last.id, fs: st.db.features.get(last.id)!, t: last.t, width: last.width, height: last.height };
      this.selector.commitKeyframe(last.t);
    }
    // rebuild the live model in the mapping queue; keyframes whose mapping never completed (no
    // pairs record: still queued when the worker died) are matched and mapped like new ones
    const ids = st.order.slice();
    this.queued++;
    this.queue = this.queue.then(async () => {
      const t0 = now();
      let remapped = 0;
      try {
        for (const id of ids) {
          if (this.disposed) return;
          if (!pairsById.has(id)) {
            try {
              await this.mapKeyframe({ id, tracking: null });
              remapped++;
            } catch (e) {
              this.deps.log('error', `mapping keyframe ${id} failed: ${(e as Error)?.message ?? e}`, e);
            }
            continue;
          }
          const fs = st.db.features.get(id)!;
          if (!st.graph.hasImage(id)) st.graph.addImage(id, fs.count);
          for (const sp of pairsById.get(id) ?? []) {
            const other = sp.a === id ? sp.b : sp.a;
            if (st.graph.hasImage(other)) st.graph.addTwoView(sp.a, sp.b, sp.inliers);
          }
          const r = this.mapper.addKeyframe(id);
          this.logRegistrations(r);
          this.consecutiveUnregistered = r.registered ? 0 : this.consecutiveUnregistered + 1;
          this.modelVersion++;
          if (!this.inline) await tick();
        }
        this.deps.log('info', `live model rebuilt from ${ids.length} keyframes (${remapped} matched again) in ${Math.round(now() - t0)} ms`);
      } finally {
        this.queued--;
      }
      if (!this.inline) this.emitModelUpdate();
    });
    if (this.inline) await this.queue;
    return st.order.length;
  }

  /** Drop the frame held by best-of-window keyframing. */
  private releaseHeld(): void {
    if (!this.held) return;
    this.held.prep.release();
    this.held = null;
  }

  dispose(): void {
    this.disposed = true;
    this.releaseHeld();
    this.ref = null;
  }
}


