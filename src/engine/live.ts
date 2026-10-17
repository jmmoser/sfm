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
 */
import type { CameraIntrinsics, FeatureSet, FrameInfo, GeoFix, ImuSample, Mat3, MatchPairs, Pose, TwoViewGeometry } from '../core/types';
import type { FeaturePipeline } from '../features/extractor';
import { deserializeFeatureSet, serializeFeatureSet } from '../io/serialize';
import { estimateTwoViewGeometry } from '../geometry/twoview';
import { mat3Mul, mat3MulTranspose } from '../math/mat3';
import { rotationAngle } from '../math/rotation';
import { median } from '../math/stats';
import { cameraFromDevice, type ScreenAngle } from '../sensors/frame-orientation';
import { angularSpeedAt, integrateGyro } from '../sensors/motion';
import { CorrespondenceGraph } from '../sfm/correspondence-graph';
import { FeatureDatabase, pairKey } from '../sfm/database';
import { KeyframeSelector, rotationCompensatedParallax, type KeyframeCandidate } from '../sfm/keyframes';
import { LiveMapper } from '../sfm/live';
import { intrinsicsFromFov } from '../sfm/mapper';
import { VocabularyIndex } from '../sfm/vocab';
import type { CaptureMode, FrameInput, Guidance, LiveCamera, LiveUpdate, QualityPreset, TrackingState } from '../workers/protocol';
import {
  OrientationTracker,
  closeSource,
  gravityDown,
  overlayPoints,
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

  /** True when the samples cover [t0, t1] (within `slackMs`). */
  covers(t0: number, t1: number, slackMs = 60): boolean {
    const n = this.samples.length;
    if (n < 2) return false;
    return this.samples[0].t <= Math.min(t0, t1) + slackMs && this.samples[n - 1].t >= Math.max(t0, t1) - slackMs;
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

/** Decide 'object' vs 'scene' from the registered cameras (optical-axis convergence). */
export function classifyScene(poses: readonly Pose[], medianDepth: number): 'object' | 'scene' | null {
  const n = poses.length;
  if (n < 6) return null;
  // least-squares point closest to all optical axes
  const A = new Float64Array(9);
  const bv = new Float64Array(3);
  const Cs: Float64Array[] = [];
  const Ds: Float64Array[] = [];
  for (const p of poses) {
    const C = poseCentre(p);
    const d = Float64Array.of(p.R[6], p.R[7], p.R[8]);
    Cs.push(C);
    Ds.push(d);
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++) {
        const m = (r === c ? 1 : 0) - d[r] * d[c];
        A[3 * r + c] += m;
        bv[r] += m * C[c];
      }
  }
  // solve A x = b (3×3)
  const det =
    A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
  // angular spread of the viewing directions
  const mean = new Float64Array(3);
  for (const d of Ds) for (let k = 0; k < 3; k++) mean[k] += d[k] / n;
  const spread = Math.acos(Math.min(1, Math.hypot(mean[0], mean[1], mean[2])));
  if (!(Math.abs(det) > 1e-9) || spread < 10 * DEG) return 'scene';
  const inv = (i: number, j: number): number => {
    const m = (r: number, c: number): number => A[3 * r + c];
    const cof = [
      m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1),
      -(m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)),
      m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0),
      -(m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1)),
      m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0),
      -(m(0, 0) * m(2, 1) - m(0, 1) * m(2, 0)),
      m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1),
      -(m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0)),
      m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0),
    ];
    return cof[3 * j + i] / det;
  };
  const P = new Float64Array(3);
  for (let i = 0; i < 3; i++) P[i] = inv(i, 0) * bv[0] + inv(i, 1) * bv[1] + inv(i, 2) * bv[2];
  let inFront = 0;
  const dist: number[] = [];
  const off: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = [P[0] - Cs[i][0], P[1] - Cs[i][1], P[2] - Cs[i][2]];
    const along = v[0] * Ds[i][0] + v[1] * Ds[i][1] + v[2] * Ds[i][2];
    const len = Math.hypot(v[0], v[1], v[2]);
    if (along > 0) inFront++;
    dist.push(len);
    off.push(Math.sqrt(Math.max(0, len * len - along * along)));
  }
  const dMed = median(dist);
  const offMed = median(off);
  const converging = inFront >= 0.9 * n && offMed < 0.35 * dMed;
  const depthOk = !(medianDepth > 0) || (dMed < 2.5 * medianDepth && dMed > 0.3 * medianDepth);
  return converging && depthOk ? 'object' : 'scene';
}

/** Live capture session. */
export class LiveSession {
  readonly state = new CaptureState();
  readonly selector: KeyframeSelector;
  readonly orientation: OrientationTracker;
  readonly mapper: LiveMapper;
  private ref: TrackingRef | null = null;
  private segment: number | null = null;
  private evaluated = 0;
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
  readonly stats = { frames: 0, keyframes: 0, mappingMs: 0, featureMs: 0, trackingMs: 0, guidedFallbacks: 0, handEye: null as HandEyeResult | null };

  constructor(private readonly deps: LiveDeps) {
    const mc = modeConfig(deps.mode);
    this.selector = new KeyframeSelector({ mode: mc.keyframeMode, ...mc.keyframe, ...(deps.config.keyframe ?? {}) });
    this.orientation = new OrientationTracker(deps.config.settleMs ?? ENGINE_DEFAULTS.settleMs);
    this.mapper = new LiveMapper(this.state.db, this.state.graph, {
      hfovPriorDeg: deps.hfovPriorDeg ?? 69,
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
    const off = this.state.timeOffsetMs ?? 0;
    if (!this.state.imu.covers(t0 + off, t1 + off)) return null;
    const G = integrateGyro(this.state.imu.samples, t0 + off, t1 + off);
    const Rcd = this.state.cameraFromDevice();
    return mat3MulTranspose(mat3Mul(Rcd, G), Rcd);
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
    const finish = (u: LiveUpdate): LiveUpdate => {
      timings.total = now() - t0;
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
      this.ref = null;
    }
    this.segment = seg;
    const verdict = this.orientation.classify({ width: f.bitmap.width, height: f.bitmap.height, screenAngle: f.screenAngle, t: f.t });
    if (!verdict.ok) {
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
      const angSpeed = angularSpeedAt(st.imu.samples, f.t + (st.timeOffsetMs ?? 0));
      const so = this.selector.options;
      const warm = this.evaluated >= Math.min(Math.max(0, so.sharpnessWarmup), so.sharpnessWindow);
      const sref = this.selector.sharpnessReference;
      this.evaluated++;
      const cand: KeyframeCandidate = {
        t: f.t,
        sharpness: prep.sharpness,
        angularSpeed: angSpeed,
        numFeatures: 0,
        numMatchesToLastKF: 0,
        numInliersToLastKF: NaN,
        medianParallaxPx: NaN,
        rotationSinceLastKFRad: NaN,
        imageWidth: W,
      };
      // pre-gate: blurry / fast-rotating frames are rejected without feature extraction
      const blurry = warm && Number.isFinite(sref) && prep.sharpness < so.blurRatio * sref;
      const fast = Number.isFinite(angSpeed) && angSpeed > so.maxAngularSpeed;
      if (blurry || fast || !(prep.sharpness >= so.minSharpness)) {
        const d = this.selector.evaluate(cand);
        return finish(this.baseUpdate(frameIndex, d.decision, this.mapGuidance(d.guidance), timings));
      }
      // features
      const pipeline = this.deps.pipeline();
      const tf = now();
      const fi = prep.featureInput(pipeline.gpu, this.featureMaxSide());
      const fs = await this.deps.gpu(() => (fi.kind === 'bitmap' ? pipeline.extractFromBitmap(fi.source, W, H) : pipeline.extract(fi.gray, W, H)));
      timings.features = now() - tf;
      this.stats.featureMs += timings.features;
      cand.numFeatures = fs.count;
      // tracking against the last keyframe
      let trackGeom: TwoViewGeometry | null = null;
      if (this.ref && (this.ref.width !== W || this.ref.height !== H)) {
        // frame size changed (camera format switch): start a new tracking segment
        this.deps.log('warn', `frame size changed ${this.ref.width}x${this.ref.height} → ${W}x${H}: new tracking segment`);
        this.selector.resetTracking();
        this.ref = null;
      }
      const ref = this.ref;
      if (ref && fs.count > 0 && ref.fs.count > 0) {
        const tm = now();
        const tr = await this.trackAgainst(ref, fs, f.t, timings);
        trackGeom = tr.geom;
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
      let mapped: Promise<void> | null = null;
      if (decision.decision === 'keyframe') {
        const tk = now();
        const id = this.createKeyframe(f, frameIndex, prep, fs, verdict.k);
        handedOff = true;
        const job: MappingJob = { id, tracking: ref && trackGeom ? { other: ref.id, geom: trackGeom } : null };
        this.ref = { id, fs, t: f.t, width: W, height: H };
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
      if (this.inline && this.modelVersion !== this.sentVersion) this.attachModel(u);
      return finish(u);
    } finally {
      if (!handedOff) prep.release();
    }
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
    const pipeline = this.deps.pipeline();
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
      matches = await this.deps.gpu(() => pipeline.matchGuided(ref.fs, fs, { kind: 'H', H: Hm, maxErrorPx: radius }));
      guided = true;
    } else {
      matches = await this.deps.gpu(() => pipeline.match(ref.fs, fs));
    }
    timings.match = now() - tm;
    const tv = now();
    let geom = matches.length >= 16 ? verify(matches) : null;
    let nInl = usable(geom, 8) ? geom!.inliers.length >> 1 : 0;
    if (guided && nInl < minInl) {
      // gyro prediction wrong (time offset, fast motion, wrong R_cam_device?) → unguided
      this.stats.guidedFallbacks++;
      const m2 = await this.deps.gpu(() => pipeline.match(ref.fs, fs));
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
    this.track(store.putBinary(`kf/${id}/colors`, colors), colors.byteLength);
    const chunk = st.imu.takeChunk();
    if (chunk.length) this.track(store.putBinary(`imu/${st.imuChunks++}`, packImu(chunk)));
    if (st.geo.length) this.track(store.putValue('geo', st.geo));
    this.persistState();
    return id;
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

  /** Resolves when the mapping queue is empty and all writes are done. */
  async flush(): Promise<void> {
    for (;;) {
      const q = this.queue;
      await q;
      if (q === this.queue) break;
    }
    while (this.writes.size) await Promise.all([...this.writes]);
  }

  /** Matching + verification of a keyframe pair (a < b in capture order). */
  async matchPair(a: number, b: number, opts: { maxIterations?: number } = {}): Promise<TwoViewGeometry | null> {
    const st = this.state;
    const fa = st.db.features.get(a);
    const fb = st.db.features.get(b);
    const ra = st.records.get(a);
    const rb = st.records.get(b);
    if (!fa || !fb || !ra || !rb || fa.count < 8 || fb.count < 8) return null;
    const pipeline = this.deps.pipeline();
    const m = await this.deps.gpu(() => pipeline.match(fa, fb));
    if (m.length >> 1 < this.minPairInliers) return null;
    const ca = this.intrinsics(ra.width, ra.height);
    const cb = this.intrinsics(rb.width, rb.height);
    const Rp = this.gyroRotation(ra.t, rb.t);
    return estimateTwoViewGeometry(fa.keypoints, fb.keypoints, m, ca, cb, { maxIterations: opts.maxIterations ?? 2000, rotationPrior: Rp ?? undefined });
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
    // vocabulary retrieval
    if (st.vocab && nRet > 0) {
      const res = st.vocab.query(fs, nRet + 4, (x: number) => chosen.has(x));
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
      if (job.tracking && job.tracking.other === c.other && usable(job.tracking.geom, this.minPairInliers)) g = job.tracking.geom;
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
    const res = this.mapper.addKeyframe(id);
    this.consecutiveUnregistered = res.registered ? 0 : this.consecutiveUnregistered + 1;
    this.modelVersion++;
    // vocabulary: train once enough keyframes exist, then index incrementally
    this.updateVocabulary(id);
    this.maybeVerifyCameraFromDevice();
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
    const ap = rec.images.get(anchor)!.pose;
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

  private updateVocabulary(id: number): void {
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
      } else if (!st.vocab.has(id)) st.vocab.add(id, st.db.features.get(id)!);
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
    const cams: LiveCamera[] = [];
    for (const { id, pose } of snap.poses.sort((a, b) => a.id - b.id)) {
      const cam = rec.cameras.get(rec.images.get(id)!.cameraId)!;
      cams.push({ pose, intrinsics: { ...cam } });
    }
    u.cameras = cams;
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

  dispose(): void {
    this.disposed = true;
    this.ref = null;
  }
}


