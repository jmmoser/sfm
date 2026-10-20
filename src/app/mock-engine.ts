/**
 * MockEngine: an in-page EngineLike for UI development and tests (`?mock=1`). It fakes the
 * live pipeline (keyframe decisions, tracking state, guidance, a growing sparse cloud and
 * camera trajectory, keypoint overlays that are projections of the scene into the current
 * mock camera), runs refine / densify / mesh as timed stages with progress, and returns REAL
 * PointCloud / TriangleMesh objects and export Blobs encoded with src/io.
 *
 * Time base: the mock trajectory advances with the frame timestamps (`FrameInput.t`) times
 * `speed`, so pauses in frame delivery pause the scan.
 */
import type { MetricInfo, PointCloud, ProgressCallback, ProgressEvent, TriangleMesh } from '../core/types';
import { createColmapZipBlob, encodeGLB, encodeMeshPLY, encodePLY, type ColmapModelInput } from '../io';
import { SessionStore, type SessionMeta } from '../io/storage';
import { Rng } from '../math/random';
import {
  ENGINE_ERRORS,
  namedError,
  type DenseOptions,
  type EngineOptions,
  type ExportFormat,
  type FrameInput,
  type Guidance,
  type InitResult,
  type KeyframeDecision,
  type LiveCamera,
  type LiveUpdate,
  type LogMessage,
  type MeshOptions,
  type QualityPreset,
  type SessionSummary,
  type SparseCamera,
  type SparseResult,
  type TrackingState,
} from '../workers/protocol';
import { ABORT_REASON_PREFIX, Emitter, type AbortOutcome, type EngineErrorInfo, type EngineLike, type EngineStatus } from './engine-client';
import { MockScene, type MockSamples } from './mock-scene';

export interface MockEngineOptions {
  /** Time multiplier for the scan and the processing stages (default 1). */
  speed?: number;
  seed?: number;
  /** Session metadata source for loadSession (default: SessionStore.open()). */
  sessions?: () => Promise<SessionStore | null>;
  /** Make a method fail (tests / error UI). */
  failOn?: Partial<Record<'init' | 'frame' | 'refine' | 'densify' | 'mesh' | 'export' | 'loadSession', string>>;
  /**
   * Make a long operation hang and IGNORE cancel (a stuck worker): it only ends through
   * `abortLongOperation` (simulated engine restart) or dispose. Tests the abort UI.
   */
  stallOn?: Partial<Record<'finishCapture' | 'refine' | 'densify' | 'mesh', boolean>>;
  /**
   * Like the real engine's capture memory cap: after this many keyframes no more are added
   * (tracking continues, `timings.memoryLimited = 1`) and one non-fatal 'out-of-memory' error
   * event (CaptureMemoryLimitError) is emitted.
   */
  memoryLimitKeyframes?: number;
}

interface Sizes {
  sparse: number;
  dense: number;
}

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, Math.max(0, ms)));

function frameSize(f: FrameInput): { w: number; h: number } {
  const b = f.bitmap as { width?: number; height?: number; displayWidth?: number; displayHeight?: number };
  return { w: b.width || b.displayWidth || 1280, h: b.height || b.displayHeight || 960 };
}

export class MockEngine implements EngineLike {
  readonly kind = 'mock' as const;
  private readonly speed: number;
  private readonly seed: number;
  private readonly failOn: NonNullable<MockEngineOptions['failOn']>;
  private readonly stallOn: NonNullable<MockEngineOptions['stallOn']>;
  private readonly memoryLimitKeyframes: number;
  private memoryLimitHit = false;
  /** Rejects the running long operation (abortLongOperation: simulated restart). */
  private abortCurrent: ((e: Error) => void) | null = null;
  /** Ends a stalled operation (abort / dispose). */
  private releaseStall: (() => void) | null = null;
  /** A densify was interrupted: the next one resumes (reported as stats.denseResumablePass). */
  private denseResumablePass: number | null = null;
  /** Quality of the interrupted densify (a resume needs the same one; another starts over). */
  private denseResumableQuality: QualityPreset | null = null;
  private readonly sessionsSrc: () => Promise<SessionStore | null>;
  private st: EngineStatus = { state: 'idle', respawns: 0 };
  private options: EngineOptions | null = null;
  private scene: MockScene | null = null;
  private samples: MockSamples | null = null;
  private sizes: Sizes = { sparse: 9000, dense: 160_000 };
  /** The init / loadSession running now: a closeSession of its scan ends it (EngineClient: 'close' disposes the engine). */
  private pendingSetup: { id: string | null; closed: boolean } | null = null;
  private busy = false;
  private hidden = false;
  private cancelled = false;
  private longOp = false;
  private disposed = false;
  // Live state.
  private frameIndex = 0;
  private lastT: number | null = null;
  private scanMs = 0;
  private keyframes: { s: number; pose: LiveCamera }[] = [];
  private lastKfS = -1;
  private revealedCount = 0;
  private lastSparseKf = -1;
  private captureDone = false;
  private sparse: SparseResult | null = null;
  private denseCloud: PointCloud | null = null;
  private meshModel: TriangleMesh | null = null;
  private sessionId: string | null = null;
  private sessionName = 'Mock scan';
  private readonly live = new Emitter<LiveUpdate>();
  private readonly progressE = new Emitter<ProgressEvent>();
  private readonly logE = new Emitter<LogMessage>();
  private readonly errorE = new Emitter<EngineErrorInfo>();
  private readonly statusE = new Emitter<EngineStatus>();

  constructor(opts: MockEngineOptions = {}) {
    this.speed = opts.speed && opts.speed > 0 ? opts.speed : 1;
    this.seed = opts.seed ?? 7;
    this.failOn = opts.failOn ?? {};
    this.stallOn = opts.stallOn ?? {};
    this.memoryLimitKeyframes = opts.memoryLimitKeyframes && opts.memoryLimitKeyframes > 0 ? opts.memoryLimitKeyframes : Infinity;
    this.sessionsSrc =
      opts.sessions ??
      (async () => {
        try {
          return SessionStore.isSupported() ? await SessionStore.open() : null;
        } catch {
          return null;
        }
      });
  }

  get status(): EngineStatus {
    return { ...this.st };
  }
  get frameInFlight(): boolean {
    return this.busy;
  }

  onLive(cb: (u: LiveUpdate) => void): () => void {
    return this.live.on(cb);
  }
  onProgress(cb: (p: ProgressEvent) => void): () => void {
    return this.progressE.on(cb);
  }
  onLog(cb: (l: LogMessage) => void): () => void {
    return this.logE.on(cb);
  }
  onError(cb: (e: EngineErrorInfo) => void): () => void {
    return this.errorE.on(cb);
  }
  onStatus(cb: (s: EngineStatus) => void): () => void {
    return this.statusE.on(cb);
  }

  private setState(state: EngineStatus['state'], reason?: string, init?: InitResult): void {
    this.st = { state, respawns: 0, ...(reason ? { reason } : {}), ...(init ? { init } : this.st.init ? { init: this.st.init } : {}) };
    this.statusE.emit({ ...this.st });
  }

  private check(method: keyof NonNullable<MockEngineOptions['failOn']>): void {
    if (this.disposed) throw namedError(ENGINE_ERRORS.crashed, 'Engine disposed');
    const msg = this.failOn[method];
    if (msg) throw new Error(msg);
  }

  private log(message: string): void {
    this.logE.emit({ type: 'log', level: 'info', message });
  }

  private reset(kind: 'object' | 'scene', quality: EngineOptions['quality']): void {
    this.scene = new MockScene(kind, this.seed);
    this.sizes =
      kind === 'object'
        ? { sparse: quality === 'fast' ? 6000 : quality === 'max' ? 14000 : 9000, dense: quality === 'fast' ? 80_000 : quality === 'max' ? 260_000 : 160_000 }
        : { sparse: quality === 'fast' ? 8000 : quality === 'max' ? 18000 : 12000, dense: quality === 'fast' ? 120_000 : quality === 'max' ? 320_000 : 200_000 };
    this.samples = this.scene.sample(this.sizes.sparse);
    this.frameIndex = 0;
    this.lastT = null;
    this.scanMs = 0;
    this.keyframes = [];
    this.lastKfS = -1;
    this.revealedCount = 0;
    this.lastSparseKf = -1;
    this.captureDone = false;
    this.sparse = null;
    this.denseCloud = null;
    this.meshModel = null;
    this.cancelled = false;
    this.memoryLimitHit = false;
    this.denseResumablePass = null;
    this.denseResumableQuality = null;
  }

  async init(options: EngineOptions): Promise<InitResult> {
    this.setState('starting');
    const setup = (this.pendingSetup = { id: options.sessionId ?? null, closed: false });
    await sleep(350 / this.speed);
    if (this.pendingSetup === setup) this.pendingSetup = null;
    if (setup.closed) throw namedError(ENGINE_ERRORS.crashed, 'Engine disposed');
    try {
      this.check('init');
    } catch (e) {
      this.setState('idle', (e as Error).message);
      throw e;
    }
    this.options = { ...options };
    this.sessionId = options.sessionId ?? null;
    this.reset(options.mode === 'scene' ? 'scene' : 'object', options.quality);
    const gpu = typeof navigator !== 'undefined' && !!(navigator as Navigator & { gpu?: unknown }).gpu;
    const res: InitResult = { gpu, adapter: 'Mock engine (no reconstruction)' };
    this.setState('ready', undefined, res);
    this.log(`mock init ${options.mode}/${options.quality}`);
    return res;
  }

  // ---- live ----------------------------------------------------------------------------------

  /** Trajectory parameter of the scan so far (1 = one full orbit / path). */
  private get progressS(): number {
    const periodMs = this.scene?.kind === 'scene' ? 60_000 : 50_000;
    return this.scanMs / periodMs;
  }

  async processFrame(f: FrameInput): Promise<LiveUpdate> {
    const b = f.bitmap as { close?: () => void };
    if (this.busy) {
      b.close?.();
      throw namedError(ENGINE_ERRORS.busy, 'A frame is already being processed');
    }
    this.busy = true;
    try {
      const { w, h } = frameSize(f);
      try {
        b.close?.();
      } catch {
        /* detached */
      }
      this.check('frame');
      if (!this.scene || !this.samples) throw namedError(ENGINE_ERRORS.notInitialized, 'init() first');
      const rng = new Rng(this.seed * 1009 + this.frameIndex);
      // Advance the scan clock; long gaps (paused / backgrounded) count as one slow frame.
      if (this.lastT !== null) {
        const dt = f.t - this.lastT;
        if (dt > 0) this.scanMs += Math.min(dt, 400) * this.speed;
      }
      this.lastT = f.t;
      await sleep((25 + rng.next() * 45) / Math.sqrt(this.speed));

      const s = this.progressS;
      const nKf = this.keyframes.length;
      const blur = this.frameIndex > 4 && rng.next() < 0.06;
      const stepS = this.scene.kind === 'object' ? 10 / 360 : 1 / 40;
      let decision: KeyframeDecision = 'skip';
      if (blur) decision = rng.next() < 0.5 ? 'reject_blur' : 'reject_motion';
      else if (this.lastKfS < 0 || s - this.lastKfS >= stepS) decision = 'keyframe';
      // Capture memory cap (real engine: CaptureMemoryLimitError once, tracking continues).
      const memoryLimited = this.keyframes.length >= this.memoryLimitKeyframes;
      if (memoryLimited && decision === 'keyframe') decision = 'skip';
      if (memoryLimited && !this.memoryLimitHit) {
        this.memoryLimitHit = true;
        this.errorE.emit({
          error: namedError('CaptureMemoryLimitError', `Capture memory limit reached after ${this.keyframes.length} keyframes: no more keyframes, tracking continues`, 'out-of-memory'),
          fatal: false,
          reason: 'out-of-memory',
        });
      }
      const cam = this.scene.intrinsics(w, h);
      const pose = this.scene.poseAt(s);
      if (decision === 'keyframe') {
        this.keyframes.push({ s, pose: { pose: this.noisyPose(pose, rng, nKf), intrinsics: cam } });
        this.lastKfS = s;
      }
      const k = this.keyframes.length;
      const tracking: TrackingState = k < 2 ? 'initializing' : 'tracking';
      // Reveal points observed so far (after initialisation).
      const reveal = this.samples.reveal;
      let revealed = 0;
      if (k >= 2) {
        const sNow = this.keyframes[k - 1].s;
        for (let i = 0; i < this.samples.count; i++) if (reveal[i] <= sNow) revealed++;
      }
      this.revealedCount = revealed;
      let guidance: Guidance = 'good';
      if (tracking === 'initializing') guidance = k === 0 ? 'initializing' : 'translate_more';
      else if (decision === 'reject_blur' || decision === 'reject_motion') guidance = 'move_slower';
      else if (Math.floor(this.scanMs / 9000) % 4 === 3 && (this.scanMs % 9000) < 1600) guidance = 'more_texture';

      const upd: LiveUpdate = {
        frameIndex: this.frameIndex++,
        decision,
        guidance,
        tracking,
        numKeyframes: k,
        numRegistered: Math.max(0, k - (k > 6 ? 1 : 0)),
        numPoints: revealed,
        timings: { total: 40 + rng.next() * 30, features: 18 + rng.next() * 8, matching: 6 + rng.next() * 4, ...(memoryLimited ? { memoryLimited: 1 } : {}) },
      };
      // Overlay: projections of the visible scene into the current (true) camera.
      upd.overlay = this.overlay(pose, cam, rng);
      // the real engine poses every tracked frame (frame PnP in the live model)
      if (tracking === 'tracking' && decision !== 'reject_blur' && decision !== 'reject_motion') upd.framePose = { pose, t: f.t };
      if (decision === 'keyframe' && k >= 2 && k !== this.lastSparseKf) {
        this.lastSparseKf = k;
        upd.sparse = this.liveSparse(k);
        upd.cameras = this.keyframes.map((x) => x.pose);
      }
      this.live.emit(upd);
      return upd;
    } finally {
      this.busy = false;
    }
  }

  private noisyPose(pose: LiveCamera['pose'], rng: Rng, k: number): LiveCamera['pose'] {
    const sc = (this.scene?.scale ?? 1) * 0.004 * (3 / (k + 3));
    const t = new Float64Array(pose.t);
    for (let i = 0; i < 3; i++) t[i] += rng.gaussian() * sc;
    return { R: new Float64Array(pose.R), t };
  }

  private overlay(pose: LiveCamera['pose'], cam: LiveCamera['intrinsics'], rng: Rng): Float32Array {
    const smp = this.samples!;
    const stride = Math.max(1, Math.floor(smp.count / 2400));
    const pts = this.scene!.project(smp.positions, smp.count, pose, cam, 260, stride);
    // Small jitter: detected keypoints are not exactly the projections.
    for (let i = 0; i < pts.length; i++) pts[i] = Math.min(1, Math.max(0, pts[i] + (rng.next() - 0.5) * 0.003));
    return pts;
  }

  private liveSparse(k: number): { positions: Float32Array; colors: Uint8Array } {
    const smp = this.samples!;
    const sNow = this.keyframes[k - 1]?.s ?? 0;
    const noise = (this.scene!.scale * 0.006 * 4) / (k + 4);
    const rng = new Rng(this.seed + k);
    const pos: number[] = [];
    const col: number[] = [];
    for (let i = 0; i < smp.count; i++) {
      if (smp.reveal[i] > sNow) continue;
      pos.push(smp.positions[3 * i] + rng.gaussian() * noise, smp.positions[3 * i + 1] + rng.gaussian() * noise, smp.positions[3 * i + 2] + rng.gaussian() * noise);
      col.push(smp.colors[3 * i], smp.colors[3 * i + 1], smp.colors[3 * i + 2]);
    }
    return { positions: new Float32Array(pos), colors: new Uint8Array(col) };
  }

  finishCapture(): Promise<void> {
    return this.longRun(async () => {
      if (this.stallOn.finishCapture) await this.stall('mapping 2 keyframes');
      await this.stage('matching', 200, undefined, 'finishing capture');
      this.captureDone = true;
    });
  }

  /** Hang (ignoring cancel) like a stuck worker; progress keeps coming like the real engine's. */
  private stall(message: string): Promise<never> {
    return new Promise<never>((_, reject) => {
      const timer = setInterval(() => this.progressE.emit({ stage: 'matching', progress: -1, message }), 1000);
      this.releaseStall = () => {
        clearInterval(timer);
        this.releaseStall = null;
        reject(namedError(ENGINE_ERRORS.cancelled, 'Aborted'));
      };
    });
  }

  // ---- processing ----------------------------------------------------------------------------

  private async stage(name: string, ms: number, onProgress?: ProgressCallback, message?: string): Promise<void> {
    const total = ms / this.speed;
    let elapsed = 0;
    const tick = 80;
    const emit = (p: number): void => {
      const e: ProgressEvent = { stage: name, progress: p };
      if (message) e.message = message;
      try {
        onProgress?.(e);
      } catch (err) {
        console.error(err);
      }
      this.progressE.emit(e);
    };
    emit(0);
    while (elapsed < total) {
      if (this.cancelled) throw namedError(ENGINE_ERRORS.cancelled, 'Cancelled');
      if (this.disposed) throw namedError(ENGINE_ERRORS.crashed, 'Engine disposed');
      await sleep(tick);
      if (this.hidden) continue; // the real engine pauses GPU work while hidden
      elapsed += tick;
      emit(Math.min(1, elapsed / total));
    }
    if (this.cancelled) throw namedError(ENGINE_ERRORS.cancelled, 'Cancelled');
  }

  private async longRun<T>(fn: () => Promise<T>): Promise<T> {
    this.cancelled = false;
    this.longOp = true;
    const aborted = new Promise<never>((_, reject) => {
      this.abortCurrent = reject;
    });
    try {
      return await Promise.race([fn(), aborted]);
    } finally {
      this.longOp = false;
      this.cancelled = false;
      this.abortCurrent = null;
    }
  }

  private ensureScene(): MockScene {
    if (!this.scene || !this.samples) throw namedError(ENGINE_ERRORS.notInitialized, 'No session loaded');
    return this.scene;
  }

  private metric(): MetricInfo {
    return this.scene?.kind === 'scene'
      ? { metric: true, source: 'imu+gps', relativeUncertainty: 0.012, gravityAligned: true, northAligned: true, originLatLonAlt: [47.3769, 8.5417, 408] }
      : // object scans: yaw from the compass without a location → MAGNETIC north (northAligned false)
        { metric: true, source: 'imu', relativeUncertainty: 0.028, gravityAligned: true, northAligned: false, originLatLonAlt: null };
  }

  private buildSparse(numKeyframes: number, timings: Record<string, number>): SparseResult {
    const scene = this.ensureScene();
    const smp = this.samples!;
    const n = Math.max(numKeyframes, 12);
    const span = Math.max(this.keyframes.length ? this.keyframes[this.keyframes.length - 1].s : 1, 0.25);
    const rng = new Rng(this.seed + 99);
    const cams: SparseCamera[] = [];
    const w = scene.kind === 'object' ? 1440 : 1920;
    const h = scene.kind === 'object' ? 1920 : 1440;
    const intr = { ...scene.intrinsics(w, h), k1: -0.0213, k2: 0.0041 };
    for (let i = 0; i < n; i++) {
      const s = this.keyframes[i]?.s ?? (span * i) / (n - 1);
      cams.push({ id: i, pose: scene.poseAt(s), intrinsics: intr });
    }
    const pos: number[] = [];
    const col: number[] = [];
    const noise = scene.scale * 0.0012;
    for (let i = 0; i < smp.count; i++) {
      if (smp.reveal[i] > span + 0.02) continue;
      pos.push(smp.positions[3 * i] + rng.gaussian() * noise, smp.positions[3 * i + 1] + rng.gaussian() * noise, smp.positions[3 * i + 2] + rng.gaussian() * noise);
      col.push(smp.colors[3 * i], smp.colors[3 * i + 1], smp.colors[3 * i + 2]);
    }
    const count = pos.length / 3;
    return {
      points: { count, positions: new Float32Array(pos), colors: new Uint8Array(col) },
      cameras: cams,
      metric: this.metric(),
      stats: {
        images: n,
        registeredImages: n,
        points: count,
        meanReprojErrorPx: 0.61,
        meanTrackLength: 5.8,
        focalPx: Math.round(intr.f * 10) / 10,
        k1: intr.k1,
        k2: intr.k2,
        scaleSource: scene.kind === 'scene' ? 'imu+gps' : 'imu',
        yawSource: scene.kind === 'scene' ? 'gps' : 'compass',
        north: scene.kind === 'scene' ? 'true' : 'magnetic',
        yawStdDeg: scene.kind === 'scene' ? 1.8 : 4.5,
        memoryMB: Math.round(150 + n * 0.6),
        ...timings,
      },
    };
  }

  refine(onProgress?: ProgressCallback): Promise<SparseResult> {
    return this.longRun(async () => {
      this.check('refine');
      this.ensureScene();
      const t: Record<string, number> = {};
      const run = async (name: string, ms: number, key: string): Promise<void> => {
        const t0 = performance.now();
        await this.stage(name, ms, onProgress);
        t[key] = Math.round(performance.now() - t0);
      };
      await run('matching', 1600, 'matchingMs');
      await run('reconstruction', 2200, 'reconstructionMs');
      await run('alignment', 700, 'alignmentMs');
      this.sparse = this.buildSparse(this.keyframes.length, t);
      this.captureDone = true;
      this.denseResumablePass = null; // refine invalidates stored depth maps
      this.denseResumableQuality = null;
      return this.sparse;
    });
  }

  densify(options?: DenseOptions, onProgress?: ProgressCallback): Promise<PointCloud> {
    return this.longRun(async () => {
      this.check('densify');
      const scene = this.ensureScene();
      if (!this.sparse) throw new Error('Run refine() first');
      const q = options?.quality ?? this.options?.quality ?? 'balanced';
      const f = q === 'fast' ? 0.6 : q === 'max' ? 1.6 : 1;
      const t0 = performance.now();
      const resumed = this.denseResumablePass !== null && (this.denseResumableQuality ?? q) === q ? this.denseResumablePass : null;
      let t1 = t0;
      try {
        await this.stage('depth_maps', 2600 * f * (resumed !== null ? 0.4 : 1), onProgress, `${this.sparse.cameras.length} views`);
        t1 = performance.now();
        await this.stage('fusion', 1100 * f, onProgress);
      } catch (e) {
        // Like the real engine: stored depth maps make an interrupted densify resumable.
        if ((e as { name?: string })?.name === ENGINE_ERRORS.cancelled) {
          this.denseResumablePass = 1;
          this.denseResumableQuality = q;
        }
        throw e;
      }
      this.denseResumablePass = null;
      this.denseResumableQuality = null;
      const n = Math.round(this.sizes.dense * (q === 'fast' ? 0.6 : q === 'max' ? 1.5 : 1));
      this.denseCloud = scene.dense(n);
      const size = q === 'fast' ? 800 : q === 'max' ? 1600 : 1280;
      this.sparse.stats = {
        ...this.sparse.stats,
        densePoints: n,
        depthMapsMs: Math.round(t1 - t0),
        fusionMs: Math.round(performance.now() - t1),
        denseMaxSize: size,
        denseMemPeakMB: Math.round(q === 'fast' ? 210 : q === 'max' ? 820 : 560),
        memoryMB: Math.round(160 + this.keyframes.length * 0.6 + n * 105e-6),
        ...(resumed !== null ? { denseResumedPass: resumed } : {}),
      };
      return this.denseCloud;
    });
  }

  mesh(options?: MeshOptions, onProgress?: ProgressCallback): Promise<TriangleMesh> {
    return this.longRun(async () => {
      this.check('mesh');
      const scene = this.ensureScene();
      if (!this.denseCloud) throw new Error('Run densify() first');
      const q = options?.quality ?? this.options?.quality ?? 'balanced';
      const t0 = performance.now();
      await this.stage('meshing', 1800 * (q === 'fast' ? 0.6 : q === 'max' ? 1.6 : 1), onProgress);
      this.meshModel = scene.mesh();
      if (this.sparse) this.sparse.stats = { ...this.sparse.stats, triangles: this.meshModel.triangleCount, meshingMs: Math.round(performance.now() - t0), memoryMB: Math.round(190 + this.keyframes.length * 0.6) };
      return this.meshModel;
    });
  }

  async exportModel(format: ExportFormat): Promise<Blob> {
    this.check('export');
    await sleep(350 / this.speed);
    const name = this.sessionName;
    switch (format) {
      case 'ply-points': {
        const pc = this.denseCloud ?? this.sparse?.points;
        if (!pc) throw new Error('No point cloud yet');
        return new Blob([encodePLY(pc, { comments: [`exported by sfm mock engine: ${name}`] })], { type: 'application/octet-stream' });
      }
      case 'ply-mesh': {
        if (!this.meshModel) throw new Error('Build the mesh first');
        return new Blob([encodeMeshPLY(this.meshModel)], { type: 'application/octet-stream' });
      }
      case 'glb': {
        const pts = this.denseCloud ?? this.sparse?.points;
        if (!this.meshModel && !pts) throw new Error('Nothing to export yet');
        const glb = encodeGLB(this.meshModel ? { mesh: this.meshModel, extras: { mock: true } } : { points: pts!, extras: { mock: true } });
        return new Blob([glb], { type: 'model/gltf-binary' });
      }
      case 'colmap-zip':
        return this.colmapZip();
    }
  }

  private async colmapZip(): Promise<Blob> {
    const sp = this.sparse;
    if (!sp) throw new Error('No sparse model yet');
    const cams = sp.cameras;
    const cam = cams[0].intrinsics;
    const model: ColmapModelInput = { cameras: [{ id: 1, cam }], images: [], points: [] };
    const names = cams.map((c) => `frame_${String(c.id).padStart(4, '0')}.jpg`);
    const pts2d: { x: number; y: number; point3D: number }[][] = cams.map(() => []);
    const n = Math.min(sp.points.count, 4000);
    const P = sp.points.positions;
    for (let i = 0; i < n; i++) {
      const track: { image: number; point2D: number }[] = [];
      for (let c = 0; c < cams.length && track.length < 6; c++) {
        const { R, t } = cams[c].pose;
        const X = P[3 * i];
        const Y = P[3 * i + 1];
        const Z = P[3 * i + 2];
        const z = R[6] * X + R[7] * Y + R[8] * Z + t[2];
        if (z <= 0.05) continue;
        const u = (cam.f * (R[0] * X + R[1] * Y + R[2] * Z + t[0])) / z + cam.cx;
        const v = (cam.f * (R[3] * X + R[4] * Y + R[5] * Z + t[1])) / z + cam.cy;
        if (u < 0 || v < 0 || u > cam.width - 1 || v > cam.height - 1) continue;
        track.push({ image: c, point2D: pts2d[c].length });
        pts2d[c].push({ x: u, y: v, point3D: i });
      }
      if (track.length >= 2) {
        model.points.push({
          id: i,
          xyz: [P[3 * i], P[3 * i + 1], P[3 * i + 2]],
          color: [sp.points.colors[3 * i], sp.points.colors[3 * i + 1], sp.points.colors[3 * i + 2]],
          error: 0.6,
          track,
        });
      } else {
        // Drop the observations of a point that is not exported.
        for (const tr of track) pts2d[tr.image][tr.point2D].point3D = -1;
      }
    }
    model.images = cams.map((c, i) => ({ id: i, cameraId: 1, name: names[i], pose: c.pose, points2D: pts2d[i] }));
    const images = await Promise.all(names.map((nm, i) => this.placeholderJpeg(i).then((data) => (data ? { name: nm, data } : null))));
    return createColmapZipBlob(
      model,
      images.filter((x): x is { name: string; data: Blob } => !!x),
      { readme: 'Generated by the sfm MockEngine (synthetic data, not a real reconstruction).\n' },
    );
  }

  private async placeholderJpeg(i: number): Promise<Blob | null> {
    try {
      if (typeof OffscreenCanvas === 'undefined') return null;
      const c = new OffscreenCanvas(160, 120);
      const ctx = c.getContext('2d');
      if (!ctx) return null;
      const g = ctx.createLinearGradient(0, 0, 160, 120);
      g.addColorStop(0, `hsl(${(i * 23) % 360} 50% 40%)`);
      g.addColorStop(1, `hsl(${(i * 23 + 80) % 360} 50% 20%)`);
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, 160, 120);
      ctx.fillStyle = '#fff';
      ctx.font = '16px sans-serif';
      ctx.fillText(`frame ${i}`, 12, 64);
      return await c.convertToBlob({ type: 'image/jpeg', quality: 0.8 });
    } catch {
      return null;
    }
  }

  async loadSession(id: string, options?: Partial<EngineOptions>): Promise<SessionSummary> {
    this.setState('starting');
    const setup = (this.pendingSetup = { id, closed: false });
    const stopIfClosed = (): void => {
      if (setup.closed) throw namedError(ENGINE_ERRORS.crashed, 'Engine disposed');
    };
    try {
      this.check('loadSession');
      await sleep(250 / this.speed);
      stopIfClosed();
      let meta: SessionMeta | null = null;
      if (this.sessionId === id && this.scene && this.options) {
        meta = null; // in-memory session: keep the current results
      } else {
        const store = await this.sessionsSrc();
        meta = store ? await store.getSession(id) : null;
        stopIfClosed();
        if (!meta) throw new Error(`Session not found: ${id}`);
        const mode = meta.mode === 'scene' ? 'scene' : 'object';
        const quality = (options?.quality ?? (meta.stats?.quality as EngineOptions['quality'] | undefined) ?? 'balanced') as EngineOptions['quality'];
        this.options = { mode: meta.mode, quality, sessionId: id, persist: true };
        this.sessionId = id;
        this.reset(mode, quality);
        // A finished capture: the whole trajectory was scanned.
        const kf = Math.max(8, Math.min(36, meta.numFrames || 24));
        const span = Math.min(1, kf / 36);
        const scene = this.scene!;
        for (let i = 0; i < kf; i++) {
          const s = (span * i) / Math.max(1, kf - 1);
          this.keyframes.push({ s, pose: { pose: scene.poseAt(s), intrinsics: scene.intrinsics(1440, 1920) } });
        }
        this.captureDone = true;
        const st = meta.status;
        if (st === 'sparse' || st === 'dense' || st === 'meshed') this.sparse = this.buildSparse(kf, {});
        if (st === 'dense' || st === 'meshed') this.denseCloud = scene.dense(this.sizes.dense);
        if (st === 'meshed') this.meshModel = scene.mesh();
        const rp = meta.stats?.denseResumablePass;
        if (typeof rp === 'number' && this.sparse && !this.denseCloud) this.denseResumablePass = rp;
        const rq = meta.stats?.denseResumableQuality;
        if (rq === 'fast' || rq === 'balanced' || rq === 'max') this.denseResumableQuality = rq;
        this.sessionName = meta.name;
      }
      this.setState('ready');
      const m = meta;
      return {
        id,
        name: m?.name ?? this.sessionName,
        mode: this.options!.mode,
        quality: this.options!.quality,
        status: this.meshModel ? 'meshed' : this.denseCloud ? 'dense' : this.sparse ? 'sparse' : 'captured',
        numFrames: m?.numFrames ?? this.keyframes.length,
        createdAt: m?.createdAt ?? Date.now(),
        updatedAt: m?.updatedAt ?? Date.now(),
        sparse: this.sparse,
        dense: this.denseCloud,
        mesh: this.meshModel,
        stats: {
          ...(this.sparse?.stats ?? {}),
          ...(this.denseResumablePass !== null ? { denseResumablePass: this.denseResumablePass } : {}),
          ...(this.denseResumablePass !== null && this.denseResumableQuality ? { denseResumableQuality: this.denseResumableQuality } : {}),
        },
      };
    } catch (e) {
      if (!setup.closed) this.setState('idle', (e as Error).message); // a close set 'closed'
      throw e;
    } finally {
      if (this.pendingSetup === setup) this.pendingSetup = null;
    }
  }

  /** Name used in export metadata. */
  setSessionName(name: string): void {
    this.sessionName = name;
  }

  cancel(): void {
    if (this.longOp) this.cancelled = true;
  }

  async abortLongOperation(opts: { graceMs?: number; reinit?: boolean } = {}): Promise<AbortOutcome> {
    if (!this.longOp) return 'none';
    this.cancel();
    const graceMs = Math.max(0, opts.graceMs ?? 5000);
    const t0 = performance.now();
    while (this.longOp && performance.now() - t0 < graceMs) await sleep(40);
    if (!this.longOp) return 'cancelled';
    // Stuck: end it here and "restart" (the in-memory session survives, like a resume from
    // IndexedDB — unless `reinit: false`: then nothing is re-opened, like EngineClient).
    this.abortCurrent?.(namedError(ENGINE_ERRORS.cancelled, 'The operation did not stop — the engine was restarted'));
    this.releaseStall?.();
    this.setState('respawning', `${ABORT_REASON_PREFIX}the operation did not respond to cancel`);
    await sleep(250 / this.speed);
    if (this.disposed) return 'restarted';
    if (opts.reinit === false) {
      this.forgetSession();
      this.setState('idle', 'restarted without re-opening the session');
    } else this.setState('ready', 'recovered after an abort');
    return 'restarted';
  }

  /** Like EngineClient.closeSession: the session is forgotten (a running operation is rejected). */
  async closeSession(id: string): Promise<void> {
    if (this.disposed) throw namedError(ENGINE_ERRORS.crashed, 'Engine disposed');
    if (this.pendingSetup?.id === id) {
      this.pendingSetup.closed = true;
      this.pendingSetup = null;
      this.setState('idle', 'closed');
    }
    if (this.sessionId !== id) return;
    if (this.longOp) {
      this.cancelled = true;
      this.abortCurrent?.(namedError(ENGINE_ERRORS.cancelled, 'The session was closed'));
      this.releaseStall?.();
    }
    this.forgetSession();
    if (this.st.state === 'ready') this.setState('idle', 'closed');
  }

  /** Forget the session (EngineClient.release: a fresh worker). A running operation is rejected. */
  async release(): Promise<void> {
    if (this.disposed) throw namedError(ENGINE_ERRORS.crashed, 'Engine disposed');
    if (this.longOp) {
      this.cancelled = true;
      this.abortCurrent?.(namedError(ENGINE_ERRORS.cancelled, 'The engine was released'));
      this.releaseStall?.();
    }
    this.forgetSession();
    this.setState('idle', 'released');
  }

  private forgetSession(): void {
    this.options = null;
    this.sessionId = null;
    this.scene = null;
    this.samples = null;
    this.keyframes = [];
    this.captureDone = false;
    this.sparse = null;
    this.denseCloud = null;
    this.meshModel = null;
    this.denseResumablePass = null;
    this.denseResumableQuality = null;
  }

  setHidden(hidden: boolean): void {
    this.hidden = hidden;
  }

  dispose(): void {
    this.disposed = true;
    this.cancelled = true;
    this.releaseStall?.();
    this.setState('disposed');
    this.live.clear();
    this.progressE.clear();
    this.logE.clear();
    this.errorE.clear();
    this.statusE.clear();
  }

  /** Test helper: the scan parameter (fraction of the orbit / path). */
  get scanProgress(): number {
    return this.progressS;
  }
  /** Test helper. */
  get revealedPoints(): number {
    return this.revealedCount;
  }
  get captureFinished(): boolean {
    return this.captureDone;
  }
}
