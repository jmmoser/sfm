/**
 * PhotogrammetryEngine (docs/ENGINE.md §2): thread-agnostic orchestration of capture → live SfM
 * → refine → dense → mesh → export, with persistence and resume. Runs inside the engine worker
 * (src/workers/engine.worker.ts) in the app and inline (main thread / Node) in tests and the
 * harness.
 *
 *   const engine = new PhotogrammetryEngine({ mode: 'object', quality: 'balanced', persist: false });
 *   await engine.init();
 *   for (const f of frames) await engine.processFrame(f);   // LiveUpdate per frame
 *   await engine.finishCapture();
 *   const sparse = await engine.refine(onProgress);          // SparseResult (+ t / frameIndex per camera)
 *   const cloud = await engine.densify();                     // PointCloud (world frame)
 *   const mesh = await engine.mesh();
 *   const blob = await engine.exportModel('ply-points');
 *
 * Model updates produced by background keyframe mapping arrive through `on('live', …)`.
 */
import type { CancelToken, FeatureSet, PointCloud, ProgressCallback, TriangleMesh } from '../core/types';
import type { FeaturePipeline } from '../features/extractor';
import type { SiftOptions } from '../features/sift-common';
import { releaseSiftCPUMemory } from '../features/sift-cpu';
import { getGPU, hasWebGPU } from '../gpu/device';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { deserializeFeatureSet, deserializePointCloud, deserializeTriangleMesh, serializePointCloud, serializeTriangleMesh } from '../io/serialize';
import {
  ENGINE_ERRORS,
  namedError,
  type DenseOptions,
  type EngineOptions,
  type ExportFormat,
  type DecodedFrameInput,
  type FrameInput,
  type InitResult,
  type LiveUpdate,
  type LogLevel,
  type MeshOptions,
  type QualityPreset,
  type SessionSummary,
} from '../workers/protocol';
import { EngineDepthStore } from './depth-store';
import { PatchMatchHolder, rebuildColourViews, runDensify, runMesh, type DenseOutput } from './dense';
import { buildExport } from './export';
import { closeSource, rotateVec } from './frame-ingest';
import { LiveSession } from './live';
import { modelKeypoints, originalKeypoints } from './refine-camera';
import {
  MemoryEngineStore,
  SessionEngineStore,
  packSparseModel,
  unpackSparseModel,
  type DenseMetaRecord,
  type EngineStateRecord,
  type EngineStore,
  type SparseMetaRecord,
} from './persistence';
import { MB, deviceMemoryBudgetMB, formatBreakdown, sumBreakdown, type GpuMemoryHints, type MemoryBreakdown } from './memory';
import { ENGINE_DEFAULTS, QUALITY_PRESETS, type QualityPresetConfig, type ResolvedMode } from './presets';
import { modelPointCloud, runRefine } from './refine';
import { CpuPool, defaultCpuWorkers } from './cpu-pool';
import type { EngineConfig, EngineEvents, KeyframeRecord, SparseModel, SparseResultEx } from './types';
import { cameraFromDevice } from '../sensors/frame-orientation';

/** Bytes of a plain-array sparse model. */
function sparseModelBytes(m: SparseModel): number {
  return m.xyz.byteLength + m.colors.byteLength + m.errors.byteLength + m.trackOffsets.byteLength + m.trackImages.byteLength + m.trackFeatures.byteLength + 300 * m.images.length;
}

function cloudBytes(pc: PointCloud | null | undefined): number {
  if (!pc) return 0;
  return pc.positions.byteLength + (pc.colors?.byteLength ?? 0) + (pc.normals?.byteLength ?? 0) + (pc.radii?.byteLength ?? 0) + (pc.confidence?.byteLength ?? 0);
}

function meshBytes(m: TriangleMesh | null | undefined): number {
  if (!m) return 0;
  let b = 0;
  for (const v of Object.values(m)) if (ArrayBuffer.isView(v)) b += v.byteLength;
  return b;
}

/**
 * GPU memory of a WebGPU feature pipeline (SIFT pyramids + matcher descriptor cache), read
 * defensively (the pipeline class keeps its SiftGPU / MatcherGPU private).
 */
function pipelineGpuBytes(p: FeaturePipeline | null): number {
  if (!p?.gpu) return 0;
  const x = p as unknown as { sift?: { memoryBytes?: () => number }; matcher?: { cacheMemoryBytes?: () => number } };
  try {
    return (x.sift?.memoryBytes?.() ?? 0) + (x.matcher?.cacheMemoryBytes?.() ?? 0);
  } catch {
    return 0;
  }
}

/**
 * Free the feature pipeline's size-dependent GPU caches (SIFT pyramids ≈ 70 MB per cached size
 * at 1600 px, matcher descriptor cache ≤ 64 MB); the compiled pipelines stay. Defensive: uses
 * `releaseCaches()` when the pipeline offers one, else its SiftGPU / MatcherGPU.
 */
function releasePipelineCaches(p: FeaturePipeline | null): void {
  if (!p?.gpu) return;
  const x = p as unknown as { releaseCaches?: () => void; sift?: { releaseCaches?: () => void }; matcher?: { clearCache?: () => void } };
  try {
    if (typeof x.releaseCaches === 'function') x.releaseCaches();
    else {
      x.sift?.releaseCaches?.();
      x.matcher?.clearCache?.();
    }
  } catch {
    /* device gone */
  }
}

type Listener<K extends keyof EngineEvents> = EngineEvents[K];

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/** IndexedDB database of scratch depth maps (non-persistent sessions). */
const SCRATCH_DB = 'sfm-scratch';

/** Key of the SIFT configuration a feature pipeline was created with (pipelines are reusable across engines with the same key). */
export function pipelineKey(quality: QualityPreset, preferGPU: boolean, config: EngineConfig = {}): string {
  const p = QUALITY_PRESETS[quality];
  const gpu = preferGPU && config.featureBackend !== 'cpu';
  return JSON.stringify({ gpu, maxFeatures: p.maxFeatures, maxImageSize: p.featureMaxSide, sift: config.sift ?? {} });
}

export class PhotogrammetryEngine {
  readonly options: EngineOptions;
  readonly config: EngineConfig;
  private store: EngineStore | null = null;
  private ownsStore = false;
  private pipeline: FeaturePipeline | null = null;
  private ownsPipeline = false;
  /**
   * pipelineKey of the SIFT configuration the pipeline was created with (an injected one: the
   * worker's spare, created for the options this engine was constructed with).
   */
  private pipelineCfgKey: string;
  private live: LiveSession | null = null;
  private model: SparseModel | null = null;
  private sparse: SparseResultEx | null = null;
  private denseOut: DenseOutput | null = null;
  private denseCloud: PointCloud | null = null;
  private denseMeta: DenseMetaRecord | null = null;
  private meshModel: TriangleMesh | null = null;
  /** A dense cloud / mesh exists (in memory or only in the store after releaseResultMemory). */
  private hasDense = false;
  private hasMesh = false;
  private readonly holder: PatchMatchHolder;
  private hiddenFlag = false;
  private visibleWaiters: (() => void)[] = [];
  private gpuChain: Promise<unknown> = Promise.resolve();
  private longOp: string | null = null;
  private storageErr: Error | null = null;
  private disposed = false;
  private initialized = false;
  private readonly listeners: { [K in keyof EngineEvents]: Set<Listener<K>> } = { live: new Set(), log: new Set(), error: new Set() };
  private stats: Record<string, number | string> = {};
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  /** Scratch IndexedDB store for depth maps of non-persistent sessions (undefined = not tried). */
  private scratch: SessionEngineStore | null | undefined = undefined;
  /** Resolved capture mode of a session opened with loadSession (no live session). */
  private loadedMode: ResolvedMode | null = null;
  /**
   * Capture state of a session without a live session (loadSession, or after the live session was
   * released for the dense stage): written back unchanged so that orientation, time offset and
   * counters survive later state writes.
   */
  private savedState: EngineStateRecord | null = null;
  private sessionName = 'Scan';
  /** Memory budget derived from the GPU adapter (iPhone RAM class), null = not known. */
  private deviceBudget: { budgetMB: number; reason: string } | null = null;
  private stateWrite: Promise<void> | null = null;
  private stateDirty = false;

  constructor(options: EngineOptions, config: EngineConfig = {}) {
    this.options = { persist: false, preferGPU: true, ...options };
    this.config = config;
    this.pipelineCfgKey = pipelineKey(this.options.quality, this.options.preferGPU !== false, config);
    this.holder = new PatchMatchHolder(this.options.preferGPU !== false, config.dense ?? {}, !!config.denseGpuTimestamps);
  }

  // ------------------------------------------------------------------------------------------
  // events

  on<K extends keyof EngineEvents>(ev: K, cb: EngineEvents[K]): () => void {
    (this.listeners[ev] as Set<EngineEvents[K]>).add(cb);
    return () => (this.listeners[ev] as Set<EngineEvents[K]>).delete(cb);
  }

  private log(level: LogLevel, message: string, data?: unknown): void {
    for (const cb of this.listeners.log) {
      try {
        cb(level, message, data);
      } catch {
        /* ignore */
      }
    }
  }

  private emitLive(u: LiveUpdate): void {
    for (const cb of this.listeners.live) {
      try {
        cb(u);
      } catch (e) {
        this.log('warn', `live listener failed: ${(e as Error)?.message ?? e}`);
      }
    }
  }

  private emitError(err: Error, fatal: boolean, reason?: 'gpu-lost' | 'out-of-memory' | 'internal' | 'storage'): void {
    for (const cb of this.listeners.error) {
      try {
        cb(err, fatal, reason);
      } catch {
        /* ignore */
      }
    }
  }

  // ------------------------------------------------------------------------------------------
  // visibility / GPU gate

  /** Page visibility: while hidden no GPU work is submitted (frames are skipped, long operations pause). */
  setHidden(hidden: boolean): void {
    this.hiddenFlag = hidden;
    if (!hidden) {
      const w = this.visibleWaiters;
      this.visibleWaiters = [];
      for (const r of w) r();
    }
  }

  get hidden(): boolean {
    return this.hiddenFlag;
  }

  /** Resolves when the page is visible (immediately if it is). */
  waitVisible(): Promise<void> {
    if (!this.hiddenFlag || this.disposed) return Promise.resolve();
    return new Promise((r) => this.visibleWaiters.push(r));
  }

  /** Serialise GPU work (one heavy command stream at a time) and hold it while hidden. */
  private gpu<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.gpuChain.then(async () => {
      await this.waitVisible();
      if (this.disposed) throw namedError(ENGINE_ERRORS.crashed, 'Engine disposed');
      return fn();
    });
    this.gpuChain = run.catch(() => undefined);
    return run;
  }

  // ------------------------------------------------------------------------------------------
  // set-up

  private get quality(): QualityPreset {
    return this.options.quality;
  }

  private get preset(): QualityPresetConfig {
    return QUALITY_PRESETS[this.quality];
  }

  /** Tab memory budget (MB): EngineConfig.memoryBudgetMB, else derived from the GPU adapter, else 1200. */
  get memoryBudgetMB(): number {
    return this.config.memoryBudgetMB ?? this.deviceBudget?.budgetMB ?? ENGINE_DEFAULTS.memoryBudgetMB;
  }

  /** Derive the memory budget from the WebGPU adapter (Apple GPUs: maxBufferSize ≈ RAM class). */
  private applyDeviceBudget(ctx: GpuMemoryHints | null): void {
    if (this.deviceBudget || !ctx) return;
    const d = deviceMemoryBudgetMB(ctx);
    if (!d) return;
    this.deviceBudget = d;
    if (this.config.memoryBudgetMB === undefined) this.log('info', `memory budget ${d.budgetMB} MB (${d.reason})`);
  }

  private async ensureDeviceBudget(): Promise<void> {
    if (this.deviceBudget || this.config.memoryBudgetMB !== undefined || this.options.preferGPU === false || !hasWebGPU()) return;
    this.applyDeviceBudget(await getGPU().catch(() => null));
  }

  /**
   * Estimated resident memory of the engine's data (bytes per component): capture state (features,
   * pairs, graph, live model, vocabulary, IMU, pending writes), sparse model, dense cloud, dense
   * colour views, mesh and the feature pipeline's GPU caches. Excludes transient stage buffers
   * (see `planDense`) and the page / JS baseline.
   */
  memoryReport(): { totalMB: number; budgetMB: number; parts: MemoryBreakdown } {
    const parts: MemoryBreakdown = this.live ? this.live.memoryBreakdown() : {};
    if (this.model) parts.sparseModel = sparseModelBytes(this.model);
    const dense = this.denseCloud ?? this.denseOut?.cloud ?? null;
    if (dense) parts.denseCloud = cloudBytes(dense);
    if (this.denseOut) parts.denseViews = this.denseOut.views.reduce((s, v) => s + v.image.data.byteLength, 0);
    if (this.meshModel) parts.mesh = meshBytes(this.meshModel);
    const gpu = pipelineGpuBytes(this.pipeline);
    if (gpu) parts.featureGpu = gpu;
    const total = sumBreakdown(parts);
    return { totalMB: Math.round((10 * total) / MB) / 10, budgetMB: this.memoryBudgetMB, parts };
  }

  private logMemory(when: string): number {
    const r = this.memoryReport();
    this.log('info', `memory after ${when} ≈ ${formatBreakdown(r.parts)} of a ${r.budgetMB} MB budget`, r.parts);
    return r.totalMB;
  }

  /** SIFT options of the feature pipeline (preset working size / feature budget + overrides). */
  private siftOptions(): Partial<SiftOptions> {
    return { maxFeatures: this.preset.maxFeatures, maxImageSize: this.preset.featureMaxSide, ...(this.config.sift ?? {}) };
  }

  private async openStore(sessionId: string | undefined): Promise<void> {
    if (this.store) return;
    if (this.config.store) {
      this.store = this.config.store;
      this.ownsStore = false;
      return;
    }
    if (this.options.persist && sessionId && SessionEngineStore.isSupported()) {
      try {
        this.store = await SessionEngineStore.open(sessionId);
        this.ownsStore = true;
        return;
      } catch (e) {
        this.log('warn', `IndexedDB unavailable, keeping the session in memory: ${(e as Error)?.message ?? e}`);
        this.emitError(e instanceof Error ? e : new Error(String(e)), false, 'storage');
      }
    }
    this.store = new MemoryEngineStore(sessionId ?? 'memory');
    this.ownsStore = true;
  }

  private async ensurePipeline(): Promise<FeaturePipeline> {
    if (this.pipeline) return this.pipeline;
    if (this.config.pipeline) {
      this.pipeline = this.config.pipeline;
      this.ownsPipeline = false;
      return this.pipeline;
    }
    const t0 = now();
    const cpuRequested = this.options.preferGPU === false || this.config.featureBackend === 'cpu';
    if (!cpuRequested && hasWebGPU()) {
      // loaded lazily: the WebGPU feature modules cannot even be imported without WebGPU
      const { createFeaturePipeline } = await import('../features/extractor');
      this.pipeline = await createFeaturePipeline({
        preferGPU: true,
        sift: this.siftOptions(),
        onFallback: (reason) => {
          this.log('warn', `feature pipeline fell back to the CPU: ${reason}`);
        },
      });
    } else {
      const reason = this.options.preferGPU === false ? 'CPU requested (preferGPU: false)' : cpuRequested ? "CPU requested (featureBackend: 'cpu')" : 'WebGPU unavailable';
      this.pipeline = createCpuFeaturePipeline(this.siftOptions(), reason);
    }
    this.ownsPipeline = true;
    this.pipelineCfgKey = pipelineKey(this.quality, this.options.preferGPU !== false, this.config);
    this.log('info', `feature pipeline ready (${this.pipeline.gpu ? 'WebGPU' : 'CPU'}, ${Math.round(now() - t0)} ms)`);
    return this.pipeline;
  }

  private createLive(): LiveSession {
    const live = new LiveSession({
      mode: this.options.mode,
      quality: this.quality,
      hfovPriorDeg: this.options.hfovPriorDeg ?? null,
      config: this.config,
      preset: this.preset,
      store: this.store!,
      pipeline: () => {
        if (!this.pipeline) throw namedError(ENGINE_ERRORS.notInitialized, 'Feature pipeline not ready');
        return this.pipeline;
      },
      gpu: (fn) => this.gpu(fn),
      hidden: () => this.hiddenFlag,
      log: (l, m, d) => this.log(l, m, d),
      emitLive: (u) => this.emitLive(u),
      storageError: (e) => this.onStorageError(e),
      memoryBudgetMB: () => this.memoryBudgetMB,
      memoryLimit: (msg) => this.emitError(namedError('CaptureMemoryLimitError', msg, 'out-of-memory'), false, 'out-of-memory'),
    });
    live.persistState = () => this.persistState();
    return live;
  }

  private onStorageError(e: unknown): void {
    const err = storageWriteError(e);
    if (!this.storageErr) this.storageErr = err;
    this.log('error', `storage write failed: ${err.message}`);
    this.emitError(err, false, 'storage');
  }

  private stateRecord(): EngineStateRecord {
    const extra: Partial<EngineStateRecord> = { hasSparse: !!this.model, hasDense: this.hasDense, hasMesh: this.hasMesh };
    if (this.live) return this.live.stateRecord(extra);
    // no live session (loadSession, or released for the dense stage): keep the stored capture
    // state (θ, verified camera←device mapping, time offset, counters) — a generic record here
    // would reset them in IndexedDB and a later refine would align with θ = 0
    if (this.savedState) return { ...this.savedState, ...extra, updatedAt: Date.now() };
    return {
      version: 1,
      mode: this.options.mode,
      quality: this.quality,
      hfovPriorDeg: this.options.hfovPriorDeg ?? null,
      thetaSession: null,
      cameraTheta: null,
      cameraThetaVerified: false,
      resolvedMode: null,
      nextKeyframeId: 0,
      frameCounter: 0,
      imuChunks: 0,
      captureFinished: false,
      vocabTrainedOn: null,
      timeOffsetMs: null,
      updatedAt: Date.now(),
      hasSparse: !!this.model,
      hasDense: this.hasDense,
      hasMesh: this.hasMesh,
    };
  }

  /**
   * Write `engine/state` (coalesced: while a write is in flight, later requests collapse into one
   * write of the then-current record — the live capture asks twice per keyframe).
   */
  private persistState(): void {
    if (!this.store || this.disposed) return;
    if (this.stateWrite) {
      this.stateDirty = true;
      return;
    }
    const store = this.store;
    this.stateWrite = store
      .putValue('engine/state', this.stateRecord())
      .catch((e: unknown) => this.onStorageError(e))
      .finally(() => {
        this.stateWrite = null;
        if (this.stateDirty && !this.disposed && this.store === store) {
          this.stateDirty = false;
          this.persistState();
        }
      });
  }

  /**
   * Create the feature pipeline (WebGPU preferred: pipelines compiled up front + self-test, CPU
   * fallback), open the session store when persisting, and restore the persisted capture when
   * `resume` is set.
   */
  async init(): Promise<InitResult> {
    if (this.disposed) throw namedError(ENGINE_ERRORS.crashed, 'Engine disposed');
    const o = this.options;
    await this.openStore(o.sessionId);
    try {
      await this.store!.ensureSession({ mode: o.mode, quality: o.quality });
    } catch (e) {
      this.log('warn', `could not create the session record: ${(e as Error)?.message ?? e}`);
      this.emitError(e instanceof Error ? e : new Error(String(e)), false, 'storage');
    }
    const pipeline = await this.ensurePipeline();
    this.live = this.createLive();
    let resumed = 0;
    if (o.resume) {
      const st = await this.store!.getValue<EngineStateRecord>('engine/state').catch(() => null);
      if (st) {
        try {
          resumed = await this.live.restore(st);
          await this.loadResults(st);
          this.log('info', `resumed session ${o.sessionId}: ${resumed} keyframes`);
        } catch (e) {
          // unreadable state: continue the capture from scratch (new keyframe ids never clash)
          this.log('error', `could not resume session ${o.sessionId}: ${(e as Error)?.message ?? e}`);
          this.emitError(e instanceof Error ? e : new Error(String(e)), false, 'storage');
          this.live.dispose();
          this.live = this.createLive();
          this.live.state.nextKeyframeId = Math.max(0, st.nextKeyframeId ?? 0);
          this.live.state.imuChunks = Math.max(0, st.imuChunks ?? 0);
        }
      }
    }
    this.initialized = true;
    this.persistState();
    let adapter: string | undefined;
    if (pipeline.gpu) {
      try {
        const ctx = await getGPU();
        if (ctx) adapter = [ctx.info.vendor, ctx.info.architecture, ctx.info.description].filter(Boolean).join(' ') || 'WebGPU';
        this.applyDeviceBudget(ctx);
      } catch {
        /* ignore */
      }
    } else await this.ensureDeviceBudget();
    const res: InitResult = { gpu: pipeline.gpu };
    if (adapter) res.adapter = adapter;
    if (!pipeline.gpu && pipeline.fallbackReason) res.fallbackReason = pipeline.fallbackReason;
    return res;
  }

  /** Recreate GPU resources after a device loss. Resolves false when no GPU can be obtained. */
  async recoverGPU(): Promise<boolean> {
    if (this.disposed || this.options.preferGPU === false) return false;
    const ctx = await getGPU().catch(() => null);
    if (!ctx) return false;
    this.holder.reset();
    if (!this.ownsPipeline && this.pipeline) return true; // injected pipeline: its owner recovers it
    if (this.pipeline && !this.pipeline.gpu && this.config.featureBackend === 'cpu') return true; // CPU features: nothing to recreate
    const old = this.pipeline;
    try {
      const { createFeaturePipeline } = await import('../features/extractor');
      // under the GPU lock (and only while visible): pipeline compilation + self-tests must not
      // overlap another command stream (dense PatchMatch recreating itself after the same loss)
      this.pipeline = await this.gpu(() => createFeaturePipeline({ preferGPU: true, sift: this.siftOptions(), context: ctx }));
      this.ownsPipeline = true;
      this.pipelineCfgKey = pipelineKey(this.quality, true, this.config);
      try {
        old?.destroy();
      } catch {
        /* ignore */
      }
      this.log('warn', `GPU recovered (feature pipeline on ${this.pipeline.gpu ? 'WebGPU' : 'CPU'})`);
      return this.pipeline.gpu;
    } catch (e) {
      this.log('error', `GPU recovery failed: ${(e as Error)?.message ?? e}`);
      return false;
    }
  }

  /** True while the feature pipeline runs on WebGPU. */
  get usesGPU(): boolean {
    return !!this.pipeline?.gpu;
  }

  // ------------------------------------------------------------------------------------------
  // capture

  private assertLive(): LiveSession {
    if (this.disposed) throw namedError(ENGINE_ERRORS.crashed, 'Engine disposed');
    if (!this.live || !this.initialized) throw namedError(ENGINE_ERRORS.notInitialized, 'Engine not initialised (call init first)');
    return this.live;
  }

  /** Analyse one frame (the engine owns and closes `f.bitmap`). */
  async processFrame(f: FrameInput): Promise<LiveUpdate> {
    let live: LiveSession;
    try {
      live = this.assertLive();
      if (this.longOp) throw namedError(ENGINE_ERRORS.busy, `Engine busy (${this.longOp})`);
      if (this.storageErr) {
        const e = this.storageErr;
        this.storageErr = null;
        throw e;
      }
    } catch (e) {
      closeSource(f.bitmap);
      throw e;
    }
    let input: DecodedFrameInput;
    let convertMs: number | null = null;
    try {
      const t0 = performance.now();
      const c = await frameToBitmap(f);
      if (c !== f) convertMs = performance.now() - t0;
      input = c;
    } catch (e) {
      closeSource(f.bitmap);
      throw e;
    }
    try {
      const upd = await live.processFrame(input);
      if (convertMs !== null) upd.timings.convert = convertMs;
      return upd;
    } finally {
      this.armIdleRelease();
    }
  }

  /** CPU SIFT keeps a large scratch arena (≈ 110 MB at 1600 px): release it when frames stop coming. */
  private armIdleRelease(): void {
    if (this.pipeline?.gpu) return;
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = setTimeout(() => {
      this.idleTimer = null;
      if (!this.disposed && !this.longOp) releaseSiftCPUMemory();
    }, 4000);
    (this.idleTimer as { unref?: () => void }).unref?.();
  }

  /** Wait for background keyframe mapping and pending writes. */
  async flush(): Promise<void> {
    if (this.live) await this.live.flush();
    await this.settleState();
  }

  /** Wait until the coalesced `engine/state` writes are done. */
  private async settleState(): Promise<void> {
    while (this.stateWrite) await this.stateWrite;
  }

  private async longRun<T>(name: string, fn: () => Promise<T>): Promise<T> {
    if (this.disposed) throw namedError(ENGINE_ERRORS.crashed, 'Engine disposed');
    if (this.longOp) throw namedError(ENGINE_ERRORS.busy, `Engine busy (${this.longOp})`);
    this.longOp = name;
    try {
      return await fn();
    } finally {
      this.longOp = null;
    }
  }

  /** End the capture: finish the keyframe mapping queue, persist, release capture-time memory. */
  finishCapture(progress?: ProgressCallback, cancel?: CancelToken): Promise<void> {
    return this.longRun('finishCapture', async () => {
      const live = this.assertLive();
      live.state.captureFinished = true;
      // a best-of-window frame still held would have become a keyframe: keep it (mapped by the flush)
      live.commitHeldKeyframe();
      const pending = live.pendingMapping;
      progress?.({ stage: 'matching', progress: 0, message: pending ? `mapping ${pending} keyframes` : 'finishing capture' });
      const timer = setInterval(() => {
        progress?.({ stage: 'matching', progress: -1, message: `mapping ${live.pendingMapping} keyframes` });
      }, 1000);
      try {
        await live.flush();
      } finally {
        clearInterval(timer);
      }
      if (cancel?.cancelled) throw namedError(ENGINE_ERRORS.cancelled, 'finishCapture cancelled');
      // capture-time scratch: CPU SIFT arena, GPU SIFT pyramids + matcher cache (pipelines stay compiled)
      releaseSiftCPUMemory();
      releasePipelineCaches(this.pipeline);
      this.persistState();
      await this.settleState();
      this.logMemory('capture');
      progress?.({ stage: 'matching', progress: 0, message: `${live.state.order.length} keyframes` });
    });
  }

  // ------------------------------------------------------------------------------------------
  // refine / dense / mesh

  /** Live session for post-capture work (restored from the store after a reload). */
  private async ensureLiveForRefine(): Promise<LiveSession> {
    if (this.live && this.initialized) return this.live;
    if (!this.store) throw namedError(ENGINE_ERRORS.notInitialized, 'No session (call init or loadSession first)');
    await this.ensurePipeline();
    const st = await this.store.getValue<EngineStateRecord>('engine/state');
    if (!st) throw new Error('This session has no capture data to refine');
    const live = this.createLive();
    await live.restore(st);
    this.live = live;
    this.savedState = null;
    this.initialized = true;
    return live;
  }

  /** Offline refinement → SparseResult (cameras carry `t` and `frameIndex`). */
  refine(progress?: ProgressCallback, cancel?: CancelToken): Promise<SparseResultEx> {
    return this.longRun('refine', async () => {
      const live = await this.ensureLiveForRefine();
      await this.ensurePipeline();
      // refine without finishCapture: the held best-of-window frame is kept as in finishCapture
      if (!live.state.captureFinished) live.commitHeldKeyframe();
      live.state.captureFinished = true;
      await live.flush();
      const t0 = now();
      // CPU helper workers for the pair verification and vocabulary indexing (refine only)
      const pool = CpuPool.create(this.config.cpuWorkers ?? defaultCpuWorkers());
      let out: Awaited<ReturnType<typeof runRefine>>;
      try {
        out = await runRefine({ live, config: this.config, store: this.store!, log: (l, m, d) => this.log(l, m, d), pool }, progress, cancel);
      } finally {
        pool?.dispose();
      }
      releaseSiftCPUMemory();
      this.model = out.model;
      this.sparse = out.result;
      out.result.stats.refineMs = Math.round(now() - t0);
      this.stats = { ...out.result.stats };
      // later stages are invalidated
      this.denseOut = null;
      this.denseCloud = null;
      this.denseMeta = null;
      this.meshModel = null;
      this.hasDense = false;
      this.hasMesh = false;
      await this.persistSparse();
      out.result.stats.memoryMB = this.logMemory('refine');
      return out.result;
    });
  }

  private async persistSparse(): Promise<void> {
    if (!this.store || !this.model || !this.sparse) return;
    const meta: SparseMetaRecord = {
      cameras: this.sparse.cameras.map((c) => ({
        id: c.id,
        pose: { R: Array.from(c.pose.R), t: Array.from(c.pose.t) },
        intrinsics: { ...c.intrinsics },
        t: c.t ?? NaN,
        frameIndex: c.frameIndex ?? -1,
      })),
      metric: this.sparse.metric,
      stats: this.sparse.stats,
      rollingShutter: this.model.rollingShutter ?? null,
    };
    try {
      await this.store.putBinary('sparse/model', packSparseModel(this.model));
      await this.store.putValue('sparse/meta', meta);
      await this.store.deleteBinary('dense/points').catch(() => undefined);
      await this.store.deleteBinary('mesh').catch(() => undefined);
      // depth maps of an earlier model (poses changed): no meshing / resuming from them, and
      // free their storage (up to ≈ 2 GB of float16 maps at `max`)
      await this.store.putValue('dense/meta', null);
      await this.store.putValue('dense/progress', null);
      await this.clearDepthMaps(this.store);
      await this.store.putValue('engine/state', this.stateRecord());
    } catch (e) {
      this.onStorageError(e);
    }
  }

  /** Delete every stored depth map (PatchMatch passes `pm<n>/`, fusion-filtered `depth/`). */
  private async clearDepthMaps(store: EngineStore): Promise<void> {
    const keys = (await store.listBinaryKeys().catch(() => [] as string[])).filter((k) => /^(pm\d+|depth)\//.test(k));
    for (const k of keys) await store.deleteBinary(k).catch(() => undefined);
  }

  /**
   * Where depth maps go: the session store when it persists; else a scratch session in a
   * separate IndexedDB database ('sfm-scratch', never listed by the app) so that hundreds of
   * depth maps (25 MB each at 1280 px) do not have to stay in memory; else memory (Node).
   */
  private async depthBackend(): Promise<EngineStore | null> {
    if (this.store?.persistent) return this.store;
    if (this.scratch !== undefined) return this.scratch;
    this.scratch = null;
    if (!SessionEngineStore.isSupported()) return null;
    try {
      const id = `scratch-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
      const sc = await SessionEngineStore.open(id, SCRATCH_DB);
      await sc.purgeOtherSessions(60 * 60 * 1000).catch(() => undefined);
      await sc.ensureSession({ mode: this.options.mode, quality: this.quality, name: 'scratch' });
      this.scratch = sc;
    } catch (e) {
      this.log('warn', `no scratch storage for depth maps (kept in memory): ${(e as Error)?.message ?? e}`);
    }
    return this.scratch;
  }

  private resolvedMode(): ResolvedMode | 'auto' {
    if (this.options.mode !== 'auto') return this.options.mode;
    return this.live?.state.resolvedMode ?? this.loadedMode ?? 'auto';
  }

  private async ensureModel(): Promise<SparseModel> {
    if (this.model) return this.model;
    if (this.store) {
      const buf = await this.store.getBinary('sparse/model');
      if (buf) {
        this.model = unpackSparseModel(buf);
        const meta = await this.store.getValue<SparseMetaRecord>('sparse/meta').catch(() => null);
        this.model.rollingShutter = meta?.rollingShutter ?? null;
        return this.model;
      }
    }
    throw new Error('No sparse model yet (run refine first)');
  }

  /** Dense point cloud (world frame). */
  densify(opts?: DenseOptions, progress?: ProgressCallback, cancel?: CancelToken): Promise<PointCloud> {
    return this.longRun('densify', async () => {
      const model = await this.ensureModel();
      if (!this.store) throw namedError(ENGINE_ERRORS.notInitialized, 'No session');
      await this.ensureDeviceBudget();
      // up direction of the views for the sky test (read before the capture state is released)
      const sky = await this.denseSkyInputs();
      // the dense stage needs the memory more than the capture state does
      this.releaseCaptureMemory();
      const resident = sumBreakdown(this.memoryReport().parts);
      // This run deletes / overwrites the stored depth maps (other settings ⇒ cleared, fusion ⇒
      // rewritten). Until it completes, mesh() must not pair them with the views of the previous
      // run (a cancelled re-densify left it with an emptied or mixed map store): forget them; the
      // previous dense CLOUD stays valid until it is replaced.
      this.denseOut = null;
      this.denseMeta = null;
      await this.store.putValue('dense/meta', null).catch((e: unknown) => this.onStorageError(e));
      const out = await runDensify(
        {
          store: this.store,
          depthBackend: await this.depthBackend(),
          model,
          quality: this.quality,
          mode: this.resolvedMode(),
          config: this.config,
          holder: this.holder,
          waitVisible: () => this.waitVisible(),
          gpuLock: (fn) => this.gpu(fn),
          log: (l, m, d) => this.log(l, m, d),
          memoryBudgetMB: this.memoryBudgetMB,
          residentBytes: resident,
          gravityAligned: sky.gravityAligned,
          gravityCam: sky.gravityCam,
        },
        opts ?? {},
        progress,
        cancel,
      );
      this.denseOut = out;
      this.denseCloud = out.cloud;
      this.denseMeta = out.meta;
      this.meshModel = null;
      this.hasDense = true;
      this.hasMesh = false;
      Object.assign(this.stats, out.stats);
      if (this.sparse) Object.assign(this.sparse.stats, out.stats);
      try {
        await this.store.putBinary('dense/points', serializePointCloud(out.cloud, { float16: true }));
        await this.store.putValue('dense/meta', out.meta);
        await this.store.deleteBinary('mesh').catch(() => undefined);
        await this.store.putValue('engine/state', this.stateRecord());
      } catch (e) {
        this.onStorageError(e);
      }
      out.stats.memoryMB = this.logMemory('densify');
      return out.cloud;
    });
  }

  /**
   * What the dense sky test needs to orient each view (`DenseDeps.gravityAligned` / `gravityCam`):
   * whether the sparse model is gravity-aligned (then each pose gives the up direction), else the
   * keyframes' gravity in the canonical camera frame — from the live session (current camera←device
   * mapping) or, after a reload, the stored keyframe records re-rotated with the stored mapping
   * (records keep the mapping of their creation time).
   */
  private async denseSkyInputs(): Promise<{ gravityAligned: boolean; gravityCam: Map<number, ArrayLike<number>> }> {
    const gravityCam = new Map<number, ArrayLike<number>>();
    let metric = this.sparse?.metric ?? null;
    if (!metric && this.store) metric = (await this.store.getValue<SparseMetaRecord>('sparse/meta').catch(() => null))?.metric ?? null;
    if (metric?.gravityAligned) return { gravityAligned: true, gravityCam };
    if (this.live) {
      for (const r of this.live.state.records.values()) if (r.info.gravityCam) gravityCam.set(r.id, r.info.gravityCam);
    } else if (this.store) {
      const st = await this.store.getValue<EngineStateRecord>('engine/state').catch(() => null);
      const theta = st ? (st.cameraTheta ?? st.thetaSession) : null;
      const Rcd = theta === 0 || theta === 90 || theta === 180 || theta === 270 ? cameraFromDevice(theta) : null;
      for (const r of await this.store.getFrameRecords().catch(() => [] as KeyframeRecord[])) {
        const g = Rcd && r.gravityDevice ? rotateVec(Rcd, r.gravityDevice) : r.info?.gravityCam;
        if (g) gravityCam.set(r.id, g);
      }
    }
    return { gravityAligned: false, gravityCam };
  }

  /**
   * Before the dense stage: free what capture / refine no longer need — the feature pipeline's GPU
   * caches and, when the session persists, the whole live session (features, pairs, graph, live
   * model, vocabulary: ≈ 1–2 MB per keyframe at `balanced`). Everything is in the store:
   * `refine` restores it (ensureLiveForRefine), exports read features from the store.
   */
  private releaseCaptureMemory(): void {
    releaseSiftCPUMemory();
    releasePipelineCaches(this.pipeline);
    const live = this.live;
    if (!live || !this.store?.persistent || !live.state.captureFinished || !this.model || live.pendingMapping > 0) return;
    const before = sumBreakdown(live.memoryBreakdown());
    this.savedState = live.stateRecord({ hasSparse: !!this.model, hasDense: this.hasDense, hasMesh: this.hasMesh });
    this.loadedMode = live.state.resolvedMode ?? this.loadedMode;
    live.dispose();
    this.live = null;
    this.log('info', `capture state released for the dense stage (≈ ${Math.round(before / MB)} MB; reloaded from storage if refine runs again)`);
  }

  /** TSDF mesh of the dense depth maps (densify runs first when needed). */
  mesh(opts?: MeshOptions, progress?: ProgressCallback, cancel?: CancelToken): Promise<TriangleMesh> {
    return this.longRun('mesh', async () => {
      const model = await this.ensureModel();
      if (!this.store) throw namedError(ENGINE_ERRORS.notInitialized, 'No session');
      let views = this.denseOut?.views ?? null;
      let filtered: EngineDepthStore | null = this.denseOut?.filtered ?? null;
      if (!views || !filtered) {
        // after a reload: persisted filtered depth maps + rebuilt colour views
        const meta = this.denseMeta ?? (await this.store.getValue<DenseMetaRecord>('dense/meta'));
        const fs = new EngineDepthStore(this.store, 'depth', 64 << 20);
        const ids = meta ? await fs.scan() : [];
        if (meta && ids.length >= 2 && this.store.persistent) {
          progress?.({ stage: 'meshing', progress: 0, message: 'loading depth maps' });
          views = await rebuildColourViews(this.store, model, meta);
          filtered = fs;
        }
      }
      if (!views || !filtered) {
        // no depth maps: run the dense stage first (outside longRun's lock: inline)
        this.longOp = null;
        try {
          await this.densify(undefined, progress, cancel);
        } finally {
          this.longOp = 'mesh';
        }
        views = this.denseOut!.views;
        filtered = this.denseOut!.filtered;
      }
      const { mesh, stats } = await runMesh(
        { quality: this.quality, mode: this.resolvedMode(), config: this.config, model, memoryBudgetMB: this.memoryBudgetMB },
        views,
        filtered,
        opts ?? {},
        progress,
        cancel,
      );
      this.meshModel = mesh;
      this.hasMesh = true;
      stats.memoryMB = this.logMemory('mesh');
      Object.assign(this.stats, stats);
      if (this.sparse) Object.assign(this.sparse.stats, stats);
      try {
        await this.store.putBinary('mesh', serializeTriangleMesh(mesh, { float16: true }));
        await this.store.putValue('engine/state', this.stateRecord());
      } catch (e) {
        this.onStorageError(e);
      }
      return mesh;
    });
  }

  // ------------------------------------------------------------------------------------------
  // export

  /**
   * Features of a keyframe for the exports, with keypoints consistent with the model's poses: refine
   * corrects the keypoints for rolling shutter IN MEMORY, the store keeps the originals — after a
   * reload / respawn they are corrected again from the persisted `SparseModel.rollingShutter`.
   */
  private async featuresOf(id: number): Promise<FeatureSet | null> {
    const st = this.live?.state;
    const mem = st?.db.features.get(id);
    if (st && mem) {
      const kp = modelKeypoints(this.model, id, originalKeypoints(st, id) ?? mem.keypoints, { st, keypoints: mem.keypoints });
      return kp === mem.keypoints ? mem : { ...mem, keypoints: kp };
    }
    const buf = await this.store?.getBinary(`kf/${id}/features`);
    if (!buf) return null;
    const fs = deserializeFeatureSet(buf);
    const kp = modelKeypoints(this.model, id, fs.keypoints, null);
    return kp === fs.keypoints ? fs : { ...fs, keypoints: kp };
  }

  /**
   * Drop the in-memory copy of a PERSISTED dense cloud / mesh (e.g. after the worker transferred
   * it to the page); it is reloaded from the store when needed (export). No-op without persistence.
   */
  releaseResultMemory(kind: 'dense' | 'mesh'): void {
    if (!this.store?.persistent) return;
    if (kind === 'dense') this.denseCloud = null;
    else this.meshModel = null;
  }

  private async reloadResults(dense = true, mesh = true): Promise<void> {
    if (!this.store) return;
    if (dense && this.hasDense && !this.denseCloud) {
      const b = await this.store.getBinary('dense/points').catch(() => null);
      if (b) this.denseCloud = deserializePointCloud(b);
    }
    if (mesh && this.hasMesh && !this.meshModel) {
      const b = await this.store.getBinary('mesh').catch(() => null);
      if (b) this.meshModel = deserializeTriangleMesh(b);
    }
  }

  /** Export the current model as a Blob. */
  async exportModel(format: ExportFormat): Promise<Blob> {
    if (this.disposed) throw namedError(ENGINE_ERRORS.crashed, 'Engine disposed');
    if (!this.model && this.store) await this.ensureModel().catch(() => null);
    // load only what this format needs (a 4 M-point cloud is ≈ 140 MB, a mesh similar), and drop
    // persisted results again afterwards: the page holds its own copy
    const needMesh = format === 'ply-mesh' || format === 'glb';
    const needDense = format === 'ply-points' || (format === 'glb' && !this.hasMesh);
    const hadDense = !!this.denseCloud;
    const hadMesh = !!this.meshModel;
    await this.reloadResults(needDense, needMesh);
    try {
      return await buildExport(format, {
        name: this.sessionName,
        sparse: this.sparse?.points ?? (this.model ? modelPointCloud(this.model) : null),
        model: this.model,
        dense: needDense ? this.denseCloud : null,
        mesh: needMesh ? this.meshModel : null,
        store: this.store,
        features: (id) => this.featuresOf(id),
      });
    } finally {
      if (this.store?.persistent) {
        if (!hadDense) this.denseCloud = null;
        if (!hadMesh) this.meshModel = null;
      }
    }
  }

  // ------------------------------------------------------------------------------------------
  // sessions

  /** Load stored results (sparse / dense / mesh) of the current store. */
  private async loadResults(st: EngineStateRecord | null): Promise<void> {
    const store = this.store!;
    // A damaged / unreadable stored result is treated as missing (logged): the scan still opens
    // and can be processed again from its photos, instead of becoming unopenable.
    const decode = <T>(what: string, f: () => T): T | null => {
      try {
        return f();
      } catch (e) {
        this.log('warn', `stored ${what} is damaged and was ignored: ${(e as Error)?.message ?? e}`);
        return null;
      }
    };
    const mb = await store.getBinary('sparse/model').catch(() => null);
    const meta = await store.getValue<SparseMetaRecord>('sparse/meta').catch(() => null);
    const loaded =
      mb && meta
        ? decode('sparse model', () => {
            const model = unpackSparseModel(mb);
            model.rollingShutter = meta.rollingShutter ?? null;
            const sparse: SparseResultEx = {
              points: modelPointCloud(model),
              cameras: meta.cameras.map((c) => ({
                id: c.id,
                pose: { R: Float64Array.from(c.pose.R), t: Float64Array.from(c.pose.t) },
                intrinsics: c.intrinsics,
                t: c.t,
                frameIndex: c.frameIndex,
              })),
              metric: meta.metric,
              stats: meta.stats,
            };
            return { model, sparse };
          })
        : null;
    if (loaded) {
      this.model = loaded.model;
      this.sparse = loaded.sparse;
      this.stats = { ...meta!.stats };
    }
    const db = await store.getBinary('dense/points').catch(() => null);
    const dense = db && this.model ? decode('dense point cloud', () => deserializePointCloud(db)) : null;
    if (dense) {
      this.denseCloud = dense;
      this.denseMeta = await store.getValue<DenseMetaRecord>('dense/meta').catch(() => null);
      this.hasDense = true;
    }
    const mb2 = await store.getBinary('mesh').catch(() => null);
    const mesh = mb2 && this.model ? decode('mesh', () => deserializeTriangleMesh(mb2)) : null;
    if (mesh) {
      this.meshModel = mesh;
      this.hasMesh = true;
    }
    void st;
  }

  /**
   * Open a stored session (may be called without init): restores the stored models so that
   * viewing, exporting, densifying and meshing work after a reload.
   */
  async loadSession(id: string, options: Partial<EngineOptions> = {}): Promise<SessionSummary> {
    if (this.disposed) throw namedError(ENGINE_ERRORS.crashed, 'Engine disposed');
    if (this.store && this.store.sessionId !== id) {
      if (this.ownsStore) this.store.close();
      this.store = null;
      this.live?.dispose();
      this.live = null;
      this.initialized = false;
      this.model = null;
      this.sparse = null;
      this.denseOut = null;
      this.denseCloud = null;
      this.denseMeta = null;
      this.meshModel = null;
      this.hasDense = false;
      this.hasMesh = false;
      // nothing of the previous session may leak into this one (stats in the summary, the
      // resolved capture mode of the dense / mesh stages, the FOV prior of a later refine)
      this.stats = {};
      this.loadedMode = null;
      this.savedState = null;
      this.options.hfovPriorDeg = undefined;
    }
    Object.assign(this.options, options, { sessionId: id });
    if (options.persist === undefined && !this.config.store) this.options.persist = true;
    await this.openStore(id);
    const meta = await this.store!.getSessionMeta();
    if (!meta) throw namedError('SessionNotFoundError', `Session not found: ${id}`);
    const st = await this.store!.getValue<EngineStateRecord>('engine/state').catch(() => null);
    if (st) {
      // values of a damaged / foreign record are ignored (defaults stay)
      if (!options.mode && (st.mode === 'object' || st.mode === 'scene' || st.mode === 'auto')) this.options.mode = st.mode;
      if (!options.quality && (st.quality === 'fast' || st.quality === 'balanced' || st.quality === 'max')) this.options.quality = st.quality;
      if (this.options.hfovPriorDeg === undefined && typeof st.hfovPriorDeg === 'number' && st.hfovPriorDeg > 0 && st.hfovPriorDeg < 180) this.options.hfovPriorDeg = st.hfovPriorDeg;
      this.loadedMode = st.resolvedMode === 'object' || st.resolvedMode === 'scene' ? st.resolvedMode : null;
      if (!this.live) this.savedState = st;
    } else {
      if (!options.mode) this.options.mode = meta.mode;
      const q = meta.stats?.quality;
      if (!options.quality && (q === 'fast' || q === 'balanced' || q === 'max')) this.options.quality = q;
    }
    this.sessionName = meta.name;
    if (!this.model) await this.loadResults(st);
    else await this.reloadResults();
    const status = this.hasMesh ? 'meshed' : this.hasDense ? 'dense' : this.sparse ? 'sparse' : st?.captureFinished ? 'captured' : meta.status;
    // a densify interrupted by a kill / cancel can resume from its stored depth maps
    const prog = this.store!.persistent ? await this.store!.getValue<{ pass?: number; quality?: string }>('dense/progress').catch(() => null) : null;
    const extra: Record<string, number | string> = {};
    if (prog && typeof prog.pass === 'number') {
      extra.denseResumablePass = prog.pass;
      // the interrupted run's preset: resuming needs the same one (another one starts over)
      if (prog.quality === 'fast' || prog.quality === 'balanced' || prog.quality === 'max') extra.denseResumableQuality = prog.quality;
    }
    return {
      id,
      name: meta.name,
      mode: this.options.mode,
      quality: this.options.quality ?? null,
      status,
      numFrames: meta.numFrames,
      createdAt: meta.createdAt,
      updatedAt: meta.updatedAt,
      sparse: this.sparse,
      dense: this.denseCloud,
      mesh: this.meshModel,
      stats: { ...(meta.stats ?? {}), ...this.stats, ...extra },
    };
  }

  // ------------------------------------------------------------------------------------------
  // accessors (harness / tests)

  /** Live capture session (null before init). */
  get liveSession(): LiveSession | null {
    return this.live;
  }

  get sparseModel(): SparseModel | null {
    return this.model;
  }

  get sparseResult(): SparseResultEx | null {
    return this.sparse;
  }

  get denseCloudResult(): PointCloud | null {
    return this.denseCloud;
  }

  get meshResult(): TriangleMesh | null {
    return this.meshModel;
  }

  get engineStore(): EngineStore | null {
    return this.store;
  }

  /** Hand the feature pipeline to a successor engine (worker re-init); it is not destroyed by dispose(). */
  releasePipeline(): { pipeline: FeaturePipeline; key: string } | null {
    const p = this.pipeline;
    if (!p) return null;
    this.pipeline = null;
    this.ownsPipeline = false;
    // the configuration it was CREATED with: loadSession may have changed the quality since
    return { pipeline: p, key: this.pipelineCfgKey };
  }

  dispose(): void {
    if (this.disposed) return;
    // a coalesced state write still pending: issue it now (IndexedDB completes transactions
    // created before close())
    if (this.stateDirty && this.store) {
      this.stateDirty = false;
      this.store.putValue('engine/state', this.stateRecord()).catch(() => undefined);
    }
    this.disposed = true;
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.live?.dispose();
    this.setHidden(false);
    try {
      if (this.ownsPipeline) this.pipeline?.destroy();
    } catch {
      /* ignore */
    }
    this.pipeline = null;
    this.holder.destroy();
    if (this.ownsStore) this.store?.close();
    if (this.scratch) {
      const sc = this.scratch;
      void sc.deleteSession().finally(() => sc.close());
    }
    this.listeners.live.clear();
    this.listeners.log.clear();
    this.listeners.error.clear();
  }
}

/**
 * A failed session-store write as the app sees it: quota errors keep their name (the app says
 * "Storage is full"); anything else (iOS UnknownError, InvalidStateError on a closed connection,
 * an aborted transaction…) becomes a StorageWriteError with code 'storage' and the original
 * name in the message — never mistaken for a full disk, but still pausing the capture.
 */
function storageWriteError(e: unknown): Error {
  const name = (e as { name?: string } | null)?.name;
  if (name === 'StorageQuotaError' || name === 'QuotaExceededError') return e as Error;
  const message = (e as { message?: string } | null)?.message ?? String(e);
  return namedError(ENGINE_ERRORS.storageWrite, name && name !== 'Error' ? `${name}: ${message}` : message, 'storage');
}

/**
 * A VideoFrame input (FrameInput.bitmap) as an ImageBitmap, downscaled to `maxSide`; the frame is
 * closed (it holds a camera buffer). Other inputs are returned unchanged.
 */
export async function frameToBitmap(f: FrameInput): Promise<DecodedFrameInput> {
  const VF = (globalThis as { VideoFrame?: { new (...a: never[]): VideoFrame } }).VideoFrame;
  if (!VF || !(f.bitmap instanceof VF)) return f as DecodedFrameInput;
  const frame = f.bitmap;
  try {
    const w = frame.displayWidth;
    const h = frame.displayHeight;
    const ms = f.maxSide;
    const s = ms && ms > 0 ? Math.min(1, ms / Math.max(w, h)) : 1;
    const bitmap =
      s < 1
        ? await createImageBitmap(frame, { resizeWidth: Math.max(1, Math.round(w * s)), resizeHeight: Math.max(1, Math.round(h * s)), resizeQuality: 'medium' })
        : await createImageBitmap(frame);
    return { ...f, bitmap };
  } finally {
    frame.close();
  }
}
