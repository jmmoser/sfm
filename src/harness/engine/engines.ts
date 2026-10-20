/**
 * Engine back-ends of the harness, behind one interface:
 *  - 'worker': EngineClient (src/app/engine-client.ts) over the real engine worker
 *    (src/workers/engine.worker.ts) — exactly what the app uses;
 *  - 'inline': PhotogrammetryEngine from src/engine on the page's main thread (loaded lazily
 *    through import.meta.glob, so this file compiles and the other back-ends work while
 *    src/engine does not exist yet);
 *  - 'mock': the in-page MockEngine of the app (plumbing checks; results are meaningless).
 */
import type { CancelToken, PointCloud, ProgressCallback, ProgressEvent, TriangleMesh } from '../../core/types';
import { EngineClient, type EngineErrorInfo, type EngineStatus } from '../../app/engine-client';
import { MockEngine } from '../../app/mock-engine';
import type {
  DenseOptions,
  EngineOptions,
  ExportFormat,
  FrameInput,
  InitResult,
  LiveUpdate,
  LogMessage,
  MeshOptions,
  SparseResult,
  SplatOptions,
  SplatResult,
} from '../../workers/protocol';
import type { HarnessEngineKind, HarnessTimeouts } from './types';

export interface HarnessEngine {
  readonly kind: HarnessEngineKind;
  init(options: EngineOptions): Promise<InitResult>;
  processFrame(frame: FrameInput): Promise<LiveUpdate>;
  finishCapture(): Promise<void>;
  refine(onProgress?: ProgressCallback): Promise<SparseResult>;
  densify(options?: DenseOptions, onProgress?: ProgressCallback): Promise<PointCloud>;
  mesh(options?: MeshOptions, onProgress?: ProgressCallback): Promise<TriangleMesh>;
  splat(options?: SplatOptions, onProgress?: ProgressCallback): Promise<SplatResult>;
  exportModel(format: ExportFormat): Promise<Blob>;
  /** Cancel the running long operation. */
  cancel(): void;
  /** Every LiveUpdate (frame responses and unsolicited updates), when the back-end reports them. */
  onLive(cb: (u: LiveUpdate) => void): () => void;
  onLog(cb: (l: LogMessage) => void): () => void;
  onError(cb: (e: EngineErrorInfo) => void): () => void;
  onStatus(cb: (s: EngineStatus) => void): () => void;
  /** Progress events not routed to a request callback (e.g. during finishCapture). */
  onProgress(cb: (p: ProgressEvent) => void): () => void;
  dispose(): void;
}

export interface CreateEngineOptions {
  timeouts: Required<HarnessTimeouts>;
  mockSpeed: number;
}

const noop = (): (() => void) => () => undefined;

/** The PhotogrammetryEngine surface of docs/ENGINE.md §2 (what the inline adapter relies on). */
interface InlineEngineInstance {
  init(): Promise<InitResult | { gpu: boolean; adapter?: string }>;
  processFrame(f: FrameInput): Promise<LiveUpdate>;
  finishCapture(progress?: ProgressCallback, cancel?: CancelToken): Promise<void>;
  refine(progress?: ProgressCallback, cancel?: CancelToken): Promise<SparseResult>;
  densify(opts?: DenseOptions, progress?: ProgressCallback, cancel?: CancelToken): Promise<PointCloud>;
  mesh(opts?: MeshOptions, progress?: ProgressCallback, cancel?: CancelToken): Promise<TriangleMesh>;
  splat?(opts?: SplatOptions, progress?: ProgressCallback, cancel?: CancelToken): Promise<SplatResult>;
  exportModel(format: ExportFormat): Promise<Blob>;
  dispose(): void;
  /** Optional event hooks (either style; src/engine EngineEvents: live / log / error). */
  on?: (event: string, cb: (...args: never[]) => void) => unknown;
  onLog?: (cb: (l: LogMessage) => void) => () => void;
  onLive?: (cb: (u: LiveUpdate) => void) => () => void;
}

type InlineEngineCtor = new (opts: EngineOptions, config?: Record<string, unknown>) => InlineEngineInstance;

/** Lazily load PhotogrammetryEngine from src/engine (index.ts preferred, else engine.ts). */
async function loadInlineEngine(): Promise<InlineEngineCtor> {
  const loaders = import.meta.glob(['../../engine/index.ts', '../../engine/engine.ts']);
  const load = loaders['../../engine/index.ts'] ?? loaders['../../engine/engine.ts'];
  if (!load) throw new Error("engine 'inline': src/engine/index.ts (or engine.ts) does not exist yet");
  const mod = (await load()) as { PhotogrammetryEngine?: unknown; default?: unknown };
  const C = mod.PhotogrammetryEngine ?? mod.default;
  if (typeof C !== 'function') throw new Error("engine 'inline': src/engine does not export PhotogrammetryEngine");
  return C as InlineEngineCtor;
}

/** Adapter: PhotogrammetryEngine (constructor options + cancel tokens) → HarnessEngine. */
class InlineEngine implements HarnessEngine {
  readonly kind = 'inline' as const;
  private engine: InlineEngineInstance | null = null;
  private token: { cancelled: boolean } = { cancelled: false };
  private readonly logCbs = new Set<(l: LogMessage) => void>();
  private readonly liveCbs = new Set<(u: LiveUpdate) => void>();
  private readonly progressCbs = new Set<(p: ProgressEvent) => void>();
  private readonly errorCbs = new Set<(e: EngineErrorInfo) => void>();
  private unsubs: (() => void)[] = [];

  constructor(private readonly Ctor: InlineEngineCtor) {}

  async init(options: EngineOptions): Promise<InitResult> {
    this.dispose();
    // The worker takes EngineConfig as the `engineConfig` field of the init options; the class
    // takes it as its second constructor argument (src/engine/types.ts).
    const { engineConfig, ...opts } = options as EngineOptions & { engineConfig?: Record<string, unknown> };
    this.engine = new this.Ctor(opts, engineConfig);
    const e = this.engine;
    const sub = (u: unknown): void => {
      if (typeof u === 'function') this.unsubs.push(u as () => void);
    };
    if (typeof e.on === 'function') {
      const on = e.on.bind(e) as (event: string, cb: (...args: unknown[]) => void) => unknown;
      sub(on('live', (u) => this.liveCbs.forEach((cb) => cb(u as LiveUpdate))));
      sub(
        on('log', (level, message, data) =>
          this.logCbs.forEach((cb) => cb({ type: 'log', level: level as LogMessage['level'], message: String(message), data })),
        ),
      );
      sub(
        on('error', (err, fatal, reason) =>
          this.errorCbs.forEach((cb) => cb({ error: err instanceof Error ? err : new Error(String(err)), fatal: !!fatal, reason: reason as string | undefined })),
        ),
      );
    } else {
      if (typeof e.onLog === 'function') sub(e.onLog((l) => this.logCbs.forEach((cb) => cb(l))));
      if (typeof e.onLive === 'function') {
        sub(e.onLive((u) => this.liveCbs.forEach((cb) => cb(u))));
      }
    }
    return (await e.init()) as InitResult;
  }

  private get e(): InlineEngineInstance {
    if (!this.engine) throw new Error('init() first');
    return this.engine;
  }

  private progress(cb?: ProgressCallback): ProgressCallback {
    return (p) => {
      cb?.(p);
      this.progressCbs.forEach((c) => c(p));
    };
  }

  private fresh(): CancelToken {
    this.token = { cancelled: false };
    return this.token;
  }

  async processFrame(frame: FrameInput): Promise<LiveUpdate> {
    const u = await this.e.processFrame(frame);
    // Like EngineClient: every frame response is also a live update. Engine hooks (if any) only
    // report unsolicited updates (protocol: frame responses are not repeated as events).
    this.liveCbs.forEach((cb) => cb(u));
    return u;
  }
  finishCapture(): Promise<void> {
    // progress like the worker's finishCapture events (mapping queue drain)
    return this.e.finishCapture(this.progress(), this.fresh());
  }
  refine(onProgress?: ProgressCallback): Promise<SparseResult> {
    return this.e.refine(this.progress(onProgress), this.fresh());
  }
  densify(options?: DenseOptions, onProgress?: ProgressCallback): Promise<PointCloud> {
    return this.e.densify(options, this.progress(onProgress), this.fresh());
  }
  mesh(options?: MeshOptions, onProgress?: ProgressCallback): Promise<TriangleMesh> {
    return this.e.mesh(options, this.progress(onProgress), this.fresh());
  }
  splat(options?: SplatOptions, onProgress?: ProgressCallback): Promise<SplatResult> {
    if (!this.e.splat) return Promise.reject(new Error("engine 'inline': this src/engine has no splat()"));
    return this.e.splat(options, this.progress(onProgress), this.fresh());
  }
  exportModel(format: ExportFormat): Promise<Blob> {
    return this.e.exportModel(format);
  }
  cancel(): void {
    this.token.cancelled = true;
  }
  onLive(cb: (u: LiveUpdate) => void): () => void {
    this.liveCbs.add(cb);
    return () => this.liveCbs.delete(cb);
  }
  onLog(cb: (l: LogMessage) => void): () => void {
    this.logCbs.add(cb);
    return () => this.logCbs.delete(cb);
  }
  onProgress(cb: (p: ProgressEvent) => void): () => void {
    this.progressCbs.add(cb);
    return () => this.progressCbs.delete(cb);
  }
  onError(cb: (e: EngineErrorInfo) => void): () => void {
    this.errorCbs.add(cb);
    return () => this.errorCbs.delete(cb);
  }
  onStatus = noop;
  dispose(): void {
    for (const u of this.unsubs) {
      try {
        u();
      } catch {
        /* ignore */
      }
    }
    this.unsubs = [];
    this.engine?.dispose();
    this.engine = null;
  }
}

export async function createHarnessEngine(kind: HarnessEngineKind, o: CreateEngineOptions): Promise<HarnessEngine> {
  switch (kind) {
    case 'worker':
      return new EngineClient({
        initTimeoutMs: o.timeouts.initMs,
        frameTimeoutMs: o.timeouts.frameMs,
        callTimeoutMs: o.timeouts.callMs,
        inactivityTimeoutMs: o.timeouts.inactivityMs,
        // A headless page is always visible; never pause the engine's timers / GPU work.
        forwardVisibility: false,
      });
    case 'mock':
      // No IndexedDB lookups: the harness never calls loadSession.
      return new MockEngine({ speed: o.mockSpeed, seed: 7, sessions: async () => null });
    case 'inline':
      return new InlineEngine(await loadInlineEngine());
    default:
      throw new Error(`unknown engine kind '${String(kind)}' (worker | inline | mock)`);
  }
}
