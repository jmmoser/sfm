/**
 * EngineClient: the PhotogrammetryEngine API (docs/ENGINE.md §2) over the engine worker
 * (src/workers/engine.worker.ts) with the typed RPC protocol of src/workers/protocol.ts.
 *
 *  - Request ids, transfer lists (ImageBitmaps / large typed arrays are transferred).
 *  - Progress events are routed to the request that caused them; `live` / `log` / `error`
 *    events go to subscribers.
 *  - Back-pressure: at most ONE frame in flight. `processFrame` resolves with the LiveUpdate;
 *    the caller (FramePump) sends the next frame only after that. A second concurrent call
 *    rejects with EngineBusyError (its bitmap is closed).
 *  - Timeouts: `init` (pipeline compilation can take a while on iOS), frames, short calls;
 *    long operations (refine/densify/mesh/export) use an INACTIVITY timeout reset by progress.
 *  - Fatal worker death (script failed to load, a `fatal` error event — e.g. the GPU process
 *    connection died, so `requestAdapter()` returns null in that realm —, a stuck init or
 *    repeated frame timeouts): pending requests reject with EngineCrashedError, the worker is
 *    terminated and RESPAWNED, and the last `init` / `loadSession` is replayed with
 *    `resume: true` (docs/research/ios-webgpu.md §5 recovery state machine). Respawns are
 *    capped (3 per 60 s); after repeated GPU failures the engine is re-initialised with
 *    `preferGPU: false` (CPU tier).
 *  - Page visibility is forwarded (`visibility` request): the worker cannot see
 *    `visibilitychange`, and iOS silently drops GPU work submitted while hidden.
 */
import type { PointCloud, ProgressCallback, ProgressEvent, TriangleMesh } from '../core/types';
import {
  ENGINE_ERRORS,
  collectTransferables,
  deserializeError,
  isEngineEvent,
  isEngineResponse,
  namedError,
  type DenseOptions,
  type EngineEvent,
  type EngineMethod,
  type EngineOptions,
  type EngineParams,
  type EngineRequest,
  type EngineResult,
  type ExportFormat,
  type FrameInput,
  type InitResult,
  type LiveUpdate,
  type LogMessage,
  type MeshOptions,
  type SessionSummary,
  type SparseResult,
} from '../workers/protocol';

// ---------------------------------------------------------------------------------------------
// EngineLike: what the UI needs (implemented by EngineClient and MockEngine)
// ---------------------------------------------------------------------------------------------

export type EngineState = 'idle' | 'starting' | 'ready' | 'respawning' | 'dead' | 'disposed';

export interface EngineStatus {
  state: EngineState;
  /** Number of respawns since construction. */
  respawns: number;
  /** Reason of the last state change (respawn cause, error message). */
  reason?: string;
  /** Result of the last successful init. */
  init?: InitResult;
}

export type AbortOutcome = 'none' | 'cancelled' | 'restarted';

/** EngineStatus.reason of a restart caused by abortLongOperation (not an engine failure). */
export const ABORT_REASON_PREFIX = 'aborted: ';

export interface EngineErrorInfo {
  error: Error;
  fatal: boolean;
  reason?: string;
}

export interface EngineLike {
  readonly kind: 'worker' | 'mock';
  readonly status: EngineStatus;
  /** True while a frame is being processed (back-pressure). */
  readonly frameInFlight: boolean;
  /** Start (or restart) an engine session. */
  init(options: EngineOptions): Promise<InitResult>;
  processFrame(frame: FrameInput): Promise<LiveUpdate>;
  finishCapture(): Promise<void>;
  refine(onProgress?: ProgressCallback): Promise<SparseResult>;
  densify(options?: DenseOptions, onProgress?: ProgressCallback): Promise<PointCloud>;
  mesh(options?: MeshOptions, onProgress?: ProgressCallback): Promise<TriangleMesh>;
  exportModel(format: ExportFormat): Promise<Blob>;
  loadSession(id: string, options?: Partial<EngineOptions>): Promise<SessionSummary>;
  /** Cancel the running long operation (it rejects with CancelledError). */
  cancel(): void;
  /**
   * Cancel the running long operation (finishCapture / refine / densify / mesh / export /
   * loadSession) and make sure it ENDS: when the engine has not settled it within `graceMs` (a
   * step that ignores cancel, a hung IndexedDB wait, a stuck worker), the request is rejected
   * locally with CancelledError and the engine is restarted (EngineClient: the worker is
   * respawned and the last init / loadSession replayed with `resume`, so the persisted capture
   * comes back and later calls do not queue behind the stuck one). Resolves with what happened:
   * 'none' (nothing running), 'cancelled' (the engine stopped it), 'restarted'.
   */
  abortLongOperation(opts?: { graceMs?: number }): Promise<AbortOutcome>;
  /** Page visibility (EngineClient forwards it automatically; MockEngine pauses timers). */
  setHidden(hidden: boolean): void;
  /** Every LiveUpdate: frame responses AND unsolicited `live` events. */
  onLive(cb: (u: LiveUpdate) => void): () => void;
  /** Every progress event (also those routed to a request's callback). */
  onProgress(cb: (p: ProgressEvent) => void): () => void;
  onLog(cb: (l: LogMessage) => void): () => void;
  onError(cb: (e: EngineErrorInfo) => void): () => void;
  onStatus(cb: (s: EngineStatus) => void): () => void;
  dispose(): void;
}

/** Tiny typed listener set. */
export class Emitter<T> {
  private readonly cbs = new Set<(v: T) => void>();
  on(cb: (v: T) => void): () => void {
    this.cbs.add(cb);
    return () => this.cbs.delete(cb);
  }
  emit(v: T): void {
    for (const cb of [...this.cbs]) {
      try {
        cb(v);
      } catch (e) {
        console.error('[engine] listener failed', e);
      }
    }
  }
  get size(): number {
    return this.cbs.size;
  }
  clear(): void {
    this.cbs.clear();
  }
}

// ---------------------------------------------------------------------------------------------
// EngineClient
// ---------------------------------------------------------------------------------------------

export interface EngineClientOptions {
  /** Worker factory (tests inject a fixture worker). Default: src/workers/engine.worker.ts. */
  createWorker?: () => Worker;
  /** Default 120 s (pipeline compilation blocks on Safari). */
  initTimeoutMs?: number;
  /** Per-frame timeout. Default 30 s. */
  frameTimeoutMs?: number;
  /** Short calls (finishCapture, cancel, visibility, dispose). Default 30 s. */
  callTimeoutMs?: number;
  /** Long operations: reject when no message for the request arrived for this long. Default 10 min. */
  inactivityTimeoutMs?: number;
  /** Respawn cap: at most this many respawns within `respawnWindowMs` (default 3 / 60 s). */
  maxRespawns?: number;
  respawnWindowMs?: number;
  /** After this many GPU-related failures re-init with preferGPU: false (default 2). */
  cpuFallbackAfter?: number;
  /** Consecutive frame timeouts that count as a stuck worker (default 2). */
  maxFrameTimeouts?: number;
  /** Replay the last init/loadSession after a respawn (default true). */
  autoReinit?: boolean;
  /** Forward document visibility to the worker (default true). */
  forwardVisibility?: boolean;
}

interface Pending {
  method: EngineMethod;
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
  timer: ReturnType<typeof setTimeout> | null;
  timeoutMs: number;
  inactivity: boolean;
  onProgress?: ProgressCallback;
  gen: number;
  fatalOnTimeout: boolean;
  frame: boolean;
}

type Setup = { kind: 'init'; options: EngineOptions } | { kind: 'load'; id: string; options?: Partial<EngineOptions> };

/**
 * Long operations: `cancel()` targets the running one, progress resets their inactivity timer.
 * finishCapture is one (it drains the keyframe mapping queue; the worker gives it a cancel token).
 */
const LONG: ReadonlySet<EngineMethod> = new Set<EngineMethod>(['finishCapture', 'refine', 'densify', 'mesh', 'export', 'loadSession']);
const FIRE_AND_FORGET: ReadonlySet<EngineMethod> = new Set<EngineMethod>(['cancel', 'visibility', 'dispose']);

function defaultWorker(): Worker {
  return new Worker(new URL('../workers/engine.worker.ts', import.meta.url), { type: 'module', name: 'engine' });
}

function closeBitmap(f: FrameInput | undefined): void {
  const b = f?.bitmap as { close?: () => void } | undefined;
  try {
    b?.close?.();
  } catch {
    /* detached */
  }
}

export class EngineClient implements EngineLike {
  readonly kind = 'worker' as const;
  private readonly opts: Required<Omit<EngineClientOptions, 'createWorker'>> & { createWorker: () => Worker };
  private worker: Worker | null = null;
  private gen = 0;
  /** A message was received from the current worker (it loaded successfully). */
  private alive = false;
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();
  private st: EngineStatus = { state: 'idle', respawns: 0 };
  private setup: Setup | null = null;
  private gate: Promise<void> = Promise.resolve();
  private respawnTimes: number[] = [];
  private gpuFailures = 0;
  private frameBusy = false;
  private frameTimeouts = 0;
  private longOp: number | null = null;
  private disposed = false;
  private hidden = false;
  /** Resolvers waiting for a request to settle (abortLongOperation). */
  private readonly settleWaiters = new Map<number, (() => void)[]>();
  private readonly live = new Emitter<LiveUpdate>();
  private readonly progressE = new Emitter<ProgressEvent>();
  private readonly logE = new Emitter<LogMessage>();
  private readonly errorE = new Emitter<{ error: Error; fatal: boolean; reason?: string }>();
  private readonly statusE = new Emitter<EngineStatus>();

  constructor(opts: EngineClientOptions = {}) {
    this.opts = {
      createWorker: opts.createWorker ?? defaultWorker,
      initTimeoutMs: opts.initTimeoutMs ?? 120_000,
      frameTimeoutMs: opts.frameTimeoutMs ?? 30_000,
      callTimeoutMs: opts.callTimeoutMs ?? 30_000,
      inactivityTimeoutMs: opts.inactivityTimeoutMs ?? 600_000,
      maxRespawns: opts.maxRespawns ?? 3,
      respawnWindowMs: opts.respawnWindowMs ?? 60_000,
      cpuFallbackAfter: opts.cpuFallbackAfter ?? 2,
      maxFrameTimeouts: opts.maxFrameTimeouts ?? 2,
      autoReinit: opts.autoReinit ?? true,
      forwardVisibility: opts.forwardVisibility ?? true,
    };
    if (this.opts.forwardVisibility && typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this.onVisibility);
      this.hidden = document.hidden;
    }
  }

  get status(): EngineStatus {
    return { ...this.st };
  }
  get frameInFlight(): boolean {
    return this.frameBusy;
  }
  /** Number of requests awaiting a response (tests / diagnostics). */
  get pendingCount(): number {
    return this.pending.size;
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

  // ---- API ------------------------------------------------------------------------------------

  async init(options: EngineOptions): Promise<InitResult> {
    this.assertUsable();
    this.setup = { kind: 'init', options: { ...options } };
    this.gpuFailures = 0;
    this.setState('starting');
    try {
      const res = await this.call('init', options, { timeoutMs: this.opts.initTimeoutMs, bypassGate: true, fatalOnTimeout: true });
      this.setState('ready', undefined, res);
      return res;
    } catch (e) {
      if (this.st.state === 'starting') this.setState('idle', (e as Error)?.message ?? String(e));
      throw e;
    }
  }

  async processFrame(frame: FrameInput): Promise<LiveUpdate> {
    if (this.frameBusy) {
      closeBitmap(frame);
      throw namedError(ENGINE_ERRORS.busy, 'A frame is already being processed');
    }
    this.frameBusy = true;
    try {
      const upd = await this.call('frame', frame, { timeoutMs: this.opts.frameTimeoutMs, frame: true });
      this.frameTimeouts = 0;
      this.live.emit(upd);
      return upd;
    } finally {
      this.frameBusy = false;
    }
  }

  async finishCapture(): Promise<void> {
    await this.call('finishCapture', undefined, { timeoutMs: this.opts.inactivityTimeoutMs, inactivity: true });
  }

  refine(onProgress?: ProgressCallback): Promise<SparseResult> {
    return this.call('refine', undefined, { onProgress });
  }

  densify(options?: DenseOptions, onProgress?: ProgressCallback): Promise<PointCloud> {
    return this.call('densify', options, { onProgress });
  }

  mesh(options?: MeshOptions, onProgress?: ProgressCallback): Promise<TriangleMesh> {
    return this.call('mesh', options, { onProgress });
  }

  exportModel(format: ExportFormat): Promise<Blob> {
    return this.call('export', { format });
  }

  async loadSession(id: string, options?: Partial<EngineOptions>): Promise<SessionSummary> {
    this.assertUsable();
    this.setup = { kind: 'load', id, options };
    this.setState('starting');
    try {
      const res = await this.call('loadSession', { id, options }, { bypassGate: true });
      this.setState('ready');
      return res;
    } catch (e) {
      if (this.st.state === 'starting') this.setState('idle', (e as Error)?.message ?? String(e));
      throw e;
    }
  }

  cancel(): void {
    const target = this.longOp;
    if (target === null || !this.worker) return;
    void this.call('cancel', { target }, { timeoutMs: this.opts.callTimeoutMs }).catch(() => undefined);
  }

  async abortLongOperation(opts: { graceMs?: number } = {}): Promise<AbortOutcome> {
    const id = this.longOp;
    const p = id !== null ? this.pending.get(id) : undefined;
    if (id === null || !p) return 'none';
    this.cancel();
    const graceMs = Math.max(0, opts.graceMs ?? 5000);
    const settled = await new Promise<boolean>((resolve) => {
      const timer = setTimeout(() => resolve(false), graceMs);
      const list = this.settleWaiters.get(id) ?? [];
      list.push(() => {
        clearTimeout(timer);
        resolve(true);
      });
      this.settleWaiters.set(id, list);
    });
    if (settled || this.disposed) return 'cancelled';
    // The engine did not react (synchronous step, a wait that ignores cancel, a hung worker): end
    // the request here and restart the worker so later calls do not queue behind it. The restart
    // replays the last init / loadSession with `resume` (the capture is persisted per keyframe).
    const stuck = this.settle(id);
    stuck?.reject(namedError(ENGINE_ERRORS.cancelled, `'${p.method}' did not stop within ${Math.round(graceMs / 1000)} s — the engine was restarted`));
    this.respawn(`${ABORT_REASON_PREFIX}'${p.method}' did not respond to cancel`, false, true);
    return 'restarted';
  }

  setHidden(hidden: boolean): void {
    if (hidden === this.hidden) return;
    this.hidden = hidden;
    // Pause (hidden) or restart (visible) the request timeouts — see armTimer.
    for (const [id, p] of this.pending) this.armTimer(id, p);
    if (!this.worker) return; // nothing to tell yet; the state is sent with the next spawn
    void this.call('visibility', { hidden }, { timeoutMs: this.opts.callTimeoutMs, bypassGate: true }).catch(() => undefined);
  }

  /** Terminate and respawn the worker now (e.g. from a "Restart engine" button). */
  restart(reason = 'manual restart'): void {
    this.assertUsable();
    this.respawn(reason, false, true);
  }

  dispose(): void {
    if (this.disposed) return;
    if (this.worker && this.alive) {
      try {
        const req: EngineRequest<'dispose'> = { id: this.nextId++, method: 'dispose', params: undefined };
        this.worker.postMessage(req);
      } catch {
        /* ignore */
      }
    }
    this.disposed = true;
    if (typeof document !== 'undefined') document.removeEventListener('visibilitychange', this.onVisibility);
    this.rejectAll(namedError(ENGINE_ERRORS.crashed, 'Engine disposed'));
    this.killWorker();
    this.setState('disposed');
    this.live.clear();
    this.progressE.clear();
    this.logE.clear();
    this.errorE.clear();
    this.statusE.clear();
  }

  // ---- RPC core -------------------------------------------------------------------------------

  private assertUsable(): void {
    if (this.disposed) throw namedError(ENGINE_ERRORS.crashed, 'Engine disposed');
  }

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const gen = ++this.gen;
    const w = this.opts.createWorker();
    this.alive = false;
    w.addEventListener('message', (ev: MessageEvent) => {
      if (gen === this.gen) this.onMessage(ev.data);
    });
    w.addEventListener('error', (ev: ErrorEvent) => {
      if (gen === this.gen) this.onWorkerError(ev);
    });
    w.addEventListener('messageerror', () => {
      if (gen !== this.gen) return;
      this.errorE.emit({ error: namedError('DataCloneError', 'Engine message could not be deserialised'), fatal: false });
    });
    this.worker = w;
    if (this.hidden) {
      // Tell a fresh worker that the page is hidden (no response expected to matter).
      const req: EngineRequest<'visibility'> = { id: this.nextId++, method: 'visibility', params: { hidden: true } };
      try {
        w.postMessage(req);
      } catch {
        /* ignore */
      }
    }
    return w;
  }

  private call<M extends EngineMethod>(
    method: M,
    params: EngineParams<M>,
    o: {
      timeoutMs?: number;
      inactivity?: boolean;
      onProgress?: ProgressCallback;
      bypassGate?: boolean;
      fatalOnTimeout?: boolean;
      frame?: boolean;
    } = {},
  ): Promise<EngineResult<M>> {
    if (this.disposed) {
      if (o.frame) closeBitmap(params as FrameInput);
      return Promise.reject(namedError(ENGINE_ERRORS.crashed, 'Engine disposed'));
    }
    if (this.st.state === 'dead') {
      if (o.frame) closeBitmap(params as FrameInput);
      return Promise.reject(namedError(ENGINE_ERRORS.crashed, `Engine stopped after repeated failures (${this.st.reason ?? 'unknown'})`));
    }
    const long = LONG.has(method);
    const timeoutMs = o.timeoutMs ?? (long ? this.opts.inactivityTimeoutMs : this.opts.callTimeoutMs);
    const inactivity = o.inactivity ?? long;
    const send = (): Promise<EngineResult<M>> =>
      new Promise<EngineResult<M>>((resolve, reject) => {
        if (this.disposed || this.st.state === 'dead') {
          if (o.frame) closeBitmap(params as FrameInput);
          reject(namedError(ENGINE_ERRORS.crashed, 'Engine unavailable'));
          return;
        }
        let w: Worker;
        try {
          w = this.ensureWorker();
        } catch (e) {
          if (o.frame) closeBitmap(params as FrameInput);
          reject(e instanceof Error ? e : new Error(String(e)));
          return;
        }
        const id = this.nextId++;
        const p: Pending = {
          method,
          resolve: resolve as (v: unknown) => void,
          reject,
          timer: null,
          timeoutMs,
          inactivity,
          onProgress: o.onProgress,
          gen: this.gen,
          fatalOnTimeout: !!o.fatalOnTimeout,
          frame: !!o.frame,
        };
        this.pending.set(id, p);
        if (long) this.longOp = id;
        this.armTimer(id, p);
        const req: EngineRequest<M> = { id, method, params };
        try {
          w.postMessage(req, collectTransferables(params));
        } catch (e) {
          this.settle(id);
          if (o.frame) closeBitmap(params as FrameInput);
          reject(e instanceof Error ? e : new Error(String(e)));
        }
      });
    if (o.bypassGate || FIRE_AND_FORGET.has(method)) return send();
    return this.gate.then(send);
  }

  /**
   * (Re)start the timeout of a pending request. No timer runs while the page is hidden: iOS
   * suspends the page and its workers (and drops GPU work), so time spent in the background
   * must not count — otherwise every request in flight "times out" the moment the user comes
   * back (a stuck-worker respawn, a failed refine). The full timeout restarts when visible.
   */
  private armTimer(id: number, p: Pending): void {
    if (p.timer) clearTimeout(p.timer);
    p.timer = null;
    if (this.hidden) return;
    if (!(p.timeoutMs > 0) || !Number.isFinite(p.timeoutMs)) return;
    p.timer = setTimeout(() => {
      if (!this.pending.has(id)) return;
      this.settle(id);
      p.reject(namedError(ENGINE_ERRORS.timeout, `Engine '${p.method}' timed out after ${Math.round(p.timeoutMs / 1000)} s`));
      if (p.fatalOnTimeout) this.respawn(`'${p.method}' timed out`, false);
      else if (p.frame && ++this.frameTimeouts >= this.opts.maxFrameTimeouts) {
        this.frameTimeouts = 0;
        this.respawn('frames time out (worker stuck)', false);
      }
    }, p.timeoutMs);
  }

  private settle(id: number): Pending | undefined {
    const p = this.pending.get(id);
    if (!p) return undefined;
    if (p.timer) clearTimeout(p.timer);
    this.pending.delete(id);
    if (this.longOp === id) this.longOp = null;
    const waiters = this.settleWaiters.get(id);
    if (waiters) {
      this.settleWaiters.delete(id);
      for (const w of waiters) w();
    }
    return p;
  }

  private onMessage(msg: unknown): void {
    this.alive = true;
    if (isEngineEvent(msg)) {
      this.onEvent(msg);
      return;
    }
    if (!isEngineResponse(msg)) return;
    const p = this.settle(msg.id);
    if (!p) return; // late response of a timed-out / fire-and-forget request
    if (msg.ok) p.resolve(msg.result);
    else p.reject(deserializeError(msg.error));
  }

  private onEvent(ev: EngineEvent): void {
    switch (ev.type) {
      case 'progress': {
        const pe: ProgressEvent = { stage: ev.stage, progress: ev.progress };
        if (ev.message !== undefined) pe.message = ev.message;
        const targetId = ev.requestId ?? this.longOp;
        const p = targetId !== null && targetId !== undefined ? this.pending.get(targetId) : undefined;
        if (p) {
          if (p.inactivity) this.armTimer(targetId as number, p);
          try {
            p.onProgress?.(pe);
          } catch (e) {
            console.error('[engine] progress callback failed', e);
          }
        }
        this.progressE.emit(pe);
        break;
      }
      case 'live':
        this.live.emit(ev.update);
        break;
      case 'log':
        this.logE.emit(ev);
        if (ev.level === 'error') console.error('[engine]', ev.message, ev.data ?? '');
        else if (ev.level === 'warn') console.warn('[engine]', ev.message, ev.data ?? '');
        break;
      case 'error': {
        const err = deserializeError(ev.error);
        this.errorE.emit({ error: err, fatal: ev.fatal, reason: ev.reason });
        if (ev.fatal) this.respawn(ev.reason ?? err.message, ev.reason === 'gpu-lost' || /gpu|device lost|adapter/i.test(err.message));
        break;
      }
    }
  }

  private onWorkerError(ev: ErrorEvent): void {
    const msg = ev.message || 'Engine worker error';
    ev.preventDefault?.();
    if (!this.alive) {
      // The module failed to load / evaluate: this worker will never answer.
      this.errorE.emit({ error: namedError(ENGINE_ERRORS.crashed, `Engine worker failed to start: ${msg}`), fatal: true, reason: 'startup' });
      this.respawn(`worker failed to start: ${msg}`, false);
    } else {
      // An uncaught exception inside the worker: it keeps running; the affected request times out.
      this.errorE.emit({ error: new Error(msg), fatal: false, reason: 'uncaught' });
    }
  }

  private rejectAll(err: Error): void {
    const ids = [...this.pending.keys()];
    for (const id of ids) {
      const p = this.settle(id);
      p?.reject(err);
    }
  }

  private killWorker(): void {
    const w = this.worker;
    this.worker = null;
    this.gen++;
    this.alive = false;
    try {
      w?.terminate();
    } catch {
      /* ignore */
    }
  }

  /**
   * Terminate the current worker, fail its pending requests and start a new one (replaying
   * the last setup). `gpu` marks GPU-related failures (CPU fallback after `cpuFallbackAfter`).
   */
  private respawn(reason: string, gpu: boolean, manual = false): void {
    if (this.disposed) return;
    const err = namedError(ENGINE_ERRORS.crashed, `Engine restarted: ${reason}`);
    this.rejectAll(err);
    this.frameBusy = false;
    this.killWorker();
    if (gpu) this.gpuFailures++;
    const now = Date.now();
    if (!manual) {
      this.respawnTimes = this.respawnTimes.filter((t) => now - t < this.opts.respawnWindowMs);
      if (this.respawnTimes.length >= this.opts.maxRespawns) {
        this.setState('dead', reason);
        return;
      }
      this.respawnTimes.push(now);
    }
    this.st = { ...this.st, respawns: this.st.respawns + 1 };
    const setup = this.setup;
    if (!setup || !this.opts.autoReinit) {
      this.setState('idle', reason);
      return;
    }
    this.setState('respawning', reason);
    const cpu = this.gpuFailures >= this.opts.cpuFallbackAfter;
    const replay = async (): Promise<void> => {
      try {
        if (setup.kind === 'init') {
          const options: EngineOptions = { ...setup.options, resume: !!setup.options.sessionId };
          if (cpu) options.preferGPU = false;
          const res = await this.call('init', options, { timeoutMs: this.opts.initTimeoutMs, bypassGate: true, fatalOnTimeout: true });
          this.setState('ready', `recovered after: ${reason}`, res);
        } else {
          const options: Partial<EngineOptions> = { ...(setup.options ?? {}) };
          if (cpu) options.preferGPU = false;
          await this.call('loadSession', { id: setup.id, options }, { bypassGate: true });
          this.setState('ready', `recovered after: ${reason}`);
        }
      } catch (e) {
        // A failed replay leaves the engine usable for a fresh init (unless it is dead).
        if (this.st.state === 'respawning') this.setState('idle', `re-initialisation failed: ${(e as Error)?.message ?? e}`);
      }
    };
    this.gate = replay();
  }

  private setState(state: EngineState, reason?: string, init?: InitResult): void {
    const next: EngineStatus = { state, respawns: this.st.respawns };
    if (reason !== undefined) next.reason = reason;
    const keepInit = init ?? (state === 'ready' || state === 'respawning' ? this.st.init : undefined);
    if (keepInit) next.init = keepInit;
    this.st = next;
    this.statusE.emit({ ...next });
  }

  private readonly onVisibility = (): void => {
    this.setHidden(document.hidden);
  };
}
