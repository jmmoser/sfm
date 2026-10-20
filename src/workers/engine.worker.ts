/**
 * Engine worker (module worker): dispatches the RPC requests of ./protocol.ts to a
 * PhotogrammetryEngine (src/engine) and forwards its events.
 *
 *  - `init` creates a new engine (the previous one is disposed; its feature pipeline is reused
 *    when the SIFT configuration matches, so a new scan does not recompile the GPU pipelines).
 *    `loadSession` works without `init` (an engine is created from the stored session).
 *  - Long operations (finishCapture / refine / densify / mesh / splat / export / loadSession)
 *    report `progress` events with their `requestId` (throttled; `detail` passed through) and are
 *    cancelled by `cancel` (target = request id, default: the running long operation).
 *  - `splatModel`: the stored splat model, decoded into fresh arrays and transferred.
 *  - Engine `live` events (background keyframe mapping) and `log` / non-fatal `error` events are
 *    forwarded. Frame responses are NOT repeated as `live` events.
 *  - Results are posted with transfer lists (collectTransferables). Results the engine keeps
 *    (sparse model, dense cloud, mesh) are cloned, except persisted dense clouds / meshes, which
 *    are transferred and dropped from the engine's memory (reloaded from IndexedDB when needed).
 *  - `visibility`: while hidden the engine submits no GPU work (frames are skipped, long
 *    operations pause between GPU batches).
 *  - GPU device loss: the engine tries to recreate its GPU resources; when no adapter can be
 *    obtained any more (GPU process gone, ios-webgpu.md §5) a fatal `{type:'error', fatal:true,
 *    reason:'gpu-lost'}` is posted and the client respawns the worker.
 *  - Extension: `init` / `loadSession` options may carry `engineConfig` (EngineConfig tuning,
 *    harness / tests only).
 */
import { PhotogrammetryEngine, pipelineKey } from '../engine/engine';
import type { EngineConfig } from '../engine/types';
import type { FeaturePipeline } from '../features/extractor';
import { onDeviceLost } from '../gpu/device';
import type { CancelToken, ProgressEvent } from '../core/types';
import {
  ENGINE_ERRORS,
  collectTransferables,
  isEngineRequest,
  namedError,
  serializeError,
  type DenseOptions,
  type EngineEvent,
  type EngineOptions,
  type EngineRequest,
  type EngineResponse,
  type ExportFormat,
  type FrameInput,
  type LogLevel,
  type MeshOptions,
  type SplatOptions,
  type WorkerToMainMessage,
} from './protocol';

interface WorkerScope {
  onmessage: ((ev: MessageEvent<unknown>) => void) | null;
  postMessage(msg: WorkerToMainMessage, transfer?: Transferable[]): void;
  addEventListener(type: string, cb: (ev: Event) => void): void;
}

const scope = self as unknown as WorkerScope;

type InitParams = EngineOptions & { engineConfig?: EngineConfig };

let engine: PhotogrammetryEngine | null = null;
let engineHadGPU = false;
let hidden = false;
let spare: { pipeline: FeaturePipeline; key: string } | null = null;
/** Cancel tokens of running long operations by request id. */
const tokens = new Map<number, { cancelled: boolean }>();
let currentLong: number | null = null;

function post(msg: WorkerToMainMessage, transfer: Transferable[] = []): void {
  try {
    scope.postMessage(msg, transfer);
  } catch (e) {
    // a transfer list problem must not lose the message: retry as a clone
    try {
      scope.postMessage(msg);
    } catch {
      console.error('[engine.worker] postMessage failed', e);
    }
  }
}

function postEvent(ev: EngineEvent, transfer?: Transferable[]): void {
  post(ev, transfer);
}

function log(level: LogLevel, message: string, data?: unknown): void {
  let d: unknown = data;
  if (d instanceof Error) d = serializeError(d);
  try {
    structuredClone(d);
  } catch {
    d = String(d);
  }
  postEvent({ type: 'log', level, message, data: d });
}

function attach(e: PhotogrammetryEngine): void {
  e.on('live', (u) => postEvent({ type: 'live', update: u }, collectTransferables(u)));
  e.on('log', (level, message, data) => log(level, message, data));
  e.on('error', (err, fatal, reason) => postEvent({ type: 'error', error: serializeError(err), fatal, reason }));
  e.setHidden(hidden);
}

function splitOptions(p: InitParams | Partial<InitParams> | undefined): { options: Partial<EngineOptions>; config: EngineConfig } {
  const { engineConfig, ...options } = (p ?? {}) as InitParams;
  return { options, config: engineConfig ?? {} };
}

function newEngine(options: EngineOptions, config: EngineConfig): PhotogrammetryEngine {
  let cfg = config;
  if (engine) {
    const rel = engine.releasePipeline();
    engine.dispose();
    engine = null;
    if (rel) {
      if (spare && spare.pipeline !== rel.pipeline) destroyPipeline(spare.pipeline);
      spare = rel;
    }
  }
  const key = pipelineKey(options.quality, options.preferGPU !== false, config);
  if (spare && spare.key === key && !config.pipeline) {
    cfg = { ...config, pipeline: spare.pipeline };
  } else if (spare) {
    destroyPipeline(spare.pipeline);
    spare = null;
  }
  const e = new PhotogrammetryEngine(options, cfg);
  attach(e);
  engine = e;
  return e;
}

function destroyPipeline(p: FeaturePipeline): void {
  try {
    p.destroy();
  } catch {
    /* ignore */
  }
}

/** Progress callback posting throttled `progress` events for request `id`. */
function progressFor(id: number): (p: ProgressEvent) => void {
  let lastT = 0;
  let lastStage = '';
  return (p) => {
    const t = performance.now();
    if (p.stage === lastStage && t - lastT < 100 && p.progress !== 1 && p.progress !== 0) return;
    lastT = t;
    lastStage = p.stage;
    const ev: EngineEvent = { type: 'progress', requestId: id, stage: p.stage, progress: p.progress };
    if (p.message !== undefined) ev.message = p.message;
    if (p.detail) ev.detail = { ...p.detail };
    postEvent(ev);
  };
}

function requireEngine(): PhotogrammetryEngine {
  if (!engine) throw namedError(ENGINE_ERRORS.notInitialized, 'Engine not initialised (send init or loadSession first)');
  return engine;
}

async function longOp<T>(id: number, fn: (progress: (p: ProgressEvent) => void, cancel: CancelToken) => Promise<T>): Promise<T> {
  const token = { cancelled: false };
  tokens.set(id, token);
  currentLong = id;
  try {
    return await fn(progressFor(id), token);
  } finally {
    tokens.delete(id);
    if (currentLong === id) currentLong = null;
  }
}

/** Run a request; returns the result and whether its large buffers may be transferred. */
async function dispatch(req: EngineRequest): Promise<{ result: unknown; transfer: boolean }> {
  switch (req.method) {
    case 'init': {
      const { options, config } = splitOptions(req.params as InitParams);
      const e = newEngine(options as EngineOptions, config);
      const res = await e.init();
      engineHadGPU = res.gpu;
      return { result: res, transfer: false };
    }
    case 'frame': {
      const f = req.params as FrameInput;
      if (!engine) {
        try {
          (f?.bitmap as { close?: () => void } | undefined)?.close?.();
        } catch {
          /* ignore */
        }
        throw namedError(ENGINE_ERRORS.notInitialized, 'Engine not initialised (send init first)');
      }
      // the LiveUpdate's arrays are fresh copies: transfer them
      return { result: await engine.processFrame(f), transfer: true };
    }
    case 'finishCapture': {
      const e = requireEngine();
      await longOp(req.id, (p, c) => e.finishCapture(p, c));
      return { result: undefined, transfer: false };
    }
    case 'refine': {
      const e = requireEngine();
      // the engine keeps its sparse result (export / resume): clone
      return { result: await longOp(req.id, (p, c) => e.refine(p, c)), transfer: false };
    }
    case 'densify': {
      const e = requireEngine();
      const pc = await longOp(req.id, (p, c) => e.densify(req.params as DenseOptions | undefined, p, c));
      const persistent = !!e.engineStore?.persistent;
      if (persistent) e.releaseResultMemory('dense');
      return { result: pc, transfer: persistent };
    }
    case 'mesh': {
      const e = requireEngine();
      const m = await longOp(req.id, (p, c) => e.mesh(req.params as MeshOptions | undefined, p, c));
      const persistent = !!e.engineStore?.persistent;
      if (persistent) e.releaseResultMemory('mesh');
      return { result: m, transfer: persistent };
    }
    case 'splat': {
      const e = requireEngine();
      // small info object (the model stays in the store: splatModel)
      return { result: await longOp(req.id, (p, c) => e.splat(req.params as SplatOptions | undefined, p, c)), transfer: false };
    }
    case 'splatModel': {
      const e = requireEngine();
      // decoded from the store into fresh arrays: transfer them
      return { result: await e.splatModel(), transfer: true };
    }
    case 'export': {
      const e = requireEngine();
      const { format } = req.params as { format: ExportFormat };
      return { result: await longOp(req.id, () => e.exportModel(format)), transfer: false };
    }
    case 'loadSession': {
      const { id, options } = req.params as { id: string; options?: Partial<InitParams> };
      const { options: o, config } = splitOptions(options);
      let e = engine;
      if (!e || e.options.sessionId !== id) {
        e = newEngine({ mode: 'auto', quality: 'balanced', persist: true, ...o, sessionId: id } as EngineOptions, config);
      }
      const ee = e;
      const sum = await longOp(req.id, () => ee.loadSession(id, o));
      const persistent = !!ee.engineStore?.persistent;
      if (persistent) {
        if (sum.dense) ee.releaseResultMemory('dense');
        if (sum.mesh) ee.releaseResultMemory('mesh');
        // transfer only the dense / mesh buffers (the sparse model stays with the engine)
        return { result: sum, transfer: true };
      }
      return { result: sum, transfer: false };
    }
    case 'cancel': {
      const p = req.params as { target?: number } | undefined;
      const target = p?.target ?? currentLong;
      if (target !== null && target !== undefined) {
        const t = tokens.get(target);
        if (t) t.cancelled = true;
      }
      return { result: undefined, transfer: false };
    }
    case 'visibility': {
      hidden = !!(req.params as { hidden: boolean }).hidden;
      engine?.setHidden(hidden);
      return { result: undefined, transfer: false };
    }
    case 'dispose': {
      for (const t of tokens.values()) t.cancelled = true;
      if (engine) {
        const rel = engine.releasePipeline();
        engine.dispose();
        engine = null;
        if (rel) destroyPipeline(rel.pipeline);
      }
      if (spare) destroyPipeline(spare.pipeline);
      spare = null;
      return { result: undefined, transfer: false };
    }
    default:
      throw namedError(ENGINE_ERRORS.notImplemented, `Unknown method ${(req as { method: string }).method}`);
  }
}

/** Transfer list: everything for `transfer`, else nothing (structured clone). */
function transfersOf(req: EngineRequest, result: unknown, transfer: boolean): Transferable[] {
  if (!transfer || result === undefined || result === null) return [];
  if (req.method === 'loadSession') {
    const s = result as { dense?: unknown; mesh?: unknown };
    return [...collectTransferables(s.dense ?? null), ...collectTransferables(s.mesh ?? null)];
  }
  return collectTransferables(result);
}

async function handle(req: EngineRequest): Promise<void> {
  let res: EngineResponse;
  let transfer: Transferable[] = [];
  try {
    const { result, transfer: tr } = await dispatch(req);
    res = { id: req.id, ok: true, result: result as never };
    transfer = transfersOf(req, result, tr);
  } catch (e) {
    const err = serializeError(e);
    if ((e as Error)?.name === 'CancelledError') err.name = ENGINE_ERRORS.cancelled;
    res = { id: req.id, ok: false, error: err };
    if (err.code === 'gpu-lost') {
      postEvent({ type: 'error', error: err, fatal: true, reason: 'gpu-lost' });
    }
  }
  post(res, transfer);
}

scope.onmessage = (ev: MessageEvent<unknown>): void => {
  const req = ev.data;
  if (!isEngineRequest(req)) return;
  void handle(req);
};

scope.addEventListener('unhandledrejection', (ev: Event) => {
  const reason = (ev as PromiseRejectionEvent).reason;
  log('error', `unhandled rejection in the engine worker: ${(reason as Error)?.message ?? reason}`);
});

// GPU device loss: try to recover in place; when no adapter can be obtained the worker is unusable.
onDeviceLost((info) => {
  if (info.reason === 'destroyed' || !engine || !engineHadGPU) return;
  const e = engine;
  log('warn', `GPU device lost (${info.reason}): ${info.message}`);
  void e
    .recoverGPU()
    .then((ok) => {
      if (ok || e !== engine) return;
      postEvent({
        type: 'error',
        fatal: true,
        reason: 'gpu-lost',
        error: { name: ENGINE_ERRORS.gpuLost, message: `GPU device lost and no adapter available (${info.message || info.reason})`, code: 'gpu-lost' },
      });
    })
    .catch(() =>
      postEvent({ type: 'error', fatal: true, reason: 'gpu-lost', error: { name: ENGINE_ERRORS.gpuLost, message: 'GPU device lost', code: 'gpu-lost' } }),
    );
});
