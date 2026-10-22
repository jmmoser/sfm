/**
 * Engine worker (module worker): dispatches the RPC requests of ./protocol.ts to a
 * PhotogrammetryEngine (src/engine) and forwards its events.
 *
 *  - `init` creates a new engine (the previous one is disposed; its feature pipeline is handed
 *    over when the SIFT configuration matches and it has not fallen back to the CPU, so a new
 *    scan does not recompile the GPU pipelines).
 *    `loadSession` works without `init` (an engine is created from the stored session).
 *  - `close`: a session that is about to be deleted is let go of (its engine is disposed, so
 *    nothing more is written to it); the feature pipeline is kept for the next `init` /
 *    `loadSession` (handed over like on a re-init, unless it fell back to the CPU meanwhile).
 *  - Long operations (finishCapture / refine / densify / mesh / splat / export / loadSession)
 *    report `progress` events with their `requestId` (throttled; `detail` passed through; an event
 *    with an `image` — the splat previews — is never throttled and its buffer is transferred) and
 *    are cancelled by `cancel` (target = request id, default: the running long operation).
 *    `finishEarly` (same targeting) sets the token's `finishEarly`: the splat training stops at its
 *    current iteration and resolves with the model.
 *  - `splatModel`: the stored splat model, decoded into fresh arrays and transferred.
 *  - Engine `live` events (background keyframe mapping) and `log` / non-fatal `error` events are
 *    forwarded. Frame responses are NOT repeated as `live` events.
 *  - Results are posted with transfer lists (collectTransferables). Results the engine keeps
 *    (sparse model, dense cloud, mesh) are cloned, except dense clouds / meshes stored in
 *    IndexedDB, which are transferred and dropped from the engine's memory (reloaded when
 *    needed); one whose write failed stays with the engine (cloned).
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
import { releaseSiftCPUMemory } from '../features/sift-cpu';
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
/** Feature pipeline of an engine closed by `close`, for the next engine (no recompilation). */
let spare: { pipeline: FeaturePipeline; key: string; gpu: boolean } | null = null;
let hidden = false;
/** Cancel tokens of running long operations by request id. */
const tokens = new Map<number, { cancelled: boolean; finishEarly: boolean }>();
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
  closeEngine();
  const s = spare;
  spare = null;
  const e = new PhotogrammetryEngine(options, config);
  const key = pipelineKey(options.quality, options.preferGPU !== false, config);
  // the new engine owns the reused pipeline (it recreates it after a device loss); one that fell
  // back to the CPU since it was handed over (a device loss while no engine held it) is not reused
  if (s && s.key === key && !config.pipeline && s.pipeline.gpu === s.gpu) e.adoptPipeline(s.pipeline, s.key);
  else if (s) destroyPipeline(s.pipeline);
  attach(e);
  engine = e;
  return e;
}

/** Dispose the engine, keeping its feature pipeline (see releasePipeline) as the next engine's spare. */
function closeEngine(): void {
  if (!engine) return;
  const rel = engine.releasePipeline();
  engine.dispose();
  engine = null;
  if (!rel) return;
  if (spare && spare.pipeline !== rel.pipeline) destroyPipeline(spare.pipeline);
  spare = { ...rel, gpu: rel.pipeline.gpu };
}

function destroyPipeline(p: FeaturePipeline): void {
  try {
    p.destroy();
  } catch {
    /* ignore */
  }
}

/** Progress callback posting throttled `progress` events for request `id` (with an image: every one, transferred). */
function progressFor(id: number): (p: ProgressEvent) => void {
  let lastT = 0;
  let lastStage = '';
  return (p) => {
    if (p.image) {
      // the engine's fresh snapshot (src/engine/splat.ts): not throttled, its pixels transferred
      const ev: EngineEvent = { type: 'progress', requestId: id, stage: p.stage, progress: p.progress, image: p.image };
      if (p.message !== undefined) ev.message = p.message;
      if (p.detail) ev.detail = { ...p.detail };
      postEvent(ev, collectTransferables(p.image));
      return;
    }
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
  const token = { cancelled: false, finishEarly: false };
  tokens.set(id, token);
  currentLong = id;
  try {
    return await fn(progressFor(id), token);
  } finally {
    tokens.delete(id);
    if (currentLong === id) currentLong = null;
  }
}

/** Run a request; returns the result and whether its large buffers may be transferred (or which). */
async function dispatch(req: EngineRequest): Promise<{ result: unknown; transfer: boolean | Transferable[] }> {
  switch (req.method) {
    case 'init': {
      const { options, config } = splitOptions(req.params as InitParams);
      const e = newEngine(options as EngineOptions, config);
      return { result: await e.init(), transfer: false };
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
      return { result: pc, transfer: e.releaseResultMemory('dense') };
    }
    case 'mesh': {
      const e = requireEngine();
      const m = await longOp(req.id, (p, c) => e.mesh(req.params as MeshOptions | undefined, p, c));
      return { result: m, transfer: e.releaseResultMemory('mesh') };
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
      return { result: await longOp(req.id, (_p, c) => e.exportModel(format, c)), transfer: false };
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
      // transfer only the dense / mesh buffers the engine dropped (the sparse model stays with it)
      const transfer = [
        ...(sum.dense && ee.releaseResultMemory('dense') ? collectTransferables(sum.dense) : []),
        ...(sum.mesh && ee.releaseResultMemory('mesh') ? collectTransferables(sum.mesh) : []),
      ];
      return { result: sum, transfer };
    }
    case 'cancel':
    case 'finishEarly': {
      const p = req.params as { target?: number } | undefined;
      const target = p?.target ?? currentLong;
      if (target !== null && target !== undefined) {
        const t = tokens.get(target);
        if (t && req.method === 'cancel') t.cancelled = true;
        else if (t) t.finishEarly = true;
      }
      return { result: undefined, transfer: false };
    }
    case 'visibility': {
      hidden = !!(req.params as { hidden: boolean }).hidden;
      engine?.setHidden(hidden);
      return { result: undefined, transfer: false };
    }
    case 'close': {
      if (!engine || engine.options.sessionId !== (req.params as { sessionId: string }).sessionId) return { result: false, transfer: false };
      for (const t of tokens.values()) t.cancelled = true;
      closeEngine();
      // CPU SIFT scratch: the disposed engine's idle timer no longer frees it (extraction is
      // synchronous, so no frame is using it now)
      releaseSiftCPUMemory();
      return { result: true, transfer: false };
    }
    case 'dispose': {
      for (const t of tokens.values()) t.cancelled = true;
      // the engine owns its feature pipeline (created or adopted): dispose() destroys it
      engine?.dispose();
      engine = null;
      if (spare) destroyPipeline(spare.pipeline);
      spare = null;
      return { result: undefined, transfer: false };
    }
    default:
      throw namedError(ENGINE_ERRORS.notImplemented, `Unknown method ${(req as { method: string }).method}`);
  }
}

/** Transfer list: everything for `transfer: true`, the given list, else nothing (structured clone). */
function transfersOf(result: unknown, transfer: boolean | Transferable[]): Transferable[] {
  if (Array.isArray(transfer)) return transfer;
  if (!transfer || result === undefined || result === null) return [];
  return collectTransferables(result);
}

async function handle(req: EngineRequest): Promise<void> {
  let res: EngineResponse;
  let transfer: Transferable[] = [];
  try {
    const { result, transfer: tr } = await dispatch(req);
    res = { id: req.id, ok: true, result: result as never };
    transfer = transfersOf(result, tr);
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
  // hadGPU, not usesGPU: the feature pipeline may already have fallen back to the CPU on this loss
  if (info.reason === 'destroyed' || !engine || !engine.hadGPU) return;
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
