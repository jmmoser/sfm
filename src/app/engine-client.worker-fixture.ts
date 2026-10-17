/**
 * Test fixture (module worker) for engine-client.gpu.test.ts: a tiny fake engine speaking the
 * protocol of src/workers/protocol.ts. Its behaviour is selected by the worker `name`:
 *   'ok'         normal
 *   'hang-init'  never answers `init`
 *   'fatal'      answers `init`, then posts a fatal GPU-lost error event
 *   'slow-frame' never answers `frame`
 *   'slow-finish'  `finishCapture` runs (progress every 30 ms) until cancelled
 *   'stuck-finish' `finishCapture` never answers and ignores `cancel` (progress keeps coming)
 */
import {
  ENGINE_ERRORS,
  collectTransferables,
  isEngineRequest,
  serializeError,
  namedError,
  type EngineRequest,
  type FrameInput,
  type LiveUpdate,
  type SparseResult,
  type WorkerToMainMessage,
} from '../workers/protocol';

const scope = self as unknown as {
  name: string;
  onmessage: ((ev: MessageEvent<unknown>) => void) | null;
  postMessage(msg: WorkerToMainMessage, transfer?: Transferable[]): void;
};

const mode = scope.name || 'ok';
let hidden = false;
let initOptions: unknown = null;
let frames = 0;
const longOps = new Map<number, () => void>();

function ok(id: number, result: unknown): void {
  scope.postMessage({ id, ok: true, result } as WorkerToMainMessage, collectTransferables(result));
}
function fail(id: number, e: unknown): void {
  scope.postMessage({ id, ok: false, error: serializeError(e) });
}

scope.onmessage = (ev: MessageEvent<unknown>): void => {
  const req = ev.data;
  if (!isEngineRequest(req)) return;
  void handle(req).catch((e) => fail(req.id, e));
};

async function handle(req: EngineRequest): Promise<void> {
  switch (req.method) {
    case 'init': {
      initOptions = req.params;
      if (mode === 'hang-init') return;
      const p = req.params as { resume?: boolean; preferGPU?: boolean };
      ok(req.id, { gpu: p.preferGPU !== false, adapter: `fixture:${mode}:resume=${!!p.resume}:gpu=${p.preferGPU !== false}` });
      if (mode === 'fatal') {
        setTimeout(() => scope.postMessage({ type: 'error', fatal: true, reason: 'gpu-lost', error: { name: 'GPUDeviceLostError', message: 'device lost' } }), 30);
      }
      return;
    }
    case 'frame': {
      const f = req.params as FrameInput;
      const b = f.bitmap as ImageBitmap;
      const w = b.width;
      const h = b.height;
      b.close?.();
      if (mode === 'slow-frame') return;
      frames++;
      await new Promise((r) => setTimeout(r, 40));
      const upd: LiveUpdate = {
        frameIndex: frames - 1,
        decision: frames % 2 ? 'keyframe' : 'skip',
        guidance: 'good',
        tracking: 'tracking',
        numKeyframes: Math.ceil(frames / 2),
        numRegistered: Math.ceil(frames / 2),
        numPoints: frames * 100,
        overlay: new Float32Array([0.25, 0.5, 0.75, 0.5]),
        sparse: { positions: new Float32Array(3 * 1000).fill(1), colors: new Uint8Array(3 * 1000).fill(200) },
        timings: { width: w, height: h, imu: f.imu.length },
      };
      ok(req.id, upd);
      return;
    }
    case 'refine': {
      for (const p of [0, 0.5, 1]) {
        scope.postMessage({ type: 'progress', requestId: req.id, stage: 'matching', progress: p });
        await new Promise((r) => setTimeout(r, 20));
      }
      scope.postMessage({ type: 'log', level: 'info', message: 'refined' });
      const n = 5000;
      const res: SparseResult = {
        points: { count: n, positions: new Float32Array(3 * n), colors: new Uint8Array(3 * n) },
        cameras: [{ id: 0, pose: { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(1, 2, 3) }, intrinsics: { width: 4, height: 3, f: 5, cx: 1.5, cy: 1, k1: 0, k2: 0 } }],
        metric: { metric: true, source: 'imu', relativeUncertainty: 0.02, gravityAligned: true, northAligned: false, originLatLonAlt: null },
        stats: { images: 1 },
      };
      ok(req.id, res);
      return;
    }
    case 'densify': {
      // Runs until cancelled (progress every 30 ms).
      await new Promise<void>((resolve) => {
        let p = 0;
        const timer = setInterval(() => {
          p = Math.min(0.99, p + 0.05);
          scope.postMessage({ type: 'progress', requestId: req.id, stage: 'patchmatch', progress: p });
        }, 30);
        longOps.set(req.id, () => {
          clearInterval(timer);
          resolve();
        });
      });
      fail(req.id, namedError(ENGINE_ERRORS.cancelled, 'Cancelled'));
      return;
    }
    case 'mesh': {
      scope.postMessage({
        type: 'live',
        update: { frameIndex: -1, decision: 'skip', guidance: 'good', tracking: 'tracking', numKeyframes: 9, numRegistered: 9, numPoints: 9, timings: {} },
      });
      ok(req.id, { vertexCount: 3, triangleCount: 1, positions: new Float32Array(9), normals: new Float32Array(9), colors: new Uint8Array(9), indices: Uint32Array.of(0, 1, 2) });
      return;
    }
    case 'export':
      ok(req.id, new Blob(['ply\n'], { type: 'application/octet-stream' }));
      return;
    case 'loadSession':
      ok(req.id, {
        id: (req.params as { id: string }).id,
        name: 'fixture',
        mode: 'object',
        quality: 'fast',
        status: 'sparse',
        numFrames: 0,
        createdAt: 0,
        updatedAt: 0,
        sparse: null,
        dense: null,
        mesh: null,
        stats: { hidden: String(hidden), init: JSON.stringify(initOptions) },
      });
      return;
    case 'cancel': {
      const target = (req.params as { target?: number } | undefined)?.target;
      if (target !== undefined) longOps.get(target)?.();
      ok(req.id, undefined);
      return;
    }
    case 'visibility':
      hidden = (req.params as { hidden: boolean }).hidden;
      ok(req.id, undefined);
      return;
    case 'finishCapture': {
      if (mode !== 'slow-finish' && mode !== 'stuck-finish') {
        ok(req.id, undefined);
        return;
      }
      await new Promise<void>((resolve) => {
        const timer = setInterval(() => scope.postMessage({ type: 'progress', requestId: req.id, stage: 'matching', progress: -1, message: 'mapping 3 keyframes' }), 30);
        // A stuck worker never registers a cancel hook: the request hangs for good.
        if (mode === 'slow-finish')
          longOps.set(req.id, () => {
            clearInterval(timer);
            resolve();
          });
      });
      fail(req.id, namedError(ENGINE_ERRORS.cancelled, 'finishCapture cancelled'));
      return;
    }
    case 'dispose':
      ok(req.id, undefined);
      return;
  }
}
