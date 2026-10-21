/**
 * Engine worker (Node; the module runs against a fake worker scope): the feature pipeline handed
 * from one engine to the next (also across `close`), and GPU device-loss recovery for engines
 * created by `init` and by `loadSession`. gpu/device and the WebGPU feature pipeline are mocked; a device loss clears the
 * fake pipelines' `gpu` the way FeaturePipelineImpl falls back to the CPU on device.lost. The splat
 * stage's preview events and `finishEarly` go through a stubbed `PhotogrammetryEngine.splat`.
 */
import { describe, expect, it, vi, type Mock } from 'vitest';
import type { CancelToken } from '../core/types';
import type { FeaturePipeline } from '../features/extractor';
import { MemoryEngineStore } from '../engine/persistence';
import type { EngineEvent, EngineResponse, WorkerToMainMessage } from './protocol';

type FakePipeline = FeaturePipeline & { gpu: boolean; destroy: Mock<() => void> };

const h = vi.hoisted(() => ({
  pipelines: [] as FakePipeline[],
  lost: [] as ((info: GPUDeviceLostInfo) => void)[],
  adapter: true,
  releaseSift: vi.fn<() => void>(),
}));

vi.mock('../features/sift-cpu', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../features/sift-cpu')>()),
  releaseSiftCPUMemory: h.releaseSift,
}));

vi.mock('../gpu/device', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../gpu/device')>()),
  hasWebGPU: () => true,
  getGPU: async () => (h.adapter ? { info: { vendor: 'test', architecture: '', description: '' } } : null),
  onDeviceLost: (cb: (info: GPUDeviceLostInfo) => void) => {
    h.lost.push(cb);
    return () => undefined;
  },
}));

vi.mock('../features/extractor', () => ({
  createFeaturePipeline: async () => {
    const unused = (): Promise<never> => Promise.reject(new Error('not used in this test'));
    const p: FakePipeline = { gpu: true, fallbackReason: null, extract: unused, extractFromBitmap: unused, match: unused, matchGuided: unused, destroy: vi.fn<() => void>() };
    h.pipelines.push(p);
    return p;
  },
}));

let onmessage: ((ev: MessageEvent<unknown>) => void) | null = null;
const waiting = new Map<number, (r: EngineResponse) => void>();
const events: EngineEvent[] = [];
/** The transfer list each event was posted with. */
const eventTransfers: Transferable[][] = [];
let nextId = 1;

/** A new worker module (fresh module state, as after a respawn). */
async function freshWorker(): Promise<void> {
  vi.resetModules();
  h.pipelines.length = 0;
  h.lost.length = 0;
  h.adapter = true;
  events.length = 0;
  eventTransfers.length = 0;
  const scope = {
    onmessage: null as ((ev: MessageEvent<unknown>) => void) | null,
    postMessage: (m: WorkerToMainMessage, transfer?: Transferable[]) => {
      if ('ok' in m) waiting.get(m.id)?.(m);
      else {
        events.push(m);
        eventTransfers.push(transfer ?? []);
      }
    },
    addEventListener: () => undefined,
  };
  vi.stubGlobal('self', scope);
  await import('./engine.worker');
  onmessage = scope.onmessage;
}

function call(method: string, params?: unknown): Promise<EngineResponse> {
  const id = nextId++;
  const res = new Promise<EngineResponse>((r) => waiting.set(id, r));
  onmessage!({ data: { id, method, params } } as MessageEvent<unknown>);
  return res;
}

function loseDevice(): void {
  for (const p of h.pipelines) p.gpu = false; // FeaturePipelineImpl's own device.lost reaction
  for (const cb of h.lost) cb({ reason: 'unknown', message: 'test loss' } as GPUDeviceLostInfo);
}

const fatal = (): EngineEvent[] => events.filter((e) => e.type === 'error' && e.fatal);

const SCAN = { mode: 'object', quality: 'fast', persist: false };

async function openStoredScan(): Promise<void> {
  const store = new MemoryEngineStore('saved');
  await store.ensureSession({ mode: 'object', quality: 'fast' });
  expect(await call('loadSession', { id: 'saved', options: { quality: 'fast', engineConfig: { store } } })).toMatchObject({ ok: true });
  // re-processing creates the GPU feature pipeline (then fails: the session has no capture data)
  expect(await call('refine')).toMatchObject({ ok: false });
  expect(h.pipelines).toHaveLength(1);
}

describe('engine worker: feature pipeline hand-over and GPU device loss', () => {
  it('a scan on the handed-over pipeline gets a new GPU pipeline after a device loss, and so does the next scan', async () => {
    await freshWorker();
    expect(await call('init', SCAN)).toMatchObject({ ok: true, result: { gpu: true } });
    expect(await call('init', SCAN)).toMatchObject({ ok: true, result: { gpu: true } });
    expect(h.pipelines).toHaveLength(1); // the second scan reuses the first one's pipeline
    loseDevice();
    await vi.waitFor(() => expect(h.pipelines).toHaveLength(2));
    await vi.waitFor(() => expect(h.pipelines[0].destroy).toHaveBeenCalledTimes(1));
    expect(await call('init', SCAN)).toMatchObject({ ok: true, result: { gpu: true } });
    expect(h.pipelines).toHaveLength(2);
    expect(fatal()).toHaveLength(0);
    await call('dispose');
    expect(h.pipelines[1].destroy).toHaveBeenCalledTimes(1);
  });

  it('a pipeline that fell back to the CPU is not handed to the next scan', async () => {
    await freshWorker();
    expect(await call('init', SCAN)).toMatchObject({ ok: true, result: { gpu: true } });
    h.pipelines[0].gpu = false; // a GPU error during the scan (no device loss)
    expect(await call('init', SCAN)).toMatchObject({ ok: true, result: { gpu: true } });
    expect(h.pipelines).toHaveLength(2);
    expect(h.pipelines[0].destroy).toHaveBeenCalledTimes(1);
    await call('dispose');
  });

  it('an engine created by loadSession recovers its pipeline after a device loss', async () => {
    await freshWorker();
    await openStoredScan();
    loseDevice();
    await vi.waitFor(() => expect(h.pipelines).toHaveLength(2));
    expect(await call('init', SCAN)).toMatchObject({ ok: true, result: { gpu: true } });
    expect(h.pipelines).toHaveLength(2);
    expect(fatal()).toHaveLength(0);
    await call('dispose');
  });

  it('an engine created by loadSession reports a fatal gpu-lost when no adapter is left', async () => {
    await freshWorker();
    await openStoredScan();
    h.adapter = false;
    loseDevice();
    await vi.waitFor(() => expect(fatal()).toHaveLength(1));
    expect(fatal()[0]).toMatchObject({ type: 'error', fatal: true, reason: 'gpu-lost' });
    await call('dispose');
  });
});

describe('engine worker: close', () => {
  const S1 = { ...SCAN, sessionId: 's1' };

  it('close lets go of that session only: its engine and CPU scratch are gone, its pipeline is destroyed by dispose', async () => {
    await freshWorker();
    expect(await call('init', S1)).toMatchObject({ ok: true });
    expect(await call('close', { sessionId: 'other' })).toMatchObject({ ok: true, result: false });
    expect(await call('finishCapture')).toMatchObject({ ok: true });
    h.releaseSift.mockClear();
    expect(await call('close', { sessionId: 's1' })).toMatchObject({ ok: true, result: true });
    expect(h.releaseSift).toHaveBeenCalled(); // the CPU SIFT scratch goes with the session
    expect(await call('finishCapture')).toMatchObject({ ok: false, error: { name: 'EngineNotInitializedError' } });
    expect(h.pipelines[0].destroy).not.toHaveBeenCalled();
    await call('dispose');
    expect(h.pipelines[0].destroy).toHaveBeenCalledTimes(1);
  });

  it('the next scan reuses the pipeline of a closed one, unless it lost its device meanwhile', async () => {
    await freshWorker();
    expect(await call('init', S1)).toMatchObject({ ok: true });
    expect(await call('close', { sessionId: 's1' })).toMatchObject({ result: true });
    expect(await call('init', S1)).toMatchObject({ ok: true, result: { gpu: true } });
    expect(h.pipelines).toHaveLength(1);
    expect(await call('close', { sessionId: 's1' })).toMatchObject({ result: true });
    loseDevice(); // no engine holds the pipeline: nothing recovers it
    expect(await call('init', S1)).toMatchObject({ ok: true, result: { gpu: true } });
    expect(h.pipelines).toHaveLength(2);
    expect(h.pipelines[0].destroy).toHaveBeenCalledTimes(1);
    expect(fatal()).toHaveLength(0);
    await call('dispose');
  });
});

describe('engine worker: splat previews and finishEarly', () => {
  it('image events are never throttled and their pixels are transferred; finishEarly reaches the running splat', async () => {
    await freshWorker();
    const { PhotogrammetryEngine } = await import('../engine/engine');
    let token: CancelToken | undefined;
    const images = [0, 1].map(() => ({ width: 16, height: 16, data: new Uint8ClampedArray(4 * 256) }));
    const spy = vi.spyOn(PhotogrammetryEngine.prototype, 'splat').mockImplementation(async (_opts, progress, cancel) => {
      token = cancel;
      progress?.({ stage: 'splat_training', progress: 0.5, detail: { iteration: 5 } });
      progress?.({ stage: 'splat_training', progress: 0.6, detail: { iteration: 6 } }); // < 100 ms later: throttled
      for (const image of images) progress?.({ stage: 'splat_preview', progress: 0.6, message: 'iteration 6', detail: { iteration: 6 }, image });
      while (!cancel?.finishEarly) await new Promise((r) => setTimeout(r, 5));
      return { count: 1, iterations: 6, ms: 1, shDegree: 0, bytes: 56, antialiased: true, stats: { splatFinishedEarly: 1 } };
    });
    expect(await call('init', SCAN)).toMatchObject({ ok: true });
    const res = call('splat', { preview: true });
    await vi.waitFor(() => expect(token).toBeDefined());
    expect(token).toMatchObject({ cancelled: false, finishEarly: false });
    expect(await call('finishEarly')).toMatchObject({ ok: true });
    expect(await res).toMatchObject({ ok: true, result: { iterations: 6, stats: { splatFinishedEarly: 1 } } });
    expect(token).toMatchObject({ cancelled: false, finishEarly: true });
    const progress = events.flatMap((e, i) => (e.type === 'progress' ? [{ e, transfer: eventTransfers[i] }] : []));
    expect(progress.map(({ e }) => e.stage)).toEqual(['splat_training', 'splat_preview', 'splat_preview']);
    for (let k = 0; k < 2; k++) {
      const { e, transfer } = progress[1 + k];
      expect(e).toMatchObject({ requestId: expect.any(Number), message: 'iteration 6', detail: { iteration: 6 }, image: images[k] });
      expect(transfer).toEqual([images[k].data.buffer]);
    }
    spy.mockRestore();
    await call('dispose');
  });
});
