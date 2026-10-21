/**
 * Feature pipeline lifecycle of the engine (Node): device-loss recovery of an adopted / injected /
 * created pipeline, hand-over to a successor engine, frames queued behind a recovery, and init() /
 * recoverGPU() overtaken by dispose(). gpu/device and the WebGPU feature pipeline are mocked:
 * `createFeaturePipeline` returns what the test supplies, and a fake GPU pipeline's `gpu` is
 * cleared the way FeaturePipelineImpl falls back to the CPU on device.lost.
 */
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import type { FeaturePipeline, FeaturePipelineOptions } from '../features/extractor';
import { ENGINE_ERRORS } from '../workers/protocol';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine, pipelineKey } from './engine';
import { MemoryEngineStore, SessionEngineStore } from './persistence';
import { frameInputs, makeCapture } from './testutil';

const h = vi.hoisted(() => ({
  create: null as null | ((opts: FeaturePipelineOptions) => Promise<FeaturePipeline>),
  calls: 0,
}));

vi.mock('../gpu/device', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../gpu/device')>()),
  hasWebGPU: () => true,
  getGPU: async () => ({ info: { vendor: 'test', architecture: '', description: '' } }),
}));

vi.mock('../features/extractor', () => ({
  createFeaturePipeline: (opts: FeaturePipelineOptions) => {
    h.calls++;
    return h.create!(opts);
  },
}));

type FakeGpuPipeline = FeaturePipeline & { gpu: boolean; destroy: Mock<() => void> };

function fakeGpuPipeline(gpu = true): FakeGpuPipeline {
  const unused = (): Promise<never> => Promise.reject(new Error('not used in this test'));
  return { gpu, fallbackReason: gpu ? null : 'GPU self-test failed', extract: unused, extractFromBitmap: unused, match: unused, matchGuided: unused, destroy: vi.fn<() => void>() };
}

function gate(): { wait: Promise<void>; open: () => void } {
  let open!: () => void;
  const wait = new Promise<void>((r) => (open = r));
  return { wait, open };
}

const OPTS = { mode: 'object', quality: 'fast', persist: false } as const;

beforeEach(() => {
  h.create = null;
  h.calls = 0;
  vi.restoreAllMocks();
});

describe('feature pipeline recovery and hand-over', () => {
  it('recreates an adopted pipeline (worker hand-over) that fell back on a device loss and destroys the old one', async () => {
    const p1 = fakeGpuPipeline();
    const p2 = fakeGpuPipeline();
    h.create = async () => p2;
    const e = new PhotogrammetryEngine({ ...OPTS });
    e.adoptPipeline(p1, pipelineKey('fast', true));
    expect((await e.init()).gpu).toBe(true);
    expect(h.calls).toBe(0);
    p1.gpu = false; // device.lost: FeaturePipelineImpl falls back to the CPU
    expect(await e.recoverGPU()).toBe(true);
    expect(h.calls).toBe(1);
    expect(e.usesGPU).toBe(true);
    expect(p1.destroy).toHaveBeenCalledTimes(1);
    expect(e.releasePipeline()?.pipeline).toBe(p2);
    e.dispose();
    expect(p2.destroy).not.toHaveBeenCalled();
  });

  it('leaves an injected pipeline (EngineConfig.pipeline) to its owner', async () => {
    const p = fakeGpuPipeline();
    const e = new PhotogrammetryEngine({ ...OPTS }, { pipeline: p });
    await e.init();
    p.gpu = false;
    expect(await e.recoverGPU()).toBe(true);
    expect(h.calls).toBe(0);
    e.dispose();
    expect(p.destroy).not.toHaveBeenCalled();
  });

  it('does not hand a pipeline that fell back to the CPU to a successor: dispose() destroys it', async () => {
    const p = fakeGpuPipeline();
    h.create = async () => p;
    const e = new PhotogrammetryEngine({ ...OPTS });
    expect((await e.init()).gpu).toBe(true);
    p.gpu = false; // GPU error / device loss
    expect(e.hadGPU).toBe(true);
    expect(e.releasePipeline()).toBeNull();
    e.dispose();
    expect(p.destroy).toHaveBeenCalledTimes(1);
  });

  it('still hands over a pipeline that never ran on the GPU (no recompile per scan)', async () => {
    const p = fakeGpuPipeline(false);
    h.create = async () => p;
    const e = new PhotogrammetryEngine({ ...OPTS });
    expect((await e.init()).gpu).toBe(false);
    expect(e.hadGPU).toBe(false);
    expect(e.releasePipeline()).toEqual({ pipeline: p, key: pipelineKey('fast', true) });
    e.dispose();
    expect(p.destroy).not.toHaveBeenCalled();
  });

  it('overlapping recoveries (two losses in a row) each destroy the pipeline they replace', async () => {
    const ps = [fakeGpuPipeline(), fakeGpuPipeline(), fakeGpuPipeline()];
    const g = gate();
    h.create = async () => {
      const i = h.calls - 1;
      if (i === 1) await g.wait;
      return ps[i];
    };
    const e = new PhotogrammetryEngine({ ...OPTS });
    await e.init();
    const first = e.recoverGPU();
    await vi.waitFor(() => expect(h.calls).toBe(2));
    const gpuJobs = vi.spyOn(e as unknown as { gpu: (fn: () => Promise<unknown>) => Promise<unknown> }, 'gpu');
    // a second loss while the first recovery compiles: its compile is queued behind the first one
    const second = e.recoverGPU();
    await vi.waitFor(() => expect(gpuJobs).toHaveBeenCalled());
    g.open();
    expect(await Promise.all([first, second])).toEqual([true, true]);
    expect(ps[0].destroy).toHaveBeenCalledTimes(1);
    expect(ps[1].destroy).toHaveBeenCalledTimes(1);
    expect(ps[2].destroy).not.toHaveBeenCalled();
    expect(e.releasePipeline()?.pipeline).toBe(ps[2]);
    e.dispose();
  });

  it('a recovery overtaken by dispose() destroys its new pipeline, not the one handed to the successor', async () => {
    const p1 = fakeGpuPipeline();
    const p2 = fakeGpuPipeline();
    const g = gate();
    h.create = async () => {
      if (h.calls === 1) return p1;
      await g.wait;
      return p2;
    };
    const e = new PhotogrammetryEngine({ ...OPTS });
    await e.init();
    const rec = e.recoverGPU();
    await vi.waitFor(() => expect(h.calls).toBe(2));
    // worker re-init while the recovery compiles
    const rel = e.releasePipeline();
    e.dispose();
    g.open();
    expect(await rec).toBe(false);
    expect(rel?.pipeline).toBe(p1);
    expect(p1.destroy).not.toHaveBeenCalled();
    expect(p2.destroy).toHaveBeenCalledTimes(1);
  });
});

describe('dispose() during init()', () => {
  it('destroys a pipeline compiled for an engine disposed meanwhile, and init rejects', async () => {
    const p = fakeGpuPipeline();
    const g = gate();
    h.create = async () => {
      await g.wait;
      return p;
    };
    const e = new PhotogrammetryEngine({ ...OPTS });
    const run = e.init();
    await vi.waitFor(() => expect(h.calls).toBe(1));
    e.dispose();
    g.open();
    await expect(run).rejects.toMatchObject({ name: ENGINE_ERRORS.crashed });
    expect(p.destroy).toHaveBeenCalledTimes(1);
    expect(e.usesGPU).toBe(false);
  });

  it('closes a session store opened after dispose() and creates no pipeline or live session', async () => {
    const store = new MemoryEngineStore('late');
    const close = vi.spyOn(store, 'close');
    vi.spyOn(SessionEngineStore, 'isSupported').mockReturnValue(true);
    vi.spyOn(SessionEngineStore, 'open').mockResolvedValue(store as unknown as SessionEngineStore);
    const e = new PhotogrammetryEngine({ mode: 'object', quality: 'fast', persist: true, sessionId: 'late' }, { featureBackend: 'cpu' });
    const run = e.init();
    e.dispose();
    await expect(run).rejects.toMatchObject({ name: ENGINE_ERRORS.crashed });
    expect(close).toHaveBeenCalledTimes(1);
    expect(e.liveSession).toBeNull();
    expect(e.releasePipeline()).toBeNull();
  });

  it('a loadSession overtaken by dispose() (a worker close / re-init) rejects and closes the store it opened', async () => {
    const store = new MemoryEngineStore('late');
    await store.ensureSession({ mode: 'object', quality: 'fast', name: 'Late' });
    const close = vi.spyOn(store, 'close');
    vi.spyOn(SessionEngineStore, 'isSupported').mockReturnValue(true);
    vi.spyOn(SessionEngineStore, 'open').mockResolvedValue(store as unknown as SessionEngineStore);
    const e = new PhotogrammetryEngine({ mode: 'object', quality: 'fast', persist: true, sessionId: 'late' }, { featureBackend: 'cpu' });
    const run = e.loadSession('late');
    e.dispose();
    await expect(run).rejects.toMatchObject({ name: ENGINE_ERRORS.crashed });
    expect(close).toHaveBeenCalledTimes(1);
  });

  it('dispose(): a failed scratch-session delete is not an unhandled rejection', async () => {
    const e = new PhotogrammetryEngine({ ...OPTS });
    const close = vi.fn();
    (e as unknown as { scratch: unknown }).scratch = { deleteSession: () => Promise.reject(new Error('UnknownError')), close };
    const unhandled = vi.fn();
    const proc = (globalThis as unknown as { process: { on(ev: string, cb: () => void): void; off(ev: string, cb: () => void): void } }).process;
    proc.on('unhandledRejection', unhandled);
    try {
      e.dispose();
      await new Promise((r) => setTimeout(r, 20));
    } finally {
      proc.off('unhandledRejection', unhandled);
    }
    expect(close).toHaveBeenCalledTimes(1);
    expect(unhandled).not.toHaveBeenCalled();
  });
});

describe('frames during a GPU recovery', () => {
  it('a frame queued behind the recovery runs on the new pipeline, not on the destroyed one', async () => {
    const cap = makeCapture({ frames: 16 });
    const sift = { maxFeatures: 1024, maxImageSize: 640 };
    const p1 = createCpuFeaturePipeline(sift);
    const p2 = createCpuFeaturePipeline(sift);
    const g = gate();
    h.create = async () => {
      if (h.calls === 1) return p1;
      await g.wait;
      return p2;
    };
    const e = new PhotogrammetryEngine({ ...OPTS, hfovPriorDeg: 55 }, { mapping: 'inline' });
    await e.init();
    const frames = frameInputs(cap);
    // until a keyframe exists: the next frame is extracted AND tracked against it
    let i = 0;
    while (i < frames.length && !(e.liveSession!.state.order.length > 0)) await e.processFrame(frames[i++]);
    expect(e.liveSession!.state.order.length).toBeGreaterThan(0);
    const rec = e.recoverGPU();
    await vi.waitFor(() => expect(h.calls).toBe(2));
    const extract = vi.spyOn(p2, 'extract');
    const gpuJobs = vi.spyOn(e as unknown as { gpu: (fn: () => Promise<unknown>) => Promise<unknown> }, 'gpu');
    const frame = e.processFrame(frames[i]);
    // the frame's extraction is queued behind the recovery (a fixed delay would not wait for it on a busy machine)
    await vi.waitFor(() => expect(gpuJobs).toHaveBeenCalled());
    g.open();
    const u = await frame;
    await rec;
    expect(u.debug?.numFeatures).toBeGreaterThan(0);
    expect(extract).toHaveBeenCalledTimes(1);
    e.dispose();
  });
});
