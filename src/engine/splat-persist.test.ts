/**
 * Persistence of the splat stage (Node): a retrain whose writes stop right after the new model
 * blob (a kill, a quota error on the meta) must not pair that blob with the previous run's meta;
 * a model ended early ("Use it now") keeps its checkpoint for a later run, a complete one clears it.
 * runSplat is mocked: it returns a model of the size (and the stats) the test asks for (no trainer,
 * no GPU).
 */
import { beforeAll, describe, expect, it, vi } from 'vitest';
import type { FeaturePipeline } from '../features/extractor';
import type { SplatModel } from '../splat/types';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import { MemoryEngineStore } from './persistence';
import { frameInputs, makeCapture, type SynthCapture } from './testutil';

const h = vi.hoisted(() => ({ count: 100, stats: {} as Record<string, number> }));

vi.mock('./splat', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./splat')>()),
  runSplat: async () => {
    const n = h.count;
    const model: SplatModel = {
      count: n,
      shDegree: 0,
      means: new Float32Array(3 * n),
      logScales: new Float32Array(3 * n),
      quats: new Float32Array(4 * n),
      opacities: new Float32Array(n),
      sh: new Float32Array(3 * n),
    };
    return { model, antialiased: true, background: [0, 0, 0] as [number, number, number], iterations: 10 * n, stats: { ...h.stats } };
  },
}));

/** Persistent in-memory store whose writes all fail once `killAfterModel` is set and 'splat/model' was written. */
class KillableStore extends MemoryEngineStore {
  killAfterModel = false;
  private dead = false;
  constructor(id: string) {
    super(id);
    Object.defineProperty(this, 'persistent', { value: true });
  }
  override async putValue(key: string, value: unknown): Promise<void> {
    if (this.dead) throw new Error('killed');
    return super.putValue(key, value);
  }
  override async putBinary(key: string, data: ArrayBuffer | ArrayBufferView): Promise<void> {
    if (this.dead) throw new Error('killed');
    await super.putBinary(key, data);
    if (this.killAfterModel && key === 'splat/model') this.dead = true;
  }
}

let pipeline: FeaturePipeline;
let cap: SynthCapture;
beforeAll(() => {
  pipeline = createCpuFeaturePipeline({ maxFeatures: 2048, maxImageSize: 1024 });
  cap = makeCapture({ frames: 36, arcDeg: 50 });
}, 120_000);

describe('splat() persistence', () => {
  it('a retrain stopped right after the model write leaves no model paired with the old meta', async () => {
    const store = new KillableStore('p');
    const opts = { mode: 'object' as const, quality: 'fast' as const, sessionId: 'p', hfovPriorDeg: 60 };
    const e = new PhotogrammetryEngine(opts, { pipeline, store, mapping: 'inline' });
    await e.init();
    for (const f of frameInputs(cap)) await e.processFrame(f);
    await e.finishCapture();
    await e.refine();
    h.count = 100;
    expect((await e.splat()).count).toBe(100);
    // retrain: the app is killed once the new model is stored
    h.count = 200;
    store.killAfterModel = true;
    await expect(e.splat()).rejects.toThrow('killed');
    e.dispose();

    const reopened = new PhotogrammetryEngine({ mode: 'auto', quality: 'balanced' }, { pipeline, store });
    const sum = await reopened.loadSession('p');
    // either both of one run, or none (here: none, the new meta never landed)
    const meta = sum.splat?.count ?? null;
    const model = (await reopened.splatModel())?.model.count ?? null;
    expect(meta).toBe(model);
    reopened.dispose();
  }, 300_000);

  it('a model ended early keeps the checkpoint it can continue from; a complete run clears it', async () => {
    const store = new KillableStore('c');
    const opts = { mode: 'object' as const, quality: 'fast' as const, sessionId: 'c', hfovPriorDeg: 60 };
    const e = new PhotogrammetryEngine(opts, { pipeline, store, mapping: 'inline' });
    await e.init();
    for (const f of frameInputs(cap)) await e.processFrame(f);
    await e.finishCapture();
    await e.refine();
    // the checkpoint runSplat wrote at the iteration it stopped at (100 Gaussians × 10 = 1000)
    const checkpoint = async (): Promise<void> => {
      await store.putValue('splat/progress', { key: 'run', quality: 'fast', iteration: 1000, count: 100, slot: 0, moments: true });
      await store.putBinary('splat/ckpt0/params', new Uint8Array(64));
    };
    await checkpoint();
    h.count = 100;
    h.stats = { splatFinishedEarly: 1, splatContinueFrom: 1000 };
    expect((await e.splat()).stats).toMatchObject({ splatFinishedEarly: 1, splatContinueFrom: 1000 });
    expect(await store.listBinaryKeys('splat/ckpt')).toEqual(['splat/ckpt0/params']);
    e.dispose();

    const reopened = new PhotogrammetryEngine({ mode: 'auto', quality: 'fast' }, { pipeline, store });
    const sum = await reopened.loadSession('c');
    expect(sum.splat?.iterations).toBe(1000);
    expect(sum.splat?.stats.splatContinueFrom).toBe(1000);
    expect(sum.stats.splatResumableIteration).toBe(1000);
    // continued to the end: cleared, and no early-finish keys left in the session's stats
    h.stats = {};
    h.count = 300;
    expect((await reopened.splat()).iterations).toBe(3000);
    expect(await store.getValue('splat/progress')).toBeNull();
    expect(await store.listBinaryKeys('splat/ckpt')).toEqual([]);
    const after = await reopened.loadSession('c');
    expect(after.stats.splatResumableIteration).toBeUndefined();
    expect(after.stats.splatFinishedEarly).toBeUndefined();
    expect(after.stats.splatContinueFrom).toBeUndefined();
    reopened.dispose();
  }, 300_000);
});
