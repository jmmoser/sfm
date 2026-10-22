/**
 * Persistence of the splat stage (Node): a retrain whose writes stop right after the new model
 * blob (a kill, a quota error on the meta) must not pair that blob with the previous run's meta;
 * a model ended early ("Use it now") keeps its checkpoint for a later run (loadSession:
 * splatContinuesModel only while the stored checkpoint is that run's), a complete one clears it,
 * and so does a densify that replaces the dense cloud. runSplat is mocked: it returns a model of
 * the size (and the stats) the test asks for (no trainer, no GPU).
 */
import { beforeAll, describe, expect, it, vi } from 'vitest';
import type { FeaturePipeline } from '../features/extractor';
import type { SplatModel } from '../splat/types';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import { MemoryEngineStore } from './persistence';
import { frameInputs, makeCapture, type SynthCapture } from './testutil';

const h = vi.hoisted(() => ({ count: 100, stats: {} as Record<string, number | string> }));

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
    h.stats = { splatFinishedEarly: 1, splatContinueFrom: 1000, splatContinueKey: 'run' };
    expect((await e.splat()).stats).toMatchObject({ splatFinishedEarly: 1, splatContinueFrom: 1000 });
    expect(await store.listBinaryKeys('splat/ckpt')).toEqual(['splat/ckpt0/params']);
    e.dispose();

    const reopened = new PhotogrammetryEngine({ mode: 'auto', quality: 'fast' }, { pipeline, store });
    const sum = await reopened.loadSession('c');
    expect(sum.splat?.iterations).toBe(1000);
    expect(sum.splat?.stats.splatContinueFrom).toBe(1000);
    expect(sum.stats.splatResumableIteration).toBe(1000);
    expect(sum.stats.splatContinuesModel).toBe(1);
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
    expect(after.stats.splatContinueKey).toBeUndefined();
    expect(after.stats.splatContinuesModel).toBeUndefined();
    reopened.dispose();
  }, 300_000);

  it('another run’s checkpoint at the model’s iteration is no "continue"; a new dense cloud drops the checkpoints (review)', async () => {
    const store = new KillableStore('k');
    const opts = { mode: 'object' as const, quality: 'fast' as const, sessionId: 'k', hfovPriorDeg: 60 };
    const e = new PhotogrammetryEngine(opts, { pipeline, store, mapping: 'inline' });
    await e.init();
    for (const f of frameInputs(cap)) await e.processFrame(f);
    await e.finishCapture();
    await e.refine();
    const checkpoint = async (key: string, quality: string): Promise<void> => {
      await store.putValue('splat/progress', { key, quality, iteration: 1000, count: 100, slot: 0, moments: true });
      await store.putBinary('splat/ckpt0/params', new Uint8Array(64));
    };
    // a Fast model ended early at 1 000 (its run 'fast-run')
    await checkpoint('fast-run', 'fast');
    h.count = 100;
    h.stats = { splatFinishedEarly: 1, splatContinueFrom: 1000, splatContinueKey: 'fast-run' };
    await e.splat();
    e.dispose();
    // a rebuild at Balanced replaced the checkpoint and was interrupted at 1 000 too
    await checkpoint('balanced-run', 'balanced');
    const a = new PhotogrammetryEngine({ mode: 'auto', quality: 'fast' }, { pipeline, store });
    const sumA = await a.loadSession('k');
    expect(sumA.splat?.iterations).toBe(1000);
    expect(sumA.stats).toMatchObject({ splatResumableIteration: 1000, splatResumableQuality: 'balanced' });
    expect(sumA.stats.splatContinuesModel).toBeUndefined();
    a.dispose();
    // the model's own checkpoint again: continuable — until a densify replaces the dense cloud
    await checkpoint('fast-run', 'fast');
    const b = new PhotogrammetryEngine({ mode: 'auto', quality: 'fast' }, { pipeline, store });
    expect((await b.loadSession('k')).stats.splatContinuesModel).toBe(1);
    await b.densify({ maxImageSize: 80, maxViews: 6 });
    expect(await store.getValue('splat/progress')).toBeNull();
    expect(await store.listBinaryKeys('splat/ckpt')).toEqual([]);
    const sumB = await b.loadSession('k');
    expect(sumB.stats.splatResumableIteration).toBeUndefined();
    expect(sumB.stats.splatContinuesModel).toBeUndefined();
    expect(sumB.splat?.count).toBe(100); // the stored model stays
    b.dispose();
  }, 300_000);
});
