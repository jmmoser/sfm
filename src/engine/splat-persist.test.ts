/**
 * Persistence of the splat stage (Node): a retrain whose writes stop right after the new model
 * blob (a kill, a quota error on the meta) must not pair that blob with the previous run's meta.
 * runSplat is mocked: it returns a model of the size the test asks for (no trainer, no GPU).
 */
import { beforeAll, describe, expect, it, vi } from 'vitest';
import type { FeaturePipeline } from '../features/extractor';
import type { SplatModel } from '../splat/types';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import { MemoryEngineStore } from './persistence';
import { frameInputs, makeCapture, type SynthCapture } from './testutil';

const h = vi.hoisted(() => ({ count: 100 }));

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
    return { model, antialiased: true, background: [0, 0, 0] as [number, number, number], iterations: 10 * n, stats: {} };
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
});
