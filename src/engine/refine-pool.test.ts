/**
 * Refine's pair verification on the CPU helper pool (review): after every worker failed the
 * remaining pairs are verified on this thread without going through the pool (one warning, not
 * one per pair), and a cancel while verifications are in flight leaves those pairs unattempted,
 * so a re-run in the same session verifies them. Node has no workers: a stand-in pool verifies
 * on this thread, fails, or never answers.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { CameraIntrinsics, FeatureSet, MatchPairs } from '../core/types';
import type { FeaturePipeline } from '../features/extractor';
import { estimateTwoViewGeometry, type TwoViewOptions } from '../geometry/twoview';
import { pairKey } from '../sfm/database';
import type { VocabularyIndex } from '../sfm/vocab';
import type { CpuPool } from './cpu-pool';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import type { LiveSession } from './live';
import { runRefine } from './refine';
import { frameInputs, makeCapture, type SynthCapture } from './testutil';

let pipeline: FeaturePipeline;
let cap: SynthCapture;
beforeAll(() => {
  pipeline = createCpuFeaturePipeline({ maxFeatures: 2048, maxImageSize: 1024 });
  cap = makeCapture({ frames: 44, arcDeg: 55 });
}, 120_000);

type Behaviour = 'verify' | 'fail' | 'hang';

/** CpuPool stand-in: `behaviour(call)` per verification ('fail' breaks every worker). */
class FakePool {
  size = 2;
  calls = 0;
  constructor(private readonly behaviour: (call: number) => Behaviour) {}
  async quantize(vocab: VocabularyIndex, fs: FeatureSet): Promise<Uint32Array> {
    return vocab.quantize(fs);
  }
  verify(kp1: Float32Array, kp2: Float32Array, m: MatchPairs, c1: CameraIntrinsics, c2: CameraIntrinsics, o: TwoViewOptions): Promise<unknown> {
    const call = ++this.calls;
    if (this.size === 0) return Promise.reject(new Error('cpu pool: no usable worker'));
    const b = this.behaviour(call);
    if (b === 'fail') {
      this.size = 0;
      return Promise.reject(new Error('cpu worker crashed'));
    }
    if (b === 'hang') return new Promise(() => undefined);
    return Promise.resolve({ geometry: estimateTwoViewGeometry(kp1, kp2, m, c1, c2, o), ms: 0 });
  }
}

async function captured(): Promise<{ e: PhotogrammetryEngine; live: LiveSession }> {
  const e = new PhotogrammetryEngine({ mode: 'object', quality: 'fast', hfovPriorDeg: 60 }, { pipeline, mapping: 'inline' });
  await e.init();
  for (const f of frameInputs(cap)) await e.processFrame(f);
  await e.finishCapture();
  const live = e.liveSession!;
  await live.flush();
  return { e, live };
}

describe('refine on the CPU pool', () => {
  it('verifies here, warning once, after every worker failed', async () => {
    const { e, live } = await captured();
    const pool = new FakePool((call) => (call === 3 ? 'fail' : 'verify'));
    const warns: string[] = [];
    const out = await runRefine({
      live,
      config: { pipeline, mapping: 'inline' },
      store: e.engineStore!,
      log: (l, m) => void (l === 'warn' && warns.push(m)),
      pool: pool as unknown as CpuPool,
    });
    expect(out.model.images.length).toBeGreaterThan(5);
    expect(Number(out.result.stats.newPairs)).toBeGreaterThan(20);
    // the failed job is verified here; no pair after it goes through the (empty) pool
    expect(pool.calls).toBe(3);
    expect(warns.filter((w) => /CPU worker failed/.test(w)).length).toBe(1);
    expect(warns.filter((w) => /no usable CPU worker/.test(w)).length).toBe(1);
    e.dispose();
  }, 300_000);

  it('a cancel while verifications are in flight leaves those pairs for a re-run', async () => {
    const { e, live } = await captured();
    const cancel = { cancelled: false };
    const pool = new FakePool((call) => {
      if (call === 3) cancel.cancelled = true;
      return 'hang';
    });
    const matched: number[] = [];
    const putative = live.putativeMatches.bind(live);
    live.putativeMatches = async (a, b, c) => {
      const m = await putative(a, b, c);
      if (m) matched.push(pairKey(a, b));
      return m;
    };
    await expect(runRefine({ live, config: { pipeline, mapping: 'inline' }, store: e.engineStore!, log: () => undefined, pool: pool as unknown as CpuPool }, undefined, cancel)).rejects.toMatchObject({
      name: 'CancelledError',
    });
    expect(pool.calls).toBe(3);
    expect(matched.length).toBeGreaterThanOrEqual(3);
    // nothing was recorded: none of the matched pairs counts as attempted
    expect(matched.filter((k) => live.state.attempted.has(k))).toEqual([]);
    e.dispose();
  }, 300_000);
});
