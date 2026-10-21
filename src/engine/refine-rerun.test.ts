/**
 * State refine leaves for a later refine of the same session (src/engine/refine.ts runRefine):
 * the final R_cam_device check marks θ verified only on a clear result (a wrong verified θ would
 * gate every later refine with mis-rotated gyro), and a pair counts as attempted only once it was
 * actually matched (a failed GPU job does not exclude the pair from later refines).
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { FeaturePipeline } from '../features/extractor';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import { handEyeDecision } from './refine';
import { frameInputs, makeCapture } from './testutil';

describe('handEyeDecision (final R_cam_device check)', () => {
  const r = (medianErrDeg: number[], best: 0 | 90 | 180 | 270, pairs = 8) => ({ medianErrDeg, best, pairs });

  it('verifies only a clear winner, after switching to it', () => {
    // best under 4° but not under half the runner-up: neither switched to nor verified
    expect(handEyeDecision(r([4.5, 3.5, 6, 6], 90), 0)).toEqual({ switchTo: null, verified: false });
    expect(handEyeDecision(r([2.9, 2.8, 2.5, 2.7], 180), 90)).toEqual({ switchTo: null, verified: false });
    // ambiguous even when the current θ happens to be the noise-best one
    expect(handEyeDecision(r([2.9, 2.8, 2.5, 2.7], 180), 180)).toEqual({ switchTo: null, verified: false });
    // clear winner: switch (from a wrong or an unset θ) and verify
    expect(handEyeDecision(r([0.4, 3, 5, 3], 0), 90)).toEqual({ switchTo: 0, verified: true });
    expect(handEyeDecision(r([0.4, 3, 5, 3], 0), null)).toEqual({ switchTo: 0, verified: true });
    expect(handEyeDecision(r([0.4, 3, 5, 3], 0), 0)).toEqual({ switchTo: null, verified: true });
    // too large an error, or too few pairs
    expect(handEyeDecision(r([5, 12, 15, 12], 0), 90)).toEqual({ switchTo: null, verified: false });
    expect(handEyeDecision(r([0.4, 3, 5, 3], 0, 3), 90)).toEqual({ switchTo: null, verified: false });
  });
});

describe('refine: pairs whose match failed', () => {
  let pipeline: FeaturePipeline;
  beforeAll(() => {
    pipeline = createCpuFeaturePipeline({ maxFeatures: 2048, maxImageSize: 1024 });
  });

  it('are matched again by the next refine of the session', async () => {
    let failMatch = false;
    // the CPU pipeline, with match() failing (a lost GPU) while the flag is set
    const flaky = new Proxy(pipeline, {
      get(t, k) {
        if (k === 'match' && failMatch) return () => Promise.reject(new Error('GPU device lost (simulated)'));
        const v = Reflect.get(t, k, t) as unknown;
        return typeof v === 'function' ? v.bind(t) : v;
      },
    });
    const cap = makeCapture({ frames: 36, arcDeg: 50 });
    const e = new PhotogrammetryEngine({ mode: 'object', quality: 'fast', hfovPriorDeg: 60 }, { pipeline: flaky, mapping: 'inline' });
    await e.init();
    for (const f of frameInputs(cap)) await e.processFrame(f);
    await e.finishCapture();
    failMatch = true;
    const first = await e.refine();
    failMatch = false;
    expect(first.stats.candidatePairs).toBeGreaterThan(0);
    expect(first.stats.newPairs).toBe(0);
    const second = await e.refine();
    expect(second.stats.candidatePairs).toBe(first.stats.candidatePairs);
    expect(second.stats.newPairs).toBeGreaterThan(0);
    e.dispose();
  }, 300_000);
});
