import { describe, expect, it } from 'vitest';
import { bundleAdjust } from './solver';
import { fixGaugeTwoPoses, makeSyntheticBA, perturbBA, syntheticErrors } from './synthetic';

/** Run a few LM iterations and return per-iteration wall times (ms). */
function timeIterations(
  cfg: Parameters<typeof makeSyntheticBA>[0],
  solver: 'auto' | 'dense_schur' | 'pcg_schur',
  iterations: number,
) {
  const s = makeSyntheticBA(cfg);
  const p = s.problem;
  fixGaugeTwoPoses(p);
  p.cameraRefine[0] = { focal: true, principalPoint: false, distortion: true };
  perturbBA(s, { rotSigma: 0.002, centreSigma: 0.02, pointSigma: 0.1, focalScale: 0.02, k1Delta: 0.01 });
  const times: number[] = [];
  let last = performance.now();
  const t0 = last;
  const r = bundleAdjust(p, {
    linearSolver: solver,
    maxIterations: iterations,
    functionTolerance: 0,
    onIteration: () => {
      const now = performance.now();
      times.push(now - last);
      last = now;
    },
  });
  const total = performance.now() - t0;
  const steady = times.slice(1).sort((a, b) => a - b);
  return { r, s, times, total, median: steady[steady.length >> 1], nObs: p.obsPose.length };
}

describe('bundleAdjust performance (logged)', () => {
  it('100 poses / 20k points / ~100k observations: dense Schur < 400 ms per iteration', () => {
    const { r, s, times, median, nObs } = timeIterations(
      { numPoses: 100, numPoints: 20000, layout: 'strafe', trackWindow: 2, radius: 0.3, k1: -0.05, noisePx: 0.5, seed: 3 },
      'auto',
      8,
    );
    const e = syntheticErrors(s);
    console.log(
      `[ba perf] 100 poses, 20000 pts, ${nObs} obs, ${r.linearSolver} (n=${r.numReducedParams}): first iteration (incl. setup) ${times[0].toFixed(0)} ms, ` +
        `median ${median.toFixed(0)} ms/iter; rms ${r.initialRmsPx.toFixed(1)} → ${r.finalRmsPx.toFixed(3)} px; f err ${(e.maxFocalRelErr * 100).toFixed(3)} %`,
    );
    expect(nObs).toBeGreaterThan(95000);
    expect(r.linearSolver).toBe('dense_schur');
    expect(median).toBeLessThan(400);
    expect(r.finalRmsPx).toBeLessThan(0.8);
  });

  it('typical local BA (12 free + 18 fixed poses, 6k points) is fast after warm-up', () => {
    const run = () => {
      const s = makeSyntheticBA({ numPoses: 30, numPoints: 6000, layout: 'orbit', orbitArc: 0.4, trackWindow: 4, noisePx: 0.5, seed: 8 });
      const p = s.problem;
      for (let i = 0; i < 18; i++) p.poseFixed[i] = 1;
      perturbBA(s, { rotSigma: 0.003, centreSigma: 0.01, pointSigma: 0.01 });
      const t = performance.now();
      const r = bundleAdjust(p, { maxIterations: 10 });
      return { r, ms: performance.now() - t, nObs: p.obsPose.length };
    };
    run(); // JIT warm-up (the live mapper calls BA repeatedly)
    const { r, ms, nObs } = run();
    console.log(
      `[ba perf] local BA: 12 free poses, 6000 pts, ${nObs} obs: ${r.iterations} iterations in ${ms.toFixed(0)} ms (${(ms / r.iterations).toFixed(1)} ms/iter), ${r.termination}`,
    );
    expect(r.success).toBe(true);
    // ≈ 50–110 ms/iter on an idle-to-loaded 4-core x86 box; the bound only catches gross
    // regressions (the unit project runs test files in parallel, timings are noisy).
    expect(ms / Math.max(1, r.iterations)).toBeLessThan(400);
  });

  it('500 poses / 60k points / ~400k observations: PCG Schur at a few s per iteration', () => {
    const cfg = { numPoses: 500, numPoints: 60000, layout: 'strafe', trackWindow: 3, radius: 0.5, k1: -0.05, noisePx: 0.5, seed: 4 } as const;
    const { r, times, median, nObs } = timeIterations(cfg, 'pcg_schur', 4);
    console.log(
      `[ba perf] 500 poses, 60000 pts, ${nObs} obs, ${r.linearSolver} (n=${r.numReducedParams}): first iteration (incl. setup) ${times[0].toFixed(0)} ms, ` +
        `median ${median.toFixed(0)} ms/iter, ${(r.pcgIterations / r.iterations).toFixed(0)} PCG it/solve; rms ${r.initialRmsPx.toFixed(1)} → ${r.finalRmsPx.toFixed(3)} px`,
    );
    expect(nObs).toBeGreaterThan(380000);
    expect(r.linearSolver).toBe('pcg_schur');
    expect(median).toBeLessThan(5000);
    expect(r.finalCost).toBeLessThan(r.initialCost / 10);
    // 'auto' recognises the banded reduced system and solves it directly (skyline Cholesky).
    const a = timeIterations(cfg, 'auto', 4);
    console.log(
      `[ba perf] same problem, 'auto' → ${a.r.linearSolver}: first iteration ${a.times[0].toFixed(0)} ms, median ${a.median.toFixed(0)} ms/iter; ` +
        `rms → ${a.r.finalRmsPx.toFixed(3)} px (cost ${a.r.finalCost.toFixed(1)} vs PCG ${r.finalCost.toFixed(1)})`,
    );
    expect(a.r.linearSolver).toBe('dense_schur');
    expect(a.r.finalCost).toBeLessThanOrEqual(r.finalCost * 1.001);
  });
});
