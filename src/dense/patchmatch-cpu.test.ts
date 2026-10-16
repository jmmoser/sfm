import { describe, expect, it } from 'vitest';
import { patchMatchCPU, lastPatchMatchCPUStats } from './patchmatch-cpu';
import { evaluateDepthMap, formatDepthStats, groundTruthDepthMap, makeSyntheticDataset, surfaceMask } from './synthetic';
import type { DepthMap } from './types';

const log = (...a: unknown[]): void => console.log('[patchmatch-cpu]', ...a);

describe('patchMatchCPU', () => {
  it('near scene 160×120, 1 ref + 6 sources: accurate depths and normals', () => {
    const ds = makeSyntheticDataset('near', { width: 160, height: 120 });
    const dm = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range);
    const { timeMs } = lastPatchMatchCPUStats();
    const mask = surfaceMask(ds.views[0].cam, ds.gtDepth[0], ds.gtNormal[0]);
    const all = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0]);
    const surf = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0], { mask });
    log(`near 160x120 photometric: ${timeMs.toFixed(0)} ms`);
    log(`  all GT px     : ${formatDepthStats(all)}`);
    log(`  surface mask  : ${formatDepthStats(surf)} (${surf.gtPixels} px)`);
    expect(surf.validFraction).toBeGreaterThanOrEqual(0.7);
    expect(surf.accurateFraction).toBeGreaterThanOrEqual(0.85);
    expect(surf.medianNormalErrorDeg).toBeLessThan(10);
    expect(all.validFraction).toBeGreaterThanOrEqual(0.7);
    expect(all.medianRelError).toBeLessThan(0.01);
    // outputs are well-formed
    for (let i = 0; i < dm.depth.length; i++) {
      if (dm.depth[i] > 0) {
        const nx = dm.normal[3 * i];
        const ny = dm.normal[3 * i + 1];
        const nz = dm.normal[3 * i + 2];
        expect(Math.abs(Math.hypot(nx, ny, nz) - 1)).toBeLessThan(1e-4);
        const x = i % 160;
        const y = (i - x) / 160;
        const c = ds.views[0].cam;
        expect(nx * (x - c.cx) / c.f + ny * (y - c.cy) / c.f + nz).toBeLessThan(0); // faces the camera
        expect(dm.depth[i]).toBeGreaterThanOrEqual(ds.range.min);
        expect(dm.depth[i]).toBeLessThanOrEqual(ds.range.max);
        expect(dm.cost[i]).toBeLessThanOrEqual(0.5);
      }
    }
  });

  it('far scene (5–60 m) 160×120 works with inverse-depth sampling', () => {
    const ds = makeSyntheticDataset('far', { width: 160, height: 120 });
    expect(ds.range.max / ds.range.min).toBeGreaterThan(10);
    const dm = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range);
    const mask = surfaceMask(ds.views[0].cam, ds.gtDepth[0], ds.gtNormal[0]);
    const s1 = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0], { mask });
    const s2 = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0], { mask, relTol: 0.02 });
    log(`far 160x120 photometric (range ${ds.range.min.toFixed(1)}–${ds.range.max.toFixed(1)} m): ${lastPatchMatchCPUStats().timeMs.toFixed(0)} ms`);
    log(`  all GT px     : ${formatDepthStats(evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0]))}`);
    log(`  surface mask  : ${formatDepthStats(s1)} | within 2%: ${(100 * s2.accurateFraction).toFixed(1)}%`);
    expect(s1.validFraction).toBeGreaterThanOrEqual(0.7);
    expect(s2.accurateFraction).toBeGreaterThanOrEqual(0.85);
    expect(s1.medianRelError).toBeLessThan(0.01);
    expect(s1.medianNormalErrorDeg).toBeLessThan(10);
  });

  it(
    'geometric-consistency pass (ACMM) reduces the error',
    () => {
      const W = 128;
      const H = 96;
      const ds = makeSyntheticDataset('near', { width: W, height: H });
      const n = ds.views.length;
      const others = (i: number) => ds.views.filter((_, j) => j !== i);
      const t0 = performance.now();
      const photo: DepthMap[] = ds.views.map((v, i) => patchMatchCPU(v, others(i), ds.range));
      const t1 = performance.now();
      const geo = patchMatchCPU(ds.views[0], others(0), ds.range, {}, { prior: photo[0], sourceDepths: photo.slice(1) });
      const t2 = performance.now();
      expect(lastPatchMatchCPUStats().geometric).toBe(true);
      expect(lastPatchMatchCPUStats().iterations).toBe(3);
      const mask = surfaceMask(ds.views[0].cam, ds.gtDepth[0], ds.gtNormal[0]);
      const pAll = evaluateDepthMap(photo[0], ds.gtDepth[0], ds.gtNormal[0]);
      const gAll = evaluateDepthMap(geo, ds.gtDepth[0], ds.gtNormal[0]);
      const pS = evaluateDepthMap(photo[0], ds.gtDepth[0], ds.gtNormal[0], { mask });
      const gS = evaluateDepthMap(geo, ds.gtDepth[0], ds.gtNormal[0], { mask });
      log(`geometric ${W}x${H}: photometric ×${n} ${(t1 - t0).toFixed(0)} ms, geometric ${(t2 - t1).toFixed(0)} ms`);
      log(`  photometric all : ${formatDepthStats(pAll)}`);
      log(`  geometric   all : ${formatDepthStats(gAll)}`);
      log(`  photometric mask: ${formatDepthStats(pS)}`);
      log(`  geometric   mask: ${formatDepthStats(gS)}`);
      expect(gAll.completeness).toBeGreaterThan(pAll.completeness + 0.01);
      expect(gAll.meanRelError).toBeLessThan(pAll.meanRelError);
      expect(gS.accurateFraction).toBeGreaterThanOrEqual(pS.accurateFraction - 0.005);
      expect(gS.medianNormalErrorDeg).toBeLessThan(10);
    },
    300_000,
  );

  it('geometric mode with ground-truth prior and source depths stays at the ground truth', () => {
    const ds = makeSyntheticDataset('near', { width: 96, height: 72 });
    const gts = ds.gtDepth.map((d, i) => groundTruthDepthMap(96, 72, d, ds.gtNormal[i]));
    const dm = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, { maxCost: 2 }, { prior: gts[0], sourceDepths: gts.slice(1) });
    const s = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0]);
    log(`geometric from GT 96x72: ${formatDepthStats(s)}`);
    expect(s.validFraction).toBeGreaterThan(0.99);
    expect(s.accurateFraction).toBeGreaterThan(0.85);
    expect(s.medianRelError).toBeLessThan(0.003);
  });

  it('handles edge cases', () => {
    const ds = makeSyntheticDataset('near', { width: 32, height: 24, numSources: 2 });
    const empty = patchMatchCPU(ds.views[0], [], ds.range);
    expect(empty.depth.every((z) => z === 0)).toBe(true);
    expect(() => patchMatchCPU(ds.views[0], ds.views.slice(1), { min: 2, max: 1 })).toThrow();
    expect(() => patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, { geometric: true })).toThrow();
    // deterministic for a fixed seed
    const a = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, { iterations: 2, seed: 7 });
    const b = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, { iterations: 2, seed: 7 });
    expect(Array.from(a.depth)).toEqual(Array.from(b.depth));
    // priors / source depth maps at a different resolution are accepted (multi-scale use)
    const small: DepthMap = { width: 16, height: 12, depth: new Float32Array(192), normal: new Float32Array(576), cost: new Float32Array(192) };
    for (let y = 0; y < 12; y++) {
      for (let x = 0; x < 16; x++) {
        const i = y * 16 + x;
        const j = 2 * y * 32 + 2 * x;
        small.depth[i] = a.depth[j];
        small.normal.set(a.normal.subarray(3 * j, 3 * j + 3), 3 * i);
      }
    }
    const up = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, { geometricIterations: 1 }, { prior: small, sourceDepths: [small, small] });
    expect(up.width).toBe(32);
    expect(up.depth.some((z) => z > 0)).toBe(true);
  });
});
