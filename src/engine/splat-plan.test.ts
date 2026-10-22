/**
 * Splat stage, the parts that run without a GPU: presets (quality table, dense-size clamp,
 * overrides), the memory plan (`planSplat` = the trainer's own estimate + host transients) and its
 * degradation (`fitSplatPlan`), the stored encodings (compact f16 model, exact checkpoint
 * parameters, bf16 Adam moments), the run key and the 'ply-splat' / 'spz' exports.
 */
import { describe, expect, it } from 'vitest';
import { Rng } from '../math/random';
import { decodeSplatPLY, encodeSplatPLY } from '../splat/ply';
import { decodeSplatSPZ } from '../splat/spz';
import { SplatTrainer } from '../splat/trainer';
import { randomScene } from '../splat/testutil';
import type { SplatModel } from '../splat/types';
import { buildExport } from './export';
import { MB, MEM, fitSplatPlan, planSplat, type SplatPlanInput } from './memory';
import { packArrays, packSplatModel, packSplatMoments, packSplatParams, unpackSplatModel, unpackSplatMoments, unpackSplatParams } from './persistence';
import { SPLAT_PRESETS, resolveDensePreset, resolveSplatPreset } from './presets';
import { splatRunKey } from './splat';
import type { SparseModel } from './types';

const dims43 = (size: number): { width: number; height: number } => ({ width: Math.round(0.75 * size), height: size });

function base(budgetMB = 1200, residentBytes = 30 * MB): Omit<SplatPlanInput, 'views' | 'width' | 'height' | 'maxGaussians'> {
  return { budgetMB, residentBytes, shDegree: 0, loadedCloudPoints: 2_000_000, initPoints: 150_000 };
}

describe('splat presets', () => {
  it('fast / balanced / max: 3k / 5k / 7k iterations, 200k / 300k / 500k Gaussians, images ≤ the dense size', () => {
    expect([SPLAT_PRESETS.fast.iterations, SPLAT_PRESETS.balanced.iterations, SPLAT_PRESETS.max.iterations]).toEqual([3000, 5000, 7000]);
    expect([SPLAT_PRESETS.fast.maxGaussians, SPLAT_PRESETS.balanced.maxGaussians, SPLAT_PRESETS.max.maxGaussians]).toEqual([200_000, 300_000, 500_000]);
    for (const q of ['fast', 'balanced', 'max'] as const) {
      const p = resolveSplatPreset(q);
      expect(p.maxImageSize).toBeLessThanOrEqual(resolveDensePreset(q).maxImageSize);
      expect(p.chunk).toBeLessThanOrEqual(64);
      expect(p.checkpointEvery).toBeGreaterThan(p.chunk);
    }
  });

  it('overrides win, but never above the dense size; the chunk stays in the dropped-work window', () => {
    const p = resolveSplatPreset('fast', { iterations: 100, maxGaussians: 5000, maxImageSize: 4000, chunk: 500 }, { maxImageSize: 640 });
    expect(p.iterations).toBe(100);
    expect(p.maxGaussians).toBe(5000);
    expect(p.maxImageSize).toBe(640);
    expect(p.chunk).toBe(64);
    expect(resolveSplatPreset('balanced', { maxImageSize: undefined }).maxImageSize).toBe(SPLAT_PRESETS.balanced.maxImageSize);
  });
});

describe('planSplat / fitSplatPlan', () => {
  it('the GPU part is the trainer’s own memory estimate', () => {
    const plan = planSplat({ ...base(), views: 90, width: 600, height: 800, maxGaussians: 200_000 });
    const est = SplatTrainer.memoryEstimate({ maxGaussians: 200_000, shDegree: 0 }, 600, 800, 90);
    expect(plan.gpuBytes).toBe(est.total);
    expect(plan.largestBuffer).toBe(est.largestBuffer);
    const training = plan.stages.find((s) => s.stage === 'training')!;
    expect(training.bytes).toBeGreaterThan(est.total);
    expect(plan.peakBytes).toBe(Math.max(...plan.stages.map((s) => s.bytes)) + MEM.reserve + 30 * MB);
  });

  it('fast on a 1.2 GB tab (object-real-like: 90 keyframes at 600×800) fits unchanged', () => {
    const p = resolveSplatPreset('fast');
    const fit = fitSplatPlan(base(), { views: 90, size: p.maxImageSize, maxGaussians: p.maxGaussians }, { dims: dims43 });
    expect(fit.degraded).toEqual([]);
    expect(fit.plan.fits).toBe(true);
    expect(fit.plan.gpuBytes).toBeLessThan(400 * MB);
  });

  it('balanced with 160 views on a 600 MB tab degrades to fit, views first, within the floors', () => {
    const p = resolveSplatPreset('balanced');
    const want = { views: 160, size: p.maxImageSize, maxGaussians: p.maxGaussians };
    const fit = fitSplatPlan(base(600), want, { dims: dims43 });
    expect(fit.plan.fits).toBe(true);
    expect(fit.views).toBeLessThan(160);
    expect(fit.views).toBeGreaterThanOrEqual(16);
    expect(fit.size).toBeGreaterThanOrEqual(480);
    expect(fit.maxGaussians).toBeGreaterThanOrEqual(50_000);
    expect(fit.degraded[0]).toMatch(/^views 160 → /);
    // the same request fits a 1.2 GB tab with fewer cuts
    const big = fitSplatPlan(base(1200), want, { dims: dims43 });
    expect(big.views).toBeGreaterThanOrEqual(fit.views);
  });

  it('an impossible budget stops at the hard floors (and says it does not fit)', () => {
    const fit = fitSplatPlan({ ...base(300), residentBytes: 200 * MB }, { views: 200, size: 1280, maxGaussians: 500_000 }, { dims: dims43, minViews: 20, minSize: 480, minGaussians: 60_000 });
    expect(fit.plan.fits).toBe(false);
    expect([fit.views, fit.size, fit.maxGaussians]).toEqual([20, 480, 60_000]);
  });

  it('keeps the largest buffer under the binding limit', () => {
    const fit = fitSplatPlan(base(8000), { views: 10, size: 1280, maxGaussians: 5_000_000 }, { dims: dims43, maxBufferBytes: 128 * MB });
    expect(fit.plan.largestBuffer).toBeLessThanOrEqual(128 * MB);
  });
});

function maxRel(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]) / Math.max(Math.abs(b[i]), 1e-30));
  return m;
}

describe('stored splat encodings', () => {
  for (const deg of [0, 1] as const) {
    it(`compact model (SH ${deg}): f32 means, f16 rest — round trip within the f16 precision`, () => {
      const m = randomScene({ count: 500, seed: 3 + deg, shDegree: deg });
      for (let i = 0; i < m.means.length; i++) m.means[i] += 1234.5; // far from the origin: still exact
      const buf = packSplatModel(m, { note: 'x' });
      expect(buf.byteLength).toBeLessThan(500 * (12 + 2 * (3 + 4 + 1 + 3 * (deg ? 4 : 1))) + 1024);
      const { model: r, extra } = unpackSplatModel(buf);
      expect(extra).toEqual({ note: 'x' });
      expect(r.count).toBe(500);
      expect(r.shDegree).toBe(deg);
      expect(Array.from(r.means)).toEqual(Array.from(m.means));
      for (let i = 0; i < m.logScales.length; i++) expect(Math.abs(r.logScales[i] - m.logScales[i])).toBeLessThan(0.004);
      for (let i = 0; i < m.opacities.length; i++) expect(Math.abs(r.opacities[i] - m.opacities[i])).toBeLessThan(0.002);
      for (let i = 0; i < m.sh.length; i++) expect(Math.abs(r.sh[i] - m.sh[i])).toBeLessThan(0.002);
      // quaternions are stored normalised (same rotation)
      for (let i = 0; i < m.count; i++) {
        const l = Math.hypot(m.quats[4 * i], m.quats[4 * i + 1], m.quats[4 * i + 2], m.quats[4 * i + 3]);
        for (let c = 0; c < 4; c++) expect(Math.abs(r.quats[4 * i + c] - m.quats[4 * i + c] / l)).toBeLessThan(1e-3);
      }
    });
  }

  it('checkpoint parameters are exact; Adam moments keep the f32 range at bf16 precision', () => {
    const m = randomScene({ count: 300, seed: 11, shDegree: 1 });
    const p = unpackSplatParams(packSplatParams(m));
    for (const k of ['means', 'logScales', 'quats', 'opacities', 'sh'] as const) expect(Array.from(p[k])).toEqual(Array.from(m[k]));
    // second moments of tiny gradients (≈ 1e-14) must not flush to zero (f16 would)
    const rng = new Rng(5);
    const v: SplatModel = { ...m, means: m.means.map(() => 10 ** rng.uniform(-15, -1)), logScales: m.logScales.map(() => rng.uniform(-1, 1) * 1e-6), quats: m.quats.slice(), opacities: m.opacities.slice(), sh: m.sh.slice() };
    const r = unpackSplatMoments(packSplatMoments(v));
    expect(maxRel(r.means, v.means)).toBeLessThanOrEqual(2 ** -8);
    expect(maxRel(r.logScales, v.logScales)).toBeLessThanOrEqual(2 ** -8);
    expect(r.means.every((x) => x > 0)).toBe(true);
    expect(packSplatMoments(v).byteLength).toBeLessThan(packSplatParams(v).byteLength * 0.6);
  });

  it('damaged or foreign containers are rejected', () => {
    const m = randomScene({ count: 20, seed: 2 });
    const buf = packSplatModel(m);
    expect(() => unpackSplatParams(buf)).toThrow(/damaged/);
    expect(() => unpackSplatModel(buf.slice(0, buf.byteLength - 40))).toThrow(/damaged/);
    expect(() => unpackSplatModel(packArrays({ kind: 'splat-model', count: 20, shDegree: 0, encoding: 'f16' }, [m.means]))).toThrow(/damaged/);
  });
});

describe('splat run key and export', () => {
  const model: SparseModel = {
    cameras: new Map([[1, { width: 640, height: 480, f: 500, cx: 320, cy: 240, k1: 0, k2: 0 }]]),
    images: [0, 1, 2].map((id) => ({ id, cameraId: 1, pose: { R: Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]), t: Float64Array.from([id * 0.1, 0, 0]) } })),
    numPoints: 2,
    xyz: Float64Array.from([0, 0, 1, 1, 1, 2]),
    colors: new Uint8Array(6),
    errors: new Float32Array(2),
    trackOffsets: Uint32Array.from([0, 2, 4]),
    trackImages: Uint32Array.from([0, 1, 1, 2]),
    trackFeatures: Uint32Array.from([0, 0, 1, 1]),
  };

  it('the key follows the poses, the dense cloud and the settings', () => {
    const k = splatRunKey(model, null, { quality: 'fast', iterations: 3000 });
    expect(splatRunKey(model, null, { quality: 'fast', iterations: 3000 })).toBe(k);
    expect(splatRunKey(model, null, { quality: 'fast', iterations: 5000 })).not.toBe(k);
    expect(splatRunKey(model, { count: 10, first: [0, 0, 0], last: [1, 1, 1] }, { quality: 'fast', iterations: 3000 })).not.toBe(k);
    const moved = { ...model, images: model.images.map((im, i) => (i === 2 ? { ...im, pose: { R: im.pose.R, t: Float64Array.from([0.3, 0, 0.001]) } } : im)) };
    expect(splatRunKey(moved, null, { quality: 'fast', iterations: 3000 })).not.toBe(k);
  });

  it("exportModel('ply-splat') writes the stored model as an INRIA PLY", async () => {
    const m = randomScene({ count: 64, seed: 9 });
    const blob = await buildExport('ply-splat', {
      name: 'scan',
      sparse: null,
      model,
      dense: null,
      mesh: null,
      store: null,
      features: async () => null,
      splat: async () => ({ model: m, antialiased: true, background: [0, 0, 0], iterations: 10 }),
    });
    const { model: r, antialiased } = decodeSplatPLY(new Uint8Array(await blob.arrayBuffer()));
    expect(antialiased).toBe(true);
    expect(Array.from(r.means)).toEqual(Array.from(m.means));
    expect(Array.from(r.sh)).toEqual(Array.from(m.sh));
    await expect(buildExport('ply-splat', { name: 'scan', sparse: null, model, dense: null, mesh: null, store: null, features: async () => null, splat: async () => null })).rejects.toThrow(
      /No Gaussian splats/,
    );
  });

  it("exportModel('spz') writes the stored model as SPZ v3 (≤ 30 % of the PLY)", async () => {
    const m = randomScene({ count: 400, seed: 10 });
    const src = {
      name: 'scan',
      sparse: null,
      model,
      dense: null,
      mesh: null,
      store: null,
      features: async () => null,
      splat: async () => ({ model: m, antialiased: true, background: [0, 0, 0] as [number, number, number], iterations: 10 }),
    };
    const blob = await buildExport('spz', src);
    const d = await decodeSplatSPZ(new Uint8Array(await blob.arrayBuffer()));
    expect(d.version).toBe(3);
    expect(d.antialiased).toBe(true);
    expect(d.model.count).toBe(400);
    for (let i = 0; i < 3 * m.count; i++) expect(Math.abs(d.model.means[i] - m.means[i])).toBeLessThanOrEqual(2 ** -(d.fractionalBits + 1) + 1e-9);
    // random Gaussians barely compress: the 20 B record against the PLY's 68 B
    expect(blob.size).toBeLessThan(0.3 * encodeSplatPLY(m).length);
    await expect(buildExport('spz', { ...src, splat: async () => null })).rejects.toThrow(/No Gaussian splats/);
  });
});
