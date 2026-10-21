/**
 * Adversarial GPU tests: CPU/GPU parity of the random initialisation (PCG streams),
 * the non-unrolled (loop) shader path, odd sizes and 1-row bands, texture
 * half-pixel conventions (signed depth bias on an analytic plane), mixed source
 * resolutions with grow-only texture arrays, non-finite geometric inputs,
 * selfTest() with non-default instance options, and device loss.
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { getGPU, type GPUContext } from '../gpu/device';
import { PatchMatchGPU } from './patchmatch-gpu';
import { patchMatchCPU } from './patchmatch-cpu';
import { evaluateDepthMap, formatDepthStats, lookAt, makeNearScene, makeSyntheticDataset, pinholeIntrinsics, renderDataset, renderSynthetic, surfaceMask, type SynthScene } from './synthetic';
import type { DenseView, DepthMap } from './types';

const log = (...a: unknown[]): void => console.log('[patchmatch-adv-gpu]', ...a);

let ctx: GPUContext | null = null;
let pm: PatchMatchGPU;

beforeAll(async () => {
  ctx = await getGPU();
  if (!ctx) throw new Error('WebGPU unavailable in the test browser');
  pm = await PatchMatchGPU.create(ctx);
}, 600_000); // pipeline compilation on SwiftShader can exceed the 30 s default hook timeout while other GPU files run

afterAll(() => {
  pm?.destroy();
});

function allFinite(dm: DepthMap): boolean {
  for (let i = 0; i < dm.depth.length; i++) {
    if (!Number.isFinite(dm.depth[i]) || !Number.isFinite(dm.cost[i])) return false;
    if (!Number.isFinite(dm.normal[3 * i]) || !Number.isFinite(dm.normal[3 * i + 1]) || !Number.isFinite(dm.normal[3 * i + 2])) return false;
  }
  return true;
}

/**
 * Per-component tolerance between a GPU normal (as `compute()` returns it) and the CPU normal of
 * the same hypothesis. Two independent terms:
 * - 1e-4: the GPU's f32 sampling of the normal (sin / cos / sqrt). SwiftShader's sin / cos are off
 *   by up to 1.9e-4 (abs, near |value| = 0.985; within WGSL's 2^-11 bound), which leaves its
 *   normalised init normals within 7.3e-5 of the CPU's (64×48 'far' set, read back unquantised).
 * - 6.5e-5: the output codec (octahedral snorm16 since the 12 B/px readback). Each coordinate
 *   rounds by <= 0.5 / 32767 and the decode is 3-Lipschitz (sigma_max of its Jacobian <= sqrt 3,
 *   1 / |P| <= sqrt 3), so |dn| <= 3 * sqrt(2) * 0.5 / 32767 = 6.474e-5; 2e7 sampled normals reach
 *   6.4726e-5 (max single component 5.67e-5).
 * A wrong PCG stream or sampling formula moves normals by O(0.1-1), far outside either term.
 */
const NORMAL_TOL = 1e-4 + 6.5e-5;

interface Parity {
  depthSame: number;
  normalSame: number;
  maxNormalDiff: number;
  costClose: number;
  meanCostDiff: number;
}

/** Per-pixel agreement of two depth maps (fractions over all pixels). */
function parity(a: DepthMap, b: DepthMap, depthTol: number, normalTol: number, costTol: number): Parity {
  const n = a.depth.length;
  let ds = 0;
  let ns = 0;
  let nd = 0;
  let cc = 0;
  let cd = 0;
  for (let i = 0; i < n; i++) {
    if (Math.abs(a.depth[i] - b.depth[i]) <= depthTol * Math.max(b.depth[i], 1e-9)) ds++;
    const e = Math.max(Math.abs(a.normal[3 * i] - b.normal[3 * i]), Math.abs(a.normal[3 * i + 1] - b.normal[3 * i + 1]), Math.abs(a.normal[3 * i + 2] - b.normal[3 * i + 2]));
    if (e <= normalTol) ns++;
    nd = Math.max(nd, e); // NaN propagates
    const d = Math.abs(a.cost[i] - b.cost[i]);
    if (d <= costTol) cc++;
    cd += d;
  }
  return { depthSame: ds / n, normalSame: ns / n, maxNormalDiff: nd, costClose: cc / n, meanCostDiff: cd / n };
}

function signedMedianRelError(dm: DepthMap, gt: Float32Array, mask: Uint8Array): number {
  const rel: number[] = [];
  for (let i = 0; i < gt.length; i++) if (mask[i] && dm.depth[i] > 0 && gt[i] > 0) rel.push((dm.depth[i] - gt[i]) / gt[i]);
  rel.sort((x, y) => x - y);
  return rel.length ? rel[rel.length >> 1] : NaN;
}

const fmtP = (p: Parity): string =>
  `depth same ${(100 * p.depthSame).toFixed(2)}% | normal same ${(100 * p.normalSame).toFixed(2)}% (max |Δn| ${p.maxNormalDiff.toExponential(2)}) | cost close ${(100 * p.costClose).toFixed(2)}% | mean |Δcost| ${p.meanCostDiff.toExponential(2)}`;

describe('PatchMatchGPU adversarial', () => {
  it('random initialisation is identical to the CPU (PCG streams, inverse-depth sampling, normals)', async () => {
    const ds = makeSyntheticDataset('far', { width: 64, height: 48, numSources: 3, supersample: 1 });
    const opts = { iterations: 0, maxCost: 2 };
    const g = await pm.compute(ds.views[0], ds.views.slice(1), ds.range, opts);
    const c = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, opts);
    const p = parity(g, c, 1e-5, NORMAL_TOL, 0.02);
    log(`init-only parity: ${fmtP(p)}`);
    expect(allFinite(g)).toBe(true);
    expect(p.depthSame).toBeGreaterThan(0.999);
    expect(p.normalSame).toBeGreaterThan(0.999);
    expect(p.costClose).toBeGreaterThan(0.97);
  });

  it('odd image size, 1-row bands: bit-identical to unbanded, statistically equal to the CPU', async () => {
    const ds = makeSyntheticDataset('near', { width: 61, height: 45, numSources: 3, supersample: 1 });
    const a = await pm.compute(ds.views[0], ds.views.slice(1), ds.range, { iterations: 2 });
    const subA = pm.lastTimings!.submits;
    const b = await pm.compute(ds.views[0], ds.views.slice(1), ds.range, { iterations: 2, maxPixelViewsPerSubmit: 1 });
    const subB = pm.lastTimings!.submits;
    log(`odd 61x45: ${subA} submits vs ${subB} with 1-row bands`);
    expect(subB).toBe(45 * 5 + 1); // init + 2 × (red + black), one row each, + the finalisation
    expect(Array.from(b.depth)).toEqual(Array.from(a.depth));
    expect(Array.from(b.normal)).toEqual(Array.from(a.normal));
    expect(Array.from(b.cost)).toEqual(Array.from(a.cost));
    const c = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, { iterations: 2 });
    const sg = evaluateDepthMap(a, ds.gtDepth[0], ds.gtNormal[0]);
    const sc = evaluateDepthMap(c, ds.gtDepth[0], ds.gtNormal[0]);
    log(`  GPU: ${formatDepthStats(sg)}`);
    log(`  CPU: ${formatDepthStats(sc)}`);
    log(`  parity: ${fmtP(parity(a, c, 1e-4, 1e-3, 0.02))}`);
    // the last column / row exist and are processed (odd width: black pixels of the last column)
    let edgeValid = 0;
    for (let y = 0; y < 45; y++) if (a.depth[y * 61 + 60] > 0) edgeValid++;
    expect(edgeValid).toBeGreaterThan(10);
    expect(Math.abs(sg.validFraction - sc.validFraction)).toBeLessThan(0.05);
    expect(Math.abs(sg.accurateFraction - sc.accurateFraction)).toBeLessThan(0.08);
  });

  it('non-unrolled shader path (window with > 81 samples) matches the CPU', async () => {
    const ds = makeSyntheticDataset('near', { width: 64, height: 48, numSources: 3, supersample: 1 });
    const opts0 = { iterations: 0, maxCost: 2, patchRadius: 5, patchStep: 1 }; // 121 samples → loop code
    const warn = console.warn;
    console.warn = () => {};
    let g0: DepthMap;
    try {
      g0 = await pm.compute(ds.views[0], ds.views.slice(1), ds.range, opts0);
    } finally {
      console.warn = warn;
    }
    const c0 = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, opts0);
    const p0 = parity(g0, c0, 1e-5, NORMAL_TOL, 0.02);
    log(`loop path, init-only parity: ${fmtP(p0)}`);
    expect(p0.depthSame).toBeGreaterThan(0.999);
    expect(p0.costClose).toBeGreaterThan(0.97);
    const opts = { iterations: 2, patchRadius: 5, patchStep: 1 };
    const g = await pm.compute(ds.views[0], ds.views.slice(1), ds.range, opts);
    const c = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, opts);
    const sg = evaluateDepthMap(g, ds.gtDepth[0], ds.gtNormal[0]);
    const sc = evaluateDepthMap(c, ds.gtDepth[0], ds.gtNormal[0]);
    log(`  loop path GPU: ${formatDepthStats(sg)}`);
    log(`  loop path CPU: ${formatDepthStats(sc)}`);
    expect(Math.abs(sg.validFraction - sc.validFraction)).toBeLessThan(0.05);
    expect(Math.abs(sg.accurateFraction - sc.accurateFraction)).toBeLessThan(0.08);
  });

  it('no systematic depth bias on an analytic slanted plane (texture half-pixel conventions)', async () => {
    const tilt = (40 * Math.PI) / 180;
    const scene: SynthScene = {
      primitives: [
        {
          kind: 'quad',
          center: [0, 4, 0],
          axisU: [1, 0, 0],
          axisV: [0, Math.sin(tilt), Math.cos(tilt)],
          halfU: 20,
          halfV: 20,
          material: { scale: 0.6, octaves: 8, seed: 5, color: [0.7, 0.6, 0.5] },
        },
      ],
    };
    const W = 128;
    const H = 96;
    const cam = pinholeIntrinsics(W, H, 1.0);
    const eye = [0.2, 0, 0.3];
    const target = [0, 4, 0];
    const poses = [lookAt(eye, target)];
    for (let i = 0; i < 4; i++) {
      const a = (Math.PI / 2) * i + 0.4;
      poses.push(lookAt([eye[0] + 0.35 * Math.cos(a), eye[1], eye[2] + 0.25 * Math.sin(a)], target));
    }
    const ds = renderDataset(scene, cam, poses, { supersample: 2 });
    const g = await pm.compute(ds.views[0], ds.views.slice(1), ds.range);
    const mask = new Uint8Array(W * H);
    for (let y = 6; y < H - 6; y++) for (let x = 6; x < W - 6; x++) mask[y * W + x] = 1;
    const s = evaluateDepthMap(g, ds.gtDepth[0], ds.gtNormal[0], { mask, relTol: 0.005 });
    const bias = signedMedianRelError(g, ds.gtDepth[0], mask);
    log(`GPU slanted plane: ${formatDepthStats(s, 0.005)} | signed median ${(100 * bias).toFixed(4)}%`);
    expect(s.validFraction).toBeGreaterThan(0.97);
    expect(s.medianRelError).toBeLessThan(0.0025);
    expect(s.medianNormalErrorDeg).toBeLessThan(4);
    expect(Math.abs(bias)).toBeLessThan(0.0005);
  });

  it('sources at other resolutions than the reference; stale larger texture arrays do not leak into results', async () => {
    const scene = makeNearScene();
    const base = makeSyntheticDataset('near', { width: 64, height: 48, numSources: 3, supersample: 1 });
    const mixed: DenseView[] = [base.views[0]];
    const scales = [1.5, 0.75, 1.25];
    for (let i = 1; i < base.views.length; i++) {
      const camS = pinholeIntrinsics(Math.round(64 * scales[i - 1]), Math.round(48 * scales[i - 1]), (60 * Math.PI) / 180);
      mixed.push({ id: i, cam: camS, pose: base.views[i].pose, image: renderSynthetic(scene, camS, base.views[i].pose, { supersample: 1 }).image });
    }
    const opts0 = { iterations: 0, maxCost: 2 };
    const g0 = await pm.compute(mixed[0], mixed.slice(1), base.range, opts0);
    const c0 = patchMatchCPU(mixed[0], mixed.slice(1), base.range, opts0);
    const p0 = parity(g0, c0, 1e-5, NORMAL_TOL, 0.02);
    log(`mixed source sizes, init-only parity: ${fmtP(p0)}`);
    expect(p0.costClose).toBeGreaterThan(0.97);
    const g = await pm.compute(mixed[0], mixed.slice(1), base.range, { iterations: 3 });
    const c = patchMatchCPU(mixed[0], mixed.slice(1), base.range, { iterations: 3 });
    const sg = evaluateDepthMap(g, base.gtDepth[0], base.gtNormal[0]);
    const sc = evaluateDepthMap(c, base.gtDepth[0], base.gtNormal[0]);
    log(`  mixed GPU: ${formatDepthStats(sg)}`);
    log(`  mixed CPU: ${formatDepthStats(sc)}`);
    expect(Math.abs(sg.validFraction - sc.validFraction)).toBeLessThan(0.05);
    expect(Math.abs(sg.accurateFraction - sc.accurateFraction)).toBeLessThan(0.08);
    // now the texture array is larger than the next (same-size) dataset: compare
    // with a fresh instance (init-only costs must agree; full runs statistically)
    const fresh = await PatchMatchGPU.create(ctx!);
    try {
      const a0 = await pm.compute(base.views[0], base.views.slice(1), base.range, opts0);
      const b0 = await fresh.compute(base.views[0], base.views.slice(1), base.range, opts0);
      const q = parity(a0, b0, 1e-6, 1e-6, 1e-3);
      log(`  stale-texture instance vs fresh, init-only: ${fmtP(q)}`);
      expect(q.depthSame).toBe(1);
      expect(q.costClose).toBeGreaterThan(0.99);
    } finally {
      fresh.destroy();
    }
  });

  it('non-finite / degenerate geometric inputs never produce NaN on the GPU', async () => {
    const W = 48;
    const H = 36;
    const ds = makeSyntheticDataset('near', { width: W, height: H, numSources: 3, supersample: 1 });
    const n = W * H;
    const prior: DepthMap = { width: W, height: H, depth: new Float32Array(n), normal: new Float32Array(3 * n), cost: new Float32Array(n) };
    for (let i = 0; i < n; i++) {
      const k = i % 5;
      prior.depth[i] = k === 0 ? NaN : k === 1 ? Infinity : k === 2 ? -1 : 3;
      prior.normal.set(k === 3 ? [0, 0, 0] : k === 4 ? [NaN, 0, -1] : [0, 0, -1], 3 * i);
    }
    const junk: DepthMap = { width: 24, height: 18, depth: new Float32Array(24 * 18), normal: new Float32Array(3 * 24 * 18), cost: new Float32Array(24 * 18) };
    for (let i = 0; i < junk.depth.length; i++) junk.depth[i] = [NaN, Infinity, -Infinity, 0, 1e30, 2.5][i % 6];
    const none: DepthMap = { width: 0, height: 0, depth: new Float32Array(0), normal: new Float32Array(0), cost: new Float32Array(0) };
    const g = await pm.compute(ds.views[0], ds.views.slice(1), ds.range, { geometricIterations: 2 }, { prior, sourceDepths: [junk, none, junk] });
    expect(allFinite(g)).toBe(true);
    for (let i = 0; i < n; i++) {
      if (g.depth[i] > 0) {
        expect(g.depth[i]).toBeGreaterThanOrEqual(ds.range.min);
        expect(g.depth[i]).toBeLessThanOrEqual(ds.range.max);
      }
    }
    const flat: DenseView = { ...ds.views[0], image: { width: W, height: H, data: new Uint8Array(n * 4).fill(77) } };
    const f = await pm.compute(flat, ds.views.slice(1), ds.range, { iterations: 1 });
    expect(allFinite(f)).toBe(true);
    expect(f.depth.every((z) => z === 0)).toBe(true);
    expect(f.cost.every((c) => c === 2)).toBe(true);
  });

  it('prior-initialised photometric pass matches the CPU', async () => {
    const full = makeSyntheticDataset('near', { width: 96, height: 72, numSources: 3, supersample: 1 });
    const half = makeSyntheticDataset('near', { width: 48, height: 36, numSources: 3, supersample: 1 });
    const coarse = patchMatchCPU(half.views[0], half.views.slice(1), half.range);
    const opts = { iterations: 0, maxCost: 2, geometric: false };
    const g0 = await pm.compute(full.views[0], full.views.slice(1), full.range, opts, { prior: coarse, sourceDepths: [] });
    const c0 = patchMatchCPU(full.views[0], full.views.slice(1), full.range, opts, { prior: coarse, sourceDepths: [] });
    const p0 = parity(g0, c0, 1e-5, NORMAL_TOL, 0.02);
    log(`prior-initialised, init-only parity: ${fmtP(p0)}`);
    expect(p0.depthSame).toBeGreaterThan(0.999);
    const g = await pm.compute(full.views[0], full.views.slice(1), full.range, { iterations: 2, geometric: false }, { prior: coarse, sourceDepths: [] });
    const c = patchMatchCPU(full.views[0], full.views.slice(1), full.range, { iterations: 2, geometric: false }, { prior: coarse, sourceDepths: [] });
    const mask = surfaceMask(full.views[0].cam, full.gtDepth[0], full.gtNormal[0]);
    const sg = evaluateDepthMap(g, full.gtDepth[0], full.gtNormal[0], { mask });
    const sc = evaluateDepthMap(c, full.gtDepth[0], full.gtNormal[0], { mask });
    log(`  GPU: ${formatDepthStats(sg)}`);
    log(`  CPU: ${formatDepthStats(sc)}`);
    expect(Math.abs(sg.accurateFraction - sc.accurateFraction)).toBeLessThan(0.05);
  });

  it('selfTest() uses the instance options (non-default patch window / sigmas / seed)', async () => {
    const custom = await PatchMatchGPU.create(ctx!, { patchRadius: 3, patchStep: 1, sigmaColor: 6, seed: 9 });
    try {
      const r = await custom.selfTest();
      log(`selfTest (r=3, step 1, σc=6, seed 9): ${JSON.stringify(r)}`);
      expect(r.ok).toBe(true);
      expect(r.agreement).toBeGreaterThan(0.85);
    } finally {
      custom.destroy();
    }
  });

  it('device loss: compute() rejects with a clear error and never hangs', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no second adapter');
    const device = await adapter.requestDevice();
    const own: GPUContext = { ...ctx!, adapter, device, limits: device.limits, features: new Set() };
    const pmLost = await PatchMatchGPU.create(own);
    const ds = makeSyntheticDataset('near', { width: 48, height: 36, numSources: 2, supersample: 1 });
    const run = pmLost.compute(ds.views[0], ds.views.slice(1), ds.range, { iterations: 4, maxPixelViewsPerSubmit: 500 });
    // lose the device while the many small bands are being processed
    setTimeout(() => device.destroy(), 20);
    const t0 = performance.now();
    const err = await run.then(
      () => null,
      (e: unknown) => e as Error,
    );
    log(`device destroyed mid-run: ${err ? err.message : 'completed before the loss'} (${(performance.now() - t0).toFixed(0)} ms)`);
    if (err) expect(err.message).toMatch(/PatchMatchGPU/);
    await expect(pmLost.compute(ds.views[0], ds.views.slice(1), ds.range, { iterations: 1 })).rejects.toThrow(/device lost/);
    pmLost.destroy();
  });
});
