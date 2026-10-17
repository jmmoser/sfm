/**
 * Memory accounting / dense memory plan (src/engine/memory.ts): the plans of the iPhone
 * scenarios fit the tab budget, the caps only bind where the full-size presets need them, the
 * device budget follows the adapter's RAM proxy, and the capture estimate counts the typed arrays
 * exactly.
 */
import { describe, expect, it } from 'vitest';
import type { FeatureSet, FrameInfo, TwoViewGeometry } from '../core/types';
import { CorrespondenceGraph } from '../sfm/correspondence-graph';
import { FeatureDatabase } from '../sfm/database';
import { MB, MEM, captureMemory, deviceMemoryBudgetMB, formatBreakdown, planDense, sumBreakdown, type DensePlanInput } from './memory';
import { resolveDensePreset } from './presets';

function denseInput(quality: 'fast' | 'balanced' | 'max', views: number, width: number, height: number, budgetMB = 1200, residentBytes = 20 * MB): DensePlanInput {
  const p = resolveDensePreset(quality);
  return {
    budgetMB,
    residentBytes,
    views,
    width,
    height,
    numSources: p.numSources,
    geometricPasses: p.geometricPasses,
    fusionMaxNeighbours: p.fusionMaxNeighbours,
    gpu: true,
    haloFilter: p.haloFilter !== false,
    tsdfMaxBlocks: p.tsdfMaxBlocks,
  };
}

describe('planDense', () => {
  it('max: 150 views at 1600×1200 fit 1.2 GB with a coarser fusion step', () => {
    const plan = planDense(denseInput('max', 150, 1600, 1200));
    expect(plan.fits).toBe(true);
    expect(plan.peakBytes).toBeLessThanOrEqual(1200 * MB);
    // ≈ 0.2 × 150 × 1.92 MP = 58 M points would not fit: a pixel step bounds the cloud
    expect(plan.fusionStep).toBeGreaterThanOrEqual(3);
    expect(plan.fusionStep).toBeLessThanOrEqual(6);
    expect(plan.expectedPoints).toBeLessThanOrEqual(plan.fusionMaxPoints);
    expect(plan.fusionMaxPoints).toBeGreaterThan(1_000_000);
    // geometric sources need ≥ numSources + 1 depth-only maps, images ≥ numSources + 2
    expect(plan.sourceDepthCacheBytes).toBeGreaterThanOrEqual(9 * 4 * 1600 * 1200);
    expect(plan.imageCacheBytes).toBeGreaterThanOrEqual(10 * 4 * 1600 * 1200);
    for (const s of plan.stages) expect(s.bytes + MEM.reserve + 20 * MB).toBeLessThanOrEqual(1200 * MB);
  });

  it('balanced: 80 views at 1280×960 → fusion step 2; fast at 800 px and the e2e 480 px runs are unchanged (step 1)', () => {
    const bal = planDense(denseInput('balanced', 80, 1280, 960));
    expect(bal.fits).toBe(true);
    expect(bal.fusionStep).toBe(2);
    const fast = planDense(denseInput('fast', 40, 800, 600));
    expect(fast.fusionStep).toBe(1);
    expect(fast.fusionMaxNeighbours).toBe(resolveDensePreset('fast').fusionMaxNeighbours);
    const e2e = planDense(denseInput('fast', 40, 360, 480));
    expect(e2e.fusionStep).toBe(1);
    expect(e2e.fits).toBe(true);
  });

  it('a 6 GB budget (850 MB) still fits max with fewer neighbours / a coarser step; an explicit step wins', () => {
    const plan = planDense(denseInput('max', 150, 1600, 1200, 850));
    expect(plan.fits).toBe(true);
    expect(plan.fusionMaxNeighbours).toBeLessThanOrEqual(8);
    expect(plan.fusionMaxNeighbours).toBeGreaterThanOrEqual(4);
    const fixed = planDense({ ...denseInput('max', 150, 1600, 1200), fusionStep: 2 });
    expect(fixed.fusionStep).toBe(2);
  });

  it('formats a breakdown largest first', () => {
    expect(formatBreakdown({ a: 3 * MB, b: 10 * MB, c: 1000 })).toBe('13 MB (b 10, a 3)');
  });
});

describe('deviceMemoryBudgetMB', () => {
  const apple = (mib: number) => ({ info: { vendor: 'apple', architecture: '', description: '' }, limits: { maxBufferSize: mib * MB } });
  it('maps the Apple maxBufferSize RAM proxy to the iPhone budget classes', () => {
    expect(deviceMemoryBudgetMB(apple(1024))!.budgetMB).toBe(1200); // 12 GB (26.2+ cap)
    expect(deviceMemoryBudgetMB(apple(650))!.budgetMB).toBe(1200); // 8 GB
    expect(deviceMemoryBudgetMB(apple(470))!.budgetMB).toBe(850); // 6 GB
    expect(deviceMemoryBudgetMB(apple(330))!.budgetMB).toBe(600); // 4 GB
    expect(deviceMemoryBudgetMB(apple(256))!.budgetMB).toBe(850); // 26.0–26.1 floor: unknown RAM
  });
  it('leaves other adapters to the default', () => {
    expect(deviceMemoryBudgetMB({ info: { vendor: 'google', architecture: 'swiftshader', description: '' }, limits: { maxBufferSize: 256 * MB } })).toBeNull();
    expect(deviceMemoryBudgetMB(null)).toBeNull();
  });
});

describe('captureMemory', () => {
  it('counts features, colours, pairs and the graph from the data structures', () => {
    const db = new FeatureDatabase();
    const graph = new CorrespondenceGraph();
    const n = 1000;
    const fs = (): FeatureSet => ({
      count: n,
      keypoints: new Float32Array(4 * n),
      descriptors: new Uint8Array(128 * n),
      scores: new Float32Array(n),
      width: 640,
      height: 480,
    });
    for (let id = 0; id < 3; id++) {
      const info = { id, t: id, width: 640, height: 480 } as unknown as FrameInfo;
      db.addFrame(info, fs(), new Uint8Array(3 * n));
      graph.addImage(id, n);
    }
    const inl = new Uint32Array(2 * 500);
    for (let k = 0; k < 500; k++) {
      inl[2 * k] = k;
      inl[2 * k + 1] = k;
    }
    const g: TwoViewGeometry = { config: 'calibrated', inliers: inl };
    db.setTwoView(0, 1, g);
    graph.addTwoView(0, 1, inl);
    const b = captureMemory({ db, graph, rec: null, vocab: null, imuSamples: 600, pendingWriteBytes: 5 * MB });
    expect(b.features).toBe(3 * n * (16 + 128 + 4));
    expect(b.colours).toBe(3 * 3 * n);
    expect(b.pairs).toBe(MEM.twoViewFixed + inl.byteLength);
    expect(b.graph).toBeGreaterThanOrEqual(3 * 6 * n + 8 * 500 + 12 * 1000);
    expect(b.imu).toBe(600 * MEM.imuSample);
    expect(b.pendingWrites).toBe(5 * MB);
    expect(sumBreakdown(b)).toBeGreaterThan(b.features);
    expect(b.rsOriginal).toBeUndefined();
    // rolling-shutter originals (a copy of the keypoints of every corrected keyframe) and the
    // best-of-window held frame are resident too
    const rsOriginal = new Map([...db.features].map(([id, f]) => [id, f.keypoints.slice()] as [number, Float32Array]));
    const b2 = captureMemory({ db, graph, rec: null, vocab: null, imuSamples: 600, rsOriginal, heldFrameBytes: 3 * MB });
    expect(b2.rsOriginal).toBe(3 * (16 * n + 16));
    expect(b2.heldFrame).toBe(3 * MB);
    expect(sumBreakdown(b2)).toBe(sumBreakdown(b) - 5 * MB + b2.rsOriginal + 3 * MB);
  });
});
