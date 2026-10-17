import { describe, expect, it } from 'vitest';
import { matchCPU } from '../../features/match';
import { estimateScanTimes } from './estimate';
import { makeMatchSets, pairOverlap, runBenchmark } from './runner';
import type { BenchRates, BenchTestResult } from './types';

describe('benchmark runner (Node: no WebGPU → CPU fallbacks)', () => {
  it('runs every test on the CPU without touching WebGPU modules, with checks and estimates', async () => {
    const seen: BenchTestResult[] = [];
    const progress: string[] = [];
    const rep = await runBenchmark({ scale: 'tiny', gpu: true, cancel: { cancelled: false }, onTest: (r) => seen.push(r), onProgress: (_id, m) => progress.push(m) });
    expect(rep.device.gpu.available).toBe(false);
    expect(rep.device.gpu.reason).toMatch(/not available/);
    expect(rep.cancelled).toBe(false);
    expect(rep.tests.map((t) => t.id)).toEqual(['sift', 'matcher', 'patchmatch', 'ba']);
    for (const t of rep.tests) {
      expect(t.backend).toBe('cpu');
      expect(t.error, `${t.id}: ${t.error}`).toBeUndefined();
      expect(t.status, `${t.id}: ${JSON.stringify(t.checks)}`).toBe('pass');
      expect(t.summary.length).toBeGreaterThan(10);
      expect(Object.keys(t.timings).length).toBeGreaterThan(0);
    }
    expect(rep.tests[0].note).toMatch(/CPU fallback measured/);
    // pending → running → final, per test
    expect(seen.filter((r) => r.id === 'ba').map((r) => r.status)).toEqual(['pending', 'running', 'pass']);
    expect(progress.length).toBeGreaterThan(4);
    expect(rep.rates.siftGpuMsPerMP).toBeNull();
    for (const k of ['siftCpuMsPerMP', 'matchCpuMsPerMPairs', 'pmCpuMsPerUnit', 'baMsPerIterPerKObs'] as const) expect(rep.rates[k]).toBeGreaterThan(0);
    expect(rep.estimates.map((e) => e.quality)).toEqual(['fast', 'balanced', 'max']);
    for (const e of rep.estimates) expect(e.features === 'cpu' && e.dense === 'cpu').toBe(true);
    expect(JSON.parse(JSON.stringify(rep))).toEqual(rep); // plain data (copyable report, postMessage)
  });

  it('cancel before the start marks every test cancelled and gives no estimates', async () => {
    const rep = await runBenchmark({ scale: 'tiny', gpu: false, cancel: { cancelled: true } });
    expect(rep.cancelled).toBe(true);
    expect(rep.tests.every((t) => t.status === 'cancelled')).toBe(true);
    expect(rep.estimates).toEqual([]);
    expect(rep.device.gpu.reason).toMatch(/disabled/);
  });

  it('synthetic descriptor sets: the CPU matcher recovers the true correspondences', () => {
    const { a, b, truth } = makeMatchSets(300, 2);
    const m = matchCPU(a, b);
    let correct = 0;
    for (let k = 0; k < m.length; k += 2) if (truth[m[k + 1]] === m[k]) correct++;
    expect(correct / (m.length / 2)).toBeGreaterThan(0.98);
    expect(correct).toBeGreaterThan(0.8 * 180);
    expect(pairOverlap(m, m)).toBe(1);
    expect(pairOverlap(m, m.subarray(0, m.length - 2))).toBeCloseTo((m.length / 2 - 1) / (m.length / 2), 9);
  });
});

describe('scan-time estimate', () => {
  const rates: BenchRates = {
    siftGpuMsPerMP: 40,
    siftCpuMsPerMP: 300,
    matchGpuMsPerMPairs: 2,
    matchCpuMsPerMPairs: 30,
    pmGpuMsPerUnit: 20,
    pmCpuMsPerUnit: 4000,
    baMsPerIterPerKObs: 2,
  };
  it('grows with the quality preset and the number of keyframes', () => {
    const e = estimateScanTimes(rates);
    expect(e.map((x) => x.quality)).toEqual(['fast', 'balanced', 'max']);
    for (let i = 1; i < 3; i++) {
      expect(e[i].frameMs).toBeGreaterThan(e[i - 1].frameMs);
      expect(e[i].denseMs).toBeGreaterThan(e[i - 1].denseMs);
      expect(e[i].processingMs).toBeGreaterThan(e[i - 1].processingMs);
    }
    for (const x of e) {
      expect(x.processingMs).toBe(x.refineMs + x.denseMs + x.meshMs);
      expect(x.captureFps).toBeLessThanOrEqual(10);
      expect(x.features).toBe('gpu');
      expect(x.dense).toBe('gpu');
    }
    const big = estimateScanTimes(rates, { keyframes: 160 });
    expect(big[1].refineMs).toBeGreaterThan(e[1].refineMs);
    expect(big[1].keyframes).toBe(160);
  });
  it('uses the CPU fallbacks without GPU rates, and needs every measurement', () => {
    const cpu = estimateScanTimes({ ...rates, siftGpuMsPerMP: null, matchGpuMsPerMPairs: null, pmGpuMsPerUnit: null });
    const gpu = estimateScanTimes(rates);
    expect(cpu[1].features).toBe('cpu');
    expect(cpu[1].dense).toBe('cpu');
    expect(cpu[1].frameMs).toBeGreaterThan(gpu[1].frameMs);
    expect(estimateScanTimes({ ...rates, baMsPerIterPerKObs: null })).toEqual([]);
    expect(estimateScanTimes({ ...rates, pmGpuMsPerUnit: null, pmCpuMsPerUnit: null })).toEqual([]);
  });
});
