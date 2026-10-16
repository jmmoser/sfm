// Runs the CPU feature path inside headless Chromium (vitest "gpu" project) to check
// browser compatibility: synchronous WASM SIMD compilation on the main thread (Chrome
// only allows sync compiles of modules < 4 KB there), and the browser-safe test helpers
// that the GPU feature tests reuse.
import { describe, expect, it } from 'vitest';
import { matchCPU, matchUsesSIMD } from './match';
import { extractSiftCPU, lastSiftCPUTimings } from './sift-cpu';
import { simdAvailable, simdModuleBytes } from './simd';
import { evaluateMatches, makeWarpedPair, similarityAboutCenter } from './testdata';

describe('CPU features in the browser', () => {
  it('compiles the SIMD kernels synchronously on the main thread', () => {
    expect(simdModuleBytes().length).toBeLessThan(4096);
    expect(simdAvailable()).toBe(true);
    expect(matchUsesSIMD()).toBe(true);
  });

  it('extracts and matches a warped pair', () => {
    const W = 320;
    const H = 240;
    const p = makeWarpedPair(W, H, similarityAboutCenter(W, H, 30, 0.9), 5);
    const t0 = performance.now();
    const a = extractSiftCPU(p.a);
    const simd = lastSiftCPUTimings.simd;
    const b = extractSiftCPU(p.b);
    const m = matchCPU(a, b);
    const ev = evaluateMatches(a, b, m, p.H);
    console.log(`[browser cpu] ${a.count}/${b.count} feats, ${ev.correct}/${ev.total} correct, ${(performance.now() - t0).toFixed(0)} ms`);
    expect(simd).toBe(1);
    expect(ev.precision).toBeGreaterThan(0.9);
    expect(ev.correct).toBeGreaterThan(200);
  });
});
