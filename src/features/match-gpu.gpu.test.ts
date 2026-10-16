// WebGPU matcher vs matchCPU / matchGuidedCPU (headless Chromium + SwiftShader).
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import type { FeatureSet, Mat3, MatchPairs } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';
import { getGPU } from '../gpu/device';
import type { GPUContext } from '../gpu/device';
import { matchCPU, matchGuidedCPU } from './match';
import type { MatchGuide, MatchOptions } from './match';
import { GPUWorkDroppedError } from './gpu-common';
import { MatcherGPU } from './match-gpu';
import { extractSiftCPU } from './sift-cpu';
import { TestRng, makeWarpedPair, similarityAboutCenter, tiltHomography } from './testdata';

const log = (...a: unknown[]): void => console.log('[match-gpu]', ...a);
const D = DESCRIPTOR_SIZE;

/** Random RootSIFT-like descriptors; b holds noisy copies of a permutation of a (as match.test.ts). */
function randomSets(na: number, nb: number, noise: number, seed: number, size = 1000): { a: FeatureSet; b: FeatureSet } {
  const rng = new TestRng(seed);
  const mk = (n: number): FeatureSet => ({
    count: n,
    keypoints: new Float32Array(4 * n),
    descriptors: new Uint8Array(D * n),
    scores: new Float32Array(n),
    width: size,
    height: size,
  });
  const a = mk(na);
  const b = mk(nb);
  const v = new Float64Array(D);
  const quant = (out: Uint8Array, o: number) => {
    let ss = 0;
    for (let k = 0; k < D; k++) ss += v[k] * v[k];
    const inv = 1 / Math.sqrt(ss);
    for (let k = 0; k < D; k++) out[o + k] = Math.min(255, Math.round(512 * v[k] * inv));
  };
  for (let i = 0; i < na; i++) {
    for (let k = 0; k < D; k++) v[k] = rng.next() < 0.3 ? 0 : rng.next();
    quant(a.descriptors, i * D);
    a.keypoints[4 * i] = rng.uniform(0, size);
    a.keypoints[4 * i + 1] = rng.uniform(0, size);
  }
  for (let j = 0; j < nb; j++) {
    const src = j < na ? (j * 7919) % na : -1;
    if (src >= 0 && j < Math.floor(nb * 0.7)) {
      for (let k = 0; k < D; k++) v[k] = Math.max(0, a.descriptors[src * D + k] / 512 + noise * rng.gaussian());
      b.keypoints[4 * j] = a.keypoints[4 * src] + rng.uniform(-1, 1);
      b.keypoints[4 * j + 1] = a.keypoints[4 * src + 1] + rng.uniform(-1, 1);
    } else {
      for (let k = 0; k < D; k++) v[k] = rng.next() < 0.3 ? 0 : rng.next();
      b.keypoints[4 * j] = rng.uniform(0, size);
      b.keypoints[4 * j + 1] = rng.uniform(0, size);
    }
    quant(b.descriptors, j * D);
  }
  return { a, b };
}

function overlap(x: MatchPairs, y: MatchPairs): { common: number; ratio: number; identical: boolean } {
  const s = new Set<number>();
  for (let k = 0; k < x.length; k += 2) s.add(x[k] * 1048576 + x[k + 1]);
  let common = 0;
  for (let k = 0; k < y.length; k += 2) if (s.has(y[k] * 1048576 + y[k + 1])) common++;
  const n = Math.max(x.length, y.length) / 2;
  const identical = x.length === y.length && x.every((v, i) => v === y[i]);
  return { common, ratio: n ? common / n : 1, identical };
}

function sortedByA(m: MatchPairs): boolean {
  for (let k = 2; k < m.length; k += 2) if (m[k] <= m[k - 2]) return false;
  return true;
}

let ctx: GPUContext;
let matcher: MatcherGPU;

beforeAll(async () => {
  const c = await getGPU();
  if (!c) throw new Error('WebGPU unavailable in the test browser');
  ctx = c;
  const t0 = performance.now();
  matcher = await MatcherGPU.create(ctx);
  log(`create ${(performance.now() - t0).toFixed(0)} ms`);
});

afterAll(() => matcher?.destroy());

async function expectSame(name: string, gpu: Promise<MatchPairs>, cpu: MatchPairs): Promise<void> {
  const t0 = performance.now();
  const g = await gpu;
  const ms = performance.now() - t0;
  const o = overlap(cpu, g);
  const T = matcher.lastTimings;
  log(
    `${name}: GPU ${g.length / 2} / CPU ${cpu.length / 2} pairs, overlap ${(100 * o.ratio).toFixed(2)}%${o.identical ? ' (identical)' : ''}; ` +
      `${ms.toFixed(0)} ms (gpu ${T?.gpuMs.toFixed(0)}, submits ${T?.submits})`,
  );
  // The final decision is exact (f64 on the host) and the GPU top-2 ranking is deterministic
  // (exact integer dots, one correctly rounded multiply): require identical pair arrays.
  expect(o.ratio).toBeGreaterThanOrEqual(0.995);
  expect(o.identical).toBe(true);
  expect(sortedByA(g)).toBe(true);
}

describe('MatcherGPU.match (brute force)', () => {
  it('equals matchCPU on random descriptors for several option sets and sizes', async () => {
    for (const [na, nb, seed] of [
      [300, 330, 1],
      [1, 70, 2],
      [65, 1, 3],
      [129, 200, 4],
    ]) {
      const { a, b } = randomSets(na, nb, 0.1, seed);
      for (const opts of [{}, { mutual: false }, { ratio: 0.95, maxDistance: 1.4 }, { ratio: 0.6, maxDistance: 0.4 }] as MatchOptions[]) {
        await expectSame(`random ${na}x${nb} ${JSON.stringify(opts)}`, matcher.match(a, b, opts), matchCPU(a, b, opts));
      }
    }
  });

  it('equals matchCPU on real SIFT features of a warped pair (both directions, self-match)', async () => {
    const W = 640;
    const H = 480;
    const p = makeWarpedPair(W, H, similarityAboutCenter(W, H, 20, 0.85), 3, { noiseSigma: 0.01 });
    const a = extractSiftCPU(p.a);
    const b = extractSiftCPU(p.b);
    await expectSame(`sift ${a.count}x${b.count}`, matcher.match(a, b), matchCPU(a, b));
    await expectSame(`sift ${b.count}x${a.count}`, matcher.match(b, a), matchCPU(b, a));
    await expectSame(`sift self`, matcher.match(a, a, { ratio: 0.95 }), matchCPU(a, a, { ratio: 0.95 }));
  });

  it('handles empty sets and re-uploads descriptors refilled in place', async () => {
    const { a, b } = randomSets(60, 70, 0.08, 15);
    const empty: FeatureSet = { count: 0, keypoints: new Float32Array(0), descriptors: new Uint8Array(0), scores: new Float32Array(0), width: 10, height: 10 };
    expect((await matcher.match(a, empty)).length).toBe(0);
    expect((await matcher.match(empty, b)).length).toBe(0);
    const opts = { ratio: 0.95, maxDistance: 1.4 };
    await matcher.match(a, b, opts); // warm the cache
    const other = randomSets(60, 70, 0.08, 16).b;
    for (let j = 0; j < other.count; j += 2) for (let k = 0; k < D; k++) other.descriptors[j * D + k] = Math.round(other.descriptors[j * D + k] * (j % 4 ? 0.3 : 0));
    b.descriptors.set(other.descriptors);
    const fresh: FeatureSet = { ...b, descriptors: new Uint8Array(b.descriptors) };
    const got = await matcher.match(a, b, opts);
    expect(Array.from(got)).toEqual(Array.from(matchCPU(a, fresh, opts)));
  });

  it('keeps the descriptor cache within its byte budget', async () => {
    const m = await MatcherGPU.create(ctx, { cacheBytes: 200 * 1024 });
    try {
      const sets = Array.from({ length: 6 }, (_, i) => randomSets(500, 10, 0.1, 100 + i).a);
      for (let i = 1; i < sets.length; i++) {
        const r = await m.match(sets[i - 1], sets[i]);
        expect(Array.from(r)).toEqual(Array.from(matchCPU(sets[i - 1], sets[i])));
        expect(m.cacheMemoryBytes()).toBeLessThanOrEqual(200 * 1024);
        expect(m.cacheMemoryBytes()).toBeGreaterThanOrEqual(2 * 500 * 128);
      }
      m.clearCache();
      expect(m.cacheMemoryBytes()).toBe(0);
    } finally {
      m.destroy();
    }
  });

  it('selfTest passes; dropped GPU work is detected and redone', async () => {
    const r = await matcher.selfTest();
    log(`selfTest: ${JSON.stringify(r)}`);
    expect(r.ok).toBe(true);
    const { a, b } = randomSets(200, 210, 0.1, 31);
    const m = await MatcherGPU.create(ctx, { testDropSubmissions: (attempt) => (attempt === 0 ? [0] : []) });
    try {
      expect(Array.from(await m.match(a, b))).toEqual(Array.from(matchCPU(a, b)));
      expect(m.lastTimings!.attempts).toBe(2);
    } finally {
      m.destroy();
    }
    const always = await MatcherGPU.create(ctx, { testDropSubmissions: () => [0] });
    await expect(always.match(a, b)).rejects.toBeInstanceOf(GPUWorkDroppedError);
    always.destroy();
  });

  it('timing: 2000 x 2000 random descriptors (informational on SwiftShader)', async () => {
    const { a, b } = randomSets(2000, 2000, 0.1, 77);
    const t0 = performance.now();
    const cpu = matchCPU(a, b);
    const tc = performance.now() - t0;
    await expectSame('2000x2000', matcher.match(a, b), cpu);
    log(`  CPU matchCPU ${tc.toFixed(0)} ms`);
  });
});

describe('MatcherGPU.matchGuided', () => {
  it('positions guide (with NaN = unconstrained, Infinity, out-of-frame) equals matchGuidedCPU', async () => {
    const { a, b } = randomSets(400, 420, 0.12, 8);
    const pred = new Float32Array(2 * a.count);
    for (let i = 0; i < a.count; i++) {
      pred[2 * i] = a.keypoints[4 * i] + 3;
      pred[2 * i + 1] = a.keypoints[4 * i + 1] - 2;
    }
    pred[2 * 5] = NaN;
    pred[2 * 6 + 1] = NaN;
    pred[2 * 7] = Infinity;
    pred[2 * 8] = -500;
    for (const R of [10, 40, 150]) {
      const g: MatchGuide = { kind: 'positions', predicted: pred, radiusPx: R };
      for (const opts of [{}, { mutual: false }, { ratio: 0.95, maxDistance: 1.4 }] as MatchOptions[]) {
        await expectSame(`positions r=${R} ${JSON.stringify(opts)}`, matcher.matchGuided(a, b, g, opts), matchGuidedCPU(a, b, g, opts));
      }
    }
    const allNaN: MatchGuide = { kind: 'positions', predicted: new Float32Array(2 * a.count).fill(NaN), radiusPx: 5 };
    await expectSame('positions all NaN', matcher.matchGuided(a, b, allNaN), matchCPU(a, b));
  });

  it('H guide equals matchGuidedCPU and is invariant to the sign / scale of H', async () => {
    const { a, b } = randomSets(300, 330, 0.1, 13);
    const H = new Float64Array([0.98, -0.05, 12, 0.04, 1.01, -9, 2e-5, -1e-5, 1]);
    const ref = matchGuidedCPU(a, b, { kind: 'H', H, maxErrorPx: 30 });
    expect(ref.length).toBeGreaterThan(40);
    for (const s of [1, -1, -1e-4, 3e5]) {
      const Hs = H.map((v) => v * s);
      await expectSame(`H x${s}`, matcher.matchGuided(a, b, { kind: 'H', H: Hs, maxErrorPx: 30 }), ref);
    }
  });

  it('F guide (epipole at infinity and inside the image) equals matchGuidedCPU', async () => {
    const { a, b } = randomSets(350, 360, 0.12, 9);
    const F1 = new Float64Array([0, 0, 0.2, 0, 0, -1, -0.2, 1, 0]);
    const e = [500.3, 400.7, 1];
    const ex = [0, -e[2], e[1], e[2], 0, -e[0], -e[1], e[0], 0];
    const A = [0.98, -0.17, 15, 0.17, 0.98, -7, 1e-5, -2e-5, 1];
    const F2 = new Float64Array(9);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) F2[3 * r + c] = ex[3 * r] * A[c] + ex[3 * r + 1] * A[3 + c] + ex[3 * r + 2] * A[6 + c];
    for (const [name, F] of [
      ['F inf', F1],
      ['F finite', F2],
    ] as [string, Mat3][]) {
      for (const tol of [2, 25]) {
        for (const opts of [{}, { ratio: 0.95, maxDistance: 1.4, mutual: false }] as MatchOptions[]) {
          const g: MatchGuide = { kind: 'F', F, maxErrorPx: tol };
          await expectSame(`${name} tol=${tol} ${JSON.stringify(opts)}`, matcher.matchGuided(a, b, g, opts), matchGuidedCPU(a, b, g, opts));
        }
      }
    }
  });

  it('guided matching of real features equals the CPU (H and positions from a known warp)', async () => {
    const W = 640;
    const H = 480;
    for (const [name, Hm] of [
      ['rotation 25°', similarityAboutCenter(W, H, 25, 0.9)],
      ['tilt 35°', tiltHomography(W, H, 35, -10)],
    ] as [string, Mat3][]) {
      const p = makeWarpedPair(W, H, Hm, 23, { noiseSigma: 0.01 });
      const a = extractSiftCPU(p.a);
      const b = extractSiftCPU(p.b);
      const gH: MatchGuide = { kind: 'H', H: Hm, maxErrorPx: 8 };
      await expectSame(`${name} H-guided`, matcher.matchGuided(a, b, gH), matchGuidedCPU(a, b, gH));
      const t0 = performance.now();
      const plain = await matcher.match(a, b);
      const tPlain = performance.now() - t0;
      const t1 = performance.now();
      await matcher.matchGuided(a, b, gH);
      const tGuided = performance.now() - t1;
      log(`  ${name}: plain ${plain.length / 2} pairs ${tPlain.toFixed(0)} ms vs H-guided ${tGuided.toFixed(0)} ms (tile culling)`);
    }
  });
});
