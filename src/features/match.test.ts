import { describe, expect, it } from 'vitest';
import type { FeatureSet, Mat3 } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';
import { descriptorDistance, matchCPU, matchGuidedCPU, matchUsesSIMD, unitDescriptors } from './match';
import type { MatchOptions } from './match';
import { extractSiftCPU } from './sift-cpu';
import {
  TestRng,
  applyHomography,
  evaluateMatches,
  makeWarpedPair,
  mat3MulH,
  similarityAboutCenter,
  tiltHomography,
} from './testdata';

const D = DESCRIPTOR_SIZE;

/** Random RootSIFT-like descriptors; features of b are noisy copies of a permutation of a. */
function randomSets(na: number, nb: number, noise: number, seed: number): { a: FeatureSet; b: FeatureSet; truth: Int32Array } {
  const rng = new TestRng(seed);
  const mk = (n: number): FeatureSet => ({
    count: n,
    keypoints: new Float32Array(4 * n),
    descriptors: new Uint8Array(D * n),
    scores: new Float32Array(n),
    width: 1000,
    height: 1000,
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
    a.keypoints[4 * i] = rng.uniform(0, 1000);
    a.keypoints[4 * i + 1] = rng.uniform(0, 1000);
  }
  const truth = new Int32Array(na).fill(-1);
  for (let j = 0; j < nb; j++) {
    const src = j < na ? (j * 7919) % na : -1; // permutation for the first na
    if (src >= 0 && j < Math.floor(nb * 0.7)) {
      for (let k = 0; k < D; k++) v[k] = Math.max(0, a.descriptors[src * D + k] / 512 + noise * rng.gaussian());
      truth[src] = j;
      b.keypoints[4 * j] = a.keypoints[4 * src] + rng.uniform(-1, 1);
      b.keypoints[4 * j + 1] = a.keypoints[4 * src + 1] + rng.uniform(-1, 1);
    } else {
      for (let k = 0; k < D; k++) v[k] = rng.next() < 0.3 ? 0 : rng.next();
      b.keypoints[4 * j] = rng.uniform(0, 1000);
      b.keypoints[4 * j + 1] = rng.uniform(0, 1000);
    }
    quant(b.descriptors, j * D);
  }
  return { a, b, truth };
}

/** Straightforward reference implementation of the documented semantics. */
function naiveMatch(a: FeatureSet, b: FeatureSet, opts: MatchOptions = {}, cand?: (i: number, j: number) => boolean): number[] {
  const ratio = opts.ratio ?? 0.8;
  const mutual = opts.mutual ?? true;
  const maxD = opts.maxDistance ?? 0.7;
  const dist = (i: number, j: number) => descriptorDistance(a, i, b, j);
  const ok = cand ?? (() => true);
  const out: number[] = [];
  for (let i = 0; i < a.count; i++) {
    let d1 = Infinity;
    let d2 = Infinity;
    let bj = -1;
    for (let j = 0; j < b.count; j++) {
      if (!ok(i, j)) continue;
      const d = dist(i, j);
      if (d < d1) {
        d2 = d1;
        d1 = d;
        bj = j;
      } else if (d < d2) d2 = d;
    }
    if (bj < 0 || d1 > maxD) continue;
    if (d2 < Infinity && !(d1 < ratio * d2)) continue;
    if (mutual) {
      let bi = -1;
      let bd = Infinity;
      for (let k = 0; k < a.count; k++) {
        if (!ok(k, bj)) continue;
        const d = dist(k, bj);
        if (d < bd) {
          bd = d;
          bi = k;
        }
      }
      if (bi !== i) continue;
    }
    out.push(i, bj);
  }
  return out;
}

describe('unitDescriptors', () => {
  it('normalises and caches', () => {
    const { a } = randomSets(10, 10, 0.05, 1);
    const u = unitDescriptors(a);
    expect(unitDescriptors(a)).toBe(u);
    for (let i = 0; i < a.count; i++) {
      let ss = 0;
      for (let k = 0; k < D; k++) ss += u[i * D + k] ** 2;
      expect(Math.sqrt(ss)).toBeCloseTo(1, 5);
    }
    expect(descriptorDistance(a, 3, a, 3)).toBeLessThan(1e-3);
  });
});

describe('matchCPU', () => {
  it('agrees with a naive reference implementation', () => {
    for (const [na, nb, noise, seed] of [
      [37, 53, 0.08, 2],
      [64, 61, 0.15, 3],
      [5, 3, 0.05, 4],
      [101, 1, 0.05, 5],
    ] as const) {
      const { a, b } = randomSets(na, nb, noise, seed);
      for (const opts of [{}, { mutual: false }, { ratio: 0.95, maxDistance: 1.2 }]) {
        expect(Array.from(matchCPU(a, b, opts))).toEqual(naiveMatch(a, b, opts));
      }
    }
  });

  it('finds the planted correspondences, sorted by a index', () => {
    const { a, b, truth } = randomSets(400, 500, 0.05, 6);
    const m = matchCPU(a, b);
    let correct = 0;
    for (let k = 0; k < m.length; k += 2) {
      if (k > 0) expect(m[k]).toBeGreaterThan(m[k - 2]);
      if (truth[m[k]] === m[k + 1]) correct++;
    }
    expect(m.length / 2).toBeGreaterThan(250);
    expect(correct / (m.length / 2)).toBeGreaterThan(0.98);
  });

  it('handles empty sets', () => {
    const { a } = randomSets(5, 5, 0.1, 7);
    const empty: FeatureSet = { ...a, count: 0, keypoints: new Float32Array(0), descriptors: new Uint8Array(0), scores: new Float32Array(0) };
    expect(matchCPU(a, empty).length).toBe(0);
    expect(matchCPU(empty, a).length).toBe(0);
    expect(matchGuidedCPU(a, empty, { kind: 'H', H: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), maxErrorPx: 3 }).length).toBe(0);
  });
});

describe('matchGuidedCPU', () => {
  it('positions guide agrees with a naive guided reference; NaN = unconstrained', () => {
    const { a, b } = randomSets(80, 90, 0.12, 8);
    const pred = new Float32Array(2 * a.count);
    for (let i = 0; i < a.count; i++) {
      pred[2 * i] = a.keypoints[4 * i] + 3;
      pred[2 * i + 1] = a.keypoints[4 * i + 1] - 2;
    }
    pred[2 * 5] = NaN;
    pred[2 * 6 + 1] = NaN;
    const R = 40;
    const cand = (i: number, j: number) => {
      const px = pred[2 * i];
      const py = pred[2 * i + 1];
      if (Number.isNaN(px) || Number.isNaN(py)) return true;
      return (b.keypoints[4 * j] - px) ** 2 + (b.keypoints[4 * j + 1] - py) ** 2 <= R * R;
    };
    for (const opts of [{}, { mutual: false }]) {
      const got = Array.from(matchGuidedCPU(a, b, { kind: 'positions', predicted: pred, radiusPx: R }, opts));
      expect(got).toEqual(naiveMatch(a, b, opts, cand));
    }
    const allNaN = new Float32Array(2 * a.count).fill(NaN);
    expect(Array.from(matchGuidedCPU(a, b, { kind: 'positions', predicted: allNaN, radiusPx: 5 }))).toEqual(
      Array.from(matchCPU(a, b)),
    );
  });

  it('F guide agrees with a naive epipolar-band reference', () => {
    const { a, b } = randomSets(70, 75, 0.12, 9);
    // F = [e]x for e = (1, 0.2, 0) (epipole at infinity): lines through x_a's direction.
    const F = new Float64Array([0, 0, 0.2, 0, 0, -1, -0.2, 1, 0]);
    const tol = 25;
    const cand = (i: number, j: number) => {
      const xa = a.keypoints[4 * i];
      const ya = a.keypoints[4 * i + 1];
      const l = [F[0] * xa + F[1] * ya + F[2], F[3] * xa + F[4] * ya + F[5], F[6] * xa + F[7] * ya + F[8]];
      const e = l[0] * b.keypoints[4 * j] + l[1] * b.keypoints[4 * j + 1] + l[2];
      return Math.abs(e) / Math.hypot(l[0], l[1]) <= tol;
    };
    const got = Array.from(matchGuidedCPU(a, b, { kind: 'F', F, maxErrorPx: tol }));
    expect(got).toEqual(naiveMatch(a, b, {}, cand));
  });

  // Real features on a warped pair: guided matching must find at least as many
  // correct matches as unguided matching.
  const W = 640;
  const H = 480;
  for (const [name, Hm] of [
    ['rotation 25°', similarityAboutCenter(W, H, 25, 0.9)],
    ['tilt 35°', tiltHomography(W, H, 35, -10)],
  ] as [string, Mat3][]) {
    it(`guided >= unguided correct matches (${name})`, () => {
      const p = makeWarpedPair(W, H, Hm, 23, { noiseSigma: 0.01 });
      const a = extractSiftCPU(p.a);
      const b = extractSiftCPU(p.b);
      const un = evaluateMatches(a, b, matchCPU(a, b), p.H);
      const gH = evaluateMatches(a, b, matchGuidedCPU(a, b, { kind: 'H', H: p.H, maxErrorPx: 4 }), p.H);
      // Predicted positions from a slightly perturbed homography (e.g. gyro prediction).
      const Hp = mat3MulH(similarityAboutCenter(W, H, 0.5, 1.005, 3, -2), p.H);
      const pred = new Float32Array(2 * a.count);
      const q = [0, 0];
      for (let i = 0; i < a.count; i++) {
        applyHomography(Hp, a.keypoints[4 * i], a.keypoints[4 * i + 1], q);
        pred[2 * i] = q[0];
        pred[2 * i + 1] = q[1];
      }
      const gP = evaluateMatches(a, b, matchGuidedCPU(a, b, { kind: 'positions', predicted: pred, radiusPx: 12 }), p.H);
      // A valid F for a planar scene: F = [e']x H for any epipole e'.
      const e = [0.3, 1, 0.002];
      const ex = new Float64Array([0, -e[2], e[1], e[2], 0, -e[0], -e[1], e[0], 0]);
      const F = mat3MulH(ex, p.H);
      const gF = evaluateMatches(a, b, matchGuidedCPU(a, b, { kind: 'F', F, maxErrorPx: 2 }), p.H);
      console.log(
        `[match] ${name}: unguided ${un.correct}/${un.total}, H ${gH.correct}/${gH.total}, ` +
          `positions ${gP.correct}/${gP.total}, F ${gF.correct}/${gF.total}`,
      );
      expect(gH.correct).toBeGreaterThanOrEqual(un.correct);
      expect(gP.correct).toBeGreaterThanOrEqual(un.correct);
      expect(gF.correct).toBeGreaterThanOrEqual(un.correct);
      // guided radii exceed the 2 px evaluation tolerance, so a few "correct" guided matches count as wrong
      expect(gH.precision).toBeGreaterThan(0.93);
      expect(gP.precision).toBeGreaterThan(0.93);
      expect(gF.precision).toBeGreaterThan(0.9);
    });
  }
});

describe('matching timing', () => {
  it('4000 x 4000 brute force (Node, informational)', () => {
    const W = 1600;
    const H = 1200;
    const p = makeWarpedPair(W, H, similarityAboutCenter(W, H, 15, 0.95), 41);
    const a = extractSiftCPU(p.a, { maxFeatures: 4000 });
    const b = extractSiftCPU(p.b, { maxFeatures: 4000 });
    matchCPU(a, b); // warm-up (JIT, caches, WASM memory)
    const t0 = performance.now();
    const m = matchCPU(a, b);
    const dt = performance.now() - t0;
    const ev = evaluateMatches(a, b, m, p.H);
    const pred = new Float32Array(2 * a.count);
    const q = [0, 0];
    for (let i = 0; i < a.count; i++) {
      applyHomography(p.H, a.keypoints[4 * i], a.keypoints[4 * i + 1], q);
      pred[2 * i] = q[0];
      pred[2 * i + 1] = q[1];
    }
    matchGuidedCPU(a, b, { kind: 'positions', predicted: pred, radiusPx: 20 });
    const t1 = performance.now();
    const g = matchGuidedCPU(a, b, { kind: 'positions', predicted: pred, radiusPx: 20 });
    const dg = performance.now() - t1;
    const evg = evaluateMatches(a, b, g, p.H);
    const e = [0.3, 1, 0.002];
    const F = mat3MulH(new Float64Array([0, -e[2], e[1], e[2], 0, -e[0], -e[1], e[0], 0]), p.H);
    matchGuidedCPU(a, b, { kind: 'F', F, maxErrorPx: 3 });
    const t2 = performance.now();
    const gf = matchGuidedCPU(a, b, { kind: 'F', F, maxErrorPx: 3 });
    const df = performance.now() - t2;
    const evf = evaluateMatches(a, b, gf, p.H);
    console.log(
      `[match] ${a.count}x${b.count} brute force (${matchUsesSIMD() ? 'WASM SIMD' : 'JS'}): ${dt.toFixed(0)} ms, ` +
        `${ev.correct}/${ev.total} correct; guided positions (r=20px): ${dg.toFixed(0)} ms, ${evg.correct}/${evg.total}; ` +
        `guided F (3px): ${df.toFixed(0)} ms, ${evf.correct}/${evf.total}`,
    );
    expect(a.count).toBe(4000);
    expect(ev.precision).toBeGreaterThan(0.95);
  });
});
