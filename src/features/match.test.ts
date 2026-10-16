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

  it('F guide agrees with the reference for a finite epipole inside the image (steep + flat lines)', () => {
    // Lines through an epipole inside the frame take every direction, exercising both
    // the column walk (flat lines) and the row walk (steep lines) of the band search.
    for (const seed of [11, 12]) {
      const { a, b } = randomSets(150, 160, 0.12, seed);
      const e = [500.3, 400.7, 1];
      const ex = [0, -e[2], e[1], e[2], 0, -e[0], -e[1], e[0], 0];
      const A = [0.98, -0.17, 15, 0.17, 0.98, -7, 1e-5, -2e-5, 1];
      const F = new Float64Array(9);
      for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++) F[3 * r + c] = ex[3 * r] * A[c] + ex[3 * r + 1] * A[3 + c] + ex[3 * r + 2] * A[6 + c];
      for (const tol of [1, 8, 40]) {
        const cand = (i: number, j: number) => {
          const xa = a.keypoints[4 * i];
          const ya = a.keypoints[4 * i + 1];
          const l = [F[0] * xa + F[1] * ya + F[2], F[3] * xa + F[4] * ya + F[5], F[6] * xa + F[7] * ya + F[8]];
          return Math.abs(l[0] * b.keypoints[4 * j] + l[1] * b.keypoints[4 * j + 1] + l[2]) <= tol * Math.hypot(l[0], l[1]);
        };
        for (const opts of [{ ratio: 0.95, maxDistance: 1.4 }, { ratio: 0.95, maxDistance: 1.4, mutual: false }]) {
          expect(Array.from(matchGuidedCPU(a, b, { kind: 'F', F, maxErrorPx: tol }, opts))).toEqual(naiveMatch(a, b, opts, cand));
        }
      }
    }
  });

  it('H guide is invariant to the (possibly negative) scale of H', () => {
    // Homographies are defined up to scale; a DLT/SVD estimate has an arbitrary sign.
    const { a, b } = randomSets(120, 130, 0.1, 13);
    const H = new Float64Array([0.98, -0.05, 12, 0.04, 1.01, -9, 2e-5, -1e-5, 1]);
    const ref = Array.from(matchGuidedCPU(a, b, { kind: 'H', H, maxErrorPx: 30 }));
    expect(ref.length).toBeGreaterThan(20);
    for (const s of [-1, -1e-4, 1e-6, 3e5, -7]) {
      const Hs = H.map((v) => v * s);
      expect(Array.from(matchGuidedCPU(a, b, { kind: 'H', H: Hs, maxErrorPx: 30 }))).toEqual(ref);
    }
    // and it agrees with the equivalent positions guide
    const pred = new Float32Array(2 * a.count);
    for (let i = 0; i < a.count; i++) {
      const x = a.keypoints[4 * i];
      const y = a.keypoints[4 * i + 1];
      const w = H[6] * x + H[7] * y + H[8];
      pred[2 * i] = (H[0] * x + H[1] * y + H[2]) / w;
      pred[2 * i + 1] = (H[3] * x + H[4] * y + H[5]) / w;
    }
    expect(Array.from(matchGuidedCPU(a, b, { kind: 'positions', predicted: pred, radiusPx: 30 }))).toEqual(ref);
  });

  it('positions guide handles predictions outside the frame of b', () => {
    const { a, b } = randomSets(90, 100, 0.12, 14);
    const pred = new Float32Array(2 * a.count);
    for (let i = 0; i < a.count; i++) {
      // shifted left/up by 60 px: many predictions fall outside b (x, y < -0.5)
      pred[2 * i] = a.keypoints[4 * i] - 60;
      pred[2 * i + 1] = a.keypoints[4 * i + 1] * 1.1 - 60;
    }
    pred[2 * 3] = Infinity;
    const R = 75;
    const cand = (i: number, j: number) => {
      const px = pred[2 * i];
      const py = pred[2 * i + 1];
      if (!Number.isFinite(px) || !Number.isFinite(py)) return false;
      return (b.keypoints[4 * j] - px) ** 2 + (b.keypoints[4 * j + 1] - py) ** 2 <= R * R;
    };
    for (const opts of [{}, { mutual: false, ratio: 0.9 }]) {
      expect(Array.from(matchGuidedCPU(a, b, { kind: 'positions', predicted: pred, radiusPx: R }, opts))).toEqual(
        naiveMatch(a, b, opts, cand),
      );
    }
  });

  it('never uses stale descriptor norms when a descriptor buffer is refilled in place', () => {
    const { a, b } = randomSets(60, 70, 0.08, 15);
    const other = randomSets(60, 70, 0.08, 16).b;
    // different norms (e.g. other quantisation / zeroed features): scale every other descriptor
    for (let j = 0; j < other.count; j += 2)
      for (let k = 0; k < D; k++) other.descriptors[j * D + k] = Math.round(other.descriptors[j * D + k] * (j % 4 ? 0.3 : 0));
    // warm any per-array caches with the original content
    matchCPU(a, b);
    matchGuidedCPU(a, b, { kind: 'positions', predicted: new Float32Array(2 * a.count).fill(NaN), radiusPx: 1 });
    // a pooled buffer is overwritten with different descriptors (same object, same count)
    b.descriptors.set(other.descriptors);
    const fresh: FeatureSet = { ...b, descriptors: new Uint8Array(b.descriptors) };
    expect(Array.from(matchCPU(a, b, { ratio: 0.95, maxDistance: 1.4 }))).toEqual(
      Array.from(matchCPU(a, fresh, { ratio: 0.95, maxDistance: 1.4 })),
    );
    const g = { kind: 'positions' as const, predicted: new Float32Array(2 * a.count).fill(NaN), radiusPx: 1 };
    expect(Array.from(matchGuidedCPU(a, b, g, { ratio: 0.95, maxDistance: 1.4 }))).toEqual(
      Array.from(matchGuidedCPU(a, fresh, g, { ratio: 0.95, maxDistance: 1.4 })),
    );
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
