import { describe, expect, it } from 'vitest';
import type { FeatureSet, Mat3, MatchPairs } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';
import { matchCPU, matchGuidedCPU } from './match';
import type { MatchGuide, MatchOptions } from './match';
import {
  descriptorStats,
  emitMatches,
  morton10,
  packDescriptorWords,
  planChunks,
  predictWithHomography,
  prepareGuide,
  reduceTop2,
  spatialOrder,
} from './match-gpu-host';
import type { GuidePrep } from './match-gpu-host';
import { MATCH_INVALID, MATCH_TILE } from './shaders/match-wgsl';
import { TestRng } from './testdata';

const D = DESCRIPTOR_SIZE;

function randomSets(na: number, nb: number, noise: number, seed: number): { a: FeatureSet; b: FeatureSet } {
  const rng = new TestRng(seed);
  const mk = (n: number): FeatureSet => ({
    count: n,
    keypoints: new Float32Array(4 * n),
    descriptors: new Uint8Array(D * n),
    scores: new Float32Array(n),
    width: 1000,
    height: 800,
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
    a.keypoints[4 * i + 1] = rng.uniform(0, 800);
  }
  for (let j = 0; j < nb; j++) {
    const src = j < na ? (j * 7919) % na : -1;
    if (src >= 0 && j < Math.floor(nb * 0.7)) {
      for (let k = 0; k < D; k++) v[k] = Math.max(0, a.descriptors[src * D + k] / 512 + noise * rng.gaussian());
      b.keypoints[4 * j] = a.keypoints[4 * src] + rng.uniform(-1, 1);
      b.keypoints[4 * j + 1] = a.keypoints[4 * src + 1] + rng.uniform(-1, 1);
    } else {
      for (let k = 0; k < D; k++) v[k] = rng.next() < 0.3 ? 0 : rng.next();
      b.keypoints[4 * j] = rng.uniform(0, 1000);
      b.keypoints[4 * j + 1] = rng.uniform(0, 800);
    }
    quant(b.descriptors, j * D);
  }
  return { a, b };
}

function idot(A: Uint8Array, i: number, B: Uint8Array, j: number): number {
  let s = 0;
  for (let k = 0; k < D; k++) s += A[i * D + k] * B[j * D + k];
  return s;
}

/** f32 emulation of the shader predicate on the guide records. */
function pred(prep: GuidePrep, i: number, j: number): boolean {
  const r = prep.aux.subarray(4 * (prep.aOff + i), 4 * (prep.aOff + i) + 4);
  const p = prep.aux.subarray(4 * (prep.bOff + j), 4 * (prep.bOff + j) + 4);
  const f = Math.fround;
  if (prep.mode === 'pos') {
    if (r[2] === 1) return true;
    if (r[2] !== 0) return false;
    const dx = f(p[0] - r[0]);
    const dy = f(p[1] - r[1]);
    return f(f(dx * dx) + f(dy * dy)) <= f(prep.r2);
  }
  if (r[3] !== 0) return false;
  return Math.abs(f(f(f(r[0] * p[2]) + f(r[1] * p[3])) + r[2])) <= f(prep.tol);
}

/** Conservative tile test of the shader (a-record i vs b tile box t). */
function tileTest(prep: GuidePrep, i: number, t: number): boolean {
  const r = prep.aux.subarray(4 * (prep.aOff + i), 4 * (prep.aOff + i) + 4);
  const bx = prep.aux.subarray(4 * (prep.boxOff + t), 4 * (prep.boxOff + t) + 4);
  if (prep.mode === 'pos') {
    if (r[2] === 1) return true;
    if (r[2] !== 0) return false;
    const R = prep.radius;
    return r[0] + R >= bx[0] && r[0] - R <= bx[2] && r[1] + R >= bx[1] && r[1] - R <= bx[3];
  }
  if (r[3] !== 0) return false;
  const d = [
    r[0] * bx[0] + r[1] * bx[1] + r[2],
    r[0] * bx[2] + r[1] * bx[1] + r[2],
    r[0] * bx[0] + r[1] * bx[3] + r[2],
    r[0] * bx[2] + r[1] * bx[3] + r[2],
  ];
  return Math.min(...d) <= prep.tol && Math.max(...d) >= -prep.tol;
}

/**
 * CPU emulation of the GPU kernel: per (row, column chunk) top-2 by f32 score with
 * ties to the smaller index, guide predicate + tile culling, then the host decision.
 */
function emulate(a: FeatureSet, b: FeatureSet, prep: GuidePrep | null, opts: MatchOptions, target: number): MatchPairs {
  const sa = descriptorStats(a);
  const sb = descriptorStats(b);
  const run = (rowsAreA: boolean) => {
    const R = rowsAreA ? a : b;
    const C = rowsAreA ? b : a;
    const invC32 = Float32Array.from(rowsAreA ? sb.invNorm : sa.invNorm);
    const c = planChunks(R.count, C.count, target);
    const out = new Uint32Array(4 * R.count * c.nChunks).fill(MATCH_INVALID);
    const rowPerm = prep ? (rowsAreA ? prep.aPerm : prep.bPerm) : null;
    const colPerm = prep ? (rowsAreA ? prep.bPerm : prep.aPerm) : null;
    for (let rt = 0; rt < c.nRowTiles; rt++) {
      for (let ch = 0; ch < c.nChunks; ch++) {
        for (let ct = ch * c.tilesPerChunk; ct < Math.min((ch + 1) * c.tilesPerChunk, c.nColTiles); ct++) {
          if (prep) {
            // tile culling: a-tile items vs b-tile box
            let hit = false;
            for (let t = 0; t < MATCH_TILE && !hit; t++) {
              const sp = (rowsAreA ? rt : ct) * MATCH_TILE + t;
              if (sp >= a.count) break;
              hit = tileTest(prep, prep.aPerm[sp], rowsAreA ? ct : rt);
            }
            if (!hit) {
              // conservativeness: no pair of the skipped tiles may pass the predicate
              for (let r = rt * MATCH_TILE; r < Math.min(R.count, (rt + 1) * MATCH_TILE); r++)
                for (let q = ct * MATCH_TILE; q < Math.min(C.count, (ct + 1) * MATCH_TILE); q++) {
                  const ri = rowPerm![r];
                  const ci = colPerm![q];
                  expect(rowsAreA ? pred(prep, ri, ci) : pred(prep, ci, ri)).toBe(false);
                }
              continue;
            }
          }
          for (let r = rt * MATCH_TILE; r < Math.min(R.count, (rt + 1) * MATCH_TILE); r++) {
            const ri = rowPerm ? rowPerm[r] : r;
            const o = 4 * (ri * c.nChunks + ch);
            for (let q = ct * MATCH_TILE; q < Math.min(C.count, (ct + 1) * MATCH_TILE); q++) {
              const ci = colPerm ? colPerm[q] : q;
              if (prep && !(rowsAreA ? pred(prep, ri, ci) : pred(prep, ci, ri))) continue;
              const d = idot(R.descriptors, ri, C.descriptors, ci);
              const s = Math.fround(d * invC32[ci]);
              const s1 = out[o] === MATCH_INVALID ? -1 : Math.fround(out[o + 1] * invC32[out[o]]);
              const s2 = out[o + 2] === MATCH_INVALID ? -1 : Math.fround(out[o + 3] * invC32[out[o + 2]]);
              if (s > s1 || (s === s1 && ci < out[o])) {
                out[o + 2] = out[o];
                out[o + 3] = out[o + 1];
                out[o] = ci;
                out[o + 1] = d;
              } else if (s > s2 || (s === s2 && ci < out[o + 2])) {
                out[o + 2] = ci;
                out[o + 3] = d;
              }
            }
          }
        }
      }
    }
    return { out, c };
  };
  const f = run(true);
  const bestJ = new Int32Array(a.count);
  const bestDot = new Float64Array(a.count);
  const secondDot = new Float64Array(a.count);
  reduceTop2(f.out, 0, a.count, f.c.nChunks, sa.invNorm, sb.invNorm, true, bestJ, bestDot, secondDot);
  let colBest: Int32Array | null = null;
  if (opts.mutual ?? true) {
    const r = run(false);
    colBest = new Int32Array(b.count);
    reduceTop2(r.out, 0, b.count, r.c.nChunks, sb.invNorm, sa.invNorm, false, colBest, new Float64Array(b.count), new Float64Array(b.count));
  }
  return emitMatches(a.count, bestJ, bestDot, secondDot, colBest, opts);
}

describe('match-gpu host helpers', () => {
  it('morton10 interleaves bits; spatialOrder is a class-ordered, local permutation', () => {
    expect(morton10(0, 0)).toBe(0);
    expect(morton10(1, 0)).toBe(1);
    expect(morton10(0, 1)).toBe(2);
    expect(morton10(3, 3)).toBe(15);
    expect(morton10(1023, 1023)).toBe(2 ** 20 - 1);
    const rng = new TestRng(2);
    const n = 1000;
    const xs = new Float64Array(n);
    const ys = new Float64Array(n);
    const cls = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = rng.uniform(0, 1000);
      ys[i] = rng.uniform(0, 1000);
      cls[i] = i % 10 === 0 ? 2 : i % 7 === 0 ? 1 : 0;
    }
    const p = spatialOrder(xs, ys, n, 0, 0, 1000, cls);
    expect(Array.from(p).sort((u, v) => u - v)).toEqual(Array.from({ length: n }, (_, i) => i));
    for (let k = 1; k < n; k++) expect(cls[p[k]]).toBeGreaterThanOrEqual(cls[p[k - 1]]);
    // tiles of 64 class-0 points cover a small part of the image
    let area = 0;
    let tiles = 0;
    const n0 = cls.filter((c) => c === 0).length;
    for (let t = 0; t + 64 <= n0; t += 64) {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let k = t; k < t + 64; k++) {
        x0 = Math.min(x0, xs[p[k]]);
        x1 = Math.max(x1, xs[p[k]]);
        y0 = Math.min(y0, ys[p[k]]);
        y1 = Math.max(y1, ys[p[k]]);
      }
      area += (x1 - x0) * (y1 - y0);
      tiles++;
    }
    expect(area / tiles / 1e6).toBeLessThan(0.2);
  });

  it('descriptorStats: norms as match.ts, hash sensitive to content, alignment independent', () => {
    const { a } = randomSets(50, 1, 0.1, 3);
    const st = descriptorStats(a);
    for (let i = 0; i < a.count; i++) {
      let ss = 0;
      for (let k = 0; k < D; k++) ss += a.descriptors[i * D + k] ** 2;
      expect(st.invNorm[i]).toBe(1 / Math.sqrt(ss));
    }
    const buf = new Uint8Array(a.descriptors.length + 1);
    buf.set(a.descriptors, 1);
    const un: FeatureSet = { ...a, descriptors: buf.subarray(1) };
    expect(descriptorStats(un).invNorm).toEqual(st.invNorm);
    expect(Array.from(packDescriptorWords(un))).toEqual(Array.from(packDescriptorWords(a)));
    const h0 = st.hash;
    a.descriptors[777] ^= 1;
    expect(descriptorStats(a).hash).not.toBe(h0);
    a.descriptors[777] ^= 1;
    expect(descriptorStats(a).hash).toBe(h0);
    expect(descriptorStats(un).hash).toBe(h0);
    // bit 31 flipped in two words of the same lane (16 bytes apart): plain FNV cancels this
    a.descriptors[3] ^= 0x80;
    a.descriptors[19] ^= 0x80;
    expect(descriptorStats(a).hash).not.toBe(h0);
    a.descriptors[3] ^= 0x80;
    a.descriptors[19] ^= 0x80;
    expect(descriptorStats(a).hash).toBe(h0);
  });

  it('predictWithHomography is invariant to the sign of H and flags points beyond infinity', () => {
    const { a } = randomSets(80, 1, 0.1, 4);
    const H = new Float64Array([1, 0.1, 5, -0.05, 1, 3, -0.0015, 0.0001, 1]); // horizon x ≈ 670 crosses the image
    const p1 = predictWithHomography(a, H);
    const p2 = predictWithHomography(a, H.map((v) => -3 * v));
    expect(Array.from(p2)).toEqual(Array.from(p1));
    let inf = 0;
    for (let i = 0; i < a.count; i++) if (!Number.isFinite(p1[2 * i])) inf++;
    expect(inf).toBeGreaterThan(0);
    expect(inf).toBeLessThan(a.count / 2);
  });

  it('planChunks covers all column tiles', () => {
    for (const [r, c, t] of [
      [10, 10, 128],
      [4096, 4096, 128],
      [64, 5000, 128],
      [100000, 3, 128],
    ]) {
      const p = planChunks(r, c, t);
      expect(p.nChunks * p.tilesPerChunk).toBeGreaterThanOrEqual(p.nColTiles);
      expect((p.nChunks - 1) * p.tilesPerChunk).toBeLessThan(p.nColTiles);
    }
  });
});

describe('kernel emulation + host decision reproduce the CPU matchers', () => {
  const optsList: MatchOptions[] = [{}, { mutual: false }, { ratio: 0.95, maxDistance: 1.4 }];

  it('brute force (several chunkings)', () => {
    const { a, b } = randomSets(150, 170, 0.1, 21);
    for (const opts of optsList) {
      const ref = Array.from(matchCPU(a, b, opts));
      for (const target of [1, 4, 1000]) expect(Array.from(emulate(a, b, null, opts, target))).toEqual(ref);
    }
  });

  it('guided: positions (NaN / Inf / out of frame), H (any sign), F (finite epipole)', () => {
    const { a, b } = randomSets(200, 210, 0.12, 22);
    const pred = new Float32Array(2 * a.count);
    for (let i = 0; i < a.count; i++) {
      pred[2 * i] = a.keypoints[4 * i] + 3;
      pred[2 * i + 1] = a.keypoints[4 * i + 1] - 2;
    }
    pred[10] = NaN;
    pred[13] = NaN;
    pred[14] = Infinity;
    pred[16] = -400;
    const e = [500.3, 400.7, 1];
    const ex = [0, -e[2], e[1], e[2], 0, -e[0], -e[1], e[0], 0];
    const A = [0.98, -0.17, 15, 0.17, 0.98, -7, 1e-5, -2e-5, 1];
    const F = new Float64Array(9);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) F[3 * r + c] = ex[3 * r] * A[c] + ex[3 * r + 1] * A[3 + c] + ex[3 * r + 2] * A[6 + c];
    const H: Mat3 = new Float64Array([0.98, -0.05, 12, 0.04, 1.01, -9, 2e-5, -1e-5, 1]);
    const guides: MatchGuide[] = [
      { kind: 'positions', predicted: pred, radiusPx: 15 },
      { kind: 'positions', predicted: pred, radiusPx: 120 },
      { kind: 'H', H, maxErrorPx: 30 },
      { kind: 'H', H: H.map((v) => -v), maxErrorPx: 30 },
      { kind: 'F', F, maxErrorPx: 3 },
      { kind: 'F', F, maxErrorPx: 40 },
    ];
    for (const g of guides) {
      const prep = prepareGuide(a, b, g);
      for (const opts of optsList) {
        const ref = Array.from(matchGuidedCPU(a, b, g, opts));
        expect(Array.from(emulate(a, b, prep, opts, 8)), `${g.kind} ${JSON.stringify(opts)}`).toEqual(ref);
      }
    }
  });

  it('exact ties (duplicate / all-zero descriptors) resolve like the CPU, incl. ratio > 1 where the tie-break is visible', () => {
    const { a, b } = randomSets(130, 150, 0.05, 23);
    for (const j of [70, 140]) b.descriptors.copyWithin(j * D, 3 * D, 4 * D); // b[3] = b[70] = b[140]
    a.descriptors.copyWithin(9 * D, 8 * D, 9 * D); // a[8] = a[9]
    a.descriptors.fill(0, 12 * D, 13 * D);
    b.descriptors.fill(0, 20 * D, 22 * D);
    for (const opts of [{}, { ratio: 1.5 }, { ratio: 1.5, maxDistance: 2 }, { ratio: 1.5, maxDistance: 2, mutual: false }] as MatchOptions[]) {
      const ref = Array.from(matchCPU(a, b, opts));
      for (const target of [1, 3, 1000]) expect(Array.from(emulate(a, b, null, opts, target)), JSON.stringify(opts)).toEqual(ref);
    }
    // guided (ratio < 1: an exact tie fails the ratio test whichever index wins)
    const g: MatchGuide = { kind: 'positions', predicted: new Float32Array(2 * a.count).fill(NaN), radiusPx: 1 };
    expect(Array.from(emulate(a, b, prepareGuide(a, b, g), {}, 5))).toEqual(Array.from(matchGuidedCPU(a, b, g)));
  });

  it('rejects predictions arrays that are too short', () => {
    const { a, b } = randomSets(10, 10, 0.1, 1);
    expect(() => prepareGuide(a, b, { kind: 'positions', predicted: new Float32Array(3), radiusPx: 2 })).toThrow(/predicted/);
  });
});
