import { describe, expect, it } from 'vitest';
import { solveSPD } from '../math/decomp';
import { Rng } from '../math/random';
import {
  BlockJacobiPreconditioner,
  BlockSparseSymmetric,
  createPcgWorkspace,
  pcgSolve,
  scatterToSkyline,
  SkylineCholesky,
  skylineEnvelope,
} from './linear';

/** Random SPD block-sparse matrix: banded pose-like blocks + a few long-range links + trailing small blocks. */
function randomSystem(rng: Rng, nPose: number, band: number, extra: number, camSizes: number[]) {
  const sizes = [...Array(nPose).fill(6), ...camSizes];
  const nB = sizes.length;
  const pattern: Set<number>[] = Array.from({ length: nB }, (_, a) => new Set([a]));
  for (let a = 0; a < nPose; a++) for (let b = a + 1; b <= Math.min(nPose - 1, a + band); b++) pattern[a].add(b);
  for (let k = 0; k < extra; k++) {
    const a = rng.int(nPose),
      b = rng.int(nPose);
    pattern[Math.min(a, b)].add(Math.max(a, b));
  }
  for (let a = 0; a < nPose; a++) for (let c = nPose; c < nB; c++) pattern[a].add(c);
  for (let c = nPose; c < nB; c++) for (let d = c; d < nB; d++) pattern[c].add(d);
  const rowStart = new Int32Array(nB + 1);
  const cols: number[] = [];
  for (let a = 0; a < nB; a++) {
    const s = [...pattern[a]].sort((x, y) => x - y);
    cols.push(...s);
    rowStart[a + 1] = cols.length;
  }
  const A = new BlockSparseSymmetric(Int32Array.from(sizes), rowStart, Int32Array.from(cols));
  const n = A.dim;
  // Dense SPD: M = B Bᵀ restricted to the pattern is not SPD in general, so build a
  // diagonally dominant symmetric matrix on the pattern.
  const D = new Float64Array(n * n);
  for (let a = 0; a < nB; a++) {
    for (let e = A.rowStart[a]; e < A.rowStart[a + 1]; e++) {
      const b = A.col[e];
      for (let r = 0; r < sizes[a]; r++)
        for (let c = 0; c < sizes[b]; c++) {
          const i = A.blockStart[a] + r,
            j = A.blockStart[b] + c;
          if (j < i) continue;
          const v = i === j ? 0 : rng.uniform(-1, 1);
          D[i * n + j] = v;
          D[j * n + i] = v;
        }
    }
  }
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += Math.abs(D[i * n + j]);
    D[i * n + i] = s + 0.1 + rng.next() * Math.pow(10, rng.uniform(0, 4)); // varying scales
  }
  // fill block values from dense
  for (let a = 0; a < nB; a++) {
    for (let e = A.rowStart[a]; e < A.rowStart[a + 1]; e++) {
      const b = A.col[e];
      const o = A.valOff[e];
      for (let r = 0; r < sizes[a]; r++)
        for (let c = 0; c < sizes[b]; c++)
          A.values[o + r * sizes[b] + c] = D[(A.blockStart[a] + r) * n + A.blockStart[b] + c];
    }
  }
  return { A, D, n };
}

describe('BlockSparseSymmetric', () => {
  it('multiply / quadraticForm match the dense matrix', () => {
    const rng = new Rng(1);
    const { A, D, n } = randomSystem(rng, 15, 3, 5, [1, 5]);
    const x = new Float64Array(n).map(() => rng.gaussian());
    const y = new Float64Array(n);
    A.multiply(x, y);
    let q = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += D[i * n + j] * x[j];
      expect(y[i]).toBeCloseTo(s, 9);
      q += x[i] * s;
    }
    expect(A.quadraticForm(x)).toBeCloseTo(q, 7);
  });

  it('rejects malformed patterns', () => {
    expect(() => new BlockSparseSymmetric(Int32Array.of(6, 6), Int32Array.of(0, 1, 1), Int32Array.of(0))).toThrow();
    expect(() => new BlockSparseSymmetric(Int32Array.of(6, 6), Int32Array.of(0, 1, 2), Int32Array.of(1, 1))).toThrow();
  });
});

describe('skyline Cholesky', () => {
  it('equals the dense SPD solve, with dropped (pinned) indices; envelope ⊇ exact row profile', () => {
    const rng = new Rng(2);
    for (const [nPose, band, extra, cams] of [
      [25, 4, 6, [3]],
      [30, 2, 0, [1, 5]],
      [12, 11, 0, []],
    ] as const) {
      const { A, D, n } = randomSystem(rng, nPose, band, extra, [...cams]);
      // drop 3 scalar indices (like pinned translation components)
      const map = new Int32Array(n);
      const drop = new Set([3, 4, 23]);
      let g = 0;
      for (let s = 0; s < n; s++) map[s] = drop.has(s) ? -1 : g++;
      const m = g;
      const Dense = new Float64Array(m * m);
      const exact = new Int32Array(m);
      A.toDenseLower(map, m, Dense, exact);
      const first = skylineEnvelope(A, map, m);
      for (let i = 0; i < m; i++) {
        expect(first[i]).toBeLessThanOrEqual(exact[i]);
        expect(first[i]).toBeGreaterThanOrEqual(0);
      }
      const sky = new SkylineCholesky(first);
      expect(sky.size).toBe(SkylineCholesky.storageSize(first));
      scatterToSkyline(A, map, sky);
      // reference: full symmetric dense matrix
      const Dr = new Float64Array(m * m);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) if (map[i] >= 0 && map[j] >= 0) Dr[map[i] * m + map[j]] = D[i * n + j];
      const b = new Float64Array(m).map(() => rng.gaussian());
      const ref = solveSPD(Dr, m, b)!;
      expect(sky.factor()).toBe(true);
      const x = b.slice();
      sky.solve(x);
      for (let i = 0; i < m; i++) expect(x[i]).toBeCloseTo(ref[i], 9);
    }
  });

  it('banded envelope is narrow; trailing dense rows (intrinsics) do not widen it', () => {
    const rng = new Rng(4);
    const { A, n } = randomSystem(rng, 50, 3, 0, [3]);
    const map = Int32Array.from({ length: n }, (_, i) => i);
    const first = skylineEnvelope(A, map, n);
    expect(SkylineCholesky.storageSize(first)).toBeLessThan(n * 30);
    expect(first[n - 1]).toBe(0);
    expect(first[6 * 25]).toBe(6 * 22);
    expect(SkylineCholesky.factorCost(first)).toBeLessThan((n * n * n) / 6 / 10);
  });

  it('factor reproduces L for arbitrary envelopes (row pairs with equal / different first columns)', () => {
    const rng = new Rng(9);
    for (const n of [1, 2, 7, 40, 61]) {
      // random envelope; runs of equal first columns like BA parameter blocks, and ragged rows
      const first = new Int32Array(n);
      for (let i = 0; i < n; i++) first[i] = i > 0 && rng.next() < 0.5 ? Math.min(i, first[i - 1]) : Math.floor(rng.next() * (i + 1));
      // L with that skyline profile (positive diagonal), A = L Lᵀ has the same envelope
      const L = new Float64Array(n * n);
      for (let i = 0; i < n; i++) {
        for (let j = first[i]; j < i; j++) L[i * n + j] = rng.gaussian();
        L[i * n + i] = 1 + rng.next();
      }
      const sky = new SkylineCholesky(first);
      for (let i = 0; i < n; i++)
        for (let j = first[i]; j <= i; j++) {
          let v = 0;
          for (let k = 0; k <= j; k++) v += L[i * n + k] * L[j * n + k];
          sky.values[sky.rowBase[i] + j] = v;
        }
      expect(sky.factor()).toBe(true);
      for (let i = 0; i < n; i++) for (let j = first[i]; j <= i; j++) expect(sky.values[sky.rowBase[i] + j]).toBeCloseTo(L[i * n + j], 8);
    }
  });

  it('rejects indefinite matrices and invalid envelopes', () => {
    const sky = new SkylineCholesky(Int32Array.of(0, 0));
    sky.values.set([1, 2, 1]);
    expect(sky.factor()).toBe(false);
    expect(() => new SkylineCholesky(Int32Array.of(0, 2))).toThrow(RangeError);
  });
});

describe('block-Jacobi PCG', () => {
  it('converges to the direct solution', () => {
    const rng = new Rng(3);
    const { A, D, n } = randomSystem(rng, 40, 5, 20, [1, 4]);
    const b = new Float64Array(n).map(() => rng.gaussian());
    const ref = solveSPD(D, n, b)!;
    const M = new BlockJacobiPreconditioner(A);
    M.update();
    const x = new Float64Array(n);
    const it = pcgSolve(A, M, b, x, 1000, 1e-12, createPcgWorkspace(n));
    expect(it).toBeGreaterThan(0);
    expect(it).toBeLessThan(200);
    let err = 0,
      nrm = 0;
    for (let i = 0; i < n; i++) {
      err = Math.max(err, Math.abs(x[i] - ref[i]));
      nrm = Math.max(nrm, Math.abs(ref[i]));
    }
    expect(err / nrm).toBeLessThan(1e-9);
  });

  it('returns x = 0 for b = 0 and handles a non-SPD diagonal block via scalar Jacobi', () => {
    const A = new BlockSparseSymmetric(Int32Array.of(2), Int32Array.of(0, 1), Int32Array.of(0));
    A.values.set([1, 2, 2, 1]); // indefinite
    const M = new BlockJacobiPreconditioner(A);
    M.update();
    const x = new Float64Array(2);
    expect(pcgSolve(A, M, new Float64Array(2), x, 10, 1e-10, createPcgWorkspace(2))).toBe(0);
    expect(Array.from(x)).toEqual([0, 0]);
    const z = new Float64Array(2);
    M.apply(Float64Array.of(2, 4), z);
    expect(Array.from(z)).toEqual([2, 4]);
  });
});
