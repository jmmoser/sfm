import { describe, expect, it } from 'vitest';
import {
  cholesky,
  choleskySolve,
  luSolve,
  nullSpace,
  nullVector,
  qr,
  solveLeastSquares,
  solveSPD,
  svd,
  svd3,
  symEig,
} from './decomp';
import { Matrix } from './matrix';
import { Rng } from './random';

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function randMatrix(rng: Rng, m: number, n: number, scale = 1): Matrix {
  const A = new Matrix(m, n);
  for (let i = 0; i < m * n; i++) A.data[i] = rng.gaussian() * scale;
  return A;
}

/** Random m x k matrix with orthonormal columns (k ≤ m), via QR of a Gaussian matrix. */
function randOrthonormal(rng: Rng, m: number, k: number): Matrix {
  return qr(randMatrix(rng, m, k)).Q;
}

/** A = U diag(s) Vᵀ with random orthonormal U (m x k), V (n x k), k = s.length. */
function withSingularValues(rng: Rng, m: number, n: number, s: number[]): Matrix {
  const k = s.length;
  const U = randOrthonormal(rng, m, k);
  const V = randOrthonormal(rng, n, k);
  const A = new Matrix(m, n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      let v = 0;
      for (let l = 0; l < k; l++) v += U.data[i * k + l] * s[l] * V.data[j * k + l];
      A.data[i * n + j] = v;
    }
  return A;
}

function maxAbs(a: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]));
  return m;
}

/** max |QᵀQ − I| */
function orthoError(Q: Matrix): number {
  const G = Q.transposeMul(Q);
  let e = 0;
  for (let i = 0; i < G.rows; i++)
    for (let j = 0; j < G.cols; j++) e = Math.max(e, Math.abs(G.get(i, j) - (i === j ? 1 : 0)));
  return e;
}

/** max |A − U diag(S) Vᵀ| */
function reconError(A: Matrix, U: Matrix, S: Float64Array, V: Matrix): number {
  const k = S.length;
  let e = 0;
  for (let i = 0; i < A.rows; i++)
    for (let j = 0; j < A.cols; j++) {
      let v = 0;
      for (let l = 0; l < k; l++) v += U.get(i, l) * S[l] * V.get(j, l);
      e = Math.max(e, Math.abs(v - A.get(i, j)));
    }
  return e;
}

function checkSvd(A: Matrix, tol = 1e-12): { U: Matrix; S: Float64Array; V: Matrix } {
  const r = svd(A);
  const m = A.rows,
    n = A.cols,
    k = Math.min(m, n);
  expect(r.U.rows).toBe(m);
  expect(r.U.cols).toBe(k);
  expect(r.V.rows).toBe(n);
  expect(r.V.cols).toBe(k);
  expect(r.S.length).toBe(k);
  for (let i = 0; i < k; i++) {
    expect(r.S[i]).toBeGreaterThanOrEqual(0);
    if (i > 0) expect(r.S[i]).toBeLessThanOrEqual(r.S[i - 1]);
  }
  const scale = Math.max(maxAbs(A.data), 1e-300);
  expect(reconError(A, r.U, r.S, r.V) / scale).toBeLessThan(tol);
  expect(orthoError(r.U)).toBeLessThan(tol);
  expect(orthoError(r.V)).toBeLessThan(tol);
  return r;
}

function det3(a: ArrayLike<number>): number {
  return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

// ---------------------------------------------------------------------------
// SVD
// ---------------------------------------------------------------------------

describe('svd', () => {
  const shapes: [number, number][] = [
    [1, 1],
    [1, 5],
    [5, 1],
    [2, 2],
    [3, 3],
    [4, 3],
    [3, 4],
    [5, 9],
    [8, 9],
    [9, 9],
    [9, 8],
    [12, 9],
    [9, 12],
    [100, 12],
    [12, 100],
    [200, 9],
    [30, 30],
    [50, 20],
    [20, 50],
  ];
  it('reconstructs random matrices of many shapes with orthonormal factors', () => {
    const rng = new Rng(1);
    for (const [m, n] of shapes) {
      for (let trial = 0; trial < 10; trial++) checkSvd(randMatrix(rng, m, n, Math.pow(10, rng.uniform(-3, 3))));
    }
  });

  it('recovers prescribed singular values (incl. graded spectra)', () => {
    const rng = new Rng(2);
    const spectra = [
      [10, 5, 1, 1e-3, 1e-8],
      [1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
      [3, 2, 1],
      [1e6, 1, 1e-6],
    ];
    for (const s of spectra) {
      for (const [m, n] of [
        [s.length, s.length],
        [s.length + 7, s.length],
        [s.length, s.length + 4],
        [60, s.length + 2],
      ]) {
        const A = withSingularValues(rng, m, n, s);
        const r = checkSvd(A);
        for (let i = 0; i < s.length; i++) expect(Math.abs(r.S[i] - s[i])).toBeLessThan(1e-13 * s[0]);
        for (let i = s.length; i < r.S.length; i++) expect(r.S[i]).toBeLessThan(1e-14 * s[0]);
      }
    }
  });

  it('handles rank-deficient matrices (3x3 rank 1/2, 10x6 rank 3, 6x10 rank 2)', () => {
    const rng = new Rng(3);
    for (let trial = 0; trial < 50; trial++) {
      for (const [m, n, rank] of [
        [3, 3, 1],
        [3, 3, 2],
        [10, 6, 3],
        [6, 10, 2],
        [9, 9, 8],
        [8, 9, 5],
      ]) {
        const s: number[] = [];
        for (let i = 0; i < rank; i++) s.push(rng.uniform(0.5, 5));
        s.sort((a, b) => b - a);
        const A = withSingularValues(rng, m, n, s);
        const r = checkSvd(A);
        for (let i = rank; i < r.S.length; i++) expect(r.S[i]).toBeLessThan(1e-14 * s[0]);
        for (let i = 0; i < rank; i++) expect(Math.abs(r.S[i] - s[i])).toBeLessThan(1e-13 * s[0]);
      }
    }
  });

  it('handles zero matrices and zero rows/columns', () => {
    for (const [m, n] of [
      [1, 1],
      [3, 3],
      [5, 2],
      [2, 5],
    ]) {
      const r = checkSvd(new Matrix(m, n));
      for (const v of r.S) expect(v).toBe(0);
    }
    const A = Matrix.fromRows([
      [1, 0, 0],
      [0, 0, 0],
      [2, 0, 3],
      [0, 0, 0],
    ]);
    const r = checkSvd(A);
    expect(r.S[2]).toBe(0);
  });

  it('handles repeated singular values', () => {
    const rng = new Rng(4);
    for (let trial = 0; trial < 20; trial++) {
      // Orthogonal matrix: all singular values 1.
      const Q = randOrthonormal(rng, 7, 7);
      const r = checkSvd(Q);
      for (const v of r.S) expect(Math.abs(v - 1)).toBeLessThan(1e-14);
      const A = withSingularValues(rng, 12, 6, [2, 2, 2, 1, 1, 0.5]);
      const r2 = checkSvd(A);
      expect(Math.abs(r2.S[1] - 2)).toBeLessThan(1e-14);
      expect(Math.abs(r2.S[4] - 1)).toBeLessThan(1e-14);
    }
    const I = Matrix.identity(5);
    const r = checkSvd(I);
    for (const v of r.S) expect(v).toBe(1);
  });

  it('is scale invariant (no overflow/underflow at 1e±200)', () => {
    const rng = new Rng(5);
    const A = randMatrix(rng, 7, 5);
    const r0 = svd(A);
    for (const sc of [1e200, 1e-200]) {
      const B = new Matrix(7, 5, A.data.map((v) => v * sc));
      const r = svd(B);
      for (let i = 0; i < 5; i++) expect(Math.abs(r.S[i] / sc - r0.S[i])).toBeLessThan(1e-13 * r0.S[0]);
      expect(orthoError(r.U)).toBeLessThan(1e-13);
      expect(orthoError(r.V)).toBeLessThan(1e-13);
    }
  });

  it('agrees with eigenvalues of AᵀA for well-conditioned matrices', () => {
    const rng = new Rng(6);
    for (let trial = 0; trial < 20; trial++) {
      const A = randMatrix(rng, 15, 6);
      const r = svd(A);
      const G = A.transposeMul(A);
      const e = symEig(G.data, 6);
      for (let i = 0; i < 6; i++) expect(Math.abs(Math.sqrt(e.values[5 - i]) - r.S[i])).toBeLessThan(1e-11 * r.S[0]);
    }
  });

  it('propagates non-finite input as NaN', () => {
    const A = Matrix.fromRows([
      [1, 2],
      [NaN, 3],
    ]);
    const r = svd(A);
    expect(Number.isNaN(r.S[0])).toBe(true);
    const B = Matrix.fromRows([
      [1, Infinity],
      [0, 3],
    ]);
    expect(Number.isNaN(svd(B).S[0])).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// null space
// ---------------------------------------------------------------------------

describe('nullVector / nullSpace', () => {
  it('finds the exact null vector of generic 8x9 and 9x9 rank-8 systems', () => {
    const rng = new Rng(10);
    for (let trial = 0; trial < 200; trial++) {
      for (const m of [8, 9, 20]) {
        // Rows orthogonal to a known x (like a noise-free DLT system).
        const x = new Float64Array(9);
        for (let i = 0; i < 9; i++) x[i] = rng.gaussian();
        const xn = Math.hypot(...x);
        for (let i = 0; i < 9; i++) x[i] /= xn;
        const A = new Matrix(m, 9);
        for (let r = 0; r < m; r++) {
          let d = 0;
          const row = new Float64Array(9);
          for (let i = 0; i < 9; i++) row[i] = rng.gaussian() * 100;
          for (let i = 0; i < 9; i++) d += row[i] * x[i];
          for (let i = 0; i < 9; i++) A.data[r * 9 + i] = row[i] - d * x[i];
        }
        const v = nullVector(A);
        expect(v.length).toBe(9);
        let dot = 0,
          nn = 0;
        for (let i = 0; i < 9; i++) {
          dot += v[i] * x[i];
          nn += v[i] * v[i];
        }
        expect(Math.abs(nn - 1)).toBeLessThan(1e-14);
        expect(1 - Math.abs(dot)).toBeLessThan(1e-12);
        const Av = A.mulVec(v);
        expect(maxAbs(Av) / maxAbs(A.data)).toBeLessThan(1e-13);
      }
    }
  });

  it('returns a full orthonormal null-space basis for wide 5x9 / 3x5 / 1x3 systems', () => {
    const rng = new Rng(11);
    for (let trial = 0; trial < 200; trial++) {
      for (const [m, n] of [
        [5, 9],
        [3, 5],
        [1, 3],
        [6, 9],
      ]) {
        const A = randMatrix(rng, m, n, Math.pow(10, rng.uniform(-2, 2)));
        const N = nullSpace(A, n - m);
        expect(N.rows).toBe(n);
        expect(N.cols).toBe(n - m);
        expect(orthoError(N)).toBeLessThan(1e-13);
        const AN = A.mul(N);
        expect(maxAbs(AN.data) / maxAbs(A.data)).toBeLessThan(1e-13);
        const v = nullVector(A);
        expect(maxAbs(A.mulVec(v)) / maxAbs(A.data)).toBeLessThan(1e-13);
      }
    }
  });

  it('rank-deficient wide systems: null space includes rank-deficiency directions', () => {
    const rng = new Rng(12);
    for (let trial = 0; trial < 50; trial++) {
      const A = withSingularValues(rng, 5, 9, [3, 2, 1]); // rank 3 → 6-D null space
      const N = nullSpace(A, 6);
      expect(orthoError(N)).toBeLessThan(1e-13);
      expect(maxAbs(A.mul(N).data) / maxAbs(A.data)).toBeLessThan(1e-13);
    }
  });

  it('least-squares null vector equals the smallest right singular vector (noisy tall system)', () => {
    const rng = new Rng(13);
    for (let trial = 0; trial < 50; trial++) {
      const A = randMatrix(rng, 100, 12);
      const v = nullVector(A);
      const r = svd(A);
      const Av = A.mulVec(v);
      expect(Math.abs(Math.hypot(...Av) - r.S[11])).toBeLessThan(1e-12 * r.S[0]);
      // ordering of nullSpace: column 0 = smallest σ
      const N = nullSpace(A, 3);
      const n1 = Math.hypot(...A.mulVec(N.getColumn(1)));
      const n2 = Math.hypot(...A.mulVec(N.getColumn(2)));
      expect(Math.abs(n1 - r.S[10])).toBeLessThan(1e-12 * r.S[0]);
      expect(Math.abs(n2 - r.S[9])).toBeLessThan(1e-12 * r.S[0]);
    }
  });

  it('validates dim', () => {
    expect(() => nullSpace(new Matrix(2, 3), 0)).toThrow(RangeError);
    expect(() => nullSpace(new Matrix(2, 3), 4)).toThrow(RangeError);
    const N = nullSpace(new Matrix(2, 3), 3); // full basis of R³ for the zero matrix
    expect(orthoError(N)).toBeLessThan(1e-15);
  });

  it('zero matrix gives a unit vector', () => {
    for (const [m, n] of [
      [3, 3],
      [2, 4],
      [6, 3],
    ]) {
      const v = nullVector(new Matrix(m, n));
      expect(Math.abs(Math.hypot(...v) - 1)).toBeLessThan(1e-15);
    }
  });

  it('timing (8x9 nullVector, 100x12 svd)', () => {
    const rng = new Rng(14);
    const mats: Matrix[] = [];
    for (let i = 0; i < 200; i++) mats.push(randMatrix(rng, 8, 9));
    let t0 = performance.now();
    const iters = 2000;
    for (let i = 0; i < iters; i++) nullVector(mats[i % mats.length]);
    const tNull = ((performance.now() - t0) / iters) * 1000;
    const big = randMatrix(rng, 100, 12);
    t0 = performance.now();
    for (let i = 0; i < 200; i++) svd(big);
    const tSvd = ((performance.now() - t0) / 200) * 1000;
    const sq = randMatrix(rng, 9, 9);
    t0 = performance.now();
    for (let i = 0; i < 1000; i++) svd(sq);
    const tSq = ((performance.now() - t0) / 1000) * 1000;
    console.log(
      `[math timing] nullVector 8x9: ${tNull.toFixed(1)} µs, svd 9x9: ${tSq.toFixed(1)} µs, svd 100x12: ${tSvd.toFixed(1)} µs`,
    );
    expect(tNull).toBeLessThan(2000);
  });
});

// ---------------------------------------------------------------------------
// svd3
// ---------------------------------------------------------------------------

function checkSvd3(A: Float64Array, tol = 1e-14): { U: Float64Array; S: Float64Array; V: Float64Array } {
  const r = svd3(A);
  const U = new Matrix(3, 3, r.U),
    V = new Matrix(3, 3, r.V);
  const scale = Math.max(maxAbs(A), 1e-300);
  expect(reconError(new Matrix(3, 3, A), U, r.S, V) / scale).toBeLessThan(tol * 4);
  expect(orthoError(U)).toBeLessThan(tol * 4);
  expect(orthoError(V)).toBeLessThan(tol * 4);
  expect(r.S[0]).toBeGreaterThanOrEqual(r.S[1]);
  expect(r.S[1]).toBeGreaterThanOrEqual(r.S[2]);
  expect(r.S[2]).toBeGreaterThanOrEqual(0);
  expect(Math.abs(Math.abs(det3(r.U)) - 1)).toBeLessThan(1e-13);
  expect(Math.abs(Math.abs(det3(r.V)) - 1)).toBeLessThan(1e-13);
  return r;
}

describe('svd3', () => {
  it('random matrices', () => {
    const rng = new Rng(20);
    for (let i = 0; i < 2000; i++) {
      const A = new Float64Array(9);
      const sc = Math.pow(10, rng.uniform(-5, 5));
      for (let k = 0; k < 9; k++) A[k] = rng.gaussian() * sc;
      checkSvd3(A);
    }
  });

  it('rank-1, rank-2, zero, identity, rotation, reflection', () => {
    const rng = new Rng(21);
    for (let i = 0; i < 300; i++) {
      const u = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
      const v = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
      const w = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
      const z = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
      const r1 = new Float64Array(9);
      const r2 = new Float64Array(9);
      for (let a = 0; a < 3; a++)
        for (let b = 0; b < 3; b++) {
          r1[3 * a + b] = u[a] * v[b];
          r2[3 * a + b] = u[a] * v[b] + w[a] * z[b];
        }
      const s1 = checkSvd3(r1);
      expect(s1.S[1]).toBeLessThan(1e-15 * s1.S[0]);
      expect(s1.S[2]).toBeLessThan(1e-15 * s1.S[0]);
      const s2 = checkSvd3(r2);
      expect(s2.S[2]).toBeLessThan(1e-15 * s2.S[0]);
    }
    const z = checkSvd3(new Float64Array(9));
    expect(Array.from(z.S)).toEqual([0, 0, 0]);
    const I = checkSvd3(new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
    expect(Array.from(I.S)).toEqual([1, 1, 1]);
    // Reflection: singular values all 1, U Vᵀ has det −1.
    const F = checkSvd3(new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, -1]));
    for (const s of F.S) expect(Math.abs(s - 1)).toBeLessThan(1e-15);
    expect(det3(F.U) * det3(F.V)).toBeLessThan(0);
    // Essential-matrix-like: singular values (1, 1, 0).
    const E = new Float64Array([0, -1, 0.3, 1, 0, -0.2, -0.3, 0.2, 0]); // skew matrix
    const Es = checkSvd3(E);
    expect(Math.abs(Es.S[0] - Es.S[1])).toBeLessThan(1e-14);
    expect(Es.S[2]).toBeLessThan(1e-15);
  });

  it('timing', () => {
    const rng = new Rng(22);
    const A = new Float64Array(9);
    for (let k = 0; k < 9; k++) A[k] = rng.gaussian();
    const t0 = performance.now();
    for (let i = 0; i < 20000; i++) {
      A[i % 9] += 1e-3;
      svd3(A);
    }
    const us = ((performance.now() - t0) / 20000) * 1000;
    console.log(`[math timing] svd3: ${us.toFixed(2)} µs`);
    expect(us).toBeLessThan(500);
  });
});

// ---------------------------------------------------------------------------
// symEig
// ---------------------------------------------------------------------------

describe('symEig', () => {
  it('random symmetric matrices: A V = V Λ, VᵀV = I, ascending', () => {
    const rng = new Rng(30);
    for (const n of [1, 2, 3, 4, 6, 9, 12, 20]) {
      for (let trial = 0; trial < 20; trial++) {
        const B = randMatrix(rng, n, n);
        const A = new Float64Array(n * n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i * n + j] = B.get(i, j) + B.get(j, i);
        const Acopy = A.slice();
        const { values, vectors } = symEig(A, n);
        expect(Array.from(A)).toEqual(Array.from(Acopy));
        for (let k = 1; k < n; k++) expect(values[k]).toBeGreaterThanOrEqual(values[k - 1]);
        const V = new Matrix(n, n, vectors);
        expect(orthoError(V)).toBeLessThan(1e-13);
        const scale = maxAbs(A);
        for (let k = 0; k < n; k++) {
          for (let i = 0; i < n; i++) {
            let s = 0;
            for (let j = 0; j < n; j++) s += A[i * n + j] * vectors[j * n + k];
            expect(Math.abs(s - values[k] * vectors[i * n + k])).toBeLessThan(1e-13 * scale);
          }
        }
      }
    }
  });

  it('repeated eigenvalues and diagonal input', () => {
    const I = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const r = symEig(I, 3);
    expect(Array.from(r.values)).toEqual([1, 1, 1]);
    const D = new Float64Array([3, 0, 0, 0, -1, 0, 0, 0, 3]);
    const rd = symEig(D, 3);
    expect(Array.from(rd.values)).toEqual([-1, 3, 3]);
    expect(Math.abs(rd.vectors[1 * 3 + 0])).toBe(1); // eigenvector of −1 is e_1
    // Known: [[2,1],[1,2]] → 1, 3
    const r2 = symEig(new Float64Array([2, 1, 1, 2]), 2);
    expect(r2.values[0]).toBeCloseTo(1, 14);
    expect(r2.values[1]).toBeCloseTo(3, 14);
    expect(Math.abs(r2.vectors[0] * r2.vectors[2] + r2.vectors[1] * r2.vectors[3])).toBeLessThan(1e-15);
    // Repeated cluster from Q diag Qᵀ
    const rng = new Rng(31);
    const Q = randOrthonormal(rng, 6, 6);
    const lam = [5, 5, 5, -2, -2, 1e-9];
    const A = new Float64Array(36);
    for (let i = 0; i < 6; i++)
      for (let j = 0; j < 6; j++) {
        let s = 0;
        for (let k = 0; k < 6; k++) s += Q.get(i, k) * lam[k] * Q.get(j, k);
        A[i * 6 + j] = s;
      }
    const e = symEig(A, 6);
    const sorted = [...lam].sort((a, b) => a - b);
    for (let k = 0; k < 6; k++) expect(Math.abs(e.values[k] - sorted[k])).toBeLessThan(1e-13);
    expect(orthoError(new Matrix(6, 6, e.vectors))).toBeLessThan(1e-13);
  });

  it('timing (12x12, as in EPnP)', () => {
    const rng = new Rng(32);
    const B = randMatrix(rng, 12, 12);
    const A = B.transposeMul(B).data;
    for (let i = 0; i < 500; i++) symEig(A, 12);
    const t0 = performance.now();
    for (let i = 0; i < 3000; i++) symEig(A, 12);
    const us = ((performance.now() - t0) / 3000) * 1000;
    console.log(`[math timing] symEig 12x12: ${us.toFixed(1)} µs`);
    expect(us).toBeLessThan(5000);
  });

  it('uses the symmetric part of a slightly asymmetric input', () => {
    const A = new Float64Array([2, 1 + 1e-9, 1 - 1e-9, 2]);
    const r = symEig(A, 2);
    expect(r.values[0]).toBeCloseTo(1, 12);
    expect(r.values[1]).toBeCloseTo(3, 12);
  });
});

// ---------------------------------------------------------------------------
// Cholesky / LU / QR / least squares
// ---------------------------------------------------------------------------

function randSPD(rng: Rng, n: number, cond = 1): Float64Array {
  const B = randMatrix(rng, n + 2, n);
  const G = B.transposeMul(B);
  for (let i = 0; i < n; i++) G.data[i * n + i] += cond;
  return G.data;
}

describe('cholesky', () => {
  it('factorises random SPD matrices and solves', () => {
    const rng = new Rng(40);
    for (const n of [1, 2, 3, 6, 12, 50]) {
      for (let trial = 0; trial < 10; trial++) {
        const A = randSPD(rng, n);
        const L = A.slice();
        expect(cholesky(L, n)).toBe(true);
        let err = 0;
        for (let i = 0; i < n; i++)
          for (let j = 0; j < n; j++) {
            if (j > i) expect(L[i * n + j]).toBe(0);
            let s = 0;
            for (let k = 0; k < n; k++) s += L[i * n + k] * L[j * n + k];
            err = Math.max(err, Math.abs(s - A[i * n + j]));
          }
        expect(err / maxAbs(A)).toBeLessThan(1e-14);
        const xTrue = new Float64Array(n).map(() => rng.gaussian());
        const b = new Float64Array(n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) b[i] += A[i * n + j] * xTrue[j];
        const x = b.slice();
        choleskySolve(L, n, x);
        for (let i = 0; i < n; i++) expect(Math.abs(x[i] - xTrue[i])).toBeLessThan(1e-10);
        const Acopy = A.slice(),
          bcopy = b.slice();
        const x2 = solveSPD(A, n, b) as Float64Array;
        expect(Array.from(A)).toEqual(Array.from(Acopy));
        expect(Array.from(b)).toEqual(Array.from(bcopy));
        for (let i = 0; i < n; i++) expect(Math.abs(x2[i] - xTrue[i])).toBeLessThan(1e-10);
      }
    }
  });

  it('rejects indefinite, semi-definite, zero and non-finite matrices', () => {
    expect(cholesky(new Float64Array([1, 2, 2, 1]), 2)).toBe(false); // indefinite
    expect(cholesky(new Float64Array([-1, 0, 0, 1]), 2)).toBe(false);
    expect(cholesky(new Float64Array([0, 0, 0, 0]), 2)).toBe(false);
    expect(cholesky(new Float64Array([1, 1, 1, 1]), 2)).toBe(false); // PSD, rank 1
    expect(cholesky(new Float64Array([NaN, 0, 0, 1]), 2)).toBe(false);
    // Rank-deficient Gram matrix (gauge freedom), computed in floating point.
    const rng = new Rng(41);
    const B = randMatrix(rng, 3, 6);
    const G = B.transposeMul(B).data;
    expect(cholesky(G.slice(), 6)).toBe(false);
    expect(solveSPD(G, 6, new Float64Array(6))).toBeNull();
    // Ill-conditioned but SPD (cond ~1e10) still factorises.
    const D = new Float64Array([1, 0, 0, 1e-10]);
    expect(cholesky(D, 2)).toBe(true);
  });
});

describe('luSolve', () => {
  it('solves random systems with pivoting', () => {
    const rng = new Rng(50);
    for (const n of [1, 2, 3, 5, 9, 30]) {
      for (let trial = 0; trial < 20; trial++) {
        const A = randMatrix(rng, n, n).data;
        const xTrue = new Float64Array(n).map(() => rng.gaussian());
        const b = new Float64Array(n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) b[i] += A[i * n + j] * xTrue[j];
        const Ac = A.slice(),
          bc = b.slice();
        const x = luSolve(A, n, b) as Float64Array;
        expect(x).not.toBeNull();
        expect(Array.from(A)).toEqual(Array.from(Ac));
        expect(Array.from(b)).toEqual(Array.from(bc));
        // residual
        let res = 0;
        for (let i = 0; i < n; i++) {
          let s = 0;
          for (let j = 0; j < n; j++) s += A[i * n + j] * x[j];
          res = Math.max(res, Math.abs(s - b[i]));
        }
        expect(res).toBeLessThan(1e-11 * (1 + maxAbs(b)));
      }
    }
  });

  it('requires pivoting, handles row scaling, detects singularity', () => {
    const x = luSolve(new Float64Array([0, 1, 1, 0]), 2, new Float64Array([2, 3])) as Float64Array;
    expect(Array.from(x)).toEqual([3, 2]);
    const y = luSolve(new Float64Array([1e20, 0, 0, 1]), 2, new Float64Array([1e20, 2])) as Float64Array;
    expect(y[0]).toBeCloseTo(1, 14);
    expect(y[1]).toBeCloseTo(2, 14);
    expect(luSolve(new Float64Array([1, 2, 2, 4]), 2, new Float64Array([1, 1]))).toBeNull();
    expect(luSolve(new Float64Array([0, 0, 0, 1]), 2, new Float64Array([1, 1]))).toBeNull();
    // Singular 3x3 (row 3 = row 1 + row 2) computed in floating point.
    expect(luSolve(new Float64Array([0.1, 0.2, 0.3, 0.7, 0.11, 0.13, 0.8, 0.31, 0.43]), 3, new Float64Array(3))).toBeNull();
  });
});

describe('qr', () => {
  it('thin QR of tall, wide and square matrices', () => {
    const rng = new Rng(60);
    for (const [m, n] of [
      [1, 1],
      [5, 3],
      [3, 5],
      [9, 9],
      [100, 12],
      [4, 1],
      [1, 4],
    ]) {
      for (let trial = 0; trial < 10; trial++) {
        const A = randMatrix(rng, m, n);
        const { Q, R } = qr(A);
        const k = Math.min(m, n);
        expect(Q.rows).toBe(m);
        expect(Q.cols).toBe(k);
        expect(R.rows).toBe(k);
        expect(R.cols).toBe(n);
        expect(orthoError(Q)).toBeLessThan(1e-14);
        for (let i = 0; i < k; i++) {
          expect(R.get(i, i)).toBeGreaterThanOrEqual(0);
          for (let j = 0; j < Math.min(i, n); j++) expect(R.get(i, j)).toBe(0);
        }
        const QR = Q.mul(R);
        for (let i = 0; i < m * n; i++) expect(Math.abs(QR.data[i] - A.data[i])).toBeLessThan(1e-13);
      }
    }
  });

  it('is scale invariant (1e±200)', () => {
    const rng = new Rng(61);
    const A = randMatrix(rng, 6, 4);
    for (const sc of [1e200, 1e-200]) {
      const B = new Matrix(6, 4, A.data.map((v) => v * sc));
      const { Q, R } = qr(B);
      expect(orthoError(Q)).toBeLessThan(1e-14);
      const QR = Q.mul(R);
      for (let i = 0; i < 24; i++) expect(Math.abs(QR.data[i] - B.data[i]) / sc).toBeLessThan(1e-13);
      const xTrue = [1, -2, 3, 0.5];
      const x = solveLeastSquares(B, B.mulVec(xTrue)) as Float64Array;
      for (let i = 0; i < 4; i++) expect(Math.abs(x[i] - xTrue[i])).toBeLessThan(1e-12);
    }
  });

  it('rank-deficient input still gives orthonormal Q and exact reconstruction', () => {
    const A = Matrix.fromRows([
      [1, 2, 3],
      [2, 4, 6],
      [1, 0, 1],
      [0, 0, 0],
    ]);
    const { Q, R } = qr(A);
    expect(orthoError(Q)).toBeLessThan(1e-14);
    const QR = Q.mul(R);
    for (let i = 0; i < 12; i++) expect(Math.abs(QR.data[i] - A.data[i])).toBeLessThan(1e-14);
    const Z = qr(new Matrix(3, 2));
    expect(orthoError(Z.Q)).toBeLessThan(1e-15);
  });
});

describe('solveLeastSquares', () => {
  it('recovers exact and noisy least-squares solutions', () => {
    const rng = new Rng(70);
    for (let trial = 0; trial < 20; trial++) {
      const A = randMatrix(rng, 40, 6);
      const xTrue = new Float64Array(6).map(() => rng.gaussian());
      const b = A.mulVec(xTrue);
      const x = solveLeastSquares(A, b) as Float64Array;
      for (let i = 0; i < 6; i++) expect(Math.abs(x[i] - xTrue[i])).toBeLessThan(1e-12);
      // Noisy: compare with the normal equations.
      for (let i = 0; i < 40; i++) b[i] += 0.1 * rng.gaussian();
      const x2 = solveLeastSquares(A, b) as Float64Array;
      const N = A.transposeMul(A);
      const x3 = solveSPD(N.data, 6, A.transposeMulVec(b)) as Float64Array;
      for (let i = 0; i < 6; i++) expect(Math.abs(x2[i] - x3[i])).toBeLessThan(1e-10);
    }
    expect(solveLeastSquares(new Matrix(2, 3), new Float64Array(2))).toBeNull();
    const D = Matrix.fromRows([
      [1, 2],
      [2, 4],
      [3, 6],
    ]);
    expect(solveLeastSquares(D, [1, 2, 3])).toBeNull();
  });
});
