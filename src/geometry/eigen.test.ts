import { describe, expect, it } from 'vitest';
import { Rng } from '../math/random';
import { luSolve } from '../math/decomp';
import { eigenvaluesNonsymmetric, eigenvectorInverseIteration } from './eigen';

/** A = V D V⁻¹ with D block-diagonal (real eigenvalues and 2×2 blocks [[a, b], [−b, a]]). */
function makeMatrix(rng: Rng, n: number, real: number[], complex: [number, number][]) {
  const D = new Float64Array(n * n);
  let k = 0;
  for (const v of real) {
    D[k * n + k] = v;
    k++;
  }
  for (const [a, b] of complex) {
    D[k * n + k] = a;
    D[k * n + k + 1] = b;
    D[(k + 1) * n + k] = -b;
    D[(k + 1) * n + k + 1] = a;
    k += 2;
  }
  const V = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) V[i] = rng.gaussian();
  // V⁻¹ column by column
  const Vi = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    const e = new Float64Array(n);
    e[j] = 1;
    const c = luSolve(V, n, e)!;
    for (let i = 0; i < n; i++) Vi[i * n + j] = c[i];
  }
  const A = new Float64Array(n * n);
  const T = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let q = 0; q < n; q++) s += V[i * n + q] * D[q * n + j];
      T[i * n + j] = s;
    }
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let q = 0; q < n; q++) s += T[i * n + q] * Vi[q * n + j];
      A[i * n + j] = s;
    }
  return A;
}

describe('nonsymmetric eigenvalues (balance + Hessenberg + Francis QR)', () => {
  it('recovers real and complex eigenvalues of random 10×10 matrices', () => {
    const rng = new Rng(21);
    const wr = new Float64Array(10),
      wi = new Float64Array(10);
    for (let trial = 0; trial < 200; trial++) {
      const real = [rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5)];
      const cpx: [number, number][] = [
        [rng.uniform(-3, 3), rng.uniform(0.5, 3)],
        [rng.uniform(-3, 3), rng.uniform(0.5, 3)],
        [rng.uniform(-3, 3), rng.uniform(0.5, 3)],
      ];
      const A = makeMatrix(rng, 10, real, cpx);
      expect(eigenvaluesNonsymmetric(A, 10, wr, wi)).toBe(true);
      const got = Array.from(wr, (r, i) => [r, wi[i]]);
      const want: [number, number][] = [...real.map((r) => [r, 0] as [number, number])];
      for (const [a, b] of cpx) want.push([a, b], [a, -b]);
      for (const [r, im] of want) {
        let best = Infinity;
        for (const [gr, gi] of got) best = Math.min(best, Math.hypot(gr - r, gi - im));
        expect(best).toBeLessThan(1e-7);
      }
    }
  });

  it('handles badly scaled matrices (balancing) and trivial sizes', () => {
    const wr = new Float64Array(3),
      wi = new Float64Array(3);
    // upper triangular with wild scaling: eigenvalues are the diagonal
    const A = Float64Array.of(1, 1e8, 1e-8, 0, 2, 1e10, 0, 0, 3);
    expect(eigenvaluesNonsymmetric(A, 3, wr, wi)).toBe(true);
    expect(Array.from(wr).sort((a, b) => a - b)).toEqual([1, 2, 3].map((v) => expect.closeTo(v, 8)));
    const w1 = new Float64Array(1),
      i1 = new Float64Array(1);
    expect(eigenvaluesNonsymmetric([7], 1, w1, i1)).toBe(true);
    expect(w1[0]).toBe(7);
    expect(i1[0]).toBe(0);
    // rotation by 90°: ±i
    const w2 = new Float64Array(2),
      i2 = new Float64Array(2);
    eigenvaluesNonsymmetric([0, -1, 1, 0], 2, w2, i2);
    expect(Math.abs(w2[0])).toBeLessThan(1e-15);
    expect(Math.abs(i2[0])).toBeCloseTo(1, 14);
  });

  it('inverse iteration gives eigenvectors of real eigenvalues', () => {
    const rng = new Rng(22);
    const wr = new Float64Array(8),
      wi = new Float64Array(8),
      v = new Float64Array(8);
    for (let trial = 0; trial < 100; trial++) {
      const A = makeMatrix(rng, 8, [1, -2, 3.5, 0.25], [
        [0.5, 1],
        [-1, 2],
      ]);
      eigenvaluesNonsymmetric(A, 8, wr, wi);
      for (let k = 0; k < 8; k++) {
        if (wi[k] !== 0) continue;
        const res = eigenvectorInverseIteration(A, 8, wr[k], v);
        expect(res).toBeLessThan(1e-10);
        expect(Math.hypot(...v)).toBeCloseTo(1, 12);
      }
    }
  });
});
