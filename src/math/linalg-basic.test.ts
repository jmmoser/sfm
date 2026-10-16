import { describe, expect, it } from 'vitest';
import {
  mat3Add,
  mat3Copy,
  mat3Det,
  mat3FrobeniusNorm,
  mat3FromColumns,
  mat3FromValues,
  mat3GetColumn,
  mat3GetRow,
  mat3Identity,
  mat3Inv,
  mat3Mul,
  mat3MulTranspose,
  mat3MulVec,
  mat3Outer,
  mat3Scale,
  mat3Skew,
  mat3Sub,
  mat3Trace,
  mat3Transpose,
  mat3TransposeMul,
  mat3TransposeMulVec,
} from './mat3';
import { Matrix } from './matrix';
import { Rng } from './random';
import {
  vec3,
  vec3Add,
  vec3AddScaled,
  vec3Angle,
  vec3Copy,
  vec3Cross,
  vec3Dist,
  vec3DistSq,
  vec3Dot,
  vec3Lerp,
  vec3Norm,
  vec3NormSq,
  vec3Normalize,
  vec3Scale,
  vec3Set,
  vec3Sub,
} from './vec';
import * as math from './index';

function rand9(rng: Rng): Float64Array {
  return Float64Array.from({ length: 9 }, () => rng.gaussian());
}

function refMul(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const o = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) o[3 * i + j] += a[3 * i + k] * b[3 * k + j];
  return o;
}

function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

describe('vec3', () => {
  it('basic operations and aliasing', () => {
    const a = vec3(1, 2, 3),
      b = vec3(-4, 5, 0.5);
    expect(Array.from(vec3())).toEqual([0, 0, 0]);
    expect(Array.from(vec3Add(a, b))).toEqual([-3, 7, 3.5]);
    expect(Array.from(vec3Sub(a, b))).toEqual([5, -3, 2.5]);
    expect(Array.from(vec3Scale(a, 2))).toEqual([2, 4, 6]);
    expect(Array.from(vec3AddScaled(a, b, 2))).toEqual([-7, 12, 4]);
    expect(vec3Dot(a, b)).toBe(-4 + 10 + 1.5);
    expect(Array.from(vec3Cross([1, 0, 0], [0, 1, 0]))).toEqual([0, 0, 1]);
    expect(vec3Norm([3, 4, 12])).toBe(13);
    expect(vec3NormSq([3, 4, 12])).toBe(169);
    expect(vec3Norm([3e200, 4e200, 0])).toBeCloseTo(5e200, -186);
    expect(vec3Norm([3e-200, 4e-200, 0]) / 5e-200).toBeCloseTo(1, 14);
    expect(Array.from(vec3Normalize([0, 0, 2]))).toEqual([0, 0, 1]);
    expect(Array.from(vec3Normalize([0, 0, 0]))).toEqual([0, 0, 0]);
    expect(vec3Dist(a, [1, 2, 5])).toBe(2);
    expect(vec3DistSq(a, [1, 2, 5])).toBe(4);
    expect(Array.from(vec3Copy([7, 8, 9, 10]))).toEqual([7, 8, 9]);
    expect(Array.from(vec3Set(vec3(), 1, 2, 3))).toEqual([1, 2, 3]);
    expect(Array.from(vec3Lerp([0, 0, 0], [2, 4, 6], 0.5))).toEqual([1, 2, 3]);
    expect(vec3Angle([1, 0, 0], [0, 3, 0])).toBeCloseTo(Math.PI / 2, 15);
    expect(vec3Angle([1, 0, 0], [1, 1e-12, 0])).toBeCloseTo(1e-12, 25);
    // aliasing: out = a
    const c = vec3(1, 2, 3);
    vec3Cross(c, [0, 0, 1], c);
    expect(Array.from(c)).toEqual([2, -1, 0]);
    const d = vec3(1, 2, 3);
    vec3Add(d, d, d);
    expect(Array.from(d)).toEqual([2, 4, 6]);
  });

  it('cross product identities on random vectors', () => {
    const rng = new Rng(1);
    for (let i = 0; i < 1000; i++) {
      const a = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
      const b = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
      const c = vec3Cross(a, b);
      expect(Math.abs(vec3Dot(c, a))).toBeLessThan(1e-14);
      expect(Math.abs(vec3Dot(c, b))).toBeLessThan(1e-14);
      expect(maxDiff(mat3MulVec(mat3Skew(a), b), c)).toBeLessThan(1e-15);
    }
  });
});

describe('mat3', () => {
  it('products, transposes and aliasing', () => {
    const rng = new Rng(2);
    for (let i = 0; i < 500; i++) {
      const a = rand9(rng),
        b = rand9(rng),
        v = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
      const ab = refMul(a, b);
      expect(maxDiff(mat3Mul(a, b), ab)).toBeLessThan(1e-14);
      expect(maxDiff(mat3MulTranspose(a, b), refMul(a, mat3Transpose(b)))).toBeLessThan(1e-14);
      expect(maxDiff(mat3TransposeMul(a, b), refMul(mat3Transpose(a), b))).toBeLessThan(1e-14);
      expect(maxDiff(mat3TransposeMulVec(a, v), mat3MulVec(mat3Transpose(a), v))).toBeLessThan(1e-14);
      // aliasing out = a and out = b
      const a2 = a.slice();
      mat3Mul(a2, b, a2);
      expect(maxDiff(a2, ab)).toBe(0);
      const b2 = b.slice();
      mat3Mul(a, b2, b2);
      expect(maxDiff(b2, ab)).toBe(0);
      const t = a.slice();
      mat3Transpose(t, t);
      expect(maxDiff(t, mat3Transpose(a))).toBe(0);
      const v2 = Float64Array.from(v);
      mat3MulVec(a, v2, v2);
      expect(maxDiff(v2, mat3MulVec(a, v))).toBe(0);
      // det(AB) = det A det B
      expect(Math.abs(mat3Det(ab) - mat3Det(a) * mat3Det(b))).toBeLessThan(1e-12 * (1 + Math.abs(mat3Det(ab))));
    }
  });

  it('inverse: A A⁻¹ = I; singular → null; scale invariant', () => {
    const rng = new Rng(3);
    for (let i = 0; i < 500; i++) {
      const a = rand9(rng);
      const inv = mat3Inv(a);
      if (!inv) continue;
      expect(maxDiff(mat3Mul(a, inv), mat3Identity())).toBeLessThan(1e-9);
    }
    expect(mat3Inv([1, 2, 3, 2, 4, 6, 0, 1, 1])).toBeNull();
    expect(mat3Inv(new Float64Array(9))).toBeNull();
    expect(mat3Inv([1, 0, 0, 0, NaN, 0, 0, 0, 1])).toBeNull();
    // Badly row-scaled but well-conditioned (K-matrix-like).
    const K = [1500, 0, 960, 0, 1500, 540, 0, 0, 1];
    const Ki = mat3Inv(K) as Float64Array;
    expect(maxDiff(mat3Mul(K, Ki), mat3Identity())).toBeLessThan(1e-13);
    const tiny = mat3Inv([1e-200, 0, 0, 0, 1e-200, 0, 0, 0, 1e-200]) as Float64Array;
    expect(tiny[0]).toBeCloseTo(1e200, -186);
    // aliasing
    const a = rand9(rng);
    const ref = mat3Inv(a) as Float64Array;
    mat3Inv(a, a);
    expect(maxDiff(a, ref)).toBe(0);
  });

  it('misc helpers', () => {
    const a = mat3FromValues(1, 2, 3, 4, 5, 6, 7, 8, 10);
    expect(mat3Det(a)).toBeCloseTo(-3, 13);
    expect(mat3Trace(a)).toBe(16);
    expect(Array.from(mat3Scale(a, 2))[8]).toBe(20);
    expect(Array.from(mat3Add(a, a))[1]).toBe(4);
    expect(Array.from(mat3Sub(a, a))).toEqual(new Array(9).fill(0));
    expect(mat3FrobeniusNorm(mat3Identity())).toBeCloseTo(Math.sqrt(3), 15);
    expect(Array.from(mat3Outer([1, 2, 3], [1, 0, -1]))).toEqual([1, 0, -1, 2, 0, -2, 3, 0, -3]);
    const c = mat3FromColumns([1, 4, 7], [2, 5, 8], [3, 6, 10]);
    expect(Array.from(c)).toEqual(Array.from(a));
    expect(Array.from(mat3GetColumn(a, 1))).toEqual([2, 5, 8]);
    expect(Array.from(mat3GetRow(a, 2))).toEqual([7, 8, 10]);
    expect(Array.from(mat3Copy(a))).toEqual(Array.from(a));
    const S = mat3Skew([1, 2, 3]);
    expect(maxDiff(mat3Transpose(S), mat3Scale(S, -1))).toBe(0);
  });
});

describe('Matrix', () => {
  it('construction, access, transpose, products', () => {
    const A = Matrix.fromRows([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(A.rows).toBe(2);
    expect(A.cols).toBe(3);
    expect(A.get(1, 2)).toBe(6);
    A.set(0, 0, 10);
    expect(A.data[0]).toBe(10);
    A.set(0, 0, 1);
    const At = A.transpose();
    expect(At.rows).toBe(3);
    expect(At.get(2, 1)).toBe(6);
    const AtA = At.mul(A);
    expect(Array.from(AtA.data)).toEqual(Array.from(A.transposeMul(A).data));
    expect(Array.from(AtA.data)).toEqual([17, 22, 27, 22, 29, 36, 27, 36, 45]);
    expect(Array.from(A.mulVec([1, 1, 1]))).toEqual([6, 15]);
    expect(Array.from(A.transposeMulVec([1, 1]))).toEqual([5, 7, 9]);
    expect(Array.from(Matrix.identity(2).data)).toEqual([1, 0, 0, 1]);
    expect(Matrix.zeros(2, 3).data.length).toBe(6);
    const B = A.clone();
    B.set(0, 0, 99);
    expect(A.get(0, 0)).toBe(1);
    expect(Array.from(A.getRow(1))).toEqual([4, 5, 6]);
    expect(Array.from(A.getColumn(1))).toEqual([2, 5]);
    expect(A.frobeniusNorm()).toBeCloseTo(Math.sqrt(91), 14);
    // Float64Array data is wrapped, not copied; number[] is copied.
    const d = new Float64Array(4);
    const W = new Matrix(2, 2, d);
    W.set(1, 1, 5);
    expect(d[3]).toBe(5);
    expect(() => new Matrix(2, 2, [1, 2, 3])).toThrow(RangeError);
    expect(() => A.mul(A)).toThrow(RangeError);
    expect(() => Matrix.fromRows([[1, 2], [3]])).toThrow(RangeError);
    expect(Matrix.fromRows([]).rows).toBe(0);
  });

  it('random products agree with naive reference', () => {
    const rng = new Rng(4);
    for (let t = 0; t < 50; t++) {
      const m = 1 + rng.int(8),
        k = 1 + rng.int(8),
        n = 1 + rng.int(8);
      const A = new Matrix(m, k, Float64Array.from({ length: m * k }, () => rng.gaussian()));
      const B = new Matrix(k, n, Float64Array.from({ length: k * n }, () => rng.gaussian()));
      const C = A.mul(B);
      for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let l = 0; l < k; l++) s += A.get(i, l) * B.get(l, j);
          expect(Math.abs(C.get(i, j) - s)).toBeLessThan(1e-13);
        }
    }
  });
});

describe('index re-exports', () => {
  it('exposes the documented API', () => {
    const names = [
      'vec3', 'vec3Add', 'vec3Sub', 'vec3Scale', 'vec3Dot', 'vec3Cross', 'vec3Norm', 'vec3Normalize', 'vec3Dist',
      'vec3Copy', 'mat3Identity', 'mat3Copy', 'mat3Mul', 'mat3MulVec', 'mat3TransposeMulVec', 'mat3MulTranspose',
      'mat3TransposeMul', 'mat3Transpose', 'mat3Det', 'mat3Inv', 'mat3Skew', 'mat3Scale', 'mat3Add',
      'mat3FrobeniusNorm', 'Matrix', 'cholesky', 'choleskySolve', 'solveSPD', 'luSolve', 'qr', 'svd', 'svd3',
      'nullVector', 'symEig', 'polyEval', 'polyRealRoots', 'solveQuadratic', 'solveCubic', 'solveQuartic',
      'quatToMat3', 'mat3ToQuat', 'quatMul', 'quatConj', 'quatNormalize', 'quatSlerp', 'rotvecToMat3',
      'mat3ToRotvec', 'axisAngleToMat3', 'rotationAngle', 'rotationAngleBetween', 'orthonormalize',
      'rotationBetweenVectors', 'Rng', 'median', 'percentile', 'mean', 'mad', 'quickselect', 'umeyama', 'applySim3',
    ];
    const m = math as unknown as Record<string, unknown>;
    for (const n of names) expect(typeof m[n], n).toBe('function');
  });
});
