/**
 * Adversarial regression tests for src/math (written during the review pass). Each
 * block names the defect it guards against; all tests are deterministic.
 */
import { describe, expect, it } from 'vitest';
import { cholesky, luSolve, nullSpace, nullVector, qr, solveLeastSquares, solveSPD, svd, svd3, symEig } from './decomp';
import { mat3Det, mat3Mul, mat3Transpose } from './mat3';
import { Matrix } from './matrix';
import { polyFromRoots, polyMul, polyRealRoots, polyRealRootsSturm, solveCubic, solveQuadratic, solveQuartic } from './poly';
import { Rng } from './random';
import {
  axisAngleToMat3,
  mat3ToQuat,
  mat3ToRotvec,
  orthonormalize,
  quatNormalize,
  quatToMat3,
  rotationAngle,
  rotationBetweenVectors,
  rotvecToMat3,
} from './rotation';
import { applySim3, umeyama } from './sim3';
import { mad, median, percentile, quickselect } from './stats';
import { vec3Normalize } from './vec';

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

/** Every returned root matches the expected one within rel·max(1,|x|); same count. */
function expectRootsRel(got: number[], want: number[], rel: number): void {
  const sorted = [...want].sort((a, b) => a - b);
  expect(got.length, `got ${JSON.stringify(got)} want ${JSON.stringify(sorted)}`).toBe(sorted.length);
  for (let i = 0; i < sorted.length; i++) {
    expect(Math.abs(got[i] - sorted[i]), `root ${i}: ${got[i]} vs ${sorted[i]}`).toBeLessThanOrEqual(
      rel * Math.max(1, Math.abs(sorted[i])),
    );
  }
}

function allNaN(a: ArrayLike<number>): boolean {
  for (let i = 0; i < a.length; i++) if (!Number.isNaN(a[i])) return false;
  return a.length > 0;
}

/** max |MᵀM − I| for a matrix with orthonormal columns. */
function orthError(M: Matrix): number {
  let e = 0;
  for (let i = 0; i < M.cols; i++)
    for (let j = 0; j < M.cols; j++) {
      let s = 0;
      for (let r = 0; r < M.rows; r++) s += M.data[r * M.cols + i] * M.data[r * M.cols + j];
      e = Math.max(e, Math.abs(s - (i === j ? 1 : 0)));
    }
  return e;
}

// ---------------------------------------------------------------------------
// Polynomials
// ---------------------------------------------------------------------------

describe('poly (adversarial)', () => {
  it('solveQuadratic: no overflow/underflow in b² − 4ac (was [-5e199], [-0.5], [NaN, NaN])', () => {
    expectRootsRel(solveQuadratic(1, 1e200, 1), [-1e200, -1e-200], 1e-15);
    // Relative check for the tiny root.
    const r = solveQuadratic(1, 1e200, 1);
    expect(Math.abs(r[1] + 1e-200) / 1e-200).toBeLessThan(1e-15);
    const g = (1 + Math.sqrt(5)) / 2;
    expectRootsRel(solveQuadratic(1e-200, 1e-200, -1e-200), [-g, g - 1], 1e-15);
    expectRootsRel(solveQuadratic(1e300, 3e300, 2e300), [-2, -1], 1e-15);
    expectRootsRel(solveQuadratic(-1e-300, -3e-300, -2e-300), [-2, -1], 1e-15);
    expectRootsRel(solveQuadratic(1e-310, 1, -1), [1], 1e-15); // other root ≈ -1e310 overflows → dropped
    expect(solveQuadratic(0, 1e-300, 1e300)).toEqual([]); // linear root overflows
    expect(solveQuadratic(NaN, 1, 1)).toEqual([]);
    expect(solveQuadratic(1, Infinity, 1)).toEqual([]);
    expectRootsRel(polyRealRoots([1, 1e200, 1]), [-1e200, -1e-200], 1e-15);
    expectRootsRel(solveCubic(1, 1e200, 1, 0), [-1e200, -1e-200, 0], 1e-15);
  });

  it('tiny leading coefficient (huge root): small roots are kept, no spurious roots', () => {
    // Before the fix: cubic with 1e-100 → [-1e100, -1.2e8, 2.5e9]; 1e-200 → [-1e200, 1.5];
    // degree 10 with 1e-40 and degree 20 with 1e-20 lost all small roots.
    for (const eps of [1e-16, 1e-50, 1e-100, 1e-150, 1e-200, 1e-300]) {
      expectRootsRel(polyRealRoots(polyMul([eps, 1], polyFromRoots([1, 2]))), [-1 / eps, 1, 2], 1e-13);
    }
    const base10 = [-1.7, -1.1, -0.5, 0.2, 0.9, 1.3, 1.9, 2.5, 3.1];
    for (const eps of [1e-16, 1e-40, 1e-60, 1e-100]) {
      expectRootsRel(polyRealRoots(polyMul([eps, 1], polyFromRoots(base10))), [-1 / eps, ...base10], 1e-9);
    }
    const base20 = Array.from({ length: 19 }, (_, i) => -2 + i * 0.23);
    for (const eps of [1e-16, 1e-20, 1e-40]) {
      expectRootsRel(polyRealRoots(polyMul([eps, 1], polyFromRoots(base20))), [-1 / eps, ...base20], 1e-6);
    }
    expectRootsRel(solveCubic(1e-100, 1, -3, 2), [-1e100, 1, 2], 1e-13);
    expectRootsRel(solveQuartic(1e-60, 1, -6, 11, -6), [-1e60, 1, 2, 3], 1e-12);
    // The deflated monic form overflows (p₁/p₀ = 1e310): coefficients must not be made monic.
    expectRootsRel(polyRealRoots([1e-300, 1e10, 1]), [-1e-10], 1e-15);
    expectRootsRel(polyRealRoots([1e-300, 1e10, -3e10, 2e10]), [1, 2], 1e-13);
  });

  it('roots spanning > 200 orders of magnitude (band split), restricted to [lo, hi]', () => {
    // (1e-100 x + 1)(x − 1)(1e100 x − 1): roots −1e100, 1e-100, 1.
    const c = polyMul(polyMul([1e-100, 1], [1, -1]), [1e100, -1]);
    const all = polyRealRoots(c);
    expect(all.length).toBe(3);
    expect(Math.abs(all[0] + 1e100) / 1e100).toBeLessThan(1e-14);
    expect(Math.abs(all[1] - 1e-100) / 1e-100).toBeLessThan(1e-14);
    expect(Math.abs(all[2] - 1)).toBeLessThan(1e-14);
    const pos = polyRealRoots(c, 0, Infinity);
    expect(pos.length).toBe(2);
    expect(Math.abs(pos[0] - 1e-100) / 1e-100).toBeLessThan(1e-14);
    expect(polyRealRoots(c, -Infinity, 0)).toEqual([all[0]]);
    expect(polyRealRoots(c, 0.5, 2)).toEqual([all[2]]);
  });

  it('roots ~1e-80 inside an O(1) bracket converge (rtsafe iteration cap was 300)', () => {
    // x³ − x² + c: roots ±√c (to first order) and ≈ 1.
    for (const c of [1e-120, 1e-160, 1e-240]) {
      const r = polyRealRoots([1, -1, 0, c]);
      const s = Math.sqrt(c);
      expect(r.length).toBe(3);
      expect(Math.abs(r[0] + s) / s).toBeLessThan(1e-12);
      expect(Math.abs(r[1] - s) / s).toBeLessThan(1e-12);
      expect(Math.abs(r[2] - 1)).toBeLessThan(1e-14);
    }
  });

  it('resolvable close roots are not merged; unresolvable clusters are', () => {
    // (x−1)(x−1−d)(x−5): |p(mid)| = d²·~1 ≫ rounding noise (~1e-15) for d ≥ 3e-7.
    // The old 8×/64× noise factors merged d = 5e-7 and 3e-7.
    for (const d of [1e-5, 1e-6, 5e-7, 3e-7]) {
      const r = polyRealRoots(polyFromRoots([1, 1 + d, 5]));
      expect(r.length, `d=${d}: ${JSON.stringify(r)}`).toBe(3);
      // forward error ≈ ε Σ|cᵢ| / |p'(root)| ≈ 2.2e-16·24 / (4d)
      const tol = (20 * 2.2e-16 * 24) / (4 * d);
      expect(Math.abs(r[0] - 1)).toBeLessThan(tol);
      expect(Math.abs(r[1] - 1 - d)).toBeLessThan(tol);
      expect(Math.abs(r[2] - 5)).toBeLessThan(1e-13);
    }
    const merged = polyRealRoots(polyFromRoots([1, 1 + 1e-8, 5]));
    expect(merged.length).toBe(2);
    expect(Math.abs(merged[0] - 1)).toBeLessThan(1e-7);
    expect(polyRealRoots(polyFromRoots([0.5, 0.5 + 1e-7, 0.5 + 2e-7])).length).toBe(1);
  });

  it('multiple roots from rounded coefficients are never lost and never split (2..4-fold)', () => {
    const rng = new Rng(12);
    let lost = 0,
      split = 0;
    for (let t = 0; t < 2000; t++) {
      const mult = 2 + rng.int(3);
      const deg = mult + 1 + rng.int(6);
      const scale = Math.pow(10, rng.uniform(-3, 3));
      const r = rng.uniform(-3, 3) * scale;
      const others: number[] = [];
      for (let i = 0; i < deg - mult; i++) {
        let s: number;
        do s = rng.uniform(-3, 3) * scale * (rng.next() < 0.3 ? 30 : 1);
        while (Math.abs(s - r) < 0.3 * scale);
        others.push(s);
      }
      const roots = polyRealRoots(polyFromRoots([...Array<number>(mult).fill(r), ...others]));
      const tol = Math.pow(1e-15, 1 / mult) * 30 * Math.max(Math.abs(r), scale);
      const near = roots.filter((x) => Math.abs(x - r) <= tol).length;
      if (near === 0) lost++;
      if (near > 1) split++;
    }
    expect(lost).toBe(0);
    expect(split).toBe(0);
  });

  it('random tiny-leading-coefficient stress: no spurious roots, all well-separated roots found', () => {
    const rng = new Rng(5);
    let spurious = 0,
      missed = 0;
    for (let t = 0; t < 1500; t++) {
      const deg = 3 + rng.int(8);
      const roots: number[] = [];
      for (let i = 0; i < deg - 1; i++) roots.push(rng.uniform(-3, 3));
      const eps = Math.pow(10, -rng.uniform(5, 300 / deg));
      const got = polyRealRoots(polyMul([eps, 1], polyFromRoots(roots)));
      roots.push(-1 / eps);
      roots.sort((a, b) => a - b);
      let minGap = Infinity;
      for (let i = 1; i < roots.length; i++) minGap = Math.min(minGap, roots[i] - roots[i - 1]);
      for (const x of got) if (!roots.some((e) => Math.abs(x - e) <= 1e-5 * Math.max(1, Math.abs(e)))) spurious++;
      if (minGap > 0.05)
        for (const e of roots) if (!got.some((x) => Math.abs(x - e) <= 1e-5 * Math.max(1, Math.abs(e)))) missed++;
    }
    expect(spurious).toBe(0);
    expect(missed).toBe(0);
  });

  it('degenerate / non-finite coefficient vectors', () => {
    expect(polyRealRoots([NaN, 1, 2, 3])).toEqual([]);
    expect(polyRealRoots([1, Infinity, 2, 3])).toEqual([]);
    expect(polyRealRoots([0, 0, 0])).toEqual([]);
    expect(polyRealRoots([0, 0, 5])).toEqual([]);
    expect(polyRealRoots([])).toEqual([]);
    expect(polyRealRoots([2, 0, 0, 0])).toEqual([0]);
    expectRootsRel(polyRealRoots([5e-324, 1, -1]), [1], 1e-15); // subnormal leading coefficient
    expect(polyRealRoots([1, -3, 2], 3, 2)).toEqual([]); // lo > hi
    expect(polyRealRoots([1, -3, 2], NaN, 2)).toEqual([]);
    // Coefficients scaled by 1e±300 give identical roots.
    for (const s of [1e-300, 1, 1e300]) expectRootsRel(polyRealRoots([s, -6 * s, 11 * s, -6 * s]), [1, 2, 3], 1e-14);
  });

  it('Sturm variant agrees on well-scaled inputs after the prepare() refactor', () => {
    const rng = new Rng(21);
    for (let t = 0; t < 200; t++) {
      const roots = Array.from({ length: 3 + rng.int(6) }, () => rng.uniform(-4, 4)).sort((a, b) => a - b);
      let minGap = Infinity;
      for (let i = 1; i < roots.length; i++) minGap = Math.min(minGap, roots[i] - roots[i - 1]);
      if (minGap < 0.05) continue;
      const c = polyFromRoots(roots);
      expectRootsRel(polyRealRootsSturm(c), roots, 1e-9);
      expectRootsRel(polyRealRoots(c), roots, 1e-9);
    }
  });
});

// ---------------------------------------------------------------------------
// Decompositions
// ---------------------------------------------------------------------------

describe('decomp (adversarial)', () => {
  it('symEig: non-finite input → NaN (was: diagonal + identity, NaN silently dropped)', () => {
    let r = symEig(new Float64Array([1, NaN, NaN, 1]), 2);
    expect(allNaN(r.values) && allNaN(r.vectors)).toBe(true);
    r = symEig(new Float64Array([Infinity, 0, 0, 1]), 2);
    expect(allNaN(r.values)).toBe(true);
    // Finite input is unaffected.
    r = symEig(new Float64Array([2, 1, 1, 2]), 2);
    expect(maxAbsDiff(r.values, [1, 3])).toBeLessThan(1e-15);
  });

  it('symEig: symmetrisation does not overflow for entries near 1e308', () => {
    const A = new Float64Array([1e307, 1.5e308, 1.5e308, 1e307]);
    const { values, vectors } = symEig(A, 2);
    expect(Math.abs(values[0] - (1e307 - 1.5e308)) / 1.5e308).toBeLessThan(1e-14);
    expect(Math.abs(values[1] - (1e307 + 1.5e308)) / 1.5e308).toBeLessThan(1e-14);
    expect(Math.abs(Math.abs(vectors[0]) - Math.SQRT1_2)).toBeLessThan(1e-14);
  });

  it('svd: strongly graded columns (1 … 1e-16) keep orthonormal U, V and reconstruct', () => {
    const rng = new Rng(31);
    for (const [m, n] of [
      [9, 9],
      [12, 9],
      [5, 9],
      [20, 4],
    ]) {
      const A = new Matrix(m, n);
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) A.set(i, j, rng.gaussian() * Math.pow(10, -2 * j));
      const { U, S, V } = svd(A);
      expect(orthError(U)).toBeLessThan(1e-14);
      expect(orthError(V)).toBeLessThan(1e-14);
      let rec = 0;
      for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let k = 0; k < S.length; k++) s += U.get(i, k) * S[k] * V.get(j, k);
          rec = Math.max(rec, Math.abs(s - A.get(i, j)));
        }
      expect(rec).toBeLessThan(1e-14 * A.frobeniusNorm());
      for (let k = 1; k < S.length; k++) expect(S[k]).toBeLessThanOrEqual(S[k - 1]);
    }
  });

  it('nullSpace / nullVector on rank-deficient wide systems (duplicate / dependent rows)', () => {
    const rng = new Rng(32);
    // 3 x 6 with row2 = row0 + row1 (rank 2): 4-D null space via the Jacobi path (dim > n − m).
    const A = new Matrix(3, 6);
    for (let i = 0; i < 2; i++) for (let j = 0; j < 6; j++) A.set(i, j, rng.gaussian());
    for (let j = 0; j < 6; j++) A.set(2, j, A.get(0, j) + A.get(1, j));
    const N = nullSpace(A, 4);
    expect(orthError(N)).toBeLessThan(1e-14);
    for (let c = 0; c < 4; c++) {
      const v = N.getColumn(c);
      expect(Math.max(...Array.from(A.mulVec(v)).map(Math.abs))).toBeLessThan(1e-14);
    }
    // 8 x 9 with two identical rows (rank 7): any unit null vector is acceptable.
    const B = new Matrix(8, 9);
    for (let i = 0; i < 8; i++) for (let j = 0; j < 9; j++) B.set(i, j, rng.gaussian());
    for (let j = 0; j < 9; j++) B.set(7, j, B.get(3, j));
    const x = nullVector(B);
    expect(Math.abs(Math.hypot(...Array.from(x)) - 1)).toBeLessThan(1e-15);
    expect(Math.max(...Array.from(B.mulVec(x)).map(Math.abs))).toBeLessThan(1e-14);
  });

  it('svd3: repeated / negative / rank-deficient spectra reconstruct with orthogonal factors', () => {
    const cases = [
      [2, 0, 0, 0, 2, 0, 0, 0, -2],
      [0, 0, 0, 0, 0, 0, 0, 0, 1e-300],
      [1, 1, 1, 1, 1, 1, 1, 1, 1 + 1e-15],
      [-1, 0, 0, 0, -1, 0, 0, 0, 1],
    ];
    for (const a of cases) {
      const A = new Float64Array(a);
      const { U, S, V } = svd3(A);
      const D = new Float64Array([S[0], 0, 0, 0, S[1], 0, 0, 0, S[2]]);
      const rec = mat3Mul(mat3Mul(U, D), mat3Transpose(V));
      const scale = Math.max(...a.map(Math.abs));
      expect(maxAbsDiff(rec, A)).toBeLessThanOrEqual(4e-16 * scale * 3);
      expect(maxAbsDiff(mat3Mul(mat3Transpose(U), U), [1, 0, 0, 0, 1, 0, 0, 0, 1])).toBeLessThan(1e-15);
      expect(maxAbsDiff(mat3Mul(mat3Transpose(V), V), [1, 0, 0, 0, 1, 0, 0, 0, 1])).toBeLessThan(1e-15);
      expect(S[0] >= S[1] && S[1] >= S[2] && S[2] >= 0).toBe(true);
    }
  });

  it('subnormal-scale input (max |a| < 2^-1024): 1/scale overflowed → NaN / zero outputs', () => {
    const s = 1e-310;
    const base = [1, 2, 0, 0, 1, 3, 1, 0, 1];
    const ref = svd(new Matrix(3, 3, base)).S;
    const A = new Matrix(
      3,
      3,
      base.map((v) => v * s),
    );
    const S = svd(A).S;
    for (let i = 0; i < 3; i++) expect(Math.abs(S[i] / s - ref[i]) / ref[0]).toBeLessThan(1e-9); // subnormals carry ~13 digits
    const S3 = svd3(A.data).S;
    for (let i = 0; i < 3; i++) expect(Math.abs(S3[i] / s - ref[i]) / ref[0]).toBeLessThan(1e-9);
    expect(Array.from(nullVector(new Matrix(2, 3, [s, 0, 0, 0, s, 0]))).map(Math.abs)).toEqual([0, 0, 1]);
    expect(Array.from(nullVector(new Matrix(3, 3, [s, 0, 0, 0, s, 0, 0, 0, 0]))).map(Math.abs)).toEqual([0, 0, 1]);
    const N = nullSpace(new Matrix(2, 3, [s, 0, 0, 0, s, 0]), 2);
    expect(orthError(N)).toBeLessThan(1e-15);
    const { Q, R } = qr(A);
    expect(orthError(Q)).toBeLessThan(1e-15);
    expect(R.data.every((v) => Number.isFinite(v))).toBe(true);
    const x = solveLeastSquares(A, [s, s, s]);
    expect(x).not.toBeNull();
    const xr = solveLeastSquares(new Matrix(3, 3, base), [1, 1, 1])!;
    expect(maxAbsDiff(x!, xr)).toBeLessThan(1e-9);
    expect(maxAbsDiff(orthonormalize(new Float64Array([s, 0, 0, 0, s, 0, 0, 0, s])), [1, 0, 0, 0, 1, 0, 0, 0, 1])).toBe(0);
  });

  it('solvers reject NaN input instead of returning garbage', () => {
    const A = new Float64Array([4, 1, 1, NaN]);
    expect(cholesky(A.slice(), 2)).toBe(false);
    expect(solveSPD(A, 2, new Float64Array([1, 1]))).toBeNull();
    expect(luSolve(A, 2, new Float64Array([1, 1]))).toBeNull();
    expect(luSolve(new Float64Array([1, 0, 0, 1]), 2, new Float64Array([NaN, 1]))).toBeNull();
    expect(solveLeastSquares(new Matrix(3, 2, [1, 0, 0, 1, NaN, 1]), [1, 2, 3])).toBeNull();
    expect(allNaN(svd(new Matrix(3, 2, [1, 0, 0, 1, NaN, 1])).S)).toBe(true);
    expect(allNaN(nullVector(new Matrix(2, 3, [1, 0, 0, 0, Infinity, 1])))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Rotations / vectors
// ---------------------------------------------------------------------------

describe('rotation (adversarial)', () => {
  it('non-finite input propagates as NaN (was silently mapped to identity / zero)', () => {
    const Rnan = new Float64Array(9).fill(NaN);
    expect(allNaN(mat3ToRotvec(Rnan))).toBe(true);
    expect(allNaN(mat3ToQuat(Rnan))).toBe(true);
    const Rinf = new Float64Array([Infinity, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(allNaN(mat3ToRotvec(Rinf))).toBe(true);
    expect(allNaN(mat3ToQuat(Rinf))).toBe(true);
    expect(allNaN(quatNormalize([NaN, 0, 0, 0]))).toBe(true);
    expect(allNaN(quatNormalize([Infinity, 0, 0, 0]))).toBe(true);
    expect(allNaN(axisAngleToMat3([NaN, 0, 1], 0.3))).toBe(true);
    expect(allNaN(rotationBetweenVectors([NaN, 0, 0], [1, 0, 0]))).toBe(true);
    expect(allNaN(rotationBetweenVectors([1, 0, 0], [0, Infinity, 0]))).toBe(true);
    expect(allNaN(vec3Normalize([NaN, 1, 0]))).toBe(true);
    expect(allNaN(rotvecToMat3([NaN, 0, 0]))).toBe(true);
    // Documented zero-input behaviour is unchanged.
    expect(Array.from(quatNormalize([0, 0, 0, 0]))).toEqual([1, 0, 0, 0]);
    expect(Array.from(vec3Normalize([0, 0, 0]))).toEqual([0, 0, 0]);
    expect(Array.from(axisAngleToMat3([0, 0, 0], 1))).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(Array.from(rotationBetweenVectors([0, 0, 0], [1, 0, 0]))).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  it('normalisation is over/underflow safe', () => {
    const h = Math.SQRT1_2;
    expect(maxAbsDiff(quatNormalize([1e200, 1e200, 0, 0]), [h, h, 0, 0])).toBeLessThan(1e-15);
    expect(maxAbsDiff(quatNormalize([1e-200, 0, 0, -1e-200]), [h, 0, 0, -h])).toBeLessThan(1e-15);
    const Rx = axisAngleToMat3([1, 0, 0], 0.7);
    expect(maxAbsDiff(axisAngleToMat3([1e-200, 0, 0], 0.7), Rx)).toBeLessThan(1e-15);
    expect(maxAbsDiff(axisAngleToMat3([1e200, 0, 0], 0.7), Rx)).toBeLessThan(1e-15);
    const Rz90 = [0, -1, 0, 1, 0, 0, 0, 0, 1];
    expect(maxAbsDiff(rotationBetweenVectors([1e200, 0, 0], [0, 1e-200, 0]), Rz90)).toBeLessThan(1e-15);
    expect(maxAbsDiff(rotationBetweenVectors([1e-200, 0, 0], [0, 1e200, 0]), Rz90)).toBeLessThan(1e-15);
    // Subnormal-range norm: 1/n overflowed to Infinity.
    expect(Array.from(vec3Normalize([1e-310, 0, 0]))).toEqual([1, 0, 0]);
    expect(maxAbsDiff(vec3Normalize([3e-320, 4e-320, 0]), [0.6, 0.8, 0])).toBeLessThan(1e-3); // subnormal inputs carry few bits
  });

  it('quaternion ↔ matrix: Hamilton handedness and Shepperd branches agree (incl. w ≈ 0)', () => {
    const rng = new Rng(41);
    for (let t = 0; t < 4000; t++) {
      // Force each Shepperd branch: one dominant component (w, x, y or z), and w ≈ 0.
      const q = new Float64Array([rng.gaussian(), rng.gaussian(), rng.gaussian(), rng.gaussian()]);
      q[t % 4] *= 10;
      if (t % 5 === 0) q[0] = rng.uniform(-1e-9, 1e-9);
      quatNormalize(q, q);
      const R = quatToMat3(q);
      const q2 = mat3ToQuat(R);
      expect(q2[0]).toBeGreaterThanOrEqual(0);
      const sgn = q[0] < 0 || (q[0] === 0 && maxAbsDiff(q, q2) > 1) ? -1 : 1;
      const err = Math.max(...Array.from(q).map((v, i) => Math.abs(sgn * v - q2[i])));
      // When w ≈ 0, q and −q are both valid; accept either sign.
      const errFlip = Math.max(...Array.from(q).map((v, i) => Math.abs(-sgn * v - q2[i])));
      expect(Math.min(err, errFlip)).toBeLessThan(1e-14);
      expect(maxAbsDiff(quatToMat3(q2), R)).toBeLessThan(1e-14);
      // Same rotation via the exp map: q = [cos θ/2, sin θ/2 n]  ⇔  R = exp(θ [n]×).
      const w = mat3ToRotvec(R);
      expect(maxAbsDiff(rotvecToMat3(w), R)).toBeLessThan(2e-14);
    }
  });

  it('log map near π: angle and axis sign consistent, continuous across the 2π/3 branch switch', () => {
    const rng = new Rng(42);
    for (let t = 0; t < 500; t++) {
      const n = new Float64Array([rng.gaussian(), rng.gaussian(), rng.gaussian()]);
      const nn = Math.hypot(n[0], n[1], n[2]);
      for (let i = 0; i < 3; i++) n[i] /= nn;
      for (const theta of [
        (2 * Math.PI) / 3 - 1e-12,
        (2 * Math.PI) / 3 + 1e-12,
        Math.PI - 1e-3,
        Math.PI - 1e-8,
        Math.PI - 1e-13,
      ]) {
        const w = new Float64Array([n[0] * theta, n[1] * theta, n[2] * theta]);
        const R = rotvecToMat3(w);
        const w2 = mat3ToRotvec(R);
        expect(Math.abs(rotationAngle(R) - theta)).toBeLessThan(1e-13);
        if (Math.PI - theta > 1e-7) {
          // Axis sign is determined: w2 must equal w (not −w).
          expect(maxAbsDiff(w2, w), `θ=${theta}`).toBeLessThan(1e-11);
        } else {
          // At θ ≈ π, ±n are equivalent up to O(π − θ); exp(log R) = R must hold.
          expect(maxAbsDiff(rotvecToMat3(w2), R)).toBeLessThan(1e-13);
        }
      }
    }
  });
});

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

describe('stats (adversarial)', () => {
  it('NaN in the data or in p → NaN (was: arbitrary order statistic)', () => {
    expect(median([1, NaN, 3])).toBeNaN();
    expect(median([NaN, 1, 2, 3])).toBeNaN();
    expect(percentile([1, 2, NaN, 4], 50)).toBeNaN();
    expect(percentile([1, 2, 3], NaN)).toBeNaN();
    expect(mad([1, 2, NaN, 4, 5])).toBeNaN();
    // ±Infinity are ordinary ordered values.
    expect(median([-Infinity, 0, Infinity])).toBe(0);
    expect(percentile([-Infinity, 1, 2], 100)).toBe(2);
  });

  it('quickselect: out-of-range / non-integer k → NaN (was undefined)', () => {
    expect(quickselect([3, 1, 2], 3)).toBeNaN();
    expect(quickselect([3, 1, 2], -1)).toBeNaN();
    expect(quickselect([3, 1, 2], 0.5)).toBeNaN();
    expect(quickselect([], 0)).toBeNaN();
    expect(quickselect([3, 1, 2], 2)).toBe(3);
  });

  it('percentile = NumPy "linear" on hand-computed cases', () => {
    const v = [10, 7, 4, 3, 2, 1]; // sorted: 1 2 3 4 7 10, rank = p/100·5
    expect(percentile(v, 0)).toBe(1);
    expect(percentile(v, 30)).toBeCloseTo(2.5, 14); // rank 1.5
    expect(percentile(v, 75)).toBeCloseTo(6.25, 14); // rank 3.75 → 4 + 0.75·3
    expect(percentile(v, 90)).toBeCloseTo(8.5, 14); // rank 4.5
    expect(percentile(v, 100)).toBe(10);
    expect(percentile(v, -5)).toBe(1); // clamped
    expect(percentile(v, 150)).toBe(10);
    expect(percentile([5], 37)).toBe(5);
    expect(percentile([2, 2, 2, 9], 50)).toBe(2);
  });

  it('quickselect-based median is correct and fast on adversarial orderings', () => {
    const n = 100001;
    const kinds: ((i: number) => number)[] = [
      () => 1,
      (i) => i,
      (i) => n - i,
      (i) => Math.min(i, n - i),
      (i) => i % 2,
      (i) => (i * 7919) % 13,
    ];
    const t0 = performance.now();
    for (const f of kinds) {
      const a = Float64Array.from({ length: n }, (_, i) => f(i));
      const ref = Float64Array.from(a).sort()[n >> 1];
      expect(median(a)).toBe(ref);
    }
    expect(performance.now() - t0).toBeLessThan(2000);
  });
});

// ---------------------------------------------------------------------------
// Rng
// ---------------------------------------------------------------------------

describe('Rng (adversarial)', () => {
  it('negative seeds do not collide with small positive ones (−1−k used to equal k)', () => {
    for (let k = 0; k < 50; k++) expect(new Rng(-1 - k).nextU32()).not.toBe(new Rng(k).nextU32());
    const seen = new Set<number>();
    for (let s = -500; s <= 500; s++) seen.add(new Rng(s).nextU32());
    expect(seen.size).toBe(1001);
    // Seeds in [0, 2³²) are unchanged by the fix (values pinned from the pre-fix code).
    const pinned: [number, number, number][] = [
      [0, 4205396811, 3543653536],
      [7, 838155491, 1550620431],
      [12345, 2345461488, 1344865159],
      [4294967295, 1413747962, 91561161],
    ];
    for (const [seed, a, b] of pinned) {
      const r = new Rng(seed);
      expect([r.nextU32(), r.nextU32()]).toEqual([a, b]);
    }
  });

  it('sample order is uniform over ORDERED k-samples (Floyd path)', () => {
    const rng = new Rng(51);
    const counts = new Map<string, number>();
    const N = 60000;
    for (let t = 0; t < N; t++) {
      const s = rng.sample(3, 2);
      const key = `${s[0]},${s[1]}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    expect(counts.size).toBe(6);
    for (const c of counts.values()) expect(Math.abs(c - N / 6)).toBeLessThan(500); // σ ≈ 91
    // Position × value table for k = 4 of n = 10: each cell ≈ N/10.
    const M = 40000;
    const tab = new Float64Array(40);
    for (let t = 0; t < M; t++) {
      const s = rng.sample(10, 4);
      for (let p = 0; p < 4; p++) tab[p * 10 + s[p]]++;
    }
    for (let i = 0; i < 40; i++) expect(Math.abs(tab[i] - M / 10)).toBeLessThan(400); // σ = 60
  });

  it('sample stays distinct with a reused, stale out buffer; sample(n, n) is a permutation', () => {
    const rng = new Rng(52);
    const out = new Uint32Array(12).fill(3);
    for (let t = 0; t < 2000; t++) {
      for (const [n, k] of [
        [100, 8],
        [9, 8],
        [20, 12],
        [1000, 12],
      ]) {
        const s = rng.sample(n, k, out);
        expect(s).toBe(out);
        expect(new Set(Array.from(s.subarray(0, k))).size).toBe(k);
        for (let i = 0; i < k; i++) expect(s[i]).toBeLessThan(n);
      }
    }
    for (const n of [1, 5, 8, 30]) {
      const s = Array.from(rng.sample(n, n)).sort((a, b) => a - b);
      expect(s).toEqual(Array.from({ length: n }, (_, i) => i));
    }
  });
});

// ---------------------------------------------------------------------------
// Sim3
// ---------------------------------------------------------------------------

describe('umeyama (adversarial)', () => {
  function transform(src: Float64Array, s: number, R: Float64Array, t: number[]): Float64Array {
    return applySim3({ s, R, t: new Float64Array(t) }, src);
  }

  it('coplanar points (rank-2 covariance) with and without reflection', () => {
    const rng = new Rng(61);
    const n = 12;
    const src = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
      src[3 * i] = rng.gaussian();
      src[3 * i + 1] = rng.gaussian();
    }
    const R = rotvecToMat3([0.3, -2.2, 0.9]);
    const dst = transform(src, 2.5, R, [1, -2, 3]);
    const est = umeyama(src, dst, n, true);
    expect(est).not.toBeNull();
    expect(Math.abs(est!.s - 2.5)).toBeLessThan(1e-12);
    expect(maxAbsDiff(est!.R, R)).toBeLessThan(1e-12);
    // Mirror the target through a plane: still a proper rotation is returned.
    const mir = Float64Array.from(dst);
    for (let i = 0; i < n; i++) mir[3 * i + 2] = -mir[3 * i + 2];
    const e2 = umeyama(src, mir, n, true);
    expect(e2).not.toBeNull();
    expect(Math.abs(mat3Det(e2!.R) - 1)).toBeLessThan(1e-12);
    // Planar data: a mirror of a plane IS a rotation of it, so the fit is exact.
    const fit = applySim3(e2!, src);
    expect(maxAbsDiff(fit, mir)).toBeLessThan(1e-11);
  });

  it('near-collinear rejection and minimal 3-point case', () => {
    const rng = new Rng(62);
    const n = 20;
    const line = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
      const u = rng.uniform(-5, 5);
      line[3 * i] = u + rng.uniform(-1e-9, 1e-9);
      line[3 * i + 1] = 2 * u + rng.uniform(-1e-9, 1e-9);
      line[3 * i + 2] = -u + rng.uniform(-1e-9, 1e-9);
    }
    const R = rotvecToMat3([0.1, 0.2, 0.3]);
    expect(umeyama(line, transform(line, 1.5, R, [0, 0, 0]), n, true)).toBeNull();
    const tri = new Float64Array([0, 0, 0, 1, 0, 0, 0, 2, 0]);
    const e = umeyama(tri, transform(tri, 0.5, R, [10, 20, 30]), 3, true);
    expect(e).not.toBeNull();
    expect(Math.abs(e!.s - 0.5)).toBeLessThan(1e-14);
    expect(maxAbsDiff(e!.R, R)).toBeLessThan(1e-14);
  });

  it('negative / NaN weights count as zero; 90 % gross outliers with zero weight are ignored', () => {
    const rng = new Rng(63);
    const n = 100;
    const src = new Float64Array(3 * n);
    for (let i = 0; i < 3 * n; i++) src[i] = rng.gaussian();
    const R = rotvecToMat3([-1.0, 0.4, 2.0]);
    const dst = transform(src, 3, R, [5, 6, 7]);
    const w = new Float64Array(n).fill(1);
    for (let i = 0; i < n; i++) {
      if (i % 10 !== 0) {
        dst[3 * i] += rng.uniform(-100, 100);
        w[i] = i % 3 === 0 ? 0 : i % 3 === 1 ? -1 : NaN;
      }
    }
    const e = umeyama(src, dst, n, true, w);
    expect(e).not.toBeNull();
    expect(Math.abs(e!.s - 3)).toBeLessThan(1e-12);
    expect(maxAbsDiff(e!.R, R)).toBeLessThan(1e-12);
    expect(maxAbsDiff(e!.t, [5, 6, 7])).toBeLessThan(1e-11);
  });
});
