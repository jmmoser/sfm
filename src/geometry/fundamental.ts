/**
 * Fundamental-matrix estimation (pixel coordinates, flat [x0, y0, x1, y1, …]):
 * x2ᵀ F x1 = 0. Hartley normalisation (Hartley 1997, "In defense of the eight-point
 * algorithm", IEEE TPAMI 19(6)); 7-point minimal solver (Hartley & Zisserman §11.1.2).
 */
import type { CameraIntrinsics, Mat3 } from '../core/types';
import { Matrix } from '../math/matrix';
import { nullSpace, svd3 } from '../math/decomp';
import { polyRealRoots } from '../math/poly';
import { denormalizeHartley, hartleyNormalize, normalizeFrobenius, smallestEigenvector } from './internal';

const Q7 = new Matrix(7, 9);
const P1 = new Float64Array(14);
const P2 = new Float64Array(14);
const T1 = new Float64Array(9);
const T2 = new Float64Array(9);

function det3cols(
  a0: number, a1: number, a2: number,
  b0: number, b1: number, b2: number,
  c0: number, c1: number, c2: number,
): number {
  // det of the matrix with columns a, b, c = a · (b × c)
  return a0 * (b1 * c2 - b2 * c1) + a1 * (b2 * c0 - b0 * c2) + a2 * (b0 * c1 - b1 * c0);
}

/**
 * Minimal 7-point solver on the first 7 correspondences: F = A + α (B − A) over the
 * 2-D null space with det F = 0 (a cubic in α). Returns 1 or 3 solutions, rank 2,
 * unit Frobenius norm.
 */
export function fundamental7pt(p1: ArrayLike<number>, p2: ArrayLike<number>): Mat3[] {
  const out: Mat3[] = [];
  if (!hartleyNormalize(p1, 7, null, P1, T1) || !hartleyNormalize(p2, 7, null, P2, T2)) return out;
  const q = Q7.data;
  for (let i = 0; i < 7; i++) {
    const a = P1[2 * i],
      b = P1[2 * i + 1],
      c = P2[2 * i],
      d = P2[2 * i + 1];
    const o = 9 * i;
    q[o] = c * a;
    q[o + 1] = c * b;
    q[o + 2] = c;
    q[o + 3] = d * a;
    q[o + 4] = d * b;
    q[o + 5] = d;
    q[o + 6] = a;
    q[o + 7] = b;
    q[o + 8] = 1;
  }
  const N = nullSpace(Q7, 2).data; // 9×2
  // A = column 1, B = column 0 − column 1; F(α) = A + α B (column-wise determinant expansion).
  const A = new Float64Array(9),
    B = new Float64Array(9);
  for (let k = 0; k < 9; k++) {
    A[k] = N[2 * k + 1];
    B[k] = N[2 * k] - N[2 * k + 1];
  }
  // columns of A: (A0,A3,A6), (A1,A4,A7), (A2,A5,A8)
  const c0 = det3cols(A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]);
  const c3 = det3cols(B[0], B[3], B[6], B[1], B[4], B[7], B[2], B[5], B[8]);
  const c1 =
    det3cols(B[0], B[3], B[6], A[1], A[4], A[7], A[2], A[5], A[8]) +
    det3cols(A[0], A[3], A[6], B[1], B[4], B[7], A[2], A[5], A[8]) +
    det3cols(A[0], A[3], A[6], A[1], A[4], A[7], B[2], B[5], B[8]);
  const c2 =
    det3cols(A[0], A[3], A[6], B[1], B[4], B[7], B[2], B[5], B[8]) +
    det3cols(B[0], B[3], B[6], A[1], A[4], A[7], B[2], B[5], B[8]) +
    det3cols(B[0], B[3], B[6], B[1], B[4], B[7], A[2], A[5], A[8]);
  const roots = polyRealRoots([c3, c2, c1, c0]);
  const Fn = new Float64Array(9);
  for (const alpha of roots) {
    for (let k = 0; k < 9; k++) Fn[k] = A[k] + alpha * B[k];
    const F = new Float64Array(9);
    denormalizeHartley(Fn, T1, T2, F);
    if (normalizeFrobenius(F)) out.push(F);
  }
  // Degree drop (c3 ≈ 0): α → ∞, i.e. F = B is a solution.
  const cmax = Math.max(Math.abs(c0), Math.abs(c1), Math.abs(c2), Math.abs(c3));
  if (Math.abs(c3) <= 1e-12 * cmax) {
    const F = new Float64Array(9);
    denormalizeHartley(B, T1, T2, F);
    if (normalizeFrobenius(F)) out.push(F);
  }
  return out;
}

let N1 = new Float64Array(64);
let N2 = new Float64Array(64);
const ATA = new Float64Array(81);
const ROW = new Float64Array(9);

/**
 * Normalised 8-point fundamental matrix on correspondences idx[0..n) (all n when idx
 * is null), rank 2 enforced by zeroing the smallest singular value, unit Frobenius
 * norm. Null for n < 8 or degenerate data.
 */
export function fundamental8ptIdx(
  p1: ArrayLike<number>,
  p2: ArrayLike<number>,
  idx: Uint32Array | null,
  n: number,
): Mat3 | null {
  if (n < 8) return null;
  if (N1.length < 2 * n) {
    N1 = new Float64Array(4 * n);
    N2 = new Float64Array(4 * n);
  }
  if (!hartleyNormalize(p1, n, idx, N1, T1) || !hartleyNormalize(p2, n, idx, N2, T2)) return null;
  ATA.fill(0);
  const r = ROW;
  for (let i = 0; i < n; i++) {
    const a = N1[2 * i],
      b = N1[2 * i + 1],
      c = N2[2 * i],
      d = N2[2 * i + 1];
    r[0] = c * a;
    r[1] = c * b;
    r[2] = c;
    r[3] = d * a;
    r[4] = d * b;
    r[5] = d;
    r[6] = a;
    r[7] = b;
    r[8] = 1;
    for (let u = 0; u < 9; u++) {
      const ru = r[u];
      const o = 9 * u;
      for (let v = u; v < 9; v++) ATA[o + v] += ru * r[v];
    }
  }
  const Fn = new Float64Array(9);
  smallestEigenvector(ATA, 9, Fn);
  if (!Number.isFinite(Fn[0])) return null;
  const Fr = enforceRank2(Fn);
  if (Fr === null) return null;
  const F = new Float64Array(9);
  denormalizeHartley(Fr, T1, T2, F);
  return normalizeFrobenius(F) ? F : null;
}

/** Normalised 8-point fundamental matrix from n ≥ 8 correspondences (pixels). */
export function fundamental8pt(p1: ArrayLike<number>, p2: ArrayLike<number>, n: number): Mat3 | null {
  return fundamental8ptIdx(p1, p2, null, n);
}

/** Closest rank-2 matrix (Frobenius): U diag(σ1, σ2, 0) Vᵀ. */
export function enforceRank2(F: ArrayLike<number>): Mat3 | null {
  const { U, S, V } = svd3(Float64Array.from(F as ArrayLike<number>));
  if (!Number.isFinite(S[0]) || !(S[0] > 0)) return null;
  const out = new Float64Array(9);
  const s0 = S[0],
    s1 = S[1];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) out[3 * i + j] = s0 * U[3 * i] * V[3 * j] + s1 * U[3 * i + 1] * V[3 * j + 1];
  return out;
}

/**
 * Fundamental matrix of two cameras' undistorted pixels from an essential matrix:
 * F = K2⁻ᵀ E K1⁻¹ (unit Frobenius norm).
 */
export function fundamentalFromEssential(E: ArrayLike<number>, cam1: CameraIntrinsics, cam2: CameraIntrinsics): Mat3 {
  // K⁻¹ = [[1/f, 0, −cx/f], [0, 1/f, −cy/f], [0, 0, 1]] — the same structure as a Hartley transform.
  const T1h = Float64Array.of(1 / cam1.f, 0, -cam1.cx / cam1.f, 0, 1 / cam1.f, -cam1.cy / cam1.f, 0, 0, 1);
  const T2h = Float64Array.of(1 / cam2.f, 0, -cam2.cx / cam2.f, 0, 1 / cam2.f, -cam2.cy / cam2.f, 0, 0, 1);
  const F = new Float64Array(9);
  denormalizeHartley(Float64Array.from(E as ArrayLike<number>), T1h, T2h, F);
  normalizeFrobenius(F);
  return F;
}
