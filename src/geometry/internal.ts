/**
 * Internal helpers shared by the geometry solvers (not re-exported from index.ts).
 */
import { symEig } from '../math/decomp';

/**
 * Hartley normalisation (Hartley 1997, "In defense of the eight-point algorithm"):
 * translate the centroid to the origin and scale so the mean distance is √2.
 * Reads points `pts[2*idx[i]]` (or `pts[2*i]` when idx is null) for i < n, writes the
 * normalised points to `out` (2n) and the 3×3 transform to `T` (x̂ = T x).
 * Returns false if all points coincide (or data is non-finite).
 */
export function hartleyNormalize(
  pts: ArrayLike<number>,
  n: number,
  idx: Uint32Array | null,
  out: Float64Array,
  T: Float64Array,
): boolean {
  let cx = 0,
    cy = 0;
  for (let i = 0; i < n; i++) {
    const k = idx ? idx[i] : i;
    cx += pts[2 * k];
    cy += pts[2 * k + 1];
  }
  cx /= n;
  cy /= n;
  let md = 0;
  for (let i = 0; i < n; i++) {
    const k = idx ? idx[i] : i;
    const dx = pts[2 * k] - cx,
      dy = pts[2 * k + 1] - cy;
    md += Math.sqrt(dx * dx + dy * dy);
  }
  md /= n;
  if (!(md > 0) || !Number.isFinite(md)) return false;
  const s = Math.SQRT2 / md;
  for (let i = 0; i < n; i++) {
    const k = idx ? idx[i] : i;
    out[2 * i] = (pts[2 * k] - cx) * s;
    out[2 * i + 1] = (pts[2 * k + 1] - cy) * s;
  }
  T[0] = s;
  T[1] = 0;
  T[2] = -s * cx;
  T[3] = 0;
  T[4] = s;
  T[5] = -s * cy;
  T[6] = 0;
  T[7] = 0;
  T[8] = 1;
  return true;
}

/**
 * Unit eigenvector of the smallest eigenvalue of the symmetric n×n matrix whose
 * UPPER triangle is stored in `A` (row-major; the lower triangle is filled here).
 */
export function smallestEigenvector(A: Float64Array, n: number, out: Float64Array): void {
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) A[i * n + j] = A[j * n + i];
  const { vectors } = symEig(A, n);
  for (let r = 0; r < n; r++) out[r] = vectors[r * n];
}

/** Determinant of a row-major 3×3 stored at offset o. */
export function det3(m: ArrayLike<number>, o = 0): number {
  return (
    m[o] * (m[o + 4] * m[o + 8] - m[o + 5] * m[o + 7]) -
    m[o + 1] * (m[o + 3] * m[o + 8] - m[o + 5] * m[o + 6]) +
    m[o + 2] * (m[o + 3] * m[o + 7] - m[o + 4] * m[o + 6])
  );
}

/** Scales a 3×3 to unit Frobenius norm in place; returns false if it is zero / non-finite. */
export function normalizeFrobenius(M: Float64Array): boolean {
  let s = 0;
  for (let i = 0; i < 9; i++) s += M[i] * M[i];
  if (!(s > 0) || !Number.isFinite(s)) return false;
  const inv = 1 / Math.sqrt(s);
  for (let i = 0; i < 9; i++) M[i] *= inv;
  return true;
}

/**
 * out = T2ᵀ M T1 for Hartley transforms T = [[s,0,a],[0,s,b],[0,0,1]] (de-normalisation
 * of F / E estimated on normalised points). out may alias M.
 */
export function denormalizeHartley(M: Float64Array, T1: Float64Array, T2: Float64Array, out: Float64Array): void {
  const s1 = T1[0],
    a1 = T1[2],
    b1 = T1[5];
  const s2 = T2[0],
    a2 = T2[2],
    b2 = T2[5];
  // m = M T1
  const m0 = M[0] * s1,
    m1 = M[1] * s1,
    m2 = M[0] * a1 + M[1] * b1 + M[2];
  const m3 = M[3] * s1,
    m4 = M[4] * s1,
    m5 = M[3] * a1 + M[4] * b1 + M[5];
  const m6 = M[6] * s1,
    m7 = M[7] * s1,
    m8 = M[6] * a1 + M[7] * b1 + M[8];
  // T2ᵀ m: rows s2·m_row0, s2·m_row1, a2·m_row0 + b2·m_row1 + m_row2
  out[0] = s2 * m0;
  out[1] = s2 * m1;
  out[2] = s2 * m2;
  out[3] = s2 * m3;
  out[4] = s2 * m4;
  out[5] = s2 * m5;
  out[6] = a2 * m0 + b2 * m3 + m6;
  out[7] = a2 * m1 + b2 * m4 + m7;
  out[8] = a2 * m2 + b2 * m5 + m8;
}

/** Collinearity test of three 2-D points (twice the triangle area vs. the squared side lengths). */
export function nearlyCollinear(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cx: number,
  cy: number,
  relTol: number,
): boolean {
  const ux = bx - ax,
    uy = by - ay,
    vx = cx - ax,
    vy = cy - ay;
  const cross = Math.abs(ux * vy - uy * vx);
  const l = Math.max(ux * ux + uy * uy, vx * vx + vy * vy, (cx - bx) * (cx - bx) + (cy - by) * (cy - by));
  return !(cross > relTol * l);
}
