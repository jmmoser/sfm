/**
 * 3x3 matrix helpers. Matrices are ROW-MAJOR `Float64Array(9)`: element (r, c) is
 * `m[3*r + c]`. Every function taking `out` writes into it (when given) and returns
 * it; `out` may alias any input.
 */
import type { Mat3, Vec3 } from '../core/types';

/** Identity matrix. */
export function mat3Identity(out: Mat3 = new Float64Array(9)): Mat3 {
  out[0] = 1;
  out[1] = 0;
  out[2] = 0;
  out[3] = 0;
  out[4] = 1;
  out[5] = 0;
  out[6] = 0;
  out[7] = 0;
  out[8] = 1;
  return out;
}

/** Copy of a. */
export function mat3Copy(a: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  for (let i = 0; i < 9; i++) out[i] = a[i];
  return out;
}

/** Build from 9 row-major values. */
export function mat3FromValues(
  m00: number,
  m01: number,
  m02: number,
  m10: number,
  m11: number,
  m12: number,
  m20: number,
  m21: number,
  m22: number,
  out: Mat3 = new Float64Array(9),
): Mat3 {
  out[0] = m00;
  out[1] = m01;
  out[2] = m02;
  out[3] = m10;
  out[4] = m11;
  out[5] = m12;
  out[6] = m20;
  out[7] = m21;
  out[8] = m22;
  return out;
}

/** out = a b */
export function mat3Mul(a: ArrayLike<number>, b: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  const a0 = a[0],
    a1 = a[1],
    a2 = a[2],
    a3 = a[3],
    a4 = a[4],
    a5 = a[5],
    a6 = a[6],
    a7 = a[7],
    a8 = a[8];
  const b0 = b[0],
    b1 = b[1],
    b2 = b[2],
    b3 = b[3],
    b4 = b[4],
    b5 = b[5],
    b6 = b[6],
    b7 = b[7],
    b8 = b[8];
  out[0] = a0 * b0 + a1 * b3 + a2 * b6;
  out[1] = a0 * b1 + a1 * b4 + a2 * b7;
  out[2] = a0 * b2 + a1 * b5 + a2 * b8;
  out[3] = a3 * b0 + a4 * b3 + a5 * b6;
  out[4] = a3 * b1 + a4 * b4 + a5 * b7;
  out[5] = a3 * b2 + a4 * b5 + a5 * b8;
  out[6] = a6 * b0 + a7 * b3 + a8 * b6;
  out[7] = a6 * b1 + a7 * b4 + a8 * b7;
  out[8] = a6 * b2 + a7 * b5 + a8 * b8;
  return out;
}

/** out = a bᵀ */
export function mat3MulTranspose(a: ArrayLike<number>, b: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  const a0 = a[0],
    a1 = a[1],
    a2 = a[2],
    a3 = a[3],
    a4 = a[4],
    a5 = a[5],
    a6 = a[6],
    a7 = a[7],
    a8 = a[8];
  const b0 = b[0],
    b1 = b[1],
    b2 = b[2],
    b3 = b[3],
    b4 = b[4],
    b5 = b[5],
    b6 = b[6],
    b7 = b[7],
    b8 = b[8];
  out[0] = a0 * b0 + a1 * b1 + a2 * b2;
  out[1] = a0 * b3 + a1 * b4 + a2 * b5;
  out[2] = a0 * b6 + a1 * b7 + a2 * b8;
  out[3] = a3 * b0 + a4 * b1 + a5 * b2;
  out[4] = a3 * b3 + a4 * b4 + a5 * b5;
  out[5] = a3 * b6 + a4 * b7 + a5 * b8;
  out[6] = a6 * b0 + a7 * b1 + a8 * b2;
  out[7] = a6 * b3 + a7 * b4 + a8 * b5;
  out[8] = a6 * b6 + a7 * b7 + a8 * b8;
  return out;
}

/** out = aᵀ b */
export function mat3TransposeMul(a: ArrayLike<number>, b: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  const a0 = a[0],
    a1 = a[1],
    a2 = a[2],
    a3 = a[3],
    a4 = a[4],
    a5 = a[5],
    a6 = a[6],
    a7 = a[7],
    a8 = a[8];
  const b0 = b[0],
    b1 = b[1],
    b2 = b[2],
    b3 = b[3],
    b4 = b[4],
    b5 = b[5],
    b6 = b[6],
    b7 = b[7],
    b8 = b[8];
  out[0] = a0 * b0 + a3 * b3 + a6 * b6;
  out[1] = a0 * b1 + a3 * b4 + a6 * b7;
  out[2] = a0 * b2 + a3 * b5 + a6 * b8;
  out[3] = a1 * b0 + a4 * b3 + a7 * b6;
  out[4] = a1 * b1 + a4 * b4 + a7 * b7;
  out[5] = a1 * b2 + a4 * b5 + a7 * b8;
  out[6] = a2 * b0 + a5 * b3 + a8 * b6;
  out[7] = a2 * b1 + a5 * b4 + a8 * b7;
  out[8] = a2 * b2 + a5 * b5 + a8 * b8;
  return out;
}

/** out = a v */
export function mat3MulVec(a: ArrayLike<number>, v: ArrayLike<number>, out: Vec3 = new Float64Array(3)): Vec3 {
  const x = v[0],
    y = v[1],
    z = v[2];
  out[0] = a[0] * x + a[1] * y + a[2] * z;
  out[1] = a[3] * x + a[4] * y + a[5] * z;
  out[2] = a[6] * x + a[7] * y + a[8] * z;
  return out;
}

/** out = aᵀ v */
export function mat3TransposeMulVec(
  a: ArrayLike<number>,
  v: ArrayLike<number>,
  out: Vec3 = new Float64Array(3),
): Vec3 {
  const x = v[0],
    y = v[1],
    z = v[2];
  out[0] = a[0] * x + a[3] * y + a[6] * z;
  out[1] = a[1] * x + a[4] * y + a[7] * z;
  out[2] = a[2] * x + a[5] * y + a[8] * z;
  return out;
}

/** out = aᵀ */
export function mat3Transpose(a: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  const a1 = a[1],
    a2 = a[2],
    a5 = a[5];
  out[0] = a[0];
  out[4] = a[4];
  out[8] = a[8];
  out[1] = a[3];
  out[3] = a1;
  out[2] = a[6];
  out[6] = a2;
  out[5] = a[7];
  out[7] = a5;
  return out;
}

/** Determinant. */
export function mat3Det(a: ArrayLike<number>): number {
  return (
    a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6])
  );
}

/** Trace. */
export function mat3Trace(a: ArrayLike<number>): number {
  return a[0] + a[4] + a[8];
}

/**
 * Inverse via the adjugate (computed on the matrix scaled to max-abs 1, so no
 * over/underflow). Returns null when the matrix is singular to working precision:
 * |det| ≤ 1e-14 · ‖row0‖‖row1‖‖row2‖ (invariant to row scaling; by Hadamard's
 * inequality the ratio is 1 for orthogonal rows), or for non-finite input.
 */
export function mat3Inv(a: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 | null {
  let sc = 0;
  for (let i = 0; i < 9; i++) {
    const v = Math.abs(a[i]);
    if (v > sc || v !== v) sc = v;
  }
  if (!(sc > 0) || !Number.isFinite(sc)) return null;
  const is = 1 / sc;
  const a0 = a[0] * is,
    a1 = a[1] * is,
    a2 = a[2] * is,
    a3 = a[3] * is,
    a4 = a[4] * is,
    a5 = a[5] * is,
    a6 = a[6] * is,
    a7 = a[7] * is,
    a8 = a[8] * is;
  const c0 = a4 * a8 - a5 * a7;
  const c1 = a5 * a6 - a3 * a8;
  const c2 = a3 * a7 - a4 * a6;
  const det = a0 * c0 + a1 * c1 + a2 * c2;
  const r0 = Math.sqrt(a0 * a0 + a1 * a1 + a2 * a2);
  const r1 = Math.sqrt(a3 * a3 + a4 * a4 + a5 * a5);
  const r2 = Math.sqrt(a6 * a6 + a7 * a7 + a8 * a8);
  if (!(Math.abs(det) > 1e-14 * r0 * r1 * r2) || !Number.isFinite(det)) return null;
  const inv = is / det; // A⁻¹ = (A/s)⁻¹ / s
  out[0] = c0 * inv;
  out[1] = (a2 * a7 - a1 * a8) * inv;
  out[2] = (a1 * a5 - a2 * a4) * inv;
  out[3] = c1 * inv;
  out[4] = (a0 * a8 - a2 * a6) * inv;
  out[5] = (a2 * a3 - a0 * a5) * inv;
  out[6] = c2 * inv;
  out[7] = (a1 * a6 - a0 * a7) * inv;
  out[8] = (a0 * a4 - a1 * a3) * inv;
  return out;
}

/** Cross-product (skew-symmetric) matrix [v]× such that [v]× w = v × w. */
export function mat3Skew(v: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  const x = v[0],
    y = v[1],
    z = v[2];
  out[0] = 0;
  out[1] = -z;
  out[2] = y;
  out[3] = z;
  out[4] = 0;
  out[5] = -x;
  out[6] = -y;
  out[7] = x;
  out[8] = 0;
  return out;
}

/** out = s a */
export function mat3Scale(a: ArrayLike<number>, s: number, out: Mat3 = new Float64Array(9)): Mat3 {
  for (let i = 0; i < 9; i++) out[i] = a[i] * s;
  return out;
}

/** out = a + b */
export function mat3Add(a: ArrayLike<number>, b: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  for (let i = 0; i < 9; i++) out[i] = a[i] + b[i];
  return out;
}

/** out = a - b */
export function mat3Sub(a: ArrayLike<number>, b: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  for (let i = 0; i < 9; i++) out[i] = a[i] - b[i];
  return out;
}

/** Frobenius norm. */
export function mat3FrobeniusNorm(a: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < 9; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}

/** Outer product out = a bᵀ. */
export function mat3Outer(a: ArrayLike<number>, b: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  const a0 = a[0],
    a1 = a[1],
    a2 = a[2],
    b0 = b[0],
    b1 = b[1],
    b2 = b[2];
  out[0] = a0 * b0;
  out[1] = a0 * b1;
  out[2] = a0 * b2;
  out[3] = a1 * b0;
  out[4] = a1 * b1;
  out[5] = a1 * b2;
  out[6] = a2 * b0;
  out[7] = a2 * b1;
  out[8] = a2 * b2;
  return out;
}

/** Build from three columns. */
export function mat3FromColumns(
  c0: ArrayLike<number>,
  c1: ArrayLike<number>,
  c2: ArrayLike<number>,
  out: Mat3 = new Float64Array(9),
): Mat3 {
  const x0 = c0[0],
    y0 = c0[1],
    z0 = c0[2],
    x1 = c1[0],
    y1 = c1[1],
    z1 = c1[2],
    x2 = c2[0],
    y2 = c2[1],
    z2 = c2[2];
  out[0] = x0;
  out[1] = x1;
  out[2] = x2;
  out[3] = y0;
  out[4] = y1;
  out[5] = y2;
  out[6] = z0;
  out[7] = z1;
  out[8] = z2;
  return out;
}

/** Column c (0..2) of a. */
export function mat3GetColumn(a: ArrayLike<number>, c: number, out: Vec3 = new Float64Array(3)): Vec3 {
  out[0] = a[c];
  out[1] = a[3 + c];
  out[2] = a[6 + c];
  return out;
}

/** Row r (0..2) of a. */
export function mat3GetRow(a: ArrayLike<number>, r: number, out: Vec3 = new Float64Array(3)): Vec3 {
  out[0] = a[3 * r];
  out[1] = a[3 * r + 1];
  out[2] = a[3 * r + 2];
  return out;
}
