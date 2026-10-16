/**
 * Minimal row-major 3x3 / vec3 helpers local to the synthetic generator (kept
 * independent of src/math on purpose: the generator is the ground truth).
 */
import type { Mat3, Vec3 } from '../../core/types';

export function m3(...v: number[]): Mat3 {
  return Float64Array.from(v);
}

export function m3Identity(): Mat3 {
  return Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
}

export function m3Mul(a: ArrayLike<number>, b: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  const a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7], a8 = a[8];
  const b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7], b8 = b[8];
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

export function m3T(a: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  const a1 = a[1], a2 = a[2], a5 = a[5];
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

/** aᵀ b */
export function m3TMul(a: ArrayLike<number>, b: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  return m3Mul(m3T(a), b, out);
}

export function m3Add(a: ArrayLike<number>, b: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  for (let i = 0; i < 9; i++) out[i] = a[i] + b[i];
  return out;
}

export function m3Vec(a: ArrayLike<number>, v: ArrayLike<number>, out: Vec3 = new Float64Array(3)): Vec3 {
  const x = v[0], y = v[1], z = v[2];
  out[0] = a[0] * x + a[1] * y + a[2] * z;
  out[1] = a[3] * x + a[4] * y + a[5] * z;
  out[2] = a[6] * x + a[7] * y + a[8] * z;
  return out;
}

/** aᵀ v */
export function m3TVec(a: ArrayLike<number>, v: ArrayLike<number>, out: Vec3 = new Float64Array(3)): Vec3 {
  const x = v[0], y = v[1], z = v[2];
  out[0] = a[0] * x + a[3] * y + a[6] * z;
  out[1] = a[1] * x + a[4] * y + a[7] * z;
  out[2] = a[2] * x + a[5] * y + a[8] * z;
  return out;
}

export function skew(v: ArrayLike<number>): Mat3 {
  return Float64Array.of(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0);
}

/** Axial vector of the antisymmetric part of m. */
export function vee(m: ArrayLike<number>): Vec3 {
  return Float64Array.of(0.5 * (m[7] - m[5]), 0.5 * (m[2] - m[6]), 0.5 * (m[3] - m[1]));
}

/** Rodrigues exp map. */
export function expSO3(w: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  const x = w[0], y = w[1], z = w[2];
  const th2 = x * x + y * y + z * z;
  const th = Math.sqrt(th2);
  let A: number, B: number;
  if (th < 1e-4) {
    A = 1 - th2 / 6 + (th2 * th2) / 120;
    B = 0.5 - th2 / 24 + (th2 * th2) / 720;
  } else {
    A = Math.sin(th) / th;
    B = (1 - Math.cos(th)) / th2;
  }
  out[0] = 1 - B * (y * y + z * z);
  out[1] = -A * z + B * x * y;
  out[2] = A * y + B * x * z;
  out[3] = A * z + B * x * y;
  out[4] = 1 - B * (x * x + z * z);
  out[5] = -A * x + B * y * z;
  out[6] = -A * y + B * x * z;
  out[7] = A * x + B * y * z;
  out[8] = 1 - B * (x * x + y * y);
  return out;
}

/** Log map (rotation vector), adequate away from π. */
export function logSO3(R: ArrayLike<number>): Vec3 {
  const c = Math.min(1, Math.max(-1, (R[0] + R[4] + R[8] - 1) / 2));
  const th = Math.acos(c);
  const v = Float64Array.of(R[7] - R[5], R[2] - R[6], R[3] - R[1]);
  const s = th < 1e-6 ? 0.5 + (th * th) / 12 : th / (2 * Math.sin(th));
  v[0] *= s;
  v[1] *= s;
  v[2] *= s;
  return v;
}

/** Geodesic angle between two rotations (radians). */
export function rotationAngleBetween(a: ArrayLike<number>, b: ArrayLike<number>): number {
  // trace(aᵀ b)
  let tr = 0;
  for (let i = 0; i < 9; i++) tr += a[i] * b[i];
  const c = Math.min(1, Math.max(-1, (tr - 1) / 2));
  // acos is ill-conditioned near 0; use the antisymmetric part for small angles.
  const M = m3TMul(a, b);
  const s = 0.5 * Math.hypot(M[7] - M[5], M[2] - M[6], M[3] - M[1]);
  return Math.atan2(s, c);
}

export function rotZ(theta: number): Mat3 {
  const c = Math.cos(theta),
    s = Math.sin(theta);
  return Float64Array.of(c, -s, 0, s, c, 0, 0, 0, 1);
}

export function v3(x = 0, y = 0, z = 0): Vec3 {
  return Float64Array.of(x, y, z);
}

export function v3Norm(a: ArrayLike<number>): number {
  return Math.hypot(a[0], a[1], a[2]);
}

export function v3Sub(a: ArrayLike<number>, b: ArrayLike<number>): Vec3 {
  return Float64Array.of(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

export function orthonormalityError(R: ArrayLike<number>): number {
  const M = m3TMul(R, R);
  let e = 0;
  for (let i = 0; i < 9; i++) e = Math.max(e, Math.abs(M[i] - (i % 4 === 0 ? 1 : 0)));
  return e;
}
