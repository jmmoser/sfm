/**
 * Small fixed-size 3-vector helpers. Inputs are any `ArrayLike<number>`; outputs
 * are `Float64Array`. Every function taking `out` writes into it (when given) and
 * returns it; `out` may alias an input.
 */
import type { Vec3 } from '../core/types';

/** New 3-vector (defaults to zero). */
export function vec3(x = 0, y = 0, z = 0): Vec3 {
  const v = new Float64Array(3);
  v[0] = x;
  v[1] = y;
  v[2] = z;
  return v;
}

/** out = a + b */
export function vec3Add(a: ArrayLike<number>, b: ArrayLike<number>, out: Vec3 = new Float64Array(3)): Vec3 {
  out[0] = a[0] + b[0];
  out[1] = a[1] + b[1];
  out[2] = a[2] + b[2];
  return out;
}

/** out = a - b */
export function vec3Sub(a: ArrayLike<number>, b: ArrayLike<number>, out: Vec3 = new Float64Array(3)): Vec3 {
  out[0] = a[0] - b[0];
  out[1] = a[1] - b[1];
  out[2] = a[2] - b[2];
  return out;
}

/** out = s * a */
export function vec3Scale(a: ArrayLike<number>, s: number, out: Vec3 = new Float64Array(3)): Vec3 {
  out[0] = a[0] * s;
  out[1] = a[1] * s;
  out[2] = a[2] * s;
  return out;
}

/** out = a + s * b (axpy). */
export function vec3AddScaled(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  s: number,
  out: Vec3 = new Float64Array(3),
): Vec3 {
  out[0] = a[0] + s * b[0];
  out[1] = a[1] + s * b[1];
  out[2] = a[2] + s * b[2];
  return out;
}

/** a · b */
export function vec3Dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** out = a × b (safe when out aliases a or b). */
export function vec3Cross(a: ArrayLike<number>, b: ArrayLike<number>, out: Vec3 = new Float64Array(3)): Vec3 {
  const ax = a[0],
    ay = a[1],
    az = a[2];
  const bx = b[0],
    by = b[1],
    bz = b[2];
  out[0] = ay * bz - az * by;
  out[1] = az * bx - ax * bz;
  out[2] = ax * by - ay * bx;
  return out;
}

/** Euclidean norm (overflow/underflow safe via Math.hypot only when needed). */
export function vec3Norm(a: ArrayLike<number>): number {
  const s = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
  if (s > 1e-300 && s < 1e300) return Math.sqrt(s);
  return Math.hypot(a[0], a[1], a[2]);
}

/** Squared Euclidean norm. */
export function vec3NormSq(a: ArrayLike<number>): number {
  return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}

/** Unit vector in the direction of a. A zero (or non-finite-norm) input yields the zero vector. */
export function vec3Normalize(a: ArrayLike<number>, out: Vec3 = new Float64Array(3)): Vec3 {
  const n = vec3Norm(a);
  if (n > 0 && Number.isFinite(n)) {
    const inv = 1 / n;
    out[0] = a[0] * inv;
    out[1] = a[1] * inv;
    out[2] = a[2] * inv;
  } else {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
  }
  return out;
}

/** Euclidean distance |a - b|. */
export function vec3Dist(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const dx = a[0] - b[0],
    dy = a[1] - b[1],
    dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** Squared Euclidean distance |a - b|². */
export function vec3DistSq(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const dx = a[0] - b[0],
    dy = a[1] - b[1],
    dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

/** Copy the first three components of a. */
export function vec3Copy(a: ArrayLike<number>, out: Vec3 = new Float64Array(3)): Vec3 {
  out[0] = a[0];
  out[1] = a[1];
  out[2] = a[2];
  return out;
}

/** Set components. */
export function vec3Set(out: Vec3, x: number, y: number, z: number): Vec3 {
  out[0] = x;
  out[1] = y;
  out[2] = z;
  return out;
}

/** out = a + t (b - a) */
export function vec3Lerp(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  t: number,
  out: Vec3 = new Float64Array(3),
): Vec3 {
  out[0] = a[0] + t * (b[0] - a[0]);
  out[1] = a[1] + t * (b[1] - a[1]);
  out[2] = a[2] + t * (b[2] - a[2]);
  return out;
}

/** Angle between two (not necessarily unit) vectors in [0, π]; 0 if either is zero. Uses atan2 for accuracy. */
export function vec3Angle(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const cx = a[1] * b[2] - a[2] * b[1];
  const cy = a[2] * b[0] - a[0] * b[2];
  const cz = a[0] * b[1] - a[1] * b[0];
  return Math.atan2(Math.sqrt(cx * cx + cy * cy + cz * cz), a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}
