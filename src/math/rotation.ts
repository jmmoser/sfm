/**
 * Rotation parameterisations: unit quaternions [w, x, y, z] (Hamilton), rotation
 * vectors (axis·angle, exp/log maps of SO(3)) and row-major 3x3 matrices. All
 * functions taking `out` write into it (when given) and return it; `out` may alias
 * inputs unless stated otherwise.
 */
import type { Mat3, Quat, Vec3 } from '../core/types';
import { svd3 } from './decomp';

/**
 * Rotation matrix of a quaternion [w, x, y, z]. Non-unit input is handled as if
 * normalised (R = I + 2/|q|² (…)); the zero quaternion gives the identity.
 */
export function quatToMat3(q: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  const w = q[0],
    x = q[1],
    y = q[2],
    z = q[3];
  const n2 = w * w + x * x + y * y + z * z;
  const s = n2 > 0 ? 2 / n2 : 0;
  const xs = x * s,
    ys = y * s,
    zs = z * s;
  const wx = w * xs,
    wy = w * ys,
    wz = w * zs;
  const xx = x * xs,
    xy = x * ys,
    xz = x * zs;
  const yy = y * ys,
    yz = y * zs,
    zz = z * zs;
  out[0] = 1 - (yy + zz);
  out[1] = xy - wz;
  out[2] = xz + wy;
  out[3] = xy + wz;
  out[4] = 1 - (xx + zz);
  out[5] = yz - wx;
  out[6] = xz - wy;
  out[7] = yz + wx;
  out[8] = 1 - (xx + yy);
  return out;
}

/**
 * Unit quaternion of a rotation matrix by Shepperd's method (1978): the largest of
 * the four quantities 1±R00±R11±R22 is used for the square root, so the division is
 * always well conditioned. Result is normalised and has w ≥ 0; non-finite input → NaN.
 */
export function mat3ToQuat(R: ArrayLike<number>, out: Quat = new Float64Array(4)): Quat {
  const r00 = R[0],
    r01 = R[1],
    r02 = R[2],
    r10 = R[3],
    r11 = R[4],
    r12 = R[5],
    r20 = R[6],
    r21 = R[7],
    r22 = R[8];
  const tr = r00 + r11 + r22;
  let w: number, x: number, y: number, z: number;
  if (tr >= r00 && tr >= r11 && tr >= r22) {
    const s = 2 * Math.sqrt(Math.max(0, 1 + tr)); // 4w
    w = 0.25 * s;
    x = (r21 - r12) / s;
    y = (r02 - r20) / s;
    z = (r10 - r01) / s;
  } else if (r00 >= r11 && r00 >= r22) {
    const s = 2 * Math.sqrt(Math.max(0, 1 + r00 - r11 - r22)); // 4x
    x = 0.25 * s;
    w = (r21 - r12) / s;
    y = (r01 + r10) / s;
    z = (r02 + r20) / s;
  } else if (r11 >= r22) {
    const s = 2 * Math.sqrt(Math.max(0, 1 - r00 + r11 - r22)); // 4y
    y = 0.25 * s;
    w = (r02 - r20) / s;
    x = (r01 + r10) / s;
    z = (r12 + r21) / s;
  } else {
    const s = 2 * Math.sqrt(Math.max(0, 1 - r00 - r11 + r22)); // 4z
    z = 0.25 * s;
    w = (r10 - r01) / s;
    x = (r02 + r20) / s;
    y = (r12 + r21) / s;
  }
  let n = Math.sqrt(w * w + x * x + y * y + z * z);
  if (!Number.isFinite(n)) {
    // Non-finite input → NaN (was silently mapped to the identity).
    out[0] = out[1] = out[2] = out[3] = NaN;
    return out;
  }
  if (!(n > 0)) {
    out[0] = 1;
    out[1] = out[2] = out[3] = 0;
    return out;
  }
  if (w < 0) n = -n;
  out[0] = w / n;
  out[1] = x / n;
  out[2] = y / n;
  out[3] = z / n;
  return out;
}

/** Hamilton product a ⊗ b (so that quatToMat3(a ⊗ b) = R(a) R(b)). */
export function quatMul(a: ArrayLike<number>, b: ArrayLike<number>, out: Quat = new Float64Array(4)): Quat {
  const aw = a[0],
    ax = a[1],
    ay = a[2],
    az = a[3];
  const bw = b[0],
    bx = b[1],
    by = b[2],
    bz = b[3];
  out[0] = aw * bw - ax * bx - ay * by - az * bz;
  out[1] = aw * bx + ax * bw + ay * bz - az * by;
  out[2] = aw * by - ax * bz + ay * bw + az * bx;
  out[3] = aw * bz + ax * by - ay * bx + az * bw;
  return out;
}

/** Conjugate [w, −x, −y, −z] (the inverse for unit quaternions). */
export function quatConj(q: ArrayLike<number>, out: Quat = new Float64Array(4)): Quat {
  out[0] = q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] = -q[3];
  return out;
}

/** Normalise to unit length; the zero quaternion maps to the identity [1, 0, 0, 0]; non-finite input → NaN. */
export function quatNormalize(q: ArrayLike<number>, out: Quat = new Float64Array(4)): Quat {
  let n = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (!(n > 1e-150 && n < 1e150)) n = Math.hypot(q[0], q[1], q[2], q[3]); // over/underflow-safe
  if (n !== n || n === Infinity) {
    out[0] = out[1] = out[2] = out[3] = NaN;
    return out;
  }
  if (!(n > 0)) {
    out[0] = 1;
    out[1] = out[2] = out[3] = 0;
    return out;
  }
  const inv = 1 / n;
  out[0] = q[0] * inv;
  out[1] = q[1] * inv;
  out[2] = q[2] * inv;
  out[3] = q[3] * inv;
  return out;
}

/**
 * Spherical linear interpolation between unit quaternions along the shortest arc
 * (b is negated when a·b < 0). t = 0 → a, t = 1 → ±b. Uses the atan2 form of the
 * angle and falls back to normalised lerp for nearly identical inputs.
 */
export function quatSlerp(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  t: number,
  out: Quat = new Float64Array(4),
): Quat {
  const aw = a[0],
    ax = a[1],
    ay = a[2],
    az = a[3];
  let bw = b[0],
    bx = b[1],
    by = b[2],
    bz = b[3];
  let dot = aw * bw + ax * bx + ay * by + az * bz;
  if (dot < 0) {
    dot = -dot;
    bw = -bw;
    bx = -bx;
    by = -by;
    bz = -bz;
  }
  // |a − b| and |a + b| give an accurate half-angle: θ = 2 atan2(|a−b|, |a+b|).
  const dw = aw - bw,
    dx = ax - bx,
    dy = ay - by,
    dz = az - bz;
  const sw = aw + bw,
    sx = ax + bx,
    sy = ay + by,
    sz = az + bz;
  const theta = 2 * Math.atan2(Math.sqrt(dw * dw + dx * dx + dy * dy + dz * dz), Math.sqrt(sw * sw + sx * sx + sy * sy + sz * sz));
  let wa: number, wb: number;
  const sinT = Math.sin(theta);
  if (theta < 1e-6 || sinT < 1e-12) {
    wa = 1 - t;
    wb = t;
  } else {
    wa = Math.sin((1 - t) * theta) / sinT;
    wb = Math.sin(t * theta) / sinT;
  }
  const rw = wa * aw + wb * bw,
    rx = wa * ax + wb * bx,
    ry = wa * ay + wb * by,
    rz = wa * az + wb * bz;
  const n = Math.sqrt(rw * rw + rx * rx + ry * ry + rz * rz);
  const inv = n > 0 ? 1 / n : 0;
  out[0] = rw * inv;
  out[1] = rx * inv;
  out[2] = ry * inv;
  out[3] = rz * inv;
  return out;
}

/** Rotate v by unit quaternion q (out = R(q) v). */
export function quatRotateVec(q: ArrayLike<number>, v: ArrayLike<number>, out: Vec3 = new Float64Array(3)): Vec3 {
  const w = q[0],
    x = q[1],
    y = q[2],
    z = q[3];
  const vx = v[0],
    vy = v[1],
    vz = v[2];
  // t = 2 q_vec × v ; v' = v + w t + q_vec × t
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  out[0] = vx + w * tx + (y * tz - z * ty);
  out[1] = vy + w * ty + (z * tx - x * tz);
  out[2] = vz + w * tz + (x * ty - y * tx);
  return out;
}

/**
 * Exponential map so(3) → SO(3) (Rodrigues): R = cos θ I + (sin θ/θ) [w]× +
 * ((1 − cos θ)/θ²) w wᵀ, θ = |w|. (1 − cos θ)/θ² is evaluated as
 * ½ (sin(θ/2)/(θ/2))² (no cancellation); Taylor series below θ = 1e-6.
 */
export function rotvecToMat3(w: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  const x = w[0],
    y = w[1],
    z = w[2];
  const t2 = x * x + y * y + z * z;
  let A: number, B: number, C: number;
  if (t2 < 1e-12) {
    A = 1 - t2 / 6; // sin θ / θ
    B = 0.5 - t2 / 24; // (1 − cos θ) / θ²
    C = 1 - 0.5 * t2; // cos θ
  } else {
    const t = Math.sqrt(t2);
    A = Math.sin(t) / t;
    const h = Math.sin(0.5 * t) / (0.5 * t);
    B = 0.5 * h * h;
    C = Math.cos(t);
  }
  const bxy = B * x * y,
    bxz = B * x * z,
    byz = B * y * z;
  out[0] = C + B * x * x;
  out[1] = bxy - A * z;
  out[2] = bxz + A * y;
  out[3] = bxy + A * z;
  out[4] = C + B * y * y;
  out[5] = byz - A * x;
  out[6] = bxz - A * y;
  out[7] = byz + A * x;
  out[8] = C + B * z * z;
  return out;
}

/**
 * Logarithm map SO(3) → so(3): rotation vector w = θ n with θ ∈ [0, π].
 * θ = atan2(|v|, (tr R − 1)/2) with v = vee((R − Rᵀ)/2) = sin θ n (accurate at all
 * angles). Near 0 the axis-scale θ/sin θ uses a Taylor series; near π (where v
 * vanishes) the axis is taken from the symmetric part (R + Rᵀ)/2 = cos θ I +
 * (1 − cos θ) n nᵀ using its largest diagonal entry, with the sign fixed by v.
 */
export function mat3ToRotvec(R: ArrayLike<number>, out: Vec3 = new Float64Array(3)): Vec3 {
  const vx = 0.5 * (R[7] - R[5]);
  const vy = 0.5 * (R[2] - R[6]);
  const vz = 0.5 * (R[3] - R[1]);
  const s = Math.sqrt(vx * vx + vy * vy + vz * vz);
  const c = 0.5 * (R[0] + R[4] + R[8] - 1);
  if (!Number.isFinite(s) || !Number.isFinite(c)) {
    // Non-finite input → NaN (was silently mapped to the zero rotation vector).
    out[0] = out[1] = out[2] = NaN;
    return out;
  }
  const theta = Math.atan2(s, c);
  if (c > -0.5) {
    // θ < 2π/3: v is a well-conditioned axis estimate.
    let f: number;
    if (s < 1e-6) {
      // θ/sin θ ≈ 1 + θ²/6 (θ ≈ s here)
      f = 1 + (s * s) / 6;
    } else {
      f = theta / s;
    }
    out[0] = vx * f;
    out[1] = vy * f;
    out[2] = vz * f;
    return out;
  }
  // Near π: n nᵀ = (S − c I)/(1 − c), S = (R + Rᵀ)/2.
  const omc = 1 - c;
  const d0 = (R[0] - c) / omc,
    d1 = (R[4] - c) / omc,
    d2 = (R[8] - c) / omc;
  const s01 = (0.5 * (R[1] + R[3])) / omc,
    s02 = (0.5 * (R[2] + R[6])) / omc,
    s12 = (0.5 * (R[5] + R[7])) / omc;
  let nx: number, ny: number, nz: number;
  if (d0 >= d1 && d0 >= d2) {
    nx = Math.sqrt(Math.max(d0, 0));
    ny = s01 / nx;
    nz = s02 / nx;
  } else if (d1 >= d2) {
    ny = Math.sqrt(Math.max(d1, 0));
    nx = s01 / ny;
    nz = s12 / ny;
  } else {
    nz = Math.sqrt(Math.max(d2, 0));
    nx = s02 / nz;
    ny = s12 / nz;
  }
  let nn = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (!(nn > 0) || !Number.isFinite(nn)) {
    out[0] = out[1] = out[2] = 0;
    return out;
  }
  if (nx * vx + ny * vy + nz * vz < 0) nn = -nn;
  const k = theta / nn;
  out[0] = nx * k;
  out[1] = ny * k;
  out[2] = nz * k;
  return out;
}

/** Rotation by `angle` (radians, right-handed) about `axis` (need not be unit; zero axis → identity; non-finite → NaN). */
export function axisAngleToMat3(axis: ArrayLike<number>, angle: number, out: Mat3 = new Float64Array(9)): Mat3 {
  let n = Math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
  if (!(n > 1e-150 && n < 1e150)) n = Math.hypot(axis[0], axis[1], axis[2]); // over/underflow-safe
  if (n !== n || n === Infinity) {
    out.fill(NaN);
    return out;
  }
  if (!(n > 0)) {
    out.fill(0);
    out[0] = out[4] = out[8] = 1;
    return out;
  }
  const k = angle / n;
  const x = axis[0] * k,
    y = axis[1] * k,
    z = axis[2] * k;
  const w = SCR_V3;
  w[0] = x;
  w[1] = y;
  w[2] = z;
  return rotvecToMat3(w, out);
}

const SCR_V3 = new Float64Array(3);
const SCR_M3 = new Float64Array(9);

/** Rotation angle θ ∈ [0, π] of R, computed as atan2(|vee(R − Rᵀ)|/2, (tr R − 1)/2) (accurate near 0 and π). */
export function rotationAngle(R: ArrayLike<number>): number {
  const vx = R[7] - R[5],
    vy = R[2] - R[6],
    vz = R[3] - R[1];
  const s = 0.5 * Math.sqrt(vx * vx + vy * vy + vz * vz);
  const c = 0.5 * (R[0] + R[4] + R[8] - 1);
  return Math.atan2(s, c);
}

/** Angle (radians, [0, π]) of the relative rotation Ra Rbᵀ. */
export function rotationAngleBetween(Ra: ArrayLike<number>, Rb: ArrayLike<number>): number {
  const M = SCR_M3;
  for (let i = 0; i < 3; i++) {
    const a0 = Ra[3 * i],
      a1 = Ra[3 * i + 1],
      a2 = Ra[3 * i + 2];
    for (let j = 0; j < 3; j++) M[3 * i + j] = a0 * Rb[3 * j] + a1 * Rb[3 * j + 1] + a2 * Rb[3 * j + 2];
  }
  return rotationAngle(M);
}

/**
 * Nearest rotation matrix (Frobenius norm) to R: U diag(1, 1, det(U Vᵀ)) Vᵀ from the
 * SVD R = U Σ Vᵀ (always det = +1, also for reflections/degenerate input).
 */
export function orthonormalize(R: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  const A = SCR_M3;
  for (let i = 0; i < 9; i++) A[i] = R[i];
  const { U, V } = svd3(A);
  // det(U Vᵀ) = det(U) det(V)
  const detU =
    U[0] * (U[4] * U[8] - U[5] * U[7]) - U[1] * (U[3] * U[8] - U[5] * U[6]) + U[2] * (U[3] * U[7] - U[4] * U[6]);
  const detV =
    V[0] * (V[4] * V[8] - V[5] * V[7]) - V[1] * (V[3] * V[8] - V[5] * V[6]) + V[2] * (V[3] * V[7] - V[4] * V[6]);
  const d = detU * detV < 0 ? -1 : 1;
  for (let i = 0; i < 3; i++) {
    const u0 = U[3 * i],
      u1 = U[3 * i + 1],
      u2 = U[3 * i + 2] * d;
    for (let j = 0; j < 3; j++) out[3 * i + j] = u0 * V[3 * j] + u1 * V[3 * j + 1] + u2 * V[3 * j + 2];
  }
  return out;
}

/**
 * Minimal rotation R with R â = b̂ (â, b̂ = a, b normalised): rotation about â × b̂ by
 * the angle atan2(|â × b̂|, â·b̂), accurate to O(ε) also for nearly (anti)parallel
 * input. For exactly antiparallel input (|â × b̂| ≤ 1e-150) a rotation by π about an
 * axis perpendicular to â is returned; for exactly parallel input, the identity.
 * Zero input yields the identity; non-finite input yields NaN.
 */
export function rotationBetweenVectors(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  out: Mat3 = new Float64Array(9),
): Mat3 {
  let na = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  let nb = Math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
  if (!(na > 1e-150 && na < 1e150)) na = Math.hypot(a[0], a[1], a[2]); // over/underflow-safe
  if (!(nb > 1e-150 && nb < 1e150)) nb = Math.hypot(b[0], b[1], b[2]);
  if (!Number.isFinite(na) || !Number.isFinite(nb)) {
    // Non-finite input → NaN (was silently mapped to the identity).
    out.fill(NaN);
    return out;
  }
  if (!(na > 0) || !(nb > 0)) {
    out.fill(0);
    out[0] = out[4] = out[8] = 1;
    return out;
  }
  const ax = a[0] / na,
    ay = a[1] / na,
    az = a[2] / na;
  const bx = b[0] / nb,
    by = b[1] / nb,
    bz = b[2] / nb;
  let cx = ay * bz - az * by,
    cy = az * bx - ax * bz,
    cz = ax * by - ay * bx;
  const s = Math.sqrt(cx * cx + cy * cy + cz * cz);
  const c = ax * bx + ay * by + az * bz;
  const theta = Math.atan2(s, c);
  if (!(s > 1e-150)) {
    // (Numerically) parallel or antiparallel: the cross product carries no direction.
    // Above this threshold the general branch stays accurate: an axis error δ is
    // multiplied by sin θ ≈ |â × b̂| in R â, so the residual is O(ε).
    if (c > 0) {
      out.fill(0);
      out[0] = out[4] = out[8] = 1;
      return out;
    }
    // Antiparallel: axis ⟂ a, from the coordinate axis least aligned with a.
    const fx = Math.abs(ax),
      fy = Math.abs(ay),
      fz = Math.abs(az);
    let ex = 0,
      ey = 0,
      ez = 0;
    if (fx <= fy && fx <= fz) ex = 1;
    else if (fy <= fz) ey = 1;
    else ez = 1;
    cx = ay * ez - az * ey;
    cy = az * ex - ax * ez;
    cz = ax * ey - ay * ex;
  } else {
    // Remove any component along â (rounding) so the axis is exactly ⟂ â.
    const d = cx * ax + cy * ay + cz * az;
    cx -= d * ax;
    cy -= d * ay;
    cz -= d * az;
  }
  const cn = Math.sqrt(cx * cx + cy * cy + cz * cz);
  const k = theta / cn;
  const w = SCR_V3;
  w[0] = cx * k;
  w[1] = cy * k;
  w[2] = cz * k;
  return rotvecToMat3(w, out);
}
