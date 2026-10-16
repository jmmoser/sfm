/**
 * Rigid world-to-camera poses: x_cam = R X + t, camera centre C = −Rᵀ t.
 */
import type { Mat3, Mat34, Pose, Vec3 } from '../core/types';

/** Identity pose (R = I, t = 0). */
export function poseIdentity(): Pose {
  const R = new Float64Array(9);
  R[0] = R[4] = R[8] = 1;
  return { R, t: new Float64Array(3) };
}

/** Deep copy. */
export function poseCopy(p: Pose): Pose {
  return { R: Float64Array.from(p.R), t: Float64Array.from(p.t) };
}

/** Camera centre C = −Rᵀ t. */
export function poseCenter(p: Pose, out: Vec3 = new Float64Array(3)): Vec3 {
  const R = p.R,
    t = p.t;
  const t0 = t[0],
    t1 = t[1],
    t2 = t[2];
  out[0] = -(R[0] * t0 + R[3] * t1 + R[6] * t2);
  out[1] = -(R[1] * t0 + R[4] * t1 + R[7] * t2);
  out[2] = -(R[2] * t0 + R[5] * t1 + R[8] * t2);
  return out;
}

/** Pose with rotation R and camera centre C (t = −R C). R is copied. */
export function poseFromCenter(R: ArrayLike<number>, C: ArrayLike<number>): Pose {
  const Rc = Float64Array.from(R as ArrayLike<number>);
  const t = new Float64Array(3);
  t[0] = -(Rc[0] * C[0] + Rc[1] * C[1] + Rc[2] * C[2]);
  t[1] = -(Rc[3] * C[0] + Rc[4] * C[1] + Rc[5] * C[2]);
  t[2] = -(Rc[6] * C[0] + Rc[7] * C[1] + Rc[8] * C[2]);
  return { R: Rc, t };
}

/** Composition a∘b : x ↦ a(b(x)) = R_a R_b x + (R_a t_b + t_a). */
export function poseCompose(a: Pose, b: Pose): Pose {
  const Ra = a.R,
    Rb = b.R;
  const R = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    const a0 = Ra[3 * i],
      a1 = Ra[3 * i + 1],
      a2 = Ra[3 * i + 2];
    R[3 * i] = a0 * Rb[0] + a1 * Rb[3] + a2 * Rb[6];
    R[3 * i + 1] = a0 * Rb[1] + a1 * Rb[4] + a2 * Rb[7];
    R[3 * i + 2] = a0 * Rb[2] + a1 * Rb[5] + a2 * Rb[8];
  }
  const tb = b.t;
  const t = new Float64Array(3);
  for (let i = 0; i < 3; i++) t[i] = Ra[3 * i] * tb[0] + Ra[3 * i + 1] * tb[1] + Ra[3 * i + 2] * tb[2] + a.t[i];
  return { R, t };
}

/** Inverse transform: x ↦ Rᵀ (x − t), i.e. {Rᵀ, −Rᵀ t}. */
export function poseInverse(p: Pose): Pose {
  const R = p.R;
  const Rt = new Float64Array(9);
  Rt[0] = R[0];
  Rt[1] = R[3];
  Rt[2] = R[6];
  Rt[3] = R[1];
  Rt[4] = R[4];
  Rt[5] = R[7];
  Rt[6] = R[2];
  Rt[7] = R[5];
  Rt[8] = R[8];
  return { R: Rt, t: poseCenter(p) };
}

/**
 * Relative pose taking camera-1 coordinates to camera-2 coordinates:
 * x2 = R x1 + t with R = R2 R1ᵀ, t = t2 − R t1 (= p2 ∘ p1⁻¹).
 */
export function poseRelative(p1: Pose, p2: Pose): Pose {
  const R1 = p1.R,
    R2 = p2.R;
  const R = new Float64Array(9);
  // R2 R1ᵀ
  for (let i = 0; i < 3; i++) {
    const a0 = R2[3 * i],
      a1 = R2[3 * i + 1],
      a2 = R2[3 * i + 2];
    for (let j = 0; j < 3; j++) R[3 * i + j] = a0 * R1[3 * j] + a1 * R1[3 * j + 1] + a2 * R1[3 * j + 2];
  }
  const t1 = p1.t;
  const t = new Float64Array(3);
  for (let i = 0; i < 3; i++) t[i] = p2.t[i] - (R[3 * i] * t1[0] + R[3 * i + 1] * t1[1] + R[3 * i + 2] * t1[2]);
  return { R, t };
}

/** 3×4 row-major [R | t]. */
export function poseToMat34(p: Pose, out: Mat34 = new Float64Array(12)): Mat34 {
  const R = p.R,
    t = p.t;
  for (let i = 0; i < 3; i++) {
    out[4 * i] = R[3 * i];
    out[4 * i + 1] = R[3 * i + 1];
    out[4 * i + 2] = R[3 * i + 2];
    out[4 * i + 3] = t[i];
  }
  return out;
}

/** Pose from a 3×4 row-major [R | t] (copied). */
export function poseFromMat34(m: ArrayLike<number>): Pose {
  const R: Mat3 = new Float64Array(9);
  const t: Vec3 = new Float64Array(3);
  for (let i = 0; i < 3; i++) {
    R[3 * i] = m[4 * i];
    R[3 * i + 1] = m[4 * i + 1];
    R[3 * i + 2] = m[4 * i + 2];
    t[i] = m[4 * i + 3];
  }
  return { R, t };
}

/** R X + t (out may alias X). */
export function poseTransformPoint(p: Pose, X: ArrayLike<number>, out: Vec3 = new Float64Array(3)): Vec3 {
  const R = p.R,
    t = p.t;
  const x = X[0],
    y = X[1],
    z = X[2];
  out[0] = R[0] * x + R[1] * y + R[2] * z + t[0];
  out[1] = R[3] * x + R[4] * y + R[5] * z + t[1];
  out[2] = R[6] * x + R[7] * y + R[8] * z + t[2];
  return out;
}
