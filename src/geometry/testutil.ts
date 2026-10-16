/**
 * Synthetic data generators for the geometry tests (browser-safe, seeded Rng only).
 * Not part of the public API (not re-exported from index.ts).
 */
import type { CameraIntrinsics, Mat3, Pose, Vec3 } from '../core/types';
import { axisAngleToMat3, rotationAngleBetween } from '../math/rotation';
import type { Rng } from '../math/random';
import { projectCam } from './camera';

/** Uniformly distributed unit vector. */
export function randomUnitVector(rng: Rng): Vec3 {
  for (;;) {
    const x = rng.gaussian(),
      y = rng.gaussian(),
      z = rng.gaussian();
    const n = Math.hypot(x, y, z);
    if (n > 1e-6) return Float64Array.of(x / n, y / n, z / n);
  }
}

/** Rotation about a random axis by an angle uniform in [minAngle, maxAngle]. */
export function randomRotation(rng: Rng, maxAngle: number, minAngle = 0): Mat3 {
  return axisAngleToMat3(randomUnitVector(rng), minAngle + (maxAngle - minAngle) * rng.next());
}

/** Rotation error in degrees. */
export function rotErrDeg(Ra: ArrayLike<number>, Rb: ArrayLike<number>): number {
  return (rotationAngleBetween(Ra, Rb) * 180) / Math.PI;
}

/** Angle (degrees) between two direction vectors. */
export function dirErrDeg(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const s = Math.hypot(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
  return (Math.atan2(s, c) * 180) / Math.PI;
}

/** Min over sign of ‖A/‖A‖ ∓ B/‖B‖‖_F. */
export function matDistUpToScale(A: ArrayLike<number>, B: ArrayLike<number>): number {
  let na = 0,
    nb = 0;
  for (let i = 0; i < 9; i++) {
    na += A[i] * A[i];
    nb += B[i] * B[i];
  }
  na = Math.sqrt(na);
  nb = Math.sqrt(nb);
  let dp = 0,
    dm = 0;
  for (let i = 0; i < 9; i++) {
    const a = A[i] / na,
      b = B[i] / nb;
    dp += (a - b) * (a - b);
    dm += (a + b) * (a + b);
  }
  return Math.sqrt(Math.min(dp, dm));
}

export interface TwoViewSceneOptions {
  n: number;
  depthMin: number;
  depthMax: number;
  /** Distance between the camera centres (0 = pure rotation). */
  baseline: number;
  /** Max relative rotation angle (rad). */
  maxAngle: number;
  /** Half field of view tangent used to sample points (default tan(35°)). */
  halfFovTan?: number;
  /** All points on a plane (through the scene centre, random tilt ≤ 60°). */
  planar?: boolean;
  /** Translation direction (unit, camera-1 frame, of camera centre 2), default random. */
  centerDir?: ArrayLike<number>;
}

export interface TwoViewScene {
  /** x2 = R x1 + t (t has length = baseline). */
  R: Mat3;
  t: Vec3;
  /** 3-D points in the camera-1 frame (3n). */
  X: Float64Array;
  /** Normalised coordinates (2n) in views 1 and 2 (noise-free). */
  x1: Float64Array;
  x2: Float64Array;
  n: number;
}

/**
 * Two-view scene: camera 1 = identity, camera 2 rotated by a random R and centred
 * at C2 = baseline·dir. Points are sampled in camera 1's frustum (depth uniform in
 * [depthMin, depthMax]) and kept only if they project into camera 2's frustum.
 */
export function makeTwoViewScene(rng: Rng, o: TwoViewSceneOptions): TwoViewScene {
  const ht = o.halfFovTan ?? Math.tan((35 * Math.PI) / 180);
  for (let attempt = 0; attempt < 1000; attempt++) {
    const R = randomRotation(rng, o.maxAngle);
    const dir = o.centerDir ? Float64Array.from(o.centerDir) : randomUnitVector(rng);
    const C2 = Float64Array.of(dir[0] * o.baseline, dir[1] * o.baseline, dir[2] * o.baseline);
    // t = −R C2
    const t = Float64Array.of(
      -(R[0] * C2[0] + R[1] * C2[1] + R[2] * C2[2]),
      -(R[3] * C2[0] + R[4] * C2[1] + R[5] * C2[2]),
      -(R[6] * C2[0] + R[7] * C2[1] + R[8] * C2[2]),
    );
    let pn: Float64Array | null = null;
    let pd = 0;
    if (o.planar) {
      const tilt = randomRotation(rng, (60 * Math.PI) / 180);
      pn = Float64Array.of(tilt[2], tilt[5], tilt[8]); // R e_z
      if (pn[2] < 0) for (let k = 0; k < 3; k++) pn[k] = -pn[k];
      pd = 0.5 * (o.depthMin + o.depthMax) * pn[2]; // plane through (0,0,dmid)
    }
    const X = new Float64Array(3 * o.n);
    const x1 = new Float64Array(2 * o.n);
    const x2 = new Float64Array(2 * o.n);
    let k = 0;
    for (let tries = 0; tries < 200 * o.n && k < o.n; tries++) {
      const u = ht * (2 * rng.next() - 1),
        v = ht * (2 * rng.next() - 1);
      let z: number;
      if (pn) {
        const den = pn[0] * u + pn[1] * v + pn[2];
        if (!(den > 1e-3)) continue;
        z = pd / den;
        if (!(z > o.depthMin * 0.2 && z < o.depthMax * 5)) continue;
      } else {
        z = o.depthMin + (o.depthMax - o.depthMin) * rng.next();
      }
      const P = [u * z, v * z, z];
      const Q0 = R[0] * P[0] + R[1] * P[1] + R[2] * P[2] + t[0];
      const Q1 = R[3] * P[0] + R[4] * P[1] + R[5] * P[2] + t[1];
      const Q2 = R[6] * P[0] + R[7] * P[1] + R[8] * P[2] + t[2];
      if (!(Q2 > 1e-3 * o.depthMin)) continue;
      const u2 = Q0 / Q2,
        v2 = Q1 / Q2;
      if (Math.abs(u2) > ht || Math.abs(v2) > ht) continue;
      X[3 * k] = P[0];
      X[3 * k + 1] = P[1];
      X[3 * k + 2] = P[2];
      x1[2 * k] = u;
      x1[2 * k + 1] = v;
      x2[2 * k] = u2;
      x2[2 * k + 1] = v2;
      k++;
    }
    if (k === o.n) return { R, t, X, x1, x2, n: o.n };
  }
  throw new Error('makeTwoViewScene: could not generate a valid scene');
}

/** Normalised coordinates → pixels with optional Gaussian noise (σ px). */
export function toPixels(
  rng: Rng,
  cam: CameraIntrinsics,
  xn: Float64Array,
  n: number,
  sigmaPx: number,
): Float64Array {
  const out = new Float64Array(2 * n);
  const P = new Float64Array(3);
  const uv = new Float64Array(2);
  for (let i = 0; i < n; i++) {
    P[0] = xn[2 * i];
    P[1] = xn[2 * i + 1];
    P[2] = 1;
    projectCam(cam, P, uv);
    out[2 * i] = uv[0] + sigmaPx * rng.gaussian();
    out[2 * i + 1] = uv[1] + sigmaPx * rng.gaussian();
  }
  return out;
}

/** Random pose looking roughly at the origin from distance `dist`. */
export function randomPoseLookingAtOrigin(rng: Rng, dist: number, jitterRad = 0.1): Pose {
  // Camera centre on a sphere, optical axis towards the origin.
  const c = randomUnitVector(rng);
  const C = Float64Array.of(c[0] * dist, c[1] * dist, c[2] * dist);
  const z = Float64Array.of(-c[0], -c[1], -c[2]);
  const tmp = Math.abs(z[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  // x = normalize(tmp × z), y = z × x
  let x0 = tmp[1] * z[2] - tmp[2] * z[1],
    x1 = tmp[2] * z[0] - tmp[0] * z[2],
    x2 = tmp[0] * z[1] - tmp[1] * z[0];
  const nx = Math.hypot(x0, x1, x2);
  x0 /= nx;
  x1 /= nx;
  x2 /= nx;
  const y0 = z[1] * x2 - z[2] * x1,
    y1 = z[2] * x0 - z[0] * x2,
    y2 = z[0] * x1 - z[1] * x0;
  const R0 = Float64Array.of(x0, x1, x2, y0, y1, y2, z[0], z[1], z[2]);
  const J = randomRotation(rng, jitterRad);
  const R = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) R[3 * i + j] = J[3 * i] * R0[j] + J[3 * i + 1] * R0[3 + j] + J[3 * i + 2] * R0[6 + j];
  const t = Float64Array.of(
    -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
    -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
    -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
  );
  return { R, t };
}
