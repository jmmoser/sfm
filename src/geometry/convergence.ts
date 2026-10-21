/**
 * Optical-axis convergence of a set of cameras: the least-squares point closest to every
 * optical axis, and whether the cameras look at it (an orbit around an object) or not (a walk).
 *
 * Shared by the engine's object / scene classification (`classifyScene`, src/engine/live.ts),
 * the capture screen's coverage ring (src/app/coverage.ts), the object crop box
 * (src/dense/object-roi.ts) and the fly-through path (src/viewer/flythrough.ts), so all of them
 * agree on where "the object" is.
 *
 * The point P minimises Σ |(I − dᵢdᵢᵀ)(P − Cᵢ)|² over the camera centres Cᵢ and unit optical
 * axes dᵢ (third row of R, OpenCV frame). The 3×3 normal matrix is singular when every axis is
 * parallel (pure translation, no rotation), so `point` is null below |det| 1e-9 or when the
 * axes spread less than 10°. `converging` is the orbit test: ≥ 90 % of the cameras have P in
 * front of them, and the median distance of P from the axes is < 0.35 × the median camera
 * distance to P.
 */
import type { Pose, Vec3 } from '../core/types';
import { median } from '../math/stats';

const DEG = Math.PI / 180;

/** Result of `opticalAxisConvergence`. */
export interface AxisConvergence {
  /** Least-squares convergence point (world), or null when the axes do not constrain it. */
  point: Vec3 | null;
  /** Angle between the mean viewing direction and the unit sphere, rad (0 = all parallel). */
  spreadRad: number;
  /** Cameras with the point in front of them (along their optical axis). */
  inFront: number;
  /** Median camera-centre distance to the point (0 when point is null). */
  distMedian: number;
  /** Median distance of the point from the optical axes (0 when point is null). */
  offAxisMedian: number;
  /** Orbit test: point in front of ≥ 90 % of cameras and offAxisMedian < 0.35 · distMedian. */
  converging: boolean;
}

/** World centre C = −Rᵀ t. */
function centre(p: Pose): Float64Array {
  const { R, t } = p;
  return Float64Array.of(
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  );
}

/** Least-squares convergence point of the optical axes and the orbit test. Needs ≥ 2 poses. */
export function opticalAxisConvergence(poses: readonly Pose[]): AxisConvergence {
  const n = poses.length;
  const A = new Float64Array(9);
  const bv = new Float64Array(3);
  const Cs: Float64Array[] = [];
  const Ds: Float64Array[] = [];
  for (const p of poses) {
    const C = centre(p);
    const d = Float64Array.of(p.R[6], p.R[7], p.R[8]);
    Cs.push(C);
    Ds.push(d);
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++) {
        const m = (r === c ? 1 : 0) - d[r] * d[c];
        A[3 * r + c] += m;
        bv[r] += m * C[c];
      }
  }
  const det =
    A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
  // angular spread of the viewing directions
  const mean = new Float64Array(3);
  for (const d of Ds) for (let k = 0; k < 3; k++) mean[k] += d[k] / Math.max(1, n);
  const spreadRad = n > 0 ? Math.acos(Math.min(1, Math.hypot(mean[0], mean[1], mean[2]))) : 0;
  const none: AxisConvergence = { point: null, spreadRad, inFront: 0, distMedian: 0, offAxisMedian: 0, converging: false };
  if (n < 2 || !(Math.abs(det) > 1e-9) || spreadRad < 10 * DEG) return none;
  const m = (r: number, c: number): number => A[3 * r + c];
  const cof = [
    m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1),
    -(m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)),
    m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0),
    -(m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1)),
    m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0),
    -(m(0, 0) * m(2, 1) - m(0, 1) * m(2, 0)),
    m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1),
    -(m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0)),
    m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0),
  ];
  const inv = (i: number, j: number): number => cof[3 * j + i] / det;
  const P = new Float64Array(3);
  for (let i = 0; i < 3; i++) P[i] = inv(i, 0) * bv[0] + inv(i, 1) * bv[1] + inv(i, 2) * bv[2];
  let inFront = 0;
  const dist: number[] = [];
  const off: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = [P[0] - Cs[i][0], P[1] - Cs[i][1], P[2] - Cs[i][2]];
    const along = v[0] * Ds[i][0] + v[1] * Ds[i][1] + v[2] * Ds[i][2];
    const len = Math.hypot(v[0], v[1], v[2]);
    if (along > 0) inFront++;
    dist.push(len);
    off.push(Math.sqrt(Math.max(0, len * len - along * along)));
  }
  const distMedian = median(dist);
  const offAxisMedian = median(off);
  const converging = inFront >= 0.9 * n && offAxisMedian < 0.35 * distMedian;
  return { point: P, spreadRad, inFront, distMedian, offAxisMedian, converging };
}
