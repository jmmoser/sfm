import { describe, expect, it } from 'vitest';
import type { Pose } from '../core/types';
import { Rng } from '../math/random';
import { createIntrinsics, projectWorld } from './camera';
import { epnp, p3p, refinePose, solvePnPRansac } from './pnp';
import { poseCenter } from './pose';
import { randomPoseLookingAtOrigin, randomRotation, randomUnitVector, rotErrDeg } from './testutil';

/** n camera-frame points in a frustum, returned in world coordinates for `pose`. */
function makePoints(
  rng: Rng,
  pose: Pose,
  n: number,
  zmin: number,
  zmax: number,
  planar = false,
): { X: Float64Array; xn: Float64Array } {
  const X = new Float64Array(3 * n),
    xn = new Float64Array(2 * n);
  const R = pose.R,
    t = pose.t;
  const tilt = randomRotation(rng, 1.0);
  const zc = 0.5 * (zmin + zmax);
  for (let i = 0; i < n; i++) {
    let P: number[];
    if (planar) {
      const a = rng.uniform(-0.4, 0.4) * zc,
        b = rng.uniform(-0.3, 0.3) * zc;
      P = [tilt[0] * a + tilt[1] * b, tilt[3] * a + tilt[4] * b, zc + tilt[6] * a + tilt[7] * b];
    } else {
      const z = rng.uniform(zmin, zmax);
      P = [rng.uniform(-0.6, 0.6) * z, rng.uniform(-0.45, 0.45) * z, z];
    }
    xn[2 * i] = P[0] / P[2];
    xn[2 * i + 1] = P[1] / P[2];
    const d = [P[0] - t[0], P[1] - t[1], P[2] - t[2]];
    for (let c = 0; c < 3; c++) X[3 * i + c] = R[c] * d[0] + R[3 + c] * d[1] + R[6 + c] * d[2];
  }
  return { X, xn };
}

function centerErr(a: Pose, b: Pose): number {
  const ca = poseCenter(a),
    cb = poseCenter(b);
  return Math.hypot(ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]);
}

function randomPose(rng: Rng, offset = 0): Pose {
  const R = randomRotation(rng, Math.PI);
  const C = randomUnitVector(rng).map((v) => v * rng.uniform(0, 5) + offset);
  const t = Float64Array.of(
    -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
    -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
    -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
  );
  return { R, t };
}

describe('p3p (Lambda Twist)', () => {
  it('the true pose is among the solutions for ≥ 99.9 % of 5000 random configurations', () => {
    const rng = new Rng(61);
    let ok = 0,
      nsol = 0,
      time = 0;
    const N = 5000;
    for (let k = 0; k < N; k++) {
      const pose = randomPose(rng);
      const { X, xn } = makePoints(rng, pose, 3, 0.5, 20);
      const B = new Float64Array(9);
      for (let i = 0; i < 3; i++) {
        const nn = Math.hypot(xn[2 * i], xn[2 * i + 1], 1);
        B[3 * i] = xn[2 * i] / nn;
        B[3 * i + 1] = xn[2 * i + 1] / nn;
        B[3 * i + 2] = 1 / nn;
      }
      const t0 = performance.now();
      const sols = p3p(X, B);
      time += performance.now() - t0;
      nsol += sols.length;
      expect(sols.length).toBeLessThanOrEqual(4);
      let best = Infinity;
      for (const s of sols) best = Math.min(best, rotErrDeg(s.R, pose.R) + centerErr(s, pose));
      if (best < 1e-6) ok++;
    }
    console.log(`[p3p] ${ok}/${N} ok, avg ${(nsol / N).toFixed(2)} solutions, ${((time / N) * 1000).toFixed(2)} µs/solve`);
    expect(ok / N).toBeGreaterThanOrEqual(0.999);
  });

  it('every solution reprojects the three points exactly; non-unit bearings accepted', () => {
    const rng = new Rng(62);
    for (let k = 0; k < 300; k++) {
      const pose = randomPose(rng);
      const { X, xn } = makePoints(rng, pose, 3, 1, 10);
      const B = Float64Array.of(xn[0], xn[1], 1, xn[2], xn[3], 1, xn[4], xn[5], 1).map((v) => v * 3);
      for (const s of p3p(X, B)) {
        for (let i = 0; i < 3; i++) {
          const P = [0, 1, 2].map(
            (r) => s.R[3 * r] * X[3 * i] + s.R[3 * r + 1] * X[3 * i + 1] + s.R[3 * r + 2] * X[3 * i + 2] + s.t[r],
          );
          expect(P[2]).toBeGreaterThan(0);
          expect(P[0] / P[2]).toBeCloseTo(xn[2 * i], 7);
          expect(P[1] / P[2]).toBeCloseTo(xn[2 * i + 1], 7);
        }
      }
    }
  });

  it('degenerate configurations give no solution', () => {
    const B = Float64Array.of(0, 0, 1, 0.1, 0, 1, 0.2, 0, 1);
    expect(p3p(Float64Array.of(0, 0, 0, 1, 1, 1, 2, 2, 2), B)).toEqual([]); // collinear
    expect(p3p(Float64Array.of(0, 0, 0, 0, 0, 0, 1, 0, 0), B)).toEqual([]); // coincident
  });
});

describe('epnp', () => {
  it('is exact on noise-free non-planar, planar and minimal (n = 4..6) point sets, also at ENU offsets', () => {
    const rng = new Rng(63);
    for (const planar of [false, true]) {
      for (const offset of [0, 1e5]) {
        let worst = 0;
        for (let k = 0; k < 100; k++) {
          const pose = randomPose(rng, offset);
          const n = [4, 5, 6, 10, 50][k % 5];
          const { X, xn } = makePoints(rng, pose, n, 2, 10, planar);
          const est = epnp(X, xn, n);
          expect(est).not.toBeNull();
          worst = Math.max(worst, rotErrDeg(est!.R, pose.R));
          expect(centerErr(est!, pose)).toBeLessThan(1e-4 * (1 + offset * 1e-5));
        }
        expect(worst).toBeLessThan(1e-5);
      }
    }
  });

  it('is accurate with 1 px noise (f = 1500)', () => {
    const rng = new Rng(64);
    for (const planar of [false, true]) {
      let sum = 0;
      for (let k = 0; k < 50; k++) {
        const pose = randomPose(rng);
        const { X, xn } = makePoints(rng, pose, 100, 2, 10, planar);
        for (let i = 0; i < 200; i++) xn[i] += rng.gaussian() / 1500;
        const est = epnp(X, xn, 100)!;
        sum += rotErrDeg(est.R, pose.R);
      }
      expect(sum / 50).toBeLessThan(0.2);
    }
  });

  it('returns null for too few or collinear points', () => {
    expect(epnp(new Float64Array(9), new Float64Array(6), 3)).toBeNull();
    const X = Float64Array.of(0, 0, 5, 1, 0, 5, 2, 0, 5, 3, 0, 5, 4, 0, 5);
    expect(epnp(X, new Float64Array(10), 5)).toBeNull();
  });
});

describe('refinePose', () => {
  it('converges from a perturbed pose (noise-free → exact; robust to outliers)', () => {
    const rng = new Rng(65);
    for (let k = 0; k < 30; k++) {
      const pose = randomPose(rng);
      const { X, xn } = makePoints(rng, pose, 80, 2, 10);
      const dR = randomRotation(rng, 0.05);
      const R0 = new Float64Array(9);
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) R0[3 * i + j] = dR[3 * i] * pose.R[j] + dR[3 * i + 1] * pose.R[3 + j] + dR[3 * i + 2] * pose.R[6 + j];
      const start: Pose = { R: R0, t: Float64Array.of(pose.t[0] + 0.1, pose.t[1] - 0.1, pose.t[2] + 0.05) };
      const ref = refinePose(start, X, xn, 80, { iterations: 30 });
      expect(rotErrDeg(ref.R, pose.R)).toBeLessThan(1e-7);
      expect(centerErr(ref, pose)).toBeLessThan(1e-7);
      // 20 % gross outliers: Huber keeps the estimate close
      const xo = Float64Array.from(xn);
      for (let i = 0; i < 16; i++) {
        xo[2 * i] += rng.uniform(-0.1, 0.1);
        xo[2 * i + 1] += rng.uniform(-0.1, 0.1);
      }
      const rob = refinePose(start, X, xo, 80, { iterations: 30, huberNormalized: 1e-3 });
      expect(rotErrDeg(rob.R, pose.R)).toBeLessThan(0.3);
    }
  });
});

describe('solvePnPRansac', () => {
  it('handles 50 % outliers, 1 px noise and radial distortion (near and far scenes)', () => {
    const rng = new Rng(66);
    const cam = createIntrinsics(1920, 1440, (70 * Math.PI) / 180, -0.1, 0.02);
    for (const [zmin, zmax] of [
      [0.3, 1.0],
      [10, 80],
    ]) {
      for (let k = 0; k < 10; k++) {
        const pose = randomPoseLookingAtOrigin(rng, 0.5 * (zmin + zmax), 0.2);
        const n = 400;
        const X = new Float64Array(3 * n),
          uv = new Float64Array(2 * n);
        const isIn = new Uint8Array(n);
        const pr = new Float64Array(2);
        const tmp = makePoints(rng, pose, n, zmin, zmax);
        X.set(tmp.X);
        for (let i = 0; i < n; i++) {
          expect(projectWorld(cam, pose, X.subarray(3 * i, 3 * i + 3), pr)).toBe(true);
          if (i % 2 === 0) {
            uv[2 * i] = pr[0] + rng.gaussian();
            uv[2 * i + 1] = pr[1] + rng.gaussian();
            isIn[i] = 1;
          } else {
            uv[2 * i] = rng.uniform(0, 1920);
            uv[2 * i + 1] = rng.uniform(0, 1440);
          }
        }
        const res = solvePnPRansac(X, uv, n, cam, { maxErrorPx: 4, rng: new Rng(k) })!;
        expect(res).not.toBeNull();
        expect(rotErrDeg(res.pose.R, pose.R)).toBeLessThan(0.1);
        expect(centerErr(res.pose, pose) / zmax).toBeLessThan(0.005);
        let tp = 0,
          fp = 0;
        for (let i = 0; i < n; i++) {
          if (res.inlierMask[i] && isIn[i]) tp++;
          else if (res.inlierMask[i]) fp++;
        }
        expect(tp).toBeGreaterThan(0.97 * 200);
        expect(fp).toBeLessThan(10);
        expect(res.numInliers).toBe(tp + fp);
      }
    }
  });

  it('returns null without enough inliers', () => {
    const rng = new Rng(67);
    const cam = createIntrinsics(640, 480, 1.2);
    const n = 100;
    const X = new Float64Array(3 * n).map(() => rng.uniform(-1, 1) + 5);
    const uv = new Float64Array(2 * n).map(() => rng.uniform(0, 480));
    expect(solvePnPRansac(X, uv, n, cam, { maxIterations: 500 })).toBeNull();
    expect(solvePnPRansac(X, uv, 5, cam)).toBeNull();
  });
});
