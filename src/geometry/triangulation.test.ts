import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, Pose } from '../core/types';
import { Rng } from '../math/random';
import { createIntrinsics, projectWorld } from './camera';
import { poseCenter, poseFromCenter } from './pose';
import { pointDepth, refinePoint, triangulateDLT, triangulateTwoView, triangulationAngle } from './triangulation';
import { randomRotation, randomUnitVector } from './testutil';

/** Camera at C looking at `target` (OpenCV frame) with a small random roll. */
function lookAt(rng: Rng, C: ArrayLike<number>, target: ArrayLike<number>): Pose {
  const z = [target[0] - C[0], target[1] - C[1], target[2] - C[2]];
  const zn = Math.hypot(z[0], z[1], z[2]);
  for (let i = 0; i < 3; i++) z[i] /= zn;
  const up = Math.abs(z[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  let x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]];
  const xn = Math.hypot(x[0], x[1], x[2]);
  x = x.map((v) => v / xn);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const R0 = Float64Array.of(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]);
  const J = randomRotation(rng, 0.05);
  const R = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) R[3 * i + j] = J[3 * i] * R0[j] + J[3 * i + 1] * R0[3 + j] + J[3 * i + 2] * R0[6 + j];
  return poseFromCenter(R, C);
}

function project(pose: Pose, X: ArrayLike<number>): [number, number] {
  const R = pose.R,
    t = pose.t;
  const P = [0, 1, 2].map((r) => R[3 * r] * X[0] + R[3 * r + 1] * X[1] + R[3 * r + 2] * X[2] + t[r]);
  return [P[0] / P[2], P[1] / P[2]];
}

describe('triangulateDLT', () => {
  it('is exact for 2–6 views, near and far points, and at large ENU offsets', () => {
    const rng = new Rng(71);
    for (const [dist, baseline, offset] of [
      [0.5, 0.1, 0],
      [50, 3, 0],
      [50, 3, 3e5],
      [5, 1, 1e6],
    ]) {
      let worst = 0;
      for (let k = 0; k < 100; k++) {
        const nv = 2 + (k % 5);
        const X = Float64Array.of(offset + rng.gaussian(), offset * 0.7 + rng.gaussian(), rng.gaussian() + 0.1 * offset);
        const poses: Pose[] = [];
        const xn = new Float64Array(2 * nv);
        const dir = randomUnitVector(rng);
        for (let v = 0; v < nv; v++) {
          const C = [0, 1, 2].map((i) => X[i] - dist * dir[i] + baseline * rng.gaussian());
          const p = lookAt(rng, C, X);
          poses.push(p);
          [xn[2 * v], xn[2 * v + 1]] = project(p, X);
        }
        const Y = triangulateDLT(poses, xn, nv)!;
        expect(Y).not.toBeNull();
        worst = Math.max(worst, Math.hypot(Y[0] - X[0], Y[1] - X[1], Y[2] - X[2]) / dist);
      }
      expect(worst).toBeLessThan(offset > 0 ? 1e-7 : 1e-10);
    }
  });

  it('returns null at infinity (identical rays) and for too few views', () => {
    const p1: Pose = { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(0, 0, 0) };
    const p2: Pose = { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(-1, 0, 0) };
    // parallel rays along +z from both centres → point at infinity
    expect(triangulateTwoView(p1, p2, [0, 0], [0, 0])).toBeNull();
    // zero baseline (pure rotation)
    expect(triangulateTwoView(p1, p1, [0.1, 0.2], [0.1, 0.2])).toBeNull();
    expect(triangulateDLT([p1], [0, 0], 1)).toBeNull();
    // a regular two-view point
    const X = triangulateTwoView(p1, p2, [0.1, 0.05], [(0.1 * 4 - 1) / 4, 0.05])!;
    expect(X[0]).toBeCloseTo(0.4, 10);
    expect(X[1]).toBeCloseTo(0.2, 10);
    expect(X[2]).toBeCloseTo(4, 10);
  });
});

describe('refinePoint', () => {
  it('minimises the pixel reprojection error with RADIAL distortion', () => {
    const rng = new Rng(72);
    const cams: CameraIntrinsics[] = [
      createIntrinsics(1920, 1440, 1.2, -0.12, 0.03),
      createIntrinsics(1920, 1440, 1.1, 0.05, -0.01),
      createIntrinsics(1920, 1440, 1.0, -0.2, 0.05),
    ];
    let errDLT = 0,
      errRef = 0;
    for (let k = 0; k < 100; k++) {
      const X = Float64Array.of(rng.gaussian(), rng.gaussian(), rng.gaussian());
      const poses: Pose[] = [];
      const uv = new Float64Array(6),
        xn = new Float64Array(6),
        pr = new Float64Array(2);
      for (let v = 0; v < 3; v++) {
        const C = [X[0] + rng.uniform(-1, 1), X[1] + rng.uniform(-1, 1), X[2] - 4 + rng.uniform(-0.5, 0.5)];
        // aim off-axis so that X lands in the strongly distorted image periphery
        const off = () => (rng.next() < 0.5 ? -1 : 1) * rng.uniform(0.8, 1.4);
        const p = lookAt(rng, C, [X[0] + off(), X[1] + off(), X[2]]);
        poses.push(p);
        expect(projectWorld(cams[v], p, X, pr)).toBe(true);
        uv[2 * v] = pr[0] + 0.7 * rng.gaussian();
        uv[2 * v + 1] = pr[1] + 0.7 * rng.gaussian();
        [xn[2 * v], xn[2 * v + 1]] = project(p, X); // noise-free normalised for the noise-free check
      }
      // noise-free: from a perturbed start back to the exact point
      const uv0 = new Float64Array(6);
      for (let v = 0; v < 3; v++) {
        projectWorld(cams[v], poses[v], X, pr);
        uv0[2 * v] = pr[0];
        uv0[2 * v + 1] = pr[1];
      }
      const Xr = refinePoint(poses, cams, uv0, 3, [X[0] + 0.05, X[1] - 0.05, X[2] + 0.1], 20);
      expect(Math.hypot(Xr[0] - X[0], Xr[1] - X[1], Xr[2] - X[2])).toBeLessThan(1e-9);
      // noisy: DLT on (wrongly) undistorted-as-pinhole normalised coordinates vs refinement
      const xd = new Float64Array(6);
      for (let v = 0; v < 3; v++) {
        xd[2 * v] = (uv[2 * v] - cams[v].cx) / cams[v].f;
        xd[2 * v + 1] = (uv[2 * v + 1] - cams[v].cy) / cams[v].f;
      }
      const Xd = triangulateDLT(poses, xd, 3)!;
      const Xn = refinePoint(poses, cams, uv, 3, Xd);
      errDLT += Math.hypot(Xd[0] - X[0], Xd[1] - X[1], Xd[2] - X[2]);
      errRef += Math.hypot(Xn[0] - X[0], Xn[1] - X[1], Xn[2] - X[2]);
    }
    expect(errRef).toBeLessThan(0.3 * errDLT);
    expect(errRef / 100).toBeLessThan(0.01);
  });

  it('returns the input when it is behind a camera', () => {
    const cam = createIntrinsics(640, 480, 1);
    const p: Pose = { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(0, 0, 0) };
    const X = refinePoint([p, p], [cam], [320, 240, 320, 240], 2, [0, 0, -1]);
    expect(Array.from(X)).toEqual([0, 0, -1]);
  });
});

describe('triangulationAngle / pointDepth', () => {
  it('computes ray angles and depths', () => {
    expect(triangulationAngle([0, 0, 0], [2, 0, 0], [1, 0, 1])).toBeCloseTo(Math.PI / 2, 14);
    expect(triangulationAngle([0, 0, 0], [0, 0, 0], [1, 2, 3])).toBe(0);
    expect(triangulationAngle([0, 0, 0], [1e-9, 0, 0], [0, 0, 1e3])).toBeCloseTo(1e-12, 20);
    const rng = new Rng(73);
    const p = lookAt(rng, [1, 2, 3], [0, 0, 0]);
    const C = poseCenter(p);
    expect(pointDepth(p, C)).toBeCloseTo(0, 12);
    expect(pointDepth(p, [0, 0, 0])).toBeCloseTo(Math.hypot(1, 2, 3) * Math.cos(Math.asin(0)), 1);
  });
});
