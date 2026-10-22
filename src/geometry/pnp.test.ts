import { describe, expect, it } from 'vitest';
import type { Pose } from '../core/types';
import { Rng } from '../math/random';
import { createIntrinsics, pixelToNormalized, projectWorld } from './camera';
import { epnp, p3p, refinePose, solvePnPKnownRotationRansac, solvePnPRansac, translationKnownRotation } from './pnp';
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

describe('solvePnPKnownRotationRansac (gyro-known rotation)', () => {
  const DEG = Math.PI / 180;
  const cam = createIntrinsics(960, 720, 65 * DEG, -0.03, 0.005);

  /**
   * A keyframe 1 m (the baseline) from a reference camera at the origin, rotated up to 35°, seeing
   * `nIn` points at zmin–zmax m (σ = noisePx) and `nOut` wrong observations (uniform pixels ≥ 40 px
   * from the true projection).
   */
  function keyframe(rng: Rng, nIn: number, nOut: number, noisePx: number, zmin = 3, zmax = 15) {
    const R = randomRotation(rng, 35 * DEG);
    const C = randomUnitVector(rng);
    const t = Float64Array.of(
      -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
      -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
      -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
    );
    const pose: Pose = { R, t };
    const n = nIn + nOut;
    const { X } = makePoints(rng, pose, n, zmin, zmax);
    const uv = new Float64Array(2 * n);
    const pr = new Float64Array(2);
    for (let i = 0; i < n; i++) {
      expect(projectWorld(cam, pose, X.subarray(3 * i, 3 * i + 3), pr)).toBe(true);
      if (i < nIn) {
        uv[2 * i] = pr[0] + noisePx * rng.gaussian();
        uv[2 * i + 1] = pr[1] + noisePx * rng.gaussian();
        continue;
      }
      // a wrong match: anywhere in the image but ≥ 40 px from the true projection
      do {
        uv[2 * i] = rng.uniform(0, cam.width);
        uv[2 * i + 1] = rng.uniform(0, cam.height);
      } while (Math.hypot(uv[2 * i] - pr[0], uv[2 * i + 1] - pr[1]) < 40);
    }
    return { pose, X, uv, n };
  }

  /** `R` turned by `deg` about a random axis (the gyro's error). */
  function perturbed(rng: Rng, R: ArrayLike<number>, deg: number): Float64Array {
    const E = randomRotation(rng, deg * DEG, deg * DEG);
    const out = new Float64Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) out[3 * i + j] = E[3 * i] * R[j] + E[3 * i + 1] * R[3 + j] + E[3 * i + 2] * R[6 + j];
    return out;
  }

  it('recovers t exactly (1e-9) from 2 noise-free points', () => {
    const rng = new Rng(201);
    const pin = createIntrinsics(960, 720, 65 * DEG);
    for (let k = 0; k < 200; k++) {
      const pose = randomPoseLookingAtOrigin(rng, rng.uniform(0.5, 20), 0.3);
      const { X, xn } = makePoints(rng, pose, 2, 0.3, 40);
      const t = translationKnownRotation(pose.R, X, xn, 2)!;
      expect(t).not.toBeNull();
      const tol = 1e-9 * Math.max(1, Math.hypot(pose.t[0], pose.t[1], pose.t[2]));
      for (let c = 0; c < 3; c++) expect(Math.abs(t[c] - pose.t[c])).toBeLessThan(tol);
      // through the pixel path and RANSAC (2 points: the sample itself)
      const uv = new Float64Array(4);
      const pr = new Float64Array(2);
      for (let i = 0; i < 2; i++) {
        projectWorld(pin, pose, X.subarray(3 * i, 3 * i + 3), pr);
        uv.set(pr, 2 * i);
      }
      const res = solvePnPKnownRotationRansac(X, uv, 2, pin, pose.R, { minInliers: 2, rng: new Rng(k) })!;
      expect(res).not.toBeNull();
      expect(res.numInliers).toBe(2);
      for (let c = 0; c < 3; c++) expect(Math.abs(res.pose.t[c] - pose.t[c])).toBeLessThan(tol);
      expect(res.rotationChangeRad).toBe(0); // < 3 points: refinePose leaves the pose as it is
    }
    // degenerate: coinciding bearings
    const X2 = Float64Array.of(0, 0, 5, 0, 0, 10);
    expect(translationKnownRotation(Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), X2, Float64Array.of(0, 0, 0, 0), 2)).toBeNull();
  });

  it('12 inliers among 70 % outliers, 0.5 px noise, rotation 0 / 1° off: the optimal pose (0.5° / 1 % of the baseline) in ≥ 99 % of 200 trials', () => {
    const huber = 6 / cam.f;
    const xn = new Float64Array(24);
    const tmp = new Float64Array(2);
    for (const [gyroErrDeg, zmin, zmax] of [
      [0, 2, 8],
      [1, 2, 8],
      [1, 5, 40], // yard-like: depth 5–40 × the baseline
    ]) {
      const rng = new Rng(202 + gyroErrDeg + zmax);
      let ok = 0,
        withinBaseline = 0;
      const trials = 200;
      for (let k = 0; k < trials; k++) {
        const s = keyframe(rng, 12, 28, 0.5, zmin, zmax);
        const res = solvePnPKnownRotationRansac(s.X, s.uv, s.n, cam, perturbed(rng, s.pose.R, gyroErrDeg), { maxErrorPx: 12, rng: new Rng(1000 + k) });
        if (!res) continue;
        // what any estimator reaches from these 12 noisy points: the robust LM optimum from the truth
        for (let i = 0; i < 12; i++) {
          pixelToNormalized(cam, s.uv[2 * i], s.uv[2 * i + 1], tmp);
          xn.set(tmp, 2 * i);
        }
        const best = refinePose(s.pose, s.X, xn, 12, { iterations: 50, huberNormalized: huber });
        let wrong = 0;
        for (let i = 12; i < s.n; i++) wrong += res.inlierMask[i];
        const found = res.numInliers === 12 && wrong === 0 && rotErrDeg(res.pose.R, s.pose.R) < 0.5 && centerErr(res.pose, best) < 0.01;
        if (found) ok++;
        if (found && centerErr(res.pose, s.pose) < 0.01) withinBaseline++;
        if (found) expect(Math.abs(res.rotationChangeRad / DEG - gyroErrDeg)).toBeLessThan(0.5);
      }
      expect(ok).toBeGreaterThanOrEqual(0.99 * trials);
      // and that optimum within 1 % of the 1 m baseline at 2–8 m (its centre error: p50 0.35 cm, p95
      // 0.75 cm; at 5–40 m p50 1.1 cm, p95 2.8 cm — the rotation ↔ translation coupling of 12 points)
      if (zmax === 8) expect(withinBaseline).toBeGreaterThanOrEqual(0.99 * trials);
    }
  });

  it('refineRotation: false keeps the given rotation exactly; seeded Rng ⇒ identical results', () => {
    const rng = new Rng(204);
    const s = keyframe(rng, 30, 30, 0.5);
    const fixed = solvePnPKnownRotationRansac(s.X, s.uv, s.n, cam, s.pose.R, { refineRotation: false, rng: new Rng(7) })!;
    expect(fixed).not.toBeNull();
    expect(Array.from(fixed.pose.R)).toEqual(Array.from(s.pose.R));
    expect(fixed.rotationChangeRad).toBe(0);
    expect(centerErr(fixed.pose, s.pose)).toBeLessThan(0.01);
    const a = solvePnPKnownRotationRansac(s.X, s.uv, s.n, cam, s.pose.R, { rng: new Rng(9) })!;
    const b = solvePnPKnownRotationRansac(s.X, s.uv, s.n, cam, s.pose.R, { rng: new Rng(9) })!;
    expect(Array.from(a.pose.t)).toEqual(Array.from(b.pose.t));
    expect(Array.from(a.inlierMask)).toEqual(Array.from(b.inlierMask));
  });

  it('returns null without enough inliers (and for a rotation far off)', () => {
    const rng = new Rng(205);
    const n = 100;
    const X = new Float64Array(3 * n).map(() => rng.uniform(-1, 1) + 5);
    const uv = new Float64Array(2 * n).map(() => rng.uniform(0, 700));
    expect(solvePnPKnownRotationRansac(X, uv, n, cam, Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), { maxIterations: 500 })).toBeNull();
    const s = keyframe(rng, 14, 0, 0.5);
    expect(solvePnPKnownRotationRansac(s.X, s.uv, 11, cam, s.pose.R)).toBeNull(); // n < minInliers
    // a prediction 25° off: no consistent translation explains 12 of the points
    expect(solvePnPKnownRotationRansac(s.X, s.uv, s.n, cam, perturbed(rng, s.pose.R, 25), { maxErrorPx: 12 })).toBeNull();
  });
});
