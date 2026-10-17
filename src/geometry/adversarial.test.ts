/**
 * Adversarial regression tests for src/geometry (review pass). Each test names the
 * defect it guards against; tests marked [regression] failed on the pre-review code.
 */
import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, Mat3 } from '../core/types';
import { svd3 } from '../math/decomp';
import { Rng } from '../math/random';
import { median } from '../math/stats';
import { createIntrinsics, distortNormalized, undistortNormalized } from './camera';
import { CHEIRALITY_MAX_DEPTH, cheiralityCount, essential5pt, essentialFromPose, recoverPose } from './essential';
import { decomposeHomography } from './homography';
import { p3p, solvePnPRansac } from './pnp';
import { ransac, ransacIterations, type RansacEstimator } from './ransac';
import { refinePoint } from './triangulation';
import { estimateTwoViewGeometry } from './twoview';
import {
  dirErrDeg,
  makeTwoViewScene,
  matDistUpToScale,
  randomRotation,
  randomUnitVector,
  rotErrDeg,
  toPixels,
  type TwoViewSceneOptions,
} from './testutil';

const CAM = createIntrinsics(1920, 1440, (70 * Math.PI) / 180, -0.08, 0.015);

function unit(t: ArrayLike<number>): Float64Array {
  const n = Math.hypot(t[0], t[1], t[2]);
  return Float64Array.of(t[0] / n, t[1] / n, t[2] / n);
}

/** Keypoints / identity matches of normalised correspondences (σ px noise, optional random outliers appended). */
function toProblem(
  rng: Rng,
  cam: CameraIntrinsics,
  x1: Float64Array,
  x2: Float64Array,
  n: number,
  sigma: number,
  nOut = 0,
): { kp1: Float32Array; kp2: Float32Array; m: Uint32Array } {
  const uv1 = toPixels(rng, cam, x1, n, sigma),
    uv2 = toPixels(rng, cam, x2, n, sigma);
  const N = n + nOut;
  const kp1 = new Float32Array(4 * N),
    kp2 = new Float32Array(4 * N),
    m = new Uint32Array(2 * N);
  for (let i = 0; i < N; i++) {
    if (i < n) {
      kp1[4 * i] = uv1[2 * i];
      kp1[4 * i + 1] = uv1[2 * i + 1];
      kp2[4 * i] = uv2[2 * i];
      kp2[4 * i + 1] = uv2[2 * i + 1];
    } else {
      kp1[4 * i] = rng.uniform(0, cam.width);
      kp1[4 * i + 1] = rng.uniform(0, cam.height);
      kp2[4 * i] = rng.uniform(0, cam.width);
      kp2[4 * i + 1] = rng.uniform(0, cam.height);
    }
    m[2 * i] = m[2 * i + 1] = i;
  }
  return { kp1, kp2, m };
}

/**
 * Points of the plane nᵀX = d seen by camera 1 (identity) and camera 2 (rotation R,
 * centre C2); returns normalised coordinates and t = −R C2 (x2 = R x1 + t).
 */
function planeScene(rng: Rng, n: number, nrm: ArrayLike<number>, d: number, C2: ArrayLike<number>, R: Mat3) {
  const x1 = new Float64Array(2 * n),
    x2 = new Float64Array(2 * n);
  const t = Float64Array.of(
    -(R[0] * C2[0] + R[1] * C2[1] + R[2] * C2[2]),
    -(R[3] * C2[0] + R[4] * C2[1] + R[5] * C2[2]),
    -(R[6] * C2[0] + R[7] * C2[1] + R[8] * C2[2]),
  );
  const hu = 0.7,
    hv = 0.52;
  for (let k = 0, tries = 0; k < n; tries++) {
    if (tries > 1000 * n) throw new Error('planeScene: no valid points');
    const u = hu * (2 * rng.next() - 1),
      v = hv * (2 * rng.next() - 1);
    const den = nrm[0] * u + nrm[1] * v + nrm[2];
    if (!(den > 1e-3)) continue;
    const z = d / den;
    if (!(z > 0 && z < 200)) continue;
    const P = [u * z, v * z, z];
    const Q = [0, 1, 2].map((r) => R[3 * r] * P[0] + R[3 * r + 1] * P[1] + R[3 * r + 2] * P[2] + t[r]);
    if (!(Q[2] > 0.1)) continue;
    const u2 = Q[0] / Q[2],
      v2 = Q[1] / Q[2];
    if (Math.abs(u2) > hu || Math.abs(v2) > hv) continue;
    x1[2 * k] = u;
    x1[2 * k + 1] = v;
    x2[2 * k] = u2;
    x2[2 * k + 1] = v2;
    k++;
  }
  return { x1, x2, t };
}

// ---------------------------------------------------------------------------
// essential.ts
// ---------------------------------------------------------------------------

describe('adversarial: recoverPose', () => {
  it('[regression] all points farther than CHEIRALITY_MAX_DEPTH baselines: correct R and sign of t (was an arbitrary candidate, often the 180°-twisted R)', () => {
    const rng = new Rng(901);
    let bad = 0;
    for (let k = 0; k < 200; k++) {
      // baseline 1, depths 2000–6000 → every finite-depth count is 0
      const sc = makeTwoViewScene(rng, { n: 40, depthMin: 2000, depthMax: 6000, baseline: 1, maxAngle: 0.5 });
      const E = essentialFromPose(sc.R, sc.t);
      const rp = recoverPose(E, sc.x1, sc.x2, sc.n);
      expect(rp.numInFront).toBe(0); // by definition (beyond the depth bound)
      if (rotErrDeg(rp.R, sc.R) > 1e-6 || dirErrDeg(rp.t, sc.t) > 1e-4) bad++;
    }
    expect(bad).toBe(0);
    expect(CHEIRALITY_MAX_DEPTH).toBe(1000);
  });

  it('[regression] exact pure rotation (parallel rays, all counts 0): the rotation is never the twisted one', () => {
    const rng = new Rng(902);
    for (let k = 0; k < 200; k++) {
      const sc = makeTwoViewScene(rng, { n: 30, depthMin: 1, depthMax: 20, baseline: 0, maxAngle: 1.0 });
      const E = essentialFromPose(sc.R, randomUnitVector(rng)); // any t: x2ᵀ[t]×R x1 = 0
      const rp = recoverPose(E, sc.x1, sc.x2, sc.n);
      expect(rotErrDeg(rp.R, sc.R)).toBeLessThan(1e-6);
      expect(Math.hypot(rp.t[0], rp.t[1], rp.t[2])).toBeCloseTo(1, 12);
    }
  });

  it('a mix of near and far points still uses the finite-depth vote (unchanged behaviour)', () => {
    const rng = new Rng(903);
    for (let k = 0; k < 100; k++) {
      const sc = makeTwoViewScene(rng, { n: 50, depthMin: 1, depthMax: 3000, baseline: 1, maxAngle: 1.5 });
      const E = essentialFromPose(sc.R, sc.t);
      const rp = recoverPose(E, sc.x1, sc.x2, sc.n);
      expect(rotErrDeg(rp.R, sc.R)).toBeLessThan(1e-6);
      expect(dirErrDeg(rp.t, sc.t)).toBeLessThan(1e-4);
    }
  });

  it('cheiralityCount with maxDepth = Infinity and with t = 0 is NaN-free', () => {
    const rng = new Rng(904);
    const sc = makeTwoViewScene(rng, { n: 20, depthMin: 1, depthMax: 5, baseline: 0.5, maxAngle: 0.3 });
    expect(cheiralityCount(sc.R, sc.t, sc.x1, sc.x2, sc.n, null, null, Infinity)).toBe(20);
    expect(cheiralityCount(sc.R, [0, 0, 0], sc.x1, sc.x2, sc.n, null, null, Infinity)).toBe(0);
    expect(cheiralityCount(sc.R, [0, 0, 0], sc.x1, sc.x2, sc.n, null, null)).toBe(0);
  });
});

describe('adversarial: essential5pt', () => {
  it('[regression] every returned solution is an essential matrix satisfying the 5 constraints, also for small baselines, planar scenes and forward motion', () => {
    // Before: real eigenvalues whose polish did not converge were returned (σ2/σ1 down
    // to 0.96, σ3/σ1 up to 1e-2) — not solutions at all.
    let total = 0;
    for (const fwd of [false, true])
      for (const planar of [false, true])
        for (const [bmin, bmax] of [
          [1e-2, 1],
          [1e-4, 1e-2],
        ]) {
          const rng = new Rng(1000 + (fwd ? 10 : 0) + (planar ? 20 : 0) + (bmin < 1e-3 ? 1 : 0));
          for (let k = 0; k < 150; k++) {
            const b = bmin * Math.pow(bmax / bmin, rng.next());
            const sc = makeTwoViewScene(rng, {
              n: 5,
              depthMin: 1,
              depthMax: 3,
              baseline: 2 * b,
              maxAngle: 0.5,
              planar,
              centerDir: fwd ? [0.01 * rng.gaussian(), 0.01 * rng.gaussian(), 1] : undefined,
            });
            const sols = essential5pt(sc.x1, sc.x2);
            expect(sols.length).toBeLessThanOrEqual(10);
            for (const E of sols) {
              total++;
              const { S } = svd3(E);
              expect(S[1] / S[0]).toBeGreaterThan(1 - 1e-5);
              expect(S[2] / S[0]).toBeLessThan(1e-5);
              for (let i = 0; i < 5; i++) {
                const a = sc.x1[2 * i],
                  bb = sc.x1[2 * i + 1],
                  c = sc.x2[2 * i],
                  d = sc.x2[2 * i + 1];
                const e = c * (E[0] * a + E[1] * bb + E[2]) + d * (E[3] * a + E[4] * bb + E[5]) + (E[6] * a + E[7] * bb + E[8]);
                expect(Math.abs(e)).toBeLessThan(1e-9);
              }
            }
          }
        }
    expect(total).toBeGreaterThan(2000);
  });

  it('forward motion and planar scenes (baseline/depth ≥ 0.01): true E found in ≥ 99 % (≥ 98 % for forward motion toward a plane)', () => {
    for (const [fwd, planar] of [
      [true, false],
      [false, true],
      [true, true],
    ]) {
      const rng = new Rng(1100 + (fwd ? 1 : 0) + (planar ? 2 : 0));
      let ok = 0;
      const N = 300;
      for (let k = 0; k < N; k++) {
        const b = 0.01 * Math.pow(100, rng.next());
        const sc = makeTwoViewScene(rng, {
          n: 5,
          depthMin: 1,
          depthMax: 3,
          baseline: 2 * b,
          maxAngle: 0.5,
          planar,
          centerDir: fwd ? [0.01 * rng.gaussian(), 0.01 * rng.gaussian(), 1] : undefined,
        });
        const Et = essentialFromPose(sc.R, sc.t);
        if (essential5pt(sc.x1, sc.x2).some((E) => matDistUpToScale(E, Et) < 1e-6)) ok++;
      }
      // forward motion toward a plane is near-degenerate (the two planar solutions merge)
      expect(ok / N).toBeGreaterThanOrEqual(fwd && planar ? 0.98 : 0.99);
    }
  });
});

// ---------------------------------------------------------------------------
// twoview.ts
// ---------------------------------------------------------------------------

describe('adversarial: estimateTwoViewGeometry, planar scenes resolvable by cheirality', () => {
  // Scenes where one branch of the planar two-fold ambiguity puts ~half the points
  // behind a camera. Before the fix the branch closest to the E-RANSAC pose was
  // reported (E cannot tell the branches apart): 'along facade' was wrong in 5/6
  // seeds (5.7° / 90° error), 'ground sideways' in 3/6 (19°).
  const cases: { name: string; nrm: number[]; d: number; C2: number[]; ang: number }[] = [
    { name: 'walking along a fronto-parallel facade (10 m, 1 m step)', nrm: [0, 0, 1], d: 10, C2: [1, 0, 0], ang: 0.05 },
    { name: 'sideways over the ground plane', nrm: [0, -0.97, 0.26], d: 1.5, C2: [0.5, 0, 0], ang: 0.05 },
  ];
  for (const c of cases) {
    it(`[regression] ${c.name}: the reported pose is the physically valid branch`, () => {
      const nrm = unit(c.nrm);
      for (let seed = 0; seed < 6; seed++) {
        const rng = new Rng(3000 + seed);
        const R = randomRotation(rng, c.ang);
        const sc = planeScene(rng, 500, nrm, c.d, c.C2, R);
        const p = toProblem(rng, CAM, sc.x1, sc.x2, 500, 0.7, 150);
        const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.m, CAM, CAM, { rng: new Rng(seed) });
        expect(g.config).toBe('planar');
        expect(rotErrDeg(g.R!, R)).toBeLessThan(0.1);
        expect(dirErrDeg(g.t!, sc.t)).toBeLessThan(1);
        expect(g.triAngle!).toBeGreaterThan((2 * Math.PI) / 180);
        // an implausible branch (points behind a camera) is not offered as the alternative
        if (g.planarAlternative) expect(rotErrDeg(g.planarAlternative.R, R)).toBeLessThan(0.1);
      }
    });
  }

  it('genuinely ambiguous planar pairs still report the other branch as planarAlternative', () => {
    // ground plane, walking forward: both branches have full cheirality support
    const nrm = unit([0, -0.97, 0.26]);
    let mainOk = 0;
    for (let seed = 0; seed < 6; seed++) {
      const rng = new Rng(3100 + seed);
      const R = randomRotation(rng, 0.05);
      const sc = planeScene(rng, 500, nrm, 1.5, [0, 0, 0.5], R);
      const p = toProblem(rng, CAM, sc.x1, sc.x2, 500, 0.7, 100);
      const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.m, CAM, CAM, { rng: new Rng(seed) });
      expect(g.config).toBe('planar');
      const e1 = rotErrDeg(g.R!, R) + 0.1 * dirErrDeg(g.t!, sc.t);
      const e2 = g.planarAlternative ? rotErrDeg(g.planarAlternative.R, R) + 0.1 * dirErrDeg(g.planarAlternative.t, sc.t) : Infinity;
      expect(Math.min(e1, e2)).toBeLessThan(0.1);
      if (e1 < 0.1) mainOk++;
    }
    expect(mainOk).toBeGreaterThan(0);
  });
});

describe('adversarial: estimateTwoViewGeometry, small observed translation classified panoramic', () => {
  it('[regression] the reported R is not biased by the translation (was the rotation-only fit: 0.11–0.28° error)', () => {
    const errs: number[] = [];
    for (let seed = 0; seed < 8; seed++) {
      const rng = new Rng(2000 + seed);
      // 3-D scene 5–15 m, 5 cm baseline: median triangulation angle ≈ 0.08° < 0.5°,
      // but the translation direction is observed (≈ all inliers in front).
      const sc = makeTwoViewScene(rng, { n: 400, depthMin: 5, depthMax: 15, baseline: 0.05, maxAngle: 0.3 });
      const p = toProblem(rng, CAM, sc.x1, sc.x2, sc.n, 0.7);
      const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.m, CAM, CAM, { rng: new Rng(seed) });
      expect(g.config).toBe('panoramic');
      expect(Array.from(g.t!)).toEqual([0, 0, 0]);
      errs.push(rotErrDeg(g.R!, sc.R));
    }
    expect(Math.max(...errs)).toBeLessThan(0.08);
    expect(median(errs)).toBeLessThan(0.05);
  });

  it('pure rotation keeps the (more accurate) rotation-only fit, also noise-free', () => {
    for (const sigma of [0, 0.1, 0.7]) {
      for (let seed = 0; seed < 4; seed++) {
        const rng = new Rng(2100 + seed);
        const sc = makeTwoViewScene(rng, { n: 400, depthMin: 1, depthMax: 50, baseline: 0, maxAngle: 0.4 });
        const p = toProblem(rng, CAM, sc.x1, sc.x2, sc.n, sigma, 150);
        const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.m, CAM, CAM, { rng: new Rng(seed) });
        expect(g.config).toBe('panoramic');
        expect(rotErrDeg(g.R!, sc.R)).toBeLessThan(sigma === 0 ? 1e-3 : 0.02);
      }
    }
  });

  it('all points beyond 1000 baselines (distant houses, 10 cm step) → panoramic with an accurate rotation', () => {
    for (let seed = 0; seed < 4; seed++) {
      const rng = new Rng(2200 + seed);
      const sc = makeTwoViewScene(rng, { n: 400, depthMin: 150, depthMax: 300, baseline: 0.1, maxAngle: 0.3 });
      const p = toProblem(rng, CAM, sc.x1, sc.x2, sc.n, 0.3, 100);
      const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.m, CAM, CAM, { rng: new Rng(seed) });
      expect(g.config).toBe('panoramic');
      expect(rotErrDeg(g.R!, sc.R)).toBeLessThan(0.05);
    }
  });
});

describe('adversarial: estimateTwoViewGeometry, motion types and robustness', () => {
  const motions: { name: string; o: TwoViewSceneOptions }[] = [
    { name: 'forward motion (epipole at the image centre)', o: { n: 400, depthMin: 2, depthMax: 20, baseline: 1, maxAngle: 0.05, centerDir: [0, 0, 1] } },
    { name: 'backward motion', o: { n: 400, depthMin: 2, depthMax: 20, baseline: 1, maxAngle: 0.05, centerDir: [0, 0, -1] } },
    { name: 'large rotation (≈ 70°) with a wide baseline', o: { n: 300, depthMin: 1, depthMax: 3, baseline: 1, maxAngle: 1.2 } },
  ];
  for (const { name, o } of motions) {
    it(`${name}: calibrated, correct sign of t`, () => {
      for (let seed = 0; seed < 4; seed++) {
        const rng = new Rng(4000 + seed);
        const sc = makeTwoViewScene(rng, o);
        const p = toProblem(rng, CAM, sc.x1, sc.x2, sc.n, 0.5, 150);
        const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.m, CAM, CAM, { rng: new Rng(seed) });
        expect(g.config).toBe('calibrated');
        expect(rotErrDeg(g.R!, sc.R)).toBeLessThan(0.1);
        expect(dirErrDeg(g.t!, sc.t)).toBeLessThan(1); // sign included (180° if flipped)
      }
    });
  }

  it('[regression] coincident, collinear and repeated correspondences → "degenerate" (was "calibrated" with an arbitrary pose)', () => {
    const rng = new Rng(4050);
    const N = 100;
    const make = (f: (i: number) => [number, number, number, number]) => {
      const kp1 = new Float32Array(4 * N),
        kp2 = new Float32Array(4 * N),
        m = new Uint32Array(2 * N);
      for (let i = 0; i < N; i++) {
        const [a, b, c, d] = f(i);
        kp1[4 * i] = a;
        kp1[4 * i + 1] = b;
        kp2[4 * i] = c;
        kp2[4 * i + 1] = d;
        m[2 * i] = m[2 * i + 1] = i;
      }
      return { kp1, kp2, m };
    };
    const cases = {
      identical: make(() => [100, 100, 200, 200]),
      tinyCluster: make(() => {
        const a = 960 + 1e-3 * rng.next(),
          b = 720 + 1e-3 * rng.next();
        return [a, b, a + 3, b];
      }),
      collinear: make(() => {
        const a = rng.uniform(0, 1900),
          c = rng.uniform(0, 1900);
        return [a, 0.5 * a + 10, c, 0.3 * c + 100];
      }),
      threeRepeated: make((i) => {
        const j = i % 3;
        return [100 + 300 * j, 50 + 400 * (j & 1), 300 + 250 * j, 70 + 380 * (j & 1)];
      }),
    };
    for (const [name, p] of Object.entries(cases)) {
      for (let seed = 0; seed < 3; seed++) {
        const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.m, CAM, CAM, { rng: new Rng(seed), maxIterations: 500 });
        expect(`${name}: ${g.config}`).toBe(`${name}: degenerate`);
        expect(g.R).toBeUndefined();
      }
    }
    // a valid scene whose points occupy a narrow (≈ 80 px tall, still ≫ threshold) strip is fine
    const sc = makeTwoViewScene(rng, { n: 3000, depthMin: 2, depthMax: 30, baseline: 1, maxAngle: 0.1 });
    const b1: number[] = [],
      b2: number[] = [];
    for (let i = 0; i < sc.n; i++) {
      if (Math.abs(sc.x1[2 * i + 1]) > 0.03) continue;
      b1.push(sc.x1[2 * i], sc.x1[2 * i + 1]);
      b2.push(sc.x2[2 * i], sc.x2[2 * i + 1]);
    }
    const nb = b1.length / 2;
    expect(nb).toBeGreaterThan(60);
    const band = toProblem(rng, CAM, Float64Array.from(b1), Float64Array.from(b2), nb, 0.5, 20);
    const gb = estimateTwoViewGeometry(band.kp1, band.kp2, band.m, CAM, CAM);
    expect(gb.config).toBe('calibrated');
    expect(rotErrDeg(gb.R!, sc.R)).toBeLessThan(0.5);
  });

  it('NaN keypoints are excluded and do not break verification', () => {
    const rng = new Rng(4100);
    const sc = makeTwoViewScene(rng, { n: 300, depthMin: 1, depthMax: 6, baseline: 0.5, maxAngle: 0.3 });
    const p = toProblem(rng, CAM, sc.x1, sc.x2, sc.n, 0.5, 50);
    for (let i = 0; i < 30; i++) p.kp1[4 * (7 * i)] = NaN;
    for (let i = 0; i < 10; i++) p.kp2[4 * (11 * i) + 1] = NaN;
    const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.m, CAM, CAM);
    expect(g.config).toBe('calibrated');
    for (let k = 0; k < g.inliers.length; k += 2) {
      expect(Number.isFinite(p.kp1[4 * g.inliers[k]])).toBe(true);
      expect(Number.isFinite(p.kp2[4 * g.inliers[k + 1] + 1])).toBe(true);
    }
    expect(rotErrDeg(g.R!, sc.R)).toBeLessThan(0.1);
  });

  it('is deterministic (explicit seed and default seed)', () => {
    const rng = new Rng(4200);
    const sc = makeTwoViewScene(rng, { n: 300, depthMin: 1, depthMax: 6, baseline: 0.5, maxAngle: 0.3 });
    const p = toProblem(rng, CAM, sc.x1, sc.x2, sc.n, 0.8, 200);
    const a = estimateTwoViewGeometry(p.kp1, p.kp2, p.m, CAM, CAM);
    const b = estimateTwoViewGeometry(p.kp1, p.kp2, p.m, CAM, CAM);
    const c = estimateTwoViewGeometry(p.kp1, p.kp2, p.m, CAM, CAM, { rng: new Rng(5) });
    const d = estimateTwoViewGeometry(p.kp1, p.kp2, p.m, CAM, CAM, { rng: new Rng(5) });
    for (const [x, y] of [
      [a, b],
      [c, d],
    ]) {
      expect(Array.from(x.R!)).toEqual(Array.from(y.R!));
      expect(Array.from(x.t!)).toEqual(Array.from(y.t!));
      expect(Array.from(x.inliers)).toEqual(Array.from(y.inliers));
      expect([x.numInliersE, x.numInliersF, x.numInliersH]).toEqual([y.numInliersE, y.numInliersF, y.numInliersH]);
    }
  });
});

// ---------------------------------------------------------------------------
// ransac.ts
// ---------------------------------------------------------------------------

/** 1-D "constant" model: data are values, inliers equal the constant exactly. */
class ConstEstimator implements RansacEstimator<number> {
  sampleSize = 2;
  numData: number;
  refines = 0;
  refits = 0;
  constructor(private v: Float64Array) {
    this.numData = v.length;
  }
  isDegenerate(s: Uint32Array): boolean {
    return this.v[s[0]] !== this.v[s[1]];
  }
  fit(s: Uint32Array, models: number[]): number {
    models[0] = this.v[s[0]];
    return 1;
  }
  residualsSq(m: number, out: Float64Array): void {
    for (let i = 0; i < this.numData; i++) out[i] = (this.v[i] - m) ** 2;
  }
  refit(inl: Uint32Array, count: number): number | null {
    this.refits++;
    let s = 0;
    for (let k = 0; k < count; k++) s += this.v[inl[k]];
    return s / count;
  }
  refine(m: number, inl: Uint32Array, count: number): number | null {
    this.refines++;
    return this.refit(inl, count) ?? m;
  }
}

describe('adversarial: ransac', () => {
  it('adaptive stopping: exactly ⌈log(1−p)/log(1−wˢ)⌉ samples once the all-inlier model is found', () => {
    for (const [w, conf] of [
      [0.5, 0.99],
      [0.3, 0.999],
      [0.8, 0.9999],
    ]) {
      const N = 1000;
      const rng = new Rng(5000);
      const v = new Float64Array(N);
      for (let i = 0; i < N; i++) v[i] = i < w * N ? 7 : 100 + 1000 * rng.next(); // distinct outliers
      const res = ransac(new ConstEstimator(v), { thresholdSq: 1e-12, confidence: conf, rng: new Rng(1) });
      expect(res.model).toBe(7);
      expect(res.numInliers).toBe(w * N);
      const k = ransacIterations(w, 2, conf, 10000);
      // ≥ k samples; more only if the first all-inlier sample came later than sample k
      expect(res.iterations).toBeGreaterThanOrEqual(k);
      expect(res.iterations).toBeLessThanOrEqual(Math.max(k, 3 * k));
    }
  });

  it('local optimisation uses `refine` (from the current model) when present, `refit` otherwise', () => {
    const v = Float64Array.from({ length: 200 }, (_, i) => (i < 150 ? 3 : 50 + i));
    const e1 = new ConstEstimator(v);
    ransac(e1, { thresholdSq: 1e-6, rng: new Rng(2) });
    expect(e1.refines).toBeGreaterThan(0);
    const e2 = new ConstEstimator(v);
    (e2 as { refine?: unknown }).refine = undefined;
    ransac(e2, { thresholdSq: 1e-6, rng: new Rng(2) });
    expect(e2.refines).toBe(0);
    expect(e2.refits).toBeGreaterThan(0);
  });

  it('maxIterations = 0 draws nothing; initial models are still scored', () => {
    const v = Float64Array.from({ length: 50 }, (_, i) => (i < 40 ? 1 : i));
    const none = ransac(new ConstEstimator(v), { thresholdSq: 1e-6, maxIterations: 0 });
    expect(none.model).toBeNull();
    expect(none.iterations).toBe(0);
    const seeded = ransac(new ConstEstimator(v), { thresholdSq: 1e-6, maxIterations: 0, initialModels: [1] });
    expect(seeded.model).toBe(1);
    expect(seeded.numInliers).toBe(40);
  });
});

// ---------------------------------------------------------------------------
// triangulation / camera / pnp / homography
// ---------------------------------------------------------------------------

describe('adversarial: misc', () => {
  it('[regression] refinePoint with no cameras / no observations returns a copy instead of throwing', () => {
    const X = Float64Array.of(1, 2, 3);
    const pose = { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: new Float64Array(3) };
    expect(Array.from(refinePoint([pose], [], Float64Array.of(0, 0), 1, X))).toEqual([1, 2, 3]);
    expect(Array.from(refinePoint([], [CAM], new Float64Array(0), 0, X))).toEqual([1, 2, 3]);
    expect(refinePoint([pose], [CAM], Float64Array.of(0, 0), 1, X)).not.toBe(X);
  });

  it('undistortNormalized converges for strong barrel / pincushion distortion up to the turning point', () => {
    const d = new Float64Array(2),
      u = new Float64Array(2);
    const rng = new Rng(6000);
    for (const [k1, k2] of [
      [-0.45, 0],
      [-0.3, 0.05],
      [0.5, 0.5],
      [0.4, -0.1],
      [0, -0.3],
      [-0.49, 0.49],
      [0.3, -0.49],
      [-0.2, -0.2],
    ]) {
      const cam = { ...CAM, k1, k2 };
      // radius of the first turning point of g(r) = r (1 + k1 r² + k2 r⁴)
      let rMax = 3;
      for (let r = 0; r < 3; r += 1e-4) {
        const s = r * r;
        if (1 + 3 * k1 * s + 5 * k2 * s * s <= 0) {
          rMax = r;
          break;
        }
      }
      for (let i = 0; i < 2000; i++) {
        const r = Math.min(1.5, 0.995 * rMax) * Math.sqrt(rng.next());
        const a = 2 * Math.PI * rng.next();
        const x = r * Math.cos(a),
          y = r * Math.sin(a);
        distortNormalized(cam, x, y, d);
        undistortNormalized(cam, d[0], d[1], u);
        expect(Math.hypot(u[0] - x, u[1] - y)).toBeLessThan(1e-9 * Math.max(r, 1e-3));
      }
      // NaN in → NaN out (terminates)
      undistortNormalized(cam, NaN, 0.1, u);
      expect(Number.isNaN(u[0])).toBe(true);
    }
  });

  it('P3P: far, tiny, offset (1e5 m), wide-angle and near-plane-through-centre configurations', () => {
    const rng = new Rng(7000);
    for (const kind of ['far', 'tiny', 'offset', 'wide', 'danger']) {
      let ok = 0;
      const N = 1000;
      for (let k = 0; k < N; k++) {
        const R = randomRotation(rng, Math.PI);
        const P: number[][] = [];
        for (let i = 0; i < 3; i++) {
          if (kind === 'far') P.push([rng.uniform(-1, 1), rng.uniform(-1, 1), 500 + 500 * rng.next()]);
          else if (kind === 'tiny') {
            const z = 1e-3 * (1 + rng.next());
            P.push([rng.uniform(-0.6, 0.6) * z, rng.uniform(-0.45, 0.45) * z, z]);
          } else if (kind === 'wide') {
            const z = 1 + rng.next();
            P.push([rng.uniform(-3, 3) * z, rng.uniform(-3, 3) * z, z]);
          } else {
            const z = 1 + 9 * rng.next();
            P.push([rng.uniform(-0.6, 0.6) * z, rng.uniform(-0.45, 0.45) * z, z]);
          }
        }
        if (kind === 'danger') {
          // the plane of the three points passes 1 mm from the camera centre
          const nrm = randomUnitVector(rng);
          for (let i = 0; i < 3; i++) {
            const dd = P[i][0] * nrm[0] + P[i][1] * nrm[1] + P[i][2] * nrm[2] - 1e-3;
            for (let c = 0; c < 3; c++) P[i][c] -= dd * nrm[c];
            if (P[i][2] < 0.1) P[i][2] = 0.1 + rng.next();
          }
        }
        const off = kind === 'offset' ? [1e5, -1e5, 5e4] : [0, 0, 0];
        const X = new Float64Array(9),
          B = new Float64Array(9);
        for (let i = 0; i < 3; i++) {
          // X = Rᵀ P + C (camera centre C = off)
          for (let c = 0; c < 3; c++) X[3 * i + c] = R[c] * P[i][0] + R[3 + c] * P[i][1] + R[6 + c] * P[i][2] + off[c];
          const nn = Math.hypot(P[i][0], P[i][1], P[i][2]);
          for (let c = 0; c < 3; c++) B[3 * i + c] = P[i][c] / nn;
        }
        const sols = p3p(X, B);
        expect(sols.length).toBeLessThanOrEqual(4);
        if (sols.some((s) => rotErrDeg(s.R, R) < 1e-3)) ok++;
      }
      expect(ok / N).toBeGreaterThanOrEqual(0.999);
    }
  });

  it('solvePnPRansac: 80 % outliers at 1e5 m ENU offsets, near and far scenes', () => {
    const rng = new Rng(7100);
    const cam = createIntrinsics(1920, 1440, (70 * Math.PI) / 180, -0.1, 0.02);
    for (const [zmin, zmax] of [
      [0.3, 1.0],
      [20, 120],
    ]) {
      for (let k = 0; k < 4; k++) {
        const R = randomRotation(rng, Math.PI);
        const C = [1e5 + rng.uniform(-10, 10), -2e5 + rng.uniform(-10, 10), 30 + rng.uniform(-1, 1)];
        const t = Float64Array.of(
          -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
          -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
          -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
        );
        const n = 500;
        const X = new Float64Array(3 * n),
          uv = new Float64Array(2 * n);
        const pc = new Float64Array(3),
          px = new Float64Array(2);
        for (let i = 0; i < n; i++) {
          const z = rng.uniform(zmin, zmax);
          pc[0] = rng.uniform(-0.6, 0.6) * z;
          pc[1] = rng.uniform(-0.45, 0.45) * z;
          pc[2] = z;
          // X = Rᵀ (pc − t)
          for (let c = 0; c < 3; c++) X[3 * i + c] = R[c] * (pc[0] - t[0]) + R[3 + c] * (pc[1] - t[1]) + R[6 + c] * (pc[2] - t[2]);
          if (i % 5 === 0) {
            const xd = new Float64Array(2);
            distortNormalized(cam, pc[0] / z, pc[1] / z, xd);
            px[0] = cam.f * xd[0] + cam.cx + 0.7 * rng.gaussian();
            px[1] = cam.f * xd[1] + cam.cy + 0.7 * rng.gaussian();
          } else {
            px[0] = rng.uniform(0, 1920);
            px[1] = rng.uniform(0, 1440);
          }
          uv[2 * i] = px[0];
          uv[2 * i + 1] = px[1];
        }
        const res = solvePnPRansac(X, uv, n, cam, { maxErrorPx: 4, rng: new Rng(k) });
        expect(res).not.toBeNull();
        expect(rotErrDeg(res!.pose.R, R)).toBeLessThan(0.15);
        const Ce = [0, 1, 2].map((c) => -(res!.pose.R[c] * res!.pose.t[0] + res!.pose.R[3 + c] * res!.pose.t[1] + res!.pose.R[6 + c] * res!.pose.t[2]));
        expect(Math.hypot(Ce[0] - C[0], Ce[1] - C[1], Ce[2] - C[2]) / zmax).toBeLessThan(0.01);
        expect(res!.numInliers).toBeGreaterThan(0.95 * 100);
        expect(res!.numInliers).toBeLessThan(110);
      }
    }
  });

  it('decomposeHomography: translation parallel to the plane normal (walking toward a wall) gives R = I and t = ±T/d', () => {
    const H = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1.3); // R = I, t/d = (0, 0, 0.3), n = e_z
    const sols = decomposeHomography(H);
    expect(sols.length).toBeGreaterThan(0);
    for (const s of sols) {
      expect(rotErrDeg(s.R, Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1))).toBeLessThan(1e-9);
      expect(Math.abs(Math.abs(s.t[2]) - 0.3)).toBeLessThan(1e-12);
      expect(Math.hypot(s.t[0], s.t[1])).toBeLessThan(1e-12);
      expect(Math.abs(s.n[2])).toBeCloseTo(1, 12);
    }
  });
});
