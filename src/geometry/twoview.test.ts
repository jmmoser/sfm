import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, Mat3 } from '../core/types';
import { Rng } from '../math/random';
import { median } from '../math/stats';
import { createIntrinsics } from './camera';
import { estimateTwoViewGeometry, rotationFromCorrespondences } from './twoview';
import {
  dirErrDeg,
  makeTwoViewScene,
  randomRotation,
  rotErrDeg,
  toPixels,
  type TwoViewScene,
  type TwoViewSceneOptions,
} from './testutil';

interface Problem {
  kp1: Float32Array;
  kp2: Float32Array;
  matches: Uint32Array;
  /** True for match k iff it is a true correspondence. */
  isInlier: Uint8Array;
  sc: TwoViewScene;
}

/**
 * Keypoints (distorted pixels, σ px noise) of a synthetic scene plus uniformly random
 * outlier matches; feature indices are shuffled in image 2 and matches shuffled.
 */
function makeProblem(rng: Rng, cam: CameraIntrinsics, o: TwoViewSceneOptions, sigma: number, outlierFrac: number): Problem {
  const sc = makeTwoViewScene(rng, o);
  const n = sc.n;
  const nOut = Math.round((n * outlierFrac) / (1 - outlierFrac));
  const N = n + nOut;
  const uv1 = toPixels(rng, cam, sc.x1, n, sigma),
    uv2 = toPixels(rng, cam, sc.x2, n, sigma);
  const kp1 = new Float32Array(4 * N),
    kp2 = new Float32Array(4 * N);
  const perm2 = new Uint32Array(N);
  for (let i = 0; i < N; i++) perm2[i] = i;
  rng.shuffle(perm2); // feature i of image 1 ↔ feature perm2[i] of image 2
  for (let i = 0; i < N; i++) {
    const j = perm2[i];
    if (i < n) {
      kp1[4 * i] = uv1[2 * i];
      kp1[4 * i + 1] = uv1[2 * i + 1];
      kp2[4 * j] = uv2[2 * i];
      kp2[4 * j + 1] = uv2[2 * i + 1];
    } else {
      kp1[4 * i] = rng.uniform(0, cam.width);
      kp1[4 * i + 1] = rng.uniform(0, cam.height);
      kp2[4 * j] = rng.uniform(0, cam.width);
      kp2[4 * j + 1] = rng.uniform(0, cam.height);
    }
    kp1[4 * i + 2] = kp2[4 * j + 2] = 2;
  }
  const order = new Uint32Array(N);
  for (let i = 0; i < N; i++) order[i] = i;
  rng.shuffle(order);
  const matches = new Uint32Array(2 * N);
  const isInlier = new Uint8Array(N);
  for (let k = 0; k < N; k++) {
    matches[2 * k] = order[k];
    matches[2 * k + 1] = perm2[order[k]];
    isInlier[k] = order[k] < n ? 1 : 0;
  }
  return { kp1, kp2, matches, isInlier, sc };
}

function inlierStats(p: Problem, inliers: Uint32Array): { precision: number; recall: number } {
  const n = p.sc.n;
  let tp = 0;
  for (let k = 0; k < inliers.length; k += 2) if (inliers[k] < n) tp++;
  return { precision: tp / Math.max(1, inliers.length / 2), recall: tp / n };
}

function mulR(A: ArrayLike<number>, B: ArrayLike<number>): Mat3 {
  const R = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) R[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
  return R;
}

const CAM = createIntrinsics(1920, 1440, (70 * Math.PI) / 180, -0.08, 0.015);

describe('estimateTwoViewGeometry: calibrated scenes', () => {
  const scenarios: { name: string; o: Omit<TwoViewSceneOptions, 'n'> }[] = [
    { name: 'object 0.5 m, wide baseline', o: { depthMin: 0.3, depthMax: 0.8, baseline: 0.15, maxAngle: 0.4 } },
    { name: 'object 0.5 m, narrow baseline', o: { depthMin: 0.3, depthMax: 0.8, baseline: 0.03, maxAngle: 0.1 } },
    { name: 'scene 50 m, wide baseline', o: { depthMin: 20, depthMax: 80, baseline: 5, maxAngle: 0.3 } },
    { name: 'scene 50 m, narrow baseline', o: { depthMin: 20, depthMax: 80, baseline: 1, maxAngle: 0.1 } },
    { name: 'backyard 3–100 m', o: { depthMin: 3, depthMax: 100, baseline: 1, maxAngle: 0.3 } },
  ];
  for (const { name, o } of scenarios) {
    it(`${name}: N(0, 0.5–1 px), 30–50 % outliers, distortion → rot < 0.3°, t < 2°`, () => {
      const rot: number[] = [],
        tr: number[] = [];
      for (let seed = 0; seed < 6; seed++) {
        const rng = new Rng(100 * seed + name.length);
        const sigma = rng.uniform(0.5, 1.0),
          outl = rng.uniform(0.3, 0.5);
        const p = makeProblem(rng, CAM, { n: 500 + rng.int(700), ...o }, sigma, outl);
        const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.matches, CAM, CAM, { rng: new Rng(seed) });
        expect(g.config).toBe('calibrated');
        expect(g.E && g.F && g.R && g.t && g.triAngle !== undefined).toBeTruthy();
        expect(Math.hypot(g.t![0], g.t![1], g.t![2])).toBeCloseTo(1, 12);
        rot.push(rotErrDeg(g.R!, p.sc.R));
        tr.push(dirErrDeg(g.t!, p.sc.t));
        const st = inlierStats(p, g.inliers);
        expect(st.recall).toBeGreaterThan(0.97);
        expect(st.precision).toBeGreaterThan(0.97);
        expect(g.numInliersE).toBeGreaterThanOrEqual(0.95 * g.numInliersF);
      }
      console.log(
        `[twoview] ${name}: rot med ${median(rot).toFixed(3)}° max ${Math.max(...rot).toFixed(3)}°, ` +
          `t med ${median(tr).toFixed(3)}° max ${Math.max(...tr).toFixed(3)}°`,
      );
      expect(Math.max(...rot)).toBeLessThan(0.3);
      expect(Math.max(...tr)).toBeLessThan(2);
    });
  }

  it('inliers are a subset of the input matches (pairs preserved)', () => {
    const rng = new Rng(7);
    const p = makeProblem(rng, CAM, { n: 400, depthMin: 1, depthMax: 5, baseline: 0.5, maxAngle: 0.3 }, 0.7, 0.3);
    const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.matches, CAM, CAM);
    const pairs = new Set<string>();
    for (let k = 0; k < p.matches.length; k += 2) pairs.add(`${p.matches[k]},${p.matches[k + 1]}`);
    for (let k = 0; k < g.inliers.length; k += 2) expect(pairs.has(`${g.inliers[k]},${g.inliers[k + 1]}`)).toBe(true);
  });

  it('two cameras with different intrinsics', () => {
    const rng = new Rng(8);
    const cam2 = { ...createIntrinsics(1440, 1080, 1.0, 0.03, 0), cx: 700, cy: 560 };
    const sc = makeTwoViewScene(rng, { n: 600, depthMin: 1, depthMax: 6, baseline: 0.5, maxAngle: 0.3, halfFovTan: 0.45 });
    const uv1 = toPixels(rng, CAM, sc.x1, 600, 0.7),
      uv2 = toPixels(rng, cam2, sc.x2, 600, 0.7);
    const kp1 = new Float32Array(2400),
      kp2 = new Float32Array(2400);
    const m = new Uint32Array(1200);
    for (let i = 0; i < 600; i++) {
      kp1[4 * i] = uv1[2 * i];
      kp1[4 * i + 1] = uv1[2 * i + 1];
      kp2[4 * i] = uv2[2 * i];
      kp2[4 * i + 1] = uv2[2 * i + 1];
      m[2 * i] = m[2 * i + 1] = i;
    }
    const g = estimateTwoViewGeometry(kp1, kp2, m, CAM, cam2);
    expect(g.config).toBe('calibrated');
    expect(rotErrDeg(g.R!, sc.R)).toBeLessThan(0.1);
    expect(dirErrDeg(g.t!, sc.t)).toBeLessThan(1);
  });
});

describe('estimateTwoViewGeometry: planar, panoramic, uncalibrated and degenerate pairs', () => {
  it('planar scene → "planar"; the true pose is R or the planar alternative', () => {
    for (let seed = 0; seed < 6; seed++) {
      const rng = new Rng(2000 + seed);
      const p = makeProblem(rng, CAM, { n: 800, depthMin: 2, depthMax: 4, baseline: 0.6, maxAngle: 0.3, planar: true }, 0.8, 0.4);
      const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.matches, CAM, CAM);
      expect(g.config).toBe('planar');
      expect(g.H).toBeDefined();
      expect(g.numInliersH).toBeGreaterThan(0.8 * g.numInliersE);
      const e1 = rotErrDeg(g.R!, p.sc.R) + 0.1 * dirErrDeg(g.t!, p.sc.t);
      const e2 = g.planarAlternative
        ? rotErrDeg(g.planarAlternative.R, p.sc.R) + 0.1 * dirErrDeg(g.planarAlternative.t, p.sc.t)
        : Infinity;
      expect(Math.min(e1, e2)).toBeLessThan(0.2);
      expect(g.triAngle!).toBeGreaterThan((0.5 * Math.PI) / 180);
    }
  });

  it('pure rotation → "panoramic" with an accurate rotation-only R and t = 0', () => {
    for (let seed = 0; seed < 5; seed++) {
      const rng = new Rng(3000 + seed);
      const p = makeProblem(rng, CAM, { n: 800, depthMin: 1, depthMax: 50, baseline: 0, maxAngle: 0.4 }, 0.8, 0.4);
      const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.matches, CAM, CAM);
      expect(g.config).toBe('panoramic');
      expect(rotErrDeg(g.R!, p.sc.R)).toBeLessThan(0.03);
      expect(Array.from(g.t!)).toEqual([0, 0, 0]);
      expect(g.triAngle!).toBeLessThan((0.1 * Math.PI) / 180);
      const st = inlierStats(p, g.inliers);
      expect(st.recall).toBeGreaterThan(0.97);
      expect(st.precision).toBeGreaterThan(0.99);
    }
  });

  it('a tiny translation (median triangulation angle < 0.5°) is also "panoramic"', () => {
    const rng = new Rng(4000);
    const p = makeProblem(rng, CAM, { n: 800, depthMin: 5, depthMax: 50, baseline: 0.02, maxAngle: 0.3 }, 0.8, 0.4);
    const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.matches, CAM, CAM);
    expect(g.config).toBe('panoramic');
    expect(rotErrDeg(g.R!, p.sc.R)).toBeLessThan(0.1);
  });

  it('rotation prior (1–1.5° gyro error) helps detect pure rotation and never changes calibrated results', () => {
    const rng = new Rng(5000);
    // pure rotation with a perturbed prior
    const p = makeProblem(rng, CAM, { n: 600, depthMin: 1, depthMax: 50, baseline: 0, maxAngle: 0.4 }, 0.8, 0.5);
    const prior = mulR(randomRotation(rng, (1.5 * Math.PI) / 180, Math.PI / 180), p.sc.R);
    const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.matches, CAM, CAM, { rotationPrior: prior });
    expect(g.config).toBe('panoramic');
    expect(rotErrDeg(g.R!, p.sc.R)).toBeLessThan(0.03);
    // a prior that is wildly off does not break the result
    const bad = estimateTwoViewGeometry(p.kp1, p.kp2, p.matches, CAM, CAM, { rotationPrior: randomRotation(rng, 3, 2) });
    expect(bad.config).toBe('panoramic');
    // general motion: identical result with and without the prior
    const q = makeProblem(rng, CAM, { n: 600, depthMin: 1, depthMax: 6, baseline: 0.5, maxAngle: 0.3 }, 0.8, 0.4);
    const a = estimateTwoViewGeometry(q.kp1, q.kp2, q.matches, CAM, CAM, { rng: new Rng(1) });
    const b = estimateTwoViewGeometry(q.kp1, q.kp2, q.matches, CAM, CAM, { rng: new Rng(1), rotationPrior: q.sc.R });
    expect(a.config).toBe('calibrated');
    expect(b.config).toBe('calibrated');
    expect(Array.from(b.R!)).toEqual(Array.from(a.R!));
    expect(Array.from(b.t!)).toEqual(Array.from(a.t!));
  });

  it('wrong intrinsics → "uncalibrated" whenever E explains < 95 % of the F inliers', () => {
    let uncal = 0;
    for (let seed = 0; seed < 6; seed++) {
      const rng = new Rng(6000 + seed);
      const p = makeProblem(rng, CAM, { n: 800, depthMin: 1, depthMax: 6, baseline: 0.6, maxAngle: 0.3 }, 0.8, 0.3);
      const wrong = { ...CAM, f: CAM.f * 1.5 };
      const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.matches, wrong, wrong);
      const ratio = g.numInliersE / g.numInliersF;
      if (ratio < 0.95) {
        expect(g.config).toBe('uncalibrated');
        expect(g.R).toBeUndefined();
        uncal++;
      } else expect(g.config).toBe('calibrated');
      const st = inlierStats(p, g.inliers);
      expect(st.recall).toBeGreaterThan(0.95);
    }
    expect(uncal).toBeGreaterThanOrEqual(4);
  });

  it('random matches and too few matches → "degenerate"', () => {
    const rng = new Rng(7000);
    const N = 400;
    const kp1 = new Float32Array(4 * N).map(() => rng.uniform(0, 1400));
    const kp2 = new Float32Array(4 * N).map(() => rng.uniform(0, 1400));
    const m = new Uint32Array(2 * N);
    for (let k = 0; k < N; k++) m[2 * k] = m[2 * k + 1] = k;
    const g = estimateTwoViewGeometry(kp1, kp2, m, CAM, CAM);
    expect(g.config).toBe('degenerate');
    expect(g.inliers.length).toBe(0);
    const few = estimateTwoViewGeometry(kp1, kp2, m.subarray(0, 20), CAM, CAM);
    expect(few.config).toBe('degenerate');
    const none = estimateTwoViewGeometry(kp1, kp2, new Uint32Array(0), CAM, CAM);
    expect(none.config).toBe('degenerate');
    // a valid geometry whose inlier ratio is below minInlierRatio
    const p = makeProblem(rng, CAM, { n: 100, depthMin: 1, depthMax: 6, baseline: 0.6, maxAngle: 0.3 }, 0.5, 0.85);
    const low = estimateTwoViewGeometry(p.kp1, p.kp2, p.matches, CAM, CAM, { minInlierRatio: 0.25 });
    expect(low.config).toBe('degenerate');
  });

  it('rotationFromCorrespondences is the Kabsch fit on bearings', () => {
    const rng = new Rng(8000);
    const sc = makeTwoViewScene(rng, { n: 50, depthMin: 1, depthMax: 9, baseline: 0, maxAngle: 0.7 });
    expect(rotErrDeg(rotationFromCorrespondences(sc.x1, sc.x2, 50, null)!, sc.R)).toBeLessThan(1e-9);
    expect(rotationFromCorrespondences(sc.x1, sc.x2, 1, null)).toBeNull();
  });
});

describe('estimateTwoViewGeometry: timing', () => {
  it('2000 matches with 40 % outliers (log only; well under 200 ms in Node)', () => {
    const rng = new Rng(9000);
    const p = makeProblem(rng, CAM, { n: 1200, depthMin: 1, depthMax: 10, baseline: 0.5, maxAngle: 0.3 }, 0.8, 0.4);
    expect(p.matches.length / 2).toBe(2000);
    estimateTwoViewGeometry(p.kp1, p.kp2, p.matches, CAM, CAM); // warm-up (JIT)
    const times: number[] = [];
    for (let rep = 0; rep < 5; rep++) {
      const t0 = performance.now();
      const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.matches, CAM, CAM, { rng: new Rng(rep) });
      times.push(performance.now() - t0);
      expect(g.config).toBe('calibrated');
    }
    console.log(`[twoview] 2000 matches, 40 % outliers: median ${median(times).toFixed(1)} ms (min ${Math.min(...times).toFixed(1)}, max ${Math.max(...times).toFixed(1)})`);
    expect(median(times)).toBeLessThan(200);
  });
});
