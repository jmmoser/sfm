import { describe, expect, it } from 'vitest';
import { mat3Mul, mat3MulTranspose } from '../math/mat3';
import { Rng } from '../math/random';
import { rotationAngleBetween, rotvecToMat3 } from '../math/rotation';
import type { BALossType, BAProblem } from './problem';
import { bundleAdjust, computeBAGradient, evaluateBACost, reprojectionErrors } from './solver';
import {
  cloneBAProblem,
  fixGaugeTwoPoses,
  makeSyntheticBA,
  perturbBA,
  poseCentre,
  syntheticErrors,
  type SyntheticBA,
} from './synthetic';

const LOSSES: BALossType[] = ['trivial', 'huber', 'cauchy', 'soft_l1'];

/** Expected RMS (px, 2-D error norm) of LS residuals at the optimum for per-axis noise σ. */
function expectedRms(sigma: number, nObs: number, nParams: number): number {
  return sigma * Math.sqrt((2 * nObs - nParams) / nObs);
}

/** Small problem exercising every parameter type, mask, weight and prior. */
function richProblem(seed = 5): SyntheticBA {
  const s = makeSyntheticBA({ numPoses: 7, numPoints: 50, layout: 'orbit', trackWindow: 3, k1: -0.1, k2: 0.02, numCameras: 2, noisePx: 3, seed });
  const p = s.problem;
  p.poseFixed[0] = 1;
  p.translationFixedMask = new Uint8Array(p.poses.length);
  p.translationFixedMask[2] = 0b101;
  p.translationFixedMask[4] = 0b010;
  p.pointFixed = new Uint8Array(p.points.length / 3);
  p.pointFixed[3] = 1;
  p.pointFixed[10] = 1;
  p.cameraRefine[0] = { focal: true, principalPoint: true, distortion: true };
  p.cameraRefine[1] = { focal: true, principalPoint: false, distortion: true };
  const rng = new Rng(seed);
  p.obsWeight = new Float32Array(p.obsPose.length).map(() => 0.5 + 1.5 * rng.next());
  p.centerPriors = [
    { pose: 1, center: Float64Array.of(0.3, -0.2, 0.1).map((v, k) => v + poseCentre(p.poses[1])[k]), sigma: 0.2 },
    { pose: 2, center: poseCentre(p.poses[2]).map((v) => v + 0.05), sigma: 0.05 },
    { pose: 0, center: Float64Array.of(1, 2, 3), sigma: 1 }, // fixed pose: constant cost
  ];
  const Rji = (i: number, j: number, w: number[]) => mat3Mul(rotvecToMat3(w), mat3MulTranspose(p.poses[j].R, p.poses[i].R));
  p.relRotationPriors = [
    { i: 1, j: 3, R_ji: Rji(1, 3, [0.05, -0.02, 0.03]), sigma: 0.01 },
    { i: 5, j: 2, R_ji: Rji(5, 2, [-0.1, 0.04, 0.0]), sigma: 0.02 }, // row block > col block
    { i: 0, j: 4, R_ji: Rji(0, 4, [0.02, 0.02, -0.01]), sigma: 0.03 }, // involves the fixed pose
  ];
  perturbBA(s, { rotSigma: 0.01, centreSigma: 0.05, pointSigma: 0.05, focalScale: 0.03, k1Delta: 0.02 });
  return s;
}

describe('computeBAGradient', () => {
  for (const loss of LOSSES) {
    it(`matches central differences of the reported cost (${loss}, weights, masks, priors, intrinsics)`, () => {
      const s = richProblem();
      const p = s.problem;
      const opts = { loss, lossScalePx: 2 };
      const g = computeBAGradient(p, opts);
      expect(g.cost).toBeCloseTo(evaluateBACost(p, opts).cost, 8);
      const costAt = (mod: (q: BAProblem) => void): number => {
        const q = cloneBAProblem(p);
        mod(q);
        return evaluateBACost(q, opts).cost;
      };
      const check = (analytic: number, mod: (q: BAProblem, h: number) => void, h: number, label: string): void => {
        const num = (costAt((q) => mod(q, h)) - costAt((q) => mod(q, -h))) / (2 * h);
        const tol = 1e-5 * Math.max(1, Math.abs(num)) + 1e-6 * g.cost;
        if (!(Math.abs(num - analytic) <= tol)) throw new Error(`${label}: analytic ${analytic} vs numeric ${num}`);
      };
      for (let i = 0; i < p.poses.length; i++) {
        if (p.poseFixed[i]) continue;
        const mask = p.translationFixedMask?.[i] ?? 0;
        for (let k = 0; k < 3; k++) {
          check(
            g.poses[6 * i + k],
            (q, h) => {
              const w = [0, 0, 0];
              w[k] = h;
              q.poses[i].R.set(mat3Mul(rotvecToMat3(w), q.poses[i].R));
            },
            1e-6,
            `pose ${i} dw${k}`,
          );
          if (!(mask & (1 << k))) check(g.poses[6 * i + 3 + k], (q, h) => (q.poses[i].t[k] += h), 1e-6, `pose ${i} dt${k}`);
        }
      }
      // fixed pose and pinned components have zero gradient
      for (let k = 0; k < 6; k++) expect(g.poses[k]).toBe(0);
      expect(g.poses[6 * 2 + 3]).toBe(0);
      expect(g.poses[6 * 2 + 5]).toBe(0);
      expect(g.poses[6 * 4 + 4]).toBe(0);
      for (let j = 0; j < p.points.length / 3; j += 2) {
        if (p.pointFixed![j]) continue;
        for (let k = 0; k < 3; k++) check(g.points[3 * j + k], (q, h) => (q.points[3 * j + k] += h), 1e-6, `point ${j}.${k}`);
      }
      expect(g.points[9]).toBe(0);
      const names = ['f', 'cx', 'cy', 'k1', 'k2'] as const;
      for (let c = 0; c < 2; c++) {
        for (let k = 0; k < 5; k++) {
          const name = names[k];
          const free = k === 0 ? p.cameraRefine[c].focal : k < 3 ? p.cameraRefine[c].principalPoint : p.cameraRefine[c].distortion;
          if (!free) {
            expect(g.cameras[5 * c + k]).toBe(0);
            continue;
          }
          check(g.cameras[5 * c + k], (q, h) => ((q.cameras[c] as unknown as Record<string, number>)[name] += h), k < 3 ? 1e-4 : 1e-8, `cam ${c} ${name}`);
        }
      }
    });
  }
});

describe('bundleAdjust: convergence', () => {
  it('self-calibrates f (+10 %) and k1 (+0.05) from perturbed poses/points back to noise level', () => {
    const s = makeSyntheticBA({ numPoses: 30, numPoints: 2000, layout: 'orbit', trackWindow: 5, k1: -0.08, k2: 0.01, noisePx: 0.5, seed: 3 });
    const p = s.problem;
    fixGaugeTwoPoses(p);
    p.cameraRefine[0] = { focal: true, principalPoint: false, distortion: true };
    perturbBA(s, { rotSigma: 0.01, centreSigma: 0.05, pointSigma: 0.05, focalScale: 0.1, k1Delta: 0.05 });
    const before = syntheticErrors(s);
    const costs: number[] = [];
    const r = bundleAdjust(p, { loss: 'trivial', maxIterations: 50, onIteration: (_i, c) => costs.push(c) });
    const after = syntheticErrors(s);
    const nParams = r.numReducedParams + 3 * r.numFreePoints;
    const exp = expectedRms(0.5, p.obsPose.length, nParams);
    console.log(
      `[ba] self-calib: ${r.termination} it=${r.iterations} rms ${r.initialRmsPx.toFixed(2)} → ${r.finalRmsPx.toFixed(4)} px (expected ${exp.toFixed(4)}), ` +
        `f err ${(after.maxFocalRelErr * 100).toFixed(4)} %, k1 ${p.cameras[0].k1.toFixed(5)}, rot ${after.maxRotErr.toExponential(2)}, ${r.timeMs.toFixed(0)} ms`,
    );
    expect(r.success).toBe(true);
    expect(['function_tolerance', 'gradient_tolerance', 'parameter_tolerance']).toContain(r.termination);
    expect(before.maxFocalRelErr).toBeGreaterThan(0.099);
    expect(after.maxFocalRelErr).toBeLessThan(0.003);
    expect(Math.abs(p.cameras[0].k1 + 0.08)).toBeLessThan(0.005);
    expect(r.finalRmsPx).toBeLessThan(exp * 1.05);
    expect(r.finalRmsPx).toBeGreaterThan(exp * 0.95);
    expect(after.maxRotErr).toBeLessThan(1e-3);
    expect(after.medianPointErr).toBeLessThan(before.medianPointErr / 10);
    for (let k = 1; k < costs.length; k++) expect(costs[k]).toBeLessThanOrEqual(costs[k - 1]);
    // fixed gauge entries untouched
    expect(Array.from(p.poses[0].R)).toEqual(Array.from(s.gtPoses[0].R));
  });

  it('self-calibration also converges with the default Cauchy loss', () => {
    const s = makeSyntheticBA({ numPoses: 30, numPoints: 2000, layout: 'orbit', trackWindow: 5, k1: -0.08, k2: 0.01, noisePx: 0.5, seed: 4 });
    const p = s.problem;
    fixGaugeTwoPoses(p);
    p.cameraRefine[0] = { focal: true, principalPoint: false, distortion: true };
    perturbBA(s, { rotSigma: 0.01, centreSigma: 0.05, pointSigma: 0.05, focalScale: -0.1, k1Delta: 0.05 });
    const r = bundleAdjust(p, { maxIterations: 100 });
    const e = syntheticErrors(s);
    expect(r.success).toBe(true);
    expect(e.maxFocalRelErr).toBeLessThan(0.003);
    expect(r.finalRmsPx).toBeLessThan(0.75);
  });

  it('large-scale scene (5–100 m, strafing camera) converges to noise level', () => {
    const s = makeSyntheticBA({ numPoses: 40, numPoints: 4000, layout: 'strafe', trackWindow: 4, radius: 0.7, noisePx: 0.7, k1: -0.03, seed: 9 });
    const p = s.problem;
    fixGaugeTwoPoses(p);
    perturbBA(s, { rotSigma: 0.003, centreSigma: 0.05, pointSigma: 0.3 });
    const r = bundleAdjust(p, { loss: 'trivial', maxIterations: 100 });
    const exp = expectedRms(0.7, p.obsPose.length, r.numReducedParams + 3 * r.numFreePoints);
    expect(r.success).toBe(true);
    expect(Math.abs(r.finalRmsPx / exp - 1)).toBeLessThan(0.05);
  });

  it('depth range 2 m – 2 km in one scene (near objects + distant houses) converges with self-calibration', () => {
    const s = makeSyntheticBA({ numPoses: 30, numPoints: 3000, layout: 'strafe', trackWindow: 4, radius: 0.3, depthRange: [2, 2000], noisePx: 0.5, k1: -0.04, seed: 5 });
    const p = s.problem;
    fixGaugeTwoPoses(p);
    p.cameraRefine[0] = { focal: true, principalPoint: false, distortion: true };
    perturbBA(s, { rotSigma: 0.003, centreSigma: 0.02, pointSigma: 0.01, focalScale: 0.05, k1Delta: 0.02 });
    const r = bundleAdjust(p, { maxIterations: 100 });
    const e = syntheticErrors(s);
    expect(r.success).toBe(true);
    expect(r.finalRmsPx).toBeLessThan(0.7);
    expect(e.maxFocalRelErr).toBeLessThan(0.003);
    expect(e.maxRotErr).toBeLessThan(1e-3);
    expect(e.medianPointErr).toBeLessThan(0.01);
  });

  it('points near infinity (1 km – 1000 km, depth unobservable) stay finite and fit to noise level', () => {
    const s = makeSyntheticBA({ numPoses: 30, numPoints: 2000, layout: 'strafe', trackWindow: 4, radius: 0.3, depthRange: [1e3, 1e6], noisePx: 0.5, seed: 6 });
    const p = s.problem;
    fixGaugeTwoPoses(p);
    perturbBA(s, { rotSigma: 0.003, centreSigma: 0.02, pointSigma: 0.01 });
    const r = bundleAdjust(p, { maxIterations: 100 });
    expect(r.success).toBe(true);
    for (const v of p.points) expect(Number.isFinite(v)).toBe(true);
    expect(r.finalRmsPx).toBeLessThan(0.75);
    expect(syntheticErrors(s).maxRotErr).toBeLessThan(3e-3);
  });

  it('multiple cameras: each camera has its own refined intrinsics', () => {
    const s = makeSyntheticBA({ numPoses: 24, numPoints: 1500, numCameras: 2, layout: 'orbit', trackWindow: 5, k1: -0.05, noisePx: 0.3, seed: 12 });
    const p = s.problem;
    fixGaugeTwoPoses(p);
    for (let c = 0; c < 2; c++) p.cameraRefine[c] = { focal: true, principalPoint: false, distortion: true };
    perturbBA(s, { rotSigma: 0.005, centreSigma: 0.02, pointSigma: 0.02, focalScale: 0.05, k1Delta: 0.02 });
    bundleAdjust(p, { loss: 'trivial' });
    for (let c = 0; c < 2; c++) expect(Math.abs(p.cameras[c].f / s.gtCameras[c].f - 1)).toBeLessThan(0.003);
  });
});

describe('bundleAdjust: robustness', () => {
  it('Cauchy loss is robust to 10 % gross outliers (trivial loss is not)', () => {
    const make = () => {
      const s = makeSyntheticBA({ numPoses: 30, numPoints: 2000, layout: 'orbit', trackWindow: 5, k1: -0.05, noisePx: 0.5, outlierFraction: 0.1, seed: 21 });
      fixGaugeTwoPoses(s.problem);
      s.problem.cameraRefine[0] = { focal: true, principalPoint: false, distortion: true };
      perturbBA(s, { rotSigma: 0.005, centreSigma: 0.03, pointSigma: 0.03, focalScale: 0.05, k1Delta: 0.02 });
      return s;
    };
    const sc = make();
    const rc = bundleAdjust(sc.problem, { loss: 'cauchy', lossScalePx: 1, maxIterations: 100 });
    const st = make();
    bundleAdjust(st.problem, { loss: 'trivial', maxIterations: 100 });
    const ec = syntheticErrors(sc),
      et = syntheticErrors(st);
    // inlier RMS
    const err = reprojectionErrors(sc.problem);
    let se = 0,
      n = 0;
    for (let o = 0; o < err.length; o++)
      if (!sc.outlier[o]) {
        se += err[o] * err[o];
        n++;
      }
    const inlierRms = Math.sqrt(se / n);
    console.log(
      `[ba] outliers 10 %: cauchy f err ${(ec.maxFocalRelErr * 100).toFixed(3)} % rot ${ec.maxRotErr.toExponential(2)} inlier RMS ${inlierRms.toFixed(3)} px ` +
        `| trivial f err ${(et.maxFocalRelErr * 100).toFixed(3)} % rot ${et.maxRotErr.toExponential(2)}`,
    );
    expect(rc.success).toBe(true);
    expect(inlierRms).toBeLessThan(0.75);
    expect(ec.maxFocalRelErr).toBeLessThan(0.003);
    expect(ec.maxRotErr).toBeLessThan(1e-3);
    expect(et.maxRotErr).toBeGreaterThan(3 * ec.maxRotErr);
  });

  it('ignores observations that are not finite and zero-weight observations', () => {
    const s = makeSyntheticBA({ numPoses: 10, numPoints: 300, layout: 'orbit', trackWindow: 3, noisePx: 0.3, outlierFraction: 0.05, seed: 2 });
    const p = s.problem;
    fixGaugeTwoPoses(p);
    p.obsWeight = new Float32Array(p.obsPose.length).fill(1);
    for (let o = 0; o < p.obsPose.length; o++) if (s.outlier[o]) p.obsWeight[o] = 0;
    p.obsUV[0] = NaN;
    perturbBA(s, { rotSigma: 0.003, centreSigma: 0.01, pointSigma: 0.01 });
    const r = bundleAdjust(p, { loss: 'trivial' });
    expect(r.success).toBe(true);
    expect(Number.isFinite(r.finalCost)).toBe(true);
    for (const v of p.points) expect(Number.isFinite(v)).toBe(true);
    expect(syntheticErrors(s).maxRotErr).toBeLessThan(2e-3);
  });

  it('handles points behind cameras and single-observation points without NaNs', () => {
    const s = makeSyntheticBA({ numPoses: 8, numPoints: 200, layout: 'orbit', trackWindow: 2, noisePx: 0.3, seed: 8 });
    const p = s.problem;
    fixGaugeTwoPoses(p);
    // point 0 moved behind its first observing camera
    const C = poseCentre(p.poses[p.obsPose[0]]);
    const j0 = p.obsPoint[0];
    for (let k = 0; k < 3; k++) p.points[3 * j0 + k] = 2 * C[k] - p.points[3 * j0 + k];
    // add single-observation points
    const nX = p.points.length / 3;
    const pts = new Float64Array(p.points.length + 6);
    pts.set(p.points);
    pts.set([0.1, 0.2, 0.3, -0.2, 0.1, 0.0], p.points.length);
    p.points = pts;
    const obsPose = Array.from(p.obsPose),
      obsPoint = Array.from(p.obsPoint),
      uv = Array.from(p.obsUV);
    obsPose.push(1, 0);
    obsPoint.push(nX, nX + 1);
    uv.push(900, 700, 1000, 650);
    p.obsPose = Uint32Array.from(obsPose);
    p.obsPoint = Uint32Array.from(obsPoint);
    p.obsUV = Float64Array.from(uv);
    perturbBA(s, { rotSigma: 0.002, centreSigma: 0.01, pointSigma: 0.01 });
    for (const loss of LOSSES) {
      const q = cloneBAProblem(p);
      const r = bundleAdjust(q, { loss });
      expect(r.success).toBe(true);
      for (const v of q.points) expect(Number.isFinite(v)).toBe(true);
      for (const ps of q.poses) for (const v of ps.t) expect(Number.isFinite(v)).toBe(true);
      expect(r.finalCost).toBeLessThanOrEqual(r.initialCost);
    }
  });
});

describe('bundleAdjust: linear solvers', () => {
  it('dense Schur and PCG Schur give the same steps and solution', () => {
    const s = makeSyntheticBA({ numPoses: 40, numPoints: 3000, layout: 'orbit', trackWindow: 4, k1: -0.05, noisePx: 0.5, seed: 31 });
    fixGaugeTwoPoses(s.problem);
    s.problem.cameraRefine[0] = { focal: true, principalPoint: true, distortion: true };
    perturbBA(s, { rotSigma: 0.005, centreSigma: 0.03, pointSigma: 0.03, focalScale: 0.03, k1Delta: 0.02 });
    const base = s.problem;
    for (const iters of [1, 30]) {
      const a = cloneBAProblem(base),
        b = cloneBAProblem(base);
      const ra = bundleAdjust(a, { linearSolver: 'dense_schur', maxIterations: iters, loss: 'huber' });
      const rb = bundleAdjust(b, { linearSolver: 'pcg_schur', maxIterations: iters, loss: 'huber', pcgTolerance: 1e-13, pcgMaxIterations: 5000 });
      expect(ra.linearSolver).toBe('dense_schur');
      expect(rb.linearSolver).toBe('pcg_schur');
      expect(rb.pcgIterations).toBeGreaterThan(0);
      expect(Math.abs(ra.finalCost - rb.finalCost) / ra.finalCost).toBeLessThan(1e-9);
      let maxDiff = 0;
      for (let k = 0; k < a.points.length; k++) maxDiff = Math.max(maxDiff, Math.abs(a.points[k] - b.points[k]));
      for (let i = 0; i < a.poses.length; i++) maxDiff = Math.max(maxDiff, rotationAngleBetween(a.poses[i].R, b.poses[i].R));
      expect(maxDiff).toBeLessThan(1e-7);
      expect(Math.abs(a.cameras[0].f - b.cameras[0].f)).toBeLessThan(1e-5);
    }
  });

  it("'auto': direct below 2400 reduced parameters; above only for a narrow envelope, else PCG", () => {
    const small = makeSyntheticBA({ numPoses: 20, numPoints: 200, seed: 1 });
    fixGaugeTwoPoses(small.problem);
    expect(bundleAdjust(small.problem, { maxIterations: 1 }).linearSolver).toBe('dense_schur');
    // banded (sequential) 402-pose capture: narrow envelope ⇒ direct solve
    const big = makeSyntheticBA({ numPoses: 402, numPoints: 2500, layout: 'strafe', trackWindow: 2, seed: 1 });
    fixGaugeTwoPoses(big.problem);
    const banded = cloneBAProblem(big.problem);
    const r = bundleAdjust(banded, { maxIterations: 1 });
    expect(r.numReducedParams).toBe(6 * 401 - 1);
    expect(r.linearSolver).toBe('dense_schur');
    // same problem with randomly ordered poses: wide envelope ⇒ PCG; the step is the same
    const perm = Array.from({ length: 402 }, (_, i) => i);
    new Rng(5).shuffle(perm);
    const shuffled = permutePoses(big.problem, perm);
    const rs = bundleAdjust(shuffled, { maxIterations: 1, pcgTolerance: 1e-12, pcgMaxIterations: 5000 });
    expect(rs.linearSolver).toBe('pcg_schur');
    expect(Math.abs(rs.finalCost - r.finalCost) / r.finalCost).toBeLessThan(1e-8);
    for (let k = 0; k < banded.points.length; k++) expect(Math.abs(shuffled.points[k] - banded.points[k])).toBeLessThan(1e-7);
    // forcing the direct solver on the shuffled problem gives the same answer as well
    const forced = permutePoses(big.problem, perm);
    const rf = bundleAdjust(forced, { maxIterations: 1, linearSolver: 'dense_schur' });
    expect(rf.linearSolver).toBe('dense_schur');
    expect(Math.abs(rf.finalCost - r.finalCost) / r.finalCost).toBeLessThan(1e-9);
  });
});

/** Reorder poses: new pose k = old pose perm[k] (observations / priors remapped). */
function permutePoses(p: BAProblem, perm: number[]): BAProblem {
  const q = cloneBAProblem(p);
  const inv = new Uint32Array(perm.length);
  perm.forEach((old, k) => (inv[old] = k));
  q.poses = perm.map((old) => ({ R: p.poses[old].R.slice(), t: p.poses[old].t.slice() }));
  q.imageCamera = Uint32Array.from(perm.map((old) => p.imageCamera[old]));
  q.poseFixed = Uint8Array.from(perm.map((old) => p.poseFixed[old]));
  if (p.translationFixedMask) q.translationFixedMask = Uint8Array.from(perm.map((old) => p.translationFixedMask![old]));
  q.obsPose = q.obsPose.map((i) => inv[i]);
  return q;
}

describe('bundleAdjust: gauge handling', () => {
  it('fixed poses, pinned translation components and fixed points stay bit-identical', () => {
    const s = makeSyntheticBA({ numPoses: 12, numPoints: 500, layout: 'orbit', trackWindow: 3, seed: 41 });
    const p = s.problem;
    p.poseFixed[3] = 1;
    p.poseFixed[7] = 1;
    p.translationFixedMask = new Uint8Array(12);
    p.translationFixedMask[0] = 0b111;
    p.translationFixedMask[5] = 0b010;
    p.pointFixed = new Uint8Array(500);
    for (let j = 0; j < 500; j += 7) p.pointFixed[j] = 1;
    perturbBA(s, { rotSigma: 0.01, centreSigma: 0.05, pointSigma: 0.05 });
    const before = cloneBAProblem(p);
    const r = bundleAdjust(p, { loss: 'trivial' });
    expect(r.success).toBe(true);
    for (const i of [3, 7]) {
      expect(Array.from(p.poses[i].R)).toEqual(Array.from(before.poses[i].R));
      expect(Array.from(p.poses[i].t)).toEqual(Array.from(before.poses[i].t));
    }
    expect(Array.from(p.poses[0].t)).toEqual(Array.from(before.poses[0].t));
    expect(rotationAngleBetween(p.poses[0].R, before.poses[0].R)).toBeGreaterThan(1e-5);
    expect(p.poses[5].t[1]).toBe(before.poses[5].t[1]);
    expect(p.poses[5].t[0]).not.toBe(before.poses[5].t[0]);
    for (let j = 0; j < 500; j++) {
      const same = p.points[3 * j] === before.points[3 * j] && p.points[3 * j + 2] === before.points[3 * j + 2];
      expect(same).toBe(j % 7 === 0);
    }
  });

  it('free gauge (nothing fixed) still converges thanks to LM damping', () => {
    const s = makeSyntheticBA({ numPoses: 15, numPoints: 800, layout: 'orbit', trackWindow: 4, noisePx: 0.5, seed: 43 });
    const p = s.problem;
    perturbBA(s, { rotSigma: 0.01, centreSigma: 0.03, pointSigma: 0.03 });
    const r = bundleAdjust(p, { loss: 'trivial', maxIterations: 100 });
    expect(r.success).toBe(true);
    // The 7-dof gauge is free: compare the cost with the gauge-fixed solution.
    const q = cloneBAProblem(p);
    fixGaugeTwoPoses(q);
    const rq = bundleAdjust(q, { loss: 'trivial', maxIterations: 100 });
    expect(Math.abs(r.finalCost - rq.finalCost) / rq.finalCost).toBeLessThan(1e-4);
    expect(r.finalRmsPx).toBeLessThan(0.75);
  });

  it('points seen only by fixed poses are optimised (incl. single-view points)', () => {
    const s = makeSyntheticBA({ numPoses: 6, numPoints: 300, layout: 'orbit', trackWindow: 2, noisePx: 0, seed: 44 });
    const p = s.problem;
    p.poseFixed.fill(1);
    perturbBA(s, { pointSigma: 0.05 });
    const r = bundleAdjust(p, { loss: 'trivial', maxIterations: 50, functionTolerance: 1e-12 });
    expect(r.success).toBe(true);
    expect(r.numReducedParams).toBe(0);
    expect(r.finalRmsPx).toBeLessThan(1e-4);
    expect(syntheticErrors(s).rmsPointErr).toBeLessThan(1e-5);
  });
});

describe('bundleAdjust: priors', () => {
  it('centre priors fix the gauge in the prior frame', () => {
    const s = makeSyntheticBA({ numPoses: 12, numPoints: 600, layout: 'orbit', trackWindow: 3, noisePx: 0.3, seed: 51 });
    const p = s.problem;
    p.centerPriors = s.gtPoses.map((g, i) => ({ pose: i, center: poseCentre(g), sigma: 0.01 }));
    // start far from the prior frame: rotate + shift everything by a similarity
    perturbBA(s, { rotSigma: 0.01, centreSigma: 0.2, pointSigma: 0.1 });
    const r = bundleAdjust(p, { loss: 'trivial', maxIterations: 100 });
    expect(r.success).toBe(true);
    const e = syntheticErrors(s);
    expect(e.maxCentreErr).toBeLessThan(0.01);
    expect(e.maxRotErr).toBeLessThan(2e-3);
  });

  it('a strong centre prior pulls the camera, a weak one does not', () => {
    for (const [sigma, expectNearPrior] of [
      [1e-5, true],
      [1e3, false],
    ] as const) {
      const s = makeSyntheticBA({ numPoses: 10, numPoints: 500, layout: 'orbit', trackWindow: 3, noisePx: 0.2, seed: 52 });
      const p = s.problem;
      fixGaugeTwoPoses(p);
      const target = poseCentre(s.gtPoses[5]).map((v, k) => v + (k === 2 ? 0.3 : 0));
      p.centerPriors = [{ pose: 5, center: target, sigma }];
      const r = bundleAdjust(p, { loss: 'cauchy', maxIterations: 100 });
      expect(r.success).toBe(true);
      const C = poseCentre(p.poses[5]);
      const dPrior = Math.hypot(C[0] - target[0], C[1] - target[1], C[2] - target[2]);
      const dGt = Math.hypot(...[0, 1, 2].map((k) => C[k] - poseCentre(s.gtPoses[5])[k]));
      if (expectNearPrior) expect(dPrior).toBeLessThan(1e-3);
      else expect(dGt).toBeLessThan(5e-3);
    }
  });

  it('relative-rotation priors determine an unobserved rotation exactly (large initial error)', () => {
    const s = makeSyntheticBA({ numPoses: 8, numPoints: 300, layout: 'orbit', trackWindow: 2, seed: 53 });
    const p = s.problem;
    p.poseFixed.fill(1);
    p.poseFixed[4] = 0;
    // drop all observations of pose 4
    const keep: number[] = [];
    for (let o = 0; o < p.obsPose.length; o++) if (p.obsPose[o] !== 4) keep.push(o);
    p.obsPose = Uint32Array.from(keep.map((o) => p.obsPose[o]));
    p.obsUV = Float64Array.from(keep.flatMap((o) => [p.obsUV[2 * o], p.obsUV[2 * o + 1]]));
    p.obsPoint = Uint32Array.from(keep.map((o) => s.problem.obsPoint[o]));
    const Rji = mat3MulTranspose(s.gtPoses[4].R, s.gtPoses[3].R);
    p.relRotationPriors = [{ i: 3, j: 4, R_ji: Rji, sigma: 0.01 }];
    p.poses[4].R.set(mat3Mul(rotvecToMat3([0.5, -0.4, 0.3]), p.poses[4].R));
    const t4 = p.poses[4].t.slice();
    // (default parameterTolerance is relative to ‖x‖ incl. all points ⇒ would stop at ~1e-9 rad)
    const r = bundleAdjust(p, { loss: 'trivial', maxIterations: 50, functionTolerance: 0, parameterTolerance: 0 });
    expect(r.success).toBe(true);
    expect(rotationAngleBetween(p.poses[4].R, s.gtPoses[4].R)).toBeLessThan(1e-9);
    expect(Array.from(p.poses[4].t)).toEqual(Array.from(t4)); // unconstrained translation is not moved
  });

  it('relative-rotation prior weight trades off against observations', () => {
    const angles: number[] = [];
    for (const sigma of [1e-6, 1e2]) {
      const s = makeSyntheticBA({ numPoses: 8, numPoints: 400, layout: 'orbit', trackWindow: 2, noisePx: 0.2, seed: 54 });
      const p = s.problem;
      fixGaugeTwoPoses(p);
      // prior deliberately 0.02 rad off the truth
      const bias = rotvecToMat3([0, 0.02, 0]);
      const Rji = mat3Mul(bias, mat3MulTranspose(s.gtPoses[5].R, s.gtPoses[4].R));
      p.relRotationPriors = [{ i: 4, j: 5, R_ji: Rji, sigma }];
      bundleAdjust(p, { loss: 'trivial', maxIterations: 100 });
      const rel = mat3MulTranspose(p.poses[5].R, p.poses[4].R);
      angles.push(rotationAngleBetween(rel, Rji));
    }
    expect(angles[0]).toBeLessThan(1e-5); // prior wins
    expect(angles[1]).toBeGreaterThan(0.015); // observations win
  });
});

describe('bundleAdjust: termination and edge cases', () => {
  const small = () => {
    const s = makeSyntheticBA({ numPoses: 8, numPoints: 300, layout: 'orbit', trackWindow: 3, noisePx: 0.5, seed: 61 });
    fixGaugeTwoPoses(s.problem);
    perturbBA(s, { rotSigma: 0.01, centreSigma: 0.03, pointSigma: 0.03 });
    return s;
  };

  it('maxIterations, cancel token, time limit', () => {
    let s = small();
    let before = cloneBAProblem(s.problem);
    let r = bundleAdjust(s.problem, { maxIterations: 0 });
    expect(r.termination).toBe('max_iterations');
    expect(r.iterations).toBe(0);
    expect(Array.from(s.problem.points)).toEqual(Array.from(before.points));
    r = bundleAdjust(s.problem, { maxIterations: 2 });
    expect(r.termination).toBe('max_iterations');
    expect(r.iterations).toBe(2);

    s = small();
    before = cloneBAProblem(s.problem);
    r = bundleAdjust(s.problem, { cancel: { cancelled: true } });
    expect(r.termination).toBe('cancelled');
    expect(r.success).toBe(true);
    expect(Array.from(s.problem.points)).toEqual(Array.from(before.points));

    s = small();
    const tok = { cancelled: false };
    r = bundleAdjust(s.problem, { cancel: tok, onIteration: (i) => (tok.cancelled = i >= 3) });
    expect(r.termination).toBe('cancelled');
    expect(r.iterations).toBe(3);
    expect(evaluateBACost(s.problem).cost).toBeCloseTo(r.finalCost, 6);
    expect(r.finalCost).toBeLessThan(r.initialCost);

    s = small();
    r = bundleAdjust(s.problem, { timeLimitMs: 0 });
    expect(r.termination).toBe('time_limit');
  });

  it('writes back the last accepted state when onIteration throws', () => {
    const s = small();
    let cost = NaN;
    expect(() =>
      bundleAdjust(s.problem, {
        onIteration: (i, c) => {
          cost = c;
          if (i === 2) throw new Error('stop');
        },
      }),
    ).toThrow('stop');
    expect(evaluateBACost(s.problem).cost).toBeCloseTo(cost, 6);
  });

  it('reports convergence by gradient / function tolerance and keeps optimum stable', () => {
    const s = small();
    const r1 = bundleAdjust(s.problem, { loss: 'trivial', maxIterations: 200 });
    expect(['function_tolerance', 'gradient_tolerance', 'parameter_tolerance']).toContain(r1.termination);
    const r2 = bundleAdjust(s.problem, { loss: 'trivial', maxIterations: 200 });
    expect(r2.success).toBe(true);
    expect(Math.abs(r2.finalCost - r1.finalCost) / r1.finalCost).toBeLessThan(1e-6);
    expect(r2.iterations).toBeLessThanOrEqual(3);
  });

  it('degenerate inputs: no free parameters, no observations, invalid problems', () => {
    const s = small();
    const p = cloneBAProblem(s.problem);
    p.poseFixed.fill(1);
    p.pointFixed = new Uint8Array(p.points.length / 3).fill(1);
    const r = bundleAdjust(p);
    expect(r.termination).toBe('no_free_parameters');
    expect(r.success).toBe(true);
    expect(r.finalCost).toBe(r.initialCost);

    const q = cloneBAProblem(s.problem);
    q.obsPose = new Uint32Array(0);
    q.obsPoint = new Uint32Array(0);
    q.obsUV = new Float64Array(0);
    const rq = bundleAdjust(q);
    expect(rq.success).toBe(true);
    expect(rq.initialCost).toBe(0);
    expect(rq.finalRmsPx).toBe(0);

    const bad = cloneBAProblem(s.problem);
    bad.obsPose[3] = 99;
    expect(() => bundleAdjust(bad)).toThrow(RangeError);
    const bad2 = cloneBAProblem(s.problem);
    bad2.relRotationPriors = [{ i: 1, j: 1, R_ji: rotvecToMat3([0, 0, 0]), sigma: 1 }];
    expect(() => bundleAdjust(bad2)).toThrow(RangeError);
    expect(() => bundleAdjust(s.problem, { loss: 'tukey' as BALossType })).toThrow(RangeError);
  });

  it('reprojectionErrors matches the RMS reported by bundleAdjust', () => {
    const s = small();
    const r = bundleAdjust(s.problem, { maxIterations: 3 });
    const out = new Float64Array(s.problem.obsPose.length + 5);
    const e = reprojectionErrors(s.problem, out);
    expect(e).toBe(out);
    let se = 0;
    for (let o = 0; o < s.problem.obsPose.length; o++) se += e[o] * e[o];
    expect(Math.sqrt(se / s.problem.obsPose.length)).toBeCloseTo(r.finalRmsPx, 9);
    expect(evaluateBACost(s.problem).rmsPx).toBeCloseTo(r.finalRmsPx, 9);
  });
});
