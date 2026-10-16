/**
 * Adversarial review tests for src/ba: an independent brute-force reference for the
 * LM step (full dense damped normal equations, no Schur complement), degenerate
 * geometry, extreme scales, outliers, pose-graph-only problems and edge cases.
 */
import { describe, expect, it } from 'vitest';
import { solveSPD } from '../math/decomp';
import { mat3Mul, mat3MulTranspose } from '../math/mat3';
import { Rng } from '../math/random';
import { mat3ToRotvec, rotationAngleBetween, rotvecToMat3 } from '../math/rotation';
import { radialResidual, radialResidualJacobian, RJ_SIZE } from './camera-model';
import { lossKind, lossRho, lossWeight } from './loss';
import { centerPriorResidualJacobian, relRotationPriorResidualJacobian } from './priors';
import type { BALossType, BAProblem } from './problem';
import { bundleAdjust, computeBAGradient, evaluateBACost, reprojectionErrors } from './solver';
import { cloneBAProblem, fixGaugeTwoPoses, makeSyntheticBA, perturbBA, poseCentre, syntheticErrors, type SyntheticBA } from './synthetic';

// ---------------------------------------------------------------------------------------------
// Brute-force reference: assemble the full IRLS-weighted Gauss–Newton system over ALL free
// parameters (poses [δω, δt] minus pinned components, free intrinsics, free points), apply the
// documented Marquardt damping H + λ·clamp(diag H, 1e-6, 1e32) and solve it densely.
// ---------------------------------------------------------------------------------------------

interface ParamIndex {
  pose: Int32Array; // 6 per pose, −1 = not a parameter
  cam: Int32Array; // 5 per camera
  point: Int32Array; // 3 per point
  n: number;
}

function indexParams(p: BAProblem, frozen?: Set<number>): ParamIndex {
  const nP = p.poses.length,
    nC = p.cameras.length,
    nX = p.points.length / 3;
  const pose = new Int32Array(6 * nP).fill(-1),
    cam = new Int32Array(5 * nC).fill(-1),
    point = new Int32Array(3 * nX).fill(-1);
  // valid observations ⇒ which points are free
  const hasObs = new Uint8Array(nX);
  const K = camK(p);
  const r = new Float64Array(2);
  for (let o = 0; o < p.obsPose.length; o++) {
    const i = p.obsPose[o];
    const w = p.obsWeight ? p.obsWeight[o] : 1;
    const ok = radialResidual(p.poses[i].R, 0, p.poses[i].t, 0, p.points, 3 * p.obsPoint[o], K, 5 * p.imageCamera[i], p.obsUV[2 * o], p.obsUV[2 * o + 1], r);
    if (ok && w >= 0 && w < Infinity) hasObs[p.obsPoint[o]] = 1;
  }
  let n = 0;
  for (let i = 0; i < nP; i++) {
    if (p.poseFixed[i]) continue;
    const m = p.translationFixedMask ? p.translationFixedMask[i] : 0;
    for (let k = 0; k < 6; k++) if (k < 3 || !(m & (1 << (k - 3)))) pose[6 * i + k] = n++;
  }
  for (let c = 0; c < nC; c++) {
    const cr = p.cameraRefine[c];
    if (cr.focal) cam[5 * c] = n++;
    if (cr.principalPoint) {
      cam[5 * c + 1] = n++;
      cam[5 * c + 2] = n++;
    }
    if (cr.distortion) {
      cam[5 * c + 3] = n++;
      cam[5 * c + 4] = n++;
    }
  }
  for (let j = 0; j < nX; j++) {
    if ((p.pointFixed && p.pointFixed[j]) || !hasObs[j] || frozen?.has(j)) continue;
    for (let k = 0; k < 3; k++) point[3 * j + k] = n++;
  }
  return { pose, cam, point, n };
}

function camK(p: BAProblem): Float64Array {
  const K = new Float64Array(5 * p.cameras.length);
  p.cameras.forEach((c, k) => K.set([c.f, c.cx, c.cy, c.k1, c.k2], 5 * k));
  return K;
}

/** Full damped LM step (H + λD) δ = −g, or the undamped gradient / Hessian if requested. */
function bruteForceStep(p: BAProblem, loss: BALossType, lossScalePx: number, lambda: number, frozen?: Set<number>) {
  const idx = indexParams(p, frozen);
  const n = idx.n;
  const H = new Float64Array(n * n);
  const g = new Float64Array(n);
  const K = camK(p);
  const J = new Float64Array(RJ_SIZE);
  const kind = lossKind(loss);
  const b = lossScalePx * lossScalePx;
  // one residual row: list of (param index, value)
  const rowIdx: number[] = [];
  const rowVal: number[] = [];
  const addRow = (res: number): void => {
    for (let a = 0; a < rowIdx.length; a++) {
      g[rowIdx[a]] += rowVal[a] * res;
      for (let c = 0; c < rowIdx.length; c++) H[rowIdx[a] * n + rowIdx[c]] += rowVal[a] * rowVal[c];
    }
    rowIdx.length = 0;
    rowVal.length = 0;
  };
  const push = (k: number, v: number): void => {
    if (k >= 0) {
      rowIdx.push(k);
      rowVal.push(v);
    }
  };
  for (let o = 0; o < p.obsPose.length; o++) {
    const i = p.obsPose[o],
      j = p.obsPoint[o],
      c = p.imageCamera[i];
    const w = p.obsWeight ? p.obsWeight[o] : 1;
    if (!(w >= 0 && w < Infinity)) continue;
    if (!radialResidualJacobian(p.poses[i].R, 0, p.poses[i].t, 0, p.points, 3 * j, K, 5 * c, p.obsUV[2 * o], p.obsUV[2 * o + 1], J)) continue;
    const s = w * w * (J[0] * J[0] + J[1] * J[1]);
    const sc = w * Math.sqrt(lossWeight(kind, s, b));
    for (let row = 0; row < 2; row++) {
      for (let k = 0; k < 6; k++) push(idx.pose[6 * i + k], sc * J[2 + 6 * row + k]);
      for (let k = 0; k < 3; k++) push(idx.point[3 * j + k], sc * J[14 + 3 * row + k]);
      for (let k = 0; k < 5; k++) push(idx.cam[5 * c + k], sc * J[20 + 5 * row + k]);
      addRow(sc * J[row]);
    }
  }
  const r3 = new Float64Array(3),
    PJ = new Float64Array(18),
    Ji = new Float64Array(9),
    Jj = new Float64Array(9);
  for (const cp of p.centerPriors ?? []) {
    centerPriorResidualJacobian(p.poses[cp.pose].R, 0, p.poses[cp.pose].t, 0, cp.center, 1 / cp.sigma, r3, PJ);
    for (let row = 0; row < 3; row++) {
      for (let k = 0; k < 6; k++) push(idx.pose[6 * cp.pose + k], PJ[6 * row + k]);
      addRow(r3[row]);
    }
  }
  for (const rp of p.relRotationPriors ?? []) {
    relRotationPriorResidualJacobian(p.poses[rp.i].R, 0, p.poses[rp.j].R, 0, rp.R_ji, 1 / rp.sigma, r3, Ji, Jj);
    for (let row = 0; row < 3; row++) {
      for (let k = 0; k < 3; k++) push(idx.pose[6 * rp.i + k], Ji[3 * row + k]);
      for (let k = 0; k < 3; k++) push(idx.pose[6 * rp.j + k], Jj[3 * row + k]);
      addRow(r3[row]);
    }
  }
  const A = H.slice();
  for (let k = 0; k < n; k++) {
    const d = Math.min(1e32, Math.max(1e-6, H[k * n + k]));
    A[k * n + k] += lambda * d;
  }
  const delta = solveSPD(A, n, g.map((v) => -v));
  if (!delta) throw new Error('brute-force system not SPD');
  return { idx, delta, H, g };
}

/** Step actually taken by bundleAdjust (after → before), expressed in the brute-force parameter order. */
function takenStep(before: BAProblem, after: BAProblem, idx: ParamIndex): Float64Array {
  const d = new Float64Array(idx.n);
  for (let i = 0; i < before.poses.length; i++) {
    const w = mat3ToRotvec(mat3MulTranspose(after.poses[i].R, before.poses[i].R));
    for (let k = 0; k < 3; k++) {
      if (idx.pose[6 * i + k] >= 0) d[idx.pose[6 * i + k]] = w[k];
      if (idx.pose[6 * i + 3 + k] >= 0) d[idx.pose[6 * i + 3 + k]] = after.poses[i].t[k] - before.poses[i].t[k];
    }
  }
  const names = ['f', 'cx', 'cy', 'k1', 'k2'] as const;
  for (let c = 0; c < before.cameras.length; c++)
    for (let k = 0; k < 5; k++) if (idx.cam[5 * c + k] >= 0) d[idx.cam[5 * c + k]] = after.cameras[c][names[k]] - before.cameras[c][names[k]];
  for (let k = 0; k < before.points.length; k++) if (idx.point[k] >= 0) d[idx.point[k]] = after.points[k] - before.points[k];
  return d;
}

/** Small but complete problem: 2 cameras with different refine flags, weights, masks, fixed points, priors, outliers. */
function adversarialProblem(seed: number, outlierFraction = 0.1): SyntheticBA {
  const s = makeSyntheticBA({ numPoses: 8, numPoints: 60, layout: 'orbit', trackWindow: 3, k1: -0.1, k2: 0.02, numCameras: 2, noisePx: 1, outlierFraction, seed });
  const p = s.problem;
  p.poseFixed[6] = 1; // a fixed pose in the middle (not the first block)
  p.translationFixedMask = new Uint8Array(p.poses.length);
  p.translationFixedMask[1] = 0b100;
  p.translationFixedMask[3] = 0b011;
  p.translationFixedMask[6] = 0b111; // on a fixed pose: must be ignored
  p.pointFixed = new Uint8Array(p.points.length / 3);
  p.pointFixed[2] = 1;
  p.pointFixed[17] = 1;
  p.cameraRefine[0] = { focal: true, principalPoint: true, distortion: true };
  p.cameraRefine[1] = { focal: true, principalPoint: false, distortion: true };
  const rng = new Rng(seed + 100);
  p.obsWeight = new Float32Array(p.obsPose.length).map(() => 0.3 + 2 * rng.next());
  p.centerPriors = [
    { pose: 2, center: poseCentre(p.poses[2]).map((v) => v + 0.1), sigma: 0.05 },
    { pose: 3, center: poseCentre(p.poses[3]).map((v) => v - 0.05), sigma: 0.2 },
  ];
  const Rji = (i: number, j: number, w: number[]) => mat3Mul(rotvecToMat3(w), mat3MulTranspose(p.poses[j].R, p.poses[i].R));
  p.relRotationPriors = [
    { i: 7, j: 1, R_ji: Rji(7, 1, [0.03, -0.02, 0.01]), sigma: 0.01 }, // swapped block order
    { i: 2, j: 6, R_ji: Rji(2, 6, [0.01, 0.0, -0.02]), sigma: 0.02 }, // involves the fixed pose
    { i: 4, j: 5, R_ji: Rji(4, 5, [0.0, 0.02, 0.0]), sigma: 0.05 },
  ];
  perturbBA(s, { rotSigma: 0.004, centreSigma: 0.02, pointSigma: 0.02, focalScale: 0.01, k1Delta: 0.01, seed: seed + 3 });
  return s;
}

describe('LM step equals the brute-force damped normal-equation solution', () => {
  for (const [loss, lambda, solver] of [
    ['trivial', 1e-4, 'dense_schur'],
    ['cauchy', 1e-4, 'dense_schur'],
    ['huber', 0.3, 'dense_schur'],
    ['soft_l1', 1e-4, 'pcg_schur'],
    ['cauchy', 5, 'pcg_schur'],
  ] as const) {
    it(`${loss}, λ = ${lambda}, ${solver}`, () => {
      // (uniform outliers make the trivial-loss GN step overshoot ⇒ rejected; no outliers there)
      const s = adversarialProblem(7, loss === 'trivial' ? 0 : 0.1);
      const p = s.problem;
      const before = cloneBAProblem(p);
      const ref = bruteForceStep(before, loss, 1.5, lambda);
      const r = bundleAdjust(p, {
        loss,
        lossScalePx: 1.5,
        initialLambda: lambda,
        maxIterations: 1,
        linearSolver: solver,
        pcgTolerance: 1e-14,
        pcgMaxIterations: 10000,
      });
      expect(r.linearSolver).toBe(solver);
      expect(r.successfulIterations).toBe(1);
      expect(r.numReducedParams + 3 * r.numFreePoints).toBe(ref.idx.n);
      const d = takenStep(before, p, ref.idx);
      let maxRef = 0,
        maxErr = 0;
      for (let k = 0; k < ref.idx.n; k++) {
        maxRef = Math.max(maxRef, Math.abs(ref.delta[k]));
        maxErr = Math.max(maxErr, Math.abs(d[k] - ref.delta[k]));
      }
      expect(maxRef).toBeGreaterThan(1e-4);
      expect(maxErr / maxRef).toBeLessThan(1e-7);
      // gain ratio: model decrease m(0) − m(δ) = −gᵀδ − ½δᵀHδ (undamped) ⇒ λ update
      let gd = 0,
        dHd = 0;
      const n = ref.idx.n;
      for (let a = 0; a < n; a++) {
        gd += ref.g[a] * ref.delta[a];
        for (let c = 0; c < n; c++) dHd += ref.delta[a] * ref.H[a * n + c] * ref.delta[c];
      }
      const pred = -gd - 0.5 * dHd;
      const rho = (r.initialCost - r.finalCost) / pred;
      expect(rho).toBeGreaterThan(1e-3);
      const expectLambda = lambda * Math.max(1 / 3, 1 - Math.pow(2 * rho - 1, 3));
      expect(Math.abs(r.finalLambda / expectLambda - 1)).toBeLessThan(1e-5);
      // fixed things untouched
      expect(Array.from(p.poses[6].R)).toEqual(Array.from(before.poses[6].R));
      expect(Array.from(p.poses[6].t)).toEqual(Array.from(before.poses[6].t));
      expect(p.poses[1].t[2]).toBe(before.poses[1].t[2]);
      expect(p.poses[3].t[0]).toBe(before.poses[3].t[0]);
      expect(p.poses[3].t[1]).toBe(before.poses[3].t[1]);
      expect(p.cameras[1].cx).toBe(before.cameras[1].cx);
      for (const j of [2, 17]) for (let k = 0; k < 3; k++) expect(p.points[3 * j + k]).toBe(before.points[3 * j + k]);
      // reported cost = cost at the written-back state
      expect(evaluateBACost(p, { loss, lossScalePx: 1.5 }).cost).toBeCloseTo(r.finalCost, 8);
    });
  }

  it('randomised configurations (fixed poses / points, masks, refine flags, priors, losses, solvers)', () => {
    const rng = new Rng(2024);
    const losses: BALossType[] = ['trivial', 'huber', 'cauchy', 'soft_l1'];
    for (let trial = 0; trial < 12; trial++) {
      const nCam = 1 + rng.int(3);
      const s = makeSyntheticBA({
        numPoses: 5 + rng.int(6),
        numPoints: 30 + rng.int(40),
        layout: rng.next() < 0.5 ? 'orbit' : 'strafe',
        trackWindow: 1 + rng.int(4),
        k1: -0.05,
        numCameras: nCam,
        noisePx: 0.7,
        seed: 300 + trial,
      });
      const p = s.problem;
      const nP = p.poses.length,
        nX = p.points.length / 3;
      p.poseFixed = Uint8Array.from({ length: nP }, () => (rng.next() < 0.25 ? 1 : 0));
      p.translationFixedMask = Uint8Array.from({ length: nP }, () => (rng.next() < 0.3 ? rng.int(8) : 0));
      p.pointFixed = Uint8Array.from({ length: nX }, () => (rng.next() < 0.1 ? 1 : 0));
      for (let c = 0; c < nCam; c++) p.cameraRefine[c] = { focal: rng.next() < 0.6, principalPoint: rng.next() < 0.4, distortion: rng.next() < 0.5 };
      if (rng.next() < 0.5) p.obsWeight = Float32Array.from({ length: p.obsPose.length }, () => 0.2 + 2 * rng.next());
      p.centerPriors = [];
      p.relRotationPriors = [];
      for (let k = 0; k < rng.int(4); k++) {
        const i = rng.int(nP);
        p.centerPriors.push({ pose: i, center: poseCentre(p.poses[i]).map((v) => v + 0.05 * rng.gaussian()), sigma: 0.02 + rng.next() });
      }
      for (let k = 0; k < rng.int(4); k++) {
        const i = rng.int(nP),
          j = (i + 1 + rng.int(nP - 1)) % nP;
        const w = [0.02 * rng.gaussian(), 0.02 * rng.gaussian(), 0.02 * rng.gaussian()];
        p.relRotationPriors.push({ i, j, R_ji: mat3Mul(rotvecToMat3(w), mat3MulTranspose(p.poses[j].R, p.poses[i].R)), sigma: 0.005 + 0.05 * rng.next() });
      }
      perturbBA(s, { rotSigma: 0.003, centreSigma: 0.01, pointSigma: 0.01, focalScale: 0.01, k1Delta: 0.01, seed: 500 + trial });
      const loss = losses[rng.int(4)];
      const lambda = Math.pow(10, rng.uniform(-3, 0)); // gauge may be free: keep λ moderate
      const solver = rng.next() < 0.5 ? 'dense_schur' : 'pcg_schur';
      const before = cloneBAProblem(p);
      const ref = bruteForceStep(before, loss, 1.3, lambda);
      const r = bundleAdjust(p, { loss, lossScalePx: 1.3, initialLambda: lambda, maxIterations: 1, linearSolver: solver, pcgTolerance: 1e-15, pcgMaxIterations: 20000 });
      const label = `trial ${trial} (${loss}, λ=${lambda.toExponential(1)}, ${solver})`;
      expect(r.numReducedParams + 3 * r.numFreePoints, label).toBe(ref.idx.n);
      if (ref.idx.n === 0) continue;
      expect(r.successfulIterations, label).toBe(1);
      const d = takenStep(before, p, ref.idx);
      let maxRef = 0,
        maxErr = 0;
      for (let k = 0; k < ref.idx.n; k++) {
        maxRef = Math.max(maxRef, Math.abs(ref.delta[k]));
        maxErr = Math.max(maxErr, Math.abs(d[k] - ref.delta[k]));
      }
      expect(maxErr / maxRef, label).toBeLessThan(1e-6);
    }
  });

  it('gain ratio uses the exact undamped model decrease (strongly non-linear first step, ρ ≈ 0.5)', () => {
    for (const solver of ['dense_schur', 'pcg_schur'] as const) {
      const s = makeSyntheticBA({ numPoses: 8, numPoints: 60, layout: 'orbit', trackWindow: 3, k1: -0.1, k2: 0.02, numCameras: 2, noisePx: 1, seed: 5 });
      fixGaugeTwoPoses(s.problem);
      s.problem.cameraRefine[0] = { focal: true, principalPoint: true, distortion: true };
      perturbBA(s, { rotSigma: 0.2, centreSigma: 0.6, pointSigma: 0.6, focalScale: 0.05, k1Delta: 0.03, seed: 9 });
      const before = cloneBAProblem(s.problem);
      const lambda = 1e-4;
      const ref = bruteForceStep(before, 'trivial', 1, lambda);
      const r = bundleAdjust(s.problem, { loss: 'trivial', initialLambda: lambda, maxIterations: 1, linearSolver: solver, pcgTolerance: 1e-14, pcgMaxIterations: 10000 });
      expect(r.successfulIterations).toBe(1);
      const n = ref.idx.n;
      let gd = 0,
        dHd = 0;
      for (let a = 0; a < n; a++) {
        gd += ref.g[a] * ref.delta[a];
        for (let c = 0; c < n; c++) dHd += ref.delta[a] * ref.H[a * n + c] * ref.delta[c];
      }
      const rho = (r.initialCost - r.finalCost) / (-gd - 0.5 * dHd);
      const factor = 1 - Math.pow(2 * rho - 1, 3);
      expect(factor).toBeGreaterThan(0.4); // informative: not clamped at 1/3
      expect(Math.abs(r.finalLambda / (lambda * factor) - 1)).toBeLessThan(1e-6);
    }
  });

  it('rank-deficient point blocks (single-view points) at minimum λ are frozen consistently', () => {
    const s = adversarialProblem(19, 0);
    const p = s.problem;
    // append 4 single-view points seen by free poses 0, 2, 4, 5
    const nX = p.points.length / 3;
    const extraPose = [0, 2, 4, 5];
    const pts = new Float64Array(p.points.length + 3 * extraPose.length);
    pts.set(p.points);
    extraPose.forEach((i, k) => {
      const C = poseCentre(p.poses[i]);
      const R = p.poses[i].R;
      // 3 m in front of the camera, slightly off-axis
      for (let c = 0; c < 3; c++) pts[p.points.length + 3 * k + c] = C[c] + R[6 + c] * 3 + R[c] * 0.2 * (k - 1.5);
    });
    p.points = pts;
    p.pointFixed = Uint8Array.from([...p.pointFixed!, 0, 0, 0, 0]);
    p.obsPose = Uint32Array.from([...p.obsPose, ...extraPose]);
    p.obsPoint = Uint32Array.from([...p.obsPoint, ...extraPose.map((_, k) => nX + k)]);
    p.obsWeight = Float32Array.from([...p.obsWeight!, 1, 1, 1, 1]);
    const K = camK(p);
    const r2 = new Float64Array(2);
    const uv = Array.from(p.obsUV);
    extraPose.forEach((i, k) => {
      radialResidual(p.poses[i].R, 0, p.poses[i].t, 0, p.points, 3 * (nX + k), K, 5 * p.imageCamera[i], 0, 0, r2);
      uv.push(r2[0] + 3, r2[1] - 2); // 3.6 px off: non-zero gradient across the ray
    });
    p.obsUV = Float64Array.from(uv);
    const before = cloneBAProblem(p);
    const frozen = new Set([nX, nX + 1, nX + 2, nX + 3]);
    const ref = bruteForceStep(before, 'trivial', 1, 1e-16, frozen);
    const r = bundleAdjust(p, { loss: 'trivial', initialLambda: 1e-16, maxIterations: 1, linearSolver: 'dense_schur' });
    expect(r.successfulIterations).toBe(1);
    for (const j of frozen) for (let c = 0; c < 3; c++) expect(p.points[3 * j + c]).toBe(before.points[3 * j + c]);
    const d = takenStep(before, p, ref.idx);
    let maxRef = 0,
      maxErr = 0;
    for (let k = 0; k < ref.idx.n; k++) {
      maxRef = Math.max(maxRef, Math.abs(ref.delta[k]));
      maxErr = Math.max(maxErr, Math.abs(d[k] - ref.delta[k]));
    }
    expect(maxErr / maxRef).toBeLessThan(1e-6);
    // and at a normal λ the single-view points move (they are not permanently frozen)
    const q = cloneBAProblem(before);
    bundleAdjust(q, { loss: 'trivial', initialLambda: 1e-3, maxIterations: 1 });
    let moved = 0;
    for (const j of frozen) moved += Math.hypot(q.points[3 * j] - before.points[3 * j], q.points[3 * j + 1] - before.points[3 * j + 1]);
    expect(moved).toBeGreaterThan(1e-4);
  });

  it('gradient used by the solver equals the brute-force Jᵀr', () => {
    const s = adversarialProblem(11);
    const p = s.problem;
    const ref = bruteForceStep(p, 'cauchy', 1.5, 1);
    const g = computeBAGradient(p, { loss: 'cauchy', lossScalePx: 1.5 });
    const flat = new Float64Array(ref.idx.n);
    for (let k = 0; k < ref.idx.pose.length; k++) if (ref.idx.pose[k] >= 0) flat[ref.idx.pose[k]] = g.poses[k];
    for (let k = 0; k < ref.idx.cam.length; k++) if (ref.idx.cam[k] >= 0) flat[ref.idx.cam[k]] = g.cameras[k];
    for (let k = 0; k < ref.idx.point.length; k++) if (ref.idx.point[k] >= 0) flat[ref.idx.point[k]] = g.points[k];
    for (let k = 0; k < ref.idx.n; k++) expect(Math.abs(flat[k] - ref.g[k])).toBeLessThanOrEqual(1e-9 * (1 + Math.abs(ref.g[k])));
  });
});

describe('scale invariance (tiny objects and large scenes)', () => {
  /** Similarity-scale the whole scene (points and translations) by s. */
  const scaleScene = (p: BAProblem, s: number): void => {
    for (let k = 0; k < p.points.length; k++) p.points[k] *= s;
    for (const ps of p.poses) for (let k = 0; k < 3; k++) ps.t[k] *= s;
  };
  it('converges to the same RMS / intrinsics at scene scales 1e-4 … 1e4', () => {
    const results: { rms: number; f: number; it: number }[] = [];
    for (const scale of [1, 1e-4, 1e4]) {
      const s = makeSyntheticBA({ numPoses: 16, numPoints: 800, layout: 'orbit', trackWindow: 4, k1: -0.05, noisePx: 0.5, seed: 77 });
      fixGaugeTwoPoses(s.problem);
      s.problem.cameraRefine[0] = { focal: true, principalPoint: false, distortion: true };
      perturbBA(s, { rotSigma: 0.005, centreSigma: 0.03, pointSigma: 0.03, focalScale: 0.05, k1Delta: 0.02 });
      scaleScene(s.problem, scale);
      const r = bundleAdjust(s.problem, { loss: 'trivial', maxIterations: 100 });
      expect(r.success).toBe(true);
      results.push({ rms: r.finalRmsPx, f: s.problem.cameras[0].f, it: r.iterations });
    }
    for (const q of results.slice(1)) {
      expect(Math.abs(q.rms / results[0].rms - 1)).toBeLessThan(1e-4);
      expect(Math.abs(q.f / results[0].f - 1)).toBeLessThan(1e-5);
    }
  });
});

describe('degenerate geometry', () => {
  it('pure rotation (zero baseline, depths unobservable): finite, monotone, rotations recovered', () => {
    // all cameras at the origin, attitudes spread over ±0.4 rad; points 2–50 m away
    const rng = new Rng(81);
    const nP = 10,
      nX = 400;
    const cam = { width: 1920, height: 1440, f: 1400, cx: 959.5, cy: 719.5, k1: 0, k2: 0 };
    const gtR = Array.from({ length: nP }, (_, i) => rotvecToMat3([0.1 * rng.gaussian(), -0.4 + (0.8 * i) / (nP - 1), 0.05 * rng.gaussian()]));
    const pts = new Float64Array(3 * nX);
    const obsPose: number[] = [],
      obsPoint: number[] = [],
      uv: number[] = [];
    const zero = new Float64Array(3);
    const K = Float64Array.of(cam.f, cam.cx, cam.cy, 0, 0);
    const r2 = new Float64Array(2);
    for (let j = 0; j < nX; j++) {
      const a = rng.int(nP);
      const d = 2 + 48 * rng.next();
      const xc = [d * rng.uniform(-0.6, 0.6), d * rng.uniform(-0.45, 0.45), d];
      for (let c = 0; c < 3; c++) pts[3 * j + c] = gtR[a][c] * xc[0] + gtR[a][3 + c] * xc[1] + gtR[a][6 + c] * xc[2];
      for (let i = 0; i < nP; i++) {
        radialResidual(gtR[i], 0, zero, 0, pts, 3 * j, K, 0, 0, 0, r2);
        const z = gtR[i][6] * pts[3 * j] + gtR[i][7] * pts[3 * j + 1] + gtR[i][8] * pts[3 * j + 2];
        if (z > 0 && r2[0] > 0 && r2[0] < 1919 && r2[1] > 0 && r2[1] < 1439) {
          obsPose.push(i);
          obsPoint.push(j);
          uv.push(r2[0] + 0.3 * rng.gaussian(), r2[1] + 0.3 * rng.gaussian());
        }
      }
    }
    const p: BAProblem = {
      cameras: [{ ...cam }],
      cameraRefine: [{ focal: false, principalPoint: false, distortion: false }],
      poses: gtR.map((R) => ({ R: mat3Mul(rotvecToMat3([0.01 * rng.gaussian(), 0.01 * rng.gaussian(), 0.01 * rng.gaussian()]), R), t: new Float64Array(3) })),
      imageCamera: new Uint32Array(nP),
      poseFixed: new Uint8Array(nP),
      points: pts.map((v) => v * (1 + 0.05 * rng.gaussian())),
      obsPose: Uint32Array.from(obsPose),
      obsPoint: Uint32Array.from(obsPoint),
      obsUV: Float64Array.from(uv),
    };
    p.poseFixed[0] = 1;
    p.poses[0].R.set(gtR[0]);
    for (const [loss, solver] of [
      ['trivial', 'dense_schur'],
      ['cauchy', 'dense_schur'],
      ['cauchy', 'pcg_schur'],
    ] as const) {
      const q = cloneBAProblem(p);
      const costs: number[] = [];
      const r = bundleAdjust(q, { loss, linearSolver: solver, maxIterations: 60, onIteration: (_i, c) => costs.push(c) });
      expect(r.success).toBe(true);
      for (const v of q.points) expect(Number.isFinite(v)).toBe(true);
      for (const ps of q.poses) for (const v of [...ps.R, ...ps.t]) expect(Number.isFinite(v)).toBe(true);
      for (let k = 1; k < costs.length; k++) expect(costs[k]).toBeLessThanOrEqual(costs[k - 1]);
      expect(r.finalRmsPx).toBeLessThan(0.4);
      let maxRot = 0;
      for (let i = 0; i < nP; i++) maxRot = Math.max(maxRot, rotationAngleBetween(q.poses[i].R, gtR[i]));
      expect(maxRot).toBeLessThan(5e-4);
    }
  });

  it('pose graph only (no points / observations): centre + relative-rotation priors are solved exactly', () => {
    const s = makeSyntheticBA({ numPoses: 6, numPoints: 0, layout: 'orbit', seed: 82 });
    const p = s.problem;
    expect(p.obsPose.length).toBe(0);
    p.centerPriors = s.gtPoses.map((g, i) => ({ pose: i, center: poseCentre(g), sigma: 0.1 }));
    p.relRotationPriors = [];
    for (let i = 0; i + 1 < 6; i++) p.relRotationPriors.push({ i, j: i + 1, R_ji: mat3MulTranspose(s.gtPoses[i + 1].R, s.gtPoses[i].R), sigma: 0.01 });
    p.poseFixed[0] = 1;
    perturbBA(s, { rotSigma: 0.2, centreSigma: 0.5 });
    const r = bundleAdjust(p, { loss: 'trivial', maxIterations: 100, functionTolerance: 0, parameterTolerance: 0 });
    expect(r.success).toBe(true);
    expect(r.finalCost).toBeLessThan(1e-18);
    const e = syntheticErrors(s);
    expect(e.maxRotErr).toBeLessThan(1e-9);
    expect(e.maxCentreErr).toBeLessThan(1e-9);
  });

  it('relative-rotation prior with an initial error close to π converges', () => {
    const s = makeSyntheticBA({ numPoses: 3, numPoints: 0, layout: 'orbit', seed: 83 });
    const p = s.problem;
    p.poseFixed[0] = 1;
    p.poseFixed[2] = 1;
    p.relRotationPriors = [{ i: 0, j: 1, R_ji: mat3MulTranspose(s.gtPoses[1].R, s.gtPoses[0].R), sigma: 0.01 }];
    p.poses[1].R.set(mat3Mul(rotvecToMat3([0, 3.0, 0.3]), p.poses[1].R)); // ≈ 3.01 rad off
    const r = bundleAdjust(p, { loss: 'trivial', maxIterations: 100, functionTolerance: 0, parameterTolerance: 0 });
    expect(r.success).toBe(true);
    expect(rotationAngleBetween(p.poses[1].R, s.gtPoses[1].R)).toBeLessThan(1e-9);
  });

  it('poses with rotation angle exactly π (e.g. diag(1, −1, −1)) and a point at the camera centre', () => {
    const s = makeSyntheticBA({ numPoses: 6, numPoints: 200, layout: 'orbit', trackWindow: 2, noisePx: 0.3, seed: 84 });
    const p = s.problem;
    // rotate the whole world by π about x: every pose rotation is multiplied by diag(1,−1,−1)
    const F = Float64Array.of(1, 0, 0, 0, -1, 0, 0, 0, -1);
    for (const ps of [...p.poses, ...s.gtPoses]) ps.R.set(mat3Mul(ps.R, F));
    for (const P of [p.points, s.gtPoints]) for (let j = 0; j < P.length; j += 3) (P[j + 1] = -P[j + 1]), (P[j + 2] = -P[j + 2]);
    fixGaugeTwoPoses(p);
    // a point exactly at the centre of camera 3 (Z_c = 0) observed by that camera
    const j0 = p.obsPoint[p.obsPose.findIndex((i) => i === 3)];
    p.points.set(poseCentre(p.poses[3]), 3 * j0);
    perturbBA(s, { rotSigma: 0.005, centreSigma: 0.01, pointSigma: 0.01 });
    const r = bundleAdjust(p, { maxIterations: 50 });
    expect(r.success).toBe(true);
    for (const v of p.points) expect(Number.isFinite(v)).toBe(true);
    expect(r.finalCost).toBeLessThan(r.initialCost);
    expect(syntheticErrors(s).maxRotErr).toBeLessThan(2e-3);
  });
});

describe('robustness', () => {
  it('40 % gross outliers: Cauchy keeps intrinsics and poses at inlier accuracy', () => {
    const s = makeSyntheticBA({ numPoses: 24, numPoints: 2000, layout: 'orbit', trackWindow: 6, k1: -0.05, noisePx: 0.5, outlierFraction: 0.4, seed: 91 });
    fixGaugeTwoPoses(s.problem);
    s.problem.cameraRefine[0] = { focal: true, principalPoint: false, distortion: true };
    perturbBA(s, { rotSigma: 0.003, centreSigma: 0.02, pointSigma: 0.02, focalScale: 0.03, k1Delta: 0.01 });
    const r = bundleAdjust(s.problem, { loss: 'cauchy', maxIterations: 100 });
    expect(r.success).toBe(true);
    const err = reprojectionErrors(s.problem);
    let se = 0,
      n = 0;
    for (let o = 0; o < err.length; o++)
      if (!s.outlier[o]) {
        se += err[o] * err[o];
        n++;
      }
    expect(Math.sqrt(se / n)).toBeLessThan(0.8);
    const e = syntheticErrors(s);
    expect(e.maxFocalRelErr).toBeLessThan(0.005);
    expect(e.maxRotErr).toBeLessThan(2e-3);
  });

  it('weights and loss scale are consistent: (k·w, k·a) gives the same minimiser and k²·cost', () => {
    const base = adversarialProblem(13).problem;
    base.centerPriors = undefined;
    base.relRotationPriors = undefined;
    const a = cloneBAProblem(base),
      b = cloneBAProblem(base);
    const k = 3;
    b.obsWeight = b.obsWeight!.map((w) => w * k);
    for (const loss of ['huber', 'cauchy', 'soft_l1'] as const) {
      const ca = evaluateBACost(a, { loss, lossScalePx: 2 }).cost;
      const cb = evaluateBACost(b, { loss, lossScalePx: 2 * k }).cost;
      expect(cb / ca).toBeCloseTo(k * k, 6);
    }
    const ra = bundleAdjust(a, { loss: 'cauchy', lossScalePx: 2, maxIterations: 30 });
    const rb = bundleAdjust(b, { loss: 'cauchy', lossScalePx: 2 * k, maxIterations: 30 });
    expect(rb.finalCost / ra.finalCost).toBeCloseTo(k * k, 4);
    let d = 0;
    for (let q = 0; q < a.points.length; q++) d = Math.max(d, Math.abs(a.points[q] - b.points[q]));
    expect(d).toBeLessThan(1e-6);
  });

  it('NaN / ±Infinity / negative / overflowing weights and NaN / huge pixels are ignored; results stay finite', () => {
    const s = makeSyntheticBA({ numPoses: 8, numPoints: 300, layout: 'orbit', trackWindow: 3, noisePx: 0.3, seed: 92 });
    const p = s.problem;
    fixGaugeTwoPoses(p);
    perturbBA(s, { rotSigma: 0.003, centreSigma: 0.01, pointSigma: 0.01 });
    p.obsWeight = new Float32Array(p.obsPose.length).fill(1);
    p.obsWeight[0] = NaN;
    p.obsWeight[1] = Infinity;
    p.obsWeight[2] = -1;
    p.obsWeight[3] = -Infinity;
    p.obsUV[2 * 4] = Infinity;
    p.obsUV[2 * 5 + 1] = NaN;
    p.obsUV[2 * 6] = 1e200; // finite residual, but its square overflows
    p.obsWeight[7] = 1e300; // finite weight, weighted square overflows
    for (const loss of ['trivial', 'cauchy'] as const) {
      const q = cloneBAProblem(p);
      const r = bundleAdjust(q, { loss });
      expect(r.success).toBe(true);
      expect(Number.isFinite(r.finalCost)).toBe(true);
      expect(Number.isFinite(r.finalRmsPx)).toBe(true);
      for (const v of q.points) expect(Number.isFinite(v)).toBe(true);
      expect(syntheticErrors({ ...s, problem: q }).maxRotErr).toBeLessThan(2e-3);
    }
  });

  it('extreme loss scales do not produce NaN costs (tiny scale ≈ L1-like, huge scale ≈ trivial)', () => {
    const s = makeSyntheticBA({ numPoses: 6, numPoints: 150, layout: 'orbit', trackWindow: 2, noisePx: 0.5, seed: 93 });
    fixGaugeTwoPoses(s.problem);
    perturbBA(s, { rotSigma: 0.003, centreSigma: 0.01, pointSigma: 0.01 });
    const trivial = evaluateBACost(s.problem, { loss: 'trivial' }).cost;
    for (const loss of ['huber', 'cauchy', 'soft_l1'] as const) {
      for (const scale of [1e-200, 1e-30, 1e30, 1e200]) {
        const c = evaluateBACost(s.problem, { loss, lossScalePx: scale }).cost;
        expect(Number.isFinite(c)).toBe(true);
        if (scale >= 1e30) expect(c / trivial).toBeCloseTo(1, 6);
        const q = cloneBAProblem(s.problem);
        const r = bundleAdjust(q, { loss, lossScalePx: scale, maxIterations: 5 });
        expect(r.success).toBe(true);
        expect(r.finalCost).toBeLessThanOrEqual(r.initialCost);
      }
    }
  });

  it('duplicate observations of the same (pose, point) are handled like a larger weight', () => {
    const s = makeSyntheticBA({ numPoses: 6, numPoints: 120, layout: 'orbit', trackWindow: 2, noisePx: 0.4, seed: 94 });
    const p = s.problem;
    fixGaugeTwoPoses(p);
    perturbBA(s, { rotSigma: 0.003, centreSigma: 0.01, pointSigma: 0.01 });
    const dup = cloneBAProblem(p);
    const nO = p.obsPose.length;
    // 4 copies ≡ weight 2 (exact in Float32, unlike √2)
    dup.obsPose = Uint32Array.from([...p.obsPose, ...p.obsPose, ...p.obsPose, ...p.obsPose]);
    dup.obsPoint = Uint32Array.from([...p.obsPoint, ...p.obsPoint, ...p.obsPoint, ...p.obsPoint]);
    dup.obsUV = Float64Array.from([...p.obsUV, ...p.obsUV, ...p.obsUV, ...p.obsUV]);
    const w = cloneBAProblem(p);
    w.obsWeight = new Float32Array(nO).fill(2);
    // identical normal equations ⇒ identical steps (up to rounding)
    for (const iters of [1, 3]) {
      const a = cloneBAProblem(dup),
        b = cloneBAProblem(w);
      // (only for the trivial loss: ½ρ(s) + ½ρ(s) ≠ ½ρ(2s) for robust losses)
      const rd = bundleAdjust(a, { loss: 'trivial', maxIterations: iters });
      const rw = bundleAdjust(b, { loss: 'trivial', maxIterations: iters });
      expect(rd.successfulIterations).toBe(iters);
      expect(Math.abs(rd.finalCost / rw.finalCost - 1)).toBeLessThan(1e-12);
      for (let k = 0; k < p.points.length; k++) expect(Math.abs(a.points[k] - b.points[k])).toBeLessThan(1e-10);
    }
  });
});

describe('edge cases', () => {
  it('completely empty problem', () => {
    const p: BAProblem = {
      cameras: [],
      cameraRefine: [],
      poses: [],
      imageCamera: new Uint32Array(0),
      poseFixed: new Uint8Array(0),
      points: new Float64Array(0),
      obsPose: new Uint32Array(0),
      obsPoint: new Uint32Array(0),
      obsUV: new Float64Array(0),
    };
    const r = bundleAdjust(p);
    expect(r.success).toBe(true);
    expect(r.termination).toBe('no_free_parameters');
    expect(r.initialCost).toBe(0);
    expect(computeBAGradient(p).cost).toBe(0);
  });

  it('only intrinsics free (all poses and points fixed): recovers f and k1', () => {
    const s = makeSyntheticBA({ numPoses: 10, numPoints: 500, layout: 'orbit', trackWindow: 3, k1: -0.06, noisePx: 0.2, seed: 95 });
    const p = s.problem;
    p.poseFixed.fill(1);
    p.pointFixed = new Uint8Array(p.points.length / 3).fill(1);
    p.cameraRefine[0] = { focal: true, principalPoint: true, distortion: true };
    perturbBA(s, { focalScale: 0.08, k1Delta: 0.05 });
    const r = bundleAdjust(p, { loss: 'trivial' });
    expect(r.success).toBe(true);
    expect(r.numFreePoints).toBe(0);
    expect(r.numReducedParams).toBe(5);
    expect(Math.abs(p.cameras[0].f / s.gtCameras[0].f - 1)).toBeLessThan(1e-3);
    expect(Math.abs(p.cameras[0].k1 + 0.06)).toBeLessThan(2e-3);
  });

  it('a free pose with no observations and no priors stays put (no NaN, no singular solve)', () => {
    const s = makeSyntheticBA({ numPoses: 8, numPoints: 300, layout: 'orbit', trackWindow: 2, noisePx: 0.3, seed: 96 });
    const p = s.problem;
    fixGaugeTwoPoses(p);
    const keep: number[] = [];
    for (let o = 0; o < p.obsPose.length; o++) if (p.obsPose[o] !== 5) keep.push(o);
    p.obsPose = Uint32Array.from(keep.map((o) => s.problem.obsPose[o]));
    p.obsUV = Float64Array.from(keep.flatMap((o) => [s.problem.obsUV[2 * o], s.problem.obsUV[2 * o + 1]]));
    p.obsPoint = Uint32Array.from(keep.map((o) => s.problem.obsPoint[o]));
    perturbBA(s, { rotSigma: 0.003, centreSigma: 0.01, pointSigma: 0.01 });
    const R5 = p.poses[5].R.slice(),
      t5 = p.poses[5].t.slice();
    for (const solver of ['dense_schur', 'pcg_schur'] as const) {
      const q = cloneBAProblem(p);
      const r = bundleAdjust(q, { loss: 'trivial', linearSolver: solver });
      expect(r.success).toBe(true);
      expect(r.finalCost).toBeLessThan(r.initialCost);
      expect(Array.from(q.poses[5].R)).toEqual(Array.from(R5));
      expect(Array.from(q.poses[5].t)).toEqual(Array.from(t5));
    }
  });

  it('rotation matrices stay orthonormal after many LM steps', () => {
    const s = makeSyntheticBA({ numPoses: 10, numPoints: 400, layout: 'orbit', trackWindow: 3, noisePx: 0.5, seed: 97 });
    const p = s.problem;
    fixGaugeTwoPoses(p);
    perturbBA(s, { rotSigma: 0.05, centreSigma: 0.05, pointSigma: 0.05 });
    // many short BA calls, like the live mapper
    for (let k = 0; k < 40; k++) bundleAdjust(p, { maxIterations: 3, functionTolerance: 0 });
    for (const ps of p.poses) {
      const RRt = mat3MulTranspose(ps.R, ps.R);
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) expect(Math.abs(RRt[3 * r + c] - (r === c ? 1 : 0))).toBeLessThan(1e-13);
    }
  });
});

describe('loss numerics', () => {
  it('soft_l1 ρ(s) is accurate for s ≪ b (no cancellation) and finite for s ≫ b', () => {
    const k = lossKind('soft_l1');
    // series: ρ = s − s²/(4b) + …
    for (const [s, b] of [
      [1e-20, 1],
      [0.25, 1e8],
      [1e-3, 1e30],
    ]) {
      const series = s - (s * s) / (4 * b);
      expect(Math.abs(lossRho(k, s, b) / series - 1)).toBeLessThan(1e-14);
    }
    expect(lossRho(k, 1e200, 1e-150) / (2 * Math.sqrt(1e200 * 1e-150)) - 1).toBeLessThan(1e-12);
    const kc = lossKind('cauchy');
    expect(lossRho(kc, 1e200, 1e-150) / (1e-150 * (Math.log(1e200) - Math.log(1e-150))) - 1).toBeLessThan(1e-12);
    for (const name of ['huber', 'cauchy', 'soft_l1'] as const) {
      const kk = lossKind(name);
      expect(Math.abs(lossRho(kk, 1e-12, 1e6) / 1e-12 - 1)).toBeLessThan(1e-12);
    }
  });
});

describe('LM termination details', () => {
  it('a step below parameterTolerance is still taken when it decreases the cost, then the run stops', () => {
    const s = makeSyntheticBA({ numPoses: 8, numPoints: 300, layout: 'orbit', trackWindow: 3, noisePx: 0.5, seed: 101 });
    fixGaugeTwoPoses(s.problem);
    perturbBA(s, { rotSigma: 0.002, centreSigma: 0.005, pointSigma: 0.005 });
    // tolerance so large that the very first step counts as "small"
    const r = bundleAdjust(s.problem, { loss: 'trivial', parameterTolerance: 10 });
    expect(r.termination).toBe('parameter_tolerance');
    expect(r.iterations).toBe(1);
    expect(r.successfulIterations).toBe(1);
    expect(r.finalCost).toBeLessThan(0.5 * r.initialCost);
    expect(evaluateBACost(s.problem, { loss: 'trivial' }).cost).toBeCloseTo(r.finalCost, 8);
  });

  it('a cancel token that is already set returns before any work and leaves the problem untouched', () => {
    const s = makeSyntheticBA({ numPoses: 6, numPoints: 100, layout: 'orbit', trackWindow: 2, seed: 102 });
    perturbBA(s, { pointSigma: 0.01 });
    const before = cloneBAProblem(s.problem);
    const r = bundleAdjust(s.problem, { cancel: { cancelled: true } });
    expect(r.termination).toBe('cancelled');
    expect(r.success).toBe(true);
    expect(r.iterations).toBe(0);
    expect(r.finalCost).toBe(r.initialCost);
    expect(Array.from(s.problem.points)).toEqual(Array.from(before.points));
  });
});

describe('extreme depths', () => {
  it('points at 1e9 … 1e15 (sky / horizon features) mixed into a normal scene: finite, rotations still accurate', () => {
    const s = makeSyntheticBA({ numPoses: 12, numPoints: 600, layout: 'strafe', trackWindow: 3, radius: 0.5, noisePx: 0.4, seed: 111 });
    const p = s.problem;
    fixGaugeTwoPoses(p);
    // move every 10th point (and its ground truth) along its anchor ray to 1e9 … 1e15
    const C0 = poseCentre(s.gtPoses[0]);
    for (let j = 0; j < p.points.length / 3; j += 10) {
      const dist = Math.pow(10, 9 + (j % 7));
      const d = [0, 1, 2].map((k) => s.gtPoints[3 * j + k] - C0[k]);
      const n = Math.hypot(...d);
      for (let k = 0; k < 3; k++) s.gtPoints[3 * j + k] = p.points[3 * j + k] = C0[k] + (d[k] / n) * dist;
    }
    // re-synthesise those observations (noise-free) for the new positions
    const K = camK(p);
    const r2 = new Float64Array(2);
    for (let o = 0; o < p.obsPose.length; o++) {
      const j = p.obsPoint[o];
      if (j % 10) continue;
      const i = p.obsPose[o];
      radialResidual(s.gtPoses[i].R, 0, s.gtPoses[i].t, 0, s.gtPoints, 3 * j, K, 0, 0, 0, r2);
      p.obsUV[2 * o] = r2[0];
      p.obsUV[2 * o + 1] = r2[1];
    }
    perturbBA(s, { rotSigma: 0.003, centreSigma: 0.01, pointSigma: 0.01 });
    for (const solver of ['dense_schur', 'pcg_schur'] as const) {
      const q = cloneBAProblem(p);
      const r = bundleAdjust(q, { maxIterations: 100, linearSolver: solver });
      expect(r.success).toBe(true);
      for (const v of q.points) expect(Number.isFinite(v)).toBe(true);
      expect(r.finalRmsPx).toBeLessThan(0.6);
      expect(syntheticErrors({ ...s, problem: q }).maxRotErr).toBeLessThan(1e-3);
    }
  });
});
