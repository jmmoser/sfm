/**
 * Bundle adjustment: Levenberg–Marquardt with analytic Jacobians and the Schur
 * complement on points (docs/ARCHITECTURE.md §4.3).
 *
 * Method (Triggs et al., "Bundle Adjustment — A Modern Synthesis", 2000; Agarwal et al.,
 * "Bundle Adjustment in the Large", ECCV 2010; Ceres Solver's trust-region LM):
 *  - Normal equations of the IRLS-weighted Gauss–Newton model
 *        [U  W] [δy]   [−g_y]
 *        [Wᵀ V] [δx] = [−g_x]
 *    with y = camera-side parameters (free poses [δω, δt] and free intrinsics per
 *    camera) and x = points. LM damping (Marquardt): H + λ·D with
 *    D = clamp(diag(H), 1e-6, 1e32) on both U and V.
 *  - Points are eliminated with 3x3 blocks: S = U − W V⁻¹ Wᵀ, assembled explicitly as
 *    a block-sparse matrix (blocks only where poses share points; intrinsics couple to
 *    all their poses). S is solved directly ('dense_schur': Cholesky in packed skyline /
 *    envelope storage — exact dense Cholesky, but memory and work only inside the row
 *    profile, which is narrow for sequential captures) when the reduced size is ≤ 2400
 *    or the envelope is small, and by block-Jacobi PCG otherwise; then
 *    δx = V⁻¹(−g_x − Wᵀ δy).
 *  - Step control: gain ratio ρ = (F(x) − F(x+δ)) / (m(0) − m(δ)) with the exact
 *    quadratic model m (undamped H). Accept if ρ > 1e-3; then λ ← λ·max(1/3, 1 −
 *    (2ρ − 1)³), ν = 2, else λ ← λ·ν, ν ← 2ν (Nielsen 1999; Madsen, Nielsen &
 *    Tingleff, "Methods for Non-Linear Least Squares Problems", 2004, §3.2).
 *  - Robust loss by IRLS: residual and Jacobian scaled by w·√ρ'(s) (exact gradient of
 *    the reported cost ½Σρ).
 *
 * Memory: no per-observation Jacobians are stored. Per iteration the solver keeps
 * U and S (block-sparse), V (6/point), and W per (point, camera-side block) (18
 * doubles); retries after a rejected step only redo the Schur complement.
 */
import type { CameraIntrinsics, Pose } from '../core/types';
import { rotvecToMat3 } from '../math/rotation';
import { radialResidual, radialResidualJacobian, RJ_SIZE } from './camera-model';
import {
  BlockJacobiPreconditioner,
  BlockSparseSymmetric,
  createPcgWorkspace,
  pcgSolve,
  scatterToSkyline,
  SkylineCholesky,
  skylineEnvelope,
  type PcgWorkspace,
} from './linear';
import { lossKind, lossRho, lossWeight, type LossKind } from './loss';
import { centerPriorResidualJacobian, relRotationPriorResidualJacobian } from './priors';
import {
  DENSE_SCHUR_MAX_SIZE,
  resolveBAOptions,
  validateBAProblem,
  type BAOptions,
  type BAProblem,
  type BAResult,
  type BATermination,
  type ResolvedBAOptions,
} from './problem';

const MIN_DIAG = 1e-6;
const MAX_DIAG = 1e32;
const MIN_LAMBDA = 1e-16;
const MAX_LAMBDA = 1e32;
const MIN_RELATIVE_DECREASE = 1e-3;
const MIN_LOSS_B = 1e-150;
const MAX_LOSS_B = 1e150;
/** Relative pivot threshold of the 3x3 point-block Cholesky (cond ≳ 1e12 ⇒ point skipped this step). */
const V_PIVOT_EPS = 1e-12;
/** A forced 'dense_schur' whose Cholesky envelope exceeds this many entries (144 MB) falls back to PCG. */
const DENSE_HARD_LIMIT_ENTRIES = 18e6;
/** 'auto' also uses the direct solver above DENSE_SCHUR_MAX_SIZE when the envelope is this small (32 MB) … */
const AUTO_SKYLINE_MAX_ENTRIES = 4e6;
/** … and its factorisation costs at most this many multiply-adds (≈ a few hundred PCG iterations). */
const AUTO_SKYLINE_MAX_COST = 3e8;

interface EvalResult {
  cost: number;
  sqErr: number;
}

/**
 * Internal solver state for one problem. Parameters are copied into flat arrays
 * (current + candidate); the problem is only written back by `writeBack`.
 */
class BAEngine {
  readonly nP: number;
  readonly nC: number;
  readonly nX: number;
  readonly nO: number;
  // Parameters: current and candidate (swapped on acceptance).
  R: Float64Array;
  T: Float64Array;
  K: Float64Array;
  X: Float64Array;
  cR: Float64Array;
  cT: Float64Array;
  cK: Float64Array;
  cX: Float64Array;

  readonly problem: BAProblem;
  readonly opts: ResolvedBAOptions;
  readonly lossKind: LossKind;
  readonly lossB: number;
  readonly obsValid: Uint8Array;
  readonly numValidObs: number;

  // ---- structure (built by buildStructure) ----
  poseBlock!: Int32Array;
  camBlock!: Int32Array;
  tMask!: Uint8Array;
  camNK!: Int32Array;
  camIdx!: Int32Array;
  pointFree!: Uint8Array;
  numFreePoints = 0;
  numPoseBlocks = 0;
  ptObsStart!: Int32Array;
  ptObs!: Int32Array;
  ptBlkStart!: Int32Array;
  ptBlk!: Int32Array;
  obsKp!: Int32Array;
  obsKc!: Int32Array;
  pairStart!: Int32Array;
  pairOff!: Int32Array;
  S!: BlockSparseSymmetric;
  U!: Float64Array;
  diagOff!: Int32Array;
  poseCamOff!: Int32Array;
  relOff!: Int32Array;
  relSwap!: Uint8Array;
  pinned!: Int32Array;
  N = 0; // scalar size of the camera-side system (incl. pinned)
  nFree = 0; // free camera-side parameters (excl. pinned)
  maxM = 0;

  // ---- linearisation / step ----
  gY!: Float64Array;
  gX!: Float64Array;
  W!: Float64Array;
  V!: Float64Array;
  Vinv!: Float64Array;
  ptSkip!: Uint8Array;
  rhs!: Float64Array;
  dy!: Float64Array;
  dx!: Float64Array;
  Ysc!: Float64Array;
  readonly jac = new Float64Array(RJ_SIZE);
  readonly jk = new Float64Array(10);
  readonly res2 = new Float64Array(2);
  readonly rot = new Float64Array(9);
  readonly pr = new Float64Array(3);
  readonly pJ = new Float64Array(18);
  readonly pJi = new Float64Array(9);
  readonly pJj = new Float64Array(9);

  // ---- linear solver ----
  useDense = true;
  denseMap!: Int32Array;
  sky: SkylineCholesky | null = null;
  denseB: Float64Array | null = null;
  precond: BlockJacobiPreconditioner | null = null;
  pcgWs: PcgWorkspace | null = null;
  pcgIterations = 0;

  constructor(problem: BAProblem, opts: ResolvedBAOptions) {
    this.problem = problem;
    this.opts = opts;
    this.lossKind = lossKind(opts.loss);
    // b = a², kept in [1e-150, 1e150]: b → 0 / ∞ would give 0·∞ = NaN costs (b·log1p(s/b)).
    this.lossB = Math.min(MAX_LOSS_B, Math.max(MIN_LOSS_B, opts.lossScalePx * opts.lossScalePx));
    const nP = (this.nP = problem.poses.length);
    const nC = (this.nC = problem.cameras.length);
    const nX = (this.nX = problem.points.length / 3);
    const nO = (this.nO = problem.obsPose.length);
    this.R = new Float64Array(9 * nP);
    this.T = new Float64Array(3 * nP);
    for (let i = 0; i < nP; i++) {
      const ps = problem.poses[i];
      for (let k = 0; k < 9; k++) this.R[9 * i + k] = ps.R[k];
      for (let k = 0; k < 3; k++) this.T[3 * i + k] = ps.t[k];
    }
    this.K = new Float64Array(5 * nC);
    for (let c = 0; c < nC; c++) {
      const cam = problem.cameras[c];
      this.K[5 * c] = cam.f;
      this.K[5 * c + 1] = cam.cx;
      this.K[5 * c + 2] = cam.cy;
      this.K[5 * c + 3] = cam.k1;
      this.K[5 * c + 4] = cam.k2;
    }
    this.X = Float64Array.from(problem.points);
    this.cR = this.R.slice();
    this.cT = this.T.slice();
    this.cK = this.K.slice();
    this.cX = this.X.slice();
    // Observations whose (weighted) residual is not finite at the input are excluded for the whole run.
    this.obsValid = new Uint8Array(nO);
    const res = this.res2;
    const ip = problem.imageCamera,
      op = problem.obsPose,
      ox = problem.obsPoint,
      uv = problem.obsUV;
    let nv = 0;
    for (let o = 0; o < nO; o++) {
      const i = op[o];
      const ok = radialResidual(this.R, 9 * i, this.T, 3 * i, this.X, 3 * ox[o], this.K, 5 * ip[i], uv[2 * o], uv[2 * o + 1], res);
      const w = problem.obsWeight ? problem.obsWeight[o] : 1;
      // also reject a finite residual whose weighted square overflows (garbage keypoints):
      // it would make the initial cost infinite and abort the whole run.
      if (ok && w === w && w >= 0 && w < Infinity && w * w * (res[0] * res[0] + res[1] * res[1]) < Infinity) {
        this.obsValid[o] = 1;
        nv++;
      }
    }
    this.numValidObs = nv;
  }

  /** Cost and Σ‖unweighted residual‖² at the current (cand = false) or candidate parameters. */
  evaluate(cand: boolean): EvalResult {
    const R = cand ? this.cR : this.R,
      T = cand ? this.cT : this.T,
      K = cand ? this.cK : this.K,
      X = cand ? this.cX : this.X;
    const p = this.problem;
    const ip = p.imageCamera,
      op = p.obsPose,
      ox = p.obsPoint,
      uv = p.obsUV,
      ow = p.obsWeight;
    const valid = this.obsValid;
    const kind = this.lossKind,
      b = this.lossB;
    const res = this.res2;
    // Neumaier-compensated sums (cost differences near convergence are tiny).
    let cost = 0,
      comp = 0,
      sq = 0;
    for (let o = 0; o < this.nO; o++) {
      if (!valid[o]) continue;
      const i = op[o];
      if (!radialResidual(R, 9 * i, T, 3 * i, X, 3 * ox[o], K, 5 * ip[i], uv[2 * o], uv[2 * o + 1], res))
        return { cost: Infinity, sqErr: Infinity };
      const e2 = res[0] * res[0] + res[1] * res[1];
      sq += e2;
      const w = ow ? ow[o] : 1;
      const term = 0.5 * lossRho(kind, w * w * e2, b);
      const s = cost + term;
      comp += Math.abs(cost) >= Math.abs(term) ? cost - s + term : term - s + cost;
      cost = s;
    }
    let prior = 0;
    const cps = p.centerPriors;
    if (cps) {
      const r = this.pr;
      for (let k = 0; k < cps.length; k++) {
        const cp = cps[k];
        centerPriorResidualJacobian(R, 9 * cp.pose, T, 3 * cp.pose, cp.center, 1 / cp.sigma, r, this.pJ);
        prior += 0.5 * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
      }
    }
    const rps = p.relRotationPriors;
    if (rps) {
      const r = this.pr;
      for (let k = 0; k < rps.length; k++) {
        const rp = rps[k];
        relRotationPriorResidualJacobian(R, 9 * rp.i, R, 9 * rp.j, rp.R_ji, 1 / rp.sigma, r, this.pJi, this.pJj);
        prior += 0.5 * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
      }
    }
    const fps = p.focalPriors;
    if (fps) {
      for (let k = 0; k < fps.length; k++) {
        const fp = fps[k];
        const r = (K[5 * fp.camera] - fp.f) / fp.sigma;
        prior += 0.5 * r * r;
      }
    }
    const pps = p.principalPointPriors;
    if (pps) {
      for (let k = 0; k < pps.length; k++) {
        const pp = pps[k];
        const rx = (K[5 * pp.camera + 1] - pp.cx) / pp.sigma,
          ry = (K[5 * pp.camera + 2] - pp.cy) / pp.sigma;
        prior += 0.5 * (rx * rx + ry * ry);
      }
    }
    const total = cost + comp + prior;
    return { cost: total === total ? total : Infinity, sqErr: sq };
  }

  /** Build parameter blocks, observation ordering and the sparsity of S. */
  buildStructure(): void {
    const p = this.problem;
    const { nP, nC, nX, nO } = this;
    const opts = this.opts;
    // ---- camera-side blocks: free poses first (input order), then cameras ----
    const poseBlock = (this.poseBlock = new Int32Array(nP).fill(-1));
    const tMask = (this.tMask = new Uint8Array(nP));
    let nB = 0;
    let nPinned = 0;
    for (let i = 0; i < nP; i++) {
      if (p.poseFixed[i]) continue;
      poseBlock[i] = nB++;
      const m = p.translationFixedMask ? p.translationFixedMask[i] & 7 : 0;
      tMask[i] = m;
      nPinned += (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1);
    }
    this.numPoseBlocks = nB;
    const camBlock = (this.camBlock = new Int32Array(nC).fill(-1));
    const camNK = (this.camNK = new Int32Array(nC));
    const camIdx = (this.camIdx = new Int32Array(5 * nC).fill(-1));
    for (let c = 0; c < nC; c++) {
      const cr = p.cameraRefine[c];
      let k = 0;
      if (cr) {
        if (cr.focal) camIdx[5 * c + k++] = 0;
        if (cr.principalPoint) {
          camIdx[5 * c + k++] = 1;
          camIdx[5 * c + k++] = 2;
        }
        if (cr.distortion) {
          camIdx[5 * c + k++] = 3;
          camIdx[5 * c + k++] = 4;
        }
      }
      camNK[c] = k;
      if (k > 0) camBlock[c] = nB++;
    }
    const blockSize = new Int32Array(nB);
    for (let b = 0; b < this.numPoseBlocks; b++) blockSize[b] = 6;
    for (let c = 0; c < nC; c++) if (camBlock[c] >= 0) blockSize[camBlock[c]] = camNK[c];

    // ---- observations by point, sorted by pose within each point (two counting sorts) ----
    const valid = this.obsValid;
    const op = p.obsPose,
      ox = p.obsPoint,
      ic = p.imageCamera;
    const poseCnt = new Int32Array(nP + 1);
    for (let o = 0; o < nO; o++) if (valid[o]) poseCnt[op[o] + 1]++;
    for (let i = 0; i < nP; i++) poseCnt[i + 1] += poseCnt[i];
    const nV = poseCnt[nP];
    const byPose = new Int32Array(nV);
    for (let o = 0; o < nO; o++) if (valid[o]) byPose[poseCnt[op[o]]++] = o;
    const ptObsStart = (this.ptObsStart = new Int32Array(nX + 1));
    for (let k = 0; k < nV; k++) ptObsStart[ox[byPose[k]] + 1]++;
    for (let j = 0; j < nX; j++) ptObsStart[j + 1] += ptObsStart[j];
    const ptObs = (this.ptObs = new Int32Array(nV));
    const cur = ptObsStart.slice(0, nX);
    for (let k = 0; k < nV; k++) {
      const o = byPose[k];
      ptObs[cur[ox[o]]++] = o;
    }

    // ---- free points and their distinct camera-side blocks ----
    const pointFree = (this.pointFree = new Uint8Array(nX));
    const ptBlkStart = (this.ptBlkStart = new Int32Array(nX + 1));
    const blkTmp = new Int32Array(2 * nV);
    const obsKp = (this.obsKp = new Int32Array(nO).fill(-1));
    const obsKc = (this.obsKc = new Int32Array(nO).fill(-1));
    const camStamp = new Int32Array(nC).fill(-1);
    let cursor = 0;
    let nFreePts = 0;
    let maxM = 0;
    for (let j = 0; j < nX; j++) {
      ptBlkStart[j] = cursor;
      const o0 = ptObsStart[j],
        o1 = ptObsStart[j + 1];
      if (o1 === o0 || (p.pointFixed && p.pointFixed[j])) continue;
      pointFree[j] = 1;
      nFreePts++;
      const start = cursor;
      let last = -1;
      for (let k = o0; k < o1; k++) {
        const o = ptObs[k];
        const pb = poseBlock[op[o]];
        if (pb < 0) continue;
        if (pb !== last) {
          blkTmp[cursor++] = pb;
          last = pb;
        }
        obsKp[o] = cursor - 1 - start;
      }
      const camStart = cursor;
      for (let k = o0; k < o1; k++) {
        const o = ptObs[k];
        const c = ic[op[o]];
        const cb = camBlock[c];
        if (cb < 0 || camStamp[c] === j) continue;
        camStamp[c] = j;
        // insertion into the sorted camera part
        let q = cursor++;
        while (q > camStart && blkTmp[q - 1] > cb) {
          blkTmp[q] = blkTmp[q - 1];
          q--;
        }
        blkTmp[q] = cb;
      }
      for (let k = o0; k < o1; k++) {
        const o = ptObs[k];
        const cb = camBlock[ic[op[o]]];
        if (cb < 0) continue;
        let q = camStart;
        while (blkTmp[q] !== cb) q++;
        obsKc[o] = q - start;
      }
      const m = cursor - start;
      if (m > maxM) maxM = m;
    }
    ptBlkStart[nX] = cursor;
    this.numFreePoints = nFreePts;
    this.maxM = maxM;
    const ptBlk = (this.ptBlk = blkTmp.slice(0, cursor));
    const nEnt = cursor;

    // ---- per-point pair index ranges ----
    const pairStart = (this.pairStart = new Int32Array(nX + 1));
    {
      let acc = 0;
      for (let j = 0; j < nX; j++) {
        pairStart[j] = acc;
        const m = ptBlkStart[j + 1] - ptBlkStart[j];
        acc += (m * (m + 1)) / 2;
      }
      if (acc > 0x7fffffff) throw new RangeError('BA: problem too large (Schur pair count overflow)');
      pairStart[nX] = acc;
    }

    // ---- U-only couplings: pose–camera, relative-rotation priors ----
    const extraRow: number[] = [];
    const extraCol: number[] = [];
    for (let i = 0; i < nP; i++) {
      const pb = poseBlock[i];
      const cb = camBlock[ic[i]];
      if (pb >= 0 && cb >= 0) {
        extraRow.push(pb);
        extraCol.push(cb);
      }
    }
    const rps = p.relRotationPriors ?? [];
    for (const rp of rps) {
      const a = poseBlock[rp.i],
        b = poseBlock[rp.j];
      if (a >= 0 && b >= 0) {
        extraRow.push(Math.min(a, b));
        extraCol.push(Math.max(a, b));
      }
    }
    const exStart = new Int32Array(nB + 1);
    for (let k = 0; k < extraRow.length; k++) exStart[extraRow[k] + 1]++;
    for (let b = 0; b < nB; b++) exStart[b + 1] += exStart[b];
    const exCol = new Int32Array(extraRow.length);
    {
      const c2 = exStart.slice(0, nB);
      for (let k = 0; k < extraRow.length; k++) exCol[c2[extraRow[k]]++] = extraCol[k];
    }

    // ---- incidence block → entries (global entry index e = ptBlkStart[j] + k) ----
    const incStart = new Int32Array(nB + 1);
    for (let e = 0; e < nEnt; e++) incStart[ptBlk[e] + 1]++;
    for (let b = 0; b < nB; b++) incStart[b + 1] += incStart[b];
    const inc = new Int32Array(nEnt);
    const entPoint = new Int32Array(nEnt);
    {
      const c2 = incStart.slice(0, nB);
      for (let j = 0; j < nX; j++) {
        for (let e = ptBlkStart[j]; e < ptBlkStart[j + 1]; e++) {
          entPoint[e] = j;
          inc[c2[ptBlk[e]]++] = e;
        }
      }
    }

    // ---- block pattern of S (upper triangle), rows via stamping ----
    const rowStart = new Int32Array(nB + 1);
    let colBuf = new Int32Array(Math.max(16, nB * 4));
    let nnz = 0;
    const stamp = new Int32Array(nB).fill(-1);
    for (let a = 0; a < nB; a++) {
      rowStart[a] = nnz;
      const push = (b: number): void => {
        if (stamp[b] === a) return;
        stamp[b] = a;
        if (nnz === colBuf.length) {
          const nb = new Int32Array(colBuf.length * 2);
          nb.set(colBuf);
          colBuf = nb;
        }
        colBuf[nnz++] = b;
      };
      push(a);
      for (let q = exStart[a]; q < exStart[a + 1]; q++) push(exCol[q]);
      for (let q = incStart[a]; q < incStart[a + 1]; q++) {
        const e = inc[q];
        const j = entPoint[e];
        const end = ptBlkStart[j + 1];
        for (let e2 = e; e2 < end; e2++) push(ptBlk[e2]);
      }
      colBuf.subarray(rowStart[a], nnz).sort();
    }
    rowStart[nB] = nnz;
    const S = (this.S = new BlockSparseSymmetric(blockSize, rowStart, colBuf.slice(0, nnz)));
    this.U = new Float64Array(S.values.length);
    const diagOff = (this.diagOff = new Int32Array(nB));
    for (let b = 0; b < nB; b++) diagOff[b] = S.valOff[S.diagEntry[b]];

    // ---- per-point pair offsets (row by row, with a column → entry scatter) ----
    const pairOff = (this.pairOff = new Int32Array(pairStart[nX]));
    const colEntry = new Int32Array(nB).fill(-1);
    for (let a = 0; a < nB; a++) {
      for (let e = rowStart[a]; e < rowStart[a + 1]; e++) colEntry[S.col[e]] = e;
      for (let q = incStart[a]; q < incStart[a + 1]; q++) {
        const e = inc[q];
        const j = entPoint[e];
        const e0 = ptBlkStart[j];
        const m = ptBlkStart[j + 1] - e0;
        const k = e - e0;
        let pos = pairStart[j] + k * m - (k * (k - 1)) / 2;
        for (let k2 = k; k2 < m; k2++) pairOff[pos++] = S.valOff[colEntry[ptBlk[e0 + k2]]];
      }
    }
    const findOff = (a: number, b: number): number => {
      for (let e = rowStart[a]; e < rowStart[a + 1]; e++) if (S.col[e] === b) return S.valOff[e];
      throw new Error('BA: internal sparsity error');
    };
    const poseCamOff = (this.poseCamOff = new Int32Array(nP).fill(-1));
    for (let i = 0; i < nP; i++) {
      const pb = poseBlock[i],
        cb = camBlock[ic[i]];
      if (pb >= 0 && cb >= 0) poseCamOff[i] = findOff(pb, cb);
    }
    this.relOff = new Int32Array(rps.length).fill(-1);
    this.relSwap = new Uint8Array(rps.length);
    for (let k = 0; k < rps.length; k++) {
      const a = poseBlock[rps[k].i],
        b = poseBlock[rps[k].j];
      if (a >= 0 && b >= 0) {
        this.relOff[k] = findOff(Math.min(a, b), Math.max(a, b));
        this.relSwap[k] = a > b ? 1 : 0;
      }
    }

    // ---- pinned translation components & dense index map ----
    const N = (this.N = S.dim);
    this.pinned = new Int32Array(nPinned);
    const denseMap = (this.denseMap = new Int32Array(N));
    {
      let q = 0;
      for (let i = 0; i < nP; i++) {
        const pb = poseBlock[i];
        if (pb < 0) continue;
        for (let bit = 0; bit < 3; bit++) if (tMask[i] & (1 << bit)) this.pinned[q++] = S.blockStart[pb] + 3 + bit;
      }
      denseMap.fill(0);
      for (let k = 0; k < nPinned; k++) denseMap[this.pinned[k]] = -1;
      let g = 0;
      for (let s = 0; s < N; s++) if (denseMap[s] === 0) denseMap[s] = g++;
      else denseMap[s] = -1;
      this.nFree = g;
    }

    // ---- work arrays ----
    this.gY = new Float64Array(N);
    this.gX = new Float64Array(3 * nX);
    this.W = new Float64Array(18 * nEnt);
    this.V = new Float64Array(6 * nX);
    this.Vinv = new Float64Array(6 * nX);
    this.ptSkip = new Uint8Array(nX);
    this.rhs = new Float64Array(N);
    this.dy = new Float64Array(N);
    this.dx = new Float64Array(3 * nX);
    this.Ysc = new Float64Array(18 * Math.max(1, maxM));

    const nF = this.nFree;
    const ls = opts.linearSolver;
    // Direct solver: packed skyline Cholesky over the (fixed) envelope of S.
    const first = nF > 0 ? skylineEnvelope(S, denseMap, nF) : new Int32Array(0);
    const skySize = SkylineCholesky.storageSize(first);
    if (ls === 'pcg_schur') this.useDense = false;
    else if (ls === 'dense_schur') this.useDense = skySize <= DENSE_HARD_LIMIT_ENTRIES;
    else
      this.useDense =
        nF <= DENSE_SCHUR_MAX_SIZE ||
        (skySize <= AUTO_SKYLINE_MAX_ENTRIES && SkylineCholesky.factorCost(first) <= AUTO_SKYLINE_MAX_COST);
    if (nF === 0) this.useDense = true;
    if (this.useDense) {
      this.sky = new SkylineCholesky(first);
      this.denseB = new Float64Array(nF);
    } else {
      this.precond = new BlockJacobiPreconditioner(S);
      this.pcgWs = createPcgWorkspace(N);
    }
  }

  /** Total number of free parameters (camera-side + 3 per free point). */
  get numParams(): number {
    return this.nFree + 3 * this.numFreePoints;
  }

  /** Gradient, U, V, W at the current parameters (IRLS-weighted). */
  linearize(): void {
    const p = this.problem;
    const R = this.R,
      T = this.T,
      K = this.K,
      X = this.X;
    const ip = p.imageCamera,
      op = p.obsPose,
      uv = p.obsUV,
      ow = p.obsWeight;
    const kind = this.lossKind,
      lb = this.lossB;
    const J = this.jac,
      jk = this.jk;
    const U = this.U,
      gY = this.gY,
      gX = this.gX,
      W = this.W,
      V = this.V;
    const poseBlock = this.poseBlock,
      camBlock = this.camBlock,
      camNK = this.camNK,
      camIdx = this.camIdx,
      tMask = this.tMask;
    const bStart = this.S.blockStart;
    const diagOff = this.diagOff,
      poseCamOff = this.poseCamOff;
    const ptObsStart = this.ptObsStart,
      ptObs = this.ptObs,
      ptBlkStart = this.ptBlkStart,
      obsKp = this.obsKp,
      obsKc = this.obsKc,
      pointFree = this.pointFree;
    U.fill(0);
    gY.fill(0);
    for (let j = 0; j < this.nX; j++) {
      const free = pointFree[j];
      const eBase = ptBlkStart[j];
      if (free) W.fill(0, 18 * eBase, 18 * ptBlkStart[j + 1]);
      let v00 = 0,
        v01 = 0,
        v02 = 0,
        v11 = 0,
        v12 = 0,
        v22 = 0,
        g0 = 0,
        g1 = 0,
        g2 = 0;
      for (let q = ptObsStart[j]; q < ptObsStart[j + 1]; q++) {
        const o = ptObs[q];
        const i = op[o];
        const c = ip[i];
        if (!radialResidualJacobian(R, 9 * i, T, 3 * i, X, 3 * j, K, 5 * c, uv[2 * o], uv[2 * o + 1], J)) continue;
        const w = ow ? ow[o] : 1;
        const s = w * w * (J[0] * J[0] + J[1] * J[1]);
        const sc = w * Math.sqrt(lossWeight(kind, s, lb));
        for (let k = 0; k < RJ_SIZE; k++) J[k] *= sc;
        const ru = J[0],
          rv = J[1];
        const pb = poseBlock[i];
        if (pb >= 0) {
          const m = tMask[i];
          if (m) {
            if (m & 1) J[5] = J[11] = 0;
            if (m & 2) J[6] = J[12] = 0;
            if (m & 4) J[7] = J[13] = 0;
          }
          // Unrolled: rows a = J[2..7], b = J[8..13]; upper triangle of the 6x6 block.
          const a0 = J[2],
            a1 = J[3],
            a2 = J[4],
            a3 = J[5],
            a4 = J[6],
            a5 = J[7];
          const b0 = J[8],
            b1 = J[9],
            b2 = J[10],
            b3 = J[11],
            b4 = J[12],
            b5 = J[13];
          const go = bStart[pb];
          gY[go] += a0 * ru + b0 * rv;
          gY[go + 1] += a1 * ru + b1 * rv;
          gY[go + 2] += a2 * ru + b2 * rv;
          gY[go + 3] += a3 * ru + b3 * rv;
          gY[go + 4] += a4 * ru + b4 * rv;
          gY[go + 5] += a5 * ru + b5 * rv;
          const u = diagOff[pb];
          U[u] += a0 * a0 + b0 * b0;
          U[u + 1] += a0 * a1 + b0 * b1;
          U[u + 2] += a0 * a2 + b0 * b2;
          U[u + 3] += a0 * a3 + b0 * b3;
          U[u + 4] += a0 * a4 + b0 * b4;
          U[u + 5] += a0 * a5 + b0 * b5;
          U[u + 7] += a1 * a1 + b1 * b1;
          U[u + 8] += a1 * a2 + b1 * b2;
          U[u + 9] += a1 * a3 + b1 * b3;
          U[u + 10] += a1 * a4 + b1 * b4;
          U[u + 11] += a1 * a5 + b1 * b5;
          U[u + 14] += a2 * a2 + b2 * b2;
          U[u + 15] += a2 * a3 + b2 * b3;
          U[u + 16] += a2 * a4 + b2 * b4;
          U[u + 17] += a2 * a5 + b2 * b5;
          U[u + 21] += a3 * a3 + b3 * b3;
          U[u + 22] += a3 * a4 + b3 * b4;
          U[u + 23] += a3 * a5 + b3 * b5;
          U[u + 28] += a4 * a4 + b4 * b4;
          U[u + 29] += a4 * a5 + b4 * b5;
          U[u + 35] += a5 * a5 + b5 * b5;
        }
        const cb = camBlock[c];
        const kc = camNK[c];
        if (cb >= 0) {
          for (let q2 = 0; q2 < kc; q2++) {
            const idx = camIdx[5 * c + q2];
            jk[q2] = J[20 + idx];
            jk[5 + q2] = J[25 + idx];
          }
          const go = bStart[cb];
          const uo = diagOff[cb];
          for (let a = 0; a < kc; a++) {
            const ja = jk[a],
              jb = jk[5 + a];
            gY[go + a] += ja * ru + jb * rv;
            const ro = uo + kc * a;
            for (let bb = a; bb < kc; bb++) U[ro + bb] += ja * jk[bb] + jb * jk[5 + bb];
          }
          if (pb >= 0) {
            const po = poseCamOff[i];
            for (let a = 0; a < 6; a++) {
              const ja = J[2 + a],
                jb = J[8 + a];
              const ro = po + kc * a;
              for (let bb = 0; bb < kc; bb++) U[ro + bb] += ja * jk[bb] + jb * jk[5 + bb];
            }
          }
        }
        if (free) {
          const p0 = J[14],
            p1 = J[15],
            p2 = J[16],
            p3 = J[17],
            p4 = J[18],
            p5 = J[19];
          v00 += p0 * p0 + p3 * p3;
          v01 += p0 * p1 + p3 * p4;
          v02 += p0 * p2 + p3 * p5;
          v11 += p1 * p1 + p4 * p4;
          v12 += p1 * p2 + p4 * p5;
          v22 += p2 * p2 + p5 * p5;
          g0 += p0 * ru + p3 * rv;
          g1 += p1 * ru + p4 * rv;
          g2 += p2 * ru + p5 * rv;
          if (pb >= 0) {
            const wo = 18 * (eBase + obsKp[o]);
            const a0 = J[2],
              a1 = J[3],
              a2 = J[4],
              a3 = J[5],
              a4 = J[6],
              a5 = J[7];
            const b0 = J[8],
              b1 = J[9],
              b2 = J[10],
              b3 = J[11],
              b4 = J[12],
              b5 = J[13];
            W[wo] += a0 * p0 + b0 * p3;
            W[wo + 1] += a0 * p1 + b0 * p4;
            W[wo + 2] += a0 * p2 + b0 * p5;
            W[wo + 3] += a1 * p0 + b1 * p3;
            W[wo + 4] += a1 * p1 + b1 * p4;
            W[wo + 5] += a1 * p2 + b1 * p5;
            W[wo + 6] += a2 * p0 + b2 * p3;
            W[wo + 7] += a2 * p1 + b2 * p4;
            W[wo + 8] += a2 * p2 + b2 * p5;
            W[wo + 9] += a3 * p0 + b3 * p3;
            W[wo + 10] += a3 * p1 + b3 * p4;
            W[wo + 11] += a3 * p2 + b3 * p5;
            W[wo + 12] += a4 * p0 + b4 * p3;
            W[wo + 13] += a4 * p1 + b4 * p4;
            W[wo + 14] += a4 * p2 + b4 * p5;
            W[wo + 15] += a5 * p0 + b5 * p3;
            W[wo + 16] += a5 * p1 + b5 * p4;
            W[wo + 17] += a5 * p2 + b5 * p5;
          }
          if (cb >= 0) {
            let wo = 18 * (eBase + obsKc[o]);
            for (let a = 0; a < kc; a++, wo += 3) {
              const ja = jk[a],
                jb = jk[5 + a];
              W[wo] += ja * p0 + jb * p3;
              W[wo + 1] += ja * p1 + jb * p4;
              W[wo + 2] += ja * p2 + jb * p5;
            }
          }
        }
      }
      if (free) {
        const vo = 6 * j;
        V[vo] = v00;
        V[vo + 1] = v01;
        V[vo + 2] = v02;
        V[vo + 3] = v11;
        V[vo + 4] = v12;
        V[vo + 5] = v22;
        gX[3 * j] = g0;
        gX[3 * j + 1] = g1;
        gX[3 * j + 2] = g2;
      }
    }

    // ---- priors (trivial loss) ----
    const cps = p.centerPriors;
    if (cps) {
      const r = this.pr,
        PJ = this.pJ;
      for (let k = 0; k < cps.length; k++) {
        const cp = cps[k];
        const i = cp.pose;
        const pb = poseBlock[i];
        if (pb < 0) continue;
        centerPriorResidualJacobian(R, 9 * i, T, 3 * i, cp.center, 1 / cp.sigma, r, PJ);
        const m = tMask[i];
        for (let bit = 0; bit < 3; bit++) if (m & (1 << bit)) PJ[3 + bit] = PJ[9 + bit] = PJ[15 + bit] = 0;
        const go = bStart[pb],
          uo = diagOff[pb];
        for (let a = 0; a < 6; a++) {
          const j0 = PJ[a],
            j1 = PJ[6 + a],
            j2 = PJ[12 + a];
          gY[go + a] += j0 * r[0] + j1 * r[1] + j2 * r[2];
          for (let bb = a; bb < 6; bb++) U[uo + 6 * a + bb] += j0 * PJ[bb] + j1 * PJ[6 + bb] + j2 * PJ[12 + bb];
        }
      }
    }
    const rps = p.relRotationPriors;
    if (rps) {
      const r = this.pr,
        Ji = this.pJi,
        Jj = this.pJj;
      for (let k = 0; k < rps.length; k++) {
        const rp = rps[k];
        const pi = poseBlock[rp.i],
          pj = poseBlock[rp.j];
        if (pi < 0 && pj < 0) continue;
        relRotationPriorResidualJacobian(R, 9 * rp.i, R, 9 * rp.j, rp.R_ji, 1 / rp.sigma, r, Ji, Jj);
        if (pi >= 0) this.addPriorRot(pi, Ji, r);
        if (pj >= 0) this.addPriorRot(pj, Jj, r);
        if (pi >= 0 && pj >= 0) {
          // cross block U(min, max)[0:3, 0:3] (row block = smaller index)
          const off = this.relOff[k];
          const A = this.relSwap[k] ? Jj : Ji; // Jacobian of the row block
          const B = this.relSwap[k] ? Ji : Jj;
          for (let a = 0; a < 3; a++)
            for (let bb = 0; bb < 3; bb++)
              U[off + 6 * a + bb] += A[a] * B[bb] + A[3 + a] * B[3 + bb] + A[6 + a] * B[6 + bb];
        }
      }
    }

    // focal priors: r = (f − f₀)/σ, ∂r/∂f = 1/σ (focal is the first parameter of a camera block when refined)
    const fps = p.focalPriors;
    if (fps) {
      for (let k = 0; k < fps.length; k++) {
        const fp = fps[k];
        const cb = this.camBlock[fp.camera];
        if (cb < 0 || this.camIdx[5 * fp.camera] !== 0) continue;
        const is = 1 / fp.sigma;
        const r = (this.K[5 * fp.camera] - fp.f) * is;
        gY[bStart[cb]] += is * r;
        U[diagOff[cb]] += is * is;
      }
    }

    // principal-point priors: r = ((cx − cx₀)/σ, (cy − cy₀)/σ); cx, cy follow the focal in a camera block
    const pps = p.principalPointPriors;
    if (pps) {
      const bs = this.S.blockSize;
      for (let k = 0; k < pps.length; k++) {
        const pp = pps[k];
        const cb = this.camBlock[pp.camera];
        if (cb < 0) continue;
        const n = bs[cb];
        const is = 1 / pp.sigma;
        for (let q = 0; q < n; q++) {
          const param = this.camIdx[5 * pp.camera + q];
          if (param !== 1 && param !== 2) continue;
          const r = (this.K[5 * pp.camera + param] - (param === 1 ? pp.cx : pp.cy)) * is;
          gY[bStart[cb] + q] += is * r;
          U[diagOff[cb] + q * n + q] += is * is;
        }
      }
    }

    // ---- mirror upper → lower in diagonal blocks ----
    const nB = this.S.numBlocks;
    const bs = this.S.blockSize;
    for (let b = 0; b < nB; b++) {
      const n = bs[b],
        o = diagOff[b];
      for (let r = 1; r < n; r++) for (let c = 0; c < r; c++) U[o + r * n + c] = U[o + c * n + r];
    }
  }

  private addPriorRot(pb: number, Jr: Float64Array, r: Float64Array): void {
    const U = this.U,
      gY = this.gY;
    const go = this.S.blockStart[pb],
      uo = this.diagOff[pb];
    for (let a = 0; a < 3; a++) {
      const j0 = Jr[a],
        j1 = Jr[3 + a],
        j2 = Jr[6 + a];
      gY[go + a] += j0 * r[0] + j1 * r[1] + j2 * r[2];
      for (let bb = a; bb < 3; bb++) U[uo + 6 * a + bb] += j0 * Jr[bb] + j1 * Jr[3 + bb] + j2 * Jr[6 + bb];
    }
  }

  /** max |gradient| over free parameters. */
  gradMaxAbs(): number {
    let m = 0;
    const gY = this.gY,
      gX = this.gX,
      pf = this.pointFree;
    for (let k = 0; k < gY.length; k++) {
      const a = Math.abs(gY[k]);
      if (a > m || a !== a) m = a !== a ? Infinity : a;
    }
    for (let j = 0; j < this.nX; j++) {
      if (!pf[j]) continue;
      for (let k = 3 * j; k < 3 * j + 3; k++) {
        const a = Math.abs(gX[k]);
        if (a > m || a !== a) m = a !== a ? Infinity : a;
      }
    }
    return m;
  }

  /** ‖x‖ over the free parameters (rotations counted as 1 each). */
  paramNorm(): number {
    let s = 0;
    for (let i = 0; i < this.nP; i++) {
      if (this.poseBlock[i] < 0) continue;
      s += 1;
      for (let k = 0; k < 3; k++) s += this.T[3 * i + k] * this.T[3 * i + k];
    }
    for (let c = 0; c < this.nC; c++)
      for (let q = 0; q < this.camNK[c]; q++) {
        const v = this.K[5 * c + this.camIdx[5 * c + q]];
        s += v * v;
      }
    for (let j = 0; j < this.nX; j++)
      if (this.pointFree[j]) for (let k = 3 * j; k < 3 * j + 3; k++) s += this.X[k] * this.X[k];
    return Math.sqrt(s);
  }

  /**
   * Solve the damped system for (dy, dx). Returns false on a linear-solver failure
   * (not positive definite / non-finite).
   */
  solve(lambda: number): boolean {
    const S = this.S;
    const Sv = S.values;
    const U = this.U;
    Sv.set(U);
    const nB = S.numBlocks;
    const bs = S.blockSize,
      bStart = S.blockStart;
    const diagOff = this.diagOff;
    for (let b = 0; b < nB; b++) {
      const n = bs[b],
        o = diagOff[b];
      for (let q = 0; q < n; q++) {
        const idx = o + q * n + q;
        let d = U[idx];
        d = d > MIN_DIAG ? (d < MAX_DIAG ? d : MAX_DIAG) : MIN_DIAG;
        Sv[idx] += lambda * d;
      }
    }
    // Pinned translation components: decoupled rows (U, W rows are zero) with unit diagonal.
    const pinned = this.pinned;
    for (let k = 0; k < pinned.length; k++) {
      const s = pinned[k];
      // locate block & local index
      const b = this.blockOfScalar(s);
      const n = bs[b],
        q = s - bStart[b];
      Sv[diagOff[b] + q * n + q] = 1;
    }
    const rhs = this.rhs,
      gY = this.gY;
    for (let k = 0; k < rhs.length; k++) rhs[k] = -gY[k];

    // ---- eliminate points ----
    const V = this.V,
      Vinv = this.Vinv,
      gX = this.gX,
      W = this.W,
      Y = this.Ysc;
    const ptBlkStart = this.ptBlkStart,
      ptBlk = this.ptBlk,
      pairStart = this.pairStart,
      pairOff = this.pairOff,
      pointFree = this.pointFree,
      ptSkip = this.ptSkip;
    for (let j = 0; j < this.nX; j++) {
      if (!pointFree[j]) continue;
      const vo = 6 * j;
      let a00 = V[vo],
        a01 = V[vo + 1],
        a02 = V[vo + 2],
        a11 = V[vo + 3],
        a12 = V[vo + 4],
        a22 = V[vo + 5];
      a00 += lambda * (a00 > MIN_DIAG ? (a00 < MAX_DIAG ? a00 : MAX_DIAG) : MIN_DIAG);
      a11 += lambda * (a11 > MIN_DIAG ? (a11 < MAX_DIAG ? a11 : MAX_DIAG) : MIN_DIAG);
      a22 += lambda * (a22 > MIN_DIAG ? (a22 < MAX_DIAG ? a22 : MAX_DIAG) : MIN_DIAG);
      // 3x3 Cholesky with relative pivot checks, then explicit inverse.
      if (!(a00 > 0) || a00 === Infinity) {
        ptSkip[j] = 1;
        continue;
      }
      const l00 = Math.sqrt(a00);
      const l10 = a01 / l00,
        l20 = a02 / l00;
      const d1 = a11 - l10 * l10;
      if (!(d1 > V_PIVOT_EPS * a11)) {
        ptSkip[j] = 1;
        continue;
      }
      const l11 = Math.sqrt(d1);
      const l21 = (a12 - l20 * l10) / l11;
      const d2 = a22 - l20 * l20 - l21 * l21;
      if (!(d2 > V_PIVOT_EPS * a22) || d2 === Infinity) {
        ptSkip[j] = 1;
        continue;
      }
      const l22 = Math.sqrt(d2);
      // L⁻¹ (lower)
      const m00 = 1 / l00,
        m11 = 1 / l11,
        m22 = 1 / l22;
      const m10 = -l10 * m00 * m11;
      const m21 = -l21 * m11 * m22;
      const m20 = -(l20 * m00 + l21 * m10) * m22;
      // V⁻¹ = L⁻ᵀ L⁻¹
      const i00 = m00 * m00 + m10 * m10 + m20 * m20;
      const i01 = m10 * m11 + m20 * m21;
      const i02 = m20 * m22;
      const i11 = m11 * m11 + m21 * m21;
      const i12 = m21 * m22;
      const i22 = m22 * m22;
      Vinv[vo] = i00;
      Vinv[vo + 1] = i01;
      Vinv[vo + 2] = i02;
      Vinv[vo + 3] = i11;
      Vinv[vo + 4] = i12;
      Vinv[vo + 5] = i22;
      ptSkip[j] = 0;
      const e0 = ptBlkStart[j];
      const m = ptBlkStart[j + 1] - e0;
      if (m === 0) continue;
      const gx0 = gX[3 * j],
        gx1 = gX[3 * j + 1],
        gx2 = gX[3 * j + 2];
      // Y_k = W_k V⁻¹ ; rhs_b += Y_k g_x
      for (let k = 0; k < m; k++) {
        const b = ptBlk[e0 + k];
        const n = bs[b];
        let wo = 18 * (e0 + k);
        let yo = 18 * k;
        const ro = bStart[b];
        for (let r = 0; r < n; r++, wo += 3, yo += 3) {
          const w0 = W[wo],
            w1 = W[wo + 1],
            w2 = W[wo + 2];
          const y0 = w0 * i00 + w1 * i01 + w2 * i02;
          const y1 = w0 * i01 + w1 * i11 + w2 * i12;
          const y2 = w0 * i02 + w1 * i12 + w2 * i22;
          Y[yo] = y0;
          Y[yo + 1] = y1;
          Y[yo + 2] = y2;
          rhs[ro + r] += y0 * gx0 + y1 * gx1 + y2 * gx2;
        }
      }
      // S(b1, b2) −= Y_k1 W_k2ᵀ for k1 ≤ k2
      let pp = pairStart[j];
      for (let k1 = 0; k1 < m; k1++) {
        const n1 = bs[ptBlk[e0 + k1]];
        const y1 = 18 * k1;
        for (let k2 = k1; k2 < m; k2++) {
          const n2 = bs[ptBlk[e0 + k2]];
          const w2 = 18 * (e0 + k2);
          let off = pairOff[pp++];
          if (n2 === 6) {
            // pose column block (hot path): W_k2 in registers, unrolled 6 columns
            const w00 = W[w2],
              w01 = W[w2 + 1],
              w02 = W[w2 + 2],
              w10 = W[w2 + 3],
              w11 = W[w2 + 4],
              w12 = W[w2 + 5],
              w20 = W[w2 + 6],
              w21 = W[w2 + 7],
              w22 = W[w2 + 8],
              w30 = W[w2 + 9],
              w31 = W[w2 + 10],
              w32 = W[w2 + 11],
              w40 = W[w2 + 12],
              w41 = W[w2 + 13],
              w42 = W[w2 + 14],
              w50 = W[w2 + 15],
              w51 = W[w2 + 16],
              w52 = W[w2 + 17];
            for (let r = 0, yo = y1; r < n1; r++, yo += 3, off += 6) {
              const ya = Y[yo],
                yb = Y[yo + 1],
                yc = Y[yo + 2];
              Sv[off] -= ya * w00 + yb * w01 + yc * w02;
              Sv[off + 1] -= ya * w10 + yb * w11 + yc * w12;
              Sv[off + 2] -= ya * w20 + yb * w21 + yc * w22;
              Sv[off + 3] -= ya * w30 + yb * w31 + yc * w32;
              Sv[off + 4] -= ya * w40 + yb * w41 + yc * w42;
              Sv[off + 5] -= ya * w50 + yb * w51 + yc * w52;
            }
          } else {
            for (let r = 0; r < n1; r++) {
              const ya = Y[y1 + 3 * r],
                yb = Y[y1 + 3 * r + 1],
                yc = Y[y1 + 3 * r + 2];
              for (let c = 0, wo = w2; c < n2; c++, wo += 3) {
                Sv[off++] -= ya * W[wo] + yb * W[wo + 1] + yc * W[wo + 2];
              }
            }
          }
        }
      }
    }

    // ---- reduced camera system ----
    const dy = this.dy;
    if (this.nFree === 0) {
      dy.fill(0);
    } else if (this.useDense) {
      const sky = this.sky!,
        B = this.denseB!;
      const map = this.denseMap;
      scatterToSkyline(S, map, sky);
      for (let s = 0; s < this.N; s++) {
        const g = map[s];
        if (g >= 0) B[g] = rhs[s];
      }
      if (!sky.factor()) return false;
      sky.solve(B);
      for (let s = 0; s < this.N; s++) {
        const g = map[s];
        const v = g >= 0 ? B[g] : 0;
        if (v !== v) return false;
        dy[s] = v;
      }
    } else {
      const M = this.precond!;
      M.update();
      const it = pcgSolve(S, M, rhs, dy, this.opts.pcgMaxIterations, this.opts.pcgTolerance, this.pcgWs!);
      if (it < 0) return false;
      this.pcgIterations += it;
      for (let k = 0; k < this.pinned.length; k++) dy[this.pinned[k]] = 0;
      for (let s = 0; s < this.N; s++) if (dy[s] !== dy[s]) return false;
    }

    // ---- back-substitution δx = V⁻¹(−g_x − Wᵀ δy) ----
    const dx = this.dx;
    for (let j = 0; j < this.nX; j++) {
      const o3 = 3 * j;
      if (!pointFree[j] || ptSkip[j]) {
        dx[o3] = dx[o3 + 1] = dx[o3 + 2] = 0;
        continue;
      }
      let t0 = -gX[o3],
        t1 = -gX[o3 + 1],
        t2 = -gX[o3 + 2];
      const e0 = ptBlkStart[j],
        e1 = ptBlkStart[j + 1];
      for (let e = e0; e < e1; e++) {
        const b = ptBlk[e];
        const n = bs[b];
        const ro = bStart[b];
        let wo = 18 * e;
        for (let r = 0; r < n; r++, wo += 3) {
          const d = dy[ro + r];
          t0 -= W[wo] * d;
          t1 -= W[wo + 1] * d;
          t2 -= W[wo + 2] * d;
        }
      }
      const vo = 6 * j;
      const x0 = Vinv[vo] * t0 + Vinv[vo + 1] * t1 + Vinv[vo + 2] * t2;
      const x1 = Vinv[vo + 1] * t0 + Vinv[vo + 3] * t1 + Vinv[vo + 4] * t2;
      const x2 = Vinv[vo + 2] * t0 + Vinv[vo + 4] * t1 + Vinv[vo + 5] * t2;
      if (x0 !== x0 || x1 !== x1 || x2 !== x2) return false;
      dx[o3] = x0;
      dx[o3 + 1] = x1;
      dx[o3 + 2] = x2;
    }
    return true;
  }

  private blockOfScalar(s: number): number {
    const st = this.S.blockStart;
    let lo = 0,
      hi = this.S.numBlocks - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (st[mid] <= s) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  /** m(0) − m(δ) = −gᵀδ − ½ δᵀ H δ for the undamped IRLS Gauss–Newton model. */
  modelDecrease(): number {
    const dy = this.dy,
      dx = this.dx,
      gY = this.gY,
      gX = this.gX;
    let lin = 0;
    for (let k = 0; k < dy.length; k++) lin += gY[k] * dy[k];
    let quad = this.S.quadraticForm(dy, this.U);
    const V = this.V,
      W = this.W;
    const ptBlkStart = this.ptBlkStart,
      ptBlk = this.ptBlk,
      bs = this.S.blockSize,
      bStart = this.S.blockStart;
    for (let j = 0; j < this.nX; j++) {
      if (!this.pointFree[j] || this.ptSkip[j]) continue;
      const o3 = 3 * j;
      const x0 = dx[o3],
        x1 = dx[o3 + 1],
        x2 = dx[o3 + 2];
      lin += gX[o3] * x0 + gX[o3 + 1] * x1 + gX[o3 + 2] * x2;
      const vo = 6 * j;
      quad +=
        V[vo] * x0 * x0 +
        V[vo + 3] * x1 * x1 +
        V[vo + 5] * x2 * x2 +
        2 * (V[vo + 1] * x0 * x1 + V[vo + 2] * x0 * x2 + V[vo + 4] * x1 * x2);
      let cross = 0;
      for (let e = ptBlkStart[j]; e < ptBlkStart[j + 1]; e++) {
        const b = ptBlk[e];
        const n = bs[b],
          ro = bStart[b];
        let wo = 18 * e;
        for (let r = 0; r < n; r++, wo += 3) cross += dy[ro + r] * (W[wo] * x0 + W[wo + 1] * x1 + W[wo + 2] * x2);
      }
      quad += 2 * cross;
    }
    return -lin - 0.5 * quad;
  }

  /** ‖δ‖ over all free parameters. */
  stepNorm(): number {
    let s = 0;
    const dy = this.dy,
      dx = this.dx;
    for (let k = 0; k < dy.length; k++) s += dy[k] * dy[k];
    for (let k = 0; k < dx.length; k++) s += dx[k] * dx[k];
    return Math.sqrt(s);
  }

  /** Candidate parameters = current ⊕ δ. */
  applyStep(): void {
    const dy = this.dy,
      dx = this.dx;
    const R = this.R,
      T = this.T,
      cR = this.cR,
      cT = this.cT;
    const rot = this.rot;
    const bStart = this.S.blockStart;
    const w = this.pr;
    for (let i = 0; i < this.nP; i++) {
      const pb = this.poseBlock[i];
      if (pb < 0) continue;
      const o = bStart[pb];
      w[0] = dy[o];
      w[1] = dy[o + 1];
      w[2] = dy[o + 2];
      rotvecToMat3(w, rot);
      const ro = 9 * i;
      for (let r = 0; r < 3; r++) {
        const a = rot[3 * r],
          b = rot[3 * r + 1],
          c = rot[3 * r + 2];
        for (let col = 0; col < 3; col++) {
          cR[ro + 3 * r + col] = a * R[ro + col] + b * R[ro + 3 + col] + c * R[ro + 6 + col];
        }
      }
      const to = 3 * i;
      cT[to] = T[to] + dy[o + 3];
      cT[to + 1] = T[to + 1] + dy[o + 4];
      cT[to + 2] = T[to + 2] + dy[o + 5];
    }
    for (let c = 0; c < this.nC; c++) {
      const cb = this.camBlock[c];
      if (cb < 0) continue;
      const o = bStart[cb];
      for (let q = 0; q < this.camNK[c]; q++) {
        const k = 5 * c + this.camIdx[5 * c + q];
        this.cK[k] = this.K[k] + dy[o + q];
      }
    }
    const X = this.X,
      cX = this.cX;
    for (let j = 0; j < this.nX; j++) {
      if (!this.pointFree[j]) continue;
      const o3 = 3 * j;
      cX[o3] = X[o3] + dx[o3];
      cX[o3 + 1] = X[o3 + 1] + dx[o3 + 1];
      cX[o3 + 2] = X[o3 + 2] + dx[o3 + 2];
    }
  }

  /** Make the candidate current (swap buffers). */
  accept(): void {
    let tmp = this.R;
    this.R = this.cR;
    this.cR = tmp;
    tmp = this.T;
    this.T = this.cT;
    this.cT = tmp;
    tmp = this.K;
    this.K = this.cK;
    this.cK = tmp;
    tmp = this.X;
    this.X = this.cX;
    this.cX = tmp;
  }

  /** Copy the current parameters back into the problem (fixed entries untouched). */
  writeBack(): void {
    const p = this.problem;
    for (let i = 0; i < this.nP; i++) {
      if (p.poseFixed[i]) continue;
      const ps: Pose = p.poses[i];
      for (let k = 0; k < 9; k++) ps.R[k] = this.R[9 * i + k];
      for (let k = 0; k < 3; k++) ps.t[k] = this.T[3 * i + k];
    }
    for (let c = 0; c < this.nC; c++) {
      if (this.camNK[c] === 0) continue;
      const cam: CameraIntrinsics = p.cameras[c];
      cam.f = this.K[5 * c];
      cam.cx = this.K[5 * c + 1];
      cam.cy = this.K[5 * c + 2];
      cam.k1 = this.K[5 * c + 3];
      cam.k2 = this.K[5 * c + 4];
    }
    const pf = this.pointFree;
    const pts = p.points;
    for (let j = 0; j < this.nX; j++) {
      if (!pf[j]) continue;
      pts[3 * j] = this.X[3 * j];
      pts[3 * j + 1] = this.X[3 * j + 1];
      pts[3 * j + 2] = this.X[3 * j + 2];
    }
  }
}

function rmsOf(sqErr: number, n: number): number {
  return n > 0 ? Math.sqrt(sqErr / n) : 0;
}

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/**
 * Levenberg–Marquardt bundle adjustment of `problem` (refined in place: poses,
 * points, intrinsics per `cameraRefine`). See the module header for the method.
 *
 * Observations whose residual is not finite at the input (or with a non-finite /
 * negative weight) are ignored for the whole run. Parameters in the problem always
 * end at the best accepted state, also on cancellation, time-out or exceptions
 * thrown by `onIteration`.
 */
export function bundleAdjust(problem: BAProblem, options?: BAOptions): BAResult {
  const ba = new BundleAdjuster(problem, options);
  ba.run();
  return ba.result;
}

/**
 * `bundleAdjust` in resumable form: `run(pause)` iterates until the solve ends or `pause()`
 * (asked after every LM iteration) returns true, and the next `run` continues with the same
 * solver state (structure of S, linearisation, damping λ and ν, iteration count, time limit
 * counted from the construction). A caller can thus yield to the event loop between iterations
 * without rebuilding the problem: the result is the one of a single uninterrupted `bundleAdjust`
 * wherever the pauses fall. The problem is written back once, when the solve ends (also on an
 * exception thrown by `onIteration`); while paused it still holds the input. `result` is final
 * once `run` returned true.
 */
export class BundleAdjuster {
  readonly result: BAResult;
  private readonly opts: ResolvedBAOptions;
  private readonly eng: BAEngine;
  private readonly t0: number;
  private phase: 'start' | 'iterating' | 'done' = 'start';
  private cost: number;
  private sqErr: number;
  private lambda: number;
  private nu = 2;
  private iter = 0;
  private successful = 0;
  private lastFailureLinear = false;
  private xNorm = 0;

  constructor(problem: BAProblem, options?: BAOptions) {
    this.t0 = now();
    validateBAProblem(problem);
    const opts = (this.opts = resolveBAOptions(options));
    const eng = (this.eng = new BAEngine(problem, opts));
    const init = eng.evaluate(false);
    const nValid = eng.numValidObs;
    this.cost = init.cost;
    this.sqErr = init.sqErr;
    this.lambda = Math.min(Math.max(opts.initialLambda, MIN_LAMBDA), MAX_LAMBDA);
    this.result = {
      success: false,
      termination: 'invalid_initial_cost',
      iterations: 0,
      initialCost: init.cost,
      finalCost: init.cost,
      initialRmsPx: rmsOf(init.sqErr, nValid),
      finalRmsPx: rmsOf(init.sqErr, nValid),
      timeMs: 0,
      successfulIterations: 0,
      linearSolver: 'none',
      numReducedParams: 0,
      numFreePoints: 0,
      pcgIterations: 0,
      finalLambda: opts.initialLambda,
    };
    if (!(init.cost < Infinity)) this.end(null);
  }

  /** True once the solve has ended (`result` is final). */
  get finished(): boolean {
    return this.phase === 'done';
  }

  /** Iterate until the solve ends (→ true) or `pause()` returns true after an iteration (→ false). */
  run(pause?: () => boolean): boolean {
    if (this.phase === 'done') return true;
    const eng = this.eng,
      opts = this.opts,
      result = this.result;
    if (this.phase === 'start') {
      if (opts.cancel?.cancelled) {
        // (before the O(#obs) structure build: cancellation should be cheap)
        result.success = true;
        this.end('cancelled', false);
        return true;
      }
      eng.buildStructure();
      result.numReducedParams = eng.nFree;
      result.numFreePoints = eng.numFreePoints;
      if (eng.numParams === 0) {
        result.success = true;
        this.end('no_free_parameters', false);
        return true;
      }
      result.linearSolver = eng.useDense ? 'dense_schur' : 'pcg_schur';
    }
    let termination: BATermination | null = null;
    try {
      if (this.phase === 'start') {
        this.phase = 'iterating';
        eng.linearize();
        if (eng.gradMaxAbs() <= opts.gradientTolerance) termination = 'gradient_tolerance';
        else this.xNorm = eng.paramNorm();
      }
      while (termination === null) {
        if (this.iter >= opts.maxIterations) {
          termination = 'max_iterations';
          break;
        }
        if (opts.cancel?.cancelled) {
          termination = 'cancelled';
          break;
        }
        if (now() - this.t0 > opts.timeLimitMs) {
          termination = 'time_limit';
          break;
        }
        this.iter++;
        let accepted = false;
        let stop: BATermination | null = null;
        if (eng.solve(this.lambda)) {
          this.lastFailureLinear = false;
          // A step below the parameter tolerance ends the run, but it is still taken when it
          // decreases the cost (one extra evaluation; Ceres would discard it).
          const smallStep = eng.stepNorm() <= opts.parameterTolerance * (this.xNorm + opts.parameterTolerance);
          const dm = eng.modelDecrease();
          if (dm > 0 && dm < Infinity) {
            eng.applyStep();
            const cand = eng.evaluate(true);
            const actual = this.cost - cand.cost;
            const rho = actual / dm;
            if (cand.cost < Infinity && rho > MIN_RELATIVE_DECREASE) {
              eng.accept();
              const prev = this.cost;
              this.cost = cand.cost;
              this.sqErr = cand.sqErr;
              this.successful++;
              accepted = true;
              const tmp = 2 * rho - 1;
              this.lambda *= Math.max(1 / 3, 1 - tmp * tmp * tmp);
              if (this.lambda < MIN_LAMBDA) this.lambda = MIN_LAMBDA;
              this.nu = 2;
              if (smallStep) {
                stop = 'parameter_tolerance';
              } else if (actual <= opts.functionTolerance * prev) {
                stop = 'function_tolerance';
              } else if (this.iter < opts.maxIterations) {
                // (skipped after the last allowed iteration: nothing would use it)
                eng.linearize();
                this.xNorm = eng.paramNorm();
                if (eng.gradMaxAbs() <= opts.gradientTolerance) stop = 'gradient_tolerance';
              }
            }
          }
          if (smallStep) stop = 'parameter_tolerance';
        } else {
          this.lastFailureLinear = true;
        }
        if (!accepted && stop === null) {
          this.lambda *= this.nu;
          this.nu *= 2;
          if (this.lambda > MAX_LAMBDA) stop = this.lastFailureLinear ? 'linear_solver_failure' : 'min_trust_region_radius';
        }
        opts.onIteration?.(this.iter, this.cost);
        if (stop !== null) {
          termination = stop;
          break;
        }
        if (pause?.()) return false;
      }
    } catch (e) {
      this.end(termination ?? 'max_iterations');
      throw e;
    }
    this.end(termination);
    return true;
  }

  /**
   * End the solve. `iterated`: the LM loop ran, so the best state is written back and the
   * iteration counters are reported; otherwise (invalid input, cancelled before the structure was
   * built, nothing free) the problem is left untouched.
   */
  private end(termination: BATermination | null, iterated = true): void {
    if (this.phase === 'done') return;
    const result = this.result;
    if (termination !== null) {
      result.termination = termination;
      if (iterated) {
        this.eng.writeBack();
        result.success = termination !== 'linear_solver_failure' && this.cost < Infinity;
        result.iterations = this.iter;
        result.successfulIterations = this.successful;
        result.finalCost = this.cost;
        result.finalRmsPx = rmsOf(this.sqErr, this.eng.numValidObs);
        result.pcgIterations = this.eng.pcgIterations;
        result.finalLambda = this.lambda;
      }
    }
    result.timeMs = now() - this.t0;
    this.phase = 'done';
  }
}

/**
 * Unweighted reprojection error (pixels) per observation at the problem's current
 * parameters (Infinity / NaN where the projection is not finite).
 */
export function reprojectionErrors(problem: BAProblem, out?: Float64Array): Float64Array {
  const nO = problem.obsPose.length;
  const res = out && out.length >= nO ? out : new Float64Array(nO);
  const K = new Float64Array(5 * problem.cameras.length);
  for (let c = 0; c < problem.cameras.length; c++) {
    const cam = problem.cameras[c];
    K[5 * c] = cam.f;
    K[5 * c + 1] = cam.cx;
    K[5 * c + 2] = cam.cy;
    K[5 * c + 3] = cam.k1;
    K[5 * c + 4] = cam.k2;
  }
  const r2 = new Float64Array(2);
  const pts = problem.points,
    uv = problem.obsUV;
  for (let o = 0; o < nO; o++) {
    const i = problem.obsPose[o];
    const ps = problem.poses[i];
    radialResidual(ps.R, 0, ps.t, 0, pts, 3 * problem.obsPoint[o], K, 5 * problem.imageCamera[i], uv[2 * o], uv[2 * o + 1], r2);
    res[o] = Math.sqrt(r2[0] * r2[0] + r2[1] * r2[1]);
  }
  return res;
}

/**
 * Total cost `½Σρ(‖w r‖²) + ½Σ‖prior‖²` (as reported by `bundleAdjust`) and the RMS
 * unweighted reprojection error at the problem's current parameters. Only the
 * `loss` / `lossScalePx` options matter.
 */
export function evaluateBACost(problem: BAProblem, options?: BAOptions): { cost: number; rmsPx: number } {
  validateBAProblem(problem);
  const eng = new BAEngine(problem, resolveBAOptions(options));
  const e = eng.evaluate(false);
  return { cost: e.cost, rmsPx: rmsOf(e.sqErr, eng.numValidObs) };
}

/** Gradient of the BA cost w.r.t. every parameter (0 for fixed / pinned ones). */
export interface BAGradient {
  cost: number;
  /** 6 per pose: [δω (left-multiplicative), δt]. */
  poses: Float64Array;
  /** 3 per point. */
  points: Float64Array;
  /** 5 per camera: [f, cx, cy, k1, k2]. */
  cameras: Float64Array;
}

/**
 * Gradient of the cost reported by `bundleAdjust` at the problem's current
 * parameters (the exact gradient the LM model uses; for diagnostics and tests).
 */
export function computeBAGradient(problem: BAProblem, options?: BAOptions): BAGradient {
  validateBAProblem(problem);
  const eng = new BAEngine(problem, resolveBAOptions(options));
  const e = eng.evaluate(false);
  eng.buildStructure();
  eng.linearize();
  const g: BAGradient = {
    cost: e.cost,
    poses: new Float64Array(6 * eng.nP),
    points: new Float64Array(3 * eng.nX),
    cameras: new Float64Array(5 * eng.nC),
  };
  const bStart = eng.S.blockStart;
  for (let i = 0; i < eng.nP; i++) {
    const pb = eng.poseBlock[i];
    if (pb < 0) continue;
    for (let k = 0; k < 6; k++) g.poses[6 * i + k] = eng.gY[bStart[pb] + k];
  }
  for (let c = 0; c < eng.nC; c++) {
    const cb = eng.camBlock[c];
    if (cb < 0) continue;
    for (let q = 0; q < eng.camNK[c]; q++) g.cameras[5 * c + eng.camIdx[5 * c + q]] = eng.gY[bStart[cb] + q];
  }
  for (let j = 0; j < eng.nX; j++) {
    if (!eng.pointFree[j]) continue;
    for (let k = 0; k < 3; k++) g.points[3 * j + k] = eng.gX[3 * j + k];
  }
  return g;
}
