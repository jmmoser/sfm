/**
 * Bundle-adjustment problem / options / result types (docs/ARCHITECTURE.md §4.3)
 * and input validation.
 *
 * Parameterisation used by the solver (all conventions from §2):
 *  - Pose i: world-to-camera `x_cam = R X + t`, updated left-multiplicatively
 *    `R ← exp([δω]×) R`, `t ← t + δt`. Local pose parameter order: [δω (3), δt (3)].
 *  - Camera c (RADIAL): `[f, cx, cy, k1, k2]`, additive updates; which of them are
 *    free is given by `cameraRefine[c]` (focal → f; principalPoint → cx, cy;
 *    distortion → k1, k2).
 *  - Point j: world XYZ, additive updates.
 *
 * Cost (Ceres convention): `½ Σ_obs ρ(‖w (π(X) − uv)‖²) + ½ Σ_priors ‖r_prior‖²`
 * (priors: camera centres, relative rotations, focal lengths, principal points)
 * where `w = obsWeight[o]` (1/σ, default 1) and ρ is the robust loss with scale
 * `lossScalePx` applied to the WEIGHTED residual (i.e. in units of σ when weights
 * are 1/σ). Priors use the trivial loss.
 */
import type { CameraIntrinsics, CancelToken, Mat3, Pose, Vec3 } from '../core/types';

/** Which intrinsics of a camera are refined. */
export interface BACameraRefine {
  focal: boolean;
  principalPoint: boolean;
  distortion: boolean;
}

/** Camera-centre prior: residual `(C − center)/sigma`, `C = −Rᵀ t` (e.g. GPS, same world frame). */
export interface BACenterPrior {
  pose: number;
  center: Vec3;
  sigma: number;
}

/**
 * Relative-rotation prior `R_j ≈ R_ji R_i` (e.g. integrated gyro): residual
 * `log(R_j R_iᵀ R_jiᵀ)/sigma` (rotation vector, radians/sigma).
 */
export interface BARelRotationPrior {
  i: number;
  j: number;
  R_ji: Mat3;
  sigma: number;
}

/**
 * Focal-length prior of a camera: residual `(f_c − f)/sigma` (pixels / pixels). Only acts when the
 * camera's focal is refined. E.g. per-image focal lengths (focus breathing) tied to a shared value.
 */
export interface BAFocalPrior {
  camera: number;
  f: number;
  sigma: number;
}

/**
 * Principal-point prior of a camera: residual `((cx_c − cx)/sigma, (cy_c − cy)/sigma)` (pixels /
 * pixels). Only acts when the camera's principal point is refined. E.g. a weak pull towards the
 * image centre or an earlier estimate, to damp a weakly observable principal-point mode.
 */
export interface BAPrincipalPointPrior {
  camera: number;
  cx: number;
  cy: number;
  sigma: number;
}

export interface BAProblem {
  /** Intrinsics per camera; refined in place. */
  cameras: CameraIntrinsics[];
  /** One entry per camera. */
  cameraRefine: BACameraRefine[];
  /** World-to-camera poses; refined in place (the R / t arrays are overwritten). */
  poses: Pose[];
  /** Camera index per pose. */
  imageCamera: Uint32Array;
  /** 1 ⇒ pose held constant. */
  poseFixed: Uint8Array;
  /** Per pose: bit0 = x, bit1 = y, bit2 = z component of t held constant (gauge). */
  translationFixedMask?: Uint8Array;
  /** 3 per point (world), refined in place. */
  points: Float64Array;
  /** 1 ⇒ point held constant. */
  pointFixed?: Uint8Array;
  /** Observation → pose index. */
  obsPose: Uint32Array;
  /** Observation → point index. */
  obsPoint: Uint32Array;
  /** 2 per observation, full-resolution pixels (pixel centre at integer coords). */
  obsUV: Float64Array;
  /** Optional per-observation weight (1/σ); multiplies the residual. */
  obsWeight?: Float32Array;
  centerPriors?: BACenterPrior[];
  relRotationPriors?: BARelRotationPrior[];
  focalPriors?: BAFocalPrior[];
  principalPointPriors?: BAPrincipalPointPrior[];
}

export type BALossType = 'trivial' | 'huber' | 'cauchy' | 'soft_l1';
export type BALinearSolverType = 'auto' | 'dense_schur' | 'pcg_schur';

export interface BAOptions {
  /** Robust loss on the weighted reprojection residual. Default 'cauchy'. */
  loss?: BALossType;
  /** Loss scale a (pixels, or σ units with 1/σ weights); a² is clamped to [1e-150, 1e150]. Default 1.0. */
  lossScalePx?: number;
  /** Maximum number of LM iterations (successful + unsuccessful). Default 50. */
  maxIterations?: number;
  /** Stop when |Δcost| ≤ functionTolerance · cost after an accepted step. Default 1e-6. */
  functionTolerance?: number;
  /** Stop when max |gradient| ≤ gradientTolerance. Default 1e-10. */
  gradientTolerance?: number;
  /**
   * Stop when ‖δ‖ ≤ parameterTolerance · (‖x‖ + parameterTolerance) (that last small step is
   * still taken if it decreases the cost). ‖x‖ is over ALL free parameters incl. points, so on
   * large scenes this can stop early; pass 0 to disable. Default 1e-8.
   */
  parameterTolerance?: number;
  /** Initial LM damping λ (= 1 / trust-region radius). Default 1e-4. */
  initialLambda?: number;
  /**
   * 'auto' (default): direct ('dense_schur', skyline Cholesky) when the reduced system has
   * ≤ 2400 free parameters — or more but a small Cholesky envelope (≤ 4M entries, ≤ 3e8
   * multiply-adds, e.g. banded sequential captures) — else block-Jacobi PCG.
   * A forced 'dense_schur' falls back to PCG if the envelope exceeds 18M entries.
   */
  linearSolver?: BALinearSolverType;
  /** Max PCG iterations per linear solve. Default 500. */
  pcgMaxIterations?: number;
  /** PCG stops at ‖r‖ ≤ pcgTolerance · ‖b‖. Default 1e-6. */
  pcgTolerance?: number;
  /** Wall-clock limit (ms); checked once per iteration. Default ∞. */
  timeLimitMs?: number;
  cancel?: CancelToken;
  /** Called after every iteration with the current (accepted) cost. */
  onIteration?: (iter: number, cost: number) => void;
}

/**
 * Why the solver stopped. Convergence criteria, iteration/time limits and
 * cancellation leave a usable (never worse than the input) solution.
 */
export type BATermination =
  | 'function_tolerance'
  | 'gradient_tolerance'
  | 'parameter_tolerance'
  | 'min_trust_region_radius'
  | 'max_iterations'
  | 'time_limit'
  | 'cancelled'
  | 'no_free_parameters'
  | 'invalid_initial_cost'
  | 'linear_solver_failure';

export interface BAResult {
  /** True when the solution is usable (finite cost, no failure). */
  success: boolean;
  termination: BATermination;
  /** LM iterations performed (successful + unsuccessful). */
  iterations: number;
  /** ½ Σ ρ + ½ Σ ‖prior‖² at the input / output parameters. */
  initialCost: number;
  finalCost: number;
  /** RMS unweighted reprojection error (px) over all observations. */
  initialRmsPx: number;
  finalRmsPx: number;
  timeMs: number;
  // ---- extensions (not in §4.3) ----
  /** Accepted LM steps. */
  successfulIterations: number;
  /** Linear solver actually used ('none' when nothing was solved). */
  linearSolver: 'dense_schur' | 'pcg_schur' | 'none';
  /** Number of free parameters in the reduced (camera) system. */
  numReducedParams: number;
  /** Number of free (eliminated) points. */
  numFreePoints: number;
  /** Total PCG iterations (0 for dense). */
  pcgIterations: number;
  /** Damping at exit. */
  finalLambda: number;
}

/** Resolved options with defaults applied. */
export interface ResolvedBAOptions {
  loss: BALossType;
  lossScalePx: number;
  maxIterations: number;
  functionTolerance: number;
  gradientTolerance: number;
  parameterTolerance: number;
  initialLambda: number;
  linearSolver: BALinearSolverType;
  pcgMaxIterations: number;
  pcgTolerance: number;
  timeLimitMs: number;
  cancel: CancelToken | undefined;
  onIteration: ((iter: number, cost: number) => void) | undefined;
}

/** Reduced-system size up to which 'auto' uses the dense Schur solver. */
export const DENSE_SCHUR_MAX_SIZE = 2400;

export const DEFAULT_BA_OPTIONS: Readonly<Omit<ResolvedBAOptions, 'cancel' | 'onIteration'>> = Object.freeze({
  loss: 'cauchy',
  lossScalePx: 1.0,
  maxIterations: 50,
  functionTolerance: 1e-6,
  gradientTolerance: 1e-10,
  parameterTolerance: 1e-8,
  initialLambda: 1e-4,
  linearSolver: 'auto',
  pcgMaxIterations: 500,
  pcgTolerance: 1e-6,
  timeLimitMs: Infinity,
});

/** Apply defaults (undefined / non-finite numbers fall back to the default). */
export function resolveBAOptions(o: BAOptions = {}): ResolvedBAOptions {
  const D = DEFAULT_BA_OPTIONS;
  const num = (v: number | undefined, d: number, min = 0): number =>
    v !== undefined && v === v && v >= min ? v : d;
  return {
    loss: o.loss ?? D.loss,
    lossScalePx: num(o.lossScalePx, D.lossScalePx, Number.MIN_VALUE),
    maxIterations: Math.floor(num(o.maxIterations, D.maxIterations)),
    functionTolerance: num(o.functionTolerance, D.functionTolerance),
    gradientTolerance: num(o.gradientTolerance, D.gradientTolerance),
    parameterTolerance: num(o.parameterTolerance, D.parameterTolerance),
    initialLambda: num(o.initialLambda, D.initialLambda, Number.MIN_VALUE),
    linearSolver: o.linearSolver ?? D.linearSolver,
    pcgMaxIterations: Math.max(1, Math.floor(num(o.pcgMaxIterations, D.pcgMaxIterations, 1))),
    pcgTolerance: num(o.pcgTolerance, D.pcgTolerance),
    timeLimitMs: num(o.timeLimitMs, D.timeLimitMs),
    cancel: o.cancel,
    onIteration: o.onIteration,
  };
}

/**
 * Throw a RangeError if array sizes / indices of `p` are inconsistent. Cheap (O(#obs)).
 */
export function validateBAProblem(p: BAProblem): void {
  const nC = p.cameras.length;
  const nP = p.poses.length;
  if (p.points.length % 3 !== 0) throw new RangeError('BA: points.length must be a multiple of 3');
  const nX = p.points.length / 3;
  const nO = p.obsPose.length;
  if (p.cameraRefine.length < nC) throw new RangeError('BA: cameraRefine must have one entry per camera');
  if (p.imageCamera.length < nP) throw new RangeError('BA: imageCamera must have one entry per pose');
  if (p.poseFixed.length < nP) throw new RangeError('BA: poseFixed must have one entry per pose');
  if (p.translationFixedMask && p.translationFixedMask.length < nP)
    throw new RangeError('BA: translationFixedMask must have one entry per pose');
  if (p.pointFixed && p.pointFixed.length < nX) throw new RangeError('BA: pointFixed must have one entry per point');
  if (p.obsPoint.length !== nO) throw new RangeError('BA: obsPoint / obsPose length mismatch');
  if (p.obsUV.length < 2 * nO) throw new RangeError('BA: obsUV must have 2 entries per observation');
  if (p.obsWeight && p.obsWeight.length < nO) throw new RangeError('BA: obsWeight must have one entry per observation');
  for (let i = 0; i < nP; i++) {
    if (p.imageCamera[i] >= nC) throw new RangeError(`BA: imageCamera[${i}] = ${p.imageCamera[i]} out of range`);
    const ps = p.poses[i];
    if (!ps || ps.R.length < 9 || ps.t.length < 3) throw new RangeError(`BA: pose ${i} malformed`);
  }
  for (let o = 0; o < nO; o++) {
    if (p.obsPose[o] >= nP) throw new RangeError(`BA: obsPose[${o}] = ${p.obsPose[o]} out of range`);
    if (p.obsPoint[o] >= nX) throw new RangeError(`BA: obsPoint[${o}] = ${p.obsPoint[o]} out of range`);
  }
  if (p.centerPriors) {
    for (const cp of p.centerPriors) {
      if (!(cp.pose >= 0 && cp.pose < nP) || (cp.pose | 0) !== cp.pose)
        throw new RangeError(`BA: centerPrior pose ${cp.pose} out of range`);
      if (!(cp.sigma > 0)) throw new RangeError('BA: centerPrior sigma must be > 0');
    }
  }
  if (p.focalPriors) {
    for (const fp of p.focalPriors) {
      if (!(fp.camera >= 0 && fp.camera < nC) || (fp.camera | 0) !== fp.camera) throw new RangeError(`BA: focalPrior camera ${fp.camera} out of range`);
      if (!(fp.sigma > 0) || !Number.isFinite(fp.f)) throw new RangeError('BA: focalPrior needs a finite f and sigma > 0');
    }
  }
  if (p.principalPointPriors) {
    for (const pp of p.principalPointPriors) {
      if (!(pp.camera >= 0 && pp.camera < nC) || (pp.camera | 0) !== pp.camera) throw new RangeError(`BA: principalPointPrior camera ${pp.camera} out of range`);
      if (!(pp.sigma > 0) || !Number.isFinite(pp.cx) || !Number.isFinite(pp.cy)) throw new RangeError('BA: principalPointPrior needs a finite cx, cy and sigma > 0');
    }
  }
  if (p.relRotationPriors) {
    for (const rp of p.relRotationPriors) {
      if (!(rp.i >= 0 && rp.i < nP) || !(rp.j >= 0 && rp.j < nP) || (rp.i | 0) !== rp.i || (rp.j | 0) !== rp.j)
        throw new RangeError(`BA: relRotationPrior (${rp.i}, ${rp.j}) out of range`);
      if (rp.i === rp.j) throw new RangeError('BA: relRotationPrior needs i ≠ j');
      if (!(rp.sigma > 0)) throw new RangeError('BA: relRotationPrior sigma must be > 0');
    }
  }
}
