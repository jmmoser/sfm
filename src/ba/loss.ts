/**
 * Robust loss functions ρ(s) of the squared residual norm s = ‖r‖² (Ceres
 * conventions; Triggs et al., "Bundle Adjustment — A Modern Synthesis", 2000, §6.8).
 * With scale a (b = a²):
 *
 *   trivial  ρ = s
 *   huber    ρ = s                    (s ≤ b)     2a√s − b              (s > b)
 *   soft_l1  ρ = 2b (√(1 + s/b) − 1)
 *   cauchy   ρ = b log(1 + s/b)
 *
 * All satisfy ρ(0) = 0, ρ'(0) = 1, so the cost ½ρ(s) ≈ ½s for inliers. The solver uses
 * IRLS weighting: residual and Jacobian scaled by √ρ'(s), which gives the exact
 * gradient ρ' Jᵀr of ½ρ and the Gauss–Newton Hessian approximation ρ' JᵀJ.
 */
import type { BALossType } from './problem';

export const LOSS_TRIVIAL = 0;
export const LOSS_HUBER = 1;
export const LOSS_CAUCHY = 2;
export const LOSS_SOFT_L1 = 3;

/** Numeric loss kind used in hot loops. */
export type LossKind = typeof LOSS_TRIVIAL | typeof LOSS_HUBER | typeof LOSS_CAUCHY | typeof LOSS_SOFT_L1;

/** Map a loss name to its numeric kind (throws on unknown names). */
export function lossKind(name: BALossType): LossKind {
  switch (name) {
    case 'trivial':
      return LOSS_TRIVIAL;
    case 'huber':
      return LOSS_HUBER;
    case 'cauchy':
      return LOSS_CAUCHY;
    case 'soft_l1':
      return LOSS_SOFT_L1;
    default:
      throw new RangeError(`BA: unknown loss '${String(name)}'`);
  }
}

const MIN_WEIGHT = 2.2250738585072014e-308; // DBL_MIN, like Ceres: keep ρ' strictly positive

/** ρ(s) for loss `kind` with squared scale `b` (= a²). */
export function lossRho(kind: LossKind, s: number, b: number): number {
  switch (kind) {
    case LOSS_HUBER:
      return s > b ? 2 * Math.sqrt(b * s) - b : s;
    case LOSS_CAUCHY: {
      const x = s / b;
      return x < Infinity ? b * Math.log1p(x) : b * (Math.log(s) - Math.log(b));
    }
    case LOSS_SOFT_L1: {
      // 2b(√(1+s/b) − 1) without cancellation (that form returns 0 for s/b ≲ 1e-16)
      const x = s / b;
      return x < Infinity ? (2 * s) / (Math.sqrt(1 + x) + 1) : 2 * Math.sqrt(s) * Math.sqrt(b);
    }
    default:
      return s;
  }
}

/** ρ'(s) (the IRLS weight) for loss `kind` with squared scale `b`. */
export function lossWeight(kind: LossKind, s: number, b: number): number {
  switch (kind) {
    case LOSS_HUBER:
      return s > b ? Math.max(MIN_WEIGHT, Math.sqrt(b / s)) : 1;
    case LOSS_CAUCHY:
      return Math.max(MIN_WEIGHT, 1 / (1 + s / b));
    case LOSS_SOFT_L1:
      return Math.max(MIN_WEIGHT, 1 / Math.sqrt(1 + s / b));
    default:
      return 1;
  }
}

/**
 * Evaluate [ρ(s), ρ'(s), ρ''(s)] into `out` (length ≥ 3). ρ'' is provided for
 * completeness / tests; the solver does not use it (IRLS, no Triggs correction).
 */
export function evaluateLoss(kind: LossKind, s: number, b: number, out: Float64Array): Float64Array {
  out[0] = lossRho(kind, s, b);
  out[1] = lossWeight(kind, s, b);
  switch (kind) {
    case LOSS_HUBER:
      out[2] = s > b ? -out[1] / (2 * s) : 0;
      break;
    case LOSS_CAUCHY: {
      const inv = 1 / (1 + s / b);
      out[2] = -(inv * inv) / b;
      break;
    }
    case LOSS_SOFT_L1: {
      const sum = 1 + s / b;
      out[2] = -1 / (2 * b * sum * Math.sqrt(sum));
      break;
    }
    default:
      out[2] = 0;
  }
  return out;
}
