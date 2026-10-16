/**
 * Pose priors for bundle adjustment: residuals and analytic Jacobians w.r.t. the
 * left-multiplicative pose update `R ← exp([δω]×) R`, `t ← t + δt`.
 *
 * Centre prior:  r = (C − c)/σ,  C = −Rᵀ t.
 *   C(δω, δt) = −Rᵀ exp(−[δω]×)(t + δt) ⇒ ∂r/∂δω = −Rᵀ[t]×/σ,  ∂r/∂δt = −Rᵀ/σ.
 *
 * Relative-rotation prior (R_j ≈ R_ji R_i):  r = φ/σ,  φ = log(E),  E = R_j R_iᵀ R_jiᵀ.
 *   R_j ← exp(b) R_j:  E ← exp(b) E                 ⇒ ∂φ/∂b = J_l⁻¹(φ)
 *   R_i ← exp(a) R_i:  E ← exp(−M a) E,  M = R_j R_iᵀ ⇒ ∂φ/∂a = −J_l⁻¹(φ) M
 *   using log(exp(δ) exp(φ)) ≈ φ + J_l⁻¹(φ) δ (left Jacobian of SO(3), Barfoot,
 *   "State Estimation for Robotics", 2017, §7.1.5). The Jacobian is exact (no
 *   small-residual approximation).
 */
import type { Mat3 } from '../core/types';
import { mat3MulTranspose } from '../math/mat3';
import { mat3ToRotvec } from '../math/rotation';

/**
 * Inverse left Jacobian of SO(3): J_l⁻¹(φ) = I − ½[φ]× + (1/θ² − (1+cos θ)/(2θ sin θ)) [φ]×²
 * (row-major 3x3), θ = ‖φ‖ ∈ [0, π]. Taylor series below θ = 1e-4.
 */
export function so3LeftJacobianInverse(phi: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  const x = phi[0],
    y = phi[1],
    z = phi[2];
  const t2 = x * x + y * y + z * z;
  let c: number;
  if (t2 < 1e-8) {
    c = 1 / 12 + t2 / 720;
  } else {
    const th = Math.sqrt(t2);
    const h = 0.5 * th;
    // (1 − h cot h)/θ²
    c = (1 - (h * Math.cos(h)) / Math.sin(h)) / t2;
  }
  // [φ]×² = φφᵀ − θ² I
  out[0] = 1 + c * (x * x - t2);
  out[1] = 0.5 * z + c * x * y;
  out[2] = -0.5 * y + c * x * z;
  out[3] = -0.5 * z + c * x * y;
  out[4] = 1 + c * (y * y - t2);
  out[5] = 0.5 * x + c * y * z;
  out[6] = 0.5 * y + c * x * z;
  out[7] = -0.5 * x + c * y * z;
  out[8] = 1 + c * (z * z - t2);
  return out;
}

/**
 * Centre prior residual `r = (C − c)·invSigma` (3) and Jacobian `J` (3x6 row-major,
 * columns [δω, δt]) at pose (R at rOff, t at tOff).
 */
export function centerPriorResidualJacobian(
  R: ArrayLike<number>,
  rOff: number,
  t: ArrayLike<number>,
  tOff: number,
  c: ArrayLike<number>,
  invSigma: number,
  outR: Float64Array,
  outJ: Float64Array,
): void {
  const t0 = t[tOff],
    t1 = t[tOff + 1],
    t2 = t[tOff + 2];
  // Rᵀ columns are R rows: (Rᵀ)[i][k] = R[k][i]
  for (let i = 0; i < 3; i++) {
    const a = R[rOff + i],
      b = R[rOff + 3 + i],
      d = R[rOff + 6 + i]; // row i of Rᵀ = column i of R
    const Ci = -(a * t0 + b * t1 + d * t2);
    outR[i] = (Ci - c[i]) * invSigma;
    // −(Rᵀ[t]×) row i: [t]× = [[0,−t2,t1],[t2,0,−t0],[−t1,t0,0]]
    // (Rᵀ[t]×)[i] = (b t2 − d t1, d t0 − a t2, a t1 − b t0)
    const o = 6 * i;
    outJ[o] = -(b * t2 - d * t1) * invSigma;
    outJ[o + 1] = -(d * t0 - a * t2) * invSigma;
    outJ[o + 2] = -(a * t1 - b * t0) * invSigma;
    outJ[o + 3] = -a * invSigma;
    outJ[o + 4] = -b * invSigma;
    outJ[o + 5] = -d * invSigma;
  }
}

const scrM = new Float64Array(9);
const scrE = new Float64Array(9);
const scrPhi = new Float64Array(3);
const scrJinv = new Float64Array(9);

/**
 * Relative-rotation prior residual `r = log(R_j R_iᵀ R_jiᵀ)·invSigma` (3) and the
 * Jacobians w.r.t. δω_i (`outJi`, 3x3) and δω_j (`outJj`, 3x3).
 */
export function relRotationPriorResidualJacobian(
  Ri: ArrayLike<number>,
  riOff: number,
  Rj: ArrayLike<number>,
  rjOff: number,
  R_ji: ArrayLike<number>,
  invSigma: number,
  outR: Float64Array,
  outJi: Float64Array,
  outJj: Float64Array,
): void {
  // M = R_j R_iᵀ
  const M = scrM;
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      M[3 * r + c] =
        Rj[rjOff + 3 * r] * Ri[riOff + 3 * c] +
        Rj[rjOff + 3 * r + 1] * Ri[riOff + 3 * c + 1] +
        Rj[rjOff + 3 * r + 2] * Ri[riOff + 3 * c + 2];
    }
  }
  mat3MulTranspose(M, R_ji, scrE);
  const phi = mat3ToRotvec(scrE, scrPhi);
  outR[0] = phi[0] * invSigma;
  outR[1] = phi[1] * invSigma;
  outR[2] = phi[2] * invSigma;
  const Jl = so3LeftJacobianInverse(phi, scrJinv);
  for (let k = 0; k < 9; k++) outJj[k] = Jl[k] * invSigma;
  // Ji = −Jl M
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      outJi[3 * r + c] =
        -(Jl[3 * r] * M[c] + Jl[3 * r + 1] * M[3 + c] + Jl[3 * r + 2] * M[6 + c]) * invSigma;
    }
  }
}
