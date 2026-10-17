/**
 * RADIAL camera projection and its analytic Jacobians, as used by the bundle
 * adjuster (kept local to `src/ba`; must match docs/ARCHITECTURE.md §2 exactly):
 *
 *   Xc = R X + t,  x = Xc/Zc, y = Yc/Zc,  r² = x² + y²,  d = 1 + k1 r² + k2 r⁴
 *   u = f d x + cx,  v = f d y + cy
 *
 * Pose perturbation is left-multiplicative, `R ← exp([δω]×) R`, `t ← t + δt`, so
 * ∂Xc/∂δω = −[R X]×, ∂Xc/∂δt = I, ∂Xc/∂X = R.
 *
 * Arrays are passed with offsets so the solver can work on flat parameter arrays
 * (R: 9 row-major, t: 3, X: 3, K: [f, cx, cy, k1, k2]).
 */

/** |Zc| is clamped to at least this value (keeps residuals finite at the camera centre). */
export const BA_MIN_DEPTH = 1e-12;

/** Offsets into the output of `radialResidualJacobian`. */
export const RJ_RES = 0; // [ru, rv] = predicted − observed
export const RJ_POSE = 2; // 2x6 row-major, columns [δω0 δω1 δω2 δt0 δt1 δt2]
export const RJ_POINT = 14; // 2x3 row-major
export const RJ_INTR = 20; // 2x5 row-major, columns [f, cx, cy, k1, k2]
export const RJ_SIZE = 30;

/** Intrinsic parameter indices within the 5-vector [f, cx, cy, k1, k2]. */
export const BA_INTR_F = 0;
export const BA_INTR_CX = 1;
export const BA_INTR_CY = 2;
export const BA_INTR_K1 = 3;
export const BA_INTR_K2 = 4;

/**
 * Reprojection residual (predicted − observed, pixels, unweighted) into out[0..1].
 * Returns false if the residual is not finite.
 */
export function radialResidual(
  R: ArrayLike<number>,
  rOff: number,
  t: ArrayLike<number>,
  tOff: number,
  X: ArrayLike<number>,
  xOff: number,
  K: ArrayLike<number>,
  kOff: number,
  uObs: number,
  vObs: number,
  out: Float64Array,
): boolean {
  const X0 = X[xOff],
    X1 = X[xOff + 1],
    X2 = X[xOff + 2];
  const xc = R[rOff] * X0 + R[rOff + 1] * X1 + R[rOff + 2] * X2 + t[tOff];
  const yc = R[rOff + 3] * X0 + R[rOff + 4] * X1 + R[rOff + 5] * X2 + t[tOff + 1];
  let zc = R[rOff + 6] * X0 + R[rOff + 7] * X1 + R[rOff + 8] * X2 + t[tOff + 2];
  if (!(zc > BA_MIN_DEPTH || zc < -BA_MIN_DEPTH)) zc = zc < 0 ? -BA_MIN_DEPTH : BA_MIN_DEPTH;
  const iz = 1 / zc;
  const x = xc * iz,
    y = yc * iz;
  const f = K[kOff];
  const r2 = x * x + y * y;
  const fd = f * (1 + r2 * (K[kOff + 3] + K[kOff + 4] * r2));
  const ru = fd * x + K[kOff + 1] - uObs;
  const rv = fd * y + K[kOff + 2] - vObs;
  out[0] = ru;
  out[1] = rv;
  return ru - ru === 0 && rv - rv === 0;
}

/**
 * Residual (predicted − observed, unweighted) and analytic Jacobians of one
 * observation, written into `out` (length ≥ RJ_SIZE) at the RJ_* offsets.
 * Returns false if the residual or Jacobian is not finite.
 */
export function radialResidualJacobian(
  R: ArrayLike<number>,
  rOff: number,
  t: ArrayLike<number>,
  tOff: number,
  X: ArrayLike<number>,
  xOff: number,
  K: ArrayLike<number>,
  kOff: number,
  uObs: number,
  vObs: number,
  out: Float64Array,
): boolean {
  const r0 = R[rOff],
    r1 = R[rOff + 1],
    r2_ = R[rOff + 2],
    r3 = R[rOff + 3],
    r4 = R[rOff + 4],
    r5 = R[rOff + 5],
    r6 = R[rOff + 6],
    r7 = R[rOff + 7],
    r8 = R[rOff + 8];
  const X0 = X[xOff],
    X1 = X[xOff + 1],
    X2 = X[xOff + 2];
  // Rotated point p = R X (the δω Jacobian needs it without t).
  const px = r0 * X0 + r1 * X1 + r2_ * X2;
  const py = r3 * X0 + r4 * X1 + r5 * X2;
  const pz = r6 * X0 + r7 * X1 + r8 * X2;
  const xc = px + t[tOff];
  const yc = py + t[tOff + 1];
  let zc = pz + t[tOff + 2];
  if (!(zc > BA_MIN_DEPTH || zc < -BA_MIN_DEPTH)) zc = zc < 0 ? -BA_MIN_DEPTH : BA_MIN_DEPTH;
  const iz = 1 / zc;
  const x = xc * iz,
    y = yc * iz;
  const f = K[kOff],
    k1 = K[kOff + 3],
    k2 = K[kOff + 4];
  const rr = x * x + y * y;
  const d = 1 + rr * (k1 + k2 * rr);
  const fd = f * d;
  const ru = fd * x + K[kOff + 1] - uObs;
  const rv = fd * y + K[kOff + 2] - vObs;
  out[RJ_RES] = ru;
  out[RJ_RES + 1] = rv;

  // ∂(u, v)/∂(x, y): f (d I + 2 (k1 + 2 k2 r²) [x y]ᵀ[x y])
  const dd = 2 * (k1 + 2 * k2 * rr);
  const fdd = f * dd;
  const a = fd + fdd * x * x;
  const b = fdd * x * y;
  const c = fd + fdd * y * y;
  // ∂(x, y)/∂Xc = [[1/z, 0, −x/z], [0, 1/z, −y/z]]  ⇒ J_proj (2x3) = rows A, B
  const A0 = a * iz,
    A1 = b * iz,
    A2 = -(a * x + b * y) * iz;
  const B0 = b * iz,
    B1 = c * iz,
    B2 = -(b * x + c * y) * iz;

  // Pose: δω column block = J_proj (−[p]×): row q ↦ p × q ; δt block = J_proj.
  let o = RJ_POSE;
  out[o] = py * A2 - pz * A1;
  out[o + 1] = pz * A0 - px * A2;
  out[o + 2] = px * A1 - py * A0;
  out[o + 3] = A0;
  out[o + 4] = A1;
  out[o + 5] = A2;
  out[o + 6] = py * B2 - pz * B1;
  out[o + 7] = pz * B0 - px * B2;
  out[o + 8] = px * B1 - py * B0;
  out[o + 9] = B0;
  out[o + 10] = B1;
  out[o + 11] = B2;

  // Point: J_proj R
  o = RJ_POINT;
  out[o] = A0 * r0 + A1 * r3 + A2 * r6;
  out[o + 1] = A0 * r1 + A1 * r4 + A2 * r7;
  out[o + 2] = A0 * r2_ + A1 * r5 + A2 * r8;
  out[o + 3] = B0 * r0 + B1 * r3 + B2 * r6;
  out[o + 4] = B0 * r1 + B1 * r4 + B2 * r7;
  out[o + 5] = B0 * r2_ + B1 * r5 + B2 * r8;

  // Intrinsics [f, cx, cy, k1, k2]
  o = RJ_INTR;
  const frr = f * rr;
  out[o] = d * x;
  out[o + 1] = 1;
  out[o + 2] = 0;
  out[o + 3] = frr * x;
  out[o + 4] = frr * rr * x;
  out[o + 5] = d * y;
  out[o + 6] = 0;
  out[o + 7] = 1;
  out[o + 8] = frr * y;
  out[o + 9] = frr * rr * y;

  // Finite check on the quantities everything else is built from.
  const chk = ru + rv + A0 + A1 + A2 + B0 + B1 + B2 + px + py + pz + d;
  return chk - chk === 0;
}
