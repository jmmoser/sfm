/**
 * Similarity transforms dst ≈ s R src + t between 3-D point sets.
 */
import type { Mat3, Vec3 } from '../core/types';
import { svd3 } from './decomp';

/** Similarity transform x ↦ s R x + t. */
export interface Sim3 {
  s: number;
  R: Mat3;
  t: Vec3;
}

/**
 * Least-squares similarity (or rigid, when `withScale` is false) transform with
 * dst ≈ s R src + t, minimising Σ wᵢ ‖dstᵢ − (s R srcᵢ + t)‖² (Umeyama 1991,
 * "Least-squares estimation of transformation parameters between two point
 * patterns", IEEE TPAMI 13(4); weighted form). `src`, `dst` are flat 3n arrays.
 * Weights must be ≥ 0 (non-finite or negative weights are treated as 0).
 *
 * Returns null for degenerate input: fewer than 3 points with positive weight,
 * (numerically) coincident or collinear source or target points (the cross-
 * covariance has rank < 2, so the rotation about the line is undetermined), or
 * non-finite data.
 */
export function umeyama(
  src: Float64Array,
  dst: Float64Array,
  n: number,
  withScale: boolean,
  weights?: Float64Array,
): Sim3 | null {
  if (n < 3) return null;
  let wsum = 0;
  let count = 0;
  let msx = 0,
    msy = 0,
    msz = 0,
    mdx = 0,
    mdy = 0,
    mdz = 0;
  for (let i = 0; i < n; i++) {
    let w = weights ? weights[i] : 1;
    if (!(w > 0) || !Number.isFinite(w)) w = 0;
    if (w === 0) continue;
    count++;
    wsum += w;
    const o = 3 * i;
    msx += w * src[o];
    msy += w * src[o + 1];
    msz += w * src[o + 2];
    mdx += w * dst[o];
    mdy += w * dst[o + 1];
    mdz += w * dst[o + 2];
  }
  if (count < 3 || !(wsum > 0)) return null;
  const iw = 1 / wsum;
  msx *= iw;
  msy *= iw;
  msz *= iw;
  mdx *= iw;
  mdy *= iw;
  mdz *= iw;
  // Second pass on centred data: cross-covariance Σ = (1/W) Σ w (d − μd)(s − μs)ᵀ and source variance.
  const C = new Float64Array(9);
  let varS = 0,
    varD = 0;
  for (let i = 0; i < n; i++) {
    let w = weights ? weights[i] : 1;
    if (!(w > 0) || !Number.isFinite(w)) w = 0;
    if (w === 0) continue;
    const o = 3 * i;
    const sx = src[o] - msx,
      sy = src[o + 1] - msy,
      sz = src[o + 2] - msz;
    const dx = dst[o] - mdx,
      dy = dst[o + 1] - mdy,
      dz = dst[o + 2] - mdz;
    varS += w * (sx * sx + sy * sy + sz * sz);
    varD += w * (dx * dx + dy * dy + dz * dz);
    const wdx = w * dx,
      wdy = w * dy,
      wdz = w * dz;
    C[0] += wdx * sx;
    C[1] += wdx * sy;
    C[2] += wdx * sz;
    C[3] += wdy * sx;
    C[4] += wdy * sy;
    C[5] += wdy * sz;
    C[6] += wdz * sx;
    C[7] += wdz * sy;
    C[8] += wdz * sz;
  }
  varS *= iw;
  varD *= iw;
  for (let i = 0; i < 9; i++) C[i] *= iw;
  if (!(varS > 0) || !(varD > 0) || !Number.isFinite(varS) || !Number.isFinite(varD)) return null;
  const { U, S, V } = svd3(C);
  if (!Number.isFinite(S[0]) || !(S[0] > 0)) return null;
  // Rank ≥ 2 required (relative to the scale-free product of the spreads).
  if (!(S[1] > 1e-12 * Math.sqrt(varS * varD))) return null;
  const detU =
    U[0] * (U[4] * U[8] - U[5] * U[7]) - U[1] * (U[3] * U[8] - U[5] * U[6]) + U[2] * (U[3] * U[7] - U[4] * U[6]);
  const detV =
    V[0] * (V[4] * V[8] - V[5] * V[7]) - V[1] * (V[3] * V[8] - V[5] * V[6]) + V[2] * (V[3] * V[7] - V[4] * V[6]);
  const d = detU * detV < 0 ? -1 : 1;
  // R = U diag(1, 1, d) Vᵀ
  const R = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    const u0 = U[3 * i],
      u1 = U[3 * i + 1],
      u2 = U[3 * i + 2] * d;
    for (let j = 0; j < 3; j++) R[3 * i + j] = u0 * V[3 * j] + u1 * V[3 * j + 1] + u2 * V[3 * j + 2];
  }
  const s = withScale ? (S[0] + S[1] + d * S[2]) / varS : 1;
  if (!(s > 0) || !Number.isFinite(s)) return null;
  const t = new Float64Array(3);
  t[0] = mdx - s * (R[0] * msx + R[1] * msy + R[2] * msz);
  t[1] = mdy - s * (R[3] * msx + R[4] * msy + R[5] * msz);
  t[2] = mdz - s * (R[6] * msx + R[7] * msy + R[8] * msz);
  return { s, R, t };
}

/** Apply x ↦ s R x + t to each of the n = p.length/3 points (out may alias p). */
export function applySim3(sim: Sim3, p: Float64Array, out: Float64Array = new Float64Array(p.length)): Float64Array {
  const R = sim.R,
    s = sim.s,
    t = sim.t;
  const r0 = s * R[0],
    r1 = s * R[1],
    r2 = s * R[2],
    r3 = s * R[3],
    r4 = s * R[4],
    r5 = s * R[5],
    r6 = s * R[6],
    r7 = s * R[7],
    r8 = s * R[8];
  const t0 = t[0],
    t1 = t[1],
    t2 = t[2];
  const n3 = p.length - (p.length % 3);
  for (let o = 0; o < n3; o += 3) {
    const x = p[o],
      y = p[o + 1],
      z = p[o + 2];
    out[o] = r0 * x + r1 * y + r2 * z + t0;
    out[o + 1] = r3 * x + r4 * y + r5 * z + t1;
    out[o + 2] = r6 * x + r7 * y + r8 * z + t2;
  }
  return out;
}

/** Inverse similarity: x ↦ (1/s) Rᵀ (x − t). */
export function sim3Inverse(sim: Sim3): Sim3 {
  const R = sim.R,
    t = sim.t,
    is = 1 / sim.s;
  const Rt = new Float64Array(9);
  Rt[0] = R[0];
  Rt[1] = R[3];
  Rt[2] = R[6];
  Rt[3] = R[1];
  Rt[4] = R[4];
  Rt[5] = R[7];
  Rt[6] = R[2];
  Rt[7] = R[5];
  Rt[8] = R[8];
  const ti = new Float64Array(3);
  ti[0] = -is * (Rt[0] * t[0] + Rt[1] * t[1] + Rt[2] * t[2]);
  ti[1] = -is * (Rt[3] * t[0] + Rt[4] * t[1] + Rt[5] * t[2]);
  ti[2] = -is * (Rt[6] * t[0] + Rt[7] * t[1] + Rt[8] * t[2]);
  return { s: is, R: Rt, t: ti };
}

/** Composition a ∘ b : x ↦ a(b(x)) = (s_a s_b) (R_a R_b) x + (s_a R_a t_b + t_a). */
export function sim3Compose(a: Sim3, b: Sim3): Sim3 {
  const Ra = a.R,
    Rb = b.R;
  const R = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      R[3 * i + j] = Ra[3 * i] * Rb[j] + Ra[3 * i + 1] * Rb[3 + j] + Ra[3 * i + 2] * Rb[6 + j];
  const tb = b.t;
  const t = new Float64Array(3);
  for (let i = 0; i < 3; i++)
    t[i] = a.s * (Ra[3 * i] * tb[0] + Ra[3 * i + 1] * tb[1] + Ra[3 * i + 2] * tb[2]) + a.t[i];
  return { s: a.s * b.s, R, t };
}
