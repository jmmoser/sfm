/**
 * Absolute pose (PnP): Lambda Twist P3P, EPnP, robust LM refinement and LO-MSAC.
 * Poses are world-to-camera (x_cam = R X + t).
 */
import type { CameraIntrinsics, Pose } from '../core/types';
import { solveSPD, symEig } from '../math/decomp';
import { polyRealRoots } from '../math/poly';
import type { Rng } from '../math/random';
import { rotvecToMat3 } from '../math/rotation';
import { umeyama } from '../math/sim3';
import { pixelToNormalized } from './camera';
import { ransac, type RansacEstimator } from './ransac';

// ---------------------------------------------------------------------------
// Lambda Twist P3P (Persson & Nordberg, "Lambda Twist: An Accurate Fast Robust
// Perspective Three Point (P3P) Solver", ECCV 2018).
//
// Unknown depths λ (P_i = λ_i y_i) satisfy λᵀ M_ij λ = a_ij = |X_i − X_j|². The pencil
// D1 + γ D2 of D1 = a23 M12 − a12 M23, D2 = a23 M13 − a13 M23 contains degenerate
// (rank-2) members (cubic in γ); a degenerate indefinite D0 factors into two planes
// through the origin, each intersected with the cone λᵀ(a13 M12 − a12 M13)λ = 0 and
// scaled by λᵀ M23 λ = a23. Depths are refined by Gauss–Newton; R from the two
// triangles' edge frames.
// ---------------------------------------------------------------------------

function detCols(
  a0: number, a1: number, a2: number,
  b0: number, b1: number, b2: number,
  c0: number, c1: number, c2: number,
): number {
  return a0 * (b1 * c2 - b2 * c1) + a1 * (b2 * c0 - b0 * c2) + a2 * (b0 * c1 - b1 * c0);
}

/** Cubic coefficients [c3, c2, c1, c0] of det(A + γ B) for row-major 3×3 A, B. */
function detPencil(A: Float64Array, B: Float64Array, out: Float64Array): void {
  const a0 = A[0], a1 = A[3], a2 = A[6], a3 = A[1], a4 = A[4], a5 = A[7], a6 = A[2], a7 = A[5], a8 = A[8];
  const b0 = B[0], b1 = B[3], b2 = B[6], b3 = B[1], b4 = B[4], b5 = B[7], b6 = B[2], b7 = B[5], b8 = B[8];
  out[3] = detCols(a0, a1, a2, a3, a4, a5, a6, a7, a8);
  out[0] = detCols(b0, b1, b2, b3, b4, b5, b6, b7, b8);
  out[2] =
    detCols(b0, b1, b2, a3, a4, a5, a6, a7, a8) +
    detCols(a0, a1, a2, b3, b4, b5, a6, a7, a8) +
    detCols(a0, a1, a2, a3, a4, a5, b6, b7, b8);
  out[1] =
    detCols(a0, a1, a2, b3, b4, b5, b6, b7, b8) +
    detCols(b0, b1, b2, a3, a4, a5, b6, b7, b8) +
    detCols(b0, b1, b2, b3, b4, b5, a6, a7, a8);
}

/** Largest of the three row cross products of a symmetric 3×3 (null vector if rank 2), unnormalised. */
function nullVec3(M: Float64Array, out: Float64Array): number {
  let best = -1;
  for (let p = 0; p < 3; p++) {
    const i = p === 2 ? 1 : 0,
      j = p === 0 ? 1 : 2;
    const x = M[3 * i + 1] * M[3 * j + 2] - M[3 * i + 2] * M[3 * j + 1];
    const y = M[3 * i + 2] * M[3 * j] - M[3 * i] * M[3 * j + 2];
    const z = M[3 * i] * M[3 * j + 1] - M[3 * i + 1] * M[3 * j];
    const nn = x * x + y * y + z * z;
    if (nn > best) {
      best = nn;
      out[0] = x;
      out[1] = y;
      out[2] = z;
    }
  }
  return best;
}

const P3_D1 = new Float64Array(9);
const P3_D2 = new Float64Array(9);
const P3_D0 = new Float64Array(9);
const P3_K = new Float64Array(9);
const P3_TMP = new Float64Array(9);
const P3_C = new Float64Array(4);
const P3_E1 = new Float64Array(3);
const P3_E2 = new Float64Array(3);
const P3_E3 = new Float64Array(3);
const P3_W = new Float64Array(3);
const P3_L = new Float64Array(12);

/**
 * Eigen-decomposition of a symmetric rank-2 3×3: e3 = null vector, σ1, σ2 the other
 * eigenvalues with |σ1| ≥ |σ2| and unit eigenvectors e1, e2. Returns [σ1, σ2] or null.
 */
function eigRank2(D: Float64Array): [number, number] | null {
  if (!(nullVec3(D, P3_E3) > 0)) return null;
  const n3 = Math.hypot(P3_E3[0], P3_E3[1], P3_E3[2]);
  for (let k = 0; k < 3; k++) P3_E3[k] /= n3;
  const tr = D[0] + D[4] + D[8];
  const c = D[0] * D[4] - D[1] * D[1] + D[0] * D[8] - D[2] * D[2] + D[4] * D[8] - D[5] * D[5];
  // σ² − tr σ + c = 0
  const disc = tr * tr - 4 * c;
  const sq = Math.sqrt(Math.max(0, disc));
  const q = 0.5 * (tr + (tr >= 0 ? sq : -sq));
  let s1 = q,
    s2 = q !== 0 ? c / q : 0;
  if (Math.abs(s2) > Math.abs(s1)) {
    const t = s1;
    s1 = s2;
    s2 = t;
  }
  // e1: null vector of D − σ1 I
  P3_TMP.set(D);
  P3_TMP[0] -= s1;
  P3_TMP[4] -= s1;
  P3_TMP[8] -= s1;
  if (!(nullVec3(P3_TMP, P3_E1) > 0)) return null;
  const n1 = Math.hypot(P3_E1[0], P3_E1[1], P3_E1[2]);
  for (let k = 0; k < 3; k++) P3_E1[k] /= n1;
  // e2 = e3 × e1 (orthonormal completion)
  P3_E2[0] = P3_E3[1] * P3_E1[2] - P3_E3[2] * P3_E1[1];
  P3_E2[1] = P3_E3[2] * P3_E1[0] - P3_E3[0] * P3_E1[2];
  P3_E2[2] = P3_E3[0] * P3_E1[1] - P3_E3[1] * P3_E1[0];
  return [s1, s2];
}

function quadForm(M: Float64Array, a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < 3; i++) s += a[i] * (M[3 * i] * b[0] + M[3 * i + 1] * b[1] + M[3 * i + 2] * b[2]);
  return s;
}

/** Gauss–Newton on the three distance equations (Lambda Twist's refineL). */
function refineLambda(L: Float64Array, o: number, a12: number, a13: number, a23: number, b12: number, b13: number, b23: number): void {
  let l1 = L[o],
    l2 = L[o + 1],
    l3 = L[o + 2];
  let r1 = l1 * l1 + l2 * l2 + b12 * l1 * l2 - a12;
  let r2 = l1 * l1 + l3 * l3 + b13 * l1 * l3 - a13;
  let r3 = l2 * l2 + l3 * l3 + b23 * l2 * l3 - a23;
  let rn = Math.abs(r1) + Math.abs(r2) + Math.abs(r3);
  for (let it = 0; it < 8 && rn > 0; it++) {
    const j11 = 2 * l1 + b12 * l2,
      j12 = 2 * l2 + b12 * l1;
    const j21 = 2 * l1 + b13 * l3,
      j23 = 2 * l3 + b13 * l1;
    const j32 = 2 * l2 + b23 * l3,
      j33 = 2 * l3 + b23 * l2;
    // J = [[j11, j12, 0], [j21, 0, j23], [0, j32, j33]]
    const det = j11 * (-j23 * j32) - j12 * (j21 * j33);
    if (!(Math.abs(det) > 0)) break;
    const id = 1 / det;
    // inverse by cofactors
    const i00 = -j23 * j32,
      i01 = -j12 * j33,
      i02 = j12 * j23;
    const i10 = -j21 * j33,
      i11 = j11 * j33,
      i12 = -j11 * j23;
    const i20 = j21 * j32,
      i21 = -j11 * j32,
      i22 = -j12 * j21;
    const n1 = l1 - id * (i00 * r1 + i01 * r2 + i02 * r3);
    const n2 = l2 - id * (i10 * r1 + i11 * r2 + i12 * r3);
    const n3 = l3 - id * (i20 * r1 + i21 * r2 + i22 * r3);
    const s1 = n1 * n1 + n2 * n2 + b12 * n1 * n2 - a12;
    const s2 = n1 * n1 + n3 * n3 + b13 * n1 * n3 - a13;
    const s3 = n2 * n2 + n3 * n3 + b23 * n2 * n3 - a23;
    const sn = Math.abs(s1) + Math.abs(s2) + Math.abs(s3);
    if (!(sn < rn)) break;
    l1 = n1;
    l2 = n2;
    l3 = n3;
    r1 = s1;
    r2 = s2;
    r3 = s3;
    rn = sn;
  }
  L[o] = l1;
  L[o + 1] = l2;
  L[o + 2] = l3;
}

/**
 * Lambda Twist P3P. X: 3 world points (9), bearings: 3 unit (or any length) vectors
 * in the camera frame (9). Returns up to 4 poses with all three points in front.
 */
export function p3p(X: ArrayLike<number>, bearings: ArrayLike<number>): Pose[] {
  const out: Pose[] = [];
  const y = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    const n = Math.hypot(bearings[3 * i], bearings[3 * i + 1], bearings[3 * i + 2]);
    if (!(n > 0)) return out;
    y[3 * i] = bearings[3 * i] / n;
    y[3 * i + 1] = bearings[3 * i + 1] / n;
    y[3 * i + 2] = bearings[3 * i + 2] / n;
  }
  const b12 = -2 * (y[0] * y[3] + y[1] * y[4] + y[2] * y[5]);
  const b13 = -2 * (y[0] * y[6] + y[1] * y[7] + y[2] * y[8]);
  const b23 = -2 * (y[3] * y[6] + y[4] * y[7] + y[5] * y[8]);
  const d12x = X[0] - X[3],
    d12y = X[1] - X[4],
    d12z = X[2] - X[5];
  const d13x = X[0] - X[6],
    d13y = X[1] - X[7],
    d13z = X[2] - X[8];
  const d23x = X[3] - X[6],
    d23y = X[4] - X[7],
    d23z = X[5] - X[8];
  const a12 = d12x * d12x + d12y * d12y + d12z * d12z;
  const a13 = d13x * d13x + d13y * d13y + d13z * d13z;
  const a23 = d23x * d23x + d23y * d23y + d23z * d23z;
  const cx = d12y * d13z - d12z * d13y,
    cy = d12z * d13x - d12x * d13z,
    cz = d12x * d13y - d12y * d13x;
  const area2 = cx * cx + cy * cy + cz * cz;
  if (!(area2 > 1e-20 * a12 * a13) || !(a23 > 0)) return out; // collinear / coincident
  // M12 = [[1, b12/2, 0], [b12/2, 1, 0], [0, 0, 0]] etc.
  const h12 = 0.5 * b12,
    h13 = 0.5 * b13,
    h23 = 0.5 * b23;
  // D1 = a23 M12 − a12 M23 ; D2 = a23 M13 − a13 M23
  P3_D1[0] = a23;
  P3_D1[1] = P3_D1[3] = a23 * h12;
  P3_D1[2] = P3_D1[6] = 0;
  P3_D1[4] = a23 - a12;
  P3_D1[5] = P3_D1[7] = -a12 * h23;
  P3_D1[8] = -a12;
  P3_D2[0] = a23;
  P3_D2[1] = P3_D2[3] = 0;
  P3_D2[2] = P3_D2[6] = a23 * h13;
  P3_D2[4] = -a13;
  P3_D2[5] = P3_D2[7] = -a13 * h23;
  P3_D2[8] = a23 - a13;
  // K = a13 M12 − a12 M13 (cone through the solutions)
  P3_K[0] = a13 - a12;
  P3_K[1] = P3_K[3] = a13 * h12;
  P3_K[2] = P3_K[6] = -a12 * h13;
  P3_K[4] = a13;
  P3_K[5] = P3_K[7] = 0;
  P3_K[8] = -a12;

  detPencil(P3_D1, P3_D2, P3_C);
  const roots = polyRealRoots(P3_C);
  // Choose the root whose degenerate conic is the best-conditioned real line pair.
  let bestG = NaN,
    bestQ = -1;
  for (const g of roots) {
    for (let k = 0; k < 9; k++) P3_D0[k] = P3_D1[k] + g * P3_D2[k];
    const ev = eigRank2(P3_D0);
    if (!ev) continue;
    const [s1, s2] = ev;
    if (!(s1 * s2 < 0)) continue;
    const qual = Math.abs(s2) / Math.abs(s1);
    if (qual > bestQ) {
      bestQ = qual;
      bestG = g;
    }
  }
  if (!(bestQ >= 0)) return out;
  for (let k = 0; k < 9; k++) P3_D0[k] = P3_D1[k] + bestG * P3_D2[k];
  const ev = eigRank2(P3_D0)!;
  const v = Math.sqrt(-ev[1] / ev[0]);
  let numL = 0;
  for (let sgn = -1; sgn <= 1; sgn += 2) {
    const s = sgn * v;
    // Plane (e1 − s e2)ᵀ λ = 0 spanned by e3 and w = s e1 + e2.
    for (let k = 0; k < 3; k++) P3_W[k] = s * P3_E1[k] + P3_E2[k];
    const k11 = quadForm(P3_K, P3_E3, P3_E3),
      k12 = quadForm(P3_K, P3_E3, P3_W),
      k22 = quadForm(P3_K, P3_W, P3_W);
    // k11 α² + 2 k12 α β + k22 β² = 0
    const swap = Math.abs(k11) < Math.abs(k22);
    const qa = swap ? k22 : k11,
      qc = swap ? k11 : k22;
    if (!(qa !== 0)) continue;
    const disc = k12 * k12 - qa * qc;
    if (disc < 0) continue;
    const sq = Math.sqrt(disc);
    const qq = -(k12 + (k12 >= 0 ? sq : -sq));
    const rs = [qq / qa, qq !== 0 ? qc / qq : 0];
    const nr = disc === 0 ? 1 : 2;
    for (let ri = 0; ri < nr; ri++) {
      const r = rs[ri];
      const lam0 = swap ? P3_E3[0] + r * P3_W[0] : r * P3_E3[0] + P3_W[0];
      const lam1 = swap ? P3_E3[1] + r * P3_W[1] : r * P3_E3[1] + P3_W[1];
      const lam2 = swap ? P3_E3[2] + r * P3_W[2] : r * P3_E3[2] + P3_W[2];
      // scale with λᵀ M23 λ = a23
      const c23 = lam1 * lam1 + lam2 * lam2 + b23 * lam1 * lam2;
      if (!(c23 > 0)) continue;
      let sc = Math.sqrt(a23 / c23);
      if (lam0 + lam1 + lam2 < 0) sc = -sc;
      const L0 = lam0 * sc,
        L1 = lam1 * sc,
        L2 = lam2 * sc;
      if (!(L0 > 0 && L1 > 0 && L2 > 0)) continue;
      P3_L[3 * numL] = L0;
      P3_L[3 * numL + 1] = L1;
      P3_L[3 * numL + 2] = L2;
      numL++;
    }
  }
  if (numL === 0) return out;
  // X⁻¹ for X = [d12, d13, d12 × d13] (columns)
  const Xm = Float64Array.of(d12x, d13x, cx, d12y, d13y, cy, d12z, d13z, cz);
  const Xi = inv3(Xm);
  if (!Xi) return out;
  const Ym = new Float64Array(9);
  for (let s = 0; s < numL; s++) {
    refineLambda(P3_L, 3 * s, a12, a13, a23, b12, b13, b23);
    const l1 = P3_L[3 * s],
      l2 = P3_L[3 * s + 1],
      l3 = P3_L[3 * s + 2];
    if (!(l1 > 0 && l2 > 0 && l3 > 0)) continue;
    // duplicates
    let dup = false;
    for (let q = 0; q < s && !dup; q++) {
      const dl =
        Math.abs(P3_L[3 * q] - l1) + Math.abs(P3_L[3 * q + 1] - l2) + Math.abs(P3_L[3 * q + 2] - l3);
      dup = dl <= 1e-9 * (l1 + l2 + l3);
    }
    if (dup) continue;
    const p1x = l1 * y[0],
      p1y = l1 * y[1],
      p1z = l1 * y[2];
    const e1x = p1x - l2 * y[3],
      e1y = p1y - l2 * y[4],
      e1z = p1z - l2 * y[5];
    const e2x = p1x - l3 * y[6],
      e2y = p1y - l3 * y[7],
      e2z = p1z - l3 * y[8];
    const fx = e1y * e2z - e1z * e2y,
      fy = e1z * e2x - e1x * e2z,
      fz = e1x * e2y - e1y * e2x;
    Ym[0] = e1x;
    Ym[1] = e2x;
    Ym[2] = fx;
    Ym[3] = e1y;
    Ym[4] = e2y;
    Ym[5] = fy;
    Ym[6] = e1z;
    Ym[7] = e2z;
    Ym[8] = fz;
    const R = new Float64Array(9);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) R[3 * i + j] = Ym[3 * i] * Xi[j] + Ym[3 * i + 1] * Xi[3 + j] + Ym[3 * i + 2] * Xi[6 + j];
    const t = new Float64Array(3);
    t[0] = p1x - (R[0] * X[0] + R[1] * X[1] + R[2] * X[2]);
    t[1] = p1y - (R[3] * X[0] + R[4] * X[1] + R[5] * X[2]);
    t[2] = p1z - (R[6] * X[0] + R[7] * X[1] + R[8] * X[2]);
    if (!Number.isFinite(R[0] + R[4] + R[8] + t[0] + t[1] + t[2])) continue;
    out.push({ R, t });
  }
  return out;
}

function inv3(m: Float64Array): Float64Array | null {
  const c00 = m[4] * m[8] - m[5] * m[7],
    c01 = m[5] * m[6] - m[3] * m[8],
    c02 = m[3] * m[7] - m[4] * m[6];
  const det = m[0] * c00 + m[1] * c01 + m[2] * c02;
  if (!(Math.abs(det) > 0) || !Number.isFinite(det)) return null;
  const id = 1 / det;
  const o = new Float64Array(9);
  o[0] = c00 * id;
  o[1] = (m[2] * m[7] - m[1] * m[8]) * id;
  o[2] = (m[1] * m[5] - m[2] * m[4]) * id;
  o[3] = c01 * id;
  o[4] = (m[0] * m[8] - m[2] * m[6]) * id;
  o[5] = (m[2] * m[3] - m[0] * m[5]) * id;
  o[6] = c02 * id;
  o[7] = (m[1] * m[6] - m[0] * m[7]) * id;
  o[8] = (m[0] * m[4] - m[1] * m[3]) * id;
  return o;
}

// ---------------------------------------------------------------------------
// EPnP (Lepetit, Moreno-Noguer & Fua, "EPnP: An Accurate O(n) Solution to the PnP
// Problem", IJCV 81(2), 2009) with Gauss–Newton refinement of the null-space
// coefficients β; 3 control points for (near-)planar point sets.
// ---------------------------------------------------------------------------

/** Squared normalised reprojection error sum of a pose (Infinity if a point is behind). */
function reprojCost(R: ArrayLike<number>, t: ArrayLike<number>, X: ArrayLike<number>, xn: ArrayLike<number>, n: number): number {
  let c = 0;
  for (let i = 0; i < n; i++) {
    const x = X[3 * i],
      y = X[3 * i + 1],
      z = X[3 * i + 2];
    const pz = R[6] * x + R[7] * y + R[8] * z + t[2];
    if (!(pz > 0)) return Infinity;
    const du = (R[0] * x + R[1] * y + R[2] * z + t[0]) / pz - xn[2 * i];
    const dv = (R[3] * x + R[4] * y + R[5] * z + t[1]) / pz - xn[2 * i + 1];
    c += du * du + dv * dv;
  }
  return c;
}

/** EPnP with nc control points (3 or 4). Returns the best pose over the β approximations. */
function epnpWith(
  X: ArrayLike<number>,
  xn: ArrayLike<number>,
  n: number,
  nc: 3 | 4,
  c0: Float64Array,
  axes: Float64Array /* row k = axis k (unit) */,
  scales: Float64Array,
): { pose: Pose; cost: number } | null {
  const dim = 3 * nc;
  // Barycentric coordinates α (n × nc)
  const alpha = new Float64Array(n * nc);
  for (let i = 0; i < n; i++) {
    const dx = X[3 * i] - c0[0],
      dy = X[3 * i + 1] - c0[1],
      dz = X[3 * i + 2] - c0[2];
    let s = 0;
    for (let j = 1; j < nc; j++) {
      const ax = axes[3 * (j - 1)],
        ay = axes[3 * (j - 1) + 1],
        az = axes[3 * (j - 1) + 2];
      const a = (dx * ax + dy * ay + dz * az) / scales[j - 1];
      alpha[i * nc + j] = a;
      s += a;
    }
    alpha[i * nc] = 1 - s;
  }
  // MᵀM
  const MtM = new Float64Array(dim * dim);
  const r1 = new Float64Array(dim),
    r2 = new Float64Array(dim);
  for (let i = 0; i < n; i++) {
    const u = xn[2 * i],
      v = xn[2 * i + 1];
    for (let j = 0; j < nc; j++) {
      const a = alpha[i * nc + j];
      r1[3 * j] = a;
      r1[3 * j + 1] = 0;
      r1[3 * j + 2] = -a * u;
      r2[3 * j] = 0;
      r2[3 * j + 1] = a;
      r2[3 * j + 2] = -a * v;
    }
    for (let p = 0; p < dim; p++) {
      const a1 = r1[p],
        a2 = r2[p];
      if (a1 === 0 && a2 === 0) continue;
      const o = p * dim;
      for (let q = p; q < dim; q++) MtM[o + q] += a1 * r1[q] + a2 * r2[q];
    }
  }
  for (let p = 0; p < dim; p++) for (let q = 0; q < p; q++) MtM[p * dim + q] = MtM[q * dim + p];
  const { vectors } = symEig(MtM, dim);
  const K = nc; // number of null-space vectors used
  // vk[k][3j + c] = vectors[(3j + c) * dim + k]
  const vk = new Float64Array(K * dim);
  for (let k = 0; k < K; k++) for (let e = 0; e < dim; e++) vk[k * dim + e] = vectors[e * dim + k];
  // World control points
  const cw = new Float64Array(3 * nc);
  cw[0] = c0[0];
  cw[1] = c0[1];
  cw[2] = c0[2];
  for (let j = 1; j < nc; j++)
    for (let c = 0; c < 3; c++) cw[3 * j + c] = c0[c] + scales[j - 1] * axes[3 * (j - 1) + c];
  // Pairs, L (pairs × monomials), rho
  const pairs: [number, number][] = [];
  for (let i = 0; i < nc; i++) for (let j = i + 1; j < nc; j++) pairs.push([i, j]);
  const np = pairs.length;
  const monos: [number, number][] = [];
  for (let a = 0; a < K; a++) for (let b = a; b < K; b++) monos.push([a, b]);
  const nm = monos.length;
  const monoIdx = (a: number, b: number): number => {
    if (a > b) {
      const t = a;
      a = b;
      b = t;
    }
    return monos.findIndex((m) => m[0] === a && m[1] === b);
  };
  const L = new Float64Array(np * nm);
  const rho = new Float64Array(np);
  const dv = new Float64Array(3 * K);
  for (let p = 0; p < np; p++) {
    const [i, j] = pairs[p];
    for (let k = 0; k < K; k++)
      for (let c = 0; c < 3; c++) dv[3 * k + c] = vk[k * dim + 3 * i + c] - vk[k * dim + 3 * j + c];
    for (let m = 0; m < nm; m++) {
      const [a, b] = monos[m];
      const d = dv[3 * a] * dv[3 * b] + dv[3 * a + 1] * dv[3 * b + 1] + dv[3 * a + 2] * dv[3 * b + 2];
      L[p * nm + m] = a === b ? d : 2 * d;
    }
    const wx = cw[3 * i] - cw[3 * j],
      wy = cw[3 * i + 1] - cw[3 * j + 1],
      wz = cw[3 * i + 2] - cw[3 * j + 2];
    rho[p] = wx * wx + wy * wy + wz * wz;
  }
  // Least-squares solve of L[:, cols] b = rho
  const lsq = (cols: number[]): Float64Array | null => {
    const m = cols.length;
    const A = new Float64Array(m * m),
      g = new Float64Array(m);
    for (let p = 0; p < np; p++) {
      for (let a = 0; a < m; a++) {
        const la = L[p * nm + cols[a]];
        g[a] += la * rho[p];
        for (let b = 0; b < m; b++) A[a * m + b] += la * L[p * nm + cols[b]];
      }
    }
    // tiny ridge for robustness
    let tr = 0;
    for (let a = 0; a < m; a++) tr += A[a * m + a];
    for (let a = 0; a < m; a++) A[a * m + a] += 1e-14 * tr;
    return solveSPD(A, m, g);
  };
  // Each candidate: initial β and the number Kg of leading β refined by Gauss–Newton.
  const candidates: { b: Float64Array; kg: number }[] = [];
  const addCandidate = (b: Float64Array, nat: number): void => {
    candidates.push({ b: Float64Array.from(b), kg: Math.min(nat, K) });
    if (nat < K) candidates.push({ b: Float64Array.from(b), kg: K });
  };
  // N = 1 (closed form of the paper)
  {
    let num = 0,
      den = 0;
    for (let p = 0; p < np; p++) {
      const [i, j] = pairs[p];
      let s = 0;
      for (let c = 0; c < 3; c++) {
        const d = vk[3 * i + c] - vk[3 * j + c];
        s += d * d;
      }
      num += Math.sqrt(s) * Math.sqrt(rho[p]);
      den += s;
    }
    if (den > 0) {
      const b = new Float64Array(K);
      b[0] = num / den;
      addCandidate(b, 1);
    }
  }
  // approx 1: [β00, β01, β02, (β03)]
  {
    const cols = [];
    for (let k = 0; k < K; k++) cols.push(monoIdx(0, k));
    if (np >= cols.length) {
      const s = lsq(cols);
      if (s && s[0] !== 0) {
        const b = new Float64Array(K);
        const b0 = Math.sqrt(Math.abs(s[0]));
        const sg = s[0] < 0 ? -1 : 1;
        b[0] = b0;
        for (let k = 1; k < K; k++) b[k] = (sg * s[k]) / b0;
        addCandidate(b, K);
      }
    }
  }
  // approx 2: [β00, β01, β11]
  {
    const s = lsq([monoIdx(0, 0), monoIdx(0, 1), monoIdx(1, 1)]);
    if (s) {
      const b = new Float64Array(K);
      if (s[0] < 0) {
        b[0] = Math.sqrt(-s[0]);
        b[1] = s[2] < 0 ? Math.sqrt(-s[2]) : 0;
      } else {
        b[0] = Math.sqrt(s[0]);
        b[1] = s[2] > 0 ? Math.sqrt(s[2]) : 0;
      }
      if (s[1] < 0) b[0] = -b[0];
      addCandidate(b, 2);
    }
  }
  // approx 3 (nc = 4): [β00, β01, β11, β02, β12]
  if (K >= 3 && np >= 5) {
    const s = lsq([monoIdx(0, 0), monoIdx(0, 1), monoIdx(1, 1), monoIdx(0, 2), monoIdx(1, 2)]);
    if (s) {
      const b = new Float64Array(K);
      if (s[0] < 0) {
        b[0] = Math.sqrt(-s[0]);
        b[1] = s[2] < 0 ? Math.sqrt(-s[2]) : 0;
      } else {
        b[0] = Math.sqrt(s[0]);
        b[1] = s[2] > 0 ? Math.sqrt(s[2]) : 0;
      }
      if (s[1] < 0) b[0] = -b[0];
      b[2] = b[0] !== 0 ? s[3] / b[0] : 0;
      addCandidate(b, 3);
    }
  }
  // Gauss–Newton on the leading kg betas: r_p = Σ_m L[p][m] β_a β_b − ρ_p
  const H = new Float64Array(K * K),
    g = new Float64Array(K),
    J = new Float64Array(K),
    prevB = new Float64Array(K);
  const distCost = (b: Float64Array): number => {
    let cost = 0;
    for (let p = 0; p < np; p++) {
      let r = -rho[p];
      for (let m = 0; m < nm; m++) r += L[p * nm + m] * b[monos[m][0]] * b[monos[m][1]];
      cost += r * r;
    }
    return cost;
  };
  const gaussNewton = (b: Float64Array, kg: number): void => {
    let cost = distCost(b);
    for (let it = 0; it < 10 && cost > 0; it++) {
      H.fill(0);
      g.fill(0);
      for (let p = 0; p < np; p++) {
        let r = -rho[p];
        J.fill(0);
        for (let m = 0; m < nm; m++) {
          const a = monos[m][0],
            c = monos[m][1];
          const l = L[p * nm + m];
          r += l * b[a] * b[c];
          J[a] += l * b[c];
          J[c] += l * b[a];
        }
        for (let a = 0; a < kg; a++) {
          g[a] -= J[a] * r;
          for (let c = 0; c < kg; c++) H[a * kg + c] += J[a] * J[c];
        }
      }
      let tr = 0;
      for (let a = 0; a < kg; a++) tr += H[a * kg + a];
      for (let a = 0; a < kg; a++) H[a * kg + a] += 1e-12 * tr + 1e-300;
      const d = solveSPD(H.subarray(0, kg * kg), kg, g.subarray(0, kg));
      if (!d) break;
      prevB.set(b);
      for (let a = 0; a < kg; a++) b[a] += d[a];
      const c2 = distCost(b);
      if (!(c2 < cost)) {
        b.set(prevB);
        break;
      }
      const rel = (cost - c2) / cost;
      cost = c2;
      if (rel < 1e-14) break;
    }
  };
  let best: { pose: Pose; cost: number } | null = null;
  const cc = new Float64Array(3 * nc);
  const Pc = new Float64Array(3 * n);
  for (const cand of candidates) {
    const b = cand.b;
    gaussNewton(b, cand.kg);
    if (!Number.isFinite(b[0])) continue;
    for (let e = 0; e < dim; e++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += b[k] * vk[k * dim + e];
      cc[e] = s;
    }
    let zsum = 0;
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let j = 0; j < nc; j++) s += alpha[i * nc + j] * cc[3 * j + c];
        Pc[3 * i + c] = s;
      }
      zsum += Pc[3 * i + 2];
    }
    if (zsum < 0) for (let e = 0; e < 3 * n; e++) Pc[e] = -Pc[e];
    const sim = umeyama(X instanceof Float64Array ? X : Float64Array.from(X), Pc, n, false);
    if (!sim) continue;
    const cost = reprojCost(sim.R, sim.t, X, xn, n);
    if (!best || cost < best.cost) best = { pose: { R: sim.R, t: sim.t }, cost };
  }
  return best;
}

/**
 * EPnP from n ≥ 4 world points X (3n) and undistorted normalised observations xn
 * (2n). Planar and near-planar point sets are handled with 3 control points (both
 * variants are tried for near-planar sets and the lower reprojection error wins).
 * Null if degenerate.
 */
export function epnp(X: ArrayLike<number>, xn: ArrayLike<number>, n: number): Pose | null {
  if (n < 4) return null;
  const c0 = new Float64Array(3);
  for (let i = 0; i < n; i++) {
    c0[0] += X[3 * i];
    c0[1] += X[3 * i + 1];
    c0[2] += X[3 * i + 2];
  }
  c0[0] /= n;
  c0[1] /= n;
  c0[2] /= n;
  const C = new Float64Array(9);
  for (let i = 0; i < n; i++) {
    const dx = X[3 * i] - c0[0],
      dy = X[3 * i + 1] - c0[1],
      dz = X[3 * i + 2] - c0[2];
    C[0] += dx * dx;
    C[1] += dx * dy;
    C[2] += dx * dz;
    C[4] += dy * dy;
    C[5] += dy * dz;
    C[8] += dz * dz;
  }
  C[3] = C[1];
  C[6] = C[2];
  C[7] = C[5];
  const { values, vectors } = symEig(C, 3);
  if (!(values[2] > 0) || !Number.isFinite(values[2])) return null;
  if (!(values[1] > 1e-12 * values[2])) return null; // collinear
  // axes ordered by decreasing variance; scales = RMS spread
  const axes = new Float64Array(9);
  const scales = new Float64Array(3);
  for (let k = 0; k < 3; k++) {
    const src = 2 - k;
    for (let c = 0; c < 3; c++) axes[3 * k + c] = vectors[c * 3 + src];
    scales[k] = Math.sqrt(Math.max(values[src], 0) / n);
  }
  const flatness = Math.sqrt(Math.max(values[0], 0) / values[2]);
  let best: { pose: Pose; cost: number } | null = null;
  if (flatness > 1e-4 && n >= 5) best = epnpWith(X, xn, n, 4, c0, axes, scales);
  if (flatness < 0.05) {
    const r3 = epnpWith(X, xn, n, 3, c0, axes, scales);
    if (r3 && (!best || r3.cost < best.cost)) best = r3;
  }
  if (!best && n === 4 && flatness > 1e-4) best = epnpWith(X, xn, n, 4, c0, axes, scales);
  if (n <= 6) {
    // Few points (EPnP's kernel is then up to 4-D and its linearisations may fail):
    // add the P3P solutions of the first three points, scored on all points.
    const X3 = new Float64Array(9),
      B3 = new Float64Array(9);
    for (let i = 0; i < 3; i++) {
      const u = xn[2 * i],
        v = xn[2 * i + 1];
      const inv = 1 / Math.sqrt(u * u + v * v + 1);
      B3[3 * i] = u * inv;
      B3[3 * i + 1] = v * inv;
      B3[3 * i + 2] = inv;
      X3[3 * i] = X[3 * i];
      X3[3 * i + 1] = X[3 * i + 1];
      X3[3 * i + 2] = X[3 * i + 2];
    }
    for (const pose of p3p(X3, B3)) {
      const cost = reprojCost(pose.R, pose.t, X, xn, n);
      if (!best || cost < best.cost) best = { pose, cost };
    }
  }
  return best ? best.pose : null;
}

// ---------------------------------------------------------------------------
// Robust LM refinement
// ---------------------------------------------------------------------------

/** Options of `refinePose`. */
export interface RefinePoseOptions {
  /** Maximum LM iterations (default 10). */
  iterations?: number;
  /** Huber threshold on the normalised reprojection error (default 2e-3 ≈ 3 px at f = 1500 px). */
  huberNormalized?: number;
}

function poseCost(
  R: ArrayLike<number>,
  t: ArrayLike<number>,
  X: ArrayLike<number>,
  xn: ArrayLike<number>,
  n: number,
  k: number,
): number {
  let c = 0;
  const k2 = k * k;
  for (let i = 0; i < n; i++) {
    const x = X[3 * i],
      y = X[3 * i + 1],
      z = X[3 * i + 2];
    const pz = R[6] * x + R[7] * y + R[8] * z + t[2];
    if (!(pz > 0)) {
      c += 2 * k - k2; // behind the camera: Huber cost of a residual of 1 (≈ 45°)
      continue;
    }
    const du = (R[0] * x + R[1] * y + R[2] * z + t[0]) / pz - xn[2 * i];
    const dv = (R[3] * x + R[4] * y + R[5] * z + t[1]) / pz - xn[2 * i + 1];
    const r2 = du * du + dv * dv;
    c += r2 <= k2 ? r2 : 2 * k * Math.sqrt(r2) - k2;
  }
  return c;
}

const RF_H = new Float64Array(36);
const RF_A = new Float64Array(36);
const RF_G = new Float64Array(6);
const RF_RHS = new Float64Array(6);
const RF_W = new Float64Array(3);
const RF_DR = new Float64Array(9);
const RF_JU = new Float64Array(6);
const RF_JV = new Float64Array(6);

/**
 * Levenberg–Marquardt refinement of a world-to-camera pose on the Huber-robustified
 * reprojection error in NORMALISED coordinates (IRLS weights). Rotation updates are
 * left-multiplicative (R ← exp([δω]×) R), t ← t + δt. Returns a new pose.
 */
export function refinePose(
  pose: Pose,
  X: ArrayLike<number>,
  xn: ArrayLike<number>,
  n: number,
  opts: RefinePoseOptions = {},
): Pose {
  const iterations = opts.iterations ?? 10;
  const k = opts.huberNormalized ?? 2e-3;
  let R = Float64Array.from(pose.R);
  let t = Float64Array.from(pose.t);
  if (n < 3) return { R, t };
  let cost = poseCost(R, t, X, xn, n, k);
  let lambda = 1e-4;
  for (let it = 0; it < iterations; it++) {
    RF_H.fill(0);
    RF_G.fill(0);
    for (let i = 0; i < n; i++) {
      const x = X[3 * i],
        y = X[3 * i + 1],
        z = X[3 * i + 2];
      const q0 = R[0] * x + R[1] * y + R[2] * z,
        q1 = R[3] * x + R[4] * y + R[5] * z,
        q2 = R[6] * x + R[7] * y + R[8] * z;
      const p0 = q0 + t[0],
        p1 = q1 + t[1],
        p2 = q2 + t[2];
      if (!(p2 > 0)) continue;
      const iz = 1 / p2;
      const u = p0 * iz,
        v = p1 * iz;
      const ru = u - xn[2 * i],
        rv = v - xn[2 * i + 1];
      const rn = Math.sqrt(ru * ru + rv * rv);
      const w = rn <= k ? 1 : k / rn;
      // ∂(u,v)/∂P = [[iz, 0, −u iz], [0, iz, −v iz]]; ∂P/∂ω = −[q]×, ∂P/∂t = I
      // −[q]× = [[0, q2, −q1], [−q2, 0, q0], [q1, −q0, 0]]
      const ju0 = iz * 0 - u * iz * q1, // column ω0: (0, −q2, q1)
        ju1 = iz * q2 - u * iz * -q0, // ω1: (q2, 0, −q0)
        ju2 = iz * -q1 - u * iz * 0, // ω2: (−q1, q0, 0)
        ju3 = iz,
        ju4 = 0,
        ju5 = -u * iz;
      const jv0 = iz * -q2 - v * iz * q1,
        jv1 = 0 - v * iz * -q0,
        jv2 = iz * q0 - v * iz * 0,
        jv3 = 0,
        jv4 = iz,
        jv5 = -v * iz;
      const Ju = RF_JU,
        Jv = RF_JV;
      Ju[0] = ju0;
      Ju[1] = ju1;
      Ju[2] = ju2;
      Ju[3] = ju3;
      Ju[4] = ju4;
      Ju[5] = ju5;
      Jv[0] = jv0;
      Jv[1] = jv1;
      Jv[2] = jv2;
      Jv[3] = jv3;
      Jv[4] = jv4;
      Jv[5] = jv5;
      for (let a = 0; a < 6; a++) {
        const wa = w * Ju[a],
          wb = w * Jv[a];
        RF_G[a] += wa * ru + wb * rv;
        for (let b = a; b < 6; b++) RF_H[6 * a + b] += wa * Ju[b] + wb * Jv[b];
      }
    }
    for (let a = 0; a < 6; a++) for (let b = 0; b < a; b++) RF_H[6 * a + b] = RF_H[6 * b + a];
    let improved = false;
    let converged = false;
    for (let tries = 0; tries < 10; tries++) {
      RF_A.set(RF_H);
      for (let a = 0; a < 6; a++) RF_A[7 * a] += lambda * Math.max(RF_H[7 * a], 1e-12);
      for (let a = 0; a < 6; a++) RF_RHS[a] = -RF_G[a];
      const d = solveSPD(RF_A, 6, RF_RHS);
      if (!d) {
        lambda *= 10;
        continue;
      }
      RF_W[0] = d[0];
      RF_W[1] = d[1];
      RF_W[2] = d[2];
      rotvecToMat3(RF_W, RF_DR);
      const Rn = new Float64Array(9);
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
          Rn[3 * i + j] = RF_DR[3 * i] * R[j] + RF_DR[3 * i + 1] * R[3 + j] + RF_DR[3 * i + 2] * R[6 + j];
      const tn = Float64Array.of(t[0] + d[3], t[1] + d[4], t[2] + d[5]);
      const cn = poseCost(Rn, tn, X, xn, n, k);
      if (cn < cost) {
        const rel = (cost - cn) / Math.max(cost, 1e-300);
        const step = Math.abs(d[0]) + Math.abs(d[1]) + Math.abs(d[2]);
        const tsc = Math.abs(t[0]) + Math.abs(t[1]) + Math.abs(t[2]) + 1e-12;
        converged = rel < 1e-12 || (step < 1e-12 && Math.abs(d[3]) + Math.abs(d[4]) + Math.abs(d[5]) < 1e-12 * tsc);
        R = Rn;
        t = tn;
        cost = cn;
        lambda = Math.max(lambda * 0.1, 1e-10);
        improved = true;
        break;
      }
      lambda *= 10;
    }
    if (!improved || converged) break;
  }
  return { R, t };
}

// ---------------------------------------------------------------------------
// RANSAC
// ---------------------------------------------------------------------------

/** Options of `solvePnPRansac`. */
export interface PnPRansacOptions {
  /** Inlier threshold on the reprojection error in pixels (default 8). */
  maxErrorPx?: number;
  confidence?: number;
  maxIterations?: number;
  /** Minimum number of inliers for success (default 12). */
  minInliers?: number;
  rng?: Rng;
}

class PnPEstimator implements RansacEstimator<Pose> {
  sampleSize = 3;
  numData: number;
  minRefitSize = 6;
  private Xs = new Float64Array(9);
  private Bs = new Float64Array(9);
  private Xi: Float64Array;
  private xi: Float64Array;
  constructor(
    private X: ArrayLike<number>,
    private xn: Float64Array,
    private bear: Float64Array,
    n: number,
    private huber: number,
  ) {
    this.numData = n;
    this.Xi = new Float64Array(3 * n);
    this.xi = new Float64Array(2 * n);
  }
  isDegenerate(s: Uint32Array): boolean {
    const X = this.X;
    const a = s[0],
      b = s[1],
      c = s[2];
    const ux = X[3 * b] - X[3 * a],
      uy = X[3 * b + 1] - X[3 * a + 1],
      uz = X[3 * b + 2] - X[3 * a + 2];
    const vx = X[3 * c] - X[3 * a],
      vy = X[3 * c + 1] - X[3 * a + 1],
      vz = X[3 * c + 2] - X[3 * a + 2];
    const cx = uy * vz - uz * vy,
      cy = uz * vx - ux * vz,
      cz = ux * vy - uy * vx;
    const l = Math.max(ux * ux + uy * uy + uz * uz, vx * vx + vy * vy + vz * vz);
    return !(cx * cx + cy * cy + cz * cz > 1e-12 * l * l);
  }
  fit(s: Uint32Array, models: Pose[]): number {
    for (let k = 0; k < 3; k++) {
      const i = s[k];
      for (let c = 0; c < 3; c++) {
        this.Xs[3 * k + c] = this.X[3 * i + c];
        this.Bs[3 * k + c] = this.bear[3 * i + c];
      }
    }
    const sols = p3p(this.Xs, this.Bs);
    for (let k = 0; k < sols.length; k++) models[k] = sols[k];
    return sols.length;
  }
  residualsSq(m: Pose, out: Float64Array): void {
    this.residualsSqRange(m, out, 0, this.numData);
  }
  residualsSqRange(m: Pose, out: Float64Array, start: number, end: number): void {
    const R = m.R,
      t = m.t,
      X = this.X,
      xn = this.xn;
    const r0 = R[0], r1 = R[1], r2 = R[2], r3 = R[3], r4 = R[4], r5 = R[5], r6 = R[6], r7 = R[7], r8 = R[8];
    const t0 = t[0], t1 = t[1], t2 = t[2];
    for (let i = start; i < end; i++) {
      const x = X[3 * i],
        y = X[3 * i + 1],
        z = X[3 * i + 2];
      const pz = r6 * x + r7 * y + r8 * z + t2;
      if (!(pz > 0)) {
        out[i] = Infinity;
        continue;
      }
      const iz = 1 / pz;
      const du = (r0 * x + r1 * y + r2 * z + t0) * iz - xn[2 * i];
      const dv = (r3 * x + r4 * y + r5 * z + t1) * iz - xn[2 * i + 1];
      out[i] = du * du + dv * dv;
    }
  }
  refine(m: Pose, inl: Uint32Array, count: number): Pose | null {
    this.gather(inl, count);
    return refinePose(m, this.Xi, this.xi, count, { iterations: 4, huberNormalized: this.huber });
  }
  gather(inl: Uint32Array, count: number): void {
    for (let k = 0; k < count; k++) {
      const i = inl[k];
      this.Xi[3 * k] = this.X[3 * i];
      this.Xi[3 * k + 1] = this.X[3 * i + 1];
      this.Xi[3 * k + 2] = this.X[3 * i + 2];
      this.xi[2 * k] = this.xn[2 * i];
      this.xi[2 * k + 1] = this.xn[2 * i + 1];
    }
  }
  get gatheredX(): Float64Array {
    return this.Xi;
  }
  get gatheredXn(): Float64Array {
    return this.xi;
  }
}

/**
 * Robust absolute pose: observed pixels are undistorted once to normalised
 * coordinates; LO-MSAC with Lambda Twist P3P minimal samples (local optimisation =
 * a few robust LM steps) at threshold maxErrorPx / f; final robust LM refinement on
 * all inliers, inlier recount and a second refinement. Returns null if fewer than
 * minInliers (default 12) inliers remain.
 */
export function solvePnPRansac(
  X: Float64Array,
  uv: Float64Array,
  n: number,
  cam: CameraIntrinsics,
  opts: PnPRansacOptions = {},
): { pose: Pose; inlierMask: Uint8Array; numInliers: number } | null {
  const maxErrorPx = opts.maxErrorPx ?? 8;
  const minInliers = Math.max(3, opts.minInliers ?? 12);
  if (n < Math.max(3, minInliers)) return null;
  const xn = new Float64Array(2 * n);
  const bear = new Float64Array(3 * n);
  const tmp = new Float64Array(2);
  for (let i = 0; i < n; i++) {
    pixelToNormalized(cam, uv[2 * i], uv[2 * i + 1], tmp);
    xn[2 * i] = tmp[0];
    xn[2 * i + 1] = tmp[1];
    const inv = 1 / Math.sqrt(tmp[0] * tmp[0] + tmp[1] * tmp[1] + 1);
    bear[3 * i] = tmp[0] * inv;
    bear[3 * i + 1] = tmp[1] * inv;
    bear[3 * i + 2] = inv;
  }
  const thr = maxErrorPx / cam.f;
  const est = new PnPEstimator(X, xn, bear, n, 0.5 * thr);
  const res = ransac(est, {
    thresholdSq: thr * thr,
    confidence: opts.confidence ?? 0.9999,
    maxIterations: opts.maxIterations ?? 10000,
    rng: opts.rng,
  });
  if (!res.model || res.numInliers < minInliers) return null;
  let pose = res.model;
  const resid = new Float64Array(n);
  const inl = new Uint32Array(n);
  let mask = res.inlierMask;
  let numInliers = res.numInliers;
  for (let pass = 0; pass < 2; pass++) {
    let c = 0;
    for (let i = 0; i < n; i++) if (mask[i]) inl[c++] = i;
    est.gather(inl, c);
    pose = refinePose(pose, est.gatheredX, est.gatheredXn, c, { iterations: 20, huberNormalized: 0.5 * thr });
    est.residualsSq(pose, resid);
    const m2 = new Uint8Array(n);
    let c2 = 0;
    const t2 = thr * thr;
    for (let i = 0; i < n; i++) {
      if (resid[i] <= t2) {
        m2[i] = 1;
        c2++;
      }
    }
    mask = m2;
    numInliers = c2;
  }
  if (numInliers < minInliers) return null;
  return { pose, inlierMask: mask, numInliers };
}

