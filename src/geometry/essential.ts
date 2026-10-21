/**
 * Essential-matrix estimation and decomposition. All inputs are UNDISTORTED
 * NORMALISED image coordinates, flat [x0, y0, x1, y1, …].
 *
 * Convention (ARCHITECTURE §2): x2 = R x1 + t, E = [t]× R, x2ᵀ E x1 = 0.
 */
import type { Mat3, Vec3 } from '../core/types';
import { Matrix } from '../math/matrix';
import { nullSpace, solveSPD, svd3 } from '../math/decomp';
import { rotvecToMat3 } from '../math/rotation';
import { eigenvaluesNonsymmetric, eigenvectorInverseIteration } from './eigen';
import { denormalizeHartley, det3, hartleyNormalize, smallestEigenvector } from './internal';

// ---------------------------------------------------------------------------
// Five-point solver: Stewénius, Engels & Nistér 2006, "Recent developments on direct
// relative orientation", ISPRS J. Photogrammetry 60 (Gröbner basis / action matrix),
// with the constraint formulation of Nistér 2004 (IEEE TPAMI 26(6)).
//
// E = x X + y Y + z Z + W spans the 4-D null space of the 5 epipolar constraints.
// The 10 cubic constraints det(E) = 0 and 2 E Eᵀ E − tr(E Eᵀ) E = 0 are written as a
// 10×20 matrix over the monomials in GRevLex order
//   x³ x²y xy² y³ x²z xyz y²z xz² yz² z³ | x² xy y² xz yz z² x y z 1
// and reduced by Gauss–Jordan elimination. The last 10 monomials b form a basis of
// the quotient ring; the action matrix M of multiplication by a generic linear form
// ℓ = c·(x, y, z) (M b = ℓ b) is read off the reduced rows; its real eigenvalues give
// the solutions and (x, y, z) are read from the eigenvectors (inverse iteration).
// The eigen-approach is used instead of Nistér's degree-10 polynomial because the
// latter suffers catastrophic cancellation for small baselines (measured: true E
// lost in 26 % of noise-free samples at baseline/depth ≈ 1e-3…1e-2, vs. ~2–10 % here;
// below 1e-3 the true root often becomes a complex eigenvalue pair and is lost too —
// such parallax is below the noise. Numbers: docs/research/solvers.md).
// Each solution is polished by Gauss–Newton on the 10 original constraints.
// ---------------------------------------------------------------------------

const CUBIC_EXP: ReadonlyArray<readonly [number, number, number]> = [
  [3, 0, 0], [2, 1, 0], [1, 2, 0], [0, 3, 0], [2, 0, 1], [1, 1, 1], [0, 2, 1], [1, 0, 2], [0, 1, 2], [0, 0, 3],
  [2, 0, 0], [1, 1, 0], [0, 2, 0], [1, 0, 1], [0, 1, 1], [0, 0, 2], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0],
];
/**
 * Multiplication by v ∈ {x, y, z} on the basis b = [x², xy, y², xz, yz, z², x, y, z, 1]:
 * v·b_0..v·b_5 are leading monomials given by reduced rows ACTION_ROWS[v]; v·b_6..v·b_9
 * are basis monomials with indices ACTION_SHIFT[v].
 */
const ACTION_ROWS = [
  [0, 1, 2, 4, 5, 7],
  [1, 2, 3, 5, 6, 8],
  [4, 5, 6, 7, 8, 9],
];
const ACTION_SHIFT = [
  [0, 1, 3, 6],
  [1, 2, 4, 7],
  [3, 4, 5, 8],
];
/**
 * Weights of the action matrix M = Σ c_v M_v. A generic combination of x, y, z makes
 * coinciding eigenvalues (solutions sharing one coordinate) unlikely.
 */
const ACTION_COEF = [0.3, 1, 0.7];
const QUAD_EXP: ReadonlyArray<readonly [number, number, number]> = [
  [2, 0, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 2, 0], [0, 1, 1], [0, 1, 0], [0, 0, 2], [0, 0, 1], [0, 0, 0],
];
const LIN_EXP: ReadonlyArray<readonly [number, number, number]> = [
  [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0],
];

function expKey(e: readonly [number, number, number]): number {
  return e[0] * 16 + e[1] * 4 + e[2];
}

/** LL[4a+b] = quadratic index of lin_a · lin_b; QL[4q+a] = cubic index of quad_q · lin_a. */
const LL = new Uint8Array(16);
const QL = new Uint8Array(40);
{
  const quadIdx = new Int16Array(64).fill(-1);
  const cubIdx = new Int16Array(64).fill(-1);
  QUAD_EXP.forEach((e, i) => (quadIdx[expKey(e)] = i));
  CUBIC_EXP.forEach((e, i) => (cubIdx[expKey(e)] = i));
  for (let a = 0; a < 4; a++)
    for (let b = 0; b < 4; b++) {
      const e = [LIN_EXP[a][0] + LIN_EXP[b][0], LIN_EXP[a][1] + LIN_EXP[b][1], LIN_EXP[a][2] + LIN_EXP[b][2]] as const;
      LL[4 * a + b] = quadIdx[expKey(e)];
    }
  for (let q = 0; q < 10; q++)
    for (let a = 0; a < 4; a++) {
      const e = [QUAD_EXP[q][0] + LIN_EXP[a][0], QUAD_EXP[q][1] + LIN_EXP[a][1], QUAD_EXP[q][2] + LIN_EXP[a][2]] as const;
      QL[4 * q + a] = cubIdx[expKey(e)];
    }
}

// Scratch (module level; all solvers are synchronous).
const Q5 = new Matrix(5, 9);
const E4 = new Float64Array(36); // entry k: coefficients of [x, y, z, 1]
const EEt = new Float64Array(90); // 3×3 of quadratics
const TRQ = new Float64Array(10);
const COF = new Float64Array(30);
const A0 = new Float64Array(200); // original 10×20 constraints
const AG = new Float64Array(200); // Gauss–Jordan reduced
const MACT = new Float64Array(100);
const WR = new Float64Array(10);
const WI = new Float64Array(10);
const EVEC = new Float64Array(10);
const MONO = new Float64Array(20);
const DMONO = new Float64Array(60);

function linMulAdd(pa: number, pb: number, s: number, out: Float64Array, oo: number): void {
  for (let a = 0; a < 4; a++) {
    const va = s * E4[pa + a];
    if (va === 0) continue;
    for (let b = 0; b < 4; b++) out[oo + LL[4 * a + b]] += va * E4[pb + b];
  }
}

function quadMulAdd(q: Float64Array, qo: number, pl: number, s: number, out: Float64Array, oo: number): void {
  for (let i = 0; i < 10; i++) {
    const v = s * q[qo + i];
    if (v === 0) continue;
    for (let a = 0; a < 4; a++) out[oo + QL[4 * i + a]] += v * E4[pl + a];
  }
}

/** Fills A0 (10×20) with the cubic constraints of E(x,y,z) = xX + yY + zZ + W. */
function buildConstraints(): void {
  A0.fill(0);
  EEt.fill(0);
  for (let i = 0; i < 3; i++)
    for (let j = i; j < 3; j++) {
      const base = (3 * i + j) * 10;
      for (let k = 0; k < 3; k++) linMulAdd((3 * i + k) * 4, (3 * j + k) * 4, 1, EEt, base);
      if (j !== i) for (let q = 0; q < 10; q++) EEt[(3 * j + i) * 10 + q] = EEt[base + q];
    }
  for (let q = 0; q < 10; q++) TRQ[q] = EEt[q] + EEt[40 + q] + EEt[80 + q];
  // Rows 1..9: (2 E Eᵀ E − tr(E Eᵀ) E)_ij
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      const ro = (1 + 3 * i + j) * 20;
      for (let k = 0; k < 3; k++) quadMulAdd(EEt, (3 * i + k) * 10, (3 * k + j) * 4, 2, A0, ro);
      quadMulAdd(TRQ, 0, (3 * i + j) * 4, -1, A0, ro);
    }
  // Row 0: det(E) by cofactors of the first row.
  COF.fill(0);
  linMulAdd(4 * 4, 8 * 4, 1, COF, 0);
  linMulAdd(5 * 4, 7 * 4, -1, COF, 0);
  linMulAdd(3 * 4, 8 * 4, 1, COF, 10);
  linMulAdd(5 * 4, 6 * 4, -1, COF, 10);
  linMulAdd(3 * 4, 7 * 4, 1, COF, 20);
  linMulAdd(4 * 4, 6 * 4, -1, COF, 20);
  quadMulAdd(COF, 0, 0, 1, A0, 0);
  quadMulAdd(COF, 10, 4, -1, A0, 0);
  quadMulAdd(COF, 20, 8, 1, A0, 0);
}

/** Gauss–Jordan elimination with partial pivoting of AG's left 10×10 block. */
function gaussJordan(): boolean {
  let scale = 0;
  for (let i = 0; i < 200; i++) scale = Math.max(scale, Math.abs(AG[i]));
  if (!(scale > 0) || !Number.isFinite(scale)) return false;
  const tiny = 1e-14 * scale;
  for (let c = 0; c < 10; c++) {
    let p = c;
    let mv = Math.abs(AG[c * 20 + c]);
    for (let r = c + 1; r < 10; r++) {
      const v = Math.abs(AG[r * 20 + c]);
      if (v > mv) {
        mv = v;
        p = r;
      }
    }
    if (!(mv > tiny)) return false;
    if (p !== c) {
      for (let j = c; j < 20; j++) {
        const t = AG[c * 20 + j];
        AG[c * 20 + j] = AG[p * 20 + j];
        AG[p * 20 + j] = t;
      }
    }
    const inv = 1 / AG[c * 20 + c];
    for (let j = c; j < 20; j++) AG[c * 20 + j] *= inv;
    for (let r = 0; r < 10; r++) {
      if (r === c) continue;
      const f = AG[r * 20 + c];
      if (f === 0) continue;
      for (let j = c; j < 20; j++) AG[r * 20 + j] -= f * AG[c * 20 + j];
    }
  }
  return true;
}

/**
 * Monomial vector (GRevLex order above) at (x, y, z) and, when `deriv`, its partial
 * derivatives w.r.t. x, y, z (DMONO rows 0, 1, 2).
 */
function monomials(x: number, y: number, z: number, deriv: boolean): void {
  const xx = x * x,
    xy = x * y,
    yy = y * y,
    xz = x * z,
    yz = y * z,
    zz = z * z;
  const m = MONO;
  m[0] = xx * x;
  m[1] = xx * y;
  m[2] = x * yy;
  m[3] = yy * y;
  m[4] = xx * z;
  m[5] = xy * z;
  m[6] = yy * z;
  m[7] = x * zz;
  m[8] = y * zz;
  m[9] = zz * z;
  m[10] = xx;
  m[11] = xy;
  m[12] = yy;
  m[13] = xz;
  m[14] = yz;
  m[15] = zz;
  m[16] = x;
  m[17] = y;
  m[18] = z;
  m[19] = 1;
  if (!deriv) return;
  const d = DMONO;
  d.fill(0);
  // ∂/∂x
  d[0] = 3 * xx;
  d[1] = 2 * xy;
  d[2] = yy;
  d[4] = 2 * xz;
  d[5] = yz;
  d[7] = zz;
  d[10] = 2 * x;
  d[11] = y;
  d[13] = z;
  d[16] = 1;
  // ∂/∂y
  d[21] = xx;
  d[22] = 2 * xy;
  d[23] = 3 * yy;
  d[25] = xz;
  d[26] = 2 * yz;
  d[28] = zz;
  d[31] = x;
  d[32] = 2 * y;
  d[34] = z;
  d[37] = 1;
  // ∂/∂z
  d[44] = xx;
  d[45] = xy;
  d[46] = yy;
  d[47] = 2 * xz;
  d[48] = 2 * yz;
  d[49] = 3 * zz;
  d[53] = x;
  d[54] = y;
  d[55] = 2 * z;
  d[58] = 1;
}

const POL_H = new Float64Array(9);
const POL_G = new Float64Array(3);

/**
 * Gauss–Newton polish of (x, y, z) on the 10 original cubic constraints (in place).
 * Returns the relative residual max_r |Σ_c a_rc m_c| / Σ_c |a_rc m_c| at the returned
 * (accepted) point — not at a rejected trial step.
 */
function polish(sol: Float64Array): number {
  let x = sol[0],
    y = sol[1],
    z = sol[2];
  let rn = residualAt(x, y, z);
  let rel = relResidual;
  for (let it = 0; it < 5 && rn > 0 && rel > 1e-12; it++) {
    monomials(x, y, z, true);
    POL_H.fill(0);
    POL_G.fill(0);
    for (let r = 0; r < 10; r++) {
      let s = 0,
        j0 = 0,
        j1 = 0,
        j2 = 0;
      const ro = r * 20;
      for (let c = 0; c < 20; c++) {
        const a = A0[ro + c];
        s += a * MONO[c];
        j0 += a * DMONO[c];
        j1 += a * DMONO[20 + c];
        j2 += a * DMONO[40 + c];
      }
      POL_H[0] += j0 * j0;
      POL_H[1] += j0 * j1;
      POL_H[2] += j0 * j2;
      POL_H[4] += j1 * j1;
      POL_H[5] += j1 * j2;
      POL_H[8] += j2 * j2;
      POL_G[0] -= j0 * s;
      POL_G[1] -= j1 * s;
      POL_G[2] -= j2 * s;
    }
    POL_H[3] = POL_H[1];
    POL_H[6] = POL_H[2];
    POL_H[7] = POL_H[5];
    const d = solveSPD(POL_H, 3, POL_G);
    if (d === null || !Number.isFinite(d[0] + d[1] + d[2])) break;
    const nx = x + d[0],
      ny = y + d[1],
      nz = z + d[2];
    const rn2 = residualAt(nx, ny, nz);
    if (!(rn2 < rn)) break;
    const small = Math.abs(d[0]) + Math.abs(d[1]) + Math.abs(d[2]) <= 1e-15 * (1 + Math.abs(x) + Math.abs(y) + Math.abs(z));
    x = nx;
    y = ny;
    z = nz;
    rn = rn2;
    rel = relResidual;
    if (small) break;
  }
  sol[0] = x;
  sol[1] = y;
  sol[2] = z;
  return rel;
}

let relResidual = 0;

/** Sum of squared constraint residuals at (x, y, z); sets relResidual. */
function residualAt(x: number, y: number, z: number): number {
  monomials(x, y, z, false);
  let rn = 0;
  let rel = 0;
  for (let r = 0; r < 10; r++) {
    let s = 0,
      sa = 0;
    const ro = r * 20;
    for (let c = 0; c < 20; c++) {
      const v = A0[ro + c] * MONO[c];
      s += v;
      sa += Math.abs(v);
    }
    rn += s * s;
    if (sa > 0) rel = Math.max(rel, Math.abs(s) / sa);
  }
  relResidual = rel;
  return rn;
}

const SOL = new Float64Array(3);

/** Solutions E (singular values (1,1,0)) from the null-space basis stored in E4. */
function solveFromBasis(out: Mat3[]): void {
  buildConstraints();
  AG.set(A0);
  if (!gaussJordan()) return;
  // Action matrix of multiplication by ℓ = c·(x, y, z) on b = [x², xy, y², xz, yz, z², x, y, z, 1].
  MACT.fill(0);
  for (let v = 0; v < 3; v++) {
    const cv = ACTION_COEF[v];
    if (cv === 0) continue;
    const rows = ACTION_ROWS[v],
      shift = ACTION_SHIFT[v];
    for (let i = 0; i < 6; i++) {
      const ro = rows[i] * 20 + 10;
      for (let j = 0; j < 10; j++) MACT[10 * i + j] -= cv * AG[ro + j];
    }
    for (let i = 0; i < 4; i++) MACT[10 * (6 + i) + shift[i]] += cv;
  }
  if (!eigenvaluesNonsymmetric(MACT, 10, WR, WI)) return;
  const first = out.length;
  for (let k = 0; k < 10; k++) {
    const lr = WR[k],
      li = WI[k];
    if (!Number.isFinite(lr) || li < 0) continue;
    // Real eigenvalues are always used. Complex pairs close to the real axis (two
    // nearby real roots perturbed by rounding — frequent for small baselines) are
    // used as starting points of the polish and kept only if it converges to a real
    // solution of the original constraints.
    const isReal = li === 0;
    if (!isReal && li > 0.25 * (1 + Math.abs(lr))) continue;
    eigenvectorInverseIteration(MACT, 10, lr, EVEC, 2);
    const w = EVEC[9];
    if (!(Math.abs(w) > 1e-12)) continue;
    SOL[0] = EVEC[6] / w;
    SOL[1] = EVEC[7] / w;
    SOL[2] = EVEC[8] / w;
    if (!Number.isFinite(SOL[0] + SOL[1] + SOL[2])) continue;
    const rel = polish(SOL);
    // Complex starts must converge to a real root. Real eigenvalues whose polish does
    // not converge (ill-conditioned clusters near pure rotation) are not roots of the
    // system — their E is not even essential (σ1 ≠ σ2, σ3 ≠ 0) — and are dropped. True
    // solutions polish to rel ≲ 1e-9 (measured over 4800 noise-free configurations
    // incl. planar / forward motion / baseline/depth down to 1e-4).
    if (!(rel < (isReal ? 1e-6 : 1e-9))) continue;
    const x = SOL[0],
      y = SOL[1],
      zz = SOL[2];
    const E = new Float64Array(9);
    let nrm = 0;
    for (let q = 0; q < 9; q++) {
      const v = x * E4[4 * q] + y * E4[4 * q + 1] + zz * E4[4 * q + 2] + E4[4 * q + 3];
      E[q] = v;
      nrm += v * v;
    }
    if (!(nrm > 0) || !Number.isFinite(nrm)) continue;
    const sc = Math.SQRT2 / Math.sqrt(nrm);
    for (let q = 0; q < 9; q++) E[q] *= sc;
    // Skip duplicates (two eigenvalues polished onto the same solution).
    let dup = false;
    for (let j = first; j < out.length && !dup; j++) {
      const F = out[j];
      let dp = 0,
        dm = 0;
      for (let q = 0; q < 9; q++) {
        dp += (F[q] - E[q]) * (F[q] - E[q]);
        dm += (F[q] + E[q]) * (F[q] + E[q]);
      }
      dup = Math.min(dp, dm) < 1e-20;
    }
    if (!dup) out.push(E);
  }
}

/**
 * Minimal 5-point essential matrix solver (uses the first 5 correspondences of the
 * flat normalised-coordinate arrays x1, x2). Returns up to 10 real solutions,
 * each scaled to singular values (1, 1, 0) (sign arbitrary).
 */
export function essential5pt(x1: ArrayLike<number>, x2: ArrayLike<number>): Mat3[] {
  const out: Mat3[] = [];
  const q = Q5.data;
  for (let i = 0; i < 5; i++) {
    const a = x1[2 * i],
      b = x1[2 * i + 1],
      c = x2[2 * i],
      d = x2[2 * i + 1];
    const o = 9 * i;
    q[o] = c * a;
    q[o + 1] = c * b;
    q[o + 2] = c;
    q[o + 3] = d * a;
    q[o + 4] = d * b;
    q[o + 5] = d;
    q[o + 6] = a;
    q[o + 7] = b;
    q[o + 8] = 1;
  }
  const N = nullSpace(Q5, 4).data; // 9×4
  for (let k = 0; k < 9; k++) {
    E4[4 * k] = N[4 * k];
    E4[4 * k + 1] = N[4 * k + 1];
    E4[4 * k + 2] = N[4 * k + 2];
    E4[4 * k + 3] = N[4 * k + 3];
  }
  solveFromBasis(out);
  return out;
}

// ---------------------------------------------------------------------------
// Linear 8-point
// ---------------------------------------------------------------------------

const LIN_N1 = { buf: new Float64Array(64) };
const LIN_N2 = { buf: new Float64Array(64) };
const LIN_T1 = new Float64Array(9);
const LIN_T2 = new Float64Array(9);
const LIN_ATA = new Float64Array(81);
const LIN_V = new Float64Array(9);

/**
 * Linear eight-point estimate on the correspondences idx[0..n) (all when idx is
 * null), Hartley-normalised, de-normalised and projected onto the essential
 * manifold (singular values (1, 1, 0)). Null for n < 8 or degenerate data.
 */
export function essentialLinearIdx(
  x1: ArrayLike<number>,
  x2: ArrayLike<number>,
  idx: Uint32Array | null,
  n: number,
): Mat3 | null {
  if (n < 8) return null;
  if (LIN_N1.buf.length < 2 * n) {
    LIN_N1.buf = new Float64Array(4 * n);
    LIN_N2.buf = new Float64Array(4 * n);
  }
  const p1 = LIN_N1.buf,
    p2 = LIN_N2.buf;
  if (!hartleyNormalize(x1, n, idx, p1, LIN_T1) || !hartleyNormalize(x2, n, idx, p2, LIN_T2)) return null;
  const F = fitBilinearNormalized(p1, p2, n);
  if (F === null) return null;
  // De-normalise: E = T2ᵀ Ê T1
  const E = new Float64Array(9);
  denormalizeHartley(F, LIN_T1, LIN_T2, E);
  return projectToEssential(E);
}

/** Smallest-eigenvector solution of the n×9 bilinear system on normalised points. */
function fitBilinearNormalized(p1: Float64Array, p2: Float64Array, n: number): Float64Array | null {
  const A = LIN_ATA;
  A.fill(0);
  const r = LIN_V;
  for (let i = 0; i < n; i++) {
    const a = p1[2 * i],
      b = p1[2 * i + 1],
      c = p2[2 * i],
      d = p2[2 * i + 1];
    r[0] = c * a;
    r[1] = c * b;
    r[2] = c;
    r[3] = d * a;
    r[4] = d * b;
    r[5] = d;
    r[6] = a;
    r[7] = b;
    r[8] = 1;
    for (let u = 0; u < 9; u++) {
      const ru = r[u];
      const o = 9 * u;
      for (let v = u; v < 9; v++) A[o + v] += ru * r[v];
    }
  }
  const out = new Float64Array(9);
  smallestEigenvector(A, 9, out);
  if (!Number.isFinite(out[0])) return null;
  return out;
}

/** Nearest essential matrix (Frobenius): U diag(1,1,0) Vᵀ. Null for non-finite input. */
export function projectToEssential(E: ArrayLike<number>): Mat3 | null {
  const { U, S, V } = svd3(Float64Array.from(E as ArrayLike<number>));
  if (!Number.isFinite(S[0]) || !(S[1] > 0)) return null;
  const out = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) out[3 * i + j] = U[3 * i] * V[3 * j] + U[3 * i + 1] * V[3 * j + 1];
  return out;
}

/**
 * Linear eight-point essential matrix from n ≥ 8 normalised correspondences,
 * projected onto the essential manifold (singular values (1, 1, 0)).
 */
export function essentialLinear(x1: ArrayLike<number>, x2: ArrayLike<number>, n: number): Mat3 | null {
  return essentialLinearIdx(x1, x2, null, n);
}

// ---------------------------------------------------------------------------
// Decomposition / pose recovery
// ---------------------------------------------------------------------------

/** E = [t]× R. */
export function essentialFromPose(R: ArrayLike<number>, t: ArrayLike<number>, out: Mat3 = new Float64Array(9)): Mat3 {
  const t0 = t[0],
    t1 = t[1],
    t2 = t[2];
  for (let j = 0; j < 3; j++) {
    const r0 = R[j],
      r1 = R[3 + j],
      r2 = R[6 + j];
    out[j] = -t2 * r1 + t1 * r2;
    out[3 + j] = t2 * r0 - t0 * r2;
    out[6 + j] = -t1 * r0 + t0 * r1;
  }
  return out;
}

/**
 * The four (R, t) candidates of an essential matrix are (R1, ±t), (R2, ±t)
 * (Hartley & Zisserman, Result 9.19): E = U diag(1,1,0) Vᵀ with det U, det V > 0,
 * R1 = U W Vᵀ, R2 = U Wᵀ Vᵀ, t = u₃ (unit).
 */
export function decomposeEssential(E: Mat3): { R1: Mat3; R2: Mat3; t: Vec3 } {
  const { U, V } = svd3(E);
  if (det3(U) < 0) for (let i = 0; i < 9; i++) U[i] = -U[i];
  if (det3(V) < 0) for (let i = 0; i < 9; i++) V[i] = -V[i];
  const R1 = new Float64Array(9),
    R2 = new Float64Array(9);
  // U W = [u2, −u1, u3]; U Wᵀ = [−u2, u1, u3] (columns u1, u2, u3 of U).
  for (let i = 0; i < 3; i++) {
    const u0 = U[3 * i],
      u1 = U[3 * i + 1],
      u2 = U[3 * i + 2];
    for (let j = 0; j < 3; j++) {
      const v0 = V[3 * j],
        v1 = V[3 * j + 1],
        v2 = V[3 * j + 2];
      R1[3 * i + j] = u1 * v0 - u0 * v1 + u2 * v2;
      R2[3 * i + j] = -u1 * v0 + u0 * v1 + u2 * v2;
    }
  }
  const t = new Float64Array(3);
  t[0] = U[2];
  t[1] = U[5];
  t[2] = U[8];
  return { R1, R2, t };
}

/**
 * Points whose depth (in units of |t|) exceeds this are treated as "at infinity" by
 * the cheirality test (COLMAP uses the same bound of 1000 baselines).
 */
export const CHEIRALITY_MAX_DEPTH = 1000;

/**
 * Midpoint triangulation depths for x2 ≃ R x1 + t: counts correspondences with
 * positive, finite (< maxDepth·|t|) depth in both cameras. Writes outMask if given.
 */
export function cheiralityCount(
  R: ArrayLike<number>,
  t: ArrayLike<number>,
  x1: ArrayLike<number>,
  x2: ArrayLike<number>,
  n: number,
  mask: Uint8Array | null | undefined,
  outMask: Uint8Array | null,
  maxDepth = CHEIRALITY_MAX_DEPTH,
): number {
  const R0 = R[0],
    R1 = R[1],
    R2 = R[2],
    R3 = R[3],
    R4 = R[4],
    R5 = R[5],
    R6 = R[6],
    R7 = R[7],
    R8 = R[8];
  const t0 = t[0],
    t1 = t[1],
    t2 = t[2];
  const tn = Math.sqrt(t0 * t0 + t1 * t1 + t2 * t2);
  const dmax = maxDepth === Infinity ? Infinity : maxDepth * tn;
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (outMask) outMask[i] = 0;
    if (mask && !mask[i]) continue;
    const u = x1[2 * i],
      v = x1[2 * i + 1];
    const ax = R0 * u + R1 * v + R2,
      ay = R3 * u + R4 * v + R5,
      az = R6 * u + R7 * v + R8;
    const bx = x2[2 * i],
      by = x2[2 * i + 1];
    const aa = ax * ax + ay * ay + az * az;
    const bb = bx * bx + by * by + 1;
    const ab = ax * bx + ay * by + az;
    const at = ax * t0 + ay * t1 + az * t2;
    const bt = bx * t0 + by * t1 + t2;
    const det = aa * bb - ab * ab;
    if (!(det > 1e-15 * aa * bb)) continue;
    const l1 = (ab * bt - at * bb) / det;
    const l2 = (aa * bt - ab * at) / det;
    if (l1 > 0 && l2 > 0 && l1 < dmax && l2 < dmax) {
      count++;
      if (outMask) outMask[i] = 1;
    }
  }
  return count;
}

/** Σ cos∠(R x1, x2) over the (masked) correspondences: rotation consistency of R. */
function rotationAlignment(
  R: ArrayLike<number>,
  x1: ArrayLike<number>,
  x2: ArrayLike<number>,
  n: number,
  mask: Uint8Array | null | undefined,
): number {
  let s = 0;
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    const u = x1[2 * i],
      v = x1[2 * i + 1];
    const ax = R[0] * u + R[1] * v + R[2],
      ay = R[3] * u + R[4] * v + R[5],
      az = R[6] * u + R[7] * v + R[8];
    const bx = x2[2 * i],
      by = x2[2 * i + 1];
    const d = Math.sqrt((ax * ax + ay * ay + az * az) * (bx * bx + by * by + 1));
    if (d > 0) s += (ax * bx + ay * by + az) / d;
  }
  return s;
}

/**
 * Relative pose from an essential matrix: the candidate of `decomposeEssential` with
 * the most correspondences (optionally restricted to `mask`) in front of both
 * cameras (positive depth < CHEIRALITY_MAX_DEPTH baselines). |t| = 1.
 * Ties (e.g. every point farther than CHEIRALITY_MAX_DEPTH baselines, or exactly
 * parallel rays under pure rotation, where all four counts are 0) are broken by the
 * in-front count without the depth bound, then by the rotation consistency
 * Σ cos∠(R x1, x2) — which rejects the 180°-twisted rotation of the pair. When the
 * sign of t is still undecided (no parallax at all), (R, +t) is returned.
 */
export function recoverPose(
  E: Mat3,
  x1: ArrayLike<number>,
  x2: ArrayLike<number>,
  n: number,
  mask?: Uint8Array,
): { R: Mat3; t: Vec3; numInFront: number; inFrontMask: Uint8Array } {
  const { R1, R2, t } = decomposeEssential(E);
  const tn = new Float64Array(3);
  tn[0] = -t[0];
  tn[1] = -t[1];
  tn[2] = -t[2];
  const cands: [Mat3, Vec3][] = [
    [R1, t],
    [R1, tn],
    [R2, t],
    [R2, tn],
  ];
  const counts = [0, 0, 0, 0];
  let bestC = -1,
    numBest = 0;
  for (let k = 0; k < 4; k++) {
    const c = cheiralityCount(cands[k][0], cands[k][1], x1, x2, n, mask, null);
    counts[k] = c;
    if (c > bestC) {
      bestC = c;
      numBest = 1;
    } else if (c === bestC) numBest++;
  }
  let bestK = counts.indexOf(bestC);
  if (numBest > 1) {
    // Secondary key: in front at any depth; tertiary: rotation consistency.
    let align1 = NaN,
      align2 = NaN;
    let bestInf = -1,
      bestAlign = -Infinity;
    for (let k = 0; k < 4; k++) {
      if (counts[k] !== bestC) continue;
      const cInf = cheiralityCount(cands[k][0], cands[k][1], x1, x2, n, mask, null, Infinity);
      let al: number;
      if (k < 2) {
        if (Number.isNaN(align1)) align1 = rotationAlignment(R1, x1, x2, n, mask);
        al = align1;
      } else {
        if (Number.isNaN(align2)) align2 = rotationAlignment(R2, x1, x2, n, mask);
        al = align2;
      }
      if (cInf > bestInf || (cInf === bestInf && al > bestAlign)) {
        bestInf = cInf;
        bestAlign = al;
        bestK = k;
      }
    }
  }
  const inFrontMask = new Uint8Array(n);
  const numInFront = cheiralityCount(cands[bestK][0], cands[bestK][1], x1, x2, n, mask, inFrontMask);
  return { R: Float64Array.from(cands[bestK][0]), t: Float64Array.from(cands[bestK][1]), numInFront, inFrontMask };
}

/**
 * Squared Sampson distance of (x1, y1) ↔ (x2, y2) for x2ᵀ M x1 = 0 (Hartley &
 * Zisserman eq. 11.9). Works for E (normalised coordinates) and F (pixels).
 */
export function sampsonDistanceSq(M: ArrayLike<number>, x1: number, y1: number, x2: number, y2: number): number {
  const a0 = M[0] * x1 + M[1] * y1 + M[2];
  const a1 = M[3] * x1 + M[4] * y1 + M[5];
  const a2 = M[6] * x1 + M[7] * y1 + M[8];
  const b0 = M[0] * x2 + M[3] * y2 + M[6];
  const b1 = M[1] * x2 + M[4] * y2 + M[7];
  const e = x2 * a0 + y2 * a1 + a2;
  const d = a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1;
  if (!(d > 0)) return e === 0 ? 0 : Infinity;
  return (e * e) / d;
}

/** Squared Sampson distances of all n correspondences (flat arrays) into out. */
export function sampsonDistancesSq(
  M: ArrayLike<number>,
  p1: ArrayLike<number>,
  p2: ArrayLike<number>,
  n: number,
  out: Float64Array,
): void {
  sampsonDistancesSqRange(M, p1, p2, 0, n, out);
}

/** Squared Sampson distances of correspondences [start, end) into out[start..end). */
export function sampsonDistancesSqRange(
  M: ArrayLike<number>,
  p1: ArrayLike<number>,
  p2: ArrayLike<number>,
  start: number,
  end: number,
  out: Float64Array,
): void {
  const m0 = M[0],
    m1 = M[1],
    m2 = M[2],
    m3 = M[3],
    m4 = M[4],
    m5 = M[5],
    m6 = M[6],
    m7 = M[7],
    m8 = M[8];
  for (let i = start; i < end; i++) {
    const x1 = p1[2 * i],
      y1 = p1[2 * i + 1],
      x2 = p2[2 * i],
      y2 = p2[2 * i + 1];
    const a0 = m0 * x1 + m1 * y1 + m2;
    const a1 = m3 * x1 + m4 * y1 + m5;
    const a2 = m6 * x1 + m7 * y1 + m8;
    const b0 = m0 * x2 + m3 * y2 + m6;
    const b1 = m1 * x2 + m4 * y2 + m7;
    const e = x2 * a0 + y2 * a1 + a2;
    const d = a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1;
    out[i] = d > 0 ? (e * e) / d : e === 0 ? 0 : Infinity;
  }
}

// ---------------------------------------------------------------------------
// Nonlinear refinement of the relative pose on the Sampson error
// ---------------------------------------------------------------------------

/** Options of `refineRelativePose`. */
export interface RelativePoseRefineOptions {
  /** Maximum LM iterations (default 15). */
  iterations?: number;
  /** Huber threshold on the Sampson residual (normalised units, default Infinity = least squares). */
  huber?: number;
  /** Cauchy scale c (normalised units): ρ(r) = c² log(1 + r²/c²); overrides `huber` when set. */
  cauchy?: number;
}

const RP_DE = new Float64Array(45); // dE/dp, 9×5 (row = entry ij)
const RP_H = new Float64Array(25);
const RP_G = new Float64Array(5);
const RP_A = new Float64Array(25);

function relPoseCost(
  E: Float64Array,
  x1: ArrayLike<number>,
  x2: ArrayLike<number>,
  idx: Uint32Array | null,
  n: number,
  huber: number,
  cauchy: number,
): number {
  let cost = 0;
  const k2 = huber * huber;
  const c2 = cauchy * cauchy;
  for (let ii = 0; ii < n; ii++) {
    const i = idx ? idx[ii] : ii;
    const r2 = sampsonDistanceSq(E, x1[2 * i], x1[2 * i + 1], x2[2 * i], x2[2 * i + 1]);
    if (!Number.isFinite(r2)) continue;
    if (cauchy > 0) cost += c2 * Math.log1p(r2 / c2);
    else cost += r2 <= k2 ? r2 : 2 * huber * Math.sqrt(r2) - k2;
  }
  return cost;
}

/**
 * Levenberg–Marquardt refinement of a relative pose (R, t), |t| = 1, minimising the
 * (optionally Huber- or Cauchy-robustified, IRLS) Sampson error of the correspondences idx[0..n)
 * (all n when idx is null). Rotation updates are left-multiplicative, t moves on
 * the unit sphere (2-D tangent update). The cost depends on E = [t]× R only, so any
 * of the four decompositions of E gives the same refined E; use `recoverPose`
 * afterwards to select the physically valid one.
 */
export function refineRelativePose(
  R0: ArrayLike<number>,
  t0: ArrayLike<number>,
  x1: ArrayLike<number>,
  x2: ArrayLike<number>,
  n: number,
  idx: Uint32Array | null = null,
  opts: RelativePoseRefineOptions = {},
): { R: Mat3; t: Vec3; E: Mat3; cost: number; initialCost: number } {
  const iterations = opts.iterations ?? 15;
  const huber = opts.huber ?? Infinity;
  const cauchy = opts.cauchy !== undefined && opts.cauchy > 0 ? opts.cauchy : 0;
  const cauchySq = cauchy * cauchy;
  let R = Float64Array.from(R0 as ArrayLike<number>);
  let t = Float64Array.from(t0 as ArrayLike<number>);
  {
    const tn = Math.hypot(t[0], t[1], t[2]);
    if (tn > 0) for (let k = 0; k < 3; k++) t[k] /= tn;
  }
  let E = essentialFromPose(R, t);
  let cost = relPoseCost(E, x1, x2, idx, n, huber, cauchy);
  const initialCost = cost;
  let lambda = 1e-4;
  const b1 = new Float64Array(3),
    b2 = new Float64Array(3);
  const tmp9 = new Float64Array(9);
  const dR = new Float64Array(9);
  const w3 = new Float64Array(3);
  const g = new Float64Array(9);
  const Jf = new Float64Array(5);
  const rhs = new Float64Array(5);
  for (let it = 0; it < iterations && n >= 5 && cost > 0; it++) {
    // Tangent basis of t.
    tangentBasis(t, b1, b2);
    // dE/dω_k = [t]× [e_k]× R ; dE/dβ_k = [b_k]× R
    for (let k = 0; k < 3; k++) {
      // [e_k]× R: rows. [e0]× = [[0,0,0],[0,0,-1],[0,1,0]] etc.
      for (let j = 0; j < 3; j++) {
        const r0 = R[j],
          r1 = R[3 + j],
          r2 = R[6 + j];
        let c0 = 0,
          c1 = 0,
          c2 = 0;
        if (k === 0) {
          c1 = -r2;
          c2 = r1;
        } else if (k === 1) {
          c0 = r2;
          c2 = -r0;
        } else {
          c0 = -r1;
          c1 = r0;
        }
        // [t]× c
        tmp9[j] = t[1] * c2 - t[2] * c1;
        tmp9[3 + j] = t[2] * c0 - t[0] * c2;
        tmp9[6 + j] = t[0] * c1 - t[1] * c0;
      }
      for (let e = 0; e < 9; e++) RP_DE[5 * e + k] = tmp9[e];
    }
    for (let k = 0; k < 2; k++) {
      const b = k === 0 ? b1 : b2;
      essentialFromPose(R, b, tmp9);
      for (let e = 0; e < 9; e++) RP_DE[5 * e + 3 + k] = tmp9[e];
    }
    RP_H.fill(0);
    RP_G.fill(0);
    for (let ii = 0; ii < n; ii++) {
      const i = idx ? idx[ii] : ii;
      const u1 = x1[2 * i],
        v1 = x1[2 * i + 1],
        u2 = x2[2 * i],
        v2 = x2[2 * i + 1];
      const a0 = E[0] * u1 + E[1] * v1 + E[2];
      const a1 = E[3] * u1 + E[4] * v1 + E[5];
      const a2 = E[6] * u1 + E[7] * v1 + E[8];
      const c0 = E[0] * u2 + E[3] * v2 + E[6];
      const c1 = E[1] * u2 + E[4] * v2 + E[7];
      const e = u2 * a0 + v2 * a1 + a2;
      const D = a0 * a0 + a1 * a1 + c0 * c0 + c1 * c1;
      if (!(D > 0)) continue;
      const sD = Math.sqrt(D);
      const r = e / sD;
      const ar = Math.abs(r);
      const w = cauchy > 0 ? 1 / (1 + (r * r) / cauchySq) : ar <= huber ? 1 : huber / ar;
      const iD = 1 / sD;
      const f = e / (D * sD);
      // ∂r/∂E_pq = x2_p x1_q / √D − f ((E x1)_p x1_q [p<2] + (Eᵀ x2)_q x2_p [q<2])
      g[0] = u2 * u1 * iD - f * (a0 * u1 + c0 * u2);
      g[1] = u2 * v1 * iD - f * (a0 * v1 + c1 * u2);
      g[2] = u2 * iD - f * a0;
      g[3] = v2 * u1 * iD - f * (a1 * u1 + c0 * v2);
      g[4] = v2 * v1 * iD - f * (a1 * v1 + c1 * v2);
      g[5] = v2 * iD - f * a1;
      g[6] = u1 * iD - f * c0;
      g[7] = v1 * iD - f * c1;
      g[8] = iD;
      for (let k = 0; k < 5; k++) {
        let s = 0;
        for (let e9 = 0; e9 < 9; e9++) s += g[e9] * RP_DE[5 * e9 + k];
        Jf[k] = s;
      }
      for (let a = 0; a < 5; a++) {
        const wa = w * Jf[a];
        RP_G[a] += wa * r;
        for (let b = a; b < 5; b++) RP_H[5 * a + b] += wa * Jf[b];
      }
    }
    for (let a = 0; a < 5; a++) for (let b = 0; b < a; b++) RP_H[5 * a + b] = RP_H[5 * b + a];
    let improved = false;
    for (let tries = 0; tries < 8; tries++) {
      RP_A.set(RP_H);
      for (let a = 0; a < 5; a++) RP_A[6 * a] += lambda * Math.max(RP_H[6 * a], 1e-12);
      for (let a = 0; a < 5; a++) rhs[a] = -RP_G[a];
      const d = solveSPD(RP_A, 5, rhs);
      if (d === null) {
        lambda *= 10;
        continue;
      }
      w3[0] = d[0];
      w3[1] = d[1];
      w3[2] = d[2];
      rotvecToMat3(w3, dR);
      const Rn = new Float64Array(9);
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
          Rn[3 * i + j] = dR[3 * i] * R[j] + dR[3 * i + 1] * R[3 + j] + dR[3 * i + 2] * R[6 + j];
      const tn = new Float64Array(3);
      for (let k = 0; k < 3; k++) tn[k] = t[k] + d[3] * b1[k] + d[4] * b2[k];
      const nt = Math.hypot(tn[0], tn[1], tn[2]);
      for (let k = 0; k < 3; k++) tn[k] /= nt;
      const En = essentialFromPose(Rn, tn);
      const cn = relPoseCost(En, x1, x2, idx, n, huber, cauchy);
      if (cn < cost) {
        const rel = (cost - cn) / cost;
        R = Rn;
        t = tn;
        E = En;
        cost = cn;
        lambda = Math.max(lambda * 0.1, 1e-12);
        improved = true;
        if (rel < 1e-10) it = iterations; // converged
        break;
      }
      lambda *= 10;
    }
    if (!improved) break;
  }
  return { R, t, E, cost, initialCost };
}

/** Orthonormal b1, b2 ⟂ unit t. */
export function tangentBasis(t: ArrayLike<number>, b1: Float64Array, b2: Float64Array): void {
  const ax = Math.abs(t[0]),
    ay = Math.abs(t[1]),
    az = Math.abs(t[2]);
  // pick the axis least aligned with t
  let ex = 0,
    ey = 0,
    ez = 0;
  if (ax <= ay && ax <= az) ex = 1;
  else if (ay <= az) ey = 1;
  else ez = 1;
  // b1 = normalize(e × t)
  let c0 = ey * t[2] - ez * t[1],
    c1 = ez * t[0] - ex * t[2],
    c2 = ex * t[1] - ey * t[0];
  const nc = Math.hypot(c0, c1, c2);
  c0 /= nc;
  c1 /= nc;
  c2 /= nc;
  b1[0] = c0;
  b1[1] = c1;
  b1[2] = c2;
  // b2 = t × b1
  b2[0] = t[1] * c2 - t[2] * c1;
  b2[1] = t[2] * c0 - t[0] * c2;
  b2[2] = t[0] * c1 - t[1] * c0;
}


