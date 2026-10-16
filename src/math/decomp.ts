/**
 * Dense matrix decompositions and solvers (row-major Float64Array storage).
 *
 *  - Cholesky (Cholesky–Banachiewicz, row oriented) + triangular solves.
 *  - LU with partial pivoting.
 *  - Householder QR (thin).
 *  - SVD by one-sided (Hestenes) Jacobi, with Householder-QR preconditioning for
 *    tall matrices. One-sided Jacobi is the most accurate dense SVD algorithm
 *    (Demmel & Veselić 1992, "Jacobi's method is more accurate than QR"; Drmač &
 *    Veselić 2008, "New fast and accurate Jacobi SVD algorithm I/II"). V is the
 *    accumulated product of plane rotations and is therefore orthonormal to
 *    machine precision even for rank-deficient input, which is what null-space
 *    extraction relies on.
 *  - Symmetric eigen-decomposition by cyclic two-sided Jacobi (Golub & Van Loan,
 *    "Matrix Computations", §8.5; Press et al., "Numerical Recipes", §11.1).
 */
import type { Mat3, Vec3 } from '../core/types';
import { Matrix } from './matrix';

const EPS = 2.220446049250313e-16;
const JACOBI_MAX_SWEEPS = 60;

// ---------------------------------------------------------------------------
// Cholesky
// ---------------------------------------------------------------------------

/**
 * In-place Cholesky factorisation A = L Lᵀ of a symmetric positive-definite n x n
 * row-major matrix. Only the LOWER triangle of A is read. On success the lower
 * triangle (incl. diagonal) holds L and the strict upper triangle is zeroed.
 *
 * Returns false (A is then partially overwritten) when a pivot is non-positive,
 * non-finite, or ≤ 1e-15 times the original diagonal entry (i.e. the matrix is
 * not numerically positive definite: cond(A) ≳ 1e15 or semi-definite).
 */
export function cholesky(A: Float64Array, n: number): boolean {
  for (let i = 0; i < n; i++) {
    const ri = i * n;
    for (let j = 0; j < i; j++) {
      const rj = j * n;
      let s = A[ri + j];
      for (let k = 0; k < j; k++) s -= A[ri + k] * A[rj + k];
      A[ri + j] = s / A[rj + j];
    }
    const aii = A[ri + i];
    let s = aii;
    for (let k = 0; k < i; k++) {
      const l = A[ri + k];
      s -= l * l;
    }
    // Also rejects NaN / ±Infinity and non-positive pivots.
    if (!(s > 1e-15 * Math.abs(aii)) || s === Infinity) return false;
    A[ri + i] = Math.sqrt(s);
    for (let j = i + 1; j < n; j++) A[ri + j] = 0;
  }
  return true;
}

/** In-place solve of L Lᵀ x = b given the Cholesky factor from `cholesky` (b is overwritten by x). */
export function choleskySolve(L: Float64Array, n: number, b: Float64Array): void {
  // Forward: L y = b
  for (let i = 0; i < n; i++) {
    const ri = i * n;
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[ri + k] * b[k];
    b[i] = s / L[ri + i];
  }
  // Backward: Lᵀ x = y, column-oriented so that rows of L are read contiguously.
  for (let i = n - 1; i >= 0; i--) {
    const ri = i * n;
    const xi = b[i] / L[ri + i];
    b[i] = xi;
    if (xi !== 0) for (let k = 0; k < i; k++) b[k] -= L[ri + k] * xi;
  }
}

/** Solve A x = b for SPD A (n x n, row-major) without modifying A or b. Null if A is not SPD. */
export function solveSPD(A: Float64Array, n: number, b: Float64Array): Float64Array | null {
  const L = A.slice(0, n * n);
  if (!cholesky(L, n)) return null;
  const x = b.slice(0, n);
  choleskySolve(L, n, x);
  return x;
}

// ---------------------------------------------------------------------------
// LU
// ---------------------------------------------------------------------------

/**
 * Solve A x = b by Gaussian elimination with partial pivoting. Non-destructive.
 * Returns null when A is singular to working precision: a pivot is ≤ 4·n·ε
 * times the max-abs of its ORIGINAL row (row-scaling invariant), or the result
 * is non-finite.
 */
export function luSolve(A: Float64Array, n: number, b: Float64Array): Float64Array | null {
  const lu = A.slice(0, n * n);
  const x = b.slice(0, n);
  const rowScale = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const ri = i * n;
    for (let j = 0; j < n; j++) {
      const v = Math.abs(lu[ri + j]);
      if (v > s) s = v;
    }
    if (!(s > 0) || !Number.isFinite(s)) return null;
    rowScale[i] = s;
  }
  const tol = 4 * n * EPS;
  for (let k = 0; k < n; k++) {
    let p = k;
    let best = Math.abs(lu[k * n + k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(lu[i * n + k]);
      if (v > best) {
        best = v;
        p = i;
      }
    }
    if (!(best > tol * rowScale[p])) return null;
    if (p !== k) {
      const rk = k * n,
        rp = p * n;
      for (let j = k; j < n; j++) {
        const t = lu[rk + j];
        lu[rk + j] = lu[rp + j];
        lu[rp + j] = t;
      }
      let t = x[k];
      x[k] = x[p];
      x[p] = t;
      t = rowScale[k];
      rowScale[k] = rowScale[p];
      rowScale[p] = t;
    }
    const rk = k * n;
    const piv = lu[rk + k];
    for (let i = k + 1; i < n; i++) {
      const ri = i * n;
      const f = lu[ri + k] / piv;
      if (f === 0) continue;
      for (let j = k + 1; j < n; j++) lu[ri + j] -= f * lu[rk + j];
      x[i] -= f * x[k];
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    const ri = i * n;
    let s = x[i];
    for (let j = i + 1; j < n; j++) s -= lu[ri + j] * x[j];
    x[i] = s / lu[ri + i];
  }
  for (let i = 0; i < n; i++) if (!Number.isFinite(x[i])) return null;
  return x;
}

// ---------------------------------------------------------------------------
// Householder QR (internal: works on the TRANSPOSED layout so columns are contiguous)
// ---------------------------------------------------------------------------

/**
 * Householder QR of the m x n matrix whose columns are the rows of `Wt` (n rows of
 * length m). In place: R(j, c) for c ≥ j ends up in Wt[c*m + j]; the essential part
 * of reflector j (v_j with implicit v_j[j] = 1) is stored in Wt[j*m + j+1 .. j*m + m-1].
 * H_j = I − tau[j] v_j v_jᵀ; A = H_0 H_1 … H_{k-1} R, k = min(m, n). (LAPACK dgeqr2 scheme.)
 */
function householderQRt(Wt: Float64Array, m: number, n: number, tau: Float64Array): void {
  const k = Math.min(m, n);
  for (let j = 0; j < k; j++) {
    const rj = j * m;
    const alpha = Wt[rj + j];
    let sigma = 0;
    for (let i = j + 1; i < m; i++) sigma += Wt[rj + i] * Wt[rj + i];
    if (sigma === 0) {
      tau[j] = 0;
      continue;
    }
    const normx = Math.sqrt(alpha * alpha + sigma);
    const beta = alpha <= 0 ? normx : -normx;
    const v0 = alpha - beta;
    const t = (beta - alpha) / beta;
    tau[j] = t;
    const inv = 1 / v0;
    for (let i = j + 1; i < m; i++) Wt[rj + i] *= inv;
    Wt[rj + j] = beta;
    for (let c = j + 1; c < n; c++) {
      const rc = c * m;
      let s = Wt[rc + j];
      for (let i = j + 1; i < m; i++) s += Wt[rj + i] * Wt[rc + i];
      s *= t;
      if (s === 0) continue;
      Wt[rc + j] -= s;
      for (let i = j + 1; i < m; i++) Wt[rc + i] -= s * Wt[rj + i];
    }
  }
}

/** Apply Q = H_0 … H_{k-1} (from householderQRt) to the length-m vector u in place: u ← Q u. */
function applyQ(Wt: Float64Array, m: number, k: number, tau: Float64Array, u: Float64Array): void {
  for (let j = k - 1; j >= 0; j--) {
    const t = tau[j];
    if (t === 0) continue;
    const rj = j * m;
    let s = u[j];
    for (let i = j + 1; i < m; i++) s += Wt[rj + i] * u[i];
    s *= t;
    u[j] -= s;
    for (let i = j + 1; i < m; i++) u[i] -= s * Wt[rj + i];
  }
}

/** Apply Qᵀ = H_{k-1} … H_0 to the length-m vector u in place: u ← Qᵀ u. */
function applyQt(Wt: Float64Array, m: number, k: number, tau: Float64Array, u: Float64Array): void {
  for (let j = 0; j < k; j++) {
    const t = tau[j];
    if (t === 0) continue;
    const rj = j * m;
    let s = u[j];
    for (let i = j + 1; i < m; i++) s += Wt[rj + i] * u[i];
    s *= t;
    u[j] -= s;
    for (let i = j + 1; i < m; i++) u[i] -= s * Wt[rj + i];
  }
}

/**
 * Thin Householder QR: A (m x n) = Q R with Q m x k (orthonormal columns), R k x n
 * upper trapezoidal, k = min(m, n). The diagonal of R is made non-negative.
 */
export function qr(A: Matrix): { Q: Matrix; R: Matrix } {
  const m = A.rows,
    n = A.cols,
    k = Math.min(m, n);
  const a = A.data;
  // Scale to max-abs 1 (no overflow in the reflector norms); R is unscaled below.
  const sc = maxAbs(a, m * n);
  const is = sc > 0 && Number.isFinite(sc) ? 1 / sc : 1;
  const Wt = new Float64Array(n * m);
  for (let r = 0; r < m; r++) for (let c = 0; c < n; c++) Wt[c * m + r] = a[r * n + c] * is;
  const tau = new Float64Array(k);
  householderQRt(Wt, m, n, tau);
  const R = new Matrix(k, n);
  const Rd = R.data;
  const us = is === 1 ? 1 : sc;
  for (let j = 0; j < k; j++) for (let c = j; c < n; c++) Rd[j * n + c] = Wt[c * m + j] * us;
  const Q = new Matrix(m, k);
  const Qd = Q.data;
  const u = new Float64Array(m);
  for (let c = 0; c < k; c++) {
    u.fill(0);
    u[c] = 1;
    applyQ(Wt, m, k, tau, u);
    // Make diag(R) ≥ 0 by flipping the sign of row c of R and column c of Q.
    const flip = Rd[c * n + c] < 0;
    if (flip) for (let j = c; j < n; j++) Rd[c * n + j] = -Rd[c * n + j];
    for (let r = 0; r < m; r++) Qd[r * k + c] = flip ? -u[r] : u[r];
  }
  return { Q, R };
}

/**
 * Linear least squares min ‖A x − b‖₂ via Householder QR (m ≥ n). Returns null when
 * m < n or A is numerically rank deficient (|R_jj| ≤ n·ε·max|R_ii|·10).
 */
export function solveLeastSquares(A: Matrix, b: ArrayLike<number>): Float64Array | null {
  const m = A.rows,
    n = A.cols;
  if (m < n || n === 0) return null;
  const a = A.data;
  // Solve with A/s (s = max|A|, no overflow), then x = x'/s.
  const sc = maxAbs(a, m * n);
  if (!(sc > 0) || !Number.isFinite(sc)) return null;
  const is = 1 / sc;
  const Wt = new Float64Array(n * m);
  for (let r = 0; r < m; r++) for (let c = 0; c < n; c++) Wt[c * m + r] = a[r * n + c] * is;
  const tau = new Float64Array(n);
  householderQRt(Wt, m, n, tau);
  const y = new Float64Array(m);
  for (let i = 0; i < m; i++) y[i] = b[i];
  applyQt(Wt, m, n, tau, y);
  let rmax = 0;
  for (let j = 0; j < n; j++) rmax = Math.max(rmax, Math.abs(Wt[j * m + j]));
  const tol = 10 * n * EPS * rmax;
  const x = new Float64Array(n);
  for (let j = n - 1; j >= 0; j--) {
    const rjj = Wt[j * m + j];
    if (!(Math.abs(rjj) > tol)) return null;
    let s = y[j];
    for (let c = j + 1; c < n; c++) s -= Wt[c * m + j] * x[c];
    x[j] = s / rjj;
  }
  for (let j = 0; j < n; j++) {
    x[j] *= is;
    if (!Number.isFinite(x[j])) return null;
  }
  return x;
}

// ---------------------------------------------------------------------------
// One-sided Jacobi SVD kernel
// ---------------------------------------------------------------------------

// Module-level scratch buffers (grown on demand, reused across calls). All
// routines below are synchronous and never call user code, so sharing is safe.
// Avoiding per-call allocation matters: off-heap typed-array allocation
// dominated the run time of small SVDs (e.g. 8 x 9 in RANSAC loops).
let scrW: Float64Array = new Float64Array(256);
let scrWt: Float64Array = new Float64Array(256);
let scrVt: Float64Array = new Float64Array(256);
let scrTau: Float64Array = new Float64Array(32);
let scrSig: Float64Array = new Float64Array(32);
let scrNrm: Float64Array = new Float64Array(32);
let scrTmp: Float64Array = new Float64Array(64);
let scrValid = new Uint8Array(32);
let scrOrder = new Int32Array(32);
let scrEigA: Float64Array = new Float64Array(64);
let scrEigV: Float64Array = new Float64Array(64);
let scrEigD: Float64Array = new Float64Array(32);

function growF64(a: Float64Array, n: number): Float64Array {
  return a.length >= n ? a : new Float64Array(Math.max(n, 2 * a.length));
}

/**
 * One-sided (Hestenes) Jacobi: orthogonalises the `nc` vectors of length `len`
 * stored as the rows of W by plane rotations applied from the right; the same
 * rotations are accumulated into the rows of Vt (nc x nc, must be initialised,
 * typically to the identity; row i of Vt = column i of V). `nrm` (≥ nc) is scratch
 * for the squared row norms, which are updated per rotation (‖w_p‖² −= tγ,
 * ‖w_q‖² += tγ; recomputed when cancellation is possible and at every sweep, as in
 * LAPACK dgesvj) so that each pair costs a single dot product.
 *
 * A pair (p, q) is rotated unless |w_p·w_q| ≤ len·ε·‖w_p‖‖w_q‖ or one of the two
 * vectors has squared norm ≤ `tiny` = ε²·‖W‖²_F (numerically zero; this keeps
 * the iteration finite when nc > len, i.e. for wide matrices).
 *
 * Input should be pre-scaled to max-abs ≈ 1 to rule out over/underflow.
 * Returns `tiny`.
 */
function jacobiOrthogonalize(W: Float64Array, nc: number, len: number, Vt: Float64Array, nrm: Float64Array): number {
  let fro2 = 0;
  for (let i = 0; i < nc; i++) {
    let s = 0;
    const o = i * len;
    for (let k = 0; k < len; k++) s += W[o + k] * W[o + k];
    nrm[i] = s;
    fro2 += s;
  }
  const tiny = fro2 * EPS * EPS;
  const tol = Math.max(1, len) * EPS;
  for (let sweep = 0; sweep < JACOBI_MAX_SWEEPS; sweep++) {
    if (sweep > 0) {
      for (let i = 0; i < nc; i++) {
        let s = 0;
        const o = i * len;
        for (let k = 0; k < len; k++) s += W[o + k] * W[o + k];
        nrm[i] = s;
      }
    }
    let rotated = false;
    for (let p = 0; p < nc - 1; p++) {
      const op = p * len;
      for (let q = p + 1; q < nc; q++) {
        const alpha = nrm[p],
          beta = nrm[q];
        if (alpha <= tiny || beta <= tiny) continue;
        const oq = q * len;
        let gamma = 0;
        for (let k = 0; k < len; k++) gamma += W[op + k] * W[oq + k];
        if (Math.abs(gamma) <= tol * Math.sqrt(alpha * beta)) continue;
        rotated = true;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;
        for (let k = 0; k < len; k++) {
          const x = W[op + k],
            y = W[oq + k];
          W[op + k] = c * x - s * y;
          W[oq + k] = s * x + c * y;
        }
        const vp = p * nc,
          vq = q * nc;
        for (let k = 0; k < nc; k++) {
          const x = Vt[vp + k],
            y = Vt[vq + k];
          Vt[vp + k] = c * x - s * y;
          Vt[vq + k] = s * x + c * y;
        }
        const tg = t * gamma;
        let na = alpha - tg,
          nb = beta + tg;
        if (!(na >= 0.25 * alpha)) {
          na = 0;
          for (let k = 0; k < len; k++) na += W[op + k] * W[op + k];
        }
        if (!(nb >= 0.25 * beta)) {
          nb = 0;
          for (let k = 0; k < len; k++) nb += W[oq + k] * W[oq + k];
        }
        nrm[p] = na;
        nrm[q] = nb;
      }
    }
    if (!rotated) break;
  }
  return tiny;
}

/**
 * Fill the rows of Ut (nvec rows of length len, nvec ≤ len) whose `valid` flag is 0
 * with unit vectors orthogonal to every valid row and to each other (Gram–Schmidt
 * of the canonical basis vector least represented in the current span, applied
 * twice for numerical orthogonality). `tmp` needs length ≥ len.
 */
function completeOrthonormal(
  Ut: Float64Array,
  nvec: number,
  len: number,
  valid: Uint8Array,
  tmp: Float64Array,
): void {
  for (let r = 0; r < nvec; r++) {
    if (valid[r]) continue;
    let bestK = 0,
      bestRes = -Infinity;
    for (let k = 0; k < len; k++) {
      let res = 1;
      for (let q = 0; q < nvec; q++) {
        if (!valid[q]) continue;
        const u = Ut[q * len + k];
        res -= u * u;
      }
      if (res > bestRes) {
        bestRes = res;
        bestK = k;
      }
    }
    for (let k = 0; k < len; k++) tmp[k] = 0;
    tmp[bestK] = 1;
    for (let pass = 0; pass < 2; pass++) {
      for (let q = 0; q < nvec; q++) {
        if (!valid[q]) continue;
        const oq = q * len;
        let d = 0;
        for (let k = 0; k < len; k++) d += Ut[oq + k] * tmp[k];
        if (d !== 0) for (let k = 0; k < len; k++) tmp[k] -= d * Ut[oq + k];
      }
      let nn = 0;
      for (let k = 0; k < len; k++) nn += tmp[k] * tmp[k];
      const inv = 1 / Math.sqrt(nn);
      for (let k = 0; k < len; k++) tmp[k] *= inv;
    }
    const or = r * len;
    for (let k = 0; k < len; k++) Ut[or + k] = tmp[k];
    valid[r] = 1;
  }
}

/** Indices 0..n-1 sorted by key descending (stable; insertion sort, n is small). */
function argsortDesc(key: Float64Array, n: number, order: Int32Array): void {
  for (let i = 0; i < n; i++) order[i] = i;
  for (let i = 1; i < n; i++) {
    const oi = order[i];
    const ki = key[oi];
    let j = i - 1;
    while (j >= 0 && key[order[j]] < ki) {
      order[j + 1] = order[j];
      j--;
    }
    order[j + 1] = oi;
  }
}

function maxAbs(a: ArrayLike<number>, len: number): number {
  let s = 0;
  for (let i = 0; i < len; i++) {
    const v = Math.abs(a[i]);
    if (v > s || v !== v) s = v; // propagate NaN
  }
  return s;
}

interface TallSvd {
  U: Matrix | null;
  S: Float64Array;
  V: Matrix;
}

/** SVD of a row-major m x n matrix with m ≥ n. U is m x n (when wanted), V n x n. */
function svdTall(a: Float64Array, m: number, n: number, wantU: boolean): TallSvd {
  const S = new Float64Array(n);
  const V = new Matrix(n, n);
  const U = wantU ? new Matrix(m, n) : null;
  const scale = maxAbs(a, m * n);
  if (!Number.isFinite(scale)) {
    S.fill(NaN);
    V.data.fill(NaN);
    if (U) U.data.fill(NaN);
    return { U, S, V };
  }
  if (scale === 0 || n === 0) {
    for (let i = 0; i < n; i++) V.data[i * n + i] = 1;
    if (U) for (let i = 0; i < n; i++) U.data[i * n + i] = 1;
    return { U, S, V };
  }
  const inv = 1 / scale;
  const nn = n * n;
  scrW = growF64(scrW, nn);
  scrVt = growF64(scrVt, nn);
  scrNrm = growF64(scrNrm, n);
  scrSig = growF64(scrSig, n);
  scrTmp = growF64(scrTmp, m);
  if (scrValid.length < n) scrValid = new Uint8Array(Math.max(n, 2 * scrValid.length));
  if (scrOrder.length < n) scrOrder = new Int32Array(Math.max(n, 2 * scrOrder.length));
  const W = scrW; // rows = columns of the (preconditioned) n x n matrix
  const qrPre = m > n;
  if (qrPre) {
    // QR preconditioning: A = Q R; the SVD of R gives Σ and V; U = Q U_R.
    scrWt = growF64(scrWt, n * m);
    scrTau = growF64(scrTau, n);
    const Wt = scrWt;
    for (let r = 0; r < m; r++) {
      const rr = r * n;
      for (let c = 0; c < n; c++) Wt[c * m + r] = a[rr + c] * inv;
    }
    householderQRt(Wt, m, n, scrTau);
    for (let c = 0; c < n; c++) {
      const rc = c * m,
        oc = c * n;
      for (let i = 0; i <= c; i++) W[oc + i] = Wt[rc + i];
      for (let i = c + 1; i < n; i++) W[oc + i] = 0;
    }
  } else {
    for (let r = 0; r < n; r++) {
      const rr = r * n;
      for (let c = 0; c < n; c++) W[c * n + r] = a[rr + c] * inv;
    }
  }
  const Vt = scrVt;
  for (let i = 0; i < nn; i++) Vt[i] = 0;
  for (let i = 0; i < n; i++) Vt[i * n + i] = 1;
  const tiny = jacobiOrthogonalize(W, n, n, Vt, scrNrm);

  const sig = scrSig,
    valid = scrValid,
    order = scrOrder;
  for (let i = 0; i < n; i++) {
    let s = 0;
    const oi = i * n;
    for (let k = 0; k < n; k++) s += W[oi + k] * W[oi + k];
    if (s > tiny) {
      sig[i] = Math.sqrt(s);
      valid[i] = 1;
    } else {
      sig[i] = 0;
      valid[i] = 0;
    }
  }
  argsortDesc(sig, n, order);
  const Vd = V.data;
  for (let j = 0; j < n; j++) {
    const oj = order[j];
    S[j] = sig[oj] * scale;
    const vo = oj * n;
    for (let r = 0; r < n; r++) Vd[r * n + j] = Vt[vo + r];
  }
  if (U) {
    // Normalise columns to get U_R (rows of W), then complete the basis.
    for (let i = 0; i < n; i++) {
      const oi = i * n;
      if (valid[i]) {
        const is = 1 / sig[i];
        for (let k = 0; k < n; k++) W[oi + k] *= is;
      } else {
        for (let k = 0; k < n; k++) W[oi + k] = 0;
      }
    }
    completeOrthonormal(W, n, n, valid, scrTmp);
    const Ud = U.data;
    if (qrPre) {
      const u = scrTmp;
      for (let j = 0; j < n; j++) {
        const oj = order[j] * n;
        for (let k = 0; k < n; k++) u[k] = W[oj + k];
        for (let k = n; k < m; k++) u[k] = 0;
        applyQ(scrWt, m, n, scrTau, u);
        for (let r = 0; r < m; r++) Ud[r * n + j] = u[r];
      }
    } else {
      for (let j = 0; j < n; j++) {
        const oj = order[j] * n;
        for (let r = 0; r < n; r++) Ud[r * n + j] = W[oj + r];
      }
    }
  }
  return { U, S, V };
}

/**
 * Thin SVD of any m x n matrix: A = U diag(S) Vᵀ with k = min(m, n), U m x k and
 * V n x k having orthonormal columns, S (length k) non-negative and descending.
 * Rank-deficient / zero input is handled: singular values at the noise floor
 * (≤ ε‖A‖_F) are returned as exactly 0 and the corresponding singular vectors are
 * completed to an orthonormal set. Non-finite input yields NaN outputs.
 * Tall input (m > n) is first reduced by Householder QR (A = QR, SVD of R).
 */
export function svd(A: Matrix): { U: Matrix; S: Float64Array; V: Matrix } {
  const m = A.rows,
    n = A.cols;
  if (m < n) {
    // Aᵀ = U' S V'ᵀ  ⇒  A = V' S U'ᵀ
    const r = svdTall(A.transpose().data, n, m, true);
    return { U: r.V, S: r.S, V: r.U as Matrix };
  }
  const r = svdTall(A.data, m, n, true);
  return { U: r.U as Matrix, S: r.S, V: r.V };
}

/**
 * Orthonormal basis of the right singular vectors belonging to the `dim` smallest
 * singular values of A (m x n, any shape), returned as an n x dim matrix whose
 * column 0 corresponds to the smallest singular value. For m < n the n − m
 * implicit zero singular values are included, so e.g. for a 5 x 9 system
 * `nullSpace(A, 4)` is its 4-D null space.
 *
 * Paths: m ≥ n → SVD (V from accumulated Jacobi rotations). m < n and
 * dim ≤ n − m → Householder QR of Aᵀ = Q R: the last n − m columns of the full Q
 * are orthogonal to every row of A (exactly, whatever the rank), so they span
 * (part of) the null space — backward stable and much cheaper. m < n and
 * dim > n − m → one-sided Jacobi directly on the n columns of A (full n x n V).
 */
export function nullSpace(A: Matrix, dim: number): Matrix {
  const m = A.rows,
    n = A.cols;
  if (!(dim >= 1 && dim <= n)) throw new RangeError(`nullSpace: dim ${dim} not in [1, ${n}]`);
  const out = new Matrix(n, dim);
  const od = out.data;
  if (m >= n) {
    const { V } = svdTall(A.data, m, n, false);
    const Vd = V.data;
    for (let j = 0; j < dim; j++) {
      const src = n - 1 - j;
      for (let r = 0; r < n; r++) od[r * dim + j] = Vd[r * n + src];
    }
    return out;
  }
  const a = A.data;
  const mn = m * n;
  const scale = maxAbs(a, mn);
  if (!Number.isFinite(scale)) {
    od.fill(NaN);
    return out;
  }
  const inv = scale > 0 ? 1 / scale : 1;
  if (dim <= n - m) {
    // Householder QR of Aᵀ (n x m). Its columns are the rows of A, i.e. exactly the
    // transposed layout householderQRt expects: copy A as is.
    scrWt = growF64(scrWt, mn);
    scrTau = growF64(scrTau, m);
    scrTmp = growF64(scrTmp, n);
    const Wt = scrWt,
      u = scrTmp;
    for (let i = 0; i < mn; i++) Wt[i] = a[i] * inv;
    householderQRt(Wt, n, m, scrTau);
    for (let j = 0; j < dim; j++) {
      for (let k = 0; k < n; k++) u[k] = 0;
      u[n - 1 - j] = 1;
      applyQ(Wt, n, m, scrTau, u);
      for (let r = 0; r < n; r++) od[r * dim + j] = u[r];
    }
    return out;
  }
  // Wide, dim > n − m: one-sided Jacobi directly on the n columns (length m) of A.
  const nn = n * n;
  scrVt = growF64(scrVt, nn);
  scrW = growF64(scrW, mn);
  scrNrm = growF64(scrNrm, n);
  scrSig = growF64(scrSig, n);
  if (scrOrder.length < n) scrOrder = new Int32Array(Math.max(n, 2 * scrOrder.length));
  const Vt = scrVt,
    W = scrW,
    key = scrSig,
    order = scrOrder;
  for (let i = 0; i < nn; i++) Vt[i] = 0;
  for (let i = 0; i < n; i++) Vt[i * n + i] = 1;
  for (let r = 0; r < m; r++) {
    const rr = r * n;
    for (let c = 0; c < n; c++) W[c * m + r] = a[rr + c] * inv;
  }
  const tiny = jacobiOrthogonalize(W, n, m, Vt, scrNrm);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const oi = i * m;
    for (let k = 0; k < m; k++) s += W[oi + k] * W[oi + k];
    // Negated so that argsortDesc yields ascending norms; noise-floor norms tie at 0.
    key[i] = s > tiny ? -s : 0;
  }
  argsortDesc(key, n, order);
  for (let j = 0; j < dim; j++) {
    const vo = order[j] * n;
    for (let r = 0; r < n; r++) od[r * dim + j] = Vt[vo + r];
  }
  return out;
}

/**
 * Unit right-singular vector of the smallest singular value: the (least-squares)
 * solution of A x = 0, ‖x‖ = 1. Works for any shape, including m < n (e.g. 8 x 9
 * DLT systems), where it returns a vector of the exact null space. Sign is arbitrary.
 */
export function nullVector(A: Matrix): Float64Array {
  const N = nullSpace(A, 1);
  const v = N.data;
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  if (s > 0 && s !== 1) {
    const inv = 1 / Math.sqrt(s);
    for (let i = 0; i < v.length; i++) v[i] *= inv;
  }
  return v;
}

// ---------------------------------------------------------------------------
// 3x3 SVD
// ---------------------------------------------------------------------------

const S3_W = new Float64Array(9);
const S3_VT = new Float64Array(9);
const S3_SIG = new Float64Array(3);
const S3_NRM = new Float64Array(3);
const S3_VALID = new Uint8Array(3);
const S3_ORDER = new Int32Array(3);
const S3_TMP = new Float64Array(3);

/**
 * SVD of a 3x3 matrix: A = U diag(S) Vᵀ exactly (to rounding), S descending and
 * ≥ 0, U and V orthogonal. NOTE: det(U) and det(V) may each be −1 (they are
 * orthogonal, not necessarily rotations); use `orthonormalize` from rotation.ts
 * or flip the last column (and handle the sign of S[2]) when a proper rotation is
 * required. Same one-sided Jacobi kernel as `svd`, allocation-free apart from the
 * returned arrays.
 */
export function svd3(A: Mat3): { U: Mat3; S: Vec3; V: Mat3 } {
  const U = new Float64Array(9);
  const S = new Float64Array(3);
  const V = new Float64Array(9);
  const scale = maxAbs(A, 9);
  if (!Number.isFinite(scale)) {
    U.fill(NaN);
    S.fill(NaN);
    V.fill(NaN);
    return { U, S, V };
  }
  if (scale === 0) {
    U[0] = U[4] = U[8] = 1;
    V[0] = V[4] = V[8] = 1;
    return { U, S, V };
  }
  const inv = 1 / scale;
  const W = S3_W,
    Vt = S3_VT;
  // Rows of W = columns of A.
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) W[c * 3 + r] = A[r * 3 + c] * inv;
  Vt.fill(0);
  Vt[0] = Vt[4] = Vt[8] = 1;
  const tiny = jacobiOrthogonalize(W, 3, 3, Vt, S3_NRM);
  for (let i = 0; i < 3; i++) {
    const o = 3 * i;
    const s = W[o] * W[o] + W[o + 1] * W[o + 1] + W[o + 2] * W[o + 2];
    if (s > tiny) {
      const sg = Math.sqrt(s);
      S3_SIG[i] = sg;
      S3_VALID[i] = 1;
      const is = 1 / sg;
      W[o] *= is;
      W[o + 1] *= is;
      W[o + 2] *= is;
    } else {
      S3_SIG[i] = 0;
      S3_VALID[i] = 0;
    }
  }
  completeOrthonormal(W, 3, 3, S3_VALID, S3_TMP);
  argsortDesc(S3_SIG, 3, S3_ORDER);
  for (let j = 0; j < 3; j++) {
    const oj = S3_ORDER[j];
    S[j] = S3_SIG[oj] * scale;
    const o = oj * 3;
    U[j] = W[o];
    U[3 + j] = W[o + 1];
    U[6 + j] = W[o + 2];
    V[j] = Vt[o];
    V[3 + j] = Vt[o + 1];
    V[6 + j] = Vt[o + 2];
  }
  return { U, S, V };
}

// ---------------------------------------------------------------------------
// Symmetric eigen-decomposition
// ---------------------------------------------------------------------------

/**
 * Eigen-decomposition of a symmetric n x n row-major matrix by cyclic Jacobi
 * rotations (threshold strategy of Numerical Recipes §11.1; the input is
 * symmetrised as (A + Aᵀ)/2). Eigenvalues are returned ASCENDING; column k of the
 * row-major n x n `vectors` (i.e. vectors[r*n + k]) is the unit eigenvector of
 * values[k]. A is not modified.
 */
export function symEig(A: Float64Array, n: number): { values: Float64Array; vectors: Float64Array } {
  const nn = n * n;
  scrEigA = growF64(scrEigA, nn);
  scrEigV = growF64(scrEigV, nn);
  scrEigD = growF64(scrEigD, 4 * n);
  if (scrOrder.length < n) scrOrder = new Int32Array(Math.max(n, 2 * scrOrder.length));
  const a = scrEigA,
    v = scrEigV;
  // d, b, z, negD are views into one scratch block.
  const d = scrEigD.subarray(0, n),
    b = scrEigD.subarray(n, 2 * n),
    z = scrEigD.subarray(2 * n, 3 * n),
    negD = scrEigD.subarray(3 * n, 4 * n);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const x = i === j ? A[i * n + i] : 0.5 * (A[i * n + j] + A[j * n + i]);
      a[i * n + j] = x;
      a[j * n + i] = x;
    }
  }
  for (let i = 0; i < nn; i++) v[i] = 0;
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let i = 0; i < n; i++) {
    d[i] = b[i] = a[i * n + i];
    z[i] = 0;
  }

  for (let sweep = 0; sweep < 100; sweep++) {
    let sm = 0;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) sm += Math.abs(a[p * n + q]);
    if (sm === 0 || !Number.isFinite(sm)) break;
    const tresh = sweep < 3 ? (0.2 * sm) / (n * n) : 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        const g = 100 * Math.abs(apq);
        if (sweep > 3 && Math.abs(d[p]) + g === Math.abs(d[p]) && Math.abs(d[q]) + g === Math.abs(d[q])) {
          a[p * n + q] = 0;
          continue;
        }
        if (!(Math.abs(apq) > tresh)) continue;
        let h = d[q] - d[p];
        let t: number;
        if (Math.abs(h) + g === Math.abs(h)) {
          t = apq / h;
        } else {
          const theta = (0.5 * h) / apq;
          t = 1 / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
          if (theta < 0) t = -t;
        }
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;
        const tau = s / (1 + c);
        h = t * apq;
        z[p] -= h;
        z[q] += h;
        d[p] -= h;
        d[q] += h;
        a[p * n + q] = 0;
        // Rotate the upper triangle (rows/cols p and q).
        for (let j = 0; j < p; j++) {
          const g1 = a[j * n + p],
            h1 = a[j * n + q];
          a[j * n + p] = g1 - s * (h1 + g1 * tau);
          a[j * n + q] = h1 + s * (g1 - h1 * tau);
        }
        for (let j = p + 1; j < q; j++) {
          const g1 = a[p * n + j],
            h1 = a[j * n + q];
          a[p * n + j] = g1 - s * (h1 + g1 * tau);
          a[j * n + q] = h1 + s * (g1 - h1 * tau);
        }
        for (let j = q + 1; j < n; j++) {
          const g1 = a[p * n + j],
            h1 = a[q * n + j];
          a[p * n + j] = g1 - s * (h1 + g1 * tau);
          a[q * n + j] = h1 + s * (g1 - h1 * tau);
        }
        for (let j = 0; j < n; j++) {
          const g1 = v[j * n + p],
            h1 = v[j * n + q];
          v[j * n + p] = g1 - s * (h1 + g1 * tau);
          v[j * n + q] = h1 + s * (g1 - h1 * tau);
        }
      }
    }
    for (let p = 0; p < n; p++) {
      b[p] += z[p];
      d[p] = b[p];
      z[p] = 0;
    }
  }
  // Sort ascending.
  const order = scrOrder;
  for (let i = 0; i < n; i++) negD[i] = -d[i];
  argsortDesc(negD, n, order);
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  for (let k = 0; k < n; k++) {
    const src = order[k];
    values[k] = d[src];
    for (let r = 0; r < n; r++) vectors[r * n + k] = v[r * n + src];
  }
  return { values, vectors };
}
