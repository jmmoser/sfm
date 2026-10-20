/**
 * Linear algebra for the reduced camera system of bundle adjustment:
 *  - `BlockSparseSymmetric`: symmetric block-sparse matrix storing the upper block
 *    triangle (incl. full square diagonal blocks), variable block sizes.
 *  - Block-Jacobi preconditioned conjugate gradients (Hestenes & Stiefel 1952;
 *    preconditioner = inverse of the diagonal blocks of the Schur complement, the
 *    "Schur–Jacobi" preconditioner of Agarwal et al., "Bundle Adjustment in the
 *    Large", ECCV 2010).
 *  - Skyline (envelope / profile) Cholesky in packed storage: identical result to a
 *    dense Cholesky, but memory Σ_i w_i and work Σ_i w_i²/2 (w_i = row profile width)
 *    instead of n²/2 and n³/6, because Cholesky fill-in stays inside the row envelope
 *    (George & Liu, "Computer Solution of Large Sparse Positive Definite Systems", 1981,
 *    ch. 4). Sequential captures give banded reduced camera matrices (intrinsics blocks
 *    are ordered last so their dense rows do not widen the profile).
 */

/** Symmetric block-sparse matrix (upper block triangle stored, rows in CSR). */
export class BlockSparseSymmetric {
  readonly numBlocks: number;
  /** Scalar dimension (Σ block sizes). */
  readonly dim: number;
  readonly blockSize: Int32Array;
  /** First scalar index of each block. */
  readonly blockStart: Int32Array;
  /** CSR over block rows: entries rowStart[a] .. rowStart[a+1]-1, columns ascending, col ≥ a. */
  readonly rowStart: Int32Array;
  readonly col: Int32Array;
  /** Offset of each entry's dense (size_a x size_b, row-major) block in `values`. */
  readonly valOff: Int32Array;
  /** Entry index of the diagonal block of each row. */
  readonly diagEntry: Int32Array;
  values: Float64Array;

  constructor(blockSize: Int32Array, rowStart: Int32Array, col: Int32Array) {
    const nB = blockSize.length;
    this.numBlocks = nB;
    this.blockSize = blockSize;
    this.rowStart = rowStart;
    this.col = col;
    this.blockStart = new Int32Array(nB + 1);
    for (let b = 0; b < nB; b++) this.blockStart[b + 1] = this.blockStart[b] + blockSize[b];
    this.dim = this.blockStart[nB];
    const nE = rowStart[nB];
    this.valOff = new Int32Array(nE + 1);
    this.diagEntry = new Int32Array(nB).fill(-1);
    let off = 0;
    for (let a = 0; a < nB; a++) {
      const sa = blockSize[a];
      for (let e = rowStart[a]; e < rowStart[a + 1]; e++) {
        const b = col[e];
        if (b < a) throw new RangeError('BlockSparseSymmetric: entries must be in the upper block triangle');
        if (e > rowStart[a] && b <= col[e - 1]) throw new RangeError('BlockSparseSymmetric: columns must ascend');
        if (b === a) this.diagEntry[a] = e;
        this.valOff[e] = off;
        off += sa * blockSize[b];
      }
      if (this.diagEntry[a] < 0) throw new RangeError('BlockSparseSymmetric: every row needs its diagonal block');
    }
    this.valOff[nE] = off;
    this.values = new Float64Array(off);
  }

  /** Number of stored block entries. */
  get numEntries(): number {
    return this.rowStart[this.numBlocks];
  }

  /** y = A x for the symmetric matrix with block values `vals` (defaults to `values`). */
  multiply(x: Float64Array, y: Float64Array, vals: Float64Array = this.values): void {
    y.fill(0, 0, this.dim);
    const nB = this.numBlocks;
    const bs = this.blockSize,
      st = this.blockStart,
      rs = this.rowStart,
      col = this.col,
      vo = this.valOff;
    for (let a = 0; a < nB; a++) {
      const sa = bs[a],
        oa = st[a];
      for (let e = rs[a]; e < rs[a + 1]; e++) {
        const b = col[e];
        const sb = bs[b],
          ob = st[b];
        let o = vo[e];
        if (b === a) {
          for (let r = 0; r < sa; r++, o += sb) {
            let s = 0;
            for (let c = 0; c < sb; c++) s += vals[o + c] * x[ob + c];
            y[oa + r] += s;
          }
        } else {
          for (let r = 0; r < sa; r++, o += sb) {
            let s = 0;
            const xr = x[oa + r];
            for (let c = 0; c < sb; c++) {
              const v = vals[o + c];
              s += v * x[ob + c];
              y[ob + c] += v * xr;
            }
            y[oa + r] += s;
          }
        }
      }
    }
  }

  /** xᵀ A x (A symmetric with block values `vals`). */
  quadraticForm(x: Float64Array, vals: Float64Array = this.values): number {
    const nB = this.numBlocks;
    const bs = this.blockSize,
      st = this.blockStart,
      rs = this.rowStart,
      col = this.col,
      vo = this.valOff;
    let total = 0;
    for (let a = 0; a < nB; a++) {
      const sa = bs[a],
        oa = st[a];
      for (let e = rs[a]; e < rs[a + 1]; e++) {
        const b = col[e];
        const sb = bs[b],
          ob = st[b];
        let o = vo[e];
        let s = 0;
        for (let r = 0; r < sa; r++, o += sb) {
          let rowDot = 0;
          for (let c = 0; c < sb; c++) rowDot += vals[o + c] * x[ob + c];
          s += x[oa + r] * rowDot;
        }
        total += b === a ? s : 2 * s;
      }
    }
    return total;
  }

  /**
   * Scatter into the LOWER triangle of a dense n x n row-major matrix (zeroed first),
   * mapping scalar index g → map[g] (−1 = dropped). `map` must be increasing over
   * kept indices. Also returns, per dense row, the first non-zero column in `first`
   * (diagnostics / tests; the solver uses `scatterToSkyline`).
   */
  toDenseLower(map: Int32Array, n: number, out: Float64Array, first: Int32Array): void {
    out.fill(0, 0, n * n);
    for (let i = 0; i < n; i++) first[i] = i;
    const nB = this.numBlocks;
    const bs = this.blockSize,
      st = this.blockStart,
      rs = this.rowStart,
      col = this.col,
      vo = this.valOff,
      vals = this.values;
    for (let a = 0; a < nB; a++) {
      const sa = bs[a],
        oa = st[a];
      for (let e = rs[a]; e < rs[a + 1]; e++) {
        const b = col[e];
        const sb = bs[b],
          ob = st[b];
        const o = vo[e];
        for (let r = 0; r < sa; r++) {
          const gi = map[oa + r];
          if (gi < 0) continue;
          for (let c = 0; c < sb; c++) {
            const gj = map[ob + c];
            if (gj < 0) continue;
            const v = vals[o + r * sb + c];
            if (gj >= gi) {
              // (gi, gj) upper → store at lower (gj, gi)
              out[gj * n + gi] = v;
              if (gi < first[gj]) first[gj] = gi;
            } else {
              out[gi * n + gj] = v;
              if (gj < first[gi]) first[gi] = gj;
            }
          }
        }
      }
    }
  }
}

/**
 * Packed skyline (profile) storage of the lower triangle of a symmetric n x n matrix
 * with in-place Cholesky. Row i stores columns first[i] … i contiguously; Cholesky
 * fill-in stays inside this envelope, so the factor needs no extra memory and costs
 * Σ_i (i − first[i])² / 2 multiply-adds (see the module header).
 */
export class SkylineCholesky {
  readonly n: number;
  readonly first: Int32Array;
  /** Element (i, j), first[i] ≤ j ≤ i, is values[rowBase[i] + j]. */
  readonly rowBase: Int32Array;
  readonly values: Float64Array;
  /** Stored entries (memory = 8 bytes each). */
  readonly size: number;

  constructor(first: Int32Array) {
    const n = (this.n = first.length);
    this.first = first;
    this.rowBase = new Int32Array(n);
    let off = 0;
    for (let i = 0; i < n; i++) {
      if (!(first[i] >= 0 && first[i] <= i)) throw new RangeError('SkylineCholesky: need 0 ≤ first[i] ≤ i');
      this.rowBase[i] = off - first[i];
      off += i - first[i] + 1;
      if (off > 0x7fffffff) throw new RangeError('SkylineCholesky: envelope too large');
    }
    this.size = off;
    this.values = new Float64Array(off);
  }

  /** Estimated factorisation work (multiply-adds) for a given envelope. */
  static factorCost(first: Int32Array): number {
    let c = 0;
    for (let i = 0; i < first.length; i++) {
      const w = i - first[i];
      c += 0.5 * w * w + w;
    }
    return c;
  }

  /** Stored entries for a given envelope. */
  static storageSize(first: Int32Array): number {
    let s = 0;
    for (let i = 0; i < first.length; i++) s += i - first[i] + 1;
    return s;
  }

  /**
   * In-place Cholesky of the stored matrix. Returns false if a pivot is ≤ 1e-15 ·
   * (original diagonal) or not finite (matrix not numerically positive definite).
   *
   * Row-oriented (L[i][j] = (A[i][j] − Σₖ L[i][k] L[j][k]) / L[j][j]), two rows × two columns at a
   * time where the envelopes allow it (rows / columns of one BA parameter block share their first
   * column): four dot products from four loads per step instead of two loads per multiply-add,
   * which bounds the plain dot product (≈ 1.7× faster for a dense 541 × 541 system, 2.4× at 1200).
   */
  factor(): boolean {
    const n = this.n,
      first = this.first,
      base = this.rowBase,
      v = this.values;
    let i = 0;
    while (i < n) {
      const fi = first[i];
      const ri = base[i];
      if (i + 1 < n && first[i + 1] === fi) {
        // rows i and i1 = i + 1
        const r1 = base[i + 1];
        let j = fi;
        for (; j + 1 < i; j += 2) {
          const fj = first[j];
          const rj = base[j];
          if (first[j + 1] !== fj) {
            rowPairColumn(v, fi, ri, r1, j, first[j], rj);
            rowPairColumn(v, fi, ri, r1, j + 1, first[j + 1], base[j + 1]);
            continue;
          }
          const rj1 = base[j + 1];
          let s00 = 0,
            s01 = 0,
            s10 = 0,
            s11 = 0;
          for (let k = fi > fj ? fi : fj; k < j; k++) {
            const x0 = v[ri + k],
              x1 = v[r1 + k],
              y0 = v[rj + k],
              y1 = v[rj1 + k];
            s00 += x0 * y0;
            s01 += x0 * y1;
            s10 += x1 * y0;
            s11 += x1 * y1;
          }
          const djj = v[rj + j];
          const l0 = (v[ri + j] - s00) / djj,
            l1 = (v[r1 + j] - s10) / djj;
          v[ri + j] = l0;
          v[r1 + j] = l1;
          // column j + 1 also has the k = j term
          const c = v[rj1 + j],
            dj1 = v[rj1 + j + 1];
          v[ri + j + 1] = (v[ri + j + 1] - (s01 + l0 * c)) / dj1;
          v[r1 + j + 1] = (v[r1 + j + 1] - (s11 + l1 * c)) / dj1;
        }
        for (; j < i; j++) rowPairColumn(v, fi, ri, r1, j, first[j], base[j]);
        // pivot of row i, then L[i1][i], then the pivot of row i1
        let sd = 0,
          sx = 0;
        for (let k = fi; k < i; k++) {
          const x = v[ri + k];
          sd += x * x;
          sx += v[r1 + k] * x;
        }
        const aii = v[ri + i];
        const dii = aii - sd;
        if (!(dii > 1e-15 * Math.abs(aii)) || dii === Infinity) return false;
        const lii = Math.sqrt(dii);
        v[ri + i] = lii;
        const l1i = (v[r1 + i] - sx) / lii;
        v[r1 + i] = l1i;
        let s2 = l1i * l1i;
        for (let k = fi; k < i; k++) {
          const x = v[r1 + k];
          s2 += x * x;
        }
        const a11 = v[r1 + i + 1];
        const d11 = a11 - s2;
        if (!(d11 > 1e-15 * Math.abs(a11)) || d11 === Infinity) return false;
        v[r1 + i + 1] = Math.sqrt(d11);
        i += 2;
        continue;
      }
      for (let j = fi; j < i; j++) {
        const fj = first[j];
        const rj = base[j];
        let s = 0;
        for (let k = fi > fj ? fi : fj; k < j; k++) s += v[ri + k] * v[rj + k];
        v[ri + j] = (v[ri + j] - s) / v[rj + j];
      }
      const aii = v[ri + i];
      let s = 0;
      for (let k = fi; k < i; k++) {
        const l = v[ri + k];
        s += l * l;
      }
      const d = aii - s;
      if (!(d > 1e-15 * Math.abs(aii)) || d === Infinity) return false;
      v[ri + i] = Math.sqrt(d);
      i++;
    }
    return true;
  }

  /** Solve L Lᵀ x = b in place (b ← x) after `factor`. */
  solve(b: Float64Array): void {
    const n = this.n,
      first = this.first,
      base = this.rowBase,
      v = this.values;
    for (let i = 0; i < n; i++) {
      const ri = base[i];
      let s = b[i];
      for (let k = first[i]; k < i; k++) s -= v[ri + k] * b[k];
      b[i] = s / v[ri + i];
    }
    for (let i = n - 1; i >= 0; i--) {
      const ri = base[i];
      const xi = b[i] / v[ri + i];
      b[i] = xi;
      if (xi !== 0) for (let k = first[i]; k < i; k++) b[k] -= v[ri + k] * xi;
    }
  }
}

/** L[i][j] and L[i1][j] of the row pair (i, i1) sharing the first column `fi` (`SkylineCholesky.factor`). */
function rowPairColumn(v: Float64Array, fi: number, ri: number, r1: number, j: number, fj: number, rj: number): void {
  let s0 = 0,
    s1 = 0;
  for (let k = fi > fj ? fi : fj; k < j; k++) {
    const y = v[rj + k];
    s0 += v[ri + k] * y;
    s1 += v[r1 + k] * y;
  }
  const d = v[rj + j];
  v[ri + j] = (v[ri + j] - s0) / d;
  v[r1 + j] = (v[r1 + j] - s1) / d;
}

/**
 * Row envelope (first stored column per compact row) of the matrix that
 * `scatterToSkyline` produces from `A` with the index map `map` (−1 = dropped,
 * increasing over kept indices). `n` = number of kept indices.
 */
export function skylineEnvelope(A: BlockSparseSymmetric, map: Int32Array, n: number): Int32Array {
  const nB = A.numBlocks;
  const minBlock = new Int32Array(nB);
  for (let b = 0; b < nB; b++) minBlock[b] = b;
  for (let a = 0; a < nB; a++)
    for (let e = A.rowStart[a]; e < A.rowStart[a + 1]; e++) {
      const b = A.col[e];
      if (a < minBlock[b]) minBlock[b] = a;
    }
  // first kept scalar of each block (or −1)
  const firstKept = new Int32Array(nB).fill(-1);
  for (let b = 0; b < nB; b++)
    for (let s = A.blockStart[b]; s < A.blockStart[b + 1]; s++)
      if (map[s] >= 0) {
        firstKept[b] = map[s];
        break;
      }
  const first = new Int32Array(n);
  for (let b = 0; b < nB; b++) {
    // the smallest block row with a kept scalar that couples to b (blocks between are ≥ minBlock)
    let a = minBlock[b];
    while (firstKept[a] < 0 && a < b) a++;
    const f = firstKept[a];
    for (let s = A.blockStart[b]; s < A.blockStart[b + 1]; s++) {
      const g = map[s];
      if (g >= 0) first[g] = f >= 0 && f <= g ? f : g;
    }
  }
  return first;
}

/** Scatter the lower triangle of `A` (index map `map`) into `sky` (zeroed first). */
export function scatterToSkyline(A: BlockSparseSymmetric, map: Int32Array, sky: SkylineCholesky): void {
  const v = sky.values,
    base = sky.rowBase;
  v.fill(0);
  const nB = A.numBlocks;
  const bs = A.blockSize,
    st = A.blockStart,
    rs = A.rowStart,
    col = A.col,
    vo = A.valOff,
    vals = A.values;
  for (let a = 0; a < nB; a++) {
    const sa = bs[a],
      oa = st[a];
    for (let e = rs[a]; e < rs[a + 1]; e++) {
      const b = col[e];
      const sb = bs[b],
        ob = st[b];
      const o = vo[e];
      for (let r = 0; r < sa; r++) {
        const gi = map[oa + r];
        if (gi < 0) continue;
        for (let c = 0; c < sb; c++) {
          const gj = map[ob + c];
          if (gj < 0) continue;
          if (gj >= gi) v[base[gj] + gi] = vals[o + r * sb + c];
          else if (b === a) v[base[gi] + gj] = vals[o + r * sb + c];
        }
      }
    }
  }
}

/** Block-Jacobi preconditioner: per diagonal block a Cholesky factor (or scalar Jacobi fallback). */
export class BlockJacobiPreconditioner {
  private readonly A: BlockSparseSymmetric;
  /** Factor storage: full sa x sa per diagonal block, at facOff[a] (only the diagonal blocks). */
  private readonly fac: Float64Array;
  private readonly facOff: Int32Array;
  /** 1 = block uses scalar Jacobi fallback. */
  private readonly scalar: Uint8Array;

  constructor(A: BlockSparseSymmetric) {
    this.A = A;
    const nB = A.numBlocks;
    this.facOff = new Int32Array(nB + 1);
    for (let a = 0; a < nB; a++) this.facOff[a + 1] = this.facOff[a] + A.blockSize[a] * A.blockSize[a];
    this.fac = new Float64Array(this.facOff[nB]);
    this.scalar = new Uint8Array(nB);
  }

  /** Factor the diagonal blocks of the matrix's current values. */
  update(): void {
    const A = this.A;
    const vals = A.values,
      fac = this.fac;
    for (let a = 0; a < A.numBlocks; a++) {
      const n = A.blockSize[a];
      const vo = A.valOff[A.diagEntry[a]];
      const o = this.facOff[a];
      for (let k = 0; k < n * n; k++) fac[o + k] = vals[vo + k];
      let ok = true;
      for (let i = 0; i < n && ok; i++) {
        const ri = o + i * n;
        for (let j = 0; j < i; j++) {
          const rj = o + j * n;
          let s = fac[ri + j];
          for (let k = 0; k < j; k++) s -= fac[ri + k] * fac[rj + k];
          fac[ri + j] = s / fac[rj + j];
        }
        const aii = fac[ri + i];
        let s = aii;
        for (let k = 0; k < i; k++) s -= fac[ri + k] * fac[ri + k];
        if (!(s > 1e-14 * Math.abs(aii)) || s === Infinity) ok = false;
        else fac[ri + i] = Math.sqrt(s);
      }
      if (ok) {
        this.scalar[a] = 0;
      } else {
        this.scalar[a] = 1;
        for (let i = 0; i < n; i++) {
          const d = vals[vo + i * n + i];
          fac[o + i * n + i] = d > 0 && d < Infinity ? 1 / d : 1;
        }
      }
    }
  }

  /** z = M⁻¹ r. */
  apply(r: Float64Array, z: Float64Array): void {
    const A = this.A;
    const fac = this.fac;
    for (let a = 0; a < A.numBlocks; a++) {
      const n = A.blockSize[a];
      const o = this.facOff[a];
      const s0 = A.blockStart[a];
      if (this.scalar[a]) {
        for (let i = 0; i < n; i++) z[s0 + i] = r[s0 + i] * fac[o + i * n + i];
        continue;
      }
      for (let i = 0; i < n; i++) {
        let s = r[s0 + i];
        for (let k = 0; k < i; k++) s -= fac[o + i * n + k] * z[s0 + k];
        z[s0 + i] = s / fac[o + i * n + i];
      }
      for (let i = n - 1; i >= 0; i--) {
        let s = z[s0 + i];
        for (let k = i + 1; k < n; k++) s -= fac[o + k * n + i] * z[s0 + k];
        z[s0 + i] = s / fac[o + i * n + i];
      }
    }
  }
}

/** Scratch vectors for `pcgSolve` (length ≥ dim). */
export interface PcgWorkspace {
  r: Float64Array;
  z: Float64Array;
  p: Float64Array;
  q: Float64Array;
}

export function createPcgWorkspace(dim: number): PcgWorkspace {
  return { r: new Float64Array(dim), z: new Float64Array(dim), p: new Float64Array(dim), q: new Float64Array(dim) };
}

/**
 * Preconditioned CG for A x = b (A SPD block-sparse), starting from x = 0.
 * Stops when ‖r‖ ≤ tol·‖b‖ or after maxIter iterations; returns the iteration count
 * (−1 if a non-finite value or a non-positive curvature pᵀAp ≤ 0 was met at the
 * first iteration, i.e. no usable solution).
 */
export function pcgSolve(
  A: BlockSparseSymmetric,
  M: BlockJacobiPreconditioner,
  b: Float64Array,
  x: Float64Array,
  maxIter: number,
  tol: number,
  ws: PcgWorkspace,
): number {
  const n = A.dim;
  const { r, z, p, q } = ws;
  let bb = 0;
  for (let i = 0; i < n; i++) {
    x[i] = 0;
    r[i] = b[i];
    bb += b[i] * b[i];
  }
  if (bb === 0) return 0;
  if (!(bb < Infinity)) return -1;
  const stop2 = tol * tol * bb;
  M.apply(r, z);
  let rz = 0;
  for (let i = 0; i < n; i++) {
    p[i] = z[i];
    rz += r[i] * z[i];
  }
  let it = 0;
  while (it < maxIter) {
    A.multiply(p, q);
    let pq = 0;
    for (let i = 0; i < n; i++) pq += p[i] * q[i];
    if (!(pq > 0) || !(rz > 0) || pq === Infinity) return it === 0 ? -1 : it;
    const alpha = rz / pq;
    let rr = 0;
    for (let i = 0; i < n; i++) {
      x[i] += alpha * p[i];
      const ri = r[i] - alpha * q[i];
      r[i] = ri;
      rr += ri * ri;
    }
    it++;
    if (rr <= stop2) break;
    M.apply(r, z);
    let rzNew = 0;
    for (let i = 0; i < n; i++) rzNew += r[i] * z[i];
    const beta = rzNew / rz;
    rz = rzNew;
    for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
  }
  return it;
}
