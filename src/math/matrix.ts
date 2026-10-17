/**
 * General dense row-major matrix (Float64). Element (i, j) is `data[i * cols + j]`.
 * Intended for small/medium problems (DLT systems, normal equations, Jacobians).
 */
export class Matrix {
  rows: number;
  cols: number;
  data: Float64Array;

  /**
   * @param data optional initial contents (row-major, length rows*cols). A
   *   `Float64Array` is wrapped WITHOUT copying; any other array-like is copied.
   */
  constructor(rows: number, cols: number, data?: Float64Array | ArrayLike<number>) {
    if (!(rows >= 0 && cols >= 0) || !Number.isInteger(rows) || !Number.isInteger(cols)) {
      throw new RangeError(`Matrix: invalid shape ${rows}x${cols}`);
    }
    this.rows = rows;
    this.cols = cols;
    if (data === undefined) {
      this.data = new Float64Array(rows * cols);
    } else {
      if (data.length !== rows * cols) {
        throw new RangeError(`Matrix: data length ${data.length} != ${rows}x${cols}`);
      }
      this.data = data instanceof Float64Array ? data : Float64Array.from(data);
    }
  }

  /** n x n identity. */
  static identity(n: number): Matrix {
    const m = new Matrix(n, n);
    for (let i = 0; i < n; i++) m.data[i * n + i] = 1;
    return m;
  }

  /** Zero matrix. */
  static zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols);
  }

  /** From an array of equal-length rows. */
  static fromRows(rows: number[][]): Matrix {
    const r = rows.length;
    const c = r > 0 ? rows[0].length : 0;
    const m = new Matrix(r, c);
    for (let i = 0; i < r; i++) {
      const row = rows[i];
      if (row.length !== c) throw new RangeError('Matrix.fromRows: ragged rows');
      for (let j = 0; j < c; j++) m.data[i * c + j] = row[j];
    }
    return m;
  }

  get(i: number, j: number): number {
    return this.data[i * this.cols + j];
  }

  set(i: number, j: number, v: number): void {
    this.data[i * this.cols + j] = v;
  }

  /** Deep copy. */
  clone(): Matrix {
    return new Matrix(this.rows, this.cols, this.data.slice());
  }

  /** New transposed matrix. */
  transpose(): Matrix {
    const r = this.rows,
      c = this.cols;
    const out = new Matrix(c, r);
    const a = this.data,
      o = out.data;
    for (let i = 0; i < r; i++) {
      const ri = i * c;
      for (let j = 0; j < c; j++) o[j * r + i] = a[ri + j];
    }
    return out;
  }

  /** this * b */
  mul(b: Matrix): Matrix {
    if (this.cols !== b.rows) throw new RangeError(`Matrix.mul: ${this.rows}x${this.cols} * ${b.rows}x${b.cols}`);
    const m = this.rows,
      k = this.cols,
      n = b.cols;
    const out = new Matrix(m, n);
    const A = this.data,
      B = b.data,
      C = out.data;
    for (let i = 0; i < m; i++) {
      const ci = i * n;
      const ai = i * k;
      for (let p = 0; p < k; p++) {
        const aip = A[ai + p];
        if (aip === 0) continue;
        const bp = p * n;
        for (let j = 0; j < n; j++) C[ci + j] += aip * B[bp + j];
      }
    }
    return out;
  }

  /** thisᵀ * b (without forming the transpose). */
  transposeMul(b: Matrix): Matrix {
    if (this.rows !== b.rows) throw new RangeError('Matrix.transposeMul: row mismatch');
    const m = this.rows,
      k = this.cols,
      n = b.cols;
    const out = new Matrix(k, n);
    const A = this.data,
      B = b.data,
      C = out.data;
    for (let r = 0; r < m; r++) {
      const ar = r * k,
        br = r * n;
      for (let i = 0; i < k; i++) {
        const a = A[ar + i];
        if (a === 0) continue;
        const ci = i * n;
        for (let j = 0; j < n; j++) C[ci + j] += a * B[br + j];
      }
    }
    return out;
  }

  /** this * x (x has length cols). */
  mulVec(x: ArrayLike<number>): Float64Array {
    if (x.length < this.cols) throw new RangeError('Matrix.mulVec: vector too short');
    const r = this.rows,
      c = this.cols,
      a = this.data;
    const out = new Float64Array(r);
    for (let i = 0; i < r; i++) {
      let s = 0;
      const ri = i * c;
      for (let j = 0; j < c; j++) s += a[ri + j] * x[j];
      out[i] = s;
    }
    return out;
  }

  /** thisᵀ * x (x has length rows). */
  transposeMulVec(x: ArrayLike<number>): Float64Array {
    const r = this.rows,
      c = this.cols,
      a = this.data;
    const out = new Float64Array(c);
    for (let i = 0; i < r; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const ri = i * c;
      for (let j = 0; j < c; j++) out[j] += a[ri + j] * xi;
    }
    return out;
  }

  /** Frobenius norm. */
  frobeniusNorm(): number {
    let s = 0;
    const a = this.data;
    for (let i = 0; i < a.length; i++) s += a[i] * a[i];
    return Math.sqrt(s);
  }

  /** Copy of row i. */
  getRow(i: number): Float64Array {
    return this.data.slice(i * this.cols, (i + 1) * this.cols);
  }

  /** Copy of column j. */
  getColumn(j: number): Float64Array {
    const out = new Float64Array(this.rows);
    for (let i = 0; i < this.rows; i++) out[i] = this.data[i * this.cols + j];
    return out;
  }
}
