/**
 * Eigenvalues of small real NONSYMMETRIC matrices (balancing + Householder-free
 * elimination to Hessenberg form + Francis double-shift QR), and eigenvectors of
 * real eigenvalues by inverse iteration. Classic EISPACK algorithms in the form of
 * Press et al., "Numerical Recipes" (2nd ed.) §11.5–11.6 (balanc, elmhes, hqr) and
 * §11.7 (inverse iteration). Used by the 5-point action-matrix solver.
 */

const EPS = 2.220446049250313e-16;

let scrH = new Float64Array(0);
let scrLU = new Float64Array(0);
let scrPiv = new Int32Array(0);
let scrV = new Float64Array(0);

/**
 * All eigenvalues (real parts wr, imaginary parts wi) of the n×n row-major matrix A
 * (not modified). Returns false if the QR iteration did not converge (wr/wi then
 * hold whatever was found so far, NaN elsewhere).
 */
export function eigenvaluesNonsymmetric(A: ArrayLike<number>, n: number, wr: Float64Array, wi: Float64Array): boolean {
  const m = n + 1; // 1-based storage a[i*m + j], i, j ∈ [1, n]
  if (scrH.length < m * m) scrH = new Float64Array(m * m);
  const a = scrH;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) a[(i + 1) * m + j + 1] = A[i * n + j];
  for (let i = 0; i < n; i++) {
    wr[i] = NaN;
    wi[i] = NaN;
  }
  balance(a, n, m);
  hessenberg(a, n, m);
  return hqr(a, n, m, wr, wi);
}

/** Balancing (Parlett & Reinsch 1969): similarity by powers of 2 equalising row/column norms. */
function balance(a: Float64Array, n: number, m: number): void {
  const RADIX = 2,
    sqrdx = 4;
  let last = false;
  let guard = 0;
  while (!last && guard++ < 100) {
    last = true;
    for (let i = 1; i <= n; i++) {
      let r = 0,
        c = 0;
      for (let j = 1; j <= n; j++) {
        if (j !== i) {
          c += Math.abs(a[j * m + i]);
          r += Math.abs(a[i * m + j]);
        }
      }
      if (c !== 0 && r !== 0) {
        let g = r / RADIX;
        let f = 1;
        const s = c + r;
        while (c < g) {
          f *= RADIX;
          c *= sqrdx;
        }
        g = r * RADIX;
        while (c > g) {
          f /= RADIX;
          c /= sqrdx;
        }
        if ((c + r) / f < 0.95 * s) {
          last = false;
          g = 1 / f;
          for (let j = 1; j <= n; j++) a[i * m + j] *= g;
          for (let j = 1; j <= n; j++) a[j * m + i] *= f;
        }
      }
    }
  }
}

/** Reduction to upper Hessenberg form by stabilised elementary similarity transforms. */
function hessenberg(a: Float64Array, n: number, mm: number): void {
  for (let m = 2; m < n; m++) {
    let x = 0;
    let i = m;
    for (let j = m; j <= n; j++) {
      if (Math.abs(a[j * mm + m - 1]) > Math.abs(x)) {
        x = a[j * mm + m - 1];
        i = j;
      }
    }
    if (i !== m) {
      for (let j = m - 1; j <= n; j++) {
        const t = a[i * mm + j];
        a[i * mm + j] = a[m * mm + j];
        a[m * mm + j] = t;
      }
      for (let j = 1; j <= n; j++) {
        const t = a[j * mm + i];
        a[j * mm + i] = a[j * mm + m];
        a[j * mm + m] = t;
      }
    }
    if (x !== 0) {
      for (i = m + 1; i <= n; i++) {
        let y = a[i * mm + m - 1];
        if (y !== 0) {
          y /= x;
          a[i * mm + m - 1] = y;
          for (let j = m; j <= n; j++) a[i * mm + j] -= y * a[m * mm + j];
          for (let j = 1; j <= n; j++) a[j * mm + m] += y * a[j * mm + i];
        }
      }
    }
  }
  // Clear the stored multipliers below the subdiagonal.
  for (let i = 3; i <= n; i++) for (let j = 1; j < i - 1; j++) a[i * mm + j] = 0;
}

function sign(a: number, b: number): number {
  return b >= 0 ? Math.abs(a) : -Math.abs(a);
}

/** Francis double-shift QR on an upper Hessenberg matrix (eigenvalues only). */
function hqr(a: Float64Array, n: number, mm: number, wr: Float64Array, wi: Float64Array): boolean {
  let anorm = 0;
  for (let i = 1; i <= n; i++) for (let j = Math.max(i - 1, 1); j <= n; j++) anorm += Math.abs(a[i * mm + j]);
  let nn = n;
  let t = 0;
  let p = 0,
    q = 0,
    r = 0,
    s = 0,
    w = 0,
    x = 0,
    y = 0,
    z = 0;
  while (nn >= 1) {
    let its = 0;
    let l: number;
    do {
      for (l = nn; l >= 2; l--) {
        s = Math.abs(a[(l - 1) * mm + l - 1]) + Math.abs(a[l * mm + l]);
        if (s === 0) s = anorm;
        if (Math.abs(a[l * mm + l - 1]) <= EPS * s) {
          a[l * mm + l - 1] = 0;
          break;
        }
      }
      x = a[nn * mm + nn];
      if (l === nn) {
        wr[nn - 1] = x + t;
        wi[nn - 1] = 0;
        nn--;
      } else {
        y = a[(nn - 1) * mm + nn - 1];
        w = a[nn * mm + nn - 1] * a[(nn - 1) * mm + nn];
        if (l === nn - 1) {
          p = 0.5 * (y - x);
          q = p * p + w;
          z = Math.sqrt(Math.abs(q));
          x += t;
          if (q >= 0) {
            z = p + sign(z, p);
            wr[nn - 2] = wr[nn - 1] = x + z;
            if (z !== 0) wr[nn - 1] = x - w / z;
            wi[nn - 2] = wi[nn - 1] = 0;
          } else {
            wr[nn - 2] = wr[nn - 1] = x + p;
            wi[nn - 2] = -z;
            wi[nn - 1] = z;
          }
          nn -= 2;
        } else {
          if (its === 60) return false;
          if (its === 10 || its === 20 || its === 40) {
            // Exceptional shift.
            t += x;
            for (let i = 1; i <= nn; i++) a[i * mm + i] -= x;
            s = Math.abs(a[nn * mm + nn - 1]) + Math.abs(a[(nn - 1) * mm + nn - 2]);
            y = x = 0.75 * s;
            w = -0.4375 * s * s;
          }
          ++its;
          let m: number;
          for (m = nn - 2; m >= l; m--) {
            z = a[m * mm + m];
            r = x - z;
            s = y - z;
            p = (r * s - w) / a[(m + 1) * mm + m] + a[m * mm + m + 1];
            q = a[(m + 1) * mm + m + 1] - z - r - s;
            r = a[(m + 2) * mm + m + 1];
            s = Math.abs(p) + Math.abs(q) + Math.abs(r);
            p /= s;
            q /= s;
            r /= s;
            if (m === l) break;
            const u = Math.abs(a[m * mm + m - 1]) * (Math.abs(q) + Math.abs(r));
            const v = Math.abs(p) * (Math.abs(a[(m - 1) * mm + m - 1]) + Math.abs(z) + Math.abs(a[(m + 1) * mm + m + 1]));
            if (u <= EPS * v) break;
          }
          for (let i = m + 2; i <= nn; i++) {
            a[i * mm + i - 2] = 0;
            if (i !== m + 2) a[i * mm + i - 3] = 0;
          }
          for (let k = m; k <= nn - 1; k++) {
            if (k !== m) {
              p = a[k * mm + k - 1];
              q = a[(k + 1) * mm + k - 1];
              r = 0;
              if (k !== nn - 1) r = a[(k + 2) * mm + k - 1];
              x = Math.abs(p) + Math.abs(q) + Math.abs(r);
              if (x !== 0) {
                p /= x;
                q /= x;
                r /= x;
              }
            }
            s = sign(Math.sqrt(p * p + q * q + r * r), p);
            if (s !== 0) {
              if (k === m) {
                if (l !== m) a[k * mm + k - 1] = -a[k * mm + k - 1];
              } else a[k * mm + k - 1] = -s * x;
              p += s;
              x = p / s;
              y = q / s;
              z = r / s;
              q /= p;
              r /= p;
              for (let j = k; j <= nn; j++) {
                p = a[k * mm + j] + q * a[(k + 1) * mm + j];
                if (k !== nn - 1) {
                  p += r * a[(k + 2) * mm + j];
                  a[(k + 2) * mm + j] -= p * z;
                }
                a[(k + 1) * mm + j] -= p * y;
                a[k * mm + j] -= p * x;
              }
              const mmin = nn < k + 3 ? nn : k + 3;
              for (let i = l; i <= mmin; i++) {
                p = x * a[i * mm + k] + y * a[i * mm + k + 1];
                if (k !== nn - 1) {
                  p += z * a[i * mm + k + 2];
                  a[i * mm + k + 2] -= p * r;
                }
                a[i * mm + k + 1] -= p * q;
                a[i * mm + k] -= p;
              }
            }
          }
        }
      }
    } while (l < nn - 1);
  }
  return true;
}

/**
 * Unit eigenvector of the real eigenvalue `lambda` of the n×n row-major matrix A by
 * inverse iteration (LU with partial pivoting of A − λI, tiny pivots replaced by
 * ε‖A‖). Writes `out` (length n) and returns the residual ‖A v − λ v‖ / ‖A‖.
 */
export function eigenvectorInverseIteration(
  A: ArrayLike<number>,
  n: number,
  lambda: number,
  out: Float64Array,
  iterations = 3,
): number {
  const nn = n * n;
  if (scrLU.length < nn) scrLU = new Float64Array(nn);
  if (scrPiv.length < n) scrPiv = new Int32Array(n);
  if (scrV.length < n) scrV = new Float64Array(n);
  const lu = scrLU,
    piv = scrPiv;
  let anorm = 0;
  for (let i = 0; i < nn; i++) anorm = Math.max(anorm, Math.abs(A[i]));
  if (!(anorm > 0) || !Number.isFinite(anorm)) {
    out.fill(NaN, 0, n);
    return NaN;
  }
  for (let i = 0; i < nn; i++) lu[i] = A[i];
  for (let i = 0; i < n; i++) lu[i * n + i] -= lambda;
  const tiny = EPS * anorm;
  for (let k = 0; k < n; k++) {
    let p = k,
      mv = Math.abs(lu[k * n + k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(lu[i * n + k]);
      if (v > mv) {
        mv = v;
        p = i;
      }
    }
    piv[k] = p;
    if (p !== k)
      for (let j = 0; j < n; j++) {
        const t = lu[k * n + j];
        lu[k * n + j] = lu[p * n + j];
        lu[p * n + j] = t;
      }
    if (!(Math.abs(lu[k * n + k]) > tiny)) lu[k * n + k] = lu[k * n + k] >= 0 ? tiny : -tiny;
    const inv = 1 / lu[k * n + k];
    for (let i = k + 1; i < n; i++) {
      const f = (lu[i * n + k] *= inv);
      if (f !== 0) for (let j = k + 1; j < n; j++) lu[i * n + j] -= f * lu[k * n + j];
    }
  }
  // Start vector: all ones (deterministic); normalised each iteration.
  const v = out;
  for (let i = 0; i < n; i++) v[i] = 1;
  for (let it = 0; it < iterations; it++) {
    // Solve LU x = P v
    for (let k = 0; k < n; k++) {
      const p = piv[k];
      if (p !== k) {
        const t = v[k];
        v[k] = v[p];
        v[p] = t;
      }
    }
    for (let i = 1; i < n; i++) {
      let s = v[i];
      for (let j = 0; j < i; j++) s -= lu[i * n + j] * v[j];
      v[i] = s;
    }
    for (let i = n - 1; i >= 0; i--) {
      let s = v[i];
      for (let j = i + 1; j < n; j++) s -= lu[i * n + j] * v[j];
      v[i] = s / lu[i * n + i];
    }
    let nrm = 0;
    for (let i = 0; i < n; i++) nrm += v[i] * v[i];
    nrm = Math.sqrt(nrm);
    if (!(nrm > 0) || !Number.isFinite(nrm)) {
      out.fill(NaN, 0, n);
      return NaN;
    }
    for (let i = 0; i < n; i++) v[i] /= nrm;
  }
  let res = 0;
  for (let i = 0; i < n; i++) {
    let s = -lambda * v[i];
    for (let j = 0; j < n; j++) s += A[i * n + j] * v[j];
    res += s * s;
  }
  return Math.sqrt(res) / anorm;
}
