/**
 * Real polynomial root finding. Coefficients are in DESCENDING powers:
 * c = [c0, c1, …, cn] ↔ c0 xⁿ + c1 xⁿ⁻¹ + … + cn.
 *
 * All root finders return DISTINCT real roots in ascending order (a root of
 * multiplicity > 1 is reported once; numerically indistinguishable roots are
 * merged). Leading zero coefficients are ignored.
 *
 * Methods
 *  - `polyRealRoots` (and `solveCubic` / `solveQuartic`, which delegate to it):
 *    the variable is scaled by a power of two R ≥ root bound (exact) so all roots
 *    lie in [−1, 1] (when deg · log2(max/min root-modulus bound) > 800 bits, the
 *    modulus range is split into power-of-two bands searched separately, so small
 *    roots never underflow next to huge ones); real roots are isolated recursively by the critical points
 *    (the real roots of p′, computed the same way): between consecutive critical
 *    points p is monotone, so an interval holds a root iff p changes sign, which is
 *    then found by bracketed, bisection-safeguarded Newton (NR "rtsafe") to full
 *    precision; a critical point where |p| is at the rounding-noise level is a
 *    multiple root. Only sign evaluations of p, p′, … are needed, which makes the
 *    method insensitive to the coefficient-growth/cancellation problems of
 *    floating-point Sturm chains and of the Cardano/Ferrari closed forms (both
 *    measured to miss or invent roots on badly scaled inputs).
 *  - `polyRealRootsSturm`: classical Sturm-sequence isolation + bisection + Newton
 *    (kept as an independent cross-check; see module notes in ARCHITECTURE.md).
 */

const EPS = 2.220446049250313e-16;
const TWO_P1000 = 2 ** 1000;
const TWO_M1000 = 2 ** -1000;

// Exponent-bit access (much cheaper than Math.log2 / Math.pow in the hot path).
const BITS_F64 = new Float64Array(1);
const BITS_U32 = new Uint32Array(BITS_F64.buffer);
const HI_WORD = (() => {
  BITS_F64[0] = 1;
  return BITS_U32[1] === 0x3ff00000 ? 1 : 0;
})();

/** floor(log2 |v|) for finite v ≠ 0 (exact). */
function ilog2(v: number): number {
  BITS_F64[0] = v;
  const e = (BITS_U32[HI_WORD] >>> 20) & 0x7ff;
  return e === 0 ? Math.floor(Math.log2(Math.abs(v))) : e - 1023; // e = 0: subnormal
}

/** 2^e for integer e. */
function pow2(e: number): number {
  if (e >= -1022 && e <= 1023) {
    BITS_U32[HI_WORD] = (e + 1023) << 20;
    BITS_U32[HI_WORD ^ 1] = 0;
    return BITS_F64[0];
  }
  return 2 ** e;
}

/** v · 2^e for integer e, exact unless the RESULT over/underflows (no spurious intermediate overflow). */
function ldexp(v: number, e: number): number {
  while (e > 1000) {
    v *= TWO_P1000;
    e -= 1000;
  }
  while (e < -1000) {
    v *= TWO_M1000;
    e += 1000;
  }
  return v * pow2(e);
}

/** Horner evaluation of c (descending powers) at x. */
export function polyEval(c: ArrayLike<number>, x: number): number {
  const n = c.length;
  if (n === 0) return 0;
  let r = c[0];
  for (let i = 1; i < n; i++) r = r * x + c[i];
  return r;
}

/** Derivative coefficients (descending powers); constant input → [0]. */
export function polyDerivative(c: ArrayLike<number>): Float64Array {
  const n = c.length - 1; // degree
  if (n <= 0) return new Float64Array(1);
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = c[i] * (n - i);
  return d;
}

/** Product of two polynomials (descending powers). */
export function polyMul(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  if (a.length === 0 || b.length === 0) return new Float64Array(0);
  const out = new Float64Array(a.length + b.length - 1);
  for (let i = 0; i < a.length; i++) {
    const ai = a[i];
    if (ai === 0) continue;
    for (let j = 0; j < b.length; j++) out[i + j] += ai * b[j];
  }
  return out;
}

/** Monic polynomial ∏ (x − r_i) (descending powers). */
export function polyFromRoots(roots: ArrayLike<number>): Float64Array {
  let p: Float64Array = new Float64Array([1]);
  const lin = new Float64Array(2);
  lin[0] = 1;
  for (let i = 0; i < roots.length; i++) {
    lin[1] = -roots[i];
    p = polyMul(p, lin);
  }
  return p;
}

/** Running error bound of Horner evaluation at x: 2nε Σ|cᵢ||x|ⁿ⁻ⁱ (Higham, "Accuracy and Stability…", §5.1). */
export function polyEvalErrorBound(c: ArrayLike<number>, x: number): number {
  const n = c.length;
  if (n === 0) return 0;
  const ax = Math.abs(x);
  let r = Math.abs(c[0]);
  for (let i = 1; i < n; i++) r = r * ax + Math.abs(c[i]);
  return 2 * n * EPS * r;
}

// Module-scratch for value+derivative evaluation (avoids tuple allocation in loops).
let evalDp = 0;
/** Returns p(x) and stores p'(x) in `evalDp`. */
function evalWithDeriv(c: ArrayLike<number>, n: number, x: number): number {
  let p = c[0];
  let dp = 0;
  for (let i = 1; i < n; i++) {
    dp = dp * x + p;
    p = p * x + c[i];
  }
  evalDp = dp;
  return p;
}

/** Newton iterations on c, accepting a step only while |p| strictly decreases. */
function newtonPolish(c: ArrayLike<number>, x: number, iters: number): number {
  const n = c.length;
  let f = evalWithDeriv(c, n, x);
  for (let it = 0; it < iters && f !== 0; it++) {
    const df = evalDp;
    if (df === 0 || !Number.isFinite(df)) break;
    const xn = x - f / df;
    if (!Number.isFinite(xn) || xn === x) break;
    const fn = evalWithDeriv(c, n, xn);
    if (!(Math.abs(fn) < Math.abs(f))) break;
    x = xn;
    f = fn;
  }
  return x;
}

/**
 * Sort ascending and merge numerically indistinguishable roots: neighbours closer
 * than 1e-6 relative are merged when |p| at their midpoint is at the rounding
 * noise level (i.e. there is no evidence of two separate roots).
 */
function sortAndMerge(roots: number[], c: ArrayLike<number>): number[] {
  roots.sort((a, b) => a - b);
  if (roots.length < 2) return roots;
  const out: number[] = [roots[0]];
  for (let i = 1; i < roots.length; i++) {
    const x1 = out[out.length - 1];
    const x2 = roots[i];
    const scale = Math.max(Math.abs(x1), Math.abs(x2));
    if (x2 - x1 <= 4 * EPS * scale) continue;
    if (x2 - x1 <= 1e-6 * scale) {
      const mid = 0.5 * (x1 + x2);
      const fm = Math.abs(polyEval(c, mid));
      const f1 = Math.abs(polyEval(c, x1)),
        f2 = Math.abs(polyEval(c, x2));
      // Same noise criterion as the critical-point test in rootsByCriticalPoints (a
      // looser one merged resolvable roots, e.g. 1 and 1 + 5e-7 of (x−1)(x−1−5e-7)(x−5)).
      // Non-finite values (overflow at huge |x|) carry no evidence: keep both.
      const noise = CLUSTER_NOISE_FACTOR * polyEvalErrorBound(c, mid);
      if (Number.isFinite(fm) && Number.isFinite(noise) && fm <= Math.max(f1, f2) + noise) {
        if (f2 < f1) out[out.length - 1] = x2; // keep the smaller residual
        continue;
      }
    }
    out.push(x2);
  }
  return out;
}

/**
 * A critical value |q(x)| ≤ CLUSTER_NOISE_FACTOR × (rigorous Horner error bound) is
 * indistinguishable from 0 (multiple root / unresolvable cluster). The bound
 * 2nε Σ|cᵢ||x|ⁿ⁻ⁱ already over-estimates the actual rounding error, so a factor ~1
 * is safe for exact multiple roots; the previous factor 8 (64 in sortAndMerge) merged
 * clearly resolvable close roots.
 */
const CLUSTER_NOISE_FACTOR = 1.5;

/**
 * Iteration cap of rtsafe. A root of magnitude 2^-s inside a bracket of width ~1 needs
 * up to ~s + 53 bisections (and rtsafe alternates Newton/bisection at worst), so the
 * cap must cover s up to ~1074 (subnormal range); the previous cap of 300 returned
 * unconverged garbage for roots below ~1e-90 relative to the bracket.
 */
const RTSAFE_MAX_ITERS = 2500;

/**
 * NR "rtsafe": Newton safeguarded by bisection on a bracket [a, b] with a sign
 * change (fa = p(a) ≠ 0, p(b) of opposite sign). Converges to full precision.
 */
function rtsafe(c: ArrayLike<number>, a: number, b: number, fa: number): number {
  const n = c.length;
  let xl: number, xh: number;
  if (fa < 0) {
    xl = a;
    xh = b;
  } else {
    xl = b;
    xh = a;
  }
  let x = 0.5 * (a + b);
  let dxold = Math.abs(b - a);
  let dx = dxold;
  let f = evalWithDeriv(c, n, x);
  let df = evalDp;
  for (let it = 0; it < RTSAFE_MAX_ITERS; it++) {
    if (f === 0) return x;
    if (((x - xh) * df - f) * ((x - xl) * df - f) > 0 || Math.abs(2 * f) > Math.abs(dxold * df)) {
      dxold = dx;
      dx = 0.5 * (xh - xl);
      const xn = xl + dx;
      if (xn === xl || xn === xh) return x;
      x = xn;
    } else {
      dxold = dx;
      dx = f / df;
      const xn = x - dx;
      if (xn === x) return x;
      x = xn;
    }
    if (Math.abs(dx) <= 2 * EPS * Math.abs(x)) return x;
    f = evalWithDeriv(c, n, x);
    df = evalDp;
    if (f < 0) xl = x;
    else xh = x;
  }
  return x;
}

function normaliseMaxAbs(p: Float64Array): Float64Array {
  let mx = 0;
  for (let i = 0; i < p.length; i++) mx = Math.max(mx, Math.abs(p[i]));
  if (mx > 0 && mx !== 1) for (let i = 0; i < p.length; i++) p[i] /= mx;
  return p;
}

// ---------------------------------------------------------------------------
// Common preprocessing
// ---------------------------------------------------------------------------

interface Prepared {
  /**
   * Coefficients of the (leading- and trailing-zero-free) polynomial, degree ≥ 1,
   * scaled by an exact power of two to max-abs ∈ [1, 2). NOT made monic: dividing by
   * a tiny leading coefficient could overflow and lose every finite root.
   */
  p: Float64Array;
  /** p with the stripped zero roots multiplied back in (for final merging). */
  pFull: ArrayLike<number>;
  deg: number;
  /** log2 of an upper bound on |roots| (Fujiwara). */
  logUp: number;
  /** log2 of a lower bound on |nonzero roots| (Fujiwara bound of the reversed polynomial). */
  logLow: number;
  /** Single-scale exponent: x = 2^k y maps every root into |y| < 1. */
  k: number;
  /** q(y) ∝ p(2^k y), exact power-of-two normalisation (max-abs ∈ [1, 2)). */
  q: Float64Array;
  ya: number;
  yb: number;
}

/**
 * Coefficients (descending powers of y) of p(2^k y) (reverse = false) or of
 * yⁿ p(2^k / y) (reverse = true), multiplied by a common power of two so that the
 * largest has magnitude in [1, 2). All scalings are exact powers of two computed by
 * exponent arithmetic, so nothing overflows; only coefficients more than ~2^-1022
 * below the largest can underflow (they are negligible at |y| ≤ 1).
 */
function scaledPoly(p: ArrayLike<number>, k: number, reverse: boolean): Float64Array {
  const n = p.length - 1;
  const out = new Float64Array(n + 1);
  let emax = -Infinity;
  for (let j = 0; j <= n; j++) {
    const v = reverse ? p[n - j] : p[j];
    if (v === 0) continue;
    const e = ilog2(v) + (reverse ? k * j : -k * j);
    if (e > emax) emax = e;
  }
  for (let j = 0; j <= n; j++) {
    const v = reverse ? p[n - j] : p[j];
    out[j] = v === 0 ? 0 : ldexp(v, (reverse ? k * j : -k * j) - emax);
  }
  return out;
}

/**
 * Upper estimate of log2 of the Fujiwara bound 2·max(|aᵢ/a₀|^{1/i}, |aₙ/(2a₀)|^{1/n}) on
 * the root moduli of a (reverse = false) or of the reversed polynomial (reverse = true;
 * its roots are the reciprocals, so the negated result is a LOWER bound on the nonzero
 * root moduli of a). Uses the exact exponents (log2|v| < ilog2(v) + 1, log2|a₀| ≥ ilog2(a₀)),
 * so it stays a valid bound (≤ 2 bits looser); no overflow for any finite coefficients.
 */
function logFujiwara(a: ArrayLike<number>, reverse: boolean): number {
  const n = a.length - 1;
  const l0 = ilog2(reverse ? a[n] : a[0]);
  let m = -Infinity;
  for (let i = 1; i <= n; i++) {
    const v = reverse ? a[n - i] : a[i];
    if (v === 0) continue;
    const t = (ilog2(v) + 1 - l0 - (i === n ? 1 : 0)) / i;
    if (t > m) m = t;
  }
  return 1 + m;
}

/**
 * Strip leading zeros and trailing zeros (zero roots pushed into `roots` when in
 * range), normalise, and scale. Returns null when no further roots can exist
 * (degree < 1, empty range, non-finite coefficients).
 */
function prepare(c: ArrayLike<number>, lo: number, hi: number, roots: number[]): Prepared | null {
  let start = 0;
  while (start < c.length && c[start] === 0) start++;
  let end = c.length;
  let zeroRoot = false;
  while (end - start > 1 && c[end - 1] === 0) {
    end--;
    zeroRoot = true;
  }
  const deg = end - start - 1;
  for (let i = start; i < end; i++) if (!Number.isFinite(c[i])) return null;
  if (zeroRoot && lo <= 0 && 0 <= hi) roots.push(0);
  if (deg < 1) return null;
  let emax = -Infinity;
  for (let i = start; i < end; i++) if (c[i] !== 0) emax = Math.max(emax, ilog2(c[i]));
  const p = new Float64Array(deg + 1);
  for (let i = 0; i <= deg; i++) p[i] = ldexp(c[start + i], -emax);
  // Root-modulus bounds (log2), Fujiwara on p and on the reversed polynomial.
  const logUp = logFujiwara(p, false);
  const logLow = -logFujiwara(p, true);
  if (!Number.isFinite(logUp) || !Number.isFinite(logLow)) return null;
  const k = Math.ceil(logUp + 1.5e-4); // 2^k ≥ 1.0001 · bound
  const q = scaledPoly(p, k, false);
  const ya = Math.max(ldexp(lo, -k), -1);
  const yb = Math.min(ldexp(hi, -k), 1);
  if (!(ya <= yb)) return null;
  return { p, pFull: zeroRoot ? polyMul(p, [1, 0]) : p, deg, logUp, logLow, k, q, ya, yb };
}

/** Polish candidate roots (x, unscaled) on p, filter to [lo, hi], merge. */
function finish(pr: Prepared, xs: number[], lo: number, hi: number, roots: number[]): number[] {
  for (const x0 of xs) {
    if (!Number.isFinite(x0)) continue;
    const x = newtonPolish(pr.p, x0, 3);
    if (x >= lo && x <= hi) roots.push(x);
  }
  return sortAndMerge(roots, pr.pFull);
}

/**
 * Root-modulus dynamic range (deg · log2(upper/lower bound), in bits) above which a
 * single scaling x = 2^k y would push the coefficients/terms that determine the
 * smallest roots towards the underflow threshold (2^-1022). Measured failure before
 * this guard: degree 10 with a leading coefficient 1e-40, degree 20 with 1e-20, cubic
 * with 1e-100 (small roots lost or replaced by spurious ones).
 */
const SINGLE_SCALE_MAX_BITS = 800;

// ---------------------------------------------------------------------------
// Critical-point isolation (default)
// ---------------------------------------------------------------------------

/**
 * Distinct real roots of q (degree n ≥ 1, q[0] ≠ 0, coefficients ~O(1)) in [a, b].
 * Recursive on q′; see module header. Roots separated only by critical points at
 * which |q| is within CLUSTER_NOISE_FACTOR × the Horner rounding-error bound are numerically
 * indistinguishable (a multiple root or an unresolvable cluster) and are reported
 * once — at that critical point (the accurate location of an even-multiplicity
 * root) when there is one.
 */
function rootsByCriticalPoints(q: Float64Array, a: number, b: number): number[] {
  const n = q.length - 1;
  if (n === 1) {
    const x = -q[1] / q[0];
    return x >= a && x <= b ? [x] : [];
  }
  if (n === 2) {
    const r = solveQuadratic(q[0], q[1], q[2]);
    const out: number[] = [];
    for (const x of r) if (x >= a && x <= b) out.push(x);
    return out;
  }
  const crit = rootsByCriticalPoints(normaliseMaxAbs(polyDerivative(q)), a, b);
  const out: number[] = [];
  // Current cluster: representative = noise-level critical point if any, else the
  // candidate with the smallest |q|.
  let have = false,
    bestX = 0,
    bestF = Infinity,
    bestIsCrit = false;
  let x0 = a;
  let f0 = polyEval(q, a);
  if (f0 === 0) {
    have = true;
    bestX = a;
    bestF = 0;
  }
  for (let i = 0; i <= crit.length; i++) {
    const isEnd = i === crit.length;
    const x1 = isEnd ? b : crit[i];
    if (!(x1 > x0)) continue;
    const f1 = polyEval(q, x1);
    if (f0 !== 0 && f1 !== 0 && f0 < 0 !== f1 < 0) {
      const r = rtsafe(q, x0, x1, f0);
      const fr = Math.abs(polyEval(q, r));
      if (!have || (!bestIsCrit && fr < bestF)) {
        bestX = r;
        bestF = fr;
      }
      have = true;
    }
    const af = Math.abs(f1);
    if (isEnd) {
      if (f1 === 0 && (!have || !bestIsCrit)) {
        bestX = b;
        bestF = 0;
        have = true;
      }
    } else if (af <= CLUSTER_NOISE_FACTOR * polyEvalErrorBound(q, x1)) {
      // Critical value at the noise level: (part of) a multiple root / cluster.
      if (!have || !bestIsCrit || af < bestF) {
        bestX = x1;
        bestF = af;
        bestIsCrit = true;
      }
      have = true;
    } else if (have) {
      // Significant excursion of |q|: closes the current cluster.
      out.push(bestX);
      have = false;
      bestF = Infinity;
      bestIsCrit = false;
    }
    x0 = x1;
    f0 = f1;
  }
  if (have) out.push(bestX);
  return out;
}

/**
 * Distinct real roots of the polynomial c (descending powers, any degree; tested to
 * 20) in [lo, hi] (default: all real roots; the bracket defaults to
 * min(Cauchy, Fujiwara) root bound), ascending. Exact zero roots (trailing zero
 * coefficients) are detected exactly. Non-finite coefficients → [].
 * See the module header for the method.
 */
export function polyRealRoots(c: ArrayLike<number>, lo = -Infinity, hi = Infinity): number[] {
  if (!(lo <= hi)) return [];
  const roots: number[] = [];
  const pr = prepare(c, lo, hi, roots);
  if (!pr) return roots;
  const { p, deg, k } = pr;
  if (deg === 1) {
    const x = -p[1] / p[0];
    if (Number.isFinite(x) && x >= lo && x <= hi) roots.push(x);
    return sortAndMerge(roots, pr.pFull);
  }
  if (deg === 2) {
    for (const x of solveQuadratic(p[0], p[1], p[2])) if (x >= lo && x <= hi) roots.push(x);
    return sortAndMerge(roots, pr.pFull);
  }
  const xs: number[] = [];
  if (deg * (pr.logUp - pr.logLow) <= SINGLE_SCALE_MAX_BITS) {
    for (const y of rootsByCriticalPoints(pr.q, pr.ya, pr.yb)) xs.push(ldexp(y, k));
    return finish(pr, xs, lo, hi, roots);
  }
  // Wide root-modulus range: split the moduli into bands [2^e0, 2^e1] with INTEGER
  // exponents, each at most SINGLE_SCALE_MAX_BITS / deg bits wide, and search band
  // (e0, e1] on p(2^e1 y) restricted to e0 ≤ log2|x| ≤ e1, i.e. 2^(e0−e1) ≤ |y| ≤ 1.
  // Power-of-two scalings leave Horner's rounding unchanged (absent underflow of
  // negligible coefficients), so adjacent bands see bit-identical scaled values at
  // their shared boundary: a root cannot fall between two bands; one exactly on a
  // boundary is found by both and merged.
  const eLow = Math.floor(pr.logLow),
    eHigh = k;
  const nb = Math.ceil((deg * (eHigh - eLow)) / SINGLE_SCALE_MAX_BITS);
  const w = Math.ceil((eHigh - eLow) / nb);
  for (let e1 = eHigh; e1 > eLow; e1 -= w) {
    const e0 = Math.max(e1 - w, eLow);
    const q = scaledPoly(p, e1, false);
    const yLo = ldexp(lo, -e1),
      yHi = ldexp(hi, -e1);
    const yMin = e0 === eLow ? 0 : pow2(e0 - e1); // lowest band: include the origin
    // Negative half [−1, −yMin] and positive half [yMin, 1], intersected with [lo, hi].
    let a = Math.max(-1, yLo),
      b = Math.min(-yMin, yHi);
    if (a <= b) for (const y of rootsByCriticalPoints(q, a, b)) xs.push(ldexp(y, e1));
    a = Math.max(yMin, yLo);
    b = Math.min(1, yHi);
    if (a <= b && !(yMin === 0 && b === 0)) for (const y of rootsByCriticalPoints(q, a, b)) xs.push(ldexp(y, e1));
  }
  return finish(pr, xs, lo, hi, roots);
}

// ---------------------------------------------------------------------------
// Closed forms / low degree
// ---------------------------------------------------------------------------

/**
 * Real roots of a x² + b x + c (distinct, ascending). Uses the cancellation-free
 * form q = −(b + sign(b)√Δ)/2, x₁ = q/a, x₂ = c/q. A discriminant that is within
 * its rounding-error bound of zero is treated as zero (double root reported once).
 * Degenerates to the linear case when a = 0. The coefficients are first scaled by
 * an exact power of two (max-abs ∈ [1, 2)) so b² − 4ac cannot overflow/underflow
 * (before: (1, 1e200, 1) → [-5e199], (1e300, 3e300, 2e300) → [NaN, NaN]). Non-finite
 * coefficients → []; roots that overflow are dropped.
 */
export function solveQuadratic(a: number, b: number, c: number): number[] {
  if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) return [];
  if (a === 0) {
    if (b === 0) return [];
    const x = -c / b;
    return Number.isFinite(x) ? [x] : [];
  }
  if (c === 0) {
    const r = -b / a;
    if (r === 0 || !Number.isFinite(r)) return [0];
    return r < 0 ? [r, 0] : [0, r];
  }
  const e = -ilog2(Math.max(Math.abs(a), Math.abs(b), Math.abs(c)));
  a = ldexp(a, e);
  b = ldexp(b, e);
  c = ldexp(c, e);
  const b2 = b * b,
    ac4 = 4 * a * c;
  const disc = b2 - ac4;
  const discErr = 4 * EPS * (b2 + Math.abs(ac4));
  if (disc < -discErr) return [];
  // |Δ| at the rounding-noise level: a (numerically) double root.
  if (disc <= discErr) {
    const x = -b / (2 * a);
    return Number.isFinite(x) ? [x] : [];
  }
  const sq = Math.sqrt(disc);
  const q = -0.5 * (b + (b >= 0 ? sq : -sq));
  const x1 = q / a; // may overflow when |a| ≪ |b| (huge root): dropped below
  const x2 = c / q;
  if (!Number.isFinite(x1)) return [x2];
  if (x1 === x2) return [x1];
  return x1 < x2 ? [x1, x2] : [x2, x1];
}

/**
 * Real roots of a x³ + b x² + c x + d (distinct, ascending; a = 0 falls back to the
 * quadratic). Computed by critical-point bracketing + safeguarded Newton (see the
 * module header), which unlike Cardano's formula stays exact for roots of widely
 * different magnitudes and near-double roots.
 */
export function solveCubic(a: number, b: number, c: number, d: number): number[] {
  if (a === 0) return solveQuadratic(b, c, d);
  return polyRealRoots([a, b, c, d]);
}

/**
 * Real roots of a x⁴ + b x³ + c x² + d x + e (distinct, ascending; a = 0 falls back
 * to the cubic). Same method as `solveCubic` (robust replacement for Ferrari; the
 * roots are Newton-polished to full precision).
 */
export function solveQuartic(a: number, b: number, c: number, d: number, e: number): number[] {
  if (a === 0) return solveCubic(b, c, d, e);
  return polyRealRoots([a, b, c, d, e]);
}

// ---------------------------------------------------------------------------
// Sturm sequences (alternative / cross-check)
// ---------------------------------------------------------------------------

const MAX_STURM_ITERS = 20000;

/** Remainder of a / b with running magnitudes; returns the stripped NEGATED remainder or null if ≈ 0. */
function negRemainder(a: Float64Array, b: Float64Array, deg: number): Float64Array | null {
  const na = a.length,
    nb = b.length;
  const w = a.slice();
  const mag = new Float64Array(na);
  for (let i = 0; i < na; i++) mag[i] = Math.abs(a[i]);
  const b0 = b[0];
  for (let i = 0; i <= na - nb; i++) {
    const qv = w[i] / b0;
    w[i] = 0;
    if (qv === 0) continue;
    for (let j = 1; j < nb; j++) {
      const t = qv * b[j];
      w[i + j] -= t;
      mag[i + j] += Math.abs(t);
    }
  }
  // Remainder occupies the last nb-1 slots; strip leading coefficients at the noise level.
  let start = na - nb + 1;
  const tolK = 32 * (deg + 1) * EPS;
  while (start < na && Math.abs(w[start]) <= tolK * mag[start]) start++;
  if (start >= na) return null;
  const r = new Float64Array(na - start);
  let mx = 0;
  for (let i = start; i < na; i++) mx = Math.max(mx, Math.abs(w[i]));
  const inv = -1 / mx; // negate and normalise to max-abs 1
  for (let i = start; i < na; i++) r[i - start] = w[i] * inv;
  return r;
}

/** Number of sign changes of the Sturm chain at x (zeros skipped). */
function signChanges(chain: Float64Array[], x: number): number {
  let changes = 0;
  let last = 0;
  for (let k = 0; k < chain.length; k++) {
    const c = chain[k];
    let v = c[0];
    for (let i = 1; i < c.length; i++) v = v * x + c[i];
    if (v !== 0) {
      const s = v > 0 ? 1 : -1;
      if (last !== 0 && s !== last) changes++;
      last = s;
    }
  }
  return changes;
}

/**
 * Same contract as `polyRealRoots`, computed by Sturm-sequence root isolation +
 * bisection + safeguarded Newton (Sturm 1829; remainders are stripped of leading
 * coefficients at the rounding-noise level, so a multiple root ends the chain at
 * gcd(p, p′) and is counted once; even-multiplicity roots are located by Sturm
 * bisection and polished by Newton on p′). NOTE: floating-point Sturm chains can
 * lose roots when the remainder sequence cancels (≈1e-4 of random badly scaled
 * inputs in our tests) and it uses a single scaling (no band split), so it also loses
 * small roots next to huge ones (deg · log2(range) ≳ 1000 bits) — prefer `polyRealRoots`.
 */
export function polyRealRootsSturm(c: ArrayLike<number>, lo = -Infinity, hi = Infinity): number[] {
  if (!(lo <= hi)) return [];
  const roots: number[] = [];
  const pr = prepare(c, lo, hi, roots);
  if (!pr) return roots;
  const { deg, q, ya, yb } = pr;
  if (deg <= 2) return polyRealRoots(c, lo, hi);

  const chain: Float64Array[] = [q];
  chain.push(normaliseMaxAbs(polyDerivative(q)));
  while (chain.length <= deg + 1) {
    const prev = chain[chain.length - 2];
    const cur = chain[chain.length - 1];
    if (cur.length < 2) break; // constant reached
    const rnext = negRemainder(prev, cur, deg);
    if (!rnext) break;
    chain.push(rnext);
  }
  const dq = polyDerivative(q);
  const yRoots: number[] = [];
  // A root exactly at the lower end is not counted by (ya, yb]; add it explicitly.
  if (polyEval(q, ya) === 0) yRoots.push(ya);

  const stA: number[] = [ya],
    stB: number[] = [yb],
    stVa: number[] = [signChanges(chain, ya)],
    stVb: number[] = [signChanges(chain, yb)];
  let iters = 0;
  while (stA.length > 0 && iters < MAX_STURM_ITERS) {
    iters++;
    const a = stA.pop() as number,
      b = stB.pop() as number,
      va = stVa.pop() as number,
      vb = stVb.pop() as number;
    const count = va - vb;
    if (count <= 0) continue;
    const width = b - a;
    const tiny = 4 * EPS * Math.max(Math.abs(a), Math.abs(b)) + 1e-300;
    if (count === 1) {
      const fb = polyEval(q, b);
      if (fb === 0) {
        yRoots.push(b);
        continue;
      }
      const fa = polyEval(q, a);
      if (fa !== 0 && fa < 0 !== fb < 0) {
        yRoots.push(rtsafe(q, a, b, fa));
        continue;
      }
    }
    if (width <= tiny) {
      yRoots.push(0.5 * (a + b));
      continue;
    }
    if (count === 1) {
      // No sign change: even-multiplicity root. Bisect by Sturm counts.
      let lo2 = a,
        hi2 = b,
        vlo = va;
      while (hi2 - lo2 > 4 * EPS * Math.max(Math.abs(lo2), Math.abs(hi2)) + 1e-300 && iters < MAX_STURM_ITERS) {
        iters++;
        const m = 0.5 * (lo2 + hi2);
        if (m <= lo2 || m >= hi2) break;
        const vm = signChanges(chain, m);
        if (vlo - vm >= 1) hi2 = m;
        else {
          lo2 = m;
          vlo = vm;
        }
      }
      let x = 0.5 * (lo2 + hi2);
      const xp = newtonPolish(dq, x, 8);
      if (Math.abs(polyEval(q, xp)) <= Math.abs(polyEval(q, x))) x = xp;
      if (Math.abs(polyEval(q, x)) <= 1e4 * polyEvalErrorBound(q, x)) yRoots.push(x);
      continue;
    }
    let m = 0.5 * (a + b);
    if (polyEval(q, m) === 0 && width > 1e-12) m = a + 0.5000001 * width;
    const vm = signChanges(chain, m);
    stA.push(a, m);
    stB.push(m, b);
    stVa.push(va, vm);
    stVb.push(vm, vb);
  }
  return finish(
    pr,
    yRoots.map((y) => ldexp(y, pr.k)),
    lo,
    hi,
    roots,
  );
}
