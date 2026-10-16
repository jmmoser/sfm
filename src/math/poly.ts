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
 *    lie in [−1, 1]; real roots are isolated recursively by the critical points
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
      if (fm <= Math.max(f1, f2) + 64 * polyEvalErrorBound(c, mid)) {
        if (f2 < f1) out[out.length - 1] = x2; // keep the smaller residual
        continue;
      }
    }
    out.push(x2);
  }
  return out;
}

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
  for (let it = 0; it < 300; it++) {
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
  /** Monic coefficients of the (trailing-zero-free) polynomial, degree ≥ 1. */
  p: Float64Array;
  /** p with the stripped zero roots multiplied back in (for final merging). */
  pFull: ArrayLike<number>;
  deg: number;
  /** Power-of-two scale: x = R y, all roots |y| < 1. */
  R: number;
  /** q(y) = p(R y)/R^deg normalised to max-abs 1. */
  q: Float64Array;
  ya: number;
  yb: number;
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
  const lead = c[start];
  const p = new Float64Array(deg + 1);
  for (let i = 0; i <= deg; i++) p[i] = c[start + i] / lead;
  for (let i = 0; i <= deg; i++) if (!Number.isFinite(p[i])) return null;
  // Root bounds (monic): Cauchy 1 + max|p_i|; Fujiwara 2 max(|p_i|^{1/i}, |p_n/2|^{1/n}).
  let cauchy = 0,
    fuji = 0;
  for (let i = 1; i <= deg; i++) {
    const ai = Math.abs(p[i]);
    cauchy = Math.max(cauchy, ai);
    fuji = Math.max(fuji, Math.pow(i === deg ? ai / 2 : ai, 1 / i));
  }
  const bound = Math.min(1 + cauchy, 2 * fuji);
  if (!(bound > 0) || !Number.isFinite(bound)) return null;
  const R = Math.pow(2, Math.ceil(Math.log2(bound * 1.0001)));
  if (!(R > 0) || !Number.isFinite(R)) return null;
  // q_i = p_i / R^i by repeated exact division (no overflow of R^i).
  const q = new Float64Array(deg + 1);
  for (let i = 0; i <= deg; i++) {
    let v = p[i];
    for (let k = 0; k < i; k++) v /= R;
    q[i] = v;
  }
  normaliseMaxAbs(q);
  const ya = Math.max(lo / R, -1);
  const yb = Math.min(hi / R, 1);
  if (!(ya <= yb)) return null;
  return { p, pFull: zeroRoot ? polyMul(p, [1, 0]) : p, deg, R, q, ya, yb };
}

/** Map scaled roots back, polish on the monic polynomial, filter to [lo, hi], merge. */
function finish(pr: Prepared, yRoots: number[], lo: number, hi: number, roots: number[]): number[] {
  for (const y of yRoots) {
    const x = newtonPolish(pr.p, y * pr.R, 3);
    if (x >= lo && x <= hi) roots.push(x);
  }
  return sortAndMerge(roots, pr.pFull);
}

// ---------------------------------------------------------------------------
// Critical-point isolation (default)
// ---------------------------------------------------------------------------

/**
 * Distinct real roots of q (degree n ≥ 1, q[0] ≠ 0, coefficients ~O(1)) in [a, b].
 * Recursive on q′; see module header. Roots separated only by critical points at
 * which |q| is within 8× the Horner rounding-error bound are numerically
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
    } else if (af <= 8 * polyEvalErrorBound(q, x1)) {
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
  const { p, deg } = pr;
  if (deg === 1) {
    const x = -p[1];
    if (x >= lo && x <= hi) roots.push(x);
    return sortAndMerge(roots, pr.pFull);
  }
  if (deg === 2) {
    for (const x of solveQuadratic(1, p[1], p[2])) if (x >= lo && x <= hi) roots.push(x);
    return sortAndMerge(roots, pr.pFull);
  }
  return finish(pr, rootsByCriticalPoints(pr.q, pr.ya, pr.yb), lo, hi, roots);
}

// ---------------------------------------------------------------------------
// Closed forms / low degree
// ---------------------------------------------------------------------------

/**
 * Real roots of a x² + b x + c (distinct, ascending). Uses the cancellation-free
 * form q = −(b + sign(b)√Δ)/2, x₁ = q/a, x₂ = c/q. A discriminant that is within
 * its rounding-error bound of zero is treated as zero (double root reported once).
 * Degenerates to the linear case when a = 0.
 */
export function solveQuadratic(a: number, b: number, c: number): number[] {
  if (a === 0) {
    if (b === 0) return [];
    return [-c / b];
  }
  if (c === 0) {
    const r = -b / a;
    if (r === 0) return [0];
    return r < 0 ? [r, 0] : [0, r];
  }
  const b2 = b * b,
    ac4 = 4 * a * c;
  const disc = b2 - ac4;
  const discErr = 4 * EPS * (b2 + Math.abs(ac4));
  if (disc < -discErr) return [];
  // |Δ| at the rounding-noise level: a (numerically) double root.
  if (disc <= discErr) return [-b / (2 * a)];
  const sq = Math.sqrt(disc);
  const q = -0.5 * (b + (b >= 0 ? sq : -sq));
  const x1 = q / a;
  const x2 = c / q;
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
 * inputs in our tests) — prefer `polyRealRoots`.
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
  return finish(pr, yRoots, lo, hi, roots);
}
