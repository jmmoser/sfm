import { describe, expect, it } from 'vitest';
import {
  polyDerivative,
  polyEval,
  polyEvalErrorBound,
  polyFromRoots,
  polyMul,
  polyRealRoots,
  polyRealRootsSturm,
  solveCubic,
  solveQuadratic,
  solveQuartic,
} from './poly';
import { Rng } from './random';

/** Expect `got` to match `want` element-wise (both ascending) within |Δ| ≤ tol·max(1,|x|). */
function expectRoots(got: number[], want: number[], tol: number): void {
  expect(got.length, `roots ${JSON.stringify(got)} vs ${JSON.stringify(want)}`).toBe(want.length);
  for (let i = 0; i < want.length; i++) {
    expect(Math.abs(got[i] - want[i]), `root ${i}: ${got[i]} vs ${want[i]}`).toBeLessThanOrEqual(
      tol * Math.max(1, Math.abs(want[i])),
    );
  }
  for (let i = 1; i < got.length; i++) expect(got[i]).toBeGreaterThan(got[i - 1]);
}

/**
 * Condition-aware comparison: root i may be off by the forward error bound of a
 * backward-stable solver, 16·errBound(x)/|p'(x)| + 4ε|x| (first-order perturbation).
 */
function expectRootsCond(got: number[], want: number[], c: ArrayLike<number>): void {
  expect(got.length, `roots ${JSON.stringify(got)} vs ${JSON.stringify(want)}`).toBe(want.length);
  const dc = polyDerivative(c);
  for (let i = 0; i < want.length; i++) {
    const x = want[i];
    const tol = (16 * polyEvalErrorBound(c, x)) / Math.abs(polyEval(dc, x)) + 4 * 2.2e-16 * Math.abs(x) + 1e-15;
    expect(Math.abs(got[i] - x), `root ${i}: ${got[i]} vs ${x}`).toBeLessThanOrEqual(tol);
  }
}

/** Chebyshev polynomial T_n coefficients (descending powers), exact integers for n ≤ 20. */
function chebyshev(n: number): Float64Array {
  let t0: Float64Array = new Float64Array([1]);
  let t1: Float64Array = new Float64Array([1, 0]);
  if (n === 0) return t0;
  for (let k = 1; k < n; k++) {
    // T_{k+1} = 2x T_k − T_{k−1}
    const a = polyMul([2, 0], t1);
    const b = new Float64Array(a.length);
    b.set(t0, a.length - t0.length);
    for (let i = 0; i < a.length; i++) a[i] -= b[i];
    t0 = t1;
    t1 = a;
  }
  return t1;
}

/** Product of real-root factors and complex-pair factors (x² − 2 re x + re² + im²). */
function polyFromMixed(real: number[], complex: [number, number][]): Float64Array {
  let p = polyFromRoots(real);
  for (const [re, im] of complex) p = polyMul(p, [1, -2 * re, re * re + im * im]);
  return p;
}

describe('polynomial helpers', () => {
  it('eval, derivative, mul, fromRoots', () => {
    const c = [2, -3, 0, 5]; // 2x³ − 3x² + 5
    expect(polyEval(c, 2)).toBe(9);
    expect(polyEval([], 3)).toBe(0);
    expect(Array.from(polyDerivative(c))).toEqual([6, -6, 0]);
    expect(Array.from(polyDerivative([7]))).toEqual([0]);
    expect(Array.from(polyMul([1, 1], [1, -1]))).toEqual([1, 0, -1]);
    expect(Array.from(polyFromRoots([1, 2, 3]))).toEqual([1, -6, 11, -6]);
    expect(polyEvalErrorBound([1, -6, 11, -6], 2)).toBeGreaterThan(0);
  });
});

describe('solveQuadratic', () => {
  it('basic cases', () => {
    expectRoots(solveQuadratic(1, -3, 2), [1, 2], 1e-15);
    expectRoots(solveQuadratic(2, 0, -8), [-2, 2], 1e-15);
    expectRoots(solveQuadratic(1, 0, 1), [], 0);
    expectRoots(solveQuadratic(1, -2, 1), [1], 1e-15); // double root once
    expectRoots(solveQuadratic(0, 2, -4), [2], 0); // linear
    expectRoots(solveQuadratic(0, 0, 1), [], 0);
    expectRoots(solveQuadratic(1, 5, 0), [-5, 0], 0);
    expectRoots(solveQuadratic(3, 0, 0), [0], 0);
  });

  it('is cancellation-free', () => {
    // x² + 1e8 x + 1 = 0 → roots ≈ −1e8, −1e-8 (naive formula loses the small one).
    const r = solveQuadratic(1, 1e8, 1);
    expect(r.length).toBe(2);
    expect(Math.abs(r[1] - -1e-8) / 1e-8).toBeLessThan(1e-15);
    expect(Math.abs(r[0] - -1e8) / 1e8).toBeLessThan(1e-15);
  });

  it('near-double roots from rounded coefficients are not lost', () => {
    const rng = new Rng(1);
    for (let i = 0; i < 1000; i++) {
      const x0 = rng.uniform(-100, 100);
      const a = rng.uniform(0.1, 10);
      // a (x − x0)² with rounded coefficients
      const r = solveQuadratic(a, -2 * a * x0, a * x0 * x0);
      expect(r.length).toBeGreaterThanOrEqual(1);
      for (const x of r) expect(Math.abs(x - x0)).toBeLessThan(1e-6 * Math.max(1, Math.abs(x0)));
    }
  });

  it('random quadratics with known roots', () => {
    const rng = new Rng(2);
    for (let i = 0; i < 2000; i++) {
      const r1 = rng.uniform(-10, 10),
        r2 = rng.uniform(-10, 10);
      if (Math.abs(r1 - r2) < 1e-3) continue;
      const a = rng.uniform(-5, 5) || 1;
      const r = solveQuadratic(a, -a * (r1 + r2), a * r1 * r2);
      expectRoots(r, [Math.min(r1, r2), Math.max(r1, r2)], 1e-11);
    }
  });
});

describe('solveCubic', () => {
  it('three real roots, one real root, multiple roots', () => {
    expectRoots(solveCubic(1, -6, 11, -6), [1, 2, 3], 1e-14);
    expectRoots(solveCubic(1, 0, 0, -8), [2], 1e-15);
    expectRoots(solveCubic(1, -4, 5, -2), [1, 2], 1e-7); // (x−1)²(x−2)
    expectRoots(solveCubic(1, -6, 12, -8), [2], 1e-5); // (x−2)³
    expectRoots(solveCubic(1, 0, 0, 0), [0], 0);
    expectRoots(solveCubic(0, 1, -3, 2), [1, 2], 1e-15);
    expectRoots(solveCubic(2, -4, 2, 0), [0, 1], 1e-15); // 2x(x−1)²
    expectRoots(solveCubic(1, 0, 1, 0), [0], 0); // x(x²+1)
    // Roots of very different magnitudes.
    const p = polyFromRoots([1e-3, 1, 1e3]);
    expectRoots(solveCubic(p[0], p[1], p[2], p[3]), [1e-3, 1, 1e3], 1e-12);
    // Close to a double root far from 0 (cancellation in the depressed form).
    const q = polyFromRoots([1000, 1000, 1001]);
    const rq = solveCubic(q[0], q[1], q[2], q[3]);
    expect(rq.length).toBe(2);
    // Inherent conditioning of a double root: |Δx| ~ sqrt(2 δp / |p''|) ≈ 1e-3 here.
    expect(Math.abs(rq[0] - 1000)).toBeLessThan(3e-3);
    expect(Math.abs(rq[1] - 1001)).toBeLessThan(2e-6); // δp/|p'| with δp ≈ ε·Σ|cᵢ||x|ⁱ ≈ 2e-6, p'(1001) = 1
  });

  it('random cubics with known real roots (and complex pairs)', () => {
    const rng = new Rng(3);
    for (let i = 0; i < 3000; i++) {
      const roots = [rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(-10, 10)].sort((a, b) => a - b);
      if (roots[1] - roots[0] < 1e-2 || roots[2] - roots[1] < 1e-2) continue;
      const s = rng.uniform(0.1, 10) * (rng.next() < 0.5 ? -1 : 1);
      const p = polyFromRoots(roots).map((v) => v * s);
      expectRoots(solveCubic(p[0], p[1], p[2], p[3]), roots, 1e-11);
      // one real root + complex pair
      const re = rng.uniform(-5, 5),
        im = rng.uniform(0.01, 5),
        x0 = rng.uniform(-5, 5);
      const c = polyFromMixed([x0], [[re, im]]);
      expectRoots(solveCubic(c[0], c[1], c[2], c[3]), [x0], 1e-11);
    }
  });
});

describe('solveQuartic', () => {
  it('special cases', () => {
    expectRoots(solveQuartic(1, -10, 35, -50, 24), [1, 2, 3, 4], 1e-13);
    expectRoots(solveQuartic(1, 0, -5, 0, 4), [-2, -1, 1, 2], 1e-14); // biquadratic
    expectRoots(solveQuartic(1, 0, 5, 0, 4), [], 0); // (x²+1)(x²+4)
    expectRoots(solveQuartic(1, 0, 0, 0, -16), [-2, 2], 1e-14);
    expectRoots(solveQuartic(0, 1, -6, 11, -6), [1, 2, 3], 1e-14); // cubic fallback
    expectRoots(solveQuartic(1, -6, 11, -6, 0), [0, 1, 2, 3], 1e-14); // e = 0
    const d = polyFromRoots([1, 1, 2, 3]); // double root
    expectRoots(solveQuartic(d[0], d[1], d[2], d[3], d[4]), [1, 2, 3], 1e-7);
    const dd = polyFromRoots([-1, -1, 2, 2]); // two double roots
    expectRoots(solveQuartic(dd[0], dd[1], dd[2], dd[3], dd[4]), [-1, 2], 1e-7);
    const t = polyFromRoots([0.5, 0.5, 0.5, 0.5]);
    expectRoots(solveQuartic(t[0], t[1], t[2], t[3], t[4]), [0.5], 1e-3);
  });

  it('random quartics vs known roots (4 real / 2 real / 0 real)', () => {
    const rng = new Rng(4);
    let n4 = 0;
    for (let i = 0; i < 3000; i++) {
      const roots = [0, 1, 2, 3].map(() => rng.uniform(-10, 10)).sort((a, b) => a - b);
      let ok = true;
      for (let k = 1; k < 4; k++) if (roots[k] - roots[k - 1] < 2e-2) ok = false;
      const s = rng.uniform(0.1, 10) * (rng.next() < 0.5 ? -1 : 1);
      if (ok) {
        const p = polyFromRoots(roots).map((v) => v * s);
        expectRoots(solveQuartic(p[0], p[1], p[2], p[3], p[4]), roots, 1e-10);
        n4++;
      }
      const r2 = [rng.uniform(-10, 10), rng.uniform(-10, 10)].sort((a, b) => a - b);
      if (r2[1] - r2[0] > 1e-2) {
        const c = polyFromMixed(r2, [[rng.uniform(-5, 5), rng.uniform(0.05, 5)]]).map((v) => v * s);
        expectRoots(solveQuartic(c[0], c[1], c[2], c[3], c[4]), r2, 1e-10);
      }
      const c0 = polyFromMixed(
        [],
        [
          [rng.uniform(-5, 5), rng.uniform(0.05, 5)],
          [rng.uniform(-5, 5), rng.uniform(0.05, 5)],
        ],
      );
      expectRoots(solveQuartic(c0[0], c0[1], c0[2], c0[3], c0[4]), [], 0);
    }
    expect(n4).toBeGreaterThan(2000);
  });

  it('badly scaled cubics/quartics (roots spanning 1e-4..1e5, tiny leading coefficient)', () => {
    const rng = new Rng(5);
    for (let i = 0; i < 3000; i++) {
      for (const deg of [3, 4]) {
        const roots: number[] = [];
        while (roots.length < deg) {
          const x = (rng.next() < 0.5 ? -1 : 1) * Math.pow(10, rng.uniform(-4, 5));
          if (roots.every((r) => Math.abs(r - x) > 1e-3 * Math.max(Math.abs(r), Math.abs(x)))) roots.push(x);
        }
        roots.sort((a, b) => a - b);
        const s = Math.pow(10, rng.uniform(-6, 6));
        const p = Array.from(polyFromRoots(roots)).map((v) => v * s);
        const got = deg === 3 ? solveCubic(p[0], p[1], p[2], p[3]) : solveQuartic(p[0], p[1], p[2], p[3], p[4]);
        expectRootsCond(got, roots, p);
      }
    }
    // Regression: Ferrari reported spurious roots / Cardano missed roots for these.
    const q1 = [-0.0007616686989027384, -16.628011705886454, -0.005114053080984986, 0.11748886732671077, 0.012032964975305654];
    const r1 = solveQuartic(q1[0], q1[1], q1[2], q1[3], q1[4]);
    expect(r1.length).toBe(2);
    for (const x of r1) expect(Math.abs(polyEval(q1, x))).toBeLessThan(1e3 * polyEvalErrorBound(q1, x));
    const c1 = [-0.0022465589304886003, 229.8643961463394, -0.0001804629631653148, 0.00006259335891834099];
    expect(solveCubic(c1[0], c1[1], c1[2], c1[3]).length).toBe(1);
  });

  it('agrees with the Sturm variant on random-coefficient quartics and degree-10 polynomials', () => {
    const rng = new Rng(55);
    for (let i = 0; i < 2000; i++) {
      const deg = i % 2 === 0 ? 4 : 10;
      const c = Array.from({ length: deg + 1 }, () => rng.gaussian());
      const a = polyRealRoots(c);
      const b = polyRealRootsSturm(c);
      expect(b.length).toBe(a.length);
      for (let k = 0; k < a.length; k++) expect(Math.abs(a[k] - b[k])).toBeLessThan(1e-9 * Math.max(1, Math.abs(a[k])));
    }
  });
});

describe('polyRealRoots', () => {
  it('low-degree and special inputs', () => {
    expectRoots(polyRealRoots([1, -3]), [3], 0);
    expectRoots(polyRealRoots([1, -3, 2]), [1, 2], 1e-15);
    expectRoots(polyRealRoots([0, 0, 1, -3, 2]), [1, 2], 1e-15); // leading zeros
    expectRoots(polyRealRoots([1, -1, 0, 0]), [0, 1], 1e-15); // x³ − x² = x²(x − 1)
    expectRoots(polyRealRoots([5]), [], 0);
    expectRoots(polyRealRoots([0, 0, 0]), [], 0);
    expectRoots(polyRealRoots([]), [], 0);
    expectRoots(polyRealRoots([1, 0, 1]), [], 0);
    expectRoots(polyRealRoots([1, 0, 0, 0, 1]), [], 0); // x⁴ + 1
    expectRoots(polyRealRoots([1, 0, 0, 0, 0]), [0], 0); // x⁴
    expectRoots(polyRealRoots([2, NaN, 1]), [], 0);
  });

  it('Chebyshev polynomials up to degree 20 (exact integer coefficients, clustered roots)', () => {
    for (let n = 1; n <= 20; n++) {
      const T = chebyshev(n);
      const want: number[] = [];
      for (let k = 1; k <= n; k++) want.push(Math.cos(((2 * k - 1) * Math.PI) / (2 * n)));
      want.sort((a, b) => a - b);
      expectRootsCond(polyRealRoots(T), want, T);
    }
  });

  it('Wilkinson-type polynomial ∏(x − k), k = 1..12', () => {
    const roots = Array.from({ length: 12 }, (_, i) => i + 1);
    const W = polyFromRoots(roots);
    expectRoots(polyRealRoots(W), roots, 1e-7);
    expectRootsCond(polyRealRoots(W), roots, W);
  });

  it('random real roots + complex pairs, degree 3..20', () => {
    const rng = new Rng(6);
    for (let trial = 0; trial < 600; trial++) {
      const deg = 3 + rng.int(18);
      const nComplexPairs = rng.int(Math.floor(deg / 2) + 1);
      const nReal = deg - 2 * nComplexPairs;
      const real: number[] = [];
      // Well separated real roots in [−1, 1] (separation ≥ 0.1 / nReal).
      while (real.length < nReal) {
        const x = rng.uniform(-1, 1);
        if (real.every((r) => Math.abs(r - x) > 0.5 / Math.max(nReal, 1))) real.push(x);
      }
      real.sort((a, b) => a - b);
      const complex: [number, number][] = [];
      for (let k = 0; k < nComplexPairs; k++) complex.push([rng.uniform(-1, 1), rng.uniform(0.2, 1)]);
      const sc = rng.uniform(0.5, 2);
      const p = polyFromMixed(real, complex).map((v) => v * sc);
      expectRootsCond(polyRealRoots(p), real, p);
    }
  });

  it('roots spanning many orders of magnitude', () => {
    const roots = [-1e4, -2.5, 1e-4, 3e-2, 0.7, 50, 2e3];
    const got = polyRealRoots(polyFromRoots(roots));
    expect(got.length).toBe(roots.length);
    for (let i = 0; i < roots.length; i++) expect(Math.abs(got[i] - roots[i]) / Math.abs(roots[i])).toBeLessThan(1e-9);
  });

  it('multiple and clustered roots', () => {
    expectRoots(polyRealRoots(polyFromRoots([1, 1, -2, 3])), [-2, 1, 3], 1e-7);
    expectRoots(polyRealRoots(polyFromRoots([0.5, 0.5, 0.5, -1])), [-1, 0.5], 1e-4);
    expectRoots(polyRealRoots(polyMul(polyFromRoots([2, 2]), [1, 0, 1])), [2], 1e-7); // even multiplicity, no sign change
    expectRoots(polyRealRoots(polyFromRoots([1, 1.001, 1.002, -1])), [-1, 1, 1.001, 1.002], 1e-9);
    expectRoots(polyRealRoots(polyFromRoots([3, 3, 3, 3, 3, 3])), [3], 2e-2);
    expectRoots(polyRealRoots(polyFromRoots([-1, -1, 1, 1, 2])), [-1, 1, 2], 1e-6);
  });

  it('Newton polish does not jump from an exact multiple root to another root (it used to drop the multiple root)', () => {
    // At a multiple root p and p′ are both rounding noise, so the step p/p′ is arbitrary; one that
    // landed near another root was accepted (|p| no larger) and the two copies were merged.
    expectRoots(solveQuartic(1, 17.5, 90.5, 66.5, -367.5), [-7, -5, 1.5], 1e-7); // (x+7)²(x+5)(x−1.5)
    expectRoots(polyRealRoots(polyFromRoots([3, 8, 8, -7, -7, -6, -6])), [-7, -6, 3, 8], 1e-6);
    expectRoots(polyRealRoots(Array.from(polyFromRoots([-39, -39, -23]), (v) => 3 * v)), [-39, -23], 1e-7);
    expectRoots(polyRealRoots(polyFromRoots([-8, -4, 1, 1, 1, 2, 3, 3])), [-8, -4, 1, 2, 3], 1e-4);
    // Small integer roots with multiplicities 1..3: the distinct-root count is exact, on the default
    // range and on a bracket around the roots.
    const rng = new Rng(18);
    for (let trial = 0; trial < 2000; trial++) {
      const roots: number[] = [];
      const nd = 2 + rng.int(3);
      while (roots.length < nd) {
        const x = rng.int(21) - 10;
        if (!roots.includes(x)) roots.push(x);
      }
      roots.sort((a, b) => a - b);
      const all = roots.flatMap((x) => Array<number>(1 + rng.int(3)).fill(x));
      if (all.length < 3 || all.length > 9) continue;
      const p = polyFromRoots(all);
      const tol = all.length - roots.length > 2 ? 1e-3 : 1e-5;
      expectRoots(polyRealRoots(p), roots, tol);
      expectRoots(polyRealRoots(p, roots[0] - 0.5, roots[nd - 1] + 0.5), roots, tol);
    }
  });

  it('respects [lo, hi]', () => {
    const p = polyFromRoots([-3, -1, 0.5, 2, 4]);
    expectRoots(polyRealRoots(p, -2, 3), [-1, 0.5, 2], 1e-13);
    expectRoots(polyRealRoots(p, 0.5, 2), [0.5, 2], 1e-13);
    expectRoots(polyRealRoots(p, 5, 10), [], 0);
    expectRoots(polyRealRoots(p, 3, -3), [], 0);
    expectRoots(polyRealRoots([1, 0, -1, 0], 0, 2), [0, 1], 1e-15);
  });

  it('never misses a sign change and never reports a non-root (random coefficients, degree 3..20)', () => {
    const rng = new Rng(7);
    for (let trial = 0; trial < 300; trial++) {
      const deg = 3 + rng.int(18);
      const c: number[] = [];
      for (let i = 0; i <= deg; i++) c.push(rng.gaussian() * Math.pow(10, rng.uniform(-2, 2)));
      const roots = polyRealRoots(c);
      for (const x of roots) {
        // Each reported root is a genuine (numerical) root: |p(x)| within a modest multiple of the rounding bound,
        // or p changes sign within a relative 1e-9 neighbourhood.
        const fx = Math.abs(polyEval(c, x));
        const h = 1e-9 * Math.max(1, Math.abs(x));
        const sc = Math.sign(polyEval(c, x - h)) !== Math.sign(polyEval(c, x + h));
        expect(fx <= 1e4 * polyEvalErrorBound(c, x) || sc, `deg ${deg} root ${x} |p|=${fx}`).toBe(true);
      }
      // Dense sampling between ±(bound): every sign change interval must contain a reported root.
      let bound = 0;
      for (let i = 1; i <= deg; i++) bound = Math.max(bound, Math.abs(c[i] / c[0]));
      bound = 1 + bound;
      const N = 4000;
      let prevX = -bound,
        prevF = polyEval(c, prevX);
      for (let i = 1; i <= N; i++) {
        const x = -bound + (2 * bound * i) / N;
        const f = polyEval(c, x);
        if (prevF !== 0 && f !== 0 && Math.sign(f) !== Math.sign(prevF)) {
          const found = roots.some((r) => r >= prevX - 1e-9 && r <= x + 1e-9);
          expect(found, `deg ${deg}: sign change in [${prevX}, ${x}] not covered by ${JSON.stringify(roots)}`).toBe(true);
        }
        prevX = x;
        prevF = f;
      }
    }
  });

  it('clusters: every true root is matched, never more roots than exist', () => {
    const rng = new Rng(9);
    for (let trial = 0; trial < 1000; trial++) {
      const deg = 3 + rng.int(10);
      const real: number[] = [];
      const center = rng.uniform(-3, 3);
      const sep = Math.pow(10, rng.uniform(-5, -1));
      const nCl = 2 + rng.int(2);
      for (let k = 0; k < nCl; k++) real.push(center + k * sep);
      while (real.length < deg) {
        const x = rng.uniform(-10, 10);
        if (real.every((r) => Math.abs(r - x) > 0.3)) real.push(x);
      }
      real.sort((a, b) => a - b);
      const p = polyFromRoots(real);
      const got = polyRealRoots(p);
      expect(got.length).toBeLessThanOrEqual(real.length);
      const dp = polyDerivative(p);
      for (const x of real) {
        const tol = (16 * polyEvalErrorBound(p, x)) / Math.abs(polyEval(dp, x)) + 1e-12;
        const nearest = got.reduce((m, g) => Math.min(m, Math.abs(g - x)), Infinity);
        expect(nearest).toBeLessThanOrEqual(Math.max(tol, nCl * sep * 1.01));
      }
    }
  });

  it('Sturm variant: basic correctness', () => {
    for (let n = 3; n <= 20; n++) {
      const T = chebyshev(n);
      const want: number[] = [];
      for (let k = 1; k <= n; k++) want.push(Math.cos(((2 * k - 1) * Math.PI) / (2 * n)));
      want.sort((a, b) => a - b);
      expectRootsCond(polyRealRootsSturm(T), want, T);
    }
    expectRoots(polyRealRootsSturm(polyFromRoots([1, 1, -2, 3])), [-2, 1, 3], 1e-7);
    expectRoots(polyRealRootsSturm(polyMul(polyFromRoots([2, 2]), [1, 0, 1])), [2], 1e-7);
    expectRoots(polyRealRootsSturm([0, 1, -1, 0, 0]), [0, 1], 1e-15);
    expectRoots(polyRealRootsSturm(polyFromRoots([-3, -1, 0.5, 2, 4]), -2, 3), [-1, 0.5, 2], 1e-13);
  });

  it('Sturm variant: a multiple root exactly at lo or hi does not zero the Sturm count there', () => {
    // Every chain member vanishes at a multiple root, so the count at that end was 0: the other
    // roots were dropped (at lo) or found twice (at hi).
    expectRoots(polyRealRootsSturm(polyFromRoots([1, 1, 2, 3]), 1, 3), [1, 2, 3], 1e-6);
    expectRoots(polyRealRootsSturm(polyFromRoots([-2, -2, 1, 4, 6]), -2, 8), [-2, 1, 4, 6], 1e-6);
    expectRoots(polyRealRootsSturm(polyFromRoots([-2, -2, -2, 8]), -Infinity, -2), [-2], 1e-12);
    expectRoots(polyRealRootsSturm(polyFromRoots([1, 1, 2, 3, 3]), 1, 3), [1, 2, 3], 1e-6);
    expect(polyRealRootsSturm(polyFromRoots([1, 1, 1]), 1, 1)).toEqual([1]);
  });

  it('timing (degree 10, as in the 5-point solver)', () => {
    const rng = new Rng(8);
    const polys: number[][] = [];
    for (let i = 0; i < 100; i++) polys.push(Array.from({ length: 11 }, () => rng.gaussian()));
    for (let i = 0; i < 200; i++) polyRealRoots(polys[i % 100]);
    const t0 = performance.now();
    const N = 3000;
    for (let i = 0; i < N; i++) polyRealRoots(polys[i % 100]);
    const us = ((performance.now() - t0) / N) * 1000;
    let t1 = performance.now();
    for (let i = 0; i < N; i++) polyRealRootsSturm(polys[i % 100]);
    const usS = ((performance.now() - t1) / N) * 1000;
    const cub = polys.map((p) => p.slice(0, 4));
    for (let i = 0; i < 2000; i++) solveCubic(cub[i % 100][0], cub[i % 100][1], cub[i % 100][2], cub[i % 100][3]);
    t1 = performance.now();
    for (let i = 0; i < 20000; i++) solveCubic(cub[i % 100][0], cub[i % 100][1], cub[i % 100][2], cub[i % 100][3]);
    const usC = ((performance.now() - t1) / 20000) * 1000;
    t1 = performance.now();
    for (let i = 0; i < 20000; i++) {
      const q = polys[i % 100];
      solveQuartic(q[0], q[1], q[2], q[3], q[4]);
    }
    const usQ = ((performance.now() - t1) / 20000) * 1000;
    console.log(
      `[math timing] polyRealRoots deg 10: ${us.toFixed(1)} µs (Sturm ${usS.toFixed(1)} µs), solveCubic ${usC.toFixed(2)} µs, solveQuartic ${usQ.toFixed(2)} µs`,
    );
    expect(us).toBeLessThan(5000);
  });
});
