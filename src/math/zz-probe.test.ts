import { it } from 'vitest';
import { polyRealRoots, polyFromRoots, polyEval, polyMul, polyRealRootsSturm } from './poly';
import { Rng } from './random';

function check(c: ArrayLike<number>, expected: number[], tolRel = 1e-8): string {
  const r = polyRealRoots(c);
  const ok = r.length === expected.length && expected.every((e, i) => Math.abs(r[i] - e) <= tolRel * Math.max(1, Math.abs(e)) || Math.abs(r[i] - e) <= tolRel * Math.abs(e));
  return `${ok ? 'OK ' : 'BAD'} got ${JSON.stringify(r)} expected ${JSON.stringify(expected)}`;
}

it('probe', () => {
  const log = (...a: unknown[]) => console.log(...a);
  for (const eps of [1e-10, 1e-16, 1e-30, 1e-50, 1e-100, 1e-150, 1e-200, 1e-300]) {
    // (eps x + 1)(x-1)(x-2)
    const c = polyMul([eps, 1], polyFromRoots([1, 2]));
    log('deg3 eps', eps, check(c, [-1 / eps, 1, 2].sort((a, b) => a - b)));
  }
  const base10 = [-1.7, -1.1, -0.5, 0.2, 0.9, 1.3, 1.9, 2.5, 3.1];
  for (const eps of [1e-10, 1e-16, 1e-20, 1e-30, 1e-35, 1e-40, 1e-60]) {
    const c = polyMul([eps, 1], polyFromRoots(base10));
    log('deg10 eps', eps, check(c, [-1 / eps, ...base10].sort((a, b) => a - b), 1e-6));
  }
  const base20 = Array.from({ length: 19 }, (_, i) => -2 + i * 0.23);
  for (const eps of [1e-8, 1e-12, 1e-16, 1e-20]) {
    const c = polyMul([eps, 1], polyFromRoots(base20));
    const r = polyRealRoots(c);
    log('deg20 eps', eps, r.length, r[0], r[1], r[r.length - 1]);
  }
  // tiny constant: roots near zero
  for (const d of [1e-20, 1e-100, 1e-200, 1e-300]) {
    const c = [1, -3, 2, d];
    log('tiny const', d, JSON.stringify(polyRealRoots(c)), 'expected ~', -d / 2);
  }
  // scaled roots
  for (const s of [1e-100, 1e100, 1e-200, 1e200]) {
    const c = polyFromRoots([-1.5 * s, 0.5 * s, 2 * s]);
    log('scaled', s, JSON.stringify(polyRealRoots(c)));
  }
  // all coefficient scales
  log('scaled coeffs', JSON.stringify(polyRealRoots([1e-300, -3e-300, 2e-300])), JSON.stringify(polyRealRoots([1e300, -3e300, 2e300])));
  log('scaled coeffs3', JSON.stringify(polyRealRoots([1e-300, -6e-300, 11e-300, -6e-300])), JSON.stringify(polyRealRoots([1e300, -6e300, 11e300, -6e300])));
  // Sturm variant extremes
  log('sturm deg3 small lead', JSON.stringify(polyRealRootsSturm(polyMul([1e-100, 1], polyFromRoots([1, 2])))));
  // close roots that are resolvable
  for (const d of [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 5e-7, 3e-7, 1e-7]) {
    const c = polyFromRoots([1, 1 + d, 5]);
    const r = polyRealRoots(c);
    // check distinguishability: value at midpoint
    const mid = 1 + d / 2;
    log('close', d, r.length, JSON.stringify(r), 'p(mid)=', polyEval(c, mid));
  }
  // random
  const rng = new Rng(5);
  let bad = 0;
  for (let t = 0; t < 2000; t++) {
    const deg = 3 + rng.int(8);
    const roots: number[] = [];
    for (let i = 0; i < deg - 1; i++) roots.push(rng.uniform(-3, 3));
    const eps = Math.pow(10, -rng.uniform(5, 60));
    const c = polyMul([eps, 1], polyFromRoots(roots));
    const r = polyRealRoots(c);
    roots.push(-1 / eps);
    roots.sort((a, b) => a - b);
    // count roots matched
    let matched = 0;
    for (const e of roots) if (r.some((x) => Math.abs(x - e) <= 1e-5 * Math.max(1, Math.abs(e)))) matched++;
    if (r.length > roots.length || matched < roots.length - 2) {
      bad++;
      if (bad < 5) log('rand bad', deg, eps, r.length, roots.length, matched);
    }
  }
  log('random small-lead bad', bad);
});
