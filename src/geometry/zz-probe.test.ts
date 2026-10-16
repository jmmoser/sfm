import { it } from 'vitest';
import { Rng } from '../math/random';
import { essential5pt, essentialFromPose, recoverPose, decomposeEssential } from './essential';
import { makeTwoViewScene, matDistUpToScale, rotErrDeg, dirErrDeg } from './testutil';
import { svd3 } from '../math/decomp';

function essResid(E: Float64Array, x1: Float64Array, x2: Float64Array, n: number) {
  let ep = 0;
  for (let i = 0; i < n; i++) {
    const a0 = E[0] * x1[2 * i] + E[1] * x1[2 * i + 1] + E[2];
    const a1 = E[3] * x1[2 * i] + E[4] * x1[2 * i + 1] + E[5];
    const a2 = E[6] * x1[2 * i] + E[7] * x1[2 * i + 1] + E[8];
    ep = Math.max(ep, Math.abs(x2[2 * i] * a0 + x2[2 * i + 1] * a1 + a2));
  }
  const { S } = svd3(E);
  return { ep, s: [S[0], S[1], S[2]] };
}

it('probe 5pt', () => {
  const classes = [
    { name: 'general', bmin: 0.1, bmax: 1 },
    { name: '1e-2..1e-1', bmin: 1e-2, bmax: 1e-1 },
    { name: '1e-3..1e-2', bmin: 1e-3, bmax: 1e-2 },
    { name: '1e-4..1e-3', bmin: 1e-4, bmax: 1e-3 },
  ];
  for (const fwd of [false, true]) {
    for (const planar of [false, true]) {
      for (const c of classes) {
        const rng = new Rng(42);
        let found = 0, total = 0, bad = 0, maxCount = 0, sumCount = 0, zero = 0;
        let worstEp = 0, worstSv = 0;
        for (let trial = 0; trial < 400; trial++) {
          const b = c.bmin * Math.pow(c.bmax / c.bmin, rng.next());
          const sc = makeTwoViewScene(rng, {
            n: 5, depthMin: 1, depthMax: 3, baseline: b * 2, maxAngle: 0.5, planar,
            centerDir: fwd ? [0.01 * rng.gaussian(), 0.01 * rng.gaussian(), 1] : undefined,
          });
          const Et = essentialFromPose(sc.R, sc.t);
          const sols = essential5pt(sc.x1, sc.x2);
          total++;
          sumCount += sols.length;
          maxCount = Math.max(maxCount, sols.length);
          if (sols.length === 0) zero++;
          let ok = false;
          for (const E of sols) {
            if (matDistUpToScale(E, Et) < 1e-6) ok = true;
            const r = essResid(E, sc.x1, sc.x2, 5);
            worstEp = Math.max(worstEp, r.ep);
            const sv = Math.abs(r.s[0] - r.s[1]) / r.s[0] + r.s[2] / r.s[0];
            worstSv = Math.max(worstSv, sv);
            if (r.ep > 1e-6 || sv > 1e-6) bad++;
          }
          if (ok) found++;
        }
        console.log(`5pt fwd=${fwd} planar=${planar} ${c.name}: found ${found}/${total}, avg ${(sumCount / total).toFixed(2)}, max ${maxCount}, zero ${zero}, bad sols ${bad}, worst ep ${worstEp.toExponential(2)} sv ${worstSv.toExponential(2)}`);
      }
    }
  }
});

it('probe recoverPose sign', () => {
  const rng = new Rng(5);
  let fails = 0;
  for (let trial = 0; trial < 2000; trial++) {
    const sc = makeTwoViewScene(rng, { n: 30, depthMin: 1, depthMax: 10 + 1000 * rng.next(), baseline: Math.pow(10, -3 * rng.next()), maxAngle: 3 });
    const E = essentialFromPose(sc.R, sc.t);
    const rp = recoverPose(E, sc.x1, sc.x2, 30);
    const tn = Math.hypot(sc.t[0], sc.t[1], sc.t[2]);
    const tu = sc.t.map((v) => v / tn);
    if (rotErrDeg(rp.R, sc.R) > 1e-5 || dirErrDeg(rp.t, tu) > 1e-5) {
      fails++;
      if (fails < 5) console.log('recoverPose fail', trial, rotErrDeg(rp.R, sc.R), dirErrDeg(rp.t, tu), rp.numInFront);
    }
  }
  console.log('recoverPose fails', fails);
  void decomposeEssential;
});
