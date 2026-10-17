import { describe, expect, it } from 'vitest';
import { svd3 } from '../math/decomp';
import { Rng } from '../math/random';
import {
  cheiralityCount,
  decomposeEssential,
  essential5pt,
  essentialFromPose,
  essentialLinear,
  essentialLinearIdx,
  projectToEssential,
  recoverPose,
  refineRelativePose,
  sampsonDistanceSq,
  sampsonDistancesSq,
  tangentBasis,
} from './essential';
import { dirErrDeg, makeTwoViewScene, matDistUpToScale, rotErrDeg } from './testutil';

function unitT(t: ArrayLike<number>): Float64Array {
  const n = Math.hypot(t[0], t[1], t[2]);
  return Float64Array.of(t[0] / n, t[1] / n, t[2] / n);
}

function addNoise(rng: Rng, x: Float64Array, sigma: number): Float64Array {
  const o = Float64Array.from(x);
  for (let i = 0; i < o.length; i++) o[i] += sigma * rng.gaussian();
  return o;
}

describe('essential5pt (Stewénius action matrix)', () => {
  it('finds the true E in ≥ 99 % of 1000 random noise-free configurations (incl. small baselines)', () => {
    const rng = new Rng(31);
    let ok = 0,
      total = 0,
      nsol = 0;
    const failByClass: Record<string, number> = {};
    let time = 0;
    // 500 general (baseline/depth 0.03–2), 300 small (0.005–0.1), 200 very small (0.001–0.01)
    const classes = [
      { name: 'general', count: 500, baseline: () => rng.uniform(0.3, 2) },
      { name: 'small', count: 300, baseline: () => rng.uniform(0.05, 0.1) },
      { name: 'verysmall', count: 200, baseline: () => rng.uniform(0.01, 0.1) },
    ];
    for (const c of classes) {
      for (let k = 0; k < c.count; k++) {
        const sc = makeTwoViewScene(rng, {
          n: 5,
          depthMin: 1,
          depthMax: 10,
          baseline: c.baseline(),
          maxAngle: rng.uniform(0, 0.8),
        });
        const Et = essentialFromPose(sc.R, sc.t);
        const t0 = performance.now();
        const sols = essential5pt(sc.x1, sc.x2);
        time += performance.now() - t0;
        nsol += sols.length;
        total++;
        let best = Infinity;
        for (const E of sols) best = Math.min(best, matDistUpToScale(E, Et));
        if (best < 1e-6) ok++;
        else failByClass[c.name] = (failByClass[c.name] ?? 0) + 1;
      }
    }
    console.log(
      `[essential5pt] success ${ok}/${total}, failures by class ${JSON.stringify(failByClass)}, ` +
        `avg #solutions ${(nsol / total).toFixed(2)}, ${((time / total) * 1000).toFixed(1)} µs/solve`,
    );
    expect(ok / total).toBeGreaterThanOrEqual(0.99);
    expect(failByClass.general ?? 0).toBe(0);
  });

  it('every solution satisfies the epipolar and essential constraints', () => {
    const rng = new Rng(32);
    for (let k = 0; k < 100; k++) {
      const sc = makeTwoViewScene(rng, { n: 5, depthMin: 1, depthMax: 8, baseline: 0.5, maxAngle: 0.5 });
      const sols = essential5pt(sc.x1, sc.x2);
      expect(sols.length).toBeGreaterThan(0);
      expect(sols.length).toBeLessThanOrEqual(10);
      for (const E of sols) {
        for (let i = 0; i < 5; i++) {
          const x1 = [sc.x1[2 * i], sc.x1[2 * i + 1], 1],
            x2 = [sc.x2[2 * i], sc.x2[2 * i + 1], 1];
          let e = 0;
          for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) e += x2[r] * E[3 * r + c] * x1[c];
          expect(Math.abs(e)).toBeLessThan(1e-9);
        }
        const { S } = svd3(E);
        expect(S[0]).toBeCloseTo(1, 6);
        expect(S[1]).toBeCloseTo(1, 6);
        expect(S[2]).toBeLessThan(1e-6);
      }
    }
  });

  it('returns no solution (rather than garbage) for degenerate input', () => {
    const z = new Float64Array(10); // all points identical
    expect(essential5pt(z, z).length).toBe(0);
  });
});

describe('essentialLinear (8-point, projected to the manifold)', () => {
  it('is exact on noise-free data (n = 8 and n = 200)', () => {
    const rng = new Rng(33);
    for (const n of [8, 200]) {
      for (let k = 0; k < 50; k++) {
        const sc = makeTwoViewScene(rng, { n, depthMin: 1, depthMax: 10, baseline: 0.8, maxAngle: 0.5 });
        const E = essentialLinear(sc.x1, sc.x2, n)!;
        expect(E).not.toBeNull();
        expect(matDistUpToScale(E, essentialFromPose(sc.R, sc.t))).toBeLessThan(1e-8);
        const { S } = svd3(E);
        expect(S[0]).toBeCloseTo(1, 12);
        expect(S[1]).toBeCloseTo(1, 12);
        expect(S[2]).toBeLessThan(1e-12);
      }
    }
  });

  it('is accurate with 1 px noise and respects the index subset', () => {
    const rng = new Rng(34);
    const f = 1500;
    const sc = makeTwoViewScene(rng, { n: 500, depthMin: 1, depthMax: 6, baseline: 0.6, maxAngle: 0.3 });
    const x1 = addNoise(rng, sc.x1, 1 / f),
      x2 = addNoise(rng, sc.x2, 1 / f);
    const E = essentialLinear(x1, x2, 500)!;
    const rp = recoverPose(E, x1, x2, 500);
    expect(rotErrDeg(rp.R, sc.R)).toBeLessThan(0.2);
    expect(dirErrDeg(rp.t, sc.t)).toBeLessThan(2);
    // subset: corrupt the first 100 points and use only the others
    const y2 = Float64Array.from(x2);
    for (let i = 0; i < 200; i++) y2[i] = rng.uniform(-0.5, 0.5);
    const idx = new Uint32Array(400);
    for (let i = 0; i < 400; i++) idx[i] = 100 + i;
    const Es = essentialLinearIdx(x1, y2, idx, 400)!;
    expect(rotErrDeg(recoverPose(Es, x1, y2, 500).R, sc.R)).toBeLessThan(0.3);
    expect(essentialLinear(x1, x2, 7)).toBeNull();
  });

  it('projectToEssential returns the nearest (1, 1, 0) matrix', () => {
    const E = projectToEssential([3, 0, 0, 0, 2, 0, 0, 0, 1])!;
    expect(Array.from(E)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 0].map((v) => expect.closeTo(v, 12)));
  });
});

describe('decomposeEssential / recoverPose', () => {
  it('the true (R, t) is among the four candidates and is selected by cheirality', () => {
    const rng = new Rng(35);
    for (let k = 0; k < 200; k++) {
      const sc = makeTwoViewScene(rng, {
        n: 50,
        depthMin: rng.uniform(0.3, 3),
        depthMax: rng.uniform(4, 60),
        baseline: rng.uniform(0.05, 1),
        maxAngle: rng.uniform(0, 1),
      });
      const E = essentialFromPose(sc.R, unitT(sc.t));
      const { R1, R2, t } = decomposeEssential(E);
      const eR = Math.min(rotErrDeg(R1, sc.R), rotErrDeg(R2, sc.R));
      expect(eR).toBeLessThan(1e-7);
      expect(Math.min(dirErrDeg(t, sc.t), 180 - dirErrDeg(t, sc.t))).toBeLessThan(1e-7);
      expect(Math.hypot(t[0], t[1], t[2])).toBeCloseTo(1, 12);
      // R1, R2 are rotations
      for (const R of [R1, R2]) {
        const { S } = svd3(R);
        expect(S[0]).toBeCloseTo(1, 12);
        expect(S[2]).toBeCloseTo(1, 12);
        const det =
          R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) + R[2] * (R[3] * R[7] - R[4] * R[6]);
        expect(det).toBeCloseTo(1, 12);
      }
      // sign of E is irrelevant
      const Eneg = E.map((v) => -v);
      for (const M of [E, Eneg]) {
        const rp = recoverPose(M, sc.x1, sc.x2, sc.n);
        expect(rotErrDeg(rp.R, sc.R)).toBeLessThan(1e-7);
        expect(dirErrDeg(rp.t, sc.t)).toBeLessThan(1e-6);
        expect(rp.numInFront).toBe(sc.n);
        expect(rp.inFrontMask.every((v) => v === 1)).toBe(true);
      }
    }
  });

  it('restricts the cheirality vote to the mask and rejects points at infinity', () => {
    const rng = new Rng(36);
    const sc = makeTwoViewScene(rng, { n: 100, depthMin: 1, depthMax: 5, baseline: 0.5, maxAngle: 0.3 });
    const E = essentialFromPose(sc.R, sc.t);
    const mask = new Uint8Array(100);
    for (let i = 0; i < 100; i += 2) mask[i] = 1;
    const rp = recoverPose(E, sc.x1, sc.x2, 100, mask);
    expect(rp.numInFront).toBe(50);
    for (let i = 1; i < 100; i += 2) expect(rp.inFrontMask[i]).toBe(0);
    // identical rays (point at infinity): not counted
    const x = Float64Array.of(0.1, 0.2);
    const R = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
    expect(cheiralityCount(R, [1, 0, 0], x, x, 1, null, null)).toBe(0);
    // far but finite (beyond 1000 baselines): not counted; within: counted
    const far = (d: number) => {
      const p = [0.05 * d, 0, d];
      return Float64Array.of(p[0] / p[2], 0, (p[0] - 1) / p[2], 0);
    };
    const a = far(500),
      b = far(5000);
    expect(cheiralityCount(R, [-1, 0, 0], a.subarray(0, 2), a.subarray(2, 4), 1, null, null)).toBe(1);
    expect(cheiralityCount(R, [-1, 0, 0], b.subarray(0, 2), b.subarray(2, 4), 1, null, null)).toBe(0);
  });
});

describe('Sampson distance', () => {
  it('is zero on the epipolar geometry, scale invariant, and ≈ the squared geometric error', () => {
    const rng = new Rng(37);
    const sc = makeTwoViewScene(rng, { n: 100, depthMin: 1, depthMax: 5, baseline: 0.5, maxAngle: 0.3 });
    const E = essentialFromPose(sc.R, sc.t);
    const E7 = E.map((v) => -7 * v);
    const out = new Float64Array(100);
    sampsonDistancesSq(E, sc.x1, sc.x2, 100, out);
    for (let i = 0; i < 100; i++) {
      expect(out[i]).toBeLessThan(1e-28);
      const d = 1e-3;
      // perturb x2 along the normal of its epipolar line by d: Sampson ≈ d² · |l2|² / (|l1|² + |l2|²)
      const x1 = [sc.x1[2 * i], sc.x1[2 * i + 1], 1];
      const l = [0, 1, 2].map((r) => E[3 * r] * x1[0] + E[3 * r + 1] * x1[1] + E[3 * r + 2]);
      const ln = Math.hypot(l[0], l[1]);
      const x2p = sc.x2[2 * i] + (d * l[0]) / ln,
        y2p = sc.x2[2 * i + 1] + (d * l[1]) / ln;
      const s = sampsonDistanceSq(E, sc.x1[2 * i], sc.x1[2 * i + 1], x2p, y2p);
      const s7 = sampsonDistanceSq(E7, sc.x1[2 * i], sc.x1[2 * i + 1], x2p, y2p);
      expect(s7).toBeCloseTo(s, 18);
      // the true minimal (both-image) squared correction is between d²·ratio and d²
      expect(s).toBeLessThanOrEqual(d * d * 1.01);
      expect(s).toBeGreaterThan(d * d * 0.2);
    }
    expect(sampsonDistanceSq(new Float64Array(9), 0, 0, 0, 0)).toBe(0);
  });
});

describe('refineRelativePose', () => {
  it('converges to the maximum-likelihood pose from a perturbed start', () => {
    const rng = new Rng(38);
    const f = 1500;
    for (let k = 0; k < 20; k++) {
      const sc = makeTwoViewScene(rng, { n: 300, depthMin: 1, depthMax: 8, baseline: 0.5, maxAngle: 0.4 });
      const x1 = addNoise(rng, sc.x1, 0.7 / f),
        x2 = addNoise(rng, sc.x2, 0.7 / f);
      const tu = unitT(sc.t);
      const ref = refineRelativePose(sc.R, tu, x1, x2, 300, null, { iterations: 50 });
      // perturbed start (2°, 10°) converges to the same optimum
      const dR = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
      const th = (2 * Math.PI) / 180;
      dR[0] = Math.cos(th);
      dR[1] = -Math.sin(th);
      dR[3] = Math.sin(th);
      dR[4] = Math.cos(th);
      const R0 = new Float64Array(9);
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) R0[3 * i + j] = dR[3 * i] * sc.R[j] + dR[3 * i + 1] * sc.R[3 + j] + dR[3 * i + 2] * sc.R[6 + j];
      const b1 = new Float64Array(3),
        b2 = new Float64Array(3);
      tangentBasis(tu, b1, b2);
      const t0 = unitT([tu[0] + 0.17 * b1[0], tu[1] + 0.17 * b1[1], tu[2] + 0.17 * b1[2]]);
      const ref2 = refineRelativePose(R0, t0, x1, x2, 300, null, { iterations: 50 });
      expect(ref2.cost).toBeLessThan(ref2.initialCost);
      expect(ref2.cost).toBeCloseTo(ref.cost, 12);
      expect(rotErrDeg(ref2.R, ref.R)).toBeLessThan(1e-4);
      expect(rotErrDeg(ref.R, sc.R)).toBeLessThan(0.1);
      expect(dirErrDeg(ref.t, sc.t)).toBeLessThan(1.5);
      expect(Math.hypot(ref2.t[0], ref2.t[1], ref2.t[2])).toBeCloseTo(1, 12);
    }
  });

  it('Cauchy / Huber kernels limit the influence of gross outliers', () => {
    const rng = new Rng(39);
    const f = 1500;
    const sc = makeTwoViewScene(rng, { n: 400, depthMin: 1, depthMax: 8, baseline: 0.5, maxAngle: 0.4 });
    const x1 = addNoise(rng, sc.x1, 0.5 / f),
      x2 = addNoise(rng, sc.x2, 0.5 / f);
    for (let i = 0; i < 40; i++) {
      x2[2 * i] += rng.uniform(-30, 30) / f;
      x2[2 * i + 1] += rng.uniform(-30, 30) / f;
    }
    const tu = unitT(sc.t);
    const ls = refineRelativePose(sc.R, tu, x1, x2, 400, null, { iterations: 50 });
    const cauchy = refineRelativePose(sc.R, tu, x1, x2, 400, null, { iterations: 50, cauchy: 1 / f });
    const huber = refineRelativePose(sc.R, tu, x1, x2, 400, null, { iterations: 50, huber: 1 / f });
    expect(rotErrDeg(cauchy.R, sc.R)).toBeLessThan(rotErrDeg(ls.R, sc.R));
    expect(rotErrDeg(huber.R, sc.R)).toBeLessThan(rotErrDeg(ls.R, sc.R));
    expect(rotErrDeg(cauchy.R, sc.R)).toBeLessThan(0.05);
  });
});
