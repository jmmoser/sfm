import { describe, expect, it } from 'vitest';
import { mat3Det, mat3Mul, mat3Transpose } from './mat3';
import { Rng } from './random';
import { rotationAngleBetween, rotvecToMat3 } from './rotation';
import { applySim3, sim3Compose, sim3Inverse, umeyama, type Sim3 } from './sim3';

function randRot(rng: Rng): Float64Array {
  const w = new Float64Array([rng.gaussian(), rng.gaussian(), rng.gaussian()]);
  const n = Math.hypot(w[0], w[1], w[2]);
  const a = rng.uniform(0, Math.PI);
  return rotvecToMat3(w.map((v) => (v / n) * a));
}

function randSim(rng: Rng, withScale = true): Sim3 {
  return {
    s: withScale ? Math.exp(rng.uniform(-3, 3)) : 1,
    R: randRot(rng),
    t: new Float64Array([rng.gaussian() * 10, rng.gaussian() * 10, rng.gaussian() * 10]),
  };
}

function randPoints(rng: Rng, n: number, spread = 1, offset = 0): Float64Array {
  const p = new Float64Array(3 * n);
  for (let i = 0; i < 3 * n; i++) p[i] = rng.gaussian() * spread + offset;
  return p;
}

function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

describe('umeyama', () => {
  it('recovers exact similarity transforms (n = 3 … 1000)', () => {
    const rng = new Rng(1);
    for (let trial = 0; trial < 300; trial++) {
      const n = [3, 4, 10, 100, 1000][trial % 5];
      const sim = randSim(rng);
      const src = randPoints(rng, n, Math.exp(rng.uniform(-2, 3)));
      const dst = applySim3(sim, src);
      const est = umeyama(src, dst, n, true) as Sim3;
      expect(est).not.toBeNull();
      expect(Math.abs(est.s / sim.s - 1)).toBeLessThan(1e-11);
      expect(rotationAngleBetween(est.R, sim.R)).toBeLessThan(1e-11);
      expect(Math.abs(mat3Det(est.R) - 1)).toBeLessThan(1e-14);
      expect(maxDiff(applySim3(est, src), dst) / (sim.s + 1)).toBeLessThan(1e-9);
    }
  });

  it('rigid mode (withScale = false) recovers rotations; scale forced to 1', () => {
    const rng = new Rng(2);
    for (let trial = 0; trial < 200; trial++) {
      const sim = randSim(rng, false);
      const src = randPoints(rng, 20, 3);
      const dst = applySim3(sim, src);
      const est = umeyama(src, dst, 20, false) as Sim3;
      expect(est.s).toBe(1);
      expect(rotationAngleBetween(est.R, sim.R)).toBeLessThan(1e-12);
      expect(maxDiff(est.t, sim.t)).toBeLessThan(1e-10);
      // Scaled target, rigid fit: rotation still right, translation absorbs the centroid.
      const dst2 = applySim3({ ...sim, s: 2 }, src);
      const est2 = umeyama(src, dst2, 20, false) as Sim3;
      expect(rotationAngleBetween(est2.R, sim.R)).toBeLessThan(1e-12);
    }
  });

  it('noisy data: errors shrink with n and match the noise level', () => {
    const rng = new Rng(3);
    const sigma = 0.01;
    for (const n of [10, 100, 1000]) {
      let rotErr = 0,
        scaleErr = 0;
      const trials = 50;
      for (let trial = 0; trial < trials; trial++) {
        const sim = randSim(rng);
        const src = randPoints(rng, n, 1);
        const dst = applySim3(sim, src);
        for (let i = 0; i < 3 * n; i++) dst[i] += sigma * sim.s * rng.gaussian();
        const est = umeyama(src, dst, n, true) as Sim3;
        rotErr += rotationAngleBetween(est.R, sim.R) ** 2;
        scaleErr += (est.s / sim.s - 1) ** 2;
      }
      rotErr = Math.sqrt(rotErr / trials);
      scaleErr = Math.sqrt(scaleErr / trials);
      // ~σ/(spread·√n) behaviour
      expect(rotErr).toBeLessThan((4 * sigma) / Math.sqrt(n));
      expect(scaleErr).toBeLessThan((4 * sigma) / Math.sqrt(n));
    }
  });

  it('weights: zero-weight outliers are ignored; weighted fit is exact on inliers', () => {
    const rng = new Rng(4);
    for (let trial = 0; trial < 100; trial++) {
      const n = 50;
      const sim = randSim(rng);
      const src = randPoints(rng, n, 2);
      const dst = applySim3(sim, src);
      const w = new Float64Array(n).fill(1);
      for (let i = 0; i < 15; i++) {
        const k = rng.int(n);
        dst[3 * k] += 100;
        w[k] = 0;
      }
      // Arbitrary positive weights on the rest do not change an exact fit.
      for (let i = 0; i < n; i++) if (w[i] > 0) w[i] = rng.uniform(0.1, 5);
      const est = umeyama(src, dst, n, true, w) as Sim3;
      expect(Math.abs(est.s / sim.s - 1)).toBeLessThan(1e-11);
      expect(rotationAngleBetween(est.R, sim.R)).toBeLessThan(1e-11);
      // Negative/NaN weights are treated as 0.
      const w2 = w.slice();
      for (let i = 0; i < n; i++) if (w2[i] === 0) w2[i] = i % 2 ? -1 : NaN;
      const est2 = umeyama(src, dst, n, true, w2) as Sim3;
      expect(rotationAngleBetween(est2.R, sim.R)).toBeLessThan(1e-11);
    }
  });

  it('reflection in the data yields a proper rotation (det = +1)', () => {
    const rng = new Rng(5);
    const src = randPoints(rng, 30, 1);
    const dst = src.slice();
    for (let i = 0; i < 30; i++) dst[3 * i + 2] = -dst[3 * i + 2]; // mirror z
    const est = umeyama(src, dst, 30, true) as Sim3;
    expect(est).not.toBeNull();
    expect(Math.abs(mat3Det(est.R) - 1)).toBeLessThan(1e-14);
    const RtR = mat3Mul(mat3Transpose(est.R), est.R);
    expect(maxDiff(RtR, [1, 0, 0, 0, 1, 0, 0, 0, 1])).toBeLessThan(1e-14);
  });

  it('degenerate input returns null', () => {
    const rng = new Rng(6);
    const sim = randSim(rng);
    // n < 3
    const two = randPoints(rng, 2);
    expect(umeyama(two, applySim3(sim, two), 2, true)).toBeNull();
    // collinear points
    const col = new Float64Array(30);
    for (let i = 0; i < 10; i++) {
      const t = rng.gaussian();
      col[3 * i] = 1 + 2 * t;
      col[3 * i + 1] = -1 + t;
      col[3 * i + 2] = 3 * t;
    }
    expect(umeyama(col, applySim3(sim, col), 10, true)).toBeNull();
    expect(umeyama(col, applySim3(sim, col), 10, false)).toBeNull();
    // coincident points
    const same = new Float64Array(15).fill(2);
    expect(umeyama(same, applySim3(sim, same), 5, true)).toBeNull();
    // fewer than 3 points with positive weight
    const pts = randPoints(rng, 10);
    const w = new Float64Array(10);
    w[0] = w[1] = 1;
    expect(umeyama(pts, applySim3(sim, pts), 10, true, w)).toBeNull();
    // target collapsed to a point
    expect(umeyama(pts, new Float64Array(30).fill(1), 10, true)).toBeNull();
    // non-finite data
    const bad = pts.slice();
    bad[4] = NaN;
    expect(umeyama(bad, applySim3(sim, pts), 10, true)).toBeNull();
    // 3 non-collinear points are enough
    const tri = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect(umeyama(tri, applySim3(sim, tri), 3, true)).not.toBeNull();
  });

  it('is accurate far from the origin (GPS/ENU-scale offsets)', () => {
    const rng = new Rng(7);
    const sim = randSim(rng);
    sim.t = new Float64Array([4.2e5, -3.1e6, 120]);
    const src = randPoints(rng, 200, 5, 1e4);
    const dst = applySim3(sim, src);
    const est = umeyama(src, dst, 200, true) as Sim3;
    expect(Math.abs(est.s / sim.s - 1)).toBeLessThan(1e-9);
    expect(rotationAngleBetween(est.R, sim.R)).toBeLessThan(1e-9);
    expect(maxDiff(applySim3(est, src), dst)).toBeLessThan(1e-6 * sim.s * 1e4);
  });
});

describe('applySim3 / inverse / compose', () => {
  it('apply in place, inverse round trip, composition', () => {
    const rng = new Rng(8);
    for (let i = 0; i < 100; i++) {
      const a = randSim(rng),
        b = randSim(rng);
      const p = randPoints(rng, 7, 3);
      const q = applySim3(a, p);
      const inPlace = p.slice();
      applySim3(a, inPlace, inPlace);
      expect(maxDiff(inPlace, q)).toBe(0);
      const back = applySim3(sim3Inverse(a), q);
      expect(maxDiff(back, p)).toBeLessThan(1e-9 * (1 + 1 / a.s) * 30);
      const ab = applySim3(sim3Compose(a, b), p);
      const ab2 = applySim3(a, applySim3(b, p));
      expect(maxDiff(ab, ab2)).toBeLessThan(1e-9 * (1 + a.s * b.s) * 30);
    }
  });
});
