import { describe, expect, it } from 'vitest';
import { Rng } from '../math/random';
import {
  decomposeHomography,
  homographyDLT,
  homographyDLTIdx,
  homographySampleDegenerate,
  homographyTransferErrorSq,
  homographyTransferErrorsSq,
} from './homography';
import { dirErrDeg, makeTwoViewScene, matDistUpToScale, randomRotation, randomUnitVector, rotErrDeg } from './testutil';

function applyH(H: ArrayLike<number>, x: number, y: number): [number, number] {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

function randomH(rng: Rng): Float64Array {
  // a well-behaved pixel homography: K2 (R + t nᵀ) K1⁻¹ with a mild plane
  const H = new Float64Array(9);
  for (let i = 0; i < 9; i++) H[i] = (i % 4 === 0 ? 1 : 0) + 0.1 * rng.gaussian();
  H[2] = rng.uniform(-50, 50);
  H[5] = rng.uniform(-50, 50);
  H[6] = rng.uniform(-1e-4, 1e-4);
  H[7] = rng.uniform(-1e-4, 1e-4);
  return H;
}

describe('homographyDLT', () => {
  it('is exact for 4 and for many noise-free correspondences', () => {
    const rng = new Rng(51);
    for (let k = 0; k < 200; k++) {
      const H = randomH(rng);
      const n = k % 2 === 0 ? 4 : 100;
      const p1 = new Float64Array(2 * n),
        p2 = new Float64Array(2 * n);
      for (let i = 0; i < n; i++) {
        p1[2 * i] = rng.uniform(0, 1920);
        p1[2 * i + 1] = rng.uniform(0, 1440);
        [p2[2 * i], p2[2 * i + 1]] = applyH(H, p1[2 * i], p1[2 * i + 1]);
      }
      if (n === 4 && homographySampleDegenerate(p1, p2, [0, 1, 2, 3])) continue;
      const Hest = homographyDLT(p1, p2, n)!;
      expect(matDistUpToScale(Hest, H)).toBeLessThan(1e-7);
      expect(Math.hypot(...Hest)).toBeCloseTo(1, 12);
      // sign: maps the first point to w > 0
      expect(Hest[6] * p1[0] + Hest[7] * p1[1] + Hest[8]).toBeGreaterThan(0);
      const r = new Float64Array(n);
      homographyTransferErrorsSq(Hest, p1, p2, n, r);
      for (let i = 0; i < n; i++) expect(r[i]).toBeLessThan(1e-12);
    }
  });

  it('is accurate with noise and supports index subsets', () => {
    const rng = new Rng(52);
    const H = randomH(rng);
    const n = 300;
    const p1 = new Float64Array(2 * n),
      p2 = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      p1[2 * i] = rng.uniform(0, 1920);
      p1[2 * i + 1] = rng.uniform(0, 1440);
      [p2[2 * i], p2[2 * i + 1]] = applyH(H, p1[2 * i], p1[2 * i + 1]);
      p2[2 * i] += i < 100 ? rng.uniform(-300, 300) : 0.7 * rng.gaussian();
      p2[2 * i + 1] += i < 100 ? rng.uniform(-300, 300) : 0.7 * rng.gaussian();
    }
    const idx = new Uint32Array(200);
    for (let i = 0; i < 200; i++) idx[i] = 100 + i;
    const Hest = homographyDLTIdx(p1, p2, idx, 200)!;
    let rms = 0;
    for (let i = 100; i < n; i++) {
      const [u, v] = applyH(Hest, p1[2 * i], p1[2 * i + 1]);
      const [u0, v0] = applyH(H, p1[2 * i], p1[2 * i + 1]);
      rms += (u - u0) ** 2 + (v - v0) ** 2;
    }
    expect(Math.sqrt(rms / 200)).toBeLessThan(0.3);
    expect(homographyDLT(p1, p2, 3)).toBeNull();
  });

  it('transfer error, sign independence and points mapped to infinity', () => {
    const H = Float64Array.of(1, 0, 10, 0, 1, -5, 0, 0, 1);
    expect(homographyTransferErrorSq(H, 0, 0, 10, -5)).toBe(0);
    expect(homographyTransferErrorSq(H, 0, 0, 13, -1)).toBeCloseTo(25, 12);
    expect(homographyTransferErrorSq(H.map((v) => -2 * v), 0, 0, 13, -1)).toBeCloseTo(25, 12);
    const Hinf = Float64Array.of(1, 0, 0, 0, 1, 0, 1, 0, 0);
    expect(homographyTransferErrorSq(Hinf, 0, 3, 0, 0)).toBe(Infinity);
  });

  it('detects collinear minimal samples', () => {
    const p1 = Float64Array.of(0, 0, 1, 1, 2, 2, 5, 0);
    const p2 = Float64Array.of(0, 0, 1, 0, 0, 1, 1, 1);
    expect(homographySampleDegenerate(p1, p2, [0, 1, 2, 3])).toBe(true);
    expect(homographySampleDegenerate(p2, p2, [0, 1, 2, 3])).toBe(false);
    expect(homographySampleDegenerate(p2, p1, [0, 1, 2, 3])).toBe(true);
  });
});

describe('decomposeHomography', () => {
  it('the true (R, t/d, n) is among the four solutions', () => {
    const rng = new Rng(53);
    for (let k = 0; k < 200; k++) {
      const R = randomRotation(rng, 0.6);
      const nrm = randomUnitVector(rng);
      if (nrm[2] < 0) for (let i = 0; i < 3; i++) nrm[i] = -nrm[i];
      const d = rng.uniform(1, 10);
      const T = randomUnitVector(rng).map((v) => v * rng.uniform(0.1, 2));
      const H = new Float64Array(9);
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) H[3 * i + j] = R[3 * i + j] + (T[i] * nrm[j]) / d;
      const scale = rng.uniform(-3, 3) || 1; // arbitrary scale and sign
      const Hs = H.map((v) => v * scale);
      // sign disambiguation from correspondences on the plane
      const x1: number[] = [],
        x2: number[] = [];
      for (let i = 0; i < 20; i++) {
        // point on the plane in front of camera 1
        const u = rng.uniform(-0.5, 0.5),
          v = rng.uniform(-0.5, 0.5);
        const den = nrm[0] * u + nrm[1] * v + nrm[2];
        if (den <= 0.05) continue;
        const z = d / den;
        const X = [u * z, v * z, z];
        const Y = [0, 1, 2].map((r) => R[3 * r] * X[0] + R[3 * r + 1] * X[1] + R[3 * r + 2] * X[2] + T[r]);
        if (Y[2] <= 0) continue;
        x1.push(u, v);
        x2.push(Y[0] / Y[2], Y[1] / Y[2]);
      }
      const sols = decomposeHomography(Hs, x1, x2, x1.length / 2);
      expect(sols.length).toBe(4);
      let best = Infinity;
      for (const s of sols) {
        const e = rotErrDeg(s.R, R) + Math.hypot(s.t[0] - T[0] / d, s.t[1] - T[1] / d, s.t[2] - T[2] / d) + dirErrDeg(s.n, nrm);
        best = Math.min(best, e);
      }
      expect(best).toBeLessThan(1e-6);
    }
  });

  it('a pure rotation decomposes to (R, 0)', () => {
    const rng = new Rng(54);
    const R = randomRotation(rng, 0.5);
    const sols = decomposeHomography(R.map((v) => 3 * v));
    expect(sols.length).toBe(1);
    expect(rotErrDeg(sols[0].R, R)).toBeLessThan(1e-9);
    expect(Math.hypot(...sols[0].t)).toBe(0);
  });

  it('works on homographies estimated from a planar two-view scene', () => {
    const rng = new Rng(55);
    const sc = makeTwoViewScene(rng, { n: 60, depthMin: 2, depthMax: 4, baseline: 0.6, maxAngle: 0.3, planar: true });
    const H = homographyDLT(sc.x1, sc.x2, 60)!;
    const sols = decomposeHomography(H, sc.x1, sc.x2, 60);
    const best = Math.min(...sols.map((s) => rotErrDeg(s.R, sc.R) + dirErrDeg(s.t, sc.t)));
    expect(best).toBeLessThan(1e-5);
  });
});
