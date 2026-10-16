import { describe, expect, it } from 'vitest';
import { Rng } from '../math/random';
import { alignSim3Ransac } from './align';
import { randomRotation, rotErrDeg } from './testutil';

function makeData(rng: Rng, n: number, outlierFrac: number, noise: number, s: number, offset = 0) {
  const R = randomRotation(rng, Math.PI);
  const t = Float64Array.of(rng.uniform(-100, 100) + offset, rng.uniform(-100, 100) + offset, rng.uniform(-10, 10));
  const src = new Float64Array(3 * n),
    dst = new Float64Array(3 * n);
  const isIn = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const p = [rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-2, 2)];
    src.set(p, 3 * i);
    for (let r = 0; r < 3; r++) dst[3 * i + r] = s * (R[3 * r] * p[0] + R[3 * r + 1] * p[1] + R[3 * r + 2] * p[2]) + t[r];
    if (rng.next() < outlierFrac) {
      for (let r = 0; r < 3; r++) dst[3 * i + r] += rng.uniform(-20, 20) * s;
    } else {
      isIn[i] = 1;
      for (let r = 0; r < 3; r++) dst[3 * i + r] += noise * rng.gaussian();
    }
  }
  return { R, t, s, src, dst, isIn };
}

describe('alignSim3Ransac', () => {
  it('recovers s, R, t with 40 % outliers and noise', () => {
    for (let seed = 0; seed < 10; seed++) {
      const rng = new Rng(80 + seed);
      const d = makeData(rng, 200, 0.4, 0.02, rng.uniform(0.1, 10), seed % 2 ? 1e5 : 0);
      const res = alignSim3Ransac(d.src, d.dst, 200, 0.1, { rng: new Rng(seed) })!;
      expect(res).not.toBeNull();
      expect(Math.abs(res.s / d.s - 1)).toBeLessThan(0.01);
      expect(rotErrDeg(res.R, d.R)).toBeLessThan(0.5);
      let tp = 0,
        fp = 0,
        fn = 0;
      for (let i = 0; i < 200; i++) {
        if (res.inlierMask[i] && d.isIn[i]) tp++;
        else if (res.inlierMask[i]) fp++;
        else if (d.isIn[i]) fn++;
      }
      expect(fn).toBeLessThan(3);
      expect(fp).toBeLessThan(3);
      expect(res.numInliers).toBe(tp + fp);
      // residuals of inliers are small
      for (let i = 0; i < 200; i++) {
        if (!res.inlierMask[i]) continue;
        let e = 0;
        for (let r = 0; r < 3; r++) {
          const v =
            res.s * (res.R[3 * r] * d.src[3 * i] + res.R[3 * r + 1] * d.src[3 * i + 1] + res.R[3 * r + 2] * d.src[3 * i + 2]) +
            res.t[r] -
            d.dst[3 * i + r];
          e += v * v;
        }
        expect(Math.sqrt(e)).toBeLessThanOrEqual(0.1);
      }
    }
  });

  it('weights, fixed scale and failure cases', () => {
    const rng = new Rng(90);
    const d = makeData(rng, 100, 0.2, 0.01, 1);
    const w = new Float64Array(100).fill(1);
    const res = alignSim3Ransac(d.src, d.dst, 100, 0.05, { weights: w, fixedScale: true })!;
    expect(res.s).toBe(1);
    expect(rotErrDeg(res.R, d.R)).toBeLessThan(0.2);
    // too few points / impossible inlier count
    expect(alignSim3Ransac(d.src, d.dst, 2, 0.05)).toBeNull();
    expect(alignSim3Ransac(d.src, d.dst, 100, 0.05, { minInliers: 99 })).toBeNull();
    expect(alignSim3Ransac(d.src, d.dst, 100, 0)).toBeNull();
    // collinear sources: no model
    const src = new Float64Array(30),
      dst = new Float64Array(30);
    for (let i = 0; i < 10; i++) {
      src[3 * i] = i;
      dst[3 * i + 1] = i;
    }
    expect(alignSim3Ransac(src, dst, 10, 0.1, { maxIterations: 200 })).toBeNull();
  });
});
