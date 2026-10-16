import { describe, expect, it } from 'vitest';
import { Rng } from '../math/random';
import { maskToIndices, ransac, ransacIterations, type RansacEstimator } from './ransac';

/** 2-D line a x + b y + c = 0 (a² + b² = 1). */
type Line = [number, number, number];

class LineEstimator implements RansacEstimator<Line> {
  sampleSize = 2;
  numData: number;
  fits = 0;
  refits = 0;
  constructor(private p: Float64Array) {
    this.numData = p.length / 2;
  }
  isDegenerate(s: Uint32Array): boolean {
    const p = this.p;
    return p[2 * s[0]] === p[2 * s[1]] && p[2 * s[0] + 1] === p[2 * s[1] + 1];
  }
  fit(s: Uint32Array, models: Line[]): number {
    this.fits++;
    const p = this.p;
    const x0 = p[2 * s[0]],
      y0 = p[2 * s[0] + 1],
      x1 = p[2 * s[1]],
      y1 = p[2 * s[1] + 1];
    let a = y1 - y0,
      b = x0 - x1;
    const n = Math.hypot(a, b);
    if (n === 0) return 0;
    a /= n;
    b /= n;
    models[0] = [a, b, -(a * x0 + b * y0)];
    return 1;
  }
  residualsSq(m: Line, out: Float64Array): void {
    const p = this.p;
    for (let i = 0; i < this.numData; i++) {
      const r = m[0] * p[2 * i] + m[1] * p[2 * i + 1] + m[2];
      out[i] = r * r;
    }
  }
  refit(inl: Uint32Array, count: number): Line | null {
    this.refits++;
    // total least squares
    let mx = 0,
      my = 0;
    for (let k = 0; k < count; k++) {
      mx += this.p[2 * inl[k]];
      my += this.p[2 * inl[k] + 1];
    }
    mx /= count;
    my /= count;
    let sxx = 0,
      sxy = 0,
      syy = 0;
    for (let k = 0; k < count; k++) {
      const dx = this.p[2 * inl[k]] - mx,
        dy = this.p[2 * inl[k] + 1] - my;
      sxx += dx * dx;
      sxy += dx * dy;
      syy += dy * dy;
    }
    // normal = eigenvector of the smallest eigenvalue of the scatter matrix
    const th = 0.5 * Math.atan2(2 * sxy, sxx - syy); // direction of largest spread
    const a = -Math.sin(th),
      b = Math.cos(th);
    return [a, b, -(a * mx + b * my)];
  }
}

function makeLineData(rng: Rng, n: number, outlierFrac: number, sigma: number) {
  const pts = new Float64Array(2 * n);
  const truth = new Uint8Array(n);
  // y = 0.5 x + 2  →  0.5 x − y + 2 = 0
  for (let i = 0; i < n; i++) {
    if (rng.next() < outlierFrac) {
      pts[2 * i] = rng.uniform(-50, 50);
      pts[2 * i + 1] = rng.uniform(-50, 50);
    } else {
      const x = rng.uniform(-50, 50);
      pts[2 * i] = x + sigma * rng.gaussian();
      pts[2 * i + 1] = 0.5 * x + 2 + sigma * rng.gaussian();
      truth[i] = 1;
    }
  }
  return { pts, truth };
}

/** Max of the normal-direction error (rad) and the offset error / 10 (distance units). */
function lineError(m: Line): number {
  // compare to normalised (0.5, −1, 2)/|(0.5, −1)|
  const n = Math.hypot(0.5, 1);
  const t = [0.5 / n, -1 / n, 2 / n];
  const s = Math.sign(m[0] * t[0] + m[1] * t[1]);
  const ang = Math.abs(Math.atan2(s * (m[0] * t[1] - m[1] * t[0]), s * (m[0] * t[0] + m[1] * t[1])));
  return Math.max(ang, Math.abs(s * m[2] - t[2]) / 10);
}

describe('ransacIterations', () => {
  it('matches log(1−p)/log(1−wˢ)', () => {
    expect(ransacIterations(0.5, 2, 0.99, 1e9)).toBe(Math.ceil(Math.log(0.01) / Math.log(0.75)));
    expect(ransacIterations(0.6, 5, 0.9999, 1e9)).toBe(Math.ceil(Math.log(1e-4) / Math.log(1 - 0.6 ** 5)));
    expect(ransacIterations(1, 5, 0.9999, 1000)).toBe(0);
    expect(ransacIterations(0, 5, 0.9999, 1000)).toBe(1000);
    expect(ransacIterations(1e-9, 7, 0.9999, 1234)).toBe(1234);
  });
});

describe('ransac (LO-MSAC)', () => {
  it('recovers a line with 60 % outliers and classifies inliers', () => {
    const rng = new Rng(10);
    const { pts, truth } = makeLineData(rng, 1000, 0.6, 0.3);
    const est = new LineEstimator(pts);
    const res = ransac(est, { thresholdSq: 1.0, rng: new Rng(1) });
    expect(res.model).not.toBeNull();
    expect(lineError(res.model!)).toBeLessThan(0.005); // σ = 0.3, ~400 inliers
    let tp = 0,
      fp = 0,
      fn = 0;
    for (let i = 0; i < truth.length; i++) {
      if (res.inlierMask[i] && truth[i]) tp++;
      else if (res.inlierMask[i]) fp++;
      else if (truth[i]) fn++;
    }
    expect(tp / (tp + fn)).toBeGreaterThan(0.98); // recall (3.3σ threshold)
    expect(fp).toBeLessThan(0.05 * tp);
    expect(res.numInliers).toBe(tp + fp);
    expect(maskToIndices(res.inlierMask).length).toBe(res.numInliers);
    // adaptive termination well below the cap; LO refits were run
    expect(res.iterations).toBeLessThan(200);
    expect(est.refits).toBeGreaterThan(0);
    expect(res.score).toBeGreaterThan(0);
  });

  it('is deterministic for a given seed', () => {
    const rng = new Rng(11);
    const { pts } = makeLineData(rng, 500, 0.5, 0.5);
    const a = ransac(new LineEstimator(pts), { thresholdSq: 2, rng: new Rng(42) });
    const b = ransac(new LineEstimator(pts), { thresholdSq: 2, rng: new Rng(42) });
    expect(a.model).toEqual(b.model);
    expect(a.iterations).toBe(b.iterations);
    expect(Array.from(a.inlierMask)).toEqual(Array.from(b.inlierMask));
    // default rng is also deterministic
    const c = ransac(new LineEstimator(pts), { thresholdSq: 2 });
    const d = ransac(new LineEstimator(pts), { thresholdSq: 2 });
    expect(c.model).toEqual(d.model);
  });

  it('local optimisation never worsens the MSAC score and improves accuracy', () => {
    let better = 0;
    let errLO = 0,
      errPlain = 0;
    for (let trial = 0; trial < 30; trial++) {
      const rng = new Rng(100 + trial);
      const { pts } = makeLineData(rng, 400, 0.4, 1.0);
      const lo = ransac(new LineEstimator(pts), { thresholdSq: 9, rng: new Rng(trial) });
      const plain = ransac(new LineEstimator(pts), { thresholdSq: 9, rng: new Rng(trial), localOptimization: false });
      expect(lo.score).toBeLessThanOrEqual(plain.score + 1e-9);
      if (lo.score < plain.score) better++;
      errLO += lineError(lo.model!);
      errPlain += lineError(plain.model!);
    }
    expect(better).toBeGreaterThan(20);
    expect(errLO).toBeLessThan(errPlain);
  });

  it('honours maxIterations, minIterations, minInlierRatio and initialModels', () => {
    const rng = new Rng(12);
    const { pts } = makeLineData(rng, 300, 0.95, 0.3);
    const capped = ransac(new LineEstimator(pts), { thresholdSq: 1, maxIterations: 37, rng: new Rng(1) });
    expect(capped.iterations).toBeLessThanOrEqual(37);
    const ratioCap = ransac(new LineEstimator(pts), { thresholdSq: 1, minInlierRatio: 0.5, rng: new Rng(1) });
    expect(ratioCap.iterations).toBeLessThanOrEqual(ransacIterations(0.5, 2, 0.9999, 10000));
    const { pts: clean } = makeLineData(rng, 300, 0.0, 0.1);
    const minIt = ransac(new LineEstimator(clean), { thresholdSq: 1, minIterations: 50, rng: new Rng(1) });
    expect(minIt.iterations).toBe(50);
    // a perfect seed terminates almost immediately
    const n = Math.hypot(0.5, 1);
    const seeded = ransac(new LineEstimator(clean), {
      thresholdSq: 1,
      rng: new Rng(1),
      initialModels: [[0.5 / n, -1 / n, 2 / n]],
    });
    expect(seeded.iterations).toBeLessThanOrEqual(2);
    expect(seeded.numInliers).toBeGreaterThan(290);
  });

  it('handles too little data and all-degenerate samples', () => {
    const one = new LineEstimator(Float64Array.of(1, 2));
    const r = ransac(one, { thresholdSq: 1 });
    expect(r.model).toBeNull();
    expect(r.numInliers).toBe(0);
    expect(r.score).toBe(Infinity);
    const same = new Float64Array(20).fill(3); // all points identical → every sample degenerate
    const r2 = ransac(new LineEstimator(same), { thresholdSq: 1, maxIterations: 100 });
    expect(r2.model).toBeNull();
    expect(r2.iterations).toBe(100);
    expect(r2.inlierMask.every((v) => v === 0)).toBe(true);
  });

  it('treats NaN / Infinity residuals as outliers', () => {
    const rng = new Rng(13);
    const { pts } = makeLineData(rng, 200, 0.2, 0.2);
    pts[0] = NaN;
    pts[3] = Infinity;
    const res = ransac(new LineEstimator(pts), { thresholdSq: 1, rng: new Rng(3) });
    expect(res.model).not.toBeNull();
    expect(res.inlierMask[0]).toBe(0);
    expect(res.inlierMask[1]).toBe(0);
    expect(Number.isFinite(res.score)).toBe(true);
  });
});
