import { describe, expect, it } from 'vitest';
import { projectWorld } from '../geometry/camera';
import { Reconstruction } from './reconstruction';
import { buildSyntheticDatabase, combineScenes, evaluateReconstruction, generateScene } from './synthetic';

describe('synthetic scenes', () => {
  it('are deterministic per seed', () => {
    const a = generateScene({ kind: 'orbit', numImages: 8, seed: 4, numPoints: 800 });
    const b = generateScene({ kind: 'orbit', numImages: 8, seed: 4, numPoints: 800 });
    const c = generateScene({ kind: 'orbit', numImages: 8, seed: 5, numPoints: 800 });
    expect(Array.from(a.features[3].keypoints)).toEqual(Array.from(b.features[3].keypoints));
    expect(Array.from(a.features[3].descriptors)).toEqual(Array.from(b.features[3].descriptors));
    expect(Array.from(a.pairs[2].matches)).toEqual(Array.from(b.pairs[2].matches));
    expect(Array.from(a.features[3].keypoints)).not.toEqual(Array.from(c.features[3].keypoints));
  });

  for (const kind of ['orbit', 'walk'] as const) {
    it(`${kind}: keypoints are noisy RADIAL projections of the GT points`, () => {
      const sigma = 0.7;
      const s = generateScene({ kind, numImages: 12, seed: 2, k1: 0.03, k2: -0.004, noisePx: sigma, clutterPerImage: 50 });
      const uv = new Float64Array(2);
      let sum = 0,
        n = 0,
        clutter = 0;
      let minDepth = Infinity,
        maxDepth = 0;
      for (let i = 0; i < s.imageIds.length; i++) {
        const fs = s.features[i];
        expect(fs.width).toBe(1920);
        expect(fs.keypoints.length).toBe(4 * fs.count);
        expect(fs.descriptors.length).toBe(128 * fs.count);
        expect(s.colors[i].length).toBe(3 * fs.count);
        const R = s.poses[i].R,
          t = s.poses[i].t;
        for (let f = 0; f < fs.count; f++) {
          const x = fs.keypoints[4 * f],
            y = fs.keypoints[4 * f + 1];
          expect(x >= -0.5 && x <= 1919.5 + 5 && y >= -0.5 && y <= 1439.5 + 5).toBe(true);
          const p = s.featurePoint[i][f];
          if (p < 0) {
            clutter++;
            continue;
          }
          const X = s.points.subarray(3 * p, 3 * p + 3);
          expect(projectWorld(s.camera, s.poses[i], X, uv)).toBe(true);
          const e = Math.hypot(uv[0] - x, uv[1] - y);
          expect(e).toBeLessThan(6 * sigma);
          sum += e;
          n++;
          const z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
          minDepth = Math.min(minDepth, z);
          maxDepth = Math.max(maxDepth, z);
        }
      }
      expect(clutter).toBe(50 * s.imageIds.length);
      expect(Math.abs(sum / n - sigma * Math.sqrt(Math.PI / 2))).toBeLessThan(0.03);
      if (kind === 'orbit') {
        expect(minDepth).toBeGreaterThan(0.2);
        expect(maxDepth).toBeLessThan(1.2);
      } else {
        expect(minDepth).toBeLessThan(5);
        expect(maxDepth).toBeGreaterThan(35);
      }
      expect(n / s.imageIds.length).toBeGreaterThan(300);
    });
  }

  it('descriptors are unit RootSIFT-like vectors, consistent across views of a point', () => {
    const s = generateScene({ kind: 'orbit', numImages: 6, seed: 3, numPoints: 1000 });
    const d = (i: number, f: number): Float64Array => {
      const out = new Float64Array(128);
      for (let k = 0; k < 128; k++) out[k] = s.features[i].descriptors[128 * f + k] / 512;
      return out;
    };
    const dot = (a: Float64Array, b: Float64Array): number => a.reduce((acc, v, k) => acc + v * b[k], 0);
    let same = 0,
      diff = 0,
      cnt = 0;
    const pair = s.pairs.find((p) => p.a === 0 && p.b === 1)!;
    for (let m = 0; m < pair.matches.length && cnt < 200; m += 2) {
      if (!pair.isTrue[m >> 1]) continue;
      const a = d(0, pair.matches[m]),
        b = d(1, pair.matches[m + 1]);
      expect(Math.sqrt(dot(a, a))).toBeCloseTo(1, 2);
      same += dot(a, b);
      diff += dot(a, d(1, (pair.matches[m + 1] + 7) % s.features[1].count));
      cnt++;
    }
    // true matches: distance √(2 − 2·dot) ≈ 0.4; random pairs ≈ 0.75 ⇒ Lowe ratio ≈ 0.5
    expect(same / cnt).toBeGreaterThan(0.9);
    expect(diff / cnt).toBeLessThan(0.8);
    expect(Math.sqrt(2 - (2 * same) / cnt) / Math.sqrt(2 - (2 * diff) / cnt)).toBeLessThan(0.7);
  });

  it('putative matches: one-to-one, sorted, with the requested outlier ratio', () => {
    const s = generateScene({ kind: 'walk', numImages: 15, seed: 8, outlierRatio: 0.3, matchDropRate: 0.1 });
    let out = 0,
      total = 0;
    for (const p of s.pairs) {
      const ka = s.imageIds.indexOf(p.a),
        kb = s.imageIds.indexOf(p.b);
      const seenA = new Set<number>(),
        seenB = new Set<number>();
      for (let m = 0; m < p.matches.length; m += 2) {
        const fa = p.matches[m],
          fb = p.matches[m + 1];
        if (m > 0) expect(fa).toBeGreaterThanOrEqual(p.matches[m - 2]);
        expect(seenA.has(fa) || seenB.has(fb)).toBe(false);
        seenA.add(fa);
        seenB.add(fb);
        const tru = s.featurePoint[ka][fa] >= 0 && s.featurePoint[ka][fa] === s.featurePoint[kb][fb];
        expect(p.isTrue[m >> 1]).toBe(tru ? 1 : 0);
        if (!tru) out++;
        total++;
      }
    }
    expect(Math.abs(out / total - 0.3)).toBeLessThan(0.02);
  });

  it('GT database (verify: false) stores consistent relative poses; evaluation of a GT model is exact', () => {
    const s = generateScene({ kind: 'orbit', numImages: 10, seed: 6, numPoints: 1000, descriptors: false });
    const { db, stats } = buildSyntheticDatabase(s, { verify: false });
    expect(stats.numStored).toBe(s.pairs.length);
    expect(stats.inlierOutlierRate).toBe(0);
    for (const [a, b, g] of db.pairs()) {
      expect(g.config).toBe('calibrated');
      // a GT point seen in both: X_b ∝ R X_a + t·λ — check the epipolar constraint
      const fa = g.inliers[0],
        fb = g.inliers[1];
      const p = s.featurePoint[a][fa];
      const X = s.points.subarray(3 * p, 3 * p + 3);
      const Pa = s.poses[a],
        Pb = s.poses[b];
      const xa = [0, 1, 2].map((r) => Pa.R[3 * r] * X[0] + Pa.R[3 * r + 1] * X[1] + Pa.R[3 * r + 2] * X[2] + Pa.t[r]);
      const xb = [0, 1, 2].map((r) => Pb.R[3 * r] * X[0] + Pb.R[3 * r + 1] * X[1] + Pb.R[3 * r + 2] * X[2] + Pb.t[r]);
      const Rxa = [0, 1, 2].map((r) => g.R![3 * r] * xa[0] + g.R![3 * r + 1] * xa[1] + g.R![3 * r + 2] * xa[2]);
      // xb − R xa must be parallel to t
      const v = [xb[0] - Rxa[0], xb[1] - Rxa[1], xb[2] - Rxa[2]];
      const t = g.t!;
      const cr = Math.hypot(v[1] * t[2] - v[2] * t[1], v[2] * t[0] - v[0] * t[2], v[0] * t[1] - v[1] * t[0]);
      expect(cr / Math.hypot(...v)).toBeLessThan(1e-9);
      expect(g.triAngle!).toBeGreaterThan(0);
    }
    const rec = new Reconstruction();
    const cid = rec.addCamera(s.camera);
    s.imageIds.forEach((id, k) => {
      rec.addImage(id, cid, s.features[k].count, s.features[k].keypoints);
      rec.registerImage(id, s.poses[k]);
    });
    // apply an arbitrary similarity: evaluation aligns it away
    rec.transform(0.3, Float64Array.of(0, -1, 0, 1, 0, 0, 0, 0, 1), Float64Array.of(5, 6, 7));
    const ev = evaluateReconstruction(rec, s);
    expect(ev.numRegistered).toBe(10);
    expect(ev.ateRelative).toBeLessThan(1e-12);
    expect(ev.maxRotErrorDeg).toBeLessThan(1e-6);
    expect(ev.focalError).toBe(0);
    expect(ev.sim!.s).toBeCloseTo(1 / 0.3, 9);
  });

  it('combineScenes merges disjoint scenes and rejects colliding ids', () => {
    const a = generateScene({ kind: 'orbit', numImages: 5, seed: 1, numPoints: 500, descriptors: false });
    const b = generateScene({ kind: 'orbit', numImages: 5, seed: 2, numPoints: 500, descriptors: false, idOffset: 100, offset: [10, 0, 0] });
    const c = combineScenes(a, b);
    expect(c.imageIds).toEqual([0, 1, 2, 3, 4, 100, 101, 102, 103, 104]);
    expect(c.points.length).toBe(a.points.length + b.points.length);
    // feature → point indices of b are shifted
    const fp = c.featurePoint[5].find((p) => p >= 0)!;
    expect(fp).toBeGreaterThanOrEqual(500);
    expect(() => combineScenes(a, a)).toThrow(RangeError);
    const { db } = buildSyntheticDatabase(c, { verify: false });
    for (const [x, y] of db.pairs()) expect(x < 100).toBe(y < 100);
  });
});
