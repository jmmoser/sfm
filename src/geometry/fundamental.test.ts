import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics } from '../core/types';
import { svd3 } from '../math/decomp';
import { Rng } from '../math/random';
import { createIntrinsics } from './camera';
import { essentialFromPose, sampsonDistancesSq } from './essential';
import { enforceRank2, fundamental7pt, fundamental8pt, fundamental8ptIdx, fundamentalFromEssential } from './fundamental';
import { makeTwoViewScene, matDistUpToScale } from './testutil';

function toPx(cam: CameraIntrinsics, xn: Float64Array): Float64Array {
  const o = new Float64Array(xn.length);
  for (let i = 0; i < xn.length; i += 2) {
    o[i] = cam.f * xn[i] + cam.cx;
    o[i + 1] = cam.f * xn[i + 1] + cam.cy;
  }
  return o;
}

function trueF(cam1: CameraIntrinsics, cam2: CameraIntrinsics, R: Float64Array, t: Float64Array): Float64Array {
  return fundamentalFromEssential(essentialFromPose(R, t), cam1, cam2);
}

describe('fundamental matrix', () => {
  const cam1 = createIntrinsics(1920, 1440, 1.2);
  const cam2 = { ...createIntrinsics(1920, 1440, 0.9), cx: 940, cy: 731 };

  it('fundamentalFromEssential satisfies x2ᵀ F x1 = 0 on pixels', () => {
    const rng = new Rng(41);
    const sc = makeTwoViewScene(rng, { n: 50, depthMin: 1, depthMax: 9, baseline: 0.7, maxAngle: 0.4 });
    const F = trueF(cam1, cam2, sc.R, sc.t);
    const p1 = toPx(cam1, sc.x1),
      p2 = toPx(cam2, sc.x2);
    const r = new Float64Array(50);
    sampsonDistancesSq(F, p1, p2, 50, r);
    for (let i = 0; i < 50; i++) expect(r[i]).toBeLessThan(1e-16);
    expect(Math.hypot(...F)).toBeCloseTo(1, 12);
  });

  it('fundamental7pt contains the true F in ≥ 99 % of random configurations; all solutions rank 2', () => {
    const rng = new Rng(42);
    let ok = 0;
    const N = 500;
    for (let k = 0; k < N; k++) {
      const sc = makeTwoViewScene(rng, {
        n: 7,
        depthMin: 1,
        depthMax: 10,
        baseline: rng.uniform(0.1, 2),
        maxAngle: rng.uniform(0, 0.7),
      });
      const F = trueF(cam1, cam2, sc.R, sc.t);
      const sols = fundamental7pt(toPx(cam1, sc.x1), toPx(cam2, sc.x2));
      expect(sols.length === 1 || sols.length === 2 || sols.length === 3).toBe(true);
      let best = Infinity;
      for (const S of sols) {
        best = Math.min(best, matDistUpToScale(S, F));
        const { S: sv } = svd3(S);
        expect(sv[2] / sv[0]).toBeLessThan(1e-9);
      }
      if (best < 1e-6) ok++;
    }
    expect(ok / N).toBeGreaterThanOrEqual(0.99);
  });

  it('fundamental8pt is exact on noise-free data and accurate with noise', () => {
    const rng = new Rng(43);
    for (const n of [8, 300]) {
      const sc = makeTwoViewScene(rng, { n, depthMin: 1, depthMax: 10, baseline: 0.8, maxAngle: 0.5 });
      const p1 = toPx(cam1, sc.x1),
        p2 = toPx(cam2, sc.x2);
      const F = fundamental8pt(p1, p2, n)!;
      expect(matDistUpToScale(F, trueF(cam1, cam2, sc.R, sc.t))).toBeLessThan(1e-7);
      expect(svd3(F).S[2]).toBeLessThan(1e-12);
    }
    // noisy: residuals near the noise level
    const sc = makeTwoViewScene(rng, { n: 400, depthMin: 1, depthMax: 10, baseline: 0.8, maxAngle: 0.5 });
    const p1 = toPx(cam1, sc.x1),
      p2 = toPx(cam2, sc.x2);
    for (let i = 0; i < p1.length; i++) {
      p1[i] += rng.gaussian();
      p2[i] += rng.gaussian();
    }
    const F = fundamental8pt(p1, p2, 400)!;
    const r = new Float64Array(400);
    sampsonDistancesSq(F, p1, p2, 400, r);
    const rms = Math.sqrt(r.reduce((a, b) => a + b, 0) / 400);
    expect(rms).toBeLessThan(1.2); // σ = 1 px per coordinate → Sampson rms ≈ 1
    expect(fundamental8pt(p1, p2, 7)).toBeNull();
    // index subset
    const idx = new Uint32Array([...Array(200).keys()].map((i) => 2 * i));
    const Fi = fundamental8ptIdx(p1, p2, idx, 200)!;
    expect(matDistUpToScale(Fi, trueF(cam1, cam2, sc.R, sc.t))).toBeLessThan(0.05);
  });

  it('enforceRank2 zeroes the smallest singular value only', () => {
    const M = Float64Array.of(3, 0, 0, 0, 2, 0, 0, 0, 1);
    expect(Array.from(enforceRank2(M)!)).toEqual([3, 0, 0, 0, 2, 0, 0, 0, 0].map((v) => expect.closeTo(v, 12)));
    expect(enforceRank2(new Float64Array(9))).toBeNull();
  });

  it('degenerate input (coincident points) yields no solution', () => {
    const p = new Float64Array(14).fill(5);
    expect(fundamental7pt(p, p).length).toBe(0);
    expect(fundamental8pt(new Float64Array(16).fill(1), new Float64Array(16).fill(1), 8)).toBeNull();
  });
});
