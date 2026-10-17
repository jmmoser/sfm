import { describe, expect, it } from 'vitest';
import type { FeatureSet, FrameInfo, TwoViewGeometry } from '../core/types';
import { Rng } from '../math/random';
import { rotvecToMat3 } from '../math/rotation';
import { FeatureDatabase, MAX_IMAGE_ID, invertTwoView, pairIds, pairKey } from './database';

function frame(id: number): FrameInfo {
  return {
    id,
    t: id,
    width: 640,
    height: 480,
    sharpness: 1,
    R_cam_device: Float64Array.of(1, 0, 0, 0, -1, 0, 0, 0, -1),
    gravityCam: null,
    heading: null,
    geo: null,
  };
}

function feats(n: number): FeatureSet {
  return {
    count: n,
    keypoints: new Float32Array(4 * n),
    descriptors: new Uint8Array(128 * n),
    scores: new Float32Array(n),
    width: 640,
    height: 480,
  };
}

describe('pairKey', () => {
  it('is symmetric, unique and invertible', () => {
    const rng = new Rng(1);
    const seen = new Map<number, string>();
    for (let k = 0; k < 2000; k++) {
      const a = rng.int(MAX_IMAGE_ID),
        b = rng.int(MAX_IMAGE_ID);
      if (a === b) continue;
      const key = pairKey(a, b);
      expect(pairKey(b, a)).toBe(key);
      expect(pairIds(key)).toEqual([Math.min(a, b), Math.max(a, b)]);
      const prev = seen.get(key);
      if (prev) expect(prev).toBe(`${Math.min(a, b)},${Math.max(a, b)}`);
      seen.set(key, `${Math.min(a, b)},${Math.max(a, b)}`);
    }
    expect(pairIds(pairKey(MAX_IMAGE_ID - 1, MAX_IMAGE_ID - 2))).toEqual([MAX_IMAGE_ID - 2, MAX_IMAGE_ID - 1]);
    expect(pairKey(0, 1)).not.toBe(pairKey(1, 2));
  });
});

describe('FeatureDatabase', () => {
  it('stores frames, features, colours; frameIds ascending', () => {
    const db = new FeatureDatabase();
    for (const id of [5, 1, 9, 3]) db.addFrame(frame(id), feats(10), id === 9 ? new Uint8Array(30) : undefined);
    expect(db.frameIds()).toEqual([1, 3, 5, 9]);
    expect(db.colors.has(9)).toBe(true);
    expect(db.colors.has(1)).toBe(false);
    expect(() => db.addFrame(frame(2), feats(10), new Uint8Array(5))).toThrow(RangeError);
    expect(() => db.addFrame(frame(-1), feats(1))).toThrow(RangeError);
    expect(() => db.addFrame(frame(MAX_IMAGE_ID), feats(1))).toThrow(RangeError);
  });

  it('orients two-view geometries: inliers swapped, pose inverted, E/F transposed, H inverted', () => {
    const db = new FeatureDatabase();
    db.addFrame(frame(2), feats(10));
    db.addFrame(frame(7), feats(10));
    const R = rotvecToMat3(Float64Array.of(0.1, -0.2, 0.05));
    const t = Float64Array.of(0.6, 0, 0.8);
    const H = Float64Array.of(1, 0.1, 3, 0.02, 0.9, -2, 0.001, 0, 1);
    const E = Float64Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9);
    // given oriented 7 → 2 (larger id first): stored canonically as 2 → 7
    const g: TwoViewGeometry = { config: 'calibrated', inliers: Uint32Array.of(1, 2, 3, 4), R, t, H, E, F: E, triAngle: 0.1 };
    db.setTwoView(7, 2, g);
    expect(db.hasPair(2, 7)).toBe(true);
    expect(db.hasPair(7, 2)).toBe(true);
    const back = db.getTwoView(7, 2)!;
    expect(Array.from(back.inliers)).toEqual([1, 2, 3, 4]);
    for (let k = 0; k < 9; k++) expect(back.R![k]).toBeCloseTo(R[k], 12);
    for (let k = 0; k < 3; k++) expect(back.t![k]).toBeCloseTo(t[k], 12);
    for (let k = 0; k < 9; k++) expect(back.H![k]).toBeCloseTo(H[k], 9);
    const fwd = db.getTwoView(2, 7)!;
    expect(Array.from(fwd.inliers)).toEqual([2, 1, 4, 3]);
    // x1 = R x7…: for a point X in camera 7, camera 2 coordinates = R X + t; the
    // canonical geometry must map camera 2 coordinates back: X = Rᵀ (x2 − t)
    const X7 = Float64Array.of(0.3, -0.4, 5);
    const X2 = Float64Array.of(
      R[0] * X7[0] + R[1] * X7[1] + R[2] * X7[2] + t[0],
      R[3] * X7[0] + R[4] * X7[1] + R[5] * X7[2] + t[1],
      R[6] * X7[0] + R[7] * X7[1] + R[8] * X7[2] + t[2],
    );
    const Ri = fwd.R!,
      ti = fwd.t!;
    for (let r = 0; r < 3; r++) expect(Ri[3 * r] * X2[0] + Ri[3 * r + 1] * X2[1] + Ri[3 * r + 2] * X2[2] + ti[r]).toBeCloseTo(X7[r], 10);
    expect(Math.hypot(ti[0], ti[1], ti[2])).toBeCloseTo(1, 12);
    expect(fwd.E![1]).toBe(E[3]); // transposed
    expect(fwd.triAngle).toBe(0.1);
    expect(fwd.config).toBe('calibrated');
    // cached inverse is stable
    expect(db.getTwoView(2, 7)).toBe(fwd);
    // pairs() yields canonical orientation
    const list = [...db.pairs()];
    expect(list.length).toBe(1);
    expect(list[0][0]).toBe(2);
    expect(list[0][1]).toBe(7);
    expect(Array.from(list[0][2].inliers)).toEqual([2, 1, 4, 3]);
    // invertTwoView twice = identity
    const twice = invertTwoView(invertTwoView(g));
    for (let k = 0; k < 9; k++) expect(twice.R![k]).toBeCloseTo(R[k], 12);
    for (let k = 0; k < 9; k++) expect(twice.H![k]).toBeCloseTo(H[k], 9);
  });

  it('replaces, deletes pairs and removes frames with their pairs', () => {
    const db = new FeatureDatabase();
    for (const id of [0, 1, 2]) db.addFrame(frame(id), feats(5));
    db.setTwoView(0, 1, { config: 'calibrated', inliers: Uint32Array.of(0, 0) });
    db.setTwoView(1, 2, { config: 'planar', inliers: Uint32Array.of(1, 1) });
    db.setTwoView(1, 0, { config: 'uncalibrated', inliers: Uint32Array.of(3, 4) });
    expect(db.numPairs).toBe(2);
    expect(db.getTwoView(0, 1)!.config).toBe('uncalibrated');
    expect(Array.from(db.getTwoView(0, 1)!.inliers)).toEqual([4, 3]);
    expect(() => db.setTwoView(1, 1, { config: 'calibrated', inliers: new Uint32Array(0) })).toThrow();
    db.removeFrame(1);
    expect(db.numPairs).toBe(0);
    expect(db.frameIds()).toEqual([0, 2]);
    expect(db.getTwoView(0, 1)).toBeUndefined();
    db.setTwoView(0, 2, { config: 'calibrated', inliers: Uint32Array.of(1, 1) });
    db.deletePair(2, 0);
    expect(db.hasPair(0, 2)).toBe(false);
  });
});
