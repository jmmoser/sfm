import { describe, expect, it } from 'vitest';
import type { FeatureSet } from '../core/types';
import { selectFeatureIndices, selectFeatures, sortByScoreDesc, subsetFeatureSet } from './distribute';
import { TestRng } from './testdata';

describe('selectFeatureIndices', () => {
  it('returns everything sorted when under budget', () => {
    const score = new Float32Array([0.1, 0.5, 0.3, 0.5]);
    const idx = selectFeatureIndices([0, 1, 2, 3], [0, 0, 0, 0], score, 4, 10, 10, 10);
    expect(Array.from(idx)).toEqual([1, 3, 2, 0]);
    expect(Array.from(sortByScoreDesc(score, 4))).toEqual([1, 3, 2, 0]);
  });

  it('sorts NaN / infinite scores deterministically (NaN ranks as -Infinity)', () => {
    const score = new Float32Array([0.2, NaN, Infinity, 0.9, NaN, -Infinity, 0.2, Infinity]);
    expect(Array.from(sortByScoreDesc(score, 8))).toEqual([2, 7, 3, 0, 6, 1, 4, 5]);
    const idx = selectFeatureIndices([0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 0, 0, 0, 0], score, 8, 10, 10, 3, false);
    expect(Array.from(idx)).toEqual([2, 7, 3]);
  });

  it('without distribution keeps the strongest', () => {
    const rng = new TestRng(3);
    const n = 500;
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const s = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      x[i] = rng.uniform(0, 640);
      y[i] = rng.uniform(0, 480);
      s[i] = rng.next();
    }
    const idx = selectFeatureIndices(x, y, s, n, 640, 480, 50, false);
    expect(idx.length).toBe(50);
    const sorted = Array.from(s).sort((a, b) => b - a);
    for (let k = 0; k < 50; k++) expect(s[idx[k]]).toBe(sorted[k]);
  });

  it('spreads features over the image and stays sorted / unique', () => {
    const rng = new TestRng(7);
    const n = 3000;
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const s = new Float32Array(n);
    // 2000 strong features in the left 20% of the image, 1000 weak ones everywhere.
    for (let i = 0; i < n; i++) {
      const strong = i < 2000;
      x[i] = strong ? rng.uniform(0, 128) : rng.uniform(0, 640);
      y[i] = rng.uniform(0, 480);
      s[i] = strong ? 1 + rng.next() : rng.next();
    }
    const plain = selectFeatureIndices(x, y, s, n, 640, 480, 400, false);
    const dist = selectFeatureIndices(x, y, s, n, 640, 480, 400, true);
    expect(dist.length).toBe(400);
    expect(new Set(Array.from(dist)).size).toBe(400);
    for (let k = 1; k < dist.length; k++) expect(s[dist[k - 1]]).toBeGreaterThanOrEqual(s[dist[k]]);
    const right = (idx: Uint32Array) => Array.from(idx).filter((i) => x[i] > 320).length;
    expect(right(plain)).toBe(0);
    // ~half the cells lie in the right half, each with a quota of 8.
    expect(right(dist)).toBeGreaterThan(80);
  });

  it('handles points on / outside the frame border and extreme aspect ratios', () => {
    const rng = new TestRng(11);
    for (const [W, H] of [[3000, 20], [20, 3000], [640, 480]]) {
      const n = 2000;
      const x = new Float32Array(n);
      const y = new Float32Array(n);
      const s = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        // includes the exact frame edges (-0.5, W-0.5) and slightly outside
        x[i] = i % 7 === 0 ? (i % 2 ? -0.5 : W - 0.5 + 0.3) : rng.uniform(-0.5, W - 0.5);
        y[i] = i % 11 === 0 ? (i % 2 ? -0.7 : H - 0.5) : rng.uniform(-0.5, H - 0.5);
        s[i] = rng.next();
      }
      const idx = selectFeatureIndices(x, y, s, n, W, H, 500, true);
      expect(idx.length).toBe(500);
      expect(new Set(Array.from(idx)).size).toBe(500);
      for (let k = 1; k < idx.length; k++) expect(s[idx[k - 1]]).toBeGreaterThanOrEqual(s[idx[k]]);
      for (const i of idx) expect(i).toBeLessThan(n);
    }
  });

  it('selectFeatures/subsetFeatureSet copy all fields', () => {
    const n = 10;
    const fs: FeatureSet = {
      count: n,
      keypoints: new Float32Array(4 * n).map((_, i) => i),
      descriptors: new Uint8Array(128 * n).map((_, i) => i % 251),
      scores: new Float32Array(n).map((_, i) => i),
      width: 100,
      height: 50,
    };
    const sub = subsetFeatureSet(fs, [3, 7]);
    expect(sub.count).toBe(2);
    expect(Array.from(sub.keypoints.subarray(4, 8))).toEqual([28, 29, 30, 31]);
    expect(sub.descriptors[128]).toBe((7 * 128) % 251);
    const sel = selectFeatures(fs, 3, false);
    expect(Array.from(sel.scores)).toEqual([9, 8, 7]);
    expect(sel.width).toBe(100);
  });
});
