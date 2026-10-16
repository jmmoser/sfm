import { describe, expect, it } from 'vitest';
import { Rng } from './random';
import { mad, mean, median, percentile, quickselect, rms, variance } from './stats';

function sortedCopy(v: ArrayLike<number>): number[] {
  return Array.from(v).sort((a, b) => a - b);
}

/** Reference percentile (NumPy "linear"). */
function refPercentile(v: ArrayLike<number>, p: number): number {
  const s = sortedCopy(v);
  const pos = (Math.min(100, Math.max(0, p)) / 100) * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo] + (pos - lo) * (s[hi] - s[lo]);
}

describe('stats', () => {
  it('median: odd/even, known values, non-destructive', () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([5])).toBe(5);
    expect(median([])).toBeNaN();
    expect(median([2, 2, 2, 2])).toBe(2);
    const v = new Float64Array([9, -1, 4, 4, 0]);
    const copy = v.slice();
    expect(median(v)).toBe(4);
    expect(Array.from(v)).toEqual(Array.from(copy));
  });

  it('median/percentile agree with sort-based references on random data with duplicates', () => {
    const rng = new Rng(1);
    for (let t = 0; t < 300; t++) {
      const n = 1 + rng.int(200);
      const v = new Float64Array(n);
      for (let i = 0; i < n; i++) v[i] = rng.next() < 0.3 ? rng.int(5) : rng.gaussian() * 10;
      const s = sortedCopy(v);
      const med = n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
      expect(median(v)).toBe(med);
      for (const p of [0, 1, 10, 25, 33.3, 50, 75, 90, 99, 100]) {
        expect(percentile(v, p)).toBeCloseTo(refPercentile(v, p), 12);
      }
      expect(percentile(v, -5)).toBe(s[0]);
      expect(percentile(v, 500)).toBe(s[n - 1]);
    }
    // plain number[] input
    expect(percentile([1, 2, 3, 4], 50)).toBe(2.5);
    expect(percentile([10, 20], 25)).toBe(12.5);
  });

  it('quickselect: k-th order statistic and partition property', () => {
    const rng = new Rng(2);
    for (let t = 0; t < 500; t++) {
      const n = 1 + rng.int(100);
      const a: number[] = [];
      for (let i = 0; i < n; i++) a.push(rng.int(10));
      const s = sortedCopy(a);
      const k = rng.int(n);
      const arr = rng.next() < 0.5 ? a.slice() : Float64Array.from(a);
      const x = quickselect(arr, k);
      expect(x).toBe(s[k]);
      expect(arr[k]).toBe(s[k]);
      for (let i = 0; i < k; i++) expect(arr[i]).toBeLessThanOrEqual(x);
      for (let i = k + 1; i < n; i++) expect(arr[i]).toBeGreaterThanOrEqual(x);
      expect(sortedCopy(arr)).toEqual(s); // a permutation
    }
    // sorted / reverse-sorted / constant inputs
    for (const base of [
      Array.from({ length: 1000 }, (_, i) => i),
      Array.from({ length: 1000 }, (_, i) => 1000 - i),
      Array.from({ length: 1000 }, () => 7),
    ]) {
      const a = base.slice();
      expect(quickselect(a, 500)).toBe(sortedCopy(base)[500]);
    }
  });

  it('mean (compensated), variance, rms', () => {
    expect(mean([1, 2, 3, 4])).toBe(2.5);
    expect(mean([])).toBeNaN();
    // Compensation: 1e16 + many 1s − 1e16
    const v = [1e16];
    for (let i = 0; i < 1000; i++) v.push(1);
    v.push(-1e16);
    expect(mean(v)).toBeCloseTo(1000 / 1002, 12);
    expect(variance([1, 2, 3, 4])).toBeCloseTo(1.25, 14);
    expect(variance([1, 2, 3, 4], 1)).toBeCloseTo(5 / 3, 14);
    expect(variance([1], 1)).toBeNaN();
    // Large offset: two-pass is accurate.
    expect(variance([1e9 + 1, 1e9 + 2, 1e9 + 3, 1e9 + 4])).toBeCloseTo(1.25, 6);
    expect(rms([3, 4])).toBeCloseTo(Math.sqrt(12.5), 14);
    expect(rms([])).toBeNaN();
  });

  it('mad: unscaled median absolute deviation, robust to outliers, non-destructive', () => {
    expect(mad([1, 1, 2, 2, 4, 6, 9])).toBe(1);
    expect(mad([])).toBeNaN();
    expect(mad([5])).toBe(0);
    const rng = new Rng(3);
    const v = new Float64Array(20001);
    for (let i = 0; i < v.length; i++) v[i] = rng.gaussian() * 2 + 10;
    for (let i = 0; i < 2000; i++) v[i] = 1e6; // 10% outliers
    const copy = v.slice();
    const m = mad(v);
    expect(Array.from(v)).toEqual(Array.from(copy));
    // σ ≈ 1.4826 · MAD (outliers inflate it only mildly)
    expect(Math.abs(1.4826 * m - 2)).toBeLessThan(0.35);
    // Reference implementation
    const med = median(v);
    expect(m).toBe(median(Array.from(v, (x) => Math.abs(x - med))));
  });

  it('scratch buffer grows and remains correct across calls of different sizes', () => {
    const rng = new Rng(4);
    for (const n of [3, 1000, 5, 100000, 7, 300000, 9]) {
      const v = Array.from({ length: n }, () => rng.gaussian());
      expect(median(v)).toBe(refPercentile(v, 50));
    }
  });
});
