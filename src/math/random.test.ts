import { describe, expect, it } from 'vitest';
import { Rng } from './random';

describe('Rng', () => {
  it('is deterministic per seed; different seeds give different streams', () => {
    const a = new Rng(42),
      b = new Rng(42),
      c = new Rng(43);
    const sa: number[] = [],
      sb: number[] = [],
      sc: number[] = [];
    for (let i = 0; i < 100; i++) {
      sa.push(a.nextU32());
      sb.push(b.nextU32());
      sc.push(c.nextU32());
    }
    expect(sa).toEqual(sb);
    expect(sa).not.toEqual(sc);
    // Default seed is fixed.
    expect(new Rng().next()).toBe(new Rng().next());
    // Re-seeding restarts the stream and clears the cached Gaussian.
    const r = new Rng(7);
    const g1 = r.gaussian();
    r.seed(7);
    expect(r.gaussian()).toBe(g1);
    // Non-integer / huge / non-finite seeds are accepted.
    expect(Number.isFinite(new Rng(1.5).next())).toBe(true);
    expect(Number.isFinite(new Rng(2 ** 40 + 3).next())).toBe(true);
    expect(Number.isFinite(new Rng(NaN).next())).toBe(true);
    expect(new Rng(2 ** 40 + 3).next()).not.toBe(new Rng(3).next());
  });

  it('next() is uniform on [0, 1)', () => {
    const r = new Rng(1);
    const N = 200000;
    let s = 0,
      s2 = 0,
      min = 1,
      max = 0;
    const bins = new Uint32Array(20);
    for (let i = 0; i < N; i++) {
      const x = r.next();
      s += x;
      s2 += x * x;
      min = Math.min(min, x);
      max = Math.max(max, x);
      bins[Math.floor(x * 20)]++;
    }
    expect(min).toBeGreaterThanOrEqual(0);
    expect(max).toBeLessThan(1);
    const mean = s / N,
      v = s2 / N - mean * mean;
    expect(Math.abs(mean - 0.5)).toBeLessThan(0.005);
    expect(Math.abs(v - 1 / 12)).toBeLessThan(0.002);
    // χ² with 19 dof: 99.9th percentile ≈ 43.8
    let chi = 0;
    for (const c of bins) chi += (c - N / 20) ** 2 / (N / 20);
    expect(chi).toBeLessThan(43.8);
    const d = r.nextDouble();
    expect(d).toBeGreaterThanOrEqual(0);
    expect(d).toBeLessThan(1);
  });

  it('int(n) covers [0, n) uniformly; edge cases', () => {
    const r = new Rng(2);
    for (const n of [1, 2, 3, 7, 10, 1000]) {
      const counts = new Uint32Array(n);
      const N = n * 2000;
      let bad = 0;
      for (let i = 0; i < N; i++) {
        const k = r.int(n);
        if (!Number.isInteger(k) || k < 0 || k >= n) bad++;
        else counts[k]++;
      }
      expect(bad).toBe(0);
      if (n > 1) {
        let chi = 0;
        for (const c of counts) chi += (c - 2000) ** 2 / 2000;
        // mean n−1, sd sqrt(2(n−1)); allow 5 sd
        expect(chi).toBeLessThan(n - 1 + 5 * Math.sqrt(2 * (n - 1)));
      }
    }
    expect(r.int(0)).toBe(0);
    expect(r.int(-5)).toBe(0);
    const big = r.int(2 ** 32);
    expect(big).toBeLessThan(2 ** 32);
    const u = r.uniform(-3, 5);
    expect(u).toBeGreaterThanOrEqual(-3);
    expect(u).toBeLessThan(5);
  });

  it('gaussian() has mean 0, variance 1, normal tails', () => {
    const r = new Rng(3);
    const N = 200000;
    let s = 0,
      s2 = 0,
      s4 = 0,
      beyond2 = 0;
    for (let i = 0; i < N; i++) {
      const g = r.gaussian();
      s += g;
      s2 += g * g;
      s4 += g ** 4;
      if (Math.abs(g) > 2) beyond2++;
    }
    const mean = s / N,
      v = s2 / N - mean * mean,
      kurt = s4 / N / (v * v);
    expect(Math.abs(mean)).toBeLessThan(0.01);
    expect(Math.abs(v - 1)).toBeLessThan(0.01);
    expect(Math.abs(kurt - 3)).toBeLessThan(0.05);
    expect(Math.abs(beyond2 / N - 0.0455)).toBeLessThan(0.003);
  });

  it('sample(n, k): distinct, in range, uniform inclusion, both algorithms', () => {
    const r = new Rng(4);
    const cases: [number, number][] = [
      [10, 0],
      [1, 1],
      [5, 5],
      [10, 3],
      [100, 8],
      [1000, 30], // Floyd (k² ≤ n)
      [50, 20], // Fisher–Yates
      [30, 30],
      [100000, 5],
    ];
    for (const [n, k] of cases) {
      const counts = new Uint32Array(n);
      const trials = n <= 100 ? 4000 : 200;
      for (let t = 0; t < trials; t++) {
        const s = r.sample(n, k);
        if (s.length !== k) expect(s.length).toBe(k);
        const seen = new Set<number>();
        for (const v of s) {
          if (v >= n || seen.has(v)) expect(`${v} invalid or duplicate`).toBe('');
          seen.add(v);
          counts[v]++;
        }
      }
      if (n <= 100 && k > 0 && k < n) {
        const expected = (trials * k) / n;
        for (const c of counts) expect(Math.abs(c - expected)).toBeLessThan(6 * Math.sqrt(expected));
      }
    }
    expect(() => r.sample(3, 4)).toThrow(RangeError);
    // Reuse of out buffer (longer than k): first k entries written.
    const out = new Uint32Array(10).fill(99);
    const res = r.sample(5, 3, out);
    expect(res).toBe(out);
    expect(out[3]).toBe(99);
  });

  it('sample of small subsets is uniform over all k-subsets (Floyd)', () => {
    const r = new Rng(5);
    const n = 6,
      k = 2;
    const counts = new Map<string, number>();
    const N = 30000;
    for (let i = 0; i < N; i++) {
      const s = Array.from(r.sample(n, k)).sort((a, b) => a - b);
      const key = s.join(',');
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    expect(counts.size).toBe(15);
    for (const c of counts.values()) expect(Math.abs(c - N / 15)).toBeLessThan(6 * Math.sqrt(N / 15));
  });

  it('shuffle is a uniform permutation (arrays and Uint32Array)', () => {
    const r = new Rng(6);
    const pos = Array.from({ length: 4 }, () => new Uint32Array(4));
    const N = 40000;
    for (let i = 0; i < N; i++) {
      const a = [0, 1, 2, 3];
      r.shuffle(a);
      if (a[0] + a[1] + a[2] + a[3] !== 6 || new Set(a).size !== 4) expect(a).toEqual([0, 1, 2, 3]);
      for (let j = 0; j < 4; j++) pos[a[j]][j]++;
    }
    for (const row of pos) for (const c of row) expect(Math.abs(c - N / 4)).toBeLessThan(6 * Math.sqrt(N / 4));
    const u = new Uint32Array([5, 6, 7, 8, 9]);
    r.shuffle(u);
    expect(Array.from(u).sort()).toEqual([5, 6, 7, 8, 9]);
  });
});
