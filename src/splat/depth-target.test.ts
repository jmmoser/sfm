import { describe, expect, it } from 'vitest';
import { Rng } from '../math/random';
import { geometryLossCPU, octDecode, octEncode, packDepthTarget, unpackTargetPixel, type DepthMapLike } from './depth-target';

function randomMap(W: number, H: number, seed: number): DepthMapLike {
  const rng = new Rng(seed);
  const depth = new Float32Array(W * H);
  const normal = new Float32Array(3 * W * H);
  const cost = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    depth[i] = i % 7 === 3 ? 0 : rng.uniform(0.3, 4);
    const n = [rng.gaussian(), rng.gaussian(), -Math.abs(rng.gaussian()) - 0.2];
    const l = Math.hypot(n[0], n[1], n[2]);
    normal.set(depth[i] > 0 ? n.map((v) => v / l) : [0, 0, 0], 3 * i);
    cost[i] = rng.uniform(0, 1.2);
  }
  return { width: W, height: H, depth, normal, cost };
}

describe('splat depth targets', () => {
  it('octahedral normal encoding round-trips (both hemispheres, axes)', () => {
    const rng = new Rng(3);
    const dirs: number[][] = [
      [0, 0, 1],
      [0, 0, -1],
      [1, 0, 0],
      [0, -1, 0],
    ];
    for (let k = 0; k < 200; k++) dirs.push([rng.gaussian(), rng.gaussian(), rng.gaussian()]);
    for (const d of dirs) {
      const l = Math.hypot(d[0], d[1], d[2]);
      const n = d.map((v) => v / l);
      const [u, v] = octEncode(n[0], n[1], n[2]);
      expect(Math.abs(u) + Math.abs(v)).toBeLessThanOrEqual(1 + 1e-12 + (n[2] < 0 ? 1 : 0));
      const r = octDecode(u, v);
      for (let q = 0; q < 3; q++) expect(r[q]).toBeCloseTo(n[q], 9);
    }
  });

  it('packs a depth map for the same camera at half size (nearest pixel, f16 / snorm16)', () => {
    const W = 40;
    const H = 30;
    const map = randomMap(W, H, 5);
    const cam = { width: W, height: H, f: 50, cx: 19.3, cy: 14.6 };
    const t = packDepthTarget(map, cam, cam, { factor: 2 });
    expect([t.width, t.height, t.factor]).toEqual([20, 15, 2]);
    let checked = 0;
    for (let j = 0; j < t.height; j++)
      for (let i = 0; i < t.width; i++) {
        // the centre of the 2×2 block maps to source pixel (2i + 0.5, 2j + 0.5) → rounds to 2i + 1 (ties up)
        const u = Math.round(2 * i + 0.5);
        const v = Math.round(2 * j + 0.5);
        const p = v * W + u;
        const px = unpackTargetPixel(t.data, j * t.width + i);
        const w = Math.max(0, Math.min(1, 1 - map.cost[p]));
        if (!(map.depth[p] > 0) || !(w > 0)) {
          expect(px.w).toBe(0);
          continue;
        }
        expect(Math.abs(px.invZ * map.depth[p] - 1)).toBeLessThan(1e-3); // f16
        expect(Math.abs(px.w - w)).toBeLessThan(1e-3);
        for (let q = 0; q < 3; q++) expect(Math.abs(px.n[q] - map.normal[3 * p + q])).toBeLessThan(1e-4);
        checked++;
      }
    expect(checked).toBeGreaterThan(100);
    expect(t.valid).toBe(checked);
  });

  it('resamples between cameras of different sizes along the rays', () => {
    // depth map of a 2× larger camera of the same pinhole (dense size 80×60 vs view 40×30)
    const big = { width: 80, height: 60, f: 100, cx: 39.5, cy: 29.5 };
    const view = { width: 40, height: 30, f: 50, cx: 19.5, cy: 14.5 };
    const map = randomMap(80, 60, 6);
    const t = packDepthTarget(map, big, view, { factor: 1 });
    expect([t.width, t.height]).toEqual([40, 30]);
    // view pixel (i, j) → big pixel 2i + 0.5 → rounds to 2i + 1
    const px = unpackTargetPixel(t.data, 7 * 40 + 9);
    const p = 15 * 80 + 19;
    if (map.depth[p] > 0 && map.cost[p] < 1) expect(Math.abs(px.invZ * map.depth[p] - 1)).toBeLessThan(1e-3);
  });

  it('geometry loss gradient matches finite differences', () => {
    const W = 24;
    const H = 18;
    const map = randomMap(W, H, 9);
    const cam = { width: W, height: H, f: 30, cx: 11.5, cy: 8.5 };
    const t = packDepthTarget(map, cam, cam, { factor: 2 });
    const rng = new Rng(11);
    // a render at level 1 (W × H): inverse depth near the target, normals near it
    const geo = new Float64Array(4 * W * H).map((_, i) => (i % 4 === 0 ? rng.uniform(0.2, 3) : rng.uniform(-1, 1)));
    const alpha = new Float64Array(W * H).map((_, i) => (i % 11 === 5 ? 0.01 : rng.uniform(0.2, 1)));
    const wD = 0.7;
    const wN = 0.2;
    const L = (g: Float64Array, a: Float64Array): number => {
      const r = geometryLossCPU(g, a, W, H, 1, t, wD, wN);
      // the kernel's partial sums are unweighted per term: L = invPx · (λ_d Σ w|e| + λ_n Σ w‖dn‖₁)
      return (wD * r.depth + wN * r.normal) / (W * H);
    };
    const r = geometryLossCPU(geo, alpha, W, H, 1, t, wD, wN);
    expect(r.valid).toBeGreaterThan(50);
    let checked = 0;
    const h = 1e-7;
    for (let idx = 0; idx < geo.length; idx += 7) {
      if (r.grad[idx] === 0) continue;
      const gp = geo.slice();
      const gm = geo.slice();
      gp[idx] += h;
      gm[idx] -= h;
      const fd = (L(gp, alpha) - L(gm, alpha)) / (2 * h);
      expect(Math.abs(fd - r.grad[idx])).toBeLessThan(1e-6 + 1e-4 * Math.abs(fd));
      checked++;
    }
    for (let o = 0; o < alpha.length; o += 3) {
      if (r.gradA[o] === 0) continue;
      const ap = alpha.slice();
      const am = alpha.slice();
      ap[o] += h;
      am[o] -= h;
      const fd = (L(geo, ap) - L(geo, am)) / (2 * h);
      expect(Math.abs(fd - r.gradA[o])).toBeLessThan(1e-6 + 1e-4 * Math.abs(fd));
      checked++;
    }
    expect(checked).toBeGreaterThan(80);
    // level 2 maps onto the target 1:1 (target factor 2)
    const r2 = geometryLossCPU(geo.subarray(0, 4 * 12 * 9), new Float64Array(12 * 9).fill(1), 12, 9, 2, t, wD, wN);
    expect(r2.valid).toBe(t.valid);
  });
});
