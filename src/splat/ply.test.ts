import { describe, expect, it } from 'vitest';
import { packParams, unpackParams } from './layout';
import { decodeSplatPLY, encodeSplatPLY } from './ply';
import { randomScene } from './testutil';
import { estimateSplatMemory } from './types';
import { initFromPointCloud, quatFromZ } from './init';
import { planMcmc, relocation } from './mcmc';
import { Rng } from '../math/random';

describe('splat PLY (INRIA layout)', () => {
  for (const deg of [0, 1] as const) {
    it(`round-trips SH degree ${deg}`, () => {
      const m = randomScene({ count: 37, seed: 3, shDegree: deg });
      const bytes = encodeSplatPLY(m, { comments: ['test'] });
      const header = new TextDecoder().decode(bytes.subarray(0, 800));
      expect(header).toContain('element vertex 37');
      expect(header).toContain('property float f_dc_2');
      expect(header.includes('f_rest_8')).toBe(deg === 1);
      const { model, antialiased } = decodeSplatPLY(bytes);
      expect(antialiased).toBe(true);
      expect(model.shDegree).toBe(deg);
      expect(Array.from(model.means)).toEqual(Array.from(m.means));
      expect(Array.from(model.sh)).toEqual(Array.from(m.sh));
      expect(Array.from(model.quats)).toEqual(Array.from(m.quats));
      expect(Array.from(model.logScales)).toEqual(Array.from(m.logScales));
      expect(Array.from(model.opacities)).toEqual(Array.from(m.opacities));
    });
  }

  it('channel-major f_rest: R coefficients first', () => {
    const m = randomScene({ count: 1, seed: 4, shDegree: 1 });
    m.sh.set([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]); // f1 = (1,2,3), f2 = (4,5,6), f3 = (7,8,9)
    const b = encodeSplatPLY(m);
    const n = b.length;
    const body = new Float32Array(b.slice(n - 4 * 26).buffer);
    expect(Array.from(body.subarray(9, 18))).toEqual([1, 4, 7, 2, 5, 8, 3, 6, 9]);
  });

  it('degree-1 file read as degree 0 keeps band 0', () => {
    const m = randomScene({ count: 5, seed: 5, shDegree: 1 });
    const { model } = decodeSplatPLY(encodeSplatPLY(m), 0);
    expect(model.shDegree).toBe(0);
    for (let i = 0; i < 5; i++) for (let c = 0; c < 3; c++) expect(model.sh[3 * i + c]).toBe(m.sh[12 * i + c]);
  });
});

describe('splat helpers', () => {
  it('pack / unpack parameter layout', () => {
    const m = randomScene({ count: 11, seed: 6, shDegree: 1 });
    const u = unpackParams(packParams(m, 16), 11, 16, 1);
    expect(Array.from(u.means)).toEqual(Array.from(m.means));
    expect(Array.from(u.sh)).toEqual(Array.from(m.sh));
    expect(Array.from(u.quats)).toEqual(Array.from(m.quats));
  });

  it('memory plan of 300k Gaussians, SH 1, 0.5 MP, 100 views stays near the design budget', () => {
    const e = estimateSplatMemory({ maxGaussians: 300_000, shDegree: 1, width: 600, height: 800, views: 100 });
    // ~0.5 KB per Gaussian (params + Adam + projection + intersections)
    const perG = (e.breakdown.params + e.breakdown.adam + e.breakdown.grad2d + e.breakdown.projection + e.breakdown.intersections) / 300_000;
    expect(perG).toBeGreaterThan(300);
    expect(perG).toBeLessThan(700);
    expect(e.largestBuffer).toBeLessThan(256 * 1024 * 1024);
  });

  it('quatFromZ maps +z onto the normal', () => {
    for (const n of [[0, 0, 1], [1, 0, 0], [0, -1, 0], [0.3, -0.5, 0.8], [0, 0, -1]]) {
      const [w, x, y, z] = quatFromZ(n[0], n[1], n[2]);
      // third column of R(q)
      const c = [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)];
      const l = Math.hypot(n[0], n[1], n[2]);
      for (let k = 0; k < 3; k++) expect(c[k]).toBeCloseTo(n[k] / l, 6);
    }
  });

  it('initFromPointCloud: discs along normals, colours to SH0, subsampling to the budget', () => {
    const rng = new Rng(7);
    const n = 2000;
    const pc = {
      count: n,
      positions: Float32Array.from({ length: 3 * n }, (_, i) => (i % 3 === 2 ? 0 : rng.uniform(-1, 1))),
      colors: Uint8Array.from({ length: 3 * n }, () => 200),
      normals: Float32Array.from({ length: 3 * n }, (_, i) => (i % 3 === 2 ? 1 : 0)),
    };
    const m = initFromPointCloud(pc, { maxCount: 500, shDegree: 0 });
    expect(m.count).toBe(500);
    for (let i = 0; i < 500; i++) {
      expect(m.logScales[3 * i + 2]).toBeCloseTo(m.logScales[3 * i] + Math.log(0.1), 5);
      expect(m.sh[3 * i] * 0.28209479177387814 + 0.5).toBeCloseTo(200 / 255, 5);
    }
  });

  it('MCMC relocation keeps a single copy unchanged and splits opacity', () => {
    expect(relocation(0.6, 1)).toEqual({ opacity: 0.6, scale: 1 });
    const r = relocation(0.6, 2);
    // two copies composite to the original opacity: 1 − (1 − o')² = o
    expect(1 - (1 - r.opacity) ** 2).toBeCloseTo(0.6, 9);
    expect(r.scale).toBeGreaterThan(0.5);
    expect(r.scale).toBeLessThan(1.5);
  });

  it('MCMC plan: dead Gaussians are relocated, growth fills new slots up to the cap', () => {
    const logits = Float32Array.from([-9, 2, 1, -8, 0.5, 3, -10, 1.5]);
    const p = planMcmc(logits, 8, { capacity: 9, growth: 0.25, minOpacity: 0.005 }, new Rng(1));
    expect(p.dead).toBe(3);
    expect(p.count).toBe(9);
    const dsts = new Set<number>();
    for (let k = 0; k < p.nCopies; k++) {
      dsts.add(p.copies[4 * k]);
      expect([1, 2, 4, 5, 7]).toContain(p.copies[4 * k + 1]);
    }
    expect([...dsts].sort()).toEqual([0, 3, 6, 8]);
  });
});
