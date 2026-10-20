import { describe, expect, it } from 'vitest';
import { Rng } from '../math/random';
import { backwardCPU, lossCPU, renderCPU, type RenderOptions } from './cpu-reference';
import { cloneModel, randomScene, testCamera } from './testutil';
import type { SplatModel } from './types';

type Field = 'means' | 'logScales' | 'quats' | 'opacities' | 'sh';
const FIELDS: Field[] = ['means', 'logScales', 'quats', 'opacities', 'sh'];

/** L = Σ w · C (+ Σ wG · G + Σ wA · (1 − T) with the geometry channels) for fixed random weights. */
function weightedLoss(model: SplatModel, cam: ReturnType<typeof testCamera>, w: Float64Array, opts: RenderOptions, wG: Float64Array | null, wA: Float64Array | null): number {
  const r = renderCPU(model, cam, opts);
  let s = 0;
  for (let i = 0; i < w.length; i++) s += w[i] * r.color[i];
  if (wG && r.geo) for (let i = 0; i < wG.length; i++) s += wG[i] * r.geo[i];
  if (wA) for (let i = 0; i < wA.length; i++) s += wA[i] * (1 - r.T[i]);
  return s;
}

function fdCheck(model: SplatModel, opts: RenderOptions, ids: number[], label: string): void {
  const cam = testCamera(40, 32);
  const rng = new Rng(7);
  const w = new Float64Array(3 * cam.width * cam.height).map(() => rng.uniform(-1, 1));
  // geometry weights: inverse depth ~1/3 → scale its weight up so its term matters
  const wG = opts.geometry ? new Float64Array(4 * cam.width * cam.height).map((_, i) => rng.uniform(-1, 1) * (i % 4 === 0 ? 3 : 1)) : null;
  const wA = opts.geometry ? new Float64Array(cam.width * cam.height).map(() => rng.uniform(-1, 1)) : null;
  const { grads } = backwardCPU(model, cam, w, opts, undefined, wG, wA);
  let checked = 0;
  for (const f of FIELDS) {
    const per = model[f].length / model.count;
    for (const i of ids) {
      for (let k = 0; k < per; k++) {
        const idx = per * i + k;
        const h = 1e-4 * Math.max(1, Math.abs(model[f][idx]));
        const mp = cloneModel(model);
        const mm = cloneModel(model);
        // Float32 storage: use the actually stored perturbation in the quotient
        mp[f][idx] = model[f][idx] + h;
        mm[f][idx] = model[f][idx] - h;
        const hh = (mp[f][idx] - mm[f][idx]) / 2;
        const fd = (weightedLoss(mp, cam, w, opts, wG, wA) - weightedLoss(mm, cam, w, opts, wG, wA)) / (2 * hh);
        const an = grads[f][idx];
        const tol = 2e-3 * Math.max(1e-3, Math.abs(fd), Math.abs(an));
        if (Math.abs(fd - an) > tol) throw new Error(`${label}: ${f}[${i}][${k}] analytic ${an} vs fd ${fd}`);
        checked++;
      }
    }
  }
  expect(checked).toBeGreaterThan(20);
}

describe('splat CPU reference', () => {
  // A tiny alpha threshold keeps the footprint continuous (no 1/255 cut-off jumps under the
  // finite-difference perturbation); the GPU parity tests use the real threshold.
  const smooth = { alphaMin: 1e-9 };

  it('forward: opaque splat covers its centre, background shows elsewhere', () => {
    const m = randomScene({ count: 1, seed: 3 });
    m.means.set([0, 0, 0]);
    m.logScales.set([-1.5, -1.5, -1.5]);
    m.opacities[0] = 10;
    m.sh.set([1.5, -1.5, -1.5]);
    const cam = testCamera(40, 32);
    const r = renderCPU(m, cam, { background: [0, 0, 1] });
    const c = r.proj[0].mean2d;
    const o = Math.round(c[1]) * 40 + Math.round(c[0]);
    expect(r.color[3 * o]).toBeGreaterThan(0.85); // colour SH_C0·1.5 + 0.5 = 0.92, alpha ≤ 0.99
    expect(r.color[3 * o + 2]).toBeLessThan(0.2); // 0.077·alpha + background·T
    expect(r.color[2]).toBeCloseTo(1, 6); // corner: background
    expect(r.n[o]).toBe(1);
    expect(r.n[0]).toBe(0);
  });

  it('backward matches finite differences (SH degree 0, antialiased)', () => {
    const m = randomScene({ count: 8, seed: 11 });
    fdCheck(m, { ...smooth, background: [0.2, 0.3, 0.4] }, [0, 3, 6], 'deg0');
  });

  it('backward matches finite differences (SH degree 1, classic low-pass)', () => {
    const m = randomScene({ count: 8, seed: 12, shDegree: 1 });
    fdCheck(m, { ...smooth, antialiased: false }, [1, 4, 7], 'deg1');
  });

  it('backward matches finite differences near the frustum border (clamped Jacobian)', () => {
    const m = randomScene({ count: 4, seed: 13 });
    // push two Gaussians beyond the image edge: their Jacobians are evaluated at the clamped point
    m.means.set([2.6, 0, 0.1], 0);
    m.means.set([-0.2, 0, 2.2], 3);
    m.logScales.set([-0.6, -0.7, -0.5, -0.6, -0.8, -0.5], 0);
    fdCheck(m, smooth, [0, 1], 'clamp');
  });

  it('geometry channels: inverse depth and smallest-axis normal of an opaque disc', () => {
    const m = randomScene({ count: 1, seed: 4 });
    m.means.set([0, 0, 0]);
    m.logScales.set([-1.5, -4, -1.4]); // thin along its local y axis
    m.quats.set([1, 0, 0, 0]);
    m.opacities[0] = 10;
    const cam = testCamera(40, 32); // at (0, −3, 0.4) looking at the origin
    const r = renderCPU(m, cam, { geometry: true });
    const p = r.proj[0];
    expect(p.normalAxis).toBe(1);
    // world +y axis, flipped to face the camera (which sits at −y): camera-frame normal of world −y
    const R = cam.R;
    const want = [-R[1], -R[4], -R[7]];
    for (let k = 0; k < 3; k++) expect(p.normal[k]).toBeCloseTo(want[k], 12);
    const o = Math.round(p.mean2d[1]) * 40 + Math.round(p.mean2d[0]);
    const a = 1 - r.T[o];
    expect(r.geo![4 * o]).toBeCloseTo(a / p.depth, 12);
    for (let k = 0; k < 3; k++) expect(r.geo![4 * o + 1 + k]).toBeCloseTo(a * want[k], 12);
    expect(r.geo![0]).toBe(0);
  });

  it('backward with geometry channels matches finite differences (SH degree 0)', () => {
    const m = randomScene({ count: 8, seed: 14 });
    fdCheck(m, { ...smooth, background: [0.2, 0.3, 0.4], geometry: true }, [0, 2, 5], 'geo0');
  });

  it('backward with geometry channels matches finite differences (SH degree 1, clamped Jacobian)', () => {
    const m = randomScene({ count: 5, seed: 15, shDegree: 1 });
    m.means.set([2.6, 0, 0.1], 0);
    m.logScales.set([-0.6, -0.9, -0.5], 0);
    fdCheck(m, { ...smooth, antialiased: false, geometry: true }, [0, 1, 3], 'geo1');
  });

  it('L1 + D-SSIM loss gradient matches finite differences', () => {
    const W = 14;
    const H = 11;
    const rng = new Rng(5);
    const x = new Float64Array(3 * W * H).map(() => rng.uniform(0, 1));
    const y = new Float64Array(3 * W * H).map(() => rng.uniform(0, 1));
    const mask = new Uint8Array(W * H).map((_, i) => (i % 17 === 3 ? 0 : 1));
    const r = lossCPU(x, y, W, H, 0.2, mask);
    expect(r.ssim).toBeGreaterThan(-1);
    expect(r.ssim).toBeLessThan(1);
    for (const idx of [0, 5, 77, 200, 3 * W * H - 1, 3 * 52 + 1]) {
      const h = 1e-6;
      const xp = x.slice();
      const xm = x.slice();
      xp[idx] += h;
      xm[idx] -= h;
      const fd = (lossCPU(xp, y, W, H, 0.2, mask).loss - lossCPU(xm, y, W, H, 0.2, mask).loss) / (2 * h);
      expect(Math.abs(fd - r.grad[idx])).toBeLessThan(1e-6 + 1e-4 * Math.abs(fd));
    }
    // identical images: SSIM 1, L1 0
    const s = lossCPU(x, x, W, H);
    expect(s.ssim).toBeCloseTo(1, 9);
    expect(s.l1).toBe(0);
  });
});
