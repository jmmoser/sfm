/**
 * src/splat/nvs.ts through the harness hook (src/harness/nvs/renderers.ts `loadSplatRenderer`):
 * small opaque Gaussians at known camera-frame points land on the pinhole projection, the colour is
 * premultiplied sRGB and alpha = 1 − transmittance.
 */
import { describe, expect, it } from 'vitest';
import { loadSplatRenderer } from '../harness/nvs/renderers';
import { encodeSplatPLY } from './ply';
import { lookAt } from './testutil';
import { SH_C0, allocSplatModel, logit } from './types';

describe('createNvsRenderer (NVS harness splat slot)', () => {
  it('opaque Gaussians land on the pinhole projection with premultiplied colour and coverage alpha', async () => {
    const W = 160;
    const H = 120;
    const { R, t } = lookAt([0.3, -2.5, 0.6], [0, 0, 0.1]);
    const cam = { width: W, height: H, fx: 150, fy: 150, cx: 83.3, cy: 57.6, R, t };
    // camera-frame points → world (X = Rᵀ (x − t))
    const toWorld = (x: number[]): number[] => {
      const d = [x[0] - t[0], x[1] - t[1], x[2] - t[2]];
      return [R[0] * d[0] + R[3] * d[1] + R[6] * d[2], R[1] * d[0] + R[4] * d[1] + R[7] * d[2], R[2] * d[0] + R[5] * d[1] + R[8] * d[2]];
    };
    const pts = [
      [0, 0, 2.5],
      [-0.9, -0.4, 3],
      [0.6, 0.4, 2.2],
    ];
    const colour = [0.8, 0.4, 0.2];
    const m = allocSplatModel(pts.length, 0);
    pts.forEach((p, i) => {
      m.means.set(toWorld(p), 3 * i);
      m.logScales.fill(Math.log(0.06), 3 * i, 3 * i + 3);
      m.opacities[i] = logit(0.995);
      for (let c = 0; c < 3; c++) m.sh[3 * i + c] = (colour[c] - 0.5) / SH_C0;
    });
    const ply = encodeSplatPLY(m, { antialiased: true });
    const r = await loadSplatRenderer(ply.slice().buffer, W, H);
    if ('skipped' in r) throw new Error(r.skipped);
    try {
      const img = await r.render(cam);
      expect(img.length).toBe(4 * W * H);
      for (const p of pts) {
        const u0 = cam.fx * (p[0] / p[2]) + cam.cx;
        const v0 = cam.fy * (p[1] / p[2]) + cam.cy;
        // alpha-weighted centroid in a box around the expected projection
        let s = 0;
        let su = 0;
        let sv = 0;
        for (let v = Math.round(v0) - 14; v <= Math.round(v0) + 14; v++)
          for (let u = Math.round(u0) - 14; u <= Math.round(u0) + 14; u++) {
            const a = img[4 * (v * W + u) + 3] / 255;
            s += a;
            su += a * u;
            sv += a * v;
          }
        const err = Math.hypot(su / s - u0, sv / s - v0);
        console.log(`[nvs] splat at (${u0.toFixed(2)}, ${v0.toFixed(2)}): centroid error ${err.toFixed(4)} px`);
        expect(err).toBeLessThan(0.05);
        // centre pixel: opaque, colour premultiplied by the coverage
        const k = 4 * (Math.round(v0) * W + Math.round(u0));
        const a = img[k + 3] / 255;
        expect(a).toBeGreaterThan(0.9);
        for (let c = 0; c < 3; c++) expect(Math.abs(img[k + c] / 255 - a * colour[c])).toBeLessThan(2.5 / 255);
      }
      // far from every Gaussian: nothing drawn
      expect(Array.from(img.subarray(0, 4))).toEqual([0, 0, 0, 0]);
    } finally {
      r.dispose?.();
    }
  });
});
