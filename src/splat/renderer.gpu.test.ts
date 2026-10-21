import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { getGPU, type GPUContext } from '../gpu/device';
import { pipelineCompileLimits } from '../features/gpu-common';
import { renderCPU } from './cpu-reference';
import { SplatRenderer } from './renderer';
import { randomScene, testCamera } from './testutil';
import { ALPHA_MIN, type SplatCamera, type SplatModel } from './types';

let ctx: GPUContext;
const renderers = new Map<string, SplatRenderer>();

beforeAll(async () => {
  const c = await getGPU();
  if (!c) throw new Error('WebGPU unavailable in the test browser');
  // SwiftShader compiles these pipelines for minutes: no async → sync fallback (it compiles twice)
  if (c.isSoftware) pipelineCompileLimits.timeoutMs = 600_000;
  ctx = c;
}, 600_000);

afterAll(() => {
  for (const r of renderers.values()) r.destroy();
});

async function renderer(deg: 0 | 1, antialiased = true): Promise<SplatRenderer> {
  const key = `${deg}-${antialiased}`;
  let r = renderers.get(key);
  if (!r) {
    r = await SplatRenderer.create(ctx, { shDegree: deg, antialiased, maxWidth: 160, maxHeight: 128 });
    renderers.set(key, r);
  }
  return r;
}

interface Parity {
  maxDiff: number;
  meanDiff: number;
  fracOver1e3: number;
  nSame: number;
}

async function parity(model: SplatModel, cam: SplatCamera, antialiased = true, bg: [number, number, number] = [0.1, 0.2, 0.3]): Promise<Parity> {
  const r = await renderer(model.shDegree, antialiased);
  r.setModel(model);
  const g = await r.render(cam, { background: bg, withN: true });
  const c = renderCPU(model, cam, { background: bg, antialiased });
  let maxDiff = 0;
  let sum = 0;
  let over = 0;
  let nSame = 0;
  const px = cam.width * cam.height;
  for (let i = 0; i < px; i++) {
    let d = Math.abs(g.data[4 * i + 3] - c.T[i]);
    for (let ch = 0; ch < 3; ch++) d = Math.max(d, Math.abs(g.data[4 * i + ch] - c.color[3 * i + ch]));
    maxDiff = Math.max(maxDiff, d);
    sum += d;
    if (d > 1e-3) over++;
    if (g.n![i] === c.n[i]) nSame++;
  }
  return { maxDiff, meanDiff: sum / px, fracOver1e3: over / px, nSame: nSame / px };
}

describe('SplatRenderer (GPU) vs CPU reference', () => {
  it('random scene, SH degree 0, antialiased', async () => {
    const m = randomScene({ count: 400, seed: 21 });
    const p = await parity(m, testCamera(150, 110));
    console.log('[splat] parity deg0', JSON.stringify(p));
    expect(p.meanDiff).toBeLessThan(2e-5);
    expect(p.fracOver1e3).toBeLessThan(2e-3);
    expect(p.maxDiff).toBeLessThan(2e-2);
    expect(p.nSame).toBeGreaterThan(0.995);
  });

  it('random scene, SH degree 1, classic low-pass, dense overlaps (early termination)', async () => {
    const m = randomScene({ count: 600, seed: 22, shDegree: 1, opacity: [0.6, 0.98], logScale: [-2.6, -1.4] });
    const p = await parity(m, testCamera(140, 128), false);
    console.log('[splat] parity deg1', JSON.stringify(p));
    expect(p.meanDiff).toBeLessThan(5e-5);
    expect(p.fracOver1e3).toBeLessThan(5e-3);
    expect(p.nSame).toBeGreaterThan(0.99);
  });

  it('degenerate input: NaN / zero-size / behind-camera / huge Gaussians render finite images', async () => {
    const m = randomScene({ count: 200, seed: 23 });
    m.means[0] = NaN;
    m.quats.fill(0, 4, 8); // zero quaternion
    m.logScales.fill(-40, 6, 9); // zero size
    m.logScales.fill(60, 9, 12); // overflow
    m.means.set([0, -10, 0], 12); // behind the camera
    m.opacities[5] = Infinity;
    m.opacities[6] = -Infinity;
    m.sh[3 * 7] = NaN;
    m.logScales.set([1.5, 1.5, 1.5], 24); // covers the whole screen
    const cam = testCamera(120, 96);
    const r = await renderer(0);
    r.setModel(m);
    const g = await r.render(cam, { withStats: true });
    let bad = 0;
    for (let i = 0; i < g.data.length; i++) if (!Number.isFinite(g.data[i])) bad++;
    // a NaN colour coefficient only poisons its own splat's pixels; everything else is finite
    const c = renderCPU(m, cam);
    let cpuBad = 0;
    for (let i = 0; i < c.color.length; i++) if (!Number.isFinite(c.color[i])) cpuBad++;
    console.log(`[splat] fuzz: non-finite GPU ${bad}, CPU ${cpuBad}, intersections ${g.intersections}`);
    expect(bad).toBeLessThanOrEqual(cpuBad + (cam.width * cam.height) / 4);
    expect(g.intersections).toBeGreaterThan(0);
  });

  it('grows the intersection reservation when a frame needs more', async () => {
    const m = randomScene({ count: 300, seed: 25, logScale: [-1.2, -0.8], opacity: [0.2, 0.4] }); // large splats (16618 intersections)
    const cam = testCamera(160, 128);
    const px = cam.width * cam.height;
    const r = await SplatRenderer.create(ctx, { maxWidth: 160, maxHeight: 128, isectCapacity: 4096 });
    // the same kernels with a reservation this frame fits from the start (it never grows)
    const fits = await SplatRenderer.create(ctx, { maxWidth: 160, maxHeight: 128, isectCapacity: 32768 });
    try {
      r.setModel(m);
      fits.setModel(m);
      const g = await r.render(cam, { withStats: true, withN: true });
      const e = await fits.render(cam, { withStats: true, withN: true });
      expect(g.intersections).toBeGreaterThan(4096);
      expect(r.isectCapacity).toBeGreaterThanOrEqual(g.intersections!);
      expect(e.intersections).toBe(g.intersections);
      expect(fits.isectCapacity).toBe(32768);
      // The forward pass is deterministic (stable scan-ranked radix sorts with count-driven dispatches,
      // per-pixel compositing) and the reservation only caps the emitted intersections: the frame
      // rendered after the growth must be, bit for bit, the frame that never overflowed. A stale or
      // truncated intersection list, or the overflowing first frame returned, would differ (the
      // 4096-capped frame is off the CPU by 0.25 max, 5.3e-2 mean, by > 1e-4 on 86.5 % of its pixels).
      const bits = (a: Float32Array | Uint32Array): Uint32Array => new Uint32Array(a.buffer, a.byteOffset, a.length);
      const gb = bits(g.data);
      const eb = bits(e.data);
      let wordsDiffer = 0;
      for (let i = 0; i < gb.length; i++) if (gb[i] !== eb[i]) wordsDiffer++;
      let nDiffer = 0;
      for (let i = 0; i < px; i++) if (g.n![i] !== e.n![i]) nDiffer++;
      expect(wordsDiffer).toBe(0);
      expect(nDiffer).toBe(0);
      // ... and that frame is complete (f32 GPU vs f64 CPU). Pixels agree to f32 rounding (≤ 6.9e-7
      // on SwiftShader) except where a splat's σ lands within the f32 error of the α cut
      // σ = ln(o / α_min): one side skips the splat, the other adds α_min·T·c ≤ 0.84·α_min = 3.3e-3
      // (T ≤ 1; degree-0 colours ≤ 0.5 + 1.2·SH_C0 = 0.84). On SwiftShader that is 1 pixel of 20480:
      // (0, 80), splat 19 at σ 4.1855038 (f64) / 4.1854987 (f32) against the cut 4.1855013 / 4.1855016,
      // which differs by 1.25e-3. A `max < 1e-4` bound over all pixels holds only when no σ happens to
      // land that close to a cut, i.e. by luck, on any f32 rasteriser.
      const c = renderCPU(m, cam);
      let maxDiff = 0;
      let sum = 0;
      let over = 0;
      for (let i = 0; i < px; i++) {
        let d = 0;
        for (let ch = 0; ch < 3; ch++) d = Math.max(d, Math.abs(g.data[4 * i + ch] - c.color[3 * i + ch]));
        maxDiff = Math.max(maxDiff, d); // NaN propagates and fails below
        sum += d;
        if (!(d <= 1e-4)) over++;
      }
      console.log(
        `[splat] intersections ${g.intersections} (reserved 4096 → ${r.isectCapacity}); same bits as the unbounded frame; vs CPU: max ${maxDiff.toExponential(2)}, mean ${(sum / px).toExponential(2)}, ${over} of ${px} pixels > 1e-4`,
      );
      // α-cut flips only: 1 pixel here; 20 (1e-3) leaves room for a GPU whose exp / FMA rounding moves σ more
      expect(over / px).toBeLessThan(1e-3);
      // ≤ two flipped splats in one pixel (2 · 3.3e-3); at 1 flip per 20480 pixels, two in one pixel is ~2e-5 per run
      expect(maxDiff).toBeLessThan(2 * ALPHA_MIN);
      // rounding (1.5e-7 here) + 20 flipped pixels at 3.3e-3 each (3.2e-6): 4 orders below a truncated frame
      expect(sum / px).toBeLessThan(5e-6);
    } finally {
      r.destroy();
      fits.destroy();
    }
  });

  it('a rejected reservation or model leaves the renderer intact (review)', async () => {
    const m = randomScene({ count: 200, seed: 26 });
    const cam = testCamera(96, 80);
    const r = await SplatRenderer.create(ctx, { maxWidth: 160, maxHeight: 128 });
    try {
      r.setModel(m);
      await r.render(cam, { background: [0.1, 0.2, 0.3] });
      const isect = r.isectCapacity;
      // beyond one sorter (64 M intersections) and beyond 1 M Gaussians: refused before anything is freed
      expect(() => r.reserveIntersections(100_000_000)).toThrow(/too large/);
      expect(r.isectCapacity).toBe(isect);
      expect(() => r.setModel({ ...m, count: 1_100_000 })).toThrow(/1M Gaussians/);
      expect(r.modelCount).toBe(m.count);
      const bg: [number, number, number] = [0.7, 0.1, 0.4];
      const g = await r.render(cam, { background: bg });
      const c = renderCPU(m, cam, { background: bg });
      let d = 0;
      for (let i = 0; i < cam.width * cam.height; i++) for (let ch = 0; ch < 3; ch++) d = Math.max(d, Math.abs(g.data[4 * i + ch] - c.color[3 * i + ch]));
      expect(d).toBeLessThan(1e-4);
    } finally {
      r.destroy();
    }
  });

  it('empty model and a model outside the view render the background', async () => {
    const m = randomScene({ count: 10, seed: 24 });
    for (let i = 0; i < 10; i++) m.means.set([0, 50, 0], 3 * i); // far behind
    const r = await renderer(0);
    r.setModel(m);
    const g = await r.render(testCamera(64, 48), { background: [0.25, 0.5, 0.75] });
    expect(g.data[0]).toBeCloseTo(0.25, 6);
    expect(g.data[4 * 100 + 2]).toBeCloseTo(0.75, 6);
    expect(g.data[3]).toBe(1);
  });
});
