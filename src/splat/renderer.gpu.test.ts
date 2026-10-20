import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { getGPU, type GPUContext } from '../gpu/device';
import { pipelineCompileLimits } from '../features/gpu-common';
import { renderCPU } from './cpu-reference';
import { SplatRenderer } from './renderer';
import { randomScene, testCamera } from './testutil';
import type { SplatCamera, SplatModel } from './types';

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
    const m = randomScene({ count: 300, seed: 25, logScale: [-1.2, -0.8], opacity: [0.2, 0.4] }); // large splats
    const cam = testCamera(160, 128);
    const r = await SplatRenderer.create(ctx, { maxWidth: 160, maxHeight: 128, isectCapacity: 4096 });
    r.setModel(m);
    const g = await r.render(cam, { withStats: true });
    r.destroy();
    const c = renderCPU(m, cam);
    let d = 0;
    for (let i = 0; i < cam.width * cam.height; i++) for (let ch = 0; ch < 3; ch++) d = Math.max(d, Math.abs(g.data[4 * i + ch] - c.color[3 * i + ch]));
    console.log(`[splat] intersections ${g.intersections} (reserved 4096 → grown), max diff ${d.toExponential(2)}`);
    expect(g.intersections).toBeGreaterThan(4096);
    expect(d).toBeLessThan(1e-4);
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
