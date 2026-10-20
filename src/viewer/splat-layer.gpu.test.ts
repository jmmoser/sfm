/**
 * The Viewer's splat layer (WebGPU canvas under the WebGL canvas): Gaussians land where the three.js
 * camera projects their means (world root, aspect, pinhole conversion), the WebGL canvas turns
 * transparent except for the overlays, measuring picks the hidden points, frames are drawn only when
 * the view changes, the pixel budget holds, dispose / setSplats(null) remove the layer, and a
 * disposed viewer is garbage-collected (the shared device's `lost` promise does not keep it).
 */
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import type { PointCloud, TriangleMesh } from '../core/types';
import { pipelineCompileLimits } from '../features/gpu-common';
import { getGPU } from '../gpu/device';
import { lookAt } from '../splat/testutil';
import { SH_C0, allocSplatModel, logit, type SplatModel } from '../splat/types';
import type { SplatView } from './splat-layer';
import { Viewer, type ViewerOptions } from './viewer';

const containers: HTMLElement[] = [];
const viewers: Viewer[] = [];

beforeAll(async () => {
  const c = await getGPU();
  if (!c) throw new Error('WebGPU unavailable in the test browser');
  if (c.isSoftware) pipelineCompileLimits.timeoutMs = 600_000;
}, 600_000);

afterEach(() => {
  for (const v of viewers.splice(0)) v.dispose();
  for (const c of containers.splice(0)) c.remove();
});

function makeViewer(w: number, h: number, opts: ViewerOptions = {}): Viewer {
  const el = document.createElement('div');
  el.style.cssText = `position:fixed;left:0;top:0;width:${w}px;height:${h}px`;
  document.body.appendChild(el);
  containers.push(el);
  const v = new Viewer(el, { background: '#000000', maxPixelRatio: 1, splatAllowSoftware: true, ...opts });
  viewers.push(v);
  return v;
}

/** Small isotropic opaque Gaussians (world Z-up) with the given sRGB colours. */
function dots(points: number[][], colours: number[][], sigma = 0.025): SplatModel {
  const m = allocSplatModel(points.length, 0);
  points.forEach((p, i) => {
    m.means.set(p, 3 * i);
    m.logScales.fill(Math.log(sigma), 3 * i, 3 * i + 3);
    m.opacities[i] = logit(0.98);
    for (let c = 0; c < 3; c++) m.sh[3 * i + c] = (colours[i][c] - 0.5) / SH_C0;
  });
  return m;
}

/** Colour-weighted centroid (pixel centres at x + 0.5, i.e. CSS px at pixel ratio 1) of channel `ch` around (x0, y0). */
function centroid(px: { width: number; height: number; data: Uint8Array }, ch: number, x0: number, y0: number, r = 12): { x: number; y: number; w: number } {
  let s = 0;
  let sx = 0;
  let sy = 0;
  for (let y = Math.max(0, Math.round(y0) - r); y <= Math.min(px.height - 1, Math.round(y0) + r); y++)
    for (let x = Math.max(0, Math.round(x0) - r); x <= Math.min(px.width - 1, Math.round(x0) + r); x++) {
      const i = 4 * (y * px.width + x);
      // this channel only (the others zero): a dot's own colour
      const w = Math.max(0, px.data[i + ch] - Math.max(px.data[i + ((ch + 1) % 3)], px.data[i + ((ch + 2) % 3)]));
      s += w;
      sx += w * (x + 0.5);
      sy += w * (y + 0.5);
    }
  return { x: sx / s, y: sy / s, w: s };
}

const PTS = [
  [0, 0, 0],
  [0.45, 0.1, 0.25],
  [-0.35, -0.3, -0.12],
];
const COLS = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

describe('Viewer splat layer', () => {
  for (const [w, h] of [
    [240, 180],
    [150, 230],
  ]) {
    it(`draws Gaussians where the three.js camera projects them (${w}×${h}, Z-up world root)`, async () => {
      const v = makeViewer(w, h);
      expect(await v.setSplats({ model: dots(PTS, COLS) })).toBe(true);
      expect(v.getStats().splatState).toBe('ready');
      v.setOrbitTarget([0.05, 0, 0.05]);
      const px = await v.readSplatPixels();
      expect(px).not.toBeNull();
      expect(px!.width).toBe(w);
      expect(px!.height).toBe(h);
      PTS.forEach((p, k) => {
        const s = v.worldToScreen(p)!;
        const c = centroid(px!, k, s.x, s.y);
        const err = Math.hypot(c.x - s.x, c.y - s.y);
        console.log(`[splat-layer] ${w}×${h} dot ${k}: expected (${s.x.toFixed(2)}, ${s.y.toFixed(2)}), centroid error ${err.toFixed(3)} px`);
        expect(c.w).toBeGreaterThan(100);
        expect(err).toBeLessThan(0.15);
      });
      expect(v.getStats().splatsShown).toBe(true);
    });
  }

  it('turns the WebGL canvas transparent (overlays only) while the splats are shown', async () => {
    const v = makeViewer(200, 160);
    const plane: PointCloud = { count: 4, positions: new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), colors: new Uint8Array(12).fill(200) };
    v.setPointCloud(plane, { pointSizePx: 8, splats: false });
    const intr = { width: 640, height: 480, f: 500, cx: 319.5, cy: 239.5, k1: 0, k2: 0 };
    const { R, t } = lookAt([0.3, -0.8, 0.5], [0, 0, 0]);
    v.setCameras([{ pose: { R: Float64Array.from(R), t: Float64Array.from(t) }, intrinsics: intr }], false);
    expect(await v.setSplats({ model: dots(PTS, COLS) })).toBe(true);
    v.setDisplay({ grid: false });
    v.fitToContent();
    const gl = v.readPixels();
    let opaque = 0;
    let lit = 0;
    for (let i = 0; i < gl.data.length; i += 4) {
      if (gl.data[i + 3] > 0) opaque++;
      if (gl.data[i] > 100 || gl.data[i + 1] > 100) lit++;
    }
    // the frustum lines only (no background, no points)
    expect(opaque).toBeGreaterThan(20);
    expect(opaque).toBeLessThan(0.15 * 200 * 160);
    expect(lit).toBeGreaterThan(0);
    // splats off: the points and the opaque background are back
    v.setDisplay({ splats: false });
    const back = v.readPixels();
    let opaque2 = 0;
    for (let i = 3; i < back.data.length; i += 4) if (back.data[i] === 255) opaque2++;
    expect(opaque2).toBe(200 * 160);
    expect(v.getStats().splatsShown).toBe(false);
  });

  it('measures on the (hidden) points while the splats are shown', async () => {
    const v = makeViewer(240, 180);
    // dense grid on z = 0, [-1, 1]²
    const k = 81;
    const pos = new Float32Array(3 * k * k);
    for (let j = 0, i = 0; j < k; j++) for (let q = 0; q < k; q++, i++) pos.set([-1 + (2 * q) / (k - 1), -1 + (2 * j) / (k - 1), 0], 3 * i);
    v.setPointCloud({ count: k * k, positions: pos, colors: new Uint8Array(3 * k * k).fill(128) }, { pointSizePx: 4, splats: false });
    expect(await v.setSplats({ model: dots(PTS, COLS) })).toBe(true);
    v.fitToContent();
    v.renderNow();
    expect(v.getStats().splatsShown).toBe(true);
    const a = v.worldToScreen([-0.5, 0.2, 0])!;
    const b = v.worldToScreen([0.5, -0.1, 0])!;
    const m = v.measureBetweenScreenPoints(a, b);
    expect(m).not.toBeNull();
    const truth = Math.hypot(1, 0.3);
    console.log(`[splat-layer] measured ${m!.distance.toFixed(4)} (truth ${truth.toFixed(4)})`);
    expect(Math.abs(m!.distance - truth)).toBeLessThan(0.01 * truth);
    expect(Math.abs(m!.a[2])).toBeLessThan(0.01);
  });

  it('draws a frame only when the view changes, within the pixel budget', async () => {
    const v = makeViewer(400, 300, { maxPixelRatio: 2, splatMaxPixels: 30_000 });
    expect(await v.setSplats({ model: dots(PTS, COLS) })).toBe(true);
    // let the framing settle (scheduled frames, damping)
    for (let i = 0; i < 60 && (i < 5 || v.framePending); i++) await new Promise((r) => requestAnimationFrame(r));
    await v.readSplatPixels();
    const st0 = v.getStats().splatLayer!;
    expect(st0.canvasWidth * st0.canvasHeight).toBeLessThanOrEqual(30_000);
    expect(Math.abs(st0.canvasWidth / st0.canvasHeight - 4 / 3)).toBeLessThan(0.02);
    const f0 = st0.frames;
    for (let i = 0; i < 3; i++) {
      v.renderNow();
      await new Promise((r) => setTimeout(r, 30));
    }
    expect(v.getStats().splatLayer!.frames).toBe(f0);
    v.setOrbitTarget([0.2, 0, 0]);
    v.renderNow();
    await new Promise((r) => setTimeout(r, 30));
    expect(v.getStats().splatLayer!.frames).toBe(f0 + 1);
  });

  it('removes the layer on setSplats(null) and dispose', async () => {
    const v = makeViewer(160, 120);
    expect(await v.setSplats({ model: dots(PTS, COLS) })).toBe(true);
    const host = v.container;
    expect(host.querySelector('canvas.viewer-splats')).not.toBeNull();
    // stacked under the WebGL canvas
    const canvases = [...host.querySelectorAll('canvas')];
    expect(canvases[0].classList.contains('viewer-splats')).toBe(true);
    expect(await v.setSplats(null)).toBe(false);
    expect(host.querySelector('canvas.viewer-splats')).toBeNull();
    expect(v.getStats().splatState).toBe('none');
    expect(await v.setSplats({ model: dots(PTS, COLS) })).toBe(true);
    v.dispose();
    expect(host.querySelector('canvas')).toBeNull();
  });

  it('screenshot() composites the splats under the overlays', async () => {
    const v = makeViewer(160, 120);
    expect(await v.setSplats({ model: dots(PTS, COLS) })).toBe(true);
    const blob = await v.screenshot();
    const bmp = await createImageBitmap(blob);
    const c = new OffscreenCanvas(bmp.width, bmp.height);
    const ctx = c.getContext('2d')!;
    ctx.drawImage(bmp, 0, 0);
    const d = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
    let red = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i] > 150 && d[i + 1] < 60 && d[i + 3] === 255) red++;
    expect(red).toBeGreaterThan(5);
  });

  it('a layer that gave up after a device loss moves to a new device on the next setModel (review)', async () => {
    const { SplatLayer } = await import('./splat-layer');
    const el = document.createElement('div');
    el.style.cssText = 'position:fixed;left:0;top:0;width:160px;height:120px';
    document.body.appendChild(el);
    containers.push(el);
    const states: string[] = [];
    const layer = await SplatLayer.create(el, null, { allowSoftware: true, maxPixelRatio: 1, maxRecoveries: 0, onState: (s) => states.push(s) });
    try {
      // three.js camera at the origin looking down −z (identity view), 60° vertical fov
      const [W, H, near, far] = [160, 120, 0.01, 100];
      const f = 1 / Math.tan(Math.PI / 6);
      const view: SplatView = {
        view: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        projection: [f / (W / H), 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, (2 * far * near) / (near - far), 0],
        cssWidth: W,
        cssHeight: H,
        near,
        background: [0, 0, 0],
        interactive: false,
      };
      const redPixels = async (): Promise<number> => {
        layer.draw(view);
        const px = await layer.readPixels();
        let n = 0;
        if (px) for (let i = 0; i < px.data.length; i += 4) if (px.data[i] > 150 && px.data[i + 1] < 60) n++;
        return n;
      };
      await layer.setModel(dots([[0, 0, -2]], [[1, 0, 0]], 0.05));
      expect(await redPixels()).toBeGreaterThan(20);
      (await getGPU())!.device.destroy();
      for (let i = 0; i < 200 && layer.stats().state !== 'unavailable'; i++) await new Promise((r) => setTimeout(r, 10));
      expect(states).toEqual(['ready', 'unavailable']);
      // a new model: built on a fresh device (not on the lost one, reporting 'ready' while drawing nothing)
      await layer.setModel(dots([[0, 0, -2]], [[1, 0, 0]], 0.05));
      expect(layer.stats().state).toBe('ready');
      expect(await redPixels()).toBeGreaterThan(20);
    } finally {
      layer.dispose();
    }
  });

  it('a disposed viewer with splats is garbage-collected: the shared device does not keep it (review)', async () => {
    const gc = (globalThis as { gc?: () => void }).gc;
    if (!gc) {
      console.warn('[splat-layer] gc() not exposed (--js-flags=--expose-gc): leak check skipped');
      return;
    }
    /** A viewer with a mesh and splats whose onSplatState captures a "screen scope", disposed. */
    const disposed = async (): Promise<{ viewer: WeakRef<object>; mesh: WeakRef<object>; scope: WeakRef<object> }> => {
      const el = document.createElement('div');
      el.style.cssText = 'position:fixed;left:0;top:0;width:160px;height:120px';
      document.body.appendChild(el);
      const scope = { exports: new Uint8Array(1 << 20) };
      const v = new Viewer(el, { background: '#000000', maxPixelRatio: 1, splatAllowSoftware: true, onSplatState: () => void scope.exports.length });
      const n = 3000;
      const mesh: TriangleMesh = {
        vertexCount: n,
        triangleCount: n / 3,
        positions: Float32Array.from({ length: 3 * n }, (_, i) => Math.sin(i) * 0.5),
        normals: new Float32Array(3 * n).fill(0.577),
        colors: new Uint8Array(3 * n).fill(180),
        indices: Uint32Array.from({ length: n }, (_, i) => i),
      };
      v.setMesh(mesh);
      expect(await v.setSplats({ model: dots(PTS, COLS) })).toBe(true);
      const refs = { viewer: new WeakRef(v), mesh: new WeakRef(mesh.positions), scope: new WeakRef(scope) };
      v.dispose();
      el.remove();
      return refs;
    };
    const refs = await disposed();
    for (let i = 0; i < 6; i++) {
      await new Promise((r) => setTimeout(r, 50));
      gc();
    }
    expect({ viewer: !!refs.viewer.deref(), mesh: !!refs.mesh.deref(), scope: !!refs.scope.deref() }).toEqual({ viewer: false, mesh: false, scope: false });
  });
});
