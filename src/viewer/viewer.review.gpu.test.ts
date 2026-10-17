/**
 * Adversarial review tests for the viewer (headless Chromium, SwiftShader GL).
 * Each block targets a specific suspected failure mode: frame conventions,
 * splat size math, EDL depth output, picking in non-trivial world frames and at
 * large scale, render-loop termination, GPU resource leaks, context loss and
 * error handling.
 */
import { afterEach, describe, expect, it } from 'vitest';
import type { CameraIntrinsics, PointCloud, Pose, TriangleMesh } from '../core/types';
import { Viewer, type ViewerOptions } from './viewer';

const W = 240;
const H = 180;
const containers: HTMLElement[] = [];
const viewers: Viewer[] = [];

function makeContainer(w = W, h = H): HTMLElement {
  const el = document.createElement('div');
  el.style.width = `${w}px`;
  el.style.height = `${h}px`;
  document.body.appendChild(el);
  containers.push(el);
  return el;
}

function makeViewer(opts: ViewerOptions = {}, el = makeContainer()): Viewer {
  const v = new Viewer(el, { background: '#000000', ...opts });
  viewers.push(v);
  return v;
}

afterEach(() => {
  for (const v of viewers.splice(0)) v.dispose();
  for (const c of containers.splice(0)) c.remove();
});

const nextFrame = () => new Promise<number>((r) => requestAnimationFrame(r));
async function settle(v: Viewer, max = 600): Promise<number> {
  let i = 0;
  for (; i < max && v.framePending; i++) await nextFrame();
  return i;
}

function planeCloud(s: number, step: number, z0: number, rgb: [number, number, number]): PointCloud {
  const k = Math.floor((2 * s) / step + 1e-9) + 1;
  const n = k * k;
  const positions = new Float32Array(3 * n);
  const colors = new Uint8Array(3 * n);
  for (let j = 0, i = 0; j < k; j++) {
    for (let q = 0; q < k; q++, i++) {
      positions[3 * i] = -s + q * step;
      positions[3 * i + 1] = -s + j * step;
      positions[3 * i + 2] = z0;
      colors.set(rgb, 3 * i);
    }
  }
  return { count: n, positions, colors };
}

function concatClouds(a: PointCloud, b: PointCloud): PointCloud {
  const positions = new Float32Array(3 * (a.count + b.count));
  positions.set(a.positions);
  positions.set(b.positions, 3 * a.count);
  const colors = new Uint8Array(3 * (a.count + b.count));
  colors.set(a.colors);
  colors.set(b.colors, 3 * a.count);
  const out: PointCloud = { count: a.count + b.count, positions, colors };
  if (a.confidence && b.confidence) {
    out.confidence = new Float32Array(a.count + b.count);
    out.confidence.set(a.confidence);
    out.confidence.set(b.confidence, a.count);
  }
  return out;
}

function planeMesh(s: number, k: number, rgb: [number, number, number]): TriangleMesh {
  const nv = (k + 1) * (k + 1);
  const positions = new Float32Array(3 * nv);
  const normals = new Float32Array(3 * nv);
  const colors = new Uint8Array(3 * nv);
  for (let j = 0, i = 0; j <= k; j++) {
    for (let q = 0; q <= k; q++, i++) {
      positions.set([-s + (2 * s * q) / k, -s + (2 * s * j) / k, 0], 3 * i);
      normals.set([0, 0, 1], 3 * i);
      colors.set(rgb, 3 * i);
    }
  }
  const indices = new Uint32Array(6 * k * k);
  let t = 0;
  for (let j = 0; j < k; j++) {
    for (let q = 0; q < k; q++) {
      const a = j * (k + 1) + q;
      indices.set([a, a + 1, a + k + 2, a, a + k + 2, a + k + 1], t);
      t += 6;
    }
  }
  return { vertexCount: nv, triangleCount: 2 * k * k, positions, normals, colors, indices };
}

function lookAtPose(C: number[], target: number[], up = [0, 0, 1]): Pose {
  const f = [target[0] - C[0], target[1] - C[1], target[2] - C[2]];
  const fn = Math.hypot(f[0], f[1], f[2]);
  const z = f.map((x) => x / fn);
  // x = z × up (image right), y = z × x (image down)
  let x = [z[1] * up[2] - z[2] * up[1], z[2] * up[0] - z[0] * up[2], z[0] * up[1] - z[1] * up[0]];
  const xn = Math.hypot(x[0], x[1], x[2]) || 1;
  x = x.map((q) => q / xn);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const R = new Float64Array([...x, ...y, ...z]);
  const t = new Float64Array([0, 1, 2].map((r) => -(R[3 * r] * C[0] + R[3 * r + 1] * C[1] + R[3 * r + 2] * C[2])));
  return { R, t };
}

const intr: CameraIntrinsics = { width: 640, height: 480, f: 500, cx: 319.5, cy: 239.5, k1: 0, k2: 0 };

function ringCameras(n: number, radius: number, height: number) {
  const cams = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    cams.push({ pose: lookAtPose([radius * Math.cos(a), radius * Math.sin(a), height], [0, 0, 0]), intrinsics: intr });
  }
  return cams;
}

/** Place the viewer camera at `eye` looking at `target` (both WORLD, Z-up, 'z' mode). */
function lookFrom(v: Viewer, eye: number[], target: number[]): void {
  v.setOrbitTarget(target);
  v.camera.position.set(eye[0], eye[2], -eye[1]); // documented (x, y, z) → (x, z, −y)
  v.controls.update();
}

function pixelAt(px: { width: number; height: number; data: Uint8Array }, x: number, y: number): number[] {
  const i = 4 * ((px.height - 1 - Math.floor(y)) * px.width + Math.floor(x));
  return Array.from(px.data.subarray(i, i + 4));
}

function countIf(px: { data: Uint8Array }, f: (r: number, g: number, b: number) => boolean): number {
  let c = 0;
  for (let i = 0; i < px.data.length; i += 4) if (f(px.data[i], px.data[i + 1], px.data[i + 2])) c++;
  return c;
}

const lit = (r: number, g: number, b: number) => r > 24 || g > 24 || b > 24;

describe('frame conventions (Z-up world → three Y-up)', () => {
  it('maps world (x, y, z) to three (x, z, −y) for targets, content and screen projection', () => {
    const v = makeViewer({ autoFit: 'off' });
    v.setOrbitTarget([1, 2, 3]);
    const t = v.controls.target;
    expect(t.x).toBeCloseTo(1, 9);
    expect(t.y).toBeCloseTo(3, 9);
    expect(t.z).toBeCloseTo(-2, 9);
    // A small bright blob at world (1, 2, 3) must be drawn where worldToScreen says.
    const blob = planeCloud(0.05, 0.01, 0, [255, 255, 255]);
    for (let i = 0; i < blob.count; i++) {
      blob.positions[3 * i] += 1;
      blob.positions[3 * i + 1] += 2;
      blob.positions[3 * i + 2] += 3;
    }
    // plus a frame of dim points so the clip sphere is not degenerate
    const frame = planeCloud(3, 0.5, 0, [30, 30, 30]);
    v.setPointCloud(concatClouds(blob, frame), { pointSizePx: 3 });
    v.setDisplay({ edl: false, grid: false });
    lookFrom(v, [1, -4, 8], [1, 2, 3]);
    const s = v.worldToScreen([1, 2, 3])!;
    expect(Math.abs(s.x - W / 2)).toBeLessThan(1);
    expect(Math.abs(s.y - H / 2)).toBeLessThan(1);
    const px = v.readPixels();
    expect(pixelAt(px, s.x, s.y)[0]).toBeGreaterThan(200);
    // world +Z is screen-up, world +X is screen-right from this view
    const up = v.worldToScreen([1, 2, 3.5])!;
    const right = v.worldToScreen([1.5, 2, 3])!;
    expect(up.y).toBeLessThan(s.y - 5);
    expect(right.x).toBeGreaterThan(s.x + 5);
  });
});

describe('splat size attenuation', () => {
  function sizeScene(v: Viewer, radius: number | null): void {
    const pc: PointCloud = {
      count: 5,
      positions: new Float32Array([0, 0, 0, -2, -2, 0, 2, -2, 0, 2, 2, 0, -2, 2, 0]),
      colors: new Uint8Array(15).fill(255),
    };
    if (radius !== null) pc.radii = new Float32Array([radius, 1e-4, 1e-4, 1e-4, 1e-4]);
    v.setPointCloud(pc, radius !== null ? { splats: true } : { pointSizePx: 15 });
    v.setDisplay({ grid: false });
    lookFrom(v, [0, -1e-4, 4], [0, 0, 0]); // straight down from 4 m
  }
  function centralArea(px: { width: number; height: number; data: Uint8Array }): number {
    let c = 0;
    for (let y = H / 2 - 40; y < H / 2 + 40; y++) for (let x = W / 2 - 40; x < W / 2 + 40; x++) if (pixelAt(px, x, y)[0] > 128) c++;
    return c;
  }
  for (const edl of [false, true]) {
    it(`world-radius splats have diameter 2·r·f_px/z (EDL ${edl})`, () => {
      const v = makeViewer({ autoFit: 'off' });
      sizeScene(v, 0.3);
      v.setDisplay({ edl });
      const fPx = H / (2 * Math.tan((v.camera.fov * Math.PI) / 360));
      const expectedD = (2 * 0.3 * fPx) / 4; // ≈ 29 px
      const d = 2 * Math.sqrt(centralArea(v.readPixels()) / Math.PI);
      expect(Math.abs(d - expectedD)).toBeLessThan(1.5);
    });
  }
  it('fixed-size points have the requested CSS diameter', () => {
    const v = makeViewer({ autoFit: 'off' });
    sizeScene(v, null);
    const d = 2 * Math.sqrt(centralArea(v.readPixels()) / Math.PI);
    expect(Math.abs(d - 15)).toBeLessThan(1.5);
  });
});

describe('EDL depth output', () => {
  it('overlays behind EDL-composited points are hidden (gl_FragDepth), visible when points are hidden', () => {
    const orange = (r: number, g: number, b: number) => r > 150 && g > 80 && b < 90;
    for (const edl of [true, false]) {
      const v = makeViewer({ autoFit: 'off' });
      v.setPointCloud(planeCloud(3, 0.02, 0, [40, 60, 200]), { pointSizePx: 4 });
      // single (highlighted, orange) camera BELOW the plane, looking down
      v.setCameras([{ pose: lookAtPose([0, 0, -0.5], [0, 0.001, -3]), intrinsics: intr }], true);
      v.setDisplay({ grid: false, edl });
      lookFrom(v, [0, -1.5, 2.5], [0, 0, -0.5]);
      expect(countIf(v.readPixels(), orange)).toBe(0);
      v.setDisplay({ points: false });
      expect(countIf(v.readPixels(), orange)).toBeGreaterThan(10);
    }
  });
});

describe('camera frustum orientation', () => {
  it('a camera looking along world +X draws its frustum towards +X', () => {
    const v = makeViewer({ autoFit: 'off' });
    v.setPointCloud(planeCloud(6, 1, -1, [30, 30, 30])); // scale reference (frustum size ∝ scene)
    v.setCameras([{ pose: lookAtPose([0, 0, 0], [1, 0, 0]), intrinsics: intr }], true);
    v.setDisplay({ points: false, grid: false });
    lookFrom(v, [0.3, -1e-3, 2], [0.3, 0, 0]);
    const px = v.readPixels();
    const s0 = v.worldToScreen([0, 0, 0])!;
    const s1 = v.worldToScreen([1, 0, 0])!;
    const ux = s1.x - s0.x;
    const uy = s1.y - s0.y;
    let fwd = 0;
    let back = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const c = pixelAt(px, x + 0.5, y + 0.5);
        if (!(c[0] > 150 && c[2] < 90)) continue;
        const d = (x + 0.5 - s0.x) * ux + (y + 0.5 - s0.y) * uy;
        if (d > 2 * Math.hypot(ux, uy)) fwd++;
        else if (d < -2 * Math.hypot(ux, uy)) back++;
      }
    }
    expect(fwd).toBeGreaterThan(20);
    expect(back).toBe(0);
  });
});

describe('picking in non-trivial frames and scales', () => {
  // Reconstruction expressed in a Y-up world: X' = Q X with Q: (x, y, z) → (x, z, −y).
  const Q = (p: number[]) => [p[0], p[2], -p[1]];
  for (const pickMode of ['auto', 'cpu'] as const) {
    it(`setWorldUp('auto') + measure returns WORLD coordinates (${pickMode})`, () => {
      const v = makeViewer({ pickMode });
      const cams = ringCameras(12, 4, 2).map(({ pose, intrinsics }) => {
        const R = pose.R;
        const Rn = new Float64Array([R[0], R[2], -R[1], R[3], R[5], -R[4], R[6], R[8], -R[7]]);
        return { pose: { R: Rn, t: pose.t }, intrinsics };
      });
      const pc = planeCloud(2, 0.02, 0, [180, 180, 180]);
      for (let i = 0; i < pc.count; i++) {
        const p = Q([pc.positions[3 * i], pc.positions[3 * i + 1], pc.positions[3 * i + 2]]);
        pc.positions.set(p, 3 * i);
      }
      v.setCameras(cams, false);
      v.setPointCloud(pc, { pointSizePx: 3 });
      v.setWorldUp('auto');
      v.fitToContent();
      v.renderNow();
      const A = Q([-1, 0.5, 0]);
      const B = Q([1, -0.5, 0]);
      const r = v.measureBetweenScreenPoints(v.worldToScreen(A)!, v.worldToScreen(B)!)!;
      expect(r).not.toBeNull();
      expect(Math.hypot(r.a[0] - A[0], r.a[1] - A[1], r.a[2] - A[2])).toBeLessThan(0.02);
      expect(Math.hypot(r.b[0] - B[0], r.b[1] - B[1], r.b[2] - B[2])).toBeLessThan(0.02);
      expect(Math.abs(r.distance - Math.hypot(2, 1)) / Math.hypot(2, 1)).toBeLessThan(0.01);
    });

    it(`backyard scale (100 m, far outliers) picks within a pixel footprint (${pickMode})`, () => {
      const v = makeViewer({ pickMode });
      const ground = planeCloud(50, 0.25, 0, [120, 160, 90]);
      const n = ground.count;
      // 0.5 % far outliers at km distances
      for (let i = 0; i < n; i += 200) ground.positions.set([3000 + i, -2000, 800], 3 * i);
      v.setPointCloud(ground, { pointSizePx: 3 });
      v.renderNow();
      const A = [-20, 10, 0];
      const B = [25, -15, 0];
      const sa = v.worldToScreen(A)!;
      const sb = v.worldToScreen(B)!;
      expect(sa && sb).toBeTruthy();
      const r = v.measureBetweenScreenPoints(sa, sb)!;
      expect(r).not.toBeNull();
      const truth = Math.hypot(45, 25);
      expect(Math.abs(r.distance - truth) / truth).toBeLessThan(0.005);
      expect(Math.abs(r.a[2])).toBeLessThan(0.1);
      expect(Math.abs(r.b[2])).toBeLessThan(0.1);
    });
  }

  it('CPU picking ignores points hidden by the confidence filter (like the GPU path)', () => {
    const front = planeCloud(1, 0.02, 0.5, [255, 0, 0]);
    front.confidence = new Float32Array(front.count).fill(0.1);
    const back = planeCloud(1, 0.02, 0, [0, 255, 0]);
    back.confidence = new Float32Array(back.count).fill(0.9);
    const pc = concatClouds(front, back);
    for (const pickMode of ['auto', 'cpu'] as const) {
      const v = makeViewer({ pickMode, autoFit: 'off' });
      v.setPointCloud(pc, { pointSizePx: 3, minConfidence: 0.5 });
      lookFrom(v, [0, -2, 4], [0, 0, 0]);
      v.renderNow();
      const p = v.pickPoint({ x: W / 2, y: H / 2 })!;
      expect(p).not.toBeNull();
      expect(Math.abs(p[2])).toBeLessThan(0.02);
    }
  });

  it('pickPoints([]) is a cheap no-op', () => {
    const v = makeViewer();
    v.setPointCloud(planeCloud(1, 0.05, 0, [255, 255, 255]));
    expect(v.pickPoints([])).toEqual([]);
  });
});

describe('content updates', () => {
  it('re-setting the same cloud with NEW radii / confidence arrays uses the new values', () => {
    const v = makeViewer({ autoFit: 'off' });
    const pc = planeCloud(1, 0.25, 0, [255, 255, 255]);
    pc.radii = new Float32Array(pc.count).fill(0.01);
    v.setPointCloud(pc, { splats: true });
    v.setDisplay({ edl: false, grid: false });
    lookFrom(v, [0, -2, 3], [0, 0, 0]);
    const small = countIf(v.readPixels(), lit);
    pc.radii = new Float32Array(pc.count).fill(0.08); // same positions/colors, new radii
    v.setPointCloud(pc, { splats: true });
    const big = countIf(v.readPixels(), lit);
    expect(big).toBeGreaterThan(small * 8);
    // new confidence array (same length) must replace the old one
    pc.confidence = new Float32Array(pc.count).fill(0.9);
    v.setPointCloud(pc, { splats: true, minConfidence: 0.5 });
    expect(countIf(v.readPixels(), lit)).toBeGreaterThan(big * 0.9);
    pc.confidence = new Float32Array(pc.count).fill(0.1);
    v.setPointCloud(pc, { splats: true, minConfidence: 0.5 });
    expect(countIf(v.readPixels(), lit)).toBeLessThan(big * 0.1);
  });

  it('a viewer created in a hidden (0×0) container frames content once it gets a size', async () => {
    const el = makeContainer(0, 0);
    const v = makeViewer({}, el);
    const pc = planeCloud(1, 0.02, 0, [255, 255, 255]);
    for (let i = 0; i < pc.count; i++) pc.positions[3 * i] *= 4; // wide: 8 × 2 m
    v.setPointCloud(pc, { pointSizePx: 2 });
    el.style.width = '120px';
    el.style.height = '240px'; // portrait
    for (let i = 0; i < 5; i++) await nextFrame();
    v.resize();
    const b = v.getContentBounds()!;
    for (let k = 0; k < 8; k++) {
      const p = [k & 1 ? b.max[0] : b.min[0], k & 2 ? b.max[1] : b.min[1], k & 4 ? b.max[2] : b.min[2]];
      const q = v.worldToScreen(p)!;
      expect(q).not.toBeNull();
      expect(q.x).toBeGreaterThan(-1);
      expect(q.x).toBeLessThan(121);
      expect(q.y).toBeGreaterThan(-1);
      expect(q.y).toBeLessThan(241);
    }
  });
});

describe('render loop', () => {
  it('a mouse drag with damping ends: frames stop and full quality is restored', async () => {
    const v = makeViewer({ interactivePointBudget: 2000 });
    v.setPointCloud(planeCloud(1, 0.01, 0, [255, 255, 255]));
    await settle(v);
    const c = v.renderer.domElement;
    c.setPointerCapture = () => {};
    c.releasePointerCapture = () => {};
    const rect = c.getBoundingClientRect();
    const pe = (type: string, x: number, y: number, target: EventTarget) =>
      target.dispatchEvent(
        new PointerEvent(type, {
          clientX: rect.left + x,
          clientY: rect.top + y,
          pointerId: 1,
          pointerType: 'mouse',
          button: 0,
          buttons: type === 'pointerup' ? 0 : 1,
          isPrimary: true,
          bubbles: true,
        }),
      );
    const t0 = v.camera.position.clone();
    pe('pointerdown', 120, 90, c);
    let sawLod = false;
    for (let k = 1; k <= 6; k++) {
      pe('pointermove', 120 + 12 * k, 90, document);
      await nextFrame();
      sawLod ||= v.getStats().lodActive;
    }
    pe('pointerup', 192, 90, document);
    expect(sawLod).toBe(true);
    const n = await settle(v);
    expect(v.framePending).toBe(false);
    expect(n).toBeLessThan(600);
    expect(v.camera.position.distanceTo(t0)).toBeGreaterThan(0.1);
    expect(v.getStats().lodActive).toBe(false);
    const f = v.getStats().frames;
    await nextFrame();
    await nextFrame();
    expect(v.getStats().frames).toBe(f);
  });

  it('an animated fit interrupted by context loss completes after restore', async () => {
    const v = makeViewer({ autoFit: 'off' });
    const pc = planeCloud(1, 0.05, 0, [255, 255, 255]);
    for (let i = 0; i < pc.count; i++) pc.positions[3 * i] += 5;
    v.setPointCloud(pc);
    v.renderNow();
    const gl = v.renderer.getContext();
    const ext = gl.getExtension('WEBGL_lose_context')!;
    const canvas = v.renderer.domElement;
    v.fitToContent(true); // 450 ms animation towards the cloud centre (5, 0, 0)
    const lost = new Promise((r) => canvas.addEventListener('webglcontextlost', r, { once: true }));
    ext.loseContext();
    await lost;
    await new Promise((r) => setTimeout(r, 50));
    const restored = new Promise((r) => canvas.addEventListener('webglcontextrestored', r, { once: true }));
    ext.restoreContext();
    await restored;
    await new Promise((r) => setTimeout(r, 600));
    await settle(v);
    const t = v.getOrbitTarget();
    expect(Math.hypot(t[0] - 5, t[1], t[2])).toBeLessThan(0.05);
  });
});

describe('resources', () => {
  function trackGL(gl: WebGL2RenderingContext) {
    const kinds = ['Buffer', 'Texture', 'Framebuffer', 'Renderbuffer', 'VertexArray'] as const;
    const live: Record<string, Set<unknown>> = {};
    const created: Record<string, number> = {};
    const g = gl as unknown as Record<string, (...a: unknown[]) => unknown>;
    for (const k of kinds) {
      live[k] = new Set();
      created[k] = 0;
      const c = g[`create${k}`].bind(gl);
      const d = g[`delete${k}`].bind(gl);
      g[`create${k}`] = (...a: unknown[]) => {
        const o = c(...a);
        if (o) {
          live[k].add(o);
          created[k]++;
        }
        return o;
      };
      g[`delete${k}`] = (o: unknown) => {
        live[k].delete(o);
        return d(o);
      };
    }
    return {
      snapshot: () => Object.fromEntries(kinds.map((k) => [k, live[k].size])) as Record<string, number>,
      created: () => ({ ...created }),
    };
  }

  it('repeated content updates do not leak GL objects or scene children', async () => {
    const el = makeContainer();
    const v = makeViewer({ interactivePointBudget: 1000 }, el);
    const tr = trackGL(v.renderer.getContext() as WebGL2RenderingContext);
    const root = (v as unknown as { contentRoot: { children: unknown[] } }).contentRoot;
    let after1: Record<string, number> | null = null;
    let children1 = 0;
    for (let r = 0; r < 6; r++) {
      const pc = planeCloud(1, 0.04 - 0.002 * r, 0, [200, 200, 200]); // > 1.2·budget → LOD
      pc.radii = new Float32Array(pc.count).fill(0.01);
      pc.confidence = new Float32Array(pc.count).fill(0.8);
      v.setPointCloud(pc, { splats: true, minConfidence: 0.5 });
      v.renderNow();
      v.controls.dispatchEvent({ type: 'start' });
      await nextFrame();
      await nextFrame();
      v.controls.dispatchEvent({ type: 'end' });
      await settle(v);
      v.pickPoint({ x: W / 2, y: H / 2 });
      v.setMesh(planeMesh(1, 4 + r, [100, 100, 100]));
      v.renderNow();
      const sp = planeCloud(1, 0.5 / (r + 1), 0.2, [255, 255, 255]);
      v.setSparsePoints(sp.positions, sp.colors);
      v.setCameras(ringCameras(4 + 12 * r, 3, 1));
      v.renderNow();
      el.style.width = `${r % 2 ? W : W - 40}px`;
      await nextFrame();
      v.resize();
      v.renderNow();
      if (r === 1) {
        after1 = tr.snapshot();
        children1 = root.children.length;
      }
    }
    el.style.width = `${W - 40}px`; // same size parity as after round 1
    v.resize();
    v.renderNow();
    const end = tr.snapshot();
    expect(root.children.length).toBe(children1);
    for (const k of Object.keys(end)) expect(end[k], k).toBeLessThanOrEqual(after1![k]);
    expect(v.renderer.info.memory.geometries).toBeLessThanOrEqual(12);
    // Removing everything frees the content buffers.
    v.setPointCloud(null);
    v.setMesh(null);
    v.setSparsePoints(new Float32Array(0), new Uint8Array(0));
    v.renderNow();
    expect(tr.snapshot().Buffer).toBeLessThan(end.Buffer - 4);
  });

  it('changing display options of an unchanged cloud does not re-upload it', () => {
    const v = makeViewer();
    const tr = trackGL(v.renderer.getContext() as WebGL2RenderingContext);
    const pc = planeCloud(1, 0.02, 0, [200, 200, 200]);
    v.setPointCloud(pc, { pointSizePx: 2 });
    v.renderNow();
    const c0 = tr.created().Buffer;
    for (const s of [3, 4, 5]) {
      v.setPointCloud(pc, { pointSizePx: s, softShading: s === 4 });
      v.renderNow();
    }
    expect(tr.created().Buffer).toBe(c0);
  });
});

describe('error handling / platform', () => {
  it('WebGL context creation failure throws and leaves the container untouched', () => {
    const el = makeContainer();
    const proto = HTMLCanvasElement.prototype as unknown as { getContext: unknown };
    const orig = proto.getContext;
    proto.getContext = function () {
      return null;
    };
    try {
      expect(() => new Viewer(el)).toThrow();
    } finally {
      proto.getContext = orig;
    }
    expect(el.querySelector('canvas')).toBeNull();
    expect(el.querySelector('.viewer-measure-label')).toBeNull();
    expect(el.style.position).toBe('');
  });

  it('canvas suppresses page gestures, text selection and the iOS callout', () => {
    const v = makeViewer();
    const s = v.renderer.domElement.style as CSSStyleDeclaration & Record<string, string>;
    expect(s.touchAction).toBe('none');
    expect(s.userSelect || s.webkitUserSelect).toBe('none');
    const ev = new Event('gesturestart', { cancelable: true });
    v.renderer.domElement.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(true);
  });

  it('screenshot() rejects while the context is lost instead of returning a blank image', async () => {
    const v = makeViewer();
    v.setPointCloud(planeCloud(1, 0.05, 0, [255, 255, 255]));
    const canvas = v.renderer.domElement;
    const ext = v.renderer.getContext().getExtension('WEBGL_lose_context')!;
    const lost = new Promise((r) => canvas.addEventListener('webglcontextlost', r, { once: true }));
    ext.loseContext();
    await lost;
    await expect(v.screenshot()).rejects.toThrow();
    const restored = new Promise((r) => canvas.addEventListener('webglcontextrestored', r, { once: true }));
    await new Promise((r) => setTimeout(r, 50));
    ext.restoreContext();
    await restored;
    const blob = await v.screenshot();
    expect(blob.type).toBe('image/png');
  });
});

describe('device pixel ratio 2 (iPhone path: DPR 3 capped to 2)', () => {
  function withDpr<T>(dpr: number, f: () => T): T {
    Object.defineProperty(window, 'devicePixelRatio', { value: dpr, configurable: true, writable: true });
    try {
      return f();
    } finally {
      delete (window as unknown as Record<string, unknown>).devicePixelRatio;
    }
  }
  for (const dpr of [2, 1.5]) {
    for (const pickMode of ['auto', 'cpu'] as const) {
      it(`DPR ${dpr}: buffer size, splat size and picking agree (${pickMode})`, () => {
        const v = withDpr(dpr, () => makeViewer({ pickMode, autoFit: 'off' }));
        expect(v.getStats().pixelRatio).toBe(dpr);
        const px0 = v.readPixels();
        expect(px0.width).toBe(Math.floor(W * dpr));
        expect(px0.height).toBe(Math.floor(H * dpr));
        v.setPointCloud(planeCloud(2, 0.02, 0, [180, 180, 180]), { pointSizePx: 3 });
        lookFrom(v, [0, -3, 3], [0, 0, 0]);
        v.renderNow();
        const A = [-1, 0.5, 0];
        const B = [1, -0.5, 0];
        const r = v.measureBetweenScreenPoints(v.worldToScreen(A)!, v.worldToScreen(B)!)!;
        expect(r).not.toBeNull();
        expect(Math.hypot(r.a[0] - A[0], r.a[1] - A[1], r.a[2] - A[2])).toBeLessThan(0.015);
        expect(Math.hypot(r.b[0] - B[0], r.b[1] - B[1], r.b[2] - B[2])).toBeLessThan(0.015);
      });
    }
  }
  it('DPR 2: fixed point size is in CSS px (15 CSS px → 30 device px)', () => {
    const v = withDpr(2, () => makeViewer({ autoFit: 'off' }));
    v.setPointCloud(
      { count: 5, positions: new Float32Array([0, 0, 0, -2, -2, 0, 2, -2, 0, 2, 2, 0, -2, 2, 0]), colors: new Uint8Array(15).fill(255) },
      { pointSizePx: 15 },
    );
    v.setDisplay({ grid: false });
    lookFrom(v, [0, -1e-4, 4], [0, 0, 0]);
    const px = v.readPixels();
    let c = 0;
    for (let y = px.height / 2 - 60; y < px.height / 2 + 60; y++)
      for (let x = px.width / 2 - 60; x < px.width / 2 + 60; x++) if (px.data[4 * (y * px.width + x)] > 128) c++;
    expect(Math.abs(2 * Math.sqrt(c / Math.PI) - 30)).toBeLessThan(2);
  });
});

describe('more picking / rendering edge cases', () => {
  it('picks at the very edge of the canvas (window partly outside)', () => {
    for (const pickMode of ['auto', 'cpu'] as const) {
      const v = makeViewer({ pickMode, autoFit: 'off' });
      v.setPointCloud(planeCloud(4, 0.02, 0, [180, 180, 180]), { pointSizePx: 3 });
      lookFrom(v, [0, -0.5, 1.5], [0, 0, 0]); // plane fills the view
      v.renderNow();
      for (const s of [{ x: 1, y: 1 }, { x: W - 1, y: H - 1 }, { x: 0.2, y: H / 2 }]) {
        const p = v.pickPoint(s)!;
        expect(p, `${pickMode} ${s.x},${s.y}`).not.toBeNull();
        expect(Math.abs(p[2])).toBeLessThan(0.01);
        const back = v.worldToScreen(p)!;
        expect(Math.hypot(back.x - s.x, back.y - s.y)).toBeLessThan(1.5);
      }
    }
  });

  it("a cloud whose points all coincide (degenerate bounds) is still drawn and pickable ('auto' up: no grid)", () => {
    const v = makeViewer({ autoFit: 'first' });
    v.setWorldUp('auto');
    const n = 10;
    v.setPointCloud({ count: n, positions: new Float32Array(3 * n).fill(2), colors: new Uint8Array(3 * n).fill(255) }, { pointSizePx: 9 });
    v.setDisplay({ grid: false });
    const s = v.worldToScreen([2, 2, 2])!;
    expect(Math.abs(s.x - W / 2)).toBeLessThan(1);
    const px = v.readPixels();
    expect(pixelAt(px, s.x, s.y)[0]).toBeGreaterThan(200);
    const p = v.pickPoint(s)!;
    expect(p).not.toBeNull();
    expect(Math.hypot(p[0] - 2, p[1] - 2, p[2] - 2)).toBeLessThan(0.05);
    const st = v.getStats();
    expect(st.far / st.near).toBeGreaterThan(1.01); // not a zero-thickness depth range
    expect(st.far / st.near).toBeLessThan(1e4);
  });

  it('EDL path: mesh pixels keep their exact (unlit) colour next to points', () => {
    const v = makeViewer({ autoFit: 'off' });
    v.setMesh(planeMesh(2, 8, [0, 200, 0]));
    const pts = planeCloud(0.3, 0.02, 0.3, [255, 255, 255]);
    for (let i = 0; i < pts.count; i++) pts.positions[3 * i] += 1.2;
    v.setPointCloud(pts, { pointSizePx: 3 });
    v.setDisplay({ unlit: true, grid: false, edl: true });
    lookFrom(v, [0, -2, 3], [0, 0, 0]);
    const px = v.readPixels();
    const s = v.worldToScreen([-0.5, 0, 0])!;
    const c = pixelAt(px, s.x, s.y);
    expect(Math.abs(c[1] - 200)).toBeLessThan(4);
    expect(c[0]).toBeLessThan(4);
    // and the points next to it are there (EDL path really used)
    const sp = v.worldToScreen([1.2, 0, 0.3])!;
    expect(pixelAt(px, sp.x, sp.y)[0]).toBeGreaterThan(150);
  });

  it("transparent background: alpha 0 where empty, 255 on content (EDL and direct)", () => {
    for (const edl of [true, false]) {
      const v = makeViewer({ background: 'transparent', autoFit: 'off' });
      v.setPointCloud(planeCloud(0.5, 0.01, 0, [255, 0, 0]), { pointSizePx: 3 });
      v.setDisplay({ grid: false, edl });
      lookFrom(v, [0, -1, 2], [0, 0, 0]);
      const px = v.readPixels();
      expect(pixelAt(px, W / 2, H / 2)[3]).toBe(255);
      expect(pixelAt(px, 1, 1)[3]).toBe(0);
    }
  });

  it('picking while the LOD subset is displayed uses the full cloud', async () => {
    const v = makeViewer({ interactivePointBudget: 500 });
    v.setPointCloud(planeCloud(1, 0.01, 0, [255, 255, 255]), { pointSizePx: 2 });
    v.controls.dispatchEvent({ type: 'start' });
    await nextFrame();
    await nextFrame();
    expect(v.getStats().lodActive).toBe(true);
    const A = [0.33, -0.21, 0];
    const p = v.pickPoint(v.worldToScreen(A)!)!;
    expect(Math.hypot(p[0] - A[0], p[1] - A[1], p[2] - A[2])).toBeLessThan(0.01);
    v.controls.dispatchEvent({ type: 'end' });
    await settle(v);
  });

  it('measure mode driven by real touch taps reports world points', () => {
    const v = makeViewer();
    v.setMesh(planeMesh(2, 16, [180, 180, 180]));
    v.renderNow();
    const c = v.renderer.domElement;
    c.setPointerCapture = () => {};
    c.releasePointerCapture = () => {};
    const results: { d: number; a: Float64Array; b: Float64Array }[] = [];
    v.startMeasure((d, a, b) => results.push({ d, a, b }));
    const rect = c.getBoundingClientRect();
    let id = 10;
    const tap = (p: { x: number; y: number }) => {
      id++;
      for (const type of ['pointerdown', 'pointerup']) {
        c.dispatchEvent(
          new PointerEvent(type, { clientX: rect.left + p.x, clientY: rect.top + p.y, pointerId: id, pointerType: 'touch', isPrimary: true, bubbles: true }),
        );
      }
    };
    const A = [-0.7, 0.3, 0];
    const B = [0.9, -0.4, 0];
    tap(v.worldToScreen(A)!);
    tap(v.worldToScreen(B)!);
    expect(results.length).toBe(1);
    const r = results[0];
    expect(Math.hypot(r.a[0] - A[0], r.a[1] - A[1], r.a[2] - A[2])).toBeLessThan(0.02);
    expect(Math.hypot(r.b[0] - B[0], r.b[1] - B[1], r.b[2] - B[2])).toBeLessThan(0.02);
    expect(r.d).toBeCloseTo(Math.hypot(1.6, 0.7), 1);
  });
});

describe('CPU fallback on meshes', () => {
  it('hits the interior of large triangles (coarse mesh, zoomed in) on the tap ray', () => {
    for (const pickMode of ['auto', 'cpu'] as const) {
      const v = makeViewer({ pickMode, autoFit: 'off' });
      v.setMesh(planeMesh(2, 4, [180, 180, 180])); // 1 m cells
      lookFrom(v, [0.25, -1.2, 1.5], [0.25, 0.25, 0]);
      v.renderNow();
      for (const A of [[0.5, 0.5, 0], [0.3, 0.2, 0], [-0.4, 0.7, 0]]) {
        const s = v.worldToScreen(A)!;
        const p = v.pickPoint(s)!;
        expect(p, `${pickMode} ${A}`).not.toBeNull();
        expect(Math.hypot(p[0] - A[0], p[1] - A[1], p[2] - A[2]), `${pickMode} ${A}`).toBeLessThan(0.01);
      }
      expect(v.getStats().pickMode).toBe(pickMode === 'auto' ? 'gpu' : 'cpu');
      // hidden mesh is not pickable
      v.setDisplay({ mesh: false });
      expect(v.pickPoint(v.worldToScreen([0.5, 0.5, 0])!)).toBeNull();
    }
  });

  it('occlusion: a mesh in front of points wins on the CPU path too', () => {
    const v = makeViewer({ pickMode: 'cpu', autoFit: 'off' });
    const m = planeMesh(1, 2, [180, 180, 180]);
    for (let i = 0; i < m.vertexCount; i++) m.positions[3 * i + 2] = 0.5;
    v.setMesh(m);
    v.setPointCloud(planeCloud(1, 0.02, 0, [255, 255, 255]), { pointSizePx: 3 });
    lookFrom(v, [0, -1.5, 3], [0, 0, 0.25]);
    v.renderNow();
    const p = v.pickPoint({ x: W / 2, y: H / 2 })!;
    expect(Math.abs(p[2] - 0.5)).toBeLessThan(0.01);
  });
});

describe('TapDetector (review)', () => {
  it('a pinch between two taps does not turn them into a double tap', async () => {
    const { TapDetector } = await import('./controls');
    const el = makeContainer(100, 100);
    const doubles: number[] = [];
    const taps: number[] = [];
    const td = new TapDetector(el, { onTap: (t) => taps.push(t.x), onDoubleTap: (t) => doubles.push(t.x) });
    const r = el.getBoundingClientRect();
    const ev = (type: string, x: number, y: number, id: number) =>
      el.dispatchEvent(new PointerEvent(type, { clientX: r.left + x, clientY: r.top + y, pointerId: id, pointerType: 'touch', bubbles: true }));
    ev('pointerdown', 10, 10, 1);
    ev('pointerup', 10, 10, 1);
    ev('pointerdown', 20, 20, 2);
    ev('pointerdown', 60, 60, 3);
    ev('pointerup', 20, 20, 2);
    ev('pointerup', 60, 60, 3);
    ev('pointerdown', 12, 11, 4);
    ev('pointerup', 12, 11, 4);
    expect(taps).toEqual([10, 12]);
    expect(doubles.length).toBe(0);
    td.dispose();
  });
});

describe('lifecycle (review)', () => {
  it('dispose() restores the container position it changed (and keeps one set by the app)', () => {
    const a = makeContainer();
    const va = new Viewer(a);
    expect(a.style.position).toBe('relative');
    va.dispose();
    expect(a.style.position).toBe('');
    const b = makeContainer();
    b.style.position = 'absolute';
    const vb = new Viewer(b);
    vb.dispose();
    expect(b.style.position).toBe('absolute');
  });
});
