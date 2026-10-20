import { afterEach, describe, expect, it } from 'vitest';
import type { CameraIntrinsics, PointCloud, Pose, TriangleMesh } from '../core/types';
import { TapDetector } from './controls';
import { Viewer } from './viewer';

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

function makeViewer(opts: ConstructorParameters<typeof Viewer>[1] = {}): Viewer {
  const v = new Viewer(makeContainer(), { background: '#000000', ...opts });
  viewers.push(v);
  return v;
}

afterEach(() => {
  for (const v of viewers.splice(0)) v.dispose();
  for (const c of containers.splice(0)) c.remove();
});

const nextFrame = () => new Promise<number>((r) => requestAnimationFrame(r));

/** Plane of points on z = z0 (world, Z-up), [-s, s]², grid spacing `step`. */
function planeCloud(s: number, step: number, z0: number, rgb: [number, number, number]): PointCloud {
  const k = Math.floor((2 * s) / step) + 1;
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
  return { count: a.count + b.count, positions, colors };
}

/** Grid mesh on z = 0, [-s, s]², `k` quads per side, constant colour. */
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

function lookAtPose(C: number[], target: number[]): Pose {
  // Camera z towards target; x horizontal (Z-up world); y = z × x (down).
  const f = [target[0] - C[0], target[1] - C[1], target[2] - C[2]];
  const fn = Math.hypot(f[0], f[1], f[2]);
  const z = f.map((v) => v / fn);
  let x = [z[1], -z[0], 0]; // z × up(0,0,1)
  const xn = Math.hypot(x[0], x[1], x[2]) || 1;
  x = x.map((v) => v / xn);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const R = new Float64Array([...x, ...y, ...z]);
  const t = new Float64Array([
    -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
    -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
    -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
  ]);
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

function countNonBackground(px: { data: Uint8Array }, thr = 24): number {
  let c = 0;
  for (let i = 0; i < px.data.length; i += 4) {
    if (px.data[i] > thr || px.data[i + 1] > thr || px.data[i + 2] > thr) c++;
  }
  return c;
}

function pixelAt(px: { width: number; height: number; data: Uint8Array }, x: number, y: number): number[] {
  // x, y in CSS px from the top-left; readPixels rows are bottom-up.
  const i = 4 * ((px.height - 1 - Math.floor(y)) * px.width + Math.floor(x));
  return Array.from(px.data.subarray(i, i + 4));
}

describe('Viewer rendering', () => {
  it('renders a point cloud with and without EDL', () => {
    const v = makeViewer();
    v.setPointCloud(planeCloud(1, 0.02, 0, [230, 60, 40]), { pointSizePx: 3 });
    const stats = v.getStats();
    expect(stats.densePoints).toBe(101 * 101);
    const withEdl = v.readPixels();
    expect(withEdl.width).toBe(W);
    expect(countNonBackground(withEdl)).toBeGreaterThan(0.1 * W * H);
    v.setDisplay({ edl: false });
    const noEdl = v.readPixels();
    expect(countNonBackground(noEdl)).toBeGreaterThan(0.1 * W * H);
    // Points are drawn in their sRGB colour (no EDL): centre pixel ≈ (230, 60, 40).
    const c = pixelAt(noEdl, W / 2, H / 2);
    expect(Math.abs(c[0] - 230)).toBeLessThan(4);
    expect(Math.abs(c[1] - 60)).toBeLessThan(4);
    expect(Math.abs(c[2] - 40)).toBeLessThan(4);
    v.setDisplay({ points: false });
    expect(countNonBackground(v.readPixels())).toBeLessThan(0.1 * W * H); // only grid lines
  });

  it('EDL darkens depth discontinuities and nothing else changes without it', () => {
    const v = makeViewer();
    expect(v.getStats().edlSupported).toBe(true);
    const lower = planeCloud(1, 0.01, 0, [255, 255, 255]);
    const upper = planeCloud(0.4, 0.01, 0.5, [255, 255, 255]);
    v.setPointCloud(concatClouds(lower, upper), { pointSizePx: 3 });
    v.setDisplay({ grid: false });
    const darkened = (px: { data: Uint8Array }) => {
      let n = 0;
      for (let i = 0; i < px.data.length; i += 4) if (px.data[i] > 24 && px.data[i] < 180) n++;
      return n;
    };
    v.setDisplay({ edl: false });
    const off = v.readPixels();
    v.setDisplay({ edl: true, edlStrength: 1 });
    const on = v.readPixels();
    expect(darkened(off)).toBe(0);
    expect(darkened(on)).toBeGreaterThan(100);
  });

  it('renders a mesh lit and unlit (no EDL shading on meshes)', () => {
    const v = makeViewer();
    v.setMesh(planeMesh(1, 8, [0, 200, 0]));
    expect(v.getStats().triangles).toBe(128);
    const lit = v.readPixels();
    expect(countNonBackground(lit)).toBeGreaterThan(0.12 * W * H);
    const cl = pixelAt(lit, W / 2, H / 2);
    expect(cl[1]).toBeGreaterThan(60);
    expect(cl[0]).toBeLessThan(40);
    v.setDisplay({ unlit: true });
    const unlit = v.readPixels();
    const cu = pixelAt(unlit, W / 2, H / 2);
    expect(Math.abs(cu[1] - 200)).toBeLessThan(4);
    expect(cu[0]).toBeLessThan(4);
    v.setDisplay({ mesh: false, grid: false });
    expect(countNonBackground(v.readPixels())).toBe(0);
  });

  it('renders camera frustums and the trajectory (cameras only)', () => {
    const v = makeViewer();
    const cams = ringCameras(8, 3, 1);
    v.setCameras(cams, true);
    expect(v.getStats().cameras).toBe(8);
    const px = v.readPixels();
    expect(countNonBackground(px)).toBeGreaterThan(150);
    // highlighted (orange) last camera present: r high, b low
    let orange = 0;
    for (let i = 0; i < px.data.length; i += 4) if (px.data[i] > 150 && px.data[i + 2] < 90) orange++;
    expect(orange).toBeGreaterThan(5);
    v.setDisplay({ cameras: false });
    expect(countNonBackground(v.readPixels())).toBe(0);
  });

  it('updates live sparse points in place and grows its buffers', () => {
    const v = makeViewer();
    const small = planeCloud(1, 0.5, 0, [200, 200, 200]);
    v.setSparsePoints(small.positions, small.colors);
    expect(v.getStats().sparsePoints).toBe(25);
    const big = planeCloud(1, 0.02, 0, [200, 200, 200]);
    v.setSparsePoints(big.positions, big.colors);
    expect(v.getStats().sparsePoints).toBe(big.count);
    expect(countNonBackground(v.readPixels())).toBeGreaterThan(0.1 * W * H);
    v.setSparsePoints(small.positions, small.colors);
    expect(v.getStats().sparsePoints).toBe(25);
    v.setSparsePoints(new Float32Array(0), new Uint8Array(0));
    expect(v.getStats().sparsePoints).toBe(0);
  });

  it('draws round splats sized by per-point radius', () => {
    const v = makeViewer();
    const pc = planeCloud(1, 0.25, 0, [255, 255, 255]); // sparse: 9×9 points
    v.setPointCloud(pc, { pointSizePx: 1 });
    v.setDisplay({ edl: false, grid: false });
    const small = countNonBackground(v.readPixels());
    pc.radii = new Float32Array(pc.count).fill(0.06);
    v.setPointCloud(pc, { splats: true });
    const big = countNonBackground(v.readPixels());
    expect(big).toBeGreaterThan(small * 8);
    // confidence threshold hides low-confidence points (here: half of them)
    pc.confidence = new Float32Array(pc.count).map((_, i) => (i % 2 ? 0.9 : 0.1));
    v.setPointCloud(pc, { splats: true, minConfidence: 0.5, softShading: true });
    const half = countNonBackground(v.readPixels());
    expect(half).toBeLessThan(big * 0.75);
    expect(half).toBeGreaterThan(big * 0.25);
  });
});

describe('Viewer interaction', () => {
  it('draws the interactive subset of huge clouds only while the view moves', async () => {
    const v = makeViewer({ interactivePointBudget: 2000 });
    v.setPointCloud(planeCloud(1, 0.01, 0, [255, 255, 255])); // 40401 points
    v.renderNow();
    expect(v.getStats().lodActive).toBe(false);
    v.controls.dispatchEvent({ type: 'start' });
    await nextFrame();
    await nextFrame();
    expect(v.getStats().lodActive).toBe(true);
    v.controls.dispatchEvent({ type: 'end' });
    for (let i = 0; i < 30 && v.framePending; i++) await nextFrame();
    expect(v.getStats().lodActive).toBe(false);
  });

  it("autoFit 'follow' tracks growing live content until the user moves the view", async () => {
    const v = makeViewer({ autoFit: 'follow' });
    const a = planeCloud(0.5, 0.1, 0, [200, 200, 200]);
    v.setSparsePoints(a.positions, a.colors);
    const t0 = v.getOrbitTarget();
    expect(Math.hypot(t0[0], t0[1])).toBeLessThan(0.1);
    const b = planeCloud(0.5, 0.1, 0, [200, 200, 200]);
    for (let i = 0; i < b.count; i++) b.positions[3 * i] += 10;
    v.setSparsePoints(b.positions, b.colors);
    for (let i = 0; i < 60 && v.framePending; i++) await nextFrame();
    const t1 = v.getOrbitTarget();
    expect(Math.abs(t1[0] - 10)).toBeLessThan(0.2);
    v.controls.dispatchEvent({ type: 'start' });
    v.controls.dispatchEvent({ type: 'end' });
    v.setSparsePoints(a.positions, a.colors);
    for (let i = 0; i < 60 && v.framePending; i++) await nextFrame();
    const t2 = v.getOrbitTarget();
    expect(Math.abs(t2[0] - 10)).toBeLessThan(0.2);
  });
});

describe('Viewer framing', () => {
  it('fitToContent frames the robust bounds and ignores outliers', () => {
    const v = makeViewer();
    const n = 20000;
    const positions = new Float32Array(3 * n);
    const colors = new Uint8Array(3 * n).fill(200);
    let s = 1;
    const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296);
    for (let i = 0; i < n; i++) {
      positions[3 * i] = 100 + (rnd() - 0.5) * 8;
      positions[3 * i + 1] = 50 + (rnd() - 0.5) * 6;
      positions[3 * i + 2] = 20 + (rnd() - 0.5) * 2;
    }
    for (let i = 0; i < n / 100; i++) positions.set([1e4 * rnd(), -1e4 * rnd(), 5e3 * rnd()], 3 * (i * 100 + 7));
    v.setPointCloud({ count: n, positions, colors });
    // Move away, then re-fit.
    v.controls.target.set(0, 0, 0);
    v.camera.position.set(500, 500, 500);
    v.controls.update();
    v.fitToContent();
    const b = v.getContentBounds()!;
    expect(b.radius).toBeLessThan(10);
    const c = v.worldToScreen(b.center)!;
    expect(Math.abs(c.x - W / 2)).toBeLessThan(0.05 * W);
    expect(Math.abs(c.y - H / 2)).toBeLessThan(0.05 * H);
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let k = 0; k < 8; k++) {
      const p = [k & 1 ? b.max[0] : b.min[0], k & 2 ? b.max[1] : b.min[1], k & 4 ? b.max[2] : b.min[2]];
      const q = v.worldToScreen(p)!;
      expect(q).not.toBeNull();
      minX = Math.min(minX, q.x);
      maxX = Math.max(maxX, q.x);
      minY = Math.min(minY, q.y);
      maxY = Math.max(maxY, q.y);
    }
    expect(minX).toBeGreaterThan(-1);
    expect(maxX).toBeLessThan(W + 1);
    expect(minY).toBeGreaterThan(-1);
    expect(maxY).toBeLessThan(H + 1);
    expect(Math.max(maxX - minX, (maxY - minY) * (W / H))).toBeGreaterThan(0.4 * W);
    const st = v.getStats();
    expect(st.far / st.near).toBeLessThan(1e5);
    const px = v.readPixels();
    expect(countNonBackground(px)).toBeGreaterThan(0.08 * W * H);
  });

  it("setWorldUp('auto') estimates up from the cameras (Y-up reconstruction)", () => {
    const v = makeViewer();
    // Cameras in a Z-up world, then the whole reconstruction rotated so that up = +Y.
    const cams = ringCameras(12, 3, 1).map(({ pose, intrinsics }) => {
      // X_w' = Q X_w with Q: (x, y, z) -> (x, z, -y) (Z-up -> Y-up); R' = R Qᵀ
      const R = pose.R;
      const Rn = new Float64Array([R[0], R[2], -R[1], R[3], R[5], -R[4], R[6], R[8], -R[7]]);
      return { pose: { R: Rn, t: pose.t }, intrinsics };
    });
    v.setCameras(cams);
    v.setWorldUp('auto');
    const up = v.getWorldUp();
    expect(up[1]).toBeGreaterThan(Math.cos((2 * Math.PI) / 180));
    v.setWorldUp('z');
    expect(Array.from(v.getWorldUp())).toEqual([0, 0, 1]);
  });

  it("setWorldUp('auto') takes each camera's upCam (images upright for another rotation)", () => {
    const v = makeViewer();
    v.setWorldUp('auto');
    // a half orbit in the Y-up reconstruction, images turned a quarter about the optical axis (raw
    // landscape buffers of a phone held in portrait): image up is horizontal, camera −x is up
    const cams = ringCameras(12, 3, 1)
      .slice(0, 6)
      .map(({ pose, intrinsics }) => {
        const R = pose.R;
        const Ry = [R[0], R[2], -R[1], R[3], R[5], -R[4], R[6], R[8], -R[7]];
        // rows (y, −x, z): Rz(−90°) · R
        const Rn = new Float64Array([Ry[3], Ry[4], Ry[5], -Ry[0], -Ry[1], -Ry[2], Ry[6], Ry[7], Ry[8]]);
        return { pose: { R: Rn, t: Float64Array.of(pose.t[1], -pose.t[0], pose.t[2]) }, intrinsics, upCam: [-1, 0, 0] };
      });
    v.setCameras(cams);
    expect(v.getWorldUp()[1]).toBeGreaterThan(Math.cos((2 * Math.PI) / 180));
    v.setCameras(cams.map(({ pose, intrinsics }) => ({ pose, intrinsics })));
    expect(Math.abs(v.getWorldUp()[1])).toBeLessThan(0.5);
  });

  it('fitToContent(overview) frames from above whatever the current view direction', () => {
    const v = makeViewer();
    v.setPointCloud(planeCloud(2, 0.25, 0, [200, 200, 200]));
    v.fitToContent(false);
    // look from the side (horizontal), then ask for the overview
    v.camera.position.set(10, 0.01, 0);
    v.controls.target.set(0, 0, 0);
    v.controls.update();
    v.fitToContent(false, true);
    const d = v.camera.position.clone().sub(v.controls.target).normalize();
    expect(d.y).toBeGreaterThan(0.3); // three's +Y = world up
    // without it, the side view direction is kept
    v.camera.position.set(10, 0.01, 0);
    v.controls.update();
    v.fitToContent(false);
    expect(Math.abs(v.camera.position.clone().sub(v.controls.target).normalize().y)).toBeLessThan(0.01);
  });

  describe('first fit from the side the photos were taken from, after the cameras arrive', () => {
    // A small cloud with cameras on an arc on its +X side, in a frame with up = −Y (e.g. the first
    // camera's): (x, y, z) → (x, −z, y). The screens set the content first, then the cameras.
    const q = (p: ArrayLike<number>) => [p[0], -p[2], p[1]];
    function content(side = 1) {
      const n = 400;
      const positions = new Float32Array(3 * n);
      for (let i = 0; i < n; i++) positions.set(q([0.3 * Math.cos(i), 0.3 * Math.sin(1.7 * i), 0.3 * Math.cos(2.3 * i)]), 3 * i);
      const pc: PointCloud = { count: n, positions, colors: new Uint8Array(3 * n).fill(200) };
      const cams = [];
      for (let i = 0; i < 9; i++) {
        const a = ((i - 4) / 4) * 0.7;
        const { R, t } = lookAtPose([side * 3 * Math.cos(a), 3 * Math.sin(a), 1], [0, 0, 0]);
        // R' = R Qᵀ, t' = t
        cams.push({ pose: { R: new Float64Array([R[0], -R[2], R[1], R[3], -R[5], R[4], R[6], -R[8], R[7]]), t }, intrinsics: intr });
      }
      return { pc, cams };
    }
    /** View direction (world, from the target) and its elevation above the world up (deg). */
    function view(v: Viewer): { d: number[]; elevDeg: number } {
      const p = v.camera.position.clone();
      (v as unknown as { contentRoot: { worldToLocal(p: unknown): void } }).contentRoot.worldToLocal(p);
      const c = v.getOrbitTarget();
      const d = [p.x - c[0], p.y - c[1], p.z - c[2]];
      const n = Math.hypot(d[0], d[1], d[2]);
      const u = v.getWorldUp();
      const dn = d.map((x) => x / n);
      return { d: dn, elevDeg: (Math.asin(dn[0] * u[0] + dn[1] * u[1] + dn[2] * u[2]) * 180) / Math.PI };
    }

    it("viewer screen order: content, cameras, then fitToContent ('auto' up)", () => {
      const v = makeViewer();
      const { pc, cams } = content();
      v.setWorldUp('auto');
      v.setPointCloud(pc);
      v.setMesh(null);
      v.setCameras(cams, false);
      v.fitToContent();
      expect(v.getWorldUp()[1]).toBeLessThan(-0.99);
      const { d, elevDeg } = view(v);
      expect(d[0]).toBeGreaterThan(0.5); // towards the cameras (+X)
      expect(Math.abs(elevDeg - 28.6)).toBeLessThan(5);
    });

    it("processing preview order: autoFit 'follow' refits when the cameras arrive", async () => {
      const v = makeViewer({ autoFit: 'follow' });
      const { pc, cams } = content();
      v.setWorldUp('auto');
      v.setPointCloud(pc);
      v.setCameras(cams, false);
      for (let i = 0; i < 60 && v.framePending; i++) await nextFrame();
      const { d, elevDeg } = view(v);
      expect(d[0]).toBeGreaterThan(0.5);
      expect(Math.abs(elevDeg - 28.6)).toBeLessThan(5);
    });

    it('a view the user moved is kept when other cameras arrive', () => {
      const v = makeViewer();
      const { pc } = content();
      v.setWorldUp('auto');
      v.setPointCloud(pc);
      v.controls.dispatchEvent({ type: 'start' });
      v.controls.dispatchEvent({ type: 'end' });
      const before = view(v).d;
      v.setCameras(content().cams, false);
      v.fitToContent();
      // the auto up turned the view with the content: its direction relative to the content is kept
      const after = view(v).d;
      expect(after[0] * before[0] + after[1] * before[1] + after[2] * before[2]).toBeGreaterThan(0.999);
    });
  });

  it('setViewPose puts the view at a camera pose (x right, y down, z forward) and gives it back', async () => {
    const v = makeViewer();
    v.setPointCloud(planeCloud(2, 0.25, 0, [200, 200, 200]));
    // camera at (0, −3, 1) looking along +Y (world), image x = +X, image y = −Z (down), rolled 30° about its axis
    const C = [0, -3, 1];
    const a = Math.PI / 6;
    const R0 = [1, 0, 0, 0, 0, -1, 0, 1, 0]; // rows: x_cam = X, y_cam = −Z, z_cam = Y
    const Rroll = [Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a), 0, 0, 0, 1];
    const R = new Float64Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) R[3 * i + j] += Rroll[3 * i + k] * R0[3 * k + j];
    const t = new Float64Array(3);
    for (let i = 0; i < 3; i++) t[i] = -(R[3 * i] * C[0] + R[3 * i + 1] * C[1] + R[3 * i + 2] * C[2]);
    v.setViewPose({ R, t }, 60);
    // project X through the pinhole of (R, t, f from the vertical FOV): the viewer must agree
    const f = H / 2 / Math.tan(Math.PI / 6);
    for (const X of [[0, 0, 1], [0.5, 0, 1.2], [-0.4, 0.5, 0.6], [1, 1, 0]]) {
      const xc = [0, 1, 2].map((i) => R[3 * i] * X[0] + R[3 * i + 1] * X[1] + R[3 * i + 2] * X[2] + t[i]);
      const s = v.worldToScreen(X)!;
      expect(s.x).toBeCloseTo(W / 2 + (f * xc[0]) / xc[2], 3);
      expect(s.y).toBeCloseTo(H / 2 + (f * xc[1]) / xc[2], 3);
    }
    // the orbit controls are off while locked: rendering frames keep the pose (and its roll)
    expect(v.controls.enabled).toBe(false);
    const before = v.worldToScreen([0.5, 0, 1.2])!;
    await nextFrame();
    await nextFrame();
    const after = v.worldToScreen([0.5, 0, 1.2])!;
    expect(after.x).toBeCloseTo(before.x, 6);
    expect(after.y).toBeCloseTo(before.y, 6);
    // framing is off while locked, and an auto-up change turns the camera with the content
    v.fitToContent(false);
    expect(v.worldToScreen([0.5, 0, 1.2])!.x).toBeCloseTo(before.x, 6);
    v.setWorldUp('auto');
    v.setCameras(ringCameras(8, 3, 1).map(({ pose, intrinsics }) => ({ pose: { R: new Float64Array([pose.R[0], pose.R[2], -pose.R[1], pose.R[3], pose.R[5], -pose.R[4], pose.R[6], pose.R[8], -pose.R[7]]), t: pose.t }, intrinsics })));
    const turned = v.worldToScreen([0.5, 0, 1.2])!;
    expect(turned.x).toBeCloseTo(before.x, 3);
    expect(turned.y).toBeCloseTo(before.y, 3);
    v.setViewPose(null);
    expect(v.controls.enabled).toBe(true);
    expect(v.camera.fov).toBe(50);
  });
});

describe('Viewer picking / measuring', () => {
  const cases = [
    ['auto', 'mesh'],
    ['auto', 'points'],
    ['cpu', 'points'],
  ] as const;
  for (const [pickMode, content] of cases) {
    it(`measures distances on a synthetic plane (${pickMode} picking, ${content})`, () => {
      const v = makeViewer({ pickMode });
      if (content === 'mesh') v.setMesh(planeMesh(2, 16, [180, 180, 180]));
      else v.setPointCloud(planeCloud(2, 0.02, 0, [180, 180, 180]), { pointSizePx: 3 });
      v.renderNow();
      const A = [-1, 0.5, 0];
      const B = [1, -0.5, 0];
      const sa = v.worldToScreen(A)!;
      const sb = v.worldToScreen(B)!;
      expect(sa).not.toBeNull();
      const t0 = performance.now();
      const r = v.measureBetweenScreenPoints(sa, sb)!;
      const dt = performance.now() - t0;
      expect(r).not.toBeNull();
      expect(v.getStats().pickMode).toBe(pickMode === 'auto' ? 'gpu' : 'cpu');
      const truth = Math.hypot(2, 1);
      expect(Math.abs(r.distance - truth) / truth).toBeLessThan(0.01);
      // 240×180 px canvas, oblique view: 1 px ≈ 2–4 cm on the plane.
      expect(Math.hypot(r.a[0] - A[0], r.a[1] - A[1], r.a[2] - A[2])).toBeLessThan(0.015);
      expect(Math.hypot(r.b[0] - B[0], r.b[1] - B[1], r.b[2] - B[2])).toBeLessThan(0.015);
      expect(dt).toBeLessThan(2000); // SwiftShader software rendering
      // Off-content tap → null.
      expect(v.pickPoint([2, 2])).toBeNull();
      // After the view changes (displayed frame or not), picks stay correct.
      for (const render of [true, false]) {
        v.camera.position.lerp(v.controls.target, 0.25);
        v.controls.update();
        if (render) v.renderNow();
        const q = v.pickPoint(v.worldToScreen(A)!)!;
        expect(Math.hypot(q[0] - A[0], q[1] - A[1], q[2] - A[2])).toBeLessThan(0.015);
      }
    });
  }

  it('picks through gaps of a sparse point cloud (nearest splat within the window)', () => {
    const v = makeViewer();
    v.setPointCloud(planeCloud(1, 0.1, 0, [200, 200, 200]), { pointSizePx: 2 });
    v.renderNow();
    const target = [0.05, 0.05, 0]; // between 4 points
    const s = v.worldToScreen(target)!;
    const p = v.pickPoint(s)!;
    expect(p).not.toBeNull();
    expect(Math.abs(p[2])).toBeLessThan(0.01);
    expect(Math.hypot(p[0] - 0.05, p[1] - 0.05)).toBeLessThan(0.09);
  });

  it('measure mode: two taps produce a callback, markers and a formatted label', async () => {
    const v = makeViewer();
    v.setMesh(planeMesh(2, 16, [180, 180, 180]));
    v.renderNow();
    const results: number[] = [];
    v.startMeasure((d) => results.push(d));
    v.setUnits(true);
    v.measureTap(v.worldToScreen([0, 0, 0])!);
    expect(results.length).toBe(0);
    v.measureTap(v.worldToScreen([0.5, 0, 0])!);
    expect(results.length).toBe(1);
    expect(results[0]).toBeCloseTo(0.5, 2);
    v.renderNow();
    const label = v.container.querySelector('.viewer-measure-label') as HTMLElement;
    expect(label.style.display).toBe('block');
    expect(label.textContent).toMatch(/^\d+(\.\d+)? cm$/);
    v.setUnits(false);
    expect(label.textContent).toMatch(/units$/);
    const shot = await v.screenshot(); // composites the label into the PNG
    expect(shot.type).toBe('image/png');
    v.stopMeasure();
    expect(label.style.display).toBe('none');
  });

  it('setOrbitTarget / getOrbitTarget round-trip in world coordinates', () => {
    const v = makeViewer();
    v.setMesh(planeMesh(2, 4, [180, 180, 180]));
    v.setOrbitTarget([0.5, -0.25, 0]);
    const t = v.getOrbitTarget();
    expect(t[0]).toBeCloseTo(0.5, 6);
    expect(t[1]).toBeCloseTo(-0.25, 6);
    expect(t[2]).toBeCloseTo(0, 6);
  });
});

describe('Viewer lifecycle', () => {
  it('screenshot() returns a PNG of the drawing-buffer size', async () => {
    const v = makeViewer();
    v.setPointCloud(planeCloud(1, 0.05, 0, [255, 255, 0]));
    const blob = await v.screenshot();
    expect(blob.type).toBe('image/png');
    const bmp = await createImageBitmap(blob);
    expect(bmp.width).toBe(W);
    expect(bmp.height).toBe(H);
    const c = document.createElement('canvas');
    c.width = W;
    c.height = H;
    const ctx = c.getContext('2d')!;
    ctx.drawImage(bmp, 0, 0);
    const d = ctx.getImageData(W / 2, H / 2, 1, 1).data;
    expect(d[0]).toBeGreaterThan(100);
    bmp.close();
  });

  it('renders on demand only and leaves no animation frames after dispose', async () => {
    const pending = new Set<number>();
    const origRaf = window.requestAnimationFrame;
    const origCaf = window.cancelAnimationFrame;
    window.requestAnimationFrame = (cb: FrameRequestCallback) => {
      const id = origRaf.call(window, (t) => {
        pending.delete(id);
        cb(t);
      });
      pending.add(id);
      return id;
    };
    window.cancelAnimationFrame = (id: number) => {
      pending.delete(id);
      origCaf.call(window, id);
    };
    try {
      const v = new Viewer(makeContainer(), { background: '#000000' });
      v.setPointCloud(planeCloud(1, 0.05, 0, [255, 255, 255]));
      // idle: a requested frame runs once, then nothing more is scheduled
      for (let i = 0; i < 5 && v.framePending; i++) await nextFrame();
      expect(v.framePending).toBe(false);
      const f0 = v.getStats().frames;
      await nextFrame();
      await nextFrame();
      expect(v.getStats().frames).toBe(f0);
      // a change schedules exactly one frame
      v.setDisplay({ edl: false });
      expect(v.framePending).toBe(true);
      v.dispose();
      expect(v.framePending).toBe(false);
      expect(pending.size).toBe(0);
      v.requestRender();
      v.setDisplay({ edl: true });
      expect(pending.size).toBe(0);
      expect(v.container.querySelector('canvas')).toBeNull();
      expect(v.container.querySelector('.viewer-measure-label')).toBeNull();
      await nextFrame();
      expect(pending.size).toBe(0);
    } finally {
      window.requestAnimationFrame = origRaf;
      window.cancelAnimationFrame = origCaf;
    }
  });

  it('survives WebGL context loss and restoration', async () => {
    const v = makeViewer();
    v.setPointCloud(planeCloud(1, 0.02, 0, [255, 255, 255]), { pointSizePx: 3 });
    expect(countNonBackground(v.readPixels())).toBeGreaterThan(0.1 * W * H);
    const gl = v.renderer.getContext();
    const ext = gl.getExtension('WEBGL_lose_context');
    expect(ext).not.toBeNull();
    const canvas = v.renderer.domElement;
    const lost = new Promise((r) => canvas.addEventListener('webglcontextlost', r, { once: true }));
    ext!.loseContext();
    await lost;
    v.setDisplay({ edl: true }); // no throw, no frame while lost
    expect(v.framePending).toBe(false);
    const restored = new Promise((r) => canvas.addEventListener('webglcontextrestored', r, { once: true }));
    // Chrome ignores restoreContext() issued in the same task as the loss dispatch.
    await new Promise((r) => setTimeout(r, 50));
    ext!.restoreContext();
    await restored;
    await nextFrame();
    const px = v.readPixels();
    expect(countNonBackground(px)).toBeGreaterThan(0.1 * W * H);
    // picking still works after restore
    expect(v.pickPoint(v.worldToScreen([0, 0, 0])!)).not.toBeNull();
  });

  it('resizes with its container', async () => {
    const v = makeViewer();
    v.setPointCloud(planeCloud(1, 0.05, 0, [255, 255, 255]));
    v.container.style.width = '160px';
    v.container.style.height = '120px';
    for (let i = 0; i < 5; i++) await nextFrame();
    const px = v.readPixels();
    expect(px.width).toBe(160);
    expect(px.height).toBe(120);
  });
});

describe('TapDetector', () => {
  it('reports taps and double taps but not drags or pinches', () => {
    const el = makeContainer(100, 100);
    const taps: number[] = [];
    const doubles: number[] = [];
    const td = new TapDetector(el, { onTap: (t) => taps.push(t.x), onDoubleTap: (t) => doubles.push(t.x) });
    let ts = 1000;
    const ev = (type: string, x: number, y: number, id = 1) =>
      el.dispatchEvent(
        new PointerEvent(type, { clientX: x + el.getBoundingClientRect().left, clientY: y + el.getBoundingClientRect().top, pointerId: id, pointerType: 'touch', bubbles: true }),
      );
    // Note: event.timeStamp is real time; taps here are fast enough.
    ev('pointerdown', 10, 10);
    ev('pointerup', 10, 10);
    expect(taps).toEqual([10]);
    ev('pointerdown', 12, 11);
    ev('pointerup', 12, 11);
    expect(doubles.length).toBe(1);
    // drag
    ev('pointerdown', 50, 50);
    ev('pointermove', 80, 50);
    ev('pointerup', 80, 50);
    expect(taps.length).toBe(2);
    // pinch (two pointers)
    ev('pointerdown', 20, 20, 1);
    ev('pointerdown', 60, 60, 2);
    ev('pointerup', 20, 20, 1);
    ev('pointerup', 60, 60, 2);
    expect(taps.length).toBe(2);
    td.dispose();
    ev('pointerdown', 10, 10);
    ev('pointerup', 10, 10);
    expect(taps.length).toBe(2);
    ts++;
  });
});
