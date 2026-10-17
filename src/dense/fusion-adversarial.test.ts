/**
 * Adversarial tests for undistortion, view selection and depth-map fusion (review pass).
 */
import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, Pose, RGBAImage } from '../core/types';
import { cameraCenter, lookAt, makeNearScene, orbitPoses, pinholeIntrinsics, type SynthScene } from './synthetic';
import { distanceToScene, renderGtViews } from './testutil-scene';
import { fuseDepthMaps } from './fusion';
import { MemoryDepthMapStore } from './types';
import { selectSourceViews } from './view-selection';
import { undistortImage, undistortedCamera } from './undistort';

const log = (...a: unknown[]): void => console.log('[fusion-adv]', ...a);
const DEG = Math.PI / 180;

function allNeighbours(n: number): Map<number, number[]> {
  const m = new Map<number, number[]>();
  for (let i = 0; i < n; i++) m.set(i, [...Array(n).keys()].filter((j) => j !== i));
  return m;
}

describe('fuseDepthMaps (adversarial)', () => {
  it('output normals are WORLD-frame and face the cameras (rotated cameras, slanted wall)', async () => {
    // wall through (0, 6, 0) with normal (−0.5, −1, 0.3)/|·| (facing the cameras near the origin);
    // cameras are rolled / pitched so that camera ≠ world axes
    const n = [-0.5, -1, 0.3];
    const ln = Math.hypot(n[0], n[1], n[2]);
    const N = n.map((v) => v / ln);
    const u0 = [N[1], -N[0], 0];
    const lu = Math.hypot(u0[0], u0[1], u0[2]);
    const U = u0.map((v) => v / lu);
    const V = [N[1] * U[2] - N[2] * U[1], N[2] * U[0] - N[0] * U[2], N[0] * U[1] - N[1] * U[0]];
    const scene: SynthScene = { primitives: [{ kind: 'quad', center: [0, 6, 0], axisU: U, axisV: V, halfU: 30, halfV: 30, material: { scale: 0.8 } }] };
    const cam = pinholeIntrinsics(96, 72, 60 * DEG);
    const poses: Pose[] = [
      lookAt([0, 0, 0], [0.5, 6, 0.4], [0.3, 0, 1]),
      lookAt([0.4, 0.1, 0.2], [0.3, 6, -0.2], [-0.2, 0, 1]),
      lookAt([-0.4, 0, -0.1], [0, 6, 0.6], [0, 0.2, 1]),
      lookAt([0.1, -0.3, 0.4], [-0.4, 6, 0], [0.5, 0, 1]),
    ];
    const gt = renderGtViews(scene, cam, poses, { normalNoiseDeg: 1, seed: 2 });
    const pc = await fuseDepthMaps(gt.views, gt.store, allNeighbours(4), { minViews: 3 });
    let worst = 1;
    let bad = 0;
    for (let i = 0; i < pc.count; i++) {
      const d = pc.normals![3 * i] * N[0] + pc.normals![3 * i + 1] * N[1] + pc.normals![3 * i + 2] * N[2];
      worst = Math.min(worst, d);
      if (d < Math.cos(5 * DEG)) bad++;
    }
    log(`slanted wall: ${pc.count} points, min cos(normal, GT) ${worst.toFixed(4)}, ${bad} > 5°`);
    expect(pc.count).toBeGreaterThan(2000);
    expect(bad / pc.count).toBeLessThan(0.01);
  });

  it('120° FOV cameras: z-depth conventions hold at the image corners', async () => {
    const scene = makeNearScene();
    const cam = pinholeIntrinsics(120, 90, 120 * DEG);
    const poses = orbitPoses([0, 0, 0.25], 1.8, 35 * DEG, 8);
    const gt = renderGtViews(scene, cam, poses, { noise: 0.001, seed: 4 });
    const pc = await fuseDepthMaps(gt.views, gt.store, allNeighbours(8), { maxNeighbours: 4 });
    const centres = gt.views.map((v) => cameraCenter(v.pose));
    let within = 0;
    for (let i = 0; i < pc.count; i++) {
      const x = pc.positions[3 * i];
      const y = pc.positions[3 * i + 1];
      const z = pc.positions[3 * i + 2];
      let dc = Infinity;
      for (const c of centres) dc = Math.min(dc, Math.hypot(x - c[0], y - c[1], z - c[2]));
      if (distanceToScene(scene, x, y, z) < 0.005 * dc) within++;
    }
    log(`120° FOV: ${pc.count} points, ${((100 * within) / pc.count).toFixed(2)} % within 0.5 %`);
    expect(pc.count).toBeGreaterThan(2000);
    expect(within / pc.count).toBeGreaterThan(0.99);
  });

  it('NaN normals, NaN / Infinity depths in the input never produce non-finite output', async () => {
    const scene = makeNearScene();
    const cam = pinholeIntrinsics(64, 48, 60 * DEG);
    const gt = renderGtViews(scene, cam, orbitPoses([0, 0, 0.25], 3, 35 * DEG, 5));
    const store = new MemoryDepthMapStore();
    for (let v = 0; v < 5; v++) {
      const dm = (await gt.store.get(v))!;
      for (let k = 0; k < dm.depth.length; k += 7) {
        if (k % 3 === 0) dm.normal[3 * k] = NaN;
        else if (k % 3 === 1) dm.depth[k] = NaN;
        else dm.depth[k] = Infinity;
      }
      await store.put(v, dm);
    }
    for (const minViews of [1, 3]) {
      const pc = await fuseDepthMaps(gt.views, store, allNeighbours(5), { minViews });
      let bad = 0;
      for (const a of [pc.positions, pc.normals!, pc.radii!, pc.confidence!]) for (const v of a) if (!Number.isFinite(v)) bad++;
      log(`non-finite input (minViews ${minViews}): ${pc.count} points, ${bad} non-finite output values`);
      expect(pc.count).toBeGreaterThan(500);
      expect(bad).toBe(0);
    }
  });

  it('cancelling from the progress callback mid-run rejects with CancelledError', async () => {
    const scene = makeNearScene();
    const cam = pinholeIntrinsics(48, 36, 1);
    const gt = renderGtViews(scene, cam, orbitPoses([0, 0, 0.25], 3, 0.6, 6));
    const token = { cancelled: false };
    const seen: string[] = [];
    const p = fuseDepthMaps(gt.views, gt.store, allNeighbours(6), {}, (e) => {
      seen.push(e.message ?? '');
      if (e.message === 'view 3/6') token.cancelled = true;
    }, token);
    await expect(p).rejects.toMatchObject({ name: 'CancelledError' });
    expect(seen).not.toContain('view 4/6');
  });
});

describe('selectSourceViews (adversarial)', () => {
  it('few visible points that project outside the pinhole frame are not discarded by the frustum fallback', () => {
    // the reference observes 6 points (SfM observations near the border of a distorted image)
    // whose pinhole projection falls just outside the frame; no other point is in its frustum
    const cam = pinholeIntrinsics(320, 240, 60 * DEG);
    const ref = { id: 0, pose: lookAt([0, 0, 0], [0, 1, 0]), cam };
    const others = [1, 2].map((id) => ({ id, pose: lookAt([id * 0.8, 0, 0], [0, 10, 0]), cam }));
    const pts: number[][] = [];
    const vis: number[][] = [];
    const halfW = Math.tan(30 * DEG); // normalised half width
    for (let k = 0; k < 6; k++) {
      const z = 10 + k * 0.4;
      pts.push([-(halfW + 0.02) * z, z, (k - 2.5) * 0.3]); // just left of the frame
      vis.push([0, 1, 2]);
    }
    const sel = selectSourceViews([ref, ...others], { xyz: Float64Array.from(pts.flat()), visibility: vis });
    const s = sel.get(0)!;
    log(`outside-frame observations: depth [${s.depthMin.toFixed(2)}, ${s.depthMax.toFixed(2)}], sources ${JSON.stringify(s.sources)}`);
    expect(s.depthMin).toBeGreaterThan(0.7 * 10);
    expect(s.depthMin).toBeLessThanOrEqual(0.8 * 10.1);
    expect(s.depthMax).toBeGreaterThanOrEqual(1.25 * 11.9);
    expect(s.depthMax).toBeLessThan(2 * 12);
  });

  it('numSources is a hard cap even when the geometric fallback runs (minSources > numSources)', () => {
    const cam = pinholeIntrinsics(320, 240, 60 * DEG);
    const views = [0, 1, 2, 3].map((x) => ({ id: x, pose: lookAt([x, 0, 0], [x, 5, 0]), cam }));
    const pts: number[][] = [];
    for (let i = 0; i < 40; i++) pts.push([-1 + (i % 8) * 0.6, 5 + (i % 3) * 0.2, -1 + Math.floor(i / 8) * 0.5]);
    const sel = selectSourceViews(views, { xyz: Float64Array.from(pts.flat()), visibility: pts.map(() => []) }, { numSources: 1, minSources: 3 });
    for (const s of sel.values()) expect(s.sources.length).toBeLessThanOrEqual(1);
  });
});

describe('undistortImage (adversarial)', () => {
  /** Smooth test image: a Gaussian dot at the distorted projection of a normalised direction. */
  function dotImage(cam: CameraIntrinsics, xn: number, yn: number, sigma: number): RGBAImage {
    const r2 = xn * xn + yn * yn;
    const dd = 1 + cam.k1 * r2 + cam.k2 * r2 * r2;
    const ud = cam.f * dd * xn + cam.cx;
    const vd = cam.f * dd * yn + cam.cy;
    const data = new Uint8ClampedArray(cam.width * cam.height * 4);
    for (let y = 0; y < cam.height; y++)
      for (let x = 0; x < cam.width; x++) {
        const g = 230 * Math.exp(-((x - ud) ** 2 + (y - vd) ** 2) / (2 * sigma * sigma));
        data.set([g + 10, g + 10, g + 10, 255], 4 * (y * cam.width + x));
      }
    return { width: cam.width, height: cam.height, data };
  }
  function centroid(img: RGBAImage, ex: number, ey: number, R: number): [number, number] {
    let sw = 0;
    let sx = 0;
    let sy = 0;
    for (let y = Math.floor(ey) - R; y <= Math.ceil(ey) + R; y++)
      for (let x = Math.floor(ex) - R; x <= Math.ceil(ex) + R; x++) {
        if (x < 0 || y < 0 || x >= img.width || y >= img.height) continue;
        const w = Math.max(0, img.data[4 * (y * img.width + x)] - 10);
        sw += w;
        sx += w * x;
        sy += w * y;
      }
    return [sx / sw, sy / sw];
  }

  it('image smaller than the calibrated size (stored at ½ resolution): intrinsics rescaled per §2', () => {
    const full: CameraIntrinsics = { width: 400, height: 300, f: 330, cx: 203.7, cy: 146.2, k1: -0.2, k2: 0.05 };
    const half: CameraIntrinsics = { width: 200, height: 150, f: 165, cx: (203.7 + 0.5) / 2 - 0.5, cy: (146.2 + 0.5) / 2 - 0.5, k1: -0.2, k2: 0.05 };
    const xn = 0.4;
    const yn = 0.27;
    const src = dotImage(half, xn, yn, 1.4);
    const { image, cam } = undistortImage(src, full, 1000);
    expect(image.width).toBe(200);
    expect(cam.f).toBeCloseTo(165, 10);
    expect(cam.cx).toBeCloseTo(half.cx, 10);
    const ex = cam.f * xn + cam.cx;
    const ey = cam.f * yn + cam.cy;
    const [gx, gy] = centroid(image, ex, ey, 6);
    log(`½-res source: dot centroid error ${Math.hypot(gx - ex, gy - ey).toFixed(4)} px`);
    expect(Math.hypot(gx - ex, gy - ey)).toBeLessThan(0.05);
  });

  it('no distortion + anisotropic rounding (sx ≠ sy): output matches its own pinhole camera', () => {
    const cam: CameraIntrinsics = { width: 400, height: 300, f: 330, cx: 201.3, cy: 148.9, k1: 0, k2: 0 };
    const xn = -0.35;
    const yn = 0.22;
    const src = dotImage(cam, xn, yn, 2.5);
    for (const antialias of [true, false]) {
      const { image, cam: oc } = undistortImage(src, cam, 133, { antialias });
      expect(oc).toEqual(undistortedCamera(cam, 133));
      expect(image.width / 400).not.toBeCloseTo(image.height / 300, 6);
      const ex = oc.f * xn + oc.cx;
      const ey = oc.f * yn + oc.cy;
      const [gx, gy] = centroid(image, ex, ey, 5);
      log(`anisotropic ${image.width}×${image.height} (antialias ${antialias}): centroid error ${Math.hypot(gx - ex, gy - ey).toFixed(4)} px`);
      expect(Math.hypot(gx - ex, gy - ey)).toBeLessThan(antialias ? 0.05 : 0.12);
    }
  });
});
