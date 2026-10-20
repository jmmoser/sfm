/**
 * Viewer splat layer helpers (Node): the pinhole SplatCamera of a three.js camera projects world
 * points exactly where three does (pose, world root, fov, aspect, zoom, view offsets), and the
 * canvas / interactive frame sizes respect the pixel budgets.
 */
import { Group, Matrix4, PerspectiveCamera, Quaternion, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { interactiveFrameSize, splatCameraFromView, splatCanvasSize, subsampleMeans } from './splat-camera';

function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('splatCameraFromView', () => {
  it('projects world points where the three.js camera does (incl. the Z-up world root)', () => {
    const r = rng(3);
    let worst = 0;
    for (let trial = 0; trial < 40; trial++) {
      const W = 100 + Math.floor(900 * r());
      const H = 100 + Math.floor(900 * r());
      const cam = new PerspectiveCamera(20 + 70 * r(), W / H, 0.05, 100);
      cam.zoom = trial % 3 === 0 ? 0.6 + r() : 1;
      if (trial % 4 === 1) cam.setViewOffset(W + 37, H + 21, 11 + 20 * r(), 7 + 10 * r(), W, H);
      cam.position.set(4 * r() - 2, 4 * r() - 2, 4 * r() - 2);
      cam.lookAt(r() - 0.5, r() - 0.5, r() - 0.5);
      cam.updateProjectionMatrix();
      cam.updateMatrixWorld();
      // world root: Z-up → Y-up ((x, y, z) → (x, z, −y)), or a random rotation
      const root = new Group();
      if (trial % 2 === 0) root.quaternion.setFromAxisAngle(new Vector3(1, 0, 0), -Math.PI / 2);
      else root.quaternion.copy(new Quaternion(r() - 0.5, r() - 0.5, r() - 0.5, r() - 0.5).normalize());
      root.updateMatrixWorld(true);
      const view = new Matrix4().multiplyMatrices(cam.matrixWorldInverse, root.matrixWorld);
      const sc = splatCameraFromView(view.elements, cam.projectionMatrix.elements, W, H, 0.01);
      for (let k = 0; k < 20; k++) {
        const X = [3 * r() - 1.5, 3 * r() - 1.5, 3 * r() - 1.5];
        // pinhole
        const R = sc.R;
        const xc = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + sc.t[0];
        const yc = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + sc.t[1];
        const zc = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + sc.t[2];
        if (zc < 0.1) continue;
        const u = (sc.fx * xc) / zc + sc.cx;
        const v = (sc.fy * yc) / zc + sc.cy;
        // three: pixel centres at integers, so CSS-style coordinates minus 0.5
        const p = new Vector3(X[0], X[1], X[2]).applyMatrix4(root.matrixWorld).project(cam);
        const u3 = ((p.x + 1) / 2) * W - 0.5;
        const v3 = ((1 - p.y) / 2) * H - 0.5;
        worst = Math.max(worst, Math.hypot(u - u3, v - v3));
      }
      // a proper rotation
      const R = sc.R;
      const det = R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) + R[2] * (R[3] * R[7] - R[4] * R[6]);
      expect(det).toBeCloseTo(1, 9);
    }
    expect(worst).toBeLessThan(1e-6 * 1000);
  });

  it('looks down +z with +y down (OpenCV) and has square pixels at the viewport aspect', () => {
    const cam = new PerspectiveCamera(60, 2, 0.1, 10);
    cam.position.set(0, 0, 5);
    cam.lookAt(0, 0, 0);
    cam.updateMatrixWorld();
    const sc = splatCameraFromView(cam.matrixWorldInverse.elements, cam.projectionMatrix.elements, 800, 400, 0.1);
    expect(sc.fx).toBeCloseTo(sc.fy, 9);
    expect(sc.fy).toBeCloseTo(200 / Math.tan(Math.PI / 6), 9);
    expect(sc.cx).toBeCloseTo(399.5, 9);
    expect(sc.cy).toBeCloseTo(199.5, 9);
    // at (0, 0, 5) looking down world −z: OpenCV +z forward = world −z, +y down = world −y
    expect(Array.from(sc.R, (x) => x + 0)).toEqual([1, 0, 0, 0, -1, 0, 0, 0, -1].map((x) => expect.closeTo(x, 12)));
    expect(sc.t[2]).toBeCloseTo(5, 12);
  });
});

describe('splat layer sizes', () => {
  it('caps the canvas by devicePixelRatio, maxPixelRatio and the pixel budget', () => {
    // iPhone 15 portrait (393 × 852 CSS, DPR 3): capped at 2 like the WebGL canvas
    expect(splatCanvasSize(393, 852, 3, 2, 2e6)).toEqual({ width: 786, height: 1704, pixelRatio: 2 });
    // large desktop window: the budget lowers the ratio below 1
    const s = splatCanvasSize(1920, 1200, 2, 2, 2e6);
    expect(s.width * s.height).toBeLessThanOrEqual(2e6);
    expect(s.width * s.height).toBeGreaterThan(1.97e6);
    expect(s.width / s.height).toBeCloseTo(1.6, 2);
    expect(splatCanvasSize(0, 100, 2, 2, 2e6).width).toBe(0);
    for (const [w, h, dpr] of [
      [393, 852, 3],
      [1024, 1366, 2],
      [777, 333, 1.75],
    ]) {
      const z = splatCanvasSize(w, h, dpr, 4, 1e6);
      expect(z.width * z.height).toBeLessThanOrEqual(1e6);
    }
  });

  it('scales a moving frame to the interactive budget at the same aspect', () => {
    expect(interactiveFrameSize(786, 1704, 2e6)).toEqual({ width: 786, height: 1704 });
    const f = interactiveFrameSize(786, 1704, 5e5);
    expect(f.width * f.height).toBeLessThanOrEqual(5.02e5);
    expect(f.width / f.height).toBeCloseTo(786 / 1704, 2);
  });

  it('subsamples the means evenly', () => {
    const m = new Float32Array(3 * 10).map((_, i) => i);
    expect(subsampleMeans(m, 10, 20)).toEqual(m);
    const s = subsampleMeans(m, 10, 5);
    expect([...s]).toEqual([0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26]);
  });
});
