/**
 * Geometric accuracy of the synthetic renderer (WebGL2 / SwiftShader in headless
 * Chromium): small bright spheres at known 3-D positions must appear where the RADIAL
 * camera model puts them (pose conversion, off-axis projection with the principal
 * point, distortion resampling), and GT depth must match.
 */
import { Mesh, MeshBasicMaterial, Scene, SphereGeometry } from 'three';
import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, Pose } from '../../core/types';
import { projectRadial } from './conventions';
import { expSO3, m3Mul, m3T, m3Vec } from './mat';
import { SynthRenderer } from './render';
import type { SynthScene } from './scene-common';

const cam: CameraIntrinsics = { width: 320, height: 240, f: 260, cx: 163.7, cy: 117.2, k1: -0.06, k2: 0.012 };

function srgbToLinear(c: number): number {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

function makeScene(pose: Pose) {
  const scene = new Scene();
  const C = m3Vec(m3T(pose.R), pose.t).map((v) => -v);
  const markers: { X: number[]; r: number; zc: number; dist: number }[] = [];
  const geo = new SphereGeometry(1, 48, 32);
  const mat = new MeshBasicMaterial({ color: 0xffffff });
  for (let j = 0; j < 4; j++)
    for (let i = 0; i < 5; i++) {
      // directions covering the image including the corners (where distortion is largest)
      const u = 12 + (i * (cam.width - 24)) / 4;
      const v = 12 + (j * (cam.height - 24)) / 3;
      const z = 2 + 0.5 * ((i + j) % 4);
      const xn = (u - cam.cx) / cam.f,
        yn = (v - cam.cy) / cam.f; // (distorted ≈ undistorted for placement)
      const xc = [xn * z, yn * z, z];
      const X = Array.from(m3Vec(m3T(pose.R), xc)).map((v2, k) => v2 + C[k]);
      const r = 0.011 * z;
      const m = new Mesh(geo, mat);
      m.position.set(X[0], X[1], X[2]);
      m.scale.setScalar(r);
      scene.add(m);
      m.updateMatrixWorld(true);
      markers.push({ X, r, zc: z, dist: Math.hypot(...xc) });
    }
  const synth: SynthScene = {
    name: 'markers',
    scene,
    surfaceMeshes: [],
    focusMeshes: [],
    depthHidden: [],
    near: 0.05,
    far: 50,
    roi: { min: [-10, -10, -10], max: [10, 10, 10] },
    focus: { min: [-1, -1, -1], max: [1, 1, 1] },
    dispose: () => {
      geo.dispose();
      mat.dispose();
    },
  };
  return { synth, markers };
}

function centroid(rgba: Uint8Array, u0: number, v0: number, rad: number): [number, number, number] {
  let sw = 0,
    su = 0,
    sv = 0;
  for (let v = Math.max(0, Math.round(v0) - rad); v <= Math.min(cam.height - 1, Math.round(v0) + rad); v++)
    for (let u = Math.max(0, Math.round(u0) - rad); u <= Math.min(cam.width - 1, Math.round(u0) + rad); u++) {
      const w = srgbToLinear(rgba[4 * (v * cam.width + u)]);
      sw += w;
      su += w * u;
      sv += w * v;
    }
  return [su / sw, sv / sw, sw];
}

describe('SynthRenderer geometry', () => {
  const pose: Pose = (() => {
    const R = expSO3([0.35, -0.6, 0.25]);
    return { R, t: Float64Array.of(0.3, -0.2, 1.1) };
  })();

  it('markers land at their RADIAL projections (< 0.12 px) and GT depth matches', async () => {
    const { synth, markers } = makeScene(pose);
    const r = new SynthRenderer(synth, cam, { noise: false, vignetting: 0, supersample: 3, motionBlurSamples: 1 });
    try {
      r.renderFrame([pose], 1, 0);
      const px = r.readPixels();
      const out = new Float64Array(3);
      let worst = 0;
      for (const m of markers) {
        expect(projectRadial(cam, pose, m.X, out)).toBe(true);
        const [cu, cv, mass] = centroid(px, out[0], out[1], 9);
        expect(mass).toBeGreaterThan(5);
        worst = Math.max(worst, Math.hypot(cu - out[0], cv - out[1]));
      }
      console.log(`[render.gpu] worst marker centroid error ${worst.toFixed(4)} px`);
      expect(worst).toBeLessThan(0.12);

      const depth = r.renderDepth(pose);
      expect(depth.length).toBe(cam.width * cam.height);
      for (const m of markers) {
        projectRadial(cam, pose, m.X, out);
        const d = depth[Math.round(out[1]) * cam.width + Math.round(out[0])];
        // front surface along the ray through the centre: z = zc − r·zc/|C|
        expect(Math.abs(d - (m.zc - (m.r * m.zc) / m.dist))).toBeLessThan(0.004 * m.zc);
      }
      expect(depth[0]).toBe(0); // background
      const jpeg = await r.toJpeg(0.92);
      expect(jpeg.type).toBe('image/jpeg');
      expect(jpeg.size).toBeGreaterThan(1000);

      const wd = r.renderWindowDepth(pose, 0.5);
      expect(wd.width).toBeGreaterThan(cam.width * 0.5);
      let nonzero = 0;
      for (const v of wd.data) if (v > 0) nonzero++;
      expect(nonzero).toBeGreaterThan(20);
    } finally {
      r.dispose();
      synth.dispose();
    }
  });

  it('motion blur averages sub-frames; noise and gain change pixel values deterministically', () => {
    const { synth, markers } = makeScene(pose);
    const r = new SynthRenderer(synth, cam, { noise: true, vignetting: 0.2, supersample: 2, motionBlurSamples: 3 });
    try {
      // sub-frame poses: small rotations about the camera y axis, R_k = exp(k·d·ŷ) R, t_k = exp(k·d·ŷ) t
      const d = 0.004;
      const poses: Pose[] = [-1, 0, 1].map((k) => {
        const Rk = expSO3([0, k * d, 0]);
        return { R: m3Mul(Rk, pose.R), t: m3Vec(Rk, pose.t) };
      });
      r.renderFrame(poses, 1, 7);
      const a = r.readPixels();
      r.renderFrame(poses, 1, 7);
      const b = r.readPixels();
      expect(Array.from(a.subarray(0, 4000))).toEqual(Array.from(b.subarray(0, 4000))); // deterministic noise
      r.renderFrame(poses, 1, 8);
      const c = r.readPixels();
      let diff = 0;
      for (let i = 0; i < a.length; i += 4) diff += a[i] !== c[i] ? 1 : 0;
      expect(diff).toBeGreaterThan(100); // different seed → different noise
      // the blurred marker centroid stays at the projection of the middle pose (symmetric blur)
      const out = new Float64Array(3);
      const m = markers[7];
      projectRadial(cam, pose, m.X, out);
      const [cu, cv] = centroid(a, out[0], out[1], 12);
      expect(Math.hypot(cu - out[0], cv - out[1])).toBeLessThan(0.6);
    } finally {
      r.dispose();
      synth.dispose();
    }
  });
});
