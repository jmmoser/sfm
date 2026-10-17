/**
 * Rolling-shutter band rendering and focus breathing in the synthetic renderer (WebGL2 /
 * SwiftShader): band-wise rendering with one pose equals the global-shutter render, markers
 * seen by a fast-rotating camera land at the projection of THEIR band's pose (not the
 * mid-readout pose), and setFocal() moves markers to the projections of the new focal.
 */
import { Mesh, MeshBasicMaterial, Scene, SphereGeometry } from 'three';
import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, Pose } from '../../core/types';
import { projectRadial } from './conventions';
import { expSO3, m3Mul, m3T, m3Vec } from './mat';
import { readoutBands, readoutGeometry, type ReadoutBand } from './realism';
import { SynthRenderer } from './render';
import type { SynthScene } from './scene-common';

// portrait θ = 0 → the readout sweeps columns right → left
const cam: CameraIntrinsics = { width: 240, height: 320, f: 260, cx: 121.3, cy: 158.6, k1: -0.05, k2: 0.01 };

function srgbToLinear(c: number): number {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

/** Markers (u, v, z-depth) of the portrait camera: centres of 15-px column bands (16 bands over 240 columns). */
const PORTRAIT_MARKERS: [number, number, number][] = [];
for (let j = 0; j < 5; j++) for (let i = 0; i < 4; i++) PORTRAIT_MARKERS.push([37 + 60 * i, 25 + (j * (cam.height - 50)) / 4, 2 + 0.4 * ((i + 2 * j) % 3)]);

function makeScene(pose: Pose, camera: CameraIntrinsics = cam, pixels: readonly (readonly [number, number, number])[] = PORTRAIT_MARKERS) {
  const scene = new Scene();
  const C = m3Vec(m3T(pose.R), pose.t).map((v) => -v);
  const markers: { X: number[] }[] = [];
  const geo = new SphereGeometry(1, 32, 24);
  const mat = new MeshBasicMaterial({ color: 0xffffff });
  for (let k = 0; k < pixels.length; k++) {
    const [u, v, z] = pixels[k];
    const xc = [((u - camera.cx) / camera.f) * z, ((v - camera.cy) / camera.f) * z, z];
    const X = Array.from(m3Vec(m3T(pose.R), xc)).map((v2, d) => v2 + C[d]);
    const m = new Mesh(geo, mat);
    m.position.set(X[0], X[1], X[2]);
    m.scale.setScalar(0.006 * z); // ≈ 1.6 px radius: much smaller than a band
    scene.add(m);
    m.updateMatrixWorld(true);
    markers.push({ X });
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

function centroid(rgba: Uint8Array, u0: number, v0: number, rad: number, camera: CameraIntrinsics = cam): [number, number, number] {
  let sw = 0,
    su = 0,
    sv = 0;
  for (let v = Math.max(0, Math.round(v0) - rad); v <= Math.min(camera.height - 1, Math.round(v0) + rad); v++)
    for (let u = Math.max(0, Math.round(u0) - rad); u <= Math.min(camera.width - 1, Math.round(u0) + rad); u++) {
      const w = srgbToLinear(rgba[4 * (v * camera.width + u)]);
      sw += w;
      su += w * u;
      sv += w * v;
    }
  return [su / sw, sv / sw, sw];
}

const pose0: Pose = (() => {
  const R = expSO3([0.3, -0.5, 0.2]);
  return { R, t: Float64Array.of(0.2, -0.1, 0.9) };
})();

/** Camera rotating about its own centre at ω (rad/s, camera axes): pose at time tMs. */
function rotatingPose(omega: [number, number, number], tMs: number): Pose {
  const s = tMs / 1000;
  const Rk = expSO3([omega[0] * s, omega[1] * s, omega[2] * s]);
  return { R: m3Mul(Rk, pose0.R), t: m3Vec(Rk, pose0.t) };
}

/** Pixel of X in the RS image: fixed point band ↔ band pose. */
function rsProjection(bands: ReadoutBand[], poseAt: (dtMs: number) => Pose, X: number[], camera: CameraIntrinsics = cam): { uv: [number, number]; band: number } | null {
  const out = new Float64Array(3);
  const find = (u: number, v: number) => bands.findIndex((b) => u >= b.rect.u0 - 0.5 && u < b.rect.u1 + 0.5 && v >= b.rect.v0 - 0.5 && v < b.rect.v1 + 0.5);
  if (!projectRadial(camera, poseAt(0), X, out)) return null;
  let k = find(out[0], out[1]);
  for (let it = 0; it < 20 && k >= 0; it++) {
    projectRadial(camera, poseAt(bands[k].dtMs), X, out);
    const k2 = find(out[0], out[1]);
    if (k2 === k) return { uv: [out[0], out[1]], band: k };
    k = k2;
  }
  return null;
}

describe('SynthRenderer rolling shutter bands + focus breathing', () => {
  it('bands with a single pose reproduce the global-shutter render', () => {
    const { synth } = makeScene(pose0);
    const r = new SynthRenderer(synth, cam, { noise: false, vignetting: 0.1, supersample: 2, motionBlurSamples: 3 });
    try {
      r.renderFrame([pose0], 1, 0);
      const gs = r.readPixels();
      const g = readoutGeometry(0, cam.width, cam.height);
      const bands = readoutBands(g, cam.width, cam.height, 32, 22);
      const n = r.renderFrameBands(bands.map((b) => ({ rect: b.rect, poses: [pose0] })), 1, 0);
      expect(n).toBe(32);
      const rs = r.readPixels();
      let diff = 0,
        maxd = 0;
      for (let i = 0; i < gs.length; i += 4) {
        const d = Math.abs(gs[i] - rs[i]);
        if (d > 0) diff++;
        maxd = Math.max(maxd, d);
      }
      console.log(`[realism.gpu] band vs GS: ${diff} differing pixels, max ${maxd}`);
      expect(diff / (cam.width * cam.height)).toBeLessThan(0.002);
      // three identical sub-frames per band (accumulation path) also reproduce it
      r.renderFrameBands(bands.map((b) => ({ rect: b.rect, poses: [pose0, pose0, pose0] })), 1, 0);
      const acc = r.readPixels();
      let diff2 = 0;
      for (let i = 0; i < gs.length; i += 4) if (Math.abs(gs[i] - acc[i]) > 1) diff2++;
      expect(diff2 / (cam.width * cam.height)).toBeLessThan(0.002);
    } finally {
      r.dispose();
      synth.dispose();
    }
  });

  it('fast rotation: markers land at their band pose projection, not at the mid-readout one', () => {
    const { synth, markers } = makeScene(pose0);
    const r = new SynthRenderer(synth, cam, { noise: false, vignetting: 0, supersample: 3, motionBlurSamples: 1 });
    try {
      const g = readoutGeometry(0, cam.width, cam.height);
      const bands = readoutBands(g, cam.width, cam.height, 16, 22); // 15 px bands
      const omega: [number, number, number] = [0.6, 1.3, 0.2];
      const poseAt = (dt: number) => rotatingPose(omega, dt);
      r.renderFrameBands(bands.map((b) => ({ rect: b.rect, poses: [poseAt(b.dtMs)] })), 1, 0);
      const px = r.readPixels();
      const out = new Float64Array(3);
      let worstRS = 0,
        minGS = Infinity,
        n = 0;
      for (const m of markers) {
        const p = rsProjection(bands, poseAt, m.X);
        if (!p) continue;
        // skip markers straddling a band boundary (their image is split between two poses)
        const b = bands[p.band].rect;
        if (p.uv[0] - b.u0 < 1.8 || b.u1 - p.uv[0] < 1.8) continue;
        const [cu, cv, mass] = centroid(px, p.uv[0], p.uv[1], 6);
        expect(mass).toBeGreaterThan(1);
        worstRS = Math.max(worstRS, Math.hypot(cu - p.uv[0], cv - p.uv[1]));
        projectRadial(cam, poseAt(0), m.X, out);
        minGS = Math.min(minGS, Math.abs(bands[p.band].dtMs) > 3 ? Math.hypot(cu - out[0], cv - out[1]) : Infinity);
        n++;
      }
      console.log(`[realism.gpu] RS markers: ${n} checked, worst error vs band projection ${worstRS.toFixed(3)} px, min offset from the GS projection ${minGS.toFixed(2)} px`);
      expect(n).toBeGreaterThan(8);
      expect(worstRS).toBeLessThan(0.15);
      expect(minGS).toBeGreaterThan(0.5);
    } finally {
      r.dispose();
      synth.dispose();
    }
  });

  it('landscape θ = 90: row bands (top row first) reproduce GS, and markers land at their band pose projection', () => {
    // exercises the row-band paths (output scissor y-flip, sub-frustum rows, accumulation scissor)
    const camL: CameraIntrinsics = { width: 320, height: 240, f: 260, cx: 160.7, cy: 118.4, k1: -0.05, k2: 0.01 };
    const g = readoutGeometry(90, camL.width, camL.height);
    expect(g).toEqual({ axis: 'rows', sign: 1, lines: 240 });
    // markers at the centres of 15-px row bands (16 bands over 240 rows)
    const pix: [number, number, number][] = [];
    for (let j = 0; j < 5; j++) for (let i = 0; i < 4; i++) pix.push([30 + (i * (camL.width - 60)) / 3, 37 + 45 * j, 2 + 0.4 * ((i + 2 * j) % 3)]);
    const { synth, markers } = makeScene(pose0, camL, pix);
    const r = new SynthRenderer(synth, camL, { noise: false, vignetting: 0, supersample: 3, motionBlurSamples: 3 });
    try {
      const bands = readoutBands(g, camL.width, camL.height, 16, 22);
      expect(bands[0].rect).toEqual({ u0: 0, u1: camL.width - 1, v0: 0, v1: 14 }); // the top row band is read first
      expect(bands[0].dtMs).toBeCloseTo((7 / 239 - 0.5) * 22, 10); // mean line 7 of 0…239: −10.36 ms
      // one pose per band, and three identical sub-frames per band, = the global-shutter render
      r.renderFrame([pose0], 1, 0);
      const gs = r.readPixels();
      r.renderFrameBands(bands.map((b) => ({ rect: b.rect, poses: [pose0] })), 1, 0);
      const one = r.readPixels();
      r.renderFrameBands(bands.map((b) => ({ rect: b.rect, poses: [pose0, pose0, pose0] })), 1, 0);
      const three = r.readPixels();
      let d1 = 0,
        d3 = 0;
      for (let i = 0; i < gs.length; i += 4) {
        if (gs[i] !== one[i]) d1++;
        if (Math.abs(gs[i] - three[i]) > 1) d3++;
      }
      expect(d1 / (camL.width * camL.height)).toBeLessThan(0.002);
      expect(d3 / (camL.width * camL.height)).toBeLessThan(0.002);
      // fast rotation (yaw-dominated: image motion across the rows → skew)
      const poseAt = (dt: number) => rotatingPose([0.5, 1.3, 0.2], dt);
      r.renderFrameBands(bands.map((b) => ({ rect: b.rect, poses: [poseAt(b.dtMs)] })), 1, 0);
      const px = r.readPixels();
      const out = new Float64Array(3);
      let worstRS = 0,
        minGS = Infinity,
        n = 0;
      for (const m of markers) {
        const p = rsProjection(bands, poseAt, m.X, camL);
        if (!p) continue;
        const b = bands[p.band].rect;
        if (p.uv[1] - b.v0 < 1.8 || b.v1 - p.uv[1] < 1.8) continue; // straddles two bands
        const [cu, cv, mass] = centroid(px, p.uv[0], p.uv[1], 6, camL);
        expect(mass).toBeGreaterThan(1);
        worstRS = Math.max(worstRS, Math.hypot(cu - p.uv[0], cv - p.uv[1]));
        projectRadial(camL, poseAt(0), m.X, out);
        if (Math.abs(bands[p.band].dtMs) > 3) minGS = Math.min(minGS, Math.hypot(cu - out[0], cv - out[1]));
        n++;
      }
      console.log(`[realism.gpu] θ=90 RS markers: ${n} checked, worst error vs band projection ${worstRS.toFixed(3)} px, min offset from the GS projection ${minGS.toFixed(2)} px`);
      expect(n).toBeGreaterThan(8);
      expect(worstRS).toBeLessThan(0.15);
      expect(minGS).toBeGreaterThan(0.5);
    } finally {
      r.dispose();
      synth.dispose();
    }
  });

  it('setFocal: markers follow the per-frame focal (focus breathing)', () => {
    const { synth, markers } = makeScene(pose0);
    const f1 = cam.f * 1.006;
    const r = new SynthRenderer(synth, cam, { noise: false, vignetting: 0, supersample: 3, motionBlurSamples: 1 }, { minFocal: cam.f * 0.994 });
    try {
      r.setFocal(f1);
      r.renderFrame([pose0], 1, 0);
      const px = r.readPixels();
      const cam1 = { ...cam, f: f1 };
      const out = new Float64Array(3);
      let worst = 0;
      for (const m of markers) {
        projectRadial(cam1, pose0, m.X, out);
        const [cu, cv] = centroid(px, out[0], out[1], 6);
        worst = Math.max(worst, Math.hypot(cu - out[0], cv - out[1]));
      }
      expect(worst).toBeLessThan(0.12);
      // the depth map follows too
      const depth = r.renderDepth(pose0);
      projectRadial(cam1, pose0, markers[0].X, out);
      expect(depth[Math.round(out[1]) * cam.width + Math.round(out[0])]).toBeGreaterThan(0);
      r.setFocal(cam.f * 0.994);
      r.setWhiteBalance([1.03, 1, 0.97]);
      r.renderFrame([pose0], 1, 0);
      const px2 = r.readPixels();
      projectRadial({ ...cam, f: cam.f * 0.994 }, pose0, markers[5].X, out);
      const c2 = centroid(px2, out[0], out[1], 6);
      expect(Math.hypot(c2[0] - out[0], c2[1] - out[1])).toBeLessThan(0.12);
      // white balance: red > blue on a white marker
      const i = 4 * (Math.round(out[1]) * cam.width + Math.round(out[0]));
      expect(px2[i]).toBeGreaterThanOrEqual(px2[i + 2]);
    } finally {
      r.dispose();
      synth.dispose();
    }
  });
});
