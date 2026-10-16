/**
 * Adversarial review of the rendered datasets on the REAL scenes (headless Chromium /
 * SwiftShader WebGL2):
 *  1. absolute geometry at the dataset's default full resolution and intrinsics
 *     (principal point off centre, k1/k2, supersample 2): marker centroids vs RADIAL;
 *  2. multi-view consistency: frame A warped into frame B through GT depth A, GT poses
 *     and the RADIAL model must align with frame B to a small fraction of a pixel
 *     (global Lucas–Kanade translation estimate), with a +1 px control;
 *  3. GT depth maps vs independently sampled GT surface points;
 *  4. texture richness of the yard lawn (the largest near-range surface).
 */
import { Mesh, MeshBasicMaterial, Scene, SphereGeometry } from 'three';
import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, Pose } from '../../core/types';
import { intrinsicsFor, resolveConfig, trajectoryFor, type SynthSceneName } from './config';
import { projectRadial, undistortNormalized } from './conventions';
import { buildScene } from './generate';
import { m3T, m3Vec } from './mat';
import { SynthRenderer } from './render';
import { SynthRng } from './rng';
import { sampleSurfaces } from './sampling';
import type { SynthScene } from './scene-common';
import { groundHeight } from './scene-yard';
import { trajectoryPose } from './trajectory';

const NO_FX = { noise: false, vignetting: 0, supersample: 2, motionBlurSamples: 1 };

function srgbToLinear(c: number): number {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

/** Luminance (0..255 sRGB-coded) of an RGBA8 image. */
function luma(rgba: Uint8Array, W: number, H: number): Float32Array {
  const out = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) out[i] = 0.2126 * rgba[4 * i] + 0.7152 * rgba[4 * i + 1] + 0.0722 * rgba[4 * i + 2];
  return out;
}

function bilinear(img: Float32Array, W: number, x: number, y: number): number {
  const x0 = Math.floor(x),
    y0 = Math.floor(y);
  const fx = x - x0,
    fy = y - y0;
  const i = y0 * W + x0;
  return (img[i] * (1 - fx) + img[i + 1] * fx) * (1 - fy) + (img[i + W] * (1 - fx) + img[i + W + 1] * fx) * fy;
}

function cameraCentre(p: Pose): number[] {
  return Array.from(m3Vec(m3T(p.R), p.t)).map((v) => -v);
}

describe('absolute geometry at the dataset defaults', () => {
  for (const scene of ['object', 'yard'] as SynthSceneName[]) {
    it(`${scene}: markers at the trajectory pose land on their RADIAL projections (full res, ss = 2)`, () => {
      const cfg = resolveConfig({ scene });
      const cam = intrinsicsFor(cfg);
      const pose = trajectoryPose(trajectoryFor(cfg), 7.3);
      const C = cameraCentre(pose);
      const scene3 = new Scene();
      const geo = new SphereGeometry(1, 48, 32);
      const mat = new MeshBasicMaterial({ color: 0xffffff });
      const markers: number[][] = [];
      const RT = m3T(pose.R);
      const tmp = new Float64Array(2);
      for (let j = 0; j < 5; j++)
        for (let i = 0; i < 6; i++) {
          // target pixel (incl. the extreme corners), back-projected exactly through RADIAL
          const u = 10 + (i * (cam.width - 21)) / 5,
            v = 10 + (j * (cam.height - 21)) / 4;
          undistortNormalized(cam.k1, cam.k2, (u - cam.cx) / cam.f, (v - cam.cy) / cam.f, tmp);
          const z = 1.5 + 0.7 * ((i + 2 * j) % 3);
          const X = Array.from(m3Vec(RT, [tmp[0] * z, tmp[1] * z, z])).map((q, k) => q + C[k]);
          const m = new Mesh(geo, mat);
          m.position.set(X[0], X[1], X[2]);
          m.scale.setScalar(0.004 * z);
          m.updateMatrixWorld(true);
          scene3.add(m);
          markers.push(X);
        }
      const synth: SynthScene = { name: 'm', scene: scene3, surfaceMeshes: [], focusMeshes: [], depthHidden: [], near: 0.05, far: 50, roi: { min: [0, 0, 0], max: [1, 1, 1] }, focus: { min: [0, 0, 0], max: [1, 1, 1] }, dispose: () => (geo.dispose(), mat.dispose()) };
      const r = new SynthRenderer(synth, cam, { ...NO_FX, supersample: cfg.effects.supersample });
      try {
        r.renderFrame([pose], 1, 0);
        const px = r.readPixels();
        const out = new Float64Array(3);
        let worst = 0,
          sum = 0;
        for (const X of markers) {
          projectRadial(cam, pose, X, out);
          let sw = 0,
            su = 0,
            sv = 0;
          for (let v = Math.max(0, Math.round(out[1]) - 8); v <= Math.min(cam.height - 1, Math.round(out[1]) + 8); v++)
            for (let u = Math.max(0, Math.round(out[0]) - 8); u <= Math.min(cam.width - 1, Math.round(out[0]) + 8); u++) {
              const w = srgbToLinear(px[4 * (v * cam.width + u)]);
              sw += w;
              su += w * u;
              sv += w * v;
            }
          expect(sw).toBeGreaterThan(3);
          const e = Math.hypot(su / sw - out[0], sv / sw - out[1]);
          worst = Math.max(worst, e);
          sum += e;
        }
        console.log(`[review.gpu] ${scene}: marker error mean ${(sum / markers.length).toFixed(4)} px, worst ${worst.toFixed(4)} px`);
        expect(worst).toBeLessThan(0.25);
        expect(sum / markers.length).toBeLessThan(0.1);
      } finally {
        r.dispose();
        synth.dispose();
      }
    });
  }
});

interface Frame {
  pose: Pose;
  img: Float32Array;
  depth: Float32Array;
}

/**
 * Warp A into B through depth A and estimate the global residual translation of B
 * (Lucas–Kanade on all well-conditioned, unoccluded pixels). `ppShift` perturbs the
 * principal point used to project into B (control: the estimate must see it).
 */
function residualShift(cam: CameraIntrinsics, A: Frame, B: Frame, ppShift = 0) {
  const W = cam.width,
    H = cam.height;
  const xy = new Float64Array(2),
    out = new Float64Array(3);
  const RT = m3T(A.pose.R);
  let gxx = 0,
    gxy = 0,
    gyy = 0,
    bx = 0,
    by = 0,
    n = 0;
  const absRes: number[] = [];
  const camB = { ...cam, cx: cam.cx + ppShift };
  for (let v = 2; v < H - 2; v += 2)
    for (let u = 2; u < W - 2; u += 2) {
      const z = A.depth[v * W + u];
      if (!(z > 0)) continue;
      // skip depth discontinuities in A
      let zmin = z,
        zmax = z;
      for (const o of [-2, 2, -2 * W, 2 * W]) {
        const q = A.depth[v * W + u + o];
        zmin = Math.min(zmin, q);
        zmax = Math.max(zmax, q);
      }
      if (!(zmin > 0) || zmax - zmin > 0.02 * z) continue;
      undistortNormalized(cam.k1, cam.k2, (u - cam.cx) / cam.f, (v - cam.cy) / cam.f, xy);
      const Xc = [xy[0] * z, xy[1] * z, z];
      const X = Array.from(m3Vec(RT, [Xc[0] - A.pose.t[0], Xc[1] - A.pose.t[1], Xc[2] - A.pose.t[2]]));
      if (!projectRadial(camB, B.pose, X, out)) continue;
      const ub = out[0],
        vb = out[1];
      if (ub < 2 || vb < 2 || ub > W - 3 || vb > H - 3) continue;
      const zb = B.depth[Math.round(vb) * W + Math.round(ub)];
      if (!(Math.abs(zb - out[2]) < 0.01 * out[2])) continue; // occluded in B
      const ib = bilinear(B.img, W, ub, vb);
      const gx = 0.5 * (bilinear(B.img, W, ub + 1, vb) - bilinear(B.img, W, ub - 1, vb));
      const gy = 0.5 * (bilinear(B.img, W, ub, vb + 1) - bilinear(B.img, W, ub, vb - 1));
      const e = A.img[v * W + u] - ib;
      if (Math.abs(e) > 40) continue; // gross outliers (mip / specular-free but still)
      gxx += gx * gx;
      gxy += gx * gy;
      gyy += gy * gy;
      bx += gx * e;
      by += gy * e;
      n++;
      if (gx * gx + gy * gy > 16) absRes.push(Math.abs(e));
    }
  const det = gxx * gyy - gxy * gxy;
  // A(p) ≈ B(w(p) + d) ⇒ e ≈ ∇B·d
  const dx = (gyy * bx - gxy * by) / det,
    dy = (gxx * by - gxy * bx) / det;
  absRes.sort((a, b) => a - b);
  return { dx, dy, n, medianAbsResidual: absRes[absRes.length >> 1] };
}

function renderFrame(r: SynthRenderer, cam: CameraIntrinsics, pose: Pose): Frame {
  r.renderFrame([pose], 1, 0);
  return { pose, img: luma(r.readPixels(), cam.width, cam.height), depth: r.renderDepth(pose) };
}

describe('multi-view consistency of rendered frames, GT depth and GT poses', () => {
  const cases: [SynthSceneName, number, number, number, number][] = [
    ['object', 360, 480, 4.0, 4.375],
    ['yard', 480, 360, 6.0, 6.5],
  ];
  for (const [scene, W, H, ta, tb] of cases) {
    it(`${scene}: A warped into B via depth A aligns with B (global shift < 0.08 px); +1 px control is detected`, () => {
      const cfg = resolveConfig({ scene, width: W, height: H, textureScale: 0.5 });
      const cam = intrinsicsFor(cfg);
      const traj = trajectoryFor(cfg);
      const synth = buildScene(cfg);
      const r = new SynthRenderer(synth, cam, NO_FX);
      try {
        const A = renderFrame(r, cam, trajectoryPose(traj, ta));
        const B = renderFrame(r, cam, trajectoryPose(traj, tb));
        const res = residualShift(cam, A, B);
        const ctl = residualShift(cam, A, B, 1);
        console.log(`[review.gpu] ${scene}: shift (${res.dx.toFixed(4)}, ${res.dy.toFixed(4)}) px over ${res.n} px, median |res| ${res.medianAbsResidual.toFixed(2)}; control (${ctl.dx.toFixed(3)}, ${ctl.dy.toFixed(3)})`);
        expect(res.n).toBeGreaterThan(5000);
        expect(Math.hypot(res.dx, res.dy)).toBeLessThan(0.08);
        expect(ctl.dx).toBeLessThan(-0.6); // the estimator sees a 1 px principal-point error
        expect(Math.abs(ctl.dy)).toBeLessThan(0.15);
        expect(res.medianAbsResidual).toBeLessThan(6); // 0..255 luma
      } finally {
        r.dispose();
        synth.dispose();
      }
    });
  }

  it('object: GT depth map agrees with independently sampled GT surface points', () => {
    const cfg = resolveConfig({ scene: 'object', width: 360, height: 480, textureScale: 0.25 });
    const cam = intrinsicsFor(cfg);
    const synth = buildScene(cfg);
    const r = new SynthRenderer(synth, cam, NO_FX);
    try {
      const pose = trajectoryPose(trajectoryFor(cfg), 9.1);
      const depth = r.renderDepth(pose);
      const gt = sampleSurfaces(synth.surfaceMeshes, 300000, new SynthRng(5));
      const out = new Float64Array(3);
      const rel: number[] = [];
      let inImage = 0;
      for (let i = 0; i < gt.count; i++) {
        if (!projectRadial(cam, pose, gt.positions.subarray(3 * i, 3 * i + 3), out)) continue;
        const u = Math.round(out[0]),
          v = Math.round(out[1]);
        if (u < 0 || v < 0 || u >= cam.width || v >= cam.height) continue;
        inImage++;
        const d = depth[v * cam.width + u];
        const e = Math.abs(d - out[2]) / out[2];
        if (e < 0.02) rel.push(e);
      }
      rel.sort((a, b) => a - b);
      const med = rel[rel.length >> 1],
        p90 = rel[Math.floor(rel.length * 0.9)];
      console.log(`[review.gpu] depth vs GT samples: ${rel.length}/${inImage} consistent, median rel ${med.toExponential(2)}, p90 ${p90.toExponential(2)}`);
      expect(rel.length).toBeGreaterThan(0.2 * inImage); // the rest is occluded (~70 % here)
      expect(med).toBeLessThan(1e-3);
      expect(p90).toBeLessThan(4e-3);
    } finally {
      r.dispose();
      synth.dispose();
    }
  });
});

/** Separable Gaussian blur (σ), clamped borders. */
function blur(img: Float32Array, W: number, H: number, sigma: number): Float32Array {
  const rad = Math.ceil(3 * sigma);
  const k = new Float32Array(2 * rad + 1);
  let s = 0;
  for (let i = -rad; i <= rad; i++) s += k[i + rad] = Math.exp((-i * i) / (2 * sigma * sigma));
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const tmp = new Float32Array(W * H),
    out = new Float32Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let a = 0;
      for (let i = -rad; i <= rad; i++) a += k[i + rad] * img[y * W + Math.min(W - 1, Math.max(0, x + i))];
      tmp[y * W + x] = a;
    }
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let a = 0;
      for (let i = -rad; i <= rad; i++) a += k[i + rad] * tmp[Math.min(H - 1, Math.max(0, y + i)) * W + x];
      out[y * W + x] = a;
    }
  return out;
}

/**
 * Density (per megapixel) of DoG blob extrema (σ = 1.6 / 2.26 / 3.2, |DoG| > thr on a
 * 0..1 image) — a SIFT-like detector, inside and outside a pixel mask.
 */
function blobDensity(img: Float32Array, W: number, H: number, mask: Uint8Array, thr = 0.02 / 3) {
  const g = img.map((v) => v / 255);
  const s = [1.6, 1.6 * Math.SQRT2, 3.2, 3.2 * Math.SQRT2].map((sg) => blur(g, W, H, sg));
  const D = [0, 1, 2].map((k) => s[k + 1].map((v, i) => v - s[k][i]));
  let inM = 0,
    outM = 0,
    nIn = 0,
    nOut = 0;
  for (let i = 0; i < W * H; i++) if (mask[i]) nIn++;
  nOut = W * H - nIn;
  const d = D[1];
  for (let y = 4; y < H - 4; y++)
    for (let x = 4; x < W - 4; x++) {
      const i = y * W + x;
      const v = d[i];
      if (Math.abs(v) <= thr) continue;
      let ext = true;
      for (let l = 0; l < 3 && ext; l++)
        for (let dy = -1; dy <= 1 && ext; dy++)
          for (let dx = -1; dx <= 1; dx++) {
            if (l === 1 && dx === 0 && dy === 0) continue;
            const q = D[l][i + dy * W + dx];
            if (v > 0 ? q >= v : q <= v) {
              ext = false;
              break;
            }
          }
      if (!ext) continue;
      if (mask[i]) inM++;
      else outM++;
    }
  return { ground: (inM / nIn) * 1e6, other: (outM / Math.max(1, nOut)) * 1e6, groundFraction: nIn / (W * H) };
}

describe('texture richness', () => {
  it('yard: the lawn near the camera has SIFT-scale structure (not a featureless green)', () => {
    const cfg = resolveConfig({ scene: 'yard' }); // full resolution + full texture scale
    const cam = intrinsicsFor(cfg);
    const traj = trajectoryFor(cfg);
    const synth = buildScene(cfg);
    const r = new SynthRenderer(synth, cam, NO_FX);
    const xy = new Float64Array(2);
    try {
      const dens: { ground: number; other: number; groundFraction: number }[] = [];
      for (const tau of [8, 26]) {
        const pose = trajectoryPose(traj, tau);
        const f = renderFrame(r, cam, pose);
        const RT = m3T(pose.R);
        const mask = new Uint8Array(cam.width * cam.height);
        for (let v = 0; v < cam.height; v++)
          for (let u = 0; u < cam.width; u++) {
            const z = f.depth[v * cam.width + u];
            if (!(z > 0)) continue;
            undistortNormalized(cam.k1, cam.k2, (u - cam.cx) / cam.f, (v - cam.cy) / cam.f, xy);
            const X = m3Vec(RT, [xy[0] * z - pose.t[0], xy[1] * z - pose.t[1], z - pose.t[2]]);
            // grass / soil / path of the lawn (not the patio, boards or plants), within 14 m
            if (Math.abs(X[2] - groundHeight(X[0], X[1])) < 0.01 && Math.hypot(X[0], X[1]) < 14) mask[v * cam.width + u] = 1;
          }
        const d = blobDensity(f.img, cam.width, cam.height, mask);
        console.log(`[review.gpu] yard τ=${tau}: ground ${(100 * d.groundFraction).toFixed(0)} % of the frame, DoG blobs/Mpx ground ${d.ground.toFixed(0)} vs other ${d.other.toFixed(0)}`);
        dens.push(d);
      }
      for (const d of dens) {
        expect(d.groundFraction).toBeGreaterThan(0.15);
        expect(d.ground).toBeGreaterThan(0.25 * d.other);
      }
    } finally {
      r.dispose();
      synth.dispose();
    }
  });
});
