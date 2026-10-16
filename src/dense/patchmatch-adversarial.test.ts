/**
 * Adversarial tests for the PatchMatch CPU reference and the shared set-up
 * (sign conventions with unequal intrinsics, random-init distribution, analytic
 * slanted plane, degenerate / non-finite inputs, option edge cases, mixed source
 * resolutions, prior-initialised multi-scale refinement).
 */
import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, Pose } from '../core/types';
import { buildSourceCamera, normalPerturbation, planeHomography, priorToPlanes, relativePose, resolvePatchMatchOptions } from './patchmatch-common';
import { patchMatchCPU } from './patchmatch-cpu';
import {
  evaluateDepthMap,
  formatDepthStats,
  lookAt,
  makeNearScene,
  makeSyntheticDataset,
  pinholeIntrinsics,
  renderDataset,
  renderSynthetic,
  surfaceMask,
  type SynthScene,
} from './synthetic';
import { MemoryDepthMapStore, type DenseView, type DepthMap } from './types';

const log = (...a: unknown[]): void => console.log('[patchmatch-adversarial]', ...a);

function emptyView(id: number, cam: CameraIntrinsics, pose: Pose): DenseView {
  return { id, cam, pose, image: { width: cam.width, height: cam.height, data: new Uint8Array(cam.width * cam.height * 4) } };
}

/** Rotation from axis-angle (Rodrigues), row-major. */
function rot(ax: number, ay: number, az: number, angle: number): Float64Array {
  const n = Math.hypot(ax, ay, az);
  const x = ax / n;
  const y = ay / n;
  const z = az / n;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const C = 1 - c;
  return new Float64Array([c + x * x * C, x * y * C - z * s, x * z * C + y * s, y * x * C + z * s, c + y * y * C, y * z * C - x * s, z * x * C - y * s, z * y * C + x * s, c + z * z * C]);
}

function toCam(p: Pose, X: ArrayLike<number>): number[] {
  const R = p.R;
  return [0, 1, 2].map((i) => R[3 * i] * X[0] + R[3 * i + 1] * X[1] + R[3 * i + 2] * X[2] + p.t[i]);
}

function toWorld(p: Pose, Xc: ArrayLike<number>): number[] {
  const R = p.R;
  const d = [Xc[0] - p.t[0], Xc[1] - p.t[1], Xc[2] - p.t[2]];
  return [0, 1, 2].map((k) => R[k] * d[0] + R[3 + k] * d[1] + R[6 + k] * d[2]);
}

/** Median of the SIGNED relative depth error over valid masked pixels. */
function signedMedianRelError(dm: DepthMap, gt: Float32Array, mask: Uint8Array): number {
  const rel: number[] = [];
  for (let i = 0; i < gt.length; i++) if (mask[i] && dm.depth[i] > 0 && gt[i] > 0) rel.push((dm.depth[i] - gt[i]) / gt[i]);
  rel.sort((a, b) => a - b);
  return rel.length ? rel[rel.length >> 1] : NaN;
}

function allFinite(dm: DepthMap): boolean {
  for (let i = 0; i < dm.depth.length; i++) {
    if (!Number.isFinite(dm.depth[i]) || !Number.isFinite(dm.cost[i])) return false;
    if (!Number.isFinite(dm.normal[3 * i]) || !Number.isFinite(dm.normal[3 * i + 1]) || !Number.isFinite(dm.normal[3 * i + 2])) return false;
  }
  return true;
}

/** Valid pixels have unit normals facing the camera and depth inside the range; invalid ones are all-zero. */
function expectWellFormed(dm: DepthMap, cam: CameraIntrinsics, range: { min: number; max: number }, maxCost: number): void {
  let bad = 0;
  for (let i = 0; i < dm.depth.length; i++) {
    const z = dm.depth[i];
    const nx = dm.normal[3 * i];
    const ny = dm.normal[3 * i + 1];
    const nz = dm.normal[3 * i + 2];
    if (!(dm.cost[i] >= 0 && dm.cost[i] <= 2)) bad++;
    if (z === 0) {
      if (nx !== 0 || ny !== 0 || nz !== 0) bad++;
      continue;
    }
    const x = i % dm.width;
    const y = (i - x) / dm.width;
    if (!(z >= range.min && z <= range.max)) bad++;
    if (Math.abs(Math.hypot(nx, ny, nz) - 1) > 1e-4) bad++;
    if (!((nx * (x - cam.cx)) / cam.f + (ny * (y - cam.cy)) / cam.f + nz < 0)) bad++;
    if (!(dm.cost[i] <= maxCost)) bad++;
  }
  expect(bad).toBe(0);
}

describe('PatchMatch set-up: sign conventions with unequal cameras', () => {
  // Reference and source with different f, principal points (off-centre) and sizes,
  // and arbitrary (rolled) rotations, not produced by lookAt.
  const camR: CameraIntrinsics = { width: 64, height: 48, f: 55, cx: 30.2, cy: 25.7, k1: 0, k2: 0 };
  const camS: CameraIntrinsics = { width: 100, height: 70, f: 91, cx: 51.3, cy: 33.1, k1: 0, k2: 0 };
  const poseR: Pose = { R: rot(0.3, -1, 0.2, 0.7), t: new Float64Array([0.4, -0.2, 3.1]) };
  const poseS: Pose = { R: rot(-0.5, 0.8, 0.4, 0.9), t: new Float64Array([-0.3, 0.5, 2.6]) };
  const ref = emptyView(0, camR, poseR);
  const src = emptyView(1, camS, poseS);

  it('relativePose composes world-to-camera poses (x_s = R_sr x_r + t_sr)', () => {
    const { R, t } = relativePose(poseR, poseS);
    for (const X of [
      [0.3, 0.2, 0.1],
      [-2, 5, 1.5],
    ]) {
      const xr = toCam(poseR, X);
      const xs = toCam(poseS, X);
      for (let i = 0; i < 3; i++) expect(R[3 * i] * xr[0] + R[3 * i + 1] * xr[1] + R[3 * i + 2] * xr[2] + t[i]).toBeCloseTo(xs[i], 10);
    }
  });

  it('plane homography and forward/backward maps hold for different K_r, K_s', () => {
    const sc = buildSourceCamera(ref, src);
    let n = [0.3, -0.4, -0.8];
    const nl = Math.hypot(n[0], n[1], n[2]);
    n = n.map((c) => c / nl);
    const u0 = 30;
    const v0 = 20;
    const z0 = 2.7;
    const H = planeHomography(camR, sc, u0, v0, z0, n)!;
    expect(H).not.toBeNull();
    const r0 = [(u0 - camR.cx) / camR.f, (v0 - camR.cy) / camR.f, 1];
    const delta = -z0 * (n[0] * r0[0] + n[1] * r0[1] + n[2] * r0[2]);
    for (const [du, dv] of [
      [0, 0],
      [5, -3],
      [-7, 6],
      [20, 15],
    ]) {
      const u = u0 + du;
      const v = v0 + dv;
      const r = [(u - camR.cx) / camR.f, (v - camR.cy) / camR.f, 1];
      const z = -delta / (n[0] * r[0] + n[1] * r[1] + n[2] * r[2]);
      const X = toWorld(poseR, [r[0] * z, r[1] * z, z]);
      const xs = toCam(poseS, X);
      const pu = (camS.f * xs[0]) / xs[2] + camS.cx;
      const pv = (camS.f * xs[1]) / xs[2] + camS.cy;
      const hz = H[6] * u + H[7] * v + H[8];
      expect((H[0] * u + H[1] * v + H[2]) / hz).toBeCloseTo(pu, 8);
      expect((H[3] * u + H[4] * v + H[5]) / hz).toBeCloseTo(pv, 8);
      // geometric-consistency maps: p_s ≅ z A p̃ + b, K_r X_r = z_s M p̃_s + v
      const q = [0, 1, 2].map((i) => z * (sc.A[3 * i] * u + sc.A[3 * i + 1] * v + sc.A[3 * i + 2]) + sc.b[i]);
      expect(q[0] / q[2]).toBeCloseTo(pu, 8);
      expect(q[1] / q[2]).toBeCloseTo(pv, 8);
      expect(q[2]).toBeCloseTo(xs[2], 8);
      const k = [0, 1, 2].map((i) => xs[2] * (sc.M[3 * i] * pu + sc.M[3 * i + 1] * pv + sc.M[3 * i + 2]) + sc.v[i]);
      expect(k[0] / k[2]).toBeCloseTo(u, 8);
      expect(k[1] / k[2]).toBeCloseTo(v, 8);
      expect(k[2]).toBeCloseTo(z, 8);
    }
    // homography is invariant to the scale of n (unnormalised prior normals)
    const H2 = planeHomography(camR, sc, u0, v0, z0, n.map((c) => 3 * c))!;
    for (let i = 0; i < 9; i++) expect(H2[i]).toBeCloseTo(H[i], 10);
  });
});

describe('priorToPlanes', () => {
  it('re-intersects upsampled slanted planes with the target pixel ray', () => {
    const cam = pinholeIntrinsics(80, 60, 1.0);
    const half = { ...cam, width: 40, height: 30, f: cam.f * 0.5, cx: (cam.cx + 0.5) * 0.5 - 0.5, cy: (cam.cy + 0.5) * 0.5 - 0.5 };
    // plane n·X + d = 0 (camera frame), strongly slanted
    const nRaw = [0, -0.8, -0.6];
    const d = 3;
    const prior: DepthMap = { width: 40, height: 30, depth: new Float32Array(1200), normal: new Float32Array(3600), cost: new Float32Array(1200) };
    for (let y = 0; y < 30; y++) {
      for (let x = 0; x < 40; x++) {
        const r = [(x - half.cx) / half.f, (y - half.cy) / half.f, 1];
        const z = -d / (nRaw[0] * r[0] + nRaw[1] * r[1] + nRaw[2] * r[2]);
        const i = y * 40 + x;
        prior.depth[i] = z;
        prior.normal.set(nRaw, 3 * i);
      }
    }
    const exact = priorToPlanes(prior, 80, 60, undefined, cam);
    const plain = priorToPlanes(prior, 80, 60);
    let maxErr = 0;
    let maxPlain = 0;
    for (let y = 0; y < 60; y++) {
      for (let x = 0; x < 80; x++) {
        const r = [(x - cam.cx) / cam.f, (y - cam.cy) / cam.f, 1];
        const z = -d / (nRaw[0] * r[0] + nRaw[1] * r[1] + nRaw[2] * r[2]);
        const i = 4 * (y * 80 + x);
        maxErr = Math.max(maxErr, Math.abs(exact[i + 3] - z) / z);
        maxPlain = Math.max(maxPlain, Math.abs(plain[i + 3] - z) / z);
      }
    }
    log(`prior upsampling, slanted plane: max rel depth error ${maxErr.toExponential(2)} (re-intersected) vs ${maxPlain.toExponential(2)} (copied)`);
    expect(maxErr).toBeLessThan(1e-5);
    expect(maxPlain).toBeGreaterThan(1e-3);
  });
});

describe('patchMatchCPU adversarial', () => {
  it('random initialisation: depths uniform in INVERSE depth, normals unit and facing the camera', () => {
    const ds = makeSyntheticDataset('far', { width: 64, height: 48, numSources: 2, supersample: 1 });
    const dm = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, { iterations: 0, maxCost: 2 });
    expect(allFinite(dm)).toBe(true);
    expectWellFormed(dm, ds.views[0].cam, ds.range, 2);
    const a = 1 / ds.range.max;
    const b = 1 / ds.range.min;
    const bins = 10;
    const hist = new Float64Array(bins);
    const histZ = new Float64Array(bins);
    let n = 0;
    for (let i = 0; i < dm.depth.length; i++) {
      const z = dm.depth[i];
      if (!(z > 0)) continue;
      n++;
      hist[Math.min(bins - 1, Math.floor(((1 / z - a) / (b - a)) * bins))]++;
      histZ[Math.min(bins - 1, Math.floor(((z - ds.range.min) / (ds.range.max - ds.range.min)) * bins))]++;
    }
    expect(n).toBe(dm.depth.length); // maxCost 2 keeps every pixel (textureless ones keep their random init)
    let chi2 = 0;
    for (const h of hist) chi2 += (h - n / bins) ** 2 / (n / bins);
    log(`init 1/z histogram chi² = ${chi2.toFixed(1)} (9 dof); z histogram first/last bin ${histZ[0]} / ${histZ[bins - 1]}`);
    expect(chi2).toBeLessThan(30);
    // uniform in 1/z ⇒ strongly skewed towards near depths in z
    expect(histZ[0]).toBeGreaterThan(5 * histZ[bins - 1]);
  });

  it('slanted textured plane: depth and normal match the analytic plane (sign conventions end to end)', () => {
    const tilt = (40 * Math.PI) / 180;
    const scene: SynthScene = {
      primitives: [
        {
          kind: 'quad',
          center: [0, 4, 0],
          axisU: [1, 0, 0],
          axisV: [0, Math.sin(tilt), Math.cos(tilt)],
          halfU: 20,
          halfV: 20,
          material: { scale: 0.6, octaves: 8, seed: 5, color: [0.7, 0.6, 0.5] },
        },
      ],
    };
    const cam = pinholeIntrinsics(96, 72, 1.0);
    const eye = [0.2, 0, 0.3];
    const target = [0, 4, 0];
    const poses = [lookAt(eye, target)];
    for (let i = 0; i < 4; i++) {
      const a = (Math.PI / 2) * i + 0.4;
      poses.push(lookAt([eye[0] + 0.35 * Math.cos(a), eye[1], eye[2] + 0.25 * Math.sin(a)], target));
    }
    const ds = renderDataset(scene, cam, poses, { supersample: 2 });
    const dm = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range);
    // interior pixels (the window is clamped at the image border)
    const mask = new Uint8Array(96 * 72);
    for (let y = 6; y < 66; y++) for (let x = 6; x < 90; x++) mask[y * 96 + x] = 1;
    const s = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0], { mask, relTol: 0.005 });
    const bias = signedMedianRelError(dm, ds.gtDepth[0], mask);
    log(`slanted plane (40°): ${formatDepthStats(s, 0.005)} | signed median ${(100 * bias).toFixed(4)}%`);
    expect(s.validFraction).toBeGreaterThan(0.97);
    expect(s.accurateFraction).toBeGreaterThan(0.8);
    expect(s.medianRelError).toBeLessThan(0.0025);
    expect(s.medianNormalErrorDeg).toBeLessThan(4);
    // no systematic bias: a half-pixel convention error (sampling, intrinsics,
    // rendering) shows up as a signed depth offset of ~0.3 % here (½ px of ~8 px disparity)
    expect(Math.abs(bias)).toBeLessThan(0.0005);
    expectWellFormed(dm, cam, ds.range, 0.5);
  });

  it('textureless reference: every pixel invalid, cost exactly 2, zero normals, no NaN', () => {
    const ds = makeSyntheticDataset('near', { width: 40, height: 30, numSources: 2, supersample: 1 });
    const flat: DenseView = { ...ds.views[0], image: { width: 40, height: 30, data: new Uint8Array(40 * 30 * 4).fill(128) } };
    const dm = patchMatchCPU(flat, ds.views.slice(1), ds.range, { iterations: 2 });
    expect(allFinite(dm)).toBe(true);
    expect(dm.depth.every((z) => z === 0)).toBe(true);
    expect(dm.cost.every((c) => c === 2)).toBe(true);
    expect(dm.normal.every((c) => c === 0)).toBe(true);
  });

  it('non-finite / degenerate geometric inputs never produce NaN', () => {
    const W = 48;
    const H = 36;
    const ds = makeSyntheticDataset('near', { width: W, height: H, numSources: 3, supersample: 1 });
    const n = W * H;
    const prior: DepthMap = { width: W, height: H, depth: new Float32Array(n), normal: new Float32Array(3 * n), cost: new Float32Array(n) };
    for (let i = 0; i < n; i++) {
      // a mix of NaN / Inf / negative depths, zero / NaN normals and plausible values
      const k = i % 5;
      prior.depth[i] = k === 0 ? NaN : k === 1 ? Infinity : k === 2 ? -1 : 3;
      prior.normal.set(k === 3 ? [0, 0, 0] : k === 4 ? [NaN, 0, -1] : [0, 0, -1], 3 * i);
    }
    const junk: DepthMap = { width: 24, height: 18, depth: new Float32Array(24 * 18), normal: new Float32Array(3 * 24 * 18), cost: new Float32Array(24 * 18) };
    for (let i = 0; i < junk.depth.length; i++) junk.depth[i] = [NaN, Infinity, -Infinity, 0, 1e30, 2.5][i % 6];
    const none: DepthMap = { width: 0, height: 0, depth: new Float32Array(0), normal: new Float32Array(0), cost: new Float32Array(0) };
    const dm = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, { geometricIterations: 2 }, { prior, sourceDepths: [junk, none, junk] });
    expect(allFinite(dm)).toBe(true);
    expectWellFormed(dm, ds.views[0].cam, ds.range, 0.5);
    const s = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0]);
    log(`junk geometric inputs: ${formatDepthStats(s)}`);
    expect(s.validFraction).toBeGreaterThan(0.5);
  });

  it('perturbation = 0 in a prior-initialised pass keeps normals finite (division-by-zero regression)', () => {
    const o = resolvePatchMatchOptions({ perturbation: 0 });
    for (let it = 0; it < 4; it++) expect(Number.isFinite(normalPerturbation(o, it, true))).toBe(true);
    const ds = makeSyntheticDataset('near', { width: 32, height: 24, numSources: 2, supersample: 1 });
    const photo = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, { iterations: 2 });
    const dm = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, { perturbation: 0, geometricIterations: 2 }, { prior: photo, sourceDepths: [photo, photo] });
    expect(allFinite(dm)).toBe(true);
    expectWellFormed(dm, ds.views[0].cam, ds.range, 0.5);
  });

  it('option edge cases and odd image sizes', () => {
    const ds = makeSyntheticDataset('near', { width: 37, height: 29, numSources: 2, supersample: 1 });
    const r = ds.range;
    const cam = ds.views[0].cam;
    // odd size, topK > N, no refinement proposals
    for (const opts of [{ iterations: 2 }, { iterations: 2, topK: 10 }, { iterations: 2, numSamplesPerIter: 0 }, { iterations: 1, patchRadius: 2, patchStep: 1 }]) {
      const dm = patchMatchCPU(ds.views[0], ds.views.slice(1), r, opts);
      expect(dm.width).toBe(37);
      expect(dm.height).toBe(29);
      expect(allFinite(dm)).toBe(true);
      expectWellFormed(dm, cam, r, 0.5);
      expect(dm.depth.some((z) => z > 0)).toBe(true);
    }
    // a window with a single sample has zero variance: everything textureless, no NaN
    const single = patchMatchCPU(ds.views[0], ds.views.slice(1), r, { iterations: 1, patchRadius: 1, patchStep: 3 });
    expect(allFinite(single)).toBe(true);
    expect(single.depth.every((z) => z === 0)).toBe(true);
    // different seeds give different (but both valid) results; same seed is deterministic
    const a = patchMatchCPU(ds.views[0], ds.views.slice(1), r, { iterations: 1, seed: 3 });
    const b = patchMatchCPU(ds.views[0], ds.views.slice(1), r, { iterations: 1, seed: 4 });
    const a2 = patchMatchCPU(ds.views[0], ds.views.slice(1), r, { iterations: 1, seed: 3 });
    expect(Array.from(a.depth)).toEqual(Array.from(a2.depth));
    expect(Array.from(a.depth)).not.toEqual(Array.from(b.depth));
    // 1-row / 1-column images
    for (const [w, h] of [
      [1, 1],
      [17, 1],
      [1, 13],
    ]) {
      const c = pinholeIntrinsics(w, h, 1.0);
      const v = (id: number, x: number): DenseView => ({ id, cam: c, pose: lookAt([x, 0, 0], [0, 5, 0]), image: { width: w, height: h, data: new Uint8Array(w * h * 4).fill(90) } });
      const dm = patchMatchCPU(v(0, 0), [v(1, 0.3)], { min: 1, max: 10 }, { iterations: 1 });
      expect(dm.depth.length).toBe(w * h);
      expect(allFinite(dm)).toBe(true);
    }
  });

  it('a zero-baseline source among informative ones neither produces NaN nor breaks the depth map', () => {
    const ds = makeSyntheticDataset('near', { width: 96, height: 72, numSources: 4 });
    const mask = surfaceMask(ds.views[0].cam, ds.gtDepth[0], ds.gtNormal[0]);
    const twin: DenseView = { ...ds.views[0], id: 99 }; // same pose: depth-independent, perfectly photo-consistent
    const a = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range);
    const b = patchMatchCPU(ds.views[0], [twin, ...ds.views.slice(1)], ds.range);
    const sa = evaluateDepthMap(a, ds.gtDepth[0], ds.gtNormal[0], { mask });
    const sb = evaluateDepthMap(b, ds.gtDepth[0], ds.gtNormal[0], { mask });
    log(`4 sources: ${formatDepthStats(sa)}`);
    log(`+ zero-baseline twin: ${formatDepthStats(sb)}`);
    expect(allFinite(b)).toBe(true);
    expect(sb.accurateFraction).toBeGreaterThan(sa.accurateFraction - 0.05);
    expect(sb.medianRelError).toBeLessThan(0.01);
  });

  it('sources with a different resolution / focal length than the reference', () => {
    const scene = makeNearScene();
    const base = makeSyntheticDataset('near', { width: 96, height: 72, numSources: 4 });
    const mask = surfaceMask(base.views[0].cam, base.gtDepth[0], base.gtNormal[0]);
    for (const scale of [0.75, 1.5]) {
      const camS = pinholeIntrinsics(Math.round(96 * scale), Math.round(72 * scale), (60 * Math.PI) / 180);
      const views: DenseView[] = [base.views[0]];
      for (let i = 1; i < base.views.length; i++) {
        views.push({ id: i, cam: camS, pose: base.views[i].pose, image: renderSynthetic(scene, camS, base.views[i].pose, { supersample: 2 }).image });
      }
      const dm = patchMatchCPU(views[0], views.slice(1), base.range);
      const s = evaluateDepthMap(dm, base.gtDepth[0], base.gtNormal[0], { mask });
      log(`sources at ${scale}× resolution: ${formatDepthStats(s)}`);
      expect(s.validFraction).toBeGreaterThan(0.9);
      expect(s.medianRelError).toBeLessThan(0.01);
      expectWellFormed(dm, views[0].cam, base.range, 0.5);
    }
  });

  it('prior-initialised photometric pass (multi-scale refinement) beats random init at equal iterations', () => {
    const full = makeSyntheticDataset('near', { width: 128, height: 96, numSources: 4 });
    const half = makeSyntheticDataset('near', { width: 64, height: 48, numSources: 4 });
    const mask = surfaceMask(full.views[0].cam, full.gtDepth[0], full.gtNormal[0]);
    const ev = (d: DepthMap) => evaluateDepthMap(d, full.gtDepth[0], full.gtNormal[0], { mask });
    const coarse = patchMatchCPU(half.views[0], half.views.slice(1), half.range);
    const rnd2 = ev(patchMatchCPU(full.views[0], full.views.slice(1), full.range, { iterations: 2 }));
    const rnd6 = ev(patchMatchCPU(full.views[0], full.views.slice(1), full.range));
    const fine = patchMatchCPU(full.views[0], full.views.slice(1), full.range, { iterations: 2, geometric: false }, { prior: coarse, sourceDepths: [] });
    const pri2 = ev(fine);
    log(`random init, 2 it : ${formatDepthStats(rnd2)}`);
    log(`random init, 6 it : ${formatDepthStats(rnd6)}`);
    log(`½-res prior, 2 it : ${formatDepthStats(pri2)}`);
    expect(pri2.accurateFraction).toBeGreaterThan(rnd2.accurateFraction + 0.02);
    expect(pri2.accurateFraction).toBeGreaterThan(rnd6.accurateFraction - 0.03);
    expectWellFormed(fine, full.views[0].cam, full.range, 0.5);
  });
});

describe('synthetic lookAt / option guards', () => {
  it('lookAt stays a proper rotation when looking along the up vector', () => {
    for (const [eye, target, up] of [
      [[0, 0, 0], [5, 0, 0], [1, 0, 0]],
      [[1, 2, 3], [1, 2, -7], [0, 0, 1]],
      [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    ]) {
      const { R, t } = lookAt(eye, target, up);
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          const d = R[3 * i] * R[3 * j] + R[3 * i + 1] * R[3 * j + 1] + R[3 * i + 2] * R[3 * j + 2];
          expect(d).toBeCloseTo(i === j ? 1 : 0, 12);
        }
      }
      const det = R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) + R[2] * (R[3] * R[7] - R[4] * R[6]);
      expect(det).toBeCloseTo(1, 12);
      const xc = toCam({ R, t }, target);
      expect(Math.hypot(xc[0], xc[1])).toBeLessThan(1e-9);
      expect(xc[2]).toBeGreaterThan(0);
    }
  });

  it('non-positive sigmas fall back to the defaults (no all-zero / NaN bilateral weights)', () => {
    const o = resolvePatchMatchOptions({ sigmaSpatial: 0, sigmaColor: -1 });
    expect(o.sigmaSpatial).toBe(5);
    expect(o.sigmaColor).toBe(3);
    const ds = makeSyntheticDataset('near', { width: 32, height: 24, numSources: 2, supersample: 1 });
    const dm = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, { iterations: 1, sigmaSpatial: 0, sigmaColor: NaN });
    expect(allFinite(dm)).toBe(true);
    expect(dm.depth.some((z) => z > 0)).toBe(true);
  });
});

describe('MemoryDepthMapStore', () => {
  it('stores, lists and deletes maps', async () => {
    const s = new MemoryDepthMapStore();
    const d: DepthMap = { width: 1, height: 1, depth: new Float32Array([2]), normal: new Float32Array([0, 0, -1]), cost: new Float32Array([0.1]) };
    expect(await s.get(3)).toBeNull();
    await s.put(3, d);
    await s.put(1, d);
    expect(await s.get(3)).toBe(d);
    expect(s.ids()).toEqual([1, 3]);
    expect(s.delete(3)).toBe(true);
    expect(s.has(3)).toBe(false);
    s.clear();
    expect(s.ids()).toEqual([]);
  });
});
