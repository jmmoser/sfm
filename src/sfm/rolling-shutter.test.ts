import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, Pose } from '../core/types';
import { projectCam } from '../geometry/camera';
import { Rng } from '../math/random';
import { rotvecToMat3 } from '../math/rotation';
import { mat3Mul, mat3MulVec } from '../math/mat3';
import { cameraFromDevice } from './extras-testutil';
import {
  correctKeypointsRS,
  estimateRollingShutter,
  rsLineTimeMs,
  rsModelFromDevice,
  rsModelToDevice,
  type RollingShutterModel,
  type RSImageObservations,
} from './rolling-shutter';

const cam: CameraIntrinsics = { width: 960, height: 720, f: 753, cx: 481, cy: 358, k1: -0.03, k2: 0.005 };

/** Pixel of camera-frame point xc seen by a rolling-shutter camera rotating with ω (rad/s, camera frame). */
function rsProject(xc: ArrayLike<number>, omega: ArrayLike<number>, model: RollingShutterModel): Float64Array | null {
  const out = new Float64Array(2);
  let tau = 0;
  for (let it = 0; it < 8; it++) {
    // x(τ) = exp(−[ω τ]×) x(0)
    const R = rotvecToMat3([-omega[0] * tau, -omega[1] * tau, -omega[2] * tau]);
    if (!projectCam(cam, mat3MulVec(R, xc), out)) return null;
    tau = rsLineTimeMs(model, out[0], out[1], cam.width, cam.height) / 1000;
  }
  return out;
}

function scene(rng: Rng, model: RollingShutterModel, nImages = 14, noisePx = 0.3) {
  const pts: number[] = [];
  for (let i = 0; i < 400; i++) pts.push(rng.uniform(-3, 3), rng.uniform(-2, 2), rng.uniform(4, 9));
  const images: RSImageObservations[] = [];
  for (let k = 0; k < nImages; k++) {
    const R = rotvecToMat3([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05)]);
    const t = Float64Array.of(rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2));
    const pose: Pose = { R, t };
    const w = Float64Array.of(rng.gaussian() * 0.6, rng.gaussian() * 0.8, rng.gaussian() * 0.3);
    const uv: number[] = [],
      X: number[] = [];
    for (let i = 0; i < pts.length / 3; i++) {
      const P = [pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]];
      const xc = mat3MulVec(R, P);
      for (let a = 0; a < 3; a++) xc[a] += t[a];
      const p = rsProject(xc, w, model);
      if (!p || p[0] < 0 || p[1] < 0 || p[0] > cam.width - 1 || p[1] > cam.height - 1) continue;
      uv.push(p[0] + noisePx * rng.gaussian(), p[1] + noisePx * rng.gaussian());
      X.push(P[0], P[1], P[2]);
    }
    // start from a slightly wrong pose (as a BA on RS-distorted data would give)
    const Rn = mat3Mul(rotvecToMat3([0.002, -0.001, 0.0015]), R);
    images.push({ id: k, cam, pose: { R: Rn, t: Float64Array.from(t) }, omega: w, uv: Float64Array.from(uv), X: Float64Array.from(X) });
  }
  return images;
}

describe('rolling-shutter correction', () => {
  it('maps rolling-shutter keypoints back to the global-shutter projection at the frame stamp', () => {
    const rng = new Rng(1);
    for (const model of [
      { axis: 'rows', readoutMs: 22 },
      { axis: 'cols', readoutMs: -30 },
    ] as RollingShutterModel[]) {
      const w = [0.4, -1.3, 0.5];
      let maxErr = 0,
        maxShift = 0;
      for (let i = 0; i < 200; i++) {
        const xc = [rng.uniform(-3, 3), rng.uniform(-2, 2), rng.uniform(3, 8)];
        const p = rsProject(xc, w, model);
        const g = new Float64Array(2);
        if (!p || !projectCam(cam, xc, g)) continue;
        const kp = Float32Array.of(p[0], p[1], 2, 0.5);
        const c = correctKeypointsRS(kp, cam, w, model);
        maxErr = Math.max(maxErr, Math.hypot(c[0] - g[0], c[1] - g[1]));
        maxShift = Math.max(maxShift, Math.hypot(p[0] - g[0], p[1] - g[1]));
        expect(c[2]).toBe(2);
        expect(c[3]).toBe(0.5);
      }
      expect(maxShift).toBeGreaterThan(5); // the RS displacement is real…
      expect(maxErr).toBeLessThan(0.01); // …and removed (float32 keypoints)
    }
    // readout 0 is the identity (in place)
    const kp = Float32Array.of(100, 200, 1, 0);
    expect(Array.from(correctKeypointsRS(kp, cam, [1, 1, 1], { axis: 'rows', readoutMs: 0 }, kp))).toEqual([100, 200, 1, 0]);
  });

  it('with the camera velocity and point depths, also removes the translation during the readout', () => {
    const rng = new Rng(4);
    const model: RollingShutterModel = { axis: 'cols', readoutMs: -22 };
    // close-range orbit: yaw 0.8 rad/s, sideways at 0.8 rad/s × 0.5 m (camera frame), slight roll
    const w = [0.02, 0.72, -0.35];
    const vel = [-0.4, 0.01, 0.02];
    let maxErr = 0,
      maxErrRot = 0,
      n = 0;
    for (let i = 0; i < 200; i++) {
      const xc = [rng.uniform(-0.2, 0.2), rng.uniform(-0.25, 0.25), rng.uniform(0.35, 0.7)];
      const g = new Float64Array(2);
      if (!projectCam(cam, xc, g)) continue;
      // RS camera: at line time τ it is rotated by −ωτ and moved by vτ (camera frame of the stamp)
      const out = new Float64Array(2);
      let tau = 0,
        ok = true;
      for (let it = 0; it < 8 && ok; it++) {
        const R = rotvecToMat3([-w[0] * tau, -w[1] * tau, -w[2] * tau]);
        ok = projectCam(cam, mat3MulVec(R, [xc[0] - vel[0] * tau, xc[1] - vel[1] * tau, xc[2] - vel[2] * tau]), out);
        tau = rsLineTimeMs(model, out[0], out[1], cam.width, cam.height) / 1000;
      }
      if (!ok) continue;
      const kp = Float32Array.of(out[0], out[1], 1, 0);
      const c = correctKeypointsRS(kp, cam, w, model, undefined, { velocity: vel, invDepth: [1 / xc[2]], defaultInvDepth: 0 });
      const r = correctKeypointsRS(kp, cam, w, model);
      maxErr = Math.max(maxErr, Math.hypot(c[0] - g[0], c[1] - g[1]));
      maxErrRot = Math.max(maxErrRot, Math.hypot(r[0] - g[0], r[1] - g[1]));
      n++;
    }
    expect(n).toBeGreaterThan(100);
    expect(maxErrRot).toBeGreaterThan(3); // rotation-only leaves the translation (≈ 0.4 m/s · 11 ms / 0.35 m)
    expect(maxErr).toBeLessThan(0.05); // first-order translational term
    // unknown depth: the default inverse depth is used; 0 = rotation only
    const kp = Float32Array.of(300, 200, 1, 0);
    const a0 = correctKeypointsRS(kp, cam, w, model, undefined, { velocity: vel, invDepth: [NaN], defaultInvDepth: 0 });
    const a1 = correctKeypointsRS(kp, cam, w, model);
    expect(Array.from(a0)).toEqual(Array.from(a1));
    const a2 = correctKeypointsRS(kp, cam, w, model, undefined, { velocity: vel, invDepth: [NaN], defaultInvDepth: 2 });
    const a3 = correctKeypointsRS(kp, cam, w, model, undefined, { velocity: vel, invDepth: [2], defaultInvDepth: 0 });
    expect(Array.from(a2)).toEqual(Array.from(a3));
  });

  it('device-frame round trip: the iPhone sweep (device +x edge → −x) is rows top→bottom at θ = 90 and columns right→left at θ = 0', () => {
    const dev = { sweepDevice: [-1, 0, 0] as [number, number, number], readoutMs: 22 };
    expect(rsModelFromDevice(dev, cameraFromDevice(90))).toEqual({ axis: 'rows', readoutMs: 22 });
    expect(rsModelFromDevice(dev, cameraFromDevice(0))).toEqual({ axis: 'cols', readoutMs: -22 });
    expect(rsModelFromDevice(dev, cameraFromDevice(180))).toEqual({ axis: 'cols', readoutMs: 22 });
    expect(rsModelFromDevice(dev, cameraFromDevice(270))).toEqual({ axis: 'rows', readoutMs: -22 });
    for (const th of [0, 90, 180, 270]) {
      const m = rsModelFromDevice(dev, cameraFromDevice(th))!;
      const back = rsModelToDevice(m, cameraFromDevice(th));
      expect(back.readoutMs).toBe(22);
      for (let k = 0; k < 3; k++) expect(back.sweepDevice[k]).toBeCloseTo(dev.sweepDevice[k], 12);
    }
  });
});

describe('estimateRollingShutter', () => {
  it('recovers axis, direction and readout time; keeps global shutter when there is none', () => {
    for (const truth of [
      { axis: 'rows', readoutMs: 22 },
      { axis: 'cols', readoutMs: -22 },
    ] as RollingShutterModel[]) {
      const est = estimateRollingShutter(scene(new Rng(7), truth));
      expect(est.accepted).toBe(true);
      expect(est.model.axis).toBe(truth.axis);
      expect(est.model.readoutMs).toBeGreaterThan(truth.readoutMs - 1.5);
      expect(est.model.readoutMs).toBeLessThan(truth.readoutMs + 1.5);
      expect(est.improvement).toBeGreaterThan(0.1);
    }
    const gs = estimateRollingShutter(scene(new Rng(8), { axis: 'rows', readoutMs: 0 }));
    expect(gs.accepted).toBe(false);
    expect(gs.model.readoutMs).toBe(0);
    expect(Math.abs(gs.best.readoutMs)).toBeLessThan(3);
  });
});

describe('RSBundleCost (short subset BA, points free)', () => {
  it('recovers the readout from a reconstruction built on rolling-shutter keypoints', async () => {
    const { generateScene, buildSyntheticDatabase } = await import('./synthetic');
    const { runIncrementalSfM } = await import('./mapper');
    const { RSBundleCost, searchRollingShutter } = await import('./rolling-shutter');
    const truth: RollingShutterModel = { axis: 'rows', readoutMs: 22 };
    const scene = generateScene({ kind: 'walk', numImages: 40, seed: 3, noisePx: 0.3, width: 960, height: 720, hfovDeg: 65, k1: 0, k2: 0, descriptors: false, outlierRatio: 0 });
    const c = scene.camera;
    const rng = new Rng(11);
    const omega = new Map<number, Float64Array>();
    scene.imageIds.forEach((id, k) => {
      // pans: every third image rotates fast
      const fast = k % 3 === 0;
      const w = Float64Array.of(rng.gaussian() * (fast ? 0.3 : 0.03), (fast ? 1.2 : 0.05) * (rng.next() < 0.5 ? -1 : 1), rng.gaussian() * (fast ? 0.2 : 0.02));
      omega.set(id, w);
      const pose = scene.poses[k];
      const kp = scene.features[k].keypoints;
      const fp = scene.featurePoint[k];
      const out = new Float64Array(2);
      for (let f = 0; f < fp.length; f++) {
        if (fp[f] < 0) continue;
        const X = [scene.points[3 * fp[f]], scene.points[3 * fp[f] + 1], scene.points[3 * fp[f] + 2]];
        const xc = mat3MulVec(pose.R, X);
        for (let a = 0; a < 3; a++) xc[a] += pose.t[a];
        let tau = 0;
        let ok = true;
        for (let it = 0; it < 8 && ok; it++) {
          const R = rotvecToMat3([-w[0] * tau, -w[1] * tau, -w[2] * tau]);
          ok = projectCam(c, mat3MulVec(R, xc), out);
          tau = rsLineTimeMs(truth, out[0], out[1], c.width, c.height) / 1000;
        }
        if (!ok) continue;
        kp[4 * f] = out[0] + 0.3 * rng.gaussian();
        kp[4 * f + 1] = out[1] + 0.3 * rng.gaussian();
      }
    });
    const original = new Map(scene.imageIds.map((id, k) => [id, scene.features[k].keypoints.slice()]));
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    const [rec] = await runIncrementalSfM(db, { intrinsics: { ...c } });
    expect(rec.numRegisteredImages).toBeGreaterThan(35);
    const fastIds = scene.imageIds.filter((_, k) => k % 3 === 0).filter((id) => rec.images.get(id)?.registered);
    const bc = new RSBundleCost(rec, fastIds, (id) => original.get(id) ?? null, (id) => omega.get(id) ?? null);
    const est = searchRollingShutter((m) => bc.cost(m), {}, { numImages: fastIds.length, numObservations: bc.numObservations });
    expect(est.accepted).toBe(true);
    expect(est.model.axis).toBe('rows');
    expect(est.model.readoutMs).toBeGreaterThan(19);
    expect(est.model.readoutMs).toBeLessThan(25);
  }, 60_000);
});

describe('RSBundleCost on a close-range orbit (translational term)', () => {
  it('finds the true readout with the camera velocities and a second pass; rotation-only finds a shorter, effective one', async () => {
    const { generateScene, buildSyntheticDatabase } = await import('./synthetic');
    const { runIncrementalSfM } = await import('./mapper');
    const { RSBundleCost, searchRollingShutter } = await import('./rolling-shutter');
    const { mat3Transpose } = await import('../math/mat3');
    const { mat3ToRotvec } = await import('../math/rotation');
    const { cameraVelocities, rsTranslations } = await import('../engine/refine-camera');
    const truth: RollingShutterModel = { axis: 'cols', readoutMs: -22 };
    const scene = generateScene({ kind: 'orbit', numImages: 36, seed: 7, noisePx: 0.2, k1: 0, k2: 0, descriptors: false, outlierRatio: 0 });
    const c = scene.camera;
    const dt = 0.12; // s between images: ≈ 1 rad/s around the object, ≈ 0.5 m/s sideways
    const timeOf = (id: number): number => 1000 * dt * scene.imageIds.indexOf(id);
    const n = scene.imageIds.length;
    const centre = (k: number): number[] => {
      const R = scene.poses[k].R,
        t = scene.poses[k].t;
      return [0, 1, 2].map((i) => -(R[i] * t[0] + R[3 + i] * t[1] + R[6 + i] * t[2]));
    };
    const omega = new Map<number, Float64Array>();
    scene.imageIds.forEach((id, k) => {
      const a = Math.max(0, k - 1),
        b = Math.min(n - 1, k + 1);
      const pose = scene.poses[k];
      // camera-frame ω: x(τ) = exp(−[ωτ]×) x(0) ⇔ R(τ) = exp(−[ωτ]×) R ⇒ ω = −log(R_b R_aᵀ) / Δt
      const rv = mat3ToRotvec(mat3Mul(scene.poses[b].R, mat3Transpose(scene.poses[a].R)));
      const w = Float64Array.from(rv, (x) => -x / ((b - a) * dt));
      omega.set(id, w);
      const Ca = centre(a),
        Cb = centre(b);
      const vc = mat3MulVec(
        pose.R,
        [0, 1, 2].map((i) => (Cb[i] - Ca[i]) / ((b - a) * dt)),
      );
      const kp = scene.features[k].keypoints;
      const fp = scene.featurePoint[k];
      const out = new Float64Array(2);
      const rng = new Rng(100 + k);
      for (let f = 0; f < fp.length; f++) {
        if (fp[f] < 0) continue;
        const X = [scene.points[3 * fp[f]], scene.points[3 * fp[f] + 1], scene.points[3 * fp[f] + 2]];
        const xc = mat3MulVec(pose.R, X);
        for (let i = 0; i < 3; i++) xc[i] += pose.t[i];
        let tau = 0,
          ok = true;
        for (let it = 0; it < 8 && ok; it++) {
          // RS camera at line time τ: rotated by −ωτ, moved by v τ
          const R = rotvecToMat3([-w[0] * tau, -w[1] * tau, -w[2] * tau]);
          ok = projectCam(c, mat3MulVec(R, [xc[0] - vc[0] * tau, xc[1] - vc[1] * tau, xc[2] - vc[2] * tau]), out);
          tau = rsLineTimeMs(truth, out[0], out[1], c.width, c.height) / 1000;
        }
        if (!ok) continue;
        kp[4 * f] = out[0] + 0.2 * rng.gaussian();
        kp[4 * f + 1] = out[1] + 0.2 * rng.gaussian();
      }
    });
    const original = new Map(scene.imageIds.map((id, k) => [id, scene.features[k].keypoints.slice()]));
    const build = async () => (await runIncrementalSfM(buildSyntheticDatabase(scene, { verify: false }).db, { intrinsics: { ...c } }))[0];
    const rec = await build();
    expect(rec.numRegisteredImages).toBeGreaterThan(32);
    const estimate = (r: typeof rec, withVel: boolean, around: RollingShutterModel | null = null) => {
      const vel = cameraVelocities(r, timeOf);
      const ids = r.registeredImageIds();
      const bc = new RSBundleCost(r, ids, (id) => original.get(id) ?? null, (id) => omega.get(id) ?? null, {
        maxObsPerImage: 40,
        maxTrack: 8,
        refineFocal: true,
        velocityOf: withVel ? (id) => vel.get(id) ?? null : undefined,
      });
      return searchRollingShutter((m) => bc.cost(m), { around }, { numImages: ids.length, numObservations: bc.numObservations });
    };
    const rot = estimate(rec, false);
    const full = estimate(rec, true);
    expect(full.accepted).toBe(true);
    expect(full.model.axis).toBe('cols');
    // the short BA starts from the reconstruction built on distorted keypoints (focal −2 %): both
    // under-estimate, rotation-only far more (its translation error cancels the yaw term)
    expect(full.model.readoutMs).toBeLessThan(-16);
    expect(rot.model.readoutMs).toBeGreaterThan(full.model.readoutMs + 3);
    // second pass on the reconstruction rebuilt from keypoints corrected with the first estimate
    const tr = rsTranslations(rec, timeOf);
    scene.features.forEach((fs, k) => {
      const id = scene.imageIds[k];
      correctKeypointsRS(original.get(id)!, c, omega.get(id)!, full.model, fs.keypoints, tr.get(id) ?? null);
    });
    const rec2 = await build();
    const pass2 = estimate(rec2, true, full.model);
    expect(pass2.accepted).toBe(true);
    expect(pass2.model.axis).toBe('cols');
    expect(Math.abs(pass2.model.readoutMs - truth.readoutMs)).toBeLessThan(1.5);
    expect(pass2.curve.length).toBeLessThan(full.curve.length); // narrowed search
  }, 180_000);
});

describe('rsCorrectPinholeImage', () => {
  it('each output pixel samples the input where the rolling-shutter camera saw that bearing', async () => {
    const { rsCorrectPinholeImage } = await import('./rolling-shutter');
    const W = 256,
      H = 192;
    const pin: CameraIntrinsics = { width: W, height: H, f: 200, cx: 127.5, cy: 95.5, k1: 0, k2: 0 };
    // input pixel values encode their own coordinates (R = u, G = v)
    const data = new Uint8ClampedArray(W * H * 4);
    for (let v = 0; v < H; v++) for (let u = 0; u < W; u++) data.set([u, v, 0, 255], 4 * (v * W + u));
    const w = [0.3, 1.2, -0.4];
    for (const model of [
      { axis: 'rows', readoutMs: 22 },
      { axis: 'cols', readoutMs: -30 },
    ] as RollingShutterModel[]) {
      const out = rsCorrectPinholeImage({ width: W, height: H, data }, pin, w, model);
      let maxErr = 0,
        maxShift = 0,
        n = 0;
      for (let v = 20; v < H - 20; v += 7)
        for (let u = 20; u < W - 20; u += 9) {
          const o = 4 * (v * W + u);
          if (out.data[o + 3] === 0) continue;
          // where does the RS camera see the bearing of (u, v)? (iterate the line time)
          const x0 = [(u - pin.cx) / pin.f, (v - pin.cy) / pin.f, 1];
          let su = u,
            sv = v;
          for (let it = 0; it < 8; it++) {
            const tau = (rsLineTimeMs(model, su, sv, W, H) / 1000);
            const R = rotvecToMat3([-w[0] * tau, -w[1] * tau, -w[2] * tau]);
            const p = mat3MulVec(R, x0);
            su = pin.f * (p[0] / p[2]) + pin.cx;
            sv = pin.f * (p[1] / p[2]) + pin.cy;
          }
          maxErr = Math.max(maxErr, Math.abs(out.data[o] - su), Math.abs(out.data[o + 1] - sv));
          maxShift = Math.max(maxShift, Math.hypot(su - u, sv - v));
          n++;
        }
      expect(n).toBeGreaterThan(300);
      expect(maxShift).toBeGreaterThan(2);
      expect(maxErr).toBeLessThan(0.75); // 8-bit rounding of the coordinate code
    }
  });
});

describe('rsCorrectPinholeImage with the translational term', () => {
  it('agrees with the keypoint correction for a fronto-parallel plane at the image depth (ν = v / Z)', async () => {
    const { rsCorrectPinholeImage } = await import('./rolling-shutter');
    const W = 256,
      H = 192;
    const pin: CameraIntrinsics = { width: W, height: H, f: 200, cx: 127.5, cy: 95.5, k1: 0, k2: 0 };
    const data = new Uint8ClampedArray(W * H * 4);
    for (let v = 0; v < H; v++) for (let u = 0; u < W; u++) data.set([u, v, 0, 255], 4 * (v * W + u));
    const w = [0.2, 0.9, -0.3];
    const vel = [0.5, -0.25, 0.15]; // m/s, camera frame
    const Z = 0.4;
    const nu = vel.map((x) => x / Z);
    for (const model of [
      { axis: 'rows', readoutMs: 30 },
      { axis: 'cols', readoutMs: -30 },
    ] as RollingShutterModel[]) {
      const out = rsCorrectPinholeImage({ width: W, height: H, data }, pin, w, model, nu);
      const rotOnly = rsCorrectPinholeImage({ width: W, height: H, data }, pin, w, model);
      let maxErr = 0,
        maxTrans = 0,
        n = 0;
      for (let v = 20; v < H - 20; v += 7)
        for (let u = 20; u < W - 20; u += 9) {
          const o = 4 * (v * W + u);
          if (out.data[o + 3] === 0 || rotOnly.data[o + 3] === 0) continue;
          // the input position this output pixel sampled = the rolling-shutter keypoint of that bearing
          const kp = Float32Array.of(out.data[o], out.data[o + 1], 1, 0);
          const c = correctKeypointsRS(kp, pin, w, model, undefined, { velocity: vel, invDepth: [1 / Z], defaultInvDepth: 0 });
          maxErr = Math.max(maxErr, Math.hypot(c[0] - u, c[1] - v));
          maxTrans = Math.max(maxTrans, Math.hypot(out.data[o] - rotOnly.data[o], out.data[o + 1] - rotOnly.data[o + 1]));
          n++;
        }
      expect(n).toBeGreaterThan(300);
      expect(maxTrans).toBeGreaterThan(2); // the translational term matters here
      expect(maxErr).toBeLessThan(0.9); // 8-bit coordinate code + first-order term
    }
  });
});
