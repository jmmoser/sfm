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
