import { describe, expect, it } from 'vitest';
import type { Mat3 } from '../core/types';
import { mat3Mul, mat3MulTranspose, mat3MulVec, mat3Transpose, mat3TransposeMul } from '../math/mat3';
import { rotationAngle, rotvecToMat3 } from '../math/rotation';
import { cameraFromDevice, uiUpInDevice, type ScreenAngle } from '../sensors/frame-orientation';
import { integrateGyro } from '../sensors/motion';
import { coverFovDeg, followViewRotation, FollowView, quarterTurned } from './follow-view';

const angle = (a: ArrayLike<number>, b: ArrayLike<number>): number => rotationAngle(mat3MulTranspose(a, b));

/** Phone attitude R_world_device and the engine's world (an arbitrary rotation W of it). */
const W = rotvecToMat3([0.3, -1.1, 0.4]);
const attitude = (t: number): Mat3 => mat3Mul(rotvecToMat3([0.2, 0.1, -0.3]), rotvecToMat3([0.4 * t, -0.25 * t, 0.15 * t]));
/** Engine pose of a keyframe at t: world (engine) → camera of session θ. */
const keyframeR = (t: number, sessionTheta: ScreenAngle): Mat3 => mat3MulTranspose(mat3MulTranspose(cameraFromDevice(sessionTheta), attitude(t)), W);

describe('followViewRotation', () => {
  it('keyframe + gyro = the phone now, in the screen frame of the current UI rotation', () => {
    for (const ts of [0, 90, 180, 270] as ScreenAngle[]) {
      for (const theta of [0, 90, 180, 270] as ScreenAngle[]) {
        const tk = 0.5,
          now = 1.7;
        const G = mat3TransposeMul(attitude(now), attitude(tk)); // R_{d(now)←d(tk)}
        const R = followViewRotation(keyframeR(tk, ts), cameraFromDevice(ts), G, theta);
        // expected: world(engine) → device now → screen frame (rear camera upright for θ)
        const expected = mat3Mul(cameraFromDevice(theta), mat3TransposeMul(attitude(now), mat3Transpose(W)));
        expect(angle(R, expected)).toBeLessThan(1e-9);
        // the view looks along the rear camera (−z of the device) and "up" on screen is the UI's up
        const dev = mat3TransposeMul(attitude(now), mat3Transpose(W)); // world → device
        const fwd = mat3MulVec(R, mat3MulVec(mat3Transpose(dev), [0, 0, -1]));
        expect(fwd[2]).toBeCloseTo(1, 9);
        const up = mat3MulVec(R, mat3MulVec(mat3Transpose(dev), uiUpInDevice(theta)));
        expect(up[1]).toBeCloseTo(-1, 9); // image y points down
      }
    }
  });

  it('matches the gyro integrator convention', () => {
    // constant body rate ω (rad/s): R_wd(t) = R0 · exp([ω t]×)
    const w = [0.6, -0.2, 0.9];
    const samples = Array.from({ length: 101 }, (_, i) => ({ t: i * 10, gyro: new Float64Array(w), accel: new Float64Array([0, 0, 0]) }));
    const G = integrateGyro(samples as never, 100, 700);
    const Rwd = (tms: number): Mat3 => rotvecToMat3(w.map((x) => (x * tms) / 1000));
    expect(angle(G, mat3TransposeMul(Rwd(700), Rwd(100)))).toBeLessThan(1e-6);
  });

  it('without gyro or engine frame: the keyframe rotation (turned for the UI rotation)', () => {
    const Rk = keyframeR(0, 0);
    expect(angle(followViewRotation(Rk, null, null, 90), Rk)).toBeLessThan(1e-12);
    expect(angle(followViewRotation(Rk, cameraFromDevice(0), null, 0), Rk)).toBeLessThan(1e-12);
    expect(angle(followViewRotation(Rk, cameraFromDevice(0), null, 90), Rk)).toBeCloseTo(Math.PI / 2, 9);
  });
});

describe('coverFovDeg / quarterTurned', () => {
  it('vertical field of view of an object-fit: cover image', () => {
    const intr = { width: 1440, height: 1920, f: 1500 };
    // portrait phone screen: the image height fills the screen, the sides are cropped
    expect(coverFovDeg(intr, false, 390, 844)).toBeCloseTo((2 * Math.atan(960 / 1500) * 180) / Math.PI, 9);
    // wider box than the image: the width fills it, the top and bottom are cropped
    const scale = 600 / 1440;
    expect(coverFovDeg(intr, false, 600, 400)).toBeCloseTo((2 * Math.atan(400 / scale / 2 / 1500) * 180) / Math.PI, 9);
    // shown a quarter turn off: width and height swap
    expect(coverFovDeg(intr, true, 844, 390)).toBeCloseTo((2 * Math.atan(390 / (844 / 1920) / 2 / 1500) * 180) / Math.PI, 9);
    expect(coverFovDeg({ width: 0, height: 1, f: 1 }, false, 1, 1)).toBeNaN();
  });

  it('frames shown a quarter turn off when the UI rotated against the session', () => {
    expect(quarterTurned(cameraFromDevice(0), 0)).toBe(false);
    expect(quarterTurned(cameraFromDevice(0), 90)).toBe(true);
    expect(quarterTurned(cameraFromDevice(90), 270)).toBe(false);
    expect(quarterTurned(cameraFromDevice(270), 180)).toBe(true);
    expect(quarterTurned(null, 90)).toBe(false);
  });
});

describe('FollowView', () => {
  const pose = (rv: number[], C: number[]) => {
    const R = rotvecToMat3(rv);
    const t = mat3MulVec(R, C);
    return { R, t: new Float64Array([-t[0], -t[1], -t[2]]) };
  };
  const centre = (p: { R: ArrayLike<number>; t: ArrayLike<number> }) => {
    const c = mat3MulVec(mat3Transpose(p.R), p.t);
    return [-c[0], -c[1], -c[2]];
  };

  it('a new anchor does not jump: the difference decays', () => {
    const f = new FollowView();
    expect(f.pose(0, null, 0)).toBeNull();
    f.setAnchor({ pose: pose([0.1, 0.2, 0], [0, 0, 0]), t: 0 });
    const a = f.pose(0, null, 0)!;
    f.setAnchor({ pose: pose([0.15, 0.2, 0], [0.3, 0, 0]), t: 500 });
    const b = f.pose(0, null, 0)!;
    expect(angle(a.R, b.R)).toBeLessThan(1e-9);
    expect(Math.hypot(...centre(b).map((x, i) => x - centre(a)[i]))).toBeLessThan(1e-9);
    const c = f.pose(2000, null, 0)!; // ≫ τ
    expect(angle(c.R, rotvecToMat3([0.15, 0.2, 0]))).toBeLessThan(1e-3);
    expect(centre(c)[0]).toBeCloseTo(0.3, 3);
    expect(f.anchorTime).toBe(500);
  });

  it('a re-initialised model (large jump) is shown at once', () => {
    const f = new FollowView();
    f.setAnchor({ pose: pose([0, 0, 0], [0, 0, 0]) });
    f.pose(0, null, 0);
    f.setAnchor({ pose: pose([0, 2, 0], [5, 0, 0]) });
    const b = f.pose(16, null, 0)!;
    expect(angle(b.R, rotvecToMat3([0, 2, 0]))).toBeLessThan(1e-9);
    expect(centre(b)[0]).toBeCloseTo(5, 9);
  });
});
