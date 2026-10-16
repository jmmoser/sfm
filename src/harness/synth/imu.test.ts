import { describe, expect, it } from 'vitest';
import { expSO3, m3Mul, m3T, m3Vec, rotationAngleBetween } from './mat';
import { GRAVITY, idealImu, simulateImu } from './imu';
import {
  cameraToWorldJet,
  createObjectTrajectory,
  createStaticTrajectory,
  createYardTrajectory,
  deviceAttitudeFromCamera,
} from './trajectory';
import { jconst, jmatSplit } from './jet';
import { rCamDevice } from './conventions';

const NOISE_FREE = { gyroNoise: 0, accelNoise: 0, gyroBias: [0, 0, 0] as [number, number, number], accelBias: [0, 0, 0] as [number, number, number] };

/** Camera-to-world rotation of a level camera with the given yaw (CCW from east). */
function levelCamera(yaw: number): Float64Array {
  return jmatSplit(cameraToWorldJet(jconst(yaw), jconst(0), jconst(0))).v;
}

describe('IMU sign conventions at rest', () => {
  it('flat face-up: accelG = (0, 0, +9.81), accel = 0, gyro = 0', () => {
    const traj = createStaticTrajectory([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1]);
    const s = simulateImu(traj, 0, 1, 0, NOISE_FREE);
    expect(s.length).toBe(61); // default 60 Hz (iOS DeviceMotion)
    for (const m of s) {
      expect(m.accelG![0]).toBeCloseTo(0, 12);
      expect(m.accelG![1]).toBeCloseTo(0, 12);
      expect(m.accelG![2]).toBeCloseTo(GRAVITY, 12);
      expect(Math.hypot(...m.accel!)).toBeLessThan(1e-12);
      expect(Math.hypot(...m.gyro)).toBeLessThan(1e-12);
    }
  });

  it('upright portrait (θ=0) camera looking north: device +y (top) is up → accelG ≈ (0, +9.81, 0)', () => {
    const R_wd = deviceAttitudeFromCamera(levelCamera(Math.PI / 2), 0);
    const m = idealImu(createStaticTrajectory(R_wd, [0, 0, 1.5], 0).evaluate(0), 0);
    expect(m.accelG![0]).toBeCloseTo(0, 12);
    expect(m.accelG![1]).toBeCloseTo(GRAVITY, 12);
    expect(m.accelG![2]).toBeCloseTo(0, 12);
    // screen faces the user: device +z points opposite to the viewing direction (south)
    expect(R_wd[5]).toBeCloseTo(-1, 12);
  });

  it('upright landscape (θ=90) camera: device +x is up → accelG ≈ (+9.81, 0, 0)', () => {
    const R_wd = deviceAttitudeFromCamera(levelCamera(0.3), 90);
    const m = idealImu(createStaticTrajectory(R_wd, [0, 0, 1.5], 90).evaluate(0), 0);
    expect(m.accelG![0]).toBeCloseTo(GRAVITY, 12);
    expect(m.accelG![1]).toBeCloseTo(0, 12);
    expect(m.accelG![2]).toBeCloseTo(0, 12);
  });

  it('R_cam_device matches ARCHITECTURE §2', () => {
    expect(Array.from(rCamDevice(0))).toEqual([1, 0, 0, 0, -1, 0, 0, 0, -1]);
    expect(Array.from(rCamDevice(90))).toEqual([0, -1, 0, -1, 0, 0, 0, 0, -1]);
  });
});

describe('IMU consistency with the trajectory', () => {
  for (const traj of [createObjectTrajectory(), createYardTrajectory()]) {
    it(`${traj.kind}: integrating the gyro reproduces the analytic attitude`, () => {
      const rate = 1000; // integrate densely so the integration error is negligible
      const s = simulateImu(traj, 2, 7, 0, { ...NOISE_FREE, rateHz: rate });
      let R = traj.evaluate(2).R_wd;
      const dt = 1 / rate;
      for (let k = 0; k + 1 < s.length; k++) {
        const a = s[k].gyro,
          b = s[k + 1].gyro;
        // trapezoid + second-order (coning) correction
        const w = [0, 1, 2].map((i) => 0.5 * (a[i] + b[i]) * dt);
        const c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]].map((x) => (x * dt * dt) / 12);
        R = m3Mul(R, expSO3([w[0] + c[0], w[1] + c[1], w[2] + c[2]]));
      }
      const err = rotationAngleBetween(R, traj.evaluate(7).R_wd);
      expect(err).toBeLessThan(1e-6);
    });

    it(`${traj.kind}: integrating R_wd·accel reproduces the analytic velocity; accelG − accel = R_wdᵀ(0,0,g)`, () => {
      const rate = 1000;
      const s = simulateImu(traj, 1, 6, 0, { ...NOISE_FREE, rateHz: rate });
      const v = Array.from(traj.evaluate(1).velocity);
      const dt = 1 / rate;
      let prev = m3Vec(traj.evaluate(1).R_wd, s[0].accel!);
      for (let k = 1; k < s.length; k++) {
        const st = traj.evaluate(1 + k / rate);
        const aw = m3Vec(st.R_wd, s[k].accel!);
        for (let i = 0; i < 3; i++) v[i] += 0.5 * (prev[i] + aw[i]) * dt;
        prev = aw;
        const g = m3Vec(m3T(st.R_wd), [0, 0, GRAVITY]);
        for (let i = 0; i < 3; i++) expect(s[k].accelG![i] - s[k].accel![i]).toBeCloseTo(g[i], 10);
      }
      const vt = traj.evaluate(6).velocity;
      expect(Math.hypot(v[0] - vt[0], v[1] - vt[1], v[2] - vt[2])).toBeLessThan(1e-4);
    });
  }

  it('noise and bias statistics match the configuration', () => {
    const traj = createYardTrajectory();
    const cfg = { gyroNoise: 0.004, accelNoise: 0.04, gyroBias: [0.01, -0.02, 0.005] as [number, number, number], accelBias: [0.1, 0.2, -0.3] as [number, number, number], seed: 3 };
    const noisy = simulateImu(traj, 0, 30, 0, { ...cfg, rateHz: 100 });
    const clean = simulateImu(traj, 0, 30, 0, { ...NOISE_FREE, rateHz: 100 });
    expect(noisy.length).toBe(3001);
    for (let i = 0; i < 3; i++) {
      const dg = noisy.map((m, k) => m.gyro[i] - clean[k].gyro[i]);
      const da = noisy.map((m, k) => m.accel![i] - clean[k].accel![i]);
      const mg = dg.reduce((a, b) => a + b, 0) / dg.length;
      const ma = da.reduce((a, b) => a + b, 0) / da.length;
      const sg = Math.sqrt(dg.reduce((a, b) => a + (b - mg) ** 2, 0) / dg.length);
      const sa = Math.sqrt(da.reduce((a, b) => a + (b - ma) ** 2, 0) / da.length);
      expect(mg).toBeCloseTo(cfg.gyroBias[i], 3);
      expect(ma).toBeCloseTo(cfg.accelBias[i], 2);
      expect(sg / cfg.gyroNoise).toBeGreaterThan(0.93);
      expect(sg / cfg.gyroNoise).toBeLessThan(1.07);
      expect(sa / cfg.accelNoise).toBeGreaterThan(0.93);
      expect(sa / cfg.accelNoise).toBeLessThan(1.07);
      // same accelerometer noise realisation on accel and accelG
      for (let k = 0; k < 50; k++) expect(noisy[k].accelG![i] - clean[k].accelG![i]).toBeCloseTo(da[k], 12);
    }
    // deterministic
    const again = simulateImu(traj, 0, 30, 0, { ...cfg, rateHz: 100 });
    expect(again[1234].gyro).toEqual(noisy[1234].gyro);
  });
});
