import { describe, expect, it } from 'vitest';
import type { ImuSample } from '../core/types';
import { mat3Mul, mat3MulVec, mat3TransposeMul } from '../math/mat3';
import { Rng } from '../math/random';
import { rotationAngle, rotvecToMat3 } from '../math/rotation';
import {
  angleDeg,
  cameraFromDevice,
  orbitTrajectory,
  simulateImu,
  simulateKeyframes,
  slowPanTrajectory,
  walkTrajectory,
  type SimTrajectory,
} from './extras-testutil';
import {
  AttitudeInterpolator,
  composeGravityAlignment,
  estimateGravity,
  estimateInertialAlignment,
  estimateMetricScale,
  estimateTimeOffset,
  filtfiltBandpass,
  GyroIntegrator,
  SplineSecondDerivative,
} from './inertial';

const LOG = !!(globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env.SFM_EXTRAS_LOG;

function scenario(kind: 'orbit' | 'walk', seed: number, offsetMs: number, imuOpts: Parameters<typeof simulateImu>[3] = {}) {
  const rng = new Rng(seed);
  const traj: SimTrajectory = kind === 'orbit' ? orbitTrajectory(rng) : walkTrajectory(rng);
  const Rcd = cameraFromDevice(kind === 'orbit' ? 0 : 90);
  const imu = simulateImu(traj, Rcd, rng, imuOpts);
  const kf = simulateKeyframes(traj, Rcd, rng, {
    offsetMs,
    s: rng.uniform(0.2, 5),
    posNoise: kind === 'orbit' ? 0.0005 : 0.002,
    rotNoise: kind === 'orbit' ? 0.0005 : 0.0005,
  });
  return { traj, Rcd, imu, kf };
}

describe('GyroIntegrator / AttitudeInterpolator', () => {
  it('integrates a constant rate exactly and matches a smooth trajectory', () => {
    const w = [0.3, -0.7, 1.1];
    const samples: ImuSample[] = [];
    for (let i = 0; i <= 200; i++) samples.push({ t: 1000 + i * 10 + (i % 3) * 0.7, gyro: [w[0], w[1], w[2]], accel: null, accelG: null });
    const g = new GyroIntegrator(samples);
    const R = g.relativeRotation(1013.3, 2777.7);
    const dt = (2777.7 - 1013.3) / 1000;
    const Rt = rotvecToMat3([w[0] * dt, w[1] * dt, w[2] * dt]);
    expect(rotationAngle(mat3TransposeMul(Rt, R))).toBeLessThan(1e-12);
    // Outside the range (held rate)
    const R2 = g.relativeRotation(900, 3100);
    const Rt2 = rotvecToMat3([w[0] * 2.2, w[1] * 2.2, w[2] * 2.2]);
    expect(rotationAngle(mat3TransposeMul(Rt2, R2))).toBeLessThan(1e-12);

    // Noise-free orbit: gyro integration over 0.5 s vs truth.
    const rng = new Rng(3);
    const traj = orbitTrajectory(rng);
    const Rcd = cameraFromDevice(0);
    const imu = simulateImu(traj, Rcd, rng, { gyroNoise: 0, gyroBias: [0, 0, 0], accelNoise: 0, jitterMs: 1 });
    const gi = new GyroIntegrator(imu);
    let maxErr = 0;
    for (let k = 0; k < 20; k++) {
      const ta = 1 + k * 0.8,
        tb = ta + 0.5;
      const A = new Float64Array(9),
        B = new Float64Array(9);
      traj.Rwc(ta, A);
      traj.Rwc(tb, B);
      const truth = mat3TransposeMul(mat3Mul(A, Rcd), mat3Mul(B, Rcd));
      const est = gi.relativeRotation(5000 + 1000 * ta, 5000 + 1000 * tb);
      maxErr = Math.max(maxErr, rotationAngle(mat3TransposeMul(truth, est)));
    }
    if (LOG) console.log('gyro integration max err (rad)', maxErr);
    expect(maxErr).toBeLessThan(1e-4); // linear-ω interpolation of 3 Hz jitter at 100 Hz
  });

  it('vision-anchored attitude is exact at keyframes and accurate in between despite gyro bias', () => {
    const rng = new Rng(11);
    const traj = orbitTrajectory(rng);
    const Rcd = cameraFromDevice(0);
    const imu = simulateImu(traj, Rcd, rng, { gyroBias: [0.02, -0.01, 0.015] });
    const kf = simulateKeyframes(traj, Rcd, rng, { offsetMs: 25, R: rotvecToMat3([0, 0, 0]), rotNoise: 0 });
    const att = new AttitudeInterpolator(kf.keyframes, new GyroIntegrator(imu), 25);
    const attB = new AttitudeInterpolator(kf.keyframes, new GyroIntegrator(imu, [0.02, -0.01, 0.015]), 25);
    const R2 = new Float64Array(9);
    let maxErrB = 0;
    const R = new Float64Array(9);
    for (const k of kf.keyframes.slice(0, 10)) {
      att.rotationAt(k.t + 25, R);
      expect(rotationAngle(mat3TransposeMul(mat3Mul(k.R_wc, Rcd), R))).toBeLessThan(1e-9);
    }
    let maxErr = 0;
    for (let i = 0; i < 200; i++) {
      const t = 1 + i * 0.09;
      const A = new Float64Array(9);
      traj.Rwc(t, A);
      att.rotationAt(5000 + 1000 * t, R);
      maxErr = Math.max(maxErr, rotationAngle(mat3TransposeMul(mat3Mul(A, Rcd), R)));
      attB.rotationAt(5000 + 1000 * t, R2);
      maxErrB = Math.max(maxErrB, rotationAngle(mat3TransposeMul(mat3Mul(A, Rcd), R2)));
    }
    if (LOG) console.log('attitude interp max err (deg)', (maxErr * 180) / Math.PI, 'bias-corrected', (maxErrB * 180) / Math.PI);
    expect(maxErr).toBeLessThan(0.0025); // 0.14°: uncorrected 0.02 rad/s bias on a rotating device
    expect(maxErrB).toBeLessThan(0.0008); // gyro noise random walk + linear-ω interpolation
  });
});

describe('signal helpers', () => {
  it('spline second derivative converges O(h²) in the interior; heavy smoothing → straight line', () => {
    const f = (s: number) => Math.sin(2 * Math.PI * 0.3 * s) + 0.1 * s * s;
    const d2 = (s: number) => -((2 * Math.PI * 0.3) ** 2) * Math.sin(2 * Math.PI * 0.3 * s) + 0.2;
    const grid = new Float64Array(200);
    for (let i = 0; i < 200; i++) grid[i] = 5 + (i * 20) / 199;
    const errAt = (h: number, jitter: boolean): number => {
      const rng = new Rng(5);
      const ts: number[] = [];
      for (let x = 0; x <= 30; x += h * (jitter ? rng.uniform(0.7, 1.3) : 1)) ts.push(x);
      const t = Float64Array.from(ts);
      const out = new SplineSecondDerivative(t, 0, grid).apply(Float64Array.from(t, f));
      let e = 0;
      for (let i = 0; i < 200; i++) e = Math.max(e, Math.abs(out[i] - d2(grid[i])));
      return e;
    };
    const e1 = errAt(0.4, false),
      e2 = errAt(0.2, false);
    expect(e1).toBeLessThan(0.2); // |f''| ≤ 3.75
    expect(e1 / e2).toBeGreaterThan(3); // second order
    expect(errAt(0.35, true)).toBeLessThan(0.25); // irregular knots
    const t = Float64Array.from({ length: 60 }, (_, i) => i * 0.37 + 0.05 * Math.sin(i));
    const opS = new SplineSecondDerivative(t, 1e6, grid);
    const o2 = opS.apply(Float64Array.from(t, (s) => 2 + 0.5 * s));
    for (let i = 0; i < 200; i++) expect(Math.abs(o2[i])).toBeLessThan(1e-9);
  });

  it('zero-phase band-pass keeps in-band sinusoids and removes DC / slow drift', () => {
    const fs = 20,
      n = 1200;
    const x = new Float64Array(n),
      inb = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const t = i / fs;
      inb[i] = Math.sin(2 * Math.PI * 1 * t + 0.3);
      x[i] = inb[i] + 5 + 2 * Math.sin(2 * Math.PI * 0.03 * t) + 0.5 * Math.sin(2 * Math.PI * 8 * t);
    }
    const y = filtfiltBandpass(x, fs, 0.3, 3);
    let err = 0;
    for (let i = 200; i < n - 200; i++) err = Math.max(err, Math.abs(y[i] - 0.98 * inb[i]));
    expect(err).toBeLessThan(0.03);
  });
});

describe('estimateTimeOffset', () => {
  for (const [kind, seed, off] of [
    ['orbit', 1, -37],
    ['orbit', 2, 180],
    ['walk', 3, 42],
    ['walk', 4, -120],
  ] as const) {
    it(`${kind}: recovers ${off} ms within 5 ms`, () => {
      const { imu, kf } = scenario(kind, seed, off);
      const t0 = performance.now();
      const res = estimateTimeOffset(kf.keyframes, imu)!;
      const ms = performance.now() - t0;
      if (LOG)
        console.log(kind, 'offset', off, '→', res.offsetMs.toFixed(2), '±', res.stdMs.toFixed(2), 'q', res.quality.toFixed(2), 'bias', Array.from(res.gyroBias).map((v) => v.toFixed(4)), `${ms.toFixed(0)} ms`, res.numPairs);
      expect(res).not.toBeNull();
      expect(Math.abs(res.offsetMs - off)).toBeLessThan(5);
      expect(res.reliable).toBe(true);
      expect(Math.abs(res.offsetMs - off)).toBeLessThan(4 * res.stdMs + 1);
    });
  }

  it('flags a featureless (constant-rate) motion as unreliable', () => {
    const traj = slowPanTrajectory(15);
    const Rcd = cameraFromDevice(0);
    const rng = new Rng(9);
    const imu = simulateImu(traj, Rcd, rng);
    const kf = simulateKeyframes(traj, Rcd, rng, { offsetMs: 30 });
    const res = estimateTimeOffset(kf.keyframes, imu);
    if (LOG && res) console.log('slow pan offset', res.offsetMs, res.stdMs, res.quality, res.reliable);
    expect(res === null || !res.reliable).toBe(true);
  });
});

describe('estimateGravity', () => {
  for (const [kind, seed] of [
    ['orbit', 21],
    ['walk', 22],
  ] as const) {
    it(`${kind}: gravity within 1°`, () => {
      const { imu, kf } = scenario(kind, seed, -30);
      const g = estimateGravity(kf.keyframes, imu, { offsetMs: -30 })!;
      const err = angleDeg(g.down, kf.downRec);
      if (LOG) console.log(kind, 'gravity err deg', err.toFixed(3), 'std deg', ((g.stdRad * 180) / Math.PI).toFixed(3), 'mag', g.magnitude.toFixed(3));
      expect(err).toBeLessThan(1);
      expect(g.reliable).toBe(true);
      const R = composeGravityAlignment(g.down);
      const z = mat3MulVec(R, g.down);
      expect(Math.abs(z[2] + 1)).toBeLessThan(1e-12);
    });
  }

  it('falls back to per-keyframe gravityCam', () => {
    const { kf } = scenario('orbit', 23, 0);
    for (const k of kf.keyframes) {
      // gravity DOWN in camera frame = R_wcᵀ down_rec (+ small noise)
      const g = mat3TransposeMul(k.R_wc, Float64Array.of(kf.downRec[0], kf.downRec[1], kf.downRec[2]).map((v) => v));
      void g;
      const d = new Float64Array(3);
      const R = k.R_wc;
      for (let i = 0; i < 3; i++) d[i] = R[i] * kf.downRec[0] + R[3 + i] * kf.downRec[1] + R[6 + i] * kf.downRec[2];
      k.gravityCam = d;
    }
    const g = estimateGravity(kf.keyframes, [])!;
    expect(g.source).toBe('gravityCam');
    expect(angleDeg(g.down, kf.downRec)).toBeLessThan(0.2);
  });
});

describe('estimateMetricScale', () => {
  const cases = [
    ['orbit', 31, 0.03, 'accel'],
    ['orbit', 32, 0.03, 'accelG'],
    ['orbit', 33, 0.03, 'accel'],
    ['walk', 34, 0.05, 'accel'],
    ['walk', 35, 0.05, 'accelG'],
    ['walk', 36, 0.05, 'accel'],
  ] as const;
  for (const [kind, seed, tol, src] of cases) {
    it(`${kind} (${src}, seed ${seed}): scale within ${tol * 100} %`, () => {
      const off = kind === 'orbit' ? -37 : 55;
      const { imu, kf } = scenario(kind, seed, off, src === 'accelG' ? { withAccel: false } : {});
      const t0 = performance.now();
      const al = estimateInertialAlignment(kf.keyframes, imu);
      const ms = performance.now() - t0;
      const sc = al.scale!;
      expect(sc).not.toBeNull();
      const truth = 1 / kf.s;
      const err = sc.scale / truth - 1;
      if (LOG)
        console.log(
          kind,
          src,
          'scale err %',
          (100 * err).toFixed(2),
          '± %',
          (100 * sc.relUncertainty).toFixed(2),
          'exc',
          sc.excitationRms.toFixed(3),
          'R2',
          sc.r2.toFixed(3),
          'neff',
          sc.numEffective.toFixed(0),
          'rows',
          sc.numRows,
          'bias',
          Array.from(sc.biasWorld).map((v) => v.toFixed(3)),
          'offset',
          al.offsetMs.toFixed(1),
          `${ms.toFixed(0)} ms`,
        );
      expect(Math.abs(err)).toBeLessThan(tol);
      expect(sc.reliable).toBe(true);
      expect(Math.abs(err)).toBeLessThan(4 * sc.relUncertainty + 0.005);
      expect(Math.abs(al.offsetMs - off)).toBeLessThan(5);
    });
  }

  it('device-bias option stays accurate', () => {
    const { imu, kf } = scenario('orbit', 37, 0, { accelBias: [0.06, -0.05, 0.04] });
    const res = estimateMetricScale(kf.keyframes, imu, { estimateDeviceBias: true })!;
    const err = res.scale * kf.s - 1;
    if (LOG) console.log('device bias', (100 * err).toFixed(2), res.biasDevice && Array.from(res.biasDevice).map((v) => v.toFixed(3)));
    expect(Math.abs(err)).toBeLessThan(0.03);
  });

  it('perfectly regular orbit without jitter: no failure, honest result', () => {
    const rng = new Rng(43);
    const traj = orbitTrajectory(rng, { regular: true, posJitterMm: [0, 0], rotJitterDeg: [0, 0] });
    const Rcd = cameraFromDevice(0);
    const imu = simulateImu(traj, Rcd, rng);
    const kf = simulateKeyframes(traj, Rcd, rng, { offsetMs: 0, s: 2 });
    const res = estimateMetricScale(kf.keyframes, imu, { offsetMs: 0 });
    expect(res).not.toBeNull();
    if (LOG) console.log('regular orbit', (100 * (res!.scale * 2 - 1)).toFixed(2), '±', (100 * res!.relUncertainty).toFixed(2), res!.reason);
    // centripetal acceleration is constant in the device frame ⇒ degenerate with the device bias
    expect(!res!.reliable || Math.abs(res!.scale * 2 - 1) < 3 * res!.relUncertainty + 0.01).toBe(true);
  });

  it('flags insufficient excitation', () => {
    const traj = slowPanTrajectory(15);
    const Rcd = cameraFromDevice(0);
    const rng = new Rng(41);
    const imu = simulateImu(traj, Rcd, rng);
    const kf = simulateKeyframes(traj, Rcd, rng, { offsetMs: 0 });
    const res = estimateMetricScale(kf.keyframes, imu, { offsetMs: 0 });
    if (LOG && res) console.log('slow pan', res.scale * kf.s, res.relUncertainty, res.excitationRms, res.reason);
    expect(res === null || !res.reliable).toBe(true);
    if (res) expect(res.reason).toMatch(/excitation/);
  });

  it('returns null without accelerations / gravity', () => {
    const { imu, kf } = scenario('orbit', 42, 0, { withAccel: false });
    expect(estimateMetricScale(kf.keyframes, imu, { offsetMs: 0, source: 'accelG' })).toBeNull();
    const none = imu.map((s) => ({ ...s, accel: null, accelG: null }));
    expect(estimateMetricScale(kf.keyframes, none, { offsetMs: 0, gravity: { down: kf.downRec } })).toBeNull();
  });
});
