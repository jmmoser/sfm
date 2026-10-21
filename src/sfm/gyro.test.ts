import { describe, expect, it } from 'vitest';
import type { ImuSample, Mat3 } from '../core/types';
import { mat3Mul, mat3MulTranspose, mat3Transpose } from '../math/mat3';
import { Rng } from '../math/random';
import { rotationAngle, rotvecToMat3 } from '../math/rotation';
import { integrateGyro } from '../sensors/motion';
import { cameraFromDevice, orbitTrajectory, simulateImu, simulateKeyframes } from './extras-testutil';
import { checkPairRotation, checkPoseRotation, gyroConsistency, gyroToleranceRad, GyroReference, IMU_MAX_GAP_MS, type TimedRotation } from './gyro';
import { estimateTimeOffset } from './inertial';

const DEG = Math.PI / 180;

function setup(seed = 3, bias: [number, number, number] = [0.004, -0.003, 0.002]) {
  const rng = new Rng(seed);
  const traj = orbitTrajectory(rng, { duration: 20 });
  const Rcd = cameraFromDevice(0);
  const imu = simulateImu(traj, Rcd, rng, { rateHz: 60, gyroNoise: 0.002, gyroBias: bias, t0Ms: 5000 });
  const sim = simulateKeyframes(traj, Rcd, rng, { offsetMs: -25, t0Ms: 5000, rotNoise: 0.0003 });
  // world-to-camera rotations of the keyframes (R = R_wcᵀ)
  const kfs: TimedRotation[] = sim.keyframes.map((k) => ({ id: k.id, t: k.t, R: mat3Transpose(k.R_wc) }));
  return { traj, Rcd, imu, sim, kfs };
}

describe('GyroReference', () => {
  it('matches integrateGyro (no bias), grows incrementally and maps camera(ta) → camera(tb)', () => {
    const { imu, Rcd, kfs } = setup(1, [0, 0, 0]);
    const full = new GyroReference(imu, { R_cam_device: Rcd });
    const inc = new GyroReference(undefined, { R_cam_device: Rcd });
    // incremental sync in chunks gives the same integration
    for (let k = 0; k <= imu.length; k += 37) inc.sync(imu.slice(0, k));
    inc.sync(imu);
    expect(inc.numSamples).toBe(full.numSamples);
    for (const [ta, tb] of [
      [6000, 6400],
      [7000, 15000],
      [12000, 9000],
    ]) {
      const G = full.relativeDevice(ta, tb);
      const ref = integrateGyro(imu, ta, tb);
      expect(rotationAngle(mat3MulTranspose(G, ref))).toBeLessThan(2e-4);
      expect(rotationAngle(mat3MulTranspose(inc.relativeDevice(ta, tb), G))).toBeLessThan(1e-12);
    }
    // vision: R_b R_aᵀ ≈ gyro camera rotation at the TRUE times (t_cam + offset with offset −25 → IMU = t − 25)
    full.offsetMs = -25;
    let maxErr = 0;
    for (let i = 0; i + 3 < kfs.length; i += 3) {
      const a = kfs[i],
        b = kfs[i + 3];
      const G = full.relative(a.t, b.t)!;
      maxErr = Math.max(maxErr, rotationAngle(mat3MulTranspose(mat3MulTranspose(b.R, a.R), G)));
    }
    expect(maxErr).toBeLessThan(0.3 * DEG);
  });

  it('a sample inserted out of order triggers a rebuild; bias is subtracted', () => {
    const { imu, Rcd } = setup(2, [0.01, 0, 0]);
    const g = new GyroReference(imu.slice(0, 300), { R_cam_device: Rcd });
    const withHole = imu.filter((_, i) => i !== 100);
    const g2 = new GyroReference(withHole.slice(0, 299), { R_cam_device: Rcd });
    // insert the missing sample (as ImuLog does) and sync: same as the full integration
    g2.sync(imu.slice(0, 300));
    expect(rotationAngle(mat3MulTranspose(g2.relativeDevice(5500, 9500), g.relativeDevice(5500, 9500)))).toBeLessThan(1e-12);
    // bias: same as integrating bias-free samples
    const nb = new GyroReference(imu, { R_cam_device: Rcd, bias: [0.01, 0, 0] });
    const unbiased = imu.map((s) => ({ ...s, gyro: [s.gyro[0] - 0.01, s.gyro[1], s.gyro[2]] as [number, number, number] }));
    const ref = integrateGyro(unbiased, 6000, 16000);
    expect(rotationAngle(mat3MulTranspose(nb.relativeDevice(6000, 16000), ref))).toBeLessThan(2e-4);
    const raw = integrateGyro(imu, 6000, 7500);
    expect(rotationAngle(mat3MulTranspose(nb.relativeDevice(6000, 7500), raw))).toBeGreaterThan(0.008);
    // setBias re-integrates
    nb.setBias(null, imu);
    expect(rotationAngle(mat3MulTranspose(nb.relativeDevice(6000, 7500), raw))).toBeLessThan(2e-4);
  });

  it('angular velocity in the camera frame: a static direction moves as ẋ = −ω × x', () => {
    const { imu, Rcd } = setup(4, [0, 0, 0]);
    const g = new GyroReference(imu, { R_cam_device: Rcd });
    const t = 11000;
    const w = g.angularVelocityCam(t, 10)!;
    const R = g.relative(t, t + 5)!; // camera(t) → camera(t + 5 ms)
    const x = [0.1, -0.2, 1];
    const y = [R[0] * x[0] + R[1] * x[1] + R[2] * x[2], R[3] * x[0] + R[4] * x[1] + R[5] * x[2], R[6] * x[0] + R[7] * x[1] + R[8] * x[2]];
    const dx = [-(w[1] * x[2] - w[2] * x[1]) * 0.005, -(w[2] * x[0] - w[0] * x[2]) * 0.005, -(w[0] * x[1] - w[1] * x[0]) * 0.005];
    for (let k = 0; k < 3; k++) expect(y[k] - x[k]).toBeCloseTo(dx[k], 4);
  });
});

describe('gyro checks', () => {
  it('pairs: correct rotations pass, a wrong one (repeated structure) fails; tolerance grows with Δt', () => {
    const { imu, Rcd, kfs } = setup(5);
    const g = new GyroReference(imu, { R_cam_device: Rcd, offsetMs: -25 });
    let bad = 0;
    for (let i = 0; i + 1 < kfs.length; i++) {
      const c = checkPairRotation(g, kfs[i].t, kfs[i + 1].t, mat3MulTranspose(kfs[i + 1].R, kfs[i].R))!;
      if (!c.ok) bad++;
    }
    expect(bad).toBe(0);
    // loop closure 15 s apart with an uncorrected bias (≈ 0.0054 rad/s ⇒ ≈ 4.6°) still passes
    const a = kfs[2],
      b = kfs.find((k) => k.t - a.t > 15000)!;
    const c = checkPairRotation(g, a.t, b.t, mat3MulTranspose(b.R, a.R))!;
    expect(c.ok).toBe(true);
    expect(c.tolRad).toBeGreaterThan(gyroToleranceRad(0, 0));
    // a rotation 40° off fails
    const wrong = mat3Mul(rotvecToMat3([0, 40 * DEG, 0]), mat3MulTranspose(b.R, a.R));
    expect(checkPairRotation(g, a.t, b.t, wrong)!.ok).toBe(false);
    // outside the IMU coverage: no verdict
    expect(checkPairRotation(g, 0, 100, wrong)).toBeNull();
  });

  it('pose: majority vote over the nearest registered images; a misregistered neighbour is outvoted', () => {
    const { imu, Rcd, kfs } = setup(6);
    const g = new GyroReference(imu, { R_cam_device: Rcd, offsetMs: -25 });
    const target = kfs[20];
    const others = kfs.filter((k) => k.id !== target.id);
    expect(checkPoseRotation(g, target.t, target.R, others)!.ok).toBe(true);
    const wrongR: Mat3 = mat3Mul(rotvecToMat3([0.9, 0, 0]), target.R);
    const c = checkPoseRotation(g, target.t, wrongR, others)!;
    expect(c.ok).toBe(false);
    expect(c.disagree).toBe(3);
    // one wrong neighbour does not block a correct registration
    const corrupted = others.map((k) => (k.id === kfs[19].id ? { ...k, R: mat3Mul(rotvecToMat3([0, 1.2, 0]), k.R) } : k));
    const c2 = checkPoseRotation(g, target.t, target.R, corrupted)!;
    expect(c2.ok).toBe(true);
    expect(c2.disagree).toBe(1);
  });

  it('model consistency: an isolated outlier image and a fragment break are told apart', () => {
    const { imu, Rcd, kfs } = setup(7);
    const g = new GyroReference(imu, { R_cam_device: Rcd, offsetMs: -25 });
    expect(gyroConsistency(kfs, g)).toMatchObject({ breaks: [], outliers: [], breaksWithoutOutliers: 0 });
    // one misplaced image
    const out = kfs.map((k) => (k.id === kfs[10].id ? { ...k, R: mat3Mul(rotvecToMat3([0, 0, 1.5]), k.R) } : k));
    const c = gyroConsistency(out, g);
    expect(c.outliers).toEqual([kfs[10].id]);
    expect(c.breaks.length).toBe(2);
    expect(c.breaksWithoutOutliers).toBe(0);
    // two rigid fragments joined by a wrong rotation (every image after index 30 rotated in the world)
    const Q = rotvecToMat3([0, 1.0, 0.3]);
    const frag = kfs.map((k, i) => (i > 30 ? { ...k, R: mat3MulTranspose(k.R, Q) } : k));
    const f = gyroConsistency(frag, g);
    expect(f.outliers).toEqual([]);
    expect(f.breaks.map((b) => b.b)).toEqual([kfs[31].id]);
    expect(f.breaksWithoutOutliers).toBe(1);
  });

  it('estimateTimeOffset bias convention matches GyroReference (bias subtracted)', () => {
    const bias: [number, number, number] = [0.006, -0.004, 0.003];
    const { imu, Rcd, sim } = setup(8, bias);
    const r = estimateTimeOffset(sim.keyframes, imu, {})!;
    expect(r.offsetMs).toBeCloseTo(-25, 0);
    for (let k = 0; k < 3; k++) expect(r.gyroBias[k]).toBeCloseTo(bias[k], 3);
    const g = new GyroReference(imu, { R_cam_device: Rcd, offsetMs: r.offsetMs, bias: r.gyroBias });
    const kfs = sim.keyframes.map((k) => ({ t: k.t, R: mat3Transpose(k.R_wc) }));
    const a = kfs[1],
      b = kfs[kfs.length - 2];
    const err = rotationAngle(mat3MulTranspose(mat3MulTranspose(b.R, a.R), g.relative(a.t, b.t)!));
    expect(err).toBeLessThan(0.3 * DEG);
  });
});

describe('IMU gaps', () => {
  it('no rotation (so no gyro verdict) across a gap of > IMU_MAX_GAP_MS; within a segment unchanged', () => {
    const { imu, Rcd, kfs } = setup(9);
    // the page was hidden 12–15 s (no DeviceMotion events): the gap's rotation is unknown
    const gapped = imu.filter((s) => s.t < 12000 || s.t > 15000);
    const full = new GyroReference(imu, { R_cam_device: Rcd, offsetMs: -25 });
    const g = new GyroReference(gapped, { R_cam_device: Rcd, offsetMs: -25 });
    const pre = kfs.filter((k) => k.t < 11900),
      post = kfs.filter((k) => k.t > 15100);
    const a = pre[pre.length - 1],
      b = post[0];
    const Rba = mat3MulTranspose(b.R, a.R);
    expect(checkPairRotation(full, a.t, b.t, Rba)!.ok).toBe(true);
    expect(g.covers(a.t, b.t)).toBe(false);
    expect(g.relative(a.t, b.t)).toBeNull();
    expect(checkPairRotation(g, a.t, b.t, Rba)).toBeNull();
    expect(checkPoseRotation(g, b.t, b.R, pre)).toBeNull();
    expect(checkPoseRotation(g, b.t, b.R, [...pre, ...post.slice(1)])!.ok).toBe(true);
    // a model spanning the gap: only same-segment pairs are checked
    const c = gyroConsistency([...pre, ...post], g);
    expect(c.breaks).toEqual([]);
    expect(c.pairs).toBe(pre.length + post.length - 2);
    expect(gyroConsistency([...pre, ...post], full).pairs).toBe(c.pairs + 1);
    // within a segment: covered and the same as without the gap
    for (const [x, y] of [
      [pre[0], pre[pre.length - 1]],
      [post[0], post[post.length - 1]],
    ]) {
      expect(rotationAngle(mat3MulTranspose(g.relative(x.t, y.t)!, full.relative(x.t, y.t)!))).toBeLessThan(1e-9);
    }
    expect(g.angularVelocityCam(13500)).toBeNull();
    expect(g.angularVelocityCam(post[0].t)).not.toBeNull();
    // gaps up to IMU_MAX_GAP_MS are still integrated
    const short = new GyroReference(imu.filter((s) => s.t < 12000 || s.t > 12000 + IMU_MAX_GAP_MS - 50), { R_cam_device: Rcd, offsetMs: -25 });
    expect(short.covers(11000, 13000)).toBe(true);
    // found when the samples arrive incrementally; kept by clone; cleared by a rebuild without it
    const inc = new GyroReference(gapped.filter((s) => s.t < 12000), { R_cam_device: Rcd, offsetMs: -25 });
    expect(inc.covers(a.t, 11900)).toBe(true);
    inc.sync(gapped);
    expect(inc.covers(a.t, b.t)).toBe(false);
    expect(inc.clone().covers(a.t, b.t)).toBe(false);
    inc.setBias([0.001, 0, 0], gapped);
    expect(inc.covers(a.t, b.t)).toBe(false);
    inc.sync(imu); // the missing samples inserted: rebuilt
    expect(inc.covers(a.t, b.t)).toBe(true);
  });

  it('a time within the slack of a gap end holds ω of the sample after it (not the step interpolated across the gap)', () => {
    // hand-held pan rates (≈ 1 rad/s) at both ends of a 15 s gap
    const imu: ImuSample[] = [];
    for (let t = 5000; t < 40000; t += 1000 / 60) {
      const s = t / 1000;
      imu.push({ t, gyro: [0.8 * Math.sin(s / 0.7), 0.8 * Math.cos(s / 1.1), 0.8 * Math.sin(s / 1.9)], accel: null, accelG: null });
    }
    const full = new GyroReference(imu);
    const gapped = imu.filter((s) => s.t < 12000 || s.t > 27000);
    const g = new GyroReference(gapped);
    const e = gapped.find((s) => s.t > 27000)!.t,
      s0 = gapped.filter((s) => s.t < 12000).pop()!.t;
    // an image up to the slack (60 ms) before the first sample after the gap, or after the last one before it
    for (const [ta, tb] of [
      [e - 10, e + 800],
      [e - 55, e + 800],
      [s0 - 800, s0 + 55],
    ]) {
      expect(rotationAngle(mat3MulTranspose(g.relative(ta, tb)!, full.relative(ta, tb)!))).toBeLessThan(0.5 * DEG);
    }
    const w = g.angularVelocityCam(e + 3)!,
      wf = full.angularVelocityCam(e + 3)!;
    expect(Math.hypot(w[0] - wf[0], w[1] - wf[1], w[2] - wf[2])).toBeLessThan(0.05);
  });
});
