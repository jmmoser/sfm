import { describe, expect, it } from 'vitest';
import type { GeoFix, ImuSample, Mat3 } from '../core/types';
import { mat3Mul, mat3MulVec, mat3Transpose } from '../math/mat3';
import { Rng } from '../math/random';
import { rotationAngle } from '../math/rotation';
import { cameraFromDevice, orbitTrajectory, simulateGps, simulateImu, simulateKeyframes, walkTrajectory, type SimKeyframes } from './extras-testutil';
import { computeWorldAlignment, transformPose, worldKeyframeFromFrame, type WorldKeyframe } from './world';

const LOG = !!(globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env.SFM_EXTRAS_LOG;
const DEG = Math.PI / 180;

function toWorldKeyframes(sim: SimKeyframes, extra?: (i: number, k: WorldKeyframe) => void): WorldKeyframe[] {
  return sim.keyframes.map((k, i) => {
    const R = mat3Transpose(k.R_wc);
    const t = mat3MulVec(R, k.C);
    for (let j = 0; j < 3; j++) t[j] = -t[j];
    const w: WorldKeyframe = { id: k.id, t: k.t, pose: { R, t }, R_cam_device: k.R_cam_device };
    extra?.(i, w);
    return w;
  });
}

/** Camera centres after applying the alignment (X' = s R X + t). */
function alignedCentres(kfs: WorldKeyframe[], s: number, R: Mat3, t: ArrayLike<number>): Float64Array {
  const out = new Float64Array(3 * kfs.length);
  kfs.forEach((k, i) => {
    const p = transformPose(k.pose, s, R, t);
    const C = mat3MulVec(mat3Transpose(p.R), p.t);
    out.set([-C[0], -C[1], -C[2]], 3 * i);
  });
  return out;
}

/** Max relative error of pairwise distances and max |Δz| error after removing the mean. */
function shapeErrors(est: Float64Array, truth: Float64Array) {
  const n = est.length / 3;
  let maxRel = 0;
  for (let i = 0; i < n; i += 3)
    for (let j = i + 5; j < n; j += 7) {
      const de = Math.hypot(est[3 * i] - est[3 * j], est[3 * i + 1] - est[3 * j + 1], est[3 * i + 2] - est[3 * j + 2]);
      const dt = Math.hypot(truth[3 * i] - truth[3 * j], truth[3 * i + 1] - truth[3 * j + 1], truth[3 * i + 2] - truth[3 * j + 2]);
      if (dt > 0.1) maxRel = Math.max(maxRel, Math.abs(de / dt - 1));
    }
  let mz = 0,
    tz = 0;
  for (let i = 0; i < n; i++) {
    mz += est[3 * i + 2];
    tz += truth[3 * i + 2];
  }
  mz /= n;
  tz /= n;
  let maxDz = 0;
  for (let i = 0; i < n; i++) maxDz = Math.max(maxDz, Math.abs(est[3 * i + 2] - mz - (truth[3 * i + 2] - tz)));
  return { maxRel, maxDz };
}

describe('computeWorldAlignment', () => {
  it('orbit + IMU: metric, gravity-aligned, first camera at the origin', () => {
    const rng = new Rng(1);
    const traj = orbitTrajectory(rng);
    const Rcd = cameraFromDevice(0);
    const imu = simulateImu(traj, Rcd, rng);
    const sim = simulateKeyframes(traj, Rcd, rng, { offsetMs: -40, s: 3.3 });
    const kfs = toWorldKeyframes(sim);
    const res = computeWorldAlignment({ keyframes: kfs, imu });
    if (LOG) console.log('orbit world: s err %', (100 * (res.s * sim.s - 1)).toFixed(2), res.metric, res.details.warnings);
    expect(res.metric.metric).toBe(true);
    expect(res.metric.source).toBe('imu');
    expect(res.metric.gravityAligned).toBe(true);
    expect(res.metric.northAligned).toBe(false);
    expect(res.metric.originLatLonAlt).toBeNull();
    expect(Math.abs(res.s * sim.s - 1)).toBeLessThan(0.03);
    expect(res.metric.relativeUncertainty!).toBeGreaterThan(0);
    expect(res.metric.relativeUncertainty!).toBeLessThan(0.03);
    // gravity → −Z
    const z = mat3MulVec(res.R, sim.downRec);
    expect(Math.acos(-z[2])).toBeLessThan(1 * DEG);
    const C = alignedCentres(kfs, res.s, res.R, res.t);
    expect(Math.hypot(C[0], C[1], C[2])).toBeLessThan(1e-9);
    const e = shapeErrors(C, sim.centresTrue);
    expect(e.maxRel).toBeLessThan(0.04);
    expect(e.maxDz).toBeLessThan(0.02); // heights (m) — needs correct up and scale
    expect(Math.abs(res.details.offsetMs + 40)).toBeLessThan(5);
  });

  it('orbit + IMU with an inverted accelerometer: sign corrected from gravityCam, not upside down', () => {
    // the accelerometer sign self-check decided late or wrongly: every specific force is negated,
    // which alone puts gravity exactly UP (a reliable-looking, upside-down model)
    const rng = new Rng(1);
    const traj = orbitTrajectory(rng);
    const Rcd = cameraFromDevice(0);
    const imu = simulateImu(traj, Rcd, rng).map((s) => ({
      ...s,
      accel: s.accel ? ([-s.accel[0], -s.accel[1], -s.accel[2]] as [number, number, number]) : null,
      accelG: s.accelG ? ([-s.accelG[0], -s.accelG[1], -s.accelG[2]] as [number, number, number]) : null,
    }));
    const sim = simulateKeyframes(traj, Rcd, rng, { offsetMs: -40, s: 3.3 });
    const Rw = new Float64Array(9);
    const withGc = toWorldKeyframes(sim, (i, k) => {
      traj.Rwc((sim.keyframes[i].tTrue - 5000) / 1000, Rw);
      k.gravityCam = mat3MulVec(mat3Transpose(Rw), [0.009 * rng.gaussian(), 0.009 * rng.gaussian(), -1]);
    });
    const res = computeWorldAlignment({ keyframes: withGc, imu });
    expect(res.details.accelSignCorrected).toBe(true);
    expect(res.details.gravity!.source).toBe('accelG');
    expect(res.metric.gravityAligned).toBe(true);
    expect(Math.acos(-mat3MulVec(res.R, sim.downRec)[2])).toBeLessThan(1 * DEG);
    // the corrected samples give the metric scale back as well
    expect(res.metric.source).toBe('imu');
    expect(Math.abs(res.s * sim.s - 1)).toBeLessThan(0.03);

    // correct sign: nothing changes
    const ok = computeWorldAlignment({ keyframes: withGc, imu: simulateImu(traj, Rcd, new Rng(1)) });
    expect(ok.details.accelSignCorrected).toBe(false);
  });

  it('orbit + IMU + compass: north-aligned within a few degrees', () => {
    const rng = new Rng(2);
    const traj = orbitTrajectory(rng);
    const Rcd = cameraFromDevice(0);
    const imu = simulateImu(traj, Rcd, rng);
    const sim = simulateKeyframes(traj, Rcd, rng, { offsetMs: 10, s: 0.8 });
    const Rw = new Float64Array(9);
    const kfs = toWorldKeyframes(sim, (i, k) => {
      // optical axis azimuth in the (true ENU) sim world, CW from north (+Y)
      traj.Rwc((sim.keyframes[i].tTrue - 5000) / 1000, Rw);
      k.heading = (Math.atan2(Rw[2], Rw[5]) * 180) / Math.PI + 6 * rng.gaussian();
      k.headingAccuracy = 10;
    });
    const res = computeWorldAlignment({ keyframes: kfs, imu });
    expect(res.details.yawSource).toBe('compass');
    expect(res.metric.northAligned).toBe(true);
    const err = rotationAngle(mat3Mul(res.R, sim.R)); // X' ∝ R R0 X_true ⇒ R R0 ≈ I
    if (LOG) console.log('compass: rotation error deg', (err / DEG).toFixed(2), 'σ deg', (res.details.compass!.stdRad / DEG).toFixed(2));
    expect(err).toBeLessThan(3 * DEG);
  });

  it('walk + IMU + GPS: fused scale, GPS yaw, ENU positions', () => {
    const rng = new Rng(3);
    const traj = walkTrajectory(rng);
    const Rcd = cameraFromDevice(90);
    const imu = simulateImu(traj, Rcd, rng);
    const sim = simulateKeyframes(traj, Rcd, rng, { offsetMs: 25, s: 0.21, posNoise: 0.002 });
    const { fixes, frame } = simulateGps(traj, rng, { sigma: 3 });
    const kfs = toWorldKeyframes(sim);
    const res = computeWorldAlignment({ keyframes: kfs, imu, gpsFixes: fixes });
    const d = res.details;
    if (LOG)
      console.log(
        'walk world:',
        d.scaleCandidates.map((c) => `${c.source} ${(100 * (c.scale * sim.s - 1)).toFixed(2)}% ±${(100 * c.relUncertainty).toFixed(2)}%`).join(', '),
        '→',
        (100 * (res.s * sim.s - 1)).toFixed(2),
        res.metric,
        d.warnings,
      );
    expect(res.metric.source).toBe('imu+gps');
    expect(res.metric.northAligned).toBe(true);
    expect(res.metric.gravityAligned).toBe(true);
    expect(res.metric.originLatLonAlt).not.toBeNull();
    expect(d.yawSource).toBe('gps');
    expect(Math.abs(res.s * sim.s - 1)).toBeLessThan(0.03);
    // inverse-variance fusion: σ below both inputs
    for (const c of d.scaleCandidates) expect(res.metric.relativeUncertainty!).toBeLessThanOrEqual(c.relUncertainty + 1e-12);
    // positions in the GPS ENU frame (sim world = ENU at the GPS origin of the simulation)
    const C = alignedCentres(kfs, res.s, res.R, res.t);
    const o = res.metric.originLatLonAlt!;
    const off = frame.fromGeodetic(o[0], o[1], o[2]); // our origin in the simulation's ENU
    let maxH = 0;
    for (let i = 0; i < kfs.length; i++) {
      const ex = C[3 * i] + off[0] - sim.centresTrue[3 * i];
      const ey = C[3 * i + 1] + off[1] - sim.centresTrue[3 * i + 1];
      maxH = Math.max(maxH, Math.hypot(ex, ey));
    }
    if (LOG) console.log('max horizontal ENU error (m)', maxH.toFixed(2));
    expect(maxH).toBeLessThan(4);
    expect(rotationAngle(mat3Mul(res.R, sim.R))).toBeLessThan(5 * DEG);
  });

  it('walk + GPS only (gravity from gravityCam), and without any sensor', () => {
    const rng = new Rng(4);
    const traj = walkTrajectory(rng);
    const Rcd = cameraFromDevice(90);
    const sim = simulateKeyframes(traj, Rcd, rng, { offsetMs: 0, s: 1.7 });
    const { fixes } = simulateGps(traj, rng, { sigma: 3 });
    const Rw = new Float64Array(9);
    const kfs = toWorldKeyframes(sim, (i, k) => {
      traj.Rwc((sim.keyframes[i].tTrue - 5000) / 1000, Rw);
      // DOWN in the camera frame: R_wcᵀ (0, 0, −1), 0.5° noise
      const g = mat3MulVec(mat3Transpose(Rw), [0.009 * rng.gaussian(), 0.009 * rng.gaussian(), -1]);
      k.gravityCam = g;
      k.geo = null;
    });
    const res = computeWorldAlignment({ keyframes: kfs, gpsFixes: fixes });
    expect(res.details.gravity!.source).toBe('gravityCam');
    expect(res.metric.source).toBe('gps');
    expect(res.details.gps!.mode).toBe('4dof');
    expect(Math.abs(res.s * sim.s - 1)).toBeLessThan(0.1);
    expect(Math.acos(-mat3MulVec(res.R, sim.downRec)[2])).toBeLessThan(0.5 * DEG);

    const none = computeWorldAlignment({ keyframes: toWorldKeyframes(sim) }, { origin: 'keep' });
    expect(none.s).toBe(1);
    expect(rotationAngle(none.R)).toBe(0);
    expect(Array.from(none.t)).toEqual([0, 0, 0]);
    expect(none.metric).toEqual({ metric: false, source: 'none', relativeUncertainty: null, gravityAligned: false, northAligned: false, originLatLonAlt: null });
    const cen = computeWorldAlignment({ keyframes: kfs }, { origin: 'centroid' });
    const Cc = alignedCentres(kfs, cen.s, cen.R, cen.t);
    let mx = 0,
      my = 0,
      mz = 0;
    for (let i = 0; i < kfs.length; i++) {
      mx += Cc[3 * i];
      my += Cc[3 * i + 1];
      mz += Cc[3 * i + 2];
    }
    expect(Math.hypot(mx, my, mz) / kfs.length).toBeLessThan(1e-9);
  });

  it('inconsistent IMU and GPS scales: keeps the more certain one and warns', () => {
    const rng = new Rng(5);
    const traj = walkTrajectory(rng);
    const Rcd = cameraFromDevice(90);
    const imu = simulateImu(traj, Rcd, rng);
    const sim = simulateKeyframes(traj, Rcd, rng, { offsetMs: 0, s: 1, posNoise: 0.002 });
    // GPS of a walk 30 % longer (wrong scale)
    const big = { duration: traj.duration, pos: (t: number, out: Float64Array) => (traj.pos(t, out), out.forEach((v, i) => (out[i] = 1.3 * v))), Rwc: traj.Rwc };
    const { fixes } = simulateGps(big, rng, { sigma: 1 });
    const res = computeWorldAlignment({ keyframes: toWorldKeyframes(sim), imu, gpsFixes: fixes as GeoFix[] });
    expect(res.details.warnings.some((w) => /disagree/.test(w))).toBe(true);
    expect(res.metric.source).not.toBe('imu+gps');
    const best = [...res.details.scaleCandidates].sort((a, b) => a.relUncertainty - b.relUncertainty)[0];
    expect(res.s).toBe(best.scale);
  });

  it('worldKeyframeFromFrame copies the frame metadata', () => {
    const pose = { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(1, 2, 3) };
    const info = {
      id: 7,
      t: 1234,
      width: 1920,
      height: 1440,
      sharpness: 10,
      R_cam_device: cameraFromDevice(90),
      gravityCam: Float64Array.of(0, 1, 0),
      heading: 45,
      geo: null,
    };
    const k = worldKeyframeFromFrame(info, pose, 12);
    expect(k).toMatchObject({ id: 7, t: 1234, heading: 45, headingAccuracy: 12, geo: null });
    expect(k.pose).toBe(pose);
    expect(k.R_cam_device).toBe(info.R_cam_device);
  });

  it('degrades gracefully with an IMU lacking accelerations', () => {
    const rng = new Rng(6);
    const traj = orbitTrajectory(rng);
    const Rcd = cameraFromDevice(0);
    const imu: ImuSample[] = simulateImu(traj, Rcd, rng).map((s) => ({ ...s, accel: null, accelG: null }));
    const sim = simulateKeyframes(traj, Rcd, rng, { offsetMs: 0, s: 2 });
    const res = computeWorldAlignment({ keyframes: toWorldKeyframes(sim), imu });
    expect(res.metric.metric).toBe(false);
    expect(res.metric.gravityAligned).toBe(false);
    expect(res.details.timeOffset).not.toBeNull();
    expect(Math.abs(res.details.offsetMs)).toBeLessThan(5);
  });
});
