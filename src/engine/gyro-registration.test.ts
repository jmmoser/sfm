/**
 * LiveSession.predictRotation (gyro-aided registration, MapperOptions.predictRotation): the
 * rotation of a keyframe propagated by the gyro from the registered keyframe nearest in time,
 * against the poses and the exact IMU of a synthetic orbit (src/engine/testutil.ts, not rendered).
 */
import { describe, expect, it } from 'vitest';
import type { Pose } from '../core/types';
import { mat3Mul, mat3Transpose } from '../math/mat3';
import { rotationAngle, rotationAngleBetween, rotvecToMat3 } from '../math/rotation';
import { Reconstruction } from '../sfm/reconstruction';
import { LiveSession } from './live';
import { MemoryEngineStore } from './persistence';
import { QUALITY_PRESETS } from './presets';
import { makeCapture } from './testutil';
import type { EngineConfig, KeyframeRecord } from './types';

const DEG = Math.PI / 180;

/** θ = 90 (raw sensor buffers): R_cam_device is not the identity, so a wrong mapping would show. */
const cap = makeCapture({ render: false, frames: 60, fps: 10, theta: 90, timeOffsetMs: 0, arcDeg: 120 });

function session(config: EngineConfig = {}, verified = true): LiveSession {
  const live = new LiveSession({
    mode: 'object',
    quality: 'fast',
    hfovPriorDeg: 55,
    config: { mapping: 'inline', ...config },
    preset: QUALITY_PRESETS.fast,
    store: new MemoryEngineStore('gyro-registration'),
    pipeline: () => {
      throw new Error('no feature pipeline in this test');
    },
    gpu: (fn) => fn(),
    hidden: () => false,
    log: () => {},
    emitLive: () => {},
    storageError: () => {},
  });
  const st = live.state;
  st.imu.load(cap.imu.slice());
  st.thetaSession = 90;
  st.cameraTheta = 90;
  st.cameraThetaVerified = verified;
  st.timeOffsetMs = 0;
  // keyframe id = frame index
  for (const f of cap.frames) st.records.set(f.index, { t: f.t } as KeyframeRecord);
  return live;
}

/** A model with the given frames registered at `pose(frame)` (default the ground truth). */
function model(registered: number[], pose: (k: number) => Pose = (k) => cap.frames[k].pose): Reconstruction {
  const rec = new Reconstruction();
  const cid = rec.addCamera({ ...cap.cam });
  for (const f of cap.frames) rec.addImage(f.index, cid, 0, new Float32Array(0));
  for (const k of registered) rec.registerImage(k, pose(k));
  return rec;
}

const turned = (p: Pose, deg: number): Pose => ({ R: mat3Mul(rotvecToMat3(Float64Array.of(0.36, 0.48, 0.8).map((v) => v * deg * DEG)), p.R), t: p.t });

describe('LiveSession.predictRotation (gyro-aided registration)', () => {
  it('propagates the nearest registered keyframe by the gyro: R̂ = G · R_n, within σ of the truth', () => {
    const live = session();
    // keyframe 30 from 25 (0.5 s, 10° of the orbit) and from 36 (0.6 s, after it)
    for (const nb of [25, 36]) {
      const p = live.predictRotation(30, model([nb]))!;
      expect(p).not.toBeNull();
      const err = rotationAngleBetween(p.R, cap.frames[30].pose.R);
      expect(err / DEG).toBeLessThan(0.05);
      // a transposed G (camera t → camera t_n) would be off by twice the motion
      const G = live.gyroRotation(cap.frames[nb].t, cap.frames[30].t)!;
      expect(rotationAngle(G) / DEG).toBeGreaterThan(8);
      expect(rotationAngleBetween(mat3Mul(mat3Transpose(G), cap.frames[nb].pose.R), cap.frames[30].pose.R)).toBeGreaterThan(1.9 * rotationAngle(G));
      // σ = 1° + 0.3°/s · |Δt| + 3 % of the gyro rotation
      const dt = Math.abs(cap.frames[nb].t - cap.frames[30].t) / 1000;
      expect(p.sigmaRad).toBeCloseTo((1 + 0.3 * dt) * DEG + 0.03 * rotationAngle(G), 12);
      expect(err).toBeLessThan(p.sigmaRad);
    }
    // its own pose never predicts it
    expect(live.predictRotation(30, model([30]))).toBeNull();
  });

  it('uses the registered keyframe nearest in time, and none further than 3 s', () => {
    const live = session();
    // 28 (0.2 s) is right, 20 (1 s) and 40 (1 s) are 10° off: the prediction follows 28 …
    const off = (k: number): Pose => (k === 28 ? cap.frames[k].pose : turned(cap.frames[k].pose, 10));
    const p = live.predictRotation(30, model([20, 28, 40], off))!;
    expect(rotationAngleBetween(p.R, cap.frames[30].pose.R) / DEG).toBeLessThan(0.05);
    // … and its error when 28 is the one that is off
    const q = live.predictRotation(30, model([20, 28, 40], (k) => (k === 28 ? turned(cap.frames[k].pose, 10) : cap.frames[k].pose)))!;
    expect(rotationAngleBetween(q.R, cap.frames[30].pose.R) / DEG).toBeCloseTo(10, 3);
    // 3.1 s and 3.5 s away: no prediction; 2.9 s: one
    expect(live.predictRotation(40, model([9, 5]))).toBeNull();
    expect(live.predictRotation(40, model([11]))).not.toBeNull();
  });

  it('needs the verified R_cam_device; works without the gyro gates; off with gyroAidedRegistration: false', () => {
    const rec = model([25]);
    expect(session({}, false).predictRotation(30, rec)).toBeNull();
    // the gates off (A/B of the gyro checks) leaves the registration aid on
    const ungated = session({ gyroGate: false });
    expect(ungated.gyroForChecks()).toBeNull();
    expect(rotationAngleBetween(ungated.predictRotation(30, rec)!.R, cap.frames[30].pose.R) / DEG).toBeLessThan(0.05);
    // wired into the live mapper unless disabled
    expect(session().mapper.opts.predictRotation).toBeTypeOf('function');
    expect(session({ gyroAidedRegistration: false }).mapper.opts.predictRotation).toBeUndefined();
    // an IMU gap (> 0.5 s) over the interval: no prediction
    const gap = session();
    gap.state.imu.load(cap.imu.filter((s) => s.t < cap.frames[26].t || s.t > cap.frames[33].t));
    expect(gap.predictRotation(30, rec)).toBeNull();
  });
});
