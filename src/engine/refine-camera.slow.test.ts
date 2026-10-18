import { describe, expect, it } from 'vitest';
import type { ImuSample } from '../core/types';
import { cameraFromDevice } from '../sensors/frame-orientation';
import { GyroReference } from '../sfm/gyro';
import { runIncrementalSfM } from '../sfm/mapper';
import { buildSyntheticDatabase, generateScene } from '../sfm/synthetic';
import { CaptureState } from './live';
import { correctKeypointsRS } from '../sfm/rolling-shutter';
import { modelFromReconstruction } from './refine';
import {
  applyRollingShutter,
  cameraVelocities,
  correctKeyframeRS,
  inverseDepths,
  modelKeypoints,
  refinePerImageFocal,
  RS_OMEGA_HALF_WINDOW_MS,
  rsTranslations,
  sparseRollingShutter,
} from './refine-camera';
import type { KeyframeRecord } from './types';

/** Orbit scene whose image i was taken with focal f·(1 + b_i) (keypoints scaled about the principal point). */
async function breathingModel(amp: number, seed = 5, shape: (i: number) => number = (i) => Math.sin(0.45 * i)) {
  const scene = generateScene({ kind: 'orbit', numImages: 30, seed, noisePx: 0.3, k1: 0, k2: 0, descriptors: false, outlierRatio: 0 });
  const cam = scene.camera;
  const b = scene.imageIds.map((_, i) => amp * shape(i));
  scene.features.forEach((fs, i) => {
    const kp = fs.keypoints;
    for (let k = 0; k < fs.count; k++) {
      kp[4 * k] = cam.cx + (kp[4 * k] - cam.cx) * (1 + b[i]);
      kp[4 * k + 1] = cam.cy + (kp[4 * k + 1] - cam.cy) * (1 + b[i]);
    }
  });
  const { db } = buildSyntheticDatabase(scene, { verify: false });
  const [rec] = await runIncrementalSfM(db, { intrinsics: { ...cam } });
  return { scene, db, rec, b };
}

describe('refinePerImageFocal', () => {
  it('recovers per-image focal lengths that breathe by ±0.6 %, and keeps the shared camera without breathing', async () => {
    const { scene, db, rec, b } = await breathingModel(0.006);
    expect(rec.numRegisteredImages).toBe(30);
    const r = await refinePerImageFocal(rec, db, rec.correspondenceGraph!, {}, { timeOrder: scene.imageIds });
    expect(r.accepted).toBe(true);
    expect(r.autocorrelation).toBeGreaterThan(0.5);
    const ids = scene.imageIds;
    const f = ids.map((id) => r.rec.cameras.get(r.rec.images.get(id)!.cameraId)!.f);
    const mf = f.reduce((a, x) => a + x, 0) / f.length;
    const mb = b.reduce((a, x) => a + x, 0) / b.length;
    let sxy = 0,
      sxx = 0,
      syy = 0;
    for (let i = 0; i < f.length; i++) {
      const x = f[i] / mf - 1,
        y = b[i] - mb;
      sxy += x * y;
      sxx += x * x;
      syy += y * y;
    }
    expect(sxy / Math.sqrt(sxx * syy)).toBeGreaterThan(0.9);
    expect(r.after).toBeLessThan(0.9 * r.before);
    // every registered image has its own camera; the input model is untouched
    expect(new Set(ids.map((id) => r.rec.images.get(id)!.cameraId)).size).toBe(30);
    expect(rec.cameras.size).toBe(1);

    const flat = await breathingModel(0);
    const r0 = await refinePerImageFocal(flat.rec, flat.db, flat.rec.correspondenceGraph!, {}, { timeOrder: flat.scene.imageIds });
    expect(r0.accepted).toBe(false);
    expect(r0.rec).toBe(flat.rec);
    // focal lengths that alternate between neighbouring frames are not focus breathing
    const jumpy = await breathingModel(0.006, 5, (i) => (i % 2 ? 1 : -1));
    const rj = await refinePerImageFocal(jumpy.rec, jumpy.db, jumpy.rec.correspondenceGraph!, {}, { timeOrder: jumpy.scene.imageIds });
    expect(rj.accepted).toBe(false);
    expect(rj.reason).toMatch(/smooth/);
  });
});

describe('refinePerImageFocal + shared principal point', () => {
  it('refines the shared camera on keypoints rescaled to the shared focal, restores them and rebuilds per-image cameras', async () => {
    const { scene, db, rec } = await breathingModel(0.006, 6);
    const before = scene.features.map((f) => f.keypoints.slice());
    let sawRescaled = false;
    let calls = 0;
    const r = await refinePerImageFocal(rec, db, rec.correspondenceGraph!, {}, {
      timeOrder: scene.imageIds,
      rounds: 2,
      refinePrincipalPoint: async (shared) => {
        calls++;
        // one shared camera, keypoints rescaled (some moved, by < 1 %)
        expect(new Set(shared.registeredImageIds().map((id) => shared.images.get(id)!.cameraId)).size).toBe(1);
        let maxRel = 0;
        scene.features.forEach((f, i) => {
          for (let k = 0; k < f.keypoints.length; k += 4) {
            const d = Math.abs(f.keypoints[k] - before[i][k]);
            if (d > 0) maxRel = Math.max(maxRel, d / Math.max(1, Math.abs(before[i][k] - scene.camera.cx)));
          }
        });
        sawRescaled = maxRel > 1e-4 && maxRel < 0.02;
        const out = shared.clone();
        for (const c of out.cameras.values()) c.cx += 2; // pretend the shared refinement moved cx
        return { rec: out, accepted: true, reason: 'test' };
      },
    });
    expect(sawRescaled).toBe(true);
    expect(r.accepted).toBe(true);
    // keypoints restored exactly; per-image cameras carry the new principal point
    scene.features.forEach((f, i) => expect(Array.from(f.keypoints)).toEqual(Array.from(before[i])));
    const cams = scene.imageIds.map((id) => r.rec.cameras.get(r.rec.images.get(id)!.cameraId)!);
    expect(new Set(cams.map((c) => c)).size).toBe(30);
    // two alternations: the (pretend) shared refinement ran twice, each moving cx by 2 px
    expect(calls).toBe(2);
    for (const c of cams) expect(c.cx).toBeCloseTo(scene.camera.cx + 4, 6);
  });
});

describe('rolling-shutter application', () => {
  it('corrects keypoints in place from saved originals and restores them exactly', () => {
    const st = new CaptureState();
    const W = 960,
      H = 720;
    const cam = { width: W, height: H, f: 750, cx: 480, cy: 360, k1: -0.02, k2: 0 };
    const kp = new Float32Array(4 * 50);
    for (let k = 0; k < 50; k++) kp.set([20 * k, 14 * k, 2, 0], 4 * k);
    const fs = { count: 50, keypoints: kp, descriptors: new Uint8Array(50 * 128), scores: new Float32Array(50), width: W, height: H };
    const Rcd = cameraFromDevice(90);
    st.db.addFrame({ id: 0, t: 1000, width: W, height: H, sharpness: 1, R_cam_device: Rcd, gravityCam: null, heading: null, geo: null }, fs);
    st.records.set(0, { id: 0, frameIndex: 0, t: 1000, width: W, height: H } as KeyframeRecord);
    st.order.push(0);
    const imu: ImuSample[] = [];
    for (let t = 0; t <= 2000; t += 10) imu.push({ t, gyro: [0, 1.2, 0], accel: null, accelG: null });
    const gyro = new GyroReference(imu, { R_cam_device: Rcd });
    const orig = kp.slice();
    expect(applyRollingShutter(st, () => cam, gyro, { axis: 'rows', readoutMs: 22 })).toBe(1);
    expect(st.rollingShutter).toEqual({ axis: 'rows', readoutMs: 22 });
    let moved = 0;
    for (let k = 0; k < 50; k++) moved = Math.max(moved, Math.abs(kp[4 * k] - orig[4 * k]) + Math.abs(kp[4 * k + 1] - orig[4 * k + 1]));
    expect(moved).toBeGreaterThan(3);
    expect(st.db.features.get(0)!.keypoints).toBe(kp); // in place: shared with reconstructions
    // the ω the keypoints were corrected with is recorded (the dense stage / exports reuse it, not
    // a later gyro query whose time offset may have moved); same window as the estimation
    const w0 = gyro.angularVelocityCam(1000, RS_OMEGA_HALF_WINDOW_MS)!;
    expect(Array.from(st.rsOmega.get(0)!)).toEqual(Array.from(w0));
    expect(Array.from(kp)).toEqual(Array.from(correctKeypointsRS(orig, cam, w0, { axis: 'rows', readoutMs: 22 })));
    const offsetBefore = gyro.offsetMs;
    gyro.offsetMs = 40; // e.g. the world alignment re-estimated the camera–IMU offset afterwards
    expect(Array.from(st.rsOmega.get(0)!)).toEqual(Array.from(w0));
    gyro.offsetMs = offsetBefore;
    // re-applying another model starts from the originals (absolute, not cumulative)
    correctKeyframeRS(st, 0, cam, gyro, { axis: 'rows', readoutMs: 22 });
    const once = kp.slice();
    correctKeyframeRS(st, 0, cam, gyro, { axis: 'rows', readoutMs: 22 });
    expect(Array.from(kp)).toEqual(Array.from(once));
    applyRollingShutter(st, () => cam, gyro, null);
    expect(Array.from(kp)).toEqual(Array.from(orig));
    expect(st.rollingShutter).toBeNull();
    expect(st.rsOmega.size).toBe(0);
    // no gyro coverage at the keyframe time ⇒ restored, no ω recorded
    correctKeyframeRS(st, 0, cam, gyro, { axis: 'rows', readoutMs: 22 });
    expect(st.rsOmega.size).toBe(1);
    st.records.set(0, { id: 0, frameIndex: 0, t: 9000, width: W, height: H } as KeyframeRecord);
    expect(correctKeyframeRS(st, 0, cam, gyro, { axis: 'rows', readoutMs: 22 })).toBe(false);
    expect(Array.from(kp)).toEqual(Array.from(orig));
    expect(st.rsOmega.size).toBe(0);
  });
});

describe('camera velocities / inverse depths (translational rolling-shutter term)', () => {
  it('match the ground truth in the camera frame, scale-free (velocity × inverse depth)', async () => {
    const { scene, rec } = await breathingModel(0);
    const dtMs = 400;
    const timeOf = (id: number): number => dtMs * scene.imageIds.indexOf(id);
    const vel = cameraVelocities(rec, timeOf);
    expect(vel.size).toBe(30);
    const centre = (R: ArrayLike<number>, t: ArrayLike<number>): number[] => [0, 1, 2].map((c) => -(R[c] * t[0] + R[3 + c] * t[1] + R[6 + c] * t[2]));
    let checked = 0;
    for (let k = 1; k + 1 < scene.imageIds.length; k++) {
      const id = scene.imageIds[k];
      // ground truth: central difference of the GT centres, in the GT camera frame
      const a = centre(scene.poses[k - 1].R, scene.poses[k - 1].t),
        b = centre(scene.poses[k + 1].R, scene.poses[k + 1].t);
      const R = scene.poses[k].R;
      const vw = [0, 1, 2].map((c) => (b[c] - a[c]) / ((2 * dtMs) / 1000));
      const vg = [0, 1, 2].map((r) => R[3 * r] * vw[0] + R[3 * r + 1] * vw[1] + R[3 * r + 2] * vw[2]);
      // median GT depth of the image's points vs the reconstruction's median inverse depth
      const d = inverseDepths(rec, id)!;
      const zs: number[] = [];
      const fp = scene.featurePoint[k];
      const t = scene.poses[k].t;
      for (let f = 0; f < fp.length; f++) {
        if (fp[f] < 0 || !(d.inv[f] > 0)) continue;
        const X = [scene.points[3 * fp[f]], scene.points[3 * fp[f] + 1], scene.points[3 * fp[f] + 2]];
        zs.push(1 / (R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2]));
      }
      zs.sort((x, y) => x - y);
      const nuGt = vg.map((x) => x * zs[zs.length >> 1]);
      const v = vel.get(id)!;
      const nu = [0, 1, 2].map((c) => v[c] * d.median);
      const err = Math.hypot(nu[0] - nuGt[0], nu[1] - nuGt[1], nu[2] - nuGt[2]) / Math.hypot(...nuGt);
      expect(err).toBeLessThan(0.03);
      checked++;
    }
    expect(checked).toBe(28);
    // a neighbour too far away in time is not used (one-sided difference / none)
    const far = cameraVelocities(rec, (id) => 10_000 * scene.imageIds.indexOf(id), 3000);
    expect(far.size).toBe(0);
  }, 60_000);
});

describe('keypoints consistent with a rolling-shutter corrected sparse model (exports)', () => {
  it('after a reload the persisted model reproduces the in-session correction (rotation + translation)', async () => {
    const { scene, db, rec } = await breathingModel(0);
    const st = new CaptureState();
    const dtMs = 400;
    scene.imageIds.forEach((id, k) => {
      const fs = db.features.get(id)!; // the SAME arrays as the reconstruction's keypoints
      st.db.addFrame({ id, t: dtMs * k, width: fs.width, height: fs.height, sharpness: 1, R_cam_device: cameraFromDevice(0), gravityCam: null, heading: null, geo: null }, fs);
      st.records.set(id, { id, frameIndex: k, t: dtMs * k, width: fs.width, height: fs.height } as KeyframeRecord);
      st.order.push(id);
    });
    const imu: ImuSample[] = [];
    for (let t = -500; t <= dtMs * scene.imageIds.length + 500; t += 10) imu.push({ t, gyro: [0.3, 1.0 + 0.3 * Math.sin(t / 700), -0.2], accel: null, accelG: null });
    const gyro = new GyroReference(imu, { R_cam_device: cameraFromDevice(0) });
    const timeOf = (id: number): number | undefined => st.records.get(id)?.t;
    const trans = rsTranslations(rec, timeOf);
    const camOf = (id: number) => rec.cameras.get(rec.images.get(id)!.cameraId);
    // unregistered-at-estimation images get rotation only: drop the translation of one image
    const noTrans = scene.imageIds[3];
    expect(applyRollingShutter(st, camOf, gyro, { axis: 'cols', readoutMs: -24 }, (id) => (id === noTrans ? null : trans.get(id) ?? null))).toBe(scene.imageIds.length);
    const order = st.order.slice();
    const model = modelFromReconstruction(rec, order);
    model.rollingShutter = sparseRollingShutter(st, rec, model.images.map((im) => im.id));
    expect(model.rollingShutter).not.toBeNull();
    expect(Object.keys(model.rollingShutter!.omega).length).toBe(scene.imageIds.length);
    expect(model.rollingShutter!.nu![noTrans]).toBeUndefined();
    let maxShift = 0,
      maxDiff = 0;
    for (const id of scene.imageIds) {
      const mem = st.db.features.get(id)!.keypoints;
      const orig = st.rsOriginal.get(id)!;
      // in session: the corrected in-memory keypoints themselves
      expect(modelKeypoints(model, id, orig, { st, keypoints: mem })).toBe(mem);
      // after a reload (store = originals): re-derived from the model
      const re = modelKeypoints(model, id, orig, null);
      for (let k = 0; k < mem.length; k += 4) {
        maxShift = Math.max(maxShift, Math.hypot(mem[k] - orig[k], mem[k + 1] - orig[k + 1]));
        maxDiff = Math.max(maxDiff, Math.hypot(re[k] - mem[k], re[k + 1] - mem[k + 1]));
      }
    }
    expect(maxShift).toBeGreaterThan(5);
    expect(maxDiff).toBeLessThan(1e-3);
    // a model without rolling shutter (e.g. an older refine) pairs with the ORIGINAL keypoints
    const id0 = scene.imageIds[0];
    const plain = { ...model, rollingShutter: null };
    expect(modelKeypoints(plain, id0, st.rsOriginal.get(id0)!, { st, keypoints: st.db.features.get(id0)!.keypoints })).toBe(st.rsOriginal.get(id0));
    // a respawned worker (originals in memory, nothing corrected) with the persisted model
    const corrected = st.db.features.get(id0)!.keypoints.slice();
    applyRollingShutter(st, camOf, gyro, null);
    const back = modelKeypoints(model, id0, st.db.features.get(id0)!.keypoints, { st, keypoints: st.db.features.get(id0)!.keypoints });
    let d = 0;
    for (let k = 0; k < back.length; k++) d = Math.max(d, Math.abs(back[k] - corrected[k]));
    expect(d).toBeLessThan(1e-3);
  }, 60_000);
});
