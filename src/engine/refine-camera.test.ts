import { describe, expect, it } from 'vitest';
import type { ImuSample } from '../core/types';
import { cameraFromDevice } from '../sensors/frame-orientation';
import { GyroReference } from '../sfm/gyro';
import { runIncrementalSfM } from '../sfm/mapper';
import { buildSyntheticDatabase, generateScene } from '../sfm/synthetic';
import { CaptureState } from './live';
import { applyRollingShutter, correctKeyframeRS, refinePerImageFocal } from './refine-camera';
import type { KeyframeRecord } from './types';

/** Orbit scene whose image i was taken with focal f·(1 + b_i) (keypoints scaled about the principal point). */
async function breathingModel(amp: number, seed = 5) {
  const scene = generateScene({ kind: 'orbit', numImages: 30, seed, noisePx: 0.3, k1: 0, k2: 0, descriptors: false, outlierRatio: 0 });
  const cam = scene.camera;
  const b = scene.imageIds.map((_, i) => amp * Math.sin(0.45 * i));
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
    const r = await refinePerImageFocal(rec, db, rec.correspondenceGraph!, {});
    expect(r.accepted).toBe(true);
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
    const r0 = await refinePerImageFocal(flat.rec, flat.db, flat.rec.correspondenceGraph!, {});
    expect(r0.accepted).toBe(false);
    expect(r0.rec).toBe(flat.rec);
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
    // re-applying another model starts from the originals (absolute, not cumulative)
    correctKeyframeRS(st, 0, cam, gyro, { axis: 'rows', readoutMs: 22 });
    const once = kp.slice();
    correctKeyframeRS(st, 0, cam, gyro, { axis: 'rows', readoutMs: 22 });
    expect(Array.from(kp)).toEqual(Array.from(once));
    applyRollingShutter(st, () => cam, gyro, null);
    expect(Array.from(kp)).toEqual(Array.from(orig));
    expect(st.rollingShutter).toBeNull();
  });
});
