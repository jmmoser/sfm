/**
 * PhotogrammetryEngine helpers that need no reconstruction: frame ingest (canonical
 * orientation, settle, raw sensor buffers), the R_cam_device hand-eye check on GT poses,
 * capture mode classification and persistence records. The end-to-end engine runs on
 * synthetic captures are in engine-orbit.slow.test.ts and engine-resume.slow.test.ts.
 */
import { describe, expect, it } from 'vitest';
import type { FeatureSet, Pose } from '../core/types';
import { rotateImagePoint } from '../sensors/frame-orientation';
import { frameToBitmap } from './engine';
import { OrientationTracker, gravityDown, overlayPoints, prepareFrame } from './frame-ingest';
import { ImuLog, classifyScene, handEyeCheck } from './live';
import { packImu, packSparseModel, unpackImu, unpackSparseModel } from './persistence';
import { makeCapture } from './testutil';

// ---------------------------------------------------------------------------------------------

describe('frame ingest', () => {
  it('a declared raw frame is upright for θ = 90 in every UI rotation and never changes the delivery-path state', () => {
    // MSTP-free VideoFrame grab: landscape-90 → landscape-270 without a portrait frame in between
    const o = new OrientationTracker(500);
    expect(o.classify({ width: 640, height: 480, screenAngle: 90, t: 0, raw: true })).toMatchObject({ ok: true, effectiveTheta: 90, k: 0, raw: true });
    expect(o.thetaSession).toBe(90);
    expect(o.classify({ width: 640, height: 480, screenAngle: 270, t: 700, raw: true }).ok).toBe(false); // settling after the rotation
    expect(o.classify({ width: 640, height: 480, screenAngle: 270, t: 1300, raw: true })).toMatchObject({ ok: true, effectiveTheta: 90, k: 0 });
    o.classify({ width: 640, height: 480, screenAngle: 0, t: 1400, raw: true });
    expect(o.classify({ width: 640, height: 480, screenAngle: 0, t: 2000, raw: true })).toMatchObject({ ok: true, effectiveTheta: 90, k: 0 });
    expect(o.rawDelivery).toBeNull(); // per frame only: nothing persisted
    expect(o.classify({ width: 480, height: 640, screenAngle: 0, t: 2100, raw: true }).ok).toBe(false); // portrait "raw" frame
    // a session started in landscape-270 on the raw path is not ambiguous
    const s = new OrientationTracker(500);
    s.classify({ width: 640, height: 480, screenAngle: 270, t: 0, raw: true });
    expect(s.thetaSession).toBe(90);
    // resumed: captured with upright frames at θ₀ = 270 (rawDelivery false) → raw frames rotate in, state kept
    const r = new OrientationTracker(500, 270);
    r.rawDelivery = false;
    expect(r.classify({ width: 640, height: 480, screenAngle: 270, t: 0, raw: true })).toMatchObject({ ok: true, effectiveTheta: 90, k: 2 });
    expect(r.thetaSession).toBe(270);
    expect(r.rawDelivery).toBe(false);
    // ambiguous resumed session (θ₀ = 270, rawDelivery null): NOT re-interpreted by raw frames…
    const a = new OrientationTracker(500, 270);
    a.classify({ width: 640, height: 480, screenAngle: 270, t: 0, raw: true });
    expect(a.thetaSession).toBe(270);
    expect(a.rawDelivery).toBeNull();
    // …and an upright frame of the <video> path at 270 afterwards is still upright for 270
    expect(a.classify({ width: 640, height: 480, screenAngle: 270, t: 100 })).toMatchObject({ ok: true, effectiveTheta: 270, k: 0, raw: false });
  });

  it('frameToBitmap converts a VideoFrame input (downscaled to maxSide) and always closes it', async () => {
    const g = globalThis as { VideoFrame?: unknown; createImageBitmap?: unknown };
    const saved = { VideoFrame: g.VideoFrame, createImageBitmap: g.createImageBitmap };
    class FakeVideoFrame {
      closed = false;
      constructor(
        readonly displayWidth: number,
        readonly displayHeight: number,
      ) {}
      close(): void {
        this.closed = true;
      }
    }
    let fail = false;
    const calls: unknown[] = [];
    g.VideoFrame = FakeVideoFrame;
    g.createImageBitmap = (src: unknown, opts?: { resizeWidth?: number; resizeHeight?: number }) => {
      calls.push(opts);
      if (fail) return Promise.reject(new Error('decode failed'));
      return Promise.resolve({ width: opts?.resizeWidth ?? 1920, height: opts?.resizeHeight ?? 1440, close() {} });
    };
    try {
      const vf = new FakeVideoFrame(1920, 1440);
      const base = { t: 1, screenAngle: 0 as const, imu: [], rawBuffer: true };
      const out = await frameToBitmap({ ...base, bitmap: vf as unknown as VideoFrame, maxSide: 1280 });
      expect([(out.bitmap as ImageBitmap).width, (out.bitmap as ImageBitmap).height]).toEqual([1280, 960]);
      expect(out.rawBuffer).toBe(true);
      expect(vf.closed).toBe(true);
      const native = new FakeVideoFrame(1920, 1440);
      await frameToBitmap({ ...base, bitmap: native as unknown as VideoFrame });
      expect(calls[1]).toBeUndefined(); // no maxSide: native size
      fail = true;
      const bad = new FakeVideoFrame(1920, 1440);
      await expect(frameToBitmap({ ...base, bitmap: bad as unknown as VideoFrame })).rejects.toThrow(/decode failed/);
      expect(bad.closed).toBe(true); // the camera buffer is released on errors too
      // an ImageBitmap / RGBA input passes through unchanged
      const rgba = { width: 2, height: 2, data: new Uint8ClampedArray(16) };
      const same = { ...base, bitmap: rgba };
      expect(await frameToBitmap(same)).toBe(same);
    } finally {
      g.VideoFrame = saved.VideoFrame;
      g.createImageBitmap = saved.createImageBitmap;
    }
  });

  it('fixes the canonical θ, drops frames after a rotation and detects raw sensor buffers', () => {
    const o = new OrientationTracker(500);
    // first frame: portrait UI, portrait bitmap
    let v = o.classify({ width: 480, height: 640, screenAngle: 0, t: 0 });
    expect(v).toMatchObject({ ok: true, effectiveTheta: 0, k: 0, raw: false });
    expect(o.thetaSession).toBe(0);
    // landscape bitmap in a portrait UI = raw sensor buffer (upright for θ = 90) → 1 clockwise turn
    v = o.classify({ width: 640, height: 480, screenAngle: 0, t: 100 });
    expect(v).toMatchObject({ ok: true, effectiveTheta: 90, k: 1, raw: true });
    // UI rotates to landscape: frames within 500 ms are dropped
    v = o.classify({ width: 640, height: 480, screenAngle: 90, t: 200 });
    expect(v.ok).toBe(false);
    v = o.classify({ width: 640, height: 480, screenAngle: 90, t: 650 });
    expect(v.ok).toBe(false);
    v = o.classify({ width: 640, height: 480, screenAngle: 90, t: 750 });
    expect(v).toMatchObject({ ok: true, effectiveTheta: 90, k: 1 });
    // portrait bitmap in a landscape UI: transient, dropped
    v = o.classify({ width: 480, height: 640, screenAngle: 90, t: 800 });
    expect(v.ok).toBe(false);
    // 270 relative to the portrait session = 3 clockwise turns
    o.classify({ width: 640, height: 480, screenAngle: 270, t: 900 });
    v = o.classify({ width: 640, height: 480, screenAngle: 270, t: 1500 });
    expect(v).toMatchObject({ ok: true, k: 3 });
  });

  it('rotates RGBA frames into the canonical orientation and maps the overlay back', () => {
    const W = 40,
      H = 30;
    const data = new Uint8ClampedArray(W * H * 4).fill(30);
    // a bright pixel at (x=5, y=7)
    const at = (x: number, y: number): number => 4 * (y * W + x);
    data.set([250, 250, 250, 255], at(5, 7));
    const prep = prepareFrame({ width: W, height: H, data }, 1);
    expect([prep.width, prep.height]).toEqual([H, W]);
    const fi = prep.featureInput(false, 1000);
    expect(fi.kind).toBe('gray');
    const g = fi.kind === 'gray' ? fi.gray : null!;
    // clockwise turn: (x, y) → (H − 1 − y, x)
    const p = rotateImagePoint(5, 7, W, H, 1);
    let best = 0,
      bi = -1;
    for (let i = 0; i < g.data.length; i++)
      if (g.data[i] > best) {
        best = g.data[i];
        bi = i;
      }
    expect([bi % g.width, Math.floor(bi / g.width)]).toEqual([p[0], p[1]]);
    // overlay of a canonical keypoint, mapped back with the inverse turn (k = 3)
    const fs: FeatureSet = { count: 1, keypoints: Float32Array.of(p[0], p[1], 1, 0), descriptors: new Uint8Array(128), scores: Float32Array.of(1), width: H, height: W };
    const ov = overlayPoints(fs, prep.width, prep.height, 3, 10);
    expect(ov[0]).toBeCloseTo(5.5 / W, 6);
    expect(ov[1]).toBeCloseTo(7.5 / H, 6);
    prep.release();
  });

  it('gravity from the IMU (spec sign) and IMU log ordering / packing', () => {
    const log = new ImuLog();
    log.append([
      { t: 0, gyro: [0, 0, 0], accel: [0, 0, 0], accelG: [0, 9.81, 0] },
      { t: 20, gyro: [0, 0, 0], accel: null, accelG: [0, 9.81, 0] },
    ]);
    log.append([
      { t: 10, gyro: [0.1, 0, 0], accel: null, accelG: [0, 9.81, 0] },
      { t: 20, gyro: [9, 9, 9], accel: null, accelG: null },
    ]);
    expect(log.samples.map((s) => s.t)).toEqual([0, 10, 20]);
    // upright portrait: specific force along +y ⇒ gravity points down = −y
    expect(gravityDown(log.samples, 10)).toEqual([-0, -1, -0]);
    const back = unpackImu(packImu(log.samples).buffer as ArrayBuffer);
    expect(back.length).toBe(3);
    expect(back[1].gyro[0]).toBeCloseTo(0.1);
    expect(back[1].accel).toBeNull();
  });
});

describe('engine: R_cam_device verification', () => {
  it('handEyeCheck ranks the hypotheses from GT poses + IMU', () => {
    const cap = makeCapture({ frames: 40, render: false });
    const log = new ImuLog();
    log.load(cap.imu.slice());
    const kfs = [];
    for (let i = 0; i < 12; i++) {
      const t = cap.opts.t0 + i * 250;
      kfs.push({ t, pose: cap.poseAt(t) });
    }
    const r = handEyeCheck(kfs, log)!;
    expect(r).not.toBeNull();
    expect(r.best).toBe(0);
    expect(r.medianErrDeg[0]).toBeLessThan(0.05);
    expect(Math.min(r.medianErrDeg[1], r.medianErrDeg[2], r.medianErrDeg[3])).toBeGreaterThan(1);
  });
});

describe('capture mode classification (auto)', () => {
  it('orbits converge on an object, walks look outward', () => {
    const cap = makeCapture({ frames: 40, render: false });
    const orbit = cap.frames.filter((_, i) => i % 4 === 0).map((f) => f.pose);
    expect(classifyScene(orbit, 1.4)).toBe('object');
    // sideways walk along x, looking at +y (a facade 10 m away) with mild panning
    const walk: Pose[] = [];
    for (let i = 0; i < 10; i++) {
      const R = Float64Array.of(1, 0, 0, 0, 0, -1, 0, 1, 0); // x right, y down (−z world), z forward (+y world)
      const C = [i * 0.5, 0, 1.5];
      walk.push({ R, t: Float64Array.of(-(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]), -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]), -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2])) });
    }
    expect(classifyScene(walk, 10)).toBe('scene');
    expect(classifyScene(walk.slice(0, 3), 10)).toBeNull();
  });
});

describe('persistence records', () => {
  it('SparseModel packs and unpacks', () => {
    const m = {
      cameras: new Map([[0, { width: 10, height: 8, f: 9, cx: 4.5, cy: 3.5, k1: 0.01, k2: -0.002 }]]),
      images: [{ id: 3, cameraId: 0, pose: { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(1, 2, 3) } }],
      numPoints: 2,
      xyz: Float64Array.of(1, 2, 3, 4, 5, 6),
      colors: Uint8Array.of(1, 2, 3, 4, 5, 6),
      errors: Float32Array.of(0.5, 0.25),
      trackOffsets: Uint32Array.of(0, 2, 4),
      trackImages: Uint32Array.of(3, 4, 3, 5),
      trackFeatures: Uint32Array.of(0, 1, 2, 3),
    };
    const b = unpackSparseModel(packSparseModel(m));
    expect(b.cameras.get(0)).toEqual(m.cameras.get(0));
    expect(Array.from(b.images[0].pose.t)).toEqual([1, 2, 3]);
    expect(Array.from(b.xyz)).toEqual(Array.from(m.xyz));
    expect(Array.from(b.trackFeatures)).toEqual([0, 1, 2, 3]);
    expect(Array.from(b.errors)).toEqual([0.5, 0.25]);
  });
});
