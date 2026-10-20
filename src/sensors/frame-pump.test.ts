import { describe, expect, it } from 'vitest';
import type { ImuSample } from '../core/types';
import { mergeGeoFixes } from '../engine/persistence';
import type { FrameCallback, FrameMeta } from './camera';
import { ScreenAngleTracker, type ScreenAngle } from './frame-orientation';
import { FramePump, type PumpFrame } from './frame-pump';
import { GeoSensor } from './geolocation';

class FakeBitmap {
  closed = false;
  constructor(
    readonly width: number,
    readonly height: number,
  ) {}
  close(): void {
    this.closed = true;
  }
}

class FakeCamera {
  cbs = new Set<FrameCallback>();
  w = 640;
  h = 480;
  grabs: FakeBitmap[] = [];
  meta: FrameMeta = { mediaTime: 0, width: 640, height: 480, callbackTime: 0, timeSource: 'captureTime', segment: 0 };
  onFrame(cb: FrameCallback): () => void {
    this.cbs.add(cb);
    return () => this.cbs.delete(cb);
  }
  grabFrame(): Promise<ImageBitmap> {
    const b = new FakeBitmap(this.w, this.h);
    this.grabs.push(b);
    return Promise.resolve(b as unknown as ImageBitmap);
  }
  emit(t: number): void {
    for (const cb of this.cbs) cb(t, this.meta);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

describe('FramePump', () => {
  it('limits the rate, applies back-pressure and passes the IMU samples since the last delivered frame', async () => {
    const cam = new FakeCamera();
    let imuCounter = 0;
    const motion = {
      drain(): ImuSample[] {
        const out: ImuSample[] = [];
        for (let k = 0; k < 3; k++) out.push({ t: imuCounter++, gyro: [0, 0, 0], accel: null, accelG: null });
        return out;
      },
      gravityAt: () => new Float64Array([0, -1, 0]),
      calibration: { version: 4 },
    };
    const geo = { nearest: (t: number) => ({ t: t - 100, lat: 1, lon: 2, alt: null, accuracy: 5, altitudeAccuracy: null }) };
    const compass = { headingAt: () => ({ heading: 12, accuracy: 7 }) };
    const tracker = new ScreenAngleTracker({ read: () => 0 as ScreenAngle });
    const frames: PumpFrame[] = [];
    let busy = 0;
    let maxBusy = 0;
    let release: (() => void) | null = null;
    const pump = new FramePump({
      camera: cam,
      motion,
      geo,
      compass,
      angleTracker: tracker,
      targetHz: 10,
      isHidden: () => false,
      consumer: async (f) => {
        busy++;
        maxBusy = Math.max(maxBusy, busy);
        frames.push(f);
        await new Promise<void>((r) => (release = r));
        busy--;
      },
    });
    pump.start();
    // 30 fps frames
    cam.emit(0);
    await sleep(0);
    expect(frames.length).toBe(1);
    expect(pump.busy).toBe(true);
    for (let i = 1; i <= 9; i++) cam.emit(i * 33.33);
    await sleep(0);
    expect(frames.length).toBe(1); // consumer still busy: everything skipped
    expect(pump.stats.skippedBusy).toBe(9);
    release!();
    await sleep(0);
    expect(pump.busy).toBe(false);
    cam.emit(10 * 33.33); // 333 ms after the last delivery → delivered
    await sleep(0);
    expect(frames.length).toBe(2);
    release!();
    await sleep(0);
    cam.emit(11 * 33.33); // 33 ms later → rate-limited
    cam.emit(12 * 33.33); // 67 ms → rate-limited
    await sleep(0);
    expect(frames.length).toBe(2);
    expect(pump.stats.skippedRate).toBe(2);
    cam.emit(13 * 33.33); // 100 ms (tolerance ½ frame) → delivered: 30 fps → 10 Hz
    await sleep(0);
    expect(frames.length).toBe(3);
    release!();
    await sleep(0);
    expect(maxBusy).toBe(1);

    const f = frames[1];
    expect(f.index).toBe(1);
    expect(f.width).toBe(640);
    expect(f.t).toBeCloseTo(333.3, 6);
    expect(f.screenAngle).toBe(0);
    expect(Array.from(f.R_cam_device)).toEqual([1, 0, 0, 0, -1, 0, 0, 0, -1]);
    expect(f.imu.map((s) => s.t)).toEqual([3, 4, 5]); // drained once per delivered frame
    expect(f.imuCalibrationVersion).toBe(4);
    expect(f.geo!.lat).toBe(1);
    expect(f.heading).toEqual({ heading: 12, accuracy: 7 });
    expect(Array.from(f.gravityDevice!)).toEqual([0, -1, 0]);
    expect(f.segment).toBe(0);
    pump.stop();
    cam.emit(5000);
    await sleep(0);
    expect(frames.length).toBe(3);
  });

  it('delivers the raw GPS fix nearest in time with its own timestamp (never restamped to the frame time)', async () => {
    const cam = new FakeCamera();
    const geo = new GeoSensor({ geolocation: null });
    // fix k at t = k·1000 ms (1 Hz), delivered 200 ms late
    const deliverFix = (k: number): void => void geo.handlePosition({ timestamp: k * 1000, coords: { latitude: 47 + k * 1e-5, longitude: 8, accuracy: 5, altitude: null, altitudeAccuracy: null } }, k * 1000 + 200, k * 1000 + 200);
    const frames: PumpFrame[] = [];
    const pump = new FramePump({ camera: cam, geo, angleTracker: new ScreenAngleTracker({ read: () => 0 }), targetHz: 10, isHidden: () => false, consumer: (f) => void frames.push(f) });
    pump.start();
    // 10 Hz frames over a 5 s walk
    for (let t = 0; t <= 5000; t += 100) {
      if (t % 1000 === 200) deliverFix((t - 200) / 1000);
      cam.emit(t);
      await sleep(0);
    }
    pump.stop();
    expect(frames).toHaveLength(51);
    // frame at 1300 ms: fixes at 0 and 1000 so far → the fix of 1000 ms as it is
    const f1300 = frames.find((f) => f.t === 1300)!;
    expect(f1300.geo).toEqual(geo.fixes[1]);
    expect(f1300.geo!.t).toBe(1000);
    // every delivered fix is one of the sensor's, and together they are all the fixes (no restamped entries)
    for (const f of frames) if (f.geo) expect(geo.fixes).toContainEqual(f.geo);
    expect(mergeGeoFixes([], frames.flatMap((f) => (f.geo ? [f.geo] : [])))).toEqual(geo.fixes);
  });

  it('skips frames right after a UI rotation and frames whose aspect contradicts it', async () => {
    const cam = new FakeCamera();
    let angle: ScreenAngle = 0;
    const tracker = new ScreenAngleTracker({ read: () => angle });
    const frames: PumpFrame[] = [];
    const pump = new FramePump({ camera: cam, angleTracker: tracker, targetHz: 1000, isHidden: () => false, consumer: (f) => void frames.push(f) });
    pump.start();
    cam.w = 480;
    cam.h = 640; // portrait UI ⇒ portrait frames
    cam.emit(1);
    await sleep(0);
    expect(frames.length).toBe(1);
    angle = 90; // UI rotated: frames within 500 ms are dropped
    tracker.poll();
    cam.emit(2);
    await sleep(0);
    expect(frames.length).toBe(1);
    expect(pump.stats.skippedOrientation).toBe(1);
    // pretend the settle time elapsed
    (tracker as unknown as { changeT: number }).changeT = -1e9;
    cam.emit(3); // still portrait pixels although the UI is landscape → transient, dropped
    await sleep(0);
    expect(frames.length).toBe(1);
    expect(pump.stats.skippedAspect).toBe(1);
    expect(cam.grabs[cam.grabs.length - 1].closed).toBe(true);
    cam.w = 640;
    cam.h = 480;
    cam.emit(4);
    await sleep(0);
    expect(frames.length).toBe(2);
    expect(frames[1].screenAngle).toBe(90);
    expect(Array.from(frames[1].R_cam_device)).toEqual([0, -1, 0, -1, 0, 0, 0, 0, -1]);
    pump.stop();
  });

  it('passes the time the UI rotation took its value, so the engine does not settle a second time', async () => {
    const cam = new FakeCamera();
    let angle: ScreenAngle = 0;
    const tracker = new ScreenAngleTracker({ read: () => angle });
    const frames: PumpFrame[] = [];
    const pump = new FramePump({ camera: cam, angleTracker: tracker, targetHz: 1000, isHidden: () => false, consumer: (f) => void frames.push(f) });
    pump.start();
    cam.w = 480;
    cam.h = 640;
    cam.emit(1);
    await sleep(0);
    angle = 90;
    const since = performance.now() - 600; // rotated 600 ms ago: settled
    tracker.poll(since);
    cam.w = 640;
    cam.h = 480;
    cam.emit(2);
    await sleep(0);
    pump.stop();
    expect(frames.map((f) => [f.screenAngle, f.screenAngleSince])).toEqual([
      [0, -Infinity],
      [90, since],
    ]);
  });

  it('closes the bitmap when the consumer throws and keeps running', async () => {
    const cam = new FakeCamera();
    const errors: unknown[] = [];
    const pump = new FramePump({
      camera: cam,
      angleTracker: new ScreenAngleTracker({ read: () => 0 }),
      targetHz: 1000,
      isHidden: () => false,
      onError: (e) => errors.push(e),
      consumer: () => {
        throw new Error('boom');
      },
    });
    pump.start();
    cam.emit(1);
    await sleep(0);
    expect(errors.length).toBe(1);
    expect(cam.grabs[0].closed).toBe(true);
    expect(pump.busy).toBe(false);
    cam.emit(100);
    await sleep(0);
    expect(errors.length).toBe(2);
    pump.stop();
  });

  it("grab 'videoFrame': delivers the camera's VideoFrame as a raw buffer in every UI rotation, drops portrait ones", async () => {
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
    const cam = new FakeCamera();
    const vfs: FakeVideoFrame[] = [];
    let portrait = false;
    const camera = Object.assign(cam, {
      result: { frameMaxSide: 1280 },
      grabVideoFrame: (): VideoFrame => {
        const f = portrait ? new FakeVideoFrame(1440, 1920) : new FakeVideoFrame(1920, 1440);
        vfs.push(f);
        return f as unknown as VideoFrame;
      },
    });
    let angle: ScreenAngle = 0;
    const tracker = new ScreenAngleTracker({ read: () => angle });
    const frames: PumpFrame[] = [];
    const pump = new FramePump({ camera, grab: 'videoFrame', angleTracker: tracker, targetHz: 1000, isHidden: () => false, consumer: (f) => void frames.push(f) });
    pump.start();
    cam.emit(1);
    await sleep(0);
    angle = 90;
    tracker.poll();
    (tracker as unknown as { changeT: number }).changeT = -1e9;
    cam.emit(2);
    await sleep(0);
    expect(frames.map((f) => [f.width, f.height, f.screenAngle, f.rawBuffer, f.maxSide])).toEqual([
      [1920, 1440, 0, true, 1280],
      [1920, 1440, 90, true, 1280],
    ]);
    expect(frames[0].bitmap).toBe(vfs[0]);
    expect(cam.grabs).toHaveLength(0); // never createImageBitmap(<video>)
    expect(pump.stats.grabMs).toBeGreaterThanOrEqual(0);
    portrait = true;
    cam.emit(3);
    await sleep(0);
    expect(frames).toHaveLength(2);
    expect(pump.stats.skippedAspect).toBe(1);
    expect(vfs[2].closed).toBe(true); // a dropped VideoFrame releases its camera buffer
    pump.stop();
    // 'upright' VideoFrames (Chromium): not raw, aspect checked like bitmaps
    const up: PumpFrame[] = [];
    portrait = false;
    angle = 0;
    tracker.poll();
    (tracker as unknown as { changeT: number }).changeT = -1e9;
    const pump2 = new FramePump({ camera, grab: 'videoFrame', videoFrameOrientation: 'upright', angleTracker: tracker, targetHz: 1000, isHidden: () => false, consumer: (f) => void up.push(f) });
    pump2.start();
    cam.emit(10);
    await sleep(0);
    expect(up.map((f) => [f.rawBuffer, f.maxSide])).toEqual([[false, 1280]]);
    pump2.stop();
  });

  it('skips while hidden', async () => {
    const cam = new FakeCamera();
    let hidden = true;
    const frames: PumpFrame[] = [];
    const pump = new FramePump({
      camera: cam,
      angleTracker: new ScreenAngleTracker({ read: () => 0 }),
      isHidden: () => hidden,
      consumer: (f) => void frames.push(f),
    });
    pump.start();
    cam.emit(0);
    await sleep(0);
    expect(frames.length).toBe(0);
    expect(pump.stats.skippedHidden).toBe(1);
    hidden = false;
    cam.emit(200);
    await sleep(0);
    expect(frames.length).toBe(1);
    pump.stop();
  });
});
