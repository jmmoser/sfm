/**
 * Browser tests (headless Chromium, fake camera via --use-fake-device-for-media-stream,
 * --use-fake-ui-for-media-stream, SwiftShader WebGPU).
 */
import { afterAll, describe, expect, it } from 'vitest';
import { CameraCapture, type FrameMeta } from './camera';
import { runCameraTimingProbe } from './camera-probe';
import { collectDiagnostics } from './diagnostics';
import { ScreenAngleTracker, screenAngle } from './frame-orientation';
import { FramePump, type PumpFrame } from './frame-pump';
import { MotionSensor } from './motion';
import { requestCapturePermissions } from './permissions';
import { WakeLock } from './wakelock';

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function makeVideo(): HTMLVideoElement {
  const v = document.createElement('video');
  v.style.width = '160px';
  document.body.appendChild(v);
  return v;
}

const videos: HTMLVideoElement[] = [];
afterAll(() => {
  for (const v of videos) v.remove();
});

describe('CameraCapture (fake camera)', () => {
  it('lists cameras with a lens classification', async () => {
    const st = await requestCapturePermissions({ geolocation: false });
    expect(st.camera).toBe('granted');
    expect(st.motion).toBe('granted'); // no permission model in Chromium
    const cams = await CameraCapture.listCameras();
    expect(cams.length).toBeGreaterThan(0);
    expect(cams[0].label).toBe('fake_device_0');
    expect(cams[0].lens).toBe('unknown');
  });

  it('start / onFrame / grabFrame / controls / stop', async () => {
    const video = makeVideo();
    videos.push(video);
    const cam = new CameraCapture();
    const states: string[] = [];
    cam.onStateChange((s) => states.push(s));
    const res = await cam.start(video, { quality: 'fast' });
    try {
    expect(states).toContain('running');
    // non-Apple 'fast': 1280×960 is not a native fake-camera format (resizeMode 'none') → 1280×720
    expect(res.width).toBe(1280);
    expect(res.height).toBe(720);
    expect(res.aspect).toBe('16:9');
    expect(res.requested).toEqual({ width: 1280, height: 720 });
    expect(res.attempts[0]).toMatch(/exact 1280x960: OverconstrainedError/);
    expect(res.settings.resizeMode).toBe('none');
    expect(res.label).toBe('fake_device_0');
    expect(res.apple).toBe(false);
    expect(res.hfovPriorDeg).toBe(63);
    expect(res.focalPriorPx).toBeCloseTo(640 / Math.tan((63 * Math.PI) / 360), 6);
    expect(video.muted).toBe(true);
    expect(video.playsInline).toBe(true);
    expect(cam.state).toBe('running');

    const ts: number[] = [];
    const metas: FrameMeta[] = [];
    const off = cam.onFrame((t, m) => {
      ts.push(t);
      metas.push({ ...m });
    });
    const t0 = performance.now();
    while (ts.length < 6 && performance.now() - t0 < 5000) await sleep(20);
    expect(ts.length).toBeGreaterThanOrEqual(6);
    for (let i = 1; i < ts.length; i++) expect(ts[i]).toBeGreaterThan(ts[i - 1]);
    expect(metas[0].width).toBe(1280);
    expect(metas[0].timeSource).toBe('captureTime');
    expect(metas[0].captureTime!).toBeLessThanOrEqual(metas[0].callbackTime);

    const bmp = await cam.grabFrame();
    expect([bmp.width, bmp.height]).toEqual([1280, 720]);
    bmp.close();
    const small = await cam.grabFrame({ maxSide: 320 });
    expect([small.width, small.height]).toEqual([320, 180]);
    small.close();

    expect(await cam.setTorch(true)).toBe(false); // fake camera has no torch
    expect(await cam.setZoom(2)).toBe(false); // nor zoom
    expect(await cam.lockFocus()).toBe(true); // manual focus at the current distance
    expect(cam.getSettings().focusMode).toBe('manual');
    // The fake device lists 'continuous' but stays in 'manual': the result must reflect reality.
    const unlocked = await cam.unlockFocus();
    expect(cam.getSettings().focusMode).toBe(unlocked ? 'continuous' : 'manual');

    const track = cam.track!;
    cam.stop();
    expect(track.readyState).toBe('ended');
    expect(cam.state).toBe('stopped');
    const n = ts.length;
    await sleep(200);
    expect(ts.length).toBe(n);
    off();
    await expect(cam.grabFrame()).rejects.toThrow(/no video frame/);
    } finally {
      cam.stop();
    }
  });

  it("'max' picks the largest native format and restart() opens a new segment", async () => {
    const video = makeVideo();
    videos.push(video);
    const cam = new CameraCapture();
    try {
      const res = await cam.start(video, { quality: 'max' });
      expect(res.attempts.slice(0, 3)).toEqual([
        'exact 2592x1944: OverconstrainedError',
        'exact 1920x1440: OverconstrainedError',
        'exact 3840x2160: ok',
      ]);
      expect(res.width).toBe(3840);
      expect(res.height).toBe(2160);
      expect(cam.segment).toBe(0);
      expect(await cam.restart('test')).toBe(true);
      expect(cam.segment).toBe(1);
      expect(cam.state).toBe('running');
      expect(video.videoWidth).toBe(3840);
    } finally {
      cam.stop();
    }
  });
});

describe('CameraCapture lifecycle races', () => {
  it('stop() during start() aborts the start and releases the camera', async () => {
    const video = makeVideo();
    videos.push(video);
    const cam = new CameraCapture();
    const p = cam.start(video, { quality: 'fast' });
    cam.stop();
    await expect(p).rejects.toThrow(/superseded/);
    expect(cam.track).toBeNull();
    // a fresh start works afterwards
    const res = await cam.start(video, { quality: 'fast' });
    expect(res.width).toBe(1280);
    cam.stop();
  });
});

describe('FramePump (fake camera + synthetic DeviceMotion events)', () => {
  it('delivers frames with back-pressure and drains the IMU per frame', async () => {
    const video = makeVideo();
    videos.push(video);
    const cam = new CameraCapture();
    await cam.start(video, { quality: 'fast' });
    const motion = new MotionSensor();
    motion.start();
    let k = 0;
    const timer = setInterval(() => {
      k++;
      window.dispatchEvent(
        new DeviceMotionEvent('devicemotion', {
          rotationRate: { alpha: 10 + 0.1 * k, beta: 0.5, gamma: -0.25 },
          acceleration: { x: 0, y: 0, z: 0 },
          accelerationIncludingGravity: { x: 0.01 * k, y: 9.8, z: 0.2 },
          interval: 16,
        }),
      );
    }, 16);
    const frames: PumpFrame[] = [];
    let busy = 0;
    let maxBusy = 0;
    const pump = new FramePump({
      camera: cam,
      motion,
      targetHz: 30,
      maxSide: 640,
      consumer: async (f) => {
        busy++;
        maxBusy = Math.max(maxBusy, busy);
        frames.push(f);
        await sleep(150); // slow consumer
        f.bitmap.close();
        busy--;
      },
    });
    pump.start();
    await sleep(1500);
    pump.stop();
    clearInterval(timer);
    await sleep(200);
    motion.stop();
    cam.stop();

    expect(frames.length).toBeGreaterThanOrEqual(4);
    expect(maxBusy).toBe(1);
    expect(pump.stats.skippedBusy).toBeGreaterThan(0);
    for (let i = 1; i < frames.length; i++) {
      expect(frames[i].t).toBeGreaterThan(frames[i - 1].t);
      expect(frames[i].t - frames[i - 1].t).toBeGreaterThanOrEqual(140);
    }
    expect(frames[0].width).toBe(640);
    expect(frames[0].height).toBe(360);
    expect(frames[0].screenAngle).toBe(0);
    // every accepted IMU sample is delivered exactly once, in order
    const imu = frames.flatMap((f) => f.imu);
    expect(imu.length).toBeGreaterThan(20);
    for (let i = 1; i < imu.length; i++) expect(imu[i].t).toBeGreaterThan(imu[i - 1].t);
    expect(imu[0].gyro[1]).toBeCloseTo((0.5 * Math.PI) / 180, 12);
    expect(imu[0].accelG![1]).toBeCloseTo(9.8, 9); // spec-sign platform, prior +1
    expect(motion.rateHz).toBeGreaterThan(20);
  });
});

describe('misc browser glue', () => {
  it('diagnostics report', async () => {
    const motion = new MotionSensor();
    const d = await collectDiagnostics({ motion, gpu: 'adapter' });
    expect(d.features.getUserMedia).toBe(true);
    expect(d.features.requestVideoFrameCallback).toBe(true);
    expect(d.features.deviceMotion).toBe(true);
    expect(d.features.motionPermissionApi).toBe(false);
    expect(d.features.wasmSimdKernels).toBe(true); // compiled + passed the runtime self-test
    expect(d.appleMobileWebKit).toBe(false);
    expect(d.cameras!.length).toBeGreaterThan(0);
    expect(d.gpu?.available).toBe(true);
    expect(Object.keys(d.gpu!.limits!).length).toBeGreaterThan(10);
    expect(d.motion!.rateHz).toBe(0);
    expect(d.hardwareConcurrency).toBeGreaterThan(0);
    expect(d.screen.screenAngle).toBe(0);
    expect(JSON.parse(JSON.stringify(d)).features.webgpu).toBe(true);
    const d2 = await collectDiagnostics({ gpu: 'device', listCameras: false });
    expect(d2.gpu?.source).toBe('device');
    expect(d2.gpu?.isSoftware).toBe(true);
  });

  it('wake lock never throws; screen angle tracker works', async () => {
    const wl = new WakeLock();
    await wl.acquire();
    expect(typeof wl.active).toBe('boolean');
    await wl.release();
    expect(wl.active).toBe(false);
    const tr = new ScreenAngleTracker();
    tr.start();
    expect(tr.angle).toBe(screenAngle());
    tr.stop();
  });
});

describe('camera timing probe (fake camera, real worker)', () => {
  it('records rVFC and MSTP frames on the same track, pairs them and times the grabs', async () => {
    const video = document.createElement('video');
    document.body.appendChild(video);
    videos.push(video);
    const cam = new CameraCapture();
    await cam.start(video, { quality: 'fast' });
    try {
      const r = await runCameraTimingProbe(cam, { durationMs: 1500 });
      expect(r.mstp.error).toBeNull();
      expect(r.mstp.transport).toBe('stream'); // Chromium: tracks are not transferable
      expect(r.mstp.frames).toBeGreaterThan(10);
      expect(r.video.frames).toBeGreaterThan(10);
      expect(r.video.timeSource).toBe('captureTime');
      expect(r.timing.pairs).toBeGreaterThan(5);
      expect(r.timing.jitterStdMs).not.toBeNull();
      expect(r.grab.syncMs!.n).toBeGreaterThan(5);
      expect(r.summary.length).toBeGreaterThan(0);
      expect(cam.track!.readyState).toBe('live'); // the probe leaves the camera running
    } finally {
      cam.stop();
    }
  });
});
