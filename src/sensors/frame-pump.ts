/**
 * FramePump: glue between CameraCapture and the engine (docs/ENGINE.md §1, §2 FrameInput).
 *
 * While running it watches presented frames (CameraCapture.onFrame), and at most `targetHz`
 * times per second — and never while the consumer is still busy (back-pressure: the consumer
 * returns a promise) — grabs the CURRENT frame as an ImageBitmap (synchronously inside the
 * frame callback, so bitmap and timestamp belong together) and delivers it with the IMU
 * samples since the previous delivered frame, the GPS fix, compass heading and attitude.
 *
 * Frames are skipped while the page is hidden, within `settleMs` (500 ms) of a UI rotation
 * (Safari 26.x transient mis-rotation) and when the bitmap aspect contradicts the UI rotation
 * (transient; re-baselined after 1.5 s of consistent contradiction).
 * The consumer OWNS the bitmap (close or transfer it).
 *
 * `grab: 'videoFrame'` (the "VideoFrame grab" setting): instead of createImageBitmap(<video>),
 * which blocks the main thread ~15 ms per grab on iOS (Diagnostics camera timing probe), the pump
 * takes `camera.grabVideoFrame()` (~0 ms: a reference to the camera buffer) and delivers it as
 * `bitmap` with `rawBuffer` (on iOS it keeps the sensor orientation in every UI rotation; the
 * engine treats it as upright for θ = 90 and converts it, downscaled to `maxSide`). Its aspect
 * does not follow the UI, so the aspect check is replaced by "a raw frame must be landscape".
 * Elsewhere (`videoFrameOrientation: 'upright'`, e.g. Chromium) VideoFrames are upright like the
 * <video>: not raw, and checked like bitmaps.
 * `stats.grabMs` = main-thread time of the grab call, whichever path.
 */
import type { GeoFix, ImuSample, Mat3, Vec3 } from '../core/types';
import type { FrameCallback, FrameMeta } from './camera';
import { ScreenAngleTracker, cameraFromDevice, type ScreenAngle } from './frame-orientation';
import type { AttitudeReading, HeadingReading } from './orientation';

export interface PumpFrame {
  /** 0-based index of delivered frames. */
  index: number;
  /**
   * Full-resolution frame (or ≤ maxSide), upright for `screenAngle` — or, with `grab:
   * 'videoFrame'`, a VideoFrame in sensor orientation (`rawBuffer`). Owned by the consumer.
   */
  bitmap: ImageBitmap | VideoFrame;
  width: number;
  height: number;
  /** Best capture-time estimate (ms, performance.now() domain). */
  t: number;
  timeSource: FrameMeta['timeSource'];
  /** UI rotation θ (ARCHITECTURE §2) of the frame. */
  screenAngle: ScreenAngle;
  /** cameraFromDevice(screenAngle). */
  R_cam_device: Mat3;
  /** Normalised IMU samples since the previous delivered frame (may extend slightly past t). */
  imu: ImuSample[];
  /** MotionSensor calibration version the samples were normalised with (-1 without motion). */
  imuCalibrationVersion: number;
  geo: GeoFix | null;
  heading: HeadingReading | null;
  attitude: AttitudeReading | null;
  /** Unit gravity direction (DOWN) in the device frame at t, if known. */
  gravityDevice: Vec3 | null;
  /** Camera capture segment (changes after an interruption ⇒ relocalise / new segment). */
  segment: number;
  /** `bitmap` is a raw sensor-orientation VideoFrame (upright for θ = 90 in any UI rotation). */
  rawBuffer: boolean;
  /** VideoFrame only: the size to downscale to when converting (null = native). */
  maxSide: number | null;
}

export interface FramePumpCamera {
  onFrame(cb: FrameCallback): () => void;
  grabFrame(opts?: { maxSide?: number }): Promise<ImageBitmap>;
  /** The current frame as a VideoFrame (synchronous; throws without a frame or VideoFrame support). */
  grabVideoFrame?(): VideoFrame;
  /** CameraCapture.result: its `frameMaxSide` is used when `maxSide` is left undefined. */
  readonly result?: { frameMaxSide: number | null } | null;
}
export interface FramePumpMotion {
  drain(): ImuSample[];
  gravityAt?(t: number): Vec3 | null;
  readonly calibration?: { version: number };
}
export interface FramePumpGeo {
  nearest(t: number): GeoFix | null;
  interpolate?(t: number): GeoFix | null;
}
export interface FramePumpCompass {
  headingAt(t: number): HeadingReading | null;
  attitudeAt?(t: number): AttitudeReading | null;
}

export interface FramePumpOptions {
  camera: FramePumpCamera;
  /** Receives each frame; the pump waits for the returned promise before delivering another. */
  consumer: (f: PumpFrame) => unknown;
  motion?: FramePumpMotion | null;
  geo?: FramePumpGeo | null;
  compass?: FramePumpCompass | null;
  /** Max delivery rate (default 10 Hz). */
  targetHz?: number;
  /** 'bitmap' (default): createImageBitmap(<video>); 'videoFrame': camera.grabVideoFrame(). */
  grab?: 'bitmap' | 'videoFrame';
  /**
   * How the VideoFrames are oriented: 'raw' (default; iOS WebKit, verified by the Diagnostics probe:
   * sensor orientation in every UI rotation) or 'upright' (like the <video>: e.g. Chromium).
   */
  videoFrameOrientation?: 'raw' | 'upright';
  /**
   * Downscale grabbed frames so max(W,H) ≤ maxSide. undefined (default) = the camera's
   * suggestion (`CameraStartResult.frameMaxSide`, e.g. 1280 for the 'fast' preset on iOS where
   * the stream is a native 1920×1440); null = always native size.
   */
  maxSide?: number | null;
  /** UI-rotation tracker (default: an internal one, started/stopped with the pump). */
  angleTracker?: ScreenAngleTracker | null;
  /** Drop frames this long after a UI rotation (default 500 ms). */
  settleMs?: number;
  /** Ignore GPS fixes further than this from the frame time (default 30000 ms). */
  geoMaxAgeMs?: number;
  isHidden?: () => boolean;
  onError?: (e: unknown) => void;
}

export interface FramePumpStats {
  framesSeen: number;
  delivered: number;
  skippedRate: number;
  skippedBusy: number;
  skippedHidden: number;
  skippedOrientation: number;
  skippedAspect: number;
  aspectRebaselines: number;
  errors: number;
  /** Exponential moving average of the consumer time (ms). */
  consumerMs: number;
  /** Exponential moving average of the main-thread time of the grab call (ms). */
  grabMs: number;
  /** Estimated camera frame period (ms). */
  framePeriodMs: number;
}

export class FramePump {
  private readonly o: FramePumpOptions;
  private readonly tracker: ScreenAngleTracker;
  private readonly ownTracker: boolean;
  private unsub: (() => void) | null = null;
  private busyFlag = false;
  /** Incremented by stop(): a grab started in an earlier run is discarded, not delivered. */
  private runGen = 0;
  /** Frame time of the last grab ATTEMPT (rate limit; failed/skipped grabs count too). */
  private lastDeliveredT = -Infinity;
  private lastFrameT = -Infinity;
  private idx = 0;
  private hz: number;
  private aspectRef: { landscape: boolean; theta: number } | null = null;
  private mismatchSince = -Infinity;
  readonly stats: FramePumpStats = {
    framesSeen: 0,
    delivered: 0,
    skippedRate: 0,
    skippedBusy: 0,
    skippedHidden: 0,
    skippedOrientation: 0,
    skippedAspect: 0,
    aspectRebaselines: 0,
    errors: 0,
    consumerMs: 0,
    grabMs: 0,
    framePeriodMs: 1000 / 30,
  };

  constructor(opts: FramePumpOptions) {
    this.o = opts;
    this.hz = opts.targetHz ?? 10;
    this.ownTracker = !opts.angleTracker;
    this.tracker = opts.angleTracker ?? new ScreenAngleTracker({ settleMs: opts.settleMs ?? 500 });
  }

  get running(): boolean {
    return this.unsub !== null;
  }
  /** True while the consumer processes a frame. */
  get busy(): boolean {
    return this.busyFlag;
  }
  get targetHz(): number {
    return this.hz;
  }

  setTargetHz(hz: number): void {
    if (hz > 0 && Number.isFinite(hz)) this.hz = hz;
  }

  start(): void {
    if (this.unsub) return;
    if (this.ownTracker) this.tracker.start();
    this.unsub = this.o.camera.onFrame(this.onFrame);
  }

  /**
   * Stop delivering. A frame already handed to the consumer is not affected; a frame still being
   * grabbed is discarded (its bitmap closed, IMU samples left buffered).
   */
  stop(): void {
    this.runGen++;
    this.unsub?.();
    this.unsub = null;
    if (this.ownTracker) this.tracker.stop();
  }

  /** Reset the aspect/orientation reference (e.g. new capture session). */
  resetReference(): void {
    this.aspectRef = null;
    this.mismatchSince = -Infinity;
  }

  private hidden(): boolean {
    if (this.o.isHidden) return this.o.isHidden();
    return typeof document !== 'undefined' && document.hidden === true;
  }

  private readonly onFrame = (t: number, meta: FrameMeta): void => {
    const st = this.stats;
    st.framesSeen++;
    if (Number.isFinite(this.lastFrameT)) {
      const dt = t - this.lastFrameT;
      if (dt > 5 && dt < 250) st.framePeriodMs += 0.1 * (dt - st.framePeriodMs);
    }
    this.lastFrameT = t;
    if (!this.unsub) return;
    if (this.hidden()) {
      st.skippedHidden++;
      return;
    }
    if (this.busyFlag) {
      st.skippedBusy++;
      return;
    }
    // Tolerance of half a frame so 30 fps → 10 Hz delivers every 3rd frame (not every 4th).
    const minInterval = 1000 / this.hz - 0.5 * st.framePeriodMs;
    if (t - this.lastDeliveredT < minInterval) {
      st.skippedRate++;
      return;
    }
    const settleMs = this.o.settleMs ?? 500;
    if (!this.tracker.isSettled(performance.now(), settleMs)) {
      st.skippedOrientation++;
      return;
    }
    const theta = this.tracker.angle;
    this.busyFlag = true;
    // Counted at the attempt so that failing grabs are retried at targetHz, not on every frame.
    this.lastDeliveredT = t;
    // grabFrame() / grabVideoFrame() are invoked synchronously here (async functions run until
    // their first await), so the frame is the one this callback reported.
    let grab: Promise<ImageBitmap | VideoFrame>;
    const ms = this.o.maxSide === undefined ? (this.o.camera.result?.frameMaxSide ?? null) : this.o.maxSide;
    const g0 = performance.now();
    try {
      const cam = this.o.camera;
      grab = this.o.grab === 'videoFrame' && cam.grabVideoFrame ? Promise.resolve(cam.grabVideoFrame()) : cam.grabFrame(ms ? { maxSide: ms } : undefined);
    } catch (e) {
      grab = Promise.reject(e);
    }
    const gms = performance.now() - g0;
    st.grabMs = st.grabMs === 0 ? gms : st.grabMs + 0.2 * (gms - st.grabMs);
    void this.deliver(grab, t, meta.timeSource, meta.segment, theta, this.runGen, ms);
  };

  private checkAspect(w: number, h: number, theta: number): boolean {
    if (w === h) return true;
    const landscape = w > h;
    const ref = this.aspectRef;
    if (!ref) {
      this.aspectRef = { landscape, theta };
      return true;
    }
    const odd = (((theta - ref.theta) / 90) % 2 + 2) % 2 === 1;
    const expected = odd ? !ref.landscape : ref.landscape;
    if (landscape === expected) {
      this.mismatchSince = -Infinity;
      return true;
    }
    const now = performance.now();
    if (!Number.isFinite(this.mismatchSince)) this.mismatchSince = now;
    if (now - this.mismatchSince < 1500) return false;
    console.warn('[sensors] frame aspect does not follow the UI rotation; re-baselining', { w, h, theta, ref });
    this.aspectRef = { landscape, theta };
    this.mismatchSince = -Infinity;
    this.stats.aspectRebaselines++;
    return true;
  }

  private async deliver(
    grab: Promise<ImageBitmap | VideoFrame>,
    t: number,
    timeSource: FrameMeta['timeSource'],
    segment: number,
    theta: ScreenAngle,
    gen: number,
    maxSide: number | null,
  ): Promise<void> {
    let bitmap: ImageBitmap | VideoFrame | null = null;
    let handedOut = false;
    try {
      bitmap = await grab;
      if (gen !== this.runGen) {
        // stop() was called while grabbing: never deliver after stop (and never leak the bitmap).
        bitmap.close();
        bitmap = null;
        return;
      }
      const b = bitmap;
      const raw = isVideoFrame(b) && (this.o.videoFrameOrientation ?? 'raw') === 'raw';
      const w = isVideoFrame(b) ? b.displayWidth : b.width;
      const h = isVideoFrame(b) ? b.displayHeight : b.height;
      // a raw buffer never follows the UI rotation: it must be (sensor-)landscape instead
      if (raw ? h > w : !this.checkAspect(w, h, theta)) {
        this.stats.skippedAspect++;
        bitmap.close();
        bitmap = null;
        return;
      }
      const m = this.o.motion;
      const imu = m ? m.drain() : [];
      let geo: GeoFix | null = null;
      if (this.o.geo) {
        geo = this.o.geo.interpolate?.(t) ?? this.o.geo.nearest(t);
        if (geo && Math.abs(geo.t - t) > (this.o.geoMaxAgeMs ?? 30000)) geo = null;
      }
      const c = this.o.compass;
      const frame: PumpFrame = {
        index: this.idx++,
        bitmap,
        width: w,
        height: h,
        t,
        timeSource,
        screenAngle: theta,
        R_cam_device: cameraFromDevice(theta),
        imu,
        imuCalibrationVersion: m?.calibration?.version ?? -1,
        geo,
        heading: c ? c.headingAt(t) : null,
        attitude: c?.attitudeAt ? c.attitudeAt(t) : null,
        gravityDevice: m?.gravityAt ? m.gravityAt(t) : null,
        segment,
        rawBuffer: raw,
        maxSide: isVideoFrame(b) ? maxSide : null,
      };
      this.stats.delivered++;
      const t0 = performance.now();
      const handed = bitmap;
      bitmap = null; // ownership passes to the consumer
      handedOut = true;
      try {
        await this.o.consumer(frame);
      } catch (e) {
        try {
          handed.close();
        } catch {
          /* detached */
        }
        throw e;
      }
      const dt = performance.now() - t0;
      this.stats.consumerMs = this.stats.consumerMs === 0 ? dt : this.stats.consumerMs + 0.2 * (dt - this.stats.consumerMs);
    } catch (e) {
      this.stats.errors++;
      bitmap?.close();
      if (!handedOut && gen !== this.runGen) return; // failure of a grab abandoned by stop()
      if (this.o.onError) this.o.onError(e);
      else console.error('[sensors] FramePump:', e);
    } finally {
      this.busyFlag = false;
    }
  }
}

function isVideoFrame(b: ImageBitmap | VideoFrame): b is VideoFrame {
  return typeof (b as VideoFrame).displayWidth === 'number' && typeof (b as ImageBitmap).width !== 'number';
}
