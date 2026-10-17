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
 * (transient; re-baselined after 1.5 s of consistent contradiction) — unless the camera reports
 * `fixedOrientation` (raw sensor buffers, which never follow the UI rotation): those frames are
 * marked `rawBuffer`, and a portrait one (the source rotated after all) is dropped instead.
 * A grab rejected with a {@link StaleFrameError} (the frame was already replaced) is not an
 * error and does not use up the rate-limit slot: the next frame is tried at once.
 * The consumer OWNS the bitmap (close or transfer it).
 */
import type { GeoFix, ImuSample, Mat3, Vec3 } from '../core/types';
import type { FrameCallback, FrameMeta } from './camera';
import { ScreenAngleTracker, cameraFromDevice, type ScreenAngle } from './frame-orientation';
import type { AttitudeReading, HeadingReading } from './orientation';

export interface PumpFrame {
  /** 0-based index of delivered frames. */
  index: number;
  /** Full-resolution frame (or ≤ maxSide), upright for `screenAngle`. Owned by the consumer. */
  bitmap: ImageBitmap;
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
  /** Raw sensor-orientation buffer (camera `fixedOrientation`): upright for θ = 90 in any UI rotation. */
  rawBuffer: boolean;
}

/** grabFrame() rejection: the reported frame was replaced before it could be grabbed (retry the next one). */
export class StaleFrameError extends Error {
  override readonly name = 'StaleFrameError';
}

export interface FramePumpCamera {
  onFrame(cb: FrameCallback): () => void;
  grabFrame(opts?: { maxSide?: number }): Promise<ImageBitmap>;
  /** CameraCapture.result: its `frameMaxSide` is used when `maxSide` is left undefined. */
  readonly result?: { frameMaxSide: number | null } | null;
  /**
   * Frames keep one orientation whatever the UI rotation (raw sensor buffers of the MSTP path,
   * ./mstp-camera.ts): the aspect-vs-rotation check is skipped (the engine classifies them).
   */
  readonly fixedOrientation?: boolean;
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
  /** Grabs whose frame had already been replaced (StaleFrameError). */
  skippedStale: number;
  aspectRebaselines: number;
  errors: number;
  /** Exponential moving average of the consumer time (ms). */
  consumerMs: number;
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
    skippedStale: 0,
    aspectRebaselines: 0,
    errors: 0,
    consumerMs: 0,
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
    // Counted at the attempt so that failing grabs are retried at targetHz, not on every frame
    // (a stale grab gives the slot back: deliver() restores prevT).
    const prevT = this.lastDeliveredT;
    this.lastDeliveredT = t;
    // grabFrame() is invoked synchronously here (async functions run until their first await),
    // so the bitmap is the frame this callback reported.
    let grab: Promise<ImageBitmap>;
    try {
      const ms = this.o.maxSide === undefined ? (this.o.camera.result?.frameMaxSide ?? null) : this.o.maxSide;
      grab = this.o.camera.grabFrame(ms ? { maxSide: ms } : undefined);
    } catch (e) {
      grab = Promise.reject(e);
    }
    void this.deliver(grab, t, meta.timeSource, meta.segment, theta, this.runGen, prevT);
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
    grab: Promise<ImageBitmap>,
    t: number,
    timeSource: FrameMeta['timeSource'],
    segment: number,
    theta: ScreenAngle,
    gen: number,
    prevT: number,
  ): Promise<void> {
    let bitmap: ImageBitmap | null = null;
    let handedOut = false;
    const raw = this.o.camera.fixedOrientation === true;
    try {
      bitmap = await grab;
      if (gen !== this.runGen) {
        // stop() was called while grabbing: never deliver after stop (and never leak the bitmap).
        bitmap.close();
        bitmap = null;
        return;
      }
      if (raw ? bitmap.height > bitmap.width : !this.checkAspect(bitmap.width, bitmap.height, theta)) {
        if (raw) console.warn('[sensors] portrait frame from a raw-buffer camera path: dropped', { w: bitmap.width, h: bitmap.height });
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
        width: bitmap.width,
        height: bitmap.height,
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
      bitmap?.close();
      if (!handedOut && e instanceof StaleFrameError) {
        this.stats.skippedStale++;
        if (this.lastDeliveredT === t) this.lastDeliveredT = prevT; // give the rate-limit slot back
        return;
      }
      this.stats.errors++;
      if (!handedOut && gen !== this.runGen) return; // failure of a grab abandoned by stop()
      if (this.o.onError) this.o.onError(e);
      else console.error('[sensors] FramePump:', e);
    } finally {
      this.busyFlag = false;
    }
  }
}
