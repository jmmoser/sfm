/**
 * MediaStreamTrackProcessor capture path (docs/research/ios-camera.md §7, ladder step A):
 * a FramePump camera that reads frames in a dedicated worker instead of the `<video>` element.
 *
 *  - A clone of CameraCapture's track is transferred to src/workers/camera.worker.ts, which reads
 *    it with MediaStreamTrackProcessor and announces every frame. Where tracks are not transferable
 *    (Chromium) but MSTP exists on the main thread, the processor is created here and its
 *    ReadableStream is transferred instead (WebKit's MSTP is worker-only, so iOS uses the former). `onFrame` subscribers get those
 *    announcements; `grabFrame` (called synchronously from the callback, as FramePump does) asks
 *    the worker for exactly that frame, converted to an ImageBitmap off the main thread. The
 *    `<video>` element keeps showing the original track as the preview.
 *  - Frame time comes from `VideoFrame.timestamp` (capture clock, µs), mapped onto the
 *    performance.now() timeline by {@link FrameClock}. rVFC `captureTime` is the post-ISP
 *    arrival time instead (ios-camera.md §7 step 7).
 *  - iOS frames keep the sensor orientation (landscape buffers in every UI rotation,
 *    `fixedOrientation`): the engine's OrientationTracker and the preview overlay classify them
 *    as raw sensor buffers. FramePump's aspect-vs-rotation check is therefore skipped.
 *  - Fallback: when the worker cannot read the track (no MSTP in workers, a non-transferable
 *    track, no frame within the timeouts) BEFORE any frame was delivered, the source forwards to
 *    CameraCapture's rVFC path for good (`mode` 'fallback'). After frames were delivered it never
 *    switches path (the engine fixed its raw-buffer decision on them); it re-attaches instead.
 *  - Camera restarts: the worker's clone is stopped while CameraCapture restarts or stops (a live
 *    clone would keep the old capture source open) and a clone of the new track is attached once
 *    the camera runs again.
 */
import type { CameraStartResult, CameraState, FrameCallback, FrameMeta } from './camera';
import type { FramePumpCamera } from './frame-pump';
import type { CameraWorkerEvent, CameraWorkerRequest } from './mstp-protocol';

/** The parts of CameraCapture this source uses. */
export interface MstpCamera {
  readonly track: MediaStreamTrack | null;
  readonly segment: number;
  readonly state: CameraState;
  readonly result: CameraStartResult | null;
  onFrame(cb: FrameCallback): () => void;
  grabFrame(opts?: { maxSide?: number; resizeQuality?: ResizeQuality }): Promise<ImageBitmap>;
  onStateChange(cb: (s: CameraState, detail?: string) => void): () => void;
}

export interface WorkerLike {
  postMessage(msg: CameraWorkerRequest, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<CameraWorkerEvent>) => void) | null;
  onerror: ((e: ErrorEvent) => void) | null;
  terminate(): void;
}

export interface MstpFrameSourceOptions {
  createWorker?: () => WorkerLike;
  /** MediaStreamTrackProcessor queue length (default 2). */
  maxBufferSize?: number;
  /** Fall back when the worker has not attached the track within this time (default 3000 ms). */
  attachTimeoutMs?: number;
  /** Fall back when no frame arrived this long after attaching (default 3000 ms). */
  firstFrameTimeoutMs?: number;
  /** Reject a grab the worker did not answer within this time (default 2000 ms). */
  grabTimeoutMs?: number;
  /** Main-thread performance.timeOrigin (tests). */
  timeOrigin?: number;
  now?: () => number;
}

export type FrameSourceMode = 'idle' | 'pending' | 'mstp' | 'fallback';

export interface MstpStats {
  /** Frames announced by the worker. */
  frames: number;
  /** Frames dispatched to onFrame subscribers. */
  dispatched: number;
  grabs: number;
  /** Grabs whose frame had already been replaced by a newer one. */
  staleGrabs: number;
  grabErrors: number;
  reattaches: number;
  /** Current capture-clock → performance.now() offset (ms), NaN before the first frame. */
  clockOffsetMs: number;
  /** Last frame: coded size and pixel format (diagnostics). */
  codedWidth: number;
  codedHeight: number;
  format: string | null;
}

/**
 * Maps `VideoFrame.timestamp` (capture clock) onto the performance.now() timeline: the offset is
 * the minimum of (arrival − timestamp) over the last `window` frames, i.e. the capture time plus
 * the smallest observed delivery latency (the engine estimates the remaining camera–IMU offset).
 */
export class FrameClock {
  private readonly ring: Float64Array;
  private n = 0;
  private head = 0;
  private lastTs = -Infinity;

  constructor(readonly window = 300) {
    this.ring = new Float64Array(window);
  }

  get offsetMs(): number {
    if (this.n === 0) return NaN;
    let m = Infinity;
    for (let i = 0; i < this.n; i++) m = Math.min(m, this.ring[i]);
    return m;
  }

  reset(): void {
    this.n = 0;
    this.head = 0;
    this.lastTs = -Infinity;
  }

  /** `t` in the performance.now() timeline; source 'callback' when the timestamp is unusable. */
  map(timestampUs: number, receivedMs: number): { t: number; source: 'frameTimestamp' | 'callback' } {
    const ts = timestampUs / 1000;
    if (!Number.isFinite(ts) || ts <= 0 || ts <= this.lastTs) return { t: receivedMs, source: 'callback' };
    this.lastTs = ts;
    this.ring[this.head] = receivedMs - ts;
    this.head = (this.head + 1) % this.window;
    this.n = Math.min(this.n + 1, this.window);
    let t = ts + this.offsetMs;
    if (receivedMs - t > 1000) {
      // the clocks jumped (or a stale minimum): start over from this frame
      this.reset();
      return this.map(timestampUs, receivedMs);
    }
    t = Math.min(t, receivedMs);
    return { t, source: 'frameTimestamp' };
  }
}

interface PendingGrab {
  resolve: (b: ImageBitmap) => void;
  reject: (e: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class MstpFrameSource implements FramePumpCamera {
  private readonly o: MstpFrameSourceOptions;
  private worker: WorkerLike | null = null;
  private md: FrameSourceMode = 'idle';
  private reason: string | null = null;
  private gen = 0;
  /** Original track the worker's clone was made from (null = nothing attached). */
  private attachedFrom: MediaStreamTrack | null = null;
  /** Clone read by a main-thread processor (stream transfer); stopped on detach. */
  private localClone: MediaStreamTrack | null = null;
  private readonly cbs = new Set<FrameCallback>();
  private cameraUnsub: (() => void) | null = null;
  private stateUnsub: (() => void) | null = null;
  private latestSeq = -1;
  /** seq of the frame being dispatched (grabFrame called from the callback grabs exactly it). */
  private dispatchSeq = -1;
  private nextGrabId = 1;
  private readonly grabs = new Map<number, PendingGrab>();
  private readonly clock = new FrameClock();
  private clockSegment = -1;
  private workerOriginDelta = 0;
  private settleStart: ((m: FrameSourceMode) => void) | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private reattachTimes: number[] = [];
  private readonly meta: FrameMeta = { mediaTime: 0, width: 0, height: 0, callbackTime: 0, timeSource: 'frameTimestamp', segment: 0 };
  readonly stats: MstpStats = {
    frames: 0,
    dispatched: 0,
    grabs: 0,
    staleGrabs: 0,
    grabErrors: 0,
    reattaches: 0,
    clockOffsetMs: NaN,
    codedWidth: 0,
    codedHeight: 0,
    format: null,
  };

  constructor(
    private readonly camera: MstpCamera,
    opts: MstpFrameSourceOptions = {},
  ) {
    this.o = opts;
  }

  /** Whether this browser could run the path at all (the worker decides for sure). */
  static isSupported(): boolean {
    return typeof Worker !== 'undefined' && typeof MediaStreamTrack !== 'undefined' && typeof MediaStreamTrack.prototype.clone === 'function';
  }

  get mode(): FrameSourceMode {
    return this.md;
  }
  /** Why the source fell back to the `<video>` path (null while it did not). */
  get fallbackReason(): string | null {
    return this.reason;
  }
  /** Frames keep the sensor orientation (MSTP buffers): FramePump skips its aspect check. */
  get fixedOrientation(): boolean {
    return this.md === 'mstp';
  }
  get result(): { frameMaxSide: number | null } | null {
    return this.camera.result;
  }

  private now(): number {
    return this.o.now ? this.o.now() : performance.now();
  }

  /**
   * Attach the camera's current track (call after CameraCapture.start resolved). Resolves with
   * 'mstp' once the first frame arrived, or 'fallback' (see `fallbackReason`).
   */
  start(): Promise<FrameSourceMode> {
    if (this.md !== 'idle') return Promise.resolve(this.md);
    if (!this.stateUnsub) this.stateUnsub = this.camera.onStateChange(this.onCameraState);
    this.md = 'pending';
    const settled = new Promise<FrameSourceMode>((resolve) => (this.settleStart = resolve));
    try {
      this.worker = (this.o.createWorker ?? defaultWorker)();
      this.worker.onmessage = (e) => this.onEvent(e.data);
      this.worker.onerror = (e) => {
        e.preventDefault?.();
        this.onWorkerFailure(`camera worker error: ${e.message || 'script error'}`);
      };
    } catch (e) {
      this.fallback(`camera worker unavailable: ${errMessage(e)}`);
      return settled;
    }
    if (!this.attach()) return settled;
    this.armTimer(this.o.attachTimeoutMs ?? 3000, 'the camera worker did not attach the track');
    return settled;
  }

  /** Stop reading, terminate the worker and drop the subscriptions to the camera. */
  stop(): void {
    this.clearTimer();
    this.stateUnsub?.();
    this.stateUnsub = null;
    this.cameraUnsub?.();
    this.cameraUnsub = null;
    this.detach();
    this.worker?.terminate();
    this.worker = null;
    if (this.md === 'pending') this.settle('fallback');
    this.md = 'idle';
  }

  onFrame(cb: FrameCallback): () => void {
    this.cbs.add(cb);
    this.syncCameraSubscription();
    return () => {
      this.cbs.delete(cb);
      this.syncCameraSubscription();
    };
  }

  grabFrame(opts: { maxSide?: number; resizeQuality?: ResizeQuality } = {}): Promise<ImageBitmap> {
    if (this.md === 'fallback') return this.camera.grabFrame(opts);
    const seq = this.dispatchSeq >= 0 ? this.dispatchSeq : this.latestSeq;
    const w = this.worker;
    if (this.md !== 'mstp' || !w || seq < 0 || !this.attachedFrom) return Promise.reject(new Error('MstpFrameSource.grabFrame: no frame available'));
    const id = this.nextGrabId++;
    this.stats.grabs++;
    return new Promise<ImageBitmap>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.grabs.delete(id);
        this.stats.grabErrors++;
        reject(new Error('MstpFrameSource.grabFrame: the camera worker did not answer'));
      }, this.o.grabTimeoutMs ?? 2000);
      this.grabs.set(id, { resolve, reject, timer });
      w.postMessage({ type: 'grab', id, gen: this.gen, seq, maxSide: opts.maxSide, resizeQuality: opts.resizeQuality });
    });
  }

  // ---- attach / detach ----------------------------------------------------------------------

  /** Transfer a clone of the camera's track to the worker. false = failed (handled). */
  private attach(): boolean {
    const w = this.worker;
    const track = this.camera.track;
    if (!w) return false;
    if (!track || track.readyState !== 'live') {
      this.onWorkerFailure('the camera has no live track');
      return false;
    }
    this.detach();
    let clone: MediaStreamTrack;
    try {
      clone = track.clone();
    } catch (e) {
      this.onWorkerFailure(`track.clone() failed: ${errMessage(e)}`);
      return false;
    }
    const gen = ++this.gen;
    const maxBufferSize = this.o.maxBufferSize ?? 2;
    try {
      w.postMessage({ type: 'attach', gen, track: clone, maxBufferSize }, [clone as unknown as Transferable]);
    } catch (e) {
      const readable = mainThreadReadable(clone, maxBufferSize);
      try {
        if (!readable) throw e;
        w.postMessage({ type: 'attach', gen, readable, maxBufferSize }, [readable as unknown as Transferable]);
        this.localClone = clone;
      } catch (e2) {
        clone.stop();
        this.onWorkerFailure(`the track cannot be transferred to a worker: ${errMessage(e2)}`);
        return false;
      }
    }
    this.attachedFrom = track;
    this.latestSeq = -1;
    return true;
  }

  private detach(): void {
    this.rejectGrabs('the camera track was detached');
    this.localClone?.stop();
    this.localClone = null;
    if (!this.attachedFrom) return;
    this.attachedFrom = null;
    this.gen++; // late events of the old clone are ignored
    this.latestSeq = -1;
    try {
      this.worker?.postMessage({ type: 'detach' });
    } catch {
      /* worker gone */
    }
  }

  private readonly onCameraState = (s: CameraState): void => {
    if (this.md !== 'mstp' && this.md !== 'pending') return;
    // ('interrupted' without a track: CameraCapture released the stream while hidden)
    if (s === 'restarting' || s === 'stopped' || s === 'error' || (s === 'interrupted' && !this.camera.track)) {
      // never keep the old capture source alive through a clone
      this.detach();
      if (this.md === 'pending') this.fallback(`the camera ${s === 'error' ? 'failed' : s} while the camera worker started`);
    } else if (s === 'running' && this.camera.track && this.camera.track !== this.attachedFrom) {
      this.stats.reattaches++;
      this.attach();
    }
  };

  /** The worker could not read the track: fall back before the first frame, re-attach after. */
  private onWorkerFailure(reason: string): void {
    if (this.md === 'pending' && this.stats.dispatched === 0) {
      this.fallback(reason);
      return;
    }
    if (this.md !== 'mstp') return;
    console.warn(`[sensors] MSTP camera path: ${reason}`);
    this.detach();
    if (this.camera.state !== 'running' || !this.worker) return; // re-attached on 'running'
    const now = Date.now();
    this.reattachTimes = this.reattachTimes.filter((t) => now - t < 10000);
    if (this.reattachTimes.length >= 3) {
      console.warn('[sensors] MSTP camera path: re-attach limit reached; frames resume after the next camera restart');
      return;
    }
    this.reattachTimes.push(now);
    this.stats.reattaches++;
    this.attach();
  }

  private fallback(reason: string): void {
    this.clearTimer();
    this.reason = reason;
    console.warn(`[sensors] MSTP camera path unavailable, using the <video> path: ${reason}`);
    this.detach();
    this.worker?.terminate();
    this.worker = null;
    this.md = 'fallback';
    this.syncCameraSubscription();
    this.settle('fallback');
  }

  private settle(m: FrameSourceMode): void {
    const s = this.settleStart;
    this.settleStart = null;
    s?.(m);
  }

  private armTimer(ms: number, reason: string): void {
    this.clearTimer();
    this.timer = setTimeout(() => {
      this.timer = null;
      if (this.md === 'pending') this.fallback(reason);
    }, ms);
  }

  private clearTimer(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  private syncCameraSubscription(): void {
    const want = this.md === 'fallback' && this.cbs.size > 0;
    if (want && !this.cameraUnsub) this.cameraUnsub = this.camera.onFrame(this.forward);
    else if (!want && this.cameraUnsub) {
      this.cameraUnsub();
      this.cameraUnsub = null;
    }
  }

  private readonly forward: FrameCallback = (t, meta) => {
    for (const cb of this.cbs) cb(t, meta);
  };

  // ---- worker events ------------------------------------------------------------------------

  private onEvent(ev: CameraWorkerEvent): void {
    switch (ev.type) {
      case 'attached':
        if (ev.gen !== this.gen) return;
        this.workerOriginDelta = ev.timeOrigin - (this.o.timeOrigin ?? performance.timeOrigin);
        if (this.md === 'pending') this.armTimer(this.o.firstFrameTimeoutMs ?? 3000, 'no camera frame reached the worker');
        return;
      case 'frame':
        if (ev.gen === this.gen) this.onWorkerFrame(ev);
        return;
      case 'ended':
        if (ev.gen === this.gen) this.onWorkerFailure('the worker track ended');
        return;
      case 'error':
        if (ev.gen === this.gen) this.onWorkerFailure(ev.message);
        return;
      case 'grabbed': {
        const g = this.grabs.get(ev.id);
        if (!g) {
          if ('bitmap' in ev) ev.bitmap.close(); // timed out meanwhile
          return;
        }
        this.grabs.delete(ev.id);
        clearTimeout(g.timer);
        if ('bitmap' in ev) g.resolve(ev.bitmap);
        else {
          if (ev.error === 'stale') this.stats.staleGrabs++;
          else this.stats.grabErrors++;
          g.reject(new Error(`MstpFrameSource.grabFrame: ${ev.error}`));
        }
        return;
      }
    }
  }

  private onWorkerFrame(ev: Extract<CameraWorkerEvent, { type: 'frame' }>): void {
    const st = this.stats;
    st.frames++;
    st.codedWidth = ev.codedWidth;
    st.codedHeight = ev.codedHeight;
    st.format = ev.format;
    this.latestSeq = ev.seq;
    if (this.md === 'pending') {
      this.clearTimer();
      this.md = 'mstp';
      this.settle('mstp');
    }
    if (this.md !== 'mstp') return;
    const segment = this.camera.segment;
    if (segment !== this.clockSegment) {
      this.clock.reset(); // re-estimate after an interruption
      this.clockSegment = segment;
    }
    const now = this.now();
    const { t, source } = this.clock.map(ev.timestampUs, ev.receivedMs + this.workerOriginDelta);
    st.clockOffsetMs = this.clock.offsetMs;
    if (this.cbs.size === 0) return;
    const m = this.meta;
    m.mediaTime = 0;
    m.captureTime = undefined;
    m.width = ev.width;
    m.height = ev.height;
    m.callbackTime = now;
    m.timeSource = source;
    m.segment = segment;
    st.dispatched++;
    this.dispatchSeq = ev.seq;
    try {
      for (const cb of this.cbs) {
        try {
          cb(t, m);
        } catch (e) {
          console.error('[sensors] frame callback failed', e);
        }
      }
    } finally {
      this.dispatchSeq = -1;
    }
  }

  private rejectGrabs(reason: string): void {
    for (const g of this.grabs.values()) {
      clearTimeout(g.timer);
      g.reject(new Error(`MstpFrameSource.grabFrame: ${reason}`));
    }
    this.grabs.clear();
  }
}

/** A main-thread MediaStreamTrackProcessor's stream (Chromium); null where MSTP is worker-only (WebKit). */
function mainThreadReadable(track: MediaStreamTrack, maxBufferSize: number): ReadableStream<unknown> | null {
  const P = (globalThis as { MediaStreamTrackProcessor?: new (i: { track: MediaStreamTrack; maxBufferSize?: number }) => { readable: ReadableStream<unknown> } })
    .MediaStreamTrackProcessor;
  if (typeof P !== 'function') return null;
  try {
    return new P({ track, maxBufferSize }).readable;
  } catch {
    return null;
  }
}

function defaultWorker(): WorkerLike {
  return new Worker(new URL('../workers/camera.worker.ts', import.meta.url), { type: 'module', name: 'camera' }) as unknown as WorkerLike;
}

function errMessage(e: unknown): string {
  return (e as { message?: string })?.message ?? String(e);
}
