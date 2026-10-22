/**
 * Raw capture recorder (Setup › Options › "Record raw capture", off by default; for testers and
 * developers): records every FrameInput the app sends to the engine — the pixels as a ≤ 1280 px
 * JPEG plus every field verbatim — so a real iPhone capture can be replayed on a desktop through
 * the identical engine (format and export: src/io/recording.ts; replay: src/harness/engine,
 * e2e/README.md "Raw capture recordings").
 *
 * It is a FrameTap (./frame-tap.ts) on the EngineLike, installed by App.startRecording while a
 * capture runs. Per frame, on the main thread, synchronously before the frame is transferred to
 * the engine: a VideoFrame is `clone()`d (no copy: a second reference to the camera buffer), an
 * ImageBitmap is drawn at the recorded size into an OffscreenCanvas and taken as a new
 * ImageBitmap (≈ a GPU blit), an RGBAImage's pixels are copied; that copy and the metadata go to
 * the recorder worker (src/workers/recorder.worker.ts), which converts (closing VideoFrames at
 * once), encodes and writes — OPFS through sync access handles, IndexedDB where OPFS is missing.
 * The engine's answer follows in live.jsonl, flagged `notSent` when the engine client rejected
 * the frame before posting it (the engine never got it: replays leave it out).
 *
 * Sensor log (sensors.jsonl): the raw DeviceMotion / DeviceOrientation events (own listeners on
 * the window, added after the capture's sensors started and removed when the recording stops,
 * before they stop: never the page's last motion listener, whose removal restarts CoreMotion on
 * iOS) and every GPS fix (`RecorderStartInfo.geo`, the capture's GeoSensor), buffered on the main
 * thread and posted every `sensorFlushMs` (500 ms) as one batch — an estimated 10–20 KB/s (not
 * measured on an iPhone) next to the 1–3 MB/s of JPEGs; at most 20 000 rows wait (more are
 * dropped and counted).
 *
 * Bounds: when more than `maxPending` (3) frames are in the worker, the frame just sent holds
 * back its processFrame resolution until the worker caught up, so at most `maxPending` + 1 are
 * in it (back-pressure: the FramePump then skips camera frames as when the engine is busy, so the
 * recording still equals the engine's input). A worker that does not catch up within `stallMs`
 * stops the recording (never the scan); like EngineClient's timeouts, that clock does not run
 * while the page is hidden (iOS suspends the page and the worker) and restarts on return.
 * Storage: recording needs `floorBytes` (1 GB) of free origin quota to start and stops itself
 * when less is left (checked every 2 s) or on any write error — the capture screen warns at
 * 300 MB, so the scan's own keyframes keep their room. Both stop only the recording.
 *
 * Recording perturbs what it measures (main-thread clone / blit, a worker encoding ≈ 10–30 ms per
 * frame next to the engine): `live.jsonl` keeps the back-pressure wait per frame, and the
 * recording's header says that it was recorded.
 */
import type { GeoFix, RGBAImage } from '../core/types';
import {
  RECORDING_FORMAT,
  RECORDING_JPEG_QUALITY,
  RECORDING_MAX_SIDE,
  RECORDING_VERSION,
  frameMeta,
  liveSummary,
  recordedSize,
  type RecordedImageKind,
  type RecordedLive,
  type RecordedSensorRow,
  type RecorderFrameMessage,
  type RecordingHeader,
  type RecordingWriterStats,
} from '../io/recording';
import type { FrameInput, LiveUpdate } from '../workers/protocol';
import type { FrameTap } from './frame-tap';

/** Free origin quota a recording needs to start / below which it stops (bytes). */
export const RECORDER_FLOOR_BYTES = 1e9;

// ---- worker protocol --------------------------------------------------------------------------

export type RecorderWorkerRequest =
  | { type: 'open'; header: RecordingHeader; prefer: 'auto' | 'opfs' | 'idb'; maxSide: number; jpegQuality: number; idbName?: string }
  /** `width` × `height`: the size to record (≤ maxSide); `image` null when it could not be copied. */
  | { type: 'frame'; msg: RecorderFrameMessage; image: VideoFrame | ImageBitmap | RGBAImage | null; width: number; height: number }
  | { type: 'live'; row: RecordedLive }
  | { type: 'sensors'; rows: RecordedSensorRow[] }
  | { type: 'close'; header: RecordingHeader };

export type RecorderWorkerEvent =
  | { type: 'opened'; sink: string }
  | { type: 'frame'; seq: number; ok: boolean; bytes: number; error?: string }
  | { type: 'closed'; stats: RecordingWriterStats }
  | { type: 'error'; fatal: boolean; name: string; message: string };

// ---- recorder ---------------------------------------------------------------------------------

export interface RecorderStartInfo {
  /** Recording name (the scan's name). */
  name: string;
  sessionId: string | null;
  engine: RecordingHeader['engine'];
  /** CameraStartResult of the capture (device ids are left out). */
  camera?: object | null;
  /** FramePump settings (grab, videoFrameOrientation, targetHz). */
  capture?: Record<string, unknown> | null;
  /** FramePump statistics, read when the recording stops. */
  pumpStats?: () => object | null;
  /** Every GPS fix for sensors.jsonl (the capture's GeoSensor; null: no location). */
  geo?: GeoFixSource | null;
}

/** Where the recorder gets every GPS fix (GeoSensor.onFix). */
export interface GeoFixSource {
  onFix(cb: (f: GeoFix) => void): () => void;
}

/** The page visibility (document). */
export interface VisibilitySource {
  readonly hidden: boolean;
  addEventListener(type: 'visibilitychange', cb: () => void): void;
  removeEventListener(type: 'visibilitychange', cb: () => void): void;
}

export interface CaptureRecorderOptions {
  createWorker?: () => Worker;
  prefer?: 'auto' | 'opfs' | 'idb';
  /** IndexedDB database of the fallback sink (tests). */
  idbName?: string;
  maxSide?: number;
  jpegQuality?: number;
  maxPending?: number;
  floorBytes?: number;
  /** Free origin quota in bytes (null: unknown). Default: navigator.storage.estimate(). */
  available?: () => Promise<number | null>;
  checkEveryMs?: number;
  stallMs?: number;
  /** Where DeviceMotion / DeviceOrientation events arrive (default the window; null: no sensor log). */
  sensorTarget?: EventTarget | null;
  /** Sensor rows are posted to the worker in batches this often (ms). */
  sensorFlushMs?: number;
  /** Page visibility (default the document; null: always visible). */
  visibility?: VisibilitySource | null;
  /** The recording ended (`auto`: by itself — low storage, a write error, a stalled worker). */
  onStop?: (reason: string, auto: boolean) => void;
}

export type RecorderState = 'recording' | 'stopping' | 'stopped';

export interface RecorderStats {
  /** Frames sent to the worker / acknowledged by it / stored with their pixels. */
  sent: number;
  acked: number;
  stored: number;
  imageBytes: number;
  /** Total back-pressure wait (ms). */
  waitMs: number;
  /** sensors.jsonl rows posted / dropped (the buffer was full). */
  sensorRows: number;
  sensorDropped: number;
}

const SENSOR_BUFFER_ROWS = 20_000;

const CAMERA_KEYS = ['label', 'width', 'height', 'frameRate', 'hfovPriorDeg', 'focalPriorPx', 'lens', 'facing', 'aspect', 'lensMaySwitch', 'apple', 'frameMaxSide', 'requested', 'minFocusDistanceM', 'torchAvailable'];

/** The camera fields a recording keeps: no device / group ids. */
export function cameraSummary(cam: object | null | undefined): Record<string, unknown> | null {
  if (!cam) return null;
  const c = cam as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const k of CAMERA_KEYS) if (c[k] !== undefined) out[k] = c[k];
  if (c.settings && typeof c.settings === 'object') {
    const { deviceId: _d, groupId: _g, ...settings } = c.settings as Record<string, unknown>;
    out.settings = settings;
  }
  return out;
}

async function defaultAvailable(): Promise<number | null> {
  try {
    const st = (globalThis.navigator as Navigator | undefined)?.storage;
    if (!st || typeof st.estimate !== 'function') return null;
    const e = await st.estimate();
    return e.quota ? Math.max(0, e.quota - (e.usage ?? 0)) : null;
  } catch {
    return null;
  }
}

function defaultWorker(): Worker {
  return new Worker(new URL('../workers/recorder.worker.ts', import.meta.url), { type: 'module', name: 'recorder' });
}

let idSeq = 0;
function newId(now: number): string {
  const c = (globalThis as { crypto?: Crypto }).crypto;
  const d = new Date(now);
  const p = (n: number): string => String(n).padStart(2, '0');
  const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  const rand = c && typeof c.randomUUID === 'function' ? c.randomUUID().slice(0, 8) : `${now.toString(36)}${(++idSeq).toString(36)}`;
  return `rec-${stamp}-${rand}`;
}

function errorText(e: unknown): string {
  const o = e as { name?: string; message?: string } | null;
  return o && typeof o === 'object' ? `${o.name && o.name !== 'Error' ? `${o.name}: ` : ''}${o.message ?? String(e)}` : String(e);
}

function isVideoFrame(x: unknown): x is VideoFrame {
  const VF = (globalThis as { VideoFrame?: abstract new (...a: never[]) => VideoFrame }).VideoFrame;
  return !!VF && x instanceof VF;
}

function isImageBitmap(x: unknown): x is ImageBitmap {
  return typeof ImageBitmap !== 'undefined' && x instanceof ImageBitmap;
}

// ---- sensor log -------------------------------------------------------------------------------

type Triple = [number | null, number | null, number | null];
const num = (v: unknown): number | null => (typeof v === 'number' ? v : null);
const xyz = (o: { x?: unknown; y?: unknown; z?: unknown } | null | undefined): Triple | null => (o ? [num(o.x), num(o.y), num(o.z)] : null);
const abg = (o: { alpha?: unknown; beta?: unknown; gamma?: unknown } | null | undefined): Triple | null => (o ? [num(o.alpha), num(o.beta), num(o.gamma)] : null);

/** A DeviceMotionEvent as a sensors.jsonl row (every field the platform gave, verbatim). */
export function motionRow(e: Event, t: number): RecordedSensorRow {
  const m = e as Partial<DeviceMotionEvent>;
  return { k: 'motion', t, ts: e.timeStamp, interval: num(m.interval), rot: abg(m.rotationRate), acc: xyz(m.acceleration), accG: xyz(m.accelerationIncludingGravity) };
}

/** A DeviceOrientationEvent as a sensors.jsonl row (iOS `webkitCompassHeading` / `webkitCompassAccuracy` included). */
export function orientationRow(e: Event, t: number, absolute: boolean): RecordedSensorRow {
  const o = e as Partial<DeviceOrientationEvent> & { webkitCompassHeading?: unknown; webkitCompassAccuracy?: unknown };
  return {
    k: 'orientation',
    t,
    ts: e.timeStamp,
    absolute,
    alpha: num(o.alpha),
    beta: num(o.beta),
    gamma: num(o.gamma),
    absoluteFlag: typeof o.absolute === 'boolean' ? o.absolute : null,
    compassHeading: num(o.webkitCompassHeading),
    compassAccuracy: num(o.webkitCompassAccuracy),
  };
}

function defaultSensorTarget(): EventTarget | null {
  return typeof window !== 'undefined' && typeof window.addEventListener === 'function' ? window : null;
}

function defaultVisibility(): VisibilitySource | null {
  const d = (globalThis as { document?: Document }).document;
  return d && typeof d.addEventListener === 'function' ? (d as unknown as VisibilitySource) : null;
}

export class CaptureRecorder implements FrameTap {
  readonly id: string;
  readonly stats: RecorderStats = { sent: 0, acked: 0, stored: 0, imageBytes: 0, waitMs: 0, sensorRows: 0, sensorDropped: 0 };
  private st: RecorderState = 'recording';
  private header: RecordingHeader;
  private readonly worker: Worker;
  private readonly o: Required<Omit<CaptureRecorderOptions, 'createWorker' | 'onStop' | 'idbName' | 'sensorTarget' | 'visibility'>> & Pick<CaptureRecorderOptions, 'onStop' | 'idbName'>;
  private readonly info: RecorderStartInfo;
  private seq = 0;
  private readonly seqOf = new WeakMap<FrameInput, number>();
  private readonly waiters: (() => void)[] = [];
  private lastAck = 0;
  private lastCheck = -Infinity;
  private checking = false;
  private canvas: OffscreenCanvas | null = null;
  private stopping: Promise<RecordingHeader | null> | null = null;
  private closedResolve: ((stats: RecordingWriterStats | null) => void) | null = null;
  private workerStats: RecordingWriterStats | null = null;
  private stallTimer: ReturnType<typeof setTimeout> | null = null;
  /** Settles start(): the worker opened its sink, or failed to. */
  private openWait: { resolve: () => void; reject: (e: Error) => void } | null = null;
  private readonly sensorTarget: EventTarget | null;
  private readonly visibility: VisibilitySource | null;
  private hidden = false;
  private sensorRows: RecordedSensorRow[] = [];
  private sensorTimer: ReturnType<typeof setInterval> | null = null;
  /** Removes the sensor / visibility listeners (begin()). */
  private readonly offs: (() => void)[] = [];

  /** Worker, OffscreenCanvas and some storage (OPFS or IndexedDB) exist. */
  static isSupported(): boolean {
    const nav = globalThis.navigator as Navigator | undefined;
    const storage = !!nav?.storage && typeof nav.storage.getDirectory === 'function';
    return typeof Worker === 'function' && typeof OffscreenCanvas === 'function' && (storage || typeof globalThis.indexedDB !== 'undefined');
  }

  /** Open a new recording (rejects with the reason it cannot record). */
  static async start(info: RecorderStartInfo, opts: CaptureRecorderOptions = {}): Promise<CaptureRecorder> {
    if (!CaptureRecorder.isSupported()) throw new Error('this browser cannot record (needs Worker, OffscreenCanvas and OPFS or IndexedDB)');
    const floor = opts.floorBytes ?? RECORDER_FLOOR_BYTES;
    const avail = await (opts.available ?? defaultAvailable)();
    if (avail !== null && avail < floor) throw new Error(`only ${Math.round(avail / 1e6)} MB of storage are free (recording needs ${Math.round(floor / 1e6)} MB)`);
    const rec = new CaptureRecorder(info, opts, (opts.createWorker ?? defaultWorker)());
    try {
      await rec.open();
    } catch (e) {
      rec.worker.terminate();
      throw e;
    }
    rec.begin();
    return rec;
  }

  private constructor(info: RecorderStartInfo, opts: CaptureRecorderOptions, worker: Worker) {
    this.info = info;
    this.o = {
      prefer: opts.prefer ?? 'auto',
      idbName: opts.idbName,
      maxSide: opts.maxSide ?? RECORDING_MAX_SIDE,
      jpegQuality: opts.jpegQuality ?? RECORDING_JPEG_QUALITY,
      maxPending: opts.maxPending ?? 3,
      floorBytes: opts.floorBytes ?? RECORDER_FLOOR_BYTES,
      available: opts.available ?? defaultAvailable,
      checkEveryMs: opts.checkEveryMs ?? 2000,
      stallMs: opts.stallMs ?? 8000,
      sensorFlushMs: opts.sensorFlushMs ?? 500,
      onStop: opts.onStop,
    };
    this.sensorTarget = opts.sensorTarget === undefined ? defaultSensorTarget() : opts.sensorTarget;
    this.visibility = opts.visibility === undefined ? defaultVisibility() : opts.visibility;
    const now = Date.now();
    this.id = newId(now);
    const nav = globalThis.navigator as (Navigator & { standalone?: boolean }) | undefined;
    this.header = {
      format: RECORDING_FORMAT,
      version: RECORDING_VERSION,
      id: this.id,
      name: info.name,
      createdAt: now,
      timeOrigin: performance.timeOrigin,
      app: { userAgent: nav?.userAgent ?? '', ...(nav?.platform ? { platform: nav.platform } : {}), ...(nav?.standalone !== undefined ? { standalone: nav.standalone } : {}), build: import.meta.env?.MODE ?? '' },
      sessionId: info.sessionId,
      engine: { ...info.engine },
      camera: cameraSummary(info.camera),
      capture: info.capture ?? null,
      recorder: { maxSide: this.o.maxSide, jpegQuality: this.o.jpegQuality, sink: '', floorBytes: this.o.floorBytes, maxPending: this.o.maxPending },
      state: 'recording',
    };
    this.worker = worker;
    worker.onmessage = (ev: MessageEvent<RecorderWorkerEvent>) => this.onWorker(ev.data);
    worker.onerror = (ev) => {
      ev.preventDefault?.();
      const msg = `recorder worker failed: ${ev.message || 'script error'}`;
      if (this.openWait) {
        this.openWait.reject(new Error(msg));
        this.openWait = null;
      } else this.autoStop(msg);
    };
  }

  get state(): RecorderState {
    return this.st;
  }

  /** Frames posted but not yet written. */
  get pending(): number {
    return this.stats.sent - this.stats.acked;
  }

  get sink(): string {
    return this.header.recorder.sink;
  }

  private open(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('the recorder worker did not start')), 15_000);
      this.openWait = {
        resolve: () => {
          clearTimeout(timer);
          resolve();
        },
        reject: (e) => {
          clearTimeout(timer);
          reject(e);
        },
      };
      this.post({ type: 'open', header: this.header, prefer: this.o.prefer, maxSide: this.o.maxSide, jpegQuality: this.o.jpegQuality, idbName: this.o.idbName });
    });
  }

  private post(req: RecorderWorkerRequest, transfer: Transferable[] = []): void {
    this.worker.postMessage(req, transfer);
  }

  /** The recording runs: sensor log and visibility listeners (removed by finish()). */
  private begin(): void {
    const on = (target: { addEventListener(t: string, cb: (e: Event) => void): void; removeEventListener(t: string, cb: (e: Event) => void): void }, type: string, cb: (e: Event) => void): void => {
      target.addEventListener(type, cb);
      this.offs.push(() => target.removeEventListener(type, cb));
    };
    const vis = this.visibility;
    if (vis) {
      this.hidden = vis.hidden;
      on(vis as unknown as EventTarget, 'visibilitychange', () => this.onVisibility());
    }
    const tgt = this.sensorTarget;
    if (tgt) {
      on(tgt, 'devicemotion', (e) => this.pushSensor(motionRow(e, performance.now())));
      on(tgt, 'deviceorientation', (e) => this.pushSensor(orientationRow(e, performance.now(), false)));
      on(tgt, 'deviceorientationabsolute', (e) => this.pushSensor(orientationRow(e, performance.now(), true)));
    }
    const geo = this.info.geo;
    if (geo) this.offs.push(geo.onFix((f) => this.pushSensor({ k: 'geo', t: performance.now(), fix: { ...f } })));
    if (tgt || geo) this.sensorTimer = setInterval(() => this.flushSensors(), Math.max(1, this.o.sensorFlushMs));
  }

  private pushSensor(row: RecordedSensorRow): void {
    if (this.st !== 'recording') return;
    if (this.sensorRows.length >= SENSOR_BUFFER_ROWS) {
      this.stats.sensorDropped++;
      return;
    }
    this.sensorRows.push(row);
  }

  private flushSensors(): void {
    if (!this.sensorRows.length) return;
    const rows = this.sensorRows;
    this.sensorRows = [];
    try {
      this.post({ type: 'sensors', rows });
      this.stats.sensorRows += rows.length;
    } catch {
      this.stats.sensorDropped += rows.length;
    }
  }

  /** Hidden: iOS suspends the page and the worker, so the stall clock stops; it restarts on return. */
  private onVisibility(): void {
    const hidden = !!this.visibility?.hidden;
    if (hidden === this.hidden) return;
    this.hidden = hidden;
    if (hidden) {
      if (this.stallTimer) clearTimeout(this.stallTimer);
      this.stallTimer = null;
      return;
    }
    this.lastAck = performance.now();
    if (this.waiters.length && this.st === 'recording') this.armStall();
  }

  private onWorker(ev: RecorderWorkerEvent): void {
    switch (ev.type) {
      case 'opened':
        this.header.recorder.sink = ev.sink;
        this.openWait?.resolve();
        this.openWait = null;
        return;
      case 'frame':
        this.stats.acked++;
        this.lastAck = performance.now();
        if (ev.ok) {
          this.stats.stored++;
          this.stats.imageBytes += ev.bytes;
        }
        this.release();
        return;
      case 'closed':
        this.workerStats = ev.stats;
        this.closedResolve?.(ev.stats);
        return;
      case 'error': {
        const msg = `${ev.name}: ${ev.message}`;
        if (this.openWait) {
          this.openWait.reject(new Error(ev.message));
          this.openWait = null;
          return;
        }
        if (ev.fatal) this.autoStop(/quota/i.test(msg) ? 'storage is full' : `write failed (${msg})`);
        return;
      }
    }
  }

  /** Wake the frames waiting for room. */
  private release(): void {
    if (this.pending > this.o.maxPending && this.st === 'recording') return;
    if (this.stallTimer) clearTimeout(this.stallTimer);
    this.stallTimer = null;
    for (const w of this.waiters.splice(0)) w();
  }

  private waitForRoom(): Promise<void> {
    const t0 = performance.now();
    return new Promise<void>((resolve) => {
      this.waiters.push(() => {
        this.stats.waitMs += performance.now() - t0;
        resolve();
      });
      this.armStall();
    });
  }

  /** While frames wait (and the page is visible): no acknowledgement for `stallMs` stops the recording. */
  private armStall(): void {
    if (this.stallTimer || this.hidden) return;
    const left = Math.max(1, this.o.stallMs - (performance.now() - this.lastAck));
    this.stallTimer = setTimeout(() => {
      this.stallTimer = null;
      if (!this.waiters.length || this.hidden) return;
      // timers may fire a little early against performance.now(): 1 ms of slack
      if (performance.now() - this.lastAck >= this.o.stallMs - 1) this.autoStop(`the recorder fell behind by ${this.pending} frames for ${(this.o.stallMs / 1000).toFixed(1)} s`);
      else this.armStall();
    }, left);
  }

  /** The pixels to send to the worker (sync: the caller transfers the original right after). */
  private copyImage(b: FrameInput['bitmap'], maxSide: number): { image: VideoFrame | ImageBitmap | RGBAImage; kind: RecordedImageKind; srcWidth: number; srcHeight: number; width: number; height: number; transfer: Transferable[] } {
    if (isVideoFrame(b)) {
      const fit = recordedSize(b.displayWidth, b.displayHeight, maxSide);
      const clone = b.clone();
      return { image: clone, kind: 'VideoFrame', srcWidth: b.displayWidth, srcHeight: b.displayHeight, ...fit, transfer: [clone] };
    }
    if (isImageBitmap(b)) {
      const fit = recordedSize(b.width, b.height, maxSide);
      if (!this.canvas) this.canvas = new OffscreenCanvas(fit.width, fit.height);
      const c = this.canvas;
      if (c.width !== fit.width || c.height !== fit.height) {
        c.width = fit.width;
        c.height = fit.height;
      }
      const ctx = c.getContext('2d') as OffscreenCanvasRenderingContext2D | null;
      if (!ctx) throw new Error('OffscreenCanvas 2D context unavailable');
      ctx.drawImage(b, 0, 0, fit.width, fit.height);
      const copy = c.transferToImageBitmap();
      return { image: copy, kind: 'ImageBitmap', srcWidth: b.width, srcHeight: b.height, ...fit, transfer: [copy] };
    }
    const img = b as RGBAImage;
    const data = new Uint8ClampedArray(img.data);
    return { image: { width: img.width, height: img.height, data }, kind: 'RGBAImage', srcWidth: img.width, srcHeight: img.height, ...recordedSize(img.width, img.height, maxSide), transfer: [data.buffer] };
  }

  // ---- FrameTap -------------------------------------------------------------------------------

  before(frame: FrameInput): Promise<void> | void {
    if (this.st !== 'recording') return;
    const seq = this.seq++;
    this.seqOf.set(frame, seq);
    const ms = frame.maxSide && frame.maxSide > 0 ? Math.min(frame.maxSide, this.o.maxSide) : this.o.maxSide;
    let copy: ReturnType<CaptureRecorder['copyImage']> | null = null;
    let error: string | undefined;
    try {
      copy = this.copyImage(frame.bitmap, ms);
    } catch (e) {
      error = `the pixels could not be copied: ${errorText(e)}`;
    }
    const msg: RecorderFrameMessage = {
      seq,
      sentAt: performance.now(),
      input: frameMeta(frame),
      kind: copy?.kind ?? (isVideoFrame(frame.bitmap) ? 'VideoFrame' : isImageBitmap(frame.bitmap) ? 'ImageBitmap' : 'RGBAImage'),
      srcWidth: copy?.srcWidth ?? 0,
      srcHeight: copy?.srcHeight ?? 0,
      ...(error ? { error } : {}),
    };
    try {
      this.post({ type: 'frame', msg, image: copy?.image ?? null, width: copy?.width ?? 0, height: copy?.height ?? 0 }, copy?.transfer ?? []);
    } catch (e) {
      (copy?.image as { close?: () => void } | undefined)?.close?.();
      this.autoStop(`the frame could not be sent to the recorder (${errorText(e)})`);
      return;
    }
    this.stats.sent++;
    if (this.stats.sent === 1) this.lastAck = performance.now();
    this.checkSpace();
    if (this.pending <= this.o.maxPending) return;
    return this.waitForRoom();
  }

  after(frame: FrameInput, result: LiveUpdate | null, error: unknown, t: { ms: number; waitMs: number; sent: boolean }): void {
    const seq = this.seqOf.get(frame);
    if (seq === undefined || this.st === 'stopped') return;
    const row: RecordedLive = {
      seq,
      index: typeof frame.index === 'number' ? frame.index : null,
      ms: Math.round(t.ms * 10) / 10,
      waitMs: Math.round(t.waitMs * 10) / 10,
      update: result ? liveSummary(result) : null,
      error: error ? errorText(error) : null,
    };
    // the engine never got it (rejected before posting): replays leave it out
    if (!t.sent) row.notSent = true;
    try {
      this.post({ type: 'live', row });
    } catch {
      /* worker gone: the recording stops through its error */
    }
  }

  // ---- storage floor / stop ---------------------------------------------------------------------

  private checkSpace(): void {
    const now = performance.now();
    if (this.checking || now - this.lastCheck < this.o.checkEveryMs) return;
    this.lastCheck = now;
    this.checking = true;
    void this.o
      .available()
      .then((avail) => {
        if (avail !== null && avail < this.o.floorBytes) this.autoStop(`storage is low (${Math.round(avail / 1e6)} MB free)`);
      })
      .catch(() => undefined)
      .finally(() => {
        this.checking = false;
      });
  }

  private autoStop(reason: string): void {
    if (this.st !== 'recording') return;
    console.warn(`[recorder] stopping: ${reason}`);
    void this.finish(reason, true);
  }

  /** Stop recording: the frames already sent are written, then the header with its summary. */
  stop(reason = 'capture ended'): Promise<RecordingHeader | null> {
    return this.finish(reason, false);
  }

  private finish(reason: string, auto: boolean): Promise<RecordingHeader | null> {
    if (this.stopping) return this.stopping;
    // synchronously: the sensor listeners go before the capture stops its own sensors
    for (const off of this.offs.splice(0)) off();
    if (this.sensorTimer) clearInterval(this.sensorTimer);
    this.sensorTimer = null;
    this.flushSensors();
    this.st = 'stopping';
    this.release();
    this.stopping = (async () => {
      let pump: Record<string, number> | null = null;
      try {
        pump = (this.info.pumpStats?.() as Record<string, number> | null) ?? null;
      } catch {
        pump = null;
      }
      const closed = new Promise<RecordingWriterStats | null>((resolve) => {
        this.closedResolve = resolve;
        setTimeout(() => resolve(null), 15_000);
      });
      const header: RecordingHeader = {
        ...this.header,
        state: 'complete',
        stoppedAt: Date.now(),
        stopReason: reason,
        summary: {
          frames: this.stats.sent,
          imageBytes: this.stats.imageBytes,
          failedImages: this.stats.acked - this.stats.stored,
          waitMs: Math.round(this.stats.waitMs),
          durationMs: Date.now() - this.header.createdAt,
          sensorRows: this.stats.sensorRows,
          sensorDropped: this.stats.sensorDropped,
          pump: pump ? { ...pump } : null,
        },
      };
      let stats: RecordingWriterStats | null = null;
      try {
        this.post({ type: 'close', header });
        stats = await closed;
      } catch {
        stats = null;
      }
      if (stats && header.summary) {
        header.summary.frames = stats.frames;
        header.summary.imageBytes = stats.imageBytes;
        header.summary.failedImages = stats.failedImages;
      }
      this.worker.terminate();
      this.st = 'stopped';
      this.header = header;
      try {
        this.o.onStop?.(reason, auto);
      } catch (e) {
        console.warn('[recorder] onStop failed', e);
      }
      return stats || this.workerStats ? header : null;
    })();
    return this.stopping;
  }
}
