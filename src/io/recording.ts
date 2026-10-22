/**
 * Raw capture recordings (the developer option "Record raw capture", src/app/recorder.ts): the
 * format, its export as an e2e dataset (e2e/README.md "Raw capture recordings") and the writer
 * core that the recorder worker (src/workers/recorder.worker.ts) runs. Pure: no DOM, no
 * storage API — the sinks live in ./recording-store.ts; unit-tested in recording.test.ts.
 *
 * What is recorded: every FrameInput the app SENT to the engine (EngineLike.setFrameTap, so the
 * frames the FramePump skipped while the engine was busy are not in it), in that order —
 *  - `frames.jsonl`: one line per frame, `{seq, sentAt, input, image, error?}` where `input` is
 *    the FrameInput without its pixels, VERBATIM (t, screenAngle, screenAngleSince, imu, geo,
 *    heading, index, segment, gravityDevice, attitude, imuCalibrationVersion, timeSource,
 *    rawBuffer, maxSide), JSON-safe (src/io/serialize.ts: typed arrays as `{$typed, data}`, −∞ /
 *    NaN / −0 as `{$num}`), so `decodeFrameRow` gives back exactly what the engine received;
 *  - the pixels as a JPEG (`RECORDING_JPEG_QUALITY`) at most `RECORDING_MAX_SIDE` px on the long
 *    side (bounds storage at ≈ 0.2–0.35 MB per frame; a 'fast' iOS capture already sends 1280 px
 *    frames, 'balanced' 1920 × 1440 ones). `image` keeps the recorded size and what the engine
 *    got: `kind` (VideoFrame / ImageBitmap / RGBAImage) and `srcWidth` × `srcHeight` (the
 *    VideoFrame's display size before its `maxSide`), the orientation flags are `input.rawBuffer`
 *    / `input.screenAngle`. The JPEG keeps the pixel layout of the input (a raw sensor-landscape
 *    VideoFrame stays landscape: `rawBuffer` says so on replay);
 *  - `live.jsonl`: the engine's answer per frame (decision, guidance, tracking, counts, timings,
 *    debug) and the recorder's own back-pressure wait — the device-side outcome a desktop replay
 *    is compared with;
 *  - `recording.json`: the header (app / device, EngineOptions, camera, recorder settings) and,
 *    once stopped, its summary (frames, bytes, why it stopped, FramePump stats).
 * Times are the main thread's performance.now() (the frame clock); `header.timeOrigin` maps
 * them to epoch ms.
 *
 * Export (`buildDatasetFiles`, `planZipParts`): the e2e dataset layout the harness reads —
 * `meta.json` (no GT: `gt: null`, `intrinsics: null`, `scene: 'recording'`, the recorded
 * `mode` / `quality` / `hfovPriorDeg` as the replay defaults), `frames.json` ({index, file, t}),
 * `frames/NNNNNN.jpg`, `imu.json` (the IMU samples of all frames, concatenated), `geo.json`
 * (distinct fixes), `heading.json` (per-frame headings with their north reference), plus
 * `frames.jsonl`, `live.jsonl` and `recording.json`. The derived files are what the ENGINE got
 * (FramePump slices), not the raw sensor streams. `stripLocation` drops the GPS fixes (also from
 * `frames.jsonl`) and the location origin. STORE ZIP parts of at most `RECORDING_PART_BYTES`
 * (the iOS share sheet reads every file fully into memory, ≈ 2× — docs/research/ios-platform.md
 * §3), each a standalone archive; extracting all parts into one directory gives the dataset.
 */
import type { GeoFix, ImuSample } from '../core/types';
import type { EngineOptions, FrameInput, HeadingSample, LiveUpdate } from '../workers/protocol';
import { parseJSONSafe, stringifyJSONSafe } from './serialize';

export const RECORDING_FORMAT = 'sfm-raw-capture';
export const RECORDING_VERSION = 1;
/** Long side of the recorded JPEGs (px): bounds storage; the engine's own frames may be larger. */
export const RECORDING_MAX_SIDE = 1280;
export const RECORDING_JPEG_QUALITY = 0.9;
/** Largest export ZIP part (bytes): ≤ 200 MB, see the header. */
export const RECORDING_PART_BYTES = 200e6;

/** A FrameInput without its pixels: what a recording keeps verbatim. */
export type FrameMeta = Omit<FrameInput, 'bitmap'>;

export type RecordedImageKind = 'VideoFrame' | 'ImageBitmap' | 'RGBAImage';

/** Where a sink keeps a frame's JPEG (internal to the store; dropped on export). */
export type ImageStoreRef = { pack: number; offset: number } | { key: string };

export interface RecordedImage {
  /** Path of the JPEG in the exported dataset ('frames/000123.jpg'). */
  file: string;
  /** Size of the recorded JPEG. */
  width: number;
  height: number;
  /** Size of the pixels the engine received (a VideoFrame's display size, before its `maxSide`). */
  srcWidth: number;
  srcHeight: number;
  /** Type of FrameInput.bitmap. */
  kind: RecordedImageKind;
  bytes: number;
  store?: ImageStoreRef;
}

/** One line of frames.jsonl. */
export interface RecordedFrame {
  /** 0-based order in which the frames were sent to the engine. */
  seq: number;
  /** performance.now() when the recorder saw the frame (just before it was sent). */
  sentAt: number;
  input: FrameMeta;
  /** null: the pixels could not be recorded (`error` says why); replays skip the frame. */
  image: RecordedImage | null;
  error?: string;
}

/** One line of live.jsonl: what the engine answered on the device. */
export interface RecordedLive {
  seq: number;
  /** FrameInput.index of the frame. */
  index: number | null;
  /** processFrame wall time on the device (ms). */
  ms: number;
  /** Time the frame waited for the recorder (back-pressure, ms). */
  waitMs: number;
  update: Pick<LiveUpdate, 'frameIndex' | 'decision' | 'guidance' | 'tracking' | 'numKeyframes' | 'numRegistered' | 'numPoints' | 'timings' | 'debug'> | null;
  error: string | null;
}

export interface RecordingSummary {
  frames: number;
  imageBytes: number;
  /** Frames whose pixels could not be recorded. */
  failedImages: number;
  /** Total back-pressure wait of the engine's frames (ms). */
  waitMs: number;
  durationMs: number;
  /** FramePump statistics at the end (frames skipped while busy etc.), when known. */
  pump?: Record<string, number> | null;
}

export interface RecordingHeader {
  format: typeof RECORDING_FORMAT;
  version: number;
  id: string;
  name: string;
  /** Epoch ms of the start. */
  createdAt: number;
  /** performance.timeOrigin of the page (epoch ms of frame time 0). */
  timeOrigin: number;
  app: { userAgent: string; platform?: string; standalone?: boolean; build?: string };
  /** The scan session the frames belong to (null: not persisted). */
  sessionId: string | null;
  /** EngineOptions of the scan (the replay defaults). */
  engine: Pick<EngineOptions, 'mode' | 'quality' | 'hfovPriorDeg' | 'preferGPU'>;
  /** Camera of the scan (src/app/recorder.ts cameraSummary: no device ids). */
  camera: Record<string, unknown> | null;
  /** How frames were grabbed (FramePump `grab`, `videoFrameOrientation`, `targetHz`). */
  capture: Record<string, unknown> | null;
  recorder: { maxSide: number; jpegQuality: number; sink: string; floorBytes: number; maxPending: number };
  state: 'recording' | 'complete';
  stoppedAt?: number;
  stopReason?: string;
  summary?: RecordingSummary;
}

// ---------------------------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------------------------

/** FrameInput without `bitmap` (a shallow copy: the arrays are the caller's). */
export function frameMeta(f: FrameInput): FrameMeta {
  const { bitmap: _bitmap, ...rest } = f;
  return rest;
}

export function encodeFrameRow(r: RecordedFrame): string {
  return stringifyJSONSafe(r);
}

export function decodeFrameRow(line: string): RecordedFrame {
  const r = parseJSONSafe<RecordedFrame>(line);
  if (!r || typeof r !== 'object' || typeof r.seq !== 'number' || !r.input || typeof r.input.t !== 'number') throw new Error('not a frames.jsonl row');
  return r;
}

export function encodeLiveRow(r: RecordedLive): string {
  return stringifyJSONSafe(r);
}

export function decodeLiveRow(line: string): RecordedLive {
  const r = parseJSONSafe<RecordedLive>(line);
  if (!r || typeof r !== 'object' || typeof r.seq !== 'number') throw new Error('not a live.jsonl row');
  return r;
}

/** The engine's answer as a live.jsonl `update` (no overlay / sparse / cameras: large, derived). */
export function liveSummary(u: LiveUpdate): RecordedLive['update'] {
  const out: NonNullable<RecordedLive['update']> = {
    frameIndex: u.frameIndex,
    decision: u.decision,
    guidance: u.guidance,
    tracking: u.tracking,
    numKeyframes: u.numKeyframes,
    numRegistered: u.numRegistered,
    numPoints: u.numPoints,
    timings: { ...u.timings },
  };
  if (u.debug) out.debug = { ...u.debug };
  return out;
}

/**
 * Parse a JSON-lines file. Unparseable lines are skipped and counted — the last line of a
 * recording whose app was killed mid-write can be truncated.
 */
export function parseJsonl<T>(text: string, decode: (line: string) => T): { rows: T[]; bad: number } {
  const rows: T[] = [];
  let bad = 0;
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    try {
      rows.push(decode(line));
    } catch {
      bad++;
    }
  }
  return { rows, bad };
}

/** Dataset path of frame `seq`'s JPEG. */
export function frameFileName(seq: number): string {
  return `frames/${String(seq).padStart(6, '0')}.jpg`;
}

/** Long-side fit (never enlarged), like the engine's `maxSide` (src/engine/engine.ts frameToBitmap). */
export function recordedSize(w: number, h: number, maxSide: number): { width: number; height: number } {
  const s = maxSide > 0 ? Math.min(1, maxSide / Math.max(w, h)) : 1;
  return { width: Math.max(1, Math.round(w * s)), height: Math.max(1, Math.round(h * s)) };
}

/**
 * Dataset frame index of each row: FrameInput.index when every row has a distinct integer one
 * (the FramePump's delivered counter), else the row's `seq`.
 */
export function datasetIndices(rows: readonly RecordedFrame[]): number[] {
  const idx = rows.map((r) => r.input.index);
  const ok = idx.every((v) => typeof v === 'number' && Number.isInteger(v)) && new Set(idx).size === idx.length;
  return ok ? (idx as number[]) : rows.map((r) => r.seq);
}

// ---------------------------------------------------------------------------------------------
// Writer core (runs in the recorder worker; Node tests drive it with an in-memory sink)
// ---------------------------------------------------------------------------------------------

/** Where a recording's bytes go (OPFS / IndexedDB: ./recording-store.ts). */
export interface RecordingSink {
  readonly kind: string;
  writeHeader(json: string): Promise<void>;
  /** Store frame `seq`'s JPEG; returns where (kept in the row's `image.store`). */
  putImage(seq: number, jpeg: Uint8Array): Promise<ImageStoreRef>;
  appendLine(file: 'frames.jsonl' | 'live.jsonl', line: string): Promise<void>;
  /** Make what was written so far durable (a killed app keeps it). */
  flush(): Promise<void>;
  close(): Promise<void>;
}

export interface EncodedImage {
  jpeg: Uint8Array;
  width: number;
  height: number;
}

/** A frame as the recorder's main-thread side posts it (the pixels travel separately). */
export interface RecorderFrameMessage {
  seq: number;
  sentAt: number;
  input: FrameMeta;
  kind: RecordedImageKind;
  srcWidth: number;
  srcHeight: number;
  /** The main thread could not copy the pixels. */
  error?: string;
}

export interface RecordingWriterStats {
  frames: number;
  imageBytes: number;
  lineBytes: number;
  failedImages: number;
}

/**
 * Writes frames in `seq` order: `writeFrame` may be called as soon as a frame arrives (its
 * conversion already running in `image`); the JPEG is stored, then its frames.jsonl line — a row
 * only ever names a stored image. A sink failure rejects that call and every later one (the
 * recording is broken: the caller stops it).
 */
export class RecordingWriter {
  readonly stats: RecordingWriterStats = { frames: 0, imageBytes: 0, lineBytes: 0, failedImages: 0 };
  private chain: Promise<unknown> = Promise.resolve();
  private failed: unknown = null;
  private sinceFlush = 0;

  constructor(
    private readonly sink: RecordingSink,
    private readonly flushEvery = 10,
  ) {}

  writeFrame(msg: RecorderFrameMessage, image: Promise<EncodedImage> | EncodedImage | null): Promise<RecordedFrame> {
    // conversion errors are this frame's, not the sink's: never an unhandled rejection meanwhile
    const img = Promise.resolve(image).then(
      (v) => ({ ok: true as const, v }),
      (e: unknown) => ({ ok: false as const, e }),
    );
    return this.enqueue(async () => {
      const got = await img;
      const row: RecordedFrame = { seq: msg.seq, sentAt: msg.sentAt, input: msg.input, image: null };
      const err = msg.error ?? (got.ok ? (got.v ? null : 'no image') : errorText(got.e));
      if (err === null && got.ok && got.v) {
        const enc = got.v;
        const store = await this.sink.putImage(msg.seq, enc.jpeg);
        row.image = { file: frameFileName(msg.seq), width: enc.width, height: enc.height, srcWidth: msg.srcWidth, srcHeight: msg.srcHeight, kind: msg.kind, bytes: enc.jpeg.byteLength, store };
        this.stats.imageBytes += enc.jpeg.byteLength;
      } else {
        row.error = err ?? 'no image';
        this.stats.failedImages++;
      }
      const line = encodeFrameRow(row);
      await this.sink.appendLine('frames.jsonl', line);
      this.stats.lineBytes += line.length + 1;
      this.stats.frames++;
      if (++this.sinceFlush >= this.flushEvery) {
        this.sinceFlush = 0;
        await this.sink.flush();
      }
      return row;
    });
  }

  writeLive(row: RecordedLive): Promise<void> {
    return this.enqueue(async () => {
      const line = encodeLiveRow(row);
      await this.sink.appendLine('live.jsonl', line);
      this.stats.lineBytes += line.length + 1;
    });
  }

  writeHeader(h: RecordingHeader): Promise<void> {
    return this.enqueue(() => this.sink.writeHeader(JSON.stringify(h, null, 1)));
  }

  /**
   * Write the final header after everything queued (its summary counts are this writer's: the
   * frames still in flight when the recording was stopped are included), flush, close the sink.
   * Also after a sink failure, best effort (a full disk may refuse the header too: the recording
   * then stays 'recording', listed as interrupted); rejects with the first error.
   */
  close(h: RecordingHeader): Promise<void> {
    const run = this.chain.then(async () => {
      const s = this.stats;
      const final: RecordingHeader = h.summary ? { ...h, summary: { ...h.summary, frames: s.frames, imageBytes: s.imageBytes, failedImages: s.failedImages } } : h;
      let err: unknown = this.failed;
      for (const step of [() => this.sink.writeHeader(JSON.stringify(final, null, 1)), () => this.sink.flush(), () => this.sink.close()]) {
        try {
          await step();
        } catch (e) {
          err ??= e;
        }
      }
      if (err) throw err;
    });
    this.chain = run.catch(() => undefined);
    return run;
  }

  private enqueue<T>(job: () => Promise<T>): Promise<T> {
    const run = this.chain.then(() => {
      if (this.failed) throw this.failed;
      return job();
    });
    this.chain = run.catch((e: unknown) => {
      this.failed ??= e;
    });
    return run;
  }
}

function errorText(e: unknown): string {
  const o = e as { name?: string; message?: string } | null;
  return o && typeof o === 'object' ? `${o.name ?? 'Error'}: ${o.message ?? String(e)}` : String(e);
}

// ---------------------------------------------------------------------------------------------
// Export as an e2e dataset
// ---------------------------------------------------------------------------------------------

/** heading.json row of a recording (per frame; `north` / `declinationDeg` as the engine got them). */
export interface RecordedHeadingRow {
  t: number;
  heading: number;
  accuracy: number;
  north: 'true' | 'magnetic';
  declinationDeg: number | null;
}

/** meta.json of an exported recording (the fields the harness and e2e/run-e2e.mjs read). */
export interface RecordingDatasetMeta {
  name: string;
  scene: 'recording';
  source: typeof RECORDING_FORMAT;
  version: number;
  /** Recorded EngineOptions: the replay defaults. */
  mode: string;
  quality: string;
  hfovPriorDeg: number | null;
  /** First recorded frame's JPEG size, and the size the engine received. */
  width: number;
  height: number;
  srcWidth: number;
  srcHeight: number;
  /** Delivered frame rate (frames / duration). */
  fps: number;
  numFrames: number;
  /** Seconds from the first to the last frame. */
  duration: number;
  startTimeMs: number;
  screenAngle: number;
  /** Epoch ms of the start (captureDateMs of a replay). */
  createdAt: number;
  originLatLonAlt: [number, number, number] | null;
  locationStripped: boolean;
  intrinsics: null;
  gt: null;
  sceneExtent: null;
  camera: Record<string, unknown> | null;
  app: RecordingHeader['app'];
  recorder: RecordingHeader['recorder'];
  summary: RecordingSummary | null;
  /** frames.jsonl lines that could not be parsed / frames without pixels (left out). */
  badLines: number;
  missingImages: number;
}

export interface DatasetBuild {
  meta: RecordingDatasetMeta;
  /** Text files of the dataset (meta.json first), each a complete file. */
  texts: { name: string; text: string }[];
  /** JPEGs to add: dataset path → the row's image (the store reads its bytes by `store`). */
  images: { name: string; image: RecordedImage }[];
}

/**
 * The e2e dataset of a recording (see the header): `framesText` / `liveText` = the stored
 * frames.jsonl / live.jsonl. Rows without pixels are left out (counted in `meta.missingImages`).
 */
export function buildDatasetFiles(header: RecordingHeader, framesText: string, liveText: string | null, opts: { stripLocation?: boolean } = {}): DatasetBuild {
  const strip = !!opts.stripLocation;
  const parsed = parseJsonl(framesText, decodeFrameRow);
  const all = parsed.rows.sort((a, b) => a.seq - b.seq);
  const rows = all.filter((r) => r.image);
  const idx = datasetIndices(rows);
  const out: RecordedFrame[] = rows.map((r) => {
    const { store: _store, ...image } = r.image!;
    const input: FrameMeta = strip ? { ...r.input, geo: r.input.geo === undefined ? undefined : null } : r.input;
    return { ...r, input, image };
  });
  const imu: ImuSample[] = [];
  for (const r of out) if (Array.isArray(r.input.imu)) imu.push(...r.input.imu);
  imu.sort((a, b) => a.t - b.t);
  const geo: GeoFix[] = [];
  const seen = new Set<string>();
  for (const r of out) {
    const g = r.input.geo;
    if (!g) continue;
    const key = `${g.t}|${g.lat}|${g.lon}`;
    if (seen.has(key)) continue;
    seen.add(key);
    geo.push(g);
  }
  geo.sort((a, b) => a.t - b.t);
  const headings: RecordedHeadingRow[] = [];
  for (const r of out) {
    const hd = r.input.heading as HeadingSample | null | undefined;
    if (hd && Number.isFinite(hd.heading)) headings.push({ t: r.input.t, heading: hd.heading, accuracy: hd.accuracy, north: hd.north ?? 'true', declinationDeg: hd.declinationDeg ?? null });
  }
  const first = out[0];
  const last = out[out.length - 1];
  const durationMs = first && last ? last.input.t - first.input.t : 0;
  const g0 = geo[0];
  const meta: RecordingDatasetMeta = {
    name: header.name,
    scene: 'recording',
    source: RECORDING_FORMAT,
    version: header.version,
    mode: header.engine.mode,
    quality: header.engine.quality,
    hfovPriorDeg: typeof header.engine.hfovPriorDeg === 'number' ? header.engine.hfovPriorDeg : null,
    width: first?.image?.width ?? 0,
    height: first?.image?.height ?? 0,
    srcWidth: first?.image?.srcWidth ?? 0,
    srcHeight: first?.image?.srcHeight ?? 0,
    fps: out.length > 1 && durationMs > 0 ? Math.round(((out.length - 1) / durationMs) * 1000 * 100) / 100 : 0,
    numFrames: out.length,
    duration: Math.round(durationMs) / 1000,
    startTimeMs: first?.input.t ?? 0,
    screenAngle: first?.input.screenAngle ?? 0,
    createdAt: header.createdAt,
    originLatLonAlt: !strip && g0 ? [g0.lat, g0.lon, g0.alt ?? 0] : null,
    locationStripped: strip,
    intrinsics: null,
    gt: null,
    sceneExtent: null,
    camera: header.camera,
    app: header.app,
    recorder: header.recorder,
    summary: header.summary ?? null,
    badLines: parsed.bad,
    missingImages: all.length - rows.length,
  };
  const texts: { name: string; text: string }[] = [
    { name: 'meta.json', text: JSON.stringify(meta, null, 1) },
    { name: 'recording.json', text: JSON.stringify(header, null, 1) },
    { name: 'frames.json', text: JSON.stringify(out.map((r, i) => ({ index: idx[i], file: r.image!.file, t: r.input.t }))) },
    { name: 'frames.jsonl', text: out.map((r) => encodeFrameRow(r)).join('\n') + (out.length ? '\n' : '') },
    { name: 'imu.json', text: JSON.stringify(imu) },
  ];
  if (geo.length) texts.push({ name: 'geo.json', text: JSON.stringify(geo) });
  if (headings.length) texts.push({ name: 'heading.json', text: JSON.stringify(headings) });
  if (liveText) texts.push({ name: 'live.jsonl', text: liveText.endsWith('\n') || !liveText ? liveText : `${liveText}\n` });
  const images = rows.map((r) => ({ name: r.image!.file, image: r.image! }));
  return { meta, texts, images };
}

// ---------------------------------------------------------------------------------------------
// ZIP parts
// ---------------------------------------------------------------------------------------------

const ZIP_ENTRY_OVERHEAD = 30 + 46; // local + central header (STORE, no extra fields: src/io/zip.ts)
const ZIP_EOCD = 22;
const utf8 = new TextEncoder();

/** Bytes an entry adds to a STORE ZIP of src/io/zip.ts. */
export function zipEntryBytes(name: string, size: number): number {
  return ZIP_ENTRY_OVERHEAD + 2 * utf8.encode(name).length + size;
}

export interface ZipPartPlan {
  /** 1-based. */
  part: number;
  names: string[];
  /** Exact archive size. */
  bytes: number;
}

/**
 * Split entries (in order) into consecutive STORE ZIPs of at most `maxBytes` each (an entry
 * larger than that alone gets its own part). The order is kept, so the text files listed first
 * land in part 1.
 */
export function planZipParts(entries: readonly { name: string; size: number }[], maxBytes = RECORDING_PART_BYTES, maxEntries = 0xffff): ZipPartPlan[] {
  const parts: ZipPartPlan[] = [];
  let cur: ZipPartPlan | null = null;
  for (const e of entries) {
    const b = zipEntryBytes(e.name, e.size);
    if (!cur || (cur.names.length > 0 && (cur.bytes + b > maxBytes || cur.names.length >= maxEntries))) {
      cur = { part: parts.length + 1, names: [], bytes: ZIP_EOCD };
      parts.push(cur);
    }
    cur.names.push(e.name);
    cur.bytes += b;
  }
  return parts;
}

/** File name of export part `part` of `count` ('<name>.zip' for a single part). */
export function partFileName(name: string, part: number, count: number): string {
  const base = name.replace(/[^\w.-]+/g, '-').replace(/^-+|-+$/g, '') || 'recording';
  return count <= 1 ? `${base}.zip` : `${base}-part${part}of${count}.zip`;
}
