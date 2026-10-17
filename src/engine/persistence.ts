/**
 * Engine persistence: a small key/value facade over `SessionStore` (IndexedDB, app / worker)
 * or plain Maps (tests, `persist: false`), plus the record formats the engine stores so that a
 * respawned worker can resume a capture and a reloaded page can view / export / densify.
 *
 * Layout (keys inside one session):
 *   frames                 keyframe image (JPEG Blob, or raw RGBA without a JPEG encoder) +
 *                          KeyframeRecord as the frame `info`
 *   value  engine/state    EngineStateRecord (options, orientation, counters, flags)
 *   bin    kf/<id>/features   serializeFeatureSet
 *   bin    kf/<id>/colors     3 B per feature (sRGB)
 *   value  kf/<id>/pairs      verified pairs (a, id) found when keyframe id was mapped
 *   bin    imu/<n>            IMU chunk (Float64 rows: t, gyro3, accel3, accelG3; NaN = null)
 *   value  geo                all GPS fixes
 *   bin    vocab              vocabulary codebook (float32)
 *   value  refine/pairs       pairs verified by refine
 *   bin    sparse/model       final SparseModel (packArrays)
 *   value  sparse/meta        SparseResult meta (cameras, metric, stats)
 *   bin    dense/points       serializePointCloud
 *   value  dense/meta         undistorted dense views (for meshing after a reload)
 *   bin    depth/<id>         fusion-filtered depth maps (float16 normals / cost)
 *   bin    mesh               serializeTriangleMesh
 */
import type { CameraIntrinsics, GeoFix, ImuSample, Mat3, MetricInfo, RGBAImage, TwoViewConfig, TwoViewGeometry, Vec3 } from '../core/types';
import { SessionStore, type SessionMeta } from '../io/storage';
import type { CaptureMode, QualityPreset } from '../workers/protocol';
import type { KeyframeRecord, SparseModel, SparseRollingShutter } from './types';

/** Image of a stored keyframe. */
export type StoredImage = Blob | RGBAImage;

export function isRGBAImage(x: unknown): x is RGBAImage {
  return !!x && typeof x === 'object' && typeof (x as RGBAImage).width === 'number' && ArrayBuffer.isView((x as RGBAImage).data);
}

/** Key/value persistence used by the engine. All methods may reject (quota, connection). */
export interface EngineStore {
  /** True when data survives a reload (IndexedDB). */
  readonly persistent: boolean;
  readonly sessionId: string;
  /** Create the session record if it does not exist. */
  ensureSession(init: { mode: CaptureMode; quality: QualityPreset; name?: string }): Promise<void>;
  getSessionMeta(): Promise<SessionMeta | null>;
  putFrame(id: number, image: StoredImage, record: KeyframeRecord): Promise<void>;
  getFrameImage(id: number): Promise<StoredImage | null>;
  getFrameRecords(): Promise<KeyframeRecord[]>;
  putValue(key: string, value: unknown): Promise<void>;
  getValue<T>(key: string): Promise<T | null>;
  putBinary(key: string, data: ArrayBuffer | ArrayBufferView): Promise<void>;
  getBinary(key: string): Promise<ArrayBuffer | null>;
  deleteBinary(key: string): Promise<void>;
  listBinaryKeys(prefix?: string): Promise<string[]>;
  listValueKeys(prefix?: string): Promise<string[]>;
  close(): void;
}

function toArrayBuffer(data: ArrayBuffer | ArrayBufferView): ArrayBuffer {
  if (data instanceof ArrayBuffer) return data;
  const u8 = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return u8.slice().buffer;
}

/** In-memory store (tests, `persist: false`). Data lives as long as the instance. */
export class MemoryEngineStore implements EngineStore {
  readonly persistent = false;
  private meta: SessionMeta | null = null;
  private readonly frames = new Map<number, { image: StoredImage; record: KeyframeRecord }>();
  private readonly values = new Map<string, unknown>();
  private readonly bins = new Map<string, ArrayBuffer>();

  constructor(readonly sessionId = 'memory') {}

  async ensureSession(init: { mode: CaptureMode; quality: QualityPreset; name?: string }): Promise<void> {
    if (this.meta) return;
    const now = Date.now();
    this.meta = {
      id: this.sessionId,
      name: init.name ?? 'Scan',
      createdAt: now,
      updatedAt: now,
      mode: init.mode,
      status: 'capturing',
      numFrames: 0,
      stats: { quality: init.quality },
    };
  }

  async getSessionMeta(): Promise<SessionMeta | null> {
    return this.meta ? { ...this.meta, numFrames: this.frames.size } : null;
  }

  async putFrame(id: number, image: StoredImage, record: KeyframeRecord): Promise<void> {
    this.frames.set(id, { image, record });
    if (this.meta) this.meta.updatedAt = Date.now();
  }

  async getFrameImage(id: number): Promise<StoredImage | null> {
    return this.frames.get(id)?.image ?? null;
  }

  async getFrameRecords(): Promise<KeyframeRecord[]> {
    return [...this.frames.values()].map((f) => f.record).sort((a, b) => a.id - b.id);
  }

  async putValue(key: string, value: unknown): Promise<void> {
    this.values.set(key, value);
  }

  async getValue<T>(key: string): Promise<T | null> {
    return this.values.has(key) ? (this.values.get(key) as T) : null;
  }

  async putBinary(key: string, data: ArrayBuffer | ArrayBufferView): Promise<void> {
    this.bins.set(key, toArrayBuffer(data));
  }

  async getBinary(key: string): Promise<ArrayBuffer | null> {
    return this.bins.get(key) ?? null;
  }

  async deleteBinary(key: string): Promise<void> {
    this.bins.delete(key);
  }

  async listBinaryKeys(prefix = ''): Promise<string[]> {
    return [...this.bins.keys()].filter((k) => k.startsWith(prefix)).sort();
  }

  async listValueKeys(prefix = ''): Promise<string[]> {
    return [...this.values.keys()].filter((k) => k.startsWith(prefix)).sort();
  }

  close(): void {
    /* nothing */
  }
}

/** Raw-RGBA frame stored as a Blob (no JPEG encoder in this realm). */
interface RawFrameMarker {
  rawRGBA?: { width: number; height: number };
}

/** IndexedDB-backed store (src/io SessionStore). */
export class SessionEngineStore implements EngineStore {
  readonly persistent = true;

  private constructor(
    private readonly store: SessionStore,
    readonly sessionId: string,
  ) {}

  static async open(sessionId: string, dbName?: string): Promise<SessionEngineStore> {
    // Keyframe JPEGs are stored as ArrayBuffers from the start: iOS WebKit intermittently fails
    // Blob writes from the worker ("UnknownError: Error preparing Blob/File data to be stored in
    // object store") during a capture, losing that keyframe. Reads return Blobs either way.
    return new SessionEngineStore(await SessionStore.open(dbName, { blobsAsArrayBuffers: true }), sessionId);
  }

  static isSupported(): boolean {
    return SessionStore.isSupported();
  }

  async ensureSession(init: { mode: CaptureMode; quality: QualityPreset; name?: string }): Promise<void> {
    const cur = await this.store.getSession(this.sessionId);
    if (cur) return;
    await this.store.createSession({ id: this.sessionId, name: init.name ?? 'Scan', mode: init.mode, status: 'capturing', stats: { quality: init.quality } });
  }

  getSessionMeta(): Promise<SessionMeta | null> {
    return this.store.getSession(this.sessionId);
  }

  async putFrame(id: number, image: StoredImage, record: KeyframeRecord): Promise<void> {
    if (isRGBAImage(image)) {
      const info: KeyframeRecord & RawFrameMarker = { ...record, rawRGBA: { width: image.width, height: image.height } };
      const bytes = new Uint8Array(image.data.buffer, image.data.byteOffset, image.width * image.height * 4).slice();
      await this.store.putFrame(this.sessionId, id, new Blob([bytes]), info);
    } else {
      await this.store.putFrame(this.sessionId, id, image, record);
    }
  }

  async getFrameImage(id: number): Promise<StoredImage | null> {
    const f = await this.store.getFrame<KeyframeRecord & RawFrameMarker>(this.sessionId, id);
    if (!f) return null;
    const raw = f.info?.rawRGBA;
    if (raw) {
      const buf = await f.jpeg.arrayBuffer();
      return { width: raw.width, height: raw.height, data: new Uint8ClampedArray(buf) };
    }
    return f.jpeg;
  }

  async getFrameRecords(): Promise<KeyframeRecord[]> {
    const infos = await this.store.getFrameInfos<KeyframeRecord & RawFrameMarker>(this.sessionId);
    return infos
      .filter((r) => r && typeof r.id === 'number')
      .map((r) => {
        const { rawRGBA: _raw, ...rest } = r;
        return rest as KeyframeRecord;
      })
      .sort((a, b) => a.id - b.id);
  }

  putValue(key: string, value: unknown): Promise<void> {
    return this.store.putJSON(this.sessionId, key, value);
  }

  getValue<T>(key: string): Promise<T | null> {
    return this.store.getJSON<T>(this.sessionId, key);
  }

  putBinary(key: string, data: ArrayBuffer | ArrayBufferView): Promise<void> {
    return this.store.putBlob(this.sessionId, key, data);
  }

  getBinary(key: string): Promise<ArrayBuffer | null> {
    return this.store.getArrayBuffer(this.sessionId, key);
  }

  deleteBinary(key: string): Promise<void> {
    return this.store.deleteBlob(this.sessionId, key);
  }

  async listBinaryKeys(prefix = ''): Promise<string[]> {
    return (await this.store.listBlobKeys(this.sessionId)).filter((k) => k.startsWith(prefix)).sort();
  }

  async listValueKeys(prefix = ''): Promise<string[]> {
    return (await this.store.listJSONKeys(this.sessionId)).filter((k) => k.startsWith(prefix)).sort();
  }

  /** Delete this session and everything stored under it. */
  deleteSession(): Promise<void> {
    return this.store.deleteSession(this.sessionId);
  }

  /** Delete the OTHER sessions of this database not updated for `maxAgeMs` (scratch databases). */
  async purgeOtherSessions(maxAgeMs: number): Promise<void> {
    const now = Date.now();
    for (const m of await this.store.listSessions()) {
      if (m.id !== this.sessionId && now - m.updatedAt > maxAgeMs) await this.store.deleteSession(m.id).catch(() => undefined);
    }
  }

  close(): void {
    try {
      this.store.close();
    } catch {
      /* ignore */
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------------------------

export const ENGINE_STATE_VERSION = 1;

/** Persisted engine state (`engine/state`). */
export interface EngineStateRecord {
  version: number;
  mode: CaptureMode;
  quality: QualityPreset;
  hfovPriorDeg: number | null;
  /** Canonical UI angle of the session (θ_s), null before the first frame. */
  thetaSession: number | null;
  /** θ whose R_cam_device describes the canonical images (= θ_s unless the hand-eye check switched it). */
  cameraTheta: number | null;
  cameraThetaVerified: boolean;
  /** Frame delivery path (raw sensor buffers / upright frames / unknown); absent in older records. */
  rawDelivery?: boolean | null;
  /** 'object' | 'scene' once 'auto' was resolved. */
  resolvedMode: 'object' | 'scene' | null;
  nextKeyframeId: number;
  frameCounter: number;
  imuChunks: number;
  captureFinished: boolean;
  /** Keyframe ids the vocabulary was trained on (null = not trained). */
  vocabTrainedOn: number[] | null;
  hasSparse: boolean;
  hasDense: boolean;
  hasMesh: boolean;
  /** Camera–IMU offset estimated by the last world alignment (ms). */
  timeOffsetMs: number | null;
  /** Gyro bias (rad/s, device frame) estimated from the live keyframes; absent in older records. */
  gyroBias?: [number, number, number] | null;
  /** Rolling-shutter model applied by the last refine / live (canonical image); informational after a reload. */
  rollingShutter?: { axis: 'rows' | 'cols'; readoutMs: number } | null;
  updatedAt: number;
}

/** A verified pair as stored (oriented a → b). */
export interface StoredPair {
  a: number;
  b: number;
  config: TwoViewConfig;
  inliers: Uint32Array;
  E?: Mat3;
  F?: Mat3;
  H?: Mat3;
  R?: Mat3;
  t?: Vec3;
  triAngle?: number;
}

export interface StoredPairs {
  pairs: StoredPair[];
  /** pairKey of every pair that was tried (verified or not). */
  attempted: number[];
}

export function pairToStored(a: number, b: number, g: TwoViewGeometry): StoredPair {
  const s: StoredPair = { a, b, config: g.config, inliers: g.inliers };
  if (g.E) s.E = g.E;
  if (g.F) s.F = g.F;
  if (g.H) s.H = g.H;
  if (g.R) s.R = g.R;
  if (g.t) s.t = g.t;
  if (g.triAngle !== undefined) s.triAngle = g.triAngle;
  return s;
}

export function storedToGeometry(s: StoredPair): TwoViewGeometry {
  const g: TwoViewGeometry = { config: s.config, inliers: s.inliers };
  if (s.E) g.E = s.E;
  if (s.F) g.F = s.F;
  if (s.H) g.H = s.H;
  if (s.R) g.R = s.R;
  if (s.t) g.t = s.t;
  if (s.triAngle !== undefined) g.triAngle = s.triAngle;
  return g;
}

/** Persisted meta of the sparse result (`sparse/meta`); points live in `sparse/model`. */
export interface SparseMetaRecord {
  cameras: { id: number; pose: { R: number[]; t: number[] }; intrinsics: CameraIntrinsics; t: number; frameIndex: number }[];
  metric: MetricInfo;
  stats: Record<string, number | string>;
  /** SparseModel.rollingShutter (absent in older records). */
  rollingShutter?: SparseRollingShutter | null;
}

/** Persisted dense views (`dense/meta`). */
export interface DenseMetaRecord {
  views: { id: number; cam: CameraIntrinsics; pose: { R: number[]; t: number[] }; depthWidth: number; depthHeight: number }[];
  maxImageSize: number;
}

// ---------------------------------------------------------------------------------------------
// IMU packing
// ---------------------------------------------------------------------------------------------

const IMU_STRIDE = 10;

export function packImu(samples: readonly ImuSample[]): Float64Array {
  const out = new Float64Array(samples.length * IMU_STRIDE);
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    const o = i * IMU_STRIDE;
    out[o] = s.t;
    out[o + 1] = s.gyro[0];
    out[o + 2] = s.gyro[1];
    out[o + 3] = s.gyro[2];
    for (let k = 0; k < 3; k++) {
      out[o + 4 + k] = s.accel ? s.accel[k] : NaN;
      out[o + 7 + k] = s.accelG ? s.accelG[k] : NaN;
    }
  }
  return out;
}

export function unpackImu(buf: ArrayBuffer): ImuSample[] {
  const a = new Float64Array(buf, 0, Math.floor(buf.byteLength / 8));
  const n = Math.floor(a.length / IMU_STRIDE);
  const out: ImuSample[] = [];
  for (let i = 0; i < n; i++) {
    const o = i * IMU_STRIDE;
    out.push({
      t: a[o],
      gyro: [a[o + 1], a[o + 2], a[o + 3]],
      accel: Number.isFinite(a[o + 4]) ? [a[o + 4], a[o + 5], a[o + 6]] : null,
      accelG: Number.isFinite(a[o + 7]) ? [a[o + 7], a[o + 8], a[o + 9]] : null,
    });
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Typed-array container (SparseModel)
// ---------------------------------------------------------------------------------------------

type PackedArray = Float64Array | Float32Array | Uint32Array | Int32Array | Uint8Array | Uint16Array;
const DTYPES: { name: string; ctor: new (buf: ArrayBuffer, off: number, len: number) => PackedArray; bytes: number }[] = [
  { name: 'f64', ctor: Float64Array, bytes: 8 },
  { name: 'f32', ctor: Float32Array, bytes: 4 },
  { name: 'u32', ctor: Uint32Array, bytes: 4 },
  { name: 'i32', ctor: Int32Array, bytes: 4 },
  { name: 'u8', ctor: Uint8Array, bytes: 1 },
  { name: 'u16', ctor: Uint16Array, bytes: 2 },
];

function dtypeOf(a: PackedArray): number {
  if (a instanceof Float64Array) return 0;
  if (a instanceof Float32Array) return 1;
  if (a instanceof Uint32Array) return 2;
  if (a instanceof Int32Array) return 3;
  if (a instanceof Uint8Array) return 4;
  return 5;
}

const PACK_MAGIC = 0x4b50454e; // 'NEPK'

/** [magic u32][header length u32][header JSON][pad 8] then per array: [dtype u32][count u32][pad 8][data][pad 8]. */
export function packArrays(header: unknown, arrays: PackedArray[]): ArrayBuffer {
  const hdr = new TextEncoder().encode(JSON.stringify({ header, n: arrays.length }));
  const pad8 = (n: number): number => (n + 7) & ~7;
  let size = pad8(8 + hdr.length);
  for (const a of arrays) size += 8 + pad8(a.byteLength);
  const buf = new ArrayBuffer(size);
  const dv = new DataView(buf);
  dv.setUint32(0, PACK_MAGIC, true);
  dv.setUint32(4, hdr.length, true);
  new Uint8Array(buf, 8, hdr.length).set(hdr);
  let off = pad8(8 + hdr.length);
  for (const a of arrays) {
    dv.setUint32(off, dtypeOf(a), true);
    dv.setUint32(off + 4, a.length, true);
    off += 8;
    new Uint8Array(buf, off, a.byteLength).set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
    off += pad8(a.byteLength);
  }
  return buf;
}

/** Thrown for a stored container that cannot be decoded (damaged, truncated, foreign format). */
function damaged(what: string): Error {
  return new Error(`Stored data is damaged (${what})`);
}

/** Every size is checked against the buffer before it is used (a damaged record never allocates). */
export function unpackArrays<H>(buf: ArrayBuffer): { header: H; arrays: PackedArray[] } {
  if (!buf || !(buf.byteLength >= 8)) throw damaged('container too short');
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== PACK_MAGIC) throw damaged('not an array container');
  const hl = dv.getUint32(4, true);
  if (hl > buf.byteLength - 8) throw damaged('header length');
  let parsed: { header: H; n: number };
  try {
    parsed = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl))) as { header: H; n: number };
  } catch {
    throw damaged('header');
  }
  if (!parsed || typeof parsed !== 'object' || !Number.isInteger(parsed.n) || parsed.n < 0) throw damaged('header');
  const pad8 = (n: number): number => (n + 7) & ~7;
  let off = pad8(8 + hl);
  const arrays: PackedArray[] = [];
  for (let i = 0; i < parsed.n; i++) {
    if (off + 8 > buf.byteLength) throw damaged('truncated');
    const dt = DTYPES[dv.getUint32(off, true)];
    if (!dt) throw damaged('array type');
    const count = dv.getUint32(off + 4, true);
    off += 8;
    if (count * dt.bytes > buf.byteLength - off) throw damaged('array length');
    // copy into its own buffer (transferable on its own)
    const bytes = new Uint8Array(buf, off, count * dt.bytes).slice();
    arrays.push(new dt.ctor(bytes.buffer, 0, count));
    off += pad8(count * dt.bytes);
  }
  return { header: parsed.header, arrays };
}

interface SparseModelHeader {
  cameras: [number, CameraIntrinsics][];
  images: { id: number; cameraId: number; R: number[]; t: number[] }[];
  numPoints: number;
}

export function packSparseModel(m: SparseModel): ArrayBuffer {
  const header: SparseModelHeader = {
    cameras: [...m.cameras.entries()],
    images: m.images.map((im) => ({ id: im.id, cameraId: im.cameraId, R: Array.from(im.pose.R), t: Array.from(im.pose.t) })),
    numPoints: m.numPoints,
  };
  return packArrays(header, [m.xyz, m.colors, m.errors, m.trackOffsets, m.trackImages, m.trackFeatures]);
}

export function unpackSparseModel(buf: ArrayBuffer): SparseModel {
  const { header, arrays } = unpackArrays<SparseModelHeader>(buf);
  // shapes the rest of the engine indexes without checks
  const n = header?.numPoints;
  const ok =
    Array.isArray(header?.cameras) &&
    Array.isArray(header?.images) &&
    header.images.every((im) => im && Array.isArray(im.R) && im.R.length === 9 && Array.isArray(im.t) && im.t.length === 3) &&
    Number.isInteger(n) &&
    n >= 0 &&
    arrays.length >= 6 &&
    arrays[0].length === 3 * n &&
    arrays[1].length === 3 * n &&
    arrays[2].length === n &&
    arrays[3].length === n + 1 &&
    arrays[4].length === arrays[5].length &&
    arrays[3][n] === arrays[4].length;
  if (!ok) throw damaged('sparse model');
  return {
    cameras: new Map(header.cameras),
    images: header.images.map((im) => ({ id: im.id, cameraId: im.cameraId, pose: { R: Float64Array.from(im.R), t: Float64Array.from(im.t) } })),
    numPoints: header.numPoints,
    xyz: arrays[0] as Float64Array,
    colors: arrays[1] as Uint8Array,
    errors: arrays[2] as Float32Array,
    trackOffsets: arrays[3] as Uint32Array,
    trackImages: arrays[4] as Uint32Array,
    trackFeatures: arrays[5] as Uint32Array,
  };
}

/** Deduplicated (by t), time-sorted union of GPS fixes. */
export function mergeGeoFixes(a: readonly GeoFix[], b: readonly GeoFix[]): GeoFix[] {
  const m = new Map<number, GeoFix>();
  for (const f of a) m.set(f.t, f);
  for (const f of b) m.set(f.t, f);
  return [...m.values()].sort((x, y) => x.t - y.t);
}
