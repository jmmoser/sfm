/**
 * Compact, versioned binary (de)serialisers for the large data structures we
 * persist in IndexedDB or move between threads: FeatureSet, PointCloud,
 * TriangleMesh and depth maps. Also: manual IEEE-754 binary16 pack/unpack and
 * a JSON-safe encoding for typed arrays (poses etc.).
 *
 * Container layout (little-endian):
 *   0  u32  magic 'SFMB' (0x424D4653)
 *   4  u16  kind   (1 FeatureSet, 2 PointCloud, 3 TriangleMesh, 4 DepthMap)
 *   6  u16  format version (currently 1)
 *   8  ...  kind-specific scalars (u32) and arrays.
 * Each array is stored as: u32 dtype, u32 byteLength, zero padding up to the
 * next multiple of 8, then the raw bytes. dtype ∈ DType below. Readers accept
 * any float dtype (f16/f32/f64) for float fields and u16/u32 for indices, so
 * encoders may change storage precision without a version bump.
 *
 * By default decoders COPY each array into its own ArrayBuffer (safe to
 * transfer individually to a worker). Pass `{ zeroCopy: true }` to get views
 * into the input buffer when it is suitably aligned.
 */
import type { FeatureSet, PointCloud, Pose, TriangleMesh } from '../core/types';

// ---------------------------------------------------------------------------
// float16
// ---------------------------------------------------------------------------

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

/**
 * Convert a number to IEEE-754 binary16 bits (round-to-nearest-even, overflow
 * to ±Infinity, gradual underflow to subnormals, NaN preserved as quiet NaN).
 * Correctly rounded for any double: the intermediate float32 is rounded to ODD
 * (an inexact result gets its last bit set as a sticky bit), which makes the
 * second rounding to 11 significant bits exact (24 ≥ 11 + 2). Plain RNE to
 * float32 first could land exactly on a half-way tie and round the wrong way.
 */
export function float32ToFloat16Bits(value: number): number {
  f32Scratch[0] = value;
  let x = u32Scratch[0];
  if ((x & 1) === 0 && f32Scratch[0] !== value && value === value && (x & 0x7f800000) !== 0x7f800000) {
    x += Math.abs(f32Scratch[0]) < Math.abs(value) ? 1 : -1; // nextafter toward value → odd
  }
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;

  if (exp === 0xff) {
    // Inf / NaN
    return sign | 0x7c00 | (mant !== 0 ? 0x200 | (mant >>> 13) : 0);
  }
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00; // overflow → Inf
  if (e <= 0) {
    // Subnormal half (or zero).
    if (e < -10) return sign; // too small → ±0 (|v| < 2^-25 rounds to 0)
    mant |= 0x800000; // implicit leading 1
    const shift = 14 - e; // 14..24
    const halfMant = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    let h = halfMant;
    if (rem > halfway || (rem === halfway && (halfMant & 1) !== 0)) h++;
    return sign | h; // may carry into the exponent field → smallest normal (correct)
  }
  // Normal.
  const halfMant = mant >>> 13;
  const rem = mant & 0x1fff;
  let h = (e << 10) | halfMant;
  if (rem > 0x1000 || (rem === 0x1000 && (halfMant & 1) !== 0)) h++; // carry may overflow into Inf (correct)
  return sign | h;
}

/** Convert IEEE-754 binary16 bits to a number. */
export function float16BitsToFloat32(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24; // ±0 / subnormal
  if (exp === 0x1f) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

let f16Table: Float32Array | null = null;
function getF16Table(): Float32Array {
  if (f16Table) return f16Table;
  const t = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) t[i] = float16BitsToFloat32(i);
  f16Table = t;
  return t;
}

/** Pack float32 values into binary16 bit patterns. */
export function packFloat16(src: ArrayLike<number>, out?: Uint16Array): Uint16Array {
  const n = src.length;
  const o = out ?? new Uint16Array(n);
  if (o.length < n) throw new RangeError('packFloat16: output too small');
  for (let i = 0; i < n; i++) o[i] = float32ToFloat16Bits(src[i]);
  return o;
}

/** Unpack binary16 bit patterns into float32 values (table lookup). */
export function unpackFloat16(src: Uint16Array, out?: Float32Array): Float32Array {
  const n = src.length;
  const o = out ?? new Float32Array(n);
  if (o.length < n) throw new RangeError('unpackFloat16: output too small');
  const t = getF16Table();
  for (let i = 0; i < n; i++) o[i] = t[src[i]];
  return o;
}

// ---------------------------------------------------------------------------
// Container writer / reader
// ---------------------------------------------------------------------------

export const SERIAL_MAGIC = 0x424d4653; // bytes 'S','F','M','B'
export const SERIAL_VERSION = 1;

export const SerialKind = {
  FeatureSet: 1,
  PointCloud: 2,
  TriangleMesh: 3,
  DepthMap: 4,
} as const;
export type SerialKind = (typeof SerialKind)[keyof typeof SerialKind];

const DType = { U8: 1, U16: 2, U32: 3, I32: 4, F16: 5, F32: 6, F64: 7 } as const;
type DType = (typeof DType)[keyof typeof DType];

const DTYPE_BYTES: Record<number, number> = { 1: 1, 2: 2, 3: 4, 4: 4, 5: 2, 6: 4, 7: 8 };

type Field = { u32: number } | { arr: ArrayBufferView; dtype: DType };

function pad8(n: number): number {
  return (n + 7) & ~7;
}

function packContainer(kind: SerialKind, fields: Field[]): ArrayBuffer {
  let size = 8;
  for (const f of fields) {
    if ('u32' in f) size += 4;
    else size = pad8(size + 8) + f.arr.byteLength;
  }
  const buf = new ArrayBuffer(size);
  const dv = new DataView(buf);
  const u8 = new Uint8Array(buf);
  dv.setUint32(0, SERIAL_MAGIC, true);
  dv.setUint16(4, kind, true);
  dv.setUint16(6, SERIAL_VERSION, true);
  let o = 8;
  for (const f of fields) {
    if ('u32' in f) {
      dv.setUint32(o, f.u32 >>> 0, true);
      o += 4;
    } else {
      dv.setUint32(o, f.dtype, true);
      dv.setUint32(o + 4, f.arr.byteLength, true);
      o = pad8(o + 8);
      u8.set(new Uint8Array(f.arr.buffer, f.arr.byteOffset, f.arr.byteLength), o);
      o += f.arr.byteLength;
    }
  }
  return buf;
}

export interface DecodeOptions {
  /** Return views into the input buffer instead of copies (when aligned). Default false. */
  zeroCopy?: boolean;
}

class Reader {
  private dv: DataView;
  private u8: Uint8Array;
  private o = 8;
  readonly kind: number;
  readonly version: number;

  constructor(input: ArrayBuffer | Uint8Array, private zeroCopy: boolean) {
    this.u8 = input instanceof Uint8Array ? input : new Uint8Array(input);
    this.dv = new DataView(this.u8.buffer, this.u8.byteOffset, this.u8.byteLength);
    if (this.u8.byteLength < 8 || this.dv.getUint32(0, true) !== SERIAL_MAGIC) {
      throw new Error('deserialize: bad magic (not an SFMB buffer)');
    }
    this.kind = this.dv.getUint16(4, true);
    this.version = this.dv.getUint16(6, true);
    if (this.version > SERIAL_VERSION || this.version === 0) {
      throw new Error(`deserialize: unsupported format version ${this.version}`);
    }
  }

  expectKind(kind: SerialKind, label: string): void {
    if (this.kind !== kind) throw new Error(`deserialize: expected ${label} (kind ${kind}), got kind ${this.kind}`);
  }

  u32(): number {
    if (this.o + 4 > this.u8.byteLength) throw new Error('deserialize: truncated');
    const v = this.dv.getUint32(this.o, true);
    this.o += 4;
    return v;
  }

  private rawArray(): { dtype: number; bytes: Uint8Array; offset: number } {
    const dtype = this.u32();
    const byteLength = this.u32();
    const start = pad8(this.o);
    if (start + byteLength > this.u8.byteLength) throw new Error('deserialize: truncated array');
    const elt = DTYPE_BYTES[dtype];
    if (!elt || byteLength % elt !== 0) throw new Error(`deserialize: bad dtype ${dtype}`);
    this.o = start + byteLength;
    return { dtype, bytes: this.u8.subarray(start, start + byteLength), offset: this.u8.byteOffset + start };
  }

  private view<T extends ArrayBufferView>(
    ctor: { new (b: ArrayBuffer, o: number, n: number): T; new (n: number): T; BYTES_PER_ELEMENT: number },
    r: { bytes: Uint8Array; offset: number },
  ): T {
    const n = r.bytes.byteLength / ctor.BYTES_PER_ELEMENT;
    if (this.zeroCopy && r.offset % ctor.BYTES_PER_ELEMENT === 0 && r.bytes.buffer instanceof ArrayBuffer) {
      return new ctor(r.bytes.buffer, r.offset, n);
    }
    const out = new ctor(n);
    new Uint8Array(out.buffer).set(r.bytes);
    return out;
  }

  floats(expected: number, label: string): Float32Array {
    const r = this.rawArray();
    let out: Float32Array;
    if (r.dtype === DType.F32) out = this.view(Float32Array, r);
    else if (r.dtype === DType.F16) out = unpackFloat16(this.copyView(Uint16Array, r));
    else if (r.dtype === DType.F64) out = Float32Array.from(this.copyView(Float64Array, r));
    else throw new Error(`deserialize: ${label}: expected float array, got dtype ${r.dtype}`);
    if (out.length !== expected) throw new Error(`deserialize: ${label}: expected ${expected} values, got ${out.length}`);
    return out;
  }

  bytes(expected: number, label: string): Uint8Array {
    const r = this.rawArray();
    if (r.dtype !== DType.U8) throw new Error(`deserialize: ${label}: expected u8 array`);
    const out = this.view(Uint8Array, r);
    if (out.length !== expected) throw new Error(`deserialize: ${label}: expected ${expected} values, got ${out.length}`);
    return out;
  }

  indices(expected: number, label: string): Uint32Array {
    const r = this.rawArray();
    let out: Uint32Array;
    if (r.dtype === DType.U32) out = this.view(Uint32Array, r);
    else if (r.dtype === DType.U16) out = Uint32Array.from(this.copyView(Uint16Array, r));
    else throw new Error(`deserialize: ${label}: expected index array, got dtype ${r.dtype}`);
    if (out.length !== expected) throw new Error(`deserialize: ${label}: expected ${expected} values, got ${out.length}`);
    return out;
  }

  /** Temporary typed view (copied if unaligned) — not returned to callers as-is. */
  private copyView<T extends ArrayBufferView>(
    ctor: { new (b: ArrayBuffer, o: number, n: number): T; new (n: number): T; BYTES_PER_ELEMENT: number },
    r: { bytes: Uint8Array; offset: number },
  ): T {
    const n = r.bytes.byteLength / ctor.BYTES_PER_ELEMENT;
    if (r.offset % ctor.BYTES_PER_ELEMENT === 0 && r.bytes.buffer instanceof ArrayBuffer) {
      return new ctor(r.bytes.buffer, r.offset, n);
    }
    const out = new ctor(n);
    new Uint8Array(out.buffer).set(r.bytes);
    return out;
  }
}

/** Kind of a serialised buffer, or null if it is not an SFMB buffer. */
export function serialKind(input: ArrayBuffer | Uint8Array): SerialKind | null {
  const u8 = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (u8.byteLength < 8) return null;
  const dv = new DataView(u8.buffer, u8.byteOffset, 8);
  if (dv.getUint32(0, true) !== SERIAL_MAGIC) return null;
  return dv.getUint16(4, true) as SerialKind;
}

function floatField(a: Float32Array, f16: boolean): Field {
  return f16 ? { arr: packFloat16(a), dtype: DType.F16 } : { arr: a, dtype: DType.F32 };
}

function check(cond: boolean, msg: string): void {
  if (!cond) throw new RangeError(`serialize: ${msg}`);
}

// ---------------------------------------------------------------------------
// FeatureSet
// ---------------------------------------------------------------------------

/** Lossless. ~4·4 + 4 + 128 bytes per feature. */
export function serializeFeatureSet(fs: FeatureSet): ArrayBuffer {
  const n = fs.count;
  check(fs.keypoints.length >= 4 * n && fs.scores.length >= n && fs.descriptors.length >= 128 * n, 'FeatureSet arrays shorter than count');
  return packContainer(SerialKind.FeatureSet, [
    { u32: n },
    { u32: fs.width },
    { u32: fs.height },
    { arr: fs.keypoints.subarray(0, 4 * n), dtype: DType.F32 },
    { arr: fs.scores.subarray(0, n), dtype: DType.F32 },
    { arr: fs.descriptors.subarray(0, 128 * n), dtype: DType.U8 },
  ]);
}

export function deserializeFeatureSet(input: ArrayBuffer | Uint8Array, opts: DecodeOptions = {}): FeatureSet {
  const r = new Reader(input, !!opts.zeroCopy);
  r.expectKind(SerialKind.FeatureSet, 'FeatureSet');
  const count = r.u32();
  const width = r.u32();
  const height = r.u32();
  const keypoints = r.floats(4 * count, 'keypoints');
  const scores = r.floats(count, 'scores');
  const descriptors = r.bytes(128 * count, 'descriptors');
  return { count, width, height, keypoints, scores, descriptors };
}

// ---------------------------------------------------------------------------
// PointCloud
// ---------------------------------------------------------------------------

export interface SerializeOptions {
  /** Store normals (and confidence / depth-map cost) as float16. Default false (lossless). */
  float16?: boolean;
}

const PC_NORMALS = 1;
const PC_RADII = 2;
const PC_CONFIDENCE = 4;

export function serializePointCloud(pc: PointCloud, opts: SerializeOptions = {}): ArrayBuffer {
  const n = pc.count;
  const f16 = !!opts.float16;
  check(pc.positions.length >= 3 * n && pc.colors.length >= 3 * n, 'PointCloud arrays shorter than count');
  let mask = 0;
  if (pc.normals) { check(pc.normals.length >= 3 * n, 'normals too short'); mask |= PC_NORMALS; }
  if (pc.radii) { check(pc.radii.length >= n, 'radii too short'); mask |= PC_RADII; }
  if (pc.confidence) { check(pc.confidence.length >= n, 'confidence too short'); mask |= PC_CONFIDENCE; }
  const fields: Field[] = [
    { u32: n },
    { u32: mask },
    { arr: pc.positions.subarray(0, 3 * n), dtype: DType.F32 },
    { arr: pc.colors.subarray(0, 3 * n), dtype: DType.U8 },
  ];
  if (pc.normals) fields.push(floatField(pc.normals.subarray(0, 3 * n), f16));
  if (pc.radii) fields.push({ arr: pc.radii.subarray(0, n), dtype: DType.F32 }); // radii stay f32 (tiny values)
  if (pc.confidence) fields.push(floatField(pc.confidence.subarray(0, n), f16));
  return packContainer(SerialKind.PointCloud, fields);
}

export function deserializePointCloud(input: ArrayBuffer | Uint8Array, opts: DecodeOptions = {}): PointCloud {
  const r = new Reader(input, !!opts.zeroCopy);
  r.expectKind(SerialKind.PointCloud, 'PointCloud');
  const count = r.u32();
  const mask = r.u32();
  const pc: PointCloud = {
    count,
    positions: r.floats(3 * count, 'positions'),
    colors: r.bytes(3 * count, 'colors'),
  };
  if (mask & PC_NORMALS) pc.normals = r.floats(3 * count, 'normals');
  if (mask & PC_RADII) pc.radii = r.floats(count, 'radii');
  if (mask & PC_CONFIDENCE) pc.confidence = r.floats(count, 'confidence');
  return pc;
}

// ---------------------------------------------------------------------------
// TriangleMesh
// ---------------------------------------------------------------------------

/** Indices are stored as u16 when every index is ≤ 65535 (lossless, also for corrupt input). */
export function serializeTriangleMesh(mesh: TriangleMesh, opts: SerializeOptions = {}): ArrayBuffer {
  const nv = mesh.vertexCount;
  const nt = mesh.triangleCount;
  check(
    mesh.positions.length >= 3 * nv && mesh.normals.length >= 3 * nv && mesh.colors.length >= 3 * nv && mesh.indices.length >= 3 * nt,
    'TriangleMesh arrays shorter than counts',
  );
  const idx = mesh.indices.subarray(0, 3 * nt);
  let maxIndex = 0;
  if (nv <= 65536) for (let i = 0; i < idx.length; i++) if (idx[i] > maxIndex) maxIndex = idx[i];
  const idxField: Field = nv <= 65536 && maxIndex <= 0xffff ? { arr: Uint16Array.from(idx), dtype: DType.U16 } : { arr: idx, dtype: DType.U32 };
  return packContainer(SerialKind.TriangleMesh, [
    { u32: nv },
    { u32: nt },
    { arr: mesh.positions.subarray(0, 3 * nv), dtype: DType.F32 },
    floatField(mesh.normals.subarray(0, 3 * nv), !!opts.float16),
    { arr: mesh.colors.subarray(0, 3 * nv), dtype: DType.U8 },
    idxField,
  ]);
}

export function deserializeTriangleMesh(input: ArrayBuffer | Uint8Array, opts: DecodeOptions = {}): TriangleMesh {
  const r = new Reader(input, !!opts.zeroCopy);
  r.expectKind(SerialKind.TriangleMesh, 'TriangleMesh');
  const vertexCount = r.u32();
  const triangleCount = r.u32();
  return {
    vertexCount,
    triangleCount,
    positions: r.floats(3 * vertexCount, 'positions'),
    normals: r.floats(3 * vertexCount, 'normals'),
    colors: r.bytes(3 * vertexCount, 'colors'),
    indices: r.indices(3 * triangleCount, 'indices'),
  };
}

// ---------------------------------------------------------------------------
// Depth map
// ---------------------------------------------------------------------------

/** Structural twin of dense/DepthMap (kept here to avoid a dependency on src/dense). */
export interface DepthMapLike {
  width: number;
  height: number;
  /** z-depth per pixel, 0 = invalid. Always stored as float32. */
  depth: Float32Array;
  /** 3 per pixel, camera frame. */
  normal: Float32Array;
  /** NCC cost per pixel (0..2). */
  cost: Float32Array;
}

/**
 * Depth is always float32 (lossless); with `float16: true` normals and cost are
 * stored as binary16 (|Δ| ≤ 2^-11 relative ≈ 0.03° normal error), cutting the
 * size from 20 to 12 bytes per pixel.
 */
export function serializeDepthMap(d: DepthMapLike, opts: SerializeOptions = {}): ArrayBuffer {
  const n = d.width * d.height;
  check(d.depth.length >= n && d.normal.length >= 3 * n && d.cost.length >= n, 'DepthMap arrays shorter than width*height');
  const f16 = !!opts.float16;
  return packContainer(SerialKind.DepthMap, [
    { u32: d.width },
    { u32: d.height },
    { arr: d.depth.subarray(0, n), dtype: DType.F32 },
    floatField(d.normal.subarray(0, 3 * n), f16),
    floatField(d.cost.subarray(0, n), f16),
  ]);
}

export function deserializeDepthMap(input: ArrayBuffer | Uint8Array, opts: DecodeOptions = {}): DepthMapLike {
  const r = new Reader(input, !!opts.zeroCopy);
  r.expectKind(SerialKind.DepthMap, 'DepthMap');
  const width = r.u32();
  const height = r.u32();
  const n = width * height;
  return {
    width,
    height,
    depth: r.floats(n, 'depth'),
    normal: r.floats(3 * n, 'normal'),
    cost: r.floats(n, 'cost'),
  };
}

// ---------------------------------------------------------------------------
// JSON-safe encoding
// ---------------------------------------------------------------------------

type TypedArrayName =
  | 'Float64Array'
  | 'Float32Array'
  | 'Int8Array'
  | 'Uint8Array'
  | 'Uint8ClampedArray'
  | 'Int16Array'
  | 'Uint16Array'
  | 'Int32Array'
  | 'Uint32Array';

const TYPED_CTORS: Record<TypedArrayName, { from(a: ArrayLike<number>): ArrayBufferView }> = {
  Float64Array,
  Float32Array,
  Int8Array,
  Uint8Array,
  Uint8ClampedArray,
  Int16Array,
  Uint16Array,
  Int32Array,
  Uint32Array,
};

function encodeNumber(v: number): number | string {
  if (Object.is(v, -0)) return '-0'; // JSON.stringify(-0) === '0'
  return Number.isFinite(v) ? v : String(v); // 'NaN' | 'Infinity' | '-Infinity'
}

/** Objects that have no JSON-safe form here and would otherwise silently become `{}`. */
const UNSUPPORTED_TAGS = new Set([
  'ArrayBuffer', 'SharedArrayBuffer', 'DataView', 'Set', 'WeakSet', 'WeakMap', 'Date', 'RegExp', 'Blob', 'File', 'Promise',
]);

/**
 * Convert a value into a JSON-safe tree:
 *  - typed arrays → `{ "$typed": "Float64Array", "data": [ ... ] }` (exact:
 *    JSON numbers round-trip doubles; non-finite entries become strings),
 *  - non-finite numbers and −0 → `{ "$num": "NaN" | "Infinity" | "-Infinity" | "-0" }`
 *    (inside typed arrays the bare string),
 *  - Map → `{ "$map": [[key, value], ...] }`,
 *  - plain objects / arrays recursively; `undefined` object members are dropped.
 * Objects whose own keys include `$typed`, `$num` or `$map` are not supported;
 * Set, Date, RegExp, ArrayBuffer, DataView, Blob… throw a TypeError instead of
 * silently becoming `{}`.
 */
export function toJSONSafe(value: unknown): unknown {
  if (typeof value === 'number') {
    if (Object.is(value, -0)) return { $num: '-0' };
    return Number.isFinite(value) ? value : { $num: String(value) };
  }
  if (value === null || typeof value !== 'object') return value;
  const tag = Object.prototype.toString.call(value).slice(8, -1); // realm-safe
  if (UNSUPPORTED_TAGS.has(tag)) throw new TypeError(`toJSONSafe: ${tag} is not supported (it would be lost)`);
  if (ArrayBuffer.isView(value)) {
    const name = tag as TypedArrayName;
    if (!(name in TYPED_CTORS)) throw new TypeError(`toJSONSafe: unsupported typed array ${name}`);
    const src = value as unknown as ArrayLike<number>;
    const data = new Array<number | string>(src.length);
    for (let i = 0; i < src.length; i++) data[i] = encodeNumber(src[i]);
    return { $typed: name, data };
  }
  if (Array.isArray(value)) return value.map(toJSONSafe);
  if (value instanceof Map) return { $map: Array.from(value, ([k, v]) => [toJSONSafe(k), toJSONSafe(v)]) };
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    if (v === undefined || typeof v === 'function') continue;
    out[k] = toJSONSafe(v);
  }
  return out;
}

/** Inverse of `toJSONSafe`. */
export function fromJSONSafe<T = unknown>(value: unknown): T {
  return decodeSafe(value) as T;
}

function decodeSafe(value: unknown): unknown {
  if (value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(decodeSafe);
  const o = value as Record<string, unknown>;
  if (typeof o.$typed === 'string' && Array.isArray(o.data)) {
    const ctor = TYPED_CTORS[o.$typed as TypedArrayName];
    if (!ctor) throw new TypeError(`fromJSONSafe: unknown typed array ${o.$typed}`);
    const data = o.data as (number | string)[];
    const nums = new Array<number>(data.length);
    for (let i = 0; i < data.length; i++) nums[i] = typeof data[i] === 'string' ? Number(data[i]) : (data[i] as number);
    return ctor.from(nums);
  }
  if (typeof o.$num === 'string' && Object.keys(o).length === 1) return Number(o.$num);
  if (Array.isArray(o.$map) && Object.keys(o).length === 1) {
    return new Map((o.$map as [unknown, unknown][]).map(([k, v]) => [decodeSafe(k), decodeSafe(v)]));
  }
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(o)) out[k] = decodeSafe(v);
  return out;
}

/** `JSON.stringify(toJSONSafe(value))`. */
export function stringifyJSONSafe(value: unknown, space?: number): string {
  return JSON.stringify(toJSONSafe(value), null, space);
}

/** `fromJSONSafe(JSON.parse(text))`. */
export function parseJSONSafe<T = unknown>(text: string): T {
  return fromJSONSafe<T>(JSON.parse(text));
}

/** Plain-array form of a pose: `{ R: number[9] (row-major), t: number[3] }`. */
export interface PoseJSON {
  R: number[];
  t: number[];
}

export function poseToJSON(p: Pose): PoseJSON {
  return { R: Array.from(p.R), t: Array.from(p.t) };
}

export function poseFromJSON(j: PoseJSON): Pose {
  if (!j || !Array.isArray(j.R) || j.R.length !== 9 || !Array.isArray(j.t) || j.t.length !== 3) {
    throw new TypeError('poseFromJSON: expected { R: number[9], t: number[3] }');
  }
  return { R: Float64Array.from(j.R), t: Float64Array.from(j.t) };
}
