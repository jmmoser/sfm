/**
 * Minimal ZIP writer/reader (STORE method only — the payloads we export, JPEG
 * frames and binary PLY/GLB, do not compress meaningfully, and STORE keeps the
 * writer allocation-free beyond the output buffer).
 *
 *  - CRC-32 (IEEE 802.3, reflected 0xEDB88320) with a slicing-by-8 table.
 *  - General-purpose flag bit 11 (UTF-8 names) is always set.
 *  - No ZIP64: the archive must stay below 4 GiB and 65 535 entries (guarded).
 *  - `createZip` builds a single Uint8Array; `createZipBlob` builds a Blob from
 *    parts so large frame Blobs (from IndexedDB) are never copied into one
 *    contiguous buffer — peak memory is one streamed chunk, which matters on iOS.
 *  - `readZip` is a small STORE-only reader used by tests and imports.
 */

export interface ZipEntryInput {
  /** Path inside the archive, '/' separated (backslashes are converted). */
  name: string;
  data: Uint8Array | ArrayBuffer | string;
  /** Modification time (Date or epoch ms). Defaults to the time of the call. */
  lastModified?: Date | number;
}

export interface ZipBlobEntryInput {
  name: string;
  data: Blob | Uint8Array | ArrayBuffer | string;
  lastModified?: Date | number;
}

export interface ZipEntry {
  name: string;
  /** View into the archive bytes (zero-copy). */
  data: Uint8Array;
  crc32: number;
  lastModified: Date;
}

export interface ZipOptions {
  /** Archive comment (UTF-8, ≤ 65 535 bytes). */
  comment?: string;
}

export class ZipError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ZipError';
  }
}

/** Largest archive we allow without ZIP64 (all offsets/sizes are u32). */
export const ZIP_MAX_SIZE = 0xffffffff;
export const ZIP_MAX_ENTRIES = 0xffff;

const SIG_LOCAL = 0x04034b50;
const SIG_CENTRAL = 0x02014b50;
const SIG_EOCD = 0x06054b50;
const SIG_ZIP64_LOCATOR = 0x07064b50;
const LOCAL_HEADER_SIZE = 30;
const CENTRAL_HEADER_SIZE = 46;
const EOCD_SIZE = 22;
const FLAG_UTF8 = 0x0800;
const VERSION = 20; // 2.0

// ---------------------------------------------------------------------------
// CRC-32 (slicing-by-8)
// ---------------------------------------------------------------------------

let crcTable: Uint32Array | null = null;

function getCrcTable(): Uint32Array {
  if (crcTable) return crcTable;
  const t = new Uint32Array(8 * 256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[i] = c >>> 0;
  }
  for (let i = 0; i < 256; i++) {
    let c = t[i];
    for (let s = 1; s < 8; s++) {
      c = t[c & 0xff] ^ (c >>> 8);
      t[s * 256 + i] = c;
    }
  }
  crcTable = t;
  return t;
}

/**
 * CRC-32 of `data`. Pass the previous result as `crc` to continue a running
 * checksum over several chunks (`crc32(b, crc32(a)) === crc32(a ++ b)`).
 */
export function crc32(data: Uint8Array, crc = 0): number {
  const t = getCrcTable();
  let c = (crc ^ 0xffffffff) >>> 0;
  const n = data.length;
  let i = 0;
  const n8 = n - (n & 7);
  while (i < n8) {
    const a = c ^ (data[i] | (data[i + 1] << 8) | (data[i + 2] << 16) | (data[i + 3] << 24));
    const b = data[i + 4] | (data[i + 5] << 8) | (data[i + 6] << 16) | (data[i + 7] << 24);
    c =
      t[1792 + (a & 0xff)] ^
      t[1536 + ((a >>> 8) & 0xff)] ^
      t[1280 + ((a >>> 16) & 0xff)] ^
      t[1024 + (a >>> 24)] ^
      t[768 + (b & 0xff)] ^
      t[512 + ((b >>> 8) & 0xff)] ^
      t[256 + ((b >>> 16) & 0xff)] ^
      t[b >>> 24];
    i += 8;
  }
  while (i < n) c = t[(c ^ data[i++]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const encoder = new TextEncoder();

/** Normalise and validate an archive path. */
export function normalizeZipPath(name: string): string {
  let p = name.replace(/\\/g, '/');
  while (p.startsWith('./')) p = p.slice(2);
  if (p.length === 0) throw new ZipError('Empty file name');
  if (p.startsWith('/') || /^[A-Za-z]:/.test(p)) throw new ZipError(`Absolute path not allowed: ${name}`);
  // '.' segments are dropped so 'a/./b' and 'a/b' collide in the duplicate check
  // (extractors resolve both to the same file).
  const parts = p.split('/').filter((s) => s !== '.');
  for (let i = 0; i < parts.length; i++) {
    const s = parts[i];
    if (s === '..') throw new ZipError(`Path traversal not allowed: ${name}`);
    if (s === '' && i !== parts.length - 1) throw new ZipError(`Empty path segment: ${name}`);
  }
  p = parts.join('/');
  if (p.length === 0) throw new ZipError(`Empty file name: ${name}`);
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001f]/.test(p)) throw new ZipError(`Control character in file name: ${name}`);
  return p;
}

function toBytes(data: Uint8Array | ArrayBuffer | string): Uint8Array {
  if (typeof data === 'string') return encoder.encode(data);
  if (data instanceof Uint8Array) return data;
  // Any other view (Float32Array, DataView, …) is stored as its raw bytes —
  // `new Uint8Array(view)` would convert element-wise and corrupt the payload.
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return new Uint8Array(data);
}

/** MS-DOS date/time (local time, 2-second resolution, clamped to 1980..2107). */
function dosDateTime(d: Date | number | undefined, fallback: Date): { time: number; date: number } {
  const date = d === undefined ? fallback : d instanceof Date ? d : new Date(d);
  let year = date.getFullYear();
  let month = date.getMonth() + 1;
  let day = date.getDate();
  let hh = date.getHours();
  let mm = date.getMinutes();
  let ss = date.getSeconds();
  if (!Number.isFinite(year) || year < 1980) {
    year = 1980; month = 1; day = 1; hh = 0; mm = 0; ss = 0;
  } else if (year > 2107) {
    year = 2107; month = 12; day = 31; hh = 23; mm = 59; ss = 58;
  }
  return {
    time: ((hh & 0x1f) << 11) | ((mm & 0x3f) << 5) | ((ss >> 1) & 0x1f),
    date: (((year - 1980) & 0x7f) << 9) | ((month & 0x0f) << 5) | (day & 0x1f),
  };
}

function dosToDate(time: number, date: number): Date {
  const year = ((date >> 9) & 0x7f) + 1980;
  const month = ((date >> 5) & 0x0f) - 1;
  const day = date & 0x1f;
  return new Date(year, month, day, (time >> 11) & 0x1f, (time >> 5) & 0x3f, (time & 0x1f) * 2);
}

interface EntryMeta {
  name: Uint8Array;
  crc: number;
  size: number;
  time: number;
  date: number;
  offset: number;
}

function writeLocalHeader(dv: DataView, o: number, e: EntryMeta): number {
  dv.setUint32(o, SIG_LOCAL, true);
  dv.setUint16(o + 4, VERSION, true);
  dv.setUint16(o + 6, FLAG_UTF8, true);
  dv.setUint16(o + 8, 0, true); // method: STORE
  dv.setUint16(o + 10, e.time, true);
  dv.setUint16(o + 12, e.date, true);
  dv.setUint32(o + 14, e.crc, true);
  dv.setUint32(o + 18, e.size, true); // compressed
  dv.setUint32(o + 22, e.size, true); // uncompressed
  dv.setUint16(o + 26, e.name.length, true);
  dv.setUint16(o + 28, 0, true); // extra length
  new Uint8Array(dv.buffer, dv.byteOffset + o + LOCAL_HEADER_SIZE, e.name.length).set(e.name);
  return o + LOCAL_HEADER_SIZE + e.name.length;
}

function writeCentralHeader(dv: DataView, o: number, e: EntryMeta): number {
  dv.setUint32(o, SIG_CENTRAL, true);
  dv.setUint16(o + 4, VERSION, true); // made by: MS-DOS / v2.0
  dv.setUint16(o + 6, VERSION, true); // needed
  dv.setUint16(o + 8, FLAG_UTF8, true);
  dv.setUint16(o + 10, 0, true);
  dv.setUint16(o + 12, e.time, true);
  dv.setUint16(o + 14, e.date, true);
  dv.setUint32(o + 16, e.crc, true);
  dv.setUint32(o + 20, e.size, true);
  dv.setUint32(o + 24, e.size, true);
  dv.setUint16(o + 28, e.name.length, true);
  dv.setUint16(o + 30, 0, true); // extra
  dv.setUint16(o + 32, 0, true); // comment
  dv.setUint16(o + 34, 0, true); // disk number start
  dv.setUint16(o + 36, 0, true); // internal attrs
  dv.setUint32(o + 38, 0, true); // external attrs
  dv.setUint32(o + 42, e.offset, true);
  new Uint8Array(dv.buffer, dv.byteOffset + o + CENTRAL_HEADER_SIZE, e.name.length).set(e.name);
  return o + CENTRAL_HEADER_SIZE + e.name.length;
}

function writeEocd(dv: DataView, o: number, count: number, cdSize: number, cdOffset: number, comment: Uint8Array): number {
  dv.setUint32(o, SIG_EOCD, true);
  dv.setUint16(o + 4, 0, true);
  dv.setUint16(o + 6, 0, true);
  dv.setUint16(o + 8, count, true);
  dv.setUint16(o + 10, count, true);
  dv.setUint32(o + 12, cdSize, true);
  dv.setUint32(o + 16, cdOffset, true);
  dv.setUint16(o + 20, comment.length, true);
  new Uint8Array(dv.buffer, dv.byteOffset + o + EOCD_SIZE, comment.length).set(comment);
  return o + EOCD_SIZE + comment.length;
}

function prepareNames(names: string[]): Uint8Array[] {
  if (names.length > ZIP_MAX_ENTRIES) throw new ZipError(`Too many entries (${names.length} > ${ZIP_MAX_ENTRIES}); ZIP64 is not supported`);
  const seen = new Set<string>();
  return names.map((n) => {
    const p = normalizeZipPath(n);
    if (seen.has(p)) throw new ZipError(`Duplicate file name: ${p}`);
    seen.add(p);
    const b = encoder.encode(p);
    if (b.length > 0xffff) throw new ZipError(`File name too long: ${p.slice(0, 64)}…`);
    return b;
  });
}

function prepareComment(comment: string | undefined): Uint8Array {
  const c = comment ? encoder.encode(comment) : new Uint8Array(0);
  if (c.length > 0xffff) throw new ZipError('Archive comment too long');
  // A comment containing the EOCD signature makes the archive ambiguous: readers
  // (Python's zipfile, …) locate the EOCD by scanning backwards for it.
  for (let i = 0; i + 3 < c.length; i++) {
    if (c[i] === 0x50 && c[i + 1] === 0x4b && c[i + 2] === 0x05 && c[i + 3] === 0x06) {
      throw new ZipError('Archive comment must not contain the end-of-central-directory signature');
    }
  }
  return c;
}

function checkTotal(total: number): void {
  if (total > ZIP_MAX_SIZE) {
    throw new ZipError(`Archive would be ${(total / 2 ** 30).toFixed(2)} GiB; ZIP64 (> 4 GiB) is not supported`);
  }
}

// ---------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------

/** Build a STORE-only ZIP archive in memory. */
export function createZip(files: ZipEntryInput[], opts: ZipOptions = {}): Uint8Array {
  const now = new Date();
  const names = prepareNames(files.map((f) => f.name));
  const comment = prepareComment(opts.comment);
  const datas = files.map((f) => toBytes(f.data));

  let total = EOCD_SIZE + comment.length;
  for (let i = 0; i < files.length; i++) {
    total += LOCAL_HEADER_SIZE + CENTRAL_HEADER_SIZE + 2 * names[i].length + datas[i].length;
  }
  checkTotal(total);

  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  const metas: EntryMeta[] = [];
  let o = 0;
  for (let i = 0; i < files.length; i++) {
    const { time, date } = dosDateTime(files[i].lastModified, now);
    const e: EntryMeta = { name: names[i], crc: crc32(datas[i]), size: datas[i].length, time, date, offset: o };
    metas.push(e);
    o = writeLocalHeader(dv, o, e);
    out.set(datas[i], o);
    o += datas[i].length;
  }
  const cdOffset = o;
  for (const e of metas) o = writeCentralHeader(dv, o, e);
  o = writeEocd(dv, o, metas.length, o - cdOffset, cdOffset, comment);
  if (o !== total) throw new ZipError('internal: size mismatch');
  return out;
}

async function crcOfBlob(blob: Blob): Promise<number> {
  // Stream the blob so a large frame is never fully resident just for the CRC.
  if (typeof blob.stream === 'function') {
    const reader = blob.stream().getReader();
    let crc = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) crc = crc32(value, crc);
    }
    return crc;
  }
  return crc32(new Uint8Array(await blob.arrayBuffer()));
}

/**
 * Build a STORE-only ZIP as a Blob. Blob inputs are referenced, not copied
 * (they are only streamed once to compute their CRC), so archives of hundreds
 * of JPEG frames stay cheap in memory. The result has type `application/zip`.
 */
export async function createZipBlob(files: ZipBlobEntryInput[], opts: ZipOptions = {}): Promise<Blob> {
  const now = new Date();
  const names = prepareNames(files.map((f) => f.name));
  const comment = prepareComment(opts.comment);

  let total = EOCD_SIZE + comment.length;
  for (let i = 0; i < files.length; i++) {
    const d = files[i].data;
    const size = typeof d === 'string' ? -1 : d instanceof Blob ? d.size : d.byteLength;
    if (size >= 0) total += size;
    total += LOCAL_HEADER_SIZE + CENTRAL_HEADER_SIZE + 2 * names[i].length;
  }
  // Strings are measured after encoding below; check what we already know.
  checkTotal(total);

  const parts: BlobPart[] = [];
  const metas: EntryMeta[] = [];
  let o = 0;
  for (let i = 0; i < files.length; i++) {
    const src = files[i].data;
    let part: Blob | Uint8Array;
    let crc: number;
    let size: number;
    if (src instanceof Blob) {
      part = src;
      size = src.size;
      crc = await crcOfBlob(src);
    } else {
      const bytes = toBytes(src);
      if (typeof src === 'string') {
        total += bytes.length;
        checkTotal(total);
      }
      part = bytes;
      size = bytes.length;
      crc = crc32(bytes);
    }
    const { time, date } = dosDateTime(files[i].lastModified, now);
    const e: EntryMeta = { name: names[i], crc, size, time, date, offset: o };
    metas.push(e);
    const header = new Uint8Array(LOCAL_HEADER_SIZE + e.name.length);
    writeLocalHeader(new DataView(header.buffer), 0, e);
    parts.push(header, part as BlobPart);
    o += header.length + size;
  }
  let cdSize = 0;
  for (const e of metas) cdSize += CENTRAL_HEADER_SIZE + e.name.length;
  const tail = new Uint8Array(cdSize + EOCD_SIZE + comment.length);
  const dv = new DataView(tail.buffer);
  let t = 0;
  for (const e of metas) t = writeCentralHeader(dv, t, e);
  writeEocd(dv, t, metas.length, cdSize, o, comment);
  parts.push(tail);
  checkTotal(o + tail.length);
  return new Blob(parts, { type: 'application/zip' });
}

// ---------------------------------------------------------------------------
// Reader (STORE only)
// ---------------------------------------------------------------------------

/**
 * Parse a ZIP archive (STORE entries only; directories are skipped). Verifies
 * CRCs. Entry data are zero-copy views into `input`.
 */
export function readZip(input: Uint8Array | ArrayBuffer): ZipEntry[] {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const n = bytes.length;
  if (n < EOCD_SIZE) throw new ZipError('Not a ZIP archive (too short)');
  // Scan backwards for the EOCD. Prefer a candidate whose comment length ends
  // exactly at the end of the input (a signature inside the comment is not
  // the record); fall back to the last plausible one (trailing junk).
  let eocd = -1;
  let loose = -1;
  const minPos = Math.max(0, n - EOCD_SIZE - 0xffff);
  for (let p = n - EOCD_SIZE; p >= minPos; p--) {
    if (dv.getUint32(p, true) !== SIG_EOCD) continue;
    const end = p + EOCD_SIZE + dv.getUint16(p + 20, true);
    if (end === n) {
      eocd = p;
      break;
    }
    if (loose < 0 && end <= n) loose = p;
  }
  if (eocd < 0) eocd = loose;
  if (eocd < 0) throw new ZipError('Not a ZIP archive (no end of central directory)');
  const count = dv.getUint16(eocd + 10, true);
  const cdSize = dv.getUint32(eocd + 12, true);
  const cdOffset = dv.getUint32(eocd + 16, true);
  // 0xFFFF / 0xFFFFFFFF are only ZIP64 markers when a ZIP64 locator precedes the
  // EOCD; a plain archive may legitimately hold exactly 65 535 entries.
  const hasZip64Locator = eocd >= 20 && dv.getUint32(eocd - 20, true) === SIG_ZIP64_LOCATOR;
  if (hasZip64Locator || cdOffset === 0xffffffff || cdSize === 0xffffffff) throw new ZipError('ZIP64 archives are not supported');
  if (cdOffset + cdSize > n) throw new ZipError('Corrupt central directory');

  const utf8 = new TextDecoder('utf-8');
  const latin1 = new TextDecoder('latin1');
  const out: ZipEntry[] = [];
  let p = cdOffset;
  for (let i = 0; i < count; i++) {
    if (p + CENTRAL_HEADER_SIZE > n || dv.getUint32(p, true) !== SIG_CENTRAL) throw new ZipError('Corrupt central directory entry');
    const flags = dv.getUint16(p + 8, true);
    const method = dv.getUint16(p + 10, true);
    const time = dv.getUint16(p + 12, true);
    const date = dv.getUint16(p + 14, true);
    const crc = dv.getUint32(p + 16, true);
    const csize = dv.getUint32(p + 20, true);
    const usize = dv.getUint32(p + 24, true);
    const nameLen = dv.getUint16(p + 28, true);
    const extraLen = dv.getUint16(p + 30, true);
    const commentLen = dv.getUint16(p + 32, true);
    const localOffset = dv.getUint32(p + 42, true);
    const nameBytes = bytes.subarray(p + CENTRAL_HEADER_SIZE, p + CENTRAL_HEADER_SIZE + nameLen);
    const name = (flags & FLAG_UTF8 ? utf8 : latin1).decode(nameBytes);
    p += CENTRAL_HEADER_SIZE + nameLen + extraLen + commentLen;
    if (name.endsWith('/')) continue; // directory
    if (flags & 1) throw new ZipError(`Encrypted entry not supported: ${name}`);
    if (method !== 0) throw new ZipError(`Unsupported compression method ${method} for ${name} (STORE only)`);
    if (csize !== usize) throw new ZipError(`Size mismatch for stored entry ${name}`);
    if (localOffset + LOCAL_HEADER_SIZE > n || dv.getUint32(localOffset, true) !== SIG_LOCAL) {
      throw new ZipError(`Corrupt local header for ${name}`);
    }
    const lNameLen = dv.getUint16(localOffset + 26, true);
    const lExtraLen = dv.getUint16(localOffset + 28, true);
    const start = localOffset + LOCAL_HEADER_SIZE + lNameLen + lExtraLen;
    if (start + csize > n) throw new ZipError(`Truncated data for ${name}`);
    const data = bytes.subarray(start, start + csize);
    if (crc32(data) !== crc) throw new ZipError(`CRC mismatch for ${name}`);
    out.push({ name, data, crc32: crc, lastModified: dosToDate(time, date) });
  }
  return out;
}
