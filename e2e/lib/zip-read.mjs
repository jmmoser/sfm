// Minimal STORE-only ZIP reader for Node (the archives src/io/zip.ts writes: raw capture
// recording exports), with CRC-32 checks and entry-name validation (no absolute paths, no '..',
// no drive letters or backslashes), so an archive cannot write outside its target directory.
// Unit-tested in src/harness/engine/e2e-lib.test.ts.

let table = null;
function crcTable() {
  if (table) return table;
  table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[i] = c >>> 0;
  }
  return table;
}

export function crc32(bytes) {
  const t = crcTable();
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = t[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** A relative '/'-separated path that stays inside its directory (throws otherwise). */
export function safeEntryName(name) {
  if (typeof name !== 'string' || !name || name.includes('\\') || name.includes('\0') || name.startsWith('/') || /^[A-Za-z]:/.test(name)) throw new Error(`unsafe ZIP entry name ${JSON.stringify(name)}`);
  const parts = name.split('/');
  if (parts.some((p) => p === '..' || p === '.' || p === '')) throw new Error(`unsafe ZIP entry name ${JSON.stringify(name)}`);
  return name;
}

/** Entries of a STORE ZIP (directories skipped): [{ name, data: Uint8Array view }]. */
export function readStoreZip(buf) {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const n = bytes.length;
  let eocd = -1;
  for (let p = n - 22; p >= Math.max(0, n - 22 - 0xffff); p--) {
    if (dv.getUint32(p, true) === 0x06054b50 && p + 22 + dv.getUint16(p + 20, true) === n) {
      eocd = p;
      break;
    }
  }
  if (eocd < 0) throw new Error('not a ZIP archive (no end of central directory)');
  const count = dv.getUint16(eocd + 10, true);
  let p = dv.getUint32(eocd + 16, true);
  const dec = new TextDecoder('utf-8');
  const out = [];
  for (let i = 0; i < count; i++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error('corrupt central directory');
    const method = dv.getUint16(p + 10, true);
    const crc = dv.getUint32(p + 16, true);
    const size = dv.getUint32(p + 20, true);
    const nameLen = dv.getUint16(p + 28, true);
    const extraLen = dv.getUint16(p + 30, true);
    const commentLen = dv.getUint16(p + 32, true);
    const local = dv.getUint32(p + 42, true);
    const name = dec.decode(bytes.subarray(p + 46, p + 46 + nameLen));
    p += 46 + nameLen + extraLen + commentLen;
    if (name.endsWith('/')) continue;
    safeEntryName(name);
    if (method !== 0) throw new Error(`${name}: compression method ${method} (STORE only)`);
    if (dv.getUint32(local, true) !== 0x04034b50) throw new Error(`${name}: corrupt local header`);
    const start = local + 30 + dv.getUint16(local + 26, true) + dv.getUint16(local + 28, true);
    if (start + size > n) throw new Error(`${name}: truncated`);
    const data = bytes.subarray(start, start + size);
    if (crc32(data) !== crc) throw new Error(`${name}: CRC mismatch`);
    out.push({ name, data });
  }
  return out;
}
