/**
 * Storage of raw capture recordings (format: ./recording.ts): the sinks the recorder worker
 * writes through, and the store the app lists / exports / deletes them with.
 *
 *  - OPFS (preferred; docs/research/ios-platform.md §2.4, §2.6): `sfm-recordings/<id>/` with
 *    `recording.json`, `frames.jsonl`, `live.jsonl`, `sensors.jsonl` and the JPEGs appended to
 *    `frames/pack-NNNN.bin` files of ≤ 64 MB (the rows hold {pack, offset}; a few large files
 *    instead of thousands keep the per-frame cost at one sync write and delete / list fast).
 *    Written through FileSystemSyncAccessHandles, which exist in dedicated workers only (one
 *    handle per file: a recording is written by one worker). The appended files stay open while
 *    recording and are flushed every few frames (RecordingWriter), so a killed app keeps what was
 *    written up to the last flush. Read on any thread through `getFile()` (disk-backed Files: an
 *    export references slices of the packs, nothing is copied).
 *  - IndexedDB fallback (no OPFS, e.g. Private Browsing): database `sfm-recordings` with
 *    `recordings` (id → header JSON), `images` ([id, seq] → JPEG Blob) and `lines`
 *    ([id, file, n] → one append: a JSON line, or a batch of sensors.jsonl lines).
 * Deleting truncates every OPFS file before `removeEntry` (removeEntry did not always free the
 * space before Safari 27, ios-platform.md §2.4) — through `createWritable` (Safari 26) where it
 * exists. `prepareRecordingExport` turns a stored recording into its ZIP parts (the stored JPEGs
 * are referenced, not copied).
 */
import { buildDatasetFiles, partFileName, planZipParts, type DatasetBuild, type ImageStoreRef, type RecordedImage, type RecordingHeader, type RecordingLineFile, type RecordingSink } from './recording';
import { createZipBlob, type ZipBlobEntryInput } from './zip';

const OPFS_DIR = 'sfm-recordings';
const IDB_NAME = 'sfm-recordings';
const PACK_BYTES = 64 << 20;
const enc = new TextEncoder();

/** FileSystemSyncAccessHandle (dedicated workers; not in the DOM lib). Before Safari 16.4 getSize/flush/close return promises. */
interface SyncHandle {
  write(buffer: Uint8Array, opts?: { at?: number }): number;
  truncate(size: number): void | Promise<void>;
  getSize(): number | Promise<number>;
  flush(): void | Promise<void>;
  close(): void | Promise<void>;
}
type SyncFileHandle = FileSystemFileHandle & { createSyncAccessHandle?: () => Promise<SyncHandle> };
type DirHandle = FileSystemDirectoryHandle & { entries?: () => AsyncIterableIterator<[string, FileSystemHandle]> };

async function opfsRoot(create: boolean): Promise<FileSystemDirectoryHandle | null> {
  const storage = (globalThis.navigator as Navigator | undefined)?.storage;
  if (!storage || typeof storage.getDirectory !== 'function') return null;
  try {
    const root = await storage.getDirectory();
    return await root.getDirectoryHandle(OPFS_DIR, { create });
  } catch {
    return null;
  }
}

/** OPFS with sync access handles is usable here (dedicated worker). */
export async function opfsSyncAvailable(): Promise<boolean> {
  const dir = await opfsRoot(true);
  if (!dir) return false;
  try {
    const fh = (await dir.getFileHandle('.probe', { create: true })) as SyncFileHandle;
    if (typeof fh.createSyncAccessHandle !== 'function') return false;
    const h = await fh.createSyncAccessHandle();
    await h.close();
    await dir.removeEntry('.probe').catch(() => undefined);
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------------------------
// OPFS sink (worker)
// ---------------------------------------------------------------------------------------------

interface OpenFile {
  h: SyncHandle;
  size: number;
}

export class OpfsRecordingSink implements RecordingSink {
  readonly kind = 'opfs';
  private pack: (OpenFile & { index: number }) | null = null;
  private readonly lines = new Map<string, OpenFile>();
  /** Packs in `frames/` (−1: not counted yet). */
  private packCount = -1;
  private closed = false;

  private constructor(
    private readonly dir: FileSystemDirectoryHandle,
    private readonly framesDir: FileSystemDirectoryHandle,
  ) {}

  static async open(id: string): Promise<OpfsRecordingSink> {
    const root = await opfsRoot(true);
    if (!root) throw new Error('OPFS is not available');
    const dir = await root.getDirectoryHandle(id, { create: true });
    const framesDir = await dir.getDirectoryHandle('frames', { create: true });
    return new OpfsRecordingSink(dir, framesDir);
  }

  private async openFile(dir: FileSystemDirectoryHandle, name: string): Promise<OpenFile> {
    const fh = (await dir.getFileHandle(name, { create: true })) as SyncFileHandle;
    if (typeof fh.createSyncAccessHandle !== 'function') throw new Error('FileSystemSyncAccessHandle is not available');
    const h = await fh.createSyncAccessHandle();
    return { h, size: await h.getSize() };
  }

  private write(f: OpenFile, bytes: Uint8Array): number {
    const at = f.size;
    const n = f.h.write(bytes, { at });
    if (n !== bytes.byteLength) throw new Error(`short write (${n} of ${bytes.byteLength} bytes)`);
    f.size += n;
    return at;
  }

  async writeHeader(json: string): Promise<void> {
    const f = await this.openFile(this.dir, 'recording.json');
    try {
      await f.h.truncate(0);
      f.size = 0;
      this.write(f, enc.encode(json));
      await f.h.flush();
    } finally {
      await f.h.close();
    }
  }

  async putImage(_seq: number, jpeg: Uint8Array): Promise<ImageStoreRef> {
    if (this.pack && this.pack.size > 0 && this.pack.size + jpeg.byteLength > PACK_BYTES) {
      const old = this.pack;
      this.pack = null;
      await old.h.flush();
      await old.h.close();
    }
    if (!this.pack) {
      const index = await this.nextPackIndex();
      this.pack = { ...(await this.openFile(this.framesDir, packName(index))), index };
    }
    const offset = this.write(this.pack, jpeg);
    return { pack: this.pack.index, offset };
  }

  private async nextPackIndex(): Promise<number> {
    // a re-opened recording appends after its packs (never rewrites one)
    if (this.packCount < 0) {
      this.packCount = 0;
      for (;;) {
        try {
          await this.framesDir.getFileHandle(packName(this.packCount));
          this.packCount++;
        } catch {
          break;
        }
      }
    }
    return this.packCount++;
  }

  async appendLine(file: RecordingLineFile, line: string): Promise<void> {
    let f = this.lines.get(file);
    if (!f) {
      f = await this.openFile(this.dir, file);
      this.lines.set(file, f);
    }
    this.write(f, enc.encode(`${line}\n`));
  }

  async flush(): Promise<void> {
    if (this.pack) await this.pack.h.flush();
    for (const f of this.lines.values()) await f.h.flush();
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    const all = [...this.lines.values(), ...(this.pack ? [this.pack] : [])];
    this.lines.clear();
    this.pack = null;
    for (const f of all) {
      try {
        await f.h.flush();
      } finally {
        await f.h.close();
      }
    }
  }
}

function packName(i: number): string {
  return `pack-${String(i).padStart(4, '0')}.bin`;
}

// ---------------------------------------------------------------------------------------------
// IndexedDB (fallback sink + its reader)
// ---------------------------------------------------------------------------------------------

function idbAvailable(): boolean {
  return typeof globalThis.indexedDB !== 'undefined' && typeof IDBKeyRange !== 'undefined';
}

function request<T>(r: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

function done(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onabort = () => reject(tx.error ?? new DOMException('transaction aborted', 'AbortError'));
    tx.onerror = () => reject(tx.error);
  });
}

function openIdb(name = IDB_NAME): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(name, 1);
    r.onupgradeneeded = () => {
      const db = r.result;
      if (!db.objectStoreNames.contains('recordings')) db.createObjectStore('recordings');
      if (!db.objectStoreNames.contains('images')) db.createObjectStore('images');
      if (!db.objectStoreNames.contains('lines')) db.createObjectStore('lines');
    };
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.onblocked = () => reject(new DOMException('database upgrade blocked', 'InvalidStateError'));
  });
}

/** Every record of recording `id` in a store keyed [id, …]. */
const prefixRange = (id: string): IDBKeyRange => IDBKeyRange.bound([id], [id, []]);

export class IdbRecordingSink implements RecordingSink {
  readonly kind = 'idb';
  private n = 0;

  private constructor(
    private readonly db: IDBDatabase,
    private readonly id: string,
  ) {}

  static async open(id: string, dbName?: string): Promise<IdbRecordingSink> {
    if (!idbAvailable()) throw new Error('IndexedDB is not available');
    const db = await openIdb(dbName);
    const sink = new IdbRecordingSink(db, id);
    // a re-opened recording appends after its lines
    const tx = db.transaction('lines', 'readonly');
    const keys = await request(tx.objectStore('lines').getAllKeys(prefixRange(id)));
    for (const k of keys) sink.n = Math.max(sink.n, Number((k as unknown[])[2]) + 1);
    return sink;
  }

  async writeHeader(json: string): Promise<void> {
    const tx = this.db.transaction('recordings', 'readwrite');
    tx.objectStore('recordings').put(json, this.id);
    await done(tx);
  }

  async putImage(seq: number, jpeg: Uint8Array): Promise<ImageStoreRef> {
    const key = String(seq);
    const tx = this.db.transaction('images', 'readwrite');
    tx.objectStore('images').put(new Blob([jpeg as Uint8Array<ArrayBuffer>], { type: 'image/jpeg' }), [this.id, key]);
    await done(tx);
    return { key };
  }

  async appendLine(file: RecordingLineFile, line: string): Promise<void> {
    const tx = this.db.transaction('lines', 'readwrite');
    tx.objectStore('lines').put(line, [this.id, file, this.n++]);
    await done(tx);
  }

  async flush(): Promise<void> {
    /* every write committed */
  }

  async close(): Promise<void> {
    this.db.close();
  }
}

// ---------------------------------------------------------------------------------------------
// Store (list / read / delete; any thread)
// ---------------------------------------------------------------------------------------------

export interface StoredRecording {
  id: string;
  sink: 'opfs' | 'idb';
  header: RecordingHeader;
  /** Bytes on disk (summary of a complete recording, else measured). */
  bytes: number;
}

/** Read access to one stored recording. */
export interface RecordingReader {
  readonly info: StoredRecording;
  /** frames.jsonl / live.jsonl / sensors.jsonl ('' when missing). */
  text(file: RecordingLineFile): Promise<string>;
  /** The JPEG of a row (a disk-backed Blob: an OPFS pack slice or the IndexedDB Blob). */
  image(img: RecordedImage): Promise<Blob>;
}

function parseHeader(text: string): RecordingHeader | null {
  try {
    const h = JSON.parse(text) as RecordingHeader;
    return h && typeof h === 'object' && typeof h.id === 'string' ? h : null;
  } catch {
    return null;
  }
}

async function fileOf(dir: FileSystemDirectoryHandle, name: string): Promise<File | null> {
  try {
    return await (await dir.getFileHandle(name)).getFile();
  } catch {
    return null;
  }
}

async function* dirEntries(dir: FileSystemDirectoryHandle): AsyncGenerator<[string, FileSystemHandle]> {
  const d = dir as DirHandle;
  if (typeof d.entries !== 'function') return;
  for await (const e of d.entries()) yield e;
}

/** Total bytes of the files under an OPFS directory. */
async function opfsBytes(dir: FileSystemDirectoryHandle): Promise<number> {
  let n = 0;
  for await (const [, h] of dirEntries(dir)) {
    if (h.kind === 'file') n += (await (h as FileSystemFileHandle).getFile()).size;
    else n += await opfsBytes(h as FileSystemDirectoryHandle);
  }
  return n;
}

export class RecordingStore {
  private constructor(
    private readonly opfs: FileSystemDirectoryHandle | null,
    private readonly idb: IDBDatabase | null,
  ) {}

  /** Open both back-ends that exist here (neither: an empty store). */
  static async open(opts: { idbName?: string } = {}): Promise<RecordingStore> {
    const opfs = await opfsRoot(false);
    let idb: IDBDatabase | null = null;
    if (idbAvailable()) idb = await openIdb(opts.idbName).catch(() => null);
    return new RecordingStore(opfs, idb);
  }

  close(): void {
    this.idb?.close();
  }

  /** Every stored recording, newest first. */
  async list(): Promise<StoredRecording[]> {
    const out: StoredRecording[] = [];
    if (this.opfs) {
      for await (const [id, h] of dirEntries(this.opfs)) {
        if (h.kind !== 'directory') continue;
        const dir = h as FileSystemDirectoryHandle;
        const f = await fileOf(dir, 'recording.json');
        const header = f ? parseHeader(await f.text()) : null;
        if (!header) continue;
        const bytes = header.state === 'complete' && header.summary ? header.summary.imageBytes + (await this.opfsTextBytes(dir)) : await opfsBytes(dir);
        out.push({ id, sink: 'opfs', header, bytes });
      }
    }
    if (this.idb) {
      const tx = this.idb.transaction(['recordings', 'images'], 'readonly');
      const keys = (await request(tx.objectStore('recordings').getAllKeys())) as string[];
      const vals = (await request(tx.objectStore('recordings').getAll())) as string[];
      for (let i = 0; i < keys.length; i++) {
        const header = parseHeader(vals[i]);
        if (!header) continue;
        let bytes = header.summary?.imageBytes ?? 0;
        if (header.state !== 'complete' || !header.summary) {
          const blobs = (await request(tx.objectStore('images').getAll(prefixRange(keys[i])))) as Blob[];
          bytes = blobs.reduce((s, b) => s + b.size, 0);
        }
        out.push({ id: keys[i], sink: 'idb', header, bytes });
      }
    }
    return out.sort((a, b) => b.header.createdAt - a.header.createdAt);
  }

  private async opfsTextBytes(dir: FileSystemDirectoryHandle): Promise<number> {
    let n = 0;
    for (const name of ['recording.json', 'frames.jsonl', 'live.jsonl', 'sensors.jsonl']) n += (await fileOf(dir, name))?.size ?? 0;
    return n;
  }

  async reader(info: StoredRecording): Promise<RecordingReader> {
    if (info.sink === 'opfs') {
      if (!this.opfs) throw new Error('OPFS is not available');
      const dir = await this.opfs.getDirectoryHandle(info.id);
      const frames = await dir.getDirectoryHandle('frames');
      const packs = new Map<number, File>();
      return {
        info,
        text: async (file) => (await (await fileOf(dir, file))?.text()) ?? '',
        image: async (img) => {
          const s = img.store;
          if (!s || !('pack' in s)) throw new Error(`${img.file}: no pack reference`);
          let f = packs.get(s.pack);
          if (!f) {
            const got = await fileOf(frames, packName(s.pack));
            if (!got) throw new Error(`${img.file}: ${packName(s.pack)} is missing`);
            packs.set(s.pack, (f = got));
          }
          if (s.offset + img.bytes > f.size) throw new Error(`${img.file}: truncated pack`);
          return f.slice(s.offset, s.offset + img.bytes, 'image/jpeg');
        },
      };
    }
    const db = this.idb;
    if (!db) throw new Error('IndexedDB is not available');
    return {
      info,
      text: async (file) => {
        const tx = db.transaction('lines', 'readonly');
        const lines = (await request(tx.objectStore('lines').getAll(IDBKeyRange.bound([info.id, file], [info.id, file, []])))) as string[];
        return lines.length ? `${lines.join('\n')}\n` : '';
      },
      image: async (img) => {
        const s = img.store;
        if (!s || !('key' in s)) throw new Error(`${img.file}: no image key`);
        const tx = db.transaction('images', 'readonly');
        const b = (await request(tx.objectStore('images').get([info.id, s.key]))) as Blob | undefined;
        if (!b) throw new Error(`${img.file} is missing`);
        return b;
      },
    };
  }

  async delete(info: Pick<StoredRecording, 'id' | 'sink'>): Promise<void> {
    if (info.sink === 'opfs') {
      if (!this.opfs) return;
      let dir: FileSystemDirectoryHandle;
      try {
        dir = await this.opfs.getDirectoryHandle(info.id);
      } catch {
        return;
      }
      await truncateAll(dir);
      await this.opfs.removeEntry(info.id, { recursive: true });
      return;
    }
    if (!this.idb) return;
    const tx = this.idb.transaction(['recordings', 'images', 'lines'], 'readwrite');
    tx.objectStore('recordings').delete(info.id);
    tx.objectStore('images').delete(prefixRange(info.id));
    tx.objectStore('lines').delete(prefixRange(info.id));
    await done(tx);
  }
}

// ---------------------------------------------------------------------------------------------
// Export (ZIP parts of the e2e dataset)
// ---------------------------------------------------------------------------------------------

export interface RecordingExportPart {
  part: number;
  count: number;
  fileName: string;
  /** Exact archive size. */
  bytes: number;
  entries: number;
  /** Build the archive (references the stored JPEGs, streams them once for the CRCs). */
  build(): Promise<Blob>;
}

export interface RecordingExport {
  meta: DatasetBuild['meta'];
  parts: RecordingExportPart[];
}

/**
 * The export of a stored recording as ZIP parts of ≤ `maxPartBytes` (src/io/recording.ts
 * buildDatasetFiles / planZipParts): part 1 holds the text files, every part extracts into the
 * same directory. The GPS fixes are left out unless `stripLocation: false` (privacy default).
 */
export async function prepareRecordingExport(store: RecordingStore, info: StoredRecording, opts: { stripLocation?: boolean; maxPartBytes?: number } = {}): Promise<RecordingExport> {
  const reader = await store.reader(info);
  const build = buildDatasetFiles(info.header, await reader.text('frames.jsonl'), await reader.text('live.jsonl'), {
    stripLocation: opts.stripLocation ?? true,
    sensorsText: await reader.text('sensors.jsonl'),
  });
  const texts = new Map(build.texts.map((t) => [t.name, t.text]));
  const images = new Map(build.images.map((i) => [i.name, i.image]));
  const plan = planZipParts(
    [...build.texts.map((t) => ({ name: t.name, size: enc.encode(t.text).length })), ...build.images.map((i) => ({ name: i.name, size: i.image.bytes }))],
    opts.maxPartBytes,
  );
  const base = `sfm-rec-${info.id.replace(/^rec-/, '')}`;
  return {
    meta: build.meta,
    parts: plan.map((p) => ({
      part: p.part,
      count: plan.length,
      fileName: partFileName(base, p.part, plan.length),
      bytes: p.bytes,
      entries: p.names.length,
      build: async () => {
        const entries: ZipBlobEntryInput[] = [];
        for (const name of p.names) {
          const text = texts.get(name);
          entries.push({ name, data: text ?? (await reader.image(images.get(name)!)), lastModified: info.header.createdAt });
        }
        return createZipBlob(entries);
      },
    })),
  };
}

/** Empty every file under `dir` (createWritable without keepExistingData, where it exists). */
async function truncateAll(dir: FileSystemDirectoryHandle): Promise<void> {
  for await (const [, h] of dirEntries(dir)) {
    if (h.kind === 'directory') {
      await truncateAll(h as FileSystemDirectoryHandle);
      continue;
    }
    const fh = h as FileSystemFileHandle & { createWritable?: (o?: { keepExistingData?: boolean }) => Promise<{ close(): Promise<void> }> };
    if (typeof fh.createWritable !== 'function') continue;
    try {
      const w = await fh.createWritable({ keepExistingData: false });
      await w.close();
    } catch {
      /* best effort: removeEntry follows */
    }
  }
}
