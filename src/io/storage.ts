/**
 * IndexedDB persistence for capture sessions. Works in Window and Worker
 * scopes (uses `globalThis.indexedDB`), no dependencies.
 *
 * Schema (DB version 1):
 *   sessions   keyPath 'id'                 SessionMeta
 *   frames     key [sessionId, frameId]     frame info (structured clone, e.g. FrameInfo)
 *   frameData  key [sessionId, frameId]     JPEG Blob
 *   blobs      key [sessionId, key]         Blob | ArrayBuffer
 *   json       key [sessionId, key]         any structured-clonable value
 * All per-session records share the compound-key prefix, so listing and the
 * cascade delete use a single key range `[sid] .. [sid, []]` per store.
 *
 * Every method resolves only when its transaction has COMMITTED (so a resolved
 * write is durable as far as the browser guarantees) and rejects with
 * `StorageQuotaError` when the origin's quota is exhausted (QuotaExceededError
 * from the request, the transaction abort, or the open call).
 *
 * Timestamps in SessionMeta are wall-clock epoch ms (`Date.now()`), not the
 * `performance.now()` domain, because they persist across page loads.
 */

export type SessionMode = 'object' | 'scene' | 'auto';
export type SessionStatus = 'capturing' | 'captured' | 'sparse' | 'dense' | 'meshed';

export interface SessionMeta {
  id: string;
  name: string;
  /** Epoch ms. */
  createdAt: number;
  /** Epoch ms; bumped by every updateSession / putFrame / deleteFrame. */
  updatedAt: number;
  mode: SessionMode;
  status: SessionStatus;
  /** Number of stored frames (maintained by putFrame / deleteFrame). */
  numFrames: number;
  stats?: Record<string, number | string>;
  thumbnail?: Blob;
}

export interface CreateSessionInit {
  id?: string;
  name?: string;
  mode?: SessionMode;
  status?: SessionStatus;
  stats?: Record<string, number | string>;
  thumbnail?: Blob;
}

export type SessionPatch = Partial<Omit<SessionMeta, 'id' | 'createdAt' | 'numFrames'>>;

export interface StorageEstimateInfo {
  /** Bytes used by this origin. */
  usage: number;
  /** Bytes available to this origin in total. */
  quota: number;
  /** quota − usage (≥ 0). */
  available: number;
  /** Whether storage is persistent (exempt from eviction); null if unknown. */
  persisted: boolean | null;
}

/** Thrown when a write fails because the origin is out of storage quota. */
export class StorageQuotaError extends Error {
  override readonly cause: unknown;
  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = 'StorageQuotaError';
    this.cause = cause;
  }
}

/** Thrown when an operation references a session that does not exist. */
export class SessionNotFoundError extends Error {
  constructor(readonly sessionId: string) {
    super(`Session not found: ${sessionId}`);
    this.name = 'SessionNotFoundError';
  }
}

const DB_VERSION = 1;
const S_SESSIONS = 'sessions';
const S_FRAMES = 'frames';
const S_FRAME_DATA = 'frameData';
const S_BLOBS = 'blobs';
const S_JSON = 'json';
const ALL_STORES = [S_SESSIONS, S_FRAMES, S_FRAME_DATA, S_BLOBS, S_JSON];

/** Fallback record used when the engine cannot store Blobs (old WebKit bugs). */
interface BufferedBlob {
  $blobBuffer: ArrayBuffer;
  type: string;
}

function isBufferedBlob(v: unknown): v is BufferedBlob {
  return !!v && typeof v === 'object' && (v as BufferedBlob).$blobBuffer instanceof ArrayBuffer;
}

function isQuotaError(e: unknown): boolean {
  if (!e || typeof e !== 'object') return false;
  const err = e as { name?: string; code?: number; message?: string };
  return err.name === 'QuotaExceededError' || err.code === 22 || (err.name === 'UnknownError' && /quota/i.test(err.message ?? ''));
}

function mapError(e: unknown, what: string): unknown {
  if (e instanceof StorageQuotaError || e instanceof SessionNotFoundError) return e;
  if (isQuotaError(e)) return new StorageQuotaError(`Storage quota exceeded while ${what}`, e);
  return e ?? new Error(`IndexedDB error while ${what}`);
}

function sessionRange(sessionId: string): IDBKeyRange {
  // Arrays sort after every other key type, so [sid, []] bounds all [sid, number|string].
  return IDBKeyRange.bound([sessionId], [sessionId, []]);
}

function newId(): string {
  const c = (globalThis as { crypto?: Crypto }).crypto;
  if (c && typeof c.randomUUID === 'function') return c.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}${Math.random().toString(36).slice(2, 10)}`;
}

function checkFrameId(frameId: number): void {
  if (typeof frameId !== 'number' || !Number.isFinite(frameId)) throw new TypeError(`Invalid frameId ${frameId}`);
}

function checkKey(key: string): void {
  if (typeof key !== 'string' || key.length === 0) throw new TypeError('Key must be a non-empty string');
}

type TxBody<T> = (tx: IDBTransaction, setResult: (v: T) => void, fail: (e: unknown) => void) => void;

export interface SessionStoreOpenOptions {
  /**
   * Store Blobs as ArrayBuffers (workaround for engines that cannot persist
   * Blobs in IndexedDB). Normally auto-detected after the first failure.
   */
  blobsAsArrayBuffers?: boolean;
}

export class SessionStore {
  static readonly DB_VERSION = DB_VERSION;

  private blobsAsBuffers: boolean;
  private closed = false;

  private constructor(
    readonly db: IDBDatabase,
    readonly dbName: string,
    opts: SessionStoreOpenOptions,
  ) {
    this.blobsAsBuffers = !!opts.blobsAsArrayBuffers;
  }

  /** True when IndexedDB is available in this realm. */
  static isSupported(): boolean {
    return typeof globalThis.indexedDB !== 'undefined' && typeof IDBKeyRange !== 'undefined';
  }

  /** Open (and create / upgrade) the database. */
  static open(dbName = 'sfm', opts: SessionStoreOpenOptions = {}): Promise<SessionStore> {
    return new Promise((resolve, reject) => {
      if (!SessionStore.isSupported()) {
        reject(new Error('IndexedDB is not available in this context'));
        return;
      }
      let req: IDBOpenDBRequest;
      try {
        req = globalThis.indexedDB.open(dbName, DB_VERSION);
      } catch (e) {
        reject(mapError(e, 'opening the database'));
        return;
      }
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(S_SESSIONS)) db.createObjectStore(S_SESSIONS, { keyPath: 'id' });
        for (const s of [S_FRAMES, S_FRAME_DATA, S_BLOBS, S_JSON]) {
          if (!db.objectStoreNames.contains(s)) db.createObjectStore(s);
        }
      };
      req.onsuccess = () => {
        const db = req.result;
        // Let a newer version (another tab) upgrade instead of blocking it.
        db.onversionchange = () => db.close();
        resolve(new SessionStore(db, dbName, opts));
      };
      req.onerror = () => reject(mapError(req.error, 'opening the database'));
      // onblocked: an older connection is open elsewhere; it closes itself on
      // versionchange (see above), after which onsuccess fires.
    });
  }

  /** Delete a whole database (tests / "reset app"). */
  static deleteDatabase(dbName = 'sfm'): Promise<void> {
    return new Promise((resolve, reject) => {
      const req = globalThis.indexedDB.deleteDatabase(dbName);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
      req.onblocked = () => {
        /* resolves once other connections close */
      };
    });
  }

  close(): void {
    if (!this.closed) {
      this.closed = true;
      this.db.close();
    }
  }

  // -------------------------------------------------------------------------
  // Transaction helper
  // -------------------------------------------------------------------------

  private tx<T>(
    stores: string[],
    mode: IDBTransactionMode,
    what: string,
    body: TxBody<T>,
    durability?: 'default' | 'relaxed' | 'strict',
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      let result: T | undefined;
      let failure: unknown;
      let failed = false;
      let tx: IDBTransaction;
      try {
        tx = durability ? this.db.transaction(stores, mode, { durability }) : this.db.transaction(stores, mode);
      } catch (e) {
        reject(mapError(e, what));
        return;
      }
      const fail = (e: unknown): void => {
        if (!failed) {
          failed = true;
          failure = mapError(e, what);
        }
        try {
          tx.abort();
        } catch {
          /* already finished */
        }
      };
      tx.oncomplete = () => (failed ? reject(failure) : resolve(result as T));
      tx.onabort = () => reject(failed ? failure : mapError(tx.error ?? new DOMException('Transaction aborted', 'AbortError'), what));
      tx.onerror = (ev) => {
        // A failed request aborts the transaction; remember the first cause.
        const target = ev.target as IDBRequest | null;
        if (!failed && target && 'error' in target && target.error) {
          failed = true;
          failure = mapError(target.error, what);
        }
      };
      try {
        body(tx, (v) => (result = v), fail);
      } catch (e) {
        fail(e);
      }
    });
  }

  /** Wrap a request success callback so exceptions abort the transaction with that error. */
  private static on<R>(req: IDBRequest<R>, fail: (e: unknown) => void, cb: (r: R) => void): void {
    req.onsuccess = () => {
      try {
        cb(req.result);
      } catch (e) {
        fail(e);
      }
    };
  }

  private async blobValue(data: Blob): Promise<Blob | BufferedBlob> {
    if (!this.blobsAsBuffers) return data;
    return { $blobBuffer: await data.arrayBuffer(), type: data.type };
  }

  /** Retry a Blob write as ArrayBuffer once if the engine refuses Blobs. */
  private async withBlobFallback<T>(hasBlob: boolean, op: () => Promise<T>): Promise<T> {
    try {
      return await op();
    } catch (e) {
      const name = (e as { name?: string } | null)?.name;
      if (!hasBlob || this.blobsAsBuffers || e instanceof StorageQuotaError || e instanceof SessionNotFoundError) throw e;
      if (name === 'DataCloneError' || name === 'UnknownError' || name === 'NotSupportedError' || name === 'DataError') {
        this.blobsAsBuffers = true;
        return op();
      }
      throw e;
    }
  }

  // -------------------------------------------------------------------------
  // Sessions
  // -------------------------------------------------------------------------

  async createSession(init: CreateSessionInit = {}): Promise<SessionMeta> {
    const now = Date.now();
    const meta: SessionMeta = {
      id: init.id ?? newId(),
      name: init.name ?? new Date(now).toISOString().slice(0, 16).replace('T', ' '),
      createdAt: now,
      updatedAt: now,
      mode: init.mode ?? 'auto',
      status: init.status ?? 'capturing',
      numFrames: 0,
    };
    if (init.stats) meta.stats = init.stats;
    if (init.thumbnail) meta.thumbnail = init.thumbnail;
    await this.tx<void>([S_SESSIONS], 'readwrite', 'creating a session', (tx, _set, fail) => {
      // add() fails with ConstraintError if the id already exists.
      const r = tx.objectStore(S_SESSIONS).add(meta);
      r.onerror = () => fail(r.error);
    });
    return meta;
  }

  /** Shallow-merge `patch` into the session (updatedAt = now unless given). */
  updateSession(id: string, patch: SessionPatch): Promise<SessionMeta> {
    return this.tx<SessionMeta>([S_SESSIONS], 'readwrite', 'updating a session', (tx, set, fail) => {
      const store = tx.objectStore(S_SESSIONS);
      SessionStore.on(store.get(id), fail, (cur: SessionMeta | undefined) => {
        if (!cur) throw new SessionNotFoundError(id);
        const next: SessionMeta = { ...cur, ...patch, id: cur.id, createdAt: cur.createdAt, numFrames: cur.numFrames };
        if (patch.updatedAt === undefined) next.updatedAt = Math.max(Date.now(), cur.updatedAt);
        store.put(next);
        set(next);
      });
    });
  }

  getSession(id: string): Promise<SessionMeta | null> {
    return this.tx<SessionMeta | null>([S_SESSIONS], 'readonly', 'reading a session', (tx, set, fail) => {
      SessionStore.on(tx.objectStore(S_SESSIONS).get(id), fail, (v: SessionMeta | undefined) => set(v ?? null));
    });
  }

  /** All sessions, newest first (by createdAt, or updatedAt when requested). */
  listSessions(opts: { orderBy?: 'createdAt' | 'updatedAt' } = {}): Promise<SessionMeta[]> {
    const key = opts.orderBy ?? 'createdAt';
    return this.tx<SessionMeta[]>([S_SESSIONS], 'readonly', 'listing sessions', (tx, set, fail) => {
      SessionStore.on(tx.objectStore(S_SESSIONS).getAll(), fail, (all: SessionMeta[]) => {
        all.sort((a, b) => b[key] - a[key] || (a.id < b.id ? 1 : a.id > b.id ? -1 : 0));
        set(all);
      });
    });
  }

  /** Delete a session and every frame, blob and JSON record under it (one transaction). */
  deleteSession(id: string): Promise<void> {
    return this.tx<void>(ALL_STORES, 'readwrite', 'deleting a session', (tx) => {
      const range = sessionRange(id);
      tx.objectStore(S_SESSIONS).delete(id);
      tx.objectStore(S_FRAMES).delete(range);
      tx.objectStore(S_FRAME_DATA).delete(range);
      tx.objectStore(S_BLOBS).delete(range);
      tx.objectStore(S_JSON).delete(range);
    });
  }

  // -------------------------------------------------------------------------
  // Frames
  // -------------------------------------------------------------------------

  /**
   * Store (or replace) a frame's JPEG and info, and update the session's
   * numFrames / updatedAt in the same transaction. Rejects with
   * SessionNotFoundError if the session does not exist.
   */
  async putFrame(sessionId: string, frameId: number, jpeg: Blob, info: unknown): Promise<void> {
    checkFrameId(frameId);
    await this.withBlobFallback(true, async () => {
      const value = await this.blobValue(jpeg);
      return this.tx<void>(
        [S_SESSIONS, S_FRAMES, S_FRAME_DATA],
        'readwrite',
        'storing a frame',
        (tx, _set, fail) => {
          const sessions = tx.objectStore(S_SESSIONS);
          const frames = tx.objectStore(S_FRAMES);
          const key = [sessionId, frameId];
          SessionStore.on(sessions.get(sessionId), fail, (meta: SessionMeta | undefined) => {
            if (!meta) throw new SessionNotFoundError(sessionId);
            frames.put(info, key);
            tx.objectStore(S_FRAME_DATA).put(value, key);
            SessionStore.on(frames.count(sessionRange(sessionId)), fail, (count: number) => {
              meta.numFrames = count;
              meta.updatedAt = Math.max(Date.now(), meta.updatedAt);
              sessions.put(meta);
            });
          });
        },
        'relaxed',
      );
    });
  }

  /** Frame JPEG + info, or null. */
  getFrame<T = unknown>(sessionId: string, frameId: number): Promise<{ jpeg: Blob; info: T } | null> {
    checkFrameId(frameId);
    return this.tx([S_FRAMES, S_FRAME_DATA], 'readonly', 'reading a frame', (tx, set, fail) => {
      const key = [sessionId, frameId];
      SessionStore.on(tx.objectStore(S_FRAMES).get(key), fail, (info: T | undefined) => {
        if (info === undefined) {
          set(null);
          return;
        }
        SessionStore.on(tx.objectStore(S_FRAME_DATA).get(key), fail, (v: Blob | BufferedBlob | undefined) => {
          if (v === undefined) {
            set(null);
            return;
          }
          const jpeg = isBufferedBlob(v) ? new Blob([v.$blobBuffer], { type: v.type || 'image/jpeg' }) : v;
          set({ jpeg, info });
        });
      });
    });
  }

  /** Infos of all frames of a session, ascending frameId. */
  getFrameInfos<T = unknown>(sessionId: string): Promise<T[]> {
    return this.tx<T[]>([S_FRAMES], 'readonly', 'reading frame infos', (tx, set, fail) => {
      SessionStore.on(tx.objectStore(S_FRAMES).getAll(sessionRange(sessionId)), fail, (v: T[]) => set(v));
    });
  }

  /** Frame ids of a session, ascending. */
  getFrameIds(sessionId: string): Promise<number[]> {
    return this.tx<number[]>([S_FRAMES], 'readonly', 'listing frames', (tx, set, fail) => {
      SessionStore.on(tx.objectStore(S_FRAMES).getAllKeys(sessionRange(sessionId)), fail, (keys: IDBValidKey[]) => {
        set(keys.map((k) => (k as [string, number])[1]));
      });
    });
  }

  /** Delete one frame (no-op if absent) and update numFrames. */
  deleteFrame(sessionId: string, frameId: number): Promise<void> {
    checkFrameId(frameId);
    return this.tx<void>([S_SESSIONS, S_FRAMES, S_FRAME_DATA], 'readwrite', 'deleting a frame', (tx, _set, fail) => {
      const key = [sessionId, frameId];
      const frames = tx.objectStore(S_FRAMES);
      frames.delete(key);
      tx.objectStore(S_FRAME_DATA).delete(key);
      const sessions = tx.objectStore(S_SESSIONS);
      SessionStore.on(sessions.get(sessionId), fail, (meta: SessionMeta | undefined) => {
        if (!meta) return;
        SessionStore.on(frames.count(sessionRange(sessionId)), fail, (count: number) => {
          meta.numFrames = count;
          meta.updatedAt = Math.max(Date.now(), meta.updatedAt);
          sessions.put(meta);
        });
      });
    });
  }

  // -------------------------------------------------------------------------
  // Binary blobs
  // -------------------------------------------------------------------------

  /**
   * Store binary data under (sessionId, key). ArrayBuffers are stored as-is
   * (typed-array views are copied to a standalone ArrayBuffer); Blobs as Blobs.
   */
  async putBlob(sessionId: string, key: string, data: Blob | ArrayBuffer | ArrayBufferView): Promise<void> {
    checkKey(key);
    let value: Blob | ArrayBuffer;
    if (data instanceof Blob || data instanceof ArrayBuffer) value = data;
    else if (ArrayBuffer.isView(data)) {
      const copy = new Uint8Array(data.byteLength);
      copy.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      value = copy.buffer;
    } else throw new TypeError('putBlob: data must be a Blob, ArrayBuffer or ArrayBufferView');
    const isBlob = value instanceof Blob;
    await this.withBlobFallback(isBlob, async () => {
      const stored = isBlob ? await this.blobValue(value as Blob) : value;
      return this.tx<void>([S_BLOBS], 'readwrite', `storing blob '${key}'`, (tx) => {
        tx.objectStore(S_BLOBS).put(stored, [sessionId, key]);
      });
    });
  }

  /** The stored Blob or ArrayBuffer (same kind as stored), or null. */
  getBlob(sessionId: string, key: string): Promise<Blob | ArrayBuffer | null> {
    checkKey(key);
    return this.tx([S_BLOBS], 'readonly', `reading blob '${key}'`, (tx, set, fail) => {
      SessionStore.on(tx.objectStore(S_BLOBS).get([sessionId, key]), fail, (v: Blob | ArrayBuffer | BufferedBlob | undefined) => {
        if (v === undefined) set(null);
        else if (isBufferedBlob(v)) set(new Blob([v.$blobBuffer], { type: v.type }));
        else set(v);
      });
    });
  }

  /** Like getBlob, always as an ArrayBuffer. */
  async getArrayBuffer(sessionId: string, key: string): Promise<ArrayBuffer | null> {
    const v = await this.getBlob(sessionId, key);
    if (v === null) return null;
    return v instanceof Blob ? v.arrayBuffer() : v;
  }

  deleteBlob(sessionId: string, key: string): Promise<void> {
    checkKey(key);
    return this.tx<void>([S_BLOBS], 'readwrite', `deleting blob '${key}'`, (tx) => {
      tx.objectStore(S_BLOBS).delete([sessionId, key]);
    });
  }

  /** Blob keys of a session, ascending. */
  listBlobKeys(sessionId: string): Promise<string[]> {
    return this.listKeys(S_BLOBS, sessionId);
  }

  // -------------------------------------------------------------------------
  // JSON / structured values
  // -------------------------------------------------------------------------

  /** Store any structured-clonable value (plain JSON, typed arrays, Maps…). */
  putJSON(sessionId: string, key: string, value: unknown): Promise<void> {
    checkKey(key);
    return this.tx<void>([S_JSON], 'readwrite', `storing '${key}'`, (tx) => {
      tx.objectStore(S_JSON).put(value, [sessionId, key]);
    });
  }

  getJSON<T = unknown>(sessionId: string, key: string): Promise<T | null> {
    checkKey(key);
    return this.tx<T | null>([S_JSON], 'readonly', `reading '${key}'`, (tx, set, fail) => {
      SessionStore.on(tx.objectStore(S_JSON).get([sessionId, key]), fail, (v: T | undefined) => set(v === undefined ? null : v));
    });
  }

  deleteJSON(sessionId: string, key: string): Promise<void> {
    checkKey(key);
    return this.tx<void>([S_JSON], 'readwrite', `deleting '${key}'`, (tx) => {
      tx.objectStore(S_JSON).delete([sessionId, key]);
    });
  }

  listJSONKeys(sessionId: string): Promise<string[]> {
    return this.listKeys(S_JSON, sessionId);
  }

  private listKeys(store: string, sessionId: string): Promise<string[]> {
    return this.tx<string[]>([store], 'readonly', 'listing keys', (tx, set, fail) => {
      SessionStore.on(tx.objectStore(store).getAllKeys(sessionRange(sessionId)), fail, (keys: IDBValidKey[]) => {
        set(keys.map((k) => String((k as [string, string])[1])));
      });
    });
  }

  /**
   * Approximate bytes held by a session's frames and blobs (Blob sizes are
   * read from the handles; their contents are not loaded).
   */
  sessionBytes(sessionId: string): Promise<number> {
    return this.tx<number>([S_FRAME_DATA, S_BLOBS], 'readonly', 'measuring a session', (tx, set, fail) => {
      let total = 0;
      const sizeOf = (v: unknown): number =>
        v instanceof Blob ? v.size : v instanceof ArrayBuffer ? v.byteLength : isBufferedBlob(v) ? v.$blobBuffer.byteLength : 0;
      const range = sessionRange(sessionId);
      SessionStore.on(tx.objectStore(S_FRAME_DATA).getAll(range), fail, (a: unknown[]) => {
        for (const v of a) total += sizeOf(v);
        SessionStore.on(tx.objectStore(S_BLOBS).getAll(range), fail, (b: unknown[]) => {
          for (const v of b) total += sizeOf(v);
          set(total);
        });
      });
    });
  }

  // -------------------------------------------------------------------------
  // Quota / persistence
  // -------------------------------------------------------------------------

  /** navigator.storage.estimate() (+ persisted()); null where unsupported. */
  async estimate(): Promise<StorageEstimateInfo | null> {
    const storage = (globalThis.navigator as Navigator | undefined)?.storage;
    if (!storage || typeof storage.estimate !== 'function') return null;
    try {
      const e = await storage.estimate();
      let persisted: boolean | null = null;
      try {
        if (typeof storage.persisted === 'function') persisted = await storage.persisted();
      } catch {
        persisted = null;
      }
      const usage = e.usage ?? 0;
      const quota = e.quota ?? 0;
      return { usage, quota, available: Math.max(0, quota - usage), persisted };
    } catch {
      return null;
    }
  }

  /**
   * Ask for persistent storage (protects captures from eviction — Safari evicts
   * script-writable storage of sites not used for 7 days unless installed to
   * the Home Screen or persisted). Window scope only; false in workers or when
   * denied/unsupported. Call after a user gesture for the best chance of a grant.
   */
  async requestPersistence(): Promise<boolean> {
    const storage = (globalThis.navigator as Navigator | undefined)?.storage;
    if (!storage) return false;
    try {
      if (typeof storage.persisted === 'function' && (await storage.persisted())) return true;
      if (typeof storage.persist !== 'function') return false;
      return await storage.persist();
    } catch {
      return false;
    }
  }
}
