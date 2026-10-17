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
 * Connection loss (docs/research/ios-platform.md §2.5): WebKit closes the
 * IDBDatabase when the IndexedDB server connection is lost (backgrounding,
 * network-process crash) and we close it ourselves on `versionchange`. A
 * transaction that cannot start (InvalidStateError) or is aborted with
 * InvalidStateError / UnknownError while the connection can no longer start
 * transactions is retried ONCE on a freshly opened connection (an aborted
 * transaction committed nothing, so the retry is safe). `close()` is final.
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

const SESSION_MODES: readonly string[] = ['object', 'scene', 'auto'] satisfies SessionMode[];
const SESSION_STATUSES: readonly string[] = ['capturing', 'captured', 'sparse', 'dense', 'meshed'] satisfies SessionStatus[];

function finiteOr(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

/**
 * A stored session record with every field of its declared type, so records written by older
 * versions or damaged ones cannot break the session list, file names or routing: name → string,
 * timestamps → finite numbers, numFrames → integer ≥ 0, unknown mode → 'auto', unknown status →
 * 'capturing' (opens like an interrupted capture), stats → only finite-number / string values,
 * thumbnail → a Blob or dropped. null when the value is not a session record at all (no object,
 * or its id is neither a string nor a number). The stored record itself is not modified.
 */
export function normalizeSessionMeta(raw: unknown): SessionMeta | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const r = raw as Record<string, unknown>;
  const id = r.id;
  if (!(typeof id === 'string' || (typeof id === 'number' && Number.isFinite(id)))) return null;
  const updatedAt = finiteOr(r.updatedAt, finiteOr(r.createdAt, 0));
  const createdAt = finiteOr(r.createdAt, updatedAt);
  const nf = finiteOr(r.numFrames, 0);
  const meta: SessionMeta = {
    id: id as string, // numeric keys stay numbers (the record must stay addressable for delete / update)
    name: typeof r.name === 'string' ? r.name : r.name === undefined || r.name === null ? '' : String(r.name),
    createdAt,
    updatedAt,
    mode: SESSION_MODES.includes(r.mode as string) ? (r.mode as SessionMode) : 'auto',
    status: SESSION_STATUSES.includes(r.status as string) ? (r.status as SessionStatus) : 'capturing',
    numFrames: nf > 0 ? Math.floor(nf) : 0,
  };
  if (!meta.name.trim()) meta.name = 'Untitled scan';
  if (r.stats && typeof r.stats === 'object' && !Array.isArray(r.stats)) {
    const stats: Record<string, number | string> = {};
    for (const [k, v] of Object.entries(r.stats as Record<string, unknown>)) {
      if (typeof v === 'string' || (typeof v === 'number' && Number.isFinite(v))) stats[k] = v;
    }
    meta.stats = stats;
  }
  if (typeof Blob !== 'undefined' && r.thumbnail instanceof Blob) meta.thumbnail = r.thumbnail;
  return meta;
}

function isBufferedBlob(v: unknown): v is BufferedBlob {
  return !!v && typeof v === 'object' && (v as BufferedBlob).$blobBuffer instanceof ArrayBuffer;
}

function isQuotaError(e: unknown): boolean {
  if (!e || typeof e !== 'object') return false;
  const err = e as { name?: string; code?: number; message?: string };
  return err.name === 'QuotaExceededError' || err.code === 22 || (err.name === 'UnknownError' && /quota/i.test(err.message ?? ''));
}

function errorName(e: unknown): string | undefined {
  return e && typeof e === 'object' ? (e as { name?: string }).name : undefined;
}

/** Errors with which an engine refuses to store a Blob (old WebKit, private modes). */
const BLOB_REFUSAL_ERRORS = new Set(['DataCloneError', 'UnknownError', 'NotSupportedError', 'DataError']);

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

/** Open (and create / upgrade) the database; the connection closes itself on versionchange. */
function openDatabase(dbName: string): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
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
      // Let a newer version / deleteDatabase (another tab) proceed instead of
      // blocking it; the store reconnects on its next operation.
      db.onversionchange = () => db.close();
      resolve(db);
    };
    req.onerror = () => reject(mapError(req.error, 'opening the database'));
    // onblocked: an older connection is open elsewhere; it closes itself on
    // versionchange (see above), after which onsuccess fires.
  });
}

async function toBufferedBlob(data: Blob): Promise<BufferedBlob> {
  return { $blobBuffer: await data.arrayBuffer(), type: data.type };
}

export interface SessionStoreOpenOptions {
  /**
   * Store Blobs as ArrayBuffers (workaround for engines that cannot persist
   * Blobs in IndexedDB). Normally auto-detected after the first failure.
   */
  blobsAsArrayBuffers?: boolean;
}

export class SessionStore {
  static readonly DB_VERSION = DB_VERSION;

  private conn: IDBDatabase;
  private blobsAsBuffers: boolean;
  private closed = false;
  private reopening: Promise<void> | null = null;

  private constructor(
    db: IDBDatabase,
    readonly dbName: string,
    opts: SessionStoreOpenOptions,
  ) {
    this.conn = db;
    this.blobsAsBuffers = !!opts.blobsAsArrayBuffers;
  }

  /** The current connection (replaced when the store reconnects after a connection loss). */
  get db(): IDBDatabase {
    return this.conn;
  }

  /** True when IndexedDB is available in this realm. */
  static isSupported(): boolean {
    return typeof globalThis.indexedDB !== 'undefined' && typeof IDBKeyRange !== 'undefined';
  }

  /** Open (and create / upgrade) the database. */
  static async open(dbName = 'sfm', opts: SessionStoreOpenOptions = {}): Promise<SessionStore> {
    if (!SessionStore.isSupported()) throw new Error('IndexedDB is not available in this context');
    return new SessionStore(await openDatabase(dbName), dbName, opts);
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

  /** Close the connection for good (later calls reject with InvalidStateError). */
  close(): void {
    if (!this.closed) {
      this.closed = true;
      this.conn.close();
    }
  }

  // -------------------------------------------------------------------------
  // Transaction helper
  // -------------------------------------------------------------------------

  /** Replace a lost connection (shared by concurrent callers). */
  private reconnect(): Promise<void> {
    if (!this.reopening) {
      const old = this.conn;
      this.reopening = openDatabase(this.dbName).then(
        (db) => {
          this.reopening = null;
          if (this.closed) {
            db.close();
            return;
          }
          try {
            old.close();
          } catch {
            /* already closed */
          }
          this.conn = db;
        },
        (e: unknown) => {
          this.reopening = null;
          throw e;
        },
      );
    }
    return this.reopening;
  }

  /**
   * True when `e` is a connection-level failure: InvalidStateError / UnknownError
   * AND `conn` can no longer start a transaction (probe). A genuine error on a
   * healthy connection is not retried.
   */
  private isConnectionLoss(e: unknown, conn: IDBDatabase): boolean {
    const name = errorName(e);
    if (name !== 'InvalidStateError' && name !== 'UnknownError') return false;
    try {
      conn.transaction(S_SESSIONS, 'readonly'); // no requests → commits immediately
      return false;
    } catch {
      return true;
    }
  }

  private async tx<T>(
    stores: string[],
    mode: IDBTransactionMode,
    what: string,
    body: TxBody<T>,
    durability?: 'default' | 'relaxed' | 'strict',
  ): Promise<T> {
    const conn = this.conn;
    try {
      return await this.runTx(conn, stores, mode, what, body, durability);
    } catch (e) {
      if (this.closed || !this.isConnectionLoss(e, conn)) throw e;
      if (this.conn === conn) await this.reconnect();
      return this.runTx(this.conn, stores, mode, what, body, durability);
    }
  }

  private runTx<T>(
    db: IDBDatabase,
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
        tx = durability ? db.transaction(stores, mode, { durability }) : db.transaction(stores, mode);
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

  /**
   * Run a write that stores a Blob; if the engine refuses it, retry once with
   * the Blob as an ArrayBuffer. Buffer mode becomes permanent ONLY if that retry
   * succeeds — an error caused by something else (e.g. a non-cloneable frame
   * info) fails both attempts and leaves Blob storage untouched.
   */
  private async withBlobFallback<T>(hasBlob: boolean, op: (asBuffers: boolean) => Promise<T>): Promise<T> {
    const asBuffers = this.blobsAsBuffers;
    try {
      return await op(asBuffers);
    } catch (e) {
      if (!hasBlob || asBuffers || e instanceof StorageQuotaError || e instanceof SessionNotFoundError) throw e;
      if (!BLOB_REFUSAL_ERRORS.has(errorName(e) ?? '')) throw e;
      let result: T;
      try {
        result = await op(true);
      } catch (e2) {
        if (e2 instanceof StorageQuotaError || e2 instanceof SessionNotFoundError) throw e2;
        // not (only) a Blob problem: report the original error, keep Blob mode — with the retry's
        // error in the message, which otherwise vanishes
        const err = new Error(`${(e as Error).message} (retry as ArrayBuffer: ${errorName(e2) ?? 'Error'}: ${(e2 as Error)?.message ?? e2})`);
        err.name = errorName(e) ?? 'Error';
        throw err;
      }
      this.blobsAsBuffers = true;
      return result;
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

  /** The session record, normalised (`normalizeSessionMeta`); null if missing or unusable. */
  getSession(id: string): Promise<SessionMeta | null> {
    return this.tx<SessionMeta | null>([S_SESSIONS], 'readonly', 'reading a session', (tx, set, fail) => {
      SessionStore.on(tx.objectStore(S_SESSIONS).get(id), fail, (v: unknown) => set(normalizeSessionMeta(v)));
    });
  }

  /**
   * All sessions, newest first (by createdAt, or updatedAt when requested), normalised
   * (`normalizeSessionMeta`: records of older versions or damaged ones never break the list).
   */
  listSessions(opts: { orderBy?: 'createdAt' | 'updatedAt' } = {}): Promise<SessionMeta[]> {
    const key = opts.orderBy ?? 'createdAt';
    return this.tx<SessionMeta[]>([S_SESSIONS], 'readonly', 'listing sessions', (tx, set, fail) => {
      SessionStore.on(tx.objectStore(S_SESSIONS).getAll(), fail, (raw: unknown[]) => {
        const all = raw.map(normalizeSessionMeta).filter((m): m is SessionMeta => m !== null);
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
    await this.withBlobFallback(true, async (asBuffers) => {
      const value = asBuffers ? await toBufferedBlob(jpeg) : jpeg; // awaited BEFORE the transaction starts
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
    await this.withBlobFallback(isBlob, async (asBuffers) => {
      const stored = isBlob && asBuffers ? await toBufferedBlob(value as Blob) : value; // before the transaction
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
