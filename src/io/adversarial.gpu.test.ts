/**
 * Adversarial regression tests for src/io browser code (review pass): IndexedDB
 * SessionStore robustness and the iOS share-sheet fallback. Each test names the
 * defect it pins down.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { saveFile } from './download';
import { SessionStore } from './storage';

const opened: { store: SessionStore; name: string }[] = [];

async function openFresh(opts?: Parameters<typeof SessionStore.open>[1]): Promise<SessionStore> {
  const name = `sfm-adv-${Math.random().toString(36).slice(2)}`;
  const store = await SessionStore.open(name, opts);
  opened.push({ store, name });
  return store;
}

afterEach(async () => {
  for (const { store, name } of opened.splice(0)) {
    store.close();
    await SessionStore.deleteDatabase(name);
  }
  vi.restoreAllMocks();
});

function jpeg(tag: number): Blob {
  return new Blob([new Uint8Array([0xff, 0xd8, tag, 0xff, 0xd9])], { type: 'image/jpeg' });
}

/** Independent raw connection (no SessionStore code) for white-box checks. */
function rawOpen(name: string): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(name);
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

function rawRequest<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

describe('SessionStore (adversarial)', () => {
  it('a DataCloneError caused by the frame INFO does not permanently switch Blob storage to ArrayBuffers', async () => {
    const store = await openFresh();
    const s = await store.createSession();
    const err = await store.putFrame(s.id, 1, jpeg(1), { cb: () => 1 }).catch((e: unknown) => e);
    expect((err as DOMException).name).toBe('DataCloneError');
    await store.putFrame(s.id, 2, jpeg(2), { id: 2 });
    expect(await store.getFrameIds(s.id)).toEqual([2]);
    expect((await store.getSession(s.id))!.numFrames).toBe(1);
    const db = await rawOpen(store.dbName);
    try {
      const raw = await rawRequest(db.transaction('frameData').objectStore('frameData').get([s.id, 2]));
      expect(raw).toBeInstanceOf(Blob); // not a { $blobBuffer } record
    } finally {
      db.close();
    }
  });

  it('cascade delete leaves no record of the session in ANY object store of the schema', async () => {
    const store = await openFresh();
    for (const id of ['s', 's2', 'r']) {
      await store.createSession({ id });
      await store.putFrame(id, 1, jpeg(1), { id: 1 });
      await store.putFrame(id, 2.5, jpeg(2), { id: 2.5 });
      await store.putBlob(id, 'k', new Uint8Array([1]).buffer);
      await store.putBlob(id, 'z', new Blob(['z']));
      await store.putJSON(id, 'j', { x: 1 });
    }
    await store.deleteSession('s');
    const db = await rawOpen(store.dbName);
    try {
      const names = Array.from(db.objectStoreNames);
      expect(names.length).toBeGreaterThanOrEqual(5);
      const tx = db.transaction(names);
      for (const n of names) {
        const keys = await rawRequest(tx.objectStore(n).getAllKeys());
        const ofS = keys.filter((k) => (Array.isArray(k) ? k[0] === 's' : k === 's'));
        expect(ofS, n).toEqual([]);
        const ofOthers = keys.filter((k) => (Array.isArray(k) ? k[0] !== 's' : k !== 's'));
        expect(ofOthers.length, n).toBeGreaterThan(0);
      }
    } finally {
      db.close();
    }
  });

  it('reconnects when the connection was closed underneath (WebKit "connection lost" / versionchange)', async () => {
    const store = await openFresh();
    await store.createSession({ id: 's' });
    store.db.close(); // what WebKit does to the IDBDatabase when the IDB server connection is lost
    expect((await store.getSession('s'))!.id).toBe('s');
    await store.putFrame('s', 1, jpeg(1), { id: 1 });
    expect((await store.getSession('s'))!.numFrames).toBe(1);
  });

  it('keeps working after another context deletes the database (versionchange → close)', async () => {
    const store = await openFresh();
    await store.createSession({ id: 's' });
    await SessionStore.deleteDatabase(store.dbName); // resolves: our connection closes itself on versionchange
    expect(await store.getSession('s')).toBeNull(); // fresh database, no InvalidStateError
    const t = await store.createSession({ id: 't' });
    expect((await store.listSessions()).map((x) => x.id)).toEqual([t.id]);
  });

  it('an aborted transaction caused by a lost connection (UnknownError) is retried once on a new connection', async () => {
    const store = await openFresh();
    await store.createSession({ id: 's' });
    const proto = IDBObjectStore.prototype;
    const orig = proto.get;
    let calls = 0;
    proto.get = function (this: IDBObjectStore, ...args: Parameters<IDBObjectStore['get']>) {
      calls++;
      if (calls === 1) {
        this.transaction.db.close();
        throw new DOMException('Connection to Indexed Database server lost. Refresh the page to try again', 'UnknownError');
      }
      return orig.apply(this, args);
    };
    try {
      expect((await store.getSession('s'))!.id).toBe('s');
      expect(calls).toBe(2);
    } finally {
      proto.get = orig;
    }
  });

  it('a genuine UnknownError on a healthy connection is surfaced, not retried', async () => {
    const store = await openFresh();
    await store.createSession({ id: 's' });
    const proto = IDBObjectStore.prototype;
    const orig = proto.get;
    let calls = 0;
    proto.get = function () {
      calls++;
      throw new DOMException('disk I/O error', 'UnknownError');
    };
    try {
      await expect(store.getSession('s')).rejects.toMatchObject({ name: 'UnknownError' });
      expect(calls).toBe(1);
    } finally {
      proto.get = orig;
    }
    expect((await store.getSession('s'))!.id).toBe('s');
  });

  it('close() is final (no silent reconnect)', async () => {
    const store = await openFresh();
    store.close();
    await expect(store.getSession('x')).rejects.toMatchObject({ name: 'InvalidStateError' });
  });
});

describe('saveFile (adversarial, iOS)', () => {
  function install(share: () => Promise<void>): void {
    Object.defineProperty(navigator, 'share', { value: share, configurable: true, writable: true });
    Object.defineProperty(navigator, 'canShare', { value: () => true, configurable: true, writable: true });
  }
  afterEach(() => {
    const n = navigator as unknown as Record<string, unknown>;
    delete n.share;
    delete n.canShare;
    delete n.standalone;
  });

  it('in an iOS Home Screen app an expired activation rethrows instead of using the broken <a download> path', async () => {
    install(async () => {
      throw new DOMException('Must be handling a user gesture', 'NotAllowedError');
    });
    Object.defineProperty(navigator, 'standalone', { value: true, configurable: true });
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    await expect(saveFile(new Blob(['x']), 'scan.glb')).rejects.toMatchObject({ name: 'NotAllowedError' });
    expect(click).not.toHaveBeenCalled();
  });

  it('in a Safari tab (navigator.standalone === false) it still falls back to a download', async () => {
    install(async () => {
      throw new DOMException('Must be handling a user gesture', 'NotAllowedError');
    });
    Object.defineProperty(navigator, 'standalone', { value: false, configurable: true });
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    expect(await saveFile(new Blob(['x']), 'scan.glb', { revokeDelayMs: 0 })).toBe('downloaded');
    expect(click).toHaveBeenCalledTimes(1);
  });
});
