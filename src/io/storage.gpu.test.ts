import { afterEach, describe, expect, it } from 'vitest';
import { SessionNotFoundError, SessionStore, StorageQuotaError } from './storage';

const opened: { store: SessionStore; name: string }[] = [];

async function openFresh(opts?: Parameters<typeof SessionStore.open>[1]): Promise<SessionStore> {
  const name = `sfm-test-${Math.random().toString(36).slice(2)}`;
  const store = await SessionStore.open(name, opts);
  opened.push({ store, name });
  return store;
}

afterEach(async () => {
  for (const { store, name } of opened.splice(0)) {
    store.close();
    await SessionStore.deleteDatabase(name);
  }
});

function jpeg(tag: number): Blob {
  return new Blob([new Uint8Array([0xff, 0xd8, tag, tag + 1, 0xff, 0xd9])], { type: 'image/jpeg' });
}

async function bytesOf(b: Blob | ArrayBuffer): Promise<number[]> {
  return Array.from(new Uint8Array(b instanceof Blob ? await b.arrayBuffer() : b));
}

describe('SessionStore (IndexedDB)', () => {
  it('creates, updates, gets and lists sessions (newest first)', async () => {
    const store = await openFresh();
    const a = await store.createSession({ name: 'garden', mode: 'scene' });
    await new Promise((r) => setTimeout(r, 5));
    const b = await store.createSession({ name: 'mug', mode: 'object' });
    expect(a.id).not.toBe(b.id);
    expect(a.status).toBe('capturing');
    expect(a.numFrames).toBe(0);

    await new Promise((r) => setTimeout(r, 5));
    const thumb = new Blob(['thumb'], { type: 'image/webp' });
    const upd = await store.updateSession(a.id, { status: 'sparse', stats: { points: 1234, model: 'RADIAL' }, thumbnail: thumb });
    expect(upd.status).toBe('sparse');
    expect(upd.createdAt).toBe(a.createdAt);
    expect(upd.updatedAt).toBeGreaterThan(a.updatedAt);

    const got = await store.getSession(a.id);
    expect(got!.stats).toEqual({ points: 1234, model: 'RADIAL' });
    expect(got!.thumbnail).toBeInstanceOf(Blob);
    expect(await got!.thumbnail!.text()).toBe('thumb');
    expect(await store.getSession('nope')).toBeNull();

    expect((await store.listSessions()).map((s) => s.name)).toEqual(['mug', 'garden']);
    expect((await store.listSessions({ orderBy: 'updatedAt' })).map((s) => s.name)).toEqual(['garden', 'mug']);

    await expect(store.updateSession('missing', { name: 'x' })).rejects.toBeInstanceOf(SessionNotFoundError);
    await expect(store.createSession({ id: a.id })).rejects.toBeTruthy(); // duplicate id
  });

  it('stores frames with infos, ordered by numeric id, and maintains numFrames', async () => {
    const store = await openFresh();
    const s = await store.createSession();
    const ids = [10, 2, 1, 33];
    // Concurrent writes must not lose numFrames updates.
    await Promise.all(ids.map((id) => store.putFrame(s.id, id, jpeg(id), { id, t: id * 100, R: Float64Array.of(id, 0, 0) })));
    expect(await store.getFrameIds(s.id)).toEqual([1, 2, 10, 33]);
    const infos = await store.getFrameInfos<{ id: number; R: Float64Array }>(s.id);
    expect(infos.map((i) => i.id)).toEqual([1, 2, 10, 33]);
    expect(infos[0].R).toBeInstanceOf(Float64Array);
    expect((await store.getSession(s.id))!.numFrames).toBe(4);

    const f = await store.getFrame<{ t: number }>(s.id, 10);
    expect(f!.info.t).toBe(1000);
    expect(f!.jpeg.type).toBe('image/jpeg');
    expect(await bytesOf(f!.jpeg)).toEqual([0xff, 0xd8, 10, 11, 0xff, 0xd9]);
    expect(await store.getFrame(s.id, 999)).toBeNull();

    // Replace keeps the count; delete decrements.
    await store.putFrame(s.id, 10, jpeg(50), { id: 10, t: 5 });
    expect((await store.getSession(s.id))!.numFrames).toBe(4);
    await store.deleteFrame(s.id, 2);
    expect(await store.getFrameIds(s.id)).toEqual([1, 10, 33]);
    expect((await store.getSession(s.id))!.numFrames).toBe(3);

    await expect(store.putFrame('missing', 1, jpeg(1), {})).rejects.toBeInstanceOf(SessionNotFoundError);
    expect(await store.getFrameIds('missing')).toEqual([]);
  });

  it('stores blobs, ArrayBuffers, views and structured JSON', async () => {
    const store = await openFresh();
    const s = await store.createSession();
    await store.putBlob(s.id, 'features/1', new Uint8Array([1, 2, 3, 4]).buffer);
    await store.putBlob(s.id, 'export.ply', new Blob([new Uint8Array([5, 6])], { type: 'application/octet-stream' }));
    const big = new Uint8Array(100);
    big.fill(7);
    await store.putBlob(s.id, 'view', big.subarray(10, 13)); // view → standalone copy
    const ab = await store.getBlob(s.id, 'features/1');
    expect(ab).toBeInstanceOf(ArrayBuffer);
    expect(await bytesOf(ab!)).toEqual([1, 2, 3, 4]);
    const bl = await store.getBlob(s.id, 'export.ply');
    expect(bl).toBeInstanceOf(Blob);
    expect(await bytesOf(bl!)).toEqual([5, 6]);
    expect(await bytesOf((await store.getArrayBuffer(s.id, 'export.ply'))!)).toEqual([5, 6]);
    expect(((await store.getBlob(s.id, 'view')) as ArrayBuffer).byteLength).toBe(3);
    expect(await store.listBlobKeys(s.id)).toEqual(['export.ply', 'features/1', 'view']);
    expect(await store.sessionBytes(s.id)).toBe(4 + 2 + 3);
    await store.deleteBlob(s.id, 'view');
    expect(await store.getBlob(s.id, 'view')).toBeNull();

    const recon = { poses: [{ R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(0, 0, 1) }], metric: { metric: true } };
    await store.putJSON(s.id, 'recon', recon);
    const back = await store.getJSON<typeof recon>(s.id, 'recon');
    expect(back!.poses[0].R).toBeInstanceOf(Float64Array);
    expect(Array.from(back!.poses[0].t)).toEqual([0, 0, 1]);
    expect(await store.getJSON(s.id, 'nope')).toBeNull();
    expect(await store.listJSONKeys(s.id)).toEqual(['recon']);
    await store.deleteJSON(s.id, 'recon');
    expect(await store.getJSON(s.id, 'recon')).toBeNull();
  });

  it('deleteSession cascades to every store without touching other sessions (incl. prefix ids)', async () => {
    const store = await openFresh();
    const a = await store.createSession({ id: 'a' });
    const ab = await store.createSession({ id: 'ab' });
    for (const s of [a, ab]) {
      await store.putFrame(s.id, 1, jpeg(1), { id: 1 });
      await store.putFrame(s.id, 2, jpeg(2), { id: 2 });
      await store.putBlob(s.id, 'k', new Uint8Array([1]).buffer);
      await store.putJSON(s.id, 'j', { x: 1 });
    }
    await store.deleteSession('a');
    expect(await store.getSession('a')).toBeNull();
    expect(await store.getFrameIds('a')).toEqual([]);
    expect(await store.getFrameInfos('a')).toEqual([]);
    expect(await store.getFrame('a', 1)).toBeNull();
    expect(await store.listBlobKeys('a')).toEqual([]);
    expect(await store.listJSONKeys('a')).toEqual([]);
    expect(await store.sessionBytes('a')).toBe(0);
    // Untouched neighbour.
    expect((await store.getSession('ab'))!.numFrames).toBe(2);
    expect(await store.getFrameIds('ab')).toEqual([1, 2]);
    expect(await store.listBlobKeys('ab')).toEqual(['k']);
    expect(await store.getJSON('ab', 'j')).toEqual({ x: 1 });
    expect((await store.listSessions()).map((s) => s.id)).toEqual(['ab']);
  });

  it('falls back to ArrayBuffer storage for Blobs transparently', async () => {
    const store = await openFresh({ blobsAsArrayBuffers: true });
    const s = await store.createSession();
    await store.putFrame(s.id, 1, jpeg(3), { id: 1 });
    const f = await store.getFrame(s.id, 1);
    expect(f!.jpeg).toBeInstanceOf(Blob);
    expect(f!.jpeg.type).toBe('image/jpeg');
    expect(await bytesOf(f!.jpeg)).toEqual([0xff, 0xd8, 3, 4, 0xff, 0xd9]);
    await store.putBlob(s.id, 'b', new Blob(['hi'], { type: 'text/plain' }));
    const b = await store.getBlob(s.id, 'b');
    expect(b).toBeInstanceOf(Blob);
    expect((b as Blob).type).toBe('text/plain');
    expect(await (b as Blob).text()).toBe('hi');
    expect(await store.sessionBytes(s.id)).toBe(6 + 2);
  });

  it('maps QuotaExceededError to StorageQuotaError and leaves no partial writes', async () => {
    const store = await openFresh();
    const s = await store.createSession();
    const proto = IDBObjectStore.prototype;
    const orig = proto.put;
    proto.put = function (this: IDBObjectStore, ...args: Parameters<IDBObjectStore['put']>) {
      if (this.name === 'frameData') throw new DOMException('The quota has been exceeded.', 'QuotaExceededError');
      return orig.apply(this, args);
    };
    try {
      const err = await store.putFrame(s.id, 1, jpeg(1), { id: 1 }).catch((e: unknown) => e);
      expect(err).toBeInstanceOf(StorageQuotaError);
      expect((err as StorageQuotaError).name).toBe('StorageQuotaError');
      expect((err as StorageQuotaError).cause).toBeInstanceOf(DOMException);
    } finally {
      proto.put = orig;
    }
    // The transaction was aborted: neither the info nor the count were written.
    expect(await store.getFrameIds(s.id)).toEqual([]);
    expect((await store.getSession(s.id))!.numFrames).toBe(0);
  });

  it('reports estimate() and requestPersistence() without throwing', async () => {
    const store = await openFresh();
    const est = await store.estimate();
    expect(est).not.toBeNull();
    expect(est!.quota).toBeGreaterThan(0);
    expect(est!.available).toBe(Math.max(0, est!.quota - est!.usage));
    expect(typeof (await store.requestPersistence())).toBe('boolean');
  });

  it('works from a module worker sharing the same database', async () => {
    const store = await openFresh();
    const s = await store.createSession();
    await store.putFrame(s.id, 1, jpeg(1), { id: 1 });
    const name = opened[opened.length - 1].name;
    const worker = new Worker(new URL('./storage.worker-fixture.ts', import.meta.url), { type: 'module' });
    try {
      const res = await new Promise<{ ok: boolean; error?: string; before: number[]; numFrames: number; hasEstimate: boolean; persisted: boolean }>(
        (resolve, reject) => {
          worker.onmessage = (ev) => resolve(ev.data);
          worker.onerror = (ev) => reject(new Error(ev.message || 'worker error'));
          worker.postMessage({ dbName: name, sessionId: s.id });
        },
      );
      expect(res.error).toBeUndefined();
      expect(res.ok).toBe(true);
      expect(res.before).toEqual([1]);
      expect(res.numFrames).toBe(2);
      expect(typeof res.persisted).toBe('boolean');
    } finally {
      worker.terminate();
    }
    // Page sees the worker's writes.
    expect(await store.getFrameIds(s.id)).toEqual([1, 7]);
    const info = (await store.getFrame<{ from: string; R: Float64Array }>(s.id, 7))!.info;
    expect(info.from).toBe('worker');
    expect(info.R).toBeInstanceOf(Float64Array);
    expect(Array.from((await store.getJSON<{ values: Float64Array }>(s.id, 'worker-state'))!.values)).toEqual([1.5, 2.5]);
    expect(await bytesOf((await store.getBlob(s.id, 'worker-bin'))!)).toEqual([9, 8, 7]);
  });
});
