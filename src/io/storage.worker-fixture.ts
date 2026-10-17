/**
 * Test fixture (module worker) for storage.gpu.test.ts: proves SessionStore
 * works in a DedicatedWorkerGlobalScope and shares the database with the page.
 */
import { SessionStore } from './storage';

interface Req {
  dbName: string;
  sessionId: string;
}

const scope = self as unknown as {
  onmessage: ((ev: MessageEvent<Req>) => void) | null;
  postMessage(msg: unknown): void;
};

scope.onmessage = async (ev) => {
  const { dbName, sessionId } = ev.data;
  try {
    const store = await SessionStore.open(dbName);
    const before = await store.getFrameIds(sessionId);
    const jpeg = new Blob([new Uint8Array([0xff, 0xd8, 0x01, 0x02, 0xff, 0xd9])], { type: 'image/jpeg' });
    await store.putFrame(sessionId, 7, jpeg, { id: 7, from: 'worker', R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1) });
    await store.putJSON(sessionId, 'worker-state', { ok: true, values: Float64Array.of(1.5, 2.5) });
    await store.putBlob(sessionId, 'worker-bin', new Uint8Array([9, 8, 7]).buffer);
    const meta = await store.getSession(sessionId);
    const estimate = await store.estimate();
    const persisted = await store.requestPersistence();
    store.close();
    scope.postMessage({ ok: true, before, numFrames: meta?.numFrames ?? -1, hasEstimate: estimate !== null, persisted });
  } catch (e) {
    scope.postMessage({ ok: false, error: String(e) });
  }
};
