/**
 * iPhone-specific engine behaviour on a PERSISTENT store (the app's IndexedDB path, emulated by
 * a MemoryEngineStore flagged persistent): dense resume after a kill / cancel, the live session
 * released before the dense stage (and restored by a later refine), the capture state kept
 * intact across state writes without a live session, store back-pressure during capture, results
 * whose write failed or was cut short by a kill, the mesh's object box (`mesh/meta`) across a
 * reload, a deleted session not re-created by a resume, and no writes after dispose.
 */
import { beforeAll, describe, expect, it, vi } from 'vitest';
import type { FeaturePipeline } from '../features/extractor';
import type { ProgressEvent } from '../core/types';
import { decodePLY } from '../io/ply';
import { readZip } from '../io/zip';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import { MemoryEngineStore, type EngineStateRecord, type EngineStore, type MeshMetaRecord, type StoredImage } from './persistence';
import { frameInputs, makeCapture, type SynthCapture } from './testutil';
import type { KeyframeRecord } from './types';
import type { SessionSummary } from '../workers/protocol';

let pipeline: FeaturePipeline;
let cap: SynthCapture;
beforeAll(() => {
  pipeline = createCpuFeaturePipeline({ maxFeatures: 2048, maxImageSize: 1024 });
  cap = makeCapture({ frames: 44, arcDeg: 55 });
}, 120_000);

/** MemoryEngineStore that reports `persistent` (write-through depth maps, resume, release paths). */
function persistentStore(id: string): MemoryEngineStore {
  const s = new MemoryEngineStore(id);
  Object.defineProperty(s, 'persistent', { value: true });
  return s;
}

const opts = { mode: 'object' as const, quality: 'fast' as const, sessionId: 'p', hfovPriorDeg: 60 };

describe('engine on a persistent store', () => {
  it('releases the capture state for dense, resumes a cancelled densify, keeps the capture state, refines again', async () => {
    const store = persistentStore('p');
    const e = new PhotogrammetryEngine(opts, { pipeline, store, mapping: 'inline' });
    const logs: string[] = [];
    e.on('log', (l, m) => logs.push(`${l}: ${m}`));
    await e.init();
    for (const f of frameInputs(cap)) await e.processFrame(f);
    await e.finishCapture();
    expect(logs.some((l) => l.startsWith('info: memory after capture ≈'))).toBe(true);
    const res = await e.refine();
    const nk = res.cameras.length;
    expect(nk).toBeGreaterThan(5);
    expect(typeof res.stats.memoryMB).toBe('number');
    const stBefore = (await store.getValue<EngineStateRecord>('engine/state'))!;
    expect(stBefore.thetaSession).toBe(0);

    // densify, cancelled in the middle of the geometric pass
    const dense = { maxImageSize: 96, maxViews: 8 };
    const cancel = { cancelled: false };
    let seen = 0;
    const onProgress = (p: ProgressEvent): void => {
      if (p.stage === 'depth_maps' && p.progress > 0.6 && ++seen === 1) cancel.cancelled = true;
    };
    await expect(e.densify(dense, onProgress, cancel)).rejects.toMatchObject({ name: 'CancelledError' });
    // the live session was released for the dense stage; the stored capture state is intact
    expect(e.liveSession).toBeNull();
    expect(logs.some((l) => l.includes('capture state released for the dense stage'))).toBe(true);
    expect(await store.getValue('dense/progress')).toMatchObject({ pass: 1, quality: 'fast' });
    const pm1 = await store.listBinaryKeys('pm1/');
    expect(pm1.length).toBeGreaterThan(0);
    // after a reload the app learns the interrupted run's preset (a resume needs the same one)
    const other = new PhotogrammetryEngine({ mode: 'auto', quality: 'max' }, { pipeline, store });
    const sum = await other.loadSession('p');
    expect(sum.stats).toMatchObject({ denseResumablePass: 1, denseResumableQuality: 'fast' });
    other.dispose();

    // a new densify resumes: the maps of pass 1 that exist are not recomputed
    const pc = await e.densify(dense);
    expect(pc.count).toBeGreaterThan(500);
    const stats = e.sparseResult!.stats;
    expect(stats.denseResumedPass).toBe(1);
    expect(stats.denseResumedMaps).toBe(pm1.length);
    expect(stats.fusionStep).toBe(1);
    // fused: no PatchMatch pass maps left, progress cleared, filtered maps kept for meshing
    expect(await store.listBinaryKeys('pm')).toEqual([]);
    expect(await store.getValue('dense/progress')).toBeNull();
    expect((await store.listBinaryKeys('depth/')).length).toBeGreaterThan(1);
    // state writes without a live session keep θ / counters / time offset
    const stAfter = (await store.getValue<EngineStateRecord>('engine/state'))!;
    expect(stAfter.hasDense).toBe(true);
    expect(stAfter.thetaSession).toBe(stBefore.thetaSession);
    expect(stAfter.nextKeyframeId).toBe(stBefore.nextKeyframeId);
    expect(stAfter.timeOffsetMs).toBe(stBefore.timeOffsetMs);
    expect(stAfter.captureFinished).toBe(true);

    const mesh = await e.mesh();
    expect(mesh.triangleCount).toBeGreaterThan(100);
    // exports read the keyframe features from the store
    const zip = readZip(new Uint8Array(await (await e.exportModel('colmap-zip')).arrayBuffer()));
    expect(zip.some((x) => x.name === 'sparse/0/points3D.txt')).toBe(true);

    // refine again: the capture is restored from the store
    const res2 = await e.refine();
    expect(e.liveSession).not.toBeNull();
    expect(res2.cameras.length).toBeGreaterThanOrEqual(nk - 1);
    expect(await store.listBinaryKeys('depth/')).toEqual([]);
    e.dispose();
  }, 300_000);

  it('loadSession + densify does not reset the stored capture state', async () => {
    const store = persistentStore('q');
    const a = new PhotogrammetryEngine({ ...opts, sessionId: 'q' }, { pipeline, store, mapping: 'inline' });
    await a.init();
    for (const f of frameInputs(cap, { to: 30 })) await a.processFrame(f);
    await a.finishCapture();
    await a.refine();
    const st0 = (await store.getValue<EngineStateRecord>('engine/state'))!;
    a.dispose();
    const b = new PhotogrammetryEngine({ mode: 'auto', quality: 'fast' }, { pipeline, store });
    await b.loadSession('q');
    await b.densify({ maxImageSize: 80, maxViews: 6 });
    const st1 = (await store.getValue<EngineStateRecord>('engine/state'))!;
    expect(st1.hasDense).toBe(true);
    expect(st1.thetaSession).toBe(st0.thetaSession);
    expect(st1.cameraTheta).toBe(st0.cameraTheta);
    expect(st1.cameraThetaVerified).toBe(st0.cameraThetaVerified);
    expect(st1.nextKeyframeId).toBe(st0.nextKeyframeId);
    expect(st1.imuChunks).toBe(st0.imuChunks);
    expect(st1.vocabTrainedOn).toEqual(st0.vocabTrainedOn);
    b.dispose();
  }, 300_000);
});

describe('store back-pressure during capture', () => {
  /** Store whose keyframe images are held until `open()` (re-armed by `close()`). */
  class GatedStore extends MemoryEngineStore implements EngineStore {
    private gate: Promise<void> = Promise.resolve();
    open: () => void = () => undefined;
    held = 0;
    constructor() {
      super('gated');
      this.close();
    }
    override close(): void {
      this.gate = new Promise((r) => (this.open = r));
    }
    override async putFrame(id: number, image: StoredImage, record: KeyframeRecord): Promise<void> {
      this.held++;
      await this.gate;
      return super.putFrame(id, image, record);
    }
  }

  it('a keyframe frame waits while too many keyframe writes are pending', async () => {
    const store = new GatedStore();
    // 240×320 keyframes hold ≈ 0.3 MB (canvas / image) + features until stored: 0.5 MB cap ⇒ ~1 in flight
    const e = new PhotogrammetryEngine(opts, { pipeline, store, mapping: 'inline', maxPendingWriteMB: 0.5 });
    await e.init();
    let waited = false;
    for (const f of frameInputs(cap)) {
      const p = e.processFrame(f);
      const r = await Promise.race([p.then(() => 'done'), new Promise((res) => setTimeout(() => res('pending'), 3000))]);
      if (r === 'done') continue;
      // still running after 3 s: blocked on the store (or just slow on a loaded machine)
      expect(store.held).toBeGreaterThanOrEqual(1);
      store.open();
      const u = await p;
      if ((u.timings.writeWait ?? 0) > 0) {
        expect(u.decision).toBe('keyframe');
        waited = true;
        break;
      }
      store.close();
    }
    expect(waited).toBe(true);
    store.open();
    await e.flush();
    e.dispose();
  }, 300_000);
});

describe('capture memory cap', () => {
  it('stops adding keyframes at the cap and reports it once (out-of-memory, non-fatal)', async () => {
    // (330 − 250 − 64) MB × 0.2 ≈ 3.2 MB
    const e = new PhotogrammetryEngine(opts, { pipeline, mapping: 'inline', memoryBudgetMB: 330, captureMemoryFraction: 0.2 });
    const errors: { fatal: boolean; reason?: string; name: string }[] = [];
    e.on('error', (err, fatal, reason) => errors.push({ fatal, reason, name: err.name }));
    await e.init();
    const ups = [];
    for (const f of frameInputs(cap)) ups.push(await e.processFrame(f));
    const kfs = ups.filter((u) => u.decision === 'keyframe').length;
    const limited = ups.filter((u) => u.timings.memoryLimited === 1).length;
    expect(kfs).toBeGreaterThanOrEqual(2);
    expect(limited).toBeGreaterThan(0);
    // ≈ 3 MB cap: the estimate after the last keyframe is past it, before it was not
    expect(e.liveSession!.memoryEstimate).toBeGreaterThan(3 * 1024 * 1024);
    expect(errors).toEqual([{ fatal: false, reason: 'out-of-memory', name: 'CaptureMemoryLimitError' }]);
    // the capture still refines
    await e.finishCapture();
    const res = await e.refine();
    expect(res.cameras.length).toBeGreaterThanOrEqual(2);
    e.dispose();
  }, 300_000);
});

/**
 * Persistent MemoryEngineStore whose writes can fail (`fail`: a full disk, or every write after a
 * simulated kill of the worker) — StorageQuotaError, like IndexedDB's QuotaExceededError.
 */
class FaultyStore extends MemoryEngineStore {
  fail: (op: 'put' | 'delete', key: string) => boolean = () => false;
  /** Called after each successful write. */
  written: (key: string) => void = () => undefined;
  constructor(id: string) {
    super(id);
    Object.defineProperty(this, 'persistent', { value: true });
  }
  private async write(op: 'put' | 'delete', key: string, run: () => Promise<void>): Promise<void> {
    if (this.fail(op, key)) throw Object.assign(new Error(`quota exceeded storing '${key}'`), { name: 'StorageQuotaError' });
    await run();
    this.written(key);
  }
  override putValue(key: string, value: unknown): Promise<void> {
    return this.write('put', key, () => super.putValue(key, value));
  }
  override putBinary(key: string, data: ArrayBuffer | ArrayBufferView): Promise<void> {
    return this.write('put', key, () => super.putBinary(key, data));
  }
  override deleteBinary(key: string): Promise<void> {
    return this.write('delete', key, () => super.deleteBinary(key));
  }
}

describe('stored results after a failed or interrupted write', () => {
  const DENSE = { maxImageSize: 80, maxViews: 6 };
  /** Another setting: a cloud that differs from the stored one. */
  const DENSE_2 = { maxImageSize: 64, maxViews: 6 };
  let base: FaultyStore;
  let baseDense = 0;
  beforeAll(async () => {
    base = new FaultyStore('r');
    const e = new PhotogrammetryEngine({ ...opts, sessionId: 'r' }, { pipeline, store: base, mapping: 'inline' });
    await e.init();
    for (const f of frameInputs(cap, { to: 30 })) await e.processFrame(f);
    await e.finishCapture();
    await e.refine();
    baseDense = (await e.densify(DENSE)).count;
    await e.mesh();
    e.dispose();
  }, 300_000);

  /** A copy of the meshed session (every test changes its own). */
  async function meshedScan(): Promise<FaultyStore> {
    const s = new FaultyStore('r');
    await s.ensureSession({ mode: 'object', quality: 'fast' });
    for (const r of await base.getFrameRecords()) await s.putFrame(r.id, structuredClone((await base.getFrameImage(r.id))!), structuredClone(r));
    for (const k of await base.listValueKeys()) await s.putValue(k, structuredClone(await base.getValue(k)));
    for (const k of await base.listBinaryKeys()) await s.putBinary(k, new Uint8Array((await base.getBinary(k))!));
    return s;
  }

  async function open(store: EngineStore): Promise<{ e: PhotogrammetryEngine; errors: string[]; sum: SessionSummary }> {
    const e = new PhotogrammetryEngine({ mode: 'auto', quality: 'fast' }, { pipeline, store, mapping: 'inline' });
    const errors: string[] = [];
    e.on('error', (err, _fatal, reason) => errors.push(`${reason}: ${err.name}`));
    return { e, errors, sum: await e.loadSession('r') };
  }

  /** What a reload (a fresh engine) finds in the store. */
  async function reload(store: FaultyStore): Promise<{ status: string; sparse: boolean; dense: number | null; mesh: boolean }> {
    store.fail = () => false;
    const { e, sum } = await open(store);
    e.dispose();
    return { status: sum.status, sparse: !!sum.sparse, dense: sum.dense?.count ?? null, mesh: !!sum.mesh };
  }

  it('a dense cloud that could not be stored stays in memory for exports; no older cloud or mesh comes back after a reload', async () => {
    const store = await meshedScan();
    const { e, errors } = await open(store);
    store.fail = (op, key) => op === 'put' && key === 'dense/points';
    const pc = await e.densify(DENSE_2);
    expect(pc.count).not.toBe(baseDense);
    expect(errors).toEqual(['storage: StorageQuotaError']);
    // the worker must not drop the only copy (it clones it instead)
    expect(e.releaseResultMemory('dense')).toBe(false);
    const ply = decodePLY(await (await e.exportModel('ply-points')).arrayBuffer());
    expect(ply.points?.count).toBe(pc.count);
    e.dispose();
    expect(await reload(store)).toEqual({ status: 'sparse', sparse: true, dense: null, mesh: false });
  }, 300_000);

  it('a mesh that could not be stored stays in memory for exports; the older mesh does not come back after a reload', async () => {
    const store = await meshedScan();
    const { e, errors } = await open(store);
    store.fail = (op, key) => op === 'put' && key === 'mesh';
    const mesh = await e.mesh();
    expect(errors).toEqual(['storage: StorageQuotaError']);
    expect(e.releaseResultMemory('mesh')).toBe(false);
    const ply = decodePLY(await (await e.exportModel('ply-mesh')).arrayBuffer());
    expect(ply.mesh?.triangleCount).toBe(mesh.triangleCount);
    e.dispose();
    expect(await reload(store)).toEqual({ status: 'dense', sparse: true, dense: baseDense, mesh: false });
  }, 300_000);

  it('the object box of the mesh comes back with it after a reload (mesh/meta) and goes with it', async () => {
    const store = await meshedScan();
    const meta = await store.getValue<MeshMetaRecord>('mesh/meta');
    expect(meta?.version).toBe(1);
    expect(typeof meta?.stats.objectRoi).toBe('string');
    console.log(`[persist] object box of the fixture mesh: ${meta?.stats.objectRoi}`);
    // (the fixture's 55° arc has none: its optical axes spread < 10°) a stored box comes back with the mesh
    const roi = { min: [-0.5, -0.5, 0.01], max: [0.5, 0.5, 0.6], planeZ: 0, centre: [0, 0, 0.2], confidence: 0.8 };
    await store.putValue('mesh/meta', { ...meta!, objectRoi: roi });
    const { e, sum } = await open(store);
    expect(sum.objectRoi).toEqual(roi);
    expect(e.objectRoi).toEqual(roi);
    // a damaged record reads as no box
    await store.putValue('mesh/meta', { version: 1, objectRoi: { min: [0, 0], max: [1, 1, 1] }, stats: {} });
    const damaged = await open(store);
    expect(damaged.sum.objectRoi).toBeNull();
    damaged.e.dispose();
    // a new dense cloud drops the mesh and its box
    await e.densify(DENSE_2);
    expect(e.objectRoi).toBeNull();
    expect(await store.getValue('mesh/meta')).toBeNull();
    e.dispose();
  }, 300_000);

  it('a kill right after a rebuilt dense cloud was stored does not pair it with the mesh of the old cloud', async () => {
    const store = await meshedScan();
    const { e } = await open(store);
    let killed = false;
    store.written = (key) => (killed ||= key === 'dense/points');
    store.fail = () => killed; // nothing is written after the kill
    const pc = await e.densify(DENSE_2);
    e.dispose();
    expect(await reload(store)).toEqual({ status: 'dense', sparse: true, dense: pc.count, mesh: false });
  }, 300_000);

  it('a kill right after a re-refined model was stored does not pair it with the old cameras, cloud or mesh', async () => {
    const store = await meshedScan();
    const { e } = await open(store);
    let killed = false;
    store.written = (key) => (killed ||= key === 'sparse/model');
    store.fail = () => killed;
    await e.refine();
    e.dispose();
    // (a model without its meta is not loaded: the scan is processed again)
    expect(await reload(store)).toEqual({ status: 'captured', sparse: false, dense: null, mesh: false });
  }, 300_000);
});

describe('session lifetime', () => {
  it('init with resume does not re-create a stored session whose record is gone (deleted); one kept in memory still resumes', async () => {
    const store = persistentStore('gone');
    const e = new PhotogrammetryEngine({ ...opts, sessionId: 'gone', resume: true }, { pipeline, store });
    await expect(e.init()).rejects.toMatchObject({ name: 'SessionNotFoundError' });
    expect(await store.getSessionMeta()).toBeNull();
    expect(await store.getValue('engine/state')).toBeNull();
    e.dispose();
    // no IndexedDB (a MemoryEngineStore fallback): nothing is stored to tell, the capture goes on
    const mem = new PhotogrammetryEngine({ ...opts, sessionId: 'mem', resume: true }, { pipeline, store: new MemoryEngineStore('mem') });
    await expect(mem.init()).resolves.toMatchObject({ gpu: false });
    mem.dispose();
  });

  it('a disposed engine stores nothing more: a keyframe still being matched in the background is dropped', async () => {
    // slow matching (CPU tier): a mapping backlog when the engine is disposed
    const slow: FeaturePipeline = {
      ...pipeline,
      match: async (...a) => {
        await new Promise((r) => setTimeout(r, 150));
        return pipeline.match(...a);
      },
    };
    const store = new FaultyStore('d');
    const e = new PhotogrammetryEngine({ ...opts, sessionId: 'd' }, { pipeline: slow, store });
    await e.init();
    for (const f of frameInputs(cap)) {
      const u = await e.processFrame(f);
      if (u.decision === 'keyframe' && u.numKeyframes >= 4) break;
    }
    const live = e.liveSession!;
    expect(live.pendingMapping).toBeGreaterThan(0);
    await new Promise((r) => setTimeout(r, 50)); // the keyframe's job is matching its pairs
    const late: string[] = [];
    e.dispose();
    store.fail = (op, key) => (late.push(`${op} ${key}`), false);
    await vi.waitFor(() => expect(live.pendingMapping).toBe(0), { timeout: 30_000 });
    expect(late).toEqual([]);
  }, 300_000);
});
