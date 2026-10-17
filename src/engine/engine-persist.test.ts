/**
 * iPhone-specific engine behaviour on a PERSISTENT store (the app's IndexedDB path, emulated by
 * a MemoryEngineStore flagged persistent): dense resume after a kill / cancel, the live session
 * released before the dense stage (and restored by a later refine), the capture state kept
 * intact across state writes without a live session, and store back-pressure during capture.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { FeaturePipeline } from '../features/extractor';
import type { ProgressEvent } from '../core/types';
import { readZip } from '../io/zip';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import { MemoryEngineStore, type EngineStateRecord, type EngineStore, type StoredImage } from './persistence';
import { frameInputs, makeCapture, type SynthCapture } from './testutil';
import type { KeyframeRecord } from './types';

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
    expect(await store.getValue('dense/progress')).toMatchObject({ pass: 1 });
    const pm1 = await store.listBinaryKeys('pm1/');
    expect(pm1.length).toBeGreaterThan(0);

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
