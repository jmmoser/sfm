/**
 * Continuing a stored capture after a page reload (EngineOptions.extend, "Add more views"): the
 * second page's frame / IMU / GPS times restart where the first page's did (performance.now()
 * restarts with every load), so without the session time base they would interleave with the
 * stored ones. With it the second segment lies after the stored capture, the reopened capture
 * registers into ONE model with the first segment (gravity aligned, no gyro break), the extension
 * loads only the sparse model (and previews it: InitResult.storedModel), a cancelled second refine
 * keeps the first model loadable, and the same run without the shift (EngineConfig.extendTimeShift:
 * false) fails the IMU-order check — the check detects the hazard. A scan without a readable
 * stored state is not continued (nothing written). With background mapping, frames are answered
 * while a slow rebuild of the stored live model runs (no back-pressure wait on it, at most
 * maxMappingBacklog keyframes behind it), model updates before the first frame carry no times, and
 * a page counting frames from 0 again gets frame indices above the stored ones. A worker respawn
 * in the same page whose first frame re-attaches a stored GPS fix continues unshifted.
 */
import { beforeAll, describe, expect, it, vi } from 'vitest';
import type { GeoFix } from '../core/types';
import type { FeaturePipeline } from '../features/extractor';
import { umeyama } from '../math/sim3';
import { LiveMapper } from '../sfm/live';
import type { FrameInput, LiveUpdate } from '../workers/protocol';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import { EXTEND_TIME_GAP_MS, frameTimeMin, segmentTimeShift, shiftFrameTimes, type CaptureState } from './live';
import { MemoryEngineStore, type EngineStateRecord } from './persistence';
import { centre, frameInputs, makeCapture, type SynthCapture } from './testutil';
import type { EngineConfig, SparseResultEx } from './types';

const DEG = Math.PI / 180;

let pipeline: FeaturePipeline;
let cap: SynthCapture;
beforeAll(() => {
  pipeline = createCpuFeaturePipeline({ maxFeatures: 2048, maxImageSize: 1024 });
  cap = makeCapture({ frames: 44, arcDeg: 55 });
}, 120_000);

/** MemoryEngineStore that reports `persistent` (the app's IndexedDB path: resume, release paths). */
function persistentStore(id: string): MemoryEngineStore {
  const s = new MemoryEngineStore(id);
  Object.defineProperty(s, 'persistent', { value: true });
  return s;
}

const opts = { mode: 'object' as const, quality: 'fast' as const, hfovPriorDeg: 60 };
/** First page: frames 0–24; the second page resumes at frame 27 (25, 26: the pause in between). */
const SPLIT = 25;
const RESUME_AT = 27;

/**
 * Frames `from`..`to` of the capture as one page delivers them: every time (frame, IMU, GPS fix)
 * moved by `dt` — a reload restarts performance.now(), so the second page's clock reads like the
 * first one's.
 */
function pageFrames(from: number, to: number, dt: number): FrameInput[] {
  return frameInputs(cap, { from, to }).map((f) => {
    const geo: GeoFix = { t: f.t + dt - 150, lat: 47.3769, lon: 8.5417, alt: 408, accuracy: 30, altitudeAccuracy: null };
    return { ...f, t: f.t + dt, imu: f.imu.map((s) => ({ ...s, t: s.t + dt })), geo };
  });
}

/** Second page's clock offset: its frame RESUME_AT is stamped like the first page's frame 0. */
const reloadDt = (): number => cap.frames[0].t - cap.frames[RESUME_AT].t;

function engineConfig(store: MemoryEngineStore, extra: EngineConfig = {}): EngineConfig {
  return { pipeline, store, mapping: 'inline', ...extra };
}

interface FirstPage {
  store: MemoryEngineStore;
  keyframes: number[];
  res?: SparseResultEx;
}

/** What the restored capture holds (the IMU after the last keyframe was never stored). */
interface Stored {
  imu: { first: number; last: number; count: number };
  /** Latest stored time (IMU, GPS, keyframes). */
  horizon: number;
}

function stored(st: CaptureState): Stored {
  const s = st.imu.samples;
  const kfLast = Math.max(...[...st.records.values()].map((r) => r.t));
  return { imu: { first: s[0].t, last: s[s.length - 1].t, count: s.length }, horizon: Math.max(s[s.length - 1].t, st.geo[st.geo.length - 1].t, kfLast) };
}

/** The first page: capture frames 0–24 (and optionally finish, refine, densify). */
async function firstPage(id: string, process: boolean): Promise<FirstPage> {
  const store = persistentStore(id);
  const a = new PhotogrammetryEngine({ ...opts, sessionId: id }, engineConfig(store));
  await a.init();
  for (const f of pageFrames(0, SPLIT, 0)) await a.processFrame(f);
  await a.flush();
  let keyframes = a.liveSession!.state.order.slice();
  let res: SparseResultEx | undefined;
  if (process) {
    await a.finishCapture();
    keyframes = a.liveSession!.state.order.slice();
    res = await a.refine();
    // (releases the capture state: the store has everything)
    const pc = await a.densify({ maxImageSize: 96, maxViews: 8 });
    expect(pc.count).toBeGreaterThan(100);
  }
  a.dispose();
  return { store, keyframes, res };
}

/**
 * The IMU-order check: the log is strictly increasing, exactly the stored samples lie in the stored
 * range [first, last] (no sample of the second page between them), and every delivered sample of
 * the second page was kept (none dropped as a duplicate of a stored time).
 */
function imuInOrder(st: CaptureState, old: Stored['imu'], added: number): boolean {
  const s = st.imu.samples;
  for (let i = 1; i < s.length; i++) if (!(s[i].t > s[i - 1].t)) return false;
  const inOld = s.filter((x) => x.t >= old.first && x.t <= old.last).length;
  return inOld === old.count && s.length === old.count + added;
}

/** Sim3 (result → GT) from camera centres: ATE (m), max gravity error (deg). */
function compare(res: SparseResultEx): { ate: number; gravityDeg: number } {
  const n = res.cameras.length;
  const src = new Float64Array(3 * n);
  const dst = new Float64Array(3 * n);
  let g = 0;
  res.cameras.forEach((c, i) => {
    const gt = cap.frames[c.frameIndex!].pose;
    src.set(centre(c.pose), 3 * i);
    dst.set(centre(gt), 3 * i);
    const d = c.pose.R[2] * gt.R[2] + c.pose.R[5] * gt.R[5] + c.pose.R[8] * gt.R[8];
    g = Math.max(g, Math.acos(Math.min(1, d)) / DEG);
  });
  const sim = umeyama(src, dst, n, true)!;
  let e = 0;
  for (let i = 0; i < n; i++) {
    for (let r = 0; r < 3; r++) {
      const p = sim.s * (sim.R[3 * r] * src[3 * i] + sim.R[3 * r + 1] * src[3 * i + 1] + sim.R[3 * r + 2] * src[3 * i + 2]) + sim.t[r];
      e += (p - dst[3 * i + r]) ** 2;
    }
  }
  return { ate: Math.sqrt(e / n), gravityDeg: g };
}

describe('session time base (pure)', () => {
  it('segmentTimeShift keeps a later clock and shifts an earlier one past the stored capture', () => {
    // same page, real pause ≥ the gap: unchanged
    expect(segmentTimeShift(80_000, 60_000, 0, EXTEND_TIME_GAP_MS)).toBe(0);
    // reloaded page (clock restarted): the segment starts EXTEND_TIME_GAP_MS after the stored capture
    expect(segmentTimeShift(1_000, 60_000, 0, EXTEND_TIME_GAP_MS)).toBe(60_000 + EXTEND_TIME_GAP_MS - 1_000);
    // a reopened capture shortly after the stored one (same page): pushed to the full gap
    expect(segmentTimeShift(62_000, 60_000, 0, EXTEND_TIME_GAP_MS)).toBe(EXTEND_TIME_GAP_MS - 2_000);
    // worker respawn in the same page (minimum gap 0): later times keep the current shift…
    expect(segmentTimeShift(62_000, 60_000, 5_000, 0)).toBe(5_000);
    // … earlier ones (another page) are shifted
    expect(segmentTimeShift(1_000, 60_000, 0, 0)).toBe(60_000 + EXTEND_TIME_GAP_MS - 1_000);
    // nothing stored yet
    expect(segmentTimeShift(1_000, -Infinity, 0, EXTEND_TIME_GAP_MS)).toBe(0);
  });

  it('shiftFrameTimes moves every time of a frame (new objects, the input untouched)', () => {
    const imu = [{ t: 90, gyro: [0, 0, 0] as [number, number, number], accel: null, accelG: null }];
    const f = {
      t: 100,
      imu,
      geo: { t: 40, lat: 1, lon: 2, alt: null, accuracy: 5, altitudeAccuracy: null },
      attitude: { t: 95, R_world_device: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), absolute: false, yawAccuracyDeg: null },
      screenAngleSince: -Infinity,
    };
    // (not the GPS fix: the FramePump attaches the nearest one, up to 30 s old)
    expect(frameTimeMin(f)).toBe(90);
    const g = shiftFrameTimes(f, 1000);
    expect([g.t, g.imu[0].t, g.geo!.t, g.attitude!.t, g.screenAngleSince]).toEqual([1100, 1090, 1040, 1095, -Infinity]);
    expect([f.t, imu[0].t, f.geo.t, f.attitude.t]).toEqual([100, 90, 40, 95]);
    expect(shiftFrameTimes(f, 0)).toBe(f);
  });
});

describe('engine: extend a stored capture after a page reload', () => {
  it('shifts the second page onto the session time base and registers both segments in one model', async () => {
    const p1 = await firstPage('extend', true);
    const res1 = p1.res!;
    expect(p1.keyframes.length).toBeGreaterThan(4);
    expect(res1.cameras.length).toBeGreaterThan(3);

    // second page: resume + extend on the same store
    const b = new PhotogrammetryEngine({ ...opts, sessionId: 'extend', resume: true, extend: true }, engineConfig(p1.store));
    const logs: string[] = [];
    b.on('log', (l, m) => logs.push(`${l}: ${m}`));
    const init = await b.init();
    expect(init.resumedKeyframes).toBe(p1.keyframes.length);
    expect(init.captureMemory!.usedMB).toBeGreaterThan(0);
    expect(init.captureMemory!.capMB).toBeGreaterThan(init.captureMemory!.usedMB);
    // the stored sparse model for the capture view (no camera times: they are on the session time base)
    expect(init.storedModel!.cameras).toHaveLength(res1.cameras.length);
    expect(init.storedModel!.cameras.every((c) => c.t === undefined)).toBe(true);
    expect(init.storedModel!.sparse.positions.length).toBe(3 * res1.points.count); // (< liveMaxPoints: not decimated)
    const live = b.liveSession!;
    const st = live.state;
    const old = stored(st);
    expect(st.captureFinished).toBe(false);
    expect(st.order).toEqual(p1.keyframes);
    // only the sparse model is in memory: the dense cloud stays in the store
    expect(b.sparseResult!.cameras.length).toBe(res1.cameras.length);
    expect(b.denseCloudResult).toBeNull();
    expect(b.memoryReport().parts.denseCloud).toBeUndefined();

    const frames2 = pageFrames(RESUME_AT, cap.frames.length, reloadDt());
    // the second page's clock reads like the first one's: without a shift its samples would land inside the stored range
    expect(frameTimeMin(frames2[0])).toBeLessThan(old.imu.last);
    const added = new Set(frames2.flatMap((f) => f.imu.map((s) => s.t))).size;
    for (const f of frames2) await b.processFrame(f);
    await b.flush();

    // IMU / GPS / keyframe times: the second segment starts EXTEND_TIME_GAP_MS after the stored capture
    expect(imuInOrder(st, old.imu, added)).toBe(true);
    const horizon = old.horizon;
    expect(st.imu.samples[old.imu.count].t).toBeGreaterThanOrEqual(horizon + EXTEND_TIME_GAP_MS - 1e-6);
    for (let i = 1; i < st.geo.length; i++) expect(st.geo[i].t).toBeGreaterThan(st.geo[i - 1].t);
    expect(st.geo.filter((g) => g.t > horizon).length).toBe(frames2.length);
    expect(st.order.length).toBeGreaterThan(p1.keyframes.length + 2);
    const newIds = st.order.slice(p1.keyframes.length);
    for (const id of newIds) expect(st.records.get(id)!.t).toBeGreaterThan(horizon + EXTEND_TIME_GAP_MS - 1);
    // dataset frame indices are above the stored counter: kept (GT lookup below)
    for (const id of newIds) expect(st.records.get(id)!.frameIndex).toBeGreaterThanOrEqual(RESUME_AT);
    expect(st.timeSegments).toHaveLength(1);
    expect(st.timeSegments[0].firstKeyframeId).toBe(newIds[0]);
    expect(st.timeSegments[0].shiftMs).toBeCloseTo(horizon + EXTEND_TIME_GAP_MS - frameTimeMin(frames2[0]), 6);
    expect(st.resultsStale).toBe(true);
    expect(logs.some((l) => /capture segment 2 from keyframe/.test(l))).toBe(true);
    // persisted: a later restore (another reload) continues on the same time base
    const rec = (await p1.store.getValue<EngineStateRecord>('engine/state'))!;
    expect(rec.timeSegments).toEqual(st.timeSegments);
    expect(rec.resultsStale).toBe(true);
    {
      // killed now: the scan opens as a capture to process, without decoding the old dense cloud
      const c = new PhotogrammetryEngine({ ...opts }, engineConfig(p1.store));
      const sum = await c.loadSession('extend');
      expect(sum.status).toBe('capturing');
      expect(sum.dense).toBeNull();
      expect(sum.sparse!.cameras.length).toBe(res1.cameras.length);
      c.dispose();
    }

    await b.finishCapture();
    // a cancelled second refine keeps the first model (and opens the scan as a capture to process)
    await expect(b.refine(undefined, { cancelled: true })).rejects.toMatchObject({ name: 'CancelledError' });
    {
      const c = new PhotogrammetryEngine({ ...opts }, engineConfig(p1.store));
      const sum = await c.loadSession('extend');
      expect(sum.sparse!.cameras.length).toBe(res1.cameras.length);
      expect(sum.status).toBe('captured');
      // the dense cloud of the first model is not read for a capture still to be processed
      expect(sum.dense).toBeNull();
      c.dispose();
    }

    const res = await b.refine();
    const n = st.order.length;
    expect(res.stats.images).toBe(n);
    // ≥ 95 % of all keyframes in ONE model
    expect(res.stats.registeredImages as number).toBeGreaterThanOrEqual(Math.ceil(0.95 * n));
    expect(res.stats.unconnectedImages).toBe(n - (res.stats.registeredImages as number));
    expect(parseInt(String(res.stats.gyroBreaksLive ?? '0'), 10)).toBe(0);
    expect(res.metric.gravityAligned).toBe(true);
    const cmp = compare(res);
    expect(cmp.gravityDeg).toBeLessThan(1);
    expect(cmp.ate).toBeLessThan(0.01); // m, on a 1.4 m orbit
    // the refine replaced the first model's later stages
    expect(await p1.store.getBinary('dense/points')).toBeNull();
    expect(st.resultsStale).toBe(false);
    b.dispose();
    {
      const c = new PhotogrammetryEngine({ ...opts }, engineConfig(p1.store));
      const sum = await c.loadSession('extend');
      expect(sum.status).toBe('sparse');
      expect(sum.sparse!.cameras.length).toBe(res.cameras.length);
      c.dispose();
    }
  }, 600_000);

  it('without the shift the second page interleaves with the stored IMU (the check detects it)', async () => {
    const p1 = await firstPage('noshift', false);
    const b = new PhotogrammetryEngine({ ...opts, sessionId: 'noshift', resume: true, extend: true }, engineConfig(p1.store, { extendTimeShift: false }));
    await b.init();
    const old = stored(b.liveSession!.state);
    const frames2 = pageFrames(RESUME_AT, RESUME_AT + 4, reloadDt());
    const added = new Set(frames2.flatMap((f) => f.imu.map((s) => s.t))).size;
    for (const f of frames2) await b.processFrame(f);
    expect(imuInOrder(b.liveSession!.state, old.imu, added)).toBe(false);
    b.dispose();
  }, 300_000);
});

describe('engine: extend — a stored state that cannot be read is not started over', () => {
  async function storeWithMeta(id: string): Promise<MemoryEngineStore> {
    const store = persistentStore(id);
    await store.ensureSession({ mode: 'object', quality: 'fast' });
    return store;
  }

  it('rejects without a stored state, and writes nothing', async () => {
    const store = await storeWithMeta('nostate');
    const b = new PhotogrammetryEngine({ ...opts, sessionId: 'nostate', resume: true, extend: true }, engineConfig(store));
    await expect(b.init()).rejects.toThrow(/no stored capture to continue/);
    expect(await store.getValue('engine/state')).toBeNull();
    b.dispose();
  });

  it('rejects on a read error of the stored state, and keeps the stored record', async () => {
    const store = await storeWithMeta('unreadable');
    const rec = { version: 1, mode: 'object', quality: 'fast', thetaSession: 0, cameraTheta: 0, nextKeyframeId: 7, frameCounter: 40, imuChunks: 3, captureFinished: true, updatedAt: 1 };
    await store.putValue('engine/state', rec);
    const get = store.getValue.bind(store);
    let failing = true;
    store.getValue = <T>(key: string): Promise<T | null> => (failing && key === 'engine/state' ? Promise.reject(new Error('IDB read failed')) : get<T>(key));
    const b = new PhotogrammetryEngine({ ...opts, sessionId: 'unreadable', resume: true, extend: true }, engineConfig(store));
    await expect(b.init()).rejects.toThrow(/could not be read: IDB read failed/);
    await b.flush().catch(() => undefined);
    failing = false;
    expect(await store.getValue('engine/state')).toEqual(rec);
    b.dispose();
  });
});

describe('engine: extend with background mapping', () => {
  it('frames are answered while a slow rebuild of the stored model runs; model times wait for the first frame; frame indices restarting at 0 get an offset', async () => {
    const id = 'rebuild';
    const p1 = await firstPage(id, false);
    const storedCounter = (await p1.store.getValue<EngineStateRecord>('engine/state'))!.frameCounter;
    const cfg = engineConfig(p1.store, { mapping: 'background' });

    {
      // the rebuild ends before the first frame (a small scan, a camera that starts late): its model
      // update carries no keyframe times — the previous page's would lie ahead of this page's clock
      const b = new PhotogrammetryEngine({ ...opts, sessionId: id, resume: true, extend: true }, cfg);
      const live: LiveUpdate[] = [];
      b.on('live', (u) => live.push(u));
      await b.init();
      await b.flush();
      const u = live.filter((x) => x.cameras?.length).pop();
      expect(u).toBeDefined();
      expect(u!.cameras!.length).toBeGreaterThan(3);
      expect(u!.cameras!.every((c) => c.t === undefined)).toBe(true);
      b.dispose();
    }

    // a slow rebuild (≈ 400 ms per keyframe — a large scan on a phone takes a minute)
    let slow = p1.keyframes.length;
    const add = LiveMapper.prototype.addKeyframe;
    const spy = vi.spyOn(LiveMapper.prototype, 'addKeyframe').mockImplementation(function (this: LiveMapper, kf: number) {
      if (slow > 0) {
        slow--;
        const end = performance.now() + 400;
        while (performance.now() < end) {
          /* busy */
        }
      }
      return add.call(this, kf);
    });
    try {
      // (one keyframe job may queue behind the rebuild: the second keyframe decision meets the cap)
      const b = new PhotogrammetryEngine({ ...opts, sessionId: id, resume: true, extend: true }, { ...cfg, maxMappingBacklog: 1 });
      const live: LiveUpdate[] = [];
      b.on('live', (u) => live.push(u));
      const tInit = performance.now();
      await b.init();
      const ls = b.liveSession!;
      const st = ls.state;
      expect(ls.rebuildProgress).not.toBeNull();
      // a reloaded page: a new FramePump counts frames from 0 again
      const frames2 = pageFrames(RESUME_AT, cap.frames.length, reloadDt()).map((f, i) => ({ ...f, index: i }));
      const answered: { u: LiveUpdate; started: boolean; rebuilding: boolean; jobs: number; ms: number }[] = [];
      let rebuildMs = NaN;
      for (const f of frames2) {
        // each frame arrives as a worker message (a macrotask: the rebuild's steps run in between)
        await new Promise((r) => setTimeout(r, 0));
        const started = ls.rebuildProgress !== null;
        const t0 = performance.now();
        const u = await b.processFrame(f);
        answered.push({ u, started, rebuilding: ls.rebuildProgress !== null, jobs: ls.pendingKeyframeJobs, ms: performance.now() - t0 });
        if (Number.isNaN(rebuildMs) && ls.rebuildProgress === null) rebuildMs = performance.now() - tInit;
      }
      await b.flush();
      if (Number.isNaN(rebuildMs)) rebuildMs = performance.now() - tInit;
      expect(rebuildMs).toBeGreaterThan(400 * p1.keyframes.length);

      // answered meanwhile, with its progress; never waiting for it; at most maxMappingBacklog
      // keyframes behind it — a later keyframe decision is dropped (tracking goes on)
      const during = answered.filter((a) => a.u.modelRebuild);
      expect(during.length).toBeGreaterThanOrEqual(p1.keyframes.length - 2);
      // a frame that came during the rebuild is answered during it (its steps run between frames)
      for (const a of answered) if (a.started) expect(a.u.modelRebuild).toBeDefined();
      expect(Math.max(...answered.map((a) => a.ms))).toBeLessThan(rebuildMs / 2);
      for (const a of during) {
        expect(a.u.modelRebuild!.total).toBe(p1.keyframes.length);
        expect(a.u.timings.mappingWait).toBeUndefined();
      }
      for (const a of answered.filter((x) => x.rebuilding)) expect(a.jobs).toBeLessThanOrEqual(1);
      expect(during.filter((a) => a.u.decision === 'keyframe').length).toBe(1);
      expect(during.some((a) => a.u.timings.rebuildLimited === 1)).toBe(true);
      const maxWait = Math.max(0, ...answered.map((a) => a.u.timings.mappingWait ?? 0));
      expect(maxWait).toBeLessThan(rebuildMs / 2);

      // the new keyframes join the rebuilt live model
      const newIds = st.order.slice(p1.keyframes.length);
      expect(newIds.length).toBeGreaterThan(2);
      const rec = ls.mapper.reconstruction;
      expect(newIds.filter((k) => rec.images.get(k)?.registered).length).toBeGreaterThanOrEqual(newIds.length - 1);

      // frame indices: above the stored counter, increasing, and they map back to the page's frames
      const shift = st.timeSegments[0].shiftMs;
      let prev = -1;
      for (const k of newIds) {
        const r = st.records.get(k)!;
        expect(r.frameIndex).toBeGreaterThanOrEqual(storedCounter);
        expect(r.frameIndex).toBeGreaterThan(prev);
        prev = r.frameIndex;
        expect(r.t).toBeCloseTo(frames2[r.frameIndex - storedCounter].t + shift, 6);
      }
      for (const k of p1.keyframes) expect(st.records.get(k)!.frameIndex).toBeLessThan(storedCounter);

      // model updates after the first frame carry the page's times: the stored keyframes before it
      const last = live.filter((x) => x.cameras?.length).pop()!;
      expect(last.cameras!.every((c) => c.t !== undefined)).toBe(true);
      const off = st.timeOffsetMs ?? 0;
      expect(Math.max(...last.cameras!.slice(0, p1.keyframes.length).map((c) => c.t!))).toBeLessThan(frames2[0].t + off);
      b.dispose();
    } finally {
      spy.mockRestore();
    }
  }, 600_000);
});

describe('engine: worker respawn in the same page', () => {
  it('a first frame re-attaching a stored GPS fix neither shifts the capture nor stores the fix twice', async () => {
    const id = 'respawn-geo';
    const p1 = await firstPage(id, false);
    const b = new PhotogrammetryEngine({ ...opts, sessionId: id, resume: true }, engineConfig(p1.store));
    await b.init();
    const st = b.liveSession!.state;
    const stored0 = st.geo.length;
    const lastFix = st.geo[stored0 - 1];
    // same page clock; fresh fixes at the frame times, the first frame carries the stored nearest one
    const frames2 = frameInputs(cap, { from: SPLIT, to: SPLIT + 10 }).map((f): FrameInput => ({ ...f, geo: { t: f.t, lat: 47.3769, lon: 8.5417, alt: 408, accuracy: 30, altitudeAccuracy: null } }));
    frames2[0] = { ...frames2[0], geo: { ...lastFix } };
    for (const f of frames2) await b.processFrame(f);
    await b.flush();
    expect(st.timeSegments).toEqual([]);
    expect(st.timeShiftMs).toBe(0);
    for (let i = 1; i < st.geo.length; i++) expect(st.geo[i].t).toBeGreaterThan(st.geo[i - 1].t);
    expect(st.geo.length).toBe(stored0 + frames2.length - 1);
    // the capture continues on the same time base
    const newIds = st.order.slice(p1.keyframes.length);
    expect(newIds.length).toBeGreaterThan(0);
    const ts = new Set(frames2.map((f) => f.t));
    for (const k of newIds) expect(ts.has(st.records.get(k)!.t)).toBe(true);
    b.dispose();
  }, 300_000);
});
