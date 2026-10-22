/**
 * CaptureRecorder (./recorder.ts) in Node against a scripted fake recorder worker: what it posts
 * per frame (the FrameInput verbatim + a COPY of the pixels), not-sent frames, back-pressure, the
 * storage floor, stalls (and the hidden page), worker errors, the sensor log and stop. The real
 * worker / OPFS / IndexedDB / JPEG path: recorder.gpu.test.ts.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { GeoFix, RGBAImage } from '../core/types';
import { frameMeta, type RecordedSensorRow } from '../io/recording';
import { pumpFrame } from '../io/recording-testdata';
import type { LiveUpdate } from '../workers/protocol';
import { CaptureRecorder, cameraSummary, type CaptureRecorderOptions, type RecorderWorkerEvent, type RecorderWorkerRequest, type VisibilitySource } from './recorder';

class FakeRecorderWorker {
  readonly posted: { req: RecorderWorkerRequest; transfer: Transferable[] }[] = [];
  onmessage: ((ev: { data: RecorderWorkerEvent }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  terminated = false;
  openReply: RecorderWorkerEvent = { type: 'opened', sink: 'opfs' };
  postMessage(req: RecorderWorkerRequest, transfer: Transferable[] = []): void {
    this.posted.push({ req, transfer });
    if (req.type === 'open') queueMicrotask(() => this.emit(this.openReply));
    if (req.type === 'close') queueMicrotask(() => this.emit({ type: 'closed', stats: { frames: this.frames().length, imageBytes: 123, lineBytes: 10, failedImages: 0 } }));
  }
  terminate(): void {
    this.terminated = true;
  }
  emit(ev: RecorderWorkerEvent): void {
    this.onmessage?.({ data: ev });
  }
  frames(): Extract<RecorderWorkerRequest, { type: 'frame' }>[] {
    return this.posted.map((p) => p.req).filter((r): r is Extract<RecorderWorkerRequest, { type: 'frame' }> => r.type === 'frame');
  }
  ack(seq: number, bytes = 100): void {
    this.emit({ type: 'frame', seq, ok: true, bytes });
  }
}

/** A document's visibility, switchable. */
class FakeVisibility implements VisibilitySource {
  hidden = false;
  private cbs = new Set<() => void>();
  get listeners(): number {
    return this.cbs.size;
  }
  addEventListener(_type: 'visibilitychange', cb: () => void): void {
    this.cbs.add(cb);
  }
  removeEventListener(_type: 'visibilitychange', cb: () => void): void {
    this.cbs.delete(cb);
  }
  set(hidden: boolean): void {
    this.hidden = hidden;
    for (const cb of [...this.cbs]) cb();
  }
}

const update: LiveUpdate = { frameIndex: 0, decision: 'keyframe', guidance: 'good', tracking: 'tracking', numKeyframes: 1, numRegistered: 0, numPoints: 0, timings: { total: 30 }, overlay: new Float32Array(10) };

let worker: FakeRecorderWorker;
const stops: [string, boolean][] = [];
async function start(opts: CaptureRecorderOptions = {}): Promise<CaptureRecorder> {
  return CaptureRecorder.start(
    { name: 'Object · Oct 21, 14:02', sessionId: 's1', engine: { mode: 'object', quality: 'balanced', hfovPriorDeg: 67.3 }, camera: { label: 'Back Camera', deviceId: 'secret', settings: { deviceId: 'secret', groupId: 'g', width: 1920 } }, pumpStats: () => ({ skippedBusy: 4 }) },
    { createWorker: () => (worker = new FakeRecorderWorker()) as unknown as Worker, available: async () => 5e9, onStop: (r, a) => void stops.push([r, a]), ...opts },
  );
}

const flush = async (): Promise<void> => {
  for (let i = 0; i < 20; i++) await Promise.resolve();
};

beforeEach(() => {
  vi.stubGlobal('Worker', function Worker() {});
  vi.stubGlobal('OffscreenCanvas', function OffscreenCanvas() {});
  vi.stubGlobal('indexedDB', {});
  stops.length = 0;
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('CaptureRecorder', () => {
  it('opens the worker with the header (no camera device ids) and needs the storage floor to start', async () => {
    const rec = await start();
    const open = worker.posted[0].req as Extract<RecorderWorkerRequest, { type: 'open' }>;
    expect(open.header).toMatchObject({ format: 'sfm-raw-capture', name: 'Object · Oct 21, 14:02', sessionId: 's1', engine: { mode: 'object', quality: 'balanced', hfovPriorDeg: 67.3 }, state: 'recording' });
    expect(JSON.stringify(open.header)).not.toContain('secret');
    expect(open.header.camera).toEqual({ label: 'Back Camera', settings: { width: 1920 } });
    expect(rec.sink).toBe('opfs');
    await expect(start({ available: async () => 400e6 })).rejects.toThrow(/400 MB of storage are free/);
    expect(cameraSummary(null)).toBeNull();
  });

  it('a worker that cannot open rejects start and is terminated', async () => {
    const created: FakeRecorderWorker[] = [];
    await expect(
      CaptureRecorder.start(
        { name: 'x', sessionId: null, engine: { mode: 'scene', quality: 'fast' } },
        {
          available: async () => null,
          createWorker: () => {
            const w = new FakeRecorderWorker();
            w.openReply = { type: 'error', fatal: true, name: 'Error', message: 'OffscreenCanvas JPEG encoding is not available in workers' };
            created.push(w);
            return w as unknown as Worker;
          },
        },
      ),
    ).rejects.toThrow(/OffscreenCanvas/);
    expect(created[0].terminated).toBe(true);
  });

  it('posts every FrameInput field verbatim with a copy of the pixels, and the engine answer per frame', async () => {
    const rec = await start();
    const f = pumpFrame(3);
    expect(rec.before(f)).toBeUndefined(); // room in the worker: no wait
    const [m] = worker.frames();
    expect(m.msg.input).toStrictEqual(frameMeta(f));
    expect(m.msg).toMatchObject({ seq: 0, kind: 'RGBAImage', srcWidth: 64, srcHeight: 48 });
    const img = m.image as RGBAImage;
    expect(img.data).not.toBe((f.bitmap as RGBAImage).data);
    expect(Array.from(img.data.slice(0, 8))).toEqual(Array.from((f.bitmap as RGBAImage).data.slice(0, 8)));
    expect(worker.posted.find((p) => p.req.type === 'frame')!.transfer).toEqual([img.data.buffer]);
    // maxSide 1280 of a 64 × 48 frame: recorded at its size
    expect([m.width, m.height]).toEqual([64, 48]);
    rec.after(f, update, null, { ms: 31.26, waitMs: 0, sent: true });
    const live = worker.posted.at(-1)!.req as Extract<RecorderWorkerRequest, { type: 'live' }>;
    expect(live.row).toEqual({ seq: 0, index: 3, ms: 31.3, waitMs: 0, update: { frameIndex: 0, decision: 'keyframe', guidance: 'good', tracking: 'tracking', numKeyframes: 1, numRegistered: 0, numPoints: 0, timings: { total: 30 } }, error: null });
    rec.after(pumpFrame(9), null, new Error('x'), { ms: 1, waitMs: 0, sent: false }); // a frame it never saw
    expect(worker.posted.at(-1)!.req.type).toBe('live');
    expect((worker.posted.at(-1)!.req as Extract<RecorderWorkerRequest, { type: 'live' }>).row.seq).toBe(0);
  });

  it('a frame the engine client rejected before posting it is flagged notSent in live.jsonl', async () => {
    const rec = await start();
    const f = pumpFrame(0);
    rec.before(f);
    rec.after(f, null, Object.assign(new Error('Engine disposed'), { name: 'EngineCrashedError' }), { ms: 0.1, waitMs: 0, sent: false });
    const live = worker.posted.at(-1)!.req as Extract<RecorderWorkerRequest, { type: 'live' }>;
    expect(live.row).toMatchObject({ seq: 0, update: null, error: 'EngineCrashedError: Engine disposed', notSent: true });
    // a frame posted that failed in the engine is not
    const g = pumpFrame(1);
    rec.before(g);
    rec.after(g, null, new Error('engine crashed'), { ms: 5, waitMs: 0, sent: true });
    expect((worker.posted.at(-1)!.req as Extract<RecorderWorkerRequest, { type: 'live' }>).row).not.toHaveProperty('notSent');
  });

  it('back-pressure: the frame that makes more than maxPending in the worker waits for an acknowledgement (≤ maxPending + 1 in it)', async () => {
    const rec = await start({ maxPending: 2 });
    expect(rec.before(pumpFrame(0))).toBeUndefined();
    expect(rec.before(pumpFrame(1))).toBeUndefined();
    let released = false;
    const hold = rec.before(pumpFrame(2));
    expect(hold).toBeInstanceOf(Promise);
    void (hold as Promise<void>).then(() => (released = true));
    await flush();
    expect(released).toBe(false);
    worker.ack(0);
    await flush();
    expect(released).toBe(true);
    expect(rec.pending).toBe(2);
    expect(rec.stats).toMatchObject({ sent: 3, acked: 1, stored: 1, imageBytes: 100 });
  });

  it('stops by itself when the free storage drops below the floor (the scan goes on)', async () => {
    let free = 5e9;
    const rec = await start({ available: async () => free, checkEveryMs: 0 });
    rec.before(pumpFrame(0));
    await flush(); // that check found enough room
    expect(rec.state).toBe('recording');
    free = 0.5e9;
    rec.before(pumpFrame(1));
    await flush();
    await vi.waitFor(() => expect(rec.state).toBe('stopped'));
    expect(stops).toEqual([[expect.stringMatching(/storage is low \(500 MB free\)/), true]]);
    const close = worker.posted.find((p) => p.req.type === 'close')!.req as Extract<RecorderWorkerRequest, { type: 'close' }>;
    expect(close.header).toMatchObject({ state: 'complete', stopReason: expect.stringMatching(/storage is low/), summary: { frames: 2, pump: { skippedBusy: 4 } } });
    expect(worker.terminated).toBe(true);
    const n = worker.posted.length;
    expect(rec.before(pumpFrame(2))).toBeUndefined(); // stopped: frames pass untapped
    expect(worker.posted.length).toBe(n);
  });

  it('a worker that falls behind for stallMs stops the recording and releases the held frame', async () => {
    const rec = await start({ maxPending: 0, stallMs: 30 });
    const hold = rec.before(pumpFrame(0)) as Promise<void>;
    await hold; // released by the stop, not by an ack
    await vi.waitFor(() => expect(rec.state).toBe('stopped'));
    expect(stops[0]).toEqual([expect.stringMatching(/fell behind by 1 frames for 0.0 s/), true]);
  });

  it('the stall clock does not run while the page is hidden and restarts on return', async () => {
    const page = new FakeVisibility();
    const rec = await start({ maxPending: 0, stallMs: 40, visibility: page });
    const hold = rec.before(pumpFrame(0)) as Promise<void>;
    page.set(true); // the user switches apps: iOS suspends the page and the worker
    await new Promise((r) => setTimeout(r, 120));
    expect(rec.state).toBe('recording');
    page.set(false);
    await new Promise((r) => setTimeout(r, 15)); // a fresh 40 ms from the return
    expect(rec.state).toBe('recording');
    worker.ack(0); // the worker caught up
    await hold;
    expect(rec.state).toBe('recording');
    // still behind after a full stallMs of visible time: stops
    const hold2 = rec.before(pumpFrame(1)) as Promise<void>;
    await hold2;
    await vi.waitFor(() => expect(rec.state).toBe('stopped'));
    expect(stops[0]).toEqual([expect.stringMatching(/fell behind/), true]);
    expect(page.listeners).toBe(0); // removed by the stop
  });

  it('sensor log: raw motion / orientation events and every GPS fix, batched; listeners removed at stop', async () => {
    const target = new EventTarget();
    const fixes = new Set<(f: GeoFix) => void>();
    const geo = { onFix: (cb: (f: GeoFix) => void) => (fixes.add(cb), () => void fixes.delete(cb)) };
    const rec = await CaptureRecorder.start(
      { name: 'x', sessionId: null, engine: { mode: 'object', quality: 'fast' }, geo },
      { createWorker: () => (worker = new FakeRecorderWorker()) as unknown as Worker, available: async () => null, sensorTarget: target, sensorFlushMs: 10, visibility: null },
    );
    const motion = Object.assign(new Event('devicemotion'), { interval: 16, rotationRate: { alpha: 1.5, beta: -0, gamma: null }, acceleration: null, accelerationIncludingGravity: { x: 0.1, y: -9.8, z: 0.2 } });
    target.dispatchEvent(motion);
    target.dispatchEvent(Object.assign(new Event('deviceorientation'), { alpha: 10, beta: 80, gamma: -2, absolute: false, webkitCompassHeading: 271.5, webkitCompassAccuracy: 15 }));
    target.dispatchEvent(Object.assign(new Event('deviceorientationabsolute'), { alpha: 11, beta: 80, gamma: -2, absolute: true }));
    const fix: GeoFix = { t: 900, lat: 47.6, lon: -122.3, alt: 50, accuracy: 5, altitudeAccuracy: 3 };
    for (const cb of fixes) cb(fix);
    const batches = (): RecordedSensorRow[][] => worker.posted.map((p) => p.req).filter((r): r is Extract<RecorderWorkerRequest, { type: 'sensors' }> => r.type === 'sensors').map((r) => r.rows);
    await vi.waitFor(() => expect(batches().flat()).toHaveLength(4));
    const [m, o, a, g] = batches().flat();
    expect(m).toMatchObject({ k: 'motion', interval: 16, rot: [1.5, -0, null], acc: null, accG: [0.1, -9.8, 0.2], ts: motion.timeStamp });
    expect(o).toMatchObject({ k: 'orientation', absolute: false, alpha: 10, beta: 80, gamma: -2, absoluteFlag: false, compassHeading: 271.5, compassAccuracy: 15 });
    expect(a).toMatchObject({ k: 'orientation', absolute: true, alpha: 11, absoluteFlag: true, compassHeading: null });
    expect(g).toEqual({ k: 'geo', t: expect.any(Number), fix });
    // stop: the rest flushed before the close, then nothing is listened to any more
    target.dispatchEvent(motion);
    const h = await rec.stop();
    const types = worker.posted.map((p) => p.req.type);
    expect(types.lastIndexOf('sensors')).toBeLessThan(types.indexOf('close'));
    expect(h?.summary).toMatchObject({ sensorRows: 5, sensorDropped: 0 });
    expect(fixes.size).toBe(0);
    const n = worker.posted.length;
    target.dispatchEvent(motion);
    await new Promise((r) => setTimeout(r, 30));
    expect(worker.posted.length).toBe(n);
  });

  it('a fatal write error (quota) stops the recording', async () => {
    const rec = await start();
    rec.before(pumpFrame(0));
    worker.emit({ type: 'error', fatal: true, name: 'QuotaExceededError', message: 'The quota has been exceeded.' });
    await vi.waitFor(() => expect(rec.state).toBe('stopped'));
    expect(stops).toEqual([['storage is full', true]]);
  });

  it('stop() writes the summary (the worker\'s counts) and is idempotent', async () => {
    const rec = await start();
    rec.before(pumpFrame(0));
    rec.before(pumpFrame(1));
    worker.ack(0, 50);
    const [a, b] = await Promise.all([rec.stop(), rec.stop()]);
    expect(a).toBe(b);
    expect(a).toMatchObject({ state: 'complete', stopReason: 'capture ended', summary: { frames: 2, imageBytes: 123, failedImages: 0, pump: { skippedBusy: 4 } } });
    expect(stops).toEqual([['capture ended', false]]);
    expect(worker.posted.filter((p) => p.req.type === 'close')).toHaveLength(1);
  });
});
