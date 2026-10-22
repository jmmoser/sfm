/**
 * CaptureRecorder (./recorder.ts) in Node against a scripted fake recorder worker: what it posts
 * per frame (the FrameInput verbatim + a COPY of the pixels), back-pressure, the storage floor,
 * stalls, worker errors and stop. The real worker / OPFS / IndexedDB / JPEG path:
 * recorder.gpu.test.ts.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { RGBAImage } from '../core/types';
import { frameMeta } from '../io/recording';
import { pumpFrame } from '../io/recording-testdata';
import type { LiveUpdate } from '../workers/protocol';
import { CaptureRecorder, cameraSummary, type CaptureRecorderOptions, type RecorderWorkerEvent, type RecorderWorkerRequest } from './recorder';

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
    rec.after(f, update, null, { ms: 31.26, waitMs: 0 });
    const live = worker.posted.at(-1)!.req as Extract<RecorderWorkerRequest, { type: 'live' }>;
    expect(live.row).toEqual({ seq: 0, index: 3, ms: 31.3, waitMs: 0, update: { frameIndex: 0, decision: 'keyframe', guidance: 'good', tracking: 'tracking', numKeyframes: 1, numRegistered: 0, numPoints: 0, timings: { total: 30 } }, error: null });
    rec.after(pumpFrame(9), null, new Error('x'), { ms: 1, waitMs: 0 }); // a frame it never saw
    expect(worker.posted.at(-1)!.req.type).toBe('live');
    expect((worker.posted.at(-1)!.req as Extract<RecorderWorkerRequest, { type: 'live' }>).row.seq).toBe(0);
  });

  it('back-pressure: beyond maxPending frames in the worker, before() waits for its acknowledgement', async () => {
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
