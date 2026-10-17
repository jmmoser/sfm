import { describe, expect, it } from 'vitest';
import type { CameraStartResult, CameraState, FrameCallback, FrameMeta } from './camera';
import { FrameClock, MstpFrameSource, type MstpCamera, type WorkerLike } from './mstp-camera';
import type { CameraWorkerEvent, CameraWorkerRequest } from './mstp-protocol';

class FakeTrack {
  readyState: MediaStreamTrackState = 'live';
  clones: FakeTrack[] = [];
  stopped = false;
  clone(): FakeTrack {
    const c = new FakeTrack();
    this.clones.push(c);
    return c;
  }
  stop(): void {
    this.stopped = true;
    this.readyState = 'ended';
  }
}

class FakeCamera {
  track: FakeTrack | null = new FakeTrack();
  segment = 0;
  state: CameraState = 'running';
  result = { frameMaxSide: 1280 } as CameraStartResult;
  frameCbs = new Set<FrameCallback>();
  stateCbs = new Set<(s: CameraState, d?: string) => void>();
  grabs = 0;
  onFrame(cb: FrameCallback): () => void {
    this.frameCbs.add(cb);
    return () => this.frameCbs.delete(cb);
  }
  grabFrame(): Promise<ImageBitmap> {
    this.grabs++;
    return Promise.resolve({ width: 1440, height: 1920 } as unknown as ImageBitmap);
  }
  onStateChange(cb: (s: CameraState, d?: string) => void): () => void {
    this.stateCbs.add(cb);
    return () => this.stateCbs.delete(cb);
  }
  setState(s: CameraState): void {
    this.state = s;
    for (const cb of this.stateCbs) cb(s);
  }
  emit(t: number): void {
    const m: FrameMeta = { mediaTime: 0, width: 1440, height: 1920, callbackTime: t, timeSource: 'captureTime', segment: this.segment };
    for (const cb of this.frameCbs) cb(t, m);
  }
}

class FakeWorker implements WorkerLike {
  sent: { msg: CameraWorkerRequest; transfer?: Transferable[] }[] = [];
  onmessage: ((e: MessageEvent<CameraWorkerEvent>) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  terminated = false;
  throwOnAttach = false;
  postMessage(msg: CameraWorkerRequest, transfer?: Transferable[]): void {
    if (msg.type === 'attach' && this.throwOnAttach) throw new DOMException('not transferable', 'DataCloneError');
    this.sent.push({ msg, transfer });
  }
  terminate(): void {
    this.terminated = true;
  }
  emit(ev: CameraWorkerEvent): void {
    this.onmessage?.({ data: ev } as MessageEvent<CameraWorkerEvent>);
  }
  lastAttach(): Extract<CameraWorkerRequest, { type: 'attach' }> {
    const a = this.sent.filter((s) => s.msg.type === 'attach').at(-1);
    return a!.msg as Extract<CameraWorkerRequest, { type: 'attach' }>;
  }
}

function frame(gen: number, seq: number, timestampUs: number, receivedMs: number): CameraWorkerEvent {
  return { type: 'frame', gen, seq, timestampUs, receivedMs, width: 1920, height: 1440, codedWidth: 1920, codedHeight: 1440, format: 'NV12' };
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function make(opts: { throwOnAttach?: boolean; firstFrameTimeoutMs?: number; attachTimeoutMs?: number } = {}) {
  const cam = new FakeCamera();
  const worker = new FakeWorker();
  worker.throwOnAttach = !!opts.throwOnAttach;
  let now = 0;
  const src = new MstpFrameSource(cam as unknown as MstpCamera, {
    createWorker: () => worker,
    timeOrigin: 1000, // worker timeOrigin 1010 ⇒ worker times are 10 ms later on the main timeline
    now: () => now,
    attachTimeoutMs: opts.attachTimeoutMs ?? 200,
    firstFrameTimeoutMs: opts.firstFrameTimeoutMs ?? 200,
    grabTimeoutMs: 100,
  });
  return { cam, worker, src, setNow: (t: number) => (now = t) };
}

describe('MstpFrameSource', () => {
  it('transfers a clone, dispatches worker frames with the capture timestamps and grabs exactly the dispatched frame', async () => {
    const { cam, worker, src, setNow } = make();
    const started = src.start();
    const att = worker.lastAttach();
    expect(att.gen).toBeGreaterThan(0);
    expect(att.track).toBe(cam.track!.clones[0]);
    expect(worker.sent[0].transfer).toEqual([cam.track!.clones[0]]);
    worker.emit({ type: 'attached', gen: att.gen, timeOrigin: 1010 });
    worker.emit(frame(att.gen, 1, 50_000_000, 100)); // capture clock 50 000 ms, arrival 110 ms (main)
    expect(await started).toBe('mstp');
    expect(src.mode).toBe('mstp');
    expect(src.fixedOrientation).toBe(true);

    const seen: { t: number; meta: FrameMeta; grab: Promise<ImageBitmap> }[] = [];
    src.onFrame((t, meta) => seen.push({ t, meta: { ...meta }, grab: src.grabFrame({ maxSide: 1280 }) }));
    setNow(150);
    // 33 ms later in capture time, delivered 5 ms faster than the first frame ⇒ new minimum offset
    worker.emit(frame(att.gen, 2, 50_033_000, 128));
    expect(seen).toHaveLength(1);
    expect(seen[0].t).toBeCloseTo(50_033 + (138 - 50_033), 6); // = arrival of the fastest frame
    expect(seen[0].meta).toMatchObject({ timeSource: 'frameTimestamp', width: 1920, height: 1440, callbackTime: 150, segment: 0 });
    const grab = worker.sent.at(-1)!.msg;
    expect(grab).toMatchObject({ type: 'grab', gen: att.gen, seq: 2, maxSide: 1280 });
    const bmp = { width: 1280, height: 960, close() {} } as unknown as ImageBitmap;
    worker.emit({ type: 'grabbed', id: (grab as { id: number }).id, bitmap: bmp });
    expect(await seen[0].grab).toBe(bmp);

    // a later frame arriving with more latency is mapped with the minimum offset
    worker.emit(frame(att.gen, 3, 50_066_000, 175));
    expect(seen[1].t).toBeCloseTo(50_066 + (138 - 50_033), 6);
    worker.emit({ type: 'grabbed', id: (worker.sent.at(-1)!.msg as { id: number }).id, error: 'stale' });
    await expect(seen[1].grab).rejects.toThrow(/stale/);
    expect(src.stats).toMatchObject({ frames: 3, dispatched: 2, grabs: 2, staleGrabs: 1, codedWidth: 1920, format: 'NV12' });

    src.stop();
    expect(worker.terminated).toBe(true);
    expect(worker.sent.some((s) => s.msg.type === 'detach')).toBe(true);
  });

  it('falls back to the camera before the first frame and forwards its frames from then on', async () => {
    const { cam, worker, src } = make();
    const ts: number[] = [];
    src.onFrame((t) => ts.push(t));
    const started = src.start();
    const att = worker.lastAttach();
    worker.emit({ type: 'error', gen: att.gen, message: 'MediaStreamTrackProcessor is not available in workers', unsupported: true });
    expect(await started).toBe('fallback');
    expect(src.fallbackReason).toMatch(/not available/);
    expect(src.fixedOrientation).toBe(false);
    expect(worker.terminated).toBe(true);
    cam.emit(42);
    expect(ts).toEqual([42]);
    const b = await src.grabFrame();
    expect(b.width).toBe(1440);
    expect(cam.grabs).toBe(1);
  });

  it('falls back when the track cannot be transferred or no frame arrives', async () => {
    const a = make({ throwOnAttach: true });
    expect(await a.src.start()).toBe('fallback');
    expect(a.src.fallbackReason).toMatch(/cannot be transferred/);
    expect(a.cam.track!.clones[0].stopped).toBe(true);

    const b = make({ firstFrameTimeoutMs: 30 });
    const started = b.src.start();
    b.worker.emit({ type: 'attached', gen: b.worker.lastAttach().gen, timeOrigin: 1000 });
    expect(await started).toBe('fallback');
    expect(b.src.fallbackReason).toMatch(/no camera frame/);

    const c = make({ attachTimeoutMs: 30 });
    expect(await c.src.start()).toBe('fallback');
    expect(c.src.fallbackReason).toMatch(/did not attach/);
  });

  it('detaches while the camera restarts and attaches the new track afterwards; never falls back after frames', async () => {
    const { cam, worker, src } = make();
    const started = src.start();
    const g1 = worker.lastAttach().gen;
    worker.emit({ type: 'attached', gen: g1, timeOrigin: 1000 });
    const ts: number[] = [];
    src.onFrame((t) => ts.push(t));
    worker.emit(frame(g1, 1, 1_000_000, 5));
    expect(await started).toBe('mstp');
    expect(ts).toHaveLength(1);

    cam.setState('restarting');
    expect(worker.sent.at(-1)!.msg.type).toBe('detach');
    worker.emit(frame(g1, 2, 1_033_000, 40)); // late frame of the old clone: ignored
    expect(ts).toHaveLength(1);
    cam.track = new FakeTrack();
    cam.segment = 1;
    cam.setState('running');
    const g2 = worker.lastAttach().gen;
    expect(g2).toBeGreaterThan(g1);
    expect(worker.lastAttach().track).toBe(cam.track.clones[0]);
    worker.emit(frame(g2, 1, 9_000_000, 900)); // new capture clock after the restart: re-estimated
    expect(ts).toHaveLength(2);
    expect(ts[1]).toBeCloseTo(900, 6);

    // the worker track ends: re-attach (same camera track), no fallback
    worker.emit({ type: 'ended', gen: g2 });
    expect(src.mode).toBe('mstp');
    expect(worker.lastAttach().gen).toBeGreaterThan(g2);
    expect(src.stats.reattaches).toBe(2);

    // the camera releases its stream while hidden: the clone must not keep the camera open
    const before = worker.sent.length;
    cam.track = null;
    cam.setState('interrupted');
    expect(worker.sent.slice(before).map((s) => s.msg.type)).toEqual(['detach']);
    src.stop();
  });

  it('rejects a grab the worker does not answer', async () => {
    const { worker, src } = make();
    const started = src.start();
    const g = worker.lastAttach().gen;
    worker.emit(frame(g, 1, 1000, 1));
    await started;
    const p = src.grabFrame().catch((e: Error) => e);
    await sleep(150);
    expect(String(await p)).toMatch(/did not answer/);
    expect(src.stats.grabErrors).toBe(1);
  });
});

describe('FrameClock', () => {
  it('maps the capture clock with the minimum observed latency and rejects unusable timestamps', () => {
    const c = new FrameClock(3);
    expect(c.map(10_000, 25)).toEqual({ t: 25, source: 'frameTimestamp' }); // offset 15
    expect(c.map(20_000, 32)).toEqual({ t: 32, source: 'frameTimestamp' }); // offset 12 (new min)
    expect(c.map(30_000, 50).t).toBe(42);
    expect(c.map(30_000, 60)).toEqual({ t: 60, source: 'callback' }); // not increasing
    expect(c.map(0, 70)).toEqual({ t: 70, source: 'callback' });
    // window of 3: the 12 ms minimum expires
    c.map(40_000, 60); // 20
    c.map(50_000, 70); // 20
    c.map(60_000, 85); // 25 → window {20, 20, 25}
    expect(c.offsetMs).toBe(20);
    // a clock jump (offset 1.5 s larger than the minimum suggests) restarts the estimate
    expect(c.map(70_000, 2_000)).toEqual({ t: 2_000, source: 'frameTimestamp' });
    expect(c.offsetMs).toBe(1930);
  });
});
