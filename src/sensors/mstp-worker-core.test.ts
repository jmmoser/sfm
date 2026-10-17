import { describe, expect, it } from 'vitest';
import type { CameraWorkerEvent } from './mstp-protocol';
import { CameraWorkerCore, grabSize, type CameraWorkerEnv, type FrameLike } from './mstp-worker-core';

class FakeFrame implements FrameLike {
  closed = false;
  readonly displayWidth = 1920;
  readonly displayHeight = 1440;
  readonly codedWidth = 1920;
  readonly codedHeight = 1440;
  readonly format = 'NV12';
  constructor(readonly timestamp: number) {}
  close(): void {
    this.closed = true;
  }
}

/** A reader fed by the test: push() delivers a frame to a pending read(), end() finishes the stream. */
class FakeReader {
  private queue: ({ value: FakeFrame; done: false } | { value: undefined; done: true })[] = [];
  private waiting: ((r: { value: FakeFrame; done: false } | { value: undefined; done: true }) => void) | null = null;
  cancelled = false;
  push(f: FakeFrame): void {
    this.deliver({ value: f, done: false });
  }
  end(): void {
    this.deliver({ value: undefined, done: true });
  }
  private deliver(r: { value: FakeFrame; done: false } | { value: undefined; done: true }): void {
    if (this.waiting) {
      const w = this.waiting;
      this.waiting = null;
      w(r);
    } else this.queue.push(r);
  }
  read(): Promise<{ value: FakeFrame; done: false } | { value: undefined; done: true }> {
    const q = this.queue.shift();
    if (q) return Promise.resolve(q);
    return new Promise((r) => (this.waiting = r));
  }
  cancel(): Promise<void> {
    this.cancelled = true;
    this.end();
    return Promise.resolve();
  }
}

class FakeTrack {
  stopped = false;
  stop(): void {
    this.stopped = true;
  }
}

function tick(): Promise<void> {
  return new Promise((r) => setTimeout(r, 0));
}

function setup(opts: { unsupported?: boolean; bitmap?: (f: FrameLike) => Promise<ImageBitmap> } = {}) {
  const events: CameraWorkerEvent[] = [];
  const transfers: (Transferable[] | undefined)[] = [];
  const readers: FakeReader[] = [];
  const env: CameraWorkerEnv = {
    post: (m, t) => {
      events.push(m);
      transfers.push(t);
    },
    createReader: () => {
      if (opts.unsupported) return null;
      const r = new FakeReader();
      readers.push(r);
      return r as unknown as ReadableStreamDefaultReader<FrameLike>;
    },
    toBitmap: opts.bitmap ?? ((f) => Promise.resolve({ width: f.displayWidth, height: f.displayHeight, close() {} } as unknown as ImageBitmap)),
    now: () => 1234,
    timeOrigin: 5000,
  };
  return { core: new CameraWorkerCore(env), events, transfers, readers };
}

describe('CameraWorkerCore', () => {
  it('announces frames, keeps only the current one open and grabs it on request', async () => {
    const { core, events, transfers, readers } = setup();
    const track = new FakeTrack();
    core.handle({ type: 'attach', gen: 1, track: track as unknown as MediaStreamTrack, maxBufferSize: 2 });
    expect(events[0]).toEqual({ type: 'attached', gen: 1, timeOrigin: 5000 });
    const f1 = new FakeFrame(1_000_000);
    const f2 = new FakeFrame(1_033_333);
    readers[0].push(f1);
    await tick();
    expect(events[1]).toMatchObject({ type: 'frame', gen: 1, seq: 1, timestampUs: 1_000_000, receivedMs: 1234, width: 1920, height: 1440, format: 'NV12' });
    expect(f1.closed).toBe(false);
    readers[0].push(f2);
    await tick();
    expect(f1.closed).toBe(true); // replaced ⇒ closed at once
    expect(f2.closed).toBe(false);

    core.handle({ type: 'grab', id: 7, gen: 1, seq: 2, maxSide: 1280 });
    await tick();
    const grabbed = events.find((e) => e.type === 'grabbed');
    expect(grabbed).toMatchObject({ type: 'grabbed', id: 7 });
    expect('bitmap' in grabbed!).toBe(true);
    expect(transfers[events.indexOf(grabbed!)]).toHaveLength(1);
    expect(f2.closed).toBe(false); // still current

    core.handle({ type: 'grab', id: 8, gen: 1, seq: 1 });
    core.handle({ type: 'grab', id: 9, gen: 0, seq: 2 });
    await tick();
    expect(events.filter((e) => e.type === 'grabbed').slice(1)).toEqual([
      { type: 'grabbed', id: 8, error: 'stale' },
      { type: 'grabbed', id: 9, error: 'no frame' },
    ]);

    core.handle({ type: 'detach' });
    await tick();
    expect(readers[0].cancelled).toBe(true);
    expect(f2.closed).toBe(true);
    expect(track.stopped).toBe(true);
    expect(events.some((e) => e.type === 'ended')).toBe(false); // a detached track is not reported
  });

  it('closes a frame replaced during its conversion only when the conversion finished', async () => {
    let finish: (() => void) | null = null;
    const { core, events, readers } = setup({
      bitmap: () => new Promise<ImageBitmap>((r) => (finish = () => r({ close() {} } as unknown as ImageBitmap))),
    });
    core.handle({ type: 'attach', gen: 3, track: new FakeTrack() as unknown as MediaStreamTrack, maxBufferSize: 2 });
    const f1 = new FakeFrame(10);
    readers[0].push(f1);
    await tick();
    core.handle({ type: 'grab', id: 1, gen: 3, seq: 1 });
    readers[0].push(new FakeFrame(20));
    await tick();
    expect(f1.closed).toBe(false); // pinned by the conversion
    finish!();
    await tick();
    expect(f1.closed).toBe(true);
    expect(events.find((e) => e.type === 'grabbed')).toMatchObject({ id: 1 });
  });

  it('reports a missing MediaStreamTrackProcessor and the end of the stream', async () => {
    const a = setup({ unsupported: true });
    const t = new FakeTrack();
    a.core.handle({ type: 'attach', gen: 1, track: t as unknown as MediaStreamTrack, maxBufferSize: 2 });
    expect(a.events).toEqual([{ type: 'error', gen: 1, message: 'MediaStreamTrackProcessor is not available in workers', unsupported: true }]);
    expect(t.stopped).toBe(true);

    const b = setup();
    b.core.handle({ type: 'attach', gen: 2, track: new FakeTrack() as unknown as MediaStreamTrack, maxBufferSize: 2 });
    const f = new FakeFrame(5);
    b.readers[0].push(f);
    b.readers[0].end();
    await tick();
    await tick();
    expect(b.events.at(-1)).toEqual({ type: 'ended', gen: 2 });
    expect(f.closed).toBe(true);
  });

  it('a new attach replaces the previous track', async () => {
    const { core, events, readers } = setup();
    const t1 = new FakeTrack();
    core.handle({ type: 'attach', gen: 1, track: t1 as unknown as MediaStreamTrack, maxBufferSize: 2 });
    core.handle({ type: 'attach', gen: 2, track: new FakeTrack() as unknown as MediaStreamTrack, maxBufferSize: 2 });
    expect(t1.stopped).toBe(true);
    expect(readers[0].cancelled).toBe(true);
    readers[1].push(new FakeFrame(1));
    await tick();
    expect(events.filter((e) => e.type === 'frame').map((e) => e.gen)).toEqual([2]);
  });
});

describe('grabSize', () => {
  it('keeps the aspect ratio and never enlarges', () => {
    expect(grabSize(1920, 1440, 1280)).toEqual({ width: 1280, height: 960 });
    expect(grabSize(1440, 1920, 1280)).toEqual({ width: 960, height: 1280 });
    expect(grabSize(1920, 1440, 4000)).toBeNull();
    expect(grabSize(1920, 1440)).toBeNull();
  });
});
