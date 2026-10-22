/**
 * FrameTap (./frame-tap.ts) on its own and on both EngineLikes: the tap sees each frame before
 * it is sent (pixels still there), may hold back processFrame's resolution, gets the answer
 * afterwards; a busy-rejected frame never reaches it; a failing tap never fails the frame.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { EngineRequest, FrameInput, InitResult, LiveUpdate } from '../workers/protocol';
import { EngineClient } from './engine-client';
import { tapFrame, type FrameTap } from './frame-tap';
import { MockEngine } from './mock-engine';

const update = (i: number): LiveUpdate => ({ frameIndex: i, decision: 'skip', guidance: 'good', tracking: 'tracking', numKeyframes: 0, numRegistered: 0, numPoints: 0, timings: {} });

function frame(i: number, log?: string[]): FrameInput {
  const bitmap = { width: 64, height: 48, closed: false, close: () => ((bitmap.closed = true), log?.push(`close ${i}`)) };
  return { bitmap: bitmap as unknown as ImageBitmap, t: 1000 + i, screenAngle: 0, imu: [], index: i };
}

const flush = async (): Promise<void> => {
  for (let i = 0; i < 20; i++) await Promise.resolve();
};

afterEach(() => vi.restoreAllMocks());

describe('tapFrame', () => {
  it('without a tap it just sends', async () => {
    expect(await tapFrame(null, frame(0), async () => update(7))).toEqual(update(7));
  });

  it('before() runs before the send with the pixels intact; after() gets the answer once the hold ended', async () => {
    const log: string[] = [];
    let release!: () => void;
    const tap: FrameTap = {
      before: (f) => {
        log.push(`before ${f.index} closed=${(f.bitmap as unknown as { closed: boolean }).closed}`);
        return new Promise<void>((r) => (release = r));
      },
      after: (f, res, err, t) => log.push(`after ${f.index} ${res?.frameIndex} ${err} ms>=0 ${t.ms >= 0} wait>=0 ${t.waitMs >= 0}`),
    };
    let resolved = false;
    const p = tapFrame(tap, frame(3, log), async (f) => {
      log.push('send');
      (f.bitmap as unknown as { close(): void }).close(); // transferred / closed by the engine
      return update(3);
    }).then((u) => {
      resolved = true;
      return u;
    });
    await flush();
    expect(log).toEqual(['before 3 closed=false', 'send', 'close 3']);
    expect(resolved).toBe(false); // held back by the tap
    release();
    expect((await p).frameIndex).toBe(3);
    expect(log.at(-1)).toBe('after 3 3 null ms>=0 true wait>=0 true');
  });

  it('a failing send reaches after() and still rejects; a throwing tap never fails the frame', async () => {
    const seen: unknown[] = [];
    const tap: FrameTap = { before: () => undefined, after: (_f, res, err) => seen.push(res, err) };
    await expect(tapFrame(tap, frame(1), async () => Promise.reject(new Error('boom')))).rejects.toThrow('boom');
    expect(seen[0]).toBeNull();
    expect((seen[1] as Error).message).toBe('boom');
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const bad: FrameTap = {
      before: () => {
        throw new Error('tap broken');
      },
      after: () => {
        throw new Error('tap broken too');
      },
    };
    expect((await tapFrame(bad, frame(2), async () => update(2))).frameIndex).toBe(2);
    const rejecting: FrameTap = { before: () => Promise.reject(new Error('hold failed')) };
    expect((await tapFrame(rejecting, frame(4), async () => update(4))).frameIndex).toBe(4);
  });
});

class FakeWorker {
  readonly posted: EngineRequest[] = [];
  private readonly listeners = new Map<string, ((ev: unknown) => void)[]>();
  addEventListener(type: string, cb: (ev: unknown) => void): void {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), cb]);
  }
  postMessage(req: EngineRequest): void {
    this.posted.push(req);
  }
  terminate(): void {}
  reply(method: string, result: unknown): void {
    const req = this.posted.filter((r) => r.method === method).at(-1)!;
    for (const cb of this.listeners.get('message') ?? []) cb({ data: { id: req.id, ok: true, result } });
  }
}

describe('EngineLike.setFrameTap', () => {
  it('EngineClient: the tap sees the frames it sends (not a busy-rejected one) and holds back their resolution', async () => {
    let worker!: FakeWorker;
    const c = new EngineClient({ forwardVisibility: false, createWorker: () => (worker = new FakeWorker()) as unknown as Worker });
    const ready = c.init({ mode: 'object', quality: 'fast' });
    await flush();
    worker.reply('init', { gpu: true } as InitResult);
    await ready;
    const seen: number[] = [];
    let release!: () => void;
    c.setFrameTap({ before: (f) => (seen.push(f.index!), new Promise<void>((r) => (release = r))), after: (f) => seen.push(-f.index!) });
    const p = c.processFrame(frame(1));
    await expect(c.processFrame(frame(2))).rejects.toMatchObject({ name: 'EngineBusyError' });
    await flush();
    worker.reply('frame', update(1));
    await flush();
    expect(c.frameInFlight).toBe(true); // waiting for the tap
    release();
    expect((await p).frameIndex).toBe(1);
    expect(seen).toEqual([1, -1]);
    c.setFrameTap(null);
    const q = c.processFrame(frame(3));
    await flush();
    worker.reply('frame', update(3));
    await q;
    expect(seen).toEqual([1, -1]);
    c.dispose();
  });

  it('MockEngine: the tap copies the frame before the mock closes it', async () => {
    const e = new MockEngine({ speed: 40, sessions: async () => null });
    await e.init({ mode: 'object', quality: 'fast' });
    const seen: string[] = [];
    e.setFrameTap({ before: (f) => void seen.push(`before closed=${(f.bitmap as unknown as { closed: boolean }).closed}`), after: (_f, u) => void seen.push(`after ${u?.decision}`) });
    const u = await e.processFrame(frame(0));
    expect(seen).toEqual(['before closed=false', `after ${u.decision}`]);
    e.dispose();
  });
});
