/**
 * EngineClient in Node against scripted fake workers (createWorker): the order of what reaches
 * each worker, and the status, when respawns overlap. The protocol round trips against a real
 * worker are in engine-client.gpu.test.ts.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { EngineRequest, FrameInput, InitResult } from '../workers/protocol';
import { EngineClient, type EngineStatus } from './engine-client';

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
  get methods(): string[] {
    return this.posted.filter((r) => r.method !== 'dispose').map((r) => r.method);
  }
  emit(msg: unknown): void {
    for (const cb of this.listeners.get('message') ?? []) cb({ data: msg });
  }
  /** Answer the last request of `method`. */
  reply(method: string, result: unknown): void {
    const req = this.posted.filter((r) => r.method === method).at(-1);
    if (!req) throw new Error(`no '${method}' request`);
    this.emit({ id: req.id, ok: true, result });
  }
  gpuLost(): void {
    this.emit({ type: 'error', error: { name: 'GPUDeviceLostError', message: 'device lost' }, fatal: true, reason: 'gpu-lost' });
  }
}

const clients: EngineClient[] = [];
afterEach(() => {
  for (const c of clients.splice(0)) c.dispose();
  vi.useRealTimers();
});

function client(opts: ConstructorParameters<typeof EngineClient>[0] = {}): { c: EngineClient; workers: FakeWorker[]; statuses: EngineStatus[] } {
  const workers: FakeWorker[] = [];
  const statuses: EngineStatus[] = [];
  const c = new EngineClient({
    forwardVisibility: false,
    ...opts,
    createWorker: () => {
      const w = new FakeWorker();
      workers.push(w);
      return w as unknown as Worker;
    },
  });
  c.onStatus((s) => statuses.push(s));
  clients.push(c);
  return { c, workers, statuses };
}

const initResult = (adapter: string): InitResult => ({ adapter, gpu: true }) as unknown as InitResult;
const frame = (t: number): FrameInput => ({ bitmap: { close() {} }, t, screenAngle: 0, imu: [] }) as unknown as FrameInput;

/** Let every pending promise continuation run (no timers involved). */
async function flush(): Promise<void> {
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

async function start(c: EngineClient, workers: FakeWorker[]): Promise<void> {
  const ready = c.init({ mode: 'object', quality: 'fast', sessionId: 's1' });
  await flush();
  workers[0].reply('init', initResult('w1'));
  await ready;
}

describe('EngineClient: overlapping respawns', () => {
  it('a frame queued behind a replay superseded by a second respawn waits for the newer replay; the state stays respawning', async () => {
    const { c, workers, statuses } = client();
    await start(c, workers);

    workers[0].gpuLost(); // respawn #1: worker 2 replays init
    const upd = c.processFrame(frame(1)); // the capture pump goes on: queued behind replay #1
    await flush();
    expect(workers[1].methods).toEqual(['init']);

    workers[1].gpuLost(); // respawn #2 before replay #1's init answered
    await flush();
    expect(workers).toHaveLength(3);
    expect(workers[2].methods).toEqual(['init']); // the frame is not sent before this init completes
    expect(c.status.state).toBe('respawning');

    workers[2].reply('init', initResult('w3'));
    await flush();
    expect(workers[2].methods).toEqual(['init', 'frame']);
    workers[2].reply('frame', { frameIndex: 0 });
    await expect(upd).resolves.toMatchObject({ frameIndex: 0 });
    expect(c.status).toMatchObject({ state: 'ready', reason: 'recovered after: gpu-lost' });
    expect(c.status.init).toMatchObject({ adapter: 'w3' });
    expect(statuses.map((s) => s.state)).toEqual(['starting', 'ready', 'respawning', 'respawning', 'ready']);
  });

  it('a frame waiting across a second respawn still counts as in flight: a second frame is refused until it is answered', async () => {
    const { c, workers } = client();
    await start(c, workers);

    const lost = c.processFrame(frame(0)); // in flight on worker 1 when its GPU is lost
    await flush();
    workers[0].gpuLost(); // respawn #1
    await expect(lost).rejects.toMatchObject({ name: 'EngineCrashedError' });
    expect(c.frameInFlight).toBe(false); // the capture pump goes on
    const upd = c.processFrame(frame(1)); // queued behind replay #1
    await flush();

    workers[1].gpuLost(); // respawn #2: the frame now waits for replay #2
    await flush();
    expect(c.frameInFlight).toBe(true); // capture's finish waits for it before finishCapture
    await expect(c.processFrame(frame(2))).rejects.toMatchObject({ name: 'EngineBusyError' });

    workers[2].reply('init', initResult('w3'));
    await flush();
    expect(workers[2].methods).toEqual(['init', 'frame']);
    workers[2].reply('frame', { frameIndex: 1 });
    await expect(upd).resolves.toMatchObject({ frameIndex: 1 });
    expect(c.frameInFlight).toBe(false);
  });

  it('a replayed init that times out: the next replay keeps the state and the queued call waits for it', async () => {
    vi.useFakeTimers();
    const { c, workers, statuses } = client({ initTimeoutMs: 1000 });
    await start(c, workers);

    workers[0].gpuLost(); // respawn #1: worker 2 never answers its init
    const sparse = c.refine();
    await flush();
    vi.advanceTimersByTime(1000); // replay #1's init times out → respawn #2
    await flush();
    expect(workers).toHaveLength(3);
    expect(workers[2].methods).toEqual(['init']);
    expect(c.status.state).toBe('respawning');

    workers[2].reply('init', initResult('w3'));
    await flush();
    expect(workers[2].methods).toEqual(['init', 'refine']);
    workers[2].reply('refine', { points: { count: 1 } });
    await expect(sparse).resolves.toMatchObject({ points: { count: 1 } });
    expect(c.status).toMatchObject({ state: 'ready', reason: "recovered after: 'init' timed out" });
    expect(statuses.some((s) => s.state === 'idle')).toBe(false);
  });
});

describe('EngineClient: superseded init / loadSession', () => {
  it('a load that ends after a newer init started leaves the status to that init', async () => {
    const { c, workers, statuses } = client();
    const load = c.loadSession('A');
    await flush();
    const init = c.init({ mode: 'object', quality: 'fast', sessionId: 'B' });
    await flush();
    expect(workers[0].methods).toEqual(['loadSession', 'init']);
    // the worker replaced A's engine with B's: A's load rejects while B's init is still running
    const req = workers[0].posted.find((r) => r.method === 'loadSession')!;
    workers[0].emit({ id: req.id, ok: false, error: { name: 'EngineCrashedError', message: 'Engine disposed' } });
    await expect(load).rejects.toMatchObject({ message: 'Engine disposed' });
    expect(c.status.state).toBe('starting');
    workers[0].reply('init', initResult('w1'));
    await init;
    expect(c.status.state).toBe('ready');
    expect(statuses.some((s) => s.state === 'idle')).toBe(false);
  });

  it('an init that succeeds after a newer load started does not report ready for it', async () => {
    const { c, workers, statuses } = client();
    const init = c.init({ mode: 'object', quality: 'fast', sessionId: 'A' });
    await flush();
    const load = c.loadSession('B');
    await flush();
    workers[0].reply('init', initResult('w1'));
    await init;
    expect(c.status.state).toBe('starting');
    workers[0].reply('loadSession', { id: 'B' });
    await load;
    expect(c.status.state).toBe('ready');
    expect(statuses.filter((s) => s.state === 'ready')).toHaveLength(1);
  });
});

describe('EngineClient: closeSession of a scan being loaded', () => {
  it('a load answered after the close (its engine was disposed) does not set ready', async () => {
    const { c, workers } = client();
    const load = c.loadSession('A').catch((e: Error) => e);
    await flush();
    const close = c.closeSession('A');
    await flush();
    expect(workers[0].methods).toEqual(['loadSession', 'close']);
    workers[0].reply('close', true);
    await close;
    expect(c.status).toMatchObject({ state: 'idle', reason: 'closed' });
    workers[0].reply('loadSession', { id: 'A' });
    await load;
    expect(c.status).toMatchObject({ state: 'idle', reason: 'closed' });
  });

  it('a replay of the closed scan answered after the close does not set ready', async () => {
    const { c, workers } = client();
    const load = c.loadSession('A');
    await flush();
    workers[0].reply('loadSession', { id: 'A' });
    await load;
    workers[0].gpuLost(); // worker 2 replays loadSession(A)
    await flush();
    expect(workers[1].methods).toEqual(['loadSession']);
    const close = c.closeSession('A');
    await flush();
    workers[1].reply('close', true);
    await close;
    workers[1].reply('loadSession', { id: 'A' });
    await flush();
    expect(c.status).toMatchObject({ state: 'idle', reason: 'closed' });
  });

  it('a replay superseded by a newer init leaves the status to that init, also when it fails', async () => {
    const { c, workers } = client();
    const load = c.loadSession('A');
    await flush();
    workers[0].reply('loadSession', { id: 'A' });
    await load;
    workers[0].gpuLost(); // worker 2 replays loadSession(A)
    await flush();
    const init = c.init({ mode: 'object', quality: 'fast', sessionId: 'B' }).catch((e: Error) => e);
    await flush();
    expect(workers[1].methods).toEqual(['loadSession', 'init']);
    workers[1].reply('loadSession', { id: 'A' }); // the replay ends after the newer init started
    await flush();
    expect(c.status.state).toBe('starting');
    const req = workers[1].posted.filter((r) => r.method === 'init').at(-1)!;
    workers[1].emit({ id: req.id, ok: false, error: { name: 'Error', message: 'compile failed' } });
    expect(await init).toMatchObject({ message: 'compile failed' });
    expect(c.status).toMatchObject({ state: 'idle', reason: 'compile failed' });
  });
});

describe('EngineClient: a continued capture (init with extend)', () => {
  it('is reopened by a replay until finishCapture resolved, then replayed as stored (resume only)', async () => {
    const { c, workers } = client();
    const ready = c.init({ mode: 'object', quality: 'fast', sessionId: 's1', resume: true, extend: true });
    await flush();
    workers[0].reply('init', initResult('w1'));
    await ready;
    const initOf = (w: FakeWorker) => w.posted.find((r) => r.method === 'init')!.params as { resume?: boolean; extend?: boolean };

    workers[0].gpuLost(); // during the capture: the replay reopens it
    await flush();
    expect(initOf(workers[1])).toMatchObject({ resume: true, extend: true });
    workers[1].reply('init', initResult('w2'));
    await flush();

    const done = c.finishCapture();
    await flush();
    workers[1].reply('finishCapture', undefined);
    await done;
    workers[1].gpuLost(); // after it: restored as stored
    await flush();
    expect(initOf(workers[2]).resume).toBe(true);
    expect(initOf(workers[2]).extend).toBeUndefined();
  });
});
