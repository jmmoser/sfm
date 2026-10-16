/**
 * EngineClient against a tiny in-test worker (engine-client.worker-fixture.ts) in Chromium:
 * request ids, transfers, progress routing, live / log events, back-pressure, cancel, init /
 * frame timeouts, fatal worker death → respawn + re-init (resume), startup failure, respawn
 * cap, visibility forwarding, dispose.
 */
import { afterEach, describe, expect, it } from 'vitest';
import type { LiveUpdate, LogMessage } from '../workers/protocol';
import { EngineClient, type EngineStatus } from './engine-client';

const clients: EngineClient[] = [];
afterEach(() => {
  for (const c of clients.splice(0)) c.dispose();
});

function fixture(name: string): Worker {
  return new Worker(new URL('./engine-client.worker-fixture.ts', import.meta.url), { type: 'module', name });
}

function client(names: string[] | string, opts: ConstructorParameters<typeof EngineClient>[0] = {}): { c: EngineClient; spawned: string[] } {
  const queue = Array.isArray(names) ? [...names] : [names];
  const spawned: string[] = [];
  const c = new EngineClient({
    forwardVisibility: false,
    ...opts,
    createWorker: () => {
      const n = queue.length > 1 ? queue.shift()! : queue[0];
      spawned.push(n);
      return n === 'broken' ? brokenWorker() : fixture(n);
    },
  });
  clients.push(c);
  return { c, spawned };
}

function brokenWorker(): Worker {
  const url = URL.createObjectURL(new Blob(['this is not ( valid javascript'], { type: 'text/javascript' }));
  return new Worker(url, { type: 'module' });
}

async function bitmap(w = 64, h = 48): Promise<ImageBitmap> {
  return createImageBitmap(new ImageData(w, h));
}

function waitFor(pred: () => boolean, timeoutMs = 10_000): Promise<void> {
  return new Promise((resolve, reject) => {
    const t0 = performance.now();
    const tick = (): void => {
      if (pred()) resolve();
      else if (performance.now() - t0 > timeoutMs) reject(new Error('waitFor timed out'));
      else setTimeout(tick, 20);
    };
    tick();
  });
}

describe('EngineClient', () => {
  it('init + frame round trip transfers the bitmap and emits the LiveUpdate', async () => {
    const { c } = client('ok');
    const statuses: EngineStatus[] = [];
    c.onStatus((s) => statuses.push(s));
    const res = await c.init({ mode: 'object', quality: 'fast' });
    expect(res.adapter).toBe('fixture:ok:resume=false:gpu=true');
    expect(c.status.state).toBe('ready');
    expect(statuses.map((s) => s.state)).toEqual(['starting', 'ready']);

    const lives: LiveUpdate[] = [];
    c.onLive((u) => lives.push(u));
    const bm = await bitmap(64, 48);
    const p = c.processFrame({ bitmap: bm, t: 1, screenAngle: 0, imu: [{ t: 0, gyro: [0, 0, 0], accel: null, accelG: null }] });
    expect(c.frameInFlight).toBe(true);
    const upd = await p;
    expect(bm.width).toBe(0); // transferred (detached) — never copied
    expect(upd.timings).toEqual({ width: 64, height: 48, imu: 1 });
    expect(upd.overlay).toBeInstanceOf(Float32Array);
    expect(upd.sparse!.positions.length).toBe(3000);
    expect(lives).toEqual([upd]);
    expect(c.frameInFlight).toBe(false);
    expect(c.pendingCount).toBe(0);
  });

  it('back-pressure: a second frame while one is in flight is rejected and closed', async () => {
    const { c } = client('ok');
    await c.init({ mode: 'object', quality: 'fast' });
    const first = c.processFrame({ bitmap: await bitmap(), t: 1, screenAngle: 0, imu: [] });
    const second = await bitmap();
    await expect(c.processFrame({ bitmap: second, t: 2, screenAngle: 0, imu: [] })).rejects.toMatchObject({ name: 'EngineBusyError' });
    expect(second.width).toBe(0); // closed
    await first;
  });

  it('routes progress to the request, forwards log and live events, returns typed arrays', async () => {
    const { c } = client('ok');
    await c.init({ mode: 'object', quality: 'fast' });
    const prog: number[] = [];
    const global: string[] = [];
    const logs: LogMessage[] = [];
    c.onProgress((p) => global.push(p.stage));
    c.onLog((l) => logs.push(l));
    const sparse = await c.refine((p) => prog.push(p.progress));
    expect(prog).toEqual([0, 0.5, 1]);
    expect(global).toEqual(['matching', 'matching', 'matching']);
    expect(logs.map((l) => l.message)).toEqual(['refined']);
    expect(sparse.points.positions).toBeInstanceOf(Float32Array);
    expect(sparse.points.positions.length).toBe(15000);
    expect(sparse.cameras[0].pose.t).toBeInstanceOf(Float64Array);
    expect(Array.from(sparse.cameras[0].pose.t)).toEqual([1, 2, 3]);

    const lives: LiveUpdate[] = [];
    c.onLive((u) => lives.push(u));
    const mesh = await c.mesh();
    expect(mesh.triangleCount).toBe(1);
    expect(lives.map((u) => u.numKeyframes)).toEqual([9]); // unsolicited live event
    const blob = await c.exportModel('ply-points');
    expect(await blob.text()).toBe('ply\n');
  });

  it('cancel stops the running long operation with CancelledError', async () => {
    const { c } = client('ok');
    await c.init({ mode: 'object', quality: 'fast' });
    let n = 0;
    const p = c.densify(undefined, () => n++);
    await waitFor(() => n >= 2);
    c.cancel();
    await expect(p).rejects.toMatchObject({ name: 'CancelledError' });
    expect(c.pendingCount).toBe(0);
  });

  it('init timeout respawns the worker and replays init with resume', async () => {
    const { c, spawned } = client(['hang-init', 'ok'], { initTimeoutMs: 300 });
    const statuses: EngineStatus[] = [];
    c.onStatus((s) => statuses.push(s));
    await expect(c.init({ mode: 'scene', quality: 'fast', sessionId: 's1' })).rejects.toMatchObject({ name: 'EngineTimeoutError' });
    await waitFor(() => c.status.state === 'ready');
    expect(spawned).toEqual(['hang-init', 'ok']);
    expect(c.status.init?.adapter).toBe('fixture:ok:resume=true:gpu=true');
    expect(c.status.respawns).toBe(1);
    expect(statuses.some((s) => s.state === 'respawning')).toBe(true);
    // Calls made after the respawn go to the new worker.
    const upd = await c.processFrame({ bitmap: await bitmap(), t: 1, screenAngle: 0, imu: [] });
    expect(upd.frameIndex).toBe(0);
  });

  it('a fatal (GPU lost) event fails pending requests, respawns and re-inits', async () => {
    const { c, spawned } = client(['fatal', 'ok']);
    await c.init({ mode: 'object', quality: 'balanced', sessionId: 'abc' });
    const pending = c.densify(); // never finishes in the fixture
    const crashed = expect(pending).rejects.toMatchObject({ name: 'EngineCrashedError' });
    await crashed;
    await waitFor(() => c.status.state === 'ready' && c.status.respawns === 1);
    expect(spawned).toEqual(['fatal', 'ok']);
    expect(c.status.init?.adapter).toContain('resume=true');
    const sum = await c.loadSession('abc');
    expect(JSON.parse(String(sum.stats.init))).toMatchObject({ mode: 'object', sessionId: 'abc' });
  });

  it('repeated GPU loss drops to the CPU tier, then gives up (respawn cap)', async () => {
    const { c } = client('fatal', { maxRespawns: 2, respawnWindowMs: 60_000, cpuFallbackAfter: 2 });
    const adapters: string[] = [];
    c.onStatus((s) => {
      if (s.state === 'ready' && s.init?.adapter) adapters.push(s.init.adapter);
    });
    await c.init({ mode: 'object', quality: 'fast' });
    await waitFor(() => c.status.state === 'dead', 15_000);
    // 1st replay: still GPU; 2nd replay: CPU fallback.
    expect(adapters).toEqual(['fixture:fatal:resume=false:gpu=true', 'fixture:fatal:resume=false:gpu=true', 'fixture:fatal:resume=false:gpu=false']);
    await expect(c.refine()).rejects.toMatchObject({ name: 'EngineCrashedError' });
  });

  it('a worker that fails to load is respawned', async () => {
    const { c, spawned } = client(['broken', 'ok'], { initTimeoutMs: 5000 });
    const errors: string[] = [];
    c.onError((e) => errors.push(e.reason ?? ''));
    await expect(c.init({ mode: 'object', quality: 'fast' })).rejects.toMatchObject({ name: 'EngineCrashedError' });
    await waitFor(() => c.status.state === 'ready');
    expect(errors).toContain('startup');
    expect(spawned).toEqual(['broken', 'ok']);
  });

  it('repeated frame timeouts count as a stuck worker', async () => {
    const { c, spawned } = client(['slow-frame', 'ok'], { frameTimeoutMs: 200, maxFrameTimeouts: 2 });
    await c.init({ mode: 'object', quality: 'fast' });
    for (let i = 0; i < 2; i++) {
      await expect(c.processFrame({ bitmap: await bitmap(), t: i, screenAngle: 0, imu: [] })).rejects.toMatchObject({ name: 'EngineTimeoutError' });
    }
    await waitFor(() => c.status.state === 'ready' && c.status.respawns === 1);
    expect(spawned).toEqual(['slow-frame', 'ok']);
    await expect(c.processFrame({ bitmap: await bitmap(), t: 3, screenAngle: 0, imu: [] })).resolves.toMatchObject({ frameIndex: 0 });
  });

  it('forwards page visibility to the worker', async () => {
    const { c } = client('ok');
    await c.init({ mode: 'object', quality: 'fast' });
    c.setHidden(true);
    expect((await c.loadSession('x')).stats.hidden).toBe('true');
    c.setHidden(false);
    expect((await c.loadSession('x')).stats.hidden).toBe('false');
  });

  it('request timeouts are paused while the page is hidden (iOS suspends the worker)', async () => {
    const { c, spawned } = client(['slow-frame', 'ok'], { frameTimeoutMs: 200, maxFrameTimeouts: 1 });
    await c.init({ mode: 'object', quality: 'fast' });
    c.setHidden(true);
    let settled: string | null = null;
    const p = c.processFrame({ bitmap: await bitmap(), t: 1, screenAngle: 0, imu: [] }).then(
      () => (settled = 'ok'),
      (e: Error) => (settled = e.name),
    );
    await new Promise((r) => setTimeout(r, 600)); // 3× the timeout in the background
    expect(settled).toBeNull();
    expect(c.status.respawns).toBe(0);
    expect(spawned).toEqual(['slow-frame']);
    // Back in the foreground the full timeout starts again.
    const t0 = performance.now();
    c.setHidden(false);
    await p;
    expect(settled).toBe('EngineTimeoutError');
    expect(performance.now() - t0).toBeGreaterThanOrEqual(150);
    await waitFor(() => c.status.state === 'ready' && c.status.respawns === 1);
  });

  it('a stuck init does not respawn while hidden', async () => {
    const { c, spawned } = client(['hang-init', 'ok'], { initTimeoutMs: 150 });
    c.setHidden(true);
    let failed = false;
    const p = c.init({ mode: 'object', quality: 'fast' }).catch(() => (failed = true));
    await new Promise((r) => setTimeout(r, 500));
    expect(failed).toBe(false);
    expect(spawned).toEqual(['hang-init']);
    c.setHidden(false);
    await p;
    expect(failed).toBe(true);
    await waitFor(() => c.status.state === 'ready');
    expect(spawned).toEqual(['hang-init', 'ok']);
  });

  it('dispose rejects pending and later calls', async () => {
    const { c } = client('ok');
    await c.init({ mode: 'object', quality: 'fast' });
    const p = c.densify();
    c.dispose();
    await expect(p).rejects.toMatchObject({ name: 'EngineCrashedError' });
    await expect(c.refine()).rejects.toMatchObject({ name: 'EngineCrashedError' });
    expect(c.status.state).toBe('disposed');
  });
});
