/**
 * App in Node, built without its DOM shell (Object.create) on a scripted engine:
 *  - ensureSessionLoaded: what `loadedSessionId` / `results` say after failed, superseded and
 *    released loads;
 *  - startRecording / stopRecording (the raw capture recorder, a scripted CaptureRecorder on a
 *    MockEngine): the tap installed and removed, an automatic stop, a capture closed while the
 *    recorder opens, a start superseded by a newer one.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { FrameInput, SessionSummary } from '../workers/protocol';
import { App } from './app';
import { toast } from './components/toast';
import type { EngineLike } from './engine-client';
import { MockEngine } from './mock-engine';
import type { CaptureRecorderOptions, RecorderStartInfo } from './recorder';

vi.mock('./components/toast', () => ({ toast: vi.fn() }));

/** CaptureRecorder.start calls, settled by the test. */
const recorders = vi.hoisted(() => ({ starts: [] as { info: unknown; opts: unknown; resolve: (r: unknown) => void; reject: (e: Error) => void }[] }));
vi.mock('./recorder', () => ({
  CaptureRecorder: {
    start: (info: unknown, opts: unknown) => new Promise((resolve, reject) => recorders.starts.push({ info, opts, resolve, reject })),
  },
}));

interface Deferred {
  id: string;
  resolve: (s: SessionSummary) => void;
  reject: (e: Error) => void;
}

/** Each loadSession waits until the test settles it (`loads`); release() ends them, like EngineClient. */
function fakeEngine(): { engine: EngineLike; loads: Deferred[] } {
  const loads: Deferred[] = [];
  const engine = {
    loadSession: (id: string) => new Promise<SessionSummary>((resolve, reject) => loads.push({ id, resolve, reject })),
    release: () => {
      for (const l of loads) l.reject(Object.assign(new Error('The engine was released'), { name: 'CancelledError' }));
      return Promise.resolve();
    },
    // the worker disposes the engine: a load of that scan still reading fails on the closed store
    closeSession: (id: string) => {
      for (const l of loads) if (l.id === id) l.reject(Object.assign(new Error('The database connection is closing.'), { name: 'InvalidStateError' }));
      return Promise.resolve();
    },
  } as unknown as EngineLike;
  return { engine, loads };
}

function fakeApp(engine: EngineLike): App {
  const config = { mode: 'object', quality: 'balanced' } as const;
  const sessionsP = Promise.resolve({ deleteSession: () => Promise.resolve() });
  return Object.assign(Object.create(App.prototype), { engine, config, results: App.emptyResults(config), loadedSessionId: null, loadSeq: 0, sessionsP }) as App;
}

const summary = (id: string): SessionSummary =>
  ({ id, name: `Scan ${id}`, mode: 'object', quality: 'balanced', sparse: null, dense: null, mesh: null, stats: {} }) as unknown as SessionSummary;

beforeEach(() => {
  vi.mocked(toast).mockClear();
  recorders.starts.length = 0;
});

describe('App.ensureSessionLoaded', () => {
  it('a failed load forgets the scan the engine held before: opening that one again reloads it', async () => {
    const { engine, loads } = fakeEngine();
    const app = fakeApp(engine);
    const a = app.ensureSessionLoaded('A');
    loads[0].resolve(summary('A'));
    expect(await a).toBe(true);
    expect(app.loadedSessionId).toBe('A');

    // the worker already replaced A's engine when B's load failed
    const b = app.ensureSessionLoaded('B');
    loads[1].reject(Object.assign(new Error('no such scan'), { name: 'SessionNotFoundError' }));
    expect(await b).toBe(false);
    expect(toast).toHaveBeenCalledTimes(1);
    expect(app.loadedSessionId).toBeNull();

    const again = app.ensureSessionLoaded('A');
    expect(loads.map((l) => l.id)).toEqual(['A', 'B', 'A']);
    loads[2].resolve(summary('A'));
    expect(await again).toBe(true);
    expect(app.loadedSessionId).toBe('A');
  });

  it('a load superseded by a newer one is not applied when it ends later', async () => {
    const { engine, loads } = fakeEngine();
    const app = fakeApp(engine);
    const a = app.ensureSessionLoaded('A'); // a large scan: Back, then another one opened
    const b = app.ensureSessionLoaded('B');
    loads[1].resolve(summary('B'));
    expect(await b).toBe(true);
    loads[0].resolve(summary('A'));
    expect(await a).toBe(false);
    expect(app.loadedSessionId).toBe('B');
    expect(app.results).toMatchObject({ sessionId: 'B', name: 'Scan B' });
  });

  it('opening the previous scan again while another one loads reloads it (the engine no longer holds it)', async () => {
    const { engine, loads } = fakeEngine();
    const app = fakeApp(engine);
    const a = app.ensureSessionLoaded('A');
    loads[0].resolve(summary('A'));
    await a;
    const b = app.ensureSessionLoaded('B'); // Back while it loads, then A again
    const again = app.ensureSessionLoaded('A');
    expect(loads.map((l) => l.id)).toEqual(['A', 'B', 'A']);
    loads[1].resolve(summary('B'));
    expect(await b).toBe(false);
    loads[2].resolve(summary('A'));
    expect(await again).toBe(true);
    expect(app.loadedSessionId).toBe('A');
    expect(app.results.name).toBe('Scan A');
  });

  it('a superseded load that fails shows no toast and leaves the newer state', async () => {
    const { engine, loads } = fakeEngine();
    const app = fakeApp(engine);
    const a = app.ensureSessionLoaded('A');
    const b = app.ensureSessionLoaded('B');
    loads[1].resolve(summary('B'));
    await b;
    loads[0].reject(Object.assign(new Error('The database connection is closing.'), { name: 'InvalidStateError' }));
    expect(await a).toBe(false);
    expect(toast).not.toHaveBeenCalled();
    expect(app.loadedSessionId).toBe('B');
    expect(app.results.name).toBe('Scan B');
  });

  it('a load still running when the engine is released is not applied', async () => {
    const { engine } = fakeEngine();
    const app = fakeApp(engine);
    const a = app.ensureSessionLoaded('A');
    await app.releaseEngine();
    expect(await a).toBe(false);
    expect(toast).not.toHaveBeenCalled();
    expect(app.loadedSessionId).toBeNull();
    expect(app.results.sessionId).toBeNull();
  });

  it('a load still running when a new scan starts is not applied', async () => {
    const { engine, loads } = fakeEngine();
    const app = fakeApp(engine);
    const navigate = vi.fn(() => true);
    Object.assign(app, { navigate, store: { get: () => ({ env: { indexedDB: true } }) } });
    const a = app.ensureSessionLoaded('A');
    app.startNewScan();
    const id = app.results.sessionId;
    expect(navigate).toHaveBeenCalledWith({ type: 'START_CAPTURE', sessionId: id });
    loads[0].resolve(summary('A'));
    expect(await a).toBe(false);
    expect(app.loadedSessionId).toBeNull();
    expect(app.results.sessionId).toBe(id);
  });

  it('a load of a scan deleted on Home meanwhile is not applied and its failure not shown', async () => {
    const { engine, loads } = fakeEngine();
    const app = fakeApp(engine);
    const a = app.ensureSessionLoaded('A'); // Back while it loads, then A deleted on Home
    expect(await app.deleteSession('A')).toBe(true);
    expect(await a).toBe(false);
    expect(loads.map((l) => l.id)).toEqual(['A']);
    expect(toast).not.toHaveBeenCalled();
    expect(app.loadedSessionId).toBeNull();
    expect(app.results.sessionId).toBeNull();
  });
});

describe('App.startRecording / stopRecording', () => {
  /** A scripted recorder: a FrameTap counting the frames it saw, stop() recorded. */
  function fakeRecorder(name: string) {
    return {
      name,
      seen: 0,
      stops: [] as string[],
      before() {
        this.seen++;
      },
      async stop(reason = 'capture ended') {
        this.stops.push(reason);
        return null;
      },
    };
  }

  async function setup(): Promise<{ app: App; engine: MockEngine; frame: () => Promise<unknown> }> {
    const engine = new MockEngine({ speed: 40, sessions: async () => null });
    await engine.init({ mode: 'object', quality: 'fast' });
    const app = Object.assign(Object.create(App.prototype), { engine, recorder: null, recordSeq: 0 }) as App;
    let t = 1000;
    const frame = (): Promise<unknown> => engine.processFrame({ bitmap: { width: 64, height: 48, close: () => undefined } as unknown as ImageBitmap, t: (t += 100), screenAngle: 0, imu: [] } as FrameInput);
    return { app, engine, frame };
  }
  const info: RecorderStartInfo = { name: 'Scan', sessionId: 's1', engine: { mode: 'object', quality: 'fast' } };
  const started = async (n: number): Promise<void> => vi.waitFor(() => expect(recorders.starts).toHaveLength(n));
  const toasts = (): string[] => vi.mocked(toast).mock.calls.map((c) => String(c[0]));

  it('installs the recorder as the frame tap until stopRecording', async () => {
    const { app, engine, frame } = await setup();
    const p = app.startRecording(info);
    await started(1);
    expect(recorders.starts[0].info).toBe(info);
    const rec = fakeRecorder('A');
    recorders.starts[0].resolve(rec);
    await p;
    expect(toasts()).toEqual([expect.stringMatching(/^Recording raw capture/)]);
    await frame();
    await frame();
    expect(rec.seen).toBe(2);
    await app.stopRecording();
    expect(rec.stops).toEqual(['capture ended']);
    await frame();
    expect(rec.seen).toBe(2); // tap removed
    await app.stopRecording(); // nothing running: no-op
    expect(rec.stops).toHaveLength(1);
    engine.dispose();
  });

  it('an automatic stop (low storage) removes the tap and says so; the scan goes on', async () => {
    const { app, engine, frame } = await setup();
    const p = app.startRecording(info);
    await started(1);
    const rec = fakeRecorder('A');
    recorders.starts[0].resolve(rec);
    await p;
    (recorders.starts[0].opts as CaptureRecorderOptions).onStop!('storage is low (900 MB free)', true);
    expect(toasts().at(-1)).toMatch(/stopped: storage is low \(900 MB free\)\. The scan continues/);
    await frame();
    expect(rec.seen).toBe(0);
    await app.stopRecording();
    expect(rec.stops).toEqual([]); // it had stopped itself
    engine.dispose();
  });

  it('a capture closed while the recorder opens: the recorder is stopped once open and never installed', async () => {
    const { app, engine, frame } = await setup();
    const p = app.startRecording(info);
    await started(1);
    await app.stopRecording(); // the capture screen's stopSensors
    const rec = fakeRecorder('A');
    recorders.starts[0].resolve(rec);
    await p;
    expect(rec.stops).toEqual([expect.stringMatching(/capture ended while the recording started/)]);
    await frame();
    expect(rec.seen).toBe(0);
    expect(toasts()).toEqual([]);
    engine.dispose();
  });

  it('a stale start that opens after a newer one does not replace it', async () => {
    const { app, engine, frame } = await setup();
    const a = app.startRecording(info); // scan A: the worker / OPFS is slow
    await started(1);
    const b = app.startRecording(info); // A cancelled, scan B started
    await started(2);
    const recB = fakeRecorder('B');
    recorders.starts[1].resolve(recB);
    await b;
    const recA = fakeRecorder('A');
    recorders.starts[0].resolve(recA);
    await a;
    expect(recA.stops).toHaveLength(1);
    expect(recB.stops).toHaveLength(0);
    await frame();
    expect([recA.seen, recB.seen]).toEqual([0, 1]);
    await app.stopRecording();
    expect(recB.stops).toEqual(['capture ended']);
    engine.dispose();
  });

  it('a start that cannot record says why, unless it was superseded meanwhile', async () => {
    const { app, engine } = await setup();
    const p = app.startRecording(info);
    await started(1);
    recorders.starts[0].reject(new Error('only 400 MB of storage are free (recording needs 1000 MB)'));
    await p;
    expect(toasts()).toEqual(['Raw capture recording is off: only 400 MB of storage are free (recording needs 1000 MB)']);
    const q = app.startRecording(info);
    await started(2);
    await app.stopRecording();
    recorders.starts[1].reject(new Error('the recorder worker did not start'));
    await q;
    expect(toasts()).toHaveLength(1);
    engine.dispose();
  });
});
