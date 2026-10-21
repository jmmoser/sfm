/**
 * App.ensureSessionLoaded in Node, on an App built without its DOM shell (Object.create) and a
 * scripted engine: what `loadedSessionId` / `results` say after failed, superseded and released
 * loads.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { SessionSummary } from '../workers/protocol';
import { App } from './app';
import { toast } from './components/toast';
import type { EngineLike } from './engine-client';
import { RunJournal } from './run-journal';

vi.mock('./components/toast', () => ({ toast: vi.fn() }));

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
  return Object.assign(Object.create(App.prototype), { engine, config, results: App.emptyResults(config), loadedSessionId: null, loadSeq: 0, sessionsP, journal: new RunJournal(null) }) as App;
}

const summary = (id: string): SessionSummary =>
  ({ id, name: `Scan ${id}`, mode: 'object', quality: 'balanced', sparse: null, dense: null, mesh: null, stats: {} }) as unknown as SessionSummary;

beforeEach(() => {
  vi.mocked(toast).mockClear();
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
