/**
 * The page side of a bridged run (./bridge-run.ts) against a fake client, a fake sessionStorage, a
 * fake wake lock and fake timers: the per-tab token, a refused start runs nothing, a query error
 * or a failing pipeline is posted as the error, the runner's abort reaches harnessAbort once, a
 * pipeline that ignores the abort is answered with a snapshot (and its late result is not
 * posted), the wake lock is held during the run and recorded in the result, a refused lock is
 * reported. Against the real server (second tab, reloads): ./e2e-bridge.test.ts.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { StartAnswer, StartInfo } from './bridge';
import { runBridged, tabToken, TAB_KEY, type BridgedRunDeps, type WakeLockLike } from './bridge-run';
import type { HarnessConfig, HarnessDevice, HarnessResult } from './types';

const DEVICE: HarnessDevice = { userAgent: 'iPhone', platform: 'iPhone', hardwareConcurrency: 4, devicePixelRatio: 3, screen: [393, 852], secureContext: true, adapter: null, adapterError: 'x' };
const res = (extra: Partial<HarnessResult> = {}): HarnessResult => ({ ok: true, aborted: false, device: { ...DEVICE }, ...extra }) as unknown as HarnessResult;
const CONFIG: HarnessConfig = { dataset: 'object', engine: 'mock' };

function memoryStorage(): Pick<Storage, 'getItem' | 'setItem'> & { map: Map<string, string> } {
  const map = new Map<string, string>();
  return { map, getItem: (k) => map.get(k) ?? null, setItem: (k, v) => void map.set(k, v) };
}

function fakeClient(answer: (info: StartInfo) => StartAnswer | Promise<StartAnswer> = () => ({ ok: true })) {
  const starts: StartInfo[] = [];
  const results: { result: unknown; error: string | null }[] = [];
  let abort: string | null = null;
  let polls = 0;
  return {
    starts,
    results,
    get polls() {
      return polls;
    },
    setAbort: (r: string) => (abort = r),
    client: {
      start: async (info: StartInfo) => {
        starts.push(info);
        return answer(info);
      },
      poll: async () => {
        polls++;
        return { abort };
      },
      result: async (env: { result: unknown; error: string | null }) => {
        results.push(env);
      },
    },
  };
}

function fakeWakeLock(grant: boolean, error = 'NotAllowedError'): WakeLockLike & { calls: string[] } {
  const w = {
    calls: [] as string[],
    active: false,
    lastError: null as string | null,
    async acquire() {
      w.calls.push('acquire');
      w.active = grant;
      w.lastError = grant ? null : error;
    },
    async release() {
      w.calls.push('release');
      w.active = false;
    },
  };
  return w;
}

function deps(over: Partial<BridgedRunDeps> & Pick<BridgedRunDeps, 'client'>): BridgedRunDeps & { lines: string[] } {
  const lines: string[] = [];
  return {
    lines,
    run: '0123456789abcdef',
    config: CONFIG,
    queryError: null,
    runPipeline: async () => res(),
    abort: () => true,
    snapshot: () => null,
    log: (m) => lines.push(m),
    storage: memoryStorage(),
    userAgent: 'ua',
    href: 'https://phone/harness.html?bridge=1',
    newToken: () => 'tok-1',
    ...over,
  };
}

afterEach(() => {
  vi.useRealTimers();
});

describe('tabToken', () => {
  it('is created once per tab and kept in sessionStorage; null when storage is missing or throws', () => {
    const s = memoryStorage();
    let n = 0;
    const next = (): string => `t${++n}`;
    expect(tabToken(s, next)).toBe('t1');
    expect(tabToken(s, next)).toBe('t1'); // a reload of the tab
    expect(s.map.get(TAB_KEY)).toBe('t1');
    expect(tabToken(memoryStorage(), next)).toBe('t2'); // another tab
    expect(tabToken(null, next)).toBeNull();
    const blocked = { getItem: () => null, setItem: () => { throw new DOMException('quota', 'QuotaExceededError'); } };
    expect(tabToken(blocked, next)).toBeNull();
    // the default token: 128 random bits as hex
    expect(tabToken(memoryStorage())).toMatch(/^[0-9a-f]{32}$/);
  });
});

describe('runBridged', () => {
  it('announces the tab, runs, posts the result with the wake lock state, then releases the lock', async () => {
    const fc = fakeClient();
    const wl = fakeWakeLock(true);
    let lockedDuringRun = false;
    const d = deps({
      client: fc.client,
      wakeLock: wl,
      runPipeline: async (c) => {
        expect(c).toBe(CONFIG);
        lockedDuringRun = wl.active;
        return res();
      },
    });
    const o = await runBridged(d);
    expect(fc.starts).toEqual([{ userAgent: 'ua', href: 'https://phone/harness.html?bridge=1', tab: 'tok-1' }]);
    expect(lockedDuringRun).toBe(true);
    expect(fc.results).toHaveLength(1);
    expect(fc.results[0].error).toBeNull();
    expect((fc.results[0].result as HarnessResult).device!.wakeLock).toEqual({ active: true, error: null });
    expect(wl.calls).toEqual(['acquire', 'release']);
    expect(o).toMatchObject({ started: true, sent: true, error: null });
    expect(d.lines.join('\n')).toMatch(/connected \(run 01234567…\)[\s\S]*result sent to the runner/);
  });

  it('a refused start (another tab, or this tab reloaded mid-run) or no server runs nothing and posts nothing', async () => {
    for (const answer of [() => ({ ok: false as const, status: 409, message: 'this run is already running in another tab' }), () => Promise.reject(new TypeError('Load failed'))]) {
      const fc = fakeClient(answer);
      const wl = fakeWakeLock(true);
      const run = vi.fn(async () => res());
      const d = deps({ client: fc.client, wakeLock: wl, runPipeline: run });
      const o = await runBridged(d);
      expect(run).not.toHaveBeenCalled();
      expect(fc.results).toEqual([]);
      expect(wl.calls).toEqual([]);
      expect(o.started).toBe(false);
      expect(d.lines.join('\n')).toMatch(/did not accept this page \((409|network error)\)/);
    }
  });

  it('a query error is posted as the run error and nothing runs; so is a missing dataset or a failing pipeline', async () => {
    const fc = fakeClient();
    const run = vi.fn(async () => res());
    await runBridged(deps({ client: fc.client, config: null, queryError: 'harness query: engine=gpu is not one of worker, inline, mock', runPipeline: run }));
    await runBridged(deps({ client: fc.client, config: null }));
    expect(run).not.toHaveBeenCalled();
    const d = deps({ client: fc.client, runPipeline: async () => Promise.reject(new Error('boom')) });
    await runBridged(d);
    expect(fc.results).toEqual([
      { result: null, error: 'harness query: engine=gpu is not one of worker, inline, mock' },
      { result: null, error: 'harness.html: the query has no dataset' },
      { result: null, error: 'runPipeline failed: boom' },
    ]);
    expect(d.lines).toContain('[harness] runPipeline failed: boom');
  });

  it("the runner's abort (poll) reaches harnessAbort once; the pipeline's partial result is posted", async () => {
    vi.useFakeTimers();
    const fc = fakeClient();
    let finish!: (r: HarnessResult) => void;
    const abort = vi.fn(() => true);
    const p = runBridged(deps({ client: fc.client, abort, pollMs: 1000, snapshotMs: 30_000, runPipeline: () => new Promise((r) => (finish = r)) }));
    await vi.advanceTimersByTimeAsync(3000);
    expect(fc.polls).toBe(3);
    expect(abort).not.toHaveBeenCalled();
    fc.setAbort('runner timeout');
    await vi.advanceTimersByTimeAsync(3000);
    expect(abort.mock.calls).toEqual([['runner timeout']]);
    finish(res({ ok: false, aborted: true }));
    const o = await p;
    expect(fc.results).toHaveLength(1);
    expect(fc.results[0]).toMatchObject({ result: { ok: false, aborted: true }, error: null });
    expect(o.sent).toBe(true);
    // the poller and the snapshot timer are gone
    const n = fc.polls;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(fc.polls).toBe(n);
    expect(fc.results).toHaveLength(1);
  });

  it('a pipeline that ignores the abort: a snapshot is posted snapshotMs later, its late result is not', async () => {
    vi.useFakeTimers();
    const fc = fakeClient();
    let finish!: (r: HarnessResult) => void;
    const snap = res({ ok: false, aborted: false });
    const d = deps({ client: fc.client, pollMs: 1000, snapshotMs: 30_000, snapshot: () => snap, runPipeline: () => new Promise((r) => (finish = r)) });
    const p = runBridged(d);
    fc.setAbort('runner timeout');
    await vi.advanceTimersByTimeAsync(1000); // abort seen
    await vi.advanceTimersByTimeAsync(29_000);
    expect(fc.results).toEqual([]);
    await vi.advanceTimersByTimeAsync(1000);
    const o = await p;
    expect(fc.results).toEqual([{ result: snap, error: 'timeout (snapshot of the unfinished run)' }]);
    expect(o).toMatchObject({ sent: true, error: 'timeout (snapshot of the unfinished run)' });
    expect(d.lines.join('\n')).toMatch(/did not return 30 s after the abort: sending a snapshot/);
    finish(res());
    await vi.advanceTimersByTimeAsync(10_000);
    expect(fc.results).toHaveLength(1);
  });

  it('a refused wake lock is reported (log + onWakeLockFailed) and recorded; the run goes on', async () => {
    const fc = fakeClient();
    const wl = fakeWakeLock(false);
    const failed: string[] = [];
    const d = deps({ client: fc.client, wakeLock: wl, onWakeLockFailed: (e) => failed.push(e) });
    const o = await runBridged(d);
    expect(failed).toEqual(['NotAllowedError']);
    expect(d.lines.join('\n')).toMatch(/WARNING: no screen wake lock \(NotAllowedError\).*Auto-Lock > Never/);
    expect((fc.results[0].result as HarnessResult).device!.wakeLock).toEqual({ active: false, error: 'NotAllowedError' });
    expect(o.sent).toBe(true);
  });

  it('a wake lock that answers only after the run is not reported (the page has moved on)', async () => {
    const fc = fakeClient();
    let grant!: () => void;
    const wl: WakeLockLike = { active: false, lastError: null, acquire: () => new Promise<void>((r) => (grant = r)), release: async () => {} };
    const failed: string[] = [];
    await runBridged(deps({ client: fc.client, wakeLock: wl, onWakeLockFailed: (e) => failed.push(e) }));
    grant();
    await Promise.resolve();
    expect(failed).toEqual([]);
  });
});
