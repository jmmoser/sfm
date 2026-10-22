/**
 * The dev-server side of the e2e bridge (e2e/lib/e2e-bridge.mjs, e2e/lib/safe-path.mjs) on a real
 * node:http server, driven by the page-side client (./bridge.ts) over Node's fetch: path safety
 * (absolute paths, '..' and paths outside the run's prefix are rejected), files streamed into the
 * run's directory, the per-run byte cap, ordered logs, /__e2e/go (none once the result is in),
 * start semantics by tab token (only the starting tab's reload fails the run; also through the
 * page flow ./bridge-run.ts with per-tab storages), the runner's timeout → abort through /poll,
 * and the plugin never applying to a build.
 */
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { createBridge, harnessPath, resolveWritePath, type Bridge } from '../../../e2e/lib/e2e-bridge.mjs';
import { safeJoin } from '../../../e2e/lib/safe-path.mjs';
import { createBridgeClient } from './bridge';
import { runBridged, type BridgedRunDeps } from './bridge-run';
import type { HarnessResult } from './types';

// Node built-ins through an untyped dynamic import: the main tsconfig is browser-typed (no @types/node).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const nodeImport = (m: string): Promise<any> => import(/* @vite-ignore */ m);
const { createServer } = (await nodeImport('node:http')) as {
  createServer(h: (req: unknown, res: { statusCode: number; end(b?: string): void }) => void): {
    listen(port: number, host: string, cb: () => void): void;
    address(): { port: number };
    close(cb: () => void): void;
    closeAllConnections(): void;
  };
};
const { mkdtempSync, readFileSync, rmSync, existsSync, readdirSync } = (await nodeImport('node:fs')) as {
  mkdtempSync(prefix: string): string;
  readFileSync(path: string): Uint8Array;
  rmSync(path: string, opts: { recursive: boolean; force: boolean }): void;
  existsSync(path: string): boolean;
  readdirSync(path: string): string[];
};
const { tmpdir } = (await nodeImport('node:os')) as { tmpdir(): string };
const { join, resolve } = (await nodeImport('node:path')) as { join(...p: string[]): string; resolve(...p: string[]): string };

let bridge: Bridge;
let base = '';
let root = '';
let server: ReturnType<typeof createServer>;

beforeAll(async () => {
  root = mkdtempSync(join(tmpdir(), 'sfm-bridge-'));
  bridge = createBridge();
  // `bridge` is read per request: a test may swap in one with other options
  server = createServer((req, res) =>
    bridge.middleware(req, res, () => {
      res.statusCode = 418; // not a bridge route
      res.end('next');
    }),
  );
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  base = `http://127.0.0.1:${server.address().port}`;
});

afterAll(async () => {
  server.closeAllConnections();
  await new Promise<void>((r) => server.close(() => r()));
  rmSync(root, { recursive: true, force: true });
});

const post = (path: string, body: BodyInit, type = 'application/octet-stream'): Promise<Response> => fetch(base + path, { method: 'POST', body, headers: { 'content-type': type } });

describe('safeJoin / resolveWritePath', () => {
  it('rejects absolute paths, .. escapes and paths outside the run prefix', () => {
    const r = resolve('/tmp/run');
    expect(safeJoin(r, 'a/b.ply')).toBe(join(r, 'a/b.ply'));
    for (const bad of ['/etc/passwd', '../x', 'a/../../x', 'a\\..\\x', '', 'a/..', null]) expect(() => safeJoin(r, bad)).toThrow(/bad path/);
    expect(resolveWritePath(r, 'object-phone/', 'object-phone/sparse.ply')).toBe(join(r, 'sparse.ply'));
    expect(resolveWritePath(r, 'object-phone/', 'object-phone/session/frames/1.jpg')).toBe(join(r, 'session/frames/1.jpg'));
    expect(() => resolveWritePath(r, 'object-phone/', 'yard-phone/sparse.ply')).toThrow(/not under object-phone/);
    expect(() => resolveWritePath(r, 'object-phone/', 'object-phone/../yard/x')).toThrow(/bad path/);
    expect(() => resolveWritePath(r, 'object-phone/', 'object-phone//etc/passwd')).toThrow(/bad path/);
    expect(() => resolveWritePath(r, '', '/abs')).toThrow(/bad path/);
  });
});

describe('bridge server + page client', () => {
  it('streams files into the run directory, keeps logs in order, settles on the result', async () => {
    const lines: string[] = [];
    const dir = join(root, 'object-phone');
    const run = bridge.expectRun({ outDir: dir, prefix: 'object-phone/', config: { dataset: 'object', engine: 'mock' }, log: (m) => lines.push(m) });
    expect(run.path).toBe(harnessPath(run.id, { dataset: 'object', engine: 'mock' }));
    // /__e2e/go → the run's harness URL (relative: works for any host the phone used)
    const go = await fetch(`${base}/__e2e/go`, { redirect: 'manual' });
    expect(go.status).toBe(302);
    expect(go.headers.get('location')).toBe(run.path);
    expect((await fetch(`${base}/harness.html`)).status).toBe(418); // everything else passes through

    const c = createBridgeClient({ run: run.id, base, chunkBytes: 1000 });
    expect(await c.start({ userAgent: 'iPhone test', href: 'x', tab: 'tab-a' })).toEqual({ ok: true });
    expect(run.info).toMatchObject({ userAgent: 'iPhone test' });
    for (let i = 0; i < 50; i++) c.log(`line ${i}`);
    const bytes = Uint8Array.from({ length: 2500 }, (_, i) => (i * 31) & 255);
    await c.writeBlob('object-phone/sparse.ply', new Blob([bytes]), false);
    await c.write('object-phone/session/manifest.json', btoa('{"a":1}'), 'base64', false);
    // overwrite (append=0) truncates
    await c.writeBlob('object-phone/small.bin', new Blob([new Uint8Array(10)]), false);
    await c.writeBlob('object-phone/small.bin', new Blob([Uint8Array.of(1, 2, 3)]), false);
    await expect(c.writeBlob('object-phone/../../escape.bin', new Blob([bytes]), false)).rejects.toThrow(/HTTP 400 bad path/);
    await expect(c.writeBlob('/etc/escape.bin', new Blob([bytes]), false)).rejects.toThrow(/HTTP 400/);
    await expect(c.writeBlob('yard-phone/x.bin', new Blob([bytes]), false)).rejects.toThrow(/not under object-phone/);
    expect(await c.poll()).toEqual({ abort: null });
    await c.result({ result: { ok: true, files: ['object-phone/sparse.ply'] }, error: null });
    // the result is in: /go no longer leads to this run while the runner scores it
    const goDone = await fetch(`${base}/__e2e/go`, { redirect: 'manual' });
    expect(goDone.status).toBe(404);
    expect(await goDone.text()).toMatch(/No e2e run is waiting/);
    expect(bridge.active).toBeNull();

    expect(await run.result).toEqual({ result: { ok: true, files: ['object-phone/sparse.ply'] }, error: null });
    expect(lines).toEqual(Array.from({ length: 50 }, (_, i) => `line ${i}`));
    expect([...readFileSync(join(dir, 'sparse.ply'))]).toEqual([...bytes]);
    expect(new TextDecoder().decode(readFileSync(join(dir, 'session/manifest.json')))).toBe('{"a":1}');
    expect([...readFileSync(join(dir, 'small.bin'))]).toEqual([1, 2, 3]);
    expect(existsSync(join(root, 'escape.bin'))).toBe(false);
    expect(readdirSync(root).sort()).toEqual(['object-phone']);
    expect(Object.fromEntries(run.files)).toEqual({ 'sparse.ply': 2500, 'session/manifest.json': 7, 'small.bin': 3 });
    // a second result is refused; once closed the run is unknown and /go has nothing
    expect((await post(`/__e2e/result?run=${run.id}`, '{"result":null}', 'application/json')).status).toBe(409);
    run.close();
    expect((await post(`/__e2e/log?run=${run.id}`, 'late', 'text/plain')).status).toBe(403);
    expect((await fetch(`${base}/__e2e/go`, { redirect: 'manual' })).status).toBe(404);
  });

  it('unknown runs, wrong methods, bad bodies and unknown routes are refused', async () => {
    const run = bridge.expectRun({ outDir: join(root, 'r2'), config: { dataset: 'object' } });
    expect((await post('/__e2e/write?run=nope&path=x', 'abc')).status).toBe(403);
    expect((await fetch(`${base}/__e2e/write?run=${run.id}&path=x`)).status).toBe(405);
    expect((await post(`/__e2e/poll?run=${run.id}`, '')).status).toBe(405);
    expect((await post(`/__e2e/start?run=${run.id}`, '{nope', 'application/json')).status).toBe(400);
    expect((await post(`/__e2e/start?run=${run.id}`, 'null', 'application/json')).status).toBe(400);
    expect((await post(`/__e2e/result?run=${run.id}`, 'nope', 'application/json')).status).toBe(400);
    expect((await post(`/__e2e/write?run=${run.id}`, 'abc')).status).toBe(400); // no path
    expect((await post(`/__e2e/bogus?run=${run.id}`, 'abc')).status).toBe(404);
    expect(run.settled).toBe(false);
    run.close();
  });

  it('start: other tabs (and their reloads) are refused and the run goes on; only a reload of the starting tab fails it', async () => {
    const run = bridge.expectRun({ outDir: join(root, 'r3'), config: { dataset: 'object' } });
    const c = createBridgeClient({ run: run.id, base });
    expect((await c.start({ userAgent: 'a', href: 'x', tab: 'tab-a' })).ok).toBe(true);
    const busy = { ok: false, status: 409, message: 'this run is already running in another tab' };
    // a second tab, then the same second tab reloaded (it kept its token): refused both times
    expect(await c.start({ userAgent: 'b', href: 'x', tab: 'tab-b' })).toEqual(busy);
    expect(await c.start({ userAgent: 'b', href: 'x', tab: 'tab-b' })).toEqual(busy);
    // a tab without storage (no token) can never claim to be the starting tab
    expect(await c.start({ userAgent: 'c', href: 'x', tab: null })).toEqual(busy);
    expect(run.settled).toBe(false);
    const again = await c.start({ userAgent: 'a', href: 'x', tab: 'tab-a' });
    expect(again).toMatchObject({ ok: false, status: 409, message: expect.stringMatching(/reloaded during the run/) });
    expect((await run.result).error).toMatch(/reloaded during the run/);
    run.close();
    // a run started by a tab without a token: a null start is never taken for its reload
    const r2 = bridge.expectRun({ outDir: join(root, 'r3b'), config: { dataset: 'object' } });
    const c2 = createBridgeClient({ run: r2.id, base });
    expect((await c2.start({ userAgent: 'a', href: 'x', tab: null })).ok).toBe(true);
    expect(await c2.start({ userAgent: 'a', href: 'x', tab: null })).toEqual(busy);
    expect(r2.settled).toBe(false);
    r2.close();
  });

  it('page flow: a second tab and its reload are refused while the first runs; only a reload of the first fails the run', async () => {
    const storage = (): BridgedRunDeps['storage'] => {
      const m = new Map<string, string>();
      return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => void m.set(k, v) };
    };
    const page = (run: string, tab: BridgedRunDeps['storage'], runPipeline: BridgedRunDeps['runPipeline']): BridgedRunDeps => ({
      client: createBridgeClient({ run, base, retryMs: 0, warn: () => {} }),
      run,
      config: { dataset: 'object', engine: 'mock' },
      queryError: null,
      runPipeline,
      abort: () => false,
      snapshot: () => null,
      log: () => {},
      storage: tab,
      userAgent: 'ua',
      href: 'x',
      pollMs: 60_000,
    });
    const done = { ok: true } as unknown as HarnessResult;
    const run = bridge.expectRun({ outDir: join(root, 'tabs'), config: { dataset: 'object' } });
    const tab1 = storage();
    const tab2 = storage();
    let finish!: (r: HarnessResult) => void;
    const first = runBridged(page(run.id, tab1, () => new Promise((r) => (finish = r))));
    await vi.waitFor(() => expect(run.startedAt).not.toBeNull());
    const others = vi.fn(async () => done);
    expect(await runBridged(page(run.id, tab2, others))).toMatchObject({ started: false, error: 'this run is already running in another tab' });
    // the user reloads the refused tab: still another tab, the run in tab 1 is untouched
    expect(await runBridged(page(run.id, tab2, others))).toMatchObject({ started: false, error: 'this run is already running in another tab' });
    expect(others).not.toHaveBeenCalled();
    expect(run.settled).toBe(false);
    finish(done);
    expect(await first).toMatchObject({ started: true, sent: true });
    expect(await run.result).toEqual({ result: { ok: true }, error: null });
    run.close();

    // tab 1 reloaded during a run it started (jetsam): the run fails, nothing runs again
    const r2 = bridge.expectRun({ outDir: join(root, 'tabs2'), config: { dataset: 'object' } });
    const tab = storage();
    let finish2!: (r: HarnessResult) => void;
    const killed = runBridged(page(r2.id, tab, () => new Promise((r) => (finish2 = r))));
    await vi.waitFor(() => expect(r2.startedAt).not.toBeNull());
    expect(await runBridged(page(r2.id, tab, others))).toMatchObject({ started: false, error: expect.stringMatching(/reloaded during the run/) });
    expect((await r2.result).error).toMatch(/reloaded during the run/);
    expect(others).not.toHaveBeenCalled();
    // (in a real reload the old page is gone; here its late result is refused)
    finish2(done);
    expect(await killed).toMatchObject({ started: true, sent: false });
    r2.close();
  });

  it('caps the bytes a run holds on disk (overwrites count once)', async () => {
    const saved = bridge;
    bridge = createBridge({ maxRunBytes: 100 });
    try {
      const run = bridge.expectRun({ outDir: join(root, 'cap'), prefix: 'cap/', config: { dataset: 'object' } });
      const c = createBridgeClient({ run: run.id, base, chunkBytes: 1 << 20 });
      await c.writeBlob('cap/a.bin', new Blob([new Uint8Array(60)]), false);
      // overwriting a.bin replaces its 60 bytes: 60 + 30 ≤ 100
      await c.writeBlob('cap/a.bin', new Blob([new Uint8Array(60)]), false);
      await c.writeBlob('cap/b.bin', new Blob([new Uint8Array(30)]), false);
      await expect(c.writeBlob('cap/c.bin', new Blob([new Uint8Array(20)]), false)).rejects.toThrow(/HTTP 413 the run's files would exceed 100 bytes/);
      await expect(c.writeBlob('cap/b.bin', new Blob([new Uint8Array(20)]), true)).rejects.toThrow(/HTTP 413/);
      expect(Object.fromEntries(run.files)).toEqual({ 'a.bin': 60, 'b.bin': 30, 'c.bin': 0 });
      await c.writeBlob('cap/b.bin', new Blob([new Uint8Array(10)]), true);
      expect(run.files.get('b.bin')).toBe(40);
      expect(readFileSync(join(root, 'cap', 'b.bin')).length).toBe(40);
      run.close();
    } finally {
      bridge = saved;
    }
  });

  it('wait: runner timeout → abort via poll → the partial result arrives flagged as timeout', async () => {
    const lines: string[] = [];
    const run = bridge.expectRun({ outDir: join(root, 'r4'), config: { dataset: 'object' }, log: (m) => lines.push(m) });
    const c = createBridgeClient({ run: run.id, base });
    await c.start({ userAgent: 'a', href: 'x', tab: 't' });
    const waiting = run.wait({ timeoutMs: 40, tickMs: 5, graceMs: 5000 });
    // the page polls until it sees the abort, then posts its partial result
    let p = await c.poll();
    for (let i = 0; i < 200 && !p?.abort; i++) {
      await new Promise((r) => setTimeout(r, 5));
      p = await c.poll();
    }
    expect(p).toEqual({ abort: 'runner timeout' });
    await c.result({ result: { ok: false, aborted: true }, error: null });
    const out = await waiting;
    expect(out).toMatchObject({ result: { ok: false, aborted: true }, error: 'timeout' });
    expect(lines.join('\n')).toMatch(/TIMEOUT after 0 s: asking the page to abort/);
    run.close();
  });

  it('wait: no page within startTimeoutMs; a page that never answers the abort; a silent page is reported', async () => {
    const a = bridge.expectRun({ outDir: join(root, 'r5'), config: { dataset: 'object' } });
    let reminders = 0;
    expect(await a.wait({ timeoutMs: 1000, startTimeoutMs: 30, remindMs: 5, tickMs: 5, remind: () => reminders++ })).toMatchObject({ result: null, error: 'no page opened the run within 0 s' });
    expect(reminders).toBeGreaterThan(0);
    a.close();
    const lines: string[] = [];
    const b = bridge.expectRun({ outDir: join(root, 'r6'), config: { dataset: 'object' }, log: (m) => lines.push(m) });
    await createBridgeClient({ run: b.id, base }).start({ userAgent: 'a', href: 'x', tab: 't' });
    expect(await b.wait({ timeoutMs: 30, graceMs: 30, staleMs: 10, tickMs: 5 })).toMatchObject({ result: null, error: 'timeout (no result from the page after the abort)' });
    expect(lines.join('\n')).toMatch(/no contact with the page for 0 s/);
    b.close();
  });

  it('the plugin applies to the dev / preview servers only, and vite.config.ts never loads it', () => {
    const p = bridge.plugin();
    expect(p.apply).toBe('serve');
    const used: unknown[] = [];
    const fake = { middlewares: { use: (m: unknown) => used.push(m) } };
    p.configureServer(fake);
    p.configurePreviewServer(fake);
    expect(used).toEqual([bridge.middleware, bridge.middleware]);
    const viteConfig = new TextDecoder().decode(readFileSync(resolve('vite.config.ts')));
    expect(viteConfig).not.toMatch(/e2e-bridge|__e2e/);
  });
});
