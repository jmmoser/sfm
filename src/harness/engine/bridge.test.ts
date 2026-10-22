/**
 * The page side of the e2e bridge (./bridge.ts) against a recording fake fetch: log lines keep
 * their order with one request in flight, writes are raw chunks with the right append flags, the
 * base64 `__e2eWrite` form decodes to the same bytes, and the result goes out after the logs.
 */
import { describe, expect, it } from 'vitest';
import { createBridgeClient } from './bridge';

interface Call {
  route: string;
  params: Record<string, string>;
  method: string;
  body: string | Uint8Array | null;
}

/** Fake fetch: records each call; `respond` decides the answer; `gate` holds log requests. */
function fakeFetch(respond: (c: Call) => Response = () => new Response(null, { status: 204 })) {
  const calls: Call[] = [];
  let release: (() => void) | null = null;
  let held = false;
  const f = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const u = new URL(String(input), 'https://phone.invalid');
    const body = init?.body;
    const c: Call = {
      route: u.pathname.replace('/__e2e/', ''),
      params: Object.fromEntries(u.searchParams),
      method: init?.method ?? 'GET',
      body: body == null ? null : body instanceof Blob ? new Uint8Array(await body.arrayBuffer()) : String(body),
    };
    calls.push(c);
    if (held && c.route === 'log') await new Promise<void>((r) => (release = r));
    return respond(c);
  }) as typeof fetch;
  return {
    f,
    calls,
    hold: () => (held = true),
    release: () => {
      held = false;
      release?.();
    },
  };
}

describe('createBridgeClient', () => {
  it('logs: in order, batched behind the request in flight, flushed before the result', async () => {
    const ff = fakeFetch();
    const b = createBridgeClient({ run: 'r1', fetch: ff.f });
    ff.hold();
    b.log('one');
    await Promise.resolve();
    b.log('two');
    b.log('three');
    expect(ff.calls.filter((c) => c.route === 'log')).toHaveLength(1); // 'two', 'three' wait
    ff.release();
    await b.result({ result: { ok: true }, error: null });
    const logs = ff.calls.filter((c) => c.route === 'log').map((c) => c.body);
    expect(logs).toEqual(['one', 'two\nthree']);
    expect(ff.calls.at(-1)).toMatchObject({ route: 'result', method: 'POST', params: { run: 'r1' } });
    expect(JSON.parse(ff.calls.at(-1)!.body as string)).toEqual({ result: { ok: true }, error: null });
    // the run is over: later lines are not sent (the runner would answer 403)
    const n = ff.calls.length;
    b.log('after');
    await b.flush();
    expect(ff.calls).toHaveLength(n);
  });

  it('a failed log batch is dropped with a warning, later lines still go out', async () => {
    const warns: string[] = [];
    let n = 0;
    const ff = fakeFetch((c) => (c.route === 'log' && n++ === 0 ? new Response('nope', { status: 500 }) : new Response(null, { status: 204 })));
    const b = createBridgeClient({ run: 'r1', fetch: ff.f, warn: (m) => warns.push(m) });
    b.log('lost');
    await b.flush();
    b.log('kept');
    await b.flush();
    expect(warns.join()).toMatch(/1 log line\(s\) lost: bridge log: HTTP 500 nope/);
    expect(ff.calls.filter((c) => c.route === 'log').map((c) => c.body)).toEqual(['lost', 'kept']);
  });

  it('writeBlob: raw chunks, append=0 then 1; write() decodes base64 / utf8', async () => {
    const ff = fakeFetch();
    const b = createBridgeClient({ run: 'r1', fetch: ff.f, chunkBytes: 4 });
    const bytes = Uint8Array.from({ length: 10 }, (_, i) => i * 7);
    await b.writeBlob('obj/sparse.ply', new Blob([bytes]), false);
    const w = ff.calls.filter((c) => c.route === 'write');
    expect(w.map((c) => c.params)).toEqual([
      { run: 'r1', path: 'obj/sparse.ply', append: '0' },
      { run: 'r1', path: 'obj/sparse.ply', append: '1' },
      { run: 'r1', path: 'obj/sparse.ply', append: '1' },
    ]);
    expect(w.flatMap((c) => [...(c.body as Uint8Array)])).toEqual([...bytes]);
    ff.calls.length = 0;
    await b.writeBlob('obj/empty.bin', new Blob([]), false);
    await b.write('obj/a.bin', btoa(String.fromCharCode(...bytes)), 'base64', true);
    await b.write('obj/t.txt', 'héllo', 'utf8', false);
    expect(ff.calls.map((c) => [c.params.path, c.params.append])).toEqual([
      ['obj/empty.bin', '0'],
      ['obj/a.bin', '1'],
      ['obj/a.bin', '1'],
      ['obj/a.bin', '1'],
      ['obj/t.txt', '0'],
      ['obj/t.txt', '1'], // 6 UTF-8 bytes in 4-byte chunks
    ]);
    expect(ff.calls.slice(1, 4).flatMap((c) => [...(c.body as Uint8Array)])).toEqual([...bytes]);
    expect(new TextDecoder().decode(new Uint8Array(ff.calls.slice(4).flatMap((c) => [...(c.body as Uint8Array)])))).toBe('héllo');
  });

  it('a refused write rejects (the pipeline records it as a write error)', async () => {
    const ff = fakeFetch(() => new Response('bad path ../x', { status: 400 }));
    const b = createBridgeClient({ run: 'r1', fetch: ff.f });
    await expect(b.writeBlob('../x', new Blob([new Uint8Array(3)]), false)).rejects.toThrow(/HTTP 400 bad path/);
  });

  it('start / poll / result answers', async () => {
    const ff = fakeFetch((c) => {
      if (c.route === 'start') return new Response('this run is already running in another tab', { status: 409 });
      if (c.route === 'poll') return Response.json({ abort: 'runner timeout' });
      return new Response('unknown run', { status: 403 });
    });
    const warns: string[] = [];
    const b = createBridgeClient({ run: 'r1', fetch: ff.f, retryMs: 0, warn: (m) => warns.push(m) });
    expect(await b.start({ userAgent: 'ua', href: 'h', tab: 'tab-1' })).toEqual({ ok: false, status: 409, message: 'this run is already running in another tab' });
    expect(JSON.parse(ff.calls[0].body as string)).toEqual({ userAgent: 'ua', href: 'h', tab: 'tab-1' });
    expect(await b.poll()).toEqual({ abort: 'runner timeout' });
    await expect(b.result({ result: null, error: 'x' })).rejects.toThrow(/HTTP 403/);
    expect(ff.calls.filter((c) => c.route === 'result')).toHaveLength(3);
    expect(warns).toHaveLength(3);
    const down = createBridgeClient({ run: 'r1', fetch: (async () => Promise.reject(new TypeError('Load failed'))) as typeof fetch });
    expect(await down.poll()).toBeNull();
  });
});
