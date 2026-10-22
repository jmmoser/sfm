/**
 * Fetch transport of the e2e bridge (harness.html?bridge=1&run=<id>, unit-tested in
 * ./bridge.test.ts; the run's flow on top of it is ./bridge-run.ts). It does over HTTP what Playwright's exposed `__e2eLog` / `__e2eWrite` do for
 * desktop runs, so the harness runs unchanged in a browser nothing automates (Safari on an
 * iPhone) and e2e/run-e2e.mjs --device phone still gets the logs, the PLY files and the result.
 * The server side is the dev-server plugin e2e/lib/e2e-bridge.mjs; every request names the run:
 *
 *   POST /__e2e/start?run=   JSON { userAgent, href, tab }  → 204, or 409 (refused: the run is
 *                            already running in another tab, or this tab (same `tab` token)
 *                            reloaded mid-run)
 *   POST /__e2e/log?run=     text lines, in order: batched, one request in flight
 *   POST /__e2e/write?run=&path=&append=0|1   raw bytes (no base64), ≤ 16 MB per request
 *   GET  /__e2e/poll?run=    JSON { abort } every few seconds: heartbeat + the runner's timeout
 *   POST /__e2e/result?run=  JSON { result: HarnessResult | null, error: string | null }
 *
 * Logs are best effort (a failed batch is dropped with a console warning); a failed write rejects,
 * so the pipeline records it like any other write error; the result is retried 3 times.
 */

export type WriteFn = (path: string, data: string, encoding: 'base64' | 'utf8', append: boolean) => Promise<void>;
export type WriteBlobFn = (path: string, data: Blob, append: boolean) => Promise<void>;

export type StartAnswer = { ok: true } | { ok: false; status: number; message: string };
export interface StartInfo {
  userAgent: string;
  href: string;
  tab: string | null;
}

export interface BridgeClient {
  /** `__e2eWrite`-compatible (base64 / utf8 strings). */
  write: WriteFn;
  /** Raw bytes, chunked. */
  writeBlob: WriteBlobFn;
  /** Queue a log line (never throws; dropped once the result was posted: the run is over). */
  log(msg: string): void;
  /** Resolves when every queued line was sent (or dropped). */
  flush(): Promise<void>;
  /** `tab`: this tab's token (./bridge-run.ts), null when sessionStorage is blocked. */
  start(info: StartInfo): Promise<StartAnswer>;
  /** null when the server could not be reached. */
  poll(): Promise<{ abort: string | null } | null>;
  /** Flushes the logs, then posts the final result. */
  result(envelope: { result: unknown; error: string | null }): Promise<void>;
}

export interface BridgeClientOptions {
  run: string;
  /** Origin + path prefix of the endpoints (default '': same origin). */
  base?: string;
  fetch?: typeof fetch;
  /** Bytes per write request (default 16 MB). */
  chunkBytes?: number;
  /** Delay between result retries (default 2 s). */
  retryMs?: number;
  /** Where the client's own failures go (default console.warn, captured before any forwarding). */
  warn?: (msg: string) => void;
}

const MAX_QUEUED_LINES = 10_000;

function decodeBase64(s: string): Uint8Array<ArrayBuffer> {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export function createBridgeClient(o: BridgeClientOptions): BridgeClient {
  const f = o.fetch ?? globalThis.fetch.bind(globalThis);
  const base = o.base ?? '';
  const chunk = o.chunkBytes ?? 16 << 20;
  const retryMs = o.retryMs ?? 2000;
  const consoleWarn = console.warn.bind(console);
  const warn = o.warn ?? ((m: string) => consoleWarn(m));

  const url = (route: string, params: Record<string, string> = {}): string => {
    const q = new URLSearchParams({ run: o.run, ...params });
    return `${base}/__e2e/${route}?${q.toString()}`;
  };
  const post = async (route: string, body: BodyInit, type: string, params?: Record<string, string>): Promise<Response> => {
    const r = await f(url(route, params), { method: 'POST', body, headers: { 'content-type': type } });
    if (!r.ok) {
      const text = await r.text().catch(() => '');
      throw new Error(`bridge ${route}: HTTP ${r.status}${text ? ` ${text}` : ''}`);
    }
    return r;
  };

  let queue: string[] = [];
  let dropped = 0;
  /** The result went out: the runner forgets the run, later lines would only get 403. */
  let done = false;
  let sending: Promise<void> | null = null;
  const pump = (): void => {
    if (sending || queue.length === 0) return;
    const lines = dropped ? [`[bridge] ${dropped} log line(s) dropped (queue full)`, ...queue] : queue;
    queue = [];
    dropped = 0;
    sending = post('log', lines.join('\n'), 'text/plain;charset=utf-8').then(
      () => {
        sending = null;
        pump();
      },
      (e: unknown) => {
        warn(`[bridge] ${lines.length} log line(s) lost: ${(e as Error)?.message ?? e}`);
        sending = null;
        pump();
      },
    );
  };
  const flush = async (): Promise<void> => {
    while (sending || queue.length) {
      pump();
      await sending;
    }
  };

  const writeBlob: WriteBlobFn = async (path, data, append) => {
    const octet = 'application/octet-stream';
    if (data.size === 0) {
      await post('write', data, octet, { path, append: append ? '1' : '0' });
      return;
    }
    for (let off = 0; off < data.size; off += chunk) {
      await post('write', data.slice(off, Math.min(data.size, off + chunk)), octet, { path, append: append || off > 0 ? '1' : '0' });
    }
  };

  return {
    writeBlob,
    write: (path, data, encoding, append) => writeBlob(path, new Blob([encoding === 'base64' ? decodeBase64(data) : new TextEncoder().encode(data)]), append),
    log(msg) {
      if (done) return;
      queue.push(String(msg));
      if (queue.length > MAX_QUEUED_LINES) {
        dropped += queue.length - MAX_QUEUED_LINES;
        queue.splice(0, queue.length - MAX_QUEUED_LINES);
      }
      pump();
    },
    flush,
    async start(info) {
      const r = await f(url('start'), { method: 'POST', body: JSON.stringify(info), headers: { 'content-type': 'application/json' } });
      if (r.ok) return { ok: true };
      return { ok: false, status: r.status, message: (await r.text().catch(() => '')) || `HTTP ${r.status}` };
    },
    async poll() {
      try {
        const r = await f(url('poll'), { cache: 'no-store' });
        if (!r.ok) return null;
        const j = (await r.json()) as { abort?: unknown };
        return { abort: typeof j.abort === 'string' ? j.abort : null };
      } catch {
        return null;
      }
    },
    async result(envelope) {
      await flush();
      const body = JSON.stringify(envelope);
      let last: unknown = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        if (attempt) await new Promise((r) => setTimeout(r, retryMs));
        try {
          await post('result', body, 'application/json');
          done = true;
          return;
        } catch (e) {
          last = e;
          warn(`[bridge] posting the result failed (attempt ${attempt + 1}/3): ${(e as Error)?.message ?? e}`);
        }
      }
      throw last;
    },
  };
}
