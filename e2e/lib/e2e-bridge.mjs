// Dev-server side of the e2e bridge (unit-tested in src/harness/engine/e2e-bridge.test.ts): a Vite
// plugin for the dev and the preview server that receives what the harness page sends over HTTP
// when no Playwright binding exists (src/harness/engine/bridge.ts, harness.html?bridge=1&run=<id>):
// progress lines, streamed files and the final result. e2e/run-e2e.mjs --device phone uses it to
// score a run made in a browser it does not drive (Safari on an iPhone) exactly like a desktop run.
//
//   GET  /__e2e/go                     302 to the harness URL of the run the runner waits for
//   POST /__e2e/start?run=ID           JSON { userAgent, href, reload } → 204. 409 when the run already
//                                      started: from another tab (the run goes on) or from a reload of
//                                      the same tab (`reload`: the run fails, on iOS that is a jetsam kill)
//   POST /__e2e/log?run=ID             text, '\n'-separated lines
//   POST /__e2e/write?run=ID&path=P&append=0|1
//                                      raw body streamed to <run dir>/<P without the run's prefix>
//                                      (safeJoin: absolute paths and '..' are rejected)
//   GET  /__e2e/poll?run=ID            JSON { abort } (heartbeat; carries the runner's timeout)
//   POST /__e2e/result?run=ID          JSON { result, error } → settles the run
//
// The run id is random (crypto) and known only to the runner and the page it sends the URL to:
// requests naming another id get 403. Cross-origin pages cannot read the id from /__e2e/go's
// redirect. The plugin is `apply: 'serve'` and is only added to the runner's inline createServer
// config (e2e/lib/e2e-browser.mjs startBridgeServer), never to vite.config.ts, so production
// builds never contain it.
import { randomBytes } from 'node:crypto';
import { createWriteStream } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { safeJoin } from './safe-path.mjs';

const LIMITS = { start: 64 << 10, log: 4 << 20, write: 512 << 20, result: 1 << 30 };

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Relative URL of the harness page for a run: bridge=1, the run id and the whole config. */
export function harnessPath(id, config) {
  const q = new URLSearchParams({ bridge: '1', run: id, config: JSON.stringify(config) });
  return `/harness.html?${q.toString()}`;
}

async function readBody(req, limit) {
  const chunks = [];
  let n = 0;
  for await (const c of req) {
    n += c.length;
    if (n > limit) throw new HttpError(413, `body over ${limit} bytes`);
    chunks.push(c);
  }
  return Buffer.concat(chunks);
}

function send(res, status, body = '', type = 'text/plain; charset=utf-8') {
  res.statusCode = status;
  res.setHeader('cache-control', 'no-store');
  if (status === 204) return res.end();
  res.setHeader('content-type', type);
  res.end(body);
}

/**
 * `path` as the page names it (the run's output prefix + a relative file) → the absolute file
 * inside `outDir`. Throws HttpError 400 for paths outside the prefix, absolute paths and '..'.
 */
export function resolveWritePath(outDir, prefix, path) {
  if (typeof path !== 'string' || !path.startsWith(prefix)) throw new HttpError(400, `bad path ${path} (not under ${prefix || 'the run'})`);
  try {
    return safeJoin(outDir, path.slice(prefix.length));
  } catch (e) {
    throw new HttpError(400, e.message);
  }
}

/**
 * The bridge: the runs the runner registered, the connect middleware and the Vite plugin.
 * `now` is injectable for tests.
 */
export function createBridge({ now = Date.now } = {}) {
  const runs = new Map();
  let active = null;

  /**
   * Register a run (and make it the one /__e2e/go points at).
   *   outDir   the run's output directory (files land here)
   *   prefix   the page's `output.prefix` (stripped from written paths, e.g. 'object-phone/')
   *   config   the HarnessConfig the page runs (sent in the URL)
   *   log(m)   page log lines
   * Returns the run: { id, path (relative harness URL), started: Promise<info>, result:
   * Promise<{ result, error }>, abort(reason), finish(envelope), wait(opts), close(), … }.
   */
  function expectRun({ outDir, prefix = '', config, log = () => {}, id = randomBytes(16).toString('hex') }) {
    let resolveStarted;
    let resolveResult;
    const run = {
      id,
      outDir,
      prefix,
      path: harnessPath(id, config),
      info: null,
      startedAt: null,
      lastSeen: null,
      abortReason: null,
      settled: false,
      /** Bytes per written file (relative to outDir). */
      files: new Map(),
      started: new Promise((r) => (resolveStarted = r)),
      result: new Promise((r) => (resolveResult = r)),
      log,
      onStart(info) {
        run.startedAt = now();
        run.info = info;
        resolveStarted(info);
      },
      finish(envelope) {
        if (run.settled) return false;
        run.settled = true;
        resolveResult({ result: envelope.result ?? null, error: envelope.error ?? null });
        return true;
      },
      abort(reason) {
        run.abortReason ??= reason;
      },
      close() {
        runs.delete(id);
        if (active === run) active = null;
      },
      /**
       * Wait for the run: until a page starts it (startTimeoutMs; a reminder every remindMs), then
       * until its result. After timeoutMs of running the page is asked to abort (it posts the
       * partial result); graceMs later the run fails. A page silent for staleMs (locked screen,
       * Safari in the background, a killed tab) is reported, not failed. Resolves
       * { result, error, waitMs, runMs }.
       */
      async wait({ timeoutMs, startTimeoutMs = Infinity, remindMs = 120_000, staleMs = 60_000, graceMs = 120_000, tickMs = 1000, remind = () => {} }) {
        const t0 = now();
        let reminded = t0;
        while (run.startedAt === null && !run.settled) {
          const t = now();
          if (t - t0 >= startTimeoutMs) {
            run.finish({ result: null, error: `no page opened the run within ${Math.round(startTimeoutMs / 1000)} s` });
            break;
          }
          if (t - reminded >= remindMs) {
            reminded = t;
            remind();
          }
          await Promise.race([run.started, run.result, sleep(tickMs)]);
        }
        const ts = run.startedAt ?? now();
        let abortAt = null;
        let stale = false;
        while (!run.settled) {
          const t = now();
          if (abortAt === null && t - ts >= timeoutMs) {
            const min = timeoutMs / 60_000;
            log(`TIMEOUT after ${min >= 2 ? `${Math.round(min)} min` : `${Math.round(timeoutMs / 1000)} s`}: asking the page to abort the pipeline`);
            run.abort('runner timeout');
            abortAt = t;
          }
          if (abortAt !== null && t - abortAt >= graceMs) {
            run.finish({ result: null, error: 'timeout (no result from the page after the abort)' });
            break;
          }
          const quiet = t - (run.lastSeen ?? ts);
          if (!stale && quiet >= staleMs) {
            stale = true;
            log(`WARNING: no contact with the page for ${Math.round(quiet / 1000)} s (screen locked, Safari in the background, or the tab was killed?)`);
          } else if (stale && quiet < staleMs) {
            stale = false;
            log('the page is back');
          }
          await Promise.race([run.result, sleep(tickMs)]);
        }
        const out = await run.result;
        const error = out.error ?? (abortAt !== null && out.result ? 'timeout' : null);
        return { result: out.result, error, waitMs: ts - t0, runMs: now() - ts };
      },
    };
    runs.set(id, run);
    active = run;
    return run;
  }

  async function handle(req, res) {
    const u = new URL(req.url, 'http://bridge.invalid');
    const route = u.pathname.slice('/__e2e/'.length);
    const method = req.method ?? 'GET';
    if (route === 'go') {
      if (method !== 'GET' && method !== 'HEAD') throw new HttpError(405, 'GET only');
      if (!active) {
        return send(
          res,
          404,
          '<!doctype html><meta name="viewport" content="width=device-width"><body style="font:16px system-ui;padding:16px">No e2e run is waiting. Start <code>node e2e/run-e2e.mjs &lt;dataset&gt; --device phone</code> on the computer, then reload this page.',
          'text/html; charset=utf-8',
        );
      }
      res.statusCode = 302;
      res.setHeader('cache-control', 'no-store');
      res.setHeader('location', active.path);
      return res.end();
    }
    const run = runs.get(u.searchParams.get('run') ?? '');
    if (!run) throw new HttpError(403, 'unknown run (the runner is not waiting for it any more)');
    run.lastSeen = now();
    const want = (m) => {
      if (method !== m) throw new HttpError(405, `${m} only`);
    };
    switch (route) {
      case 'start': {
        want('POST');
        let info;
        try {
          info = JSON.parse((await readBody(req, LIMITS.start)).toString('utf8') || '{}');
        } catch (e) {
          if (e instanceof HttpError) throw e;
          throw new HttpError(400, 'start: body is not JSON');
        }
        if (run.settled) return send(res, 409, 'this run is over; open /__e2e/go for the current one');
        if (run.startedAt !== null) {
          if (info.reload) {
            run.log('the page was reloaded during the run: the run is lost (tab killed for memory, or reloaded by hand)');
            run.finish({ result: null, error: 'the harness page was reloaded during the run (tab killed: out of memory? or reloaded by hand)' });
            return send(res, 409, 'reloaded during the run: this run is lost, start a new one on the computer');
          }
          return send(res, 409, 'this run is already running in another tab');
        }
        run.onStart(info);
        return send(res, 204);
      }
      case 'log': {
        want('POST');
        const text = (await readBody(req, LIMITS.log)).toString('utf8');
        for (const line of text.split('\n')) if (line) run.log(line);
        return send(res, 204);
      }
      case 'write': {
        want('POST');
        const rel = u.searchParams.get('path');
        const file = resolveWritePath(run.outDir, run.prefix, rel);
        const append = u.searchParams.get('append') === '1';
        await mkdir(dirname(file), { recursive: true });
        let n = 0;
        await pipeline(
          req,
          async function* (src) {
            for await (const c of src) {
              n += c.length;
              if (n > LIMITS.write) throw new HttpError(413, `write over ${LIMITS.write} bytes per request`);
              yield c;
            }
          },
          createWriteStream(file, { flags: append ? 'a' : 'w' }),
        );
        const key = rel.slice(run.prefix.length);
        run.files.set(key, (append ? run.files.get(key) ?? 0 : 0) + n);
        return send(res, 204);
      }
      case 'poll':
        want('GET');
        return send(res, 200, JSON.stringify({ abort: run.abortReason }), 'application/json');
      case 'result': {
        want('POST');
        let env;
        try {
          env = JSON.parse((await readBody(req, LIMITS.result)).toString('utf8'));
        } catch (e) {
          if (e instanceof HttpError) throw e;
          throw new HttpError(400, 'result: body is not JSON');
        }
        if (!env || typeof env !== 'object') throw new HttpError(400, 'result: expected { result, error }');
        if (!run.finish({ result: env.result ?? null, error: typeof env.error === 'string' ? env.error : null })) return send(res, 409, 'this run already has a result');
        return send(res, 204);
      }
      default:
        throw new HttpError(404, `unknown bridge route ${route}`);
    }
  }

  /** Connect middleware: handles /__e2e/*, passes everything else on. */
  function middleware(req, res, next) {
    if (!req.url || !req.url.startsWith('/__e2e/')) return next();
    handle(req, res).catch((e) => {
      const status = e instanceof HttpError ? e.status : 500;
      if (!res.headersSent) send(res, status, e.message ?? String(e));
      else res.end();
      // drain an unread body so the connection is usable again
      req.resume?.();
    });
  }

  return { runs, expectRun, middleware, plugin: () => bridgePlugin(middleware), get active() { return active; } };
}

/** Vite plugin: the middleware in front of the dev server's and the preview server's own. */
export function bridgePlugin(middleware) {
  return {
    name: 'sfm-e2e-bridge',
    // dev and preview servers only: `vite build` never applies it
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use(middleware);
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware);
    },
  };
}
