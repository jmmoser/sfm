/**
 * Engine harness page (harness.html): replays a synthetic dataset (e2e/data/<name>/, served by
 * Vite at /e2e/data/<name>/) through the engine and returns a JSON result.
 *
 *   window.runPipeline(config)  → Promise<HarnessResult>   (see ./types.ts, ./pipeline.ts)
 *   window.harnessAbort(reason) → abort the running pipeline (partial result is returned)
 *   window.harnessSnapshot()    → result so far (for runner timeouts)
 *   window.harnessReady === true once loaded
 *
 * Driven headlessly by e2e/run-e2e.mjs, which installs `window.__e2eLog(msg)` (progress to
 * stdout) and `window.__e2eWrite(path, base64, encoding, append)` (PLY streaming). Opened in a
 * browser by hand, `harness.html?dataset=object&engine=mock&maxFrames=40&densify=1` runs on
 * load and shows the result summary (query keys: ./query.ts).
 *
 * Phone runs (`e2e/run-e2e.mjs --device phone`): `?bridge=1&run=<id>&config=<JSON>` with no
 * Playwright binding installs the fetch bridge (./bridge.ts) as `__e2eLog` / `__e2eWrite` /
 * `__e2eWriteBlob`, forwards page errors and console warnings to the runner, announces the run
 * (a reload of the same tab during the run is reported and nothing runs again: on iOS that is a
 * jetsam kill), polls for the runner's abort, and posts `{ result, error }` at the end.
 */
import { createBridgeClient, type BridgeClient } from './bridge';
import { startPipeline, writeBinary, type RunHandle } from './pipeline';
import { bridgeFromQuery, configFromQuery } from './query';
import { replaySplat, type ReplaySplatConfig, type ReplaySplatResult } from './replay';
import type { HarnessConfig, HarnessResult } from './types';

declare global {
  interface Window {
    runPipeline: (config: HarnessConfig) => Promise<HarnessResult>;
    harnessAbort: (reason?: string) => boolean;
    harnessSnapshot: () => HarnessResult | null;
    harnessLast?: HarnessResult;
    harnessReady?: boolean;
    /** Experiments: the splat stage on a dumped session (./replay.ts, e2e/splat-replay.mjs). */
    replaySplat: (config: ReplaySplatConfig) => Promise<ReplaySplatResult>;
  }
}

const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');
const LOG_LINES = 400;
/** Poll period of a bridged run (heartbeat + abort). */
const POLL_MS = 5000;

function log(msg: string): void {
  if (window.__e2eLog) void window.__e2eLog(msg);
  else console.log(msg);
  if (statusEl) statusEl.textContent = msg;
  if (logEl) {
    logEl.textContent += `${msg}\n`;
    const lines = logEl.textContent!.split('\n');
    if (lines.length > LOG_LINES) logEl.textContent = lines.slice(-LOG_LINES).join('\n');
  }
}

let current: RunHandle | null = null;

async function runPipeline(config: HarnessConfig): Promise<HarnessResult> {
  if (current) throw new Error('a pipeline is already running');
  const handle = startPipeline(config, log);
  current = handle;
  try {
    const r = await handle.done;
    window.harnessLast = r;
    log(`[harness] finished: ok=${r.ok}${r.errors.length ? `, ${r.errors.length} error(s)` : ''}, total ${Math.round(r.timings.total ?? 0)} ms`);
    return r;
  } finally {
    current = null;
  }
}

window.runPipeline = runPipeline;
window.harnessAbort = (reason?: string): boolean => {
  if (!current) return false;
  current.abort(reason);
  return true;
};
window.harnessSnapshot = (): HarnessResult | null => current?.snapshot() ?? window.harnessLast ?? null;
window.replaySplat = (config: ReplaySplatConfig): Promise<ReplaySplatResult> => replaySplat(config, log, writeBinary);

// ---- fetch bridge (phone runs) ----
function str(a: unknown): string {
  if (a instanceof Error) return `${a.name}: ${a.message}`;
  if (typeof a === 'string') return a;
  try {
    return JSON.stringify(a) ?? String(a);
  } catch {
    return String(a);
  }
}

function installBridge(run: string): BridgeClient {
  // created first: its own failures go to the original console.warn, not back into the bridge
  const b = createBridgeClient({ run });
  window.__e2eLog = b.log;
  window.__e2eWrite = b.write;
  window.__e2eWriteBlob = b.writeBlob;
  // nobody watches the phone's console: send what would land there
  for (const level of ['error', 'warn'] as const) {
    const orig = console[level].bind(console);
    console[level] = (...args: unknown[]): void => {
      orig(...args);
      b.log(`[page:${level}] ${args.map(str).join(' ')}`);
    };
  }
  window.addEventListener('error', (e) => b.log(`[page:exception] ${e.message} (${e.filename}:${e.lineno})`));
  window.addEventListener('unhandledrejection', (e) => b.log(`[page:exception] unhandled rejection: ${str(e.reason)}`));
  return b;
}

function showResult(r: HarnessResult): void {
  const out = document.getElementById('result');
  if (out) out.textContent = JSON.stringify({ ...r, frames: `${r.frames.length} frames`, progress: `${r.progress.length} events`, densePoints: r.densePoints ? `${r.densePoints.count} points` : undefined }, null, 2);
}

/** "Open the next run" link after a bridged run (the runner serves the current one at /__e2e/go). */
function showNextLink(): void {
  const out = document.getElementById('result');
  if (!out?.parentElement) return;
  const a = document.createElement('a');
  a.href = '/__e2e/go';
  a.textContent = 'Next run (/__e2e/go)';
  a.style.cssText = 'display:block;margin:12px 0;color:#9cf;font-size:16px';
  out.parentElement.insertBefore(a, out);
}

async function runBridged(b: BridgeClient, run: string, config: HarnessConfig | null, queryError: string | null): Promise<void> {
  // a reload of this tab keeps sessionStorage: tell the runner, which then fails the run (jetsam)
  const key = 'sfm-e2e-bridge-run';
  let reload = false;
  try {
    reload = sessionStorage.getItem(key) === run;
    sessionStorage.setItem(key, run);
  } catch {
    /* storage blocked: a reload looks like a new tab (the runner then only times out) */
  }
  const st = await b.start({ userAgent: navigator.userAgent, href: location.href, reload }).catch((e: unknown) => ({ ok: false as const, status: 0, message: (e as Error)?.message ?? String(e) }));
  if (!st.ok) {
    log(`[bridge] the runner did not accept this page (${st.status || 'network error'}): ${st.message}`);
    showNextLink();
    return;
  }
  log(`[bridge] connected (run ${run.slice(0, 8)}…); keep this tab in front and the screen on until the result is sent`);
  let result: HarnessResult | null = null;
  let error: string | null = queryError;
  const timer = setInterval(() => {
    void b.poll().then((p) => {
      if (p?.abort && current) window.harnessAbort(p.abort);
    });
  }, POLL_MS);
  try {
    if (config) {
      result = await runPipeline(config);
      showResult(result);
    } else error ??= 'harness.html: the query has no dataset';
  } catch (e) {
    error = `runPipeline failed: ${(e as Error)?.message ?? e}`;
    log(`[harness] ${error}`);
  } finally {
    clearInterval(timer);
  }
  try {
    await b.result({ result, error });
    log('[bridge] result sent to the runner');
  } catch (e) {
    log(`[bridge] could not send the result: ${(e as Error)?.message ?? e}`);
  }
  showNextLink();
}

// ---- autorun from the query: harness.html?dataset=object&engine=mock&… ----
const query = new URLSearchParams(location.search);
let auto: HarnessConfig | null = null;
let queryError: string | null = null;
try {
  auto = configFromQuery(query);
} catch (e) {
  queryError = (e as Error).message;
}
// bridge=1 never falls back to a local run: its results would reach nobody
const bridgeMode = !window.__e2eWrite && query.get('bridge') === '1';
if (bridgeMode) {
  try {
    const bq = bridgeFromQuery(query)!;
    void runBridged(installBridge(bq.run), bq.run, auto, queryError);
  } catch (e) {
    log(`[bridge] ${(e as Error).message}`);
  }
}
window.harnessReady = true;
if (!bridgeMode) {
  if (queryError) log(`[harness] ${queryError}`);
  else if (auto) void runPipeline(auto).then(showResult, (e) => log(`[harness] failed: ${(e as Error)?.message ?? e}`));
  else log('engine harness ready (window.runPipeline)');
}
