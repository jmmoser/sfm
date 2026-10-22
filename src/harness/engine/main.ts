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
 * `__e2eWriteBlob`, forwards page errors and console warnings to the runner, and runs
 * ./bridge-run.ts: announce the run with this tab's token (a reload of the same tab during the
 * run fails it and nothing runs again: on iOS that is a jetsam kill; another tab is refused),
 * hold a screen wake lock (a "Keep the screen on" button when the request needs a tap), poll for
 * the runner's abort (a snapshot is sent when the pipeline does not return), post
 * `{ result, error }`.
 */
import { WakeLock } from '../../sensors/wakelock';
import { createBridgeClient, type BridgeClient } from './bridge';
import { runBridged } from './bridge-run';
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

/** An element above the result (the "Next run" link, the wake-lock button). */
function insertAboveResult(el: HTMLElement): void {
  const out = document.getElementById('result');
  out?.parentElement?.insertBefore(el, out);
}

/** "Open the next run" link after a bridged run (the runner serves the current one at /__e2e/go). */
function showNextLink(): void {
  const a = document.createElement('a');
  a.href = '/__e2e/go';
  a.textContent = 'Next run (/__e2e/go)';
  a.style.cssText = 'display:block;margin:12px 0;color:#9cf;font-size:16px';
  insertAboveResult(a);
}

let wakeButton: HTMLButtonElement | null = null;

/** The wake lock was refused without a gesture: a tap on this button requests it again (removed with the run). */
function showWakeButton(wake: WakeLock): void {
  if (wakeButton) return;
  const btn = document.createElement('button');
  wakeButton = btn;
  btn.textContent = 'Keep the screen on';
  btn.style.cssText = 'display:block;margin:12px 0;padding:10px 16px;font:600 16px system-ui,sans-serif';
  btn.addEventListener('click', () => {
    void wake.acquire().then(() => {
      log(wake.active ? '[bridge] screen wake lock held' : `[bridge] the screen wake lock was refused again (${wake.lastError}): set Auto-Lock to Never`);
      if (wake.active) btn.remove();
    });
  });
  insertAboveResult(btn);
}

function sessionStore(): Storage | null {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
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
    const { run } = bridgeFromQuery(query)!;
    const wake = new WakeLock();
    void runBridged({
      client: installBridge(run),
      run,
      config: auto,
      queryError,
      runPipeline,
      abort: (reason) => window.harnessAbort(reason),
      snapshot: () => window.harnessSnapshot(),
      log,
      storage: sessionStore(),
      userAgent: navigator.userAgent,
      href: location.href,
      wakeLock: wake,
      onWakeLockFailed: () => showWakeButton(wake),
    }).then((o) => {
      wakeButton?.remove();
      if (o.result) showResult(o.result);
      showNextLink();
    });
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
