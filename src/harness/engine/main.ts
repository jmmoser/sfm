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
 * load and shows the result summary.
 */
import { startPipeline, type RunHandle } from './pipeline';
import type { HarnessConfig, HarnessEngineKind, HarnessResult } from './types';

declare global {
  interface Window {
    runPipeline: (config: HarnessConfig) => Promise<HarnessResult>;
    harnessAbort: (reason?: string) => boolean;
    harnessSnapshot: () => HarnessResult | null;
    harnessLast?: HarnessResult;
    harnessReady?: boolean;
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
window.harnessReady = true;

// ---- manual use: harness.html?dataset=object&engine=mock&… ----
function configFromQuery(q: URLSearchParams): HarnessConfig | null {
  const dataset = q.get('dataset');
  if (!dataset) return null;
  const num = (k: string): number | undefined => (q.has(k) && Number.isFinite(Number(q.get(k))) ? Number(q.get(k)) : undefined);
  const bool = (k: string): boolean | undefined => (q.has(k) ? q.get(k) !== '0' && q.get(k) !== 'false' : undefined);
  const c: HarnessConfig = { dataset, engine: (q.get('engine') as HarnessEngineKind | null) ?? 'worker' };
  const quality = q.get('quality');
  if (quality === 'fast' || quality === 'balanced' || quality === 'max') c.quality = quality;
  const mode = q.get('mode');
  if (mode === 'object' || mode === 'scene' || mode === 'auto') c.mode = mode;
  c.frameStride = num('stride');
  c.maxFrames = num('maxFrames');
  c.startFrame = num('start');
  c.fps = num('fps');
  c.useImu = bool('imu');
  c.useGps = bool('gps');
  c.useHeading = bool('heading');
  const north = q.get('headingNorth');
  if (north === 'magnetic' || north === 'app' || north === 'true') c.headingNorth = north;
  c.densify = bool('densify');
  c.mesh = bool('mesh');
  c.exports = bool('exports');
  c.mockSpeed = num('mockSpeed');
  return c;
}

const auto = configFromQuery(new URLSearchParams(location.search));
if (auto) {
  void runPipeline(auto).then(
    (r) => {
      const out = document.getElementById('result');
      if (out) out.textContent = JSON.stringify({ ...r, frames: `${r.frames.length} frames`, progress: `${r.progress.length} events`, densePoints: r.densePoints ? `${r.densePoints.count} points` : undefined }, null, 2);
    },
    (e) => log(`[harness] failed: ${(e as Error)?.message ?? e}`),
  );
} else {
  log('engine harness ready (window.runPipeline)');
}
