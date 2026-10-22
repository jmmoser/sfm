/**
 * Query string of harness.html (pure; unit-tested in ./query.test.ts).
 *
 *   configFromQuery(q)  — the HarnessConfig of a manual or phone run, null without `dataset`:
 *     `config=<JSON>`    the whole HarnessConfig (what e2e/run-e2e.mjs --device phone sends), then
 *     single keys on top: engine, quality, mode, stride, maxFrames, start, fps, imu, gps, heading,
 *     headingNorth, densify, mesh, splat, splatIterations (implies splat), exports, mockSpeed,
 *     engineConfig=<JSON object>. Malformed JSON throws (a silent default would run the wrong
 *     configuration on a phone nobody watches).
 *   bridgeFromQuery(q)  — `bridge=1&run=<id>`: post logs, files and the result to the dev server
 *     (./bridge.ts, e2e/lib/e2e-bridge.mjs); null otherwise.
 */
import type { HarnessConfig, HarnessEngineKind } from './types';

function jsonObject(q: URLSearchParams, key: string): Record<string, unknown> | undefined {
  const s = q.get(key);
  if (s === null) return undefined;
  let v: unknown;
  try {
    v = JSON.parse(s);
  } catch (e) {
    throw new Error(`harness query: ${key} is not JSON (${(e as Error).message})`);
  }
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error(`harness query: ${key} must be a JSON object`);
  return v as Record<string, unknown>;
}

export function configFromQuery(q: URLSearchParams): HarnessConfig | null {
  const base = jsonObject(q, 'config') as Partial<HarnessConfig> | undefined;
  const dataset = q.get('dataset') ?? (typeof base?.dataset === 'string' ? base.dataset : null);
  if (!dataset) return null;
  const num = (k: string): number | undefined => (q.has(k) && Number.isFinite(Number(q.get(k))) ? Number(q.get(k)) : undefined);
  const bool = (k: string): boolean | undefined => (q.has(k) ? q.get(k) !== '0' && q.get(k) !== 'false' : undefined);
  const c: HarnessConfig = { ...base, dataset };
  // only keys present in the query override the `config` JSON
  const set = <K extends keyof HarnessConfig>(k: K, v: HarnessConfig[K] | undefined): void => {
    if (v !== undefined) c[k] = v;
  };
  const engine = q.get('engine');
  if (engine === 'worker' || engine === 'inline' || engine === 'mock') c.engine = engine as HarnessEngineKind;
  c.engine ??= 'worker';
  const quality = q.get('quality');
  if (quality === 'fast' || quality === 'balanced' || quality === 'max') c.quality = quality;
  const mode = q.get('mode');
  if (mode === 'object' || mode === 'scene' || mode === 'auto') c.mode = mode;
  set('frameStride', num('stride'));
  set('maxFrames', num('maxFrames'));
  set('startFrame', num('start'));
  set('fps', num('fps'));
  set('useImu', bool('imu'));
  set('useGps', bool('gps'));
  set('useHeading', bool('heading'));
  const north = q.get('headingNorth');
  if (north === 'magnetic' || north === 'app' || north === 'true') c.headingNorth = north;
  set('densify', bool('densify'));
  set('mesh', bool('mesh'));
  set('splat', bool('splat'));
  const its = num('splatIterations');
  if (its !== undefined && Number.isInteger(its) && its >= 1) c.splat = { ...(typeof c.splat === 'object' ? c.splat : {}), iterations: its };
  set('exports', bool('exports'));
  set('mockSpeed', num('mockSpeed'));
  const ec = jsonObject(q, 'engineConfig');
  if (ec) c.engineConfig = { ...(c.engineConfig ?? {}), ...ec };
  return c;
}

export interface BridgeQuery {
  /** Run id the runner registered (e2e/lib/e2e-bridge.mjs `expectRun`). */
  run: string;
}

export function bridgeFromQuery(q: URLSearchParams): BridgeQuery | null {
  if (q.get('bridge') !== '1') return null;
  const run = q.get('run');
  if (!run || !/^[\w-]{8,128}$/.test(run)) throw new Error('harness query: bridge=1 needs run=<id> (open the URL e2e/run-e2e.mjs --device phone prints)');
  return { run };
}
