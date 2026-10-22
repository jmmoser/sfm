/**
 * The page side of a bridged run (harness.html?bridge=1&run=<id>, e2e/run-e2e.mjs --device phone |
 * local-bridge; unit-tested in ./bridge-run.test.ts). main.ts wires it to the real page: the fetch
 * client (./bridge.ts), sessionStorage, runPipeline / harnessAbort / harnessSnapshot and a screen
 * wake lock (src/sensors/wakelock.ts). In order:
 *
 *  1. Announce the run (POST /__e2e/start) with this tab's token: a random id kept in
 *     sessionStorage, which survives a reload of the tab but is not shared with a new tab. The
 *     runner (e2e/lib/e2e-bridge.mjs) keeps the token of the tab that started the run: a later
 *     start with the same token is that tab reloaded during the run (on iOS, a tab jetsam killed)
 *     and fails the run; a start from any other tab (a second tap on the link) is refused and the
 *     run goes on. A refused page runs nothing: its results would reach nobody.
 *  2. Keep the screen on. iOS Auto-Lock would otherwise lock the phone mid-run and Safari suspends
 *     the tab. Whether WebKit wants a user gesture for the request is unverified
 *     (docs/research/ios-camera.md §6), so it is requested at once without waiting for it, and
 *     when it fails `onWakeLockFailed` lets the page offer a button and the runner is told. The
 *     lock's state when the result goes out is recorded in `result.device.wakeLock`.
 *  3. Run the pipeline, polling GET /__e2e/poll every pollMs (heartbeat). When the runner asks to
 *     abort (its --timeout-min), call harnessAbort; if the pipeline still has not returned
 *     snapshotMs later (an engine call that ignores the abort), post harnessSnapshot() with the
 *     desktop runner's 'timeout (snapshot of the unfinished run)'. SNAPSHOT_MS + POLL_MS stays
 *     inside the runner's grace period (run.wait graceMs, 120 s), after which it gives up.
 *  4. Post { result, error } (a query error is posted as the error and nothing runs), then release
 *     the wake lock.
 * Never throws.
 */
import type { BridgeClient, StartAnswer } from './bridge';
import type { HarnessConfig, HarnessResult } from './types';

/** Poll period (heartbeat + abort). */
export const POLL_MS = 5000;
/** After the runner's abort, how long the pipeline may take to return before a snapshot is posted. */
export const SNAPSHOT_MS = 90_000;
/** sessionStorage key of this tab's token. */
export const TAB_KEY = 'sfm-e2e-bridge-tab';

/** What runBridged needs of src/sensors/wakelock.ts WakeLock. */
export interface WakeLockLike {
  acquire(): Promise<void>;
  release(): Promise<void>;
  readonly active: boolean;
  readonly lastError: string | null;
}

export interface BridgedRunDeps {
  client: Pick<BridgeClient, 'start' | 'poll' | 'result'>;
  /** The run id of the URL. */
  run: string;
  /** configFromQuery's config; null with a query error or without a dataset. */
  config: HarnessConfig | null;
  /** configFromQuery's error: posted as the run's error, nothing runs. */
  queryError: string | null;
  runPipeline(config: HarnessConfig): Promise<HarnessResult>;
  /** window.harnessAbort: false when no pipeline runs. */
  abort(reason: string): boolean;
  /** window.harnessSnapshot. */
  snapshot(): HarnessResult | null;
  log(msg: string): void;
  /** sessionStorage; null (or throwing) when blocked: a reload then looks like another tab. */
  storage: Pick<Storage, 'getItem' | 'setItem'> | null;
  userAgent: string;
  href: string;
  wakeLock?: WakeLockLike;
  /** The wake lock request failed (e.g. offer a button: a tap is a user gesture). */
  onWakeLockFailed?(error: string): void;
  /** Token of a new tab (default: 128 random bits from crypto, not the seeded Rng: every tab must differ). */
  newToken?(): string;
  pollMs?: number;
  snapshotMs?: number;
}

export interface BridgedRunOutcome {
  /** The runner accepted this tab. */
  started: boolean;
  /** The result reached the runner. */
  sent: boolean;
  /** What was posted (null when refused). */
  result: HarnessResult | null;
  error: string | null;
}

type Envelope = { result: HarnessResult | null; error: string | null };

const message = (e: unknown): string => (e as Error)?.message ?? String(e);

function randomToken(): string {
  const b = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
}

/** This tab's token: kept in sessionStorage (created on the first visit), null when storage is blocked. */
export function tabToken(storage: BridgedRunDeps['storage'], newToken: () => string = randomToken): string | null {
  if (!storage) return null;
  try {
    const t = storage.getItem(TAB_KEY);
    if (t) return t;
    const n = newToken();
    storage.setItem(TAB_KEY, n);
    return n;
  } catch {
    return null;
  }
}

export async function runBridged(d: BridgedRunDeps): Promise<BridgedRunOutcome> {
  const pollMs = d.pollMs ?? POLL_MS;
  const snapshotMs = d.snapshotMs ?? SNAPSHOT_MS;
  const tab = tabToken(d.storage, d.newToken);
  const st = await d.client.start({ userAgent: d.userAgent, href: d.href, tab }).catch((e: unknown): StartAnswer => ({ ok: false, status: 0, message: message(e) }));
  if (!st.ok) {
    d.log(`[bridge] the runner did not accept this page (${st.status || 'network error'}): ${st.message}`);
    return { started: false, sent: false, result: null, error: st.message };
  }
  d.log(`[bridge] connected (run ${d.run.slice(0, 8)}…); keep this tab in front and the screen on until the result is sent`);

  const wl = d.wakeLock;
  let finished = false;
  if (wl) {
    void wl.acquire().then(() => {
      if (wl.active || finished) return;
      const why = wl.lastError ?? 'not granted';
      d.log(`[bridge] WARNING: no screen wake lock (${why}): tap "Keep the screen on", or set Settings > Display & Brightness > Auto-Lock > Never; a locked phone suspends the run`);
      d.onWakeLockFailed?.(why);
    });
  }

  // the snapshot posted when the pipeline does not return after the runner's abort
  let late!: (env: Envelope) => void;
  const snapshotted = new Promise<Envelope>((r) => (late = r));
  let snapshotTimer: ReturnType<typeof setTimeout> | null = null;
  let aborting = false;
  const poller = setInterval(() => {
    void d.client.poll().then((p) => {
      if (!p?.abort || aborting) return;
      aborting = true;
      d.log(`[bridge] the runner asks to abort (${p.abort})`);
      d.abort(p.abort);
      snapshotTimer = setTimeout(() => {
        d.log(`[bridge] the pipeline did not return ${Math.round(snapshotMs / 1000)} s after the abort: sending a snapshot`);
        late({ result: d.snapshot(), error: 'timeout (snapshot of the unfinished run)' });
      }, snapshotMs);
    });
  }, pollMs);

  const ran = (async (): Promise<Envelope> => {
    if (!d.config) return { result: null, error: d.queryError ?? 'harness.html: the query has no dataset' };
    try {
      return { result: await d.runPipeline(d.config), error: null };
    } catch (e) {
      const error = `runPipeline failed: ${message(e)}`;
      d.log(`[harness] ${error}`);
      return { result: null, error };
    }
  })();
  const env = await Promise.race([ran, snapshotted]);
  clearInterval(poller);
  if (snapshotTimer !== null) clearTimeout(snapshotTimer);

  let result = env.result;
  if (result && wl) result = { ...result, device: result.device ? { ...result.device, wakeLock: { active: wl.active, error: wl.lastError } } : result.device };
  let sent = false;
  try {
    await d.client.result({ result, error: env.error });
    sent = true;
    d.log('[bridge] result sent to the runner');
  } catch (e) {
    d.log(`[bridge] could not send the result: ${message(e)}`);
  }
  finished = true;
  await wl?.release();
  return { started: true, sent, result, error: env.error };
}
