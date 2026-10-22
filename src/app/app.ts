/**
 * App shell: owns the engine (EngineClient or MockEngine), the session store, the shared
 * capture configuration and results, and hosts the screens behind the pure navigation state
 * machine of router.ts. Single document, no History API (see router.ts).
 *
 * Also owns the run journal (run-journal.ts): at start it classifies a long run that never ended
 * (killed while visible / stopped in the background) for Home's "run interrupted" card, and it
 * feeds the journal the page's visibility (visibilitychange, pagehide / pageshow) while a run is open.
 */
import type { PointCloud, TriangleMesh } from '../core/types';
import { SessionStore, type SessionMeta, type SessionPatch } from '../io/storage';
import { isAppleMobileWebKit } from '../sensors/camera';
import { parseConfig } from './config';
import { requestCapturePermissions, type CapturePermissionStatus } from '../sensors/permissions';
import { WakeLock } from '../sensors/wakelock';
import type { CaptureMode, LiveCamera, QualityPreset, SparseResult, SplatResult } from '../workers/protocol';
import { isQuotaError } from './capture-policy';
import { closeAllModals, initLayers } from './components/layers';
import { toast } from './components/toast';
import { h, reducedMotion } from './dom';
import { ABORT_REASON_PREFIX, type EngineLike, type EngineStatus } from './engine-client';
import { detectEnvironment, probeWebGPU, type EnvInfo } from './env';
import { defaultSessionName } from './format';
import { nextRoute, type NavEvent, type Route } from './router';
import { Store } from './store';
import { createCaptureScreen } from './screens/capture';
import { createDiagnosticsScreen } from './screens/diagnostics';
import { createHomeScreen } from './screens/home';
import { createProcessingScreen } from './screens/processing';
import { createSetupScreen } from './screens/setup';
import { createViewerScreen } from './screens/viewer';
import { RunJournal } from './run-journal';
import { prefetchViewer } from './viewer-loader';

export interface Screen {
  readonly el: HTMLElement;
  /** Called after the element is attached. */
  enter?(): void;
  /** Called before the element is removed; must release cameras, viewers, timers. */
  leave(): void;
}

export interface CaptureConfig {
  mode: CaptureMode;
  quality: QualityPreset;
  useGps: boolean;
  torch: boolean;
  /** Build the dense cloud right after capture. */
  autoDense: boolean;
  /** Show the engine diagnostics panel while capturing. */
  debugOverlay: boolean;
  /**
   * Grab camera frames as VideoFrames (no main-thread copy; the engine converts them) instead of
   * createImageBitmap(<video>), which blocks the main thread ~6–16 ms per grab on iOS. Default ON
   * on iOS WebKit, where the raw sensor orientation of these frames is verified (Diagnostics
   * camera timing probe); off elsewhere.
   */
  videoFrameGrab: boolean;
  /** Capture shows the live point cloud full screen, following the phone, instead of the camera preview. */
  cloudView: boolean;
  /**
   * Version of the stored defaults. Options saved before version 2 kept `videoFrameGrab: false`
   * from when it was opt-in: they get the current default once.
   */
  configVersion: number;
}

/** Large data shared between screens (kept out of the reactive store). */
export interface Results {
  sessionId: string | null;
  name: string;
  mode: CaptureMode;
  quality: QualityPreset;
  sparse: SparseResult | null;
  dense: PointCloud | null;
  mesh: TriangleMesh | null;
  /** The stored Gaussian-splat model's info (the model itself: engine.splatModel()); null when none. */
  splat: SplatResult | null;
  /** Last live sparse model / keyframe cameras of the capture (processing preview). */
  liveSparse: { positions: Float32Array; colors: Uint8Array } | null;
  liveCameras: LiveCamera[] | null;
  stats: Record<string, number | string>;
  /**
   * Set by ensureSessionLoaded: the scan has keyframes its stored model does not cover (loadSession
   * status 'capturing' / 'captured' with a model — a continued capture not processed yet). It is to
   * be processed, not shown as built: its old model is only the preview (`liveSparse`), not `sparse`.
   * Cleared by the refine.
   */
  toProcess?: { status: 'capturing' | 'captured'; numFrames: number };
}

export interface AppState {
  route: Route;
  env: EnvInfo;
  engine: EngineStatus;
}

export interface AppOptions {
  engine: EngineLike;
  mock?: boolean;
  /** Open this route first (dev / tests: `?screen=diagnostics`). */
  initialEvent?: NavEvent | null;
}

const CONFIG_KEY = 'sfm.captureConfig.v1';

function loadConfig(): CaptureConfig {
  let raw: string | null = null;
  try {
    raw = localStorage.getItem(CONFIG_KEY);
  } catch {
    /* storage unavailable */
  }
  return parseConfig(raw, isAppleMobileWebKit());
}

function newSessionId(): string {
  const c = (globalThis as { crypto?: Crypto }).crypto;
  if (c && typeof c.randomUUID === 'function') return c.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

export class App {
  readonly root: HTMLElement;
  readonly engine: EngineLike;
  readonly mock: boolean;
  readonly store: Store<AppState>;
  readonly wakeLock = new WakeLock();
  /** The long run in progress (localStorage): written by Processing / the viewer's photoreal build. */
  readonly journal = RunJournal.local();
  config: CaptureConfig;
  results: Results;
  /** Session whose state the engine currently holds (init or loadSession). */
  loadedSessionId: string | null = null;
  /**
   * Debug / e2e hooks installed by screens (e.g. hold frame delivery while a headless test
   * takes a screenshot: software GPU emulation starves the compositor while frames are grabbed).
   */
  readonly debug: {
    holdFrames?: (hold: boolean) => void;
    /** Last Diagnostics benchmark report (e2e). */
    benchReport?: unknown;
    /** The viewer screen's 3D view (e2e screenshots / measurements); undefined elsewhere. */
    viewer?: unknown;
  } = {};
  /**
   * A long engine job runs outside the Processing screen (the viewer's photoreal build): leaving
   * the page asks first, like during capture / processing.
   */
  busy = false;
  /** Bumped by each ensureSessionLoaded and by what supersedes it (new scan, release, delete): a stale load is not applied. */
  private loadSeq = 0;
  private permissions: Promise<CapturePermissionStatus> | null = null;
  private sessionsP: Promise<SessionStore | null> | null = null;
  private screen: Screen | null = null;
  private readonly host: HTMLElement;

  constructor(root: HTMLElement, opts: AppOptions) {
    this.root = root;
    this.engine = opts.engine;
    this.mock = !!opts.mock;
    // a run still open in the journal was stopped by a kill / eviction: Home reports it
    this.journal.recover();
    this.config = loadConfig();
    this.results = App.emptyResults(this.config);
    const env = detectEnvironment();
    this.store = new Store<AppState>({ route: { screen: 'home' }, env, engine: this.engine.status });
    root.classList.add('app');
    if (this.mock) root.classList.add('is-mock');
    this.host = h('div', { class: 'screen-host' });
    root.appendChild(this.host);
    initLayers(root);

    this.engine.onStatus((s) => {
      this.store.set({ engine: s });
      // A restart the user asked for (abortLongOperation of a stuck step) is not an error.
      if (s.state === 'respawning' && !s.reason?.startsWith(ABORT_REASON_PREFIX)) toast('The processing engine restarted after an error. Recovering…', { tone: 'warn' });
      else if (s.state === 'dead') toast('The processing engine stopped after repeated errors. Reload the app to continue.', { tone: 'bad', durationMs: 9000 });
    });
    this.engine.onError((e) => {
      if (e.fatal) console.warn('[app] engine fatal error', e.error);
      if (isQuotaError(e.error)) {
        toast('Storage is full — free up space on this device, or delete old scans.', { tone: 'bad', durationMs: 9000 });
      } else if (e.reason === 'storage') {
        toast(`Could not save to on-device storage: ${e.error.message}`, { tone: 'bad', durationMs: 9000 });
      } else if (e.reason === 'out-of-memory' && this.route.screen !== 'capture') {
        // The capture screen asks "finish scan now?" itself (its own prompt); elsewhere: say it.
        toast(`Memory limit reached: ${e.error.message}`, { tone: 'warn', durationMs: 7000 });
      }
    });

    // Workers cannot see page visibility; iOS drops GPU work submitted while hidden.
    document.addEventListener('visibilitychange', () => {
      this.engine.setHidden(document.hidden);
      // a run stopped while hidden was evicted, not killed in front of the user
      this.journal.setVisible(!document.hidden);
    });
    window.addEventListener('pagehide', (e: PageTransitionEvent) => this.journal.pageHide(e.persisted));
    window.addEventListener('pageshow', () => this.journal.pageShow());
    // Desktop / Android: the browser Back button or closing the tab would silently drop a running
    // capture or processing job (there is no History API state to go back to) — ask first.
    // (iOS Safari shows no beforeunload prompt; the capture itself is persisted as it goes.)
    window.addEventListener('beforeunload', (e: BeforeUnloadEvent) => {
      const screen = this.route.screen;
      if (screen !== 'capture' && screen !== 'processing' && !this.busy) return;
      e.preventDefault();
      e.returnValue = '';
    });

    void probeWebGPU().then((r) => {
      const cur = this.store.get().env;
      this.store.set({ env: { ...cur, webgpu: r.state, webgpuAdapter: r.adapter } });
    });

    this.show({ screen: 'home' });
    if (opts.initialEvent) this.navigate(opts.initialEvent);
    prefetchViewer();
  }

  static emptyResults(cfg: Pick<CaptureConfig, 'mode' | 'quality'>): Results {
    return {
      sessionId: null,
      name: defaultSessionName(cfg.mode),
      mode: cfg.mode,
      quality: cfg.quality,
      sparse: null,
      dense: null,
      mesh: null,
      splat: null,
      liveSparse: null,
      liveCameras: null,
      stats: {},
    };
  }

  get route(): Route {
    return this.store.get().route;
  }

  get env(): EnvInfo {
    return this.store.get().env;
  }

  // ---- navigation ------------------------------------------------------------------------------

  /** Apply a navigation event; false if not allowed in the current route. */
  navigate(ev: NavEvent): boolean {
    const next = nextRoute(this.route, ev);
    if (!next) {
      console.warn('[app] navigation not allowed', this.route.screen, ev.type);
      return false;
    }
    this.show(next);
    return true;
  }

  private show(route: Route): void {
    closeAllModals();
    const prev = this.screen;
    this.screen = null;
    if (prev) {
      try {
        prev.leave();
      } catch (e) {
        console.error('[app] screen leave failed', e);
      }
    }
    this.store.set({ route });
    const screen = this.createScreen(route);
    this.screen = screen;
    const el = screen.el;
    el.classList.add('screen');
    el.dataset.screen = route.screen;
    const animate = !!prev && !reducedMotion();
    // Focus was in the old screen (the tapped button): move it to the new one afterwards, so
    // keyboard / VoiceOver users are not left on a removed element (focus falls to <body>).
    const hadFocus = !!prev && (prev.el.contains(document.activeElement) || document.activeElement === document.body);
    if (prev) {
      const old = prev.el;
      // The fading old screen must not take taps, keyboard focus or VoiceOver attention.
      old.inert = true;
      old.setAttribute('aria-hidden', 'true');
      if (animate) {
        old.classList.add('screen-exit');
        setTimeout(() => old.remove(), 180);
      } else old.remove();
    }
    if (animate) {
      el.classList.add('screen-enter');
      // Two frames so the start state is painted, plus a timer fallback: rAF does not run while
      // the compositor is starved or the page is hidden, which left the new screen invisible.
      const reveal = (): void => el.classList.remove('screen-enter');
      requestAnimationFrame(() => requestAnimationFrame(reveal));
      setTimeout(reveal, 120);
    }
    this.host.appendChild(el);
    document.title = titleFor(route);
    try {
      screen.enter?.();
    } catch (e) {
      console.error('[app] screen enter failed', e);
      toast(`Could not open this screen: ${(e as Error)?.message ?? e}`, { tone: 'bad' });
    }
    if (hadFocus) focusScreen(el);
  }

  private createScreen(route: Route): Screen {
    switch (route.screen) {
      case 'home':
        return createHomeScreen(this);
      case 'setup':
        return createSetupScreen(this);
      case 'capture':
        return createCaptureScreen(this, route);
      case 'processing':
        return createProcessingScreen(this, route);
      case 'viewer':
        return createViewerScreen(this, route);
      case 'diagnostics':
        return createDiagnosticsScreen(this, route);
    }
  }

  // ---- capture configuration / permissions -------------------------------------------------------

  saveConfig(patch: Partial<CaptureConfig>): void {
    this.config = { ...this.config, ...patch };
    try {
      localStorage.setItem(CONFIG_KEY, JSON.stringify(this.config));
    } catch {
      /* ignore */
    }
  }

  /**
   * MUST be called synchronously inside the Start tap (no await before it): motion permission
   * first (iOS requires the gesture), then the camera; also the wake lock and persistent
   * storage, which like a gesture. Geolocation is requested by the capture screen right after
   * the camera started (GeoSensor), so a slow first fix never delays the camera.
   */
  requestCapturePermissions(): Promise<CapturePermissionStatus> {
    const p = requestCapturePermissions({ geolocation: false });
    void this.wakeLock.acquire();
    try {
      const st = navigator.storage;
      if (st && typeof st.persist === 'function') void st.persist().catch(() => false);
    } catch {
      /* ignore */
    }
    this.permissions = p;
    return p;
  }

  /** The permission request made by the Start tap (consumed once). */
  takePermissions(): Promise<CapturePermissionStatus> | null {
    const p = this.permissions;
    this.permissions = null;
    return p;
  }

  /** Begin a new scan from the Setup screen's tap handler. */
  startNewScan(): void {
    const id = this.env.indexedDB ? newSessionId() : null;
    this.results = App.emptyResults(this.config);
    this.results.sessionId = id;
    this.loadedSessionId = null;
    this.loadSeq++;
    this.navigate({ type: 'START_CAPTURE', sessionId: id });
  }

  /**
   * Continue the capture of the stored scan `sessionId` ("Add more views" / "Continue scanning";
   * from the tap that called requestCapturePermissions). The page lets go of the scan's dense
   * cloud, mesh and splat info (the capture needs the memory and the next refine replaces them; a
   * capture stopped without new views loads the scan again); its sparse model stays for the
   * capture's preview.
   */
  startExtendScan(sessionId: string): boolean {
    if (!nextRoute(this.route, { type: 'EXTEND_CAPTURE', sessionId })) return false;
    this.loadSeq++; // a load still running is superseded by the capture's engine init
    if (this.results.sessionId === sessionId) {
      this.results.dense = null;
      this.results.mesh = null;
      this.results.splat = null;
      delete this.results.toProcess;
    } else {
      this.results = { ...App.emptyResults(this.config), sessionId };
      this.loadedSessionId = null;
    }
    return this.navigate({ type: 'EXTEND_CAPTURE', sessionId });
  }

  // ---- sessions -----------------------------------------------------------------------------------

  sessions(): Promise<SessionStore | null> {
    if (!this.sessionsP) {
      this.sessionsP = SessionStore.isSupported()
        ? SessionStore.open().catch((e) => {
            console.warn('[app] IndexedDB unavailable', e);
            this.sessionsP = null;
            return null;
          })
        : Promise.resolve(null);
    }
    return this.sessionsP;
  }

  /**
   * Create the session record of a new capture. Never throws: 'quota' when the origin is out of
   * storage (the capture cannot be saved — the caller blocks it), 'error' / 'unavailable' when
   * IndexedDB failed or is missing (the caller warns and continues).
   */
  async createSession(
    id: string,
    init: { name: string; mode: CaptureMode; quality: QualityPreset },
  ): Promise<{ status: 'ok'; meta: SessionMeta } | { status: 'quota' | 'error' | 'unavailable'; error?: unknown }> {
    const s = await this.sessions();
    if (!s) return { status: 'unavailable' };
    try {
      return { status: 'ok', meta: await s.createSession({ id, name: init.name, mode: init.mode, status: 'capturing', stats: { quality: init.quality } }) };
    } catch (e) {
      console.warn('[app] createSession failed', e);
      return { status: isQuotaError(e) ? 'quota' : 'error', error: e };
    }
  }

  /** Bytes still available to this origin (null when unknown). */
  async storageAvailable(): Promise<number | null> {
    const s = await this.sessions();
    const est = s ? await s.estimate() : null;
    return est && est.quota > 0 ? est.available : null;
  }

  /** Patch a session's metadata (stats are merged). Never throws. */
  async updateSession(id: string | null, patch: SessionPatch): Promise<void> {
    if (!id) return;
    const s = await this.sessions();
    if (!s) return;
    try {
      if (patch.stats) {
        const cur = await s.getSession(id);
        if (!cur) return;
        patch = { ...patch, stats: { ...(cur.stats ?? {}), ...patch.stats } };
      }
      await s.updateSession(id, patch);
    } catch (e) {
      if ((e as Error)?.name !== 'SessionNotFoundError') console.warn('[app] updateSession failed', e);
    }
  }

  async deleteSession(id: string): Promise<boolean> {
    const s = await this.sessions();
    if (!s) return false;
    // (a load still running may be of this scan: it is not applied, and its failure is not shown)
    this.loadSeq++;
    // The engine lets go of the scan first: none of its writes may land after the deletion, and no
    // later engine restart may replay its init (`resume` would re-create the deleted scan).
    await this.engine.closeSession(id).catch(() => undefined);
    if (this.loadedSessionId === id) this.loadedSessionId = null;
    this.journal.forget(id);
    try {
      await s.deleteSession(id);
      if (this.results.sessionId === id) this.results = App.emptyResults(this.config);
      return true;
    } catch (e) {
      toast(`Could not delete the scan: ${(e as Error)?.message ?? e}`, { tone: 'bad' });
      return false;
    }
  }

  /**
   * True while the engine holds a scan in memory: a capture or an opened scan (`loadedSessionId`),
   * or any initialised engine (its capture state, models and GPU pipelines stay until released).
   */
  engineHoldsData(): boolean {
    const st = this.engine.status.state;
    return this.loadedSessionId !== null || st === 'ready' || st === 'starting' || st === 'respawning';
  }

  /**
   * Free the engine's memory (EngineLike.release: the worker and everything in it) and the models
   * the page holds, e.g. before the on-device benchmark. Stored scans stay on the device and are
   * loaded again when opened.
   */
  async releaseEngine(): Promise<void> {
    this.loadSeq++; // a load still running ends with the release: not applied, no toast
    await this.engine.release();
    this.loadedSessionId = null;
    this.results = App.emptyResults(this.config);
  }

  /**
   * Make the engine hold `sessionId` (loadSession when it holds another one) and copy the
   * stored models into `results`. Returns false when it failed (a toast explains) or was
   * superseded before it ended (another scan opened, a new scan, a release): its result is dropped.
   * The engine's status wins over the session record's: a continued capture whose new keyframes
   * the page never recorded (killed, or a frame answered after Stop) is still processed/sparse/
   * meshed in the record — it is set back to the capture to process (`results.toProcess`).
   */
  async ensureSessionLoaded(sessionId: string | null): Promise<boolean> {
    if (!sessionId || this.loadedSessionId === sessionId) return true;
    const seq = ++this.loadSeq;
    // The engine lets go of the scan it held as soon as it loads another one, also when that fails.
    this.loadedSessionId = null;
    try {
      const sum = await this.engine.loadSession(sessionId);
      if (seq !== this.loadSeq) return false;
      this.loadedSessionId = sessionId;
      this.results = {
        ...App.emptyResults({ mode: sum.mode, quality: sum.quality ?? this.config.quality }),
        sessionId,
        name: sum.name,
        sparse: sum.sparse,
        dense: sum.dense,
        mesh: sum.mesh,
        splat: sum.splat ?? null,
        stats: sum.stats,
      };
      if ((sum.status === 'capturing' || sum.status === 'captured') && sum.sparse) {
        const r = this.results;
        r.toProcess = { status: sum.status, numFrames: sum.numFrames };
        r.liveSparse = { positions: sum.sparse.points.positions, colors: sum.sparse.points.colors };
        r.liveCameras = sum.sparse.cameras.map((c) => ({ pose: c.pose, intrinsics: c.intrinsics, ...(c.upCam ? { upCam: c.upCam } : {}) }));
        r.sparse = null;
        const cur = await this.sessions()
          .then((s) => s?.getSession(sessionId) ?? null)
          .catch(() => null);
        if (cur && cur.status !== sum.status && cur.status !== 'capturing' && cur.status !== 'captured') await this.updateSession(sessionId, { status: sum.status });
        if (seq !== this.loadSeq) return false;
      }
      return true;
    } catch (e) {
      if (seq !== this.loadSeq) return false;
      toast(`Couldn’t open the scan: ${(e as Error)?.message ?? e}`, { tone: 'bad' });
      return false;
    }
  }
}

/** Focus the screen's heading (or the screen itself) without scrolling or a visible ring. */
function focusScreen(el: HTMLElement): void {
  const target = el.querySelector<HTMLElement>('h1') ?? el;
  if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
  try {
    target.focus({ preventScroll: true });
  } catch {
    /* ignore */
  }
}

function titleFor(route: Route): string {
  const base = '3D Scan';
  switch (route.screen) {
    case 'home':
      return base;
    case 'setup':
      return `New scan · ${base}`;
    case 'capture':
      return `Capturing · ${base}`;
    case 'processing':
      return `Processing · ${base}`;
    case 'viewer':
      return `Model · ${base}`;
    case 'diagnostics':
      return `Diagnostics · ${base}`;
  }
}
