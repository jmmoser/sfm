/**
 * App shell: owns the engine (EngineClient or MockEngine), the session store, the shared
 * capture configuration and results, and hosts the screens behind the pure navigation state
 * machine of router.ts. Single document, no History API (see router.ts).
 */
import type { PointCloud, TriangleMesh } from '../core/types';
import { SessionStore, type SessionMeta, type SessionPatch } from '../io/storage';
import { requestCapturePermissions, type CapturePermissionStatus } from '../sensors/permissions';
import { WakeLock } from '../sensors/wakelock';
import type { CaptureMode, LiveCamera, QualityPreset, SparseResult } from '../workers/protocol';
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
  /** Last live sparse model / keyframe cameras of the capture (processing preview). */
  liveSparse: { positions: Float32Array; colors: Uint8Array } | null;
  liveCameras: LiveCamera[] | null;
  stats: Record<string, number | string>;
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

const DEFAULT_CONFIG: CaptureConfig = { mode: 'object', quality: 'balanced', useGps: false, torch: false, autoDense: true };

function loadConfig(): CaptureConfig {
  try {
    const raw = localStorage.getItem(CONFIG_KEY);
    if (raw) {
      const c = JSON.parse(raw) as Partial<CaptureConfig>;
      return {
        mode: c.mode === 'object' || c.mode === 'scene' || c.mode === 'auto' ? c.mode : DEFAULT_CONFIG.mode,
        quality: c.quality === 'fast' || c.quality === 'balanced' || c.quality === 'max' ? c.quality : DEFAULT_CONFIG.quality,
        useGps: typeof c.useGps === 'boolean' ? c.useGps : DEFAULT_CONFIG.useGps,
        torch: false, // never start with the torch on
        autoDense: typeof c.autoDense === 'boolean' ? c.autoDense : DEFAULT_CONFIG.autoDense,
      };
    }
  } catch {
    /* storage unavailable */
  }
  return { ...DEFAULT_CONFIG };
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
  config: CaptureConfig;
  results: Results;
  /** Session whose state the engine currently holds (init or loadSession). */
  loadedSessionId: string | null = null;
  /**
   * Debug / e2e hooks installed by screens (e.g. hold frame delivery while a headless test
   * takes a screenshot: software GPU emulation starves the compositor while frames are grabbed).
   */
  readonly debug: { holdFrames?: (hold: boolean) => void; /** Last Diagnostics benchmark report (e2e). */ benchReport?: unknown } = {};
  private permissions: Promise<CapturePermissionStatus> | null = null;
  private sessionsP: Promise<SessionStore | null> | null = null;
  private screen: Screen | null = null;
  private readonly host: HTMLElement;

  constructor(root: HTMLElement, opts: AppOptions) {
    this.root = root;
    this.engine = opts.engine;
    this.mock = !!opts.mock;
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
      if (e.reason === 'storage' || isQuotaError(e.error)) {
        toast('Storage is full — free up space on this device, or delete old scans.', { tone: 'bad', durationMs: 9000 });
      } else if (e.reason === 'out-of-memory' && this.route.screen !== 'capture') {
        // The capture screen asks "finish scan now?" itself (its own prompt); elsewhere: say it.
        toast(`Memory limit reached: ${e.error.message}`, { tone: 'warn', durationMs: 7000 });
      }
    });

    // Workers cannot see page visibility; iOS drops GPU work submitted while hidden.
    document.addEventListener('visibilitychange', () => this.engine.setHidden(document.hidden));
    // Desktop / Android: the browser Back button or closing the tab would silently drop a running
    // capture or processing job (there is no History API state to go back to) — ask first.
    // (iOS Safari shows no beforeunload prompt; the capture itself is persisted as it goes.)
    window.addEventListener('beforeunload', (e: BeforeUnloadEvent) => {
      const screen = this.route.screen;
      if (screen !== 'capture' && screen !== 'processing') return;
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
    this.navigate({ type: 'START_CAPTURE', sessionId: id });
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
    try {
      await s.deleteSession(id);
      if (this.results.sessionId === id) this.results = App.emptyResults(this.config);
      if (this.loadedSessionId === id) this.loadedSessionId = null;
      return true;
    } catch (e) {
      toast(`Could not delete the scan: ${(e as Error)?.message ?? e}`, { tone: 'bad' });
      return false;
    }
  }

  /**
   * Make the engine hold `sessionId` (loadSession when it holds another one) and copy the
   * stored models into `results`. Returns false when it failed (a toast explains).
   */
  async ensureSessionLoaded(sessionId: string | null): Promise<boolean> {
    if (!sessionId || this.loadedSessionId === sessionId) return true;
    try {
      const sum = await this.engine.loadSession(sessionId);
      this.loadedSessionId = sessionId;
      this.results = {
        ...App.emptyResults({ mode: sum.mode, quality: sum.quality ?? this.config.quality }),
        sessionId,
        name: sum.name,
        sparse: sum.sparse,
        dense: sum.dense,
        mesh: sum.mesh,
        stats: sum.stats,
      };
      return true;
    } catch (e) {
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
