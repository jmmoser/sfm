/**
 * App entry: styles, engine selection (`?mock=1` → in-page MockEngine, otherwise the engine
 * worker through EngineClient), global error toasts, service worker (production builds only,
 * never on localhost), persistent storage in Home Screen mode.
 *
 * URL parameters (development / tests): `mock=1`, `mockSpeed=<x>` (MockEngine time scale),
 * `mockFail=<method>[:message]` (make a MockEngine method fail), `mockStall=<method>[,…]` (a
 * MockEngine long operation hangs and ignores cancel: finishCapture | refine | densify | mesh),
 * `mockMemoryLimit=<n>` (capture memory cap after n keyframes), `screen=diagnostics|setup`,
 * `benchScale=small` (Diagnostics benchmark with CI-sized inputs).
 */
import './app/styles.css';
import { App } from './app/app';
import { toast } from './app/components/toast';
import { EngineClient, type EngineLike } from './app/engine-client';
import type { NavEvent } from './app/router';
import { isCancelledError } from './workers/protocol';

async function boot(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const mock = params.get('mock') === '1' || params.get('mock') === 'true';
  const speed = Number(params.get('mockSpeed') ?? '1');
  const fail = params.get('mockFail');
  const failOn = fail ? { [fail.split(':')[0]]: fail.split(':').slice(1).join(':') || 'Simulated failure' } : undefined;
  const stallOn = Object.fromEntries((params.get('mockStall') ?? '').split(',').filter(Boolean).map((m) => [m, true]));
  const memoryLimitKeyframes = Number(params.get('mockMemoryLimit') ?? '0') || undefined;
  // The MockEngine (dev / tests) is loaded on demand: it is not part of the production entry chunk.
  const engine: EngineLike = mock
    ? new (await import('./app/mock-engine')).MockEngine({ speed: Number.isFinite(speed) && speed > 0 ? speed : 1, failOn, stallOn, memoryLimitKeyframes })
    : new EngineClient();
  const root = document.getElementById('app') ?? document.body.appendChild(document.createElement('div'));
  root.id = 'app';
  installErrorToasts();
  const screen = params.get('screen');
  const initialEvent: NavEvent | null = screen === 'diagnostics' ? { type: 'OPEN_DIAGNOSTICS' } : screen === 'setup' ? { type: 'NEW_SCAN' } : null;
  const app = new App(root, { engine, mock, initialEvent });
  (globalThis as { __sfmApp?: App }).__sfmApp = app; // debugging / e2e hooks
  registerServiceWorker();
  requestPersistenceIfInstalled();
}

/** Uncaught errors / rejections → a toast (deduplicated by the toast host), never a silent failure. */
function installErrorToasts(): void {
  const ignore = (msg: string): boolean => /ResizeObserver loop|Script error\.?$|AbortError|The play\(\) request was interrupted/i.test(msg);
  window.addEventListener('error', (e: ErrorEvent) => {
    const msg = e.message || String(e.error ?? 'Unknown error');
    if (ignore(msg)) return;
    toast(`Something went wrong: ${msg}`, { tone: 'bad' });
  });
  window.addEventListener('unhandledrejection', (e: PromiseRejectionEvent) => {
    const r = e.reason as { name?: string; message?: string } | undefined;
    if (isCancelledError(r)) {
      e.preventDefault();
      return;
    }
    const msg = r?.message ?? String(r);
    if (ignore(msg) || r?.name === 'AbortError') return;
    toast(`Something went wrong: ${msg}`, { tone: 'bad' });
  });
}

function registerServiceWorker(): void {
  if (!import.meta.env.PROD || !('serviceWorker' in navigator)) return;
  const host = location.hostname;
  if (host === 'localhost' || host === '[::1]' || host.endsWith('.localhost') || /^127\./.test(host)) return;
  const register = (): void => {
    navigator.serviceWorker.register(new URL('sw.js', document.baseURI), { scope: './' }).catch((e) => console.warn('[sw] registration failed', e));
  };
  // boot() may run after 'load' (it awaits the MockEngine chunk with ?mock=1): never miss it.
  if (document.readyState === 'complete') register();
  else window.addEventListener('load', register, { once: true });
}

/**
 * In a Home Screen app `persist()` is granted (ios-platform.md §2.2) — ask on every launch.
 * In a tab it is usually denied; the Start tap asks again (with a gesture).
 */
function requestPersistenceIfInstalled(): void {
  const nav = navigator as Navigator & { standalone?: boolean };
  const standalone = nav.standalone === true || matchMedia('(display-mode: standalone)').matches;
  if (!standalone || !nav.storage?.persist) return;
  void nav.storage.persist().catch(() => false);
}

void boot();
