/**
 * App entry: styles, engine selection (`?mock=1` → in-page MockEngine, otherwise the engine
 * worker through EngineClient), global error toasts, service worker (production builds only,
 * never on localhost), persistent storage in Home Screen mode.
 *
 * URL parameters (development / tests): `mock=1`, `mockSpeed=<x>` (MockEngine time scale),
 * `screen=diagnostics|setup` (initial screen).
 */
import './app/styles.css';
import { App } from './app/app';
import { toast } from './app/components/toast';
import { EngineClient, type EngineLike } from './app/engine-client';
import { MockEngine } from './app/mock-engine';
import type { NavEvent } from './app/router';
import { isCancelledError } from './workers/protocol';

function boot(): void {
  const params = new URLSearchParams(location.search);
  const mock = params.get('mock') === '1' || params.get('mock') === 'true';
  const speed = Number(params.get('mockSpeed') ?? '1');
  const engine: EngineLike = mock ? new MockEngine({ speed: Number.isFinite(speed) && speed > 0 ? speed : 1 }) : new EngineClient();
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
  window.addEventListener('load', () => {
    navigator.serviceWorker.register(new URL('sw.js', document.baseURI), { scope: './' }).catch((e) => console.warn('[sw] registration failed', e));
  });
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

boot();
