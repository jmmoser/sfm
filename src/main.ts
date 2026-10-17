/**
 * App entry: styles, engine selection (`?mock=1` → in-page MockEngine, otherwise the engine
 * worker through EngineClient), global error toasts, service worker (production builds only,
 * never on localhost), persistent storage in Home Screen mode.
 *
 * URL parameters (development / tests): `mock=1`, `mockSpeed=<x>` (MockEngine time scale),
 * `mockFail=<method>[:message]` (make a MockEngine method fail; the custom message only in dev
 * builds), `mockStall=<method>[,…]` (a MockEngine long operation hangs and ignores cancel:
 * finishCapture | refine | densify | mesh), `mockMemoryLimit=<n>` (capture memory cap after n
 * keyframes), `screen=diagnostics|setup`, `benchScale=small` (Diagnostics benchmark with CI-sized
 * inputs). All validated in src/app/boot-params.ts.
 */
import './app/styles.css';
import { App } from './app/app';
import { toast } from './app/components/toast';
import { EngineClient, type EngineLike } from './app/engine-client';
import type { NavEvent } from './app/router';
import { isCancelledError } from './workers/protocol';
import { parseBootParams } from './app/boot-params';
import { registerServiceWorker, requestPersistenceIfInstalled } from './app/pwa';

async function boot(): Promise<void> {
  // Validated (src/app/boot-params.ts): unknown values are ignored; the mockFail message is a
  // dev-build hook only.
  const { mock, mockOptions, screen, cameraPath } = parseBootParams(location.search, import.meta.env.DEV);
  // The MockEngine (dev / tests) is loaded on demand: it is not part of the production entry chunk.
  const engine: EngineLike = mock ? new (await import('./app/mock-engine')).MockEngine(mockOptions) : new EngineClient();
  const root = document.getElementById('app') ?? document.body.appendChild(document.createElement('div'));
  root.id = 'app';
  installErrorToasts();
  const initialEvent: NavEvent | null = screen === 'diagnostics' ? { type: 'OPEN_DIAGNOSTICS' } : screen === 'setup' ? { type: 'NEW_SCAN' } : null;
  const app = new App(root, { engine, mock, initialEvent, cameraPath });
  (globalThis as { __sfmApp?: App }).__sfmApp = app; // debugging / e2e hooks
  // the entry chunk's own (content-hashed) URL versions the service worker
  registerServiceWorker(import.meta.url);
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

void boot();
