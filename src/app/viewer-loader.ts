/**
 * Lazy loading of the three.js Viewer: Home and Setup do not need it, so it lives in its own
 * chunk (most of the bundle). It is prefetched when the browser is idle after the first screen
 * rendered, so the capture / processing / viewer screens rarely wait for it.
 */
type ViewerModule = typeof import('../viewer');

let pending: Promise<ViewerModule> | null = null;

export function loadViewerModule(): Promise<ViewerModule> {
  if (!pending) {
    pending = import('../viewer').catch((e) => {
      pending = null; // allow a retry (e.g. offline before the service worker cached the chunk)
      throw e;
    });
  }
  return pending;
}

/** Start loading in the background (idle time). */
export function prefetchViewer(delayMs = 1200): void {
  const run = (): void => void loadViewerModule().catch(() => undefined);
  const ric = (globalThis as { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number }).requestIdleCallback;
  setTimeout(() => (ric ? ric(run, { timeout: 3000 }) : run()), delayMs);
}
