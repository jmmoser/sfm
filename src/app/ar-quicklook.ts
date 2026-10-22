/**
 * "View in AR": AR Quick Look for a prepared USDZ Blob (iOS / iPadOS Safari; the Export sheet's
 * USDZ row, src/app/screens/viewer.ts).
 *
 *  - Offered only on Apple mobile WebKit whose anchors support rel="ar"
 *    (`relList.supports('ar')` with `EnvInfo.apple`, which only excludes macOS Safari: rel="ar"
 *    there, but no AR). Neither proves that AR Quick Look opens: WKWebView in-app browsers
 *    (Instagram, Gmail, …) have the iPhone UA and report rel="ar" without opening it (WebKit
 *    239135). So a launch watches the page: when it neither hides nor loses focus within
 *    `failMs` (2 s), `onNotOpened` runs (the viewer's hint toast: Share / Save, then Files).
 *    Whether AR Quick Look blurs / hides the page is unverified (docs/research/ios-platform.md §8
 *    item 11); the hint is worded so that showing it behind a working AR view does no harm.
 *  - Not offered in a Home Screen (standalone) web app until that device check: if WebKit does not
 *    route rel="ar" there, the click is a plain `<a download href="blob:…">`, which is broken in
 *    standalone mode (WebKit 275288 / 290847; src/io/download.ts never uses it there). The device
 *    test opts in with `localStorage['sfm.arQuickLookStandalone'] = '1'` (Web Inspector console).
 *  - Launch: a hidden `<a rel="ar" href="blob:…" download="name.usdz"><img></a>` clicked inside
 *    the tap (as <model-viewer> does): Safari only hands a rel="ar" link to AR Quick Look when the
 *    anchor's first child is an <img>; `download` names the file AR Quick Look shares (otherwise
 *    the blob id — WebKit 263878).
 *  - The object URL lives as long as its Blob is the prepared export (AR Quick Look reads it
 *    lazily, and a second "View in AR" reuses it): `dispose()` revokes them all and drops a
 *    pending hint (screen exit).
 */
import { sanitizeFilename } from '../io/download';
import { button } from './components/controls';
import { h } from './dom';

/** localStorage key that offers "View in AR" in the Home Screen app too (device test only). */
export const AR_STANDALONE_KEY = 'sfm.arQuickLookStandalone';

/** AR Quick Look can be launched from this page (see the file header). */
export function arQuickLookAvailable(env: { apple: boolean; standalone?: boolean }): boolean {
  if (!env.apple || typeof document === 'undefined') return false;
  if (env.standalone && !standaloneOptIn()) return false;
  try {
    const rl = document.createElement('a').relList as DOMTokenList | undefined;
    return !!rl && typeof rl.supports === 'function' && rl.supports('ar');
  } catch {
    return false;
  }
}

function standaloneOptIn(): boolean {
  try {
    return globalThis.localStorage?.getItem(AR_STANDALONE_KEY) === '1';
  } catch {
    return false;
  }
}

export interface ARQuickLookOptions {
  /** A launch after which the page neither hid nor lost focus within `failMs` (AR Quick Look did not open, probably). */
  onNotOpened?: () => void;
  /** Default 2000 ms. */
  failMs?: number;
}

export interface ARQuickLook {
  /** Open AR Quick Look on `blob` (call it inside a tap handler). */
  open(blob: Blob, filename: string): void;
  /** Revoke the object URLs; a pending `onNotOpened` is dropped. */
  dispose(): void;
}

export function createARQuickLook(opts: ARQuickLookOptions = {}): ARQuickLook {
  const urls = new Map<Blob, string>();
  const failMs = opts.failMs ?? 2000;
  let stopWatch: (() => void) | null = null;

  /** After a launch: did AR Quick Look take the page over (blur / hidden)? */
  const watch = (): void => {
    stopWatch?.();
    const onNotOpened = opts.onNotOpened;
    if (!onNotOpened) return;
    let left = false;
    const onLeave = (): void => {
      left = true;
    };
    const onVisibility = (): void => {
      if (document.hidden) left = true;
    };
    window.addEventListener('blur', onLeave);
    window.addEventListener('pagehide', onLeave);
    document.addEventListener('visibilitychange', onVisibility);
    const stop = (): void => {
      clearTimeout(timer);
      window.removeEventListener('blur', onLeave);
      window.removeEventListener('pagehide', onLeave);
      document.removeEventListener('visibilitychange', onVisibility);
      if (stopWatch === stop) stopWatch = null;
    };
    const timer = setTimeout(() => {
      stop();
      if (!left && !document.hidden) onNotOpened();
    }, failMs);
    stopWatch = stop;
  };

  return {
    open(blob, filename) {
      let url = urls.get(blob);
      if (!url) {
        // a Blob without a type would not be recognised as a USDZ
        const typed = blob.type ? blob : new Blob([blob], { type: 'model/vnd.usdz+zip' });
        url = URL.createObjectURL(typed);
        // one prepared USDZ at a time: the older ones are no longer offered
        for (const [b, u] of urls) {
          URL.revokeObjectURL(u);
          urls.delete(b);
        }
        urls.set(blob, url);
      }
      const a = document.createElement('a');
      a.rel = 'ar';
      a.href = url;
      a.download = sanitizeFilename(filename);
      a.style.display = 'none';
      a.appendChild(document.createElement('img'));
      document.body.appendChild(a);
      // (listening before the click: AR Quick Look may take the page over synchronously)
      watch();
      try {
        a.click();
      } finally {
        // (kept attached a moment: AR Quick Look reports banner taps to the anchor)
        setTimeout(() => a.remove(), 10_000);
      }
    },
    dispose() {
      stopWatch?.();
      for (const u of urls.values()) URL.revokeObjectURL(u);
      urls.clear();
    },
  };
}

/**
 * An Export-sheet action with a "View in AR" button above it (`save`: the row's Share / Save
 * button, demoted to secondary so the row has one primary action).
 */
export function withViewInAR(save: HTMLElement, onView: () => void): HTMLElement {
  save.classList.replace('btn-primary', 'btn-secondary');
  const view = button({ label: 'View in AR', icon: 'object', variant: 'primary', size: 'sm', testId: 'export-ar-usdz', onClick: onView });
  return h('div', { class: 'export-actions' }, view, save);
}
