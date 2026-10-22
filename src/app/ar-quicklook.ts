/**
 * "View in AR": AR Quick Look for a prepared USDZ Blob (iOS / iPadOS Safari; the Export sheet's
 * USDZ row, src/app/screens/viewer.ts).
 *
 *  - Available only on Apple mobile WebKit whose anchors support rel="ar"
 *    (`relList.supports('ar')`). The check alone is not enough — WebKit reports it in contexts
 *    that cannot open AR Quick Look (WebKit 239135) and on macOS (no AR) — so it is combined
 *    with the iPhone / iPad detection (`EnvInfo.apple`). Elsewhere the button is not shown;
 *    Share / Save of the .usdz stays (Files / Messages open it in Quick Look).
 *  - Launch: a hidden `<a rel="ar" href="blob:…" download="name.usdz"><img></a>` clicked inside
 *    the tap (as <model-viewer> does): Safari only hands a rel="ar" link to AR Quick Look when the
 *    anchor's first child is an <img>; `download` names the file AR Quick Look shares (otherwise
 *    the blob id — WebKit 263878).
 *  - The object URL lives as long as its Blob is the prepared export (AR Quick Look reads it
 *    lazily, and a second "View in AR" reuses it): `dispose()` revokes them all (screen exit).
 *  - Unverified on iOS 26 (docs/research/ios-platform.md §3, §8): rel="ar" with a blob URL from a
 *    Home Screen (standalone) web app.
 */
import { sanitizeFilename } from '../io/download';
import { button } from './components/controls';
import { h } from './dom';

/** AR Quick Look can be launched from this page (see the file header). */
export function arQuickLookAvailable(env: { apple: boolean }): boolean {
  if (!env.apple || typeof document === 'undefined') return false;
  try {
    const rl = document.createElement('a').relList as DOMTokenList | undefined;
    return !!rl && typeof rl.supports === 'function' && rl.supports('ar');
  } catch {
    return false;
  }
}

export interface ARQuickLook {
  /** Open AR Quick Look on `blob` (call it inside a tap handler). */
  open(blob: Blob, filename: string): void;
  /** Revoke the object URLs. */
  dispose(): void;
}

export function createARQuickLook(): ARQuickLook {
  const urls = new Map<Blob, string>();
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
      try {
        a.click();
      } finally {
        // (kept attached a moment: AR Quick Look reports banner taps to the anchor)
        setTimeout(() => a.remove(), 10_000);
      }
    },
    dispose() {
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
