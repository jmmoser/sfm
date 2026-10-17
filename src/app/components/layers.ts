/**
 * Shared overlay layer (dialogs, sheets, toasts live above the screens) and modal plumbing:
 * the app root is made `inert` while a modal is open, focus is moved into the modal and
 * restored afterwards, Escape closes.
 */
import { h } from '../dom';

let layer: HTMLDivElement | null = null;
let appRoot: HTMLElement | null = null;
const modalStack: HTMLElement[] = [];

export function initLayers(root: HTMLElement): void {
  appRoot = root;
  overlayLayer();
}

export function overlayLayer(): HTMLDivElement {
  if (layer && layer.isConnected) return layer;
  layer = h('div', { class: 'overlay-layer' });
  document.body.appendChild(layer);
  return layer;
}

/**
 * Register a modal element (already in the overlay layer). Returns a release function that
 * restores focus / inertness. `onEscape` is called for the Escape key on the topmost modal.
 */
export function pushModal(el: HTMLElement, onEscape: () => void): () => void {
  const prevFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  modalStack.push(el);
  if (appRoot) appRoot.inert = true;
  const onKey = (e: KeyboardEvent): void => {
    if (e.key === 'Escape' && modalStack[modalStack.length - 1] === el) {
      e.preventDefault();
      onEscape();
    }
  };
  document.addEventListener('keydown', onKey);
  // Focus the first focusable control (after the element is laid out).
  requestAnimationFrame(() => {
    const f = el.querySelector<HTMLElement>('[autofocus], input, button:not([disabled]), [tabindex]:not([tabindex="-1"])');
    (f ?? el).focus?.({ preventScroll: true });
  });
  return () => {
    document.removeEventListener('keydown', onKey);
    const i = modalStack.indexOf(el);
    if (i >= 0) modalStack.splice(i, 1);
    if (appRoot && modalStack.length === 0) appRoot.inert = false;
    try {
      prevFocus?.focus({ preventScroll: true });
    } catch {
      /* ignore */
    }
  };
}

/** True while any dialog / sheet is open. */
export function modalOpen(): boolean {
  return modalStack.length > 0;
}

/** Close every open modal (e.g. when the screen changes). */
const closers = new Set<() => void>();
export function registerCloser(fn: () => void): () => void {
  closers.add(fn);
  return () => closers.delete(fn);
}
export function closeAllModals(): void {
  for (const c of [...closers]) c();
}
