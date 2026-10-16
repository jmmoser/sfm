/**
 * Bottom sheets (iOS-style): a panel sliding up from the bottom with a grabber, a title, a
 * scrollable body and safe-area padding. Tap on the backdrop, the close button, Escape or a
 * downward swipe on the grabber/header closes it. `actionSheet` builds a list of actions.
 */
import { h, type Child } from '../dom';
import { icon, type IconName } from '../icons';
import { overlayLayer, pushModal, registerCloser } from './layers';

export interface SheetHandle {
  el: HTMLElement;
  body: HTMLElement;
  close: () => void;
  /** Resolves when the sheet has closed. */
  closed: Promise<void>;
  setTitle: (t: string) => void;
}

export interface SheetOptions {
  title: string;
  subtitle?: string;
  content?: Child;
  onClose?: () => void;
  className?: string;
  testId?: string;
}

let seq = 0;

export function openSheet(o: SheetOptions): SheetHandle {
  const id = `sheet-${++seq}`;
  let resolveClosed!: () => void;
  const closed = new Promise<void>((r) => (resolveClosed = r));
  const backdrop = h('div', { class: 'sheet-backdrop' });
  const titleEl = h('h2', { class: 'sheet-title', id: `${id}-t`, text: o.title });
  const closeBtn = h('button', { type: 'button', class: 'icon-btn icon-btn-solid sheet-close', 'aria-label': 'Close' }, icon('close', { size: 20 }));
  const header = h(
    'div',
    { class: 'sheet-header' },
    h('div', { class: 'sheet-grabber', 'aria-hidden': 'true' }),
    h('div', { class: 'sheet-heading' }, titleEl, o.subtitle ? h('p', { class: 'sheet-subtitle', text: o.subtitle }) : null),
    closeBtn,
  );
  const body = h('div', { class: 'sheet-body' });
  if (o.content) {
    const c = Array.isArray(o.content) ? o.content : [o.content];
    for (const x of c) if (x instanceof Node) body.appendChild(x);
  }
  const panel = h(
    'div',
    { class: `sheet${o.className ? ` ${o.className}` : ''}`, role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': `${id}-t`, tabindex: '-1', 'data-testid': o.testId },
    header,
    body,
  );
  const wrap = h('div', { class: 'sheet-wrap' }, backdrop, panel);
  overlayLayer().appendChild(wrap);
  let isClosed = false;
  const close = (): void => {
    if (isClosed) return;
    isClosed = true;
    release();
    unregister();
    backdrop.classList.remove('show');
    panel.classList.remove('show');
    panel.style.transform = '';
    setTimeout(() => {
      wrap.remove();
      resolveClosed();
    }, 260);
    o.onClose?.();
  };
  backdrop.addEventListener('click', close);
  closeBtn.addEventListener('click', close);
  const release = pushModal(panel, close);
  const unregister = registerCloser(close);

  // Swipe down on the header to dismiss.
  let startY = NaN;
  let dy = 0;
  header.addEventListener('pointerdown', (e: PointerEvent) => {
    if ((e.target as HTMLElement).closest('button')) return;
    startY = e.clientY;
    dy = 0;
    header.setPointerCapture?.(e.pointerId);
    panel.classList.add('dragging');
  });
  header.addEventListener('pointermove', (e: PointerEvent) => {
    if (Number.isNaN(startY)) return;
    dy = Math.max(0, e.clientY - startY);
    panel.style.transform = `translateY(${dy}px)`;
  });
  const end = (): void => {
    if (Number.isNaN(startY)) return;
    startY = NaN;
    panel.classList.remove('dragging');
    if (dy > 90) close();
    else panel.style.transform = '';
  };
  header.addEventListener('pointerup', end);
  header.addEventListener('pointercancel', end);

  requestAnimationFrame(() => {
    backdrop.classList.add('show');
    panel.classList.add('show');
  });
  return { el: panel, body, close, closed, setTitle: (t) => (titleEl.textContent = t) };
}

export interface SheetAction {
  label: string;
  icon?: IconName;
  description?: string;
  destructive?: boolean;
  disabled?: boolean;
  testId?: string;
  onSelect: () => void;
}

/** Action list in a sheet; selecting an action closes the sheet first. */
export function actionSheet(title: string, actions: SheetAction[], subtitle?: string): SheetHandle {
  const list = h('div', { class: 'action-list', role: 'menu' });
  const sheet = openSheet({ title, subtitle, content: list, className: 'sheet-actions' });
  for (const a of actions) {
    const b = h(
      'button',
      { type: 'button', role: 'menuitem', class: `action${a.destructive ? ' action-destructive' : ''}`, 'data-testid': a.testId },
      a.icon ? h('span', { class: 'action-icon' }, icon(a.icon, { size: 22 })) : null,
      h('span', { class: 'action-text' }, h('span', { class: 'action-title', text: a.label }), a.description ? h('span', { class: 'action-sub', text: a.description }) : null),
    );
    b.disabled = !!a.disabled;
    b.addEventListener('click', () => {
      sheet.close();
      a.onSelect();
    });
    list.appendChild(b);
  }
  return sheet;
}
