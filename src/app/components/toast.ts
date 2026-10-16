/**
 * Toasts: short, non-blocking messages at the bottom (above the home indicator), announced
 * by screen readers (role=status / alert). Identical messages are coalesced.
 */
import { h } from '../dom';
import { icon, type IconName } from '../icons';
import type { Tone } from '../format';
import { modalOpen, overlayLayer } from './layers';

export interface ToastOptions {
  tone?: Tone;
  icon?: IconName;
  durationMs?: number;
  action?: { label: string; onClick: () => void };
}

let host: HTMLDivElement | null = null;
const active = new Map<string, { el: HTMLElement; timer: ReturnType<typeof setTimeout> }>();

function toastHost(): HTMLDivElement {
  if (host && host.isConnected) return host;
  host = h('div', { class: 'toast-host', 'aria-live': 'polite' });
  overlayLayer().appendChild(host);
  return host;
}

export function toast(message: string, opts: ToastOptions = {}): void {
  if (typeof document === 'undefined') return;
  const tone = opts.tone ?? 'neutral';
  const key = `${tone}|${message}`;
  const existing = active.get(key);
  const duration = opts.durationMs ?? (tone === 'bad' ? 6000 : 3000);
  if (existing) {
    clearTimeout(existing.timer);
    existing.timer = setTimeout(() => dismiss(key), duration);
    return;
  }
  const iconName: IconName | null = opts.icon ?? (tone === 'bad' || tone === 'warn' ? 'warning' : tone === 'good' ? 'check' : null);
  const el = h(
    'div',
    { class: `toast toast-${tone}`, role: tone === 'bad' ? 'alert' : 'status' },
    iconName ? icon(iconName, { size: 18 }) : null,
    h('span', { class: 'toast-text', text: message }),
  );
  if (opts.action) {
    const a = opts.action;
    const b = h('button', { type: 'button', class: 'toast-action', text: a.label });
    b.addEventListener('click', () => {
      dismiss(key);
      a.onClick();
    });
    el.appendChild(b);
  }
  const host = toastHost();
  // Over an open sheet / dialog the bottom is taken: show toasts at the top instead.
  host.classList.toggle('toast-top', modalOpen());
  host.appendChild(el);
  // Keep at most 2 visible (the newest replaces the oldest).
  while (active.size >= 2) {
    const first = active.keys().next().value as string;
    dismiss(first);
  }
  active.set(key, { el, timer: setTimeout(() => dismiss(key), duration) });
  requestAnimationFrame(() => el.classList.add('show'));
}

function dismiss(key: string): void {
  const t = active.get(key);
  if (!t) return;
  active.delete(key);
  clearTimeout(t.timer);
  t.el.classList.remove('show');
  t.el.classList.add('hide');
  setTimeout(() => t.el.remove(), 250);
}
