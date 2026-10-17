/**
 * Basic controls: buttons, icon buttons, switches, segmented controls, chips, progress bars,
 * spinners, banners. Every control is a real <button>/<input> (keyboard + VoiceOver), with
 * touch targets ≥ 44 px (styles.css).
 */
import { h, type Child } from '../dom';
import { icon, type IconName } from '../icons';
import type { Tone } from '../format';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'glass';

export interface ButtonOptions {
  label?: string;
  icon?: IconName;
  variant?: ButtonVariant;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  block?: boolean;
  ariaLabel?: string;
  disabled?: boolean;
  className?: string;
  onClick?: (e: MouseEvent) => void;
  testId?: string;
}

export function button(o: ButtonOptions): HTMLButtonElement {
  const b = h(
    'button',
    {
      type: 'button',
      class: ['btn', `btn-${o.variant ?? 'secondary'}`, `btn-${o.size ?? 'md'}`, o.block ? 'btn-block' : '', o.className ?? ''].filter(Boolean).join(' '),
      'aria-label': o.ariaLabel,
      'data-testid': o.testId,
    },
    o.icon ? icon(o.icon, { size: o.size === 'sm' ? 18 : 22 }) : null,
    o.label ? h('span', { class: 'btn-label', text: o.label }) : null,
  );
  b.disabled = !!o.disabled;
  if (o.onClick) b.addEventListener('click', o.onClick);
  return b;
}

export function setButtonLabel(b: HTMLButtonElement, label: string, iconName?: IconName): void {
  const l = b.querySelector('.btn-label');
  if (l) l.textContent = label;
  if (iconName) {
    const old = b.querySelector('.icon-wrap');
    const n = icon(iconName, { size: b.classList.contains('btn-sm') ? 18 : 22 });
    if (old) old.replaceWith(n);
    else b.prepend(n);
  }
}

/** Round icon-only button (44×44 min) with an accessible label. */
export function iconButton(name: IconName, label: string, onClick?: (e: MouseEvent) => void, opts: { variant?: 'plain' | 'glass' | 'solid'; className?: string; testId?: string } = {}): HTMLButtonElement {
  const b = h(
    'button',
    {
      type: 'button',
      class: `icon-btn icon-btn-${opts.variant ?? 'plain'}${opts.className ? ` ${opts.className}` : ''}`,
      'aria-label': label,
      title: label,
      'data-testid': opts.testId,
    },
    icon(name, { size: 22 }),
  );
  if (onClick) b.addEventListener('click', onClick);
  return b;
}

export interface SwitchOptions {
  label: string;
  description?: string;
  checked: boolean;
  disabled?: boolean;
  icon?: IconName;
  onChange?: (checked: boolean) => void;
  testId?: string;
}

/** A list row with a label and an iOS-style switch (input type=checkbox role=switch). */
export function switchRow(o: SwitchOptions): { el: HTMLLabelElement; input: HTMLInputElement } {
  const input = h('input', { type: 'checkbox', role: 'switch', class: 'switch-input', 'data-testid': o.testId });
  input.checked = o.checked;
  input.disabled = !!o.disabled;
  input.addEventListener('change', () => o.onChange?.(input.checked));
  const el = h(
    'label',
    { class: 'row row-switch' },
    o.icon ? h('span', { class: 'row-icon' }, icon(o.icon, { size: 20 })) : null,
    h('span', { class: 'row-text' }, h('span', { class: 'row-title', text: o.label }), o.description ? h('span', { class: 'row-sub', text: o.description }) : null),
    h('span', { class: 'switch' }, input, h('span', { class: 'switch-track', 'aria-hidden': 'true' }, h('span', { class: 'switch-thumb' }))),
  );
  return { el, input };
}

export interface SegmentOption<T extends string> {
  value: T;
  label: string;
}

/** Segmented control (radiogroup of buttons). */
export function segmented<T extends string>(
  name: string,
  options: SegmentOption<T>[],
  value: T,
  onChange: (v: T) => void,
): { el: HTMLDivElement; set: (v: T) => void } {
  const el = h('div', { class: 'segmented', role: 'radiogroup', 'aria-label': name });
  const buttons = options.map((o) => {
    const b = h('button', { type: 'button', role: 'radio', class: 'segment', 'data-value': o.value, text: o.label });
    b.addEventListener('click', () => {
      set(o.value);
      onChange(o.value);
    });
    el.appendChild(b);
    return b;
  });
  const set = (v: T): void => {
    buttons.forEach((b, i) => {
      const on = options[i].value === v;
      b.setAttribute('aria-checked', on ? 'true' : 'false');
      b.classList.toggle('active', on);
      b.tabIndex = on ? 0 : -1;
    });
    const idx = Math.max(0, options.findIndex((o) => o.value === v));
    el.style.setProperty('--seg-index', String(idx));
    el.style.setProperty('--seg-count', String(options.length));
  };
  el.addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    const cur = buttons.findIndex((b) => b.classList.contains('active'));
    const n = (cur + (e.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
    buttons[n].click();
    buttons[n].focus();
    e.preventDefault();
  });
  set(value);
  return { el, set };
}

export function chip(label: string, tone: Tone = 'neutral', iconName?: IconName): HTMLSpanElement {
  return h('span', { class: `chip chip-${tone}` }, iconName ? icon(iconName, { size: 14 }) : null, h('span', { text: label }));
}

export function spinner(size: 'sm' | 'md' | 'lg' = 'md', label = 'Loading'): HTMLSpanElement {
  return h('span', { class: `spinner spinner-${size}`, role: 'progressbar', 'aria-label': label, 'aria-busy': 'true' });
}

/** Linear progress bar; `set(-1)` = indeterminate. */
export function progressBar(label: string): { el: HTMLDivElement; set: (p: number) => void } {
  const fill = h('div', { class: 'progress-fill' });
  const el = h('div', { class: 'progress', role: 'progressbar', 'aria-label': label, 'aria-valuemin': '0', 'aria-valuemax': '100' }, fill);
  const set = (p: number): void => {
    if (p < 0 || !Number.isFinite(p)) {
      el.classList.add('indeterminate');
      el.removeAttribute('aria-valuenow');
      fill.style.transform = '';
      return;
    }
    el.classList.remove('indeterminate');
    const c = Math.max(0, Math.min(1, p));
    fill.style.transform = `scaleX(${c})`;
    el.setAttribute('aria-valuenow', String(Math.round(c * 100)));
  };
  set(0);
  return { el, set };
}

export interface BannerOptions {
  tone: Tone;
  title: string;
  body?: Child;
  icon?: IconName;
  actions?: HTMLElement[];
  onDismiss?: () => void;
  testId?: string;
}

export function banner(o: BannerOptions): HTMLDivElement {
  const el = h(
    'div',
    { class: `banner banner-${o.tone}`, role: o.tone === 'bad' ? 'alert' : 'status', 'data-testid': o.testId },
    h('span', { class: 'banner-icon' }, icon(o.icon ?? (o.tone === 'bad' || o.tone === 'warn' ? 'warning' : 'info'), { size: 20 })),
    h(
      'div',
      { class: 'banner-text' },
      h('div', { class: 'banner-title', text: o.title }),
      o.body ? h('div', { class: 'banner-body' }, o.body) : null,
      o.actions?.length ? h('div', { class: 'banner-actions' }, ...o.actions) : null,
    ),
  );
  if (o.onDismiss) {
    const close = iconButton('close', 'Dismiss', () => {
      el.remove();
      o.onDismiss?.();
    });
    close.classList.add('banner-close');
    el.appendChild(close);
  }
  return el;
}

/** Section with an optional header and a grouped (inset) list body. */
export function section(title: string | null, ...children: Child[]): HTMLElement {
  return h('section', { class: 'section' }, title ? h('h2', { class: 'section-title', text: title }) : null, ...children);
}

export function group(...children: Child[]): HTMLDivElement {
  return h('div', { class: 'group' }, ...children);
}

/** Key/value row for info lists. */
export function kvRow(key: string, value: Child, opts: { mono?: boolean; tone?: Tone } = {}): HTMLDivElement {
  return h(
    'div',
    { class: 'row row-kv' },
    h('span', { class: 'row-title', text: key }),
    h('span', { class: `row-value${opts.mono ? ' mono' : ''}${opts.tone ? ` tone-${opts.tone}` : ''}` }, value),
  );
}

/** Large tappable choice card (radio). */
export function choiceCard(o: {
  icon: IconName;
  title: string;
  subtitle: string;
  selected: boolean;
  onSelect: () => void;
  testId?: string;
}): HTMLButtonElement {
  const b = h(
    'button',
    { type: 'button', role: 'radio', class: 'choice', 'aria-checked': o.selected ? 'true' : 'false', 'data-testid': o.testId },
    h('span', { class: 'choice-icon' }, icon(o.icon, { size: 26 })),
    h('span', { class: 'choice-text' }, h('span', { class: 'choice-title', text: o.title }), h('span', { class: 'choice-sub', text: o.subtitle })),
    h('span', { class: 'choice-check' }, icon('check', { size: 16 })),
  );
  b.addEventListener('click', o.onSelect);
  return b;
}
