/**
 * Inline SVG icon set (24×24, stroke = currentColor). Decorative by default (aria-hidden);
 * pass a label for standalone meaningful icons.
 */

const P = {
  plus: '<path d="M12 5v14M5 12h14"/>',
  close: '<path d="M6 6l12 12M18 6L6 18"/>',
  back: '<path d="M15 5l-7 7 7 7"/>',
  chevronRight: '<path d="M9 5l7 7-7 7"/>',
  chevronDown: '<path d="M6 9l6 6 6-6"/>',
  more: '<circle cx="5" cy="12" r="1.4" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none"/><circle cx="19" cy="12" r="1.4" fill="currentColor" stroke="none"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/>',
  pencil: '<path d="M4 20h4L19 9l-4-4L4 16v4z"/><path d="M13.5 6.5l4 4"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/>',
  activity: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
  camera: '<path d="M4 8a2 2 0 0 1 2-2h2l1.5-2h5L16 6h2a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8z"/><circle cx="12" cy="13" r="3.5"/>',
  cube: '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/><path d="M12 12l8-4.5M12 12L4 7.5M12 12v9"/>',
  object: '<path d="M12 7l5 2.8v5.6L12 18l-5-2.6V9.8L12 7z"/><path d="M12 12.6l5-2.8M12 12.6L7 9.8M12 12.6V18"/><path d="M4.5 6.5A9.5 4 0 0 1 19.5 6.5" stroke-dasharray="2 2.2"/><path d="M19.5 17.5a9.5 4 0 0 1-15 0" stroke-dasharray="2 2.2"/>',
  area: '<path d="M3 19l5-9 4 6 3-4 6 7H3z"/><circle cx="16.5" cy="6.5" r="2"/>',
  auto: '<path d="M5 19L15 9"/><path d="M14 4v2M19 9h2M17.5 5.5l1.5-1.5M12 7l1 1"/><path d="M16 11l2 2"/>',
  layers: '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/>',
  points: '<circle cx="6" cy="7" r="1.3" fill="currentColor"/><circle cx="12" cy="5" r="1.3" fill="currentColor"/><circle cx="18" cy="8" r="1.3" fill="currentColor"/><circle cx="8" cy="13" r="1.3" fill="currentColor"/><circle cx="15" cy="13" r="1.3" fill="currentColor"/><circle cx="5" cy="18" r="1.3" fill="currentColor"/><circle cx="12" cy="19" r="1.3" fill="currentColor"/><circle cx="19" cy="17" r="1.3" fill="currentColor"/>',
  mesh: '<path d="M4 18L12 4l8 14H4z"/><path d="M8 11h8M12 18l-4-7M12 18l4-7"/>',
  frustum: '<path d="M4 6h7v6H4zM11 7.5l8-3.5v10l-8-3.5"/><path d="M4 16h16" stroke-dasharray="2 2.4"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  ruler: '<path d="M3 16.5L16.5 3 21 7.5 7.5 21 3 16.5z"/><path d="M7 12.5l2 2M10 9.5l1.5 1.5M13 6.5l2 2M5.5 15l1 1M8.5 12l1 1M11.5 9l1 1"/>',
  share: '<path d="M12 3v12M8 7l4-4 4 4"/><path d="M6 11H5a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-8a1 1 0 0 0-1-1h-1"/>',
  download: '<path d="M12 4v11M7.5 10.5L12 15l4.5-4.5"/><path d="M5 19h14"/>',
  pause: '<path d="M9 6v12M15 6v12"/>',
  play: '<path d="M8 5.5v13l11-6.5-11-6.5z" fill="currentColor"/>',
  stop: '<rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  torch: '<path d="M8 3h8v3l-2 4v10a1 1 0 0 1-1 1h-2a1 1 0 0 1-1-1V10L8 6V3z"/><path d="M8 6h8M12 13v2"/>',
  torchOff: '<path d="M8 3h8v3l-2 4v10a1 1 0 0 1-1 1h-2a1 1 0 0 1-1-1V10L8 6V3z"/><path d="M3 3l18 18"/>',
  expand: '<path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7"/>',
  collapse: '<path d="M20 10h-6V4M4 14h6v6M14 10l7-7M10 14l-7 7"/>',
  warning: '<path d="M12 4L2.5 20h19L12 4z"/><path d="M12 10v4.5M12 17.5v.01"/>',
  location: '<path d="M12 21s-6.5-6.2-6.5-11a6.5 6.5 0 0 1 13 0c0 4.8-6.5 11-6.5 11z"/><circle cx="12" cy="10" r="2.3"/>',
  compass: '<circle cx="12" cy="12" r="9"/><path d="M15.5 8.5l-2 5-5 2 2-5 5-2z"/>',
  cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/><path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
  sparkles: '<path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8L12 3z"/><path d="M18.5 15l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8.8-2z"/>',
  refresh: '<path d="M20 11a8 8 0 0 0-14.3-4.9L4 8"/><path d="M4 4v4h4"/><path d="M4 13a8 8 0 0 0 14.3 4.9L20 16"/><path d="M20 20v-4h-4"/>',
  storage: '<ellipse cx="12" cy="6" rx="7.5" ry="3"/><path d="M4.5 6v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3V6"/><path d="M4.5 12v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6"/>',
  home: '<path d="M4 11l8-7 8 7v9a1 1 0 0 1-1 1h-5v-6h-4v6H5a1 1 0 0 1-1-1v-9z"/>',
  lock: '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
  addSquare: '<rect x="4" y="4" width="16" height="16" rx="4"/><path d="M12 8.5v7M8.5 12h7"/>',
  motion: '<rect x="7" y="3" width="10" height="18" rx="2.5"/><path d="M3 9c-1 2-1 4 0 6M21 9c1 2 1 4 0 6"/>',
  gpu: '<rect x="3" y="7" width="18" height="10" rx="2"/><circle cx="9" cy="12" r="2.3"/><path d="M14 10.5h4M14 13.5h4M6 17v3M10 17v3M14 17v3"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.6 3.6 5.6 3.6 9s-1.1 6.4-3.6 9c-2.5-2.6-3.6-5.6-3.6-9S9.5 5.6 12 3z"/>',
  target: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
  file: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5z"/><path d="M14 3v5h5"/>',
  zip: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5z"/><path d="M11 5h2M11 8h2M11 11h2M11 14h2v3h-2z"/>',
  dot: '<circle cx="12" cy="12" r="4" fill="currentColor" stroke="none"/>',
} as const;

export type IconName = keyof typeof P;

export function iconSvg(name: IconName, size = 24): string {
  // innerHTML sink (icon()): only the static markup above, never a value from outside (own
  // properties only — no `constructor` / `__proto__` lookups — and a numeric size).
  const body: string = hasIcon(name) ? P[name] : '';
  const px = Number.isFinite(size) && size > 0 ? Math.round(size * 100) / 100 : 24;
  return `<svg class="icon" width="${px}" height="${px}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${body}</svg>`;
}

/** An icon element (span wrapper). */
export function icon(name: IconName, opts: { size?: number; label?: string; className?: string } = {}): HTMLSpanElement {
  const s = document.createElement('span');
  s.className = `icon-wrap${opts.className ? ` ${opts.className}` : ''}`;
  s.innerHTML = iconSvg(name, opts.size ?? 24);
  if (opts.label) {
    s.setAttribute('role', 'img');
    s.setAttribute('aria-label', opts.label);
  } else s.setAttribute('aria-hidden', 'true');
  return s;
}

export function hasIcon(name: string): name is IconName {
  return Object.prototype.hasOwnProperty.call(P, name);
}
