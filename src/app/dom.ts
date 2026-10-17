/**
 * Tiny DOM helpers (no framework, CSP-safe: no inline handlers / eval / style attributes).
 */

export type Child = Node | string | number | null | undefined | false | Child[];

export type Props = {
  class?: string;
  text?: string;
  /** Trusted static markup only (icons). */
  html?: string;
  style?: Partial<Record<string, string>>;
  dataset?: Record<string, string>;
  [attr: string]: unknown;
};

/** Create an element: h('button', { class: 'btn', onclick: fn, 'aria-label': 'Close' }, 'Text'). */
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, props?: Props | null, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (props) applyProps(el, props);
  append(el, children);
  return el;
}

export function applyProps(el: HTMLElement, props: Props): void {
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = String(v);
    else if (k === 'text') el.textContent = String(v);
    else if (k === 'html') el.innerHTML = String(v);
    else if (k === 'style') {
      for (const [sk, sv] of Object.entries(v as Record<string, string>)) {
        if (sv === undefined) continue;
        if (sk.startsWith('--') || sk.includes('-')) el.style.setProperty(sk, sv);
        else (el.style as unknown as Record<string, string>)[sk] = sv;
      }
    } else if (k === 'dataset') Object.assign(el.dataset, v as Record<string, string>);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v as EventListener);
    // never an inline event-handler attribute from a string (the CSP would refuse it anyway)
    else if (/^on/i.test(k)) throw new TypeError(`h(): '${k}' must be a function`);
    else if (k in el && typeof v !== 'string' && k !== 'role') (el as unknown as Record<string, unknown>)[k] = v;
    else el.setAttribute(k, v === true ? '' : String(v));
  }
}

export function append(el: Node, children: Child[]): void {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else if (c instanceof Node) el.appendChild(c);
    else el.appendChild(document.createTextNode(String(c)));
  }
}

/** Remove all children. */
export function clear(el: Node): void {
  while (el.firstChild) el.removeChild(el.firstChild);
}

/** Replace all children. */
export function replaceChildren(el: Element, ...children: Child[]): void {
  clear(el);
  append(el, children);
}

/** Toggle a boolean attribute / class conveniently. */
export function setHidden(el: HTMLElement, hidden: boolean): void {
  el.hidden = hidden;
}

/** Collects disposers (event listeners, subscriptions, timers) for a screen. */
export class Disposer {
  private readonly fns: (() => void)[] = [];
  add(fn: (() => void) | null | undefined): void {
    if (fn) this.fns.push(fn);
  }
  listen<K extends keyof WindowEventMap>(t: Window, type: K, fn: (e: WindowEventMap[K]) => void, opts?: AddEventListenerOptions): void;
  listen<K extends keyof DocumentEventMap>(t: Document, type: K, fn: (e: DocumentEventMap[K]) => void, opts?: AddEventListenerOptions): void;
  listen(t: EventTarget, type: string, fn: (e: Event) => void, opts?: AddEventListenerOptions): void;
  listen(t: EventTarget, type: string, fn: (e: never) => void, opts?: AddEventListenerOptions): void {
    t.addEventListener(type, fn as EventListener, opts);
    this.fns.push(() => t.removeEventListener(type, fn as EventListener, opts));
  }
  timeout(fn: () => void, ms: number): void {
    const id = setTimeout(fn, ms);
    this.fns.push(() => clearTimeout(id));
  }
  interval(fn: () => void, ms: number): void {
    const id = setInterval(fn, ms);
    this.fns.push(() => clearInterval(id));
  }
  dispose(): void {
    while (this.fns.length) {
      const f = this.fns.pop()!;
      try {
        f();
      } catch (e) {
        console.error(e);
      }
    }
  }
}

/** prefers-reduced-motion. */
export function reducedMotion(): boolean {
  try {
    return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch {
    return false;
  }
}
