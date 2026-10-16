/**
 * Minimal reactive state container (no framework): `get`, `set` (shallow merge or updater),
 * `subscribe`, and `select` (fires only when the selected slice changes, compared with
 * Object.is or a custom equality). Listeners run synchronously after each change, in
 * subscription order; a listener may call `set` again (the nested change is delivered after
 * the current round).
 */
export type Listener<T> = (state: T, prev: T) => void;
export type Updater<T> = Partial<T> | ((s: T) => Partial<T>);

export class Store<T extends object> {
  private state: T;
  private readonly listeners = new Set<Listener<T>>();
  private notifying = false;
  private queued: T[] = [];

  constructor(initial: T) {
    this.state = initial;
  }

  get(): T {
    return this.state;
  }

  /** Shallow-merge a patch (or the result of an updater). No-op when nothing changed. */
  set(u: Updater<T>): void {
    const patch = typeof u === 'function' ? u(this.state) : u;
    let changed = false;
    for (const k of Object.keys(patch) as (keyof T)[]) {
      if (!Object.is(this.state[k], patch[k])) {
        changed = true;
        break;
      }
    }
    if (!changed) return;
    const prev = this.state;
    this.state = { ...prev, ...patch };
    this.emit(prev);
  }

  /** Replace the whole state. */
  replace(next: T): void {
    if (Object.is(next, this.state)) return;
    const prev = this.state;
    this.state = next;
    this.emit(prev);
  }

  subscribe(cb: Listener<T>): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }

  /** Subscribe to a derived value; `cb` runs only when it changes (and once at once if `immediate`). */
  select<S>(selector: (s: T) => S, cb: (value: S, prev: S) => void, opts: { equals?: (a: S, b: S) => boolean; immediate?: boolean } = {}): () => void {
    const eq = opts.equals ?? Object.is;
    let cur = selector(this.state);
    if (opts.immediate) cb(cur, cur);
    return this.subscribe((s) => {
      const next = selector(s);
      if (eq(next, cur)) return;
      const prev = cur;
      cur = next;
      cb(next, prev);
    });
  }

  private emit(prev: T): void {
    if (this.notifying) {
      this.queued.push(prev);
      return;
    }
    this.notifying = true;
    try {
      let p: T | undefined = prev;
      while (p !== undefined) {
        const state = this.state;
        for (const l of [...this.listeners]) {
          try {
            l(state, p);
          } catch (e) {
            console.error('[store] listener failed', e);
          }
        }
        p = this.queued.shift();
      }
    } finally {
      this.notifying = false;
      this.queued = [];
    }
  }
}

/** Shallow equality for arrays / plain objects (for `select` on derived tuples). */
export function shallowEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (const k of ka) if (!Object.is((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k])) return false;
  return true;
}
