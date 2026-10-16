import { describe, expect, it, vi } from 'vitest';
import { Store, shallowEqual } from './store';

describe('Store', () => {
  it('merges patches and notifies with (state, prev)', () => {
    const s = new Store({ a: 1, b: 'x' });
    const cb = vi.fn();
    s.subscribe(cb);
    s.set({ a: 2 });
    expect(s.get()).toEqual({ a: 2, b: 'x' });
    expect(cb).toHaveBeenCalledWith({ a: 2, b: 'x' }, { a: 1, b: 'x' });
  });

  it('skips no-op updates and supports updaters', () => {
    const s = new Store({ n: 1 });
    const cb = vi.fn();
    s.subscribe(cb);
    s.set({ n: 1 });
    expect(cb).not.toHaveBeenCalled();
    s.set((st) => ({ n: st.n + 1 }));
    expect(s.get().n).toBe(2);
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it('select fires only when the slice changes', () => {
    const s = new Store({ a: 1, b: 1 });
    const cb = vi.fn();
    s.select((st) => st.a, cb);
    s.set({ b: 2 });
    expect(cb).not.toHaveBeenCalled();
    s.set({ a: 5 });
    expect(cb).toHaveBeenCalledWith(5, 1);
  });

  it('select with immediate and custom equality', () => {
    const s = new Store({ list: [1, 2] });
    const cb = vi.fn();
    s.select((st) => st.list.slice(), cb, { equals: shallowEqual, immediate: true });
    expect(cb).toHaveBeenCalledTimes(1);
    s.set({ list: [1, 2] }); // new array, same content
    expect(cb).toHaveBeenCalledTimes(1);
    s.set({ list: [1, 3] });
    expect(cb).toHaveBeenCalledTimes(2);
  });

  it('unsubscribe stops notifications', () => {
    const s = new Store({ v: 0 });
    const cb = vi.fn();
    const off = s.subscribe(cb);
    off();
    s.set({ v: 1 });
    expect(cb).not.toHaveBeenCalled();
  });

  it('nested set inside a listener is delivered after the current round, in order', () => {
    const s = new Store({ v: 0 });
    const seen: number[][] = [];
    s.subscribe((st, prev) => {
      seen.push([1, st.v, prev.v]);
      if (st.v === 1) s.set({ v: 2 });
    });
    s.subscribe((st, prev) => seen.push([2, st.v, prev.v]));
    s.set({ v: 1 });
    // Second listener sees the latest state in the first round; then a second round for the nested change.
    expect(seen[0]).toEqual([1, 1, 0]);
    expect(seen[seen.length - 1][1]).toBe(2);
    expect(s.get().v).toBe(2);
  });

  it('a throwing listener does not break the others', () => {
    const s = new Store({ v: 0 });
    const err = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const cb = vi.fn();
    s.subscribe(() => {
      throw new Error('boom');
    });
    s.subscribe(cb);
    s.set({ v: 1 });
    expect(cb).toHaveBeenCalled();
    err.mockRestore();
  });

  it('shallowEqual', () => {
    expect(shallowEqual({ a: 1 }, { a: 1 })).toBe(true);
    expect(shallowEqual({ a: 1 }, { a: 2 })).toBe(false);
    expect(shallowEqual([1], { 0: 1 })).toBe(false);
    expect(shallowEqual(null, null)).toBe(true);
  });
});
