import { describe, expect, it } from 'vitest';
import {
  ENGINE_ERRORS,
  collectTransferables,
  deserializeError,
  isCancelledError,
  isEngineEvent,
  isEngineRequest,
  isEngineResponse,
  namedError,
  serializeError,
} from '../workers/protocol';

describe('collectTransferables', () => {
  it('collects the buffers of large typed arrays, once each', () => {
    const positions = new Float32Array(3000);
    const colors = new Uint8Array(3000);
    const shared = new ArrayBuffer(8192);
    const a = new Float32Array(shared, 0, 512);
    const b = new Uint32Array(shared, 4096, 256);
    const t = collectTransferables({ sparse: { positions, colors }, list: [a, b], again: positions }, { transferPartialViews: true });
    expect(t).toHaveLength(3);
    expect(t).toContain(positions.buffer);
    expect(t).toContain(colors.buffer);
    expect(t).toContain(shared);
  });

  it('never detaches a pool behind a partial view by default', () => {
    // e.g. a mapper returning `this.points.subarray(0, 3 * n)` of its growing buffer.
    const pool = new Float32Array(30_000);
    const live = pool.subarray(0, 9000);
    const whole = new Float32Array(3000);
    const t = collectTransferables({ positions: live, colors: whole });
    expect(t).toEqual([whole.buffer]);
    structuredClone({ positions: live }, { transfer: t as Transferable[] });
    expect(pool.length).toBe(30_000); // still usable by the sender
    expect(collectTransferables({ positions: live }, { transferPartialViews: true })).toEqual([pool.buffer]);
  });

  it('clones small arrays (poses, intrinsics) by default', () => {
    const pose = { R: new Float64Array(9), t: new Float64Array(3) };
    expect(collectTransferables({ cameras: [{ pose }] })).toHaveLength(0);
    expect(collectTransferables({ cameras: [{ pose }] }, { minByteLength: 0 })).toHaveLength(2);
  });

  it('skips SharedArrayBuffers, detached buffers and Blobs', () => {
    const out: unknown[] = [];
    if (typeof SharedArrayBuffer !== 'undefined') out.push(new Float32Array(new SharedArrayBuffer(4096)));
    const det = new ArrayBuffer(4096);
    structuredClone(det, { transfer: [det] }); // detaches
    out.push(det, new Blob([new Uint8Array(4096)]));
    expect(collectTransferables(out)).toHaveLength(0);
  });

  it('handles cycles, Maps and Sets, and respects maxDepth', () => {
    const big = new Float64Array(1024);
    const o: Record<string, unknown> = { big };
    o.self = o;
    const m = new Map([['k', { arr: new Uint8Array(2048) }]]);
    const s = new Set([new Int16Array(1024)]);
    expect(collectTransferables([o, m, s])).toHaveLength(3);
    const deep = { a: { b: { c: { d: new Float32Array(1024) } } } };
    expect(collectTransferables(deep, { maxDepth: 2 })).toHaveLength(0);
    expect(collectTransferables(deep, { maxDepth: 8 })).toHaveLength(1);
  });

  it('collects ImageBitmap-like transferables when the class exists', () => {
    class FakeBitmap {
      width = 4;
      height = 4;
    }
    const g = globalThis as unknown as { ImageBitmap?: unknown };
    const had = 'ImageBitmap' in g;
    const prev = g.ImageBitmap;
    g.ImageBitmap = FakeBitmap;
    try {
      const bm = new FakeBitmap();
      const t = collectTransferables({ bitmap: bm, t: 1, imu: [{ gyro: [0, 0, 0] }] });
      expect(t).toEqual([bm]);
    } finally {
      if (had) g.ImageBitmap = prev;
      else delete g.ImageBitmap;
    }
  });

  it('ignores primitives and class instances like Date', () => {
    expect(collectTransferables(null)).toEqual([]);
    expect(collectTransferables(42)).toEqual([]);
    expect(collectTransferables({ d: new Date(), e: new Error('x') })).toEqual([]);
  });
});

describe('envelopes', () => {
  it('tells responses, events and requests apart', () => {
    expect(isEngineResponse({ id: 1, ok: true, result: 3 })).toBe(true);
    expect(isEngineResponse({ id: 1, ok: false, error: { name: 'E', message: 'm' } })).toBe(true);
    expect(isEngineResponse({ type: 'progress', id: 1, ok: true })).toBe(false);
    expect(isEngineEvent({ type: 'progress', stage: 'matching', progress: 0.5 })).toBe(true);
    expect(isEngineEvent({ type: 'live', update: {} })).toBe(true);
    expect(isEngineEvent({ type: 'other' })).toBe(false);
    expect(isEngineEvent({ id: 1, ok: true })).toBe(false);
    expect(isEngineRequest({ id: 1, method: 'frame', params: {} })).toBe(true);
    expect(isEngineRequest({ id: 1, method: 'nope', params: {} })).toBe(false);
  });

  it('errors round-trip with name, message and code', () => {
    const e = namedError(ENGINE_ERRORS.cancelled, 'stopped', 'user');
    const s = serializeError(e);
    expect(s).toMatchObject({ name: 'CancelledError', message: 'stopped', code: 'user' });
    const back = deserializeError(structuredClone(s));
    expect(back).toBeInstanceOf(Error);
    expect(back.name).toBe('CancelledError');
    expect(back.message).toBe('stopped');
    expect(back.code).toBe('user');
    expect(isCancelledError(back)).toBe(true);
    expect(isCancelledError(new Error('x'))).toBe(false);
    expect(serializeError('plain')).toEqual({ name: 'Error', message: 'plain' });
    expect(deserializeError(null).message).toMatch(/Unknown/);
  });
});
