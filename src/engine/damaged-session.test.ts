/**
 * A stored session with damaged / foreign-format result records (sparse model, dense cloud, mesh,
 * engine state) must still open: the damaged results are treated as missing (logged) so the scan
 * can be processed again from its photos, instead of the whole scan becoming unopenable.
 */
import { describe, expect, it } from 'vitest';
import { PhotogrammetryEngine } from './engine';
import { MemoryEngineStore, packArrays, unpackArrays, unpackSparseModel } from './persistence';

async function damagedStore(): Promise<MemoryEngineStore> {
  const store = new MemoryEngineStore('damaged');
  await store.ensureSession({ mode: 'object', quality: 'fast', name: 'Damaged' });
  await store.putValue('engine/state', { version: 1, mode: 'drone', quality: 'ultra', captureFinished: true, hasSparse: true, hfovPriorDeg: 'wide', resolvedMode: 'boat' });
  await store.putBinary('sparse/model', new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
  await store.putValue('sparse/meta', { cameras: 'not an array', metric: null, stats: {} });
  await store.putBinary('dense/points', new Uint8Array(64).fill(7));
  await store.putBinary('mesh', new Uint8Array([0x53, 0x46, 0x4d, 0x42, 9, 9, 9, 9]));
  return store;
}

describe('damaged stored results', () => {
  it('unpackArrays / unpackSparseModel reject damaged containers with a clear error (no huge allocation, no TypeError)', () => {
    const good = packArrays({ n: 1 }, [new Float64Array([1, 2]), new Uint8Array([3])]);
    expect(unpackArrays<{ n: number }>(good).arrays.map((a) => Array.from(a))).toEqual([[1, 2], [3]]);
    const bad = (mutate: (dv: DataView, u8: Uint8Array) => void): ArrayBuffer => {
      const b = good.slice(0);
      mutate(new DataView(b), new Uint8Array(b));
      return b;
    };
    const hl = new DataView(good).getUint32(4, true);
    const first = (8 + hl + 7) & ~7;
    for (const b of [
      new ArrayBuffer(4),
      bad((dv) => dv.setUint32(4, 0xffffffff, true)), // header length beyond the buffer
      bad((_dv, u8) => u8.fill(0x20, 8, 8 + hl)), // header JSON blank
      bad((dv) => dv.setUint32(first, 77, true)), // unknown dtype
      bad((dv) => dv.setUint32(first + 4, 0x7fffffff, true)), // count beyond the buffer
      good.slice(0, good.byteLength - 8), // truncated
    ]) {
      expect(() => unpackArrays(b)).toThrow(/damaged|corrupt/i);
    }
    // a well-formed container whose arrays do not match the header
    const mismatch = packArrays({ cameras: [], images: [], numPoints: 5 }, [new Float64Array(3), new Uint8Array(3), new Float32Array(1), new Uint32Array(2), new Uint32Array(0), new Uint32Array(0)]);
    expect(() => unpackSparseModel(mismatch)).toThrow(/damaged|corrupt/i);
  });

  it('loadSession opens the scan without the damaged results (re-processable), with a warning', async () => {
    const store = await damagedStore();
    const e = new PhotogrammetryEngine({ mode: 'auto', quality: 'fast', sessionId: 'damaged', persist: true }, { store, mapping: 'inline' });
    const logs: string[] = [];
    e.on('log', (level, msg) => logs.push(`${level}: ${msg}`));
    const sum = await e.loadSession('damaged');
    expect(sum.name).toBe('Damaged');
    expect(sum.sparse).toBeNull();
    expect(sum.dense).toBeNull();
    expect(sum.mesh).toBeNull();
    expect(sum.status).toBe('captured'); // → Processing (refine) from the stored photos
    expect([sum.mode, sum.quality]).toEqual(['auto', 'fast']); // invalid state values ignored
    expect(logs.filter((l) => l.startsWith('warn:') && /damaged/i.test(l)).length).toBeGreaterThanOrEqual(1);
    e.dispose();
  });
});
