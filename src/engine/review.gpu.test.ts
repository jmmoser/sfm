/**
 * Integration-review regression (IndexedDB, Chromium): `loadSession` of another session on the
 * same engine must not carry anything of the previous one over (summary stats, FOV prior used by
 * a later refine, resolved capture mode).
 */
import { describe, expect, it } from 'vitest';
import { SessionStore } from '../io/storage';
import { PhotogrammetryEngine } from './engine';
import { packSparseModel, type SparseMetaRecord } from './persistence';

describe('engine loadSession switching sessions (IndexedDB)', () => {
  it('resets the previous session’s stats, FOV prior and resolved mode', async () => {
    const s1 = `review-a-${Date.now()}`;
    const s2 = `review-b-${Date.now()}`;
    try {
      // session 1: a stored sparse model with stats, FOV prior 50°
      const a = new PhotogrammetryEngine({ mode: 'object', quality: 'fast', persist: true, preferGPU: false, sessionId: s1, hfovPriorDeg: 50 });
      await a.init();
      const store = a.engineStore!;
      await store.putBinary(
        'sparse/model',
        packSparseModel({
          cameras: new Map([[0, { width: 4, height: 3, f: 4, cx: 1.5, cy: 1, k1: 0, k2: 0 }]]),
          images: [{ id: 0, cameraId: 0, pose: { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(0, 0, 0) } }],
          numPoints: 1,
          xyz: Float64Array.of(0, 0, 1),
          colors: Uint8Array.of(1, 2, 3),
          errors: Float32Array.of(0.5),
          trackOffsets: Uint32Array.of(0, 1),
          trackImages: Uint32Array.of(0),
          trackFeatures: Uint32Array.of(0),
        }),
      );
      const meta: SparseMetaRecord = {
        cameras: [],
        metric: { metric: false, source: 'none', relativeUncertainty: null, gravityAligned: false, northAligned: false, originLatLonAlt: null },
        stats: { leakCheck: 7 },
      };
      await store.putValue('sparse/meta', meta);
      await a.flush();
      a.dispose();
      // session 2: captured with an 80° prior, nothing refined yet
      const b = new PhotogrammetryEngine({ mode: 'scene', quality: 'fast', persist: true, preferGPU: false, sessionId: s2, hfovPriorDeg: 80 });
      await b.init();
      await b.flush();
      b.dispose();

      const c = new PhotogrammetryEngine({ mode: 'auto', quality: 'balanced', preferGPU: false });
      const sum1 = await c.loadSession(s1);
      expect(sum1.stats.leakCheck).toBe(7);
      expect(sum1.sparse).not.toBeNull();
      expect(c.options.hfovPriorDeg).toBe(50);
      const sum2 = await c.loadSession(s2);
      expect(sum2.sparse).toBeNull();
      expect(sum2.mode).toBe('scene');
      expect(sum2.stats.leakCheck).toBeUndefined();
      expect(c.options.hfovPriorDeg).toBe(80);
      c.dispose();
    } finally {
      const s = await SessionStore.open();
      await s.deleteSession(s1).catch(() => undefined);
      await s.deleteSession(s2).catch(() => undefined);
      s.close();
    }
  }, 120_000);
});
