/**
 * Dense preset resolution: the engine table ← `DENSE_PRESETS` of src/dense (tuned by the dense
 * owner, read defensively) ← EngineConfig.dense overrides; the option blocks of the tuned presets
 * reach the dense module.
 */
import { describe, expect, it } from 'vitest';
import * as dense from '../dense';
import { QUALITY_PRESETS, externalDensePreset, resolveDensePreset } from './presets';

const tuned = (dense as unknown as { DENSE_PRESETS?: Record<string, Record<string, unknown>> }).DENSE_PRESETS;

describe('dense presets', () => {
  it('takes the flat numbers and the option blocks of DENSE_PRESETS when src/dense exports it', () => {
    for (const q of ['fast', 'balanced', 'max'] as const) {
      const ext = externalDensePreset(q);
      const r = resolveDensePreset(q);
      if (!tuned) {
        expect(ext).toEqual({});
        expect(r).toEqual(QUALITY_PRESETS[q].dense);
        continue;
      }
      const t = tuned[q];
      for (const k of ['maxImageSize', 'maxViews', 'numSources', 'iterations', 'geometricPasses', 'geometricIterations', 'sorK'] as const) {
        if (typeof t[k] === 'number') expect(r[k]).toBe(t[k]);
      }
      if (typeof t.haloFilter === 'boolean') expect(r.haloFilter).toBe(t.haloFilter);
      for (const k of ['patchMatch', 'viewSelection', 'fusion'] as const) {
        if (t[k] && typeof t[k] === 'object') expect(r[k]).toEqual(t[k]);
      }
    }
  });

  it('explicit overrides win over both tables', () => {
    const r = resolveDensePreset('fast', { maxImageSize: 480, haloFilter: false, fusion: { maxNormalErrorDeg: 9 } });
    expect(r.maxImageSize).toBe(480);
    expect(r.haloFilter).toBe(false);
    expect(r.fusion).toEqual({ maxNormalErrorDeg: 9 });
  });
});
