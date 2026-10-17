import { describe, expect, it } from 'vitest';
import { lumaStats } from './frame-ingest';

describe('lumaStats', () => {
  it('is 0 ± 0 for a black frame', () => {
    expect(lumaStats({ width: 4, height: 2, data: new Uint8ClampedArray(32) })).toEqual({ mean: 0, std: 0 });
  });

  it('measures mean and spread of a two-tone frame', () => {
    const data = new Uint8ClampedArray(16);
    data.set([0, 0, 0, 255, 200, 200, 200, 255, 0, 0, 0, 255, 200, 200, 200, 255]);
    const s = lumaStats({ width: 2, height: 2, data });
    expect(s.mean).toBeCloseTo(100, 6);
    expect(s.std).toBeCloseTo(100, 6);
  });
});
