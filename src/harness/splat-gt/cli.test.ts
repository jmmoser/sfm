import { describe, expect, it } from 'vitest';
import { parseArgs } from '../../../e2e/run-splat-gt.mjs';

describe('run-splat-gt parseArgs', () => {
  it('--train-frames / --eval-at take comma-separated integers ≥ 0, without empty items or repeats', () => {
    expect(parseArgs(['object-real', '--train-frames', '0,8,16']).cfg.trainFrames).toEqual([0, 8, 16]);
    expect(parseArgs(['object-real', '--eval-at', '500,1000']).cfg.evalAt).toEqual([500, 1000]);
    // an empty item (Number('') is 0) adds frame / iteration 0; a repeat trains a view twice
    for (const bad of ['4,8,', ',', '4,4,8', 'a,b', '-1,3']) {
      expect(() => parseArgs(['object-real', '--train-frames', bad])).toThrow();
      expect(() => parseArgs(['object-real', '--eval-at', bad])).toThrow();
    }
  });
});
