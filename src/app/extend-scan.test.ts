/**
 * "Add more views": what Finish of a continued capture runs again (rebuildSteps) and how the
 * confirm sheet says it (rebuiltPhrase) — the stages the scan had, not more and not fewer.
 */
import { describe, expect, it } from 'vitest';
import { isProcessed, rebuildSteps, rebuiltPhrase } from './extend-scan';

describe('extend scan: stages rebuilt after the refine', () => {
  it('rebuilds the stages the scan had, and densify with auto-dense', () => {
    expect(rebuildSteps('meshed', false)).toEqual(['densify', 'mesh']);
    expect(rebuildSteps('dense', false)).toEqual(['densify']);
    expect(rebuildSteps('sparse', false)).toEqual([]);
    expect(rebuildSteps('sparse', true)).toEqual(['densify']);
    // an interrupted capture (or an unknown status): what a new scan would run
    expect(rebuildSteps('capturing', true)).toEqual(['densify']);
    expect(rebuildSteps(null, false)).toEqual([]);
  });

  it('says what is rebuilt', () => {
    expect(rebuiltPhrase([])).toBe('the 3D model');
    expect(rebuiltPhrase(['densify'])).toBe('the 3D model and the dense cloud');
    expect(rebuiltPhrase(['densify', 'mesh'])).toBe('the 3D model, the dense cloud and the mesh');
    expect(isProcessed('meshed') && isProcessed('sparse') && !isProcessed('captured')).toBe(true);
  });
});
