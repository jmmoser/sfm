import { describe, expect, it } from 'vitest';
import { LowLightMonitor, lumaStats } from './frame-ingest';

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

describe('LowLightMonitor', () => {
  const run = (m: LowLightMonitor, frames: [number, boolean][]): boolean[] => frames.map(([l, b]) => m.push(l, b));

  it('a dark room: the median of 5 frames below 40, cleared above 50', () => {
    const m = new LowLightMonitor({ warmupFrames: 0 });
    // bright, then the light goes: dark from the 3rd dark frame (median of the window)
    expect(run(m, [[110, false], [105, false], [30, false], [28, false], [25, false]])).toEqual([false, false, false, false, true]);
    // 45 is not bright enough to clear (hysteresis), ≥ 50 is
    expect(run(m, [[45, false], [46, false], [47, false]])).toEqual([true, true, true]);
    expect(run(m, [[60, false], [62, false], [64, false]])).toEqual([true, true, false]);
  });

  it('one dark frame (a hand over the lens) is not a dark room; NaN frames are ignored', () => {
    const m = new LowLightMonitor({ warmupFrames: 0 });
    expect(run(m, [[0, false], [0, false]])).toEqual([false, false]); // < 3 frames
    m.reset();
    // after a reset (start / new camera segment) the first 8 frames are auto-exposure settling
    const w = new LowLightMonitor();
    expect(run(w, new Array<[number, boolean]>(10).fill([5, false]))).toEqual([false, false, false, false, false, false, false, false, false, false]);
    expect(w.push(5, false)).toBe(true); // the 3rd dark frame after the warm-up
    expect(run(m, [[100, false], [100, false], [3, false], [100, false], [NaN, false], [100, false]]).some(Boolean)).toBe(false);
  });

  it('dim and blurry: below 70 with 3 of 5 frames blur-rejected', () => {
    const m = new LowLightMonitor({ warmupFrames: 0 });
    expect(run(m, [[60, false], [62, false], [61, true], [63, true]])).toEqual([false, false, false, false]);
    expect(m.push(60, true)).toBe(true); // 3 blur rejects in the window
    // stays while ≥ 2 blur rejects remain in the window (dim), clears once the blur is gone
    expect(run(m, [[62, false], [64, false], [63, false], [65, false]])).toEqual([true, true, true, false]);
    // dim but sharp is fine; bright and blurry is motion blur, not darkness
    const b = new LowLightMonitor({ warmupFrames: 0 });
    expect(run(b, [[120, true], [125, true], [118, true], [121, true], [119, true]]).some(Boolean)).toBe(false);
  });
});
