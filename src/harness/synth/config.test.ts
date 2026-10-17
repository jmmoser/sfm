import { describe, expect, it } from 'vitest';
import { exposureGainFn, frameTime, intrinsicsFor, resolveConfig, screenAngleFor, trajectoryFor } from './config';

describe('synth config', () => {
  it('scene defaults: object portrait 720x960 8 fps x 25 s; yard landscape 960x720 6 fps x 40 s', () => {
    const o = resolveConfig({ scene: 'object' });
    expect([o.width, o.height, o.fps, o.duration, o.frames]).toEqual([720, 960, 8, 25, 200]);
    expect(screenAngleFor(o)).toBe(0);
    expect(o.gps).toBeNull();
    const y = resolveConfig({ scene: 'yard' });
    expect([y.width, y.height, y.fps, y.duration, y.frames]).toEqual([960, 720, 6, 40, 240]);
    expect(screenAngleFor(y)).toBe(90);
    expect(y.gps?.originLatLonAlt).toEqual([47.6062, -122.3321, 56]);
    expect(y.timeOffsetMs).toBe(25);
  });

  it('intrinsics: same sensor (f) for both orientations, distortion defaults', () => {
    const o = intrinsicsFor(resolveConfig({ scene: 'object' }));
    const y = intrinsicsFor(resolveConfig({ scene: 'yard' }));
    expect(o.f).toBeCloseTo(y.f, 12);
    expect(o.f).toBeCloseTo(480 / Math.tan((32.5 * Math.PI) / 180), 9);
    expect([o.k1, o.k2]).toEqual([-0.03, 0.005]);
    expect(o.cx).toBeCloseTo(359.5 + 1.5, 12);
  });

  it('frame times, trajectory duration and exposure gain range', () => {
    const c = resolveConfig({ scene: 'yard', frames: 40, fps: 1, duration: 40 });
    expect(frameTime(c, 10)).toBe(10);
    expect(trajectoryFor(c).duration).toBe(40);
    const g = exposureGainFn(c);
    for (let i = 0; i < 200; i++) {
      const v = g(i * 0.2, i);
      expect(v).toBeGreaterThanOrEqual(0.95 - 1e-12);
      expect(v).toBeLessThanOrEqual(1.05 + 1e-12);
    }
  });
});
