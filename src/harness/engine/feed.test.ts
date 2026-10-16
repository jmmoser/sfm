import { describe, expect, it } from 'vitest';
import type { ImuSample } from '../../core/types';
import { cleanGeoFix, compassHeading, gravityDownAt, hfovLongSideDeg, latestAtOrBefore, numStats, planFrames, samplesBetween, upperBound } from './feed';

const frames = (n: number, fps: number, t0 = 10_000): { index: number; t: number }[] =>
  Array.from({ length: n }, (_, i) => ({ index: i, t: t0 + (1000 * i) / fps }));

describe('planFrames', () => {
  it('keeps everything by default, in time order', () => {
    const f = frames(10, 8).reverse();
    expect(planFrames(f).map((x) => x.index)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('applies start, stride and maxFrames', () => {
    const f = frames(40, 8);
    expect(planFrames(f, { startFrame: 2, frameStride: 4 }).map((x) => x.index)).toEqual([2, 6, 10, 14, 18, 22, 26, 30, 34, 38]);
    expect(planFrames(f, { frameStride: 4, maxFrames: 3 }).map((x) => x.index)).toEqual([0, 4, 8]);
  });

  it('rate-limits like a pump with half a frame of tolerance', () => {
    // 30 fps source, 10 Hz pump → exactly every 3rd frame.
    expect(planFrames(frames(30, 30), { fps: 10, datasetFps: 30 }).map((x) => x.index)).toEqual([0, 3, 6, 9, 12, 15, 18, 21, 24, 27]);
    // 8 fps source, 10 Hz pump → every frame.
    expect(planFrames(frames(8, 8), { fps: 10, datasetFps: 8 })).toHaveLength(8);
    // 8 fps source, 3 Hz pump: 333 ms − 62.5 ms tolerance ⇒ every 3rd frame (375 ms).
    expect(planFrames(frames(9, 8), { fps: 3, datasetFps: 8 }).map((x) => x.index)).toEqual([0, 3, 6]);
  });
});

describe('time lookups', () => {
  const arr = [{ t: 1 }, { t: 2 }, { t: 2 }, { t: 5 }];
  it('upperBound / samplesBetween use (t0, t1]', () => {
    expect(upperBound(arr, 0)).toBe(0);
    expect(upperBound(arr, 2)).toBe(3);
    expect(upperBound(arr, 9)).toBe(4);
    expect(samplesBetween(arr, 1, 5)).toEqual([{ t: 2 }, { t: 2 }, { t: 5 }]);
    expect(samplesBetween(arr, 2, 4)).toEqual([]);
    expect(samplesBetween(arr, -Infinity, 1)).toEqual([{ t: 1 }]);
    expect(samplesBetween(arr, 5, 5)).toEqual([]);
  });

  it('latestAtOrBefore', () => {
    expect(latestAtOrBefore(arr, 0.5)).toBeNull();
    expect(latestAtOrBefore(arr, 3)).toEqual({ t: 2 });
    expect(latestAtOrBefore(null, 3)).toBeNull();
  });

  it('consecutive frames get disjoint IMU slices covering the whole stream', () => {
    const imu = Array.from({ length: 200 }, (_, i) => ({ t: 9500 + i * (1000 / 60) }));
    const plan = planFrames(frames(20, 8), { frameStride: 3 });
    let prev = plan[0].t - 1000;
    let total = 0;
    for (const f of plan) {
      const s = samplesBetween(imu, prev, f.t);
      for (const x of s) expect(x.t > prev && x.t <= f.t).toBe(true);
      total += s.length;
      prev = f.t;
    }
    expect(total).toBe(upperBound(imu, plan[plan.length - 1].t) - upperBound(imu, plan[0].t - 1000));
  });
});

describe('gravityDownAt', () => {
  const s = (t: number, accel: [number, number, number] | null, accelG: [number, number, number] | null): ImuSample => ({ t, gyro: [0, 0, 0], accel, accelG });

  it('is the unit DOWN vector from accelG − accel', () => {
    // Upright portrait, phone accelerating sideways: specific force of gravity = +y (up) in the device frame.
    const g = gravityDownAt([s(0, [1, 0, 0], [1, 9.81, 0])], 10)!;
    expect(g).toBeInstanceOf(Float64Array);
    expect(g[0]).toBeCloseTo(0, 12);
    expect(g[1]).toBeCloseTo(-1, 12);
    expect(g[2]).toBeCloseTo(0, 12);
  });

  it('falls back to accelG without linear acceleration and needs a recent sample', () => {
    const g = gravityDownAt([s(0, null, [0, 0, 9.81])], 50)!;
    expect(g[2]).toBeCloseTo(-1, 12);
    expect(gravityDownAt([s(0, null, [0, 0, 9.81])], 500)).toBeNull();
    expect(gravityDownAt([s(0, null, null)], 10)).toBeNull();
    expect(gravityDownAt([], 10)).toBeNull();
  });
});

describe('misc', () => {
  it('hfovLongSideDeg matches the synthetic datasets (65° on the long side)', () => {
    const f = 480 / Math.tan((32.5 * Math.PI) / 180);
    expect(hfovLongSideDeg(f, 720, 960)).toBeCloseTo(65, 10);
    expect(hfovLongSideDeg(f, 960, 720)).toBeCloseTo(65, 10);
  });

  it('cleanGeoFix strips dataset-only fields', () => {
    const g = { t: 1, lat: 2, lon: 3, alt: 4, accuracy: 5, altitudeAccuracy: 6, trueEnu: [1, 2, 3] };
    expect(cleanGeoFix(g)).toEqual({ t: 1, lat: 2, lon: 3, alt: 4, accuracy: 5, altitudeAccuracy: 6 });
    expect(cleanGeoFix(null)).toBeNull();
  });

  it('numStats', () => {
    expect(numStats([])).toEqual({ n: 0, mean: null, median: null, p90: null, max: null });
    const st = numStats([5, 1, 3, NaN, 2, 4]);
    expect(st).toEqual({ n: 5, mean: 3, median: 3, p90: 5, max: 5 });
  });
});

describe('compassHeading (CompassSensor reference: device top within 45° of horizontal, else optical axis)', () => {
  const row = { t: 0, heading: 85, accuracy: 5, headingDeviceTop: 355 };
  it('landscape, upright phone (device top horizontal): device-top azimuth', () => {
    // θ = 90 upright: gravity DOWN along device +x
    expect(compassHeading(row, [1, 0, 0])).toEqual({ heading: 355, accuracy: 5 });
  });
  it('portrait, upright or tilted (device top > 45° above horizontal): optical-axis azimuth', () => {
    expect(compassHeading(row, [0, -1, 0])).toEqual({ heading: 85, accuracy: 5 });
    expect(compassHeading(row, [0, -0.8, -0.6])).toEqual({ heading: 85, accuracy: 5 });
  });
  it('falls back to the optical axis without gravity or device-top heading', () => {
    expect(compassHeading(row, null)).toEqual({ heading: 85, accuracy: 5 });
    expect(compassHeading({ ...row, headingDeviceTop: null }, [1, 0, 0])).toEqual({ heading: 85, accuracy: 5 });
    expect(compassHeading(null, [1, 0, 0])).toBeNull();
  });
});
