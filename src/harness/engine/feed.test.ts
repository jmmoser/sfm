import { describe, expect, it } from 'vitest';
import type { ImuSample } from '../../core/types';
import { declination } from '../../sensors/wmm';
import { frameMeta, type RecordedFrame } from '../../io/recording';
import { pumpFrame } from '../../io/recording-testdata';
import { HeadingNorthEmulator, cleanGeoFix, compassHeading, gravityDownAt, hfovLongSideDeg, latestAtOrBefore, numStats, planFrames, samplesBetween, upperBound, verbatimInput } from './feed';

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

  it('keepOrder keeps the given order (a recording: the order the frames were sent in)', () => {
    const f = [{ index: 0, t: 10 }, { index: 1, t: 5 }, { index: 2, t: 20 }];
    expect(planFrames(f, { keepOrder: true }).map((x) => x.index)).toEqual([0, 1, 2]);
    expect(planFrames(f).map((x) => x.index)).toEqual([1, 0, 2]);
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

describe('HeadingNorthEmulator (dataset true-north headings → what the phone reports)', () => {
  const yard: [number, number, number] = [47.6062, -122.3321, 56];
  const date = Date.UTC(2026, 9, 17);
  const D = declination(yard[0], yard[1], yard[2] / 1000, 2026 + (289 + 0.0) / 365); // ≈ +15°
  const fix = { t: 0, lat: 47.60623, lon: -122.33199, alt: 57, accuracy: 3, altitudeAccuracy: 5 };
  const ad = (a: number, b: number): number => Math.abs(((((a - b) % 360) + 540) % 360) - 180);

  it("'magnetic': WMM at the dataset location, flagged magnetic", () => {
    const e = new HeadingNorthEmulator('magnetic', yard, date);
    expect(Math.abs(e.datasetDeclinationDeg - D)).toBeLessThan(0.01);
    const h = e.convert({ heading: 10, accuracy: 5 }, fix)!;
    expect(h.north).toBe('magnetic');
    expect(ad(h.heading, 10 - e.datasetDeclinationDeg)).toBeLessThan(1e-9);
    expect(h.accuracy).toBe(5);
    expect(e.convert(null, fix)).toBeNull();
  });

  it("'app': magnetic until a usable fix was delivered, then true north (CompassSensor.setLocationFromFix)", () => {
    const e = new HeadingNorthEmulator('app', yard, date);
    const before = e.convert({ heading: 100, accuracy: 5 }, null)!;
    expect(before.north).toBe('magnetic');
    expect(ad(before.heading, 100 - e.datasetDeclinationDeg)).toBeLessThan(1e-9);
    const after = e.convert({ heading: 100, accuracy: 5 }, fix)!;
    expect(after.north).toBe('true');
    // declination at the fix (a few metres from the origin) = the dataset's to ≪ 0.01°
    expect(ad(after.heading, 100)).toBeLessThan(1e-3);
    expect(Math.abs(after.declinationDeg! - e.datasetDeclinationDeg)).toBeLessThan(1e-3);
    // stays true north afterwards, also without a fix
    expect(e.convert({ heading: 50, accuracy: 5 }, null)!.north).toBe('true');
  });

  it("'true' and datasets without a location replay the dataset headings unchanged", () => {
    expect(new HeadingNorthEmulator('true', yard, date).convert({ heading: 33, accuracy: 4 }, fix)).toEqual({ heading: 33, accuracy: 4, north: 'true', declinationDeg: null });
    const none = new HeadingNorthEmulator('magnetic', null, date);
    expect(none.mode).toBe('true');
    expect(none.convert({ heading: 33, accuracy: 4 }, null)!.north).toBe('true');
  });
});

describe('verbatimInput', () => {
  const rows: RecordedFrame[] = Array.from({ length: 5 }, (_, i) => ({ seq: i, sentAt: 0, input: frameMeta(pumpFrame(i)), image: null }));
  const all = { imu: true, gps: true, heading: true, gravity: true };

  it('is the recorded FrameInput, unchanged, when every row is replayed', () => {
    for (let i = 0; i < rows.length; i++) {
      const v = verbatimInput(rows, i, i - 1, all);
      expect(v).toStrictEqual(rows[i].input);
      expect(v.imu).toBe(rows[i].input.imu);
    }
  });

  it('prepends the IMU samples of the rows skipped since the previous replayed one', () => {
    expect(verbatimInput(rows, 3, 0, all).imu).toEqual([...rows[1].input.imu, ...rows[2].input.imu, ...rows[3].input.imu]);
    expect(verbatimInput(rows, 2, -1, all).imu).toHaveLength(6);
    expect(rows[3].input.imu).toHaveLength(2); // the rows themselves are untouched
  });

  it('withholds the streams turned off', () => {
    const v = verbatimInput(rows, 0, -1, { imu: false, gps: false, heading: false, gravity: true });
    expect(v).toMatchObject({ imu: [], imuCalibrationVersion: -1, geo: null, heading: null, attitude: null, gravityDevice: null });
    expect(v.t).toBe(rows[0].input.t);
    expect(verbatimInput(rows, 0, -1, { ...all, gravity: false }).gravityDevice).toBeNull();
    // a field the frame never had stays absent
    const bare: RecordedFrame[] = [{ seq: 0, sentAt: 0, input: { t: 1, screenAngle: 0, imu: [] }, image: null }];
    expect(Object.keys(verbatimInput(bare, 0, -1, { imu: false, gps: false, heading: false, gravity: false })).sort()).toEqual(['imu', 'screenAngle', 't']);
  });
});
