import { describe, expect, it } from 'vitest';
import { GeoSensor, epochToPerfMs, type GeoPositionLike } from './geolocation';

function pos(epoch: number, lat: number, lon: number, acc = 5, alt: number | null = 30, altAcc: number | null = 8): GeoPositionLike {
  return { timestamp: epoch, coords: { latitude: lat, longitude: lon, accuracy: acc, altitude: alt, altitudeAccuracy: altAcc } };
}

describe('GeoSensor', () => {
  it('converts epoch timestamps to the performance.now() domain', () => {
    expect(epochToPerfMs(1_700_000_001_000, 1_700_000_002_000, 5000)).toBe(4000);
  });

  it('stores fixes in time order, dedupes, nearest() and interpolate()', () => {
    const g = new GeoSensor({ geolocation: null, maxInterpolationGapMs: 1500 });
    const E0 = 1_700_000_000_000;
    const P0 = 10_000; // performance.now() when Date.now() = E0 + 10_000
    const now = (dt: number): [number, number] => [E0 + 10_000 + dt, P0 + dt];
    // fixes at perf times 1000, 2000, 6000 (+ one delivered out of order at 1500)
    g.handlePosition(pos(E0 + 1000, 47.0, 8.0), ...now(0));
    g.handlePosition(pos(E0 + 2000, 47.001, 8.002, 7, 32, 10), ...now(10));
    g.handlePosition(pos(E0 + 6000, 47.004, 8.004, 9, null, null), ...now(20));
    g.handlePosition(pos(E0 + 1500, 47.0005, 8.001), ...now(30));
    expect(g.handlePosition(pos(E0 + 6000, 47.004, 8.004, 9, null, null), ...now(40))).toBeNull(); // duplicate
    expect(g.fixes.map((f) => f.t)).toEqual([1000, 1500, 2000, 6000]);
    expect(g.latest()!.t).toBe(6000);
    expect(g.nearest(1700)!.t).toBe(1500);
    expect(g.nearest(1900)!.t).toBe(2000);
    expect(g.nearest(-1e9)!.t).toBe(1000);
    expect(g.nearest(1e9)!.t).toBe(6000);

    const m = g.interpolate(1750)!;
    expect(m.t).toBe(1750);
    expect(m.lat).toBeCloseTo(47.00075, 10);
    expect(m.lon).toBeCloseTo(8.0015, 10);
    expect(m.alt).toBeCloseTo(31, 10);
    expect(m.accuracy).toBeCloseTo(6, 10);
    // gap 2000 → 6000 is longer than 1500: nearest within 1500 ms or null
    expect(g.interpolate(2500)!.lat).toBeCloseTo(47.001, 10);
    expect(g.interpolate(4000)).toBeNull();
    expect(g.interpolate(7000)!.t).toBe(7000);
    expect(g.interpolate(8000)).toBeNull();
    // altitude null on one side → no altitude interpolation
    expect(g.fixes[3].alt).toBeNull();
    expect(g.fixes[3].altitudeAccuracy).toBeNull();
  });

  it('interpolates across the antimeridian', () => {
    const g = new GeoSensor({ geolocation: null });
    g.handlePosition(pos(1000, 0, 179.9), 1000, 1000);
    g.handlePosition(pos(2000, 0, -179.9), 2000, 2000);
    expect(Math.abs(g.interpolate(1500)!.lon)).toBeCloseTo(180, 9);
  });

  it('flags approximate locations and clamps future timestamps', () => {
    const g = new GeoSensor({ geolocation: null });
    g.handlePosition(pos(1e13, 1, 2, 3000), 1_000_000, 500);
    expect(g.approximate).toBe(true);
    expect(g.fixes[0].t).toBe(500);
  });

  it('drives watchPosition with high accuracy and maximumAge 0; handles denial', () => {
    const calls: { opts?: PositionOptions }[] = [];
    let ok: PositionCallback | null = null;
    let err: PositionErrorCallback | null | undefined = null;
    let cleared = -1;
    const geo = {
      watchPosition: (s: PositionCallback, e?: PositionErrorCallback | null, o?: PositionOptions) => {
        calls.push({ opts: o });
        ok = s;
        err = e;
        return 7;
      },
      clearWatch: (id: number) => {
        cleared = id;
      },
      getCurrentPosition: () => undefined,
    } as unknown as Geolocation;
    const g = new GeoSensor({ geolocation: geo });
    const got: number[] = [];
    g.onFix((f) => got.push(f.lat));
    g.start();
    expect(g.status).toBe('watching');
    expect(calls[0].opts).toMatchObject({ enableHighAccuracy: true, maximumAge: 0 });
    ok!({ timestamp: Date.now(), coords: { latitude: 1, longitude: 2, accuracy: 4, altitude: null, altitudeAccuracy: null } } as unknown as GeolocationPosition);
    expect(got).toEqual([1]);
    err!({ code: 1, message: 'denied' } as GeolocationPositionError);
    expect(g.status).toBe('denied');
    expect(cleared).toBe(7);
  });
});
