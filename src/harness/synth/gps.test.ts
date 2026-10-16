import { describe, expect, it } from 'vitest';
import { enuToGeodetic, geodeticToEnu, simulateGps } from './gps';
import { createStaticTrajectory, createYardTrajectory } from './trajectory';

const ORIGIN: [number, number, number] = [47.6062, -122.3321, 56];

describe('geodesy', () => {
  it('ENU ↔ geodetic round trip (sub-mm up to 2 km)', () => {
    for (const [e, n, u] of [
      [0, 0, 0],
      [10, -20, 1.5],
      [1500, 800, -30],
      [-2000, 1999, 120],
    ]) {
      const [lat, lon, h] = enuToGeodetic(ORIGIN, e, n, u);
      const back = geodeticToEnu(ORIGIN, lat, lon, h);
      expect(Math.abs(back[0] - e)).toBeLessThan(1e-4);
      expect(Math.abs(back[1] - n)).toBeLessThan(1e-4);
      expect(Math.abs(back[2] - u)).toBeLessThan(1e-4);
    }
  });

  it('north / east scale is physically right (1e-3 deg lat ≈ 111.2 m)', () => {
    const [n1] = [geodeticToEnu(ORIGIN, ORIGIN[0] + 0.001, ORIGIN[1], ORIGIN[2])[1]];
    expect(n1).toBeGreaterThan(110.9);
    expect(n1).toBeLessThan(111.4);
    const e1 = geodeticToEnu(ORIGIN, ORIGIN[0], ORIGIN[1] + 0.001, ORIGIN[2])[0];
    expect(e1).toBeCloseTo(111.32 * Math.cos((ORIGIN[0] * Math.PI) / 180) * 1, 0);
    // +lon is east, +lat is north
    expect(e1).toBeGreaterThan(0);
  });
});

describe('simulateGps', () => {
  it('1 Hz fixes around the true path with 3 m horizontal / 5 m vertical RMS', () => {
    const traj = createStaticTrajectory([1, 0, 0, 0, 1, 0, 0, 0, 1], [5, -3, 1.5], 0, 4000);
    const fixes = simulateGps(traj, 0, 3000, 1000, {});
    expect(fixes.length).toBeGreaterThan(2995);
    expect(fixes[1].t - fixes[0].t).toBeCloseTo(1000, 6);
    let h2 = 0,
      v2 = 0,
      me = 0;
    for (const f of fixes) {
      const [e, n, u] = geodeticToEnu(ORIGIN, f.lat, f.lon, f.alt!);
      h2 += (e - 5) ** 2 + (n + 3) ** 2;
      v2 += (u - 1.5) ** 2;
      me += e - 5;
      expect(f.accuracy).toBeGreaterThan(2.6);
      expect(f.accuracy).toBeLessThan(3.4);
      expect(f.altitudeAccuracy!).toBeGreaterThan(4.4);
    }
    const hr = Math.sqrt(h2 / fixes.length),
      vr = Math.sqrt(v2 / fixes.length);
    expect(hr).toBeGreaterThan(2.8);
    expect(hr).toBeLessThan(3.2);
    expect(vr).toBeGreaterThan(4.6);
    expect(vr).toBeLessThan(5.4);
    expect(Math.abs(me / fixes.length)).toBeLessThan(0.2);
  });

  it('records the true ENU position and is deterministic', () => {
    const traj = createYardTrajectory();
    const a = simulateGps(traj, 0, 40, 10000, { seed: 5 });
    const b = simulateGps(traj, 0, 40, 10000, { seed: 5 });
    expect(a).toEqual(b);
    const p = traj.evaluate((a[3].t - 10000) / 1000).position;
    expect(a[3].trueEnu[0]).toBeCloseTo(p[0], 12);
    expect(a.length).toBe(40);
  });
});
