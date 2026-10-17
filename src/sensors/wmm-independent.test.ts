/**
 * Independent cross-check of `wmm.ts` (review): the field is synthesised a DIFFERENT way and
 * compared with the official WMM2025 test values and with `magneticField` at random places, dates
 * and heights (poles included).
 *
 * Method (shares nothing with wmm.ts but the parsed coefficient table, which wmm.test.ts checks
 * against the official WMM.COF): the scalar potential
 *   V(r, θ, λ) = a Σₙ (a/r)ⁿ⁺¹ Σₘ (gₙᵐ cos mλ + hₙᵐ sin mλ) P̆ₙᵐ(cos θ),
 * with the Schmidt semi-normalised P̆ₙᵐ built from the explicit Legendre polynomial coefficients
 * (Rodrigues: Pₙ(x) = 2⁻ⁿ Σₖ (−1)ᵏ C(n,k) C(2n−2k, n) xⁿ⁻²ᵏ, m-th derivative × sinᵐθ ×
 * √(2 (n−m)!/(n+m)!)), B = −∇V by central differences in Cartesian ECEF coordinates, projected
 * onto the geodetic north / east / down axes of the WGS84 position (no geocentric latitude, no
 * recursion, no ψ rotation).
 */
import { describe, expect, it } from 'vitest';
import { Rng } from '../math/random';
import { WMM2025, magneticField } from './wmm';

const N = WMM2025.nMax;
const RE = 6371.2;
const A = 6378.137;
const F = 1 / 298.257223563;
const E2 = F * (2 - F);
const DEG = Math.PI / 180;

function binom(n: number, k: number): number {
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return r;
}
function factorial(n: number): number {
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

/** Per (n, m): coefficients of dᵐPₙ/dxᵐ (ascending powers) and the Schmidt factor. */
const LEG: { c: number[]; s: number }[] = [];
for (let n = 0; n <= N; n++) {
  const p = new Array<number>(n + 1).fill(0);
  for (let k = 0; 2 * k <= n; k++) p[n - 2 * k] = ((k % 2 ? -1 : 1) * binom(n, k) * binom(2 * n - 2 * k, n)) / 2 ** n;
  let q = p;
  for (let m = 0; m <= n; m++) {
    LEG[(n * (n + 1)) / 2 + m] = { c: q.slice(), s: m === 0 ? 1 : Math.sqrt((2 * factorial(n - m)) / factorial(n + m)) };
    q = q.slice(1).map((v, i) => v * (i + 1));
  }
}

function potential(x: number, y: number, z: number, g: ArrayLike<number>, h: ArrayLike<number>): number {
  const r = Math.hypot(x, y, z);
  const ct = z / r;
  const st = Math.hypot(x, y) / r; // sin θ without cancellation near the poles
  const lam = Math.atan2(y, x);
  let v = 0;
  for (let n = 1; n <= N; n++) {
    let t = 0;
    for (let m = 0; m <= n; m++) {
      const k = (n * (n + 1)) / 2 + m;
      const { c, s } = LEG[k];
      let poly = 0;
      for (let i = c.length - 1; i >= 0; i--) poly = poly * ct + c[i];
      t += (g[k] * Math.cos(m * lam) + h[k] * Math.sin(m * lam)) * s * st ** m * poly;
    }
    v += (RE / r) ** (n + 1) * t;
  }
  return RE * v;
}

/** X, Y, Z (north, east, down; nT) of −∇V at a geodetic position. */
function fieldNED(latDeg: number, lonDeg: number, altKm: number, g: ArrayLike<number>, h: ArrayLike<number>): [number, number, number] {
  const la = latDeg * DEG,
    lo = lonDeg * DEG;
  const sla = Math.sin(la),
    cla = Math.abs(latDeg) === 90 ? 0 : Math.cos(la);
  const Nr = A / Math.sqrt(1 - E2 * sla * sla);
  const P = [(Nr + altKm) * cla * Math.cos(lo), (Nr + altKm) * cla * Math.sin(lo), (Nr * (1 - E2) + altKm) * sla];
  const e = 0.002; // km
  const grad = [0, 1, 2].map((i) => {
    const a = P.slice(),
      b = P.slice();
    a[i] += e;
    b[i] -= e;
    return -(potential(a[0], a[1], a[2], g, h) - potential(b[0], b[1], b[2], g, h)) / (2 * e);
  });
  const north = [-sla * Math.cos(lo), -sla * Math.sin(lo), cla];
  const east = [-Math.sin(lo), Math.cos(lo), 0];
  const down = [-cla * Math.cos(lo), -cla * Math.sin(lo), -sla];
  const dot = (u: number[]): number => u[0] * grad[0] + u[1] * grad[1] + u[2] * grad[2];
  return [dot(north), dot(east), dot(down)];
}

function independent(latDeg: number, lonDeg: number, altKm: number, year: number) {
  const dt = year - WMM2025.epoch;
  const g = WMM2025.g.map((v, k) => v + dt * WMM2025.gDot[k]);
  const h = WMM2025.h.map((v, k) => v + dt * WMM2025.hDot[k]);
  const [X, Y, Z] = fieldNED(latDeg, lonDeg, altKm, g, h);
  const [Xd, Yd, Zd] = fieldNED(latDeg, lonDeg, altKm, WMM2025.gDot, WMM2025.hDot);
  const H = Math.hypot(X, Y);
  return { X, Y, Z, D: Math.atan2(Y, X) / DEG, I: Math.atan2(Z, H) / DEG, Xd, Yd, Zd, Dd: (X * Yd - Y * Xd) / (H * H) / DEG };
}

const angleDiff = (a: number, b: number): number => Math.abs((((a - b) % 360) + 540) % 360 - 180);

// A few official rows (WMM2025_TestValues.txt): year, alt km, lat, lon, D, I, X, Y, Z (nT)
const OFFICIAL = [
  [2025.0, 28, 89, -121, -99.77, 88.47, -255.388723, -1482.460628, 56194.288771],
  [2025.0, 3, -50, -103, 27.96, -54.89, 19526.532799, 10362.990980, -31437.562789],
  [2026.0, 82, -64, 87, -81.74, -75.4, 2007.07487, -13829.362571, -53663.514288],
  [2026.5, 12, -79, 115, -137.58, -77.37, -9613.650418, -8785.714482, -58104.306533],
  [2027.5, 0, -13, -59, -17.49, -15.26, 21365.849306, -6732.56062, -6112.31352],
  [2028.0, 75, 79, 125, -18.59, 87.42, 2447.5959, -823.235047, 57308.751756],
  [2029.0, 50, 87, -154, -73.48, 89.07, 257.87742, -869.484879, 55992.308183],
  [2029.5, 77, -18, 138, 4.45, -47.55, 31751.49758, 2472.257962, -34817.395113],
] as const;

describe('WMM2025 — independent field synthesis (review cross-check)', () => {
  it('the independent synthesis reproduces official test values', () => {
    for (const [yr, alt, lat, lon, D, I, X, Y, Z] of OFFICIAL) {
      const e = independent(lat, lon, alt, yr);
      expect(angleDiff(e.D, D)).toBeLessThanOrEqual(0.0051);
      expect(Math.abs(e.I - I)).toBeLessThanOrEqual(0.0051);
      expect(Math.abs(e.X - X)).toBeLessThan(2e-3);
      expect(Math.abs(e.Y - Y)).toBeLessThan(2e-3);
      expect(Math.abs(e.Z - Z)).toBeLessThan(2e-3);
    }
  });

  it('wmm.ts agrees with it at random places, dates and heights (and at / next to the poles)', () => {
    const rng = new Rng(2025);
    const pts: [number, number, number, number][] = [];
    for (let i = 0; i < 300; i++) pts.push([rng.uniform(-90, 90), rng.uniform(-180, 180), rng.uniform(-0.5, 600), rng.uniform(2025, 2030)]);
    for (const lat of [90, -90, 89.9999999, -89.9999999, 89.999]) for (const lon of [0, 33, -120]) pts.push([lat, lon, 0, 2027.3]);
    // Seattle, the e2e yard location
    pts.push([47.6062, -122.3321, 0.056, 2026.79]);
    let maxNt = 0,
      maxD = 0,
      maxRate = 0;
    for (const [lat, lon, alt, yr] of pts) {
      const a = magneticField(lat, lon, alt, yr);
      const b = independent(lat, lon, alt, yr);
      maxNt = Math.max(maxNt, Math.abs(a.x - b.X), Math.abs(a.y - b.Y), Math.abs(a.z - b.Z));
      maxD = Math.max(maxD, angleDiff(a.declination, b.D), Math.abs(a.inclination - b.I));
      maxRate = Math.max(maxRate, Math.abs(a.xRate - b.Xd), Math.abs(a.yRate - b.Yd), Math.abs(a.zRate - b.Zd));
      if (Math.abs(lat) < 89.9) expect(Math.abs(a.declinationRate - b.Dd)).toBeLessThan(1e-5);
    }
    expect(maxNt).toBeLessThan(5e-3);
    expect(maxD).toBeLessThan(1e-4);
    expect(maxRate).toBeLessThan(1e-3);
  });
});
