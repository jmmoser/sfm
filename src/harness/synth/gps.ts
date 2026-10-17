/**
 * GPS simulation and WGS84 geodesy (ENU ↔ geodetic).
 *
 * World frame = local ENU (x east, y north, z up) at `originLatLonAlt`. Fix errors are
 * first-order Gauss–Markov per axis (correlation time `correlationS`, 0 = white) with
 * stationary σ: horizontal per-axis σ = horizontalSigma/√2 (so the RMS 2-D horizontal
 * error is `horizontalSigma`), vertical σ = verticalSigma. The reported `accuracy` /
 * `altitudeAccuracy` fields are the configured sigmas with ±10 % jitter (like iOS,
 * which reports a slowly varying estimate rather than the actual error).
 */
import type { GeoFix } from '../../core/types';
import { SynthRng } from './rng';
import type { Trajectory } from './trajectory';

const WGS84_A = 6378137.0;
const WGS84_F = 1 / 298.257223563;
const WGS84_E2 = WGS84_F * (2 - WGS84_F);
const DEG = Math.PI / 180;

export function geodeticToEcef(latDeg: number, lonDeg: number, h: number): [number, number, number] {
  const lat = latDeg * DEG,
    lon = lonDeg * DEG;
  const s = Math.sin(lat),
    c = Math.cos(lat);
  const N = WGS84_A / Math.sqrt(1 - WGS84_E2 * s * s);
  return [(N + h) * c * Math.cos(lon), (N + h) * c * Math.sin(lon), (N * (1 - WGS84_E2) + h) * s];
}

export function ecefToGeodetic(x: number, y: number, z: number): [number, number, number] {
  const lon = Math.atan2(y, x);
  const p = Math.hypot(x, y);
  let lat = Math.atan2(z, p * (1 - WGS84_E2));
  let h = 0;
  for (let i = 0; i < 10; i++) {
    const s = Math.sin(lat);
    const N = WGS84_A / Math.sqrt(1 - WGS84_E2 * s * s);
    h = p / Math.cos(lat) - N;
    const next = Math.atan2(z, p * (1 - (WGS84_E2 * N) / (N + h)));
    if (Math.abs(next - lat) < 1e-14) {
      lat = next;
      break;
    }
    lat = next;
  }
  const s = Math.sin(lat);
  const N = WGS84_A / Math.sqrt(1 - WGS84_E2 * s * s);
  h = p / Math.cos(lat) - N;
  return [lat / DEG, lon / DEG, h];
}

/** ENU offset (m) at origin → geodetic [latDeg, lonDeg, h]. */
export function enuToGeodetic(origin: readonly [number, number, number], e: number, n: number, u: number): [number, number, number] {
  const [lat0, lon0, h0] = origin;
  const [x0, y0, z0] = geodeticToEcef(lat0, lon0, h0);
  const sl = Math.sin(lat0 * DEG),
    cl = Math.cos(lat0 * DEG);
  const so = Math.sin(lon0 * DEG),
    co = Math.cos(lon0 * DEG);
  // Columns of the ENU→ECEF rotation are the east, north, up unit vectors.
  const dx = -so * e - sl * co * n + cl * co * u;
  const dy = co * e - sl * so * n + cl * so * u;
  const dz = cl * n + sl * u;
  return ecefToGeodetic(x0 + dx, y0 + dy, z0 + dz);
}

/** Geodetic → ENU offset (m) at origin. */
export function geodeticToEnu(
  origin: readonly [number, number, number],
  latDeg: number,
  lonDeg: number,
  h: number,
): [number, number, number] {
  const [lat0, lon0, h0] = origin;
  const [x0, y0, z0] = geodeticToEcef(lat0, lon0, h0);
  const [x, y, z] = geodeticToEcef(latDeg, lonDeg, h);
  const dx = x - x0,
    dy = y - y0,
    dz = z - z0;
  const sl = Math.sin(lat0 * DEG),
    cl = Math.cos(lat0 * DEG);
  const so = Math.sin(lon0 * DEG),
    co = Math.cos(lon0 * DEG);
  return [-so * dx + co * dy, -sl * co * dx - sl * so * dy + cl * dz, cl * co * dx + cl * so * dy + sl * dz];
}

export interface GpsConfig {
  /** WGS84 origin of the ENU world frame. Default Seattle (47.6062, −122.3321, 56 m). */
  originLatLonAlt: [number, number, number];
  rateHz: number;
  /** RMS horizontal (2-D) error, m. Default 3. */
  horizontalSigma: number;
  /** Vertical error σ, m. Default 5. */
  verticalSigma: number;
  /** Error correlation time (s); 0 = white noise. Default 0. */
  correlationS: number;
  seed: number;
}

export const DEFAULT_GPS_CONFIG: GpsConfig = {
  originLatLonAlt: [47.6062, -122.3321, 56],
  rateHz: 1,
  horizontalSigma: 3,
  verticalSigma: 5,
  correlationS: 0,
  seed: 11,
};

export interface SimulatedFix extends GeoFix {
  /** Ground-truth ENU position of the device at t (not available on a real device). */
  trueEnu: [number, number, number];
}

/** 1 Hz (default) fixes of the device position over trajectory time [startS, endS]. */
export function simulateGps(
  traj: Trajectory,
  startS: number,
  endS: number,
  t0Ms: number,
  config: Partial<GpsConfig> = {},
): SimulatedFix[] {
  const c = { ...DEFAULT_GPS_CONFIG, ...config };
  const rng = new SynthRng(c.seed).derive('gps');
  const dt = 1 / c.rateHz;
  const rho = c.correlationS > 0 ? Math.exp(-dt / c.correlationS) : 0;
  const q = Math.sqrt(1 - rho * rho);
  const sh = c.horizontalSigma / Math.SQRT2;
  const sv = c.verticalSigma;
  let ex = sh * rng.gaussian(),
    ey = sh * rng.gaussian(),
    ez = sv * rng.gaussian();
  const out: SimulatedFix[] = [];
  const phase = 0.37 * dt; // fixes are not aligned with frames
  for (let tau = startS + phase; tau <= endS + 1e-9; tau += dt) {
    if (out.length > 0) {
      ex = rho * ex + q * sh * rng.gaussian();
      ey = rho * ey + q * sh * rng.gaussian();
      ez = rho * ez + q * sv * rng.gaussian();
    }
    const p = traj.evaluate(tau).position;
    const [lat, lon, alt] = enuToGeodetic(c.originLatLonAlt, p[0] + ex, p[1] + ey, p[2] + ez);
    out.push({
      t: t0Ms + tau * 1000,
      lat,
      lon,
      alt,
      accuracy: Math.round(c.horizontalSigma * (1 + 0.1 * (2 * rng.next() - 1)) * 10) / 10,
      altitudeAccuracy: Math.round(c.verticalSigma * (1 + 0.1 * (2 * rng.next() - 1)) * 10) / 10,
      trueEnu: [p[0], p[1], p[2]],
    });
  }
  return out;
}
