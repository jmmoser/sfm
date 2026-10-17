import { describe, expect, it } from 'vitest';
import { CompassSensor, referenceAzimuth } from './orientation';
import { azimuthDeg, deviceOrientationFromMatrix, deviceOrientationMatrix } from './normalize';

const D = Math.PI / 180;
function mul(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const o = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) o[3 * r + c] += a[3 * r + k] * b[3 * k + c];
  return o;
}
function rz(d: number): number[] {
  return [Math.cos(d * D), -Math.sin(d * D), 0, Math.sin(d * D), Math.cos(d * D), 0, 0, 0, 1];
}
function rx(d: number): number[] {
  return [1, 0, 0, 0, Math.cos(d * D), -Math.sin(d * D), 0, Math.sin(d * D), Math.cos(d * D)];
}
function angleBetween(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let tr = 0;
  for (let k = 0; k < 9; k++) tr += a[k] * b[k];
  return Math.acos(Math.min(1, Math.max(-1, 0.5 * (tr - 1)))) / D;
}
function angDiff(a: number, b: number): number {
  let d = (a - b) % 360;
  if (d > 180) d -= 360;
  if (d < -180) d += 360;
  return Math.abs(d);
}
function noise(i: number, k: number): number {
  const x = Math.sin(i * 12.9898 + k * 78.233) * 43758.5453;
  return (x - Math.floor(x) - 0.5) * 2;
}

/**
 * Feed an iOS-like session: true ENU attitude R_enu(t), arbitrary frame R_arb = Rz(ψ)·R_enu,
 * webkitCompassHeading = azimuth of the reported reference direction.
 */
function feedIOS(
  c: CompassSensor,
  psi: number,
  n: number,
  pose: (i: number) => Float64Array,
  reported: (Renu: Float64Array, i: number) => number,
  acc = 10,
  t0 = 1000,
): { t: number; Renu: Float64Array }[] {
  const out: { t: number; Renu: Float64Array }[] = [];
  for (let i = 0; i < n; i++) {
    const t = t0 + i * (1000 / 60);
    const Renu = pose(i);
    const Rarb = mul(rz(psi), Renu);
    const e = deviceOrientationFromMatrix(Rarb);
    c.handleOrientation({
      timeStamp: t,
      alpha: e.alpha,
      beta: e.beta,
      gamma: e.gamma,
      webkitCompassHeading: (reported(Renu, i) + 2 * noise(i, 1) + 360) % 360,
      webkitCompassAccuracy: acc,
    });
    out.push({ t, Renu });
  }
  return out;
}
const topAzimuth = (R: Float64Array): number => azimuthDeg(R[1], R[4]);
const camAzimuth = (R: Float64Array): number => azimuthDeg(-R[2], -R[5]);

describe('referenceAzimuth', () => {
  it('uses the device top when it is near horizontal, the camera axis when upright', () => {
    const flat = mul(rz(-30), rx(0)); // top at azimuth 30
    expect(referenceAzimuth(flat).azimuth).toBeCloseTo(30, 9);
    expect(referenceAzimuth(flat).weight).toBeCloseTo(1, 9);
    const upright = mul(rz(-75), rx(90)); // camera (−z) at azimuth 75
    expect(referenceAzimuth(upright).azimuth).toBeCloseTo(75, 9);
    expect(camAzimuth(upright)).toBeCloseTo(75, 9);
  });
});

describe('CompassSensor (iOS: arbitrary alpha + webkitCompassHeading)', () => {
  it('estimates the yaw offset ψ and returns ENU attitude and headings (tilted phone)', () => {
    const c = new CompassSensor({ target: null, screenAngle: () => 0 });
    const psi = 40;
    const pose = (i: number): Float64Array => mul(rz(-(100 + 0.15 * i)), rx(30 + 5 * Math.sin(i / 50)));
    expect(c.attitudeAt(1000)).toBeNull();
    const rec = feedIOS(c, psi, 400, pose, topAzimuth);
    expect(angDiff(c.yawOffsetDeg!, psi)).toBeLessThan(0.5);
    expect(c.yawAccuracyDeg!).toBeGreaterThan(5);
    expect(c.headingModel).toBe('device-top');
    const { t, Renu } = rec[250];
    const att = c.attitudeAt(t + 3)!; // interpolated between samples
    expect(att.absolute).toBe(true);
    expect(angleBetween(att.R_world_device, Renu)).toBeLessThan(1);
    const h = c.headingAt(t)!;
    expect(h.source).toBe('compass');
    expect(angDiff(h.heading, topAzimuth(Renu))).toBeLessThan(3);
    expect(h.accuracy).toBe(10);
    const cam = c.cameraHeadingAt(t)!;
    expect(angDiff(cam.heading, camAzimuth(Renu))).toBeLessThan(1);
    c.setDeclination(10);
    expect(angDiff(c.headingAt(t)!.heading, topAzimuth(Renu) + 10)).toBeLessThan(3);
    expect(angDiff(topAzimuth(c.attitudeAt(t)!.R_world_device), topAzimuth(Renu) + 10)).toBeLessThan(1);
  });

  it('upright capture pose: ψ from the camera-direction model', () => {
    const c = new CompassSensor({ target: null, screenAngle: () => 0 });
    const psi = 290;
    const pose = (i: number): Float64Array => mul(rz(-(20 + 0.3 * i)), rx(88));
    const rec = feedIOS(c, psi, 600, pose, camAzimuth);
    expect(angDiff(c.yawOffsetDeg!, psi)).toBeLessThan(1);
    const { t, Renu } = rec[500];
    expect(angleBetween(c.attitudeAt(t)!.R_world_device, Renu)).toBeLessThan(1.5);
  });

  it('ignores invalid (accuracy −1) or inaccurate readings', () => {
    const c = new CompassSensor({ target: null });
    const pose = (): Float64Array => mul(rz(-10), rx(20));
    feedIOS(c, 10, 100, pose, topAzimuth, -1);
    expect(c.yawOffsetDeg).toBeNull();
    const a = c.attitudeAt(1500)!;
    expect(a.absolute).toBe(false);
    expect(a.yawAccuracyDeg).toBeNull();
    expect(c.headingAt(1500)).toBeNull();
    feedIOS(c, 10, 100, pose, topAzimuth, 60, 5000);
    expect(c.yawOffsetDeg).toBeNull();
    expect(c.stats.rejectedAccuracy).toBe(100);
  });

  it('switches to the UI-top heading model when that one is consistent across UI rotations', () => {
    let theta = 0;
    const c = new CompassSensor({ target: null, screenAngle: () => theta });
    const psi = 120;
    const pose = (i: number): Float64Array => mul(rz(-(0.2 * i)), rx(15));
    // Platform reports the heading of the UI top: device top at θ=0, right edge (+x) at θ=90.
    const uiTopAz = (R: Float64Array): number => (theta === 90 ? azimuthDeg(R[0], R[3]) : topAzimuth(R));
    for (let k = 0; k < 6; k++) {
      theta = k % 2 === 0 ? 0 : 90;
      feedIOS(c, psi, 120, (i) => pose(i + 120 * k), uiTopAz, 8, 1000 + k * 2000);
    }
    expect(c.headingModel).toBe('ui-top');
    expect(angDiff(c.yawOffsetDeg!, psi)).toBeLessThan(1);
  });
});

describe('CompassSensor north reference (magnetic → true via WMM)', () => {
  it('reports magnetic headings until a location / declination is known, then true north', () => {
    const c = new CompassSensor({ target: null, screenAngle: () => 0 });
    const pose = (i: number): Float64Array => mul(rz(-(40 + 0.2 * i)), rx(25));
    const rec = feedIOS(c, 200, 300, pose, topAzimuth, 8);
    const { t, Renu } = rec[250];
    const mag = c.headingAt(t)!;
    expect(mag.north).toBe('magnetic');
    expect(mag.declinationDeg).toBe(0);
    expect(c.northReference).toBe('magnetic');
    expect(c.declinationInfo).toBeNull();
    expect(c.attitudeAt(t)!.north).toBe('magnetic');
    // Seattle, 2026-10-17: WMM2025 declination ≈ +15.0° (east)
    const e = c.setLocation(47.6062, -122.3321, 56, Date.UTC(2026, 9, 17))!;
    expect(e.declination).toBeGreaterThan(14.5);
    expect(e.declination).toBeLessThan(15.5);
    const tru = c.headingAt(t)!;
    expect(tru.north).toBe('true');
    expect(tru.declinationDeg).toBe(e.declination);
    expect(angDiff(tru.heading, mag.heading + e.declination)).toBeLessThan(1e-9);
    expect(angDiff(topAzimuth(c.attitudeAt(t)!.R_world_device), topAzimuth(Renu) + e.declination)).toBeLessThan(1);
    expect(c.attitudeAt(t)!.north).toBe('true');
    expect(c.declinationInfo).toMatchObject({ source: 'wmm', latDeg: 47.6062, lonDeg: -122.3321, altM: 56, blackoutZone: false, inValidityPeriod: true });
    // invalid location: nothing changes
    expect(c.setLocation(NaN, 0)).toBeNull();
    expect(c.northReference).toBe('true');
    // explicit declination, then back to magnetic
    c.setDeclination(-3);
    expect(c.headingAt(t)!).toMatchObject({ north: 'true', declinationDeg: -3 });
    expect(c.declinationInfo).toEqual({ deg: -3, source: 'manual' });
    c.setDeclination(null);
    expect(c.headingAt(t)!).toMatchObject({ north: 'magnetic', declinationDeg: 0 });
    expect(angDiff(c.headingAt(t)!.heading, mag.heading)).toBeLessThan(1e-9);
    // a declination in the constructor means true north from the start
    expect(new CompassSensor({ target: null, declinationDeg: 0 }).northReference).toBe('true');
  });

  it('setLocationFromFix: first usable fix, refined once by a precise fix, never over a manual declination', () => {
    const c = new CompassSensor({ target: null });
    const date = Date.UTC(2026, 9, 17);
    const fix = (lat: number, lon: number, accuracy: number) => ({ t: 0, lat, lon, alt: 50, accuracy, altitudeAccuracy: null });
    expect(c.setLocationFromFix(null)).toBe(false);
    expect(c.setLocationFromFix(fix(47.6, -122.3, 80_000), date)).toBe(false); // too coarse
    expect(c.northReference).toBe('magnetic');
    expect(c.setLocationFromFix(fix(47.7, -122.2, 3000), date)).toBe(true); // approximate (Precise Location off)
    const d0 = c.declinationDeg;
    expect(c.northReference).toBe('true');
    expect(c.setLocationFromFix(fix(47.6, -122.3, 500), date)).toBe(false); // not precise enough to refine
    expect(c.setLocationFromFix(fix(47.6062, -122.3321, 8), date)).toBe(true); // precise: refined once
    expect(Math.abs(c.declinationDeg - d0)).toBeLessThan(0.2);
    expect(c.declinationInfo!.latDeg).toBe(47.6062);
    expect(c.setLocationFromFix(fix(47.61, -122.33, 5), date)).toBe(false); // first precise fix kept
    c.setDeclination(12);
    expect(c.setLocationFromFix(fix(47.6, -122.3, 5), date)).toBe(false);
    expect(c.declinationDeg).toBe(12);
  });
});

describe('CompassSensor (deviceorientationabsolute)', () => {
  it('uses absolute alpha/beta/gamma as ENU attitude', () => {
    const c = new CompassSensor({ target: null });
    for (let i = 0; i < 30; i++) c.handleOrientation({ timeStamp: 100 + i * 16, alpha: 30, beta: 0, gamma: 0, absolute: true }, true);
    const a = c.attitudeAt(200)!;
    expect(a.absolute).toBe(true);
    expect(angleBetween(a.R_world_device, deviceOrientationMatrix(30, 0, 0))).toBeLessThan(1e-9);
    const h = c.headingAt(200)!;
    expect(h.source).toBe('absolute');
    expect(h.heading).toBeCloseTo(330, 9); // compass heading = 360 − alpha for a flat phone
    c.handleOrientation({ timeStamp: 1000, alpha: 30, beta: 90, gamma: 0, absolute: true }, true);
    const cam = c.cameraHeadingAt(1000)!;
    expect(cam.heading).toBeCloseTo(330, 6);
    expect(cam.elevation).toBeCloseTo(0, 6);
  });

  it('listens on the given target', () => {
    const target = new EventTarget();
    const c = new CompassSensor({ target });
    c.start();
    const e = new Event('deviceorientation');
    for (const [k, v] of Object.entries({ timeStamp: 5, alpha: 1, beta: 2, gamma: 3, webkitCompassHeading: 50, webkitCompassAccuracy: 5 }))
      Object.defineProperty(e, k, { value: v });
    target.dispatchEvent(e);
    expect(c.stats.events).toBe(1);
    expect(c.latest()!.heading).toBe(50);
    c.stop();
    target.dispatchEvent(e);
    expect(c.stats.events).toBe(1);
  });
});
