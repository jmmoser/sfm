/**
 * True-north heading in the world alignment: magnetic → true conversion with the WMM2025
 * declination (first GPS fix / coarse location / none), GPS ⊕ compass yaw fusion with outlier
 * rejection, and the reported north reference.
 */
import { describe, expect, it } from 'vitest';
import type { GeoFix, Mat3 } from '../core/types';
import { mat3Mul, mat3MulVec, mat3Transpose } from '../math/mat3';
import { Rng } from '../math/random';
import { decimalYearFromDate, declination as wmmDeclination } from '../sensors/wmm';
import { cameraFromDevice, simulateGps, simulateImu, simulateKeyframes, walkTrajectory, type SimKeyframes, type SimTrajectory } from './extras-testutil';
import { fuseYaw } from './geo';
import { computeWorldAlignment, type WorldAlignmentOptions, type WorldKeyframe } from './world';

const LOG = !!(globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env.SFM_EXTRAS_LOG;
const DEG = Math.PI / 180;
const SEATTLE: [number, number, number] = [47.6062, -122.3321, 56];
const DATE = Date.UTC(2026, 9, 17);
/** WMM2025 declination in Seattle on the test date (≈ +15° east). */
const DECL = wmmDeclination(SEATTLE[0], SEATTLE[1], SEATTLE[2] / 1000, decimalYearFromDate(DATE));

function wrapDeg(d: number): number {
  let r = (d + 180) % 360;
  if (r < 0) r += 360;
  return r - 180;
}

/** Signed yaw error (deg) of an alignment R: R · R0 maps the true world to the aligned one. */
function yawErrorDeg(R: Mat3, sim: SimKeyframes): number {
  const M = mat3Mul(R, sim.R);
  return Math.atan2(M[3], M[0]) / DEG;
}

interface HeadingSim {
  noiseDeg: number;
  accuracy: number;
  biasDeg?: number;
  north: 'true' | 'magnetic';
  /** Flag written into the keyframes (default: `north`). */
  flag?: 'true' | 'magnetic' | null;
}

/** Keyframes with optical-axis compass headings (true azimuth + bias + noise, minus the declination when magnetic). */
function keyframes(sim: SimKeyframes, traj: SimTrajectory, rng: Rng, h: HeadingSim | null): WorldKeyframe[] {
  const Rw = new Float64Array(9);
  return sim.keyframes.map((k) => {
    const R = mat3Transpose(k.R_wc);
    const t = mat3MulVec(R, k.C);
    for (let j = 0; j < 3; j++) t[j] = -t[j];
    const w: WorldKeyframe = { id: k.id, t: k.t, pose: { R, t }, R_cam_device: k.R_cam_device };
    if (h) {
      traj.Rwc((k.tTrue - 5000) / 1000, Rw);
      const trueHeading = Math.atan2(Rw[2], Rw[5]) / DEG + (h.biasDeg ?? 0) + h.noiseDeg * rng.gaussian();
      w.heading = h.north === 'magnetic' ? trueHeading - DECL : trueHeading;
      w.headingAccuracy = h.accuracy;
      w.headingNorth = h.flag === undefined ? h.north : h.flag;
    }
    return w;
  });
}

function walkScene(seed: number, opts: { distance?: number; gpsSigma?: number } = {}) {
  const rng = new Rng(seed);
  const traj = walkTrajectory(rng, { distance: opts.distance ?? 40 });
  const Rcd = cameraFromDevice(90);
  const imu = simulateImu(traj, Rcd, rng);
  const sim = simulateKeyframes(traj, Rcd, rng, { offsetMs: 20, s: 0.6, posNoise: 0.002 });
  const { fixes } = simulateGps(traj, rng, { sigma: opts.gpsSigma ?? 3, origin: SEATTLE });
  return { rng, traj, imu, sim, fixes };
}

const BASE: Partial<WorldAlignmentOptions> = { compassReference: 'optical-axis', captureDate: DATE };

describe('world alignment: true-north heading', () => {
  it('Seattle declination from WMM2025 is ≈ +15° (sanity of the test setup)', () => {
    expect(DECL).toBeGreaterThan(14.5);
    expect(DECL).toBeLessThan(15.5);
  });

  it('magnetic headings + GPS: converted with the WMM declination at the first fix, fused with the GPS yaw', () => {
    const { rng, traj, imu, sim, fixes } = walkScene(101);
    const kfs = keyframes(sim, traj, rng, { noiseDeg: 5, accuracy: 5, north: 'magnetic' });
    const res = computeWorldAlignment({ keyframes: kfs, imu, gpsFixes: fixes }, BASE);
    const d = res.details;
    const gpsOnly = computeWorldAlignment({ keyframes: kfs, imu, gpsFixes: fixes }, { ...BASE, fuseCompassYaw: false });
    if (LOG)
      console.log('fused: err', yawErrorDeg(res.R, sim).toFixed(2), '° σ', (d.yawStdRad! / DEG).toFixed(2), '| gps-only err', yawErrorDeg(gpsOnly.R, sim).toFixed(2), d.headingStats);
    expect(d.declination).not.toBeNull();
    expect(d.declination!.source).toBe('gps');
    // first fix is a few metres from the GPS origin: same declination to 1e-3°
    expect(Math.abs(d.declination!.deg - DECL)).toBeLessThan(1e-3);
    expect(d.declination!.numConverted).toBe(kfs.length);
    expect(d.north).toBe('true');
    expect(res.metric.northAligned).toBe(true);
    expect(d.yawSource).toBe('gps+compass');
    expect(d.yawFusion!.rejected).toBeNull();
    // inverse-variance fusion: σ below both inputs
    expect(d.yawStdRad!).toBeLessThan(d.gps!.yawStdRad);
    expect(d.yawStdRad!).toBeLessThan(d.compass!.stdRad);
    expect(Math.abs(yawErrorDeg(res.R, sim))).toBeLessThan(2);
    expect(gpsOnly.details.yawSource).toBe('gps');
    expect(d.headingStats).toMatchObject({ north: 'true', declinationSource: 'gps', headingsConverted: kfs.length, compassNorth: 'true' });
    expect(typeof d.headingStats.yawStdDeg).toBe('number');
    expect(typeof d.headingStats.compassMinusGpsYawDeg).toBe('number');
    // translation still consistent with the GPS track (fused yaw used for the offset)
    expect(res.metric.originLatLonAlt).not.toBeNull();
  });

  it('magnetic headings mislabelled as true (≈ 15° off) are rejected against a confident GPS yaw', () => {
    const { rng, traj, imu, sim, fixes } = walkScene(102);
    const kfs = keyframes(sim, traj, rng, { noiseDeg: 5, accuracy: 5, north: 'magnetic', flag: 'true' });
    const res = computeWorldAlignment({ keyframes: kfs, imu, gpsFixes: fixes }, BASE);
    const d = res.details;
    if (LOG) console.log('mislabelled: diff', (d.yawFusion!.differenceRad / DEG).toFixed(1), '° z', d.yawFusion!.disagreementSigma.toFixed(1), 'gps σ', (d.gps!.yawStdRad / DEG).toFixed(2));
    expect(d.yawFusion!.rejected).toBe('compass');
    expect(d.yawSource).toBe('gps');
    expect(d.warnings.some((w) => /compass and GPS yaws disagree/.test(w))).toBe(true);
    expect(Math.abs(yawErrorDeg(res.R, sim))).toBeLessThan(6);
    expect(res.metric.northAligned).toBe(true);
  });

  it('a magnetic disturbance (+40° compass bias) is rejected; a confident GPS wins even against a smaller compass σ', () => {
    const { rng, traj, imu, sim, fixes } = walkScene(103);
    const kfs = keyframes(sim, traj, rng, { noiseDeg: 2, accuracy: 2, biasDeg: 40, north: 'true' });
    const res = computeWorldAlignment({ keyframes: kfs, imu, gpsFixes: fixes }, BASE);
    expect(res.details.compass!.stdRad).toBeLessThan(res.details.gps!.yawStdRad); // compass claims more precision
    expect(res.details.yawFusion!.rejected).toBe('compass');
    expect(res.details.yawSource).toBe('gps');
    expect(Math.abs(yawErrorDeg(res.R, sim))).toBeLessThan(6);
  });

  it('short, noisy GPS track: the compass dominates the fused yaw', () => {
    const { rng, traj, imu, sim, fixes } = walkScene(104, { distance: 22, gpsSigma: 3 });
    const kfs = keyframes(sim, traj, rng, { noiseDeg: 4, accuracy: 4, north: 'magnetic' });
    const res = computeWorldAlignment({ keyframes: kfs, imu, gpsFixes: fixes }, BASE);
    const d = res.details;
    if (LOG) console.log('short track: gps σ', (d.gps!.yawStdRad / DEG).toFixed(2), 'compass σ', (d.compass!.stdRad / DEG).toFixed(2), 'err', yawErrorDeg(res.R, sim).toFixed(2), d.yawFusion);
    expect(d.gps).not.toBeNull();
    expect(d.yawSource === 'gps+compass' || d.yawSource === 'compass').toBe(true);
    expect(d.yawFusion!.compassWeight).toBeGreaterThan(0.5);
    expect(Math.abs(yawErrorDeg(res.R, sim))).toBeLessThan(3);
  });

  it('magnetic headings without any location: magnetic yaw applied, northAligned false; a coarse location or explicit declination makes it true north', () => {
    const { rng, traj, imu, sim } = walkScene(105);
    const kfs = keyframes(sim, traj, rng, { noiseDeg: 5, accuracy: 5, north: 'magnetic' });
    const mag = computeWorldAlignment({ keyframes: kfs, imu }, BASE);
    expect(mag.details.yawSource).toBe('compass');
    expect(mag.details.north).toBe('magnetic');
    expect(mag.metric.northAligned).toBe(false);
    expect(mag.metric.gravityAligned).toBe(true);
    expect(mag.details.declination).toBeNull();
    expect(mag.details.warnings.some((w) => /magnetic/.test(w))).toBe(true);
    expect(mag.details.headingStats.north).toBe('magnetic');
    // +Y is magnetic north: the true-north error is the declination (magnetic north lies D east of true north)
    const eMag = yawErrorDeg(mag.R, sim);
    expect(Math.abs(wrapDeg(eMag - DECL))).toBeLessThan(2);

    const coarse = computeWorldAlignment({ keyframes: kfs, imu }, { ...BASE, coarseLocation: [SEATTLE[0] + 0.1, SEATTLE[1] - 0.1] });
    expect(coarse.details.north).toBe('true');
    expect(coarse.metric.northAligned).toBe(true);
    expect(coarse.details.declination!.source).toBe('coarse');
    expect(Math.abs(coarse.details.declination!.deg - DECL)).toBeLessThan(0.2);
    expect(Math.abs(yawErrorDeg(coarse.R, sim))).toBeLessThan(2);
    // the two differ by exactly the declination
    expect(Math.abs(wrapDeg(eMag - yawErrorDeg(coarse.R, sim) - coarse.details.declination!.deg))).toBeLessThan(1e-6);

    const explicit = computeWorldAlignment({ keyframes: kfs, imu }, { ...BASE, declinationDeg: DECL });
    expect(explicit.details.declination!.source).toBe('explicit');
    expect(Math.abs(yawErrorDeg(explicit.R, sim))).toBeLessThan(2);

    const off = computeWorldAlignment({ keyframes: kfs, imu }, { ...BASE, useMagneticYaw: false });
    expect(off.details.yawSource).toBe('none');
    expect(off.metric.northAligned).toBe(false);
  });

  it('approximate fixes (km accuracy) still give the declination; fixes beyond 50 km accuracy do not', () => {
    const { rng, traj, imu, sim, fixes } = walkScene(106);
    const kfs = keyframes(sim, traj, rng, { noiseDeg: 5, accuracy: 5, north: 'magnetic' });
    // coarse, inaccurate fixes: no GPS alignment (track too short for 5 km σ), but a declination
    const approx: GeoFix[] = fixes.map((f) => ({ ...f, accuracy: 5000, lat: f.lat + 0.02 }));
    const a = computeWorldAlignment({ keyframes: kfs, imu, gpsFixes: approx }, BASE);
    expect(a.details.gps).toBeNull();
    expect(a.details.declination!.source).toBe('gps');
    expect(Math.abs(a.details.declination!.deg - DECL)).toBeLessThan(0.2);
    expect(a.details.yawSource).toBe('compass');
    expect(a.metric.northAligned).toBe(true);
    const useless = fixes.map((f) => ({ ...f, accuracy: 80_000 }));
    const b = computeWorldAlignment({ keyframes: kfs, imu, gpsFixes: useless }, BASE);
    expect(b.details.declination).toBeNull();
    expect(b.details.north).toBe('magnetic');
  });

  it('mixed true / magnetic keyframe headings are all brought to true north', () => {
    const { rng, traj, imu, sim, fixes } = walkScene(107);
    const kfs = keyframes(sim, traj, rng, { noiseDeg: 4, accuracy: 4, north: 'magnetic' });
    // the second half arrived after the app knew the declination (CompassSensor.setLocation)
    const half = kfs.length >> 1;
    for (let i = half; i < kfs.length; i++) {
      kfs[i].heading = kfs[i].heading! + DECL;
      kfs[i].headingNorth = 'true';
    }
    const res = computeWorldAlignment({ keyframes: kfs, imu, gpsFixes: fixes }, { ...BASE, fuseCompassYaw: false });
    expect(res.details.declination!.numConverted).toBe(half);
    // compass alone (not fused) is within a few degrees of true north
    const compassOnly = computeWorldAlignment({ keyframes: kfs, imu }, { ...BASE, coarseLocation: SEATTLE });
    expect(compassOnly.details.yawSource).toBe('compass');
    expect(Math.abs(yawErrorDeg(compassOnly.R, sim))).toBeLessThan(2);
    expect(Math.abs(compassOnly.details.compass!.scatterRad / DEG - 4)).toBeLessThan(1.5); // no 15° split
  });

  it('WMM blackout zone (near the magnetic pole): the compass is not used', () => {
    const { rng, traj, imu, sim } = walkScene(108);
    const kfs = keyframes(sim, traj, rng, { noiseDeg: 5, accuracy: 5, north: 'magnetic' });
    const res = computeWorldAlignment({ keyframes: kfs, imu }, { ...BASE, coarseLocation: [85.7, 139] });
    expect(res.details.declination!.blackoutZone).toBe(true);
    expect(res.details.compass).toBeNull();
    expect(res.details.yawSource).toBe('none');
    expect(res.details.warnings.some((w) => /blackout/.test(w))).toBe(true);
  });

  it('headings default to true north when no flag is given (FrameInfo.heading contract)', () => {
    const { rng, traj, imu, sim } = walkScene(109);
    const kfs = keyframes(sim, traj, rng, { noiseDeg: 5, accuracy: 5, north: 'true', flag: null });
    const res = computeWorldAlignment({ keyframes: kfs, imu }, BASE);
    expect(res.details.north).toBe('true');
    expect(res.metric.northAligned).toBe(true);
    expect(res.details.declination).toBeNull(); // no location: nothing to report, nothing needed
    expect(Math.abs(yawErrorDeg(res.R, sim))).toBeLessThan(2);
  });

  it('calibration: fused yaw error / σ over seeds (z-scores) and RMS vs GPS-only', () => {
    const z: number[] = [];
    let se2 = 0,
      sg2 = 0;
    const seeds = [201, 202, 203, 204, 205, 206];
    for (const seed of seeds) {
      const { rng, traj, imu, sim, fixes } = walkScene(seed, { distance: 30 });
      const kfs = keyframes(sim, traj, rng, { noiseDeg: 5, accuracy: 5, north: 'magnetic' });
      const res = computeWorldAlignment({ keyframes: kfs, imu, gpsFixes: fixes }, BASE);
      const g = computeWorldAlignment({ keyframes: kfs, imu, gpsFixes: fixes }, { ...BASE, fuseCompassYaw: false });
      const e = yawErrorDeg(res.R, sim);
      z.push((e * DEG) / res.details.yawStdRad!);
      se2 += e * e;
      sg2 += yawErrorDeg(g.R, sim) ** 2;
    }
    const zr = Math.sqrt(z.reduce((a, b) => a + b * b, 0) / z.length);
    if (LOG) console.log('fused rms', Math.sqrt(se2 / seeds.length).toFixed(2), '° gps-only rms', Math.sqrt(sg2 / seeds.length).toFixed(2), '° z rms', zr.toFixed(2), z.map((v) => v.toFixed(2)));
    expect(Math.sqrt(se2)).toBeLessThan(Math.sqrt(sg2));
    expect(Math.max(...z.map(Math.abs))).toBeLessThan(3.5);
  });
});

describe('fuseYaw', () => {
  it('inverse-variance weights on the circle (wrap-around at ±π)', () => {
    const f = fuseYaw({ yaw: 179 * DEG, stdRad: 2 * DEG }, { yaw: -179 * DEG, stdRad: 2 * DEG });
    expect(f.used).toBe('both');
    expect(Math.abs(wrapDeg(f.yaw / DEG - 180))).toBeLessThan(1e-9);
    expect(f.differenceRad / DEG).toBeCloseTo(2, 9);
    expect(f.stdRad / DEG).toBeCloseTo(Math.SQRT2, 9);
    expect(f.compassWeight).toBeCloseTo(0.5, 12);
    const g = fuseYaw({ yaw: 0, stdRad: 4 * DEG }, { yaw: 3 * DEG, stdRad: 2 * DEG });
    expect(g.compassWeight).toBeCloseTo(0.8, 12);
    expect(g.yaw / DEG).toBeCloseTo(2.4, 9);
    expect(g.stdRad / DEG).toBeCloseTo(Math.sqrt(1 / (1 / 16 + 1 / 4)), 9);
    expect(g.disagreementSigma).toBeCloseTo(3 / Math.sqrt(20), 9);
  });

  it('rejection: compass dropped against a confident GPS, else the less certain one', () => {
    // confident GPS (3°) vs a compass claiming 1°, 20° apart (6.3σ)
    const a = fuseYaw({ yaw: 0, stdRad: 3 * DEG }, { yaw: 20 * DEG, stdRad: 1 * DEG });
    expect(a.rejected).toBe('compass');
    expect(a.used).toBe('gps');
    expect(a.yaw).toBe(0);
    expect(a.compassWeight).toBe(0);
    // unconfident GPS (15° > 10°) vs a 2° compass, 60° apart
    const b = fuseYaw({ yaw: 0, stdRad: 15 * DEG }, { yaw: 60 * DEG, stdRad: 2 * DEG });
    expect(b.rejected).toBe('gps');
    expect(b.used).toBe('compass');
    expect(b.yaw / DEG).toBeCloseTo(60, 9);
    // a looser gate fuses them
    expect(fuseYaw({ yaw: 0, stdRad: 3 * DEG }, { yaw: 20 * DEG, stdRad: 1 * DEG }, 10).used).toBe('both');
    expect(fuseYaw({ yaw: 0, stdRad: 15 * DEG }, { yaw: 60 * DEG, stdRad: 2 * DEG }, { confidentGpsStdRad: 20 * DEG }).rejected).toBe('compass');
  });

  it('non-finite inputs fall back to the other estimate', () => {
    expect(fuseYaw({ yaw: 0.1, stdRad: NaN }, { yaw: 0.2, stdRad: 0.05 })).toMatchObject({ used: 'compass', rejected: null, yaw: 0.2 });
    expect(fuseYaw({ yaw: 0.1, stdRad: 0.05 }, { yaw: NaN, stdRad: 0.05 })).toMatchObject({ used: 'gps', rejected: null, yaw: 0.1 });
  });
});
