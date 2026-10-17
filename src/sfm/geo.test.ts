import { describe, expect, it } from 'vitest';
import type { GeoFix } from '../core/types';
import { Rng } from '../math/random';
import { rotvecToMat3 } from '../math/rotation';
import { orbitTrajectory, simulateGps, walkTrajectory, type SimTrajectory } from './extras-testutil';
import {
  alignToGps,
  ecefToGeodetic,
  EnuFrame,
  enuToGeodetic,
  geodeticToEcef,
  geodeticToEnu,
  trackExtent,
  WGS84_A,
  WGS84_B,
  WGS84_E2,
  yawFromCompass,
  type CompassObservation,
} from './geo';

const LOG = !!(globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env.SFM_EXTRAS_LOG;

describe('WGS84 conversions', () => {
  it('matches closed forms at reference points', () => {
    const e = geodeticToEcef(0, 0, 0);
    expect(e[0]).toBeCloseTo(WGS84_A, 6);
    expect(Math.abs(e[1]) + Math.abs(e[2])).toBeLessThan(1e-6);
    const p = geodeticToEcef(90, 0, 0);
    expect(p[2]).toBeCloseTo(WGS84_B, 6);
    expect(Math.hypot(p[0], p[1])).toBeLessThan(1e-6);
    const q = geodeticToEcef(0, 90, 100);
    expect(q[1]).toBeCloseTo(WGS84_A + 100, 6);
    // lat 45, lon 0, h = 0: x = N cos φ, z = N (1 − e²) sin φ
    const N = WGS84_A / Math.sqrt(1 - WGS84_E2 * 0.5);
    const r = geodeticToEcef(45, 0, 0);
    expect(r[0]).toBeCloseTo(N * Math.SQRT1_2, 6);
    expect(r[2]).toBeCloseTo(N * (1 - WGS84_E2) * Math.SQRT1_2, 6);
  });

  it('ECEF ⇄ geodetic round trip (incl. poles, depths, orbital heights)', () => {
    const rng = new Rng(1);
    let maxAng = 0,
      maxH = 0;
    const pts: [number, number, number][] = [
      [90, 0, 0],
      [-90, 123, 500],
      [89.9999999, -179.9, 10],
      [0, 180, -400],
      [0, 0, 2e7],
    ];
    for (let i = 0; i < 2000; i++) pts.push([rng.uniform(-90, 90), rng.uniform(-180, 180), rng.uniform(-500, 1e5)]);
    for (const [lat, lon, h] of pts) {
      const x = geodeticToEcef(lat, lon, h);
      const g = ecefToGeodetic(x[0], x[1], x[2]);
      maxAng = Math.max(maxAng, Math.abs(g[0] - lat));
      if (Math.abs(lat) < 89.9999) {
        let dl = Math.abs(g[1] - lon);
        if (dl > 180) dl = 360 - dl;
        maxAng = Math.max(maxAng, dl * Math.cos((lat * Math.PI) / 180));
      }
      maxH = Math.max(maxH, Math.abs(g[2] - h));
    }
    expect(maxAng).toBeLessThan(1e-10); // degrees (≈ 1e-5 m)
    expect(maxH).toBeLessThan(1e-6);
  });

  it('local ENU: axes, round trip, distances', () => {
    const f = new EnuFrame(47.3769, 8.5417, 410);
    const R = f.R;
    // orthonormal
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        const d = R[3 * i] * R[3 * j] + R[3 * i + 1] * R[3 * j + 1] + R[3 * i + 2] * R[3 * j + 2];
        expect(d).toBeCloseTo(i === j ? 1 : 0, 14);
      }
    expect(Math.hypot(...f.fromGeodetic(47.3769, 8.5417, 410))).toBeLessThan(1e-8);
    // 0.001° north ≈ 111.2 m north, slightly down (curvature)
    const nrt = f.fromGeodetic(47.3779, 8.5417, 410);
    expect(Math.abs(nrt[0])).toBeLessThan(1e-6);
    expect(nrt[1]).toBeGreaterThan(111);
    expect(nrt[1]).toBeLessThan(111.4);
    expect(nrt[2]).toBeLessThan(0);
    expect(nrt[2]).toBeGreaterThan(-0.002);
    const up = f.fromGeodetic(47.3769, 8.5417, 510);
    expect(up[2]).toBeCloseTo(100, 8);
    // east: 0.001° east at lat φ ≈ 111.32 km cos φ / 1000
    const est = f.fromGeodetic(47.3769, 8.5427, 410);
    expect(est[0]).toBeCloseTo((Math.PI / 180) * 0.001 * (WGS84_A / Math.sqrt(1 - WGS84_E2 * Math.sin((47.3769 * Math.PI) / 180) ** 2)) * Math.cos((47.3769 * Math.PI) / 180), 2);
    const rng = new Rng(2);
    for (let i = 0; i < 200; i++) {
      const e = rng.uniform(-5000, 5000),
        n = rng.uniform(-5000, 5000),
        u = rng.uniform(-100, 300);
      const g = f.toGeodetic(e, n, u);
      const back = f.fromGeodetic(g[0], g[1], g[2]);
      expect(Math.hypot(back[0] - e, back[1] - n, back[2] - u)).toBeLessThan(1e-6);
    }
    // convenience wrappers
    const a = geodeticToEnu(47.38, 8.55, 420, 47.3769, 8.5417, 410);
    const b = enuToGeodetic(a[0], a[1], a[2], 47.3769, 8.5417, 410);
    expect(b[0]).toBeCloseTo(47.38, 10);
    expect(b[1]).toBeCloseTo(8.55, 10);
    expect(b[2]).toBeCloseTo(420, 6);
  });
});

/** Keyframe centres in a reconstruction frame X_rec = s Rz(ψ) X + t (gravity-aligned) or with a general rotation. */
function recCentres(traj: SimTrajectory, rng: Rng, s: number, general: boolean) {
  const times: number[] = [];
  const C: number[] = [];
  const R = general ? rotvecToMat3([rng.gaussian(), rng.gaussian(), rng.gaussian()]) : rotvecToMat3([0, 0, rng.uniform(-Math.PI, Math.PI)]);
  const t = [rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5)];
  const p = new Float64Array(3);
  for (let tt = 0.3; tt < traj.duration - 0.3; tt += rng.uniform(0.25, 0.5)) {
    traj.pos(tt, p);
    times.push(5000 + 1000 * tt);
    for (let i = 0; i < 3; i++) C.push(s * (R[3 * i] * p[0] + R[3 * i + 1] * p[1] + R[3 * i + 2] * p[2]) + t[i]);
  }
  return { times: Float64Array.from(times), C: Float64Array.from(C), R, t };
}

describe('alignToGps', () => {
  it('4-DOF on a 40 m walk with 3 m GPS noise: scale within 10 %, honest σ', () => {
    const errs: number[] = [];
    let maxZ = 0;
    for (let seed = 0; seed < 12; seed++) {
      const rng = new Rng(100 + seed);
      const traj = walkTrajectory(rng);
      const s = rng.uniform(0.1, 10);
      const rc = recCentres(traj, rng, s, false);
      const { fixes } = simulateGps(traj, rng, { sigma: 3 });
      const res = alignToGps(rc.C, rc.times, fixes, { gravityAligned: true })!;
      expect(res).not.toBeNull();
      expect(res.mode).toBe('4dof');
      const err = res.s * s - 1;
      errs.push(err);
      maxZ = Math.max(maxZ, Math.abs(err) / res.scaleRelUncertainty);
      // yaw: R ≈ Rz(−ψ) where rc.R = Rz(ψ)
      const yawTrue = -Math.atan2(rc.R[3], rc.R[0]);
      const yawEst = Math.atan2(res.R[3], res.R[0]);
      let dy = yawEst - yawTrue;
      dy = Math.atan2(Math.sin(dy), Math.cos(dy));
      expect(Math.abs(dy)).toBeLessThan((10 * Math.PI) / 180);
      expect(res.rmsResidual).toBeGreaterThan(2);
      expect(res.rmsResidual).toBeLessThan(5);
      expect(Math.abs(err)).toBeLessThan(0.1);
    }
    if (LOG) console.log('gps 4dof scale errs %', errs.map((e) => (100 * e).toFixed(1)).join(' '), 'max |z|', maxZ.toFixed(2));
    expect(maxZ).toBeLessThan(3.5);
  });

  it('robust to gross GPS outliers and time-correlated error', () => {
    const rng = new Rng(7);
    const traj = walkTrajectory(rng);
    const rc = recCentres(traj, rng, 2.5, false);
    const { fixes } = simulateGps(traj, rng, { sigma: 3, correlatedSigma: 3, tauS: 8 });
    for (let i = 3; i < fixes.length; i += 9) fixes[i] = { ...fixes[i], lat: fixes[i].lat + 0.0006 }; // ~65 m jumps
    const res = alignToGps(rc.C, rc.times, fixes, { gravityAligned: true })!;
    if (LOG) console.log('outliers+correlated: err %', (100 * (res.s * 2.5 - 1)).toFixed(2), 'σ %', (100 * res.scaleRelUncertainty).toFixed(2), 'ρ', res.residualAutocorrelation.toFixed(2), 'inl', res.numInliers, '/', res.numFixes);
    expect(res.numInliers).toBeLessThan(res.numFixes - 3);
    expect(Math.abs(res.s * 2.5 - 1)).toBeLessThan(0.15);
    expect(Math.abs(res.s * 2.5 - 1)).toBeLessThan(3 * res.scaleRelUncertainty + 0.01);
  });

  it('full Sim3 on a 2-D yard loop with a general rotation', () => {
    const rng = new Rng(9);
    // rectangle-ish loop 30 × 20 m walked at 1.2 m/s
    const loop: SimTrajectory = {
      duration: 90,
      pos(t, out) {
        const a = (2 * Math.PI * t) / 90;
        out[0] = 15 * Math.cos(a);
        out[1] = 10 * Math.sin(a);
        out[2] = 1.5 + 0.5 * Math.sin(3 * a);
      },
      Rwc() {},
    };
    const rc = recCentres(loop, rng, 0.37, true);
    const { fixes } = simulateGps(loop, rng, { sigma: 2 });
    const res = alignToGps(rc.C, rc.times, fixes, { gravityAligned: false })!;
    expect(res.mode).toBe('sim3');
    expect(Math.abs(res.s * 0.37 - 1)).toBeLessThan(0.05);
    // R ≈ rc.Rᵀ
    let tr = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) tr += res.R[3 * i + j] * rc.R[3 * j + i]; // trace(R̂ · R_true)
    const ang = Math.acos(Math.min(1, (tr - 1) / 2));
    expect(ang).toBeLessThan((5 * Math.PI) / 180);
  });

  it('refuses short tracks and too few fixes; handles missing altitude', () => {
    const rng = new Rng(11);
    const traj = orbitTrajectory(rng); // 0.8 m wide — far below 4 × 3 m
    const rc = recCentres(traj, rng, 1, false);
    const { fixes } = simulateGps(traj, rng, { sigma: 3 });
    expect(alignToGps(rc.C, rc.times, fixes, { gravityAligned: true })).toBeNull();
    const walk = walkTrajectory(rng);
    const rw = recCentres(walk, rng, 1, false);
    const g = simulateGps(walk, rng, { sigma: 3, withAlt: false });
    expect(alignToGps(rw.C, rw.times, g.fixes.slice(0, 4), { gravityAligned: true })).toBeNull();
    const res = alignToGps(rw.C, rw.times, g.fixes, { gravityAligned: true })!;
    expect(res).not.toBeNull();
    expect(res.origin[2]).toBe(0);
    expect(res.hasAlt.every((v) => v === 0)).toBe(true);
    // fixes outside the keyframe time span are not associated
    const late: GeoFix[] = g.fixes.map((f) => ({ ...f, t: f.t + 1e6 }));
    expect(alignToGps(rw.C, rw.times, late, { gravityAligned: true })).toBeNull();
    const ext = trackExtent(res.dst, res.numFixes, 9);
    expect(ext).toBeGreaterThan(33);
    expect(ext).toBeLessThan(47);
    expect(res.extent).toBeCloseTo(ext, 6);
  });
});

describe('yawFromCompass', () => {
  it('recovers the yaw from noisy headings with outliers', () => {
    const rng = new Rng(21);
    for (const yaw of [0.3, -2.9, 1.7]) {
      const obs: CompassObservation[] = [];
      for (let i = 0; i < 80; i++) {
        // direction in the gravity-aligned reconstruction frame
        const az = rng.uniform(0, 2 * Math.PI); // CW from +Y
        const el = rng.uniform(-0.5, 0.5);
        const dir = [Math.cos(el) * Math.sin(az), Math.cos(el) * Math.cos(az), Math.sin(el)];
        // heading in ENU: azimuth after Rz(yaw) (CCW) = az − yaw
        let h = ((az - yaw) * 180) / Math.PI + 10 * rng.gaussian();
        if (i % 7 === 0) h += rng.uniform(60, 180); // magnetic disturbance
        obs.push({ headingDeg: h, accuracyDeg: 10, direction: dir });
      }
      obs.push({ headingDeg: 0, direction: [0, 0, 1] }); // vertical axis: skipped
      const r = yawFromCompass(obs)!;
      let d = r.yaw - yaw;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      if (LOG) console.log('compass yaw err deg', ((d * 180) / Math.PI).toFixed(2), 'σ', ((r.stdRad * 180) / Math.PI).toFixed(2));
      expect(Math.abs(d)).toBeLessThan((3 * Math.PI) / 180);
      expect(r.numUsed).toBe(80);
      expect(r.numInliers).toBeLessThan(80);
      expect(r.numInliers).toBeGreaterThan(60);
    }
    expect(yawFromCompass([{ headingDeg: 10, direction: [0, 0, -1] }])).toBeNull();
  });
});
