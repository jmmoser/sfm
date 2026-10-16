/**
 * Adversarial review tests: check the generator's outputs the way a DOWNSTREAM CONSUMER
 * sees them (GT world-to-camera poses + meta R_cam_device + ImuSample[] + GeoFix[] +
 * heading samples), using only ARCHITECTURE §2 conventions — never the trajectory's
 * internal R_wd. A sign / transposition / frame error anywhere in the chain
 * (poseFromDevice, R_cam_device, IMU, ENU, heading) shows up as degrees or metres.
 */
import { describe, expect, it } from 'vitest';
import type { ImuSample, Pose } from '../../core/types';
import { applySim3, axisAngle, mat3Mul, mat3T } from '../../../e2e/lib/linalg.mjs';
import { computeMetrics, type LoadedDataset } from '../../../e2e/lib/metrics-core.mjs';
import { CliError, parseArgs, sceneConfig } from '../../../e2e/lib/synth-cli.mjs';
import { frameTime, resolveConfig, screenAngleFor, trajectoryFor, type SynthConfig } from './config';
import { cameraCenter, headingDegOfDirection, rCamDevice } from './conventions';
import { enuToGeodetic, geodeticToEnu, simulateGps } from './gps';
import { simulateHeadings } from './heading';
import { GRAVITY, simulateImu } from './imu';
import { expSO3, m3Mul, m3T, m3Vec, rotationAngleBetween } from './mat';
import { SynthRng } from './rng';
import { trajectoryPose } from './trajectory';

const NOISE_FREE = { gyroNoise: 0, accelNoise: 0, gyroBias: [0, 0, 0] as [number, number, number], accelBias: [0, 0, 0] as [number, number, number] };

/** Catmull-Rom interpolation of a uniformly sampled 3-vector signal at time tMs. */
function interp(samples: ImuSample[], tMs: number, pick: (s: ImuSample) => readonly number[]): [number, number, number] {
  const t0 = samples[0].t;
  const h = samples[1].t - t0;
  const x = (tMs - t0) / h;
  const i = Math.min(samples.length - 3, Math.max(1, Math.floor(x)));
  const u = x - i;
  const p0 = pick(samples[i - 1]),
    p1 = pick(samples[i]),
    p2 = pick(samples[i + 1]),
    p3 = pick(samples[i + 2]);
  const out: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 3; k++)
    out[k] = 0.5 * (2 * p1[k] + (-p0[k] + p2[k]) * u + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * u * u + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * u * u * u);
  return out;
}

/**
 * Consumer-side strapdown integration between two frames: R_wd from GT pose i via
 * R_wd = Rᵀ R_cam_device, then dR_wd/dt = R_wd [ω]× with the gyro, and position by
 * double integration of a_world = R_wd accelG + g_world (spec sign, Z-up world).
 */
function strapdown(imu: ImuSample[], R_cd: Float64Array, poseI: Pose, tI: number, tJ: number, v0: ArrayLike<number>) {
  let R_wd = m3Mul(m3T(poseI.R), R_cd);
  const p = Array.from(cameraCenter(poseI));
  const v = Array.from(v0);
  const h = 0.5; // ms
  const n = Math.round((tJ - tI) / h);
  const aw = (t: number, R: Float64Array) => {
    const ag = interp(imu, t, (s) => s.accelG!);
    const a = m3Vec(R, ag);
    a[2] -= GRAVITY; // + g_world = (0, 0, −9.81)
    return a;
  };
  let aPrev = aw(tI, R_wd);
  for (let k = 0; k < n; k++) {
    const t = tI + k * h;
    const w = interp(imu, t + h / 2, (s) => s.gyro); // midpoint exponential integrator
    const dt = h / 1000;
    const Rn = m3Mul(R_wd, expSO3([w[0] * dt, w[1] * dt, w[2] * dt]));
    const aNext = aw(t + h, Rn);
    for (let c = 0; c < 3; c++) {
      p[c] += v[c] * dt + ((2 * aPrev[c] + aNext[c]) / 6) * dt * dt;
      v[c] += 0.5 * (aPrev[c] + aNext[c]) * dt;
    }
    R_wd = Rn;
    aPrev = aNext;
  }
  return { R_wd, p };
}

describe('consumer view: IMU ↔ GT poses through R_cam_device (ARCHITECTURE §2)', () => {
  for (const scene of ['object', 'yard'] as const) {
    const cfg: SynthConfig = resolveConfig({ scene, imu: NOISE_FREE });
    const traj = trajectoryFor(cfg);
    const theta = screenAngleFor(cfg);
    const R_cd = rCamDevice(theta);
    const T0 = cfg.startTimeMs;
    const lastTau = frameTime(cfg, cfg.frames - 1);
    // exactly what generate.ts writes to imu.json (100 Hz, true clock)
    const imu = simulateImu(traj, -0.5, lastTau + 0.5, T0, cfg.imu);
    const pose = (i: number) => trajectoryPose(traj, frameTime(cfg, i));
    const tTrue = (i: number) => T0 + frameTime(cfg, i) * 1000;

    it(`${scene}: integrated gyro reproduces GT relative camera rotations R_j R_iᵀ = R_cd ΔR_dev R_cdᵀ`, () => {
      let worst = 0;
      for (const [i, j] of [
        [3, 4],
        [10, 17],
        [40, 52],
        [100, 101],
      ] as const) {
        const Pi = pose(i),
          Pj = pose(j);
        const { R_wd } = strapdown(imu, R_cd, Pi, tTrue(i), tTrue(j), [0, 0, 0]);
        // camera rotation predicted by the IMU: R_j = R_cd R_wd(t_j)ᵀ
        const Rpred = m3Mul(R_cd, m3T(R_wd));
        const err = rotationAngleBetween(Rpred, Pj.R);
        worst = Math.max(worst, err);
        // sanity: the cameras really did rotate (the check is not vacuous)
        if (j - i > 1) expect(rotationAngleBetween(Pi.R, Pj.R)).toBeGreaterThan(0.01);
      }
      expect(worst).toBeLessThan(2e-4); // rad; any convention error is O(0.1–1) rad
    });

    it(`${scene}: (accelG − accel)/g is the world up vector in the device frame; R_cd maps it to R·ẑ`, () => {
      for (const i of [0, 7, 33, 120]) {
        // the IMU sample nearest to frame i, and the GT camera pose at that sample's time
        const s = imu[Math.round(((tTrue(i) - imu[0].t) / 1000) * cfg.imu.rateHz)];
        expect(Math.abs(s.t - tTrue(i))).toBeLessThan(1000 / cfg.imu.rateHz);
        const Pn = trajectoryPose(traj, (s.t - T0) / 1000);
        const upDev = [0, 1, 2].map((k) => (s.accelG![k] - s.accel![k]) / GRAVITY);
        const upCam = m3Vec(R_cd, upDev);
        const expected = [Pn.R[2], Pn.R[5], Pn.R[8]]; // R (0,0,1)
        for (let k = 0; k < 3; k++) expect(upCam[k]).toBeCloseTo(expected[k], 10);
        // image is upright: world up projects to camera −y
        expect(upCam[1]).toBeLessThan(-0.6);
      }
    });

    it(`${scene}: double-integrated accelG (+ g_world) reproduces GT camera-centre displacement`, () => {
      let worst = 0;
      for (const [i, j] of [
        [5, 11],
        [60, 66],
      ] as const) {
        const Pi = pose(i),
          Pj = pose(j);
        const v0 = traj.evaluate(frameTime(cfg, i)).velocity; // initial velocity (state, not a sensor)
        const { p } = strapdown(imu, R_cd, Pi, tTrue(i), tTrue(j), v0);
        const Cj = cameraCenter(Pj);
        const Ci = cameraCenter(Pi);
        const disp = Math.hypot(Cj[0] - Ci[0], Cj[1] - Ci[1], Cj[2] - Ci[2]);
        expect(disp).toBeGreaterThan(0.05);
        worst = Math.max(worst, Math.hypot(p[0] - Cj[0], p[1] - Cj[1], p[2] - Cj[2]));
      }
      // a gravity sign error would be ~g t²/2 ≈ metres, a frame error decimetres
      expect(worst).toBeLessThan(3e-3);
    });
  }

  it('time-offset semantics: the IMU agrees with frame poses at tTrue, not at t = tTrue + offset', () => {
    const cfg = resolveConfig({ scene: 'object', imu: NOISE_FREE });
    const traj = trajectoryFor(cfg);
    const R_cd = rCamDevice(0);
    const imu = simulateImu(traj, -0.5, 10, cfg.startTimeMs, cfg.imu);
    const i = 20,
      j = 28;
    const Pi = trajectoryPose(traj, frameTime(cfg, i));
    const Pj = trajectoryPose(traj, frameTime(cfg, j));
    const ti = cfg.startTimeMs + frameTime(cfg, i) * 1000;
    const tj = cfg.startTimeMs + frameTime(cfg, j) * 1000;
    const errAt = (shift: number) => {
      // integrate from the (shifted) start time; the start attitude is the GT pose i
      const { R_wd } = strapdown(imu, R_cd, Pi, ti + shift, tj + shift, [0, 0, 0]);
      return rotationAngleBetween(m3Mul(R_cd, m3T(R_wd)), Pj.R);
    };
    expect(errAt(0)).toBeLessThan(1e-4);
    expect(errAt(cfg.timeOffsetMs)).toBeGreaterThan(20 * errAt(0));
  });
});

describe('IMU timing (iOS-like)', () => {
  it('defaults to 60 Hz; optional timestamp jitter keeps samples strictly increasing and values on the true grid', () => {
    const cfg = resolveConfig({ scene: 'yard', imu: { ...NOISE_FREE, timeJitterMs: 2 } });
    const traj = trajectoryFor(cfg);
    expect(resolveConfig({ scene: 'object' }).imu.rateHz).toBe(60);
    const clean = simulateImu(traj, 0, 20, 1000, { ...cfg.imu, timeJitterMs: 0 });
    const jit = simulateImu(traj, 0, 20, 1000, cfg.imu);
    expect(jit.length).toBe(1201);
    let s2 = 0;
    for (let k = 0; k < jit.length; k++) {
      if (k > 0) expect(jit[k].t).toBeGreaterThan(jit[k - 1].t);
      const d = jit[k].t - clean[k].t;
      expect(Math.abs(d)).toBeLessThanOrEqual(5 + 1e-9);
      s2 += d * d;
      expect(jit[k].gyro).toEqual(clean[k].gyro); // values belong to the true sample time
    }
    const sd = Math.sqrt(s2 / jit.length);
    expect(sd).toBeGreaterThan(1.7);
    expect(sd).toBeLessThan(2.2);
  });
});

describe('consumer view: heading and GPS conventions', () => {
  it('yard (θ = 90, landscape): headingDeviceTop ≈ camera heading − 90° (device top points left)', () => {
    const cfg = resolveConfig({ scene: 'yard' });
    const traj = trajectoryFor(cfg);
    const hs = simulateHeadings(traj, 0, 20, 0, { sigmaDeg: 0 });
    let worst = 0;
    for (const h of hs) {
      expect(h.heading).toBeCloseTo(h.trueHeading, 9);
      // camera heading from the GT pose as a consumer computes it (third row of R)
      const P = trajectoryPose(traj, h.t / 1000);
      const camH = headingDegOfDirection([P.R[6], P.R[7], P.R[8]]);
      expect(Math.abs(((camH - h.heading + 540) % 360) - 180)).toBeLessThan(1e-9);
      expect(h.headingDeviceTop).not.toBeNull();
      const d = (((h.headingDeviceTop! - (h.heading - 90)) % 360) + 540) % 360 - 180;
      worst = Math.max(worst, Math.abs(d));
    }
    expect(worst).toBeLessThan(6); // roll ±2.3°, pitch ≤ 8° only
  });

  it('heading 90° means the camera looks east (+x), 0° north (+y) in the ENU world', () => {
    expect(headingDegOfDirection([1, 0, 0])).toBeCloseTo(90, 12);
    expect(headingDegOfDirection([0, 1, -0.2])).toBeCloseTo(0, 12);
    expect(headingDegOfDirection([-1, -1, 0])).toBeCloseTo(225, 12);
  });

  it('ENU ↔ WGS84 uses the exact meridian / prime-vertical radii and the right axis order', () => {
    const O: [number, number, number] = [47.6062, -122.3321, 56];
    const a = 6378137,
      f = 1 / 298.257223563,
      e2 = f * (2 - f);
    const phi = (O[0] * Math.PI) / 180;
    const w = 1 - e2 * Math.sin(phi) ** 2;
    const M = (a * (1 - e2)) / w ** 1.5,
      N = a / Math.sqrt(w);
    // 1 m north / east / up
    const [latN, lonN, hN] = enuToGeodetic(O, 0, 1, 0);
    expect(((latN - O[0]) * Math.PI) / 180).toBeCloseTo(1 / (M + O[2]), 12);
    expect(lonN).toBeCloseTo(O[1], 12);
    expect(hN - O[2]).toBeCloseTo(0, 6);
    const [latE, lonE] = enuToGeodetic(O, 1, 0, 0);
    expect(((lonE - O[1]) * Math.PI) / 180).toBeCloseTo(1 / ((N + O[2]) * Math.cos(phi)), 12);
    expect(latE).toBeCloseTo(O[0], 9);
    const [latU, lonU, hU] = enuToGeodetic(O, 0, 0, 1);
    expect(hU - O[2]).toBeCloseTo(1, 8);
    expect(latU).toBeCloseTo(O[0], 11);
    expect(lonU).toBeCloseTo(O[1], 12);
    // geodeticToEnu is the inverse on the yard scale
    const back = geodeticToEnu(O, ...enuToGeodetic(O, 12.3, -45.6, 1.5));
    expect(Math.hypot(back[0] - 12.3, back[1] + 45.6, back[2] - 1.5)).toBeLessThan(1e-7);
  });

  it('noise-free GPS fixes map back to the GT camera centre at the fix time', () => {
    const cfg = resolveConfig({ scene: 'yard' });
    const traj = trajectoryFor(cfg);
    const fixes = simulateGps(traj, -0.5, 39.5, cfg.startTimeMs, { ...cfg.gps!, horizontalSigma: 0, verticalSigma: 0 });
    for (const fx of fixes) {
      const enu = geodeticToEnu(cfg.gps!.originLatLonAlt, fx.lat, fx.lon, fx.alt!);
      const C = cameraCenter(trajectoryPose(traj, (fx.t - cfg.startTimeMs) / 1000));
      expect(Math.hypot(enu[0] - C[0], enu[1] - C[1], enu[2] - C[2])).toBeLessThan(1e-6);
    }
  });
});

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/** Ring of cameras around GT samples on a unit sphere centred at (20, −30, 2). */
function ringDataset(): LoadedDataset {
  const O = [20, -30, 2];
  const frames = [];
  for (let i = 0; i < 24; i++) {
    const a = (i / 24) * 2 * Math.PI;
    const C = [O[0] + 3 * Math.cos(a), O[1] + 3 * Math.sin(a), O[2] + 0.5 + 0.2 * Math.sin(3 * a)];
    const fw = [O[0] - C[0], O[1] - C[1], O[2] - C[2]];
    const nf = Math.hypot(...fw);
    for (let k = 0; k < 3; k++) fw[k] /= nf;
    let r = [fw[1], -fw[0], 0];
    const nr = Math.hypot(...r);
    r = r.map((v) => v / nr);
    const d = [fw[1] * r[2] - fw[2] * r[1], fw[2] * r[0] - fw[0] * r[2], fw[0] * r[1] - fw[1] * r[0]];
    const R = [...r, ...d, ...fw];
    const t = [0, 1, 2].map((k) => -(R[3 * k] * C[0] + R[3 * k + 1] * C[1] + R[3 * k + 2] * C[2]));
    frames.push({ index: i, pose: { R, t } });
  }
  const rng = new SynthRng(77);
  const n = 20000;
  const pts = new Float32Array(3 * n),
    nrm = new Int8Array(3 * n);
  for (let i = 0; i < n; i++) {
    const v = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
    const l = Math.hypot(...v);
    for (let k = 0; k < 3; k++) {
      pts[3 * i + k] = O[k] + v[k] / l;
      nrm[3 * i + k] = Math.round((127 * v[k]) / l);
    }
  }
  return {
    meta: {
      name: 'ring',
      scene: 'yard',
      width: 640,
      height: 480,
      intrinsics: { f: 500, cx: 319.5, cy: 239.5, k1: 0, k2: 0 },
      sceneExtent: 6,
      roi: { min: [-100, -100, -100], max: [100, 100, 100] },
      denseTau: 0.02,
      gt: { spacing: Math.sqrt((4 * Math.PI) / n) },
    },
    frames,
    gt: { points: pts, normals: nrm, visibility: null },
  };
}

/** Express GT poses / points in a result frame X_res = s·R·X_gt + t. */
function similarResult(ds: LoadedDataset, S: { s: number; R: number[]; t: number[] }) {
  const images = ds.frames.map((f) => {
    const Rres = mat3Mul(f.pose.R, mat3T(S.R));
    const Rt = [0, 1, 2].map((k) => Rres[3 * k] * S.t[0] + Rres[3 * k + 1] * S.t[1] + Rres[3 * k + 2] * S.t[2]);
    return { index: f.index, pose: { R: Rres, t: [0, 1, 2].map((k) => S.s * f.pose.t[k] - Rt[k]) } };
  });
  const g = ds.gt!.points;
  const pts = new Float32Array(g.length);
  for (let i = 0; i < g.length / 3; i++) pts.set(applySim3(S, [g[3 * i], g[3 * i + 1], g[3 * i + 2]]), 3 * i);
  return { images, pts };
}

describe('metrics: adversarial cases', () => {
  const ds = ringDataset();

  it('scaleError is the exact relative scale error of the result (+ = too large)', () => {
    for (const s of [1.1, 0.8, 1.0]) {
      const { images } = similarResult(ds, { s, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] });
      const m = computeMetrics(ds, { images, metric: { metric: true, gravityAligned: true, northAligned: true } });
      expect(m.metric!.scaleError).toBeCloseTo(s - 1, 9);
      expect(m.metric!.scale).toBeCloseTo(1 / s, 9); // Sim3 scale mapping result → GT
    }
  });

  it("dense 'se3' alignment is a true rigid fit (not the Sim3's R, t with the scale dropped)", () => {
    // metric result with a 2 % scale error, far from its origin (ENU offsets are large outdoors)
    const S = { s: 1.02, R: axisAngle([0, 0, 1], 0.7), t: [150, -80, 3] };
    const { images, pts } = similarResult(ds, S);
    const m = computeMetrics(ds, { images, metric: { metric: true, gravityAligned: true, northAligned: false } }, { points: pts, denseAlign: 'se3', tau: 0.05 });
    // with the best rigid fit the only error is the 2 % scale about the scene centre:
    // points on the unit sphere (≤ ~3.5 m from the camera centroid) move ≤ ~7 cm
    expect(m.dense!.accuracyMedian).toBeLessThan(0.06);
    expect(m.dense!.numPointsEvaluated).toBe(20000);
  });

  it('unknown --dense-align modes are rejected instead of silently skipping the alignment', () => {
    const { images, pts } = similarResult(ds, { s: 2, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [5, 5, 5] });
    expect(() => computeMetrics(ds, { images }, { points: pts, denseAlign: 'sim3icp' as never })).toThrow(/dense-align/);
  });

  it('heading / gravity error signs: result world yawed +7° CCW ⇒ result north is 7° clockwise of GT north', () => {
    // X_res = Rz(+7°) X_gt: GT north (0,1,0) has result coordinates rotated CCW, i.e. the
    // result's +Y axis, seen in GT coordinates, is rotated CLOCKWISE by 7° (towards east).
    const S = { s: 1, R: axisAngle([0, 0, 1], (7 * Math.PI) / 180), t: [1, 2, 3] };
    const { images } = similarResult(ds, S);
    const m = computeMetrics(ds, { images, metric: { metric: true, gravityAligned: true, northAligned: true } });
    expect(m.frame!.headingErrorDeg).toBeCloseTo(7, 6);
    expect(m.frame!.gravityErrorDeg).toBeLessThan(1e-6);
    expect(m.frame!.headingErrorDegSim3!).toBeCloseTo(7, 6);
  });
});

describe('metrics: dense accuracy search bound', () => {
  it('a focus sample farther (Euclidean) than the base hit can still win after the point-to-plane refinement', () => {
    // base: one hidden-layer sample 1.5 mm below the query; focus: an on-surface sample
    // 2 mm to the side (refined distance 2 mm − h = 1 mm with h = 1 mm)
    const ds: LoadedDataset = {
      meta: { name: 'p', scene: 'object', width: 10, height: 10, intrinsics: { f: 10, cx: 4.5, cy: 4.5, k1: 0, k2: 0 }, sceneExtent: 1, denseTau: 0.01, gt: { spacing: 0.001, focus: { spacing: 0.001 } } },
      frames: [],
      gt: { points: Float32Array.of(0, 0, -0.0015), normals: Int8Array.of(0, 0, 127), visibility: null },
      gtFocus: { points: Float32Array.of(0.002, 0, 0), normals: Int8Array.of(0, 0, 127), visibility: null },
    };
    const m = computeMetrics(ds, { images: [] }, { points: Float32Array.of(0, 0, 0), denseAlign: 'none' });
    expect(m.dense!.accuracyMedian).toBeCloseTo(0.001, 6);
  });
});

describe('nearest-neighbour structures: adversarial inputs', () => {
  it('HashGrid / KdTree agree with brute force at ENU-scale offsets, negative cells and heavy duplicates', async () => {
    const { HashGrid } = await import('../../../e2e/lib/grid.mjs');
    const { KdTree } = await import('../../../e2e/lib/kdtree.mjs');
    const rng = new SynthRng(99);
    const n = 4000;
    const P = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      // a few hundred distinct points, each repeated (duplicates), around (−1234, 987, −3)
      const k = i % 317;
      const r = new SynthRng(1000 + k);
      P[3 * i] = -1234 + r.uniform(-0.5, 0.5);
      P[3 * i + 1] = 987 + r.uniform(-0.5, 0.5);
      P[3 * i + 2] = -3 + (k % 2 ? 0 : r.uniform(-0.05, 0.05)); // half of them coplanar
    }
    const grid = new HashGrid(P, 0.013);
    const tree = new KdTree(P);
    for (let q = 0; q < 300; q++) {
      const x = -1234 + rng.uniform(-0.7, 0.7),
        y = 987 + rng.uniform(-0.7, 0.7),
        z = -3 + rng.uniform(-0.2, 0.2);
      let best = Infinity;
      for (let i = 0; i < n; i++) best = Math.min(best, Math.hypot(P[3 * i] - x, P[3 * i + 1] - y, P[3 * i + 2] - z));
      const t = tree.nearest(x, y, z);
      expect(t.dist).toBeCloseTo(best, 9);
      const g = grid.nearest(x, y, z, 0.2);
      if (best <= 0.2) expect(g.dist).toBeCloseTo(best, 9);
      else expect(g.index).toBe(-1);
      expect(grid.hasWithin(x, y, z, 0.05)).toBe(best <= 0.05);
    }
    expect(new KdTree(new Float32Array(0)).nearest(0, 0, 0).index).toBe(-1);
  });
});

describe('generate-synth CLI safety', () => {
  it('rejects dataset names that would escape the output directory (the generator rm -rf’s the target)', () => {
    for (const bad of ['..', '.', '../x', 'a/b', '/tmp/x', 'a\\b']) {
      expect(() => sceneConfig('object', parseArgs(['object', '--name', bad]))).toThrow(CliError);
    }
    expect(sceneConfig('object', parseArgs(['object', '--name', 'my-set_1.v2'])).name).toBe('my-set_1.v2');
  });

  it('dataset R_cam_device in meta equals ARCHITECTURE §2 for both scenes', () => {
    expect(Array.from(rCamDevice(screenAngleFor(resolveConfig({ scene: 'object' }))))).toEqual([1, 0, 0, 0, -1, 0, 0, 0, -1]);
    expect(Array.from(rCamDevice(screenAngleFor(resolveConfig({ scene: 'yard' }))))).toEqual([0, -1, 0, -1, 0, 0, 0, 0, -1]);
    // and v_cam = R_cam_device · v_dev, R = R_cam_device · R_wdᵀ: the device top (+y) in the
    // yard (θ = 90) appears as image LEFT (camera −x)
    const v = m3Vec(rCamDevice(90), [0, 1, 0]);
    expect(Array.from(v)).toEqual([-1, 0, 0]);
  });
});
