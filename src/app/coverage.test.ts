import { describe, expect, it } from 'vitest';
import type { Pose } from '../core/types';
import { lookAt } from '../dense/synthetic';
import { OBJECT_REAL_SHAKE } from '../harness/synth/config';
import { createObjectTrajectory, trajectoryPose, type ObjectTrajectoryOptions } from '../harness/synth/trajectory';
import { axisAngleToMat3 } from '../math/rotation';
import { Rng } from '../math/random';
import type { LiveCamera } from '../workers/protocol';
import { COVERAGE_BAND_EDGES, computeCoverage, coverageBand, coverageCell, coverageSector, largestGap } from './coverage';

const DEG = Math.PI / 180;

/** World up (+Z) in the camera frame: R · (0, 0, 1). */
function upCamOf(p: Pose): [number, number, number] {
  return [p.R[2], p.R[5], p.R[8]];
}

/** Keyframes of a synthetic object orbit sampled at `kfPerS` (the live engine's ≈ 3.6 keyframes/s on object-real). */
function orbitKeyframes(opts: Partial<ObjectTrajectoryOptions>, kfPerS = 3.6): LiveCamera[] {
  const traj = createObjectTrajectory(opts);
  const out: LiveCamera[] = [];
  for (let k = 0; k <= traj.duration * kfPerS; k++) {
    const pose = trajectoryPose(traj, k / kfPerS);
    out.push({ pose, intrinsics: { width: 600, height: 800, f: 600, cx: 299.5, cy: 399.5, k1: 0, k2: 0 }, upCam: upCamOf(pose) });
  }
  return out;
}

/** object-real (e2e/generate-synth.mjs object-real: the default orbit + shake). */
const OBJECT_REAL: Partial<ObjectTrajectoryOptions> = { shake: OBJECT_REAL_SHAKE };
/** A capture that adds the higher loop the prompt asks for (spans ≈ 16–48° elevation). */
const TWO_TIER: Partial<ObjectTrajectoryOptions> = { ...OBJECT_REAL, duration: 50, revolutions: 2.4, heightAbove: 0.3, heightAmplitude: 0.15 };

/** The cameras in a world moved by the similarity X' = s Rs X + T (poses: R' = R Rsᵀ, t' = s t − R' T). */
function sim3(cams: LiveCamera[], s: number, Rs: Float64Array, T: number[]): LiveCamera[] {
  return cams.map((c) => {
    const R = c.pose.R;
    const Rp = new Float64Array(9);
    for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) Rp[3 * r + k] = R[3 * r] * Rs[3 * k] + R[3 * r + 1] * Rs[3 * k + 1] + R[3 * r + 2] * Rs[3 * k + 2];
    const tp = new Float64Array(3);
    for (let r = 0; r < 3; r++) tp[r] = s * c.pose.t[r] - (Rp[3 * r] * T[0] + Rp[3 * r + 1] * T[1] + Rp[3 * r + 2] * T[2]);
    return { ...c, pose: { R: Rp, t: tp } };
  });
}

describe('coverage bins', () => {
  it('bands at 10° / 40°, sectors centred on azimuth 0, circular gaps', () => {
    expect([coverageBand(-0.01), coverageBand(0), coverageBand(9.9 * DEG), coverageBand(10 * DEG), coverageBand(39.9 * DEG), coverageBand(40 * DEG), coverageBand(NaN)]).toEqual([-1, 0, 0, 1, 1, 2, -1]);
    expect([coverageSector(0), coverageSector(-14 * DEG), coverageSector(14 * DEG), coverageSector(16 * DEG), coverageSector(-16 * DEG), coverageSector(2 * Math.PI)]).toEqual([0, 0, 0, 1, 11, 0]);
    expect(largestGap([2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 0, 1], 2)).toBe(2);
    expect(largestGap([0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0], 2)).toBe(4); // wraps around
    expect(largestGap(new Array(12).fill(0), 2)).toBe(12);
  });
});

describe('computeCoverage', () => {
  it('object-real: the mid band is full, the top is not — "raise the phone, look down"', () => {
    const cams = orbitKeyframes(OBJECT_REAL);
    expect(cams.length).toBe(91);
    const cov = computeCoverage(cams);
    expect(cov.stable).toBe(true);
    // the convergence point is the object (target (0, 0, 0.86); the aim wanders by ±1.5 cm)
    expect(Math.hypot(cov.centre![0], cov.centre![1], cov.centre![2] - 0.86)).toBeLessThan(0.03);
    expect(cov.up![2]).toBeGreaterThan(Math.cos(1 * DEG));
    expect(cov.covered[1]).toBe(12);
    expect(cov.covered[2]).toBeLessThanOrEqual(1);
    expect(cov.prompt).toBe('look_down');
    expect(cov.completeness).toBeCloseTo((12 + cov.covered[2]) / 24, 9);
    // the orbit spans ≈ 13–35° elevation
    const el = Array.from(cov.elevationRad, (e) => e / DEG);
    expect(Math.min(...el)).toBeGreaterThan(10);
    expect(Math.max(...el)).toBeLessThan(38);
  });

  it('a second, higher loop fills the high band', () => {
    const cov = computeCoverage(orbitKeyframes(TWO_TIER));
    expect(cov.stable).toBe(true);
    expect(cov.covered[1]).toBe(12);
    expect(cov.covered[2]).toBeGreaterThanOrEqual(10);
    expect(cov.prompt).toBe('done');
    expect(cov.completeness).toBeGreaterThan(0.9);
  });

  it('azimuth: referenced to the first keyframe, counter-clockwise seen from above (the orbit direction)', () => {
    const cams = orbitKeyframes(OBJECT_REAL);
    const cov = computeCoverage(cams, { current: cams[0].pose });
    expect(cov.azimuthRad[0]).toBeCloseTo(0, 12);
    expect(cov.currentCell).toEqual({ sector: 0, band: 1 });
    // a quarter orbit later (25 s / 1.2 revolutions / 4 ≈ 5.2 s ≈ 19 keyframes): ≈ 90°
    expect(cov.azimuthRad[19] / DEG).toBeGreaterThan(70);
    expect(cov.azimuthRad[19] / DEG).toBeLessThan(110);
    expect(coverageCell(cov, cams[19].pose)!.sector).toBe(3);
  });

  it('a partial orbit asks for the other side; the azimuth-only ring ignores height', () => {
    const cams = orbitKeyframes(OBJECT_REAL).slice(0, 40); // ≈ 11 s ≈ 190°
    const cov = computeCoverage(cams);
    expect(cov.stable).toBe(true);
    expect(cov.prompt).toBe('other_side');
    expect(cov.covered[1]).toBeGreaterThanOrEqual(5);
    expect(cov.covered[1]).toBeLessThanOrEqual(8);
    const flat = computeCoverage(orbitKeyframes(OBJECT_REAL), { elevation: false });
    expect(flat.elevation).toBe(false);
    expect(flat.coveredSectors).toBe(12);
    expect(flat.completeness).toBe(1);
    expect(flat.prompt).toBe('done');
  });

  it('is invariant to a similarity transform of the live model', () => {
    const cams = orbitKeyframes(TWO_TIER);
    const ref = computeCoverage(cams);
    const rng = new Rng(7);
    for (let trial = 0; trial < 3; trial++) {
      const axis = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
      const Rs = axisAngleToMat3(Float64Array.from(axis.map((v) => v / Math.hypot(...axis))), rng.next() * Math.PI);
      const moved = computeCoverage(sim3(cams, 0.2 + 3 * rng.next(), Rs, [rng.gaussian() * 5, rng.gaussian() * 5, rng.gaussian() * 5]));
      expect(moved.stable).toBe(true);
      expect(moved.counts).toEqual(ref.counts);
      expect(moved.prompt).toBe(ref.prompt);
      for (let i = 0; i < cams.length; i++) {
        expect(Math.abs(moved.elevationRad[i] - ref.elevationRad[i])).toBeLessThan(1e-9);
        expect(Math.abs(Math.sin(moved.azimuthRad[i] - ref.azimuthRad[i]))).toBeLessThan(1e-9);
      }
    }
  });

  it('half the cameras without upCam (no gravity): elevation within 0.5°, same bins', () => {
    for (const opts of [OBJECT_REAL, TWO_TIER]) {
      const cams = orbitKeyframes(opts);
      const ref = computeCoverage(cams);
      const half = computeCoverage(cams.map((c, i) => (i % 2 ? { pose: c.pose, intrinsics: c.intrinsics } : c)));
      expect(half.stable).toBe(true);
      for (let i = 0; i < cams.length; i++) {
        const e = ref.elevationRad[i];
        expect(Math.abs(half.elevationRad[i] - e) / DEG).toBeLessThan(0.5);
        expect(Math.abs(Math.sin(half.azimuthRad[i] - ref.azimuthRad[i])) / DEG).toBeLessThan(0.5);
        // a different band only for a camera within 0.5° of a band edge
        if (coverageBand(half.elevationRad[i]) !== coverageBand(e)) expect(Math.min(...COVERAGE_BAND_EDGES.map((b) => Math.abs(e - b))) / DEG).toBeLessThan(0.5);
      }
      expect(half.covered).toEqual(ref.covered);
      expect(half.prompt).toBe(ref.prompt);
    }
  });

  it('is not stable before 6 converging cameras, nor for a walk', () => {
    const cams = orbitKeyframes(OBJECT_REAL);
    const few = computeCoverage(cams.slice(0, 5));
    expect(few.stable).toBe(false);
    expect(few.prompt).toBe('locating');
    expect(few.completeness).toBe(0);
    expect(Array.from(few.elevationRad).every(Number.isNaN)).toBe(true);
    const walk: LiveCamera[] = [];
    for (let i = 0; i < 20; i++) {
      const pose = lookAt([0.3 * i, 0, 1.5], [0.3 * i + 0.2, 8, 1.2]);
      walk.push({ pose, intrinsics: cams[0].intrinsics, upCam: upCamOf(pose) });
    }
    expect(computeCoverage(walk).stable).toBe(false);
  });

  it('280 keyframes take < 1 ms', () => {
    const cams = orbitKeyframes({ ...TWO_TIER, duration: 78 });
    expect(cams.length).toBeGreaterThanOrEqual(280);
    const use = cams.slice(0, 280);
    for (let i = 0; i < 30; i++) computeCoverage(use); // warm up the JIT
    const ms: number[] = [];
    for (let i = 0; i < 41; i++) {
      const t0 = performance.now();
      computeCoverage(use, { current: use[100].pose });
      ms.push(performance.now() - t0);
    }
    ms.sort((a, b) => a - b);
    // median (the 4 cores of the test machine are shared): ≈ 0.2–0.4 ms on a loaded x86 core
    expect(ms[20]).toBeLessThan(1);
  });
});
