import { describe, expect, it } from 'vitest';
import { cameraCenter, cameraForward } from './conventions';
import { logSO3, m3TMul, orthonormalityError } from './mat';
import {
  createObjectTrajectory,
  createYardTrajectory,
  trajectoryPose,
  yardLoopPoint,
  type Trajectory,
} from './trajectory';

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function checkDerivatives(traj: Trajectory) {
  const h1 = 1e-5;
  const h2 = 1e-3;
  let worst = { v: 0, a: 0, dR: 0, ddR: 0, w: 0, wd: 0, orth: 0 };
  for (let k = 0; k <= 40; k++) {
    const t = -0.3 + (k / 40) * (traj.duration + 0.6);
    const s = traj.evaluate(t);
    const sp = traj.evaluate(t + h1),
      sm = traj.evaluate(t - h1);
    const Sp = traj.evaluate(t + h2),
      Sm = traj.evaluate(t - h2);
    const v = s.position.map((_, i) => (sp.position[i] - sm.position[i]) / (2 * h1));
    const a = s.position.map((_, i) => (Sp.position[i] - 2 * s.position[i] + Sm.position[i]) / (h2 * h2));
    const dR = s.R_wd.map((_, i) => (sp.R_wd[i] - sm.R_wd[i]) / (2 * h1));
    const ddR = s.R_wd.map((_, i) => (Sp.R_wd[i] - 2 * s.R_wd[i] + Sm.R_wd[i]) / (h2 * h2));
    // body rate from the relative rotation R(t−h)ᵀ R(t+h) = exp(2h [ω]×)
    const w = logSO3(m3TMul(sm.R_wd, sp.R_wd)).map((x) => x / (2 * h1));
    const wd = s.omegaDev.map((_, i) => (sp.omegaDev[i] - sm.omegaDev[i]) / (2 * h1));
    worst = {
      v: Math.max(worst.v, maxAbsDiff(v, s.velocity)),
      a: Math.max(worst.a, maxAbsDiff(a, s.acceleration)),
      dR: Math.max(worst.dR, maxAbsDiff(dR, s.dR_wd)),
      ddR: Math.max(worst.ddR, maxAbsDiff(ddR, s.ddR_wd)),
      w: Math.max(worst.w, maxAbsDiff(w, s.omegaDev)),
      wd: Math.max(worst.wd, maxAbsDiff(wd, s.omegaDotDev)),
      orth: Math.max(worst.orth, orthonormalityError(s.R_wd)),
    };
  }
  return worst;
}

describe('trajectory analytic derivatives', () => {
  for (const traj of [createObjectTrajectory(), createYardTrajectory()]) {
    it(`${traj.kind}: velocity, acceleration, dR, d²R, ω, ω̇ match finite differences`, () => {
      const w = checkDerivatives(traj);
      expect(w.v).toBeLessThan(1e-7);
      expect(w.a).toBeLessThan(5e-4);
      expect(w.dR).toBeLessThan(1e-7);
      expect(w.ddR).toBeLessThan(5e-4);
      expect(w.w).toBeLessThan(1e-7);
      expect(w.wd).toBeLessThan(1e-6);
      expect(w.orth).toBeLessThan(1e-12);
    });

    it(`${traj.kind}: C2 — acceleration and angular acceleration are continuous`, () => {
      let jumpA = 0,
        jumpW = 0;
      for (let k = 0; k < 2000; k++) {
        const t = (k / 2000) * traj.duration;
        const a = traj.evaluate(t),
          b = traj.evaluate(t + 1e-7);
        jumpA = Math.max(jumpA, maxAbsDiff(a.acceleration, b.acceleration));
        jumpW = Math.max(jumpW, maxAbsDiff(a.omegaDotDev, b.omegaDotDev));
      }
      expect(jumpA).toBeLessThan(1e-4);
      expect(jumpW).toBeLessThan(1e-4);
    });
  }
});

describe('object orbit', () => {
  const traj = createObjectTrajectory();
  it('orbits ~0.45 m around the object, 1.1–1.3 revolutions, looking at it', () => {
    const target = [0, 0, 0.86];
    let yawPrev = NaN,
      total = 0,
      maxAngle = 0;
    let rMin = Infinity,
      rMax = 0;
    const N = 500;
    for (let k = 0; k <= N; k++) {
      const t = (k / N) * traj.duration;
      const pose = trajectoryPose(traj, t);
      const C = cameraCenter(pose);
      const r = Math.hypot(C[0] - target[0], C[1] - target[1]);
      rMin = Math.min(rMin, r);
      rMax = Math.max(rMax, r);
      const yaw = Math.atan2(C[1], C[0]);
      if (k > 0) {
        let d = yaw - yawPrev;
        if (d > Math.PI) d -= 2 * Math.PI;
        if (d < -Math.PI) d += 2 * Math.PI;
        total += d;
      }
      yawPrev = yaw;
      const f = cameraForward(pose);
      const to = [target[0] - C[0], target[1] - C[1], target[2] - C[2]];
      const cos = (f[0] * to[0] + f[1] * to[1] + f[2] * to[2]) / Math.hypot(...to);
      maxAngle = Math.max(maxAngle, (Math.acos(Math.min(1, cos)) * 180) / Math.PI);
      expect(C[2]).toBeGreaterThan(0.86); // above the object centre, never below the table
    }
    const revs = total / (2 * Math.PI);
    expect(revs).toBeGreaterThan(1.1);
    expect(revs).toBeLessThan(1.3);
    expect(rMin).toBeGreaterThan(0.38);
    expect(rMax).toBeLessThan(0.52);
    expect(maxAngle).toBeLessThan(6);
  });

  it('is portrait (θ = 0): the device top edge points mostly up', () => {
    const s = traj.evaluate(3);
    expect(s.R_wd[7]).toBeGreaterThan(0.7); // device +y · world up
  });
});

describe('yard walk', () => {
  const traj = createYardTrajectory();
  it('walks ~1 m/s at ~1.5 m with a 2 Hz ±3 cm bob and closes the loop', () => {
    let len = 0,
      zMin = Infinity,
      zMax = -Infinity;
    let prev = traj.evaluate(0).position;
    const N = 4000;
    for (let k = 1; k <= N; k++) {
      const p = traj.evaluate((k / N) * traj.duration).position;
      len += Math.hypot(p[0] - prev[0], p[1] - prev[1]);
      zMin = Math.min(zMin, p[2]);
      zMax = Math.max(zMax, p[2]);
      prev = p;
    }
    const speed = len / traj.duration;
    expect(speed).toBeGreaterThan(0.85);
    expect(speed).toBeLessThan(1.25);
    expect(zMin).toBeGreaterThan(1.42);
    expect(zMax).toBeLessThan(1.58);
    expect(zMax - zMin).toBeGreaterThan(0.06);
    // loop closure: some late frame comes back within 1 m of the start
    const p0 = traj.evaluate(0).position;
    let best = Infinity;
    for (let t = 0.85 * traj.duration; t <= traj.duration; t += 0.05) {
      const p = traj.evaluate(t).position;
      best = Math.min(best, Math.hypot(p[0] - p0[0], p[1] - p0[1]));
    }
    expect(best).toBeLessThan(1);
    // extent ~15 x 10 m
    let xMax = 0,
      yMax = 0;
    for (let s = 0; s < 2 * Math.PI; s += 0.01) {
      const [x, y] = yardLoopPoint(s);
      xMax = Math.max(xMax, Math.abs(x));
      yMax = Math.max(yMax, Math.abs(y));
    }
    expect(2 * xMax).toBeGreaterThan(14);
    expect(2 * xMax).toBeLessThan(18);
    expect(2 * yMax).toBeGreaterThan(9);
    expect(2 * yMax).toBeLessThan(12);
  });

  it('looks mostly outward and roughly level, landscape (device +x up)', () => {
    let outward = 0;
    const N = 200;
    for (let k = 0; k < N; k++) {
      const t = (k / N) * traj.duration;
      const s = traj.evaluate(t);
      const pose = trajectoryPose(traj, t);
      const f = cameraForward(pose);
      const r = Math.hypot(s.position[0], s.position[1]);
      outward += (f[0] * s.position[0] + f[1] * s.position[1]) / (r * Math.hypot(f[0], f[1]));
      expect(Math.abs(f[2])).toBeLessThan(0.3);
      expect(s.R_wd[6]).toBeGreaterThan(0.9); // device +x · world up (landscape)
    }
    expect(outward / N).toBeGreaterThan(0.6);
  });
});
