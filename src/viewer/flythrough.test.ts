/**
 * Fly-through path (src/viewer/flythrough.ts) on the keyframe poses of the e2e object-real and
 * yard-real runs (flythrough.testdata.ts) and on synthetic captures: deterministic, every frame
 * centre within 5 cm (object) / 0.25 m (yard) of the capture polyline, the view direction within
 * 20° of the nearest keyframe, no jerks (per-frame linear and angular steps ≤ 2× their median),
 * a level horizon, 180–300 frames, and the video orientation / field of view from the photos.
 */
import { describe, expect, it } from 'vitest';
import type { Pose } from '../core/types';
import { Rng } from '../math/random';
import { lookAt } from '../splat/testutil';
import { buildFlythrough, easedProgress, flythroughCamera, flythroughTimestampUs, rotationDistance, viewingDepth, type FlythroughCamera, type FlythroughPath } from './flythrough';
import { OBJECT_REAL, YARD_REAL, type PoseFixture } from './flythrough.testdata';

const DEG = Math.PI / 180;

function fixtureCams(f: PoseFixture): FlythroughCamera[] {
  return f.poses.map((p) => ({ pose: { R: Float64Array.from(p.slice(0, 9)), t: Float64Array.from(p.slice(9, 12)) }, intrinsics: f.intrinsics }));
}

const centre = (p: Pose): number[] => {
  const { R, t } = p;
  return [-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])];
};

/** Distance from p to the polyline through `pts`. */
function polylineDistance(p: number[], pts: number[][]): number {
  let best = Infinity;
  for (let k = 0; k + 1 < pts.length; k++) {
    const a = pts[k];
    const b = pts[k + 1];
    const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const l2 = ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2;
    const w = l2 > 0 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1] + (p[2] - a[2]) * ab[2]) / l2)) : 0;
    best = Math.min(best, Math.hypot(p[0] - a[0] - w * ab[0], p[1] - a[1] - w * ab[1], p[2] - a[2] - w * ab[2]));
  }
  return best;
}

interface PathStats {
  maxOffPath: number;
  maxViewDeg: number;
  linStepRatio: number;
  angStepRatio: number;
  maxRollDeg: number;
}

function med(v: number[]): number {
  const s = v.slice().sort((a, b) => a - b);
  return s[s.length >> 1];
}

function stats(path: FlythroughPath, cams: FlythroughCamera[], up = [0, 0, 1]): PathStats {
  const kc = cams.map((c) => centre(c.pose));
  let maxOffPath = 0;
  let maxViewDeg = 0;
  let maxRollDeg = 0;
  const lin: number[] = [];
  const ang: number[] = [];
  path.poses.forEach((p, i) => {
    const c = centre(p);
    maxOffPath = Math.max(maxOffPath, polylineDistance(c, kc));
    let nk = 0;
    for (let k = 1; k < kc.length; k++) if (Math.hypot(c[0] - kc[k][0], c[1] - kc[k][1], c[2] - kc[k][2]) < Math.hypot(c[0] - kc[nk][0], c[1] - kc[nk][1], c[2] - kc[nk][2])) nk = k;
    const R = cams[nk].pose.R;
    const cosA = p.R[6] * R[6] + p.R[7] * R[7] + p.R[8] * R[8];
    maxViewDeg = Math.max(maxViewDeg, Math.acos(Math.min(1, cosA)) / DEG);
    // level horizon: the camera x axis is perpendicular to up
    maxRollDeg = Math.max(maxRollDeg, Math.asin(Math.min(1, Math.abs(p.R[0] * up[0] + p.R[1] * up[1] + p.R[2] * up[2]))) / DEG);
    if (i > 0) {
      const q = path.poses[i - 1];
      const c0 = centre(q);
      lin.push(Math.hypot(c[0] - c0[0], c[1] - c0[1], c[2] - c0[2]));
      ang.push(rotationDistance(p.R, q.R));
    }
  });
  return { maxOffPath, maxViewDeg, linStepRatio: Math.max(...lin) / med(lin), angStepRatio: Math.max(...ang) / med(ang), maxRollDeg };
}

function log(name: string, p: FlythroughPath, s: PathStats): void {
  console.log(
    `[flythrough] ${name}: ${p.mode}, ${p.poses.length} frames ${p.width}×${p.height} f ${p.fx.toFixed(0)}, coverage ${p.coverage.map((c) => c.toFixed(2)).join('–')}, ` +
      `off-path ≤ ${(100 * s.maxOffPath).toFixed(1)} cm, view ≤ ${s.maxViewDeg.toFixed(1)}°, step ratio lin ${s.linStepRatio.toFixed(2)} ang ${s.angStepRatio.toFixed(2)}, roll ${s.maxRollDeg.toFixed(3)}°`,
  );
}

/** Synthetic orbit around (0, 0, 0.1): radius 0.45 m, height 0.3 m, hand jitter. */
function orbit(n: number, sweepDeg: number, seed: number, jitter = 0.004): FlythroughCamera[] {
  const rng = new Rng(seed);
  const intr = { width: 1440, height: 1920, f: 1500 };
  return Array.from({ length: n }, (_, k) => {
    const a = (k / (n - 1)) * sweepDeg * DEG + 0.3;
    const eye: [number, number, number] = [0.45 * Math.cos(a) + jitter * rng.gaussian(), 0.45 * Math.sin(a) + jitter * rng.gaussian(), 0.4 + 0.05 * Math.sin(2 * a) + jitter * rng.gaussian()];
    const target: [number, number, number] = [0.02 * rng.gaussian(), 0.02 * rng.gaussian(), 0.1];
    const { R, t } = lookAt(eye, target);
    return { pose: { R: Float64Array.from(R), t: Float64Array.from(t) }, intrinsics: intr };
  });
}

describe('fly-through path', () => {
  it('object-real (e2e run poses): an orbit about the convergence point that hugs the capture', () => {
    const cams = fixtureCams(OBJECT_REAL);
    const p = buildFlythrough(cams)!;
    const s = stats(p, cams);
    log('object-real', p, s);
    expect(p.mode).toBe('object');
    expect(p.poses.length).toBeGreaterThanOrEqual(180);
    expect(p.poses.length).toBeLessThanOrEqual(300);
    // portrait photos (720 × 960, image y vertical) → portrait video
    expect([p.width, p.height]).toEqual([720, 1280]);
    expect(s.maxOffPath).toBeLessThan(0.05);
    expect(s.maxViewDeg).toBeLessThan(20);
    expect(s.linStepRatio).toBeLessThanOrEqual(2);
    expect(s.angStepRatio).toBeLessThanOrEqual(2);
    expect(s.maxRollDeg).toBeLessThan(0.01);
  });

  it('yard-real (e2e run poses): a smoothed replay of the walk', () => {
    const cams = fixtureCams(YARD_REAL);
    // depth: viewingDepth of the run's sparse.ply (8.47 m)
    const p = buildFlythrough(cams, { depth: 8.5 })!;
    const s = stats(p, cams);
    log('yard-real', p, s);
    expect(p.mode).toBe('scene');
    expect(p.poses.length).toBeGreaterThanOrEqual(180);
    expect(p.poses.length).toBeLessThanOrEqual(300);
    // landscape photos (960 × 720) → landscape video
    expect([p.width, p.height]).toEqual([1280, 720]);
    expect(s.maxOffPath).toBeLessThan(0.25);
    expect(s.maxViewDeg).toBeLessThan(20);
    expect(s.linStepRatio).toBeLessThanOrEqual(2);
    expect(s.angStepRatio).toBeLessThanOrEqual(2);
    expect(s.maxRollDeg).toBeLessThan(0.01);
  });

  it('is deterministic', () => {
    for (const f of [OBJECT_REAL, YARD_REAL]) {
      const a = buildFlythrough(fixtureCams(f))!;
      const b = buildFlythrough(fixtureCams(f))!;
      expect(b.poses.length).toBe(a.poses.length);
      a.poses.forEach((p, i) => {
        expect(Array.from(b.poses[i].R)).toEqual(Array.from(p.R));
        expect(Array.from(b.poses[i].t)).toEqual(Array.from(p.t));
      });
    }
  });

  it('synthetic orbits: a full noisy orbit, a partial one and the frame camera', () => {
    for (const [sweep, seed] of [
      [360, 1],
      [140, 2],
    ] as const) {
      const cams = orbit(60, sweep, seed);
      const p = buildFlythrough(cams)!;
      const s = stats(p, cams);
      log(`orbit ${sweep}°`, p, s);
      expect(p.mode).toBe('object');
      expect(Math.hypot(p.centre![0], p.centre![1], p.centre![2] - 0.1)).toBeLessThan(0.03);
      expect(s.maxOffPath).toBeLessThan(0.05);
      expect(s.maxViewDeg).toBeLessThan(20);
      expect(s.linStepRatio).toBeLessThanOrEqual(2);
      expect(s.angStepRatio).toBeLessThanOrEqual(2);
      // at the 40 °/s orbit limit: the full orbit in ≈ 9 s + 0.6 s of ramps, the partial one over minSeconds
      if (sweep === 360) expect(Math.abs(p.poses.length - 30 * 9.6)).toBeLessThan(6);
      else expect(p.poses.length).toBe(180);
      expect(p.coverage[1] - p.coverage[0]).toBeGreaterThan(0.999);
    }
    const p = buildFlythrough(orbit(30, 90, 3), { longSide: 640, shortSide: 360 })!;
    const cam = flythroughCamera(p, 10);
    expect([cam.width, cam.height, cam.cx, cam.cy]).toEqual([360, 640, 179.5, 319.5]);
    expect(cam.R).toBe(p.poses[10].R);
    expect(cam.near).toBeCloseTo(0.02 * p.depth, 12);
    // the frame fits the photos' field of view (portrait 1440 × 1920 at f 1500), × zoom 1.05
    expect(cam.fx).toBeCloseTo(1.05 * Math.max(320 / (960 / 1500), 180 / (720 / 1500)), 6);
    expect(flythroughTimestampUs(p, 3)).toBe(100000);
  });

  it('synthetic walk: a curved path at 1.6 m height looking ahead and to the side', () => {
    const rng = new Rng(9);
    const cams: FlythroughCamera[] = [];
    for (let k = 0; k < 70; k++) {
      const s = k * 0.5;
      const eye: [number, number, number] = [s + 0.03 * rng.gaussian(), 3 * Math.sin(s / 8) + 0.03 * rng.gaussian(), 1.6 + 0.02 * rng.gaussian()];
      const look: [number, number, number] = [eye[0] + 6, eye[1] + 8 * Math.cos(s / 10), 1.2];
      const { R, t } = lookAt(eye, look);
      cams.push({ pose: { R: Float64Array.from(R), t: Float64Array.from(t) }, intrinsics: { width: 1920, height: 1440, f: 1500 } });
    }
    const p = buildFlythrough(cams, { depth: 8 })!;
    const st = stats(p, cams);
    log('walk', p, st);
    expect(p.mode).toBe('scene');
    expect(st.maxOffPath).toBeLessThan(0.25);
    expect(st.maxViewDeg).toBeLessThan(20);
    expect(st.linStepRatio).toBeLessThanOrEqual(2);
    expect(st.angStepRatio).toBeLessThanOrEqual(2);
    // 35 m at 0.2 · 8 m/s: cropped to the middle 10 s
    expect(p.poses.length).toBe(300);
    expect(p.coverage[0]).toBeGreaterThan(0.1);
    expect(p.coverage[1]).toBeLessThan(0.9);
    expect(Math.abs(p.coverage[0] + p.coverage[1] - 1)).toBeLessThan(0.1);
  });

  it('a pan from one spot turns in place; fewer than two cameras give no path', () => {
    const cams = Array.from({ length: 12 }, (_, k) => {
      const a = (k * 8 * Math.PI) / 180;
      const { R, t } = lookAt([1, 2, 1.5], [1 + Math.cos(a), 2 + Math.sin(a), 1.5]);
      return { pose: { R: Float64Array.from(R), t: Float64Array.from(t) }, intrinsics: { width: 1440, height: 1920, f: 1500 } };
    });
    const p = buildFlythrough(cams)!;
    expect(p.mode).toBe('scene');
    for (const q of p.poses) expect(Math.hypot(...centre(q).map((v, i) => v - [1, 2, 1.5][i]))).toBeLessThan(1e-9);
    const first = Math.atan2(p.poses[0].R[7], p.poses[0].R[6]) / DEG;
    const last = Math.atan2(p.poses[p.poses.length - 1].R[7], p.poses[p.poses.length - 1].R[6]) / DEG;
    expect(last - first).toBeGreaterThan(70);
    expect(buildFlythrough(cams.slice(0, 1))).toBeNull();
  });

  it('viewingDepth: the median depth of the points the photos see', () => {
    const cams = orbit(24, 360, 4, 0);
    // points on a 0.2 m sphere around the orbit centre: seen at ≈ the camera distance (0.54 m) − 0.2 m … + 0.2 m
    const rng = new Rng(8);
    const pos = new Float32Array(3 * 3000);
    for (let i = 0; i < 3000; i++) {
      const v = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
      const l = Math.hypot(v[0], v[1], v[2]);
      pos.set([(0.2 * v[0]) / l, (0.2 * v[1]) / l, 0.1 + (0.2 * v[2]) / l], 3 * i);
    }
    const d = viewingDepth(cams, pos)!;
    const dist = Math.hypot(0.45, 0.3);
    expect(d).toBeGreaterThan(dist - 0.2);
    expect(d).toBeLessThan(dist + 0.05);
    expect(viewingDepth(cams, new Float32Array(30))).toBeNull();
  });

  it('easedProgress ramps smoothly from 0 to 1', () => {
    expect(easedProgress(0, 0.1)).toBe(0);
    expect(easedProgress(1, 0.1)).toBe(1);
    expect(easedProgress(0.5, 0.1)).toBeCloseTo(0.5, 12);
    let prev = 0;
    let maxStep = 0;
    for (let i = 1; i <= 100; i++) {
      const v = easedProgress(i / 100, 0.1);
      expect(v).toBeGreaterThanOrEqual(prev);
      maxStep = Math.max(maxStep, v - prev);
      prev = v;
    }
    // cruise speed 1 / (1 − r)
    expect(maxStep).toBeLessThanOrEqual(0.01 / 0.9 + 1e-12);
  });
});
