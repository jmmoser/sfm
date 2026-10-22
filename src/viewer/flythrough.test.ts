/**
 * Fly-through path (src/viewer/flythrough.ts) on the keyframe poses of the e2e object-real and
 * yard-real runs (flythrough.testdata.ts) and on synthetic captures: deterministic, every frame
 * centre within 5 cm (object) / 0.25 m (yard) of the capture polyline, the view direction within
 * 20° of the nearest keyframe, no jerks (per-frame linear and angular steps ≤ 2× their median),
 * a level horizon, 180–300 frames, and the video orientation / field of view from the photos.
 * Captures the README asks for or that users make anyway: walks that turn on the spot (out and
 * back, a pan in the middle: the clip shows walking, pans take ≤ a quarter of it), orbits at two
 * heights in either direction, a top-down shot and an azimuth gap (one ring is orbited, no
 * azimuth without photos is shown).
 */
import { describe, expect, it } from 'vitest';
import type { Pose } from '../core/types';
import { Rng } from '../math/random';
import { lookAt } from '../splat/testutil';
import { buildFlythrough, easedProgress, flythroughCamera, flythroughTimestampUs, rotationDistance, viewingDepth, type FlythroughCamera, type FlythroughOptions, type FlythroughPath } from './flythrough';
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

/**
 * Largest angle between a frame's view and the views of the keyframes near it: those within `tol`
 * of the nearest one (an out-and-back walk passes each spot twice, looking both ways).
 */
function nearViewDeg(path: FlythroughPath, cams: FlythroughCamera[], tol: number): number {
  const kc = cams.map((c) => centre(c.pose));
  let worst = 0;
  for (const p of path.poses) {
    const c = centre(p);
    const d = kc.map((k) => Math.hypot(c[0] - k[0], c[1] - k[1], c[2] - k[2]));
    const dMin = Math.min(...d);
    let best = 180;
    cams.forEach((cam, k) => {
      if (d[k] > dMin + tol) return;
      const R = cam.pose.R;
      best = Math.min(best, Math.acos(Math.min(1, p.R[6] * R[6] + p.R[7] * R[7] + p.R[8] * R[8])) / DEG);
    });
    worst = Math.max(worst, best);
  }
  return worst;
}

/** Translation shown (sum of frame steps) and the share of frames that turn on the spot. */
function motion(path: FlythroughPath): { moved: number; turningShare: number } {
  const lin: number[] = [];
  const ang: number[] = [];
  for (let i = 1; i < path.poses.length; i++) {
    const a = centre(path.poses[i - 1]);
    const b = centre(path.poses[i]);
    lin.push(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
    ang.push(rotationDistance(path.poses[i].R, path.poses[i - 1].R));
  }
  const cruise = Math.max(...lin);
  const turning = lin.filter((l, i) => l < 0.2 * cruise && ang[i] > 0.1 * DEG).length;
  return { moved: lin.reduce((x, y) => x + y, 0), turningShare: turning / lin.length };
}

const LANDSCAPE = { width: 1920, height: 1440, f: 1500 };

/** A camera 1.6 m high looking along yaw `yawDeg` and slightly down. */
function walkCam(eye: [number, number, number], yawDeg: number): FlythroughCamera {
  const a = yawDeg * DEG;
  const { R, t } = lookAt(eye, [eye[0] + 6 * Math.cos(a), eye[1] + 6 * Math.sin(a), eye[2] - 0.6]);
  return { pose: { R: Float64Array.from(R), t: Float64Array.from(t) }, intrinsics: LANDSCAPE };
}

/** Out along +x looking ahead (0.5 m keyframes), a 180° turn on the spot (11 keyframes, `turnJitter` m), back the same way. */
function outAndBack(legM: number, seed: number, turnJitter: number): FlythroughCamera[] {
  const rng = new Rng(seed);
  const j = (s: number): number => s * rng.gaussian();
  const cams: FlythroughCamera[] = [];
  const kf = Math.round(legM / 0.5);
  for (let k = 0; k <= kf; k++) cams.push(walkCam([0.5 * k + j(0.01), j(0.01), 1.6 + j(0.01)], j(1)));
  for (let k = 1; k < 12; k++) cams.push(walkCam([legM + j(turnJitter), j(turnJitter), 1.6 + j(turnJitter)], 15 * k + j(1)));
  for (let k = 0; k <= kf; k++) cams.push(walkCam([legM - 0.5 * k + j(0.01), j(0.01), 1.6 + j(0.01)], 180 + j(1)));
  return cams;
}

/**
 * Object capture around (0, 0, 0.1): keyframes at azimuths `az` (deg) and heights `h` (m), radius
 * `r` (m), aimed at the object with 4 mm / 1 cm of jitter.
 */
function objectShots(az: number[], h: number[], seed: number, r: number[] = az.map(() => 0.45)): FlythroughCamera[] {
  const rng = new Rng(seed);
  const intr = { width: 1440, height: 1920, f: 1500 };
  return az.map((a, k) => {
    const eye: [number, number, number] = [r[k] * Math.cos(a * DEG) + 0.004 * rng.gaussian(), r[k] * Math.sin(a * DEG) + 0.004 * rng.gaussian(), h[k] + 0.004 * rng.gaussian()];
    const { R, t } = lookAt(eye, [0.01 * rng.gaussian(), 0.01 * rng.gaussian(), 0.1]);
    return { pose: { R: Float64Array.from(R), t: Float64Array.from(t) }, intrinsics: intr };
  });
}

const range = (a: number, b: number, step: number): number[] => Array.from({ length: Math.round((b - a) / step) + 1 }, (_, i) => a + i * step);

/** Largest azimuth (about the z axis through the origin) of a frame away from every keyframe's (deg). */
function azimuthOffDeg(path: FlythroughPath, cams: FlythroughCamera[]): number {
  const az = (c: number[]): number => Math.atan2(c[1], c[0]);
  const ka = cams.map((c) => az(centre(c.pose)));
  let worst = 0;
  for (const p of path.poses) {
    const a = az(centre(p));
    worst = Math.max(worst, Math.min(...ka.map((k) => Math.abs(Math.atan2(Math.sin(a - k), Math.cos(a - k))))) / DEG);
  }
  return worst;
}

function expectHugs(name: string, cams: FlythroughCamera[], o: FlythroughOptions, offPath: number): FlythroughPath {
  const p = buildFlythrough(cams, o)!;
  const s = stats(p, cams);
  log(name, p, s);
  expect(s.maxOffPath).toBeLessThan(offPath);
  expect(s.linStepRatio).toBeLessThanOrEqual(2);
  expect(s.maxRollDeg).toBeLessThan(0.01);
  return p;
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

  it('an out-and-back walk shows the walking, not 10 s of turning on the spot', () => {
    // 20 m out, a 180° turn on the spot with 1 cm of hand jitter, 20 m back (depth 8 m: 1.6 m/s)
    for (const jitter of [0.01, 0]) {
      const cams = outAndBack(20, 5, jitter);
      const p = expectHugs(`out and back 20 m (turn jitter ${jitter} m)`, cams, { depth: 8 }, 0.25);
      const m = motion(p);
      expect(p.mode).toBe('scene');
      expect(p.poses.length).toBe(300);
      // ≥ 75 % of the clip at the 1.6 m/s limit; the turn (if shown at all) ≤ a quarter
      expect(m.moved).toBeGreaterThan(0.75 * 1.6 * 9.4);
      expect(m.turningShare).toBeLessThanOrEqual(0.25);
      expect(p.coverage[1] - p.coverage[0]).toBeGreaterThan(0.25);
      expect(nearViewDeg(p, cams, 0.25)).toBeLessThan(20);
    }
    // 4 m legs: the whole capture fits once the turn is shortened, so the turn is shown, eased
    for (const jitter of [0.01, 0]) {
      const cams = outAndBack(4, 6, jitter);
      const p = expectHugs(`out and back 4 m (turn jitter ${jitter} m)`, cams, { depth: 8 }, 0.25);
      const m = motion(p);
      expect(p.coverage[0]).toBeLessThan(0.01);
      expect(p.coverage[1]).toBeGreaterThan(0.99);
      // 8 m at 1.6 m/s (5 s) + 180° at ≤ 90 °/s peak with smoothstep ramps (2.9 s)
      expect(m.turningShare).toBeLessThan(0.4);
      expect(m.moved).toBeGreaterThan(7.5);
      const yaw = (q: Pose): number => Math.atan2(q.R[7], q.R[6]) / DEG;
      expect(Math.abs(Math.abs(yaw(p.poses[p.poses.length - 1]) - yaw(p.poses[0])) - 180)).toBeLessThan(10);
      // every frame looks the way a photo near it looked (no frozen frame at the turn)
      expect(nearViewDeg(p, cams, 0.25)).toBeLessThan(20);
      for (let i = 1; i < p.poses.length; i++) expect(rotationDistance(p.poses[i].R, p.poses[i - 1].R) / DEG).toBeLessThan(3.1);
    }
  });

  it('a walk with a 90° pan in the middle: the pan takes at most a quarter of the clip', () => {
    // 15 m along +x looking ahead, a 90° pan on the spot, 15 m on looking to the left
    const rng = new Rng(7);
    const j = (v: number): number => v * rng.gaussian();
    const cams: FlythroughCamera[] = [];
    for (let k = 0; k <= 30; k++) cams.push(walkCam([0.5 * k + j(0.01), j(0.01), 1.6 + j(0.01)], j(1)));
    for (let k = 1; k < 6; k++) cams.push(walkCam([15 + j(0.01), j(0.01), 1.6 + j(0.01)], 15 * k + j(1)));
    for (let k = 0; k <= 30; k++) cams.push(walkCam([15 + 0.5 * k + j(0.01), j(0.01), 1.6 + j(0.01)], 90 + j(1)));
    const p = expectHugs('walk with a pan', cams, { depth: 8, mode: 'scene' }, 0.25);
    const m = motion(p);
    expect(p.poses.length).toBe(300);
    expect(m.moved).toBeGreaterThan(0.75 * 1.6 * 9.4);
    expect(m.turningShare).toBeLessThanOrEqual(0.25);
    expect(p.coverage[1] - p.coverage[0]).toBeGreaterThan(0.25);
    expect(nearViewDeg(p, cams, 0.25)).toBeLessThan(20);
    // its optical axes converge (inFront 67 / 67, 6.9 m away), so 'auto' calls it an object, but
    // no pass covers 20° of azimuth about that point: the walk again (the app passes the user's /
    // the engine's mode anyway, flythroughForModel)
    const auto = expectHugs('walk with a pan (auto)', cams, { depth: 8 }, 0.25);
    expect(auto.mode).toBe('scene');
    expect(nearViewDeg(auto, cams, 0.25)).toBeLessThan(20);
  });

  it('orbits at two heights (either direction), a top-down shot and an azimuth gap: one ring, covered azimuths only', () => {
    // low ring 0 → 360°, a rise at ≈ 360° (the README's "cover it from 2–3 heights"), high ring on
    const rise = { az: [362, 364, 366], h: [0.33, 0.42, 0.5] };
    const same = objectShots([...range(0, 360, 10), ...rise.az, ...range(370, 720, 10)], [...range(0, 360, 10).map(() => 0.25), ...rise.h, ...range(370, 720, 10).map(() => 0.55)], 11);
    const p1 = expectHugs('two heights, same direction', same, {}, 0.05);
    expect(p1.mode).toBe('object');
    expect(stats(p1, same).maxViewDeg).toBeLessThan(20);
    // … and back: low ring 0 → 360°, high ring 360 → 0°
    const back = objectShots([...range(0, 360, 10), ...rise.az.map((a) => 720 - a), ...range(350, 0, -10)], [...range(0, 360, 10).map(() => 0.25), ...rise.h, ...range(350, 0, -10).map(() => 0.55)], 12);
    const p2 = expectHugs('two heights, back and forth', back, {}, 0.05);
    expect(p2.mode).toBe('object');
    expect(stats(p2, back).maxViewDeg).toBeLessThan(20);
    // one ring (with the rise it starts from), not the average of both
    expect(p2.poses.filter((q) => Math.min(Math.abs(centre(q)[2] - 0.25), Math.abs(centre(q)[2] - 0.55)) > 0.03).length).toBeLessThan(0.15 * p2.poses.length);
    // a 180° arc ending in shots over the top (the last 1 cm from the axis)
    const top = objectShots([...range(0, 180, 8), 182, 184, 186], [...range(0, 180, 8).map(() => 0.3), 0.45, 0.58, 0.65], 13, [...range(0, 180, 8).map(() => 0.45), 0.3, 0.15, 0.01]);
    const p3 = expectHugs('arc + top-down', top, {}, 0.05);
    expect(p3.mode).toBe('object');
    expect(azimuthOffDeg(p3, top)).toBeLessThan(8);
    // registered azimuths 0–90° and 180–300° (tracking lost in between): the larger arc only
    const gap = objectShots([...range(0, 90, 6), ...range(180, 300, 6)], [...range(0, 90, 6), ...range(180, 300, 6)].map(() => 0.3), 14);
    const p4 = expectHugs('azimuth gap', gap, {}, 0.05);
    expect(p4.mode).toBe('object');
    expect(azimuthOffDeg(p4, gap)).toBeLessThan(5);
    expect(p4.coverage[0]).toBeGreaterThanOrEqual(16 / 36 - 1e-9);
    for (const [name, p, cams] of [
      ['two heights, same direction', p1, same],
      ['back and forth', p2, back],
      ['top-down', p3, top],
      ['gap', p4, gap],
    ] as const)
      expect(stats(p, [...cams]).angStepRatio, name).toBeLessThanOrEqual(2);
  });

  it("'auto' is the engine's object / scene test: converging cameras at a plausible scene depth", () => {
    const cams = orbit(40, 300, 21);
    expect(buildFlythrough(cams)!.mode).toBe('object');
    expect(buildFlythrough(cams, { depth: 0.5 })!.mode).toBe('object');
    // the convergence point 0.54 m away while the photos see 5 m deep: not an object
    expect(buildFlythrough(cams, { depth: 5 })!.mode).toBe('scene');
    expect(buildFlythrough(cams, { mode: 'scene' })!.mode).toBe('scene');
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
