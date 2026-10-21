/**
 * Adversarial regression tests for the sfm "extras" (keyframes, pairs, vocab, inertial, geo,
 * world). Each test names the defect it guards against.
 */
import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, GeoFix, ImuSample } from '../core/types';
import { mat3MulVec, mat3Transpose } from '../math/mat3';
import { Rng } from '../math/random';
import { rotvecToMat3 } from '../math/rotation';
import { cameraFromDevice, lookAt, orbitTrajectory, simulateGps, simulateImu, simulateKeyframes, walkTrajectory, type SimTrajectory } from './extras-testutil';
import { alignToGps, EnuFrame, yawFromCompass } from './geo';
import { estimateInertialAlignment, estimateMetricScale, estimateTimeOffset, estimateTimeOffsetFromAcceleration, GyroIntegrator } from './inertial';
import { KeyframeSelector, rotationCompensatedParallax, type KeyframeCandidate } from './keyframes';
import { mergePairs, spatialPairs } from './pairs';
import { quantizeDescriptors, trainVocabulary, VocabularyIndex } from './vocab';
import { computeWorldAlignment, type WorldKeyframe } from './world';
import { perfSlack } from '../core/perf-slack';

const DEG = Math.PI / 180;

function cand(t: number, inliers: number, extra: Partial<KeyframeCandidate> = {}): KeyframeCandidate {
  return {
    t,
    sharpness: 100,
    angularSpeed: 0,
    numFeatures: 1000,
    numMatchesToLastKF: inliers,
    numInliersToLastKF: inliers,
    medianParallaxPx: 0,
    rotationSinceLastKFRad: 0,
    imageWidth: 1000,
    ...extra,
  };
}

/** Walk / orbit whose camera never rotates: the rotation-based time offset is unobservable. */
function noRotation(base: SimTrajectory): SimTrajectory {
  const L = new Float64Array(9);
  lookAt(1, 0.3, -0.1, L);
  return { duration: base.duration, pos: base.pos, Rwc: (_t, out) => out.set(L) };
}

describe('KeyframeSelector (review)', () => {
  it('sharpnessWarmup > sharpnessWindow no longer blocks the first keyframe forever', () => {
    const s = new KeyframeSelector({ sharpnessWindow: 3, sharpnessWarmup: 5 });
    const ds = [];
    for (let i = 0; i < 10; i++) ds.push(s.evaluate(cand(i * 100, 500)).decision);
    expect(ds.indexOf('keyframe')).toBe(3); // after the 3 frames the ring can hold
    expect(s.numKeyframes).toBe(1);
  });

  it('repeated revertLastKeyframe walks back through the keyframe history', () => {
    const s = new KeyframeSelector();
    for (let i = 0; i < 6; i++) s.evaluate(cand(i * 100, 500));
    expect(s.lastKeyframeTime).toBe(500);
    expect(s.evaluate(cand(1000, 10)).trigger).toBe('tracking_lost');
    expect(s.evaluate(cand(1300, 10)).trigger).toBe('tracking_lost');
    expect(s.numKeyframes).toBe(3);
    s.revertLastKeyframe();
    expect(s.lastKeyframeTime).toBe(1000);
    s.revertLastKeyframe();
    // before the fix: lastKeyframeTime = null and the next frame became a 'first' keyframe
    expect(s.lastKeyframeTime).toBe(500);
    expect(s.numKeyframes).toBe(1);
    const d = s.evaluate(cand(1400, 500));
    expect(d.trigger).not.toBe('first');
    s.revertLastKeyframe();
    expect(s.lastKeyframeTime).toBeNull();
    expect(s.numKeyframes).toBe(0);
    expect(s.evaluate(cand(1500, 500)).trigger).toBe('first');
    s.revertLastKeyframe();
    s.revertLastKeyframe(); // no-op without a keyframe
    expect(s.numKeyframes).toBe(0);
    // history is bounded but keeps the most recent keyframes
    for (let i = 0; i < 40; i++) s.commitKeyframe(10000 + 1000 * i);
    for (let i = 0; i < 5; i++) s.revertLastKeyframe();
    expect(s.lastKeyframeTime).toBe(10000 + 1000 * 34);
  });

  it('a non-finite timestamp is skipped without corrupting the state', () => {
    const s = new KeyframeSelector();
    for (let i = 0; i < 6; i++) s.evaluate(cand(i * 100, 500));
    const d = s.evaluate(cand(NaN, 5)); // would be a tracking_lost keyframe at t = NaN
    expect(d.decision).toBe('skip');
    expect(s.lastKeyframeTime).toBe(500);
    expect(s.evaluate(cand(1000, 5)).trigger).toBe('tracking_lost');
    expect(s.lastKeyframeTime).toBe(1000);
  });

  it('an unknown (NaN) inlier count is neither tracking loss nor an overlap drop', () => {
    const s = new KeyframeSelector();
    for (let i = 0; i < 6; i++) s.evaluate(cand(i * 100, 500));
    for (let i = 1; i <= 10; i++) expect(s.evaluate(cand(500 + i * 100, NaN)).decision).toBe('skip');
    // the parallax trigger still works
    expect(s.evaluate(cand(1700, NaN, { medianParallaxPx: 100 })).trigger).toBe('parallax');
  });
});

describe('rotationCompensatedParallax (review)', () => {
  const cam: CameraIntrinsics = { width: 1000, height: 750, f: 800, cx: 499.5, cy: 374.5, k1: 0, k2: 0 };
  function scene(rng: Rng, R: Float64Array, n: number) {
    const kpA = new Float32Array(4 * n),
      kpB = new Float32Array(4 * n),
      m = new Uint32Array(2 * n);
    for (let i = 0; i < n; i++) {
      const X = Float64Array.of(rng.uniform(-1, 1), rng.uniform(-0.8, 0.8), rng.uniform(2, 6));
      const Y = mat3MulVec(R, X);
      kpA[4 * i] = (cam.f * X[0]) / X[2] + cam.cx;
      kpA[4 * i + 1] = (cam.f * X[1]) / X[2] + cam.cy;
      kpB[4 * i] = (cam.f * Y[0]) / Y[2] + cam.cx;
      kpB[4 * i + 1] = (cam.f * Y[1]) / Y[2] + cam.cy;
      m[2 * i] = i;
      m[2 * i + 1] = i;
    }
    return { kpA, kpB, m };
  }

  it('ignores NaN keypoints / out-of-range indices; K matrix path = intrinsics path', () => {
    const rng = new Rng(11);
    const R = rotvecToMat3([0.02, -0.05, 0.01]);
    const { kpA, kpB, m } = scene(rng, R, 51);
    kpB[4 * 7] = NaN; // corrupt keypoint
    const mm = new Uint32Array(m.length + 2);
    mm.set(m);
    mm[m.length] = 9999; // out-of-range index
    mm[m.length + 1] = 3;
    const p = rotationCompensatedParallax(kpA, kpB, mm, cam, R);
    expect(Number.isFinite(p)).toBe(true);
    expect(p).toBeLessThan(1e-3); // pure rotation ⇒ no parallax (float32 keypoints)
    const K = Float64Array.of(cam.f, 0, cam.cx, 0, cam.f, cam.cy, 0, 0, 1);
    const p2 = rotationCompensatedParallax(kpA, kpB, mm, K, R);
    expect(Math.abs(p2 - p)).toBeLessThan(1e-6);
    // even number of usable matches ⇒ mean of the two middle values
    const two = rotationCompensatedParallax(
      Float32Array.of(0, 0, 0, 0, 10, 0, 0, 0),
      Float32Array.of(3, 0, 0, 0, 17, 0, 0, 0),
      Uint32Array.of(0, 0, 1, 1),
      K,
      Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1),
    );
    expect(two).toBeCloseTo(5, 9);
  });

  it('rotation by π about the vertical axis: every point is behind ⇒ NaN', () => {
    const rng = new Rng(12);
    const { kpA, kpB, m } = scene(rng, rotvecToMat3([0, 0.1, 0]), 20);
    const Rpi = rotvecToMat3([0, Math.PI, 0]);
    expect(rotationCompensatedParallax(kpA, kpB, m, cam, Rpi)).toBeNaN();
    expect(rotationCompensatedParallax(kpA, kpB, new Uint32Array(0), cam, Rpi)).toBeNaN();
  });
});

describe('pairs (review)', () => {
  it('mergePairs: fast path matches a reference and is fast; generic path for large / negative ids', () => {
    const rng = new Rng(1);
    const big: [number, number][] = [];
    for (let i = 0; i < 200000; i++) big.push([rng.int(3000), rng.int(3000)]);
    // Best of 3 calls on each side: the first call of either includes JIT warm-up, and the minimum
    // is the noise-robust estimate under machine load (single cold calls measured 0.35–0.7×).
    let r: ReturnType<typeof mergePairs> = [];
    let ms = Infinity;
    for (let rep = 0; rep < 3; rep++) {
      const t0 = performance.now();
      r = mergePairs([big, big]);
      ms = Math.min(ms, performance.now() - t0);
    }
    const ref = new Set<string>();
    for (const [a, b] of big) if (a !== b) ref.add(`${Math.min(a, b)},${Math.max(a, b)}`);
    expect(r.length).toBe(ref.size);
    let sorted = true;
    for (let i = 1; i < r.length; i++) sorted &&= r[i - 1][0] < r[i][0] || (r[i - 1][0] === r[i][0] && r[i - 1][1] < r[i][1]);
    expect(sorted).toBe(true);
    for (let i = 0; i < r.length; i += 997) expect(ref.has(`${r[i][0]},${r[i][1]}`)).toBe(true);
    // Relative to the pre-review algorithm (Set<number> + sort of pair objects) on the same data, so
    // the check is robust to machine load: ≈ 50 ms vs ≈ 150–250 ms (Node 22, 4-core VM).
    let old: [number, number][] = [];
    let oldMs = Infinity;
    for (let rep = 0; rep < 3; rep++) {
      const t1 = performance.now();
      const seen = new Set<number>();
      old = [];
      for (const list of [big, big])
        for (const [x, y] of list) {
          const a = Math.min(x, y),
            b = Math.max(x, y);
          if (a === b) continue;
          const key = a * 67108864 + b;
          if (seen.has(key)) continue;
          seen.add(key);
          old.push([a, b]);
        }
      old.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
      oldMs = Math.min(oldMs, performance.now() - t1);
    }
    expect(old).toEqual(r);
    expect(ms).toBeLessThan(0.5 * oldMs * perfSlack());
    // ids beyond the 2^26 key range and negative ids: no key collisions
    const L = 2 ** 26;
    const g = mergePairs([
      [
        [L, 1],
        [1, L],
        [0, L + 1],
        [-5, 3],
        [3, -5],
        [NaN, 2],
        [7, 7],
      ],
    ]);
    expect(g).toEqual([
      [-5, 3],
      [0, L + 1],
      [1, L],
    ]);
    // ids ≥ 2^16 (Float64 keys) and the 2^16 boundary of the Uint32 keys; NaN / Infinity dropped
    expect(
      mergePairs([
        [
          [70000, 5],
          [5, 70000],
          [L - 1, 65536],
          [Infinity, 1],
          [3, 2],
        ],
      ]),
    ).toEqual([
      [2, 3],
      [5, 70000],
      [65536, L - 1],
    ]);
    expect(
      mergePairs([
        [
          [65535, 0],
          [65534, 65535],
          [0, 65535],
          [1, NaN],
        ],
      ]),
    ).toEqual([
      [0, 65535],
      [65534, 65535],
    ]);
    // exclude is called once per distinct pair
    let calls = 0;
    mergePairs(
      [
        [
          [1, 2],
          [2, 1],
          [1, 2],
        ],
      ],
      () => (calls++, false),
    );
    expect(calls).toBe(1);
  });

  it('spatialPairs never pairs cameras with a non-finite pose; 180° accepts opposite views', () => {
    const I = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
    const cams = [0, 1, 2].map((i) => ({ id: i, pose: { R: I, t: Float64Array.of(i, 0, 0) } }));
    cams.push({ id: 9, pose: { R: Float64Array.of(NaN, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(0, 0, 0) } });
    cams.push({ id: 8, pose: { R: I, t: Float64Array.of(0, NaN, 0) } });
    const p = spatialPairs(cams, 3, 30);
    expect(p.some(([a, b]) => a >= 8 || b >= 8)).toBe(false);
    expect(p).toEqual([
      [0, 1],
      [0, 2],
      [1, 2],
    ]);
    const back = { id: 5, pose: { R: Float64Array.of(-1, 0, 0, 0, 1, 0, 0, 0, -1), t: Float64Array.of(0, 0, 0.5) } };
    expect(spatialPairs([cams[0], back], 1, 180)).toEqual([[0, 5]]);
    expect(spatialPairs([cams[0], back], 1, 179)).toEqual([]);
  });
});

describe('vocabulary (review)', () => {
  it('degenerate training data (all descriptors identical): finite, deterministic, same on both paths', () => {
    const n = 500;
    const desc = new Uint8Array(n * 128).fill(37);
    const a = trainVocabulary([{ descriptors: desc, count: n }], { numWords: 16, simd: true });
    const b = trainVocabulary([{ descriptors: desc, count: n }], { numWords: 16, simd: false });
    expect(a.every((v) => v === 37)).toBe(true);
    expect(b.every((v) => v === 37)).toBe(true);
    const w = quantizeDescriptors(a, 16, desc, 10);
    expect(Array.from(w)).toEqual(new Array(10).fill(0)); // ties → lowest index
  });

  it('SIMD and JS k-means give the same codebook (determinism across devices)', () => {
    const rng = new Rng(5);
    const n = 3000;
    const desc = new Uint8Array(n * 128);
    const centres = Array.from({ length: 12 }, () => Array.from({ length: 128 }, () => rng.int(200)));
    for (let i = 0; i < n; i++) {
      const c = centres[i % 12];
      for (let d = 0; d < 128; d++) desc[i * 128 + d] = Math.min(255, Math.max(0, c[d] + Math.round(8 * rng.gaussian())));
    }
    const a = trainVocabulary([{ descriptors: desc, count: n }], { numWords: 48, simd: true });
    const b = trainVocabulary([{ descriptors: desc, count: n }], { numWords: 48, simd: false });
    let maxd = 0;
    for (let i = 0; i < a.length; i++) maxd = Math.max(maxd, Math.abs(a[i] - b[i]));
    expect(maxd).toBeLessThan(1e-9);
  });

  it('out-of-range word ids / bad tf in a BowVector do not produce NaN scores', () => {
    const rng = new Rng(9);
    const cb = new Float64Array(8 * 128);
    for (let i = 0; i < cb.length; i++) cb[i] = rng.int(256);
    const idx = new VocabularyIndex(cb);
    idx.addBow(1, { words: Uint32Array.of(0, 2, 5), tf: Float32Array.of(1, 1, 1) });
    idx.addBow(2, { words: Uint32Array.of(0, 2, 9, 4000), tf: Float32Array.of(1, 2, 1, 1) }); // 9, 4000 ≥ k
    idx.addBow(3, { words: Uint32Array.of(1, 3, 6), tf: Float32Array.of(1, NaN, 1) });
    idx.addBow(4, { words: Uint32Array.of(1, 7), tf: Float32Array.of(1, 1) });
    for (const id of [1, 2, 3, 4]) {
      const r = idx.query(id, 3);
      expect(r.length).toBe(3);
      for (const x of r) expect(Number.isFinite(x.score)).toBe(true);
    }
    expect(idx.query(1, 1)[0].id).toBe(2);
    const q = idx.query({ words: Uint32Array.of(0, 2, 77), tf: Float32Array.of(1, 1, 1) }, 2);
    expect(q.map((x) => x.id)).toEqual([2, 1]);
    expect(Number.isFinite(q[0].score)).toBe(true);
    expect(idx.getBow(2)!.words.length).toBe(2);
  });
});

describe('inertial (review)', () => {
  it('time offset unobservable from rotation ⇒ estimated from linear acceleration (scale was 4–9 % off)', () => {
    for (const [kind, seed, off, accelG] of [
      ['walk', 1, 40, false],
      ['walk', 2, -90, true],
      ['orbit', 3, 70, false],
    ] as const) {
      const rng = new Rng(seed * 31);
      const traj = noRotation(kind === 'walk' ? walkTrajectory(rng) : orbitTrajectory(rng));
      const Rcd = cameraFromDevice(90);
      const imu = simulateImu(traj, Rcd, rng, accelG ? { withAccel: false } : {});
      const kf = simulateKeyframes(traj, Rcd, rng, { offsetMs: off, s: 2 });
      const al = estimateInertialAlignment(kf.keyframes, imu);
      expect(al.timeOffset === null || !al.timeOffset.reliable).toBe(true);
      expect(al.offsetSource).toBe('accel');
      expect(Math.abs(al.offsetMs - off)).toBeLessThan(3);
      expect(al.accelTimeOffset!.stdMs).toBeLessThan(3);
      const err = al.scale!.scale * kf.s - 1;
      expect(al.scale!.reliable).toBe(true);
      expect(Math.abs(err)).toBeLessThan(0.02);
      if (kind === 'walk') {
        // the bias the fallback removes: the default offset (0) gives a clearly wrong scale
        const wrong = estimateMetricScale(kf.keyframes, imu, { offsetMs: 0, gravity: al.gravity })!;
        expect(Math.abs(wrong.scale * kf.s - 1)).toBeGreaterThan(0.03);
      }
    }
  });

  it('acceleration offset: agrees with the gyro one on rich motion, refuses a static camera, unused when the gyro offset is reliable', () => {
    const rng = new Rng(77);
    const traj = orbitTrajectory(rng);
    const Rcd = cameraFromDevice(0);
    const imu = simulateImu(traj, Rcd, rng);
    const kf = simulateKeyframes(traj, Rcd, rng, { offsetMs: -20, s: 2 });
    const al = estimateInertialAlignment(kf.keyframes, imu);
    expect(al.offsetSource).toBe('gyro');
    expect(al.accelTimeOffset).toBeNull();
    const acc = estimateTimeOffsetFromAcceleration(kf.keyframes, imu, { scale: { gravity: al.gravity } })!;
    expect(Math.abs(acc.offsetMs + 20)).toBeLessThan(3);
    // static camera: no excitation ⇒ unreliable
    const still: SimTrajectory = { duration: 15, pos: (_t, o) => o.set([0, 0, 1.5]), Rwc: (_t, o) => lookAt(1, 0, 0, o) };
    const imu2 = simulateImu(still, Rcd, rng);
    const kf2 = simulateKeyframes(still, Rcd, rng, { offsetMs: 0, s: 2 });
    const acc2 = estimateTimeOffsetFromAcceleration(kf2.keyframes, imu2, { scale: { gravity: { down: kf2.downRec } } });
    expect(acc2 === null || !acc2.reliable).toBe(true);
  });

  it('non-finite gyro / accel samples are dropped instead of poisoning every estimate', () => {
    const rng = new Rng(21);
    const traj = orbitTrajectory(rng);
    const Rcd = cameraFromDevice(0);
    const imu: ImuSample[] = simulateImu(traj, Rcd, rng);
    const kf = simulateKeyframes(traj, Rcd, rng, { offsetMs: 15, s: 3 });
    imu[300] = { ...imu[300], gyro: [NaN, 0, 0] };
    imu[900] = { ...imu[900], accel: [0, NaN, 0], accelG: [Infinity, 0, 0] };
    const g = new GyroIntegrator(imu);
    expect(g.n).toBe(imu.length - 1);
    expect(Number.isFinite(g.q[4 * g.n - 1])).toBe(true);
    const al = estimateInertialAlignment(kf.keyframes, imu);
    expect(al.offsetSource).toBe('gyro');
    expect(Math.abs(al.offsetMs - 15)).toBeLessThan(3);
    expect(Math.abs(al.scale!.scale * 3 - 1)).toBeLessThan(0.03);
  });

  it('scale is invariant to the reconstruction units (1e-3 … 1e3) and works with a 60 Hz jittery IMU', () => {
    for (const [s, seed] of [
      [1e-3, 1],
      [1e3, 2],
    ] as const) {
      const rng = new Rng(500 + seed);
      const traj = orbitTrajectory(rng);
      const Rcd = cameraFromDevice(0);
      const imu = simulateImu(traj, Rcd, rng, { rateHz: 60, jitterMs: 3 });
      const kf = simulateKeyframes(traj, Rcd, rng, { offsetMs: 25, s, t: [s * 3, -s, 7 * s] });
      const al = estimateInertialAlignment(kf.keyframes, imu);
      expect(al.offsetSource).toBe('gyro');
      expect(Math.abs(al.offsetMs - 25)).toBeLessThan(3);
      expect(al.scale!.reliable).toBe(true);
      expect(Math.abs(al.scale!.scale * s - 1)).toBeLessThan(0.03);
    }
  });

  it('time offset near the edge of the search range: found inside, flagged outside', () => {
    const rng = new Rng(77);
    const traj = orbitTrajectory(rng);
    const Rcd = cameraFromDevice(0);
    const imu = simulateImu(traj, Rcd, rng);
    for (const off of [-240, 240]) {
      const r = estimateTimeOffset(simulateKeyframes(traj, Rcd, new Rng(1), { offsetMs: off, s: 2 }).keyframes, imu)!;
      expect(Math.abs(r.offsetMs - off)).toBeLessThan(2);
      expect(r.reliable).toBe(true);
    }
    for (const off of [320, -400]) {
      const r = estimateTimeOffset(simulateKeyframes(traj, Rcd, new Rng(1), { offsetMs: off, s: 2 }).keyframes, imu);
      expect(r === null || !r.reliable).toBe(true);
    }
  });
});

describe('geo (review)', () => {
  function track(frame: EnuFrame, n: number, mut?: (i: number, f: GeoFix) => void) {
    const fixes: GeoFix[] = [];
    const centres: number[] = [];
    const times: number[] = [];
    for (let i = 0; i < n; i++) {
      const e = i * 2,
        nn = i * 0.5 + 0.05 * i * i; // curved: a full Sim3 needs a non-collinear track
      const g = frame.toGeodetic(e, nn, 3);
      const f: GeoFix = { t: 1000 * i, lat: g[0], lon: g[1], alt: g[2], accuracy: 3, altitudeAccuracy: 4 };
      mut?.(i, f);
      fixes.push(f);
      centres.push(e / 2, nn / 2, 1);
      times.push(1000 * i);
    }
    return { fixes, centres, times };
  }

  it('a fix with NaN accuracy / altitudeAccuracy no longer turns the whole alignment into null / NaN', () => {
    const frame = new EnuFrame(47.3, 8.5, 400);
    const { fixes, centres, times } = track(frame, 40, (i, f) => {
      if (i === 3) f.accuracy = NaN;
      if (i === 4) f.altitudeAccuracy = NaN;
    });
    for (const gravityAligned of [true, false]) {
      const r = alignToGps(centres, times, fixes, { gravityAligned })!;
      expect(r).not.toBeNull();
      expect(r.numFixes).toBe(39);
      expect(r.origin.every(Number.isFinite)).toBe(true);
      expect(Math.abs(r.s - 2)).toBeLessThan(1e-6);
      expect(Array.from(r.t).every(Number.isFinite)).toBe(true);
    }
  });

  it('antimeridian track: continuous ENU, origin near ±180°, exact similarity', () => {
    const frame = new EnuFrame(-16.5, 179.9997, 10);
    const { fixes, centres, times } = track(frame, 40);
    expect(fixes.some((f) => f.lon < 0) && fixes.some((f) => f.lon > 0)).toBe(true);
    const r = alignToGps(centres, times, fixes, { gravityAligned: true })!;
    expect(Math.abs(Math.abs(r.origin[1]) - 180)).toBeLessThan(0.01);
    expect(Math.abs(r.s - 2)).toBeLessThan(1e-6);
    expect(r.rmsResidual).toBeLessThan(1e-4);
  });

  it('ENU round trip is sub-micrometre at the poles, the antimeridian and 100 km away', () => {
    for (const [lat, lon] of [
      [90, 0],
      [-89.9999999, 123],
      [0, -180],
    ]) {
      const f = new EnuFrame(lat, lon, 50);
      for (const p of [
        [100000, -50000, 300],
        [-3, 4, -2],
      ]) {
        const g = f.toGeodetic(p[0], p[1], p[2]);
        const back = f.fromGeodetic(g[0], g[1], g[2]);
        expect(Math.hypot(back[0] - p[0], back[1] - p[1], back[2] - p[2])).toBeLessThan(1e-6);
      }
    }
  });

  it('compass observations with a non-finite accuracy use the 20° default', () => {
    const r = yawFromCompass([
      { headingDeg: 10, accuracyDeg: NaN, direction: [0, 1, 0] },
      { headingDeg: 10, accuracyDeg: 5, direction: [0, 1, 0] },
    ])!;
    expect(r.numInliers).toBe(2);
    expect(r.yaw).toBeCloseTo(-10 * DEG, 12);
  });
});

describe('world (review)', () => {
  function toWorld(sim: ReturnType<typeof simulateKeyframes>): WorldKeyframe[] {
    return sim.keyframes.map((k) => {
      const R = mat3Transpose(k.R_wc);
      const t = mat3MulVec(R, k.C);
      for (let j = 0; j < 3; j++) t[j] = -t[j];
      return { id: k.id, t: k.t, pose: { R, t }, R_cam_device: k.R_cam_device };
    });
  }

  it('steady-camera walk: offset from acceleration, metric scale within 2 % (was ≈ 5 % off)', () => {
    const rng = new Rng(62);
    const traj = noRotation(walkTrajectory(rng));
    const Rcd = cameraFromDevice(90);
    const imu = simulateImu(traj, Rcd, rng);
    const sim = simulateKeyframes(traj, Rcd, rng, { offsetMs: 45, s: 0.7 });
    const res = computeWorldAlignment({ keyframes: toWorld(sim), imu });
    expect(res.details.offsetSource).toBe('accel');
    expect(Math.abs(res.details.offsetMs - 45)).toBeLessThan(3);
    expect(res.metric.source).toBe('imu');
    expect(Math.abs(res.s * sim.s - 1)).toBeLessThan(0.02);
    expect(res.details.warnings.some((w) => /time offset not observable/.test(w))).toBe(false);
  });

  it('GPS with a corrupt (NaN accuracy) fix still geo-references the walk', () => {
    const rng = new Rng(63);
    const traj = walkTrajectory(rng);
    const Rcd = cameraFromDevice(90);
    const imu = simulateImu(traj, Rcd, rng);
    const sim = simulateKeyframes(traj, Rcd, rng, { offsetMs: -10, s: 1.9 });
    const { fixes } = simulateGps(traj, rng, { sigma: 3 });
    fixes[5] = { ...fixes[5], accuracy: NaN };
    const res = computeWorldAlignment({ keyframes: toWorld(sim), imu, gpsFixes: fixes });
    expect(res.details.gps).not.toBeNull();
    expect(res.metric.northAligned).toBe(true);
    expect(res.metric.originLatLonAlt!.every(Number.isFinite)).toBe(true);
    expect(Array.from(res.t).every(Number.isFinite)).toBe(true);
  });
});
