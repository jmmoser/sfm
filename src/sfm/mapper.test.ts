import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, FeatureSet, FrameInfo, Pose } from '../core/types';
import { projectWorld } from '../geometry/camera';
import { mat3Mul, mat3MulTranspose } from '../math/mat3';
import { Rng } from '../math/random';
import { rotationAngle, rotationAngleBetween, rotvecToMat3 } from '../math/rotation';
import { estimateTwoViewGeometry } from '../geometry/twoview';
import { CorrespondenceGraph } from './correspondence-graph';
import { reconstructionInvariantErrors } from './core-testutil';
import { FeatureDatabase } from './database';
import { LiveMapper, type LiveMapperOptions } from './live';
import {
  IncrementalMapper,
  SfMCancelledError,
  buildCorrespondenceGraph,
  intrinsicsFromFov,
  resolveMapperOptions,
  runIncrementalSfM,
  type MapperOptions,
} from './mapper';
import { Reconstruction } from './reconstruction';
import { buildSyntheticDatabase, combineScenes, evaluateReconstruction, generateScene } from './synthetic';
import { ROTATION_ONLY_CAMERA, rotationOnlyDb } from './e2e-testutil';

const DEG = Math.PI / 180;

function smallOrbit(numImages = 20, seed = 1) {
  const scene = generateScene({
    kind: 'orbit',
    numImages,
    seed,
    numPoints: 2500,
    noisePx: 0.5,
    outlierRatio: 0.2,
    descriptors: false,
    k1: 0.02,
  });
  return scene;
}

describe('mapper building blocks', () => {
  it('resolves options (nested triangulation merged, bogus thresholds shared)', () => {
    const o = resolveMapperOptions({ triangulation: { minAngleDeg: 3 } as never, maxExtraParam: 0.5 });
    expect(o.triangulation.minAngleDeg).toBe(3);
    expect(o.triangulation.createMaxAngleErrorDeg).toBe(2);
    expect(o.triangulation.maxExtraParam).toBe(0.5);
    expect(o.absPoseMaxErrorPx).toBe(12);
    expect(o.initMinNumInliers).toBe(100);
  });

  it('intrinsicsFromFov uses the long side', () => {
    const c = intrinsicsFromFov(1440, 1920, 69);
    expect(c.f).toBeCloseTo(960 / Math.tan(34.5 * DEG), 9);
    expect(c.cx).toBe(719.5);
    expect(c.cy).toBe(959.5);
  });

  it('buildCorrespondenceGraph skips degenerate and small pairs', () => {
    const scene = smallOrbit(6);
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    db.setTwoView(0, 5, { config: 'degenerate', inliers: Uint32Array.of(0, 0, 1, 1) });
    db.setTwoView(1, 5, { config: 'calibrated', inliers: Uint32Array.of(0, 0, 1, 1) });
    const g = buildCorrespondenceGraph(db, 15);
    expect(g.numCorrespondencesBetween(0, 5)).toBe(0);
    expect(g.numCorrespondencesBetween(1, 5)).toBe(0);
    expect(g.numCorrespondencesBetween(0, 1)).toBeGreaterThan(100);
  });

  it('shares one camera per image size (or per cameraKey) and scales the intrinsics guess', () => {
    const scene = smallOrbit(4);
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    const half: CameraIntrinsics = { ...scene.camera, width: 960, height: 720, f: scene.camera.f / 2, cx: 479.5, cy: 359.5 };
    const g = buildCorrespondenceGraph(db);
    const rec = new Reconstruction();
    const m = new IncrementalMapper(db, g, rec, { intrinsics: half });
    for (const id of db.frameIds()) m.ensureImage(id);
    expect(rec.cameras.size).toBe(1);
    const cam = rec.cameras.get(0)!;
    expect(cam.width).toBe(1920);
    expect(cam.f).toBeCloseTo(scene.camera.f, 9);
    expect(cam.cx).toBeCloseTo(959.5, 9);
    const rec2 = new Reconstruction();
    const m2 = new IncrementalMapper(db, g, rec2, { cameraKey: (id) => id % 2, intrinsics: (id) => (id === 1 ? { ...scene.camera, f: 1000 } : undefined) });
    for (const id of db.frameIds()) m2.ensureImage(id);
    expect(rec2.cameras.size).toBe(2);
    expect(rec2.cameras.get(rec2.images.get(1)!.cameraId)!.f).toBe(1000);
    expect(rec2.cameras.get(rec2.images.get(0)!.cameraId)!.f).toBeCloseTo(intrinsicsFromFov(1920, 1440, 69).f, 9);
  });

  it('initial pair: prefers wide-baseline pairs of a video sequence; re-estimates with current intrinsics', () => {
    const scene = smallOrbit(60, 2); // 6° steps: adjacent frames have ~6° triangulation angle
    const guess = { ...scene.camera, f: scene.camera.f * 1.08, k1: 0 };
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    const g = buildCorrespondenceGraph(db);
    const rec = new Reconstruction();
    const m = new IncrementalMapper(db, g, rec, { intrinsics: guess });
    for (const id of db.frameIds()) m.ensureImage(id);
    const init = m.findInitialPair(100, 16)!;
    expect(init).not.toBeNull();
    expect(init.geom.triAngle).toBeGreaterThanOrEqual(16 * DEG);
    expect(init.geom.numInliers).toBeGreaterThanOrEqual(100);
    expect(Math.abs(init.a - init.b)).toBeGreaterThan(1);
    // relative rotation close to the truth despite the 8 % focal error
    const Pa = scene.poses[init.a],
      Pb = scene.poses[init.b];
    const Rrel = new Float64Array(9);
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++) Rrel[3 * r + c] = Pb.R[3 * r] * Pa.R[3 * c] + Pb.R[3 * r + 1] * Pa.R[3 * c + 1] + Pb.R[3 * r + 2] * Pa.R[3 * c + 2];
    expect(rotationAngleBetween(Rrel, init.geom.R) / DEG).toBeLessThan(1.5);
    // each pair is tried only once per level
    const again = m.findInitialPair(100, 16);
    if (again) expect(again.a * 1000 + again.b).not.toBe(init.a * 1000 + init.b);
  });

  it('initial pair from an uncalibrated two-view geometry (focal 40 % too short) via E = K₂ᵀFK₁', () => {
    const scene = smallOrbit(12, 3);
    const guess = { ...scene.camera, f: scene.camera.f * 0.6, k1: 0 };
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    const g = buildCorrespondenceGraph(db);
    const rec = new Reconstruction();
    const m = new IncrementalMapper(db, g, rec, { intrinsics: guess });
    for (const id of db.frameIds()) m.ensureImage(id);
    const configs: string[] = [];
    for (let b = 1; b < 6; b++) {
      const ig = m.estimateInitPair(0, b);
      if (!ig) continue;
      configs.push(ig.config);
      expect(ig.numInliers).toBeGreaterThan(200);
      const Pa = scene.poses[0],
        Pb = scene.poses[b];
      const Rrel = new Float64Array(9);
      for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++) Rrel[3 * r + c] = Pb.R[3 * r] * Pa.R[3 * c] + Pb.R[3 * r + 1] * Pa.R[3 * c + 1] + Pb.R[3 * r + 2] * Pa.R[3 * c + 2];
      // a wrong focal biases the rotation, but the pose must be the right branch
      // (errors grow with the focal error and the rotation: ~35 % of it at f·0.6; a wrong branch is ≫)
      expect(rotationAngleBetween(Rrel, ig.R)).toBeLessThan(0.4 * rotationAngle(Rrel));
      expect(ig.triAngle).toBeGreaterThan(0);
      expect(Math.hypot(ig.t[0], ig.t[1], ig.t[2])).toBeCloseTo(1, 12);
    }
    expect(configs).toContain('uncalibrated');
  });
});

describe('runIncrementalSfM', () => {
  it('reconstructs a small orbit (GT matches with outliers removed by verification)', async () => {
    const scene = smallOrbit(20, 5);
    const guess = { ...scene.camera, f: scene.camera.f * 0.95, k1: 0 };
    const { db } = buildSyntheticDatabase(scene, { camera: guess });
    const events: number[] = [];
    // the run must yield to the event loop (worker stays responsive)
    let ticks = 0;
    const timer = setInterval(() => ticks++, 5);
    const t0 = performance.now();
    const models = await runIncrementalSfM(db, { intrinsics: guess }, (e) => events.push(e.progress));
    const ms = performance.now() - t0;
    clearInterval(timer);
    // without yields the timer could not fire at all during the run (load-independent bound)
    expect(ticks).toBeGreaterThan(5);
    expect(models.length).toBe(1);
    const rec = models[0];
    const ev = evaluateReconstruction(rec, scene);
    console.log(`small orbit: ${ms.toFixed(0)} ms, reg ${ev.numRegistered}, ATE/extent ${ev.ateRelative.toExponential(2)}, f err ${ev.focalError.toExponential(2)}, k1 ${ev.k1.toFixed(4)}, mre ${rec.meanReprojectionError().toFixed(3)}`);
    expect(ev.numRegistered).toBe(20);
    expect(ev.ateRelative).toBeLessThan(0.002);
    expect(ev.focalError).toBeLessThan(0.01);
    expect(rec.meanReprojectionError()).toBeLessThan(1);
    expect(ev.wrongObservationRate).toBeLessThan(0.01);
    expect(events.length).toBeGreaterThan(5);
    expect(events[events.length - 1]).toBe(1);
    // colours = mean of the track's feature colours
    const p = rec.points.values().next().value!;
    let r = 0;
    for (const el of p.track) r += db.colors.get(el.image)![3 * el.feature];
    expect(p.color[0]).toBe(Math.round(r / p.track.length));
    for (const q of rec.points.values()) expect(q.error).toBeGreaterThanOrEqual(0);
  });

  it('cancels with a CancelledError', async () => {
    const scene = smallOrbit(14, 6);
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    const token = { cancelled: false };
    const p = runIncrementalSfM(db, { intrinsics: scene.camera }, (e) => {
      if (e.progress > 0.3) token.cancelled = true;
    }, token);
    await expect(p).rejects.toBeInstanceOf(SfMCancelledError);
    await expect(p).rejects.toMatchObject({ name: 'CancelledError' });
  });

  it('returns no model for empty input, unconnected frames or pure rotation', async () => {
    expect(await runIncrementalSfM(new FeatureDatabase())).toEqual([]);
    const scene = smallOrbit(5, 7);
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    for (const [a, b] of [...db.pairs()]) db.deletePair(a, b);
    expect(await runIncrementalSfM(db)).toEqual([]);
    const rot = rotationOnlyDb();
    let panoramic = 0;
    for (const [, , g] of rot.pairs()) if (g.config === 'panoramic') panoramic++;
    expect(panoramic).toBeGreaterThan(0);
    expect(await runIncrementalSfM(rot, { intrinsics: ROTATION_ONLY_CAMERA })).toEqual([]);
  });

  it('a disconnected capture gives two models (largest first)', async () => {
    const a = generateScene({ kind: 'orbit', numImages: 24, seed: 11, k1: 0.03, descriptors: false, numPoints: 3000 });
    const b = generateScene({ kind: 'walk', numImages: 36, seed: 12, k1: 0.03, descriptors: false, idOffset: 1000 });
    const s = combineScenes(a, b);
    const guess = { ...s.camera, f: s.camera.f * 1.08, k1: 0 };
    const { db } = buildSyntheticDatabase(s, { camera: guess });
    const t0 = performance.now();
    const models = await runIncrementalSfM(db, { intrinsics: guess });
    console.log(`disconnected: ${(performance.now() - t0).toFixed(0)} ms, models ${models.map((m) => m.numRegisteredImages).join('+')}`);
    expect(models.length).toBe(2);
    expect(models[0].numRegisteredImages).toBe(36);
    expect(models[1].numRegisteredImages).toBe(24);
    expect(models[0].registeredImageIds().every((id) => id >= 1000)).toBe(true);
    expect(models[1].registeredImageIds().every((id) => id < 1000)).toBe(true);
    const ea = evaluateReconstruction(models[1], a),
      eb = evaluateReconstruction(models[0], b);
    expect(ea.ateRelative).toBeLessThan(0.005);
    expect(eb.ateRelative).toBeLessThan(0.005);
    expect(ea.focalError).toBeLessThan(0.01);
    expect(eb.focalError).toBeLessThan(0.01);
  });

  it('a given initial pair builds one model (not one per threshold level)', async () => {
    const a = generateScene({ kind: 'orbit', numImages: 24, seed: 11, k1: 0.03, descriptors: false, numPoints: 3000 });
    const b = generateScene({ kind: 'walk', numImages: 36, seed: 12, k1: 0.03, descriptors: false, idOffset: 1000 });
    const s = combineScenes(a, b);
    const guess = { ...s.camera, f: s.camera.f * 1.08, k1: 0 };
    const { db } = buildSyntheticDatabase(s, { camera: guess });
    const models = await runIncrementalSfM(db, { intrinsics: guess, initImagePair: [0, 3] });
    expect(models.length).toBe(1);
    expect(models[0].numRegisteredImages).toBe(24);
    expect(models[0].registeredImageIds().every((id) => id < 1000)).toBe(true);
    expect(models[0].images.get(0)!.registered && models[0].images.get(3)!.registered).toBe(true);
  });

  it('a given initial pair that fails at the strict level is retried at the relaxed ones', async () => {
    // median angle ≈ 2°: the strict level's 4° minimum point angle leaves < 15 points, 2° hundreds
    const s = generateScene({ kind: 'walk', numImages: 36, seed: 7, k1: 0.03, descriptors: false });
    const guess = { ...s.camera, f: s.camera.f * 1.08, k1: 0 };
    const { db } = buildSyntheticDatabase(s, { camera: guess });
    const models = await runIncrementalSfM(db, { intrinsics: guess, initImagePair: [0, 1] });
    expect(models.length).toBe(1);
    expect(models[0].numRegisteredImages).toBe(36);
  });
});

describe('rotationCheck (external rotation consistency, e.g. the gyro)', () => {
  it('rejected registrations / initial pairs stay unregistered; the check sees the PnP pose', async () => {
    const scene = smallOrbit(16, 4);
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    const banned = scene.imageIds[9];
    const seen: number[] = [];
    const models = await runIncrementalSfM(db, {
      intrinsics: { ...scene.camera },
      multipleModels: false,
      rotationCheck: {
        pose: (id, pose) => {
          seen.push(id);
          // the PnP pose is the true one up to the model's similarity: plausible rotation matrix
          expect(Math.abs(rotationAngle(pose.R))).toBeLessThanOrEqual(Math.PI);
          return id !== banned;
        },
        pair: (a, b) => a !== banned && b !== banned,
      },
    });
    const rec = models[0];
    expect(seen).toContain(banned);
    expect(rec.images.get(banned)!.registered).toBe(false);
    expect(rec.numRegisteredImages).toBe(15);
  });
});

describe('predictRotation (gyro-aided registration with a known rotation)', () => {
  const CAM: CameraIntrinsics = { width: 1600, height: 1200, f: 1300, cx: 799.5, cy: 599.5, k1: 0, k2: 0 };
  const poseAt = (C: ArrayLike<number>, R: Float64Array): Pose => ({
    R,
    t: Float64Array.of(-(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]), -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]), -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2])),
  });
  /** R turned by `deg` about a fixed oblique axis (a gyro prediction error). */
  const turned = (R: ArrayLike<number>, deg: number): Float64Array => mat3Mul(rotvecToMat3(Float64Array.of(0.6, -0.48, 0.64).map((v) => v * deg * DEG)), R);

  /**
   * Images 0, 1 registered (GT poses) with `nPts` triangulated points (feature j of both = point j);
   * each test image sees `good` of them (0.3 px noise; `cluster`: all in one 7 × 9 px patch), the
   * next `wrong.count` exactly where `wrong.pose` sees them (false matches on repeated structure,
   * consistent with a wrong pose) and `bad` wrong observations, through correspondences with image 0.
   */
  function rig(
    nPts: number,
    tests: { id: number; pose: Pose; good: number; bad: number; cluster?: boolean; wrong?: { pose: Pose; count: number } }[],
    opts: Partial<MapperOptions> = {},
    db = new FeatureDatabase(),
  ) {
    const rng = new Rng(3);
    const X = new Float64Array(3 * nPts);
    for (let j = 0; j < nPts; j++) X.set([-2 + 4 * rng.next(), -1.5 + 3 * rng.next(), 6 + 4 * rng.next()], 3 * j);
    const poses = [poseAt([0, 0, 0], rotvecToMat3(Float64Array.of(0, 0, 0))), poseAt([1, 0, 0], rotvecToMat3(Float64Array.of(0, -0.1, 0)))];
    const rec = new Reconstruction();
    const graph = new CorrespondenceGraph();
    const cid = rec.addCamera(CAM);
    const uv = new Float64Array(2);
    for (let i = 0; i < 2; i++) {
      const kp = new Float32Array(4 * nPts);
      for (let j = 0; j < nPts; j++) {
        projectWorld(CAM, poses[i], X.subarray(3 * j, 3 * j + 3), uv);
        kp[4 * j] = uv[0];
        kp[4 * j + 1] = uv[1];
      }
      rec.addImage(i, cid, nPts, kp);
      graph.addImage(i, nPts);
    }
    for (const t of tests) {
      const nw = t.wrong?.count ?? 0;
      const n = t.good + nw + t.bad;
      const kp = new Float32Array(4 * n);
      const m: number[] = [];
      for (let k = 0; k < n; k++) {
        if (k < t.good) {
          projectWorld(CAM, t.pose, X.subarray(3 * k, 3 * k + 3), uv);
          kp[4 * k] = t.cluster ? 10 + (k % 7) : uv[0] + 0.3 * rng.gaussian();
          kp[4 * k + 1] = t.cluster ? 10 + ((k / 7) | 0) : uv[1] + 0.3 * rng.gaussian();
        } else if (k < t.good + nw) {
          projectWorld(CAM, t.wrong!.pose, X.subarray(3 * k, 3 * k + 3), uv);
          kp[4 * k] = uv[0];
          kp[4 * k + 1] = uv[1];
        } else {
          kp[4 * k] = rng.next() * (CAM.width - 1);
          kp[4 * k + 1] = rng.next() * (CAM.height - 1);
        }
        m.push(k, k);
      }
      rec.addImage(t.id, cid, n, kp);
      graph.addImage(t.id, n);
      graph.addTwoView(0, t.id, Uint32Array.from(m));
    }
    rec.attachGraph(graph);
    rec.registerImage(0, poses[0]);
    rec.registerImage(1, poses[1]);
    for (let j = 0; j < nPts; j++)
      rec.addPoint(X.subarray(3 * j, 3 * j + 3), [
        { image: 0, feature: j },
        { image: 1, feature: j },
      ]);
    const mapper = new IncrementalMapper(db, graph, rec, { seed: 5, ...opts });
    return { rec, mapper };
  }

  const P2 = poseAt([0.4, 0.2, -0.3], rotvecToMat3(Float64Array.of(0.05, -0.04, 0.02)));
  const tests = [
    { id: 22, pose: P2, good: 26, bad: 6 }, // 81 % but 26 < 30 inliers: P3P rejects it
    { id: 24, pose: P2, good: 17, bad: 5 }, // 22 correspondences: below the P3P floor altogether
    { id: 25, pose: P2, good: 14, bad: 30 }, // 32 %: below the known-rotation ratio (0.5)
    { id: 26, pose: P2, good: 40, bad: 0, cluster: true }, // all in a 7 × 9 px patch
    { id: 27, pose: P2, good: 26, bad: 6 }, // prediction 20° off
    { id: 28, pose: P2, good: 26, bad: 6 }, // no prediction
  ];

  it('registers what P3P rejects only with a prediction it confirms, enough inliers, ratio and spread', () => {
    // without a predictor: the P3P floor (30 inliers) applies as before
    const base = rig(400, tests);
    for (const t of tests) expect(base.mapper.registerNextImage(t.id)).toBe(false);
    expect(base.mapper.registrable(24, 22)).toBe(false);
    // with one: the truth turned by 0.8° (σ 1°), 20° off for image 27, none for 28
    const asked: number[] = [];
    let recRef: Reconstruction | null = null;
    const { mapper, rec } = rig(400, tests, {
      predictRotation: (id, r) => {
        asked.push(id);
        recRef = r;
        if (id === 28) return null;
        return { R: turned(P2.R, id === 27 ? 20 : 0.8), sigmaRad: 1 * DEG };
      },
    });
    expect(mapper.registrable(24, 22)).toBe(true);
    expect(mapper.registrable(28, 22)).toBe(false); // no prediction
    expect(mapper.registrable(24, 14)).toBe(false); // < knownRotationMinNumInliers
    expect(recRef).toBe(rec);
    for (const id of [25, 26, 27, 28]) expect(mapper.registerNextImage(id)).toBe(false);
    expect(mapper.numKnownRotationRegistrations).toBe(0);
    for (const [id, good] of [
      [22, 26],
      [24, 17],
    ]) {
      expect(mapper.registerNextImage(id)).toBe(true);
      const img = rec.images.get(id)!;
      // the inliers became observations; the refined pose is the true one, not the prediction
      expect(img.numPoints3D).toBe(good);
      expect(rotationAngleBetween(img.pose.R, P2.R) / DEG).toBeLessThan(0.1);
      for (let k = 0; k < 3; k++) expect(Math.abs(img.pose.t[k] - P2.t[k])).toBeLessThan(0.01);
    }
    expect(asked).toContain(22);
    expect(mapper.numKnownRotationRegistrations).toBe(2);
    expect(reconstructionInvariantErrors(rec, { graph: mapper.graph })).toEqual([]);
  });

  it('a P3P pose the rotation check rejects falls back to the known rotation: the true pose, or none when its inliers are too few a share', () => {
    // 40 false matches consistent with a pose 25° off (P3P's best: more inliers than the 32 true
    // ones, noise-free) and 32 true ones (0.3 px)
    const W = turned(P2.R, 25);
    const wrongPose = poseAt([0.9, -0.4, 0.2], W);
    const t = [{ id: 30, pose: P2, good: 32, bad: 0, wrong: { pose: wrongPose, count: 40 } }];
    // the "gyro": the truth 0.8° off; the pose check rejects rotations more than 5° from it
    const gyroR = turned(P2.R, 0.8);
    const checked: number[] = [];
    const recovered: number[] = [];
    const opts = (ratio: number): Partial<MapperOptions> => ({
      knownRotationMinInlierRatio: ratio,
      rotationCheck: {
        pose: (_id, pose) => (checked.push(rotationAngleBetween(pose.R, gyroR) / DEG), rotationAngleBetween(pose.R, gyroR) < 5 * DEG),
        poseRecovered: (id) => recovered.push(id),
      },
      predictRotation: () => ({ R: gyroR, sigmaRad: 1 * DEG }),
    });
    // without a predictor the rejected P3P pose is the end of it
    const plain = rig(400, t, { rotationCheck: opts(0.5).rotationCheck });
    expect(plain.mapper.registerNextImage(30)).toBe(false);
    expect(plain.mapper.numRotationRejects).toBe(1);
    expect(checked[0]).toBeGreaterThan(20); // P3P chose the wrong pose
    // with one: 32 of 72 (44 %) is too small a share for the default ratio (0.5) …
    checked.length = 0;
    const strict = rig(400, t, opts(0.5));
    expect(strict.mapper.registerNextImage(30)).toBe(false);
    expect(strict.mapper.lastRegistration).toMatchObject({ correspondences: 72, via: null });
    expect(strict.mapper.lastRegistration!.p3pInliers).toBeGreaterThanOrEqual(40);
    expect(strict.mapper.lastRegistration!.knownRotationInliers).toBe(32);
    expect(strict.mapper.numRotationRejects).toBe(1);
    expect(strict.mapper.lastRegistration).toMatchObject({ p3pRotationRejected: true, knownRotationRejected: false });
    expect(strict.mapper.numRotationRejectsRecovered).toBe(0);
    expect(strict.mapper.numKnownRotationRegistrations).toBe(0);
    expect(recovered).toEqual([]);
    // … and enough at 0.4: registered with the true pose, the check passed it
    checked.length = 0;
    const { mapper, rec } = rig(400, t, opts(0.4));
    expect(mapper.registerNextImage(30)).toBe(true);
    expect(mapper.lastRegistration).toMatchObject({ via: 'knownRotation', knownRotationInliers: 32, p3pRotationRejected: true, knownRotationRejected: false });
    expect(checked).toHaveLength(2);
    expect(checked[1]).toBeLessThan(1);
    // the rejected P3P pose is counted, and reported as recovered (not a failed registration)
    expect(mapper.numRotationRejects).toBe(1);
    expect(mapper.numRotationRejectsRecovered).toBe(1);
    expect(recovered).toEqual([30]);
    const img = rec.images.get(30)!;
    expect(rotationAngleBetween(img.pose.R, P2.R) / DEG).toBeLessThan(0.1);
    for (let k = 0; k < 3; k++) expect(Math.abs(img.pose.t[k] - P2.t[k])).toBeLessThan(0.01);
    expect(img.numPoints3D).toBe(32);
    expect([...mapper.knownRotationImages]).toEqual([30]);
    // P3P alone for this call (LiveMapper before re-triangulating): no fallback
    const p3pOnly = rig(400, t, opts(0.4));
    expect(p3pOnly.mapper.registerNextImage(30, { knownRotation: false })).toBe(false);
    expect(p3pOnly.mapper.lastRegistration!.knownRotationInliers).toBeNaN();
    expect(p3pOnly.mapper.registerNextImage(30, { p3p: false })).toBe(true);
    expect(p3pOnly.mapper.lastRegistration!.p3pInliers).toBeNaN();
  });

  it('the known-rotation pose must agree with the baseline of its best verified pair', () => {
    const t = [{ id: 22, pose: P2, good: 26, bad: 6 }];
    const opts: Partial<MapperOptions> = { predictRotation: () => ({ R: turned(P2.R, 0.8), sigmaRad: 1 * DEG }) };
    // image 0 is at the origin with R = I: the pair 0 → 22 is (R2, t2 / |t2|)
    const dir = (sign: number): Float64Array => P2.t.map((v) => (sign * v) / Math.hypot(P2.t[0], P2.t[1], P2.t[2]));
    const withPair = (sign: number, triAngleDeg: number, config: 'calibrated' | 'planar' = 'calibrated') => {
      const db = new FeatureDatabase();
      db.setTwoView(0, 22, { config, inliers: new Uint32Array(400), R: P2.R, t: dir(sign), triAngle: triAngleDeg * DEG });
      return rig(400, t, opts, db).mapper;
    };
    const right = withPair(1, 3);
    expect(right.registerNextImage(22)).toBe(true);
    expect(right.lastRegistration!.pairBaselineDeg).toBeLessThan(1);
    // reversed baseline (a pose the rotation–translation coupling of far points allows): refused
    const reversed = withPair(-1, 3);
    expect(reversed.registerNextImage(22)).toBe(false);
    expect(reversed.lastRegistration).toMatchObject({ via: null, knownRotationInliers: 26 });
    expect(reversed.lastRegistration!.pairBaselineDeg).toBeGreaterThan(170);
    // a pair at too small a triangulation angle, or a planar one, says nothing about the baseline
    for (const m of [withPair(-1, 1), withPair(-1, 3, 'planar')]) {
      expect(m.registerNextImage(22)).toBe(true);
      expect(m.lastRegistration!.pairBaselineDeg).toBeNaN();
    }
  });

  it('LiveMapper: a keyframe that sees only a few map points (after a pan) and the walk after it register instead of staying pending', () => {
    const scene = generateScene({ kind: 'walk', numImages: 30, seed: 6, numPoints: 6000, noisePx: 0.5, descriptors: false });
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    const ids = db.frameIds();
    const K = 12;
    const gt = new Map(ids.map((id, k) => [id, scene.poses[k]]));
    // What a fast pan at a low analysis rate leaves (e2e yard-real at stride 3): keyframe K looks at a
    // region only K − 1 and K − 2 saw, at too little parallax to triangulate it (here: its map points
    // are dropped but for 24), and the keyframes after it overlap only K and later ones. (Without
    // re-triangulating the neighbours, which would rebuild the dropped points: src/sfm/live.test.ts.)
    const stream = (opts: Partial<LiveMapperOptions>): LiveMapper => {
      const graph = new CorrespondenceGraph();
      const live = new LiveMapper(db, graph, { intrinsics: { ...scene.camera }, retriangulateNeighbours: 0, ...opts });
      for (const id of ids) {
        graph.addImage(id, db.features.get(id)!.count);
        const near = id === ids[K] ? [ids[K - 1], ids[K - 2]] : null;
        if (near) {
          const rec = live.reconstruction;
          const keep = new Set<number>();
          for (const j of near) {
            const g = db.getTwoView(j, id)!;
            const img = rec.images.get(j)!;
            for (let k = 0; k < g.inliers.length; k += 2) {
              const pid = img.point3D[g.inliers[k]];
              if (pid < 0 || keep.has(pid)) continue;
              if (keep.size < 24) keep.add(pid);
              else rec.deletePoint(pid);
            }
          }
        }
        for (const j of ids) {
          if (j >= id) break;
          const g = db.getTwoView(j, id);
          if (!g || (near && !near.includes(j)) || (id > ids[K] && j < ids[K])) continue;
          graph.addTwoView(j, id, g.inliers);
        }
        live.addKeyframe(id);
      }
      return live;
    };
    const plain = stream({});
    expect(plain.pendingIds()).toEqual(ids.slice(K)); // lost for good
    // the "gyro": GT relative rotation from the registered keyframe nearest in capture order, 0.5° off
    const rng = new Rng(8);
    const live = stream({
      predictRotation: (id, rec) => {
        let nb = -1;
        for (const r of rec.registeredImageIds()) if (r !== id && (nb < 0 || Math.abs(r - id) < Math.abs(nb - id))) nb = r;
        if (nb < 0 || Math.abs(nb - id) > 5) return null;
        const G = mat3MulTranspose(gt.get(id)!.R, gt.get(nb)!.R);
        const E = rotvecToMat3(Float64Array.of(rng.gaussian(), rng.gaussian(), rng.gaussian()).map((v) => (0.5 / Math.sqrt(3)) * DEG * v));
        return { R: mat3Mul(mat3Mul(E, G), rec.images.get(nb)!.pose.R), sigmaRad: 1 * DEG };
      },
    });
    expect(live.pendingIds()).toEqual([]);
    expect(live.incrementalMapper.numKnownRotationRegistrations).toBeGreaterThanOrEqual(1);
    const ev = evaluateReconstruction(live.reconstruction, scene);
    expect(ev.numRegistered).toBe(30);
    expect(ev.ateRelative).toBeLessThan(0.005);
    expect(ev.maxRotErrorDeg).toBeLessThan(0.5);
  });
});
