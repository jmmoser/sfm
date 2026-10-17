import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, FeatureSet, FrameInfo, Pose } from '../core/types';
import { projectWorld } from '../geometry/camera';
import { Rng } from '../math/random';
import { rotationAngle, rotationAngleBetween, rotvecToMat3 } from '../math/rotation';
import { estimateTwoViewGeometry } from '../geometry/twoview';
import { FeatureDatabase } from './database';
import {
  IncrementalMapper,
  SfMCancelledError,
  buildCorrespondenceGraph,
  intrinsicsFromFov,
  resolveMapperOptions,
  runIncrementalSfM,
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
