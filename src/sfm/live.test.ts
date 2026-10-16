import { describe, expect, it } from 'vitest';
import { CorrespondenceGraph } from './correspondence-graph';
import { LiveMapper } from './live';
import { buildSyntheticDatabase, evaluateReconstruction, generateScene } from './synthetic';
import { ROTATION_ONLY_CAMERA, rotationOnlyDb } from './e2e-testutil';

function smallStream(numImages: number, seed: number) {
  const scene = generateScene({ kind: 'orbit', numImages, seed, numPoints: 2500, noisePx: 0.5, outlierRatio: 0.2, descriptors: false, k1: 0.02 });
  const guess = { ...scene.camera, f: scene.camera.f * 1.05, k1: 0 };
  const { db } = buildSyntheticDatabase(scene, { camera: guess });
  return { scene, db, guess };
}

describe('LiveMapper', () => {
  it('initialises once there is enough baseline, then registers keyframes in order', () => {
    const { scene, db, guess } = smallStream(36, 1);
    const graph = new CorrespondenceGraph();
    const live = new LiveMapper(db, graph, { intrinsics: guess, liveInitMinTriAngleDeg: 8 });
    const ids = db.frameIds();
    const results = [];
    for (const id of ids) {
      graph.addImage(id, db.features.get(id)!.count);
      for (const j of ids) {
        if (j >= id) break;
        const g = db.getTwoView(j, id);
        if (g) graph.addTwoView(j, id, g.inliers);
      }
      results.push(live.addKeyframe(id));
    }
    // 10° steps: the first pair with ≥ 8° baseline exists at keyframe 1
    const firstInit = results.findIndex((r) => r.initialized);
    expect(firstInit).toBeGreaterThanOrEqual(1);
    expect(firstInit).toBeLessThanOrEqual(3);
    for (let k = 0; k < firstInit; k++) expect(results[k].initialized).toBe(false);
    for (let k = firstInit; k < results.length; k++) {
      expect(results[k].initialized).toBe(true);
      expect(results[k].registered).toBe(true);
      expect(results[k].numPoints).toBeGreaterThan(100);
      expect(results[k].reprojError).toBeGreaterThan(0.2);
      expect(results[k].reprojError).toBeLessThan(1.5);
    }
    // earlier keyframes were registered after initialisation
    expect(live.pendingIds()).toEqual([]);
    const ev = evaluateReconstruction(live.reconstruction, scene);
    expect(ev.numRegistered).toBe(36);
    expect(ev.ateRelative).toBeLessThan(0.002);
    // snapshot: points / colours / poses copies
    const snap = live.snapshot();
    expect(snap.points.length).toBe(3 * live.reconstruction.numPoints);
    const p = live.reconstruction.points.values().next().value!;
    expect(snap.points[0]).toBeCloseTo(p.xyz[0], 5);
    expect([snap.colors[0], snap.colors[1], snap.colors[2]]).toEqual(p.color);
    expect(snap.poses.length).toBe(36);
    snap.poses[0].pose.t[0] += 1;
    expect(live.reconstruction.images.get(snap.poses[0].id)!.pose.t[0]).not.toBe(snap.poses[0].pose.t[0]);
    // a full refinement keeps everything registered
    live.refine();
    expect(live.reconstruction.numRegisteredImages).toBe(36);
    // re-adding a known keyframe is a no-op
    expect(live.addKeyframe(ids[5]).registered).toBe(true);
  });

  it('keeps a disconnected keyframe pending and relocalises it when later keyframes connect', () => {
    const { db, guess } = smallStream(30, 2);
    const graph = new CorrespondenceGraph();
    const live = new LiveMapper(db, graph, { intrinsics: guess });
    const ids = db.frameIds();
    const LOST = 12;
    const late: [number, number][] = [];
    for (const id of ids) {
      graph.addImage(id, db.features.get(id)!.count);
      for (const j of ids) {
        if (j >= id) break;
        const g = db.getTwoView(j, id);
        if (!g) continue;
        // keyframe LOST is first matched to nothing (e.g. motion blur / failed matching);
        // its pairs with earlier keyframes are only verified when keyframe LOST + 2 arrives
        if (id === LOST || (j === LOST && id === LOST + 1)) {
          late.push([j, id]);
          continue;
        }
        graph.addTwoView(j, id, g.inliers);
      }
      if (id === LOST + 2) for (const [a, b] of late) graph.addTwoView(a, b, db.getTwoView(a, b)!.inliers);
      const r = live.addKeyframe(id);
      if (id === LOST) {
        expect(r.registered).toBe(false);
        expect(live.pendingIds()).toContain(LOST);
      }
    }
    expect(live.pendingIds()).toEqual([]);
    expect(live.reconstruction.images.get(LOST)!.registered).toBe(true);
    expect(live.reconstruction.numRegisteredImages).toBe(30);
  });

  it('never initialises from pure rotation', () => {
    const db = rotationOnlyDb();
    const graph = new CorrespondenceGraph();
    const live = new LiveMapper(db, graph, { intrinsics: ROTATION_ONLY_CAMERA });
    const ids = db.frameIds();
    for (const id of ids) {
      graph.addImage(id, db.features.get(id)!.count);
      for (const j of ids) {
        if (j >= id) break;
        const g = db.getTwoView(j, id);
        if (g && g.config !== 'degenerate') graph.addTwoView(j, id, g.inliers);
      }
      expect(live.addKeyframe(id).initialized).toBe(false);
    }
    expect(live.reconstruction.numRegisteredImages).toBe(0);
  });

  it('reports unregistered keyframes without correspondences', () => {
    const { db, guess } = smallStream(10, 3);
    // drop every pair: no correspondences at all
    const graph = new CorrespondenceGraph();
    const live = new LiveMapper(db, graph, { intrinsics: guess });
    for (const id of db.frameIds()) {
      graph.addImage(id, db.features.get(id)!.count);
      const r = live.addKeyframe(id);
      expect(r).toEqual({ registered: false, initialized: false, numPoints: 0, reprojError: 0 });
    }
    expect(live.pendingIds().length).toBe(10);
    expect(live.snapshot().points.length).toBe(0);
    expect(() => live.addKeyframe(999)).toThrow(RangeError);
  });
});
