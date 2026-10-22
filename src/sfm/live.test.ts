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

  it('re-triangulates the registered neighbours of a keyframe whose correspondences turned their two-view tracks multi-view', () => {
    const scene = generateScene({ kind: 'walk', numImages: 30, seed: 6, numPoints: 6000, noisePx: 0.5, descriptors: false });
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    const ids = db.frameIds();
    const K = 14;
    const point = (id: number, f: number): number => scene.featurePoint[ids.indexOf(id)][f];
    // "New region" S: the points keyframes K − 2, K − 1 and K all see. After a fast pan only those
    // keyframes overlap there: S is matched between K − 2 and K − 1 only until K arrives (two-view
    // tracks, which the triangulator does not create points from), K sees just 10 other map points,
    // and the keyframes after K overlap K and later ones (and S in K − 2 / K − 1).
    const seen = (k: number): Set<number> => new Set(Array.from(scene.featurePoint[k]).filter((p) => p >= 0));
    const s2 = seen(K - 2),
      s1 = seen(K - 1);
    const S = new Set([...seen(K)].filter((p) => s2.has(p) && s1.has(p)));
    expect(S.size).toBeGreaterThan(200);
    const filter = (j: number, id: number, inl: Uint32Array): Uint32Array | null => {
      const near = j === ids[K - 2] || j === ids[K - 1];
      if (id >= ids[K] && j < ids[K - 2]) return null;
      const out: number[] = [];
      let others = 0;
      for (let k = 0; k < inl.length; k += 2) {
        const inS = S.has(point(j, inl[k]));
        if (id >= ids[K - 2] && j < ids[K - 2] && inS) continue;
        if (id >= ids[K] && near && !inS && (id > ids[K] || others++ >= 10)) continue;
        out.push(inl[k], inl[k + 1]);
      }
      return out.length >= 30 ? Uint32Array.from(out) : null;
    };
    const stream = (retriangulateNeighbours: number): { live: LiveMapper; atK: { registered: boolean; points: number } } => {
      const graph = new CorrespondenceGraph();
      const live = new LiveMapper(db, graph, { intrinsics: { ...scene.camera }, retriangulateNeighbours });
      let atK = { registered: false, points: 0 };
      for (const id of ids) {
        graph.addImage(id, db.features.get(id)!.count);
        for (const j of ids) {
          if (j >= id) break;
          const g = db.getTwoView(j, id);
          const inl = g ? filter(j, id, g.inliers) : null;
          if (inl) graph.addTwoView(j, id, inl);
        }
        const r = live.addKeyframe(id);
        if (id === ids[K]) atK = { registered: r.registered, points: live.reconstruction.images.get(id)!.numVisiblePoints3D };
      }
      return { live, atK };
    };
    // as before: K sees 10 map points and the rest of the walk is lost
    const off = stream(0);
    expect(off.atK.registered).toBe(false);
    expect(off.atK.points).toBeLessThanOrEqual(10);
    expect(off.live.pendingIds()).toEqual(ids.slice(K));
    // re-triangulated: K − 2 / K − 1 give S, K registers on it, and so does the rest
    const { live: on, atK } = stream(6);
    expect(atK.registered).toBe(true);
    expect(atK.points).toBeGreaterThan(100);
    expect(on.pendingIds()).toEqual([]);
    const ev = evaluateReconstruction(on.reconstruction, scene);
    expect(ev.numRegistered).toBe(30);
    expect(ev.ateRelative).toBeLessThan(0.005);
    expect(ev.maxRotErrorDeg).toBeLessThan(0.5);
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
