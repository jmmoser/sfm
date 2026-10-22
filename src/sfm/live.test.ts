import { describe, expect, it } from 'vitest';
import { mat3Mul, mat3MulTranspose } from '../math/mat3';
import { Rng } from '../math/random';
import { rotvecToMat3 } from '../math/rotation';
import { CorrespondenceGraph } from './correspondence-graph';
import { LiveMapper, type LiveKeyframeResult, type LiveMapperOptions } from './live';
import { buildSyntheticDatabase, evaluateReconstruction, generateScene, lookPose } from './synthetic';
import { ROTATION_ONLY_CAMERA, rotationOnlyDb } from './e2e-testutil';

const DEG = Math.PI / 180;

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

  it('a keyframe P3P rejects on the map as it is: re-triangulation and P3P again before the gyro-predicted rotation', () => {
    const scene = generateScene({ kind: 'walk', numImages: 30, seed: 6, numPoints: 6000, noisePx: 0.5, descriptors: false });
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    const ids = db.frameIds();
    const K = 14;
    const point = (k: number, f: number): number => scene.featurePoint[k][f];
    const seen = (k: number): Set<number> => new Set(Array.from(scene.featurePoint[k]).filter((p) => p >= 0));
    const s3 = seen(K - 3),
      s2 = seen(K - 2),
      s1 = seen(K - 1);
    // As in the test above, the "new region" S (seen by K − 2, K − 1 and K) stays two-view between
    // K − 2 and K − 1 until K arrives, but for 36 points that K − 3 sees too (map points before K):
    // enough for P3P to be tried before re-triangulating (≥ 30). 9 of K's 36 correspondences with
    // them are false matches (features of K that see no point of K − 1 / K − 2): too few P3P inliers
    // (27 < 30), a share the known rotation accepts (75 %; e2e yard-real at stride 3, the keyframe
    // past the pan: 36 map points, 27 P3P inliers).
    const S = new Set([...seen(K)].filter((p) => s2.has(p) && s1.has(p)));
    const mapped = [...S].filter((p) => s3.has(p)).slice(0, 36);
    expect(mapped).toHaveLength(36);
    const spare: number[] = [];
    scene.featurePoint[K].forEach((p, f) => {
      if (p >= 0 && !s1.has(p) && !s2.has(p)) spare.push(f);
    });
    const falseMatch = new Map(mapped.slice(27).map((p, q) => [p, spare[q]]));
    const isMapped = new Set(mapped);
    const stream = (opts: { retriangulateNeighbours: number }): { live: LiveMapper; atK: LiveKeyframeResult } => {
      const graph = new CorrespondenceGraph();
      // the "gyro": GT relative rotation from the registered keyframe nearest in capture order
      const live = new LiveMapper(db, graph, {
        intrinsics: { ...scene.camera },
        ...opts,
        predictRotation: (id, rec) => {
          let nb = -1;
          for (const r of rec.registeredImageIds()) if (r !== id && (nb < 0 || Math.abs(r - id) < Math.abs(nb - id))) nb = r;
          if (nb < 0) return null;
          const k = ids.indexOf(id),
            kn = ids.indexOf(nb);
          return { R: mat3Mul(mat3MulTranspose(scene.poses[k].R, scene.poses[kn].R), rec.images.get(nb)!.pose.R), sigmaRad: Math.PI / 180 };
        },
      });
      let atK: LiveKeyframeResult | null = null;
      for (const [k, id] of ids.entries()) {
        graph.addImage(id, db.features.get(id)!.count);
        for (const [kj, j] of ids.entries()) {
          if (j >= id) break;
          const g = db.getTwoView(j, id);
          if (!g) continue;
          const out: number[] = [];
          for (let q = 0; q < g.inliers.length; q += 2) {
            const p = point(kj, g.inliers[q]);
            if (k < K - 2 || kj >= K) out.push(g.inliers[q], g.inliers[q + 1]);
            else if (k < K) {
              if (kj >= K - 2 || !S.has(p) || isMapped.has(p)) out.push(g.inliers[q], g.inliers[q + 1]);
            } else if (kj >= K - 2 && S.has(p)) out.push(g.inliers[q], (k === K && falseMatch.get(p)) || g.inliers[q + 1]);
          }
          if (out.length >= 60) graph.addTwoView(j, id, Uint32Array.from(out));
        }
        const r = live.addKeyframe(id);
        if (k === K) atK = r;
      }
      return { live, atK: atK! };
    };
    // re-triangulation off: P3P fails (27 inliers), the known rotation registers K
    const off = stream({ retriangulateNeighbours: 0 });
    const offK = off.atK.registrations.find((a) => a.id === ids[K])!;
    expect(off.atK.registered).toBe(true);
    expect(offK.visible).toBeGreaterThanOrEqual(30); // of the 36 (a point may have been filtered)
    expect(offK.calls).toHaveLength(1);
    expect(offK.calls[0]).toMatchObject({ p3pInliers: 0, via: 'knownRotation' });
    expect(offK.calls[0].knownRotationInliers).toBeGreaterThanOrEqual(offK.visible - 9);
    expect(offK.calls[0].knownRotationInliers).toBeLessThan(30);
    // on: P3P alone first, re-triangulation (S), then P3P registers K; the gyro path is never used
    const { live, atK } = stream({ retriangulateNeighbours: 6 });
    const a = atK.registrations.find((r) => r.id === ids[K])!;
    expect(atK.registered).toBe(true);
    expect(a.visible).toBeGreaterThanOrEqual(30);
    expect(a.visibleRetriangulated).toBeGreaterThan(100);
    expect(a.calls).toHaveLength(2);
    expect(a.calls[0]).toMatchObject({ p3pInliers: 0, via: null });
    expect(a.calls[0].knownRotationInliers).toBeNaN();
    expect(a.calls[1].via).toBe('p3p');
    expect(a.calls[1].p3pInliers).toBeGreaterThan(100);
    expect(live.incrementalMapper.numKnownRotationRegistrations).toBe(0);
    expect(live.pendingIds()).toEqual([]);
    const ev = evaluateReconstruction(live.reconstruction, scene);
    expect(ev.numRegistered).toBe(30);
    expect(ev.maxRotErrorDeg).toBeLessThan(0.5);
  });

  it('a walk with a 35° pan between keyframes: what P3P leaves pending registers after re-triangulation, with the gyro-predicted rotation when a pair checks its baseline', () => {
    // Walk east through the yard looking north (yaw 90°); keyframe K − 2 is panned 35° (to 55°),
    // K − 1 and K turn further (to 38–38.5°, then 26°) and the walk goes on at 26°. K looks at a
    // region only K − 2 and K − 1 saw (two-view tracks; 0.5 m apart, houses 20–40 m away) plus a
    // sliver of the map — e2e yard-real at stride 3, the keyframes after its 33.7° pan.
    const N = 26,
      K = 12;
    const walk = (yawK1: number) => {
      const poses = [];
      for (let k = 0; k < N; k++) {
        const yaw = k <= K - 3 ? 90 : k === K - 2 ? 55 : k === K - 1 ? yawK1 : 26;
        poses.push(lookPose([-10 + 0.5 * k, -6 + 0.6 * Math.sin(k / 3), 1.6 + 0.1 * Math.cos(k / 2)], yaw * DEG, -0.08));
      }
      const scene = generateScene({ kind: 'walk', poses, seed: 6, numPoints: 9000, noisePx: 0.5, descriptors: false });
      return { scene, db: buildSyntheticDatabase(scene, { verify: false }).db };
    };
    /** `gyro`: GT relative rotation from the registered keyframe nearest in capture order, turned `errDeg` about a seeded random axis. */
    const stream = ({ scene, db }: ReturnType<typeof walk>, retriangulateNeighbours: number, gyro: { errDeg: number; seed: number } | null, opts: Partial<LiveMapperOptions> = {}) => {
      const ids = db.frameIds();
      const graph = new CorrespondenceGraph();
      const rng = new Rng(gyro?.seed ?? 0);
      const predictRotation = (id: number, rec: LiveMapper['reconstruction']) => {
        let nb = -1;
        for (const r of rec.registeredImageIds()) if (r !== id && (nb < 0 || Math.abs(r - id) < Math.abs(nb - id))) nb = r;
        if (nb < 0) return null;
        const axis = Float64Array.of(rng.gaussian(), rng.gaussian(), rng.gaussian());
        const e = (gyro!.errDeg * DEG) / Math.hypot(axis[0], axis[1], axis[2]);
        const G = mat3MulTranspose(scene.poses[ids.indexOf(id)].R, scene.poses[ids.indexOf(nb)].R);
        return { R: mat3Mul(mat3Mul(rotvecToMat3(axis.map((v) => v * e)), G), rec.images.get(nb)!.pose.R), sigmaRad: DEG };
      };
      const live = new LiveMapper(db, graph, { intrinsics: { ...scene.camera }, retriangulateNeighbours, ...(gyro ? { predictRotation } : {}), ...opts });
      let atK: LiveKeyframeResult | null = null;
      let kCentreErr = NaN;
      for (const [k, id] of ids.entries()) {
        graph.addImage(id, db.features.get(id)!.count);
        for (const j of ids) {
          if (j >= id) break;
          const g = db.getTwoView(j, id);
          if (g) graph.addTwoView(j, id, g.inliers);
        }
        const r = live.addKeyframe(id);
        if (k === K) {
          atK = r;
          // K's centre right after its registration, in the GT frame of the keyframes registered so far
          const ev = evaluateReconstruction(live.reconstruction, scene);
          if (r.registered && ev.sim) {
            const c = live.reconstruction.imageCenter(id),
              { s, R, t } = ev.sim;
            const g = scene.poses[k];
            const gc = [0, 1, 2].map((q) => -(g.R[q] * g.t[0] + g.R[3 + q] * g.t[1] + g.R[6 + q] * g.t[2]));
            kCentreErr = Math.hypot(...[0, 1, 2].map((q) => s * (R[3 * q] * c[0] + R[3 * q + 1] * c[1] + R[3 * q + 2] * c[2]) + t[q] - gc[q]));
          }
        }
      }
      const a = atK!.registrations.find((x) => x.id === ids[K])!;
      return { live, a, ids, kCentreErr, ev: evaluateReconstruction(live.reconstruction, scene) };
    };
    // K − 1 at 38.5°: K sees 20 map points, 27 after re-triangulating K − 2 / K − 1 (small angles)
    const w = walk(38.5);
    const ids = w.db.frameIds();
    const tri = (b: number): number => w.db.getTwoView(ids[b], ids[K])!.triAngle! / DEG;
    expect(tri(K - 2)).toBeLessThan(1.5); // the pairs that see K's region: low parallax
    expect(tri(K - 1)).toBeLessThan(1.5);
    for (const retri of [0, 6]) {
      const r = stream(w, retri, null); // P3P: 20 points, 27 after re-triangulation (< 30)
      expect(r.a.visible).toBeLessThan(30);
      expect(r.a.registered).toBe(false);
      expect(r.live.pendingIds()).toEqual(ids.slice(K)); // lost for good
    }
    // the gyro path at the default pair threshold (1.5°): no pair checks K's baseline (median
    // triangulation angles 1.26° / 0.57°), so K is refused before the solver runs (fails closed) …
    const closed = stream(w, 0, { errDeg: 0, seed: 1 });
    expect(closed.a.registered).toBe(false);
    expect(closed.a.calls.at(-1)!.knownRotationRefused).toBe('noPair');
    expect(closed.live.pendingIds()).toEqual(ids.slice(K)); // without re-triangulation: lost
    // … and with re-triangulation K + 1 registers with the gyro rotation instead — its pair with
    // K − 2 (1.8°) checks its baseline — and K by P3P after it
    const next = stream(w, 6, { errDeg: 1, seed: 6 });
    expect(next.a.registered).toBe(false);
    expect(next.a.calls.at(-1)!.knownRotationRefused).toBe('noPair');
    expect([...next.live.incrementalMapper.knownRotationImages]).toEqual([ids[K + 1]]);
    expect(next.live.pendingIds()).toEqual([]);
    expect(next.ev.numRegistered).toBe(N);
    expect(next.ev.ateRelative).toBeLessThan(0.002);
    expect(next.ev.maxRotErrorDeg).toBeLessThan(0.5);
    // with those pairs admitted (their directions are exact here): re-triangulation off, 20 inliers
    // all in the sliver (inlier spread < 5 % of the image): refused
    const lowTri: Partial<LiveMapperOptions> = { knownRotationPairMinTriAngleDeg: 0.5 };
    const sliver = stream(w, 0, { errDeg: 0, seed: 1 }, lowTri);
    expect(sliver.a.registered).toBe(false);
    expect(sliver.a.calls.at(-1)!.knownRotationRefused).toBe('spread');
    // … and both together register K with the gyro-predicted rotation, then the rest of the walk —
    // also with the gyro 1° off the truth
    for (const gyro of [
      { errDeg: 0, seed: 1 },
      { errDeg: 1, seed: 2 },
      { errDeg: 1, seed: 3 },
      { errDeg: 1, seed: 4 },
    ]) {
      const both = stream(w, 6, gyro, lowTri);
      expect(both.a.visibleRetriangulated).toBeGreaterThanOrEqual(15);
      expect(both.a.visibleRetriangulated).toBeLessThan(30);
      expect(both.a.registered).toBe(true);
      expect(both.a.calls.at(-1)).toMatchObject({ via: 'knownRotation', knownRotationRefused: null });
      expect(both.a.calls.at(-1)!.pairBaselineDeg).toBeLessThan(30);
      expect(both.kCentreErr).toBeLessThan(0.05); // m, of a 0.5 m step
      expect(both.live.pendingIds()).toEqual([]);
      expect(both.ev.numRegistered).toBe(N);
      expect(both.ev.ateRelative).toBeLessThan(0.002);
      expect(both.ev.maxRotErrorDeg).toBeLessThan(0.5);
    }
    // K − 1 at 38°: re-triangulation alone gives P3P its 30 points — the gyro is not needed (nor used)
    const w2 = walk(38);
    expect(stream(w2, 0, null).live.pendingIds()).toEqual(w2.db.frameIds().slice(K));
    const p3p = stream(w2, 6, { errDeg: 0, seed: 1 }, lowTri);
    expect(p3p.a.registered).toBe(true);
    expect(p3p.a.calls.at(-1)!.via).toBe('p3p');
    expect(p3p.live.incrementalMapper.numKnownRotationRegistrations).toBe(0);
    expect(p3p.ev.numRegistered).toBe(N);
    expect(p3p.ev.maxRotErrorDeg).toBeLessThan(0.5);
  });

  it('a P3P pose the gyro check rejected in an earlier call of the attempt counts as recovered once the attempt registers', () => {
    const scene = generateScene({ kind: 'walk', numImages: 20, seed: 6, numPoints: 6000, noisePx: 0.5, descriptors: false });
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    const ids = db.frameIds();
    const K = 10;
    const run = (rejectFirst: number) => {
      const graph = new CorrespondenceGraph();
      let checksOfK = 0;
      const recovered: number[] = [];
      const live = new LiveMapper(db, graph, {
        intrinsics: { ...scene.camera },
        // the "gyro check" rejects the first `rejectFirst` poses of keyframe K
        rotationCheck: { pose: (id) => id !== ids[K] || ++checksOfK > rejectFirst, poseRecovered: (id) => recovered.push(id) },
        predictRotation: (id, rec) => {
          let nb = -1;
          for (const r of rec.registeredImageIds()) if (r !== id && (nb < 0 || Math.abs(r - id) < Math.abs(nb - id))) nb = r;
          if (nb < 0) return null;
          const G = mat3MulTranspose(scene.poses[ids.indexOf(id)].R, scene.poses[ids.indexOf(nb)].R);
          return { R: mat3Mul(G, rec.images.get(nb)!.pose.R), sigmaRad: DEG };
        },
      });
      let atK: LiveKeyframeResult | null = null;
      for (const [k, id] of ids.entries()) {
        graph.addImage(id, db.features.get(id)!.count);
        for (const j of ids) {
          if (j >= id) break;
          const g = db.getTwoView(j, id);
          if (g) graph.addTwoView(j, id, g.inliers);
        }
        const r = live.addKeyframe(id);
        if (k === K) atK = r;
      }
      return { live, a: atK!.registrations.find((x) => x.id === ids[K])!, recovered };
    };
    // first P3P pose rejected; after re-triangulating the neighbours P3P registers K (a second call)
    const once = run(1);
    expect(once.a.registered).toBe(true);
    expect(once.a.calls.map((c) => [c.via, c.p3pRotationRejected])).toEqual([
      [null, true],
      ['p3p', false],
    ]);
    // both P3P poses rejected (before and after re-triangulation), the known rotation registers K
    const twice = run(2);
    expect(twice.a.registered).toBe(true);
    expect(twice.a.calls.map((c) => [c.via, c.p3pRotationRejected])).toEqual([
      [null, true],
      ['knownRotation', true],
    ]);
    // every rejected P3P pose of K is reported recovered, once: rejected − recovered = poses of
    // keyframes that stayed unregistered (0 here)
    for (const [r, n] of [
      [once, 1],
      [twice, 2],
    ] as const) {
      expect(r.live.incrementalMapper.numRotationRejects).toBe(n);
      expect(r.live.incrementalMapper.numRotationRejectsRecovered).toBe(n);
      expect(r.recovered).toEqual(Array(n).fill(ids[K]));
      expect(r.live.pendingIds()).toEqual([]);
    }
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
      expect(r).toEqual({ registered: false, initialized: false, numPoints: 0, reprojError: 0, registrations: [] });
    }
    expect(live.pendingIds().length).toBe(10);
    expect(live.snapshot().points.length).toBe(0);
    expect(() => live.addKeyframe(999)).toThrow(RangeError);
  });
});
