import { describe, expect, it } from 'vitest';
import type { Pose } from '../core/types';
import { Rng } from '../math/random';
import { rotvecToMat3 } from '../math/rotation';
import { buildCorrespondenceGraph } from './mapper';
import { Reconstruction } from './reconstruction';
import { buildSyntheticDatabase, generateScene, type SyntheticScene } from './synthetic';
import { DEFAULT_TRIANGULATOR_OPTIONS, IncrementalTriangulator, RayTriangulator, angularError } from './triangulator';

const DEG = Math.PI / 180;

/** Rays from n random cameras (around the origin, ~5 units away) to X; returns C, D, Z arrays. */
function rays(n: number, X: number[], rng: Rng, noiseRad = 0, spread = 1) {
  const C = new Float64Array(3 * n),
    D = new Float64Array(3 * n),
    Z = new Float64Array(3 * n);
  for (let k = 0; k < n; k++) {
    const cx = spread * (rng.next() * 4 - 2),
      cy = spread * (rng.next() * 4 - 2),
      cz = -5 + rng.next();
    C.set([cx, cy, cz], 3 * k);
    let dx = X[0] - cx + noiseRad * 5 * rng.gaussian(),
      dy = X[1] - cy + noiseRad * 5 * rng.gaussian(),
      dz = X[2] - cz;
    const n2 = Math.hypot(dx, dy, dz);
    dx /= n2;
    dy /= n2;
    dz /= n2;
    D.set([dx, dy, dz], 3 * k);
    Z.set([dx, dy, dz], 3 * k); // optical axis along the ray (depth > 0 in front)
  }
  return { C, D, Z };
}

describe('RayTriangulator', () => {
  it('recovers the exact point from noise-free rays (2 views and many views)', () => {
    const rng = new Rng(1);
    const tri = new RayTriangulator(new Rng(2));
    for (const n of [2, 3, 7, 15, 16, 40]) {
      const X = [rng.gaussian(), rng.gaussian(), 2 + rng.next()];
      const { C, D, Z } = rays(n, X, rng);
      const cnt = tri.estimate(C, D, Z, n, 1.5 * DEG, 2 * DEG);
      expect(cnt).toBe(n);
      for (let k = 0; k < 3; k++) expect(tri.X[k]).toBeCloseTo(X[k], 9);
    }
  });

  it('rejects outlier rays (30 %) and keeps the inliers', () => {
    const rng = new Rng(3);
    const tri = new RayTriangulator(new Rng(4));
    for (const n of [6, 12, 30]) {
      const X = [0.3, -0.2, 2.5];
      const { C, D, Z } = rays(n, X, rng, 0.002);
      const out = new Set<number>();
      while (out.size < Math.floor(0.3 * n)) out.add(rng.int(n));
      for (const k of out) {
        // random direction in front
        const dx = rng.gaussian() * 0.5,
          dy = rng.gaussian() * 0.5;
        const nn = Math.hypot(dx, dy, 1);
        D.set([dx / nn, dy / nn, 1 / nn], 3 * k);
      }
      const cnt = tri.estimate(C, D, Z, n, 1.5 * DEG, 2 * DEG);
      expect(cnt).toBe(n - out.size);
      for (let k = 0; k < n; k++) expect(tri.inliers[k]).toBe(out.has(k) ? 0 : 1);
      expect(Math.hypot(tri.X[0] - X[0], tri.X[1] - X[1], tri.X[2] - X[2])).toBeLessThan(0.05);
    }
  });

  it('enforces the minimum triangulation angle and cheirality', () => {
    const rng = new Rng(5);
    const tri = new RayTriangulator();
    // far point, tiny baseline: angle ≈ 0.02 rad·… below 1.5°
    const { C, D, Z } = rays(4, [0, 0, 200], rng, 0, 0.5);
    expect(tri.estimate(C, D, Z, 4, 1.5 * DEG, 2 * DEG)).toBe(0);
    // rays pointing away from each other's intersection: point behind the cameras
    const C2 = Float64Array.of(-1, 0, 0, 1, 0, 0);
    const D2 = Float64Array.of(-0.6, 0, 0.8, 0.6, 0, 0.8); // diverging
    const Z2 = Float64Array.of(0, 0, 1, 0, 0, 1);
    expect(tri.estimate(C2, D2, Z2, 2, 1.5 * DEG, 2 * DEG)).toBe(0);
    expect(tri.estimate(C2, D2, Z2, 1, 1.5 * DEG, 2 * DEG)).toBe(0);
  });

  it('angularError is zero for an exact observation and grows with the offset', () => {
    const R = rotvecToMat3(Float64Array.of(0.1, 0.2, -0.3));
    const pose: Pose = { R, t: Float64Array.of(0.5, -0.2, 1) };
    const X = [0.4, 0.3, 4];
    const p = [0, 1, 2].map((r) => R[3 * r] * X[0] + R[3 * r + 1] * X[1] + R[3 * r + 2] * X[2] + pose.t[r]);
    expect(angularError(pose, p[0] / p[2], p[1] / p[2], X)).toBeLessThan(1e-12);
    const x = p[0] / p[2],
      y = p[1] / p[2];
    const e = angularError(pose, x + 0.01, y, X);
    const exact = Math.acos(((x + 0.01) * x + y * y + 1) / (Math.hypot(x + 0.01, y, 1) * Math.hypot(x, y, 1)));
    expect(e).toBeCloseTo(exact, 9);
    expect(angularError(pose, -p[0] / p[2], -p[1] / p[2], [2 * pose.t[0], 0, -100])).toBeGreaterThan(Math.PI / 2);
  });
});

/** Small orbit scene; reconstruction with GT camera and GT poses for the first `nReg` images. */
function setup(nImages: number, nReg: number, seed = 1) {
  const scene = generateScene({
    kind: 'orbit',
    numImages: nImages,
    seed,
    numPoints: 1200,
    noisePx: 0.3,
    outlierRatio: 0,
    clutterPerImage: 0,
    descriptors: false,
    pairWindow: 4,
    extraPairsPerImage: 0,
    k1: 0.02,
  });
  const { db } = buildSyntheticDatabase(scene, { verify: false });
  const graph = buildCorrespondenceGraph(db, 15);
  const rec = new Reconstruction();
  const cid = rec.addCamera(scene.camera);
  scene.imageIds.forEach((id, k) => rec.addImage(id, cid, scene.features[k].count, scene.features[k].keypoints));
  rec.attachGraph(graph);
  for (let k = 0; k < nReg; k++) rec.registerImage(scene.imageIds[k], scene.poses[k]);
  const tri = new IncrementalTriangulator(graph, rec, db, new Rng(9));
  return { scene, db, graph, rec, tri };
}

function pointAccuracy(rec: Reconstruction, scene: SyntheticScene): { maxErr: number; impure: number; n: number } {
  let maxErr = 0,
    impure = 0;
  for (const p of rec.points.values()) {
    const gp = scene.featurePoint[p.track[0].image][p.track[0].feature];
    for (const el of p.track) if (scene.featurePoint[el.image][el.feature] !== gp) impure++;
    const G = scene.points.subarray(3 * gp, 3 * gp + 3);
    maxErr = Math.max(maxErr, Math.hypot(p.xyz[0] - G[0], p.xyz[1] - G[1], p.xyz[2] - G[2]));
  }
  return { maxErr, impure, n: rec.points.size };
}

describe('IncrementalTriangulator', () => {
  const o = DEFAULT_TRIANGULATOR_OPTIONS;

  it('triangulates registered images: creates then continues tracks, accurately', () => {
    const { scene, rec, tri } = setup(10, 6);
    const n0 = tri.triangulateImage(o, 0);
    expect(n0).toBeGreaterThan(0); // creates points with already registered images 1..5
    const pts0 = rec.numPoints;
    let continued = 0;
    for (let k = 1; k < 6; k++) {
      const before = rec.numObservations;
      tri.triangulateImage(o, k);
      continued += rec.numObservations - before;
    }
    expect(continued).toBeGreaterThan(0);
    const acc = pointAccuracy(rec, scene);
    expect(acc.impure).toBe(0);
    // object ~0.5 m away, 0.3 px noise: mm-level points
    expect(acc.maxErr).toBeLessThan(0.02);
    expect(acc.n).toBeGreaterThanOrEqual(pts0);
    // tracks only contain registered images, one observation per image
    for (const p of rec.points.values()) {
      const imgs = new Set(p.track.map((e) => e.image));
      expect(imgs.size).toBe(p.track.length);
      for (const el of p.track) expect(rec.images.get(el.image)!.registered).toBe(true);
    }
    expect(tri.getModifiedPoints().length).toBeGreaterThan(0);
    // unregistered images are never triangulated
    expect(tri.triangulateImage(o, 8)).toBe(0);
  });

  it('completes tracks through the graph (transitively) and merges split tracks', () => {
    const { scene, rec, tri, graph } = setup(10, 3, 2);
    for (let k = 0; k < 3; k++) tri.triangulateImage(o, k);
    // register more images WITHOUT triangulating them: completion must find their observations
    for (let k = 3; k < 7; k++) rec.registerImage(scene.imageIds[k], scene.poses[k]);
    const before = rec.numObservations;
    const added = tri.completeAllTracks(o);
    expect(added).toBeGreaterThan(100);
    expect(rec.numObservations - before).toBe(added);
    // image 6 is not directly matched to images 0..2 (pair window 4 ⇒ 2–6 is), but transitively
    expect(rec.images.get(6)!.numPoints3D).toBeGreaterThan(50);
    expect(pointAccuracy(rec, scene).impure).toBe(0);
    // split a track into two points and merge them back
    let target = -1;
    for (const p of rec.points.values())
      if (p.track.length >= 4) {
        target = p.id;
        break;
      }
    const p = rec.points.get(target)!;
    const tr = p.track.slice();
    const xyz = Float64Array.from(p.xyz);
    rec.deletePoint(target);
    const a = rec.addPoint(xyz, tr.slice(0, 2));
    const b = rec.addPoint([xyz[0] + 1e-4, xyz[1], xyz[2]], tr.slice(2));
    const m = tri.mergeAllTracks(o);
    expect(m).toBe(tr.length);
    expect(rec.points.has(a)).toBe(false);
    expect(rec.points.has(b)).toBe(false);
    const merged = rec.images.get(tr[0].image)!.point3D[tr[0].feature];
    expect(rec.points.get(merged)!.track.length).toBe(tr.length);
    // a wrong split point (5 cm off) is not merged
    const p2 = rec.points.get(merged)!;
    const tr2 = p2.track.slice();
    rec.deletePoint(merged);
    rec.addPoint(xyz, tr2.slice(0, 2));
    const bad = rec.addPoint([xyz[0] + 0.05, xyz[1], xyz[2]], tr2.slice(2));
    tri.mergeAllTracks(o);
    expect(rec.points.has(bad)).toBe(true);
    void graph;
  });

  it('retriangulates under-reconstructed pairs', () => {
    const { rec, tri } = setup(8, 4, 3);
    // only a two-view model of images 0, 1 exists; 2, 3 registered without points
    tri.triangulateImage(o, 0);
    const pairsBefore = rec.numSharedPoints(2, 3);
    expect(pairsBefore).toBeGreaterThan(0); // created with registered 2, 3 as well
    // wipe all points, then retriangulate from scratch
    for (const id of [...rec.points.keys()]) rec.deletePoint(id);
    const n = tri.retriangulate(o);
    expect(n).toBeGreaterThan(200);
    expect(rec.numSharedPoints(2, 3)).toBeGreaterThan(50);
    // pairs that were well reconstructed when their turn came were not tried yet; each
    // pair is tried at most once (reMaxTrials = 1), so wiping + retriangulating ends at 0
    let calls = 0;
    for (;;) {
      for (const id of [...rec.points.keys()]) rec.deletePoint(id);
      calls++;
      if (tri.retriangulate(o) === 0) break;
      expect(calls).toBeLessThan(12);
    }
    expect(tri.retriangulate({ ...o, reMaxTrials: 2 })).toBeGreaterThan(200);
  });

  it('ignores two-view tracks when asked and skips bogus cameras', () => {
    const { rec, tri, graph } = setup(6, 2, 4);
    // with only two registered images, a feature with exactly one correspondence is a two-view observation
    const n1 = tri.triangulateImage({ ...o, ignoreTwoViewTracks: true }, 0);
    let twoViewOnly = 0;
    for (const p of rec.points.values()) if (graph.isTwoViewObservation(p.track[0].image, p.track[0].feature)) twoViewOnly++;
    expect(twoViewOnly).toBe(0);
    for (const id of [...rec.points.keys()]) rec.deletePoint(id);
    const n2 = tri.triangulateImage({ ...o, ignoreTwoViewTracks: false }, 0);
    expect(n2).toBeGreaterThan(n1);
    for (const id of [...rec.points.keys()]) rec.deletePoint(id);
    rec.cameras.get(0)!.f = 1e6;
    expect(tri.triangulateImage(o, 0)).toBe(0);
  });
});
