/**
 * Adversarial tests of the SfM core (review pass): id validation, geometric edge
 * cases, bookkeeping invariants under random edits, deterministic time-sliced global
 * BA, forward-motion initialisation, extreme similarity transforms during live mapping.
 */
import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, Pose, TwoViewGeometry } from '../core/types';
import { projectWorld } from '../geometry/camera';
import { Rng } from '../math/random';
import { rotvecToMat3 } from '../math/rotation';
import { CorrespondenceGraph } from './correspondence-graph';
import { reconstructionInvariantErrors } from './core-testutil';
import { FeatureDatabase, MAX_IMAGE_ID, invertTwoView, pairKey } from './database';
import { LiveMapper } from './live';
import { IncrementalMapper, SfMCancelledError, buildCorrespondenceGraph, runIncrementalSfM } from './mapper';
import { Reconstruction, type Track } from './reconstruction';
import { buildSyntheticDatabase, evaluateReconstruction, generateScene, type SyntheticScene } from './synthetic';
import { RayTriangulator, angularError } from './triangulator';

const DEG = Math.PI / 180;

function poseAt(C: ArrayLike<number>, R: Float64Array): Pose {
  return {
    R,
    t: Float64Array.of(
      -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
      -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
      -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
    ),
  };
}

function frameInfo(id: number) {
  return {
    id,
    t: id,
    width: 640,
    height: 480,
    sharpness: 1,
    R_cam_device: Float64Array.of(1, 0, 0, 0, -1, 0, 0, 0, -1),
    gravityCam: null,
    heading: null,
    geo: null,
  };
}

function featureSet(n: number) {
  return { count: n, keypoints: new Float32Array(4 * n), descriptors: new Uint8Array(128 * n), scores: new Float32Array(n), width: 640, height: 480 };
}

/** Streams a scene into a LiveMapper like the engine (pairs with earlier keyframes first). */
function stream(db: FeatureDatabase, live: LiveMapper, graph: CorrespondenceGraph, ids: number[], onKeyframe?: (id: number) => void): void {
  const all = db.frameIds();
  for (const id of ids) {
    if (!graph.hasImage(id)) graph.addImage(id, db.features.get(id)!.count);
    for (const j of all) {
      if (j >= id) break;
      if (!graph.hasImage(j)) continue;
      const g = db.getTwoView(j, id);
      if (g && g.config !== 'degenerate' && g.inliers.length >= 30) graph.addTwoView(j, id, g.inliers);
    }
    live.addKeyframe(id);
    onKeyframe?.(id);
  }
}

describe('FeatureDatabase / CorrespondenceGraph: id validation', () => {
  it('ids ≥ 2^20 are rejected everywhere (pairKey would alias other pairs)', () => {
    // the aliasing the validation prevents:
    expect(pairKey(0, MAX_IMAGE_ID + 5)).toBe(pairKey(1, 5));
    const db = new FeatureDatabase();
    const g: TwoViewGeometry = { config: 'calibrated', inliers: Uint32Array.of(0, 0) };
    expect(() => db.setTwoView(0, MAX_IMAGE_ID + 5, g)).toThrow(RangeError);
    expect(() => db.setTwoView(-1, 3, g)).toThrow(RangeError);
    expect(() => db.setTwoView(1.5, 3, g)).toThrow(RangeError);
    expect(() => db.addFrame(frameInfo(MAX_IMAGE_ID), featureSet(1))).toThrow(RangeError);
    const graph = new CorrespondenceGraph();
    expect(() => graph.addImage(MAX_IMAGE_ID + 5, 3)).toThrow(RangeError);
    expect(() => graph.addImage(-2, 3)).toThrow(RangeError);
    expect(() => graph.addImage(NaN, 3)).toThrow(RangeError);
    expect(() => graph.addImage(4, -1)).toThrow(RangeError);
    expect(() => graph.addImage(4, 2.5)).toThrow(RangeError);
    graph.addImage(MAX_IMAGE_ID - 1, 3); // largest valid id
    graph.addImage(0, 3);
    graph.addTwoView(MAX_IMAGE_ID - 1, 0, Uint32Array.of(1, 2));
    expect(graph.matchesBetween(0, MAX_IMAGE_ID - 1)).toEqual(Uint32Array.of(2, 1));
    const rec = new Reconstruction();
    const cid = rec.addCamera({ width: 640, height: 480, f: 500, cx: 319.5, cy: 239.5, k1: 0, k2: 0 });
    expect(() => rec.addImage(MAX_IMAGE_ID, cid, 3)).toThrow(RangeError);
    expect(() => rec.addImage(-1, cid, 3)).toThrow(RangeError);
  });

  it('invertTwoView is an involution and keeps E, F, H and (R, t) consistent; a singular H is dropped', () => {
    const rng = new Rng(5);
    for (let trial = 0; trial < 20; trial++) {
      // rotations up to (and including) ≈ π
      const ang = trial === 0 ? Math.PI - 1e-9 : rng.next() * Math.PI;
      const ax = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
      const an = Math.hypot(ax[0], ax[1], ax[2]);
      const R = rotvecToMat3(ax.map((v) => (v / an) * ang));
      const t0 = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
      const tn = Math.hypot(t0[0], t0[1], t0[2]);
      const t = Float64Array.from(t0.map((v) => v / tn));
      // E = [t]x R  (x2ᵀ E x1 = 0 for x2 = R x1 + t)
      const tx = [0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0];
      const E = new Float64Array(9);
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) E[3 * r + c] = tx[3 * r] * R[c] + tx[3 * r + 1] * R[3 + c] + tx[3 * r + 2] * R[6 + c];
      const H = Float64Array.of(1, 0.1, 2, 0.05, 1.2, -1, 0.001, 0.002, 1);
      const g: TwoViewGeometry = { config: 'calibrated', inliers: Uint32Array.of(3, 7, 4, 9), E, F: E, H, R, t, triAngle: 0.1 };
      const inv = invertTwoView(g);
      for (let k = 0; k < 5; k++) {
        const x1 = [rng.gaussian(), rng.gaussian(), 3 + rng.next()];
        const x2 = [0, 1, 2].map((r) => R[3 * r] * x1[0] + R[3 * r + 1] * x1[1] + R[3 * r + 2] * x1[2] + t[r]);
        // inverse pose maps x2 back to x1
        for (let r = 0; r < 3; r++) {
          const back = inv.R![3 * r] * x2[0] + inv.R![3 * r + 1] * x2[1] + inv.R![3 * r + 2] * x2[2] + inv.t![r];
          expect(back).toBeCloseTo(x1[r], 10);
        }
        // epipolar constraint of the inverse: x1ᵀ E' x2 = 0
        let s = 0;
        for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) s += x1[r] * inv.E![3 * r + c] * x2[c];
        expect(Math.abs(s)).toBeLessThan(1e-9);
      }
      const back = invertTwoView(inv);
      for (let k = 0; k < 9; k++) {
        expect(back.R![k]).toBeCloseTo(R[k], 12);
        expect(back.E![k]).toBeCloseTo(E[k], 12);
        expect(back.H![k]).toBeCloseTo(H[k], 10);
      }
      for (let k = 0; k < 3; k++) expect(back.t![k]).toBeCloseTo(t[k], 12);
      expect(Array.from(back.inliers)).toEqual([3, 7, 4, 9]);
    }
    const singular = invertTwoView({ config: 'planar', inliers: new Uint32Array(0), H: Float64Array.of(1, 2, 3, 2, 4, 6, 0, 0, 1) });
    expect(singular.H).toBeUndefined();
  });
});

describe('synthetic scenes: loop / forward kinds', () => {
  const centre = (p: Pose): number[] => [0, 1, 2].map((c) => -(p.R[c] * p.t[0] + p.R[3 + c] * p.t[1] + p.R[6 + c] * p.t[2]));

  it('loop closes on itself (last keyframes matched to the first), forward walks along the optical axis', () => {
    const loop = generateScene({ kind: 'loop', numImages: 60, seed: 2, descriptors: false });
    const c0 = centre(loop.poses[0]),
      cN = centre(loop.poses[59]);
    expect(Math.hypot(c0[0] - cN[0], c0[1] - cN[1], c0[2] - cN[2])).toBeLessThan(3); // one step (~1.8 m) apart
    expect(loop.pairs.some((p) => p.a < 3 && p.b > 56)).toBe(true);
    for (const fp of loop.featurePoint) expect(fp.filter((q) => q >= 0).length).toBeGreaterThan(150);
    const fwd = generateScene({ kind: 'forward', numImages: 20, seed: 2, descriptors: false });
    for (let i = 1; i < 20; i++) {
      const a = centre(fwd.poses[i - 1]),
        b = centre(fwd.poses[i]);
      const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const R = fwd.poses[i].R; // optical axis = third row of R
      const cos = (R[6] * d[0] + R[7] * d[1] + R[8] * d[2]) / Math.hypot(d[0], d[1], d[2]);
      expect(cos).toBeGreaterThan(0.95);
    }
    expect(() => generateScene({ kind: 'spiral' as never })).toThrow(RangeError);
    // existing kinds keep their random streams (other tests depend on them)
    const w = generateScene({ kind: 'walk', numImages: 3, seed: 1, descriptors: false });
    expect(w.features[0].count).toBe(generateScene({ kind: 'walk', numImages: 3, seed: 1, descriptors: false }).features[0].count);
  });
});

describe('CorrespondenceGraph: adversarial inputs', () => {
  it('duplicate / many-to-one matches, empty replacement and capped output agree with a brute-force model', () => {
    const g = new CorrespondenceGraph();
    for (let i = 0; i < 3; i++) g.addImage(i, 6);
    // feature 0 of image 0 matched twice to the same feature and to two features of image 1
    g.addTwoView(0, 1, Uint32Array.of(0, 0, 0, 0, 0, 3, 1, 1));
    g.addTwoView(2, 0, Uint32Array.of(5, 0, 4, 0));
    expect(g.numCorrespondencesOf(0, 0)).toBe(5);
    expect(g.numCorrespondences(0)).toBe(6);
    expect(g.numObservations(0)).toBe(2);
    const outI = new Int32Array(2),
      outF = new Int32Array(2);
    expect(g.getCorrespondences(0, 0, outI, outF)).toBe(5); // total, though only 2 written
    expect(g.numCorrespondencesBetween(0, 1)).toBe(4);
    // replacing with an empty set removes the pair entirely (and its correspondences)
    g.addTwoView(1, 0, new Uint32Array(0));
    expect(g.numCorrespondencesBetween(0, 1)).toBe(0);
    expect(g.numPairs).toBe(1);
    expect(g.numCorrespondencesOf(0, 0)).toBe(2);
    expect(g.numCorrespondences(1)).toBe(0);
    expect(g.numObservations(1)).toBe(0);
    expect(g.neighbours(1).size).toBe(0);
    expect(g.correspondences(0, 0).map((c) => `${c.image}:${c.feature}`).sort()).toEqual(['2:4', '2:5']);
    g.removeTwoView(0, 2);
    expect(g.numCorrespondences(0)).toBe(0);
    expect(g.numObservations(0)).toBe(0);
    expect([...g.imagePairs()]).toEqual([]);
    expect(() => g.addTwoView(0, 1, Uint32Array.of(6, 0))).toThrow(RangeError);
    expect(() => g.addTwoView(0, 9, Uint32Array.of(0, 0))).toThrow(RangeError);
  });
});

describe('Reconstruction: transform and bookkeeping', () => {
  const CAM: CameraIntrinsics = { width: 800, height: 600, f: 700, cx: 399.5, cy: 299.5, k1: 0.02, k2: -0.003 };

  function smallModel(seed: number) {
    const rng = new Rng(seed);
    const rec = new Reconstruction();
    const cid = rec.addCamera(CAM);
    const nCam = 5,
      nPts = 40;
    const X = new Float64Array(3 * nPts);
    for (let j = 0; j < nPts; j++) X.set([-1 + 2 * rng.next(), -1 + 2 * rng.next(), 5 + 3 * rng.next()], 3 * j);
    const uv = new Float64Array(2);
    for (let i = 0; i < nCam; i++) {
      const pose = poseAt([0.5 * i - 1, 0.1 * rng.gaussian(), 0], rotvecToMat3([0.02 * rng.gaussian(), 0.05 * (i - 2), 0.01 * rng.gaussian()]));
      const kp = new Float32Array(4 * nPts);
      for (let j = 0; j < nPts; j++) {
        projectWorld(CAM, pose, X.subarray(3 * j, 3 * j + 3), uv);
        kp[4 * j] = uv[0] + 0.3 * rng.gaussian();
        kp[4 * j + 1] = uv[1] + 0.3 * rng.gaussian();
      }
      rec.addImage(i, cid, nPts, kp);
      rec.registerImage(i, pose);
    }
    for (let j = 0; j < nPts; j++) {
      const tr: Track[] = [];
      for (let i = 0; i < nCam; i++) tr.push({ image: i, feature: j });
      rec.addPoint(X.subarray(3 * j, 3 * j + 3), tr);
    }
    return rec;
  }

  it('extreme similarities (rotation ≈ π, s = 1e-3 / 1e3, |t| = 1e5) keep every reprojection error and map the centres', () => {
    for (const [s, ang, tt] of [
      [1e-3, Math.PI - 1e-10, 1e5],
      [1e3, Math.PI, -3e4],
      [7.5, 2.9, 12],
    ] as const) {
      const rec = smallModel(3);
      const R = rotvecToMat3([(0.6 * ang) / Math.hypot(0.6, 0.8), 0, (0.8 * ang) / Math.hypot(0.6, 0.8)]);
      const t = Float64Array.of(tt, -0.5 * tt, 0.25 * tt);
      const errBefore: number[] = [];
      for (const p of rec.points.values()) for (const el of p.track) errBefore.push(rec.observationError(p.id, el));
      const C0 = rec.registeredImageIds().map((id) => rec.imageCenter(id));
      rec.transform(s, R, t);
      // (|t| / s = 1e8 times the scene size costs ~1e-8 relative precision: ~1e-5 px)
      const tol = 1e-9 + 1e-13 * (Math.abs(tt) / s);
      let k = 0;
      for (const p of rec.points.values())
        for (const el of p.track) {
          const e = rec.observationError(p.id, el);
          expect(Math.abs(e - errBefore[k++])).toBeLessThan(tol);
        }
      rec.registeredImageIds().forEach((id, i) => {
        const C = rec.imageCenter(id);
        const c = C0[i];
        for (let r = 0; r < 3; r++) {
          const want = s * (R[3 * r] * c[0] + R[3 * r + 1] * c[1] + R[3 * r + 2] * c[2]) + t[r];
          expect(Math.abs(C[r] - want)).toBeLessThan(1e-9 * Math.max(1, Math.abs(tt), s));
        }
        // poses stay rotations
        const Ri = rec.images.get(id)!.pose.R;
        const det =
          Ri[0] * (Ri[4] * Ri[8] - Ri[5] * Ri[7]) - Ri[1] * (Ri[3] * Ri[8] - Ri[5] * Ri[6]) + Ri[2] * (Ri[3] * Ri[7] - Ri[4] * Ri[6]);
        expect(det).toBeCloseTo(1, 9);
      });
      // the inverse similarity restores the model
      const Rt = Float64Array.of(R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]);
      const ti = Float64Array.of(
        -(Rt[0] * t[0] + Rt[1] * t[1] + Rt[2] * t[2]) / s,
        -(Rt[3] * t[0] + Rt[4] * t[1] + Rt[5] * t[2]) / s,
        -(Rt[6] * t[0] + Rt[7] * t[1] + Rt[8] * t[2]) / s,
      );
      rec.transform(1 / s, Rt, ti);
      rec.registeredImageIds().forEach((id, i) => {
        const C = rec.imageCenter(id);
        for (let r = 0; r < 3; r++) expect(Math.abs(C[r] - C0[i][r])).toBeLessThan(1e-6);
      });
      expect(reconstructionInvariantErrors(rec)).toEqual([]);
    }
  });

  it('addPoint rejects a track listing an element twice / an out-of-range feature, without side effects', () => {
    const rec = smallModel(6);
    const img0 = rec.images.get(0)!;
    rec.deletePoint(img0.point3D[1]);
    rec.deletePoint(img0.point3D[2]);
    const before = reconstructionInvariantErrors(rec);
    expect(before).toEqual([]);
    const n0 = rec.numPoints;
    expect(() =>
      rec.addPoint([0, 0, 6], [
        { image: 0, feature: 1 },
        { image: 1, feature: 1 },
        { image: 0, feature: 1 },
      ]),
    ).toThrow(/twice/);
    expect(() =>
      rec.addPoint([0, 0, 6], [
        { image: 0, feature: 2 },
        { image: 1, feature: 999 },
      ]),
    ).toThrow(RangeError);
    expect(() =>
      rec.addPoint([0, 0, 6], [
        { image: 0, feature: 2 },
        { image: 77, feature: 0 },
      ]),
    ).toThrow(RangeError);
    // nothing changed; the features are still free
    expect(rec.numPoints).toBe(n0);
    expect(img0.point3D[1]).toBe(-1);
    expect(img0.point3D[2]).toBe(-1);
    expect(reconstructionInvariantErrors(rec)).toEqual([]);
    rec.addPoint([0, 0, 6], [
      { image: 0, feature: 1 },
      { image: 1, feature: 1 },
    ]);
    expect(reconstructionInvariantErrors(rec)).toEqual([]);
  });

  it('transform rejects a non-positive or non-finite scale (would put points behind the cameras)', () => {
    const rec = smallModel(4);
    const I = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
    const z = new Float64Array(3);
    for (const s of [0, -1, NaN, Infinity]) expect(() => rec.transform(s, I, z)).toThrow(RangeError);
    expect(rec.meanReprojectionError()).toBeLessThan(1);
  });

  it('random edits through every mutator keep all invariants (tracks ↔ features, counters, visibility)', () => {
    const rng = new Rng(77);
    const nImg = 10,
      nf = 50;
    const g = new CorrespondenceGraph();
    const rec = new Reconstruction();
    const cid = rec.addCamera(CAM);
    for (let i = 0; i < nImg; i++) g.addImage(i, nf);
    for (let a = 0; a < nImg; a++)
      for (let b = a + 1; b < nImg; b++) {
        const m: number[] = [];
        for (let f = 0; f < nf; f++) if (rng.next() < 0.4) m.push(f, rng.next() < 0.9 ? f : rng.int(nf));
        g.addTwoView(a, b, Uint32Array.from(m));
      }
    rec.attachGraph(g);
    const X = new Float64Array(3 * nf);
    for (let f = 0; f < nf; f++) X.set([-1 + 2 * rng.next(), -1 + 2 * rng.next(), 6 + rng.next()], 3 * f);
    const uv = new Float64Array(2);
    for (let i = 0; i < nImg; i++) {
      const pose = poseAt([0.4 * i, 0, 0], rotvecToMat3([0, -0.03 * i, 0]));
      const kp = new Float32Array(4 * nf);
      for (let f = 0; f < nf; f++) {
        projectWorld(CAM, pose, X.subarray(3 * f, 3 * f + 3), uv);
        // a few gross outliers so that the filters have work to do
        kp[4 * f] = uv[0] + (rng.next() < 0.05 ? 40 : 0.3 * rng.gaussian());
        kp[4 * f + 1] = uv[1] + 0.3 * rng.gaussian();
      }
      rec.addImage(i, cid, nf, kp);
      rec.registerImage(i, pose);
    }
    const check = (tag: string): void => {
      const errs = reconstructionInvariantErrors(rec, { graph: g });
      if (errs.length) throw new Error(`${tag}: ${errs.join('; ')}`);
    };
    check('start');
    for (let step = 0; step < 600; step++) {
      const op = rng.int(9);
      const f = rng.int(nf);
      const reg = rec.registeredImageIds();
      if (op <= 2) {
        const track: Track[] = [];
        for (const i of reg) if (rec.images.get(i)!.point3D[f] < 0 && rng.next() < 0.6) track.push({ image: i, feature: f });
        if (track.length >= 2) rec.addPoint(X.subarray(3 * f, 3 * f + 3), track);
      } else if (op === 3) {
        const i = reg[rng.int(reg.length)];
        if (i !== undefined && rec.images.get(i)!.point3D[f] >= 0) rec.deleteObservation(i, f);
      } else if (op === 4) {
        const i = reg[rng.int(reg.length)],
          j = reg[rng.int(reg.length)];
        if (i === undefined || j === undefined) continue;
        const pid = rec.images.get(i)!.point3D[f];
        if (pid >= 0 && rec.images.get(j)!.point3D[f] < 0) rec.addObservation(pid, { image: j, feature: f });
      } else if (op === 5) {
        const ids = [...rec.points.keys()];
        if (ids.length >= 2) {
          const a = ids[rng.int(ids.length)],
            b = ids[rng.int(ids.length)];
          rec.mergePoints(a, b);
        }
      } else if (op === 6) {
        if (rng.next() < 0.15 && reg.length > 3) rec.deregisterImage(reg[rng.int(reg.length)]);
        else {
          const un = [...rec.images.values()].filter((im) => !im.registered);
          if (un.length) rec.registerImage(un[0].id, poseAt([0.4 * un[0].id, 0, 0], rotvecToMat3([0, -0.03 * un[0].id, 0])));
        }
      } else if (op === 7) {
        if (rng.next() < 0.5) rec.filterAllPoints3D(4, 0.5 * DEG);
        else rec.filterObservationsInImages(4, [reg[rng.int(reg.length)]]);
      } else {
        const ids = [...rec.points.keys()];
        if (ids.length) rec.deletePoint(ids[rng.int(ids.length)]);
      }
      if (step % 20 === 0) check(`step ${step} op ${op}`);
    }
    check('end');
    rec.filterObservationsWithNegativeDepth();
    rec.filterImages(0.1, 10, 1);
    check('filters');
    const c = rec.clone();
    expect(reconstructionInvariantErrors(c, { graph: g })).toEqual([]);
    // the clone is independent
    const pid = [...c.points.keys()][0];
    if (pid !== undefined) c.deletePoint(pid);
    check('after clone edit');
  });
});

describe('RayTriangulator / angularError: degenerate geometry', () => {
  function raysTo(X: number[], centers: number[][], outlier: (k: number) => boolean, rng: Rng) {
    const n = centers.length;
    const C = new Float64Array(3 * n),
      D = new Float64Array(3 * n),
      Z = new Float64Array(3 * n);
    centers.forEach((c, k) => {
      let d = [X[0] - c[0], X[1] - c[1], X[2] - c[2]];
      if (outlier(k)) d = [d[0] + 0.3 * Math.hypot(...d) * rng.gaussian(), d[1] + 0.3 * Math.hypot(...d) * rng.gaussian(), d[2]];
      const dn = Math.hypot(d[0], d[1], d[2]);
      C.set(c, 3 * k);
      D.set([d[0] / dn, d[1] / dn, d[2] / dn], 3 * k);
      Z.set([0, 0, 1], 3 * k); // all cameras look along +z
    });
    return { C, D, Z };
  }

  it('points at infinity (parallel rays) and points behind every camera give no point', () => {
    const tri = new RayTriangulator(new Rng(1));
    const n = 6;
    const C = new Float64Array(3 * n),
      D = new Float64Array(3 * n),
      Z = new Float64Array(3 * n);
    for (let k = 0; k < n; k++) {
      C.set([k, 0, 0], 3 * k);
      D.set([0, 0, 1], 3 * k);
      Z.set([0, 0, 1], 3 * k);
    }
    expect(tri.estimate(C, D, Z, n, 1.5 * DEG, 2 * DEG)).toBe(0);
    // rays converge, but behind the cameras (z-axis points away)
    const r = raysTo([0, 0, 10], [[-2, 0, 0], [2, 0, 0], [0, 2, 0]], () => false, new Rng(2));
    for (let k = 0; k < 3; k++) r.Z[3 * k + 2] = -1;
    expect(tri.estimate(r.C, r.D, r.Z, 3, 1.5 * DEG, 2 * DEG)).toBe(0);
    // a single ray / zero rays
    expect(tri.estimate(r.C, r.D, r.Z, 1, 1.5 * DEG, 2 * DEG)).toBe(0);
    expect(tri.estimate(r.C, r.D, r.Z, 0, 1.5 * DEG, 2 * DEG)).toBe(0);
  });

  it('is accurate at huge offsets (1e6) and tiny scales (1e-4) and finds the inliers among 60 % outliers', () => {
    const rng = new Rng(3);
    for (const [scale, off] of [
      [1, 1e6],
      [1e-4, 0],
      [1e-4, 1e3],
    ] as const) {
      const X = [off + 0.3 * scale, off - 0.2 * scale, off + 8 * scale];
      const centers: number[][] = [];
      for (let k = 0; k < 50; k++) centers.push([off + scale * (rng.next() * 6 - 3), off + scale * (rng.next() * 2 - 1), off + scale * rng.next() * 0.5]);
      const bad = (k: number): boolean => k % 5 < 3; // 60 % outliers
      const { C, D, Z } = raysTo(X, centers, bad, rng);
      const tri = new RayTriangulator(new Rng(4));
      const cnt = tri.estimate(C, D, Z, 50, 1.5 * DEG, 2 * DEG);
      expect(cnt).toBe(20);
      for (let k = 0; k < 50; k++) expect(tri.inliers[k]).toBe(bad(k) ? 0 : 1);
      for (let r = 0; r < 3; r++) expect(Math.abs(tri.X[r] - X[r])).toBeLessThan(1e-6 * scale + 1e-9 * Math.abs(off));
    }
  });

  it('angularError is π for a point at the camera centre or behind it', () => {
    const pose = poseAt([1, 2, 3], rotvecToMat3([0.1, -0.2, 0.3]));
    expect(angularError(pose, 0.1, -0.1, [1, 2, 3])).toBe(Math.PI);
    // just in front of the camera on the observed ray: ~0
    const Rt = pose.R;
    const dirCam = [0.1, -0.1, 1];
    const Xw = [0, 1, 2].map((c) => 1 * [1, 2, 3][c] + 1e-3 * (Rt[c] * dirCam[0] + Rt[3 + c] * dirCam[1] + Rt[6 + c] * dirCam[2]));
    expect(angularError(pose, 0.1, -0.1, Xw)).toBeLessThan(1e-9);
    const Xb = [0, 1, 2].map((c) => [1, 2, 3][c] - (Rt[c] * dirCam[0] + Rt[3 + c] * dirCam[1] + Rt[6 + c] * dirCam[2]));
    expect(angularError(pose, 0.1, -0.1, Xb)).toBe(Math.PI);
  });
});

describe('IncrementalMapper: determinism, forward motion', () => {
  it('time-sliced global BA gives bit-identical results wherever the slices fall', async () => {
    const scene = generateScene({ kind: 'orbit', numImages: 22, seed: 9, numPoints: 2000, noisePx: 0.7, outlierRatio: 0.25, descriptors: false, k1: 0.03 });
    const guess = { ...scene.camera, f: scene.camera.f * 1.08, k1: 0 };
    const { db } = buildSyntheticDatabase(scene, { camera: guess });
    const run = async (sliceMs: number) => {
      const [rec] = await runIncrementalSfM(db, { intrinsics: guess, globalBASliceMs: sliceMs });
      const out: number[] = [];
      const cam = rec.cameras.get(0)!;
      out.push(cam.f, cam.k1, cam.k2, rec.numPoints, rec.numObservations);
      for (const id of rec.registeredImageIds()) out.push(id, ...rec.images.get(id)!.pose.R, ...rec.images.get(id)!.pose.t);
      for (const p of rec.points.values()) out.push(p.id, ...p.xyz);
      return out;
    };
    const a = await run(Infinity);
    // a 0.01 ms slice ends after (almost) every accepted LM step
    const b = await run(0.01);
    expect(b.length).toBe(a.length);
    let maxDiff = 0;
    for (let k = 0; k < a.length; k++) maxDiff = Math.max(maxDiff, Math.abs(a[k] - b[k]));
    expect(maxDiff).toBe(0);
  }, 300_000);

  function forwardData(n = 50, seed = 3) {
    const scene = generateScene({ kind: 'forward', numImages: n, seed, k1: 0.03, noisePx: 0.7, outlierRatio: 0.25, descriptors: false });
    const guess = { ...scene.camera, f: scene.camera.f * 1.08, k1: 0 };
    const { db } = buildSyntheticDatabase(scene, { camera: guess });
    return { scene, db, guess };
  }

  it('walking straight ahead: no regular initial pair exists, the last-resort level reconstructs everything', async () => {
    const { scene, db, guess } = forwardData();
    // every verified pair is forward motion with a small median angle
    let maxAngle = 0;
    for (const [, , g] of db.pairs()) maxAngle = Math.max(maxAngle, g.triAngle ?? 0);
    expect(maxAngle).toBeLessThan(4 * DEG);
    expect(await runIncrementalSfM(db, { intrinsics: guess, initFallback: false })).toEqual([]);
    const models = await runIncrementalSfM(db, { intrinsics: guess });
    expect(models.length).toBe(1);
    const rec = models[0];
    const ev = evaluateReconstruction(rec, scene);
    expect(ev.numRegistered).toBe(scene.imageIds.length);
    expect(ev.ateRelative).toBeLessThan(5e-4);
    expect(ev.maxRotErrorDeg).toBeLessThan(0.5);
    expect(ev.focalError).toBeLessThan(0.02); // forward motion constrains f weakly
    expect(rec.meanReprojectionError()).toBeLessThan(1);
    expect(reconstructionInvariantErrors(rec, { graph: rec.correspondenceGraph! })).toEqual([]);
  }, 300_000);

  it('the last-resort level is not used when a regular model exists (orbit unaffected)', async () => {
    const scene = generateScene({ kind: 'orbit', numImages: 14, seed: 4, numPoints: 2000, descriptors: false, k1: 0.02 });
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    const a = await runIncrementalSfM(db, { intrinsics: scene.camera });
    const b = await runIncrementalSfM(db, { intrinsics: scene.camera, initFallback: false });
    expect(a.length).toBe(1);
    expect(a[0].numRegisteredImages).toBe(14);
    expect(Array.from(a[0].images.get(5)!.pose.t)).toEqual(Array.from(b[0].images.get(5)!.pose.t));
  }, 300_000);

  it('LiveMapper initialises a forward walk once the relaxed thresholds apply', () => {
    const { scene, db, guess } = forwardData();
    const graph = new CorrespondenceGraph();
    const live = new LiveMapper(db, graph, { intrinsics: guess });
    let firstInit = -1;
    stream(db, live, graph, db.frameIds(), (id) => {
      if (live.initialized && firstInit < 0) firstInit = id;
    });
    expect(firstInit).toBeGreaterThanOrEqual(DEFAULT_FALLBACK_AFTER);
    const ev = evaluateReconstruction(live.reconstruction, scene);
    expect(ev.numRegistered).toBeGreaterThanOrEqual(scene.imageIds.length - 2);
    expect(ev.ateRelative).toBeLessThan(1e-3);
    // strict thresholds only: never initialises
    const graph2 = new CorrespondenceGraph();
    const strict = new LiveMapper(db, graph2, { intrinsics: guess, initFallback: false });
    stream(db, strict, graph2, db.frameIds().slice(0, 30));
    expect(strict.initialized).toBe(false);
  }, 300_000);

  it('LiveMapper: after initialising from an older pair, the pending keyframes register outward from that pair', () => {
    // walking away from the scene with 4 sequential pairs (CPU tier): the initial pair (50, 54) is
    // found while adding keyframe 40, so the keyframes between them need the pair's side first
    const { db, guess } = forwardData(60, 7);
    const graph = new CorrespondenceGraph();
    const live = new LiveMapper(db, graph, { intrinsics: guess });
    const ids = db.frameIds().reverse();
    let atInit: { id: number; registered: boolean; pending: number[] } | null = null;
    for (let k = 0; k < ids.length; k++) {
      const id = ids[k];
      graph.addImage(id, db.features.get(id)!.count);
      for (const j of ids.slice(Math.max(0, k - 4), k)) {
        const g = db.getTwoView(j, id);
        if (g && g.config !== 'degenerate' && g.inliers.length >= 30) graph.addTwoView(j, id, g.inliers);
      }
      const r = live.addKeyframe(id);
      if (r.initialized && !atInit) atInit = { id, registered: r.registered, pending: live.pendingIds() };
    }
    expect(atInit).not.toBeNull();
    expect(atInit!.registered).toBe(true);
    expect(atInit!.pending).toEqual([]);
    expect(live.reconstruction.numRegisteredImages).toBe(60);
  }, 300_000);
});

const DEFAULT_FALLBACK_AFTER = 10;

describe('LiveMapper: invariants and extreme world transforms during capture', () => {
  function orbitData(n: number, seed: number): { scene: SyntheticScene; db: FeatureDatabase; guess: CameraIntrinsics } {
    const scene = generateScene({ kind: 'orbit', numImages: n, seed, numPoints: 2500, noisePx: 0.5, outlierRatio: 0.2, descriptors: false, k1: 0.02 });
    const guess = { ...scene.camera, f: scene.camera.f * 1.05, k1: 0 };
    const { db } = buildSyntheticDatabase(scene, { camera: guess });
    return { scene, db, guess };
  }

  it('keeps every invariant (incl. brute-force visibility statistics) after each keyframe', () => {
    const { scene, db, guess } = orbitData(24, 12);
    const graph = new CorrespondenceGraph();
    const live = new LiveMapper(db, graph, { intrinsics: guess });
    stream(db, live, graph, db.frameIds(), (id) => {
      const errs = reconstructionInvariantErrors(live.reconstruction, { graph });
      if (errs.length) throw new Error(`keyframe ${id}: ${errs.join('; ')}`);
    });
    expect(evaluateReconstruction(live.reconstruction, scene).numRegistered).toBe(24);
  }, 300_000);

  it('refine(): a keyframe dropped by the image filter is reported as pending again', () => {
    const { db, guess } = orbitData(24, 15);
    const graph = new CorrespondenceGraph();
    const live = new LiveMapper(db, graph, { intrinsics: guess });
    stream(db, live, graph, db.frameIds());
    const rec = live.reconstruction;
    expect(rec.numRegisteredImages).toBe(24);
    expect(live.pendingIds()).toEqual([]);
    // turn keyframe 7 around: all its points end up behind it ⇒ filtered ⇒ image deregistered
    const img = rec.images.get(7)!;
    const C = rec.imageCenter(7);
    const flip = rotvecToMat3([0, Math.PI, 0]);
    const R = img.pose.R;
    const Rn = new Float64Array(9);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) Rn[3 * r + c] = flip[3 * r] * R[c] + flip[3 * r + 1] * R[3 + c] + flip[3 * r + 2] * R[6 + c];
    img.pose = poseAt(C, Rn); // same centre, looking away from the object
    live.refine();
    expect(img.registered).toBe(false);
    expect(live.pendingIds()).toEqual([7]);
    expect(reconstructionInvariantErrors(rec, { graph })).toEqual([]);
  }, 300_000);

  it('a keyframe dropped by a refinement through incrementalMapper (engine refine) is pending again', async () => {
    const { db, guess } = orbitData(24, 15);
    const graph = new CorrespondenceGraph();
    const live = new LiveMapper(db, graph, { intrinsics: guess });
    const ids = db.frameIds();
    stream(db, live, graph, ids.slice(0, 23));
    const rec = live.reconstruction;
    expect(rec.numRegisteredImages).toBe(23);
    const img = rec.images.get(7)!;
    const dropKeyframe7 = async () => {
      const flip = rotvecToMat3([0, Math.PI, 0]);
      const R = img.pose.R;
      const Rn = new Float64Array(9);
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) Rn[3 * r + c] = flip[3 * r] * R[c] + flip[3 * r + 1] * R[3 + c] + flip[3 * r + 2] * R[6 + c];
      img.pose = poseAt(rec.imageCenter(7), Rn);
      await live.incrementalMapper.iterativeGlobalRefinement(undefined, undefined, true);
      expect(img.registered).toBe(false);
    };
    // the next registered keyframe retries it
    await dropKeyframe7();
    stream(db, live, graph, ids.slice(23));
    expect(img.registered).toBe(true);
    expect(rec.numRegisteredImages).toBe(24);
    // the pending loop of a later engine refine retries it
    await dropKeyframe7();
    expect(live.pendingIds()).toEqual([7]);
    for (const id of live.pendingIds()) live.addKeyframe(id);
    expect(img.registered).toBe(true);
    expect(live.pendingIds()).toEqual([]);
  }, 300_000);

  it('a similarity applied mid-capture (rotation ≈ π, s = 1e3, |t| = 1e5) does not disturb further mapping', () => {
    const { scene, db, guess } = orbitData(36, 13);
    const graph = new CorrespondenceGraph();
    const live = new LiveMapper(db, graph, { intrinsics: guess });
    const ids = db.frameIds();
    stream(db, live, graph, ids.slice(0, 16));
    const rec = live.reconstruction;
    const mreBefore = rec.meanReprojectionError();
    const R = rotvecToMat3([0, Math.PI - 1e-7, 0]);
    rec.transform(1e3, R, Float64Array.of(1e5, -2e5, 5e4));
    expect(Math.abs(rec.meanReprojectionError() - mreBefore)).toBeLessThan(1e-6);
    stream(db, live, graph, ids.slice(16));
    live.refine();
    const ev = evaluateReconstruction(rec, scene);
    expect(ev.numRegistered).toBe(36);
    expect(ev.ateRelative).toBeLessThan(0.002);
    expect(ev.focalError).toBeLessThan(0.01);
    expect(rec.meanReprojectionError()).toBeLessThan(1);
    expect(reconstructionInvariantErrors(rec, { graph })).toEqual([]);
  }, 300_000);
});

describe('IncrementalMapper: next-best view, registration thresholds, BA gauge / intrinsics flags, cancel', () => {
  const CAM: CameraIntrinsics = { width: 1600, height: 1200, f: 1300, cx: 799.5, cy: 599.5, k1: 0, k2: 0 };

  /**
   * Images 0, 1 registered (GT poses) with `nPts` triangulated points (feature j of
   * both = point j); unregistered test images get correspondences to image 0 as
   * configured. Returns the parts and a mapper on them.
   */
  function rig(nPts: number, tests: { id: number; pose: Pose; good: number; bad: number; cluster?: boolean }[], seed = 1) {
    const rng = new Rng(seed);
    const X = new Float64Array(3 * nPts);
    for (let j = 0; j < nPts; j++) X.set([-2 + 4 * rng.next(), -1.5 + 3 * rng.next(), 6 + 4 * rng.next()], 3 * j);
    const poses = [poseAt([0, 0, 0], rotvecToMat3([0, 0, 0])), poseAt([1, 0, 0], rotvecToMat3([0, -0.1, 0]))];
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
      const n = t.good + t.bad;
      const kp = new Float32Array(4 * n);
      const m: number[] = [];
      for (let k = 0; k < n; k++) {
        const j = k; // corresponds to point j
        if (k < t.good) {
          projectWorld(CAM, t.pose, X.subarray(3 * j, 3 * j + 3), uv);
          kp[4 * k] = t.cluster ? 10 + (k % 7) : uv[0] + 0.3 * rng.gaussian();
          kp[4 * k + 1] = t.cluster ? 10 + ((k / 7) | 0) : uv[1] + 0.3 * rng.gaussian();
        } else {
          kp[4 * k] = rng.next() * (CAM.width - 1);
          kp[4 * k + 1] = rng.next() * (CAM.height - 1);
        }
        m.push(j, k);
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
    const mapper = new IncrementalMapper(new FeatureDatabase(), graph, rec, { seed: 5 });
    return { rec, graph, mapper, X };
  }

  const P2 = poseAt([0.4, 0.2, -0.3], rotvecToMat3([0.05, -0.04, 0.02]));

  it('next-best view: visibility-pyramid score ranks spread-out over clustered correspondences; tried images go last', () => {
    const { mapper } = rig(300, [
      { id: 10, pose: P2, good: 60, bad: 0, cluster: true }, // 60 visible points in one corner
      { id: 11, pose: P2, good: 50, bad: 0 }, // fewer, but spread over the image
      { id: 12, pose: P2, good: 25, bad: 0 }, // too few visible points (< 30): never proposed
    ]);
    expect(mapper.findNextImages()).toEqual([11, 10]);
    // a failed trial moves the image behind untried ones
    mapper.rec.images.get(11)!.keypoints!.fill(1e6); // all observations at one (far) pixel: degenerate
    mapper.rec.syncVisibility(11); // (keypoints are immutable by contract; this test changed them)
    expect(mapper.registerNextImage(11)).toBe(false);
    expect(mapper.findNextImages()).toEqual([10, 11]);
    // after maxRegTrials (3) attempts it is no longer proposed
    expect(mapper.registerNextImage(11)).toBe(false);
    expect(mapper.registerNextImage(11)).toBe(false);
    expect(mapper.findNextImages()).toEqual([10]);
    // a failed registration leaves no trace
    const img = mapper.rec.images.get(11)!;
    expect(img.registered).toBe(false);
    expect(img.numPoints3D).toBe(0);
    expect(reconstructionInvariantErrors(mapper.rec, { graph: mapper.graph })).toEqual([]);
  });

  it('registration needs ≥ absPoseMinNumInliers inliers AND an inlier ratio ≥ absPoseMinInlierRatio', () => {
    const { mapper, rec } = rig(400, [
      { id: 20, pose: P2, good: 40, bad: 160 }, // 40 inliers but 20 %: rejected
      { id: 21, pose: P2, good: 66, bad: 134 }, // 33 %: accepted
      { id: 22, pose: P2, good: 26, bad: 6 }, // 81 % but 26 < 30 inliers: rejected
      // 60 observations within a 7 × 9 px patch: a camera ~1e8 units away explains them all
      // within 12 px (PnP + refinement drift there) — degenerate, must be rejected
      { id: 23, pose: P2, good: 60, bad: 0, cluster: true },
    ]);
    expect(mapper.registerNextImage(20)).toBe(false);
    expect(mapper.registerNextImage(22)).toBe(false);
    expect(mapper.registerNextImage(23)).toBe(false);
    expect(rec.images.get(23)!.registered).toBe(false);
    expect(mapper.registerNextImage(21)).toBe(true);
    const img = rec.images.get(21)!;
    // only the inliers became observations; the pose is the true one
    expect(img.numPoints3D).toBe(66);
    for (let k = 0; k < 9; k++) expect(img.pose.R[k]).toBeCloseTo(P2.R[k], 3);
    for (let k = 0; k < 3; k++) expect(img.pose.t[k]).toBeCloseTo(P2.t[k], 2);
    expect(reconstructionInvariantErrors(rec, { graph: mapper.graph })).toEqual([]);
  });

  it('BA gauge and intrinsics flags: constant / masked parameters stay bit-identical', async () => {
    const scene = generateScene({ kind: 'orbit', numImages: 16, seed: 31, numPoints: 2000, descriptors: false, k1: 0.02, outlierRatio: 0 });
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    const graph = buildCorrespondenceGraph(db);
    const rec = new Reconstruction();
    const m = new IncrementalMapper(db, graph, rec, { intrinsics: { ...scene.camera, f: scene.camera.f * 1.03, k1: 0 } });
    for (const id of db.frameIds()) m.ensureImage(id);
    const init = m.findInitialPair(100, 8)!;
    expect(await m.reconstructFrom(init.a, init.b, init.geom, 2)).toBe(true);
    expect(rec.numRegisteredImages).toBe(16);
    const cam = rec.cameras.get(0)!;
    const relAxis = (a: Pose, b: Pose): number => {
      const w = [0, 1, 2].map((c) => a.R[c] * a.t[0] + a.R[3 + c] * a.t[1] + a.R[6 + c] * a.t[2]);
      const r = [0, 1, 2].map((q) => Math.abs(b.t[q] - (b.R[3 * q] * w[0] + b.R[3 * q + 1] * w[1] + b.R[3 * q + 2] * w[2])));
      return r[0] >= r[1] && r[0] >= r[2] ? 0 : r[1] >= r[2] ? 1 : 2;
    };
    // perturb every pose a little so that the BAs have work to do
    const rng = new Rng(2);
    for (const id of rec.registeredImageIds()) for (let k = 0; k < 3; k++) rec.images.get(id)!.pose.t[k] += 1e-3 * rng.gaussian();
    // local BA: 16 registered > 7 bundle images ⇒ intrinsics constant; gauge = last bundle image fixed + one t component
    const id = rec.registeredImageIds()[8];
    const bundle = m.findLocalBundle(id);
    expect(bundle.length).toBe(6);
    const i1 = bundle[5],
      i2 = bundle[4];
    const ax = relAxis(rec.images.get(i1)!.pose, rec.images.get(i2)!.pose);
    const snap = (k: number) => [...rec.images.get(k)!.pose.R, ...rec.images.get(k)!.pose.t];
    const p1 = snap(i1),
      t2 = rec.images.get(i2)!.pose.t[ax],
      camBefore = { ...cam },
      tId = Array.from(rec.images.get(id)!.pose.t);
    const rep = m.adjustLocalBundle(id, [], true);
    expect(rep.ba?.success).toBe(true);
    expect(rep.ba!.iterations).toBeGreaterThan(0);
    expect(snap(i1)).toEqual(p1);
    expect(rec.images.get(i2)!.pose.t[ax]).toBe(t2);
    expect({ ...cam }).toEqual(camBefore);
    expect(Array.from(rec.images.get(id)!.pose.t)).not.toEqual(tId);
    // global BA: first registered image fixed, the largest relative-translation component of the second fixed, intrinsics free
    const reg = rec.registeredImageIds();
    const g0 = snap(reg[0]);
    const gax = relAxis(rec.images.get(reg[0])!.pose, rec.images.get(reg[1])!.pose);
    const g1 = rec.images.get(reg[1])!.pose.t[gax];
    cam.f *= 1.01;
    const fBefore = cam.f;
    const res = m.adjustGlobalBundle();
    expect(res?.success).toBe(true);
    expect(snap(reg[0])).toEqual(g0);
    expect(rec.images.get(reg[1])!.pose.t[gax]).toBe(g1);
    expect(cam.f).not.toBe(fBefore);
    expect(Math.abs(cam.f - scene.camera.f) / scene.camera.f).toBeLessThan(0.002);
    expect(reconstructionInvariantErrors(rec, { graph })).toEqual([]);
  }, 120_000);

  it('cancellation at an arbitrary point (k-th check of the token) always rejects with CancelledError', async () => {
    const scene = generateScene({ kind: 'orbit', numImages: 12, seed: 14, numPoints: 1500, descriptors: false, k1: 0.02 });
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    let total = 0;
    {
      const counting = {
        get cancelled() {
          total++;
          return false;
        },
      };
      const models = await runIncrementalSfM(db, { intrinsics: scene.camera, globalBASliceMs: 0.01 }, undefined, counting);
      expect(models[0].numRegisteredImages).toBe(12);
    }
    expect(total).toBeGreaterThan(20);
    for (const k of [0, 1, 2, 5, Math.floor(total / 4), Math.floor(total / 2), total - 2]) {
      let reads = 0;
      const token = {
        get cancelled() {
          return ++reads > k;
        },
      };
      const p = runIncrementalSfM(db, { intrinsics: scene.camera, globalBASliceMs: 0.01 }, undefined, token);
      await expect(p).rejects.toBeInstanceOf(SfMCancelledError);
    }
  }, 300_000);
});
