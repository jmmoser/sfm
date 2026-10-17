import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, Pose } from '../core/types';
import { projectWorld } from '../geometry/camera';
import { Rng } from '../math/random';
import { rotvecToMat3 } from '../math/rotation';
import { CorrespondenceGraph } from './correspondence-graph';
import { Reconstruction, maxVisibilityScore, hasBogusParams, type Track } from './reconstruction';

const CAM: CameraIntrinsics = { width: 800, height: 600, f: 700, cx: 399.5, cy: 299.5, k1: 0.02, k2: -0.003 };

function poseAt(cx: number, cy: number, cz: number, rv: [number, number, number] = [0, 0, 0]): Pose {
  const R = rotvecToMat3(Float64Array.from(rv));
  return {
    R,
    t: Float64Array.of(
      -(R[0] * cx + R[1] * cy + R[2] * cz),
      -(R[3] * cx + R[4] * cy + R[5] * cz),
      -(R[6] * cx + R[7] * cy + R[8] * cz),
    ),
  };
}

/** n cameras on a line, m points in front; every point seen by every camera (feature index = point index). */
function scene(nCams: number, nPts: number, noise: number, seed = 1) {
  const rng = new Rng(seed);
  const poses: Pose[] = [];
  for (let i = 0; i < nCams; i++) poses.push(poseAt(0.4 * i, 0.05 * rng.gaussian(), 0, [0.02 * rng.gaussian(), -0.05 * i, 0.01 * rng.gaussian()]));
  const X = new Float64Array(3 * nPts);
  for (let j = 0; j < nPts; j++) {
    X[3 * j] = -1 + 3 * rng.next();
    X[3 * j + 1] = -1 + 2 * rng.next();
    X[3 * j + 2] = 4 + 4 * rng.next();
  }
  const kps: Float32Array[] = [];
  const uv = new Float64Array(2);
  for (let i = 0; i < nCams; i++) {
    const kp = new Float32Array(4 * nPts);
    for (let j = 0; j < nPts; j++) {
      projectWorld(CAM, poses[i], X.subarray(3 * j, 3 * j + 3), uv);
      kp[4 * j] = uv[0] + noise * rng.gaussian();
      kp[4 * j + 1] = uv[1] + noise * rng.gaussian();
    }
    kps.push(kp);
  }
  return { poses, X, kps };
}

function buildRec(nCams: number, nPts: number, noise: number, seed = 1) {
  const s = scene(nCams, nPts, noise, seed);
  const rec = new Reconstruction();
  const cid = rec.addCamera(CAM);
  for (let i = 0; i < nCams; i++) {
    rec.addImage(i, cid, nPts, s.kps[i]);
    rec.registerImage(i, s.poses[i]);
  }
  for (let j = 0; j < nPts; j++) {
    const track: Track[] = [];
    for (let i = 0; i < nCams; i++) track.push({ image: i, feature: j });
    rec.addPoint(s.X.subarray(3 * j, 3 * j + 3), track, [j % 256, 2, 3]);
  }
  return { rec, ...s };
}

/** Brute-force visibility statistics of `img` (same definitions as the reconstruction). */
function bruteVisibility(rec: Reconstruction, g: CorrespondenceGraph, id: number): { visible: number; score: number } {
  const img = rec.images.get(id)!;
  const cam = rec.cameras.get(img.cameraId)!;
  const cells: Map<string, number>[] = [];
  for (let l = 0; l < 6; l++) cells.push(new Map());
  let visible = 0;
  for (let f = 0; f < img.point3D.length; f++) {
    let cnt = 0;
    for (const c of g.correspondences(id, f)) {
      const o = rec.images.get(c.image);
      if (o && o.point3D[c.feature] >= 0) cnt++;
    }
    if (cnt === 0) continue;
    visible++;
    const kp = img.keypoints!;
    let cx = Math.min(63, Math.max(0, Math.floor(((kp[4 * f] + 0.5) * 64) / cam.width)));
    let cy = Math.min(63, Math.max(0, Math.floor(((kp[4 * f + 1] + 0.5) * 64) / cam.height)));
    for (let l = 5; l >= 0; l--) {
      cells[l].set(`${cx},${cy}`, 1);
      cx >>= 1;
      cy >>= 1;
    }
  }
  let score = 0;
  for (let l = 0; l < 6; l++) score += cells[l].size * (1 << (2 * (l + 1)));
  return { visible, score };
}

describe('Reconstruction bookkeeping', () => {
  it('adds / deletes observations and points with COLMAP semantics', () => {
    const { rec } = buildRec(4, 10, 0);
    expect(rec.numPoints).toBe(10);
    expect(rec.numObservations).toBe(40);
    expect(rec.meanTrackLength()).toBe(4);
    const img0 = rec.images.get(0)!;
    expect(img0.numPoints3D).toBe(10);
    const pid = img0.point3D[3];
    rec.deleteObservation(0, 3);
    expect(img0.point3D[3]).toBe(-1);
    expect(rec.points.get(pid)!.track.length).toBe(3);
    rec.deleteObservation(1, 3);
    expect(rec.points.get(pid)!.track.length).toBe(2);
    // deleting from a 2-track deletes the point and frees the other feature
    rec.deleteObservation(2, 3);
    expect(rec.points.has(pid)).toBe(false);
    expect(rec.images.get(3)!.point3D[3]).toBe(-1);
    expect(rec.numObservations).toBe(36);
    expect(() => rec.addPoint([0, 0, 1], [{ image: 0, feature: 4 }])).toThrow(); // feature taken
    expect(() => rec.addObservation(img0.point3D[5], { image: 1, feature: 5 })).toThrow();
    // addObservation on a free feature
    const p5 = img0.point3D[5];
    rec.deleteObservation(1, 5);
    rec.addObservation(p5, { image: 1, feature: 5 });
    expect(rec.points.get(p5)!.track.length).toBe(4);
    // deregister removes all observations of the image
    rec.deregisterImage(1);
    expect(rec.images.get(1)!.numPoints3D).toBe(0);
    expect(rec.registeredImageIds()).toEqual([0, 2, 3]);
    for (const p of rec.points.values()) for (const el of p.track) expect(el.image).not.toBe(1);
  });

  it('mergePoints averages position / colour by track length and concatenates tracks', () => {
    const rec = new Reconstruction();
    const cid = rec.addCamera(CAM);
    for (let i = 0; i < 5; i++) rec.addImage(i, cid, 4);
    const a = rec.addPoint([0, 0, 3], [
      { image: 0, feature: 0 },
      { image: 1, feature: 0 },
      { image: 2, feature: 0 },
    ], [30, 60, 90]);
    const b = rec.addPoint([4, 0, 3], [
      { image: 3, feature: 1 },
      { image: 4, feature: 1 },
    ], [130, 160, 190]);
    const m = rec.mergePoints(a, b);
    expect(rec.points.has(a) || rec.points.has(b)).toBe(false);
    const p = rec.points.get(m)!;
    expect(p.xyz[0]).toBeCloseTo((3 * 0 + 2 * 4) / 5, 12);
    expect(p.color).toEqual([70, 100, 130]);
    expect(p.track.length).toBe(5);
    for (const el of p.track) expect(rec.images.get(el.image)!.point3D[el.feature]).toBe(m);
    expect(rec.numObservations).toBe(5);
  });

  it('transform keeps projections invariant and maps centres (sim3 on poses)', () => {
    const { rec } = buildRec(5, 30, 0.3, 3);
    const errBefore = rec.meanReprojectionError();
    const centers = rec.registeredImageIds().map((id) => rec.imageCenter(id));
    const s = 2.7,
      R = rotvecToMat3(Float64Array.of(0.3, -1.1, 0.4)),
      t = Float64Array.of(100, -50, 1e3);
    const uv0 = new Float64Array(2),
      uv1 = new Float64Array(2);
    const p0 = rec.points.values().next().value!;
    projectWorld(CAM, rec.images.get(2)!.pose, p0.xyz, uv0);
    rec.transform(s, R, t);
    projectWorld(CAM, rec.images.get(2)!.pose, p0.xyz, uv1);
    expect(uv1[0]).toBeCloseTo(uv0[0], 6);
    expect(uv1[1]).toBeCloseTo(uv0[1], 6);
    expect(rec.meanReprojectionError()).toBeCloseTo(errBefore, 6);
    rec.registeredImageIds().forEach((id, k) => {
      const c = rec.imageCenter(id);
      const C = centers[k];
      for (let r = 0; r < 3; r++) expect(c[r]).toBeCloseTo(s * (R[3 * r] * C[0] + R[3 * r + 1] * C[1] + R[3 * r + 2] * C[2]) + t[r], 6);
      // R stays a rotation
      const Ri = rec.images.get(id)!.pose.R;
      expect(Ri[0] * Ri[0] + Ri[1] * Ri[1] + Ri[2] * Ri[2]).toBeCloseTo(1, 12);
    });
  });

  it('meanReprojectionError matches the noise level; clone is independent', () => {
    const sigma = 0.7;
    const { rec } = buildRec(6, 400, sigma, 5);
    const e = rec.meanReprojectionError();
    // E‖n‖ of a 2-D Gaussian = σ √(π/2)
    expect(Math.abs(e - sigma * Math.sqrt(Math.PI / 2))).toBeLessThan(0.05);
    rec.updatePointErrors();
    for (const p of rec.points.values()) expect(p.error).toBeGreaterThan(0);
    const c = rec.clone();
    expect(c.meanReprojectionError()).toBe(e);
    c.transform(1, Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), Float64Array.of(1, 0, 0));
    const pid = [...c.points.keys()][0];
    c.deletePoint(pid);
    expect(rec.points.has(pid)).toBe(true);
    expect(rec.meanReprojectionError()).toBe(e);
    expect(rec.numObservations).toBe(2400);
    expect(c.numObservations).toBe(2394);
  });

  it('filters large reprojection errors, small triangulation angles and negative depth', () => {
    const { rec, kps } = buildRec(4, 50, 0.2, 9);
    // corrupt one observation of point 0 (feature 0 of image 2) by 10 px
    kps[2][0] += 10;
    // corrupt point 1 in 3 of 4 images → point deleted (≤ 1 good observation left)
    kps[0][4] += 20;
    kps[1][4] += 20;
    kps[2][4] -= 20;
    const p0 = rec.images.get(2)!.point3D[0];
    const p1 = rec.images.get(0)!.point3D[1];
    const removed = rec.filterAllPoints3D(4, (1 * Math.PI) / 180);
    expect(rec.points.get(p0)!.track.length).toBe(3);
    expect(rec.points.has(p1)).toBe(false);
    expect(removed).toBe(1 + 4);
    // tiny triangulation angle (point 1000 units away, baseline 0.01) and negative depth
    const rec2 = new Reconstruction();
    const cid = rec2.addCamera(CAM);
    for (let i = 0; i < 2; i++) {
      rec2.addImage(i, cid, 2, new Float32Array([399.5, 299.5, 1, 0, 399.5, 299.5, 1, 0]));
      rec2.registerImage(i, poseAt(0.01 * i, 0, 0));
    }
    rec2.addPoint([0.005, 0, 1e3], [
      { image: 0, feature: 0 },
      { image: 1, feature: 0 },
    ]);
    // point behind both cameras
    rec2.addPoint([0.005, 0, -5], [
      { image: 0, feature: 1 },
      { image: 1, feature: 1 },
    ]);
    expect(rec2.filterObservationsWithNegativeDepth()).toBe(1); // first deletion removes the 2-track point
    expect(rec2.numPoints).toBe(1);
    expect(rec2.filterAllPoints3D(4, (1.5 * Math.PI) / 180)).toBe(2);
    expect(rec2.numPoints).toBe(0);
  });

  it('filterObservationsInImages only checks observations in the given images', () => {
    const { rec, kps } = buildRec(4, 20, 0.2, 13);
    kps[1][0] += 8; // point 0 in image 1
    kps[2][4] += 8; // point 1 in image 2
    const p0 = rec.images.get(1)!.point3D[0],
      p1 = rec.images.get(2)!.point3D[1];
    expect(rec.filterObservationsInImages(4, [1])).toBe(1);
    expect(rec.points.get(p0)!.track.length).toBe(3);
    expect(rec.points.get(p1)!.track.length).toBe(4); // image 2 not checked
    expect(rec.filterObservationsInImages(4, [0, 2, 3])).toBe(1);
    expect(rec.points.get(p1)!.track.length).toBe(3);
    // a 2-track point loses both observations
    const q = rec.images.get(0)!.point3D[2];
    rec.deleteObservation(2, 2);
    rec.deleteObservation(3, 2);
    kps[0][8] += 50;
    expect(rec.filterObservationsInImages(4, [0])).toBe(2);
    expect(rec.points.has(q)).toBe(false);
  });

  it('filterImages deregisters images without points or with bogus intrinsics', () => {
    const { rec } = buildRec(3, 5, 0, 2);
    rec.addImage(10, 0, 3, null);
    rec.registerImage(10, poseAt(5, 0, 0));
    expect(rec.filterImages(0.1, 10, 1)).toEqual([10]);
    expect(rec.images.get(10)!.registered).toBe(false);
    rec.cameras.get(0)!.f = 1e5; // bogus focal
    expect(hasBogusParams(rec.cameras.get(0)!)).toBe(true);
    expect(rec.filterImages(0.1, 10, 1).sort()).toEqual([0, 1, 2]);
    expect(rec.numPoints).toBe(0);
  });
});

describe('Reconstruction visibility statistics', () => {
  it('equal a brute-force recount after random edits (fuzz)', () => {
    const rng = new Rng(11);
    const nImg = 8,
      nf = 60;
    const g = new CorrespondenceGraph();
    const rec = new Reconstruction();
    const cid = rec.addCamera(CAM);
    for (let i = 0; i < nImg; i++) {
      g.addImage(i, nf);
      const kp = new Float32Array(4 * nf);
      for (let f = 0; f < nf; f++) {
        kp[4 * f] = rng.next() * (CAM.width - 1);
        kp[4 * f + 1] = rng.next() * (CAM.height - 1);
      }
      rec.addImage(i, cid, nf, kp);
      rec.registerImage(i, poseAt(i, 0, 0));
    }
    // graph: each feature f of every image corresponds to feature f of other images (random subset)
    for (let a = 0; a < nImg; a++)
      for (let b = a + 1; b < nImg; b++) {
        const m: number[] = [];
        for (let f = 0; f < nf; f++) if (rng.next() < 0.5) m.push(f, f);
        g.addTwoView(a, b, Uint32Array.from(m));
      }
    rec.attachGraph(g);
    const check = (): void => {
      for (let i = 0; i < nImg; i++) {
        const ref = bruteVisibility(rec, g, i);
        expect(rec.images.get(i)!.numVisiblePoints3D).toBe(ref.visible);
        expect(rec.images.get(i)!.visibilityScore).toBe(ref.score);
      }
    };
    check();
    for (let step = 0; step < 300; step++) {
      const op = rng.int(5);
      const f = rng.int(nf);
      if (op <= 1) {
        // create a point from free observations of feature f
        const track: Track[] = [];
        for (let i = 0; i < nImg; i++) if (rec.images.get(i)!.point3D[f] < 0 && rng.next() < 0.5) track.push({ image: i, feature: f });
        if (track.length >= 2) rec.addPoint([0, 0, 5], track);
      } else if (op === 2) {
        const i = rng.int(nImg);
        const pid = rec.images.get(i)!.point3D[f];
        if (pid >= 0) rec.deleteObservation(i, f);
      } else if (op === 3) {
        const i = rng.int(nImg);
        const pid = rec.images.get(i)!.point3D[f];
        // extend with a free observation
        const j = rng.int(nImg);
        if (pid >= 0 && rec.images.get(j)!.point3D[f] < 0) rec.addObservation(pid, { image: j, feature: f });
      } else {
        // merge two points of different features
        const ids = [...rec.points.keys()];
        if (ids.length >= 2) {
          const a = ids[rng.int(ids.length)],
            b = ids[rng.int(ids.length)];
          if (a !== b) rec.mergePoints(a, b);
        }
      }
      if (step % 25 === 0) check();
    }
    check();
    rec.deregisterImage(3);
    check();
    const c = rec.clone();
    for (let i = 0; i < nImg; i++) expect(c.images.get(i)!.visibilityScore).toBe(rec.images.get(i)!.visibilityScore);
    // syncVisibility recomputes from scratch
    rec.syncVisibility(2);
    check();
    expect(rec.images.get(0)!.visibilityScore).toBeLessThanOrEqual(maxVisibilityScore());
  });

  it('images added after attachGraph see existing points', () => {
    const g = new CorrespondenceGraph();
    const rec = new Reconstruction();
    const cid = rec.addCamera(CAM);
    g.addImage(0, 3);
    g.addImage(1, 3);
    g.addImage(2, 3);
    g.addTwoView(0, 1, Uint32Array.of(0, 0, 1, 1));
    g.addTwoView(0, 2, Uint32Array.of(0, 2));
    rec.attachGraph(g);
    const kp = new Float32Array(12);
    rec.addImage(0, cid, 3, kp);
    rec.addImage(1, cid, 3, kp);
    rec.addPoint([0, 0, 1], [
      { image: 0, feature: 0 },
      { image: 1, feature: 0 },
    ]);
    rec.addImage(2, cid, 3, kp);
    expect(rec.images.get(2)!.numVisiblePoints3D).toBe(1);
    expect(rec.images.get(0)!.numVisiblePoints3D).toBe(1); // feature 0 ↔ (1,0)
    expect(rec.visibleCounts(2)![2]).toBe(1);
  });
});
