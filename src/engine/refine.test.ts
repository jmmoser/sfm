/**
 * refine's final principal-point bundle adjustment (src/engine/refine.ts refinePrincipalPoint):
 * a capture whose true principal point is offset from the image centre is reconstructed with the
 * centred guess (the mapper keeps it fixed, as COLMAP does) — every camera then absorbs the offset
 * as a constant rotation; the final BA with the principal point free removes it.
 */
import { describe, expect, it } from 'vitest';
import { runIncrementalSfM } from '../sfm/mapper';
import { buildSyntheticDatabase, evaluateReconstruction, generateScene } from '../sfm/synthetic';
import { explainedCorrespondenceRatio, liveModelHealth, refinePrincipalPoint, splitPairs } from './refine';
import type { GyroConsistency } from '../sfm/gyro';

describe('refinePrincipalPoint', () => {
  it('recovers an offset principal point and removes the constant rotation error', async () => {
    const scene = generateScene({ kind: 'orbit', numImages: 30, seed: 5, numPoints: 2500, noisePx: 0.5, outlierRatio: 0.2, descriptors: false, k1: 0.02 });
    const centred = { ...scene.camera };
    // true principal point (+4, −3) px from the centre: shift every observation
    const dx = 4,
      dy = -3;
    for (const fs of scene.features)
      for (let i = 0; i < fs.count; i++) {
        fs.keypoints[4 * i] += dx;
        fs.keypoints[4 * i + 1] += dy;
      }
    const { db } = buildSyntheticDatabase(scene, { camera: centred });
    const [rec] = await runIncrementalSfM(db, { intrinsics: centred });
    expect(rec.numRegisteredImages).toBe(30);
    const before = evaluateReconstruction(rec, scene);
    const fixedCam = [...rec.cameras.values()][0];
    expect(fixedCam.cx).toBeCloseTo(centred.cx, 6); // kept fixed by the mapper
    // ≈ atan(5 px / f) of constant rotation error
    expect(before.meanRotErrorDeg).toBeGreaterThan(0.1);

    const r = await refinePrincipalPoint(rec, db, rec.correspondenceGraph!, { intrinsics: centred });
    expect(r.accepted).toBe(true);
    const cam = [...r.rec.cameras.values()][0];
    expect(Math.abs(cam.cx - (centred.cx + dx))).toBeLessThan(0.5);
    expect(Math.abs(cam.cy - (centred.cy + dy))).toBeLessThan(0.5);
    const after = evaluateReconstruction(r.rec, scene);
    expect(after.numRegistered).toBe(30);
    expect(after.meanRotErrorDeg).toBeLessThan(0.3 * before.meanRotErrorDeg);
    expect(after.ate).toBeLessThanOrEqual(before.ate * 1.05);
    // the input model is untouched (refine works on a clone)
    expect([...rec.cameras.values()][0].cx).toBeCloseTo(centred.cx, 6);
  }, 120_000);
});

describe('explainedCorrespondenceRatio (model choice)', () => {
  it('drops when the tracks of the two halves of a capture are kept apart (an unclosed loop)', async () => {
    const scene = generateScene({ kind: 'orbit', numImages: 24, seed: 9, noisePx: 0.3, descriptors: false, outlierRatio: 0 });
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    const [rec] = await runIncrementalSfM(db, { intrinsics: { ...scene.camera } });
    const full = explainedCorrespondenceRatio(rec, db);
    expect(full).toBeGreaterThan(0.8);
    // split every track spanning both halves into two points
    const split = rec.clone();
    const half = new Set(scene.imageIds.slice(0, 12));
    for (const p of [...split.points.values()]) {
      const a = p.track.filter((el) => half.has(el.image));
      const b = p.track.filter((el) => !half.has(el.image));
      if (a.length < 2 || b.length < 2) continue;
      const xyz = Float64Array.from(p.xyz);
      split.deletePoint(p.id);
      split.addPoint(xyz, a);
      split.addPoint(xyz, b);
    }
    const parted = explainedCorrespondenceRatio(split, db);
    expect(parted).toBeLessThan(full - 0.05);
    // unregistered images do not count
    expect(explainedCorrespondenceRatio(new (rec.constructor as new () => typeof rec)(), db)).toBe(0);
  });
});

describe('liveModelHealth (offline SfM skipped for a healthy live model)', () => {
  it('healthy closed model; symptoms: unclosed loop, unregistered keyframe, gyro breaks / outliers, no gyro', async () => {
    const scene = generateScene({ kind: 'loop', numImages: 30, seed: 4, noisePx: 0.3, descriptors: false, outlierRatio: 0 });
    const { db } = buildSyntheticDatabase(scene, { verify: false });
    const [rec] = await runIncrementalSfM(db, { intrinsics: { ...scene.camera } });
    const ids = scene.imageIds;
    expect(rec.numRegisteredImages).toBe(ids.length);
    const gyro: GyroConsistency = { pairs: 29, breaks: [], outliers: [], breaksWithoutOutliers: 0, medianErrDeg: 0.05 };
    const ok = liveModelHealth(rec, db, ids, gyro);
    expect(ok.healthy).toBe(true);
    expect(ok.split!.splitPairs).toBe(0);
    expect(ok.reason).toMatch(/all 30 keyframes registered/);
    // the loop left open: the second half of the capture drifted 30 cm (its cameras and the points of
    // its tracks), and every track spanning both halves is kept as two points
    const open = rec.clone();
    const half = new Set(ids.slice(0, 15));
    const d = [0.3, 0, 0];
    for (const id of ids.slice(15)) {
      const p = open.images.get(id)!.pose;
      for (let r = 0; r < 3; r++) p.t[r] -= p.R[3 * r] * d[0] + p.R[3 * r + 1] * d[1] + p.R[3 * r + 2] * d[2];
    }
    let parted = 0;
    for (const p of [...open.points.values()]) {
      const a = p.track.filter((el) => half.has(el.image));
      const b = p.track.filter((el) => !half.has(el.image));
      const moved = [p.xyz[0] + d[0], p.xyz[1] + d[1], p.xyz[2] + d[2]];
      if (a.length === 0) p.xyz.set(moved);
      if (a.length < 2 || b.length < 2) continue;
      const xyz = Float64Array.from(p.xyz);
      open.deletePoint(p.id);
      open.addPoint(xyz, a);
      open.addPoint(moved, b);
      parted++;
    }
    expect(parted).toBeGreaterThan(100);
    const sp = splitPairs(open, db);
    expect(sp.loopPairs).toBeGreaterThanOrEqual(2);
    expect(sp.splitRatio).toBeGreaterThan(splitPairs(rec, db).splitRatio);
    const h = liveModelHealth(open, db, ids, gyro);
    expect(h.healthy).toBe(false);
    expect(h.reason).toMatch(/unclosed loop/);
    // with too few split correspondences for the count limits (a phone-rate capture's short loop
    // tracks), the loop-like pairs that join the two ends of the capture still tell
    const lax = { maxLoopCorrespondences: 1e9, maxSplitCorrespondences: 1e9 };
    const far = liveModelHealth(open, db, ids, gyro, lax);
    expect(far.healthy).toBe(false);
    expect(far.reason).toMatch(/loop-like split pairs across the capture/);
    const span = (l: { a: number; b: number }): number => Math.abs(ids.indexOf(l.a) - ids.indexOf(l.b));
    expect(Math.max(...far.split!.loops.map(span))).toBeGreaterThanOrEqual(10);
    expect(liveModelHealth(open, db, ids, gyro, { ...lax, maxLoopSpan: 1 }).healthy).toBe(true);
    expect(liveModelHealth(rec, db, ids, gyro, lax).healthy).toBe(true);
    // a keyframe the live model never registered
    const missing = rec.clone();
    missing.deregisterImage(ids[7]);
    expect(liveModelHealth(missing, db, ids, gyro).reason).toMatch(/1 of 30 keyframes not registered/);
    // gyro: unverified, breaks, outliers
    expect(liveModelHealth(rec, db, ids, null).healthy).toBe(false);
    expect(liveModelHealth(rec, db, ids, { ...gyro, breaksWithoutOutliers: 1 }).reason).toMatch(/gyro break/);
    expect(liveModelHealth(rec, db, ids, { ...gyro, outliers: [ids[3]] }).reason).toMatch(/disagree with the gyro/);
    // keyframes registered with the gyro-predicted rotation: the gyro checks are no evidence for them
    // (one the model no longer registers, or an unknown id, does not count)
    expect(liveModelHealth(rec, db, ids, gyro, {}, new Set([ids[3], ids[9]])).reason).toMatch(/2 keyframe\(s\) registered with the gyro-predicted rotation/);
    expect(liveModelHealth(missing, db, ids.filter((id) => id !== ids[7]), gyro, {}, [ids[7], 999]).healthy).toBe(true);
    // too few images to judge
    expect(liveModelHealth(rec, db, ids, gyro, { minImages: 40 }).healthy).toBe(false);
  }, 120_000);
});
