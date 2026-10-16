/**
 * refine's final principal-point bundle adjustment (src/engine/refine.ts refinePrincipalPoint):
 * a capture whose true principal point is offset from the image centre is reconstructed with the
 * centred guess (the mapper keeps it fixed, as COLMAP does) — every camera then absorbs the offset
 * as a constant rotation; the final BA with the principal point free removes it.
 */
import { describe, expect, it } from 'vitest';
import { runIncrementalSfM } from '../sfm/mapper';
import { buildSyntheticDatabase, evaluateReconstruction, generateScene } from '../sfm/synthetic';
import { refinePrincipalPoint } from './refine';

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
