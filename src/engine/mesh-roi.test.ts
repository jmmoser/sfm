/**
 * runMesh with the automatic object box (object mode): the TSDF is bounded to the box that
 * `proposeObjectRoi` finds on the sparse model, so the support plane and the clutter beside the
 * object stay out of the mesh; without gravity alignment, in scene mode or with
 * `MeshOptions.objectRoi: false` the whole scene within the depth cut is meshed as before.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { Pose, TriangleMesh } from '../core/types';
import { lookAt, pinholeIntrinsics, type SynthScene } from '../dense/synthetic';
import { distanceToScene, renderGtViews, type GtViews } from '../dense/testutil-scene';
import { Rng } from '../math/random';
import { runMesh, type MeshDeps } from './dense';
import type { SparseModel } from './types';

// ground ("table"), a 2-part object at the orbit centre, a block at the camera radius, a wall
const scene: SynthScene = {
  primitives: [
    { kind: 'quad', center: [0, 0, 0], axisU: [1, 0, 0], axisV: [0, 1, 0], halfU: 3, halfV: 3, material: { scale: 0.3, octaves: 8, color: [0.62, 0.55, 0.42], seed: 1 } },
    { kind: 'sphere', center: [0, 0, 0.15], radius: 0.15, material: { scale: 0.08, octaves: 8, color: [0.45, 0.75, 0.4], seed: 2 } },
    { kind: 'box', center: [0.22, 0.12, 0.06], halfSize: [0.06, 0.05, 0.06], yaw: 0.4, material: { scale: 0.08, octaves: 8, color: [0.8, 0.42, 0.33], seed: 3 } },
    { kind: 'box', center: [-0.55, -0.55, 0.08], halfSize: [0.06, 0.06, 0.08], yaw: 0.2, material: { scale: 0.08, octaves: 8, color: [0.38, 0.55, 0.8], seed: 4 } },
    { kind: 'quad', center: [0, 2.2, 1], axisU: [1, 0, 0], axisV: [0, 0, 1], halfU: 3, halfV: 1, material: { scale: 0.3, octaves: 8, color: [0.7, 0.7, 0.65], seed: 5 } },
  ],
};
/** AABB of the 2-part object (sphere + yawed box). */
const OBJECT = { min: [-0.15, -0.15, 0], max: [0.285, 0.189, 0.3] };

/** Sparse model: surface samples of the scene (0.5 mm noise), each observed by the 3 cameras that face it most. */
function sparseModel(poses: Pose[], cam: ReturnType<typeof pinholeIntrinsics>): SparseModel {
  const rng = new Rng(3);
  const xyz: number[] = [];
  const add = (x: number, y: number, z: number): void => {
    xyz.push(x + 0.0005 * rng.gaussian(), y + 0.0005 * rng.gaussian(), z + 0.0005 * rng.gaussian());
  };
  for (let i = 0; i < 3000; i++) add(3 * (2 * rng.next() - 1) * 0.4, 3 * (2 * rng.next() - 1) * 0.4, 0);
  for (let i = 0; i < 600; i++) {
    const v = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
    const l = Math.hypot(v[0], v[1], v[2]);
    if (v[2] / l < -0.8) continue; // the underside is not seen
    add((0.15 * v[0]) / l, (0.15 * v[1]) / l, 0.15 + (0.15 * v[2]) / l);
  }
  for (const [c, h, yaw] of [
    [[0.22, 0.12], [0.06, 0.05, 0.06], 0.4],
    [[-0.55, -0.55], [0.06, 0.06, 0.08], 0.2],
  ] as const) {
    const cs = Math.cos(yaw);
    const sn = Math.sin(yaw);
    for (let i = 0; i < 300; i++) {
      const p = [(2 * rng.next() - 1) * h[0], (2 * rng.next() - 1) * h[1], 2 * h[2] * rng.next()];
      const f = rng.int(5);
      if (f < 2) p[0] = f ? h[0] : -h[0];
      else if (f < 4) p[1] = f === 3 ? h[1] : -h[1];
      else p[2] = 2 * h[2];
      add(c[0] + cs * p[0] - sn * p[1], c[1] + sn * p[0] + cs * p[1], p[2]);
    }
  }
  for (let i = 0; i < 400; i++) add(3 * (2 * rng.next() - 1), 2.2, 2 * rng.next());
  const n = xyz.length / 3;
  const TRACK = 3;
  const trackOffsets = new Uint32Array(n + 1);
  const trackImages = new Uint32Array(TRACK * n);
  for (let i = 0; i < n; i++) {
    trackOffsets[i + 1] = TRACK * (i + 1);
    // the cameras that see the point most head-on (largest depth along their axis), best first
    const z = poses.map((p) => p.R[6] * xyz[3 * i] + p.R[7] * xyz[3 * i + 1] + p.R[8] * xyz[3 * i + 2] + p.t[2]);
    const order = poses.map((_, k) => k).sort((a, b) => z[b] - z[a] || a - b);
    for (let k = 0; k < TRACK; k++) trackImages[TRACK * i + k] = order[k];
  }
  return {
    cameras: new Map([[0, cam]]),
    images: poses.map((pose, id) => ({ id, cameraId: 0, pose })),
    numPoints: n,
    xyz: Float64Array.from(xyz),
    colors: new Uint8Array(3 * n),
    errors: new Float32Array(n),
    trackOffsets,
    trackImages,
    trackFeatures: new Uint32Array(TRACK * n),
  };
}

describe('runMesh object box', () => {
  let gt: GtViews;
  let model: SparseModel;
  const deps = (over: Partial<MeshDeps> = {}): MeshDeps => ({ quality: 'fast', mode: 'object', config: {}, model, memoryBudgetMB: 1200, gravityAligned: true, ...over });
  let full: { mesh: TriangleMesh; stats: Record<string, number | string> };

  beforeAll(async () => {
    const cam = pinholeIntrinsics(160, 120, (60 * Math.PI) / 180);
    const poses: Pose[] = [];
    for (let i = 0; i < 18; i++) {
      const a = (2 * Math.PI * i) / 18;
      poses.push(lookAt([0.8 * Math.cos(a), 0.8 * Math.sin(a), 0.12 + (i % 2 ? 0.4 : 0.55)], [0, 0, 0.12]));
    }
    gt = renderGtViews(scene, cam, poses);
    model = sparseModel(poses, cam);
    full = await runMesh(deps(), gt.views, gt.store, { objectRoi: false });
  }, 60_000);

  it('object mode, gravity-aligned: the mesh is the object only, with fewer blocks', async () => {
    const r = await runMesh(deps(), gt.views, gt.store);
    const fmt = (s: Record<string, number | string>, m: TriangleMesh): string =>
      `${m.vertexCount} vertices, voxel ${Number(s.voxelSize).toPrecision(3)}, ${s.tsdfBlocks} blocks, ${s.tsdfCoarsenLevels} coarsen levels, ${s.meshingMs} ms, box ${s.objectRoi}`;
    console.log(`[mesh-roi] bounded:   ${fmt(r.stats, r.mesh)}\n[mesh-roi] unbounded: ${fmt(full.stats, full.mesh)}`);
    const roi = r.objectRoi!;
    expect(roi).not.toBeNull();
    expect(String(r.stats.objectRoi)).toMatch(/^min \(/);
    expect(r.stats.objectRoiConfidence).toBeGreaterThan(0.3);
    expect(full.stats.objectRoi).toBe('off');
    // the box: around both object parts, above the ground, not reaching the block at the camera radius
    expect(Math.abs(roi.planeZ)).toBeLessThan(0.002);
    for (let k = 0; k < 2; k++) {
      expect(roi.min[k]).toBeLessThan(OBJECT.min[k]);
      expect(roi.max[k]).toBeGreaterThan(OBJECT.max[k]);
      expect(roi.min[k]).toBeGreaterThan(-0.45);
    }
    const v = Number(r.stats.voxelSize);
    expect(roi.min[2] - roi.planeZ).toBeGreaterThan(v); // ≥ 2 estimated voxels of clearance
    // every vertex inside the box, on the object (not the ground, the block or the wall)
    const m = r.mesh;
    expect(m.vertexCount).toBeGreaterThan(300);
    let out = 0;
    let err = 0;
    for (let i = 0; i < m.vertexCount; i++) {
      const p = [m.positions[3 * i], m.positions[3 * i + 1], m.positions[3 * i + 2]];
      if (p.some((c, a) => c < roi.min[a] || c > roi.max[a])) out++;
      if (distanceToScene(scene, p[0], p[1], p[2]) > 0.5 * v) err++;
    }
    expect(out).toBe(0);
    expect(err / m.vertexCount).toBeLessThan(0.05);
    expect(Number(r.stats.tsdfBlocks)).toBeLessThan(0.3 * Number(full.stats.tsdfBlocks));
    expect(r.stats.tsdfCoarsenLevels).toBe(0);
    expect(full.mesh.vertexCount).toBeGreaterThan(3 * m.vertexCount);
  });

  it('no box without gravity alignment or in scene mode: the scene within the depth cut, as before', async () => {
    const ng = await runMesh(deps({ gravityAligned: false }), gt.views, gt.store);
    expect(ng.objectRoi).toBeNull();
    expect(ng.stats.objectRoi).toBe('none: model not gravity-aligned');
    expect(ng.mesh.vertexCount).toBe(full.mesh.vertexCount);
    const sc = await runMesh(deps({ mode: 'scene' }), gt.views, gt.store);
    expect(sc.objectRoi).toBeNull();
    expect(sc.stats.objectRoi).toBe('off');
  });

  it('no box from too few object points (a textureless object): the whole scene, not a crop', async () => {
    // keep 60 of the object's sparse points (the 2-part object: above the ground, near the axis)
    const keep: number[] = [];
    let objectKept = 0;
    for (let i = 0; i < model.numPoints; i++) {
      const [x, y, z] = [model.xyz[3 * i], model.xyz[3 * i + 1], model.xyz[3 * i + 2]];
      const onObject = z > 0.005 && Math.hypot(x, y) < 0.4;
      if (onObject && objectKept++ >= 60) continue;
      keep.push(i);
    }
    const n = keep.length;
    const trackOffsets = new Uint32Array(n + 1);
    const trackImages: number[] = [];
    keep.forEach((i, j) => {
      for (let k = model.trackOffsets[i]; k < model.trackOffsets[i + 1]; k++) trackImages.push(model.trackImages[k]);
      trackOffsets[j + 1] = trackImages.length;
    });
    const thin: SparseModel = {
      ...model,
      numPoints: n,
      xyz: Float64Array.from(keep.flatMap((i) => [model.xyz[3 * i], model.xyz[3 * i + 1], model.xyz[3 * i + 2]])),
      colors: new Uint8Array(3 * n),
      errors: new Float32Array(n),
      trackOffsets,
      trackImages: Uint32Array.from(trackImages),
      trackFeatures: new Uint32Array(trackImages.length),
    };
    const r = await runMesh(deps({ model: thin }), gt.views, gt.store);
    expect(r.objectRoi).toBeNull();
    expect(String(r.stats.objectRoi)).toMatch(/^none: no object on the support plane/);
    // (the depth cut follows the thin model's median depth)
    const off = await runMesh(deps({ model: thin }), gt.views, gt.store, { objectRoi: false });
    expect(r.mesh.vertexCount).toBe(off.mesh.vertexCount);
    expect(r.mesh.vertexCount).toBeGreaterThan(10_000);
  });
});
