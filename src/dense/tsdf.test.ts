import { describe, expect, it } from 'vitest';
import type { Pose, TriangleMesh } from '../core/types';
import { KdTree3 } from './kdtree';
import { meshComponents, meshEdgeStats } from './marching-cubes';
import { lookAt, makeFarScene, makeNearScene, orbitPoses, pinholeIntrinsics, type SynthScene } from './synthetic';
import { distanceToScene, renderGtViews } from './testutil-scene';
import { tsdfMesh, type TsdfStats } from './tsdf';
import { MemoryDepthMapStore } from './types';

const log = (...a: unknown[]): void => console.log('[tsdf]', ...a);

/** Cameras on rings at several elevations around `target` (all-around coverage incl. top/bottom). */
function shellPoses(target: number[], dist: number, elevationsDeg: number[], perRing: number): Pose[] {
  const poses: Pose[] = [];
  // orbitPoses' radius is the 3-D camera–target distance
  elevationsDeg.forEach((el, k) => poses.push(...orbitPoses(target, dist, (el * Math.PI) / 180, perRing, k * 0.4)));
  return poses;
}

function vertexErrors(mesh: TriangleMesh, dist: (x: number, y: number, z: number) => number): { mean: number; p95: number; max: number } {
  const e = new Float64Array(mesh.vertexCount);
  let s = 0;
  for (let i = 0; i < mesh.vertexCount; i++) {
    e[i] = dist(mesh.positions[3 * i], mesh.positions[3 * i + 1], mesh.positions[3 * i + 2]);
    s += e[i];
  }
  e.sort();
  return { mean: s / Math.max(1, e.length), p95: e[Math.floor(0.95 * (e.length - 1))] ?? NaN, max: e[e.length - 1] ?? NaN };
}

function fmtStats(s: TsdfStats): string {
  return (
    `voxel ${s.voxelSize.toPrecision(3)} (auto ${s.autoVoxelSize.toPrecision(3)}, ${s.coarsenLevels} coarsen levels), τ ${s.truncation.toPrecision(3)}, ` +
    `${s.blocks} blocks (${(s.volumeBytes / 1e6).toFixed(1)} MB), V ${s.vertices} F ${s.triangles}, ` +
    `alloc ${s.allocateMs.toFixed(0)} ms, integrate ${s.integrateMs.toFixed(0)} ms, mesh ${s.meshMs.toFixed(0)} ms, total ${s.totalMs.toFixed(0)} ms`
  );
}

async function sphereRun(scale: number, opts: Parameters<typeof tsdfMesh>[2] = {}, noise = 0): Promise<{ mesh: TriangleMesh; stats: TsdfStats; R: number; c: number[] }> {
  const R = 0.5 * scale;
  const c = [0.05 * scale, -0.03 * scale, 0.02 * scale];
  const scene: SynthScene = { primitives: [{ kind: 'sphere', center: c, radius: R, material: { scale: 0.2 * scale } }] };
  const cam = pinholeIntrinsics(160, 120, (60 * Math.PI) / 180);
  const poses = shellPoses(c, 2 * scale, [-55, -18, 18, 55], 8);
  const gt = renderGtViews(scene, cam, poses, { noise, normalNoiseDeg: noise > 0 ? 3 : 0, seed: 3 });
  let stats: TsdfStats | null = null;
  const mesh = await tsdfMesh(gt.views, gt.store, { ...opts, onStats: (s) => (stats = s) });
  return { mesh, stats: stats!, R, c };
}

describe('tsdfMesh', () => {
  it('sphere from 32 GT depth maps: closed mesh, radius error < voxel/2', async () => {
    const { mesh, stats, R, c } = await sphereRun(1);
    const st = meshEdgeStats(mesh);
    const err = vertexErrors(mesh, (x, y, z) => Math.abs(Math.hypot(x - c[0], y - c[1], z - c[2]) - R));
    log(`sphere r=0.5 @2 m: ${fmtStats(stats)}`);
    log(`  radius error mean ${(err.mean / stats.voxelSize).toFixed(3)} vox, p95 ${(err.p95 / stats.voxelSize).toFixed(3)}, max ${(err.max / stats.voxelSize).toFixed(3)} vox; edges ${JSON.stringify(st)}`);
    expect(stats.coarsenLevels).toBe(0);
    expect(st.boundary).toBe(0);
    expect(st.nonManifold).toBe(0);
    expect(st.misoriented).toBe(0);
    expect(mesh.vertexCount - st.edges + mesh.triangleCount).toBe(2);
    expect(meshComponents(mesh).sizes.length).toBe(1);
    expect(err.max).toBeLessThan(0.5 * stats.voxelSize);
    expect(err.mean).toBeLessThan(0.1 * stats.voxelSize);
    // normals point outwards (towards the cameras)
    let minDot = 1;
    for (let i = 0; i < mesh.vertexCount; i++) {
      const x = mesh.positions[3 * i] - c[0];
      const y = mesh.positions[3 * i + 1] - c[1];
      const z = mesh.positions[3 * i + 2] - c[2];
      const r = Math.hypot(x, y, z);
      minDot = Math.min(minDot, (mesh.normals[3 * i] * x + mesh.normals[3 * i + 1] * y + mesh.normals[3 * i + 2] * z) / r);
    }
    expect(minDot).toBeGreaterThan(0.95);
    // colours come from the images (not the default grey / black)
    let sum = 0;
    for (let i = 0; i < mesh.colors.length; i++) sum += mesh.colors[i];
    expect(sum / mesh.colors.length).toBeGreaterThan(20);
  });

  it('same sphere at ×50 scale (25 m radius, 100 m away): scale-invariant result', async () => {
    const { mesh, stats, R, c } = await sphereRun(50);
    const st = meshEdgeStats(mesh);
    const err = vertexErrors(mesh, (x, y, z) => Math.abs(Math.hypot(x - c[0], y - c[1], z - c[2]) - R));
    log(`sphere r=25 @100 m: ${fmtStats(stats)}; radius error mean ${(err.mean / stats.voxelSize).toFixed(3)} vox, max ${(err.max / stats.voxelSize).toFixed(3)} vox`);
    expect(stats.voxelSize).toBeGreaterThan(0.5);
    expect(st.boundary).toBe(0);
    expect(st.nonManifold).toBe(0);
    expect(err.max).toBeLessThan(0.5 * stats.voxelSize);
  });

  it('noisy depth maps (σ = 0.3 %) are averaged out', async () => {
    const { mesh, stats, R, c } = await sphereRun(1, {}, 0.003);
    const st = meshEdgeStats(mesh);
    const err = vertexErrors(mesh, (x, y, z) => Math.abs(Math.hypot(x - c[0], y - c[1], z - c[2]) - R));
    log(`noisy sphere: voxel ${stats.voxelSize.toPrecision(3)}, single-depth σ ≈ ${(0.003 * 1.6).toFixed(4)}; error mean ${err.mean.toFixed(5)} (${(err.mean / stats.voxelSize).toFixed(3)} vox), p95 ${(err.p95 / stats.voxelSize).toFixed(3)} vox, max ${(err.max / stats.voxelSize).toFixed(3)} vox, components ${meshComponents(mesh).sizes.length}`);
    expect(st.nonManifold).toBe(0);
    expect(err.mean).toBeLessThan(0.15 * stats.voxelSize);
    expect(err.p95).toBeLessThan(0.5 * stats.voxelSize);
  });

  it('block budget forces automatic coarsening and still yields a closed mesh', async () => {
    const { mesh, stats, R, c } = await sphereRun(1, { maxBlocks: 120 });
    const st = meshEdgeStats(mesh);
    const err = vertexErrors(mesh, (x, y, z) => Math.abs(Math.hypot(x - c[0], y - c[1], z - c[2]) - R));
    log(`budget 120 blocks: ${fmtStats(stats)}; radius error max ${(err.max / stats.voxelSize).toFixed(3)} vox`);
    expect(stats.coarsenLevels).toBeGreaterThan(0);
    expect(stats.blocks).toBeLessThanOrEqual(120);
    expect(stats.blockCapHit).toBe(false);
    expect(st.boundary).toBe(0);
    expect(err.max).toBeLessThan(0.5 * stats.voxelSize);
  });

  it('yawed box: faces within voxel/2, closed', async () => {
    const box = { kind: 'box' as const, center: [0.02, -0.01, 0.0], halfSize: [0.4, 0.3, 0.25], yaw: 0.3, material: { scale: 0.2 } };
    const scene: SynthScene = { primitives: [box] };
    const cam = pinholeIntrinsics(160, 120, (60 * Math.PI) / 180);
    const poses = shellPoses(box.center, 2.2, [-55, -18, 18, 55], 8);
    const gt = renderGtViews(scene, cam, poses);
    let stats: TsdfStats | null = null;
    const mesh = await tsdfMesh(gt.views, gt.store, { onStats: (s) => (stats = s) });
    const s = stats!;
    const st = meshEdgeStats(mesh);
    const err = vertexErrors(mesh, (x, y, z) => distanceToScene(scene, x, y, z));
    log(`box: ${fmtStats(s)}; error mean ${(err.mean / s.voxelSize).toFixed(3)} vox, p95 ${(err.p95 / s.voxelSize).toFixed(3)} vox, max ${(err.max / s.voxelSize).toFixed(3)} vox; edges ${JSON.stringify(st)}`);
    expect(st.boundary).toBe(0);
    expect(st.nonManifold).toBe(0);
    // p95 0.48 → 0.54 voxel since grazing views (cos θ ≥ 0.1) use the plane distance (needed for
    // ground planes seen from eye height, see tsdf-adversarial): slightly rounder box edges
    expect(err.p95).toBeLessThan(0.6 * s.voxelSize);
    expect(err.max).toBeLessThan(1.5 * s.voxelSize); // corners are rounded by ~1 voxel
  });

  it('near scene (ground + boxes + sphere), 12 orbit views with noise and outliers', async () => {
    const scene = makeNearScene();
    const cam = pinholeIntrinsics(200, 150, (60 * Math.PI) / 180);
    const poses = orbitPoses([0, 0, 0.25], 2.6, (40 * Math.PI) / 180, 12);
    const gt = renderGtViews(scene, cam, poses, { noise: 0.002, normalNoiseDeg: 3, outliers: 0.02, seed: 11 });
    let stats: TsdfStats | null = null;
    const mesh = await tsdfMesh(gt.views, gt.store, { onStats: (s) => (stats = s) });
    const s = stats!;
    const err = vertexErrors(mesh, (x, y, z) => distanceToScene(scene, x, y, z));
    log(`near scene: ${fmtStats(s)}; error mean ${(err.mean / s.voxelSize).toFixed(3)} vox, p95 ${(err.p95 / s.voxelSize).toFixed(3)} vox, max ${(err.max / s.voxelSize).toFixed(2)} vox`);
    expect(mesh.triangleCount).toBeGreaterThan(1000);
    expect(err.p95).toBeLessThan(0.5 * s.voxelSize);
    expect(err.mean).toBeLessThan(0.2 * s.voxelSize);
  });

  it('depth maps at half the image resolution give the same surface', async () => {
    const R = 0.5;
    const scene: SynthScene = { primitives: [{ kind: 'sphere', center: [0, 0, 0], radius: R, material: { scale: 0.2 } }] };
    const cam = pinholeIntrinsics(240, 180, (60 * Math.PI) / 180);
    const half = { ...cam, width: 120, height: 90, f: cam.f * 0.5, cx: (cam.cx + 0.5) * 0.5 - 0.5, cy: (cam.cy + 0.5) * 0.5 - 0.5 };
    const poses = shellPoses([0, 0, 0], 2, [-55, -18, 18, 55], 6);
    const full = renderGtViews(scene, cam, poses);
    const halfGt = renderGtViews(scene, half, poses);
    let stats: TsdfStats | null = null;
    const mesh = await tsdfMesh(full.views, halfGt.store, { onStats: (s) => (stats = s) });
    const err = vertexErrors(mesh, (x, y, z) => Math.abs(Math.hypot(x, y, z) - R));
    log(`half-res depth: voxel ${stats!.voxelSize.toPrecision(3)}, error max ${(err.max / stats!.voxelSize).toFixed(3)} vox`);
    // the auto voxel follows the DEPTH-map footprint (2 px at 120 px wide)
    expect(stats!.voxelSize).toBeGreaterThan(0.025);
    expect(err.max).toBeLessThan(0.5 * stats!.voxelSize);
    expect(meshEdgeStats(mesh).boundary).toBe(0);
  });

  it('yard scene (houses 10–60 m) from a walk, 320×240: timing and accuracy', async () => {
    const scene = makeFarScene();
    const cam = pinholeIntrinsics(320, 240, (65 * Math.PI) / 180);
    const poses: Pose[] = [];
    for (let i = 0; i < 16; i++) poses.push(lookAt([-10 + i * 1.3, 0, 1.6], [-6 + i * 0.8, 30, 2]));
    const t0 = performance.now();
    const gt = renderGtViews(scene, cam, poses, { noise: 0.002, normalNoiseDeg: 3, seed: 4 });
    const t1 = performance.now();
    let stats: TsdfStats | null = null;
    const mesh = await tsdfMesh(gt.views, gt.store, { onStats: (s) => (stats = s) });
    const s = stats!;
    const err = vertexErrors(mesh, (x, y, z) => distanceToScene(scene, x, y, z));
    log(`yard 16 × 320×240 (render ${(t1 - t0).toFixed(0)} ms): ${fmtStats(s)}; error mean ${(err.mean / s.voxelSize).toFixed(3)} vox, p95 ${(err.p95 / s.voxelSize).toFixed(3)} vox`);
    expect(mesh.triangleCount).toBeGreaterThan(10000);
    expect(err.p95).toBeLessThan(0.75 * s.voxelSize);
  });

  describe('bounds (object box)', () => {
    // a sphere resting on the ground with a box beside it; cameras look down from 20–70°
    const scene: SynthScene = {
      primitives: [
        { kind: 'quad', center: [0, 0, 0], axisU: [1, 0, 0], axisV: [0, 1, 0], halfU: 3, halfV: 3, material: { scale: 0.5, octaves: 7, color: [0.6, 0.55, 0.45], seed: 1 } },
        { kind: 'sphere', center: [0, 0, 0.3], radius: 0.3, material: { scale: 0.2, octaves: 7, color: [0.4, 0.7, 0.4], seed: 2 } },
        { kind: 'box', center: [0.9, 0.6, 0.2], halfSize: [0.2, 0.2, 0.2], yaw: 0.3, material: { scale: 0.2, octaves: 7, color: [0.8, 0.4, 0.3], seed: 3 } },
      ],
    };
    const bounds = { min: [-0.36, -0.36, 0.02], max: [0.36, 0.36, 0.66] };
    const cam = pinholeIntrinsics(160, 120, (60 * Math.PI) / 180);
    const gt = renderGtViews(scene, cam, shellPoses([0, 0, 0.3], 1.8, [20, 45, 70], 10));
    const run = async (opts: Parameters<typeof tsdfMesh>[2]): Promise<{ mesh: TriangleMesh; stats: TsdfStats }> => {
      let stats: TsdfStats | null = null;
      const mesh = await tsdfMesh(gt.views, gt.store, { ...opts, onStats: (s) => (stats = s) });
      return { mesh, stats: stats! };
    };
    const outside = (m: TriangleMesh, pad: number): number => {
      let n = 0;
      for (let i = 0; i < m.vertexCount; i++) {
        for (let a = 0; a < 3; a++) {
          const v = m.positions[3 * i + a];
          if (v < bounds.min[a] - pad || v > bounds.max[a] + pad) {
            n++;
            break;
          }
        }
      }
      return n;
    };
    /** Distance from each vertex of `a` to the nearest vertex of `b`. */
    const nearest = (a: TriangleMesh, b: TriangleMesh, keep: (x: number, y: number, z: number) => boolean): number[] => {
      const tree = new KdTree3(b.positions, b.vertexCount);
      const d = new Float64Array(1);
      const idx = new Int32Array(1);
      const out: number[] = [];
      for (let i = 0; i < a.vertexCount; i++) {
        const [x, y, z] = [a.positions[3 * i], a.positions[3 * i + 1], a.positions[3 * i + 2]];
        if (!keep(x, y, z)) continue;
        tree.knn(x, y, z, 1, d, idx);
        out.push(Math.sqrt(d[0]));
      }
      return out;
    };

    it('no vertex outside the bounds; inside, the unbounded surface within 1 voxel, with far fewer blocks', async () => {
      const auto = await run({});
      const autoBox = await run({ bounds });
      log(`unbounded: ${fmtStats(auto.stats)}`);
      log(`bounded:   ${fmtStats(autoBox.stats)}`);
      // the bounded scene scale: 2 px at the sphere (≈ 1.5–1.8 m), not at the median of ground + sphere + box
      expect(autoBox.stats.medianDepth).toBeLessThan(auto.stats.medianDepth);
      expect(autoBox.stats.voxelSize).toBeLessThan(auto.stats.voxelSize);
      expect(outside(autoBox.mesh, 0)).toBe(0);
      // same voxel: the bounded volume is the unbounded one inside the bounds
      const voxelSize = auto.stats.voxelSize;
      const full = await run({ voxelSize });
      const box = await run({ voxelSize, bounds });
      const v = voxelSize;
      expect(full.stats.blockCapHit).toBe(false);
      expect(outside(full.mesh, v)).toBeGreaterThan(1000); // ground + box
      expect(outside(box.mesh, 0)).toBe(0);
      expect(box.mesh.triangleCount).toBeGreaterThan(2000);
      expect(box.stats.blocks).toBeLessThan(0.5 * full.stats.blocks);
      // bounded vertices lie on the unbounded mesh …
      const dIn = nearest(box.mesh, full.mesh, () => true);
      expect(Math.max(...dIn)).toBeLessThan(v);
      // … which is complete inside the bounds (shrunk by 2 voxels: the cut faces end on cube corners)
      const inner = (x: number, y: number, z: number): boolean => [x, y, z].every((c, a) => c > bounds.min[a] + 2 * v && c < bounds.max[a] - 2 * v);
      const dOut = nearest(full.mesh, box.mesh, inner);
      expect(dOut.length).toBeGreaterThan(1000);
      expect(Math.max(...dOut)).toBeLessThan(v);
      // the sphere, not the ground: every vertex within half a voxel of the scene, none on the ground
      const err = vertexErrors(box.mesh, (x, y, z) => distanceToScene(scene, x, y, z));
      expect(err.p95).toBeLessThan(0.5 * v);
      let low = 0;
      for (let i = 0; i < box.mesh.vertexCount; i++) if (box.mesh.positions[3 * i + 2] < 0.02 + v) low++;
      expect(low / box.mesh.vertexCount).toBeLessThan(0.02);
    });

    it('a block budget that coarsens the whole scene fits the bounded volume at a finer voxel', async () => {
      const full = await run({ maxBlocks: 600 });
      const box = await run({ maxBlocks: 600, bounds });
      log(`budget 600, unbounded: ${fmtStats(full.stats)}`);
      log(`budget 600, bounded:   ${fmtStats(box.stats)}`);
      expect(full.stats.coarsenLevels).toBeGreaterThan(0);
      expect(box.stats.coarsenLevels).toBeLessThan(full.stats.coarsenLevels);
      expect(box.stats.voxelSize).toBeLessThan(full.stats.voxelSize);
      expect(outside(box.mesh, 0)).toBe(0);
    });

    it('an empty or non-finite box gives an empty mesh', async () => {
      for (const b of [
        { min: [0.2, 0, 0], max: [0.1, 1, 1] },
        { min: [NaN, 0, 0], max: [1, 1, 1] },
      ]) {
        const { mesh, stats } = await run({ bounds: b });
        expect(mesh.vertexCount).toBe(0);
        expect(stats.blocks).toBe(0);
      }
    });
  });

  it('no valid depth: empty mesh', async () => {
    const store = new MemoryDepthMapStore();
    const cam = pinholeIntrinsics(32, 24, 1);
    await store.put(0, { width: 32, height: 24, depth: new Float32Array(32 * 24), normal: new Float32Array(3 * 32 * 24), cost: new Float32Array(32 * 24).fill(2) });
    const mesh = await tsdfMesh([{ id: 0, cam, pose: lookAt([0, 0, 0], [0, 1, 0]), image: { width: 32, height: 24, data: new Uint8ClampedArray(32 * 24 * 4) } }], store);
    expect(mesh.triangleCount).toBe(0);
    expect(mesh.vertexCount).toBe(0);
  });

  it('cancellation throws', async () => {
    const scene: SynthScene = { primitives: [{ kind: 'sphere', center: [0, 0, 0], radius: 0.5, material: { scale: 0.2 } }] };
    const cam = pinholeIntrinsics(64, 48, 1);
    const gt = renderGtViews(scene, cam, orbitPoses([0, 0, 0], 2, 0.3, 4));
    await expect(tsdfMesh(gt.views, gt.store, {}, undefined, { cancelled: true })).rejects.toThrow(/cancelled/);
  });
});
