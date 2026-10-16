/**
 * Adversarial tests for the TSDF mesher (review pass): block allocation completeness for
 * scenes with a wide depth range, memory cap with a user voxel size, tile-culling
 * conservativeness, face winding, cancellation in the middle of a run.
 */
import { describe, expect, it } from 'vitest';
import type { Pose, TriangleMesh } from '../core/types';
import { BLOCK_SIZE, meshComponents, meshEdgeStats, removeSmallComponents } from './marching-cubes';
import { cameraCenter, lookAt, orbitPoses, pinholeIntrinsics, type SynthScene } from './synthetic';
import { distanceToScene, renderGtViews } from './testutil-scene';
import { buildTsdfVolume, integrateDepthMap, tsdfMesh, TsdfVolume, type TsdfStats } from './tsdf';
import { MemoryDepthMapStore, type DenseView, type DepthMap } from './types';
import { Rng } from '../math/random';

const log = (...a: unknown[]): void => console.log('[tsdf-adv]', ...a);

/** Near box covering most of the image + far wall: median depth ≈ near, far wall ≈ 15× further. */
function nearFarScene(): SynthScene {
  return {
    primitives: [
      { kind: 'box', center: [0, 1.6, 0.1], halfSize: [0.62, 0.2, 0.46], yaw: 0.15, material: { scale: 0.2 } },
      { kind: 'quad', center: [0, 25, 0], axisU: [1, 0, 0], axisV: [0, 0, 1], halfU: 40, halfV: 30, material: { scale: 3 } },
    ],
  };
}

/** World point of pixel (x, y) at z-depth d. */
function backproject(v: DenseView, x: number, y: number, d: number): number[] {
  const c = v.cam;
  const R = v.pose.R;
  const C = cameraCenter(v.pose);
  const rx = ((x - c.cx) / c.f) * d;
  const ry = ((y - c.cy) / c.f) * d;
  return [0, 1, 2].map((a) => C[a] + R[a] * rx + R[3 + a] * ry + R[6 + a] * d);
}

function blockOf(vol: TsdfVolume, X: number[]): number {
  const b = X.map((x) => Math.floor(Math.floor(x / vol.voxelSize + 0.5) / BLOCK_SIZE));
  return vol.findBlock(b[0], b[1], b[2]);
}

/** Fraction of GT surface samples with a mesh vertex within `r`. */
function meshCoverage(mesh: TriangleMesh, samples: number[][], r: number): number {
  const grid = new Map<string, number[]>();
  for (let i = 0; i < mesh.vertexCount; i++) {
    const k = `${Math.floor(mesh.positions[3 * i] / r)},${Math.floor(mesh.positions[3 * i + 1] / r)},${Math.floor(mesh.positions[3 * i + 2] / r)}`;
    let l = grid.get(k);
    if (!l) grid.set(k, (l = []));
    l.push(i);
  }
  let hit = 0;
  for (const p of samples) {
    const cx = Math.floor(p[0] / r);
    const cy = Math.floor(p[1] / r);
    const cz = Math.floor(p[2] / r);
    let ok = false;
    for (let dx = -1; dx <= 1 && !ok; dx++)
      for (let dy = -1; dy <= 1 && !ok; dy++)
        for (let dz = -1; dz <= 1 && !ok; dz++) {
          const l = grid.get(`${cx + dx},${cy + dy},${cz + dz}`);
          if (!l) continue;
          for (const i of l) {
            if (Math.hypot(mesh.positions[3 * i] - p[0], mesh.positions[3 * i + 1] - p[1], mesh.positions[3 * i + 2] - p[2]) <= r) {
              ok = true;
              break;
            }
          }
        }
    if (ok) hit++;
  }
  return samples.length ? hit / samples.length : 1;
}

describe('TSDF allocation (adversarial)', () => {
  it('every valid pixel of a wide-depth-range scene gets its surface block allocated (near object + wall 15× further)', async () => {
    const scene = nearFarScene();
    const cam = pinholeIntrinsics(160, 120, (60 * Math.PI) / 180);
    const poses: Pose[] = [-0.15, 0, 0.15].map((dx) => lookAt([dx, 0, 0.1], [0, 10, 0]));
    const gt = renderGtViews(scene, cam, poses);
    const stats: Partial<TsdfStats> = {};
    const vol = await buildTsdfVolume(gt.views, gt.store, {}, undefined, undefined, stats);
    let valid = 0;
    let missNear = 0;
    let missFar = 0;
    let far = 0;
    for (let v = 0; v < gt.views.length; v++) {
      const view = gt.views[v];
      const d = gt.gtDepth[v];
      for (let y = 0; y < cam.height; y++) {
        for (let x = 0; x < cam.width; x++) {
          const z = d[y * cam.width + x];
          if (!(z > 0)) continue;
          valid++;
          const isFar = z > 10;
          if (isFar) far++;
          if (blockOf(vol, backproject(view, x, y, z)) < 0) {
            if (isFar) missFar++;
            else missNear++;
          }
        }
      }
    }
    log(
      `near+far: voxel ${stats.voxelSize!.toPrecision(3)} (auto ${stats.autoVoxelSize!.toPrecision(3)}, ${stats.coarsenLevels} levels), median depth ${stats.medianDepth!.toFixed(2)}, ` +
        `${stats.blocks} blocks; surface blocks missing: near ${missNear}, far ${missFar}/${far} of ${valid} valid pixels`,
    );
    expect(far).toBeGreaterThan(1000);
    expect(missNear).toBe(0);
    expect(missFar).toBe(0);
  });

  it('far wall is meshed without holes (mesh covers the GT surface of the far wall)', async () => {
    const scene = nearFarScene();
    const cam = pinholeIntrinsics(160, 120, (60 * Math.PI) / 180);
    const poses: Pose[] = [-0.15, 0, 0.15].map((dx) => lookAt([dx, 0, 0.1], [0, 10, 0]));
    const gt = renderGtViews(scene, cam, poses);
    let stats: TsdfStats | null = null;
    const mesh = await tsdfMesh(gt.views, gt.store, { minComponentFraction: 0, onStats: (s) => (stats = s) });
    const s = stats!;
    const farSamples: number[][] = [];
    const v = 1;
    for (let y = 2; y < cam.height - 2; y += 3) {
      for (let x = 2; x < cam.width - 2; x += 3) {
        const z = gt.gtDepth[v][y * cam.width + x];
        if (z > 10) farSamples.push(backproject(gt.views[v], x, y, z));
      }
    }
    // a pixel footprint at 25 m is larger than a voxel: allow the footprint as the radius
    const r = Math.max(2 * s.voxelSize, (25 / cam.f) * 1.5);
    const cov = meshCoverage(mesh, farSamples, r);
    log(`far wall coverage ${(100 * cov).toFixed(1)} % of ${farSamples.length} samples within ${r.toFixed(3)} (voxel ${s.voxelSize.toPrecision(3)}, ${s.blocks} blocks, ${s.triangles} triangles)`);
    expect(cov).toBeGreaterThan(0.97);
  });

  it('ground plane seen from eye height (grazing rays) is meshed as ONE connected sheet', async () => {
    const scene: SynthScene = {
      primitives: [{ kind: 'quad', center: [0, 30, 0], axisU: [1, 0, 0], axisV: [0, 1, 0], halfU: 30, halfV: 30, material: { scale: 1.5, octaves: 8 } }],
    };
    const cam = pinholeIntrinsics(192, 144, (65 * Math.PI) / 180);
    const poses: Pose[] = [];
    for (let i = 0; i < 8; i++) poses.push(lookAt([-2 + 0.5 * i, 0, 1.6], [-1 + 0.3 * i, 20, 0.8]));
    const gt = renderGtViews(scene, cam, poses, { noise: 0.001, normalNoiseDeg: 2, seed: 8 });
    let stats: TsdfStats | null = null;
    const mesh = await tsdfMesh(gt.views, gt.store, { minComponentFraction: 0, onStats: (s) => (stats = s) });
    const s = stats!;
    const { sizes } = meshComponents(mesh);
    const largest = Math.max(...sizes);
    // GT ground samples (pixels of view 4 up to 25 m) covered by the mesh
    const samples: number[][] = [];
    const v = 4;
    for (let y = 1; y < cam.height; y += 2)
      for (let x = 1; x < cam.width; x += 2) {
        const z = gt.gtDepth[v][y * cam.width + x];
        if (z > 0 && z < 25) samples.push(backproject(gt.views[v], x, y, z));
      }
    const cov = meshCoverage(mesh, samples, 1.5 * s.voxelSize);
    // the same samples covered by the LARGEST component only (connectivity of the ground sheet)
    const main = removeSmallComponents(mesh, 0.999).mesh;
    const covMain = meshCoverage(main, samples, 1.5 * s.voxelSize);
    log(
      `grazing ground: voxel ${s.voxelSize.toPrecision(3)}, ${s.blocks} blocks, F ${mesh.triangleCount}, ${sizes.length} components, ` +
        `largest ${((100 * largest) / mesh.triangleCount).toFixed(1)} %; ground ≤ 25 m covered ${(100 * cov).toFixed(1)} %, by the largest component ${(100 * covMain).toFixed(1)} %`,
    );
    expect(cov).toBeGreaterThan(0.98);
    expect(covMain).toBeGreaterThan(0.95);
  });

  it('a user voxel size far below the auto size respects maxBlocks quickly (no unbounded allocation maps)', async () => {
    const scene: SynthScene = { primitives: [{ kind: 'sphere', center: [0, 0, 0], radius: 0.5, material: { scale: 0.2 } }] };
    const cam = pinholeIntrinsics(96, 72, (60 * Math.PI) / 180);
    const gt = renderGtViews(scene, cam, orbitPoses([0, 0, 0], 2, 0.3, 6));
    let stats: TsdfStats | null = null;
    const t0 = performance.now();
    // auto voxel ≈ 0.05; 1e-4 would need ~10^8 blocks
    const mesh = await tsdfMesh(gt.views, gt.store, { voxelSize: 1e-4, maxBlocks: 300, onStats: (s) => (stats = s) });
    const ms = performance.now() - t0;
    log(`tiny user voxel: ${stats!.blocks} blocks, cap hit ${stats!.blockCapHit}, ${ms.toFixed(0)} ms, ${mesh.triangleCount} triangles`);
    expect(stats!.blocks).toBeLessThanOrEqual(300);
    expect(stats!.blockCapHit).toBe(true);
    expect(ms).toBeLessThan(15000);
  });
});

/**
 * Brute-force count of the voxels `integrateDepthMap` must update (same per-voxel rules as
 * the implementation, no block culling; refDepth = 1, maxWeight large). Mirrors: plane
 * distance for cos θ ≥ 0.1 with along-ray reach max(τ_obs, min(10 τ_obs, 1.5 voxel / cos θ)),
 * ray distance (reach τ_obs) otherwise; fall-off behind −τ/4 drops an update only when the
 * rounded weight reaches 0.
 */
function bruteForceUpdated(vol: TsdfVolume, dm: DepthMap, cam: { f: number; cx: number; cy: number }, pose: Pose, tau: number, kPx: number, planeMode: boolean): number {
  const vs = vol.voxelSize;
  let n = 0;
  for (let b = 0; b < vol.blockCount; b++) {
    for (let lz = 0; lz < 8; lz++)
      for (let ly = 0; ly < 8; ly++)
        for (let lx = 0; lx < 8; lx++) {
          const P = [(8 * vol.blockCoords[3 * b] + lx) * vs, (8 * vol.blockCoords[3 * b + 1] + ly) * vs, (8 * vol.blockCoords[3 * b + 2] + lz) * vs];
          const R = pose.R;
          const t = pose.t;
          const X = R[0] * P[0] + R[1] * P[1] + R[2] * P[2] + t[0];
          const Y = R[3] * P[0] + R[4] * P[1] + R[5] * P[2] + t[1];
          const Z = R[6] * P[0] + R[7] * P[1] + R[8] * P[2] + t[2];
          if (!(Z > 1e-9)) continue;
          const u = (cam.f * X) / Z + cam.cx;
          const v = (cam.f * Y) / Z + cam.cy;
          if (!(u >= -0.5 && v >= -0.5 && u < dm.width - 0.5 && v < dm.height - 0.5)) continue;
          const px = Math.floor(u + 0.5);
          const py = Math.floor(v + 0.5);
          const q = py * dm.width + px;
          const d = dm.depth[q];
          if (!(d > 0)) continue;
          const tauObs = Math.max(tau, (kPx * d) / cam.f);
          const rqx = (px - cam.cx) / cam.f;
          const rqy = (py - cam.cy) / cam.f;
          const L = Math.sqrt(rqx * rqx + rqy * rqy + 1);
          const nx = dm.normal[3 * q];
          const ny = dm.normal[3 * q + 1];
          const nz = dm.normal[3 * q + 2];
          const nr = nx * rqx + ny * rqy + nz;
          const cosN = -nr / L;
          const plane = planeMode && cosN >= 0.1;
          const reach = plane ? Math.max(tauObs, Math.min(10 * tauObs, (1.5 * vs) / cosN)) : tauObs;
          if ((Z - d) * L > reach) continue;
          const sdf = plane ? nx * X + ny * Y + nz * Z - d * nr : (d - Z) * L;
          if (sdf < -tauObs) continue;
          const tv = Math.min(1, sdf / tauObs);
          if (tv < -0.25) {
            let cosT = 1;
            if (plane) cosT = cosN;
            else if (nx !== 0 || ny !== 0 || nz !== 0) cosT = Math.max(0.1, Math.abs(nr) / L);
            const wObs = Math.min(64, cosT / (d * d));
            const wq = Math.max(1, (wObs * 64 + 0.5) | 0);
            if (((wq * (1 + tv)) / 0.75 + 0.5) | 0) n++;
            continue;
          }
          n++;
        }
  }
  return n;
}

describe('TSDF integration culling (adversarial)', () => {
  it('block culling is conservative for a 120° FOV camera and a depth map with large per-tile depth jumps', () => {
    const W = 96;
    const H = 72;
    const f = W / 2 / Math.tan(Math.PI / 3); // 120° hfov
    const cam = { width: W, height: H, f, cx: (W - 1) / 2, cy: (H - 1) / 2, k1: 0, k2: 0 };
    const pose = lookAt([0, 0, 0], [0, 1, 0]);
    const rng = new Rng(4);
    const dm: DepthMap = { width: W, height: H, depth: new Float32Array(W * H), normal: new Float32Array(3 * W * H), cost: new Float32Array(W * H) };
    // checkerboard of 16×16 tiles alternating near / far with jitter, a few invalid tiles
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const tx = (x / 16) | 0;
        const ty = (y / 16) | 0;
        const k = (tx * 7 + ty * 3) % 5;
        dm.depth[y * W + x] = k === 4 ? 0 : k % 2 ? 0.8 + 0.2 * rng.next() : 4 + 2 * rng.next();
      }
    const vs = 0.05;
    const vol = new TsdfVolume(vs, 4 * vs, 1 << 20, false, 4096);
    // all blocks of a slab in front of the camera (wide lateral extent: off-axis blocks)
    for (let bz = -6; bz <= 6; bz++) for (let by = 1; by <= 16; by++) for (let bx = -24; bx <= 24; bx++) vol.allocateBlock(bx, by, bz);
    const params = { truncation: 4 * vs, adaptiveTruncationPx: 2, sdfMode: 'ray' as const, refDepth: 1, maxDepth: Infinity, maxCost: 2, maxWeight: 256 };
    const updated = integrateDepthMap(vol, dm, cam, pose, null, params);
    const expected = bruteForceUpdated(vol, dm, cam, pose, 4 * vs, 2, false);
    log(`culling: updated ${updated}, brute force ${expected} (${vol.blockCount} blocks)`);
    expect(updated).toBe(expected);
  });

  it('block culling is conservative in plane mode with grazing normals (extended reach behind the surface)', () => {
    const W = 96;
    const H = 72;
    const f = W / 2 / Math.tan(Math.PI / 3);
    const cam = { width: W, height: H, f, cx: (W - 1) / 2, cy: (H - 1) / 2, k1: 0, k2: 0 };
    const pose = lookAt([0, 0, 0], [0, 1, 0]);
    const rng = new Rng(6);
    const dm: DepthMap = { width: W, height: H, depth: new Float32Array(W * H), normal: new Float32Array(3 * W * H), cost: new Float32Array(W * H) };
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const tx = (x / 16) | 0;
        const ty = (y / 16) | 0;
        dm.depth[i] = (tx + ty) % 2 ? 0.8 + 0.2 * rng.next() : 2 + 2 * rng.next();
        // normal at a random angle to the pixel ray, facing the camera; a third nearly grazing
        const rx = (x - cam.cx) / f;
        const ry = (y - cam.cy) / f;
        const L = Math.hypot(rx, ry, 1);
        const r = [rx / L, ry / L, 1 / L];
        const cosT = rng.next() < 0.33 ? 0.1 + 0.1 * rng.next() : 0.2 + 0.8 * rng.next();
        // a unit vector perpendicular to r
        let p = [r[1], -r[0], 0];
        const lp = Math.hypot(p[0], p[1], p[2]) || 1;
        p = p.map((c) => c / lp);
        const a = 2 * Math.PI * rng.next();
        const q = [r[1] * p[2] - r[2] * p[1], r[2] * p[0] - r[0] * p[2], r[0] * p[1] - r[1] * p[0]];
        const sinT = Math.sqrt(1 - cosT * cosT);
        for (let k = 0; k < 3; k++) dm.normal[3 * i + k] = -cosT * r[k] + sinT * (Math.cos(a) * p[k] + Math.sin(a) * q[k]);
      }
    const vs = 0.05;
    const vol = new TsdfVolume(vs, 4 * vs, 1 << 20, false, 4096);
    for (let bz = -5; bz <= 5; bz++) for (let by = 1; by <= 14; by++) for (let bx = -20; bx <= 20; bx++) vol.allocateBlock(bx, by, bz);
    const params = { truncation: 4 * vs, adaptiveTruncationPx: 2, sdfMode: 'plane' as const, refDepth: 1, maxDepth: Infinity, maxCost: 2, maxWeight: 1e9 };
    const updated = integrateDepthMap(vol, dm, cam, pose, null, params);
    const expected = bruteForceUpdated(vol, dm, cam, pose, 4 * vs, 2, true);
    log(`culling (plane mode): updated ${updated}, brute force ${expected}`);
    expect(updated).toBe(expected);
  });
});

describe('TSDF orientation and cancellation (adversarial)', () => {
  it('face winding is outward (CCW seen from the cameras) on a TSDF sphere, not only the vertex normals', async () => {
    const scene: SynthScene = { primitives: [{ kind: 'sphere', center: [0.1, 0, 0], radius: 0.5, material: { scale: 0.2 } }] };
    const cam = pinholeIntrinsics(96, 72, (60 * Math.PI) / 180);
    const poses: Pose[] = [];
    for (const el of [-50, -15, 15, 50]) poses.push(...orbitPoses([0.1, 0, 0], 2, (el * Math.PI) / 180, 6, el * 0.01));
    const gt = renderGtViews(scene, cam, poses);
    const mesh = await tsdfMesh(gt.views, gt.store);
    let neg = 0;
    const P = mesh.positions;
    for (let t = 0; t < mesh.triangleCount; t++) {
      const [a, b, c] = [0, 1, 2].map((k) => 3 * mesh.indices[3 * t + k]);
      const e1 = [P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]];
      const e2 = [P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]];
      const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      const g = [(P[a] + P[b] + P[c]) / 3 - 0.1, (P[a + 1] + P[b + 1] + P[c + 1]) / 3, (P[a + 2] + P[b + 2] + P[c + 2]) / 3];
      if (n[0] * g[0] + n[1] * g[1] + n[2] * g[2] < 0) neg++;
    }
    expect(mesh.triangleCount).toBeGreaterThan(500);
    expect(neg).toBe(0);
    expect(meshEdgeStats(mesh).misoriented).toBe(0);
  });

  it('cancelling from the progress callback in the middle of integration rejects with CancelledError', async () => {
    const scene: SynthScene = { primitives: [{ kind: 'sphere', center: [0, 0, 0], radius: 0.5, material: { scale: 0.2 } }] };
    const cam = pinholeIntrinsics(64, 48, 1);
    const gt = renderGtViews(scene, cam, orbitPoses([0, 0, 0], 2, 0.3, 6));
    const token = { cancelled: false };
    const msgs: string[] = [];
    const p = tsdfMesh(gt.views, gt.store, {}, (e) => {
      msgs.push(e.message ?? '');
      if ((e.message ?? '').startsWith('integrating 3/')) token.cancelled = true;
    }, token);
    await expect(p).rejects.toMatchObject({ name: 'CancelledError' });
    expect(msgs.some((m) => m.startsWith('integrating 3/'))).toBe(true);
    expect(msgs.some((m) => m.startsWith('integrating 4/'))).toBe(false);
  });

  it('Infinity / NaN depths and NaN normals in the input are ignored (no hang, no budget blow-up)', async () => {
    const scene: SynthScene = { primitives: [{ kind: 'sphere', center: [0, 0, 0], radius: 0.5, material: { scale: 0.2 } }] };
    const cam = pinholeIntrinsics(64, 48, (60 * Math.PI) / 180);
    const gt = renderGtViews(scene, cam, orbitPoses([0, 0, 0], 2, 0.3, 8));
    const store = new MemoryDepthMapStore();
    for (let v = 0; v < 8; v++) {
      const dm = (await gt.store.get(v))!;
      for (let k = 0; k < dm.depth.length; k += 11) {
        if (k % 3 === 0) dm.depth[k] = Infinity;
        else if (k % 3 === 1) dm.depth[k] = NaN;
        else dm.normal[3 * k + 1] = NaN;
      }
      await store.put(v, dm);
    }
    let stats: TsdfStats | null = null;
    const t0 = performance.now();
    const mesh = await tsdfMesh(gt.views, store, { onStats: (s) => (stats = s) });
    const ms = performance.now() - t0;
    let maxErr = 0;
    for (let i = 0; i < mesh.vertexCount; i++) {
      const r = Math.hypot(mesh.positions[3 * i], mesh.positions[3 * i + 1], mesh.positions[3 * i + 2]);
      maxErr = Math.max(maxErr, Math.abs(r - 0.5));
    }
    log(`non-finite input: ${ms.toFixed(0)} ms, voxel ${stats!.voxelSize.toPrecision(3)}, ${stats!.coarsenLevels} levels, ${stats!.blocks} blocks, F ${mesh.triangleCount}, max radius error ${(maxErr / stats!.voxelSize).toFixed(2)} vox`);
    expect(stats!.coarsenLevels).toBe(0);
    expect(Number.isFinite(stats!.medianDepth)).toBe(true);
    expect(mesh.triangleCount).toBeGreaterThan(500);
    expect(maxErr).toBeLessThan(stats!.voxelSize);
  });

  it('empty store / no views: empty mesh, no throw', async () => {
    const mesh = await tsdfMesh([], new MemoryDepthMapStore());
    expect(mesh.triangleCount).toBe(0);
    void distanceToScene;
  });
});

describe('TSDF small-component removal (adversarial)', () => {
  it('a detached object in a large scene survives the default fraction rule; a small blob does not', async () => {
    // huge wall (largest component) + a separate 1 m sphere (≪ 2 % of the wall's triangles)
    // + a tiny floating sphere (noise-blob sized)
    const scene: SynthScene = {
      primitives: [
        { kind: 'quad', center: [0, 12, 0], axisU: [1, 0, 0], axisV: [0, 0, 1], halfU: 40, halfV: 20, material: { scale: 1 } },
        { kind: 'sphere', center: [1.5, 7, 0], radius: 0.7, material: { scale: 0.3 } },
        { kind: 'sphere', center: [-1.5, 6, 0.4], radius: 0.15, material: { scale: 0.05 } },
      ],
    };
    const cam = pinholeIntrinsics(240, 180, (90 * Math.PI) / 180);
    const poses: Pose[] = [-1, -0.5, 0, 0.5, 1].map((x) => lookAt([x, 0, 0.2], [0, 10, 0]));
    const gt = renderGtViews(scene, cam, poses);
    let stats: TsdfStats | null = null;
    const mesh = await tsdfMesh(gt.views, gt.store, { voxelSize: 0.1, maxBlocks: 20000, onStats: (s) => (stats = s) });
    const all = await tsdfMesh(gt.views, gt.store, { voxelSize: 0.1, maxBlocks: 20000, minComponentFraction: 0 });
    const { sizes } = meshComponents(all);
    log(`component sizes (no removal): ${[...sizes].sort((x, y) => y - x).slice(0, 5).join(', ')}`);
    const near = (cx: number, cy: number, cz: number, r: number): number => {
      let n = 0;
      for (let i = 0; i < mesh.vertexCount; i++) if (Math.hypot(mesh.positions[3 * i] - cx, mesh.positions[3 * i + 1] - cy, mesh.positions[3 * i + 2] - cz) < r) n++;
      return n;
    };
    const big = near(1.5, 7, 0, 0.9);
    const tiny = near(-1.5, 6, 0.4, 0.3);
    log(`components: voxel ${stats!.voxelSize.toPrecision(3)}, removed ${stats!.removedComponents}, sphere vertices ${big}, blob vertices ${tiny}, F ${mesh.triangleCount}`);
    expect(big).toBeGreaterThan(100);
    expect(tiny).toBe(0);
  });
});
