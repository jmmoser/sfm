import { describe, expect, it } from 'vitest';
import { MC_EDGE_CORNERS, MC_TABLES, marchingCubes, meshComponents, meshEdgeStats, removeSmallComponents } from './marching-cubes';
import { TsdfVolume } from './tsdf';
import type { TriangleMesh } from '../core/types';

const log = (...a: unknown[]): void => console.log('[marching-cubes]', ...a);

function faceNormalDots(mesh: TriangleMesh, dir: (x: number, y: number, z: number) => [number, number, number]): { neg: number; min: number } {
  let neg = 0;
  let min = Infinity;
  const P = mesh.positions;
  for (let t = 0; t < mesh.triangleCount; t++) {
    const a = 3 * mesh.indices[3 * t];
    const b = 3 * mesh.indices[3 * t + 1];
    const c = 3 * mesh.indices[3 * t + 2];
    const e1 = [P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]];
    const e2 = [P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]];
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const l = Math.hypot(n[0], n[1], n[2]);
    if (!(l > 0)) continue;
    const g = dir((P[a] + P[b] + P[c]) / 3, (P[a + 1] + P[b + 1] + P[c + 1]) / 3, (P[a + 2] + P[b + 2] + P[c + 2]) / 3);
    const d = (n[0] * g[0] + n[1] * g[1] + n[2] * g[2]) / l;
    if (d < 0) neg++;
    min = Math.min(min, d);
  }
  return { neg, min };
}

describe('marching cubes tables', () => {
  it('generated tables are consistent', () => {
    expect(MC_EDGE_CORNERS.length).toBe(12);
    const { triStart, triEdges, edgeMask } = MC_TABLES;
    expect(triStart[1] - triStart[0]).toBe(0);
    expect(triStart[256] - triStart[255]).toBe(0);
    let maxTris = 0;
    for (let cs = 0; cs < 256; cs++) {
      // crossing edges = edges whose corners differ
      let mask = 0;
      MC_EDGE_CORNERS.forEach(([a, b], e) => {
        if (((cs >> a) & 1) !== ((cs >> b) & 1)) mask |= 1 << e;
      });
      expect(edgeMask[cs]).toBe(mask);
      expect(edgeMask[255 - cs]).toBe(mask);
      const n = (triStart[cs + 1] - triStart[cs]) / 3;
      maxTris = Math.max(maxTris, n);
      // every crossing edge is used, and only those
      let used = 0;
      for (let k = triStart[cs]; k < triStart[cs + 1]; k++) if (triEdges[k] < 12) used |= 1 << triEdges[k];
      expect(used).toBe(mask);
      // within one case, each loop edge appears once and each fan diagonal twice (closed fans)
      const cnt = new Map<string, number>();
      for (let k = triStart[cs]; k < triStart[cs + 1]; k += 3) {
        for (let j = 0; j < 3; j++) {
          const a = triEdges[k + j];
          const b = triEdges[k + ((j + 1) % 3)];
          const key = `${Math.min(a, b)},${Math.max(a, b)}`;
          cnt.set(key, (cnt.get(key) ?? 0) + 1);
        }
      }
      for (const c of cnt.values()) expect(c === 1 || c === 2).toBe(true);
    }
    const centred = MC_TABLES.centreStart[256];
    log(`max triangles per cube: ${maxTris}, loops needing a centre vertex: ${centred}`);
    expect(maxTris).toBeLessThanOrEqual(6);
  });
});

describe('marchingCubes on analytic SDFs', () => {
  it('sphere: closed, genus 0, consistent outward winding, sub-voxel accurate', () => {
    const R = 1;
    const vs = 0.05;
    const c = [0.013, -0.021, 0.007];
    const sdf = (x: number, y: number, z: number): number => Math.hypot(x - c[0], y - c[1], z - c[2]) - R;
    const t0 = performance.now();
    const vol = TsdfVolume.fromFunction(sdf, [-1.3, -1.3, -1.3], [1.3, 1.3, 1.3], vs, 4, (x, y, z) => [128 + 100 * x, 128 + 100 * y, 128 + 100 * z]);
    const t1 = performance.now();
    const mesh = marchingCubes(vol);
    const t2 = performance.now();
    const st = meshEdgeStats(mesh);
    const euler = mesh.vertexCount - st.edges + mesh.triangleCount;
    let maxErr = 0;
    let sumErr = 0;
    let minNDot = 1;
    let maxColErr = 0;
    for (let i = 0; i < mesh.vertexCount; i++) {
      const x = mesh.positions[3 * i] - c[0];
      const y = mesh.positions[3 * i + 1] - c[1];
      const z = mesh.positions[3 * i + 2] - c[2];
      const r = Math.hypot(x, y, z);
      const e = Math.abs(r - R);
      maxErr = Math.max(maxErr, e);
      sumErr += e;
      const d = (mesh.normals[3 * i] * x + mesh.normals[3 * i + 1] * y + mesh.normals[3 * i + 2] * z) / r;
      minNDot = Math.min(minNDot, d);
      const px = mesh.positions[3 * i];
      maxColErr = Math.max(maxColErr, Math.abs(mesh.colors[3 * i] - (128 + 100 * px)));
    }
    const fd = faceNormalDots(mesh, (x, y, z) => [x - c[0], y - c[1], z - c[2]]);
    log(
      `sphere R=1 voxel=${vs}: ${vol.blockCount} blocks, V=${mesh.vertexCount} F=${mesh.triangleCount} E=${st.edges} χ=${euler}, ` +
        `radius err mean ${(sumErr / mesh.vertexCount / vs).toFixed(4)} vox max ${(maxErr / vs).toFixed(4)} vox, min n·r ${minNDot.toFixed(5)}, ` +
        `volume ${(t1 - t0).toFixed(0)} ms, MC ${(t2 - t1).toFixed(0)} ms`,
    );
    expect(st.boundary).toBe(0);
    expect(st.nonManifold).toBe(0);
    expect(st.misoriented).toBe(0);
    expect(euler).toBe(2);
    expect(meshComponents(mesh).sizes.length).toBe(1);
    expect(maxErr).toBeLessThan(0.05 * vs);
    expect(minNDot).toBeGreaterThan(0.995);
    expect(fd.neg).toBe(0);
    expect(maxColErr).toBeLessThan(2);
  });

  it('torus: closed with Euler characteristic 0', () => {
    const Rm = 0.8;
    const rm = 0.3;
    const sdf = (x: number, y: number, z: number): number => Math.hypot(Math.hypot(x, y) - Rm, z) - rm;
    const vol = TsdfVolume.fromFunction(sdf, [-1.3, -1.3, -0.5], [1.3, 1.3, 0.5], 0.04);
    const mesh = marchingCubes(vol);
    const st = meshEdgeStats(mesh);
    const euler = mesh.vertexCount - st.edges + mesh.triangleCount;
    log(`torus: V=${mesh.vertexCount} F=${mesh.triangleCount} χ=${euler}`);
    expect(st.boundary).toBe(0);
    expect(st.nonManifold).toBe(0);
    expect(st.misoriented).toBe(0);
    expect(euler).toBe(0);
    const fd = faceNormalDots(mesh, (x, y, z) => {
      const q = Math.hypot(x, y);
      return [x - (Rm * x) / q, y - (Rm * y) / q, z];
    });
    expect(fd.neg).toBe(0);
  });

  it('noisy SDF with many ambiguous cubes stays watertight and manifold', () => {
    // high-frequency perturbation creates face/interior ambiguities and small separate pieces
    const sdf = (x: number, y: number, z: number): number =>
      Math.hypot(x, y, z) - 1 + 0.09 * Math.sin(37 * x + 1) * Math.sin(41 * y + 2) * Math.sin(43 * z + 3);
    const vol = TsdfVolume.fromFunction(sdf, [-1.3, -1.3, -1.3], [1.3, 1.3, 1.3], 0.05);
    const mesh = marchingCubes(vol);
    const st = meshEdgeStats(mesh);
    // count cubes with ambiguous faces for the log
    log(`noisy sphere: V=${mesh.vertexCount} F=${mesh.triangleCount}, components ${meshComponents(mesh).sizes.length}, edges ${JSON.stringify(st)}`);
    expect(st.boundary).toBe(0);
    expect(st.nonManifold).toBe(0);
    expect(st.misoriented).toBe(0);
  });

  it('vertex colours come from the coloured end of an edge when the other end was never coloured', () => {
    const vs = 0.05;
    const vol = TsdfVolume.fromFunction((x) => x - 0.3 * vs, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], vs, 4, () => [200, 50, 50]);
    // free-space side: observed but never coloured (colour 0, colour weight 0)
    for (let vi = 0; vi < vol.tsdf.length; vi++) {
      if (vol.tsdf[vi] <= 0) continue;
      vol.color!.fill(0, 3 * vi, 3 * vi + 3);
      vol.colorWeight![vi] = 0;
    }
    const mesh = marchingCubes(vol);
    expect(mesh.vertexCount).toBeGreaterThan(100);
    let bad = 0;
    for (let i = 0; i < mesh.vertexCount; i++) if (mesh.colors[3 * i] !== 200 || mesh.colors[3 * i + 1] !== 50 || mesh.colors[3 * i + 2] !== 50) bad++;
    expect(bad).toBe(0);
  });

  it('removeSmallComponents keeps the large component', () => {
    const sdf = (x: number, y: number, z: number): number => Math.min(Math.hypot(x, y, z) - 1, Math.hypot(x - 2, y, z) - 0.15);
    const vol = TsdfVolume.fromFunction(sdf, [-1.3, -1.3, -1.3], [2.3, 1.3, 1.3], 0.05);
    const mesh = marchingCubes(vol);
    expect(meshComponents(mesh).sizes.length).toBe(2);
    const { mesh: m2, removedComponents } = removeSmallComponents(mesh, 0.05);
    expect(removedComponents).toBe(1);
    expect(meshComponents(m2).sizes.length).toBe(1);
    let maxX = -Infinity;
    for (let i = 0; i < m2.vertexCount; i++) maxX = Math.max(maxX, m2.positions[3 * i]);
    expect(maxX).toBeLessThan(1.01);
    expect(meshEdgeStats(m2).boundary).toBe(0);
  });
});
