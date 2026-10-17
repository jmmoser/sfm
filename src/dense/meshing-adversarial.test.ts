/**
 * Adversarial tests for marching cubes, mesh simplification and point-cloud filters
 * (review pass).
 */
import { describe, expect, it } from 'vitest';
import type { PointCloud, TriangleMesh } from '../core/types';
import { Rng } from '../math/random';
import { BLOCK_VOXELS, MC_EDGE_CORNERS, TSDF_SCALE, WEIGHT_UNIT, marchingCubes, meshComponents, meshEdgeStats } from './marching-cubes';
import { EdgeHeap, simplifyMesh } from './simplify';
import { TsdfVolume } from './tsdf';
import { KdTree3 } from './kdtree';
import { filterPointCloud, statisticalOutlierMask, voxelDownsample } from './pointcloud';

const log = (...a: unknown[]): void => console.log('[meshing-adv]', ...a);

function signedVolume(mesh: TriangleMesh): number {
  const P = mesh.positions;
  let v = 0;
  for (let t = 0; t < mesh.triangleCount; t++) {
    const a = 3 * mesh.indices[3 * t];
    const b = 3 * mesh.indices[3 * t + 1];
    const c = 3 * mesh.indices[3 * t + 2];
    v += P[a] * (P[b + 1] * P[c + 2] - P[b + 2] * P[c + 1]) - P[a + 1] * (P[b] * P[c + 2] - P[b + 2] * P[c]) + P[a + 2] * (P[b] * P[c + 1] - P[b + 1] * P[c]);
  }
  return v / 6;
}

function validIndices(mesh: TriangleMesh): boolean {
  for (const i of mesh.indices) if (!(i < mesh.vertexCount)) return false;
  for (const v of mesh.positions) if (!Number.isFinite(v)) return false;
  for (const v of mesh.normals) if (!Number.isFinite(v)) return false;
  return true;
}

describe('marchingCubes (adversarial)', () => {
  it('random field over 3×3×3 blocks: all 256 cases occur, closed, edge-manifold, consistently and outwardly oriented', () => {
    const vs = 0.1;
    const vol = new TsdfVolume(vs, 0.4, 1000, false, 64);
    const rng = new Rng(9);
    const N = 24;
    const val = new Float64Array(N * N * N);
    for (let z = 0; z < N; z++)
      for (let y = 0; y < N; y++)
        for (let x = 0; x < N; x++) {
          const border = x === 0 || y === 0 || z === 0 || x === N - 1 || y === N - 1 || z === N - 1;
          // positive shell → the zero set is closed; interior values never saturate (no ghost cubes)
          val[x + N * (y + N * z)] = border ? 0.5 : 0.9 * (2 * rng.next() - 1) || 0.01;
        }
    for (let bz = 0; bz < 3; bz++)
      for (let by = 0; by < 3; by++)
        for (let bx = 0; bx < 3; bx++) {
          const b = vol.allocateBlock(bx, by, bz);
          for (let lz = 0; lz < 8; lz++)
            for (let ly = 0; ly < 8; ly++)
              for (let lx = 0; lx < 8; lx++) {
                const vi = b * BLOCK_VOXELS + lx + 8 * (ly + 8 * lz);
                vol.tsdf[vi] = Math.round(val[8 * bx + lx + N * (8 * by + ly + N * (8 * bz + lz))] * TSDF_SCALE);
                vol.weight[vi] = WEIGHT_UNIT;
              }
        }
    const cases = new Set<number>();
    let negVoxels = 0;
    for (let z = 0; z < N - 1; z++)
      for (let y = 0; y < N - 1; y++)
        for (let x = 0; x < N - 1; x++) {
          let cs = 0;
          for (let c = 0; c < 8; c++) if (val[x + (c & 1) + N * (y + ((c >> 1) & 1) + N * (z + ((c >> 2) & 1)))] < 0) cs |= 1 << c;
          cases.add(cs);
        }
    for (const v of val) if (v < 0) negVoxels++;
    const mesh = marchingCubes(vol);
    const st = meshEdgeStats(mesh);
    const vol6 = signedVolume(mesh);
    const est = negVoxels * vs * vs * vs;
    log(`random field: ${cases.size} cases, V ${mesh.vertexCount} F ${mesh.triangleCount}, edges ${JSON.stringify(st)}, enclosed volume ${vol6.toFixed(3)} vs ≈ ${est.toFixed(3)}`);
    expect(cases.size).toBe(256);
    expect(validIndices(mesh)).toBe(true);
    expect(st.boundary).toBe(0);
    expect(st.nonManifold).toBe(0);
    expect(st.misoriented).toBe(0);
    expect(vol6).toBeGreaterThan(0.6 * est);
    expect(vol6).toBeLessThan(1.4 * est);
    void MC_EDGE_CORNERS;
  });

  it('exact zeros on the voxel grid (box faces through voxel centres): closed genus-0, finite', () => {
    const vs = 0.1;
    const sdf = (x: number, y: number, z: number): number => Math.max(Math.abs(x), Math.abs(y), Math.abs(z)) - 0.5;
    const vol = TsdfVolume.fromFunction(sdf, [-1, -1, -1], [1, 1, 1], vs, 4);
    let zeros = 0;
    for (let i = 0; i < vol.blockCount * BLOCK_VOXELS; i++) if (vol.tsdf[i] === 0 && vol.weight[i] > 0) zeros++;
    const mesh = marchingCubes(vol);
    const st = meshEdgeStats(mesh);
    const euler = mesh.vertexCount - st.edges + mesh.triangleCount;
    log(`zero-valued voxels ${zeros}: V ${mesh.vertexCount} F ${mesh.triangleCount}, χ ${euler}, edges ${JSON.stringify(st)}, volume ${signedVolume(mesh).toFixed(4)}`);
    expect(zeros).toBeGreaterThan(500);
    expect(validIndices(mesh)).toBe(true);
    expect(st.boundary).toBe(0);
    expect(st.nonManifold).toBe(0);
    expect(st.misoriented).toBe(0);
    expect(euler).toBe(2);
    // zeros count as outside: the 12 box edges are chamfered by one voxel (≈ −5 % volume)
    expect(signedVolume(mesh)).toBeGreaterThan(0.93);
    expect(signedVolume(mesh)).toBeLessThanOrEqual(1 + 1e-6);
  });

  it('unobserved voxels (weight 0) cut holes with clean boundaries, never non-manifold edges', () => {
    const vol = TsdfVolume.fromFunction((x, y, z) => Math.hypot(x, y, z) - 1, [-1.3, -1.3, -1.3], [1.3, 1.3, 1.3], 0.05);
    const rng = new Rng(3);
    for (let i = 0; i < vol.blockCount * BLOCK_VOXELS; i++) if (rng.next() < 0.03) vol.weight[i] = 0;
    const mesh = marchingCubes(vol);
    const st = meshEdgeStats(mesh);
    log(`3 % unobserved voxels: F ${mesh.triangleCount}, edges ${JSON.stringify(st)}`);
    expect(st.boundary).toBeGreaterThan(0);
    expect(st.nonManifold).toBe(0);
    expect(st.misoriented).toBe(0);
  });
});

describe('EdgeHeap (adversarial)', () => {
  it('pops in non-decreasing cost order under random interleaved push / pop (incl. ties), payload intact', () => {
    const rng = new Rng(17);
    const heap = new EdgeHeap();
    const ref: { c: number; u: number }[] = [];
    const out = new Float64Array(5);
    let id = 0;
    for (let step = 0; step < 60000; step++) {
      if (rng.next() < 0.6 || ref.length === 0) {
        const c = rng.next() < 0.2 ? Math.floor(rng.next() * 5) : rng.next() * 100;
        heap.push(c, id, id + 1, id + 2, id + 3);
        ref.push({ c, u: id });
        id++;
      } else {
        expect(heap.pop(out)).toBe(true);
        let mi = 0;
        for (let k = 1; k < ref.length; k++) if (ref[k].c < ref[mi].c) mi = k;
        expect(out[0]).toBe(ref[mi].c);
        // payload of the popped slot belongs to an entry with that cost
        const k = ref.findIndex((e) => e.u === out[1] && e.c === out[0]);
        expect(k).toBeGreaterThanOrEqual(0);
        expect(out[2]).toBe(out[1] + 1);
        expect(out[3]).toBe(out[1] + 2);
        expect(out[4]).toBe(out[1] + 3);
        ref.splice(k, 1);
      }
      expect(heap.size).toBe(ref.length);
    }
    let prev = -Infinity;
    while (heap.pop(out)) {
      expect(out[0]).toBeGreaterThanOrEqual(prev);
      prev = out[0];
    }
  });
});

/** n×n grid of unit squares in z = 0, CCW seen from +z, with a smooth colour ramp. */
function gridMesh(n: number): TriangleMesh {
  const V = (n + 1) * (n + 1);
  const positions = new Float32Array(3 * V);
  const normals = new Float32Array(3 * V);
  const colors = new Uint8Array(3 * V);
  for (let j = 0; j <= n; j++)
    for (let i = 0; i <= n; i++) {
      const v = j * (n + 1) + i;
      positions[3 * v] = i;
      positions[3 * v + 1] = j;
      normals[3 * v + 2] = 1;
      colors[3 * v] = Math.round((255 * i) / n);
      colors[3 * v + 1] = Math.round((255 * j) / n);
      colors[3 * v + 2] = 7;
    }
  const indices = new Uint32Array(6 * n * n);
  let o = 0;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const a = j * (n + 1) + i;
      const b = a + 1;
      const c = a + n + 1;
      const d = c + 1;
      indices.set([a, b, d, a, d, c], o);
      o += 6;
    }
  return { vertexCount: V, triangleCount: 2 * n * n, positions, normals, colors, indices };
}

describe('simplifyMesh (adversarial)', () => {
  it('flat grid with a boundary: stays planar, keeps its outline and orientation, area preserved', () => {
    const n = 40;
    const mesh = gridMesh(n);
    const out = simplifyMesh(mesh, 50);
    let area = 0;
    let flipped = 0;
    const P = out.positions;
    for (let t = 0; t < out.triangleCount; t++) {
      const [a, b, c] = [0, 1, 2].map((k) => 3 * out.indices[3 * t + k]);
      const cz = (P[b] - P[a]) * (P[c + 1] - P[a + 1]) - (P[b + 1] - P[a + 1]) * (P[c] - P[a]);
      if (cz <= 0) flipped++;
      area += 0.5 * cz;
    }
    let offPlane = 0;
    let corners = 0;
    for (let v = 0; v < out.vertexCount; v++) {
      if (Math.abs(P[3 * v + 2]) > 1e-6) offPlane++;
      const x = P[3 * v];
      const y = P[3 * v + 1];
      if ((Math.abs(x) < 1e-6 || Math.abs(x - n) < 1e-6) && (Math.abs(y) < 1e-6 || Math.abs(y - n) < 1e-6)) corners++;
    }
    const st = meshEdgeStats(out);
    log(`grid ${mesh.triangleCount} → ${out.triangleCount} triangles, area ${area.toFixed(6)} (exp ${n * n}), flipped ${flipped}, off-plane ${offPlane}, corners ${corners}, edges ${JSON.stringify(st)}`);
    expect(out.triangleCount).toBeLessThanOrEqual(60);
    expect(validIndices(out)).toBe(true);
    expect(flipped).toBe(0);
    expect(offPlane).toBe(0);
    expect(corners).toBe(4);
    expect(area).toBeCloseTo(n * n, 3);
    expect(st.nonManifold).toBe(0);
    expect(st.misoriented).toBe(0);
  });

  it('closed sphere simplified to ~1 %: still closed, genus 0, manifold', () => {
    const vol = TsdfVolume.fromFunction((x, y, z) => Math.hypot(x, y, z) - 1, [-1.3, -1.3, -1.3], [1.3, 1.3, 1.3], 0.05);
    const mesh = marchingCubes(vol);
    const out = simplifyMesh(mesh, Math.floor(mesh.triangleCount / 100));
    const st = meshEdgeStats(out);
    const euler = out.vertexCount - st.edges + out.triangleCount;
    log(`sphere ${mesh.triangleCount} → ${out.triangleCount}, χ ${euler}, edges ${JSON.stringify(st)}, volume ${signedVolume(out).toFixed(3)} (4.19)`);
    expect(out.triangleCount).toBeLessThanOrEqual(Math.floor(mesh.triangleCount / 100) + 2);
    expect(st.boundary).toBe(0);
    expect(st.nonManifold).toBe(0);
    expect(st.misoriented).toBe(0);
    expect(euler).toBe(2);
    expect(meshComponents(out).sizes.length).toBe(1);
    expect(signedVolume(out)).toBeGreaterThan(0.9 * (4 / 3) * Math.PI);
  });

  it('degenerate input (non-manifold fan, duplicate and degenerate triangles, unused vertices, target 0) does not crash', () => {
    const base = gridMesh(6);
    const extra: number[] = [];
    // a third triangle on an interior edge (non-manifold), a duplicate, a degenerate one
    extra.push(base.indices[0], base.indices[1], base.vertexCount); // new apex vertex above the grid
    extra.push(base.indices[3], base.indices[4], base.indices[5]);
    extra.push(5, 5, 6);
    const V = base.vertexCount + 2; // +1 apex, +1 unused
    const positions = new Float32Array(3 * V);
    positions.set(base.positions);
    positions.set([0.5, 0.2, 2], 3 * base.vertexCount);
    positions.set([99, 99, 99], 3 * (base.vertexCount + 1));
    const normals = new Float32Array(3 * V);
    normals.set(base.normals);
    const colors = new Uint8Array(3 * V);
    colors.set(base.colors);
    const mesh: TriangleMesh = {
      vertexCount: V,
      triangleCount: base.triangleCount + 3,
      positions,
      normals,
      colors,
      indices: Uint32Array.from([...base.indices, ...extra]),
    };
    for (const target of [0, 10, 40]) {
      const out = simplifyMesh(mesh, target);
      expect(validIndices(out)).toBe(true);
      for (let t = 0; t < out.triangleCount; t++) {
        const a = out.indices[3 * t];
        const b = out.indices[3 * t + 1];
        const c = out.indices[3 * t + 2];
        expect(a !== b && b !== c && a !== c).toBe(true);
      }
    }
  });
});

function cloudOf(pts: number[][], radii?: number[]): PointCloud {
  const n = pts.length;
  const pc: PointCloud = { count: n, positions: Float32Array.from(pts.flat()), colors: new Uint8Array(3 * n).fill(90) };
  if (radii) pc.radii = Float32Array.from(radii);
  return pc;
}

describe('point-cloud filters (adversarial)', () => {
  it('SOR keeps every point when all statistics are identical (σ = 0, floating-point mean)', () => {
    // 1000 well separated pairs 0.1 apart (float32): every point has the SAME kNN distance
    // (k = 1), whose float64 mean / variance only differ from it by rounding
    for (const n of [999, 1000, 1001, 4097]) {
      const pts: number[][] = [];
      // integer x (exact in float32), pair offset along y: every distance is exactly 0.1f
      for (let i = 0; i < n; i++) pts.push([Math.floor(i / 2) * 10, (i % 2) * 0.1, -0.7]);
      if (n % 2) pts.push([Math.floor(n / 2) * 10, 0.1, -0.7]);
      const { keep, threshold } = statisticalOutlierMask(cloudOf(pts), { k: 1, std: 0 });
      let kept = 0;
      for (const x of keep) kept += x;
      log(`identical stats n=${pts.length}: kept ${kept}, threshold ${threshold}`);
      expect(kept).toBe(pts.length);
    }
    // k = 3 on far-apart squares: every value is (2 s + s√2) / 3, not representable → the
    // float64 mean of n copies is off by rounding
    for (const m of [251, 1000, 1777]) {
      const pts: number[][] = [];
      for (let i = 0; i < m; i++) for (const [a, b] of [[0, 0], [1, 0], [1, 1], [0, 1]]) pts.push([i * 10, a * 0.1, b * 0.1]);
      const { keep, meanDist, threshold } = statisticalOutlierMask(cloudOf(pts), { k: 3, std: 0 });
      let kept = 0;
      for (const x of keep) kept += x;
      log(`squares m=${m}: kept ${kept}/${4 * m}, value ${meanDist[0]}, threshold ${threshold}`);
      expect(kept).toBe(4 * m);
    }
    // exactly identical values (duplicated points): nothing may be removed
    const dup = cloudOf(Array.from({ length: 200 }, () => [0.1, 0.2, 0.3]));
    const r = statisticalOutlierMask(dup, { k: 4, std: 0 });
    let kept = 0;
    for (const x of r.keep) kept += x;
    expect(kept).toBe(200);
  });

  it('scale-adaptive SOR with missing (0) radii does not delete those points wholesale', () => {
    const rng = new Rng(2);
    const pts: number[][] = [];
    const radii: number[] = [];
    for (let i = 0; i < 4000; i++) {
      pts.push([rng.next(), rng.next(), 0.001 * rng.gaussian()]);
      radii.push(i % 4 === 0 ? 0 : 0.01); // a quarter without a radius
    }
    for (let i = 0; i < 40; i++) {
      pts.push([rng.next(), rng.next(), 0.5 + rng.next()]); // outliers above the plane
      radii.push(0.01);
    }
    const out = filterPointCloud(cloudOf(pts, radii), { statistical: { k: 8, std: 2 } });
    const { keep } = statisticalOutlierMask(cloudOf(pts, radii), { k: 8, std: 2 });
    let keptNoRadius = 0;
    let keptOutliers = 0;
    for (let i = 0; i < 4000; i += 4) keptNoRadius += keep[i];
    for (let i = 4000; i < 4040; i++) keptOutliers += keep[i];
    log(`0-radius points kept ${keptNoRadius}/1000, outliers kept ${keptOutliers}/40, output ${out.count}`);
    expect(keptNoRadius).toBeGreaterThan(950);
    expect(keptOutliers).toBeLessThan(10);
  });

  it('k-d tree on fully duplicated / collinear points: exact kNN, terminates', () => {
    const P = new Float32Array(3 * 5000);
    for (let i = 0; i < 5000; i++) P[3 * i] = i < 2500 ? 1 : (i % 7) * 0.5; // half identical, rest collinear
    const tree = new KdTree3(P);
    const d = new Float64Array(16);
    const ix = new Int32Array(16);
    const got = tree.knn(1, 0, 0, 16, d, ix);
    expect(got).toBe(16);
    for (let k = 0; k < 16; k++) expect(d[k]).toBe(0);
    const g2 = tree.knn(0.25, 0, 0, 3, d, ix);
    expect(g2).toBe(3);
    for (let k = 0; k < 3; k++) expect(d[k]).toBeCloseTo(0.0625, 10);
  });

  it('voxelDownsample: cells straddling 0 are not merged (floor, not truncation) and NaN points do not crash', () => {
    const pc = cloudOf([
      [-0.01, 0.01, 0.01],
      [0.01, 0.01, 0.01],
      [NaN, 0, 0],
      [0.02, 0.02, 0.02],
    ]);
    const out = voxelDownsample(pc, 0.1);
    let finite = 0;
    for (let i = 0; i < out.count; i++) if (Number.isFinite(out.positions[3 * i])) finite++;
    expect(finite).toBe(2);
  });
});
