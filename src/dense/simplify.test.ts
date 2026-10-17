import { describe, expect, it } from 'vitest';
import type { TriangleMesh } from '../core/types';
import { marchingCubes, meshEdgeStats } from './marching-cubes';
import { simplifyMesh } from './simplify';
import { TsdfVolume } from './tsdf';

const log = (...a: unknown[]): void => console.log('[simplify]', ...a);

function closestPointTriangleDistSq(p: number[], a: number[], b: number[], c: number[]): number {
  // Ericson, Real-Time Collision Detection 5.1.5
  const sub = (u: number[], v: number[]): number[] => [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
  const dot = (u: number[], v: number[]): number => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const ab = sub(b, a);
  const ac = sub(c, a);
  const ap = sub(p, a);
  const d1 = dot(ab, ap);
  const d2 = dot(ac, ap);
  const at = (q: number[]): number => dot(sub(p, q), sub(p, q));
  if (d1 <= 0 && d2 <= 0) return at(a);
  const bp = sub(p, b);
  const d3 = dot(ab, bp);
  const d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return at(b);
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return at([a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]]);
  }
  const cp = sub(p, c);
  const d5 = dot(ab, cp);
  const d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return at(c);
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return at([a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]]);
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
    return at([b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])]);
  }
  if (!(va + vb + vc > 0)) return Math.min(at(a), at(b), at(c)); // degenerate triangle
  const denom = 1 / (va + vb + vc);
  const v = vb * denom;
  const w = vc * denom;
  return at([a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w]);
}

/** Max distance from the vertices of `from` to the surface of `to` (grid-accelerated, search radius r). */
function oneSidedHausdorff(from: TriangleMesh, to: TriangleMesh, r: number): number {
  const cell = r;
  const grid = new Map<string, number[]>();
  const P = to.positions;
  for (let t = 0; t < to.triangleCount; t++) {
    let mn = [Infinity, Infinity, Infinity];
    let mx = [-Infinity, -Infinity, -Infinity];
    for (let k = 0; k < 3; k++) {
      const v = to.indices[3 * t + k];
      mn = mn.map((m, a) => Math.min(m, P[3 * v + a]));
      mx = mx.map((m, a) => Math.max(m, P[3 * v + a]));
    }
    for (let x = Math.floor(mn[0] / cell); x <= Math.floor(mx[0] / cell); x++)
      for (let y = Math.floor(mn[1] / cell); y <= Math.floor(mx[1] / cell); y++)
        for (let z = Math.floor(mn[2] / cell); z <= Math.floor(mx[2] / cell); z++) {
          const key = `${x},${y},${z}`;
          if (!grid.has(key)) grid.set(key, []);
          grid.get(key)!.push(t);
        }
  }
  let worst = 0;
  for (let i = 0; i < from.vertexCount; i++) {
    const p = [from.positions[3 * i], from.positions[3 * i + 1], from.positions[3 * i + 2]];
    const cx = Math.floor(p[0] / cell);
    const cy = Math.floor(p[1] / cell);
    const cz = Math.floor(p[2] / cell);
    let best = r * r;
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dz = -1; dz <= 1; dz++) {
          const l = grid.get(`${cx + dx},${cy + dy},${cz + dz}`);
          if (!l) continue;
          for (const t of l) {
            const tri = [0, 1, 2].map((k) => {
              const v = to.indices[3 * t + k];
              return [P[3 * v], P[3 * v + 1], P[3 * v + 2]];
            });
            let d2 = closestPointTriangleDistSq(p, tri[0], tri[1], tri[2]);
            if (!Number.isFinite(d2)) d2 = Math.min(...tri.map((q) => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2));
            best = Math.min(best, d2);
          }
        }
    worst = Math.max(worst, Math.sqrt(best));
  }
  return worst;
}

/** Faces whose normal points inwards (degenerate zero-area faces are not counted). */
function outwardFlips(mesh: TriangleMesh): number {
  let flips = 0;
  const P = mesh.positions;
  for (let t = 0; t < mesh.triangleCount; t++) {
    const [a, b, c] = [0, 1, 2].map((k) => 3 * mesh.indices[3 * t + k]);
    const e1 = [P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]];
    const e2 = [P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]];
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const g = [P[a] + P[b] + P[c], P[a + 1] + P[b + 1] + P[c + 1], P[a + 2] + P[b + 2] + P[c + 2]];
    if (n[0] * g[0] + n[1] * g[1] + n[2] * g[2] < 0) flips++;
  }
  return flips;
}

describe('simplifyMesh', () => {
  it('sphere to 25 % triangles: closed, no flips, small Hausdorff error, colours preserved', () => {
    const vs = 0.04;
    const colorFn = (x: number, y: number, z: number): [number, number, number] => [128 + 100 * x, 128 + 100 * y, 128 + 100 * z];
    const vol = TsdfVolume.fromFunction((x, y, z) => Math.hypot(x, y, z) - 1, [-1.3, -1.3, -1.3], [1.3, 1.3, 1.3], vs, 4, colorFn);
    const mesh = marchingCubes(vol);
    const target = Math.floor(mesh.triangleCount / 4);
    const t0 = performance.now();
    const flips0 = outwardFlips(mesh);
    const s = simplifyMesh(mesh, target);
    const t1 = performance.now();
    expect(flips0).toBe(0);
    const st = meshEdgeStats(s);
    let maxR = 0;
    let maxCol = 0;
    for (let i = 0; i < s.vertexCount; i++) {
      const x = s.positions[3 * i];
      const y = s.positions[3 * i + 1];
      const z = s.positions[3 * i + 2];
      maxR = Math.max(maxR, Math.abs(Math.hypot(x, y, z) - 1));
      const c = colorFn(x, y, z);
      for (let k = 0; k < 3; k++) maxCol = Math.max(maxCol, Math.abs(s.colors[3 * i + k] - c[k]));
    }
    const hOrigToSimp = oneSidedHausdorff(mesh, s, 4 * vs);
    const hSimpToOrig = oneSidedHausdorff(s, mesh, 4 * vs);
    log(
      `sphere ${mesh.triangleCount} → ${s.triangleCount} triangles (target ${target}) in ${(t1 - t0).toFixed(0)} ms; ` +
        `vertex radius error max ${(maxR / vs).toFixed(4)} vox; Hausdorff orig→simp ${(hOrigToSimp / vs).toFixed(4)} vox, simp→orig ${(hSimpToOrig / vs).toFixed(4)} vox; ` +
        `max colour error ${maxCol.toFixed(1)}; edges ${JSON.stringify(st)}`,
    );
    expect(s.triangleCount).toBeLessThanOrEqual(target);
    expect(s.triangleCount).toBeGreaterThan(0.95 * target);
    expect(st.boundary).toBe(0);
    expect(st.nonManifold).toBe(0);
    expect(st.misoriented).toBe(0);
    expect(s.vertexCount - st.edges + s.triangleCount).toBe(2);
    expect(outwardFlips(s)).toBe(0);
    expect(Math.max(hOrigToSimp, hSimpToOrig)).toBeLessThan(0.1 * vs);
    expect(maxCol).toBeLessThan(4);
  });

  it('planar patch with a bump: plane stays exact, open boundary preserved', () => {
    const N = 60;
    const positions: number[] = [];
    const indices: number[] = [];
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const x = i / (N - 1);
        const y = j / (N - 1);
        const r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2;
        positions.push(x, y, r2 < 0.04 ? 0.1 * (0.04 - r2) * 25 : 0);
      }
    }
    for (let j = 0; j + 1 < N; j++) {
      for (let i = 0; i + 1 < N; i++) {
        const a = j * N + i;
        indices.push(a, a + 1, a + N + 1, a, a + N + 1, a + N);
      }
    }
    const V = N * N;
    const mesh: TriangleMesh = {
      vertexCount: V,
      triangleCount: indices.length / 3,
      positions: Float32Array.from(positions),
      normals: new Float32Array(3 * V).map((_, k) => (k % 3 === 2 ? 1 : 0)),
      colors: new Uint8Array(3 * V).fill(128),
      indices: Uint32Array.from(indices),
    };
    const s = simplifyMesh(mesh, Math.floor(mesh.triangleCount / 10));
    const st = meshEdgeStats(s);
    let offBorder = 0;
    let borderVerts = 0;
    let maxFlatErr = 0;
    // boundary vertices must stay on the square's border
    const edgeCount = new Map<string, number>();
    for (let t = 0; t < s.triangleCount; t++) {
      for (let k = 0; k < 3; k++) {
        const a = s.indices[3 * t + k];
        const b = s.indices[3 * t + ((k + 1) % 3)];
        const key = `${Math.min(a, b)},${Math.max(a, b)}`;
        edgeCount.set(key, (edgeCount.get(key) ?? 0) + 1);
      }
    }
    const isB = new Uint8Array(s.vertexCount);
    for (const [key, c] of edgeCount) if (c === 1) for (const v of key.split(',').map(Number)) isB[v] = 1;
    for (let i = 0; i < s.vertexCount; i++) {
      const x = s.positions[3 * i];
      const y = s.positions[3 * i + 1];
      const z = s.positions[3 * i + 2];
      if (isB[i]) {
        borderVerts++;
        const d = Math.min(Math.abs(x), Math.abs(1 - x), Math.abs(y), Math.abs(1 - y));
        if (d > 1e-5) offBorder++;
      }
      const r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2;
      if (r2 > 0.05) maxFlatErr = Math.max(maxFlatErr, Math.abs(z));
    }
    log(`plane+bump ${mesh.triangleCount} → ${s.triangleCount}: ${borderVerts} border vertices (${offBorder} off the border), max |z| on the flat part ${maxFlatErr.toExponential(2)}`);
    expect(s.triangleCount).toBeLessThanOrEqual(mesh.triangleCount / 10);
    expect(st.nonManifold).toBe(0);
    expect(offBorder).toBe(0);
    expect(maxFlatErr).toBeLessThan(1e-5);
  });

  it('timing: ~470k-triangle sphere to 25 %', () => {
    const vol = TsdfVolume.fromFunction((x, y, z) => Math.hypot(x, y, z) - 1, [-1.1, -1.1, -1.1], [1.1, 1.1, 1.1], 0.0115);
    const mesh = marchingCubes(vol);
    const t0 = performance.now();
    const s = simplifyMesh(mesh, Math.floor(mesh.triangleCount / 4));
    const dt = performance.now() - t0;
    log(`${mesh.triangleCount} → ${s.triangleCount} triangles: ${dt.toFixed(0)} ms (${((1000 * dt) / mesh.triangleCount).toFixed(2)} µs per input triangle)`);
    expect(meshEdgeStats(s).nonManifold).toBe(0);
    expect(outwardFlips(s)).toBe(0);
  });

  it('target ≥ count returns an equivalent mesh', () => {
    const vol = TsdfVolume.fromFunction((x, y, z) => Math.hypot(x, y, z) - 0.5, [-0.7, -0.7, -0.7], [0.7, 0.7, 0.7], 0.1);
    const mesh = marchingCubes(vol);
    const s = simplifyMesh(mesh, mesh.triangleCount + 10);
    expect(s.triangleCount).toBe(mesh.triangleCount);
    expect(Array.from(s.positions)).toEqual(Array.from(mesh.positions));
  });
});
