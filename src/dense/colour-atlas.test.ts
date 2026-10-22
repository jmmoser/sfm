/**
 * Colour atlas (src/dense/colour-atlas.ts): layout invariants and the colour transfer from the
 * undecimated mesh, on a TSDF mesh of a synthetic textured scene (src/dense/testutil-scene.ts);
 * a mesh the simplifier cannot reduce (over budget, smaller cell layouts); the cooperative build
 * (same result, stops on cancel).
 */
import { describe, expect, it } from 'vitest';
import type { Pose, RGBAImage, TriangleMesh } from '../core/types';
import { Rng } from '../math/random';
import { buildColourAtlas, buildColourAtlasAsync, ClosestTriangleFinder } from './colour-atlas';
import { KdTree3 } from './kdtree';
import { marchingCubes } from './marching-cubes';
import { orbitPoses, pinholeIntrinsics, textureLuminance, type SynthScene } from './synthetic';
import { distanceToScene, renderGtViews } from './testutil-scene';
import { tsdfMesh, TsdfVolume } from './tsdf';

const log = (...a: unknown[]): void => console.log('[colour-atlas]', ...a);

/**
 * A textured box and sphere on a textured ground patch, seen from rings of cameras (depth noise
 * like real fused maps — exactly planar GT faces are a slow worst case of the simplifier). The
 * vertex colours are then replaced by the scene texture at a ~voxel footprint, so they vary at
 * vertex scale like a printed surface (the rendered photos are band-limited to ≈ 1.3 cm here).
 */
async function sceneMesh(): Promise<TriangleMesh> {
  const scene: SynthScene = {
    primitives: [
      { kind: 'quad', center: [0, 0, 0], axisU: [1, 0, 0], axisV: [0, 1, 0], halfU: 0.9, halfV: 0.9, material: { scale: 0.5, octaves: 7, color: [0.62, 0.55, 0.42], seed: 1 } },
      { kind: 'box', center: [0.2, 0.15, 0.2], halfSize: [0.25, 0.18, 0.2], yaw: 0.4, material: { scale: 0.25, octaves: 7, color: [0.8, 0.42, 0.33], seed: 2 } },
      { kind: 'sphere', center: [-0.35, -0.3, 0.22], radius: 0.22, material: { scale: 0.2, octaves: 7, color: [0.45, 0.75, 0.4], seed: 4 } },
    ],
  };
  const cam = pinholeIntrinsics(200, 150, (60 * Math.PI) / 180);
  const poses: Pose[] = [];
  [25, 50, 75].forEach((el, k) => poses.push(...orbitPoses([0, 0, 0.15], 2.2, (el * Math.PI) / 180, 10, k * 0.3)));
  const gt = renderGtViews(scene, cam, poses, { seed: 5, noise: 0.001 });
  const mesh = await tsdfMesh(gt.views, gt.store, { voxelSize: 0.01 });
  for (let i = 0; i < mesh.vertexCount; i++) {
    const x = mesh.positions[3 * i];
    const y = mesh.positions[3 * i + 1];
    const z = mesh.positions[3 * i + 2];
    let best = 0;
    let bestD = Infinity;
    scene.primitives.forEach((p, k) => {
      const d = distanceToScene({ primitives: [p] }, x, y, z);
      if (d < bestD) {
        bestD = d;
        best = k;
      }
    });
    const m = scene.primitives[best].material;
    const lum = textureLuminance(m, x, y, z, 0.005);
    for (let c = 0; c < 3; c++) mesh.colors[3 * i + c] = Math.max(0, Math.min(255, Math.round(255 * lum * (m.color?.[c] ?? 0.6))));
  }
  return mesh;
}

/** Bilinear sample (texel centres at (i + 0.5) / W, clamped at the page border) of an image-convention UV. */
function sampleBilinear(img: RGBAImage, u: number, v: number, out: Float64Array): void {
  const x = u * img.width - 0.5;
  const y = v * img.height - 0.5;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  out.fill(0);
  for (let dy = 0; dy < 2; dy++) {
    for (let dx = 0; dx < 2; dx++) {
      const w = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy);
      const xi = Math.min(img.width - 1, Math.max(0, x0 + dx));
      const yi = Math.min(img.height - 1, Math.max(0, y0 + dy));
      const o = 4 * (yi * img.width + xi);
      for (let c = 0; c < 3; c++) out[c] += w * img.data[o + c];
    }
  }
}

/**
 * Reference colour of `mesh` at the point closest to (x, y, z): exhaustive over the one-rings of
 * the 16 nearest vertices (independent of the atlas' walk).
 */
function referenceColour(mesh: TriangleMesh, tree: KdTree3, ring: number[][], p: number[], out: Float64Array): void {
  const kd = new Float64Array(16);
  const ki = new Int32Array(16);
  const n = tree.knn(p[0], p[1], p[2], 16, kd, ki);
  let best = Infinity;
  const P = mesh.positions;
  for (let i = 0; i < n; i++) {
    for (const t of ring[ki[i]]) {
      const ia = mesh.indices[3 * t];
      const ib = mesh.indices[3 * t + 1];
      const ic = mesh.indices[3 * t + 2];
      const [d, u, v, w] = closestOnTriangle(p, [P[3 * ia], P[3 * ia + 1], P[3 * ia + 2]], [P[3 * ib], P[3 * ib + 1], P[3 * ib + 2]], [P[3 * ic], P[3 * ic + 1], P[3 * ic + 2]]);
      if (d < best) {
        best = d;
        for (let c = 0; c < 3; c++) out[c] = u * mesh.colors[3 * ia + c] + v * mesh.colors[3 * ib + c] + w * mesh.colors[3 * ic + c];
      }
    }
  }
}

/** [squared distance, barycentrics] of the closest point on triangle abc (brute force over regions). */
function closestOnTriangle(p: number[], a: number[], b: number[], c: number[]): [number, number, number, number] {
  const sub = (x: number[], y: number[]): number[] => [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
  const dot = (x: number[], y: number[]): number => x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  const ab = sub(b, a);
  const ac = sub(c, a);
  const ap = sub(p, a);
  // interior projection
  const d00 = dot(ab, ab);
  const d01 = dot(ab, ac);
  const d11 = dot(ac, ac);
  const d20 = dot(ap, ab);
  const d21 = dot(ap, ac);
  const den = d00 * d11 - d01 * d01;
  const cands: [number, number, number][] = [];
  if (den > 0) {
    const v = (d11 * d20 - d01 * d21) / den;
    const w = (d00 * d21 - d01 * d20) / den;
    if (v >= 0 && w >= 0 && v + w <= 1) cands.push([1 - v - w, v, w]);
  }
  const seg = (s0: number[], s1: number[], i0: number, i1: number): void => {
    const d = sub(s1, s0);
    const l = dot(d, d);
    const t = l > 0 ? Math.min(1, Math.max(0, dot(sub(p, s0), d) / l)) : 0;
    const bc = [0, 0, 0];
    bc[i0] = 1 - t;
    bc[i1] = t;
    cands.push(bc as [number, number, number]);
  };
  seg(a, b, 0, 1);
  seg(b, c, 1, 2);
  seg(a, c, 0, 2);
  let best: [number, number, number, number] = [Infinity, 1, 0, 0];
  for (const [u, v, w] of cands) {
    const q = [u * a[0] + v * b[0] + w * c[0], u * a[1] + v * b[1] + w * c[1], u * a[2] + v * b[2] + w * c[2]];
    const e = sub(q, p);
    const d = dot(e, e);
    if (d < best[0]) best = [d, u, v, w];
  }
  return best;
}

describe('buildColourAtlas', () => {
  it('transfers the undecimated colours: atlas error ≤ 3/255 per channel at random surface points', async () => {
    const mesh = await sceneMesh();
    const target = Math.round(mesh.triangleCount / 6);
    // page sized like the app's: ≈ 14 texels per original vertex (2048² for 300 k vertices)
    const size = Math.round(Math.sqrt(14 * mesh.vertexCount) / 8) * 8;
    const atlas = buildColourAtlas(mesh, { maxTriangles: target, size });
    const st = atlas.stats;
    log(
      `input ${mesh.triangleCount} triangles / ${mesh.vertexCount} vertices → ${st.triangles} triangles, page ${size}², ${st.pairs} pairs + ${st.singles} singles, ` +
        `L ${st.minSide}…${st.meanSide.toFixed(1)}…${st.maxSide}, fill ${(100 * st.pageFill).toFixed(1)} %, simplify ${st.simplifyMs.toFixed(0)} ms, pack ${st.packMs.toFixed(0)} ms, fill ${st.fillMs.toFixed(0)} ms`,
    );
    const dec = atlas.mesh;
    expect(dec.triangleCount).toBeLessThanOrEqual(target);
    expect(st.triangles).toBe(dec.triangleCount);
    // most triangles share a square with an edge neighbour
    expect(st.pairs * 2).toBeGreaterThan(0.8 * dec.triangleCount);
    expect(atlas.uvs.length).toBe(6 * dec.triangleCount);
    for (const u of atlas.uvs) {
      expect(u).toBeGreaterThanOrEqual(0);
      expect(u).toBeLessThanOrEqual(1);
    }

    // reference: the ORIGINAL mesh's barycentric colour at the closest point
    const tree = new KdTree3(mesh.positions, mesh.vertexCount);
    const ring: number[][] = Array.from({ length: mesh.vertexCount }, () => []);
    for (let t = 0; t < mesh.triangleCount; t++) for (let k = 0; k < 3; k++) ring[mesh.indices[3 * t + k]].push(t);
    const rng = new Rng(11);
    const N = 10_000;
    const ref = new Float64Array(3);
    const got = new Float64Array(3);
    let errAtlas = 0;
    let errVertex = 0;
    let maxAtlas = 0;
    const I = dec.indices;
    const P = dec.positions;
    for (let s = 0; s < N; s++) {
      const t = rng.int(dec.triangleCount);
      let b1 = rng.next();
      let b2 = rng.next();
      if (b1 + b2 > 1) {
        b1 = 1 - b1;
        b2 = 1 - b2;
      }
      const b0 = 1 - b1 - b2;
      const w = [b0, b1, b2];
      const p = [0, 0, 0];
      let u = 0;
      let v = 0;
      const vc = [0, 0, 0];
      for (let k = 0; k < 3; k++) {
        const vi = I[3 * t + k];
        for (let c = 0; c < 3; c++) {
          p[c] += w[k] * P[3 * vi + c];
          vc[c] += w[k] * dec.colors[3 * vi + c];
        }
        u += w[k] * atlas.uvs[6 * t + 2 * k];
        v += w[k] * atlas.uvs[6 * t + 2 * k + 1];
      }
      referenceColour(mesh, tree, ring, p, ref);
      sampleBilinear(atlas.texture, u, v, got);
      let e = 0;
      for (let c = 0; c < 3; c++) {
        e += Math.abs(got[c] - ref[c]);
        errVertex += Math.abs(vc[c] - ref[c]);
      }
      errAtlas += e;
      maxAtlas = Math.max(maxAtlas, e / 3);
    }
    const meanAtlas = errAtlas / (3 * N);
    const meanVertex = errVertex / (3 * N);
    log(`mean |Δ| per channel at ${N} random points: atlas ${meanAtlas.toFixed(2)}, simplified-mesh vertex colours ${meanVertex.toFixed(2)} (of 255); atlas max ${maxAtlas.toFixed(1)}`);
    expect(meanAtlas).toBeLessThanOrEqual(3);
    expect(meanAtlas).toBeLessThan(meanVertex);
  });

  it('keeps an already small mesh, pairs a quad and leaves the unused page area at the mean colour', () => {
    // two triangles of a unit quad, red at x = 0 and blue at x = 1
    const mesh: TriangleMesh = {
      vertexCount: 4,
      triangleCount: 2,
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
      colors: new Uint8Array([255, 0, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0]),
      indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    };
    const atlas = buildColourAtlas(mesh, { size: 64 });
    expect(atlas.mesh).toBe(mesh);
    expect(atlas.stats.pairs).toBe(1);
    expect(atlas.stats.singles).toBe(0);
    // the shared edge (vertices 0 and 2) has the same UVs in both triangles
    const uv = (t: number, k: number): [number, number] => [atlas.uvs[6 * t + 2 * k], atlas.uvs[6 * t + 2 * k + 1]];
    expect(uv(0, 0)).toEqual(uv(1, 0));
    expect(uv(0, 2)).toEqual(uv(1, 1));
    // colour along x: red → blue, sampled through the UVs at the quad's centre line
    const got = new Float64Array(3);
    for (const [b, x] of [
      [0.0, 0],
      [0.5, 0.5],
      [1.0, 1],
    ] as const) {
      // point on edge 0–1 of triangle 0 at parameter b: x = b, y = 0
      const u = (1 - b) * uv(0, 0)[0] + b * uv(0, 1)[0];
      const v = (1 - b) * uv(0, 0)[1] + b * uv(0, 1)[1];
      sampleBilinear(atlas.texture, u, v, got);
      expect(got[0]).toBeCloseTo(255 * (1 - x), -0.5);
      expect(got[2]).toBeCloseTo(255 * x, -0.5);
    }
    // far corner of the page: mean colour
    const o = 4 * (64 * 64 - 1);
    expect([...atlas.texture.data.subarray(o, o + 4)]).toEqual([128, 0, 128, 255]);
  });

  it('rejects meshes that do not fit and bad options', () => {
    const mesh: TriangleMesh = {
      vertexCount: 3,
      triangleCount: 1,
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array(9),
      colors: new Uint8Array(9),
      indices: new Uint32Array([0, 1, 2]),
    };
    expect(() => buildColourAtlas(mesh, { size: 4 })).toThrow(/page size/);
    expect(() => buildColourAtlas({ ...mesh, triangleCount: 0 })).toThrow(/empty/);
    expect(() => buildColourAtlas({ ...mesh, indices: new Uint32Array([0, 1, 3]) })).toThrow(/out of range/);
    // 2000 single triangles of ≥ 5×5 texels (L ≥ 2 + gutters) cannot fit a 64² page
    const many: TriangleMesh = {
      vertexCount: 6000,
      triangleCount: 2000,
      positions: Float32Array.from({ length: 18000 }, (_, i) => (i % 9 === 3 || i % 9 === 7 ? 1 : 0) + Math.floor(i / 9) * 2),
      normals: new Float32Array(18000),
      colors: new Uint8Array(18000),
      indices: Uint32Array.from({ length: 6000 }, (_, i) => i),
    };
    expect(() => buildColourAtlas(many, { size: 64 })).toThrow(/do not fit/);
    // … but a 160² page holds them without gutters (3×3 texels), a 96² page at 1-texel sides (2×2)
    const noGutter = buildColourAtlas(many, { size: 160 });
    expect([noGutter.stats.gutter, noGutter.stats.minSide >= 2]).toEqual([0, true]);
    const tiny = buildColourAtlas(many, { size: 96 });
    expect([tiny.stats.gutter, tiny.stats.minSide]).toEqual([0, 1]);
    expect(buildColourAtlas(many, { size: 256 }).stats.gutter).toBe(1);
  });

  it('keeps what the simplifier cannot remove (over budget) and still fits the page', () => {
    // 500 closed tetrahedra 2 m apart: no edge collapse passes the link condition, relaxed or not
    const N = 500;
    const corners = [0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1];
    const positions = new Float32Array(12 * N);
    const colors = new Uint8Array(12 * N);
    const indices = new Uint32Array(12 * N);
    const faces = [0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2];
    const colourOf = (i: number): number[] => [(37 * i) % 256, (101 * i) % 256, (173 * i) % 256];
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < 4; k++) {
        positions.set([corners[3 * k] + 2 * (i % 25), corners[3 * k + 1] + 2 * Math.floor(i / 25), corners[3 * k + 2]], 12 * i + 3 * k);
        colors.set(colourOf(i), 12 * i + 3 * k);
      }
      for (let k = 0; k < 12; k++) indices[12 * i + k] = 4 * i + faces[k];
    }
    const mesh: TriangleMesh = { vertexCount: 4 * N, triangleCount: 4 * N, positions, normals: new Float32Array(12 * N), colors, indices };
    // 1000 cells (2 pairs per tetrahedron) need the 2×2-texel layout in 72²
    const atlas = buildColourAtlas(mesh, { maxTriangles: 500, size: 72 });
    const st = atlas.stats;
    expect(st.triangles).toBe(4 * N);
    expect(st.overBudget).toBe(true);
    expect(st.relaxedSimplify).toBe(true);
    expect([st.gutter, st.minSide, st.pairs, st.singles]).toEqual([0, 1, 2 * N, 0]);
    // every triangle's centre samples its tetrahedron's colour
    const got = new Float64Array(3);
    const P = atlas.mesh.positions;
    for (let t = 0; t < atlas.mesh.triangleCount; t++) {
      const u = (atlas.uvs[6 * t] + atlas.uvs[6 * t + 2] + atlas.uvs[6 * t + 4]) / 3;
      const v = (atlas.uvs[6 * t + 1] + atlas.uvs[6 * t + 3] + atlas.uvs[6 * t + 5]) / 3;
      sampleBilinear(atlas.texture, u, v, got);
      const a = atlas.mesh.indices[3 * t];
      expect([...got].map(Math.round)).toEqual(colourOf(Math.round(P[3 * a] / 2) + 25 * Math.round(P[3 * a + 1] / 2)));
    }
    // a page too small even for 2×2 cells
    expect(() => buildColourAtlas(mesh, { maxTriangles: 500, size: 56 })).toThrow(/2000 triangles \(1000 cells\) do not fit a 56² page/);
  });

  it('cooperative build: the same atlas, and a cancel stops it at the next step', async () => {
    const vol = TsdfVolume.fromFunction((x, y, z) => Math.hypot(x, y, z) - 0.3, [-0.35, -0.35, -0.35], [0.35, 0.35, 0.35], 0.02);
    const mesh = marchingCubes(vol);
    for (let i = 0; i < mesh.vertexCount; i++) mesh.colors.fill(Math.round(127 + 400 * mesh.positions[3 * i + 2]), 3 * i, 3 * i + 3);
    const opts = { maxTriangles: Math.round(mesh.triangleCount / 3), size: 256 };
    const sync = buildColourAtlas(mesh, opts);
    const co = await buildColourAtlasAsync(mesh, opts, { cancelled: false });
    expect(co.mesh.triangleCount).toBe(sync.mesh.triangleCount);
    expect(co.uvs).toEqual(sync.uvs);
    expect(co.texture.data).toEqual(sync.texture.data);
    // cancelled before / after a few steps
    await expect(buildColourAtlasAsync(mesh, opts, { cancelled: true })).rejects.toMatchObject({ name: 'CancelledError' });
    let reads = 0;
    const token = {
      get cancelled(): boolean {
        return ++reads > 5;
      },
    };
    await expect(buildColourAtlasAsync(mesh, opts, token)).rejects.toThrow(/colour atlas: cancelled/);
  });
});

describe('ClosestTriangleFinder', () => {
  it('finds the closest triangle of a folded strip from any start', () => {
    // a zig-zag strip of 40 triangles; queries above random points
    const n = 21;
    const pos: number[] = [];
    for (let i = 0; i < n; i++) pos.push(i, 0, Math.sin(i), i, 1, Math.sin(i));
    const idx: number[] = [];
    for (let i = 0; i + 1 < n; i++) idx.push(2 * i, 2 * i + 2, 2 * i + 1, 2 * i + 1, 2 * i + 2, 2 * i + 3);
    const mesh: TriangleMesh = {
      vertexCount: 2 * n,
      triangleCount: idx.length / 3,
      positions: Float32Array.from(pos),
      normals: new Float32Array(6 * n),
      colors: Uint8Array.from({ length: 6 * n }, (_, i) => (i * 37) % 256),
      indices: Uint32Array.from(idx),
    };
    const f = new ClosestTriangleFinder(mesh);
    const rng = new Rng(3);
    for (let q = 0; q < 200; q++) {
      const x = rng.next() * (n - 1);
      const y = rng.next();
      const z = Math.sin(x) + 0.05 * (rng.next() - 0.5);
      if (q % 7 === 0) f.reset();
      const t = f.closest(x, y, z);
      let best = Infinity;
      for (let s = 0; s < mesh.triangleCount; s++) {
        const c = [0, 1, 2].map((k) => Array.from(mesh.positions.subarray(3 * mesh.indices[3 * s + k], 3 * mesh.indices[3 * s + k] + 3)));
        best = Math.min(best, closestOnTriangle([x, y, z], c[0], c[1], c[2])[0]);
      }
      const ct = [0, 1, 2].map((k) => Array.from(mesh.positions.subarray(3 * mesh.indices[3 * t + k], 3 * mesh.indices[3 * t + k] + 3)));
      expect(closestOnTriangle([x, y, z], ct[0], ct[1], ct[2])[0]).toBeLessThanOrEqual(best + 1e-9);
    }
  });
});
