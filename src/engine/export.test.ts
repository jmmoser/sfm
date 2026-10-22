/**
 * Exports (src/engine/export.ts buildExport): 'ply-points' and points-only GLB use the dense
 * cloud when it has points, else the sparse cloud — an empty dense cloud (a degenerate densify)
 * does not hide the sparse one; 'usdz' simplifies to the AR budget, bakes the atlas and puts the
 * origin at the AR pivot, warns when a mesh stays above the budget, and stops on cancel.
 */
import { describe, expect, it } from 'vitest';
import type { MetricInfo, PointCloud, TriangleMesh } from '../core/types';
import { marchingCubes } from '../dense/marching-cubes';
import { TsdfVolume } from '../dense/tsdf';
import { parseGLB } from '../io/glb';
import { decodePLY } from '../io/ply';
import { readZip } from '../io/zip';
import { buildExport, USDZ_MAX_TRIANGLES, type ExportSources } from './export';

function cloud(n: number): PointCloud {
  return { count: n, positions: Float32Array.from({ length: 3 * n }, (_, i) => i), colors: new Uint8Array(3 * n).fill(100) };
}

function sources(sparse: PointCloud | null, dense: PointCloud | null): ExportSources {
  return { name: 'export-test', sparse, model: null, dense, mesh: null, store: null, features: async () => null };
}

const bytes = async (b: Blob): Promise<Uint8Array> => new Uint8Array(await b.arrayBuffer());

describe('buildExport: point clouds', () => {
  it('falls back to the sparse cloud when the dense one is empty', async () => {
    const ply = decodePLY(await bytes(await buildExport('ply-points', sources(cloud(2), cloud(0)))));
    expect(ply.points?.count).toBe(2);
    const glb = parseGLB(await bytes(await buildExport('glb', sources(cloud(2), cloud(0)))));
    expect(glb.json.accessors?.[0]?.count).toBe(2);
    // a non-empty dense cloud wins
    expect(decodePLY(await bytes(await buildExport('ply-points', sources(cloud(2), cloud(5))))).points?.count).toBe(5);
  });

  it('still rejects when there are no points at all', async () => {
    await expect(buildExport('ply-points', sources(cloud(0), cloud(0)))).rejects.toThrow(/No point cloud yet/);
    await expect(buildExport('ply-points', sources(null, null))).rejects.toThrow(/No point cloud yet/);
    await expect(buildExport('glb', sources(cloud(0), cloud(0)))).rejects.toThrow(/Nothing to export yet/);
  });
});

describe('buildExport: usdz', () => {
  /** A ~130 k-triangle sphere of radius 0.3 centred at (1, 2, 0.5), coloured by height. */
  function sphere(): TriangleMesh {
    const vol = TsdfVolume.fromFunction((x, y, z) => Math.hypot(x - 1, y - 2, z - 0.5) - 0.3, [0.65, 1.65, 0.15], [1.35, 2.35, 0.85], 0.0047);
    const mesh = marchingCubes(vol);
    for (let i = 0; i < mesh.vertexCount; i++) mesh.colors.fill(Math.round(255 * (mesh.positions[3 * i + 2] - 0.2) / 0.6), 3 * i, 3 * i + 3);
    return mesh;
  }
  const textEntry = (zip: Uint8Array): string => new TextDecoder().decode(readZip(zip)[0].data);
  const usdArray = (usda: string, decl: string): number[] => {
    const open = usda.indexOf(`${decl} = [`) + decl.length + 4;
    return usda.slice(open, usda.indexOf(']', open)).replace(/[()]/g, '').split(',').map(Number);
  };

  it('simplifies to the AR budget, aligns the package and puts the origin at the base centre', async () => {
    const mesh = sphere();
    expect(mesh.triangleCount).toBeGreaterThan(USDZ_MAX_TRIANGLES);
    const textures: number[] = [];
    const src: ExportSources = {
      ...sources(null, null),
      mesh,
      metric: { metric: false } as MetricInfo,
      encodeTexture: async (img) => {
        textures.push(img.width);
        return new Blob([new Uint8Array([0xff, 0xd8, 0xff, 0xd9])], { type: 'image/jpeg' });
      },
    };
    const blob = await buildExport('usdz', src);
    expect(blob.type).toBe('model/vnd.usdz+zip');
    expect(textures).toEqual([2048]);
    const zip = await bytes(blob);
    const entries = readZip(zip);
    expect(entries.map((e) => e.name)).toEqual(['model.usda', 'textures/albedo.jpg']);
    for (const e of entries) expect((e.data.byteOffset - zip.byteOffset) % 64).toBe(0);
    const usda = textEntry(zip);
    expect(usda).toContain('bool metricScale = 0');
    const faces = usdArray(usda, 'int[] faceVertexCounts');
    expect(faces.length).toBeLessThanOrEqual(USDZ_MAX_TRIANGLES);
    expect(usdArray(usda, 'texCoord2f[] primvars:st').length).toBe(6 * faces.length);
    // Y-up about the pivot: the sphere's bottom (z = 0.2) at y ≈ 0 (the 2nd-percentile height of
    // a sphere — z is uniform over its area — is 0.02 · 0.6 m above the bottom), centred in x and z
    const pts = usdArray(usda, 'point3f[] points');
    let minY = Infinity;
    let maxY = -Infinity;
    let sx = 0;
    let sz = 0;
    for (let i = 0; i < pts.length; i += 3) {
      minY = Math.min(minY, pts[i + 1]);
      maxY = Math.max(maxY, pts[i + 1]);
      sx += pts[i];
      sz += pts[i + 2];
    }
    expect(minY).toBeCloseTo(-0.012, 2);
    expect(maxY).toBeCloseTo(0.6 - 0.012, 2);
    expect(Math.abs(sx / (pts.length / 3))).toBeLessThan(0.01);
    expect(Math.abs(sz / (pts.length / 3))).toBeLessThan(0.01);
  });

  it('an object ROI moves the pivot to its box bottom centre on the support plane', async () => {
    const src: ExportSources = {
      ...sources(null, null),
      mesh: sphere(),
      objectRoi: { min: [0.9, 1.8, 0.11], max: [1.3, 2.2, 0.6], centre: [1.0, 2.1, 0.5], planeZ: 0.1, confidence: 0.8 },
      encodeTexture: async () => new Blob([new Uint8Array(4)], { type: 'image/png' }),
    };
    const zip = await bytes(await buildExport('usdz', src));
    expect(readZip(zip)[1].name).toBe('textures/albedo.png');
    const pts = usdArray(textEntry(zip), 'point3f[] points');
    let minX = Infinity;
    let minY = Infinity;
    for (let i = 0; i < pts.length; i += 3) {
      minX = Math.min(minX, pts[i]);
      minY = Math.min(minY, pts[i + 1]);
    }
    // box centre (1.1, 2.0), not the orbit centre (1.0, 2.1): sphere x ∈ [0.7, 1.3] − 1.1, bottom z 0.2 − planeZ 0.1
    expect(minX).toBeCloseTo(-0.4, 2);
    expect(minY).toBeCloseTo(0.1, 2);
  });

  it('stops on cancel (before the atlas, and after the texture encode)', async () => {
    await expect(buildExport('usdz', { ...sources(null, null), mesh: sphere() }, { cancelled: true })).rejects.toMatchObject({ name: 'CancelledError' });
    const token = { cancelled: false };
    const src: ExportSources = {
      ...sources(null, null),
      mesh: sphere(),
      encodeTexture: async () => {
        token.cancelled = true;
        return new Blob([new Uint8Array(4)], { type: 'image/jpeg' });
      },
    };
    await expect(buildExport('usdz', src, token)).rejects.toMatchObject({ name: 'CancelledError', message: 'export: cancelled' });
  });

  it('warns when the simplifier cannot reach the AR budget, and still exports', async () => {
    // 26 000 closed tetrahedra (104 k triangles): no edge collapse passes the link condition
    const N = 26_000;
    const corners = [0, 0, 0, 0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01];
    const faces = [0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2];
    const mesh: TriangleMesh = { vertexCount: 4 * N, triangleCount: 4 * N, positions: new Float32Array(12 * N), normals: new Float32Array(12 * N), colors: new Uint8Array(12 * N).fill(90), indices: new Uint32Array(12 * N) };
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < 4; k++) mesh.positions.set([corners[3 * k] + 0.05 * (i % 160), corners[3 * k + 1] + 0.05 * Math.floor(i / 160), corners[3 * k + 2]], 12 * i + 3 * k);
      for (let k = 0; k < 12; k++) mesh.indices[12 * i + k] = 4 * i + faces[k];
    }
    const logs: string[] = [];
    const src: ExportSources = {
      ...sources(null, null),
      mesh,
      encodeTexture: async () => new Blob([new Uint8Array(4)], { type: 'image/jpeg' }),
      log: (level, message) => logs.push(`${level}: ${message}`),
    };
    const usda = textEntry(await bytes(await buildExport('usdz', src)));
    expect(usdArray(usda, 'int[] faceVertexCounts').length).toBe(4 * N);
    expect(logs).toEqual([`warn: usdz: ${4 * N} triangles after simplifying (AR budget ${USDZ_MAX_TRIANGLES})`]);
  });

  it('needs a mesh and an image encoder', async () => {
    await expect(buildExport('usdz', sources(cloud(3), null))).rejects.toThrow(/No mesh yet/);
    // Node has no OffscreenCanvas: the default JPEG path reports it
    const tri: TriangleMesh = { vertexCount: 3, triangleCount: 1, positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), normals: new Float32Array(9), colors: new Uint8Array(9), indices: new Uint32Array([0, 1, 2]) };
    await expect(buildExport('usdz', { ...sources(null, null), mesh: tri })).rejects.toThrow(/OffscreenCanvas/);
  });
});
