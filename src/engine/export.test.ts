/**
 * Point exports (src/engine/export.ts buildExport): 'ply-points' and points-only GLB use the dense
 * cloud when it has points, else the sparse cloud — an empty dense cloud (a degenerate densify)
 * does not hide the sparse one.
 */
import { describe, expect, it } from 'vitest';
import type { PointCloud } from '../core/types';
import { parseGLB } from '../io/glb';
import { decodePLY } from '../io/ply';
import { buildExport, type ExportSources } from './export';

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
