import { describe, expect, it } from 'vitest';
import type { SparseResult } from '../../workers/protocol';
import { exportAvailability } from './viewer';

const cloud = (count: number) => ({ count, positions: new Float32Array(3 * count), colors: new Uint8Array(3 * count) });
const sparse = { points: cloud(2), cameras: [{}, {}, {}], stats: {} } as unknown as SparseResult;

describe('exportAvailability', () => {
  it('point exports describe the dense cloud when there is one', () => {
    const r = { sparse, dense: cloud(5), mesh: null };
    expect(exportAvailability(r, 'ply-points')).toEqual({ ok: true, sub: 'Dense cloud · 5 points, normals, colours' });
    expect(exportAvailability(r, 'glb')).toEqual({ ok: true, sub: 'Coloured point cloud (glTF binary)' });
  });
  it('an empty dense cloud (fusion kept no point) counts as none: the sparse points are exported', () => {
    const r = { sparse, dense: cloud(0), mesh: null };
    expect(exportAvailability(r, 'ply-points')).toEqual({ ok: true, sub: 'Sparse points · 2' });
    expect(exportAvailability(r, 'glb').ok).toBe(true);
    expect(exportAvailability({ sparse: null, dense: cloud(0), mesh: null }, 'ply-points')).toEqual({ ok: false, sub: 'No points yet' });
    expect(exportAvailability({ sparse: null, dense: cloud(0), mesh: null }, 'glb')).toEqual({ ok: false, sub: 'Nothing to export yet' });
  });
  it('mesh and COLMAP exports need their model', () => {
    expect(exportAvailability({ sparse, dense: null, mesh: null }, 'ply-mesh').ok).toBe(false);
    expect(exportAvailability({ sparse, dense: null, mesh: null }, 'colmap-zip')).toEqual({ ok: true, sub: 'Cameras, sparse model and 3 photos' });
    expect(exportAvailability({ sparse: null, dense: null, mesh: null }, 'colmap-zip').ok).toBe(false);
  });
  it('splat exports (PLY and SPZ) need the photoreal model', () => {
    const splat = { count: 1200 } as unknown as NonNullable<Parameters<typeof exportAvailability>[0]['splat']>;
    for (const f of ['ply-splat', 'spz'] as const) {
      expect(exportAvailability({ sparse, dense: null, mesh: null }, f)).toEqual({ ok: false, sub: 'Build the photoreal model first' });
      expect(exportAvailability({ sparse, dense: null, mesh: null, splat }, f).ok).toBe(true);
    }
    expect(exportAvailability({ sparse, dense: null, mesh: null, splat }, 'spz').sub).toMatch(/^1.2k Gaussians · compact/);
  });
});
