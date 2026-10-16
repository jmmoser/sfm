import { describe, expect, it } from 'vitest';
import { loadSynthDataset } from './dataset';

describe('loadSynthDataset', () => {
  it('loads JSON files relative to the base URL; geo.json is optional', async () => {
    const files: Record<string, unknown> = {
      '/d/meta.json': { name: 'x', intrinsics: { f: 500, cx: 1, cy: 2, k1: 0, k2: 0, width: 4, height: 3 } },
      '/d/frames.json': [{ index: 0, file: 'frames/000000.jpg', t: 25, tTrue: 0, pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [1, 2, 3] }, center: [0, 0, 0], gain: 1 }],
      '/d/imu.json': [],
      '/d/heading.json': [],
    };
    const fetchFn = async (u: string) => ({
      ok: u in files,
      status: u in files ? 200 : 404,
      json: async () => files[u],
      arrayBuffer: async () => new ArrayBuffer(0),
    });
    const ds = await loadSynthDataset('/d/', fetchFn);
    expect(ds.geo).toBeNull();
    expect(ds.frameUrl(0)).toBe('/d/frames/000000.jpg');
    expect(Array.from(ds.pose(0).t)).toEqual([1, 2, 3]);
    expect(ds.intrinsics().f).toBe(500);
    await expect(loadSynthDataset('/missing', fetchFn)).rejects.toThrow(/meta.json/);
  });
});
