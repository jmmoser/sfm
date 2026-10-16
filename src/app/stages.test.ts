import { describe, expect, it } from 'vitest';
import { STAGES, estimateRemainingMs, stageIdFor, stageWeights, stagesForSteps, type StageId, type StageState } from './stages';

describe('stageIdFor', () => {
  it.each([
    ['matching', 'matching'],
    ['Matching', 'matching'],
    ['feature extraction', 'matching'],
    ['pair verification', 'matching'],
    ['retrieval', 'matching'],
    ['reconstruction', 'reconstruction'],
    ['incremental sfm', 'reconstruction'],
    ['global BA', 'reconstruction'],
    ['bundle adjustment', 'reconstruction'],
    ['retriangulate', 'reconstruction'],
    ['alignment', 'alignment'],
    ['gravity / scale', 'alignment'],
    ['gps', 'alignment'],
    ['depth_maps', 'depth_maps'],
    ['Depth maps', 'depth_maps'],
    ['patchmatch', 'depth_maps'],
    ['undistort', 'depth_maps'],
    ['fusion', 'fusion'],
    ['depth map fusion', 'fusion'],
    ['tsdf', 'meshing'],
    ['marching cubes', 'meshing'],
    ['meshing', 'meshing'],
  ])('%s → %s', (name, id) => {
    expect(stageIdFor(name)).toBe(id);
  });
  it('unknown / empty → null', () => {
    expect(stageIdFor('')).toBeNull();
    expect(stageIdFor('frobnicate')).toBeNull();
  });
});

describe('stages', () => {
  it('six stages in pipeline order, grouped by step', () => {
    expect(STAGES.map((s) => s.id)).toEqual(['matching', 'reconstruction', 'alignment', 'depth_maps', 'fusion', 'meshing']);
    expect(stagesForSteps(['refine'])).toEqual(['matching', 'reconstruction', 'alignment']);
    expect(stagesForSteps(['densify', 'mesh'])).toEqual(['depth_maps', 'fusion', 'meshing']);
  });
  it('dense stages weigh more at higher quality', () => {
    expect(stageWeights('max').depth_maps).toBeGreaterThan(stageWeights('fast').depth_maps);
    expect(stageWeights('max').matching).toBe(stageWeights('fast').matching);
  });
});

function st(id: StageId, status: StageState['status'], progress = 0, startedAt: number | null = null, endedAt: number | null = null): StageState {
  return { id, status, progress, startedAt, endedAt };
}

describe('estimateRemainingMs', () => {
  const w = { matching: 1, reconstruction: 1, alignment: 1, depth_maps: 2, fusion: 1, meshing: 1 } as Record<StageId, number>;

  it('unknown before any measurable progress', () => {
    expect(estimateRemainingMs([st('matching', 'active', 0.01, 0), st('reconstruction', 'pending')], w, 1000)).toBeNull();
    expect(estimateRemainingMs([st('matching', 'pending')], w, 0)).toBeNull();
  });

  it('extrapolates the active stage and the pending ones from its rate', () => {
    // matching 50 % after 10 s → 10 s left in matching; 1 weight unit = 20 s ⇒ +20 s for reconstruction.
    const r = estimateRemainingMs([st('matching', 'active', 0.5, 0), st('reconstruction', 'pending')], w, 10_000);
    expect(r).toBeCloseTo(30_000, 6);
  });

  it('uses finished stages as the rate', () => {
    // matching took 10 s (weight 1); depth maps (weight 2) pending ⇒ 20 s; reconstruction just started.
    const r = estimateRemainingMs([st('matching', 'done', 1, 0, 10_000), st('reconstruction', 'active', 0, 10_000), st('depth_maps', 'pending')], w, 10_000);
    expect(r).toBeCloseTo(10_000 + 20_000, 6);
  });

  it('decreases as time passes on a steady pipeline', () => {
    const at = (t: number): number => {
      const p = Math.min(1, t / 20_000);
      return estimateRemainingMs([st('matching', 'done', 1, 0, 10_000), st('reconstruction', 'active', p, 10_000), st('alignment', 'pending')], w, 10_000 + t)!;
    };
    expect(at(5000)).toBeGreaterThan(at(10_000));
    expect(at(10_000)).toBeGreaterThan(0);
  });

  it('zero when everything is done', () => {
    expect(estimateRemainingMs([st('matching', 'done', 1, 0, 5000), st('reconstruction', 'done', 1, 5000, 9000)], w, 9000)).toBe(0);
  });
});
