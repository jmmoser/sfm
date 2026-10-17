import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics } from '../../core/types';
import { bbox, keyframeSourceIndex, mapCamerasToFrames, medianIntrinsics, normalizeIntrinsics, poseToJSON, stageTimings, type CameraLike, type DeliveredRef } from './result';
import type { ProgressRecord } from './types';

const K: CameraIntrinsics = { width: 720, height: 960, f: 750, cx: 360, cy: 480, k1: -0.03, k2: 0.005 };
const pose = (x: number): { R: Float64Array; t: Float64Array } => ({ R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(x, 0, 0) });

/** 10 delivered frames (dataset indices 0, 4, 8, …), keyframes at delivered 0, 3, 6, 9; engine counter = delivered. */
function delivered(engineOffset = 0): DeliveredRef[] {
  return Array.from({ length: 10 }, (_, k) => ({ delivered: k, index: 4 * k, t: 10_000 + 500 * k, engineFrameIndex: k + engineOffset, keyframe: k % 3 === 0 }));
}

describe('mapCamerasToFrames', () => {
  it('uses frameIndex = FrameInput.index (SparseCameraEx)', () => {
    const cams: CameraLike[] = [3, 0, 9].map((k, i) => ({ id: i, pose: pose(k), intrinsics: K, frameIndex: k, t: 10_000 + 500 * k }));
    const m = mapCamerasToFrames(cams, delivered(100));
    expect(m.mapping).toBe('frameIndex');
    expect(m.images.map((x) => x.index)).toEqual([0, 12, 36]);
    expect(m.images[1].pose.t).toEqual([3, 0, 0]);
    expect(m.unmapped).toEqual([]);
  });

  it('falls back to LiveUpdate.frameIndex, then to the capture time', () => {
    const a = mapCamerasToFrames([{ id: 0, pose: pose(0), intrinsics: K, frameIndex: 105 }], delivered(100));
    expect(a.images.map((x) => x.index)).toEqual([20]);
    const b = mapCamerasToFrames([{ id: 7, pose: pose(0), intrinsics: K, t: 10_000 + 500 * 6 + 0.4 }], delivered());
    expect(b.mapping).toBe('t');
    expect(b.images.map((x) => x.index)).toEqual([24]);
    const c = mapCamerasToFrames([{ id: 7, pose: pose(0), intrinsics: K, t: 10_250 }], delivered());
    expect(c.unmapped).toEqual([7]);
  });

  it('treats plain ids as keyframe ordinals when they fit', () => {
    const cams: CameraLike[] = [0, 1, 3].map((id) => ({ id, pose: pose(id), intrinsics: K }));
    const m = mapCamerasToFrames(cams, delivered());
    expect(m.mapping).toBe('keyframe-ordinal');
    // keyframes: delivered 0, 3, 6, 9 → dataset 0, 12, 24, 36
    expect(m.images.map((x) => [x.id, x.index])).toEqual([
      [0, 0],
      [1, 12],
      [3, 36],
    ]);
  });

  it('reports unmappable and duplicate cameras', () => {
    const cams: CameraLike[] = [
      { id: 0, pose: pose(0), intrinsics: K, frameIndex: 3 },
      { id: 1, pose: pose(1), intrinsics: K, frameIndex: 3 },
      { id: 2, pose: pose(2), intrinsics: K, frameIndex: 99 },
    ];
    const m = mapCamerasToFrames(cams, delivered());
    expect(m.images).toHaveLength(1);
    expect(m.duplicates).toEqual([1]);
    expect(m.unmapped).toEqual([2]);
    // ids beyond the keyframe count are not ordinals; id-as-frame needs a keyframe there.
    const n = mapCamerasToFrames([{ id: 6, pose: pose(0), intrinsics: K }, { id: 5, pose: pose(0), intrinsics: K }], delivered());
    expect(n.images.map((x) => x.index)).toEqual([24]);
    expect(n.unmapped).toEqual([5]);
    expect(n.mapping).toBe('id-as-frame');
  });
});

describe('intrinsics', () => {
  it('keeps matching sizes, rescales by the §2 rule, flags transposed ones', () => {
    expect(normalizeIntrinsics(K, 720, 960)).toEqual({ intrinsics: { f: 750, cx: 360, cy: 480, k1: -0.03, k2: 0.005, width: 720, height: 960 }, note: null });
    const big = { ...K, width: 1440, height: 1920, f: 1500, cx: 720.5, cy: 960.5 };
    const r = normalizeIntrinsics(big, 720, 960);
    expect(r.intrinsics.f).toBeCloseTo(750, 12);
    expect(r.intrinsics.cx).toBeCloseTo((720.5 + 0.5) * 0.5 - 0.5, 12);
    expect(r.note).toMatch(/rescaled/);
    const tr = normalizeIntrinsics({ ...K, width: 960, height: 720, cx: 480, cy: 361 }, 720, 960);
    expect(tr.intrinsics.cx).toBe(361);
    expect(tr.intrinsics.cy).toBe(480);
    expect(tr.note).toMatch(/TRANSPOSED/);
  });

  it('medianIntrinsics', () => {
    expect(medianIntrinsics([])).toBeNull();
    const m = medianIntrinsics([K, { ...K, f: 760 }, { ...K, f: 740 }, { ...K, f: 1000 }])!;
    expect(m.f).toBe(755);
    expect(m.k1).toBe(-0.03);
  });
});

describe('stageTimings / misc', () => {
  it('stage duration runs until the next stage of the same step or the step end', () => {
    const p = (step: string, stage: string, ms: number, progress = 0.5): ProgressRecord => ({ step, stage, uiStage: null, progress, ms });
    const st = stageTimings([p('refine', 'matching', 10), p('refine', 'matching', 40), p('refine', 'reconstruction', 50), p('refine', 'reconstruction', 70), p('densify', 'depth_maps', 100)], {
      refine: 90,
      densify: 160,
    });
    expect(st.map((s) => [s.step, s.stage, s.durationMs, s.events])).toEqual([
      ['refine', 'matching', 40, 2],
      ['refine', 'reconstruction', 40, 2],
      ['densify', 'depth_maps', 60, 1],
    ]);
  });

  it('poseToJSON and bbox', () => {
    expect(poseToJSON(pose(2))).toEqual({ R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [2, 0, 0] });
    expect(bbox(Float32Array.of(0, 1, 2, -1, NaN, 0, 3, -2, 1), 3)).toEqual({ min: [0, -2, 1], max: [3, 1, 2] });
    expect(bbox(new Float32Array(0), 0)).toBeNull();
  });
});

describe('keyframeSourceIndex (live-trajectory mapping of keyframes)', () => {
  it('a held (best-of-window) keyframe maps to its own, earlier frame; others to the update frame', () => {
    const plan = [0, 4, 8, 12, 16].map((index) => ({ index }));
    const of = (d: number): number | undefined => plan[d]?.index;
    // the update of delivered frame 3 (dataset 12) created a keyframe from delivered frame 1 (dataset 4)
    expect(keyframeSourceIndex({ keyframeFrameIndex: 1, total: 5 }, of, 12)).toBe(4);
    expect(keyframeSourceIndex({ total: 5 }, of, 12)).toBe(12);
    expect(keyframeSourceIndex(undefined, of, 12)).toBe(12);
    expect(keyframeSourceIndex({ keyframeFrameIndex: 99 }, of, 12)).toBe(12); // unknown frame
  });
});
