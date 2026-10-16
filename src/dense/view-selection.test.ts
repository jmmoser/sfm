import { describe, expect, it } from 'vitest';
import type { Pose } from '../core/types';
import { lookAt, pinholeIntrinsics } from './synthetic';
import { neighboursFromSelection, scaleWeight, selectSourceViews, triangulationAngleWeight } from './view-selection';
import { Rng } from '../math/random';

const log = (...a: unknown[]): void => console.log('[view-selection]', ...a);
const DEG = Math.PI / 180;

function visibleIn(pose: Pose, cam: { width: number; height: number; f: number; cx: number; cy: number }, X: number[]): boolean {
  const R = pose.R;
  const t = pose.t;
  const z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
  if (!(z > 0)) return false;
  const u = (cam.f * (R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0])) / z + cam.cx;
  const v = (cam.f * (R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1])) / z + cam.cy;
  return u >= 0 && v >= 0 && u <= cam.width - 1 && v <= cam.height - 1;
}

describe('selectSourceViews', () => {
  it('weights', () => {
    expect(triangulationAngleWeight(0, 3 * DEG, 45 * DEG)).toBe(0);
    expect(triangulationAngleWeight(1.5 * DEG, 3 * DEG, 45 * DEG)).toBeCloseTo(0.25, 10);
    expect(triangulationAngleWeight(20 * DEG, 3 * DEG, 45 * DEG)).toBe(1);
    expect(triangulationAngleWeight(60 * DEG, 3 * DEG, 45 * DEG)).toBeCloseTo(Math.exp(-1), 10);
    expect(scaleWeight(0.5)).toBe(0.5);
    expect(scaleWeight(1.5)).toBe(1);
    expect(scaleWeight(4)).toBe(0.5);
  });

  it('picks overlapping views with useful baselines; rejects zero-baseline and non-overlapping views', () => {
    const cam = pinholeIntrinsics(640, 480, 60 * DEG);
    // wall at y = 10
    const rng = new Rng(3);
    const pts: number[][] = [];
    for (let i = 0; i < 3000; i++) pts.push([-12 + 24 * rng.next(), 10 + 0.3 * rng.next(), -3 + 6 * rng.next()]);
    // cameras at y = 0 (x offsets), all looking along +y; view 9 looks backwards; view 10 is a far, coarse view
    const xs = [0, 0.02, 0.3, 1, 2, 3, 4, 8, -2];
    const poses: Pose[] = xs.map((x) => lookAt([x, 0, 0], [x, 10, 0]));
    poses.push(lookAt([0, 0, 0], [0, -10, 0]));
    poses.push(lookAt([0, -30, 0], [0, 10, 0]));
    const views = poses.map((pose, id) => ({ id: 100 + id, pose, cam }));
    const visibility = pts.map((X) => views.filter((v) => visibleIn(v.pose, cam, X)).map((v) => v.id));
    const xyz = Float64Array.from(pts.flat());
    const sel = selectSourceViews(views, { xyz, visibility }, { numSources: 5 });
    const s0 = sel.get(100)!;
    log(`ref 100 sources ${JSON.stringify(s0.sources)} scores ${s0.scores.map((x) => x.toFixed(0)).join(',')}, depth [${s0.depthMin.toFixed(2)}, ${s0.depthMax.toFixed(2)}]`);
    expect(s0.sources).not.toContain(101); // 0.1° baseline: hard-rejected
    expect(s0.sources).not.toContain(109); // looks the other way
    expect(s0.sources.length).toBe(5);
    for (const id of [103, 104, 105, 108]) expect(s0.sources).toContain(id); // 1–3 m baselines both sides
    // 0.3 m (1.7°) ranks below the 1–4 m baselines
    const rank = (id: number): number => (s0.sources.includes(id) ? s0.sources.indexOf(id) : 99);
    expect(rank(102)).toBeGreaterThan(rank(103));
    // depth range ≈ [0.8 · 10, 1.25 · 10.3]
    expect(s0.depthMin).toBeGreaterThan(0.8 * 9.99);
    expect(s0.depthMin).toBeLessThan(0.8 * 10.05);
    expect(s0.depthMax).toBeGreaterThan(1.25 * 10.2);
    expect(s0.depthMax).toBeLessThan(1.25 * 10.31);
    // every view gets ≥ 2 sources except the backwards one (nothing overlaps)
    for (const [id, s] of sel) {
      if (id === 109) continue;
      expect(s.sources.length).toBeGreaterThanOrEqual(2);
      expect(s.depthMin).toBeGreaterThan(0);
      expect(s.depthMax).toBeGreaterThan(s.depthMin);
    }
    const nb = neighboursFromSelection(sel);
    for (const [id, l] of nb) for (const j of l) expect(nb.get(j)!.includes(id)).toBe(true);
  });

  it('falls back to frustum points when the reference has no visible sparse points', () => {
    const cam = pinholeIntrinsics(320, 240, 60 * DEG);
    const rng = new Rng(5);
    const pts: number[][] = [];
    for (let i = 0; i < 500; i++) pts.push([-4 + 8 * rng.next(), 5 + rng.next(), -2 + 4 * rng.next()]);
    const poses = [-1, 0, 1, 2].map((x) => lookAt([x, 0, 0], [x, 5, 0]));
    const views = poses.map((pose, id) => ({ id, pose, cam }));
    // view 0 is not listed in any visibility list
    const visibility = pts.map((X) => views.filter((v) => v.id !== 0 && visibleIn(v.pose, cam, X)).map((v) => v.id));
    const sel = selectSourceViews(views, { xyz: Float64Array.from(pts.flat()), visibility });
    const s = sel.get(0)!;
    log(`fallback: sources ${JSON.stringify(s.sources)}, depth [${s.depthMin.toFixed(2)}, ${s.depthMax.toFixed(2)}]`);
    expect(s.sources.length).toBeGreaterThanOrEqual(2);
    expect(s.depthMin).toBeLessThan(5);
    expect(s.depthMax).toBeGreaterThan(6);
    expect(s.depthMax).toBeLessThan(8);
  });

  it('no sparse points at all: baseline-derived range', () => {
    const cam = pinholeIntrinsics(320, 240, 60 * DEG);
    const views = [0, 1, 2].map((x) => ({ id: x, pose: lookAt([x, 0, 0], [x, 5, 0]), cam }));
    const sel = selectSourceViews(views, { xyz: new Float64Array(0), visibility: [] });
    for (const s of sel.values()) {
      expect(s.depthMin).toBeGreaterThan(0);
      expect(s.depthMax).toBeGreaterThan(s.depthMin * 10);
    }
  });
});
