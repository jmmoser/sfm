import { describe, expect, it } from 'vitest';
import type { Pose } from '../core/types';
import { Rng } from '../math/random';
import { mergePairs, pairId, retrievalPairs, sequentialPairs, spatialPairs, type RetrievalSource, type SpatialCamera } from './pairs';
import { perfSlack } from '../core/perf-slack';

/** Camera at C looking along unit direction d (world), up hint +Z. */
function lookPose(C: number[], d: number[]): Pose {
  const f = d.map((v) => v / Math.hypot(d[0], d[1], d[2]));
  let r = [f[1], -f[0], 0];
  const rn = Math.hypot(r[0], r[1]) || 1;
  r = r.map((v) => v / rn);
  const dn = [f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0]];
  // R rows = camera axes in world (world → camera)
  const R = Float64Array.of(r[0], r[1], r[2], dn[0], dn[1], dn[2], f[0], f[1], f[2]);
  const t = Float64Array.of(-(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]), -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]), -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]));
  return { R, t };
}

describe('pair selection', () => {
  it('pairId is order-independent and collision-free for large ids', () => {
    expect(pairId(3, 7)).toBe(pairId(7, 3));
    const seen = new Set<number>();
    for (let a = 0; a < 60; a++) for (let b = a + 1; b < 60; b++) seen.add(pairId(a * 1_000_003, b * 1_000_003));
    expect(seen.size).toBe((60 * 59) / 2);
  });

  it('sequentialPairs: window and exponential offsets', () => {
    const ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
    const p = sequentialPairs(ids, 2);
    expect(p.length).toBe(9 + 8);
    expect(p[0]).toEqual([10, 11]);
    expect(p).toContainEqual([17, 19]);
    expect(p).not.toContainEqual([10, 13]);
    const e = sequentialPairs(ids, 2, { exponential: true });
    expect(e).toContainEqual([10, 14]); // offset 4
    expect(e).toContainEqual([10, 18]); // offset 8
    expect(e).not.toContainEqual([10, 15]);
    expect(sequentialPairs([5, 3], 3)).toEqual([[3, 5]]);
    expect(sequentialPairs(ids, 0)).toEqual([]);
  });

  it('retrievalPairs: top-k with exclusion and score floor', () => {
    const sim = (a: number, b: number) => (Math.abs(a - b) === 50 ? 0.9 : 1 / (1 + Math.abs(a - b)));
    const index: RetrievalSource = {
      imageIds: () => Array.from({ length: 100 }, (_, i) => i),
      query(id, k, exclude) {
        const pred = typeof exclude === 'function' ? exclude : (x: number) => !!exclude && [...exclude].includes(x);
        return this.imageIds()
          .filter((j) => j !== id && !pred(j))
          .map((j) => ({ id: j, score: sim(id, j) }))
          .sort((a, b) => b.score - a.score)
          .slice(0, k);
      },
    };
    const p = retrievalPairs(index, 2, { exclude: (a, b) => Math.abs(a - b) <= 5 });
    // every image's best non-neighbour is its "loop" partner at ±50
    for (let i = 0; i < 50; i++) expect(p).toContainEqual([i, i + 50]);
    expect(p.every(([a, b]) => b - a > 5)).toBe(true);
    const q = retrievalPairs(index, 3, { ids: [0], minScore: 0.5 });
    expect(q).toEqual([
      [0, 1],
      [0, 50],
    ]);
  });

  it('spatialPairs: nearest cameras with similar viewing direction', () => {
    // 36 cameras on a circle of radius 2 looking at the centre, plus 4 looking outward
    const cams: SpatialCamera[] = [];
    for (let i = 0; i < 36; i++) {
      const a = (2 * Math.PI * i) / 36;
      const C = [2 * Math.cos(a), 2 * Math.sin(a), 0.3];
      cams.push({ id: i, pose: lookPose(C, [-C[0], -C[1], -0.3]) });
    }
    for (let i = 0; i < 4; i++) {
      const a = (2 * Math.PI * i) / 36;
      const C = [2 * Math.cos(a), 2 * Math.sin(a), 0.3];
      cams.push({ id: 100 + i, pose: lookPose(C, C) });
    }
    const p = spatialPairs(cams, 2, 30);
    expect(p).toContainEqual([0, 1]);
    expect(p).toContainEqual([0, 35]);
    // the outward cameras sit at the same place but look the other way → never paired with inward ones
    expect(p.some(([a, b]) => (a < 100) !== (b < 100))).toBe(false);
    expect(p).toContainEqual([100, 101]);
    // angle limit 5°: neighbours are 10° apart → no pairs between inward cameras
    expect(spatialPairs(cams, 2, 5).filter(([a, b]) => a < 100 && b < 100)).toEqual([]);
    // max distance
    expect(spatialPairs(cams, 4, 60, { maxDistance: 0.4 }).every(([a, b]) => a >= 100 || Math.min(Math.abs(a - b), 36 - Math.abs(a - b)) <= 1)).toBe(true);
  });

  it('spatialPairs with depths uses look-at points', () => {
    // two cameras 10 m apart looking at the same point (converging), and one close by looking elsewhere
    const target = [0, 10, 0];
    const cams: SpatialCamera[] = [
      { id: 1, pose: lookPose([-3, 0, 0], [target[0] + 3, target[1], 0]), depth: Math.hypot(3, 10) },
      { id: 2, pose: lookPose([3, 0, 0], [target[0] - 3, target[1], 0]), depth: Math.hypot(3, 10) },
      { id: 3, pose: lookPose([-2.5, 0, 0], [0.3, 1, 0]), depth: 1 },
    ];
    const p = spatialPairs(cams, 1, 45);
    expect(p).toContainEqual([1, 2]);
  });

  it('mergePairs: dedupe, normalise, exclude, sort; scales to many pairs', () => {
    const m = mergePairs(
      [
        [
          [3, 1],
          [1, 3],
          [2, 2],
        ],
        [
          [5, 4],
          [1, 3],
          [0, 9],
        ],
      ],
      (a, b) => a === 0 && b === 9,
    );
    expect(m).toEqual([
      [1, 3],
      [4, 5],
    ]);
    const rng = new Rng(1);
    const big: [number, number][] = [];
    for (let i = 0; i < 200000; i++) big.push([rng.int(3000), rng.int(3000)]);
    const t0 = performance.now();
    const r = mergePairs([big, big]);
    const ms = performance.now() - t0;
    expect(ms).toBeLessThan(1500 * perfSlack());
    let sorted = true;
    for (let i = 1; i < r.length; i++) sorted &&= r[i - 1][0] < r[i][0] || (r[i - 1][0] === r[i][0] && r[i - 1][1] < r[i][1]);
    expect(sorted).toBe(true);
    expect(r.length).toBeGreaterThan(190000);
  });
});
