/**
 * Stress tests of the SfM core: a 200-keyframe closed walk around a yard (loop
 * closure through the correspondence graph; ~95 m extent) in batch and live mode, and
 * a long close-range object orbit. Acceptance settings: 0.7 px noise, 25 % outlier
 * matches, focal initialised 8 % off, true k1 = 0.03, verification with the wrong camera.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import { CorrespondenceGraph } from './correspondence-graph';
import { reconstructionInvariantErrors } from './core-testutil';
import type { FeatureDatabase } from './database';
import { LiveMapper } from './live';
import { runIncrementalSfM, type MapperTimings } from './mapper';
import type { Reconstruction } from './reconstruction';
import { buildSyntheticDatabase, evaluateReconstruction, generateScene, type SyntheticScene, type SyntheticSceneKind } from './synthetic';
import type { CameraIntrinsics } from '../core/types';
import { perfSlack } from '../core/perf-slack';

interface Data {
  scene: SyntheticScene;
  db: FeatureDatabase;
  guess: CameraIntrinsics;
  verifyMs: number;
}

function makeData(kind: SyntheticSceneKind, numImages: number, seed: number, numPoints?: number): Data {
  const scene = generateScene({ kind, numImages, seed, k1: 0.03, noisePx: 0.7, outlierRatio: 0.25, descriptors: false, ...(numPoints ? { numPoints } : {}) });
  const guess = { ...scene.camera, f: scene.camera.f * 1.08, k1: 0, k2: 0 };
  const { db, stats } = buildSyntheticDatabase(scene, { camera: guess });
  return { scene, db, guess, verifyMs: stats.timeMs };
}

async function batch(data: Data): Promise<{ rec: Reconstruction; models: Reconstruction[]; ms: number; maxStall: number; timings?: MapperTimings }> {
  let timings: MapperTimings | undefined;
  let last = performance.now(),
    maxStall = 0;
  const timer = setInterval(() => {
    const t = performance.now();
    maxStall = Math.max(maxStall, t - last);
    last = t;
  }, 2);
  const t0 = performance.now();
  const models = await runIncrementalSfM(data.db, { intrinsics: data.guess, onModel: (_, t) => (timings ??= t) });
  const ms = performance.now() - t0;
  clearInterval(timer);
  return { rec: models[0], models, ms, maxStall, timings };
}

function liveStream(data: Data): { live: LiveMapper; ms: number[]; graph: CorrespondenceGraph } {
  const { db } = data;
  const graph = new CorrespondenceGraph();
  const live = new LiveMapper(db, graph, { intrinsics: data.guess });
  const ids = db.frameIds();
  const ms: number[] = [];
  for (const id of ids) {
    graph.addImage(id, db.features.get(id)!.count);
    for (const j of ids) {
      if (j >= id) break;
      const g = db.getTwoView(j, id);
      if (g && g.config !== 'degenerate' && g.inliers.length >= 30) graph.addTwoView(j, id, g.inliers);
    }
    const t0 = performance.now();
    live.addKeyframe(id);
    ms.push(performance.now() - t0);
  }
  return { live, ms, graph };
}

const fmt = (v: unknown): string => JSON.stringify(v, (_, x) => (typeof x === 'number' ? Math.round(x) : x));

describe('stress: 200-keyframe closed walk around a yard (loop closure)', () => {
  const N = 200;
  let data: Data;
  beforeAll(() => {
    data = makeData('loop', N, 3);
  }, 300_000);

  it('the scene closes the loop: the last keyframes are matched to the first ones', () => {
    let loopPairs = 0;
    for (const [a, b] of data.db.pairs()) if (b - a > N / 2) loopPairs++;
    expect(loopPairs).toBeGreaterThanOrEqual(5);
    expect(data.scene.extent).toBeGreaterThan(80);
  });

  it('runIncrementalSfM: one model, every keyframe, mm-level trajectory, self-calibrated', async () => {
    const { rec, models, ms, maxStall, timings } = await batch(data);
    const ev = evaluateReconstruction(rec, data.scene);
    const mre = rec.meanReprojectionError();
    console.log(
      `[loop batch] ${ms.toFixed(0)} ms (verification ${data.verifyMs.toFixed(0)} ms), max stall ${maxStall.toFixed(0)} ms | reg ${ev.numRegistered}/${N} pts ${rec.numPoints} obs ${rec.numObservations} track ${rec.meanTrackLength().toFixed(1)} | ATE ${ev.ate.toExponential(2)} (${ev.ateRelative.toExponential(2)} of ${data.scene.extent.toFixed(1)}) max ${ev.maxCenterError.toFixed(4)} rot ${ev.maxRotErrorDeg.toFixed(4)}° | f ${ev.focalError.toExponential(2)} k1 ${ev.k1.toFixed(4)} | mre ${mre.toFixed(3)} wrong ${(100 * ev.wrongObservationRate).toFixed(3)} % | ${fmt(timings)}`,
    );
    expect(models.length).toBe(1);
    expect(ev.numRegistered).toBe(N);
    expect(ev.ateRelative).toBeLessThan(2e-4);
    // loop closure: no drift accumulates along the 110 m loop
    expect(ev.maxCenterError / data.scene.extent).toBeLessThan(1e-3);
    expect(ev.maxRotErrorDeg).toBeLessThan(0.2);
    expect(ev.focalError).toBeLessThan(0.005);
    expect(Math.abs(ev.k1 - 0.03)).toBeLessThan(0.005);
    expect(mre).toBeLessThan(1);
    expect(ev.wrongObservationRate).toBeLessThan(0.01);
    expect(reconstructionInvariantErrors(rec, { graph: rec.correspondenceGraph! })).toEqual([]);
    // generous bounds (shared CI machines): ≈ 25 s / 0.7 s on an idle desktop
    expect(ms).toBeLessThan(240_000 * perfSlack());
    expect(maxStall).toBeLessThan(8_000);
  }, 600_000);

  it('LiveMapper: registers the stream as it arrives, bounded per-keyframe time, closes the loop after refine()', () => {
    const t0 = performance.now();
    const { live, ms, graph } = liveStream(data);
    const total = performance.now() - t0;
    const rec = live.reconstruction;
    const ev = evaluateReconstruction(rec, data.scene);
    const sorted = ms.slice().sort((a, b) => a - b);
    const t1 = performance.now();
    live.refine();
    const refineMs = performance.now() - t1;
    const ev2 = evaluateReconstruction(rec, data.scene);
    console.log(
      `[loop live] ${total.toFixed(0)} ms, per keyframe median ${sorted[N >> 1].toFixed(0)} p90 ${sorted[Math.floor(0.9 * N)].toFixed(0)} max ${sorted[N - 1].toFixed(0)} ms | reg ${ev.numRegistered}/${N} pending ${live.pendingIds().length} | ATE ${ev.ateRelative.toExponential(2)} max ${ev.maxCenterError.toFixed(4)} f ${ev.focalError.toExponential(2)} | refine ${refineMs.toFixed(0)} ms → ATE ${ev2.ateRelative.toExponential(2)} max ${ev2.maxCenterError.toFixed(4)} f ${ev2.focalError.toExponential(2)} mre ${rec.meanReprojectionError().toFixed(3)} | ${fmt(live.timings)}`,
    );
    expect(ev.numRegistered).toBeGreaterThanOrEqual(Math.ceil(0.98 * N));
    expect(ev.ateRelative).toBeLessThan(5e-4);
    expect(ev2.ateRelative).toBeLessThan(2e-4);
    expect(ev2.maxCenterError / data.scene.extent).toBeLessThan(1e-3);
    expect(ev2.focalError).toBeLessThan(0.005);
    expect(rec.meanReprojectionError()).toBeLessThan(1);
    expect(reconstructionInvariantErrors(rec, { graph })).toEqual([]);
    // median keyframe ≈ 30 ms idle; generous bound for loaded machines
    expect(sorted[N >> 1]).toBeLessThan(1_000);
  }, 600_000);
});

describe('stress: long close-range object orbit', () => {
  const N = 120;
  let data: Data;
  beforeAll(() => {
    data = makeData('orbit', N, 8, 4000);
  }, 300_000);

  it('runIncrementalSfM and LiveMapper reconstruct a 120-view orbit (3° steps, long tracks)', async () => {
    const { rec, ms, maxStall, timings } = await batch(data);
    const ev = evaluateReconstruction(rec, data.scene);
    console.log(
      `[orbit batch] ${ms.toFixed(0)} ms (verification ${data.verifyMs.toFixed(0)} ms), max stall ${maxStall.toFixed(0)} ms | reg ${ev.numRegistered}/${N} pts ${rec.numPoints} obs ${rec.numObservations} track ${rec.meanTrackLength().toFixed(1)} | ATE ${ev.ateRelative.toExponential(2)} max ${ev.maxCenterError.toExponential(2)} rot ${ev.maxRotErrorDeg.toFixed(4)}° | f ${ev.focalError.toExponential(2)} k1 ${ev.k1.toFixed(4)} mre ${rec.meanReprojectionError().toFixed(3)} | ${fmt(timings)}`,
    );
    expect(ev.numRegistered).toBe(N);
    expect(ev.ateRelative).toBeLessThan(2e-4);
    expect(ev.maxRotErrorDeg).toBeLessThan(0.1);
    expect(ev.focalError).toBeLessThan(0.005);
    expect(Math.abs(ev.k1 - 0.03)).toBeLessThan(0.005);
    expect(rec.meanReprojectionError()).toBeLessThan(1);
    expect(reconstructionInvariantErrors(rec, { graph: rec.correspondenceGraph! })).toEqual([]);
    expect(ms).toBeLessThan(240_000 * perfSlack());

    const t0 = performance.now();
    const { live, ms: kms, graph } = liveStream(data);
    live.refine();
    const total = performance.now() - t0;
    const lr = live.reconstruction;
    const el = evaluateReconstruction(lr, data.scene);
    const sorted = kms.slice().sort((a, b) => a - b);
    console.log(
      `[orbit live] ${total.toFixed(0)} ms incl. refine, per keyframe median ${sorted[N >> 1].toFixed(0)} p90 ${sorted[Math.floor(0.9 * N)].toFixed(0)} max ${sorted[N - 1].toFixed(0)} ms | reg ${el.numRegistered}/${N} | ATE ${el.ateRelative.toExponential(2)} f ${el.focalError.toExponential(2)} mre ${lr.meanReprojectionError().toFixed(3)}`,
    );
    expect(el.numRegistered).toBeGreaterThanOrEqual(Math.ceil(0.98 * N));
    expect(el.ateRelative).toBeLessThan(3e-4);
    expect(el.focalError).toBeLessThan(0.005);
    expect(reconstructionInvariantErrors(lr, { graph })).toEqual([]);
  }, 900_000);
});
