import { beforeAll, describe, expect, it } from 'vitest';
import { runIncrementalSfM, type MapperTimings } from './mapper';
import { evaluateReconstruction } from './synthetic';
import { makeE2EData, streamLive, type E2EData } from './e2e-testutil';

const N = 60;

describe('SfM end-to-end: orbit (60 images, 0.7 px noise, 25 % outlier matches, f +8 %, k1 = 0.03)', () => {
  let data: E2EData;
  beforeAll(() => {
    data = makeE2EData('orbit', N, 3);
  }, 120_000);

  it('runIncrementalSfM registers all images with ATE < 0.5 % of the extent, focal error < 1 %, reprojection < 1 px', async () => {
    let timings: MapperTimings | undefined;
    // longest event-loop stall during the run (worker responsiveness)
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
    expect(models.length).toBe(1);
    const rec = models[0];
    const ev = evaluateReconstruction(rec, data.scene);
    const mre = rec.meanReprojectionError();
    console.log(
      `[orbit batch] ${ms.toFixed(0)} ms (verification ${data.verifyMs.toFixed(0)} ms) reg ${ev.numRegistered}/${N} pts ${rec.numPoints} track ${rec.meanTrackLength().toFixed(1)} | ATE ${ev.ate.toExponential(2)} (${(100 * ev.ateRelative).toFixed(4)} % of ${data.scene.extent.toFixed(2)}) rot ${ev.maxRotErrorDeg.toFixed(4)}° | f err ${(100 * ev.focalError).toFixed(4)} % k1 ${ev.k1.toFixed(4)} k2 ${ev.k2.toFixed(4)} | mre ${mre.toFixed(3)} px | wrong obs ${(100 * ev.wrongObservationRate).toFixed(3)} %`,
    );
    console.log(`[orbit batch] max event-loop stall ${maxStall.toFixed(0)} ms; timings ${JSON.stringify(timings, (_, v) => (typeof v === 'number' ? Math.round(v) : v))}`);
    expect(ev.numRegistered).toBe(N);
    expect(ev.ateRelative).toBeLessThan(0.005);
    expect(ev.focalError).toBeLessThan(0.01);
    expect(Math.abs(ev.k1 - 0.03)).toBeLessThan(0.005);
    expect(mre).toBeLessThan(1);
    expect(ev.wrongObservationRate).toBeLessThan(0.01);
  }, 300_000);

  it('LiveMapper on the keyframe stream registers ≥ 95 % with bounded per-keyframe time', () => {
    const t0 = performance.now();
    const { live, ms } = streamLive(data);
    const total = performance.now() - t0;
    const rec = live.reconstruction;
    const ev = evaluateReconstruction(rec, data.scene);
    const sorted = ms.slice().sort((a, b) => a - b);
    console.log(
      `[orbit live] ${total.toFixed(0)} ms, per keyframe median ${sorted[sorted.length >> 1].toFixed(0)} p90 ${sorted[Math.floor(0.9 * sorted.length)].toFixed(0)} max ${sorted[sorted.length - 1].toFixed(0)} ms | reg ${rec.numRegisteredImages}/${N} pts ${rec.numPoints} | ATE ${(100 * ev.ateRelative).toFixed(4)} % f err ${(100 * ev.focalError).toFixed(4)} % k1 ${ev.k1.toFixed(4)} mre ${rec.meanReprojectionError().toFixed(3)} px | ${JSON.stringify(live.timings, (_, v) => (typeof v === 'number' ? Math.round(v) : v))}`,
    );
    expect(rec.numRegisteredImages).toBeGreaterThanOrEqual(Math.ceil(0.95 * N));
    expect(ev.ateRelative).toBeLessThan(0.005);
    expect(ev.focalError).toBeLessThan(0.01);
    expect(rec.meanReprojectionError()).toBeLessThan(1);
    const snap = live.snapshot();
    expect(snap.points.length).toBe(3 * rec.numPoints);
    expect(snap.colors.length).toBe(3 * rec.numPoints);
    expect(snap.poses.length).toBe(rec.numRegisteredImages);
  }, 300_000);
});
