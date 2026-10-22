/**
 * The engine's splat stage on a tiny synthetic orbit (src/engine/testutil.ts, CPU features, GPU
 * PatchMatch + splats; SwiftShader or E2E_GPU=host, workload: `SPLAT`):
 *  - persistent session (real IndexedDB): densify, splat cancelled mid-training (the readbacks of
 *    its checkpoint dropped first: read again, never stored stale), the engine dropped without
 *    cleanup ("killed"), a NEW engine reports the resumable iteration, resumes from the
 *    checkpoint (its first preview shows the checkpoint) and is ended early ("Use it now"): the
 *    model is persisted with its checkpoint kept (and its run key); a third engine reports it as
 *    continuable, continues to the end, clears the checkpoints and exports 'ply-splat';
 *  - in-memory session without a dense cloud: previews (the first before the training, PSNR
 *    rising to the final model's), dropped GPU work rolls back to the last checkpoint, a destroyed device is replaced
 *    and training continues from the checkpoint; dropped readbacks of the appearance normalisation
 *    are read again.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { ProgressEvent } from '../core/types';
import { getGPU } from '../gpu/device';
import { SessionStore } from '../io/storage';
import { decodeSplatPLY } from '../splat/ply';
import type { SplatTrainer } from '../splat/trainer';
import { COLOUR_FLOATS } from '../splat/wgsl/loss';
import type { EngineOptions, SplatPreviewDetail, SplatTrainingDetail } from '../workers/protocol';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import { unpackSplatParams, type EngineStateRecord } from './persistence';
import { frameInputs, makeCapture, type SynthCapture } from './testutil';
import type { EngineConfig } from './types';

let cap: SynthCapture;
beforeAll(() => {
  cap = makeCapture({ frames: 44, arcDeg: 55 });
});

/**
 * Training schedule of every test, sized for SwiftShader (4 CPU cores) within the 600 s timeout.
 * SwiftShader per iteration (dense-seeded model, 16.5 k Gaussians): 2.2 s at 240×320 (loss 0.62 s,
 * rasteriser backward 0.97 s, forward 0.36 s: per-pixel work) and 1.0 s at 120×160; 6 k
 * Gaussians: 1.7 s / 0.67 s. 300 iterations at 240×320 (checkpoints every 100) took 725 / 528 /
 * 1066 s per test alone (160 + 200, ≈ 380 and 800 trained iterations); this schedule 280 / 143 /
 * 267 s alone, 377 / 182 / 281 s next to four other GPU test files (timings vary with the load).
 * 120×160 views (the 240×320 dense depth maps are resampled to the targets), 200 iterations,
 * checkpoints at 0, 50, 100, 150 (a later checkpoint after the first mid-run one) and MCMC
 * refinements at 100 and 150 (refineEvery 50: one in the checkpoint at 100 that is resumed or
 * rolled back to, one after it, as at 100 / 200 of the 300 iterations before).
 */
const SPLAT = { iterations: 200, maxGaussians: 30_000, maxImageSize: 160, checkpointEvery: 50, chunk: 10, trainer: { mcmc: { refineEvery: 50 } } };

function config(extra: Partial<EngineConfig> = {}): EngineConfig {
  return {
    pipeline: createCpuFeaturePipeline({ maxFeatures: 2048, maxImageSize: 1024 }),
    featureBackend: 'cpu',
    mapping: 'inline',
    dense: { maxImageSize: 320, maxViews: 8 },
    splat: SPLAT,
    ...extra,
  };
}

async function captureAndRefine(e: PhotogrammetryEngine): Promise<void> {
  await e.init();
  for (const f of frameInputs(cap)) await e.processFrame(f);
  await e.finishCapture();
  await e.refine();
}

const training = (p: ProgressEvent): SplatTrainingDetail | null => (p.stage === 'splat_training' && p.detail ? (p.detail as SplatTrainingDetail) : null);
const preview = (p: ProgressEvent): SplatPreviewDetail | null => (p.stage === 'splat_preview' && p.detail && p.image ? (p.detail as SplatPreviewDetail) : null);
/** The training reports of a run from iteration `from` to `to` (one per `SPLAT.chunk`; no chunk crosses a checkpoint). */
const reports = (from: number, to: number): number[] => Array.from({ length: (to - from) / SPLAT.chunk + 1 }, (_, i) => from + i * SPLAT.chunk);

describe('engine splat stage', () => {
  it('cancel → kill → a new engine resumes from the checkpoint, persists and exports', async () => {
    const id = `splat-resume-${Date.now()}`;
    const opts: EngineOptions = { mode: 'object', quality: 'fast', persist: true, sessionId: id, hfovPriorDeg: 60 };
    try {
      let trainerA: SplatTrainer | null = null;
      const a = new PhotogrammetryEngine(opts, config({ splatDebug: { onTrainer: (t) => (trainerA = t) } }));
      await captureAndRefine(a);
      const cloud = await a.densify();
      expect(cloud.count).toBeGreaterThan(500);
      const cancel = { cancelled: false };
      const seen: SplatTrainingDetail[] = [];
      const stages = new Set<string>();
      const onProgress = (p: ProgressEvent): void => {
        stages.add(p.stage);
        const d = training(p);
        if (!d) return;
        seen.push(d);
        // the checkpoint at iteration 100 (written right after this report) loses its first two
        // readbacks (iOS: the app went to the background): its staging data would be stale
        if (d.iteration === 100) trainerA!.simulateDroppedWork({ readbacks: 2 });
        // cancelled between the checkpoints at 100 and 150 (< 100 iterations after the last one:
        // no checkpoint on cancel)
        if (d.iteration >= 120) cancel.cancelled = true;
      };
      await expect(a.splat(undefined, onProgress, cancel)).rejects.toMatchObject({ name: 'CancelledError' });
      expect([...stages]).toEqual(['splat_prepare', 'splat_training']);
      expect(seen[0].iterations).toBe(200);
      expect(seen.every((d) => Number.isFinite(d.loss) && d.itPerSec > 0 && d.gaussians > 0)).toBe(true);
      // the checkpoint at 100 holds the model grown by the MCMC refinement at 100
      expect(seen.find((d) => d.iteration === 100)!.gaussians).toBeGreaterThan(seen[0].gaussians);
      const prog = await a.engineStore!.getValue<{ iteration: number; slot: number }>('splat/progress');
      expect(prog?.iteration).toBe(100);
      expect((await a.engineStore!.listBinaryKeys(`splat/ckpt${prog!.slot}/`)).length).toBe(4);
      // the checkpoint holds the trainer's data, not stale staging contents (loss sums, stamps)
      const colour = new Float32Array((await a.engineStore!.getBinary(`splat/ckpt${prog!.slot}/colour`))!);
      for (let v = 0; v < colour.length / COLOUR_FLOATS; v++)
        for (const k of [0, 5, 10]) expect(Math.abs(colour[v * COLOUR_FLOATS + k] - 1)).toBeLessThan(0.5);
      const params = unpackSplatParams((await a.engineStore!.getBinary(`splat/ckpt${prog!.slot}/params`))!);
      expect(params.means.every((x) => Number.isFinite(x) && Math.abs(x) < 100)).toBe(true);
      a.dispose(); // "killed": no cleanup

      // reload: a new engine opens the session, resumes (its first preview shows the checkpoint)
      // and is ended early ("Use it now" at the report of iteration 130, past finishFrom = 60)
      const b = new PhotogrammetryEngine({ mode: 'auto', quality: 'fast' }, config());
      const sum = await b.loadSession(id);
      expect(sum.stats.splatResumableIteration).toBe(100);
      expect(sum.splat ?? null).toBeNull();
      const finish = { cancelled: false, finishEarly: false };
      const resumed: SplatTrainingDetail[] = [];
      const order: string[] = [];
      const previewsB: SplatPreviewDetail[] = [];
      const early = await b.splat(
        { preview: true },
        (p) => {
          const d = training(p);
          const v = preview(p);
          if (d || v) order.push(p.stage);
          if (v) previewsB.push(v);
          if (!d) return;
          resumed.push(d);
          if (d.iteration >= 130) finish.finishEarly = true;
        },
        finish,
      );
      console.log(`[splat-engine] resumed, ended early: ${JSON.stringify(early)}`);
      expect(previewsB[0].iteration).toBe(100);
      expect(order[0]).toBe('splat_preview');
      expect(resumed[0]).toMatchObject({ iteration: 110, finishFrom: 60 });
      expect(resumed.map((d) => d.iteration)).toEqual([110, 120, 130]);
      expect(early.iterations).toBe(130);
      expect(early.stats).toMatchObject({ splatResumedFrom: 100, splatFinishedEarly: 1, splatContinueFrom: 130, splatIterations: 200 });
      // the model is persisted, the checkpoint at 130 kept (its run key with the model)
      const progB = await b.engineStore!.getValue<{ iteration: number; key: string }>('splat/progress');
      expect(progB?.iteration).toBe(130);
      expect(early.stats.splatContinueKey).toBe(progB!.key);
      expect((await b.engineStore!.getValue<EngineStateRecord>('engine/state'))?.hasSplat).toBe(true);
      expect((await b.splatModel())?.model.count).toBe(early.count);
      b.dispose();

      // a third engine: the stored model is continuable; it continues to the end
      const c = new PhotogrammetryEngine({ mode: 'auto', quality: 'fast' }, config());
      const sumC = await c.loadSession(id);
      expect(sumC.splat).toMatchObject({ count: early.count, iterations: 130 });
      expect(sumC.stats.splatResumableIteration).toBe(130);
      expect(sumC.stats.splatContinuesModel).toBe(1);
      const continued: SplatTrainingDetail[] = [];
      const info = await c.splat(undefined, (p) => {
        const d = training(p);
        if (d) continued.push(d);
      });
      console.log(`[splat-engine] continued: ${JSON.stringify(info)}`);
      expect(info.iterations).toBe(200);
      expect(info.stats.splatResumedFrom).toBe(130);
      expect(info.stats.splatFinishedEarly).toBeUndefined();
      // default geometry supervision: the persisted filtered depth maps of the reloaded session
      expect(Number(info.stats.splatDepthViews)).toBeGreaterThan(0);
      expect(continued[0].iteration).toBe(140);
      // the resumed trainer refines again (MCMC at 150)
      expect(continued[continued.length - 1].gaussians).toBeGreaterThan(continued[0].gaussians);
      expect(info.count).toBeGreaterThan(1000);
      expect(Number(info.stats.splatLoss)).toBeLessThan(0.25);
      expect(info.bytes).toBe(56 * info.count);
      // checkpoints cleared, model persisted
      expect(await c.engineStore!.getValue('splat/progress')).toBeNull();
      expect(await c.engineStore!.listBinaryKeys('splat/ckpt')).toEqual([]);
      expect((await c.engineStore!.getValue<EngineStateRecord>('engine/state'))?.hasSplat).toBe(true);
      const payload = await c.splatModel();
      expect(payload?.model.count).toBe(info.count);
      expect(payload?.antialiased).toBe(true);
      const ply = decodeSplatPLY(new Uint8Array(await (await c.exportModel('ply-splat')).arrayBuffer()));
      expect(ply.model.count).toBe(info.count);
      c.dispose();

      // a fourth engine sees the stored model
      const d = new PhotogrammetryEngine({ mode: 'auto', quality: 'fast' }, config());
      const sum2 = await d.loadSession(id);
      expect(sum2.splat).toMatchObject({ count: info.count, iterations: 200 });
      expect(sum2.stats.splatResumableIteration).toBeUndefined();
      expect(sum2.stats.splatContinuesModel).toBeUndefined();
      expect((await d.splatModel())?.model.count).toBe(info.count);
      d.dispose();
    } finally {
      const s = await SessionStore.open();
      await s.deleteSession(id).catch(() => undefined);
      s.close();
    }
  }, 600_000);

  it('previews; rolls back dropped work and survives a device loss (sparse init + sky shell)', async () => {
    let drops = 0;
    let current: SplatTrainer | null = null;
    const e = new PhotogrammetryEngine(
      { mode: 'object', quality: 'fast', persist: false, hfovPriorDeg: 60 },
      config({
        splatDebug: {
          onTrainer: (t) => {
            current = t;
            // the first trainer drops the command buffers of two iterations of its first chunk
            if (drops++ === 0) t.simulateDroppedWork({ iterations: 2 });
          },
        },
      }),
    );
    await captureAndRefine(e);
    let lost = false;
    const iters: number[] = [];
    const order: string[] = [];
    const previews: { d: SplatPreviewDetail; width: number; height: number; sum: number }[] = [];
    const info = await e.splat({ backgroundShell: true, preview: true }, (p) => {
      // the colour-state readbacks of the appearance normalisation are dropped twice
      if (p.stage === 'splat_finishing' && p.progress === 0) current!.simulateDroppedWork({ readbacks: 2 });
      const v = preview(p);
      if (v) previews.push({ d: v, width: p.image!.width, height: p.image!.height, sum: Array.from(p.image!.data).reduce((s, x) => s + x, 0) });
      const d = training(p);
      if (d || v) order.push(p.stage);
      if (d) iters.push(d.iteration);
      if (d && d.iteration >= 120 && !lost) {
        lost = true;
        // (destroyed in a microtask: before the next step's GPU work completes)
        void getGPU().then((ctx) => ctx?.device.destroy());
      }
    });
    console.log(`[splat-engine] drop + loss: ${JSON.stringify(info.stats)}`);
    expect(info.iterations).toBe(200);
    expect(info.stats.splatDroppedWork).toBe(1);
    expect(info.stats.splatDeviceLosses).toBe(1);
    // the dropped first chunk is trained again from the checkpoint at 0 (first report 10, not 20);
    // the device lost in the step after 120 is rebuilt from the checkpoint at 100 (next report 110)
    expect(iters).toEqual([...reports(10, 120), ...reports(110, 200)]);
    expect(info.stats.splatReadRetries).toBe(2);
    // the mean colour matrix of the training views (stale staging data: loss sums, stamps)
    expect(Math.abs(Number(info.stats.splatMeanColourGain) - 1)).toBeLessThan(0.5);
    expect(info.stats.splatInitDense).toBe(0);
    expect(Number(info.stats.splatShell)).toBeGreaterThan(0);
    expect(drops).toBe(2); // the trainer was rebuilt once
    expect((await e.splatModel())?.model.count).toBe(info.count);
    // previews: the initial Gaussians before the first training report, then between chunks
    // (≥ 2 s apart; a chunk takes longer here) at the training size (≤ 400 px: factor 1)
    console.log(`[splat-engine] previews: ${previews.map((p) => `${p.d.iteration}: ${p.d.psnr} dB`).join(', ')}`);
    expect(previews.length).toBeGreaterThanOrEqual(3);
    expect(order[0]).toBe('splat_preview');
    expect(previews[0].d.iteration).toBe(0);
    expect(info.stats.splatPreviews).toBe(previews.length);
    for (const p of previews) expect([p.width, p.height, p.d.viewId]).toEqual([info.stats.splatWidth, info.stats.splatHeight, previews[0].d.viewId]);
    expect(new Set(previews.map((p) => p.sum)).size).toBeGreaterThan(2);
    expect(previews[previews.length - 1].d.psnr).toBeGreaterThan(previews[0].d.psnr + 3);
    // the trained model from that view (not posted): what the previews converge to
    expect(Number(info.stats.splatPreviewFinalPsnr)).toBeGreaterThan(previews[previews.length - 1].d.psnr - 1);
    e.dispose();
  }, 600_000);

  it('a device loss in the first checkpoint, a later checkpoint or the finishing readbacks rebuilds and trains on (review)', async () => {
    const trainers: SplatTrainer[] = [];
    const deviceOf = (t: SplatTrainer): GPUDevice => (t as unknown as { device: GPUDevice }).device;
    let loseOnTrainer = false;
    const e = new PhotogrammetryEngine(
      { mode: 'object', quality: 'fast', persist: false, hfovPriorDeg: 60 },
      config({
        splatDebug: {
          onTrainer: (t) => {
            trainers.push(t);
            // lost right after the uploads: the iteration-0 checkpoint's readbacks fail
            if (loseOnTrainer) {
              loseOnTrainer = false;
              deviceOf(t).destroy();
            }
          },
        },
      }),
    );
    await captureAndRefine(e);
    // run 1: the first checkpoint (rebuilt from the initial Gaussians), then the appearance
    // normalisation (rebuilt from the checkpoint at iteration 150, trained again to 200)
    loseOnTrainer = true;
    let finishLost = false;
    const itersA: number[] = [];
    const a = await e.splat(undefined, (p) => {
      const d = training(p);
      if (d) itersA.push(d.iteration);
      if (p.stage === 'splat_finishing' && p.progress === 0 && !finishLost) {
        finishLost = true;
        deviceOf(trainers[trainers.length - 1]).destroy();
      }
    });
    console.log(`[splat-engine] losses at the first checkpoint + finishing: ${JSON.stringify(a.stats)}`);
    expect(a.iterations).toBe(200);
    expect(a.stats.splatDeviceLosses).toBe(2);
    expect(trainers.length).toBe(3);
    // nothing trained before the first loss; the finishing loss trains again from the checkpoint at 150
    expect(itersA).toEqual([...reports(10, 200), ...reports(160, 200)]);
    expect(Math.abs(Number(a.stats.splatMeanColourGain) - 1)).toBeLessThan(0.5);
    expect((await e.splatModel())?.model.count).toBe(a.count);
    // run 2: the checkpoint at iteration 150 (written right after its report): back to 100
    let ckptLost = false;
    const iters: number[] = [];
    const b = await e.splat(undefined, (p) => {
      const d = training(p);
      if (!d) return;
      iters.push(d.iteration);
      if (d.iteration === 150 && !ckptLost) {
        ckptLost = true;
        deviceOf(trainers[trainers.length - 1]).destroy();
      }
    });
    console.log(`[splat-engine] loss at a checkpoint: ${JSON.stringify(b.stats)}`);
    expect(b.iterations).toBe(200);
    expect(b.stats.splatDeviceLosses).toBe(1);
    expect(trainers.length).toBe(5);
    expect(iters.filter((i) => i === 150).length).toBe(2); // trained again from 100
    expect(iters).toEqual([...reports(10, 150), ...reports(110, 200)]);
    expect((await e.splatModel())?.model.count).toBe(b.count);
    e.dispose();
  }, 600_000);
});
