/**
 * The engine's splat stage on a tiny synthetic orbit (src/engine/testutil.ts, CPU features, GPU
 * PatchMatch + splats; run on the host GPU: E2E_GPU=host — SwiftShader compiles the splat
 * pipelines for minutes):
 *  - persistent session (real IndexedDB): densify, splat cancelled mid-training (the readbacks of
 *    its checkpoint dropped first: read again, never stored stale), the engine dropped without
 *    cleanup ("killed"), a NEW engine reports the resumable iteration, resumes from the
 *    checkpoint, persists the model, clears the checkpoints and exports 'ply-splat';
 *  - in-memory session without a dense cloud: dropped GPU work rolls back to the last checkpoint,
 *    a destroyed device is replaced and training continues from the checkpoint; dropped readbacks
 *    of the appearance normalisation are read again.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { ProgressEvent } from '../core/types';
import { getGPU } from '../gpu/device';
import { SessionStore } from '../io/storage';
import { decodeSplatPLY } from '../splat/ply';
import type { SplatTrainer } from '../splat/trainer';
import { COLOUR_FLOATS } from '../splat/wgsl/loss';
import type { EngineOptions, SplatTrainingDetail } from '../workers/protocol';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import { unpackSplatParams, type EngineStateRecord } from './persistence';
import { frameInputs, makeCapture, type SynthCapture } from './testutil';
import type { EngineConfig } from './types';

let cap: SynthCapture;
beforeAll(() => {
  cap = makeCapture({ frames: 44, arcDeg: 55 });
});

const SPLAT = { iterations: 300, maxGaussians: 30_000, maxImageSize: 320, checkpointEvery: 100, chunk: 20 };

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
        if (d.iteration >= 150) cancel.cancelled = true;
      };
      await expect(a.splat(undefined, onProgress, cancel)).rejects.toMatchObject({ name: 'CancelledError' });
      expect([...stages]).toEqual(['splat_prepare', 'splat_training']);
      expect(seen[0].iterations).toBe(300);
      expect(seen.every((d) => Number.isFinite(d.loss) && d.itPerSec > 0 && d.gaussians > 0)).toBe(true);
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

      // reload: a new engine opens the session and resumes
      const b = new PhotogrammetryEngine({ mode: 'auto', quality: 'fast' }, config());
      const sum = await b.loadSession(id);
      expect(sum.stats.splatResumableIteration).toBe(100);
      expect(sum.splat ?? null).toBeNull();
      const resumed: SplatTrainingDetail[] = [];
      const info = await b.splat(undefined, (p) => {
        const d = training(p);
        if (d) resumed.push(d);
      });
      console.log(`[splat-engine] resumed: ${JSON.stringify(info)}`);
      expect(info.iterations).toBe(300);
      expect(info.stats.splatResumedFrom).toBe(100);
      // default geometry supervision: the persisted filtered depth maps of the reloaded session
      expect(Number(info.stats.splatDepthViews)).toBeGreaterThan(0);
      expect(resumed[0].iteration).toBe(120);
      expect(info.count).toBeGreaterThan(1000);
      expect(Number(info.stats.splatLoss)).toBeLessThan(0.25);
      expect(info.bytes).toBe(56 * info.count);
      // checkpoints cleared, model persisted
      expect(await b.engineStore!.getValue('splat/progress')).toBeNull();
      expect(await b.engineStore!.listBinaryKeys('splat/ckpt')).toEqual([]);
      expect((await b.engineStore!.getValue<EngineStateRecord>('engine/state'))?.hasSplat).toBe(true);
      const payload = await b.splatModel();
      expect(payload?.model.count).toBe(info.count);
      expect(payload?.antialiased).toBe(true);
      const ply = decodeSplatPLY(new Uint8Array(await (await b.exportModel('ply-splat')).arrayBuffer()));
      expect(ply.model.count).toBe(info.count);
      b.dispose();

      // a third engine sees the stored model
      const c = new PhotogrammetryEngine({ mode: 'auto', quality: 'fast' }, config());
      const sum2 = await c.loadSession(id);
      expect(sum2.splat?.count).toBe(info.count);
      expect(sum2.stats.splatResumableIteration).toBeUndefined();
      expect((await c.splatModel())?.model.count).toBe(info.count);
      c.dispose();
    } finally {
      const s = await SessionStore.open();
      await s.deleteSession(id).catch(() => undefined);
      s.close();
    }
  }, 600_000);

  it('rolls back dropped work and survives a device loss (sparse init + sky shell)', async () => {
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
    const info = await e.splat({ backgroundShell: true }, (p) => {
      // the colour-state readbacks of the appearance normalisation are dropped twice
      if (p.stage === 'splat_finishing' && p.progress === 0) current!.simulateDroppedWork({ readbacks: 2 });
      const d = training(p);
      if (d && d.iteration >= 140 && !lost) {
        lost = true;
        void getGPU().then((ctx) => ctx?.device.destroy());
      }
    });
    console.log(`[splat-engine] drop + loss: ${JSON.stringify(info.stats)}`);
    expect(info.iterations).toBe(300);
    expect(info.stats.splatDroppedWork).toBe(1);
    expect(info.stats.splatDeviceLosses).toBe(1);
    expect(info.stats.splatReadRetries).toBe(2);
    // the mean colour matrix of the training views (stale staging data: loss sums, stamps)
    expect(Math.abs(Number(info.stats.splatMeanColourGain) - 1)).toBeLessThan(0.5);
    expect(info.stats.splatInitDense).toBe(0);
    expect(Number(info.stats.splatShell)).toBeGreaterThan(0);
    expect(drops).toBe(2); // the trainer was rebuilt once
    expect((await e.splatModel())?.model.count).toBe(info.count);
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
    // normalisation (rebuilt from the checkpoint at iteration 200, trained again to 300)
    loseOnTrainer = true;
    let finishLost = false;
    const a = await e.splat(undefined, (p) => {
      if (p.stage === 'splat_finishing' && p.progress === 0 && !finishLost) {
        finishLost = true;
        deviceOf(trainers[trainers.length - 1]).destroy();
      }
    });
    console.log(`[splat-engine] losses at the first checkpoint + finishing: ${JSON.stringify(a.stats)}`);
    expect(a.iterations).toBe(300);
    expect(a.stats.splatDeviceLosses).toBe(2);
    expect(trainers.length).toBe(3);
    expect(Math.abs(Number(a.stats.splatMeanColourGain) - 1)).toBeLessThan(0.5);
    expect((await e.splatModel())?.model.count).toBe(a.count);
    // run 2: the checkpoint at iteration 200 (written right after its report): back to 100
    let ckptLost = false;
    const iters: number[] = [];
    const b = await e.splat(undefined, (p) => {
      const d = training(p);
      if (!d) return;
      iters.push(d.iteration);
      if (d.iteration === 200 && !ckptLost) {
        ckptLost = true;
        deviceOf(trainers[trainers.length - 1]).destroy();
      }
    });
    console.log(`[splat-engine] loss at a checkpoint: ${JSON.stringify(b.stats)}`);
    expect(b.iterations).toBe(300);
    expect(b.stats.splatDeviceLosses).toBe(1);
    expect(trainers.length).toBe(5);
    expect(iters.filter((i) => i === 200).length).toBe(2); // trained again from 100
    expect((await e.splatModel())?.model.count).toBe(b.count);
    e.dispose();
  }, 600_000);
});
