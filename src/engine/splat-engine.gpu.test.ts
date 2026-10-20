/**
 * The engine's splat stage on a tiny synthetic orbit (src/engine/testutil.ts, CPU features, GPU
 * PatchMatch + splats; run on the host GPU: E2E_GPU=host — SwiftShader compiles the splat
 * pipelines for minutes):
 *  - persistent session (real IndexedDB): densify, splat cancelled mid-training, the engine
 *    dropped without cleanup ("killed"), a NEW engine reports the resumable iteration, resumes from
 *    the checkpoint, persists the model, clears the checkpoints and exports 'ply-splat';
 *  - in-memory session without a dense cloud: dropped GPU work rolls back to the last checkpoint,
 *    a destroyed device is replaced and training continues from the checkpoint.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { ProgressEvent } from '../core/types';
import { getGPU } from '../gpu/device';
import { SessionStore } from '../io/storage';
import { decodeSplatPLY } from '../splat/ply';
import type { EngineOptions, SplatTrainingDetail } from '../workers/protocol';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import type { EngineStateRecord } from './persistence';
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
      const a = new PhotogrammetryEngine(opts, config());
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
        if (d.iteration >= 150) cancel.cancelled = true;
      };
      await expect(a.splat(undefined, onProgress, cancel)).rejects.toMatchObject({ name: 'CancelledError' });
      expect([...stages]).toEqual(['splat_prepare', 'splat_training']);
      expect(seen[0].iterations).toBe(300);
      expect(seen.every((d) => Number.isFinite(d.loss) && d.itPerSec > 0 && d.gaussians > 0)).toBe(true);
      const prog = await a.engineStore!.getValue<{ iteration: number; slot: number }>('splat/progress');
      expect(prog?.iteration).toBe(100);
      expect((await a.engineStore!.listBinaryKeys(`splat/ckpt${prog!.slot}/`)).length).toBe(4);
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
    const e = new PhotogrammetryEngine(
      { mode: 'object', quality: 'fast', persist: false, hfovPriorDeg: 60 },
      config({
        splatDebug: {
          onTrainer: (t) => {
            // the first trainer drops the command buffers of two iterations of its first chunk
            if (drops++ === 0) t.simulateDroppedWork({ iterations: 2 });
          },
        },
      }),
    );
    await captureAndRefine(e);
    let lost = false;
    const info = await e.splat({ backgroundShell: true }, (p) => {
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
    expect(info.stats.splatInitDense).toBe(0);
    expect(Number(info.stats.splatShell)).toBeGreaterThan(0);
    expect(drops).toBe(2); // the trainer was rebuilt once
    expect((await e.splatModel())?.model.count).toBe(info.count);
    e.dispose();
  }, 600_000);
});
