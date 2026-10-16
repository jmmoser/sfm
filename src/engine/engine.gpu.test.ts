/**
 * End to end in Chromium (SwiftShader WebGPU): the real engine worker
 * (src/workers/engine.worker.ts) driven through EngineClient on ~20 frames of the synthetic
 * `object` dataset (e2e/data/object, served by Vite at /e2e/data/object/), with GPU SIFT /
 * matching and GPU PatchMatch; plus IndexedDB persistence / resume / loadSession of the inline
 * engine. SwiftShader is a slow CPU emulation, so working sizes are tiny.
 *
 * Generate the dataset first if missing: `node e2e/generate-synth.mjs object`.
 */
import { describe, expect, it } from 'vitest';
import type { ImuSample } from '../core/types';
import { EngineClient } from '../app/engine-client';
import type { SynthFrame, SynthMeta } from '../harness/synth/dataset';
import { decodePLY } from '../io/ply';
import { SessionStore } from '../io/storage';
import type { EngineOptions, FrameInput, LiveUpdate } from '../workers/protocol';
import type { ProgressEvent } from '../core/types';
import { PhotogrammetryEngine } from './engine';
import type { EngineConfig, SparseCameraEx } from './types';

const BASE = '/e2e/data/object';

interface Dataset {
  meta: SynthMeta;
  frames: SynthFrame[];
  imu: ImuSample[];
  frameUrl(i: number): string;
}

/** meta / frames / imu of the dataset (loadSynthDataset also wants geo.json, which the object scene lacks — Vite answers it with index.html). */
async function dataset(): Promise<Dataset | null> {
  try {
    const get = async <T>(f: string): Promise<T> => {
      const r = await fetch(`${BASE}/${f}`);
      if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
      return (await r.json()) as T;
    };
    const [meta, fr, imu] = await Promise.all([get<SynthMeta>('meta.json'), get<SynthFrame[]>('frames.json'), get<ImuSample[]>('imu.json')]);
    return { meta, frames: fr, imu, frameUrl: (i) => `${BASE}/${fr[i].file}` };
  } catch (e) {
    console.warn(`${BASE}: ${(e as Error)?.message ?? e}`);
    return null;
  }
}

/** FrameInputs for the dataset frames `indices` (IMU samples since the previous frame, as the pump drains them). */
async function* frames(ds: Dataset, indices: number[], asBitmap = true): AsyncGenerator<FrameInput> {
  let k = 0;
  let lastT = -Infinity;
  for (const i of indices) {
    const fr = ds.frames[i];
    const blob = await (await fetch(ds.frameUrl(i))).blob();
    const bitmap = await createImageBitmap(blob);
    const until = fr.t + 5;
    const imu: ImuSample[] = [];
    while (k < ds.imu.length && ds.imu[k].t <= until) {
      if (ds.imu[k].t > lastT) imu.push(ds.imu[k]);
      k++;
    }
    lastT = until;
    let src: ImageBitmap | { width: number; height: number; data: Uint8ClampedArray } = bitmap;
    if (!asBitmap) {
      const c = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = c.getContext('2d')!;
      ctx.drawImage(bitmap, 0, 0);
      const d = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
      bitmap.close();
      src = { width: d.width, height: d.height, data: d.data };
    }
    yield { bitmap: src, t: fr.t, screenAngle: ds.meta.screenAngle, imu, index: fr.index };
  }
}

const TINY: EngineConfig = {
  sift: { maxImageSize: 400, maxFeatures: 1024, upsampleBelow: 0 },
  livePairs: { sequential: 3, retrieval: 0, spatial: 1 },
  refinePairs: { sequential: 4, retrieval: 2, spatial: 2 },
  dense: { iterations: 2, geometricPasses: 0, numSources: 4, maxViews: 6 },
  memoryBudgetMB: 600,
};

describe('engine worker end to end (e2e/data/object)', () => {
  it('captures 20 frames, refines, densifies (GPU) and exports through EngineClient', async (ctx) => {
    const ds = await dataset();
    if (!ds) {
      console.warn(`dataset ${BASE} missing — run: node e2e/generate-synth.mjs object`);
      ctx.skip();
      return;
    }
    const client = new EngineClient({ forwardVisibility: false, frameTimeoutMs: 180_000, initTimeoutMs: 300_000 });
    const live: LiveUpdate[] = [];
    const logs: string[] = [];
    client.onLive((u) => live.push(u));
    client.onLog((l) => logs.push(`${l.level}: ${l.message}`));
    try {
      const opts: EngineOptions & { engineConfig: EngineConfig } = {
        mode: 'object',
        quality: 'fast',
        persist: false,
        preferGPU: true,
        hfovPriorDeg: 65,
        engineConfig: TINY,
      };
      const init = await client.init(opts);
      expect(init.gpu).toBe(true);
      const indices = Array.from({ length: 20 }, (_, i) => 2 * i);
      const updates: LiveUpdate[] = [];
      for await (const f of frames(ds, indices)) updates.push(await client.processFrame(f));
      expect(updates.map((u) => u.frameIndex)).toEqual(indices);
      const kfs = updates.filter((u) => u.decision === 'keyframe').length;
      expect(kfs).toBeGreaterThanOrEqual(4);
      await client.finishCapture();
      // background mapping delivered model updates as live events
      expect(live.some((u) => !!u.sparse && !!u.cameras && u.cameras.length >= 2)).toBe(true);
      const stages: string[] = [];
      const sparse = await client.refine((p: ProgressEvent) => {
        if (!stages.includes(p.stage)) stages.push(p.stage);
      });
      expect(stages).toEqual(expect.arrayContaining(['matching', 'reconstruction', 'alignment']));
      const cams = sparse.cameras as SparseCameraEx[];
      expect(cams.length).toBeGreaterThanOrEqual(Math.min(kfs, 4));
      for (const c of cams) {
        expect(indices).toContain(c.frameIndex);
        expect(c.t).toBe(ds.frames[c.frameIndex!].t);
      }
      const f = cams[0].intrinsics.f;
      const fTrue = ds.meta.intrinsics.f;
      expect(Math.abs(f / fTrue - 1)).toBeLessThan(0.1);
      expect(sparse.points.count).toBeGreaterThan(100);
      expect(sparse.metric.gravityAligned).toBe(true);
      const dense = await client.densify({ maxImageSize: 128, maxViews: 6 });
      expect(dense.count).toBeGreaterThan(100);
      const ply = decodePLY(await (await client.exportModel('ply-points')).arrayBuffer());
      expect(ply.points!.count).toBe(dense.count);
      const zip = await client.exportModel('colmap-zip');
      expect(zip.size).toBeGreaterThan(10_000);
    } finally {
      console.info(logs.filter((l) => !l.startsWith('debug')).join('\n'));
      client.dispose();
    }
  }, 600_000);
});

describe('engine persistence in IndexedDB (inline, CPU)', () => {
  it('resumes a capture and loads the session in a new engine', async (ctx) => {
    const ds = await dataset();
    if (!ds) {
      ctx.skip();
      return;
    }
    const id = `engine-test-${Date.now()}`;
    const opts: EngineOptions = { mode: 'object', quality: 'fast', persist: true, preferGPU: false, sessionId: id, hfovPriorDeg: 65 };
    const cfg: EngineConfig = { ...TINY, mapping: 'inline' };
    const a = new PhotogrammetryEngine(opts, cfg);
    await a.init();
    const idx = Array.from({ length: 16 }, (_, i) => 2 * i);
    for await (const f of frames(ds, idx.slice(0, 10), false)) await a.processFrame(f);
    await a.flush();
    const kfA = a.liveSession!.state.order.slice();
    expect(kfA.length).toBeGreaterThanOrEqual(2);
    a.dispose();
    // respawn: resume from IndexedDB, keyframes are JPEGs in the store
    const b = new PhotogrammetryEngine({ ...opts, resume: true }, cfg);
    await b.init();
    expect(b.liveSession!.state.order).toEqual(kfA);
    const img = await b.engineStore!.getFrameImage(kfA[0]);
    expect(img instanceof Blob).toBe(true);
    for await (const f of frames(ds, idx.slice(10), false)) await b.processFrame(f);
    await b.finishCapture();
    b.dispose();
    const c = new PhotogrammetryEngine({ mode: 'auto', quality: 'balanced', preferGPU: false }, cfg);
    const sum = await c.loadSession(id);
    expect(sum.id).toBe(id);
    expect(sum.mode).toBe('object');
    expect(sum.numFrames).toBeGreaterThanOrEqual(kfA.length);
    expect(sum.status).toBe('captured');
    c.dispose();
    const s = await SessionStore.open();
    await s.deleteSession(id);
    s.close();
  }, 600_000);
});
