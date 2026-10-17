/**
 * Dense resume across a "jetsam kill" with the real IndexedDB store (Chromium): a densify is
 * interrupted in its geometric pass, a NEW engine opens the session (as after a page reload),
 * `loadSession` reports the resumable pass and the next densify skips the stored depth maps.
 * CPU feature pipeline + CPU PatchMatch (preferGPU: false) on the small synthetic orbit of
 * src/engine/testutil.ts, so the test does not depend on SwiftShader speed.
 */
import { describe, expect, it } from 'vitest';
import type { ProgressEvent } from '../core/types';
import { SessionStore } from '../io/storage';
import type { EngineOptions } from '../workers/protocol';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import type { EngineStateRecord } from './persistence';
import { frameInputs, makeCapture } from './testutil';
import type { EngineConfig } from './types';

describe('dense resume after a kill (IndexedDB)', () => {
  it('a new engine resumes an interrupted densify from the stored depth maps', async () => {
    const cap = makeCapture({ frames: 44, arcDeg: 55 });
    const id = `dense-resume-${Date.now()}`;
    const opts: EngineOptions = { mode: 'object', quality: 'fast', persist: true, preferGPU: false, sessionId: id, hfovPriorDeg: 60 };
    const pipeline = createCpuFeaturePipeline({ maxFeatures: 2048, maxImageSize: 1024 });
    const cfg: EngineConfig = { pipeline, mapping: 'inline' };
    const dense = { maxImageSize: 96, maxViews: 8 };
    try {
      const a = new PhotogrammetryEngine(opts, cfg);
      await a.init();
      for (const f of frameInputs(cap)) await a.processFrame(f);
      await a.finishCapture();
      await a.refine();
      const st0 = (await a.engineStore!.getValue<EngineStateRecord>('engine/state'))!;
      // "killed" in the geometric pass: cancel, then drop the engine without any cleanup
      const cancel = { cancelled: false };
      const onProgress = (p: ProgressEvent): void => {
        if (p.stage === 'depth_maps' && p.progress > 0.6) cancel.cancelled = true;
      };
      await expect(a.densify(dense, onProgress, cancel)).rejects.toMatchObject({ name: 'CancelledError' });
      const stored = (await a.engineStore!.listBinaryKeys('pm1/')).length;
      expect(stored).toBeGreaterThan(0);
      a.dispose();

      // reload: a new engine opens the session
      const b = new PhotogrammetryEngine({ mode: 'auto', quality: 'fast', preferGPU: false }, cfg);
      const sum = await b.loadSession(id);
      expect(sum.status).toBe('sparse');
      expect(sum.stats.denseResumablePass).toBe(1);
      const pc = await b.densify(dense);
      expect(pc.count).toBeGreaterThan(500);
      expect(b.sparseResult!.stats.denseResumedPass).toBe(1);
      expect(b.sparseResult!.stats.denseResumedMaps).toBe(stored);
      expect(await b.engineStore!.listBinaryKeys('pm')).toEqual([]);
      expect(await b.engineStore!.getValue('dense/progress')).toBeNull();
      const st1 = (await b.engineStore!.getValue<EngineStateRecord>('engine/state'))!;
      expect(st1.hasDense).toBe(true);
      expect(st1.thetaSession).toBe(st0.thetaSession);
      expect(st1.cameraThetaVerified).toBe(st0.cameraThetaVerified);
      expect(st1.timeOffsetMs).toBe(st0.timeOffsetMs);
      const mesh = await b.mesh();
      expect(mesh.triangleCount).toBeGreaterThan(100);
      b.dispose();
    } finally {
      const s = await SessionStore.open();
      await s.deleteSession(id).catch(() => undefined);
      s.close();
    }
  }, 600_000);
});
