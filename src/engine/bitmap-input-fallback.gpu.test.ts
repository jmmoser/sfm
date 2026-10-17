/**
 * GPU bitmap input that comes out blank (copyExternalImageToTexture yielding an empty texture for
 * camera frames on some WebKit builds): the live session must notice that gray input finds the
 * features and switch to it, instead of reporting "not enough detail" for the whole capture.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { FeatureSet } from '../core/types';
import type { FeaturePipeline } from '../features/extractor';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import { MemoryEngineStore } from './persistence';
import { frameInputs, makeCapture, type SynthCapture } from './testutil';

let cap: SynthCapture;
beforeAll(() => {
  cap = makeCapture({ frames: 24 });
}, 120_000);

/** A "GPU" pipeline whose bitmap path finds nothing (blank upload); the gray path works. */
function blankBitmapPipeline(): FeaturePipeline & { bitmapCalls: number } {
  const cpu = createCpuFeaturePipeline({ maxFeatures: 2048, maxImageSize: 1024 });
  const p = Object.create(cpu) as FeaturePipeline & { bitmapCalls: number };
  Object.defineProperty(p, 'gpu', { value: true });
  p.bitmapCalls = 0;
  p.extractFromBitmap = async (_b, w = 0, h = 0): Promise<FeatureSet> => {
    p.bitmapCalls++;
    return { count: 0, keypoints: new Float32Array(0), descriptors: new Uint8Array(0), scores: new Float32Array(0), width: w, height: h };
  };
  return p;
}

describe('blank GPU bitmap input', () => {
  it('switches to gray input and makes keyframes', async () => {
    const pipeline = blankBitmapPipeline();
    const engine = new PhotogrammetryEngine(
      { mode: 'object', quality: 'fast', persist: false, sessionId: 'blank-bitmap', hfovPriorDeg: 60 },
      { pipeline, mapping: 'inline', store: new MemoryEngineStore('blank-bitmap') },
    );
    await engine.init();
    const live = () => engine.liveSession!;
    for (const f of frameInputs(cap)) {
      const rgba = f.bitmap as { width: number; height: number; data: Uint8ClampedArray };
      const bitmap = await createImageBitmap(new ImageData(new Uint8ClampedArray(rgba.data), rgba.width, rgba.height));
      const u = await engine.processFrame({ ...f, bitmap });
      expect(u.guidance).not.toBe('more_texture');
    }
    // one blank bitmap extraction, then gray input for the rest of the session
    expect(pipeline.bitmapCalls).toBe(1);
    expect(live().state.order.length).toBeGreaterThan(2);
    engine.dispose();
  }, 300_000);
});
