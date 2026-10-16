// createFeaturePipeline: backend selection and CPU fallback (headless Chromium + SwiftShader).
import { describe, expect, it } from 'vitest';
import type { FeatureSet } from '../core/types';
import { getGPU } from '../gpu/device';
import type { GPUContext } from '../gpu/device';
import { createFeaturePipeline } from './extractor';
import { rgbaToGray } from './image';
import { matchCPU, matchGuidedCPU } from './match';
import { extractSiftCPU } from './sift-cpu';
import { makeTexturedImage, makeWarpedPair, similarityAboutCenter } from './testdata';

const log = (...a: unknown[]): void => console.log('[extractor]', ...a);

function sameFeatures(a: FeatureSet, b: FeatureSet): boolean {
  return (
    a.count === b.count &&
    a.width === b.width &&
    a.height === b.height &&
    a.keypoints.every((v, i) => v === b.keypoints[i]) &&
    a.descriptors.every((v, i) => v === b.descriptors[i])
  );
}

/** Fraction of a's keypoints with a keypoint of b within 1 px. */
function posAgreement(a: FeatureSet, b: FeatureSet): number {
  let n = 0;
  for (let i = 0; i < a.count; i++) {
    for (let j = 0; j < b.count; j++) {
      if (Math.hypot(a.keypoints[4 * i] - b.keypoints[4 * j], a.keypoints[4 * i + 1] - b.keypoints[4 * j + 1]) <= 1) {
        n++;
        break;
      }
    }
  }
  return a.count ? n / a.count : 1;
}

/** A GPUContext on a private device (so destroying it does not affect other tests). */
async function privateContext(): Promise<GPUContext> {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('no adapter');
  const device = await adapter.requestDevice();
  return {
    adapter,
    device,
    features: new Set(device.features as unknown as Iterable<string>),
    limits: device.limits,
    info: { vendor: 'test', architecture: 'swiftshader', description: '' },
    isSoftware: true,
  };
}

describe('createFeaturePipeline', () => {
  const W = 320;
  const H = 240;
  const opts = { maxFeatures: 800 };

  it('uses the GPU when available; results agree with the CPU reference', async () => {
    expect(await getGPU()).not.toBeNull();
    const fp = await createFeaturePipeline({ sift: opts });
    try {
      expect(fp.gpu).toBe(true);
      expect(fp.fallbackReason).toBeNull();
      const p = makeWarpedPair(W, H, similarityAboutCenter(W, H, 15, 0.9), 4);
      const a = await fp.extract(p.a, 640, 480);
      const b = await fp.extract(p.b, 640, 480);
      const ca = extractSiftCPU(p.a, opts, 640, 480);
      log(`GPU ${a.count} vs CPU ${ca.count} features, position agreement ${(100 * posAgreement(ca, a)).toFixed(1)}%`);
      expect([a.width, a.height]).toEqual([640, 480]);
      expect(posAgreement(ca, a)).toBeGreaterThan(0.97);
      expect(Array.from(await fp.match(a, b))).toEqual(Array.from(matchCPU(a, b)));
      const guide = { kind: 'positions' as const, predicted: new Float32Array(2 * a.count).fill(NaN), radiusPx: 4 };
      expect(Array.from(await fp.matchGuided(a, b, guide))).toEqual(Array.from(matchGuidedCPU(a, b, guide)));
      // bitmap path
      const rgba = new ImageData(W, H);
      for (let i = 0; i < W * H; i++) {
        const v = Math.round(255 * (p.a.data[i] as number));
        rgba.data.set([v, v, v, 255], 4 * i);
      }
      const bmp = await createImageBitmap(rgba);
      const fb = await fp.extractFromBitmap(bmp);
      const cb = extractSiftCPU(rgbaToGray(rgba), opts);
      expect([fb.width, fb.height]).toEqual([W, H]);
      expect(posAgreement(cb, fb)).toBeGreaterThan(0.97);
      bmp.close();
    } finally {
      fp.destroy();
    }
  });

  it('preferGPU: false and context: null use the CPU (identical to the CPU functions)', async () => {
    const img = makeTexturedImage(W, H, 5);
    for (const fp of [await createFeaturePipeline({ preferGPU: false, sift: opts }), await createFeaturePipeline({ context: null, sift: opts })]) {
      try {
        expect(fp.gpu).toBe(false);
        expect(fp.fallbackReason).toMatch(/CPU|unavailable/);
        const a = await fp.extract(img, W, H);
        expect(sameFeatures(a, extractSiftCPU(img, opts, W, H))).toBe(true);
        const b = await fp.extract(makeTexturedImage(W, H, 6), W, H);
        expect(Array.from(await fp.match(a, b))).toEqual(Array.from(matchCPU(a, b)));
        const rgba = new ImageData(W, H);
        for (let i = 0; i < W * H; i++) {
          const v = Math.round(255 * (img.data[i] as number));
          rgba.data.set([v, v, v, 255], 4 * i);
        }
        const bmp = await createImageBitmap(rgba);
        const fb = await fp.extractFromBitmap(bmp);
        expect(sameFeatures(fb, extractSiftCPU(rgbaToGray(rgba), opts))).toBe(true);
        bmp.close();
      } finally {
        fp.destroy();
      }
    }
  });

  it('falls back to the CPU on device loss (logged once, callback once) and keeps working', async () => {
    const ctx = await privateContext();
    const reasons: string[] = [];
    const fp = await createFeaturePipeline({ context: ctx, sift: opts, onFallback: (r) => reasons.push(r) });
    const img = makeTexturedImage(W, H, 7);
    expect(fp.gpu).toBe(true);
    const g = await fp.extract(img, W, H);
    expect(g.count).toBeGreaterThan(100);
    ctx.device.destroy();
    await ctx.device.lost;
    await new Promise((r) => setTimeout(r, 0));
    expect(fp.gpu).toBe(false);
    const c = await fp.extract(img, W, H);
    expect(sameFeatures(c, extractSiftCPU(img, opts, W, H))).toBe(true);
    expect(Array.from(await fp.match(c, c))).toEqual(Array.from(matchCPU(c, c)));
    expect(reasons.length).toBe(1);
    log(`fallback reason: ${reasons[0]}`);
    fp.destroy();
  });

  it('falls back when a GPU call fails mid-operation (device destroyed during extraction)', async () => {
    const ctx = await privateContext();
    const fp = await createFeaturePipeline({ context: ctx, sift: opts });
    const img = makeTexturedImage(W, H, 8);
    expect(fp.gpu).toBe(true);
    const pending = fp.extract(img, W, H);
    ctx.device.destroy();
    const r = await pending;
    expect(fp.gpu).toBe(false);
    expect(sameFeatures(r, extractSiftCPU(img, opts, W, H))).toBe(true);
    fp.destroy();
  });

  it('GPU work dropped by the OS: that call runs on the CPU, the GPU stays enabled', async () => {
    const fp = await createFeaturePipeline({ sift: opts, siftGPU: { testDropSubmissions: () => [0] }, selfTest: false });
    try {
      const img = makeTexturedImage(W, H, 9);
      const r = await fp.extract(img, W, H);
      expect(sameFeatures(r, extractSiftCPU(img, opts, W, H))).toBe(true);
      expect(fp.gpu).toBe(true);
    } finally {
      fp.destroy();
    }
  });

  it('input errors propagate without disabling the GPU', async () => {
    const fp = await createFeaturePipeline({ sift: opts });
    try {
      await expect(fp.extract({ width: 50, height: 50, data: new Float32Array(3) }, 50, 50)).rejects.toThrow(/shorter/);
      expect(fp.gpu).toBe(true);
      const a = makeTexturedImage(64, 48, 1);
      await expect(fp.matchGuided(await fp.extract(a, 64, 48), await fp.extract(a, 64, 48), { kind: 'positions', predicted: new Float32Array(1), radiusPx: 3 })).rejects.toThrow(
        /predicted/,
      );
      expect(fp.gpu).toBe(true);
    } finally {
      fp.destroy();
    }
  });
});
