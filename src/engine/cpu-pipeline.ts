/**
 * CPU-only FeaturePipeline (extractSiftCPU + matchCPU / matchGuidedCPU) that does NOT import
 * the WebGPU modules: `sift-gpu.ts` / `match-gpu.ts` read `GPUBufferUsage` at module load, which
 * throws in realms without WebGPU (older iOS, Lockdown mode, Node). The engine therefore loads
 * `features/extractor` dynamically and only when WebGPU exists, and uses this pipeline otherwise.
 */
import type { FeatureSet, GrayImage, MatchPairs, RGBAImage } from '../core/types';
import type { FeaturePipeline } from '../features/extractor';
import { rgbaToGray } from '../features/image';
import { matchCPU, matchGuidedCPU, type MatchGuide, type MatchOptions } from '../features/match';
import type { SiftOptions } from '../features/sift-common';
import { extractSiftCPU } from '../features/sift-cpu';
import type { SiftBitmapSource } from '../features/sift-gpu';

function bitmapToGray(bitmap: SiftBitmapSource): GrayImage {
  if (typeof ImageData !== 'undefined' && bitmap instanceof ImageData) return rgbaToGray(bitmap as unknown as RGBAImage);
  if (typeof OffscreenCanvas === 'undefined') throw new Error('extractFromBitmap: OffscreenCanvas unavailable for the CPU path');
  const c = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = c.getContext('2d', { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D | null;
  if (!ctx) throw new Error('extractFromBitmap: 2D context unavailable');
  ctx.drawImage(bitmap as CanvasImageSource, 0, 0);
  const d = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  return rgbaToGray({ width: d.width, height: d.height, data: d.data });
}

export function createCpuFeaturePipeline(sift: Partial<SiftOptions> = {}, reason = 'CPU requested'): FeaturePipeline {
  let destroyed = false;
  const check = (): void => {
    if (destroyed) throw new Error('FeaturePipeline: used after destroy()');
  };
  return {
    gpu: false,
    fallbackReason: reason,
    async extract(img: GrayImage, fullWidth: number, fullHeight: number): Promise<FeatureSet> {
      check();
      return extractSiftCPU(img, sift, fullWidth, fullHeight);
    },
    async extractFromBitmap(bitmap: SiftBitmapSource, fullWidth?: number, fullHeight?: number): Promise<FeatureSet> {
      check();
      return extractSiftCPU(bitmapToGray(bitmap), sift, fullWidth, fullHeight);
    },
    async match(a: FeatureSet, b: FeatureSet, opts?: MatchOptions): Promise<MatchPairs> {
      check();
      return matchCPU(a, b, opts);
    },
    async matchGuided(a: FeatureSet, b: FeatureSet, guide: MatchGuide, opts?: MatchOptions): Promise<MatchPairs> {
      check();
      return matchGuidedCPU(a, b, guide, opts);
    },
    destroy(): void {
      destroyed = true;
    },
  };
}
