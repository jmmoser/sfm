/**
 * Snapshot previews of a model while it trains (the engine's 'splat_preview' events,
 * src/engine/splat.ts). Pure CPU helpers, unit-tested in preview.test.ts:
 *
 *   previewViewIndex  the training view nearest the middle registered image (the viewer's default
 *                     `viewFromCamera` photo), so the preview shows what the finished model opens on;
 *   previewFactor     the integer downscale that brings the training size to ≤ the preview size: the
 *                     camera is `downscaleSplatCamera(cam, factor)`, so the photo box-downscaled by
 *                     the same factor (`boxDownscale`, §2 resampling rule) is the render's ground truth;
 *   rgbPsnr           PSNR of the RGB channels of two RGBA8 images.
 *
 * The engine turns the trainer's f32 render into RGBA8 with `splatImageToRGBA8` (renderer.ts). The
 * snapshot shows raw trainer colours: no per-view colour correction and no appearance
 * normalisation (the finished model's colours can shift slightly).
 */
import type { RGBAImage } from '../core/types';

/** PSNR reported for identical images (dB; keeps the value finite for JSON consumers). */
const PSNR_MAX = 99;

/**
 * Index into `viewIds` of the view nearest (by position in `imageIds`, the registration order the
 * viewer lists its cameras in) the middle image `imageIds[⌊n / 2⌋]`; ties take the earlier view.
 * Views missing from `imageIds` count as farthest.
 */
export function previewViewIndex(viewIds: readonly number[], imageIds: readonly number[]): number {
  if (!viewIds.length) throw new Error('previewViewIndex: no views');
  const pos = new Map(imageIds.map((id, i) => [id, i]));
  const mid = Math.floor(imageIds.length / 2);
  let best = 0;
  let bestD = Infinity;
  viewIds.forEach((id, i) => {
    const p = pos.get(id);
    const d = p === undefined ? Infinity : Math.abs(p - mid);
    if (d < bestD) {
      best = i;
      bestD = d;
    }
  });
  return best;
}

/** Smallest integer factor that brings `width × height` to ≤ `maxSize` px on the long side (≥ 1). */
export function previewFactor(width: number, height: number, maxSize: number): number {
  return Math.max(1, Math.ceil(Math.max(width, height) / Math.max(1, maxSize)));
}

/**
 * `img` downscaled by the integer `factor`: floor size, pixel (x, y) = the rounded mean of the
 * factor × factor block starting at (factor·x, factor·y) — the pixels of `downscaleSplatCamera`.
 */
export function boxDownscale(img: RGBAImage, factor: number): RGBAImage {
  const f = Math.max(1, Math.round(factor));
  const w = Math.max(1, Math.floor(img.width / f));
  const h = Math.max(1, Math.floor(img.height / f));
  const out = new Uint8ClampedArray(4 * w * h);
  if (f === 1) {
    for (let y = 0; y < h; y++) out.set(img.data.subarray(4 * y * img.width, 4 * (y * img.width + w)), 4 * y * w);
    return { width: w, height: h, data: out };
  }
  const n = f * f;
  const src = img.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;
      for (let dy = 0; dy < f; dy++) {
        let o = 4 * ((y * f + dy) * img.width + x * f);
        for (let dx = 0; dx < f; dx++, o += 4) {
          r += src[o];
          g += src[o + 1];
          b += src[o + 2];
          a += src[o + 3];
        }
      }
      const q = 4 * (y * w + x);
      out[q] = Math.round(r / n);
      out[q + 1] = Math.round(g / n);
      out[q + 2] = Math.round(b / n);
      out[q + 3] = Math.round(a / n);
    }
  }
  return { width: w, height: h, data: out };
}

/** PSNR (dB, peak 255) of the RGB channels of two RGBA8 images of one size (≤ 99 dB). */
export function rgbPsnr(a: RGBAImage, b: RGBAImage): number {
  if (a.width !== b.width || a.height !== b.height) throw new Error(`rgbPsnr: ${a.width}×${a.height} vs ${b.width}×${b.height}`);
  const px = a.width * a.height;
  let se = 0;
  for (let i = 0; i < px; i++) {
    for (let c = 0; c < 3; c++) {
      const e = a.data[4 * i + c] - b.data[4 * i + c];
      se += e * e;
    }
  }
  const mse = se / (3 * Math.max(1, px));
  return mse > 0 ? Math.min(PSNR_MAX, 10 * Math.log10((255 * 255) / mse)) : PSNR_MAX;
}
