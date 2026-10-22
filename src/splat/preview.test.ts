import { describe, expect, it } from 'vitest';
import type { RGBAImage } from '../core/types';
import { Rng } from '../math/random';
import { boxDownscale, previewFactor, previewViewIndex, rgbPsnr } from './preview';
import { downscaleSplatCamera } from './types';

function image(w: number, h: number, f: (x: number, y: number, c: number) => number): RGBAImage {
  const data = new Uint8ClampedArray(4 * w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 4; c++) data[4 * (y * w + x) + c] = f(x, y, c);
  return { width: w, height: h, data };
}

describe('splat previews', () => {
  it('previewViewIndex: the training view nearest the middle registered image', () => {
    const images = [10, 11, 12, 13, 14, 15, 16];
    // the middle image (13) is a training view
    expect(previewViewIndex([10, 13, 16], images)).toBe(1);
    // not a training view: the nearest one (12 and 14 tie: the earlier)
    expect(previewViewIndex([10, 12, 14, 16], images)).toBe(1);
    expect(previewViewIndex([10, 11, 14, 16], images)).toBe(2);
    // views unknown to the model count as farthest; a single view is the answer
    expect(previewViewIndex([99, 16], images)).toBe(1);
    expect(previewViewIndex([99], images)).toBe(0);
    expect(() => previewViewIndex([], images)).toThrow();
  });

  it('previewFactor: the smallest integer factor to ≤ the preview size', () => {
    expect(previewFactor(600, 800, 400)).toBe(2);
    expect(previewFactor(768, 1024, 400)).toBe(3);
    expect(previewFactor(960, 1280, 400)).toBe(4);
    expect(previewFactor(120, 160, 400)).toBe(1);
    expect(previewFactor(801, 600, 400)).toBe(3);
  });

  it('boxDownscale: the block means of downscaleSplatCamera, at its size', () => {
    const img = image(7, 5, (x, y, c) => (c === 3 ? 255 : 10 * x + 40 * y + c));
    const cam = downscaleSplatCamera({ width: 7, height: 5, fx: 1, fy: 1, cx: 3, cy: 2, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] }, 2);
    const d = boxDownscale(img, 2);
    expect([d.width, d.height]).toEqual([cam.width, cam.height]);
    // pixel (1, 1): the block (2..3, 2..3) — x̄ = 2.5, ȳ = 2.5
    expect(Array.from(d.data.subarray(4 * (1 * 3 + 1), 4 * (1 * 3 + 1) + 4))).toEqual([Math.round(25 + 100), Math.round(25 + 100 + 1), 25 + 100 + 2, 255]);
    // factor 1: the same pixels (a copy)
    const same = boxDownscale(img, 1);
    expect(Array.from(same.data)).toEqual(Array.from(img.data));
    expect(same.data).not.toBe(img.data);
  });

  it('rgbPsnr: RGB only, peak 255, capped for identical images', () => {
    const a = image(4, 4, (_x, _y, c) => (c === 3 ? 255 : 100));
    expect(rgbPsnr(a, a)).toBe(99);
    // every RGB value off by 10: 10·log10(255² / 100)
    const b = image(4, 4, (_x, _y, c) => (c === 3 ? 0 : 110));
    expect(rgbPsnr(a, b)).toBeCloseTo(10 * Math.log10((255 * 255) / 100), 9);
    const rng = new Rng(3);
    const noisy = image(4, 4, (_x, _y, c) => (c === 3 ? 255 : 100 + Math.round(rng.uniform(-40, 40))));
    expect(rgbPsnr(a, noisy)).toBeLessThan(rgbPsnr(a, b));
    expect(() => rgbPsnr(a, image(2, 2, () => 0))).toThrow();
  });
});
