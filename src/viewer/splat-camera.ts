/**
 * Pure helpers of the viewer's splat layer (unit-tested in splat-camera.test.ts): the pinhole
 * SplatCamera that sees exactly what a three.js camera sees, and the frame size of the pixel
 * budget. No three.js / WebGPU imports: matrices are plain column-major arrays (three's
 * `Matrix4.elements`).
 */
import type { SplatCamera } from '../splat/types';

/**
 * The SplatCamera (OpenCV frame, world-to-camera R row-major + t, pinhole fx fy cx cy) of a
 * three.js perspective camera for an image of `width` × `height` pixels.
 *
 * `view`: world (the reconstruction's frame, under the viewer's world root) → three camera, i.e.
 * `camera.matrixWorldInverse × root.matrixWorld`; `projection`: `camera.projectionMatrix`
 * (any perspective projection: fov, aspect, zoom and view offsets are all in it). Both
 * column-major. three's camera looks down −z with +y up; OpenCV's down +z with +y down, so rows 1
 * and 2 of the view change sign. Projection: x_ndc = P00·x/z + (−P02) in OpenCV coordinates, and
 * NDC [−1, 1] spans the image edges [−0.5, W − 0.5] (pixel centres at integers, ARCHITECTURE §2):
 * u = W/2·(x_ndc + 1) − 0.5, v = H/2·(1 − y_ndc) − 0.5.
 */
export function splatCameraFromView(view: ArrayLike<number>, projection: ArrayLike<number>, width: number, height: number, near: number): SplatCamera {
  const m = view;
  // column-major: element (r, c) at m[4c + r]
  const R = [m[0], m[4], m[8], -m[1], -m[5], -m[9], -m[2], -m[6], -m[10]];
  const t = [m[12], -m[13], -m[14]];
  const p = projection;
  const fx = 0.5 * width * p[0];
  const fy = 0.5 * height * p[5];
  const cx = 0.5 * width * (1 - p[8]) - 0.5;
  const cy = 0.5 * height * (1 + p[9]) - 0.5;
  return { width, height, fx, fy, cx, cy, R, t, near };
}

export interface SplatFrameSize {
  /** Canvas (backing store) size in pixels. */
  width: number;
  height: number;
  /** Canvas pixels per CSS pixel. */
  pixelRatio: number;
}

/**
 * Canvas size for a CSS box: devicePixelRatio capped by `maxPixelRatio` and by the pixel budget
 * (`maxPixels`: the ratio is lowered so width × height ≤ maxPixels).
 */
export function splatCanvasSize(cssWidth: number, cssHeight: number, devicePixelRatio: number, maxPixelRatio: number, maxPixels: number): SplatFrameSize {
  if (!(cssWidth > 0) || !(cssHeight > 0)) return { width: 0, height: 0, pixelRatio: 0 };
  const dpr = devicePixelRatio > 0 && Number.isFinite(devicePixelRatio) ? devicePixelRatio : 1;
  let pr = Math.min(dpr, maxPixelRatio > 0 ? maxPixelRatio : dpr, Math.sqrt(maxPixels / (cssWidth * cssHeight)));
  let width = Math.max(1, Math.floor(cssWidth * pr));
  let height = Math.max(1, Math.floor(cssHeight * pr));
  // rounding up cannot push it over the budget (floor), but keep the invariant explicit
  while (width * height > maxPixels && pr > 0) {
    pr *= 0.99;
    width = Math.max(1, Math.floor(cssWidth * pr));
    height = Math.max(1, Math.floor(cssHeight * pr));
  }
  return { width, height, pixelRatio: pr };
}

/**
 * Render size of a frame drawn while the view moves: the canvas size scaled down to at most
 * `pixels` (same aspect; at least 1 × 1). Returns the canvas size when it already fits.
 */
export function interactiveFrameSize(width: number, height: number, pixels: number): { width: number; height: number } {
  if (width * height <= pixels) return { width, height };
  const s = Math.sqrt(pixels / (width * height));
  return { width: Math.max(1, Math.round(width * s)), height: Math.max(1, Math.round(height * s)) };
}

/** Robust bounds input for a splat model: an evenly spread subsample of the Gaussian means. */
export function subsampleMeans(means: Float32Array, count: number, max: number): Float32Array {
  const n = Math.min(count, Math.floor(means.length / 3));
  if (n <= max) return means.slice(0, 3 * n);
  const out = new Float32Array(3 * max);
  const step = n / max;
  for (let i = 0; i < max; i++) {
    const k = Math.floor(i * step);
    out[3 * i] = means[3 * k];
    out[3 * i + 1] = means[3 * k + 1];
    out[3 * i + 2] = means[3 * k + 2];
  }
  return out;
}
