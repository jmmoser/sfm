/**
 * Splat renderer of the novel-view evaluation (nvs.html, e2e/nvs.mjs; the hook is
 * `loadSplatRenderer` in src/harness/nvs/renderers.ts): a run's `splats.ply` (INRIA 3DGS layout,
 * the run's world frame — what `exportModel('ply-splat')` writes) drawn by `SplatRenderer` at the
 * test cameras.
 *
 * Colour convention: the model is fitted to the sRGB training images, so the rasteriser's colour is
 * already sRGB (no transfer function). Rendered over a black background it is the PREMULTIPLIED
 * colour Σᵢ Tᵢ αᵢ cᵢ, and 1 − T (final transmittance) is the coverage: exactly the RGBA8 the
 * harness composites over its background. Rows run top to bottom (pixel (x, y) at y·W + x, +y
 * down as in the OpenCV pinhole camera), like the rasteriser's output. The exported model is
 * appearance-normalised (the mean training-view colour matrix is folded into the colours).
 */
import { getGPU } from '../gpu/device';
import { decodeSplatPLY } from './ply';
import { SplatRenderer } from './renderer';
import type { SplatCamera } from './types';

export interface NvsCameraInput {
  width: number;
  height: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  /** World-to-camera rotation (row-major 3×3) and translation (OpenCV camera frame). */
  R: ArrayLike<number>;
  t: ArrayLike<number>;
}

export interface NvsSplatRenderer {
  readonly name: 'splat';
  /** RGBA8 (premultiplied sRGB, alpha = 1 − transmittance), cam.width × cam.height, rows top to bottom. */
  render(cam: NvsCameraInput): Promise<Uint8Array>;
  dispose(): void;
}

/** Premultiplied RGBA8 of a render over a black background (colour, transmittance in A). */
export function premultipliedRGBA8(data: Float32Array, pixels: number, out = new Uint8Array(4 * pixels)): Uint8Array {
  const q = (v: number): number => (v > 0 ? (v < 1 ? Math.round(255 * v) : 255) : 0);
  for (let i = 0; i < pixels; i++) {
    out[4 * i] = q(data[4 * i]);
    out[4 * i + 1] = q(data[4 * i + 1]);
    out[4 * i + 2] = q(data[4 * i + 2]);
    out[4 * i + 3] = q(1 - data[4 * i + 3]);
  }
  return out;
}

export async function createNvsRenderer(input: { ply: ArrayBuffer | Uint8Array; width: number; height: number }): Promise<NvsSplatRenderer> {
  const gpu = await getGPU();
  if (!gpu) throw new Error('splat renderer: no WebGPU device');
  const bytes = input.ply instanceof Uint8Array ? input.ply : new Uint8Array(input.ply);
  const { model, antialiased } = decodeSplatPLY(bytes, 1);
  const r = await SplatRenderer.create(gpu, { shDegree: model.shDegree, antialiased, maxWidth: input.width, maxHeight: input.height });
  r.setModel(model);
  return {
    name: 'splat',
    async render(cam) {
      if (cam.width > input.width || cam.height > input.height) throw new Error(`splat renderer: ${cam.width}×${cam.height} exceeds the ${input.width}×${input.height} it was created for`);
      const c: SplatCamera = { width: cam.width, height: cam.height, fx: cam.fx, fy: cam.fy, cx: cam.cx, cy: cam.cy, R: cam.R, t: cam.t };
      // black background ⇒ the colour channels are the premultiplied radiance
      const img = await r.render(c, { background: [0, 0, 0] });
      return premultipliedRGBA8(img.data, img.width * img.height);
    },
    dispose: () => r.destroy(),
  };
}
