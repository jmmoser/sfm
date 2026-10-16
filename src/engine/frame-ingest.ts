/**
 * Frame ingest: orientation handling and per-frame image preparation.
 *
 * Orientation (ARCHITECTURE §2 / §2.2, docs/research/ios-camera.md §1.4):
 *  - the session's canonical orientation is the first accepted frame's (effective) θ₀; a frame
 *    captured at another θ is rotated by `canonicalRotation(θ₀, θ)` CLOCKWISE quarter turns so
 *    every keyframe shares one image frame and R_cam_device = cameraFromDevice(θ₀);
 *  - frames within `settleMs` (500 ms) of a UI rotation are dropped (Safari 26 transient
 *    mis-rotation);
 *  - RAW sensor buffers: iOS pixel buffers are sensor-landscape and only rotated by metadata.
 *    A landscape bitmap in a portrait UI (θ ∈ {0, 180}) is such a raw buffer, which is upright
 *    for θ = 90 — it is treated as captured at θ = 90. A portrait bitmap in a landscape UI is
 *    not explained by any delivery path (transient) and is dropped. The remaining ambiguity
 *    (raw buffer vs upright frame in landscape-270) is resolved by the engine's hand-eye check.
 *
 * Image preparation (ENGINE.md "Frame flow"): a ≤ 640 px RGBA copy (per-feature colours +
 * sharpness), the feature input (ImageBitmap / rotated OffscreenCanvas for the GPU SIFT, a gray
 * image at the working size for the CPU SIFT) and — for keyframes only — the full-resolution
 * canonical image as JPEG (OffscreenCanvas.convertToBlob, quality 0.92) or raw RGBA when no
 * JPEG encoder exists (Node tests). The source bitmap is closed by `release()`.
 */
import type { FeatureSet, GrayImage, ImuSample, RGBAImage, Vec3 } from '../core/types';
import { areaResampleRGBA } from '../dense/undistort';
import { rgbaToGray, rotateRGBA90, sharpness } from '../features/image';
import type { SiftBitmapSource } from '../features/sift-gpu';
import { canonicalRotation, rotateImagePoint, rotatedSize, type QuarterTurns, type ScreenAngle } from '../sensors/frame-orientation';
import { isRGBAImage, type StoredImage } from './persistence';

export type FrameSource = ImageBitmap | RGBAImage;

// ---------------------------------------------------------------------------------------------
// Orientation
// ---------------------------------------------------------------------------------------------

export type OrientationVerdict =
  | { ok: false; reason: string }
  | {
      ok: true;
      /** θ the pixels are upright for (90 for raw sensor buffers in a portrait UI). */
      effectiveTheta: ScreenAngle;
      /** Clockwise quarter turns into the canonical orientation. */
      k: QuarterTurns;
      raw: boolean;
    };

function snap(theta: number): ScreenAngle {
  const q = ((Math.round(theta / 90) % 4) + 4) % 4;
  return (q * 90) as ScreenAngle;
}

/** Tracks θ changes, detects raw sensor buffers and fixes the session's canonical θ₀. */
export class OrientationTracker {
  thetaSession: ScreenAngle | null;
  private lastUI: ScreenAngle | null = null;
  private changeT = -Infinity;

  constructor(
    readonly settleMs = 500,
    thetaSession: number | null = null,
  ) {
    this.thetaSession = thetaSession === null ? null : snap(thetaSession);
  }

  /** Decide for one frame (`width`/`height` = bitmap size as submitted). */
  classify(f: { width: number; height: number; screenAngle: number; t: number }): OrientationVerdict {
    const ui = snap(f.screenAngle);
    if (this.lastUI !== null && ui !== this.lastUI) this.changeT = f.t;
    this.lastUI = ui;
    if (Number.isFinite(f.t) && f.t - this.changeT < this.settleMs) return { ok: false, reason: `within ${this.settleMs} ms of a UI rotation` };
    const portraitUI = ui === 0 || ui === 180;
    const landscapeBitmap = f.width > f.height;
    const portraitBitmap = f.height > f.width;
    let eff: ScreenAngle = ui;
    let raw = false;
    if (portraitUI && landscapeBitmap) {
      // raw sensor-landscape buffer: upright for θ = 90 (ios-camera.md §1.4)
      eff = 90;
      raw = true;
    } else if (!portraitUI && portraitBitmap) {
      return { ok: false, reason: `portrait ${f.width}x${f.height} frame in a landscape UI (θ=${ui}): transient mis-rotation` };
    }
    if (this.thetaSession === null) this.thetaSession = eff;
    return { ok: true, effectiveTheta: eff, k: canonicalRotation(this.thetaSession, eff), raw };
  }
}

// ---------------------------------------------------------------------------------------------
// Image helpers
// ---------------------------------------------------------------------------------------------

export function hasOffscreenCanvas(): boolean {
  return typeof OffscreenCanvas !== 'undefined';
}

/** Area-averaged RGBA resize (8-bit output). */
export function resizeRGBA(src: RGBAImage, w: number, h: number): RGBAImage {
  if (w === src.width && h === src.height) return src;
  const f = areaResampleRGBA(src, w, h);
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < out.length; i++) out[i] = f[i];
  return { width: w, height: h, data: out };
}

/** Size with the long side ≤ maxSide (never enlarged). */
export function fitSize(w: number, h: number, maxSide: number): { width: number; height: number; scale: number } {
  const s = Math.min(1, maxSide / Math.max(w, h));
  return { width: Math.max(1, Math.round(w * s)), height: Math.max(1, Math.round(h * s)), scale: s };
}

/** Draw `src` rotated by k clockwise quarter turns onto a new canvas of the rotated size (scaled to w × h). */
function drawRotated(src: CanvasImageSource & { width: number; height: number }, k: QuarterTurns, w?: number, h?: number): OffscreenCanvas {
  const rs = rotatedSize(src.width, src.height, k);
  const W = w ?? rs.width;
  const H = h ?? rs.height;
  const canvas = new OffscreenCanvas(W, H);
  const ctx = canvas.getContext('2d', { willReadFrequently: false }) as OffscreenCanvasRenderingContext2D | null;
  if (!ctx) throw new Error('OffscreenCanvas 2D context unavailable');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  const sx = W / rs.width;
  const sy = H / rs.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(sx, sy);
  // rotation about the canvas origin, then translation back into view (rotateRGBA90 convention)
  if (k === 1) {
    ctx.translate(rs.width, 0);
    ctx.rotate(Math.PI / 2);
  } else if (k === 2) {
    ctx.translate(rs.width, rs.height);
    ctx.rotate(Math.PI);
  } else if (k === 3) {
    ctx.translate(0, rs.height);
    ctx.rotate(-Math.PI / 2);
  }
  ctx.drawImage(src, 0, 0);
  return canvas;
}

function canvasToRGBA(canvas: OffscreenCanvas): RGBAImage {
  const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
  const d = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return { width: d.width, height: d.height, data: d.data };
}

/** JPEG of an RGBA image (null without OffscreenCanvas). */
export async function encodeJpegRGBA(img: RGBAImage, quality = 0.92): Promise<Blob | null> {
  if (!hasOffscreenCanvas()) return null;
  const c = new OffscreenCanvas(img.width, img.height);
  const ctx = c.getContext('2d') as OffscreenCanvasRenderingContext2D | null;
  if (!ctx) return null;
  const data = img.data instanceof Uint8ClampedArray ? img.data : new Uint8ClampedArray(img.data.buffer, img.data.byteOffset, img.data.byteLength);
  ctx.putImageData(new ImageData(data.slice(0, img.width * img.height * 4), img.width, img.height), 0, 0);
  return c.convertToBlob({ type: 'image/jpeg', quality });
}

/** Feature-extraction input of a frame. */
export type FeatureInput = { kind: 'bitmap'; source: SiftBitmapSource } | { kind: 'gray'; gray: GrayImage };

/** A frame after orientation handling, ready for analysis. */
export interface PreparedFrame {
  /** Canonical full-resolution size (keypoint frame). */
  width: number;
  height: number;
  /** Size as submitted. */
  srcWidth: number;
  srcHeight: number;
  k: QuarterTurns;
  /** ≤ 640 px canonical RGBA copy and its scale (small = full × scale per axis). */
  small: RGBAImage;
  smallScaleX: number;
  smallScaleY: number;
  sharpness: number;
  /** Feature input: GPU pipelines take the bitmap / rotated canvas, CPU ones a working-size gray image. */
  featureInput(gpu: boolean, maxSide: number): FeatureInput;
  /** Full-resolution canonical image for storage (JPEG Blob, or RGBA without an encoder). */
  fullImage(quality?: number): Promise<StoredImage>;
  /** Close the source bitmap and drop canvases (idempotent). */
  release(): void;
}

/** Prepare a frame rotated by k quarter turns into the canonical orientation. */
export function prepareFrame(source: FrameSource, k: QuarterTurns, smallMaxSide = 640): PreparedFrame {
  return isRGBAImage(source) ? prepareRGBA(source, k, smallMaxSide) : prepareBitmap(source, k, smallMaxSide);
}

function smallSharpness(small: RGBAImage): number {
  return sharpness(rgbaToGray(small));
}

function prepareRGBA(src: RGBAImage, k: QuarterTurns, smallMaxSide: number): PreparedFrame {
  if (!(src.width > 0 && src.height > 0) || src.data.length < src.width * src.height * 4) throw new Error(`invalid RGBA frame ${src.width}x${src.height}`);
  let img: RGBAImage | null = k ? rotateRGBA90(src, k) : src;
  const W = img.width;
  const H = img.height;
  const fs = fitSize(W, H, smallMaxSide);
  const small = resizeRGBA(img, fs.width, fs.height);
  return {
    width: W,
    height: H,
    srcWidth: src.width,
    srcHeight: src.height,
    k,
    small,
    smallScaleX: fs.width / W,
    smallScaleY: fs.height / H,
    sharpness: smallSharpness(small),
    featureInput(_gpu: boolean, maxSide: number): FeatureInput {
      if (!img) throw new Error('frame released');
      const ws = fitSize(W, H, maxSide);
      const work = ws.width === W && ws.height === H ? img : resizeRGBA(img, ws.width, ws.height);
      return { kind: 'gray', gray: rgbaToGray(work) };
    },
    async fullImage(quality = 0.92): Promise<StoredImage> {
      if (!img) throw new Error('frame released');
      const cur = img;
      const jpeg = await encodeJpegRGBA(cur, quality);
      return jpeg ?? cur;
    },
    release(): void {
      img = null;
    },
  };
}

function prepareBitmap(bitmap: ImageBitmap, k: QuarterTurns, smallMaxSide: number): PreparedFrame {
  if (!hasOffscreenCanvas()) throw new Error('ImageBitmap frames need OffscreenCanvas');
  const sw = bitmap.width;
  const sh = bitmap.height;
  if (!(sw > 0 && sh > 0)) throw new Error(`invalid frame bitmap ${sw}x${sh}`);
  const rs = rotatedSize(sw, sh, k);
  const W = rs.width;
  const H = rs.height;
  let bmp: ImageBitmap | null = bitmap;
  let rotated: OffscreenCanvas | null = null;
  const canonical = (): CanvasImageSource & { width: number; height: number } => {
    if (!bmp) throw new Error('frame released');
    if (k === 0) return bmp;
    rotated ??= drawRotated(bmp, k);
    return rotated;
  };
  const fs = fitSize(W, H, smallMaxSide);
  // small copy straight from the bitmap (rotation + downscale in one draw)
  const small = canvasToRGBA(drawRotated(bitmap, k, fs.width, fs.height));
  return {
    width: W,
    height: H,
    srcWidth: sw,
    srcHeight: sh,
    k,
    small,
    smallScaleX: fs.width / W,
    smallScaleY: fs.height / H,
    sharpness: smallSharpness(small),
    featureInput(gpu: boolean, maxSide: number): FeatureInput {
      if (!bmp) throw new Error('frame released');
      if (gpu) {
        const src = canonical();
        return { kind: 'bitmap', source: src as SiftBitmapSource };
      }
      const ws = fitSize(W, H, maxSide);
      const c = drawRotated(bmp, k, ws.width, ws.height);
      return { kind: 'gray', gray: rgbaToGray(canvasToRGBA(c)) };
    },
    async fullImage(quality = 0.92): Promise<StoredImage> {
      if (!bmp) throw new Error('frame released');
      const c = k === 0 ? drawRotated(bmp, 0) : (canonical() as OffscreenCanvas);
      return c.convertToBlob({ type: 'image/jpeg', quality });
    },
    release(): void {
      if (bmp) {
        try {
          bmp.close();
        } catch {
          /* detached */
        }
      }
      bmp = null;
      if (rotated) {
        // shrink the backing store right away (the canvas may outlive us in a closure)
        rotated.width = 1;
        rotated.height = 1;
      }
      rotated = null;
    },
  };
}

/** Close a frame source that will not be processed. */
export function closeSource(source: FrameSource | null | undefined): void {
  if (!source || isRGBAImage(source)) return;
  try {
    (source as ImageBitmap).close?.();
  } catch {
    /* ignore */
  }
}

/** Size of a frame source as submitted. */
export function sourceSize(source: FrameSource): { width: number; height: number } {
  return { width: source.width, height: source.height };
}

// ---------------------------------------------------------------------------------------------
// Per-feature data
// ---------------------------------------------------------------------------------------------

/** sRGB colour per feature (3 B), bilinear in the small copy (§2 resampling rule). */
export function sampleFeatureColors(fs: FeatureSet, small: RGBAImage, sx: number, sy: number): Uint8Array {
  const out = new Uint8Array(3 * fs.count);
  const w = small.width;
  const h = small.height;
  const d = small.data;
  for (let i = 0; i < fs.count; i++) {
    const x = Math.min(w - 1, Math.max(0, (fs.keypoints[4 * i] + 0.5) * sx - 0.5));
    const y = Math.min(h - 1, Math.max(0, (fs.keypoints[4 * i + 1] + 0.5) * sy - 0.5));
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const x1 = Math.min(w - 1, x0 + 1);
    const y1 = Math.min(h - 1, y0 + 1);
    const ax = x - x0;
    const ay = y - y0;
    for (let c = 0; c < 3; c++) {
      const v00 = d[4 * (y0 * w + x0) + c];
      const v10 = d[4 * (y0 * w + x1) + c];
      const v01 = d[4 * (y1 * w + x0) + c];
      const v11 = d[4 * (y1 * w + x1) + c];
      out[3 * i + c] = Math.round((1 - ay) * ((1 - ax) * v00 + ax * v10) + ay * ((1 - ax) * v01 + ax * v11));
    }
  }
  return out;
}

/**
 * Overlay keypoints (strongest first — FeatureSets are sorted by score) in [0,1] coordinates of
 * the frame AS SUBMITTED: canonical pixels rotated by `kOut` clockwise quarter turns, the inverse
 * of the frame's canonical rotation k (kOut = (4 − k) mod 4).
 */
export function overlayPoints(fs: FeatureSet, width: number, height: number, kOut: number, maxPoints: number): Float32Array {
  const n = Math.min(fs.count, Math.max(0, maxPoints));
  const out = new Float32Array(2 * n);
  const rs = rotatedSize(width, height, kOut);
  const p = new Float64Array(2);
  for (let i = 0; i < n; i++) {
    rotateImagePoint(fs.keypoints[4 * i], fs.keypoints[4 * i + 1], width, height, kOut, p);
    out[2 * i] = (p[0] + 0.5) / rs.width;
    out[2 * i + 1] = (p[1] + 0.5) / rs.height;
  }
  return out;
}

/**
 * Unit gravity direction (DOWN) in the device frame at time t: the caller's `gravityDevice`
 * when given, else the mean specific force of the samples within ±`windowMs` (spec sign:
 * accelG points UP when still, so down = −accelG / |accelG|). null when unknown.
 */
export function gravityDown(samples: readonly ImuSample[], t: number, given?: ArrayLike<number> | null, windowMs = 80): [number, number, number] | null {
  if (given && given.length >= 3) {
    const n = Math.hypot(given[0], given[1], given[2]);
    if (n > 1e-6 && Number.isFinite(n)) return [given[0] / n, given[1] / n, given[2] / n];
  }
  let sx = 0,
    sy = 0,
    sz = 0,
    m = 0;
  for (let i = samples.length - 1; i >= 0; i--) {
    const s = samples[i];
    if (s.t < t - windowMs) break;
    if (s.t > t + windowMs || !s.accelG) continue;
    sx += s.accelG[0];
    sy += s.accelG[1];
    sz += s.accelG[2];
    m++;
  }
  if (m === 0) return null;
  const n = Math.hypot(sx, sy, sz);
  if (!(n > 1e-6)) return null;
  return [-sx / n, -sy / n, -sz / n];
}

/** R · v for a row-major 3×3. */
export function rotateVec(R: ArrayLike<number>, v: ArrayLike<number>): Vec3 {
  return Float64Array.of(R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]);
}

export { canonicalRotation, rotatedSize };
