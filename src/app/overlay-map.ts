/**
 * Mapping of normalised frame coordinates (LiveUpdate.overlay, [0,1]², relative to the frame
 * as submitted) to CSS pixels of a `<video>` element drawn with `object-fit: cover` (or
 * `contain`) and `object-position: 50% 50%`.
 */
import { canonicalRotation, snapScreenAngle, type ScreenAngle } from '../sensors/frame-orientation';

/**
 * Which way the pixels of each submitted frame are upright — the SAME rule as the engine's
 * `OrientationTracker` (src/engine/frame-ingest.ts), so a keypoint overlay given relative to the
 * bitmap AS SUBMITTED (protocol) can be turned onto the `<video>` preview, which is always upright
 * for the current UI rotation:
 *  - a landscape bitmap in a portrait UI (θ ∈ {0, 180}) is a RAW sensor buffer (iOS pixel buffers
 *    are sensor-landscape, rotated by metadata only): upright for θ = 90;
 *  - raw delivery is a property of the capture path, fixed by the FIRST portrait-UI frame: with raw
 *    delivery the landscape bitmaps of a landscape UI are raw too (upright for 90 also at θ = 270);
 *  - a portrait bitmap in a landscape UI is a transient mis-rotation the engine drops: no overlay;
 *  - frames within `settleMs` of a UI rotation are ignored for the delivery decision (as the engine
 *    drops them before classifying);
 *  - a frame declared raw (`raw`, FrameInput.rawBuffer) is upright for θ = 90 in every UI rotation,
 *    without changing `rawDelivery` (the engine's stateless rule).
 * The preview shown at θ_display needs `canonicalRotation(θ_display, θ_upright)` CLOCKWISE quarter
 * turns of the frame (θ = 90 raw buffer in a portrait UI: 1 turn).
 */
export class FrameOrientationTracker {
  /** true = raw sensor buffers, false = frames rotated upright, null = not known yet. */
  rawDelivery: boolean | null = null;
  private lastUI: ScreenAngle | null = null;
  private changeT = -Infinity;

  constructor(readonly settleMs = 500) {}

  /**
   * Observe a submitted frame (call for EVERY frame, in submission order) and return the θ its
   * pixels are upright for, or null when the engine drops such a frame (no overlay).
   */
  observe(f: { width: number; height: number; screenAngle: number; t?: number; raw?: boolean }): ScreenAngle | null {
    const ui = snapScreenAngle(f.screenAngle);
    const t = f.t ?? NaN;
    if (this.lastUI !== null && ui !== this.lastUI) this.changeT = t;
    this.lastUI = ui;
    const settling = Number.isFinite(t) && t - this.changeT < this.settleMs;
    // a frame declared raw (FrameInput.rawBuffer): upright for θ = 90; per frame, no state change
    if (f.raw) return settling || f.height > f.width ? null : 90;
    return this.uprightTheta(f.width, f.height, ui, !settling);
  }

  /** θ the pixels of a `width × height` bitmap submitted at UI rotation `ui` are upright for (null: dropped). */
  uprightTheta(width: number, height: number, ui: number, noteDelivery = false): ScreenAngle | null {
    const s = snapScreenAngle(ui);
    const portraitUI = s === 0 || s === 180;
    if (portraitUI) {
      if (width > height) {
        if (noteDelivery && this.rawDelivery === null) this.rawDelivery = true;
        return 90;
      }
      if (height > width && noteDelivery && this.rawDelivery === null) this.rawDelivery = false;
      return s;
    }
    if (height > width) return null;
    if (width > height && this.rawDelivery === true) return 90;
    return s;
  }

  /**
   * Clockwise quarter turns that map the overlay of a frame (relative to the bitmap as
   * submitted, upright for `uprightTheta`) onto the preview shown at UI rotation `displayAngle`.
   */
  static overlayTurns(uprightTheta: number, displayAngle: number): number {
    return canonicalRotation(displayAngle, uprightTheta);
  }
}

export interface CoverTransform {
  /** CSS px per frame pixel. */
  scale: number;
  /** Displayed frame size (CSS px). */
  width: number;
  height: number;
  /** Top-left of the displayed frame relative to the element (≤ 0 when cropped). */
  offsetX: number;
  offsetY: number;
}

/**
 * Transform of a `srcW × srcH` frame drawn into an `elW × elH` box. `cover` scales by the larger
 * ratio and crops the overflow symmetrically; `contain` by the smaller one (letterbox).
 */
export function fitTransform(srcW: number, srcH: number, elW: number, elH: number, fit: 'cover' | 'contain' = 'cover'): CoverTransform {
  if (!(srcW > 0 && srcH > 0 && elW > 0 && elH > 0)) return { scale: 0, width: 0, height: 0, offsetX: 0, offsetY: 0 };
  const sx = elW / srcW;
  const sy = elH / srcH;
  const scale = fit === 'cover' ? Math.max(sx, sy) : Math.min(sx, sy);
  const width = srcW * scale;
  const height = srcH * scale;
  return { scale, width, height, offsetX: (elW - width) / 2, offsetY: (elH - height) / 2 };
}

/** Normalised frame point → element CSS px. */
export function mapNormalizedPoint(u: number, v: number, t: CoverTransform): { x: number; y: number } {
  return { x: t.offsetX + u * t.width, y: t.offsetY + v * t.height };
}

/** Element CSS px → normalised frame point (inverse of mapNormalizedPoint). */
export function unmapPoint(x: number, y: number, t: CoverTransform): { u: number; v: number } {
  return { u: t.width > 0 ? (x - t.offsetX) / t.width : NaN, v: t.height > 0 ? (y - t.offsetY) / t.height : NaN };
}

/**
 * Rotate a normalised point by `k` CLOCKWISE quarter turns (as displayed) — the same convention
 * as `rotateRGBA90` / `canonicalRotation`: k = 1 maps (u, v) → (1 − v, u).
 */
export function rotateNormalizedPoint(u: number, v: number, k: number): { u: number; v: number } {
  const q = ((Math.round(k) % 4) + 4) % 4;
  switch (q) {
    case 1:
      return { u: 1 - v, v: u };
    case 2:
      return { u: 1 - u, v: 1 - v };
    case 3:
      return { u: v, v: 1 - u };
    default:
      return { u, v };
  }
}

/**
 * Map an interleaved overlay (u0, v0, u1, v1, …) to element coordinates. Points outside the
 * visible (cropped) area are dropped. `quarterTurns` rotates the points first (frame captured
 * at a different UI rotation than the current one). Returns interleaved CSS px (x, y, …).
 */
export function mapOverlay(
  overlay: ArrayLike<number>,
  frameW: number,
  frameH: number,
  elW: number,
  elH: number,
  opts: { fit?: 'cover' | 'contain'; quarterTurns?: number; margin?: number } = {},
): Float32Array {
  const k = ((Math.round(opts.quarterTurns ?? 0) % 4) + 4) % 4;
  // A rotated frame swaps its aspect.
  const w = k % 2 ? frameH : frameW;
  const h = k % 2 ? frameW : frameH;
  const t = fitTransform(w, h, elW, elH, opts.fit ?? 'cover');
  const m = opts.margin ?? 0;
  const n = Math.floor(overlay.length / 2);
  const out = new Float32Array(n * 2);
  let j = 0;
  for (let i = 0; i < n; i++) {
    let u = overlay[2 * i];
    let v = overlay[2 * i + 1];
    if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
    if (k) ({ u, v } = rotateNormalizedPoint(u, v, k));
    const x = t.offsetX + u * t.width;
    const y = t.offsetY + v * t.height;
    if (x < -m || y < -m || x > elW + m || y > elH + m) continue;
    out[j++] = x;
    out[j++] = y;
  }
  return out.subarray(0, j);
}
