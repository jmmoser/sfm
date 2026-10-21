/**
 * Options, scale-space planning, coordinate mapping and descriptor finalisation
 * shared by the CPU (sift-cpu.ts) and GPU (sift-gpu.ts) SIFT implementations,
 * so both produce features in the same frames and the same descriptor layout.
 *
 * Scale-space conventions (both implementations MUST follow them):
 *  - The input is area-downscaled so its long side <= maxImageSize ("processed" image,
 *    procWidth x procHeight). Processed pixel coordinates follow the resampling rule.
 *  - Octave o has a pixel grid with x_proc = x_oct * 2^o (o may be -1). Octave -1 is
 *    a 2x bilinear upsample with up(2i) = in(i), up(2i+1) = (in(i) + in(i+1)) / 2
 *    (edge clamped); octave o+1 is the decimation of level S of octave o:
 *    next(i, j) = cur(2i, 2j), size floor(w/2) x floor(h/2).
 *  - Octaves are generated while min(w, h) >= 16.
 *  - Gaussian level i (0..S+2) of an octave has blur sigma * 2^(i/S) in octave pixels,
 *    DoG level i = L(i+1) - L(i) (i = 0..S+1), extrema searched at DoG levels 1..S.
 *    A keypoint at continuous DoG level s has scale sigma * 2^(s/S) (octave pixels),
 *    its orientation / descriptor use Gaussian level clamp(round(s), 1, S).
 *  - Descriptor layout: index = (by * 4 + bx) * 8 + bo where bx (by) is the spatial bin
 *    along the keypoint's rotated x (y) axis — x' = (cos θ, sin θ), y' = (-sin θ, cos θ)
 *    in image coordinates (y down) — and bo the bin of (gradient angle - θ) in [0, 2π),
 *    bin bo centred at bo * 45°. θ follows the image rotation: warping an image by +α
 *    (clockwise as displayed) increases θ by α; an exact 90° turn gives θ + π/2 and
 *    identical descriptors (pinned by tests).
 *  - Reported keypoint scale is the Lowe / VLFeat sigma (≈ 0.89 x the sigma of an ideal
 *    Gaussian blob); score = |DoG| at the refined extremum.
 *  - maxFeatures counts ORIENTED features. The distributed selection (distribute.ts) runs
 *    twice: on keypoints before orientation (descriptors are only computed for survivors)
 *    and on the oriented features at the end.
 */

import { DESCRIPTOR_SIZE } from '../core/types';

export interface SiftOptions {
  /** Maximum number of (oriented) features returned. Default 4096. */
  maxFeatures: number;
  /** Input is area-downscaled so its long side <= this. Default 1600. */
  maxImageSize: number;
  /**
   * -1 = upsample x2 first, 0 = start at the processed resolution (default),
   * >= 1 = additionally area-downscale by 2^firstOctave. With the default 0, the
   * extractor automatically uses -1 when the processed long side < upsampleBelow.
   */
  firstOctave: number;
  /** Scales per octave S. Default 3. */
  scalesPerOctave: number;
  /** Base blur of level 0 (octave pixels). Default 1.6. */
  sigma: number;
  /** DoG peak threshold on 0..1 images. Default 0.02 / scalesPerOctave. */
  peakThreshold: number;
  /** Edge rejection ratio r (principal curvature ratio). Default 10. */
  edgeThreshold: number;
  /** Max orientations per keypoint. Default 2. */
  maxOrientations: number;
  /** Grid-based spatial distribution of the kept features. Default true. */
  distribute: boolean;
  /**
   * (Extension) When firstOctave === 0 and the processed long side is below this,
   * firstOctave -1 is used instead. Default 800. Set 0 to disable auto-upsampling.
   */
  upsampleBelow?: number;
}

export const DEFAULT_SIFT_OPTIONS: Readonly<Required<SiftOptions>> = Object.freeze({
  maxFeatures: 4096,
  maxImageSize: 1600,
  firstOctave: 0,
  scalesPerOctave: 3,
  sigma: 1.6,
  peakThreshold: 0.02 / 3,
  edgeThreshold: 10,
  maxOrientations: 2,
  distribute: true,
  upsampleBelow: 800,
});

/** Fills defaults. peakThreshold defaults to 0.02 / scalesPerOctave when not given. */
export function resolveSiftOptions(opts?: Partial<SiftOptions>): Required<SiftOptions> {
  const o = opts ?? {};
  const S = Math.max(1, Math.round(o.scalesPerOctave ?? DEFAULT_SIFT_OPTIONS.scalesPerOctave));
  return {
    maxFeatures: Math.max(0, Math.floor(o.maxFeatures ?? DEFAULT_SIFT_OPTIONS.maxFeatures)),
    maxImageSize: Math.max(16, Math.floor(o.maxImageSize ?? DEFAULT_SIFT_OPTIONS.maxImageSize)),
    firstOctave: Math.max(-1, Math.round(o.firstOctave ?? DEFAULT_SIFT_OPTIONS.firstOctave)),
    scalesPerOctave: S,
    sigma: o.sigma ?? DEFAULT_SIFT_OPTIONS.sigma,
    peakThreshold: o.peakThreshold ?? 0.02 / S,
    edgeThreshold: o.edgeThreshold ?? DEFAULT_SIFT_OPTIONS.edgeThreshold,
    maxOrientations: Math.max(1, Math.round(o.maxOrientations ?? DEFAULT_SIFT_OPTIONS.maxOrientations)),
    distribute: o.distribute ?? DEFAULT_SIFT_OPTIONS.distribute,
    upsampleBelow: o.upsampleBelow ?? DEFAULT_SIFT_OPTIONS.upsampleBelow,
  };
}

/** Nominal blur of the (processed) input image, in processed pixels. */
export const SIFT_NOMINAL_BLUR = 0.5;
/** Minimum octave side length. */
export const SIFT_MIN_OCTAVE_SIZE = 16;

export interface SiftPlan {
  inputWidth: number;
  inputHeight: number;
  /** Processed image size (input area-downscaled; octave 0 grid). */
  procWidth: number;
  procHeight: number;
  /** Effective first octave: -1 or 0 (positive requests are folded into procWidth/Height). */
  firstOctave: number;
  /** Octave sizes, index k = octave firstOctave + k. */
  octaves: { octave: number; width: number; height: number }[];
  /** Full-resolution frame the output coordinates refer to. */
  fullWidth: number;
  fullHeight: number;
  /** procWidth / fullWidth, procHeight / fullHeight. */
  kx: number;
  ky: number;
  /** Blur already present in the first octave's base image (octave pixels). */
  baseBlur: number;
}

/** Computes processed size, octaves and the full-resolution mapping. */
export function planSift(
  inputWidth: number,
  inputHeight: number,
  opts: Required<SiftOptions>,
  fullWidth = inputWidth,
  fullHeight = inputHeight,
): SiftPlan {
  if (!(inputWidth >= 1 && inputHeight >= 1)) throw new Error(`planSift: invalid image size ${inputWidth}x${inputHeight}`);
  const longSide = Math.max(inputWidth, inputHeight);
  let s = longSide > opts.maxImageSize ? opts.maxImageSize / longSide : 1;
  let fo = opts.firstOctave;
  const autoUpsample = fo === 0;
  if (fo > 0) {
    s /= 2 ** fo;
    fo = 0;
  }
  const procWidth = s === 1 ? inputWidth : Math.max(1, Math.round(inputWidth * s));
  const procHeight = s === 1 ? inputHeight : Math.max(1, Math.round(inputHeight * s));
  if (autoUpsample && Math.max(procWidth, procHeight) < opts.upsampleBelow) fo = -1;
  const octaves: SiftPlan['octaves'] = [];
  let w = fo < 0 ? procWidth * 2 : procWidth;
  let h = fo < 0 ? procHeight * 2 : procHeight;
  let o = fo;
  while (Math.min(w, h) >= SIFT_MIN_OCTAVE_SIZE) {
    octaves.push({ octave: o, width: w, height: h });
    w >>= 1;
    h >>= 1;
    o++;
  }
  return {
    inputWidth,
    inputHeight,
    procWidth,
    procHeight,
    firstOctave: fo,
    octaves,
    fullWidth,
    fullHeight,
    kx: procWidth / fullWidth,
    ky: procHeight / fullHeight,
    baseBlur: fo < 0 ? 2 * SIFT_NOMINAL_BLUR : SIFT_NOMINAL_BLUR,
  };
}

/**
 * Map a keypoint from octave coordinates to the full-resolution frame.
 * Writes [x, y, scale] into out[offset..offset+2].
 */
export function octaveToFull(
  plan: SiftPlan,
  octave: number,
  xOct: number,
  yOct: number,
  sigmaOct: number,
  out: Float32Array,
  offset: number,
): void {
  const f = 2 ** octave;
  const xp = xOct * f;
  const yp = yOct * f;
  out[offset] = (xp + 0.5) / plan.kx - 0.5;
  out[offset + 1] = (yp + 0.5) / plan.ky - 0.5;
  out[offset + 2] = (sigmaOct * f * 0.5) * (1 / plan.kx + 1 / plan.ky);
}

/**
 * Finalises a raw 128-bin SIFT histogram into a quantised RootSIFT descriptor:
 * L2 normalise, clamp at 0.2, L2 renormalise, L1 normalise, sqrt (unit L2),
 * quantise min(255, round(512 v)). `hist` is modified in place.
 */
export function finalizeRootSift(hist: Float32Array, out: Uint8Array, offset: number): void {
  const n = DESCRIPTOR_SIZE;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += hist[i] * hist[i];
  if (!(ss > 0)) {
    out.fill(0, offset, offset + n);
    return;
  }
  let inv = 1 / Math.sqrt(ss);
  ss = 0;
  for (let i = 0; i < n; i++) {
    let v = hist[i] * inv;
    if (v > 0.2) v = 0.2;
    hist[i] = v;
    ss += v * v;
  }
  inv = 1 / Math.sqrt(ss);
  let l1 = 0;
  for (let i = 0; i < n; i++) {
    const v = hist[i] * inv;
    hist[i] = v;
    l1 += v;
  }
  const il1 = 1 / l1;
  for (let i = 0; i < n; i++) {
    const q = Math.round(512 * Math.sqrt(hist[i] * il1));
    out[offset + i] = q > 255 ? 255 : q;
  }
}
