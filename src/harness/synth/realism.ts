/**
 * iPhone video-capture artefacts for the synthetic generator (pure; Node-testable):
 * rolling-shutter readout geometry and band timing, focus breathing, white-balance drift,
 * and the adaptive motion-blur sample count. The renderer (render.ts) and the generator
 * (generate.ts) consume these; see e2e/README.md "Real-capture artefacts".
 *
 * ROLLING SHUTTER — direction, sign and timestamp (exact definition)
 * ------------------------------------------------------------------
 * The iPhone sensor buffer is "sensor landscape" (docs/research/ios-camera.md §1.4: MSTP
 * frames keep codedWidth > codedHeight in every UI orientation, portrait frames are tagged
 * with a 90° rotation). The native buffer is upright for θ = 90 (window.orientation 90,
 * device top edge pointing LEFT, AVCaptureVideoOrientationLandscapeRight), and the sensor
 * reads its native rows top (buffer row 0) to bottom.
 *
 * In the device frame (ARCHITECTURE §2: +x right edge, +y top edge, +z out of the screen):
 * at θ = 90, `R_cam_device` gives y_cam = −x_dev, so the first native row (smallest image v,
 * most negative y_cam) images the scene on the device's +x side. The readout therefore
 * sweeps the scene from the device +x (right-edge) side towards −x, whatever the UI
 * orientation. With a = R_cam_device(θ)·x̂_dev (device +x in camera axes):
 *
 *   θ =   0 (portrait):           a = (+1, 0) → image COLUMNS, from the RIGHT edge (u = W−1) to u = 0
 *   θ =  90 (landscape, top left): a = (0, −1) → image ROWS, from the TOP (v = 0) to v = H−1
 *   θ = 180:                      a = (−1, 0) → columns, from u = 0 to u = W−1
 *   θ = 270:                      a = (0, +1) → rows, from the bottom (v = H−1) to v = 0
 *
 * `reverse: true` flips the sweep (if on-device measurements show the opposite order).
 * Line index i ∈ [0, L−1] in readout order (L = W for columns, H for rows) of the pixel
 * centre (u, v); readout coordinate s = i / (L − 1) ∈ [0, 1]; line i is exposed around
 *
 *   t_line(i) = t_frame + (s − 0.5) · readoutMs        (line delay = readoutMs / (L − 1)),
 *
 * i.e. the first line at t_frame − readout/2 and the last at t_frame + readout/2. The frame
 * timestamp (frames.json `tTrue`, and `t` = tTrue + timeOffsetMs) and the GT pose are those
 * of MID-READOUT (s = 0.5), which is also the centre of the exposure of the middle line.
 * Rendering is band-wise: L lines are split into N contiguous bands; every pixel of band k
 * is rendered with the pose(s) at t_frame + (s̄_k − 0.5)·readout, s̄_k = mean s of the band's
 * lines (exact on average; per-line timing error ≤ ±(L/N/2)·line delay). N may adapt per
 * frame (rsBandCount) so that the image step between neighbouring bands stays ≤ ~0.3 px.
 */
import type { CameraIntrinsics, Pose } from '../../core/types';
import { cameraCenter, projectRadial, rCamDevice, undistortNormalized } from './conventions';
import { SynthRng } from './rng';
import type { WindowDepth } from './sampling';

// ---------------------------------------------------------------------------
// Rolling shutter
// ---------------------------------------------------------------------------

export interface ReadoutGeometry {
  /** 'cols': lines are image columns (index from u); 'rows': lines are image rows (from v). */
  axis: 'rows' | 'cols';
  /** +1: line index = pixel coordinate; −1: line index = L − 1 − pixel coordinate. */
  sign: 1 | -1;
  /** Number of lines L along the readout. */
  lines: number;
}

/** Readout geometry of the UI-upright image for screen angle θ (see the header). */
export function readoutGeometry(theta: number, width: number, height: number, reverse = false): ReadoutGeometry {
  const R = rCamDevice(theta);
  // a = R_cam_device · x̂_dev = first column of R (camera x, y components).
  const ax = R[0],
    ay = R[3];
  let g: ReadoutGeometry;
  if (Math.abs(ax) > 0.5) g = { axis: 'cols', sign: ax > 0 ? -1 : 1, lines: width };
  else if (Math.abs(ay) > 0.5) g = { axis: 'rows', sign: ay > 0 ? -1 : 1, lines: height };
  else throw new Error(`readoutGeometry: unsupported screen angle ${theta}`);
  if (reverse) g.sign = g.sign === 1 ? -1 : 1;
  return g;
}

/** Line index (continuous for non-integer pixel coordinates) of pixel (u, v) in readout order. */
export function readoutLine(g: ReadoutGeometry, u: number, v: number): number {
  const c = g.axis === 'cols' ? u : v;
  return g.sign === 1 ? c : g.lines - 1 - c;
}

/** Readout coordinate s ∈ [0, 1] of pixel (u, v): 0 = first line read, 1 = last. */
export function readoutCoordinate(g: ReadoutGeometry, u: number, v: number): number {
  return g.lines > 1 ? readoutLine(g, u, v) / (g.lines - 1) : 0.5;
}

/** Time offset (ms) of the line through pixel (u, v) relative to the frame (mid-readout) time. */
export function readoutTimeOffsetMs(g: ReadoutGeometry, readoutMs: number, u: number, v: number): number {
  return (readoutCoordinate(g, u, v) - 0.5) * readoutMs;
}

/** Inclusive rectangle of integer pixel centres. */
export interface PixelRect {
  u0: number;
  u1: number;
  v0: number;
  v1: number;
}

export interface ReadoutBand {
  index: number;
  /** Lines [line0, line1) in readout order. */
  line0: number;
  line1: number;
  /** Mean readout coordinate s̄ of the band's lines. */
  s: number;
  /** Time offset (ms) of the band from the frame (mid-readout) time: (s̄ − 0.5)·readoutMs. */
  dtMs: number;
  /** Output pixels of the band. */
  rect: PixelRect;
}

/**
 * Split the readout into `n` contiguous bands of ⌊L/n⌋ or ⌈L/n⌉ lines (band k = lines
 * [⌊kL/n⌋, ⌊(k+1)L/n⌋)); bands are listed in readout order and partition the image.
 */
export function readoutBands(g: ReadoutGeometry, width: number, height: number, n: number, readoutMs: number): ReadoutBand[] {
  const L = g.lines;
  const nb = Math.max(1, Math.min(L, Math.round(n)));
  const out: ReadoutBand[] = [];
  for (let k = 0; k < nb; k++) {
    const line0 = Math.floor((k * L) / nb),
      line1 = Math.floor(((k + 1) * L) / nb);
    const sMean = L > 1 ? (line0 + line1 - 1) / 2 / (L - 1) : 0.5;
    // pixel coordinate range of the lines
    const c0 = g.sign === 1 ? line0 : L - line1,
      c1 = g.sign === 1 ? line1 - 1 : L - 1 - line0;
    const rect: PixelRect = g.axis === 'cols' ? { u0: c0, u1: c1, v0: 0, v1: height - 1 } : { u0: 0, u1: width - 1, v0: c0, v1: c1 };
    out.push({ index: k, line0, line1, s: sMean, dtMs: (sMean - 0.5) * readoutMs, rect });
  }
  return out;
}

/**
 * Adaptive band count of a frame: the image moves by `rsPx` over the readout, so N bands leave
 * steps of rsPx/N px at the band edges. N = ⌈rsPx / step⌉ with step = max(stepPx, 0.1·blurPx)
 * (a blurred frame hides proportionally larger steps), clamped to [min(minBands, maxBands),
 * maxBands]. stepPx ≤ 0 → always maxBands.
 */
export function rsBandCount(rsPx: number, blurPx: number, maxBands: number, stepPx: number, minBands = 4): number {
  const max = Math.max(1, Math.round(maxBands));
  if (!(stepPx > 0)) return max;
  const step = Math.max(stepPx, 0.1 * (blurPx > 0 ? blurPx : 0));
  const n = Math.ceil((rsPx > 0 ? rsPx : 0) / step);
  return Math.max(Math.min(minBands, max), Math.min(max, n));
}

/** One band covering the whole image at the frame time (global shutter). */
export function globalShutterBand(width: number, height: number): ReadoutBand {
  return { index: 0, line0: 0, line1: 1, s: 0.5, dtMs: 0, rect: { u0: 0, u1: width - 1, v0: 0, v1: height - 1 } };
}

// ---------------------------------------------------------------------------
// Motion blur: sub-frame times and adaptive sample count
// ---------------------------------------------------------------------------

/**
 * Sub-frame time offsets (ms) across an exposure of `exposureMs` centred on 0.
 * 'endpoints' (legacy fixed-count blur): k/(n−1) − 0.5; 'midpoints' (adaptive): (k+0.5)/n − 0.5.
 */
export function subframeOffsetsMs(n: number, exposureMs: number, scheme: 'endpoints' | 'midpoints'): number[] {
  if (n <= 1) return [0];
  const out: number[] = [];
  for (let k = 0; k < n; k++) out.push((scheme === 'endpoints' ? k / (n - 1) - 0.5 : (k + 0.5) / n - 0.5) * exposureMs);
  return out;
}

/**
 * Adaptive blur sample count for an image motion of `dispPx` over the exposure: 1 below
 * 0.25 px, else 1 + ⌈disp / pxPerSample⌉, at most `maxSamples` (≥ 1). pxPerSample ≤ 0 means
 * the fixed count `maxSamples`.
 */
export function blurSampleCount(dispPx: number, pxPerSample: number, maxSamples: number): number {
  const max = Math.max(1, Math.round(maxSamples));
  if (!(pxPerSample > 0)) return max;
  if (!(dispPx >= 0.25)) return 1;
  return Math.min(max, 1 + Math.ceil(dispPx / pxPerSample));
}

/**
 * Largest image displacement (px) between two poses of the probe pixels, for scene points at
 * z-depth `depth` along the probe rays of pose A (RADIAL model of `cam`).
 */
export function imageMotionPx(cam: CameraIntrinsics, a: Pose, b: Pose, probes: readonly (readonly [number, number])[], depth: number): number {
  const C = cameraCenter(a);
  const xy = new Float64Array(2);
  const X = new Float64Array(3);
  const out = new Float64Array(3);
  let max = 0;
  for (const [u, v] of probes) {
    undistortNormalized(cam.k1, cam.k2, (u - cam.cx) / cam.f, (v - cam.cy) / cam.f, xy);
    // X = C + Rᵀ (x·d, y·d, d)
    const cx = xy[0] * depth,
      cy = xy[1] * depth,
      cz = depth;
    const R = a.R;
    X[0] = C[0] + R[0] * cx + R[3] * cy + R[6] * cz;
    X[1] = C[1] + R[1] * cx + R[4] * cy + R[7] * cz;
    X[2] = C[2] + R[2] * cx + R[5] * cy + R[8] * cz;
    if (!projectRadial(cam, b, X, out)) return Infinity;
    max = Math.max(max, Math.hypot(out[0] - u, out[1] - v));
  }
  return max;
}

/** Probe pixels of a rectangle: its centre, the 4 corners and the 4 edge midpoints. */
export function rectProbes(r: PixelRect): [number, number][] {
  const us = [r.u0, (r.u0 + r.u1) / 2, r.u1],
    vs = [r.v0, (r.v0 + r.v1) / 2, r.v1];
  const out: [number, number][] = [];
  for (const v of vs) for (const u of us) out.push([u, v]);
  return out;
}

// ---------------------------------------------------------------------------
// Focus breathing
// ---------------------------------------------------------------------------

export interface FocusBreathingConfig {
  /**
   * Largest relative focal change: f_i = f0·(1 + b_i) with b_i ∈ [−amplitude, +amplitude]
   * (0.006 = ±0.6 %). Near focus → longer focal (b > 0).
   */
  amplitude: number;
  /** Autofocus lag: time constant (s) of the first-order follower (in diopters) of the target distance. */
  lagS: number;
  /** Lower bound (diopters) of the half-range used to normalise the series (avoids amplifying tiny changes). */
  minHalfRangeD: number;
  /** Closest focus distance (m); nearer targets are clamped. */
  minFocusM: number;
  /** Target distance (m) when the AF region sees no surface (sky). */
  skyM: number;
  /** Half-size of the centred AF region as a fraction of the image width / height. */
  regionFrac: number;
}

export const DEFAULT_FOCUS_BREATHING: FocusBreathingConfig = {
  amplitude: 0.006,
  lagS: 0.25,
  minHalfRangeD: 0.02,
  minFocusM: 0.1,
  skyM: 100,
  regionFrac: 0.15,
};

/**
 * Autofocus target distance (m): median z-depth of the centred region (±regionFrac of the
 * image size around the principal point) of an undistorted window depth map at the frame pose.
 */
export function centerFocusDistance(wd: WindowDepth, cam: CameraIntrinsics, cfg: Pick<FocusBreathingConfig, 'regionFrac' | 'skyM'> = DEFAULT_FOCUS_BREATHING): number {
  const hx = (cfg.regionFrac * cam.width) / cam.f,
    hy = (cfg.regionFrac * cam.height) / cam.f;
  const { xmin, xmax, ymin, ymax } = wd.window;
  const px = (xmax - xmin) / wd.width,
    py = (ymax - ymin) / wd.height;
  const vals: number[] = [];
  let total = 0;
  for (let r = 0; r < wd.height; r++) {
    const y = ymin + (r + 0.5) * py;
    if (Math.abs(y) > hy) continue;
    for (let c = 0; c < wd.width; c++) {
      const x = xmin + (c + 0.5) * px;
      if (Math.abs(x) > hx) continue;
      total++;
      const z = wd.data[r * wd.width + c];
      if (z > 0) vals.push(z);
    }
  }
  if (vals.length === 0 || vals.length < 0.25 * total) return cfg.skyM;
  vals.sort((a, b) => a - b);
  return vals[vals.length >> 1];
}

export interface FocusSeries {
  /** AF target distance per frame (m), as given (clamped to minFocusM). */
  targetM: number[];
  /** AF state per frame in diopters (1/m) after the lag filter. */
  diopters: number[];
  /** Relative focal change per frame: f_i = f0·(1 + b_i). */
  b: number[];
}

/**
 * Focus-breathing series from per-frame AF target distances at times `timesS` (s, increasing).
 * The AF state follows the target in diopters with a first-order lag (exact step response at
 * the frame times): D_i = D_{i−1} + (1 − exp(−Δt/lagS))·(D*_i − D_{i−1}), D_0 = D*_0. The focal
 * change is linear in diopters (lens extension ∝ 1/d), scaled so that the sequence spans exactly
 * ±amplitude about the middle of its diopter range: b_i = amplitude·(D_i − D_mid)/max(D_half, minHalfRangeD).
 */
export function focusBreathingSeries(targetM: readonly number[], timesS: readonly number[], cfg: FocusBreathingConfig): FocusSeries {
  const n = targetM.length;
  const tgt = targetM.map((d) => Math.max(cfg.minFocusM, d));
  const D: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const Dt = 1 / tgt[i];
    if (i === 0) D[i] = Dt;
    else {
      const dt = Math.max(0, timesS[i] - timesS[i - 1]);
      const a = cfg.lagS > 0 ? 1 - Math.exp(-dt / cfg.lagS) : 1;
      D[i] = D[i - 1] + a * (Dt - D[i - 1]);
    }
  }
  let lo = Infinity,
    hi = -Infinity;
  for (const d of D) {
    lo = Math.min(lo, d);
    hi = Math.max(hi, d);
  }
  const mid = (lo + hi) / 2,
    half = Math.max((hi - lo) / 2, cfg.minHalfRangeD);
  const b = D.map((d) => (cfg.amplitude * (d - mid)) / half);
  return { targetM: tgt, diopters: D, b };
}

// ---------------------------------------------------------------------------
// White balance drift
// ---------------------------------------------------------------------------

/**
 * Slow white-balance drift: per-channel linear gains (r, g, b) with a colour-temperature swing
 * r = 1 + A·w(τ), b = 1 − A·w(τ) (w = 0.8 sin(2πτ/13 + φ1) + 0.2 sin(2πτ/4.1 + φ2)) and a smaller
 * green/magenta tint g = 1 + 0.3·A·sin(2πτ/9.7 + φ3). A = 0 → (1, 1, 1).
 */
export function whiteBalanceFn(seed: number, amplitude: number): (tau: number) => [number, number, number] {
  if (!(amplitude > 0)) return () => [1, 1, 1];
  const rng = new SynthRng(seed).derive('white-balance');
  const p1 = rng.next() * 6.283,
    p2 = rng.next() * 6.283,
    p3 = rng.next() * 6.283;
  return (tau) => {
    const w = 0.8 * Math.sin((2 * Math.PI * tau) / 13 + p1) + 0.2 * Math.sin((2 * Math.PI * tau) / 4.1 + p2);
    const g = Math.sin((2 * Math.PI * tau) / 9.7 + p3);
    return [1 + amplitude * w, 1 + 0.3 * amplitude * g, 1 - amplitude * w];
  };
}
