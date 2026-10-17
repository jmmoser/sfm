/**
 * RADIAL → pinhole undistortion for dense stereo (ARCHITECTURE §4.6).
 *
 * The output camera keeps the focal length and principal point of the input camera,
 * resampled with the §2 rule (`scaleIntrinsics` semantics: f' = f·(sx + sy)/2,
 * c' = (c + 0.5)·s − 0.5) to an image whose long side is ≤ maxSize, with k1 = k2 = 0.
 * Every output pixel is inverse-mapped through the RADIAL model into the source and
 * sampled bilinearly. When the output is smaller than the source, the source is first
 * box-filtered (exact area averaging) to the output scale, so downscaling does not
 * alias. Output pixels that map outside the source image (or beyond the fold radius
 * of a strongly distorted lens, where the RADIAL model stops being monotone) get
 * RGBA = (0, 0, 0, 0).
 */
import type { CameraIntrinsics, RGBAImage } from '../core/types';
import { scaleCamera } from './dense-util';

export interface UndistortOptions {
  /** Box-filter the source when the output is smaller (default true). */
  antialias?: boolean;
}

/** Size of the undistorted image for a camera and a long-side limit (never upscales). */
export function undistortedSize(width: number, height: number, maxSize: number): { width: number; height: number } {
  const s = maxSize > 0 ? Math.min(1, maxSize / Math.max(width, height)) : 1;
  return { width: Math.max(1, Math.round(width * s)), height: Math.max(1, Math.round(height * s)) };
}

/** Pinhole intrinsics produced by `undistortImage(·, cam, maxSize)` (without processing pixels). */
export function undistortedCamera(cam: CameraIntrinsics, maxSize: number): CameraIntrinsics {
  const { width, height } = undistortedSize(cam.width, cam.height, maxSize);
  const c = scaleCamera(cam, width, height);
  c.k1 = 0;
  c.k2 = 0;
  return c;
}

/**
 * Squared normalised radius beyond which r·(1 + k1 r² + k2 r⁴) stops increasing
 * (first positive root of 1 + 3 k1 s + 5 k2 s², s = r²); Infinity if monotone.
 */
export function radialFoldRadiusSq(k1: number, k2: number): number {
  if (k2 === 0) return k1 < 0 ? -1 / (3 * k1) : Infinity;
  const a = 5 * k2;
  const b = 3 * k1;
  const disc = b * b - 4 * a;
  if (disc < 0) return Infinity;
  const sq = Math.sqrt(disc);
  const r1 = (-b - sq) / (2 * a);
  const r2 = (-b + sq) / (2 * a);
  let best = Infinity;
  if (r1 > 0) best = Math.min(best, r1);
  if (r2 > 0) best = Math.min(best, r2);
  return best;
}

/** Horizontal / vertical box-filter weights (CSR) for shrinking n → m samples (m ≤ n). */
function areaWeights(n: number, m: number): { start: Int32Array; idx: Int32Array; w: Float32Array } {
  const s = m / n;
  const start = new Int32Array(m + 1);
  const idxList: number[] = [];
  const wList: number[] = [];
  for (let o = 0; o < m; o++) {
    start[o] = idxList.length;
    const a = o / s;
    const b = (o + 1) / s;
    const i0 = Math.max(0, Math.floor(a));
    const i1 = Math.min(n - 1, Math.ceil(b) - 1);
    let sum = 0;
    const first = idxList.length;
    for (let i = i0; i <= i1; i++) {
      const ov = Math.min(i + 1, b) - Math.max(i, a);
      if (ov <= 1e-9) continue;
      idxList.push(i);
      wList.push(ov);
      sum += ov;
    }
    for (let k = first; k < wList.length; k++) wList[k] /= sum;
  }
  start[m] = idxList.length;
  return { start, idx: Int32Array.from(idxList), w: Float32Array.from(wList) };
}

/** Exact area-average resampling of an RGBA8 image to w × h (shrinking), Float32 RGBA output. */
export function areaResampleRGBA(src: RGBAImage, w: number, h: number): Float32Array {
  const W = src.width;
  const H = src.height;
  const d = src.data;
  const hw = areaWeights(W, w);
  const vw = areaWeights(H, h);
  const out = new Float32Array(w * h * 4);
  const row = new Float32Array(w * 4);
  let rowIndex = -1;
  const hres = (y: number): void => {
    if (rowIndex === y) return;
    rowIndex = y;
    const base = y * W * 4;
    for (let o = 0; o < w; o++) {
      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;
      for (let k = hw.start[o]; k < hw.start[o + 1]; k++) {
        const j = base + 4 * hw.idx[k];
        const wt = hw.w[k];
        r += wt * d[j];
        g += wt * d[j + 1];
        b += wt * d[j + 2];
        a += wt * d[j + 3];
      }
      row[4 * o] = r;
      row[4 * o + 1] = g;
      row[4 * o + 2] = b;
      row[4 * o + 3] = a;
    }
  };
  for (let oy = 0; oy < h; oy++) {
    const ob = oy * w * 4;
    for (let k = vw.start[oy]; k < vw.start[oy + 1]; k++) {
      hres(vw.idx[k]);
      const wt = vw.w[k];
      for (let i = 0; i < w * 4; i++) out[ob + i] += wt * row[i];
    }
  }
  return out;
}

/**
 * Undistort a RADIAL image to a pinhole camera (k1 = k2 = 0) whose long side is ≤ maxSize.
 * `cam` describes `src` (if its size differs from the image, it is rescaled first with the
 * §2 rule). Returns the undistorted RGBA image and its intrinsics.
 */
export function undistortImage(
  src: RGBAImage,
  cam: CameraIntrinsics,
  maxSize: number,
  opts: UndistortOptions = {},
): { image: RGBAImage; cam: CameraIntrinsics } {
  const W = src.width;
  const H = src.height;
  if (!(W > 0 && H > 0) || src.data.length < W * H * 4) throw new Error('undistortImage: invalid source image');
  const camSrc = cam.width === W && cam.height === H ? cam : scaleCamera(cam, W, H);
  const { width: w, height: h } = undistortedSize(W, H, maxSize);
  const outCam = scaleCamera(camSrc, w, h);
  outCam.k1 = 0;
  outCam.k2 = 0;
  const k1 = camSrc.k1;
  const k2 = camSrc.k2;
  const out = new Uint8ClampedArray(w * h * 4);
  const antialias = opts.antialias ?? true;
  const shrink = w !== W || h !== H;

  // Base image to sample from (source, or its box-filtered copy at the output scale).
  let baseF: Float32Array | null = null;
  let baseW = W;
  let baseH = H;
  if (shrink && antialias) {
    baseF = areaResampleRGBA(src, w, h);
    baseW = w;
    baseH = h;
  }

  const isotropic = Math.abs(w / W - h / H) < 1e-12;
  if (k1 === 0 && k2 === 0) {
    if (!shrink) {
      out.set(src.data.subarray(0, W * H * 4));
      return { image: { width: w, height: h, data: out }, cam: outCam };
    }
    // the box-filtered image IS the pinhole output when the scale is the same on both axes
    if (baseF && isotropic) {
      for (let i = 0; i < out.length; i++) out[i] = Math.round(baseF[i]);
      return { image: { width: w, height: h, data: out }, cam: outCam };
    }
  }

  // Output pixel → normalised ray (output pinhole camera) → distorted source pixel (camSrc)
  // → base pixel (§2 rule per axis: the box-filtered base may be scaled anisotropically).
  const fo = outCam.f;
  const cxo = outCam.cx;
  const cyo = outCam.cy;
  const fs = camSrc.f;
  const cxs = camSrc.cx;
  const cys = camSrc.cy;
  const bsx = baseW / W;
  const bsy = baseH / H;
  const foldSq = radialFoldRadiusSq(k1, k2);
  const maxUs = W - 0.5;
  const maxVs = H - 0.5;
  const sd = src.data;

  for (let y = 0; y < h; y++) {
    const yn = (y - cyo) / fo;
    for (let x = 0; x < w; x++) {
      const xn = (x - cxo) / fo;
      const r2 = xn * xn + yn * yn;
      const o = 4 * (y * w + x);
      if (r2 >= foldSq) continue; // stays (0,0,0,0)
      const dd = 1 + r2 * (k1 + k2 * r2);
      const us = fs * dd * xn + cxs;
      const vs = fs * dd * yn + cys;
      if (!(us >= -0.5 && us <= maxUs && vs >= -0.5 && vs <= maxVs)) continue;
      const u = (us + 0.5) * bsx - 0.5;
      const v = (vs + 0.5) * bsy - 0.5;
      const fx = Math.floor(u);
      const fy = Math.floor(v);
      const ax = u - fx;
      const ay = v - fy;
      const x0 = fx < 0 ? 0 : fx;
      const y0 = fy < 0 ? 0 : fy;
      const x1 = fx + 1 > baseW - 1 ? baseW - 1 : fx + 1;
      const y1 = fy + 1 > baseH - 1 ? baseH - 1 : fy + 1;
      const w00 = (1 - ax) * (1 - ay);
      const w10 = ax * (1 - ay);
      const w01 = (1 - ax) * ay;
      const w11 = ax * ay;
      const i00 = 4 * (y0 * baseW + x0);
      const i10 = 4 * (y0 * baseW + x1);
      const i01 = 4 * (y1 * baseW + x0);
      const i11 = 4 * (y1 * baseW + x1);
      if (baseF) {
        for (let c = 0; c < 4; c++) {
          out[o + c] = Math.round(w00 * baseF[i00 + c] + w10 * baseF[i10 + c] + w01 * baseF[i01 + c] + w11 * baseF[i11 + c]);
        }
      } else {
        // no antialiasing (or no shrink): sample the source directly
        for (let c = 0; c < 4; c++) {
          out[o + c] = Math.round(w00 * sd[i00 + c] + w10 * sd[i10 + c] + w01 * sd[i01 + c] + w11 * sd[i11 + c]);
        }
      }
    }
  }
  return { image: { width: w, height: h, data: out }, cam: outCam };
}
