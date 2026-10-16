/**
 * Basic single-channel image utilities shared by the CPU SIFT, the GPU feature
 * pipeline (host-side preparation), keyframe selection and tests.
 *
 * Conventions (docs/ARCHITECTURE.md §2):
 *  - Pixel centres at integer coordinates; an image covers [-0.5, W-0.5] x [-0.5, H-0.5].
 *  - Resampling from W_full to W_small uses s = W_small / W_full and
 *    x_small = (x_full + 0.5) * s - 0.5 (per axis).
 *  - GrayImage data is either Uint8 (0..255) or Float32 (0..1). Every function
 *    here accepts both and returns Float32 0..1 data.
 */

import type { GrayImage, RGBAImage } from '../core/types';

/**
 * Returns the pixels of `img` as Float32 in 0..1. If the data already is a
 * Float32Array it is returned AS IS (aliasing the input — do not modify it).
 */
export function grayToFloat32(img: GrayImage): Float32Array {
  const d = img.data;
  if (d instanceof Float32Array) return d;
  const n = img.width * img.height;
  const out = new Float32Array(n);
  const s = 1 / 255;
  for (let i = 0; i < n; i++) out[i] = d[i] * s;
  return out;
}

/** RGBA8 → gray Float32 0..1 using Rec.709 luma weights on the (sRGB-encoded) values. */
export function rgbaToGray(img: RGBAImage): GrayImage {
  const { width, height, data } = img;
  const n = width * height;
  const out = new Float32Array(n);
  const wr = 0.2126 / 255;
  const wg = 0.7152 / 255;
  const wb = 0.0722 / 255;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    out[i] = wr * data[p] + wg * data[p + 1] + wb * data[p + 2];
  }
  return { width, height, data: out };
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/** Sparse 1-D resampling filter: `taps` (index, weight) pairs per output sample. */
interface AxisFilter {
  taps: number;
  index: Int32Array;
  weight: Float32Array;
}

/**
 * Builds the 1-D resampling filter from `srcN` to `dstN` samples using the
 * project's resampling rule. Area averaging (exact box overlap) when shrinking,
 * bilinear (edge-clamped) when enlarging, identity when equal.
 */
function axisFilter(srcN: number, dstN: number): AxisFilter {
  if (srcN === dstN) {
    const index = new Int32Array(dstN);
    const weight = new Float32Array(dstN).fill(1);
    for (let i = 0; i < dstN; i++) index[i] = i;
    return { taps: 1, index, weight };
  }
  const s = dstN / srcN;
  if (s < 1) {
    // Output pixel i covers [i/s, (i+1)/s] in source "edge" coordinates
    // (source pixel j covers [j, j+1]).
    const inv = srcN / dstN;
    const taps = Math.ceil(inv) + 1;
    const index = new Int32Array(dstN * taps);
    const weight = new Float32Array(dstN * taps);
    for (let i = 0; i < dstN; i++) {
      const lo = i * inv;
      const hi = Math.min(srcN, (i + 1) * inv);
      const j0 = Math.floor(lo);
      let t = 0;
      let sum = 0;
      for (let j = j0; j < hi && t < taps; j++, t++) {
        const ov = Math.min(hi, j + 1) - Math.max(lo, j);
        const jj = Math.min(j, srcN - 1);
        index[i * taps + t] = jj;
        weight[i * taps + t] = ov;
        sum += ov;
      }
      for (; t < taps; t++) {
        index[i * taps + t] = Math.min(j0, srcN - 1);
        weight[i * taps + t] = 0;
      }
      const inorm = sum > 0 ? 1 / sum : 0;
      for (let k = 0; k < taps; k++) weight[i * taps + k] *= inorm;
    }
    return { taps, index, weight };
  }
  // Bilinear enlarge: x_src = (x_dst + 0.5) / s - 0.5, clamped to the valid range.
  const taps = 2;
  const index = new Int32Array(dstN * 2);
  const weight = new Float32Array(dstN * 2);
  for (let i = 0; i < dstN; i++) {
    let x = (i + 0.5) / s - 0.5;
    if (x < 0) x = 0;
    if (x > srcN - 1) x = srcN - 1;
    const j0 = Math.floor(x);
    const f = x - j0;
    const j1 = Math.min(j0 + 1, srcN - 1);
    index[2 * i] = j0;
    index[2 * i + 1] = j1;
    weight[2 * i] = 1 - f;
    weight[2 * i + 1] = f;
  }
  return { taps, index, weight };
}

/**
 * Resample a Float32 plane (sw x sh) into `dst` (dw x dh). Separable; area
 * averaging per axis when shrinking, bilinear when enlarging.
 */
export function resizeFloat32(
  src: Float32Array,
  sw: number,
  sh: number,
  dw: number,
  dh: number,
  dst?: Float32Array,
): Float32Array {
  const out = dst ?? new Float32Array(dw * dh);
  if (sw === dw && sh === dh) {
    out.set(src.subarray(0, sw * sh));
    return out;
  }
  const fx = axisFilter(sw, dw);
  const fy = axisFilter(sh, dh);
  // Horizontal pass: sw x sh -> dw x sh.
  const tmp = new Float32Array(dw * sh);
  const tx = fx.taps;
  const ix = fx.index;
  const wx = fx.weight;
  for (let y = 0; y < sh; y++) {
    const ro = y * sw;
    const to = y * dw;
    for (let x = 0; x < dw; x++) {
      let acc = 0;
      const b = x * tx;
      for (let t = 0; t < tx; t++) acc += wx[b + t] * src[ro + ix[b + t]];
      tmp[to + x] = acc;
    }
  }
  // Vertical pass: dw x sh -> dw x dh (row-wise accumulation for locality).
  const ty = fy.taps;
  const iy = fy.index;
  const wy = fy.weight;
  for (let y = 0; y < dh; y++) {
    const oo = y * dw;
    const b = y * ty;
    const w0 = wy[b];
    const r0 = iy[b] * dw;
    for (let x = 0; x < dw; x++) out[oo + x] = w0 * tmp[r0 + x];
    for (let t = 1; t < ty; t++) {
      const w = wy[b + t];
      if (w === 0) continue;
      const r = iy[b + t] * dw;
      for (let x = 0; x < dw; x++) out[oo + x] += w * tmp[r + x];
    }
  }
  return out;
}

/** Area-averaging when shrinking, bilinear when enlarging (per axis). Returns Float32 0..1. */
export function resizeGray(img: GrayImage, w: number, h: number): GrayImage {
  if (!(w >= 1 && h >= 1)) throw new Error(`resizeGray: invalid size ${w}x${h}`);
  const src = grayToFloat32(img);
  const data = resizeFloat32(src, img.width, img.height, w, h);
  return { width: w, height: h, data };
}

// ---------------------------------------------------------------------------
// Gaussian blur
// ---------------------------------------------------------------------------

/** Normalised half-kernel k[0..r] of a Gaussian with radius r = ceil(4 sigma). */
export function gaussianHalfKernel(sigma: number): Float32Array {
  const r = Math.max(1, Math.ceil(4 * sigma));
  const k = new Float32Array(r + 1);
  const inv = -1 / (2 * sigma * sigma);
  let sum = 0;
  for (let i = 0; i <= r; i++) {
    const v = Math.exp(i * i * inv);
    k[i] = v;
    sum += i === 0 ? v : 2 * v;
  }
  for (let i = 0; i <= r; i++) k[i] /= sum;
  return k;
}

let padScratch = new Float32Array(0);

/**
 * Separable Gaussian blur of a Float32 plane with replicated borders.
 * `dst` may NOT alias `src`; `tmp` (w*h) is scratch (allocated if omitted).
 * Kernel radius = ceil(4 sigma). sigma <= 0 copies.
 */
export function blurFloat32(
  src: Float32Array,
  w: number,
  h: number,
  sigma: number,
  dst: Float32Array,
  tmp?: Float32Array,
): Float32Array {
  const n = w * h;
  if (!(sigma > 0)) {
    dst.set(src.subarray(0, n));
    return dst;
  }
  const k = gaussianHalfKernel(sigma);
  const r = k.length - 1;
  const t = tmp ?? new Float32Array(n);
  const k0 = k[0];

  // Horizontal pass src -> t using a padded row buffer.
  const pw = w + 2 * r;
  if (padScratch.length < pw) padScratch = new Float32Array(pw);
  const pad = padScratch;
  for (let y = 0; y < h; y++) {
    const ro = y * w;
    const first = src[ro];
    const last = src[ro + w - 1];
    for (let i = 0; i < r; i++) {
      pad[i] = first;
      pad[r + w + i] = last;
    }
    for (let x = 0; x < w; x++) pad[r + x] = src[ro + x];
    if (r <= 4) {
      for (let x = 0; x < w; x++) {
        const c = x + r;
        let acc = k0 * pad[c];
        for (let j = 1; j <= r; j++) acc += k[j] * (pad[c - j] + pad[c + j]);
        t[ro + x] = acc;
      }
    } else {
      for (let x = 0; x < w; x++) {
        const c = x + r;
        let acc = k0 * pad[c];
        let j = 1;
        for (; j + 1 <= r; j += 2) {
          acc += k[j] * (pad[c - j] + pad[c + j]) + k[j + 1] * (pad[c - j - 1] + pad[c + j + 1]);
        }
        for (; j <= r; j++) acc += k[j] * (pad[c - j] + pad[c + j]);
        t[ro + x] = acc;
      }
    }
  }

  // Vertical pass t -> dst, accumulated row-wise (contiguous inner loops).
  const hm1 = h - 1;
  for (let y = 0; y < h; y++) {
    const oo = y * w;
    const co = oo;
    for (let x = 0; x < w; x++) dst[oo + x] = k0 * t[co + x];
    for (let j = 1; j <= r; j++) {
      const kj = k[j];
      const ya = (y - j < 0 ? 0 : y - j) * w;
      const yb = (y + j > hm1 ? hm1 : y + j) * w;
      for (let x = 0; x < w; x++) dst[oo + x] += kj * (t[ya + x] + t[yb + x]);
    }
  }
  return dst;
}

/** Gaussian blur (radius ceil(4 sigma), replicated borders). Returns Float32 0..1. */
export function gaussianBlur(img: GrayImage, sigma: number): GrayImage {
  const src = grayToFloat32(img);
  const out = new Float32Array(img.width * img.height);
  blurFloat32(src, img.width, img.height, sigma, out);
  return { width: img.width, height: img.height, data: out };
}

// ---------------------------------------------------------------------------
// Sharpness
// ---------------------------------------------------------------------------

/**
 * Variance of the 4-neighbour Laplacian, computed on a copy whose long side is
 * at most 640 px (area-downscaled), on the 0..255 intensity scale (the usual
 * OpenCV "Laplacian variance" convention; typical sharp phone frames give
 * several hundred, motion-blurred frames tens). Border pixels are excluded.
 */
export function sharpness(img: GrayImage): number {
  let w = img.width;
  let h = img.height;
  let data: Float32Array;
  const longSide = Math.max(w, h);
  if (longSide > 640) {
    const s = 640 / longSide;
    const nw = Math.max(1, Math.round(w * s));
    const nh = Math.max(1, Math.round(h * s));
    data = resizeFloat32(grayToFloat32(img), w, h, nw, nh);
    w = nw;
    h = nh;
  } else {
    data = grayToFloat32(img);
  }
  if (w < 3 || h < 3) return 0;
  let sum = 0;
  let sum2 = 0;
  for (let y = 1; y < h - 1; y++) {
    const o = y * w;
    for (let x = 1; x < w - 1; x++) {
      const i = o + x;
      const l = (data[i - 1] + data[i + 1] + data[i - w] + data[i + w] - 4 * data[i]) * 255;
      sum += l;
      sum2 += l * l;
    }
  }
  const n = (w - 2) * (h - 2);
  const m = sum / n;
  return Math.max(0, sum2 / n - m * m);
}

// ---------------------------------------------------------------------------
// Rotation
// ---------------------------------------------------------------------------

/**
 * Rotate an RGBA image CLOCKWISE (as displayed, y down) by `quarterTurns` x 90°.
 * Negative values rotate counter-clockwise. For odd turns width/height swap.
 * Clockwise 90°: dst(x', y') = src(x = y', y = H - 1 - x').
 * The output data has the same array type as the input.
 */
export function rotateRGBA90(img: RGBAImage, quarterTurns: number): RGBAImage {
  const k = (((Math.round(quarterTurns) % 4) + 4) % 4) as 0 | 1 | 2 | 3;
  const { width: W, height: H, data } = img;
  const n = W * H;
  const Ctor = data instanceof Uint8ClampedArray ? Uint8ClampedArray : Uint8Array;
  const out = new Ctor(n * 4);
  if (k === 0) {
    out.set(data.subarray(0, n * 4));
    return { width: W, height: H, data: out };
  }
  // 32-bit views (copy the source if it is not 4-byte aligned).
  const srcBytes = data.byteOffset % 4 === 0 ? data : data.slice(0, n * 4);
  const s = new Uint32Array(srcBytes.buffer, srcBytes.byteOffset, n);
  const d = new Uint32Array(out.buffer, 0, n);
  if (k === 2) {
    for (let i = 0; i < n; i++) d[i] = s[n - 1 - i];
    return { width: W, height: H, data: out };
  }
  const dw = H; // output width for odd turns
  const dh = W;
  if (k === 1) {
    // dst(x', y') = src(y', H-1-x')  <=>  src(x, y) -> dst(H-1-y, x)
    for (let y = 0; y < H; y++) {
      const so = y * W;
      const dx = H - 1 - y;
      for (let x = 0; x < W; x++) d[x * dw + dx] = s[so + x];
    }
  } else {
    // counter-clockwise: dst(x', y') = src(W-1-y', x')  <=>  src(x, y) -> dst(y, W-1-x)
    for (let y = 0; y < H; y++) {
      const so = y * W;
      for (let x = 0; x < W; x++) d[(W - 1 - x) * dw + y] = s[so + x];
    }
  }
  return { width: dw, height: dh, data: out };
}
