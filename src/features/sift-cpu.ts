/**
 * Reference-quality CPU SIFT (Lowe 2004, VLFeat/COLMAP-style) producing RootSIFT
 * descriptors. Used as the fallback when WebGPU is unavailable and as the
 * reference the GPU implementation is validated against.
 *
 * Pipeline: area-downscale to maxImageSize → (optional 2x upsample) → Gaussian
 * scale space (S+3 levels per octave, incremental separable blurs) → DoG →
 * 3x3x3 extrema → Taylor refinement in (x, y, s) with position moves → contrast
 * and edge rejection → spatially distributed preselection → orientation
 * histograms (36 bins) → 4x4x8 descriptors → RootSIFT quantisation → final
 * distributed selection → keypoints mapped to the full-resolution frame.
 *
 * See sift-common.ts for the exact scale-space / descriptor conventions.
 */

import type { FeatureSet, GrayImage } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';
import { selectFeatureIndices } from './distribute';
import { blurFloat32, gaussianHalfKernel, grayToFloat32, resizeFloat32 } from './image';
import { createSimdKernels, simdAvailable } from './simd';
import type { SimdKernels } from './simd';
import { finalizeRootSift, octaveToFull, planSift, resolveSiftOptions } from './sift-common';
import type { SiftOptions, SiftPlan } from './sift-common';

const TWO_PI = Math.PI * 2;
const HALF_PI = Math.PI / 2;
const ORI_BINS = 36;
const MAX_REFINE_ITERS = 5;
const OFFSET_LIMIT = 0.6;

// ---------------------------------------------------------------------------
// Arena: all image planes of one extraction live in one buffer, reused across
// calls (extraction is synchronous, so a module-level arena is safe). When WASM
// SIMD is available the arena is the kernels' linear memory, so the blur passes
// run in SIMD on the same planes the JS stages read.
// ---------------------------------------------------------------------------

/** Bytes reserved for a plane of n floats: 16-byte aligned plus 16 bytes of slack. */
function planeBytes(n: number): number {
  return ((n * 4 + 15) & ~15) + 16;
}

class Arena {
  private kernels: SimdKernels | null | undefined;
  private jsBuffer: ArrayBuffer | null = null;
  private buffer: ArrayBuffer | null = null;
  private top = 0;
  private cap = 0;
  simd: SimdKernels | null = null;

  begin(totalBytes: number): void {
    this.simd = null;
    if (simdAvailable()) {
      if (this.kernels === undefined) this.kernels = createSimdKernels(totalBytes);
      if (this.kernels) {
        try {
          this.kernels.ensure(totalBytes);
          this.simd = this.kernels;
        } catch {
          // Memory growth refused (e.g. memory pressure): use plain JS buffers.
          this.kernels = null;
        }
      }
    }
    if (this.simd) {
      this.buffer = this.simd.memory.buffer;
    } else {
      if (!this.jsBuffer || this.jsBuffer.byteLength < totalBytes) this.jsBuffer = new ArrayBuffer(totalBytes);
      this.buffer = this.jsBuffer;
    }
    this.top = 0;
    this.cap = totalBytes;
  }

  f32(n: number): Float32Array {
    const off = this.top;
    this.top += planeBytes(n);
    if (this.top > this.cap || !this.buffer) throw new Error('sift-cpu: arena overflow');
    return new Float32Array(this.buffer, off, n);
  }

  release(): void {
    this.kernels = undefined;
    this.jsBuffer = null;
    this.buffer = null;
    this.simd = null;
  }
}

const arena = new Arena();

/** Frees the scratch memory kept between extractSiftCPU calls. */
export function releaseSiftCPUMemory(): void {
  arena.release();
}

/** Separable Gaussian blur of an arena plane (WASM SIMD when available). */
function blurPlane(
  src: Float32Array,
  dst: Float32Array,
  w: number,
  h: number,
  sigma: number,
  tmp: Float32Array,
  kernel: Float32Array,
  pad: Float32Array,
): void {
  const k = arena.simd;
  if (!k || !(sigma > 0)) {
    blurFloat32(src, w, h, sigma, dst, tmp);
    return;
  }
  const kern = gaussianHalfKernel(sigma);
  const r = kern.length - 1;
  if (r + 1 > kernel.length || w + 2 * r + 8 > pad.length) throw new Error('sift-cpu: blur scratch too small');
  kernel.set(kern);
  k.blurH(src.byteOffset, tmp.byteOffset, w, h, kernel.byteOffset, r, pad.byteOffset);
  k.blurV(tmp.byteOffset, dst.byteOffset, w, h, kernel.byteOffset, r);
}

/** Stage timings (ms) of the most recent extractSiftCPU call (diagnostics). */
export const lastSiftCPUTimings: Record<string, number> = {};

// ---------------------------------------------------------------------------
// Growable keypoint list (structure of arrays)
// ---------------------------------------------------------------------------

class KeypointList {
  n = 0;
  x = new Float64Array(1024);
  y = new Float64Array(1024);
  s = new Float64Array(1024); // continuous DoG level
  sigma = new Float64Array(1024); // octave pixels
  score = new Float32Array(1024);
  octIdx = new Int32Array(1024); // index into plan.octaves
  level = new Int32Array(1024); // Gaussian level used for orientation / descriptor

  push(x: number, y: number, s: number, sigma: number, score: number, octIdx: number, level: number): void {
    if (this.n === this.x.length) this.grow();
    const i = this.n++;
    this.x[i] = x;
    this.y[i] = y;
    this.s[i] = s;
    this.sigma[i] = sigma;
    this.score[i] = score;
    this.octIdx[i] = octIdx;
    this.level[i] = level;
  }

  private grow(): void {
    const m = this.x.length * 2;
    const g64 = (a: Float64Array) => {
      const b = new Float64Array(m);
      b.set(a);
      return b;
    };
    this.x = g64(this.x);
    this.y = g64(this.y);
    this.s = g64(this.s);
    this.sigma = g64(this.sigma);
    const sc = new Float32Array(m);
    sc.set(this.score);
    this.score = sc;
    const oi = new Int32Array(m);
    oi.set(this.octIdx);
    this.octIdx = oi;
    const lv = new Int32Array(m);
    lv.set(this.level);
    this.level = lv;
  }
}

// ---------------------------------------------------------------------------
// Scale-space helpers
// ---------------------------------------------------------------------------

/** 2x upsample: up(2i) = in(i), up(2i+1) = (in(i) + in(i+1)) / 2, edge clamped. */
function upsample2x(src: Float32Array, w: number, h: number, dst: Float32Array, tmp: Float32Array): void {
  const w2 = 2 * w;
  for (let y = 0; y < h; y++) {
    const so = y * w;
    const to = y * w2;
    for (let x = 0; x < w; x++) {
      const a = src[so + x];
      const b = x + 1 < w ? src[so + x + 1] : a;
      tmp[to + 2 * x] = a;
      tmp[to + 2 * x + 1] = 0.5 * (a + b);
    }
  }
  for (let y = 0; y < h; y++) {
    const ra = y * w2;
    const rb = (y + 1 < h ? y + 1 : y) * w2;
    const d0 = 2 * y * w2;
    const d1 = d0 + w2;
    for (let x = 0; x < w2; x++) {
      const a = tmp[ra + x];
      dst[d0 + x] = a;
      dst[d1 + x] = 0.5 * (a + tmp[rb + x]);
    }
  }
}

/** D = B - A. */
function subtractInto(D: Float32Array, B: Float32Array, A: Float32Array, n: number): void {
  for (let j = 0; j < n; j++) D[j] = B[j] - A[j];
}

/** Decimation next(i, j) = cur(2i, 2j). */
function decimate(src: Float32Array, w: number, dst: Float32Array, dw: number, dh: number): void {
  for (let y = 0; y < dh; y++) {
    const so = 2 * y * w;
    const d = y * dw;
    for (let x = 0; x < dw; x++) dst[d + x] = src[so + 2 * x];
  }
}

/** atan2 approximation, |error| < 1e-5 rad (Abramowitz & Stegun 4.4.49). Returns [0, 2π). */
function fastAtan2Pos(y: number, x: number): number {
  const ax = x < 0 ? -x : x;
  const ay = y < 0 ? -y : y;
  if (ax === 0 && ay === 0) return 0;
  let r: number;
  if (ay > ax) {
    const z = ax / ay;
    const z2 = z * z;
    r = HALF_PI - z * (0.999866 + z2 * (-0.3302995 + z2 * (0.180141 + z2 * (-0.085133 + z2 * 0.0208351))));
  } else {
    const z = ay / ax;
    const z2 = z * z;
    r = z * (0.999866 + z2 * (-0.3302995 + z2 * (0.180141 + z2 * (-0.085133 + z2 * 0.0208351))));
  }
  if (x < 0) r = Math.PI - r;
  if (y < 0) r = TWO_PI - r;
  return r >= TWO_PI ? 0 : r;
}

/** Gradient magnitude and angle ([0, 2π)) with central differences (one-sided at borders). */
function computeGradient(L: Float32Array, w: number, h: number, mag: Float32Array, ang: Float32Array): void {
  const k = arena.simd;
  if (k && w >= 3 && h >= 3) {
    // SIMD interior, then the border pixels (the kernel's 4-pixel blocks overrun them).
    k.grad(L.byteOffset, mag.byteOffset, ang.byteOffset, w, h);
    gradientRow(L, 0, 0, w, 1, w, mag, ang);
    const last = (h - 1) * w;
    gradientRow(L, last, last - w, last, 1, w, mag, ang);
    for (let y = 1; y < h - 1; y++) {
      const i0 = y * w;
      const gx0 = L[i0 + 1] - L[i0];
      const gy0 = 0.5 * (L[i0 + w] - L[i0 - w]);
      mag[i0] = Math.sqrt(gx0 * gx0 + gy0 * gy0);
      ang[i0] = fastAtan2Pos(gy0, gx0);
      const i1 = i0 + w - 1;
      const gx1 = L[i1] - L[i1 - 1];
      const gy1 = 0.5 * (L[i1 + w] - L[i1 - w]);
      mag[i1] = Math.sqrt(gx1 * gx1 + gy1 * gy1);
      ang[i1] = fastAtan2Pos(gy1, gx1);
    }
    return;
  }
  for (let y = 0; y < h; y++) {
    const o = y * w;
    const up = y > 0 ? o - w : o;
    const dn = y < h - 1 ? o + w : o;
    const fy = y > 0 && y < h - 1 ? 0.5 : 1;
    gradientRow(L, o, up, dn, fy, w, mag, ang);
  }
}

function gradientRow(
  L: Float32Array,
  o: number,
  up: number,
  dn: number,
  fy: number,
  w: number,
  mag: Float32Array,
  ang: Float32Array,
): void {
  // Interior: central differences, no branches on x.
  for (let x = 1; x < w - 1; x++) {
    const i = o + x;
    const gx = 0.5 * (L[i + 1] - L[i - 1]);
    const gy = fy * (L[dn + x] - L[up + x]);
    mag[i] = Math.sqrt(gx * gx + gy * gy);
    ang[i] = fastAtan2Pos(gy, gx);
  }
  // Borders: one-sided differences.
  const gx0 = w > 1 ? L[o + 1] - L[o] : 0;
  const gy0 = fy * (L[dn] - L[up]);
  mag[o] = Math.sqrt(gx0 * gx0 + gy0 * gy0);
  ang[o] = fastAtan2Pos(gy0, gx0);
  if (w > 1) {
    const x = w - 1;
    const gx1 = L[o + x] - L[o + x - 1];
    const gy1 = fy * (L[dn + x] - L[up + x]);
    mag[o + x] = Math.sqrt(gx1 * gx1 + gy1 * gy1);
    ang[o + x] = fastAtan2Pos(gy1, gx1);
  }
}

// ---------------------------------------------------------------------------
// Detection + refinement
// ---------------------------------------------------------------------------

function isMax(Dm: Float32Array, D: Float32Array, Dp: Float32Array, i: number, w: number, v: number): boolean {
  return (
    v > D[i - 1] && v > D[i + 1] && v > D[i - w] && v > D[i + w] &&
    v > D[i - w - 1] && v > D[i - w + 1] && v > D[i + w - 1] && v > D[i + w + 1] &&
    v > Dm[i] && v > Dm[i - 1] && v > Dm[i + 1] && v > Dm[i - w] && v > Dm[i + w] &&
    v > Dm[i - w - 1] && v > Dm[i - w + 1] && v > Dm[i + w - 1] && v > Dm[i + w + 1] &&
    v > Dp[i] && v > Dp[i - 1] && v > Dp[i + 1] && v > Dp[i - w] && v > Dp[i + w] &&
    v > Dp[i - w - 1] && v > Dp[i - w + 1] && v > Dp[i + w - 1] && v > Dp[i + w + 1]
  );
}

function isMin(Dm: Float32Array, D: Float32Array, Dp: Float32Array, i: number, w: number, v: number): boolean {
  return (
    v < D[i - 1] && v < D[i + 1] && v < D[i - w] && v < D[i + w] &&
    v < D[i - w - 1] && v < D[i - w + 1] && v < D[i + w - 1] && v < D[i + w + 1] &&
    v < Dm[i] && v < Dm[i - 1] && v < Dm[i + 1] && v < Dm[i - w] && v < Dm[i + w] &&
    v < Dm[i - w - 1] && v < Dm[i - w + 1] && v < Dm[i + w - 1] && v < Dm[i + w + 1] &&
    v < Dp[i] && v < Dp[i - 1] && v < Dp[i + 1] && v < Dp[i - w] && v < Dp[i + w] &&
    v < Dp[i - w - 1] && v < Dp[i - w + 1] && v < Dp[i + w - 1] && v < Dp[i + w + 1]
  );
}

/** Output of refineExtremum: [x, y, s, value, xi, yi, si]. */
const refined = new Float64Array(7);

/**
 * Quadratic (Taylor) refinement in (x, y, s), moving the integer position by one
 * sample per iteration while an offset exceeds 0.6. Applies contrast and edge
 * rejection. Returns false if rejected.
 */
function refineExtremum(
  dog: Float32Array[],
  w: number,
  h: number,
  S: number,
  x0: number,
  y0: number,
  s0: number,
  peakThreshold: number,
  edgeThreshold: number,
): boolean {
  let x = x0;
  let y = y0;
  let s = s0;
  let converged = false;
  let v = 0;
  let dx = 0, dy = 0, ds = 0, dxx = 0, dyy = 0, dxy = 0;
  let bx = 0, by = 0, bs = 0;
  for (let iter = 0; iter < MAX_REFINE_ITERS; iter++) {
    const D = dog[s];
    const Dm = dog[s - 1];
    const Dp = dog[s + 1];
    const i = y * w + x;
    v = D[i];
    dx = 0.5 * (D[i + 1] - D[i - 1]);
    dy = 0.5 * (D[i + w] - D[i - w]);
    ds = 0.5 * (Dp[i] - Dm[i]);
    dxx = D[i + 1] + D[i - 1] - 2 * v;
    dyy = D[i + w] + D[i - w] - 2 * v;
    const dss = Dp[i] + Dm[i] - 2 * v;
    dxy = 0.25 * (D[i + w + 1] - D[i + w - 1] - D[i - w + 1] + D[i - w - 1]);
    const dxs = 0.25 * (Dp[i + 1] - Dp[i - 1] - Dm[i + 1] + Dm[i - 1]);
    const dys = 0.25 * (Dp[i + w] - Dp[i - w] - Dm[i + w] + Dm[i - w]);
    // Solve H b = -g (H symmetric) by Cramer's rule.
    const c00 = dyy * dss - dys * dys;
    const c01 = dxs * dys - dxy * dss;
    const c02 = dxy * dys - dxs * dyy;
    const det = dxx * c00 + dxy * c01 + dxs * c02;
    if (!(Math.abs(det) > 1e-30)) return false;
    const c11 = dxx * dss - dxs * dxs;
    const c12 = dxs * dxy - dxx * dys;
    const c22 = dxx * dyy - dxy * dxy;
    const id = -1 / det;
    bx = (c00 * dx + c01 * dy + c02 * ds) * id;
    by = (c01 * dx + c11 * dy + c12 * ds) * id;
    bs = (c02 * dx + c12 * dy + c22 * ds) * id;
    if (Math.abs(bx) <= OFFSET_LIMIT && Math.abs(by) <= OFFSET_LIMIT && Math.abs(bs) <= OFFSET_LIMIT) {
      converged = true;
      break;
    }
    if (!(Number.isFinite(bx) && Number.isFinite(by) && Number.isFinite(bs))) return false;
    const mx = bx > OFFSET_LIMIT && x < w - 2 ? 1 : bx < -OFFSET_LIMIT && x > 1 ? -1 : 0;
    const my = by > OFFSET_LIMIT && y < h - 2 ? 1 : by < -OFFSET_LIMIT && y > 1 ? -1 : 0;
    const ms = bs > OFFSET_LIMIT && s < S ? 1 : bs < -OFFSET_LIMIT && s > 1 ? -1 : 0;
    if (mx === 0 && my === 0 && ms === 0) break;
    x += mx;
    y += my;
    s += ms;
  }
  if (!converged) return false;
  const value = v + 0.5 * (dx * bx + dy * by + ds * bs);
  if (!(Math.abs(value) >= peakThreshold)) return false;
  const det2 = dxx * dyy - dxy * dxy;
  const tr = dxx + dyy;
  const r1 = edgeThreshold + 1;
  if (!(det2 > 0) || tr * tr * edgeThreshold >= r1 * r1 * det2) return false;
  refined[0] = x + bx;
  refined[1] = y + by;
  refined[2] = s + bs;
  refined[3] = value;
  refined[4] = x;
  refined[5] = y;
  refined[6] = s;
  return true;
}

function detectOctave(
  dog: Float32Array[],
  w: number,
  h: number,
  S: number,
  sigma: number,
  peakThreshold: number,
  edgeThreshold: number,
  octIdx: number,
  out: KeypointList,
): void {
  const pre = 0.8 * peakThreshold;
  const seen = new Set<number>();
  for (let s = 1; s <= S; s++) {
    const D = dog[s];
    const Dm = dog[s - 1];
    const Dp = dog[s + 1];
    for (let y = 1; y < h - 1; y++) {
      const ro = y * w;
      for (let x = 1; x < w - 1; x++) {
        const i = ro + x;
        const v = D[i];
        if (v > pre) {
          if (!isMax(Dm, D, Dp, i, w, v)) continue;
        } else if (v < -pre) {
          if (!isMin(Dm, D, Dp, i, w, v)) continue;
        } else continue;
        if (!refineExtremum(dog, w, h, S, x, y, s, peakThreshold, edgeThreshold)) continue;
        const key = (refined[6] * h + refined[5]) * w + refined[4];
        if (seen.has(key)) continue;
        seen.add(key);
        const xr = refined[0];
        const yr = refined[1];
        if (xr < 0 || xr > w - 1 || yr < 0 || yr > h - 1) continue;
        const sr = refined[2];
        let lv = Math.round(sr);
        if (lv < 1) lv = 1;
        else if (lv > S) lv = S;
        out.push(xr, yr, sr, sigma * 2 ** (sr / S), Math.abs(refined[3]), octIdx, lv);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Orientation assignment
// ---------------------------------------------------------------------------

const oriHist = new Float64Array(ORI_BINS);
const oriTmp = new Float64Array(ORI_BINS);
let wxScratch = new Float64Array(256);
let wyScratch = new Float64Array(256);
const peakAngle = new Float64Array(ORI_BINS);
const peakValue = new Float64Array(ORI_BINS);

function ensureWeightScratch(n: number): void {
  if (wxScratch.length < n) {
    wxScratch = new Float64Array(n * 2);
    wyScratch = new Float64Array(n * 2);
  }
}

/**
 * Dominant gradient orientations around (x, y) at scale sigma (octave pixels).
 * Writes up to maxOri angles (strongest first) into `out`; returns the count.
 */
function computeOrientations(
  mag: Float32Array,
  ang: Float32Array,
  w: number,
  h: number,
  x: number,
  y: number,
  sigma: number,
  maxOri: number,
  out: Float64Array,
): number {
  const sigmaw = 1.5 * sigma;
  const W = Math.max(1, Math.floor(3 * sigmaw));
  const xi = Math.floor(x + 0.5);
  const yi = Math.floor(y + 0.5);
  if (xi < 0 || xi >= w || yi < 0 || yi >= h) return 0;
  const hist = oriHist;
  hist.fill(0);
  const dx0 = Math.max(-W, -xi);
  const dx1 = Math.min(W, w - 1 - xi);
  const dy0 = Math.max(-W, -yi);
  const dy1 = Math.min(W, h - 1 - yi);
  ensureWeightScratch(2 * W + 1);
  const wx = wxScratch;
  const wy = wyScratch;
  const g = -1 / (2 * sigmaw * sigmaw);
  for (let d = dx0; d <= dx1; d++) {
    const t = xi + d - x;
    wx[d - dx0] = Math.exp(t * t * g);
  }
  for (let d = dy0; d <= dy1; d++) {
    const t = yi + d - y;
    wy[d - dy0] = Math.exp(t * t * g);
  }
  const r2max = W * W + 0.6;
  const binScale = ORI_BINS / TWO_PI;
  for (let dy = dy0; dy <= dy1; dy++) {
    const ty = yi + dy - y;
    const ty2 = ty * ty;
    const wyv = wy[dy - dy0];
    const ro = (yi + dy) * w + xi;
    const rem = r2max - ty2;
    if (!(rem > 0)) continue;
    const half = Math.sqrt(rem);
    let a0 = Math.ceil(x - xi - half);
    let a1 = Math.floor(x - xi + half);
    if (a0 < dx0) a0 = dx0;
    if (a1 > dx1) a1 = dx1;
    for (let dx = a0; dx <= a1; dx++) {
      const tx = xi + dx - x;
      if (tx * tx + ty2 >= r2max) continue;
      const i = ro + dx;
      const wgt = mag[i] * wx[dx - dx0] * wyv;
      const fbin = ang[i] * binScale - 0.5;
      const b = Math.floor(fbin);
      const rb = fbin - b;
      const b0 = b < 0 ? b + ORI_BINS : b;
      const b1 = b0 + 1 === ORI_BINS ? 0 : b0 + 1;
      hist[b0] += (1 - rb) * wgt;
      hist[b1] += rb * wgt;
    }
  }
  // Six passes of circular [1/3, 1/3, 1/3] smoothing.
  const tmp = oriTmp;
  for (let it = 0; it < 6; it++) {
    tmp.set(hist);
    for (let i = 0; i < ORI_BINS; i++) {
      const im = i === 0 ? ORI_BINS - 1 : i - 1;
      const ip = i === ORI_BINS - 1 ? 0 : i + 1;
      hist[i] = (tmp[im] + tmp[i] + tmp[ip]) / 3;
    }
  }
  let maxh = 0;
  for (let i = 0; i < ORI_BINS; i++) if (hist[i] > maxh) maxh = hist[i];
  if (!(maxh > 0)) return 0;
  const thr = 0.8 * maxh;
  let np = 0;
  for (let i = 0; i < ORI_BINS; i++) {
    const h0 = hist[i];
    if (h0 < thr) continue;
    const hm = hist[i === 0 ? ORI_BINS - 1 : i - 1];
    const hp = hist[i === ORI_BINS - 1 ? 0 : i + 1];
    if (!(h0 > hm && h0 > hp)) continue;
    const den = hp + hm - 2 * h0;
    const di = den < 0 ? (-0.5 * (hp - hm)) / den : 0;
    let th = (TWO_PI * (i + di + 0.5)) / ORI_BINS;
    if (th < 0) th += TWO_PI;
    else if (th >= TWO_PI) th -= TWO_PI;
    // Insertion into a list sorted by descending peak value.
    let k = np++;
    while (k > 0 && peakValue[k - 1] < h0) {
      peakValue[k] = peakValue[k - 1];
      peakAngle[k] = peakAngle[k - 1];
      k--;
    }
    peakValue[k] = h0;
    peakAngle[k] = th;
  }
  const n = Math.min(np, maxOri);
  for (let k = 0; k < n; k++) out[k] = peakAngle[k];
  return n;
}

// ---------------------------------------------------------------------------
// Descriptor
// ---------------------------------------------------------------------------

const descHist = new Float32Array(DESCRIPTOR_SIZE);
/** Padded accumulation histogram: spatial bins -1..4 (6 x 6), orientation bins 0..8 (9). */
const PAD_X = 9;
const PAD_Y = 6 * 9;
const PAD_SIZE = 6 * 6 * 9;
const padHist = new Float64Array(PAD_SIZE);
/** Max window width / rows handled by the WASM descriptor kernel (larger windows use JS). */
const DESC_MAX = 1024;

interface DescScratch {
  /** f32 column weights (arena). */
  wx: Float32Array;
  /** DESC_MAX row records of 8 x 32-bit words (arena), as f32 and i32 views. */
  rows: Float32Array;
  rowsI: Int32Array;
  /** Padded f32 histogram (arena). */
  P: Float32Array;
}

/** Fold the padded 6x6x9 histogram into the 4x4x8 layout (orientation bin 8 wraps to 0). */
function foldPadded(P: Float32Array | Float64Array, hist: Float32Array): void {
  for (let by = 0; by < 4; by++) {
    for (let bx = 0; bx < 4; bx++) {
      const pb = (by + 1) * PAD_Y + (bx + 1) * PAD_X;
      const ob = (by * 4 + bx) * 8;
      hist[ob] = P[pb] + P[pb + 8];
      for (let t = 1; t < 8; t++) hist[ob + t] = P[pb + t];
    }
  }
}

/** Interval of t with |c t + d| < lim, intersected into range[0..1]. */
function slab(c: number, d: number, lim: number, range: Float64Array): void {
  if (Math.abs(c) < 1e-12) {
    if (!(Math.abs(d) < lim)) {
      range[0] = 1;
      range[1] = 0;
    }
    return;
  }
  let lo = (-lim - d) / c;
  let hi = (lim - d) / c;
  if (lo > hi) {
    const t = lo;
    lo = hi;
    hi = t;
  }
  if (lo > range[0]) range[0] = lo;
  if (hi < range[1]) range[1] = hi;
}

const slabRange = new Float64Array(2);

/**
 * Raw 4x4x8 SIFT histogram (VLFeat-style: magnification 3, Gaussian window with
 * sigma = half the descriptor width, trilinear interpolation) into `hist`.
 */
function computeDescriptor(
  mag: Float32Array,
  ang: Float32Array,
  w: number,
  h: number,
  x: number,
  y: number,
  sigma: number,
  theta: number,
  hist: Float32Array,
  scratch?: DescScratch,
): void {
  hist.fill(0);
  const P = padHist;
  const SBP = 3 * sigma;
  const W = Math.floor(Math.SQRT2 * SBP * 2.5 + 0.5);
  const xi = Math.floor(x + 0.5);
  const yi = Math.floor(y + 0.5);
  const ct = Math.cos(theta);
  const st = Math.sin(theta);
  const isbp = 1 / SBP;
  const dx0 = Math.max(-W, -xi);
  const dx1 = Math.min(W, w - 1 - xi);
  const dy0 = Math.max(-W, -yi);
  const dy1 = Math.min(W, h - 1 - yi);
  if (dx0 > dx1 || dy0 > dy1) return;
  ensureWeightScratch(2 * W + 1);
  const wx = wxScratch;
  const wy = wyScratch;
  const sw = 2 * SBP; // window sigma = NBP/2 bins
  const g = -1 / (2 * sw * sw);
  for (let d = dx0; d <= dx1; d++) {
    const t = xi + d - x;
    wx[d - dx0] = Math.exp(t * t * g);
  }
  for (let d = dy0; d <= dy1; d++) {
    const t = yi + d - y;
    wy[d - dy0] = Math.exp(t * t * g);
  }
  const oScale = 8 / TWO_PI;
  const lim = 2.5 * SBP;
  const xoff = x - xi; // tx = dx - xoff
  const range = slabRange;
  const k = arena.simd;
  if (k && scratch && dx1 - dx0 + 1 <= DESC_MAX && dy1 - dy0 + 1 <= DESC_MAX) {
    // WASM path: JS prepares per-row records, the kernel runs the sample loop.
    const wxv = scratch.wx;
    for (let d = dx0; d <= dx1; d++) wxv[d - dx0] = wx[d - dx0];
    const rf = scratch.rows;
    const ri = scratch.rowsI;
    let nr = 0;
    for (let dy = dy0; dy <= dy1; dy++) {
      const ty = yi + dy - y;
      range[0] = -Infinity;
      range[1] = Infinity;
      slab(ct, st * ty, lim, range);
      slab(-st, ct * ty, lim, range);
      if (!(range[0] <= range[1])) continue;
      let a0 = Math.ceil(range[0] + xoff);
      let a1 = Math.floor(range[1] + xoff);
      if (a0 < dx0) a0 = dx0;
      if (a1 > dx1) a1 = dx1;
      if (a0 > a1) continue;
      const rec = nr * 8;
      const tx0 = a0 - xoff;
      ri[rec] = ((yi + dy) * w + xi + a0) * 4;
      ri[rec + 1] = a1 - a0 + 1;
      ri[rec + 2] = (a0 - dx0) * 4;
      rf[rec + 3] = wy[dy - dy0];
      rf[rec + 4] = (ct * tx0 + st * ty) * isbp;
      rf[rec + 5] = (ct * ty - st * tx0) * isbp;
      nr++;
    }
    const P32 = scratch.P;
    P32.fill(0);
    k.desc(mag.byteOffset, ang.byteOffset, wxv.byteOffset, rf.byteOffset, nr, ct * isbp, -st * isbp, theta, P32.byteOffset);
    foldPadded(P32, hist);
    return;
  }
  P.fill(0);
  for (let dy = dy0; dy <= dy1; dy++) {
    const ty = yi + dy - y;
    // Restrict dx to the rotated support square |nx| < 2.5, |ny| < 2.5.
    range[0] = -Infinity;
    range[1] = Infinity;
    slab(ct, st * ty, lim, range);
    slab(-st, ct * ty, lim, range);
    if (!(range[0] <= range[1])) continue;
    let a0 = Math.ceil(range[0] + xoff);
    let a1 = Math.floor(range[1] + xoff);
    if (a0 < dx0) a0 = dx0;
    if (a1 > dx1) a1 = dx1;
    if (a0 > a1) continue;
    const wyv = wy[dy - dy0];
    const rxo = st * ty;
    const ryo = ct * ty;
    const ro = (yi + dy) * w + xi;
    for (let dx = a0; dx <= a1; dx++) {
      const tx = dx - xoff;
      const nx = (ct * tx + rxo) * isbp;
      const ny = (ryo - st * tx) * isbp;
      if (nx <= -2.5 || nx >= 2.5 || ny <= -2.5 || ny >= 2.5) continue;
      const i = ro + dx;
      const m = mag[i] * wx[dx - dx0] * wyv;
      let a = ang[i] - theta;
      if (a < 0) a += TWO_PI;
      const nt = a * oScale;
      const fx = nx + 1.5;
      const fy = ny + 1.5;
      const bx = Math.floor(fx);
      const by = Math.floor(fy);
      let bt = Math.floor(nt);
      if (bt > 7) bt = 7;
      const rx = fx - bx;
      const ry = fy - by;
      const rt = nt - bt;
      // Padded indices: spatial bin b -> b + 1, orientation bins 0..8 (8 wraps to 0).
      const base = (by + 1) * PAD_Y + (bx + 1) * PAD_X + bt;
      const m0 = m * (1 - ry);
      const m1 = m * ry;
      const m00 = m0 * (1 - rx);
      const m01 = m0 * rx;
      const m10 = m1 * (1 - rx);
      const m11 = m1 * rx;
      const rt0 = 1 - rt;
      P[base] += m00 * rt0;
      P[base + 1] += m00 * rt;
      P[base + PAD_X] += m01 * rt0;
      P[base + PAD_X + 1] += m01 * rt;
      P[base + PAD_Y] += m10 * rt0;
      P[base + PAD_Y + 1] += m10 * rt;
      P[base + PAD_Y + PAD_X] += m11 * rt0;
      P[base + PAD_Y + PAD_X + 1] += m11 * rt;
    }
  }
  foldPadded(P, hist);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

function emptyFeatureSet(width: number, height: number): FeatureSet {
  return {
    count: 0,
    keypoints: new Float32Array(0),
    descriptors: new Uint8Array(0),
    scores: new Float32Array(0),
    width,
    height,
  };
}

/**
 * Extract RootSIFT features on the CPU.
 * @param img gray image (Uint8 0..255 or Float32 0..1)
 * @param opts SIFT options (defaults: see DEFAULT_SIFT_OPTIONS)
 * @param fullWidth width of the full-resolution frame the keypoints are expressed in (default img.width)
 * @param fullHeight height of that frame (default img.height)
 * @returns features sorted by descending score, at most opts.maxFeatures
 */
export function extractSiftCPU(
  img: GrayImage,
  opts?: Partial<SiftOptions>,
  fullWidth?: number,
  fullHeight?: number,
): FeatureSet {
  const T = lastSiftCPUTimings;
  for (const k of Object.keys(T)) delete T[k];
  let tPrev = performance.now();
  const tick = (name: string) => {
    const t = performance.now();
    T[name] = (T[name] ?? 0) + (t - tPrev);
    tPrev = t;
  };

  const o = resolveSiftOptions(opts);
  if (img.data.length < img.width * img.height) throw new Error('extractSiftCPU: image data shorter than width*height');
  const FW = fullWidth ?? img.width;
  const FH = fullHeight ?? img.height;
  if (o.maxFeatures === 0) return emptyFeatureSet(FW, FH);
  const plan: SiftPlan = planSift(img.width, img.height, o, FW, FH);
  if (plan.octaves.length === 0) return emptyFeatureSet(FW, FH);
  const S = o.scalesPerOctave;
  const sigma = o.sigma;

  // --- Arena layout --------------------------------------------------------
  const pw = plan.procWidth;
  const ph = plan.procHeight;
  const oct0 = plan.octaves[0];
  const maxN = oct0.width * oct0.height;
  const incSigma: number[] = [];
  for (let i = 1; i <= S + 2; i++) {
    incSigma[i] = sigma * Math.sqrt(2 ** ((2 * i) / S) - 2 ** ((2 * (i - 1)) / S));
  }
  const b0 = plan.baseBlur;
  const sigmaBase = sigma > b0 ? Math.sqrt(sigma * sigma - b0 * b0) : 0;
  const rMax = Math.max(1, Math.ceil(4 * Math.max(sigmaBase, incSigma[S + 2])));
  {
    let total = planeBytes(pw * ph) + planeBytes(maxN) * (1 + 3 + (S + 2)) + planeBytes(rMax + 1);
    total += planeBytes(DESC_MAX) + planeBytes(DESC_MAX * 8) + planeBytes(PAD_SIZE);
    total += planeBytes(oct0.width + 2 * rMax + 8);
    if (plan.firstOctave < 0) total += planeBytes(2 * pw * ph) + planeBytes(maxN);
    for (const oc of plan.octaves) total += S * planeBytes(oc.width * oc.height);
    arena.begin(total + 64);
  }
  const descRows = arena.f32(DESC_MAX * 8);
  const descScratch: DescScratch = {
    wx: arena.f32(DESC_MAX),
    rows: descRows,
    rowsI: new Int32Array(descRows.buffer, descRows.byteOffset, descRows.length),
    P: arena.f32(PAD_SIZE),
  };
  const kernelBuf = arena.f32(rMax + 1);
  const padBuf = arena.f32(oct0.width + 2 * rMax + 8);
  const tmp = arena.f32(maxN);
  const scratch = [arena.f32(maxN), arena.f32(maxN), arena.f32(maxN)];
  const dog: Float32Array[] = [];
  for (let i = 0; i <= S + 1; i++) dog.push(arena.f32(maxN));

  // --- Processed image ---------------------------------------------------
  const proc = arena.f32(pw * ph);
  if (pw === img.width && ph === img.height) {
    const d = img.data;
    if (d instanceof Float32Array) proc.set(d.subarray(0, pw * ph));
    else for (let i = 0, n = pw * ph; i < n; i++) proc[i] = d[i] * (1 / 255);
  } else {
    resizeFloat32(grayToFloat32(img), img.width, img.height, pw, ph, proc);
  }
  let base: Float32Array;
  if (plan.firstOctave < 0) {
    base = arena.f32(maxN);
    upsample2x(proc, pw, ph, base, arena.f32(2 * pw * ph));
  } else {
    base = proc;
  }
  tick('prepare');

  // --- Scale space + detection, octave by octave ------------------------
  const kps = new KeypointList();
  // Kept Gaussian levels 1..S per octave for orientation / descriptors.
  const kept: Float32Array[][] = [];
  for (let k = 0; k < plan.octaves.length; k++) {
    const { width: w, height: h } = plan.octaves[k];
    const n = w * h;
    const L: Float32Array[] = new Array(S + 3);
    const levels: Float32Array[] = [];
    for (let i = 1; i <= S; i++) levels.push(arena.f32(n));
    L[0] = scratch[0].subarray(0, n);
    for (let i = 1; i <= S; i++) L[i] = levels[i - 1];
    L[S + 1] = scratch[1].subarray(0, n);
    L[S + 2] = scratch[2].subarray(0, n);
    const t = tmp.subarray(0, n);
    if (k === 0) {
      blurPlane(base, L[0], w, h, sigmaBase, t, kernelBuf, padBuf);
    } else {
      const prev = plan.octaves[k - 1];
      decimate(kept[k - 1][S - 1], prev.width, L[0], w, h);
    }
    for (let i = 1; i <= S + 2; i++) blurPlane(L[i - 1], L[i], w, h, incSigma[i], t, kernelBuf, padBuf);
    tick('gaussian');
    const D: Float32Array[] = [];
    for (let i = 0; i <= S + 1; i++) {
      const d = dog[i].subarray(0, n);
      subtractInto(d, L[i + 1], L[i], n);
      D.push(d);
    }
    kept.push(levels);
    tick('dog');
    detectOctave(D, w, h, S, sigma, o.peakThreshold, o.edgeThreshold, k, kps);
    tick('detect');
  }

  // --- Preselection (keypoints) -----------------------------------------
  const nk = kps.n;
  const px = new Float32Array(nk);
  const py = new Float32Array(nk);
  for (let i = 0; i < nk; i++) {
    const f = 2 ** plan.octaves[kps.octIdx[i]].octave;
    px[i] = kps.x[i] * f;
    py[i] = kps.y[i] * f;
  }
  const pre = selectFeatureIndices(px, py, kps.score, nk, pw, ph, o.maxFeatures, o.distribute);
  tick('select');

  // Group the preselected keypoints by (octave, level) so each gradient image is computed once.
  const groupKey = new Int32Array(pre.length);
  const order = new Uint32Array(pre.length);
  for (let m = 0; m < pre.length; m++) {
    const i = pre[m];
    groupKey[m] = kps.octIdx[i] * (S + 1) + kps.level[i];
    order[m] = m;
  }
  // Within a group, visit keypoints in raster order for cache locality of mag/ang.
  order.sort((a, b) => groupKey[a] - groupKey[b] || kps.y[pre[a]] - kps.y[pre[b]] || kps.x[pre[a]] - kps.x[pre[b]] || a - b);

  // --- Orientations + descriptors ---------------------------------------
  const maxOut = pre.length * o.maxOrientations;
  const outKp = new Float32Array(4 * maxOut);
  const outDesc = new Uint8Array(DESCRIPTOR_SIZE * maxOut);
  const outScore = new Float32Array(maxOut);
  const outPx = new Float32Array(maxOut);
  const outPy = new Float32Array(maxOut);
  const angles = new Float64Array(o.maxOrientations);
  // The DoG planes are free now: reuse two of them for gradient magnitude / angle.
  const mag = dog[0];
  const ang = dog[1];
  let nOut = 0;
  let curGroup = -1;
  for (let q = 0; q < order.length; q++) {
    const m = order[q];
    const i = pre[m];
    const k = kps.octIdx[i];
    const { width: w, height: h, octave } = plan.octaves[k];
    if (groupKey[m] !== curGroup) {
      curGroup = groupKey[m];
      tick('describe');
      computeGradient(kept[k][kps.level[i] - 1], w, h, mag, ang);
      tick('gradient');
    }
    const x = kps.x[i];
    const y = kps.y[i];
    const sg = kps.sigma[i];
    const na = computeOrientations(mag, ang, w, h, x, y, sg, o.maxOrientations, angles);
    for (let a = 0; a < na; a++) {
      computeDescriptor(mag, ang, w, h, x, y, sg, angles[a], descHist, descScratch);
      finalizeRootSift(descHist, outDesc, nOut * DESCRIPTOR_SIZE);
      octaveToFull(plan, octave, x, y, sg, outKp, 4 * nOut);
      outKp[4 * nOut + 3] = angles[a];
      outScore[nOut] = kps.score[i];
      outPx[nOut] = px[i];
      outPy[nOut] = py[i];
      nOut++;
    }
  }
  tick('describe');

  // --- Final selection, sorted by descending score ----------------------
  const fin = selectFeatureIndices(outPx, outPy, outScore, nOut, pw, ph, o.maxFeatures, o.distribute);
  const n = fin.length;
  const keypoints = new Float32Array(4 * n);
  const descriptors = new Uint8Array(DESCRIPTOR_SIZE * n);
  const scores = new Float32Array(n);
  for (let m = 0; m < n; m++) {
    const i = fin[m];
    keypoints[4 * m] = outKp[4 * i];
    keypoints[4 * m + 1] = outKp[4 * i + 1];
    keypoints[4 * m + 2] = outKp[4 * i + 2];
    keypoints[4 * m + 3] = outKp[4 * i + 3];
    descriptors.set(outDesc.subarray(i * DESCRIPTOR_SIZE, (i + 1) * DESCRIPTOR_SIZE), m * DESCRIPTOR_SIZE);
    scores[m] = outScore[i];
  }
  tick('finalize');
  T.simd = arena.simd ? 1 : 0;
  T.candidates = nk;
  T.preselected = pre.length;
  T.oriented = nOut;
  return { count: n, keypoints, descriptors, scores, width: FW, height: FH };
}
