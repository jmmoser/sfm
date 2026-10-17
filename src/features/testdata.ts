/**
 * Procedural test imagery for feature extraction / matching tests (CPU and GPU).
 * No Node-only APIs: usable from both the Node unit tests and the browser GPU tests.
 *
 * makeTexturedImage renders a rich synthetic texture (multi-scale value noise,
 * Gaussian blobs, anti-aliased rectangles (edges/corners), rings and text-like
 * strokes). makeWarpedPair renders a "world" texture at 2x resolution and produces
 * two views related by a known homography (with proper anti-aliasing through
 * supersampled warping), optionally with photometric changes and noise.
 *
 * Homographies map pixel coordinates (pixel centres at integers) of image A to
 * image B: x_b ~ H x_a, row-major Float64Array(9).
 */

import type { FeatureSet, GrayImage, Mat3, MatchPairs } from '../core/types';

// ---------------------------------------------------------------------------
// Deterministic RNG (local, so tests do not depend on src/math)
// ---------------------------------------------------------------------------

export class TestRng {
  private s: number;
  constructor(seed = 1) {
    this.s = seed >>> 0 || 0x9e3779b9;
  }
  /** Uniform [0, 1). mulberry32. */
  next(): number {
    let t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  uniform(a: number, b: number): number {
    return a + (b - a) * this.next();
  }
  int(n: number): number {
    return Math.floor(this.next() * n);
  }
  gaussian(): number {
    let u = 0;
    while (u === 0) u = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * this.next());
  }
}

// ---------------------------------------------------------------------------
// Procedural texture
// ---------------------------------------------------------------------------

export interface TextureOptions {
  /** Multiplier on the number of shapes (blobs, rectangles, strokes, rings). Default 1. */
  density?: number;
  /** Global contrast multiplier around 0.5 (default 1). */
  contrast?: number;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Synthetic textured image, Float32 0..1. Deterministic for a given seed. */
export function makeTexturedImage(width: number, height: number, seed = 1, opts: TextureOptions = {}): GrayImage {
  const rng = new TestRng(seed);
  const density = opts.density ?? 1;
  const img = new Float32Array(width * height).fill(0.5);
  const area = width * height;

  // Multi-scale value noise (smoothstep-interpolated lattice).
  const cells = [128, 64, 32, 16, 8, 4];
  const amps = [0.2, 0.14, 0.1, 0.07, 0.05, 0.035];
  for (let o = 0; o < cells.length; o++) {
    const c = cells[o];
    const gw = Math.ceil(width / c) + 2;
    const gh = Math.ceil(height / c) + 2;
    const lat = new Float32Array(gw * gh);
    for (let i = 0; i < lat.length; i++) lat[i] = rng.uniform(-1, 1);
    const a = amps[o];
    for (let y = 0; y < height; y++) {
      const fy = y / c;
      const iy = Math.floor(fy);
      let ty = fy - iy;
      ty = ty * ty * (3 - 2 * ty);
      for (let x = 0; x < width; x++) {
        const fx = x / c;
        const ix = Math.floor(fx);
        let tx = fx - ix;
        tx = tx * tx * (3 - 2 * tx);
        const p = iy * gw + ix;
        const v0 = lat[p] + (lat[p + 1] - lat[p]) * tx;
        const v1 = lat[p + gw] + (lat[p + gw + 1] - lat[p + gw]) * tx;
        img[y * width + x] += a * (v0 + (v1 - v0) * ty);
      }
    }
  }

  // Anti-aliased rotated rectangles (edges, corners).
  const nRect = Math.round((area / 9000) * density);
  for (let r = 0; r < nRect; r++) {
    const cx = rng.uniform(0, width);
    const cy = rng.uniform(0, height);
    const hx = rng.uniform(3, 45);
    const hy = rng.uniform(3, 45);
    const th = rng.uniform(0, Math.PI);
    const val = rng.uniform(0.05, 0.95);
    const op = rng.uniform(0.5, 1);
    const ct = Math.cos(th);
    const st = Math.sin(th);
    const ext = Math.ceil(Math.hypot(hx, hy)) + 2;
    const x0 = Math.max(0, Math.floor(cx - ext));
    const x1 = Math.min(width - 1, Math.ceil(cx + ext));
    const y0 = Math.max(0, Math.floor(cy - ext));
    const y1 = Math.min(height - 1, Math.ceil(cy + ext));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx;
        const dy = y - cy;
        const u = Math.abs(ct * dx + st * dy) - hx;
        const v = Math.abs(-st * dx + ct * dy) - hy;
        const d = u > 0 && v > 0 ? Math.hypot(u, v) : Math.max(u, v);
        const cov = clamp01(0.5 - d) * op;
        if (cov > 0) {
          const i = y * width + x;
          img[i] += (val - img[i]) * cov;
        }
      }
    }
  }

  // Gaussian blobs of many sizes.
  const nBlob = Math.round((area / 2500) * density);
  for (let b = 0; b < nBlob; b++) {
    const cx = rng.uniform(0, width);
    const cy = rng.uniform(0, height);
    const sg = Math.exp(rng.uniform(Math.log(1.2), Math.log(22)));
    const amp = (rng.next() < 0.5 ? -1 : 1) * rng.uniform(0.12, 0.4);
    const ext = Math.ceil(3 * sg);
    const x0 = Math.max(0, Math.floor(cx - ext));
    const x1 = Math.min(width - 1, Math.ceil(cx + ext));
    const y0 = Math.max(0, Math.floor(cy - ext));
    const y1 = Math.min(height - 1, Math.ceil(cy + ext));
    const g = -1 / (2 * sg * sg);
    for (let y = y0; y <= y1; y++) {
      const dy2 = (y - cy) * (y - cy);
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx;
        img[y * width + x] += amp * Math.exp((dx * dx + dy2) * g);
      }
    }
  }

  // Rings.
  const nRing = Math.round((area / 14000) * density);
  for (let r = 0; r < nRing; r++) {
    const cx = rng.uniform(0, width);
    const cy = rng.uniform(0, height);
    const rad = rng.uniform(3, 28);
    const hw = rng.uniform(0.6, 2.5);
    const val = rng.next() < 0.5 ? rng.uniform(0, 0.25) : rng.uniform(0.75, 1);
    const ext = Math.ceil(rad + hw + 2);
    const x0 = Math.max(0, Math.floor(cx - ext));
    const x1 = Math.min(width - 1, Math.ceil(cx + ext));
    const y0 = Math.max(0, Math.floor(cy - ext));
    const y1 = Math.min(height - 1, Math.ceil(cy + ext));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const d = Math.abs(Math.hypot(x - cx, y - cy) - rad) - hw;
        const cov = clamp01(0.5 - d);
        if (cov > 0) {
          const i = y * width + x;
          img[i] += (val - img[i]) * cov;
        }
      }
    }
  }

  // Text-like strokes: short anti-aliased polylines.
  const nStroke = Math.round((area / 1800) * density);
  const seg = new Float64Array(5 * 4);
  for (let s = 0; s < nStroke; s++) {
    let px = rng.uniform(0, width);
    let py = rng.uniform(0, height);
    let dir = rng.uniform(0, 2 * Math.PI);
    const ns = 1 + rng.int(4);
    const hw = rng.uniform(0.5, 1.6);
    const val = rng.next() < 0.7 ? rng.uniform(0, 0.2) : rng.uniform(0.8, 1);
    let bx0 = px, bx1 = px, by0 = py, by1 = py;
    for (let k = 0; k < ns; k++) {
      const len = rng.uniform(4, 16);
      dir += rng.uniform(-2, 2);
      const qx = px + Math.cos(dir) * len;
      const qy = py + Math.sin(dir) * len;
      seg[4 * k] = px;
      seg[4 * k + 1] = py;
      seg[4 * k + 2] = qx;
      seg[4 * k + 3] = qy;
      px = qx;
      py = qy;
      bx0 = Math.min(bx0, px);
      bx1 = Math.max(bx1, px);
      by0 = Math.min(by0, py);
      by1 = Math.max(by1, py);
    }
    const x0 = Math.max(0, Math.floor(bx0 - hw - 2));
    const x1 = Math.min(width - 1, Math.ceil(bx1 + hw + 2));
    const y0 = Math.max(0, Math.floor(by0 - hw - 2));
    const y1 = Math.min(height - 1, Math.ceil(by1 + hw + 2));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        let dmin = Infinity;
        for (let k = 0; k < ns; k++) {
          const ax = seg[4 * k], ay = seg[4 * k + 1];
          const ux = seg[4 * k + 2] - ax, uy = seg[4 * k + 3] - ay;
          const l2 = ux * ux + uy * uy;
          let t = l2 > 0 ? ((x - ax) * ux + (y - ay) * uy) / l2 : 0;
          t = t < 0 ? 0 : t > 1 ? 1 : t;
          const d = Math.hypot(x - ax - t * ux, y - ay - t * uy);
          if (d < dmin) dmin = d;
        }
        const cov = clamp01(hw + 0.5 - dmin);
        if (cov > 0) {
          const i = y * width + x;
          img[i] += (val - img[i]) * cov;
        }
      }
    }
  }

  const contrast = opts.contrast ?? 1;
  for (let i = 0; i < img.length; i++) img[i] = clamp01(0.5 + (img[i] - 0.5) * contrast);
  return { width, height, data: img };
}

// ---------------------------------------------------------------------------
// Homographies
// ---------------------------------------------------------------------------

export function mat3MulH(a: ArrayLike<number>, b: ArrayLike<number>): Mat3 {
  const o = new Float64Array(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) o[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c];
  return o;
}

export function mat3InvH(m: ArrayLike<number>): Mat3 {
  const [a, b, c, d, e, f, g, h, i] = Array.from(m);
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-300) throw new Error('singular homography');
  const id = 1 / det;
  return new Float64Array([
    A * id, -(b * i - c * h) * id, (b * f - c * e) * id,
    B * id, (a * i - c * g) * id, -(a * f - c * d) * id,
    C * id, -(a * h - b * g) * id, (a * e - b * d) * id,
  ]);
}

/** Apply x' ~ H x. Returns false if the point maps to (or behind) infinity. */
export function applyHomography(H: ArrayLike<number>, x: number, y: number, out: Float64Array | number[]): boolean {
  const w = H[6] * x + H[7] * y + H[8];
  if (!(Math.abs(w) > 1e-12)) return false;
  out[0] = (H[0] * x + H[1] * y + H[2]) / w;
  out[1] = (H[3] * x + H[4] * y + H[5]) / w;
  return w > 0;
}

/** Local area scale sqrt(|det J|) of the homography at (x, y). */
export function homographyLocalScale(H: ArrayLike<number>, x: number, y: number): number {
  const w = H[6] * x + H[7] * y + H[8];
  const u = (H[0] * x + H[1] * y + H[2]) / w;
  const v = (H[3] * x + H[4] * y + H[5]) / w;
  const a = (H[0] - u * H[6]) / w;
  const b = (H[1] - u * H[7]) / w;
  const c = (H[3] - v * H[6]) / w;
  const d = (H[4] - v * H[7]) / w;
  return Math.sqrt(Math.abs(a * d - b * c));
}

/** Resampling-rule scale: x' = (x + 0.5) s - 0.5. */
export function scaleHomography(s: number, sy = s): Mat3 {
  return new Float64Array([s, 0, 0.5 * s - 0.5, 0, sy, 0.5 * sy - 0.5, 0, 0, 1]);
}

/**
 * Similarity about the image centre ((W-1)/2, (H-1)/2): rotate by angleDeg (clockwise
 * as displayed, i.e. from +x towards +y), scale, then translate by (tx, ty).
 */
export function similarityAboutCenter(width: number, height: number, angleDeg: number, scale = 1, tx = 0, ty = 0): Mat3 {
  const cx = (width - 1) / 2;
  const cy = (height - 1) / 2;
  const a = (angleDeg * Math.PI) / 180;
  const c = Math.cos(a) * scale;
  const s = Math.sin(a) * scale;
  return new Float64Array([c, -s, cx - c * cx + s * cy + tx, s, c, cy - s * cx - c * cy + ty, 0, 0, 1]);
}

/**
 * Perspective "tilt": homography induced by rotating a pinhole camera (focal
 * `focal` px, principal point at the image centre) by tiltXDeg about its x axis
 * and tiltYDeg about its y axis, i.e. H = K R K^-1, re-centred so that the image
 * centre stays fixed.
 */
export function tiltHomography(width: number, height: number, tiltXDeg: number, tiltYDeg = 0, focal = 1.2 * Math.max(width, height)): Mat3 {
  const cx = (width - 1) / 2;
  const cy = (height - 1) / 2;
  const ax = (tiltXDeg * Math.PI) / 180;
  const ay = (tiltYDeg * Math.PI) / 180;
  const Rx = new Float64Array([1, 0, 0, 0, Math.cos(ax), -Math.sin(ax), 0, Math.sin(ax), Math.cos(ax)]);
  const Ry = new Float64Array([Math.cos(ay), 0, Math.sin(ay), 0, 1, 0, -Math.sin(ay), 0, Math.cos(ay)]);
  const K = new Float64Array([focal, 0, cx, 0, focal, cy, 0, 0, 1]);
  const H = mat3MulH(mat3MulH(K, mat3MulH(Ry, Rx)), mat3InvH(K));
  // Re-centre: translate so the centre maps to itself.
  const p = [0, 0];
  applyHomography(H, cx, cy, p);
  const T = new Float64Array([1, 0, cx - p[0], 0, 1, cy - p[1], 0, 0, 1]);
  const out = mat3MulH(T, H);
  const s = 1 / out[8];
  for (let i = 0; i < 9; i++) out[i] *= s;
  return out;
}

// ---------------------------------------------------------------------------
// Warping and photometric changes
// ---------------------------------------------------------------------------

export interface WarpOptions {
  /** Supersampling factor per axis (box filter). Default: automatic from the local scale. */
  supersample?: number;
  /** Value outside the source image. Default 0.5. */
  fill?: number;
}

/**
 * Renders dst(x) = src(H^-1 x) with bilinear sampling and ss x ss box supersampling
 * (anti-aliasing for minification). H maps src pixel coords to dst pixel coords.
 */
export function warpHomography(src: GrayImage, H: Mat3, outW: number, outH: number, opts: WarpOptions = {}): GrayImage {
  const Hi = mat3InvH(H);
  const sw = src.width;
  const sh = src.height;
  const sd = src.data;
  const norm = sd instanceof Float32Array ? 1 : 1 / 255;
  const fill = opts.fill ?? 0.5;
  let ss = opts.supersample ?? 0;
  if (!ss) {
    // Max local minification over a few probe points.
    let m = 0;
    for (const [px, py] of [[0.5, 0.5], [0, 0], [1, 0], [0, 1], [1, 1]]) {
      m = Math.max(m, homographyLocalScale(Hi, px * (outW - 1), py * (outH - 1)));
    }
    ss = Math.max(1, Math.min(6, Math.ceil(2 * m - 1e-6)));
  }
  const out = new Float32Array(outW * outH);
  const inv = 1 / (ss * ss);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      let acc = 0;
      for (let j = 0; j < ss; j++) {
        const yy = y + (j + 0.5) / ss - 0.5;
        for (let i = 0; i < ss; i++) {
          const xx = x + (i + 0.5) / ss - 0.5;
          const w = Hi[6] * xx + Hi[7] * yy + Hi[8];
          const u = (Hi[0] * xx + Hi[1] * yy + Hi[2]) / w;
          const v = (Hi[3] * xx + Hi[4] * yy + Hi[5]) / w;
          if (!(w > 0) || u < -0.5 || v < -0.5 || u > sw - 0.5 || v > sh - 0.5) {
            acc += fill;
            continue;
          }
          let x0 = Math.floor(u);
          let y0 = Math.floor(v);
          let fx = u - x0;
          let fy = v - y0;
          if (x0 < 0) {
            x0 = 0;
            fx = 0;
          } else if (x0 >= sw - 1) {
            x0 = sw - 1;
            fx = 0;
          }
          if (y0 < 0) {
            y0 = 0;
            fy = 0;
          } else if (y0 >= sh - 1) {
            y0 = sh - 1;
            fy = 0;
          }
          const p = y0 * sw + x0;
          const x1 = fx > 0 ? 1 : 0;
          const y1 = fy > 0 ? sw : 0;
          const a = sd[p] + (sd[p + x1] - sd[p]) * fx;
          const b = sd[p + y1] + (sd[p + y1 + x1] - sd[p + y1]) * fx;
          acc += (a + (b - a) * fy) * norm;
        }
      }
      out[y * outW + x] = acc * inv;
    }
  }
  return { width: outW, height: outH, data: out };
}

export interface PhotometricOptions {
  /** v' = gain * v + bias, then optional gamma. */
  gain?: number;
  bias?: number;
  gamma?: number;
  /** Additive Gaussian noise sigma (0..1 scale). */
  noiseSigma?: number;
  seed?: number;
}

/** Photometric change + noise, clamped to 0..1. Returns a new Float32 image. */
export function adjustPhotometric(img: GrayImage, o: PhotometricOptions): GrayImage {
  const src = img.data;
  const norm = src instanceof Float32Array ? 1 : 1 / 255;
  const out = new Float32Array(img.width * img.height);
  const gain = o.gain ?? 1;
  const bias = o.bias ?? 0;
  const gamma = o.gamma ?? 1;
  const ns = o.noiseSigma ?? 0;
  const rng = new TestRng(o.seed ?? 12345);
  for (let i = 0; i < out.length; i++) {
    let v = clamp01(gain * src[i] * norm + bias);
    if (gamma !== 1) v = Math.pow(v, gamma);
    if (ns > 0) v += ns * rng.gaussian();
    out[i] = clamp01(v);
  }
  return { width: img.width, height: img.height, data: out };
}

export interface WarpedPair {
  a: GrayImage;
  b: GrayImage;
  /** x_b ~ H x_a. */
  H: Mat3;
}

/**
 * Two views of the same synthetic planar scene. The world texture is rendered at
 * 2x the view size; view A is the world downscaled 2x, view B the world seen through
 * H (relative to A). Both are anti-aliased by supersampling.
 */
export function makeWarpedPair(
  width: number,
  height: number,
  H: Mat3,
  seed = 1,
  photometric?: PhotometricOptions,
  texture?: TextureOptions,
): WarpedPair {
  const world = makeTexturedImage(2 * width, 2 * height, seed, texture);
  const S = scaleHomography(0.5);
  const a = warpHomography(world, S, width, height);
  let b = warpHomography(world, mat3MulH(H, S), width, height);
  if (photometric) b = adjustPhotometric(b, photometric);
  return { a, b, H: new Float64Array(H) };
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

export interface MatchEval {
  total: number;
  correct: number;
  precision: number;
}

/** A match (i, j) is correct when |H x_a(i) - x_b(j)| <= tolPx (in image b pixels). */
export function evaluateMatches(a: FeatureSet, b: FeatureSet, matches: MatchPairs, H: Mat3, tolPx = 2): MatchEval {
  const p = [0, 0];
  const n = matches.length >> 1;
  let correct = 0;
  const t2 = tolPx * tolPx;
  for (let k = 0; k < n; k++) {
    const i = matches[2 * k];
    const j = matches[2 * k + 1];
    if (!applyHomography(H, a.keypoints[4 * i], a.keypoints[4 * i + 1], p)) continue;
    const dx = p[0] - b.keypoints[4 * j];
    const dy = p[1] - b.keypoints[4 * j + 1];
    if (dx * dx + dy * dy <= t2) correct++;
  }
  return { total: n, correct, precision: n > 0 ? correct / n : 0 };
}

export interface RepeatabilityResult {
  /** Distinct keypoint positions of A that project inside B (with margin). */
  visible: number;
  /** Distinct keypoint positions of B that project inside A (with margin). */
  visibleB: number;
  /** Visible A keypoints with a B keypoint within tolPx and consistent scale. */
  repeated: number;
  /** repeated / min(visible, visibleB) (Mikolajczyk et al. definition). */
  ratio: number;
}

/**
 * Keypoint repeatability A → B under H: a visible A keypoint is repeated if some B
 * keypoint lies within tolPx of its projection and the scale ratio matches the local
 * homography scale within a factor `scaleTol` (default 1.5).
 */
export function repeatability(a: FeatureSet, b: FeatureSet, H: Mat3, tolPx = 2, marginPx = 10, scaleTol = 1.5): RepeatabilityResult {
  // Simple grid over B for neighbour queries.
  const cs = Math.max(4, tolPx * 2);
  const gw = Math.ceil(b.width / cs) + 1;
  const gh = Math.ceil(b.height / cs) + 1;
  const buckets: number[][] = Array.from({ length: gw * gh }, () => []);
  for (let j = 0; j < b.count; j++) {
    const gx = Math.min(gw - 1, Math.max(0, Math.floor((b.keypoints[4 * j] + 0.5) / cs)));
    const gy = Math.min(gh - 1, Math.max(0, Math.floor((b.keypoints[4 * j + 1] + 0.5) / cs)));
    buckets[gy * gw + gx].push(j);
  }
  const p = [0, 0];
  let visible = 0;
  let repeated = 0;
  const seenPos = new Set<string>();
  for (let i = 0; i < a.count; i++) {
    const xa = a.keypoints[4 * i];
    const ya = a.keypoints[4 * i + 1];
    const key = `${xa.toFixed(2)},${ya.toFixed(2)}`;
    if (seenPos.has(key)) continue; // multiple orientations of one keypoint count once
    seenPos.add(key);
    if (!applyHomography(H, xa, ya, p)) continue;
    if (p[0] < marginPx || p[1] < marginPx || p[0] > b.width - 1 - marginPx || p[1] > b.height - 1 - marginPx) continue;
    if (xa < marginPx || ya < marginPx || xa > a.width - 1 - marginPx || ya > a.height - 1 - marginPx) continue;
    visible++;
    const expScale = a.keypoints[4 * i + 2] * homographyLocalScale(H, xa, ya);
    const gx = Math.floor((p[0] + 0.5) / cs);
    const gy = Math.floor((p[1] + 0.5) / cs);
    let found = false;
    for (let yy = gy - 1; yy <= gy + 1 && !found; yy++) {
      if (yy < 0 || yy >= gh) continue;
      for (let xx = gx - 1; xx <= gx + 1 && !found; xx++) {
        if (xx < 0 || xx >= gw) continue;
        for (const j of buckets[yy * gw + xx]) {
          const dx = b.keypoints[4 * j] - p[0];
          const dy = b.keypoints[4 * j + 1] - p[1];
          if (dx * dx + dy * dy > tolPx * tolPx) continue;
          const r = b.keypoints[4 * j + 2] / expScale;
          if (r < 1 / scaleTol || r > scaleTol) continue;
          found = true;
          break;
        }
      }
    }
    if (found) repeated++;
  }
  const Hi = mat3InvH(H);
  let visibleB = 0;
  seenPos.clear();
  for (let j = 0; j < b.count; j++) {
    const xb = b.keypoints[4 * j];
    const yb = b.keypoints[4 * j + 1];
    const key = `${xb.toFixed(2)},${yb.toFixed(2)}`;
    if (seenPos.has(key)) continue;
    seenPos.add(key);
    if (!applyHomography(Hi, xb, yb, p)) continue;
    if (p[0] < marginPx || p[1] < marginPx || p[0] > a.width - 1 - marginPx || p[1] > a.height - 1 - marginPx) continue;
    if (xb < marginPx || yb < marginPx || xb > b.width - 1 - marginPx || yb > b.height - 1 - marginPx) continue;
    visibleB++;
  }
  const den = Math.min(visible, visibleB);
  return { visible, visibleB, repeated, ratio: den > 0 ? Math.min(1, repeated / den) : 0 };
}

/** Fraction of gx x gy grid cells containing at least one keypoint. */
export function gridCoverage(fs: FeatureSet, gx = 8, gy = 6): number {
  const occ = new Uint8Array(gx * gy);
  for (let i = 0; i < fs.count; i++) {
    const cx = Math.min(gx - 1, Math.max(0, Math.floor(((fs.keypoints[4 * i] + 0.5) / fs.width) * gx)));
    const cy = Math.min(gy - 1, Math.max(0, Math.floor(((fs.keypoints[4 * i + 1] + 0.5) / fs.height) * gy)));
    occ[cy * gx + cx] = 1;
  }
  let n = 0;
  for (let i = 0; i < occ.length; i++) n += occ[i];
  return n / occ.length;
}
