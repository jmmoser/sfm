/**
 * Host-side (CPU) logic of the WebGPU SIFT, kept free of WebGPU calls so it can be
 * unit-tested in Node. Everything here mirrors extractSiftCPU (sift-cpu.ts) exactly:
 * the resampling tables are those of image.ts, detection candidates are put back
 * into the CPU's raster order before de-duplication, and preselection, describe
 * order, coordinate mapping and the final selection use the same code / comparators.
 */

import type { FeatureSet, GrayImage } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';
import { selectFeatureIndices } from './distribute';
import { octaveToFull } from './sift-common';
import type { SiftOptions, SiftPlan } from './sift-common';
import { CANDIDATE_WORDS } from './shaders/sift-wgsl';

// ---------------------------------------------------------------------------
// Resampling tables (identical to image.ts axisFilter)
// ---------------------------------------------------------------------------

export interface AxisTable {
  taps: number;
  index: Int32Array;
  weight: Float32Array;
}

/**
 * 1-D resampling filter from srcN to dstN samples: exact box overlap when shrinking,
 * edge-clamped bilinear when enlarging, identity when equal (same as image.ts).
 */
export function axisResampleTable(srcN: number, dstN: number): AxisTable {
  if (srcN === dstN) {
    const index = new Int32Array(dstN);
    const weight = new Float32Array(dstN).fill(1);
    for (let i = 0; i < dstN; i++) index[i] = i;
    return { taps: 1, index, weight };
  }
  const s = dstN / srcN;
  if (s < 1) {
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
        index[i * taps + t] = Math.min(j, srcN - 1);
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
  const index = new Int32Array(dstN * 2);
  const weight = new Float32Array(dstN * 2);
  for (let i = 0; i < dstN; i++) {
    let x = (i + 0.5) / s - 0.5;
    if (x < 0) x = 0;
    if (x > srcN - 1) x = srcN - 1;
    const j0 = Math.floor(x);
    const f = x - j0;
    index[2 * i] = j0;
    index[2 * i + 1] = Math.min(j0 + 1, srcN - 1);
    weight[2 * i] = 1 - f;
    weight[2 * i + 1] = f;
  }
  return { taps: 2, index, weight };
}

export interface ResizeTables {
  /** [x index | x weight bits | y index | y weight bits] as u32 words. */
  data: Uint32Array;
  tx: number;
  ty: number;
  oxi: number;
  oxw: number;
  oyi: number;
  oyw: number;
}

/** Packs the x / y resampling tables for the GPU resize kernel. */
export function packResizeTables(sw: number, sh: number, dw: number, dh: number): ResizeTables {
  const fx = axisResampleTable(sw, dw);
  const fy = axisResampleTable(sh, dh);
  const nx = fx.index.length;
  const ny = fy.index.length;
  const data = new Uint32Array(2 * nx + 2 * ny);
  const oxi = 0;
  const oxw = nx;
  const oyi = 2 * nx;
  const oyw = 2 * nx + ny;
  data.set(new Uint32Array(fx.index.buffer, fx.index.byteOffset, nx), oxi);
  data.set(new Uint32Array(fx.weight.buffer, fx.weight.byteOffset, nx), oxw);
  data.set(new Uint32Array(fy.index.buffer, fy.index.byteOffset, ny), oyi);
  data.set(new Uint32Array(fy.weight.buffer, fy.weight.byteOffset, ny), oyw);
  return { data, tx: fx.taps, ty: fy.taps, oxi, oxw, oyi, oyw };
}

/**
 * Resamples a plane on the CPU with the packed tables, using the same arithmetic as
 * the GPU kernel (for tests).
 */
export function applyResizeTables(src: ArrayLike<number>, sw: number, dw: number, dh: number, t: ResizeTables, scale = 1): Float32Array {
  const out = new Float32Array(dw * dh);
  const w = new Float32Array(t.data.buffer, t.data.byteOffset, t.data.length);
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      let acc = 0;
      for (let a = 0; a < t.ty; a++) {
        const wy = w[t.oyw + y * t.ty + a];
        if (wy === 0) continue;
        const iy = t.data[t.oyi + y * t.ty + a];
        let row = 0;
        for (let b = 0; b < t.tx; b++) row += w[t.oxw + x * t.tx + b] * src[iy * sw + t.data[t.oxi + x * t.tx + b]];
        acc += wy * row;
      }
      out[y * dw + x] = acc * scale;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Blur kernels
// ---------------------------------------------------------------------------

/** Incremental blur sigmas of a scale space: sigmaBase (L0 from the base image) and inc[1..S+2]. */
export function scaleSpaceSigmas(o: Required<SiftOptions>, baseBlur: number): { sigmaBase: number; inc: number[] } {
  const S = o.scalesPerOctave;
  const sigma = o.sigma;
  const inc: number[] = [0];
  for (let i = 1; i <= S + 2; i++) inc[i] = sigma * Math.sqrt(2 ** ((2 * i) / S) - 2 ** ((2 * (i - 1)) / S));
  const sigmaBase = sigma > baseBlur ? Math.sqrt(sigma * sigma - baseBlur * baseBlur) : 0;
  return { sigmaBase, inc };
}

// ---------------------------------------------------------------------------
// Detection candidates
// ---------------------------------------------------------------------------

/** Keypoint candidates in the CPU's order (structure of arrays). */
export interface SiftCandidates {
  n: number;
  /** Refined position in octave pixels. */
  x: Float64Array;
  y: Float64Array;
  /** Continuous DoG level. */
  s: Float64Array;
  /** Detection scale in octave pixels: sigma * 2^(s / S). */
  sigma: Float64Array;
  /** |DoG| at the refined extremum. */
  score: Float32Array;
  /** Index into plan.octaves. */
  octIdx: Int32Array;
  /** Gaussian level used for orientation / descriptor: clamp(round(s), 1, S). */
  level: Int32Array;
}

/**
 * Decodes raw GPU detection records. The GPU appends candidates in arbitrary order;
 * they are sorted back into the CPU's detection order (octave, start level, start
 * row, start column) and refined extrema that converged to the same integer location
 * are de-duplicated keeping the first, exactly like detectOctave's `seen` set.
 */
export function decodeCandidates(raw: Uint32Array, count: number, plan: SiftPlan, S: number, sigma: number): SiftCandidates {
  const n0 = Math.min(count, Math.floor(raw.length / CANDIDATE_WORDS));
  const f = new Float32Array(raw.buffer, raw.byteOffset, n0 * CANDIDATE_WORDS);
  let W0 = 1;
  let H0 = 1;
  for (const oc of plan.octaves) {
    W0 = Math.max(W0, oc.width);
    H0 = Math.max(H0, oc.height);
  }
  const L = S + 2;
  const startKey = (k: number): number => {
    const o = k * CANDIDATE_WORDS;
    const w5 = raw[o + 5];
    const w6 = raw[o + 6];
    const s0 = (w5 >>> 8) & 0xff;
    const oct = w5 >>> 16;
    return ((oct * L + s0) * H0 + (w6 >>> 16)) * W0 + (w6 & 0xffff);
  };
  // Sort indices by start key (unique per start location).
  const order = new Uint32Array(n0);
  const maxKey = plan.octaves.length * L * H0 * W0;
  if (maxKey * Math.max(1, n0) < 2 ** 52) {
    const keys = new Float64Array(n0);
    for (let k = 0; k < n0; k++) keys[k] = startKey(k) * n0 + k;
    keys.sort();
    for (let k = 0; k < n0; k++) order[k] = keys[k] % n0;
  } else {
    const keys = new Float64Array(n0);
    for (let k = 0; k < n0; k++) {
      keys[k] = startKey(k);
      order[k] = k;
    }
    order.sort((a, b) => keys[a] - keys[b] || a - b);
  }
  const seen = new Set<number>();
  const out: SiftCandidates = {
    n: 0,
    x: new Float64Array(n0),
    y: new Float64Array(n0),
    s: new Float64Array(n0),
    sigma: new Float64Array(n0),
    score: new Float32Array(n0),
    octIdx: new Int32Array(n0),
    level: new Int32Array(n0),
  };
  let m = 0;
  for (let q = 0; q < n0; q++) {
    const k = order[q];
    const o = k * CANDIDATE_WORDS;
    const w4 = raw[o + 4];
    const w5 = raw[o + 5];
    const oct = w5 >>> 16;
    const si = w5 & 0xff;
    const fkey = ((oct * L + si) * H0 + (w4 >>> 16)) * W0 + (w4 & 0xffff);
    if (seen.has(fkey)) continue;
    seen.add(fkey);
    const sr = f[o + 2];
    out.x[m] = f[o];
    out.y[m] = f[o + 1];
    out.s[m] = sr;
    out.sigma[m] = sigma * 2 ** (sr / S);
    out.score[m] = Math.abs(f[o + 3]);
    out.octIdx[m] = oct;
    let lv = Math.round(sr);
    if (lv < 1) lv = 1;
    else if (lv > S) lv = S;
    out.level[m] = lv;
    m++;
  }
  out.n = m;
  return out;
}

export interface Preselection {
  /** Indices into the candidates, sorted by descending score (distributed). */
  pre: Uint32Array;
  /** Processed-image positions of all candidates (x_oct * 2^octave). */
  px: Float32Array;
  py: Float32Array;
}

/** Spatially distributed preselection of keypoints (before orientation), as extractSiftCPU. */
export function preselectCandidates(c: SiftCandidates, plan: SiftPlan, o: Required<SiftOptions>): Preselection {
  const nk = c.n;
  const px = new Float32Array(nk);
  const py = new Float32Array(nk);
  for (let i = 0; i < nk; i++) {
    const f = 2 ** plan.octaves[c.octIdx[i]].octave;
    px[i] = c.x[i] * f;
    py[i] = c.y[i] * f;
  }
  const pre = selectFeatureIndices(px, py, c.score, nk, plan.procWidth, plan.procHeight, o.maxFeatures, o.distribute);
  return { pre, px, py };
}

export interface DescribeGroup {
  octIdx: number;
  level: number;
  /** Slot range [start, end) in describe order. */
  start: number;
  end: number;
}

export interface DescribeOrder {
  /** order[q] = index into `pre` of the keypoint in slot q. */
  order: Uint32Array;
  /** Contiguous slot ranges sharing (octave, level). */
  groups: DescribeGroup[];
  /** Per slot: x, y (octave pixels), sigma (octave pixels), 0 — upload for the GPU. */
  kp: Float32Array;
}

/** The CPU's describe order: grouped by (octave, level), raster order inside a group. */
export function describeOrder(c: SiftCandidates, pre: Uint32Array, S: number): DescribeOrder {
  const np = pre.length;
  const groupKey = new Int32Array(np);
  const order = new Uint32Array(np);
  for (let m = 0; m < np; m++) {
    const i = pre[m];
    groupKey[m] = c.octIdx[i] * (S + 1) + c.level[i];
    order[m] = m;
  }
  const { x, y } = c;
  order.sort((a, b) => groupKey[a] - groupKey[b] || y[pre[a]] - y[pre[b]] || x[pre[a]] - x[pre[b]] || a - b);
  const groups: DescribeGroup[] = [];
  const kp = new Float32Array(4 * np);
  for (let q = 0; q < np; q++) {
    const m = order[q];
    const i = pre[m];
    kp[4 * q] = c.x[i];
    kp[4 * q + 1] = c.y[i];
    kp[4 * q + 2] = c.sigma[i];
    const g = groups[groups.length - 1];
    if (g && g.octIdx === c.octIdx[i] && g.level === c.level[i]) g.end = q + 1;
    else groups.push({ octIdx: c.octIdx[i], level: c.level[i], start: q, end: q + 1 });
  }
  return { order, groups, kp };
}

/**
 * Builds the final FeatureSet from the GPU orientation / descriptor output exactly
 * like extractSiftCPU: oriented features in describe order, keypoints mapped to the
 * full-resolution frame, final distributed selection sorted by descending score.
 * `descWords` holds maxOri * 32 u32 per slot (4 descriptor bytes per word).
 */
export function assembleFeatures(
  plan: SiftPlan,
  o: Required<SiftOptions>,
  c: SiftCandidates,
  sel: Preselection,
  ord: DescribeOrder,
  counts: Uint32Array,
  angles: Float32Array,
  descBytes: Uint8Array,
): FeatureSet {
  const maxOri = o.maxOrientations;
  const np = ord.order.length;
  const maxOut = np * maxOri;
  const outKp = new Float32Array(4 * maxOut);
  const outDesc = new Uint8Array(DESCRIPTOR_SIZE * maxOut);
  const outScore = new Float32Array(maxOut);
  const outPx = new Float32Array(maxOut);
  const outPy = new Float32Array(maxOut);
  let nOut = 0;
  for (let q = 0; q < np; q++) {
    const i = sel.pre[ord.order[q]];
    const octave = plan.octaves[c.octIdx[i]].octave;
    const na = Math.min(counts[q], maxOri);
    for (let a = 0; a < na; a++) {
      const slot = q * maxOri + a;
      outDesc.set(descBytes.subarray(slot * DESCRIPTOR_SIZE, (slot + 1) * DESCRIPTOR_SIZE), nOut * DESCRIPTOR_SIZE);
      octaveToFull(plan, octave, c.x[i], c.y[i], c.sigma[i], outKp, 4 * nOut);
      outKp[4 * nOut + 3] = angles[slot];
      outScore[nOut] = c.score[i];
      outPx[nOut] = sel.px[i];
      outPy[nOut] = sel.py[i];
      nOut++;
    }
  }
  const fin = selectFeatureIndices(outPx, outPy, outScore, nOut, plan.procWidth, plan.procHeight, o.maxFeatures, o.distribute);
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
  return { count: n, keypoints, descriptors, scores, width: plan.fullWidth, height: plan.fullHeight };
}

/** Empty FeatureSet in a frame of width x height. */
export function emptyFeatures(width: number, height: number): FeatureSet {
  return { count: 0, keypoints: new Float32Array(0), descriptors: new Uint8Array(0), scores: new Float32Array(0), width, height };
}

/**
 * Full-resolution frame size given optional fullWidth / fullHeight (one given: the
 * other follows the input aspect), as extractSiftCPU. Throws on invalid sizes.
 */
export function resolveFullSize(w: number, h: number, fullWidth?: number, fullHeight?: number, who = 'SiftGPU'): [number, number] {
  const FW = fullWidth ?? (fullHeight !== undefined ? Math.round((w * fullHeight) / h) : w);
  const FH = fullHeight ?? (fullWidth !== undefined ? Math.round((h * fullWidth) / w) : h);
  if (!(FW > 0 && FH > 0 && Number.isFinite(FW) && Number.isFinite(FH))) {
    throw new Error(`${who}: invalid full-resolution size ${FW}x${FH}`);
  }
  return [FW, FH];
}

// ---------------------------------------------------------------------------
// Self-test support
// ---------------------------------------------------------------------------

/**
 * Deterministic synthetic test image (smooth background, Gaussian blobs of several
 * sizes, anti-aliased rectangles), Float32 0..1. Used by SiftGPU.selfTest.
 */
export function selfTestImage(W = 128, H = 96): GrayImage {
  const d = new Float32Array(W * H);
  let seed = 0x2545f491;
  const rnd = () => {
    seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
    return seed / 4294967296;
  };
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) d[y * W + x] = 0.5 + 0.08 * Math.sin(x * 0.21) * Math.cos(y * 0.17);
  const nBlob = Math.round((W * H) / 220);
  for (let b = 0; b < nBlob; b++) {
    const cx = rnd() * W;
    const cy = rnd() * H;
    const s = 1.3 + rnd() * 4;
    const a = (rnd() < 0.5 ? -1 : 1) * (0.15 + rnd() * 0.25);
    const e = Math.ceil(3 * s);
    for (let y = Math.max(0, Math.floor(cy - e)); y <= Math.min(H - 1, Math.ceil(cy + e)); y++)
      for (let x = Math.max(0, Math.floor(cx - e)); x <= Math.min(W - 1, Math.ceil(cx + e)); x++)
        d[y * W + x] += a * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s));
  }
  const nRect = Math.round((W * H) / 1500);
  for (let r = 0; r < nRect; r++) {
    const x0 = rnd() * W;
    const y0 = rnd() * H;
    const x1 = x0 + 4 + rnd() * 20;
    const y1 = y0 + 4 + rnd() * 20;
    const v = 0.1 + 0.8 * rnd();
    for (let y = Math.max(0, Math.floor(y0)); y <= Math.min(H - 1, Math.ceil(y1)); y++)
      for (let x = Math.max(0, Math.floor(x0)); x <= Math.min(W - 1, Math.ceil(x1)); x++) {
        const cov = Math.max(0, Math.min(1, x + 0.5 - x0, x1 - x + 0.5)) * Math.max(0, Math.min(1, y + 0.5 - y0, y1 - y + 0.5));
        d[y * W + x] += (v - d[y * W + x]) * cov * 0.7;
      }
  }
  for (let i = 0; i < d.length; i++) d[i] = d[i] < 0 ? 0 : d[i] > 1 ? 1 : d[i];
  return { width: W, height: H, data: d };
}

export interface FeatureParity {
  /** Reference keypoints with a test keypoint within 1 px and 20 % scale. */
  posFrac: number;
  /** Median cosine similarity of the descriptors of corresponding (same orientation ±0.2 rad) features. */
  medianCos: number;
  pairs: number;
}

/** Agreement of a feature set with a reference set (e.g. GPU vs CPU). */
export function featureParity(ref: FeatureSet, test: FeatureSet): FeatureParity {
  const cs = 4;
  const gw = Math.ceil(Math.max(1, test.width) / cs) + 2;
  const gh = Math.ceil(Math.max(1, test.height) / cs) + 2;
  const cellOf = (x: number, y: number) =>
    Math.min(gh - 1, Math.max(0, Math.floor(y / cs) + 1)) * gw + Math.min(gw - 1, Math.max(0, Math.floor(x / cs) + 1));
  const start = new Int32Array(gw * gh + 1);
  const cell = new Int32Array(test.count);
  for (let j = 0; j < test.count; j++) {
    cell[j] = cellOf(test.keypoints[4 * j], test.keypoints[4 * j + 1]);
    start[cell[j] + 1]++;
  }
  for (let c = 0; c < gw * gh; c++) start[c + 1] += start[c];
  const items = new Int32Array(test.count);
  const fill = start.slice(0, gw * gh);
  for (let j = 0; j < test.count; j++) items[fill[cell[j]]++] = j;
  let pos = 0;
  const cos: number[] = [];
  const D = DESCRIPTOR_SIZE;
  for (let i = 0; i < ref.count; i++) {
    const x = ref.keypoints[4 * i];
    const y = ref.keypoints[4 * i + 1];
    const s = ref.keypoints[4 * i + 2];
    const th = ref.keypoints[4 * i + 3];
    let found = false;
    let bestJ = -1;
    let bestA = Infinity;
    const cx = Math.floor(x / cs) + 1;
    const cy = Math.floor(y / cs) + 1;
    for (let yy = cy - 1; yy <= cy + 1; yy++) {
      if (yy < 0 || yy >= gh) continue;
      for (let xx = cx - 1; xx <= cx + 1; xx++) {
        if (xx < 0 || xx >= gw) continue;
        const c = yy * gw + xx;
        for (let t = start[c]; t < start[c + 1]; t++) {
          const j = items[t];
          const r = test.keypoints[4 * j + 2] / s;
          if (Math.hypot(test.keypoints[4 * j] - x, test.keypoints[4 * j + 1] - y) > 1 || r < 1 / 1.2 || r > 1.2) continue;
          found = true;
          let a = Math.abs(test.keypoints[4 * j + 3] - th) % (2 * Math.PI);
          if (a > Math.PI) a = 2 * Math.PI - a;
          if (a < bestA) {
            bestA = a;
            bestJ = j;
          }
        }
      }
    }
    if (!found) continue;
    pos++;
    if (bestA < 0.2) {
      let dt = 0;
      let na = 0;
      let nb = 0;
      for (let k = 0; k < D; k++) {
        const u = ref.descriptors[i * D + k];
        const v = test.descriptors[bestJ * D + k];
        dt += u * v;
        na += u * u;
        nb += v * v;
      }
      cos.push(na > 0 && nb > 0 ? dt / Math.sqrt(na * nb) : na === nb ? 1 : 0);
    }
  }
  cos.sort((a, b) => a - b);
  return { posFrac: ref.count ? pos / ref.count : 1, medianCos: cos.length ? cos[cos.length >> 1] : ref.count ? 0 : 1, pairs: cos.length };
}
