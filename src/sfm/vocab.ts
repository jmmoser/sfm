/**
 * Bag-of-visual-words image retrieval for loop closure (ARCHITECTURE §5 "Refine").
 *
 * Vocabulary: flat k-means codebook (k ≈ 256–1024) trained on a random subsample of the
 * RootSIFT descriptors (uint8, compared as floats): k-means++ seeding (Arthur & Vassilvitskii
 * 2007) on a smaller subsample, then Lloyd iterations accelerated with Hamerly's bounds
 * (Hamerly 2010, "Making k-means even faster", SDM) — after the first pass most points skip
 * the full search. Features are quantised by EXACT nearest-centroid search
 * (argmaxₖ x·cₖ − ½‖cₖ‖², 4-centroid register blocking).
 *
 * Index: TF-IDF weighting with L2-normalised BoW vectors and cosine scoring (Sivic &
 * Zisserman 2003, "Video Google"); burstiness handled by power-law term frequencies
 * (tf → √tf, Jégou, Douze & Schmid 2009, "On the burstiness of visual elements") or binary
 * tf. IDF = ln(N / nᵥᵥ) is recomputed lazily when images are added, so `add` is O(features·k·128)
 * for the quantisation plus O(1) bookkeeping.
 *
 * Complexity (D = 128, measured in Node/V8 x86, ≈ 1 GMAC/s; phones similar to ½×):
 *  - quantisation: n·k·D MACs per image — 4000 features × k = 512 ≈ 0.26 s, so
 *    500 images × 4000 features ≈ 130 s total. `maxFeaturesPerImage` (strongest by detector
 *    score) bounds it: 1500 ⇒ ≈ 0.1 s / image ≈ 50 s for 500 images, spread over capture.
 *  - training: seeding s·k·D (s = seedPerWord·k), first Lloyd pass M·k·D (M = trainingPerWord·k),
 *    later passes typically 5–20 % of that: k = 512 ⇒ ≈ 1–1.5 s.
 *  - query: O(total non-zeros) = O(images · min(k, features)) — 500 images × 512 words ≈ 0.3 ms.
 */
import { DESCRIPTOR_SIZE, type FeatureSet } from '../core/types';
import { Rng } from '../math/random';
import type { RetrievalSource } from './pairs';

const D = DESCRIPTOR_SIZE;

/** Options of the vocabulary and the index. */
export interface VocabularyOptions {
  /** Number of visual words k (default 512). */
  numWords: number;
  /** Lloyd iterations after seeding (default 8; stops early when no assignment changes). */
  iterations: number;
  /** Training sample size per word (default 24 ⇒ M = 24·k descriptors). */
  trainingPerWord: number;
  /** k-means++ seeding sample per word (default 8). */
  seedPerWord: number;
  /** Rng seed for sampling / seeding (default 0x5eed). */
  seed: number;
  /** Term-frequency transform against bursts (default 'sqrt'). */
  burstiness: 'sqrt' | 'binary' | 'none';
  /** Index / query only the strongest N features per image by detector score (0 = all; default 0). */
  maxFeaturesPerImage: number;
}

/** Default `VocabularyOptions`. */
export const DEFAULT_VOCABULARY_OPTIONS: VocabularyOptions = {
  numWords: 512,
  iterations: 8,
  trainingPerWord: 24,
  seedPerWord: 8,
  seed: 0x5eed,
  burstiness: 'sqrt',
  maxFeaturesPerImage: 0,
};

/** Sparse bag of words: sorted unique word ids and their (burstiness-adjusted) term frequencies. */
export interface BowVector {
  words: Uint32Array;
  tf: Float32Array;
}

/** Retrieval hit. */
export interface RetrievalResult {
  id: number;
  score: number;
}

/** Descriptor source for training: FeatureSets or raw descriptor arrays (128 bytes / feature). */
export type DescriptorSource = FeatureSet | { descriptors: Uint8Array; count: number };

// ------------------------------------------------------------------------------------
// Kernels
// ------------------------------------------------------------------------------------

/** ½‖c‖² per centroid. */
function halfNorms(cb: Float64Array, k: number, out: Float64Array): void {
  for (let c = 0; c < k; c++) {
    let s = 0;
    const o = c * D;
    for (let d = 0; d < D; d++) s += cb[o + d] * cb[o + d];
    out[c] = 0.5 * s;
  }
}

/** Nearest centroid of xb (Float64 scratch, D values): argmax x·c − ½‖c‖². */
function nearest(xb: Float64Array, cb: Float64Array, hn: Float64Array, k: number): number {
  let best = -Infinity,
    bi = 0;
  let c = 0;
  for (; c + 4 <= k; c += 4) {
    let s0 = 0,
      s1 = 0,
      s2 = 0,
      s3 = 0;
    const o0 = c * D,
      o1 = o0 + D,
      o2 = o1 + D,
      o3 = o2 + D;
    for (let d = 0; d < D; d++) {
      const x = xb[d];
      s0 += x * cb[o0 + d];
      s1 += x * cb[o1 + d];
      s2 += x * cb[o2 + d];
      s3 += x * cb[o3 + d];
    }
    s0 -= hn[c];
    s1 -= hn[c + 1];
    s2 -= hn[c + 2];
    s3 -= hn[c + 3];
    if (s0 > best) {
      best = s0;
      bi = c;
    }
    if (s1 > best) {
      best = s1;
      bi = c + 1;
    }
    if (s2 > best) {
      best = s2;
      bi = c + 2;
    }
    if (s3 > best) {
      best = s3;
      bi = c + 3;
    }
  }
  for (; c < k; c++) {
    let s = 0;
    const o = c * D;
    for (let d = 0; d < D; d++) s += xb[d] * cb[o + d];
    s -= hn[c];
    if (s > best) {
      best = s;
      bi = c;
    }
  }
  return bi;
}

/**
 * Nearest and second-nearest centroid with true Euclidean distances (for Hamerly bounds).
 * Writes [index, d1, d2] into res.
 */
function nearest2(xb: Float64Array, xn2: number, cb: Float64Array, hn: Float64Array, k: number, res: Float64Array): void {
  // maximise g = x·c − ½‖c‖²  ⇔ minimise d² = ‖x‖² − 2g
  let g1 = -Infinity,
    g2 = -Infinity,
    i1 = 0;
  let c = 0;
  for (; c + 4 <= k; c += 4) {
    let s0 = 0,
      s1 = 0,
      s2 = 0,
      s3 = 0;
    const o0 = c * D,
      o1 = o0 + D,
      o2 = o1 + D,
      o3 = o2 + D;
    for (let d = 0; d < D; d++) {
      const x = xb[d];
      s0 += x * cb[o0 + d];
      s1 += x * cb[o1 + d];
      s2 += x * cb[o2 + d];
      s3 += x * cb[o3 + d];
    }
    const v0 = s0 - hn[c],
      v1 = s1 - hn[c + 1],
      v2 = s2 - hn[c + 2],
      v3 = s3 - hn[c + 3];
    if (v0 > g2) {
      if (v0 > g1) {
        g2 = g1;
        g1 = v0;
        i1 = c;
      } else g2 = v0;
    }
    if (v1 > g2) {
      if (v1 > g1) {
        g2 = g1;
        g1 = v1;
        i1 = c + 1;
      } else g2 = v1;
    }
    if (v2 > g2) {
      if (v2 > g1) {
        g2 = g1;
        g1 = v2;
        i1 = c + 2;
      } else g2 = v2;
    }
    if (v3 > g2) {
      if (v3 > g1) {
        g2 = g1;
        g1 = v3;
        i1 = c + 3;
      } else g2 = v3;
    }
  }
  for (; c < k; c++) {
    let s = 0;
    const o = c * D;
    for (let d = 0; d < D; d++) s += xb[d] * cb[o + d];
    const v = s - hn[c];
    if (v > g2) {
      if (v > g1) {
        g2 = g1;
        g1 = v;
        i1 = c;
      } else g2 = v;
    }
  }
  res[0] = i1;
  res[1] = Math.sqrt(Math.max(0, xn2 - 2 * g1));
  res[2] = g2 === -Infinity ? Infinity : Math.sqrt(Math.max(0, xn2 - 2 * g2));
}

function loadDescriptor(desc: Uint8Array, i: number, xb: Float64Array): number {
  const o = i * D;
  let n2 = 0;
  for (let d = 0; d < D; d++) {
    const v = desc[o + d];
    xb[d] = v;
    n2 += v * v;
  }
  return n2;
}

/**
 * Exact nearest-word quantisation of `count` descriptors (uint8, 128 per feature).
 * If `indices` is given, only those features are quantised (out[j] ↔ indices[j]).
 */
export function quantizeDescriptors(
  codebook: Float64Array,
  numWords: number,
  descriptors: Uint8Array,
  count: number,
  out?: Uint32Array,
  indices?: ArrayLike<number>,
  halfNormsCache?: Float64Array,
): Uint32Array {
  const m = indices ? indices.length : count;
  const res = out && out.length >= m ? out : new Uint32Array(m);
  let hn = halfNormsCache;
  if (!hn) {
    hn = new Float64Array(numWords);
    halfNorms(codebook, numWords, hn);
  }
  const xb = new Float64Array(D);
  for (let j = 0; j < m; j++) {
    const i = indices ? indices[j] : j;
    loadDescriptor(descriptors, i, xb);
    res[j] = nearest(xb, codebook, hn, numWords);
  }
  return res;
}

// ------------------------------------------------------------------------------------
// Training
// ------------------------------------------------------------------------------------

function descCount(s: DescriptorSource): number {
  return Math.min(s.count, Math.floor(s.descriptors.length / D));
}

/** Uniform random subsample of ≈ m descriptors across sources (proportional per source). */
function sampleDescriptors(sources: readonly DescriptorSource[], m: number, rng: Rng): Uint8Array {
  let total = 0;
  for (const s of sources) total += descCount(s);
  if (total === 0) return new Uint8Array(0);
  if (m >= total) {
    const out = new Uint8Array(total * D);
    let o = 0;
    for (const s of sources) {
      const c = descCount(s);
      out.set(s.descriptors.subarray(0, c * D), o);
      o += c * D;
    }
    return out;
  }
  // Largest-remainder apportionment so that exactly m are drawn.
  const quotas = sources.map((s) => (descCount(s) * m) / total);
  const take = quotas.map((q) => Math.floor(q));
  let rem = m - take.reduce((a, b) => a + b, 0);
  const order = quotas.map((q, i) => [q - Math.floor(q), i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  for (let r = 0; r < order.length && rem > 0; r++) {
    const i = order[r][1];
    if (take[i] < descCount(sources[i])) {
      take[i]++;
      rem--;
    }
  }
  const out = new Uint8Array(m * D);
  let o = 0;
  for (let si = 0; si < sources.length; si++) {
    const s = sources[si];
    const c = descCount(s);
    const t = Math.min(take[si], c);
    if (t === 0) continue;
    const idx = rng.sample(c, t);
    for (let j = 0; j < t; j++) {
      out.set(s.descriptors.subarray(idx[j] * D, idx[j] * D + D), o);
      o += D;
    }
  }
  return o === out.length ? out : out.subarray(0, o);
}

/** k-means++ seeding (D² sampling) on `data` (uint8, n × D). Returns k × D float centroids. */
function kmeansPlusPlus(data: Uint8Array, n: number, k: number, rng: Rng): Float64Array {
  const cb = new Float64Array(k * D);
  const minD2 = new Float64Array(n).fill(Infinity);
  const xn = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const o = i * D;
    for (let d = 0; d < D; d++) s += data[o + d] * data[o + d];
    xn[i] = s;
  }
  let pick = rng.int(n);
  const cbuf = new Float64Array(D);
  for (let c = 0; c < k; c++) {
    const po = pick * D;
    let cn = 0;
    for (let d = 0; d < D; d++) {
      const v = data[po + d];
      cb[c * D + d] = v;
      cbuf[d] = v;
      cn += v * v;
    }
    if (c === k - 1) break;
    // update D² and its sum
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const o = i * D;
      let dot = 0;
      for (let d = 0; d < D; d++) dot += data[o + d] * cbuf[d];
      const d2 = Math.max(0, xn[i] + cn - 2 * dot);
      if (d2 < minD2[i]) minD2[i] = d2;
      sum += minD2[i];
    }
    if (!(sum > 0)) {
      // all remaining points coincide with centres: pick uniformly
      pick = rng.int(n);
      continue;
    }
    let r = rng.nextDouble() * sum;
    pick = n - 1;
    for (let i = 0; i < n; i++) {
      r -= minD2[i];
      if (r < 0) {
        pick = i;
        break;
      }
    }
  }
  return cb;
}

/**
 * Hamerly-accelerated Lloyd iterations on `data` (uint8, n × D), refining `cb` (k × D)
 * in place. Empty clusters are re-seeded with the point farthest from its centre.
 */
function lloydHamerly(data: Uint8Array, n: number, cb: Float64Array, k: number, iterations: number): void {
  const assign = new Int32Array(n);
  const upper = new Float64Array(n);
  const lower = new Float64Array(n);
  const xn = new Float64Array(n);
  const hn = new Float64Array(k);
  const xb = new Float64Array(D);
  const res = new Float64Array(3);
  halfNorms(cb, k, hn);
  for (let i = 0; i < n; i++) {
    xn[i] = loadDescriptor(data, i, xb);
    nearest2(xb, xn[i], cb, hn, k, res);
    assign[i] = res[0];
    upper[i] = res[1];
    lower[i] = res[2];
  }
  const sums = new Float64Array(k * D);
  const counts = new Int32Array(k);
  const moved = new Float64Array(k);
  const sHalf = new Float64Array(k);
  const old = new Float64Array(D);
  for (let it = 0; it < iterations; it++) {
    // ---- update step
    sums.fill(0);
    counts.fill(0);
    for (let i = 0; i < n; i++) {
      const a = assign[i];
      counts[a]++;
      const o = i * D,
        so = a * D;
      for (let d = 0; d < D; d++) sums[so + d] += data[o + d];
    }
    let maxMove = 0,
      maxMove2 = 0,
      argMax = -1;
    for (let c = 0; c < k; c++) {
      const co = c * D;
      for (let d = 0; d < D; d++) old[d] = cb[co + d];
      if (counts[c] > 0) {
        const inv = 1 / counts[c];
        for (let d = 0; d < D; d++) cb[co + d] = sums[co + d] * inv;
      } else {
        // re-seed: farthest point (by its upper bound) — becomes exactly this centre
        let far = 0,
          fd = -1;
        for (let i = 0; i < n; i++)
          if (upper[i] > fd && counts[assign[i]] > 1) {
            fd = upper[i];
            far = i;
          }
        const prev = assign[far];
        counts[prev]--;
        for (let d = 0; d < D; d++) {
          sums[prev * D + d] -= data[far * D + d];
          cb[co + d] = data[far * D + d];
        }
        assign[far] = c;
        upper[far] = 0;
        lower[far] = 0; // force a re-check next pass
        counts[c] = 1;
      }
      let mv = 0;
      for (let d = 0; d < D; d++) {
        const e = cb[co + d] - old[d];
        mv += e * e;
      }
      mv = Math.sqrt(mv);
      moved[c] = mv;
      if (mv > maxMove) {
        maxMove2 = maxMove;
        maxMove = mv;
        argMax = c;
      } else if (mv > maxMove2) maxMove2 = mv;
    }
    if (maxMove === 0) break;
    halfNorms(cb, k, hn);
    for (let i = 0; i < n; i++) {
      upper[i] += moved[assign[i]];
      lower[i] -= assign[i] === argMax ? maxMove2 : maxMove;
    }
    // s(c) = ½ min distance to another centre
    for (let c = 0; c < k; c++) sHalf[c] = Infinity;
    for (let c = 0; c < k; c++) {
      const co = c * D;
      for (let c2 = c + 1; c2 < k; c2++) {
        const c2o = c2 * D;
        let dot = 0;
        for (let d = 0; d < D; d++) dot += cb[co + d] * cb[c2o + d];
        const dist = 0.5 * Math.sqrt(Math.max(0, 2 * hn[c] + 2 * hn[c2] - 2 * dot));
        if (dist < sHalf[c]) sHalf[c] = dist;
        if (dist < sHalf[c2]) sHalf[c2] = dist;
      }
    }
    // ---- assignment step with bounds
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const a = assign[i];
      const m = Math.max(sHalf[a], lower[i]);
      if (upper[i] <= m) continue;
      // tighten upper bound
      const o = i * D,
        ao = a * D;
      let dot = 0;
      for (let d = 0; d < D; d++) dot += data[o + d] * cb[ao + d];
      upper[i] = Math.sqrt(Math.max(0, xn[i] - 2 * dot + 2 * hn[a]));
      if (upper[i] <= m) continue;
      loadDescriptor(data, i, xb);
      nearest2(xb, xn[i], cb, hn, k, res);
      if (res[0] !== a) changed++;
      assign[i] = res[0];
      upper[i] = res[1];
      lower[i] = res[2];
    }
    if (changed === 0) break;
  }
  // final centre update for the last assignment
  sums.fill(0);
  counts.fill(0);
  for (let i = 0; i < n; i++) {
    const a = assign[i];
    counts[a]++;
    const o = i * D,
      so = a * D;
    for (let d = 0; d < D; d++) sums[so + d] += data[o + d];
  }
  for (let c = 0; c < k; c++) {
    if (counts[c] === 0) continue;
    const inv = 1 / counts[c];
    for (let d = 0; d < D; d++) cb[c * D + d] = sums[c * D + d] * inv;
  }
}

/**
 * Train a k-means codebook (k × 128 floats, row-major) from a random subsample of the
 * descriptors. k is reduced to the number of available descriptors if necessary.
 */
export function trainVocabulary(sources: readonly DescriptorSource[], opts: Partial<VocabularyOptions> = {}): Float64Array {
  const o = { ...DEFAULT_VOCABULARY_OPTIONS, ...opts };
  const rng = new Rng(o.seed);
  let total = 0;
  for (const s of sources) total += descCount(s);
  if (total === 0) throw new RangeError('trainVocabulary: no descriptors');
  const k = Math.max(1, Math.min(Math.floor(o.numWords), total));
  const sample = sampleDescriptors(sources, Math.max(k, Math.floor(o.trainingPerWord * k)), rng);
  const n = sample.length / D;
  // Seeding subsample (a prefix of a shuffled order is a uniform subsample).
  const ns = Math.min(n, Math.max(k, Math.floor(o.seedPerWord * k)));
  let seedData = sample;
  if (ns < n) {
    const idx = rng.sample(n, ns);
    seedData = new Uint8Array(ns * D);
    for (let j = 0; j < ns; j++) seedData.set(sample.subarray(idx[j] * D, idx[j] * D + D), j * D);
  }
  const cb = kmeansPlusPlus(seedData, ns, k, rng);
  if (o.iterations > 0) lloydHamerly(sample, n, cb, k, o.iterations);
  return cb;
}

// ------------------------------------------------------------------------------------
// Index
// ------------------------------------------------------------------------------------

interface Doc {
  words: Uint32Array;
  tf: Float32Array;
  invNorm: number;
}

/**
 * Inverted-file-free BoW index (forward scan over sparse document vectors; the vectors are
 * dense-ish for k ≤ 1024, where scanning is as fast as an inverted file and simpler).
 */
export class VocabularyIndex implements RetrievalSource {
  readonly numWords: number;
  /** Codebook, numWords × 128 (row-major, float64). */
  readonly codebook: Float64Array;
  readonly options: VocabularyOptions;
  private hn: Float64Array;
  private docs = new Map<number, Doc>();
  private df: Uint32Array;
  private idf: Float64Array;
  private dirty = true;
  private counts: Float64Array;
  private qv: Float64Array;

  /** Wrap an existing codebook (k × 128, float32 or float64). */
  constructor(codebook: Float32Array | Float64Array, opts: Partial<VocabularyOptions> = {}) {
    if (codebook.length === 0 || codebook.length % D !== 0) throw new RangeError('VocabularyIndex: codebook length must be a positive multiple of 128');
    this.codebook = codebook instanceof Float64Array ? codebook : Float64Array.from(codebook);
    this.numWords = this.codebook.length / D;
    this.options = { ...DEFAULT_VOCABULARY_OPTIONS, ...opts, numWords: this.numWords };
    this.hn = new Float64Array(this.numWords);
    halfNorms(this.codebook, this.numWords, this.hn);
    this.df = new Uint32Array(this.numWords);
    this.idf = new Float64Array(this.numWords);
    this.counts = new Float64Array(this.numWords);
    this.qv = new Float64Array(this.numWords);
  }

  /** Train a codebook on (a subsample of) the given features and return an empty index. */
  static train(sources: readonly DescriptorSource[], opts: Partial<VocabularyOptions> = {}): VocabularyIndex {
    return new VocabularyIndex(trainVocabulary(sources, opts), opts);
  }

  /** Codebook as float32 (for persistence). */
  exportCodebook(): Float32Array {
    return Float32Array.from(this.codebook);
  }

  /** Number of indexed images. */
  get size(): number {
    return this.docs.size;
  }

  /** Indexed image ids, ascending. */
  imageIds(): number[] {
    return [...this.docs.keys()].sort((a, b) => a - b);
  }

  has(id: number): boolean {
    return this.docs.has(id);
  }

  /** Word id per feature (the selected subset when `maxFeaturesPerImage` > 0; see `selectFeatures`). */
  quantize(features: FeatureSet): Uint32Array {
    const sel = this.selectFeatures(features);
    const m = sel ? sel.length : features.count;
    const out = new Uint32Array(m);
    quantizeDescriptors(this.codebook, this.numWords, features.descriptors, features.count, out, sel ?? undefined, this.hn);
    return out;
  }

  /** Indices of the strongest `maxFeaturesPerImage` features (null = all). */
  selectFeatures(features: FeatureSet): Uint32Array | null {
    const cap = this.options.maxFeaturesPerImage;
    if (!(cap > 0) || features.count <= cap) return null;
    const idx = new Uint32Array(features.count);
    for (let i = 0; i < idx.length; i++) idx[i] = i;
    const sc = features.scores;
    idx.sort((a, b) => sc[b] - sc[a] || a - b);
    return idx.slice(0, cap).sort();
  }

  /** Sparse BoW (term frequencies after the burstiness transform, no IDF). */
  computeBow(features: FeatureSet): BowVector {
    return this.bowFromWords(this.quantize(features));
  }

  /** Sparse BoW from word ids. */
  bowFromWords(words: ArrayLike<number>): BowVector {
    const cnt = this.counts;
    let nnz = 0;
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      if (w >= this.numWords) continue;
      if (cnt[w] === 0) nnz++;
      cnt[w]++;
    }
    const ws = new Uint32Array(nnz);
    const tf = new Float32Array(nnz);
    const mode = this.options.burstiness;
    let j = 0;
    for (let w = 0; w < this.numWords && j < nnz; w++) {
      const c = cnt[w];
      if (c === 0) continue;
      ws[j] = w;
      tf[j] = mode === 'sqrt' ? Math.sqrt(c) : mode === 'binary' ? 1 : c;
      j++;
      cnt[w] = 0;
    }
    return { words: ws, tf };
  }

  /** Add (or replace) an image. */
  add(imageId: number, features: FeatureSet): void {
    this.addBow(imageId, this.computeBow(features));
  }

  /** Add (or replace) an image from a precomputed BoW. */
  addBow(imageId: number, bow: BowVector): void {
    this.remove(imageId);
    for (let i = 0; i < bow.words.length; i++) this.df[bow.words[i]]++;
    this.docs.set(imageId, { words: bow.words, tf: bow.tf, invNorm: 0 });
    this.dirty = true;
  }

  /** Remove an image; returns whether it was indexed. */
  remove(imageId: number): boolean {
    const d = this.docs.get(imageId);
    if (!d) return false;
    for (let i = 0; i < d.words.length; i++) this.df[d.words[i]]--;
    this.docs.delete(imageId);
    this.dirty = true;
    return true;
  }

  /** Stored BoW of an image. */
  getBow(imageId: number): BowVector | undefined {
    const d = this.docs.get(imageId);
    return d ? { words: d.words, tf: d.tf } : undefined;
  }

  private refresh(): void {
    if (!this.dirty) return;
    const N = this.docs.size;
    const lnN = Math.log(Math.max(N, 1));
    for (let w = 0; w < this.numWords; w++) {
      const n = this.df[w];
      this.idf[w] = n > 0 ? lnN - Math.log(n) : Math.log(N + 1);
    }
    for (const d of this.docs.values()) {
      let s = 0;
      for (let i = 0; i < d.words.length; i++) {
        const v = d.tf[i] * this.idf[d.words[i]];
        s += v * v;
      }
      d.invNorm = s > 0 ? 1 / Math.sqrt(s) : 0;
    }
    this.dirty = false;
  }

  /** Cosine similarity of the TF-IDF vectors of two indexed images (NaN if unknown). */
  similarity(a: number, b: number): number {
    const da = this.docs.get(a),
      db = this.docs.get(b);
    if (!da || !db) return NaN;
    this.refresh();
    let i = 0,
      j = 0,
      s = 0;
    while (i < da.words.length && j < db.words.length) {
      const wa = da.words[i],
        wb = db.words[j];
      if (wa === wb) {
        const idf = this.idf[wa];
        s += da.tf[i] * db.tf[j] * idf * idf;
        i++;
        j++;
      } else if (wa < wb) i++;
      else j++;
    }
    return s * da.invNorm * db.invNorm;
  }

  /**
   * Top-k most similar indexed images (cosine of L2-normalised TF-IDF vectors, descending;
   * ties by ascending id). `q` = indexed image id (excluded from its own results), a BoW or a
   * FeatureSet. `exclude` = ids to skip (iterable or predicate).
   */
  query(q: number | BowVector | FeatureSet, k: number, exclude?: Iterable<number> | ((id: number) => boolean)): RetrievalResult[] {
    const kk = Math.max(0, Math.floor(k));
    if (kk === 0 || this.docs.size === 0) return [];
    this.refresh();
    let bow: BowVector;
    let selfId: number | null = null;
    if (typeof q === 'number') {
      const d = this.docs.get(q);
      if (!d) return [];
      bow = d;
      selfId = q;
    } else if ('words' in q) bow = q;
    else bow = this.computeBow(q);
    const qv = this.qv;
    let qn = 0;
    for (let i = 0; i < bow.words.length; i++) {
      const w = bow.words[i];
      const v = bow.tf[i] * this.idf[w];
      qv[w] = v;
      qn += v * v;
    }
    const pred = typeof exclude === 'function' ? exclude : null;
    const set = exclude && typeof exclude !== 'function' ? new Set(exclude) : null;
    const bestS = new Float64Array(kk).fill(-Infinity);
    const bestId = new Float64Array(kk);
    let cnt = 0;
    if (qn > 0) {
      const qInv = 1 / Math.sqrt(qn);
      for (const [id, d] of this.docs) {
        if (id === selfId || (set && set.has(id)) || (pred && pred(id))) continue;
        let s = 0;
        const ws = d.words,
          tf = d.tf;
        for (let i = 0; i < ws.length; i++) {
          const w = ws[i];
          s += qv[w] * tf[i] * this.idf[w];
        }
        s *= qInv * d.invNorm;
        // insert if better than the current k-th (ties: smaller id first)
        if (cnt < kk || s > bestS[kk - 1] || (s === bestS[kk - 1] && id < bestId[kk - 1])) {
          let p = cnt < kk ? cnt++ : kk - 1;
          while (p > 0 && (bestS[p - 1] < s || (bestS[p - 1] === s && bestId[p - 1] > id))) {
            bestS[p] = bestS[p - 1];
            bestId[p] = bestId[p - 1];
            p--;
          }
          bestS[p] = s;
          bestId[p] = id;
        }
      }
    }
    for (let i = 0; i < bow.words.length; i++) qv[bow.words[i]] = 0;
    const out: RetrievalResult[] = [];
    for (let i = 0; i < cnt; i++) out.push({ id: bestId[i], score: bestS[i] });
    return out;
  }
}
