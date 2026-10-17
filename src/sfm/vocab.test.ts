import { describe, expect, it } from 'vitest';
import type { FeatureSet } from '../core/types';
import { Rng } from '../math/random';
import { Quantizer, quantizeDescriptors, trainVocabulary, VocabularyIndex } from './vocab';

const LOG = !!(globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env.SFM_EXTRAS_LOG;
const D = 128;

/** Random RootSIFT-like prototype: non-negative, unit L2 (float). */
function prototype(rng: Rng): Float64Array {
  const v = new Float64Array(D);
  let n = 0;
  for (let d = 0; d < D; d++) {
    const g = Math.max(0, rng.gaussian());
    v[d] = g * g;
    n += v[d] * v[d];
  }
  n = Math.sqrt(n);
  for (let d = 0; d < D; d++) v[d] /= n;
  return v;
}

/** Perturb a unit vector (relative noise σ), renormalise, quantise like RootSIFT (min(255, round(512 v))). */
function perturb(rng: Rng, p: Float64Array, sigma: number, out: Uint8Array, o: number): Float64Array {
  const v = new Float64Array(D);
  let n = 0;
  for (let d = 0; d < D; d++) {
    v[d] = Math.max(0, p[d] + sigma * rng.gaussian() * 0.09);
    n += v[d] * v[d];
  }
  n = Math.sqrt(n) || 1;
  for (let d = 0; d < D; d++) {
    v[d] /= n;
    out[o + d] = Math.min(255, Math.round(512 * v[d]));
  }
  return v;
}

function featureSet(desc: Uint8Array, scores?: Float32Array): FeatureSet {
  const count = desc.length / D;
  return { count, keypoints: new Float32Array(4 * count), descriptors: desc, scores: scores ?? new Float32Array(count).fill(1), width: 1920, height: 1440 };
}

interface Places {
  images: FeatureSet[];
  place: number[];
}

/** A loop: places 0..P−1 (3 images each), then revisits places 0..R−1 (3 images each). */
function makeLoop(rng: Rng, P: number, R: number, perImage = 300): Places {
  const protos = Array.from({ length: 600 }, () => prototype(rng));
  const common = Array.from({ length: 10 }, () => prototype(rng));
  // landmarks: (prototype + 0.6 noise) per place
  const land: Float64Array[][] = [];
  const tmp = new Uint8Array(D);
  for (let p = 0; p < P; p++) {
    const L: Float64Array[] = [];
    for (let l = 0; l < 250; l++) L.push(perturb(rng, protos[rng.int(protos.length)], 0.6, tmp, 0));
    land.push(L);
  }
  const images: FeatureSet[] = [];
  const place: number[] = [];
  const shoot = (p: number) => {
    const desc = new Uint8Array(perImage * D);
    for (let i = 0; i < perImage; i++) {
      const r = rng.next();
      const src = r < 0.7 ? land[p][rng.int(250)] : r < 0.85 ? common[rng.int(common.length)] : protos[rng.int(protos.length)];
      perturb(rng, src, 0.25, desc, i * D);
    }
    images.push(featureSet(desc));
    place.push(p);
  };
  for (let p = 0; p < P; p++) for (let k = 0; k < 3; k++) shoot(p);
  for (let p = 0; p < R; p++) for (let k = 0; k < 3; k++) shoot(p);
  return { images, place };
}

describe('quantizeDescriptors / trainVocabulary', () => {
  it('exact nearest centroid (incl. k not a multiple of 4)', () => {
    const rng = new Rng(1);
    const k = 37;
    const cb = new Float64Array(k * D);
    for (let i = 0; i < cb.length; i++) cb[i] = rng.uniform(0, 80);
    const n = 500;
    const desc = new Uint8Array(n * D);
    for (let i = 0; i < desc.length; i++) desc[i] = rng.int(160);
    const q = quantizeDescriptors(cb, k, desc, n);
    for (let i = 0; i < n; i++) {
      let best = Infinity,
        bi = -1;
      for (let c = 0; c < k; c++) {
        let d2 = 0;
        for (let d = 0; d < D; d++) d2 += (desc[i * D + d] - cb[c * D + d]) ** 2;
        if (d2 < best) {
          best = d2;
          bi = c;
        }
      }
      expect(q[i]).toBe(bi);
    }
    const sub = quantizeDescriptors(cb, k, desc, n, undefined, [5, 7, 400]);
    expect(Array.from(sub)).toEqual([q[5], q[7], q[400]]);
  });

  it('SIMD quantiser gives exactly the plain-JS result (rounding-error re-check)', () => {
    const rng = new Rng(11);
    for (const k of [37, 512]) {
      const cb = new Float64Array(k * D);
      // near-ties: pairs of almost identical centroids
      for (let c = 0; c < k; c++)
        for (let d = 0; d < D; d++) cb[c * D + d] = c % 2 === 1 ? cb[(c - 1) * D + d] + rng.uniform(-0.01, 0.01) : rng.uniform(0, 60);
      for (let i = 0; i < cb.length; i++) cb[i] = Math.min(255, Math.max(0, cb[i]));
      const n = 3001;
      const desc = new Uint8Array(n * D);
      for (let i = 0; i < desc.length; i++) desc[i] = rng.next() < 0.3 ? 0 : rng.int(120);
      const ref = quantizeDescriptors(cb, k, desc, n);
      const q = new Quantizer(cb, k, true);
      const got = q.quantize(desc, n);
      expect(Array.from(got)).toEqual(Array.from(ref));
      const sel = [3000, 1, 17, 2999, 5];
      expect(Array.from(q.quantize(desc, n, undefined, sel))).toEqual(sel.map((i) => ref[i]));
      expect(new Quantizer(cb, k, false).usesSimd).toBe(false);
      if (LOG) console.log('simd available:', q.usesSimd);
    }
    // out-of-range codebook → JS path
    const bad = new Float64Array(4 * D).fill(300);
    expect(new Quantizer(bad, 4, true).usesSimd).toBe(false);
  });

  it('k-means++ / Hamerly recovers well-separated clusters', () => {
    const rng = new Rng(2);
    const K = 64;
    const centers = Array.from({ length: K }, () => prototype(rng));
    const per = 40;
    const desc = new Uint8Array(K * per * D);
    const label = new Int32Array(K * per);
    for (let c = 0; c < K; c++)
      for (let j = 0; j < per; j++) {
        perturb(rng, centers[c], 0.15, desc, (c * per + j) * D);
        label[c * per + j] = c;
      }
    for (const simd of [true, false]) {
    const cb = trainVocabulary([{ descriptors: desc, count: K * per }], { numWords: K, trainingPerWord: per, seedPerWord: per, iterations: 20, simd });
    const q = quantizeDescriptors(cb, K, desc, K * per);
    // purity: majority word of each true cluster, and words used by one cluster only
    let pure = 0;
    const owner = new Map<number, Set<number>>();
    for (let c = 0; c < K; c++) {
      const cnt = new Map<number, number>();
      for (let j = 0; j < per; j++) cnt.set(q[c * per + j], (cnt.get(q[c * per + j]) ?? 0) + 1);
      const [w, m] = [...cnt.entries()].sort((a, b) => b[1] - a[1])[0];
      pure += m;
      if (!owner.has(w)) owner.set(w, new Set());
      owner.get(w)!.add(c);
    }
    expect(pure / (K * per)).toBeGreaterThan(0.97);
    expect(owner.size).toBeGreaterThanOrEqual(0.9 * K); // k-means may merge/split a few clusters (local minima)
    }
  });

  it('is deterministic and handles fewer descriptors than words', () => {
    const rng = new Rng(3);
    const desc = new Uint8Array(300 * D);
    for (let i = 0; i < 300; i++) perturb(rng, prototype(rng), 0.1, desc, i * D);
    const a = trainVocabulary([{ descriptors: desc, count: 300 }], { numWords: 32, seed: 9 });
    const b = trainVocabulary([{ descriptors: desc, count: 300 }], { numWords: 32, seed: 9 });
    expect(Array.from(a)).toEqual(Array.from(b));
    const small = trainVocabulary([{ descriptors: desc.subarray(0, 10 * D), count: 10 }], { numWords: 64 });
    expect(small.length).toBe(10 * D);
    expect(() => trainVocabulary([], {})).toThrow(RangeError);
  });
});

describe('VocabularyIndex', () => {
  it('finds revisited places in a synthetic loop (TF-IDF, incremental add)', () => {
    const rng = new Rng(4);
    const P = 24,
      R = 6;
    const { images, place } = makeLoop(rng, P, R);
    const firstVisit = images.slice(0, 3 * P);
    const t0 = performance.now();
    const index = VocabularyIndex.train(firstVisit, { numWords: 256 });
    const tTrain = performance.now() - t0;
    images.forEach((f, i) => index.add(i, f)); // incremental adds after training
    expect(index.size).toBe(images.length);
    let hits = 0,
      total = 0,
      top1 = 0;
    for (let i = 3 * P; i < images.length; i++) {
      const res = index.query(i, 3, (id) => Math.abs(id - i) <= 6);
      total++;
      if (res.some((r) => place[r.id] === place[i] && r.id < 3 * P)) hits++;
      if (place[res[0].id] === place[i]) top1++;
      for (let k = 1; k < res.length; k++) expect(res[k - 1].score).toBeGreaterThanOrEqual(res[k].score);
    }
    if (LOG) console.log(`loop retrieval recall@3 ${hits}/${total}, top-1 ${top1}/${total}, train ${tTrain.toFixed(0)} ms`);
    expect(hits / total).toBeGreaterThanOrEqual(0.95);
    expect(top1 / total).toBeGreaterThanOrEqual(0.9);
    // a query by FeatureSet scores like the stored image (self excluded explicitly)
    const q = index.query(images[3 * P], 2, [3 * P]);
    const qi = index.query(3 * P, 2);
    expect(q.map((r) => r.id)).toEqual(qi.map((r) => r.id));
    expect(q[0].score).toBeCloseTo(qi[0].score, 6);
    // similarity is symmetric and in [0, 1]
    const s1 = index.similarity(0, 1),
      s2 = index.similarity(1, 0);
    expect(s1).toBeCloseTo(s2, 12);
    expect(s1).toBeGreaterThan(0);
    expect(s1).toBeLessThanOrEqual(1 + 1e-9);
    expect(index.similarity(0, 0)).toBeCloseTo(1, 9);
    expect(Number.isNaN(index.similarity(0, 9999))).toBe(true);
  });

  it('add / remove / replace keep IDF consistent; exclusion forms', () => {
    const rng = new Rng(5);
    const { images } = makeLoop(rng, 6, 0, 150);
    const index = VocabularyIndex.train(images, { numWords: 64 });
    images.forEach((f, i) => index.add(10 + i, f));
    const before = index.query(10, 5).map((r) => r.id);
    index.add(99, images[3]);
    expect(index.query(10, 5).map((r) => r.id)).not.toEqual([]);
    expect(index.remove(99)).toBe(true);
    expect(index.remove(99)).toBe(false);
    expect(index.query(10, 5).map((r) => r.id)).toEqual(before);
    index.add(10, images[0]); // replace with identical content
    expect(index.query(10, 5).map((r) => r.id)).toEqual(before);
    const ex = index.query(10, 3, [before[0]]);
    expect(ex.map((r) => r.id)).not.toContain(before[0]);
    const ex2 = index.query(10, 3, (id) => id === before[1]);
    expect(ex2.map((r) => r.id)).not.toContain(before[1]);
    expect(index.imageIds()).toEqual(images.map((_, i) => 10 + i));
    expect(index.query(12345, 3)).toEqual([]);
    expect(index.query(10, 0)).toEqual([]);
  });

  it('burstiness: sqrt/binary tf limit the influence of repeated words', () => {
    const rng = new Rng(6);
    const { images } = makeLoop(rng, 4, 0, 200);
    const cb = trainVocabulary(images, { numWords: 64 });
    for (const mode of ['none', 'sqrt', 'binary'] as const) {
      const index = new VocabularyIndex(cb, { burstiness: mode });
      images.forEach((f, i) => index.add(i, f));
      const w = index.quantize(images[0]);
      // image with a single word repeated 200 times + 20 words of image 0
      const words = new Uint32Array(220);
      words.fill(w[0]);
      words.set(w.subarray(0, 20), 200);
      const bow = index.bowFromWords(words);
      const tfMax = Math.max(...bow.tf);
      if (mode === 'none') expect(tfMax).toBeGreaterThanOrEqual(200);
      if (mode === 'sqrt') expect(tfMax).toBeCloseTo(Math.sqrt(200 + w.subarray(0, 20).filter((x) => x === w[0]).length), 6);
      if (mode === 'binary') expect(tfMax).toBe(1);
    }
  });

  it('codebook export round trip and maxFeaturesPerImage subset', () => {
    const rng = new Rng(7);
    const { images } = makeLoop(rng, 4, 0, 200);
    const index = VocabularyIndex.train(images, { numWords: 48 });
    const copy = new VocabularyIndex(index.exportCodebook());
    expect(copy.numWords).toBe(48);
    const a = index.quantize(images[0]),
      b = copy.quantize(images[0]);
    let same = 0;
    for (let i = 0; i < a.length; i++) same += a[i] === b[i] ? 1 : 0;
    expect(same / a.length).toBeGreaterThan(0.995); // float32 rounding of the centroids
    const scores = new Float32Array(images[0].count);
    for (let i = 0; i < scores.length; i++) scores[i] = (i * 37) % 101;
    const fs = { ...images[0], scores };
    const capped = new VocabularyIndex(index.codebook, { maxFeaturesPerImage: 50 });
    const sel = capped.selectFeatures(fs)!;
    expect(sel.length).toBe(50);
    const minSel = Math.min(...Array.from(sel, (i) => scores[i]));
    let above = 0;
    for (let i = 0; i < scores.length; i++) if (scores[i] > minSel) above++;
    expect(above).toBeLessThanOrEqual(50);
    expect(capped.quantize(fs).length).toBe(50);
    expect(() => new VocabularyIndex(new Float32Array(100))).toThrow(RangeError);
  });

  it('timing: 4000 features × 512 words, training on a subsample', () => {
    const rng = new Rng(8);
    const protos = Array.from({ length: 2000 }, () => prototype(rng));
    const sets: FeatureSet[] = [];
    for (let s = 0; s < 12; s++) {
      const desc = new Uint8Array(4000 * D);
      for (let i = 0; i < 4000; i++) perturb(rng, protos[rng.int(protos.length)], 0.3, desc, i * D);
      sets.push(featureSet(desc));
    }
    let t0 = performance.now();
    const index = VocabularyIndex.train(sets, { numWords: 512 });
    const tTrain = performance.now() - t0;
    t0 = performance.now();
    const jsIndex = VocabularyIndex.train(sets, { numWords: 512, simd: false });
    const tTrainJs = performance.now() - t0;
    t0 = performance.now();
    jsIndex.add(0, sets[0]);
    const tQJs = performance.now() - t0;
    t0 = performance.now();
    for (let s = 0; s < 4; s++) index.add(s, sets[s]);
    const tQ = (performance.now() - t0) / 4;
    for (let s = 4; s < 12; s++) index.add(s, sets[s]);
    t0 = performance.now();
    for (let q = 0; q < 100; q++) index.query(q % 12, 10);
    const tQuery = (performance.now() - t0) / 100;
    if (LOG)
      console.log(
        `train k=512 on ${16 * 512} descriptors: ${tTrain.toFixed(0)} ms (JS ${tTrainJs.toFixed(0)} ms); add (4000 features): ${tQ.toFixed(0)} ms (JS ${tQJs.toFixed(0)} ms); query (12 images): ${tQuery.toFixed(3)} ms`,
      );
    expect(tTrain).toBeLessThan(20000);
    expect(tQ).toBeLessThan(3000);
  });
});
