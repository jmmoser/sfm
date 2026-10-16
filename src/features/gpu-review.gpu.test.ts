// Adversarial review tests of the WebGPU feature module (SiftGPU, MatcherGPU, pipeline).
// Each test targets a specific failure mode found or suspected in review; the defect it
// pins is named in the test title. Headless Chromium + SwiftShader (CPU emulation):
// sizes are kept small, timings are not asserted.
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import type { FeatureSet, GrayImage, Mat3, MatchPairs } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';
import { getGPU } from '../gpu/device';
import type { GPUContext } from '../gpu/device';
import { createFeaturePipeline } from './extractor';
import { GPUFeatureError, GPUWorkDroppedError, compilePipeline } from './gpu-common';
import { rgbaToGray } from './image';
import { matchCPU, matchGuidedCPU } from './match';
import type { MatchGuide, MatchOptions } from './match';
import { MatcherGPU } from './match-gpu';
import type { SiftOptions } from './sift-common';
import { extractSiftCPU } from './sift-cpu';
import { SiftGPU } from './sift-gpu';
import { featureParity } from './sift-gpu-host';
import { TestRng, makeTexturedImage } from './testdata';

const log = (...a: unknown[]): void => console.log('[gpu-review]', ...a);
const D = DESCRIPTOR_SIZE;

let ctx: GPUContext;
/** Uncaptured WebGPU errors on the shared device (must stay empty: every error must be attributed). */
const uncaptured: string[] = [];

beforeAll(async () => {
  const c = await getGPU();
  if (!c) throw new Error('WebGPU unavailable in the test browser');
  ctx = c;
  c.device.addEventListener('uncapturederror', (ev) => uncaptured.push((ev as GPUUncapturedErrorEvent).error.message));
});

afterAll(() => {
  expect(uncaptured).toEqual([]);
});

function sameFeatures(a: FeatureSet, b: FeatureSet): boolean {
  return (
    a.count === b.count &&
    a.width === b.width &&
    a.height === b.height &&
    a.keypoints.every((v, i) => v === b.keypoints[i]) &&
    a.descriptors.every((v, i) => v === b.descriptors[i]) &&
    a.scores.every((v, i) => v === b.scores[i])
  );
}

function toU8(img: GrayImage): Uint8Array {
  const d = new Uint8Array(img.width * img.height);
  for (let i = 0; i < d.length; i++) d[i] = Math.round((img.data[i] as number) * 255);
  return d;
}

function sameMatches(x: MatchPairs, y: MatchPairs): boolean {
  return x.length === y.length && x.every((v, i) => v === y[i]);
}

/** Random RootSIFT-like descriptors; b holds noisy copies of a permutation of a. */
function randomSets(na: number, nb: number, noise: number, seed: number, size = 600): { a: FeatureSet; b: FeatureSet } {
  const rng = new TestRng(seed);
  const mk = (n: number): FeatureSet => ({
    count: n,
    keypoints: new Float32Array(4 * n),
    descriptors: new Uint8Array(D * n),
    scores: new Float32Array(n),
    width: size,
    height: size,
  });
  const a = mk(na);
  const b = mk(nb);
  const v = new Float64Array(D);
  const quant = (out: Uint8Array, o: number) => {
    let ss = 0;
    for (let k = 0; k < D; k++) ss += v[k] * v[k];
    const inv = 1 / Math.sqrt(ss);
    for (let k = 0; k < D; k++) out[o + k] = Math.min(255, Math.round(512 * v[k] * inv));
  };
  for (let i = 0; i < na; i++) {
    for (let k = 0; k < D; k++) v[k] = rng.next() < 0.3 ? 0 : rng.next();
    quant(a.descriptors, i * D);
    a.keypoints[4 * i] = rng.uniform(0, size);
    a.keypoints[4 * i + 1] = rng.uniform(0, size);
  }
  for (let j = 0; j < nb; j++) {
    const src = j < na ? (j * 7919) % na : -1;
    if (src >= 0 && j < Math.floor(nb * 0.7)) {
      for (let k = 0; k < D; k++) v[k] = Math.max(0, a.descriptors[src * D + k] / 512 + noise * rng.gaussian());
      b.keypoints[4 * j] = a.keypoints[4 * src] + rng.uniform(-1, 1);
      b.keypoints[4 * j + 1] = a.keypoints[4 * src + 1] + rng.uniform(-1, 1);
    } else {
      for (let k = 0; k < D; k++) v[k] = rng.next() < 0.3 ? 0 : rng.next();
      b.keypoints[4 * j] = rng.uniform(0, size);
      b.keypoints[4 * j + 1] = rng.uniform(0, size);
    }
    quant(b.descriptors, j * D);
  }
  return { a, b };
}

/**
 * Guides exercising every kernel variant, consistent with randomSets (b ≈ a ± 1 px):
 * positions with NaN / Inf / out-of-frame entries, a negatively scaled near-identity H,
 * and F = [e]x (epipolar lines through a finite epipole inside the image and x_a).
 */
function guidesFor(a: FeatureSet): [string, MatchGuide][] {
  const pred = new Float32Array(2 * a.count);
  for (let i = 0; i < a.count; i++) {
    pred[2 * i] = a.keypoints[4 * i] + 2;
    pred[2 * i + 1] = a.keypoints[4 * i + 1] - 1;
  }
  pred[0] = NaN;
  pred[3] = Infinity;
  pred[5] = -900;
  const H = new Float64Array([1, 0.01, 3, -0.01, 1, 2, 1e-6, -1e-6, 1]);
  const e = [300.3, 280.7, 1];
  const F = new Float64Array([0, -e[2], e[1], e[2], 0, -e[0], -e[1], e[0], 0]);
  return [
    ['positions r=25', { kind: 'positions', predicted: pred, radiusPx: 25 }],
    ['-H', { kind: 'H', H: H.map((v) => -v) as Mat3, maxErrorPx: 30 }],
    ['F', { kind: 'F', F, maxErrorPx: 6 }],
  ];
}

// ---------------------------------------------------------------------------
// SiftGPU
// ---------------------------------------------------------------------------

describe('SiftGPU review: non-default options and odd geometry stay CPU-equivalent', () => {
  // Every option that changes kernel tables, plane counts, dispatch shapes or the fixed-point
  // scale: S (planes, detect z-range, kernel radii up to 45 > tiled-V limit 24), sigma below
  // the base blur (copy path instead of a blur), firstOctave > 0 (folded downscale),
  // maxOrientations 4, distribute off, odd / elongated sizes (odd octave sizes, 3 octaves only).
  const cases: [string, Partial<SiftOptions>, number, number, 'f32' | 'u8'][] = [
    ['S=2', { scalesPerOctave: 2 }, 211, 157, 'f32'],
    ['S=4, 4 orientations', { scalesPerOctave: 4, maxOrientations: 4 }, 240, 180, 'u8'],
    ['S=1 (radius 45)', { scalesPerOctave: 1 }, 180, 136, 'f32'],
    ['sigma 0.9 (< base blur 1.0: copy path)', { sigma: 0.9 }, 200, 150, 'f32'],
    ['firstOctave 1 (folded downscale)', { firstOctave: 1 }, 480, 360, 'u8'],
    ['distribute off, 300 features', { distribute: false, maxFeatures: 300 }, 301, 199, 'f32'],
    ['elongated 700x40', {}, 700, 40, 'f32'],
  ];
  for (const [name, opts, W, H, kind] of cases) {
    it(name, async () => {
      const ex = await SiftGPU.create(ctx, opts, { blur: 'tiled' });
      try {
        const f = makeTexturedImage(W, H, W + H);
        const img: GrayImage = kind === 'u8' ? { width: W, height: H, data: toU8(f) } : f;
        const g = await ex.extract(img);
        const c = extractSiftCPU(img, opts);
        const p = featureParity(c, g);
        const q = featureParity(g, c);
        log(`${name} ${W}x${H} ${kind}: GPU ${g.count} / CPU ${c.count}, pos ${p.posFrac.toFixed(4)} (rev ${q.posFrac.toFixed(4)}), median cos ${p.medianCos.toFixed(5)}`);
        expect(c.count).toBeGreaterThan(20);
        expect(Math.abs(g.count - c.count)).toBeLessThanOrEqual(Math.max(3, 0.02 * c.count));
        expect(p.posFrac).toBeGreaterThanOrEqual(0.97);
        expect(q.posFrac).toBeGreaterThanOrEqual(0.97);
        expect(p.medianCos).toBeGreaterThanOrEqual(0.995);
        expect(g.count).toBeLessThanOrEqual(ex.options.maxFeatures);
        for (let i = 0; i < g.count; i++) {
          const th = g.keypoints[4 * i + 3];
          expect(th >= 0 && th < 2 * Math.PI + 1e-6).toBe(true);
        }
      } finally {
        ex.destroy();
      }
    });
  }

  it('tiled and direct blurs are bit-identical at odd sizes and with radius > tiled-V limit (S=1)', async () => {
    for (const opts of [{}, { scalesPerOctave: 1 }] as Partial<SiftOptions>[]) {
      const t = await SiftGPU.create(ctx, opts, { blur: 'tiled' });
      const d = await SiftGPU.create(ctx, opts, { blur: 'direct' });
      try {
        const img = makeTexturedImage(301, 173, 77);
        const a = await t.extract(img);
        const b = await d.extract(img);
        expect(a.count).toBeGreaterThan(20);
        expect(sameFeatures(a, b)).toBe(true);
      } finally {
        t.destroy();
        d.destroy();
      }
    }
  });

  it('many small command buffers (tiny submit budget) give bit-identical results', async () => {
    const img = makeTexturedImage(200, 150, 5);
    const ref = await SiftGPU.create(ctx);
    const small = await SiftGPU.create(ctx, {}, { submitBudget: 1e5 });
    try {
      const a = await ref.extract(img);
      const b = await small.extract(img);
      log(`submits: default ${ref.lastTimings!.submits}, tiny budget ${small.lastTimings!.submits}`);
      expect(small.lastTimings!.submits).toBeGreaterThan(20);
      expect(small.lastTimings!.attempts).toBe(1);
      expect(sameFeatures(a, b)).toBe(true);
    } finally {
      ref.destroy();
      small.destroy();
    }
  });

  it('Uint8 data at an odd byte offset / Uint8ClampedArray upload correctly; other typed arrays are input errors', async () => {
    const ex = await SiftGPU.create(ctx, { maxFeatures: 500 });
    try {
      const W = 157;
      const H = 99; // W*H odd: exercises the tail upload too
      const u8 = toU8(makeTexturedImage(W, H, 9));
      const ref = await ex.extract({ width: W, height: H, data: u8 });
      expect(ref.count).toBeGreaterThan(20);
      const backing = new Uint8Array(u8.length + 3);
      backing.set(u8, 3);
      const odd = await ex.extract({ width: W, height: H, data: backing.subarray(3) });
      expect(sameFeatures(ref, odd)).toBe(true);
      const clamped = await ex.extract({ width: W, height: H, data: new Uint8ClampedArray(u8) as unknown as Uint8Array });
      expect(sameFeatures(ref, clamped)).toBe(true);
      const f64 = { width: W, height: H, data: Float64Array.from(u8, (v) => v / 255) as unknown as Float32Array };
      const err = await ex.extract(f64).then(
        () => null,
        (e: unknown) => e,
      );
      expect(err).toBeInstanceOf(Error);
      expect(err).not.toBeInstanceOf(GPUFeatureError);
      expect(sameFeatures(ref, await ex.extract({ width: W, height: H, data: u8 }))).toBe(true);
    } finally {
      ex.destroy();
    }
  });

  it('selfTest passes with maxFeatures < 10 (it used to require > 10 CPU features)', async () => {
    const ex = await SiftGPU.create(ctx, { maxFeatures: 8 });
    try {
      const r = await ex.selfTest();
      log(`selfTest maxFeatures 8: ${JSON.stringify(r)}`);
      expect(r.ok).toBe(true);
    } finally {
      ex.destroy();
    }
  });

  it('releaseCaches() / destroy() during a running extraction: the extraction completes correctly, no GPU errors', async () => {
    const img = makeTexturedImage(240, 180, 13);
    const ref = await SiftGPU.create(ctx);
    const expected = await ref.extract(img);
    ref.destroy();
    // Many command buffers so that later ones still use the pyramid when the caches are freed.
    const ex = await SiftGPU.create(ctx, {}, { submitBudget: 1e5 });
    try {
      await ex.extract(img); // warm: pyramid cached
      const cached = ex.memoryBytes();
      let done = false;
      const running = ex.extract(img).finally(() => (done = true));
      await new Promise((r) => setTimeout(r, 30));
      expect(done).toBe(false); // really in flight
      ex.releaseCaches();
      expect(sameFeatures(await running, expected)).toBe(true);
      log(`memory: cached ${cached}, after release ${ex.memoryBytes()}`);
      expect(ex.memoryBytes()).toBeLessThan(cached - 4e6); // caches freed once it ended
      expect(sameFeatures(await ex.extract(img), expected)).toBe(true);
      let done2 = false;
      const running2 = ex.extract(img).finally(() => (done2 = true));
      await new Promise((r) => setTimeout(r, 30));
      expect(done2).toBe(false);
      ex.destroy();
      expect(sameFeatures(await running2, expected)).toBe(true);
      await expect(ex.extract(img)).rejects.toBeInstanceOf(GPUFeatureError);
    } finally {
      ex.destroy();
    }
    expect(uncaptured).toEqual([]);
  });

  it('a detached ImageData is an input error on both backends (Chromium crashed the renderer on it)', async () => {
    const W = 64;
    const H = 48;
    const id = new ImageData(W, H);
    // Detach the pixel buffer (as after transferring it to a worker): the ImageData keeps
    // its size but has no data. copyExternalImageToTexture on it crashes Chromium's renderer.
    structuredClone(id.data.buffer, { transfer: [id.data.buffer] });
    expect(id.data.length).toBe(0);
    const fp = await createFeaturePipeline({ sift: { maxFeatures: 300 }, selfTest: false });
    const cpu = await createFeaturePipeline({ sift: { maxFeatures: 300 }, preferGPU: false });
    try {
      expect(fp.gpu).toBe(true);
      for (const p of [fp, cpu]) {
        const err = await p.extractFromBitmap(id).then(
          () => null,
          (e: unknown) => e,
        );
        expect(err).toBeInstanceOf(Error);
        expect(err).not.toBeInstanceOf(GPUFeatureError);
        expect((err as Error).message).toMatch(/detached/);
      }
      expect(fp.gpu).toBe(true);
      expect(fp.fallbackReason).toBeNull();
    } finally {
      fp.destroy();
      cpu.destroy();
    }
  });
});

// ---------------------------------------------------------------------------
// MatcherGPU
// ---------------------------------------------------------------------------

describe('MatcherGPU review', () => {
  it('multi-pass / multi-submission (rowTile0 > 0) and extreme column chunking equal the CPU', async () => {
    const { a, b } = randomSets(203, 141, 0.1, 3);
    const configs = [
      { pairBudget: 64 * 141, targetWorkgroups: 128 }, // one row tile per pass and per command buffer
      { pairBudget: 64 * 141 * 2, targetWorkgroups: 1 }, // 2 row tiles per pass, single column chunk
      { targetWorkgroups: 100000 }, // one column tile per chunk
    ];
    for (const cfg of configs) {
      const m = await MatcherGPU.create(ctx, cfg);
      try {
        for (const opts of [{}, { mutual: false }, { ratio: 0.95, maxDistance: 1.4 }] as MatchOptions[]) {
          expect(sameMatches(await m.match(a, b, opts), matchCPU(a, b, opts))).toBe(true);
          expect(sameMatches(await m.match(b, a, opts), matchCPU(b, a, opts))).toBe(true);
        }
        log(`cfg ${JSON.stringify(cfg)}: ${m.lastTimings!.submits} submits`);
        if (cfg.pairBudget) expect(m.lastTimings!.submits).toBeGreaterThan(2);
        for (const [name, g] of guidesFor(a)) {
          const cpu = matchGuidedCPU(a, b, g);
          expect(cpu.length).toBeGreaterThan(0);
          expect(sameMatches(await m.matchGuided(a, b, g), cpu), `${name} ${JSON.stringify(cfg)}`).toBe(true);
        }
      } finally {
        m.destroy();
      }
    }
  });

  it('exact ties (duplicate / all-zero descriptors) resolve like matchCPU, also with ratio > 1', async () => {
    const { a, b } = randomSets(90, 100, 0.05, 11);
    // duplicates: b[10] = b[40] = b[70] (exact ties for whoever matches them), a[5] = a[6]
    for (const j of [40, 70]) b.descriptors.copyWithin(j * D, 10 * D, 11 * D);
    a.descriptors.copyWithin(6 * D, 5 * D, 6 * D);
    // all-zero descriptors in both sets (invNorm 0 -> similarity 0 for every pair)
    a.descriptors.fill(0, 20 * D, 21 * D);
    b.descriptors.fill(0, 30 * D, 32 * D);
    const m = await MatcherGPU.create(ctx);
    try {
      for (const opts of [{}, { ratio: 1.5 }, { ratio: 1.5, maxDistance: 2 }, { ratio: 1.5, maxDistance: 2, mutual: false }] as MatchOptions[]) {
        const cpu = matchCPU(a, b, opts);
        expect(sameMatches(await m.match(a, b, opts), cpu), JSON.stringify(opts)).toBe(true);
      }
      // guided with all predictions unknown = brute force over the same candidate sets
      const allNaN: MatchGuide = { kind: 'positions', predicted: new Float32Array(2 * a.count).fill(NaN), radiusPx: 3 };
      expect(sameMatches(await m.matchGuided(a, b, allNaN, { ratio: 1.5, maxDistance: 2 }), matchCPU(a, b, { ratio: 1.5, maxDistance: 2 }))).toBe(true);
      // radius 0: every tile culled except unknown rows
      const pred = new Float32Array(2 * a.count).fill(-1e4);
      pred[2 * 7] = NaN;
      const g0: MatchGuide = { kind: 'positions', predicted: pred, radiusPx: 0 };
      expect(sameMatches(await m.matchGuided(a, b, g0), matchGuidedCPU(a, b, g0))).toBe(true);
    } finally {
      m.destroy();
    }
  });

  it('malformed FeatureSets are input errors: no GPUFeatureError, matcher and pipeline keep the GPU', async () => {
    const { a, b } = randomSets(70, 80, 0.1, 21);
    const shortDesc: FeatureSet = { ...b, descriptors: b.descriptors.subarray(0, 50 * D) };
    const shortKp: FeatureSet = { ...a, keypoints: a.keypoints.subarray(0, 10) };
    const m = await MatcherGPU.create(ctx);
    try {
      for (const p of [
        m.match(a, shortDesc),
        m.matchGuided(shortKp, b, { kind: 'H', H: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), maxErrorPx: 4 }),
        m.match({ ...a, count: 1.5 }, b),
      ]) {
        const err = await p.then(
          () => null,
          (e: unknown) => e,
        );
        expect(err).toBeInstanceOf(Error);
        expect(err).not.toBeInstanceOf(GPUFeatureError);
      }
      expect(sameMatches(await m.match(a, b), matchCPU(a, b))).toBe(true);
    } finally {
      m.destroy();
    }
    const fp = await createFeaturePipeline({ selfTest: false });
    try {
      await expect(fp.match(a, shortDesc)).rejects.toThrow(/descriptor/);
      expect(fp.gpu).toBe(true);
      expect(sameMatches(await fp.match(a, b), matchCPU(a, b))).toBe(true);
    } finally {
      fp.destroy();
    }
  });

  it('clearCache() / destroy() during a running match: correct result, no GPU errors', async () => {
    const { a, b } = randomSets(400, 380, 0.1, 31);
    const expected = matchCPU(a, b);
    const m = await MatcherGPU.create(ctx, { pairBudget: 64 * 380 });
    try {
      let done = false;
      const running = m.match(a, b).finally(() => (done = true));
      await new Promise((r) => setTimeout(r, 20));
      expect(done).toBe(false); // really in flight
      m.clearCache();
      expect(sameMatches(await running, expected)).toBe(true);
      expect(m.cacheMemoryBytes()).toBe(0);
      let done2 = false;
      const running2 = m.match(a, b).finally(() => (done2 = true));
      await new Promise((r) => setTimeout(r, 20));
      expect(done2).toBe(false);
      m.destroy();
      expect(sameMatches(await running2, expected)).toBe(true);
      await expect(m.match(a, b)).rejects.toBeInstanceOf(GPUFeatureError);
    } finally {
      m.destroy();
    }
    expect(uncaptured).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Shared helpers + pipeline
// ---------------------------------------------------------------------------

describe('GPU errors are hard failures, not dropped work', () => {
  // An invalid command buffer is not executed, so the work stamp also reports a mismatch;
  // before the fix the scope's validation error was ignored in favour of that mismatch:
  // 3 retries, then a *transient* GPUWorkDroppedError, so the pipeline kept the broken GPU
  // backend forever (every call: 3 failing GPU attempts, then the CPU).
  it('SiftGPU / MatcherGPU: one attempt, GPUFeatureError (not GPUWorkDroppedError)', async () => {
    const attempts: number[] = [];
    const ex = await SiftGPU.create(ctx, { maxFeatures: 300 }, { testDropSubmissions: (a) => (attempts.push(a), []) });
    const img = makeTexturedImage(160, 120, 3);
    try {
      expect((await ex.extract(img)).count).toBeGreaterThan(20);
      attempts.length = 0;
      (ex as unknown as { kernBuf: GPUBuffer }).kernBuf.destroy(); // every blur now uses a destroyed buffer
      const err = await ex.extract(img).then(
        () => null,
        (e: unknown) => e,
      );
      log(`SiftGPU with a destroyed internal buffer: ${(err as Error)?.name}: ${(err as Error)?.message?.slice(0, 100)}`);
      expect(err).toBeInstanceOf(GPUFeatureError);
      expect(err).not.toBeInstanceOf(GPUWorkDroppedError);
      expect(attempts).toEqual([0]);
    } finally {
      ex.destroy();
    }
    const { a, b } = randomSets(100, 90, 0.1, 5);
    const mAttempts: number[] = [];
    const m = await MatcherGPU.create(ctx, { testDropSubmissions: (t) => (mAttempts.push(t), []) });
    try {
      expect(sameMatches(await m.match(a, b), matchCPU(a, b))).toBe(true);
      mAttempts.length = 0;
      (m as unknown as { out: { buffer: GPUBuffer } }).out.buffer.destroy();
      const err = await m.match(a, b).then(
        () => null,
        (e: unknown) => e,
      );
      expect(err).toBeInstanceOf(GPUFeatureError);
      expect(err).not.toBeInstanceOf(GPUWorkDroppedError);
      expect(mAttempts).toEqual([0]);
    } finally {
      m.destroy();
    }
  });

  it('the pipeline falls back to the CPU for good (once) on such an error', async () => {
    const reasons: string[] = [];
    const opts = { maxFeatures: 300 };
    const fp = await createFeaturePipeline({ sift: opts, selfTest: false, onFallback: (r) => reasons.push(r) });
    try {
      expect(fp.gpu).toBe(true);
      const img = makeTexturedImage(160, 120, 4);
      (fp as unknown as { sift: { kernBuf: GPUBuffer } }).sift.kernBuf.destroy();
      expect(sameFeatures(await fp.extract(img, 160, 120), extractSiftCPU(img, opts, 160, 120))).toBe(true);
      expect(fp.gpu).toBe(false);
      expect(sameFeatures(await fp.extract(img, 160, 120), extractSiftCPU(img, opts, 160, 120))).toBe(true);
      expect(reasons.length).toBe(1);
      log(`fallback reason: ${reasons[0].slice(0, 120)}`);
    } finally {
      fp.destroy();
    }
  });
});

describe('gpu-common / pipeline review', () => {
  it('concurrent pipeline compilation attributes a shader error to its own pipeline only', async () => {
    const good = (i: number) => `@group(0) @binding(0) var<storage, read_write> s: array<u32>;
@compute @workgroup_size(1) fn main() { s[0] = ${i}u; }`;
    const bad = '@compute @workgroup_size(1) fn main() { let x: u32 = 1.5; }';
    const r = await Promise.allSettled([
      compilePipeline(ctx.device, 'good-0', good(0), ctx.device.lost),
      compilePipeline(ctx.device, 'bad', bad, ctx.device.lost),
      compilePipeline(ctx.device, 'good-1', good(1), ctx.device.lost),
      compilePipeline(ctx.device, 'good-2', good(2), ctx.device.lost),
    ]);
    log(r.map((x) => (x.status === 'fulfilled' ? 'ok' : (x.reason as Error).message.slice(0, 80))).join(' | '));
    expect(r.map((x) => x.status)).toEqual(['fulfilled', 'rejected', 'fulfilled', 'fulfilled']);
    expect((r[1] as PromiseRejectedResult).reason).toBeInstanceOf(GPUFeatureError);
    expect(((r[1] as PromiseRejectedResult).reason as Error).message).toMatch(/bad/);
  });

  it('extractFromBitmap with only fullWidth: the CPU fallback uses the same (aspect-derived) frame as the GPU', async () => {
    const W = 200;
    const H = 150;
    const gray = makeTexturedImage(W, H, 3);
    const id = new ImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      const v = Math.round(255 * (gray.data[i] as number));
      id.data.set([v, v, v, 255], 4 * i);
    }
    const opts = { maxFeatures: 400 };
    const gpu = await createFeaturePipeline({ sift: opts, selfTest: false });
    const cpu = await createFeaturePipeline({ sift: opts, preferGPU: false });
    try {
      expect(gpu.gpu).toBe(true);
      const g = await gpu.extractFromBitmap(id, 800);
      const c = await cpu.extractFromBitmap(id, 800);
      expect([g.width, g.height]).toEqual([800, 600]);
      expect([c.width, c.height]).toEqual([800, 600]);
      expect(sameFeatures(c, extractSiftCPU(rgbaToGray(id), opts, 800, 600))).toBe(true);
      const p = featureParity(c, g);
      expect(p.posFrac).toBeGreaterThan(0.95);
      const c2 = await cpu.extractFromBitmap(id, undefined, 300);
      expect([c2.width, c2.height]).toEqual([400, 300]);
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  });

  it('concurrent extract / match / matchGuided through one pipeline equal the sequential results', async () => {
    const opts = { maxFeatures: 500 };
    const fp = await createFeaturePipeline({ sift: opts, selfTest: false });
    try {
      const i1 = makeTexturedImage(200, 150, 41);
      const i2 = makeTexturedImage(200, 150, 42);
      const { a, b } = randomSets(150, 160, 0.1, 43);
      const [g] = guidesFor(a);
      const [f1, m1, f2, m2] = await Promise.all([fp.extract(i1, 200, 150), fp.match(a, b), fp.extract(i2, 200, 150), fp.matchGuided(a, b, g[1])]);
      expect(fp.gpu).toBe(true);
      expect(sameFeatures(f1, await fp.extract(i1, 200, 150))).toBe(true);
      expect(sameFeatures(f2, await fp.extract(i2, 200, 150))).toBe(true);
      expect(sameMatches(m1, matchCPU(a, b))).toBe(true);
      expect(sameMatches(m2, matchGuidedCPU(a, b, g[1]))).toBe(true);
    } finally {
      fp.destroy();
    }
    expect(uncaptured).toEqual([]);
  });
});
