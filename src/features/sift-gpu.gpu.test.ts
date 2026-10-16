// WebGPU SIFT vs the CPU reference (headless Chromium + SwiftShader WebGPU).
// SwiftShader is a CPU emulation: timings are informational only.
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import type { FeatureSet, GrayImage, Mat3 } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';
import { getGPU } from '../gpu/device';
import type { GPUContext } from '../gpu/device';
import { GPUFeatureError, GPUWorkDroppedError } from './gpu-common';
import { rgbaToGray } from './image';
import { matchCPU } from './match';
import { MatcherGPU } from './match-gpu';
import { extractSiftCPU } from './sift-cpu';
import type { SiftOptions } from './sift-common';
import { SiftGPU } from './sift-gpu';
import { evaluateMatches, makeTexturedImage, makeWarpedPair, similarityAboutCenter, tiltHomography } from './testdata';

const log = (...a: unknown[]): void => console.log('[sift-gpu]', ...a);

interface Parity {
  /** CPU keypoints with a GPU keypoint within 1 px and 20 % scale. */
  posFrac: number;
  /** ... and the same orientation (within 0.2 rad): descriptor correspondences. */
  pairs: number;
  medianCos: number;
  p05Cos: number;
  identicalDesc: number;
  maxPosErr: number;
}

function angDiff(a: number, b: number): number {
  let d = Math.abs(a - b) % (2 * Math.PI);
  if (d > Math.PI) d = 2 * Math.PI - d;
  return d;
}

function cosine(a: Uint8Array, ao: number, b: Uint8Array, bo: number): number {
  let d = 0;
  let na = 0;
  let nb = 0;
  for (let k = 0; k < DESCRIPTOR_SIZE; k++) {
    d += a[ao + k] * b[bo + k];
    na += a[ao + k] * a[ao + k];
    nb += b[bo + k] * b[bo + k];
  }
  return na > 0 && nb > 0 ? d / Math.sqrt(na * nb) : na === nb ? 1 : 0;
}

function parity(cpu: FeatureSet, gpu: FeatureSet): Parity {
  // grid over GPU keypoints
  const cs = 4;
  const gw = Math.ceil(gpu.width / cs) + 2;
  const gh = Math.ceil(gpu.height / cs) + 2;
  const cells: number[][] = Array.from({ length: gw * gh }, () => []);
  const cell = (x: number, y: number) => Math.min(gh - 1, Math.max(0, Math.floor(y / cs) + 1)) * gw + Math.min(gw - 1, Math.max(0, Math.floor(x / cs) + 1));
  for (let j = 0; j < gpu.count; j++) cells[cell(gpu.keypoints[4 * j], gpu.keypoints[4 * j + 1])].push(j);
  let pos = 0;
  const cos: number[] = [];
  let identical = 0;
  let maxErr = 0;
  for (let i = 0; i < cpu.count; i++) {
    const x = cpu.keypoints[4 * i];
    const y = cpu.keypoints[4 * i + 1];
    const s = cpu.keypoints[4 * i + 2];
    const th = cpu.keypoints[4 * i + 3];
    let found = false;
    let bestJ = -1;
    let bestA = Infinity;
    let bestD = Infinity;
    const cx = Math.floor(x / cs) + 1;
    const cy = Math.floor(y / cs) + 1;
    for (let yy = cy - 1; yy <= cy + 1; yy++)
      for (let xx = cx - 1; xx <= cx + 1; xx++) {
        if (yy < 0 || yy >= gh || xx < 0 || xx >= gw) continue;
        for (const j of cells[yy * gw + xx]) {
          const d = Math.hypot(gpu.keypoints[4 * j] - x, gpu.keypoints[4 * j + 1] - y);
          const r = gpu.keypoints[4 * j + 2] / s;
          if (d > 1 || r < 1 / 1.2 || r > 1.2) continue;
          found = true;
          const a = angDiff(gpu.keypoints[4 * j + 3], th);
          if (a < bestA) {
            bestA = a;
            bestJ = j;
            bestD = d;
          }
        }
      }
    if (!found) continue;
    pos++;
    if (bestA < 0.2) {
      maxErr = Math.max(maxErr, bestD);
      const c = cosine(cpu.descriptors, i * DESCRIPTOR_SIZE, gpu.descriptors, bestJ * DESCRIPTOR_SIZE);
      cos.push(c);
      let same = true;
      for (let k = 0; k < DESCRIPTOR_SIZE && same; k++) same = cpu.descriptors[i * DESCRIPTOR_SIZE + k] === gpu.descriptors[bestJ * DESCRIPTOR_SIZE + k];
      if (same) identical++;
    }
  }
  cos.sort((a, b) => a - b);
  return {
    posFrac: cpu.count ? pos / cpu.count : 1,
    pairs: cos.length,
    medianCos: cos.length ? cos[cos.length >> 1] : 0,
    p05Cos: cos.length ? cos[Math.floor(cos.length * 0.05)] : 0,
    identicalDesc: cos.length ? identical / cos.length : 0,
    maxPosErr: maxErr,
  };
}

function fmt(p: Parity): string {
  return `pos ${(100 * p.posFrac).toFixed(1)}%, desc pairs ${p.pairs}, median cos ${p.medianCos.toFixed(5)}, p5 cos ${p.p05Cos.toFixed(4)}, identical ${(100 * p.identicalDesc).toFixed(1)}%, max pos err ${p.maxPosErr.toExponential(1)}`;
}

function checkContract(fs: FeatureSet, maxFeatures: number, W: number, H: number): void {
  expect(fs.count).toBeLessThanOrEqual(maxFeatures);
  expect(fs.keypoints.length).toBe(4 * fs.count);
  expect(fs.descriptors.length).toBe(DESCRIPTOR_SIZE * fs.count);
  expect(fs.scores.length).toBe(fs.count);
  expect([fs.width, fs.height]).toEqual([W, H]);
  for (let i = 1; i < fs.count; i++) expect(fs.scores[i - 1]).toBeGreaterThanOrEqual(fs.scores[i]);
  for (let i = 0; i < fs.count; i++) {
    const [x, y, s, o] = fs.keypoints.subarray(4 * i, 4 * i + 4);
    expect(x >= -0.5 && x <= W - 0.5 && y >= -0.5 && y <= H - 0.5).toBe(true);
    expect(s).toBeGreaterThan(0);
    expect(o >= 0 && o < 2 * Math.PI).toBe(true);
    // unit-ish RootSIFT: sum of squares of the quantised values ≈ 512²
    let ss = 0;
    for (let k = 0; k < DESCRIPTOR_SIZE; k++) ss += fs.descriptors[i * DESCRIPTOR_SIZE + k] ** 2;
    expect(Math.abs(Math.sqrt(ss) / 512 - 1)).toBeLessThan(0.05);
  }
}

function toU8(img: GrayImage): GrayImage {
  const d = new Uint8Array(img.width * img.height);
  for (let i = 0; i < d.length; i++) d[i] = Math.round(img.data[i] * 255);
  return { width: img.width, height: img.height, data: d };
}

let ctx: GPUContext;
let sift: SiftGPU;

beforeAll(async () => {
  const c = await getGPU();
  if (!c) throw new Error('WebGPU unavailable in the test browser');
  ctx = c;
  const t0 = performance.now();
  sift = await SiftGPU.create(ctx);
  log(`adapter ${ctx.info.vendor}/${ctx.info.architecture} software=${ctx.isSoftware}; create ${(performance.now() - t0).toFixed(0)} ms`);
});

afterAll(() => sift?.destroy());

async function compare(name: string, img: GrayImage, opts: Partial<SiftOptions>, ex: SiftGPU, fw?: number, fh?: number) {
  const t0 = performance.now();
  const g = await ex.extract(img, fw, fh);
  const tg = performance.now() - t0;
  const t1 = performance.now();
  const c = extractSiftCPU(img, opts, fw, fh);
  const tc = performance.now() - t1;
  const p = parity(c, g);
  const T = ex.lastTimings!;
  log(
    `${name}: GPU ${g.count} / CPU ${c.count} features; ${fmt(p)}; GPU ${tg.toFixed(0)} ms (detect ${T.detectMs.toFixed(0)}, describe ${T.describeMs.toFixed(0)}, ` +
      `${T.candidates} cand, ${T.preselected} pre, ${T.oriented} oriented, ${T.submits} submits, ${T.readbacks} readbacks), CPU ${tc.toFixed(0)} ms`,
  );
  return { g, c, p };
}

describe('SiftGPU vs extractSiftCPU', () => {
  it('Float32 640x480 (auto octave -1): keypoints, scales, orientations and descriptors agree', async () => {
    const img = makeTexturedImage(640, 480, 5);
    const { g, c, p } = await compare('f32 640x480 o-1', img, {}, sift);
    checkContract(g, 4096, 640, 480);
    expect(Math.abs(g.count - c.count)).toBeLessThanOrEqual(Math.max(3, 0.02 * c.count));
    expect(p.posFrac).toBeGreaterThanOrEqual(0.97);
    expect(p.medianCos).toBeGreaterThanOrEqual(0.995);
    expect(p.p05Cos).toBeGreaterThan(0.98);
  });

  it('Uint8 input, octave 0, maxFeatures 600, maxOrientations 1', async () => {
    const opts = { upsampleBelow: 0, maxFeatures: 600, maxOrientations: 1 };
    const ex = await SiftGPU.create(ctx, opts);
    try {
      const img = toU8(makeTexturedImage(640, 480, 6));
      const { g, p } = await compare('u8 640x480 o0 600', img, opts, ex);
      checkContract(g, 600, 640, 480);
      expect(g.count).toBe(600);
      expect(p.posFrac).toBeGreaterThanOrEqual(0.97);
      expect(p.medianCos).toBeGreaterThanOrEqual(0.995);
    } finally {
      ex.destroy();
    }
  });

  it('internal downscale (maxImageSize) and external full-resolution mapping', async () => {
    const opts = { maxImageSize: 400 };
    const ex = await SiftGPU.create(ctx, opts);
    try {
      const img = makeTexturedImage(640, 480, 7);
      const a = await compare('640x480 -> 400x300', img, opts, ex);
      expect(a.p.posFrac).toBeGreaterThanOrEqual(0.97);
      expect(a.p.medianCos).toBeGreaterThanOrEqual(0.995);
      const small = makeTexturedImage(320, 240, 8);
      const b = await compare('320x240 in 1280x960 frame', small, opts, ex, 1280, 960);
      expect([b.g.width, b.g.height]).toEqual([1280, 960]);
      expect(b.p.posFrac).toBeGreaterThanOrEqual(0.97);
      expect(b.p.medianCos).toBeGreaterThanOrEqual(0.995);
      checkContract(b.g, 4096, 1280, 960);
    } finally {
      ex.destroy();
    }
  });

  it('shared-memory (tiled) blur kernels give the same features as the direct ones', async () => {
    const tiled = await SiftGPU.create(ctx, {}, { blur: 'tiled' });
    const direct = await SiftGPU.create(ctx, {}, { blur: 'direct' });
    try {
      const img = makeTexturedImage(256, 192, 9);
      const t0 = performance.now();
      const a = await tiled.extract(img);
      const t1 = performance.now();
      const b = await direct.extract(img);
      const t2 = performance.now();
      log(`tiled ${(t1 - t0).toFixed(0)} ms vs direct ${(t2 - t1).toFixed(0)} ms (SwiftShader emulates barriers slowly)`);
      expect(a.count).toBe(b.count);
      expect(Array.from(a.keypoints)).toEqual(Array.from(b.keypoints));
      expect(Array.from(a.descriptors)).toEqual(Array.from(b.descriptors));
      const p = parity(extractSiftCPU(img), a);
      log(`tiled 256x192: ${fmt(p)}`);
      expect(p.posFrac).toBeGreaterThanOrEqual(0.97);
    } finally {
      tiled.destroy();
      direct.destroy();
    }
  });

  it('matching precision on warped pairs equals the CPU pipeline within a few %', async () => {
    const matcher = await MatcherGPU.create(ctx);
    try {
      const W = 640;
      const H = 480;
      for (const [name, Hm] of [
        ['similarity 30° x0.9', similarityAboutCenter(W, H, 30, 0.9)],
        ['tilt 35°', tiltHomography(W, H, 35, -10)],
      ] as [string, Mat3][]) {
        const pair = makeWarpedPair(W, H, Hm, 31, { noiseSigma: 0.01, gain: 0.9, bias: 0.03 });
        const ga = await sift.extract(pair.a);
        const gb = await sift.extract(pair.b);
        const gm = await matcher.match(ga, gb);
        const ge = evaluateMatches(ga, gb, gm, pair.H);
        const ca = extractSiftCPU(pair.a);
        const cb = extractSiftCPU(pair.b);
        const ce = evaluateMatches(ca, cb, matchCPU(ca, cb), pair.H);
        log(`${name}: GPU ${ge.correct}/${ge.total} (${(100 * ge.precision).toFixed(1)}%), CPU ${ce.correct}/${ce.total} (${(100 * ce.precision).toFixed(1)}%)`);
        expect(ge.precision).toBeGreaterThan(0.9);
        expect(Math.abs(ge.precision - ce.precision)).toBeLessThan(0.03);
        expect(ge.correct).toBeGreaterThan(0.9 * ce.correct);
      }
    } finally {
      matcher.destroy();
    }
  });
});

describe('SiftGPU descriptor precision', () => {
  it('identical keypoints get descriptors within ±1 byte, also on low-contrast images', async () => {
    for (const contrast of [1, 0.15]) {
      const img = makeTexturedImage(320, 240, 9, { contrast });
      const g = await sift.extract(img);
      const c = extractSiftCPU(img);
      let n = 0;
      let ident = 0;
      let maxDiff = 0;
      let sumDiff = 0;
      for (let i = 0; i < c.count; i++) {
        let best = -1;
        let bd = Infinity;
        for (let j = 0; j < g.count; j++) {
          const d =
            Math.hypot(c.keypoints[4 * i] - g.keypoints[4 * j], c.keypoints[4 * i + 1] - g.keypoints[4 * j + 1]) +
            angDiff(c.keypoints[4 * i + 3], g.keypoints[4 * j + 3]);
          if (d < bd) {
            bd = d;
            best = j;
          }
        }
        if (bd > 0.01) continue;
        n++;
        let same = true;
        for (let k = 0; k < DESCRIPTOR_SIZE; k++) {
          const df = Math.abs(c.descriptors[i * DESCRIPTOR_SIZE + k] - g.descriptors[best * DESCRIPTOR_SIZE + k]);
          maxDiff = Math.max(maxDiff, df);
          sumDiff += df;
          if (df) same = false;
        }
        if (same) ident++;
      }
      log(
        `contrast ${contrast}: CPU ${c.count} / GPU ${g.count}, ${n} identical keypoints (±0.01), descriptors byte-identical ${((100 * ident) / n).toFixed(1)}%, ` +
          `max byte diff ${maxDiff}, mean abs byte diff ${(sumDiff / (n * DESCRIPTOR_SIZE)).toFixed(4)}`,
      );
      expect(n).toBeGreaterThan(0.95 * c.count);
      expect(maxDiff).toBeLessThanOrEqual(2);
      expect(sumDiff / (n * DESCRIPTOR_SIZE)).toBeLessThan(0.02);
    }
  });
});

describe('SiftGPU.extractFromBitmap', () => {
  function rgbaImage(W: number, H: number, seed: number): ImageData {
    // Colour image whose channels differ (exercises the luma weights).
    const r = makeTexturedImage(W, H, seed).data;
    const g = makeTexturedImage(W, H, seed + 100).data;
    const id = new ImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      id.data[4 * i] = Math.round(255 * r[i]);
      id.data[4 * i + 1] = Math.round(255 * (0.7 * r[i] + 0.3 * g[i]));
      id.data[4 * i + 2] = Math.round(255 * g[i]);
      id.data[4 * i + 3] = 255;
    }
    return id;
  }

  it('ImageBitmap at the processed size matches rgbaToGray + CPU', async () => {
    const id = rgbaImage(480, 360, 41);
    const bmp = await createImageBitmap(id);
    const t0 = performance.now();
    const g = await sift.extractFromBitmap(bmp);
    const tg = performance.now() - t0;
    const c = extractSiftCPU(rgbaToGray(id));
    const p = parity(c, g);
    log(`bitmap 480x360: GPU ${g.count} / CPU ${c.count}; ${fmt(p)}; ${tg.toFixed(0)} ms`);
    checkContract(g, 4096, 480, 360);
    expect(p.posFrac).toBeGreaterThanOrEqual(0.97);
    expect(p.medianCos).toBeGreaterThanOrEqual(0.995);
    bmp.close();
  });

  it('downscales on the GPU and maps to the requested full-resolution frame', async () => {
    const opts = { maxImageSize: 320 };
    const ex = await SiftGPU.create(ctx, opts);
    try {
      const id = rgbaImage(640, 480, 43);
      const bmp = await createImageBitmap(id);
      const g = await ex.extractFromBitmap(bmp, 1280, 960);
      const c = extractSiftCPU(rgbaToGray(id), opts, 1280, 960);
      const p = parity(c, g);
      log(`bitmap 640x480 -> 320x240 in 1280x960: GPU ${g.count} / CPU ${c.count}; ${fmt(p)}`);
      checkContract(g, 4096, 1280, 960);
      expect(p.posFrac).toBeGreaterThanOrEqual(0.97);
      expect(p.medianCos).toBeGreaterThanOrEqual(0.995);
      // ImageData is accepted directly too.
      const g2 = await ex.extractFromBitmap(id, 1280, 960);
      expect(parity(g, g2).posFrac).toBeGreaterThan(0.99);
      bmp.close();
    } finally {
      ex.destroy();
    }
  });
});

describe('SiftGPU robustness', () => {
  it('handles edge cases (tiny / blank images, maxFeatures 0, invalid input)', async () => {
    const blank = { width: 64, height: 48, data: new Float32Array(64 * 48).fill(0.5) };
    const e1 = await sift.extract(blank);
    expect(e1.count).toBe(0);
    expect([e1.width, e1.height]).toEqual([64, 48]);
    const tiny = makeTexturedImage(12, 9, 1);
    const e2 = await sift.extract(tiny);
    expect(e2.count).toBe(extractSiftCPU(tiny).count);
    const small = makeTexturedImage(40, 30, 2);
    const e3 = await sift.extract(small);
    const c3 = extractSiftCPU(small);
    log(`40x30: GPU ${e3.count}, CPU ${c3.count}`);
    expect(Math.abs(e3.count - c3.count)).toBeLessThanOrEqual(2);
    const zero = await SiftGPU.create(ctx, { maxFeatures: 0 });
    expect((await zero.extract(small)).count).toBe(0);
    zero.destroy();
    // Input errors are plain Errors (no CPU fallback trigger), the extractor stays usable.
    const bad = { width: 100, height: 100, data: new Float32Array(10) };
    await expect(sift.extract(bad)).rejects.toThrow(/shorter/);
    await expect(sift.extract(bad)).rejects.not.toBeInstanceOf(GPUFeatureError);
    expect((await sift.extract(small)).count).toBe(e3.count);
    // odd-length Uint8 data (tail bytes are uploaded separately)
    const odd = toU8(makeTexturedImage(37, 29, 3));
    const g = await sift.extract(odd);
    const c = extractSiftCPU(odd);
    expect(Math.abs(g.count - c.count)).toBeLessThanOrEqual(2);
  });

  it('grows the candidate buffer when the capacity is exceeded (same result)', async () => {
    const ex = await SiftGPU.create(ctx, {}, { initialCandidateCapacity: 256 });
    try {
      const img = makeTexturedImage(400, 300, 12, { density: 2 });
      const a = await ex.extract(img);
      log(`capacity growth: ${ex.lastTimings!.candidates} candidates, ${ex.lastTimings!.readbacks} readbacks`);
      expect(ex.lastTimings!.candidates).toBeGreaterThan(256);
      const b = await sift.extract(img);
      expect(a.count).toBe(b.count);
      expect(Array.from(a.keypoints)).toEqual(Array.from(b.keypoints));
      expect(Array.from(a.descriptors)).toEqual(Array.from(b.descriptors));
      // count > readback guess (but within capacity): a second readback of the records
      const ex2 = await SiftGPU.create(ctx, {}, { initialReadbackGuess: 64 });
      const c = await ex2.extract(img);
      expect(ex2.lastTimings!.readbacks).toBe(3);
      expect(Array.from(c.keypoints)).toEqual(Array.from(b.keypoints));
      expect(Array.from(c.descriptors)).toEqual(Array.from(b.descriptors));
      ex2.destroy();
    } finally {
      ex.destroy();
    }
  });

  it('selfTest passes on this adapter', async () => {
    const r = await sift.selfTest();
    log(`selfTest: ${JSON.stringify(r)}`);
    expect(r.ok).toBe(true);
  });

  it('detects GPU work dropped by the OS and redoes the extraction', async () => {
    const img = makeTexturedImage(200, 150, 33);
    const ref = await sift.extract(img);
    for (const drop of [[0], [1]]) {
      const ex = await SiftGPU.create(ctx, {}, { testDropSubmissions: (attempt) => (attempt === 0 ? drop : []) });
      try {
        const r = await ex.extract(img);
        expect(ex.lastTimings!.attempts).toBe(2);
        expect(Array.from(r.keypoints)).toEqual(Array.from(ref.keypoints));
        expect(Array.from(r.descriptors)).toEqual(Array.from(ref.descriptors));
      } finally {
        ex.destroy();
      }
    }
    const always = await SiftGPU.create(ctx, {}, { testDropSubmissions: () => [1] });
    await expect(always.extract(img)).rejects.toBeInstanceOf(GPUWorkDroppedError);
    // still usable afterwards
    always.destroy();
  });

  it('caches pyramids per size (LRU), results are deterministic, destroy() frees', async () => {
    const ex = await SiftGPU.create(ctx, {}, { maxCachedSizes: 1 });
    const i1 = makeTexturedImage(200, 150, 21);
    const i2 = makeTexturedImage(160, 200, 22);
    const r1 = await ex.extract(i1);
    const m1 = ex.memoryBytes();
    const r2 = await ex.extract(i2);
    const r1b = await ex.extract(i1);
    expect(Array.from(r1b.keypoints)).toEqual(Array.from(r1.keypoints));
    expect(Array.from(r1b.descriptors)).toEqual(Array.from(r1.descriptors));
    expect(r2.count).toBeGreaterThan(0);
    expect(m1).toBeGreaterThan(0);
    // concurrent calls are serialised
    const [p, q] = await Promise.all([ex.extract(i1), ex.extract(i2)]);
    expect(p.count).toBe(r1.count);
    expect(q.count).toBe(r2.count);
    ex.releaseCaches();
    expect((await ex.extract(i2)).count).toBe(r2.count);
    ex.destroy();
    await expect(ex.extract(i1)).rejects.toBeInstanceOf(GPUFeatureError);
  });
});
