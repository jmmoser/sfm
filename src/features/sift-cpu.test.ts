import { describe, expect, it } from 'vitest';
import type { FeatureSet, GrayImage, Mat3 } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';
import { resizeGray } from './image';
import { matchCPU } from './match';
import { extractSiftCPU, lastSiftCPUTimings } from './sift-cpu';
import { planSift, resolveSiftOptions } from './sift-common';
import {
  TestRng,
  evaluateMatches,
  gridCoverage,
  makeTexturedImage,
  makeWarpedPair,
  repeatability,
  similarityAboutCenter,
  tiltHomography,
} from './testdata';

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

interface Blob {
  x: number;
  y: number;
  s: number;
}

/** Grid of bright/dark Gaussian blobs at random sub-pixel positions. */
function blobImage(W: number, H: number, n: number, sMin: number, sMax: number, seed: number): { img: GrayImage; blobs: Blob[] } {
  const rng = new TestRng(seed);
  const d = new Float32Array(W * H).fill(0.5);
  const blobs: Blob[] = [];
  const gx = Math.floor(Math.sqrt((n * W) / H));
  const gy = Math.ceil(n / gx);
  for (let j = 0; j < gy; j++)
    for (let i = 0; i < gx; i++) {
      const s = rng.uniform(sMin, sMax);
      const x = ((i + 0.5) * W) / gx + rng.uniform(-0.5, 0.5);
      const y = ((j + 0.5) * H) / gy + rng.uniform(-0.5, 0.5);
      const amp = (i + j) % 2 ? 0.4 : -0.4;
      blobs.push({ x, y, s });
      const e = Math.ceil(4 * s);
      for (let yy = Math.max(0, Math.floor(y - e)); yy <= Math.min(H - 1, y + e); yy++)
        for (let xx = Math.max(0, Math.floor(x - e)); xx <= Math.min(W - 1, x + e); xx++)
          d[yy * W + xx] += amp * Math.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s));
    }
  return { img: { width: W, height: H, data: d }, blobs };
}

function evalBlobs(fs: FeatureSet, blobs: Blob[]) {
  let found = 0;
  let bx = 0;
  let by = 0;
  let se = 0;
  const ratios: number[] = [];
  for (const b of blobs) {
    let best = Infinity;
    let bi = -1;
    for (let i = 0; i < fs.count; i++) {
      const r = fs.keypoints[4 * i + 2] / b.s;
      if (r < 0.6 || r > 1.6) continue;
      const d = Math.hypot(fs.keypoints[4 * i] - b.x, fs.keypoints[4 * i + 1] - b.y);
      if (d < best) {
        best = d;
        bi = i;
      }
    }
    if (bi < 0 || best > 2) continue;
    found++;
    const ex = fs.keypoints[4 * bi] - b.x;
    const ey = fs.keypoints[4 * bi + 1] - b.y;
    bx += ex;
    by += ey;
    se += ex * ex + ey * ey;
    ratios.push(fs.keypoints[4 * bi + 2] / b.s);
  }
  ratios.sort((p, q) => p - q);
  return { found, biasX: bx / found, biasY: by / found, rms: Math.sqrt(se / found), scaleMedian: ratios[ratios.length >> 1] };
}

/** Exact clockwise 90° rotation of a gray image (pixel permutation). */
function rotateGray90(img: GrayImage): GrayImage {
  const { width: W, height: H } = img;
  const src = img.data;
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) out[x * H + (H - 1 - y)] = src[y * W + x];
  return { width: H, height: W, data: out };
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

describe('planSift', () => {
  it('downsizes to maxImageSize and auto-upsamples small images', () => {
    const o = resolveSiftOptions();
    expect(o.peakThreshold).toBeCloseTo(0.02 / 3, 12);
    const p1 = planSift(4032, 3024, o);
    expect([p1.procWidth, p1.procHeight, p1.firstOctave]).toEqual([1600, 1200, 0]);
    expect(p1.octaves[0]).toEqual({ octave: 0, width: 1600, height: 1200 });
    expect(Math.min(p1.octaves.at(-1)!.width, p1.octaves.at(-1)!.height)).toBeGreaterThanOrEqual(16);
    const p2 = planSift(640, 480, o);
    expect([p2.procWidth, p2.firstOctave, p2.octaves[0].width]).toEqual([640, -1, 1280]);
    expect(p2.baseBlur).toBe(1);
    const p3 = planSift(640, 480, resolveSiftOptions({ upsampleBelow: 0 }));
    expect(p3.firstOctave).toBe(0);
    const p4 = planSift(1600, 1200, resolveSiftOptions({ firstOctave: 1 }), 3200, 2400);
    expect([p4.procWidth, p4.firstOctave, p4.kx]).toEqual([800, 0, 0.25]);
    expect(resolveSiftOptions({ scalesPerOctave: 4 }).peakThreshold).toBeCloseTo(0.005, 12);
  });
});

describe('extractSiftCPU — localisation and coordinate frames', () => {
  it('localises Gaussian blobs with sub-pixel accuracy (octave -1 and 0)', () => {
    const { img, blobs } = blobImage(400, 300, 48, 2.5, 6, 1);
    for (const opts of [{}, { upsampleBelow: 0 }]) {
      const r = evalBlobs(extractSiftCPU(img, opts), blobs);
      expect(r.found).toBe(blobs.length);
      expect(Math.abs(r.biasX)).toBeLessThan(0.02);
      expect(Math.abs(r.biasY)).toBeLessThan(0.02);
      expect(r.rms).toBeLessThan(0.1);
      // DoG scale convention (VLFeat/Lowe): sigma of the lower bracketing level ≈ σ_blob·2^(-1/6).
      expect(r.scaleMedian).toBeGreaterThan(0.8);
      expect(r.scaleMedian).toBeLessThan(1.0);
    }
  });

  it('maps keypoints to the full-resolution frame (internal and external downscaling)', () => {
    const { img, blobs } = blobImage(1600, 1200, 48, 6, 14, 2);
    const cases: [string, FeatureSet][] = [
      ['full', extractSiftCPU(img)],
      ['max800', extractSiftCPU(img, { maxImageSize: 800 })],
      ['max700', extractSiftCPU(img, { maxImageSize: 700 })],
      ['max600', extractSiftCPU(img, { maxImageSize: 600 })],
      ['ext800', extractSiftCPU(resizeGray(img, 800, 600), {}, 1600, 1200)],
      ['ext533', extractSiftCPU(resizeGray(img, 533, 400), {}, 1600, 1200)],
    ];
    for (const [name, fs] of cases) {
      expect(fs.width).toBe(1600);
      expect(fs.height).toBe(1200);
      const r = evalBlobs(fs, blobs);
      // A wrong half-pixel convention would show up as a bias of ~0.5-1.5 px here.
      expect(r.found, name).toBe(blobs.length);
      expect(Math.abs(r.biasX), name).toBeLessThan(0.06);
      expect(Math.abs(r.biasY), name).toBeLessThan(0.06);
      expect(r.rms, name).toBeLessThan(0.25);
      expect(r.scaleMedian, name).toBeGreaterThan(0.8);
      expect(r.scaleMedian, name).toBeLessThan(1.0);
    }
  });

  it('is consistent between full-res and downscaled extraction on textures', () => {
    const img = makeTexturedImage(1280, 960, 21);
    const full = extractSiftCPU(img, { maxFeatures: 3000 });
    const half = extractSiftCPU(resizeGray(img, 640, 480), { maxFeatures: 3000, upsampleBelow: 0 }, 1280, 960);
    const m = matchCPU(half, full);
    let n = 0;
    let bx = 0;
    let by = 0;
    let lr = 0;
    for (let k = 0; k < m.length; k += 2) {
      const i = m[k];
      const j = m[k + 1];
      const ex = half.keypoints[4 * i] - full.keypoints[4 * j];
      const ey = half.keypoints[4 * i + 1] - full.keypoints[4 * j + 1];
      if (Math.hypot(ex, ey) > 2) continue;
      n++;
      bx += ex;
      by += ey;
      lr += Math.log(half.keypoints[4 * i + 2] / full.keypoints[4 * j + 2]);
    }
    expect(n).toBeGreaterThan(300);
    expect(Math.abs(bx / n)).toBeLessThan(0.1);
    expect(Math.abs(by / n)).toBeLessThan(0.1);
    expect(Math.abs(Math.exp(lr / n) - 1)).toBeLessThan(0.05);
  });
});

describe('extractSiftCPU — output contract', () => {
  const img = makeTexturedImage(640, 480, 11);
  const fs = extractSiftCPU(img, { maxFeatures: 2000 });

  it('respects maxFeatures, frame size and sorting', () => {
    expect(fs.count).toBeGreaterThan(1000);
    expect(fs.count).toBeLessThanOrEqual(2000);
    expect(fs.keypoints.length).toBe(4 * fs.count);
    expect(fs.descriptors.length).toBe(DESCRIPTOR_SIZE * fs.count);
    expect(fs.scores.length).toBe(fs.count);
    expect([fs.width, fs.height]).toEqual([640, 480]);
    for (let i = 1; i < fs.count; i++) expect(fs.scores[i - 1]).toBeGreaterThanOrEqual(fs.scores[i]);
    for (let i = 0; i < fs.count; i++) {
      const [x, y, s, o] = fs.keypoints.subarray(4 * i, 4 * i + 4);
      expect(x).toBeGreaterThanOrEqual(-0.5);
      expect(x).toBeLessThanOrEqual(639.5);
      expect(y).toBeGreaterThanOrEqual(-0.5);
      expect(y).toBeLessThanOrEqual(479.5);
      expect(s).toBeGreaterThan(0.5);
      expect(o).toBeGreaterThanOrEqual(0);
      expect(o).toBeLessThan(2 * Math.PI);
      expect(fs.scores[i]).toBeGreaterThanOrEqual(resolveSiftOptions().peakThreshold);
    }
  });

  it('produces unit-norm RootSIFT descriptors', () => {
    let worst = 0;
    for (let i = 0; i < fs.count; i++) {
      let ss = 0;
      for (let k = 0; k < DESCRIPTOR_SIZE; k++) {
        const v = fs.descriptors[i * DESCRIPTOR_SIZE + k] / 512;
        ss += v * v;
      }
      worst = Math.max(worst, Math.abs(Math.sqrt(ss) - 1));
    }
    // quantisation error of 128 rounded components (+ rare 255 clamps)
    expect(worst).toBeLessThan(0.03);
  });

  it('is deterministic and accepts Uint8 input', () => {
    const again = extractSiftCPU(img, { maxFeatures: 2000 });
    expect(Array.from(again.keypoints)).toEqual(Array.from(fs.keypoints));
    expect(Array.from(again.descriptors)).toEqual(Array.from(fs.descriptors));
    const u8 = new Uint8Array(img.data.length);
    for (let i = 0; i < u8.length; i++) u8[i] = Math.round(img.data[i] * 255);
    const fu = extractSiftCPU({ width: 640, height: 480, data: u8 }, { maxFeatures: 2000 });
    expect(Math.abs(fu.count - fs.count)).toBeLessThan(0.05 * fs.count);
  });

  it('handles tiny and degenerate images', () => {
    expect(extractSiftCPU({ width: 8, height: 8, data: new Float32Array(64) }).count).toBe(0);
    expect(extractSiftCPU({ width: 64, height: 48, data: new Float32Array(64 * 48).fill(0.5) }).count).toBe(0);
    expect(extractSiftCPU(img, { maxFeatures: 0 }).count).toBe(0);
    const few = extractSiftCPU(img, { maxFeatures: 10 });
    expect(few.count).toBe(10);
  });

  it('distributes features spatially', () => {
    // Left half strongly textured, right half low contrast.
    const W = 640;
    const H = 480;
    const hi = makeTexturedImage(W, H, 4);
    const lo = makeTexturedImage(W, H, 5, { contrast: 0.15 });
    const d = new Float32Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) d[y * W + x] = (x < W / 2 ? hi : lo).data[y * W + x];
    const img2: GrayImage = { width: W, height: H, data: d };
    const plain = extractSiftCPU(img2, { maxFeatures: 500, distribute: false });
    const dist = extractSiftCPU(img2, { maxFeatures: 500, distribute: true });
    const right = (f: FeatureSet) => {
      let n = 0;
      for (let i = 0; i < f.count; i++) if (f.keypoints[4 * i] > W / 2) n++;
      return n;
    };
    expect(dist.count).toBe(500);
    expect(right(dist)).toBeGreaterThan(right(plain) + 50);
    expect(gridCoverage(dist, 8, 6)).toBeGreaterThan(gridCoverage(plain, 8, 6));
    expect(gridCoverage(dist, 8, 6)).toBeGreaterThan(0.95);
  });
});

describe('extractSiftCPU — invariance (repeatability + matching precision)', () => {
  it('exact 90° rotation', () => {
    const img = makeTexturedImage(640, 480, 13);
    const rot = rotateGray90(img);
    const a = extractSiftCPU(img);
    const b = extractSiftCPU(rot);
    // x_b = H - 1 - y_a, y_b = x_a
    const H: Mat3 = new Float64Array([0, -1, 479, 1, 0, 0, 0, 0, 1]);
    const m = matchCPU(a, b);
    const ev = evaluateMatches(a, b, m, H, 2);
    const rep = repeatability(a, b, H, 1.5);
    console.log(`[sift] exact rot90: ${a.count}/${b.count} feats, ${ev.correct}/${ev.total} correct, rep ${rep.ratio.toFixed(3)}`);
    expect(ev.precision).toBeGreaterThan(0.99);
    expect(ev.correct).toBeGreaterThan(0.5 * a.count);
    expect(rep.ratio).toBeGreaterThan(0.85);
  });

  const W = 640;
  const Hh = 480;
  const cases: { name: string; H: Mat3; photo?: Parameters<typeof makeWarpedPair>[4]; minCorrect: number; minRep: number }[] = [
    { name: 'identity+noise σ=0.02', H: similarityAboutCenter(W, Hh, 0), photo: { noiseSigma: 0.02, seed: 3 }, minCorrect: 800, minRep: 0.6 },
    { name: 'rotation 30°', H: similarityAboutCenter(W, Hh, 30), minCorrect: 800, minRep: 0.6 },
    { name: 'rotation 90°', H: similarityAboutCenter(W, Hh, 90), minCorrect: 800, minRep: 0.6 },
    { name: 'scale 0.6', H: similarityAboutCenter(W, Hh, 0, 0.6), minCorrect: 500, minRep: 0.5 },
    { name: 'scale 1.5', H: similarityAboutCenter(W, Hh, 0, 1.5), minCorrect: 500, minRep: 0.5 },
    { name: 'scale 0.6 + rotation 45°', H: similarityAboutCenter(W, Hh, 45, 0.6), minCorrect: 400, minRep: 0.45 },
    { name: 'perspective tilt 40°/10°', H: tiltHomography(W, Hh, 40, 10), minCorrect: 400, minRep: 0.45 },
    {
      name: 'brightness/contrast/gamma',
      H: similarityAboutCenter(W, Hh, 10, 1, 7, -4),
      photo: { gain: 0.55, bias: 0.2, gamma: 1.4 },
      minCorrect: 800,
      minRep: 0.6,
    },
    {
      name: 'rotation 20° + noise σ=0.04',
      H: similarityAboutCenter(W, Hh, 20),
      photo: { noiseSigma: 0.04, seed: 9 },
      minCorrect: 300,
      minRep: 0.4,
    },
  ];
  for (const c of cases) {
    it(c.name, () => {
      const p = makeWarpedPair(W, Hh, c.H, 17, c.photo);
      const a = extractSiftCPU(p.a);
      const b = extractSiftCPU(p.b);
      const m = matchCPU(a, b);
      const ev = evaluateMatches(a, b, m, p.H, 2);
      const rep = repeatability(a, b, p.H, 2);
      console.log(
        `[sift] ${c.name}: feats ${a.count}/${b.count}, matches ${ev.total}, correct ${ev.correct}, ` +
          `precision ${ev.precision.toFixed(3)}, repeatability ${rep.ratio.toFixed(3)} (${rep.repeated}/${Math.min(rep.visible, rep.visibleB)})`,
      );
      expect(ev.precision).toBeGreaterThanOrEqual(0.9);
      expect(ev.correct).toBeGreaterThanOrEqual(c.minCorrect);
      expect(rep.ratio).toBeGreaterThanOrEqual(c.minRep);
    });
  }
});

describe('extractSiftCPU — timing', () => {
  it('1600x1200 extraction (Node, informational)', () => {
    const img = makeTexturedImage(1600, 1200, 31);
    extractSiftCPU(img); // warm-up (JIT + buffer pool)
    const t0 = performance.now();
    const fs = extractSiftCPU(img);
    const dt = performance.now() - t0;
    const st = Object.entries(lastSiftCPUTimings)
      .map(([k, v]) => `${k}=${Math.round(v)}`)
      .join(' ');
    console.log(`[sift] 1600x1200 CPU extraction: ${dt.toFixed(0)} ms, ${fs.count} features (${st})`);
    expect(fs.count).toBe(4096);
  });
});

// ---------------------------------------------------------------------------
// Adversarial convention tests (review): these pin down conventions that plain
// matching tests cannot see (a consistently wrong sign still matches perfectly),
// but that the GPU implementation and the SfM code rely on.
// ---------------------------------------------------------------------------

function wrapAngle(a: number): number {
  a %= 2 * Math.PI;
  if (a > Math.PI) a -= 2 * Math.PI;
  else if (a <= -Math.PI) a += 2 * Math.PI;
  return a;
}

function percentile(v: number[], p: number): number {
  const s = [...v].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))];
}

/** Matched pairs whose positions agree with H within tol px. */
function geometricallyCorrect(a: FeatureSet, b: FeatureSet, H: Mat3, tol: number): [number, number][] {
  const m = matchCPU(a, b);
  const out: [number, number][] = [];
  for (let k = 0; k < m.length; k += 2) {
    const i = m[k];
    const j = m[k + 1];
    const xa = a.keypoints[4 * i];
    const ya = a.keypoints[4 * i + 1];
    const w = H[6] * xa + H[7] * ya + H[8];
    const u = (H[0] * xa + H[1] * ya + H[2]) / w;
    const v = (H[3] * xa + H[4] * ya + H[5]) / w;
    if (Math.hypot(u - b.keypoints[4 * j], v - b.keypoints[4 * j + 1]) <= tol) out.push([i, j]);
  }
  return out;
}

describe('extractSiftCPU — conventions (adversarial)', () => {
  it('orientation is measured from +x towards +y (image frame) and scale follows the warp', () => {
    // similarityAboutCenter(angle) rotates clockwise as displayed (from +x towards +y),
    // so every orientation must increase by +angle; a flipped sign would give -angle.
    const W = 640;
    const Hh = 480;
    for (const [deg, sc] of [[35, 0.85], [-60, 1.2], [150, 1]] as const) {
      const p = makeWarpedPair(W, Hh, similarityAboutCenter(W, Hh, deg, sc), 29);
      const a = extractSiftCPU(p.a);
      const b = extractSiftCPU(p.b);
      const pairs = geometricallyCorrect(a, b, p.H, 1.5);
      expect(pairs.length).toBeGreaterThan(500);
      const target = (deg * Math.PI) / 180;
      const dth = pairs.map(([i, j]) => Math.abs(wrapAngle(b.keypoints[4 * j + 3] - a.keypoints[4 * i + 3] - target)));
      const lr = pairs.map(([i, j]) => Math.log(b.keypoints[4 * j + 2] / a.keypoints[4 * i + 2]));
      console.log(`[sift] orientation ${deg > 0 ? '+' : ''}${deg}° x${sc}: ${pairs.length} pairs, |Δθ err| p50 ${percentile(dth, 0.5).toFixed(4)} p90 ${percentile(dth, 0.9).toFixed(4)} rad, scale ratio ${(Math.exp(percentile(lr, 0.5)) / sc).toFixed(4)}`);
      // natural orientation noise is ~1°; a flipped convention would be off by 2·angle
      expect(percentile(dth, 0.5)).toBeLessThan(0.05);
      expect(percentile(dth, 0.9)).toBeLessThan(0.2);
      expect(Math.abs(Math.exp(percentile(lr, 0.5)) / sc - 1)).toBeLessThan(0.02);
    }
  });

  it('exact 90° rotation adds π/2 to orientations and preserves descriptors', () => {
    // Odd height => decimation parity is preserved under the rotation, so the scale
    // space is (up to float rounding) exactly equivariant.
    const img = makeTexturedImage(641, 481, 13);
    const a = extractSiftCPU(img);
    const b = extractSiftCPU(rotateGray90(img));
    const H: Mat3 = new Float64Array([0, -1, 480, 1, 0, 0, 0, 0, 1]);
    const pairs = geometricallyCorrect(a, b, H, 0.05);
    expect(pairs.length).toBeGreaterThan(0.9 * a.count);
    const dth = pairs.map(([i, j]) => Math.abs(wrapAngle(b.keypoints[4 * j + 3] - a.keypoints[4 * i + 3] - Math.PI / 2)));
    expect(percentile(dth, 0.95)).toBeLessThan(1e-3);
    let identical = 0;
    for (const [i, j] of pairs) {
      let d = 0;
      for (let k = 0; k < DESCRIPTOR_SIZE; k++) d = Math.max(d, Math.abs(a.descriptors[128 * i + k] - b.descriptors[128 * j + k]));
      if (d <= 1) identical++;
    }
    expect(identical).toBeGreaterThan(0.95 * pairs.length);
  });

  // Descriptor layout (normative for sift-gpu): index (by*4 + bx)*8 + bo, bx along
  // x' = (cos θ, sin θ), by along y' = (−sin θ, cos θ), bo = bin of (gradient angle − θ).
  // A dark blob (keypoint) with step edges at known offsets / gradient angles; the
  // expected bins are derived from the detected θ, so the test does not depend on
  // which structure dominates the orientation.
  function layoutImage(withTopEdge: boolean): GrayImage {
    const W = 128;
    const H = 128;
    const sig = (t: number) => 1 / (1 + Math.exp(-t));
    const d = new Float32Array(W * H);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let v = 0.5 - 0.3 * Math.exp(-((x - 63.3) ** 2 + (y - 64.2) ** 2) / 32);
        // vertical edge 12 px right of the blob, bright -> dark: gradient angle π
        v -= 0.3 * sig((x - 75.3) / 0.7);
        // horizontal edge segment 12 px above the blob, bright below: gradient angle π/2
        if (withTopEdge) v += 0.12 * sig((y - 52.2) / 0.7) * sig(6 - Math.abs(x - 63.3));
        d[y * W + x] = v;
      }
    return { width: W, height: H, data: d };
  }
  const g = globalThis as { SFM_DISABLE_SIMD?: boolean };
  for (const simd of [true, false]) {
    for (const withTopEdge of [false, true]) {
      it(`descriptor layout: rotated bins and orientation direction (${withTopEdge ? 'two edges' : 'one edge'}, ${simd ? 'SIMD' : 'JS'})`, () => {
        if (!simd) g.SFM_DISABLE_SIMD = true;
        try {
          const fs = extractSiftCPU(layoutImage(withTopEdge), { upsampleBelow: 0 });
          let k = -1;
          let bd = Infinity;
          for (let i = 0; i < fs.count; i++) {
            const dd = Math.hypot(fs.keypoints[4 * i] - 63.3, fs.keypoints[4 * i + 1] - 64.2);
            if (fs.keypoints[4 * i + 2] > 2 && dd < bd) {
              bd = dd;
              k = i;
            }
          }
          expect(bd).toBeLessThan(0.5);
          const [, , scale, th] = fs.keypoints.subarray(4 * k, 4 * k + 4);
          const desc = fs.descriptors.subarray(128 * k, 128 * k + 128);
          const E = (bx: number, by: number, bo: number) => desc[(by * 4 + bx) * 8 + bo] ** 2;
          // The vertical edge: offset (+12, 0), gradient angle π.
          const SBP = 3 * scale;
          const nx = (Math.cos(th) * 12) / SBP;
          const ny = (-Math.sin(th) * 12) / SBP;
          const bo = ((Math.round(wrapAngle(Math.PI - th) / (Math.PI / 4)) % 8) + 8) % 8;
          const half = (bo2: number, axis: 'x' | 'y', lowSide: boolean) => {
            let e = 0;
            for (let by = 0; by < 4; by++)
              for (let bx = 0; bx < 4; bx++) if ((axis === 'x' ? bx : by) < 2 === lowSide) e += E(bx, by, bo2);
            return e;
          };
          let checked = 0;
          if (Math.abs(nx) > 0.8) {
            expect(half(bo, 'x', nx < 0)).toBeGreaterThan(2 * half(bo, 'x', nx >= 0));
            checked++;
          }
          if (Math.abs(ny) > 0.8) {
            expect(half(bo, 'y', ny < 0)).toBeGreaterThan(2 * half(bo, 'y', ny >= 0));
            // orientation-bin direction: (gradient − θ), not (θ − gradient)
            const flip = (8 - bo) % 8;
            if (flip !== bo) expect(half(bo, 'y', ny < 0)).toBeGreaterThan(5 * half(flip, 'y', ny < 0));
            checked++;
          }
          expect(checked).toBeGreaterThan(0);
        } finally {
          delete g.SFM_DISABLE_SIMD;
        }
      });
    }
  }

  it('other scalesPerOctave (S = 2, 4): sub-pixel localisation and DoG scale convention', () => {
    const { img, blobs } = blobImage(400, 300, 48, 2.5, 6, 1);
    for (const S of [2, 4]) {
      for (const upsampleBelow of [800, 0]) {
        const r = evalBlobs(extractSiftCPU(img, { scalesPerOctave: S, upsampleBelow }), blobs);
        expect(r.found, `S=${S}`).toBeGreaterThanOrEqual(blobs.length - 1);
        expect(Math.abs(r.biasX)).toBeLessThan(0.03);
        expect(Math.abs(r.biasY)).toBeLessThan(0.03);
        expect(r.rms).toBeLessThan(0.12);
        // ideal blob: lower bracketing level sigma = σ_blob · 2^(-1/(2S))
        expect(Math.abs(r.scaleMedian / 2 ** (-1 / (2 * S)) - 1)).toBeLessThan(0.03);
      }
    }
  });

  it('odd sizes, tiny images and extreme aspect ratios produce valid output', () => {
    for (const [w, h] of [[641, 481], [17, 33], [16, 16], [15, 400], [1001, 23], [23, 1001], [100, 15], [1, 1], [3, 2000]]) {
      const img = makeTexturedImage(w, h, 3);
      const fs = extractSiftCPU(img);
      expect([fs.width, fs.height]).toEqual([w, h]);
      for (let i = 0; i < fs.count; i++) {
        const [x, y, s, o] = fs.keypoints.subarray(4 * i, 4 * i + 4);
        expect(x >= -0.5 && x <= w - 0.5 && y >= -0.5 && y <= h - 0.5, `${w}x${h}`).toBe(true);
        expect(s > 0 && Number.isFinite(s) && o >= 0 && o < 2 * Math.PI).toBe(true);
      }
    }
  });

  it('a single full-resolution dimension implies the other from the input aspect', () => {
    const { img, blobs } = blobImage(800, 600, 24, 3, 6, 5);
    const both = extractSiftCPU(img, {}, 1600, 1200);
    const onlyW = extractSiftCPU(img, {}, 1600);
    const onlyH = extractSiftCPU(img, {}, undefined, 1200);
    for (const fs of [onlyW, onlyH]) {
      expect([fs.width, fs.height]).toEqual([1600, 1200]);
      expect(Array.from(fs.keypoints)).toEqual(Array.from(both.keypoints));
    }
    const scaled = blobs.map((b) => ({ x: (b.x + 0.5) * 2 - 0.5, y: (b.y + 0.5) * 2 - 0.5, s: 2 * b.s }));
    expect(evalBlobs(both, scaled).found).toBe(blobs.length);
    expect(() => extractSiftCPU(img, {}, 0, 0)).toThrow();
    expect(() => extractSiftCPU(img, {}, NaN)).toThrow();
  });

  it('Uint8 input with internal downscaling matches the Float32 path', () => {
    const src = makeTexturedImage(1000, 700, 4);
    const u8 = new Uint8Array(src.data.length);
    for (let i = 0; i < u8.length; i++) u8[i] = Math.round(src.data[i] * 255);
    const f32 = new Float32Array(u8.length);
    for (let i = 0; i < u8.length; i++) f32[i] = u8[i] / 255;
    const opts = { maxImageSize: 640, maxFeatures: 1500 };
    const a = extractSiftCPU({ width: 1000, height: 700, data: u8 }, opts);
    const b = extractSiftCPU({ width: 1000, height: 700, data: f32 }, opts);
    expect(Math.abs(a.count - b.count)).toBeLessThanOrEqual(3);
    let same = 0;
    const n = Math.min(a.count, b.count);
    for (let i = 0; i < n; i++) {
      if (Math.hypot(a.keypoints[4 * i] - b.keypoints[4 * i], a.keypoints[4 * i + 1] - b.keypoints[4 * i + 1]) < 1e-3) same++;
    }
    expect(same).toBeGreaterThan(0.98 * n);
  });
});
