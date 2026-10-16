import { describe, expect, it } from 'vitest';
import type { GrayImage, RGBAImage } from '../core/types';
import { blurFloat32, gaussianBlur, grayToFloat32, resizeGray, rgbaToGray, rotateRGBA90, sharpness } from './image';
import { makeTexturedImage } from './testdata';

function gray(w: number, h: number, f: (x: number, y: number) => number): GrayImage {
  const d = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d[y * w + x] = f(x, y);
  return { width: w, height: h, data: d };
}

describe('rgbaToGray', () => {
  it('uses Rec.709 weights on sRGB values', () => {
    const img: RGBAImage = {
      width: 4,
      height: 1,
      data: new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 0]),
    };
    const g = rgbaToGray(img);
    expect(g.data).toBeInstanceOf(Float32Array);
    expect(g.data[0]).toBeCloseTo(0.2126, 5);
    expect(g.data[1]).toBeCloseTo(0.7152, 5);
    expect(g.data[2]).toBeCloseTo(0.0722, 5);
    expect(g.data[3]).toBeCloseTo(1, 5);
  });

  it('grayToFloat32 scales Uint8 and passes Float32 through', () => {
    const u: GrayImage = { width: 2, height: 1, data: new Uint8Array([0, 255]) };
    expect(Array.from(grayToFloat32(u))).toEqual([0, 1]);
    const f = new Float32Array([0.25, 0.5]);
    expect(grayToFloat32({ width: 2, height: 1, data: f })).toBe(f);
  });
});

describe('resizeGray', () => {
  it('area-averages exact 2x2 blocks', () => {
    const img = gray(4, 4, (x, y) => x + 10 * y);
    const r = resizeGray(img, 2, 2);
    expect(Array.from(r.data)).toEqual([5.5, 7.5, 25.5, 27.5]);
  });

  it('follows the resampling rule (ramp) when shrinking', () => {
    // f(x) = x sampled at pixel centres. For an integer factor the box average equals
    // the ramp at the output centre mapped by x_full = (x_small + 0.5)/s - 0.5; for a
    // non-integer factor it is within the pixel-step quantisation (< 0.1 here).
    for (const [W, w, tol] of [[36, 12, 1e-4], [30, 12, 0.06]]) {
      const img = gray(W, 3, (x) => x);
      const r = resizeGray(img, w, 3);
      const s = w / W;
      let bias = 0;
      for (let i = 0; i < w; i++) {
        const e = r.data[w + i] - ((i + 0.5) / s - 0.5);
        expect(Math.abs(e)).toBeLessThan(tol);
        bias += e;
      }
      expect(Math.abs(bias / w)).toBeLessThan(0.01);
    }
  });

  it('preserves the mean when area-averaging', () => {
    const img = makeTexturedImage(97, 61, 3);
    const r = resizeGray(img, 40, 27);
    const mean = (a: ArrayLike<number>) => Array.from(a).reduce((p, c) => p + c, 0) / a.length;
    expect(mean(r.data)).toBeCloseTo(mean(img.data), 4);
  });

  it('bilinear enlargement follows the resampling rule', () => {
    const img = gray(10, 10, (x, y) => 2 * x + 3 * y);
    const r = resizeGray(img, 20, 20);
    // Interior output pixel i maps to source (i + 0.5)/2 - 0.5.
    for (const [i, j] of [[3, 4], [10, 7], [17, 15]]) {
      const xs = (i + 0.5) / 2 - 0.5;
      const ys = (j + 0.5) / 2 - 0.5;
      expect(r.data[j * 20 + i]).toBeCloseTo(2 * xs + 3 * ys, 4);
    }
  });

  it('keeps constant images constant (both directions)', () => {
    const img: GrayImage = { width: 13, height: 7, data: new Uint8Array(91).fill(128) };
    for (const [w, h] of [[5, 3], [29, 17], [13, 20]]) {
      const r = resizeGray(img, w, h);
      for (const v of r.data) expect(v).toBeCloseTo(128 / 255, 5);
    }
  });
});

describe('resizeGray — reference checks (review)', () => {
  /** Exact 2-D box-overlap area average (shrink) / clamped bilinear (enlarge), per axis. */
  function axisWeights(n: number, m: number): number[][] {
    const s = m / n;
    const W: number[][] = [];
    for (let i = 0; i < m; i++) {
      const row = new Array(n).fill(0);
      if (s < 1) {
        const lo = i / s;
        const hi = (i + 1) / s;
        let sum = 0;
        for (let j = 0; j < n; j++) {
          const ov = Math.max(0, Math.min(hi, j + 1) - Math.max(lo, j));
          row[j] = ov;
          sum += ov;
        }
        for (let j = 0; j < n; j++) row[j] /= sum;
      } else {
        const x = Math.min(n - 1, Math.max(0, (i + 0.5) / s - 0.5));
        const j0 = Math.floor(x);
        const f = x - j0;
        row[j0] += 1 - f;
        row[Math.min(n - 1, j0 + 1)] += f;
      }
      W.push(row);
    }
    return W;
  }

  it('matches a brute-force separable reference for mixed shrink/enlarge factors', () => {
    const src = makeTexturedImage(37, 29, 4);
    for (const [w, h] of [[16, 11], [37, 50], [80, 13], [5, 29], [36, 28]]) {
      const r = resizeGray(src, w, h);
      const Wx = axisWeights(37, w);
      const Wy = axisWeights(29, h);
      let maxErr = 0;
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
          let v = 0;
          for (let j = 0; j < 29; j++) {
            if (!Wy[y][j]) continue;
            for (let i = 0; i < 37; i++) v += Wy[y][j] * Wx[x][i] * src.data[j * 37 + i];
          }
          maxErr = Math.max(maxErr, Math.abs(v - r.data[y * w + x]));
        }
      expect(maxErr, `${w}x${h}`).toBeLessThan(1e-5);
    }
  });

  it('accepts Uint8 input directly (same result as the Float32 path, no full-size copy)', () => {
    const src = makeTexturedImage(101, 77, 6);
    const u8 = new Uint8Array(src.data.length);
    for (let i = 0; i < u8.length; i++) u8[i] = Math.round(src.data[i] * 255);
    const f = grayToFloat32({ width: 101, height: 77, data: u8 });
    for (const [w, h] of [[40, 30], [101, 77], [150, 33], [64, 77]]) {
      const a = resizeGray({ width: 101, height: 77, data: u8 }, w, h);
      const b = resizeGray({ width: 101, height: 77, data: f }, w, h);
      expect(a.data).toBeInstanceOf(Float32Array);
      for (let i = 0; i < a.data.length; i++) expect(Math.abs(a.data[i] - b.data[i])).toBeLessThan(1e-6);
    }
    // sharpness takes the same path
    const big = new Uint8Array(1300 * 700).map((_, i) => (i * 2654435761) >>> 24);
    const bigF = grayToFloat32({ width: 1300, height: 700, data: big });
    expect(sharpness({ width: 1300, height: 700, data: big })).toBeCloseTo(sharpness({ width: 1300, height: 700, data: bigF }), 3);
  });
});

describe('gaussianBlur', () => {
  it('preserves constants and mass, and has the expected variance', () => {
    const N = 81;
    const imp = gray(N, N, (x, y) => (x === 40 && y === 40 ? 1 : 0));
    for (const sigma of [0.8, 1.6, 3.2]) {
      const b = gaussianBlur(imp, sigma);
      let sum = 0;
      let m2 = 0;
      for (let y = 0; y < N; y++)
        for (let x = 0; x < N; x++) {
          const v = b.data[y * N + x];
          sum += v;
          m2 += v * (x - 40) * (x - 40);
        }
      expect(sum).toBeCloseTo(1, 5);
      expect(Math.sqrt(m2)).toBeCloseTo(sigma, 2);
      // symmetry
      expect(b.data[40 * N + 37]).toBeCloseTo(b.data[40 * N + 43], 6);
      expect(b.data[37 * N + 40]).toBeCloseTo(b.data[43 * N + 40], 6);
    }
    const c = gaussianBlur(gray(20, 9, () => 0.3), 2.5);
    for (const v of c.data) expect(v).toBeCloseTo(0.3, 5);
  });

  it('blurFloat32 handles kernels wider than the image', () => {
    const src = new Float32Array([0, 1, 0, 0, 1, 0]);
    const dst = new Float32Array(6);
    blurFloat32(src, 3, 2, 5, dst);
    for (const v of dst) expect(Number.isFinite(v)).toBe(true);
    // replicated borders keep values within the input range
    for (const v of dst) expect(v).toBeGreaterThanOrEqual(0);
  });
});

describe('sharpness', () => {
  it('decreases with blur and is zero for flat images', () => {
    const img = makeTexturedImage(320, 240, 5);
    const s0 = sharpness(img);
    const s1 = sharpness(gaussianBlur(img, 1));
    const s2 = sharpness(gaussianBlur(img, 2.5));
    expect(s0).toBeGreaterThan(s1);
    expect(s1).toBeGreaterThan(s2);
    expect(sharpness(gray(50, 40, () => 0.7))).toBeCloseTo(0, 8);
  });

  it('is computed on a <= 640 px copy (resolution independent for large inputs)', () => {
    const big = makeTexturedImage(1280, 960, 9);
    const small = resizeGray(big, 640, 480);
    expect(sharpness(big)).toBeCloseTo(sharpness(small), 6);
  });
});

describe('rotateRGBA90', () => {
  const W = 5;
  const H = 3;
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) data.set([i, 2 * i, 3 * i, 255], 4 * i);
  const img: RGBAImage = { width: W, height: H, data };
  const px = (im: RGBAImage, x: number, y: number) => im.data[4 * (y * im.width + x)];

  it('rotates clockwise for positive quarter turns', () => {
    const r = rotateRGBA90(img, 1);
    expect(r.width).toBe(H);
    expect(r.height).toBe(W);
    expect(r.data).toBeInstanceOf(Uint8ClampedArray);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        // src(x, y) -> dst(H-1-y, x)
        expect(px(r, H - 1 - y, x)).toBe(px(img, x, y));
      }
  });

  it('composes correctly', () => {
    const r2 = rotateRGBA90(img, 2);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) expect(px(r2, W - 1 - x, H - 1 - y)).toBe(px(img, x, y));
    const r11 = rotateRGBA90(rotateRGBA90(img, 1), 1);
    expect(Array.from(r11.data)).toEqual(Array.from(r2.data));
    const r3 = rotateRGBA90(img, 3);
    const rm1 = rotateRGBA90(img, -1);
    expect(Array.from(r3.data)).toEqual(Array.from(rm1.data));
    const r4 = rotateRGBA90(rotateRGBA90(r3, 1), 0);
    expect(Array.from(r4.data)).toEqual(Array.from(img.data));
    expect(r3.width).toBe(H);
  });

  it('rejects non-finite quarter turns (instead of silently rotating)', () => {
    expect(() => rotateRGBA90(img, NaN)).toThrow();
    expect(() => rotateRGBA90(img, Infinity)).toThrow();
  });

  it('handles unaligned input views', () => {
    const backing = new Uint8Array(W * H * 4 + 1);
    backing.set(data, 1);
    const view = backing.subarray(1);
    const r = rotateRGBA90({ width: W, height: H, data: view }, 1);
    expect(Array.from(r.data)).toEqual(Array.from(rotateRGBA90(img, 1).data));
  });
});
