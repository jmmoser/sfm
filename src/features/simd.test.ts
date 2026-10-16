import { afterEach, describe, expect, it } from 'vitest';
import { blurFloat32, gaussianHalfKernel } from './image';
import { matchCPU, matchUsesSIMD } from './match';
import { extractSiftCPU, lastSiftCPUTimings } from './sift-cpu';
import { createSimdKernels, simdAvailable } from './simd';
import { TestRng, makeTexturedImage, makeWarpedPair, similarityAboutCenter } from './testdata';

const g = globalThis as { SFM_DISABLE_SIMD?: boolean };

afterEach(() => {
  delete g.SFM_DISABLE_SIMD;
});

describe('WASM SIMD kernels', () => {
  it('are available in Node and can be disabled', () => {
    expect(simdAvailable()).toBe(true);
    g.SFM_DISABLE_SIMD = true;
    expect(simdAvailable()).toBe(false);
    expect(createSimdKernels()).toBeNull();
    expect(matchUsesSIMD()).toBe(false);
  });

  it('blur kernels match the JS blur', () => {
    const k = createSimdKernels(1 << 20)!;
    const rng = new TestRng(5);
    for (const [w, h] of [[37, 23], [16, 16], [5, 70], [64, 3]]) {
      const n = w * h;
      const src = new Float32Array(n).map(() => rng.next());
      const ref = new Float32Array(n);
      for (const sigma of [0.6, 1.6, 3.3, 7]) {
        blurFloat32(src, w, h, sigma, ref);
        const kern = gaussianHalfKernel(sigma);
        const r = kern.length - 1;
        const mem = k.memory.buffer;
        const srcOff = 0;
        const tmpOff = 65536;
        const dstOff = 131072;
        const kvOff = 196608;
        const padOff = 200704;
        new Float32Array(mem, srcOff, n).set(src);
        new Float32Array(mem, kvOff, r + 1).set(kern);
        k.blurH(srcOff, tmpOff, w, h, kvOff, r, padOff);
        k.blurV(tmpOff, dstOff, w, h, kvOff, r);
        const out = new Float32Array(mem, dstOff, n);
        for (let i = 0; i < n; i++) expect(Math.abs(out[i] - ref[i])).toBeLessThan(1e-6);
      }
    }
  });

  it('dot4 computes exact integer dot products', () => {
    const k = createSimdKernels(1 << 20)!;
    const rng = new TestRng(6);
    const nb = 13;
    const A = new Int16Array(4 * 128).map(() => rng.int(256));
    const B = new Int16Array(nb * 128).map(() => rng.int(256));
    new Int16Array(k.memory.buffer, 0, A.length).set(A);
    new Int16Array(k.memory.buffer, 4096, B.length).set(B);
    k.dot4(0, 4096, nb, 16384);
    const out = new Int32Array(k.memory.buffer, 16384, 4 * nb);
    for (let j = 0; j < nb; j++)
      for (let q = 0; q < 4; q++) {
        let s = 0;
        for (let t = 0; t < 128; t++) s += A[q * 128 + t] * B[j * 128 + t];
        expect(out[4 * j + q]).toBe(s);
      }
  });
});

describe('SIMD gradient kernel', () => {
  it('matches central differences + atan2 on interior pixels', () => {
    const k = createSimdKernels(1 << 20)!;
    const rng = new TestRng(9);
    const w = 23;
    const h = 11;
    const n = w * h;
    const L = new Float32Array(n).map(() => rng.next());
    // exact zeros / axis-aligned gradients too
    for (let x = 0; x < w; x++) L[5 * w + x] = 0.5;
    const mem = k.memory.buffer;
    new Float32Array(mem, 0, n).set(L);
    new Float32Array(mem, 8192, n).fill(-1);
    new Float32Array(mem, 16384, n).fill(-1);
    k.grad(0, 8192, 16384, w, h);
    const mag = new Float32Array(mem, 8192, n);
    const ang = new Float32Array(mem, 16384, n);
    for (let y = 1; y < h - 1; y++)
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const gx = 0.5 * (L[i + 1] - L[i - 1]);
        const gy = 0.5 * (L[i + w] - L[i - w]);
        expect(mag[i]).toBeCloseTo(Math.hypot(gx, gy), 6);
        let a = Math.atan2(gy, gx);
        if (a < 0) a += 2 * Math.PI;
        expect(ang[i]).toBeGreaterThanOrEqual(0);
        expect(ang[i]).toBeLessThan(2 * Math.PI);
        let d = Math.abs(ang[i] - a);
        d = Math.min(d, 2 * Math.PI - d);
        if (gx !== 0 || gy !== 0) expect(d).toBeLessThan(3e-5);
      }
  });
});

describe('SIMD and JS paths agree', () => {
  it('matchCPU is bit-identical', () => {
    const p = makeWarpedPair(640, 480, similarityAboutCenter(640, 480, 20, 0.9), 3);
    const a = extractSiftCPU(p.a, { maxFeatures: 1501 });
    const b = extractSiftCPU(p.b, { maxFeatures: 1203 });
    expect(a.count % 4).not.toBe(0);
    expect(matchUsesSIMD()).toBe(true);
    const ms = matchCPU(a, b);
    const ms2 = matchCPU(a, b, { mutual: false, ratio: 0.9 });
    g.SFM_DISABLE_SIMD = true;
    const mj = matchCPU(a, b);
    const mj2 = matchCPU(a, b, { mutual: false, ratio: 0.9 });
    expect(Array.from(ms)).toEqual(Array.from(mj));
    expect(Array.from(ms2)).toEqual(Array.from(mj2));
    expect(ms.length).toBeGreaterThan(1000);
  });

  it('extractSiftCPU gives (numerically) the same features', () => {
    const img = makeTexturedImage(500, 400, 8);
    const s = extractSiftCPU(img);
    expect(lastSiftCPUTimings.simd).toBe(1);
    g.SFM_DISABLE_SIMD = true;
    const j = extractSiftCPU(img);
    expect(lastSiftCPUTimings.simd).toBe(0);
    expect(Math.abs(s.count - j.count)).toBeLessThanOrEqual(Math.ceil(0.01 * j.count));
    // Same keypoints (blur differs only by float rounding) and descriptors (f32 vs f64
    // accumulation: at most +-1 quantisation steps).
    let same = 0;
    let maxDiff = 0;
    let diffBytes = 0;
    const n = Math.min(s.count, j.count);
    for (let i = 0; i < n; i++) {
      const dx = s.keypoints[4 * i] - j.keypoints[4 * i];
      const dy = s.keypoints[4 * i + 1] - j.keypoints[4 * i + 1];
      const dth = s.keypoints[4 * i + 3] - j.keypoints[4 * i + 3];
      if (Math.hypot(dx, dy) > 1e-3 || Math.abs(dth) > 1e-3) continue;
      same++;
      for (let k = 0; k < 128; k++) {
        const d = Math.abs(s.descriptors[128 * i + k] - j.descriptors[128 * i + k]);
        maxDiff = Math.max(maxDiff, d);
        if (d) diffBytes++;
      }
    }
    console.log(`[simd] ${same}/${n} identical keypoints, descriptor bytes differing: ${diffBytes}, max diff ${maxDiff}`);
    expect(same).toBeGreaterThan(0.97 * n);
    expect(maxDiff).toBeLessThanOrEqual(2);
    expect(diffBytes).toBeLessThan(0.01 * same * 128);
  });
});
