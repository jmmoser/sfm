import { describe, expect, it } from 'vitest';
import { DESCRIPTOR_SIZE } from '../core/types';
import { resizeFloat32 } from './image';
import { CANDIDATE_WORDS } from './shaders/sift-wgsl';
import { octaveToFull, planSift, resolveSiftOptions } from './sift-common';
import {
  applyResizeTables,
  assembleFeatures,
  decodeCandidates,
  describeOrder,
  packResizeTables,
  preselectCandidates,
  resolveFullSize,
  scaleSpaceSigmas,
} from './sift-gpu-host';
import { TestRng } from './testdata';

describe('resize tables', () => {
  it('reproduce resizeFloat32 (area shrink, identity axes, bilinear enlarge, u8 scaling)', () => {
    const rng = new TestRng(3);
    for (const [sw, sh, dw, dh] of [
      [97, 61, 40, 25],
      [100, 80, 100, 33],
      [64, 48, 64, 48],
      [40, 30, 97, 71],
      [4032, 3, 1600, 2],
    ]) {
      const src = new Float32Array(sw * sh);
      for (let i = 0; i < src.length; i++) src[i] = rng.next();
      const ref = resizeFloat32(src, sw, sh, dw, dh);
      const t = packResizeTables(sw, sh, dw, dh);
      const got = applyResizeTables(src, sw, dw, dh, t);
      let maxErr = 0;
      for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(ref[i] - got[i]));
      expect(maxErr, `${sw}x${sh}->${dw}x${dh}`).toBeLessThan(2e-6);
      const u8 = new Uint8Array(sw * sh);
      for (let i = 0; i < u8.length; i++) u8[i] = Math.floor(rng.next() * 256);
      const r8 = resizeFloat32(u8, sw, sh, dw, dh);
      const g8 = applyResizeTables(u8, sw, dw, dh, t, 1 / 255);
      for (let i = 0; i < r8.length; i++) expect(Math.abs(r8[i] - g8[i])).toBeLessThan(2e-6);
    }
  });
});

describe('scale-space sigmas', () => {
  it('match the incremental blur schedule of the CPU', () => {
    const o = resolveSiftOptions();
    const { sigmaBase, inc } = scaleSpaceSigmas(o, 0.5);
    expect(sigmaBase).toBeCloseTo(Math.sqrt(1.6 * 1.6 - 0.25), 12);
    // composing the increments reproduces sigma * 2^(i/S)
    let s2 = 1.6 * 1.6;
    for (let i = 1; i <= 5; i++) {
      s2 += inc[i] * inc[i];
      expect(Math.sqrt(s2)).toBeCloseTo(1.6 * 2 ** (i / 3), 10);
    }
    expect(scaleSpaceSigmas(resolveSiftOptions({ sigma: 0.4 }), 0.5).sigmaBase).toBe(0);
  });
});

describe('resolveFullSize', () => {
  it('follows extractSiftCPU (one dimension -> aspect), rejects invalid sizes', () => {
    expect(resolveFullSize(320, 240)).toEqual([320, 240]);
    expect(resolveFullSize(320, 240, 1280)).toEqual([1280, 960]);
    expect(resolveFullSize(320, 240, undefined, 480)).toEqual([640, 480]);
    expect(() => resolveFullSize(320, 240, 0, 10)).toThrow();
    expect(() => resolveFullSize(320, 240, NaN, 10)).toThrow();
  });
});

/** Builds a raw GPU candidate record. */
function record(
  out: Uint32Array,
  k: number,
  c: { xr: number; yr: number; sr: number; v: number; xi: number; yi: number; si: number; s0: number; oct: number; x0: number; y0: number },
): void {
  const f = new Float32Array(out.buffer);
  const o = k * CANDIDATE_WORDS;
  f[o] = c.xr;
  f[o + 1] = c.yr;
  f[o + 2] = c.sr;
  f[o + 3] = c.v;
  out[o + 4] = c.xi | (c.yi << 16);
  out[o + 5] = c.si | (c.s0 << 8) | (c.oct << 16);
  out[o + 6] = c.x0 | (c.y0 << 16);
}

describe('decodeCandidates', () => {
  const o = resolveSiftOptions();
  const plan = planSift(640, 480, o);

  it('restores the CPU detection order and de-duplicates by refined integer location (first wins)', () => {
    const recs = [
      // (oct, s0, y0, x0) start order: B < A < C < D ; A and D converge to the same location
      { name: 'A', xr: 10.2, yr: 20.1, sr: 1.3, v: -0.05, xi: 10, yi: 20, si: 1, s0: 1, oct: 0, x0: 11, y0: 20 },
      { name: 'B', xr: 5.5, yr: 3.4, sr: 2.2, v: 0.03, xi: 5, yi: 3, si: 2, s0: 1, oct: 0, x0: 5, y0: 3 },
      { name: 'C', xr: 7, yr: 8, sr: 2.6, v: 0.02, xi: 7, yi: 8, si: 3, s0: 3, oct: 0, x0: 7, y0: 8 },
      { name: 'D', xr: 10.2, yr: 20.1, sr: 1.3, v: -0.05, xi: 10, yi: 20, si: 1, s0: 2, oct: 0, x0: 10, y0: 20 },
      { name: 'E', xr: 4, yr: 4, sr: 0.6, v: 0.04, xi: 4, yi: 4, si: 1, s0: 1, oct: 1, x0: 4, y0: 4 },
    ];
    const raw = new Uint32Array(recs.length * CANDIDATE_WORDS);
    // shuffled GPU order
    const gpuOrder = [3, 4, 0, 2, 1];
    gpuOrder.forEach((r, k) => record(raw, k, recs[r]));
    const c = decodeCandidates(raw, recs.length, plan, 3, 1.6);
    expect(c.n).toBe(4);
    const xs = Array.from(c.x.subarray(0, c.n)).map((v) => +v.toFixed(2));
    expect(xs).toEqual([5.5, 10.2, 7, 4]); // B, A (D dropped), C, E
    expect(Array.from(c.octIdx.subarray(0, 4))).toEqual([0, 0, 0, 1]);
    expect(Array.from(c.level.subarray(0, 4))).toEqual([2, 1, 3, 1]); // clamp(round(sr), 1, S)
    expect(c.score[1]).toBeCloseTo(0.05, 6);
    expect(c.sigma[0]).toBeCloseTo(1.6 * 2 ** (Math.fround(2.2) / 3), 10);
  });

  it('ignores records beyond the buffer and handles empty input', () => {
    const c = decodeCandidates(new Uint32Array(0), 10, plan, 3, 1.6);
    expect(c.n).toBe(0);
  });
});

describe('preselection, describe order and assembly', () => {
  it('mirror extractSiftCPU (grouping, raster order, mapping, final selection)', () => {
    const o = resolveSiftOptions({ maxFeatures: 5, maxOrientations: 2 });
    const plan = planSift(200, 100, o, 400, 200); // octave -1 first (small image)
    const rng = new TestRng(5);
    const n = 9;
    const raw = new Uint32Array(n * CANDIDATE_WORDS);
    for (let k = 0; k < n; k++) {
      const oct = k % 2;
      const x0 = 3 + 7 * k;
      const y0 = 2 + ((5 * k) % 40);
      record(raw, k, { xr: x0 + 0.25, yr: y0 - 0.25, sr: 1 + (k % 3) * 0.8, v: 0.01 + rng.next() * 0.05, xi: x0, yi: y0, si: 1 + (k % 3), s0: 1 + (k % 3), oct, x0, y0 });
    }
    const c = decodeCandidates(raw, n, plan, o.scalesPerOctave, o.sigma);
    expect(c.n).toBe(n);
    const sel = preselectCandidates(c, plan, o);
    expect(sel.pre.length).toBe(5);
    for (let m = 1; m < sel.pre.length; m++) expect(c.score[sel.pre[m - 1]]).toBeGreaterThanOrEqual(c.score[sel.pre[m]]);
    const ord = describeOrder(c, sel.pre, o.scalesPerOctave);
    // groups are contiguous, sorted by (octave, level), raster inside
    let prevKey = -1;
    for (const g of ord.groups) {
      const key = g.octIdx * 4 + g.level;
      expect(key).toBeGreaterThan(prevKey);
      prevKey = key;
      for (let q = g.start + 1; q < g.end; q++) {
        const i0 = sel.pre[ord.order[q - 1]];
        const i1 = sel.pre[ord.order[q]];
        expect(c.y[i0] < c.y[i1] || (c.y[i0] === c.y[i1] && c.x[i0] <= c.x[i1])).toBe(true);
      }
    }
    expect(ord.kp[0]).toBeCloseTo(c.x[sel.pre[ord.order[0]]], 4);
    // fake GPU output: first slot 2 orientations, others 1, slot 2 none
    const np = ord.order.length;
    const counts = new Uint32Array(np).fill(1);
    counts[0] = 2;
    counts[2] = 0;
    const angles = new Float32Array(np * 2);
    const desc = new Uint8Array(np * 2 * DESCRIPTOR_SIZE);
    for (let s = 0; s < np * 2; s++) {
      angles[s] = 0.1 * s;
      desc.fill(s + 1, s * DESCRIPTOR_SIZE, (s + 1) * DESCRIPTOR_SIZE);
    }
    const fs = assembleFeatures(plan, o, c, sel, ord, counts, angles, desc);
    expect(fs.count).toBe(5); // 6 oriented -> final selection to maxFeatures
    expect([fs.width, fs.height]).toEqual([400, 200]);
    for (let m = 1; m < fs.count; m++) expect(fs.scores[m - 1]).toBeGreaterThanOrEqual(fs.scores[m]);
    // every output keypoint is octaveToFull of its candidate, with its slot's angle / descriptor
    const tmp = new Float32Array(4);
    for (let m = 0; m < fs.count; m++) {
      const slot = fs.descriptors[m * DESCRIPTOR_SIZE] - 1;
      const q = slot >> 1;
      const i = sel.pre[ord.order[q]];
      octaveToFull(plan, plan.octaves[c.octIdx[i]].octave, c.x[i], c.y[i], c.sigma[i], tmp, 0);
      expect(fs.keypoints[4 * m]).toBe(tmp[0]);
      expect(fs.keypoints[4 * m + 1]).toBe(tmp[1]);
      expect(fs.keypoints[4 * m + 2]).toBe(tmp[2]);
      expect(fs.keypoints[4 * m + 3]).toBeCloseTo(0.1 * slot, 6);
      expect(fs.scores[m]).toBe(c.score[i]);
      expect(q).not.toBe(2);
    }
  });
});
