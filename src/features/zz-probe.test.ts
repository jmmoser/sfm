import { describe, expect, it } from 'vitest';
import type { FeatureSet, GrayImage, Mat3 } from '../core/types';
import { matchCPU, matchGuidedCPU } from './match';
import { extractSiftCPU } from './sift-cpu';
import { applyHomography, evaluateMatches, makeTexturedImage, makeWarpedPair, similarityAboutCenter } from './testdata';

function wrap(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a <= -Math.PI) a += 2 * Math.PI;
  return a;
}

function correctPairs(a: FeatureSet, b: FeatureSet, m: Uint32Array, H: Mat3): [number, number][] {
  const out: [number, number][] = [];
  const q = [0, 0];
  for (let k = 0; k < m.length; k += 2) {
    const i = m[k];
    const j = m[k + 1];
    applyHomography(H, a.keypoints[4 * i], a.keypoints[4 * i + 1], q);
    if (Math.hypot(q[0] - b.keypoints[4 * j], q[1] - b.keypoints[4 * j + 1]) < 1.5) out.push([i, j]);
  }
  return out;
}

describe('probe', () => {
  it('orientation sign under +30 deg (clockwise displayed) rotation', () => {
    const W = 640, H = 480;
    for (const ang of [30, -50, 90]) {
      const p = makeWarpedPair(W, H, similarityAboutCenter(W, H, ang, 0.8), 17);
      const a = extractSiftCPU(p.a);
      const b = extractSiftCPU(p.b);
      const cp = correctPairs(a, b, matchCPU(a, b), p.H);
      const d: number[] = cp.map(([i, j]) => wrap(b.keypoints[4 * j + 3] - a.keypoints[4 * i + 3]));
      d.sort((x, y) => x - y);
      const lr = cp.map(([i, j]) => Math.log(b.keypoints[4 * j + 2] / a.keypoints[4 * i + 2])).sort((x, y) => x - y);
      console.log(`ang ${ang}: n=${cp.length} median dtheta=${((d[d.length >> 1] * 180) / Math.PI).toFixed(2)} deg; scale ratio median ${Math.exp(lr[lr.length >> 1]).toFixed(4)}`);
    }
  });

  it('negated H guide', () => {
    const W = 640, H = 480;
    const p = makeWarpedPair(W, H, similarityAboutCenter(W, H, 20, 0.9), 3);
    const a = extractSiftCPU(p.a);
    const b = extractSiftCPU(p.b);
    const Hn = new Float64Array(p.H).map((v) => -v);
    const g1 = matchGuidedCPU(a, b, { kind: 'H', H: p.H, maxErrorPx: 4 });
    const g2 = matchGuidedCPU(a, b, { kind: 'H', H: Hn, maxErrorPx: 4 });
    console.log(`H: ${g1.length / 2}, -H: ${g2.length / 2}`);
  });

  it('odd sizes / tiny', () => {
    for (const [w, h] of [[641, 481], [17, 33], [16, 16], [33, 17], [1001, 23], [23, 1001], [100, 15]]) {
      const img = makeTexturedImage(w, h, 3);
      const fs = extractSiftCPU(img);
      let bad = 0;
      for (let i = 0; i < fs.count; i++) {
        const x = fs.keypoints[4 * i], y = fs.keypoints[4 * i + 1];
        if (!(x >= -0.5 && x <= w - 0.5 && y >= -0.5 && y <= h - 0.5)) bad++;
        for (let k = 0; k < 4; k++) if (!Number.isFinite(fs.keypoints[4 * i + k])) bad++;
      }
      console.log(`${w}x${h}: ${fs.count} feats, bad ${bad}`);
    }
  });

  it('descriptor absolute layout: edge to the right of a blob', () => {
    // dark blob at the centre; a bright->dark vertical edge to its right
    const W = 128, H = 128;
    const d = new Float32Array(W * H);
    const cx = 63.3, cy = 64.2, s = 4;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let v = 0.5 - 0.35 * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s));
        // vertical step edge at x = cx + 12 (dark on right), smooth
        v += -0.2 / (1 + Math.exp(-(x - (cx + 12)) / 0.7)) ;
        d[y * W + x] = v;
      }
    const img: GrayImage = { width: W, height: H, data: d };
    const fs = extractSiftCPU(img, { upsampleBelow: 0 });
    let best = -1, bd = 1e9;
    for (let i = 0; i < fs.count; i++) {
      const dd = Math.hypot(fs.keypoints[4 * i] - cx, fs.keypoints[4 * i + 1] - cy);
      if (dd < bd && fs.keypoints[4*i+2] > 2.5) { bd = dd; best = i; }
    }
    console.log('n', fs.count, 'best', best, bd, Array.from(fs.keypoints.subarray(4 * best, 4 * best + 4)));
    for (let i = 0; i < fs.count; i++) console.log(Array.from(fs.keypoints.subarray(4*i, 4*i+4)).map(v=>v.toFixed(2)).join(' '));
    const desc = fs.descriptors.subarray(128 * best, 128 * best + 128);
    // energy per spatial bin and per orientation bin
    const sp: number[] = [];
    for (let by = 0; by < 4; by++) {
      const row: string[] = [];
      for (let bx = 0; bx < 4; bx++) {
        let e = 0;
        for (let o = 0; o < 8; o++) e += desc[(by * 4 + bx) * 8 + o] ** 2;
        row.push(e.toFixed(0).padStart(7));
      }
      sp.push(row.join(' '));
    }
    console.log(sp.join('\n'));
    const ob: number[] = new Array(8).fill(0);
    for (let k = 0; k < 128; k++) ob[k % 8] += desc[k] ** 2;
    console.log('orient bins', ob.map((v) => v.toFixed(0)).join(' '));
  });
});
