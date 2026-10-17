import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, RGBAImage } from '../../core/types';
import { removeDepthHalos } from '../../dense/depth-filter';
import type { DepthMap } from '../../dense/types';
import { Rng } from '../../math/random';
import { removeDepthHalosRotated, rotateDepthMap } from './halo-rotation';

const W = 80;
const H = 60;
const SKY = 22;
const cam: CameraIntrinsics = { width: W, height: H, f: 70, cx: 39.5, cy: 29.5, k1: 0, k2: 0 };

/** Smooth sky (sparse random depths, a 10-px band on the wall's plane) above a slanted textured wall. */
function scene(): { dm: DepthMap; image: RGBAImage } {
  const rng = new Rng(7);
  const n = W * H;
  const depth = new Float32Array(n);
  const normal = new Float32Array(3 * n);
  const cost = new Float32Array(n).fill(0.2);
  const data = new Uint8ClampedArray(4 * n);
  const nl = Math.hypot(0.3, -0.2, -1);
  const nrm = [0.3 / nl, -0.2 / nl, -1 / nl];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const sky = y < SKY;
      const t = sky ? 2 * rng.next() - 1 : 25 * (2 * rng.next() - 1);
      data.set(sky ? [150 + 0.5 * y + t, 185 + 0.3 * y + t, 230 + t, 255] : [150 + t, 60 + t, 40 + t, 255], 4 * i);
      const nr = (nrm[0] * (x - cam.cx)) / cam.f + (nrm[1] * (y - cam.cy)) / cam.f + nrm[2];
      if (!sky || y >= SKY - 10) {
        depth[i] = (10 * nrm[2]) / nr;
        normal.set(nrm, 3 * i);
      } else if (i % 3 === 0) {
        depth[i] = 2 + 30 * rng.next();
        normal[3 * i + 2] = -1;
      }
    }
  }
  return { dm: { width: W, height: H, depth, normal, cost }, image: { width: W, height: H, data } };
}

describe('removeDepthHalosRotated (orientation experiment of the dense GT harness)', () => {
  it('rotateDepthMap: four turns are the identity; one turn moves the top row to the right column', () => {
    const { dm } = scene();
    const r1 = rotateDepthMap(dm, 1);
    expect([r1.width, r1.height]).toEqual([H, W]);
    // pixel (x, 0) → (H − 1, x); normal (nx, ny, nz) → (−ny, nx, nz)
    const i = 5;
    const j = 5 * H + (H - 1);
    expect(r1.depth[j]).toBe(dm.depth[i]);
    expect(Array.from(r1.normal.subarray(3 * j, 3 * j + 3))).toEqual([-dm.normal[3 * i + 1], dm.normal[3 * i], dm.normal[3 * i + 2]]);
    const r4 = rotateDepthMap(rotateDepthMap(r1, 2), 1);
    expect(r4.depth).toEqual(dm.depth);
    expect(r4.normal).toEqual(dm.normal);
  });

  it('with skyUp rotated along, the result matches the upright filter for every quarter turn; without it, 180° misses the sky', () => {
    const { dm, image } = scene();
    const up = [0, -1, 0];
    const ref = removeDepthHalos(dm, image, cam, { skyUp: up });
    // every sky depth (sparse + band) is removed, the wall stays
    const valid = { sky: 0, wall: 0 };
    const removed = { sky: 0, wall: 0 };
    for (let i = 0; i < W * H; i++) {
      if (!(dm.depth[i] > 0)) continue;
      const r = i < SKY * W ? 'sky' : 'wall';
      valid[r]++;
      if (!(ref.depth[i] > 0)) removed[r]++;
    }
    expect(removed.sky).toBe(valid.sky);
    expect(removed.wall).toBeLessThan(0.01 * valid.wall);
    for (const k of [1, 2, 3]) {
      const out = removeDepthHalosRotated(dm, image, cam, { skyUp: up }, k);
      let same = 0;
      for (let i = 0; i < W * H; i++) same += (out.depth[i] > 0) === (ref.depth[i] > 0) ? 1 : 0;
      expect(same / (W * H)).toBeGreaterThan(0.995);
    }
    // upside down without skyUp: the top-row seeds see only the wall, the band survives
    const blind = removeDepthHalosRotated(dm, image, cam, {}, 2);
    let bandKept = 0;
    for (let y = SKY - 10; y < SKY - 2; y++) for (let x = 10; x < W - 10; x++) bandKept += blind.depth[y * W + x] > 0 ? 1 : 0;
    expect(bandKept).toBeGreaterThan(0);
  });
});
