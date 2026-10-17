import { describe, expect, it } from 'vitest';
import {
  BandController,
  CHECKER_OFFSETS,
  CHECKER_REGION_SIZES,
  GPU_SOURCE_CAMERA_FLOATS,
  buildSourceCamera,
  checkerOffset,
  packSourceCameras,
  passSeed,
  patchOffsets,
  pcgHash,
  planeHomography,
  planesToDepthMap,
  priorToPlanes,
  relativePose,
  resolvePatchMatchOptions,
  rgbaToGray8,
} from './patchmatch-common';
import { lookAt, pinholeIntrinsics } from './synthetic';
import type { DenseView, DepthMap } from './types';

function view(id: number, eye: number[], target: number[], w = 64, h = 48): DenseView {
  const cam = pinholeIntrinsics(w, h, 1.0);
  return { id, cam, pose: lookAt(eye, target), image: { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) } };
}

function project(v: DenseView, X: number[]): [number, number, number] {
  const R = v.pose.R;
  const t = v.pose.t;
  const xc = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
  const yc = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
  const zc = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
  return [v.cam.f * (xc / zc) + v.cam.cx, v.cam.f * (yc / zc) + v.cam.cy, zc];
}

/** World point of ref pixel (u, v) at depth z. */
function backproject(v: DenseView, u: number, w: number, z: number): number[] {
  const xc = ((u - v.cam.cx) / v.cam.f) * z;
  const yc = ((w - v.cam.cy) / v.cam.f) * z;
  const R = v.pose.R;
  const t = v.pose.t;
  const d = [xc - t[0], yc - t[1], z - t[2]];
  return [R[0] * d[0] + R[3] * d[1] + R[6] * d[2], R[1] * d[0] + R[4] * d[1] + R[7] * d[2], R[2] * d[0] + R[5] * d[1] + R[8] * d[2]];
}

describe('patchmatch-common', () => {
  it('relativePose maps reference-frame points to the source frame', () => {
    const ref = view(0, [0, -3, 1], [0, 0, 0]);
    const src = view(1, [0.7, -2.8, 1.3], [0.1, 0, 0]);
    const { R, t } = relativePose(ref.pose, src.pose);
    const X = [0.3, 0.2, 0.1];
    const Rr = ref.pose.R;
    const xr = [0, 1, 2].map((i) => Rr[3 * i] * X[0] + Rr[3 * i + 1] * X[1] + Rr[3 * i + 2] * X[2] + ref.pose.t[i]);
    const Rs = src.pose.R;
    const xs = [0, 1, 2].map((i) => Rs[3 * i] * X[0] + Rs[3 * i + 1] * X[1] + Rs[3 * i + 2] * X[2] + src.pose.t[i]);
    for (let i = 0; i < 3; i++) {
      expect(R[3 * i] * xr[0] + R[3 * i + 1] * xr[1] + R[3 * i + 2] * xr[2] + t[i]).toBeCloseTo(xs[i], 10);
    }
  });

  it('plane-induced homography matches explicit ray-plane intersection + projection', () => {
    const ref = view(0, [0, -3, 1], [0, 0, 0]);
    const src = view(1, [0.8, -2.6, 1.4], [0.2, 0.1, 0]);
    const sc = buildSourceCamera(ref, src);
    // a slanted plane through the point at pixel (30, 20), depth 2.7
    const u0 = 30;
    const v0 = 20;
    const z0 = 2.7;
    let n = [0.3, -0.4, -0.8];
    const nl = Math.hypot(n[0], n[1], n[2]);
    n = n.map((c) => c / nl);
    const H = planeHomography(ref.cam, sc, u0, v0, z0, n)!;
    expect(H).not.toBeNull();
    // plane: n·X_r + δ = 0 with δ = −z0 n·r(p0)
    const r0 = [(u0 - ref.cam.cx) / ref.cam.f, (v0 - ref.cam.cy) / ref.cam.f, 1];
    const delta = -z0 * (n[0] * r0[0] + n[1] * r0[1] + n[2] * r0[2]);
    expect(delta).toBeGreaterThan(0);
    for (const [du, dv] of [
      [0, 0],
      [5, -3],
      [-7, 6],
      [12, 9],
    ]) {
      const u = u0 + du;
      const v = v0 + dv;
      const r = [(u - ref.cam.cx) / ref.cam.f, (v - ref.cam.cy) / ref.cam.f, 1];
      const z = -delta / (n[0] * r[0] + n[1] * r[1] + n[2] * r[2]);
      const X = backproject(ref, u, v, z);
      const [pu, pv] = project(src, X);
      const hx = H[0] * u + H[1] * v + H[2];
      const hy = H[3] * u + H[4] * v + H[5];
      const hz = H[6] * u + H[7] * v + H[8];
      expect(hx / hz).toBeCloseTo(pu, 8);
      expect(hy / hz).toBeCloseTo(pv, 8);
    }
    // plane seen from behind -> null
    expect(planeHomography(ref.cam, sc, u0, v0, z0, [-n[0], -n[1], -n[2]])).toBeNull();
  });

  it('back-projection matrices (M, v) invert (A, b)', () => {
    const ref = view(0, [0, -3, 1], [0, 0, 0]);
    const src = view(1, [0.8, -2.6, 1.4], [0.2, 0.1, 0]);
    const sc = buildSourceCamera(ref, src);
    const u = 21;
    const v = 33;
    const z = 3.1;
    const X = backproject(ref, u, v, z);
    const [pu, pv, zs] = project(src, X);
    // K_r X_r' = zs M p̃_s + v
    const p = [pu, pv, 1];
    const k = [0, 1, 2].map((i) => zs * (sc.M[3 * i] * p[0] + sc.M[3 * i + 1] * p[1] + sc.M[3 * i + 2] * p[2]) + sc.v[i]);
    expect(k[0] / k[2]).toBeCloseTo(u, 8);
    expect(k[1] / k[2]).toBeCloseTo(v, 8);
    expect(k[2]).toBeCloseTo(z, 8);
    // forward: p_s ≅ z A p̃ + b
    const q = [0, 1, 2].map((i) => z * (sc.A[3 * i] * u + sc.A[3 * i + 1] * v + sc.A[3 * i + 2]) + sc.b[i]);
    expect(q[0] / q[2]).toBeCloseTo(pu, 8);
    expect(q[1] / q[2]).toBeCloseTo(pv, 8);
  });

  it('packs source cameras in the documented 40-float layout', () => {
    const ref = view(0, [0, -3, 1], [0, 0, 0]);
    const src = view(1, [0.8, -2.6, 1.4], [0.2, 0.1, 0], 50, 40);
    const sc = buildSourceCamera(ref, src, { width: 25, height: 20 });
    const packed = packSourceCameras([sc, sc], 64, 48);
    expect(packed.length).toBe(16 * GPU_SOURCE_CAMERA_FLOATS);
    const b = GPU_SOURCE_CAMERA_FLOATS;
    expect(packed[b + 0]).toBeCloseTo(sc.A[0], 5);
    expect(packed[b + 6]).toBeCloseTo(sc.A[5], 5);
    expect(packed[b + 14]).toBeCloseTo(sc.b[2], 4);
    expect(packed[b + 16 + 9]).toBeCloseTo(sc.M[7], 5);
    expect(packed[b + 30]).toBeCloseTo(sc.v[2], 4);
    expect([packed[b + 32], packed[b + 33]]).toEqual([50, 40]);
    expect(packed[b + 34]).toBeCloseTo(1 / 64, 7);
    expect([packed[b + 36], packed[b + 37], packed[b + 38], packed[b + 39]]).toEqual([0.5, 0.5, 25, 20]);
  });

  it('checkerboard pattern: procedural form equals the table, all on the opposite colour', () => {
    let k = 0;
    expect(CHECKER_OFFSETS.length).toBe(2 * 72);
    const seen = new Set<string>();
    for (let r = 0; r < 8; r++) {
      for (let i = 0; i < CHECKER_REGION_SIZES[r]; i++, k++) {
        const [dx, dy] = checkerOffset(r, i);
        expect([dx, dy]).toEqual([CHECKER_OFFSETS[2 * k], CHECKER_OFFSETS[2 * k + 1]]);
        expect(Math.abs(dx + dy) % 2).toBe(1);
        seen.add(`${dx},${dy}`);
      }
    }
    expect(seen.size).toBe(72);
  });

  it('PCG hash is deterministic and roughly uniform', () => {
    expect(pcgHash(0)).toBe(pcgHash(0));
    expect(pcgHash(1)).not.toBe(pcgHash(2));
    expect(passSeed(1, 3)).not.toBe(passSeed(1, 4));
    let s = 12345;
    let sum = 0;
    const n = 20000;
    const hist = new Uint32Array(10);
    for (let i = 0; i < n; i++) {
      s = pcgHash(s);
      const u = (s >>> 8) / 16777216;
      sum += u;
      hist[Math.floor(u * 10)]++;
    }
    expect(sum / n).toBeGreaterThan(0.48);
    expect(sum / n).toBeLessThan(0.52);
    for (const h of hist) expect(Math.abs(h - n / 10)).toBeLessThan(n / 50);
  });

  it('resamples a prior depth map (nearest, §2 rule) and finalises depth maps', () => {
    const prior: DepthMap = {
      width: 2,
      height: 2,
      depth: new Float32Array([1, 2, 0, 4]),
      normal: new Float32Array([0, 0, -1, 0, 0.6, -0.8, 0, 0, -1, 0.6, 0, -0.8]),
      cost: new Float32Array(4),
    };
    const planes = priorToPlanes(prior, 4, 4);
    // pixel (3, 0) -> prior (1, 0)
    expect(Array.from(planes.subarray(4 * 3, 4 * 3 + 4)).map((v) => +v.toFixed(3))).toEqual([0, 0.6, -0.8, 2]);
    // pixel (0, 3) -> prior (0, 1): invalid
    expect(planes[4 * 12 + 3]).toBe(0);
    const photo = new Float32Array(16).fill(0.1);
    photo[15] = 0.9;
    const dm = planesToDepthMap(planes, photo, 4, 4, { min: 0.5, max: 3.5 }, 0.5);
    expect(dm.depth[0]).toBe(1);
    expect(dm.depth[3]).toBe(2);
    expect(dm.depth[12]).toBe(0); // invalid prior
    expect(dm.depth[15]).toBe(0); // depth 4 out of range (and cost too high)
    expect(dm.normal[3 * 3 + 1]).toBeCloseTo(0.6, 6);
  });

  it('resolves options and patch offsets', () => {
    const o = resolvePatchMatchOptions({ iterations: 3, numSamplesPerIter: 10 });
    expect(o.iterations).toBe(3);
    expect(o.patchRadius).toBe(5);
    expect(o.numSamplesPerIter).toBe(6);
    expect(patchOffsets(5, 2)).toEqual([-5, -3, -1, 1, 3, 5]);
    expect(patchOffsets(4, 2)).toEqual([-4, -2, 0, 2, 4]);
  });

  it('converts RGBA to Rec.709 grey', () => {
    const g = rgbaToGray8({ width: 2, height: 1, data: new Uint8ClampedArray([255, 255, 255, 255, 255, 0, 0, 255]) });
    expect(Array.from(g)).toEqual([255, 54]);
  });

  it('BandController adapts band height to the target time per submission', () => {
    const c = new BandController(20, 32, 400, 8);
    expect(c.next()).toBe(32);
    c.update(32, 64); // 2 ms/row -> 10 rows
    expect(c.next()).toBe(10);
    c.update(10, 2); // fast now: 0.2 ms/row -> average 1.1 ms/row -> 18 rows
    expect(c.next()).toBe(18);
    for (let i = 0; i < 20; i++) c.update(c.next(), 0.01 * c.next());
    expect(c.next()).toBe(400); // capped at maxRows
    for (let i = 0; i < 20; i++) c.update(c.next(), 100 * c.next());
    expect(c.next()).toBe(8); // floored at minRows
    const fixed = new BandController(Infinity, 77, 100, 1);
    fixed.update(77, 1000);
    expect(fixed.next()).toBe(77);
  });
});
