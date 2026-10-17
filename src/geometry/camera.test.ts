import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics } from '../core/types';
import { Rng } from '../math/random';
import {
  createIntrinsics,
  distortNormalized,
  intrinsicsMatrix,
  intrinsicsMatrixInverse,
  keypointsToNormalized,
  normalizedToPixel,
  pixelToBearing,
  pixelToNormalized,
  projectCam,
  projectWorld,
  scaleIntrinsics,
  undistortNormalized,
  undistortPixel,
} from './camera';
import { randomPoseLookingAtOrigin } from './testutil';

describe('camera intrinsics', () => {
  it('createIntrinsics follows the documented formulas', () => {
    const cam = createIntrinsics(1920, 1440, (70 * Math.PI) / 180, 0.1, -0.02);
    expect(cam.f).toBeCloseTo(960 / Math.tan((35 * Math.PI) / 180), 10);
    expect(cam.cx).toBe(959.5);
    expect(cam.cy).toBe(719.5);
    expect(cam.k1).toBe(0.1);
    expect(cam.k2).toBe(-0.02);
    expect(cam.width).toBe(1920);
    expect(cam.height).toBe(1440);
  });

  it('scaleIntrinsics is consistent with the §2 pixel resampling rule (incl. distortion)', () => {
    const rng = new Rng(1);
    const cam = createIntrinsics(4032, 3024, 1.2, -0.12, 0.03);
    cam.cx += 7.3;
    cam.cy -= 4.1;
    const small = scaleIntrinsics(cam, 1008, 756);
    const s = 1008 / 4032;
    const uv = new Float64Array(2),
      uvs = new Float64Array(2);
    for (let k = 0; k < 200; k++) {
      const X = [rng.uniform(-1, 1), rng.uniform(-0.8, 0.8), rng.uniform(1, 5)];
      expect(projectCam(cam, X, uv)).toBe(true);
      expect(projectCam(small, X, uvs)).toBe(true);
      expect(uvs[0]).toBeCloseTo((uv[0] + 0.5) * s - 0.5, 9);
      expect(uvs[1]).toBeCloseTo((uv[1] + 0.5) * s - 0.5, 9);
    }
    // round trip back to full resolution
    const back = scaleIntrinsics(small, 4032, 3024);
    expect(back.f).toBeCloseTo(cam.f, 9);
    expect(back.cx).toBeCloseTo(cam.cx, 9);
    expect(back.cy).toBeCloseTo(cam.cy, 9);
  });

  it('K and K⁻¹ are inverse', () => {
    const cam = createIntrinsics(640, 480, 1.0);
    const K = intrinsicsMatrix(cam),
      Ki = intrinsicsMatrixInverse(cam);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += K[3 * i + k] * Ki[3 * k + j];
        expect(s).toBeCloseTo(i === j ? 1 : 0, 12);
      }
  });
});

describe('radial distortion', () => {
  it('undistortNormalized inverts distortNormalized (|k| < 0.5) to 1e-10', () => {
    const rng = new Rng(2);
    const out = new Float64Array(2),
      back = new Float64Array(2);
    let worst = 0;
    for (let trial = 0; trial < 4000; trial++) {
      const k1 = rng.uniform(-0.5, 0.5),
        k2 = rng.uniform(-0.5, 0.5);
      const cam: CameraIntrinsics = { width: 100, height: 100, f: 100, cx: 50, cy: 50, k1, k2 };
      // stay inside the monotone branch of g(r) = r (1 + k1 r² + k2 r⁴)
      let rMax = 1.0;
      for (let r = 0.01; r <= 1.0; r += 0.01) {
        const s = r * r;
        if (1 + 3 * k1 * s + 5 * k2 * s * s <= 0.05) {
          rMax = r - 0.02;
          break;
        }
      }
      const r = rng.uniform(0, rMax),
        th = rng.uniform(0, 2 * Math.PI);
      const x = r * Math.cos(th),
        y = r * Math.sin(th);
      distortNormalized(cam, x, y, out);
      undistortNormalized(cam, out[0], out[1], back);
      worst = Math.max(worst, Math.hypot(back[0] - x, back[1] - y));
    }
    expect(worst).toBeLessThan(1e-10);
  });

  it('distort ∘ undistort is the identity on distorted coordinates', () => {
    const rng = new Rng(3);
    const cam = createIntrinsics(1920, 1440, 1.2, -0.15, 0.04);
    const a = new Float64Array(2),
      b = new Float64Array(2);
    for (let k = 0; k < 1000; k++) {
      const xd = rng.uniform(-0.6, 0.6),
        yd = rng.uniform(-0.45, 0.45);
      undistortNormalized(cam, xd, yd, a);
      distortNormalized(cam, a[0], a[1], b);
      expect(Math.hypot(b[0] - xd, b[1] - yd)).toBeLessThan(1e-12);
    }
  });

  it('beyond the invertible radius of a barrel lens the result is clamped (finite, monotone)', () => {
    const cam: CameraIntrinsics = { width: 10, height: 10, f: 1, cx: 0, cy: 0, k1: -0.4, k2: 0 };
    const out = new Float64Array(2);
    // g'(r) = 1 − 1.2 r² = 0 at r* = 0.9129; g(r*) = 0.6086
    const rStar = Math.sqrt(1 / 1.2);
    undistortNormalized(cam, 2, 0, out);
    expect(Number.isFinite(out[0])).toBe(true);
    expect(out[0]).toBeCloseTo(rStar, 9);
    expect(out[1]).toBe(0);
    undistortNormalized(cam, 0, 0, out);
    expect(out[0]).toBe(0);
    expect(out[1]).toBe(0);
  });

  it('zero distortion is the identity and exact', () => {
    const cam = createIntrinsics(640, 480, 1.0);
    const out = new Float64Array(2);
    undistortNormalized(cam, 0.123456789, -0.987654321, out);
    expect(out[0]).toBe(0.123456789);
    expect(out[1]).toBe(-0.987654321);
  });
});

describe('projection', () => {
  it('projectCam / pixelToNormalized / pixelToBearing round trip with distortion', () => {
    const rng = new Rng(4);
    const cam = createIntrinsics(1920, 1440, (68 * Math.PI) / 180, -0.08, 0.02);
    const uv = new Float64Array(2),
      xn = new Float64Array(2),
      b = new Float64Array(3),
      up = new Float64Array(2),
      p2 = new Float64Array(2);
    for (let k = 0; k < 500; k++) {
      const Z = rng.uniform(0.2, 100);
      const X = [rng.uniform(-0.6, 0.6) * Z, rng.uniform(-0.45, 0.45) * Z, Z];
      expect(projectCam(cam, X, uv)).toBe(true);
      pixelToNormalized(cam, uv[0], uv[1], xn);
      expect(xn[0]).toBeCloseTo(X[0] / Z, 10);
      expect(xn[1]).toBeCloseTo(X[1] / Z, 10);
      pixelToBearing(cam, uv[0], uv[1], b);
      const n = Math.hypot(X[0], X[1], X[2]);
      expect(Math.hypot(b[0], b[1], b[2])).toBeCloseTo(1, 14);
      expect(b[0]).toBeCloseTo(X[0] / n, 10);
      expect(b[2]).toBeCloseTo(X[2] / n, 10);
      normalizedToPixel(cam, xn[0], xn[1], p2);
      expect(p2[0]).toBeCloseTo(uv[0], 7);
      expect(p2[1]).toBeCloseTo(uv[1], 7);
      undistortPixel(cam, uv[0], uv[1], up);
      expect(up[0]).toBeCloseTo((cam.f * X[0]) / Z + cam.cx, 7);
    }
  });

  it('projectCam rejects points at or behind the camera', () => {
    const cam = createIntrinsics(640, 480, 1);
    const out = Float64Array.of(-7, -7);
    expect(projectCam(cam, [0, 0, 0], out)).toBe(false);
    expect(projectCam(cam, [1, 1, -1], out)).toBe(false);
    expect(projectCam(cam, [1, 1, 1e-13], out)).toBe(false);
    expect(out[0]).toBe(-7);
    expect(projectCam(cam, [0, 0, 2e-12], out)).toBe(true);
    expect(out[0]).toBeCloseTo(cam.cx, 9);
  });

  it('projectWorld equals projectCam of the transformed point (principal point at integer pixel convention)', () => {
    const rng = new Rng(5);
    const cam = createIntrinsics(1000, 800, 1.1, 0.05, -0.01);
    const pose = randomPoseLookingAtOrigin(rng, 5);
    const a = new Float64Array(2),
      b = new Float64Array(2);
    for (let k = 0; k < 100; k++) {
      const X = [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)];
      const R = pose.R,
        t = pose.t;
      const P = [0, 1, 2].map((i) => R[3 * i] * X[0] + R[3 * i + 1] * X[1] + R[3 * i + 2] * X[2] + t[i]);
      expect(projectWorld(cam, pose, X, a)).toBe(projectCam(cam, P, b));
      expect(a[0]).toBeCloseTo(b[0], 9);
      expect(a[1]).toBeCloseTo(b[1], 9);
    }
    // the optical axis hits the principal point (cx, cy) = ((W−1)/2, (H−1)/2)
    expect(projectCam(cam, [0, 0, 3], a)).toBe(true);
    expect(a[0]).toBe(499.5);
    expect(a[1]).toBe(399.5);
  });

  it('keypointsToNormalized matches pixelToNormalized and accepts a preallocated buffer', () => {
    const rng = new Rng(6);
    const cam = createIntrinsics(1920, 1440, 1.2, -0.1, 0.01);
    const n = 300;
    const kp = new Float32Array(4 * n);
    for (let i = 0; i < n; i++) {
      kp[4 * i] = rng.uniform(0, 1920);
      kp[4 * i + 1] = rng.uniform(0, 1440);
      kp[4 * i + 2] = 2;
      kp[4 * i + 3] = 0.3;
    }
    const buf = new Float64Array(2 * n + 10);
    const out = keypointsToNormalized(cam, kp, buf);
    expect(out).toBe(buf);
    const xn = new Float64Array(2);
    for (let i = 0; i < n; i++) {
      pixelToNormalized(cam, kp[4 * i], kp[4 * i + 1], xn);
      expect(out[2 * i]).toBe(xn[0]);
      expect(out[2 * i + 1]).toBe(xn[1]);
    }
    expect(keypointsToNormalized(cam, kp).length).toBe(2 * n);
  });
});
