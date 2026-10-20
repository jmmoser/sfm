import { describe, expect, it } from 'vitest';
import {
  fitInverseDepthPlane,
  surfaceDepthAt,
  boundsSphere,
  cameraCenter,
  cameraUp,
  clampedFullBounds,
  estimateUpVector,
  formatDistance,
  goldenStridePermutation,
  linearizeDepth,
  niceStep,
  percentileAlong,
  pickNearestProjected,
  robustBounds,
  subsamplePositions,
  symEig3,
  unpackRGBAToDepth,
} from './helpers';
import { frustumCorners, writeFrustumLines, VERTICES_PER_CAMERA } from './cameras';
import type { Mat3, Pose } from '../core/types';
import { cameraFromDevice } from '../sensors/frame-orientation';

// Small deterministic PRNG for test data (mulberry32).
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function mul3(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const o = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c];
  return o;
}

function axisAngle(axis: number[], ang: number): Float64Array {
  const n = Math.hypot(axis[0], axis[1], axis[2]);
  const [x, y, z] = axis.map((v) => v / n);
  const c = Math.cos(ang);
  const s = Math.sin(ang);
  const C = 1 - c;
  return new Float64Array([
    c + x * x * C, x * y * C - z * s, x * z * C + y * s,
    y * x * C + z * s, c + y * y * C, y * z * C - x * s,
    z * x * C - y * s, z * y * C + x * s, c + z * z * C,
  ]);
}

/** World-to-camera rotation for a camera in a Z-up world with yaw ψ, pitch θ (up +), roll φ. */
function camRotation(yaw: number, pitch: number, roll: number): Float64Array {
  const f = [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)];
  const r = [Math.sin(yaw), -Math.cos(yaw), 0];
  const d = [f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0]];
  const R = new Float64Array([...r, ...d, ...f]);
  return mul3(axisAngle([0, 0, 1], roll), R); // roll about the optical axis (camera frame)
}

function rotate(R: ArrayLike<number>, v: ArrayLike<number>): number[] {
  return [0, 1, 2].map((r) => R[3 * r] * v[0] + R[3 * r + 1] * v[1] + R[3 * r + 2] * v[2]);
}

function angleDeg(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const d = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (Math.hypot(a[0], a[1], a[2]) * Math.hypot(b[0], b[1], b[2]));
  return (Math.acos(Math.min(1, Math.max(-1, d))) * 180) / Math.PI;
}

describe('robustBounds', () => {
  it('ignores outliers via 2–98 percentiles and reports full extent', () => {
    const r = rng(1);
    const n = 10000;
    const pos = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      pos[3 * i] = 10 + (r() - 0.5) * 2; // x in [9, 11]
      pos[3 * i + 1] = -5 + (r() - 0.5) * 4; // y in [-7, -3]
      pos[3 * i + 2] = r(); // z in [0, 1]
    }
    // 1 % far outliers
    for (let i = 0; i < 100; i++) {
      pos[3 * i * 97] = 1e5 * (r() - 0.5);
      pos[3 * i * 97 + 2] = -1e4;
    }
    const b = robustBounds(pos)!;
    expect(b).not.toBeNull();
    expect(b.count).toBe(n);
    expect(b.min[0]).toBeGreaterThan(8.9);
    expect(b.max[0]).toBeLessThan(11.1);
    expect(b.min[1]).toBeGreaterThan(-7.1);
    expect(b.max[1]).toBeLessThan(-2.9);
    expect(b.center[0]).toBeCloseTo(10, 1);
    expect(b.center[1]).toBeCloseTo(-5, 1);
    expect(b.radius).toBeLessThan(3);
    expect(b.full.min[2]).toBeLessThanOrEqual(-1e4);
    const clamped = clampedFullBounds(b);
    expect(clamped.min[2]).toBeGreaterThan(-20);
    expect(clamped.max[0]).toBeLessThan(40);
  });

  it('handles empty, single and non-finite input; subsamples large input', () => {
    expect(robustBounds(new Float32Array(0))).toBeNull();
    const one = robustBounds(new Float32Array([1, 2, 3]))!;
    expect(Array.from(one.center)).toEqual([1, 2, 3]);
    expect(one.radius).toBeGreaterThan(0);
    const withNaN = robustBounds(new Float32Array([NaN, 0, 0, 1, 1, 1, 3, 3, 3]))!;
    expect(withNaN.full.min[0]).toBe(1);
    const big = new Float32Array(3 * 200000).map((_, i) => i % 3);
    const s = subsamplePositions(big, 200000, 1000);
    expect(s.length).toBe(3000);
    expect(s[0]).toBe(0);
    expect(s[1]).toBe(1);
    expect(s[2]).toBe(2);
  });

  it('percentileAlong gives the ground height', () => {
    const r = rng(2);
    const n = 5000;
    const pos = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      pos[3 * i] = r() * 10;
      pos[3 * i + 1] = r() * 10;
      pos[3 * i + 2] = i < n * 0.9 ? 1.5 + 0.01 * r() : 1.5 + 3 * r(); // ground at 1.5 + objects
    }
    const h = percentileAlong(pos, [0, 0, 1], 2);
    expect(h).toBeGreaterThan(1.49);
    expect(h).toBeLessThan(1.52);
    const s = boundsSphere({ min: new Float64Array([0, 0, 0]), max: new Float64Array([2, 2, 1]) });
    expect(s.radius).toBeCloseTo(1.5, 10);
  });
});

describe('symEig3', () => {
  it('diagonalises a symmetric matrix', () => {
    const A = [4, 1, 2, 1, 3, 0.5, 2, 0.5, 1];
    const { values, vectors } = symEig3(A);
    expect(values[0]).toBeLessThanOrEqual(values[1]);
    expect(values[1]).toBeLessThanOrEqual(values[2]);
    for (let k = 0; k < 3; k++) {
      const v = [vectors[k], vectors[3 + k], vectors[6 + k]];
      for (let r = 0; r < 3; r++) {
        const Av = A[3 * r] * v[0] + A[3 * r + 1] * v[1] + A[3 * r + 2] * v[2];
        expect(Av).toBeCloseTo(values[k] * v[r], 10);
      }
    }
    expect(values[0] + values[1] + values[2]).toBeCloseTo(8, 10);
  });
});

describe('estimateUpVector', () => {
  const tilts: [string, Float64Array][] = [
    ['identity', new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])],
    ['tilted world', axisAngle([0.3, -1, 0.5], 1.1)],
  ];
  for (const [name, W] of tilts) {
    // Reconstruction frame = W · gravity frame → true up = W e_z.
    const trueUp = [W[2], W[5], W[8]];
    const make = (yaw: number, pitch: number, roll: number): Pose => {
      const Rg = camRotation(yaw, pitch, roll); // world(gravity)->cam
      // x_cam = Rg X_g, X_g = Wᵀ X_w  ⇒ R = Rg Wᵀ
      const Wt = new Float64Array([W[0], W[3], W[6], W[1], W[4], W[7], W[2], W[5], W[8]]);
      return { R: mul3(Rg, Wt) as Mat3, t: new Float64Array(3) };
    };

    it(`partial orbit looking down (${name}) — unbiased by pitch`, () => {
      const r = rng(3);
      const poses: Pose[] = [];
      for (let i = 0; i < 40; i++) {
        const yaw = (i / 40) * (Math.PI / 2); // 90° arc only
        poses.push(make(yaw, -0.6 + 0.1 * (r() - 0.5), 0.04 * (r() - 0.5)));
      }
      const up = estimateUpVector(poses)!;
      expect(angleDeg(up, trueUp)).toBeLessThan(2);
      // The naive mean of camera up vectors is clearly biased in this setup.
      const mean = [0, 0, 0];
      for (const p of poses) {
        const u = cameraUp(p.R);
        mean[0] += u[0];
        mean[1] += u[1];
        mean[2] += u[2];
      }
      expect(angleDeg(mean, trueUp)).toBeGreaterThan(20);
    });

    it(`full orbit with a few rolled outliers (${name})`, () => {
      const r = rng(4);
      const poses: Pose[] = [];
      for (let i = 0; i < 60; i++) {
        const roll = i % 15 === 0 ? 0.6 : 0.03 * (r() - 0.5);
        poses.push(make(r() * 2 * Math.PI, -0.3 + 0.2 * r(), roll));
      }
      expect(angleDeg(estimateUpVector(poses)!, trueUp)).toBeLessThan(2);
    });

    it(`degenerate forward walk, no yaw change (${name})`, () => {
      const r = rng(5);
      const poses: Pose[] = [];
      // Pitch about the (common) right axis is unobservable here: symmetric pitches.
      for (let i = 0; i < 20; i++) poses.push(make(0.7, -0.4 + (0.8 * i) / 19, 0.02 * (r() - 0.5)));
      expect(angleDeg(estimateUpVector(poses)!, trueUp)).toBeLessThan(1);
    });
  }

  // Canonical images upright for θ₀, the phone held in portrait (raw iOS buffers: θ₀ = 90): image up
  // is not physical up; the camera's up axis upCam = R_cam_device(θ₀) · (0, 1, 0) (device +y) is.
  for (const theta0 of [90, 180, 270]) {
    it(`portrait half orbit of images upright for θ₀ = ${theta0} — up from upCam`, () => {
      const r = rng(6);
      const W = axisAngle([0.3, -1, 0.5], 1.1);
      const trueUp = [W[2], W[5], W[8]];
      const Wt = new Float64Array([W[0], W[3], W[6], W[1], W[4], W[7], W[2], W[5], W[8]]);
      // R = R_cam_device(θ₀) · R_wdᵀ, the device frame of an upright camera: R_wdᵀ = R_cam_device(0) · R
      const toCanonical = mul3(cameraFromDevice(theta0), cameraFromDevice(0));
      const upCam = rotate(cameraFromDevice(theta0), [0, 1, 0]);
      const poses: Pose[] = [];
      for (let i = 0; i < 12; i++) poses.push({ R: mul3(toCanonical, mul3(camRotation((i / 12) * Math.PI, -0.3 + 0.1 * r(), 0.03 * (r() - 0.5)), Wt)) as Mat3, t: new Float64Array(3) });
      expect(angleDeg(estimateUpVector(poses.map((p) => ({ R: p.R, upCam })))!, trueUp)).toBeLessThan(1);
      // camera −y (image up) is horizontal here
      expect(angleDeg(estimateUpVector(poses)!, trueUp)).toBeGreaterThan(45);
    });
  }

  it('upCam = camera −y gives exactly the result without upCam', () => {
    const r = rng(7);
    const poses: Pose[] = [];
    for (let i = 0; i < 30; i++) poses.push({ R: camRotation(r() * 2 * Math.PI, -0.3 + 0.2 * r(), i % 10 === 0 ? 0.5 : 0.03 * (r() - 0.5)) as Mat3, t: new Float64Array(3) });
    const up = estimateUpVector(poses)!;
    expect(angleDeg(up, [0, 0, 1])).toBeLessThan(2);
    expect(estimateUpVector(poses.map((p) => ({ R: p.R, upCam: [0, -1, 0] })))).toEqual(up);
  });

  it('single camera returns its up vector; empty returns null', () => {
    const R = camRotation(0.3, 0, 0);
    const up = estimateUpVector([{ R }])!;
    expect(angleDeg(up, [0, 0, 1])).toBeLessThan(1e-6);
    expect(estimateUpVector([])).toBeNull();
  });
});

describe('formatting & depth helpers', () => {
  it('niceStep picks 1/2/5 × 10^k', () => {
    expect(niceStep(1)).toBe(1);
    expect(niceStep(0.3)).toBeCloseTo(0.2, 12);
    expect(niceStep(4.2)).toBe(5);
    expect(niceStep(80)).toBe(100);
    expect(niceStep(0)).toBe(1);
  });

  it('formatDistance', () => {
    expect(formatDistance(1.2345, true)).toBe('1.23 m');
    expect(formatDistance(0.2345, true)).toBe('23.4 cm');
    expect(formatDistance(0.0234, true)).toBe('2.34 cm');
    expect(formatDistance(123.45, true)).toBe('123.5 m');
    expect(formatDistance(1.2345, false)).toBe('1.23 units');
    expect(formatDistance(0.012345, false)).toBe('0.0123 units');
  });

  it('unpackRGBAToDepth inverts packDepthToRGBA', () => {
    // Mirror of three's packDepthToRGBA (packing.glsl) in float64.
    const pack = (v: number): number[] => {
      if (v <= 0) return [0, 0, 0, 0];
      if (v >= 1) return [255, 255, 255, 255];
      let vuf = Math.floor(v * 16777216);
      const af = v * 16777216 - vuf;
      const bf = (vuf / 256) % 1;
      vuf = Math.floor(vuf / 256);
      const gf = (vuf / 256) % 1;
      vuf = Math.floor(vuf / 256);
      return [Math.round(vuf), Math.round(gf * 256), Math.round(bf * 256), Math.round(af * 255)];
    };
    for (const v of [0.123456789, 0.5, 0.99999, 0.000123]) {
      const [r, g, b, a] = pack(v);
      expect(Math.abs(unpackRGBAToDepth(r, g, b, a) - v)).toBeLessThan(1e-7);
    }
    expect(unpackRGBAToDepth(255, 255, 255, 255)).toBeCloseTo(1, 12);
  });

  it('linearizeDepth maps 0 → near, 1 → far', () => {
    expect(linearizeDepth(0, 0.1, 100)).toBeCloseTo(0.1, 10);
    expect(linearizeDepth(1, 0.1, 100)).toBeCloseTo(100, 8);
  });

  it('goldenStridePermutation yields distinct, evenly spread indices', () => {
    const n = 1000003;
    const k = 1000;
    const p = goldenStridePermutation(n, k);
    expect(new Set(p).size).toBe(k);
    const sorted = Array.from(p).sort((a, b) => a - b);
    let maxGap = 0;
    for (let i = 1; i < k; i++) maxGap = Math.max(maxGap, sorted[i] - sorted[i - 1]);
    expect(maxGap).toBeLessThan((3 * n) / k);
    const full = goldenStridePermutation(12, 12);
    expect(new Set(full).size).toBe(12);
  });
});

describe('surface depth fit', () => {
  it('recovers the depth of a plane from asymmetric samples (1/z affine)', () => {
    const a = 1 / 6.5;
    const b = -0.004;
    const c = 0.0025;
    const u: number[] = [];
    const v: number[] = [];
    const z: number[] = [];
    for (const [du, dv] of [[1, 2], [3, 1], [4, 4], [2, 5], [6, 0.5], [0.5, 3]]) {
      u.push(du);
      v.push(dv);
      z.push(1 / (a + b * du + c * dv));
    }
    expect(surfaceDepthAt(u, v, z, u.length)).toBeCloseTo(6.5, 9);
    const p = fitInverseDepthPlane(u, v, z, u.length)!;
    expect(p.b).toBeCloseTo(b, 9);
    expect(p.c).toBeCloseTo(c, 9);
  });

  it('handles collinear and tiny sample sets', () => {
    const a = 0.2;
    const u = [1, 2, 3, 4];
    const v = [2, 4, 6, 8];
    const z = u.map((x, i) => 1 / (a + 0.01 * x - 0.003 * v[i]));
    expect(surfaceDepthAt(u, v, z, 4)).toBeCloseTo(5, 6);
    expect(surfaceDepthAt([0.2], [0.1], [3], 1)).toBeCloseTo(3, 9);
    expect(Number.isNaN(surfaceDepthAt([], [], [], 0))).toBe(true);
  });
});

describe('pickNearestProjected', () => {
  // Simple column-major projection: camera at origin looking down -Z (three.js convention).
  function perspective(fovY: number, aspect: number, near: number, far: number): Float64Array {
    const f = 1 / Math.tan(fovY / 2);
    const m = new Float64Array(16);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) / (near - far);
    m[11] = -1;
    m[14] = (2 * far * near) / (near - far);
    return m;
  }

  it('returns the front-most point under the cursor', () => {
    const mvp = perspective(Math.PI / 3, 1, 0.1, 100);
    const pos = new Float32Array([
      0, 0, -5, // centre, far
      0, 0, -2, // centre, near (expected)
      1, 1, -2, // off-centre
    ]);
    expect(pickNearestProjected(pos, 3, mvp, 200, 200, 100, 100, 5)).toBe(1);
    expect(pickNearestProjected(pos, 3, mvp, 200, 200, 10, 10, 5)).toBe(-1);
  });

  it('estimates the surface depth on the tap ray for an oblique dense plane', () => {
    // Camera at origin looking down −Z; plane through (0, 0, −5) tilted 60° about X.
    const mvp = perspective(Math.PI / 3, 1, 0.1, 100);
    const pts: number[] = [];
    const tilt = (60 * Math.PI) / 180;
    for (let i = -100; i <= 100; i++) {
      for (let j = -100; j <= 100; j++) {
        const s = i * 0.01;
        const t = j * 0.01;
        pts.push(s, t * Math.cos(tilt), -5 - t * Math.sin(tilt));
      }
    }
    const pos = new Float32Array(pts);
    const out = { covered: false, w: 0 };
    // tap slightly off-centre; true depth from the ray/plane intersection
    const px = 205.3;
    const py = 188.7;
    const idx = pickNearestProjected(pos, pos.length / 3, mvp, 400, 400, px, py, 14, { sizePx: 4 }, out);
    expect(idx).toBeGreaterThanOrEqual(0);
    expect(out.covered).toBe(true);
    const f = 1 / Math.tan(Math.PI / 6);
    const rx = ((px / 400) * 2 - 1) / f;
    const ry = (1 - (py / 400) * 2) / f;
    // ray (rx, ry, −1)·w ; plane: y·sin + (z + 5)·cos = 0 ⇒ w (ry sin − cos) + 5 cos = 0
    const wTrue = (5 * Math.cos(tilt)) / (Math.cos(tilt) - ry * Math.sin(tilt));
    expect(Math.abs(out.w - wTrue) / wTrue).toBeLessThan(1e-3);
    expect(Math.abs(rx)).toBeLessThan(1);
  });

  it('is fast enough for 2 M points (fallback path)', () => {
    const n = 2_000_000;
    const r = rng(7);
    const pos = new Float32Array(3 * n);
    for (let i = 0; i < 3 * n; i += 3) {
      pos[i] = (r() - 0.5) * 10;
      pos[i + 1] = (r() - 0.5) * 10;
      pos[i + 2] = -5 - r() * 10;
    }
    const mvp = perspective(Math.PI / 3, 1, 0.1, 100);
    pickNearestProjected(pos, n, mvp, 400, 400, 200, 200, 10); // warm-up
    let best = Infinity;
    for (let k = 0; k < 3; k++) {
      const t0 = performance.now();
      pickNearestProjected(pos, n, mvp, 400, 400, 200 + k, 200, 10);
      best = Math.min(best, performance.now() - t0);
    }
    console.log(`[viewer] CPU pick over 2M points: ${best.toFixed(1)} ms`);
    expect(best).toBeLessThan(60);
  });
});

describe('camera frustum geometry', () => {
  it('corners lie on the image rays at the requested depth', () => {
    const R = camRotation(0.4, -0.2, 0.1);
    const C = [1, 2, 3];
    // t = -R C
    const t = new Float64Array([
      -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
      -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
      -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
    ]);
    const pose: Pose = { R, t };
    expect(Array.from(cameraCenter(pose)).map((v) => +v.toFixed(9))).toEqual(C);
    const cam = { width: 640, height: 480, f: 500, cx: 319.5, cy: 239.5, k1: 0, k2: 0 };
    const k = frustumCorners(pose, cam, 2);
    // back to camera frame: x_cam = R X + t
    const toCam = (i: number) => {
      const X = [k[3 * i], k[3 * i + 1], k[3 * i + 2]];
      return [0, 1, 2].map((r) => R[3 * r] * X[0] + R[3 * r + 1] * X[1] + R[3 * r + 2] * X[2] + t[r]);
    };
    const tl = toCam(0);
    const br = toCam(2);
    expect(tl[2]).toBeCloseTo(2, 9);
    expect((tl[0] / tl[2]) * cam.f + cam.cx).toBeCloseTo(-0.5, 6);
    expect((tl[1] / tl[2]) * cam.f + cam.cy).toBeCloseTo(-0.5, 6);
    expect((br[0] / br[2]) * cam.f + cam.cx).toBeCloseTo(639.5, 6);
    expect((br[1] / br[2]) * cam.f + cam.cy).toBeCloseTo(479.5, 6);
  });

  it('writes VERTICES_PER_CAMERA vertices per camera; highlight is larger', () => {
    const cam = { width: 4, height: 3, f: 4, cx: 1.5, cy: 1, k1: 0, k2: 0 };
    const pose: Pose = { R: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), t: new Float64Array(3) };
    const cams = [{ pose, intrinsics: cam }, { pose, intrinsics: cam }];
    const pos = new Float32Array(3 * VERTICES_PER_CAMERA * 2);
    const col = new Float32Array(3 * VERTICES_PER_CAMERA * 2);
    const n = writeFrustumLines(cams, 1, true, pos, col, [0, 0, 1], [1, 0, 0]);
    expect(n).toBe(2 * VERTICES_PER_CAMERA);
    expect(pos[3 * 1 + 2]).toBeCloseTo(1, 6); // first ray end at depth 1
    expect(pos[3 * (VERTICES_PER_CAMERA + 1) + 2]).toBeCloseTo(1.6, 6); // highlighted
    expect(col[3 * VERTICES_PER_CAMERA]).toBe(1);
    expect(col[2]).toBe(1);
    // "up" marker apex is above the top edge (camera −y, world −y for identity pose)
    const apexY = pos[3 * 17 + 1];
    expect(apexY).toBeLessThan((-1 - 0.5) / 4);
  });
});
