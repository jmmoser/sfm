/**
 * Adversarial review unit tests (Node) for viewer helpers: CPU ray–mesh
 * intersection, confidence-aware CPU picking and frustum orientation in a Z-up
 * world.
 */
import { describe, expect, it } from 'vitest';
import type { Pose } from '../core/types';
import { frustumCorners, writeFrustumLines, VERTICES_PER_CAMERA } from './cameras';
import { cameraCenter, pickNearestProjected, raycastTriangles } from './helpers';

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

/** World-to-camera pose looking from C at target in a Z-up world (x right & level, y down). */
function lookAt(C: number[], target: number[]): Pose {
  const f = [target[0] - C[0], target[1] - C[1], target[2] - C[2]];
  const fn = Math.hypot(f[0], f[1], f[2]);
  const z = f.map((v) => v / fn);
  let x = [z[1], -z[0], 0];
  const xn = Math.hypot(x[0], x[1]) || 1;
  x = x.map((v) => v / xn);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const R = new Float64Array([...x, ...y, ...z]);
  const t = new Float64Array([0, 1, 2].map((r) => -(R[3 * r] * C[0] + R[3 * r + 1] * C[1] + R[3 * r + 2] * C[2])));
  return { R, t };
}

describe('raycastTriangles', () => {
  it('finds the nearest hit of random rays against random triangles (both faces)', () => {
    const r = rng(3);
    for (let trial = 0; trial < 200; trial++) {
      const A = [r(), r(), r()].map((v) => v * 4 - 2);
      const B = [r(), r(), r()].map((v) => v * 4 - 2);
      const Cc = [r(), r(), r()].map((v) => v * 4 - 2);
      const pos = new Float32Array([...A, ...B, ...Cc]);
      // point inside the triangle (barycentric) and a ray through it from outside
      let u = r();
      let v = r();
      if (u + v > 1) {
        u = 1 - u;
        v = 1 - v;
      }
      const P = [0, 1, 2].map((k) => pos[k] + u * (pos[3 + k] - pos[k]) + v * (pos[6 + k] - pos[k]));
      const O = [r(), r(), r()].map((x) => x * 20 - 10);
      const d = [P[0] - O[0], P[1] - O[1], P[2] - O[2]];
      const t = raycastTriangles(pos, [0, 1, 2], 1, O[0], O[1], O[2], d[0], d[1], d[2]);
      expect(t).toBeCloseTo(1, 5);
      // reversed ray from P outwards: no hit beyond tMin
      const t2 = raycastTriangles(pos, [0, 1, 2], 1, P[0] - 1e-3 * d[0], P[1] - 1e-3 * d[1], P[2] - 1e-3 * d[2], -d[0], -d[1], -d[2]);
      expect(t2).toBe(Infinity);
    }
  });

  it('returns the closest of several layers, skips bad indices and misses outside', () => {
    // two unit squares at z = 0 and z = 1 (2 triangles each), ray down from z = 5
    const pos = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]);
    const idx = new Uint32Array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 99, 0, 1]);
    expect(raycastTriangles(pos, idx, 5, 0.3, 0.6, 5, 0, 0, -1)).toBeCloseTo(4, 9);
    expect(raycastTriangles(pos, idx, 2, 0.3, 0.6, 5, 0, 0, -1)).toBeCloseTo(5, 9); // only the lower square
    expect(raycastTriangles(pos, idx, 5, 1.3, 0.6, 5, 0, 0, -1)).toBe(Infinity);
    expect(raycastTriangles(pos, idx, 5, 0.3, 0.6, 5, 1, 0, 0)).toBe(Infinity); // parallel
  });

  it('is fast enough for 1 M triangles (CPU fallback)', () => {
    const k = 708; // ≈ 1 M triangles
    const nv = (k + 1) * (k + 1);
    const pos = new Float32Array(3 * nv);
    for (let j = 0, i = 0; j <= k; j++) for (let q = 0; q <= k; q++, i++) pos.set([q / k, j / k, 0.01 * Math.sin(q)], 3 * i);
    const idx = new Uint32Array(6 * k * k);
    for (let j = 0, t = 0; j < k; j++) {
      for (let q = 0; q < k; q++, t += 6) {
        const a = j * (k + 1) + q;
        idx.set([a, a + 1, a + k + 2, a, a + k + 2, a + k + 1], t);
      }
    }
    raycastTriangles(pos, idx, 2 * k * k, 0.5, 0.5, 2, 0.01, 0, -1);
    let best = Infinity;
    for (let n = 0; n < 3; n++) {
      const t0 = performance.now();
      const t = raycastTriangles(pos, idx, 2 * k * k, 0.37, 0.61, 2, 0.01, 0, -1);
      best = Math.min(best, performance.now() - t0);
      expect(t).toBeGreaterThan(1.9);
      expect(t).toBeLessThan(2.1);
    }
    console.log(`[viewer] CPU raycast over ${2 * k * k} triangles: ${best.toFixed(1)} ms`);
    expect(best).toBeLessThan(150);
  });
});

describe('pickNearestProjected confidence filter', () => {
  it('skips points below minConfidence exactly like the shader (conf < min hidden)', () => {
    const mvp = perspective(Math.PI / 3, 1, 0.1, 100);
    const pos = new Float32Array([0, 0, -2, 0, 0, -5]); // near (low conf) in front of far
    const confidence = new Float32Array([0.1, 0.5]);
    const out = { covered: false, w: 0 };
    expect(pickNearestProjected(pos, 2, mvp, 200, 200, 100, 100, 5, { sizePx: 4 }, out)).toBe(0);
    expect(pickNearestProjected(pos, 2, mvp, 200, 200, 100, 100, 5, { sizePx: 4, confidence, minConfidence: 0.5 }, out)).toBe(1);
    expect(out.w).toBeCloseTo(5, 6);
    // minConfidence 0 disables the filter
    expect(pickNearestProjected(pos, 2, mvp, 200, 200, 100, 100, 5, { sizePx: 4, confidence, minConfidence: 0 })).toBe(0);
    // everything filtered → miss
    expect(pickNearestProjected(pos, 2, mvp, 200, 200, 100, 100, 5, { sizePx: 4, confidence, minConfidence: 0.9 })).toBe(-1);
  });
});

describe('frustum orientation in a Z-up world', () => {
  it('rays go along the optical axis (row 2 of R); the up marker points to world +Z for a level camera', () => {
    const C = [2, -1, 1.5];
    const pose = lookAt(C, [6, 3, 1.5]); // level, looking along (1, 1, 0)/√2
    expect(Array.from(cameraCenter(pose)).map((v) => +v.toFixed(9))).toEqual(C);
    const cam = { width: 400, height: 300, f: 350, cx: 199.5, cy: 149.5, k1: 0, k2: 0 };
    const k = frustumCorners(pose, cam, 1);
    // rectangle centre ≈ C + forward (principal point at the image centre)
    const cx = (k[0] + k[3] + k[6] + k[9]) / 4;
    const cy = (k[1] + k[4] + k[7] + k[10]) / 4;
    const cz = (k[2] + k[5] + k[8] + k[11]) / 4;
    expect(cx - C[0]).toBeCloseTo(Math.SQRT1_2, 9);
    expect(cy - C[1]).toBeCloseTo(Math.SQRT1_2, 9);
    expect(cz - C[2]).toBeCloseTo(0, 9);
    // top edge (image −y) is higher in the world than the bottom edge
    expect(k[2]).toBeGreaterThan(k[11]);
    // top-right corner is to the right of the camera: (forward × up) side
    const right = [Math.SQRT1_2, -Math.SQRT1_2, 0];
    expect((k[3] - C[0]) * right[0] + (k[4] - C[1]) * right[1]).toBeGreaterThan(0);
    const pos = new Float32Array(3 * VERTICES_PER_CAMERA);
    const col = new Float32Array(3 * VERTICES_PER_CAMERA);
    writeFrustumLines([{ pose, intrinsics: cam }], 1, false, pos, col, [0, 0, 1], [1, 0, 0]);
    const apexZ = pos[3 * 17 + 2];
    expect(apexZ).toBeGreaterThan(k[2] + 0.1);
    // every ray starts at the camera centre
    for (let j = 0; j < 4; j++) expect(Math.hypot(pos[6 * j] - C[0], pos[6 * j + 1] - C[1], pos[6 * j + 2] - C[2])).toBeLessThan(1e-6);
  });
});
