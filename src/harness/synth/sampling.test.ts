import { Mesh, MeshBasicMaterial, PlaneGeometry } from 'three';
import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics } from '../../core/types';
import { pinholeWindow, poseFromDevice } from './conventions';
import { SynthRng } from './rng';
import { accumulateVisibility, collectTriangles, quantizeNormals, sampleSurfaces, sampleTriangles } from './sampling';
import { deviceAttitudeFromCamera } from './trajectory';

describe('area-uniform surface sampling', () => {
  it('distributes samples proportionally to triangle area, on the triangles', () => {
    // two triangles in z=0 (area 0.5) and x=5 (area 1.5)
    const tris = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 5, 0, 0, 5, 3, 0, 5, 0, 1]);
    const s = sampleTriangles(tris, 200_000, new SynthRng(1));
    expect(s.totalArea).toBeCloseTo(2, 6);
    let first = 0;
    for (let i = 0; i < s.count; i++) {
      const x = s.positions[3 * i],
        y = s.positions[3 * i + 1],
        z = s.positions[3 * i + 2];
      if (x < 2) {
        first++;
        expect(Math.abs(z)).toBeLessThan(1e-6);
        expect(x + y).toBeLessThanOrEqual(1 + 1e-6);
        expect(Math.abs(s.normals[3 * i + 2])).toBeCloseTo(1, 6);
      } else expect(x).toBeCloseTo(5, 5);
    }
    expect(first / s.count).toBeGreaterThan(0.245);
    expect(first / s.count).toBeLessThan(0.255);
    // uniform inside a triangle: mean of the first triangle's samples = centroid (1/3, 1/3)
    let mx = 0,
      my = 0;
    for (let i = 0; i < s.count; i++)
      if (s.positions[3 * i] < 2) {
        mx += s.positions[3 * i];
        my += s.positions[3 * i + 1];
      }
    expect(mx / first).toBeCloseTo(1 / 3, 2);
    expect(my / first).toBeCloseTo(1 / 3, 2);
  });

  it('uses world matrices of meshes (scaled plane → scaled area)', () => {
    const m = new Mesh(new PlaneGeometry(2, 1, 3, 2), new MeshBasicMaterial());
    m.scale.set(2, 3, 1);
    m.position.set(10, 0, 1);
    m.updateMatrixWorld(true);
    const tris = collectTriangles([m]);
    expect(tris.length).toBe(9 * 12);
    const s = sampleSurfaces([m], 1000, new SynthRng(2));
    expect(s.totalArea).toBeCloseTo(12, 5);
    for (let i = 0; i < s.count; i++) {
      expect(s.positions[3 * i + 2]).toBeCloseTo(1, 6);
      expect(Math.abs(s.positions[3 * i] - 10)).toBeLessThanOrEqual(2 + 1e-6);
    }
    const q = quantizeNormals(s.normals);
    expect(q[2]).toBe(127);
  });
});

describe('visibility from a depth map', () => {
  it('marks unoccluded in-frustum samples only', () => {
    const cam: CameraIntrinsics = { width: 64, height: 48, f: 50, cx: 31.5, cy: 23.5, k1: -0.02, k2: 0 };
    // level camera at the origin looking north (+Y): right = +X, down = −Z; landscape θ = 90
    const R_wc = [1, 0, 0, 0, 0, 1, 0, -1, 0]; // columns: right, down, forward
    const R_wd = deviceAttitudeFromCamera(R_wc, 90);
    const pose = poseFromDevice(R_wd, [0, 0, 0], 90);
    // depth map: a wall at y = 5 (z-depth 5 everywhere)
    const win = pinholeWindow({ ...cam, k1: 0 });
    const grow = 0.1;
    const window = { xmin: win.xmin - grow, xmax: win.xmax + grow, ymin: win.ymin - grow, ymax: win.ymax + grow };
    const W = 40,
      H = 30;
    const depth = { data: new Float32Array(W * H).fill(5), width: W, height: H, window };
    const pts = new Float32Array([
      0, 5, 0, // on the wall, centre → visible
      1, 5, 0.5, // on the wall → visible
      0, 8, 0, // behind the wall → occluded
      0, -5, 0, // behind the camera
      50, 5, 0, // outside the image
    ]);
    const counts = new Uint8Array(5);
    const n = accumulateVisibility(pts, counts, pose, cam, depth);
    expect(n).toBe(2);
    expect(Array.from(counts)).toEqual([1, 1, 0, 0, 0]);
    counts.fill(255);
    accumulateVisibility(pts, counts, pose, cam, depth);
    expect(counts[0]).toBe(255); // saturates
  });
});
