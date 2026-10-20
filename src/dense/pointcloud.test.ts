import { describe, expect, it } from 'vitest';
import type { PointCloud } from '../core/types';
import { Rng } from '../math/random';
import { KdTree3 } from './kdtree';
import { filterPointCloud, footprintDownsample, statisticalOutlierMask, voxelDownsample } from './pointcloud';

const log = (...a: unknown[]): void => console.log('[pointcloud]', ...a);

function cloud(pts: number[][], radii?: number[]): PointCloud {
  const n = pts.length;
  const pc: PointCloud = {
    count: n,
    positions: Float32Array.from(pts.flat()),
    colors: new Uint8Array(3 * n).fill(100),
    normals: new Float32Array(3 * n).map((_, k) => (k % 3 === 2 ? 1 : 0)),
  };
  if (radii) pc.radii = Float32Array.from(radii);
  return pc;
}

describe('KdTree3', () => {
  it('exact kNN against brute force (clustered data with duplicates)', () => {
    const rng = new Rng(1);
    const pts: number[] = [];
    for (let i = 0; i < 3000; i++) {
      const c = i % 3;
      const s = c === 0 ? 0.001 : c === 1 ? 0.1 : 10;
      pts.push(c * 5 + s * rng.gaussian(), s * rng.gaussian(), s * rng.gaussian());
    }
    for (let i = 0; i < 50; i++) pts.push(pts[0], pts[1], pts[2]); // duplicates
    const P = Float32Array.from(pts);
    const n = P.length / 3;
    const tree = new KdTree3(P);
    const k = 10;
    const d = new Float64Array(k);
    const ix = new Int32Array(k);
    for (let q = 0; q < n; q += 37) {
      const got = tree.knn(P[3 * q], P[3 * q + 1], P[3 * q + 2], k, d, ix, q);
      expect(got).toBe(k);
      const brute: number[] = [];
      for (let j = 0; j < n; j++) {
        if (j === q) continue;
        brute.push((P[3 * j] - P[3 * q]) ** 2 + (P[3 * j + 1] - P[3 * q + 1]) ** 2 + (P[3 * j + 2] - P[3 * q + 2]) ** 2);
      }
      brute.sort((a, b) => a - b);
      const mine = Array.from(d).sort((a, b) => a - b);
      for (let t = 0; t < k; t++) expect(mine[t]).toBeCloseTo(brute[t], 12);
    }
  });
});

describe('filterPointCloud', () => {
  it('voxel downsampling averages attributes', () => {
    const pts: number[][] = [];
    const cols: number[] = [];
    for (let i = 0; i < 100; i++) {
      for (let j = 0; j < 100; j++) {
        pts.push([i * 0.01 + 0.001, j * 0.01 + 0.001, 0]);
        cols.push(i * 2, j * 2, 50);
      }
    }
    const pc = cloud(pts, new Array(pts.length).fill(0.004));
    pc.colors = Uint8Array.from(cols);
    pc.confidence = new Float32Array(pc.count).fill(0.5);
    const out = filterPointCloud(pc, { voxelSize: 0.05 });
    expect(out.count).toBe(400); // 20 × 20 cells of 5 × 5 points
    for (let c = 0; c < out.count; c++) {
      const x = out.positions[3 * c];
      const ci = Math.round((x - 0.001) / 0.01); // mean column index of the 5 points
      expect(out.colors[3 * c]).toBe(ci * 2);
      expect(out.normals![3 * c + 2]).toBeCloseTo(1, 6);
      expect(out.radii![c]).toBeCloseTo(0.03, 6); // max(mean 0.004, 0.6 · voxel)
      expect(out.confidence![c]).toBeCloseTo(0.5, 6);
    }
  });

  it('statistical outlier removal removes injected outliers', () => {
    const rng = new Rng(2);
    const pts: number[][] = [];
    // sphere r = 1 and ground plane, ~uniform spacing ≈ 1.5 cm, small noise
    for (let i = 0; i < 25000; i++) {
      const u = 2 * rng.next() - 1;
      const a = 2 * Math.PI * rng.next();
      const s = Math.sqrt(1 - u * u);
      pts.push([s * Math.cos(a) + 0.002 * rng.gaussian(), s * Math.sin(a), u + 1.5]);
    }
    for (let i = 0; i < 25000; i++) pts.push([-2 + 4 * rng.next(), -2 + 4 * rng.next(), 0.002 * rng.gaussian()]);
    const nIn = pts.length;
    const nOut = 1000;
    for (let i = 0; i < nOut; i++) pts.push([-2 + 4 * rng.next(), -2 + 4 * rng.next(), 0.1 + 3 * rng.next()]);
    const pc = cloud(pts);
    const t0 = performance.now();
    const { keep, threshold } = statisticalOutlierMask(pc, { k: 8, std: 2 });
    const dt = performance.now() - t0;
    let inRemoved = 0;
    let outRemoved = 0;
    for (let i = 0; i < nIn; i++) if (!keep[i]) inRemoved++;
    for (let i = nIn; i < pts.length; i++) if (!keep[i]) outRemoved++;
    log(`SOR k=8 std=2 on ${pts.length} pts: ${dt.toFixed(0)} ms, threshold ${threshold.toFixed(4)}, outliers removed ${outRemoved}/${nOut}, inliers removed ${inRemoved}/${nIn}`);
    expect(outRemoved / nOut).toBeGreaterThan(0.9);
    expect(inRemoved / nIn).toBeLessThan(0.02);
    const filtered = filterPointCloud(pc, { statistical: { k: 8, std: 2 } });
    expect(filtered.count).toBe(pts.length - inRemoved - outRemoved);
  });

  it('scale-adaptive SOR keeps sparse far surfaces of a multi-scale cloud', () => {
    const rng = new Rng(3);
    const pts: number[][] = [];
    const radii: number[] = [];
    // close-range object: 1 mm spacing (0.1 m patch at 0.5 m), radius 0.7 mm
    for (let i = 0; i < 100; i++) for (let j = 0; j < 100; j++) {
      pts.push([i * 0.001, j * 0.001, 0.5]);
      radii.push(0.0007);
    }
    // far wall: 5 cm spacing (2.5 m × 2.5 m at 30 m), radius 3.5 cm
    for (let i = 0; i < 50; i++) for (let j = 0; j < 50; j++) {
      pts.push([-1 + i * 0.05, -1 + j * 0.05, 30]);
      radii.push(0.035);
    }
    const nNear = 10000;
    const nIn = pts.length;
    // outliers near the object (floating 1–3 cm off it, footprint-scale radii)
    for (let i = 0; i < 100; i++) {
      pts.push([0.1 * rng.next(), 0.1 * rng.next(), 0.5 - 0.01 - 0.02 * rng.next()]);
      radii.push(0.0007);
    }
    const pc = cloud(pts, radii);
    const ad = statisticalOutlierMask(pc, { k: 8, std: 2 });
    const plain = statisticalOutlierMask(pc, { k: 8, std: 2, scaleAdaptive: false });
    const count = (keep: Uint8Array, a: number, b: number): number => {
      let c = 0;
      for (let i = a; i < b; i++) if (!keep[i]) c++;
      return c;
    };
    log(
      `multi-scale: adaptive removes far ${count(ad.keep, nNear, nIn)}/2500, near ${count(ad.keep, 0, nNear)}, outliers ${count(ad.keep, nIn, pts.length)}/100; ` +
        `plain removes far ${count(plain.keep, nNear, nIn)}/2500, outliers ${count(plain.keep, nIn, pts.length)}/100`,
    );
    expect(count(ad.keep, nNear, nIn)).toBeLessThan(0.02 * 2500);
    expect(count(ad.keep, 0, nNear)).toBeLessThan(0.02 * nNear);
    expect(count(ad.keep, nIn, pts.length)).toBeGreaterThan(90);
    // why the adaptive mode exists: plain SOR removes far-wall points and keeps the outliers
    expect(count(plain.keep, nNear, nIn)).toBeGreaterThan(0.05 * 2500);
    expect(count(plain.keep, nIn, pts.length)).toBeLessThan(50);
  });

  it('footprint LOD keeps a close-range object at its own resolution under the point cap', () => {
    const rng = new Rng(5);
    const pts: number[][] = [];
    const radii: number[] = [];
    // close-range object: sphere r = 5 cm at 0.5 m, ~1 mm spacing (radius 0.7 mm)
    const nObj = 30_000;
    for (let i = 0; i < nObj; i++) {
      const u = 2 * rng.next() - 1;
      const a = 2 * Math.PI * rng.next();
      const s = Math.sqrt(1 - u * u);
      pts.push([0.05 * s * Math.cos(a), 0.05 * s * Math.sin(a), 0.5 + 0.05 * u]);
      radii.push(0.0007);
    }
    // background: wall at 3 m, 6 mm spacing (radius 4 mm), 6 m × 0.6 m
    for (let i = 0; i < 1000; i++) for (let j = 0; j < 100; j++) {
      pts.push([-3 + 0.006 * i, -0.3 + 0.006 * j, 3]);
      radii.push(0.004);
    }
    const pc = cloud(pts, radii);
    const cap = 60_000;
    const t0 = performance.now();
    const lod = footprintDownsample(pc, cap);
    const ms = performance.now() - t0;
    // the old rule: one voxel from the median radius (the wall's) for the same cap
    const uni = voxelDownsample(pc, 2 * 0.004 * Math.sqrt(pc.count / cap));
    const near = (c: PointCloud): number => {
      let k = 0;
      for (let i = 0; i < c.count; i++) if (c.positions[3 * i + 2] < 1) k++;
      return k;
    };
    // distance of the merged object points from the sphere
    const sphereErr = (c: PointCloud): number => {
      const e: number[] = [];
      for (let i = 0; i < c.count; i++) {
        const z = c.positions[3 * i + 2];
        if (z >= 1) continue;
        e.push(Math.abs(Math.hypot(c.positions[3 * i], c.positions[3 * i + 1], z - 0.5) - 0.05));
      }
      e.sort((a, b) => a - b);
      return e[Math.floor(0.9 * e.length)];
    };
    log(
      `LOD ${pc.count} → ${lod.count} (cap ${cap}, ${ms.toFixed(0)} ms): object ${near(lod)}/${nObj} points, p90 off-surface ${(1000 * sphereErr(lod)).toFixed(2)} mm; ` +
        `uniform voxel → ${uni.count}: object ${near(uni)}, p90 ${(1000 * sphereErr(uni)).toFixed(2)} mm`,
    );
    expect(lod.count).toBeLessThanOrEqual(cap);
    expect(lod.count).toBeGreaterThan(0.85 * cap);
    // both surfaces lose a similar share; the object keeps far more points than with one voxel
    expect(near(lod)).toBeGreaterThan(0.3 * nObj);
    expect(near(lod)).toBeGreaterThan(5 * near(uni));
    expect(sphereErr(lod)).toBeLessThan(0.0005);
    expect(lod.radii!.every((r) => r >= 0.0007)).toBe(true);
    // under the cap: unchanged
    expect(footprintDownsample(pc, pc.count)).toBe(pc);
    // no radii: uniform searched voxel, still under the cap
    const noR = cloud(pts);
    const u2 = footprintDownsample(noR, cap);
    expect(u2.count).toBeLessThanOrEqual(cap);
    expect(u2.count).toBeGreaterThan(0.5 * cap);
  });

  it('timing: 1M points voxel + SOR', () => {
    const rng = new Rng(4);
    const n = 1_000_000;
    const P = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      P[3 * i] = 10 * rng.next();
      P[3 * i + 1] = 10 * rng.next();
      P[3 * i + 2] = 0.01 * rng.gaussian();
    }
    const pc: PointCloud = { count: n, positions: P, colors: new Uint8Array(3 * n) };
    const t0 = performance.now();
    const v = voxelDownsample(pc, 0.02);
    const t1 = performance.now();
    const f = filterPointCloud(pc, { statistical: { k: 8, std: 2 } });
    const t2 = performance.now();
    log(`1M pts: voxel 2 cm → ${v.count} in ${(t1 - t0).toFixed(0)} ms; SOR k=8 → ${f.count} in ${(t2 - t1).toFixed(0)} ms`);
    expect(f.count).toBeGreaterThan(0.9 * n);
  });
});
