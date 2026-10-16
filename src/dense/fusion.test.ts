import { describe, expect, it } from 'vitest';
import type { PointCloud } from '../core/types';
import { cameraCenter, makeNearScene, makeFarScene, orbitPoses, pinholeIntrinsics, lookAt, type SynthScene } from './synthetic';
import { distanceToScene, renderGtViews } from './testutil-scene';
import { fuseDepthMaps, type FusionStats } from './fusion';
import { MemoryDepthMapStore, type DenseView } from './types';
import { neighboursFromSelection, selectSourceViews } from './view-selection';
import { Rng } from '../math/random';

const log = (...a: unknown[]): void => console.log('[fusion]', ...a);

/** Relative error of each point: distance to the true surface / distance to the nearest camera. */
function relErrors(pc: PointCloud, scene: SynthScene, views: DenseView[]): Float64Array {
  const centres = views.map((v) => cameraCenter(v.pose));
  const out = new Float64Array(pc.count);
  for (let i = 0; i < pc.count; i++) {
    const x = pc.positions[3 * i];
    const y = pc.positions[3 * i + 1];
    const z = pc.positions[3 * i + 2];
    let dc = Infinity;
    for (const c of centres) dc = Math.min(dc, Math.hypot(x - c[0], y - c[1], z - c[2]));
    out[i] = distanceToScene(scene, x, y, z) / dc;
  }
  return out;
}

function ringNeighbours(n: number, k: number): Map<number, number[]> {
  const m = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const l: number[] = [];
    for (let d = 1; d <= k; d++) l.push((i + d) % n, (i - d + n) % n);
    m.set(i, l);
  }
  return m;
}

function frac(a: Float64Array, pred: (v: number) => boolean): number {
  let c = 0;
  for (const v of a) if (pred(v)) c++;
  return a.length ? c / a.length : 0;
}

describe('fuseDepthMaps', () => {
  it('noisy GT depth maps with 10 % outliers → points within 0.5 % of the true surfaces, outliers removed', async () => {
    const scene = makeNearScene();
    const cam = pinholeIntrinsics(160, 120, (60 * Math.PI) / 180);
    const poses = orbitPoses([0, 0, 0.25], 3, (35 * Math.PI) / 180, 16);
    const gt = renderGtViews(scene, cam, poses, { noise: 0.002, normalNoiseDeg: 3, outliers: 0.1, seed: 5 });
    let stats: FusionStats | null = null;
    const filtered = new MemoryDepthMapStore();
    const progress: number[] = [];
    const pc = await fuseDepthMaps(gt.views, gt.store, ringNeighbours(16, 2), { onStats: (s) => (stats = s), filteredStore: filtered }, (e) => progress.push(e.progress));
    const s = stats!;
    const err = relErrors(pc, scene, gt.views);
    const sorted = Float64Array.from(err).sort();
    let totalValid = 0;
    for (const d of gt.gtDepth) for (const z of d) if (z > 0) totalValid++;
    log(
      `16 views 160×120, σ 0.2 %, 10 % outliers: ${pc.count} points from ${totalValid} GT pixels, mean views ${s.meanViews.toFixed(2)}, ` +
        `${s.timeMs.toFixed(0)} ms; rel. error median ${(100 * sorted[sorted.length >> 1]).toFixed(3)} %, p99 ${(100 * sorted[Math.floor(0.99 * sorted.length)]).toFixed(3)} %, ` +
        `max ${(100 * sorted[sorted.length - 1]).toFixed(2)} %, within 0.5 %: ${(100 * frac(err, (v) => v < 0.005)).toFixed(2)} %, > 2 %: ${frac(err, (v) => v > 0.02) * pc.count} pts`,
    );
    expect(pc.count).toBeGreaterThan(0.15 * totalValid);
    expect(frac(err, (v) => v < 0.005)).toBeGreaterThan(0.99);
    expect(frac(err, (v) => v > 0.02)).toBeLessThan(0.001);
    expect(s.meanViews).toBeGreaterThanOrEqual(3);
    // attributes
    expect(pc.normals!.length).toBe(3 * pc.count);
    expect(pc.radii!.length).toBe(pc.count);
    expect(pc.confidence!.length).toBe(pc.count);
    for (let i = 0; i < pc.count; i += 97) {
      expect(Math.abs(Math.hypot(pc.normals![3 * i], pc.normals![3 * i + 1], pc.normals![3 * i + 2]) - 1)).toBeLessThan(1e-5);
      expect(pc.confidence![i]).toBeGreaterThan(0);
      expect(pc.confidence![i]).toBeLessThanOrEqual(1);
      expect(pc.radii![i]).toBeGreaterThan(0);
    }
    // colours are image colours (scene albedo is never pure black)
    let csum = 0;
    for (let i = 0; i < pc.colors.length; i++) csum += pc.colors[i];
    expect(csum / pc.colors.length).toBeGreaterThan(30);
    expect(progress.at(-1)).toBe(1);

    // the filtered depth maps contain only consistent pixels
    let kept = 0;
    let bad = 0;
    for (let v = 0; v < 16; v++) {
      const dm = await filtered.get(v);
      expect(dm).not.toBeNull();
      for (let k = 0; k < dm!.depth.length; k++) {
        if (!(dm!.depth[k] > 0)) continue;
        kept++;
        if (Math.abs(dm!.depth[k] - gt.gtDepth[v][k]) > 0.02 * gt.gtDepth[v][k]) bad++;
      }
    }
    log(`filtered depth maps: ${kept} pixels kept (${((100 * kept) / totalValid).toFixed(1)} % of GT pixels), ${bad} with > 2 % error`);
    expect(bad / kept).toBeLessThan(0.002);
  });

  it('duplicates are avoided: each surface patch is fused once', async () => {
    // fronto-parallel textured wall seen by 5 nearly identical cameras
    const scene: SynthScene = { primitives: [{ kind: 'quad', center: [0, 5, 0], axisU: [1, 0, 0], axisV: [0, 0, 1], halfU: 20, halfV: 20, material: { scale: 1 } }] };
    const cam = pinholeIntrinsics(80, 60, 1);
    const poses = [-0.2, -0.1, 0, 0.1, 0.2].map((dx) => lookAt([dx, 0, 0], [dx, 1, 0]));
    const gt = renderGtViews(scene, cam, poses);
    const nb = new Map<number, number[]>();
    for (let i = 0; i < 5; i++) nb.set(i, [0, 1, 2, 3, 4].filter((j) => j !== i));
    const pc = await fuseDepthMaps(gt.views, gt.store, nb, { minViews: 3 });
    // each wall pixel footprint holds ~1 point: count ≈ union area / pixel area
    // (5 views, shifted by ≤ 0.4 m on a 4.6 m wide field of view): ≤ ~1.2 × one view
    log(`wall: ${pc.count} points for 5 × ${80 * 60} pixels`);
    expect(pc.count).toBeLessThan(1.2 * 80 * 60);
    expect(pc.count).toBeGreaterThan(0.7 * 80 * 60);
  });

  it('end-to-end with selectSourceViews (sparse points from GT) on the far (yard) scene', async () => {
    const scene = makeFarScene();
    const cam = pinholeIntrinsics(192, 144, (65 * Math.PI) / 180);
    // walk along the yard looking at the houses
    const poses = [];
    for (let i = 0; i < 12; i++) poses.push(lookAt([-6 + i * 1.1, 0, 1.6 + 0.1 * (i % 2)], [-3 + i * 0.6, 30, 2]));
    const gt = renderGtViews(scene, cam, poses, { noise: 0.002, normalNoiseDeg: 3, outliers: 0.1, seed: 9 });
    // sparse points: GT surface samples, visible where they agree with each view's GT depth
    const rng = new Rng(1);
    const xyz: number[] = [];
    const vis: number[][] = [];
    for (let v = 0; v < poses.length; v++) {
      const R = poses[v].R;
      const C = cameraCenter(poses[v]);
      for (let s = 0; s < 150; s++) {
        const k = rng.int(cam.width * cam.height);
        const z = gt.gtDepth[v][k];
        if (!(z > 0)) continue;
        const x = k % cam.width;
        const y = (k - x) / cam.width;
        const rx = ((x - cam.cx) / cam.f) * z;
        const ry = ((y - cam.cy) / cam.f) * z;
        const X = [0, 1, 2].map((a) => C[a] + R[a] * rx + R[3 + a] * ry + R[6 + a] * z);
        const seen: number[] = [];
        for (let w = 0; w < poses.length; w++) {
          const Rw = poses[w].R;
          const tw = poses[w].t;
          const zc = Rw[6] * X[0] + Rw[7] * X[1] + Rw[8] * X[2] + tw[2];
          if (!(zc > 0)) continue;
          const u = Math.round((cam.f * (Rw[0] * X[0] + Rw[1] * X[1] + Rw[2] * X[2] + tw[0])) / zc + cam.cx);
          const vv = Math.round((cam.f * (Rw[3] * X[0] + Rw[4] * X[1] + Rw[5] * X[2] + tw[1])) / zc + cam.cy);
          if (u < 0 || vv < 0 || u >= cam.width || vv >= cam.height) continue;
          const zg = gt.gtDepth[w][vv * cam.width + u];
          if (Math.abs(zg - zc) < 0.01 * zc) seen.push(w);
        }
        if (seen.length >= 2) {
          xyz.push(...X);
          vis.push(seen);
        }
      }
    }
    const sel = selectSourceViews(gt.views, { xyz: Float64Array.from(xyz), visibility: vis }, { numSources: 6 });
    for (const [id, s] of sel) {
      expect(s.sources.length).toBeGreaterThanOrEqual(2);
      expect(s.sources).not.toContain(id);
      // GT depth range of the view is inside the selected range (with the margins)
      let lo = Infinity;
      let hi = 0;
      const d = gt.gtDepth[id];
      const sorted = Float64Array.from(d.filter((z) => z > 0)).sort();
      lo = sorted[Math.floor(0.02 * sorted.length)];
      hi = sorted[Math.floor(0.98 * sorted.length)];
      expect(s.depthMin).toBeLessThanOrEqual(lo);
      expect(s.depthMax).toBeGreaterThanOrEqual(hi * 0.9);
    }
    let stats: FusionStats | null = null;
    const pc = await fuseDepthMaps(gt.views, gt.store, neighboursFromSelection(sel), { onStats: (s) => (stats = s) });
    const err = relErrors(pc, scene, gt.views);
    const sorted = Float64Array.from(err).sort();
    log(
      `yard 12 views 192×144 (${xyz.length / 3} sparse pts): ${pc.count} points, mean views ${stats!.meanViews.toFixed(2)}, ${stats!.timeMs.toFixed(0)} ms; ` +
        `rel. error median ${(100 * sorted[sorted.length >> 1]).toFixed(3)} %, p99 ${(100 * sorted[Math.floor(0.99 * sorted.length)]).toFixed(3)} %, within 0.5 %: ${(100 * frac(err, (v) => v < 0.005)).toFixed(2)} %`,
    );
    expect(pc.count).toBeGreaterThan(5000);
    expect(frac(err, (v) => v < 0.005)).toBeGreaterThan(0.98);
    expect(frac(err, (v) => v > 0.02)).toBeLessThan(0.002);
  });

  it('timing: 16 views at 640×480', async () => {
    const scene = makeNearScene();
    const cam = pinholeIntrinsics(640, 480, (60 * Math.PI) / 180);
    const poses = orbitPoses([0, 0, 0.25], 3, (35 * Math.PI) / 180, 16);
    const t0 = performance.now();
    const gt = renderGtViews(scene, cam, poses, { noise: 0.002, normalNoiseDeg: 3, outliers: 0.1, seed: 5 });
    const t1 = performance.now();
    let stats: FusionStats | null = null;
    const pc = await fuseDepthMaps(gt.views, gt.store, ringNeighbours(16, 3), { onStats: (s) => (stats = s) });
    const err = relErrors(pc, scene, gt.views);
    log(
      `640×480 × 16 views, 6 neighbours: render ${(t1 - t0).toFixed(0)} ms, fusion ${stats!.timeMs.toFixed(0)} ms ` +
        `(${((1000 * stats!.timeMs) / (16 * 640 * 480)).toFixed(3)} µs/px), ${pc.count} points, mean views ${stats!.meanViews.toFixed(2)}, ` +
        `within 0.5 %: ${(100 * frac(err, (v) => v < 0.005)).toFixed(2)} %`,
    );
    expect(frac(err, (v) => v < 0.005)).toBeGreaterThan(0.99);
  });

  it('cancellation throws and missing depth maps are skipped', async () => {
    const scene = makeNearScene();
    const cam = pinholeIntrinsics(48, 36, 1);
    const gt = renderGtViews(scene, cam, orbitPoses([0, 0, 0.25], 3, 0.6, 6));
    await expect(fuseDepthMaps(gt.views, gt.store, ringNeighbours(6, 2), {}, undefined, { cancelled: true })).rejects.toThrow(/cancelled/);
    const partial = new MemoryDepthMapStore();
    for (const id of [0, 1, 2]) await partial.put(id, (await gt.store.get(id))!);
    const pc = await fuseDepthMaps(gt.views, partial, ringNeighbours(6, 2));
    expect(pc.count).toBeGreaterThan(0);
  });
});

describe('fuseDepthMaps with depth maps at half the image resolution', () => {
  it('scales cameras per §2 and stays accurate', async () => {
    const scene = makeNearScene();
    const cam = pinholeIntrinsics(240, 180, (60 * Math.PI) / 180);
    const poses = orbitPoses([0, 0, 0.25], 3, (35 * Math.PI) / 180, 12);
    const full = renderGtViews(scene, cam, poses);
    // exact GT depth maps rendered with the half-resolution camera (§2 rule)
    const half = { ...cam, width: 120, height: 90, f: cam.f * 0.5, cx: (cam.cx + 0.5) * 0.5 - 0.5, cy: (cam.cy + 0.5) * 0.5 - 0.5 };
    const halfGt = renderGtViews(scene, half, poses, { noise: 0.001, seed: 2 });
    const pc = await fuseDepthMaps(full.views, halfGt.store, ringNeighbours(12, 2));
    const err = relErrors(pc, scene, full.views);
    const sorted = Float64Array.from(err).sort();
    log(`half-res depth maps (120×90) with 240×180 images: ${pc.count} points, rel. error median ${(100 * sorted[sorted.length >> 1]).toFixed(3)} %, within 0.5 %: ${(100 * frac(err, (v) => v < 0.005)).toFixed(2)} %`);
    expect(pc.count).toBeGreaterThan(5000);
    expect(frac(err, (v) => v < 0.005)).toBeGreaterThan(0.99);
    // colours sampled from the full-resolution images: compare with the image at the projection
    let csum = 0;
    for (let i = 0; i < pc.colors.length; i++) csum += pc.colors[i];
    expect(csum / pc.colors.length).toBeGreaterThan(30);
  });
});
