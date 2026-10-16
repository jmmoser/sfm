/**
 * End-to-end CPU dense pipeline on real PatchMatch output (not GT + noise):
 * sparse points → selectSourceViews → patchMatchCPU (photometric + geometric) →
 * fuseDepthMaps (+ filtered depth maps) → filterPointCloud → tsdfMesh → simplifyMesh.
 */
import { describe, expect, it } from 'vitest';
import type { DepthMap } from './types';
import { MemoryDepthMapStore } from './types';
import { Rng } from '../math/random';
import { cameraCenter, makeNearScene, orbitPoses, pinholeIntrinsics } from './synthetic';
import { distanceToScene, renderGtViews } from './testutil-scene';
import { selectSourceViews, neighboursFromSelection } from './view-selection';
import { patchMatchCPU } from './patchmatch-cpu';
import { fuseDepthMaps, type FusionStats } from './fusion';
import { filterPointCloud } from './pointcloud';
import { tsdfMesh, type TsdfStats } from './tsdf';
import { simplifyMesh } from './simplify';

const log = (...a: unknown[]): void => console.log('[dense-e2e]', ...a);

describe('dense pipeline end to end (CPU)', () => {
  it('near scene, 10 views 128×96: PatchMatch → fusion → filter → TSDF → simplify', async () => {
    const scene = makeNearScene();
    const cam = pinholeIntrinsics(128, 96, (60 * Math.PI) / 180);
    const poses = orbitPoses([0, 0, 0.25], 3, (35 * Math.PI) / 180, 10, 0.2);
    // views rendered with 2×2 supersampling (textures need it for stereo); GT kept for scoring
    const gt = renderGtViews(scene, cam, poses, { supersample: 2 });
    const views = gt.views;

    // sparse points from GT (stand-in for SfM): random pixels, visibility by depth test
    const rng = new Rng(2);
    const xyz: number[] = [];
    const vis: number[][] = [];
    for (let v = 0; v < views.length; v++) {
      const R = poses[v].R;
      const C = cameraCenter(poses[v]);
      for (let s = 0; s < 120; s++) {
        const k = rng.int(cam.width * cam.height);
        const z = gt.gtDepth[v][k];
        if (!(z > 0)) continue;
        const x = k % cam.width;
        const y = (k - x) / cam.width;
        const rx = ((x - cam.cx) / cam.f) * z;
        const ry = ((y - cam.cy) / cam.f) * z;
        const X = [0, 1, 2].map((a) => C[a] + R[a] * rx + R[3 + a] * ry + R[6 + a] * z);
        const seen: number[] = [];
        for (let w = 0; w < views.length; w++) {
          const Rw = poses[w].R;
          const tw = poses[w].t;
          const zc = Rw[6] * X[0] + Rw[7] * X[1] + Rw[8] * X[2] + tw[2];
          if (!(zc > 0)) continue;
          const u = Math.round((cam.f * (Rw[0] * X[0] + Rw[1] * X[1] + Rw[2] * X[2] + tw[0])) / zc + cam.cx);
          const vv = Math.round((cam.f * (Rw[3] * X[0] + Rw[4] * X[1] + Rw[5] * X[2] + tw[1])) / zc + cam.cy);
          if (u < 0 || vv < 0 || u >= cam.width || vv >= cam.height) continue;
          if (Math.abs(gt.gtDepth[w][vv * cam.width + u] - zc) < 0.01 * zc) seen.push(w);
        }
        if (seen.length >= 2) {
          xyz.push(...X);
          vis.push(seen);
        }
      }
    }
    const t0 = performance.now();
    const sel = selectSourceViews(views, { xyz: Float64Array.from(xyz), visibility: vis }, { numSources: 4 });
    const t1 = performance.now();

    // photometric pass, then geometric-consistency pass
    const photo = new Map<number, DepthMap>();
    for (const v of views) {
      const s = sel.get(v.id)!;
      photo.set(v.id, patchMatchCPU(v, s.sources.map((id) => views[id]), { min: s.depthMin, max: s.depthMax }, { iterations: 4 }));
    }
    const t2 = performance.now();
    const store = new MemoryDepthMapStore();
    for (const v of views) {
      const s = sel.get(v.id)!;
      const dm = patchMatchCPU(
        v,
        s.sources.map((id) => views[id]),
        { min: s.depthMin, max: s.depthMax },
        { geometricIterations: 2 },
        { prior: photo.get(v.id)!, sourceDepths: s.sources.map((id) => photo.get(id)!) },
      );
      await store.put(v.id, dm);
    }
    const t3 = performance.now();

    let fstats: FusionStats | null = null;
    const filtered = new MemoryDepthMapStore();
    const pc = await fuseDepthMaps(views, store, neighboursFromSelection(sel), { filteredStore: filtered, onStats: (s) => (fstats = s) });
    const t4 = performance.now();
    const pcf = filterPointCloud(pc, { statistical: { k: 8, std: 2 } });
    const t5 = performance.now();
    let tstats: TsdfStats | null = null;
    const mesh = await tsdfMesh(views, filtered, { onStats: (s) => (tstats = s) });
    const t6 = performance.now();
    const simp = simplifyMesh(mesh, Math.floor(mesh.triangleCount / 4));
    const t7 = performance.now();

    const centres = views.map((v) => cameraCenter(v.pose));
    const relErr = (x: number, y: number, z: number): number => {
      let dc = Infinity;
      for (const c of centres) dc = Math.min(dc, Math.hypot(x - c[0], y - c[1], z - c[2]));
      return distanceToScene(scene, x, y, z) / dc;
    };
    const stats = (P: Float32Array, n: number): { within05: number; within1: number; median: number } => {
      const e = new Float64Array(n);
      for (let i = 0; i < n; i++) e[i] = relErr(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
      const s = Float64Array.from(e).sort();
      let a = 0;
      let b = 0;
      for (const v of e) {
        if (v < 0.005) a++;
        if (v < 0.01) b++;
      }
      return { within05: a / n, within1: b / n, median: s[n >> 1] };
    };
    let rawValid = 0;
    let rawGood = 0;
    for (const v of views) {
      const dm = (await store.get(v.id))!;
      for (let k = 0; k < dm.depth.length; k++) {
        const z = dm.depth[k];
        const zg = gt.gtDepth[v.id][k];
        if (!(z > 0) || !(zg > 0)) continue;
        rawValid++;
        if (Math.abs(z - zg) < 0.01 * zg) rawGood++;
      }
    }
    const sp = stats(pc.positions, pc.count);
    const sf = stats(pcf.positions, pcf.count);
    const sm = stats(mesh.positions, mesh.vertexCount);
    const ts = tstats!;
    log(
      `timings: selection ${(t1 - t0).toFixed(0)} ms, PatchMatch photometric ${(t2 - t1).toFixed(0)} ms, geometric ${(t3 - t2).toFixed(0)} ms, ` +
        `fusion ${(t4 - t3).toFixed(0)} ms, SOR ${(t5 - t4).toFixed(0)} ms, TSDF ${(t6 - t5).toFixed(0)} ms, simplify ${(t7 - t6).toFixed(0)} ms`,
    );
    log(`raw depth maps: ${((100 * rawGood) / rawValid).toFixed(1)} % of valid pixels within 1 %`);
    log(
      `fused: ${pc.count} pts (mean views ${fstats!.meanViews.toFixed(2)}), within 0.5 % ${(100 * sp.within05).toFixed(2)} %, within 1 % ${(100 * sp.within1).toFixed(2)} %, median ${(100 * sp.median).toFixed(3)} %`,
    );
    log(`filtered: ${pcf.count} pts, within 0.5 % ${(100 * sf.within05).toFixed(2)} %, within 1 % ${(100 * sf.within1).toFixed(2)} %`);
    log(
      `mesh: voxel ${ts.voxelSize.toPrecision(3)}, ${mesh.triangleCount} tris → ${simp.triangleCount} simplified; vertices within 0.5 % ${(100 * sm.within05).toFixed(2)} %, within 1 % ${(100 * sm.within1).toFixed(2)} %, median ${(100 * sm.median).toFixed(3)} %`,
    );
    expect(pc.count).toBeGreaterThan(5000);
    // fusion removes inconsistent depths. (At this tiny resolution PatchMatch's 11×11 window
    // fattens edges; at 256×192 the same pipeline gives 99.5 % within 1 %, 96.9 % within 0.5 %.)
    expect(sp.within1).toBeGreaterThan(rawGood / rawValid + 0.2);
    expect(sp.within1).toBeGreaterThan(0.93);
    expect(sf.within1).toBeGreaterThanOrEqual(sp.within1 - 0.002);
    expect(mesh.triangleCount).toBeGreaterThan(2000);
    expect(sm.within1).toBeGreaterThan(0.9);
  });
});
