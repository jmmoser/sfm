/**
 * Test helpers for the fusion / meshing tests: exact distance from a point to the
 * surfaces of a synthetic scene, GT depth-map stores with noise and outliers.
 * Browser-safe (no Node APIs).
 */
import type { Pose } from '../core/types';
import { Rng } from '../math/random';
import { groundTruthDepthMap, renderSynthetic, type SynthScene } from './synthetic';
import { MemoryDepthMapStore, type DenseView } from './types';

/** Unsigned distance from (x, y, z) to the nearest primitive surface of `scene`. */
export function distanceToScene(scene: SynthScene, x: number, y: number, z: number): number {
  let best = Infinity;
  for (const p of scene.primitives) {
    let d: number;
    if (p.kind === 'sphere') {
      d = Math.abs(Math.hypot(x - p.center[0], y - p.center[1], z - p.center[2]) - p.radius);
    } else if (p.kind === 'box') {
      const yaw = p.yaw ?? 0;
      const c = Math.cos(yaw);
      const s = Math.sin(yaw);
      const rx = x - p.center[0];
      const ry = y - p.center[1];
      const lx = c * rx + s * ry;
      const ly = -s * rx + c * ry;
      const lz = z - p.center[2];
      const qx = Math.abs(lx) - p.halfSize[0];
      const qy = Math.abs(ly) - p.halfSize[1];
      const qz = Math.abs(lz) - p.halfSize[2];
      const outside = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0));
      const inside = Math.min(Math.max(qx, qy, qz), 0);
      d = Math.abs(outside + inside);
    } else {
      const dx = x - p.center[0];
      const dy = y - p.center[1];
      const dz = z - p.center[2];
      const u = dx * p.axisU[0] + dy * p.axisU[1] + dz * p.axisU[2];
      const v = dx * p.axisV[0] + dy * p.axisV[1] + dz * p.axisV[2];
      // normal component = residual after removing the in-plane parts
      const wx = dx - u * p.axisU[0] - v * p.axisV[0];
      const wy = dy - u * p.axisU[1] - v * p.axisV[1];
      const wz = dz - u * p.axisU[2] - v * p.axisV[2];
      const cu = Math.max(-p.halfU, Math.min(p.halfU, u));
      const cv = Math.max(-p.halfV, Math.min(p.halfV, v));
      d = Math.sqrt((u - cu) ** 2 + (v - cv) ** 2 + wx * wx + wy * wy + wz * wz);
    }
    if (d < best) best = d;
  }
  return best;
}

export interface GtViews {
  views: DenseView[];
  gtDepth: Float32Array[];
  gtNormal: Float32Array[];
  store: MemoryDepthMapStore;
}

/**
 * Render `poses` with the pinhole camera and put (optionally corrupted) GT depth maps in a
 * MemoryDepthMapStore: Gaussian relative depth noise `noise` (σ, fraction of depth),
 * normals tilted by ≈ `normalNoiseDeg`, and a fraction `outliers` of valid pixels replaced
 * by random depths in [0.3, 3] × GT with random camera-facing normals.
 */
export function renderGtViews(
  scene: SynthScene,
  cam: { width: number; height: number; f: number; cx: number; cy: number; k1: number; k2: number },
  poses: Pose[],
  opts: { noise?: number; normalNoiseDeg?: number; outliers?: number; seed?: number; supersample?: number } = {},
): GtViews {
  const rng = new Rng(opts.seed ?? 7);
  const views: DenseView[] = [];
  const gtDepth: Float32Array[] = [];
  const gtNormal: Float32Array[] = [];
  const store = new MemoryDepthMapStore();
  const noise = opts.noise ?? 0;
  const nn = ((opts.normalNoiseDeg ?? 0) * Math.PI) / 180;
  const outl = opts.outliers ?? 0;
  for (let i = 0; i < poses.length; i++) {
    const r = renderSynthetic(scene, cam, poses[i], { supersample: opts.supersample ?? 1 });
    views.push({ id: i, cam: { ...cam }, pose: poses[i], image: r.image });
    gtDepth.push(r.depth);
    gtNormal.push(r.normal);
    const dm = groundTruthDepthMap(cam.width, cam.height, r.depth, r.normal);
    for (let k = 0; k < dm.depth.length; k++) {
      if (!(dm.depth[k] > 0)) continue;
      const x = k % cam.width;
      const y = (k - x) / cam.width;
      const rx = (x - cam.cx) / cam.f;
      const ry = (y - cam.cy) / cam.f;
      if (outl > 0 && rng.next() < outl) {
        dm.depth[k] *= 0.3 + 2.7 * rng.next();
        // random normal facing the camera
        let nx = rng.gaussian();
        let ny = rng.gaussian();
        let nz = rng.gaussian();
        if (nx * rx + ny * ry + nz > 0) {
          nx = -nx;
          ny = -ny;
          nz = -nz;
        }
        const l = Math.hypot(nx, ny, nz);
        dm.normal[3 * k] = nx / l;
        dm.normal[3 * k + 1] = ny / l;
        dm.normal[3 * k + 2] = nz / l;
        dm.cost[k] = 0.4;
        continue;
      }
      if (noise > 0) dm.depth[k] *= 1 + noise * rng.gaussian();
      if (nn > 0) {
        let nx = dm.normal[3 * k] + nn * rng.gaussian() * 0.7;
        let ny = dm.normal[3 * k + 1] + nn * rng.gaussian() * 0.7;
        let nz = dm.normal[3 * k + 2] + nn * rng.gaussian() * 0.7;
        if (nx * rx + ny * ry + nz > 0) {
          nx = dm.normal[3 * k];
          ny = dm.normal[3 * k + 1];
          nz = dm.normal[3 * k + 2];
        }
        const l = Math.hypot(nx, ny, nz);
        dm.normal[3 * k] = nx / l;
        dm.normal[3 * k + 1] = ny / l;
        dm.normal[3 * k + 2] = nz / l;
      }
    }
    void store.put(i, dm);
  }
  return { views, gtDepth, gtNormal, store };
}
