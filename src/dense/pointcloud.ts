/**
 * Point-cloud filters (ARCHITECTURE §4.6): voxel-grid downsampling and statistical
 * outlier removal (SOR).
 *
 *  - `voxelDownsample`: points are bucketed into a hash grid of `voxelSize` cells; each
 *    occupied cell yields one point with the mean position and colour, the mean normal
 *    (normals sign-aligned to the cell's first normal before averaging), the mean
 *    confidence and radius max(mean radius, 0.6 · voxelSize) so splats still close gaps.
 *  - `statisticalOutlierMask`: mean distance m_i to the k nearest neighbours (exact kNN
 *    with a k-d tree, so mixed close-range / far-range densities work — a uniform grid
 *    degenerates when the spacing varies by 100×); points with m_i > μ + std·σ are removed.
 *    With `scaleAdaptive` (default when the cloud has radii) the statistic is m_i / r_i,
 *    i.e. the distance in units of the point's own pixel footprint, so sparse far-away
 *    surfaces are not mistaken for outliers of a dense close-range object.
 */
import type { PointCloud } from '../core/types';
import { Int3HashMap } from './dense-util';
import { KdTree3 } from './kdtree';

export interface StatisticalFilterOptions {
  /** Neighbours per point. */
  k: number;
  /** Threshold multiplier σ: remove m_i > μ + std · σ. */
  std: number;
  /** Normalise distances by the point radii (default: true when `radii` exist). */
  scaleAdaptive?: boolean;
}

export interface PointCloudFilterOptions {
  /** Voxel-grid downsampling cell size (world units); omitted / ≤ 0 = off. Applied first. */
  voxelSize?: number;
  /** Statistical outlier removal; omitted = off. */
  statistical?: StatisticalFilterOptions;
  /** Drop points with confidence below this (needs `confidence`). Applied first. */
  minConfidence?: number;
}

/** Subset of a point cloud (keep[i] ≠ 0). */
export function selectPoints(pc: PointCloud, keep: Uint8Array): PointCloud {
  let n = 0;
  for (let i = 0; i < pc.count; i++) if (keep[i]) n++;
  const positions = new Float32Array(3 * n);
  const colors = new Uint8Array(3 * n);
  const normals = pc.normals ? new Float32Array(3 * n) : undefined;
  const radii = pc.radii ? new Float32Array(n) : undefined;
  const confidence = pc.confidence ? new Float32Array(n) : undefined;
  let o = 0;
  for (let i = 0; i < pc.count; i++) {
    if (!keep[i]) continue;
    for (let k = 0; k < 3; k++) {
      positions[3 * o + k] = pc.positions[3 * i + k];
      colors[3 * o + k] = pc.colors[3 * i + k];
      if (normals) normals[3 * o + k] = pc.normals![3 * i + k];
    }
    if (radii) radii[o] = pc.radii![i];
    if (confidence) confidence[o] = pc.confidence![i];
    o++;
  }
  const out: PointCloud = { count: n, positions, colors };
  if (normals) out.normals = normals;
  if (radii) out.radii = radii;
  if (confidence) out.confidence = confidence;
  return out;
}

/** Voxel-grid downsampling (see file header). */
export function voxelDownsample(pc: PointCloud, voxelSize: number): PointCloud {
  if (!(voxelSize > 0) || pc.count === 0) return pc;
  const n = pc.count;
  const inv = 1 / voxelSize;
  const map = new Int3HashMap(Math.min(n, 1 << 20));
  const cellOf = new Int32Array(n);
  let cells = 0;
  const P = pc.positions;
  for (let i = 0; i < n; i++) {
    const c = map.getOrInsert(Math.floor(P[3 * i] * inv), Math.floor(P[3 * i + 1] * inv), Math.floor(P[3 * i + 2] * inv), cells);
    if (map.inserted) cells++;
    cellOf[i] = c;
  }
  const cnt = new Uint32Array(cells);
  const sp = new Float64Array(3 * cells);
  const sc = new Float64Array(3 * cells);
  const sn = pc.normals ? new Float64Array(3 * cells) : null;
  const first = pc.normals ? new Int32Array(cells).fill(-1) : null;
  const sr = pc.radii ? new Float64Array(cells) : null;
  const sconf = pc.confidence ? new Float64Array(cells) : null;
  for (let i = 0; i < n; i++) {
    const c = cellOf[i];
    cnt[c]++;
    for (let k = 0; k < 3; k++) {
      sp[3 * c + k] += P[3 * i + k];
      sc[3 * c + k] += pc.colors[3 * i + k];
    }
    if (sn) {
      const N = pc.normals!;
      let s = 1;
      if (first![c] < 0) first![c] = i;
      else {
        const f = first![c];
        if (N[3 * i] * N[3 * f] + N[3 * i + 1] * N[3 * f + 1] + N[3 * i + 2] * N[3 * f + 2] < 0) s = -1;
      }
      sn[3 * c] += s * N[3 * i];
      sn[3 * c + 1] += s * N[3 * i + 1];
      sn[3 * c + 2] += s * N[3 * i + 2];
    }
    if (sr) sr[c] += pc.radii![i];
    if (sconf) sconf[c] += pc.confidence![i];
  }
  const positions = new Float32Array(3 * cells);
  const colors = new Uint8Array(3 * cells);
  const normals = sn ? new Float32Array(3 * cells) : undefined;
  const radii = sr ? new Float32Array(cells) : undefined;
  const confidence = sconf ? new Float32Array(cells) : undefined;
  for (let c = 0; c < cells; c++) {
    const ic = 1 / cnt[c];
    for (let k = 0; k < 3; k++) {
      positions[3 * c + k] = sp[3 * c + k] * ic;
      colors[3 * c + k] = Math.round(sc[3 * c + k] * ic);
    }
    if (normals) {
      let nx = sn![3 * c];
      let ny = sn![3 * c + 1];
      let nz = sn![3 * c + 2];
      let l = Math.hypot(nx, ny, nz);
      if (!(l > 1e-12)) {
        const f = first![c];
        nx = pc.normals![3 * f];
        ny = pc.normals![3 * f + 1];
        nz = pc.normals![3 * f + 2];
        l = Math.hypot(nx, ny, nz) || 1;
      }
      normals[3 * c] = nx / l;
      normals[3 * c + 1] = ny / l;
      normals[3 * c + 2] = nz / l;
    }
    if (radii) radii[c] = Math.max(sr![c] * ic, 0.6 * voxelSize);
    if (confidence) confidence[c] = sconf![c] * ic;
  }
  const out: PointCloud = { count: cells, positions, colors };
  if (normals) out.normals = normals;
  if (radii) out.radii = radii;
  if (confidence) out.confidence = confidence;
  return out;
}

/**
 * Statistical outlier removal mask (1 = keep). Also returns the per-point statistic
 * (mean kNN distance, divided by the radius when scale-adaptive) and the threshold.
 */
export function statisticalOutlierMask(
  pc: PointCloud,
  opts: StatisticalFilterOptions,
): { keep: Uint8Array; meanDist: Float64Array; threshold: number } {
  const n = pc.count;
  const k = Math.max(1, Math.floor(opts.k));
  const keep = new Uint8Array(n).fill(1);
  const meanDist = new Float64Array(n);
  if (n <= k) return { keep, meanDist, threshold: Infinity };
  const adaptive = (opts.scaleAdaptive ?? !!pc.radii) && !!pc.radii;
  const tree = new KdTree3(pc.positions, n);
  const d2 = new Float64Array(k);
  const ii = new Int32Array(k);
  let sum = 0;
  let sum2 = 0;
  let m = 0;
  // queries in tree order (spatially coherent → cache friendly)
  const order = tree.perm;
  for (let o = 0; o < n; o++) {
    const i = order[o];
    const got = tree.knnAt(o, k, d2, ii);
    let s = 0;
    for (let j = 0; j < got; j++) s += Math.sqrt(d2[j]);
    let v = got > 0 ? s / got : Infinity;
    if (adaptive) {
      const r = pc.radii![i];
      v = r > 0 ? v / r : Infinity;
    }
    meanDist[i] = v;
    if (Number.isFinite(v)) {
      sum += v;
      sum2 += v * v;
      m++;
    }
  }
  if (m === 0) return { keep, meanDist, threshold: Infinity };
  const mu = sum / m;
  const sigma = Math.sqrt(Math.max(0, sum2 / m - mu * mu));
  const threshold = mu + opts.std * sigma;
  for (let i = 0; i < n; i++) if (!(meanDist[i] <= threshold)) keep[i] = 0;
  return { keep, meanDist, threshold };
}

/** Filter a point cloud: confidence threshold → voxel downsampling → statistical outlier removal. */
export function filterPointCloud(pc: PointCloud, opts: PointCloudFilterOptions = {}): PointCloud {
  let out = pc;
  if (opts.minConfidence !== undefined && pc.confidence) {
    const keep = new Uint8Array(pc.count);
    for (let i = 0; i < pc.count; i++) keep[i] = pc.confidence[i] >= opts.minConfidence ? 1 : 0;
    out = selectPoints(out, keep);
  }
  if (opts.voxelSize !== undefined && opts.voxelSize > 0) out = voxelDownsample(out, opts.voxelSize);
  if (opts.statistical && opts.statistical.k > 0) {
    const { keep } = statisticalOutlierMask(out, opts.statistical);
    out = selectPoints(out, keep);
  }
  return out;
}
