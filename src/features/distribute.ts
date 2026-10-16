/**
 * Spatially distributed feature selection.
 *
 * Plain "keep the N strongest" selection concentrates features on the most
 * textured part of the image, which hurts pose estimation (poor geometric
 * conditioning) and reconstruction coverage. We overlay a grid whose cell aspect
 * follows the image aspect, give every cell a quota of its strongest responses,
 * then fill the remaining budget globally by response.
 */

import type { FeatureSet } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';

export interface DistributeOptions {
  /** Approximate number of grid cells. Default clamp(round(maxCount / 16), 1, 1024). */
  cells?: number;
  /** Fraction of maxCount reserved for per-cell quotas. Default 0.5. */
  quotaFraction?: number;
}

/**
 * Indices 0..n-1 sorted by descending score (ties: ascending index). NaN scores rank
 * as -Infinity (a NaN-producing comparator would make the whole order unspecified).
 */
export function sortByScoreDesc(score: ArrayLike<number>, n: number): Uint32Array {
  const idx = new Uint32Array(n);
  const key = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    idx[i] = i;
    const v = score[i];
    key[i] = v === v ? v : -Infinity;
  }
  // n is a few tens of thousands at most; a comparator sort is fine. The comparator
  // is consistent (±Infinity ties fall back to the index) and fully deterministic.
  idx.sort((a, b) => {
    const d = key[b] - key[a];
    return d > 0 ? 1 : d < 0 ? -1 : a - b;
  });
  return idx;
}

/** Grid dimensions (gx, gy) with gx/gy ≈ width/height and gx*gy ≈ cells. */
export function gridDims(width: number, height: number, cells: number): [number, number] {
  const aspect = Math.max(1e-6, width) / Math.max(1e-6, height);
  const gx = Math.max(1, Math.round(Math.sqrt(cells * aspect)));
  const gy = Math.max(1, Math.round(cells / gx));
  return [gx, gy];
}

/**
 * Selects at most `maxCount` of `count` features, returning their indices sorted by
 * descending score. With `distribute` (default true) a per-cell quota of the
 * strongest responses is taken first, then the budget is filled globally by score.
 * Positions are in an image frame of `width` x `height` (any consistent frame).
 */
export function selectFeatureIndices(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  score: ArrayLike<number>,
  count: number,
  width: number,
  height: number,
  maxCount: number,
  distribute = true,
  opts: DistributeOptions = {},
): Uint32Array {
  const order = sortByScoreDesc(score, count);
  if (count <= maxCount) return order;
  if (maxCount <= 0) return new Uint32Array(0);
  if (!distribute) return order.slice(0, maxCount);

  const cells = opts.cells ?? Math.min(1024, Math.max(1, Math.round(maxCount / 16)));
  const [gx, gy] = gridDims(width, height, cells);
  const quota = Math.floor(((opts.quotaFraction ?? 0.5) * maxCount) / (gx * gy));
  const selected = new Uint8Array(count);
  let nsel = 0;
  if (quota > 0) {
    const cellCount = new Int32Array(gx * gy);
    const sx = gx / Math.max(1e-6, width);
    const sy = gy / Math.max(1e-6, height);
    for (let k = 0; k < count && nsel < maxCount; k++) {
      const i = order[k];
      // Frame covers [-0.5, W-0.5]; shift by 0.5 so cells partition it evenly.
      let cx = Math.floor((x[i] + 0.5) * sx);
      let cy = Math.floor((y[i] + 0.5) * sy);
      if (cx < 0) cx = 0;
      else if (cx >= gx) cx = gx - 1;
      if (cy < 0) cy = 0;
      else if (cy >= gy) cy = gy - 1;
      const c = cy * gx + cx;
      if (cellCount[c] < quota) {
        cellCount[c]++;
        selected[i] = 1;
        nsel++;
      }
    }
  }
  for (let k = 0; k < count && nsel < maxCount; k++) {
    const i = order[k];
    if (!selected[i]) {
      selected[i] = 1;
      nsel++;
    }
  }
  const out = new Uint32Array(nsel);
  let m = 0;
  for (let k = 0; k < count; k++) {
    const i = order[k];
    if (selected[i]) out[m++] = i;
  }
  return out;
}

/** New FeatureSet with the features at `indices` (in that order). */
export function subsetFeatureSet(fs: FeatureSet, indices: ArrayLike<number>): FeatureSet {
  const n = indices.length;
  const keypoints = new Float32Array(4 * n);
  const descriptors = new Uint8Array(DESCRIPTOR_SIZE * n);
  const scores = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    const i = indices[k];
    keypoints.set(fs.keypoints.subarray(4 * i, 4 * i + 4), 4 * k);
    descriptors.set(fs.descriptors.subarray(DESCRIPTOR_SIZE * i, DESCRIPTOR_SIZE * (i + 1)), DESCRIPTOR_SIZE * k);
    scores[k] = fs.scores[i];
  }
  return { count: n, keypoints, descriptors, scores, width: fs.width, height: fs.height };
}

/**
 * Keeps at most `maxCount` features of a FeatureSet (spatially distributed when
 * `distribute`), sorted by descending score. Useful after GPU extraction.
 */
export function selectFeatures(fs: FeatureSet, maxCount: number, distribute = true, opts?: DistributeOptions): FeatureSet {
  const n = fs.count;
  const xs = new Float32Array(n);
  const ys = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = fs.keypoints[4 * i];
    ys[i] = fs.keypoints[4 * i + 1];
  }
  const idx = selectFeatureIndices(xs, ys, fs.scores, n, fs.width, fs.height, maxCount, distribute, opts);
  return subsetFeatureSet(fs, idx);
}
