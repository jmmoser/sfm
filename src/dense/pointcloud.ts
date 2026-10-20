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
 *    surfaces are not mistaken for outliers of a dense close-range object (points without a
 *    usable radius use the median radius). NaN positions are removed.
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
  return mergeCells(pc, cellOf, cells, () => voxelSize);
}

/** Levels of `footprintDownsample` (half octaves of the radius; cells of different levels never merge). */
const LOD_LEVELS = 64;
const LOD_LEVELS_PER_OCTAVE = 2;

/**
 * Footprint-adaptive voxel LOD (the dense point cap). Every point is merged with the points of
 * its cell in a grid whose cell size is `s` × its own splat radius (`radii`, ≈ 0.7 pixel
 * footprints on the surface), quantised to half-octave levels. Near / fine geometry thus keeps
 * its own resolution instead of being merged at one cloud-wide voxel size (which, sized from the
 * median radius, merged a close-range object ~10× and averaged its points across surfaces), and a
 * merged centroid moves by at most about one cell, i.e. a few footprints. `s` is searched so
 * that at most `maxPoints` cells remain (≥ 90 % of it when the search converges). Merged
 * attributes as in `voxelDownsample` (radius ≥ 0.6 × the cell). Points without a finite positive
 * radius use the median radius; without `radii` the grid is uniform (searched voxel size).
 * Non-finite positions are dropped.
 */
export function footprintDownsample(pc: PointCloud, maxPoints: number): PointCloud {
  const n = pc.count;
  const target = Math.max(1, Math.floor(maxPoints));
  if (n <= target) return pc;
  const P = pc.positions;
  const radii = pc.radii;
  // median / minimum radius (subsample) and coordinate range
  let rMed = 1;
  let rMin = 1;
  if (radii) {
    const rs: number[] = [];
    const stride = Math.max(1, Math.floor(n / 50_000));
    for (let i = 0; i < n; i += stride) if (radii[i] > 0 && radii[i] < Infinity) rs.push(radii[i]);
    if (rs.length) {
      rs.sort((a, b) => a - b);
      rMed = rs[rs.length >> 1];
      rMin = rs[0];
    }
  }
  let maxAbs = 0;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < 3 * n; i++) {
    const v = P[i];
    if (!Number.isFinite(v)) continue;
    maxAbs = Math.max(maxAbs, Math.abs(v));
    lo[i % 3] = Math.min(lo[i % 3], v);
    hi[i % 3] = Math.max(hi[i % 3], v);
  }
  if (!radii) {
    // no footprints: a uniform grid, starting from the spacing of a surface filling the box
    rMed = rMin = Math.max(1e-12, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / Math.sqrt(n)) || 1;
  }
  const r0 = Math.max(rMin * 0.5, rMed * 1e-4, 1e-12);
  // smallest scale: cell indices × levels stay inside int32
  const sMin = Math.max(1e-3, (maxAbs * 64) / 2 ** 30 / r0);
  const level = new Uint8Array(n);
  const valid = new Uint8Array(n);
  let nValid = 0;
  for (let i = 0; i < n; i++) {
    if (!(Number.isFinite(P[3 * i]) && Number.isFinite(P[3 * i + 1]) && Number.isFinite(P[3 * i + 2]))) continue;
    valid[i] = 1;
    nValid++;
    const r = radii && radii[i] > 0 && radii[i] < Infinity ? radii[i] : rMed;
    const l = Math.floor(LOD_LEVELS_PER_OCTAVE * Math.log2(Math.max(r, r0) / r0));
    level[i] = l < 0 ? 0 : l >= LOD_LEVELS ? LOD_LEVELS - 1 : l;
  }
  // cell size of level l at scale s: s · r0 · 2^((l + ½) / levelsPerOctave) (the level's geometric centre)
  const levelSize = new Float64Array(LOD_LEVELS);
  for (let l = 0; l < LOD_LEVELS; l++) levelSize[l] = r0 * Math.pow(2, (l + 0.5) / LOD_LEVELS_PER_OCTAVE);
  const inv = new Float64Array(LOD_LEVELS);
  // a pass stops once it has more cells than `abortAt` (its count is then extrapolated): bounds
  // the table to ≈ 1.25 × target entries
  const abortAt = Math.min(nValid, Math.ceil(1.25 * target));
  const tableRef = (): CellTable => table!;
  let table: CellTable | null = new CellTable(abortAt + 1);
  /** Cells at scale `scale` (into `cellOf`); > target ⇒ possibly an extrapolated estimate. */
  const assign = (scale: number, cellOf: Int32Array): number => {
    for (let l = 0; l < LOD_LEVELS; l++) inv[l] = 1 / (scale * levelSize[l]);
    const table = tableRef();
    table.clear();
    for (let i = 0; i < n; i++) {
      if (!valid[i]) {
        cellOf[i] = -1;
        continue;
      }
      const l = level[i];
      const k = inv[l];
      cellOf[i] = table.getOrInsert(Math.floor(P[3 * i] * k), Math.floor(P[3 * i + 1] * k), Math.floor(P[3 * i + 2] * k) * LOD_LEVELS + l);
      if (table.size > abortAt) return Math.max(table.size, Math.round((table.size * n) / (i + 1)));
    }
    return table.size;
  };
  // search s (cells fall roughly as 1 / s² on surfaces): secant steps inside a bracket; any count
  // in [0.9, 1] × target is accepted
  let cellOf = new Int32Array(n);
  let spare = new Int32Array(n);
  let s = Math.max(sMin, 1.8 * Math.sqrt(nValid / target));
  let okS = Infinity; // smallest s seen with cells ≤ target (its assignment is in `cellOf`)
  let okCells = 0;
  let failS = sMin; // largest s seen with cells > target
  for (let it = 0; it < 8; it++) {
    const cells = assign(s, spare);
    if (cells <= target) {
      if (s < okS) {
        okS = s;
        okCells = cells;
        [cellOf, spare] = [spare, cellOf];
      }
      if (cells >= 0.9 * target) break;
    } else failS = Math.max(failS, s);
    let next = s * Math.sqrt(cells / (0.9 * target));
    if (!(next > failS && next < okS)) next = okS < Infinity ? Math.sqrt(failS * okS) : 2 * s;
    s = next;
  }
  for (s = okS < Infinity ? okS : 2 * s; !(okS < Infinity); s *= 1.5) {
    // never under the target (degenerate radii): grow until it is
    const cells = assign(s, cellOf);
    if (cells <= target) {
      okS = s;
      okCells = cells;
    }
  }
  const cells = okCells;
  // search scratch is garbage now (the merge allocates the output)
  spare = new Int32Array(0);
  table = null;
  // per-cell level (cell size) for the merged radius
  const cellLevel = new Uint8Array(cells);
  for (let i = 0; i < n; i++) if (cellOf[i] >= 0) cellLevel[cellOf[i]] = level[i];
  return mergeCells(pc, cellOf, cells, (c) => okS * levelSize[cellLevel[c]]);
}

/**
 * Set of integer cells (x, y, z) → dense index (insertion order): open addressing, linear
 * probing, one interleaved (x, y, z, index) slot per entry (one cache line per probe), sized once
 * and cleared between the scale-search passes (no reallocation).
 */
class CellTable {
  private readonly slots: Int32Array;
  private readonly cap: number;
  size = 0;

  /** `maxEntries`: upper bound of the entries (no growth; load factor ≤ 0.6). */
  constructor(maxEntries: number) {
    this.cap = Math.max(1024, Math.ceil(maxEntries / 0.6));
    this.slots = new Int32Array(4 * this.cap);
    this.clear();
  }

  clear(): void {
    const s = this.slots;
    for (let i = 3; i < s.length; i += 4) s[i] = -1;
    this.size = 0;
  }

  getOrInsert(x: number, y: number, z: number): number {
    let h = Math.imul(x, 0x8da6b343) ^ Math.imul(y, 0xd8163841) ^ Math.imul(z, 0xcb1ab31f);
    h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
    h ^= h >>> 13;
    const cap = this.cap;
    const s = this.slots;
    let i = Math.floor(((h >>> 0) / 4294967296) * cap);
    for (;;) {
      const k = 4 * i;
      const v = s[k + 3];
      if (v < 0) {
        s[k] = x;
        s[k + 1] = y;
        s[k + 2] = z;
        s[k + 3] = this.size;
        return this.size++;
      }
      if (s[k] === x && s[k + 1] === y && s[k + 2] === z) return v;
      if (++i === cap) i = 0;
    }
  }
}

/**
 * One point per cell (`cellOf[i]` in [0, cells), −1 = drop): mean position / colour / confidence,
 * mean normal (sign-aligned to the cell's first normal), radius max(mean radius, 0.6 · cell size).
 * The points are counting-sorted by cell first, so the only per-cell scratch is the bucket
 * offsets (8 B per cell besides the output, instead of ~100 B of accumulators).
 */
function mergeCells(pc: PointCloud, cellOf: Int32Array, cells: number, cellSize: (c: number) => number): PointCloud {
  const n = pc.count;
  const P = pc.positions;
  const C = pc.colors;
  const N = pc.normals;
  const R = pc.radii;
  const F = pc.confidence;
  const start = new Int32Array(cells + 1);
  for (let i = 0; i < n; i++) if (cellOf[i] >= 0) start[cellOf[i] + 1]++;
  for (let c = 0; c < cells; c++) start[c + 1] += start[c];
  const cursor = start.slice(0, cells);
  const order = new Int32Array(start[cells]);
  for (let i = 0; i < n; i++) {
    const c = cellOf[i];
    if (c >= 0) order[cursor[c]++] = i;
  }
  const positions = new Float32Array(3 * cells);
  const colors = new Uint8Array(3 * cells);
  const normals = N ? new Float32Array(3 * cells) : undefined;
  const radii = R ? new Float32Array(cells) : undefined;
  const confidence = F ? new Float32Array(cells) : undefined;
  for (let c = 0; c < cells; c++) {
    const a = start[c];
    const b = start[c + 1];
    const ic = 1 / (b - a);
    let px = 0;
    let py = 0;
    let pz = 0;
    let cr = 0;
    let cg = 0;
    let cb = 0;
    let nx = 0;
    let ny = 0;
    let nz = 0;
    let sr = 0;
    let sf = 0;
    const f = order[a];
    for (let k = a; k < b; k++) {
      const i = order[k];
      px += P[3 * i];
      py += P[3 * i + 1];
      pz += P[3 * i + 2];
      cr += C[3 * i];
      cg += C[3 * i + 1];
      cb += C[3 * i + 2];
      if (N) {
        const sgn = N[3 * i] * N[3 * f] + N[3 * i + 1] * N[3 * f + 1] + N[3 * i + 2] * N[3 * f + 2] < 0 ? -1 : 1;
        nx += sgn * N[3 * i];
        ny += sgn * N[3 * i + 1];
        nz += sgn * N[3 * i + 2];
      }
      if (R) sr += R[i];
      if (F) sf += F[i];
    }
    positions[3 * c] = px * ic;
    positions[3 * c + 1] = py * ic;
    positions[3 * c + 2] = pz * ic;
    colors[3 * c] = Math.round(cr * ic);
    colors[3 * c + 1] = Math.round(cg * ic);
    colors[3 * c + 2] = Math.round(cb * ic);
    if (normals) {
      let l = Math.hypot(nx, ny, nz);
      if (!(l > 1e-12)) {
        nx = N![3 * f];
        ny = N![3 * f + 1];
        nz = N![3 * f + 2];
        l = Math.hypot(nx, ny, nz) || 1;
      }
      normals[3 * c] = nx / l;
      normals[3 * c + 1] = ny / l;
      normals[3 * c + 2] = nz / l;
    }
    if (radii) radii[c] = Math.max(sr * ic, 0.6 * cellSize(c));
    if (confidence) confidence[c] = sf * ic;
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
  let adaptive = (opts.scaleAdaptive ?? !!pc.radii) && !!pc.radii;
  // Points without a usable radius (0, negative, NaN — e.g. merged from another source) are
  // normalised by the median radius instead of getting an infinite statistic (which removed
  // them all).
  let rSub = 1;
  if (adaptive) {
    const radii = pc.radii!;
    const stride = Math.max(1, Math.floor(n / 20000));
    const rs: number[] = [];
    for (let i = 0; i < n; i += stride) if (radii[i] > 0 && radii[i] < Infinity) rs.push(radii[i]);
    if (rs.length === 0) adaptive = false;
    else {
      rs.sort((a, b) => a - b);
      rSub = rs[rs.length >> 1];
    }
  }
  const tree = new KdTree3(pc.positions, n);
  const d2 = new Float64Array(k);
  const ii = new Int32Array(k);
  let sum = 0;
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
      v /= r > 0 && r < Infinity ? r : rSub;
    }
    meanDist[i] = v;
    if (Number.isFinite(v)) {
      sum += v;
      m++;
    }
  }
  if (m === 0) return { keep, meanDist, threshold: Infinity };
  // two-pass variance (Σv² / m − μ² cancels catastrophically for near-identical values)
  const mu = sum / m;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const v = meanDist[i];
    if (Number.isFinite(v)) ss += (v - mu) * (v - mu);
  }
  const sigma = Math.sqrt(ss / m);
  const threshold = mu + opts.std * sigma;
  // rounding tolerance: identical statistics must never fall above their own mean
  const keepBelow = threshold + 1e-9 * Math.abs(mu);
  for (let i = 0; i < n; i++) if (!(meanDist[i] <= keepBelow)) keep[i] = 0;
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
