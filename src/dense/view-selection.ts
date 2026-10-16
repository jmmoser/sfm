/**
 * Source-view selection and per-view depth ranges for PatchMatch MVS (ARCHITECTURE §4.6).
 *
 * COLMAP-like: for each reference view, every other view is scored over the sparse
 * points both observe:
 *
 *   score_j = Σ_p  w_angle(θ_p) · w_scale(ρ_p)
 *
 *  - θ_p: triangulation angle at p between the two viewing rays. Smooth prior: rises as
 *    (θ/θ_lo)² below `minTriAngleDeg` (small baselines carry little depth information and
 *    a zero-baseline source is photo-consistent at every depth), 1 up to `maxTriAngleDeg`,
 *    then a Gaussian fall-off (σ = 15°; strong perspective distortion / occlusions).
 *  - ρ_p = footprint_ref / footprint_j (footprint = distance / f): Goesele et al. 2007 scale
 *    term, 1 for 1 ≤ ρ < 2, ρ for ρ < 1 (source sees p coarser), 2/ρ for ρ ≥ 2.
 *
 * Pairs whose 75th-percentile triangulation angle is below `hardMinTriAngleDeg`
 * (COLMAP's filter) or that share fewer than `minSharedPoints` points are discarded.
 * Up to `numSources` views are then picked greedily; after each pick the remaining
 * scores of views seeing the scene from (almost) the same direction as an already
 * selected one are down-weighted (near-duplicate keyframes add little). When fewer than
 * `minSources` views qualify, views whose frustum contains the reference's points are
 * added (geometric fallback).
 *
 * Depth range: robust percentiles (default 1st / 99th) of the z-depths of the sparse
 * points visible in the reference, widened ×0.8 / ×1.25. With fewer than
 * `minPointsForRange` visible points, all sparse points projecting into the reference
 * image in front of it are used; without any, a range is derived from the baselines.
 */
import type { CameraIntrinsics, Pose } from '../core/types';
import { poseCentre, quantile } from './dense-util';

export interface ViewSelectionView {
  id: number;
  pose: Pose;
  cam: CameraIntrinsics;
}

export interface SparsePointsInput {
  /** 3 per point, world frame. */
  xyz: Float64Array;
  /** View ids observing each point. */
  visibility: number[][];
}

export interface ViewSelectionOptions {
  /** Maximum number of sources per reference. Default 8. */
  numSources: number;
  /** Minimum number of sources (geometric fallback fills up to this). Default 2. */
  minSources: number;
  /** Lower knee of the triangulation-angle prior (degrees). Default 3. */
  minTriAngleDeg: number;
  /** Upper knee of the triangulation-angle prior (degrees). Default 45. */
  maxTriAngleDeg: number;
  /** Hard filter on the 75th-percentile triangulation angle of a pair (degrees). Default 1. */
  hardMinTriAngleDeg: number;
  /** Minimum number of shared sparse points for a candidate. Default 5. */
  minSharedPoints: number;
  /** Viewing directions closer than this (degrees) count as redundant. Default 3. */
  diversityAngleDeg: number;
  /** Depth range percentiles (0..100). Defaults 1 and 99. */
  depthPercentileLow: number;
  depthPercentileHigh: number;
  /** Margins applied to the percentiles. Defaults 0.8 and 1.25. */
  depthMarginMin: number;
  depthMarginMax: number;
  /** Below this many visible points the range falls back to all points in the frustum. Default 10. */
  minPointsForRange: number;
}

export const DEFAULT_VIEW_SELECTION_OPTIONS: Readonly<ViewSelectionOptions> = Object.freeze({
  numSources: 8,
  minSources: 2,
  minTriAngleDeg: 3,
  maxTriAngleDeg: 45,
  hardMinTriAngleDeg: 1,
  minSharedPoints: 5,
  diversityAngleDeg: 3,
  depthPercentileLow: 1,
  depthPercentileHigh: 99,
  depthMarginMin: 0.8,
  depthMarginMax: 1.25,
  minPointsForRange: 10,
});

export interface SourceViewSelection {
  /** Selected source view ids, best first. */
  sources: number[];
  depthMin: number;
  depthMax: number;
  /** Score of each selected source (same order). */
  scores: number[];
}

const DEG = Math.PI / 180;

/** Triangulation-angle prior (θ in radians). */
export function triangulationAngleWeight(theta: number, loRad: number, hiRad: number): number {
  if (theta < loRad) {
    const r = theta / loRad;
    return r * r;
  }
  if (theta <= hiRad) return 1;
  const s = (theta - hiRad) / (15 * DEG);
  return Math.exp(-s * s);
}

/** Goesele et al. scale weight for footprint ratio ρ = footprint_ref / footprint_src. */
export function scaleWeight(rho: number): number {
  if (rho >= 2) return 2 / rho;
  if (rho >= 1) return 1;
  return Math.max(0, rho);
}

/**
 * Select source views and depth ranges for every view (see file header). Returns a map
 * view id → selection. Views without any usable source get `sources: []`.
 */
export function selectSourceViews(
  views: ViewSelectionView[],
  sparsePoints: SparsePointsInput,
  options: Partial<ViewSelectionOptions> = {},
): Map<number, SourceViewSelection> {
  const o: ViewSelectionOptions = { ...DEFAULT_VIEW_SELECTION_OPTIONS, ...options };
  // numSources is a hard cap (the geometric fallback fills up to minSources ≤ numSources)
  o.minSources = Math.min(o.minSources, o.numSources);
  const nv = views.length;
  const result = new Map<number, SourceViewSelection>();
  if (nv === 0) return result;
  const index = new Map<number, number>();
  views.forEach((v, i) => index.set(v.id, i));
  const centres = new Float64Array(3 * nv);
  const c3 = new Float64Array(3);
  for (let i = 0; i < nv; i++) {
    poseCentre(views[i].pose, c3);
    centres.set(c3, 3 * i);
  }
  const xyz = sparsePoints.xyz;
  const np = Math.min(sparsePoints.visibility.length, Math.floor(xyz.length / 3));

  // CSR: per view, the points it observes; per point, the view indices observing it.
  const visStart = new Int32Array(np + 1);
  const visList: number[] = [];
  const perViewCount = new Int32Array(nv);
  for (let p = 0; p < np; p++) {
    visStart[p] = visList.length;
    const vis = sparsePoints.visibility[p];
    if (!vis) continue;
    for (const id of vis) {
      const vi = index.get(id);
      if (vi === undefined) continue;
      visList.push(vi);
      perViewCount[vi]++;
    }
  }
  visStart[np] = visList.length;
  const vis = Int32Array.from(visList);
  const viewStart = new Int32Array(nv + 1);
  for (let i = 0; i < nv; i++) viewStart[i + 1] = viewStart[i] + perViewCount[i];
  const viewPts = new Int32Array(viewStart[nv]);
  const fill = viewStart.slice(0, nv);
  for (let p = 0; p < np; p++) {
    for (let k = visStart[p]; k < visStart[p + 1]; k++) viewPts[fill[vis[k]]++] = p;
  }

  const lo = o.minTriAngleDeg * DEG;
  const hi = o.maxTriAngleDeg * DEG;
  const hardMin = o.hardMinTriAngleDeg * DEG;
  const score = new Float64Array(nv);
  const shared = new Int32Array(nv);
  const angles: number[][] = Array.from({ length: nv }, () => []);
  // per-candidate mean viewing direction (unit vectors from the points to the camera)
  const dirSum = new Float64Array(3 * nv);

  for (let r = 0; r < nv; r++) {
    const ref = views[r];
    score.fill(0);
    shared.fill(0);
    dirSum.fill(0);
    for (const a of angles) a.length = 0;
    const crx = centres[3 * r];
    const cry = centres[3 * r + 1];
    const crz = centres[3 * r + 2];
    const fr = ref.cam.f;
    for (let k = viewStart[r]; k < viewStart[r + 1]; k++) {
      const p = viewPts[k];
      const X = xyz[3 * p];
      const Y = xyz[3 * p + 1];
      const Z = xyz[3 * p + 2];
      const ax = crx - X;
      const ay = cry - Y;
      const az = crz - Z;
      const da = Math.hypot(ax, ay, az);
      if (!(da > 0)) continue;
      for (let q = visStart[p]; q < visStart[p + 1]; q++) {
        const j = vis[q];
        if (j === r) continue;
        const bx = centres[3 * j] - X;
        const by = centres[3 * j + 1] - Y;
        const bz = centres[3 * j + 2] - Z;
        const db = Math.hypot(bx, by, bz);
        if (!(db > 0)) continue;
        const cx = ay * bz - az * by;
        const cy = az * bx - ax * bz;
        const cz = ax * by - ay * bx;
        const theta = Math.atan2(Math.hypot(cx, cy, cz), ax * bx + ay * by + az * bz);
        const rho = (da / fr) / (db / views[j].cam.f);
        score[j] += triangulationAngleWeight(theta, lo, hi) * scaleWeight(rho);
        shared[j]++;
        angles[j].push(theta);
        dirSum[3 * j] += bx / db;
        dirSum[3 * j + 1] += by / db;
        dirSum[3 * j + 2] += bz / db;
      }
    }

    // candidates
    const cand: number[] = [];
    for (let j = 0; j < nv; j++) {
      if (j === r || shared[j] < o.minSharedPoints || !(score[j] > 0)) continue;
      if (quantile(angles[j], 0.75) < hardMin) continue;
      cand.push(j);
    }
    // unit mean viewing directions for diversity
    const dirs = new Map<number, [number, number, number]>();
    for (const j of cand) {
      const n = Math.hypot(dirSum[3 * j], dirSum[3 * j + 1], dirSum[3 * j + 2]) || 1;
      dirs.set(j, [dirSum[3 * j] / n, dirSum[3 * j + 1] / n, dirSum[3 * j + 2] / n]);
    }
    const eff = new Map<number, number>();
    for (const j of cand) eff.set(j, score[j]);
    const selected: number[] = [];
    const selScores: number[] = [];
    const divRad = Math.max(1e-6, o.diversityAngleDeg * DEG);
    while (selected.length < o.numSources && eff.size > 0) {
      let best = -1;
      let bestS = -Infinity;
      for (const [j, s] of eff) {
        if (s > bestS || (s === bestS && j < best)) {
          bestS = s;
          best = j;
        }
      }
      if (best < 0 || !(bestS > 0)) break;
      selected.push(best);
      selScores.push(bestS);
      eff.delete(best);
      const db = dirs.get(best)!;
      for (const [j, s] of eff) {
        const dj = dirs.get(j)!;
        const ang = Math.acos(Math.max(-1, Math.min(1, dj[0] * db[0] + dj[1] * db[1] + dj[2] * db[2])));
        if (ang < divRad) eff.set(j, s * (0.5 + 0.5 * (ang / divRad)));
      }
    }

    // depth samples of points visible in the reference
    const R = ref.pose.R;
    const t = ref.pose.t;
    const depths: number[] = [];
    for (let k = viewStart[r]; k < viewStart[r + 1]; k++) {
      const p = viewPts[k];
      const z = R[6] * xyz[3 * p] + R[7] * xyz[3 * p + 1] + R[8] * xyz[3 * p + 2] + t[2];
      if (z > 0 && Number.isFinite(z)) depths.push(z);
    }
    const frustumPts = (): number[] => {
      // all points projecting into the reference image in front of it
      const out: number[] = [];
      const c = ref.cam;
      for (let p = 0; p < np; p++) {
        const X = xyz[3 * p];
        const Y = xyz[3 * p + 1];
        const Z = xyz[3 * p + 2];
        const z = R[6] * X + R[7] * Y + R[8] * Z + t[2];
        if (!(z > 0)) continue;
        const u = (c.f * (R[0] * X + R[1] * Y + R[2] * Z + t[0])) / z + c.cx;
        const v = (c.f * (R[3] * X + R[4] * Y + R[5] * Z + t[1])) / z + c.cy;
        if (u >= -0.5 && u <= c.width - 0.5 && v >= -0.5 && v <= c.height - 0.5) out.push(p);
      }
      return out;
    };
    let rangePts: number[] | null = null;
    if (depths.length < o.minPointsForRange) {
      // too few observed points: ADD every point projecting into the image (observed points
      // stay: their pinhole projection may fall outside the frame for a distorted camera)
      rangePts = frustumPts();
      const observed = new Set<number>();
      for (let k = viewStart[r]; k < viewStart[r + 1]; k++) observed.add(viewPts[k]);
      for (const p of rangePts) {
        if (observed.has(p)) continue;
        depths.push(R[6] * xyz[3 * p] + R[7] * xyz[3 * p + 1] + R[8] * xyz[3 * p + 2] + t[2]);
      }
    }

    // geometric fallback: views whose frustum contains the reference's points
    if (selected.length < o.minSources) {
      const pts: number[] = [];
      if (viewStart[r + 1] - viewStart[r] >= o.minPointsForRange) {
        for (let k = viewStart[r]; k < viewStart[r + 1]; k++) pts.push(viewPts[k]);
      } else {
        for (const p of rangePts ?? frustumPts()) pts.push(p);
      }
      const fb: { j: number; s: number }[] = [];
      for (let j = 0; j < nv; j++) {
        if (j === r || selected.includes(j)) continue;
        const v = views[j];
        const Rj = v.pose.R;
        const tj = v.pose.t;
        const cj = v.cam;
        let s = 0;
        const angs: number[] = [];
        for (const p of pts) {
          const X = xyz[3 * p];
          const Y = xyz[3 * p + 1];
          const Z = xyz[3 * p + 2];
          const z = Rj[6] * X + Rj[7] * Y + Rj[8] * Z + tj[2];
          if (!(z > 0)) continue;
          const u = (cj.f * (Rj[0] * X + Rj[1] * Y + Rj[2] * Z + tj[0])) / z + cj.cx;
          const vv = (cj.f * (Rj[3] * X + Rj[4] * Y + Rj[5] * Z + tj[1])) / z + cj.cy;
          if (!(u >= -0.5 && u <= cj.width - 0.5 && vv >= -0.5 && vv <= cj.height - 0.5)) continue;
          const ax = crx - X;
          const ay = cry - Y;
          const az = crz - Z;
          const bx = centres[3 * j] - X;
          const by = centres[3 * j + 1] - Y;
          const bz = centres[3 * j + 2] - Z;
          const da = Math.hypot(ax, ay, az);
          const db = Math.hypot(bx, by, bz);
          const theta = Math.atan2(Math.hypot(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), ax * bx + ay * by + az * bz);
          angs.push(theta);
          s += triangulationAngleWeight(theta, lo, hi) * scaleWeight((da / fr) / (db / cj.f));
        }
        if (s > 0 && angs.length > 0 && quantile(angs, 0.75) >= hardMin) fb.push({ j, s });
      }
      fb.sort((a, b) => b.s - a.s || a.j - b.j);
      for (const { j, s } of fb) {
        if (selected.length >= o.minSources) break;
        selected.push(j);
        selScores.push(s);
      }
    }

    // depth range
    let dMin: number;
    let dMax: number;
    if (depths.length > 0) {
      const pLo = quantile(depths, o.depthPercentileLow / 100);
      const pHi = quantile(depths, o.depthPercentileHigh / 100);
      dMin = pLo * o.depthMarginMin;
      dMax = pHi * o.depthMarginMax;
    } else {
      // no structure at all: derive from baselines to the sources / nearest cameras
      const dists: number[] = [];
      const pool = selected.length > 0 ? selected : [...Array(nv).keys()].filter((j) => j !== r);
      for (const j of pool) {
        dists.push(Math.hypot(centres[3 * j] - crx, centres[3 * j + 1] - cry, centres[3 * j + 2] - crz));
      }
      const b = dists.length > 0 ? quantile(dists, 0.5) : 1;
      const base = b > 0 ? b : 1;
      dMin = 0.5 * base;
      dMax = 200 * base;
    }
    if (!(dMax > 0)) dMax = 1;
    if (!(dMin > 0) || dMin >= dMax) dMin = dMax * 1e-3;
    result.set(ref.id, { sources: selected.map((j) => views[j].id), depthMin: dMin, depthMax: dMax, scores: selScores });
  }
  return result;
}

/** Neighbour lists (view id → source ids) for `fuseDepthMaps` from a selection result. */
export function neighboursFromSelection(sel: Map<number, SourceViewSelection>, symmetric = true): Map<number, number[]> {
  const out = new Map<number, number[]>();
  for (const [id, s] of sel) out.set(id, [...s.sources]);
  if (symmetric) {
    for (const [id, s] of sel) {
      for (const j of s.sources) {
        const l = out.get(j);
        if (l && !l.includes(id)) l.push(id);
      }
    }
  }
  return out;
}
