/**
 * Automatic object box of an object-mode capture (ARCHITECTURE §4.6, §5 "Dense"): where the
 * object is, from the camera orbit and the sparse points, in the gravity-aligned world frame
 * (+Z up). Pure CPU and deterministic; it needs only the sparse model, so it works before dense.
 * runMesh (src/engine/dense.ts) passes the box to the TSDF as `TsdfOptions.bounds`.
 *
 *  1. Orbit: the optical-axis convergence point P (src/geometry/convergence.ts) must exist and
 *     pass the orbit test. The orbit axis is the vertical line through P, `ringRadius` the median
 *     horizontal camera distance from it (≥ 0.25 d, else the cameras look straight down), and d the
 *     median camera distance to P. d scales every threshold below, so non-metric models work too.
 *  2. Support plane: the sparse points in the annulus 0.6–1.5 × ringRadius around the axis with
 *     z in [P.z − 1.0 d, P.z + 0.1 d]. The annulus lies outside the object's footprint, so the top
 *     of a flat object (a book) is never taken for the plane, and mostly inside the camera ring,
 *     so the room is not either. z-histogram (bin 0.004 d ≈ 2 mm at 0.5 m, [1 2 1] smoothed) →
 *     peak → least-squares plane z = a x + b y + c over the ±3-bin band, refitted 3× on the
 *     ±3-bin residual band (a gravity error of a few degrees still gives one plane). Rejected
 *     with fewer than 30 points, under 20 % of the annulus points (the peak's share; a peak /
 *     median-bin ratio is undefined on a sparse histogram whose median bin is empty) or a tilt
 *     above 2° (see MAX_TILT).
 *  3. Object: the points within 0.6 × ringRadius of the axis, above the plane + clearance
 *     (max(3σ of the plane residuals, `minClearance`; runMesh passes 2 TSDF voxels at d)) and below
 *     P.z + 1.5 d. Two filters drop the points that would stretch the box: weak points (seen in
 *     < 3 images, when the input has tracks) and isolated ones (6th-nearest object neighbour
 *     farther than max(3 × its median, 0.02 d): floaters, and the thin noise of a placemat or the
 *     table just above the clearance, which otherwise spreads the box over the whole 0.6 × ring
 *     disc). No connectivity test: the parts of a multi-part object (or an object and its stand)
 *     connect only through the plane. At least 100 points must remain, and the lowest (2nd
 *     percentile) must lie within 0.25 d of the plane, else the object does not rest on it
 *     (hand-held, hanging).
 *  4. Box: 0.2 / 99.8 percentiles per axis + a max(12 % of the extent, 0.03 d) margin (sparse
 *     points cover the extremes thinly; see MARGIN). The bottom is flat at the plane's highest
 *     point under the box (a bottom corner; the plane may be tilted by up to 2°) + clearance, so
 *     the support stays out of the box; `planeZ` is the plane height under the box centre.
 *     confidence = the plane's share of the annulus points × min(1, object points / 200); below
 *     `minConfidence` (default 0.3) there is no box.
 *
 * The result is null (`reason` says why) when the model is not gravity-aligned, the cameras do
 * not orbit a point (axes spread < 10°, orbit test failed, cameras look straight down), there is
 * no plane, the object does not rest on it, or the confidence is low. Callers then keep the whole
 * scene; runMesh keeps the 2 × median depth cut alone.
 *
 * Limits: the box is axis-aligned in the world frame, so its yaw is the world's (north when
 * known). Objects wider than ~0.6 × the camera ring are trimmed, and clutter within that radius
 * is kept. White, glossy or glass supports give few sparse points, hence no plane and no box.
 * On a tilted plane the flat bottom cuts the low side of the object's base (see MAX_TILT).
 * Validated on synthetic data only: object-roi.test.ts, mesh-roi.test.ts, a 40-frame e2e 'object'
 * run (worker engine, 240 px dense, the box before the point filters): no mesh vertex outside the
 * GT object box ± 2 cm (61 % unbounded), its completeness within 0.004 of the unbounded mesh; and
 * offline on the stored sparse models of 6 object-real runs (7–12 k points; sparse.ply has no
 * tracks, so without the track filter) and of that e2e run, at the clearances of 240, 800 (fast)
 * and 1280 px (balanced) dense: the box contains meta.focus − 1 cm at IoU 0.70–0.79 (object-real
 * at fast / balanced 0.73–0.77), unchanged with 1–3 % floaters added in the object cylinder.
 * Without the isolation filter: IoU 0.31–0.39 at fast / balanced (placemat noise above the 3.4 mm
 * clearance spread the box over the 0.6 × ring disc), 0.13–0.18 with 1–3 % floaters.
 */
import type { ObjectRoi, Pose } from '../core/types';
import { opticalAxisConvergence } from '../geometry/convergence';
import { solveSPD } from '../math/decomp';
import { mad, median, percentile } from '../math/stats';
import { KdTree3 } from './kdtree';

/** What the proposal reads: the registered cameras and the sparse points (a SparseModel fits). */
export interface ObjectRoiInput {
  images: readonly { pose: Pose }[];
  /** 3 per point, world frame. */
  xyz: ArrayLike<number>;
  numPoints: number;
  /**
   * CSR track offsets (`SparseModel.trackOffsets`: point i is seen in trackOffsets[i + 1] −
   * trackOffsets[i] images). Given, object points seen in fewer than 3 images are dropped.
   */
  trackOffsets?: ArrayLike<number>;
}

export interface ObjectRoiOptions {
  /** World +Z is up (`MetricInfo.gravityAligned`). Without it there is no box. */
  gravityAligned: boolean;
  /**
   * Lower bound of the box bottom's clearance above the plane: world units, or `perDistance` × d
   * (runMesh: 2 TSDF voxels, i.e. 2 · autoVoxelPixels / focal px per unit of the object distance d,
   * or 2 × a given voxel size). Default 0.008 d (2 histogram bins).
   */
  minClearance?: number | { perDistance: number };
  /** No box below this confidence (default 0.3, see MIN_CONFIDENCE). */
  minConfidence?: number;
}

/** Result of `proposeObjectRoi`: the box or why there is none, and the numbers behind it. */
export interface ObjectRoiProposal {
  roi: ObjectRoi | null;
  /** Why `roi` is null ('' when it is not). */
  reason: string;
  /** Median camera distance to the orbit centre (the length scale d; 0 without one). */
  distMedian: number;
  /** Median horizontal camera distance from the orbit axis (0 without an orbit). */
  ringRadius: number;
  /** Points of the annulus (plane candidates) and the plane's inliers among them. */
  annulusPoints: number;
  planePoints: number;
  /** Plane tilt from horizontal (rad) and robust σ of its residuals (NaN without a plane). */
  planeTiltRad: number;
  planeSigma: number;
  /** Plane slopes (a, b) of z = a x + b y + c (NaN without a plane). */
  planeSlope: [number, number];
  /** Clearance of the box bottom above the plane (NaN without a plane). */
  clearance: number;
  /** Object points the box is taken from (after the two filters below). */
  objectPoints: number;
  /** Object candidates dropped as weak (seen in < 3 images; 0 without tracks). */
  weakPoints: number;
  /** Object candidates dropped as isolated (floaters, near-plane noise). */
  isolatedPoints: number;
  /** The box's confidence, also when it is below `minConfidence` (0 before the box step). */
  confidence: number;
}

const DEG = Math.PI / 180;
/** Plane candidates: horizontal distance from the orbit axis in [0.6, 1.5] × ringRadius. */
const ANNULUS_INNER = 0.6;
const ANNULUS_OUTER = 1.5;
/** Plane search range below / above the orbit centre (× d). */
const PLANE_BELOW = 1.0;
const PLANE_ABOVE = 0.1;
/** Histogram bin (× d) and the inlier band half-width (bins). */
const BIN = 0.004;
const BAND_BINS = 3;
const PLANE_REFITS = 3;
const MIN_PLANE_POINTS = 30;
const MIN_PLANE_FRACTION = 0.2;
/**
 * The box bottom is flat, so a tilted plane costs the low side of the object's base 2 (|a| hx +
 * |b| hy) of height (≈ 1 cm for a 15 cm half-width at 2°); beyond that there is no box.
 */
const MAX_TILT = 2 * DEG;
/** Object points: within 0.6 × ringRadius of the axis, below P.z + 1.5 d. */
const OBJECT_RADIUS = 0.6;
const OBJECT_ABOVE = 1.5;
/** Object points need a track of ≥ 3 images (two-view points are the usual SfM floaters). */
const MIN_TRACK = 3;
/**
 * Isolation filter: a point's distance to its 6th-nearest object neighbour must be ≤ max(3 × the
 * median of that distance, 0.02 d). Floaters (alone or in clusters of up to 6), and the thin noise
 * of a placemat or the table just above the clearance, lie much farther from their neighbours
 * than surface samples of the object. On the 6 object-real sparse models of the header it drops
 * 3–4 % of the 2.3–4.6 k candidates; a 4× or 5× limit let the placemat noise through (IoU down to
 * 0.53), 4 neighbours instead of 6 gave the same boxes.
 */
const ISOLATION_K = 6;
const ISOLATION_FACTOR = 3;
const ISOLATION_MIN = 0.02;
/** Object points needed after both filters (fewer: a box from a handful of points). */
const MIN_OBJECT_POINTS = 100;
/** No box below this confidence: a weak plane or few object points crop the mesh wrongly. */
const MIN_CONFIDENCE = 0.3;
/** The 2nd-percentile object point lies at most 0.25 d above the plane. */
const MAX_GAP = 0.25;
/**
 * Box: percentiles 0.2 / 99.8 per axis + max(12 % of the extent, 0.03 d). Sparse points cover an
 * object's extremes thinly: on a 40-frame e2e object run (856 object points) 0.5 / 99.5 + max(5 %,
 * 0.02 d) left the box 0.9–1.3 cm inside the GT object box (79 % of it covered); 10 % covered
 * 94.5 % (the rest is the bottom cut) at IoU 0.78, but once the isolation filter drops the odd
 * extreme point it no longer contains the GT box − 1 cm there; 12 % does, on that run and the 6
 * object-real models (IoU 0.73–0.79 there).
 */
const PCT_LO = 0.2;
const PCT_HI = 99.8;
const MARGIN = 0.12;
const MIN_MARGIN = 0.03;
/** Cameras closer than 0.25 d to the axis (horizontally) look straight down: no ring. */
const MIN_RING = 0.25;
/** Object points at which the confidence stops growing. */
const FULL_SUPPORT_POINTS = 200;

/** Propose the object box of an orbit capture (see the file header). */
export function proposeObjectRoi(model: ObjectRoiInput, opts: ObjectRoiOptions): ObjectRoiProposal {
  const out: ObjectRoiProposal = {
    roi: null,
    reason: '',
    distMedian: 0,
    ringRadius: 0,
    annulusPoints: 0,
    planePoints: 0,
    planeTiltRad: NaN,
    planeSigma: NaN,
    planeSlope: [NaN, NaN],
    clearance: NaN,
    objectPoints: 0,
    weakPoints: 0,
    isolatedPoints: 0,
    confidence: 0,
  };
  const fail = (reason: string): ObjectRoiProposal => {
    out.reason = reason;
    return out;
  };
  if (!opts.gravityAligned) return fail('model not gravity-aligned');
  const poses = model.images.map((im) => im.pose);
  if (poses.length < 2) return fail('fewer than 2 cameras');
  const conv = opticalAxisConvergence(poses);
  if (!conv.point) return fail('optical axes do not converge (spread < 10°)');
  if (!conv.converging) return fail('cameras do not orbit a common point');
  const P = conv.point;
  const d = conv.distMedian;
  out.distMedian = d;
  const radii: number[] = [];
  for (const { R, t } of poses) {
    // camera centre C = −Rᵀ t, horizontal part relative to the axis
    const cx = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]) - P[0];
    const cy = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]) - P[1];
    radii.push(Math.hypot(cx, cy));
  }
  const ring = median(radii);
  out.ringRadius = ring;
  if (!(d > 0) || !(ring >= MIN_RING * d)) return fail('cameras look straight down (no camera ring)');

  // ---- 2. support plane
  const X = model.xyz;
  const n = Math.min(model.numPoints, Math.floor(X.length / 3));
  const bin = BIN * d;
  const r0 = ANNULUS_INNER * ring;
  const r1 = ANNULUS_OUTER * ring;
  const zLo = P[2] - PLANE_BELOW * d;
  const zHi = P[2] + PLANE_ABOVE * d;
  const cand: number[] = [];
  for (let i = 0; i < n; i++) {
    const z = X[3 * i + 2];
    if (!(z >= zLo && z <= zHi)) continue;
    const r = Math.hypot(X[3 * i] - P[0], X[3 * i + 1] - P[1]);
    if (r >= r0 && r <= r1) cand.push(i);
  }
  out.annulusPoints = cand.length;
  if (cand.length < MIN_PLANE_POINTS) return fail(`no support plane (${cand.length} points around the object)`);
  const nb = Math.floor((zHi - zLo) / bin) + 1;
  const hist = new Float64Array(nb);
  for (const i of cand) hist[Math.min(nb - 1, Math.floor((X[3 * i + 2] - zLo) / bin))]++;
  let peak = 0;
  let peakScore = -1;
  for (let b = 0; b < nb; b++) {
    const s = (b > 0 ? hist[b - 1] : 0) + 2 * hist[b] + (b + 1 < nb ? hist[b + 1] : 0);
    if (s > peakScore) {
      peakScore = s;
      peak = b;
    }
  }
  // plane z = pa·x + pb·y + pc in coordinates relative to the axis
  let pa = 0;
  let pb = 0;
  let pc = zLo + (peak + 0.5) * bin;
  const band = BAND_BINS * bin;
  const residual = (i: number): number => X[3 * i + 2] - (pa * (X[3 * i] - P[0]) + pb * (X[3 * i + 1] - P[1]) + pc);
  let inliers = cand.filter((i) => Math.abs(residual(i)) <= band);
  for (let it = 0; it < PLANE_REFITS && inliers.length >= 3; it++) {
    const A = new Float64Array(9);
    const rhs = new Float64Array(3);
    for (const i of inliers) {
      const x = X[3 * i] - P[0];
      const y = X[3 * i + 1] - P[1];
      const z = X[3 * i + 2];
      A[0] += x * x;
      A[1] += x * y;
      A[2] += x;
      A[4] += y * y;
      A[5] += y;
      A[8] += 1;
      rhs[0] += x * z;
      rhs[1] += y * z;
      rhs[2] += z;
    }
    A[3] = A[1];
    A[6] = A[2];
    A[7] = A[5];
    const sol = solveSPD(A, 3, rhs);
    if (sol && sol.every(Number.isFinite)) {
      pa = sol[0];
      pb = sol[1];
      pc = sol[2];
    } else {
      // inliers on a line (a strip of table): keep the plane horizontal
      pa = 0;
      pb = 0;
      pc = median(inliers.map((i) => X[3 * i + 2]));
    }
    inliers = cand.filter((i) => Math.abs(residual(i)) <= band);
  }
  out.planePoints = inliers.length;
  const tilt = Math.atan(Math.hypot(pa, pb));
  out.planeTiltRad = tilt;
  out.planeSlope = [pa, pb];
  const sigma = 1.4826 * mad(inliers.map(residual));
  out.planeSigma = sigma;
  if (inliers.length < MIN_PLANE_POINTS || inliers.length < MIN_PLANE_FRACTION * cand.length) {
    return fail(`no support plane (${inliers.length} of ${cand.length} points around the object on the best plane)`);
  }
  if (!(tilt <= MAX_TILT)) return fail(`support plane tilted ${(tilt / DEG).toFixed(1)}°`);
  const mc = opts.minClearance;
  const minClearance = mc === undefined ? 2 * bin : typeof mc === 'number' ? mc : mc.perDistance * d;
  const clearance = Math.max(3 * (sigma || 0), minClearance >= 0 ? minClearance : 0);
  out.clearance = clearance;

  // ---- 3. object points
  const ro = OBJECT_RADIUS * ring;
  const zTop = P[2] + OBJECT_ABOVE * d;
  const tracks = model.trackOffsets && model.trackOffsets.length > n ? model.trackOffsets : null;
  const objCand: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = X[3 * i] - P[0];
    const y = X[3 * i + 1] - P[1];
    const z = X[3 * i + 2];
    if (!(Math.hypot(x, y) <= ro && z <= zTop)) continue;
    if (!(z - (pa * x + pb * y + pc) > clearance)) continue;
    if (tracks && tracks[i + 1] - tracks[i] < MIN_TRACK) {
      out.weakPoints++;
      continue;
    }
    objCand.push(i);
  }
  const keep = isolationMask(X, objCand, P, ISOLATION_MIN * d);
  const ox: number[] = [];
  const oy: number[] = [];
  const oz: number[] = [];
  const above: number[] = [];
  for (let j = 0; j < objCand.length; j++) {
    if (!keep[j]) {
      out.isolatedPoints++;
      continue;
    }
    const i = objCand[j];
    ox.push(X[3 * i]);
    oy.push(X[3 * i + 1]);
    oz.push(X[3 * i + 2]);
    above.push(X[3 * i + 2] - (pa * (X[3 * i] - P[0]) + pb * (X[3 * i + 1] - P[1]) + pc));
  }
  out.objectPoints = oz.length;
  if (oz.length < MIN_OBJECT_POINTS) return fail(`no object on the support plane (${oz.length} points above it)`);
  const gap = percentile(above, 2);
  if (gap > MAX_GAP * d) return fail(`object does not rest on the support plane (gap ${gap.toPrecision(3)})`);

  // ---- 4. box
  const lo = [percentile(ox, PCT_LO), percentile(oy, PCT_LO), percentile(oz, PCT_LO)];
  const hi = [percentile(ox, PCT_HI), percentile(oy, PCT_HI), percentile(oz, PCT_HI)];
  const min: [number, number, number] = [0, 0, 0];
  const max: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const m = Math.max(MARGIN * (hi[k] - lo[k]), MIN_MARGIN * d);
    min[k] = lo[k] - m;
    max[k] = hi[k] + m;
  }
  const planeZ = pa * (0.5 * (min[0] + max[0]) - P[0]) + pb * (0.5 * (min[1] + max[1]) - P[1]) + pc;
  // the plane's highest point under the box (a bottom corner) + clearance
  min[2] = planeZ + 0.5 * (Math.abs(pa) * (max[0] - min[0]) + Math.abs(pb) * (max[1] - min[1])) + clearance;
  if (!(max[2] > min[2])) return fail('empty box');
  const confidence = Math.min(1, inliers.length / cand.length) * Math.min(1, oz.length / FULL_SUPPORT_POINTS);
  out.confidence = confidence;
  const minConfidence = opts.minConfidence ?? MIN_CONFIDENCE;
  if (!(confidence >= minConfidence)) return fail(`low confidence (${confidence.toFixed(2)} < ${minConfidence})`);
  out.roi = { min, max, planeZ, centre: [P[0], P[1], P[2]], confidence };
  return out;
}

/**
 * Isolation filter of the object candidates `idx` (see ISOLATION_K): 1 = keep. Positions relative
 * to the orbit centre `P`, so Float32 keeps sub-mm precision far from the world origin.
 */
function isolationMask(X: ArrayLike<number>, idx: readonly number[], P: ArrayLike<number>, minRadius: number): Uint8Array {
  const m = idx.length;
  const keep = new Uint8Array(m).fill(1);
  if (m <= ISOLATION_K) return keep;
  const pts = new Float32Array(3 * m);
  for (let j = 0; j < m; j++) for (let k = 0; k < 3; k++) pts[3 * j + k] = X[3 * idx[j] + k] - P[k];
  const tree = new KdTree3(pts, m);
  const d2 = new Float64Array(ISOLATION_K);
  const ii = new Int32Array(ISOLATION_K);
  // distance to the K-th nearest neighbour, per input index
  const rk = new Float64Array(m);
  for (let o = 0; o < m; o++) {
    const got = tree.knnAt(o, ISOLATION_K, d2, ii);
    let r2 = 0;
    for (let j = 0; j < got; j++) r2 = Math.max(r2, d2[j]);
    rk[tree.perm[o]] = got < ISOLATION_K ? Infinity : Math.sqrt(r2);
  }
  const limit = Math.max(ISOLATION_FACTOR * median(rk), minRadius);
  for (let j = 0; j < m; j++) if (!(rk[j] <= limit)) keep[j] = 0;
  return keep;
}

/** Bottom centre of the box on the support plane: the origin of exports that stand the object on the ground. */
export function objectRoiPivot(roi: ObjectRoi): [number, number, number] {
  return [0.5 * (roi.min[0] + roi.max[0]), 0.5 * (roi.min[1] + roi.max[1]), roi.planeZ];
}
