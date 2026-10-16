/**
 * Two-view geometry verification with model selection between E, F and H, after
 * COLMAP's `EstimateCalibratedTwoViewGeometry` (Schönberger & Frahm 2016,
 * "Structure-from-Motion Revisited").
 */
import type { CameraIntrinsics, Mat3, MatchPairs, TwoViewConfig, TwoViewGeometry, Vec3 } from '../core/types';
import { orthonormalize, rotationAngleBetween } from '../math/rotation';
import { Rng } from '../math/random';
import { undistortNormalized } from './camera';
import {
  decomposeEssential,
  essential5pt,
  essentialLinearIdx,
  recoverPose,
  refineRelativePose,
  sampsonDistanceSq,
  sampsonDistancesSq,
  sampsonDistancesSqRange,
  cheiralityCount,
} from './essential';
import { median } from '../math/stats';
import { fundamental7pt, fundamental8ptIdx, fundamentalFromEssential } from './fundamental';
import {
  decomposeHomography,
  homographyDLTIdx,
  homographySampleDegenerate,
  homographyTransferErrorsSq,
  homographyTransferErrorsSqRange,
} from './homography';
import { ransac, type RansacEstimator } from './ransac';

/** Options of `estimateTwoViewGeometry`. */
export interface TwoViewOptions {
  /** Inlier threshold in pixels (default 4). */
  maxErrorPx?: number;
  /** Minimum number of inliers of the selected model (default 15). */
  minNumInliers?: number;
  /** Minimum inlier ratio of the selected model; also caps RANSAC iterations (default 0.25). */
  minInlierRatio?: number;
  /** RANSAC confidence (default 0.999, as COLMAP's two-view verification). */
  confidence?: number;
  /** RANSAC iteration cap per model (default 10000). */
  maxIterations?: number;
  rng?: Rng;
  /**
   * Relative rotation R_2←1 (e.g. integrated gyro), used ONLY to help detect pure
   * rotation: (1) the homography K2 R K1⁻¹ seeds the H RANSAC; (2) a rotation-only
   * model refined from the prior (Kabsch on bearings) that explains more than
   * `maxHInlierRatio` of the inliers of a planar/panoramic pair classifies it as
   * 'panoramic'. It never changes the estimated E, F, H, R or t otherwise.
   */
  rotationPrior?: Mat3;
  /** Extension: E is preferred if E_inl / F_inl ≥ this (default 0.95). */
  minEFInlierRatio?: number;
  /** Extension: planar/panoramic if H_inl / (E or F)_inl > this (default 0.8). */
  maxHInlierRatio?: number;
  /** Extension: median triangulation angle below which a planar/panoramic pair is panoramic (default 0.5°). */
  minTriAngleDeg?: number;
}

/** Extended result: the contract's TwoViewGeometry plus diagnostics. */
export interface TwoViewGeometryResult extends TwoViewGeometry {
  numInliersE: number;
  numInliersF: number;
  numInliersH: number;
  /**
   * 'planar' only: the other relative pose that explains a planar scene equally well
   * (two-view planar ambiguity, Longuet-Higgins 1986; Faugeras & Lustman 1988). It
   * can only be resolved with a third view or a prior on the plane normal. Absent when
   * cheirality already resolves the ambiguity (the other branch of the homography
   * decomposition has < 90 % of the best branch's points in front of both cameras).
   */
  planarAlternative?: { R: Mat3; t: Vec3 };
}

const DEFAULT_SEED = 0x7a0b1e5;

/**
 * 'panoramic' pairs whose full-model pose has at least this fraction of the inliers in
 * front of both cameras report the full model's rotation instead of a rotation-only
 * fit (pure rotation measures ≤ 0.52, small observed translations 0.87–1.0).
 */
const PANORAMIC_FULL_MODEL_FRONT_RATIO = 0.8;

// ---------------------------------------------------------------------------
// Estimators
// ---------------------------------------------------------------------------

/** LO-MSAC estimator for E on normalised coordinates (5-point, Sampson residual, linear 8-point refit). */
export class EssentialEstimator implements RansacEstimator<Mat3> {
  sampleSize = 5;
  numData: number;
  minRefitSize = 8;
  private s1 = new Float64Array(10);
  private s2 = new Float64Array(10);
  constructor(
    private x1: Float64Array,
    private x2: Float64Array,
    n: number,
  ) {
    this.numData = n;
  }
  fit(sample: Uint32Array, models: Mat3[]): number {
    for (let k = 0; k < 5; k++) {
      const i = sample[k];
      this.s1[2 * k] = this.x1[2 * i];
      this.s1[2 * k + 1] = this.x1[2 * i + 1];
      this.s2[2 * k] = this.x2[2 * i];
      this.s2[2 * k + 1] = this.x2[2 * i + 1];
    }
    const sols = essential5pt(this.s1, this.s2);
    for (let k = 0; k < sols.length; k++) models[k] = sols[k];
    return sols.length;
  }
  residualsSq(E: Mat3, out: Float64Array): void {
    sampsonDistancesSq(E, this.x1, this.x2, this.numData, out);
  }
  residualsSqRange(E: Mat3, out: Float64Array, start: number, end: number): void {
    sampsonDistancesSqRange(E, this.x1, this.x2, start, end, out);
  }
  refit(inliers: Uint32Array, count: number): Mat3 | null {
    return essentialLinearIdx(this.x1, this.x2, inliers, count);
  }
}

/** LO-MSAC estimator for F on pixels (7-point, Sampson residual, normalised 8-point refit). */
export class FundamentalEstimator implements RansacEstimator<Mat3> {
  sampleSize = 7;
  numData: number;
  minRefitSize = 8;
  private s1 = new Float64Array(14);
  private s2 = new Float64Array(14);
  constructor(
    private p1: Float64Array,
    private p2: Float64Array,
    n: number,
  ) {
    this.numData = n;
  }
  fit(sample: Uint32Array, models: Mat3[]): number {
    for (let k = 0; k < 7; k++) {
      const i = sample[k];
      this.s1[2 * k] = this.p1[2 * i];
      this.s1[2 * k + 1] = this.p1[2 * i + 1];
      this.s2[2 * k] = this.p2[2 * i];
      this.s2[2 * k + 1] = this.p2[2 * i + 1];
    }
    const sols = fundamental7pt(this.s1, this.s2);
    for (let k = 0; k < sols.length; k++) models[k] = sols[k];
    return sols.length;
  }
  residualsSq(F: Mat3, out: Float64Array): void {
    sampsonDistancesSq(F, this.p1, this.p2, this.numData, out);
  }
  residualsSqRange(F: Mat3, out: Float64Array, start: number, end: number): void {
    sampsonDistancesSqRange(F, this.p1, this.p2, start, end, out);
  }
  refit(inliers: Uint32Array, count: number): Mat3 | null {
    return fundamental8ptIdx(this.p1, this.p2, inliers, count);
  }
}

/** LO-MSAC estimator for H (4-point DLT, one-sided transfer error, DLT refit). */
export class HomographyEstimator implements RansacEstimator<Mat3> {
  sampleSize = 4;
  numData: number;
  minRefitSize = 4;
  constructor(
    private p1: Float64Array,
    private p2: Float64Array,
    n: number,
  ) {
    this.numData = n;
  }
  isDegenerate(sample: Uint32Array): boolean {
    return homographySampleDegenerate(this.p1, this.p2, sample);
  }
  fit(sample: Uint32Array, models: Mat3[]): number {
    const H = homographyDLTIdx(this.p1, this.p2, sample, 4);
    if (!H) return 0;
    models[0] = H;
    return 1;
  }
  residualsSq(H: Mat3, out: Float64Array): void {
    homographyTransferErrorsSq(H, this.p1, this.p2, this.numData, out);
  }
  residualsSqRange(H: Mat3, out: Float64Array, start: number, end: number): void {
    homographyTransferErrorsSqRange(H, this.p1, this.p2, start, end, out);
  }
  refit(inliers: Uint32Array, count: number): Mat3 | null {
    return homographyDLTIdx(this.p1, this.p2, inliers, count);
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function countMask(m: Uint8Array): number {
  let c = 0;
  for (let i = 0; i < m.length; i++) c += m[i];
  return c;
}

/** Kabsch / Wahba rotation R minimising Σ ‖b2 − R b1‖² over bearings of the masked correspondences. */
export function rotationFromCorrespondences(
  x1: ArrayLike<number>,
  x2: ArrayLike<number>,
  n: number,
  mask: Uint8Array | null,
): Mat3 | null {
  const B = new Float64Array(9);
  let c = 0;
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    const a0 = x1[2 * i],
      a1 = x1[2 * i + 1];
    const b0 = x2[2 * i],
      b1 = x2[2 * i + 1];
    const ia = 1 / Math.sqrt(a0 * a0 + a1 * a1 + 1),
      ib = 1 / Math.sqrt(b0 * b0 + b1 * b1 + 1);
    const u0 = a0 * ia,
      u1 = a1 * ia,
      u2 = ia;
    const v0 = b0 * ib,
      v1 = b1 * ib,
      v2 = ib;
    B[0] += v0 * u0;
    B[1] += v0 * u1;
    B[2] += v0 * u2;
    B[3] += v1 * u0;
    B[4] += v1 * u1;
    B[5] += v1 * u2;
    B[6] += v2 * u0;
    B[7] += v2 * u1;
    B[8] += v2 * u2;
    c++;
  }
  if (c < 2) return null;
  const R = orthonormalize(B);
  return Number.isFinite(R[0]) ? R : null;
}

/** Squared pixel error of the rotation-only model x2 ~ K2 R K1⁻¹ x1 (normalised inputs). */
function rotationResidualsSq(
  R: ArrayLike<number>,
  x1: Float64Array,
  x2: Float64Array,
  n: number,
  f2: number,
  out: Float64Array,
): void {
  for (let i = 0; i < n; i++) {
    const a = x1[2 * i],
      b = x1[2 * i + 1];
    const z = R[6] * a + R[7] * b + R[8];
    if (!(z > 0)) {
      out[i] = Infinity;
      continue;
    }
    const du = ((R[0] * a + R[1] * b + R[2]) / z - x2[2 * i]) * f2;
    const dv = ((R[3] * a + R[4] * b + R[5]) / z - x2[2 * i + 1]) * f2;
    out[i] = du * du + dv * dv;
  }
}

/** Median angle (rad) between the rays x1 and Rᵀ x2 (cam-1 frame) over the masked correspondences. */
function medianRayAngle(R: ArrayLike<number>, x1: Float64Array, x2: Float64Array, n: number, mask: Uint8Array): number {
  const angles = new Float64Array(countMask(mask));
  let c = 0;
  for (let i = 0; i < n; i++) {
    if (!mask[i]) continue;
    const a0 = x1[2 * i],
      a1 = x1[2 * i + 1],
      a2 = 1;
    const b0 = x2[2 * i],
      b1 = x2[2 * i + 1];
    // d2 = Rᵀ (b0, b1, 1)
    const d0 = R[0] * b0 + R[3] * b1 + R[6],
      d1 = R[1] * b0 + R[4] * b1 + R[7],
      d2 = R[2] * b0 + R[5] * b1 + R[8];
    const cx = a1 * d2 - a2 * d1,
      cy = a2 * d0 - a0 * d2,
      cz = a0 * d1 - a1 * d0;
    angles[c++] = Math.atan2(Math.sqrt(cx * cx + cy * cy + cz * cz), a0 * d0 + a1 * d1 + a2 * d2);
  }
  if (c === 0) return 0;
  const m = median(c === angles.length ? angles : angles.subarray(0, c));
  return Number.isFinite(m) ? m : 0;
}

/**
 * True if the masked correspondences cannot support a two-view model: in either image
 * the points lie within ±minStd (px, RMS) of a single line or point (minor-axis standard
 * deviation of their 2-D scatter), or there are fewer than `minDistinct` distinct
 * correspondences (positions quantised to 1 px in both images).
 */
function supportDegenerate(p1: Float64Array, p2: Float64Array, mask: Uint8Array, minStd: number, minDistinct: number): boolean {
  for (let img = 0; img < 2; img++) {
    const p = img === 0 ? p1 : p2;
    let c = 0,
      mx = 0,
      my = 0;
    for (let i = 0; i < mask.length; i++) {
      if (!mask[i]) continue;
      mx += p[2 * i];
      my += p[2 * i + 1];
      c++;
    }
    if (c === 0) return true;
    mx /= c;
    my /= c;
    let sxx = 0,
      sxy = 0,
      syy = 0;
    for (let i = 0; i < mask.length; i++) {
      if (!mask[i]) continue;
      const dx = p[2 * i] - mx,
        dy = p[2 * i + 1] - my;
      sxx += dx * dx;
      sxy += dx * dy;
      syy += dy * dy;
    }
    sxx /= c;
    sxy /= c;
    syy /= c;
    const h = 0.5 * (sxx - syy);
    const lmin = 0.5 * (sxx + syy) - Math.sqrt(h * h + sxy * sxy);
    if (!(lmin >= minStd * minStd)) return true;
  }
  // Distinct correspondences (duplicates arise e.g. from multi-orientation keypoints):
  // the smaller of the numbers of distinct 1-px cells occupied in image 1 and image 2.
  return Math.min(distinctCells(p1, mask), distinctCells(p2, mask)) < minDistinct;
}

/** Number of distinct 1-px cells (rounded coordinates) occupied by the masked points. */
function distinctCells(p: Float64Array, mask: Uint8Array): number {
  const keys = new Float64Array(countMask(mask));
  let c = 0;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    // exact integer keys for |coordinates| < 2^20 px (larger values only merge cells)
    keys[c++] = (Math.round(p[2 * i]) + 1048576) * 4194304 + (Math.round(p[2 * i + 1]) + 1048576);
  }
  keys.sort();
  let d = 0;
  for (let i = 0; i < c; i++) if (i === 0 || keys[i] !== keys[i - 1]) d++;
  return d;
}

function extractInliers(matches: MatchPairs, mask: Uint8Array): MatchPairs {
  const c = countMask(mask);
  const out = new Uint32Array(2 * c);
  let k = 0;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    out[2 * k] = matches[2 * i];
    out[2 * k + 1] = matches[2 * i + 1];
    k++;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Main entry
// ---------------------------------------------------------------------------

/**
 * Verifies putative matches between two keyframes and classifies the pair
 * (COLMAP-style model selection):
 *
 * 1. Keypoints → undistorted normalised coordinates (E) and undistorted pixels of the
 *    pinhole cameras (F, H).
 * 2. E: 5-point LO-MSAC at maxErrorPx / f (mean of both cameras, Sampson error), then
 *    LM refinement of (R, t) on the inliers (Huber at the threshold, then Cauchy at
 *    the robust noise scale on the inliers in front of both cameras — see
 *    `refineEssentialInliers`), inliers recounted.
 *    F: 7-point LO-MSAC at maxErrorPx (seeded with the F implied by E).
 *    H: 4-point LO-MSAC at maxErrorPx (seeded with K2 R K1⁻¹ when a rotation prior
 *    is given; its iteration budget only needs to find an H that matters for step 3).
 *    RANSAC budgets are capped for the inlier ratio max(minInlierRatio,
 *    minNumInliers / N); confidence defaults to 0.999 (COLMAP).
 * 3. If E_inl/F_inl ≥ 0.95 and E_inl ≥ minNumInliers → (H_inl/E_inl > 0.8 ?
 *    planar_or_panoramic : calibrated); else if F_inl ≥ minNumInliers → (H_inl/F_inl
 *    > 0.8 ? planar_or_panoramic : uncalibrated); else if H_inl ≥ minNumInliers →
 *    planar_or_panoramic; else degenerate. Also degenerate when the selected model's
 *    inliers are fewer than minNumInliers or below minInlierRatio·N. The inliers are
 *    those of the selected model (E or F, or H if it has more for planar/panoramic).
 * 4. calibrated: the refined, cheirality-consistent (R, t) (|t| = 1); triAngle =
 *    median angle between the two rays of the inliers in front of both cameras.
 *    planar_or_panoramic: pose from E when E is reliable (E_inl ≥ minNumInliers and
 *    E_inl/F_inl ≥ 0.95), else from the decomposition of H — but when E's pose is an H
 *    branch with clearly fewer points in front of both cameras than the best H branch
 *    (E cannot tell the planar branches apart; cheirality can), the best H branch;
 *    resolved to 'panoramic' if the median triangulation angle is < minTriAngleDeg,
 *    translation is unobservable (fewer than half of the inliers triangulate in front,
 *    or H is a pure rotation), or the rotation prior's rotation-only model explains the
 *    inliers; otherwise 'planar', whose R, t come from the H decomposition: among the
 *    branches with maximal cheirality support (±max(2, 1 % of the H inliers)) the one
 *    closest to the selected pose; another branch with ≥ 90 % of that support is
 *    returned as `planarAlternative`. Inlier sets that are coincident / collinear
 *    within maxErrorPx in either image, or have fewer than minNumInliers distinct
 *    correspondences, are 'degenerate'. For 'panoramic', R is the rotation-only (Kabsch) fit
 *    refined on its inliers when the translation is unobserved (< 80 % of the inliers
 *    in front of both cameras for the full model's pose), else the full model's
 *    rotation (a rotation-only fit would absorb the small translation); t = (0, 0, 0),
 *    inliers = H inliers, and triAngle the median rotation-compensated ray angle.
 *    'uncalibrated' pairs get no R, t.
 */
export function estimateTwoViewGeometry(
  kp1: Float32Array,
  kp2: Float32Array,
  matches: MatchPairs,
  cam1: CameraIntrinsics,
  cam2: CameraIntrinsics,
  opts: TwoViewOptions = {},
): TwoViewGeometryResult {
  const maxErrorPx = opts.maxErrorPx ?? 4;
  const minNumInliers = opts.minNumInliers ?? 15;
  const minInlierRatio = opts.minInlierRatio ?? 0.25;
  const confidence = opts.confidence ?? 0.999;
  const maxIterations = opts.maxIterations ?? 10000;
  const minEF = opts.minEFInlierRatio ?? 0.95;
  const maxHRatio = opts.maxHInlierRatio ?? 0.8;
  const minTriAngle = ((opts.minTriAngleDeg ?? 0.5) * Math.PI) / 180;
  const rng = opts.rng ?? new Rng(DEFAULT_SEED);

  const N = matches.length >> 1;
  const degenerate = (nE = 0, nF = 0, nH = 0): TwoViewGeometryResult => ({
    config: 'degenerate',
    inliers: new Uint32Array(0),
    numInliersE: nE,
    numInliersF: nF,
    numInliersH: nH,
  });
  if (N < Math.max(minNumInliers, 8)) return degenerate();

  // 1. Coordinates
  const xn1 = new Float64Array(2 * N),
    xn2 = new Float64Array(2 * N);
  const up1 = new Float64Array(2 * N),
    up2 = new Float64Array(2 * N);
  const tmp = new Float64Array(2);
  const if1 = 1 / cam1.f,
    if2 = 1 / cam2.f;
  for (let m = 0; m < N; m++) {
    const i = matches[2 * m],
      j = matches[2 * m + 1];
    undistortNormalized(cam1, (kp1[4 * i] - cam1.cx) * if1, (kp1[4 * i + 1] - cam1.cy) * if1, tmp);
    xn1[2 * m] = tmp[0];
    xn1[2 * m + 1] = tmp[1];
    up1[2 * m] = cam1.f * tmp[0] + cam1.cx;
    up1[2 * m + 1] = cam1.f * tmp[1] + cam1.cy;
    undistortNormalized(cam2, (kp2[4 * j] - cam2.cx) * if2, (kp2[4 * j + 1] - cam2.cy) * if2, tmp);
    xn2[2 * m] = tmp[0];
    xn2[2 * m + 1] = tmp[1];
    up2[2 * m] = cam2.f * tmp[0] + cam2.cx;
    up2[2 * m + 1] = cam2.f * tmp[1] + cam2.cy;
  }

  // 2a. Essential matrix
  const thrE = 0.5 * (maxErrorPx / cam1.f + maxErrorPx / cam2.f);
  const thrE2 = thrE * thrE;
  const eEst = new EssentialEstimator(xn1, xn2, N);
  // A model is only useful with ≥ minNumInliers inliers AND ratio ≥ minInlierRatio, so
  // the iteration budget is capped for the larger of the two ratios (this makes small,
  // hopeless match sets cheap).
  const capRatio = Math.min(1, Math.max(minInlierRatio, minNumInliers / N));
  const eRes = ransac(eEst, { thresholdSq: thrE2, confidence, maxIterations, minInlierRatio: capRatio, rng });
  let E: Mat3 | null = eRes.model;
  let maskE = eRes.inlierMask;
  let nE = eRes.numInliers;
  const resid = new Float64Array(N);
  // Relative pose of the refined E (cheirality-consistent), when available.
  let poseR: Mat3 | null = null;
  let poseT: Float64Array | null = null;
  if (E && nE >= 8) {
    const r = refineEssentialInliers(E, xn1, xn2, N, maskE, eRes.score, thrE, resid);
    E = r.E;
    maskE = r.mask;
    nE = r.count;
    poseR = r.R;
    poseT = r.t;
  }

  // 2b. Fundamental matrix (seeded with the F implied by E)
  const fEst = new FundamentalEstimator(up1, up2, N);
  const fSeed = E ? [fundamentalFromEssential(E, cam1, cam2)] : undefined;
  const fRes = ransac(fEst, {
    thresholdSq: maxErrorPx * maxErrorPx,
    confidence,
    maxIterations,
    minInlierRatio: capRatio,
    rng,
    initialModels: fSeed,
  });
  const F = fRes.model;
  const maskF = fRes.inlierMask;
  const nF = fRes.numInliers;

  // 2c. Homography
  const hEst = new HomographyEstimator(up1, up2, N);
  let hSeed: Mat3[] | undefined;
  const Rp = opts.rotationPrior;
  if (Rp) {
    // H = K2 R K1⁻¹ (pinhole parts)
    const K1i = [if1, 0, -cam1.cx * if1, 0, if1, -cam1.cy * if1, 0, 0, 1];
    const RK = new Float64Array(9);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) RK[3 * i + j] = Rp[3 * i] * K1i[j] + Rp[3 * i + 1] * K1i[3 + j] + Rp[3 * i + 2] * K1i[6 + j];
    const Hs = new Float64Array(9);
    for (let j = 0; j < 3; j++) {
      Hs[j] = cam2.f * RK[j] + cam2.cx * RK[6 + j];
      Hs[3 + j] = cam2.f * RK[3 + j] + cam2.cy * RK[6 + j];
      Hs[6 + j] = RK[6 + j];
    }
    hSeed = [Hs];
  }
  // H only matters if it explains > maxHRatio of the E/F inliers, or (alone) at least
  // minInlierRatio of the matches: its iteration budget is capped accordingly.
  const hMinRatio = Math.max(1e-3, maxHRatio * capRatio, (maxHRatio * Math.max(nE, nF)) / N);
  const hRes = ransac(hEst, {
    thresholdSq: maxErrorPx * maxErrorPx,
    confidence,
    maxIterations,
    minInlierRatio: hMinRatio,
    rng,
    initialModels: hSeed,
  });
  const H = hRes.model;
  const maskH = hRes.inlierMask;
  const nH = hRes.numInliers;

  // 3. Model selection
  const eDecent = E !== null && nE >= minNumInliers && (nF === 0 || nE / nF >= minEF);
  let config: TwoViewConfig;
  let best: Uint8Array;
  if (eDecent) {
    best = nE >= nF ? maskE : maskF;
    if (nH / nE > maxHRatio) {
      config = 'planar_or_panoramic';
      if (nH > countMask(best)) best = maskH;
    } else config = 'calibrated';
  } else if (F !== null && nF >= minNumInliers) {
    best = maskF;
    if (nH / nF > maxHRatio) {
      config = 'planar_or_panoramic';
      if (nH > nF) best = maskH;
    } else config = 'uncalibrated';
  } else if (H !== null && nH >= minNumInliers) {
    best = maskH;
    config = 'planar_or_panoramic';
  } else {
    return degenerate(nE, nF, nH);
  }
  const nBest = countMask(best);
  if (nBest < minNumInliers || nBest / N < minInlierRatio) return degenerate(nE, nF, nH);
  // Degenerate support: coincident / collinear inliers (within the pixel threshold) or
  // repeated correspondences fit every model of a whole family — the selected E/F/H and
  // the pose would be arbitrary.
  if (supportDegenerate(up1, up2, best, maxErrorPx, minNumInliers)) return degenerate(nE, nF, nH);

  const out: TwoViewGeometryResult = {
    config,
    inliers: extractInliers(matches, best),
    numInliersE: nE,
    numInliersF: nF,
    numInliersH: nH,
  };
  if (E) out.E = E;
  if (F) out.F = F;
  if (H) out.H = H;

  // 4. Relative pose
  if (config === 'calibrated') {
    const pose = poseFromE(E!, poseR, poseT, xn1, xn2, N, maskE);
    out.R = pose.R;
    out.t = pose.t;
    out.triAngle = medianRayAngle(pose.R, xn1, xn2, N, pose.inFront);
  } else if (config === 'planar_or_panoramic') {
    let R: Mat3 | null = null;
    let t: Float64Array | null = null;
    let inFront: Uint8Array | null = null;
    let unobservable = false;
    // Fraction of the model's inliers in front of both cameras for (R, t): ≈ 0.5 or less
    // under pure rotation (each point is in front for exactly one of ±t), ≈ 1 when the
    // translation direction is really observed.
    let frontRatio = 0;
    // Decomposition of H (cheirality support on the H inliers, best first). A pure
    // rotation H yields one entry with count −1.
    const hp = H ? homographyPoses(H, cam1, cam2, xn1, xn2, N, maskH) : [];
    const hPure = hp.length === 1 && hp[0].count < 0;
    const hc = hPure ? [] : hp;
    // Candidates whose cheirality support is within `tieTol` of the best are
    // indistinguishable (two-fold planar ambiguity); a clearly lower support means the
    // candidate puts points behind a camera, i.e. it is not a valid explanation.
    const tieTol = Math.max(2, Math.ceil(0.01 * nH));
    if (eDecent) {
      const pose = poseFromE(E!, poseR, poseT, xn1, xn2, N, maskE);
      R = pose.R;
      t = pose.t;
      inFront = pose.inFront;
      frontRatio = pose.numInFront / nE;
      // E-RANSAC cannot tell the two branches of a planar scene apart (both explain every
      // point of the plane). If its pose is the H branch with clearly less cheirality
      // support than the best one, cheirality resolves the ambiguity: switch.
      const k = closestPose(hc, R, t);
      if (k >= 0 && hc[k].dist < 0.2 && hc[k].count < hc[0].count - tieTol && hc[0].count >= 0.5 * nH) {
        R = hc[0].R;
        t = hc[0].t;
        inFront = hc[0].inFront;
        frontRatio = hc[0].count / nH;
      }
    } else if (hPure) {
      R = hp[0].R;
      t = hp[0].t;
    } else if (hc.length > 0) {
      R = hc[0].R;
      t = hc[0].t;
      inFront = hc[0].inFront;
      frontRatio = hc[0].count / nH;
    }
    unobservable = hPure && !eDecent ? true : frontRatio < 0.5;
    const triAngle = R !== null && inFront !== null ? medianRayAngle(R, xn1, xn2, N, inFront) : 0;
    let panoramic = unobservable || triAngle < minTriAngle || R === null;
    // Rotation prior: rotation-only model refined from the prior.
    if (!panoramic && Rp) {
      const rot = rotationOnlyInliers(Rp, xn1, xn2, N, cam2.f, maxErrorPx, resid);
      if (rot && countMask(rot.mask) > maxHRatio * nBest) panoramic = true;
    }
    if (panoramic) {
      out.config = 'panoramic';
      let Rr: Mat3 | null;
      if (R !== null && !unobservable && frontRatio >= PANORAMIC_FULL_MODEL_FRONT_RATIO) {
        // Small but OBSERVED translation (classified panoramic for its small triangulation
        // angle or by the rotation prior): the rotation of the full model (E, or H
        // decomposition) is unbiased, whereas a rotation-only fit absorbs the translation
        // (bias ≈ baseline/depth: measured 0.1–0.28° vs 0.01–0.05° for 3-D scenes at
        // baseline/depth ≈ 0.003–0.005).
        Rr = R;
      } else {
        // (Near-)pure rotation: rotation-only fit refined from the E/H rotation (or a
        // Kabsch fit of the inliers) — 2 constraints per point instead of 1 for E.
        const R0 = R ?? rotationFromCorrespondences(xn1, xn2, N, best);
        Rr = R0;
        if (R0) {
          const rot = rotationOnlyInliers(R0, xn1, xn2, N, cam2.f, maxErrorPx, resid);
          if (rot && countMask(rot.mask) >= Math.max(minNumInliers, 0.5 * nBest)) Rr = rot.R;
        }
      }
      if (H && nH >= minNumInliers) out.inliers = extractInliers(matches, maskH);
      if (Rr) {
        out.R = Rr;
        out.t = new Float64Array(3);
        out.triAngle = medianRayAngle(Rr, xn1, xn2, N, H && nH >= minNumInliers ? maskH : best);
      }
    } else {
      out.config = 'planar';
      out.R = R!;
      out.t = t!;
      out.triAngle = triAngle;
      // The homography uses the full 2-D constraint of every point on the plane and
      // is more accurate than E here: among the H branches with (tied) maximal
      // cheirality support, report the one closest to the selected pose; the other
      // tied branch (two-fold planar ambiguity) is `planarAlternative`.
      if (hc.length > 0) {
        const maxC = hc[0].count;
        const k = closestPose(hc, R!, t!, Math.max(maxC - tieTol, 0.5 * nH));
        let bi = -1;
        if (k >= 0 && hc[k].dist < 0.2) {
          bi = k;
          out.R = hc[k].R;
          out.t = hc[k].t;
          out.triAngle = medianRayAngle(hc[k].R, xn1, xn2, N, hc[k].inFront);
        }
        for (let j = 0; j < hc.length; j++) {
          if (j === bi || hc[j].count < Math.max(0.9 * maxC, 0.5 * nH)) continue;
          if (rotationAngleBetween(hc[j].R, out.R) > 1e-3) {
            out.planarAlternative = { R: hc[j].R, t: hc[j].t };
            break;
          }
        }
      }
    }
  }
  return out;
}

/**
 * Inliers (pixel threshold) of the rotation-only model x2 ~ K2 R K1⁻¹ x1, starting from
 * R0 and refined by Kabsch on the current inliers with shrinking gates (the first,
 * max(3·maxErrorPx, 2° of view angle), tolerates gyro-integration errors of a prior).
 */
function rotationOnlyInliers(
  R0: ArrayLike<number>,
  x1: Float64Array,
  x2: Float64Array,
  n: number,
  f2: number,
  maxErrorPx: number,
  resid: Float64Array,
): { mask: Uint8Array; R: Mat3 } | null {
  let R: Mat3 = Float64Array.from(R0 as ArrayLike<number>);
  const gates = [Math.max(3 * maxErrorPx, 0.035 * f2), Math.max(2 * maxErrorPx, 0.01 * f2), maxErrorPx, maxErrorPx];
  let mask = new Uint8Array(n);
  for (let pass = 0; pass < gates.length; pass++) {
    rotationResidualsSq(R, x1, x2, n, f2, resid);
    const g2 = gates[pass] * gates[pass];
    mask = new Uint8Array(n);
    let c = 0;
    for (let i = 0; i < n; i++)
      if (resid[i] <= g2) {
        mask[i] = 1;
        c++;
      }
    if (c < 3) return null;
    if (pass < gates.length - 1) {
      const Rn = rotationFromCorrespondences(x1, x2, n, mask);
      if (!Rn) return null;
      R = Rn;
    }
  }
  return { mask, R };
}

/**
 * Refinement of a RANSAC essential matrix on its inliers: (1) Huber-robust LM on the
 * Sampson error (threshold T) accepted while the MSAC score improves; (2) the
 * cheirality-consistent pose (`recoverPose`) is refined again with a Cauchy kernel at
 * the robust noise scale σ̂ = 1.4826·median|r| on the inliers in front of both
 * cameras — epipolar-consistent outliers have high leverage on t for small
 * baselines, and about half of them triangulate behind a camera. Inliers recounted.
 */
function refineEssentialInliers(
  E0: Mat3,
  x1: Float64Array,
  x2: Float64Array,
  N: number,
  mask0: Uint8Array,
  score0: number,
  thr: number,
  resid: Float64Array,
): { E: Mat3; mask: Uint8Array; count: number; R: Mat3 | null; t: Float64Array | null } {
  const thr2 = thr * thr;
  let E = E0;
  let mask = mask0;
  let count = countMask(mask0);
  let bestScore = score0;
  const recount = (M: Mat3): { m: Uint8Array; c: number; sc: number } => {
    sampsonDistancesSq(M, x1, x2, N, resid);
    const m = new Uint8Array(N);
    let c = 0,
      sc = 0;
    for (let i = 0; i < N; i++) {
      const r = resid[i];
      if (r <= thr2) {
        m[i] = 1;
        c++;
        sc += r;
      } else sc += thr2;
    }
    return { m, c, sc };
  };
  for (let pass = 0; pass < 2; pass++) {
    const idx = maskIndices(mask);
    const { R1, t } = decomposeEssential(E);
    const ref = refineRelativePose(R1, t, x1, x2, idx.length, idx, { huber: thr, iterations: 10 });
    const rc = recount(ref.E);
    if (!(rc.sc < bestScore)) break;
    bestScore = rc.sc;
    E = ref.E;
    mask = rc.m;
    count = rc.c;
  }
  const rp = recoverPose(E, x1, x2, N, mask);
  if (rp.numInFront < 8) return { E, mask, count, R: null, t: null };
  const idx = maskIndices(rp.inFrontMask);
  // robust scale of the Sampson residuals of the in-front inliers
  const r = new Float64Array(idx.length);
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k];
    r[k] = Math.sqrt(sampsonDistanceSq(E, x1[2 * i], x1[2 * i + 1], x2[2 * i], x2[2 * i + 1]));
  }
  const sigma = Math.min(thr, Math.max(1e-3 * thr, 1.4826 * median(r)));
  const ref = refineRelativePose(rp.R, rp.t, x1, x2, idx.length, idx, { cauchy: sigma, iterations: 20 });
  const rc = recount(ref.E);
  // Accept unless it clearly loses support (different objective than MSAC).
  if (rc.c >= 0.97 * count && rc.sc <= 1.05 * bestScore) {
    return { E: ref.E, mask: rc.m, count: rc.c, R: ref.R, t: ref.t };
  }
  return { E, mask, count, R: rp.R, t: rp.t };
}

/**
 * Relative poses from a pixel homography (H between undistorted pixels): decomposition
 * of K2⁻¹ H K1, each candidate with its cheirality count on the H inliers, sorted by
 * decreasing count. A pure rotation yields a single entry with t = 0 and count −1.
 */
function homographyPoses(
  H: Mat3,
  cam1: CameraIntrinsics,
  cam2: CameraIntrinsics,
  xn1: Float64Array,
  xn2: Float64Array,
  N: number,
  maskH: Uint8Array,
): HPose[] {
  const if2 = 1 / cam2.f;
  const K1 = [cam1.f, 0, cam1.cx, 0, cam1.f, cam1.cy, 0, 0, 1];
  const HK = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) HK[3 * i + j] = H[3 * i] * K1[j] + H[3 * i + 1] * K1[3 + j] + H[3 * i + 2] * K1[6 + j];
  const Hn = new Float64Array(9);
  for (let j = 0; j < 3; j++) {
    Hn[j] = (HK[j] - cam2.cx * HK[6 + j]) * if2;
    Hn[3 + j] = (HK[3 + j] - cam2.cy * HK[6 + j]) * if2;
    Hn[6 + j] = HK[6 + j];
  }
  const nH = countMask(maskH);
  const sx1 = new Float64Array(2 * nH),
    sx2 = new Float64Array(2 * nH);
  let c = 0;
  for (let i = 0; i < N; i++) {
    if (!maskH[i]) continue;
    sx1[2 * c] = xn1[2 * i];
    sx1[2 * c + 1] = xn1[2 * i + 1];
    sx2[2 * c] = xn2[2 * i];
    sx2[2 * c + 1] = xn2[2 * i + 1];
    c++;
  }
  const out: HPose[] = [];
  for (const sol of decomposeHomography(Hn, sx1, sx2, c)) {
    const tn = Math.hypot(sol.t[0], sol.t[1], sol.t[2]);
    if (!(tn > 1e-8)) return [{ R: sol.R, t: new Float64Array(3), count: -1, inFront: new Uint8Array(N), dist: Infinity }];
    const tu = Float64Array.of(sol.t[0] / tn, sol.t[1] / tn, sol.t[2] / tn);
    const inFront = new Uint8Array(N);
    const cnt = cheiralityCount(sol.R, tu, xn1, xn2, N, maskH, inFront);
    out.push({ R: sol.R, t: tu, count: cnt, inFront, dist: Infinity });
  }
  out.sort((a, b) => b.count - a.count);
  return out;
}

interface HPose {
  R: Mat3;
  t: Float64Array;
  count: number;
  inFront: Uint8Array;
  /** Set by `closestPose`: rotation angle + translation-direction angle (rad). */
  dist: number;
}

/**
 * Index of the candidate (with count ≥ minCount) closest to (R, t) in rotation angle +
 * translation-direction angle (stored in `dist`), or −1.
 */
function closestPose(cands: HPose[], R: ArrayLike<number>, t: ArrayLike<number>, minCount = -Infinity): number {
  let bi = -1,
    bd = Infinity;
  for (let k = 0; k < cands.length; k++) {
    const c = cands[k];
    if (c.count < minCount) continue;
    const ct = c.t[0] * t[0] + c.t[1] * t[1] + c.t[2] * t[2];
    c.dist = rotationAngleBetween(c.R, R) + Math.acos(Math.min(1, Math.max(-1, ct)));
    if (c.dist < bd) {
      bd = c.dist;
      bi = k;
    }
  }
  return bi;
}

function maskIndices(mask: Uint8Array): Uint32Array {
  const out = new Uint32Array(countMask(mask));
  let c = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) out[c++] = i;
  return out;
}

/** Pose of E: the refined cheirality-consistent (R, t) when available, else `recoverPose`. */
function poseFromE(
  E: Mat3,
  R: Mat3 | null,
  t: Float64Array | null,
  x1: Float64Array,
  x2: Float64Array,
  N: number,
  mask: Uint8Array,
): { R: Mat3; t: Float64Array; inFront: Uint8Array; numInFront: number } {
  if (R && t) {
    const inFront = new Uint8Array(N);
    const c = cheiralityCount(R, t, x1, x2, N, mask, inFront);
    // Guard: the refined pose must still be the best of the four decompositions.
    const rp = recoverPose(E, x1, x2, N, mask);
    if (c >= rp.numInFront) return { R, t, inFront, numInFront: c };
    return { R: rp.R, t: rp.t, inFront: rp.inFrontMask, numInFront: rp.numInFront };
  }
  const rp = recoverPose(E, x1, x2, N, mask);
  return { R: rp.R, t: rp.t, inFront: rp.inFrontMask, numInFront: rp.numInFront };
}
