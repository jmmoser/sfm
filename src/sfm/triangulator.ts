/**
 * Incremental triangulation (COLMAP `IncrementalTriangulator`, Schönberger & Frahm,
 * "Structure-from-Motion Revisited", CVPR 2016, §4.4–4.5):
 *  - triangulate a newly registered image: continue existing tracks through its
 *    correspondences and create new tracks with correspondences in registered images;
 *  - robust multi-view triangulation: RANSAC over track elements with 2-view minimal
 *    samples (exhaustive for ≤ 15 elements), angular residuals, minimum triangulation
 *    angle and positive depth (cheirality);
 *  - track completion (transitively through the graph) and track merging (all
 *    observations must reproject within a threshold);
 *  - re-triangulation of under-reconstructed image pairs after global BA.
 */
import type { CameraIntrinsics, Pose } from '../core/types';
import { undistortNormalized } from '../geometry/camera';
import { Rng } from '../math/random';
import type { CorrespondenceGraph } from './correspondence-graph';
import type { FeatureDatabase } from './database';
import { pairKey } from './database';
import { hasBogusParams, reprojectionErrorSq, type RecImage, type Reconstruction, type Track } from './reconstruction';

export interface TriangulatorOptions {
  /** Transitivity when collecting correspondences of a feature (COLMAP max_transitivity). */
  maxTransitivity: number;
  /** Max angular error (deg) of observations of a created point. */
  createMaxAngleErrorDeg: number;
  /** Max angular error (deg) when continuing an existing track. */
  continueMaxAngleErrorDeg: number;
  /** Max reprojection error (px) of all observations of a merged point. */
  mergeMaxReprojErrorPx: number;
  /** Max reprojection error (px) of observations added by track completion. */
  completeMaxReprojErrorPx: number;
  /** Graph hops followed by track completion. */
  completeMaxTransitivity: number;
  /** Max angular error (deg) when continuing tracks during re-triangulation. */
  reMaxAngleErrorDeg: number;
  /** Re-triangulate image pairs whose triangulated-correspondence ratio is below this. */
  reMinRatio: number;
  /** Max re-triangulation attempts per image pair. */
  reMaxTrials: number;
  /** Minimum triangulation angle (deg) of a created point. */
  minAngleDeg: number;
  /** Do not create points from two observations whose features have a single correspondence. */
  ignoreTwoViewTracks: boolean;
  /** Bogus-camera thresholds (images with bogus cameras are ignored). */
  minFocalLengthRatio: number;
  maxFocalLengthRatio: number;
  maxExtraParam: number;
}

export const DEFAULT_TRIANGULATOR_OPTIONS: Readonly<TriangulatorOptions> = Object.freeze({
  maxTransitivity: 1,
  createMaxAngleErrorDeg: 2,
  continueMaxAngleErrorDeg: 2,
  mergeMaxReprojErrorPx: 4,
  completeMaxReprojErrorPx: 4,
  completeMaxTransitivity: 5,
  reMaxAngleErrorDeg: 5,
  reMinRatio: 0.2,
  reMaxTrials: 1,
  minAngleDeg: 1.5,
  ignoreTwoViewTracks: true,
  minFocalLengthRatio: 0.1,
  maxFocalLengthRatio: 10,
  maxExtraParam: 1,
});

const DEG = Math.PI / 180;
/** Exhaustive 2-view sampling up to this many observations (COLMAP kExhaustiveSamplingThreshold). */
const EXHAUSTIVE_THRESHOLD = 15;
/** Recurse in `create` when at least this many observations were left out (COLMAP kMinRecursiveTrackLength). */
const MIN_RECURSIVE_TRACK_LENGTH = 3;

interface NormCache {
  xn: Float64Array;
  f: number;
  cx: number;
  cy: number;
  k1: number;
  k2: number;
}

/**
 * Robust multi-view point triangulation from world-frame rays.
 * Inputs: n observations with camera centres C (3n), unit world ray directions D (3n)
 * and camera z-axes Z (3n, rows of R) for the depth test.
 */
export class RayTriangulator {
  /** Output: inlier flag per observation of the last successful `estimate`. */
  inliers = new Uint8Array(64);
  /** Output: the point. */
  readonly X = new Float64Array(3);
  private res = new Float64Array(64);
  private w = new Float64Array(64);
  private idx = new Int32Array(64);
  private tmp = new Float64Array(3);
  private readonly rng: Rng;
  constructor(rng?: Rng) {
    this.rng = rng ?? new Rng(0x7a1a);
  }

  private ensure(n: number): void {
    if (this.inliers.length >= n) return;
    const cap = Math.max(n, 2 * this.inliers.length);
    this.inliers = new Uint8Array(cap);
    this.res = new Float64Array(cap);
    this.w = new Float64Array(cap);
    this.idx = new Int32Array(cap);
  }

  /**
   * Weighted least-squares intersection of rays (minimises Σ wᵢ‖(I − dᵢdᵢᵀ)(X − Cᵢ)‖²),
   * re-weighted once by 1/distance² (≈ angular error). Uses observations idx[0..m).
   * Returns false if the rays are (numerically) parallel.
   */
  solveLS(C: Float64Array, D: Float64Array, idx: Int32Array, m: number, out: Float64Array): boolean {
    const o0 = 3 * idx[0];
    const ox = C[o0],
      oy = C[o0 + 1],
      oz = C[o0 + 2];
    for (let k = 0; k < m; k++) this.w[k] = 1;
    for (let pass = 0; pass < 2; pass++) {
      let a00 = 0,
        a01 = 0,
        a02 = 0,
        a11 = 0,
        a12 = 0,
        a22 = 0,
        b0 = 0,
        b1 = 0,
        b2 = 0;
      for (let k = 0; k < m; k++) {
        const o = 3 * idx[k];
        const dx = D[o],
          dy = D[o + 1],
          dz = D[o + 2];
        const cx = C[o] - ox,
          cy = C[o + 1] - oy,
          cz = C[o + 2] - oz;
        const w = this.w[k];
        // P = I − d dᵀ
        const p00 = 1 - dx * dx,
          p01 = -dx * dy,
          p02 = -dx * dz,
          p11 = 1 - dy * dy,
          p12 = -dy * dz,
          p22 = 1 - dz * dz;
        a00 += w * p00;
        a01 += w * p01;
        a02 += w * p02;
        a11 += w * p11;
        a12 += w * p12;
        a22 += w * p22;
        b0 += w * (p00 * cx + p01 * cy + p02 * cz);
        b1 += w * (p01 * cx + p11 * cy + p12 * cz);
        b2 += w * (p02 * cx + p12 * cy + p22 * cz);
      }
      const c00 = a11 * a22 - a12 * a12,
        c01 = a02 * a12 - a01 * a22,
        c02 = a01 * a12 - a02 * a11;
      const det = a00 * c00 + a01 * c01 + a02 * c02;
      const tr = (a00 + a11 + a22) / 3;
      if (!(det > 1e-13 * tr * tr * tr)) return false;
      const c11 = a00 * a22 - a02 * a02,
        c12 = a01 * a02 - a00 * a12,
        c22 = a00 * a11 - a01 * a01;
      const id = 1 / det;
      const x = (c00 * b0 + c01 * b1 + c02 * b2) * id;
      const y = (c01 * b0 + c11 * b1 + c12 * b2) * id;
      const z = (c02 * b0 + c12 * b1 + c22 * b2) * id;
      out[0] = x + ox;
      out[1] = y + oy;
      out[2] = z + oz;
      if (pass === 0) {
        for (let k = 0; k < m; k++) {
          const o = 3 * idx[k];
          const vx = out[0] - C[o],
            vy = out[1] - C[o + 1],
            vz = out[2] - C[o + 2];
          const d2 = vx * vx + vy * vy + vz * vz;
          this.w[k] = d2 > 0 ? 1 / d2 : 0;
        }
        let ws = 0;
        for (let k = 0; k < m; k++) ws += this.w[k];
        if (!(ws > 0) || !Number.isFinite(ws)) return Number.isFinite(out[0] + out[1] + out[2]);
        const s = m / ws;
        for (let k = 0; k < m; k++) this.w[k] *= s;
      }
    }
    return Number.isFinite(out[0] + out[1] + out[2]);
  }

  /** Squared angular residuals of all n observations (π² if behind the camera). */
  private residuals(C: Float64Array, D: Float64Array, Z: Float64Array, n: number, X: Float64Array): void {
    for (let k = 0; k < n; k++) {
      const o = 3 * k;
      const vx = X[0] - C[o],
        vy = X[1] - C[o + 1],
        vz = X[2] - C[o + 2];
      const depth = Z[o] * vx + Z[o + 1] * vy + Z[o + 2] * vz;
      if (!(depth > 0)) {
        this.res[k] = Math.PI * Math.PI;
        continue;
      }
      const dx = D[o],
        dy = D[o + 1],
        dz = D[o + 2];
      const cr0 = dy * vz - dz * vy,
        cr1 = dz * vx - dx * vz,
        cr2 = dx * vy - dy * vx;
      const a = Math.atan2(Math.sqrt(cr0 * cr0 + cr1 * cr1 + cr2 * cr2), dx * vx + dy * vy + dz * vz);
      this.res[k] = a * a;
    }
  }

  private positiveDepth(C: Float64Array, Z: Float64Array, idx: Int32Array, m: number, X: Float64Array): boolean {
    for (let k = 0; k < m; k++) {
      const o = 3 * idx[k];
      if (!(Z[o] * (X[0] - C[o]) + Z[o + 1] * (X[1] - C[o + 1]) + Z[o + 2] * (X[2] - C[o + 2]) > 0)) return false;
    }
    return true;
  }

  /**
   * RANSAC triangulation: minimal samples are pairs of observations whose rays meet
   * at ≥ minAngle with positive depth in both cameras; residual = angular error;
   * inliers within maxAngle; local optimisation = LS on the inliers (with a
   * cheirality check). Exhaustive for n ≤ 15 pairs. Returns the number of inliers
   * (0 on failure); the point is in `X`, the mask in `inliers`.
   */
  estimate(C: Float64Array, D: Float64Array, Z: Float64Array, n: number, minAngle: number, maxAngle: number): number {
    if (n < 2) return 0;
    this.ensure(n);
    const T2 = maxAngle * maxAngle;
    const cosMin = Math.cos(minAngle);
    const Xc = this.tmp;
    let bestScore = Infinity,
      bestCount = 0;
    const best = this.X;
    const pair = this.idx;
    const exhaustive = n <= EXHAUSTIVE_THRESHOLD;
    const numPairs = (n * (n - 1)) >> 1;
    let maxIter = exhaustive ? numPairs : 10000;
    let i = 0,
      j = 1;
    for (let it = 0; it < maxIter; it++) {
      let a: number, b: number;
      if (exhaustive) {
        a = i;
        b = j;
        if (++j >= n) {
          i++;
          j = i + 1;
        }
      } else {
        a = this.rng.int(n);
        b = this.rng.int(n - 1);
        if (b >= a) b++;
      }
      // angle between the two rays
      const oa = 3 * a,
        ob = 3 * b;
      const cosAB = D[oa] * D[ob] + D[oa + 1] * D[ob + 1] + D[oa + 2] * D[ob + 2];
      if (cosAB > cosMin) continue; // rays too parallel: triangulation angle at X ≤ angle between rays
      pair[0] = a;
      pair[1] = b;
      if (!this.solveLS(C, D, pair, 2, Xc)) continue;
      if (!this.positiveDepth(C, Z, pair, 2, Xc)) continue;
      // triangulation angle at the point
      if (!(triAngleCos(C, oa, ob, Xc) <= cosMin)) continue;
      this.residuals(C, D, Z, n, Xc);
      let score = 0,
        cnt = 0;
      for (let k = 0; k < n; k++) {
        const r = this.res[k];
        if (r <= T2) {
          score += r;
          cnt++;
        } else score += T2;
      }
      if (score < bestScore) {
        bestScore = score;
        bestCount = cnt;
        best[0] = Xc[0];
        best[1] = Xc[1];
        best[2] = Xc[2];
        if (!exhaustive) {
          const w = cnt / n;
          const p = w * w;
          const need = p >= 1 ? 1 : Math.ceil(Math.log(1e-4) / Math.log1p(-p));
          maxIter = Math.min(10000, Math.max(need, 20));
        }
      }
    }
    if (bestCount < 2) return 0;
    // local optimisation: LS on inliers, accepted if the score does not get worse
    for (let lo = 0; lo < 2; lo++) {
      this.residuals(C, D, Z, n, best);
      let m = 0;
      for (let k = 0; k < n; k++) if (this.res[k] <= T2) pair[m++] = k;
      if (m < 2 || !this.solveLS(C, D, pair, m, Xc) || !this.positiveDepth(C, Z, pair, m, Xc)) break;
      this.residuals(C, D, Z, n, Xc);
      let score = 0;
      for (let k = 0; k < n; k++) score += Math.min(this.res[k], T2);
      if (score <= bestScore) {
        bestScore = score;
        best[0] = Xc[0];
        best[1] = Xc[1];
        best[2] = Xc[2];
      } else break;
    }
    this.residuals(C, D, Z, n, best);
    let cnt = 0;
    for (let k = 0; k < n; k++) {
      const ok = this.res[k] <= T2 ? 1 : 0;
      this.inliers[k] = ok;
      cnt += ok;
    }
    if (cnt < 2) return 0;
    // the inlier set must contain a pair with sufficient triangulation angle
    let okAngle = false;
    for (let a = 0; a < n && !okAngle; a++) {
      if (!this.inliers[a]) continue;
      for (let b = a + 1; b < n; b++)
        if (this.inliers[b] && triAngleCos(C, 3 * a, 3 * b, best) <= cosMin) {
          okAngle = true;
          break;
        }
    }
    return okAngle ? cnt : 0;
  }
}

function triAngleCos(C: Float64Array, a: number, b: number, X: ArrayLike<number>): number {
  const ax = X[0] - C[a],
    ay = X[1] - C[a + 1],
    az = X[2] - C[a + 2];
  const bx = X[0] - C[b],
    by = X[1] - C[b + 1],
    bz = X[2] - C[b + 2];
  const den = Math.sqrt((ax * ax + ay * ay + az * az) * (bx * bx + by * by + bz * bz));
  return den > 0 ? (ax * bx + ay * by + az * bz) / den : 1;
}

/** Angular error (rad) between the observed normalised point and the ray to X in camera `pose`. */
export function angularError(pose: Pose, xn: number, yn: number, X: ArrayLike<number>): number {
  const R = pose.R,
    t = pose.t;
  const px = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
  const py = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
  const pz = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
  const c0 = yn * pz - py,
    c1 = px - xn * pz,
    c2 = xn * py - yn * px;
  return Math.atan2(Math.sqrt(c0 * c0 + c1 * c1 + c2 * c2), xn * px + yn * py + pz);
}

/**
 * COLMAP-style incremental triangulator operating on a `Reconstruction` (with the
 * correspondence graph attached) and the feature database (keypoints / colours).
 * All methods are synchronous; they record created / changed points in
 * `modifiedPoints` (used by the mapper for local BA and filtering).
 */
export class IncrementalTriangulator {
  readonly modifiedPoints = new Set<number>();
  private readonly mergeTrials = new Map<number, Set<number>>();
  private readonly reTrials = new Map<number, number>();
  private readonly normCache = new Map<number, NormCache>();
  private readonly tri: RayTriangulator;
  // correspondence scratch
  private cImg = new Int32Array(256);
  private cFeat = new Int32Array(256);
  private cN = 0;
  private sImg = new Int32Array(256);
  private sFeat = new Int32Array(256);
  private C = new Float64Array(768);
  private D = new Float64Array(768);
  private Z = new Float64Array(768);
  private readonly tmp2 = new Float64Array(2);

  constructor(
    private readonly graph: CorrespondenceGraph,
    private readonly rec: Reconstruction,
    private readonly db?: FeatureDatabase,
    rng?: Rng,
  ) {
    this.tri = new RayTriangulator(rng ?? new Rng(0x7213));
  }

  /** Clears the per-step caches (merge trials). */
  clearCaches(): void {
    this.mergeTrials.clear();
  }

  addModifiedPoint(id: number): void {
    this.modifiedPoints.add(id);
  }

  /** Ids of modified points that still exist. */
  getModifiedPoints(): number[] {
    const out: number[] = [];
    for (const id of this.modifiedPoints) if (this.rec.points.has(id)) out.push(id);
    return out;
  }

  clearModifiedPoints(): void {
    this.modifiedPoints.clear();
  }

  /** Invalidates cached normalised coordinates (call after intrinsics changed — done automatically by value check). */
  invalidate(imageId?: number): void {
    if (imageId === undefined) this.normCache.clear();
    else this.normCache.delete(imageId);
  }

  /** Undistorted normalised coordinates of all features of an image (cached per intrinsics). */
  normalized(img: RecImage): Float64Array {
    const cam = this.rec.cameras.get(img.cameraId)!;
    let c = this.normCache.get(img.id);
    if (c && c.f === cam.f && c.cx === cam.cx && c.cy === cam.cy && c.k1 === cam.k1 && c.k2 === cam.k2) return c.xn;
    const kp = img.keypoints;
    const n = img.point3D.length;
    const xn = c && c.xn.length === 2 * n ? c.xn : new Float64Array(2 * n);
    if (kp) {
      const inf = 1 / cam.f;
      const t = this.tmp2;
      const noDist = cam.k1 === 0 && cam.k2 === 0;
      for (let k = 0; k < n; k++) {
        const x = (kp[4 * k] - cam.cx) * inf,
          y = (kp[4 * k + 1] - cam.cy) * inf;
        if (noDist) {
          xn[2 * k] = x;
          xn[2 * k + 1] = y;
        } else {
          undistortNormalized(cam, x, y, t);
          xn[2 * k] = t[0];
          xn[2 * k + 1] = t[1];
        }
      }
    }
    c = { xn, f: cam.f, cx: cam.cx, cy: cam.cy, k1: cam.k1, k2: cam.k2 };
    this.normCache.set(img.id, c);
    return xn;
  }

  private bogus(cam: CameraIntrinsics, o: TriangulatorOptions): boolean {
    return hasBogusParams(cam, o.minFocalLengthRatio, o.maxFocalLengthRatio, o.maxExtraParam);
  }

  private ensureCorr(n: number): void {
    if (this.cImg.length >= n) return;
    const cap = Math.max(n, 2 * this.cImg.length);
    const ci = new Int32Array(cap),
      cf = new Int32Array(cap);
    ci.set(this.cImg);
    cf.set(this.cFeat);
    this.cImg = ci;
    this.cFeat = cf;
  }

  private ensureRays(n: number): void {
    if (this.sImg.length >= n) return;
    const cap = Math.max(n, 2 * this.sImg.length);
    this.sImg = new Int32Array(cap);
    this.sFeat = new Int32Array(cap);
    this.C = new Float64Array(3 * cap);
    this.D = new Float64Array(3 * cap);
    this.Z = new Float64Array(3 * cap);
  }

  /**
   * Collects correspondences of (image, feature) in registered images with valid
   * cameras (up to `transitivity` graph hops) into cImg/cFeat; returns how many of
   * them already have a point.
   */
  private find(o: TriangulatorOptions, image: number, feature: number, transitivity: number): number {
    this.cN = 0;
    const g = this.graph;
    const heads = g.heads(image);
    if (!heads) return 0;
    let numTri = 0;
    if (transitivity <= 1) {
      const nx = g.nextOf,
        im = g.imageOf,
        ft = g.featureOf;
      for (let e = heads[feature]; e >= 0; e = nx[e]) numTri += this.consider(o, image, im[e], ft[e]);
      return numTri;
    }
    // breadth-first search through the graph (also through unregistered images)
    const KEY = 1 << 22;
    const seen = new Set<number>([image * KEY + feature]);
    let frontier: number[] = [image, feature];
    for (let hop = 0; hop < transitivity && frontier.length > 0; hop++) {
      const next: number[] = [];
      const nx = g.nextOf,
        im = g.imageOf,
        ft = g.featureOf;
      for (let s = 0; s < frontier.length; s += 2) {
        const h = g.heads(frontier[s]);
        if (!h) continue;
        for (let e = h[frontier[s + 1]]; e >= 0; e = nx[e]) {
          const j = im[e],
            fj = ft[e];
          const key = j * KEY + fj;
          if (seen.has(key)) continue;
          seen.add(key);
          next.push(j, fj);
          numTri += this.consider(o, image, j, fj);
        }
      }
      frontier = next;
    }
    return numTri;
  }

  /** Appends (j, fj) to the correspondence scratch if usable; returns 1 if it has a point. */
  private consider(o: TriangulatorOptions, image: number, j: number, fj: number): number {
    if (j === image) return 0;
    const ri = this.rec.images.get(j);
    if (!ri || !ri.registered) return 0;
    if (this.bogus(this.rec.cameras.get(ri.cameraId)!, o)) return 0;
    this.ensureCorr(this.cN + 1);
    this.cImg[this.cN] = j;
    this.cFeat[this.cN] = fj;
    this.cN++;
    return ri.point3D[fj] >= 0 ? 1 : 0;
  }

  /** Fills ray data (C, D, Z) for the observations sImg/sFeat[0..n). */
  private setupRays(n: number): void {
    const C = this.C,
      D = this.D,
      Z = this.Z;
    for (let k = 0; k < n; k++) {
      const img = this.rec.images.get(this.sImg[k])!;
      const xn = this.normalized(img);
      const f = this.sFeat[k];
      const x = xn[2 * f],
        y = xn[2 * f + 1];
      const R = img.pose.R,
        t = img.pose.t;
      const o = 3 * k;
      C[o] = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]);
      C[o + 1] = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]);
      C[o + 2] = -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);
      // world ray = Rᵀ (x, y, 1) normalised
      const inv = 1 / Math.sqrt(x * x + y * y + 1);
      const bx = x * inv,
        by = y * inv,
        bz = inv;
      D[o] = R[0] * bx + R[3] * by + R[6] * bz;
      D[o + 1] = R[1] * bx + R[4] * by + R[7] * bz;
      D[o + 2] = R[2] * bx + R[5] * by + R[8] * bz;
      Z[o] = R[6];
      Z[o + 1] = R[7];
      Z[o + 2] = R[8];
    }
  }

  private trackColor(track: Track[]): number[] | undefined {
    if (!this.db) return undefined;
    let r = 0,
      g = 0,
      b = 0,
      n = 0;
    for (const el of track) {
      const col = this.db.colors.get(el.image);
      if (!col) continue;
      const o = 3 * el.feature;
      r += col[o];
      g += col[o + 1];
      b += col[o + 2];
      n++;
    }
    if (n === 0) return undefined;
    return [Math.round(r / n), Math.round(g / n), Math.round(b / n)];
  }

  /**
   * Creates a point from the observations in (img[k], feat[k]) k < n that have no
   * point yet (RANSAC triangulation); recurses on the left-out observations.
   */
  private create(o: TriangulatorOptions, img: Int32Array, feat: Int32Array, n: number): number {
    this.ensureRays(n);
    let m = 0;
    for (let k = 0; k < n; k++) {
      const ri = this.rec.images.get(img[k])!;
      if (ri.point3D[feat[k]] >= 0) continue;
      // one observation per image
      let dup = false;
      for (let q = 0; q < m; q++)
        if (this.sImg[q] === img[k]) {
          dup = true;
          break;
        }
      if (dup) continue;
      this.sImg[m] = img[k];
      this.sFeat[m] = feat[k];
      m++;
    }
    if (m < 2) return 0;
    if (o.ignoreTwoViewTracks && m === 2 && this.graph.isTwoViewObservation(this.sImg[0], this.sFeat[0])) return 0;
    this.setupRays(m);
    const cnt = this.tri.estimate(this.C, this.D, this.Z, m, o.minAngleDeg * DEG, o.createMaxAngleErrorDeg * DEG);
    if (cnt < 2) return 0;
    const track: Track[] = [];
    for (let k = 0; k < m; k++) if (this.tri.inliers[k]) track.push({ image: this.sImg[k], feature: this.sFeat[k] });
    const id = this.rec.addPoint(this.tri.X, track, this.trackColor(track));
    this.modifiedPoints.add(id);
    if (m - track.length >= MIN_RECURSIVE_TRACK_LENGTH) {
      const ri = Int32Array.from(img.subarray(0, n)),
        rf = Int32Array.from(feat.subarray(0, n));
      return track.length + this.create(o, ri, rf, n);
    }
    return track.length;
  }

  /**
   * Adds (refImage, refFeature) to the existing point (among the correspondences'
   * points) with the smallest angular error, if ≤ maxAngle.
   */
  private continueTrack(o: TriangulatorOptions, refImage: number, refFeature: number, maxAngle: number, img: Int32Array, feat: Int32Array, n: number): number {
    const ref = this.rec.images.get(refImage)!;
    if (ref.point3D[refFeature] >= 0) return 0;
    const xn = this.normalized(ref);
    const x = xn[2 * refFeature],
      y = xn[2 * refFeature + 1];
    let bestErr = Infinity,
      bestPid = -1;
    for (let k = 0; k < n; k++) {
      const ri = this.rec.images.get(img[k])!;
      const pid = ri.point3D[feat[k]];
      if (pid < 0) continue;
      const p = this.rec.points.get(pid)!;
      const err = angularError(ref.pose, x, y, p.xyz);
      if (err < bestErr) {
        // the point must not be observed in the reference image already
        let seen = false;
        for (const el of p.track)
          if (el.image === refImage) {
            seen = true;
            break;
          }
        if (seen) continue;
        bestErr = err;
        bestPid = pid;
      }
    }
    if (bestPid >= 0 && bestErr <= maxAngle) {
      this.rec.addObservation(bestPid, { image: refImage, feature: refFeature });
      this.modifiedPoints.add(bestPid);
      return 1;
    }
    return 0;
  }

  /**
   * Triangulates the observations of a registered image: continues existing tracks
   * and creates new ones. Returns the number of added observations.
   */
  triangulateImage(o: TriangulatorOptions, imageId: number): number {
    this.clearCaches();
    const img = this.rec.images.get(imageId);
    if (!img || !img.registered) return 0;
    if (this.bogus(this.rec.cameras.get(img.cameraId)!, o)) return 0;
    let num = 0;
    const nf = img.point3D.length;
    const contAngle = o.continueMaxAngleErrorDeg * DEG;
    for (let f = 0; f < nf; f++) {
      const numTri = this.find(o, imageId, f, o.maxTransitivity);
      if (this.cN === 0) continue;
      const n = this.cN;
      const ci = this.cImg,
        cf = this.cFeat;
      if (numTri > 0) num += this.continueTrack(o, imageId, f, contAngle, ci, cf, n);
      this.ensureCorr(n + 1);
      this.cImg[n] = imageId;
      this.cFeat[n] = f;
      num += this.create(o, this.cImg, this.cFeat, n + 1);
    }
    return num;
  }

  /**
   * Completes the tracks of the image's points and triangulates its untriangulated
   * features whose correspondences have no point yet (COLMAP `CompleteImage`).
   */
  completeImage(o: TriangulatorOptions, imageId: number): number {
    this.clearCaches();
    const img = this.rec.images.get(imageId);
    if (!img || !img.registered) return 0;
    if (this.bogus(this.rec.cameras.get(img.cameraId)!, o)) return 0;
    let num = 0;
    const nf = img.point3D.length;
    for (let f = 0; f < nf; f++) {
      const pid = img.point3D[f];
      if (pid >= 0) {
        num += this.complete(o, pid);
        continue;
      }
      if (o.ignoreTwoViewTracks && this.graph.isTwoViewObservation(imageId, f)) continue;
      const numTri = this.find(o, imageId, f, o.maxTransitivity);
      if (numTri > 0 || this.cN === 0) continue;
      const n = this.cN;
      this.ensureCorr(n + 1);
      this.cImg[n] = imageId;
      this.cFeat[n] = f;
      // single estimate, no recursion (COLMAP CompleteImage)
      this.ensureRays(n + 1);
      let m = 0;
      for (let k = 0; k <= n; k++) {
        let dup = false;
        for (let q = 0; q < m; q++)
          if (this.sImg[q] === this.cImg[k]) {
            dup = true;
            break;
          }
        if (dup) continue;
        this.sImg[m] = this.cImg[k];
        this.sFeat[m] = this.cFeat[k];
        m++;
      }
      if (m < 2) continue;
      this.setupRays(m);
      const cnt = this.tri.estimate(this.C, this.D, this.Z, m, o.minAngleDeg * DEG, o.createMaxAngleErrorDeg * DEG);
      if (cnt < 2) continue;
      const track: Track[] = [];
      for (let k = 0; k < m; k++) if (this.tri.inliers[k]) track.push({ image: this.sImg[k], feature: this.sFeat[k] });
      const id = this.rec.addPoint(this.tri.X, track, this.trackColor(track));
      this.modifiedPoints.add(id);
      num += track.length;
    }
    return num;
  }

  /**
   * Track completion: adds observations of registered images that correspond
   * (transitively, up to completeMaxTransitivity hops) to the point's observations
   * and reproject within completeMaxReprojErrorPx.
   */
  complete(o: TriangulatorOptions, pointId: number): number {
    const p = this.rec.points.get(pointId);
    if (!p) return 0;
    const g = this.graph;
    const max2 = o.completeMaxReprojErrorPx * o.completeMaxReprojErrorPx;
    let num = 0;
    let queue: Track[] = p.track.slice();
    const X0 = p.xyz[0],
      X1 = p.xyz[1],
      X2 = p.xyz[2];
    for (let tr = 0; tr < o.completeMaxTransitivity && queue.length > 0; tr++) {
      const prev = queue;
      queue = [];
      for (const q of prev) {
        const heads = g.heads(q.image);
        if (!heads) continue;
        const nx = g.nextOf,
          im = g.imageOf,
          ft = g.featureOf;
        for (let e = heads[q.feature]; e >= 0; e = nx[e]) {
          const j = im[e],
            fj = ft[e];
          const ri = this.rec.images.get(j);
          if (!ri || !ri.registered || ri.point3D[fj] >= 0 || !ri.keypoints) continue;
          const cam = this.rec.cameras.get(ri.cameraId)!;
          if (this.bogus(cam, o)) continue;
          const e2 = reprojectionErrorSq(cam, ri.pose, X0, X1, X2, ri.keypoints[4 * fj], ri.keypoints[4 * fj + 1]);
          if (!(e2 <= max2)) continue;
          let seen = false;
          for (const el of p.track)
            if (el.image === j) {
              seen = true;
              break;
            }
          if (seen) continue;
          this.rec.addObservation(pointId, { image: j, feature: fj });
          this.modifiedPoints.add(pointId);
          if (tr < o.completeMaxTransitivity - 1) queue.push({ image: j, feature: fj });
          num++;
        }
      }
    }
    return num;
  }

  /**
   * Merges the point with corresponding points (through the graph) if all
   * observations of the merged point reproject within mergeMaxReprojErrorPx;
   * recursive. Returns the number of merged observations.
   */
  merge(o: TriangulatorOptions, pointId: number): number {
    const p = this.rec.points.get(pointId);
    if (!p) return 0;
    const g = this.graph;
    const max2 = o.mergeMaxReprojErrorPx * o.mergeMaxReprojErrorPx;
    let trialsA = this.mergeTrials.get(pointId);
    if (!trialsA) {
      trialsA = new Set();
      this.mergeTrials.set(pointId, trialsA);
    }
    for (const el of p.track) {
      const heads = g.heads(el.image);
      if (!heads) continue;
      for (let e = heads[el.feature]; e >= 0; e = g.nextOf[e]) {
        const j = g.imageOf[e],
          fj = g.featureOf[e];
        const ri = this.rec.images.get(j);
        if (!ri || !ri.registered) continue;
        const qid = ri.point3D[fj];
        if (qid < 0 || qid === pointId || trialsA.has(qid)) continue;
        const q = this.rec.points.get(qid)!;
        trialsA.add(qid);
        let trialsB = this.mergeTrials.get(qid);
        if (!trialsB) {
          trialsB = new Set();
          this.mergeTrials.set(qid, trialsB);
        }
        trialsB.add(pointId);
        const wa = p.track.length,
          wb = q.track.length;
        const iw = 1 / (wa + wb);
        const M0 = (wa * p.xyz[0] + wb * q.xyz[0]) * iw,
          M1 = (wa * p.xyz[1] + wb * q.xyz[1]) * iw,
          M2 = (wa * p.xyz[2] + wb * q.xyz[2]) * iw;
        let ok = true;
        for (const tr of [p.track, q.track]) {
          for (const t of tr) {
            const ti = this.rec.images.get(t.image)!;
            if (!ti.keypoints) continue;
            const e2 = reprojectionErrorSq(
              this.rec.cameras.get(ti.cameraId)!,
              ti.pose,
              M0,
              M1,
              M2,
              ti.keypoints[4 * t.feature],
              ti.keypoints[4 * t.feature + 1],
            );
            if (!(e2 <= max2)) {
              ok = false;
              break;
            }
          }
          if (!ok) break;
        }
        if (!ok) continue;
        const numObs = wa + wb;
        const mid = this.rec.mergePoints(pointId, qid);
        this.modifiedPoints.delete(pointId);
        this.modifiedPoints.delete(qid);
        this.modifiedPoints.add(mid);
        const rec = this.merge(o, mid);
        return rec > 0 ? rec : numObs;
      }
    }
    return 0;
  }

  completeTracks(o: TriangulatorOptions, pointIds: Iterable<number>): number {
    let n = 0;
    for (const id of pointIds) n += this.complete(o, id);
    return n;
  }

  mergeTracks(o: TriangulatorOptions, pointIds: Iterable<number>): number {
    let n = 0;
    for (const id of pointIds) n += this.merge(o, id);
    return n;
  }

  completeAllTracks(o: TriangulatorOptions): number {
    this.clearCaches();
    return this.completeTracks(o, [...this.rec.points.keys()]);
  }

  mergeAllTracks(o: TriangulatorOptions): number {
    this.clearCaches();
    return this.mergeTracks(o, [...this.rec.points.keys()]);
  }

  /**
   * Re-triangulates image pairs (both registered) whose ratio of triangulated
   * correspondences is below reMinRatio (at most reMaxTrials times per pair):
   * continue tracks with the relaxed angular threshold, create points between
   * untriangulated correspondences.
   */
  retriangulate(o: TriangulatorOptions): number {
    this.clearCaches();
    let num = 0;
    const reAngle = o.reMaxAngleErrorDeg * DEG;
    const pi = new Int32Array(2),
      pf = new Int32Array(2);
    for (const [a, b] of this.graph.imagePairs()) {
      const ia = this.rec.images.get(a),
        ib = this.rec.images.get(b);
      if (!ia || !ib || !ia.registered || !ib.registered) continue;
      const m = this.graph.matchesBetween(a, b);
      const total = m.length >> 1;
      if (total === 0) continue;
      let numTri = 0;
      for (let k = 0; k < m.length; k += 2) {
        const pa = ia.point3D[m[k]];
        if (pa >= 0 && pa === ib.point3D[m[k + 1]]) numTri++;
      }
      if (numTri / total >= o.reMinRatio) continue;
      const key = pairKey(a, b);
      const trials = this.reTrials.get(key) ?? 0;
      if (trials >= o.reMaxTrials) continue;
      this.reTrials.set(key, trials + 1);
      if (this.bogus(this.rec.cameras.get(ia.cameraId)!, o) || this.bogus(this.rec.cameras.get(ib.cameraId)!, o)) continue;
      for (let k = 0; k < m.length; k += 2) {
        const fa = m[k],
          fb = m[k + 1];
        const ha = ia.point3D[fa] >= 0,
          hb = ib.point3D[fb] >= 0;
        if (ha && hb) continue;
        if (ha && !hb) {
          pi[0] = a;
          pf[0] = fa;
          num += this.continueTrack(o, b, fb, reAngle, pi, pf, 1);
        } else if (!ha && hb) {
          pi[0] = b;
          pf[0] = fb;
          num += this.continueTrack(o, a, fa, reAngle, pi, pf, 1);
        } else {
          pi[0] = a;
          pf[0] = fa;
          pi[1] = b;
          pf[1] = fb;
          num += this.create(o, pi, pf, 2);
        }
      }
    }
    return num;
  }
}
