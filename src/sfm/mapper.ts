/**
 * Incremental Structure-from-Motion (COLMAP's incremental mapper: Schönberger &
 * Frahm, "Structure-from-Motion Revisited", CVPR 2016):
 *
 *  1. initial pair: images with many correspondences, two-view geometry re-estimated
 *     with the current intrinsics (calibrated / planar, or uncalibrated with E = K₂ᵀFK₁),
 *     enough inliers and median triangulation angle, limited forward motion; the
 *     thresholds are relaxed progressively (inliers 100 → 30, angle 16° → 4°);
 *  2. next-best view: unregistered images ranked by the multi-resolution visibility
 *     score of their correspondences to triangulated points (§4.2 of the paper);
 *  3. registration: P3P LO-RANSAC + robust refinement (solvePnPRansac);
 *  4. triangulation of the new image (IncrementalTriangulator);
 *  5. local BA over the image and its most connected neighbours, track merging /
 *     completion, filtering (reprojection error, triangulation angle);
 *  6. global BA (self-calibration of focal length and k1, k2 of the shared camera)
 *     whenever the model grew by 10 % (images or points), followed by
 *     re-triangulation, completion / merging and filtering, iterated while > 0.5 %
 *     of the observations change;
 *  7. multiple models when the graph is disconnected (or registration stalls).
 */
import type {
  CameraIntrinsics,
  CancelToken,
  Mat3,
  Pose,
  ProgressCallback,
  TwoViewConfig,
  Vec3,
} from '../core/types';
import type { BACameraRefine, BALinearSolverType, BALossType, BAProblem, BAResult } from '../ba/problem';
import { bundleAdjust } from '../ba/solver';
import { scaleIntrinsics } from '../geometry/camera';
import { projectToEssential, recoverPose } from '../geometry/essential';
import { solvePnPRansac } from '../geometry/pnp';
import { estimateTwoViewGeometry } from '../geometry/twoview';
import { Rng } from '../math/random';
import { CorrespondenceGraph } from './correspondence-graph';
import { pairKey, type FeatureDatabase } from './database';
import { hasBogusParams, Reconstruction } from './reconstruction';
import { DEFAULT_TRIANGULATOR_OPTIONS, IncrementalTriangulator, type TriangulatorOptions } from './triangulator';

const DEG = Math.PI / 180;

/** Options of the incremental mapper (COLMAP names in comments where they differ). */
export interface MapperOptions {
  // ---- cameras
  /**
   * Initial intrinsics guess: one for all frames, or per frame (a function returning
   * undefined falls back to `hfovPriorDeg`). Scaled to the frame size if needed.
   * Frames that share a `cameraKey` share one camera (the guess of the first frame).
   */
  intrinsics?: CameraIntrinsics | ((frameId: number) => CameraIntrinsics | undefined);
  /** Camera grouping (default: one shared camera per image size). */
  cameraKey?: (frameId: number) => string | number;
  /** FOV prior (deg) across the LONG image side when no intrinsics are given (default 69). */
  hfovPriorDeg: number;
  // ---- graph
  /** Pairs with fewer inliers are ignored (default 15). */
  minNumMatches: number;
  // ---- initialisation
  initMinNumInliers: number;
  initMinNumInliersFloor: number;
  initMinTriAngleDeg: number;
  initMinTriAngleFloorDeg: number;
  initMaxErrorPx: number;
  initMaxForwardMotion: number;
  initMaxRegTrials: number;
  /** Max two-view re-estimations per initial-pair search (bounds the cost on video input). */
  initMaxPairEvaluations: number;
  /**
   * Last-resort initialisation (runIncrementalSfM: only when the regular levels gave no
   * model at all; LiveMapper: after `liveInitFallbackAfter` keyframes without one):
   * accept pairs with median triangulation angle ≥ initFallbackMinTriAngleDeg and
   * forward motion < initFallbackMaxForwardMotion. Without it a capture walking
   * straight ahead (epipole inside the image, small median angles) yields nothing.
   */
  initFallback: boolean;
  initFallbackMinTriAngleDeg: number;
  initFallbackMaxForwardMotion: number;
  /** Fixed initial pair (optional). */
  initImagePair?: [number, number];
  // ---- registration
  absPoseMaxErrorPx: number;
  absPoseMinNumInliers: number;
  absPoseMinInlierRatio: number;
  maxRegTrials: number;
  // ---- local BA
  /** Number of neighbour images in the local bundle (besides the new image). */
  localBANumImages: number;
  localBAMinTriAngleDeg: number;
  localBAMaxIterations: number;
  localBAMaxRefinements: number;
  localBAMaxRefinementChange: number;
  localBALoss: BALossType;
  localBALossScalePx: number;
  localBATimeLimitMs: number;
  /** Relative cost decrease below which local BA stops (default 1e-4). */
  localBAFunctionTolerance: number;
  /** Observations of constant (outside-anchored) points per image in local BA (default 600). */
  localBAMaxConstantObsPerImage: number;
  // ---- global BA
  globalBAImagesRatio: number;
  globalBAPointsRatio: number;
  globalBAImagesFreq: number;
  globalBAPointsFreq: number;
  globalBAMaxIterations: number;
  globalBAMaxRefinements: number;
  globalBAMaxRefinementChange: number;
  globalBALoss: BALossType;
  globalBALossScalePx: number;
  /**
   * Linear solver of the global BA. 'auto' (default): block-Jacobi PCG when more than
   * 150 images are registered and the mean track is longer than 8 (dense reduced
   * camera matrix: measured 1.5–2× faster than the skyline Cholesky at 160–250
   * images), otherwise the BA module's own 'auto'.
   */
  globalBALinearSolver: BALinearSolverType;
  globalBATimeLimitMs: number;
  /**
   * Intermediate global BAs use at most this many observations per track (every
   * ⌈L/cap⌉-th; Σ m² Schur cost); the final global BA uses all (default 12).
   */
  globalBAMaxTrackLength: number;
  /** BA / filter rounds of intermediate global refinements (the final one uses globalBAMaxRefinements; default 2). */
  intermediateGlobalBAMaxRefinements: number;
  /** runIncrementalSfM: global BA runs in slices of this many ms with yields in between (default 500). */
  globalBASliceMs: number;
  refineFocal: boolean;
  refineDistortion: boolean;
  refinePrincipalPoint: boolean;
  /**
   * Optional weak prior on the principal point per camera id (BA `principalPointPriors`, residual
   * (c − c₀)/σ in px): acts in every BA of this mapper that refines that camera's principal point.
   * Absent = none (default).
   */
  principalPointPriors?: ReadonlyMap<number, { cx: number; cy: number; sigma: number }>;
  /** Intrinsics are refined (global BA) only once this many images are registered. */
  minImagesForIntrinsics: number;
  // ---- filtering
  filterMaxReprojErrorPx: number;
  filterMinTriAngleDeg: number;
  minFocalLengthRatio: number;
  maxFocalLengthRatio: number;
  maxExtraParam: number;
  /** Registered images with fewer points are deregistered (once ≥ 20 images are registered). */
  minImagePoints3D: number;
  // ---- models
  multipleModels: boolean;
  maxNumModels: number;
  minModelSize: number;
  maxModelOverlap: number;
  // ---- external consistency checks
  /**
   * Optional checks of rotations against an independent sensor (the engine passes the gyro,
   * src/sfm/gyro.ts): `pair` vets the relative rotation x_b = R x_a of a candidate initial pair,
   * `pose` the absolute pose of an image about to be registered (PnP). Returning false rejects
   * the pair / registration (the image stays unregistered and can be retried later).
   */
  rotationCheck?: {
    pair?(a: number, b: number, R_ba: Mat3): boolean;
    pose?(id: number, pose: Pose, rec: Reconstruction): boolean;
  };
  // ---- misc
  triangulation: TriangulatorOptions;
  seed: number;
  /** Yield to the event loop at least this often (ms) in runIncrementalSfM. */
  yieldIntervalMs: number;
}

export const DEFAULT_MAPPER_OPTIONS: Readonly<MapperOptions> = Object.freeze({
  hfovPriorDeg: 69,
  minNumMatches: 15,
  initMinNumInliers: 100,
  initMinNumInliersFloor: 30,
  initMinTriAngleDeg: 16,
  initMinTriAngleFloorDeg: 4,
  initMaxErrorPx: 4,
  initMaxForwardMotion: 0.95,
  initMaxRegTrials: 2,
  initMaxPairEvaluations: 100,
  initFallback: true,
  initFallbackMinTriAngleDeg: 1.5,
  initFallbackMaxForwardMotion: 0.9999,
  absPoseMaxErrorPx: 12,
  absPoseMinNumInliers: 30,
  absPoseMinInlierRatio: 0.25,
  maxRegTrials: 3,
  localBANumImages: 6,
  localBAMinTriAngleDeg: 6,
  localBAMaxIterations: 25,
  localBAMaxRefinements: 2,
  localBAMaxRefinementChange: 0.001,
  localBALoss: 'soft_l1' as BALossType,
  localBALossScalePx: 1,
  localBATimeLimitMs: Infinity,
  localBAFunctionTolerance: 1e-4,
  localBAMaxConstantObsPerImage: 600,
  globalBAImagesRatio: 1.1,
  globalBAPointsRatio: 1.1,
  globalBAImagesFreq: 500,
  globalBAPointsFreq: 250000,
  globalBAMaxIterations: 50,
  globalBAMaxRefinements: 5,
  globalBAMaxRefinementChange: 0.005,
  globalBALoss: 'trivial' as BALossType,
  globalBALinearSolver: 'auto' as BALinearSolverType,
  globalBALossScalePx: 1,
  globalBATimeLimitMs: Infinity,
  globalBASliceMs: 500,
  globalBAMaxTrackLength: 12,
  intermediateGlobalBAMaxRefinements: 2,
  refineFocal: true,
  refineDistortion: true,
  refinePrincipalPoint: false,
  minImagesForIntrinsics: 3,
  filterMaxReprojErrorPx: 4,
  filterMinTriAngleDeg: 1.5,
  minFocalLengthRatio: 0.1,
  maxFocalLengthRatio: 10,
  maxExtraParam: 1,
  minImagePoints3D: 6,
  multipleModels: true,
  maxNumModels: 50,
  minModelSize: 10,
  maxModelOverlap: 20,
  triangulation: DEFAULT_TRIANGULATOR_OPTIONS,
  seed: 0x5f3d,
  yieldIntervalMs: 25,
});

/** Applies defaults (the nested `triangulation` options are merged too). */
export function resolveMapperOptions(o: Partial<MapperOptions> = {}): MapperOptions {
  const out = { ...DEFAULT_MAPPER_OPTIONS, ...o } as MapperOptions;
  out.triangulation = { ...DEFAULT_TRIANGULATOR_OPTIONS, ...(o.triangulation ?? {}) };
  out.triangulation.minFocalLengthRatio = out.minFocalLengthRatio;
  out.triangulation.maxFocalLengthRatio = out.maxFocalLengthRatio;
  out.triangulation.maxExtraParam = out.maxExtraParam;
  return out;
}

/** Thrown by `runIncrementalSfM` / mapper loops when the cancel token is set. */
export class SfMCancelledError extends Error {
  constructor(stage = 'sfm') {
    super(`${stage}: cancelled`);
    this.name = 'CancelledError';
  }
}

/** Re-estimated geometry of a candidate initial pair. */
export interface InitPairGeometry {
  config: TwoViewConfig;
  /** Pose of the second image relative to the first (x2 = R x1 + t, |t| = 1). */
  R: Mat3;
  t: Vec3;
  /** Inlier matches (first-image feature, second-image feature). */
  inliers: Uint32Array;
  numInliers: number;
  /** Median triangulation angle (rad) of the inliers in front of both cameras. */
  triAngle: number;
  /** |t_z|: forward-motion component of the unit baseline. */
  forwardMotion: number;
}

/** State shared between the mappers of successive models of one run. */
export interface MapperSharedState {
  /** Image → number of models it is registered in. */
  registrations: Map<number, number>;
  /** Image → number of times it was used in an initial pair. */
  initRegTrials: Map<number, number>;
  /** Cache of re-estimated initial-pair geometries (keyed by pairKey; valid for the prior intrinsics). */
  initGeometry: Map<number, InitPairGeometry | null>;
  /** Pairs already used for initialisation, keyed by `${level}:${pairKey}`. */
  initPairsTried: Set<string>;
  /** Number of two-view re-estimations done. */
  initEvaluations: number;
}

export function createMapperSharedState(): MapperSharedState {
  return { registrations: new Map(), initRegTrials: new Map(), initGeometry: new Map(), initPairsTried: new Set(), initEvaluations: 0 };
}

export interface LocalBAReport {
  numAdjustedObservations: number;
  numMergedObservations: number;
  numCompletedObservations: number;
  numFilteredObservations: number;
  ba: BAResult | null;
}

/** Accumulated wall-clock times (ms) of the mapper stages. */
export interface MapperTimings {
  init: number;
  register: number;
  triangulate: number;
  localBA: number;
  globalBA: number;
  retriangulate: number;
  filter: number;
  numLocalBA: number;
  numGlobalBA: number;
  /** LM iterations summed over all local / global BA runs. */
  localBAIterations: number;
  globalBAIterations: number;
}

/** Configuration of one bundle adjustment over a subset of the reconstruction. */
interface BundleConfig {
  /** Images whose observations are all added (poses free unless constant). */
  images: number[];
  constantPoses: Set<number>;
  /** Image → translationFixedMask bits. */
  translationMask: Map<number, number>;
  /** Points added with ALL their observations (outside images as constant poses). */
  variablePoints: number[];
  refine: (cameraId: number) => BACameraRefine;
  /** Cap on observations of constant points per image (subsampled; Infinity = all). */
  maxConstObsPerImage: number;
  /**
   * Global BA only (all registered images, no variable points): tracks longer than
   * this contribute only every ⌈L/cap⌉-th observation (Infinity = all).
   */
  trackCap?: number;
  /** Focal priors per camera id (BA `focalPriors`). */
  focalPriors?: Map<number, { f: number; sigma: number }>;
}

/** Cancellation / per-iteration hook forwarded to `bundleAdjust`. */
interface BAControl {
  cancel?: CancelToken;
  onIteration?: (iter: number, cost: number) => void;
}

/** Two-view re-estimations per chunk of the async initial-pair search. */
const INIT_CHUNK = 6;

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const yieldToEventLoop = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

/** Default intrinsics from a FOV prior across the long side. */
export function intrinsicsFromFov(width: number, height: number, hfovLongDeg: number): CameraIntrinsics {
  const f = (0.5 * Math.max(width, height)) / Math.tan(0.5 * hfovLongDeg * DEG);
  return { width, height, f, cx: 0.5 * (width - 1), cy: 0.5 * (height - 1), k1: 0, k2: 0 };
}

/** Builds the correspondence graph of all frames and stored pairs (≥ minNumMatches inliers, not degenerate). */
export function buildCorrespondenceGraph(db: FeatureDatabase, minNumMatches = 15): CorrespondenceGraph {
  const g = new CorrespondenceGraph();
  for (const id of db.frameIds()) g.addImage(id, db.features.get(id)!.count);
  for (const [a, b, geom] of db.pairs()) {
    if (geom.config === 'degenerate') continue;
    if (geom.inliers.length >> 1 < minNumMatches) continue;
    if (!g.hasImage(a) || !g.hasImage(b)) continue;
    g.addTwoView(a, b, geom.inliers);
  }
  return g;
}

/** Sets every point colour to the mean of its observations' feature colours (db.colors). */
export function extractColors(rec: Reconstruction, db: FeatureDatabase): void {
  for (const p of rec.points.values()) {
    let r = 0,
      g = 0,
      b = 0,
      n = 0;
    for (const el of p.track) {
      const c = db.colors.get(el.image);
      if (!c) continue;
      r += c[3 * el.feature];
      g += c[3 * el.feature + 1];
      b += c[3 * el.feature + 2];
      n++;
    }
    if (n > 0) p.color = [Math.round(r / n), Math.round(g / n), Math.round(b / n)];
  }
}

/** Median of an array (destructive sort). */
function medianOf(a: number[]): number {
  if (a.length === 0) return 0;
  a.sort((x, y) => x - y);
  return a[a.length >> 1];
}

function percentileOf(a: number[], p: number): number {
  if (a.length === 0) return 0;
  a.sort((x, y) => x - y);
  const k = Math.min(a.length - 1, Math.max(0, Math.round((p / 100) * (a.length - 1))));
  return a[k];
}

/** Index (0..2) of the largest |component| of the translation of `b` relative to `a` (gauge choice). */
function largestRelTranslationAxis(a: Pose, b: Pose): number {
  // t_rel = t_b − R_b R_aᵀ t_a
  const Ra = a.R,
    ta = a.t,
    Rb = b.R,
    tb = b.t;
  const w0 = Ra[0] * ta[0] + Ra[3] * ta[1] + Ra[6] * ta[2];
  const w1 = Ra[1] * ta[0] + Ra[4] * ta[1] + Ra[7] * ta[2];
  const w2 = Ra[2] * ta[0] + Ra[5] * ta[1] + Ra[8] * ta[2];
  const r0 = Math.abs(tb[0] - (Rb[0] * w0 + Rb[1] * w1 + Rb[2] * w2));
  const r1 = Math.abs(tb[1] - (Rb[3] * w0 + Rb[4] * w1 + Rb[5] * w2));
  const r2 = Math.abs(tb[2] - (Rb[6] * w0 + Rb[7] * w1 + Rb[8] * w2));
  return r0 >= r1 && r0 >= r2 ? 0 : r1 >= r2 ? 1 : 2;
}

/**
 * Rejects degenerate absolute poses: non-finite values, or inlier observations so
 * concentrated (RMS spread < maxErrorPx) that a camera arbitrarily far away explains
 * them all within the threshold — PnP + refinement then drifts the camera to ~1e8
 * scene units (seen with coinciding keypoints / tight clusters of wrong matches).
 */
function poseIsUsable(pose: Pose, inlierMask: Uint8Array, uv: Float64Array, n: number, maxErrorPx: number): boolean {
  for (let k = 0; k < 9; k++) if (!Number.isFinite(pose.R[k])) return false;
  for (let k = 0; k < 3; k++) if (!Number.isFinite(pose.t[k])) return false;
  let mu = 0,
    mv = 0,
    m = 0;
  for (let k = 0; k < n; k++)
    if (inlierMask[k]) {
      mu += uv[2 * k];
      mv += uv[2 * k + 1];
      m++;
    }
  if (m === 0) return false;
  mu /= m;
  mv /= m;
  let s2 = 0;
  for (let k = 0; k < n; k++)
    if (inlierMask[k]) {
      const du = uv[2 * k] - mu,
        dv = uv[2 * k + 1] - mv;
      s2 += du * du + dv * dv;
    }
  return Math.sqrt(s2 / m) >= maxErrorPx;
}

/**
 * COLMAP-style incremental mapper for ONE model. The reconstruction must have the
 * correspondence graph attached (`ensureImage` adds frames on demand, used by the
 * live mapper). Synchronous building blocks + an async driver (`reconstruct`).
 */
export class IncrementalMapper {
  readonly opts: MapperOptions;
  readonly triangulator: IncrementalTriangulator;
  readonly shared: MapperSharedState;
  readonly timings: MapperTimings = {
    init: 0,
    register: 0,
    triangulate: 0,
    localBA: 0,
    globalBA: 0,
    retriangulate: 0,
    filter: 0,
    numLocalBA: 0,
    numGlobalBA: 0,
    localBAIterations: 0,
    globalBAIterations: 0,
  };
  private readonly rng: Rng;
  private readonly priorCameras = new Map<number, CameraIntrinsics>();
  private readonly cameraIds = new Map<string | number, number>();
  private readonly numRegTrials = new Map<number, number>();
  private readonly filteredImages = new Set<number>();
  private readonly numRegPerCamera = new Map<number, number>();
  /** Registered images of this model that are also registered in other models. */
  numSharedRegImages = 0;
  /** Registrations / initial pairs rejected by `rotationCheck`. */
  numRotationRejects = 0;

  constructor(
    readonly db: FeatureDatabase,
    readonly graph: CorrespondenceGraph,
    readonly rec: Reconstruction,
    opts: Partial<MapperOptions> = {},
    shared?: MapperSharedState,
  ) {
    this.opts = resolveMapperOptions(opts);
    this.rng = new Rng(this.opts.seed);
    this.shared = shared ?? createMapperSharedState();
    if (rec.correspondenceGraph !== graph) rec.attachGraph(graph);
    this.triangulator = new IncrementalTriangulator(graph, rec, db, new Rng(this.opts.seed + 1));
    for (const [cid, cam] of rec.cameras) this.priorCameras.set(cid, { ...cam });
    for (const id of rec.registeredImageIds()) {
      const cid = rec.images.get(id)!.cameraId;
      this.numRegPerCamera.set(cid, (this.numRegPerCamera.get(cid) ?? 0) + 1);
    }
  }

  // ------------------------------------------------------------------ setup

  private initialIntrinsics(id: number): CameraIntrinsics {
    const info = this.db.frames.get(id)!;
    const fs = this.db.features.get(id)!;
    const W = info.width || fs.width,
      H = info.height || fs.height;
    const g = this.opts.intrinsics;
    let cam: CameraIntrinsics | undefined;
    if (typeof g === 'function') cam = g(id);
    else if (g) cam = g;
    if (cam) return cam.width === W && cam.height === H ? { ...cam } : scaleIntrinsics(cam, W, H);
    return intrinsicsFromFov(W, H, this.opts.hfovPriorDeg);
  }

  /** Adds the frame to the reconstruction (creating / sharing its camera) if missing. */
  ensureImage(id: number): void {
    if (this.rec.images.has(id)) return;
    const info = this.db.frames.get(id);
    const fs = this.db.features.get(id);
    if (!info || !fs) throw new RangeError(`IncrementalMapper: frame ${id} not in the database`);
    const key = this.opts.cameraKey ? this.opts.cameraKey(id) : `${info.width || fs.width}x${info.height || fs.height}`;
    let cid = this.cameraIds.get(key);
    if (cid === undefined) {
      const cam = this.initialIntrinsics(id);
      cid = this.rec.addCamera(cam);
      this.cameraIds.set(key, cid);
      this.priorCameras.set(cid, { ...cam });
    }
    this.rec.addImage(id, cid, fs.count, fs.keypoints);
  }

  private bogus(cid: number): boolean {
    const o = this.opts;
    return hasBogusParams(this.rec.cameras.get(cid)!, o.minFocalLengthRatio, o.maxFocalLengthRatio, o.maxExtraParam);
  }

  private onRegistered(id: number): void {
    const cid = this.rec.images.get(id)!.cameraId;
    this.numRegPerCamera.set(cid, (this.numRegPerCamera.get(cid) ?? 0) + 1);
    const n = this.shared.registrations.get(id) ?? 0;
    if (n > 0) this.numSharedRegImages++;
    this.shared.registrations.set(id, n + 1);
  }

  private onDeregistered(id: number): void {
    const cid = this.rec.images.get(id)!.cameraId;
    this.numRegPerCamera.set(cid, Math.max(0, (this.numRegPerCamera.get(cid) ?? 1) - 1));
    const n = this.shared.registrations.get(id) ?? 1;
    if (n > 1) this.numSharedRegImages = Math.max(0, this.numSharedRegImages - 1);
    this.shared.registrations.set(id, Math.max(0, n - 1));
  }

  /** Deregisters an image (and its observations) with bookkeeping. */
  deregisterImage(id: number): void {
    const img = this.rec.images.get(id);
    if (!img || !img.registered) return;
    this.rec.deregisterImage(id);
    this.onDeregistered(id);
  }

  // ------------------------------------------------------------------ initialisation

  /**
   * Two-view geometry of (a, b) re-estimated on the graph correspondences with the
   * current intrinsics. Null for panoramic / degenerate pairs.
   */
  estimateInitPair(a: number, b: number): InitPairGeometry | null {
    const matches = this.graph.matchesBetween(a, b);
    const n = matches.length >> 1;
    if (n < 8) return null;
    const ia = this.rec.images.get(a)!,
      ib = this.rec.images.get(b)!;
    const ca = this.rec.cameras.get(ia.cameraId)!,
      cb = this.rec.cameras.get(ib.cameraId)!;
    const kpa = this.db.features.get(a)!.keypoints,
      kpb = this.db.features.get(b)!.keypoints;
    this.shared.initEvaluations++;
    const g = estimateTwoViewGeometry(kpa, kpb, matches, ca, cb, {
      maxErrorPx: this.opts.initMaxErrorPx,
      minNumInliers: 15,
      rng: this.rng,
    });
    let R: Mat3 | undefined = g.R,
      t: Vec3 | undefined = g.t;
    const inl = g.inliers;
    const m = inl.length >> 1;
    if (m < 8) return null;
    const xa = this.triangulator.normalized(ia),
      xb = this.triangulator.normalized(ib);
    const x1 = new Float64Array(2 * m),
      x2 = new Float64Array(2 * m);
    for (let k = 0; k < m; k++) {
      x1[2 * k] = xa[2 * inl[2 * k]];
      x1[2 * k + 1] = xa[2 * inl[2 * k] + 1];
      x2[2 * k] = xb[2 * inl[2 * k + 1]];
      x2[2 * k + 1] = xb[2 * inl[2 * k + 1] + 1];
    }
    if (g.config === 'uncalibrated' && g.F) {
      // E = K_bᵀ F K_a (F acts on undistorted pixels)
      const F = g.F;
      const Ka = [ca.f, 0, ca.cx, 0, ca.f, ca.cy, 0, 0, 1],
        Kb = [cb.f, 0, cb.cx, 0, cb.f, cb.cy, 0, 0, 1];
      const FK = new Float64Array(9),
        E0 = new Float64Array(9);
      for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++) FK[3 * r + c] = F[3 * r] * Ka[c] + F[3 * r + 1] * Ka[3 + c] + F[3 * r + 2] * Ka[6 + c];
      for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++) E0[3 * r + c] = Kb[r] * FK[c] + Kb[3 + r] * FK[3 + c] + Kb[6 + r] * FK[6 + c];
      const E = projectToEssential(E0);
      if (!E) return null;
      const rp = recoverPose(E, x1, x2, m);
      R = rp.R;
      t = rp.t;
    } else if (g.config !== 'calibrated' && g.config !== 'planar') return null;
    if (!R || !t) return null;
    const tn = Math.hypot(t[0], t[1], t[2]);
    if (!(tn > 0)) return null;
    // median triangulation angle of the inliers in front of both cameras
    const angles: number[] = [];
    const tx = t[0] / tn,
      ty = t[1] / tn,
      tz = t[2] / tn;
    // camera 1 at origin, camera 2 centre C2 = −Rᵀ t
    const C2x = -(R[0] * tx + R[3] * ty + R[6] * tz),
      C2y = -(R[1] * tx + R[4] * ty + R[7] * tz),
      C2z = -(R[2] * tx + R[5] * ty + R[8] * tz);
    for (let k = 0; k < m; k++) {
      // ray 1: d1 = (x1, 1); ray 2 in world: Rᵀ (x2, 1)
      const d1x = x1[2 * k],
        d1y = x1[2 * k + 1],
        d1z = 1;
      const u = x2[2 * k],
        v = x2[2 * k + 1];
      const d2x = R[0] * u + R[3] * v + R[6],
        d2y = R[1] * u + R[4] * v + R[7],
        d2z = R[2] * u + R[5] * v + R[8];
      // midpoint: minimise |s d1 − C2 − r d2|
      const aa = d1x * d1x + d1y * d1y + d1z * d1z,
        bb = d2x * d2x + d2y * d2y + d2z * d2z,
        ab = d1x * d2x + d1y * d2y + d1z * d2z;
      const ac = d1x * C2x + d1y * C2y + d1z * C2z,
        bc = d2x * C2x + d2y * C2y + d2z * C2z;
      const det = aa * bb - ab * ab;
      if (!(det > 1e-14 * aa * bb)) continue;
      const s = (ac * bb - ab * bc) / det,
        r = (ab * ac - aa * bc) / det;
      if (!(s > 0 && r > 0)) continue;
      const Xx = 0.5 * (s * d1x + C2x + r * d2x),
        Xy = 0.5 * (s * d1y + C2y + r * d2y),
        Xz = 0.5 * (s * d1z + C2z + r * d2z);
      const ax = Xx,
        ay = Xy,
        az = Xz;
      const bx = Xx - C2x,
        by = Xy - C2y,
        bz = Xz - C2z;
      const c = (ax * bx + ay * by + az * bz) / Math.sqrt((ax * ax + ay * ay + az * az) * (bx * bx + by * by + bz * bz));
      angles.push(Math.acos(Math.max(-1, Math.min(1, c))));
    }
    if (angles.length < 8) return null;
    return {
      config: g.config,
      R: Float64Array.from(R),
      t: Float64Array.of(tx, ty, tz),
      inliers: Uint32Array.from(inl),
      numInliers: m,
      triAngle: medianOf(angles),
      forwardMotion: Math.abs(tz),
    };
  }

  /** Two-view re-estimations left in the current `findInitialPair` search. */
  private initBudget = Infinity;

  private initGeometry(a: number, b: number): InitPairGeometry | null {
    const key = pairKey(a, b);
    let g = this.shared.initGeometry.get(key);
    if (g === undefined) {
      this.initBudget--;
      g = this.estimateInitPair(Math.min(a, b), Math.max(a, b)); // oriented min → max
      this.shared.initGeometry.set(key, g);
    }
    if (!g) return null;
    if (a < b) return g;
    // invert to a → b
    const R = g.R,
      t = g.t;
    const Rt = Float64Array.of(R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]);
    const ti = Float64Array.of(
      -(Rt[0] * t[0] + Rt[1] * t[1] + Rt[2] * t[2]),
      -(Rt[3] * t[0] + Rt[4] * t[1] + Rt[5] * t[2]),
      -(Rt[6] * t[0] + Rt[7] * t[1] + Rt[8] * t[2]),
    );
    const inl = new Uint32Array(g.inliers.length);
    for (let k = 0; k < inl.length; k += 2) {
      inl[k] = g.inliers[k + 1];
      inl[k + 1] = g.inliers[k];
    }
    return { ...g, R: Rt, t: ti, inliers: inl, forwardMotion: Math.abs(ti[2]) };
  }

  /** True if the image may take part in an initial pair. */
  private initCandidate(id: number): boolean {
    if ((this.shared.registrations.get(id) ?? 0) > 0) return false;
    if ((this.shared.initRegTrials.get(id) ?? 0) >= this.opts.initMaxRegTrials) return false;
    const img = this.rec.images.get(id);
    return !!img && !img.registered;
  }

  /** True if the last `findInitialPair` stopped because its evaluation budget ran out. */
  initSearchExhausted = false;

  /**
   * Finds an initial pair satisfying the given thresholds (COLMAP order: first image
   * by number of correspondences; second image by correspondences, preferring pairs
   * whose stored two-view triangulation angle already passes — i.e. pairs further
   * apart in a video sequence). Each pair is tried once per threshold level. At most
   * `budget` two-view re-estimations are done (cached results are free); when the
   * budget runs out, `initSearchExhausted` is set and a later call continues.
   */
  findInitialPair(
    minInliers: number,
    minTriAngleDeg: number,
    levelKey?: string,
    budget = this.opts.initMaxPairEvaluations,
    maxForwardMotion = this.opts.initMaxForwardMotion,
  ): { a: number; b: number; geom: InitPairGeometry } | null {
    levelKey ??= `${minInliers}/${minTriAngleDeg}/${maxForwardMotion}`;
    const minAngle = minTriAngleDeg * DEG;
    this.initBudget = budget;
    this.initSearchExhausted = false;
    const check = this.opts.rotationCheck?.pair;
    const accept = (g: InitPairGeometry | null, a: number, b: number): g is InitPairGeometry => {
      if (!g || !(g.numInliers >= minInliers && g.triAngle >= minAngle && g.forwardMotion < maxForwardMotion)) return false;
      if (check && !check(a, b, g.R)) {
        this.numRotationRejects++;
        return false;
      }
      return true;
    };
    const evaluate = (a: number, b: number): InitPairGeometry | null | undefined => {
      // undefined = budget exhausted (pair not marked as tried)
      if (this.initBudget <= 0 && !this.shared.initGeometry.has(pairKey(a, b))) {
        this.initSearchExhausted = true;
        return undefined;
      }
      this.shared.initPairsTried.add(`${levelKey}:${pairKey(a, b)}`);
      return this.initGeometry(a, b);
    };
    if (this.opts.initImagePair) {
      const [a, b] = this.opts.initImagePair;
      if (this.shared.initPairsTried.has(`${levelKey}:${pairKey(a, b)}`)) return null;
      const g = evaluate(a, b);
      return g && g.numInliers >= 15 ? { a, b, geom: g } : null;
    }
    const firsts = [...this.rec.images.keys()]
      .filter((id) => this.initCandidate(id) && this.graph.numCorrespondences(id) > 0)
      .sort((x, y) => this.graph.numCorrespondences(y) - this.graph.numCorrespondences(x) || x - y);
    for (const a of firsts) {
      const cands: { b: number; n: number; ok: boolean }[] = [];
      for (const [b, n] of this.graph.neighbours(a)) {
        if (n < minInliers || !this.initCandidate(b)) continue;
        // (canonical order: no inverted copy is built / cached; config and angle are symmetric)
        const stored = this.db.getTwoView(Math.min(a, b), Math.max(a, b));
        if (stored && stored.config === 'panoramic') continue;
        const sa = stored?.triAngle;
        // stored angle far below the threshold: hopeless (skip the re-estimation)
        if (sa !== undefined && sa < 0.5 * minAngle) continue;
        cands.push({ b, n, ok: sa === undefined || sa >= minAngle });
      }
      cands.sort((x, y) => Number(y.ok) - Number(x.ok) || y.n - x.n || x.b - y.b);
      for (const { b } of cands) {
        if (this.shared.initPairsTried.has(`${levelKey}:${pairKey(a, b)}`)) continue;
        const g = evaluate(a, b);
        if (g === undefined) return null;
        if (accept(g, a, b)) return { a, b, geom: g };
      }
    }
    return null;
  }

  /**
   * Registers the initial pair (first image at the origin) and triangulates its
   * inliers (positive depth, triangulation angle ≥ minPointAngleDeg).
   */
  registerInitialPair(a: number, b: number, geom: InitPairGeometry, minPointAngleDeg: number): number {
    const t0 = now();
    this.shared.initRegTrials.set(a, (this.shared.initRegTrials.get(a) ?? 0) + 1);
    this.shared.initRegTrials.set(b, (this.shared.initRegTrials.get(b) ?? 0) + 1);
    const ia = this.rec.images.get(a)!,
      ib = this.rec.images.get(b)!;
    this.rec.registerImage(a, { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: new Float64Array(3) });
    this.rec.registerImage(b, { R: geom.R, t: geom.t });
    this.onRegistered(a);
    this.onRegistered(b);
    const xa = this.triangulator.normalized(ia),
      xb = this.triangulator.normalized(ib);
    const R = geom.R,
      t = geom.t;
    const C2x = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
      C2y = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
      C2z = -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);
    const cosMin = Math.cos(minPointAngleDeg * DEG);
    const inl = geom.inliers;
    let num = 0;
    const X = new Float64Array(3);
    const colA = this.db.colors.get(a),
      colB = this.db.colors.get(b);
    for (let k = 0; k < inl.length; k += 2) {
      const fa = inl[k],
        fb = inl[k + 1];
      if (ia.point3D[fa] >= 0 || ib.point3D[fb] >= 0) continue;
      const d1x = xa[2 * fa],
        d1y = xa[2 * fa + 1];
      const u = xb[2 * fb],
        v = xb[2 * fb + 1];
      const d2x = R[0] * u + R[3] * v + R[6],
        d2y = R[1] * u + R[4] * v + R[7],
        d2z = R[2] * u + R[5] * v + R[8];
      const aa = d1x * d1x + d1y * d1y + 1,
        bb = d2x * d2x + d2y * d2y + d2z * d2z,
        ab = d1x * d2x + d1y * d2y + d2z;
      const ac = d1x * C2x + d1y * C2y + C2z,
        bc = d2x * C2x + d2y * C2y + d2z * C2z;
      const det = aa * bb - ab * ab;
      if (!(det > 1e-14 * aa * bb)) continue;
      const s = (ac * bb - ab * bc) / det,
        r = (ab * ac - aa * bc) / det;
      if (!(s > 0 && r > 0)) continue;
      X[0] = 0.5 * (s * d1x + C2x + r * d2x);
      X[1] = 0.5 * (s * d1y + C2y + r * d2y);
      X[2] = 0.5 * (s + C2z + r * d2z);
      const bx = X[0] - C2x,
        by = X[1] - C2y,
        bz = X[2] - C2z;
      const c = (X[0] * bx + X[1] * by + X[2] * bz) / Math.sqrt((X[0] * X[0] + X[1] * X[1] + X[2] * X[2]) * (bx * bx + by * by + bz * bz));
      if (!(c <= cosMin)) continue;
      // positive depth in camera 2
      if (!(R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2] > 0) || !(X[2] > 0)) continue;
      let color: number[] | undefined;
      if (colA && colB)
        color = [
          (colA[3 * fa] + colB[3 * fb]) >> 1,
          (colA[3 * fa + 1] + colB[3 * fb + 1]) >> 1,
          (colA[3 * fa + 2] + colB[3 * fb + 2]) >> 1,
        ];
      const pid = this.rec.addPoint(X, [
        { image: a, feature: fa },
        { image: b, feature: fb },
      ], color);
      this.triangulator.addModifiedPoint(pid);
      num++;
    }
    this.timings.init += now() - t0;
    return num;
  }

  // ------------------------------------------------------------------ registration

  /** Unregistered images ranked for registration (COLMAP FindNextImages, MIN_UNCERTAINTY). */
  findNextImages(): number[] {
    const fresh: [number, number][] = [],
      other: [number, number][] = [];
    for (const img of this.rec.images.values()) {
      if (img.registered) continue;
      if (img.numVisiblePoints3D < this.opts.absPoseMinNumInliers) continue;
      const trials = this.numRegTrials.get(img.id) ?? 0;
      if (trials >= this.opts.maxRegTrials) continue;
      const rank = img.visibilityScore;
      if (!this.filteredImages.has(img.id) && trials === 0) fresh.push([img.id, rank]);
      else other.push([img.id, rank]);
    }
    const byRank = (x: [number, number], y: [number, number]): number => y[1] - x[1] || x[0] - y[0];
    fresh.sort(byRank);
    other.sort(byRank);
    return fresh.map((e) => e[0]).concat(other.map((e) => e[0]));
  }

  /** Resets the registration trial counter of an image (e.g. after the graph grew). */
  resetRegTrials(id: number): void {
    this.numRegTrials.delete(id);
  }

  /**
   * Registers an image by PnP against the triangulated points visible through its
   * correspondences; adds the inlier observations to their points. Returns success.
   */
  registerNextImage(id: number): boolean {
    const t0 = now();
    try {
      return this.registerNextImageImpl(id);
    } finally {
      this.timings.register += now() - t0;
    }
  }

  private registerNextImageImpl(id: number): boolean {
    const o = this.opts;
    const img = this.rec.images.get(id);
    if (!img || img.registered) return false;
    this.numRegTrials.set(id, (this.numRegTrials.get(id) ?? 0) + 1);
    if (img.numVisiblePoints3D < o.absPoseMinNumInliers && this.rec.correspondenceGraph) {
      // the counts can be stale when the graph grew after the image was added
      this.rec.syncVisibility(id);
      if (img.numVisiblePoints3D < o.absPoseMinNumInliers) return false;
    }
    const cid = img.cameraId;
    if ((this.numRegPerCamera.get(cid) ?? 0) === 0 || this.bogus(cid)) {
      const prior = this.priorCameras.get(cid);
      if (prior) Object.assign(this.rec.cameras.get(cid)!, prior);
    }
    const cam = this.rec.cameras.get(cid)!;
    const kp = img.keypoints!;
    const g = this.graph;
    const heads = g.heads(id);
    if (!heads) return false;
    const nx = g.nextOf,
      im = g.imageOf,
      ft = g.featureOf;
    const feats: number[] = [],
      pids: number[] = [];
    const nf = img.point3D.length;
    for (let f = 0; f < nf; f++) {
      const start = pids.length;
      for (let e = heads[f]; e >= 0; e = nx[e]) {
        const ri = this.rec.images.get(im[e]);
        if (!ri || !ri.registered) continue;
        const pid = ri.point3D[ft[e]];
        if (pid < 0) continue;
        let dup = false;
        for (let q = start; q < pids.length; q++)
          if (pids[q] === pid) {
            dup = true;
            break;
          }
        if (dup || this.bogus(ri.cameraId)) continue;
        feats.push(f);
        pids.push(pid);
      }
    }
    const n = pids.length;
    if (n < o.absPoseMinNumInliers) return false;
    const X = new Float64Array(3 * n),
      uv = new Float64Array(2 * n);
    for (let k = 0; k < n; k++) {
      const p = this.rec.points.get(pids[k])!;
      X[3 * k] = p.xyz[0];
      X[3 * k + 1] = p.xyz[1];
      X[3 * k + 2] = p.xyz[2];
      uv[2 * k] = kp[4 * feats[k]];
      uv[2 * k + 1] = kp[4 * feats[k] + 1];
    }
    const res = solvePnPRansac(X, uv, n, cam, {
      maxErrorPx: o.absPoseMaxErrorPx,
      minInliers: o.absPoseMinNumInliers,
      confidence: 0.99999,
      maxIterations: 10000,
      minInlierRatio: o.absPoseMinInlierRatio,
      rng: this.rng,
    });
    if (!res || res.numInliers < o.absPoseMinNumInliers || res.numInliers < o.absPoseMinInlierRatio * n) return false;
    if (!poseIsUsable(res.pose, res.inlierMask, uv, n, o.absPoseMaxErrorPx)) return false;
    if (o.rotationCheck?.pose && !o.rotationCheck.pose(id, res.pose, this.rec)) {
      this.numRotationRejects++;
      return false;
    }
    this.rec.registerImage(id, res.pose);
    this.onRegistered(id);
    const added = new Set<number>();
    for (let k = 0; k < n; k++) {
      if (!res.inlierMask[k]) continue;
      const f = feats[k],
        pid = pids[k];
      if (img.point3D[f] >= 0 || added.has(pid) || !this.rec.points.has(pid)) continue;
      added.add(pid);
      this.rec.addObservation(pid, { image: id, feature: f });
      this.triangulator.addModifiedPoint(pid);
    }
    return true;
  }

  /** Triangulates the observations of a (registered) image. */
  triangulateImage(id: number): number {
    const t0 = now();
    const n = this.triangulator.triangulateImage(this.opts.triangulation, id);
    this.timings.triangulate += now() - t0;
    return n;
  }

  // ------------------------------------------------------------------ bundle adjustment

  private refineFlags(global: boolean, cameraId: number, imagesInBundle: number): BACameraRefine {
    const o = this.opts;
    const numReg = this.numRegPerCamera.get(cameraId) ?? 0;
    const enough = this.rec.numRegisteredImages >= o.minImagesForIntrinsics;
    if (!enough || (!global && imagesInBundle < numReg)) return { focal: false, principalPoint: false, distortion: false };
    return { focal: o.refineFocal, principalPoint: o.refinePrincipalPoint, distortion: o.refineDistortion };
  }

  /** Builds and solves a BA problem for the given configuration; returns the BA result and #observations. */
  private solveBundle(
    cfg: BundleConfig,
    loss: BALossType,
    lossScalePx: number,
    maxIterations: number,
    timeLimitMs: number,
    functionTolerance = 1e-6,
    initialLambda?: number,
    linearSolver: BALinearSolverType = 'auto',
    control?: BAControl,
  ): { result: BAResult | null; numObs: number } {
    const rec = this.rec;
    const imageSet = new Set(cfg.images);
    // pose set: config images + outside images observing variable points
    const poseSet = new Set(cfg.images);
    const variable = new Set<number>();
    for (const pid of cfg.variablePoints) {
      const p = rec.points.get(pid);
      if (!p) continue;
      variable.add(pid);
      for (const el of p.track) if (!poseSet.has(el.image) && rec.images.get(el.image)!.registered) poseSet.add(el.image);
    }
    const poseIds = [...poseSet].sort((x, y) => x - y);
    const poseIndex = new Map<number, number>();
    poseIds.forEach((id, k) => poseIndex.set(id, k));
    const camIndex = new Map<number, number>();
    const cameras: CameraIntrinsics[] = [];
    const cameraRefine: BACameraRefine[] = [];
    const camCount = new Map<number, number>();
    for (const id of cfg.images) {
      const cid = rec.images.get(id)!.cameraId;
      camCount.set(cid, (camCount.get(cid) ?? 0) + 1);
    }
    const imageCamera = new Uint32Array(poseIds.length);
    const poses: Pose[] = new Array(poseIds.length);
    const poseFixed = new Uint8Array(poseIds.length);
    const tmask = new Uint8Array(poseIds.length);
    let anyMask = false;
    poseIds.forEach((id, k) => {
      const img = rec.images.get(id)!;
      let ci = camIndex.get(img.cameraId);
      if (ci === undefined) {
        ci = cameras.length;
        camIndex.set(img.cameraId, ci);
        cameras.push(rec.cameras.get(img.cameraId)!);
        cameraRefine.push(cfg.refine(img.cameraId));
      }
      imageCamera[k] = ci;
      poses[k] = img.pose;
      poseFixed[k] = !imageSet.has(id) || cfg.constantPoses.has(id) ? 1 : 0;
      const m = cfg.translationMask.get(id);
      if (m) {
        tmask[k] = m;
        anyMask = true;
      }
    });
    // observations (typed scratch buffers; point id → slot table, reset afterwards)
    this.ensurePointTables(rec);
    const slot = this.pointSlot,
      cls = this.pointClass;
    let nX = 0,
      nO = 0;
    const getPoint = (pid: number): number => {
      let k = slot[pid] - 1;
      if (k < 0) {
        k = nX++;
        slot[pid] = k + 1;
        if (this.pointIdBuf.length <= k) this.growPointIds();
        this.pointIdBuf[k] = pid;
        this.obsCountBuf[k] = 0;
      }
      return k;
    };
    const pushObs = (pi: number, k: number, u: number, v: number): void => {
      if (nO >= this.obsPoseBuf.length) this.growObs();
      this.obsPoseBuf[nO] = pi;
      this.obsPointBuf[nO] = k;
      this.obsUVBuf[2 * nO] = u;
      this.obsUVBuf[2 * nO + 1] = v;
      this.obsCountBuf[k]++;
      nO++;
    };
    // Points with observations outside the bundle are held constant; their
    // observations in the bundle images are subsampled to maxConstObsPerImage.
    // cls[pid]: 0 unknown, 1 constant, 2 free / variable
    const touched: number[] = [];
    const constOf = (pid: number): boolean => {
      let c = cls[pid];
      if (c === 0) {
        c = 2;
        if (!variable.has(pid))
          for (const el of rec.points.get(pid)!.track)
            if (!imageSet.has(el.image)) {
              c = 1;
              break;
            }
        cls[pid] = c;
        touched.push(pid);
      }
      return c === 1;
    };
    const capped = cfg.trackCap !== undefined && cfg.trackCap < Infinity;
    if (capped) {
      // global BA with long tracks subsampled: every point is free, all its images are in the problem
      const cap = Math.max(2, cfg.trackCap!);
      for (const p of rec.points.values()) {
        const tr = p.track;
        const L = tr.length;
        const stride = L > cap ? Math.ceil(L / cap) : 1;
        let k = -1;
        for (let i = 0; i < L; i += stride) {
          const el = tr[i];
          const pi = poseIndex.get(el.image);
          if (pi === undefined) continue;
          if (k < 0) k = getPoint(p.id);
          const kp = rec.images.get(el.image)!.keypoints!;
          pushObs(pi, k, kp[4 * el.feature], kp[4 * el.feature + 1]);
        }
      }
    }
    for (const id of capped ? [] : cfg.images) {
      const img = rec.images.get(id)!;
      const kp = img.keypoints!;
      const pi = poseIndex.get(id)!;
      const p3 = img.point3D;
      let stride = 1;
      if (cfg.maxConstObsPerImage < Infinity) {
        let nConst = 0;
        for (let f = 0; f < p3.length; f++) if (p3[f] >= 0 && constOf(p3[f])) nConst++;
        stride = Math.max(1, Math.ceil(nConst / Math.max(1, cfg.maxConstObsPerImage)));
      }
      let cc = 0;
      for (let f = 0; f < p3.length; f++) {
        const pid = p3[f];
        if (pid < 0) continue;
        if (stride > 1 && constOf(pid) && cc++ % stride !== 0) continue;
        pushObs(pi, getPoint(pid), kp[4 * f], kp[4 * f + 1]);
      }
    }
    for (const pid of variable) {
      const p = rec.points.get(pid)!;
      const k = getPoint(pid);
      for (const el of p.track) {
        if (imageSet.has(el.image)) continue;
        const pi = poseIndex.get(el.image);
        if (pi === undefined) continue;
        const kp = rec.images.get(el.image)!.keypoints!;
        pushObs(pi, k, kp[4 * el.feature], kp[4 * el.feature + 1]);
      }
    }
    for (const pid of touched) cls[pid] = 0;
    const ids = this.pointIdBuf;
    for (let k = 0; k < nX; k++) slot[ids[k]] = 0;
    if (nO === 0 || nX === 0) return { result: null, numObs: 0 };
    const points = new Float64Array(3 * nX);
    const pointFixed = new Uint8Array(nX);
    for (let k = 0; k < nX; k++) {
      const p = rec.points.get(ids[k])!;
      points[3 * k] = p.xyz[0];
      points[3 * k + 1] = p.xyz[1];
      points[3 * k + 2] = p.xyz[2];
      if (!capped && !variable.has(ids[k]) && this.obsCountBuf[k] < p.track.length) pointFixed[k] = 1;
    }
    const problem: BAProblem = {
      cameras,
      cameraRefine,
      poses,
      imageCamera,
      poseFixed,
      translationFixedMask: anyMask ? tmask : undefined,
      points,
      pointFixed,
      obsPose: this.obsPoseBuf.subarray(0, nO),
      obsPoint: this.obsPointBuf.subarray(0, nO),
      obsUV: this.obsUVBuf.subarray(0, 2 * nO),
    };
    if (cfg.focalPriors && cfg.focalPriors.size) {
      problem.focalPriors = [];
      for (const [cid, fp] of cfg.focalPriors) {
        const ci = camIndex.get(cid);
        if (ci !== undefined) problem.focalPriors.push({ camera: ci, f: fp.f, sigma: fp.sigma });
      }
    }
    const ppp = this.opts.principalPointPriors;
    if (ppp && ppp.size) {
      for (const [cid, pp] of ppp) {
        const ci = camIndex.get(cid);
        if (ci !== undefined && cameraRefine[ci].principalPoint) (problem.principalPointPriors ??= []).push({ camera: ci, cx: pp.cx, cy: pp.cy, sigma: pp.sigma });
      }
    }
    const result = bundleAdjust(problem, {
      loss,
      lossScalePx,
      maxIterations,
      timeLimitMs,
      functionTolerance,
      parameterTolerance: 0,
      initialLambda,
      linearSolver,
      cancel: control?.cancel,
      onIteration: control?.onIteration,
    });
    for (let k = 0; k < nX; k++) {
      if (pointFixed[k]) continue;
      const p = rec.points.get(ids[k])!;
      p.xyz[0] = points[3 * k];
      p.xyz[1] = points[3 * k + 1];
      p.xyz[2] = points[3 * k + 2];
    }
    return { result, numObs: nO };
  }

  // scratch for solveBundle (grow-only)
  private pointSlot = new Int32Array(0);
  private pointClass = new Uint8Array(0);
  private pointIdBuf = new Int32Array(1024);
  private obsCountBuf = new Int32Array(1024);
  private obsPoseBuf = new Uint32Array(16384);
  private obsPointBuf = new Uint32Array(16384);
  private obsUVBuf = new Float64Array(32768);

  private ensurePointTables(rec: Reconstruction): void {
    let maxId = 0;
    for (const id of rec.points.keys()) if (id > maxId) maxId = id;
    if (this.pointSlot.length > maxId) return;
    const n = Math.max(1024, 2 * (maxId + 1));
    this.pointSlot = new Int32Array(n);
    this.pointClass = new Uint8Array(n);
  }

  private growPointIds(): void {
    const n = 2 * this.pointIdBuf.length;
    const a = new Int32Array(n),
      b = new Int32Array(n);
    a.set(this.pointIdBuf);
    b.set(this.obsCountBuf);
    this.pointIdBuf = a;
    this.obsCountBuf = b;
  }

  private growObs(): void {
    const n = 2 * this.obsPoseBuf.length;
    const a = new Uint32Array(n),
      b = new Uint32Array(n),
      c = new Float64Array(2 * n);
    a.set(this.obsPoseBuf);
    b.set(this.obsPointBuf);
    c.set(this.obsUVBuf);
    this.obsPoseBuf = a;
    this.obsPointBuf = b;
    this.obsUVBuf = c;
  }

  /**
   * Images sharing the most points with `id` (COLMAP FindLocalBundle: prefer images
   * with sufficient triangulation angle, relaxing the angle / overlap thresholds).
   */
  findLocalBundle(id: number): number[] {
    const img = this.rec.images.get(id)!;
    const shared = new Map<number, number>();
    const pointSet = new Set<number>();
    const p3 = img.point3D;
    for (let f = 0; f < p3.length; f++) {
      const pid = p3[f];
      if (pid < 0) continue;
      pointSet.add(pid);
      for (const el of this.rec.points.get(pid)!.track) if (el.image !== id) shared.set(el.image, (shared.get(el.image) ?? 0) + 1);
    }
    const overlapping = [...shared.entries()].sort((x, y) => y[1] - x[1] || x[0] - y[0]);
    const numEff = Math.min(this.opts.localBANumImages, overlapping.length);
    if (overlapping.length === numEff) return overlapping.map((e) => e[0]);
    const C = this.rec.imageCenter(id);
    const minAngle = this.opts.localBAMinTriAngleDeg * DEG;
    const np = img.numPoints3D;
    const thresholds: [number, number][] = [
      [minAngle, 0.6 * np],
      [minAngle / 1.5, 0.6 * np],
      [minAngle / 2, 0.5 * np],
      [minAngle / 2.5, 0.4 * np],
      [minAngle / 3, 0.3 * np],
      [minAngle / 4, 0.2 * np],
      [minAngle / 5, 0.1 * np],
      [minAngle / 6, 0.1 * np],
    ];
    const triAngles = new Float64Array(overlapping.length).fill(-1);
    const used = new Uint8Array(overlapping.length);
    const out: number[] = [];
    const C2 = new Float64Array(3);
    for (const [angleThr, overlapThr] of thresholds) {
      for (let k = 0; k < overlapping.length; k++) {
        if (overlapping[k][1] < overlapThr) break;
        if (used[k]) continue;
        if (triAngles[k] < 0) {
          const other = this.rec.images.get(overlapping[k][0])!;
          this.rec.imageCenter(other.id, C2);
          const angles: number[] = [];
          const q3 = other.point3D;
          for (let f = 0; f < q3.length; f++) {
            const pid = q3[f];
            if (pid < 0 || !pointSet.has(pid)) continue;
            const X = this.rec.points.get(pid)!.xyz;
            const ax = X[0] - C[0],
              ay = X[1] - C[1],
              az = X[2] - C[2];
            const bx = X[0] - C2[0],
              by = X[1] - C2[1],
              bz = X[2] - C2[2];
            const d = Math.sqrt((ax * ax + ay * ay + az * az) * (bx * bx + by * by + bz * bz));
            angles.push(d > 0 ? Math.acos(Math.max(-1, Math.min(1, (ax * bx + ay * by + az * bz) / d))) : 0);
          }
          triAngles[k] = percentileOf(angles, 75);
        }
        if (triAngles[k] >= angleThr) {
          out.push(overlapping[k][0]);
          used[k] = 1;
          if (out.length >= numEff) break;
        }
      }
      if (out.length >= numEff) break;
    }
    for (let k = 0; k < overlapping.length && out.length < numEff; k++) {
      if (!used[k]) {
        out.push(overlapping[k][0]);
        used[k] = 1;
      }
    }
    return out;
  }

  /**
   * Local BA around a newly registered image (COLMAP AdjustLocalBundle), then
   * merge / complete the adjusted tracks and filter the local points.
   */
  adjustLocalBundle(id: number, pointIds: number[], refineIntrinsics: boolean, timeLimitMs?: number): LocalBAReport {
    const t0 = now();
    const o = this.opts;
    const report: LocalBAReport = {
      numAdjustedObservations: 0,
      numMergedObservations: 0,
      numCompletedObservations: 0,
      numFilteredObservations: 0,
      ba: null,
    };
    const bundle = this.findLocalBundle(id);
    const tri = o.triangulation;
    if (bundle.length > 0) {
      const images = [id, ...bundle];
      const constantPoses = new Set<number>();
      const translationMask = new Map<number, number>();
      if (bundle.length === 1) {
        constantPoses.add(bundle[0]);
        const ax = largestRelTranslationAxis(this.rec.images.get(bundle[0])!.pose, this.rec.images.get(id)!.pose);
        translationMask.set(id, 1 << ax);
      } else {
        const i1 = bundle[bundle.length - 1],
          i2 = bundle[bundle.length - 2];
        constantPoses.add(i1);
        const ax = largestRelTranslationAxis(this.rec.images.get(i1)!.pose, this.rec.images.get(i2)!.pose);
        translationMask.set(i2, 1 << ax);
      }
      const camCount = new Map<number, number>();
      for (const iid of images) {
        const cid = this.rec.images.get(iid)!.cameraId;
        camCount.set(cid, (camCount.get(cid) ?? 0) + 1);
      }
      const variablePoints: number[] = [];
      for (const pid of pointIds) {
        const p = this.rec.points.get(pid);
        if (p && (p.error < 0 || p.track.length <= 15)) variablePoints.push(pid);
      }
      const { result, numObs } = this.solveBundle(
        {
          images,
          constantPoses,
          translationMask,
          variablePoints,
          refine: (cid) =>
            refineIntrinsics ? this.refineFlags(false, cid, camCount.get(cid) ?? 0) : { focal: false, principalPoint: false, distortion: false },
          maxConstObsPerImage: o.localBAMaxConstantObsPerImage,
        },
        o.localBALoss,
        o.localBALossScalePx,
        o.localBAMaxIterations,
        timeLimitMs ?? o.localBATimeLimitMs,
        o.localBAFunctionTolerance,
      );
      report.ba = result;
      report.numAdjustedObservations = numObs;
      if (result) this.timings.localBAIterations += result.iterations;
      this.triangulator.clearCaches();
      report.numMergedObservations = this.triangulator.mergeTracks(tri, variablePoints);
      report.numCompletedObservations = this.triangulator.completeTracks(tri, variablePoints);
      report.numCompletedObservations += this.triangulator.completeImage(tri, id);
    }
    const ft = now();
    // COLMAP filters every observation of every point seen by the bundle images; with long
    // tracks that is most of the model. Only the bundle poses and the variable points moved:
    // check the observations in the bundle images, and the modified points completely.
    report.numFilteredObservations = this.rec.filterObservationsInImages(o.filterMaxReprojErrorPx, [id, ...bundle]);
    report.numFilteredObservations += this.rec.filterPoints3D(o.filterMaxReprojErrorPx, o.filterMinTriAngleDeg * DEG, pointIds);
    this.timings.filter += now() - ft;
    this.timings.localBA += now() - t0;
    this.timings.numLocalBA++;
    return report;
  }

  /** Repeated local BA while it changes > localBAMaxRefinementChange of the observations. */
  iterativeLocalRefinement(id: number, timeLimitMs?: number): void {
    const o = this.opts;
    const tEnd = timeLimitMs !== undefined ? now() + timeLimitMs : Infinity;
    for (let i = 0; i < o.localBAMaxRefinements; i++) {
      const remaining = tEnd - now();
      if (remaining <= 0) break;
      const rep = this.adjustLocalBundle(id, this.triangulator.getModifiedPoints(), i === 0, Math.min(o.localBATimeLimitMs, remaining));
      const changed =
        (rep.numMergedObservations + rep.numCompletedObservations + rep.numFilteredObservations) /
        Math.max(1, rep.numAdjustedObservations);
      if (changed < o.localBAMaxRefinementChange) break;
    }
    this.triangulator.clearModifiedPoints();
  }

  /**
   * Global BA over all registered images (gauge: first image fixed, one translation
   * component of the second). Observations behind their camera are removed first
   * unless `filterNegativeDepth` is false (continuation slices of one BA).
   */
  adjustGlobalBundle(
    timeLimitMs?: number,
    maxIterations?: number,
    initialLambda?: number,
    trackCap = Infinity,
    control?: BAControl,
    filterNegativeDepth = true,
  ): BAResult | null {
    const t0 = now();
    const o = this.opts;
    const reg = this.rec.registeredImageIds();
    if (reg.length < 2) return null;
    if (filterNegativeDepth) this.rec.filterObservationsWithNegativeDepth();
    const constantPoses = new Set<number>([reg[0]]);
    const ax = largestRelTranslationAxis(this.rec.images.get(reg[0])!.pose, this.rec.images.get(reg[1])!.pose);
    const translationMask = new Map<number, number>([[reg[1], 1 << ax]]);
    const { result } = this.solveBundle(
      {
        images: reg,
        constantPoses,
        translationMask,
        variablePoints: [],
        refine: (cid) => this.refineFlags(true, cid, 0),
        maxConstObsPerImage: Infinity,
        trackCap,
      },
      o.globalBALoss,
      o.globalBALossScalePx,
      maxIterations ?? o.globalBAMaxIterations,
      timeLimitMs ?? o.globalBATimeLimitMs,
      1e-6,
      initialLambda,
      o.globalBALinearSolver === 'auto' && reg.length > 150 && this.rec.meanTrackLength() > 8 ? 'pcg_schur' : o.globalBALinearSolver,
      control,
    );
    this.timings.globalBA += now() - t0;
    this.timings.numGlobalBA++;
    if (result) this.timings.globalBAIterations += result.iterations;
    return result;
  }

  /**
   * Global BA with per-camera focal priors and ONLY the focal of those cameras free (principal
   * point / distortion fixed; other cameras constant): e.g. per-image cameras of a final refinement
   * whose focal lengths breathe around a shared value, `priors` = camera id → shared focal,
   * σ = sigmaRel · focal. Same gauge as `adjustGlobalBundle`; trivial loss; all observations.
   */
  async adjustGlobalBundleFocalPriors(priors: Map<number, number>, sigmaRel: number, cancel?: CancelToken): Promise<BAResult | null> {
    const t0 = now();
    const o = this.opts;
    const reg = this.rec.registeredImageIds();
    if (reg.length < 2) return null;
    this.rec.filterObservationsWithNegativeDepth();
    const constantPoses = new Set<number>([reg[0]]);
    const ax = largestRelTranslationAxis(this.rec.images.get(reg[0])!.pose, this.rec.images.get(reg[1])!.pose);
    const translationMask = new Map<number, number>([[reg[1], 1 << ax]]);
    const fp = new Map<number, { f: number; sigma: number }>();
    for (const [cid, f] of priors) fp.set(cid, { f, sigma: Math.max(1e-9, sigmaRel * f) });
    const free: BACameraRefine = { focal: true, principalPoint: false, distortion: false };
    const none: BACameraRefine = { focal: false, principalPoint: false, distortion: false };
    await yieldToEventLoop();
    const { result } = this.solveBundle(
      { images: reg, constantPoses, translationMask, variablePoints: [], refine: (cid) => (fp.has(cid) ? free : none), maxConstObsPerImage: Infinity, focalPriors: fp },
      'trivial',
      o.globalBALossScalePx,
      o.globalBAMaxIterations,
      o.globalBATimeLimitMs,
      1e-6,
      undefined,
      o.globalBALinearSolver === 'auto' && reg.length > 150 && this.rec.meanTrackLength() > 8 ? 'pcg_schur' : o.globalBALinearSolver,
      { cancel },
    );
    if (cancel?.cancelled) throw new SfMCancelledError();
    this.timings.globalBA += now() - t0;
    this.timings.numGlobalBA++;
    if (result) this.timings.globalBAIterations += result.iterations;
    return result;
  }

  completeAndMergeTracks(): number {
    const tri = this.opts.triangulation;
    const n = this.triangulator.completeAllTracks(tri);
    return n + this.triangulator.mergeAllTracks(tri);
  }

  retriangulate(): number {
    const t0 = now();
    const n = this.triangulator.retriangulate(this.opts.triangulation);
    this.timings.retriangulate += now() - t0;
    return n;
  }

  filterPoints(): number {
    const t0 = now();
    const o = this.opts;
    const n = this.rec.filterAllPoints3D(o.filterMaxReprojErrorPx, o.filterMinTriAngleDeg * DEG);
    this.timings.filter += now() - t0;
    return n;
  }

  /** Deregisters images with bogus intrinsics or too few points (only once ≥ 20 images are registered). */
  filterImages(): number[] {
    const o = this.opts;
    if (this.rec.numRegisteredImages < 20) return [];
    const ids = this.rec.filterImages(o.minFocalLengthRatio, o.maxFocalLengthRatio, o.maxExtraParam);
    for (const id of ids) {
      this.onDeregistered(id);
      this.filteredImages.add(id);
    }
    if (o.minImagePoints3D > 1) {
      for (const id of this.rec.registeredImageIds()) {
        if (this.rec.images.get(id)!.numPoints3D < o.minImagePoints3D) {
          this.deregisterImage(id);
          this.filteredImages.add(id);
          ids.push(id);
        }
      }
    }
    return ids;
  }

  /**
   * COLMAP IterativeGlobalRefinement: complete/merge + re-triangulate, then global BA
   * → complete/merge → filter, repeated while > globalBAMaxRefinementChange of the
   * observations change; finally filter images.
   */
  async iterativeGlobalRefinement(timeLimitMs?: number, cancel?: CancelToken, full = true): Promise<void> {
    const o = this.opts;
    const tEnd = timeLimitMs !== undefined ? now() + timeLimitMs : Infinity;
    this.completeAndMergeTracks();
    this.retriangulate();
    await yieldToEventLoop();
    const maxRef = full ? o.globalBAMaxRefinements : Math.min(o.globalBAMaxRefinements, o.intermediateGlobalBAMaxRefinements);
    for (let i = 0; i < maxRef; i++) {
      if (cancel?.cancelled) throw new SfMCancelledError();
      if (tEnd - now() <= 0) break;
      const numObs = Math.max(1, this.rec.numObservations);
      await this.adjustGlobalBundleSliced(tEnd, cancel, full);
      let changed = this.completeAndMergeTracks();
      changed += this.filterPoints();
      await yieldToEventLoop();
      if (changed / numObs < o.globalBAMaxRefinementChange) break;
    }
    this.filterImages();
  }

  /**
   * Global BA in slices of ≈ `globalBASliceMs` with event-loop yields in between, up to
   * globalBAMaxIterations in total and within `globalBATimeLimitMs` (and `tEnd`).
   *
   * Slices end only right after an ACCEPTED LM step: the solver's state is then fully
   * described by its parameters and damping λ (the damping growth factor is reset by
   * an accepted step), so the next slice — started from the written-back parameters
   * with the carried λ — continues exactly as one uninterrupted solve would. Where the
   * slices fall (which depends on CPU speed / load) therefore does not change the
   * result (deterministic runs). A slice is cut regardless after 3 × globalBASliceMs.
   */
  async adjustGlobalBundleSliced(tEnd = Infinity, cancel?: CancelToken, full = true): Promise<void> {
    const o = this.opts;
    const deadline = Math.min(tEnd, now() + o.globalBATimeLimitMs);
    let used = 0;
    let lambda: number | undefined;
    let lastCost = NaN;
    let first = true;
    while (used < o.globalBAMaxIterations) {
      const t0 = now();
      const remaining = deadline - t0;
      if (remaining <= 0) break;
      const sliceEnd = t0 + o.globalBASliceMs,
        hardEnd = t0 + 3 * o.globalBASliceMs;
      const stop = { cancelled: false };
      let prev = lastCost,
        sliced = false;
      const res = this.adjustGlobalBundle(
        remaining,
        o.globalBAMaxIterations - used,
        lambda,
        full ? Infinity : o.globalBAMaxTrackLength,
        {
          cancel: stop,
          onIteration: (_it, cost) => {
            // an accepted step strictly lowers the cost; a rejected one leaves it unchanged
            const accepted = cost < prev;
            prev = cost;
            if (cancel?.cancelled) {
              stop.cancelled = true;
              return;
            }
            const t = now();
            if ((accepted && t >= sliceEnd) || t >= hardEnd) {
              stop.cancelled = true;
              sliced = true;
            }
          },
        },
        first,
      );
      first = false;
      if (!res) break;
      used += Math.max(1, res.iterations);
      lastCost = res.finalCost;
      // continue the next slice with the adapted damping (LM state carried over)
      lambda = res.finalLambda > 0 && Number.isFinite(res.finalLambda) ? res.finalLambda : undefined;
      if (cancel?.cancelled) throw new SfMCancelledError();
      if (res.termination !== 'cancelled' || !sliced) break;
      await yieldToEventLoop();
      if (cancel?.cancelled) throw new SfMCancelledError();
    }
  }

  /** Synchronous variant (for time-bounded live use); `full = false` caps long tracks. */
  iterativeGlobalRefinementSync(timeLimitMs?: number, full = true): void {
    const o = this.opts;
    const tEnd = timeLimitMs !== undefined ? now() + timeLimitMs : Infinity;
    this.completeAndMergeTracks();
    this.retriangulate();
    const maxRef = full ? o.globalBAMaxRefinements : Math.min(o.globalBAMaxRefinements, o.intermediateGlobalBAMaxRefinements);
    for (let i = 0; i < maxRef; i++) {
      const remaining = tEnd - now();
      if (remaining <= 0) break;
      const numObs = Math.max(1, this.rec.numObservations);
      this.adjustGlobalBundle(Math.min(o.globalBATimeLimitMs, remaining), undefined, undefined, full ? Infinity : o.globalBAMaxTrackLength);
      let changed = this.completeAndMergeTracks();
      changed += this.filterPoints();
      if (changed / numObs < o.globalBAMaxRefinementChange) break;
    }
    this.filterImages();
  }

  // ------------------------------------------------------------------ driver

  /**
   * Initialises from the given pair and grows the model (COLMAP ReconstructSubModel).
   * Returns false if the initial pair failed.
   */
  async reconstructFrom(
    a: number,
    b: number,
    geom: InitPairGeometry,
    minPointAngleDeg: number,
    progress?: ProgressCallback,
    cancel?: CancelToken,
    totalImages = this.rec.images.size,
  ): Promise<boolean> {
    const o = this.opts;
    const t0 = now();
    this.registerInitialPair(a, b, geom, minPointAngleDeg);
    this.adjustGlobalBundle();
    this.filterPoints();
    this.triangulator.clearModifiedPoints();
    this.timings.init += now() - t0;
    if (this.rec.numRegisteredImages < 2 || this.rec.numPoints < Math.min(o.absPoseMinNumInliers, 15)) return false;
    let prevImages = this.rec.numRegisteredImages,
      prevPoints = this.rec.numPoints;
    let regSuccess = true,
      prevRegSuccess = true;
    let lastYield = now();
    const maybeYield = async (): Promise<void> => {
      if (now() - lastYield >= o.yieldIntervalMs) {
        await yieldToEventLoop();
        lastYield = now();
      }
      if (cancel?.cancelled) throw new SfMCancelledError();
    };
    while (regSuccess) {
      await maybeYield();
      regSuccess = false;
      const next = this.findNextImages();
      if (next.length === 0) break;
      for (let trial = 0; trial < next.length; trial++) {
        const id = next[trial];
        if (this.registerNextImage(id)) {
          this.triangulateImage(id);
          await maybeYield();
          this.iterativeLocalRefinement(id);
          await maybeYield();
          const nImg = this.rec.numRegisteredImages,
            nPts = this.rec.numPoints;
          if (
            nImg >= o.globalBAImagesRatio * prevImages ||
            nImg >= o.globalBAImagesFreq + prevImages ||
            nPts >= o.globalBAPointsRatio * prevPoints ||
            nPts >= o.globalBAPointsFreq + prevPoints
          ) {
            await this.iterativeGlobalRefinement(undefined, cancel, false);
            prevImages = this.rec.numRegisteredImages;
            prevPoints = this.rec.numPoints;
          }
          progress?.({
            stage: 'sfm',
            progress: Math.min(1, this.rec.numRegisteredImages / Math.max(1, totalImages)),
            message: `registered ${this.rec.numRegisteredImages}/${totalImages} images, ${this.rec.numPoints} points`,
          });
          regSuccess = true;
          break;
        } else if (trial >= 30 && this.rec.numRegisteredImages < o.minModelSize) break;
        await maybeYield();
      }
      if (this.numSharedRegImages >= o.maxModelOverlap) break;
      if (!regSuccess && prevRegSuccess) {
        regSuccess = true;
        prevRegSuccess = false;
        await this.iterativeGlobalRefinement(undefined, cancel, false);
        prevImages = this.rec.numRegisteredImages;
        prevPoints = this.rec.numPoints;
      } else prevRegSuccess = regSuccess;
    }
    // final global refinement with all observations (intermediate ones cap long tracks)
    if (this.rec.numRegisteredImages >= 2) await this.iterativeGlobalRefinement(undefined, cancel, true);
    return true;
  }
}

/** Options accepted by `runIncrementalSfM` (mapper options + diagnostics hook). */
export interface RunSfMOptions extends MapperOptions {
  /** Called with per-model timings (diagnostics). */
  onModel?: (rec: Reconstruction, timings: MapperTimings) => void;
}

/**
 * Full incremental SfM on the database (COLMAP IncrementalMapperController):
 * builds the correspondence graph, then repeatedly finds an initial pair (with
 * progressively relaxed thresholds) and grows a model until no images are left or
 * no initial pair is found. Yields to the event loop periodically. Returns the
 * models, largest first (point colours = mean feature colour of the track; point
 * errors updated). Throws `SfMCancelledError` (name 'CancelledError') on cancel.
 */
export async function runIncrementalSfM(
  db: FeatureDatabase,
  opts: Partial<RunSfMOptions> = {},
  progress?: ProgressCallback,
  cancel?: CancelToken,
): Promise<Reconstruction[]> {
  const o = resolveMapperOptions(opts);
  const graph = buildCorrespondenceGraph(db, o.minNumMatches);
  const frameIds = db.frameIds();
  const shared = createMapperSharedState();
  const models: Reconstruction[] = [];
  // [min inliers, min median triangulation angle (deg), max forward motion, last resort]
  const levels: [number, number, number, boolean][] = [];
  {
    let inl = o.initMinNumInliers,
      ang = o.initMinTriAngleDeg;
    const floorI = Math.min(o.initMinNumInliersFloor, inl),
      floorA = Math.min(o.initMinTriAngleFloorDeg, ang);
    const fwd = o.initMaxForwardMotion;
    levels.push([inl, ang, fwd, false]);
    while (inl > floorI || ang > floorA) {
      if (inl > floorI) {
        inl = Math.max(floorI, Math.floor(inl / 2));
        levels.push([inl, ang, fwd, false]);
      }
      if (ang > floorA) {
        ang = Math.max(floorA, ang / 2);
        levels.push([inl, ang, fwd, false]);
      }
    }
    const fa = Math.min(floorA, o.initFallbackMinTriAngleDeg),
      ff = Math.max(fwd, o.initFallbackMaxForwardMotion);
    if (o.initFallback && (fa < floorA || ff > fwd)) levels.push([floorI, fa, ff, true]);
  }
  const minModelSize = Math.max(2, Math.min(Math.floor(0.8 * frameIds.length), o.minModelSize));
  progress?.({ stage: 'sfm', progress: 0, message: `${frameIds.length} images, ${graph.numPairs} pairs` });
  let done = false;
  for (const [minInl, minAng, maxFwd, lastResort] of levels) {
    if (done) break;
    // the last-resort level only runs when the regular levels found no model at all
    if (lastResort && models.length > 0) break;
    for (;;) {
      if (cancel?.cancelled) throw new SfMCancelledError();
      const numRegAnywhere = frameIds.filter((id) => (shared.registrations.get(id) ?? 0) > 0).length;
      if (models.length >= o.maxNumModels || numRegAnywhere >= frameIds.length - 1) {
        done = true;
        break;
      }
      const rec = new Reconstruction();
      const mapper = new IncrementalMapper(db, graph, rec, o, shared);
      for (const id of frameIds) mapper.ensureImage(id);
      rec.attachGraph(graph);
      // chunked search with yields (each re-estimation costs ~10–30 ms)
      let init: ReturnType<IncrementalMapper['findInitialPair']> = null;
      for (let spent = 0; spent < o.initMaxPairEvaluations; spent += INIT_CHUNK) {
        init = mapper.findInitialPair(minInl, minAng, undefined, Math.min(INIT_CHUNK, o.initMaxPairEvaluations - spent), maxFwd);
        if (init || !mapper.initSearchExhausted) break;
        await yieldToEventLoop();
        if (cancel?.cancelled) throw new SfMCancelledError();
      }
      if (!init) break;
      const ok = await mapper.reconstructFrom(
        init.a,
        init.b,
        init.geom,
        Math.max(o.filterMinTriAngleDeg, 0.25 * minAng),
        progress,
        cancel,
        frameIds.length,
      );
      const size = rec.numRegisteredImages;
      if (ok && size >= 2 && (size >= minModelSize || models.length === 0)) {
        extractColors(rec, db);
        rec.updatePointErrors();
        models.push(rec);
        opts.onModel?.(rec, mapper.timings);
      } else {
        // discard: release the registrations of this attempt
        for (const id of rec.registeredImageIds()) mapper.deregisterImage(id);
      }
      if (!o.multipleModels && models.length > 0) {
        done = true;
        break;
      }
      await yieldToEventLoop();
    }
  }
  // a tiny first model is dropped if larger ones exist
  const out = models.filter((m) => m.numRegisteredImages >= minModelSize || models.length === 1);
  out.sort((x, y) => y.numRegisteredImages - x.numRegisteredImages || y.numPoints - x.numPoints);
  progress?.({ stage: 'sfm', progress: 1, message: `${out.length} model(s)` });
  return out;
}
