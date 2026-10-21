/**
 * Camera-model refinements of refine (ENGINE.md "Refine"): gyro-based rolling-shutter correction
 * of the keypoints and per-image focal lengths (focus breathing).
 *
 * Rolling shutter (src/sfm/rolling-shutter.ts): the readout direction / time is estimated on the
 * reconstruction (hypotheses axis × readout, robust reprojection after a short BA of every registered
 * image and a subsample of the points; gyro rotation + translation, the camera velocities from the
 * neighbouring keyframes and the depths from the reconstruction) and, when it lowers the cost
 * significantly, every keyframe's keypoints are corrected IN PLACE (the FeatureSet, the live / offline reconstructions and their clones share
 * the arrays) from the originals kept in `CaptureState.rsOriginal`; estimates are always absolute
 * (from the originals), so refine can run again. The model is persisted in the engine state and
 * reported in `stats` (canonical image and device frame). A model known in advance (e.g. from an
 * earlier capture on the same device, `EngineConfig.rollingShutter`) is applied live to every new
 * keyframe (`correctKeyframeRS`).
 *
 * Per-image focal (focus breathing, src/ba focal priors): every registered image gets its own
 * camera whose focal is free but tied to the shared focal by a Gaussian prior (σ = 0.5 % by
 * default); principal point and distortion stay shared and fixed. Kept only when the mean
 * reprojection error drops significantly AND the focal lengths are smooth in time (focus breathing
 * follows the lagged focus distance; jumps mean they absorb another error, e.g. rolling shutter).
 */
import type { CameraIntrinsics, CancelToken } from '../core/types';
import type { CorrespondenceGraph } from '../sfm/correspondence-graph';
import type { FeatureDatabase } from '../sfm/database';
import type { GyroReference } from '../sfm/gyro';
import { IncrementalMapper, type MapperOptions } from '../sfm/mapper';
import type { Reconstruction } from '../sfm/reconstruction';
import {
  correctKeypointsRS,
  rsModelToDevice,
  RSBundleCost,
  searchRollingShutter,
  searchRollingShutterBatched,
  type RollingShutterEstimate,
  type RollingShutterEstimateOptions,
  type RollingShutterModel,
  type RSTranslation,
} from '../sfm/rolling-shutter';
import type { CpuPool } from './cpu-pool';
import type { CaptureState } from './live';
import type { SparseModel, SparseRollingShutter } from './types';

/**
 * Half window (ms) of the gyro angular velocity used for the rolling-shutter correction: the SAME
 * for the estimation (hypothesis costs), the keypoint correction and the dense images, so that the
 * readout estimated is the one applied (a 22 ms readout spans ±11 ms around the stamp).
 */
export const RS_OMEGA_HALF_WINDOW_MS = 12;

/** Original (uncorrected) keypoints of a keyframe: the saved copy, else the current array. */
export function originalKeypoints(st: CaptureState, id: number): Float32Array | null {
  return st.rsOriginal.get(id) ?? st.db.features.get(id)?.keypoints ?? null;
}

/**
 * Correct one keyframe's keypoints in place for `model` from its originals (saved on first use);
 * readout 0 (or no gyro coverage) restores the originals. `trans` adds the translational term
 * (velocity + inverse depths from a reconstruction). Returns true when corrected.
 */
export function correctKeyframeRS(
  st: CaptureState,
  id: number,
  cam: CameraIntrinsics,
  gyro: GyroReference,
  model: RollingShutterModel | null,
  trans: RSTranslation | null = null,
): boolean {
  const fs = st.db.features.get(id);
  const r = st.records.get(id);
  if (!fs || !r) return false;
  let orig = st.rsOriginal.get(id);
  const w = model && model.readoutMs !== 0 ? gyro.angularVelocityCam(r.t, RS_OMEGA_HALF_WINDOW_MS) : null;
  if (!model || !w) {
    if (orig) fs.keypoints.set(orig);
    st.rsOmega.delete(id);
    st.rsTranslated.delete(id);
    return false;
  }
  if (!orig) {
    orig = fs.keypoints.slice();
    st.rsOriginal.set(id, orig);
  }
  correctKeypointsRS(orig, cam, w, model, fs.keypoints, trans);
  st.rsOmega.set(id, Float64Array.from(w));
  if (trans) st.rsTranslated.add(id);
  else st.rsTranslated.delete(id);
  return true;
}

/**
 * Correct every keyframe for `model` (null / 0 restores the originals), with the translational term
 * of `transFor` where it has one. Returns the number corrected.
 */
export function applyRollingShutter(
  st: CaptureState,
  camFor: (id: number) => CameraIntrinsics | undefined,
  gyro: GyroReference,
  model: RollingShutterModel | null,
  transFor: ((id: number) => RSTranslation | null) | null = null,
): number {
  let n = 0;
  for (const id of st.order) {
    const cam = camFor(id);
    if (!cam) continue;
    if (correctKeyframeRS(st, id, cam, gyro, model, model && transFor ? transFor(id) : null)) n++;
  }
  st.rollingShutter = model && model.readoutMs !== 0 ? { ...model } : null;
  return n;
}

/**
 * World velocity of each registered image's camera centre turned into its camera frame (scene units
 * / s): central difference of the registered images nearest in time before and after (one-sided at
 * the ends and across gaps), neighbours at most `maxGapMs` away. Keyframes are 0.2–1 s apart: the
 * smooth part of the motion (an orbit, a walk), which is what moves the image during a readout.
 */
export function cameraVelocities(rec: Reconstruction, timeOf: (id: number) => number | undefined, maxGapMs = 3000): Map<number, Float64Array> {
  const items: { id: number; t: number; C: Float64Array; R: ArrayLike<number> }[] = [];
  for (const id of rec.registeredImageIds()) {
    const t = timeOf(id);
    const p = rec.images.get(id)?.pose;
    if (t === undefined || !Number.isFinite(t) || !p) continue;
    const R = p.R,
      tt = p.t;
    const C = Float64Array.of(-(R[0] * tt[0] + R[3] * tt[1] + R[6] * tt[2]), -(R[1] * tt[0] + R[4] * tt[1] + R[7] * tt[2]), -(R[2] * tt[0] + R[5] * tt[1] + R[8] * tt[2]));
    items.push({ id, t, C, R });
  }
  items.sort((a, b) => a.t - b.t);
  const out = new Map<number, Float64Array>();
  for (let k = 0; k < items.length; k++) {
    const a = k > 0 && items[k].t - items[k - 1].t <= maxGapMs && items[k].t > items[k - 1].t ? items[k - 1] : items[k];
    const b = k + 1 < items.length && items[k + 1].t - items[k].t <= maxGapMs && items[k + 1].t > items[k].t ? items[k + 1] : items[k];
    const dt = (b.t - a.t) / 1000;
    if (!(dt > 0)) continue;
    const vw = [0, 1, 2].map((c) => (b.C[c] - a.C[c]) / dt);
    const R = items[k].R;
    out.set(items[k].id, Float64Array.of(R[0] * vw[0] + R[1] * vw[1] + R[2] * vw[2], R[3] * vw[0] + R[4] * vw[1] + R[5] * vw[2], R[6] * vw[0] + R[7] * vw[1] + R[8] * vw[2]));
  }
  return out;
}

/** Inverse depth (1 / scene units) of each feature of a registered image that observes a 3-D point (NaN otherwise) and their median. */
export function inverseDepths(rec: Reconstruction, id: number): { inv: Float64Array; median: number } | null {
  const img = rec.images.get(id);
  if (!img || !img.registered) return null;
  const R = img.pose.R,
    t = img.pose.t;
  const inv = new Float64Array(img.point3D.length).fill(NaN);
  const vals: number[] = [];
  for (let f = 0; f < img.point3D.length; f++) {
    const p = img.point3D[f] >= 0 ? rec.points.get(img.point3D[f]) : undefined;
    if (!p) continue;
    const X = p.xyz;
    const z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
    if (z > 0) {
      inv[f] = 1 / z;
      vals.push(1 / z);
    }
  }
  if (!vals.length) return null;
  vals.sort((a, b) => a - b);
  return { inv, median: vals[vals.length >> 1] };
}

/**
 * Translational rolling-shutter term of every registered image of `rec` (velocity from the
 * neighbouring camera centres, inverse depths from its points; keypoints without a point use the
 * image's median inverse depth).
 */
export function rsTranslations(rec: Reconstruction, timeOf: (id: number) => number | undefined): Map<number, RSTranslation> {
  const vel = cameraVelocities(rec, timeOf);
  const out = new Map<number, RSTranslation>();
  for (const [id, v] of vel) {
    const d = inverseDepths(rec, id);
    if (d) out.set(id, { velocity: v, invDepth: d.inv, defaultInvDepth: d.median });
  }
  return out;
}

// ------------------------------------------------------------------------------------------
// keypoints consistent with a sparse model (exports)

/** Per image of a sparse model: its observations (feature index, point index), built once per model. */
const modelObsCache = new WeakMap<SparseModel, Map<number, { features: Uint32Array; points: Uint32Array }>>();

function modelObservations(model: SparseModel): Map<number, { features: Uint32Array; points: Uint32Array }> {
  let m = modelObsCache.get(model);
  if (m) return m;
  const lists = new Map<number, { f: number[]; p: number[] }>();
  for (let i = 0; i < model.numPoints; i++) {
    for (let k = model.trackOffsets[i]; k < model.trackOffsets[i + 1]; k++) {
      const im = model.trackImages[k];
      let l = lists.get(im);
      if (!l) lists.set(im, (l = { f: [], p: [] }));
      l.f.push(model.trackFeatures[k]);
      l.p.push(i);
    }
  }
  m = new Map();
  for (const [im, l] of lists) m.set(im, { features: Uint32Array.from(l.f), points: Uint32Array.from(l.p) });
  modelObsCache.set(model, m);
  return m;
}

/**
 * Inverse depth of each feature of image `id` that observes a point of the sparse model (NaN
 * otherwise) and their median: `inverseDepths` of the reconstruction the model was made from.
 */
export function modelInverseDepths(model: SparseModel, id: number, numFeatures: number): { inv: Float64Array; median: number } | null {
  const im = model.images.find((x) => x.id === id);
  const obs = modelObservations(model).get(id);
  if (!im || !obs) return null;
  const R = im.pose.R,
    t = im.pose.t;
  const inv = new Float64Array(numFeatures).fill(NaN);
  const vals: number[] = [];
  for (let k = 0; k < obs.features.length; k++) {
    const f = obs.features[k],
      j = 3 * obs.points[k];
    const X = model.xyz;
    const z = R[6] * X[j] + R[7] * X[j + 1] + R[8] * X[j + 2] + t[2];
    if (!(z > 0)) continue;
    if (f < numFeatures) inv[f] = 1 / z;
    vals.push(1 / z);
  }
  if (!vals.length) return null;
  vals.sort((a, b) => a - b);
  return { inv, median: vals[vals.length >> 1] };
}

/**
 * ORIGINAL keypoints of image `id` corrected for the rolling shutter of a sparse model
 * (`SparseModel.rollingShutter`: the ω each keyframe was corrected with, and the translational term
 * ν = v · median inverse depth ⇒ v · ρ_f = ν · ρ_f / ρ_median with ρ_f from the model's points), with
 * the image's camera of the model. What refine corrected in place, re-derived from the persisted
 * model (after a reload, or a worker respawn whose store holds the originals). Null when the model
 * has no correction for this image.
 */
export function rsCorrectModelKeypoints(model: SparseModel, id: number, original: Float32Array): Float32Array | null {
  const rs = model.rollingShutter;
  const w = rs?.omega[id];
  const im = model.images.find((x) => x.id === id);
  const cam = im ? model.cameras.get(im.cameraId) : undefined;
  if (!rs || !w || !cam || rs.readoutMs === 0) return null;
  let trans: RSTranslation | null = null;
  const nu = rs.nu?.[id];
  if (nu) {
    const d = modelInverseDepths(model, id, original.length >> 2);
    if (d && d.median > 0) {
      const rel = d.inv.map((x) => x / d.median);
      trans = { velocity: nu, invDepth: rel, defaultInvDepth: 1 };
    }
  }
  return correctKeypointsRS(original, cam, w, rs, new Float32Array(original.length), trans);
}

/**
 * Keypoints of image `id` consistent with the poses of `model` (the COLMAP export's POINTS2D):
 *  - the in-memory keypoints when the live state corrected them for exactly the model's rolling
 *    shutter (refine in this session: same model, same ω) — or when neither is corrected;
 *  - otherwise the ORIGINAL keypoints (the store's, or the live state's saved originals),
 *    corrected from the model's `rollingShutter` when it has one (`rsCorrectModelKeypoints`).
 * `live` is null after a reload (`original` = the store's features).
 */
export function modelKeypoints(
  model: SparseModel | null,
  id: number,
  original: Float32Array,
  live: { st: CaptureState; keypoints: Float32Array } | null,
): Float32Array {
  const rs = model?.rollingShutter ?? null;
  if (live) {
    const { st } = live;
    const w = st.rsOmega.get(id);
    const corrected = !!w && !!st.rollingShutter;
    if (!corrected && !(rs && rs.omega[id])) return live.keypoints;
    const mw = rs?.omega[id];
    if (corrected && rs && mw && st.rollingShutter!.axis === rs.axis && st.rollingShutter!.readoutMs === rs.readoutMs && mw[0] === w![0] && mw[1] === w![1] && mw[2] === w![2])
      return live.keypoints;
  }
  return (model && rsCorrectModelKeypoints(model, id, original)) ?? original;
}

/**
 * `SparseModel.rollingShutter` of a model built from `rec` after refine corrected the keypoints
 * (null when nothing is corrected): the ω each image's keypoints were corrected with
 * (`CaptureState.rsOmega` — not a new gyro query: the alignment may have moved the time offset
 * since, and a readout known in advance may have been applied with an unverified R_cam_device) and,
 * for the images corrected with the translational term, ν = camera velocity × the image's median
 * inverse depth in `rec` (the dense stage applies it at that depth; exports re-derive the
 * per-keypoint term from it).
 */
export function sparseRollingShutter(st: CaptureState, rec: Reconstruction, imageIds: readonly number[]): SparseRollingShutter | null {
  if (!st.rollingShutter || !st.rsOmega.size) return null;
  const omega: Record<number, [number, number, number]> = {};
  for (const id of imageIds) {
    const w = st.rsOmega.get(id);
    if (w) omega[id] = [w[0], w[1], w[2]];
  }
  const out: SparseRollingShutter = { axis: st.rollingShutter.axis, readoutMs: st.rollingShutter.readoutMs, omega };
  if (st.rsTranslated.size) {
    const nu: Record<number, [number, number, number]> = {};
    const vel = cameraVelocities(rec, (id) => st.records.get(id)?.t);
    for (const [id, v] of vel) {
      if (!st.rsTranslated.has(id) || !omega[id]) continue;
      const d = inverseDepths(rec, id);
      if (d) nu[id] = [v[0] * d.median, v[1] * d.median, v[2] * d.median];
    }
    out.nu = nu;
  }
  return out;
}

/** Registered images that rotated fastest (|ω| ≥ minOmega at their frame time), at most maxImages. */
export function fastestImages(rec: Reconstruction, st: CaptureState, gyro: GyroReference, opts: { maxImages?: number; minOmega?: number } = {}): number[] {
  const maxImages = opts.maxImages ?? 24;
  const minOmega = opts.minOmega ?? 0.15;
  const cands: { id: number; speed: number }[] = [];
  for (const id of rec.registeredImageIds()) {
    const r = st.records.get(id);
    const w = r ? gyro.angularVelocityCam(r.t, RS_OMEGA_HALF_WINDOW_MS) : null;
    if (!w) continue;
    const speed = Math.hypot(w[0], w[1], w[2]);
    if (speed >= minOmega) cands.push({ id, speed });
  }
  cands.sort((a, b) => b.speed - a.speed || a.id - b.id);
  return cands.slice(0, maxImages).map((c) => c.id);
}

export interface RefineRSResult {
  estimate: RollingShutterEstimate | null;
  /** Model now applied to the keypoints (null = none). */
  applied: RollingShutterModel | null;
  /** Keyframes corrected. */
  corrected: number;
  /** Whether the translational term was used (estimation and correction). */
  translation: boolean;
  reason: string;
}

/**
 * Estimate the rolling shutter on `rec` and apply it to every keyframe when accepted (otherwise
 * the originals are restored). At least `minImages` fast-rotating registered images are needed.
 * With a CPU `pool`, independent hypotheses are scored in parallel (`searchRollingShutterBatched`).
 */
export async function refineRollingShutter(
  rec: Reconstruction,
  st: CaptureState,
  gyro: GyroReference,
  opts: Partial<RollingShutterEstimateOptions> & {
    minImages?: number;
    maxImages?: number;
    minOmega?: number;
    maxObsPerImage?: number;
    refineFocal?: boolean;
    refinePrincipalPoint?: boolean;
    /** Translational term (camera velocities × inverse depths of `rec`; default true). */
    translation?: boolean;
  } = {},
  known: RollingShutterModel | null = null,
  pool: CpuPool | null = null,
): Promise<RefineRSResult> {
  const camFor = (id: number): CameraIntrinsics | undefined => {
    const img = rec.images.get(id);
    if (img) return rec.cameras.get(img.cameraId);
    const r = st.records.get(id);
    if (!r) return undefined;
    for (const c of rec.cameras.values()) if (c.width === r.width && c.height === r.height) return c;
    return undefined;
  };
  const useTrans = opts.translation !== false;
  const trans = useTrans ? rsTranslations(rec, (id) => st.records.get(id)?.t) : null;
  const transFor = trans ? (id: number): RSTranslation | null => trans.get(id) ?? null : null;
  const ids = fastestImages(rec, st, gyro, { maxImages: opts.maxImages, minOmega: opts.minOmega });
  const minImages = opts.minImages ?? 6;
  if (ids.length < minImages) {
    const n = applyRollingShutter(st, camFor, gyro, known, transFor);
    const why = `only ${ids.length} registered images rotate fast enough (≥ ${minImages} needed)`;
    return { estimate: null, applied: known, corrected: n, translation: useTrans, reason: known ? `${why}: kept the known readout` : why };
  }
  const omegaOf = (id: number): Float64Array | null => {
    const r = st.records.get(id);
    return r ? gyro.angularVelocityCam(r.t, RS_OMEGA_HALF_WINDOW_MS) : null;
  };
  // short BA on a subset of the points: every registered image free (with a gauge), ≤ 40 points per
  // image, ≤ 8 observations per point, shared focal free. All poses must be free: on an orbit every
  // frame rotates at a similar rate, so held (RS-biased) poses would resist the correction.
  const bc = new RSBundleCost(rec, rec.registeredImageIds(), (id) => originalKeypoints(st, id), omegaOf, {
    maxObsPerImage: opts.maxObsPerImage ?? 40,
    maxTrack: 8,
    refineFocal: opts.refineFocal ?? true,
    refinePrincipalPoint: opts.refinePrincipalPoint ?? false,
    velocityOf: trans ? (id) => trans.get(id)?.velocity ?? null : undefined,
  });
  const info = { numImages: rec.numRegisteredImages, numObservations: bc.numObservations };
  // with CPU workers, the hypotheses that do not depend on each other are evaluated in parallel (same
  // estimate); a failed batch is evaluated here
  const est =
    pool && pool.size > 0
      ? await searchRollingShutterBatched((models) => pool.rsCosts(bc, models).catch(() => models.map((m) => bc.cost(m))), opts, info)
      : searchRollingShutter((m) => bc.cost(m), opts, info);
  let model = est.accepted ? est.model : null;
  let reason = est.reason;
  if (!model && known && known.readoutMs !== 0) {
    // a readout known in advance (device constant) is kept unless the data contradict it
    const c = bc.cost(known);
    if (c <= est.cost0) {
      model = known;
      reason = `${est.reason}; kept the known ${known.axis} ${known.readoutMs.toFixed(1)} ms (cost ${c.toFixed(1)} ≤ ${est.cost0.toFixed(1)})`;
    }
  }
  const n = applyRollingShutter(st, camFor, gyro, model, transFor);
  return { estimate: est, applied: model, corrected: n, translation: useTrans, reason };
}

/** Stats fields describing a rolling-shutter result (canonical image + device frame). */
export function rsStats(r: RefineRSResult, R_cam_device: ArrayLike<number>): Record<string, number | string> {
  const out: Record<string, number | string> = { rollingShutter: r.applied ? `${r.applied.axis} ${r.applied.readoutMs.toFixed(1)} ms` : `none (${r.reason})` };
  if (r.estimate) {
    out.rsBest = `${r.estimate.best.axis} ${r.estimate.best.readoutMs.toFixed(1)} ms`;
    out.rsImprovementPct = Math.round(1000 * r.estimate.improvement) / 10;
    out.rsImages = r.estimate.numImages;
  }
  if (r.applied) {
    const d = rsModelToDevice(r.applied, R_cam_device);
    out.rsReadoutMs = Math.round(10 * d.readoutMs) / 10;
    out.rsSweepDevice = d.sweepDevice.map((v) => Math.round(v * 1000) / 1000).join(',');
    out.rsTranslation = r.translation ? 'on' : 'off';
  }
  return out;
}

// ------------------------------------------------------------------------------------------
// per-image focal

export interface PerImageFocalResult {
  rec: Reconstruction;
  accepted: boolean;
  reason: string;
  before: number;
  after: number;
  /** Relative spread (std / mean) of the per-image focal lengths when accepted. */
  spread: number;
  /** Lag-1 autocorrelation of the per-image focal lengths in time order (NaN when not computed). */
  autocorrelation: number;
}

/**
 * Per-image focal refinement (focus breathing) on a clone of `rec`: one camera per registered
 * image (copy of its shared camera), focal free with a prior N(f_shared, (sigmaRel·f_shared)²),
 * principal point and distortion fixed; one global BA (poses + points free). Accepted when the
 * mean reprojection error drops by at least `minImprovement` (relative) and the focal lengths are
 * smooth in time (lag-1 autocorrelation ≥ `minAutocorrelation`, when `timeOrder` is given).
 *
 * With `refinePrincipalPoint` (the shared (f, cx, cy, k1, k2) refinement of refine), the shared
 * camera is then re-estimated WITHOUT the breathing: a per-image focal f_i is exactly a scaling of
 * the image's keypoints about the principal point by f̄/f_i (RADIAL distortion acts on normalised
 * coordinates), so the keypoints are rescaled temporarily (in place, restored), the shared camera
 * refined on them, and the per-image focal lengths re-expressed around the new shared focal and
 * polished by a second per-image BA. Breathing that correlates with the trajectory otherwise
 * biases the shared principal point (e2e abl-object-breath: 5.6 px ⇒ 0.44° rotation error).
 */
export async function refinePerImageFocal(
  rec: Reconstruction,
  db: FeatureDatabase,
  graph: CorrespondenceGraph,
  opts: Partial<MapperOptions>,
  o: {
    sigmaRel?: number;
    minImprovement?: number;
    minAutocorrelation?: number;
    timeOrder?: readonly number[];
    /**
     * Shared (f, c, k) refinement of the rescaled model; `priors` (shared camera id → weak principal-
     * point prior, `ppPriorSigmaPx`) are to be passed to its BA (`MapperOptions.principalPointPriors`).
     * `rec` is a throwaway copy owned by the callee (it may be refined in place).
     */
    refinePrincipalPoint?: (
      rec: Reconstruction,
      priors?: ReadonlyMap<number, { cx: number; cy: number; sigma: number }>,
    ) => Promise<{ rec: Reconstruction; accepted: boolean; reason: string }>;
    /** Maximum alternations of the shared re-refinement and the per-image focal lengths (default 6). */
    rounds?: number;
    /** Stop alternating once the shared principal point moves less than this (px; default 0.25) and f̄ < 0.02 %. */
    ppTolerancePx?: number;
    /**
     * Weak prior σ (px) on the shared principal point in every shared re-refinement of the
     * alternation (0 / absent = none), anchored at the shared principal point of the input model
     * (`ppPriorAnchor` 'input', default) or the image centre ('centre').
     */
    ppPriorSigmaPx?: number;
    ppPriorAnchor?: 'input' | 'centre';
    cancel?: CancelToken;
  } = {},
): Promise<PerImageFocalResult> {
  const sigmaRel = o.sigmaRel ?? 0.005;
  const minImp = o.minImprovement ?? 0.005;
  const minAc = o.minAutocorrelation ?? 0.6;
  const before = rec.meanReprojectionError();
  const work = rec.clone();
  const sharedOf = new Map<number, number>(); // image id → shared camera id
  const priors = new Map<number, number>(); // per-image camera id → shared focal
  for (const id of work.registeredImageIds()) {
    const img = work.images.get(id)!;
    const cam = work.cameras.get(img.cameraId)!;
    sharedOf.set(id, img.cameraId);
    const cid = work.addCamera({ ...cam });
    priors.set(cid, cam.f);
    img.cameraId = cid;
  }
  const m = new IncrementalMapper(db, graph, work, { ...opts });
  await m.adjustGlobalBundleFocalPriors(priors, sigmaRel, o.cancel);
  const after = work.meanReprojectionError();
  const stats = (r: Reconstruction): { spread: number; ac: number } => {
    const fs: number[] = [];
    for (const id of r.registeredImageIds()) fs.push(r.cameras.get(r.images.get(id)!.cameraId)!.f);
    const mean = fs.reduce((a, b) => a + b, 0) / Math.max(1, fs.length);
    const sd = Math.sqrt(fs.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, fs.length));
    const ac = o.timeOrder ? lag1Autocorrelation(o.timeOrder.filter((id) => r.images.get(id)?.registered).map((id) => r.cameras.get(r.images.get(id)!.cameraId)!.f)) : NaN;
    return { spread: mean > 0 ? sd / mean : 0, ac };
  };
  const { spread, ac } = stats(work);
  const imp = before > 0 ? (before - after) / before : 0;
  const summary = `reprojection ${before.toFixed(3)} → ${after.toFixed(3)} px (${imp >= 0 ? '−' : '+'}${Math.abs(100 * imp).toFixed(1)} %), focal spread ${(100 * spread).toFixed(2)} %, lag-1 autocorrelation ${ac.toFixed(2)}`;
  if (!(imp >= minImp)) return { rec, accepted: false, reason: `${summary}: improvement < ${(100 * minImp).toFixed(1)} %`, before, after, spread, autocorrelation: ac };
  // Focus breathing follows the (lagged) focus distance: smooth in time. Per-image focal lengths
  // that jump between consecutive keyframes absorb something else (e2e object-real without a good
  // rolling-shutter correction: correlation −0.84 with the angular speed, lag-1 autocorrelation
  // −0.14 against 0.94 for the true focal lengths) and are rejected.
  if (o.timeOrder && !(ac >= minAc)) return { rec, accepted: false, reason: `${summary}: not smooth in time (< ${minAc})`, before, after, spread, autocorrelation: ac };
  let out = work;
  let ppNote = '';
  if (o.refinePrincipalPoint) {
    // Alternate: shared camera WITHOUT the breathing (keypoints rescaled about the principal point
    // by f̄/f_i, temporarily, in place), then per-image focal lengths around the new shared focal.
    // The per-image focal lengths are only partly observable (focal ↔ dolly), so one round leaves part
    // of the breathing in the shared camera; the alternation converges linearly (principal point
    // 0.4–1 px per round on the e2e captures) — up to `rounds` (6) until it moves < 0.25 px.
    // The cap stops before convergence on purpose: c_y still moves at round 6 and walks past the
    // truth with more rounds (object-real, 12 rounds: 1.46 px vs 0.87 px at 6), so it acts as an
    // early-stopping regulariser; a joint BA with intrinsics shared across cameras would be the
    // principled fix (docs/research/real-capture-robustness.md).
    const shared = new Map<number, CameraIntrinsics>(); // shared camera id → its current values
    for (const sid of new Set(sharedOf.values())) shared.set(sid, { ...rec.cameras.get(sid)! });
    const rounds = Math.max(1, o.rounds ?? 6);
    const notes: string[] = [];
    // weak principal-point prior: damps the slow drift along the weakly observable c ↔ pitch mode
    let ppPriors: Map<number, { cx: number; cy: number; sigma: number }> | undefined;
    const ppSigma = o.ppPriorSigmaPx ?? 0;
    if (ppSigma > 0) {
      ppPriors = new Map();
      for (const [sid, c] of shared) {
        const centre = o.ppPriorAnchor === 'centre';
        ppPriors.set(sid, { cx: centre ? (c.width - 1) / 2 : c.cx, cy: centre ? (c.height - 1) / 2 : c.cy, sigma: ppSigma });
      }
    }
    let stop = `stopped at the ${rounds}-round cap`;
    for (let round = 0; round < rounds; round++) {
      const before0 = [...shared.values()][0];
      const prevF = before0.f,
        prevCx = before0.cx,
        prevCy = before0.cy;
      const saved = new Map<number, Float32Array>();
      const rescaled = out.clone();
      const rel = new Map<number, number>(); // image → f_i / f̄
      for (const id of out.registeredImageIds()) {
        const img = out.images.get(id)!;
        const kp = img.keypoints;
        const sid = sharedOf.get(id);
        if (!kp || sid === undefined || saved.has(id)) continue;
        const ci = out.cameras.get(img.cameraId)!;
        const sh = shared.get(sid)!;
        rel.set(id, ci.f / sh.f);
        saved.set(id, kp.slice());
        const sc = sh.f / ci.f;
        for (let k = 0; k < kp.length; k += 4) {
          kp[k] = ci.cx + (kp[k] - ci.cx) * sc;
          kp[k + 1] = ci.cy + (kp[k + 1] - ci.cy) * sc;
        }
      }
      // shared cameras (current values) replace the per-image ones in the rescaled copy
      for (const [sid, cam] of shared) {
        if (rescaled.cameras.has(sid)) Object.assign(rescaled.cameras.get(sid)!, cam);
        else rescaled.addCamera(cam, sid);
      }
      for (const id of rel.keys()) rescaled.images.get(id)!.cameraId = sharedOf.get(id)!;
      for (const cid of [...rescaled.cameras.keys()]) if (!shared.has(cid) && ![...rescaled.images.values()].some((im) => im.cameraId === cid)) rescaled.cameras.delete(cid);
      let pp: { rec: Reconstruction; accepted: boolean; reason: string } | null = null;
      try {
        pp = await o.refinePrincipalPoint(rescaled, ppPriors);
      } finally {
        for (const [id, kp] of saved) out.images.get(id)!.keypoints!.set(kp);
      }
      if (!pp || !pp.accepted) {
        if (pp) notes.push(`round ${round + 1}: shared camera kept (${pp.reason})`);
        stop = `stopped in round ${round + 1}`;
        break;
      }
      // per-image cameras around the new shared camera, then polish the focal lengths
      const r2 = pp.rec;
      const priors2 = new Map<number, number>();
      for (const [sid] of shared) {
        const c = r2.cameras.get(sid);
        if (c) shared.set(sid, { ...c });
      }
      for (const [id, fr] of rel) {
        const img = r2.images.get(id)!;
        const shNew = shared.get(sharedOf.get(id)!)!;
        const cid = r2.addCamera({ ...shNew, f: shNew.f * fr });
        priors2.set(cid, shNew.f);
        img.cameraId = cid;
      }
      const m2 = new IncrementalMapper(db, graph, r2, { ...opts });
      await m2.adjustGlobalBundleFocalPriors(priors2, sigmaRel, o.cancel);
      out = r2;
      const c0 = [...shared.values()][0];
      notes.push(`round ${round + 1}: f̄ ${c0.f.toFixed(1)} px, principal point offset (${(c0.cx - (c0.width - 1) / 2).toFixed(2)}, ${(c0.cy - (c0.height - 1) / 2).toFixed(2)}) px`);
      // converged: the shared camera hardly moves any more (the alternation converges linearly)
      if (Math.hypot(c0.cx - prevCx, c0.cy - prevCy) < (o.ppTolerancePx ?? 0.25) && Math.abs(c0.f / prevF - 1) < 2e-4) {
        stop = `converged in ${round + 1} rounds`;
        break;
      }
    }
    if (notes.length) {
      const prior = ppPriors ? `, principal-point prior σ ${ppSigma} px (${o.ppPriorAnchor === 'centre' ? 'image centre' : 'input'})` : '';
      ppNote = `; shared camera re-refined without the breathing (${stop}${prior}): ${notes.join('; ')}`;
    }
  }
  // drop the now unused shared cameras
  const used = new Set<number>();
  for (const img of out.images.values()) used.add(img.cameraId);
  for (const cid of [...out.cameras.keys()]) if (!used.has(cid)) out.cameras.delete(cid);
  const fin = stats(out);
  return {
    rec: out,
    accepted: true,
    reason: `${summary}${ppNote}; final ${out.meanReprojectionError().toFixed(3)} px, spread ${(100 * fin.spread).toFixed(2)} %`,
    before,
    after: out.meanReprojectionError(),
    spread: fin.spread,
    autocorrelation: fin.ac,
  };
}

/** Lag-1 autocorrelation of a series (NaN for < 3 values or no variance). */
export function lag1Autocorrelation(x: readonly number[]): number {
  const n = x.length;
  if (n < 3) return NaN;
  const m = x.reduce((a, b) => a + b, 0) / n;
  let num = 0,
    den = 0;
  for (let i = 0; i < n; i++) {
    den += (x[i] - m) ** 2;
    if (i + 1 < n) num += (x[i] - m) * (x[i + 1] - m);
  }
  return den > 0 ? num / den : NaN;
}
