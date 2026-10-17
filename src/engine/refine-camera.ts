/**
 * Camera-model refinements of refine (ENGINE.md "Refine"): gyro-based rolling-shutter correction
 * of the keypoints and per-image focal lengths (focus breathing).
 *
 * Rolling shutter (src/sfm/rolling-shutter.ts): the readout direction / time is estimated on the
 * reconstruction (hypotheses axis × readout, robust reprojection after a motion-only BA of the
 * fastest-rotating images) and, when it lowers the cost significantly, every keyframe's keypoints
 * are corrected IN PLACE (the FeatureSet, the live / offline reconstructions and their clones share
 * the arrays) from the originals kept in `CaptureState.rsOriginal`; estimates are always absolute
 * (from the originals), so refine can run again. The model is persisted in the engine state and
 * reported in `stats` (canonical image and device frame). A model known in advance (e.g. from an
 * earlier capture on the same device, `EngineConfig.rollingShutter`) is applied live to every new
 * keyframe (`correctKeyframeRS`).
 *
 * Per-image focal (focus breathing, src/ba focal priors): every registered image gets its own
 * camera whose focal is free but tied to the shared focal by a Gaussian prior (σ = 0.5 % by
 * default); principal point and distortion stay shared and fixed. Kept only when the mean
 * reprojection error drops significantly.
 */
import type { CameraIntrinsics, CancelToken } from '../core/types';
import type { CorrespondenceGraph } from '../sfm/correspondence-graph';
import type { FeatureDatabase } from '../sfm/database';
import type { GyroReference } from '../sfm/gyro';
import { IncrementalMapper, type MapperOptions } from '../sfm/mapper';
import type { Reconstruction } from '../sfm/reconstruction';
import {
  correctKeypointsRS,
  estimateRollingShutter,
  rsHypothesisCost,
  rsModelToDevice,
  type RollingShutterEstimate,
  type RollingShutterEstimateOptions,
  type RollingShutterModel,
  type RSImageObservations,
} from '../sfm/rolling-shutter';
import type { CaptureState } from './live';

/** Original (uncorrected) keypoints of a keyframe: the saved copy, else the current array. */
export function originalKeypoints(st: CaptureState, id: number): Float32Array | null {
  return st.rsOriginal.get(id) ?? st.db.features.get(id)?.keypoints ?? null;
}

/**
 * Correct one keyframe's keypoints in place for `model` from its originals (saved on first use);
 * readout 0 (or no gyro coverage) restores the originals. Returns true when corrected.
 */
export function correctKeyframeRS(st: CaptureState, id: number, cam: CameraIntrinsics, gyro: GyroReference, model: RollingShutterModel | null): boolean {
  const fs = st.db.features.get(id);
  const r = st.records.get(id);
  if (!fs || !r) return false;
  let orig = st.rsOriginal.get(id);
  if (!model || model.readoutMs === 0) {
    if (orig) fs.keypoints.set(orig);
    return false;
  }
  const w = gyro.angularVelocityCam(r.t, Math.max(8, 0.5 * Math.abs(model.readoutMs)));
  if (!w) {
    if (orig) fs.keypoints.set(orig);
    return false;
  }
  if (!orig) {
    orig = fs.keypoints.slice();
    st.rsOriginal.set(id, orig);
  }
  correctKeypointsRS(orig, cam, w, model, fs.keypoints);
  return true;
}

/** Correct every keyframe for `model` (null / 0 restores the originals). Returns the number corrected. */
export function applyRollingShutter(st: CaptureState, camFor: (id: number) => CameraIntrinsics | undefined, gyro: GyroReference, model: RollingShutterModel | null): number {
  let n = 0;
  for (const id of st.order) {
    const cam = camFor(id);
    if (!cam) continue;
    if (correctKeyframeRS(st, id, cam, gyro, model)) n++;
  }
  st.rollingShutter = model && model.readoutMs !== 0 ? { ...model } : null;
  return n;
}

/**
 * Observations of the registered images that rotated fastest (|ω| ≥ minOmega, at most maxImages)
 * from the ORIGINAL keypoints, for `estimateRollingShutter`.
 */
export function rsObservations(rec: Reconstruction, st: CaptureState, gyro: GyroReference, opts: { maxImages?: number; minOmega?: number; maxObsPerImage?: number } = {}): RSImageObservations[] {
  const maxImages = opts.maxImages ?? 30;
  const minOmega = opts.minOmega ?? 0.15;
  const maxObs = opts.maxObsPerImage ?? 600;
  const cands: { id: number; w: Float64Array; speed: number }[] = [];
  for (const id of rec.registeredImageIds()) {
    const r = st.records.get(id);
    if (!r) continue;
    const w = gyro.angularVelocityCam(r.t, 12);
    if (!w) continue;
    const speed = Math.hypot(w[0], w[1], w[2]);
    if (speed >= minOmega) cands.push({ id, w: Float64Array.from(w), speed });
  }
  cands.sort((a, b) => b.speed - a.speed || a.id - b.id);
  const out: RSImageObservations[] = [];
  for (const c of cands.slice(0, maxImages)) {
    const img = rec.images.get(c.id)!;
    const kp = originalKeypoints(st, c.id);
    const cam = rec.cameras.get(img.cameraId);
    if (!kp || !cam) continue;
    const feats: number[] = [];
    for (let f = 0; f < img.point3D.length; f++) if (img.point3D[f] >= 0 && rec.points.has(img.point3D[f])) feats.push(f);
    const stride = Math.max(1, Math.ceil(feats.length / maxObs));
    const uv: number[] = [],
      X: number[] = [];
    for (let k = 0; k < feats.length; k += stride) {
      const f = feats[k];
      const p = rec.points.get(img.point3D[f])!;
      uv.push(kp[4 * f], kp[4 * f + 1]);
      X.push(p.xyz[0], p.xyz[1], p.xyz[2]);
    }
    if (uv.length < 2 * 20) continue;
    out.push({ id: c.id, cam: { ...cam }, pose: { R: Float64Array.from(img.pose.R), t: Float64Array.from(img.pose.t) }, omega: c.w, uv: Float64Array.from(uv), X: Float64Array.from(X) });
  }
  return out;
}

export interface RefineRSResult {
  estimate: RollingShutterEstimate | null;
  /** Model now applied to the keypoints (null = none). */
  applied: RollingShutterModel | null;
  /** Keyframes corrected. */
  corrected: number;
  reason: string;
}

/**
 * Estimate the rolling shutter on `rec` and apply it to every keyframe when accepted (otherwise
 * the originals are restored). At least `minImages` fast-rotating registered images are needed.
 */
export function refineRollingShutter(
  rec: Reconstruction,
  st: CaptureState,
  gyro: GyroReference,
  opts: Partial<RollingShutterEstimateOptions> & { minImages?: number; maxImages?: number; minOmega?: number } = {},
  known: RollingShutterModel | null = null,
): RefineRSResult {
  const camFor = (id: number): CameraIntrinsics | undefined => {
    const img = rec.images.get(id);
    if (img) return rec.cameras.get(img.cameraId);
    const r = st.records.get(id);
    if (!r) return undefined;
    for (const c of rec.cameras.values()) if (c.width === r.width && c.height === r.height) return c;
    return undefined;
  };
  const obs = rsObservations(rec, st, gyro, { maxImages: opts.maxImages, minOmega: opts.minOmega });
  const minImages = opts.minImages ?? 6;
  if (obs.length < minImages) {
    const n = applyRollingShutter(st, camFor, gyro, known);
    const why = `only ${obs.length} registered images rotate fast enough (≥ ${minImages} needed)`;
    return { estimate: null, applied: known, corrected: n, reason: known ? `${why}: kept the known readout` : why };
  }
  const est = estimateRollingShutter(obs, opts);
  let model = est.accepted ? est.model : null;
  let reason = est.reason;
  if (!model && known && known.readoutMs !== 0) {
    // a readout known in advance (device constant) is kept unless the data contradict it
    const c = rsHypothesisCost(obs, known, opts);
    if (c <= est.cost0) {
      model = known;
      reason = `${est.reason}; kept the known ${known.axis} ${known.readoutMs.toFixed(1)} ms (cost ${c.toFixed(1)} ≤ ${est.cost0.toFixed(1)})`;
    }
  }
  const n = applyRollingShutter(st, camFor, gyro, model);
  return { estimate: est, applied: model, corrected: n, reason };
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
}

/**
 * Per-image focal refinement (focus breathing) on a clone of `rec`: one camera per registered
 * image (copy of its shared camera), focal free with a prior N(f_shared, (sigmaRel·f_shared)²),
 * principal point and distortion fixed; one global BA (poses + points free). Accepted when the
 * mean reprojection error drops by at least `minImprovement` (relative).
 */
export async function refinePerImageFocal(
  rec: Reconstruction,
  db: FeatureDatabase,
  graph: CorrespondenceGraph,
  opts: Partial<MapperOptions>,
  o: { sigmaRel?: number; minImprovement?: number; cancel?: CancelToken } = {},
): Promise<PerImageFocalResult> {
  const sigmaRel = o.sigmaRel ?? 0.005;
  const minImp = o.minImprovement ?? 0.03;
  const before = rec.meanReprojectionError();
  const work = rec.clone();
  const shared = new Map<number, number>(); // new camera id → shared focal
  for (const id of work.registeredImageIds()) {
    const img = work.images.get(id)!;
    const cam = work.cameras.get(img.cameraId)!;
    const cid = work.addCamera({ ...cam });
    shared.set(cid, cam.f);
    img.cameraId = cid;
  }
  const m = new IncrementalMapper(db, graph, work, { ...opts });
  await m.adjustGlobalBundleFocalPriors(shared, sigmaRel, o.cancel);
  const after = work.meanReprojectionError();
  const fs: number[] = [];
  for (const id of work.registeredImageIds()) fs.push(work.cameras.get(work.images.get(id)!.cameraId)!.f);
  const mean = fs.reduce((a, b) => a + b, 0) / Math.max(1, fs.length);
  const sd = Math.sqrt(fs.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, fs.length));
  const spread = mean > 0 ? sd / mean : 0;
  const imp = before > 0 ? (before - after) / before : 0;
  if (!(imp >= minImp)) {
    return { rec, accepted: false, reason: `reprojection ${before.toFixed(3)} → ${after.toFixed(3)} px (${(100 * imp).toFixed(1)} % < ${(100 * minImp).toFixed(0)} %)`, before, after, spread };
  }
  // drop the now unused shared cameras
  const used = new Set<number>();
  for (const img of work.images.values()) used.add(img.cameraId);
  for (const cid of [...work.cameras.keys()]) if (!used.has(cid)) work.cameras.delete(cid);
  return { rec: work, accepted: true, reason: `reprojection ${before.toFixed(3)} → ${after.toFixed(3)} px (−${(100 * imp).toFixed(1)} %), focal spread ${(100 * spread).toFixed(2)} %`, before, after, spread };
}
