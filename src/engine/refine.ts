/**
 * Refine after capture (ENGINE.md "Refine"): vocabulary retrain / extend → candidate pairs
 * (sequential 10, retrieval 15, spatial 10) → match + verify → refined live model → unless it is
 * healthy (`liveModelHealth`), a full incremental SfM from scratch (live-refined intrinsics as the
 * initial guess, gyro-aided registration as live) and the better of the live and the offline model
 * (gyro breaks, registered images, explained correspondences, mean reprojection error) → final
 * global BA with self-calibration (f, k1, k2), re-triangulation and filtering → world alignment
 * (IMU gravity / time offset / metric scale, GPS, compass; verified R_cam_device) → SparseModel.
 *
 * The offline SfM can return several models: only the largest is kept (no merging); stats
 * `offlineModels` / `offlineDroppedImages` report the others, and `unconnectedImages` the
 * keyframes missing from the final model (the viewer offers "Add more views" for them).
 *
 * Progress stages: 'matching', 'reconstruction', 'alignment' (src/app/stages.ts).
 */
import type { CancelToken, MatchPairs, PointCloud, ProgressCallback, TwoViewGeometry } from '../core/types';
import { estimateTwoViewGeometry } from '../geometry/twoview';
import { mergePairs, retrievalPairs, sequentialPairs, spatialPairs, type ImagePair, type SpatialCamera } from '../sfm/pairs';
import type { CorrespondenceGraph } from '../sfm/correspondence-graph';
import { pairKey, type FeatureDatabase } from '../sfm/database';
import { extractColors, IncrementalMapper, runIncrementalSfM, SfMCancelledError, type MapperOptions, type RunSfMOptions } from '../sfm/mapper';
import { reprojectionErrorSq, type Reconstruction } from '../sfm/reconstruction';
import { VocabularyIndex } from '../sfm/vocab';
import { computeWorldAlignment, type WorldAlignment, type WorldKeyframe } from '../sfm/world';
import { gyroConsistency, timedRotations, type GyroConsistency } from '../sfm/gyro';
import { median } from '../math/stats';
import type { ScreenAngle } from '../sensors/frame-orientation';
import { handEyeCheck, keyframeUpCam, type HandEyeResult, type LiveSession } from './live';
import { refinePerImageFocal, refineRollingShutter, rsStats, sparseRollingShutter, type RefineRSResult } from './refine-camera';
import { pairToStored, type EngineStore, type StoredPair, type StoredPairs } from './persistence';
import type { CpuPool } from './cpu-pool';
import { ENGINE_DEFAULTS } from './presets';
import type { EngineConfig, SparseCameraEx, SparseModel, SparseResultEx } from './types';

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

/**
 * Yield one macrotask (every pending microtask runs first) without the ≥ 4 ms clamp of nested
 * timers: a message to ourselves. Used to let a queued GPU job encode and submit its work before
 * the caller blocks the thread with CPU work.
 */
const taskChannel = typeof MessageChannel !== 'undefined' ? new MessageChannel() : null;
const taskQueue: (() => void)[] = [];
if (taskChannel) {
  taskChannel.port1.onmessage = () => taskQueue.shift()?.();
  // (Node: the open port must not keep the process alive)
  for (const port of [taskChannel.port1, taskChannel.port2]) (port as unknown as { unref?: () => void }).unref?.();
}
const yieldTask = (): Promise<void> =>
  taskChannel
    ? new Promise((r) => {
        taskQueue.push(r);
        taskChannel.port2.postMessage(null);
      })
    : tick();

export function cancelledError(stage: string): Error {
  return new SfMCancelledError(stage);
}

function checkCancel(cancel: CancelToken | undefined, stage: string): void {
  if (cancel?.cancelled) throw cancelledError(stage);
}

export interface RefineDeps {
  live: LiveSession;
  config: EngineConfig;
  store: EngineStore;
  /** CPU helper workers for the pair verification and the vocabulary indexing (null / absent: this thread). */
  pool?: CpuPool | null;
  log(level: 'debug' | 'info' | 'warn' | 'error', message: string, data?: unknown): void;
  /** Await visibility between GPU batches (the engine's gate does it per call). */
}

export interface RefineOutput {
  model: SparseModel;
  result: SparseResultEx;
  rec: Reconstruction;
  alignment: WorldAlignment | null;
}

/** Plain-array model of the registered part of a reconstruction (capture order). */
export function modelFromReconstruction(rec: Reconstruction, order: readonly number[]): SparseModel {
  const pos = new Map<number, number>();
  order.forEach((id, i) => pos.set(id, i));
  const ids = rec.registeredImageIds().sort((a, b) => (pos.get(a) ?? a) - (pos.get(b) ?? b));
  const n = rec.points.size;
  const xyz = new Float64Array(3 * n);
  const colors = new Uint8Array(3 * n);
  const errors = new Float32Array(n);
  const offsets = new Uint32Array(n + 1);
  let total = 0;
  for (const p of rec.points.values()) total += p.track.length;
  const ti = new Uint32Array(total);
  const tf = new Uint32Array(total);
  let i = 0,
    k = 0;
  for (const p of rec.points.values()) {
    xyz[3 * i] = p.xyz[0];
    xyz[3 * i + 1] = p.xyz[1];
    xyz[3 * i + 2] = p.xyz[2];
    colors[3 * i] = p.color[0];
    colors[3 * i + 1] = p.color[1];
    colors[3 * i + 2] = p.color[2];
    errors[i] = p.error;
    offsets[i] = k;
    for (const el of p.track) {
      ti[k] = el.image;
      tf[k] = el.feature;
      k++;
    }
    i++;
  }
  offsets[n] = k;
  const cameras = new Map(rec.cameras);
  return {
    cameras,
    images: ids.map((id) => {
      const im = rec.images.get(id)!;
      return { id, cameraId: im.cameraId, pose: { R: Float64Array.from(im.pose.R), t: Float64Array.from(im.pose.t) } };
    }),
    numPoints: n,
    xyz,
    colors,
    errors,
    trackOffsets: offsets,
    trackImages: ti,
    trackFeatures: tf,
  };
}

/** Point cloud (float32) of a sparse model. */
export function modelPointCloud(m: SparseModel): PointCloud {
  const positions = new Float32Array(3 * m.numPoints);
  for (let i = 0; i < positions.length; i++) positions[i] = m.xyz[i];
  return { count: m.numPoints, positions, colors: m.colors.slice() };
}

/**
 * When the only reason the band-pass IMU scale was rejected is low excitation, but the fit itself
 * is excellent, return the relaxed threshold to retry with: relative σ ≤ 3 % and either R² ≥ 0.9
 * with excitation ≥ 0.03 m/s², or R² ≥ 0.95 with excitation ≥ 0.015 m/s². Reached when the
 * alignment found no metric scale at all — the keyframe-interval estimate that
 * `computeWorldAlignment` falls back to (`estimateMetricScaleFromIntervals`, not limited by the
 * keyframe rate) was unreliable as well and no GPS scale exists, e.g. a gentle orbit, whose slow
 * centripetal acceleration is degenerate with a device-frame accelerometer bias — or when the scale
 * is that interval estimate alone and the retry would claim a smaller σ (`lowExcitationSigma`):
 * the more precise of the two wins, not the one `computeWorldAlignment` happens to try first
 * (e2e object-real: interval σ 2.4 % with κ just above its 2 % gate vs the retry's 2.0 %). Not
 * when the interval regression also supplied the time offset (`intervalTimeOffset`): the re-run
 * at the relaxed threshold takes the band-pass path and would fall back to the default offset.
 *
 * The excitation the inertial module measures is the band-limited IMU acceleration AS SAMPLED AT
 * THE KEYFRAMES: it falls with the keyframe rate, not only with the hand motion (e2e `object`, GT
 * poses, same motion: 0.110 m/s² with all 200 frames, 0.058 at 90 keyframes, 0.029 at 53 —
 * R² 0.993 / 0.988 / 0.977, scale error +0.4 / +0.3 / +1.8 %). What the floor guards against is
 * errors-in-variables attenuation of the regression (noise on the IMU side biases the scale), and
 * that is bounded by the fit itself: attenuation ≤ 1 − R². So R² decides, and the caller reports
 * σ ≥ 1 − R² (`lowExcitationSigma`).
 */
export function lowExcitationRetry(a: WorldAlignment): { minExcitation: number; excitation: number; r2: number } | null {
  const s = a.details.imuScale;
  if (!s || s.reliable) return null;
  if (a.metric.metric) {
    // (not when the interval regression also found the time offset: the re-run would lose it)
    const iv = a.details.imuScaleIntervals;
    if (a.metric.source !== 'imu' || !iv || !iv.reliable || a.details.intervalTimeOffset?.reliable) return null;
    if (!(lowExcitationSigma(s.relUncertainty, s.r2) < iv.relUncertainty)) return null;
  }
  const reasons = s.reason.split(';').map((r) => r.trim()).filter(Boolean);
  if (reasons.length !== 1 || !reasons[0].startsWith('insufficient excitation')) return null;
  if (!(s.scale > 0 && s.relUncertainty <= 0.03)) return null;
  if (!((s.excitationRms >= 0.03 && s.r2 >= 0.9) || (s.excitationRms >= 0.015 && s.r2 >= 0.95))) return null;
  return { minExcitation: Math.min(0.03, s.excitationRms * 0.99), excitation: s.excitationRms, r2: s.r2 };
}

/** σ claimed for a scale accepted by `lowExcitationRetry`: max(2σ, 2 %, 1 − R² — the EIV attenuation bound). */
export function lowExcitationSigma(relUncertainty: number | null, r2: number): number {
  return Math.max(2 * (relUncertainty ?? 0), 0.02, 1 - r2);
}

/**
 * One global bundle adjustment of `rec` with the principal point refined (in place on a clone).
 * Accepted when every camera's principal point stays within 10 % of the image size of the image
 * centre and the mean reprojection error does not grow; otherwise the input model is kept.
 * With `consumeInput` the adjustment runs on `rec` itself (no clone): for a throwaway input the
 * caller discards when not accepted, since `rec` is then modified whatever the outcome.
 */
export async function refinePrincipalPoint(
  rec: Reconstruction,
  db: FeatureDatabase,
  graph: CorrespondenceGraph,
  opts: Partial<MapperOptions>,
  cancel?: CancelToken,
  consumeInput = false,
): Promise<{ rec: Reconstruction; accepted: boolean; reason: string }> {
  const before = rec.meanReprojectionError();
  const work = consumeInput ? rec : rec.clone();
  const m = new IncrementalMapper(db, graph, work, { ...opts, refinePrincipalPoint: true, refineFocal: true });
  await m.adjustGlobalBundleSliced(Infinity, cancel, true);
  m.filterPoints();
  const after = work.meanReprojectionError();
  const shifts: string[] = [];
  for (const cam of work.cameras.values()) {
    const dx = cam.cx - (cam.width - 1) / 2;
    const dy = cam.cy - (cam.height - 1) / 2;
    if (!(Math.abs(dx) <= 0.1 * cam.width && Math.abs(dy) <= 0.1 * cam.height)) {
      return { rec, accepted: false, reason: `principal point (${cam.cx.toFixed(1)}, ${cam.cy.toFixed(1)}) too far from the image centre` };
    }
    shifts.push(`(${dx >= 0 ? '+' : ''}${dx.toFixed(2)}, ${dy >= 0 ? '+' : ''}${dy.toFixed(2)}) px`);
  }
  if (!(after <= before * 1.02 + 1e-9)) return { rec, accepted: false, reason: `reprojection error grew ${before.toFixed(3)} → ${after.toFixed(3)} px` };
  return { rec: work, accepted: true, reason: `offset from the image centre ${shifts.join(', ')}; reprojection ${before.toFixed(3)} → ${after.toFixed(3)} px` };
}

/**
 * Add the not-yet-indexed keyframes to a vocabulary index, yielding every ~30 ms. With a CPU pool
 * the keyframes are quantised on its workers (≤ 2 per worker in flight) and added here in order:
 * the same index as quantising here (exact quantiser), which is also the fallback for a failed job.
 */
async function indexVocabulary(
  vocab: VocabularyIndex,
  ids: readonly number[],
  db: FeatureDatabase,
  cancel: CancelToken | undefined,
  onProgress?: (done: number) => void,
  pool?: CpuPool | null,
): Promise<void> {
  let last = now();
  const todo = ids.filter((id) => db.features.has(id) && !vocab.has(id));
  const jobs: Promise<Uint32Array | null>[] = [];
  const launch = (): void => {
    const fs = db.features.get(todo[jobs.length])!;
    jobs.push(pool!.quantize(vocab, fs).catch(() => null));
  };
  if (pool && pool.size > 0) while (jobs.length < Math.min(todo.length, 2 * pool.size)) launch();
  for (let k = 0; k < todo.length; k++) {
    const fs = db.features.get(todo[k])!;
    const words = k < jobs.length ? await jobs[k] : null;
    if (k < jobs.length && jobs.length < todo.length) launch();
    if (words) vocab.addBow(todo[k], vocab.bowFromWords(words));
    else vocab.add(todo[k], fs);
    if (now() - last > 30) {
      checkCancel(cancel, 'matching');
      onProgress?.(ids.length - todo.length + k + 1);
      await tick();
      last = now();
    }
  }
}

/**
 * Fraction of the verified two-view correspondences between registered images that the model
 * explains: both features observe the SAME 3-D point. A model that failed to close a loop keeps the
 * start and the end of the loop in separate tracks (the loop pairs' correspondences are split), so
 * its ratio is lower, while its reprojection error can be as low as that of a closed model (e2e
 * yard-real: live 0.392 px with a 17 cm loop gap vs offline 0.393 px).
 */
export function explainedCorrespondenceRatio(rec: Reconstruction, db: FeatureDatabase): number {
  let total = 0,
    explained = 0;
  for (const [a, b, g] of db.pairs()) {
    const ia = rec.images.get(a),
      ib = rec.images.get(b);
    if (!ia || !ib || !ia.registered || !ib.registered) continue;
    const inl = g.inliers;
    for (let k = 0; k < inl.length; k += 2) {
      const pa = ia.point3D[inl[k]];
      total++;
      if (pa >= 0 && pa === ib.point3D[inl[k + 1]]) explained++;
    }
  }
  return total > 0 ? explained / total : 0;
}

/**
 * Final R_cam_device check on the refined poses, by the rule of the live check
 * (`LiveSession.maybeVerifyCameraFromDevice`): only a hypothesis that explains the gyro clearly
 * best (median < 4° and under half the runner-up's) is switched to and marked verified. An
 * ambiguous result verifies nothing: the gyro gates of later refines trust a verified θ.
 */
export function handEyeDecision(r: HandEyeResult, current: ScreenAngle | null): { switchTo: ScreenAngle | null; verified: boolean } {
  const sorted = [...r.medianErrDeg].sort((x, y) => x - y);
  const eb = r.medianErrDeg[r.best / 90];
  if (r.pairs < 4 || !(eb < 4 && eb < 0.5 * sorted[1])) return { switchTo: null, verified: false };
  return { switchTo: r.best !== current ? r.best : null, verified: true };
}

/** Options of `splitPairs`. */
export interface SplitPairOptions {
  /** A SPLIT pair has at least this many split correspondences and this fraction of its inliers split (defaults 15, 0.25). */
  minSplit: number;
  minSplitFraction: number;
  /**
   * A split pair is LOOP-LIKE when the median reprojection error (px) of a's points in b over its
   * split correspondences is above `minReprojPx` (default 8: twice the 4 px track-merge threshold)
   * and at most `maxReprojFocal` × the focal length (default 0.4 ⇒ 300 px at f 750).
   */
  minReprojPx: number;
  maxReprojFocal: number;
}

export const DEFAULT_SPLIT_PAIR_OPTIONS: Readonly<SplitPairOptions> = Object.freeze({ minSplit: 15, minSplitFraction: 0.25, minReprojPx: 8, maxReprojFocal: 0.4 });

/** Result of `splitPairs`. */
export interface SplitPairStats {
  /** Verified pairs with both images registered, their correspondences, and the share split between different 3-D points. */
  pairs: number;
  correspondences: number;
  splitRatio: number;
  /** Split pairs and their split correspondences. */
  splitPairs: number;
  splitCorrespondences: number;
  /** Loop-like split pairs and their split correspondences. */
  loopPairs: number;
  loopCorrespondences: number;
  /** The loop-like split pairs (image ids) and their split correspondences. */
  loops: { a: number; b: number; split: number }[];
  /** The split pairs, most split first: "a-b split/inliers rotation° reprojection px" (diagnostics). */
  detail: string[];
}

/**
 * Verified pairs whose correspondences a model keeps on DIFFERENT 3-D points (both features observe
 * a point, but not the same one). A model that did not close a loop triangulates both ends of it
 * separately, so the loop pairs are split — many of them, each mostly — and a's points land in b
 * a few to a few hundred pixels from b's features (the loop gap: e2e yard-real live models 12–15 px,
 * 50–255 px and beyond 0.4 f; rotation 0–8° off the model's). A closed model's split pairs are false pairs on
 * repeated structure, which it rightly keeps apart: few correspondences in total, mostly hundreds to
 * thousands of pixels off (e2e yard-real, 168 k correspondences: 40–85 split pairs with 950–1940
 * correspondences, 60–190 of them loop-like; unclosed live models: 95–130 split pairs with
 * 2950–5650, of them ≥ 1200 loop-like when the gap is ≤ 0.4 f; object-real: none).
 */
export function splitPairs(rec: Reconstruction, db: FeatureDatabase, opts: Partial<SplitPairOptions> = {}): SplitPairStats {
  const o = { ...DEFAULT_SPLIT_PAIR_OPTIONS, ...opts };
  let pairs = 0,
    total = 0,
    split = 0,
    nSplit = 0,
    splitCorr = 0,
    nLoop = 0,
    loopCorr = 0;
  const loops: SplitPairStats['loops'] = [];
  const detail: { key: string; n: number }[] = [];
  const Rba = new Float64Array(9);
  const errs: number[] = [];
  for (const [a, b, g] of db.pairs()) {
    const ia = rec.images.get(a),
      ib = rec.images.get(b);
    if (!ia || !ib || !ia.registered || !ib.registered) continue;
    pairs++;
    const inl = g.inliers;
    const n = inl.length >> 1;
    let ns = 0;
    for (let k = 0; k < inl.length; k += 2) {
      const pa = ia.point3D[inl[k]],
        pb = ib.point3D[inl[k + 1]];
      if (pa >= 0 && pb >= 0 && pa !== pb) ns++;
    }
    total += n;
    split += ns;
    if (ns < Math.max(o.minSplit, o.minSplitFraction * n)) continue;
    nSplit++;
    splitCorr += ns;
    // where a's point of each split correspondence lands in b
    const cam = rec.cameras.get(ib.cameraId)!;
    const kp = ib.keypoints;
    errs.length = 0;
    for (let k = 0; k < inl.length && kp; k += 2) {
      const pa = ia.point3D[inl[k]],
        pb = ib.point3D[inl[k + 1]];
      if (!(pa >= 0 && pb >= 0 && pa !== pb)) continue;
      const X = rec.points.get(pa)?.xyz;
      if (X) errs.push(Math.sqrt(reprojectionErrorSq(cam, ib.pose, X[0], X[1], X[2], kp[4 * inl[k + 1]], kp[4 * inl[k + 1] + 1])));
    }
    const med = errs.length ? median(errs) : NaN;
    if (med > o.minReprojPx && med <= o.maxReprojFocal * cam.f) {
      nLoop++;
      loopCorr += ns;
      loops.push({ a, b, split: ns });
    }
    // diagnostics: the pair's rotation against the model's relative rotation R_b R_aᵀ
    let errDeg = NaN;
    if (g.R) {
      const A = ia.pose.R,
        B = ib.pose.R;
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) Rba[3 * i + j] = B[3 * i] * A[3 * j] + B[3 * i + 1] * A[3 * j + 1] + B[3 * i + 2] * A[3 * j + 2];
      let tr = 0;
      for (let k = 0; k < 9; k++) tr += Rba[k] * g.R[k];
      errDeg = (Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2))) * 180) / Math.PI;
    }
    detail.push({ key: `${a}-${b} ${ns}/${n} ${errDeg.toFixed(1)}° ${med.toFixed(0)}px`, n: ns });
  }
  detail.sort((x, y) => y.n - x.n);
  return {
    pairs,
    correspondences: total,
    splitRatio: total > 0 ? split / total : 0,
    splitPairs: nSplit,
    splitCorrespondences: splitCorr,
    loopPairs: nLoop,
    loopCorrespondences: loopCorr,
    loops,
    detail: detail.map((d) => d.key),
  };
}

/** Verdict of `liveModelHealth`. */
export interface LiveModelHealth {
  /** True: the live model needs no offline SfM to be checked against. */
  healthy: boolean;
  /** The first symptom found, or the evidence of health. */
  reason: string;
  explained: number;
  split: SplitPairStats | null;
}

/** Thresholds of `liveModelHealth`. */
export interface LiveHealthOptions {
  /** Registered images needed to judge (default 20). */
  minImages: number;
  /** Explained share of the verified correspondences (default 0.7; e2e live and offline models 0.83–0.96). */
  minExplained: number;
  /** Split correspondences on loop-like split pairs: at most max(abs, fraction × all) (defaults 150, 0.25 %). */
  maxLoopCorrespondences: number;
  maxLoopFraction: number;
  /** Split correspondences on any split pair: at most max(abs, fraction × all) (defaults 1000, 1.5 %). */
  maxSplitCorrespondences: number;
  maxSplitFraction: number;
  /**
   * No loop-like split pair may join keyframes this fraction of the capture apart or more (in
   * `keyframes` order; default 1/3): the two ends of a loop the model did not close.
   */
  maxLoopSpan: number;
  split: Partial<SplitPairOptions>;
}

export const DEFAULT_LIVE_HEALTH_OPTIONS: Readonly<Omit<LiveHealthOptions, 'split'>> = Object.freeze({
  minImages: 20,
  minExplained: 0.7,
  maxLoopCorrespondences: 150,
  maxLoopFraction: 0.0025,
  maxSplitCorrespondences: 1000,
  maxSplitFraction: 0.015,
  maxLoopSpan: 1 / 3,
});

/**
 * Whether the refined live model needs the from-scratch offline SfM as an independent check. That
 * SfM rescued live models made of fragments joined by a wrong rotation (gyro breaks), with
 * misplaced (gyro outliers) or unregistered keyframes, and with an unclosed loop (e2e/README
 * "Real-capture robustness"); a live model without those symptoms has been kept in every e2e run
 * (it ties with the offline model: object-real 0.582 vs 0.587 px, 95.7 vs 96.4 % explained;
 * yard-real 0.382 vs 0.384 px, 86.6 vs 87.3 %) — and the offline SfM is 40–50 % of refine's
 * reconstruction time. Healthy when ALL hold (a deterministic function of the model and the pairs):
 *  - every keyframe of `keyframes` is registered, and at least `minImages`;
 *  - the gyro is verified (`gyro` not null) and the model has no gyro breaks or outliers;
 *  - no keyframe of the model was registered with the gyro-predicted rotation (`gyroAided`,
 *    IncrementalMapper.knownRotationImages): its rotation agrees with the gyro by construction, so
 *    the gyro checks are no evidence for it, and it makes "every keyframe registered" easy. A live
 *    model registered through the gyro path drifted unnoticed (e2e yard-real at stride 3 from frame
 *    1, the gyro path alone: 49/49 registered, 0.5 m / 3° apart at the loop's ends, ATE 6.9e-3, 16
 *    loop-like split correspondences — healthy by the other rules);
 *  - at least `minExplained` of the verified correspondences are explained (same 3-D point);
 *  - no unclosed loop (`splitPairs`): few correspondences on loop-like split pairs (a loop gap of a
 *    few to a few hundred pixels), and few on split pairs at all (any gap). The thresholds sit
 *    between the e2e closed models (≤ 190 loop-like / ≤ 1940 split correspondences of 168 k on
 *    yard-real: limits 420 / 2520; none on object-real) and the unclosed live models (≥ 1200
 *    loop-like, or 2950–5650 split);
 *  - and no loop-like split pair across ≥ `maxLoopSpan` of the capture: the two ends of a loop
 *    the model did not close. A phone-rate capture has few, short loop tracks, so the counts above
 *    miss it: e2e yard-real at stride 3 from frame 1 is mapped live with the focal 2.8 % long (the
 *    self-calibration of the first ≈ 8 keyframes: weak geometry, far points), its loop's ends ≈ 11°
 *    apart before refine; refine closed it in 11 of 13 runs (ATE 9.6e-4–1.0e-3) and left it open
 *    in 2 (ATE 1.8e-2, rotation 2.4°) on only 47–69 loop-like correspondences (limit 150) — on the
 *    loop's own pairs, keyframes 1–2 ↔ 42–44 (≈ 210 px). The closed models' loop-like pairs join
 *    keyframes ≤ 17 % of the capture apart (yard-real: 14 of 81 at stride 1, 7 of 48 at stride 3);
 *    a 12 px split across the capture in one closed model only costs the offline SfM's time.
 * Any doubt runs the offline SfM, as before: a wrong "unhealthy" only costs time.
 */
export function liveModelHealth(
  rec: Reconstruction,
  db: FeatureDatabase,
  keyframes: readonly number[],
  gyro: GyroConsistency | null,
  opts: Partial<LiveHealthOptions> = {},
  gyroAided: Iterable<number> = [],
): LiveModelHealth {
  const o = { ...DEFAULT_LIVE_HEALTH_OPTIONS, ...opts };
  const out = (healthy: boolean, reason: string, explained = NaN, split: SplitPairStats | null = null): LiveModelHealth => ({ healthy, reason, explained, split });
  let missing = 0;
  for (const id of keyframes) if (db.features.has(id) && !rec.images.get(id)?.registered) missing++;
  if (missing > 0) return out(false, `${missing} of ${keyframes.length} keyframes not registered`);
  const n = rec.numRegisteredImages;
  if (n < o.minImages) return out(false, `only ${n} registered images (< ${o.minImages})`);
  if (!gyro) return out(false, 'no verified gyro to check it against');
  let aided = 0;
  for (const id of gyroAided) if (rec.images.get(id)?.registered) aided++;
  if (aided > 0) return out(false, `${aided} keyframe(s) registered with the gyro-predicted rotation (no independent check)`);
  if (gyro.breaksWithoutOutliers > 0) return out(false, `${gyro.breaksWithoutOutliers} gyro break(s)`);
  if (gyro.outliers.length > 0) return out(false, `${gyro.outliers.length} keyframe(s) disagree with the gyro`);
  const explained = explainedCorrespondenceRatio(rec, db);
  if (!(explained >= o.minExplained)) return out(false, `${(100 * explained).toFixed(1)} % of the correspondences explained (< ${(100 * o.minExplained).toFixed(0)} %)`, explained);
  const sp = splitPairs(rec, db, opts.split);
  const maxLoop = Math.max(o.maxLoopCorrespondences, o.maxLoopFraction * sp.correspondences);
  const maxSplit = Math.max(o.maxSplitCorrespondences, o.maxSplitFraction * sp.correspondences);
  const counts = `${sp.loopCorrespondences} correspondences on ${sp.loopPairs} loop-like and ${sp.splitCorrespondences} on ${sp.splitPairs} split pairs`;
  if (sp.loopCorrespondences > maxLoop || sp.splitCorrespondences > maxSplit)
    return out(false, `unclosed loop? ${counts} (limits ${Math.round(maxLoop)} / ${Math.round(maxSplit)})`, explained, sp);
  const pos = new Map(keyframes.map((id, k) => [id, k]));
  const span = (l: { a: number; b: number }): number => Math.abs((pos.get(l.a) ?? NaN) - (pos.get(l.b) ?? NaN));
  const far = sp.loops.filter((l) => span(l) >= o.maxLoopSpan * keyframes.length);
  if (far.length > 0)
    return out(false, `unclosed loop? loop-like split pairs across the capture: ${far.map((l) => `${l.a}-${l.b} (${l.split})`).join(', ')}`, explained, sp);
  return out(
    true,
    `all ${n} keyframes registered, gyro-consistent (median ${gyro.medianErrDeg.toFixed(2)}°), ${(100 * explained).toFixed(1)} % explained, ${counts}`,
    explained,
    sp,
  );
}

/** Full refine pipeline. */
export async function runRefine(deps: RefineDeps, progress?: ProgressCallback, cancel?: CancelToken): Promise<RefineOutput> {
  const live = deps.live;
  const st = live.state;
  const cfg = deps.config;
  const order = st.order.slice();
  if (order.length < 2) throw new Error(`refine needs at least 2 keyframes (have ${order.length})`);
  const stats: Record<string, number | string> = {};
  const report = (stage: string, p: number, message?: string): void => progress?.({ stage, progress: Math.max(0, Math.min(1, p)), message });

  // ---------------------------------------------------------------- matching
  const tMatch = now();
  report('matching', 0, 'vocabulary');
  await tick();
  const vc = cfg.vocabulary ?? {};
  try {
    const trained = st.vocabTrainedOn?.length ?? 0;
    if (!st.vocab || trained < 0.5 * order.length) {
      const sources = order.map((id) => st.db.features.get(id)!).filter((f) => f && f.count > 0);
      if (sources.length >= 2) {
        const vocab = VocabularyIndex.train(sources, { numWords: vc.refineWords ?? ENGINE_DEFAULTS.refineVocabularyWords });
        // indexing quantises every feature (≈ 35 ms per 4k-feature keyframe with SIMD, 260 ms in
        // JS): yield between keyframes so cancel / progress / visibility messages get through
        await indexVocabulary(vocab, order, st.db, cancel, (k) => report('matching', 0.04 * (k / order.length), `vocabulary ${k}/${order.length}`), deps.pool);
        st.vocab = vocab;
        st.vocabTrainedOn = order.slice();
        void deps.store.putBinary('vocab', st.vocab.exportCodebook()).catch(() => undefined);
      }
    } else await indexVocabulary(st.vocab, order, st.db, cancel, undefined, deps.pool);
  } catch (e) {
    if ((e as Error)?.name === 'CancelledError') throw e;
    deps.log('warn', `vocabulary failed: ${(e as Error)?.message ?? e}`);
  }
  checkCancel(cancel, 'matching');
  stats.vocabularyMs = Math.round(now() - tMatch);
  const rp = cfg.refinePairs ?? {};
  const seq = sequentialPairs(order, rp.sequential ?? ENGINE_DEFAULTS.refineSequential);
  const exclude = (a: number, b: number): boolean => st.attempted.has(pairKey(a, b));
  const ret: ImagePair[] = st.vocab ? retrievalPairs(st.vocab, rp.retrieval ?? ENGINE_DEFAULTS.refineRetrieval, { exclude }) : [];
  const spa: ImagePair[] = [];
  if (live.mapper.initialized) {
    const rec = live.mapper.reconstruction;
    const depth = live.medianSceneDepth();
    const cams: SpatialCamera[] = rec.registeredImageIds().map((id) => ({ id, pose: rec.images.get(id)!.pose, depth }));
    spa.push(...spatialPairs(cams, rp.spatial ?? ENGINE_DEFAULTS.refineSpatial, 75));
  }
  const speculative = new Set<number>();
  for (const [a, b] of [...ret, ...spa]) speculative.add(pairKey(a, b));
  for (const [a, b] of seq) speculative.delete(pairKey(a, b));
  // per candidate source (sequential, retrieval, spatial — first that proposed it): pairs, verified, verification ms
  const source = new Map<number, number>();
  for (const [k, list] of [seq, ret, spa].entries()) for (const [a, b] of list) if (!source.has(pairKey(a, b))) source.set(pairKey(a, b), k);
  const bySource = [0, 1, 2].map(() => ({ pairs: 0, verified: 0, ms: 0, failedMs: 0 }));
  const pairs = mergePairs([seq, ret, spa], exclude).filter(([a, b]) => st.db.features.has(a) && st.db.features.has(b));
  stats.candidatePairs = pairs.length;
  const stored: StoredPair[] = [];
  const attempted: number[] = [];
  let verified = 0;
  let lastYield = now();
  // verification compute time (engine thread or pool workers), waiting for the GPU matcher, and
  // (pool) waiting for verification results
  let verifyMs = 0,
    matchWaitMs = 0,
    resultWaitMs = 0;
  // One-deep prefetch: pair i+1 is matched on the GPU while pair i is verified on the CPU
  // (the GPU chain still runs one job at a time). The queued job needs the event loop to encode
  // and submit its work: one macrotask after pair i's matches arrived lets it reach the GPU
  // before the (synchronous) verification starts — otherwise the GPU idles during every
  // verification and the CPU during every match (e2e: 7–10 s of waiting per refine).
  // Rejections surface when awaited; after a cancel the queued job skips its match.
  const putative = (i: number): Promise<MatchPairs | null> | null => {
    if (i >= pairs.length) return null;
    const p = live.putativeMatches(pairs[i][0], pairs[i][1], cancel);
    p.catch(() => undefined);
    return p;
  };
  // With the CPU pool the verifications run on its workers, up to `maxInflight` at a time, while this
  // thread feeds the GPU matcher; their results are gated and recorded in candidate order (the
  // same database / graph as verifying here, which is the fallback for a failed job and for every
  // pair once no worker is left).
  const pool = deps.pool && deps.pool.size > 0 ? deps.pool : null;
  const maxInflight = pool ? 2 * pool.size : 0;
  let poolGone = false;
  /** `skipped`: a job failed after a cancel (nothing to record). */
  type Verification = { geometry: TwoViewGeometry | null; ms: number; skipped?: boolean };
  const verifyHere = (a: number, b: number, m: MatchPairs, maxIterations: number): Verification => {
    const t0 = now();
    const v = live.verifyInputs(a, b, m, { maxIterations });
    const geometry = v ? estimateTwoViewGeometry(v.kp1, v.kp2, m, v.cam1, v.cam2, v.opts) : null;
    return { geometry, ms: now() - t0 };
  };
  const verifyPooled = (a: number, b: number, m: MatchPairs, maxIterations: number): Promise<Verification> => {
    const v = live.verifyInputs(a, b, m, { maxIterations });
    if (!v) return Promise.resolve({ geometry: null, ms: 0 });
    return pool!.verify(v.kp1, v.kp2, m, v.cam1, v.cam2, v.opts).catch((e: unknown) => {
      if (cancel?.cancelled) return { geometry: null, ms: 0, skipped: true };
      deps.log('warn', `pair ${a}-${b}: CPU worker failed (${(e as Error)?.message ?? e}), verifying here`);
      return verifyHere(a, b, m, maxIterations);
    });
  };
  const record = (a: number, b: number, key: number, r: Verification): void => {
    const g = r.geometry ? live.gatePair(a, b, r.geometry) : null;
    const src = bySource[source.get(key) ?? 0];
    src.pairs++;
    src.ms += r.ms;
    verifyMs += r.ms;
    if (g && g.config !== 'degenerate' && g.inliers.length >> 1 >= ENGINE_DEFAULTS.minPairInliers) {
      src.verified++;
      live.addPair(a, b, g);
      stored.push(pairToStored(a, b, g));
      verified++;
    } else src.failedMs += r.ms;
  };
  // pairs of this loop in st.attempted whose outcome is not recorded (pool jobs in flight, a match
  // skipped after a cancel): a cancel takes them out again, so a re-run in this session verifies them
  const unrecorded = new Set<number>();
  const inflight: { a: number; b: number; key: number; done: boolean; result: Promise<Verification> }[] = [];
  const drain = async (keep: number): Promise<void> => {
    while (inflight.length > keep || (inflight.length > 0 && inflight[0].done)) {
      const job = inflight.shift()!;
      const t0 = now();
      const r = await job.result;
      resultWaitMs += now() - t0;
      if (r.skipped) continue;
      record(job.a, job.b, job.key, r);
      unrecorded.delete(job.key);
    }
  };
  let next = putative(0);
  try {
    for (let i = 0; i < pairs.length; i++) {
      checkCancel(cancel, 'matching');
      const [a, b] = pairs[i];
      const key = pairKey(a, b);
      st.attempted.add(key);
      attempted.push(key);
      unrecorded.add(key);
      const cur = next!;
      next = putative(i + 1);
      const maxIterations = speculative.has(key) ? 1000 : 2000;
      try {
        const tw = now();
        const m = await cur;
        if (next) await yieldTask();
        matchWaitMs += now() - tw;
        if (!m) {
          if (!cancel?.cancelled) unrecorded.delete(key); // else: the match was skipped
        } else if (pool && pool.size > 0) {
          const job = { a, b, key, done: false, result: verifyPooled(a, b, m, maxIterations) };
          void job.result.then(() => (job.done = true));
          inflight.push(job);
        } else {
          if (pool && !poolGone) {
            poolGone = true;
            deps.log('warn', 'no usable CPU worker left: verifying the remaining pairs here');
          }
          if (inflight.length) await drain(0); // candidate order
          record(a, b, key, verifyHere(a, b, m, maxIterations));
          unrecorded.delete(key);
        }
        if (inflight.length) await drain(maxInflight);
      } catch (e) {
        // a failed GPU match job: not attempted, the next refine tries the pair again
        unrecorded.delete(key);
        st.attempted.delete(key);
        attempted.splice(attempted.lastIndexOf(key), 1);
        deps.log('warn', `pair ${a}-${b}: ${(e as Error)?.message ?? e}`);
      }
      if (now() - lastYield > 40) {
        report('matching', 0.05 + 0.95 * ((i + 1) / pairs.length), `verified ${verified} of ${i + 1}/${pairs.length} pairs`);
        await tick();
        lastYield = now();
      }
    }
    await drain(0);
    if (unrecorded.size) checkCancel(cancel, 'matching');
  } catch (e) {
    for (const key of unrecorded) st.attempted.delete(key);
    throw e;
  }
  const rec: StoredPairs = { pairs: stored, attempted };
  // (not read back yet: a refine on a live session restored from the store — e.g. after densify
  // released it — matches these pairs again)
  await deps.store.putValue('refine/pairs', rec).catch((e: unknown) => deps.log('warn', `could not persist refine pairs: ${(e as Error)?.message ?? e}`));
  stats.newPairs = verified;
  stats.pairs = st.db.numPairs;
  stats.matchingMs = Math.round(now() - tMatch);
  stats.verifyMs = Math.round(verifyMs);
  stats.matchWaitMs = Math.round(matchWaitMs);
  if (pool) {
    stats.cpuWorkers = pool.size;
    stats.verifyResultWaitMs = Math.round(resultWaitMs);
  }
  stats.verifyBySource = ['sequential', 'retrieval', 'spatial']
    .map((n, k) => `${n} ${bySource[k].verified}/${bySource[k].pairs} ${Math.round(bySource[k].ms)} ms (failed ${Math.round(bySource[k].failedMs)} ms)`)
    .join(', ');
  report('matching', 1, `${st.db.numPairs} verified pairs`);

  // ---------------------------------------------------------------- reconstruction
  const tRec = now();
  report('reconstruction', 0, 'refining the live model');
  await tick();
  let liveRec: Reconstruction | null = null;
  if (live.mapper.initialized) {
    // keyframes that never registered live get another chance with the enriched graph (with the
    // live mapper's neighbour re-triangulation and gyro-aided registration)
    const aided = live.mapper.incrementalMapper.numKnownRotationRegistrations,
      retri = live.mapper.retriangulation.registered;
    for (const id of live.mapper.pendingIds()) {
      checkCancel(cancel, 'reconstruction');
      try {
        live.mapper.addKeyframe(id);
      } catch {
        /* stays unregistered */
      }
    }
    stats.gyroAidedRegistrations = `${aided} live, ${live.mapper.incrementalMapper.numKnownRotationRegistrations - aided} in refine`;
    stats.retriangulatedRegistrations = `${retri} live, ${live.mapper.retriangulation.registered - retri} in refine (${live.mapper.retriangulation.attempts} attempts)`;
    await live.mapper.incrementalMapper.iterativeGlobalRefinement(undefined, cancel, true);
    liveRec = live.mapper.reconstruction;
  }
  checkCancel(cancel, 'reconstruction');
  // gyro bias / camera–IMU offset from the refined live model (the rolling-shutter estimation needs
  // ω at the right time; the live calibration may be off or stale)
  if (liveRec && liveRec.numRegisteredImages >= 12 && cfg.estimateRollingShutter !== false) live.calibrateGyro('refine');
  stats.liveRefineMs = Math.round(now() - tRec);
  // Model choice: gyro consistency first (a model whose consecutive keyframes disagree with the
  // gyro beyond isolated outliers is made of fragments joined by a wrong rotation), then the
  // registered images, then the mean reprojection error.
  const gyro = live.gyroForChecks();
  const consistency = (r: Reconstruction | null): GyroConsistency | null => {
    if (!r || !gyro) return null;
    const imgs = timedRotations(r.registeredImageIds(), (id) => r.images.get(id)?.pose, (id) => st.records.get(id)?.t);
    return gyroConsistency(imgs, gyro, { tolerance: live.gyroTolerance() });
  };
  // Rolling shutter: estimated on the refined live model (when it is gyro-consistent) so that the
  // offline SfM already runs on corrected keypoints; otherwise on the chosen model below.
  const rsEnabled = cfg.estimateRollingShutter !== false && !!gyro;
  let rs: RefineRSResult | null = null;
  const runRS = async (rec: Reconstruction, mapper: IncrementalMapper): Promise<void> => {
    const tRS = now();
    const tm0 = { ...mapper.timings };
    const prev = st.rollingShutter;
    report('reconstruction', 0.08, 'rolling shutter');
    await tick();
    const rsOpts = cfg.rollingShutterOptions ?? {};
    let searchMs = 0,
      readjustMs = 0,
      evaluations = 0;
    const readjust = async (): Promise<void> => {
      const t0 = now();
      // keypoints changed in place: drop cached normalised coordinates and re-adjust
      live.mapper.incrementalMapper.triangulator.invalidate();
      mapper.triangulator.invalidate();
      await mapper.iterativeGlobalRefinement(undefined, cancel, true);
      readjustMs += now() - t0;
    };
    const estimate = async (opts: typeof rsOpts, known: RefineRSResult['applied']): Promise<RefineRSResult> => {
      const t0 = now();
      const out = await refineRollingShutter(rec, st, gyro!, opts, known, deps.pool ?? null);
      searchMs += now() - t0;
      evaluations += out.estimate?.curve.length ?? 0;
      return out;
    };
    let r = await estimate(rsOpts, live.liveRollingShutter());
    const first = r;
    if (r.applied || prev) await readjust();
    // further passes on the model re-adjusted with the estimate (the short BAs start from the input
    // model: built on distorted keypoints it pulls the readout towards 0), narrowed around it
    let passes = 1;
    for (let p = 2; p <= Math.max(1, rsOpts.passes ?? 2) && r.applied && r.estimate; p++) {
      checkCancel(cancel, 'reconstruction');
      // NOT re-calibrated here although the offset calibrated on the rolling-shutter DISTORTED live
      // model is biased by where the features sit in the frame (e2e yard-real −30.5 / abl-yard-rs
      // −27.9 ms, truth −25; re-calibrating on the corrected model gives −25.1 / −24.9 ms and the
      // true bias): measured neutral on abl-*-rs / object-real and worse on yard-real (principal
      // point 0.4 → 1.3 px, rotation 0.10 → 0.13°) — docs/research/real-capture-robustness.md
      const before = r.applied;
      r = await estimate({ ...rsOpts, around: before }, before);
      passes = p;
      await readjust();
      if (!r.applied || r.applied.axis !== before.axis || Math.abs(r.applied.readoutMs - before.readoutMs) < 0.5) break;
    }
    rs = r;
    Object.assign(stats, rsStats(r, st.cameraFromDevice()));
    if (first.estimate) stats.rsImprovementPct = Math.round(1000 * first.estimate.improvement) / 10;
    stats.rsPasses = passes;
    stats.rsMs = Math.round(now() - tRS);
    const tm = mapper.timings;
    stats.rsGlobalBA = `${tm.numGlobalBA - tm0.numGlobalBA} BAs, ${tm.globalBAIterations - tm0.globalBAIterations} iterations, ${Math.round(tm.globalBA - tm0.globalBA)} ms`;
    stats.rsSearchMs = Math.round(searchMs);
    stats.rsReadjustMs = Math.round(readjustMs);
    stats.rsEvaluations = evaluations;
    if (passes > 1 && first.applied) stats.rsPass1 = `${first.applied.axis} ${first.applied.readoutMs.toFixed(1)} ms`;
    deps.log(
      'info',
      `rolling shutter: ${r.applied ? `${r.applied.axis} ${r.applied.readoutMs.toFixed(1)} ms (${r.corrected} keyframes corrected, ${passes} pass(es)${r.translation ? ', with translation' : ''})` : 'none'} — ${first.reason}`,
    );
  };
  if (rsEnabled && liveRec && liveRec.numRegisteredImages >= 10 && (consistency(liveRec)?.breaksWithoutOutliers ?? 0) === 0) {
    await runRS(liveRec, live.mapper.incrementalMapper);
  }
  if (liveRec) {
    extractColors(liveRec, st.db);
    liveRec.updatePointErrors();
  }
  checkCancel(cancel, 'reconstruction');
  const liveCam = liveRec && liveRec.numRegisteredImages >= 3 ? [...liveRec.cameras.values()][0] : undefined;
  const mapperOpts: Partial<RunSfMOptions> = {
    hfovPriorDeg: live.mapper.opts.hfovPriorDeg,
    ...(liveCam ? { intrinsics: { ...liveCam } } : {}),
    // gyro gate of registrations / initial pairs (active once R_cam_device is verified)
    rotationCheck: live.rotationCheck(),
    // gyro-aided registration as live (EngineConfig.gyroAidedRegistration): at a phone's analysis
    // rate the from-scratch SfM otherwise loses the keyframes the live mapper registered with it or
    // with re-triangulated neighbours (e2e yard-real at stride 3 from frame 1: 3 models, the largest
    // 32 of 48 keyframes; with it 1 model of 48), and so could never replace a live model whose
    // loop did not close (docs/research/real-capture-robustness.md "Phone-rate capture")
    ...(cfg.gyroAidedRegistration === false ? {} : { predictRotation: (id: number, r: Reconstruction) => live.predictRotation(id, r) }),
    ...(cfg.mapper ?? {}),
  };
  // the offline SfM checks the live model independently: skipped when the live model is healthy
  const lc = consistency(liveRec);
  const health = liveRec && cfg.refineOfflineSfM !== 'always' ? liveModelHealth(liveRec, st.db, order, lc, {}, live.mapper.incrementalMapper.knownRotationImages) : null;
  stats.offlineSfM = health?.healthy ? `skipped: ${health.reason}` : health ? `run: ${health.reason}` : 'run';
  deps.log('info', `refine: offline SfM ${stats.offlineSfM}`);
  if (health?.split) deps.log('info', `refine: live model split pairs: ${health.split.detail.slice(0, 40).join('; ')}`);
  let offline: Reconstruction | null = null;
  if (!health?.healthy) {
    const tOff = now();
    try {
      const gyroAided = new Map<Reconstruction, number>();
      const models = await runIncrementalSfM(
        st.db,
        { ...mapperOpts, onModel: (r, _t, m) => gyroAided.set(r, m.numKnownRotationRegistrations) },
        (e) => report('reconstruction', 0.1 + 0.85 * Math.max(0, e.progress), e.message),
        cancel,
      );
      offline = models[0] ?? null;
      if (offline) stats.gyroAidedRegistrations = `${stats.gyroAidedRegistrations ?? '0 live, 0 in refine'}, ${gyroAided.get(offline) ?? 0} offline`;
      // only the largest model is kept: report what the others held (split frequency on real captures)
      stats.offlineModels = models.length;
      if (models.length > 1) {
        const kept = new Set(models[0].registeredImageIds());
        const dropped = new Set<number>();
        for (const m of models.slice(1)) for (const id of m.registeredImageIds()) if (!kept.has(id)) dropped.add(id);
        stats.offlineDroppedImages = dropped.size;
        deps.log('warn', `refine: offline SfM found ${models.length} models; ${dropped.size} keyframes only in the ${models.length - 1} smaller one(s) were dropped`);
      }
    } catch (e) {
      if ((e as Error)?.name === 'CancelledError') throw e;
      deps.log('warn', `offline SfM failed: ${(e as Error)?.message ?? e}`);
    }
    stats.offlineSfMMs = Math.round(now() - tOff);
  }
  const score = (r: Reconstruction | null): [number, number] => (r ? [r.numRegisteredImages, r.meanReprojectionError()] : [0, Infinity]);
  const [lr, le] = score(liveRec);
  const [or, oe] = score(offline);
  const oc = consistency(offline);
  stats.liveRegistered = lr;
  if (offline) stats.offlineRegistered = or;
  if (lc) stats.gyroBreaksLive = `${lc.breaksWithoutOutliers} (+${lc.outliers.length} outliers, median ${lc.medianErrDeg.toFixed(2)}°)`;
  if (oc) stats.gyroBreaksOffline = `${oc.breaksWithoutOutliers} (+${oc.outliers.length} outliers, median ${oc.medianErrDeg.toFixed(2)}°)`;
  let chosen: Reconstruction;
  let which: 'live' | 'offline';
  if (!offline && !liveRec) throw new Error('reconstruction failed: no initial pair found (move the camera sideways while scanning)');
  const lb = lc ? lc.breaksWithoutOutliers : 0,
    ob = oc ? oc.breaksWithoutOutliers : 0;
  // explained correspondences: a model that did not close a loop splits the loop pairs' tracks
  const lx = liveRec ? (health && Number.isFinite(health.explained) ? health.explained : explainedCorrespondenceRatio(liveRec, st.db)) : 0,
    ox = offline ? explainedCorrespondenceRatio(offline, st.db) : 0;
  stats.liveExplained = Math.round(1000 * lx) / 1000;
  if (offline) stats.offlineExplained = Math.round(1000 * ox) / 1000;
  const fmtSplit = (sp: SplitPairStats): string =>
    `${(100 * sp.splitRatio).toFixed(2)} % of ${sp.correspondences} correspondences split; ${sp.splitCorrespondences} on ${sp.splitPairs} split pairs, ${sp.loopCorrespondences} on ${sp.loopPairs} loop-like`;
  if (liveRec) {
    const sp = health?.split ?? splitPairs(liveRec, st.db);
    stats.liveSplit = fmtSplit(sp);
    if (!health?.split) deps.log('info', `refine: live model ${stats.liveSplit}; split pairs: ${sp.detail.slice(0, 40).join('; ')}`);
  }
  if (offline) {
    const sp = splitPairs(offline, st.db);
    stats.offlineSplit = fmtSplit(sp);
    deps.log('info', `refine: offline model ${stats.offlineSplit}; split pairs: ${sp.detail.slice(0, 40).join('; ')}`);
  }
  const minExplainedGap = cfg.modelChoiceExplainedGap ?? 0.015;
  const liveBetter = (): boolean => {
    if (lb !== ob) return lb < ob;
    if (lr !== or) return lr > or;
    if (Math.abs(lx - ox) > minExplainedGap) return lx > ox;
    return le <= oe;
  };
  if (!offline || (liveRec && liveBetter())) {
    chosen = liveRec!;
    which = 'live';
  } else {
    chosen = offline;
    which = 'offline';
  }
  const cc = which === 'live' ? lc : oc;
  // work on a copy so the live model stays in its own frame (refine can run again); the offline
  // model is used by nobody else (no copy), and when it lost it is dropped right away: every
  // reconstruction is ≈ 300 B per point + 40 B per observation, and refine otherwise held four
  // (live, offline, chosen copy, principal-point copy) at its peak
  if (which === 'live') chosen = chosen.clone();
  offline = null;
  stats.model = which;
  // isolated gyro outliers (an image misplaced by false matches) are deregistered; the global
  // refinement below re-triangulates without them
  if (cc && cc.outliers.length) {
    const m = new IncrementalMapper(st.db, st.graph, chosen, { ...(which === 'live' ? live.mapper.incrementalMapper.opts : mapperOpts) });
    for (const id of cc.outliers) m.deregisterImage(id);
    await m.iterativeGlobalRefinement(undefined, cancel, true);
    stats.gyroOutliers = cc.outliers.join(',');
    deps.log('warn', `refine: ${cc.outliers.length} keyframe(s) disagree with the gyro and were removed: ${cc.outliers.join(', ')}`);
  }
  if (rsEnabled && rs === null && chosen.numRegisteredImages >= 10) {
    await runRS(chosen, new IncrementalMapper(st.db, st.graph, chosen, { ...(which === 'live' ? live.mapper.incrementalMapper.opts : mapperOpts) }));
  }
  stats.gyroRejectedPairs = live.stats.gyroRejectedPairs;
  stats.gyroRejectedPoses = live.stats.gyroRejectedPoses;
  stats.gyroRecoveredPoses = live.stats.gyroRecoveredPoses;
  stats.forcedKeyframes = live.stats.forcedKeyframes;
  stats.heldKeyframes = live.stats.heldKeyframes;
  // final global BA with the principal point free (well constrained by the many views of a
  // finished capture; fixed during mapping as in COLMAP). A fixed, offset principal point is
  // absorbed by a constant rotation of every camera (≈ 0.14° per 1.8 px on the e2e datasets).
  const ppMin = cfg.refinePrincipalPoint === false ? Infinity : cfg.refinePrincipalPoint === true ? 3 : ENGINE_DEFAULTS.refinePrincipalPointMinImages;
  if (chosen.numRegisteredImages >= ppMin) {
    report('reconstruction', 0.97, 'final bundle adjustment (principal point)');
    const tPP = now();
    try {
      const r = await refinePrincipalPoint(chosen, st.db, st.graph, { ...(which === 'live' ? live.mapper.incrementalMapper.opts : mapperOpts) }, cancel);
      if (r.accepted) chosen = r.rec;
      stats.principalPoint = r.accepted ? 'refined' : `kept (${r.reason})`;
      deps.log(r.accepted ? 'info' : 'warn', `principal point ${r.accepted ? 'refined' : 'kept'}: ${r.reason}`);
    } catch (e) {
      if ((e as Error)?.name === 'CancelledError') throw e;
      stats.principalPoint = 'kept (BA failed)';
      deps.log('warn', `principal point BA failed: ${(e as Error)?.message ?? e}`);
    }
    stats.principalPointMs = Math.round(now() - tPP);
  }
  // per-image focal lengths (focus breathing): only when they lower the reprojection error clearly
  if (cfg.perImageFocal !== false && chosen.numRegisteredImages >= 10) {
    report('reconstruction', 0.985, 'per-image focal');
    const tPF = now();
    try {
      const mo = { ...(which === 'live' ? live.mapper.incrementalMapper.opts : mapperOpts) };
      const ppOn = cfg.refinePrincipalPoint !== false && chosen.numRegisteredImages >= ppMin;
      const pf = await refinePerImageFocal(chosen, st.db, st.graph, mo, {
        sigmaRel: cfg.perImageFocalSigma ?? 0.005,
        minImprovement: cfg.perImageFocalMinImprovement,
        minAutocorrelation: cfg.perImageFocalMinAutocorrelation,
        timeOrder: order,
        rounds: cfg.perImageFocalRounds,
        ppPriorSigmaPx: cfg.perImageFocalPPSigmaPx,
        ppPriorAnchor: cfg.perImageFocalPPAnchor,
        // the per-image rounds pass a throwaway rescaled copy: refine it in place
        refinePrincipalPoint: ppOn
          ? (r, priors) => refinePrincipalPoint(r, st.db, st.graph, priors ? { ...mo, principalPointPriors: priors } : mo, cancel, true)
          : undefined,
        cancel,
      });
      if (pf.accepted) chosen = pf.rec;
      stats.perImageFocal = pf.accepted ? `refined: ${pf.reason}` : `kept shared (${pf.reason})`;
      deps.log('info', `per-image focal ${pf.accepted ? 'refined' : 'kept shared'}: ${pf.reason}`);
    } catch (e) {
      if ((e as Error)?.name === 'CancelledError') throw e;
      stats.perImageFocal = 'kept shared (BA failed)';
      deps.log('warn', `per-image focal BA failed: ${(e as Error)?.message ?? e}`);
    }
    stats.perImageFocalMs = Math.round(now() - tPF);
  }
  stats.reconstructionMs = Math.round(now() - tRec);
  report('reconstruction', 1, `${which} model: ${chosen.numRegisteredImages}/${order.length} images, ${chosen.numPoints} points`);
  deps.log(
    'info',
    `refine: live ${lr} images (${le.toFixed(3)} px, ${(100 * lx).toFixed(1)} % explained), ${health?.healthy ? 'offline SfM skipped' : `offline ${or} images (${oe.toFixed(3)} px, ${(100 * ox).toFixed(1)} % explained)`} → ${which}`,
  );

  // ---------------------------------------------------------------- alignment
  const tAl = now();
  report('alignment', 0, 'gravity, scale and north');
  await tick();
  checkCancel(cancel, 'alignment');
  const kfs = chosen
    .registeredImageIds()
    .map((id) => ({ id, rec: st.records.get(id)!, pose: chosen.images.get(id)!.pose }))
    .filter((k) => !!k.rec);
  // final R_cam_device check with the refined poses
  if (cfg.verifyCameraFromDevice !== false && !st.cameraThetaVerified && st.imu.samples.length > 10) {
    const r = handEyeCheck(
      kfs.map((k) => ({ t: k.rec.t, pose: k.pose })),
      st.imu,
      { offsetMs: st.timeOffsetMs ?? 0 },
    );
    if (r) {
      const d = handEyeDecision(r, st.cameraTheta);
      if (d.switchTo !== null) {
        deps.log('warn', `R_cam_device check (refined poses): switching θ ${st.cameraTheta} → ${d.switchTo}`, r);
        live.setCameraTheta(d.switchTo);
      }
      if (d.verified) st.cameraThetaVerified = true;
    }
  }
  let alignment: WorldAlignment | null = null;
  try {
    const wk: WorldKeyframe[] = kfs.map((k) => ({
      id: k.id,
      t: k.rec.t,
      pose: k.pose,
      R_cam_device: k.rec.info.R_cam_device,
      gravityCam: k.rec.info.gravityCam,
      heading: k.rec.info.heading,
      headingAccuracy: k.rec.headingAccuracy,
      // Absent `headingNorth` = 'true': the FrameInfo.heading / HeadingSample contract (true north
      // unless flagged 'magnetic'; callers without a north reference, e.g. tests and the harness'
      // true-north replays, send none). Since the WMM change the app flags every heading it sends
      // (capture.ts: 'magnetic' until CompassSensor knows the declination). App builds before that
      // change sent unflagged MAGNETIC headings, so their keyframes would be read as true north here
      // (yaw off by the local declination) — the app was never released, so no such stored session
      // exists; a released app would need a session-version check to read them as 'magnetic'.
      headingNorth: k.rec.headingNorth ?? 'true',
      geo: k.rec.info.geo,
    }));
    const input = { keyframes: wk, imu: st.imu.samples, gpsFixes: st.geo };
    alignment = computeWorldAlignment(input, { timeOffsetMs: st.timeOffsetMs ?? 0 });
    const lowEx = lowExcitationRetry(alignment);
    if (lowEx) {
      // gentle hand-held motion (smooth orbits) or sparse keyframes: the IMU scale fit is excellent
      // but below the excitation floor of the inertial module → accept it with an inflated σ, unless
      // the keyframe-interval scale already used is more precise
      const a2 = computeWorldAlignment(input, { timeOffsetMs: st.timeOffsetMs ?? 0, scale: { minExcitation: lowEx.minExcitation } });
      const sigma = lowExcitationSigma(a2.metric.relativeUncertainty, lowEx.r2);
      const prior = alignment.metric.metric ? alignment.metric.relativeUncertainty : null;
      if (a2.metric.metric && a2.metric.source === 'imu' && (prior === null || sigma < prior)) {
        a2.metric.relativeUncertainty = sigma;
        a2.details.warnings.push(
          `IMU scale accepted at low excitation (${lowEx.excitation.toFixed(3)} m/s², R² ${lowEx.r2.toFixed(2)}, σ ${(100 * sigma).toFixed(1)} %)` +
            (prior !== null ? ` instead of the keyframe-interval scale (σ ${(100 * prior).toFixed(1)} %)` : ''),
        );
        alignment = a2;
      }
    }
    chosen.transform(alignment.s, alignment.R, alignment.t);
    chosen.metric = alignment.metric;
    if (alignment.details.offsetSource !== 'default') st.timeOffsetMs = alignment.details.offsetMs;
    stats.timeOffsetMs = Math.round(alignment.details.offsetMs * 10) / 10;
    stats.timeOffsetSource = alignment.details.offsetSource;
    stats.scaleSource = alignment.metric.source;
    if (alignment.metric.source === 'imu' || alignment.metric.source === 'imu+gps')
      stats.scaleMethod = alignment.details.imuScale?.reliable ? 'bandpass' : 'intervals';
    stats.yawSource = alignment.details.yawSource;
    Object.assign(stats, alignment.details.headingStats); // north reference, declination, yaw σ / fusion
    if (alignment.metric.relativeUncertainty !== null) stats.scaleUncertainty = alignment.metric.relativeUncertainty;
    if (alignment.details.gravity) stats.gravityStdDeg = (alignment.details.gravity.stdRad * 180) / Math.PI;
    if (alignment.details.warnings.length) {
      stats.alignmentWarnings = alignment.details.warnings.join('; ');
      deps.log('info', `alignment: ${alignment.details.warnings.join('; ')}`);
    }
  } catch (e) {
    deps.log('warn', `world alignment failed: ${(e as Error)?.message ?? e}`);
  }
  stats.alignmentMs = Math.round(now() - tAl);
  report('alignment', 1);

  // ---------------------------------------------------------------- result
  const model = modelFromReconstruction(chosen, order);
  // the keypoints were corrected for rolling shutter: the dense stage corrects the images too (with
  // the ω / translation each keyframe's keypoints were corrected with), exports re-derive them
  const srs = sparseRollingShutter(st, chosen, model.images.map((im) => im.id));
  if (srs) model.rollingShutter = srs;
  const cameras: SparseCameraEx[] = model.images.map((im) => {
    const r = st.records.get(im.id);
    const upCam = r ? keyframeUpCam(r) : undefined;
    return {
      id: im.id,
      pose: { R: Float64Array.from(im.pose.R), t: Float64Array.from(im.pose.t) },
      intrinsics: { ...model.cameras.get(im.cameraId)! },
      t: r?.t ?? NaN,
      frameIndex: r?.frameIndex ?? -1,
      ...(upCam ? { upCam } : {}),
    };
  });
  const cam0 = cameras[0]?.intrinsics;
  const trackLens: number[] = [];
  for (let i = 0; i < model.numPoints; i++) trackLens.push(model.trackOffsets[i + 1] - model.trackOffsets[i]);
  // keyframes the final model could not connect (the viewer says so and offers to add views)
  const unconnected = order.length - model.images.length;
  if (unconnected > 0) deps.log('warn', `refine: ${unconnected} of ${order.length} keyframes could not be connected to the model`);
  Object.assign(stats, {
    images: order.length,
    registeredImages: model.images.length,
    unconnectedImages: unconnected,
    points: model.numPoints,
    meanReprojErrorPx: Math.round(chosen.meanReprojectionError() * 1000) / 1000,
    meanTrackLength: Math.round(chosen.meanTrackLength() * 100) / 100,
    medianTrackLength: trackLens.length ? median(trackLens) : 0,
    ...(cam0 ? { focalPx: Math.round(median(cameras.map((c) => c.intrinsics.f)) * 10) / 10, k1: cam0.k1, k2: cam0.k2 } : {}),
  });
  const result: SparseResultEx = {
    points: modelPointCloud(model),
    cameras,
    metric: { ...chosen.metric },
    stats,
  };
  return { model, result, rec: chosen, alignment };
}
