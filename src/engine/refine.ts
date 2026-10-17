/**
 * Refine after capture (ENGINE.md "Refine"): vocabulary retrain / extend → candidate pairs
 * (sequential 10, retrieval 15, spatial 10) → match + verify → full incremental SfM from scratch
 * (live-refined intrinsics as the initial guess) → keep the better of the live and the offline
 * model (registered images, then mean reprojection error) → final global BA with
 * self-calibration (f, k1, k2), re-triangulation and filtering → world alignment (IMU gravity /
 * time offset / metric scale, GPS, compass; verified R_cam_device) → SparseModel.
 *
 * Progress stages: 'matching', 'reconstruction', 'alignment' (src/app/stages.ts).
 */
import type { CancelToken, PointCloud, ProgressCallback, TwoViewGeometry } from '../core/types';
import { mergePairs, retrievalPairs, sequentialPairs, spatialPairs, type ImagePair, type SpatialCamera } from '../sfm/pairs';
import type { CorrespondenceGraph } from '../sfm/correspondence-graph';
import { pairKey, type FeatureDatabase } from '../sfm/database';
import { extractColors, IncrementalMapper, runIncrementalSfM, SfMCancelledError, type MapperOptions, type RunSfMOptions } from '../sfm/mapper';
import type { Reconstruction } from '../sfm/reconstruction';
import { VocabularyIndex } from '../sfm/vocab';
import { computeWorldAlignment, type WorldAlignment, type WorldKeyframe } from '../sfm/world';
import { median } from '../math/stats';
import { handEyeCheck, type LiveSession } from './live';
import { pairToStored, type EngineStore, type StoredPair, type StoredPairs } from './persistence';
import { ENGINE_DEFAULTS } from './presets';
import type { EngineConfig, SparseCameraEx, SparseModel, SparseResultEx } from './types';

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

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
 * When the only reason the IMU scale was rejected is low excitation, but the fit itself is
 * excellent, return the relaxed threshold to retry with: relative σ ≤ 3 % and either R² ≥ 0.9 with
 * excitation ≥ 0.03 m/s², or R² ≥ 0.95 with excitation ≥ 0.015 m/s².
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
  if (!s || s.reliable || a.metric.metric) return null;
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
 */
export async function refinePrincipalPoint(
  rec: Reconstruction,
  db: FeatureDatabase,
  graph: CorrespondenceGraph,
  opts: Partial<MapperOptions>,
  cancel?: CancelToken,
): Promise<{ rec: Reconstruction; accepted: boolean; reason: string }> {
  const work = rec.clone();
  const before = rec.meanReprojectionError();
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

/** Add the not-yet-indexed keyframes to a vocabulary index, yielding every ~30 ms. */
async function indexVocabulary(
  vocab: VocabularyIndex,
  ids: readonly number[],
  db: FeatureDatabase,
  cancel: CancelToken | undefined,
  onProgress?: (done: number) => void,
): Promise<void> {
  let last = now();
  for (let k = 0; k < ids.length; k++) {
    const fs = db.features.get(ids[k]);
    if (fs && !vocab.has(ids[k])) vocab.add(ids[k], fs);
    if (now() - last > 30) {
      checkCancel(cancel, 'matching');
      onProgress?.(k + 1);
      await tick();
      last = now();
    }
  }
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
        await indexVocabulary(vocab, order, st.db, cancel, (k) => report('matching', 0.04 * (k / order.length), `vocabulary ${k}/${order.length}`));
        st.vocab = vocab;
        st.vocabTrainedOn = order.slice();
        void deps.store.putBinary('vocab', st.vocab.exportCodebook()).catch(() => undefined);
      }
    } else await indexVocabulary(st.vocab, order, st.db, cancel);
  } catch (e) {
    if ((e as Error)?.name === 'CancelledError') throw e;
    deps.log('warn', `vocabulary failed: ${(e as Error)?.message ?? e}`);
  }
  checkCancel(cancel, 'matching');
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
  const pairs = mergePairs([seq, ret, spa], exclude).filter(([a, b]) => st.db.features.has(a) && st.db.features.has(b));
  stats.candidatePairs = pairs.length;
  const stored: StoredPair[] = [];
  const attempted: number[] = [];
  let verified = 0;
  let lastYield = now();
  for (let i = 0; i < pairs.length; i++) {
    checkCancel(cancel, 'matching');
    const [a, b] = pairs[i];
    const key = pairKey(a, b);
    st.attempted.add(key);
    attempted.push(key);
    let g: TwoViewGeometry | null = null;
    try {
      g = await live.matchPair(a, b, { maxIterations: speculative.has(key) ? 1000 : 2000 });
    } catch (e) {
      deps.log('warn', `pair ${a}-${b}: ${(e as Error)?.message ?? e}`);
    }
    if (g && g.config !== 'degenerate' && g.inliers.length >> 1 >= ENGINE_DEFAULTS.minPairInliers) {
      live.addPair(a, b, g);
      stored.push(pairToStored(a, b, g));
      verified++;
    }
    if (now() - lastYield > 40) {
      report('matching', 0.05 + 0.95 * ((i + 1) / pairs.length), `verified ${verified} of ${i + 1}/${pairs.length} pairs`);
      await tick();
      lastYield = now();
    }
  }
  const rec: StoredPairs = { pairs: stored, attempted };
  await deps.store.putValue('refine/pairs', rec).catch((e: unknown) => deps.log('warn', `could not persist refine pairs: ${(e as Error)?.message ?? e}`));
  stats.newPairs = verified;
  stats.pairs = st.db.numPairs;
  stats.matchingMs = Math.round(now() - tMatch);
  report('matching', 1, `${st.db.numPairs} verified pairs`);

  // ---------------------------------------------------------------- reconstruction
  const tRec = now();
  report('reconstruction', 0, 'refining the live model');
  await tick();
  let liveRec: Reconstruction | null = null;
  if (live.mapper.initialized) {
    // keyframes that never registered live get another chance with the enriched graph
    for (const id of live.mapper.pendingIds()) {
      checkCancel(cancel, 'reconstruction');
      try {
        live.mapper.addKeyframe(id);
      } catch {
        /* stays unregistered */
      }
    }
    await live.mapper.incrementalMapper.iterativeGlobalRefinement(undefined, cancel, true);
    liveRec = live.mapper.reconstruction;
    extractColors(liveRec, st.db);
    liveRec.updatePointErrors();
  }
  checkCancel(cancel, 'reconstruction');
  const liveCam = liveRec && liveRec.numRegisteredImages >= 3 ? [...liveRec.cameras.values()][0] : undefined;
  const mapperOpts: Partial<RunSfMOptions> = {
    hfovPriorDeg: live.mapper.opts.hfovPriorDeg,
    ...(liveCam ? { intrinsics: { ...liveCam } } : {}),
    ...(cfg.mapper ?? {}),
  };
  let offline: Reconstruction | null = null;
  try {
    const models = await runIncrementalSfM(
      st.db,
      mapperOpts,
      (e) => report('reconstruction', 0.1 + 0.85 * Math.max(0, e.progress), e.message),
      cancel,
    );
    offline = models[0] ?? null;
  } catch (e) {
    if ((e as Error)?.name === 'CancelledError') throw e;
    deps.log('warn', `offline SfM failed: ${(e as Error)?.message ?? e}`);
  }
  const score = (r: Reconstruction | null): [number, number] => (r ? [r.numRegisteredImages, r.meanReprojectionError()] : [0, Infinity]);
  const [lr, le] = score(liveRec);
  const [or, oe] = score(offline);
  stats.liveRegistered = lr;
  stats.offlineRegistered = or;
  let chosen: Reconstruction;
  let which: 'live' | 'offline';
  if (!offline && !liveRec) throw new Error('reconstruction failed: no initial pair found (move the camera sideways while scanning)');
  if (!offline || (liveRec && (lr > or || (lr === or && le <= oe)))) {
    chosen = liveRec!;
    which = 'live';
  } else {
    chosen = offline;
    which = 'offline';
  }
  // work on a copy so the live model stays in its own frame (refine can run again); the offline
  // model is used by nobody else (no copy), and when it lost it is dropped right away: every
  // reconstruction is ≈ 300 B per point + 40 B per observation, and refine otherwise held four
  // (live, offline, chosen copy, principal-point copy) at its peak
  if (which === 'live') chosen = chosen.clone();
  offline = null;
  stats.model = which;
  // final global BA with the principal point free (well constrained by the many views of a
  // finished capture; fixed during mapping as in COLMAP). A fixed, offset principal point is
  // absorbed by a constant rotation of every camera (≈ 0.14° per 1.8 px on the e2e datasets).
  const ppMin = cfg.refinePrincipalPoint === false ? Infinity : cfg.refinePrincipalPoint === true ? 3 : ENGINE_DEFAULTS.refinePrincipalPointMinImages;
  if (chosen.numRegisteredImages >= ppMin) {
    report('reconstruction', 0.97, 'final bundle adjustment (principal point)');
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
  }
  stats.reconstructionMs = Math.round(now() - tRec);
  report('reconstruction', 1, `${which} model: ${chosen.numRegisteredImages}/${order.length} images, ${chosen.numPoints} points`);
  deps.log('info', `refine: live ${lr} images (${le.toFixed(3)} px), offline ${or} images (${oe.toFixed(3)} px) → ${which}`);

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
    if (r && r.pairs >= 4) {
      const sorted = [...r.medianErrDeg].sort((x, y) => x - y);
      const best = r.best;
      if (best !== st.cameraTheta && r.medianErrDeg[best / 90] < 4 && r.medianErrDeg[best / 90] < 0.5 * sorted[1]) {
        deps.log('warn', `R_cam_device check (refined poses): switching θ ${st.cameraTheta} → ${best}`, r);
        live.setCameraTheta(best);
      }
      if (r.medianErrDeg[best / 90] < 4) st.cameraThetaVerified = true;
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
      geo: k.rec.info.geo,
    }));
    const input = { keyframes: wk, imu: st.imu.samples, gpsFixes: st.geo };
    alignment = computeWorldAlignment(input, { timeOffsetMs: st.timeOffsetMs ?? 0 });
    const lowEx = lowExcitationRetry(alignment);
    if (lowEx) {
      // gentle hand-held motion (smooth orbits) or sparse keyframes: the IMU scale fit is excellent
      // but below the excitation floor of the inertial module → accept it with an inflated σ
      const a2 = computeWorldAlignment(input, { timeOffsetMs: st.timeOffsetMs ?? 0, scale: { minExcitation: lowEx.minExcitation } });
      if (a2.metric.metric && a2.metric.source === 'imu') {
        a2.metric.relativeUncertainty = lowExcitationSigma(a2.metric.relativeUncertainty, lowEx.r2);
        a2.details.warnings.push(`IMU scale accepted at low excitation (${lowEx.excitation.toFixed(3)} m/s², R² ${lowEx.r2.toFixed(2)})`);
        alignment = a2;
      }
    }
    chosen.transform(alignment.s, alignment.R, alignment.t);
    chosen.metric = alignment.metric;
    if (alignment.details.offsetSource !== 'default') st.timeOffsetMs = alignment.details.offsetMs;
    stats.timeOffsetMs = Math.round(alignment.details.offsetMs * 10) / 10;
    stats.timeOffsetSource = alignment.details.offsetSource;
    stats.scaleSource = alignment.metric.source;
    stats.yawSource = alignment.details.yawSource;
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
  const cameras: SparseCameraEx[] = model.images.map((im) => {
    const r = st.records.get(im.id);
    return {
      id: im.id,
      pose: { R: Float64Array.from(im.pose.R), t: Float64Array.from(im.pose.t) },
      intrinsics: { ...model.cameras.get(im.cameraId)! },
      t: r?.t ?? NaN,
      frameIndex: r?.frameIndex ?? -1,
    };
  });
  const cam0 = cameras[0]?.intrinsics;
  const trackLens: number[] = [];
  for (let i = 0; i < model.numPoints; i++) trackLens.push(model.trackOffsets[i + 1] - model.trackOffsets[i]);
  Object.assign(stats, {
    images: order.length,
    registeredImages: model.images.length,
    points: model.numPoints,
    meanReprojErrorPx: Math.round(chosen.meanReprojectionError() * 1000) / 1000,
    meanTrackLength: Math.round(chosen.meanTrackLength() * 100) / 100,
    medianTrackLength: trackLens.length ? median(trackLens) : 0,
    ...(cam0 ? { focalPx: Math.round(cam0.f * 10) / 10, k1: cam0.k1, k2: cam0.k2 } : {}),
  });
  const result: SparseResultEx = {
    points: modelPointCloud(model),
    cameras,
    metric: { ...chosen.metric },
    stats,
  };
  return { model, result, rec: chosen, alignment };
}
