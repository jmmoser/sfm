/**
 * Gaussian-splatting stage (docs/ENGINE.md "Splat", src/splat/index.ts): trains
 * `SplatTrainer` on the registered keyframes after densify (or after refine without a dense cloud).
 *
 *   prepare   training views = registered keyframes with a stored photo, evenly spread, as the
 *             dense stage builds them (`loadDenseViewImage`: undistorted pinhole + rolling-shutter
 *             corrected, at the training size ≤ the dense size) → memory plan (`fitSplatPlan`:
 *             fewer views / Gaussians / smaller images until it fits) → initial Gaussians: the fused
 *             dense cloud reduced by the footprint LOD to `denseInitFraction` × the cap + the sparse
 *             SfM points (`initFromPointCloud`) + a far background shell coloured from view
 *             thumbnails (scene captures: sky) → pipelines compiled → views uploaded one at a time.
 *   training  `trainer.step(chunk)` under the engine's GPU lock, waiting while the page is hidden;
 *             cancel between chunks; a checkpoint (f32 parameters, bf16 Adam moments, colour state)
 *             every `checkpointEvery` iterations in the session store (resume after a kill or a
 *             cancel) or the scratch store; dropped work (`SplatWorkDroppedError`) rolls back to
 *             the last checkpoint in place; a dropped readback of a checkpoint or of the final
 *             model is read again once visible (stale data is never stored); a lost device (in a
 *             step, a checkpoint or the finishing readbacks) rebuilds the trainer on the new one
 *             from it (from the initial Gaussians before the first one; twice at most, then
 *             'gpu-lost'), trains again from there and finishes. Finish early (`CancelToken.finishEarly`,
 *             the app's "Use it now"): from `finishFrom` (the trainer's `fullResolutionFrom`, the end
 *             of the coarse-to-fine schedule; an earlier request waits for it) the loop stops between
 *             chunks, writes a checkpoint at that iteration and finishes like a complete run.
 *   finishing appearance normalisation (mean training-view colour matrix folded into the colours),
 *             read-back; the engine persists the model (`packSplatModel`) and clears the checkpoints
 *             — except after a finish early with a stored checkpoint at the model's iteration
 *             (`stats.splatContinueFrom`, its run key `splatContinueKey`): a later run of the same
 *             settings continues from it.
 *   previews  (`SplatOptions.preview`) 'splat_preview' events: `trainer.render` of the training view
 *             nearest the middle photo (the viewer's default view), box-downscaled by an integer to
 *             ≤ 400 px, as RGBA8 with its PSNR against that photo; once right after the first upload
 *             (that view goes first: the initial Gaussians, ~12 s before the first iteration on an M4)
 *             and then between chunks every ≥ 2 s. Forward only, no new pipeline, under the GPU lock
 *             (it waits while the page is hidden), one ≤ 1.9 MB stamped readback through the trainer's
 *             staging buffer: work iOS drops as the page hides (`SplatWorkDroppedError`) or any other
 *             failure only skips that one. At the end of the training the same view is rendered once
 *             more (not posted): `stats.splatPreviewFinalPsnr`, what the previews' PSNR converges to.
 *
 * Choices: every registered keyframe trains, not just the dense subset (40 at fast): novel views
 * need the viewpoints, and a 600×800 view is only 1.9 MB of GPU memory. The dense cloud is seeded
 * through the footprint LOD (keeps the MVS cloud's screen-space density, unlike a random subset);
 * the sparse points add the walls / far structure MVS misses. The shell radius
 * max(20 × camera extent, 3 × p98 sparse-point distance) is a heuristic, and an unresolved 'auto'
 * mode gets no shell. ≤ `chunk` (25) iterations per GPU-lock hold: cancel / page-hidden latency
 * ≈ 0.5 s on an M4. Measurements: docs/research/splat-tuning.md.
 *
 * Known limits: view preparation (JPEG decode, undistortion, rolling-shutter remap, ≈ 128 ms per
 * 600×800 view on an M4, ~19 % of the fast stage) and the depth-target packing (1–1.5 s for 40
 * views) run on the engine thread; only the views that are dense views get depth targets, and the
 * targets stay allocated after `until` (a rollback may need them). Not yet measured on an iPhone
 * (iteration time, IndexedDB checkpoint writes of 22–56 MB every 500 iterations, `max` headroom,
 * the preview readbacks next to the viewer). A finish early skips the end of the learning-rate
 * decay (~2 dB below the full schedule at 1 000 of 3 000 iterations in the GT-pose snapshots of
 * e2e/README.md "Splat results"; not measured through this path). Previews are raw trainer
 * colours from a training viewpoint (flattering), outside the memory plan (≈ 4 MB transient).
 */
import type { CameraIntrinsics, CancelToken, PointCloud, ProgressCallback, RGBAImage } from '../core/types';
import type { DepthMap } from '../dense/types';
import { footprintDownsample } from '../dense/pointcloud';
import { undistortedCamera } from '../dense/undistort';
import type { GPUContext } from '../gpu/device';
import { SplatTrainer, SplatWorkDroppedError, DEFAULT_SPLAT_GEOMETRY, geometryMode, type SplatTrainerOptions, type SplatTrainerState, type SplatTrainingView } from '../splat/trainer';
import { packDepthTarget } from '../splat/depth-target';
import { backgroundShell, concatModels, initFromPointCloud, type ColourSource } from '../splat/init';
import { boxDownscale, previewFactor, previewViewIndex, rgbPsnr } from '../splat/preview';
import { splatImageToRGBA8 } from '../splat/renderer';
import { downscaleSplatCamera, splatCamera, type SplatCamera, type SplatModel } from '../splat/types';
import { ENGINE_ERRORS, SPLAT_STAGES, namedError, type QualityPreset, type SplatOptions, type SplatPreviewDetail, type SplatTrainingDetail } from '../workers/protocol';
import { loadDenseViewImage, spreadSubset } from './dense';
import { MB, fitSplatPlan, formatBreakdown, planSplat, type SplatPlan } from './memory';
import {
  MemoryEngineStore,
  packSplatMoments,
  packSplatParams,
  unpackSplatMoments,
  unpackSplatParams,
  type EngineStore,
  type SplatProgressRecord,
} from './persistence';
import { resolveSplatPreset, type ResolvedMode, type SplatPresetConfig } from './presets';
import { cancelledError, modelPointCloud } from './refine';
import type { EngineConfig, SparseModel } from './types';

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

export const SPLAT_PROGRESS_KEY = 'splat/progress';
const ckptKey = (slot: number, part: 'params' | 'm' | 'v' | 'colour'): string => `splat/ckpt${slot}/${part}`;
/** Long side of the view thumbnails that colour the background shell. */
const THUMB_SIZE = 160;
/** Attempts of a trainer readback whose copy was dropped (SplatWorkDroppedError) before the stage fails. */
const MAX_READ_ATTEMPTS = 5;
/** Long side of the 'splat_preview' snapshots (px; never above the training size). */
const PREVIEW_SIZE = 400;
/**
 * Training wall time between two previews (ms): ≤ 0.5 Hz keeps their forward pass and readback
 * (≈ 5 ms on an M4) far below 1 % of the training and the readback stalls off the viewer's frames.
 */
const PREVIEW_EVERY_MS = 2000;

/** The previews of a run (`SplatOptions.preview`). */
interface PreviewState {
  /** Training view index and keyframe id. */
  index: number;
  id: number;
  /** Downscale factor of the training camera and the camera rendered. */
  factor: number;
  cam: SplatCamera;
  /** That view's photo at the preview size (set when it is uploaded). */
  target: RGBAImage | null;
  /** now() of the last preview, previews posted. */
  last: number;
  count: number;
}

export interface SplatDeps {
  /** Keyframe images. */
  store: EngineStore;
  /**
   * Checkpoints: the session store when it persists (resume after a kill), else the scratch
   * IndexedDB store, else null (kept in memory for the rollbacks of this run).
   */
  checkpoints: EngineStore | null;
  model: SparseModel;
  /** The fused dense cloud (null = none) and whether the engine keeps it resident anyway. Called once. */
  denseCloud(): Promise<PointCloud | null>;
  /**
   * The dense stage's fusion-filtered depth maps (undistorted pinhole views at the dense size,
   * keyframe ids), for the trainer's geometry supervision (`SplatTrainerOptions.geometry`); called
   * only when that is on; null / absent = none.
   */
  depthMaps?: (() => Promise<SplatDepthMaps | null>) | null;
  /** Points of the dense cloud (planning before it is loaded; 0 = none) and whether it is resident. */
  densePoints: number;
  denseResident: boolean;
  quality: QualityPreset;
  mode: ResolvedMode | 'auto';
  config: EngineConfig;
  /** Resolves while the page is visible (GPU work must not be submitted while hidden). */
  waitVisible(): Promise<void>;
  /** The engine's GPU lock (one heavy command stream at a time). */
  gpuLock<T>(fn: () => Promise<T>): Promise<T>;
  log(level: 'debug' | 'info' | 'warn' | 'error', message: string, data?: unknown): void;
  memoryBudgetMB: number;
  /** Engine data resident during the stage (bytes). */
  residentBytes: number;
  /** The shared WebGPU context (a fresh device after a loss); null without WebGPU. */
  gpu(): Promise<GPUContext | null>;
}

/** Depth maps of the dense stage (EngineDepthStore / MemoryDepthMapStore). */
export interface SplatDepthMaps {
  get(id: number): Promise<DepthMap | null>;
  /** Keyframe ids with a map. */
  ids(): number[];
}

export interface SplatOutput {
  /** Appearance-normalised trained model. */
  model: SplatModel;
  antialiased: boolean;
  background: [number, number, number];
  iterations: number;
  stats: Record<string, number | string>;
}

/** FNV-1a 32-bit of a string, hex. */
function fnv1a(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

/**
 * Identity of a splat run (what a checkpoint must match to be resumed): poses, cameras, sparse
 * points, rolling shutter, the dense cloud's identity, the preset / options and trainer overrides.
 * The fitted plan (views, size, Gaussians: budget dependent) is stored with the checkpoint instead.
 */
export function splatRunKey(model: SparseModel, dense: { count: number; first: number[]; last: number[] } | null, settings: Record<string, unknown>): string {
  const r6 = (v: number): number => Math.round(v * 1e6) / 1e6;
  return fnv1a(
    JSON.stringify({
      poses: model.images.map((im) => [im.id, im.cameraId, ...Array.from(im.pose.R, r6), ...Array.from(im.pose.t, r6)]),
      cams: [...model.cameras.entries()],
      points: model.numPoints,
      xyz: model.numPoints ? [r6(model.xyz[0]), r6(model.xyz[model.xyz.length - 1])] : [],
      rs: model.rollingShutter ? [model.rollingShutter.axis, model.rollingShutter.readoutMs, !!model.rollingShutter.nu] : null,
      dense,
      settings,
    }),
  );
}

/** Delete the stored checkpoint (both slots) and its progress record. */
export async function clearSplatCheckpoints(store: EngineStore): Promise<void> {
  await store.putValue(SPLAT_PROGRESS_KEY, null).catch(() => undefined);
  const keys = (await store.listBinaryKeys('splat/ckpt').catch(() => [] as string[])).filter((k) => /^splat\/ckpt\d\//.test(k));
  for (const k of keys) await store.deleteBinary(k).catch(() => undefined);
}

function isDeviceLost(e: unknown): boolean {
  return /device lost|device was lost|GPU device|instance destroyed|destroyed buffer/i.test(String((e as Error)?.message ?? e));
}

/** Concatenation of two clouds (positions / colours; normals and radii where both have them, else zero). */
function mergeClouds(a: PointCloud, b: PointCloud): PointCloud {
  const n = a.count + b.count;
  const positions = new Float32Array(3 * n);
  positions.set(a.positions.subarray(0, 3 * a.count));
  positions.set(b.positions.subarray(0, 3 * b.count), 3 * a.count);
  const colors = new Uint8Array(3 * n);
  colors.set(a.colors.subarray(0, 3 * a.count));
  colors.set(b.colors.subarray(0, 3 * b.count), 3 * a.count);
  const out: PointCloud = { count: n, positions, colors };
  if (a.normals || b.normals) {
    out.normals = new Float32Array(3 * n);
    if (a.normals) out.normals.set(a.normals.subarray(0, 3 * a.count));
    if (b.normals) out.normals.set(b.normals.subarray(0, 3 * b.count), 3 * a.count);
  }
  if (a.radii || b.radii) {
    out.radii = new Float32Array(n);
    if (a.radii) out.radii.set(a.radii.subarray(0, a.count));
    if (b.radii) out.radii.set(b.radii.subarray(0, b.count), a.count);
  }
  return out;
}

/** Centre of the cameras, their largest distance from it, and the 98th percentile distance of the sparse points. */
function sceneSpan(model: SparseModel): { centre: [number, number, number]; cameraExtent: number; pointsP98: number } {
  const cs = model.images.map((im) => {
    const R = im.pose.R;
    const t = im.pose.t;
    return [-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])];
  });
  const c = [0, 1, 2].map((k) => cs.reduce((s, x) => s + x[k], 0) / Math.max(1, cs.length)) as [number, number, number];
  const cameraExtent = Math.max(1e-6, ...cs.map((x) => Math.hypot(x[0] - c[0], x[1] - c[1], x[2] - c[2])));
  const d: number[] = [];
  const stride = Math.max(1, Math.floor(model.numPoints / 20_000));
  for (let i = 0; i < model.numPoints; i += stride) {
    const r = Math.hypot(model.xyz[3 * i] - c[0], model.xyz[3 * i + 1] - c[1], model.xyz[3 * i + 2] - c[2]);
    if (Number.isFinite(r)) d.push(r);
  }
  d.sort((a, b) => a - b);
  return { centre: c, cameraExtent, pointsP98: d.length ? d[Math.min(d.length - 1, Math.floor(0.98 * d.length))] : cameraExtent };
}

interface CheckpointData {
  state: SplatTrainerState;
  params: SplatModel;
  moments: { m: SplatModel; v: SplatModel } | null;
}

/** Train Gaussian splats on the sparse (+ dense) reconstruction. */
export async function runSplat(deps: SplatDeps, opts: SplatOptions = {}, progress?: ProgressCallback, cancel?: CancelToken): Promise<SplatOutput> {
  const t0 = now();
  const model = deps.model;
  const quality = opts.quality ?? deps.quality;
  const { trainer: trainerConfig = {}, ...presetOverrides } = deps.config.splat ?? {};
  const preset: SplatPresetConfig = resolveSplatPreset(
    quality,
    { ...presetOverrides, iterations: opts.iterations ?? presetOverrides.iterations, maxGaussians: opts.maxGaussians ?? presetOverrides.maxGaussians, maxImageSize: opts.maxImageSize ?? presetOverrides.maxImageSize, maxViews: opts.maxViews ?? presetOverrides.maxViews },
    deps.config.dense ?? {},
  );
  // the preset's trainer defaults, EngineConfig.splat.trainer field by field on top
  const trainerOverrides: Partial<SplatTrainerOptions> = { ...preset.trainer, ...trainerConfig };
  const iterations = Math.max(1, Math.round(preset.iterations));
  const shDegree = trainerOverrides.shDegree ?? preset.shDegree;
  const stats: Record<string, number | string> = {};
  const report = (stage: string, p: number, message?: string, detail?: Record<string, number>, image?: RGBAImage): void => {
    const e = { stage, progress: Math.max(0, Math.min(1, p)), ...(message !== undefined ? { message } : {}), ...(detail ? { detail } : {}), ...(image ? { image } : {}) };
    progress?.(e);
  };
  const check = (stage: string): void => {
    if (cancel?.cancelled) throw cancelledError(stage);
  };
  if (model.images.length < 2) throw new Error('splat needs a sparse model with at least 2 registered images (run refine first)');

  // ---- prepare: views, plan
  report(SPLAT_STAGES.prepare, 0, 'planning');
  let ctx = await deps.gpu();
  if (!ctx) throw namedError('NotSupportedError', 'Gaussian splatting needs WebGPU', 'no-webgpu');
  const storedIds = new Set((await deps.store.getFrameRecords()).map((r) => r.id));
  const withImage = model.images.filter((im) => storedIds.has(im.id));
  if (withImage.length < 2) throw new Error(`splat needs at least 2 registered images with stored photos (${withImage.length} of ${model.images.length} available)`);
  if (withImage.length < model.images.length) stats.splatMissingImages = model.images.length - withImage.length;
  const camOf = (id: number): CameraIntrinsics => model.cameras.get(model.images.find((im) => im.id === id)!.cameraId)!;
  const dims = (size: number, ids: readonly number[] = withImage.map((im) => im.id)): { width: number; height: number } => {
    let best = { width: 1, height: 1 };
    for (const cid of new Set(ids.map((id) => model.images.find((im) => im.id === id)!.cameraId))) {
      const c = undistortedCamera(model.cameras.get(cid)!, size);
      if (c.width * c.height > best.width * best.height) best = { width: c.width, height: c.height };
    }
    return best;
  };
  const shellOn = (opts.backgroundShell ?? deps.mode === 'scene') && preset.shellCount > 0;
  const span = sceneSpan(model);
  const shellRadius = Math.max(20 * span.cameraExtent, 3 * span.pointsP98);
  const denseTarget = deps.densePoints > 0 ? Math.min(deps.densePoints, Math.round(preset.denseInitFraction * preset.maxGaussians)) : 0;
  // geometry supervision: the dense stage's depth maps of the registered keyframes with photos
  const geoOpt = trainerOverrides.geometry ? { ...DEFAULT_SPLAT_GEOMETRY, ...trainerOverrides.geometry } : null;
  const depthStore = geoOpt && deps.depthMaps ? await deps.depthMaps() : null;
  const depthIds = new Set(depthStore ? depthStore.ids().filter((id) => withImage.some((im) => im.id === id)) : []);
  const planBase = {
    budgetMB: deps.memoryBudgetMB,
    residentBytes: deps.residentBytes,
    shDegree,
    loadedCloudPoints: deps.denseResident ? 0 : deps.densePoints,
    initPoints: denseTarget + model.numPoints + (shellOn ? preset.shellCount : 0),
    geometry: geoOpt && depthIds.size ? { views: depthIds.size, factor: geoOpt.factor, normals: geometryMode(geoOpt) === true } : undefined,
  };
  const fit = fitSplatPlan(planBase, { views: Math.min(preset.maxViews, withImage.length), size: preset.maxImageSize, maxGaussians: preset.maxGaussians }, { dims: (s) => dims(s), maxBufferBytes: ctx.limits.maxStorageBufferBindingSize });
  // the dense cloud (read from the store unless resident) seeds the Gaussians and identifies the run
  let cloud: PointCloud | null = deps.densePoints ? await deps.denseCloud() : null;
  if (cloud && !cloud.count) cloud = null;
  const r6 = (v: number): number => Math.round(v * 1e6) / 1e6;
  const denseId = cloud ? { count: cloud.count, first: Array.from(cloud.positions.subarray(0, 3), r6), last: Array.from(cloud.positions.subarray(3 * cloud.count - 3, 3 * cloud.count), r6) } : null;
  const runKey = splatRunKey(model, denseId, {
    quality,
    iterations,
    shDegree,
    maxViews: preset.maxViews,
    maxImageSize: preset.maxImageSize,
    maxGaussians: preset.maxGaussians,
    denseInitFraction: preset.denseInitFraction,
    shell: shellOn ? [preset.shellCount, Math.round(shellRadius * 1e3) / 1e3] : null,
    trainer: trainerOverrides,
  });

  // ---- resume from a checkpoint of the same run (plan stored with it)
  const ckStore: EngineStore = deps.checkpoints ?? new MemoryEngineStore('splat-checkpoints');
  let resume: (CheckpointData & { viewIds: number[]; size: number; maxGaussians: number; slot: 0 | 1 }) | null = null;
  const prog = opts.restart ? null : await ckStore.getValue<SplatProgressRecord>(SPLAT_PROGRESS_KEY).catch(() => null);
  if (prog && prog.key === runKey && prog.plan && prog.iteration < iterations) {
    try {
      const ids = prog.plan.viewIds;
      if (!ids.every((id) => storedIds.has(id) && model.images.some((im) => im.id === id))) throw new Error('views changed');
      const d = dims(prog.plan.size, ids);
      const p = planSplat({ ...planBase, views: ids.length, width: d.width, height: d.height, maxGaussians: prog.plan.maxGaussians });
      if (!p.fits && fit.plan.fits) throw new Error(`its plan (${ids.length} views, ${prog.plan.size} px, ${prog.plan.maxGaussians} Gaussians) no longer fits the memory budget`);
      const data = await readCheckpoint(ckStore, prog);
      resume = { ...data, viewIds: ids, size: prog.plan.size, maxGaussians: prog.plan.maxGaussians, slot: prog.slot };
    } catch (e) {
      deps.log('warn', `splat: checkpoint at iteration ${prog.iteration} not resumed (${(e as Error)?.message ?? e}); starting over`);
    }
  }
  if (!resume) await clearSplatCheckpoints(ckStore);

  const views = resume ? resume.viewIds.map((id) => withImage.find((im) => im.id === id)!) : spreadSubset(withImage, fit.views);
  const size = resume ? resume.size : fit.size;
  const maxGaussians = resume ? resume.maxGaussians : fit.maxGaussians;
  const vd = dims(size, views.map((v) => v.id));
  const plan: SplatPlan = resume ? planSplat({ ...planBase, views: views.length, width: vd.width, height: vd.height, maxGaussians }) : fit.plan;
  if (!resume && fit.degraded.length) {
    stats.splatDegraded = fit.degraded.join(', ');
    deps.log('warn', `splat: lowered to fit the ${deps.memoryBudgetMB} MB budget: ${fit.degraded.join(', ')}`);
  }
  stats.splatViews = views.length;
  stats.splatSize = size;
  stats.splatWidth = vd.width;
  stats.splatHeight = vd.height;
  stats.splatMaxGaussians = maxGaussians;
  stats.splatIterations = iterations;
  stats.splatMemPeakMB = Math.round(plan.peakBytes / MB);
  stats.splatGpuMB = Math.round(plan.gpuBytes / MB);
  // previews: the training view nearest the viewer's default photo, at ≤ PREVIEW_SIZE
  let preview: PreviewState | null = null;
  if (opts.preview && progress) {
    const index = previewViewIndex(
      views.map((v) => v.id),
      model.images.map((im) => im.id),
    );
    const im = views[index];
    const full = splatCamera(undistortedCamera(camOf(im.id), size), im.pose);
    const factor = previewFactor(full.width, full.height, PREVIEW_SIZE);
    preview = { index, id: im.id, factor, cam: downscaleSplatCamera(full, factor), target: null, last: -Infinity, count: 0 };
  }
  deps.log(
    plan.fits ? 'info' : 'warn',
    `splat memory plan (${views.length} views ${vd.width}×${vd.height}, ≤ ${maxGaussians} Gaussians, SH ${shDegree}, budget ${deps.memoryBudgetMB} MB, peak ≈ ${Math.round(plan.peakBytes / MB)} MB, GPU ≈ ${Math.round(plan.gpuBytes / MB)} MB): ` +
      plan.stages.map((st) => `${st.stage} ${formatBreakdown(st.parts, st.bytes)}`).join('; '),
  );
  check(SPLAT_STAGES.prepare);

  // ---- initial Gaussians (or the checkpoint's)
  let init: SplatModel;
  const tInit = now();
  if (resume) {
    init = resume.params;
    stats.splatResumedFrom = resume.state.iteration;
    deps.log('info', `splat: resuming at iteration ${resume.state.iteration} of ${iterations} (${resume.state.count} Gaussians)`);
    cloud = null;
  } else {
    report(SPLAT_STAGES.prepare, 0.05, 'initial Gaussians');
    await tick();
    const shellCount = shellOn ? preset.shellCount : 0;
    let pc: PointCloud = modelPointCloud(model);
    let fromDense = 0;
    if (cloud && cloud.count) {
      const lod = footprintDownsample(cloud, Math.min(cloud.count, Math.round(preset.denseInitFraction * maxGaussians)));
      fromDense = lod.count;
      pc = mergeClouds(lod, pc);
    }
    cloud = null; // the engine keeps a resident cloud; one read from the store is released here
    check(SPLAT_STAGES.prepare);
    const seed = trainerOverrides.seed ?? 1;
    init = initFromPointCloud(pc, { maxCount: Math.max(1, Math.min(pc.count, maxGaussians - shellCount)), shDegree, seed });
    stats.splatInitDense = fromDense;
    stats.splatInitSparse = model.numPoints;
    if (shellCount) {
      // sky: shell coloured from small thumbnails of the training views
      const sources: ColourSource[] = [];
      for (const im of views) {
        const t = await loadDenseViewImage(deps.store, model, im.id, camOf(im.id), THUMB_SIZE);
        sources.push({ cam: splatCamera(t.cam, im.pose), data: t.image.data });
      }
      const shell = backgroundShell(sources, { count: shellCount, centre: span.centre, radius: shellRadius, shDegree });
      init = concatModels(init, shell);
      stats.splatShell = shell.count;
      stats.splatShellRadius = Math.round(shellRadius * 1000) / 1000;
    }
    stats.splatInitGaussians = init.count;
    deps.log('info', `splat: ${init.count} initial Gaussians (dense ${fromDense}, sparse ${model.numPoints}${shellCount ? `, shell ${stats.splatShell} at r = ${shellRadius.toFixed(2)}` : ''}) in ${Math.round(now() - tInit)} ms`);
  }

  // ---- trainer: pipelines, model, views
  const trainerOpts: SplatTrainerOptions = { ...trainerOverrides, maxGaussians, iterations, shDegree };
  if (trainerOpts.geometry && !depthIds.size) {
    deps.log('info', 'splat: geometry supervision requested but there are no dense depth maps: off');
    trainerOpts.geometry = false;
  }
  /** Geometry targets of the views with a dense depth map (resampled to the training size). */
  const uploadDepth = async (t: SplatTrainer): Promise<void> => {
    if (!trainerOpts.geometry || !depthStore) return;
    const td = now();
    let n = 0;
    for (let i = 0; i < views.length; i++) {
      check(SPLAT_STAGES.prepare);
      const id = views[i].id;
      if (!depthIds.has(id)) continue;
      const map = await depthStore.get(id);
      if (!map) continue;
      const mapCam = undistortedCamera(camOf(id), Math.max(map.width, map.height));
      if (mapCam.width !== map.width || mapCam.height !== map.height) {
        deps.log('warn', `splat: depth map ${id} is ${map.width}×${map.height}, its dense camera ${mapCam.width}×${mapCam.height}: skipped`);
        continue;
      }
      t.setViewDepth(i, packDepthTarget(map, mapCam, undistortedCamera(camOf(id), size), { factor: geoOpt!.factor, grazing: geoOpt!.grazing }));
      n++;
      if (n % 4 === 0) await tick();
    }
    stats.splatDepthViews = n;
    stats.splatDepthMs = Math.round(now() - td);
    deps.log('info', `splat: geometry targets for ${n} of ${views.length} views (1/${geoOpt!.factor} size) in ${Math.round(now() - td)} ms`);
  };
  const viewMeta = (): SplatTrainingView[] => views.map((im) => ({ id: im.id, cam: undistortedCamera(camOf(im.id), size), pose: im.pose, image: null }));
  /**
   * Trainer `t` rendered from the preview view (a forward pass and a ≤ 1.9 MB stamped readback
   * under the GPU lock, which waits while the page is hidden) as RGBA8, with its PSNR against that
   * view's photo. Throws like `render` (SplatWorkDroppedError: the work was dropped as the page hid).
   */
  const snapshot = async (t: SplatTrainer, p: PreviewState, target: RGBAImage): Promise<{ image: RGBAImage; psnr: number }> => {
    const out = await deps.gpuLock(() => t.render(p.cam));
    const image: RGBAImage = { width: out.width, height: out.height, data: splatImageToRGBA8(out) };
    return { image, psnr: Math.round(100 * rgbPsnr(image, target)) / 100 };
  };
  /**
   * One 'splat_preview' snapshot of trainer `t`. A failure (dropped work, or a device loss, which
   * the next step then recovers from) only skips it: never a stale image.
   */
  const renderPreview = async (t: SplatTrainer): Promise<void> => {
    const p = preview;
    if (!p?.target) return;
    try {
      const { image, psnr } = await snapshot(t, p, p.target);
      const detail: SplatPreviewDetail = { iteration: t.iteration, iterations, viewId: p.id, psnr };
      report(SPLAT_STAGES.preview, t.iteration / iterations, `iteration ${t.iteration}`, detail, image);
      p.count++;
    } catch (e) {
      deps.log('warn', `splat: preview at iteration ${t.iteration} skipped (${(e as Error)?.message ?? e})`);
    }
    p.last = now();
  };
  const uploadViews = async (t: SplatTrainer, from: number, to: number): Promise<void> => {
    // the preview's view first: the first snapshot (the initial Gaussians) follows its upload
    const order = preview ? [preview.index, ...views.keys()].filter((i, k) => k === 0 || i !== preview!.index) : [...views.keys()];
    for (let k = 0; k < order.length; k++) {
      check(SPLAT_STAGES.prepare);
      const i = order[k];
      const im = views[i];
      const v = await loadDenseViewImage(deps.store, model, im.id, camOf(im.id), size);
      await deps.waitVisible();
      t.setViewImage(i, v.image);
      if (preview && i === preview.index) {
        preview.target ??= boxDownscale(v.image, preview.factor);
        if (!preview.count) await renderPreview(t);
      }
      report(SPLAT_STAGES.prepare, from + ((to - from) * (k + 1)) / views.length, `uploading views ${k + 1}/${views.length}`);
      if (k % 4 === 3) await tick();
    }
  };
  /** A trainer on `params` (+ a checkpoint's state, restored before the uploads: the first preview's iteration). */
  const createTrainer = async (params: SplatModel, from: number, to: number, ck?: Pick<CheckpointData, 'state' | 'moments'>): Promise<SplatTrainer> => {
    report(SPLAT_STAGES.prepare, from, 'compiling GPU pipelines');
    const c = ctx!;
    const t = await deps.gpuLock(() => SplatTrainer.create(c, trainerOpts));
    try {
      t.init(params, viewMeta());
      if (ck) t.restore(ck.state, ck.moments);
      await uploadViews(t, from + 0.1 * (to - from), to);
      await uploadDepth(t);
      deps.config.splatDebug?.onTrainer?.(t);
    } catch (e) {
      t.destroy();
      throw e;
    }
    return t;
  };
  const tPrep = now();
  let trainer = await createTrainer(init, 0.2, 1, resume ?? undefined);
  try {
    stats.splatPrepareMs = Math.round(now() - t0);
    deps.log('info', `splat: prepared in ${((now() - t0) / 1000).toFixed(1)} s (trainer + ${views.length} view uploads ${((now() - tPrep) / 1000).toFixed(1)} s)`);

    // ---- checkpoints (slot alternates; the progress record names the complete one)
    let slot: 0 | 1 = resume ? resume.slot : 1;
    let lastCkpt: { iteration: number; slot: 0 | 1 } | null = resume ? { iteration: resume.state.iteration, slot: resume.slot } : null;
    let ckptMs = 0;
    let ckpts = 0;
    let readRetries = 0;
    /**
     * A trainer readback under the GPU lock, read again (once the page is visible) when its copy
     * was dropped: stale staging data must never become a checkpoint or the saved model.
     */
    const readBack = async <T>(what: string, fn: (t: SplatTrainer) => Promise<T>): Promise<T> => {
      const t = trainer;
      for (let attempt = 1; ; attempt++) {
        try {
          return await deps.gpuLock(() => fn(t));
        } catch (e) {
          if (!(e instanceof SplatWorkDroppedError) || attempt >= MAX_READ_ATTEMPTS) throw e;
          readRetries++;
          deps.log('warn', `splat: ${what}: ${e.message} — reading again`);
          await deps.waitVisible();
        }
      }
    };
    const writeCheckpoint = async (): Promise<void> => {
      const tc = now();
      const next: 0 | 1 = slot === 0 ? 1 : 0;
      const state = await readBack('checkpoint', (t) => t.readState());
      await ckStore.putBinary(ckptKey(next, 'params'), packSplatParams(await readBack('checkpoint', (t) => t.readModel())));
      const moments = state.iteration > 0;
      if (moments) {
        await ckStore.putBinary(ckptKey(next, 'm'), packSplatMoments(await readBack('checkpoint', (t) => t.readMoments('m'))));
        await ckStore.putBinary(ckptKey(next, 'v'), packSplatMoments(await readBack('checkpoint', (t) => t.readMoments('v'))));
      }
      await ckStore.putBinary(ckptKey(next, 'colour'), state.colour);
      const rec: SplatProgressRecord = {
        key: runKey,
        quality,
        iteration: state.iteration,
        count: state.count,
        slot: next,
        moments,
        background: state.background,
        mcmc: state.mcmc,
        maxIntersections: state.maxIntersections,
        updatedAt: Date.now(),
        plan: { viewIds: views.map((v) => v.id), size, maxGaussians },
      };
      await ckStore.putValue(SPLAT_PROGRESS_KEY, rec);
      for (const part of ['params', 'm', 'v', 'colour'] as const) await ckStore.deleteBinary(ckptKey(slot, part)).catch(() => undefined);
      slot = next;
      lastCkpt = { iteration: state.iteration, slot: next };
      ckpts++;
      ckptMs += now() - tc;
    };
    /** The last checkpoint (its iteration, parameters, moments). */
    const loadLast = async (): Promise<CheckpointData> => {
      const rec = await ckStore.getValue<SplatProgressRecord>(SPLAT_PROGRESS_KEY);
      if (!rec || rec.key !== runKey) throw new Error('no checkpoint to roll back to');
      return readCheckpoint(ckStore, rec);
    };
    let losses = 0;
    const lostError = (e: unknown): Error => namedError(ENGINE_ERRORS.gpuLost, `GPU device lost during splat training: ${(e as Error)?.message ?? e}`, 'gpu-lost');
    /**
     * After a failed GPU operation (a step, a checkpoint, the finishing readbacks): a device loss
     * rebuilds the trainer on the new device from the last checkpoint (`from`: from these initial
     * Gaussians, before the first checkpoint), twice at most, then 'gpu-lost'. Anything else is
     * rethrown.
     */
    const recoverLoss = async (e: unknown, from?: SplatModel): Promise<void> => {
      if (!isDeviceLost(e)) throw e;
      if (losses >= 2) throw lostError(e);
      losses++;
      deps.log('warn', `splat: GPU device lost (${(e as Error)?.message ?? e}) — rebuilding the trainer from ${from ? 'the initial Gaussians' : `the checkpoint at iteration ${lastCkpt?.iteration ?? 0}`}`);
      try {
        trainer.destroy();
      } catch {
        /* device gone */
      }
      ctx = await deps.gpu();
      if (!ctx) throw lostError(e);
      if (from) {
        trainer = await createTrainer(from, 0, 1);
        return;
      }
      const ck = await loadLast();
      trainer = await createTrainer(ck.params, 0, 1, ck);
    };
    // iteration 0: the initial Gaussians (rollback target)
    while (!resume) {
      try {
        await writeCheckpoint();
        break;
      } catch (e) {
        await recoverLoss(e, init);
      }
    }

    // ---- training, then finishing (a device loss there trains again from the last checkpoint)
    const tTrain = now();
    let trainItersThisRun = 0;
    let trainMsThisRun = 0;
    let drops = 0;
    let last = { loss: NaN, count: trainer.gaussians, itPerSec: 0 };
    const every = Math.max(1, Math.round(preset.checkpointEvery));
    // finish early (CancelToken.finishEarly) from the end of the coarse-to-fine schedule on
    const finishFrom = Math.min(iterations, trainer.fullResolutionFrom);
    let finishedEarly = false;
    let mean: Float64Array | null = null;
    let out: SplatModel | null = null;
    let tf = 0;
    while (!out) {
      while (trainer.iteration < iterations) {
        if (cancel?.cancelled) {
          // keep the progress since the last checkpoint for the resume (cheap next to the training)
          if (lastCkpt && trainer.iteration - lastCkpt.iteration >= 100) await writeCheckpoint().catch((e: unknown) => deps.log('warn', `splat: checkpoint on cancel failed: ${(e as Error)?.message ?? e}`));
          throw cancelledError(SPLAT_STAGES.training);
        }
        if (cancel?.finishEarly && trainer.iteration >= finishFrom) {
          // the checkpoint a later run continues from, before the finishing changes the trainer
          if (lastCkpt?.iteration !== trainer.iteration) {
            try {
              await writeCheckpoint();
            } catch (e) {
              if (isDeviceLost(e)) {
                await recoverLoss(e);
                continue;
              }
              deps.log('warn', `splat: checkpoint at iteration ${trainer.iteration} failed (${(e as Error)?.message ?? e}); finishing without one to continue from`);
            }
          }
          finishedEarly = true;
          break;
        }
        await deps.waitVisible();
        const it = trainer.iteration;
        const nextCkpt = (Math.floor(it / every) + 1) * every;
        const n = Math.max(1, Math.min(preset.chunk, iterations - it, nextCkpt - it));
        const t = trainer;
        try {
          const r = await deps.gpuLock(() => t.step(n));
          trainItersThisRun += n;
          trainMsThisRun += r.ms;
          last = { loss: r.loss, count: r.count, itPerSec: r.iterationsPerSecond };
        } catch (e) {
          if (e instanceof SplatWorkDroppedError && drops < 8) {
            drops++;
            deps.log('warn', `splat: ${e.message} — rolling back to the checkpoint at iteration ${lastCkpt?.iteration ?? 0}`);
            const ck = await loadLast();
            await deps.waitVisible();
            trainer.restore(ck.state, ck.moments, ck.params);
            continue;
          }
          await recoverLoss(e);
          continue;
        }
        const cur = trainer.iteration;
        const detail: SplatTrainingDetail = { iteration: cur, iterations, loss: last.loss, itPerSec: last.itPerSec, gaussians: last.count, finishFrom };
        report(SPLAT_STAGES.training, cur / iterations, `iteration ${cur}/${iterations} · ${last.count} Gaussians · ${last.itPerSec.toFixed(1)} it/s`, detail);
        if (cur % every === 0 && cur < iterations) await writeCheckpoint().catch((e: unknown) => recoverLoss(e));
        // (none right before the loop stops: the model itself comes next)
        const stopping = !!cancel?.cancelled || (!!cancel?.finishEarly && cur >= finishFrom);
        if (preview && cur < iterations && !stopping && now() - preview.last >= PREVIEW_EVERY_MS) await renderPreview(trainer);
      }
      stats.splatTrainMs = Math.round(now() - tTrain);
      stats.splatItPerSec = trainMsThisRun > 0 ? Math.round((10_000 * trainItersThisRun) / trainMsThisRun) / 10 : 0;
      stats.splatLoss = Math.round(last.loss * 1e5) / 1e5;
      stats.splatCheckpoints = ckpts;
      stats.splatCheckpointMs = Math.round(ckptMs);
      if (drops) stats.splatDroppedWork = drops;
      if (losses) stats.splatDeviceLosses = losses;
      if (preview) stats.splatPreviews = preview.count;
      // the trained model from the preview view (same raw colours, not posted): the previews' end point
      if (preview?.target) {
        try {
          stats.splatPreviewFinalPsnr = (await snapshot(trainer, preview, preview.target)).psnr;
        } catch (e) {
          delete stats.splatPreviewFinalPsnr;
          deps.log('warn', `splat: final preview PSNR skipped (${(e as Error)?.message ?? e})`);
        }
      }
      deps.log(
        'info',
        `splat: trained ${trainItersThisRun} iterations in ${(stats.splatTrainMs / 1000).toFixed(1)} s (${stats.splatItPerSec} it/s in step, checkpoints ${ckpts} × ≈ ${ckpts ? Math.round(ckptMs / ckpts) : 0} ms), ${last.count} Gaussians, loss ${last.loss.toFixed(4)}` +
          `${finishedEarly ? `; finished early at iteration ${trainer.iteration} of ${iterations}` : ''}` +
          `${preview ? `; ${preview.count} previews${stats.splatPreviewFinalPsnr !== undefined ? ` (final model ${stats.splatPreviewFinalPsnr} dB at that view)` : ''}` : ''}`,
      );

      // ---- finishing (a cancel arriving now is ignored: the training is complete)
      report(SPLAT_STAGES.finishing, 0, 'normalising appearance');
      tf = now();
      try {
        // (a dropped readback of normaliseAppearance throws before it changes anything: read again)
        mean = await readBack('appearance normalisation', (t) => t.normaliseAppearance());
        report(SPLAT_STAGES.finishing, 0.4, 'reading the model');
        out = await readBack('model read-back', (t) => t.readModel());
      } catch (e) {
        await recoverLoss(e);
      }
    }
    if (mean) stats.splatMeanColourGain = Math.round(((mean[0] + mean[5] + mean[10]) / 3) * 1000) / 1000;
    if (readRetries) stats.splatReadRetries = readRetries;
    if (finishedEarly) {
      stats.splatFinishedEarly = 1;
      // a stored checkpoint at the model's iteration: the engine keeps it for a later run to continue
      // from (its run key: loadSession tells it from the checkpoint of another, interrupted run)
      if (deps.checkpoints && lastCkpt?.iteration === trainer.iteration) Object.assign(stats, { splatContinueFrom: trainer.iteration, splatContinueKey: runKey });
    }
    stats.splatCount = out.count;
    stats.splatFinishMs = Math.round(now() - tf);
    stats.splatMs = Math.round(now() - t0);
    return { model: out, antialiased: trainer.antialiased, background: trainer.backgroundColour, iterations: trainer.iteration, stats };
  } finally {
    try {
      trainer.destroy();
    } catch {
      /* device gone */
    }
  }
}

/** Read a checkpoint's binaries (throws when one is missing or damaged). */
async function readCheckpoint(store: EngineStore, rec: SplatProgressRecord): Promise<CheckpointData> {
  const get = async (part: 'params' | 'm' | 'v' | 'colour'): Promise<ArrayBuffer> => {
    const b = await store.getBinary(ckptKey(rec.slot, part));
    if (!b) throw new Error(`checkpoint ${part} missing`);
    return b;
  };
  const params = unpackSplatParams(await get('params'));
  const moments = rec.moments ? { m: unpackSplatMoments(await get('m')), v: unpackSplatMoments(await get('v')) } : null;
  const colour = new Float32Array(await get('colour'));
  if (params.count !== rec.count || (moments && (moments.m.count !== rec.count || moments.v.count !== rec.count))) throw new Error('checkpoint sizes disagree');
  return {
    params,
    moments,
    state: { iteration: rec.iteration, count: rec.count, background: rec.background, colour, mcmc: rec.mcmc, maxIntersections: rec.maxIntersections },
  };
}
