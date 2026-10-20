/**
 * Dense reconstruction and meshing (ENGINE.md "Dense / mesh", ARCHITECTURE §5 "Dense"):
 * view subset by preset → undistortion per view (keyframe JPEGs decoded on demand through an
 * LRU image cache sized to the memory budget) → source-view selection from the sparse model →
 * PatchMatch (photometric, then geometric-consistency passes; WebGPU with a CPU fallback at low
 * resolution) into DepthMapStores (memory, or IndexedDB + LRU when persisting) → fusion →
 * point-cloud filtering. Meshing: TSDF over the fusion-filtered depth maps (+ simplification).
 *
 * GPU scheduling (one command buffer in flight, retries of dropped work) is done inside
 * PatchMatchGPU; between views the engine waits while the page is hidden.
 */
import type { CameraIntrinsics, CancelToken, PointCloud, Pose, ProgressCallback, RGBAImage, TriangleMesh } from '../core/types';
import { removeDepthHalos, worldUpInCamera } from '../dense/depth-filter';
import { fuseDepthMaps } from '../dense/fusion';
import { patchMatchCPU } from '../dense/patchmatch-cpu';
import { PatchMatchGPU } from '../dense/patchmatch-gpu';
import { filterPointCloud, footprintDownsample } from '../dense/pointcloud';
import { simplifyMesh } from '../dense/simplify';
import { tsdfMesh } from '../dense/tsdf';
import type { DenseView, DepthMap, DepthMapStore, PatchMatchOptions } from '../dense/types';
import { undistortImage, undistortedCamera } from '../dense/undistort';
import { rsCorrectPinholeImage } from '../sfm/rolling-shutter';
import { neighboursFromSelection, selectSourceViews, type SourceViewSelection } from '../dense/view-selection';
import { getGPU } from '../gpu/device';
import { median } from '../math/stats';
import type { DenseOptions, MeshOptions, QualityPreset } from '../workers/protocol';
import { EngineDepthStore } from './depth-store';
import { fitSize, resizeRGBA } from './frame-ingest';
import { LruCache, decodeStoredImage, rgbaBytes } from './image-cache';
import { MB, formatBreakdown, planDense, type DensePlan } from './memory';
import type { DenseMetaRecord, EngineStore } from './persistence';
import { modeConfig, resolveDensePreset, type DensePresetConfig, type ResolvedMode } from './presets';
import { cancelledError } from './refine';
import type { EngineConfig, SparseModel } from './types';

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

/** Lazily created, self-tested PatchMatchGPU (null ⇒ CPU fallback). */
export class PatchMatchHolder {
  private pm: PatchMatchGPU | null = null;
  private tried = false;
  reason: string | null = null;

  /** `denseOverrides`: EngineConfig.dense, so the precompiled source buckets cover an overridden numSources. */
  constructor(
    private readonly preferGPU: boolean,
    private readonly denseOverrides: Partial<DensePresetConfig> = {},
    private readonly gpuTimestamps = false,
  ) {}

  async get(): Promise<PatchMatchGPU | null> {
    if (this.pm) return this.pm;
    if (this.tried || !this.preferGPU) {
      if (!this.preferGPU) this.reason ??= 'CPU requested (preferGPU: false)';
      return null;
    }
    this.tried = true;
    try {
      const ctx = await getGPU();
      if (!ctx) {
        this.reason = 'WebGPU unavailable';
        return null;
      }
      // Precompile only the source buckets and kernel shapes (window, affine warp) densify can
      // use (any quality, with the config's overrides): a compile failure then still falls back
      // to the CPU here, not mid-densify. The self-test runs the first preset's kernel shape.
      const presets = (['fast', 'balanced', 'max'] as const).map((q) => resolveDensePreset(q, this.denseOverrides));
      const maxSources = Math.max(...presets.map((p) => Math.min(16, p.numSources)));
      const shapes = new Map<string, Partial<PatchMatchOptions>>();
      for (const p of presets) {
        const k = (p.patchMatch ?? {}) as Partial<PatchMatchOptions>;
        const shape = { patchRadius: k.patchRadius, patchStep: k.patchStep, affineWindow: k.affineWindow };
        shapes.set(JSON.stringify(shape), shape);
      }
      const [first, ...rest] = [...shapes.values()];
      const pm = await PatchMatchGPU.create(ctx, first, maxSources, { timestamps: this.gpuTimestamps });
      for (const shape of rest) await pm.precompile(shape, maxSources);
      const st = await pm.selfTest();
      if (!st.ok) {
        pm.destroy();
        this.reason = `PatchMatch GPU self-test failed (agreement ${st.agreement.toFixed(2)}${st.error ? `, ${st.error}` : ''})`;
        return null;
      }
      this.pm = pm;
      return pm;
    } catch (e) {
      this.reason = `PatchMatchGPU unavailable: ${(e as Error)?.message ?? e}`;
      return null;
    }
  }

  /** Forget the instance (device lost): the next `get()` creates a new one. */
  reset(): void {
    try {
      this.pm?.destroy();
    } catch {
      /* device gone */
    }
    this.pm = null;
    this.tried = false;
  }

  destroy(): void {
    this.reset();
    this.tried = true;
  }
}

export interface DenseDeps {
  /** Keyframe images (and persisted results). */
  store: EngineStore;
  /**
   * Backend of the depth maps: the session store when it persists, else a scratch IndexedDB
   * store (bounded memory), or null = keep them in memory (Node tests).
   */
  depthBackend: EngineStore | null;
  model: SparseModel;
  quality: QualityPreset;
  mode: ResolvedMode | 'auto';
  config: EngineConfig;
  holder: PatchMatchHolder;
  /** Resolves while the page is visible (GPU work must not be submitted while hidden). */
  waitVisible(): Promise<void>;
  /** The engine's GPU lock (one heavy command stream at a time). */
  gpuLock<T>(fn: () => Promise<T>): Promise<T>;
  log(level: 'debug' | 'info' | 'warn' | 'error', message: string, data?: unknown): void;
  memoryBudgetMB: number;
  /** Engine data resident during the dense stage (bytes; capture state if kept, sparse model). */
  residentBytes?: number;
  /**
   * Up direction of each view for the halo filter's sky test (`denseSkyUp`): the model is
   * gravity-aligned (`MetricInfo.gravityAligned`: world +Z is up), else the keyframes' gravity
   * (`FrameInfo.gravityCam`, DOWN in the canonical camera frame, by keyframe id). Neither ⇒ the
   * sky test assumes upright images (top row), which is wrong for frames that are not upright.
   */
  gravityAligned?: boolean;
  gravityCam?: ReadonlyMap<number, ArrayLike<number> | null>;
}

/**
 * Up direction (opposite gravity) in a view's camera frame for the halo filter's sky test
 * (`HaloFilterOptions.skyUp`): world +Z through the pose when the model is gravity-aligned, else
 * −gravityCam when known, else null (the filter then takes the image as upright).
 */
export function denseSkyUp(pose: Pose, gravityAligned: boolean, gravityCam?: ArrayLike<number> | null): [number, number, number] | null {
  if (gravityAligned) {
    const up = worldUpInCamera(pose.R);
    if (up.every(Number.isFinite)) return up;
  }
  if (gravityCam && gravityCam.length >= 3) {
    const up: [number, number, number] = [-gravityCam[0], -gravityCam[1], -gravityCam[2]];
    const n = Math.hypot(up[0], up[1], up[2]);
    if (n > 0 && n < Infinity) return [up[0] / n, up[1] / n, up[2] / n];
  }
  return null;
}

/** PatchMatch pass: ½-resolution photometric (multi-scale), photometric, geometric. */
type PassKind = 'half' | 'photo' | 'geo';

/** Persisted dense progress (`dense/progress` in the depth backend): resume after a kill / cancel. */
interface DenseProgress {
  /** Hash of the model + view selection inputs + settings the maps were computed with. */
  key: string;
  /** Pass in progress (index into the pass kinds: [½ photometric,] photometric, geometric…), or = number of passes when fusing. */
  pass: number;
  /**
   * Quality preset of the run (the key covers its settings): a resume must use the same one — the
   * engine reports it after a reload (`stats.denseResumableQuality`). Absent in older records.
   */
  quality?: QualityPreset;
}

const PROGRESS_KEY = 'dense/progress';

/** FNV-1a 32-bit of a string, hex. */
function fnv1a(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

/** Identity of a dense run: model (poses, cameras, points) + reference views + settings. */
function denseKey(
  model: SparseModel,
  refs: number[],
  settings: { maxSize: number; passes: string; mode: string; preset: DensePresetConfig; halo: boolean; sky?: string },
): string {
  const r6 = (v: number): number => Math.round(v * 1e6) / 1e6;
  const poses = model.images.map((im) => [im.id, im.cameraId, ...Array.from(im.pose.R, r6), ...Array.from(im.pose.t, r6)]);
  const p = settings.preset;
  return fnv1a(
    JSON.stringify({
      poses,
      cams: [...model.cameras.entries()],
      points: model.numPoints,
      xyz: model.numPoints ? [r6(model.xyz[0]), r6(model.xyz[model.xyz.length - 1])] : [],
      refs,
      maxSize: settings.maxSize,
      passes: settings.passes,
      mode: settings.mode,
      halo: settings.halo,
      sky: settings.sky ?? null,
      rs: model.rollingShutter ? [model.rollingShutter.axis, model.rollingShutter.readoutMs, !!model.rollingShutter.nu] : null,
      pm: [p.iterations, p.geometricIterations, p.numSources, p.patchMatch ?? {}, p.viewSelection ?? {}, p.fullResIterations ?? null],
    }),
  );
}

/** Delete stored PatchMatch-pass / filtered depth maps of an earlier, different dense run. */
async function clearStoredDepthMaps(store: EngineStore): Promise<void> {
  const keys = (await store.listBinaryKeys().catch(() => [] as string[])).filter((k) => /^(pm\d+|depth)\//.test(k));
  for (const k of keys) await store.deleteBinary(k).catch(() => undefined);
}

/** The depth channel only (geometric-pass sources read nothing else). */
function depthOnly(d: DepthMap | null): DepthMap {
  if (!d) return emptyDepth();
  return { width: d.width, height: d.height, depth: d.depth, normal: new Float32Array(0), cost: new Float32Array(0) };
}

/** Result of densify (kept by the engine for meshing / export). */
export interface DenseOutput {
  cloud: PointCloud;
  /** Views with colour images (downscaled) — inputs of TSDF meshing. */
  views: DenseView[];
  /** Fusion-filtered depth maps. */
  filtered: EngineDepthStore;
  meta: DenseMetaRecord;
  stats: Record<string, number | string>;
}

/** Median z-depth of the sparse points in their observing (registered) cameras. */
export function medianSparseDepth(model: SparseModel): number {
  const poses = new Map(model.images.map((im) => [im.id, im.pose]));
  const ds: number[] = [];
  const stride = Math.max(1, Math.floor(model.numPoints / 4000));
  for (let i = 0; i < model.numPoints; i += stride) {
    const pose = poses.get(model.trackImages[model.trackOffsets[i]]);
    if (!pose) continue;
    const R = pose.R,
      t = pose.t;
    const z = R[6] * model.xyz[3 * i] + R[7] * model.xyz[3 * i + 1] + R[8] * model.xyz[3 * i + 2] + t[2];
    if (z > 0) ds.push(z);
  }
  return ds.length ? median(ds) : NaN;
}

/** Evenly spread subset of `n` items (capture order kept). */
export function spreadSubset<T>(items: readonly T[], n: number): T[] {
  if (items.length <= n) return items.slice();
  if (n <= 1) return [items[0]];
  const out: T[] = [];
  let last = -1;
  for (let i = 0; i < n; i++) {
    const k = Math.round((i * (items.length - 1)) / (n - 1));
    if (k !== last) out.push(items[k]);
    last = k;
  }
  return out;
}

function emptyDepth(): DepthMap {
  return { width: 0, height: 0, depth: new Float32Array(0), normal: new Float32Array(0), cost: new Float32Array(0) };
}

function isDeviceLost(e: unknown): boolean {
  const m = String((e as Error)?.message ?? e);
  // 'instance destroyed': the worker's device-loss handler (recoverGPU) reset the holder under us
  return /device lost|device was lost|GPU device|instance destroyed/i.test(m);
}

/** Dense point cloud of the sparse model (world frame). */
export async function runDensify(deps: DenseDeps, opts: DenseOptions = {}, progress?: ProgressCallback, cancel?: CancelToken): Promise<DenseOutput> {
  const t0 = now();
  const model = deps.model;
  const quality = opts.quality ?? deps.quality;
  const preset: DensePresetConfig = resolveDensePreset(quality, deps.config.dense ?? {});
  const mc = modeConfig(deps.mode === 'auto' ? 'auto' : deps.mode);
  const stats: Record<string, number | string> = {};
  const report = (stage: string, p: number, message?: string): void => progress?.({ stage, progress: Math.max(0, Math.min(1, p)), message });
  const check = (stage: string): void => {
    if (cancel?.cancelled) throw cancelledError(stage);
  };
  if (model.images.length < 2) throw new Error('densify needs a sparse model with at least 2 registered images (run refine first)');

  report('depth_maps', 0, 'preparing views');
  let pm = await deps.holder.get();
  if (!pm && deps.holder.reason) deps.log('info', `dense: CPU PatchMatch (${deps.holder.reason})`);
  let maxSize = Math.max(32, Math.round(opts.maxImageSize ?? (pm ? preset.maxImageSize : Math.min(preset.maxImageSize, preset.cpuMaxImageSize))));
  const maxViews = Math.max(2, Math.round(opts.maxViews ?? preset.maxViews));
  // A keyframe whose photo never reached storage (failed JPEG encode / write during capture) is
  // left out: it would fail every densify (and every "Try again") at the same view.
  const storedIds = new Set((await deps.store.getFrameRecords()).map((r) => r.id));
  const withImage = model.images.filter((im) => storedIds.has(im.id));
  if (withImage.length < model.images.length) {
    const missing = model.images.filter((im) => !storedIds.has(im.id)).map((im) => im.id);
    deps.log('warn', `dense: ${missing.length} registered keyframe(s) without a stored image, skipped (${missing.join(', ')})`);
    stats.denseMissingImages = missing.length;
  }
  if (withImage.length < 2) throw new Error(`densify needs at least 2 registered images with stored photos (${withImage.length} of ${model.images.length} available)`);
  const images = spreadSubset(withImage, maxViews);
  stats.denseViews = images.length;
  stats.patchMatch = pm ? 'gpu' : 'cpu';

  // ---- memory plan (iPhone tab budget): cache sizes, fusion neighbours / pixel step / point cap;
  // the image size is lowered (× 0.85 steps, not below 480 px) when even that does not fit
  // PatchMatch passes: [½-resolution photometric (multi-scale),] photometric (prior-initialised
  // from the ½-resolution map when multi-scale), geometric × geometricPasses
  const geometricPasses = Math.max(0, Math.round(preset.geometricPasses));
  const multiScale = !!preset.multiScale;
  const passKinds: PassKind[] = [...(multiScale ? (['half'] as const) : []), 'photo', ...new Array<PassKind>(geometricPasses).fill('geo')];
  const passes = passKinds.length;
  if (multiScale) stats.denseMultiScale = 1;
  // colour-guided halo / sky clean-up of the final depth maps before fusion
  const halo = preset.haloFilter === false ? null : removeDepthHalos;
  if (halo) stats.haloFilter = 1;
  // sky test orientation per view (frames are not always upright: canonical orientation, portrait lock)
  const skyUpOf = (id: number, pose: Pose): [number, number, number] | null => denseSkyUp(pose, !!deps.gravityAligned, deps.gravityCam?.get(id));
  let skySource = 'top row';
  if (deps.gravityAligned) skySource = 'gravity-aligned model';
  else if (images.some((im) => denseSkyUp(im.pose, false, deps.gravityCam?.get(im.id)))) skySource = 'keyframe gravity';
  if (halo) stats.haloSkyUp = skySource;
  const presetStep = Number((preset.fusion as { step?: unknown } | undefined)?.step);
  const largest = (size: number): { width: number; height: number } => {
    let best = { width: 1, height: 1 };
    for (const cam of model.cameras.values()) {
      const c = undistortedCamera(cam, size);
      if (c.width * c.height > best.width * best.height) best = { width: c.width, height: c.height };
    }
    return best;
  };
  const planFor = (size: number): DensePlan => {
    const d = largest(size);
    return planDense({
      budgetMB: deps.memoryBudgetMB,
      residentBytes: deps.residentBytes ?? 0,
      views: images.length,
      width: d.width,
      height: d.height,
      numSources: Math.min(16, preset.numSources),
      geometricPasses,
      fusionMaxNeighbours: preset.fusionMaxNeighbours,
      gpu: !!pm,
      haloFilter: !!halo,
      tsdfMaxBlocks: preset.tsdfMaxBlocks,
      fusionStep: preset.fusionStep ?? (Number.isFinite(presetStep) && presetStep >= 1 ? presetStep : undefined),
    });
  };
  let plan = planFor(maxSize);
  const requested = maxSize;
  while (!plan.fits && maxSize > 480) {
    maxSize = Math.max(480, Math.round(maxSize * 0.85));
    plan = planFor(maxSize);
  }
  if (maxSize !== requested) {
    stats.denseMaxSizeRequested = requested;
    deps.log('warn', `dense: ${requested} px does not fit the ${deps.memoryBudgetMB} MB budget — using ${maxSize} px`);
  }
  stats.denseMaxSize = maxSize;
  stats.denseMemPeakMB = Math.round(plan.peakBytes / MB);
  stats.fusionStep = plan.fusionStep;
  stats.fusionMaxNeighbours = plan.fusionMaxNeighbours;
  {
    const d = largest(maxSize);
    deps.log(
      plan.fits ? 'info' : 'warn',
      `dense memory plan (${images.length} views ${d.width}×${d.height}, budget ${deps.memoryBudgetMB} MB, peak ≈ ${Math.round(plan.peakBytes / MB)} MB, storage ≈ ${Math.round(plan.storageBytes / MB)} MB): ` +
        plan.stages.map((st) => `${st.stage} ${formatBreakdown(st.parts, st.bytes)}`).join('; ') +
        `; fusion step ${plan.fusionStep}, ≤ ${plan.fusionMaxPoints} points (≈ ${plan.expectedPoints} expected), ${plan.fusionMaxNeighbours} neighbours`,
    );
  }

  // ---- geometry of the undistorted views (no pixels yet)
  const geom = images.map((im) => {
    const cam = model.cameras.get(im.cameraId)!;
    return { id: im.id, pose: im.pose, distorted: cam, cam: undistortedCamera(cam, maxSize) };
  });
  const viewIds = new Set(geom.map((g) => g.id));
  const visibility: number[][] = [];
  for (let i = 0; i < model.numPoints; i++) {
    const vis: number[] = [];
    for (let k = model.trackOffsets[i]; k < model.trackOffsets[i + 1]; k++) {
      const id = model.trackImages[k];
      if (viewIds.has(id)) vis.push(id);
    }
    visibility.push(vis);
  }
  const sel: Map<number, SourceViewSelection> = selectSourceViews(
    geom.map((g) => ({ id: g.id, pose: g.pose, cam: g.cam })),
    { xyz: model.xyz, visibility },
    // capture-mode depth margins, then the tuned DENSE_PRESETS selection options
    { depthMarginMin: mc.depthMarginMin, depthMarginMax: mc.depthMarginMax, ...(preset.viewSelection ?? {}), numSources: Math.min(16, preset.numSources) },
  );

  // ---- images: undistorted at the dense size (LRU) + colour thumbnails for fusion / TSDF
  const cache = new LruCache<number, DenseView>(plan.imageCacheBytes, (v) => rgbaBytes(v.image));
  const colourBytesPerView = plan.colourBytes / Math.max(1, geom.length);
  const colour = new Map<number, RGBAImage>();
  const geomById = new Map(geom.map((g) => [g.id, g]));
  // wall-time breakdown of the depth-map stage (ms; logged at its end)
  const perf = { decodes: 0, load: 0, inputs: 0, compute: 0, upload: 0, gpu: 0, gpuTs: 0, readback: 0, halo: 0, store: 0 };
  const loadView = (id: number): Promise<DenseView> =>
    cache.getOrLoad(id, async () => {
      const tl = now();
      perf.decodes++;
      const g = geomById.get(id)!;
      const und = await loadDenseViewImage(deps.store, model, id, g.distorted, maxSize);
      const view: DenseView = { id, cam: und.cam, pose: g.pose, image: und.image };
      if (!colour.has(id)) {
        // colour thumbnail of the SAME (rolling-shutter corrected) image: fusion / TSDF sample it
        // at the projection with the frame-stamp pose
        const w = und.image.width,
          h = und.image.height;
        const s = Math.min(1, Math.sqrt(colourBytesPerView / (4 * w * h)));
        const fs = fitSize(w, h, Math.max(64, Math.floor(s * Math.max(w, h))));
        colour.set(id, resizeRGBA(und.image, fs.width, fs.height));
      }
      perf.load += now() - tl;
      return view;
    });
  // ½-resolution views of the multi-scale pass: decoded and undistorted at half the size (exact
  // §2-rule pinhole of the full view), cached separately (¼ of the bytes: all views usually fit)
  const halfSize = Math.max(32, Math.round(maxSize / 2));
  const halfCache = new LruCache<number, DenseView>(multiScale ? plan.imageCacheBytes : 0, (v) => rgbaBytes(v.image));
  const loadHalfView = (id: number): Promise<DenseView> =>
    halfCache.getOrLoad(id, async () => {
      const tl = now();
      perf.decodes++;
      const g = geomById.get(id)!;
      const und = await loadDenseViewImage(deps.store, model, id, g.distorted, halfSize);
      perf.load += now() - tl;
      return { id, cam: und.cam, pose: g.pose, image: und.image };
    });

  // ---- depth maps
  // Resident maps are bounded explicitly: the pass stores keep ONE full map (write-through to
  // IndexedDB), geometric-pass sources come from a depth-only LRU (sources only need `depth`), and
  // a previous-pass map is deleted after its last use (peak storage ≈ one pass, not two).
  const ordered = geom.filter((g) => (sel.get(g.id)?.sources.length ?? 0) > 0);
  stats.viewsWithSources = ordered.length;
  const backend = deps.depthBackend;
  const persistent = !!backend?.persistent;
  // resume after a kill / cancel: same model + settings ⇒ keep the maps already computed
  const planKey = denseKey(model, ordered.map((g) => g.id), { maxSize, passes: passKinds.join(','), mode: deps.mode, preset, halo: !!halo, sky: halo ? skySource : undefined });
  let resumePass = 0;
  if (persistent) {
    const prog = await backend!.getValue<DenseProgress>(PROGRESS_KEY).catch(() => null);
    if (prog && prog.key === planKey && prog.pass >= 0 && prog.pass <= passes) {
      resumePass = prog.pass;
      deps.log('info', `dense: resuming ${resumePass < passes ? `pass ${resumePass + 1}/${passes}` : 'at fusion'} from stored depth maps`);
    } else await clearStoredDepthMaps(backend!);
  }
  stats.denseResumedPass = resumePass;
  const setProgress = async (pass: number | null): Promise<void> => {
    if (!persistent) return;
    await backend!.putValue(PROGRESS_KEY, pass === null ? null : ({ key: planKey, pass, quality } satisfies DenseProgress));
  };
  const srcDepth = new LruCache<number, DepthMap>(plan.sourceDepthCacheBytes, (d) => d.depth.byteLength);
  let prev: EngineDepthStore | null = resumePass > 0 ? new EngineDepthStore(backend, `pm${resumePass - 1}`, 0) : null;
  let cur: EngineDepthStore | null = resumePass >= passes ? prev : null;
  const total = passes * ordered.length;
  let done = resumePass * ordered.length;
  let skipped = 0;
  const tDepth = now();
  for (let pass = resumePass; pass < passes; pass++) {
    const kind = passKinds[pass];
    // maps of the previous pass this pass reads: the reference's (prior) and, geometric, the sources'
    const reads = (id: number): number[] => (pass === 0 ? [] : kind === 'geo' ? [id, ...sel.get(id)!.sources] : [id]);
    await setProgress(pass);
    cur = new EngineDepthStore(backend, `pm${pass}`, 0);
    const have = new Set(persistent && pass === resumePass ? await cur.scan() : []);
    // Full-resolution passes alternate direction (serpentine, the last one forward): the next
    // pass starts with the views the image LRU (and PatchMatchGPU's upload caches) still hold.
    const fullBefore = passKinds.slice(pass + 1).filter((k) => k !== 'half').length;
    const order = kind !== 'half' && fullBefore % 2 === 1 ? [...ordered].reverse() : ordered;
    // last reference (index in `order`) that reads each previous-pass map
    const lastUse = new Map<number, number>();
    order.forEach((g, i) => reads(g.id).forEach((id) => lastUse.set(id, i)));
    const releasePrev = async (i: number): Promise<void> => {
      if (!prev) return;
      for (const id of reads(order[i].id)) {
        if (lastUse.get(id) !== i) continue;
        srcDepth.delete(id);
        await prev.delete(id);
      }
    };
    for (let i = 0; i < order.length; i++) {
      const g = order[i];
      check('depth_maps');
      if (have.has(g.id)) {
        skipped++;
        done++;
        await releasePrev(i);
        continue;
      }
      const s = sel.get(g.id)!;
      const load = kind === 'half' ? loadHalfView : loadView;
      const ref = await load(g.id);
      const srcs: DenseView[] = [];
      for (const sid of s.sources) srcs.push(await load(sid));
      const range = { min: s.depthMin, max: s.depthMax };
      let geometric: { prior: DepthMap; sourceDepths: DepthMap[] } | undefined;
      const ti = now();
      if (pass > 0 && prev) {
        const prior = await prev.get(g.id);
        if (!prior) {
          await releasePrev(i);
          continue;
        }
        const sourceDepths: DepthMap[] = [];
        const p0 = prev;
        if (kind === 'geo') {
          for (const sid of s.sources) {
            const d = await srcDepth.getOrLoad(sid, async () => depthOnly(await p0.get(sid)));
            sourceDepths.push(d.width > 0 ? d : emptyDepth());
          }
        }
        geometric = { prior, sourceDepths };
      }
      perf.inputs += now() - ti;
      const pmBase = preset.patchMatch ?? {};
      const pmOpts: Partial<PatchMatchOptions> =
        kind === 'geo'
          ? { ...pmBase, geometric: true, geometricIterations: preset.geometricIterations }
          : // the full-resolution photometric pass of multi-scale starts from the ½-resolution map
            kind === 'photo' && geometric
            ? { ...pmBase, geometric: false, iterations: Math.max(1, Math.round(preset.fullResIterations ?? 2)) }
            : { ...pmBase, iterations: preset.iterations };
      let dm: DepthMap;
      const tv = now();
      if (pm) {
        await deps.waitVisible();
        try {
          const inst: PatchMatchGPU = pm;
          dm = await deps.gpuLock(() => inst.compute(ref, srcs, range, pmOpts, geometric));
        } catch (e) {
          if (!isDeviceLost(e)) throw e;
          deps.log('warn', `PatchMatchGPU: device lost — recreating (${(e as Error)?.message ?? e})`);
          deps.holder.reset();
          pm = await deps.holder.get();
          if (!pm) {
            const err = new Error(`GPU device lost during dense reconstruction: ${(e as Error)?.message ?? e}`) as Error & { code?: string };
            err.name = 'GPUDeviceLostError';
            err.code = 'gpu-lost';
            throw err;
          }
          const inst: PatchMatchGPU = pm;
          dm = await deps.gpuLock(() => inst.compute(ref, srcs, range, pmOpts, geometric));
        }
      } else {
        dm = patchMatchCPU(ref, srcs, range, pmOpts, geometric);
      }
      perf.compute += now() - tv;
      const lt = pm?.lastTimings;
      if (lt) {
        perf.upload += lt.uploadMs;
        perf.gpu += lt.initMs + lt.iterationsMs.reduce((a, b) => a + b, 0);
        perf.readback += lt.readbackMs;
        perf.gpuTs += lt.gpuMs ?? 0;
      }
      geometric = undefined;
      const th = now();
      if (halo && pass === passes - 1) {
        try {
          const skyUp = skyUpOf(g.id, g.pose);
          dm = skyUp ? halo(dm, ref.image, ref.cam, { skyUp }) : halo(dm, ref.image, ref.cam);
        } catch (e) {
          deps.log('warn', `depth halo filter failed: ${(e as Error)?.message ?? e}`);
        }
      }
      const tp = now();
      perf.halo += tp - th;
      await cur.put(g.id, dm);
      await releasePrev(i);
      perf.store += now() - tp;
      done++;
      const label = kind === 'half' ? 'photometric ½' : kind === 'photo' ? 'photometric' : `geometric ${pass - passKinds.indexOf('geo') + 1}`;
      report('depth_maps', done / total, `${label} ${done}/${total} (${Math.round(now() - tv)} ms)`);
      await tick();
    }
    if (prev) await prev.clear();
    srcDepth.clear();
    halfCache.clear();
    prev = cur;
  }
  if (skipped) stats.denseResumedMaps = skipped;
  const depthMapsMs = now() - tDepth;
  stats.depthMapsMs = Math.round(depthMapsMs);
  {
    const s = (ms: number): string => (ms / 1000).toFixed(1);
    deps.log(
      'info',
      `dense depth maps ${s(depthMapsMs)} s: view loads ${s(perf.load)} s (${perf.decodes} decodes), pass inputs ${s(perf.inputs)} s, ` +
        `PatchMatch ${s(perf.compute)} s (upload ${s(perf.upload)}, GPU passes ${s(perf.gpu)}${perf.gpuTs > 0 ? ` [GPU timestamps ${s(perf.gpuTs)}]` : ''}, readback ${s(perf.readback)}), ` +
        `halo filter ${s(perf.halo)} s, store ${s(perf.store)} s` +
        (pm ? `; GPU upload caches: images ${pm.cacheStats.imageHits} hits / ${pm.cacheStats.imageMisses} uploads, depth maps ${pm.cacheStats.depthHits} / ${pm.cacheStats.depthMisses}` : ''),
    );
  }
  const finalStore = cur!;
  cache.clear();
  // PatchMatchGPU's buffers (≈ 32 B per reference px + 5 B per source px: ~140 MB at 1600 px) are
  // not needed for fusion / meshing: free them (recreated by the next densify)
  if (pm) deps.holder.reset();
  await setProgress(passes);

  // ---- fusion
  const tFuse = now();
  check('fusion');
  report('fusion', 0, 'fusing depth maps');
  const filtered = new EngineDepthStore(backend, 'depth', 0);
  await filtered.clear();
  // colour views: thumbnails (fusion / TSDF sample colours at any scale)
  const views: DenseView[] = [];
  for (const g of ordered) {
    let img = colour.get(g.id);
    if (!img) {
      const v = await loadView(g.id);
      img = colour.get(g.id) ?? v.image;
    }
    views.push({ id: g.id, cam: g.cam, pose: g.pose, image: img });
  }
  cache.clear();
  const neighbours = neighboursFromSelection(sel);
  const minViews = deps.mode === 'scene' ? preset.fusionMinViewsScene : preset.fusionMinViewsObject;
  const tFuseMaps = now();
  let cloud = await fuseDepthMaps(
    views,
    finalStore,
    neighbours,
    // halo clean-up happens once: on the depth maps above (engine) or in fusion (preset), never both
    {
      ...(preset.fusion ?? {}),
      ...(halo ? { haloFilter: null } : {}),
      minViews,
      maxNeighbours: plan.fusionMaxNeighbours,
      step: plan.fusionStep,
      maxPoints: plan.fusionMaxPoints,
      filteredStore: filtered,
    },
    (e) => report('fusion', 0.9 * Math.max(0, e.progress), e.message),
    cancel,
  );
  if (cloud.count >= plan.fusionMaxPoints) deps.log('warn', `fusion stopped at the memory cap of ${plan.fusionMaxPoints} points (later views not fused)`);
  stats.fusedPoints = cloud.count;
  const fuseMs = now() - tFuseMaps;
  // fused: a later densify starts over (progress first, so a kill in between never resumes into
  // deleted maps)
  await setProgress(null);
  await finalStore.clear();
  check('fusion');
  report('fusion', 0.92, `filtering ${cloud.count} points`);
  await tick();
  const tCap = now();
  // point cap: footprint-adaptive LOD (cells ∝ each point's own radius). A single voxel size
  // from the median radius merged the close-range object of a scan ~10× (object-real 800 px:
  // 3.5 M → 0.57 M points, accuracy p50 0.31 → 0.45 mm; yard-real precision 0.96 → 0.80).
  if (cloud.count > preset.maxPoints) cloud = footprintDownsample(cloud, preset.maxPoints);
  if (preset.sorK > 0 && cloud.count > preset.sorK + 1) cloud = filterPointCloud(cloud, { statistical: { k: preset.sorK, std: preset.sorStd } });
  stats.densePoints = cloud.count;
  stats.fusionMs = Math.round(now() - tFuse);
  deps.log(
    'info',
    `dense fusion ${(stats.fusionMs / 1000).toFixed(1)} s: fuseDepthMaps ${(fuseMs / 1000).toFixed(1)} s → ${stats.fusedPoints} points, point cap / filter ${((now() - tCap) / 1000).toFixed(1)} s → ${cloud.count} points`,
  );
  stats.denseMs = Math.round(now() - t0);
  report('fusion', 1, `${cloud.count} points`);
  const meta: DenseMetaRecord = {
    views: views.map((v) => ({
      id: v.id,
      cam: v.cam,
      pose: { R: Array.from(v.pose.R), t: Array.from(v.pose.t) },
      depthWidth: v.cam.width,
      depthHeight: v.cam.height,
    })),
    maxImageSize: maxSize,
  };
  return { cloud, views, filtered, meta, stats };
}

/**
 * Colour views for meshing after a reload (dense views without images in memory): decode the
 * keyframes again at a small size and undistort.
 */
export async function rebuildColourViews(store: EngineStore, model: SparseModel, meta: DenseMetaRecord, maxSide = 480): Promise<DenseView[]> {
  const out: DenseView[] = [];
  for (const v of meta.views) {
    const im = model.images.find((x) => x.id === v.id);
    const cam = im ? model.cameras.get(im.cameraId) : undefined;
    const stored = await store.getFrameImage(v.id);
    const pose = { R: Float64Array.from(v.pose.R), t: Float64Array.from(v.pose.t) };
    if (!cam || !stored) continue;
    const fs = fitSize(v.cam.width, v.cam.height, maxSide);
    const rgba = await decodeStoredImage(stored, fs.width, fs.height);
    out.push({ id: v.id, cam: v.cam, pose, image: denseViewImage(rgba, cam as CameraIntrinsics, maxSide, model, v.id).image });
  }
  return out;
}

/**
 * Pinhole (undistorted, ≤ maxSize) image of a keyframe for the dense stage. When the model's
 * keypoints were rolling-shutter corrected (`SparseModel.rollingShutter`), its poses are those of
 * the frame stamp and the pixels are corrected too (`rsCorrectPinholeImage`), else every
 * fast-rotating view is skewed against its pose. Used for the PatchMatch views AND their colour
 * thumbnails (fusion / TSDF), in-session and after a reload alike.
 */
export function denseViewImage(rgba: RGBAImage, distorted: CameraIntrinsics, maxSize: number, model: SparseModel, id: number): { image: RGBAImage; cam: CameraIntrinsics } {
  const und = undistortImage(rgba, distorted, maxSize);
  const rs = model.rollingShutter;
  const w = rs ? rs.omega[id] : undefined;
  return { cam: und.cam, image: rs && w ? rsCorrectPinholeImage(und.image, und.cam, w, rs, rs.nu?.[id] ?? null) : und.image };
}

/**
 * A stored keyframe as a dense view image: decoded at the undistorted size, undistorted and
 * rolling-shutter corrected (`denseViewImage`) — the views of PatchMatch and of the splat stage.
 */
export async function loadDenseViewImage(store: EngineStore, model: SparseModel, id: number, distorted: CameraIntrinsics, maxSize: number): Promise<{ image: RGBAImage; cam: CameraIntrinsics }> {
  const stored = await store.getFrameImage(id);
  if (!stored) throw new Error(`keyframe image ${id} missing`);
  const c = undistortedCamera(distorted, maxSize);
  return denseViewImage(await decodeStoredImage(stored, c.width, c.height), distorted, maxSize, model, id);
}

export interface MeshDeps {
  quality: QualityPreset;
  mode: ResolvedMode | 'auto';
  config: EngineConfig;
  model: SparseModel | null;
  memoryBudgetMB: number;
}

/** TSDF mesh of the fusion-filtered depth maps (+ optional simplification). */
export async function runMesh(
  deps: MeshDeps,
  views: DenseView[],
  filtered: DepthMapStore,
  opts: MeshOptions = {},
  progress?: ProgressCallback,
  cancel?: CancelToken,
): Promise<{ mesh: TriangleMesh; stats: Record<string, number | string> }> {
  const t0 = now();
  const preset = resolveDensePreset(opts.quality ?? deps.quality, deps.config.dense ?? {});
  const mc = modeConfig(deps.mode === 'auto' ? 'auto' : deps.mode);
  const stats: Record<string, number | string> = {};
  let maxDepth = Infinity;
  if (Number.isFinite(mc.tsdfMaxDepthFactor) && deps.model) {
    const d = medianSparseDepth(deps.model);
    if (d > 0) maxDepth = mc.tsdfMaxDepthFactor * d;
  }
  // block budget scaled to the memory budget (≈ 5.6 KB per block)
  const maxBlocks = Math.min(preset.tsdfMaxBlocks, Math.floor((0.25 * deps.memoryBudgetMB * 1024 * 1024) / 5632));
  progress?.({ stage: 'meshing', progress: 0, message: 'TSDF' });
  let mesh = await tsdfMesh(
    views,
    filtered,
    {
      voxelSize: opts.voxelSize && opts.voxelSize > 0 ? opts.voxelSize : undefined,
      maxDepth,
      maxBlocks,
      autoVoxelPixels: mc.tsdfAutoVoxelPixels,
      onStats: (s) => {
        stats.voxelSize = s.voxelSize;
      },
    },
    (e) => progress?.({ stage: 'meshing', progress: 0.85 * Math.max(0, e.progress), message: e.message }),
    cancel,
  );
  if (cancel?.cancelled) throw cancelledError('meshing');
  if (opts.targetTriangles && opts.targetTriangles > 0 && mesh.triangleCount > opts.targetTriangles) {
    progress?.({ stage: 'meshing', progress: 0.9, message: `simplifying ${mesh.triangleCount} → ${opts.targetTriangles} triangles` });
    await tick();
    mesh = simplifyMesh(mesh, Math.round(opts.targetTriangles));
  }
  stats.triangles = mesh.triangleCount;
  stats.vertices = mesh.vertexCount;
  stats.meshingMs = Math.round(now() - t0);
  progress?.({ stage: 'meshing', progress: 1, message: `${mesh.triangleCount} triangles` });
  return { mesh, stats };
}
