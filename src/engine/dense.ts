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
import type { CameraIntrinsics, CancelToken, PointCloud, ProgressCallback, RGBAImage, TriangleMesh } from '../core/types';
import * as denseModule from '../dense';
import { fuseDepthMaps } from '../dense/fusion';
import { patchMatchCPU } from '../dense/patchmatch-cpu';
import { PatchMatchGPU } from '../dense/patchmatch-gpu';
import { filterPointCloud, voxelDownsample } from '../dense/pointcloud';
import { simplifyMesh } from '../dense/simplify';
import { tsdfMesh } from '../dense/tsdf';
import type { DenseView, DepthMap, DepthMapStore } from '../dense/types';
import { undistortImage, undistortedCamera } from '../dense/undistort';
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

  constructor(private readonly preferGPU: boolean) {}

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
      const pm = await PatchMatchGPU.create(ctx);
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
}

/** Persisted dense progress (`dense/progress` in the depth backend): resume after a kill / cancel. */
interface DenseProgress {
  /** Hash of the model + view selection inputs + settings the maps were computed with. */
  key: string;
  /** Pass in progress (0 = photometric), or = number of passes when fusing. */
  pass: number;
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
function denseKey(model: SparseModel, refs: number[], settings: { maxSize: number; passes: number; mode: string; preset: DensePresetConfig; halo: boolean }): string {
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
      pm: [p.iterations, p.geometricIterations, p.numSources, p.patchMatch ?? {}, p.viewSelection ?? {}],
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

type HaloFilter = (dm: DepthMap, image: RGBAImage, cam: CameraIntrinsics, options?: Record<string, unknown>) => DepthMap;

/** `removeDepthHalos` of src/dense when that module exports it (colour-guided halo clean-up before fusion). */
function haloFilter(): HaloFilter | null {
  const f = (denseModule as unknown as Record<string, unknown>)[['remove', 'DepthHalos'].join('')];
  return typeof f === 'function' ? (f as HaloFilter) : null;
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
  const images = spreadSubset(model.images, maxViews);
  stats.denseViews = images.length;
  stats.patchMatch = pm ? 'gpu' : 'cpu';

  // ---- memory plan (iPhone tab budget): cache sizes, fusion neighbours / pixel step / point cap;
  // the image size is lowered (× 0.85 steps, not below 480 px) when even that does not fit
  const passes = 1 + Math.max(0, Math.round(preset.geometricPasses));
  const halo = preset.haloFilter === false ? null : haloFilter();
  if (halo) stats.haloFilter = 1;
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
      geometricPasses: passes - 1,
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
  const loadView = (id: number): Promise<DenseView> =>
    cache.getOrLoad(id, async () => {
      const g = geomById.get(id)!;
      const stored = await deps.store.getFrameImage(id);
      if (!stored) throw new Error(`keyframe image ${id} missing`);
      const rgba = await decodeStoredImage(stored, g.cam.width, g.cam.height);
      const und = undistortImage(rgba, g.distorted, maxSize);
      const view: DenseView = { id, cam: und.cam, pose: g.pose, image: und.image };
      if (!colour.has(id)) {
        const w = und.image.width,
          h = und.image.height;
        const s = Math.min(1, Math.sqrt(colourBytesPerView / (4 * w * h)));
        const fs = fitSize(w, h, Math.max(64, Math.floor(s * Math.max(w, h))));
        colour.set(id, resizeRGBA(und.image, fs.width, fs.height));
      }
      return view;
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
  const planKey = denseKey(model, ordered.map((g) => g.id), { maxSize, passes, mode: deps.mode, preset, halo: !!halo });
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
    await backend!.putValue(PROGRESS_KEY, pass === null ? null : ({ key: planKey, pass } satisfies DenseProgress));
  };
  const srcDepth = new LruCache<number, DepthMap>(plan.sourceDepthCacheBytes, (d) => d.depth.byteLength);
  let prev: EngineDepthStore | null = resumePass > 0 ? new EngineDepthStore(backend, `pm${resumePass - 1}`, 0) : null;
  let cur: EngineDepthStore | null = resumePass >= passes ? prev : null;
  const total = passes * ordered.length;
  let done = resumePass * ordered.length;
  let skipped = 0;
  const tDepth = now();
  for (let pass = resumePass; pass < passes; pass++) {
    await setProgress(pass);
    cur = new EngineDepthStore(backend, `pm${pass}`, 0);
    const have = new Set(persistent && pass === resumePass ? await cur.scan() : []);
    // last reference (index in `ordered`) that reads each previous-pass map
    const lastUse = new Map<number, number>();
    if (pass > 0) ordered.forEach((g, i) => [g.id, ...sel.get(g.id)!.sources].forEach((id) => lastUse.set(id, i)));
    const releasePrev = async (i: number): Promise<void> => {
      if (!prev) return;
      const gi = ordered[i];
      const ids = [gi.id, ...sel.get(gi.id)!.sources];
      for (const id of ids) {
        if (lastUse.get(id) !== i) continue;
        srcDepth.delete(id);
        await prev.delete(id);
      }
    };
    for (let i = 0; i < ordered.length; i++) {
      const g = ordered[i];
      check('depth_maps');
      if (have.has(g.id)) {
        skipped++;
        done++;
        await releasePrev(i);
        continue;
      }
      const s = sel.get(g.id)!;
      const ref = await loadView(g.id);
      const srcs: DenseView[] = [];
      for (const sid of s.sources) srcs.push(await loadView(sid));
      const range = { min: s.depthMin, max: s.depthMax };
      let geometric: { prior: DepthMap; sourceDepths: DepthMap[] } | undefined;
      if (pass > 0 && prev) {
        const prior = await prev.get(g.id);
        if (!prior) {
          await releasePrev(i);
          continue;
        }
        const sourceDepths: DepthMap[] = [];
        const p0 = prev;
        for (const sid of s.sources) {
          const d = await srcDepth.getOrLoad(sid, async () => depthOnly(await p0.get(sid)));
          sourceDepths.push(d.width > 0 ? d : emptyDepth());
        }
        geometric = { prior, sourceDepths };
      }
      const pmOpts =
        pass === 0
          ? { ...(preset.patchMatch ?? {}), iterations: preset.iterations }
          : { ...(preset.patchMatch ?? {}), geometric: true, geometricIterations: preset.geometricIterations };
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
      geometric = undefined;
      if (halo && pass === passes - 1) {
        try {
          dm = halo(dm, ref.image, ref.cam);
        } catch (e) {
          deps.log('warn', `depth halo filter failed: ${(e as Error)?.message ?? e}`);
        }
      }
      await cur.put(g.id, dm);
      await releasePrev(i);
      done++;
      report('depth_maps', done / total, `${pass === 0 ? 'photometric' : `geometric ${pass}`} ${done}/${total} (${Math.round(now() - tv)} ms)`);
      await tick();
    }
    if (prev) await prev.clear();
    srcDepth.clear();
    prev = cur;
  }
  if (skipped) stats.denseResumedMaps = skipped;
  stats.depthMapsMs = Math.round(now() - tDepth);
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
  // fused: a later densify starts over (progress first, so a kill in between never resumes into
  // deleted maps)
  await setProgress(null);
  await finalStore.clear();
  check('fusion');
  report('fusion', 0.92, `filtering ${cloud.count} points`);
  await tick();
  if (cloud.count > preset.maxPoints) {
    // voxel size from the median radius so that about maxPoints remain
    const r = cloud.radii ? median(Array.from(cloud.radii.subarray(0, Math.min(cloud.count, 100_000)))) : 0;
    if (r > 0) cloud = voxelDownsample(cloud, 2 * r * Math.sqrt(cloud.count / preset.maxPoints));
  }
  if (preset.sorK > 0 && cloud.count > preset.sorK + 1) cloud = filterPointCloud(cloud, { statistical: { k: preset.sorK, std: preset.sorStd } });
  stats.densePoints = cloud.count;
  stats.fusionMs = Math.round(now() - tFuse);
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
    const und = undistortImage(rgba, cam as CameraIntrinsics, maxSide);
    out.push({ id: v.id, cam: v.cam, pose, image: und.image });
  }
  return out;
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
