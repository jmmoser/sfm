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
  return /device lost|device was lost|GPU device/i.test(m);
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
  const maxSize = Math.max(32, Math.round(opts.maxImageSize ?? (pm ? preset.maxImageSize : Math.min(preset.maxImageSize, preset.cpuMaxImageSize))));
  const maxViews = Math.max(2, Math.round(opts.maxViews ?? preset.maxViews));
  const images = spreadSubset(model.images, maxViews);
  stats.denseViews = images.length;
  stats.denseMaxSize = maxSize;
  stats.patchMatch = pm ? 'gpu' : 'cpu';

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
  const budgetBytes = deps.memoryBudgetMB * 1024 * 1024;
  const cache = new LruCache<number, DenseView>(Math.max(64 << 20, 0.15 * budgetBytes), (v) => rgbaBytes(v.image));
  const colourBudget = Math.max(16 << 20, 0.08 * budgetBytes);
  const colourBytesPerView = colourBudget / Math.max(1, geom.length);
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
  const depthCacheBytes = Math.max(48 << 20, 0.25 * budgetBytes);
  const passes = 1 + Math.max(0, Math.round(preset.geometricPasses));
  const halo = preset.haloFilter === false ? null : haloFilter();
  if (halo) stats.haloFilter = 1;
  const ordered = geom.filter((g) => (sel.get(g.id)?.sources.length ?? 0) > 0);
  stats.viewsWithSources = ordered.length;
  let prev: EngineDepthStore | null = null;
  let cur: EngineDepthStore | null = null;
  const total = passes * ordered.length;
  let done = 0;
  const tDepth = now();
  for (let pass = 0; pass < passes; pass++) {
    cur = new EngineDepthStore(deps.depthBackend, `pm${pass}`, depthCacheBytes);
    for (const g of ordered) {
      check('depth_maps');
      const s = sel.get(g.id)!;
      const ref = await loadView(g.id);
      const srcs: DenseView[] = [];
      for (const sid of s.sources) srcs.push(await loadView(sid));
      const range = { min: s.depthMin, max: s.depthMax };
      let geometric: { prior: DepthMap; sourceDepths: DepthMap[] } | undefined;
      if (pass > 0 && prev) {
        const prior = await prev.get(g.id);
        if (!prior) continue;
        const sourceDepths: DepthMap[] = [];
        for (const sid of s.sources) sourceDepths.push((await prev.get(sid)) ?? emptyDepth());
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
      if (halo && pass === passes - 1) {
        try {
          dm = halo(dm, ref.image, ref.cam);
        } catch (e) {
          deps.log('warn', `depth halo filter failed: ${(e as Error)?.message ?? e}`);
        }
      }
      await cur.put(g.id, dm);
      done++;
      report('depth_maps', done / total, `${pass === 0 ? 'photometric' : `geometric ${pass}`} ${done}/${total} (${Math.round(now() - tv)} ms)`);
      await tick();
    }
    if (prev) await prev.clear();
    prev = cur;
  }
  stats.depthMapsMs = Math.round(now() - tDepth);
  const finalStore = cur!;
  cache.clear();

  // ---- fusion
  const tFuse = now();
  check('fusion');
  report('fusion', 0, 'fusing depth maps');
  const filtered = new EngineDepthStore(deps.depthBackend, 'depth', depthCacheBytes);
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
    { ...(preset.fusion ?? {}), ...(halo ? { haloFilter: null } : {}), minViews, maxNeighbours: preset.fusionMaxNeighbours, filteredStore: filtered },
    (e) => report('fusion', 0.9 * Math.max(0, e.progress), e.message),
    cancel,
  );
  stats.fusedPoints = cloud.count;
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
