/**
 * Memory accounting for the iPhone tab budget (ARCHITECTURE §2.1; docs/research/ios-platform.md
 * §1: the WebContent process — main thread, every worker, JS heaps, Wasm, canvases, decoded
 * images, VideoFrames / ImageBitmaps AND WebGPU buffers / textures — is killed by jetsam without
 * warning above ≈ 1.5 GB; design for ≤ 1.2 GB on 8 / 12 GB iPhones, ≤ 0.85 GB on 6 GB, ≤ 0.6 GB on
 * 4 GB).
 *
 *  - `captureMemory`: estimate of what the live capture keeps resident (features, colours,
 *    verified pairs, correspondence graph, live reconstruction, vocabulary, IMU log, writes not
 *    yet in IndexedDB), computed from the actual data structures (typed arrays exactly, JS objects
 *    with per-object constants for JavaScriptCore).
 *  - `deviceMemoryBudgetMB`: budget from the WebGPU adapter (`maxBufferSize` is the only RAM
 *    proxy iOS exposes: 26.2+ ≈ min(1 GiB, 23–25 % of RAM / 3); ios-webgpu.md §2).
 *  - `planDense`: analytic peak of each dense stage (PatchMatch, fusion, TSDF) and the caps that
 *    keep it inside the budget: image / depth-map cache sizes, fusion neighbours, the fusion
 *    pixel step and point cap, TSDF blocks.
 *  - `planSplat` / `fitSplatPlan`: the Gaussian-splatting stage (trainer GPU buffers from
 *    `SplatTrainer.memoryEstimate` = `estimateSplatMemory`, initialisation, view decoding,
 *    checkpoints) and the degradation (fewer views, Gaussians, smaller images) that fits it.
 *
 * All sizes are bytes unless a name ends in `MB`.
 *
 * Limits: every number here is an estimate — iOS exposes no `performance.memory`, so there is no
 * on-device memory profile, the JSC heap slack on top of live data is covered only by the 250 MB
 * page reserve, and the `maxBufferSize` budget classes are unverified on devices. Per-stage budget
 * estimates and the measured per-keyframe cost: docs/research/on-device-benchmark.md. Descriptors
 * are ≈ 86 % of the feature bytes and only needed for matching: paging them to the store would
 * roughly double the keyframe cap.
 */
import type { FeatureSet } from '../core/types';
import type { CorrespondenceGraph } from '../sfm/correspondence-graph';
import type { FeatureDatabase } from '../sfm/database';
import type { Reconstruction } from '../sfm/reconstruction';
import type { VocabularyIndex } from '../sfm/vocab';
import { estimateSplatMemory, paramFloatsPerGaussian } from '../splat/types';

export const MB = 1024 * 1024;

/** Per-unit costs used by the estimates (bytes). */
export const MEM = Object.freeze({
  /** RecPoint object + xyz Float64Array + colour array + track array + Map entry (JSC, approx.). */
  recPoint: 300,
  /** One track element object ({ image, feature }) + its slot in the track array. */
  recObservation: 40,
  /** ImuSample object with three 3-element arrays. */
  imuSample: 200,
  /** Fixed part of a stored two-view geometry (object, E / F / H / R / t). */
  twoViewFixed: 400,
  /** Correspondence-graph pool: 2 directed links × (next, image, feature) Int32. */
  graphPerMatch: 24,
  /** Depth map in JS: depth f32 + normal 3 × f32 + cost f32. */
  depthMapPx: 20,
  /** Depth map as serialised for IndexedDB (float16 normals / cost) — also the put() copy. */
  depthStoredPx: 12,
  /** Depth-only map (geometric-pass sources need only `depth`). */
  depthOnlyPx: 4,
  rgbaPx: 4,
  /**
   * PatchMatchGPU (src/dense/patchmatch-gpu.ts): 32 B per reference px, 1 B per source px, +4 B
   * per source px geometric; fixed: uniforms, ≤ 8 MB staging and the upload caches' slack layers
   * (≤ 8 MB each, `PatchMatchGPU.cacheSlackBytes`).
   */
  pmGpuRefPx: 32,
  pmGpuSrcPx: 1,
  pmGpuGeomSrcPx: 4,
  pmGpuFixed: 28 * MB,
  /** Fused point arrays: position + normal 3 × f32, colour 3 B, radius + confidence f32. */
  fusedPoint: 35,
  /**
   * Peak bytes per fused point at the end of `fuseDepthMaps`: the growing sink (capacity ≤ 2 ×
   * count) plus the sliced result.
   */
  fusedPointPeak: 105,
  /** Fusion masks: 1 bit per pixel per view. */
  fusionMaskPx: 1 / 8,
  /** TSDF block (8³ voxels × 11 B). */
  tsdfBlock: 5632,
  /** TSDF per-pixel scratch of buildTsdfVolume. */
  tsdfScratchPx: 16,
  /**
   * Everything outside the engine's data: page (three.js viewer, DOM, video element, UI), both JS
   * heaps' baseline, code, the SIMD Wasm arena when idle. Kept free in every plan.
   */
  reserve: 250 * MB,
  /**
   * Fraction of reference pixels that become fused points (measured: 0.19 object, 0.23 yard at
   * 40 views; it falls with more overlapping views, so this is conservative for 80–150 views).
   */
  fusedFraction: 0.2,
  /**
   * Splat initialisation per input point: the dense cloud as stored (float16 container) + as
   * decoded (35 B) when it is not resident, the footprint LOD's per-point scratch, and per seeded
   * point the k-d tree, σ and the Gaussian (`initFromPointCloud`).
   */
  splatCloudLoadPoint: 55,
  splatLodPoint: 32,
  splatInitPoint: 120,
  /** Decoding one training view: the full-size JPEG bitmap + canvas (≈ 2 × 12 MB at 1920×1440) … */
  splatDecodeFixed: 24 * MB,
  /** … and decoded / undistorted / rolling-shutter-corrected copies at the training size (RGBA). */
  splatDecodePx: 12,
  /** Background-shell colour thumbnails (all views, ≤ 200 px). */
  splatThumbBytes: 4 * 200 * 150,
});

/** Named byte counts. */
export type MemoryBreakdown = Record<string, number>;

export function featureSetBytes(fs: FeatureSet): number {
  return fs.keypoints.byteLength + fs.descriptors.byteLength + fs.scores.byteLength;
}

export function sumBreakdown(b: MemoryBreakdown): number {
  let s = 0;
  for (const v of Object.values(b)) s += v;
  return s;
}

/** "312 MB (features 180, pairs 60, …)" — components ≥ 1 MB, largest first. */
export function formatBreakdown(b: MemoryBreakdown, total = sumBreakdown(b)): string {
  const parts = Object.entries(b)
    .filter(([, v]) => v >= MB)
    .sort((x, y) => y[1] - x[1])
    .map(([k, v]) => `${k} ${Math.round(v / MB)}`);
  return `${Math.round(total / MB)} MB${parts.length ? ` (${parts.join(', ')})` : ''}`;
}

/** Resident size of a reconstruction (points with their tracks, per-image arrays; keypoints are shared with the DB). */
export function reconstructionBytes(rec: Reconstruction): number {
  let b = rec.numPoints * MEM.recPoint + rec.numObservations * MEM.recObservation;
  for (const img of rec.images.values()) b += img.point3D.byteLength + 2 * img.point3D.length + 200;
  return b;
}

/** Inputs of `captureMemory` (the CaptureState / LiveMapper parts that grow with the capture). */
export interface CaptureMemoryInput {
  db: FeatureDatabase;
  graph: CorrespondenceGraph;
  rec: Reconstruction | null;
  vocab: VocabularyIndex | null;
  imuSamples: number;
  /** Bytes of writes handed to the store and not yet completed (JPEG blobs, features, …). */
  pendingWriteBytes?: number;
  /** Original keypoints of rolling-shutter corrected keyframes (`CaptureState.rsOriginal`). */
  rsOriginal?: ReadonlyMap<number, Float32Array>;
  /** Frame held by best-of-window keyframing (source bitmap, rotated canvas, small copy). */
  heldFrameBytes?: number;
}

/** Estimated resident memory of a capture, by component. */
export function captureMemory(c: CaptureMemoryInput): MemoryBreakdown {
  let features = 0;
  let colours = 0;
  let graphNodes = 0;
  for (const [id, fs] of c.db.features) {
    features += featureSetBytes(fs);
    colours += c.db.colors.get(id)?.byteLength ?? 0;
    graphNodes += 6 * fs.count;
  }
  let pairs = 0;
  for (const [, , g] of c.db.pairs()) pairs += MEM.twoViewFixed + g.inliers.byteLength;
  // inverted copies cached by FeatureDatabase.getTwoView(b > a) (private: read defensively)
  const inv = (c.db as unknown as { invCache?: Map<number, { inliers: Uint32Array }> }).invCache;
  if (inv instanceof Map) for (const g of inv.values()) pairs += MEM.twoViewFixed + (g?.inliers?.byteLength ?? 0);
  let graphPairs = 0;
  for (const [, , n] of c.graph.imagePairs()) graphPairs += 8 * n + 64;
  // the pool arrays at their current capacity (they grow by doubling)
  const graphPool = 12 * c.graph.nextOf.length;
  const out: MemoryBreakdown = {
    features,
    colours,
    pairs,
    graph: graphNodes + graphPairs + graphPool,
    reconstruction: c.rec ? reconstructionBytes(c.rec) : 0,
    vocabulary: c.vocab ? c.vocab.numWords * 128 * 16 + c.vocab.size * Math.min(c.vocab.numWords, 2048) * 8 : 0,
    imu: c.imuSamples * MEM.imuSample,
  };
  if (c.pendingWriteBytes) out.pendingWrites = c.pendingWriteBytes;
  if (c.rsOriginal && c.rsOriginal.size) {
    let b = 0;
    for (const kp of c.rsOriginal.values()) b += kp.byteLength + 16;
    out.rsOriginal = b;
  }
  if (c.heldFrameBytes) out.heldFrame = c.heldFrameBytes;
  return out;
}

// ---------------------------------------------------------------------------------------------
// Device budget
// ---------------------------------------------------------------------------------------------

/** Subset of the GPU context the budget is derived from. */
export interface GpuMemoryHints {
  info?: { vendor?: string; architecture?: string; description?: string };
  limits?: { maxBufferSize?: number } | null;
}

/**
 * Tab budget (MB) for an Apple GPU from `maxBufferSize` (ios-webgpu.md §2, ios-platform.md §1):
 * Safari 26.2+ reports max(256 MiB, min(1 GiB, MTLDevice.maxBufferLength / 3)) with
 * maxBufferLength ≈ 23–25 % of RAM ⇒ 4 GB ≈ 330 MiB → 600 MB; 6 GB ≈ 450–480 MiB → 850 MB;
 * 8 GB ≈ 0.6–0.7 GiB and 12 GB = 1 GiB → 1200 MB. Exactly 256 MiB (the floor: Safari 26.0–26.1
 * on any iPhone, or a small device) is ambiguous → 850 MB. Non-Apple adapters → null (the
 * caller keeps its default).
 */
export function deviceMemoryBudgetMB(ctx: GpuMemoryHints | null | undefined): { budgetMB: number; reason: string } | null {
  if (!ctx) return null;
  const desc = `${ctx.info?.vendor ?? ''} ${ctx.info?.architecture ?? ''} ${ctx.info?.description ?? ''}`;
  if (!/apple/i.test(desc)) return null;
  const mbs = ctx.limits?.maxBufferSize;
  if (typeof mbs !== 'number' || !(mbs > 0)) return null;
  const mib = mbs / MB;
  let budgetMB: number;
  if (mib >= 560) budgetMB = 1200;
  else if (mib >= 400) budgetMB = 850;
  else if (mib > 257) budgetMB = 600;
  else budgetMB = 850;
  return { budgetMB, reason: `Apple GPU, maxBufferSize ${Math.round(mib)} MiB` };
}

// ---------------------------------------------------------------------------------------------
// Dense plan
// ---------------------------------------------------------------------------------------------

export interface DensePlanInput {
  budgetMB: number;
  /** Engine data that stays resident during the dense stage (sparse model, capture state if kept). */
  residentBytes: number;
  /** Reference views and the undistorted dense image size. */
  views: number;
  width: number;
  height: number;
  numSources: number;
  geometricPasses: number;
  fusionMaxNeighbours: number;
  /** PatchMatch on the GPU (GPU buffers count against the same process on iOS). */
  gpu: boolean;
  /** Halo filter on the final maps (one more map-sized copy per view). */
  haloFilter: boolean;
  tsdfMaxBlocks: number;
  /** Explicit fusion pixel step (else derived from the budget). */
  fusionStep?: number;
}

export interface DenseStagePeak {
  stage: 'patchmatch' | 'fusion' | 'mesh';
  bytes: number;
  parts: MemoryBreakdown;
}

export interface DensePlan {
  /** Undistorted RGBA images (LRU). */
  imageCacheBytes: number;
  /** Depth-only maps of geometric-pass sources (LRU). */
  sourceDepthCacheBytes: number;
  /** Colour thumbnails kept for fusion / TSDF (all views). */
  colourBytes: number;
  /** Fusion neighbours (≤ the preset's; lowered only when the resident maps alone would not fit). */
  fusionMaxNeighbours: number;
  /** Fusion pixel step (1 = every reference pixel). */
  fusionStep: number;
  /** Hard cap on fused points (fusion stops there; sized so the cloud fits). */
  fusionMaxPoints: number;
  /** Expected fused points at `fusionStep`. */
  expectedPoints: number;
  tsdfMaxBlocks: number;
  /** Bytes available to the engine (budget − reserve − resident). */
  shareBytes: number;
  stages: DenseStagePeak[];
  /** Largest stage peak + reserve + resident ≤ budget. */
  fits: boolean;
  peakBytes: number;
  /**
   * IndexedDB space the dense stage needs at its peak: the maps of the running pass, the
   * not-yet-released part of the previous pass (≈ 20 %) and the fusion-filtered maps.
   */
  storageBytes: number;
}

/** Peak-memory plan of densify + mesh, and the caps that keep it inside the budget. */
export function planDense(p: DensePlanInput): DensePlan {
  const budget = p.budgetMB * MB;
  const px = Math.max(1, p.width * p.height);
  const views = Math.max(1, p.views);
  const S = Math.max(1, p.numSources);
  const share = Math.max(64 * MB, budget - MEM.reserve - p.residentBytes);
  const img = MEM.rgbaPx * px;
  const map = MEM.depthMapPx * px;

  // ---- PatchMatch passes
  const imageCacheBase = Math.max((S + 2) * img, Math.min(0.15 * budget, 24 * img));
  const geometric = p.geometricPasses > 0;
  const sourceDepthCacheBytes = geometric ? Math.max((S + 1) * MEM.depthOnlyPx * px, Math.min(0.1 * budget, 32 * MEM.depthOnlyPx * px)) : 0;
  const colourBytes = Math.max(16 * MB, 0.08 * budget);
  const gpuBytes = p.gpu ? MEM.pmGpuRefPx * px + S * px * (MEM.pmGpuSrcPx + (geometric ? MEM.pmGpuGeomSrcPx : 0)) + MEM.pmGpuFixed : 0;
  const pmParts: MemoryBreakdown = {
    images: imageCacheBase,
    sourceDepths: sourceDepthCacheBytes,
    // prior (full map) + the new map + its serialised copy (+ the halo-filtered copy)
    depthMaps: (geometric ? map : 0) + map + MEM.depthStoredPx * px + (p.haloFilter ? map : 0),
    colour: colourBytes,
    gpu: gpuBytes,
  };

  // ---- fusion: reference + neighbours resident, masks, colour, filtered-map output, points
  let nb = Math.max(1, Math.round(p.fusionMaxNeighbours));
  const fusionFixed = (n: number): MemoryBreakdown => ({
    depthMaps: (n + 2) * map + map /* store LRU: last map */,
    masks: Math.ceil(views * px * MEM.fusionMaskPx),
    colour: colourBytes,
    filteredOutput: map + MEM.depthStoredPx * px,
  });
  while (nb > 4 && sumBreakdown(fusionFixed(nb)) > 0.65 * share) nb--;
  const fixed = fusionFixed(nb);
  const pointsBytes = Math.max(0, share - sumBreakdown(fixed));
  const pointCap = Math.max(250_000, Math.floor(pointsBytes / MEM.fusedPointPeak));
  const fullExpected = MEM.fusedFraction * views * px;
  let step = p.fusionStep && p.fusionStep >= 1 ? Math.round(p.fusionStep) : 1;
  if (!(p.fusionStep && p.fusionStep >= 1) && fullExpected > pointCap) step = Math.min(6, Math.ceil(Math.sqrt(fullExpected / pointCap)));
  const expectedPoints = Math.round(fullExpected / (step * step));
  const fusionParts: MemoryBreakdown = { ...fixed, points: Math.min(pointCap, expectedPoints) * MEM.fusedPointPeak };

  // ---- mesh: TSDF blocks + scratch + two maps (TSDF cache) + colour
  const tsdfMaxBlocks = Math.max(1000, Math.min(p.tsdfMaxBlocks, Math.floor((0.25 * budget) / MEM.tsdfBlock)));
  const meshParts: MemoryBreakdown = {
    tsdf: tsdfMaxBlocks * MEM.tsdfBlock,
    scratch: MEM.tsdfScratchPx * px,
    depthMaps: 3 * map,
    colour: colourBytes,
    // marching cubes + simplification of ≈ 1–2 M triangles
    mesh: 120 * MB,
  };

  // The image LRU may use what the PatchMatch stage leaves below the larger of the other two
  // stages' peaks (it never raises the plan's peak): up to every view, so each view is decoded
  // and undistorted once per resolution instead of once per pass.
  let imageCacheBytes = imageCacheBase;
  const otherPeak = Math.max(sumBreakdown(fusionParts), sumBreakdown(meshParts));
  const pmSlack = otherPeak - sumBreakdown(pmParts);
  if (pmSlack > 0 && views * img > imageCacheBytes) {
    imageCacheBytes = Math.min(views * img, imageCacheBytes + pmSlack);
    pmParts.images = imageCacheBytes;
  }

  const stages: DenseStagePeak[] = [
    { stage: 'patchmatch', bytes: sumBreakdown(pmParts), parts: pmParts },
    { stage: 'fusion', bytes: sumBreakdown(fusionParts), parts: fusionParts },
    { stage: 'mesh', bytes: sumBreakdown(meshParts), parts: meshParts },
  ];
  const peakBytes = Math.max(...stages.map((s) => s.bytes)) + MEM.reserve + p.residentBytes;
  return {
    imageCacheBytes,
    sourceDepthCacheBytes,
    colourBytes,
    fusionMaxNeighbours: nb,
    fusionStep: step,
    fusionMaxPoints: pointCap,
    expectedPoints,
    tsdfMaxBlocks,
    shareBytes: share,
    stages,
    fits: peakBytes <= budget,
    peakBytes,
    storageBytes: Math.round(2.2 * views * px * MEM.depthStoredPx),
  };
}

// ---------------------------------------------------------------------------------------------
// Splat plan
// ---------------------------------------------------------------------------------------------

export interface SplatPlanInput {
  budgetMB: number;
  /** Engine data that stays resident during the stage (sparse model, results kept in memory). */
  residentBytes: number;
  /** Training views and their (largest) image size. */
  views: number;
  width: number;
  height: number;
  maxGaussians: number;
  shDegree: 0 | 1;
  /** Dense-cloud points read from the store for the initialisation (0 when the cloud is resident or absent). */
  loadedCloudPoints: number;
  /** Points the initialisation seeds Gaussians from (dense LOD + sparse points + shell). */
  initPoints: number;
  /** Trainer geometry supervision: views with a depth target, the target downscale, normal channels. */
  geometry?: { views: number; factor: number; normals?: boolean };
}

export interface SplatStagePeak {
  stage: 'prepare' | 'training';
  bytes: number;
  parts: MemoryBreakdown;
}

export interface SplatPlan {
  /** GPU buffers of the trainer (`SplatTrainer.memoryEstimate`, ±1 %), by buffer group. */
  gpuBytes: number;
  gpuParts: MemoryBreakdown;
  /** Largest single GPU buffer (must stay ≤ maxStorageBufferBindingSize: 256 MiB on iOS 26.0–26.1). */
  largestBuffer: number;
  /** Bytes available to the engine (budget − reserve − resident). */
  shareBytes: number;
  stages: SplatStagePeak[];
  /** Largest stage peak + reserve + resident. */
  peakBytes: number;
  fits: boolean;
}

/**
 * Peak memory of the splat stage. 'prepare' (before the trainer exists): the dense cloud read from
 * the store, its footprint LOD and the initial Gaussians; 'training': the trainer's GPU buffers
 * (on iOS they count against the same process) plus the largest host transient — decoding one
 * training view, a checkpoint (parameters read back as f32 + packed + the IndexedDB copy) or the
 * final model (read back + compact copy).
 */
export function planSplat(p: SplatPlanInput): SplatPlan {
  const budget = p.budgetMB * MB;
  const share = Math.max(64 * MB, budget - MEM.reserve - p.residentBytes);
  const px = Math.max(1, p.width * p.height);
  const geometry = p.geometry ? { views: Math.min(p.views, p.geometry.views), factor: p.geometry.factor, normals: p.geometry.normals } : undefined;
  const est = estimateSplatMemory({ maxGaussians: p.maxGaussians, shDegree: p.shDegree, width: p.width, height: p.height, views: p.views, geometry });
  // a SplatModel row: mean 3 + log-scale 3 + quaternion 4 + opacity 1 + SH (f32)
  const modelRow = 4 * (paramFloatsPerGaussian(p.shDegree) - 2);
  const prepare: MemoryBreakdown = {
    cloud: p.loadedCloudPoints * MEM.splatCloudLoadPoint,
    lod: p.loadedCloudPoints * MEM.splatLodPoint,
    init: p.initPoints * MEM.splatInitPoint,
    thumbnails: p.views * MEM.splatThumbBytes,
  };
  const transient = Math.max(
    MEM.splatDecodeFixed + MEM.splatDecodePx * px,
    // checkpoint: section readback + SplatModel + packed container + IndexedDB copy
    p.maxGaussians * modelRow * 3 + p.maxGaussians * 16,
    // final model: read back + compact container (f32 means, f16 rest) + IndexedDB copy
    p.maxGaussians * (modelRow + 2 * (modelRow / 2 + 6)),
  );
  const training: MemoryBreakdown = { ...prefixed('gpu.', est.breakdown), host: transient };
  const stages: SplatStagePeak[] = [
    { stage: 'prepare', bytes: sumBreakdown(prepare), parts: prepare },
    { stage: 'training', bytes: sumBreakdown(training), parts: training },
  ];
  const peakBytes = Math.max(...stages.map((s) => s.bytes)) + MEM.reserve + p.residentBytes;
  return { gpuBytes: est.total, gpuParts: est.breakdown, largestBuffer: est.largestBuffer, shareBytes: share, stages, peakBytes, fits: peakBytes <= budget };
}

function prefixed(prefix: string, b: MemoryBreakdown): MemoryBreakdown {
  const out: MemoryBreakdown = {};
  for (const [k, v] of Object.entries(b)) out[prefix + k] = v;
  return out;
}

export interface SplatFitRequest {
  views: number;
  /** Long side of the training images (px). */
  size: number;
  maxGaussians: number;
}

export interface SplatFitOptions {
  /** Training image size (undistorted pinhole) for a long side. */
  dims(size: number): { width: number; height: number };
  /** Never below these (defaults 16 views, 480 px, 50 000 Gaussians). */
  minViews?: number;
  minSize?: number;
  minGaussians?: number;
  /** Largest GPU buffer allowed (default 256 MiB, the iOS 26.0–26.1 binding limit). */
  maxBufferBytes?: number;
}

export interface SplatFit extends SplatFitRequest {
  width: number;
  height: number;
  plan: SplatPlan;
  /** What was lowered ("views 160 → 120", …); empty when the request fits. */
  degraded: string[];
}

/**
 * Lower the request until `planSplat` fits: first down to soft floors (half the views, half the
 * Gaussians, 0.75 × the size) taking from whichever of the training images and the Gaussians uses
 * more GPU memory, then to the hard floors in the same order. Returns the last plan even when it
 * still does not fit (the caller warns and tries anyway).
 */
export function fitSplatPlan(base: Omit<SplatPlanInput, 'views' | 'width' | 'height' | 'maxGaussians'>, want: SplatFitRequest, o: SplatFitOptions): SplatFit {
  const minViews = Math.min(want.views, Math.max(2, o.minViews ?? 16));
  const minSize = Math.min(want.size, Math.max(64, o.minSize ?? 480));
  const minG = Math.min(want.maxGaussians, Math.max(1000, o.minGaussians ?? 50_000));
  const maxBuf = o.maxBufferBytes ?? 256 * MB;
  const soft = { views: Math.max(minViews, Math.ceil(0.5 * want.views)), size: Math.max(minSize, Math.round(0.75 * want.size)), g: Math.max(minG, Math.round(0.5 * want.maxGaussians)) };
  let { views, size, maxGaussians } = want;
  const evaluate = (): { plan: SplatPlan; width: number; height: number } => {
    const d = o.dims(size);
    return { plan: planSplat({ ...base, views, width: d.width, height: d.height, maxGaussians }), ...d };
  };
  let cur = evaluate();
  const ok = (): boolean => cur.plan.fits && cur.plan.largestBuffer <= maxBuf;
  for (let guard = 0; guard < 200 && !ok(); guard++) {
    const imageBytes = cur.plan.gpuParts.images ?? 0;
    const gaussBytes = cur.plan.gpuBytes - imageBytes - (cur.plan.gpuParts.pixels ?? 0);
    const floors = views > soft.views || maxGaussians > soft.g || size > soft.size ? soft : { views: minViews, size: minSize, g: minG };
    if (imageBytes >= gaussBytes && views > floors.views) views = Math.max(floors.views, Math.floor(views * 0.85));
    else if (maxGaussians > floors.g) maxGaussians = Math.max(floors.g, Math.floor(maxGaussians * 0.85));
    else if (views > floors.views) views = Math.max(floors.views, Math.floor(views * 0.85));
    else if (size > floors.size) size = Math.max(floors.size, Math.round(size * 0.9));
    else break;
    cur = evaluate();
  }
  const degraded: string[] = [];
  if (views !== want.views) degraded.push(`views ${want.views} → ${views}`);
  if (maxGaussians !== want.maxGaussians) degraded.push(`Gaussians ${want.maxGaussians} → ${maxGaussians}`);
  if (size !== want.size) degraded.push(`image size ${want.size} → ${size} px`);
  return { views, size, maxGaussians, width: cur.width, height: cur.height, plan: cur.plan, degraded };
}
