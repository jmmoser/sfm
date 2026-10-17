/**
 * Quality presets (docs/ENGINE.md §2 table) and capture-mode parameters.
 *
 * | preset   | feature max side | max features | dense max side | PM iterations | geom passes |
 * |----------|------------------|--------------|----------------|---------------|-------------|
 * | fast     | 1024             | 2048         | 800            | 4             | 0           |
 * | balanced | 1600             | 4096         | 1280           | 6             | 1           |
 * | max      | 2048             | 8192         | 1600           | 8             | 2           |
 *
 * When `src/dense` exports `DENSE_PRESETS` (tuned from the e2e results by the dense owner),
 * its values override the dense part of this table (read defensively: only known numeric /
 * boolean fields are taken, under either the engine's or a few common alternative names).
 */
import * as denseModule from '../dense';
import type { KeyframeSelectorOptions } from '../sfm/keyframes';
import type { CaptureMode, QualityPreset } from '../workers/protocol';

export interface FeaturePresetConfig {
  /** SIFT working resolution (long side, px). */
  featureMaxSide: number;
  maxFeatures: number;
  /** Pairs matched per new keyframe during capture. */
  liveSequentialPairs: number;
  liveRetrievalPairs: number;
  liveSpatialPairs: number;
}

export interface DensePresetConfig {
  /** Long side of the undistorted stereo images (px). */
  maxImageSize: number;
  /** Photometric PatchMatch iterations. */
  iterations: number;
  /** Geometric-consistency passes after the photometric pass. */
  geometricPasses: number;
  /** Iterations of each geometric pass. */
  geometricIterations: number;
  /** Maximum number of reference views. */
  maxViews: number;
  /** Source views per reference. */
  numSources: number;
  /** Long side used by the CPU PatchMatch fallback (no WebGPU). */
  cpuMaxImageSize: number;
  /** Fusion: minimum consistent views per point (object / scene). */
  fusionMinViewsObject: number;
  fusionMinViewsScene: number;
  /** Fusion neighbours per reference (bounds the resident depth maps). */
  fusionMaxNeighbours: number;
  /** TSDF block budget (≈ 5.6 KB per block). */
  tsdfMaxBlocks: number;
  /** Statistical outlier removal on the fused cloud (k neighbours, σ multiplier; k = 0 disables). */
  sorK: number;
  sorStd: number;
  /** Fused clouds above this many points are voxel-downsampled. */
  maxPoints: number;
  /** Apply src/dense `removeDepthHalos` (when exported) to the final depth maps before fusion. Default true. */
  haloFilter?: boolean;
  /**
   * Fusion pixel step (every n-th reference pixel). Default: derived from the memory budget
   * (`planDense`): 1 unless the expected fused cloud would not fit (balanced / max at full size).
   */
  fusionStep?: number;
  /**
   * Option blocks of the tuned `DENSE_PRESETS` (src/dense/presets.ts) passed through to the
   * dense module: PatchMatch kernel options (every pass), `selectSourceViews` options (over the
   * capture mode's depth margins) and `fuseDepthMaps` options (minViews / maxNeighbours /
   * haloFilter are set by the engine).
   */
  patchMatch?: Record<string, unknown>;
  viewSelection?: Record<string, unknown>;
  fusion?: Record<string, unknown>;
}

export interface QualityPresetConfig extends FeaturePresetConfig {
  dense: DensePresetConfig;
}

export const QUALITY_PRESETS: Readonly<Record<QualityPreset, QualityPresetConfig>> = Object.freeze({
  fast: {
    featureMaxSide: 1024,
    maxFeatures: 2048,
    liveSequentialPairs: 6,
    liveRetrievalPairs: 2,
    liveSpatialPairs: 2,
    dense: {
      maxImageSize: 800,
      iterations: 4,
      geometricPasses: 0,
      geometricIterations: 3,
      maxViews: 40,
      numSources: 6,
      cpuMaxImageSize: 320,
      fusionMinViewsObject: 3,
      fusionMinViewsScene: 2,
      fusionMaxNeighbours: 6,
      tsdfMaxBlocks: 20_000,
      sorK: 8,
      sorStd: 2.5,
      maxPoints: 2_000_000,
    },
  },
  balanced: {
    featureMaxSide: 1600,
    maxFeatures: 4096,
    liveSequentialPairs: 6,
    liveRetrievalPairs: 3,
    liveSpatialPairs: 3,
    dense: {
      maxImageSize: 1280,
      iterations: 6,
      geometricPasses: 1,
      geometricIterations: 3,
      maxViews: 80,
      numSources: 8,
      cpuMaxImageSize: 400,
      fusionMinViewsObject: 3,
      fusionMinViewsScene: 2,
      fusionMaxNeighbours: 8,
      tsdfMaxBlocks: 30_000,
      sorK: 8,
      sorStd: 2.5,
      maxPoints: 4_000_000,
    },
  },
  max: {
    featureMaxSide: 2048,
    maxFeatures: 8192,
    liveSequentialPairs: 6,
    liveRetrievalPairs: 3,
    liveSpatialPairs: 3,
    dense: {
      maxImageSize: 1600,
      iterations: 8,
      geometricPasses: 2,
      geometricIterations: 3,
      maxViews: 150,
      numSources: 8,
      cpuMaxImageSize: 480,
      fusionMinViewsObject: 3,
      fusionMinViewsScene: 2,
      fusionMaxNeighbours: 8,
      tsdfMaxBlocks: 40_000,
      sorK: 8,
      sorStd: 2.5,
      maxPoints: 6_000_000,
    },
  },
});

/** Alternative field names accepted from `DENSE_PRESETS` entries. */
const DENSE_ALIASES: Record<string, keyof DensePresetConfig> = {
  maxImageSize: 'maxImageSize',
  denseMaxSide: 'maxImageSize',
  maxSize: 'maxImageSize',
  iterations: 'iterations',
  pmIterations: 'iterations',
  geometricPasses: 'geometricPasses',
  geomPasses: 'geometricPasses',
  geometricIterations: 'geometricIterations',
  maxViews: 'maxViews',
  numSources: 'numSources',
  sources: 'numSources',
  cpuMaxImageSize: 'cpuMaxImageSize',
  fusionMinViewsObject: 'fusionMinViewsObject',
  fusionMinViewsScene: 'fusionMinViewsScene',
  fusionMaxNeighbours: 'fusionMaxNeighbours',
  maxNeighbours: 'fusionMaxNeighbours',
  tsdfMaxBlocks: 'tsdfMaxBlocks',
  maxBlocks: 'tsdfMaxBlocks',
  sorK: 'sorK',
  sorStd: 'sorStd',
  maxPoints: 'maxPoints',
};

/** Numeric fields of a `DENSE_PRESETS[quality]` entry (if src/dense exports one), mapped to our names. */
export function externalDensePreset(quality: QualityPreset): Partial<DensePresetConfig> {
  // computed key: the export may not exist (yet); a literal would make the bundler warn
  const ext = (denseModule as unknown as Record<string, unknown>)[['DENSE', 'PRESETS'].join('_')];
  if (!ext || typeof ext !== 'object') return {};
  return parseDensePresetEntry((ext as Record<string, unknown>)[quality]);
}

/**
 * Engine fields of one `DENSE_PRESETS` entry: the option blocks (`patchMatch`, `viewSelection`,
 * `fusion`), the flat numeric fields (engine names or aliases) and a few values derivable from the
 * nested blocks (`fusion.minViews`, `fusion.maxNeighbours`, `tsdf.maxBlocks`, `patchMatch.*`). The
 * flat fields are authoritative (src/dense/presets.ts: "minViews = fusionMinViews*"): a nested
 * value only fills a field the entry does not set flat, whatever the key order.
 */
export function parseDensePresetEntry(entry: unknown): Partial<DensePresetConfig> {
  if (!entry || typeof entry !== 'object') return {};
  const out: Partial<DensePresetConfig> = {};
  const nested: Partial<DensePresetConfig> = {};
  // option blocks for the dense module (plain JSON-like values only)
  for (const k of ['patchMatch', 'viewSelection', 'fusion'] as const) {
    const b = (entry as Record<string, unknown>)[k];
    if (!b || typeof b !== 'object' || Array.isArray(b)) continue;
    const block: Record<string, unknown> = {};
    for (const [bk, bv] of Object.entries(b as Record<string, unknown>)) {
      if ((typeof bv === 'number' && Number.isFinite(bv)) || typeof bv === 'boolean' || bv === null) block[bk] = bv;
      else if (bv && typeof bv === 'object' && !Array.isArray(bv)) block[bk] = { ...(bv as Record<string, unknown>) };
    }
    out[k] = block;
  }
  const visit = (obj: Record<string, unknown>, depth: number): void => {
    const dst = depth === 0 ? out : nested;
    for (const [k, v] of Object.entries(obj)) {
      if (typeof v === 'number' && Number.isFinite(v) && DENSE_ALIASES[k]) (dst as Record<string, number>)[DENSE_ALIASES[k]] = v;
      else if (typeof v === 'boolean' && k === 'haloFilter') dst.haloFilter = v;
      else if (typeof v === 'boolean' && k === 'geometric' && !v) dst.geometricPasses = 0;
      else if (v && typeof v === 'object' && !Array.isArray(v) && depth < 2) {
        // nested blocks like { patchMatch: { iterations }, fusion: { minViews } }
        const sub = v as Record<string, unknown>;
        if (k === 'fusion' && typeof sub.minViews === 'number') {
          nested.fusionMinViewsObject = sub.minViews;
          nested.fusionMinViewsScene = sub.minViews;
        }
        if (k === 'fusion' && typeof sub.maxNeighbours === 'number') nested.fusionMaxNeighbours = sub.maxNeighbours;
        if (k === 'tsdf' && typeof sub.maxBlocks === 'number') nested.tsdfMaxBlocks = sub.maxBlocks;
        if (k === 'patchMatch' || k === 'patchmatch' || k === 'pm') visit(sub, depth + 1);
      }
    }
  };
  visit(entry as Record<string, unknown>, 0);
  for (const [k, v] of Object.entries(nested)) if ((out as Record<string, unknown>)[k] === undefined) (out as Record<string, unknown>)[k] = v;
  return out;
}

/** The dense preset: engine table ← DENSE_PRESETS (if exported) ← explicit overrides. */
export function resolveDensePreset(quality: QualityPreset, overrides: Partial<DensePresetConfig> = {}): DensePresetConfig {
  return { ...QUALITY_PRESETS[quality].dense, ...externalDensePreset(quality), ...stripUndefined(overrides) };
}

function stripUndefined<T extends object>(o: T): Partial<T> {
  const out: Partial<T> = {};
  for (const [k, v] of Object.entries(o)) if (v !== undefined) (out as Record<string, unknown>)[k] = v;
  return out;
}

/** Concrete capture mode (after 'auto' resolved). */
export type ResolvedMode = 'object' | 'scene';

export interface ModeConfig {
  /** KeyframeSelector mode (its parallax trigger). */
  keyframeMode: CaptureMode;
  keyframe: Partial<KeyframeSelectorOptions>;
  /** TSDF: maxDepth = factor × median scene depth (object mode; Infinity = off). */
  tsdfMaxDepthFactor: number;
  /** View selection depth margins (× 1st / 99th percentile). */
  depthMarginMin: number;
  depthMarginMax: number;
  /** TSDF auto voxel size in pixel footprints at the median depth. */
  tsdfAutoVoxelPixels: number;
}

export function modeConfig(mode: CaptureMode): ModeConfig {
  switch (mode) {
    case 'object':
      return { keyframeMode: 'object', keyframe: {}, tsdfMaxDepthFactor: 2, depthMarginMin: 0.8, depthMarginMax: 1.25, tsdfAutoVoxelPixels: 2 };
    case 'scene':
      return { keyframeMode: 'scene', keyframe: {}, tsdfMaxDepthFactor: Infinity, depthMarginMin: 0.7, depthMarginMax: 1.5, tsdfAutoVoxelPixels: 2 };
    default:
      return { keyframeMode: 'auto', keyframe: {}, tsdfMaxDepthFactor: Infinity, depthMarginMin: 0.75, depthMarginMax: 1.4, tsdfAutoVoxelPixels: 2 };
  }
}

/** Default numbers of the engine (not preset dependent). */
export const ENGINE_DEFAULTS = Object.freeze({
  settleMs: 500,
  guidedRadiusFrac: 0.08,
  overlayMaxPoints: 300,
  liveMaxPoints: 50_000,
  trackingMaxIterations: 400,
  /** Registered images needed before refine frees the principal point in its final BA. */
  refinePrincipalPointMinImages: 20,
  /** Keyframes queued / being mapped before a keyframe frame's response waits (background mapping). */
  maxMappingBacklog: 2,
  vocabularyTrainAfter: 25,
  vocabularyWords: 256,
  refineVocabularyWords: 512,
  refineSequential: 10,
  refineRetrieval: 15,
  refineSpatial: 10,
  memoryBudgetMB: 1200,
  /**
   * Capture state (features, pairs, graph, live model, …) cap as a fraction of the memory budget
   * minus the page / JS reserve (250 MB) and the feature GPU caches (64 MB): refine peaks at ≈ 1.8–1.9×
   * the capture state (more pairs, a second reconstruction + the working copy, global BA:
   * measured on the e2e object capture), so 0.45 keeps the refine peak ≈ 0.85 of what is left.
   * 1200 MB → ≈ 400 MB (≈ 280 keyframes at balanced on 1920×1440), 850 MB → ≈ 240 MB.
   */
  captureMemoryFraction: 0.45,
  /** Two-view pairs with fewer inliers are not stored. */
  minPairInliers: 15,
  /** Long side of the per-frame colour / sharpness copy. */
  smallMaxSide: 640,
  /** Registered keyframes needed before the R_cam_device hand-eye check. */
  handEyeMinKeyframes: 6,
});
