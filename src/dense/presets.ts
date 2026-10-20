/**
 * Recommended dense-reconstruction settings per quality preset (`fast` / `balanced` / `max`),
 * tuned on the GT-pose dense harness (e2e/run-dense-gt.mjs; ARCHITECTURE §6 "Dense tuning
 * results"). Flat fields use the engine's names (src/engine/presets.ts merges them over its base
 * table); the nested blocks hold the full option sets for direct users of src/dense:
 *
 *   selectSourceViews(views, sparse, { numSources, ...viewSelection })
 *   PatchMatch photometric: { ...patchMatch, iterations }      (multiScale: see below)
 *   PatchMatch geometric × geometricPasses: { ...patchMatch, geometric: true, geometricIterations },
 *     prior + sources = the maps of the previous pass
 *   haloFilter ⇒ removeDepthHalos(map, view.image, view.cam) on the final maps, with the default
 *     options: halo vote + sky test (or fuseDepthMaps(..., { haloFilter: {} }) — not both: a
 *     second pass erodes edges). With a gravity-aligned model pass
 *     { skyUp: worldUpInCamera(view.pose.R) } per map: without it the sky test assumes upright frames.
 *   fuseDepthMaps(views, store, neighboursFromSelection(sel), { ...fusion, minViews })
 *   sorK > 0 ⇒ filterPointCloud(cloud, { statistical: { k: sorK, std: sorStd } })
 *
 * The defaults of the individual functions (`DEFAULT_FUSION_OPTIONS`, …) are unchanged for
 * compatibility (existing tests pin them); these presets are the tuned combination.
 *
 * Multi-scale (2026-10, real GPU): ½-resolution photometric (`iterations`) → full-resolution
 * prior-initialised photometric (`fullResIterations` 2) → geometric, run by the engine
 * (src/engine/dense.ts). GT-pose harness at 800 px / 40 views (fast), M4: quality-neutral
 * (object F 0.9992 → 0.9990, yard 0.6184 → 0.6190) at 1.2–1.5× less GPU time; ARCHITECTURE §6
 * "Dense on a real GPU". The cheaper kernel (`patchRadius` 6 / `patchStep` 3, `affineWindow`,
 * 3 proposals) halves the GPU time again but costs yard completeness (e2e yard-real F −0.75 pt),
 * so the presets keep the 11 px / 36-sample window.
 *
 * Evidence (480 px, GT poses; P / C at τ): fusion `minViews` 2 + halo filter + normal 15°
 * took object 0.9991 / 0.960 → 0.9996 / 0.988 and yard 0.9885 / 0.231 → 0.979 / 0.347;
 * `depthMarginMin` 0.5 (yard near ground) → yard 0.981 / 0.352; the sky test of the halo filter
 * (untextured top-connected regions without coherent depth, ARCHITECTURE §6 "Sky / low-texture
 * halo removal") → yard 0.985 / 0.352, object unchanged. SOR lowers completeness more
 * than it raises precision (off). 4 instead of 6 photometric iterations and 5 instead of 8
 * sources cost 0.2–0.8 pt of completeness (`fast` only); 2 instead of 3 geometric iterations
 * and a second geometric pass stay within ±0.2 pt with this fusion (`fast` uses 2 iterations,
 * `max` keeps one pass and spends its budget on image size and views). Image size and view
 * count are the main levers for large scenes (GT depth into fusion, yard: 480 → 960 px
 * C 0.461 → 0.526, 30 → 60 views C 0.461 → 0.570); a 480 px object is already saturated.
 */
import type { FusionOptions } from './fusion';
import type { PatchMatchOptions } from './types';
import type { ViewSelectionOptions } from './view-selection';

export type DenseQuality = 'fast' | 'balanced' | 'max';

export interface DensePreset {
  /** Long side of the undistorted stereo images (px). */
  maxImageSize: number;
  /** Maximum number of reference views (keyframes spread over the trajectory). */
  maxViews: number;
  /** Source views per reference (`selectSourceViews` numSources, ≤ 16). */
  numSources: number;
  /** Photometric PatchMatch iterations (full resolution). */
  iterations: number;
  /** Geometric-consistency passes after the photometric pass (0 = none). */
  geometricPasses: number;
  /** Iterations of each geometric pass. */
  geometricIterations: number;
  /**
   * Multi-scale photometric initialisation: photometric at ½ resolution for all views
   * (`iterations`), then a prior-initialised photometric pass at full resolution
   * (`fullResIterations`), then the geometric passes.
   */
  multiScale: boolean;
  /** Full-resolution prior-initialised iterations when `multiScale`. */
  fullResIterations: number;
  /** Long side used by the CPU PatchMatch fallback (no WebGPU). */
  cpuMaxImageSize: number;
  /** Fusion: minimum consistent views per point (reference included), object / large-scene mode. */
  fusionMinViewsObject: number;
  fusionMinViewsScene: number;
  /** Fusion neighbours per reference (bounds resident depth maps: (n + 2) × 20 B/px). */
  fusionMaxNeighbours: number;
  /** Apply `removeDepthHalos` (default options: halo vote + sky test) to the final depth maps before fusion. */
  haloFilter: boolean;
  /** Statistical outlier removal on the fused cloud: k neighbours (0 = off), σ multiplier. */
  sorK: number;
  sorStd: number;
  /** Kernel options for every PatchMatch pass. */
  patchMatch: Partial<PatchMatchOptions>;
  /** Extra `selectSourceViews` options. */
  viewSelection: Partial<ViewSelectionOptions>;
  /** `fuseDepthMaps` options (minViews = fusionMinViews*; no haloFilter — see `haloFilter`). */
  fusion: Partial<FusionOptions>;
}

const FUSION: Partial<FusionOptions> = Object.freeze({ minViews: 2, maxNormalErrorDeg: 15 });
const SELECTION: Partial<ViewSelectionOptions> = Object.freeze({ depthMarginMin: 0.5 });

export const DENSE_PRESETS: Readonly<Record<DenseQuality, Readonly<DensePreset>>> = Object.freeze({
  fast: Object.freeze({
    maxImageSize: 800,
    maxViews: 40,
    numSources: 5,
    iterations: 4,
    geometricPasses: 1,
    geometricIterations: 2,
    multiScale: true,
    fullResIterations: 2,
    cpuMaxImageSize: 320,
    fusionMinViewsObject: 2,
    fusionMinViewsScene: 2,
    fusionMaxNeighbours: 6,
    haloFilter: true,
    sorK: 0,
    sorStd: 3,
    patchMatch: Object.freeze({}),
    viewSelection: SELECTION,
    fusion: FUSION,
  }),
  balanced: Object.freeze({
    maxImageSize: 1280,
    maxViews: 80,
    numSources: 8,
    iterations: 6,
    geometricPasses: 1,
    geometricIterations: 3,
    multiScale: true,
    fullResIterations: 2,
    cpuMaxImageSize: 400,
    fusionMinViewsObject: 2,
    fusionMinViewsScene: 2,
    fusionMaxNeighbours: 8,
    haloFilter: true,
    sorK: 0,
    sorStd: 3,
    patchMatch: Object.freeze({}),
    viewSelection: SELECTION,
    fusion: FUSION,
  }),
  max: Object.freeze({
    maxImageSize: 1600,
    maxViews: 150,
    numSources: 8,
    iterations: 6,
    geometricPasses: 1,
    geometricIterations: 3,
    multiScale: true,
    fullResIterations: 2,
    cpuMaxImageSize: 480,
    fusionMinViewsObject: 2,
    fusionMinViewsScene: 2,
    fusionMaxNeighbours: 8,
    haloFilter: true,
    sorK: 0,
    sorStd: 3,
    patchMatch: Object.freeze({}),
    viewSelection: SELECTION,
    fusion: FUSION,
  }),
});
