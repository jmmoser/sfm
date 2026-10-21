/**
 * Dense reconstruction data types (ARCHITECTURE §4.6).
 *
 * Conventions: OpenCV camera frame (x right, y down, z forward), world-to-camera
 * poses (x_cam = R X + t), pixel centres at integer coordinates.
 */
import type { CameraIntrinsics, Pose, RGBAImage } from '../core/types';

/** One undistorted view used by dense stereo. */
export interface DenseView {
  id: number;
  /** Pinhole intrinsics (k1 = k2 = 0) of `image`. */
  cam: CameraIntrinsics;
  /** World-to-camera pose. */
  pose: Pose;
  /** Undistorted image at cam.width × cam.height. */
  image: RGBAImage;
}

/** Per-pixel depth / normal / cost map of one view (row-major, width × height). */
export interface DepthMap {
  width: number;
  height: number;
  /** z-depth in the camera frame; 0 = invalid. */
  depth: Float32Array;
  /** Unit normal per pixel (3 floats), camera frame, facing the camera (n · X < 0). 0,0,0 when invalid. */
  normal: Float32Array;
  /** Aggregated photometric cost (1 − bilateral NCC), 0..2. */
  cost: Float32Array;
}

/** Async key-value store of depth maps (memory, IndexedDB, …). */
export interface DepthMapStore {
  get(id: number): Promise<DepthMap | null>;
  put(id: number, d: DepthMap): Promise<void>;
}

/** Simple in-memory store. */
export class MemoryDepthMapStore implements DepthMapStore {
  private readonly maps = new Map<number, DepthMap>();

  async get(id: number): Promise<DepthMap | null> {
    return this.maps.get(id) ?? null;
  }

  async put(id: number, d: DepthMap): Promise<void> {
    this.maps.set(id, d);
  }

  has(id: number): boolean {
    return this.maps.has(id);
  }

  delete(id: number): boolean {
    return this.maps.delete(id);
  }

  ids(): number[] {
    return [...this.maps.keys()].sort((a, b) => a - b);
  }

  clear(): void {
    this.maps.clear();
  }
}

/** Optional prior + source depth maps for the geometric-consistency (ACMM) pass. */
export interface GeometricInput {
  /**
   * Depth map of the reference view used as initialisation (any resolution; nearest
   * resampling per §2, planes re-intersected with the target pixel's ray).
   */
  prior: DepthMap;
  /** Depth maps of the source views, same order as `sources` (any resolution each; a 0×0 map = unavailable). */
  sourceDepths: DepthMap[];
}

/**
 * PatchMatch MVS options (ACMH view selection + ACMM geometric consistency).
 * See `DEFAULT_PATCHMATCH_OPTIONS` for defaults.
 */
export interface PatchMatchOptions {
  /** Photometric iterations (one iteration = red pass + black pass). Default 6. */
  iterations: number;
  /** Matching window radius in pixels. Default 5. */
  patchRadius: number;
  /** Sampling step inside the window (offsets −r, −r+step, …, ≤ r). Default 2. */
  patchStep: number;
  /**
   * Warp the window into a source with the homography's affine approximation at the window
   * centre (its Jacobian) instead of a projective divide per sample: two multiply-adds per
   * sample. The error is second order in the window offset (well below 0.1 px for a 13 px window
   * except on grazing planes). Default false; `DENSE_PRESETS` keep it off too: with the cheaper
   * 13 px / 25-sample window and 3 proposals it halves GPU time but costs ≈ 1 pt of yard
   * completeness (docs/research/dense-tuning.md "Dense on a real GPU").
   */
  affineWindow: boolean;
  /**
   * Number of refinement proposals evaluated per pixel per iteration (1..6), in
   * this order: (perturbed depth, current normal), (current depth, perturbed normal),
   * (random depth, random normal), (perturbed both), (random depth, current normal),
   * (current depth, random normal). Default 6.
   */
  numSamplesPerIter?: number;
  /**
   * Geometric-consistency (ACMM) mode. Only meaningful when the `geometric`
   * argument (prior + source depth maps) is passed to compute(); when that
   * argument is present, geometric mode is used unless this is explicitly false.
   * With the argument AND `geometric: false`, a PHOTOMETRIC pass runs that is
   * initialised from `prior` (sourceDepths are ignored): the multi-scale refinement
   * step (coarse photometric → this at full resolution with `iterations` ≈ 2 →
   * geometric pass with full-resolution maps). Do NOT feed coarse maps as
   * `sourceDepths` of a full-resolution geometric pass: their reprojection error
   * dominates the cost (measured 62 % vs 88 % within 1 % on the near scene).
   */
  geometric: boolean;
  /** Iterations of the geometric pass (replaces `iterations` in geometric mode). Default 3. */
  geometricIterations: number;
  /** Weight λ of the forward-backward reprojection error term. Default 0.2. */
  geomLambda: number;
  /** Truncation τ (pixels) of the reprojection error. Default 3. */
  geomMaxError: number;
  /** Bilateral weight spatial sigma (pixels): w ∝ exp(−dist / (2σs²)). Default 5. */
  sigmaSpatial: number;
  /** Bilateral weight colour sigma (0..255 units): w ∝ exp(−ΔC / (2σc²)), ΔC = mean |ΔR|,|ΔG|,|ΔB| to the centre pixel. Default 3. */
  sigmaColor: number;
  /** Pixels whose final photometric cost exceeds this are invalidated (NCC < 1 − maxCost). Default 0.5. */
  maxCost: number;
  /** Views used for the initial cost and as fallback when view selection selects none. Default 3. */
  topK: number;
  /** ACMH "good" cost threshold at iteration t: goodCost · exp(−t²/goodCostDecay). Default 0.8. */
  goodCost: number;
  /** α of the good-cost schedule. Default 90. */
  goodCostDecay: number;
  /** ACMH "bad" cost threshold. Default 1.2. */
  badCost: number;
  /** Initial relative depth perturbation (fraction of depth). Default 0.05. */
  perturbation: number;
  /** Initial normal perturbation (radians, per component). Default 0.15. */
  normalPerturbation: number;
  /** Multiplicative decay of the perturbation per iteration. Default 0.5. */
  perturbationDecay: number;
  /** Lower bound of the relative depth perturbation. Default 0.002. */
  minPerturbation: number;
  /** Starting relative depth perturbation when initialised from a prior (geometric or prior-initialised photometric pass). Default 0.01. */
  geometricPerturbation: number;
  /** Minimum weighted variance of a 0..1 grey patch (textureless rejection). Default (0.75/255)². */
  minVariance: number;
  /** RNG seed. Default 1. */
  seed: number;
  /** GPU only: max (pixels × source views) per command buffer (hard cap / first band). Default 1.5M. */
  maxPixelViewsPerSubmit: number;
  /**
   * GPU only: target GPU time per command buffer in ms; band heights adapt to it
   * (ARCHITECTURE §2.1: ≤ 30 ms in processing mode, 2–8 ms while the camera preview
   * animates). Ignored on software adapters. Default 20.
   */
  targetSubmitMs: number;
}

export const MAX_PATCHMATCH_SOURCES = 16;

export const DEFAULT_PATCHMATCH_OPTIONS: Readonly<PatchMatchOptions> = Object.freeze({
  iterations: 6,
  patchRadius: 5,
  patchStep: 2,
  affineWindow: false,
  numSamplesPerIter: 6,
  geometric: false,
  geometricIterations: 3,
  geomLambda: 0.2,
  geomMaxError: 3,
  sigmaSpatial: 5,
  sigmaColor: 3,
  maxCost: 0.5,
  topK: 3,
  goodCost: 0.8,
  goodCostDecay: 90,
  badCost: 1.2,
  perturbation: 0.05,
  normalPerturbation: 0.15,
  perturbationDecay: 0.5,
  minPerturbation: 0.002,
  geometricPerturbation: 0.01,
  minVariance: (0.75 / 255) * (0.75 / 255),
  seed: 1,
  maxPixelViewsPerSubmit: 1_500_000,
  targetSubmitMs: 20,
});

/** Depth search range (z-depth, same units as poses). */
export interface DepthRange {
  min: number;
  max: number;
}
