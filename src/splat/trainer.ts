/**
 * On-device 3D Gaussian Splatting trainer (WebGPU). Everything per iteration stays on the GPU:
 *
 *   forward (forward.ts) → target (view image box-downscaled to the current level)
 *   → loss: per-view affine colour correction, (1 − λ)·L1 + λ·D-SSIM, ∂loss/∂C, colour-matrix
 *     Adam step [+ geometry: inverse-depth / normal L1 against the dense stage's depth maps,
 *     `geometry`] → rasteriser backward (no per-pixel atomics; one float CAS add per
 *     splat × tile × component) → projection backward fused with Adam (visible Gaussians only)
 *     + MCMC regularisers, shape regularisers (`flatten`, `anisotropy`) and position noise.
 *
 * Every `mcmc.refineEvery` steps (between `start` and `stop`) the host reads the opacities
 * back once and plans an MCMC relocation / growth step (mcmc.ts) that the GPU applies.
 * Training views are seen in random order (a fresh permutation per epoch), at a coarse-to-fine
 * resolution schedule (default ¼ for the first 10 % of the iterations, ½ until 30 %, then full).
 *
 * Driving it: `create` compiles every pipeline (stage start), `init` uploads the initial model
 * and the views (images now or one by one with `setViewImage`, so the host never holds them all),
 * then `step(n)` runs n iterations — one command buffer each, awaited before the next (one heavy
 * command buffer in flight, iOS rule) — and can be called repeatedly in time slices from a worker.
 * `render`, `readModel` / `exportPLY` (ply.ts), `memory()`.
 *
 * Dropped work (iOS may silently drop command buffers submitted while the page is hidden): every
 * iteration stamps its stats slot with a submission counter and every MCMC relocation copies its
 * own stamp next to them; `step` checks the stamps of its iterations and throws
 * `SplatWorkDroppedError` when one is missing (the caller restores a checkpoint: a dropped
 * relocation leaves grown slots uninitialised). Every readback copies its own stamp behind the
 * data: a dropped one throws `SplatWorkDroppedError` instead of returning stale staging contents
 * (`readState` / `readModel` / `readMoments` / `normaliseAppearance` can then simply be called
 * again). Checkpoints: `readState` + `readMoments` + `readModel`, restored by
 * `init(params, views)` + `restore(state, moments)`.
 */
import type { CameraIntrinsics, Pose, RGBAImage } from '../core/types';
import type { GPUContext } from '../gpu/device';
import { Rng } from '../math/random';
import { compileForwardPipelines, getForwardPipelines, SplatForward, type ForwardPipelines } from './forward';
import { PassProfiler, bindGroup, compileAll, makeBuffer } from './gpu-util';
import { packParams, paramOffsets, unpackParams } from './layout';
import { planMcmc } from './mcmc';
import { encodeSplatPLY } from './ply';
import type { SplatImage } from './renderer';
import {
  SH_C0,
  allocSplatModel,
  downscaleSplatCamera,
  estimateSplatMemory,
  shCoeffCount,
  splatCamera,
  TILE_SIZE,
  type ShDegree,
  type SplatCamera,
  type SplatMemoryEstimate,
  type SplatModel,
} from './types';
import { OPT_BYTES, backwardWGSL, relocateWGSL } from './wgsl/backward';
import { COLOUR_ADAM_WGSL, COLOUR_CFG_BYTES, COLOUR_FLOATS, GEO_CFG_BYTES, GEO_LOSS_WGSL, LOSS_CFG_BYTES, SSIM_BWD_WGSL, SSIM_FWD_WGSL, TARGET_WGSL } from './wgsl/loss';
import { grad2dStride, shaderKey, type SplatShaderConfig } from './wgsl/common';
import { rasterBackwardWGSL } from './wgsl/raster';
import type { DepthTarget } from './depth-target';

/**
 * A training image with its pinhole camera (k1 = k2 = 0) and pose — `DenseView` compatible. A
 * null / absent image is uploaded later with `SplatTrainer.setViewImage` (before `step`).
 */
export interface SplatTrainingView {
  id: number;
  cam: CameraIntrinsics;
  pose: Pose;
  image?: RGBAImage | null;
}

/**
 * GPU work of `step` was not executed (command buffers dropped while the page was hidden, or a
 * stale readback). Transient: the trainer state is inconsistent — restore a checkpoint (or start
 * over) and continue.
 */
export class SplatWorkDroppedError extends Error {
  readonly transient = true;
  constructor(
    message: string,
    /** Iterations of the checked window whose stamp was missing. */
    readonly iterations: number,
    /** The MCMC relocation of this call was not executed. */
    readonly relocation: boolean,
  ) {
    super(message);
    this.name = 'GPUWorkDroppedError';
  }
}

/**
 * Small part of a training checkpoint (the rest: `readModel()` parameters and `readMoments()`
 * Adam moments, in the `SplatModel` layout).
 */
export interface SplatTrainerState {
  iteration: number;
  count: number;
  background: [number, number, number];
  /** Per-view colour-correction state (`COLOUR_FLOATS` per view: 3×4 matrix, Adam m, v, step). */
  colour: Float32Array;
  mcmc: { refinements: number; relocated: number; added: number };
  /** Largest intersection count seen (pre-sizes the reservation after a restore). */
  maxIntersections: number;
}

export interface SplatLearningRates {
  /** Mean learning rate at the start / end (× scene extent; exponential decay in between). */
  mean: number;
  meanFinal: number;
  scale: number;
  quat: number;
  opacity: number;
  sh0: number;
  shN: number;
}

export const DEFAULT_SPLAT_LR: SplatLearningRates = {
  mean: 1.6e-4,
  meanFinal: 1.6e-6,
  scale: 5e-3,
  quat: 1e-3,
  opacity: 5e-2,
  sh0: 2.5e-3,
  shN: 2.5e-3 / 20,
};

export interface SplatMcmcOptions {
  refineEvery: number;
  /** First / last refinement step (iterations; default 10 % / 75 % of `iterations`). */
  start: number;
  stop: number;
  growth: number;
  minOpacity: number;
  /** Position-noise learning-rate multiplier (paper: 5e5). */
  noiseLr: number;
  opacityReg: number;
  /** Scale regulariser (applied to scale / scene extent, so it is unit-free). */
  scaleReg: number;
}

export interface SplatColourCorrection {
  lr: number;
  /** L2 pull of each view's matrix towards [I | 0]. */
  reg: number;
}

/**
 * Geometry supervision from the dense stage's depth maps (`setViewDepth`; DN-Splatter-style):
 * L = (1 / pixels) Σ w (λ_d |D̂·z_t − 1| + λ_n ‖N̂ − n_t‖₁) over the target pixels, D̂ / N̂ = rendered
 * alpha-blended inverse depth / normal (smallest-scale axis) divided by the accumulated alpha,
 * z_t / n_t / w the target depth, normal and weight (depth-target.ts). Weights decay exponentially
 * from `[0]` to `[1]` over the first `until` of the run (linearly when one end is 0), then 0 and
 * the trainer switches to the plain kernels. Defaults (measured, ARCHITECTURE "Geometry
 * supervision …"): inverse depth only, 0.5 → 0.05 over the first 30 %; the normal term
 * (e.g. 0.1 → 0.01) did not add anything on the test scenes and costs more.
 */
export interface SplatGeometryOptions {
  depth: [number, number];
  normal: [number, number];
  /** Fraction of the iterations the losses run for (default 0.3: the ¼ / ½-resolution part). */
  until: number;
  /** Targets at 1/factor of the training-view size (default 2; memory 8 B per target pixel). */
  factor: number;
  /**
   * Target weight × |cos θ|^grazing (θ: target normal vs the pixel's ray; depth-target.ts), so
   * surfaces the view sees at grazing angles are supervised less. Default 0 (off). Applied when
   * the engine builds the targets.
   */
  grazing: number;
}

export const DEFAULT_SPLAT_GEOMETRY: SplatGeometryOptions = { depth: [0.5, 0.05], normal: [0, 0], until: 0.3, factor: 2, grazing: 0 };

/**
 * Shader configuration of a geometry option: full channels (inverse depth + normal), 'depth' when
 * both normal weights are 0 (cheaper kernels), false without geometry.
 */
export function geometryMode(g: Partial<SplatGeometryOptions> | false | undefined | null): boolean | 'depth' {
  if (!g) return false;
  const n = g.normal ?? DEFAULT_SPLAT_GEOMETRY.normal;
  return n[0] > 0 || n[1] > 0 ? true : 'depth';
}

export interface SplatTrainerOptions {
  /** Hard cap on the number of Gaussians (memory plan). */
  maxGaussians: number;
  /** Planned number of iterations (learning-rate, resolution and MCMC schedules). */
  iterations: number;
  shDegree?: ShDegree;
  antialiased?: boolean;
  background?: [number, number, number];
  /** SSIM weight λ (default 0.2). */
  ssimLambda?: number;
  lr?: Partial<SplatLearningRates>;
  /** Scene extent (default 1.1 × largest camera distance from the cameras' centroid). */
  sceneExtent?: number;
  /** Resolution levels: downscale factor (1, 2, 4, …) until a fraction of the iterations. */
  resolution?: { factor: number; until: number }[];
  mcmc?: Partial<SplatMcmcOptions> | false;
  /** Per-view affine colour correction (default on). */
  colourCorrection?: Partial<SplatColourCorrection> | false;
  /** Depth / normal supervision (default off; targets through `setViewDepth`). */
  geometry?: Partial<SplatGeometryOptions> | false;
  /**
   * Flatten regulariser: weight of the smallest scale's L1 (÷ scene extent, mean over the
   * Gaussians) — surfel-like discs. Default 0.
   */
  flatten?: number;
  /** Anisotropy cap: penalise ln s_max − ln s_mid above ln(maxRatio) (mean over the Gaussians). Default off. */
  anisotropy?: { maxRatio: number; weight: number } | false;
  seed?: number;
  /** Tile intersections reserved per Gaussian at the start (default 12; grows on demand). */
  intersectionsPerGaussian?: number;
  /** Record per-pass GPU times (timestamp queries) for every step. */
  profile?: boolean;
  /**
   * Iteration command buffers allowed in flight (default 1, the iOS rule; 2 hides the host
   * encode / completion latency on desktop GPUs). Profiling forces 1.
   */
  inFlight?: 1 | 2;
}

export interface SplatStepResult {
  iteration: number;
  count: number;
  /** Loss terms of the last iteration of the call. */
  loss: number;
  l1: number;
  ssim: number;
  ms: number;
  iterationsPerSecond: number;
  /** MCMC refinements done in this call. */
  refinements: number;
  /** Intersection overflow seen (an iteration dropped tile intersections). */
  overflow: boolean;
}

export interface SplatTrainerStats {
  iteration: number;
  count: number;
  intersectionCapacity: number;
  /** Largest intersection count seen. */
  maxIntersections: number;
  /** Accumulated GPU ms per pass (profile mode) and the iterations it covers. */
  gpuMs: Record<string, number>;
  profiledIterations: number;
  mcmc: { refinements: number; relocated: number; added: number };
}

interface TrainPipelines {
  fwd: ForwardPipelines;
  /** Geometry loss (only with the geometry channels). */
  geoLoss: GPUComputePipeline | null;
  rasterBwd: GPUComputePipeline;
  adam: GPUComputePipeline;
  grads: GPUComputePipeline;
  target: GPUComputePipeline;
  ssimFwd: GPUComputePipeline;
  ssimBwd: GPUComputePipeline;
  colourAdam: GPUComputePipeline;
  relocCopy: GPUComputePipeline;
  relocUpdate: GPUComputePipeline;
}

/** Ring of per-iteration loss / stamp slots; slot STAT_SLOTS holds the last relocation's stamp. */
const STAT_SLOTS = 64;
/** Rasteriser backward: pixel groups (32 pixels each) per workgroup, splats per thread step (wgsl/raster.ts). */
export const BWD_GROUPS = 2;
export const BWD_PER_STEP = 1;

const pipelineCache = new WeakMap<GPUDevice, Map<string, Promise<TrainPipelines>>>();

async function compileTrainPipelines(device: GPUDevice, cfg: SplatShaderConfig): Promise<TrainPipelines> {
  const lost = device.lost;
  const [fwd, p, g] = await Promise.all([
    getForwardPipelines(device, cfg),
    compileAll(
      device,
      lost,
      {
        rasterBwd: rasterBackwardWGSL(cfg, BWD_GROUPS, BWD_PER_STEP),
        adam: backwardWGSL(cfg, 'adam'),
        grads: backwardWGSL(cfg, 'grads'),
        target: TARGET_WGSL,
        ssimFwd: SSIM_FWD_WGSL,
        ssimBwd: SSIM_BWD_WGSL,
        colourAdam: COLOUR_ADAM_WGSL,
        relocCopy: relocateWGSL(cfg, 'copy'),
        relocUpdate: relocateWGSL(cfg, 'update'),
      },
      'splat-train',
    ),
    cfg.geometry ? compileAll(device, lost, { geoLoss: GEO_LOSS_WGSL }, 'splat-train') : Promise.resolve(null),
  ]);
  return { fwd, geoLoss: g?.geoLoss ?? null, ...p };
}

/**
 * The plain (no geometry channels) kernels with the geometry run's projection layout, sharing its
 * layout-independent pipelines (loss, MCMC relocation, sort): used after the geometry losses end.
 */
async function compilePostGeometryPipelines(device: GPUDevice, cfg: SplatShaderConfig, geo: TrainPipelines): Promise<TrainPipelines> {
  const lost = device.lost;
  const [fwd, p] = await Promise.all([
    compileForwardPipelines(device, lost, cfg, geo.fwd.sort),
    compileAll(device, lost, { rasterBwd: rasterBackwardWGSL(cfg, BWD_GROUPS, BWD_PER_STEP), adam: backwardWGSL(cfg, 'adam'), grads: backwardWGSL(cfg, 'grads') }, 'splat-train-post'),
  ]);
  return { ...geo, fwd, geoLoss: null, ...p };
}

/**
 * Valid target pixels of an RGBA8 training image at each level `factor` (TARGET_WGSL: the f × f
 * box's minimum alpha ≥ 0.5, i.e. every alpha byte ≥ 128, over ⌊W / f⌋ × ⌊H / f⌋ pixels); null
 * when no alpha byte is below 128 (every pixel of every level is valid).
 */
function validTargetPixels(img: RGBAImage, factors: number[]): Map<number, number> | null {
  const d = img.data;
  const W = img.width;
  const H = img.height;
  let masked = false;
  for (let i = 3; i < 4 * W * H && !masked; i += 4) masked = d[i] < 128;
  if (!masked) return null;
  const out = new Map<number, number>();
  for (const f of new Set(factors)) {
    const w = Math.max(1, Math.floor(W / f));
    const h = Math.max(1, Math.floor(H / f));
    let n = 0;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let ok = true;
        for (let dy = 0; dy < f && ok; dy++) for (let dx = 0; dx < f && ok; dx++) ok = d[4 * ((y * f + dy) * W + x * f + dx) + 3] >= 128;
        if (ok) n++;
      }
    out.set(f, n);
  }
  return out;
}

export class SplatTrainer {
  readonly shDegree: ShDegree;
  readonly antialiased: boolean;
  readonly capacity: number;
  private readonly lr: SplatLearningRates;
  private readonly mcmc: SplatMcmcOptions | null;
  private readonly cc: SplatColourCorrection | null;
  private readonly geo: SplatGeometryOptions | null;
  /** Geometry run with `until` < 1: the geometry pipelines and the plain ones of the same layout. */
  private geoPipes: TrainPipelines | null = null;
  private postPipes: TrainPipelines | null = null;
  private readonly lambda: number;
  private background: [number, number, number];
  private readonly levels: { factor: number; until: number }[];
  private rng: Rng;

  private fwd!: SplatForward;
  private params!: GPUBuffer;
  private adamM!: GPUBuffer;
  private adamV!: GPUBuffer;
  private grad2d!: GPUBuffer;
  private target!: GPUBuffer;
  private pqr!: GPUBuffer;
  private dLdC!: GPUBuffer;
  private lossPartial!: GPUBuffer;
  private colourPartial!: GPUBuffer;
  private colour!: GPUBuffer;
  private statsBuf!: GPUBuffer;
  private optU!: GPUBuffer;
  private lossU!: GPUBuffer;
  private colourU!: GPUBuffer;
  private relocU!: GPUBuffer;
  private relocList: GPUBuffer | null = null;
  private images: GPUBuffer[] = [];
  /** Geometry supervision: uniform, per-workgroup sums, per-view targets (null = none), bind groups. */
  private geoU: GPUBuffer | null = null;
  private geoPartial: GPUBuffer | null = null;
  private geoDummy: GPUBuffer | null = null;
  private depthTargets: (GPUBuffer | null)[] = [];
  private depthDims: ({ width: number; height: number; factor: number } | null)[] = [];
  private bgGeo: GPUBindGroup[] = [];
  private staging: GPUBuffer | null = null;
  private gradsOut: GPUBuffer | null = null;
  private profiler: PassProfiler | null = null;

  private bgTarget: GPUBindGroup[] = [];
  private bgSsimFwd!: GPUBindGroup;
  private bgSsimBwd!: GPUBindGroup;
  private bgColourAdam!: GPUBindGroup;
  private bgRasterBwd: GPUBindGroup[] = [];
  private bgAdam!: GPUBindGroup;
  private bgReloc: GPUBindGroup[] = [];

  private views: { cam: SplatCamera; width: number; height: number }[] = [];
  private order: number[] = [];
  private orderPos = 0;
  private extent = 1;
  private count = 0;
  private iter = 0;
  private maxIsect = 0;
  private gpuMs: Record<string, number> = {};
  private profiledIterations = 0;
  private mcmcStats = { refinements: 0, relocated: 0, added: 0 };
  private initialised = false;
  /** Views whose image has not been uploaded yet. */
  private missingImages = new Set<number>();
  /**
   * Per view, its valid target pixels (alpha ≥ 0.5 over the whole box, as TARGET_WGSL) at each
   * level factor of the schedule; null: every pixel is valid (no alpha below 0.5 in the image).
   */
  private validPx: (Map<number, number> | null)[] = [];
  /** Stamp written by the iteration of each stats slot, and the submission counter. */
  private readonly slotStamp = new Uint32Array(STAT_SLOTS);
  private stamps = 0;
  /** Stamp of the last MCMC relocation (copied into slot STAT_SLOTS by its command buffer). */
  private relocStamp = 0;
  /** Stamp of the last readback (copied behind its data, see readBuffer) and its source buffer. */
  private readStamp = 0;
  private readStampBuf: GPUBuffer | null = null;
  /** Test hook: command buffers to discard instead of submitting (simulated dropped work). */
  private dropNext = { iterations: 0, relocations: 0, readbacks: 0 };

  private constructor(
    private readonly device: GPUDevice,
    private readonly lost: Promise<unknown>,
    private pipes: TrainPipelines,
    readonly opts: SplatTrainerOptions,
  ) {
    this.shDegree = opts.shDegree ?? 0;
    this.antialiased = opts.antialiased ?? true;
    this.capacity = opts.maxGaussians;
    this.lr = { ...DEFAULT_SPLAT_LR, ...(opts.lr ?? {}) };
    const T = opts.iterations;
    this.mcmc =
      opts.mcmc === false
        ? null
        : {
            refineEvery: 100,
            start: Math.max(100, Math.round(0.1 * T)),
            stop: Math.round(0.75 * T),
            growth: 0.05,
            minOpacity: 0.005,
            noiseLr: 5e5,
            opacityReg: 0.01,
            scaleReg: 0.01,
            ...(opts.mcmc ?? {}),
          };
    this.cc = opts.colourCorrection === false ? null : { lr: 2e-3, reg: 1e-2, ...(opts.colourCorrection ?? {}) };
    this.geo = opts.geometry ? { ...DEFAULT_SPLAT_GEOMETRY, ...opts.geometry } : null;
    this.lambda = opts.ssimLambda ?? 0.2;
    this.background = opts.background ?? [0, 0, 0];
    this.levels = opts.resolution ?? [
      { factor: 4, until: 0.1 },
      { factor: 2, until: 0.3 },
    ];
    this.rng = new Rng(opts.seed ?? 1);
  }

  /**
   * Compiles every pipeline (call at stage start; compilation blocks the queue on Safari).
   * Pipelines are cached per device and shader configuration.
   */
  static async create(ctx: Pick<GPUContext, 'device'>, opts: SplatTrainerOptions): Promise<SplatTrainer> {
    const device = ctx.device;
    const cfg: SplatShaderConfig = { shDegree: opts.shDegree ?? 0, antialiased: opts.antialiased ?? true, geometry: geometryMode(opts.geometry) };
    let perDevice = pipelineCache.get(device);
    if (!perDevice) pipelineCache.set(device, (perDevice = new Map()));
    const key = shaderKey(cfg);
    let pipes = perDevice.get(key);
    if (!pipes) {
      pipes = compileTrainPipelines(device, cfg);
      perDevice.set(key, pipes);
      pipes.catch(() => perDevice!.delete(key));
    }
    const t = new SplatTrainer(device, device.lost, await pipes, opts);
    // geometry losses that end before the run does: plain kernels of the same layout afterwards
    if (t.geo && t.geo.until < 1) {
      const wideKey = `${key}:after`;
      let post = perDevice.get(wideKey);
      if (!post) {
        const geoPipes = await pipes;
        post = compilePostGeometryPipelines(device, { ...cfg, geometry: false, wideProj: cfg.geometry === true }, geoPipes);
        perDevice.set(wideKey, post);
        post.catch(() => perDevice!.delete(wideKey));
      }
      t.geoPipes = await pipes;
      t.postPipes = await post;
    }
    return t;
  }

  /** GPU memory plan for these options and views (`depthViews`: views with a geometry target). */
  static memoryEstimate(
    opts: Pick<SplatTrainerOptions, 'maxGaussians' | 'shDegree' | 'intersectionsPerGaussian' | 'geometry'>,
    width: number,
    height: number,
    views: number,
    depthViews = views,
  ): SplatMemoryEstimate {
    const mode = geometryMode(opts.geometry);
    const geometry = mode ? { views: depthViews, factor: (opts.geometry && opts.geometry.factor) || DEFAULT_SPLAT_GEOMETRY.factor, normals: mode === true } : undefined;
    return estimateSplatMemory({ maxGaussians: opts.maxGaussians, shDegree: opts.shDegree ?? 0, width, height, views, intersectionsPerGaussian: opts.intersectionsPerGaussian, geometry });
  }

  memory(): SplatMemoryEstimate {
    const w = Math.max(...this.views.map((v) => v.width));
    const h = Math.max(...this.views.map((v) => v.height));
    return estimateSplatMemory({
      maxGaussians: this.capacity,
      shDegree: this.shDegree,
      width: w,
      height: h,
      views: this.views.length,
      intersectionsPerGaussian: this.fwd ? this.fwd.isectCapacity / this.capacity : this.opts.intersectionsPerGaussian,
      geometry: this.geo ? { views: this.depthTargets.filter((t) => t).length, factor: this.geo.factor, normals: geometryMode(this.geo) === true } : undefined,
    });
  }

  get iteration(): number {
    return this.iter;
  }

  get gaussians(): number {
    return this.count;
  }

  get sceneExtent(): number {
    return this.extent;
  }

  /**
   * Uploads the initial model and the training views (pinhole cameras; RGBA8 images now or later
   * through `setViewImage`).
   */
  init(model: SplatModel, views: SplatTrainingView[]): void {
    if (this.initialised) throw new Error('SplatTrainer: init called twice');
    if (model.shDegree !== this.shDegree) throw new Error(`SplatTrainer: model SH degree ${model.shDegree}, trainer ${this.shDegree}`);
    if (!views.length) throw new Error('SplatTrainer: no training views');
    const d = this.device;
    const cap = this.capacity;
    const n = Math.min(model.count, cap);
    if (model.count > cap) console.warn(`[splat] init: ${model.count} Gaussians truncated to the cap ${cap}`);
    this.count = n;
    const P = this.pipes;
    const layout = paramOffsets(cap, this.shDegree);
    // (throws before anything is allocated: what init allocates is freed by destroy() even when it fails)
    SplatForward.check({ capacity: cap, isectCapacity: Math.ceil(cap * (this.opts.intersectionsPerGaussian ?? 12)) });
    this.params = makeBuffer(d, layout.total * 4, 'splat-params');
    this.adamM = makeBuffer(d, layout.total * 4, 'splat-adam-m');
    this.adamV = makeBuffer(d, layout.total * 4, 'splat-adam-v');
    this.grad2d = makeBuffer(d, cap * grad2dStride(P.fwd.cfg) * 4, 'splat-grad2d');
    d.queue.writeBuffer(this.params, 0, packParams({ ...model, count: n }, cap));
    this.views = views.map((v) => ({ cam: splatCamera(v.cam, v.pose), width: v.cam.width, height: v.cam.height }));
    const maxW = Math.max(...views.map((v) => v.cam.width));
    const maxH = Math.max(...views.map((v) => v.cam.height));
    this.fwd = new SplatForward(d, P.fwd, {
      capacity: cap,
      maxWidth: maxW,
      maxHeight: maxH,
      isectCapacity: Math.ceil(cap * (this.opts.intersectionsPerGaussian ?? 12)),
      params: this.params,
      label: 'splat-train',
    });
    const maxPx = maxW * maxH;
    const maxTiles = Math.ceil(maxW / TILE_SIZE) * Math.ceil(maxH / TILE_SIZE);
    this.target = makeBuffer(d, maxPx * 4, 'splat-target');
    this.pqr = makeBuffer(d, maxPx * 48, 'splat-pqr');
    this.dLdC = makeBuffer(d, maxPx * 16, 'splat-dldc');
    this.lossPartial = makeBuffer(d, maxTiles * 16, 'splat-loss-partial');
    this.colourPartial = makeBuffer(d, maxTiles * 48, 'splat-colour-partial');
    this.colour = makeBuffer(d, views.length * COLOUR_FLOATS * 4, 'splat-colour');
    const col = new Float32Array(views.length * COLOUR_FLOATS);
    for (let v = 0; v < views.length; v++) {
      col[v * COLOUR_FLOATS] = 1;
      col[v * COLOUR_FLOATS + 5] = 1;
      col[v * COLOUR_FLOATS + 10] = 1;
    }
    d.queue.writeBuffer(this.colour, 0, col);
    this.statsBuf = makeBuffer(d, (STAT_SLOTS + 1) * 16, 'splat-stats');
    const uni = (bytes: number, label: string, extra = 0): GPUBuffer => d.createBuffer({ size: bytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | extra, label });
    this.optU = uni(OPT_BYTES, 'splat-opt');
    this.lossU = uni(LOSS_CFG_BYTES, 'splat-losscfg');
    this.colourU = uni(COLOUR_CFG_BYTES, 'splat-colourcfg');
    // copied into the stats buffer by each relocation (its dropped-work stamp)
    this.relocU = uni(16, 'splat-reloccfg', GPUBufferUsage.COPY_SRC);
    views.forEach((v, i) => {
      const b = makeBuffer(d, v.cam.width * v.cam.height * 4, `splat-view-${v.id}`);
      this.images.push(b);
      this.bgTarget.push(bindGroup(d, P.target, [this.fwd.frame, this.lossU, b, this.target]));
      this.missingImages.add(i);
    });
    this.bgSsimFwd = bindGroup(d, P.ssimFwd, [this.fwd.frame, this.lossU, this.fwd.outColor, this.target, this.colour, this.pqr, this.lossPartial]);
    this.bgSsimBwd = bindGroup(d, P.ssimBwd, [this.fwd.frame, this.lossU, this.fwd.outColor, this.target, this.colour, this.pqr, this.dLdC, this.colourPartial]);
    this.bgColourAdam = bindGroup(d, P.colourAdam, [this.colourU, this.colourPartial, this.lossPartial, this.colour, this.statsBuf]);
    this.bgAdam = bindGroup(d, P.adam, [this.fwd.frame, this.optU, this.params, this.adamM, this.adamV, this.grad2d, this.fwd.tileCount]);
    this.makeRasterBwdBindGroups();
    if (this.geo) {
      this.geoU = uni(GEO_CFG_BYTES, 'splat-geocfg');
      this.geoPartial = makeBuffer(d, maxTiles * 16, 'splat-geo-partial');
      this.geoDummy = makeBuffer(d, 16, 'splat-geo-none');
      this.depthTargets = views.map(() => null);
      this.depthDims = views.map(() => null);
      this.bgGeo = views.map(() => bindGroup(d, P.geoLoss!, [this.fwd.frame, this.geoU!, this.fwd.outGeo!, this.geoDummy!, this.geoPartial!, this.fwd.outColor, this.dLdC]));
    }
    // scene extent (3DGS): 1.1 × largest distance of a camera centre from the centroid
    if (this.opts.sceneExtent) this.extent = this.opts.sceneExtent;
    else {
      const cs = this.views.map((v) => {
        const R = v.cam.R;
        const t = v.cam.t;
        return [-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])];
      });
      const m = [0, 1, 2].map((k) => cs.reduce((s, c) => s + c[k], 0) / cs.length);
      this.extent = 1.1 * Math.max(1e-6, ...cs.map((c) => Math.hypot(c[0] - m[0], c[1] - m[1], c[2] - m[2])));
    }
    if (this.opts.profile && PassProfiler.supported(d)) this.profiler = new PassProfiler(d);
    this.initialised = true;
    views.forEach((v, i) => {
      if (v.image) this.setViewImage(i, v.image);
    });
  }

  /** Uploads the RGBA8 image of training view `index` (camera size); the host copy can be dropped afterwards. */
  setViewImage(index: number, img: RGBAImage): void {
    if (!this.initialised) throw new Error('SplatTrainer: init first');
    const v = this.views[index];
    if (!v) throw new Error(`SplatTrainer: no view ${index}`);
    if (img.width !== v.width || img.height !== v.height) throw new Error(`SplatTrainer: view ${index} image ${img.width}×${img.height} ≠ camera ${v.width}×${v.height}`);
    this.device.queue.writeBuffer(this.images[index], 0, img.data.buffer, img.data.byteOffset, img.width * img.height * 4);
    this.validPx[index] = validTargetPixels(img, [1, ...this.levels.map((l) => l.factor)]);
    this.missingImages.delete(index);
  }

  /**
   * Geometry target of training view `index` (depth-target.ts `packDepthTarget` at the trainer's
   * `geometry.factor` of the view size), or null to drop it. Needs `geometry`.
   */
  setViewDepth(index: number, target: DepthTarget | null): void {
    if (!this.initialised) throw new Error('SplatTrainer: init first');
    if (!this.geo) throw new Error('SplatTrainer: setViewDepth needs the geometry option');
    const v = this.views[index];
    if (!v) throw new Error(`SplatTrainer: no view ${index}`);
    this.depthTargets[index]?.destroy();
    this.depthTargets[index] = null;
    this.depthDims[index] = null;
    let buf = this.geoDummy!;
    if (target) {
      const f = this.geo.factor;
      if (target.factor !== f || target.width !== Math.max(1, Math.floor(v.width / f)) || target.height !== Math.max(1, Math.floor(v.height / f)))
        throw new Error(`SplatTrainer: depth target ${target.width}×${target.height} /${target.factor} ≠ view ${index} ${v.width}×${v.height} /${f}`);
      buf = makeBuffer(this.device, Math.max(16, target.data.byteLength), `splat-depth-${index}`);
      this.device.queue.writeBuffer(buf, 0, target.data.buffer, target.data.byteOffset, target.data.byteLength);
      this.depthTargets[index] = buf;
      this.depthDims[index] = { width: target.width, height: target.height, factor: target.factor };
    }
    this.bgGeo[index] = bindGroup(this.device, this.pipes.geoLoss!, [this.fwd.frame, this.geoU!, this.fwd.outGeo!, buf, this.geoPartial!, this.fwd.outColor, this.dLdC]);
  }

  /** Whether the geometry kernels are active (false after `geometry.until`, or without geometry). */
  get geometryActive(): boolean {
    return !!this.pipes.fwd.cfg.geometry;
  }

  /** Training views with a geometry target. */
  get depthViews(): number {
    return this.depthTargets.filter((t) => t).length;
  }

  /** Number of training views. */
  get numViews(): number {
    return this.views.length;
  }

  private makeRasterBwdBindGroups(): void {
    const f = this.fwd;
    this.bgRasterBwd = [0, 1].map((k) =>
      bindGroup(this.device, this.pipes.rasterBwd, [f.frame, f.ranges, f.isectSort.vals[k], f.proj, f.outColor, f.outN, this.dLdC, this.grad2d, ...(this.pipes.fwd.cfg.geometry ? [f.outGeo!] : [])]),
    );
  }

  /** Geometry pipelines while the geometry losses run, the plain ones of the same layout after. */
  private syncPhase(): void {
    if (!this.geoPipes || !this.postPipes) return;
    const want = this.iter < this.geoEndIteration() ? this.geoPipes : this.postPipes;
    if (want === this.pipes) return;
    this.pipes = want;
    this.fwd.setPipelines(want.fwd);
    this.makeRasterBwdBindGroups();
    this.bgAdam = bindGroup(this.device, want.adam, [this.fwd.frame, this.optU, this.params, this.adamM, this.adamV, this.grad2d, this.fwd.tileCount]);
  }

  /** First iteration without the geometry losses (Infinity: they run to the end). */
  private geoEndIteration(): number {
    const g = this.geo;
    if (!g || g.until >= 1) return Infinity;
    return Math.max(1, this.opts.iterations) * Math.max(0, g.until);
  }

  /** Geometry-loss weights (depth, normal) at iteration `it`. */
  geometryWeightsAt(it: number): [number, number] {
    const g = this.geo;
    if (!g) return [0, 0];
    const T = Math.max(1, this.opts.iterations) * Math.max(0, Math.min(1, g.until));
    if (!(it < T)) return [0, 0];
    const f = it / T;
    const interp = ([a, b]: [number, number]): number => (a > 0 && b > 0 ? Math.exp(Math.log(a) * (1 - f) + Math.log(b) * f) : a + (b - a) * f);
    return [interp(g.depth), interp(g.normal)];
  }

  private nextView(): number {
    if (this.orderPos >= this.order.length) {
      this.order = Array.from({ length: this.views.length }, (_, i) => i);
      this.rng.shuffle(this.order);
      this.orderPos = 0;
    }
    return this.order[this.orderPos++];
  }

  /** Downscale factor of the resolution schedule at iteration `it`. */
  levelAt(it: number): number {
    const frac = it / Math.max(1, this.opts.iterations);
    for (const l of this.levels) if (frac < l.until) return l.factor;
    return 1;
  }

  /** Mean learning rate (world units) at iteration `it`. */
  meanLrAt(it: number): number {
    const T = Math.max(1, this.opts.iterations);
    const f = Math.min(1, it / T);
    return this.extent * Math.exp(Math.log(this.lr.mean) * (1 - f) + Math.log(this.lr.meanFinal) * f);
  }

  private writeUniforms(view: number, cam: SplatCamera, factor: number): void {
    const d = this.device;
    const it = this.iter;
    const t = it + 1;
    const b1 = 0.9;
    const b2 = 0.999;
    const lrMean = this.meanLrAt(it);
    const N = Math.max(1, this.count);
    const mc = this.mcmc;
    const opt = new ArrayBuffer(OPT_BYTES);
    const of = new Float32Array(opt);
    const ou = new Uint32Array(opt);
    of.set([lrMean, this.lr.scale, this.lr.quat, this.lr.opacity], 0);
    of.set([this.lr.sh0, this.lr.shN, mc ? mc.noiseLr * lrMean : 0, 0], 4);
    of.set([b1, b2, 1e-15, 0], 8);
    of.set([1 - Math.pow(b1, t), 1 - Math.pow(b2, t), 0, 0], 12);
    of.set([mc ? mc.opacityReg / N : 0, mc ? mc.scaleReg / (3 * N * this.extent) : 0, 0, 0], 16);
    ou.set([(Math.imul(it + 1, 0x9e3779b1) ^ 0x7f4a7c15) >>> 0, 0, 0, 0], 20);
    const an = this.opts.anisotropy;
    of.set([(this.opts.flatten ?? 0) / (N * this.extent), an ? an.weight / N : 0, an ? Math.log(Math.max(1, an.maxRatio)) : 0, 0], 24);
    d.queue.writeBuffer(this.optU, 0, opt);
    if (this.geo) {
      const t = this.depthDims[view];
      const [wd, wn] = this.geometryWeightsAt(it);
      const gb = new ArrayBuffer(GEO_CFG_BYTES);
      new Uint32Array(gb).set([t?.width ?? 0, t?.height ?? 0, factor, t?.factor ?? 1], 0);
      new Float32Array(gb).set([wd, wn, 1 / (cam.width * cam.height)], 4);
      new Uint32Array(gb)[7] = t ? 1 : 0;
      d.queue.writeBuffer(this.geoU!, 0, gb);
    }
    const v = this.views[view];
    const lc = new ArrayBuffer(LOSS_CFG_BYTES);
    new Uint32Array(lc).set([view, v.width, v.height, factor], 0);
    // ∂loss/∂C normalised by the valid pixels, like the loss (masked targets: undistortion borders)
    const nv = this.validPx[view]?.get(factor) ?? cam.width * cam.height;
    new Float32Array(lc).set([this.lambda, 1 / (3 * Math.max(1, nv))], 4);
    d.queue.writeBuffer(this.lossU, 0, lc);
    const cc = new ArrayBuffer(COLOUR_CFG_BYTES);
    const slot = it % STAT_SLOTS;
    // dropped-work stamp of this iteration (exact as f32 below 2²⁴ submissions)
    this.stamps = (this.stamps % 0xfffff0) + 1;
    this.slotStamp[slot] = this.stamps;
    const ccu = new Uint32Array(cc);
    ccu.set([Math.ceil(cam.width / TILE_SIZE) * Math.ceil(cam.height / TILE_SIZE), view, this.cc ? 1 : 0, slot], 0);
    new Float32Array(cc).set([this.cc?.lr ?? 0, 0.9, 0.999, 1e-15, this.cc?.reg ?? 0], 4);
    ccu[9] = this.stamps;
    d.queue.writeBuffer(this.colourU, 0, cc);
  }

  private encodeIteration(view: number): GPUCommandEncoder {
    this.syncPhase();
    const factor = this.levelAt(this.iter);
    const cam = downscaleSplatCamera(this.views[view].cam, factor);
    this.fwd.setFrame(cam, this.count, this.background);
    this.writeUniforms(view, cam, factor);
    const P = this.pipes;
    const prof = this.profiler;
    const enc = this.device.createCommandEncoder({ label: 'splat-iteration' });
    this.fwd.encode(enc, prof);
    const tx = this.fwd.tilesX;
    const ty = this.fwd.tilesY;
    let pass = prof ? prof.pass(enc, 'loss') : enc.beginComputePass({ label: 'loss' });
    pass.setPipeline(P.target);
    pass.setBindGroup(0, this.bgTarget[view]);
    pass.dispatchWorkgroups(tx, ty);
    pass.setPipeline(P.ssimFwd);
    pass.setBindGroup(0, this.bgSsimFwd);
    pass.dispatchWorkgroups(tx, ty);
    pass.setPipeline(P.ssimBwd);
    pass.setBindGroup(0, this.bgSsimBwd);
    pass.dispatchWorkgroups(tx, ty);
    if (P.geoLoss) {
      pass.setPipeline(P.geoLoss);
      pass.setBindGroup(0, this.bgGeo[view]);
      pass.dispatchWorkgroups(tx, ty);
    }
    pass.setPipeline(P.colourAdam);
    pass.setBindGroup(0, this.bgColourAdam);
    pass.dispatchWorkgroups(1);
    pass.end();
    pass = prof ? prof.pass(enc, 'raster-bwd') : enc.beginComputePass({ label: 'raster-bwd' });
    pass.setPipeline(P.rasterBwd);
    pass.setBindGroup(0, this.bgRasterBwd[this.fwd.isectResult]);
    pass.dispatchWorkgroups(tx, ty, 8 / BWD_GROUPS);
    pass.end();
    pass = prof ? prof.pass(enc, 'adam') : enc.beginComputePass({ label: 'adam' });
    pass.setPipeline(P.adam);
    pass.setBindGroup(0, this.bgAdam);
    pass.dispatchWorkgroups(Math.ceil(this.count / 256));
    pass.end();
    return enc;
  }

  private async waitGPU(): Promise<void> {
    const r = await Promise.race([this.device.queue.onSubmittedWorkDone().then(() => 'ok' as const), this.lost.then(() => 'lost' as const)]);
    if (r === 'lost') throw new Error('GPU device lost during splat training');
  }

  private ensureStaging(bytes: number): GPUBuffer {
    if (!this.staging || this.staging.size < bytes) {
      this.staging?.destroy();
      this.staging = this.device.createBuffer({ size: Math.ceil(bytes / 256) * 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'splat-train-staging' });
    }
    return this.staging;
  }

  /**
   * Reads `bytes` of `src` back. The same command buffer copies a fresh stamp behind the data:
   * a dropped copy (iOS drops work submitted while the app goes to the background, and mapAsync
   * still resolves) leaves the previous staging contents and throws `SplatWorkDroppedError`
   * instead of returning them. Nothing on the GPU changes, so the read can simply be repeated.
   */
  private async readBuffer(src: GPUBuffer, offset: number, bytes: number): Promise<ArrayBuffer> {
    const size = Math.ceil(bytes / 4) * 4;
    const st = this.ensureStaging(size + 4);
    if (!this.readStampBuf) this.readStampBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label: 'splat-read-stamp' });
    this.readStamp = (this.readStamp % 0xfffffff0) + 1;
    const stamp = this.readStamp;
    this.device.queue.writeBuffer(this.readStampBuf, 0, new Uint32Array([stamp]));
    const enc = this.device.createCommandEncoder({ label: 'splat-readback' });
    enc.copyBufferToBuffer(src, offset, st, 0, size);
    enc.copyBufferToBuffer(this.readStampBuf, 0, st, size, 4);
    if (this.dropNext.readbacks > 0) {
      this.dropNext.readbacks--;
      enc.finish();
    } else this.device.queue.submit([enc.finish()]);
    const r = await Promise.race([st.mapAsync(GPUMapMode.READ, 0, size + 4).then(() => 'ok' as const), this.lost.then(() => 'lost' as const)]);
    if (r === 'lost') throw new Error('GPU device lost during readback');
    const mapped = st.getMappedRange(0, size + 4);
    const got = new Uint32Array(mapped, size, 1)[0];
    const out = mapped.slice(0, size);
    st.unmap();
    if (got !== stamp) throw new SplatWorkDroppedError('splat readback: GPU work was not executed (page hidden?)', 0, false);
    return out;
  }

  /**
   * Runs `n` iterations (one command buffer each, awaited in turn). Returns the loss of the
   * last one; MCMC refinements and intersection-capacity checks happen in between. Throws
   * `SplatWorkDroppedError` when the stamps show that one of the last ≤ 64 iterations or this
   * call's relocation did not execute (restore a checkpoint before continuing).
   */
  async step(n = 1): Promise<SplatStepResult> {
    if (!this.initialised) throw new Error('SplatTrainer: init first');
    if (this.missingImages.size) throw new Error(`SplatTrainer: ${this.missingImages.size} view image(s) not uploaded (setViewImage)`);
    const t0 = performance.now();
    const reloc0 = this.relocStamp;
    let refinements = 0;
    let overflow = false;
    const inFlight = this.opts.inFlight ?? 1;
    const m = this.mcmc;
    const refineDue = (it: number): boolean => !!m && it >= m.start && it <= m.stop && it % m.refineEvery === 0;
    // host work after iteration `it` that must see its result (and may change buffers / the count)
    const syncAfter = (it: number, k: number): boolean => k === n - 1 || it % 25 === 0 || refineDue(it);
    let prev: Promise<void> | null = null;
    let enc: GPUCommandEncoder | null = null;
    for (let k = 0; k < n; k++) {
      if (!enc) enc = this.encodeIteration(this.nextView());
      this.profiler?.resolve(enc);
      if (this.dropNext.iterations > 0) {
        this.dropNext.iterations--;
        enc.finish();
      } else this.device.queue.submit([enc.finish()]);
      enc = null;
      const done = this.waitGPU();
      const it = this.iter + 1;
      this.iter = it;
      const sync = syncAfter(it, k);
      if (inFlight <= 1 || this.profiler || sync) {
        // encode the next iteration while this one runs (still one command buffer in flight)
        if (!sync) enc = this.encodeIteration(this.nextView());
        await done;
        if (prev) await prev;
        prev = null;
      } else {
        // keep this iteration queued behind the previous one (two command buffers in flight)
        if (prev) await prev;
        prev = done;
      }
      if (this.profiler) {
        const t = await this.profiler.read();
        if (t) {
          for (const [l, ms] of t) this.gpuMs[l] = (this.gpuMs[l] ?? 0) + ms;
          this.profiledIterations++;
        }
      }
      if (it % 25 === 0 || k === n - 1) overflow = (await this.checkIntersections()) || overflow;
      if (refineDue(it)) {
        await this.refine();
        refinements++;
      }
    }
    const ms = performance.now() - t0;
    // every stats slot of this call's last ≤ 64 iterations + the relocation stamp, one readback
    const raw = await this.readBuffer(this.statsBuf, 0, (STAT_SLOTS + 1) * 16);
    const stats = new Float32Array(raw);
    const checked = Math.min(n, STAT_SLOTS);
    let dropped = 0;
    for (let i = this.iter - checked; i < this.iter; i++) {
      const slot = i % STAT_SLOTS;
      if (stats[4 * slot + 3] !== this.slotStamp[slot]) dropped++;
    }
    const relocDropped = this.relocStamp !== reloc0 && new Uint32Array(raw)[4 * STAT_SLOTS + 3] !== this.relocStamp;
    if (dropped || relocDropped) {
      throw new SplatWorkDroppedError(
        `splat training: GPU work was not executed (${dropped} of the last ${checked} iterations${relocDropped ? ', an MCMC relocation' : ''}; page hidden?)`,
        dropped,
        relocDropped,
      );
    }
    const so = 4 * ((this.iter - 1) % STAT_SLOTS);
    const nv = Math.max(1, stats[so + 2]);
    const l1 = stats[so] / (3 * nv);
    const ssim = stats[so + 1] / (3 * nv);
    return {
      iteration: this.iter,
      count: this.count,
      loss: (1 - this.lambda) * l1 + this.lambda * (1 - ssim),
      l1,
      ssim,
      ms,
      iterationsPerSecond: (1000 * n) / ms,
      refinements,
      overflow,
    };
  }

  /** Reads the last frame's (uncapped) intersection count; grows the reservation when close to it. */
  private async checkIntersections(): Promise<boolean> {
    const total = new Uint32Array(await this.readBuffer(this.fwd.binCtl, 13 * 4, 4))[0];
    this.maxIsect = Math.max(this.maxIsect, total);
    const cap = this.fwd.isectCapacity;
    const over = total > cap;
    if (total > 0.85 * cap) {
      const next = Math.ceil(Math.max(total, this.maxIsect) * 1.5);
      if (next * 4 <= this.device.limits.maxStorageBufferBindingSize && next <= 64 * 1024 * 1024) {
        this.fwd.resizeIntersections(next);
        this.makeRasterBwdBindGroups();
      }
    }
    return over;
  }

  /** One MCMC relocation / growth step (host plan from the opacities, applied on the GPU). */
  private async refine(): Promise<void> {
    const m = this.mcmc!;
    const lay = paramOffsets(this.capacity, this.shDegree);
    const op = new Float32Array(await this.readBuffer(this.params, lay.opacity * 4, this.count * 4));
    const plan = planMcmc(op, this.count, { capacity: this.capacity, growth: m.growth, minOpacity: m.minOpacity }, this.rng);
    this.mcmcStats.refinements++;
    this.mcmcStats.relocated += Math.min(plan.dead, plan.nCopies);
    this.mcmcStats.added += plan.added;
    if (plan.nCopies === 0 && plan.nUpdates === 0) return;
    const entries = plan.nCopies + plan.nUpdates;
    const d = this.device;
    if (!this.relocList || this.relocList.size < entries * 16) {
      this.relocList?.destroy();
      this.relocList = makeBuffer(d, Math.max(entries, 1024) * 16 * 2, 'splat-reloc-list');
      this.bgReloc = [this.pipes.relocCopy, this.pipes.relocUpdate].map((p) => bindGroup(d, p, [this.relocU, this.relocList!, this.params, this.adamM, this.adamV]));
    }
    const list = new Uint32Array(entries * 4);
    list.set(plan.copies.subarray(0, 4 * plan.nCopies), 0);
    list.set(plan.updates, 4 * plan.nCopies);
    d.queue.writeBuffer(this.relocList, 0, list);
    this.relocStamp = (this.relocStamp % 0x7ffffff0) + 1;
    d.queue.writeBuffer(this.relocU, 0, new Uint32Array([plan.nCopies, plan.nUpdates, this.capacity, this.relocStamp]));
    const enc = d.createCommandEncoder({ label: 'splat-mcmc' });
    const pass = enc.beginComputePass({ label: 'mcmc' });
    pass.setPipeline(this.pipes.relocCopy);
    pass.setBindGroup(0, this.bgReloc[0]);
    pass.dispatchWorkgroups(Math.ceil(plan.nCopies / 64));
    pass.setPipeline(this.pipes.relocUpdate);
    pass.setBindGroup(0, this.bgReloc[1]);
    pass.dispatchWorkgroups(Math.ceil(plan.nUpdates / 64));
    pass.end();
    // dropped-work stamp: [nCopies, nUpdates, capacity, relocStamp] → stats slot STAT_SLOTS
    enc.copyBufferToBuffer(this.relocU, 0, this.statsBuf, STAT_SLOTS * 16, 16);
    if (this.dropNext.relocations > 0) {
      this.dropNext.relocations--;
      enc.finish();
    } else d.queue.submit([enc.finish()]);
    await this.waitGPU();
    this.count = plan.count;
  }

  /** Renders the current model (forward only; between steps). */
  async render(cam: SplatCamera, background: ArrayLike<number> = this.background): Promise<SplatImage> {
    const f = this.fwd;
    f.setFrame(cam, this.count, background);
    const px = cam.width * cam.height;
    const st = this.ensureStaging(px * 16);
    const enc = this.device.createCommandEncoder({ label: 'splat-train-render' });
    f.encode(enc, null);
    enc.copyBufferToBuffer(f.outColor, 0, st, 0, px * 16);
    this.device.queue.submit([enc.finish()]);
    const r = await Promise.race([st.mapAsync(GPUMapMode.READ, 0, px * 16).then(() => 'ok' as const), this.lost.then(() => 'lost' as const)]);
    if (r === 'lost') throw new Error('GPU device lost during render');
    const data = new Float32Array(st.getMappedRange(0, px * 16).slice(0));
    st.unmap();
    return { width: cam.width, height: cam.height, data };
  }

  /** The current model (host copy). */
  readModel(): Promise<SplatModel> {
    return this.readLayout(this.params);
  }

  /**
   * Adam first ('m') or second ('v') moments of the live Gaussians in the `SplatModel` layout
   * (each field holds the moments of that parameter) — part of a training checkpoint.
   */
  readMoments(which: 'm' | 'v'): Promise<SplatModel> {
    return this.readLayout(which === 'm' ? this.adamM : this.adamV);
  }

  /** The first `count` entries of each section of a parameter-layout buffer (one readback per section). */
  private async readLayout(buf: GPUBuffer): Promise<SplatModel> {
    const lay = paramOffsets(this.capacity, this.shDegree);
    const n = this.count;
    const SN = 3 * shCoeffCount(this.shDegree);
    const read = async (off: number, floats: number): Promise<Float32Array> => (floats ? new Float32Array(await this.readBuffer(buf, off * 4, floats * 4)) : new Float32Array(0));
    const m = allocSplatModel(n, this.shDegree);
    const mean = await read(lay.mean, 4 * n);
    const scale = await read(lay.scale, 4 * n);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < 3; k++) {
        m.means[3 * i + k] = mean[4 * i + k];
        m.logScales[3 * i + k] = scale[4 * i + k];
      }
    }
    m.quats.set(await read(lay.quat, 4 * n));
    m.opacities.set(await read(lay.opacity, n));
    m.sh.set(await read(lay.sh, SN * n));
    return m;
  }

  /** Iteration, count, colour correction, background and counters (with readModel + readMoments: a checkpoint). */
  async readState(): Promise<SplatTrainerState> {
    const colour = new Float32Array(await this.readBuffer(this.colour, 0, this.views.length * COLOUR_FLOATS * 4));
    return {
      iteration: this.iter,
      count: this.count,
      background: [...this.background] as [number, number, number],
      colour,
      mcmc: { ...this.mcmcStats },
      maxIntersections: this.maxIsect,
    };
  }

  /**
   * Continues a checkpoint: after `init(params, views)` with the checkpoint's parameters as the
   * model, or on a running trainer with `params` (rolls it back in place: the view images stay).
   * Same views and options. Writes the Adam moments (zero when absent), colour state and
   * background, and sets the iteration (schedules, Adam bias correction). The view order and the
   * MCMC sampling restart from a seed derived from the iteration.
   */
  restore(state: SplatTrainerState, moments?: { m: SplatModel; v: SplatModel } | null, params?: SplatModel): void {
    if (!this.initialised) throw new Error('SplatTrainer: init first');
    const d = this.device;
    if (params) {
      if (params.count > this.capacity || params.shDegree !== this.shDegree) throw new Error('SplatTrainer.restore: parameters do not fit the trainer');
      d.queue.writeBuffer(this.params, 0, packParams(params, this.capacity));
      this.count = params.count;
    }
    if (state.count !== this.count) throw new Error(`SplatTrainer.restore: state has ${state.count} Gaussians, the model ${this.count}`);
    if (state.colour.length !== this.views.length * COLOUR_FLOATS) throw new Error('SplatTrainer.restore: colour state does not match the views');
    if (moments) {
      for (const [mm, buf] of [
        [moments.m, this.adamM],
        [moments.v, this.adamV],
      ] as const) {
        if (mm.count !== this.count || mm.shDegree !== this.shDegree) throw new Error('SplatTrainer.restore: moments do not match the model');
        d.queue.writeBuffer(buf, 0, packParams(mm, this.capacity));
      }
    } else {
      const zero = new Float32Array(paramOffsets(this.capacity, this.shDegree).total);
      d.queue.writeBuffer(this.adamM, 0, zero);
      d.queue.writeBuffer(this.adamV, 0, zero);
    }
    d.queue.writeBuffer(this.colour, 0, state.colour);
    this.background = [...state.background] as [number, number, number];
    this.iter = state.iteration;
    this.mcmcStats = { ...state.mcmc };
    this.maxIsect = state.maxIntersections;
    if (this.maxIsect > 0.85 * this.fwd.isectCapacity) {
      const next = Math.ceil(this.maxIsect * 1.5);
      if (next * 4 <= d.limits.maxStorageBufferBindingSize && next <= 64 * 1024 * 1024) {
        this.fwd.resizeIntersections(next);
        this.makeRasterBwdBindGroups();
      }
    }
    this.rng = new Rng((this.opts.seed ?? 1) + 7919 * (state.iteration + 1));
    this.order = [];
    this.orderPos = 0;
  }

  /** Training background colour (changes with `normaliseAppearance`). */
  get backgroundColour(): [number, number, number] {
    return [...this.background] as [number, number, number];
  }

  /**
   * The current model as an INRIA-layout splat PLY (ply.ts). With colour correction the
   * model is first normalised to the mean appearance of the training views
   * (`normaliseAppearance`), so other viewers show the average exposure.
   */
  async exportPLY(comments: string[] = []): Promise<Uint8Array> {
    if (this.cc) await this.normaliseAppearance();
    return encodeSplatPLY(await this.readModel(), { antialiased: this.antialiased, comments });
  }

  /**
   * Gauge transform of the per-view colour correction: with M̄ = (A, b) the mean of the
   * training views' colour matrices, the splat colours become A·c + b (SH band 0 affine, higher
   * bands × A), the background A·bg + b and every view matrix M_v ∘ M̄⁻¹, so the corrected
   * renders of the training views are unchanged while the raw render shows the average
   * exposure / white balance (what a held-out view or a viewer should get). Returns M̄
   * (row-major 3×4), or null when it is not invertible. A dropped readback throws
   * `SplatWorkDroppedError` before anything is changed (it can be called again).
   */
  async normaliseAppearance(): Promise<Float64Array | null> {
    const V = this.views.length;
    const col = new Float32Array(await this.readBuffer(this.colour, 0, V * COLOUR_FLOATS * 4));
    const Mb = new Float64Array(12);
    for (let v = 0; v < V; v++) for (let k = 0; k < 12; k++) Mb[k] += col[v * COLOUR_FLOATS + k] / V;
    const A = [Mb[0], Mb[1], Mb[2], Mb[4], Mb[5], Mb[6], Mb[8], Mb[9], Mb[10]];
    const b = [Mb[3], Mb[7], Mb[11]];
    const det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
    if (!(Math.abs(det) > 1e-6)) return null;
    const Ai = [
      (A[4] * A[8] - A[5] * A[7]) / det,
      (A[2] * A[7] - A[1] * A[8]) / det,
      (A[1] * A[5] - A[2] * A[4]) / det,
      (A[5] * A[6] - A[3] * A[8]) / det,
      (A[0] * A[8] - A[2] * A[6]) / det,
      (A[2] * A[3] - A[0] * A[5]) / det,
      (A[3] * A[7] - A[4] * A[6]) / det,
      (A[1] * A[6] - A[0] * A[7]) / det,
      (A[0] * A[4] - A[1] * A[3]) / det,
    ];
    const mul = (M: number[], x: number[]): number[] => [0, 1, 2].map((r) => M[3 * r] * x[0] + M[3 * r + 1] * x[1] + M[3 * r + 2] * x[2]);
    // splat colours
    const lay = paramOffsets(this.capacity, this.shDegree);
    const SN = 3 * (this.shDegree === 0 ? 1 : 4);
    const sh = new Float32Array(await this.readBuffer(this.params, lay.sh * 4, this.count * SN * 4));
    for (let i = 0; i < this.count; i++) {
      const o = i * SN;
      const c = mul(A, [0, 1, 2].map((k) => SH_C0 * sh[o + k] + 0.5));
      for (let k = 0; k < 3; k++) sh[o + k] = (c[k] + b[k] - 0.5) / SH_C0;
      for (let q = 3; q < SN; q += 3) {
        const f = mul(A, [sh[o + q], sh[o + q + 1], sh[o + q + 2]]);
        for (let k = 0; k < 3; k++) sh[o + q + k] = f[k];
      }
    }
    this.device.queue.writeBuffer(this.params, lay.sh * 4, sh);
    // view matrices: M_v ∘ M̄⁻¹ = (A_v A⁻¹, b_v − A_v A⁻¹ b)
    for (let v = 0; v < V; v++) {
      const o = v * COLOUR_FLOATS;
      const Av = [col[o], col[o + 1], col[o + 2], col[o + 4], col[o + 5], col[o + 6], col[o + 8], col[o + 9], col[o + 10]];
      const bv = [col[o + 3], col[o + 7], col[o + 11]];
      const An = [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => Av[3 * r] * Ai[c] + Av[3 * r + 1] * Ai[3 + c] + Av[3 * r + 2] * Ai[6 + c]));
      const bn = mul(An, b).map((x, k) => bv[k] - x);
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) col[o + 4 * r + c] = An[3 * r + c];
        col[o + 4 * r + 3] = bn[r];
      }
    }
    this.device.queue.writeBuffer(this.colour, 0, col);
    const bg = mul(A, this.background);
    this.background = [bg[0] + b[0], bg[1] + b[1], bg[2] + b[2]];
    return Mb;
  }

  /** The learned per-view colour matrices (row-major 3×4 each). */
  async readColourMatrices(): Promise<Float32Array[]> {
    const d = new Float32Array(await this.readBuffer(this.colour, 0, this.views.length * COLOUR_FLOATS * 4));
    return this.views.map((_, v) => d.slice(v * COLOUR_FLOATS, v * COLOUR_FLOATS + 12));
  }

  stats(): SplatTrainerStats {
    return {
      iteration: this.iter,
      count: this.count,
      intersectionCapacity: this.fwd?.isectCapacity ?? 0,
      maxIntersections: this.maxIsect,
      gpuMs: { ...this.gpuMs },
      profiledIterations: this.profiledIterations,
      mcmc: { ...this.mcmcStats },
    };
  }

  /**
   * Test hook: the next `iterations` iteration command buffers, `relocations` MCMC relocation
   * command buffers and `readbacks` readback command buffers are discarded instead of submitted
   * (work dropped while the page is hidden).
   */
  simulateDroppedWork(o: { iterations?: number; relocations?: number; readbacks?: number }): void {
    this.dropNext = { iterations: Math.max(0, o.iterations ?? 0), relocations: Math.max(0, o.relocations ?? 0), readbacks: Math.max(0, o.readbacks ?? 0) };
  }

  resetProfile(): void {
    this.gpuMs = {};
    this.profiledIterations = 0;
  }

  /**
   * Test hook: parameter gradients of L = Σ dLdC · C (RGB per pixel) [+ Σ dLdG · G, the geometry
   * channels: 4 per pixel] for camera `cam` through the GPU rasteriser backward and projection
   * backward (no loss kernels, no update). Also returns the rendered geometry channels.
   */
  async debugGradients(cam: SplatCamera, dLdC: Float32Array, dLdG?: Float32Array, dLdA?: Float32Array): Promise<{ grad2d: Float32Array; grads: SplatModel; geo: Float32Array | null }> {
    const d = this.device;
    const f = this.fwd;
    f.setFrame(cam, this.count, this.background);
    const px = cam.width * cam.height;
    const g4 = new Float32Array(4 * px);
    for (let i = 0; i < px; i++) g4.set([dLdC[3 * i], dLdC[3 * i + 1], dLdC[3 * i + 2], dLdA ? dLdA[i] : 0], 4 * i);
    d.queue.writeBuffer(this.dLdC, 0, g4);
    if (f.outGeo) {
      // [2o + 1] = ∂L/∂G (the forward pass of this submission fills [2o])
      const g8 = new Float32Array(8 * px);
      if (dLdG) for (let i = 0; i < px; i++) g8.set(dLdG.subarray(4 * i, 4 * i + 4), 8 * i + 4);
      d.queue.writeBuffer(f.outGeo, 0, g8);
    }
    const lay = paramOffsets(this.capacity, this.shDegree);
    if (!this.gradsOut) this.gradsOut = makeBuffer(d, lay.total * 4, 'splat-grads-out');
    const G2 = grad2dStride(this.pipes.fwd.cfg);
    const g2Bytes = this.count * 4 * G2;
    const g2Copy = makeBuffer(d, g2Bytes, 'splat-grad2d-copy');
    const bg = bindGroup(d, this.pipes.grads, [f.frame, this.params, this.grad2d, f.tileCount, this.gradsOut]);
    const enc = d.createCommandEncoder();
    f.encode(enc, null);
    enc.clearBuffer(this.gradsOut);
    let pass = enc.beginComputePass();
    pass.setPipeline(this.pipes.rasterBwd);
    pass.setBindGroup(0, this.bgRasterBwd[f.isectResult]);
    pass.dispatchWorkgroups(f.tilesX, f.tilesY, 8 / BWD_GROUPS);
    pass.end();
    enc.copyBufferToBuffer(this.grad2d, 0, g2Copy, 0, Math.ceil(g2Bytes / 16) * 16);
    pass = enc.beginComputePass();
    pass.setPipeline(this.pipes.grads);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(this.count / 256));
    pass.end();
    d.queue.submit([enc.finish()]);
    const g2 = new Float32Array(await this.readBuffer(g2Copy, 0, g2Bytes));
    g2Copy.destroy();
    const gr = new Float32Array(await this.readBuffer(this.gradsOut, 0, lay.total * 4));
    let geo: Float32Array | null = null;
    if (f.outGeo) {
      const g8 = new Float32Array(await this.readBuffer(f.outGeo, 0, 32 * px));
      geo = new Float32Array(4 * px);
      for (let i = 0; i < px; i++) geo.set(g8.subarray(8 * i, 8 * i + 4), 4 * i);
    }
    return { grad2d: g2, grads: unpackParams(gr, this.count, this.capacity, this.shDegree), geo };
  }

  /**
   * Test hook: forward + geometry loss of training view `view` at level `factor` with the weights
   * of iteration `it`: the rendered geometry channels (4 per pixel), ∂L/∂(them) and the summed
   * per-workgroup partials (depth, normal, pixels, weight).
   */
  async debugGeoLoss(view: number, factor = 1, it = 0): Promise<{ geo: Float32Array; alpha: Float32Array; grad: Float32Array; gradA: Float32Array; sums: [number, number, number, number]; width: number; height: number }> {
    if (!this.geo || !this.pipes.geoLoss) throw new Error('SplatTrainer: no geometry option');
    const cam = downscaleSplatCamera(this.views[view].cam, factor);
    this.fwd.setFrame(cam, this.count, this.background);
    const iter = this.iter;
    this.iter = it;
    this.writeUniforms(view, cam, factor);
    this.iter = iter;
    const enc = this.device.createCommandEncoder();
    this.fwd.encode(enc, null);
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipes.geoLoss);
    pass.setBindGroup(0, this.bgGeo[view]);
    pass.dispatchWorkgroups(this.fwd.tilesX, this.fwd.tilesY);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    const px = cam.width * cam.height;
    const g8 = new Float32Array(await this.readBuffer(this.fwd.outGeo!, 0, 32 * px));
    const geo = new Float32Array(4 * px);
    const grad = new Float32Array(4 * px);
    for (let i = 0; i < px; i++) {
      geo.set(g8.subarray(8 * i, 8 * i + 4), 4 * i);
      grad.set(g8.subarray(8 * i + 4, 8 * i + 8), 4 * i);
    }
    const tiles = this.fwd.tilesX * this.fwd.tilesY;
    const gp = new Float32Array(await this.readBuffer(this.geoPartial!, 0, tiles * 16));
    const sums: [number, number, number, number] = [0, 0, 0, 0];
    for (let i = 0; i < tiles; i++) for (let k = 0; k < 4; k++) sums[k] += gp[4 * i + k];
    const col = new Float32Array(await this.readBuffer(this.fwd.outColor, 0, px * 16));
    const dc = new Float32Array(await this.readBuffer(this.dLdC, 0, px * 16));
    const alpha = new Float32Array(px);
    const gradA = new Float32Array(px);
    for (let i = 0; i < px; i++) {
      alpha[i] = 1 - col[4 * i + 3];
      gradA[i] = dc[4 * i + 3];
    }
    return { geo, alpha, grad, gradA, sums, width: cam.width, height: cam.height };
  }

  /**
   * Test hook: runs the loss kernels for training view `view` on the current render at level
   * `factor`; returns the loss sums and ∂loss/∂C (RGB per pixel) plus the render and target.
   */
  async debugLoss(view: number, factor = 1): Promise<{ l1: number; ssim: number; valid: number; dLdC: Float32Array; render: Float32Array; target: Uint8Array; width: number; height: number }> {
    const cam = downscaleSplatCamera(this.views[view].cam, factor);
    this.fwd.setFrame(cam, this.count, this.background);
    this.writeUniforms(view, cam, factor);
    const P = this.pipes;
    const enc = this.device.createCommandEncoder();
    this.fwd.encode(enc, null);
    const pass = enc.beginComputePass();
    pass.setPipeline(P.target);
    pass.setBindGroup(0, this.bgTarget[view]);
    pass.dispatchWorkgroups(this.fwd.tilesX, this.fwd.tilesY);
    pass.setPipeline(P.ssimFwd);
    pass.setBindGroup(0, this.bgSsimFwd);
    pass.dispatchWorkgroups(this.fwd.tilesX, this.fwd.tilesY);
    pass.setPipeline(P.ssimBwd);
    pass.setBindGroup(0, this.bgSsimBwd);
    pass.dispatchWorkgroups(this.fwd.tilesX, this.fwd.tilesY);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    const px = cam.width * cam.height;
    const tiles = this.fwd.tilesX * this.fwd.tilesY;
    const lp = new Float32Array(await this.readBuffer(this.lossPartial, 0, tiles * 16));
    let l1 = 0;
    let ssim = 0;
    let valid = 0;
    for (let i = 0; i < tiles; i++) {
      l1 += lp[4 * i];
      ssim += lp[4 * i + 1];
      valid += lp[4 * i + 2];
    }
    const g = new Float32Array(await this.readBuffer(this.dLdC, 0, px * 16));
    const dLdC = new Float32Array(3 * px);
    for (let i = 0; i < px; i++) dLdC.set(g.subarray(4 * i, 4 * i + 3), 3 * i);
    const r = new Float32Array(await this.readBuffer(this.fwd.outColor, 0, px * 16));
    const t = new Uint8Array(await this.readBuffer(this.target, 0, px * 4));
    return { l1, ssim, valid, dLdC, render: r, target: t, width: cam.width, height: cam.height };
  }

  /** Frees the GPU buffers (also those of an init that threw half-way). */
  destroy(): void {
    this.fwd?.destroy();
    for (const b of [this.params, this.adamM, this.adamV, this.grad2d, this.target, this.pqr, this.dLdC, this.lossPartial, this.colourPartial, this.colour, this.statsBuf, this.optU, this.lossU, this.colourU, this.relocU, ...this.images])
      b?.destroy();
    this.relocList?.destroy();
    this.staging?.destroy();
    this.readStampBuf?.destroy();
    this.readStampBuf = null;
    this.gradsOut?.destroy();
    this.profiler?.destroy();
    for (const b of [this.geoU, this.geoPartial, this.geoDummy, ...this.depthTargets]) b?.destroy();
    this.depthTargets = [];
    this.images = [];
    this.initialised = false;
  }
}
