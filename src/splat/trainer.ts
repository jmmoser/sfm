/**
 * On-device 3D Gaussian Splatting trainer (WebGPU). Everything per iteration stays on the GPU:
 *
 *   forward (forward.ts) → target (view image box-downscaled to the current level)
 *   → loss: per-view affine colour correction, (1 − λ)·L1 + λ·D-SSIM, ∂loss/∂C, colour-matrix
 *     Adam step → rasteriser backward (no per-pixel atomics; one float CAS add per
 *     splat × tile × component) → projection backward fused with Adam (visible Gaussians only)
 *     + MCMC regularisers and position noise.
 *
 * Every `mcmc.refineEvery` steps (between `start` and `stop`) the host reads the opacities
 * back once and plans an MCMC relocation / growth step (mcmc.ts) that the GPU applies.
 * Training views are seen in random order (a fresh permutation per epoch), at a coarse-to-fine
 * resolution schedule (default ¼ for the first 10 % of the iterations, ½ until 30 %, then full).
 *
 * Driving it: `create` compiles every pipeline (stage start), `init` uploads the initial model
 * and the views, then `step(n)` runs n iterations — one command buffer each, awaited before the
 * next (one heavy command buffer in flight, iOS rule) — and can be called repeatedly in time
 * slices from a worker. `render`, `readModel` / `exportPLY` (ply.ts), `memory()`.
 */
import type { CameraIntrinsics, Pose, RGBAImage } from '../core/types';
import type { GPUContext } from '../gpu/device';
import { Rng } from '../math/random';
import { getForwardPipelines, SplatForward, type ForwardPipelines } from './forward';
import { PassProfiler, bindGroup, compileAll, makeBuffer } from './gpu-util';
import { packParams, paramOffsets, unpackParams } from './layout';
import { planMcmc } from './mcmc';
import { encodeSplatPLY } from './ply';
import type { SplatImage } from './renderer';
import {
  downscaleSplatCamera,
  estimateSplatMemory,
  splatCamera,
  TILE_SIZE,
  type ShDegree,
  type SplatCamera,
  type SplatMemoryEstimate,
  type SplatModel,
} from './types';
import { OPT_BYTES, backwardWGSL, relocateWGSL } from './wgsl/backward';
import { COLOUR_ADAM_WGSL, COLOUR_CFG_BYTES, COLOUR_FLOATS, LOSS_CFG_BYTES, SSIM_BWD_WGSL, SSIM_FWD_WGSL, TARGET_WGSL } from './wgsl/loss';
import { rasterBackwardWGSL } from './wgsl/raster';

/** A training image with its pinhole camera (k1 = k2 = 0) and pose — `DenseView` compatible. */
export interface SplatTrainingView {
  id: number;
  cam: CameraIntrinsics;
  pose: Pose;
  image: RGBAImage;
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

const STAT_SLOTS = 64;
/** Rasteriser backward: pixel groups (32 pixels each) per workgroup, splats per thread step (wgsl/raster.ts). */
export const BWD_GROUPS = 2;
export const BWD_PER_STEP = 1;

const pipelineCache = new WeakMap<GPUDevice, Map<string, Promise<TrainPipelines>>>();

async function compileTrainPipelines(device: GPUDevice, cfg: { shDegree: ShDegree; antialiased: boolean }): Promise<TrainPipelines> {
  const lost = device.lost;
  const [fwd, p] = await Promise.all([
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
  ]);
  return { fwd, ...p };
}

export class SplatTrainer {
  readonly shDegree: ShDegree;
  readonly antialiased: boolean;
  readonly capacity: number;
  private readonly lr: SplatLearningRates;
  private readonly mcmc: SplatMcmcOptions | null;
  private readonly cc: SplatColourCorrection | null;
  private readonly lambda: number;
  private readonly background: [number, number, number];
  private readonly levels: { factor: number; until: number }[];
  private readonly rng: Rng;

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

  private constructor(
    private readonly device: GPUDevice,
    private readonly lost: Promise<unknown>,
    private readonly pipes: TrainPipelines,
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
    const cfg = { shDegree: opts.shDegree ?? 0, antialiased: opts.antialiased ?? true };
    let perDevice = pipelineCache.get(device);
    if (!perDevice) pipelineCache.set(device, (perDevice = new Map()));
    const key = `${cfg.shDegree}-${cfg.antialiased}`;
    let pipes = perDevice.get(key);
    if (!pipes) {
      pipes = compileTrainPipelines(device, cfg);
      perDevice.set(key, pipes);
      pipes.catch(() => perDevice!.delete(key));
    }
    return new SplatTrainer(device, device.lost, await pipes, opts);
  }

  /** GPU memory plan for these options and views. */
  static memoryEstimate(opts: Pick<SplatTrainerOptions, 'maxGaussians' | 'shDegree' | 'intersectionsPerGaussian'>, width: number, height: number, views: number): SplatMemoryEstimate {
    return estimateSplatMemory({ maxGaussians: opts.maxGaussians, shDegree: opts.shDegree ?? 0, width, height, views, intersectionsPerGaussian: opts.intersectionsPerGaussian });
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

  /** Uploads the initial model and the training views (RGBA8 images, pinhole cameras). */
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
    this.params = makeBuffer(d, layout.total * 4, 'splat-params');
    this.adamM = makeBuffer(d, layout.total * 4, 'splat-adam-m');
    this.adamV = makeBuffer(d, layout.total * 4, 'splat-adam-v');
    this.grad2d = makeBuffer(d, cap * 9 * 4, 'splat-grad2d');
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
    this.statsBuf = makeBuffer(d, STAT_SLOTS * 16, 'splat-stats');
    const uni = (bytes: number, label: string): GPUBuffer => d.createBuffer({ size: bytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label });
    this.optU = uni(OPT_BYTES, 'splat-opt');
    this.lossU = uni(LOSS_CFG_BYTES, 'splat-losscfg');
    this.colourU = uni(COLOUR_CFG_BYTES, 'splat-colourcfg');
    this.relocU = uni(16, 'splat-reloccfg');
    for (const v of views) {
      const img = v.image;
      if (img.width !== v.cam.width || img.height !== v.cam.height) throw new Error(`SplatTrainer: view ${v.id} image ${img.width}×${img.height} ≠ camera ${v.cam.width}×${v.cam.height}`);
      const b = makeBuffer(d, img.width * img.height * 4, `splat-view-${v.id}`);
      d.queue.writeBuffer(b, 0, img.data.buffer, img.data.byteOffset, img.width * img.height * 4);
      this.images.push(b);
      this.bgTarget.push(bindGroup(d, P.target, [this.fwd.frame, this.lossU, b, this.target]));
    }
    this.bgSsimFwd = bindGroup(d, P.ssimFwd, [this.fwd.frame, this.lossU, this.fwd.outColor, this.target, this.colour, this.pqr, this.lossPartial]);
    this.bgSsimBwd = bindGroup(d, P.ssimBwd, [this.fwd.frame, this.lossU, this.fwd.outColor, this.target, this.colour, this.pqr, this.dLdC, this.colourPartial]);
    this.bgColourAdam = bindGroup(d, P.colourAdam, [this.colourU, this.colourPartial, this.lossPartial, this.colour, this.statsBuf]);
    this.bgAdam = bindGroup(d, P.adam, [this.fwd.frame, this.optU, this.params, this.adamM, this.adamV, this.grad2d, this.fwd.tileCount]);
    this.makeRasterBwdBindGroups();
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
  }

  private makeRasterBwdBindGroups(): void {
    const f = this.fwd;
    this.bgRasterBwd = [0, 1].map((k) =>
      bindGroup(this.device, this.pipes.rasterBwd, [f.frame, f.ranges, f.isectSort.vals[k], f.proj, f.outColor, f.outN, this.dLdC, this.grad2d]),
    );
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
    d.queue.writeBuffer(this.optU, 0, opt);
    const v = this.views[view];
    const lc = new ArrayBuffer(LOSS_CFG_BYTES);
    new Uint32Array(lc).set([view, v.width, v.height, factor], 0);
    new Float32Array(lc).set([this.lambda, 1 / (3 * cam.width * cam.height)], 4);
    d.queue.writeBuffer(this.lossU, 0, lc);
    const cc = new ArrayBuffer(COLOUR_CFG_BYTES);
    new Uint32Array(cc).set([Math.ceil(cam.width / TILE_SIZE) * Math.ceil(cam.height / TILE_SIZE), view, this.cc ? 1 : 0, it % STAT_SLOTS], 0);
    new Float32Array(cc).set([this.cc?.lr ?? 0, 0.9, 0.999, 1e-15, this.cc?.reg ?? 0], 4);
    d.queue.writeBuffer(this.colourU, 0, cc);
  }

  private encodeIteration(view: number): GPUCommandEncoder {
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

  private async readBuffer(src: GPUBuffer, offset: number, bytes: number): Promise<ArrayBuffer> {
    const st = this.ensureStaging(bytes);
    const enc = this.device.createCommandEncoder({ label: 'splat-readback' });
    enc.copyBufferToBuffer(src, offset, st, 0, Math.ceil(bytes / 4) * 4);
    this.device.queue.submit([enc.finish()]);
    const r = await Promise.race([st.mapAsync(GPUMapMode.READ, 0, Math.ceil(bytes / 4) * 4).then(() => 'ok' as const), this.lost.then(() => 'lost' as const)]);
    if (r === 'lost') throw new Error('GPU device lost during readback');
    const out = st.getMappedRange(0, Math.ceil(bytes / 4) * 4).slice(0);
    st.unmap();
    return out;
  }

  /**
   * Runs `n` iterations (one command buffer each, awaited in turn). Returns the loss of the
   * last one; MCMC refinements and intersection-capacity checks happen in between.
   */
  async step(n = 1): Promise<SplatStepResult> {
    if (!this.initialised) throw new Error('SplatTrainer: init first');
    const t0 = performance.now();
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
      this.device.queue.submit([enc.finish()]);
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
    const s = new Float32Array(await this.readBuffer(this.statsBuf, ((this.iter - 1) % STAT_SLOTS) * 16, 16));
    const nv = Math.max(1, s[2]);
    const l1 = s[0] / (3 * nv);
    const ssim = s[1] / (3 * nv);
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
    d.queue.writeBuffer(this.relocU, 0, new Uint32Array([plan.nCopies, plan.nUpdates, this.capacity, 0]));
    const enc = d.createCommandEncoder({ label: 'splat-mcmc' });
    const pass = enc.beginComputePass({ label: 'mcmc' });
    pass.setPipeline(this.pipes.relocCopy);
    pass.setBindGroup(0, this.bgReloc[0]);
    pass.dispatchWorkgroups(Math.ceil(plan.nCopies / 64));
    pass.setPipeline(this.pipes.relocUpdate);
    pass.setBindGroup(0, this.bgReloc[1]);
    pass.dispatchWorkgroups(Math.ceil(plan.nUpdates / 64));
    pass.end();
    d.queue.submit([enc.finish()]);
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
  async readModel(): Promise<SplatModel> {
    const lay = paramOffsets(this.capacity, this.shDegree);
    const d = new Float32Array(await this.readBuffer(this.params, 0, lay.total * 4));
    return unpackParams(d, this.count, this.capacity, this.shDegree);
  }

  /** The current model as an INRIA-layout splat PLY (ply.ts). */
  async exportPLY(comments: string[] = []): Promise<Uint8Array> {
    return encodeSplatPLY(await this.readModel(), { antialiased: this.antialiased, comments });
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

  resetProfile(): void {
    this.gpuMs = {};
    this.profiledIterations = 0;
  }

  /**
   * Test hook: parameter gradients of L = Σ dLdC · C (RGB per pixel) for camera `cam` through
   * the GPU rasteriser backward and projection backward (no loss kernels, no update).
   */
  async debugGradients(cam: SplatCamera, dLdC: Float32Array): Promise<{ grad2d: Float32Array; grads: SplatModel }> {
    const d = this.device;
    const f = this.fwd;
    f.setFrame(cam, this.count, this.background);
    const px = cam.width * cam.height;
    const g4 = new Float32Array(4 * px);
    for (let i = 0; i < px; i++) g4.set([dLdC[3 * i], dLdC[3 * i + 1], dLdC[3 * i + 2], 0], 4 * i);
    d.queue.writeBuffer(this.dLdC, 0, g4);
    const lay = paramOffsets(this.capacity, this.shDegree);
    if (!this.gradsOut) this.gradsOut = makeBuffer(d, lay.total * 4, 'splat-grads-out');
    const g2Bytes = this.count * 36;
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
    return { grad2d: g2, grads: unpackParams(gr, this.count, this.capacity, this.shDegree) };
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

  destroy(): void {
    if (!this.initialised) return;
    this.fwd.destroy();
    for (const b of [this.params, this.adamM, this.adamV, this.grad2d, this.target, this.pqr, this.dLdC, this.lossPartial, this.colourPartial, this.colour, this.statsBuf, this.optU, this.lossU, this.colourU, this.relocU, ...this.images])
      b.destroy();
    this.relocList?.destroy();
    this.staging?.destroy();
    this.gradsOut?.destroy();
    this.profiler?.destroy();
    this.images = [];
    this.initialised = false;
  }
}
