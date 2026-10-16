/**
 * WebGPU SIFT producing RootSIFT features identical in layout and conventions to
 * extractSiftCPU (sift-cpu.ts): same options, scale space, refinement, orientation,
 * descriptor, RootSIFT quantisation, spatially distributed selection (the selection
 * itself runs on the CPU with the same code) and full-resolution coordinate mapping.
 *
 * GPU stages (one extraction):
 *   1. upload (Uint8 / Float32 gray, or an ImageBitmap via copyExternalImageToTexture
 *      with the Rec.709 luma computed on the GPU), area-resample to the processed size;
 *   2. optional 2x upsample (octave -1), Gaussian scale space per octave with
 *      separable shared-memory blurs (S + 3 levels kept per octave), decimation;
 *   3. DoG extrema + Taylor refinement + contrast / edge tests, candidates appended
 *      through an atomic counter (capacity guarded, grown and re-run if exceeded);
 *   4. readback of the compact candidate list -> CPU: restore the CPU detection order,
 *      de-duplicate, distributed preselection, describe order;
 *   5. orientation (one workgroup per keypoint) and descriptors (one workgroup per
 *      oriented slot, fixed-point workgroup atomics, RootSIFT + quantisation on the
 *      GPU) -> one readback -> CPU assembles the FeatureSet and applies the final
 *      distributed selection.
 * Work is split into several submissions by a cost budget (iOS GPU watchdog), the
 * pyramids of the most recent processed sizes stay allocated between calls.
 */

import type { FeatureSet, GrayImage } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';
import type { GPUContext } from '../gpu/device';
import {
  BudgetEncoder,
  FeatureInputError,
  GPUFeatureError,
  GPUWorkDroppedError,
  GrowBuffer,
  ParamBlock,
  PARAM_SLOT,
  STAMP_WGSL,
  Staging,
  WorkStamp,
  compilePipeline,
  dispatch2D,
  errorScope,
} from './gpu-common';
import { gaussianHalfKernel } from './image';
import {
  BLUR_H_DIRECT_WGSL,
  BLUR_H_MAX_RADIUS,
  BLUR_H_WGSL,
  BLUR_V_DIRECT_WGSL,
  BLUR_V_TILED_MAX_RADIUS,
  BLUR_V_TILED_WGSL,
  CANDIDATE_WORDS,
  DECIMATE_WGSL,
  DESCRIBE_WGSL,
  DETECT_WGSL,
  ORIENT_WGSL,
  UPSAMPLE_WGSL,
  descriptorFixedPoint,
  resizeShader,
} from './shaders/sift-wgsl';
import { SIFT_NOMINAL_BLUR, planSift, resolveSiftOptions } from './sift-common';
import type { SiftOptions, SiftPlan } from './sift-common';
import {
  assembleFeatures,
  decodeCandidates,
  describeOrder,
  emptyFeatures,
  featureParity,
  packResizeTables,
  preselectCandidates,
  resolveFullSize,
  scaleSpaceSigmas,
  selfTestImage,
} from './sift-gpu-host';
import { extractSiftCPU } from './sift-cpu';

/** Extra (GPU-only) configuration of SiftGPU. */
export interface SiftGPUConfig {
  /** Number of processed image sizes whose pyramids stay allocated between calls. Default 2. */
  maxCachedSizes?: number;
  /** Work per command buffer before submitting and waiting (≈ pixel-taps). Default 2e8. */
  submitBudget?: number;
  /** Initial candidate capacity (records, >= 256). Default 65536; grows automatically. */
  initialCandidateCapacity?: number;
  /**
   * Candidate records read back together with the counter (one round trip when the
   * count fits; adapts to 1.25x the last count). Default 8192.
   */
  initialReadbackGuess?: number;
  /**
   * Blur kernels: 'tiled' = shared-memory row / column tiles (default on hardware
   * adapters), 'direct' = direct global loads (default on software adapters such as
   * SwiftShader, whose workgroup-barrier emulation makes tiles ~10x slower). Both give
   * identical results.
   */
  blur?: 'auto' | 'tiled' | 'direct';
  /**
   * Diagnostics: submit and await every dispatch separately and record the wall time
   * per kernel in `lastProfile` (slow; for profiling on a device only).
   */
  profile?: boolean;
  /**
   * Attempts per extraction when the GPU did not execute submitted work (iOS drops
   * command buffers submitted while the page is hidden). Default 3.
   */
  maxAttempts?: number;
  /**
   * @internal Testing hook: command buffers (0-based per attempt) to discard instead of
   * submitting, simulating work dropped by the OS.
   */
  testDropSubmissions?: (attempt: number) => readonly number[];
}

/** Result of SiftGPU.selfTest(). */
export interface SiftSelfTestResult {
  ok: boolean;
  gpuCount: number;
  cpuCount: number;
  /** CPU keypoints with a GPU keypoint within 1 px / 20 % scale. */
  posFrac: number;
  /** Median cosine similarity of corresponding descriptors. */
  medianCos: number;
  ms: number;
  error?: string;
  /**
   * The test could not run because submitted GPU work was dropped (page hidden): it says
   * nothing about the GPU's correctness, run it again later.
   */
  transient?: boolean;
}

/** Image sources accepted by SiftGPU.extractFromBitmap (copyExternalImageToTexture). */
export type SiftBitmapSource = ImageBitmap | ImageData | OffscreenCanvas | HTMLCanvasElement;

/** Stage timings and counts of the most recent extraction (diagnostics). */
export interface SiftGPUTimings {
  totalMs: number;
  uploadMs: number;
  /** Encoding + GPU execution of resize, pyramid and detection, incl. readback wait. */
  detectMs: number;
  /** Host work: candidate decoding, de-duplication, preselection, describe order. */
  selectMs: number;
  /** Orientation + descriptor GPU work incl. readback. */
  describeMs: number;
  /** Final assembly + selection on the CPU. */
  assembleMs: number;
  candidates: number;
  preselected: number;
  oriented: number;
  submits: number;
  readbacks: number;
  /** Attempts needed (> 1 when submitted GPU work was dropped and redone). */
  attempts: number;
}

interface Pipelines {
  resizeU8: GPUComputePipeline;
  resizeF32: GPUComputePipeline;
  resizeRGBA: GPUComputePipeline;
  upsample: GPUComputePipeline;
  blurH: GPUComputePipeline;
  blurHD: GPUComputePipeline;
  blurVT: GPUComputePipeline;
  blurVD: GPUComputePipeline;
  decimate: GPUComputePipeline;
  detect: GPUComputePipeline;
  orient: GPUComputePipeline;
  describe: GPUComputePipeline;
  stamp: GPUComputePipeline;
}

type Cmd =
  | { kind: 'dispatch'; pipe: GPUComputePipeline; bg: GPUBindGroup; x: number; y: number; z: number; cost: number }
  | { kind: 'copy'; src: GPUBuffer; srcOff: number; dst: GPUBuffer; dstOff: number; size: number };

interface OctaveBuf {
  buf: GPUBuffer;
  w: number;
  h: number;
  /** Floats per plane. */
  N: number;
}

interface Pyramid {
  key: string;
  firstOctave: number;
  procWidth: number;
  procHeight: number;
  oct: OctaveBuf[];
  temp: GPUBuffer;
  params: GPUBuffer;
  cmds: Cmd[];
  detectParams: GPUBuffer | null;
  detectCmds: Cmd[];
  detectCapacity: number;
  detectCandBuf: GPUBuffer | null;
  bytes: number;
}

interface InputState {
  key: string;
  width: number;
  height: number;
  kind: 'u8' | 'f32' | 'rgba';
  procWidth: number;
  procHeight: number;
  tables: GPUBuffer;
  tablesTaps: [number, number];
  params: GPUBuffer;
  buf: GPUBuffer | null;
  tex: GPUTexture | null;
  bytes: number;
}

interface InputDesc {
  kind: 'u8' | 'f32' | 'rgba';
  width: number;
  height: number;
  gray?: GrayImage;
  bitmap?: SiftBitmapSource;
}

const STORAGE = GPUBufferUsage.STORAGE;
const COPY_DST = GPUBufferUsage.COPY_DST;
const COPY_SRC = GPUBufferUsage.COPY_SRC;
const UNIFORM = GPUBufferUsage.UNIFORM;

function ceilDiv(a: number, b: number): number {
  return Math.floor((a + b - 1) / b);
}

/** WebGPU SIFT / RootSIFT extractor (see file header). Create once, reuse, destroy(). */
export class SiftGPU {
  /** Resolved options (same semantics as extractSiftCPU). */
  readonly options: Required<SiftOptions>;
  /** Timings of the most recent extraction. */
  lastTimings: SiftGPUTimings | null = null;
  /** Per-kernel wall times (ms) of the most recent extraction when `profile` is enabled. */
  lastProfile: Map<string, number> | null = null;

  private readonly device: GPUDevice;
  private readonly lostPromise: Promise<unknown>;
  private isLost = false;
  private destroyed = false;
  private readonly maxCached: number;
  private readonly budget: number;
  private readonly profile: boolean;
  private readonly tiledBlur: boolean;
  private readonly maxAttempts: number;
  private readonly testDrop: ((attempt: number) => readonly number[]) | null;
  private readonly stamp: WorkStamp;
  private readonly pipes: Pipelines;
  private readonly kernBuf: GPUBuffer;
  /** Kernel table: [offset, radius] per entry (radius -1 = copy). 0/1 = base (fo 0 / -1), 1 + i = inc[i]. */
  private readonly kernTab: [number, number][];
  private pyramids: Pyramid[] = [];
  private inputs: InputState[] = [];
  private candCapacity: number;
  private candBuf: GPUBuffer;
  private readonly counterBuf: GPUBuffer;
  private readonly kpBuf: GPUBuffer;
  private readonly oriBuf: GPUBuffer;
  private readonly descBuf: GPUBuffer;
  private readonly dynParams: GrowBuffer;
  private readonly staging: Staging;
  private readonly slotCap: number;
  private candGuess: number;
  private readonly minGuess: number;
  private chain: Promise<unknown> = Promise.resolve();
  /** An extraction is running (its GPU resources must not be freed under it). */
  private busy = false;
  /** releaseCaches() was called while busy: free the caches when the extraction ends. */
  private releasePending = false;
  /** GPU resources freed (destroy() frees them when the running extraction ends). */
  private freed = false;

  private constructor(
    ctx: GPUContext,
    o: Required<SiftOptions>,
    pipes: Pipelines,
    kern: { data: Float32Array; tab: [number, number][] },
    cfg: SiftGPUConfig,
  ) {
    const device = ctx.device;
    this.device = device;
    this.options = o;
    this.pipes = pipes;
    this.maxCached = Math.max(1, cfg.maxCachedSizes ?? 2);
    this.budget = cfg.submitBudget ?? 2e8;
    this.profile = !!cfg.profile;
    this.maxAttempts = Math.max(1, cfg.maxAttempts ?? 3);
    this.testDrop = cfg.testDropSubmissions ?? null;
    this.tiledBlur = cfg.blur === 'tiled' || ((cfg.blur ?? 'auto') === 'auto' && !ctx.isSoftware);
    this.lostPromise = device.lost.then(() => {
      this.isLost = true;
    });
    this.kernBuf = device.createBuffer({ size: Math.max(16, kern.data.byteLength), usage: STORAGE | COPY_DST, label: 'sift-kernels' });
    device.queue.writeBuffer(this.kernBuf, 0, kern.data);
    this.kernTab = kern.tab;
    this.candCapacity = Math.max(256, cfg.initialCandidateCapacity ?? 65536);
    this.candGuess = Math.max(16, cfg.initialReadbackGuess ?? 8192);
    this.minGuess = Math.min(4096, this.candGuess);
    this.candBuf = device.createBuffer({ size: this.candCapacity * CANDIDATE_WORDS * 4, usage: STORAGE | COPY_SRC, label: 'sift-candidates' });
    // word 0: candidate counter; words 1-2: work stamp (attempt id, executed command buffers)
    this.counterBuf = device.createBuffer({ size: 16, usage: STORAGE | COPY_SRC | COPY_DST, label: 'sift-counter' });
    this.stamp = new WorkStamp(device, pipes.stamp, this.counterBuf, 1);
    const cap = Math.max(1, o.maxFeatures);
    this.slotCap = cap;
    const mo = o.maxOrientations;
    this.kpBuf = device.createBuffer({ size: cap * 16, usage: STORAGE | COPY_DST, label: 'sift-keypoints' });
    this.oriBuf = device.createBuffer({ size: Math.ceil((cap * (1 + mo) * 4) / 16) * 16, usage: STORAGE | COPY_SRC, label: 'sift-orientations' });
    this.descBuf = device.createBuffer({ size: cap * mo * DESCRIPTOR_SIZE, usage: STORAGE | COPY_SRC, label: 'sift-descriptors' });
    this.dynParams = new GrowBuffer(device, UNIFORM | COPY_DST, 'sift-dyn-params');
    this.staging = new Staging(device, 'sift-staging', device.lost);
  }

  /**
   * Compiles all pipelines up front (pipeline compilation blocks the queue on Safari:
   * create the extractor before the capture loop starts). Throws GPUFeatureError when
   * the options need unsupported blur radii or compilation fails.
   */
  static async create(ctx: GPUContext, opts?: Partial<SiftOptions>, cfg: SiftGPUConfig = {}): Promise<SiftGPU> {
    const o = resolveSiftOptions(opts);
    const device = ctx.device;
    // Blur kernels: base for baseBlur 0.5 (octave 0 first) and 1.0 (octave -1 first), then inc[1..S+2].
    const kernels: Float32Array[] = [];
    const tab: [number, number][] = [];
    const add = (sigma: number) => {
      if (!(sigma > 0)) {
        tab.push([0, -1]);
        return;
      }
      const k = gaussianHalfKernel(sigma);
      const r = k.length - 1;
      if (r > BLUR_H_MAX_RADIUS) throw new GPUFeatureError(`SiftGPU: blur radius ${r} exceeds ${BLUR_H_MAX_RADIUS} (sigma ${sigma.toFixed(2)})`);
      let off = 0;
      for (const kk of kernels) off += kk.length;
      kernels.push(k);
      tab.push([off, r]);
    };
    const s0 = scaleSpaceSigmas(o, SIFT_NOMINAL_BLUR);
    const s1 = scaleSpaceSigmas(o, 2 * SIFT_NOMINAL_BLUR);
    add(s0.sigmaBase);
    add(s1.sigmaBase);
    for (let i = 1; i <= o.scalesPerOctave + 2; i++) add(s0.inc[i]);
    let total = 0;
    for (const k of kernels) total += k.length;
    const data = new Float32Array(Math.max(4, total));
    let off = 0;
    for (const k of kernels) {
      data.set(k, off);
      off += k.length;
    }
    const lost = device.lost;
    const names: [keyof Pipelines, string][] = [
      ['resizeU8', resizeShader('u8')],
      ['resizeF32', resizeShader('f32')],
      ['resizeRGBA', resizeShader('rgba')],
      ['upsample', UPSAMPLE_WGSL],
      ['blurH', BLUR_H_WGSL],
      ['blurHD', BLUR_H_DIRECT_WGSL],
      ['blurVT', BLUR_V_TILED_WGSL],
      ['blurVD', BLUR_V_DIRECT_WGSL],
      ['decimate', DECIMATE_WGSL],
      ['detect', DETECT_WGSL],
      ['orient', ORIENT_WGSL],
      ['describe', DESCRIBE_WGSL],
      ['stamp', STAMP_WGSL],
    ];
    const compiled = await Promise.all(names.map(([n, code]) => compilePipeline(device, `sift-${n}`, code, lost)));
    const pipes = {} as Pipelines;
    names.forEach(([n], i) => {
      pipes[n] = compiled[i];
    });
    return new SiftGPU(ctx, o, pipes, { data, tab }, cfg);
  }

  /**
   * Extracts RootSIFT features (same contract as extractSiftCPU).
   * @param img gray image, Uint8 0..255 or Float32 0..1
   * @param fullWidth width of the full-resolution frame of the output keypoints (default img.width)
   * @param fullHeight height of that frame (default img.height)
   */
  extract(img: GrayImage, fullWidth?: number, fullHeight?: number): Promise<FeatureSet> {
    const d = img.data as unknown;
    if (!(d instanceof Float32Array || d instanceof Uint8Array || d instanceof Uint8ClampedArray)) {
      // Any other typed array would be uploaded as raw bytes (garbage features).
      return Promise.reject(new Error('SiftGPU.extract: image data must be a Uint8Array (0..255) or Float32Array (0..1)'));
    }
    if (!(Number.isInteger(img.width) && Number.isInteger(img.height) && img.width >= 1 && img.height >= 1)) {
      // (A fractional size reached createBuffer as a TypeError, reported as a GPU failure.)
      return Promise.reject(new Error(`SiftGPU.extract: invalid image size ${img.width}x${img.height}`));
    }
    if (img.data.length < img.width * img.height) return Promise.reject(new Error('SiftGPU.extract: image data shorter than width*height'));
    let FW: number;
    let FH: number;
    try {
      [FW, FH] = resolveFullSize(img.width, img.height, fullWidth, fullHeight, 'SiftGPU.extract');
    } catch (e) {
      return Promise.reject(e);
    }
    const kind = img.data instanceof Float32Array ? 'f32' : 'u8';
    return this.exclusive(() => this.run({ kind, width: img.width, height: img.height, gray: img }, FW, FH));
  }

  /**
   * Extracts features from an ImageBitmap (or canvas / ImageData) entirely on the GPU:
   * copyExternalImageToTexture, Rec.709 luma of the sRGB values (as rgbaToGray) and
   * area resampling in one kernel — no CPU pixel work.
   */
  extractFromBitmap(bitmap: SiftBitmapSource, fullWidth?: number, fullHeight?: number): Promise<FeatureSet> {
    const w = bitmap.width;
    const h = bitmap.height;
    if (!(w >= 1 && h >= 1)) return Promise.reject(new Error(`SiftGPU.extractFromBitmap: invalid bitmap size ${w}x${h}`));
    if (typeof ImageData !== 'undefined' && bitmap instanceof ImageData && bitmap.data.length < 4 * w * h) {
      // Detached pixel buffer (transferred to a worker): Chromium's copyExternalImageToTexture
      // crashes the renderer on it, so reject it as an input error before touching the GPU.
      return Promise.reject(new Error(`SiftGPU.extractFromBitmap: ImageData pixels are detached or shorter than ${w}x${h}x4`));
    }
    let FW: number;
    let FH: number;
    try {
      [FW, FH] = resolveFullSize(w, h, fullWidth, fullHeight, 'SiftGPU.extractFromBitmap');
    } catch (e) {
      return Promise.reject(e);
    }
    return this.exclusive(() => this.run({ kind: 'rgba', width: w, height: h, bitmap }, FW, FH));
  }

  /** GPU memory currently held (bytes, approximate). */
  memoryBytes(): number {
    let b = this.candCapacity * CANDIDATE_WORDS * 4 + this.kpBuf.size + this.oriBuf.size + this.descBuf.size + this.dynParams.size;
    for (const p of this.pyramids) b += p.bytes;
    for (const i of this.inputs) b += i.bytes;
    return b;
  }

  /**
   * Frees the cached per-size pyramids and input buffers (pipelines are kept). While an
   * extraction is running, the caches are freed when it ends (freeing buffers that its
   * remaining command buffers use would make them invalid).
   */
  releaseCaches(): void {
    if (this.busy) {
      this.releasePending = true;
      return;
    }
    this.freeCaches();
  }

  /**
   * Frees all GPU resources. Later calls reject with GPUFeatureError; an extraction that
   * is already running completes first (its resources are freed when it ends).
   */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    if (!this.busy) this.freeAll();
  }

  private freeCaches(): void {
    this.releasePending = false;
    for (const p of this.pyramids) this.destroyPyramid(p);
    for (const i of this.inputs) this.destroyInput(i);
    this.pyramids = [];
    this.inputs = [];
  }

  private freeAll(): void {
    if (this.freed) return;
    this.freed = true;
    this.freeCaches();
    this.kernBuf.destroy();
    this.candBuf.destroy();
    this.counterBuf.destroy();
    this.kpBuf.destroy();
    this.oriBuf.destroy();
    this.descBuf.destroy();
    this.dynParams.destroy();
    this.staging.destroy();
    this.stamp.destroy();
  }

  /**
   * Compares this extractor with extractSiftCPU on a small synthetic image (guards
   * against GPU driver / compiler bugs on untested devices). ok = feature counts within
   * 5 %, >= 90 % of CPU keypoints reproduced (1 px, 20 % scale), median descriptor
   * cosine >= 0.97. Never throws.
   */
  async selfTest(): Promise<SiftSelfTestResult> {
    const t0 = performance.now();
    const img = selfTestImage();
    try {
      const g = await this.extract(img);
      const c = extractSiftCPU(img, this.options);
      const p = featureParity(c, g);
      // Enough reference features to be conclusive; with maxFeatures < 10 the CPU can only
      // return maxFeatures of them (a fixed "> 10" made every such extractor fail).
      const minCount = Math.max(1, Math.min(10, this.options.maxFeatures));
      const ok = Math.abs(g.count - c.count) <= Math.max(2, 0.05 * c.count) && p.posFrac >= 0.9 && p.medianCos >= 0.97 && c.count >= minCount;
      return { ok, gpuCount: g.count, cpuCount: c.count, posFrac: p.posFrac, medianCos: p.medianCos, ms: performance.now() - t0 };
    } catch (e) {
      return { ok: false, gpuCount: 0, cpuCount: 0, posFrac: 0, medianCos: 0, ms: performance.now() - t0, error: (e as Error)?.message ?? String(e), transient: e instanceof GPUWorkDroppedError };
    }
  }

  // -------------------------------------------------------------------------

  private exclusive<T>(fn: () => Promise<T>): Promise<T> {
    const guarded = async (): Promise<T> => {
      this.busy = true;
      try {
        return await fn();
      } finally {
        this.busy = false;
        // Deferred releaseCaches() / destroy() (see there).
        if (this.destroyed) this.freeAll();
        else if (this.releasePending) this.freeCaches();
      }
    };
    const r = this.chain.then(guarded, guarded);
    this.chain = r.then(
      () => undefined,
      () => undefined,
    );
    return r;
  }

  private checkUsable(): void {
    if (this.destroyed) throw new GPUFeatureError('SiftGPU: used after destroy()');
    if (this.isLost) throw new GPUFeatureError('SiftGPU: GPU device lost (recreate via getGPU() + SiftGPU.create)');
  }

  private async run(input: InputDesc, FW: number, FH: number): Promise<FeatureSet> {
    const o = this.options;
    const tStart = performance.now();
    if (o.maxFeatures === 0) return emptyFeatures(FW, FH);
    const plan = planSift(input.width, input.height, o, FW, FH);
    if (plan.octaves.length === 0) return emptyFeatures(FW, FH);
    const device = this.device;
    for (let attempt = 0; ; attempt++) {
      this.checkUsable();
      const pop = errorScope(device, 'SiftGPU');
      let result: FeatureSet | null = null;
      let failure: unknown = null;
      try {
        this.stamp.begin();
        result = await this.runGPU(input, plan, tStart, attempt);
      } catch (e) {
        failure = e;
      }
      try {
        await pop();
      } catch (e) {
        // A validation / OOM error outranks the stamp mismatch it causes (an invalid command
        // buffer is not executed): it is NOT transient dropped work — retrying would reuse the
        // same broken resources, and the pipeline would never switch to the CPU for good.
        if (!(failure instanceof FeatureInputError)) failure = e;
      }
      if (!failure) return result!;
      if (failure instanceof GPUWorkDroppedError && attempt + 1 < this.maxAttempts && !this.isLost) continue;
      if (failure instanceof FeatureInputError) throw failure;
      // Hard GPU failure: drop the cached per-size resources (possibly invalid objects) so a
      // later call starts from fresh allocations. Nothing of this call is still in flight.
      if (!(failure instanceof GPUWorkDroppedError)) this.freeCaches();
      if (failure instanceof GPUFeatureError) throw failure;
      if (this.isLost) throw new GPUFeatureError('SiftGPU: GPU device lost', { cause: failure });
      throw new GPUFeatureError(`SiftGPU: ${(failure as Error)?.message ?? failure}`, { cause: failure });
    }
  }

  private async runGPU(input: InputDesc, plan: SiftPlan, tStart: number, attempt: number): Promise<FeatureSet> {
    const o = this.options;
    const S = o.scalesPerOctave;
    const device = this.device;
    const T: SiftGPUTimings = {
      totalMs: 0,
      uploadMs: 0,
      detectMs: 0,
      selectMs: 0,
      describeMs: 0,
      assembleMs: 0,
      candidates: 0,
      preselected: 0,
      oriented: 0,
      submits: 0,
      readbacks: 0,
      attempts: attempt + 1,
    };
    let t = performance.now();
    const pyr = this.getPyramid(plan);
    const inp = this.getInput(input, plan);
    const oct0 = pyr.oct[0];
    const procPlane = plan.firstOctave < 0 ? S + 1 : S + 2;

    // --- upload ------------------------------------------------------------
    let direct = false;
    if (input.kind === 'rgba') {
      try {
        device.queue.copyExternalImageToTexture({ source: input.bitmap! }, { texture: inp.tex! }, [input.width, input.height]);
      } catch (e) {
        // Synchronous exceptions concern the source (closed / detached / not origin-clean);
        // GPU-side validation errors go to the error scope instead.
        if (this.isLost) throw e;
        throw new FeatureInputError(`SiftGPU.extractFromBitmap: cannot read the image source: ${(e as Error)?.message ?? e}`);
      }
    } else {
      const d = input.gray!.data;
      const n = input.width * input.height;
      if (input.kind === 'f32') {
        const f = d as Float32Array;
        if (inp.procWidth === input.width && inp.procHeight === input.height) {
          device.queue.writeBuffer(oct0.buf, procPlane * oct0.N * 4, f.buffer, f.byteOffset, n * 4);
          direct = true;
        } else {
          device.queue.writeBuffer(inp.buf!, 0, f.buffer, f.byteOffset, n * 4);
        }
      } else {
        const n4 = n & ~3;
        if (n4 > 0) device.queue.writeBuffer(inp.buf!, 0, d.buffer, d.byteOffset, n4);
        if (n4 < n) {
          const tail = new Uint8Array(4);
          for (let i = n4; i < n; i++) tail[i - n4] = d[i];
          device.queue.writeBuffer(inp.buf!, n4, tail);
        }
      }
    }
    T.uploadMs = performance.now() - t;
    t = performance.now();

    // --- resize, pyramid, detection ----------------------------------------
    this.lastProfile = this.profile ? new Map() : null;
    const drop = this.testDrop ? new Set(this.testDrop(attempt)) : null;
    const be = new BudgetEncoder(device, this.budget, this.lostPromise, 'sift', this.lastProfile, this.stamp, drop);
    if (!direct) {
      const pipe = input.kind === 'u8' ? this.pipes.resizeU8 : input.kind === 'f32' ? this.pipes.resizeF32 : this.pipes.resizeRGBA;
      const bg = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inp.params, offset: 0, size: PARAM_SLOT } },
          { binding: 1, resource: { buffer: inp.tables } },
          { binding: 2, resource: { buffer: oct0.buf } },
          { binding: 3, resource: input.kind === 'rgba' ? inp.tex!.createView() : { buffer: inp.buf! } },
        ],
      });
      const cost = inp.procWidth * inp.procHeight * inp.tablesTaps[0] * inp.tablesTaps[1];
      await be.dispatch(pipe, bg, ceilDiv(inp.procWidth, 16), ceilDiv(inp.procHeight, 16), 1, cost);
    }
    await this.encodeCmds(be, pyr.cmds);
    this.ensureDetect(pyr);
    let count = await this.detect(be, pyr, T);
    if (count > this.candCapacity) {
      // Capacity exceeded: grow and re-run detection on the (still valid) pyramid.
      this.growCandidates(Math.ceil(count * 1.3) + 1024);
      this.ensureDetect(pyr);
      count = await this.detect(be, pyr, T);
    }
    const n0 = Math.min(count, this.candCapacity);
    let raw: Uint32Array;
    if (n0 <= this.lastGuess) {
      raw = this.lastRaw!.subarray(0, n0 * CANDIDATE_WORDS);
    } else {
      const enc = be.encoder();
      const bytes = 16 + n0 * CANDIDATE_WORDS * 4;
      this.staging.ensure(bytes);
      enc.copyBufferToBuffer(this.counterBuf, 0, this.staging.buffer, 0, 16);
      enc.copyBufferToBuffer(this.candBuf, 0, this.staging.buffer, 16, n0 * CANDIDATE_WORDS * 4);
      be.submit();
      const ab = await this.staging.read(bytes);
      T.readbacks++;
      this.stamp.check(new Uint32Array(ab, 4, 2), be.submits, 'candidates');
      raw = new Uint32Array(ab, 16, n0 * CANDIDATE_WORDS);
    }
    this.candGuess = Math.max(this.minGuess, Math.ceil(count * 1.25) + 512);
    T.detectMs = performance.now() - t;
    t = performance.now();

    // --- host: order, de-duplicate, preselect --------------------------------
    const cands = decodeCandidates(raw, n0, plan, S, o.sigma);
    this.lastRaw = null;
    T.candidates = cands.n;
    if (cands.n === 0) {
      T.submits = be.submits;
      T.totalMs = performance.now() - tStart;
      this.lastTimings = T;
      return emptyFeatures(plan.fullWidth, plan.fullHeight);
    }
    const sel = preselectCandidates(cands, plan, o);
    const ord = describeOrder(cands, sel.pre, S);
    const np = ord.order.length;
    T.preselected = np;
    if (np > this.slotCap) throw new GPUFeatureError('SiftGPU: internal error (preselection exceeds capacity)');
    T.selectMs = performance.now() - t;
    t = performance.now();

    // --- orientation + descriptors ------------------------------------------
    const mo = o.maxOrientations;
    device.queue.writeBuffer(this.kpBuf, 0, ord.kp);
    const pb = new ParamBlock(2 * ord.groups.length + 2);
    const slots: [number, number][] = [];
    const fix = descriptorFixedPoint(o.sigma, S);
    for (const g of ord.groups) {
      const oc = pyr.oct[g.octIdx];
      const u = [oc.w, oc.h, g.level * oc.N, g.start, g.end - g.start, mo, this.slotCap];
      slots.push([pb.add(u), pb.add(u, [fix])]);
    }
    this.dynParams.ensure(pb.byteLength);
    device.queue.writeBuffer(this.dynParams.buffer!, 0, pb.bytes());
    for (let gi = 0; gi < ord.groups.length; gi++) {
      const g = ord.groups[gi];
      const oc = pyr.oct[g.octIdx];
      const n = g.end - g.start;
      const sig = o.sigma * 2 ** (g.level / S);
      const wo = Math.max(1, Math.floor(4.5 * sig));
      const wd = Math.floor(Math.SQRT2 * 3 * sig * 2.5 + 0.5);
      const bgO = device.createBindGroup({
        layout: this.pipes.orient.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.dynParams.buffer!, offset: slots[gi][0], size: PARAM_SLOT } },
          { binding: 1, resource: { buffer: oc.buf } },
          { binding: 2, resource: { buffer: this.kpBuf } },
          { binding: 3, resource: { buffer: this.oriBuf } },
        ],
      });
      const [ox, oy] = dispatch2D(n);
      await be.dispatch(this.pipes.orient, bgO, ox, oy, 1, n * (2 * wo + 1) ** 2 * 8);
      const bgD = device.createBindGroup({
        layout: this.pipes.describe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.dynParams.buffer!, offset: slots[gi][1], size: PARAM_SLOT } },
          { binding: 1, resource: { buffer: oc.buf } },
          { binding: 2, resource: { buffer: this.kpBuf } },
          { binding: 3, resource: { buffer: this.oriBuf } },
          { binding: 4, resource: { buffer: this.descBuf } },
        ],
      });
      const [dx, dy] = dispatch2D(n * mo);
      await be.dispatch(this.pipes.describe, bgD, dx, dy, 1, n * (2 * wd + 1) ** 2 * 6);
    }
    const cntBytes = np * 4;
    const angBytes = np * mo * 4;
    const descBytes = np * mo * DESCRIPTOR_SIZE;
    const total = cntBytes + angBytes + descBytes + 16;
    this.staging.ensure(total);
    const enc = be.encoder();
    enc.copyBufferToBuffer(this.oriBuf, 0, this.staging.buffer, 0, cntBytes);
    enc.copyBufferToBuffer(this.oriBuf, this.slotCap * 4, this.staging.buffer, cntBytes, angBytes);
    enc.copyBufferToBuffer(this.descBuf, 0, this.staging.buffer, cntBytes + angBytes, descBytes);
    enc.copyBufferToBuffer(this.counterBuf, 0, this.staging.buffer, total - 16, 16);
    be.submit();
    const ab = await this.staging.read(total);
    T.readbacks++;
    this.stamp.check(new Uint32Array(ab, total - 12, 2), be.submits, 'describe');
    T.describeMs = performance.now() - t;
    t = performance.now();

    const counts = new Uint32Array(ab, 0, np);
    const angles = new Float32Array(ab, cntBytes, np * mo);
    const desc = new Uint8Array(ab, cntBytes + angBytes, descBytes);
    let oriented = 0;
    for (let q = 0; q < np; q++) oriented += Math.min(counts[q], mo);
    T.oriented = oriented;
    const fs = assembleFeatures(plan, o, cands, sel, ord, counts, angles, desc);
    T.assembleMs = performance.now() - t;
    T.submits = be.submits;
    T.totalMs = performance.now() - tStart;
    this.lastTimings = T;
    return fs;
  }

  private lastRaw: Uint32Array | null = null;
  private lastGuess = 0;

  /**
   * Runs the detection dispatches (after the pyramid commands already recorded in
   * `be`), copies the counter and the first `candGuess` records to staging, submits and
   * reads back. Returns the total candidate count; the records read are kept in
   * lastRaw / lastGuess.
   */
  private async detect(be: BudgetEncoder, pyr: Pyramid, T: SiftGPUTimings): Promise<number> {
    be.encoder().clearBuffer(this.counterBuf, 0, 4);
    await this.encodeCmds(be, pyr.detectCmds);
    const guess = Math.min(this.candCapacity, this.candGuess);
    const bytes = 16 + guess * CANDIDATE_WORDS * 4;
    this.staging.ensure(bytes);
    const enc = be.encoder();
    enc.copyBufferToBuffer(this.counterBuf, 0, this.staging.buffer, 0, 16);
    enc.copyBufferToBuffer(this.candBuf, 0, this.staging.buffer, 16, guess * CANDIDATE_WORDS * 4);
    be.submit();
    const ab = await this.staging.read(bytes);
    T.readbacks++;
    this.stamp.check(new Uint32Array(ab, 4, 2), be.submits, 'detect');
    const count = new Uint32Array(ab, 0, 1)[0];
    this.lastRaw = new Uint32Array(ab, 16, guess * CANDIDATE_WORDS);
    this.lastGuess = guess;
    return count;
  }

  private async encodeCmds(be: BudgetEncoder, cmds: Cmd[]): Promise<void> {
    for (const c of cmds) {
      if (c.kind === 'dispatch') await be.dispatch(c.pipe, c.bg, c.x, c.y, c.z, c.cost);
      else be.encoder().copyBufferToBuffer(c.src, c.srcOff, c.dst, c.dstOff, c.size);
    }
  }

  private growCandidates(capacity: number): void {
    this.candBuf.destroy();
    this.candCapacity = capacity;
    this.candBuf = this.device.createBuffer({ size: capacity * CANDIDATE_WORDS * 4, usage: STORAGE | COPY_SRC, label: 'sift-candidates' });
  }

  // -------------------------------------------------------------------------
  // Per-size resources
  // -------------------------------------------------------------------------

  private getPyramid(plan: SiftPlan): Pyramid {
    const key = `${plan.procWidth}x${plan.procHeight}|${plan.firstOctave}`;
    const i = this.pyramids.findIndex((p) => p.key === key);
    if (i >= 0) {
      const p = this.pyramids[i];
      this.pyramids.splice(i, 1);
      this.pyramids.push(p);
      return p;
    }
    while (this.pyramids.length >= this.maxCached) this.destroyPyramid(this.pyramids.shift()!);
    const p = this.createPyramid(plan, key);
    this.pyramids.push(p);
    return p;
  }

  private destroyPyramid(p: Pyramid): void {
    for (const oc of p.oct) oc.buf.destroy();
    p.temp.destroy();
    p.params.destroy();
    p.detectParams?.destroy();
  }

  private createPyramid(plan: SiftPlan, key: string): Pyramid {
    const device = this.device;
    const o = this.options;
    const S = o.scalesPerOctave;
    const planes = S + 3;
    const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    // Check every size before allocating anything (a throw after some allocations leaked them).
    for (const oc of plan.octaves) {
      const size = planes * oc.width * oc.height * 4;
      if (size > limit) {
        throw new GPUFeatureError(`SiftGPU: octave buffer ${(size / 2 ** 20).toFixed(0)} MiB exceeds the device limit (${(limit / 2 ** 20).toFixed(0)} MiB); lower maxImageSize`);
      }
    }
    const oct: OctaveBuf[] = [];
    let bytes = 0;
    for (const oc of plan.octaves) {
      const N = oc.width * oc.height;
      const size = planes * N * 4;
      oct.push({ buf: device.createBuffer({ size, usage: STORAGE | COPY_DST | COPY_SRC, label: `sift-octave${oc.octave}` }), w: oc.width, h: oc.height, N });
      bytes += size;
    }
    const N0 = oct[0].N;
    const temp = device.createBuffer({ size: N0 * 4, usage: STORAGE | COPY_DST | COPY_SRC, label: 'sift-temp' });
    bytes += N0 * 4;

    const pb = new ParamBlock(64);
    type Pending = { pipe: GPUComputePipeline; slot: number; buffers: GPUBuffer[]; x: number; y: number; z: number; cost: number };
    const pending: (Pending | Extract<Cmd, { kind: 'copy' }>)[] = [];
    const blur = (src: OctaveBuf, srcPlane: number, dst: OctaveBuf, dstPlane: number, kIdx: number) => {
      const [kOff, r] = this.kernTab[kIdx];
      const { w, h, N } = dst;
      if (r < 0) {
        pending.push({ kind: 'copy', src: src.buf, srcOff: srcPlane * N * 4, dst: temp, dstOff: 0, size: N * 4 });
        pending.push({ kind: 'copy', src: temp, srcOff: 0, dst: dst.buf, dstOff: dstPlane * N * 4, size: N * 4 });
        return;
      }
      const cost = w * h * (2 * r + 1);
      const hSlot = pb.add([w, h, r, kOff, srcPlane * N, 0]);
      if (this.tiledBlur) {
        pending.push({ pipe: this.pipes.blurH, slot: hSlot, buffers: [this.kernBuf, src.buf, temp], x: ceilDiv(w, 256), y: h, z: 1, cost });
      } else {
        pending.push({ pipe: this.pipes.blurHD, slot: hSlot, buffers: [this.kernBuf, src.buf, temp], x: ceilDiv(w, 64), y: ceilDiv(h, 4), z: 1, cost });
      }
      if (this.tiledBlur && r <= BLUR_V_TILED_MAX_RADIUS) {
        pending.push({ pipe: this.pipes.blurVT, slot: pb.add([w, h, r, kOff, 0, dstPlane * N]), buffers: [this.kernBuf, temp, dst.buf], x: ceilDiv(w, 32), y: ceilDiv(h, 32), z: 1, cost });
      } else {
        pending.push({ pipe: this.pipes.blurVD, slot: pb.add([w, h, r, kOff, 0, dstPlane * N]), buffers: [this.kernBuf, temp, dst.buf], x: ceilDiv(w, 64), y: ceilDiv(h, 4), z: 1, cost });
      }
    };
    // Octave 0 base image (the resize writes the processed image into plane procPlane).
    if (plan.firstOctave < 0) {
      const pw = plan.procWidth;
      const ph = plan.procHeight;
      pending.push({
        pipe: this.pipes.upsample,
        slot: pb.add([pw, ph, (S + 1) * N0, (S + 2) * N0]),
        buffers: [oct[0].buf],
        x: ceilDiv(2 * pw, 16),
        y: ceilDiv(2 * ph, 16),
        z: 1,
        cost: N0 * 4,
      });
    }
    blur(oct[0], S + 2, oct[0], 0, plan.firstOctave < 0 ? 1 : 0);
    for (let k = 0; k < oct.length; k++) {
      const oc = oct[k];
      if (k > 0) {
        const prev = oct[k - 1];
        pending.push({
          pipe: this.pipes.decimate,
          slot: pb.add([oc.w, oc.h, prev.w, S * prev.N, 0]),
          buffers: [prev.buf, oc.buf],
          x: ceilDiv(oc.w, 16),
          y: ceilDiv(oc.h, 16),
          z: 1,
          cost: oc.N,
        });
      }
      for (let i = 1; i <= S + 2; i++) blur(oc, i - 1, oc, i, 1 + i);
    }
    const params = device.createBuffer({ size: Math.max(PARAM_SLOT, pb.byteLength), usage: UNIFORM | COPY_DST, label: 'sift-pyramid-params' });
    device.queue.writeBuffer(params, 0, pb.bytes());
    const cmds: Cmd[] = pending.map((c) => {
      if ('kind' in c) return c;
      const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: params, offset: c.slot, size: PARAM_SLOT } }];
      c.buffers.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b } }));
      const bg = device.createBindGroup({ layout: c.pipe.getBindGroupLayout(0), entries });
      return { kind: 'dispatch', pipe: c.pipe, bg, x: c.x, y: c.y, z: c.z, cost: c.cost };
    });
    return {
      key,
      firstOctave: plan.firstOctave,
      procWidth: plan.procWidth,
      procHeight: plan.procHeight,
      oct,
      temp,
      params,
      cmds,
      detectParams: null,
      detectCmds: [],
      detectCapacity: 0,
      detectCandBuf: null,
      bytes,
    };
  }

  /** (Re)builds the detection dispatches of a pyramid for the current candidate buffer. */
  private ensureDetect(pyr: Pyramid): void {
    if (pyr.detectCandBuf === this.candBuf && pyr.detectCapacity === this.candCapacity) return;
    const device = this.device;
    const o = this.options;
    const S = o.scalesPerOctave;
    pyr.detectParams?.destroy();
    const pb = new ParamBlock(pyr.oct.length);
    const slots = pyr.oct.map((oc, k) => pb.add([oc.w, oc.h, S, oc.N, k, this.candCapacity], [0.8 * o.peakThreshold, o.peakThreshold, o.edgeThreshold]));
    const params = device.createBuffer({ size: Math.max(PARAM_SLOT, pb.byteLength), usage: UNIFORM | COPY_DST, label: 'sift-detect-params' });
    device.queue.writeBuffer(params, 0, pb.bytes());
    pyr.detectParams = params;
    pyr.detectCmds = pyr.oct.map((oc, k) => {
      const bg = device.createBindGroup({
        layout: this.pipes.detect.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params, offset: slots[k], size: PARAM_SLOT } },
          { binding: 1, resource: { buffer: oc.buf } },
          { binding: 2, resource: { buffer: this.candBuf } },
          { binding: 3, resource: { buffer: this.counterBuf } },
        ],
      });
      return { kind: 'dispatch' as const, pipe: this.pipes.detect, bg, x: ceilDiv(oc.w - 2, 16), y: ceilDiv(oc.h - 2, 16), z: S, cost: oc.N * S * 4 };
    });
    pyr.detectCandBuf = this.candBuf;
    pyr.detectCapacity = this.candCapacity;
  }

  private getInput(input: InputDesc, plan: SiftPlan): InputState {
    const key = `${input.width}x${input.height}|${input.kind}|${plan.procWidth}x${plan.procHeight}`;
    const i = this.inputs.findIndex((s) => s.key === key);
    if (i >= 0) {
      const s = this.inputs[i];
      this.inputs.splice(i, 1);
      this.inputs.push(s);
      return s;
    }
    while (this.inputs.length >= this.maxCached + 1) this.destroyInput(this.inputs.shift()!);
    const device = this.device;
    const S = this.options.scalesPerOctave;
    const t = packResizeTables(input.width, input.height, plan.procWidth, plan.procHeight);
    const tables = device.createBuffer({ size: Math.max(16, t.data.byteLength), usage: STORAGE | COPY_DST, label: 'sift-resize-tables' });
    device.queue.writeBuffer(tables, 0, t.data);
    const N0 = plan.octaves[0].width * plan.octaves[0].height;
    const procPlane = plan.firstOctave < 0 ? S + 1 : S + 2;
    const pb = new ParamBlock(1);
    pb.add(
      [input.width, input.height, plan.procWidth, plan.procHeight, t.tx, t.ty, t.oxi, t.oxw, t.oyi, t.oyw, procPlane * N0],
      [input.kind === 'u8' ? 1 / 255 : 1],
    );
    const params = device.createBuffer({ size: PARAM_SLOT, usage: UNIFORM | COPY_DST, label: 'sift-resize-params' });
    device.queue.writeBuffer(params, 0, pb.bytes());
    let buf: GPUBuffer | null = null;
    let tex: GPUTexture | null = null;
    let bytes = t.data.byteLength + PARAM_SLOT;
    const n = input.width * input.height;
    if (input.kind === 'rgba') {
      tex = device.createTexture({
        size: [input.width, input.height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        label: 'sift-input-rgba',
      });
      bytes += n * 4;
    } else if (!(input.kind === 'f32' && plan.procWidth === input.width && plan.procHeight === input.height)) {
      // (Float32 input at the processed size is written straight into the pyramid.)
      const size = input.kind === 'f32' ? n * 4 : Math.ceil(n / 4) * 4;
      buf = device.createBuffer({ size: Math.max(16, size), usage: STORAGE | COPY_DST, label: `sift-input-${input.kind}` });
      bytes += size;
    }
    const s: InputState = {
      key,
      width: input.width,
      height: input.height,
      kind: input.kind,
      procWidth: plan.procWidth,
      procHeight: plan.procHeight,
      tables,
      tablesTaps: [t.tx, t.ty],
      params,
      buf,
      tex,
      bytes,
    };
    this.inputs.push(s);
    return s;
  }

  private destroyInput(s: InputState): void {
    s.tables.destroy();
    s.params.destroy();
    s.buf?.destroy();
    s.tex?.destroy();
  }
}
