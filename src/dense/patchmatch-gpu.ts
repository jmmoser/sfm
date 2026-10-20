/**
 * WebGPU PatchMatch MVS (ACMH view selection + ACMM geometric consistency).
 * Same algorithm as patchMatchCPU (see patchmatch-cpu.ts / patchmatch-wgsl.ts).
 *
 * iOS scheduling rules (ARCHITECTURE §2.1):
 *  - every pass (init, red, black) is split into row bands, ONE command buffer per
 *    band, and the next band is only submitted after `onSubmittedWorkDone` of the
 *    previous one (at most one heavy command buffer in flight);
 *  - the band height is adapted on line so each command buffer takes about
 *    `targetSubmitMs` of GPU time (multiplicative estimate from the measured time
 *    per row), capped by `maxPixelViewsPerSubmit`;
 *  - every dispatch increments a sentinel counter; if the count read back with the
 *    results does not match the number of submissions (work silently dropped while
 *    the page was hidden) the whole computation is retried (max 2×, every source
 *    uploaded again: uploads may have been dropped too) and then fails;
 *  - all pipelines are compiled in `create()` (compilation blocks the GPU process
 *    on Safari 26), raced against device loss and a timeout.
 * Textures and buffers are reused across compute() calls. Source grey images and source depth
 * maps live in LRU texture-array caches keyed by the uploaded object (`LayerCache`): a view that
 * is a source of several references is converted and uploaded once while it stays resident (the
 * caller must not modify an image / depth array in place after passing it). The reference image is
 * uploaded as is (its grey is derived in the shader). A `finalize` pass converts the hypotheses to
 * depth / octahedral normal / cost on the GPU, and the result is read back at 12 B/px together
 * with the dropped-work sentinel in one mapping when it fits the staging buffer.
 */
import type { GPUContext } from '../gpu/device';
import { createBuffer } from '../gpu/device';
import type { DenseView, DepthMap, DepthRange, GeometricInput, PatchMatchOptions } from './types';
import { MAX_PATCHMATCH_SOURCES } from './types';
import {
  GPU_SOURCE_CAMERA_FLOATS,
  buildSourceCamera,
  checkRange,
  clampSources,
  depthPerturbation,
  emptyDepthMap,
  goodCostThreshold,
  isGeometricActive,
  normalPerturbation,
  octCoords,
  octDecode,
  packSourceCameras,
  passSeed,
  priorToPlanes,
  resolvePatchMatchOptions,
  rgbaToGray8,
  BandController,
  type ResolvedPatchMatchOptions,
} from './patchmatch-common';
import { patchMatchCPU } from './patchmatch-cpu';
import { PARAMS_BYTES, PARAMS_STRIDE, WORKGROUP_X, WORKGROUP_Y, patchMatchWGSL } from './patchmatch-wgsl';
import { makeSyntheticDataset } from './synthetic';

export interface PatchMatchGPUTimings {
  /** CPU-side preparation + uploads. */
  uploadMs: number;
  initMs: number;
  /** Wall time per iteration (red + black passes). */
  iterationsMs: number[];
  readbackMs: number;
  totalMs: number;
  /** Command buffers submitted (last attempt). */
  submits: number;
  /** Longest single command buffer (wall time submit → done), ms. */
  maxSubmitMs: number;
  /** Band height (rows) of the last propagation submit. */
  bandRows: number;
  /** Full recomputations because GPU work was dropped. */
  retries: number;
  /** GPU execution time of the compute passes (timestamp queries; only with `create(…, { timestamps: true })`). */
  gpuMs?: number;
}

/** Instance options of `PatchMatchGPU.create`. */
export interface PatchMatchGPUCreateOptions {
  /**
   * Measure the GPU time of every compute pass with timestamp queries (`lastTimings.gpuMs`) when
   * the device has 'timestamp-query'. Diagnostics: costs one extra readback per compute(), and
   * WebKit's timestamps are expensive. Default false.
   */
  timestamps?: boolean;
}

/** Timestamp-query state (2 queries per compute pass). */
interface Timestamps {
  set: GPUQuerySet;
  resolve: GPUBuffer;
  read: GPUBuffer;
  /** Capacity in passes. */
  passes: number;
}

export interface PatchMatchSelfTestResult {
  ok: boolean;
  /** Fraction of pixels where GPU and CPU agree within 2 % (among pixels valid in either). */
  agreement: number;
  gpuValid: number;
  cpuValid: number;
  /** Median relative depth error of the GPU result vs ground truth. */
  gpuMedianRelError: number;
  ms: number;
  error?: string;
}

interface Pipelines {
  init: GPUComputePipeline;
  propagate: GPUComputePipeline;
  finalize: GPUComputePipeline;
  priorPlanes: GPUComputePipeline;
}

/** Per-reference-size resources (reused while W × H is unchanged). */
interface Resources {
  width: number;
  height: number;
  refTex: GPUTexture;
  planes: GPUBuffer;
  costs: GPUBuffer;
  photo: GPUBuffer;
  views: GPUBuffer;
  staging: GPUBuffer;
}

/**
 * 2D-array texture whose layers cache uploads keyed by the uploaded object (a source image, a
 * depth array): LRU over the layers, never evicting a layer the current compute() uses. Keys are
 * held weakly (ids through a WeakMap), so the cache never keeps a caller's data alive. Grows
 * (reallocated, every entry dropped) when a larger image or more layers are needed.
 */
class LayerCache {
  tex: GPUTexture | null = null;
  width = 0;
  height = 0;
  layers = 0;
  hits = 0;
  misses = 0;
  private readonly ids = new WeakMap<object, number>();
  private nextId = 1;
  private slotId: number[] = [];
  private slotUse: number[] = [];
  private clock = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly format: GPUTextureFormat,
    private readonly label: string,
  ) {}

  /** At least W × H × `layers` (grow-only in every dimension). */
  ensure(W: number, H: number, layers: number): void {
    if (this.tex && this.width >= W && this.height >= H && this.layers >= layers) return;
    const w = Math.max(W, this.width);
    const h = Math.max(H, this.height);
    const l = Math.max(layers, this.layers);
    this.tex?.destroy();
    this.tex = this.device.createTexture({
      label: this.label,
      size: [w, h, l],
      dimension: '2d',
      format: this.format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.width = w;
    this.height = h;
    this.layers = l;
    this.slotId = new Array<number>(l).fill(0);
    this.slotUse = new Array<number>(l).fill(0);
  }

  /** Layer holding `key`'s data; on a miss `upload(layer)` fills the least recently used free layer. */
  acquire(key: object, taken: Set<number>, upload: (layer: number) => void): number {
    let id = this.ids.get(key);
    if (!id) {
      id = this.nextId++;
      this.ids.set(key, id);
    }
    const t = ++this.clock;
    const hit = this.slotId.indexOf(id);
    if (hit >= 0 && !taken.has(hit)) {
      this.slotUse[hit] = t;
      this.hits++;
      taken.add(hit);
      return hit;
    }
    let best = -1;
    for (let i = 0; i < this.layers; i++) if (!taken.has(i) && (best < 0 || this.slotUse[i] < this.slotUse[best])) best = i;
    if (best < 0) throw new Error(`PatchMatchGPU: ${this.label} has no free layer`);
    upload(best);
    this.slotId[best] = id;
    this.slotUse[best] = t;
    this.misses++;
    taken.add(best);
    return best;
  }

  /** Forget every layer's content (the texture stays): the next acquire of any key uploads again. */
  invalidate(): void {
    this.slotId.fill(0);
    this.slotUse.fill(0);
  }

  destroy(): void {
    this.tex?.destroy();
    this.tex = null;
    this.width = this.height = this.layers = 0;
    this.slotId = [];
    this.slotUse = [];
  }
}

/** Number of Params blocks in the uniform ring (one per submission, reused cyclically). */
const PARAMS_RING = 64;
/** Source-count buckets compiled up front (array capacities in the shader). */
const SOURCE_BUCKETS = [4, 8, 12, 16];

class DroppedWorkError extends Error {}

export class PatchMatchGPU {
  /** Readback staging-buffer cap in bytes (outputs larger than this, 12 B/px, are read in row chunks). */
  static stagingBytes = 8 << 20;
  /**
   * Upload caches: GPU bytes each may hold beyond the layers of the current call (up to 4 slack
   * layers: the next references' sources are mostly the same views).
   */
  static cacheSlackBytes = 8 << 20;
  /** Pipeline compilation timeout before falling back to synchronous creation. */
  static pipelineTimeoutMs = 30_000;
  /** Use the adaptive band controller also on software adapters (tests). */
  static adaptiveOnSoftware = false;

  private readonly device: GPUDevice;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly pipelines = new Map<string, Promise<Pipelines>>();
  private readonly sampler: GPUSampler;
  private readonly camsBuf: GPUBuffer;
  private readonly paramsBuf: GPUBuffer;
  private readonly statusBuf: GPUBuffer;
  private res: Resources | null = null;
  private readonly srcCache: LayerCache;
  private readonly depthCache: LayerCache;
  private grayBuf: Uint8Array | undefined;
  private ts: Timestamps | null = null;
  private lost = false;
  private destroyed = false;
  private busy: Promise<unknown> = Promise.resolve();
  /** Timings of the last compute() call. */
  lastTimings: PatchMatchGPUTimings | null = null;

  private constructor(
    private readonly ctx: GPUContext,
    private readonly defaults: Partial<PatchMatchOptions>,
  ) {
    const device = ctx.device;
    this.device = device;
    device.lost.then(() => {
      this.lost = true;
    });
    const cs = GPUShaderStage.COMPUTE;
    this.layout = device.createBindGroupLayout({
      label: 'patchmatch-bgl',
      entries: [
        { binding: 0, visibility: cs, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAMS_BYTES } },
        { binding: 1, visibility: cs, buffer: { type: 'uniform' } },
        { binding: 2, visibility: cs, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: cs, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 4, visibility: cs, sampler: { type: 'filtering' } },
        { binding: 5, visibility: cs, texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' } },
        { binding: 6, visibility: cs, buffer: { type: 'storage' } },
        { binding: 7, visibility: cs, buffer: { type: 'storage' } },
        { binding: 8, visibility: cs, buffer: { type: 'storage' } },
        { binding: 9, visibility: cs, buffer: { type: 'storage' } },
        { binding: 10, visibility: cs, buffer: { type: 'storage' } },
      ],
    });
    this.pipelineLayout = device.createPipelineLayout({ label: 'patchmatch-pl', bindGroupLayouts: [this.layout] });
    this.sampler = device.createSampler({
      label: 'patchmatch-linear',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    this.camsBuf = createBuffer(device, MAX_PATCHMATCH_SOURCES * GPU_SOURCE_CAMERA_FLOATS * 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'patchmatch-cams');
    this.paramsBuf = createBuffer(device, PARAMS_RING * PARAMS_STRIDE, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'patchmatch-params');
    this.statusBuf = createBuffer(device, 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'patchmatch-status');
    this.srcCache = new LayerCache(device, 'r8unorm', 'pm-src');
    this.depthCache = new LayerCache(device, 'r32float', 'pm-src-depth');
  }

  /** Upload-cache statistics (source images, source depth maps). */
  get cacheStats(): { imageHits: number; imageMisses: number; depthHits: number; depthMisses: number } {
    return { imageHits: this.srcCache.hits, imageMisses: this.srcCache.misses, depthHits: this.depthCache.hits, depthMisses: this.depthCache.misses };
  }

  /**
   * Create the matcher and compile the pipelines for the (patchRadius, patchStep, affineWindow) of
   * `opts` up front: the source-count buckets (4/8/12/16) up to the one holding
   * `maxSources` (default all). Larger buckets compile lazily on first use; Safari's
   * shader compile is slow, so callers that know their source count should pass it.
   */
  static async create(ctx: GPUContext, opts?: Partial<PatchMatchOptions>, maxSources = MAX_PATCHMATCH_SOURCES, inst: PatchMatchGPUCreateOptions = {}): Promise<PatchMatchGPU> {
    const pm = new PatchMatchGPU(ctx, opts ?? {});
    if (inst.timestamps && ctx.device.features.has('timestamp-query')) {
      const passes = 1024;
      pm.ts = {
        set: ctx.device.createQuerySet({ label: 'pm-timestamps', type: 'timestamp', count: 2 * passes }),
        resolve: createBuffer(ctx.device, 16 * passes, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC, 'pm-ts-resolve'),
        read: ctx.device.createBuffer({ label: 'pm-ts-read', size: 16 * passes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        passes,
      };
    }
    const o = resolvePatchMatchOptions(opts);
    const top = SOURCE_BUCKETS.find((b) => b >= maxSources) ?? MAX_PATCHMATCH_SOURCES;
    try {
      for (const b of SOURCE_BUCKETS) if (b <= top) await pm.getPipelines(b, o.patchRadius, o.patchStep, o.affineWindow);
    } catch (e) {
      pm.destroy();
      throw e;
    }
    return pm;
  }

  /** Compile the pipelines of another kernel shape (patchRadius, patchStep, affineWindow) up front. */
  async precompile(opts: Partial<PatchMatchOptions>, maxSources = MAX_PATCHMATCH_SOURCES): Promise<void> {
    const o = resolvePatchMatchOptions({ ...this.defaults, ...opts });
    const top = SOURCE_BUCKETS.find((b) => b >= maxSources) ?? MAX_PATCHMATCH_SOURCES;
    for (const b of SOURCE_BUCKETS) if (b <= top) await this.getPipelines(b, o.patchRadius, o.patchStep, o.affineWindow);
  }

  /**
   * Compute the depth/normal map of `ref`. Calls are serialised per instance.
   * With `geometric` (prior + source depth maps, same order as `sources`) the
   * ACMM geometric-consistency pass runs instead of the photometric one.
   */
  compute(
    ref: DenseView,
    sources: DenseView[],
    range: DepthRange,
    opts?: Partial<PatchMatchOptions>,
    geometric?: GeometricInput,
  ): Promise<DepthMap> {
    const run = this.busy.then(
      () => this.computeWithRetry(ref, sources, range, opts, geometric),
      () => this.computeWithRetry(ref, sources, range, opts, geometric),
    );
    this.busy = run.catch(() => undefined);
    return run;
  }

  /**
   * GPU-vs-CPU self-test on a tiny synthetic scene (80×60, 3 sources, 2
   * iterations). Run once after creation (Safari has had WGSL miscompiles); fall
   * back to the CPU path when `ok` is false.
   */
  async selfTest(): Promise<PatchMatchSelfTestResult> {
    const t0 = performance.now();
    try {
      const ds = makeSyntheticDataset('near', { width: 80, height: 60, numSources: 3, supersample: 1 });
      // The CPU reference must run with exactly the options the GPU run resolves to
      // (instance defaults from create() included), otherwise a GPU created with e.g.
      // a different patch radius / seed / sigma is compared against a different
      // algorithm. Photometric mode is forced (no geometric input here).
      const opts: Partial<PatchMatchOptions> = { ...this.defaults, iterations: 2, maxCost: 2, geometric: false };
      const g = await this.compute(ds.views[0], ds.views.slice(1), ds.range, opts);
      const c = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, opts);
      let both = 0;
      let agree = 0;
      let gv = 0;
      let cv = 0;
      let n = 0;
      const rel: number[] = [];
      for (let i = 0; i < g.depth.length; i++) {
        const zg = ds.gtDepth[0][i];
        if (!(zg > 0)) continue;
        n++;
        const a = g.depth[i];
        const b = c.depth[i];
        if (a > 0) {
          gv++;
          rel.push(Math.abs(a - zg) / zg);
        }
        if (b > 0) cv++;
        if (a > 0 || b > 0) {
          both++;
          if (a > 0 && b > 0 && Math.abs(a - b) <= 0.02 * b) agree++;
        }
      }
      rel.sort((x, y) => x - y);
      const med = rel.length ? rel[rel.length >> 1] : 1;
      const agreement = both ? agree / both : 0;
      const gpuValid = n ? gv / n : 0;
      const cpuValid = n ? cv / n : 0;
      const ok = agreement >= 0.7 && Math.abs(gpuValid - cpuValid) < 0.1 && med < 0.06;
      return { ok, agreement, gpuValid, cpuValid, gpuMedianRelError: med, ms: performance.now() - t0 };
    } catch (e) {
      return { ok: false, agreement: 0, gpuValid: 0, cpuValid: 0, gpuMedianRelError: 1, ms: performance.now() - t0, error: String((e as Error)?.message ?? e) };
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.releaseResources();
    this.paramsBuf.destroy();
    this.camsBuf.destroy();
    this.statusBuf.destroy();
    if (this.ts) {
      this.ts.set.destroy();
      this.ts.resolve.destroy();
      this.ts.read.destroy();
      this.ts = null;
    }
    this.pipelines.clear();
  }

  // ---------------------------------------------------------------------------

  private checkAlive(): void {
    if (this.destroyed) throw new Error('PatchMatchGPU: instance destroyed');
    if (this.lost) throw new Error('PatchMatchGPU: GPU device lost (recreate via getGPU() + PatchMatchGPU.create)');
  }

  private async createPipeline(desc: GPUComputePipelineDescriptor): Promise<GPUComputePipeline> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<'timeout'>((resolve) => {
      timer = setTimeout(() => resolve('timeout'), PatchMatchGPU.pipelineTimeoutMs);
    });
    const lost = this.device.lost.then(() => 'lost' as const);
    try {
      const r = await Promise.race([this.device.createComputePipelineAsync(desc), lost, timeout]);
      if (r === 'lost') throw new Error('PatchMatchGPU: device lost during pipeline creation');
      if (r === 'timeout') {
        this.checkAlive();
        console.warn('[PatchMatchGPU] async pipeline creation timed out; creating synchronously');
        return this.device.createComputePipeline(desc);
      }
      return r;
    } finally {
      clearTimeout(timer);
    }
  }

  private getPipelines(maxSources: number, patchRadius: number, patchStep: number, affine: boolean): Promise<Pipelines> {
    const key = pipelineKey(maxSources, patchRadius, patchStep, affine);
    let p = this.pipelines.get(key);
    if (!p) {
      const code = patchMatchWGSL({ maxSources, patchRadius, patchStep, affine });
      const module = this.device.createShaderModule({ code, label: `patchmatch-${key}` });
      p = (async () => {
        const init = await this.createPipeline({ label: `pm-init-${key}`, layout: this.pipelineLayout, compute: { module, entryPoint: 'init' } });
        const propagate = await this.createPipeline({ label: `pm-prop-${key}`, layout: this.pipelineLayout, compute: { module, entryPoint: 'propagate' } });
        const finalize = await this.createPipeline({ label: `pm-final-${key}`, layout: this.pipelineLayout, compute: { module, entryPoint: 'finalize' } });
        const priorPlanes = await this.createPipeline({ label: `pm-prior-${key}`, layout: this.pipelineLayout, compute: { module, entryPoint: 'priorPlanes' } });
        return { init, propagate, finalize, priorPlanes };
      })();
      p.catch(() => this.pipelines.delete(key));
      this.pipelines.set(key, p);
    }
    return p;
  }

  private releaseResources(): void {
    const r = this.res;
    if (r) {
      r.refTex.destroy();
      r.planes.destroy();
      r.costs.destroy();
      r.photo.destroy();
      r.views.destroy();
      r.staging.destroy();
      this.res = null;
    }
    this.srcCache.destroy();
    this.depthCache.destroy();
  }

  private ensureResources(W: number, H: number): Resources {
    const r = this.res;
    if (r && r.width === W && r.height === H) return r;
    if (r) {
      r.refTex.destroy();
      r.planes.destroy();
      r.costs.destroy();
      r.photo.destroy();
      r.views.destroy();
      r.staging.destroy();
      this.res = null;
    }
    const d = this.device;
    const n = W * H;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.res = {
      width: W,
      height: H,
      refTex: d.createTexture({ label: 'pm-ref', size: [W, H, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST }),
      planes: createBuffer(d, 16 * n, storage, 'pm-planes'),
      costs: createBuffer(d, 4 * n, storage, 'pm-costs'),
      photo: createBuffer(d, 4 * n, storage, 'pm-photo'),
      views: createBuffer(d, 4 * n, storage, 'pm-views'),
      staging: d.createBuffer({
        label: 'pm-staging',
        size: Math.max(32, Math.min(12 * n + 16, Math.floor(PatchMatchGPU.stagingBytes / 16) * 16)),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
    };
    return this.res;
  }

  /** Sum of the GPU durations (ms) of the first `passes` timed compute passes. */
  private async readTimestamps(ts: Timestamps, passes: number): Promise<number> {
    const enc = this.device.createCommandEncoder({ label: 'pm-ts' });
    enc.resolveQuerySet(ts.set, 0, 2 * passes, ts.resolve, 0);
    enc.copyBufferToBuffer(ts.resolve, 0, ts.read, 0, 16 * passes);
    this.device.queue.submit([enc.finish()]);
    await ts.read.mapAsync(GPUMapMode.READ, 0, 16 * passes);
    const t = new BigUint64Array(ts.read.getMappedRange(0, 16 * passes).slice(0));
    ts.read.unmap();
    let ns = 0;
    for (let k = 0; k < passes; k++) if (t[2 * k + 1] > t[2 * k]) ns += Number(t[2 * k + 1] - t[2 * k]);
    return ns / 1e6;
  }

  /**
   * Finalised outputs (`costs` = depth, `views` = octahedral normal, `photo` = cost) → DepthMap,
   * in row chunks through the staging buffer; the first chunk also carries the dropped-work
   * sentinel, which is checked before the rest is read.
   */
  private async readOutputs(res: Resources, W: number, H: number, submits: number, range: DepthRange): Promise<DepthMap> {
    const n = W * H;
    const depth = new Float32Array(n);
    const cost = new Float32Array(n);
    const packed = new Uint32Array(n);
    const staging = res.staging;
    const rowsPerChunk = Math.floor((staging.size - 16) / (12 * W));
    if (rowsPerChunk < 1) {
      // staging smaller than one row (tests): buffer by buffer
      const status = new Uint32Array(4);
      await this.readInto(this.statusBuf, status, staging);
      if (status[0] !== submits) throw new DroppedWorkError(`PatchMatchGPU: ${submits - status[0]} of ${submits} command buffers were not executed (page hidden?)`);
      await this.readInto(res.costs, depth, staging);
      await this.readInto(res.views, packed, staging);
      await this.readInto(res.photo, cost, staging);
    } else {
      for (let y0 = 0; y0 < H; y0 += rowsPerChunk) {
        const rows = Math.min(rowsPerChunk, H - y0);
        const off = 4 * y0 * W;
        const len = 4 * rows * W;
        const enc = this.device.createCommandEncoder({ label: 'pm-readback' });
        if (y0 === 0) enc.copyBufferToBuffer(this.statusBuf, 0, staging, 0, 16);
        enc.copyBufferToBuffer(res.costs, off, staging, 16, len);
        enc.copyBufferToBuffer(res.views, off, staging, 16 + len, len);
        enc.copyBufferToBuffer(res.photo, off, staging, 16 + 2 * len, len);
        this.device.queue.submit([enc.finish()]);
        try {
          await staging.mapAsync(GPUMapMode.READ, 0, 16 + 3 * len);
        } catch (e) {
          this.checkAlive();
          throw new Error(`PatchMatchGPU: readback failed (device lost?): ${(e as Error)?.message ?? e}`);
        }
        const m = staging.getMappedRange(0, 16 + 3 * len);
        if (y0 === 0) {
          const executed = new Uint32Array(m, 0, 1)[0];
          if (executed !== submits) {
            staging.unmap();
            throw new DroppedWorkError(`PatchMatchGPU: ${submits - executed} of ${submits} command buffers were not executed (page hidden?)`);
          }
        }
        const i0 = y0 * W;
        depth.set(new Float32Array(m, 16, rows * W), i0);
        packed.set(new Uint32Array(m, 16 + len, rows * W), i0);
        cost.set(new Float32Array(m, 16 + 2 * len, rows * W), i0);
        staging.unmap();
      }
    }
    const normal = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      const z = depth[i];
      if (!(z > 0)) continue;
      // the range test again in double precision (the shader's bounds are f32-rounded)
      if (!(z >= range.min && z <= range.max)) depth[i] = 0;
      else octDecode(packed[i], normal, 3 * i);
    }
    return { width: W, height: H, depth, normal, cost };
  }

  /** Chunked GPU → CPU copy through the (bounded) staging buffer; one getMappedRange per map. */
  private async readInto(src: GPUBuffer, dst: Float32Array | Uint32Array, staging: GPUBuffer): Promise<void> {
    const total = dst.byteLength;
    const out = new Uint8Array(dst.buffer, dst.byteOffset, total);
    for (let off = 0; off < total; off += staging.size) {
      const len = Math.min(staging.size, total - off);
      const enc = this.device.createCommandEncoder({ label: 'pm-readback' });
      enc.copyBufferToBuffer(src, off, staging, 0, len);
      this.device.queue.submit([enc.finish()]);
      try {
        await staging.mapAsync(GPUMapMode.READ, 0, len);
      } catch (e) {
        this.checkAlive();
        throw new Error(`PatchMatchGPU: readback failed (device lost?): ${(e as Error)?.message ?? e}`);
      }
      out.set(new Uint8Array(staging.getMappedRange(0, len)), off);
      staging.unmap();
    }
  }

  /**
   * Run the SYNCHRONOUS `fn` inside validation (+ optionally out-of-memory) error
   * scopes. The scopes are pushed and popped synchronously around `fn` (no await in
   * between): the error-scope stack is shared by everything using the device in this
   * realm, so a scope left open across an await would capture (and could be popped
   * by) unrelated GPU work interleaved from other modules. GPU errors are rethrown.
   */
  private async scoped<T>(label: string, oom: boolean, fn: () => T): Promise<T> {
    const device = this.device;
    device.pushErrorScope('validation');
    if (oom) device.pushErrorScope('out-of-memory');
    let result: T | undefined;
    let thrown: unknown = undefined;
    let threw = false;
    try {
      result = fn();
    } catch (e) {
      thrown = e;
      threw = true;
    }
    const oomP = oom ? device.popErrorScope() : Promise.resolve(null);
    const valP = device.popErrorScope();
    const [oomErr, valErr] = await Promise.all([oomP, valP]);
    if (threw) throw thrown;
    if (valErr) throw new Error(`PatchMatchGPU: validation error during ${label}: ${valErr.message}`);
    if (oomErr) throw new Error(`PatchMatchGPU: out of GPU memory during ${label}: ${oomErr.message}`);
    return result as T;
  }

  private async computeWithRetry(
    ref: DenseView,
    sources: DenseView[],
    range: DepthRange,
    opts: Partial<PatchMatchOptions> | undefined,
    geometric: GeometricInput | undefined,
  ): Promise<DepthMap> {
    const maxRetries = 2;
    for (let attempt = 0; ; attempt++) {
      try {
        const out = await this.computeImpl(ref, sources, range, opts, geometric);
        if (this.lastTimings) this.lastTimings.retries = attempt;
        return out;
      } catch (e) {
        // the uploads of a dropped call may have been dropped too: cached layers are not trusted
        if (e instanceof DroppedWorkError) {
          this.srcCache.invalidate();
          this.depthCache.invalidate();
        }
        if (!(e instanceof DroppedWorkError) || attempt >= maxRetries) {
          if (e instanceof DroppedWorkError) throw new Error(`${e.message} (after ${attempt + 1} attempts)`);
          throw e;
        }
        console.warn(`[PatchMatchGPU] ${e.message}; retrying`);
      }
    }
  }

  private async computeImpl(
    ref: DenseView,
    sourcesIn: DenseView[],
    range: DepthRange,
    optsIn: Partial<PatchMatchOptions> | undefined,
    geometric: GeometricInput | undefined,
  ): Promise<DepthMap> {
    this.checkAlive();
    const tStart = performance.now();
    checkRange(range);
    const merged: Partial<PatchMatchOptions> = { ...this.defaults, ...(optsIn ?? {}) };
    const o = resolvePatchMatchOptions(merged);
    const geo = isGeometricActive(merged, geometric);
    const usePrior = !!geometric; // see patchMatchCPU
    const sources = clampSources(sourcesIn, 'PatchMatchGPU');
    const W = ref.cam.width;
    const H = ref.cam.height;
    if (ref.image.width !== W || ref.image.height !== H) throw new Error('PatchMatchGPU: ref image size does not match ref.cam');
    const N = sources.length;
    if (N === 0 || W === 0 || H === 0) return emptyDepthMap(W, H);
    if (geo && geometric!.sourceDepths.length < N) throw new Error('PatchMatchGPU: geometric.sourceDepths must have one depth map per source');
    const maxDim = this.ctx.limits.maxTextureDimension2D;
    if (W > maxDim || H > maxDim) throw new Error(`PatchMatchGPU: image ${W}×${H} exceeds maxTextureDimension2D ${maxDim}`);

    const maxSources = SOURCE_BUCKETS.find((b) => b >= N) ?? MAX_PATCHMATCH_SOURCES;
    const key = pipelineKey(maxSources, o.patchRadius, o.patchStep, o.affineWindow);
    if (!this.pipelines.has(key)) console.warn(`[PatchMatchGPU] compiling pipeline ${key} lazily (pass patchRadius/patchStep/affineWindow to create() to precompile)`);
    const pipes = await this.getPipelines(maxSources, o.patchRadius, o.patchStep, o.affineWindow);
    this.checkAlive();

    let texW = 1;
    let texH = 1;
    for (const s of sources) {
      if (s.image.width !== s.cam.width || s.image.height !== s.cam.height) throw new Error('PatchMatchGPU: source image size does not match cam');
      texW = Math.max(texW, s.cam.width);
      texH = Math.max(texH, s.cam.height);
    }
    let depthW = 1;
    let depthH = 1;
    if (geo) {
      for (let i = 0; i < N; i++) {
        depthW = Math.max(depthW, geometric!.sourceDepths[i].width);
        depthH = Math.max(depthH, geometric!.sourceDepths[i].height);
      }
    }
    if (Math.max(texW, texH, depthW, depthH) > maxDim) {
      throw new Error(`PatchMatchGPU: a source image / depth map exceeds maxTextureDimension2D ${maxDim}`);
    }
    const device = this.device;
    const q = device.queue;
    const iterations = geo ? o.geometricIterations : o.iterations;

    // ---- resources & uploads (inside error scopes) ------------------------------
    let priorSize: { width: number; height: number } | null = null;
    // On failure (e.g. out of memory) the cached resources may be invalid objects:
    // release them so the next call reallocates instead of failing forever.
    const { res, bindGroup } = await this.scoped('set-up', true, () => {
      const res = this.ensureResources(W, H);
      // upload caches: every source of this call resident at once, plus slack layers for the
      // next references' sources (trajectory-local); photometric passes keep the depth cache
      const slack = (bytesPerLayer: number): number => Math.min(4, Math.floor(PatchMatchGPU.cacheSlackBytes / bytesPerLayer));
      const maxLayers = this.ctx.limits.maxTextureArrayLayers;
      this.srcCache.ensure(texW, texH, Math.min(maxLayers, N + slack(Math.max(texW, this.srcCache.width) * Math.max(texH, this.srcCache.height))));
      if (geo) this.depthCache.ensure(depthW, depthH, Math.min(maxLayers, N + slack(4 * Math.max(depthW, this.depthCache.width) * Math.max(depthH, this.depthCache.height))));
      else this.depthCache.ensure(1, 1, 1);
      const srcTex = this.srcCache.tex!;
      const depthTex = this.depthCache.tex!;
      // reference: RGBA as is (the shader derives the grey)
      q.writeTexture({ texture: res.refTex }, ref.image.data, { bytesPerRow: 4 * W, rowsPerImage: H }, [W, H, 1]);
      const cams = [];
      const imageLayers: number[] = [];
      const depthLayers: number[] = [];
      const takenImg = new Set<number>();
      const takenDepth = new Set<number>();
      for (let s = 0; s < N; s++) {
        const v = sources[s];
        imageLayers.push(
          this.srcCache.acquire(v.image, takenImg, (layer) => {
            this.grayBuf = rgbaToGray8(v.image, this.grayBuf);
            q.writeTexture({ texture: srcTex, origin: [0, 0, layer] }, this.grayBuf, { bytesPerRow: v.cam.width, rowsPerImage: v.cam.height }, [v.cam.width, v.cam.height, 1]);
          }),
        );
        const dm = geo ? geometric!.sourceDepths[s] : undefined;
        if (dm && dm.width > 0 && dm.height > 0) {
          depthLayers.push(
            this.depthCache.acquire(dm.depth, takenDepth, (layer) => {
              q.writeTexture({ texture: depthTex, origin: [0, 0, layer] }, dm.depth, { bytesPerRow: 4 * dm.width, rowsPerImage: dm.height }, [dm.width, dm.height, 1]);
            }),
          );
        } else depthLayers.push(0); // 0 × 0 map: never read (dsize = 0)
        cams.push(buildSourceCamera(ref, v, dm ? { width: dm.width, height: dm.height } : undefined));
      }
      q.writeBuffer(this.camsBuf, 0, packSourceCameras(cams, this.srcCache.width, this.srcCache.height, undefined, { image: imageLayers, depth: depthLayers }));
      if (usePrior) {
        const pr = geometric!.prior;
        const pn = pr.width * pr.height;
        if (pn > 0 && pn <= W * H && pr.depth.length >= pn && pr.normal.length >= 3 * pn) {
          // compact prior (depth + the normal's octahedral coordinates, 12 B/px at its own size)
          // into the not yet used costs / photo / views buffers; the priorPlanes pass resamples it
          const nrm = pr.normal;
          const ox = new Float32Array(pn);
          const oy = new Float32Array(pn);
          const o2 = new Float32Array(2);
          for (let i = 0; i < pn; i++) {
            if (!(pr.depth[i] > 0)) continue;
            octCoords(nrm[3 * i], nrm[3 * i + 1], nrm[3 * i + 2], o2, 0);
            ox[i] = o2[0];
            oy[i] = o2[1];
          }
          q.writeBuffer(res.costs, 0, pr.depth, 0, pn);
          q.writeBuffer(res.photo, 0, ox);
          q.writeBuffer(res.views, 0, oy);
          priorSize = { width: pr.width, height: pr.height };
        } else q.writeBuffer(res.planes, 0, priorToPlanes(pr, W, H, undefined, ref.cam));
      }
      q.writeBuffer(this.statusBuf, 0, new Uint32Array(4));
      const bindGroup = device.createBindGroup({
        label: 'pm-bg',
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuf, offset: 0, size: PARAMS_BYTES } },
          { binding: 1, resource: { buffer: this.camsBuf } },
          { binding: 2, resource: res.refTex.createView() },
          { binding: 3, resource: srcTex.createView({ dimension: '2d-array' }) },
          { binding: 4, resource: this.sampler },
          { binding: 5, resource: depthTex.createView({ dimension: '2d-array' }) },
          { binding: 6, resource: { buffer: res.planes, size: 16 * W * H } },
          { binding: 7, resource: { buffer: res.costs, size: 4 * W * H } },
          { binding: 8, resource: { buffer: res.photo, size: 4 * W * H } },
          { binding: 9, resource: { buffer: res.views, size: 4 * W * H } },
          { binding: 10, resource: { buffer: this.statusBuf, size: 16 } },
        ],
      });
      return { res, bindGroup };
    }).catch((e: unknown) => {
      if (!this.destroyed) this.releaseResources();
      throw e;
    });
    this.checkAlive();
    const tUpload = performance.now();

    // ---- dispatch: one command buffer in flight, adaptive row bands ---------------
    const budget = Math.max(1, o.maxPixelViewsPerSubmit);
    const halfW = Math.ceil(W / 2);
    const maxRowsProp = Math.max(1, Math.min(H, Math.floor(budget / (halfW * N))));
    const maxRowsInit = Math.max(1, Math.min(H, Math.floor(budget / (W * N))));
    // software adapters (tests) have no watchdog: use the budget-only bands
    const target = this.ctx.isSoftware && !PatchMatchGPU.adaptiveOnSoftware ? Infinity : o.targetSubmitMs > 0 ? o.targetSubmitMs : Infinity;
    const minRows = Math.min(H, WORKGROUP_Y);
    const finite = Number.isFinite(target);
    const ctlInitRows = finite
      ? new BandController(target, Math.min(maxRowsInit, 64), maxRowsInit, Math.min(minRows, maxRowsInit))
      : new BandController(Infinity, maxRowsInit, maxRowsInit, 1);
    const ctlPropRows = finite
      ? new BandController(target, Math.min(maxRowsProp, 32), maxRowsProp, Math.min(minRows, maxRowsProp))
      : new BandController(Infinity, maxRowsProp, maxRowsProp, 1);

    let submits = 0;
    let maxSubmitMs = 0;
    const ts = this.ts;
    let timed = 0;
    let lastRows = 0;
    const iterTimes: number[] = [];
    let tInit = tUpload;
    let tPrev = tUpload;
    const params = new ArrayBuffer(PARAMS_BYTES);
    const pu = new Uint32Array(params);
    const pf = new Float32Array(params);
    const runPass = async (pipeline: GPUComputePipeline, ctl: BandController, colour: number, it: number, seed: number, isInit: boolean): Promise<void> => {
      let row = 0;
      while (row < H) {
        const rows = Math.min(H - row, ctl.next());
        fillParams(pu, pf, o, ref, range, N, colour, row, row + rows, it, seed, geo, usePrior, isInit, priorSize);
        const slot = submits % PARAMS_RING;
        // error scope opened and closed synchronously around encode + submit
        device.pushErrorScope('validation');
        q.writeBuffer(this.paramsBuf, slot * PARAMS_STRIDE, params);
        const enc = device.createCommandEncoder({ label: 'pm-band' });
        const tsw = ts && timed < ts.passes ? { querySet: ts.set, beginningOfPassWriteIndex: 2 * timed, endOfPassWriteIndex: 2 * timed + 1 } : undefined;
        if (tsw) timed++;
        const pass = enc.beginComputePass({ label: 'pm', ...(tsw ? { timestampWrites: tsw } : {}) });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup, [slot * PARAMS_STRIDE]);
        pass.dispatchWorkgroups(Math.ceil((isInit ? W : halfW) / WORKGROUP_X), Math.ceil(rows / WORKGROUP_Y), 1);
        pass.end();
        const t0 = performance.now();
        q.submit([enc.finish()]);
        const errP = device.popErrorScope().catch(() => null);
        submits++;
        // raced against device loss so a lost device can never hang compute()
        const [err] = await Promise.all([errP, Promise.race([q.onSubmittedWorkDone(), device.lost])]);
        this.checkAlive();
        if (err) throw new Error(`PatchMatchGPU: validation error during dispatch: ${err.message}`);
        const ms = performance.now() - t0;
        maxSubmitMs = Math.max(maxSubmitMs, ms);
        ctl.update(rows, ms);
        if (!isInit) lastRows = rows;
        row += rows;
      }
    };
    if (priorSize) await runPass(pipes.priorPlanes, new BandController(Infinity, H, H, 1), 0, 0, 0, true);
    await runPass(pipes.init, ctlInitRows, 0, 0, passSeed(o.seed, geo ? 0x1ffff : 0xffff), true);
    tInit = performance.now();
    tPrev = tInit;
    for (let it = 0; it < iterations; it++) {
      for (let colour = 0; colour < 2; colour++) {
        await runPass(pipes.propagate, ctlPropRows, colour, it, passSeed(o.seed, (geo ? 0x100 : 0) + 2 * it + colour), false);
      }
      const t = performance.now();
      iterTimes.push(t - tPrev);
      tPrev = t;
    }

    // ---- finalisation (depth / normal / cost on the GPU), readback + dropped-work check
    await runPass(pipes.finalize, new BandController(Infinity, H, H, 1), 0, 0, 0, true);
    let gpuMs: number | undefined;
    if (ts && timed > 0) gpuMs = await this.readTimestamps(ts, timed);
    const out = await this.readOutputs(res, W, H, submits, range);
    this.checkAlive();
    const tEnd = performance.now();
    this.lastTimings = {
      uploadMs: tUpload - tStart,
      initMs: tInit - tUpload,
      iterationsMs: iterTimes,
      readbackMs: tEnd - tPrev,
      totalMs: tEnd - tStart,
      submits,
      maxSubmitMs,
      bandRows: lastRows,
      retries: 0,
      ...(gpuMs !== undefined ? { gpuMs } : {}),
    };
    return out;
  }
}

function pipelineKey(maxSources: number, patchRadius: number, patchStep: number, affine: boolean): string {
  return `${maxSources}|${patchRadius}|${patchStep}|${affine ? 'affine' : 'projective'}`;
}

function fillParams(
  u: Uint32Array,
  f: Float32Array,
  o: ResolvedPatchMatchOptions,
  ref: DenseView,
  range: DepthRange,
  N: number,
  colour: number,
  rowStart: number,
  rowEnd: number,
  it: number,
  seed: number,
  geo: boolean,
  usePrior: boolean,
  init: boolean,
  prior: { width: number; height: number } | null,
): void {
  u[0] = ref.cam.width;
  u[1] = ref.cam.height;
  u[2] = N;
  u[3] = colour;
  u[4] = rowStart;
  u[5] = rowEnd;
  u[6] = it;
  u[7] = seed >>> 0;
  f[8] = ref.cam.f;
  f[9] = ref.cam.cx;
  f[10] = ref.cam.cy;
  f[11] = 1 / ref.cam.f;
  f[12] = range.min;
  f[13] = range.max;
  f[14] = 1 / range.min;
  f[15] = 1 / range.max;
  f[16] = goodCostThreshold(o, it);
  f[17] = o.badCost;
  f[18] = depthPerturbation(o, it, usePrior);
  f[19] = normalPerturbation(o, it, usePrior);
  f[20] = o.geomLambda;
  f[21] = o.geomMaxError;
  f[22] = 1 / (3 * 2 * o.sigmaColor * o.sigmaColor);
  f[23] = o.minVariance;
  u[24] = Math.min(o.topK, N);
  u[25] = o.numSamplesPerIter;
  u[26] = usePrior && init ? 1 : 0;
  u[27] = geo ? 1 : 0;
  f[28] = 1 / (2 * o.sigmaSpatial * o.sigmaSpatial);
  f[29] = o.maxCost;
  u[30] = prior ? prior.width : 0;
  u[31] = prior ? prior.height : 0;
}
