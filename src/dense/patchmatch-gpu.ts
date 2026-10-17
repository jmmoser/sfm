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
 *    the page was hidden) the whole computation is retried (max 2×) and then fails;
 *  - all pipelines are compiled in `create()` (compilation blocks the GPU process
 *    on Safari 26), raced against device loss and a timeout.
 * Textures and buffers are reused across compute() calls (grow-only arrays).
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
  packSourceCameras,
  passSeed,
  planesToDepthMap,
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

/** A 2D-array texture that is only reallocated when it has to grow. */
interface ArrayTex {
  tex: GPUTexture;
  width: number;
  height: number;
  layers: number;
}

/** Number of Params blocks in the uniform ring (one per submission, reused cyclically). */
const PARAMS_RING = 64;
/** Source-count buckets compiled up front (array capacities in the shader). */
const SOURCE_BUCKETS = [4, 8, 12, 16];

class DroppedWorkError extends Error {}

export class PatchMatchGPU {
  /** Readback staging-buffer cap in bytes (outputs larger than this are read in chunks). */
  static stagingBytes = 4 << 20;
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
  private srcArr: ArrayTex | null = null;
  private depthArr: ArrayTex | null = null;
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
  }

  /**
   * Create the matcher and compile the pipelines for the (patchRadius, patchStep) of
   * `opts` up front: the source-count buckets (4/8/12/16) up to the one holding
   * `maxSources` (default all). Larger buckets compile lazily on first use; Safari's
   * shader compile is slow, so callers that know their source count should pass it.
   */
  static async create(ctx: GPUContext, opts?: Partial<PatchMatchOptions>, maxSources = MAX_PATCHMATCH_SOURCES): Promise<PatchMatchGPU> {
    const pm = new PatchMatchGPU(ctx, opts ?? {});
    const o = resolvePatchMatchOptions(opts);
    const top = SOURCE_BUCKETS.find((b) => b >= maxSources) ?? MAX_PATCHMATCH_SOURCES;
    try {
      for (const b of SOURCE_BUCKETS) if (b <= top) await pm.getPipelines(b, o.patchRadius, o.patchStep);
    } catch (e) {
      pm.destroy();
      throw e;
    }
    return pm;
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

  private getPipelines(maxSources: number, patchRadius: number, patchStep: number): Promise<Pipelines> {
    const key = `${maxSources}|${patchRadius}|${patchStep}`;
    let p = this.pipelines.get(key);
    if (!p) {
      const code = patchMatchWGSL({ maxSources, patchRadius, patchStep });
      const module = this.device.createShaderModule({ code, label: `patchmatch-${key}` });
      p = (async () => {
        const init = await this.createPipeline({ label: `pm-init-${key}`, layout: this.pipelineLayout, compute: { module, entryPoint: 'init' } });
        const propagate = await this.createPipeline({ label: `pm-prop-${key}`, layout: this.pipelineLayout, compute: { module, entryPoint: 'propagate' } });
        return { init, propagate };
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
    this.srcArr?.tex.destroy();
    this.srcArr = null;
    this.depthArr?.tex.destroy();
    this.depthArr = null;
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
        size: Math.max(16, Math.min(20 * n, Math.floor(PatchMatchGPU.stagingBytes / 16) * 16)),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
    };
    return this.res;
  }

  /**
   * Grow-only 2D-array texture: reused when it is at least W × H × layers (the
   * shaders bound every access by the per-source image / depth-map size).
   */
  private ensureArray(cur: ArrayTex | null, W: number, H: number, layers: number, format: GPUTextureFormat, label: string): ArrayTex {
    if (cur && cur.width >= W && cur.height >= H && cur.layers >= layers) return cur;
    // grow-only in every dimension (avoids reallocating back and forth when views of
    // different aspect ratios alternate)
    const keep = cur ? Math.max(cur.layers, layers) : layers;
    if (cur) {
      W = Math.max(W, cur.width);
      H = Math.max(H, cur.height);
    }
    cur?.tex.destroy();
    const tex = this.device.createTexture({
      label,
      size: [W, H, keep],
      dimension: '2d',
      format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    return { tex, width: W, height: H, layers: keep };
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
    const key = `${maxSources}|${o.patchRadius}|${o.patchStep}`;
    if (!this.pipelines.has(key)) console.warn(`[PatchMatchGPU] compiling pipeline ${key} lazily (pass patchRadius/patchStep to create() to precompile)`);
    const pipes = await this.getPipelines(maxSources, o.patchRadius, o.patchStep);
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
    // On failure (e.g. out of memory) the cached resources may be invalid objects:
    // release them so the next call reallocates instead of failing forever.
    const { res, bindGroup } = await this.scoped('set-up', true, () => {
      const res = this.ensureResources(W, H);
      this.srcArr = this.ensureArray(this.srcArr, texW, texH, N, 'r8unorm', 'pm-src');
      // photometric passes never read the depth array: keep whatever exists (or a 1×1 dummy)
      this.depthArr = geo
        ? this.ensureArray(this.depthArr, depthW, depthH, N, 'r32float', 'pm-src-depth')
        : (this.depthArr ?? this.ensureArray(null, 1, 1, 1, 'r32float', 'pm-src-depth'));
      const srcTex = this.srcArr.tex;
      const depthTex = this.depthArr.tex;
      // reference: RGB + grey in alpha
      const gray = rgbaToGray8(ref.image);
      const refData = new Uint8Array(W * H * 4);
      refData.set(ref.image.data);
      for (let i = 0; i < W * H; i++) refData[4 * i + 3] = gray[i];
      q.writeTexture({ texture: res.refTex }, refData, { bytesPerRow: 4 * W, rowsPerImage: H }, [W, H, 1]);
      const cams = [];
      let grayBuf: Uint8Array | undefined;
      for (let s = 0; s < N; s++) {
        const v = sources[s];
        grayBuf = rgbaToGray8(v.image, grayBuf);
        q.writeTexture({ texture: srcTex, origin: [0, 0, s] }, grayBuf, { bytesPerRow: v.cam.width, rowsPerImage: v.cam.height }, [v.cam.width, v.cam.height, 1]);
        const dm = geo ? geometric!.sourceDepths[s] : undefined;
        if (dm && dm.width > 0 && dm.height > 0) {
          q.writeTexture({ texture: depthTex, origin: [0, 0, s] }, dm.depth, { bytesPerRow: 4 * dm.width, rowsPerImage: dm.height }, [dm.width, dm.height, 1]);
        }
        cams.push(buildSourceCamera(ref, v, dm ? { width: dm.width, height: dm.height } : undefined));
      }
      q.writeBuffer(this.camsBuf, 0, packSourceCameras(cams, this.srcArr.width, this.srcArr.height));
      if (usePrior) q.writeBuffer(res.planes, 0, priorToPlanes(geometric!.prior, W, H, undefined, ref.cam));
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
        fillParams(pu, pf, o, ref, range, N, colour, row, row + rows, it, seed, geo, usePrior, isInit);
        const slot = submits % PARAMS_RING;
        // error scope opened and closed synchronously around encode + submit
        device.pushErrorScope('validation');
        q.writeBuffer(this.paramsBuf, slot * PARAMS_STRIDE, params);
        const enc = device.createCommandEncoder({ label: 'pm-band' });
        const pass = enc.beginComputePass({ label: 'pm' });
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

    // ---- readback + dropped-work check -------------------------------------------
    const n = W * H;
    const status = new Uint32Array(4);
    await this.readInto(this.statusBuf, status, res.staging);
    if (status[0] !== submits) {
      throw new DroppedWorkError(`PatchMatchGPU: ${submits - status[0]} of ${submits} command buffers were not executed (page hidden?)`);
    }
    const planes = new Float32Array(4 * n);
    const photo = new Float32Array(n);
    await this.readInto(res.planes, planes, res.staging);
    await this.readInto(res.photo, photo, res.staging);
    this.checkAlive();
    const out = planesToDepthMap(planes, photo, W, H, range, o.maxCost);
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
    };
    return out;
  }
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
  f[29] = 0;
  f[30] = 0;
  f[31] = 0;
}
