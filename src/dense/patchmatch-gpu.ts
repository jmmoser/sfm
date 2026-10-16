/**
 * WebGPU PatchMatch MVS (ACMH view selection + ACMM geometric consistency).
 * Same algorithm as patchMatchCPU (see patchmatch-cpu.ts / patchmatch-wgsl.ts).
 *
 * Work is split into row bands, one queue submission per band and checkerboard
 * colour, with `gpuIdle` awaited after every iteration so no single command
 * buffer runs long enough to trip the iOS GPU watchdog. Textures and buffers are
 * reused across compute() calls with the same image sizes.
 */
import type { GPUContext } from '../gpu/device';
import { createBuffer, gpuIdle } from '../gpu/device';
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
  type ResolvedPatchMatchOptions,
} from './patchmatch-common';
import { PARAMS_BYTES, PARAMS_STRIDE, WORKGROUP_X, WORKGROUP_Y, patchMatchWGSL } from './patchmatch-wgsl';

export interface PatchMatchGPUTimings {
  uploadMs: number;
  initMs: number;
  iterationsMs: number[];
  readbackMs: number;
  totalMs: number;
  submits: number;
  bandRows: number;
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

export class PatchMatchGPU {
  /** Readback staging-buffer cap in bytes (outputs larger than this are read in chunks). */
  static stagingBytes = 4 << 20;
  private readonly device: GPUDevice;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly pipelines = new Map<string, Promise<Pipelines>>();
  private readonly sampler: GPUSampler;
  private readonly camsBuf: GPUBuffer;
  private paramsBuf: GPUBuffer | null = null;
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
    this.layout = device.createBindGroupLayout({
      label: 'patchmatch-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAMS_BYTES } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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
  }

  /** Create the matcher; compiles the default pipeline eagerly. */
  static async create(ctx: GPUContext, opts?: Partial<PatchMatchOptions>): Promise<PatchMatchGPU> {
    const pm = new PatchMatchGPU(ctx, opts ?? {});
    const o = resolvePatchMatchOptions(opts);
    await pm.getPipelines(8, o.patchRadius, o.patchStep);
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
      () => this.computeImpl(ref, sources, range, opts, geometric),
      () => this.computeImpl(ref, sources, range, opts, geometric),
    );
    this.busy = run.catch(() => undefined);
    return run;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.releaseResources();
    this.paramsBuf?.destroy();
    this.paramsBuf = null;
    this.camsBuf.destroy();
    this.pipelines.clear();
  }

  // ---------------------------------------------------------------------------

  private checkAlive(): void {
    if (this.destroyed) throw new Error('PatchMatchGPU: instance destroyed');
    if (this.lost) throw new Error('PatchMatchGPU: GPU device lost (recreate via getGPU() + PatchMatchGPU.create)');
  }

  private getPipelines(maxSources: number, patchRadius: number, patchStep: number): Promise<Pipelines> {
    const key = `${maxSources}|${patchRadius}|${patchStep}`;
    let p = this.pipelines.get(key);
    if (!p) {
      const code = patchMatchWGSL({ maxSources, patchRadius, patchStep });
      const module = this.device.createShaderModule({ code, label: `patchmatch-${key}` });
      p = Promise.all([
        this.device.createComputePipelineAsync({ label: `pm-init-${key}`, layout: this.pipelineLayout, compute: { module, entryPoint: 'init' } }),
        this.device.createComputePipelineAsync({ label: `pm-prop-${key}`, layout: this.pipelineLayout, compute: { module, entryPoint: 'propagate' } }),
      ]).then(([init, propagate]) => ({ init, propagate }));
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
    const keep = cur && cur.width >= W && cur.height >= H ? Math.max(cur.layers, layers) : layers;
    if (cur && cur.width >= W && cur.height >= H) {
      W = cur.width;
      H = cur.height;
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

  private ensureParams(bytes: number): GPUBuffer {
    if (!this.paramsBuf || this.paramsBuf.size < bytes) {
      this.paramsBuf?.destroy();
      this.paramsBuf = this.device.createBuffer({ label: 'pm-params', size: Math.max(bytes, PARAMS_STRIDE), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    return this.paramsBuf;
  }

  /** Chunked GPU → CPU copy through the (bounded) staging buffer. */
  private async readInto(src: GPUBuffer, dst: Float32Array, staging: GPUBuffer): Promise<void> {
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
   * Run `fn` inside validation (+ optionally out-of-memory) error scopes; the
   * scopes are always popped, and GPU errors are rethrown as Errors.
   */
  private async scoped<T>(label: string, oom: boolean, fn: () => T | Promise<T>): Promise<T> {
    const device = this.device;
    device.pushErrorScope('validation');
    if (oom) device.pushErrorScope('out-of-memory');
    let result: T | undefined;
    let thrown: unknown = null;
    try {
      result = await fn();
    } catch (e) {
      thrown = e;
    }
    const oomErr = oom ? await device.popErrorScope() : null;
    const valErr = await device.popErrorScope();
    if (thrown) throw thrown;
    if (valErr) throw new Error(`PatchMatchGPU: validation error during ${label}: ${valErr.message}`);
    if (oomErr) throw new Error(`PatchMatchGPU: out of GPU memory during ${label}: ${oomErr.message}`);
    return result as T;
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
    const sources = clampSources(sourcesIn, 'PatchMatchGPU');
    const W = ref.cam.width;
    const H = ref.cam.height;
    if (ref.image.width !== W || ref.image.height !== H) throw new Error('PatchMatchGPU: ref image size does not match ref.cam');
    const N = sources.length;
    if (N === 0 || W === 0 || H === 0) return emptyDepthMap(W, H);
    if (geo && geometric!.sourceDepths.length < N) throw new Error('PatchMatchGPU: geometric.sourceDepths must have one depth map per source');
    const maxDim = this.ctx.limits.maxTextureDimension2D;
    if (W > maxDim || H > maxDim) throw new Error(`PatchMatchGPU: image ${W}×${H} exceeds maxTextureDimension2D ${maxDim}`);

    const maxSources = Math.min(MAX_PATCHMATCH_SOURCES, Math.max(4, Math.ceil(N / 4) * 4));
    const pipes = await this.getPipelines(maxSources, o.patchRadius, o.patchStep);
    this.checkAlive();

    // ---- resources & uploads -----------------------------------------------
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
    const device = this.device;
    const q = device.queue;
    const iterations = geo ? o.geometricIterations : o.iterations;
    const budget = Math.max(1, o.maxPixelViewsPerSubmit);
    const halfW = Math.ceil(W / 2);
    const bandProp = Math.max(1, Math.min(H, Math.floor(budget / (halfW * N))));
    const bandInit = Math.max(1, Math.min(H, Math.floor(budget / (W * N))));
    const initBands = Math.ceil(H / bandInit);
    const propBands = Math.ceil(H / bandProp);

    // ---- resources, uploads, per-pass parameters (inside error scopes) --------
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
        q.writeTexture({ texture: srcTex, origin: [0, 0, s] }, grayBuf, { bytesPerRow: v.cam.width, rowsPerImage: v.cam.height }, [
          v.cam.width,
          v.cam.height,
          1,
        ]);
        const dm = geo ? geometric!.sourceDepths[s] : undefined;
        if (dm && dm.width > 0 && dm.height > 0) {
          q.writeTexture({ texture: depthTex, origin: [0, 0, s] }, dm.depth, { bytesPerRow: 4 * dm.width, rowsPerImage: dm.height }, [
            dm.width,
            dm.height,
            1,
          ]);
        }
        cams.push(buildSourceCamera(ref, v, dm ? { width: dm.width, height: dm.height } : undefined));
      }
      q.writeBuffer(this.camsBuf, 0, packSourceCameras(cams, this.srcArr.width, this.srcArr.height));
      if (geo) q.writeBuffer(res.planes, 0, priorToPlanes(geometric!.prior, W, H));

      const numPasses = initBands + iterations * 2 * propBands;
      const paramsBuf = this.ensureParams(numPasses * PARAMS_STRIDE);
      const pdata = new ArrayBuffer(numPasses * PARAMS_STRIDE);
      let slot = 0;
      const writePass = (colour: number, rowStart: number, rowEnd: number, it: number, seed: number, init: boolean): void => {
        const u = new Uint32Array(pdata, slot * PARAMS_STRIDE, PARAMS_BYTES / 4);
        const f = new Float32Array(pdata, slot * PARAMS_STRIDE, PARAMS_BYTES / 4);
        fillParams(u, f, o, ref, range, N, colour, rowStart, rowEnd, it, seed, geo, init);
        slot++;
      };
      const initSeed = passSeed(o.seed, geo ? 0x1ffff : 0xffff);
      for (let b = 0; b < initBands; b++) writePass(0, b * bandInit, Math.min(H, (b + 1) * bandInit), 0, initSeed, true);
      for (let it = 0; it < iterations; it++) {
        for (let colour = 0; colour < 2; colour++) {
          const seed = passSeed(o.seed, (geo ? 0x100 : 0) + 2 * it + colour);
          for (let b = 0; b < propBands; b++) writePass(colour, b * bandProp, Math.min(H, (b + 1) * bandProp), it, seed, false);
        }
      }
      q.writeBuffer(paramsBuf, 0, pdata);
      const bindGroup = device.createBindGroup({
        label: 'pm-bg',
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuf, offset: 0, size: PARAMS_BYTES } },
          { binding: 1, resource: { buffer: this.camsBuf } },
          { binding: 2, resource: res.refTex.createView() },
          { binding: 3, resource: srcTex.createView({ dimension: '2d-array' }) },
          { binding: 4, resource: this.sampler },
          { binding: 5, resource: depthTex.createView({ dimension: '2d-array' }) },
          { binding: 6, resource: { buffer: res.planes, size: 16 * W * H } },
          { binding: 7, resource: { buffer: res.costs, size: 4 * W * H } },
          { binding: 8, resource: { buffer: res.photo, size: 4 * W * H } },
          { binding: 9, resource: { buffer: res.views, size: 4 * W * H } },
        ],
      });
      return { res, bindGroup };
    });
    this.checkAlive();
    const tUpload = performance.now();

    // ---- dispatch: one submission per band and colour, gpuIdle per iteration ----
    let submits = 0;
    const iterTimes: number[] = [];
    let tInit = tUpload;
    let tPrev = tUpload;
    await this.scoped('dispatch', false, async () => {
      let slot = 0;
      const submitPass = (pipeline: GPUComputePipeline, gx: number, gy: number): void => {
        const enc = device.createCommandEncoder({ label: 'pm-pass' });
        const pass = enc.beginComputePass({ label: 'pm' });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup, [slot * PARAMS_STRIDE]);
        pass.dispatchWorkgroups(gx, gy, 1);
        pass.end();
        q.submit([enc.finish()]);
        slot++;
        submits++;
      };
      for (let b = 0; b < initBands; b++) {
        const rows = Math.min(H, (b + 1) * bandInit) - b * bandInit;
        submitPass(pipes.init, Math.ceil(W / WORKGROUP_X), Math.ceil(rows / WORKGROUP_Y));
      }
      await gpuIdle(device);
      this.checkAlive();
      tInit = performance.now();
      tPrev = tInit;
      for (let it = 0; it < iterations; it++) {
        for (let colour = 0; colour < 2; colour++) {
          for (let b = 0; b < propBands; b++) {
            const rows = Math.min(H, (b + 1) * bandProp) - b * bandProp;
            submitPass(pipes.propagate, Math.ceil(halfW / WORKGROUP_X), Math.ceil(rows / WORKGROUP_Y));
          }
        }
        await gpuIdle(device);
        this.checkAlive();
        const t = performance.now();
        iterTimes.push(t - tPrev);
        tPrev = t;
      }
    });

    // ---- readback ---------------------------------------------------------------
    const n = W * H;
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
      bandRows: bandProp,
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
  f[18] = depthPerturbation(o, it, geo);
  f[19] = normalPerturbation(o, it, geo);
  f[20] = o.geomLambda;
  f[21] = o.geomMaxError;
  f[22] = 1 / (3 * 2 * o.sigmaColor * o.sigmaColor);
  f[23] = o.minVariance;
  u[24] = Math.min(o.topK, N);
  u[25] = o.numSamplesPerIter;
  u[26] = geo && init ? 1 : 0;
  u[27] = geo ? 1 : 0;
  f[28] = 1 / (2 * o.sigmaSpatial * o.sigmaSpatial);
  f[29] = 0;
  f[30] = 0;
  f[31] = 0;
}
