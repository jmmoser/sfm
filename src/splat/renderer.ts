/**
 * Forward-only splat renderer (viewer, evaluation): `render(camera)` → RGB + transmittance per
 * pixel. Holds its own copy of a `SplatModel` on the GPU. The trainer renders with the same
 * kernels (`SplatForward`) on its live parameters. `encodeFrame` records a frame without a
 * readback (the viewer presents the colour buffer with `SplatPresenter`, ./present.ts).
 */
import type { GPUContext } from '../gpu/device';
import { getForwardPipelines, SplatForward, type ForwardPipelines } from './forward';
import { PassProfiler, makeBuffer } from './gpu-util';
import { packParams, paramOffsets } from './layout';
import { TILE_SIZE, type ShDegree, type SplatCamera, type SplatModel } from './types';

export interface SplatRendererOptions {
  shDegree?: ShDegree;
  /** Mip-Splatting opacity compensation (default true; must match how the model was trained). */
  antialiased?: boolean;
  /** Gaussian capacity (default: the first model's count). */
  capacity?: number;
  maxWidth: number;
  maxHeight: number;
  /** Tile intersections reserved (default 12 per Gaussian). */
  isectCapacity?: number;
}

export interface SplatImage {
  width: number;
  height: number;
  /** RGBA float per pixel: colour (background included) and final transmittance in A. */
  data: Float32Array;
  /** Index of the last contributing splat + 1 per pixel (when requested). */
  n?: Uint32Array;
  /** Tile intersections of the frame (uncapped) — when requested. */
  intersections?: number;
  /** GPU ms per pass (timestamp queries, when requested and supported). */
  gpuMs?: Record<string, number>;
}

export interface RenderCallOptions {
  background?: ArrayLike<number>;
  withN?: boolean;
  withStats?: boolean;
  profile?: boolean;
}

export class SplatRenderer {
  private fwd: SplatForward | null = null;
  private params: GPUBuffer | null = null;
  private count = 0;
  private staging: GPUBuffer | null = null;
  private profiler: PassProfiler | null = null;

  private constructor(
    readonly device: GPUDevice,
    private readonly lost: Promise<unknown>,
    private readonly pipes: ForwardPipelines,
    private readonly opts: SplatRendererOptions,
  ) {}

  /** Compiles the pipelines, cached per device (do this at stage start: compilation blocks the queue on Safari). */
  static async create(ctx: Pick<GPUContext, 'device'>, opts: SplatRendererOptions): Promise<SplatRenderer> {
    const pipes = await getForwardPipelines(ctx.device, { shDegree: opts.shDegree ?? 0, antialiased: opts.antialiased ?? true });
    return new SplatRenderer(ctx.device, ctx.device.lost, pipes, { ...opts });
  }

  /** Uploads a model (re-allocates when it exceeds the capacity). */
  setModel(model: SplatModel): void {
    if (model.shDegree !== this.pipes.cfg.shDegree) throw new Error(`SplatRenderer: model SH degree ${model.shDegree}, renderer ${this.pipes.cfg.shDegree}`);
    const cap = Math.max(1, this.opts.capacity ?? 0, model.count);
    if (!this.fwd || this.fwd.capacity < model.count) {
      SplatForward.check({ capacity: cap, isectCapacity: this.opts.isectCapacity }); // before the current model is freed
      this.fwd?.destroy();
      this.params?.destroy();
      this.params = makeBuffer(this.device, paramOffsets(cap, model.shDegree).total * 4, 'splat-render-params');
      this.fwd = this.makeForward(cap, this.opts.isectCapacity);
    }
    this.device.queue.writeBuffer(this.params!, 0, packParams(model, this.fwd.capacity));
    this.count = model.count;
  }

  private makeForward(capacity: number, isectCapacity: number | undefined): SplatForward {
    return new SplatForward(this.device, this.pipes, {
      capacity,
      maxWidth: this.opts.maxWidth,
      maxHeight: this.opts.maxHeight,
      isectCapacity,
      params: this.params!,
      label: 'splat-render',
    });
  }

  get shDegree(): ShDegree {
    return this.pipes.cfg.shDegree;
  }

  get antialiased(): boolean {
    return this.pipes.cfg.antialiased;
  }

  /** Gaussians of the current model. */
  get modelCount(): number {
    return this.count;
  }

  get maxWidth(): number {
    return this.opts.maxWidth;
  }

  get maxHeight(): number {
    return this.opts.maxHeight;
  }

  /** Tile intersections reserved (0 before the first model). */
  get isectCapacity(): number {
    return this.fwd?.isectCapacity ?? 0;
  }

  /**
   * Changes the largest frame (between frames): the per-pixel / per-tile buffers are reallocated,
   * the model and the intersection reservation are kept.
   */
  setMaxSize(maxWidth: number, maxHeight: number): void {
    if (maxWidth === this.opts.maxWidth && maxHeight === this.opts.maxHeight) return;
    this.opts.maxWidth = maxWidth;
    this.opts.maxHeight = maxHeight;
    const f = this.fwd;
    if (!f) return;
    const isect = f.isectCapacity;
    f.destroy();
    this.fwd = this.makeForward(f.capacity, isect);
  }

  /** Reserves `capacity` tile intersections (between frames; only grows). */
  reserveIntersections(capacity: number): void {
    const f = this.fwd;
    if (f && capacity > f.isectCapacity) f.resizeIntersections(Math.ceil(capacity));
  }

  /** GPU memory held (bytes, ±1 %): parameters, projection / sort / intersection and per-pixel buffers. */
  get gpuBytes(): number {
    const f = this.fwd;
    if (!f) return 0;
    const N = f.capacity;
    const px = f.maxWidth * f.maxHeight;
    const tiles = Math.ceil(f.maxWidth / TILE_SIZE) * Math.ceil(f.maxHeight / TILE_SIZE);
    const params = paramOffsets(N, this.pipes.cfg.shDegree).total * 4;
    return params + N * (48 + 8 + 4 + 4 + 16) + f.isectCapacity * 16 + px * 20 + tiles * 8;
  }

  /**
   * Records the forward pass of `cam` without a readback (viewer): when the encoder has run,
   * the returned buffer holds per pixel (x, y), at index y·width + x (rows top to bottom), the
   * colour with T·background added and the final transmittance T (vec4 f32). The frame's tile
   * intersections (uncapped — above `isectCapacity` the farthest were dropped) are copied to
   * `countTo` when given (4 bytes).
   */
  encodeFrame(enc: GPUCommandEncoder, cam: SplatCamera, o: { background?: ArrayLike<number>; countTo?: { buffer: GPUBuffer; offset: number } } = {}): GPUBuffer {
    const f = this.fwd;
    if (!f) throw new Error('SplatRenderer: setModel first');
    f.setFrame(cam, this.count, o.background);
    f.encode(enc);
    if (o.countTo) enc.copyBufferToBuffer(f.binCtl, 13 * 4, o.countTo.buffer, o.countTo.offset, 4);
    return f.outColor;
  }

  /**
   * Renders the model. When the frame needs more tile intersections than reserved, the
   * reservation grows and the frame is rendered again.
   */
  async render(cam: SplatCamera, o: RenderCallOptions = {}): Promise<SplatImage> {
    let r = await this.renderOnce(cam, o);
    const f = this.fwd!;
    const grown = grownIntersectionCapacity(r.intersections!, f.isectCapacity, this.device.limits.maxStorageBufferBindingSize);
    if (grown !== null) {
      f.resizeIntersections(grown);
      r = await this.renderOnce(cam, o);
    }
    if (!o.withStats) delete r.intersections;
    return r;
  }

  private async renderOnce(cam: SplatCamera, o: RenderCallOptions): Promise<SplatImage> {
    const f = this.fwd;
    if (!f) throw new Error('SplatRenderer: setModel first');
    f.setFrame(cam, this.count, o.background);
    const px = cam.width * cam.height;
    const nBytes = o.withN ? px * 4 : 0;
    const need = px * 16 + nBytes + 16;
    if (!this.staging || this.staging.size < need) {
      this.staging?.destroy();
      this.staging = this.device.createBuffer({ size: Math.ceil(need / 16) * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'splat-render-staging' });
    }
    if (o.profile && !this.profiler && PassProfiler.supported(this.device)) this.profiler = new PassProfiler(this.device);
    const prof = o.profile ? this.profiler : null;
    const enc = this.device.createCommandEncoder({ label: 'splat-render' });
    f.encode(enc, prof);
    enc.copyBufferToBuffer(f.outColor, 0, this.staging, 0, px * 16);
    if (o.withN) enc.copyBufferToBuffer(f.outN, 0, this.staging, px * 16, nBytes);
    enc.copyBufferToBuffer(f.binCtl, 13 * 4, this.staging, px * 16 + nBytes, 4);
    prof?.resolve(enc);
    this.device.queue.submit([enc.finish()]);
    const staging = this.staging;
    const r = await Promise.race([staging.mapAsync(GPUMapMode.READ, 0, need).then(() => 'ok' as const), this.lost.then(() => 'lost' as const)]);
    if (r === 'lost') throw new Error('GPU device lost during splat render');
    const buf = staging.getMappedRange(0, need).slice(0);
    staging.unmap();
    const out: SplatImage = { width: cam.width, height: cam.height, data: new Float32Array(buf, 0, 4 * px) };
    if (o.withN) out.n = new Uint32Array(buf, px * 16, px);
    out.intersections = new Uint32Array(buf, px * 16 + nBytes, 1)[0];
    if (prof) {
      const t = await prof.read();
      if (t) out.gpuMs = Object.fromEntries(t);
    }
    return out;
  }

  destroy(): void {
    this.fwd?.destroy();
    this.params?.destroy();
    this.staging?.destroy();
    this.profiler?.destroy();
    this.fwd = null;
  }
}

/** sRGB bytes (RGBA8, alpha 255) of a rendered image. */
export function splatImageToRGBA8(img: SplatImage): Uint8ClampedArray {
  const n = img.width * img.height;
  const out = new Uint8ClampedArray(4 * n);
  for (let i = 0; i < n; i++) {
    out[4 * i] = Math.round(255 * img.data[4 * i]);
    out[4 * i + 1] = Math.round(255 * img.data[4 * i + 1]);
    out[4 * i + 2] = Math.round(255 * img.data[4 * i + 2]);
    out[4 * i + 3] = 255;
  }
  return out;
}

/**
 * The intersection reservation a frame that needed `intersections` regrows to (25 % headroom), or
 * null when it fits or cannot be honoured: each sort buffer holds 4 B per intersection and must stay
 * within `maxBindingBytes` (maxStorageBufferBindingSize), headroom included.
 */
export function grownIntersectionCapacity(intersections: number, capacity: number, maxBindingBytes: number): number | null {
  const max = Math.floor(maxBindingBytes / 4);
  if (!(intersections > capacity) || intersections > max) return null;
  return Math.min(Math.ceil(intersections * 1.25), max);
}
