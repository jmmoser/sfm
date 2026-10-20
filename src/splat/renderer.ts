/**
 * Forward-only splat renderer (viewer, evaluation): `render(camera)` → RGB + transmittance per
 * pixel. Holds its own copy of a `SplatModel` on the GPU. The trainer renders with the same
 * kernels (`SplatForward`) on its live parameters.
 */
import type { GPUContext } from '../gpu/device';
import { getForwardPipelines, SplatForward, type ForwardPipelines } from './forward';
import { PassProfiler, makeBuffer } from './gpu-util';
import { packParams, paramOffsets } from './layout';
import type { ShDegree, SplatCamera, SplatModel } from './types';

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
    private readonly device: GPUDevice,
    private readonly lost: Promise<unknown>,
    private readonly pipes: ForwardPipelines,
    private readonly opts: SplatRendererOptions,
  ) {}

  /** Compiles the pipelines, cached per device (do this at stage start: compilation blocks the queue on Safari). */
  static async create(ctx: Pick<GPUContext, 'device'>, opts: SplatRendererOptions): Promise<SplatRenderer> {
    const pipes = await getForwardPipelines(ctx.device, { shDegree: opts.shDegree ?? 0, antialiased: opts.antialiased ?? true });
    return new SplatRenderer(ctx.device, ctx.device.lost, pipes, opts);
  }

  /** Uploads a model (re-allocates when it exceeds the capacity). */
  setModel(model: SplatModel): void {
    if (model.shDegree !== this.pipes.cfg.shDegree) throw new Error(`SplatRenderer: model SH degree ${model.shDegree}, renderer ${this.pipes.cfg.shDegree}`);
    const cap = Math.max(1, this.opts.capacity ?? 0, model.count);
    if (!this.fwd || this.fwd.capacity < model.count) {
      this.fwd?.destroy();
      this.params?.destroy();
      this.params = makeBuffer(this.device, paramOffsets(cap, model.shDegree).total * 4, 'splat-render-params');
      this.fwd = new SplatForward(this.device, this.pipes, {
        capacity: cap,
        maxWidth: this.opts.maxWidth,
        maxHeight: this.opts.maxHeight,
        isectCapacity: this.opts.isectCapacity,
        params: this.params,
        label: 'splat-render',
      });
    }
    this.device.queue.writeBuffer(this.params!, 0, packParams(model, this.fwd.capacity));
    this.count = model.count;
  }

  async render(cam: SplatCamera, o: RenderCallOptions = {}): Promise<SplatImage> {
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
    if (o.withStats) out.intersections = new Uint32Array(buf, px * 16 + nBytes, 1)[0];
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
