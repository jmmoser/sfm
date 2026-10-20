/**
 * Splat display layer of the Viewer: the trained Gaussians drawn by the WebGPU `SplatRenderer`
 * (the trainer's own forward kernels, so the viewer shows exactly what was trained) into a canvas
 * stacked UNDER the three.js WebGL canvas, which then clears to transparent and draws only the
 * overlays (camera frustums, measurement markers). Loaded lazily by the Viewer (own chunk).
 *
 * Frames: `draw(view)` per viewer frame; a frame is rendered only when the camera, the size, the
 * model or the background changed since the last one (the canvas keeps showing the last
 * presented frame). One command buffer per frame (forward + present, no readback but a 4-byte
 * intersection count) and at most one in flight (iOS rule, ARCHITECTURE §2.1): a frame asked for
 * while one runs is coalesced into the latest view and drawn when it completes. Pixel budget:
 * the canvas resolution is devicePixelRatio capped by `maxPixelRatio` and `maxPixels`; while the
 * view moves a frame that took > 16 ms at full size is drawn at ≤ `interactivePixels` and
 * upscaled (bilinear), the still view at full size. Intersections beyond the reservation (the
 * farthest are dropped for that frame) grow it up to `maxIntersections` and redraw.
 *
 * Pipelines are compiled when the model is set. While the page is hidden nothing is submitted
 * (iOS may drop such work); on return the frame is redrawn. A device loss rebuilds the layer on
 * the new device from the host copy of the model (twice), then reports 'unavailable'. Software
 * adapters (SwiftShader) are refused: compiling and drawing would block for minutes.
 */
import { getGPU, type GPUContext } from '../gpu/device';
import { SplatPresenter } from '../splat/present';
import { SplatRenderer } from '../splat/renderer';
import type { SplatModel } from '../splat/types';
import { interactiveFrameSize, splatCameraFromView, splatCanvasSize } from './splat-camera';

export type SplatLayerState = 'loading' | 'ready' | 'lost' | 'unavailable';

export interface SplatLayerOptions {
  /** Most canvas pixels (default 2 M). */
  maxPixels?: number;
  /** Cap of the canvas pixels per CSS pixel (default 2, like the WebGL canvas). */
  maxPixelRatio?: number;
  /** Most pixels of a frame drawn while the view moves (default 0.5 M). */
  interactivePixels?: number;
  /** Cap on the tile-intersection reservation (default 6 M = 96 MB). */
  maxIntersections?: number;
  /** Device loss: rebuild attempts before giving up (default 2). */
  maxRecoveries?: number;
  /** State changes after the first model upload ('lost' while recovering, 'unavailable' when it gave up). */
  onState?: (state: SplatLayerState, detail?: string) => void;
  /** The layer needs a new frame (grown reservation, page visible again, recovered device). */
  onInvalidate?: () => void;
  /** Accept a software adapter (tests). */
  allowSoftware?: boolean;
}

/** What a frame shows (from the viewer). */
export interface SplatView {
  /** World → three camera (camera.matrixWorldInverse × worldRoot.matrixWorld), column-major. */
  view: ArrayLike<number>;
  /** camera.projectionMatrix, column-major. */
  projection: ArrayLike<number>;
  cssWidth: number;
  cssHeight: number;
  /** Near plane (world units, camera z). */
  near: number;
  /** sRGB background in [0, 1]; null: transparent (premultiplied canvas). */
  background: [number, number, number] | null;
  /** The view is moving (a reduced frame may be drawn). */
  interactive: boolean;
}

export interface SplatLayerStats {
  state: SplatLayerState;
  gaussians: number;
  /** Frames submitted. */
  frames: number;
  /** Size of the last frame rendered (may be below the canvas while moving). */
  frameWidth: number;
  frameHeight: number;
  canvasWidth: number;
  canvasHeight: number;
  /** Wall time of the last full-size frame, submit → done (ms). */
  fullFrameMs: number;
  /** Tile intersections of the last frame and the reservation. */
  intersections: number;
  isectCapacity: number;
  gpuMB: number;
}

const SIG = 40;

export class SplatLayer {
  readonly canvas: HTMLCanvasElement;
  private readonly opts: Required<Omit<SplatLayerOptions, 'onState' | 'onInvalidate'>> & Pick<SplatLayerOptions, 'onState' | 'onInvalidate'>;
  private gpu: GPUContext;
  private context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private presenter: SplatPresenter | null = null;
  private renderer: SplatRenderer | null = null;
  private model: SplatModel | null = null;
  private antialiased = true;
  private modelVersion = 0;
  private state: SplatLayerState = 'loading';
  private visible = true;
  private disposed = false;
  private recoveries = 0;
  private readonly lastSig = new Float64Array(SIG).fill(NaN);
  private readonly sig = new Float64Array(SIG);
  private lastView: SplatView | null = null;
  private pending: SplatView | null = null;
  private inFlight = false;
  private countBuf: GPUBuffer | null = null;
  private countDevice: GPUDevice | null = null;
  /** The canvas texture can be copied (readPixels / screenshots). */
  private copySrc = true;
  private frames = 0;
  private frameW = 0;
  private frameH = 0;
  private fullFrameMs = 0;
  private intersections = 0;
  private readonly onVisibility = (): void => {
    if (document.hidden || this.disposed) return;
    // work submitted while hidden may have been dropped: draw again
    this.lastSig.fill(NaN);
    this.opts.onInvalidate?.();
  };

  private constructor(host: HTMLElement, before: Element | null, gpu: GPUContext, context: GPUCanvasContext, canvas: HTMLCanvasElement, opts: SplatLayerOptions) {
    this.opts = {
      maxPixels: opts.maxPixels ?? 2_000_000,
      maxPixelRatio: opts.maxPixelRatio ?? 2,
      interactivePixels: opts.interactivePixels ?? 500_000,
      maxIntersections: opts.maxIntersections ?? 6_000_000,
      maxRecoveries: opts.maxRecoveries ?? 2,
      allowSoftware: opts.allowSoftware ?? false,
      onState: opts.onState,
      onInvalidate: opts.onInvalidate,
    };
    this.gpu = gpu;
    this.canvas = canvas;
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    host.insertBefore(canvas, before);
    document.addEventListener('visibilitychange', this.onVisibility);
    this.watchDevice(gpu.device);
  }

  /** Creates the canvas (inserted into `host` before `before`). Rejects without a usable WebGPU device. */
  static async create(host: HTMLElement, before: Element | null, opts: SplatLayerOptions = {}): Promise<SplatLayer> {
    const gpu = await getGPU();
    if (!gpu) throw new Error('WebGPU is not available');
    if (gpu.isSoftware && !opts.allowSoftware) throw new Error('software WebGPU adapter: splat rendering disabled');
    const canvas = document.createElement('canvas');
    canvas.className = 'viewer-splats';
    Object.assign(canvas.style, {
      position: 'absolute',
      left: '0',
      top: '0',
      width: '100%',
      height: '100%',
      display: 'block',
      pointerEvents: 'none',
    } satisfies Partial<CSSStyleDeclaration>);
    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('WebGPU canvas context unavailable');
    const layer = new SplatLayer(host, before, gpu, context, canvas, opts);
    // COPY_SRC (readPixels: tests, screenshots) is checked once: without it the layer still draws
    const device = gpu.device;
    device.pushErrorScope('validation');
    layer.configure();
    const err = await device.popErrorScope().catch(() => null);
    if (err) {
      console.warn('[splat-layer] canvas without COPY_SRC:', err.message);
      layer.copySrc = false;
      layer.configure();
    }
    return layer;
  }

  private configure(): void {
    this.context.configure({
      device: this.gpu.device,
      format: this.format,
      alphaMode: 'premultiplied',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | (this.copySrc ? GPUTextureUsage.COPY_SRC : 0),
    });
  }

  /**
   * Uploads a model (host arrays are kept for a device-loss rebuild, not copied). Compiles the
   * renderer's pipelines on first use (or for another SH degree / antialiasing mode).
   */
  async setModel(model: SplatModel, antialiased = true): Promise<void> {
    if (this.disposed) throw new Error('SplatLayer disposed');
    this.model = model;
    this.antialiased = antialiased;
    await this.build();
    this.modelVersion++;
    this.lastSig.fill(NaN);
  }

  /** (Re)creates presenter + renderer on the current device and uploads the model. */
  private async build(): Promise<void> {
    const model = this.model;
    if (!model) return;
    const device = this.gpu.device;
    const r0 = this.renderer;
    const reuse = r0 && r0.shDegree === model.shDegree && r0.antialiased === this.antialiased && r0.device === device;
    if (!this.presenter || this.presenter.device !== device) {
      this.presenter?.destroy();
      this.presenter = null;
      const p = await SplatPresenter.create(device, this.format);
      if (this.disposed) {
        p.destroy();
        return;
      }
      this.presenter = p;
    }
    if (!reuse) {
      const w = Math.max(16, this.canvas.width);
      const h = Math.max(16, this.canvas.height);
      const r = await SplatRenderer.create(this.gpu, {
        shDegree: model.shDegree,
        antialiased: this.antialiased,
        maxWidth: w,
        maxHeight: h,
        // a close-up at viewer resolution needs more than the trainer's 12 per Gaussian
        isectCapacity: Math.min(this.opts.maxIntersections, Math.max(1 << 20, 16 * model.count)),
      });
      if (this.disposed) {
        r.destroy();
        return;
      }
      this.renderer?.destroy();
      this.renderer = r;
    }
    this.renderer!.setModel(model);
    if (!this.countBuf || this.countDevice !== device) {
      this.countBuf?.destroy();
      this.countBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'splat-layer-count' });
      this.countDevice = device;
    }
    this.setState('ready');
  }

  private setState(s: SplatLayerState, detail?: string): void {
    if (this.state === s) return;
    this.state = s;
    this.opts.onState?.(s, detail);
  }

  get ready(): boolean {
    return this.state === 'ready' && !!this.renderer && !this.disposed;
  }

  setVisible(v: boolean): void {
    if (this.visible === v) return;
    this.visible = v;
    this.canvas.style.visibility = v ? 'visible' : 'hidden';
    if (v) this.lastSig.fill(NaN);
  }

  /** Draws `view` unless it equals the last frame (no-op while hidden, not ready, or invisible). */
  draw(view: SplatView): void {
    if (!this.ready || !this.visible) return;
    this.lastView = view;
    if (typeof document !== 'undefined' && document.hidden) {
      this.lastSig.fill(NaN);
      return;
    }
    if (this.inFlight) {
      this.pending = view;
      return;
    }
    const size = splatCanvasSize(view.cssWidth, view.cssHeight, devicePixelRatio, this.opts.maxPixelRatio, this.opts.maxPixels);
    if (size.width === 0) return;
    const frame = view.interactive && this.fullFrameMs > 16 ? interactiveFrameSize(size.width, size.height, this.opts.interactivePixels) : size;
    const sig = this.signature(view, size.width, size.height, frame.width, frame.height);
    let same = true;
    for (let i = 0; i < SIG && same; i++) same = sig[i] === this.lastSig[i];
    if (same) return;
    this.submit(view, size.width, size.height, frame.width, frame.height, null);
  }

  private signature(v: SplatView, cw: number, ch: number, fw: number, fh: number): Float64Array {
    const s = this.sig;
    for (let i = 0; i < 16; i++) s[i] = v.view[i];
    for (let i = 0; i < 16; i++) s[16 + i] = v.projection[i];
    s[32] = cw;
    s[33] = ch;
    s[34] = fw;
    s[35] = fh;
    s[36] = v.near;
    s[37] = this.modelVersion;
    s[38] = v.background ? v.background[0] + 2 * v.background[1] + 4 * v.background[2] : -1;
    s[39] = v.background ? v.background[0] * 7 + v.background[2] : -1;
    return s;
  }

  /**
   * Encodes and submits one frame; with `readback` the canvas texture is also copied into it
   * (bytes per row aligned to 256).
   */
  private submit(view: SplatView, cw: number, ch: number, fw: number, fh: number, readback: { buffer: GPUBuffer; bytesPerRow: number } | null): Promise<void> {
    const r = this.renderer!;
    const device = this.gpu.device;
    if (this.canvas.width !== cw || this.canvas.height !== ch) {
      this.canvas.width = cw;
      this.canvas.height = ch;
    }
    // per-pixel buffers for the canvas size (reallocated when it does not fit or is far too big)
    if (cw > r.maxWidth || ch > r.maxHeight || 2 * cw * ch < r.maxWidth * r.maxHeight) r.setMaxSize(cw, ch);
    const cam = splatCameraFromView(view.view, view.projection, fw, fh, view.near);
    const enc = device.createCommandEncoder({ label: 'splat-layer' });
    const color = r.encodeFrame(enc, cam, { background: view.background ?? [0, 0, 0], countTo: { buffer: this.countBuf!, offset: 0 } });
    const tex = this.context.getCurrentTexture();
    this.presenter!.encode(enc, tex.createView(), cw, ch, { buffer: color, width: fw, height: fh }, !view.background);
    if (readback) enc.copyTextureToBuffer({ texture: tex }, { buffer: readback.buffer, bytesPerRow: readback.bytesPerRow }, { width: cw, height: ch });
    device.queue.submit([enc.finish()]);
    this.lastSig.set(this.signature(view, cw, ch, fw, fh));
    this.inFlight = true;
    this.frames++;
    this.frameW = fw;
    this.frameH = fh;
    const t0 = performance.now();
    const countBuf = this.countBuf!;
    const done = countBuf.mapAsync(GPUMapMode.READ, 0, 4).then(
      () => {
        const n = new Uint32Array(countBuf.getMappedRange(0, 4))[0];
        countBuf.unmap();
        return n;
      },
      () => -1, // device lost / destroyed: handled by the lost watcher
    );
    return done.then((n) => {
      this.inFlight = false;
      if (this.disposed || n < 0) return;
      if (fw === cw && fh === ch) this.fullFrameMs = performance.now() - t0;
      this.intersections = n;
      const cap = r.isectCapacity;
      if (n > cap && cap < this.opts.maxIntersections && n * 4 <= device.limits.maxStorageBufferBindingSize) {
        // the farthest intersections of this frame were dropped: grow and draw it again
        r.reserveIntersections(Math.min(this.opts.maxIntersections, Math.ceil(n * 1.25)));
        this.lastSig.fill(NaN);
        if (!this.pending) this.pending = this.lastView;
      }
      const p = this.pending;
      this.pending = null;
      if (p) this.draw(p);
    });
  }

  /**
   * Draws the last view at full size now and reads the canvas back (RGBA8, rows top to bottom,
   * premultiplied when the background is transparent). Null when nothing can be drawn.
   */
  async readPixels(): Promise<{ width: number; height: number; data: Uint8Array } | null> {
    const view = this.lastView;
    if (!this.ready || !view || !this.copySrc) return null;
    while (this.inFlight) await new Promise((res) => setTimeout(res, 4));
    if (!this.ready) return null;
    const size = splatCanvasSize(view.cssWidth, view.cssHeight, devicePixelRatio, this.opts.maxPixelRatio, this.opts.maxPixels);
    if (size.width === 0) return null;
    const bytesPerRow = Math.ceil((4 * size.width) / 256) * 256;
    const device = this.gpu.device;
    const buffer = device.createBuffer({ size: bytesPerRow * size.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'splat-layer-readback' });
    try {
      await this.submit(view, size.width, size.height, size.width, size.height, { buffer, bytesPerRow });
      await buffer.mapAsync(GPUMapMode.READ);
      const src = new Uint8Array(buffer.getMappedRange());
      const data = new Uint8Array(4 * size.width * size.height);
      const bgra = this.format === 'bgra8unorm';
      for (let y = 0; y < size.height; y++) {
        for (let x = 0; x < size.width; x++) {
          const s = y * bytesPerRow + 4 * x;
          const d = 4 * (y * size.width + x);
          data[d] = src[bgra ? s + 2 : s];
          data[d + 1] = src[s + 1];
          data[d + 2] = src[bgra ? s : s + 2];
          data[d + 3] = src[s + 3];
        }
      }
      buffer.unmap();
      return { width: size.width, height: size.height, data };
    } finally {
      buffer.destroy();
    }
  }

  stats(): SplatLayerStats {
    const r = this.renderer;
    return {
      state: this.state,
      gaussians: r?.modelCount ?? 0,
      frames: this.frames,
      frameWidth: this.frameW,
      frameHeight: this.frameH,
      canvasWidth: this.canvas.width,
      canvasHeight: this.canvas.height,
      fullFrameMs: this.fullFrameMs,
      intersections: this.intersections,
      isectCapacity: r?.isectCapacity ?? 0,
      gpuMB: r ? Math.round(r.gpuBytes / 1e5) / 10 : 0,
    };
  }

  // ---------------------------------------------------------------- device loss

  private watchDevice(device: GPUDevice): void {
    void device.lost.then((info) => {
      if (this.disposed || this.gpu.device !== device) return;
      if (info.reason === 'destroyed' && this.disposed) return;
      void this.recover(info.message || String(info.reason));
    });
  }

  private async recover(reason: string): Promise<void> {
    this.inFlight = false;
    this.pending = null;
    this.renderer = null; // its buffers died with the device
    this.presenter = null;
    this.countBuf = null;
    if (this.recoveries >= this.opts.maxRecoveries) {
      this.setState('unavailable', `GPU device lost: ${reason}`);
      return;
    }
    this.recoveries++;
    this.setState('lost', reason);
    try {
      const gpu = await getGPU();
      if (this.disposed) return;
      if (!gpu || (gpu.isSoftware && !this.opts.allowSoftware)) throw new Error('no WebGPU device after the loss');
      this.gpu = gpu;
      this.configure();
      this.watchDevice(gpu.device);
      await this.build();
      if (this.disposed) return;
      this.modelVersion++;
      this.lastSig.fill(NaN);
      this.opts.onInvalidate?.();
    } catch (e) {
      if (!this.disposed) this.setState('unavailable', (e as Error)?.message ?? String(e));
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.renderer?.destroy();
    this.presenter?.destroy();
    this.countBuf?.destroy();
    this.renderer = null;
    this.presenter = null;
    this.countBuf = null;
    this.model = null;
    try {
      this.context.unconfigure();
    } catch {
      /* ignore */
    }
    this.canvas.remove();
  }
}
