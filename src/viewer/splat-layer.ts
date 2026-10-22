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
 * Offscreen frames (`openOffscreen`, the fly-through video): the same renderer and presenter draw
 * SplatCameras at a fixed size outside the visible canvas, one command buffer per frame awaited
 * (count readback + onSubmittedWorkDone) before the next, no new pipelines. While a session is open
 * the viewer's own frames are not drawn (the canvas keeps its last frame; it is redrawn on close),
 * and the per-pixel buffers are sized to the video frame. A frame waits while the page is hidden,
 * and is drawn again when the page was hidden while it ran (dropped work) or when it overflowed the
 * intersection reservation. Frame source: `new VideoFrame(OffscreenCanvas 'webgpu')`. The first
 * frame of an 'auto' session checks it against a copy of the same texture (PSNR ≥ 35 dB), because
 * WebKit's VideoFrame-from-WebGPU-canvas path is unverified. The fallback copies the texture into a
 * MAP_READ buffer and builds the frame from those bytes ('RGBX' / 'BGRX' with the 256-byte row
 * stride, 3.9 MB per 720p frame).
 *
 * Pipelines are compiled when the model is set. While the page is hidden nothing is submitted
 * (iOS may drop such work); on return the frame is redrawn. A device loss rebuilds the layer on
 * the new device from the host copy of the model (twice), then reports 'unavailable' (the next
 * `setModel` starts over on a new device). Software adapters (SwiftShader) are refused:
 * compiling and drawing would block for minutes.
 *
 * Frame times and GPU memory: docs/research/viewer-rendering.md.
 */
import { getGPU, type GPUContext } from '../gpu/device';
import { SplatPresenter } from '../splat/present';
import { SplatRenderer } from '../splat/renderer';
import type { SplatCamera, SplatModel } from '../splat/types';
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

/** Where an offscreen session's VideoFrames come from. */
export type SplatFramePath = 'canvas' | 'readback';

export interface SplatOffscreenOptions {
  width: number;
  height: number;
  /** sRGB background in [0, 1] (default black). */
  background?: [number, number, number];
  /** 'auto' (default): the canvas path when the session's first frame verifies, else the readback. */
  path?: 'auto' | SplatFramePath;
}

/** Offscreen frames of the layer's model (`SplatLayer.openOffscreen`). */
export interface SplatOffscreen {
  readonly width: number;
  readonly height: number;
  /** Source of the VideoFrames (null until the first `videoFrame`). */
  readonly path: SplatFramePath | null;
  /** Why the canvas path is not used (null while it is, or before the first frame). */
  readonly fallbackReason: string | null;
  /** Submit → done of the last frame and of all frames so far (ms; redraws not counted). */
  readonly lastFrameMs: number;
  readonly totalFrameMs: number;
  readonly frames: number;
  /** Draws `cam` (world frame, OpenCV, `width` × `height`) as a VideoFrame; the caller closes it. */
  videoFrame(cam: SplatCamera, timestampUs: number, durationUs?: number): Promise<VideoFrame>;
  /** Draws `cam` and reads it back (RGBA8, rows top to bottom; tests, quality checks). */
  pixels(cam: SplatCamera): Promise<{ width: number; height: number; data: Uint8Array }>;
  /** Frees the session's GPU resources; the viewer draws again. */
  close(): void;
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
  /** SplatRenderer.gpuBytes: per-Gaussian parameters + projection / sort (~80 B), 16 B per reserved intersection, 20 B per canvas pixel. */
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
  /** `gpu`'s device was lost and the layer has not moved to a new one (it gave up: 'unavailable'). */
  private deviceLost = false;
  private readonly lastSig = new Float64Array(SIG).fill(NaN);
  private readonly sig = new Float64Array(SIG);
  private lastView: SplatView | null = null;
  private pending: SplatView | null = null;
  private inFlight = false;
  /** An offscreen session (video export) is open: the viewer's own frames wait. */
  private offscreenOpen = false;
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
    // a layer that gave up after a device loss starts over on a new device (throws without one)
    if (this.deviceLost && !(await this.moveToNewDevice())) return;
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
    if ((typeof document !== 'undefined' && document.hidden) || this.offscreenOpen) {
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

  /**
   * Opens an offscreen session at `width` × `height` (video export). Throws when the layer is not
   * ready or a session is already open. See the file header.
   */
  openOffscreen(o: SplatOffscreenOptions): SplatOffscreen {
    if (!this.ready || !this.presenter || !this.countBuf) throw new Error('the photoreal model is not loaded');
    if (this.offscreenOpen) throw new Error('an offscreen session is already open');
    const W = Math.round(o.width);
    const H = Math.round(o.height);
    if (!(W >= 16 && H >= 16 && W <= 4096 && H <= 4096)) throw new Error(`offscreen size ${W}×${H}`);
    const layer = this;
    const device0 = this.gpu.device;
    const bg = o.background ?? [0, 0, 0];
    const bgra = this.format === 'bgra8unorm';
    const bytesPerRow = Math.ceil((4 * W) / 256) * 256;
    let path: SplatFramePath | null = o.path === 'canvas' || o.path === 'readback' ? o.path : null;
    let fallbackReason: string | null = null;
    let closed = false;
    let hiddenDuring = false;
    let wake: (() => void) | null = null;
    const onVis = (): void => {
      if (document.hidden) hiddenDuring = true;
      else wake?.();
    };
    document.addEventListener('visibilitychange', onVis);
    const rb = device0.createBuffer({ size: bytesPerRow * H, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'splat-offscreen-readback' });
    let texture: GPUTexture | null = null;
    let canvas: OffscreenCanvas | null = null;
    let ctx: GPUCanvasContext | null = null;
    const stats = { lastFrameMs: 0, totalFrameMs: 0, frames: 0 };
    this.offscreenOpen = true;

    const check = (): void => {
      if (closed) throw new DOMException('offscreen session closed', 'AbortError');
      if (layer.disposed || layer.gpu.device !== device0 || !layer.renderer || !layer.presenter || !layer.countBuf) throw new Error('the photoreal model was unloaded or the GPU device was lost');
    };
    const whenVisible = async (): Promise<void> => {
      while (typeof document !== 'undefined' && document.hidden && !closed) await new Promise<void>((res) => (wake = res));
      wake = null;
    };
    const ownTexture = (): GPUTexture => {
      texture ??= device0.createTexture({ size: [W, H], format: layer.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC, label: 'splat-offscreen' });
      return texture;
    };
    /** The WebGPU OffscreenCanvas (null when it cannot be created / configured here). */
    const canvasContext = (): GPUCanvasContext | null => {
      if (ctx || canvas) return ctx;
      try {
        canvas = new OffscreenCanvas(W, H);
        const c = canvas.getContext('webgpu');
        if (c) {
          c.configure({ device: device0, format: layer.format, alphaMode: 'opaque', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
          ctx = c;
        }
      } catch (e) {
        fallbackReason = `OffscreenCanvas webgpu: ${(e as Error)?.message ?? e}`;
      }
      if (!ctx && !fallbackReason) fallbackReason = 'OffscreenCanvas has no webgpu context';
      return ctx;
    };

    /** Draws one frame into `target` (+ the readback copy), awaited; drawn again after dropped work or an overflow. */
    const render = async (cam: SplatCamera, target: 'canvas' | 'texture', readback: boolean): Promise<void> => {
      if (cam.width !== W || cam.height !== H) throw new Error(`camera ${cam.width}×${cam.height} for a ${W}×${H} session`);
      for (;;) {
        // one heavy command buffer in flight: a viewer frame finishes first
        while (layer.inFlight) await new Promise((r) => setTimeout(r, 2));
        await whenVisible();
        check();
        const r = layer.renderer!;
        if (r.maxWidth !== W || r.maxHeight !== H) r.setMaxSize(W, H);
        const tex = target === 'canvas' ? ctx!.getCurrentTexture() : ownTexture();
        const enc = device0.createCommandEncoder({ label: 'splat-offscreen' });
        const color = r.encodeFrame(enc, cam, { background: bg, countTo: { buffer: layer.countBuf!, offset: 0 } });
        layer.presenter!.encode(enc, tex.createView(), W, H, { buffer: color, width: W, height: H }, false);
        if (readback) enc.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow }, { width: W, height: H });
        hiddenDuring = false;
        const t0 = performance.now();
        device0.queue.submit([enc.finish()]);
        layer.inFlight = true;
        const countBuf = layer.countBuf!;
        let n = -1;
        try {
          const [count] = await Promise.all([
            countBuf.mapAsync(GPUMapMode.READ, 0, 4).then(() => {
              const v = new Uint32Array(countBuf.getMappedRange(0, 4))[0];
              countBuf.unmap();
              return v;
            }),
            device0.queue.onSubmittedWorkDone(),
          ]);
          n = count;
        } finally {
          layer.inFlight = false;
        }
        stats.lastFrameMs = performance.now() - t0;
        check();
        // the page was hidden while the GPU ran it: the work may have been dropped
        if (hiddenDuring || document.hidden) continue;
        const cap = r.isectCapacity;
        if (n > cap && cap < layer.opts.maxIntersections && n * 4 <= device0.limits.maxStorageBufferBindingSize) {
          // the farthest intersections were dropped: grow and draw the frame again
          r.reserveIntersections(Math.min(layer.opts.maxIntersections, Math.ceil(n * 1.25)));
          continue;
        }
        stats.totalFrameMs += stats.lastFrameMs;
        stats.frames++;
        return;
      }
    };

    /** RGBA8 of the mapped readback buffer. */
    const rgba = (src: Uint8Array): Uint8Array => {
      const out = new Uint8Array(4 * W * H);
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++) {
          const s = y * bytesPerRow + 4 * x;
          const d = 4 * (y * W + x);
          out[d] = src[bgra ? s + 2 : s];
          out[d + 1] = src[s + 1];
          out[d + 2] = src[bgra ? s : s + 2];
          out[d + 3] = 255;
        }
      return out;
    };

    /** The readback-path VideoFrame of the frame just rendered with `readback`. */
    const readbackFrame = async (timestamp: number, duration: number | undefined): Promise<VideoFrame> => {
      await rb.mapAsync(GPUMapMode.READ);
      try {
        // (the constructor copies the bytes; the 256-byte-aligned row stride goes in the layout)
        return new VideoFrame(rb.getMappedRange(), { format: bgra ? 'BGRX' : 'RGBX', codedWidth: W, codedHeight: H, timestamp, duration, layout: [{ offset: 0, stride: bytesPerRow }] });
      } finally {
        rb.unmap();
      }
    };

    /** First frame of an 'auto' session: is the canvas VideoFrame the rendered image? */
    const probeCanvas = async (cam: SplatCamera, timestamp: number, duration: number | undefined): Promise<VideoFrame | null> => {
      if (!canvasContext()) return null;
      let frame: VideoFrame | null = null;
      try {
        device0.pushErrorScope('validation');
        let renderError: unknown = null;
        await render(cam, 'canvas', true).catch((e) => (renderError = e));
        const err = await device0.popErrorScope();
        if (renderError) throw renderError;
        if (err) throw new Error(err.message);
        frame = new VideoFrame(canvas!, { timestamp, duration });
        await rb.mapAsync(GPUMapMode.READ);
        const direct = rgba(new Uint8Array(rb.getMappedRange()));
        rb.unmap();
        const c2 = new OffscreenCanvas(W, H).getContext('2d', { willReadFrequently: true });
        if (!c2) throw new Error('no 2D canvas to check the frame');
        c2.drawImage(frame, 0, 0, W, H);
        const seen = c2.getImageData(0, 0, W, H).data;
        let se = 0;
        for (let i = 0; i < direct.length; i += 4) for (let c = 0; c < 3; c++) se += (seen[i + c] - direct[i + c]) ** 2;
        const mse = se / (3 * W * H);
        const psnr = mse > 0 ? 10 * Math.log10((255 * 255) / mse) : Infinity;
        if (frame.codedWidth !== W || frame.codedHeight !== H || psnr < 35)
          throw new Error(`the canvas VideoFrame differs from the rendered image (${frame.codedWidth}×${frame.codedHeight}, PSNR ${psnr.toFixed(1)} dB)`);
        return frame;
      } catch (e) {
        frame?.close();
        if (closed || (e as Error)?.name === 'AbortError') throw e;
        fallbackReason = (e as Error)?.message ?? String(e);
        return null;
      }
    };

    const session: SplatOffscreen = {
      width: W,
      height: H,
      get path() {
        return path;
      },
      get fallbackReason() {
        return fallbackReason;
      },
      get lastFrameMs() {
        return stats.lastFrameMs;
      },
      get totalFrameMs() {
        return stats.totalFrameMs;
      },
      get frames() {
        return stats.frames;
      },
      async videoFrame(cam, timestampUs, durationUs) {
        if (path === null) {
          const f = await probeCanvas(cam, timestampUs, durationUs);
          path = f ? 'canvas' : 'readback';
          if (f) return f;
        }
        if (path === 'canvas') {
          if (!canvasContext()) throw new Error(fallbackReason ?? 'no WebGPU OffscreenCanvas');
          await render(cam, 'canvas', false);
          return new VideoFrame(canvas!, { timestamp: timestampUs, duration: durationUs });
        }
        await render(cam, 'texture', true);
        return readbackFrame(timestampUs, durationUs);
      },
      async pixels(cam) {
        await render(cam, 'texture', true);
        await rb.mapAsync(GPUMapMode.READ);
        try {
          return { width: W, height: H, data: rgba(new Uint8Array(rb.getMappedRange())) };
        } finally {
          rb.unmap();
        }
      },
      close() {
        if (closed) return;
        closed = true;
        wake?.();
        document.removeEventListener('visibilitychange', onVis);
        layer.offscreenOpen = false;
        // (a frame still running keeps them until it completes: destroy defers the free)
        rb.destroy();
        texture?.destroy();
        try {
          ctx?.unconfigure();
        } catch {
          /* ignore */
        }
        if (!layer.disposed) {
          layer.lastSig.fill(NaN);
          layer.opts.onInvalidate?.();
        }
      },
    };
    return session;
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
    // The reaction stays attached to the shared device's `lost` promise for the page's lifetime
    // (pending until a loss): it must not keep a disposed layer — and through its callbacks the
    // Viewer and its screen — reachable.
    const ref = new WeakRef(this);
    void device.lost.then((info) => {
      const layer = ref.deref();
      if (!layer || layer.disposed || layer.gpu.device !== device) return;
      void layer.recover(info.message || String(info.reason));
    });
  }

  private async recover(reason: string): Promise<void> {
    this.deviceLost = true;
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
      if (!(await this.moveToNewDevice())) return;
      await this.build();
      if (this.disposed) return;
      this.modelVersion++;
      this.lastSig.fill(NaN);
      this.opts.onInvalidate?.();
    } catch (e) {
      if (!this.disposed) this.setState('unavailable', (e as Error)?.message ?? String(e));
    }
  }

  /** Adopts the shared context's new device after a loss (false: disposed meanwhile; throws without a usable one). */
  private async moveToNewDevice(): Promise<boolean> {
    const gpu = await getGPU();
    if (this.disposed) return false;
    if (!this.deviceLost) return true; // moved meanwhile (a recovery and a setModel at once)
    if (!gpu || gpu.device === this.gpu.device || (gpu.isSoftware && !this.opts.allowSoftware)) throw new Error('no WebGPU device after the loss');
    this.gpu = gpu;
    this.deviceLost = false;
    this.configure();
    this.watchDevice(gpu.device);
    return true;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.opts.onState = undefined;
    this.opts.onInvalidate = undefined;
    this.lastView = null;
    this.pending = null;
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
