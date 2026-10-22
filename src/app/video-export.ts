/**
 * Fly-through video export (MP4, H.264), on the main thread next to the viewer. The camera path is
 * src/viewer/flythrough.ts (from the registered keyframes; depth from the sparse points; up = +Z
 * when gravity aligned, else estimated from the cameras). The frames come from the viewer's splat
 * layer (`Viewer.openSplatFrames`: one command buffer per frame, awaited). They are encoded with
 * WebCodecs `VideoEncoder` and muxed by src/io/mp4.ts.
 *
 * Codec (`chooseVideoConfig`): `VideoEncoder.isConfigSupported` is asked for H.264 High 4.0, then
 * Main 4.0, then Baseline 3.1 (avc1.640028 → 4d0028 → 42e01f) at the frame size, 30 fps and
 * 6 Mbit/s, with `avc: { format: 'avc' }` (AVCC chunks plus the description that becomes the avcC
 * box). Level 3.1 fits 1280 × 720 at 30 fps exactly. The Export sheet shows the entry only when a
 * splat model exists and one of these is supported (`videoExportSupported`, cached). WebCodecs is
 * in Safari 16.4+ and is disabled in Lockdown Mode (docs/research/ios-platform.md).
 *
 * Loop (`renderFlythroughVideo`): per frame, wait until the encoder's queue is empty (so at most one
 * raw frame exists), render, encode (a key frame every 2 s), then close the frame. Encoded chunks
 * stream into the muxer (≈ 7.5 MB for 10 s at 6 Mbit/s). Cancel: an AbortSignal, checked per frame
 * (the encoder is closed and nothing is kept). The UI (`createVideoExport`) stops the export when
 * the page is hidden (iOS may drop GPU work and reclaim the encoder), and so does leaving the
 * viewer or starting a photoreal build. The frame source itself never submits while hidden.
 *
 * `probeVideoExport` (Diagnostics) needs no model. It reports the codec, whether a VideoFrame of a
 * WebGPU OffscreenCanvas holds the rendered pixels (a cleared texture), the encode time per 720p
 * frame, and whether VideoDecoder reads the MP4 back. The first iPhone run thus tells which frame
 * path and codec the export will use. Measurements so far: docs/research/flythrough-video.md.
 */
import { saveFile } from '../io/download';
import { Mp4Writer, readMp4 } from '../io/mp4';
import { getGPU } from '../gpu/device';
import type { SparseResult } from '../workers/protocol';
import { buildFlythrough, flythroughCamera, flythroughTimestampUs, viewingDepth, type FlythroughCamera, type FlythroughPath } from '../viewer/flythrough';
import { estimateUpVector } from '../viewer/helpers';
import type { SplatFramePath, SplatOffscreen, SplatOffscreenOptions } from '../viewer/splat-layer';
import type { App } from './app';
import { button, chip, progressBar } from './components/controls';
import { toast } from './components/toast';
import { h } from './dom';
import { exportFilename, formatBytes } from './format';
import { icon } from './icons';

/** H.264 profiles tried in order: High 4.0, Main 4.0, Baseline 3.1. */
export const VIDEO_CODECS = ['avc1.640028', 'avc1.4d0028', 'avc1.42e01f'] as const;
export const VIDEO_BITRATE = 6_000_000;
export const VIDEO_FPS = 30;

/** The first supported H.264 encoder configuration at this size (null: none, or no WebCodecs). */
export async function chooseVideoConfig(width: number, height: number, fps = VIDEO_FPS, bitrate = VIDEO_BITRATE): Promise<VideoEncoderConfig | null> {
  if (typeof VideoEncoder === 'undefined' || typeof VideoEncoder.isConfigSupported !== 'function') return null;
  for (const codec of VIDEO_CODECS) {
    const config: VideoEncoderConfig = { codec, width, height, bitrate, framerate: fps, avc: { format: 'avc' } };
    try {
      const r = await VideoEncoder.isConfigSupported(config);
      if (r.supported) return { ...config, ...r.config, avc: { format: 'avc' } };
    } catch {
      /* an invalid config for this browser: try the next */
    }
  }
  return null;
}

let supportedPromise: Promise<boolean> | null = null;

/** H.264 encoding at 720p in portrait or landscape (cached for the page). */
export function videoExportSupported(): Promise<boolean> {
  supportedPromise ??= (async () => !!((await chooseVideoConfig(720, 1280)) ?? (await chooseVideoConfig(1280, 720))))().catch(() => false);
  return supportedPromise;
}

/** Where `renderFlythroughVideo` takes its frames (a `SplatOffscreen`). */
export type FlythroughFrameSource = Pick<SplatOffscreen, 'videoFrame' | 'path' | 'totalFrameMs' | 'frames'>;

export interface FlythroughVideoOptions {
  frames: FlythroughFrameSource;
  path: FlythroughPath;
  config: VideoEncoderConfig;
  signal?: AbortSignal;
  onProgress?: (done: number, total: number) => void;
  /** Key frame interval (default 2 s). */
  keyFrameEverySec?: number;
  /** Encoder factory (tests). */
  createEncoder?: (init: VideoEncoderInit) => VideoEncoder;
}

export interface FlythroughVideoResult {
  blob: Blob;
  frames: number;
  width: number;
  height: number;
  seconds: number;
  codec: string;
  framePath: SplatFramePath | null;
  /** Wall time per frame (render + encode + waits) and GPU submit → done per frame (ms). */
  msPerFrame: number;
  renderMsPerFrame: number;
  /** Most raw frames alive at once (1 by construction). */
  maxRawFrames: number;
}

const abortError = (): DOMException => new DOMException('The video export was cancelled', 'AbortError');

/** Renders, encodes and muxes the fly-through. See the file header. */
export async function renderFlythroughVideo(o: FlythroughVideoOptions): Promise<FlythroughVideoResult> {
  const { path, config, signal } = o;
  const N = path.poses.length;
  const mux = new Mp4Writer({ width: path.width, height: path.height, fps: path.fps });
  let failure: unknown = null;
  const encoder = (o.createEncoder ?? ((init) => new VideoEncoder(init)))({
    output: (chunk, meta) => {
      try {
        const desc = meta?.decoderConfig?.description;
        if (desc) mux.setDecoderConfig(desc);
        const b = new Uint8Array(chunk.byteLength);
        chunk.copyTo(b);
        mux.addChunk(b, { key: chunk.type === 'key', timestampUs: chunk.timestamp });
      } catch (e) {
        failure ??= e;
      }
    },
    error: (e) => {
      failure ??= e;
    },
  });
  const keyEvery = Math.max(1, Math.round((o.keyFrameEverySec ?? 2) * path.fps));
  const t0 = performance.now();
  let alive = 0;
  let maxAlive = 0;
  const throwIfStopped = (): void => {
    if (signal?.aborted) throw abortError();
    if (failure) throw failure instanceof Error || failure instanceof DOMException ? failure : new Error(String(failure));
    if (encoder.state === 'closed') throw new Error('The video encoder was closed (the app may have been in the background)');
  };
  try {
    encoder.configure(config);
    for (let i = 0; i < N; i++) {
      throwIfStopped();
      // at most one raw frame: the encoder has taken the previous one before the next is drawn
      while (encoder.encodeQueueSize > 0) {
        await new Promise((r) => setTimeout(r, 2));
        throwIfStopped();
      }
      const ts = flythroughTimestampUs(path, i);
      const frame = await o.frames.videoFrame(flythroughCamera(path, i), ts, flythroughTimestampUs(path, i + 1) - ts);
      alive++;
      maxAlive = Math.max(maxAlive, alive);
      try {
        throwIfStopped();
        encoder.encode(frame, { keyFrame: i % keyEvery === 0 });
      } finally {
        frame.close();
        alive--;
      }
      o.onProgress?.(i + 1, N);
    }
    await encoder.flush();
    throwIfStopped();
  } finally {
    if (encoder.state !== 'closed') encoder.close();
  }
  const blob = mux.finish();
  const ms = performance.now() - t0;
  return {
    blob,
    frames: N,
    width: path.width,
    height: path.height,
    seconds: N / path.fps,
    codec: config.codec,
    framePath: o.frames.path,
    msPerFrame: ms / N,
    renderMsPerFrame: o.frames.frames ? o.frames.totalFrameMs / o.frames.frames : 0,
    maxRawFrames: maxAlive,
  };
}

/** The fly-through path of a sparse model (cameras in capture order). Null with fewer than 2 cameras. */
export function flythroughForModel(sparse: Pick<SparseResult, 'cameras' | 'metric' | 'points'>): FlythroughPath | null {
  const sorted = sparse.cameras.slice().sort((a, b) => a.id - b.id);
  const cams: FlythroughCamera[] = sorted.map((c) => ({ pose: c.pose, intrinsics: c.intrinsics }));
  const up = sparse.metric.gravityAligned ? [0, 0, 1] : (estimateUpVector(sorted.map((c) => ({ R: c.pose.R, upCam: c.upCam ?? null }))) ?? [0, 0, 1]);
  const depth = viewingDepth(cams, sparse.points.positions) ?? undefined;
  return buildFlythrough(cams, { up, depth, fps: VIDEO_FPS });
}

// ------------------------------------------------------------------------------- export sheet row

export interface VideoExportHost {
  app: App;
  /** The viewer's offscreen splat frames (null: no viewer). */
  openFrames(o: SplatOffscreenOptions): SplatOffscreen | null;
  /** Why the splats cannot be rendered now (null: they can). */
  blocked(): string | null;
  /**
   * Another job holds `app.busy` / the wake lock now (the photoreal build): a stopped export leaves
   * them to it.
   */
  othersBusy(): boolean;
  /** Re-render the open Export sheet. */
  rerender(): void;
}

export interface VideoExportController {
  /** The Export sheet row; null when there is no splat model or no H.264 encoder. */
  row(): HTMLElement | null;
  /** Stop a running export (keeps nothing). */
  cancel(): void;
  /** The model changed: drop a prepared video. */
  invalidate(): void;
  dispose(): void;
}

type VideoState =
  | { status: 'idle'; note?: string }
  | { status: 'rendering'; done: number; total: number }
  | { status: 'ready'; blob: Blob; result: FlythroughVideoResult }
  | { status: 'error'; error: string };

/** The "Fly-through video (MP4)" entry of the viewer's Export sheet (self-contained state). */
export function createVideoExport(host: VideoExportHost): VideoExportController {
  const app = host.app;
  let supported: boolean | null = null;
  let disposed = false;
  let state: VideoState = { status: 'idle' };
  let abort: AbortController | null = null;
  let sharingUntil = 0;
  const bar = progressBar('Fly-through video');
  bar.el.style.marginTop = '6px';
  void videoExportSupported().then((ok) => {
    supported = ok;
    if (!disposed) host.rerender();
  });
  const onVisibility = (): void => {
    if (document.hidden && abort) stop('Stopped: the app went to the background. Tap Prepare to start again.');
  };
  document.addEventListener('visibilitychange', onVisibility);
  /** This export holds `app.busy` (leaving asks first) and the wake lock. */
  let holding = false;
  const hold = (): void => {
    if (holding) return;
    holding = true;
    app.busy = true;
    void app.wakeLock.acquire();
  };
  /** Released at once on dispose / a build start: the export settles later, when the next owner may hold them. */
  const unhold = (): void => {
    if (!holding) return;
    holding = false;
    if (host.othersBusy()) return;
    app.busy = false;
    void app.wakeLock.release();
  };

  function stop(note?: string): void {
    if (!abort) return;
    abort.abort();
    state = { status: 'idle', note };
  }

  async function start(): Promise<void> {
    if (abort || disposed) return;
    const why = host.blocked();
    const sparse = app.results.sparse;
    if (why || !sparse) return;
    const path = flythroughForModel(sparse);
    if (!path) {
      state = { status: 'error', error: 'Not enough registered photos for a camera path' };
      return host.rerender();
    }
    const ctl = new AbortController();
    abort = ctl;
    state = { status: 'rendering', done: 0, total: path.poses.length };
    host.rerender();
    let frames: SplatOffscreen | null = null;
    hold();
    try {
      const config = await chooseVideoConfig(path.width, path.height, path.fps);
      if (!config) throw new Error(`This browser cannot encode H.264 video at ${path.width} × ${path.height}`);
      if (ctl.signal.aborted) throw abortError();
      frames = host.openFrames({ width: path.width, height: path.height });
      if (!frames) throw new Error('The photoreal model is not loaded');
      const f = frames;
      // a cancel also stops a frame waiting for the page to become visible
      ctl.signal.addEventListener('abort', () => f.close(), { once: true });
      let lastPaint = 0;
      const result = await renderFlythroughVideo({
        frames,
        path,
        config,
        signal: ctl.signal,
        onProgress: (done, total) => {
          if (state.status !== 'rendering' || abort !== ctl) return;
          state = { status: 'rendering', done, total };
          bar.set(done / total);
          const now = performance.now();
          if (now - lastPaint > 250 || done === total) {
            lastPaint = now;
            host.rerender();
          }
        },
      });
      console.info(
        `[video] ${result.frames} frames ${result.width}×${result.height} ${result.codec}, frame path ${result.framePath}${frames.fallbackReason ? ` (${frames.fallbackReason})` : ''}, ` +
          `${result.msPerFrame.toFixed(1)} ms per frame (GPU ${result.renderMsPerFrame.toFixed(1)} ms), ${formatBytes(result.blob.size)}, mode ${path.mode}, coverage ${path.coverage.map((c) => c.toFixed(2)).join('–')}`,
      );
      if (abort === ctl && !ctl.signal.aborted) state = { status: 'ready', blob: result.blob, result };
    } catch (e) {
      // a cancel already set the idle state (with its note)
      if (abort === ctl && !ctl.signal.aborted) state = { status: 'error', error: (e as Error)?.message ?? String(e) };
    } finally {
      frames?.close();
      if (abort === ctl) {
        abort = null;
        unhold();
      }
      if (!disposed) host.rerender();
    }
  }

  function save(blob: Blob, filename: string): void {
    if (performance.now() < sharingUntil) return;
    sharingUntil = performance.now() + 2000;
    // fresh user activation: saveFile is called synchronously in this tap (iOS share rule)
    saveFile(blob, filename)
      .then(
        (res) => {
          if (res === 'shared') toast('Shared', { tone: 'good' });
          else if (res === 'downloaded') toast(`Saved ${filename}`, { tone: 'good', icon: 'download' });
        },
        (err) => {
          const n = (err as { name?: string })?.name;
          if (n === 'NotAllowedError') toast(`Tap “${app.env.share ? 'Share / Save' : 'Save'}” again to save the video.`, { tone: 'warn' });
          else if (n !== 'InvalidStateError') toast(`Could not save: ${(err as Error)?.message ?? err}`, { tone: 'bad' });
        },
      )
      .finally(() => {
        sharingUntil = 0;
      });
  }

  function row(): HTMLElement | null {
    const r = app.results;
    if (!r.splat || !r.sparse || supported !== true) return null;
    const filename = exportFilename(r.name, 'mp4', 'flythrough');
    const why = host.blocked();
    let sub = state.status === 'idle' && state.note ? state.note : 'A 6–10 s clip along your capture path · 720p H.264, plays anywhere';
    let action: HTMLElement;
    const text = h('div', { class: 'export-text' }, h('span', { class: 'export-title', text: 'Fly-through video (MP4)' }));
    const subEl = h('span', { class: 'export-sub' });
    text.appendChild(subEl);
    const el = h('div', { class: `export-item${state.status === 'error' ? ' error' : ''}`, 'data-testid': 'export-video' });
    if (state.status === 'rendering') {
      sub = `Rendering frame ${state.done} of ${state.total}`;
      bar.set(state.total ? state.done / state.total : -1);
      text.appendChild(bar.el);
      action = button({ label: 'Cancel', variant: 'secondary', size: 'sm', testId: 'export-cancel-video', onClick: () => (stop(), host.rerender()) });
    } else if (why && state.status !== 'ready') {
      sub = why;
      el.classList.add('disabled');
      action = chip('Unavailable', 'neutral');
    } else if (state.status === 'ready') {
      const res = state.result;
      const blob = state.blob;
      sub = `${filename} · ${formatBytes(blob.size)} · ${res.seconds.toFixed(1)} s`;
      el.dataset.codec = res.codec;
      el.dataset.framePath = res.framePath ?? '';
      el.dataset.msPerFrame = res.msPerFrame.toFixed(1);
      action = button({
        label: app.env.share ? 'Share / Save' : 'Save',
        icon: app.env.share ? 'share' : 'download',
        variant: 'primary',
        size: 'sm',
        testId: 'export-save-video',
        onClick: () => save(blob, filename),
      });
    } else {
      if (state.status === 'error') sub = state.error;
      // (disabled while a stopped export is still letting go of the GPU)
      action = button({ label: state.status === 'error' ? 'Retry' : 'Prepare', variant: 'secondary', size: 'sm', testId: 'export-prepare-video', disabled: !!abort, onClick: () => void start() });
    }
    subEl.textContent = sub;
    el.append(h('span', { class: 'export-icon' }, icon('play', { size: 22 })), text, action);
    return el;
  }

  return {
    row,
    cancel() {
      stop();
      unhold();
    },
    invalidate() {
      stop();
      unhold();
      state = { status: 'idle' };
    },
    dispose() {
      disposed = true;
      stop();
      unhold();
      document.removeEventListener('visibilitychange', onVisibility);
    },
  };
}

// ------------------------------------------------------------------------------- Diagnostics probe

export interface VideoProbeReport {
  /** VideoEncoder exists. */
  webCodecs: boolean;
  /** First supported codec of the ladder at 720 × 1280 / 1280 × 720 (null: none). */
  codecPortrait: string | null;
  codecLandscape: string | null;
  /** 'canvas' when a VideoFrame of a WebGPU OffscreenCanvas holds the rendered pixels. */
  framePath: SplatFramePath | null;
  /** Why the canvas path failed (or WebGPU is missing). */
  canvasNote: string | null;
  frames: number;
  /** Wall time per 720 × 1280 frame: draw (a cleared texture), VideoFrame, encode (ms). */
  msPerFrame: number | null;
  bytes: number;
  /** Frames VideoDecoder read back from the MP4 (null: no decoder for it). */
  decoded: number | null;
  error: string | null;
}

/**
 * Encodes `frames` cleared 720 × 1280 frames (a colour ramp) through the export's frame path and
 * encoder into an MP4 and decodes it again. Needs no model; see the file header.
 */
export async function probeVideoExport(frames = 60): Promise<VideoProbeReport> {
  const W = 720;
  const H = 1280;
  const rep: VideoProbeReport = {
    webCodecs: typeof VideoEncoder !== 'undefined',
    codecPortrait: null,
    codecLandscape: null,
    framePath: null,
    canvasNote: null,
    frames: 0,
    msPerFrame: null,
    bytes: 0,
    decoded: null,
    error: null,
  };
  if (!rep.webCodecs) return { ...rep, error: 'WebCodecs VideoEncoder is not available (older iOS or Lockdown Mode)' };
  const cfg = await chooseVideoConfig(W, H);
  rep.codecPortrait = cfg?.codec ?? null;
  rep.codecLandscape = (await chooseVideoConfig(H, W))?.codec ?? null;
  const gpu = typeof navigator !== 'undefined' && navigator.gpu ? await getGPU() : null;
  const colour = (i: number): [number, number, number] => [0.8, 0.2 + (0.6 * i) / Math.max(1, frames - 1), 0.4];
  let make: (i: number, ts: number) => Promise<VideoFrame>;
  let cleanup = (): void => {};
  if (gpu) {
    const device = gpu.device;
    const format = navigator.gpu.getPreferredCanvasFormat();
    const clear = (enc: GPUCommandEncoder, view: GPUTextureView, i: number): void => {
      const [r, g, b] = colour(i);
      enc.beginRenderPass({ colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: { r, g, b, a: 1 } }] }).end();
    };
    let ctx: GPUCanvasContext | null = null;
    let canvas: OffscreenCanvas | null = null;
    try {
      canvas = new OffscreenCanvas(W, H);
      ctx = canvas.getContext('webgpu');
      ctx?.configure({ device, format, alphaMode: 'opaque' });
      if (!ctx) rep.canvasNote = 'OffscreenCanvas has no webgpu context';
    } catch (e) {
      rep.canvasNote = `OffscreenCanvas webgpu: ${(e as Error)?.message ?? e}`;
      ctx = null;
    }
    if (ctx && canvas) {
      // does the canvas VideoFrame show the cleared colour?
      const enc = device.createCommandEncoder();
      clear(enc, ctx.getCurrentTexture().createView(), 0);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      try {
        const f = new VideoFrame(canvas, { timestamp: 0 });
        const c2 = new OffscreenCanvas(8, 8).getContext('2d', { willReadFrequently: true })!;
        c2.drawImage(f, 0, 0, 8, 8);
        f.close();
        const px = c2.getImageData(4, 4, 1, 1).data;
        const want = colour(0).map((v) => Math.round(255 * v));
        const ok = Math.abs(px[0] - want[0]) <= 8 && Math.abs(px[1] - want[1]) <= 8 && Math.abs(px[2] - want[2]) <= 8;
        if (ok) rep.framePath = 'canvas';
        else rep.canvasNote = `the canvas VideoFrame shows rgb(${px[0]}, ${px[1]}, ${px[2]}), not rgb(${want.join(', ')})`;
      } catch (e) {
        rep.canvasNote = `new VideoFrame(OffscreenCanvas): ${(e as Error)?.message ?? e}`;
      }
    }
    if (rep.framePath === 'canvas') {
      const c = ctx!;
      const cv = canvas!;
      make = async (i, ts) => {
        const enc = device.createCommandEncoder();
        clear(enc, c.getCurrentTexture().createView(), i);
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        return new VideoFrame(cv, { timestamp: ts });
      };
    } else {
      rep.framePath = 'readback';
      const bytesPerRow = Math.ceil((4 * W) / 256) * 256;
      const tex = device.createTexture({ size: [W, H], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      const buf = device.createBuffer({ size: bytesPerRow * H, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      make = async (i, ts) => {
        const enc = device.createCommandEncoder();
        clear(enc, tex.createView(), i);
        enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow }, { width: W, height: H });
        device.queue.submit([enc.finish()]);
        await buf.mapAsync(GPUMapMode.READ);
        try {
          return new VideoFrame(buf.getMappedRange(), { format: format === 'bgra8unorm' ? 'BGRX' : 'RGBX', codedWidth: W, codedHeight: H, timestamp: ts, layout: [{ offset: 0, stride: bytesPerRow }] });
        } finally {
          buf.unmap();
        }
      };
      cleanup = () => {
        tex.destroy();
        buf.destroy();
      };
    }
  } else {
    // no WebGPU: CPU frames, the encoder alone
    rep.canvasNote = 'WebGPU is not available';
    const data = new Uint8Array(4 * W * H);
    make = async (i, ts) => {
      const [r, g, b] = colour(i).map((v) => Math.round(255 * v));
      for (let k = 0; k < data.length; k += 4) data.set([r, g, b, 255], k);
      return new VideoFrame(data, { format: 'RGBX', codedWidth: W, codedHeight: H, timestamp: ts });
    };
  }
  try {
    // (the frame path is known by now: reported also without an encoder)
    if (!cfg) throw new Error('No supported H.264 encoder configuration at 720p');
    const path: FlythroughPath = { mode: 'scene', width: W, height: H, fps: VIDEO_FPS, fx: 1, fy: 1, cx: 0, cy: 0, near: 1, poses: [], centre: null, coverage: [0, 1], depth: 1 };
    path.poses = Array.from({ length: frames }, () => ({ R: new Float64Array(9), t: new Float64Array(3) }));
    let i = 0;
    const source: FlythroughFrameSource = { path: rep.framePath, totalFrameMs: 0, frames: 0, videoFrame: (_cam, ts) => make(i++, ts) };
    const res = await renderFlythroughVideo({ frames: source, path, config: cfg });
    rep.frames = res.frames;
    rep.msPerFrame = res.msPerFrame;
    rep.bytes = res.blob.size;
    rep.decoded = await decodeCount(new Uint8Array(await res.blob.arrayBuffer()));
  } catch (e) {
    rep.error = (e as Error)?.message ?? String(e);
  } finally {
    cleanup();
  }
  return rep;
}

/** Frames VideoDecoder decodes from an MP4 of `Mp4Writer` (null when no decoder takes it). */
export async function decodeCount(mp4: Uint8Array, onFrame?: (f: VideoFrame, index: number) => void | Promise<void>): Promise<number | null> {
  if (typeof VideoDecoder === 'undefined') return null;
  const info = readMp4(mp4);
  const config: VideoDecoderConfig = { codec: info.codec, codedWidth: info.width, codedHeight: info.height, description: info.avcC };
  const sup = await VideoDecoder.isConfigSupported(config).catch(() => null);
  if (!sup?.supported) return null;
  let n = 0;
  let failure: unknown = null;
  const pending: Promise<void>[] = [];
  const dec = new VideoDecoder({
    output: (f) => {
      const k = n++;
      if (onFrame) pending.push(Promise.resolve(onFrame(f, k)).finally(() => f.close()));
      else f.close();
    },
    error: (e) => {
      failure = e;
    },
  });
  dec.configure(config);
  for (const s of info.samples) {
    dec.decode(new EncodedVideoChunk({ type: s.key ? 'key' : 'delta', timestamp: Math.round((s.pts * 1e6) / info.timescale), duration: Math.round(1e6 / (info.timescale / info.stts[0][1])), data: mp4.subarray(s.offset, s.offset + s.size) }));
  }
  await dec.flush();
  dec.close();
  await Promise.all(pending);
  if (failure) throw failure;
  return n;
}
