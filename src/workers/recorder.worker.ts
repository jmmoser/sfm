/**
 * Raw capture recorder worker (src/app/recorder.ts): turns the frames the app sends to the engine
 * into JPEGs and writes the recording (format: src/io/recording.ts) to OPFS through sync access
 * handles, or to IndexedDB where OPFS is missing (src/io/recording-store.ts).
 *
 *  - A VideoFrame is converted to a downscaled ImageBitmap AS SOON AS it arrives and closed (it
 *    holds a camera buffer: a held clone can stall the camera, docs/research/ios-camera.md),
 *    exactly like the engine converts its own (`createImageBitmap` with `resizeQuality:
 *    'medium'`, src/engine/engine.ts frameToBitmap); ImageBitmaps arrive already downscaled
 *    (copied on the main thread), RGBAImages as a copy of their pixels.
 *  - Encoding (one reused OffscreenCanvas, `convertToBlob` JPEG) and writing run one frame at a
 *    time in arrival order (RecordingWriter); every frame is acknowledged (`frame`), which is the
 *    main thread's back-pressure signal.
 *  - A sink failure (quota, I/O) is reported once as a fatal `error`; later frames are dropped
 *    (closed) and the main thread stops the recording.
 * Times are never taken here (a worker's performance.now() has its own origin): `sentAt` and the
 * frame times come from the main thread.
 */
import type { RGBAImage } from '../core/types';
import { RecordingWriter, recordedSize, type EncodedImage, type RecordingSink } from '../io/recording';
import { IdbRecordingSink, OpfsRecordingSink, opfsSyncAvailable } from '../io/recording-store';
import type { RecorderWorkerEvent, RecorderWorkerRequest } from '../app/recorder';

interface WorkerScope {
  postMessage(msg: RecorderWorkerEvent): void;
  onmessage: ((ev: MessageEvent<RecorderWorkerRequest>) => void) | null;
}

const scope = self as unknown as WorkerScope;
let writer: RecordingWriter | null = null;
let quality = 0.9;
let maxSide = 1280;
let broken = false;
let canvas: OffscreenCanvas | null = null;
/** Encodes run one at a time (one canvas). */
let encodeChain: Promise<unknown> = Promise.resolve();

function errorInfo(e: unknown): { name: string; message: string } {
  const o = e as { name?: string; message?: string } | null;
  return { name: typeof o?.name === 'string' ? o.name : 'Error', message: typeof o?.message === 'string' ? o.message : String(e) };
}

function isVideoFrame(x: unknown): x is VideoFrame {
  const VF = (globalThis as { VideoFrame?: abstract new (...a: never[]) => VideoFrame }).VideoFrame;
  return !!VF && x instanceof VF;
}

function isRGBA(x: unknown): x is RGBAImage {
  const o = x as RGBAImage | null;
  return !!o && typeof o.width === 'number' && typeof o.height === 'number' && ArrayBuffer.isView(o.data);
}

function closeImage(x: unknown): void {
  try {
    (x as { close?: () => void } | null)?.close?.();
  } catch {
    /* detached */
  }
}

/** The pixels as a drawable at most `w` × `h` (VideoFrames converted and closed right away). */
async function toDrawable(image: VideoFrame | ImageBitmap | RGBAImage, w: number, h: number): Promise<ImageBitmap | OffscreenCanvas> {
  if (isVideoFrame(image)) {
    try {
      const sw = image.displayWidth;
      const sh = image.displayHeight;
      return sw === w && sh === h ? await createImageBitmap(image) : await createImageBitmap(image, { resizeWidth: w, resizeHeight: h, resizeQuality: 'medium' });
    } finally {
      image.close();
    }
  }
  if (isRGBA(image)) {
    const c = new OffscreenCanvas(image.width, image.height);
    const ctx = c.getContext('2d') as OffscreenCanvasRenderingContext2D | null;
    if (!ctx) throw new Error('OffscreenCanvas 2D context unavailable');
    const px = new Uint8ClampedArray(image.data.buffer as ArrayBuffer, image.data.byteOffset, image.data.byteLength);
    ctx.putImageData(new ImageData(px, image.width, image.height), 0, 0);
    return c;
  }
  return image;
}

async function encode(src: ImageBitmap | OffscreenCanvas, w: number, h: number): Promise<EncodedImage> {
  try {
    if (!canvas) canvas = new OffscreenCanvas(w, h);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D | null;
    if (!ctx) throw new Error('OffscreenCanvas 2D context unavailable');
    ctx.drawImage(src, 0, 0, w, h);
    const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
    return { jpeg: new Uint8Array(await blob.arrayBuffer()), width: w, height: h };
  } finally {
    if (!(src instanceof OffscreenCanvas)) src.close();
  }
}

async function openSink(id: string, prefer: 'auto' | 'opfs' | 'idb', idbName?: string): Promise<RecordingSink> {
  if (prefer !== 'idb' && (await opfsSyncAvailable())) return OpfsRecordingSink.open(id);
  if (prefer === 'opfs') throw new Error('OPFS sync access handles are not available');
  return IdbRecordingSink.open(id, idbName);
}

async function handle(req: RecorderWorkerRequest): Promise<void> {
  switch (req.type) {
    case 'open': {
      try {
        if (typeof OffscreenCanvas === 'undefined' || typeof OffscreenCanvas.prototype.convertToBlob !== 'function') throw new Error('OffscreenCanvas JPEG encoding is not available in workers');
        quality = req.jpegQuality;
        maxSide = req.maxSide;
        const sink = await openSink(req.header.id, req.prefer, req.idbName);
        writer = new RecordingWriter(sink);
        const header = { ...req.header, recorder: { ...req.header.recorder, sink: sink.kind } };
        await writer.writeHeader(header);
        scope.postMessage({ type: 'opened', sink: sink.kind });
      } catch (e) {
        scope.postMessage({ type: 'error', fatal: true, ...errorInfo(e) });
      }
      return;
    }
    case 'frame': {
      const { msg, image } = req;
      if (!writer || broken) {
        closeImage(image);
        scope.postMessage({ type: 'frame', seq: msg.seq, ok: false, bytes: 0, error: 'recording stopped' });
        return;
      }
      let enc: Promise<EncodedImage> | null = null;
      if (image) {
        const fit = recordedSize(req.width, req.height, maxSide);
        const drawable = toDrawable(image, fit.width, fit.height);
        drawable.catch(() => undefined);
        const job = encodeChain.then(async () => encode(await drawable, fit.width, fit.height));
        encodeChain = job.catch(() => undefined);
        enc = job;
      }
      try {
        const row = await writer.writeFrame(msg, enc);
        scope.postMessage({ type: 'frame', seq: msg.seq, ok: !!row.image, bytes: row.image?.bytes ?? 0, ...(row.error ? { error: row.error } : {}) });
      } catch (e) {
        scope.postMessage({ type: 'frame', seq: msg.seq, ok: false, bytes: 0, error: errorInfo(e).message });
        if (!broken) {
          broken = true;
          scope.postMessage({ type: 'error', fatal: true, ...errorInfo(e) });
        }
      }
      return;
    }
    case 'live': {
      if (!writer || broken) return;
      try {
        await writer.writeLive(req.row);
      } catch (e) {
        if (!broken) {
          broken = true;
          scope.postMessage({ type: 'error', fatal: true, ...errorInfo(e) });
        }
      }
      return;
    }
    case 'close': {
      const w = writer;
      writer = null;
      try {
        if (w) await w.close(req.header);
        scope.postMessage({ type: 'closed', stats: w ? { ...w.stats } : { frames: 0, imageBytes: 0, lineBytes: 0, failedImages: 0 } });
      } catch (e) {
        scope.postMessage({ type: 'error', fatal: true, ...errorInfo(e) });
        scope.postMessage({ type: 'closed', stats: w ? { ...w.stats } : { frames: 0, imageBytes: 0, lineBytes: 0, failedImages: 0 } });
      }
      return;
    }
  }
}

scope.onmessage = (ev) => {
  void handle(ev.data);
};
