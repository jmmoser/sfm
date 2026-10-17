/**
 * Camera worker logic (docs/research/ios-camera.md §5.3, §7 step 6): reads a transferred
 * MediaStreamTrack with MediaStreamTrackProcessor (DedicatedWorker-only in WebKit, Safari 18+)
 * and converts frames to ImageBitmaps on request.
 *
 *  - Every frame is announced to the main thread (`frame` event, a few numbers) and the last
 *    `KEEP_FRAMES` stay open, so the main thread can ask for exactly the frame whose timestamp it
 *    paired with the IMU (`grab` with that `seq`). A frame being converted is closed only when its
 *    conversion finished. Older frames are closed at once: the capture buffer pool is small —
 *    keeping 2 open (next to the `<video>` preview's) stalled Chromium's fake camera after 3
 *    frames, so 1 it is; the main thread skips announcements that are already old instead.
 *  - WebKit ignores `resizeQuality` for `createImageBitmap(VideoFrame)` (ImageBitmap.cpp draws
 *    with the default filter), so downscaled grabs may be filtered differently than on the
 *    `<video>` path.
 *  - Frames keep the sensor orientation on iOS (raw sensor-landscape buffers, no
 *    `VideoFrame.rotation`); the engine's OrientationTracker handles that (frame-ingest.ts).
 *  - Environment access is injected (`CameraWorkerEnv`) so the logic runs in Node tests.
 */
import type { CameraWorkerEvent, CameraWorkerRequest } from './mstp-protocol';

export interface FrameLike {
  readonly timestamp: number;
  readonly displayWidth: number;
  readonly displayHeight: number;
  readonly codedWidth: number;
  readonly codedHeight: number;
  readonly format: string | null;
  close(): void;
}

/** Frames kept open for grabs (more starve the capture buffer pool, see above). */
export const KEEP_FRAMES = 1;

export interface CameraWorkerEnv {
  post(msg: CameraWorkerEvent, transfer?: Transferable[]): void;
  /** null = MediaStreamTrackProcessor is not available. */
  createReader(track: MediaStreamTrack, maxBufferSize: number): ReadableStreamDefaultReader<FrameLike> | null;
  toBitmap(frame: FrameLike, opts: { maxSide?: number; resizeQuality?: ResizeQuality }): Promise<ImageBitmap>;
  now(): number;
  /** performance.timeOrigin, read when called (WebKit converts it to wall time at each read). */
  timeOrigin(): number;
}

interface Current {
  frame: FrameLike;
  seq: number;
}

export class CameraWorkerCore {
  private gen = -1;
  private track: MediaStreamTrack | null = null;
  private reader: ReadableStreamDefaultReader<FrameLike> | null = null;
  /** Open frames, oldest first; the last one is the current frame. */
  private frames: Current[] = [];
  private seq = 0;
  /** Frames being converted (pin count); closed when the last conversion finished, unless current. */
  private readonly pinned = new Map<FrameLike, number>();

  constructor(private readonly env: CameraWorkerEnv) {}

  handle(msg: CameraWorkerRequest): void {
    switch (msg.type) {
      case 'attach':
        this.attach(msg.gen, msg.track ?? null, (msg.readable as ReadableStream<FrameLike> | undefined) ?? null, msg.maxBufferSize);
        return;
      case 'grab':
        void this.grab(msg);
        return;
      case 'detach':
        this.detach();
        return;
    }
  }

  private attach(gen: number, track: MediaStreamTrack | null, readable: ReadableStream<FrameLike> | null, maxBufferSize: number): void {
    this.detach();
    this.gen = gen;
    this.track = track;
    let reader: ReadableStreamDefaultReader<FrameLike> | null = null;
    try {
      if (readable) reader = readable.getReader();
      else if (track) reader = this.env.createReader(track, maxBufferSize);
      else throw new Error('nothing to read');
    } catch (e) {
      this.fail(gen, `MediaStreamTrackProcessor failed: ${errMessage(e)}`);
      return;
    }
    if (!reader) {
      this.fail(gen, 'MediaStreamTrackProcessor is not available in workers', true);
      return;
    }
    this.reader = reader;
    this.env.post({ type: 'attached', gen, timeOrigin: this.env.timeOrigin() });
    void this.readLoop(gen, reader);
  }

  private fail(gen: number, message: string, unsupported = false): void {
    this.stopTrack();
    this.env.post({ type: 'error', gen, message, ...(unsupported ? { unsupported } : {}) });
  }

  private async readLoop(gen: number, reader: ReadableStreamDefaultReader<FrameLike>): Promise<void> {
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        if (gen !== this.gen) {
          value.close();
          break;
        }
        const seq = this.setCurrent(value);
        this.env.post({
          type: 'frame',
          gen,
          seq,
          timestampUs: value.timestamp,
          receivedMs: this.env.now(),
          width: value.displayWidth,
          height: value.displayHeight,
          codedWidth: value.codedWidth,
          codedHeight: value.codedHeight,
          format: value.format ?? null,
        });
      }
      if (gen === this.gen) {
        this.dropCurrent();
        this.env.post({ type: 'ended', gen });
      }
    } catch (e) {
      if (gen === this.gen) {
        this.dropCurrent();
        this.env.post({ type: 'error', gen, message: `reading frames failed: ${errMessage(e)}` });
      }
    }
  }

  private setCurrent(frame: FrameLike): number {
    const seq = ++this.seq;
    this.frames.push({ frame, seq });
    while (this.frames.length > KEEP_FRAMES) this.release(this.frames.shift()!.frame);
    return seq;
  }

  private dropCurrent(): void {
    const old = this.frames;
    this.frames = [];
    for (const f of old) this.release(f.frame);
  }

  /** Close a frame that is no longer kept, unless a conversion still uses it. */
  private release(frame: FrameLike): void {
    if (this.pinned.has(frame) || this.frames.some((f) => f.frame === frame)) return;
    try {
      frame.close();
    } catch {
      /* already closed */
    }
  }

  private async grab(msg: Extract<CameraWorkerRequest, { type: 'grab' }>): Promise<void> {
    const cur = msg.gen === this.gen ? this.frames.find((f) => f.seq === msg.seq) : undefined;
    if (!cur) {
      const stale = msg.gen === this.gen && this.frames.length > 0 && msg.seq < this.frames[0].seq;
      this.env.post({ type: 'grabbed', id: msg.id, error: stale ? 'stale' : 'no frame' });
      return;
    }
    const frame = cur.frame;
    this.pinned.set(frame, (this.pinned.get(frame) ?? 0) + 1);
    try {
      const bitmap = await this.env.toBitmap(frame, { maxSide: msg.maxSide, resizeQuality: msg.resizeQuality });
      this.env.post({ type: 'grabbed', id: msg.id, bitmap }, [bitmap]);
    } catch (e) {
      this.env.post({ type: 'grabbed', id: msg.id, error: errMessage(e) });
    } finally {
      const n = (this.pinned.get(frame) ?? 1) - 1;
      if (n > 0) this.pinned.set(frame, n);
      else {
        this.pinned.delete(frame);
        this.release(frame);
      }
    }
  }

  private stopTrack(): void {
    try {
      this.track?.stop();
    } catch {
      /* ignore */
    }
    this.track = null;
  }

  private detach(): void {
    const reader = this.reader;
    this.reader = null;
    this.gen = -1;
    if (reader) reader.cancel().catch(() => undefined);
    this.dropCurrent();
    this.stopTrack();
  }
}

/** Long side ≤ maxSide (never enlarged); null = keep the native size. */
export function grabSize(w: number, h: number, maxSide?: number): { width: number; height: number } | null {
  if (!maxSide || maxSide <= 0 || Math.max(w, h) <= maxSide) return null;
  const s = maxSide / Math.max(w, h);
  return { width: Math.max(1, Math.round(w * s)), height: Math.max(1, Math.round(h * s)) };
}

/** VideoFrame → ImageBitmap: createImageBitmap (resized when asked), else an OffscreenCanvas copy. */
export async function videoFrameToBitmap(frame: FrameLike, opts: { maxSide?: number; resizeQuality?: ResizeQuality }): Promise<ImageBitmap> {
  const src = frame as unknown as VideoFrame;
  const size = grabSize(frame.displayWidth, frame.displayHeight, opts.maxSide);
  try {
    return size
      ? await createImageBitmap(src, { resizeWidth: size.width, resizeHeight: size.height, resizeQuality: opts.resizeQuality ?? 'medium' })
      : await createImageBitmap(src);
  } catch (e) {
    if (typeof OffscreenCanvas === 'undefined') throw e;
    const w = size?.width ?? frame.displayWidth;
    const h = size?.height ?? frame.displayHeight;
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw e;
    ctx.drawImage(src, 0, 0, w, h);
    return canvas.transferToImageBitmap();
  }
}

function errMessage(e: unknown): string {
  return (e as { message?: string })?.message ?? String(e);
}
