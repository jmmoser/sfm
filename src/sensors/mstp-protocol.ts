/**
 * Messages between MstpFrameSource (main thread, ./mstp-camera.ts) and the camera worker
 * (src/workers/camera.worker.ts → ./mstp-worker-core.ts).
 *
 * `gen` identifies one attached track: the main thread bumps it on every attach (camera restart,
 * re-attach after an error), so events and grab results of an older track are ignored.
 */

export type CameraWorkerRequest =
  /**
   * Read `track` (a clone, transferred) with MediaStreamTrackProcessor, or — where tracks cannot be
   * transferred (Chromium) — the already-created processor's `readable`. Replaces any previous one.
   */
  | { type: 'attach'; gen: number; track?: MediaStreamTrack; readable?: ReadableStream<unknown>; maxBufferSize: number }
  /** Convert frame `seq` of track `gen` (if it is still the current one) to an ImageBitmap. */
  | { type: 'grab'; id: number; gen: number; seq: number; maxSide?: number; resizeQuality?: ResizeQuality }
  /** Stop reading and stop the worker's track (if it holds one). */
  | { type: 'detach' };

export type CameraWorkerEvent =
  /** The processor is reading; `timeOrigin` = the worker's performance.timeOrigin, read now. */
  | { type: 'attached'; gen: number; timeOrigin: number }
  | {
      type: 'frame';
      gen: number;
      seq: number;
      /** VideoFrame.timestamp (µs, capture clock — not the performance.now() timeline). */
      timestampUs: number;
      /** Worker performance.now() when the frame was read. */
      receivedMs: number;
      /** displayWidth × displayHeight. */
      width: number;
      height: number;
      codedWidth: number;
      codedHeight: number;
      format: string | null;
    }
  /** The readable stream finished (track ended or was stopped). */
  | { type: 'ended'; gen: number }
  /** `attach` failed (`unsupported`: no MediaStreamTrackProcessor in this worker) or reading failed. */
  | { type: 'error'; gen: number; message: string; unsupported?: boolean }
  | { type: 'grabbed'; id: number; bitmap: ImageBitmap }
  | { type: 'grabbed'; id: number; error: string };
