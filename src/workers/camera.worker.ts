/**
 * Camera worker (module worker): MediaStreamTrackProcessor capture path. All logic lives in
 * src/sensors/mstp-worker-core.ts; the main-thread side is src/sensors/mstp-camera.ts.
 */
import { CameraWorkerCore, videoFrameToBitmap, type FrameLike } from '../sensors/mstp-worker-core';
import type { CameraWorkerEvent, CameraWorkerRequest } from '../sensors/mstp-protocol';

type ProcessorCtor = new (init: { track: MediaStreamTrack; maxBufferSize?: number }) => { readable: ReadableStream<FrameLike> };

interface WorkerScope {
  postMessage(msg: CameraWorkerEvent, transfer: Transferable[]): void;
  onmessage: ((ev: MessageEvent<CameraWorkerRequest>) => void) | null;
}

const scope = self as unknown as WorkerScope;

const core = new CameraWorkerCore({
  post: (msg: CameraWorkerEvent, transfer?: Transferable[]) => scope.postMessage(msg, transfer ?? []),
  createReader: (track, maxBufferSize) => {
    const Processor = (globalThis as { MediaStreamTrackProcessor?: ProcessorCtor }).MediaStreamTrackProcessor;
    if (typeof Processor !== 'function') return null;
    return new Processor({ track, maxBufferSize }).readable.getReader();
  },
  toBitmap: videoFrameToBitmap,
  now: () => performance.now(),
  timeOrigin: () => performance.timeOrigin,
});

scope.onmessage = (e: MessageEvent<CameraWorkerRequest>) => core.handle(e.data);
