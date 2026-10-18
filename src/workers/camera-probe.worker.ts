/**
 * Camera timing probe worker (src/sensors/camera-probe.ts): reads a transferred track (or a
 * main-thread processor's stream) with MediaStreamTrackProcessor, records each frame's timestamp,
 * arrival time and size, and closes it at once. `stop` returns the samples and stops the track.
 */
import type { ProbeMstpSample, ProbeWorkerEvent, ProbeWorkerRequest } from '../sensors/camera-probe';

interface FrameLike {
  readonly timestamp: number;
  readonly codedWidth: number;
  readonly codedHeight: number;
  readonly displayWidth: number;
  readonly displayHeight: number;
  readonly format: string | null;
  close(): void;
}
type ProcessorCtor = new (init: { track: MediaStreamTrack }) => { readable: ReadableStream<FrameLike> };
interface WorkerScope {
  postMessage(msg: ProbeWorkerEvent): void;
  onmessage: ((ev: MessageEvent<ProbeWorkerRequest>) => void) | null;
}

const scope = self as unknown as WorkerScope;
const samples: Omit<ProbeMstpSample, 'receivedMs'>[] = [];
const received: number[] = [];
let reader: ReadableStreamDefaultReader<FrameLike> | null = null;
let track: MediaStreamTrack | null = null;

async function read(r: ReadableStreamDefaultReader<FrameLike>): Promise<void> {
  try {
    for (;;) {
      const { value, done } = await r.read();
      if (done) return;
      received.push(performance.now());
      samples.push({
        timestampUs: value.timestamp,
        codedWidth: value.codedWidth,
        codedHeight: value.codedHeight,
        displayWidth: value.displayWidth,
        displayHeight: value.displayHeight,
        format: value.format ?? null,
      });
      value.close();
    }
  } catch (e) {
    scope.postMessage({ type: 'error', message: `reading frames failed: ${(e as Error)?.message ?? e}` });
  }
}

scope.onmessage = (e: MessageEvent<ProbeWorkerRequest>) => {
  const msg = e.data;
  if (msg.type === 'start') {
    try {
      if (msg.readable) reader = (msg.readable as ReadableStream<FrameLike>).getReader();
      else {
        const Processor = (globalThis as { MediaStreamTrackProcessor?: ProcessorCtor }).MediaStreamTrackProcessor;
        track = msg.track ?? null;
        if (typeof Processor !== 'function' || !track) {
          track?.stop();
          scope.postMessage({ type: 'error', message: 'MediaStreamTrackProcessor is not available in workers' });
          return;
        }
        reader = new Processor({ track }).readable.getReader();
      }
    } catch (err) {
      track?.stop();
      scope.postMessage({ type: 'error', message: `MediaStreamTrackProcessor failed: ${(err as Error)?.message ?? err}` });
      return;
    }
    scope.postMessage({ type: 'started', timeOrigin: performance.timeOrigin });
    void read(reader);
  } else {
    reader?.cancel().catch(() => undefined);
    track?.stop();
    scope.postMessage({ type: 'samples', samples, receivedMs: received });
  }
};
