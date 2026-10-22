/**
 * FrameTap: an observer of the FrameInput stream an EngineLike sends to its engine
 * (`EngineLike.setFrameTap`; used by the raw capture recorder, ./recorder.ts). It sees exactly
 * the frames that reach the engine, in order — not the ones the engine client rejects as busy,
 * nor the ones the FramePump skipped.
 *
 *  - `before(frame)` runs synchronously before the frame is sent, while the caller still owns the
 *    pixels (they are transferred afterwards): the tap copies what it needs there. A returned
 *    promise holds back the RESOLUTION of processFrame (not the engine), so the FramePump sends
 *    the next frame only after it — back-pressure for a tap that falls behind.
 *  - `after(frame, result, error, t)` gets the engine's answer (or error) once that wait ended:
 *    `t.ms` = processFrame time of the engine, `t.waitMs` = extra time spent waiting for the tap.
 * A tap that throws never fails the frame (logged).
 */
import type { FrameInput, LiveUpdate } from '../workers/protocol';

export interface FrameTap {
  before(frame: FrameInput): Promise<void> | void;
  after?(frame: FrameInput, result: LiveUpdate | null, error: unknown, t: { ms: number; waitMs: number }): void;
}

/** `send(frame)` with `tap` around it (see the header); without a tap just `send(frame)`. */
export async function tapFrame(tap: FrameTap | null, frame: FrameInput, send: (f: FrameInput) => Promise<LiveUpdate>): Promise<LiveUpdate> {
  if (!tap) return send(frame);
  let hold: Promise<void> | void = undefined;
  try {
    hold = tap.before(frame);
  } catch (e) {
    console.warn('[engine] frame tap failed', e);
  }
  const t0 = performance.now();
  let result: LiveUpdate | null = null;
  let error: unknown = null;
  try {
    result = await send(frame);
    return result;
  } catch (e) {
    error = e;
    throw e;
  } finally {
    const t1 = performance.now();
    if (hold) await hold.catch((e: unknown) => console.warn('[engine] frame tap failed', e));
    try {
      tap.after?.(frame, result, error, { ms: t1 - t0, waitMs: performance.now() - t1 });
    } catch (e) {
      console.warn('[engine] frame tap failed', e);
    }
  }
}
