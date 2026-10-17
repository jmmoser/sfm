/**
 * Pure decisions of the capture screen (unit-tested in capture-policy.test.ts).
 */
import { ENGINE_ERRORS } from '../workers/protocol';
import type { EngineState } from './engine-client';

/**
 * What the capture screen does when `processFrame` rejects:
 *  - 'ignore'      back-pressure (EngineBusyError) or a cancelled frame;
 *  - 'unavailable' the engine cannot process frames at all (placeholder worker, not
 *                  initialised) or stopped for good after repeated crashes (state 'dead'):
 *                  stop the pump and show an error card — retrying at 10 Hz only spams toasts;
 *  - 'storage'     the keyframe could not be stored (quota, or any other failed write): pause
 *                  and offer to finish with what was captured — further keyframes would likely
 *                  fail the same way, and carrying on would silently lose them;
 *  - 'restarted'   the worker crashed and is being respawned (the client replays init): warn
 *                  once and keep going;
 *  - 'transient'   anything else: count it, warn after a few in a row.
 */
export type FrameErrorAction = 'ignore' | 'unavailable' | 'storage' | 'restarted' | 'transient';

/** QuotaExceededError (DOM), StorageQuotaError (src/io) or an engine error with code 'quota'. */
export function isQuotaError(e: unknown): boolean {
  const o = e as { name?: string; code?: unknown } | null;
  return !!o && (o.name === 'QuotaExceededError' || o.name === 'StorageQuotaError' || o.code === 'quota');
}

/** A session-store write that failed for another reason than quota (StorageWriteError / code 'storage'). */
export function isStorageWriteError(e: unknown): boolean {
  const o = e as { name?: string; code?: unknown } | null;
  return !!o && (o.name === ENGINE_ERRORS.storageWrite || o.code === 'storage');
}

export function frameErrorAction(e: unknown, engineState: EngineState): FrameErrorAction {
  const name = (e as { name?: string } | null)?.name;
  if (name === ENGINE_ERRORS.busy || name === ENGINE_ERRORS.cancelled || name === 'AbortError') return 'ignore';
  if (isQuotaError(e) || isStorageWriteError(e)) return 'storage';
  if (engineState === 'dead' || engineState === 'disposed') return 'unavailable';
  if (name === ENGINE_ERRORS.notImplemented || name === ENGINE_ERRORS.notInitialized) return 'unavailable';
  if (name === ENGINE_ERRORS.crashed) return 'restarted';
  return 'transient';
}

/**
 * Whether the keypoint overlay of a frame captured at UI rotation `frameAngle` can be drawn on
 * the live preview shown at `currentAngle`. Both the frame bitmap and the `<video>` are upright
 * for THEIR UI rotation, so after a rotation the old frame's keypoints are not a quarter turn
 * of the new view (the UI rotation compensates the device roll) and the frame aspect no longer
 * matches the video: such a stale overlay is dropped (the next frame arrives ≤ 0.6 s later).
 */
export function overlayMatchesView(frameAngle: number, currentAngle: number): boolean {
  return ((((frameAngle - currentAngle) % 360) + 360) % 360) === 0;
}
