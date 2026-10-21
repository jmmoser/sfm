/**
 * Pure decisions of the capture screen (unit-tested in capture-policy.test.ts): what a failed
 * frame means, whether a frame's overlay matches the preview, and the guidance banner's
 * priority + dwell gate.
 */
import { ENGINE_ERRORS, type Guidance } from '../workers/protocol';
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

/**
 * Guidance codes in the order they take the banner (highest first). 'too_dark' outranks
 * 'tracking_lost': in the dark, tracking is lost BECAUSE of the light, and "go back to a part
 * you already captured" cannot fix that — the torch can (the status pill still says Lost).
 */
export const GUIDANCE_PRIORITY: readonly Guidance[] = ['too_dark', 'tracking_lost', 'move_slower', 'more_texture', 'translate_more', 'initializing', 'good'];

/** Codes shown on their first frame: 'tracking_lost' must not wait, 'too_dark' is already debounced by the engine. */
const IMMEDIATE: ReadonlySet<Guidance> = new Set<Guidance>(['tracking_lost', 'too_dark']);

function guidanceRank(g: Guidance): number {
  const i = GUIDANCE_PRIORITY.indexOf(g);
  return i < 0 ? GUIDANCE_PRIORITY.length : i;
}

/** Severity class for screen-reader announcements: 0 starting, 1 good, 2 warning, 3 bad. */
function guidanceSeverity(g: Guidance): number {
  return g === 'initializing' ? 0 : g === 'good' ? 1 : g === 'tracking_lost' || g === 'too_dark' ? 3 : 2;
}

export interface GuidanceGateOptions {
  /** Consecutive frames a new code needs before it shows (default 2; IMMEDIATE codes: 1). */
  confirmFrames?: number;
  /** A shown code stays at least this long (ms, default 1500) unless a higher-priority code replaces it. */
  dwellMs?: number;
}

export interface GuidanceGateResult {
  /** The code to show (null: none — after a reset without one). */
  code: Guidance | null;
  /** It differs from what was shown before this update. */
  changed: boolean;
  /** Announce it to screen readers: the first code, or a change of severity (start / good / warning / bad). */
  announce: boolean;
}

/**
 * The guidance banner's hysteresis. The engine reports a guidance code per analysed frame
 * (≤ 10 Hz) and neighbouring frames often disagree (one blurred frame says 'move_slower'); a
 * banner that follows every frame flickers and floods VoiceOver. The gate shows a new code only
 * after it persisted `confirmFrames` frames, and keeps a shown code for `dwellMs` unless a
 * higher-priority one (GUIDANCE_PRIORITY) replaces it. 'tracking_lost' and 'too_dark' show at
 * once. Feed it one code per analysed frame (not the engine's extra model updates).
 */
export class GuidanceGate {
  private readonly confirmFrames: number;
  private readonly dwellMs: number;
  private shown: Guidance | null = null;
  private shownAt = -Infinity;
  private candidate: Guidance | null = null;
  private candidateFrames = 0;

  constructor(opts: GuidanceGateOptions = {}) {
    this.confirmFrames = Math.max(1, opts.confirmFrames ?? 2);
    this.dwellMs = Math.max(0, opts.dwellMs ?? 1500);
  }

  /** The code on the banner (null: none). */
  get code(): Guidance | null {
    return this.shown;
  }

  /** Show `code` now, without hysteresis (null: nothing; the next frame's code then shows at once). */
  reset(code: Guidance | null = null, now = -Infinity): void {
    this.shown = code;
    this.shownAt = now;
    this.candidate = null;
    this.candidateFrames = 0;
  }

  /** One analysed frame's guidance at time `now` (ms). */
  update(code: Guidance, now: number): GuidanceGateResult {
    const prev = this.shown;
    if (code === prev) {
      this.candidate = null;
      this.candidateFrames = 0;
      return { code, changed: false, announce: false };
    }
    if (code === this.candidate) this.candidateFrames++;
    else {
      this.candidate = code;
      this.candidateFrames = 1;
    }
    const unchanged: GuidanceGateResult = { code: prev, changed: false, announce: false };
    if (prev !== null) {
      if (!IMMEDIATE.has(code) && this.candidateFrames < this.confirmFrames) return unchanged;
      if (guidanceRank(code) >= guidanceRank(prev) && now - this.shownAt < this.dwellMs) return unchanged;
    }
    this.reset(code, now);
    return { code, changed: true, announce: prev === null || guidanceSeverity(code) !== guidanceSeverity(prev) };
  }
}
