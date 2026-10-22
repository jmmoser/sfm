/**
 * Pure helpers that turn engine outputs into the harness result (and e2e/metrics.mjs input):
 * SparseCamera → dataset frame mapping, intrinsics normalisation to the dataset image size,
 * per-stage timings from progress events. No DOM, unit-tested in result.test.ts.
 */
import type { CameraIntrinsics, Pose } from '../../core/types';
import type { IntrinsicsJSON, MappedImage, PoseJSON, ProgressRecord, StageTiming } from './types';

/** A frame the harness delivered, with what the engine said about it. */
export interface DeliveredRef {
  /** FrameInput.index sent: the delivered-frame counter (a verbatim replay of a recording: the recorded index). */
  delivered: number;
  /** Dataset frame index. */
  index: number;
  /** Capture time (ms). */
  t: number;
  /** LiveUpdate.frameIndex of its response (null when the frame failed). */
  engineFrameIndex: number | null;
  keyframe: boolean;
}

/** SparseCamera plus the optional extension fields the harness understands. */
export interface CameraLike {
  id: number;
  pose: { R: ArrayLike<number>; t: ArrayLike<number> };
  intrinsics: CameraIntrinsics;
  /** Extension (src/engine SparseCameraEx, not in protocol.ts): FrameInput.index of the source frame. */
  frameIndex?: unknown;
  /** Extension: capture time of the source frame (ms). */
  t?: unknown;
}

export function poseToJSON(p: { R: ArrayLike<number>; t: ArrayLike<number> } | Pose): PoseJSON {
  return { R: Array.from(p.R as ArrayLike<number>), t: Array.from(p.t as ArrayLike<number>) };
}

const isInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v);
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/**
 * Dataset frame index of the keyframe a 'keyframe' LiveUpdate created. Best-of-window keyframing
 * (src/sfm/keyframes.ts) can make an EARLIER, held frame the keyframe: the update of the later
 * frame then names its source in `timings.keyframeFrameIndex` (FrameInput.index of the held frame,
 * mapped back by `datasetIndexOfDelivered`). Otherwise the keyframe is the update's own frame (`current`).
 */
export function keyframeSourceIndex(timings: Record<string, number> | undefined, datasetIndexOfDelivered: (delivered: number) => number | undefined, current: number): number {
  const k = timings?.keyframeFrameIndex;
  if (isInt(k)) {
    const idx = datasetIndexOfDelivered(k);
    if (idx !== undefined) return idx;
  }
  return current;
}

/**
 * Map every camera to the dataset frame it came from. Per camera, in this order:
 *  1. `frameIndex` (extension field, src/engine SparseCameraEx): FrameInput.index of the source
 *     frame (= DeliveredRef.delivered), else LiveUpdate.frameIndex;
 *  2. `t` (extension field): the delivered frame with the nearest capture time (within 1 ms);
 *  3. `id` as the keyframe ordinal (k-th frame answered with decision 'keyframe') — used when the
 *     ids of the cameras without 1./2. are distinct integers below the number of keyframes;
 *  4. `id` as LiveUpdate.frameIndex of a keyframe.
 * Cameras mapped to a frame that is already taken are reported in `duplicates` and dropped.
 */
export function mapCamerasToFrames(
  cams: readonly CameraLike[],
  frames: readonly DeliveredRef[],
): { images: MappedImage[]; mapping: string; unmapped: number[]; duplicates: number[] } {
  const byEngine = new Map<number, DeliveredRef>();
  const byDelivered = new Map<number, DeliveredRef>();
  for (const f of frames) {
    if (f.engineFrameIndex !== null) byEngine.set(f.engineFrameIndex, f);
    byDelivered.set(f.delivered, f);
  }
  const keyframes = frames.filter((f) => f.keyframe);
  const sortedT = [...frames].sort((a, b) => a.t - b.t);
  const nearestT = (t: number): DeliveredRef | null => {
    let best: DeliveredRef | null = null;
    for (const f of sortedT) if (!best || Math.abs(f.t - t) < Math.abs(best.t - t)) best = f;
    return best && Math.abs(best.t - t) <= 1 ? best : null;
  };
  // Decide globally whether ids look like keyframe ordinals.
  const rest = cams.filter((c) => !isInt(c.frameIndex) && !isNum(c.t));
  const restIds = rest.map((c) => c.id);
  const ordinal =
    rest.length > 0 && restIds.every((id) => isInt(id) && id >= 0 && id < keyframes.length) && new Set(restIds).size === restIds.length;
  const used = new Set<number>();
  const images: MappedImage[] = [];
  const unmapped: number[] = [];
  const duplicates: number[] = [];
  const via = new Set<string>();
  for (const c of cams) {
    let ref: DeliveredRef | null = null;
    let how: MappedImage['via'] | null = null;
    if (isInt(c.frameIndex)) {
      ref = byDelivered.get(c.frameIndex) ?? byEngine.get(c.frameIndex) ?? null;
      how = 'frameIndex';
    } else if (isNum(c.t)) {
      ref = nearestT(c.t);
      how = 't';
    } else if (ordinal) {
      ref = keyframes[c.id] ?? null;
      how = 'keyframe-ordinal';
    } else if (isInt(c.id)) {
      const r = byEngine.get(c.id);
      ref = r && r.keyframe ? r : null;
      how = 'id-as-frame';
    }
    if (!ref || !how) {
      unmapped.push(c.id);
      continue;
    }
    if (used.has(ref.index)) {
      duplicates.push(c.id);
      continue;
    }
    used.add(ref.index);
    via.add(how);
    images.push({ index: ref.index, id: c.id, pose: poseToJSON(c.pose), f: c.intrinsics.f, via: how });
  }
  images.sort((a, b) => a.index - b.index);
  return { images, mapping: via.size ? [...via].join('+') : 'none', unmapped, duplicates };
}

/**
 * Express engine intrinsics in the dataset's full-resolution pixel frame (W × H):
 *  - same size: as is;
 *  - same aspect, other size: rescaled with the §2 resampling rule (f' = f·s, c' = (c + ½)·s − ½);
 *  - transposed (a quarter turn): f, k1, k2 kept, cx/cy swapped (approximate) — the poses are then in
 *    a rotated camera frame too, which the note flags (rotation errors will be ~90°; integrator must check).
 */
export function normalizeIntrinsics(intr: CameraIntrinsics, W: number, H: number): { intrinsics: IntrinsicsJSON; note: string | null } {
  const w = intr.width || W;
  const h = intr.height || H;
  const base = { k1: intr.k1 ?? 0, k2: intr.k2 ?? 0 };
  if (w === W && h === H) return { intrinsics: { f: intr.f, cx: intr.cx, cy: intr.cy, ...base, width: W, height: H }, note: null };
  const same = Math.abs(w / h - W / H) < 1e-3;
  if (same) {
    const s = W / w;
    return {
      intrinsics: { f: intr.f * s, cx: (intr.cx + 0.5) * s - 0.5, cy: (intr.cy + 0.5) * s - 0.5, ...base, width: W, height: H },
      note: `engine intrinsics at ${w}x${h}, rescaled by ${s.toFixed(4)} to ${W}x${H}`,
    };
  }
  if (Math.abs(h / w - W / H) < 1e-3) {
    const s = W / h;
    return {
      intrinsics: { f: intr.f * s, cx: (intr.cy + 0.5) * s - 0.5, cy: (intr.cx + 0.5) * s - 0.5, ...base, width: W, height: H },
      note: `engine intrinsics at ${w}x${h} are TRANSPOSED w.r.t. the dataset ${W}x${H} (engine works in another orientation); cx/cy swapped (approximate: the turn direction is unknown), poses NOT rotated`,
    };
  }
  return {
    intrinsics: { f: intr.f, cx: intr.cx, cy: intr.cy, ...base, width: w, height: h },
    note: `engine intrinsics at ${w}x${h} do not match the dataset ${W}x${H} (aspect differs); used as is`,
  };
}

/** Element-wise median of the cameras' intrinsics (a shared camera gives its own values). */
export function medianIntrinsics(list: readonly CameraIntrinsics[]): CameraIntrinsics | null {
  if (list.length === 0) return null;
  const med = (pick: (c: CameraIntrinsics) => number): number => {
    const v = list.map(pick).sort((a, b) => a - b);
    const m = v.length >> 1;
    return v.length % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
  };
  return {
    width: list[0].width,
    height: list[0].height,
    f: med((c) => c.f),
    cx: med((c) => c.cx),
    cy: med((c) => c.cy),
    k1: med((c) => c.k1 ?? 0),
    k2: med((c) => c.k2 ?? 0),
  };
}

/**
 * Per-stage timings from progress events: within a step, a stage lasts from its first event to
 * the first event of the next stage (or the end of the step, `stepEnds[step]`, ms since start).
 */
export function stageTimings(progress: readonly ProgressRecord[], stepEnds: Record<string, number>): StageTiming[] {
  const order: string[] = [];
  const map = new Map<string, StageTiming>();
  for (const p of progress) {
    const key = `${p.step}\u0000${p.stage}`;
    let s = map.get(key);
    if (!s) {
      s = { step: p.step, stage: p.stage, uiStage: p.uiStage, firstMs: p.ms, lastMs: p.ms, durationMs: 0, events: 0 };
      map.set(key, s);
      order.push(key);
    }
    s.lastMs = Math.max(s.lastMs, p.ms);
    s.firstMs = Math.min(s.firstMs, p.ms);
    s.events++;
  }
  const list = order.map((k) => map.get(k)!);
  for (const s of list) {
    const next = list.filter((o) => o.step === s.step && o.firstMs > s.firstMs).reduce((m, o) => Math.min(m, o.firstMs), Infinity);
    const end = Number.isFinite(next) ? next : (stepEnds[s.step] ?? s.lastMs);
    s.durationMs = Math.max(0, end - s.firstMs);
  }
  return list;
}

/** Bounding box of a Float32 xyz array (null when empty or all non-finite). */
export function bbox(positions: Float32Array, count: number): { min: number[]; max: number[] } | null {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  let any = false;
  for (let i = 0; i < count; i++) {
    const x = positions[3 * i];
    const y = positions[3 * i + 1];
    const z = positions[3 * i + 2];
    if (!Number.isFinite(x + y + z)) continue;
    any = true;
    if (x < min[0]) min[0] = x;
    if (y < min[1]) min[1] = y;
    if (z < min[2]) min[2] = z;
    if (x > max[0]) max[0] = x;
    if (y > max[1]) max[1] = y;
    if (z > max[2]) max[2] = z;
  }
  return any ? { min, max } : null;
}
