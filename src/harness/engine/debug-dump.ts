/**
 * Diagnostics of `engine: 'inline'` runs (`HarnessConfig.debugDump`): reads PhotogrammetryEngine
 * internals — the live session's keyframe records, verified pairs and live reconstruction —
 * through untyped access. Debugging aid only (NOT an engine API: field names may change); every
 * reader returns null when the internals are not reachable (worker / mock engines).
 */
import type { PoseJSON } from './types';

/** State of one keyframe in the live model. */
export interface DebugKeyframe {
  id: number;
  frameIndex: number;
  registered: boolean;
  /** Observations of 3D points in this image (0 when unregistered). */
  numPoints3D: number;
  /** Features whose correspondences reach a 3D point (what PnP registration can use; may be stale). */
  numVisiblePoints3D: number;
  pose?: PoseJSON;
}

/** One verified pair (keyframe ids a < b; geometry oriented a → b, x_b = R x_a + t). */
export interface DebugPair {
  a: number;
  b: number;
  frameA: number;
  frameB: number;
  config: string;
  inliers: number;
  triAngleDeg: number | null;
  R: number[] | null;
  t: number[] | null;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
function liveOf(harnessEngine: unknown): any | null {
  const inner = (harnessEngine as { engine?: unknown } | null)?.engine as { live?: unknown } | undefined;
  const live = inner?.live as any;
  return live && typeof live === 'object' && live.mapper && live.state ? live : null;
}

const arr = (m: ArrayLike<number> | undefined | null): number[] | null => (m ? Array.from(m, (v) => Math.round(v * 1e9) / 1e9) : null);

/** Every keyframe (capture order) with its registration state and pose in the live model. */
export function debugKeyframes(harnessEngine: unknown): DebugKeyframe[] | null {
  const live = liveOf(harnessEngine);
  if (!live) return null;
  const st = live.state;
  const rec = live.mapper.reconstruction;
  const out: DebugKeyframe[] = [];
  for (const id of st.order as number[]) {
    const img = rec.images.get(id);
    const k: DebugKeyframe = {
      id,
      frameIndex: st.records.get(id)?.frameIndex ?? -1,
      registered: !!img?.registered,
      numPoints3D: img?.registered ? img.numPoints3D : 0,
      numVisiblePoints3D: img?.numVisiblePoints3D ?? 0,
    };
    if (img?.registered) k.pose = { R: arr(img.pose.R)!, t: arr(img.pose.t)! };
    out.push(k);
  }
  return out;
}

/** Every verified pair of the feature database. */
export function debugPairs(harnessEngine: unknown): DebugPair[] | null {
  const live = liveOf(harnessEngine);
  if (!live) return null;
  const st = live.state;
  const out: DebugPair[] = [];
  for (const [a, b, g] of st.db.pairs() as Iterable<[number, number, any]>) {
    out.push({
      a,
      b,
      frameA: st.records.get(a)?.frameIndex ?? -1,
      frameB: st.records.get(b)?.frameIndex ?? -1,
      config: String(g.config),
      inliers: g.inliers ? g.inliers.length >> 1 : 0,
      triAngleDeg: typeof g.triAngle === 'number' ? (g.triAngle * 180) / Math.PI : null,
      R: arr(g.R),
      t: arr(g.t),
    });
  }
  return out;
}
/* eslint-enable @typescript-eslint/no-explicit-any */
