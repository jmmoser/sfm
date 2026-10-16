/**
 * Image-pair selection for matching (ARCHITECTURE §5 "Refine"): sequential neighbours,
 * vocabulary retrieval (loop closure) and spatial neighbours from a reconstruction, plus a
 * deduplicating merge. Pairs are normalised to [min id, max id].
 */
import type { Pose } from '../core/types';

/** Unordered image pair, normalised so that pair[0] < pair[1]. */
export type ImagePair = [number, number];

/** Retrieval interface implemented by `VocabularyIndex` (decoupled for testing). */
export interface RetrievalSource {
  imageIds(): number[];
  query(id: number, k: number, exclude?: Iterable<number> | ((id: number) => boolean)): { id: number; score: number }[];
}

const KEY_BASE = 67108864; // 2^26

/**
 * Order-independent numeric key of a pair of non-negative integer ids < 2^26 (exact in a
 * double). Not necessarily equal to `database.pairKey`; use it only within this module's
 * outputs or for your own sets.
 */
export function pairId(a: number, b: number): number {
  return a < b ? a * KEY_BASE + b : b * KEY_BASE + a;
}

function norm(a: number, b: number): ImagePair {
  return a < b ? [a, b] : [b, a];
}

/**
 * Pairs of images that are within `window` positions of each other in `ids` (capture
 * order). With `exponential`, also pairs at offsets window·2^k (k ≥ 1) up to `maxOffset`
 * (COLMAP-style "quadratic overlap" for cheap long-range links in long sequences).
 */
export function sequentialPairs(
  ids: ArrayLike<number>,
  window: number,
  opts: { exponential?: boolean; maxOffset?: number } = {},
): ImagePair[] {
  const n = ids.length;
  const w = Math.max(0, Math.floor(window));
  const out: ImagePair[] = [];
  for (let i = 0; i < n; i++) {
    for (let d = 1; d <= w && i + d < n; d++) if (ids[i] !== ids[i + d]) out.push(norm(ids[i], ids[i + d]));
    if (opts.exponential && w > 0) {
      const maxOff = opts.maxOffset ?? n;
      for (let d = 2 * w; d <= maxOff && i + d < n; d *= 2) if (ids[i] !== ids[i + d]) out.push(norm(ids[i], ids[i + d]));
    }
  }
  return dedupe(out);
}

/**
 * Top-k retrieval candidates for every image of `index` (or of `opts.ids`). Pairs for which
 * `exclude(a, b)` is true (e.g. already matched or temporal neighbours) are filtered before
 * taking the top k, results below `minScore` are dropped.
 */
export function retrievalPairs(
  index: RetrievalSource,
  k: number,
  opts: { ids?: ArrayLike<number>; minScore?: number; exclude?: (a: number, b: number) => boolean } = {},
): ImagePair[] {
  const ids = opts.ids ?? index.imageIds();
  const minScore = opts.minScore ?? -Infinity;
  const exclude = opts.exclude;
  const out: ImagePair[] = [];
  for (let i = 0; i < ids.length; i++) {
    const q = ids[i];
    const res = index.query(q, k, exclude ? (id: number) => id === q || exclude(q, id) : undefined);
    for (const r of res) if (r.id !== q && r.score >= minScore) out.push(norm(q, r.id));
  }
  return dedupe(out);
}

/** Camera for `spatialPairs`. `depth` (optional) = typical scene depth seen by the camera. */
export interface SpatialCamera {
  id: number;
  pose: Pose;
  depth?: number;
}

/**
 * For every camera, the k nearest other cameras whose viewing directions differ by at most
 * `maxAngleDeg`. Distance = between camera centres, or — when both cameras carry a `depth`
 * — between their look-at points C + depth·d (better for inward/outward-facing captures).
 * `maxDistance` optionally bounds the distance. O(n²) brute force (n = 1000: ~5 ms).
 */
export function spatialPairs(
  cams: readonly SpatialCamera[],
  k: number,
  maxAngleDeg: number,
  opts: { maxDistance?: number } = {},
): ImagePair[] {
  const n = cams.length;
  const kk = Math.max(0, Math.floor(k));
  if (n < 2 || kk === 0) return [];
  const C = new Float64Array(3 * n);
  const D = new Float64Array(3 * n);
  const P = new Float64Array(3 * n);
  const hasDepth = new Uint8Array(n);
  const valid = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const { R, t } = cams[i].pose;
    // C = −Rᵀ t, viewing direction = Rᵀ e_z = third row of R.
    const cx = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]);
    const cy = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]);
    const cz = -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);
    C[3 * i] = cx;
    C[3 * i + 1] = cy;
    C[3 * i + 2] = cz;
    const dn = Math.hypot(R[6], R[7], R[8]) || 1;
    D[3 * i] = R[6] / dn;
    D[3 * i + 1] = R[7] / dn;
    D[3 * i + 2] = R[8] / dn;
    // cameras with a non-finite pose (e.g. not registered) are never paired
    valid[i] = Number.isFinite(cx + cy + cz + D[3 * i] + D[3 * i + 1] + D[3 * i + 2]) ? 1 : 0;
    const dep = cams[i].depth;
    if (dep !== undefined && Number.isFinite(dep) && dep > 0) {
      hasDepth[i] = 1;
      P[3 * i] = cx + dep * D[3 * i];
      P[3 * i + 1] = cy + dep * D[3 * i + 1];
      P[3 * i + 2] = cz + dep * D[3 * i + 2];
    }
  }
  const cosMax = maxAngleDeg >= 180 ? -Infinity : Math.cos((Math.min(Math.max(maxAngleDeg, 0), 180) * Math.PI) / 180);
  const maxD2 = opts.maxDistance !== undefined ? opts.maxDistance * opts.maxDistance : Infinity;
  const bestD = new Float64Array(kk);
  const bestJ = new Int32Array(kk);
  const out: ImagePair[] = [];
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue;
    let cnt = 0;
    const dx = D[3 * i],
      dy = D[3 * i + 1],
      dz = D[3 * i + 2];
    for (let j = 0; j < n; j++) {
      if (j === i || !valid[j]) continue;
      if (dx * D[3 * j] + dy * D[3 * j + 1] + dz * D[3 * j + 2] < cosMax) continue;
      const A = hasDepth[i] && hasDepth[j] ? P : C;
      const ex = A[3 * i] - A[3 * j],
        ey = A[3 * i + 1] - A[3 * j + 1],
        ez = A[3 * i + 2] - A[3 * j + 2];
      const d2 = ex * ex + ey * ey + ez * ez;
      if (d2 > maxD2) continue;
      if (cnt < kk) {
        let p = cnt++;
        while (p > 0 && bestD[p - 1] > d2) {
          bestD[p] = bestD[p - 1];
          bestJ[p] = bestJ[p - 1];
          p--;
        }
        bestD[p] = d2;
        bestJ[p] = j;
      } else if (d2 < bestD[kk - 1]) {
        let p = kk - 1;
        while (p > 0 && bestD[p - 1] > d2) {
          bestD[p] = bestD[p - 1];
          bestJ[p] = bestJ[p - 1];
          p--;
        }
        bestD[p] = d2;
        bestJ[p] = j;
      }
    }
    for (let q = 0; q < cnt; q++) if (cams[i].id !== cams[bestJ[q]].id) out.push(norm(cams[i].id, cams[bestJ[q]].id));
  }
  return dedupe(out);
}

/**
 * Union of pair lists: normalised to [min, max], self-pairs and duplicates removed, pairs
 * with `exclude(a, b)` true dropped (e.g. `(a, b) => db.hasPair(a, b)`; called once per
 * distinct pair), sorted by (a, b). Pairs with a non-finite id are dropped.
 * Ids that are integers in [0, 2^26) take a fast path (typed-array sort of numeric keys,
 * ≈ 10× faster than sorting pair objects); other ids fall back to a generic merge.
 */
export function mergePairs(lists: Iterable<readonly (readonly [number, number])[]>, exclude?: (a: number, b: number) => boolean): ImagePair[] {
  let total = 0;
  let small = true;
  const arr: (readonly (readonly [number, number])[])[] = [];
  for (const list of lists) {
    arr.push(list);
    total += list.length;
    if (small)
      for (const p of list) {
        const a = p[0],
          b = p[1];
        if (!(a >= 0 && a < KEY_BASE && b >= 0 && b < KEY_BASE && Number.isInteger(a) && Number.isInteger(b))) {
          if (Number.isFinite(a) && Number.isFinite(b)) small = false;
        }
      }
  }
  if (!small) return mergePairsGeneric(arr, exclude);
  const keys = new Float64Array(total);
  let n = 0;
  for (const list of arr)
    for (const p of list) {
      const a = p[0],
        b = p[1];
      if (a === b || !Number.isFinite(a) || !Number.isFinite(b)) continue;
      keys[n++] = a < b ? a * KEY_BASE + b : b * KEY_BASE + a;
    }
  const ks = keys.subarray(0, n).sort();
  const out: ImagePair[] = [];
  let prev = -1;
  for (let i = 0; i < n; i++) {
    const k = ks[i];
    if (k === prev) continue;
    prev = k;
    const b = k % KEY_BASE,
      a = (k - b) / KEY_BASE;
    if (exclude && exclude(a, b)) continue;
    out.push([a, b]);
  }
  return out;
}

function mergePairsGeneric(lists: readonly (readonly (readonly [number, number])[])[], exclude?: (a: number, b: number) => boolean): ImagePair[] {
  const seen = new Map<number, Set<number>>();
  const out: ImagePair[] = [];
  for (const list of lists) {
    for (const p of list) {
      const a = Math.min(p[0], p[1]),
        b = Math.max(p[0], p[1]);
      if (a === b || !Number.isFinite(a) || !Number.isFinite(b)) continue;
      let s = seen.get(a);
      if (!s) seen.set(a, (s = new Set<number>()));
      if (s.has(b)) continue;
      s.add(b);
      if (exclude && exclude(a, b)) continue;
      out.push([a, b]);
    }
  }
  out.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  return out;
}

function dedupe(pairs: ImagePair[]): ImagePair[] {
  return mergePairs([pairs]);
}
