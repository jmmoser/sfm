/**
 * Feature database: keyframe metadata, local features, optional per-feature colours
 * and verified two-view geometries (docs/ARCHITECTURE.md §4.5).
 *
 * Two-view geometries are stored canonically (smaller id first). `getTwoView(a, b)`
 * returns the geometry ORIENTED from a to b: inliers as (a-feature, b-feature) pairs
 * and, for a > b, the inverted relative pose (x_a = Rᵀ x_b − Rᵀ t), Eᵀ, Fᵀ and H⁻¹.
 */
import type { FeatureSet, FrameInfo, Mat3, TwoViewGeometry } from '../core/types';
import { mat3Inv, mat3Transpose } from '../math/mat3';

/** Ids must be integers in [0, 2^20). */
export const MAX_IMAGE_ID = 1 << 20;
const PAIR_SHIFT = MAX_IMAGE_ID;

/** True for a valid image id (integer in [0, MAX_IMAGE_ID)); larger ids would make `pairKey` collide. */
export function isValidImageId(id: number): boolean {
  return Number.isInteger(id) && id >= 0 && id < MAX_IMAGE_ID;
}

/**
 * Order-independent integer key of an image pair: min·2^20 + max (exact in a double,
 * ids < 2^20). pairKey(a, b) === pairKey(b, a).
 */
export function pairKey(a: number, b: number): number {
  return a < b ? a * PAIR_SHIFT + b : b * PAIR_SHIFT + a;
}

/** Inverse of `pairKey`: [min id, max id]. */
export function pairIds(key: number): [number, number] {
  const a = Math.floor(key / PAIR_SHIFT);
  return [a, key - a * PAIR_SHIFT];
}

/** Swap (i, j) → (j, i) in a flat match array. */
export function swapMatches(m: Uint32Array): Uint32Array {
  const out = new Uint32Array(m.length);
  for (let k = 0; k < m.length; k += 2) {
    out[k] = m[k + 1];
    out[k + 1] = m[k];
  }
  return out;
}

/** Geometry of the pair seen from the other image (b → a). */
export function invertTwoView(g: TwoViewGeometry): TwoViewGeometry {
  const out: TwoViewGeometry = { config: g.config, inliers: swapMatches(g.inliers) };
  if (g.E) out.E = mat3Transpose(g.E);
  if (g.F) out.F = mat3Transpose(g.F);
  if (g.H) {
    const Hi = mat3Inv(g.H);
    if (Hi) out.H = Hi;
  }
  if (g.R && g.t) {
    // x2 = R x1 + t  ⇒  x1 = Rᵀ x2 − Rᵀ t   (|t| unchanged)
    const R = g.R,
      t = g.t;
    out.R = mat3Transpose(R) as Mat3;
    out.t = Float64Array.of(
      -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
      -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
      -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
    );
  }
  if (g.triAngle !== undefined) out.triAngle = g.triAngle;
  return out;
}

/** In-memory store of frames, features and verified pairs. */
export class FeatureDatabase {
  readonly frames = new Map<number, FrameInfo>();
  readonly features = new Map<number, FeatureSet>();
  /** Optional per-feature sRGB colours (3 bytes per feature), keyed by frame id. */
  readonly colors = new Map<number, Uint8Array>();
  private readonly twoViews = new Map<number, TwoViewGeometry>();
  private readonly invCache = new Map<number, TwoViewGeometry>();

  /** Adds (or replaces) a frame with its features and optional per-feature colours. */
  addFrame(info: FrameInfo, features: FeatureSet, colors?: Uint8Array): void {
    const id = info.id;
    if (!isValidImageId(id)) throw new RangeError(`FeatureDatabase: bad frame id ${id}`);
    if (colors && colors.length < 3 * features.count)
      throw new RangeError('FeatureDatabase: colors must have 3 bytes per feature');
    this.frames.set(id, info);
    this.features.set(id, features);
    if (colors) this.colors.set(id, colors);
    else this.colors.delete(id);
  }

  /** Removes a frame and every pair it takes part in. */
  removeFrame(id: number): void {
    this.frames.delete(id);
    this.features.delete(id);
    this.colors.delete(id);
    for (const key of [...this.twoViews.keys()]) {
      const [a, b] = pairIds(key);
      if (a === id || b === id) {
        this.twoViews.delete(key);
        this.invCache.delete(key);
      }
    }
  }

  /** Stores the verified geometry of (a, b) — given oriented a → b — canonically (min id first). */
  setTwoView(a: number, b: number, geom: TwoViewGeometry): void {
    if (a === b) throw new RangeError('FeatureDatabase.setTwoView: a === b');
    if (!isValidImageId(a) || !isValidImageId(b)) throw new RangeError(`FeatureDatabase.setTwoView: bad image id ${isValidImageId(a) ? b : a}`);
    const key = pairKey(a, b);
    this.twoViews.set(key, a < b ? geom : invertTwoView(geom));
    this.invCache.delete(key);
  }

  /** Geometry oriented a → b (inliers as (a-feature, b-feature)); undefined if the pair is unknown. */
  getTwoView(a: number, b: number): TwoViewGeometry | undefined {
    const key = pairKey(a, b);
    const g = this.twoViews.get(key);
    if (!g || a < b) return g;
    let inv = this.invCache.get(key);
    if (!inv) {
      inv = invertTwoView(g);
      this.invCache.set(key, inv);
    }
    return inv;
  }

  hasPair(a: number, b: number): boolean {
    return this.twoViews.has(pairKey(a, b));
  }

  /** Removes the pair (a, b) if present. */
  deletePair(a: number, b: number): void {
    const key = pairKey(a, b);
    this.twoViews.delete(key);
    this.invCache.delete(key);
  }

  get numPairs(): number {
    return this.twoViews.size;
  }

  /** All stored pairs as [min id, max id, geometry oriented min → max]. */
  *pairs(): IterableIterator<[number, number, TwoViewGeometry]> {
    for (const [key, g] of this.twoViews) {
      const [a, b] = pairIds(key);
      yield [a, b, g];
    }
  }

  /** Frame ids in ascending order. */
  frameIds(): number[] {
    return [...this.frames.keys()].sort((x, y) => x - y);
  }
}
