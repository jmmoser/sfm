/**
 * Feature-level correspondence graph (COLMAP `CorrespondenceGraph`, Schönberger &
 * Frahm 2016) that grows incrementally: images and verified pairs can be added at any
 * time (live capture) without rebuilding.
 *
 * Storage: one singly linked list per (image, feature) in a shared pool of typed
 * arrays (12 bytes per directed correspondence) — no per-correspondence objects.
 * Hot loops iterate without allocation:
 *
 * ```ts
 * const heads = graph.heads(img)!, nx = graph.nextOf, im = graph.imageOf, ft = graph.featureOf;
 * for (let e = heads[f]; e >= 0; e = nx[e]) { const j = im[e], g = ft[e]; … }
 * ```
 * (re-read `nextOf` / `imageOf` / `featureOf` after `addTwoView`: the pool may grow).
 */
import type { MatchPairs } from '../core/types';
import { isValidImageId, pairIds, pairKey } from './database';

export interface Correspondence {
  image: number;
  feature: number;
}

interface ImageNode {
  numFeatures: number;
  /** Pool index of the first correspondence of each feature, -1 if none. */
  head: Int32Array;
  /** Number of correspondences per feature. */
  count: Uint16Array;
  numCorrespondences: number;
  /** Features with at least one correspondence. */
  numObservations: number;
  /** Connected images → number of correspondences. */
  adj: Map<number, number>;
}

interface PairEntry {
  /** Smaller image id. */
  a: number;
  b: number;
  /** Matches oriented a → b. */
  matches: Uint32Array;
}

const EMPTY: ReadonlyArray<Correspondence> = Object.freeze([]);
const EMPTY_U32 = new Uint32Array(0);
const EMPTY_MAP: ReadonlyMap<number, number> = new Map();

export class CorrespondenceGraph {
  private readonly nodes = new Map<number, ImageNode>();
  private readonly pairMap = new Map<number, PairEntry>();
  private next = new Int32Array(1024);
  private img = new Int32Array(1024);
  private feat = new Int32Array(1024);
  private poolSize = 0;
  /** Incremented on every structural change (lets caches detect growth). */
  version = 0;

  /** Adds an image with `numFeatures` features (no-op if it already exists with the same count). */
  addImage(id: number, numFeatures: number): void {
    if (!isValidImageId(id)) throw new RangeError(`CorrespondenceGraph.addImage: bad image id ${id} (integer in [0, 2^20))`);
    if (!(Number.isInteger(numFeatures) && numFeatures >= 0))
      throw new RangeError(`CorrespondenceGraph.addImage: bad feature count ${numFeatures}`);
    const old = this.nodes.get(id);
    if (old) {
      if (old.numFeatures !== numFeatures)
        throw new RangeError(`CorrespondenceGraph.addImage: image ${id} re-added with a different feature count`);
      return;
    }
    this.nodes.set(id, {
      numFeatures,
      head: new Int32Array(numFeatures).fill(-1),
      count: new Uint16Array(numFeatures),
      numCorrespondences: 0,
      numObservations: 0,
      adj: new Map(),
    });
    this.version++;
  }

  hasImage(id: number): boolean {
    return this.nodes.has(id);
  }

  /** Image ids in ascending order. */
  imageIds(): number[] {
    return [...this.nodes.keys()].sort((x, y) => x - y);
  }

  numFeatures(id: number): number {
    return this.nodes.get(id)?.numFeatures ?? 0;
  }

  private grow(min: number): void {
    let cap = this.next.length;
    if (cap >= min) return;
    while (cap < min) cap *= 2;
    const n = new Int32Array(cap),
      i = new Int32Array(cap),
      f = new Int32Array(cap);
    n.set(this.next.subarray(0, this.poolSize));
    i.set(this.img.subarray(0, this.poolSize));
    f.set(this.feat.subarray(0, this.poolSize));
    this.next = n;
    this.img = i;
    this.feat = f;
  }

  private link(node: ImageNode, feature: number, otherImage: number, otherFeature: number): void {
    const e = this.poolSize++;
    this.img[e] = otherImage;
    this.feat[e] = otherFeature;
    this.next[e] = node.head[feature];
    node.head[feature] = e;
    if (node.count[feature]++ === 0) node.numObservations++;
    node.numCorrespondences++;
  }

  private unlink(node: ImageNode, feature: number, otherImage: number, otherFeature: number): void {
    let prev = -1;
    for (let e = node.head[feature]; e >= 0; e = this.next[e]) {
      if (this.img[e] === otherImage && this.feat[e] === otherFeature) {
        if (prev < 0) node.head[feature] = this.next[e];
        else this.next[prev] = this.next[e];
        if (--node.count[feature] === 0) node.numObservations--;
        node.numCorrespondences--;
        return;
      }
      prev = e;
    }
  }

  /**
   * Adds the verified inlier matches of the pair (a, b), oriented (a-feature,
   * b-feature). Both images must have been added. Replaces a previous entry of the
   * same pair. Out-of-range feature indices throw.
   */
  addTwoView(a: number, b: number, inliers: MatchPairs): void {
    if (a === b) throw new RangeError('CorrespondenceGraph.addTwoView: a === b');
    const na = this.nodes.get(a),
      nb = this.nodes.get(b);
    if (!na || !nb) throw new RangeError(`CorrespondenceGraph.addTwoView: unknown image ${na ? b : a}`);
    const n = inliers.length >> 1;
    for (let k = 0; k < n; k++) {
      if (inliers[2 * k] >= na.numFeatures || inliers[2 * k + 1] >= nb.numFeatures)
        throw new RangeError('CorrespondenceGraph.addTwoView: feature index out of range');
    }
    const key = pairKey(a, b);
    if (this.pairMap.has(key)) this.removeTwoView(a, b);
    if (n === 0) return;
    this.grow(this.poolSize + 2 * n);
    for (let k = 0; k < n; k++) {
      const fa = inliers[2 * k],
        fb = inliers[2 * k + 1];
      this.link(na, fa, b, fb);
      this.link(nb, fb, a, fa);
    }
    let matches: Uint32Array;
    if (a < b) matches = Uint32Array.from(inliers.subarray(0, 2 * n));
    else {
      matches = new Uint32Array(2 * n);
      for (let k = 0; k < n; k++) {
        matches[2 * k] = inliers[2 * k + 1];
        matches[2 * k + 1] = inliers[2 * k];
      }
    }
    this.pairMap.set(key, { a: Math.min(a, b), b: Math.max(a, b), matches });
    na.adj.set(b, n);
    nb.adj.set(a, n);
    this.version++;
  }

  /** Removes the pair (a, b) and its correspondences (pool entries are not reclaimed). */
  removeTwoView(a: number, b: number): void {
    const key = pairKey(a, b);
    const p = this.pairMap.get(key);
    if (!p) return;
    const na = this.nodes.get(p.a)!,
      nb = this.nodes.get(p.b)!;
    const m = p.matches;
    for (let k = 0; k < m.length; k += 2) {
      this.unlink(na, m[k], p.b, m[k + 1]);
      this.unlink(nb, m[k + 1], p.a, m[k]);
    }
    this.pairMap.delete(key);
    na.adj.delete(p.b);
    nb.adj.delete(p.a);
    this.version++;
  }

  /** Correspondences of (image, feature) as fresh objects (convenience; hot loops use `heads`). */
  correspondences(image: number, feature: number): ReadonlyArray<Correspondence> {
    const node = this.nodes.get(image);
    if (!node || feature < 0 || feature >= node.numFeatures || node.head[feature] < 0) return EMPTY;
    const out: Correspondence[] = [];
    for (let e = node.head[feature]; e >= 0; e = this.next[e]) out.push({ image: this.img[e], feature: this.feat[e] });
    return out;
  }

  /**
   * Writes up to `outImages.length` correspondences of (image, feature) into the
   * output arrays; returns the total number of correspondences (may exceed capacity).
   */
  getCorrespondences(image: number, feature: number, outImages: Int32Array, outFeatures: Int32Array): number {
    const node = this.nodes.get(image);
    if (!node || feature < 0 || feature >= node.numFeatures) return 0;
    let c = 0;
    const cap = Math.min(outImages.length, outFeatures.length);
    for (let e = node.head[feature]; e >= 0; e = this.next[e]) {
      if (c < cap) {
        outImages[c] = this.img[e];
        outFeatures[c] = this.feat[e];
      }
      c++;
    }
    return c;
  }

  /** Per-feature list heads of an image (pool indices, -1 = none), or undefined for an unknown image. */
  heads(image: number): Int32Array | undefined {
    return this.nodes.get(image)?.head;
  }

  /** Per-feature correspondence counts of an image. */
  featureCounts(image: number): Uint16Array | undefined {
    return this.nodes.get(image)?.count;
  }

  /** Pool: next entry index (-1 terminates). */
  get nextOf(): Int32Array {
    return this.next;
  }
  /** Pool: image id of the corresponding observation. */
  get imageOf(): Int32Array {
    return this.img;
  }
  /** Pool: feature index of the corresponding observation. */
  get featureOf(): Int32Array {
    return this.feat;
  }

  /** Number of correspondences of a single feature. */
  numCorrespondencesOf(image: number, feature: number): number {
    const node = this.nodes.get(image);
    return node && feature >= 0 && feature < node.numFeatures ? node.count[feature] : 0;
  }

  /** True if the feature has exactly one correspondence (COLMAP `IsTwoViewObservation`). */
  isTwoViewObservation(image: number, feature: number): boolean {
    return this.numCorrespondencesOf(image, feature) === 1;
  }

  /** Total number of correspondences of an image (sum over its features). */
  numCorrespondences(image: number): number {
    return this.nodes.get(image)?.numCorrespondences ?? 0;
  }

  /** Number of features of an image with at least one correspondence. */
  numObservations(image: number): number {
    return this.nodes.get(image)?.numObservations ?? 0;
  }

  numCorrespondencesBetween(a: number, b: number): number {
    const p = this.pairMap.get(pairKey(a, b));
    return p ? p.matches.length >> 1 : 0;
  }

  /** Matches of the pair oriented (a-feature, b-feature); empty if unconnected. */
  matchesBetween(a: number, b: number): Uint32Array {
    const p = this.pairMap.get(pairKey(a, b));
    if (!p) return EMPTY_U32;
    if (a < b) return p.matches;
    const m = p.matches;
    const out = new Uint32Array(m.length);
    for (let k = 0; k < m.length; k += 2) {
      out[k] = m[k + 1];
      out[k + 1] = m[k];
    }
    return out;
  }

  /** All connected pairs as [min id, max id, number of correspondences]. */
  *imagePairs(): IterableIterator<[number, number, number]> {
    for (const [key, p] of this.pairMap) {
      const [a, b] = pairIds(key);
      yield [a, b, p.matches.length >> 1];
    }
  }

  get numPairs(): number {
    return this.pairMap.size;
  }

  /** Images connected to `image` → number of correspondences (live view; do not mutate). */
  neighbours(image: number): ReadonlyMap<number, number> {
    return this.nodes.get(image)?.adj ?? EMPTY_MAP;
  }
}
