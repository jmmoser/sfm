/**
 * Depth-map stores for the dense stage (ENGINE.md "Dense / mesh"): depth maps are large
 * (20 B/px; 1280×960 ≈ 25 MB), so with persistence they go to IndexedDB (float16 normals /
 * cost, 12 B/px) behind a small LRU of resident maps; without persistence (tests) they stay in
 * memory.
 */
import type { DepthMap, DepthMapStore } from '../dense/types';
import { deserializeDepthMap, serializeDepthMap } from '../io/serialize';
import { LruCache } from './image-cache';
import type { EngineStore } from './persistence';

export function depthMapBytes(d: DepthMap): number {
  return d.depth.byteLength + d.normal.byteLength + d.cost.byteLength;
}

/**
 * DepthMapStore under a key prefix. `store` persistent ⇒ write-through to IndexedDB + LRU of
 * `cacheBytes`; otherwise every map is kept in memory.
 */
export class EngineDepthStore implements DepthMapStore {
  private readonly memory = new Map<number, DepthMap>();
  private readonly cache: LruCache<number, DepthMap>;
  private readonly known = new Set<number>();

  constructor(
    private readonly store: EngineStore | null,
    readonly prefix: string,
    cacheBytes: number,
  ) {
    this.cache = new LruCache<number, DepthMap>(cacheBytes, depthMapBytes);
  }

  get persistent(): boolean {
    return !!this.store && this.store.persistent;
  }

  private key(id: number): string {
    return `${this.prefix}/${id}`;
  }

  async get(id: number): Promise<DepthMap | null> {
    if (!this.persistent) return this.memory.get(id) ?? null;
    const hit = this.cache.get(id);
    if (hit) return hit;
    const buf = await this.store!.getBinary(this.key(id));
    if (!buf) return null;
    const d = deserializeDepthMap(buf) as DepthMap;
    this.cache.set(id, d);
    return d;
  }

  async put(id: number, d: DepthMap): Promise<void> {
    this.known.add(id);
    if (!this.persistent) {
      this.memory.set(id, d);
      return;
    }
    this.cache.set(id, d);
    await this.store!.putBinary(this.key(id), serializeDepthMap(d, { float16: true }));
  }

  /** Ids written through this instance (or found in the store by `scan`). */
  ids(): number[] {
    return [...this.known].sort((a, b) => a - b);
  }

  /** Discover persisted maps of this prefix (after a reload). */
  async scan(): Promise<number[]> {
    if (this.persistent) {
      for (const k of await this.store!.listBinaryKeys(`${this.prefix}/`)) {
        const id = Number(k.slice(this.prefix.length + 1));
        if (Number.isInteger(id)) this.known.add(id);
      }
    } else for (const id of this.memory.keys()) this.known.add(id);
    return this.ids();
  }

  /** Delete every map of this prefix (memory + store). */
  async clear(): Promise<void> {
    const ids = await this.scan();
    this.memory.clear();
    this.cache.clear();
    if (this.persistent) for (const id of ids) await this.store!.deleteBinary(this.key(id)).catch(() => undefined);
    this.known.clear();
  }

  /** Drop resident maps (persisted copies stay). */
  releaseMemory(): void {
    this.cache.clear();
  }
}
