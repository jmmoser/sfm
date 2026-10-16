/**
 * Byte-bounded LRU cache and keyframe image decoding (JPEG Blob → RGBA through
 * createImageBitmap + OffscreenCanvas in the worker; raw RGBA in Node tests).
 *
 * Memory discipline (iPhone tab budget ≈ 1.2 GB, ARCHITECTURE §2.1): full-resolution RGBA of
 * all keyframes is never kept — keyframes live as JPEGs in the store and are decoded on demand
 * at the size the consumer needs, through a cache with a byte budget.
 */
import type { RGBAImage } from '../core/types';
import { hasOffscreenCanvas, resizeRGBA } from './frame-ingest';
import { isRGBAImage, type StoredImage } from './persistence';

export class LruCache<K, V> {
  private readonly map = new Map<K, { v: V; bytes: number }>();
  private readonly pending = new Map<K, Promise<V>>();
  private used = 0;
  hits = 0;
  misses = 0;

  constructor(
    public capacityBytes: number,
    private readonly sizeOf: (v: V) => number,
  ) {}

  get bytes(): number {
    return this.used;
  }

  get size(): number {
    return this.map.size;
  }

  has(k: K): boolean {
    return this.map.has(k);
  }

  get(k: K): V | undefined {
    const e = this.map.get(k);
    if (!e) return undefined;
    this.map.delete(k);
    this.map.set(k, e);
    return e.v;
  }

  set(k: K, v: V): void {
    const old = this.map.get(k);
    if (old) {
      this.used -= old.bytes;
      this.map.delete(k);
    }
    const bytes = Math.max(0, this.sizeOf(v));
    this.map.set(k, { v, bytes });
    this.used += bytes;
    this.evict(k);
  }

  delete(k: K): void {
    const e = this.map.get(k);
    if (!e) return;
    this.used -= e.bytes;
    this.map.delete(k);
  }

  /** Cached value, or the loader's result (concurrent loads of one key are shared). */
  async getOrLoad(k: K, load: () => Promise<V>): Promise<V> {
    const hit = this.get(k);
    if (hit !== undefined) {
      this.hits++;
      return hit;
    }
    const p = this.pending.get(k);
    if (p) return p;
    this.misses++;
    const run = load().then(
      (v) => {
        this.pending.delete(k);
        this.set(k, v);
        return v;
      },
      (e: unknown) => {
        this.pending.delete(k);
        throw e;
      },
    );
    this.pending.set(k, run);
    return run;
  }

  /** Evict least recently used entries until within budget (`keep` is never evicted). */
  private evict(keep?: K): void {
    if (this.used <= this.capacityBytes) return;
    for (const [k, e] of this.map) {
      if (this.used <= this.capacityBytes) break;
      if (k === keep) continue;
      this.map.delete(k);
      this.used -= e.bytes;
    }
  }

  clear(): void {
    this.map.clear();
    this.used = 0;
  }

  keys(): K[] {
    return [...this.map.keys()];
  }
}

/** Bytes of an RGBA image. */
export function rgbaBytes(img: RGBAImage): number {
  return img.data.byteLength;
}

/**
 * Decode a stored keyframe image to RGBA, optionally resampled to `w × h` (drawImage with
 * high-quality smoothing in the browser, area averaging for raw RGBA).
 */
export async function decodeStoredImage(img: StoredImage, w?: number, h?: number): Promise<RGBAImage> {
  if (isRGBAImage(img)) {
    if (w === undefined || h === undefined || (w === img.width && h === img.height)) return img;
    return resizeRGBA(img, w, h);
  }
  if (typeof createImageBitmap !== 'function' || !hasOffscreenCanvas()) throw new Error('decoding JPEG keyframes needs createImageBitmap + OffscreenCanvas');
  const bmp = await createImageBitmap(img);
  try {
    const W = w ?? bmp.width;
    const H = h ?? bmp.height;
    const c = new OffscreenCanvas(W, H);
    const ctx = c.getContext('2d', { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D | null;
    if (!ctx) throw new Error('OffscreenCanvas 2D context unavailable');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bmp, 0, 0, W, H);
    const d = ctx.getImageData(0, 0, W, H);
    c.width = 1;
    c.height = 1;
    return { width: d.width, height: d.height, data: d.data };
  } finally {
    bmp.close();
  }
}

/** Natural size of a stored image (decodes the JPEG header via createImageBitmap). */
export async function storedImageSize(img: StoredImage): Promise<{ width: number; height: number }> {
  if (isRGBAImage(img)) return { width: img.width, height: img.height };
  const bmp = await createImageBitmap(img);
  const s = { width: bmp.width, height: bmp.height };
  bmp.close();
  return s;
}
