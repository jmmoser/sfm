/**
 * Small helpers shared by the dense fusion / meshing modules (undistortion, view
 * selection, fusion, TSDF, marching cubes, simplification, point-cloud filters).
 * Kept local to `src/dense` so the module does not depend on the state of
 * `src/math` / `src/geometry`.
 */
import type { CameraIntrinsics, CancelToken, Pose } from '../core/types';
import type { DepthMap, DepthMapStore } from './types';

// ---------------------------------------------------------------------------
// Cancellation / cooperative yielding

/** Thrown by the async dense stages when their CancelToken is set. */
export class DenseCancelledError extends Error {
  constructor(stage: string) {
    super(`${stage}: cancelled`);
    this.name = 'CancelledError';
  }
}

/** Yield to the event loop (lets a worker process cancel / progress messages). */
export function yieldToEventLoop(): Promise<void> {
  const s = (globalThis as { scheduler?: { yield?: () => Promise<void> } }).scheduler;
  if (s && typeof s.yield === 'function') return s.yield();
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

/** Time-sliced yielding: `await y.tick()` yields at most every `sliceMs` and throws on cancel. */
export class Yielder {
  private last = now();
  constructor(
    private readonly stage: string,
    private readonly cancel?: CancelToken,
    private readonly sliceMs = 40,
  ) {}

  /** True when the slice is used up (call `tick()` then). Cheap enough for inner loops every ~1k items. */
  due(): boolean {
    return now() - this.last >= this.sliceMs;
  }

  async tick(force = false): Promise<void> {
    if (this.cancel?.cancelled) throw new DenseCancelledError(this.stage);
    if (!force && !this.due()) return;
    await yieldToEventLoop();
    this.last = now();
    if (this.cancel?.cancelled) throw new DenseCancelledError(this.stage);
  }

  check(): void {
    if (this.cancel?.cancelled) throw new DenseCancelledError(this.stage);
  }
}

// ---------------------------------------------------------------------------
// Cameras

/**
 * Intrinsics of the same camera for an image resampled to newWidth × newHeight
 * (ARCHITECTURE §2 rule, identical to geometry/scaleIntrinsics): c' = (c + 0.5)·s − 0.5
 * per axis, f' = f·(sx + sy)/2. Distortion unchanged.
 */
export function scaleCamera(cam: CameraIntrinsics, newWidth: number, newHeight: number): CameraIntrinsics {
  if (newWidth === cam.width && newHeight === cam.height) return { ...cam };
  const sx = newWidth / cam.width;
  const sy = newHeight / cam.height;
  return {
    width: newWidth,
    height: newHeight,
    f: cam.f * 0.5 * (sx + sy),
    cx: (cam.cx + 0.5) * sx - 0.5,
    cy: (cam.cy + 0.5) * sy - 0.5,
    k1: cam.k1,
    k2: cam.k2,
  };
}

/** Camera centre C = −Rᵀ t. */
export function poseCentre(pose: Pose, out: Float64Array = new Float64Array(3)): Float64Array {
  const R = pose.R;
  const t = pose.t;
  out[0] = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]);
  out[1] = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]);
  out[2] = -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);
  return out;
}

/**
 * Relative transform ref-camera → other-camera: X_o = R X_r + t
 * (R = R_o R_rᵀ, t = t_o − R t_r).
 */
export function relativeTransform(ref: Pose, other: Pose): { R: Float64Array; t: Float64Array } {
  const a = other.R;
  const b = ref.R;
  const R = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      R[3 * i + j] = a[3 * i] * b[3 * j] + a[3 * i + 1] * b[3 * j + 1] + a[3 * i + 2] * b[3 * j + 2];
    }
  }
  const tr = ref.t;
  const t = new Float64Array(3);
  for (let i = 0; i < 3; i++) t[i] = other.t[i] - (R[3 * i] * tr[0] + R[3 * i + 1] * tr[1] + R[3 * i + 2] * tr[2]);
  return { R, t };
}

// ---------------------------------------------------------------------------
// Depth-map cache

/**
 * Small LRU cache in front of a DepthMapStore (depth maps are large: 20 B/px).
 * Concurrent requests for the same id share one load.
 */
export class DepthMapCache {
  private readonly entries = new Map<number, DepthMap | null>();
  private readonly pending = new Map<number, Promise<DepthMap | null>>();
  hits = 0;
  misses = 0;

  constructor(
    private readonly store: DepthMapStore,
    public capacity: number,
  ) {
    this.capacity = Math.max(1, Math.floor(capacity));
  }

  peek(id: number): DepthMap | null | undefined {
    return this.entries.get(id);
  }

  async get(id: number): Promise<DepthMap | null> {
    if (this.entries.has(id)) {
      const v = this.entries.get(id)!;
      // refresh LRU position
      this.entries.delete(id);
      this.entries.set(id, v);
      this.hits++;
      return v;
    }
    const p = this.pending.get(id);
    if (p) return p;
    this.misses++;
    const load = this.store
      .get(id)
      .then((d) => {
        this.pending.delete(id);
        this.entries.set(id, d);
        this.evict();
        return d;
      })
      .catch((e: unknown) => {
        this.pending.delete(id);
        throw e;
      });
    this.pending.set(id, load);
    return load;
  }

  /** Load several ids (in parallel) and return them in order. */
  async getMany(ids: readonly number[]): Promise<(DepthMap | null)[]> {
    if (ids.length + 1 > this.capacity) this.capacity = ids.length + 1;
    return Promise.all(ids.map((id) => this.get(id)));
  }

  private evict(): void {
    while (this.entries.size > this.capacity) {
      const first = this.entries.keys().next().value as number;
      this.entries.delete(first);
    }
  }

  clear(): void {
    this.entries.clear();
  }
}

// ---------------------------------------------------------------------------
// Hash map for integer triples (open addressing, linear probing)

/**
 * Hash map (int32, int32, int32) → non-negative int32, typed-array backed (no per-entry
 * allocation). Used for voxel blocks, mesh edges and point-cloud voxel grids.
 */
export class Int3HashMap {
  private keys: Int32Array;
  private vals: Int32Array;
  private mask: number;
  size = 0;

  constructor(expected = 1024) {
    let cap = 16;
    while (cap < expected * 2) cap *= 2;
    this.keys = new Int32Array(cap * 3);
    this.vals = new Int32Array(cap).fill(-1);
    this.mask = cap - 1;
  }

  private static hash(x: number, y: number, z: number): number {
    let h = Math.imul(x, 0x8da6b343) ^ Math.imul(y, 0xd8163841) ^ Math.imul(z, 0xcb1ab31f);
    h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
    h ^= h >>> 13;
    return h >>> 0;
  }

  /** Value for the key, −1 if absent. */
  get(x: number, y: number, z: number): number {
    const keys = this.keys;
    const vals = this.vals;
    const mask = this.mask;
    let s = Int3HashMap.hash(x, y, z) & mask;
    for (;;) {
      const v = vals[s];
      if (v < 0) return -1;
      const k = 3 * s;
      if (keys[k] === x && keys[k + 1] === y && keys[k + 2] === z) return v;
      s = (s + 1) & mask;
    }
  }

  /**
   * Returns the value stored for the key; inserts `value` first if the key is absent
   * (`inserted` then becomes true).
   */
  getOrInsert(x: number, y: number, z: number, value: number): number {
    if ((this.size + 1) * 2 > this.vals.length) this.grow();
    const keys = this.keys;
    const vals = this.vals;
    const mask = this.mask;
    let s = Int3HashMap.hash(x, y, z) & mask;
    for (;;) {
      const v = vals[s];
      const k = 3 * s;
      if (v < 0) {
        keys[k] = x;
        keys[k + 1] = y;
        keys[k + 2] = z;
        vals[s] = value;
        this.size++;
        this.inserted = true;
        return value;
      }
      if (keys[k] === x && keys[k + 1] === y && keys[k + 2] === z) {
        this.inserted = false;
        return v;
      }
      s = (s + 1) & mask;
    }
  }

  /** Set by `getOrInsert`: whether the last call inserted a new key. */
  inserted = false;

  private grow(): void {
    const oldKeys = this.keys;
    const oldVals = this.vals;
    const cap = oldVals.length * 2;
    this.keys = new Int32Array(cap * 3);
    this.vals = new Int32Array(cap).fill(-1);
    this.mask = cap - 1;
    const keys = this.keys;
    const vals = this.vals;
    const mask = this.mask;
    for (let i = 0; i < oldVals.length; i++) {
      const v = oldVals[i];
      if (v < 0) continue;
      const x = oldKeys[3 * i];
      const y = oldKeys[3 * i + 1];
      const z = oldKeys[3 * i + 2];
      let s = Int3HashMap.hash(x, y, z) & mask;
      while (vals[s] >= 0) s = (s + 1) & mask;
      keys[3 * s] = x;
      keys[3 * s + 1] = y;
      keys[3 * s + 2] = z;
      vals[s] = v;
    }
  }

  /** Calls `cb(x, y, z, value)` for every entry (table order). */
  forEach(cb: (x: number, y: number, z: number, value: number) => void): void {
    const keys = this.keys;
    const vals = this.vals;
    for (let s = 0; s < vals.length; s++) {
      const v = vals[s];
      if (v >= 0) cb(keys[3 * s], keys[3 * s + 1], keys[3 * s + 2], v);
    }
  }

  /** Approximate memory footprint in bytes. */
  bytes(): number {
    return this.keys.byteLength + this.vals.byteLength;
  }

  clear(): void {
    this.vals.fill(-1);
    this.size = 0;
  }
}

// ---------------------------------------------------------------------------
// Misc numeric helpers

/** Value at fraction q ∈ [0, 1] of a numeric sample (sorts a copy; NaN-free input). */
export function quantile(values: ArrayLike<number>, q: number): number {
  const n = values.length;
  if (n === 0) return NaN;
  const s = Float64Array.from(values as ArrayLike<number>).sort();
  const pos = Math.min(n - 1, Math.max(0, q * (n - 1)));
  const lo = Math.floor(pos);
  const hi = Math.min(n - 1, lo + 1);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/** Grow a typed array to at least `minLength` (doubling), preserving contents. */
export function growF32(a: Float32Array<ArrayBuffer>, minLength: number): Float32Array<ArrayBuffer> {
  if (a.length >= minLength) return a;
  const b = new Float32Array(Math.max(minLength, a.length * 2, 64));
  b.set(a);
  return b;
}

export function growU8(a: Uint8Array<ArrayBuffer>, minLength: number): Uint8Array<ArrayBuffer> {
  if (a.length >= minLength) return a;
  const b = new Uint8Array(Math.max(minLength, a.length * 2, 64));
  b.set(a);
  return b;
}

export function growU32(a: Uint32Array<ArrayBuffer>, minLength: number): Uint32Array<ArrayBuffer> {
  if (a.length >= minLength) return a;
  const b = new Uint32Array(Math.max(minLength, a.length * 2, 64));
  b.set(a);
  return b;
}

/**
 * RGB (3 bytes / px) of `image` resampled to the depth-map grid w × h (box filter when
 * shrinking, nearest when the sizes match / enlarging). Used to colour fused points
 * and TSDF voxels when depth maps were computed at a lower resolution than the image.
 */
export function rgbAtResolution(image: { width: number; height: number; data: Uint8Array | Uint8ClampedArray }, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h * 3);
  const W = image.width;
  const H = image.height;
  const d = image.data;
  if (W === w && H === h) {
    for (let i = 0, j = 0; i < w * h; i++, j += 4) {
      out[3 * i] = d[j];
      out[3 * i + 1] = d[j + 1];
      out[3 * i + 2] = d[j + 2];
    }
    return out;
  }
  const sx = W / w;
  const sy = H / h;
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, Math.min(H - 1, Math.floor(y * sy)));
    const y1 = Math.max(y0 + 1, Math.min(H, Math.ceil((y + 1) * sy)));
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, Math.min(W - 1, Math.floor(x * sx)));
      const x1 = Math.max(x0 + 1, Math.min(W, Math.ceil((x + 1) * sx)));
      let r = 0;
      let g = 0;
      let b = 0;
      let n = 0;
      for (let yy = y0; yy < y1; yy++) {
        for (let xx = x0; xx < x1; xx++) {
          const j = 4 * (yy * W + xx);
          r += d[j];
          g += d[j + 1];
          b += d[j + 2];
          n++;
        }
      }
      const o = 3 * (y * w + x);
      out[o] = Math.round(r / n);
      out[o + 1] = Math.round(g / n);
      out[o + 2] = Math.round(b / n);
    }
  }
  return out;
}
