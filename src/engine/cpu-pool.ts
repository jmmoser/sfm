/**
 * Pool of CPU helper workers (src/workers/cpu.worker.ts) for independent CPU jobs that otherwise
 * run one after another on the engine worker's single thread:
 *  - refine: the two-view verification of the candidate pairs, the visual-word quantisation of the
 *    keyframes and the rolling-shutter hypotheses (short subset BAs) (e2e Metal object-real /
 *    yard-real: 10–23 s, ≈ 7.5 s and ≈ 12 s of refine);
 *  - densify (`DenseHelpers`, src/engine/dense-jobs.ts): keyframe decoding + undistortion +
 *    rolling-shutter warp of the next views and the final maps' halo filter + float16
 *    serialisation, while the engine thread drives the GPU PatchMatch passes.
 * The jobs are pure and deterministic, so the results are those of the engine thread; callers
 * apply them in their own order.
 *
 * Nested dedicated workers: Safari 15.5+ (docs/research/ios-platform.md). Without `Worker` in the
 * engine's scope (Node tests, old browsers) or when a worker cannot be created, `create` returns
 * null and the callers run the jobs inline; a worker that fails later rejects its jobs (callers fall
 * back to inline for them) and is not used again.
 */
import type { CameraIntrinsics, FeatureSet, MatchPairs, RGBAImage } from '../core/types';
import type { DepthMap } from '../dense/types';
import type { TwoViewGeometryResult, TwoViewOptions } from '../geometry/twoview';
import type { RollingShutterModel, RSBundleCost } from '../sfm/rolling-shutter';
import type { VocabularyIndex } from '../sfm/vocab';
import type { CpuWorkerRequest, CpuWorkerResponse } from '../workers/cpu.worker';
import type { DenseHelpers, DenseViewRS, HaloJobInput, HaloStoreResult } from './dense-jobs';
import type { StoredImage } from './persistence';

interface PoolWorker {
  worker: Worker;
  pending: Map<number, { resolve: (r: CpuWorkerResponse) => void; reject: (e: Error) => void }>;
  /** Vocabularies whose codebook this worker holds, rolling-shutter cost functions it holds. */
  codebooks: Set<number>;
  rsCosts: Set<number>;
  broken: boolean;
}

/** Workers for a pool on this device: hardware threads − 2 (the engine thread, the page), 0–3. */
export function defaultCpuWorkers(): number {
  const hc = typeof navigator !== 'undefined' && navigator.hardwareConcurrency > 0 ? navigator.hardwareConcurrency : 2;
  return Math.max(0, Math.min(3, hc - 2));
}

export class CpuPool implements DenseHelpers {
  private readonly workers: PoolWorker[];
  private nextId = 1;
  /** Compute time reported by the workers (ms), per operation. */
  readonly busyMs = { verify: 0, quantize: 0, rsCost: 0, denseView: 0, haloStore: 0 };
  private readonly vocabKeys = new WeakMap<VocabularyIndex, number>();
  private nextVocabKey = 1;
  private readonly rsKeys = new WeakMap<RSBundleCost, number>();
  private nextRsKey = 1;

  private constructor(workers: Worker[]) {
    this.workers = workers.map((worker) => {
      const w: PoolWorker = { worker, pending: new Map(), codebooks: new Set(), rsCosts: new Set(), broken: false };
      worker.onmessage = (ev: MessageEvent<CpuWorkerResponse>) => {
        const job = w.pending.get(ev.data.id);
        if (!job) return;
        w.pending.delete(ev.data.id);
        job.resolve(ev.data);
      };
      worker.onerror = (ev: ErrorEvent) => {
        ev.preventDefault?.();
        this.fail(w, new Error(`cpu worker: ${ev.message || 'error'}`));
      };
      // a reply that cannot be deserialised would otherwise leave its job pending for ever
      worker.onmessageerror = () => this.fail(w, new Error('cpu worker: undeliverable message'));
      return w;
    });
  }

  /** Whether this realm can create the workers (`Worker` exists: not in Node tests). */
  static available(): boolean {
    return typeof Worker !== 'undefined';
  }

  /** A pool of `size` workers (default `defaultCpuWorkers()`), or null when workers are unavailable. */
  static create(size = defaultCpuWorkers()): CpuPool | null {
    if (!(size >= 1) || !CpuPool.available()) return null;
    const workers: Worker[] = [];
    try {
      for (let k = 0; k < size; k++) workers.push(new Worker(new URL('../workers/cpu.worker.ts', import.meta.url), { type: 'module', name: `cpu-${k}` }));
    } catch {
      for (const w of workers) w.terminate();
      return null;
    }
    return new CpuPool(workers);
  }

  /** Workers still in use. */
  get size(): number {
    return this.workers.filter((w) => !w.broken).length;
  }

  private fail(w: PoolWorker, e: Error): void {
    w.broken = true;
    for (const job of w.pending.values()) job.reject(e);
    w.pending.clear();
    w.worker.terminate();
  }

  /** The usable worker with the fewest jobs in flight. */
  private pick(): PoolWorker {
    let best: PoolWorker | null = null;
    for (const w of this.workers) if (!w.broken && (!best || w.pending.size < best.pending.size)) best = w;
    if (!best) throw new Error('cpu pool: no usable worker');
    return best;
  }

  private post(w: PoolWorker, req: CpuWorkerRequest, transfer: Transferable[] = []): Promise<CpuWorkerResponse> {
    return new Promise((resolve, reject) => {
      w.pending.set(req.id, { resolve, reject });
      try {
        w.worker.postMessage(req, transfer);
      } catch (e) {
        w.pending.delete(req.id);
        reject(e as Error);
      }
    });
  }

  private static result<T>(r: CpuWorkerResponse): T {
    if (r.error !== undefined) throw new Error(`cpu worker: ${r.error}`);
    return r.result as T;
  }

  /** `estimateTwoViewGeometry(kp1, kp2, matches, cam1, cam2, opts)` on a worker (inputs are copied), with its compute time. */
  async verify(
    kp1: Float32Array,
    kp2: Float32Array,
    matches: MatchPairs,
    cam1: CameraIntrinsics,
    cam2: CameraIntrinsics,
    opts: TwoViewOptions,
  ): Promise<{ geometry: TwoViewGeometryResult; ms: number }> {
    const w = this.pick();
    const r = await this.post(w, { id: this.nextId++, op: 'verify', kp1, kp2, matches, cam1, cam2, opts });
    this.busyMs.verify += r.ms;
    return { geometry: CpuPool.result<TwoViewGeometryResult>(r), ms: r.ms };
  }

  /** `vocab.quantize(features)` on a worker (the codebook is sent to each worker once). */
  async quantize(vocab: VocabularyIndex, features: FeatureSet): Promise<Uint32Array> {
    let key = this.vocabKeys.get(vocab);
    if (key === undefined) this.vocabKeys.set(vocab, (key = this.nextVocabKey++));
    const w = this.pick();
    if (!w.codebooks.has(key)) {
      w.codebooks.add(key);
      // ordered before the quantise request on the same worker; a failure surfaces through it
      this.post(w, { id: this.nextId++, op: 'codebook', key, codebook: vocab.codebook, options: { ...vocab.options } }).catch(() => undefined);
    }
    const r = await this.post(w, { id: this.nextId++, op: 'quantize', key, count: features.count, descriptors: features.descriptors, scores: features.scores });
    this.busyMs.quantize += r.ms;
    return CpuPool.result<Uint32Array>(r);
  }

  /** `bc.cost(m)` for every hypothesis, in parallel over the workers (the cost data sent once per worker). */
  rsCosts(bc: RSBundleCost, models: RollingShutterModel[]): Promise<number[]> {
    let key = this.rsKeys.get(bc);
    if (key === undefined) this.rsKeys.set(bc, (key = this.nextRsKey++));
    const k = key;
    return Promise.all(
      models.map(async (model) => {
        const w = this.pick();
        const first = !w.rsCosts.has(k);
        if (first) w.rsCosts.add(k);
        const r = await this.post(w, { id: this.nextId++, op: 'rsCost', key: k, model, ...(first ? { data: bc.toData() } : {}) });
        this.busyMs.rsCost += r.ms;
        return CpuPool.result<number>(r);
      }),
    );
  }

  /**
   * `decodeDenseView` (+ `denseColourThumbnail` when `thumbBytes` > 0) on a worker: the stored
   * keyframe is posted (a Blob is shared, raw RGBA copied), the images come back transferred.
   */
  async denseView(
    stored: StoredImage,
    distorted: CameraIntrinsics,
    maxSize: number,
    rs: DenseViewRS | null,
    thumbBytes: number,
  ): Promise<{ image: RGBAImage; cam: CameraIntrinsics; thumb: RGBAImage | null }> {
    const w = this.pick();
    const r = await this.post(w, { id: this.nextId++, op: 'denseView', stored, distorted, maxSize, rs, thumbBytes });
    this.busyMs.denseView += r.ms;
    return CpuPool.result<{ image: RGBAImage; cam: CameraIntrinsics; thumb: RGBAImage | null }>(r);
  }

  /**
   * `haloStoreJob(copy of map, halo, serialize)` on a worker. The map and the image are COPIED (not
   * transferred): the caller keeps its own, so a worker that fails never loses a depth map.
   */
  async haloStore(map: DepthMap, halo: HaloJobInput | null, serialize: boolean): Promise<HaloStoreResult> {
    const w = this.pick();
    const r = await this.post(w, { id: this.nextId++, op: 'haloStore', map, halo, serialize });
    this.busyMs.haloStore += r.ms;
    return CpuPool.result<HaloStoreResult>(r);
  }

  dispose(): void {
    for (const w of this.workers) if (!w.broken) this.fail(w, new Error('cpu pool disposed'));
  }
}
