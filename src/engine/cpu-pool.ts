/**
 * Pool of CPU helper workers (src/workers/cpu.worker.ts) for refine's independent CPU jobs: the
 * two-view verification of the candidate pairs and the visual-word quantisation of the keyframes,
 * which otherwise run one after another on the engine worker's single thread (e2e Metal
 * object-real / yard-real: 10–23 s and ≈ 7.5 s of refine). The jobs are pure and deterministic, so
 * the results are those of the engine thread; callers apply them in their own order.
 *
 * Nested dedicated workers: Safari 15.5+ (docs/research/ios-platform.md). Without `Worker` in the
 * engine's scope (Node tests, old browsers) or when a worker cannot be created, `create` returns
 * null and the callers run the jobs inline; a worker that fails later rejects its jobs (callers fall
 * back to inline for them) and is not used again.
 */
import type { CameraIntrinsics, FeatureSet, MatchPairs } from '../core/types';
import type { TwoViewGeometryResult, TwoViewOptions } from '../geometry/twoview';
import type { VocabularyIndex } from '../sfm/vocab';
import type { CpuWorkerRequest, CpuWorkerResponse } from '../workers/cpu.worker';

interface PoolWorker {
  worker: Worker;
  pending: Map<number, { resolve: (r: CpuWorkerResponse) => void; reject: (e: Error) => void }>;
  /** Vocabularies whose codebook this worker holds. */
  codebooks: Set<number>;
  broken: boolean;
}

/** Workers for a pool on this device: hardware threads − 2 (the engine thread, the page), 1–3. */
export function defaultCpuWorkers(): number {
  const hc = typeof navigator !== 'undefined' && navigator.hardwareConcurrency > 0 ? navigator.hardwareConcurrency : 2;
  return Math.max(0, Math.min(3, hc - 2));
}

export class CpuPool {
  private readonly workers: PoolWorker[];
  private nextId = 1;
  /** Compute time reported by the workers (ms), per operation. */
  readonly busyMs = { verify: 0, quantize: 0 };
  private readonly vocabKeys = new WeakMap<VocabularyIndex, number>();
  private nextVocabKey = 1;

  private constructor(workers: Worker[]) {
    this.workers = workers.map((worker) => {
      const w: PoolWorker = { worker, pending: new Map(), codebooks: new Set(), broken: false };
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
      return w;
    });
  }

  /** A pool of `size` workers (default `defaultCpuWorkers()`), or null when workers are unavailable. */
  static create(size = defaultCpuWorkers()): CpuPool | null {
    if (!(size >= 1) || typeof Worker === 'undefined') return null;
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

  dispose(): void {
    for (const w of this.workers) if (!w.broken) this.fail(w, new Error('cpu pool disposed'));
  }
}
