/**
 * WebGPU descriptor matcher with the semantics of matchCPU / matchGuidedCPU
 * (match.ts): Euclidean distance of unit RootSIFT vectors, Lowe ratio test on
 * best / second best, max distance, mutual nearest neighbour (over the same guided
 * candidate sets), output sorted by index in a.
 *
 * The GPU computes exact integer descriptor dot products (GEMM-style tiles, see
 * shaders/match-wgsl.ts) and returns, per query and column chunk, the indices and
 * integer dots of the best two candidates (ranked by an f32 similarity). The host
 * recomputes those few similarities exactly in f64 like the CPU matcher and applies
 * the final decision with the same code, so results coincide with matchCPU except for
 * candidates whose f32 similarities tie within rounding (third-best swaps), which is
 * vanishingly rare. Both directions run on the GPU for the mutual check (skipped when
 * `mutual: false`). Guided matching evaluates the constraint in the kernel and culls
 * whole tiles using spatially sorted features ('positions', 'H' -> predicted
 * positions, 'F' -> epipolar band).
 *
 * Descriptor buffers (packed 4 bytes per u32) and inverse norms (GPU f32 and host f64)
 * are cached per FeatureSet (WeakMap + LRU byte budget + FinalizationRegistry); a cheap
 * content hash is verified on every call, so descriptor arrays refilled in place are
 * re-uploaded and their norms recomputed.
 */

import type { FeatureSet, MatchPairs } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';
import type { GPUContext } from '../gpu/device';
import {
  BudgetEncoder,
  FeatureInputError,
  GPUFeatureError,
  GPUWorkDroppedError,
  GrowBuffer,
  ParamBlock,
  PARAM_SLOT,
  STAMP_WGSL,
  Staging,
  WorkStamp,
  compilePipeline,
  errorScope,
} from './gpu-common';
import { matchCPU, matchGuidedCPU } from './match';
import type { MatchGuide, MatchOptions } from './match';
import { descriptorHash, descriptorInvNorms, emitMatches, packDescriptorWords, planChunks, prepareGuide, reduceTop2, validateFeatureSet } from './match-gpu-host';
import type { GuidePrep } from './match-gpu-host';
import { MATCH_TILE, matchShader } from './shaders/match-wgsl';
import type { MatchKernelMode } from './shaders/match-wgsl';

export interface MatcherGPUConfig {
  /** GPU memory budget of the per-FeatureSet descriptor cache (bytes). Default 64 MiB. */
  cacheBytes?: number;
  /** Descriptor pairs per command buffer before submitting and waiting. Default 6.4e7. */
  pairBudget?: number;
  /** Target number of workgroups per pass (column chunking for small row counts). Default 128. */
  targetWorkgroups?: number;
  /** Attempts per call when the GPU did not execute submitted work (page hidden). Default 3. */
  maxAttempts?: number;
  /** @internal Testing hook: command buffers (0-based per attempt) to discard instead of submitting. */
  testDropSubmissions?: (attempt: number) => readonly number[];
}

/** Result of MatcherGPU.selfTest(). */
export interface MatcherSelfTestResult {
  ok: boolean;
  /** Fraction of CPU pairs reproduced (worst of the plain / guided checks). */
  overlap: number;
  ms: number;
  error?: string;
  /** The test could not run because submitted GPU work was dropped (page hidden): run it again later. */
  transient?: boolean;
}

export interface MatcherGPUTimings {
  totalMs: number;
  /** Norms, hashing, uploads, guide preparation. */
  prepareMs: number;
  /** GPU work + readback. */
  gpuMs: number;
  /** Exact host decision. */
  finalizeMs: number;
  submits: number;
  uploadedSets: number;
  attempts: number;
}

interface CacheEntry {
  count: number;
  hash: number;
  /** f64 inverse norms (host re-rank), kept so a cache hit skips recomputing them. */
  invNorm: Float64Array;
  desc: GPUBuffer;
  inv: GPUBuffer;
  bytes: number;
  alive: boolean;
}

const STORAGE = GPUBufferUsage.STORAGE;
const COPY_DST = GPUBufferUsage.COPY_DST;
const COPY_SRC = GPUBufferUsage.COPY_SRC;
const UNIFORM = GPUBufferUsage.UNIFORM;

type PipeKey = 'plain' | 'pos-f' | 'pos-r' | 'epi-f' | 'epi-r' | 'stamp';

/** WebGPU brute-force / guided matcher (see file header). */
export class MatcherGPU {
  lastTimings: MatcherGPUTimings | null = null;

  private readonly device: GPUDevice;
  private readonly lostPromise: Promise<unknown>;
  private isLost = false;
  private destroyed = false;
  private readonly pipes: Record<PipeKey, GPUComputePipeline>;
  private readonly cache = new WeakMap<FeatureSet, CacheEntry>();
  private readonly lru = new Set<CacheEntry>();
  private cacheUsed = 0;
  private readonly cacheBytes: number;
  private readonly pairBudget: number;
  private readonly targetWG: number;
  private readonly registry: FinalizationRegistry<CacheEntry> | null;
  private readonly out: GrowBuffer;
  private readonly params: GrowBuffer;
  private readonly aux: GrowBuffer;
  private readonly aPerm: GrowBuffer;
  private readonly bPerm: GrowBuffer;
  private readonly staging: Staging;
  private readonly stampBuf: GPUBuffer;
  private readonly stamp: WorkStamp;
  private readonly maxAttempts: number;
  private readonly testDrop: ((attempt: number) => readonly number[]) | null;
  private chain: Promise<unknown> = Promise.resolve();
  /** A match is running (its GPU buffers must not be freed under it). */
  private busy = false;
  /** clearCache() was called while busy: clear when the match ends. */
  private clearPending = false;
  private freed = false;

  private constructor(ctx: GPUContext, pipes: Record<PipeKey, GPUComputePipeline>, cfg: MatcherGPUConfig) {
    const device = ctx.device;
    this.device = device;
    this.pipes = pipes;
    this.cacheBytes = cfg.cacheBytes ?? 64 * 2 ** 20;
    this.pairBudget = cfg.pairBudget ?? 6.4e7;
    this.targetWG = cfg.targetWorkgroups ?? 128;
    this.lostPromise = device.lost.then(() => {
      this.isLost = true;
    });
    this.registry =
      typeof FinalizationRegistry !== 'undefined' ? new FinalizationRegistry<CacheEntry>((e) => this.evict(e)) : null;
    this.out = new GrowBuffer(device, STORAGE | COPY_SRC, 'match-out');
    this.params = new GrowBuffer(device, UNIFORM | COPY_DST, 'match-params');
    this.aux = new GrowBuffer(device, STORAGE | COPY_DST, 'match-aux');
    this.aPerm = new GrowBuffer(device, STORAGE | COPY_DST, 'match-perm-a');
    this.bPerm = new GrowBuffer(device, STORAGE | COPY_DST, 'match-perm-b');
    this.staging = new Staging(device, 'match-staging', device.lost);
    this.stampBuf = device.createBuffer({ size: 16, usage: STORAGE | COPY_SRC, label: 'match-stamp' });
    this.stamp = new WorkStamp(device, pipes.stamp, this.stampBuf, 0);
    this.maxAttempts = Math.max(1, cfg.maxAttempts ?? 3);
    this.testDrop = cfg.testDropSubmissions ?? null;
  }

  /** Compiles the 5 matching pipelines up front. */
  static async create(ctx: GPUContext, cfg: MatcherGPUConfig = {}): Promise<MatcherGPU> {
    const device = ctx.device;
    const defs: [PipeKey, MatchKernelMode, boolean][] = [
      ['plain', 'plain', false],
      ['pos-f', 'pos', false],
      ['pos-r', 'pos', true],
      ['epi-f', 'epi', false],
      ['epi-r', 'epi', true],
      ['stamp', 'plain', false],
    ];
    const compiled = await Promise.all(
      defs.map(([k, m, r]) => compilePipeline(device, `match-${k}`, k === 'stamp' ? STAMP_WGSL : matchShader(m, r), device.lost)),
    );
    const pipes = {} as Record<PipeKey, GPUComputePipeline>;
    defs.forEach(([k], i) => {
      pipes[k] = compiled[i];
    });
    return new MatcherGPU(ctx, pipes, cfg);
  }

  /**
   * Brute-force matching with the semantics of matchCPU. Malformed FeatureSets (arrays
   * shorter than `count`) reject with a plain Error (input error, not a GPU failure).
   */
  match(a: FeatureSet, b: FeatureSet, opts?: MatchOptions): Promise<MatchPairs> {
    try {
      validateFeatureSet(a, 'a', false);
      validateFeatureSet(b, 'b', false);
    } catch (e) {
      return Promise.reject(e);
    }
    return this.exclusive(() => this.run(a, b, null, opts));
  }

  /** Guided matching with the semantics of matchGuidedCPU. */
  matchGuided(a: FeatureSet, b: FeatureSet, guide: MatchGuide, opts?: MatchOptions): Promise<MatchPairs> {
    let prep: GuidePrep;
    try {
      validateFeatureSet(a, 'a', true);
      validateFeatureSet(b, 'b', true);
      if (a.count === 0 || b.count === 0) return Promise.resolve(new Uint32Array(0));
      prep = prepareGuide(a, b, guide);
    } catch (e) {
      return Promise.reject(e);
    }
    return this.exclusive(() => this.run(a, b, prep, opts));
  }

  /** GPU memory of the descriptor cache (bytes). */
  cacheMemoryBytes(): number {
    return this.cacheUsed;
  }

  /**
   * Drops all cached descriptor buffers (deferred to the end of a running match, whose
   * remaining command buffers still use them).
   */
  clearCache(): void {
    if (this.busy) {
      this.clearPending = true;
      return;
    }
    this.clearPending = false;
    for (const e of [...this.lru]) this.evict(e);
  }

  /** Frees all GPU resources (after a running match ends); later calls reject. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    if (!this.busy) this.freeAll();
  }

  private freeAll(): void {
    if (this.freed) return;
    this.freed = true;
    this.clearPending = false;
    for (const e of [...this.lru]) this.evict(e);
    this.out.destroy();
    this.params.destroy();
    this.aux.destroy();
    this.aPerm.destroy();
    this.bPerm.destroy();
    this.staging.destroy();
    this.stampBuf.destroy();
    this.stamp.destroy();
  }

  /**
   * Compares the GPU matcher with matchCPU / matchGuidedCPU on deterministic random
   * descriptors (guards against GPU driver / compiler bugs). ok = >= 99 % of the CPU
   * pairs reproduced in both modes. Never throws.
   */
  async selfTest(): Promise<MatcherSelfTestResult> {
    const t0 = performance.now();
    try {
      const { a, b } = selfTestSets();
      const pred = new Float32Array(2 * a.count);
      for (let i = 0; i < a.count; i++) {
        pred[2 * i] = i % 9 === 0 ? NaN : a.keypoints[4 * i] + 2;
        pred[2 * i + 1] = a.keypoints[4 * i + 1] - 1;
      }
      const guide: MatchGuide = { kind: 'positions', predicted: pred, radiusPx: 60 };
      const o1 = pairOverlap(matchCPU(a, b), await this.match(a, b));
      const o2 = pairOverlap(matchGuidedCPU(a, b, guide), await this.matchGuided(a, b, guide));
      const overlap = Math.min(o1, o2);
      return { ok: overlap >= 0.99, overlap, ms: performance.now() - t0 };
    } catch (e) {
      return { ok: false, overlap: 0, ms: performance.now() - t0, error: (e as Error)?.message ?? String(e), transient: e instanceof GPUWorkDroppedError };
    }
  }

  // -------------------------------------------------------------------------

  private exclusive<T>(fn: () => Promise<T>): Promise<T> {
    const guarded = async (): Promise<T> => {
      this.busy = true;
      try {
        return await fn();
      } finally {
        this.busy = false;
        if (this.destroyed) this.freeAll();
        else if (this.clearPending) this.clearCache();
      }
    };
    const r = this.chain.then(guarded, guarded);
    this.chain = r.then(
      () => undefined,
      () => undefined,
    );
    return r;
  }

  private evict(e: CacheEntry): void {
    if (!e.alive) return;
    e.alive = false;
    this.registry?.unregister(e);
    e.desc.destroy();
    e.inv.destroy();
    this.cacheUsed -= e.bytes;
    this.lru.delete(e);
  }

  /** GPU buffers of a FeatureSet's descriptors, uploaded on first use or content change. */
  private entry(fs: FeatureSet, hash: number, T: MatcherGPUTimings): CacheEntry {
    let e = this.cache.get(fs);
    if (e && e.alive && e.count === fs.count && e.hash === hash) {
      this.lru.delete(e);
      this.lru.add(e);
      return e;
    }
    if (e) this.evict(e);
    const device = this.device;
    const n = fs.count;
    const descBytes = Math.max(16, n * DESCRIPTOR_SIZE);
    const invBytes = Math.max(16, Math.ceil((n * 4) / 16) * 16);
    const desc = device.createBuffer({ size: descBytes, usage: STORAGE | COPY_DST, label: 'match-desc' });
    device.queue.writeBuffer(desc, 0, packDescriptorWords(fs));
    const invNorm = descriptorInvNorms(fs);
    const inv32 = new Float32Array(n);
    for (let i = 0; i < n; i++) inv32[i] = invNorm[i];
    const inv = device.createBuffer({ size: invBytes, usage: STORAGE | COPY_DST, label: 'match-inv' });
    device.queue.writeBuffer(inv, 0, inv32);
    e = { count: n, hash, invNorm, desc, inv, bytes: descBytes + invBytes, alive: true };
    this.cache.set(fs, e);
    this.lru.add(e);
    this.cacheUsed += e.bytes;
    this.registry?.register(fs, e, e);
    T.uploadedSets++;
    return e;
  }

  /** Evicts least recently used entries (never those in `keep`) until within budget. */
  private trimCache(keep: CacheEntry[]): void {
    for (const e of [...this.lru]) {
      if (this.cacheUsed <= this.cacheBytes) break;
      if (keep.includes(e)) continue;
      this.evict(e);
    }
  }

  private async run(a: FeatureSet, b: FeatureSet, prep: GuidePrep | null, opts?: MatchOptions): Promise<MatchPairs> {
    const na = a.count;
    const nb = b.count;
    if (na === 0 || nb === 0) return new Uint32Array(0);
    if (this.destroyed) throw new GPUFeatureError('MatcherGPU: used after destroy()');
    for (let attempt = 0; ; attempt++) {
      if (this.isLost) throw new GPUFeatureError('MatcherGPU: GPU device lost (recreate via getGPU() + MatcherGPU.create)');
      const pop = errorScope(this.device, 'MatcherGPU');
      let result: MatchPairs | null = null;
      let failure: unknown = null;
      try {
        this.stamp.begin();
        result = await this.runGPU(a, b, prep, opts, attempt);
      } catch (e) {
        failure = e;
      }
      try {
        await pop();
      } catch (e) {
        // A validation / OOM error outranks the stamp mismatch it causes (see SiftGPU.run).
        if (!(failure instanceof FeatureInputError)) failure = e;
      }
      if (!failure) return result!;
      if (failure instanceof GPUWorkDroppedError && attempt + 1 < this.maxAttempts && !this.isLost) continue;
      if (failure instanceof FeatureInputError) throw failure;
      // Hard GPU failure: drop cached descriptor buffers (possibly invalid objects).
      if (!(failure instanceof GPUWorkDroppedError)) for (const e of [...this.lru]) this.evict(e);
      if (failure instanceof GPUFeatureError) throw failure;
      if (this.isLost) throw new GPUFeatureError('MatcherGPU: GPU device lost', { cause: failure });
      throw new GPUFeatureError(`MatcherGPU: ${(failure as Error)?.message ?? failure}`, { cause: failure });
    }
  }

  private async runGPU(a: FeatureSet, b: FeatureSet, prep: GuidePrep | null, opts: MatchOptions | undefined, attempt: number): Promise<MatchPairs> {
    const t0 = performance.now();
    const T: MatcherGPUTimings = { totalMs: 0, prepareMs: 0, gpuMs: 0, finalizeMs: 0, submits: 0, uploadedSets: 0, attempts: attempt + 1 };
    const device = this.device;
    const na = a.count;
    const nb = b.count;
    const mutual = opts?.mutual ?? true;
    const ea = this.entry(a, descriptorHash(a), T);
    const eb = a === b ? ea : this.entry(b, descriptorHash(b), T);
    this.trimCache([ea, eb]);

    const fwd = planChunks(na, nb, this.targetWG);
    const rev = mutual ? planChunks(nb, na, this.targetWG) : null;
    const fwdEntries = na * fwd.nChunks;
    const totalEntries = fwdEntries + (rev ? nb * rev.nChunks : 0);
    this.out.ensure(totalEntries * 16);
    if (prep) {
      this.aux.ensure(prep.aux.byteLength);
      device.queue.writeBuffer(this.aux.buffer!, 0, prep.aux);
      this.aPerm.ensure(na * 4);
      device.queue.writeBuffer(this.aPerm.buffer!, 0, prep.aPerm);
      this.bPerm.ensure(nb * 4);
      device.queue.writeBuffer(this.bPerm.buffer!, 0, prep.bPerm);
    }

    // Parameter slots: one per (direction, batch of row tiles).
    const pb = new ParamBlock(8);
    interface Pass {
      slot: number;
      x: number;
      y: number;
      reverse: boolean;
      cost: number;
    }
    const passes: Pass[] = [];
    const addPasses = (nRows: number, nCols: number, c: ReturnType<typeof planChunks>, outOff: number, reverse: boolean) => {
      const tilesPerBatch = Math.max(1, Math.floor(this.pairBudget / (MATCH_TILE * Math.max(1, nCols))));
      for (let t = 0; t < c.nRowTiles; t += tilesPerBatch) {
        const x = Math.min(tilesPerBatch, c.nRowTiles - t);
        const slot = pb.add(
          [nRows, nCols, t, c.tilesPerChunk, c.nColTiles, c.nChunks, outOff, prep?.aOff ?? 0, prep?.bOff ?? 0, prep?.boxOff ?? 0],
          [prep?.r2 ?? 0, prep?.radius ?? 0, prep?.tol ?? 0],
        );
        passes.push({ slot, x, y: c.nChunks, reverse, cost: x * MATCH_TILE * nCols });
      }
    };
    addPasses(na, nb, fwd, 0, false);
    if (rev) addPasses(nb, na, rev, fwdEntries, true);
    this.params.ensure(pb.byteLength);
    device.queue.writeBuffer(this.params.buffer!, 0, pb.bytes());
    T.prepareMs = performance.now() - t0;

    const t1 = performance.now();
    const drop = this.testDrop ? new Set(this.testDrop(attempt)) : null;
    const be = new BudgetEncoder(device, this.pairBudget, this.lostPromise, 'match', null, this.stamp, drop);
    for (const ps of passes) {
      const key: PipeKey = !prep ? 'plain' : `${prep.mode}-${ps.reverse ? 'r' : 'f'}`;
      const pipe = this.pipes[key];
      const rowE = ps.reverse ? eb : ea;
      const colE = ps.reverse ? ea : eb;
      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: this.params.buffer!, offset: ps.slot, size: PARAM_SLOT } },
        { binding: 1, resource: { buffer: rowE.desc } },
        { binding: 2, resource: { buffer: colE.desc } },
        { binding: 3, resource: { buffer: colE.inv } },
        { binding: 4, resource: { buffer: this.out.buffer! } },
      ];
      if (prep) {
        entries.push({ binding: 5, resource: { buffer: (ps.reverse ? this.bPerm : this.aPerm).buffer! } });
        entries.push({ binding: 6, resource: { buffer: (ps.reverse ? this.aPerm : this.bPerm).buffer! } });
        entries.push({ binding: 7, resource: { buffer: this.aux.buffer! } });
      }
      const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
      await be.dispatch(pipe, bg, ps.x, ps.y, 1, ps.cost);
    }
    const bytes = totalEntries * 16;
    this.staging.ensure(bytes + 16);
    be.encoder().copyBufferToBuffer(this.out.buffer!, 0, this.staging.buffer, 0, bytes);
    be.encoder().copyBufferToBuffer(this.stampBuf, 0, this.staging.buffer, bytes, 16);
    be.submit();
    const ab = await this.staging.read(bytes + 16);
    this.stamp.check(new Uint32Array(ab, bytes, 2), be.submits, 'match');
    const out = new Uint32Array(ab, 0, totalEntries * 4);
    T.submits = be.submits;
    T.gpuMs = performance.now() - t1;

    const t2 = performance.now();
    const bestJ = new Int32Array(na);
    const bestDot = new Float64Array(na);
    const secondDot = new Float64Array(na);
    reduceTop2(out, 0, na, fwd.nChunks, ea.invNorm, eb.invNorm, true, bestJ, bestDot, secondDot);
    let colBestI: Int32Array | null = null;
    if (rev) {
      colBestI = new Int32Array(nb);
      reduceTop2(out, 4 * fwdEntries, nb, rev.nChunks, eb.invNorm, ea.invNorm, false, colBestI, new Float64Array(nb), new Float64Array(nb));
    }
    const pairs = emitMatches(na, bestJ, bestDot, secondDot, colBestI, opts);
    T.finalizeMs = performance.now() - t2;
    T.totalMs = performance.now() - t0;
    this.lastTimings = T;
    return pairs;
  }
}

/** Deterministic random descriptor sets for the self-test (b = noisy permuted copies of a). */
function selfTestSets(): { a: FeatureSet; b: FeatureSet } {
  let seed = 0x9e3779b9;
  const rnd = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const D = DESCRIPTOR_SIZE;
  const mk = (n: number): FeatureSet => ({
    count: n,
    keypoints: new Float32Array(4 * n),
    descriptors: new Uint8Array(D * n),
    scores: new Float32Array(n),
    width: 640,
    height: 480,
  });
  const na = 150;
  const nb = 170;
  const a = mk(na);
  const b = mk(nb);
  const v = new Float64Array(D);
  const quant = (out: Uint8Array, o: number) => {
    let ss = 0;
    for (let k = 0; k < D; k++) ss += v[k] * v[k];
    const inv = 1 / Math.sqrt(ss);
    for (let k = 0; k < D; k++) out[o + k] = Math.min(255, Math.round(512 * v[k] * inv));
  };
  for (let i = 0; i < na; i++) {
    for (let k = 0; k < D; k++) v[k] = rnd() < 0.3 ? 0 : rnd();
    quant(a.descriptors, i * D);
    a.keypoints[4 * i] = rnd() * 640;
    a.keypoints[4 * i + 1] = rnd() * 480;
  }
  for (let j = 0; j < nb; j++) {
    const src = (j * 37) % na;
    if (j < 120) {
      for (let k = 0; k < D; k++) v[k] = Math.max(0, a.descriptors[src * D + k] / 512 + 0.1 * (rnd() - 0.5));
      b.keypoints[4 * j] = a.keypoints[4 * src] + 2 * rnd();
      b.keypoints[4 * j + 1] = a.keypoints[4 * src + 1] - 2 * rnd();
    } else {
      for (let k = 0; k < D; k++) v[k] = rnd() < 0.3 ? 0 : rnd();
      b.keypoints[4 * j] = rnd() * 640;
      b.keypoints[4 * j + 1] = rnd() * 480;
    }
    quant(b.descriptors, j * D);
  }
  return { a, b };
}

/** |reference ∩ test| / max(|reference|, |test|) over match pairs. */
function pairOverlap(ref: MatchPairs, test: MatchPairs): number {
  const s = new Set<number>();
  for (let k = 0; k < ref.length; k += 2) s.add(ref[k] * 4294967296 + ref[k + 1]);
  let common = 0;
  for (let k = 0; k < test.length; k += 2) if (s.has(test[k] * 4294967296 + test[k + 1])) common++;
  const n = Math.max(ref.length, test.length) / 2;
  return n ? common / n : 1;
}
