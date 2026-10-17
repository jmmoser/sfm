/**
 * Live (during capture) incremental mapping of time-ordered keyframes on top of the
 * COLMAP-style `IncrementalMapper`:
 *  - initialisation from the best pair among recent keyframes once the two-view
 *    geometry has enough inliers and triangulation angle (baseline);
 *  - registration of every new keyframe by PnP against the current model,
 *    triangulation, time-bounded local BA;
 *  - keyframes that fail to register stay pending and are retried (relocalisation)
 *    when more of their features see triangulated points;
 *  - periodic time-bounded global BA with self-calibration (more often early on).
 *
 * The caller (engine) adds each keyframe to the FeatureDatabase and its verified
 * pairs to the CorrespondenceGraph BEFORE calling `addKeyframe`.
 */
import type { Pose } from '../core/types';
import type { CorrespondenceGraph } from './correspondence-graph';
import { pairIds, pairKey, type FeatureDatabase } from './database';
import { IncrementalMapper, resolveMapperOptions, type InitPairGeometry, type MapperOptions } from './mapper';
import { Reconstruction } from './reconstruction';

export interface LiveMapperOptions extends MapperOptions {
  /** Initial pairs are searched among the last `initWindow` keyframes (default 30). */
  initWindow: number;
  /** Minimum inliers of the initial pair (default 60). */
  liveInitMinNumInliers: number;
  /** Minimum median triangulation angle (deg) of the initial pair (default 5). */
  liveInitMinTriAngleDeg: number;
  /** Two-view re-estimations per `addKeyframe` while not initialised (default 6). */
  maxInitEvaluationsPerKeyframe: number;
  /**
   * With `initFallback`: after this many keyframes without initialisation the median
   * angle threshold drops to max(initFallbackMinTriAngleDeg, ½ liveInitMinTriAngleDeg)
   * and forward motion up to initFallbackMaxForwardMotion is accepted; after twice as
   * many the angle threshold is initFallbackMinTriAngleDeg (default 10; 0 = never).
   */
  liveInitFallbackAfter: number;
  /** Minimum number of points of the initial model (default 50). */
  minInitPoints: number;
  /** Time budget (ms) of the local BA after each registration (default 120). */
  localBATimeMs: number;
  /** Global BA after this many registrations (default 15). */
  globalBAEvery: number;
  /** Time budget (ms) of a periodic global BA (default 800). */
  globalBATimeMs: number;
  /** Pending keyframes retried after a successful registration (default 4). */
  retryPendingPerKeyframe: number;
}

export const DEFAULT_LIVE_OPTIONS: Readonly<
  Pick<
    LiveMapperOptions,
    | 'initWindow'
    | 'liveInitMinNumInliers'
    | 'liveInitMinTriAngleDeg'
    | 'maxInitEvaluationsPerKeyframe'
    | 'liveInitFallbackAfter'
    | 'minInitPoints'
    | 'localBATimeMs'
    | 'globalBAEvery'
    | 'globalBATimeMs'
    | 'retryPendingPerKeyframe'
  >
> = Object.freeze({
  initWindow: 30,
  liveInitMinNumInliers: 60,
  liveInitMinTriAngleDeg: 5,
  maxInitEvaluationsPerKeyframe: 6,
  liveInitFallbackAfter: 10,
  minInitPoints: 50,
  localBATimeMs: 120,
  globalBAEvery: 15,
  globalBATimeMs: 800,
  retryPendingPerKeyframe: 4,
});

export interface LiveKeyframeResult {
  registered: boolean;
  initialized: boolean;
  numPoints: number;
  /** Mean reprojection error (px) of the model (0 before initialisation). */
  reprojError: number;
}

export interface LiveSnapshot {
  /** 3 floats per point (world). */
  points: Float32Array;
  /** 3 bytes per point (sRGB). */
  colors: Uint8Array;
  poses: { id: number; pose: Pose }[];
}

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());

export class LiveMapper {
  readonly opts: LiveMapperOptions;
  private rec: Reconstruction;
  private mapper: IncrementalMapper;
  private readonly keyframes: number[] = [];
  /** Unregistered keyframes → visible-point count at the last attempt. */
  private readonly pending = new Map<number, number>();
  /** Re-estimated geometry of every pair evaluated for initialisation (pairKey → geometry / null). */
  private readonly initCache = new Map<number, InitPairGeometry | null>();
  /** Pairs an initialisation was attempted from (never retried). */
  private readonly initAttempted = new Set<number>();
  private isInit = false;
  private regSinceGlobal = 0;
  private numRegAtGlobal = 0;
  /** Accumulated timings (ms) for diagnostics. */
  readonly timings = { init: 0, register: 0, localBA: 0, globalBA: 0, total: 0, keyframes: 0 };

  constructor(
    readonly db: FeatureDatabase,
    readonly graph: CorrespondenceGraph,
    opts: Partial<LiveMapperOptions> = {},
  ) {
    const base = resolveMapperOptions(opts);
    this.opts = { ...base, ...DEFAULT_LIVE_OPTIONS, ...opts, triangulation: base.triangulation } as LiveMapperOptions;
    this.rec = new Reconstruction();
    this.mapper = new IncrementalMapper(db, graph, this.rec, this.opts);
  }

  get reconstruction(): Reconstruction {
    return this.rec;
  }

  get initialized(): boolean {
    return this.isInit;
  }

  /** The underlying incremental mapper (e.g. for a final refinement). */
  get incrementalMapper(): IncrementalMapper {
    return this.mapper;
  }

  /** Keyframes not (yet) registered. */
  pendingIds(): number[] {
    return [...this.pending.keys()];
  }

  /** Processes a new keyframe (already in the database and the graph). */
  addKeyframe(id: number): LiveKeyframeResult {
    const t0 = now();
    if (!this.db.frames.has(id)) throw new RangeError(`LiveMapper.addKeyframe: frame ${id} not in the database`);
    if (!this.graph.hasImage(id)) this.graph.addImage(id, this.db.features.get(id)!.count);
    const known = this.rec.images.has(id);
    this.mapper.ensureImage(id);
    if (!known) this.keyframes.push(id);
    const img = this.rec.images.get(id)!;
    if (!img.registered) {
      if (!this.isInit) {
        this.pending.set(id, -1);
        this.tryInitialize(id);
      } else if (this.tryRegister(id)) {
        this.retryPending(id);
      } else this.pending.set(id, img.numVisiblePoints3D);
    }
    this.maybeGlobalBA();
    this.timings.total += now() - t0;
    this.timings.keyframes++;
    return {
      registered: this.rec.images.get(id)!.registered,
      initialized: this.isInit,
      numPoints: this.rec.numPoints,
      reprojError: this.isInit ? this.rec.meanReprojectionError() : 0,
    };
  }

  private tryRegister(id: number): boolean {
    const t0 = now();
    this.rec.syncVisibility(id);
    this.mapper.resetRegTrials(id);
    const ok = this.mapper.registerNextImage(id);
    this.timings.register += now() - t0;
    if (!ok) return false;
    this.pending.delete(id);
    const t1 = now();
    this.mapper.triangulateImage(id);
    this.mapper.iterativeLocalRefinement(id, this.opts.localBATimeMs);
    this.timings.localBA += now() - t1;
    this.regSinceGlobal++;
    return true;
  }

  /** Retries pending keyframes whose visible-point count increased (most recent first). */
  private retryPending(justRegistered: number): void {
    if (this.pending.size === 0) return;
    // nearest keyframes in capture order first
    const pos = this.keyframes.indexOf(justRegistered);
    const cands = [...this.pending.keys()].sort(
      (a, b) => Math.abs(this.keyframes.indexOf(a) - pos) - Math.abs(this.keyframes.indexOf(b) - pos),
    );
    let tries = 0;
    for (const id of cands) {
      if (tries >= this.opts.retryPendingPerKeyframe) break;
      const img = this.rec.images.get(id)!;
      this.rec.syncVisibility(id);
      const last = this.pending.get(id) ?? -1;
      if (img.numVisiblePoints3D < this.opts.absPoseMinNumInliers || img.numVisiblePoints3D <= last) continue;
      tries++;
      if (!this.tryRegister(id)) this.pending.set(id, img.numVisiblePoints3D);
    }
  }

  private tryInitialize(newId: number): void {
    const t0 = now();
    const o = this.opts;
    const window = this.keyframes.slice(Math.max(0, this.keyframes.length - o.initWindow));
    // thresholds: strict, then relaxed (fallback) while initialisation keeps failing
    const stage =
      o.initFallback && o.liveInitFallbackAfter > 0 ? Math.min(2, Math.floor(this.keyframes.length / o.liveInitFallbackAfter)) : 0;
    const minAngleDeg =
      stage === 0
        ? o.liveInitMinTriAngleDeg
        : Math.min(o.liveInitMinTriAngleDeg, stage === 1 ? Math.max(o.initFallbackMinTriAngleDeg, 0.5 * o.liveInitMinTriAngleDeg) : o.initFallbackMinTriAngleDeg);
    const maxForward = stage === 0 ? o.initMaxForwardMotion : Math.max(o.initMaxForwardMotion, o.initFallbackMaxForwardMotion);
    const minAngle = (minAngleDeg * Math.PI) / 180;
    const accept = (g: InitPairGeometry | null | undefined): g is InitPairGeometry =>
      !!g && g.numInliers >= o.liveInitMinNumInliers && g.triAngle >= minAngle && g.forwardMotion < maxForward;
    const attempt = (key: number, a: number, b: number, g: InitPairGeometry): boolean => {
      this.initAttempted.add(key);
      // external rotation check (gyro) of the initial pair, oriented a → b
      if (o.rotationCheck?.pair && !o.rotationCheck.pair(a, b, g.R)) return false;
      return this.initializeFrom(a, b, g);
    };
    // 1. relaxed thresholds: pairs evaluated earlier may pass now (cached, free), largest angle first
    if (stage > 0) {
      const inWindow = new Set(window);
      const ok: [number, InitPairGeometry][] = [];
      for (const [key, g] of this.initCache) {
        if (this.initAttempted.has(key) || !accept(g)) continue;
        const [a, b] = pairIds(key);
        if (inWindow.has(a) && inWindow.has(b)) ok.push([key, g]);
      }
      ok.sort((x, y) => y[1].triAngle - x[1].triAngle || x[0] - y[0]);
      for (const [key, g] of ok) {
        const [a, b] = pairIds(key);
        if (attempt(key, a, b, g)) break;
      }
    }
    // 2. candidate pairs with the new keyframe, plus unevaluated pairs within the window
    const cands: { a: number; b: number; n: number; ok: boolean }[] = [];
    const consider = (a: number, b: number): void => {
      const key = pairKey(a, b);
      if (this.initCache.has(key)) return;
      const n = this.graph.numCorrespondencesBetween(a, b);
      if (n < o.liveInitMinNumInliers) return;
      const st = this.db.getTwoView(a, b);
      if (st && st.config === 'panoramic') return;
      const sa = st?.triAngle;
      if (sa !== undefined && sa < 0.5 * minAngle) return;
      cands.push({ a, b, n, ok: sa === undefined || sa >= minAngle });
    };
    if (!this.isInit) {
      for (const k of window) if (k !== newId) consider(Math.min(k, newId), Math.max(k, newId));
      if (cands.length === 0)
        for (let i = 0; i < window.length; i++)
          for (let j = i + 1; j < window.length; j++) consider(Math.min(window[i], window[j]), Math.max(window[i], window[j]));
    }
    cands.sort((x, y) => Number(y.ok) - Number(x.ok) || y.n - x.n);
    let evals = 0;
    for (const c of cands) {
      if (evals >= o.maxInitEvaluationsPerKeyframe) break;
      evals++;
      const key = pairKey(c.a, c.b);
      const g = this.mapper.estimateInitPair(c.a, c.b);
      this.initCache.set(key, g);
      if (!accept(g)) continue;
      if (attempt(key, c.a, c.b, g)) break;
    }
    this.timings.init += now() - t0;
    if (this.isInit) {
      // register the other pending keyframes, nearest to the initial pair first
      const ids = [...this.pending.keys()];
      const pa = this.keyframes.indexOf(newId);
      ids.sort((x, y) => Math.abs(this.keyframes.indexOf(x) - pa) - Math.abs(this.keyframes.indexOf(y) - pa));
      for (let pass = 0; pass < 2; pass++) {
        let any = false;
        for (const id of ids) {
          if (this.rec.images.get(id)!.registered) continue;
          if (this.tryRegister(id)) any = true;
          else this.pending.set(id, this.rec.images.get(id)!.numVisiblePoints3D);
        }
        if (!any) break;
      }
    }
  }

  private initializeFrom(a: number, b: number, g: InitPairGeometry): boolean {
    const o = this.opts;
    const minPointAngle = Math.max(o.filterMinTriAngleDeg, 0.25 * o.liveInitMinTriAngleDeg);
    this.mapper.registerInitialPair(a, b, g, minPointAngle);
    this.mapper.adjustGlobalBundle(o.globalBATimeMs);
    this.mapper.filterPoints();
    this.mapper.triangulator.clearModifiedPoints();
    if (this.rec.numPoints < o.minInitPoints || this.rec.numRegisteredImages < 2) {
      for (const id of this.rec.registeredImageIds()) this.mapper.deregisterImage(id);
      for (const pid of [...this.rec.points.keys()]) this.rec.deletePoint(pid);
      return false;
    }
    this.isInit = true;
    this.pending.delete(a);
    this.pending.delete(b);
    this.numRegAtGlobal = 2;
    this.regSinceGlobal = 0;
    return true;
  }

  private maybeGlobalBA(): void {
    if (!this.isInit) return;
    const n = this.rec.numRegisteredImages;
    const early = n <= 30 && n >= Math.max(this.opts.minImagesForIntrinsics, Math.ceil(1.3 * this.numRegAtGlobal));
    if (!early && this.regSinceGlobal < this.opts.globalBAEvery) return;
    const t0 = now();
    this.mapper.iterativeGlobalRefinementSync(this.opts.globalBATimeMs, false);
    this.syncPending();
    this.timings.globalBA += now() - t0;
    this.numRegAtGlobal = this.rec.numRegisteredImages;
    this.regSinceGlobal = 0;
  }

  /** After a global refinement: images deregistered by its image filter become pending again. */
  private syncPending(): void {
    for (const id of this.rec.registeredImageIds()) this.pending.delete(id);
    for (const id of this.keyframes) if (!this.rec.images.get(id)!.registered && !this.pending.has(id)) this.pending.set(id, -1);
  }

  /** Full (not time-bounded) global refinement, e.g. at the end of capture. */
  refine(timeLimitMs?: number): void {
    if (!this.isInit) return;
    this.mapper.iterativeGlobalRefinementSync(timeLimitMs);
    this.syncPending();
    this.numRegAtGlobal = this.rec.numRegisteredImages;
    this.regSinceGlobal = 0;
  }

  /** Sparse points, colours (from the per-feature colours) and registered poses (copies). */
  snapshot(): LiveSnapshot {
    const n = this.rec.points.size;
    const points = new Float32Array(3 * n);
    const colors = new Uint8Array(3 * n);
    let k = 0;
    for (const p of this.rec.points.values()) {
      points[3 * k] = p.xyz[0];
      points[3 * k + 1] = p.xyz[1];
      points[3 * k + 2] = p.xyz[2];
      colors[3 * k] = p.color[0];
      colors[3 * k + 1] = p.color[1];
      colors[3 * k + 2] = p.color[2];
      k++;
    }
    const poses = this.rec.registeredImageIds().map((id) => {
      const pose = this.rec.images.get(id)!.pose;
      return { id, pose: { R: Float64Array.from(pose.R), t: Float64Array.from(pose.t) } };
    });
    return { points, colors, poses };
  }
}
