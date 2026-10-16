/**
 * Generic LO-MSAC: RANSAC with MSAC scoring (Torr & Zisserman 2000, "MLESAC: A new
 * robust estimator…", truncated-quadratic cost Σ min(r², T²)) and local
 * optimisation of every so-far-the-best model (Chum, Matas & Kittler 2003,
 * "Locally optimized RANSAC"; iterative least squares with a shrinking threshold
 * as in Lebeda, Matas & Chum 2012, "Fixing the locally optimized RANSAC").
 *
 * Deterministic: all randomness comes from the supplied (or a fixed-seed) `Rng`.
 */
import { Rng } from '../math/random';

/** Problem definition consumed by `ransac`. */
export interface RansacEstimator<M> {
  /** Minimal sample size. */
  sampleSize: number;
  /** Number of data points. */
  numData: number;
  /**
   * Minimal solver: fills `models[0..k)` with FRESH model objects (RANSAC keeps
   * references to the best one) and returns k (0 if the sample yields no model).
   */
  fit(sample: Uint32Array, models: M[]): number;
  /** Squared residual of every datum under `model` (out has length numData). */
  residualsSq(model: M, out: Float64Array): void;
  /**
   * Extension: squared residuals of data [start, end) only (out[start..end)). When
   * present, hypotheses are scored in blocks and abandoned as soon as their partial
   * MSAC cost exceeds the best cost (an exact bail-out: results are unchanged).
   */
  residualsSqRange?(model: M, out: Float64Array, start: number, end: number): void;
  /** Least-squares fit on the given inliers (local optimisation / final refit). */
  refit?(inliers: Uint32Array, count: number): M | null;
  /** True if the minimal sample is degenerate (skipped without fitting). */
  isDegenerate?(sample: Uint32Array): boolean;
  /**
   * Extension: refinement STARTING from `model` (e.g. nonlinear least squares). When
   * present it is used for local optimisation instead of `refit`.
   */
  refine?(model: M, inliers: Uint32Array, count: number): M | null;
  /** Extension: minimum inlier count for refit/refine (default: sampleSize). */
  minRefitSize?: number;
}

/** RANSAC options. `M` only matters for the `initialModels` extension. */
export interface RansacOptions<M = unknown> {
  /** Squared inlier threshold (same units as `residualsSq`). */
  thresholdSq: number;
  /** Probability of having drawn at least one all-inlier sample (default 0.9999). */
  confidence?: number;
  /** Maximum number of samples drawn (default 10000). */
  maxIterations?: number;
  /** Minimum number of samples drawn (default 0). */
  minIterations?: number;
  /** Random source (default: a fixed-seed Rng, so results are reproducible). */
  rng?: Rng;
  /** Local optimisation of each new best model (default true; needs refit or refine). */
  localOptimization?: boolean;
  /**
   * Extension: a-priori minimum inlier ratio. Caps the number of iterations at the
   * count that finds a model with this inlier ratio with the given confidence
   * (COLMAP's `min_inlier_ratio`). Default 0 = no cap besides maxIterations.
   */
  minInlierRatio?: number;
  /** Extension: hypotheses scored (and locally optimised) before sampling starts. */
  initialModels?: M[];
  /** Extension: number of local-optimisation steps (default 4). */
  loIterations?: number;
  /**
   * Extension: the first LO step uses inliers within loThresholdMultiplier·T, shrinking
   * linearly to T in the last step (default 2).
   */
  loThresholdMultiplier?: number;
}

/** Result of `ransac`. */
export interface RansacResult<M> {
  model: M | null;
  /** 1 for inliers (residual² ≤ thresholdSq) of the returned model. */
  inlierMask: Uint8Array;
  numInliers: number;
  /** Number of samples drawn (including degenerate ones). */
  iterations: number;
  /** MSAC cost Σ min(r², T²) of the returned model (Infinity if none). */
  score: number;
}

const DEFAULT_SEED = 0x5eed1234;

/**
 * Number of samples needed to draw an all-inlier minimal sample with probability
 * `confidence` when the inlier ratio is `inlierRatio`: log(1 − p) / log(1 − wˢ).
 */
export function ransacIterations(inlierRatio: number, sampleSize: number, confidence: number, maxIterations: number): number {
  if (!(inlierRatio > 0)) return maxIterations;
  const ws = Math.pow(Math.min(1, inlierRatio), sampleSize);
  if (ws >= 1) return 0;
  const denom = Math.log1p(-ws);
  if (!(denom < 0)) return maxIterations;
  const k = Math.log(Math.max(1e-300, 1 - confidence)) / denom;
  if (!Number.isFinite(k)) return maxIterations;
  return Math.min(maxIterations, Math.max(0, Math.ceil(k)));
}

/** Indices of the set entries of a mask. */
export function maskToIndices(mask: Uint8Array): Uint32Array {
  let c = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) c++;
  const out = new Uint32Array(c);
  c = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) out[c++] = i;
  return out;
}

/**
 * LO-MSAC. For every model of every minimal sample the MSAC cost is computed; a new
 * best model is locally optimised (iterative refits on its inliers with a threshold
 * shrinking from loThresholdMultiplier·T to T, each accepted only if it lowers the
 * cost). The number of iterations adapts to the best inlier ratio. A final
 * local-optimisation pass at threshold T is run on the winner.
 */
export function ransac<M>(est: RansacEstimator<M>, opts: RansacOptions<M>): RansacResult<M> {
  const N = est.numData;
  const s = est.sampleSize;
  const mask = new Uint8Array(N);
  if (!(N >= s) || s <= 0 || !(opts.thresholdSq >= 0)) {
    return { model: null, inlierMask: mask, numInliers: 0, iterations: 0, score: Infinity };
  }
  const T2 = opts.thresholdSq;
  const confidence = opts.confidence ?? 0.9999;
  const maxIterationsOpt = Math.max(0, Math.floor(opts.maxIterations ?? 10000));
  const minIterations = Math.max(0, Math.floor(opts.minIterations ?? 0));
  const rng = opts.rng ?? new Rng(DEFAULT_SEED);
  const canLO = (opts.localOptimization ?? true) && (est.refit !== undefined || est.refine !== undefined);
  const loIterations = Math.max(1, Math.floor(opts.loIterations ?? 4));
  const loMult = Math.max(1, opts.loThresholdMultiplier ?? 2);
  const minRefit = Math.max(1, est.minRefitSize ?? s);
  let maxIterations = maxIterationsOpt;
  if (opts.minInlierRatio !== undefined && opts.minInlierRatio > 0) {
    maxIterations = Math.min(maxIterations, ransacIterations(opts.minInlierRatio, s, confidence, maxIterationsOpt));
  }
  maxIterations = Math.max(maxIterations, minIterations);

  let bestRes = new Float64Array(N);
  let curRes = new Float64Array(N);
  let loRes = new Float64Array(N);
  const inl = new Uint32Array(N);
  const sample = new Uint32Array(s);
  const models: M[] = [];

  let best = null as M | null;
  let bestScore = Infinity;
  let bestCount = 0;
  let lastCount = 0;

  // MSAC score of the residuals in `r`; sets lastCount.
  const score = (r: Float64Array): number => {
    let sc = 0;
    let c = 0;
    for (let i = 0; i < N; i++) {
      const v = r[i];
      if (v <= T2) {
        sc += v;
        c++;
      } else sc += T2; // also NaN / Infinity → T²
    }
    lastCount = c;
    return sc;
  };

  // Local optimisation of the current best (residuals in bestRes).
  const localOptimize = (finalPass: boolean): void => {
    const iters = finalPass ? 2 : loIterations;
    for (let k = 0; k < iters; k++) {
      const m = finalPass ? 1 : loMult - ((loMult - 1) * k) / Math.max(1, iters - 1);
      const thr = T2 * m * m;
      let c = 0;
      for (let i = 0; i < N; i++) if (bestRes[i] <= thr) inl[c++] = i;
      if (c < minRefit) break;
      const cand = est.refine !== undefined && best !== null ? est.refine(best, inl, c) : est.refit!(inl, c);
      if (cand === null) continue;
      est.residualsSq(cand, loRes);
      const sc = score(loRes);
      if (sc < bestScore) {
        best = cand;
        bestScore = sc;
        bestCount = lastCount;
        const tmp = bestRes;
        bestRes = loRes;
        loRes = tmp;
      } else if (finalPass) break;
    }
  };

  const BLOCK = 128;
  // Full residuals of m into curRes and its MSAC score, or Infinity as soon as the
  // partial score exceeds the best (sets lastCount when complete).
  const scoreBounded = (m: M): number => {
    if (est.residualsSqRange === undefined || bestScore === Infinity) {
      est.residualsSq(m, curRes);
      return score(curRes);
    }
    let sc = 0;
    let c = 0;
    for (let start = 0; start < N; start += BLOCK) {
      const end = Math.min(N, start + BLOCK);
      est.residualsSqRange(m, curRes, start, end);
      for (let i = start; i < end; i++) {
        const v = curRes[i];
        if (v <= T2) {
          sc += v;
          c++;
        } else sc += T2;
      }
      if (sc >= bestScore) return Infinity;
    }
    lastCount = c;
    return sc;
  };

  const consider = (m: M): boolean => {
    const sc = scoreBounded(m);
    if (sc < bestScore) {
      best = m;
      bestScore = sc;
      bestCount = lastCount;
      const tmp = bestRes;
      bestRes = curRes;
      curRes = tmp;
      if (canLO) localOptimize(false);
      return true;
    }
    return false;
  };

  let dynIterations = maxIterations;
  const updateIterations = (): void => {
    dynIterations = Math.max(minIterations, ransacIterations(bestCount / N, s, confidence, maxIterations));
  };

  if (opts.initialModels) {
    for (const m of opts.initialModels) if (consider(m)) updateIterations();
  }

  let it = 0;
  for (; it < dynIterations; it++) {
    rng.sample(N, s, sample);
    if (est.isDegenerate !== undefined && est.isDegenerate(sample)) continue;
    const k = est.fit(sample, models);
    let improved = false;
    for (let j = 0; j < k; j++) if (consider(models[j])) improved = true;
    if (improved) updateIterations();
  }

  if (best !== null && canLO) localOptimize(true);

  let numInliers = 0;
  if (best !== null) {
    for (let i = 0; i < N; i++) {
      if (bestRes[i] <= T2) {
        mask[i] = 1;
        numInliers++;
      }
    }
  }
  return { model: best, inlierMask: mask, numInliers, iterations: it, score: bestScore };
}
