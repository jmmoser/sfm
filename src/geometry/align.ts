/**
 * Robust similarity alignment dst ≈ s R src + t between 3-D point sets: LO-MSAC
 * with 3-point Umeyama minimal samples and weighted Umeyama refits on the inliers.
 */
import type { Mat3, Vec3 } from '../core/types';
import { Rng } from '../math/random';
import { umeyama, type Sim3 } from '../math/sim3';
import { ransac, type RansacEstimator } from './ransac';

/** Options of `alignSim3Ransac`. */
export interface AlignSim3Options {
  /** Per-point weights for the refits (≥ 0), e.g. 1/σ² of GPS fixes. */
  weights?: Float64Array;
  rng?: Rng;
  /** Minimum inlier count for success (default 3). */
  minInliers?: number;
  /** RANSAC confidence (default 0.9999). */
  confidence?: number;
  /** RANSAC iteration cap (default 10000). */
  maxIterations?: number;
  /** Estimate a rigid transform (s = 1) instead (default false). */
  fixedScale?: boolean;
}

class Sim3Estimator implements RansacEstimator<Sim3> {
  sampleSize = 3;
  numData: number;
  minRefitSize = 3;
  private s = new Float64Array(9);
  private d = new Float64Array(9);
  private gs: Float64Array;
  private gd: Float64Array;
  private gw: Float64Array | undefined;
  constructor(
    private src: Float64Array,
    private dst: Float64Array,
    n: number,
    private weights: Float64Array | undefined,
    private withScale: boolean,
  ) {
    this.numData = n;
    this.gs = new Float64Array(3 * n);
    this.gd = new Float64Array(3 * n);
    this.gw = weights ? new Float64Array(n) : undefined;
  }
  fit(sample: Uint32Array, models: Sim3[]): number {
    for (let k = 0; k < 3; k++) {
      const i = sample[k];
      for (let c = 0; c < 3; c++) {
        this.s[3 * k + c] = this.src[3 * i + c];
        this.d[3 * k + c] = this.dst[3 * i + c];
      }
    }
    const m = umeyama(this.s, this.d, 3, this.withScale);
    if (!m) return 0;
    models[0] = m;
    return 1;
  }
  residualsSq(m: Sim3, out: Float64Array): void {
    this.residualsSqRange(m, out, 0, this.numData);
  }
  residualsSqRange(m: Sim3, out: Float64Array, start: number, end: number): void {
    const R = m.R,
      s = m.s,
      t = m.t;
    const r0 = s * R[0], r1 = s * R[1], r2 = s * R[2], r3 = s * R[3], r4 = s * R[4], r5 = s * R[5], r6 = s * R[6], r7 = s * R[7], r8 = s * R[8];
    const src = this.src,
      dst = this.dst;
    for (let i = start; i < end; i++) {
      const x = src[3 * i],
        y = src[3 * i + 1],
        z = src[3 * i + 2];
      const dx = r0 * x + r1 * y + r2 * z + t[0] - dst[3 * i];
      const dy = r3 * x + r4 * y + r5 * z + t[1] - dst[3 * i + 1];
      const dz = r6 * x + r7 * y + r8 * z + t[2] - dst[3 * i + 2];
      out[i] = dx * dx + dy * dy + dz * dz;
    }
  }
  refit(inl: Uint32Array, count: number): Sim3 | null {
    for (let k = 0; k < count; k++) {
      const i = inl[k];
      for (let c = 0; c < 3; c++) {
        this.gs[3 * k + c] = this.src[3 * i + c];
        this.gd[3 * k + c] = this.dst[3 * i + c];
      }
      if (this.gw && this.weights) this.gw[k] = this.weights[i];
    }
    const gs = this.gs.subarray(0, 3 * count),
      gd = this.gd.subarray(0, 3 * count);
    return umeyama(gs, gd, count, this.withScale, this.gw ? this.gw.subarray(0, count) : undefined);
  }
}

/**
 * Robust Sim3 (dst ≈ s R src + t) with inlier threshold `maxError` on the 3-D
 * residual (dst units). Returns null if no model with ≥ minInliers inliers exists.
 */
export function alignSim3Ransac(
  src: Float64Array,
  dst: Float64Array,
  n: number,
  maxError: number,
  opts: AlignSim3Options = {},
): { s: number; R: Mat3; t: Vec3; inlierMask: Uint8Array; numInliers: number } | null {
  const minInliers = Math.max(3, opts.minInliers ?? 3);
  if (n < minInliers || !(maxError > 0)) return null;
  const est = new Sim3Estimator(src, dst, n, opts.weights, !(opts.fixedScale ?? false));
  const res = ransac(est, {
    thresholdSq: maxError * maxError,
    confidence: opts.confidence ?? 0.9999,
    maxIterations: opts.maxIterations ?? 10000,
    rng: opts.rng ?? new Rng(0x51a3),
  });
  if (!res.model || res.numInliers < minInliers) return null;
  return { s: res.model.s, R: res.model.R, t: res.model.t, inlierMask: res.inlierMask, numInliers: res.numInliers };
}
