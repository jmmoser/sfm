/**
 * 3DGS-MCMC densification (Kheradmand et al., NeurIPS 2024), host side: the training state is
 * a fixed-capacity set of Gaussians; every `refineEvery` steps the "dead" ones
 * (opacity ≤ minOpacity) are moved onto live Gaussians sampled ∝ opacity, and the count grows
 * by `growth` (until the cap) the same way. A Gaussian sampled n times is split into N = n + 1
 * copies that keep the rendered contribution: o' = 1 − (1 − o)^(1/N) and the scales are
 * multiplied by o / Σ_{i=1..N} Σ_{k=0..i−1} C(i−1, k) (−1)^k o'^(k+1) / √(k+1) (eq. 9 of the
 * paper). The GPU applies the plan (wgsl/backward.ts `relocateWGSL`): copy entries
 * (dst ← src, new opacity, log-scale delta), then the sources' own updates; Adam moments of
 * both are reset. Implemented from the paper (the reference code has a non-commercial licence).
 * Chosen over 3DGS clone / split / prune for its fixed memory budget (the cap is planned up front)
 * and its robustness to the MVS initialisation.
 */
import { Rng } from '../math/random';
import { logit, sigmoid } from './types';

export interface McmcPlan {
  /** Entries [dst, src, opacity-logit bits, log-scale-delta bits] (dst ← copy of src). */
  copies: Uint32Array;
  /** Entries [index, 0, opacity-logit bits, log-scale-delta bits] for each sampled source. */
  updates: Uint32Array;
  nCopies: number;
  nUpdates: number;
  /** Gaussian count after the step. */
  count: number;
  dead: number;
  added: number;
}

export interface McmcPlanOptions {
  capacity: number;
  /** Growth factor per refinement step (default 0.05). */
  growth: number;
  /** Opacity at or below which a Gaussian is dead (default 0.005). */
  minOpacity: number;
  /** Largest number of copies of one source per step (default 51). */
  maxCopies?: number;
}

const binomCache: number[][] = [];
function binom(n: number, k: number): number {
  while (binomCache.length <= n) {
    const m = binomCache.length;
    const row = new Array<number>(m + 1).fill(1);
    for (let j = 1; j < m; j++) row[j] = binomCache[m - 1][j - 1] + binomCache[m - 1][j];
    binomCache.push(row);
  }
  return binomCache[n][k];
}

/** New opacity and scale factor of a Gaussian (opacity o) split into n copies (paper eq. 9). */
export function relocation(o: number, n: number, minOpacity = 0.005): { opacity: number; scale: number } {
  if (n <= 1) return { opacity: o, scale: 1 };
  const on = Math.min(1 - 1e-6, Math.max(minOpacity, 1 - Math.pow(1 - o, 1 / n)));
  let denom = 0;
  for (let i = 1; i <= n; i++) {
    for (let k = 0; k < i; k++) denom += (binom(i - 1, k) * (k % 2 === 0 ? 1 : -1) * Math.pow(on, k + 1)) / Math.sqrt(k + 1);
  }
  const scale = denom > 0 ? o / denom : 1;
  return { opacity: on, scale: Number.isFinite(scale) && scale > 0 ? scale : 1 };
}

/** Alias-free multinomial sampling ∝ w (with replacement) via binary search on the CDF. */
function sampleProportional(w: Float64Array, idx: Uint32Array, k: number, rng: Rng): Uint32Array {
  const cdf = new Float64Array(w.length);
  let s = 0;
  for (let i = 0; i < w.length; i++) cdf[i] = s += w[i];
  const out = new Uint32Array(k);
  for (let j = 0; j < k; j++) {
    const u = rng.next() * s;
    let lo = 0;
    let hi = w.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid] < u) lo = mid + 1;
      else hi = mid;
    }
    out[j] = idx[lo];
  }
  return out;
}

/** Relocation + growth plan from the current opacity logits of the first `count` Gaussians. */
export function planMcmc(opacityLogits: Float32Array, count: number, o: McmcPlanOptions, rng: Rng): McmcPlan {
  const op = new Float64Array(count);
  const dead: number[] = [];
  const alive: number[] = [];
  for (let i = 0; i < count; i++) {
    const v = sigmoid(opacityLogits[i]);
    op[i] = Number.isFinite(v) ? v : 0;
    if (op[i] <= o.minOpacity) dead.push(i);
    else alive.push(i);
  }
  const target = Math.min(o.capacity, Math.max(count, Math.floor(count * (1 + o.growth))));
  const nAdd = target - count;
  const nSamples = alive.length ? dead.length + nAdd : 0;
  const empty = { copies: new Uint32Array(0), updates: new Uint32Array(0), nCopies: 0, nUpdates: 0, count, dead: dead.length, added: 0 };
  if (nSamples === 0) return empty;
  const aliveIdx = Uint32Array.from(alive);
  const w = new Float64Array(alive.length);
  for (let j = 0; j < alive.length; j++) w[j] = op[alive[j]];
  const src = sampleProportional(w, aliveIdx, nSamples, rng);
  const maxCopies = o.maxCopies ?? 51;
  // copies per source (capped: excess samples are dropped)
  const times = new Map<number, number>();
  const keep: number[] = [];
  for (const s of src) {
    const n = times.get(s) ?? 0;
    if (n + 1 >= maxCopies) continue;
    times.set(s, n + 1);
    keep.push(s);
  }
  const dsts: number[] = dead.slice();
  for (let k = 0; k < nAdd; k++) dsts.push(count + k);
  const nCopies = Math.min(keep.length, dsts.length);
  // relocations fill the dead slots first; additions only the slots actually filled
  let newCount = count;
  const copies = new Uint32Array(4 * nCopies);
  const f = new Float32Array(1);
  const fu = new Uint32Array(f.buffer);
  const bits = (x: number): number => {
    f[0] = x;
    return fu[0];
  };
  const params = new Map<number, { logit: number; delta: number }>();
  for (const [s, n] of times) {
    const r = relocation(op[s], n + 1, o.minOpacity);
    params.set(s, { logit: logit(r.opacity), delta: Math.log(r.scale) });
  }
  for (let k = 0; k < nCopies; k++) {
    const s = keep[k];
    const p = params.get(s)!;
    copies.set([dsts[k], s, bits(p.logit), bits(p.delta)], 4 * k);
    if (dsts[k] >= newCount) newCount = dsts[k] + 1;
  }
  const updates = new Uint32Array(4 * params.size);
  let u = 0;
  for (const [s, p] of params) {
    updates.set([s, 0, bits(p.logit), bits(p.delta)], 4 * u);
    u++;
  }
  return { copies, updates, nCopies, nUpdates: params.size, count: newCount, dead: dead.length, added: newCount - count };
}
