/**
 * Order statistics and simple summaries. `median`, `percentile`, `mad` never modify
 * their input (they copy into a reused module-level scratch buffer and run
 * quickselect there). Inputs must not contain NaN. Empty input → NaN.
 */

let scratch = new Float64Array(64);
/** Inputs larger than this get a temporary copy instead of growing the persistent scratch. */
const MAX_PERSISTENT_SCRATCH = 1 << 18;

function copyToScratch(v: ArrayLike<number>): Float64Array {
  const n = v.length;
  let s: Float64Array;
  if (n > MAX_PERSISTENT_SCRATCH) s = new Float64Array(n);
  else {
    if (scratch.length < n) scratch = new Float64Array(Math.max(n, Math.min(scratch.length * 2, MAX_PERSISTENT_SCRATCH)));
    s = scratch;
  }
  for (let i = 0; i < n; i++) s[i] = v[i];
  return s;
}

/**
 * In-place selection (Hoare's FIND with median-of-three pivot): returns the k-th
 * smallest element (0-based) of a[0 .. a.length-1] and partially orders `a` so that
 * a[k] holds it, everything before is ≤ and everything after is ≥. Expected O(n).
 */
export function quickselect(a: Float64Array | number[], k: number): number {
  return selectRange(a, 0, a.length - 1, k);
}

function selectRange(a: Float64Array | number[], lo: number, hi: number, k: number): number {
  while (hi > lo) {
    const mid = (lo + hi) >>> 1;
    let t: number;
    if (a[mid] < a[lo]) {
      t = a[mid];
      a[mid] = a[lo];
      a[lo] = t;
    }
    if (a[hi] < a[lo]) {
      t = a[hi];
      a[hi] = a[lo];
      a[lo] = t;
    }
    if (a[hi] < a[mid]) {
      t = a[hi];
      a[hi] = a[mid];
      a[mid] = t;
    }
    const pivot = a[mid];
    let i = lo,
      j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) {
        t = a[i];
        a[i] = a[j];
        a[j] = t;
        i++;
        j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return a[k];
  }
  return a[k];
}

/** Minimum of a[from .. to-1]. */
function minRange(a: Float64Array, from: number, to: number): number {
  let m = a[from];
  for (let i = from + 1; i < to; i++) if (a[i] < m) m = a[i];
  return m;
}

/** Maximum of a[from .. to-1]. */
function maxRange(a: Float64Array, from: number, to: number): number {
  let m = a[from];
  for (let i = from + 1; i < to; i++) if (a[i] > m) m = a[i];
  return m;
}

/** Median of the scratch prefix of length n (destroys the order of the prefix). */
function medianOfScratch(s: Float64Array, n: number): number {
  const k = n >>> 1;
  const hiMid = selectRange(s, 0, n - 1, k);
  if (n & 1) return hiMid;
  // Elements before k are ≤ s[k]; the lower middle is their maximum.
  return 0.5 * (maxRange(s, 0, k) + hiMid);
}

/** Median (average of the two middle values for even length). Non-destructive. */
export function median(v: ArrayLike<number>): number {
  const n = v.length;
  if (n === 0) return NaN;
  return medianOfScratch(copyToScratch(v), n);
}

/**
 * p-th percentile, p ∈ [0, 100] (clamped), with linear interpolation between order
 * statistics at rank p/100·(n − 1) (NumPy's default "linear" method). Non-destructive.
 */
export function percentile(v: ArrayLike<number>, p: number): number {
  const n = v.length;
  if (n === 0) return NaN;
  const pc = p <= 0 ? 0 : p >= 100 ? 100 : p;
  const pos = (pc / 100) * (n - 1);
  const k = Math.floor(pos);
  const frac = pos - k;
  const s = copyToScratch(v);
  const lo = selectRange(s, 0, n - 1, k);
  if (frac === 0 || k + 1 >= n) return lo;
  const hiVal = minRange(s, k + 1, n);
  return lo + frac * (hiVal - lo);
}

/** Arithmetic mean (Neumaier-compensated summation). */
export function mean(v: ArrayLike<number>): number {
  const n = v.length;
  if (n === 0) return NaN;
  let sum = 0,
    comp = 0;
  for (let i = 0; i < n; i++) {
    const x = v[i];
    const t = sum + x;
    if (Math.abs(sum) >= Math.abs(x)) comp += sum - t + x;
    else comp += x - t + sum;
    sum = t;
  }
  return (sum + comp) / n;
}

/**
 * Median absolute deviation median(|v − median(v)|), UNSCALED (multiply by 1.4826
 * for a consistent estimate of σ under Gaussian noise). Non-destructive.
 */
export function mad(v: ArrayLike<number>): number {
  const n = v.length;
  if (n === 0) return NaN;
  const s = copyToScratch(v);
  const med = medianOfScratch(s, n);
  // Overwrite the scratch prefix with absolute deviations (order is irrelevant).
  for (let i = 0; i < n; i++) s[i] = Math.abs(s[i] - med);
  return medianOfScratch(s, n);
}

/** Variance with `ddof` delta degrees of freedom (0 = population, 1 = sample). Two-pass. */
export function variance(v: ArrayLike<number>, ddof = 0): number {
  const n = v.length;
  if (n - ddof <= 0) return NaN;
  const m = mean(v);
  let s = 0,
    c = 0;
  for (let i = 0; i < n; i++) {
    const d = v[i] - m;
    s += d * d;
    c += d;
  }
  return (s - (c * c) / n) / (n - ddof);
}

/** Root mean square. */
export function rms(v: ArrayLike<number>): number {
  const n = v.length;
  if (n === 0) return NaN;
  let s = 0;
  for (let i = 0; i < n; i++) s += v[i] * v[i];
  return Math.sqrt(s / n);
}
