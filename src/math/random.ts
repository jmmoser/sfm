/**
 * Deterministic, seedable pseudo-random numbers. ALL randomness in the code base
 * goes through `Rng` so that tests and RANSAC runs are reproducible.
 *
 * Generator: sfc32 (Chris Doty-Humphrey's "Small Fast Chaotic" PRNG, PractRand-clean,
 * 128-bit state, 32-bit output) seeded through splitmix32 (Steele et al. 2014,
 * 32-bit finaliser of MurmurHash3) so that nearby seeds give unrelated streams.
 */

const TWO_POW_32 = 4294967296;
const INV_2_32 = 1 / TWO_POW_32;
const DEFAULT_SEED = 0x2545f491;

export class Rng {
  private a = 0;
  private b = 0;
  private c = 0;
  private d = 0;
  private spare = 0;
  private hasSpare = false;
  /** Scratch for partial Fisher–Yates in `sample` (grown on demand, reused). */
  private perm: Uint32Array = new Uint32Array(0);

  /** @param seed any finite number (integer part of the low 32 bits and of seed/2³² are mixed in). */
  constructor(seed: number = DEFAULT_SEED) {
    this.seed(seed);
  }

  /** Re-seed (resets the cached Gaussian). */
  seed(seed: number): void {
    let s = Number.isFinite(seed) ? (seed >>> 0) ^ (Math.floor(seed / TWO_POW_32) >>> 0) : DEFAULT_SEED;
    const next = (): number => {
      s = (s + 0x9e3779b9) | 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      return (z ^ (z >>> 16)) | 0;
    };
    this.a = next();
    this.b = next();
    this.c = next();
    this.d = next();
    this.hasSpare = false;
    for (let i = 0; i < 12; i++) this.nextU32();
  }

  /** Uniform 32-bit unsigned integer. */
  nextU32(): number {
    const a = this.a,
      b = this.b,
      c = this.c,
      d = this.d;
    const t = (((a + b) | 0) + d) | 0;
    this.d = (d + 1) | 0;
    this.a = b ^ (b >>> 9);
    this.b = (c + (c << 3)) | 0;
    const cr = (c << 21) | (c >>> 11);
    this.c = (cr + t) | 0;
    return t >>> 0;
  }

  /** Uniform in [0, 1) with 32 bits of resolution. */
  next(): number {
    return this.nextU32() * INV_2_32;
  }

  /** Uniform in [0, 1) with full 53-bit resolution (two draws). */
  nextDouble(): number {
    const hi = this.nextU32() >>> 5; // 27 bits
    const lo = this.nextU32() >>> 6; // 26 bits
    return (hi * 67108864 + lo) / 9007199254740992;
  }

  /** Uniform integer in [0, n) for integer 1 ≤ n ≤ 2³² (returns 0 for n ≤ 1). */
  int(n: number): number {
    if (!(n > 1)) return 0;
    return Math.floor(this.nextU32() * INV_2_32 * n);
  }

  /** Uniform in [lo, hi). */
  uniform(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next();
  }

  /** Standard normal N(0, 1) via the Box–Muller transform (second value cached). */
  gaussian(): number {
    if (this.hasSpare) {
      this.hasSpare = false;
      return this.spare;
    }
    const u1 = 1 - this.next(); // (0, 1]
    const u2 = this.next();
    const r = Math.sqrt(-2 * Math.log(u1));
    const th = 2 * Math.PI * u2;
    this.spare = r * Math.sin(th);
    this.hasSpare = true;
    return r * Math.cos(th);
  }

  /**
   * k distinct indices drawn uniformly from [0, n) (a uniformly random k-subset; the
   * ORDER of the output is not uniformly random for small k). Uses Floyd's algorithm
   * (Bentley & Floyd, CACM 1987) when k² ≤ n or k ≤ 8 (O(k²), no O(n) memory), and a
   * partial Fisher–Yates shuffle over a reused scratch buffer otherwise.
   * Writes the first k entries of `out` when it is given and long enough (else
   * allocates). Throws RangeError if k > n.
   */
  sample(n: number, k: number, out?: Uint32Array): Uint32Array {
    if (!(k >= 0) || !(k <= n)) throw new RangeError(`Rng.sample: k=${k} not in [0, n=${n}]`);
    const res = out && out.length >= k ? out : new Uint32Array(k);
    if (k === 0) return res;
    if (k <= 8 || k * k <= n) {
      // Floyd: for j = n−k … n−1, pick t ∈ [0, j]; insert t, or j if t was taken.
      let m = 0;
      for (let j = n - k; j < n; j++) {
        const t = this.int(j + 1);
        let taken = false;
        for (let i = 0; i < m; i++) {
          if (res[i] === t) {
            taken = true;
            break;
          }
        }
        res[m++] = taken ? j : t;
      }
      return res;
    }
    if (this.perm.length < n) this.perm = new Uint32Array(Math.max(n, this.perm.length * 2));
    const p = this.perm;
    for (let i = 0; i < n; i++) p[i] = i;
    for (let i = 0; i < k; i++) {
      const j = i + this.int(n - i);
      const t = p[i];
      p[i] = p[j];
      p[j] = t;
      res[i] = p[i];
    }
    return res;
  }

  /** In-place uniform Fisher–Yates shuffle. */
  shuffle<T>(a: T[] | Uint32Array): void {
    const arr = a as unknown as T[]; // same element access semantics for both
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const t = arr[i];
      arr[i] = arr[j];
      arr[j] = t;
    }
  }
}
