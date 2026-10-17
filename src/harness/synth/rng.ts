/**
 * Small local deterministic PRNG for the synthetic dataset generator.
 *
 * The generator is the GROUND TRUTH for the whole pipeline, so it deliberately does
 * not depend on the modules it is used to test (src/math etc.). sfc32 seeded through
 * splitmix32; `derive()` produces independent, reproducible sub-streams so that e.g.
 * changing the number of frames does not change the textures or the IMU noise.
 */

export class SynthRng {
  private a = 0;
  private b = 0;
  private c = 0;
  private d = 0;
  private spare = 0;
  private hasSpare = false;
  readonly seedValue: number;

  constructor(seed = 1) {
    this.seedValue = seed >>> 0;
    let s = this.seedValue ^ 0x5bd1e995;
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
    for (let i = 0; i < 12; i++) this.u32();
  }

  /** Uniform 32-bit unsigned integer. */
  u32(): number {
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

  /** Uniform in [0, 1). */
  next(): number {
    return this.u32() / 4294967296;
  }

  /** Uniform in [lo, hi). */
  uniform(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next();
  }

  /** Uniform integer in [0, n). */
  int(n: number): number {
    return n > 1 ? Math.floor(this.next() * n) : 0;
  }

  /** Random element of an array. */
  pick<T>(arr: readonly T[]): T {
    return arr[this.int(arr.length)];
  }

  /** Standard normal (Box–Muller, second value cached). */
  gaussian(): number {
    if (this.hasSpare) {
      this.hasSpare = false;
      return this.spare;
    }
    const u1 = 1 - this.next();
    const u2 = this.next();
    const r = Math.sqrt(-2 * Math.log(u1));
    const th = 2 * Math.PI * u2;
    this.spare = r * Math.sin(th);
    this.hasSpare = true;
    return r * Math.cos(th);
  }

  /** Independent sub-stream keyed by a label (stable across runs). */
  derive(label: string): SynthRng {
    return new SynthRng(hashString(label, this.seedValue));
  }
}

/** FNV-1a style 32-bit string hash mixed with a seed. */
export function hashString(s: string, seed = 0): number {
  let h = (0x811c9dc5 ^ seed) >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d) >>> 0;
  h ^= h >>> 15;
  return h >>> 0;
}
