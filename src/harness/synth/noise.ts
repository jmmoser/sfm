/**
 * Procedural noise on Float32 rasters (pure, no DOM): tileable value-noise fBm,
 * Worley (cellular) distance, and small raster helpers used by textures.ts.
 */
import { SynthRng } from './rng';

export interface Raster {
  width: number;
  height: number;
  /** One float per pixel, row-major. */
  data: Float32Array;
}

function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

/**
 * Tileable value noise: lattice of `cellsX × cellsY` random values, quintic
 * interpolation. Adds `amplitude · (noise − 0.5)·2` into `out` (range ±amplitude).
 */
export function addValueNoise(out: Raster, cellsX: number, cellsY: number, amplitude: number, rng: SynthRng): void {
  const W = out.width,
    H = out.height,
    d = out.data;
  cellsX = Math.max(1, Math.round(cellsX));
  cellsY = Math.max(1, Math.round(cellsY));
  const lat = new Float32Array(cellsX * cellsY);
  for (let i = 0; i < lat.length; i++) lat[i] = rng.next() * 2 - 1;
  const fxs = new Float32Array(W);
  const ix0 = new Int32Array(W);
  const ix1 = new Int32Array(W);
  for (let x = 0; x < W; x++) {
    const gx = ((x + 0.5) / W) * cellsX;
    const i0 = Math.floor(gx);
    fxs[x] = fade(gx - i0);
    ix0[x] = i0 % cellsX;
    ix1[x] = (i0 + 1) % cellsX;
  }
  for (let y = 0; y < H; y++) {
    const gy = ((y + 0.5) / H) * cellsY;
    const j0 = Math.floor(gy);
    const fy = fade(gy - j0);
    const r0 = (j0 % cellsY) * cellsX,
      r1 = ((j0 + 1) % cellsY) * cellsX;
    const row = y * W;
    for (let x = 0; x < W; x++) {
      const a = lat[r0 + ix0[x]],
        b = lat[r0 + ix1[x]],
        c = lat[r1 + ix0[x]],
        e = lat[r1 + ix1[x]];
      const fx = fxs[x];
      const top = a + (b - a) * fx,
        bot = c + (e - c) * fx;
      d[row + x] += amplitude * (top + (bot - top) * fy);
    }
  }
}

export interface FbmOptions {
  /** Lattice cells across the width for the lowest octave. */
  baseCells: number;
  octaves: number;
  /** Amplitude multiplier per octave (default 0.55). */
  gain?: number;
  /** Frequency multiplier per octave (default 2). */
  lacunarity?: number;
}

/** Fractal value noise, approximately in [−1, 1] (tileable). */
export function fbm(width: number, height: number, rng: SynthRng, o: FbmOptions): Raster {
  const r: Raster = { width, height, data: new Float32Array(width * height) };
  const gain = o.gain ?? 0.55,
    lac = o.lacunarity ?? 2;
  let amp = 1,
    cells = o.baseCells,
    norm = 0;
  const aspect = height / width;
  for (let k = 0; k < o.octaves; k++) {
    if (cells > width) break;
    addValueNoise(r, cells, Math.max(1, cells * aspect), amp, rng);
    norm += amp;
    amp *= gain;
    cells *= lac;
  }
  const inv = 1 / Math.max(norm * 0.6, 1e-6);
  const d = r.data;
  for (let i = 0; i < d.length; i++) d[i] = Math.max(-1, Math.min(1, d[i] * inv));
  return r;
}

/**
 * Worley noise: for each pixel the distance (in pixels) to the nearest of `n`
 * random feature points (tileable), and the index of that point.
 */
export function worley(width: number, height: number, n: number, rng: SynthRng): { dist: Float32Array; id: Int32Array; dist2: Float32Array } {
  const px = new Float32Array(n),
    py = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    px[i] = rng.next() * width;
    py[i] = rng.next() * height;
  }
  // Bucket grid for speed.
  const cell = Math.max(4, Math.sqrt((width * height) / n));
  const gw = Math.ceil(width / cell),
    gh = Math.ceil(height / cell);
  const buckets: number[][] = Array.from({ length: gw * gh }, () => []);
  for (let i = 0; i < n; i++) buckets[Math.min(gh - 1, Math.floor(py[i] / cell)) * gw + Math.min(gw - 1, Math.floor(px[i] / cell))].push(i);
  const dist = new Float32Array(width * height);
  const dist2 = new Float32Array(width * height);
  const id = new Int32Array(width * height);
  for (let y = 0; y < height; y++) {
    const gy = Math.floor(y / cell);
    for (let x = 0; x < width; x++) {
      const gx = Math.floor(x / cell);
      let best = Infinity,
        second = Infinity,
        bi = 0;
      for (let dy = -2; dy <= 2; dy++) {
        const by = (((gy + dy) % gh) + gh) % gh;
        for (let dx = -2; dx <= 2; dx++) {
          const bx = (((gx + dx) % gw) + gw) % gw;
          for (const i of buckets[by * gw + bx]) {
            let ddx = Math.abs(px[i] - x),
              ddy = Math.abs(py[i] - y);
            if (ddx > width / 2) ddx = width - ddx;
            if (ddy > height / 2) ddy = height - ddy;
            const d2 = ddx * ddx + ddy * ddy;
            if (d2 < best) {
              second = best;
              best = d2;
              bi = i;
            } else if (d2 < second) second = d2;
          }
        }
      }
      const o = y * width + x;
      dist[o] = Math.sqrt(best);
      dist2[o] = Math.sqrt(second);
      id[o] = bi;
    }
  }
  return { dist, id, dist2 };
}
