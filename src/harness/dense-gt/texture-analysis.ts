/**
 * Low-texture / sky analysis for the dense GT harness: how many depth-map pixels of each GT
 * error class a segment-texture gate (`segmentTextureStd` of src/dense/depth-filter.ts) would
 * remove, for several thresholds, either everywhere ('all') or only in low-texture regions
 * connected to the image border ('border', the sky heuristic).
 */
import type { DepthMap } from '../../dense/types';
import { rgbAtResolution } from '../../dense/dense-util';
import { borderComponents, borderConnectedMask, segmentTextureSampler, segmentTextureStd } from '../../dense/depth-filter';
import type { RGBAImage } from '../../core/types';

export interface TextureAnalysisConfig {
  radius?: number;
  step?: number;
  segmentColor?: number;
  minSamples?: number;
  /** Grey-std thresholds to evaluate (default 0.5 … 6). */
  thresholds?: number[];
  /** Border bits for the sky heuristic (1 top, 2 bottom, 4 left, 8 right; default 1). */
  borders?: number;
  /** Texture threshold of the region (coherence) statistics (default 2). */
  skyMaxStd?: number;
}

export type ClassCounts = { noGt: number; bad: number; mid: number; good: number };

export interface TextureCounts {
  /** All valid pixels by class. */
  all: ClassCounts;
  /** Per threshold: valid pixels with std < T ('low') and low ∧ border-connected ('border'). */
  low: ClassCounts[];
  border: ClassCounts[];
}

export function emptyCounts(nT: number): TextureCounts {
  const z = (): ClassCounts => ({ noGt: 0, bad: 0, mid: 0, good: 0 });
  return { all: z(), low: Array.from({ length: nT }, z), border: Array.from({ length: nT }, z) };
}

export function addCounts(a: TextureCounts, b: TextureCounts): void {
  const add = (x: ClassCounts, y: ClassCounts): void => {
    x.noGt += y.noGt;
    x.bad += y.bad;
    x.mid += y.mid;
    x.good += y.good;
  };
  add(a.all, b.all);
  b.low.forEach((c, k) => add(a.low[k], c));
  b.border.forEach((c, k) => add(a.border[k], c));
}

export const DEFAULT_THRESHOLDS = [0.5, 1, 1.5, 2, 3, 4, 6];

/** Texture maps of one view (shared by the input and the fused map). */
export function textureMasks(image: RGBAImage, W: number, H: number, cfg: TextureAnalysisConfig): { std: Float32Array; border: Uint8Array[] } {
  const rgb = rgbAtResolution(image, W, H);
  const seg = cfg.segmentColor ?? 10;
  const std = segmentTextureStd(rgb, W, H, cfg.radius ?? 5, cfg.step ?? 1, seg, cfg.minSamples ?? 8);
  const th = cfg.thresholds ?? DEFAULT_THRESHOLDS;
  const border = th.map((T) => {
    const low = new Uint8Array(W * H);
    for (let i = 0; i < low.length; i++) low[i] = std[i] < T ? 1 : 0;
    return borderConnectedMask(low, rgb, W, H, seg, cfg.borders ?? 1);
  });
  return { std, border };
}

/** Class counts of the valid pixels of `dm` vs GT depth. */
export function countTexture(dm: DepthMap | null, gtDepth: Float32Array, masks: { std: Float32Array; border: Uint8Array[] }, thresholds: number[]): TextureCounts {
  const c = emptyCounts(thresholds.length);
  if (!dm) return c;
  for (let i = 0; i < dm.depth.length; i++) {
    const z = dm.depth[i];
    if (!(z > 0)) continue;
    const zg = gtDepth[i];
    const cls: keyof ClassCounts = !(zg > 0) ? 'noGt' : Math.abs(z - zg) > 0.05 * zg ? 'bad' : Math.abs(z - zg) > 0.01 * zg ? 'mid' : 'good';
    c.all[cls]++;
    const s = masks.std[i];
    for (let k = 0; k < thresholds.length; k++) {
      if (!(s < thresholds[k])) continue;
      c.low[k][cls]++;
      if (masks.border[k][i]) c.border[k][cls]++;
    }
  }
  return c;
}

/** Human-readable table: per threshold, removed / valid per class (and %). */
export function formatTextureCounts(label: string, c: TextureCounts, thresholds: number[]): string {
  const pc = (a: number, b: number): string => `${a}/${b} (${b ? ((100 * a) / b).toFixed(1) : '—'}%)`;
  const lines = [`${label}: valid noGt ${c.all.noGt}, >5% ${c.all.bad}, 1-5% ${c.all.mid}, <=1% ${c.all.good}`];
  thresholds.forEach((T, k) => {
    const l = c.low[k];
    const b = c.border[k];
    lines.push(
      `  std<${T}: all-low  noGt ${pc(l.noGt, c.all.noGt)}  >5% ${pc(l.bad, c.all.bad)}  1-5% ${pc(l.mid, c.all.mid)}  <=1% ${pc(l.good, c.all.good)}`,
      `  std<${T}: border   noGt ${pc(b.noGt, c.all.noGt)}  >5% ${pc(b.bad, c.all.bad)}  1-5% ${pc(b.mid, c.all.mid)}  <=1% ${pc(b.good, c.all.good)}`,
    );
  });
  return lines.join('\n');
}

/** One top-connected untextured region of a view: area, coherent depths, GT-sky pixels, valid depths. */
export interface SkyComponent {
  frame: number;
  area: number;
  coherent: number;
  gtSky: number;
  valid: number;
}

/**
 * Top-connected untextured regions (std < maxStd, `borderComponents`) of one depth map with the
 * coherence measure of the sky test (valid pixels with ≥ 3 of 4 neighbours valid and within 5 % of
 * the pixel's plane) and the number of GT-sky pixels (no GT surface).
 */
export function skyComponents(
  frame: number,
  dm: DepthMap,
  image: RGBAImage,
  cam: { f: number; cx: number; cy: number; width: number; height: number },
  gtDepth: Float32Array,
  maxStd: number,
  cfg: TextureAnalysisConfig = {},
): SkyComponent[] {
  const W = dm.width;
  const H = dm.height;
  const rgb = rgbAtResolution(image, W, H);
  const seg = cfg.segmentColor ?? 10;
  const std = segmentTextureSampler(rgb, W, H, cfg.radius ?? 5, cfg.step ?? 1, seg, cfg.minSamples ?? 8);
  const { labels, count } = borderComponents((i) => std(i) < maxStd, rgb, W, H, seg, 1);
  const f = cam.f * (W / cam.width);
  const cx = (cam.cx + 0.5) * (W / cam.width) - 0.5;
  const cy = (cam.cy + 0.5) * (H / cam.height) - 0.5;
  const comps: SkyComponent[] = Array.from({ length: count }, () => ({ frame, area: 0, coherent: 0, gtSky: 0, valid: 0 }));
  const onPlane = (i: number, x: number, y: number, xq: number, yq: number, dq: number): boolean => {
    const d = dm.depth[i];
    const nx = dm.normal[3 * i], ny = dm.normal[3 * i + 1], nz = dm.normal[3 * i + 2];
    const nrp = (nx * (x - cx)) / f + (ny * (y - cy)) / f + nz;
    const den = (nx * (xq - cx)) / f + (ny * (yq - cy)) / f + nz;
    const z = nrp < 0 && den < -0.05 && nrp < 0.5 * den && nrp > 2 * den ? (d * nrp) / den : d;
    return Math.abs(dq - z) <= 0.05 * dq;
  };
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const l = labels[i];
      if (!l) continue;
      const c = comps[l - 1];
      c.area++;
      if (!(gtDepth[i] > 0)) c.gtSky++;
      const d = dm.depth[i];
      if (!(d > 0 && d < Infinity)) continue;
      c.valid++;
      let agree = 0;
      for (let k = 0; k < 4; k++) {
        const xx = k === 0 ? x - 1 : k === 1 ? x + 1 : x;
        const yy = k === 2 ? y - 1 : k === 3 ? y + 1 : y;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        const dj = dm.depth[yy * W + xx];
        if (dj > 0 && dj < Infinity && onPlane(i, x, y, xx, yy, dj)) agree++;
      }
      if (agree >= 3) c.coherent++;
    }
  }
  return comps;
}

/** Area-weighted histogram of the coherence ratio, split into GT-sky (> 50 % sky pixels) and surface regions. */
export function formatSkyComponents(comps: SkyComponent[], minArea = 100): string {
  const bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01];
  const lines: string[] = [];
  for (const [label, ratio] of [
    ['coherent / area', (c: SkyComponent): number => c.coherent / c.area],
    ['coherent / valid', (c: SkyComponent): number => c.coherent / Math.max(1, c.valid)],
    ['valid / area', (c: SkyComponent): number => c.valid / c.area],
  ] as const) {
    const h = { sky: bins.map(() => 0), surf: bins.map(() => 0) };
    const n = { sky: 0, surf: 0 };
    for (const c of comps) {
      if (c.area < minArea) continue;
      const k = bins.findIndex((b) => ratio(c) < b);
      const cls = c.gtSky > 0.5 * c.area ? 'sky' : 'surf';
      h[cls][k] += c.area;
      n[cls]++;
    }
    const row = (a: number[]): string => a.map((v) => String(v).padStart(8)).join('');
    lines.push(
      `top-connected untextured regions (area >= ${minArea} px), area by ${label}  <${bins.map((b) => String(Math.min(1, b)).padStart(6)).join('  ')}`,
      `  GT sky   (${n.sky} regions): ${row(h.sky)}`,
      `  surfaces (${n.surf} regions): ${row(h.surf)}`,
    );
  }
  return lines.join('\n');
}
