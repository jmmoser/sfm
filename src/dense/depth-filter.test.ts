import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, RGBAImage } from '../core/types';
import { Rng } from '../math/random';
import { borderComponents, borderConnectedMask, depthHaloMask, removeDepthHalos, segmentTextureStd } from './depth-filter';
import { rgbAtResolution } from './dense-util';
import { fuseDepthMaps, type FusionStats } from './fusion';
import { makeNearScene, orbitPoses, pinholeIntrinsics } from './synthetic';
import { renderGtViews } from './testutil-scene';
import { MemoryDepthMapStore, type DepthMap } from './types';

const log = (...a: unknown[]): void => console.log('[depth-filter]', ...a);

const W = 96;
const H = 96;
const cam: CameraIntrinsics = { width: W, height: H, f: 80, cx: 47.5, cy: 47.5, k1: 0, k2: 0 };
/** Foreground square [S0, S1)² at depth 1 in front of a plane at depth 3. */
const S0 = 30;
const S1 = 66;
const HALO = 4;

type Region = 'fg' | 'halo' | 'bg';

function regionOf(x: number, y: number): Region {
  if (x >= S0 && x < S1 && y >= S0 && y < S1) return 'fg';
  const dx = x < S0 ? S0 - x : x >= S1 ? x - S1 + 1 : 0;
  const dy = y < S0 ? S0 - y : y >= S1 ? y - S1 + 1 : 0;
  return Math.max(dx, dy) <= HALO ? 'halo' : 'bg';
}

/**
 * Textured foreground (dark red, ±25 levels) in front of a weakly textured light-grey
 * background (±1 level, `sameColour`: the foreground's texture instead); the 4-px band of
 * background pixels around the square carries the foreground's depth (a PatchMatch halo).
 */
function haloScene(sameColour = false): { dm: DepthMap; image: RGBAImage } {
  const rng = new Rng(3);
  const n = W * H;
  const depth = new Float32Array(n);
  const normal = new Float32Array(3 * n);
  const cost = new Float32Array(n).fill(0.1);
  const data = new Uint8ClampedArray(4 * n);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const r = regionOf(x, y);
      depth[i] = r === 'bg' ? 3 : 1;
      normal[3 * i + 2] = -1;
      const fgColour = r === 'fg' || sameColour;
      const t = fgColour ? 25 * (2 * rng.next() - 1) : 2 * rng.next() - 1;
      data[4 * i] = fgColour ? 150 + t : 200 + t;
      data[4 * i + 1] = fgColour ? 50 + t : 200 + t;
      data[4 * i + 2] = fgColour ? 40 + t : 205 + t;
      data[4 * i + 3] = 255;
    }
  }
  return { dm: { width: W, height: H, depth, normal, cost }, image: { width: W, height: H, data } };
}

function rejectedByRegion(mask: Uint32Array | null): Record<Region, number> {
  const tot: Record<Region, number> = { fg: 0, halo: 0, bg: 0 };
  const rej: Record<Region, number> = { fg: 0, halo: 0, bg: 0 };
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const r = regionOf(x, y);
      tot[r]++;
      if (mask && (mask[i >>> 5] >>> (i & 31)) & 1) rej[r]++;
    }
  }
  return { fg: rej.fg / tot.fg, halo: rej.halo / tot.halo, bg: rej.bg / tot.bg };
}

describe('depthHaloMask / removeDepthHalos', () => {
  it('removes a background-coloured band carrying the foreground depth, keeps both surfaces', () => {
    const { dm, image } = haloScene();
    const f = rejectedByRegion(depthHaloMask(dm, image, cam));
    log(`halo band: rejected halo ${(100 * f.halo).toFixed(1)} %, foreground ${(100 * f.fg).toFixed(1)} %, background ${(100 * f.bg).toFixed(1)} %`);
    expect(f.halo).toBeGreaterThan(0.9);
    expect(f.fg).toBeLessThan(0.02);
    expect(f.bg).toBeLessThan(0.01);
  });

  it('does not erode same-coloured surfaces at different depths', () => {
    const { dm, image } = haloScene(true);
    const f = rejectedByRegion(depthHaloMask(dm, image, cam));
    log(`same colour: rejected halo ${(100 * f.halo).toFixed(1)} %, foreground ${(100 * f.fg).toFixed(1)} %, background ${(100 * f.bg).toFixed(1)} %`);
    expect(f.fg).toBeLessThan(0.02);
    expect(f.bg).toBeLessThan(0.02);
  });

  it('keeps slanted planes (plane-predicted depth), handles resolution changes and empty input', () => {
    // one plane slanted 55° about the x axis: depth 1.9 → 22 from the bottom to the top row
    // (up to 6 % per pixel), no discontinuity
    const n = W * H;
    const depth = new Float32Array(n);
    const normal = new Float32Array(3 * n);
    const a = (55 * Math.PI) / 180;
    const nrm = [0, -Math.sin(a), -Math.cos(a)];
    const d0 = 2; // plane n·X + d0 = 0
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const nr = (nrm[0] * (x - cam.cx)) / cam.f + (nrm[1] * (y - cam.cy)) / cam.f + nrm[2];
        depth[i] = -d0 / nr;
        normal.set(nrm, 3 * i);
      }
    }
    expect(Math.min(...depth)).toBeGreaterThan(1);
    const { image } = haloScene();
    const dm = { width: W, height: H, depth, normal, cost: new Float32Array(n) };
    const m = depthHaloMask(dm, image, cam);
    let rej = 0;
    if (m) for (let i = 0; i < n; i++) rej += (m[i >>> 5] >>> (i & 31)) & 1;
    expect(rej).toBe(0);
    // camera given at 2× the depth-map resolution, image at 2× too
    const big: CameraIntrinsics = { width: 2 * W, height: 2 * H, f: 160, cx: 95.5, cy: 95.5, k1: 0, k2: 0 };
    const s = haloScene();
    const img2 = { width: 2 * W, height: 2 * H, data: new Uint8ClampedArray(16 * W * H) };
    for (let y = 0; y < 2 * H; y++) for (let x = 0; x < 2 * W; x++) for (let k = 0; k < 4; k++) img2.data[4 * (y * 2 * W + x) + k] = s.image.data[4 * ((y >> 1) * W + (x >> 1)) + k];
    const f2 = rejectedByRegion(depthHaloMask(s.dm, img2, big));
    expect(f2.halo).toBeGreaterThan(0.85);
    expect(f2.fg).toBeLessThan(0.02);
    // no valid pixels / filter off
    const empty = { width: W, height: H, depth: new Float32Array(n), normal: new Float32Array(3 * n), cost: new Float32Array(n) };
    expect(depthHaloMask(empty, image, cam)).toBeNull();
    expect(depthHaloMask(s.dm, s.image, cam, { radius: 0 })).toBeNull();
  });

  it('removeDepthHalos returns an invalidated copy and leaves the input untouched', () => {
    const { dm, image } = haloScene();
    const before = dm.depth.slice();
    const out = removeDepthHalos(dm, image, cam);
    expect(Array.from(dm.depth)).toEqual(Array.from(before));
    let removed = 0;
    for (let i = 0; i < W * H; i++) {
      if (out.depth[i] === 0 && dm.depth[i] > 0) {
        removed++;
        expect(out.cost[i]).toBe(2);
        expect(out.normal[3 * i + 2]).toBe(0);
      }
    }
    expect(removed).toBeGreaterThan(0);
  });

  it('fusion with haloFilter: clean GT maps lose almost nothing, rejected pixels are not in the filtered maps', async () => {
    const scene = makeNearScene();
    const c = pinholeIntrinsics(160, 120, (60 * Math.PI) / 180);
    const poses = orbitPoses([0, 0, 0.25], 3, (35 * Math.PI) / 180, 12);
    const gt = renderGtViews(scene, c, poses, { noise: 0.002, normalNoiseDeg: 3, seed: 5 });
    const nb = new Map<number, number[]>();
    for (let i = 0; i < 12; i++) nb.set(i, [(i + 1) % 12, (i + 11) % 12, (i + 2) % 12, (i + 10) % 12]);
    const plain = await fuseDepthMaps(gt.views, gt.store, nb, { minViews: 2 });
    let stats: FusionStats | null = null;
    const filtered = new MemoryDepthMapStore();
    const halo = await fuseDepthMaps(gt.views, gt.store, nb, { minViews: 2, haloFilter: {}, filteredStore: filtered, onStats: (s) => (stats = s) });
    const s = stats! as FusionStats;
    log(`clean maps: ${plain.count} → ${halo.count} points with the halo filter (${s.haloRejected} px rejected, ${s.haloMs!.toFixed(0)} ms)`);
    expect(halo.count).toBeGreaterThan(0.97 * plain.count);
    expect(s.haloRejected).toBeGreaterThanOrEqual(0);
    // rejected pixels never appear in the filtered maps
    for (let k = 0; k < 12; k++) {
      const src = (await gt.store.get(k))!;
      const m = depthHaloMask(src, gt.views[k].image, gt.views[k].cam);
      const fm = (await filtered.get(k))!;
      if (!m) continue;
      for (let i = 0; i < fm.depth.length; i++) if ((m[i >>> 5] >>> (i & 31)) & 1) expect(fm.depth[i]).toBe(0);
    }
  });
});

/**
 * "Sky" scene: rows [0, SKY) are a smooth light-blue sky (±1 level, vertical gradient), rows
 * [SKY, H) a textured dark-red wall at depth 10 facing the camera. PatchMatch-style output: the
 * wall is valid; a band of BAND sky rows above the wall carries the wall's plane (a halo wider
 * than the halo filter's window), the sky above it has scattered random depths (every 3rd
 * pixel) or none. A smooth grey patch inside the wall (rows 70–85, cols 30–60) carries the
 * wall's depth (a real untextured surface, not connected to the top).
 */
const SKY = 40;
const BAND = 12;
type SkyRegion = 'band' | 'sky' | 'wall' | 'patch';
function skyScene(): { dm: DepthMap; image: RGBAImage; region: (x: number, y: number) => SkyRegion } {
  const rng = new Rng(11);
  const n = W * H;
  const depth = new Float32Array(n);
  const normal = new Float32Array(3 * n);
  const cost = new Float32Array(n).fill(0.2);
  const data = new Uint8ClampedArray(4 * n);
  const region = (x: number, y: number): SkyRegion => (y >= SKY ? (y >= 70 && y < 86 && x >= 30 && x < 60 ? 'patch' : 'wall') : y >= SKY - BAND ? 'band' : 'sky');
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const r = region(x, y);
      const sky = r === 'sky' || r === 'band';
      const t = r === 'wall' ? 25 * (2 * rng.next() - 1) : 2 * rng.next() - 1;
      const c = sky ? [150 + 0.5 * y + t, 185 + 0.3 * y + t, 230 + t] : r === 'patch' ? [180 + t, 180 + t, 180 + t] : [150 + t, 50 + t, 40 + t];
      data.set([c[0], c[1], c[2], 255], 4 * i);
      if (r === 'sky') {
        if (i % 3 === 0) {
          depth[i] = 2 + 30 * rng.next();
          normal[3 * i + 2] = -1;
        }
      } else {
        depth[i] = 10;
        normal[3 * i + 2] = -1;
      }
    }
  }
  return { dm: { width: W, height: H, depth, normal, cost }, image: { width: W, height: H, data }, region };
}

function rejectedFractions(mask: Uint32Array | null, dm: DepthMap, region: (x: number, y: number) => string): Record<string, number> {
  const tot: Record<string, number> = {};
  const rej: Record<string, number> = {};
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!(dm.depth[i] > 0)) continue;
      const r = region(x, y);
      tot[r] = (tot[r] ?? 0) + 1;
      if (mask && (mask[i >>> 5] >>> (i & 31)) & 1) rej[r] = (rej[r] ?? 0) + 1;
    }
  }
  return Object.fromEntries(Object.keys(tot).map((k) => [k, (rej[k] ?? 0) / tot[k]]));
}

describe('sky test (untextured top-connected regions without coherent depth)', () => {
  it('segmentTextureStd: the sky next to a strong edge measures the sky, texture is high, isolated colours are Infinity', () => {
    const { image } = skyScene();
    const rgb = rgbAtResolution(image, W, H);
    for (const step of [1, 2]) {
      const std = segmentTextureStd(rgb, W, H, 5, step, 10, 8);
      // sky pixel directly above the wall edge, sky pixel far from it, wall pixel
      expect(std[(SKY - 1) * W + 48]).toBeLessThan(2);
      expect(std[10 * W + 48]).toBeLessThan(2);
      const wall = std[(SKY + 15) * W + 10];
      expect(wall === Infinity || wall > 3).toBe(true);
    }
    // a lone pixel of a unique colour
    const solo = { width: W, height: H, data: new Uint8ClampedArray(image.data) };
    solo.data.set([0, 255, 0, 255], 4 * (20 * W + 20));
    expect(segmentTextureStd(rgbAtResolution(solo, W, H), W, H, 5, 1, 10, 8)[20 * W + 20]).toBe(Infinity);
  });

  it('borderConnectedMask / borderComponents keep only regions connected to the selected border through similar colours', () => {
    const { image, region } = skyScene();
    const rgb = rgbAtResolution(image, W, H);
    const std = segmentTextureStd(rgb, W, H, 5, 1, 10, 8);
    const low = Uint8Array.from(std, (v) => (v < 3 ? 1 : 0));
    const top = borderConnectedMask(low, rgb, W, H, 10, 1);
    const comp = borderComponents((i) => std[i] < 3, rgb, W, H, 10, 1);
    expect(comp.count).toBe(1);
    expect(Array.from(comp.labels, (l) => (l > 0 ? 1 : 0))).toEqual(Array.from(top));
    let sky = 0, skyIn = 0, patch = 0, patchLow = 0, patchIn = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const r = region(x, y);
        if (r === 'sky' || r === 'band') {
          sky++;
          skyIn += top[i];
        } else if (r === 'patch') {
          patch++;
          patchLow += low[i];
          patchIn += top[i];
        }
      }
    }
    expect(skyIn / sky).toBeGreaterThan(0.97);
    expect(patchLow / patch).toBeGreaterThan(0.5);
    expect(patchIn).toBe(0);
    // bottom border: at most a few isolated wall pixels, never the sky
    const bottom = borderConnectedMask(low, rgb, W, H, 10, 2);
    expect(bottom.reduce((a, b) => a + b, 0)).toBeLessThan(0.01 * W * H);
    for (let i = 0; i < SKY * W; i++) expect(bottom[i]).toBe(0);
  });

  it('removes a halo band wider than the halo window and the random sky depths, keeps the wall and the enclosed smooth patch', () => {
    const { dm, image, region } = skyScene();
    const plain = rejectedFractions(depthHaloMask(dm, image, cam, { skyMaxStd: 0 }), dm, region);
    const sky = rejectedFractions(depthHaloMask(dm, image, cam), dm, region);
    log(`sky scene: halo vote only ${JSON.stringify(plain)}, with the sky test ${JSON.stringify(sky)}`);
    // the 8-px halo vote leaves the 12-px band (consistent plane extension)
    expect(plain.band).toBeLessThan(0.5);
    expect(sky.band).toBe(1);
    expect(sky.sky).toBe(1);
    expect(sky.wall).toBeLessThan(0.01);
    expect(sky.patch).toBe(0);
    // sky test alone (halo vote off)
    const only = rejectedFractions(depthHaloMask(dm, image, cam, { radius: 0 }), dm, region);
    expect(only.band).toBe(1);
    expect(only.wall).toBe(0);
    expect(depthHaloMask(dm, image, cam, { radius: 0, skyMaxStd: 0 })).toBeNull();
  });

  it('keeps an untextured top-connected surface with a coherent depth (a smooth wall), removes it when its depths are incoherent', () => {
    const rng = new Rng(5);
    const n = W * H;
    const data = new Uint8ClampedArray(4 * n);
    for (let i = 0; i < n; i++) {
      const t = 2 * rng.next() - 1;
      data.set([215 + t, 212 + t, 205 + t, 255], 4 * i);
    }
    const image = { width: W, height: H, data };
    const make = (randomDepth: boolean): DepthMap => {
      const depth = new Float32Array(n);
      const normal = new Float32Array(3 * n);
      for (let i = 0; i < n; i++) {
        if (rng.next() < 0.15) continue; // 15 % holes
        depth[i] = randomDepth ? 1 + 5 * rng.next() : 2;
        normal[3 * i + 2] = -1;
      }
      return { width: W, height: H, depth, normal, cost: new Float32Array(n) };
    };
    const count = (m: Uint32Array | null): number => (m ? m.reduce((a, w) => a + popcount(w), 0) : 0);
    const coherent = make(false);
    const noisy = make(true);
    let validNoisy = 0;
    for (let i = 0; i < n; i++) validNoisy += noisy.depth[i] > 0 ? 1 : 0;
    expect(count(depthHaloMask(coherent, image, cam, { radius: 0 }))).toBe(0);
    expect(count(depthHaloMask(noisy, image, cam, { radius: 0 }))).toBe(validNoisy);
  });
});

function popcount(v: number): number {
  let c = 0;
  let b = v >>> 0;
  while (b) {
    b &= b - 1;
    c++;
  }
  return c;
}
