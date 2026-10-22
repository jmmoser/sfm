/**
 * Depth-map clean-up before fusion: colour-guided removal of "halos" (foreground
 * fattening) next to depth discontinuities, and of depths on the sky.
 *
 * Window-based matching (PatchMatch NCC over a (2r+1)² window) lets pixels of a weakly
 * textured background next to a textured foreground take the FOREGROUND's plane: a band
 * of several pixels around silhouettes (object in front of a plaster wall) and around roof
 * and tree outlines against the sky, widened further by plane propagation. The band lies on
 * the extension of the foreground plane, so it is consistent across views and survives
 * fusion's geometric checks; it shows up as background-coloured points floating next to
 * every silhouette (synthetic yard, `minViews` 2: 38 % of the points farther than τ from the
 * surface were sky-coloured).
 *
 * 1. Halo vote. A halo pixel has the colour of one surface and the depth of another. For every
 * valid pixel p within `radius` pixels of a depth discontinuity (an invalid or off-plane
 * 4-neighbour), the window samples of p's colour (mean |ΔR|,|ΔG|,|ΔB| < `segmentColor`)
 * vote: a sample is on p's plane when |d_q − z_p(q)| ≤ maxRelDepth · d_q, where z_p(q) is
 * the depth of p's plane (from its normal) on q's ray, and off-plane when invalid or
 * farther / nearer. p is rejected when more than `voteFraction` of at least
 * `minSegmentSamples` same-coloured samples are off its plane. Pixels of a textured surface
 * keep their same-coloured neighbours on their plane; same-coloured surfaces at different
 * depths are judged by the side p's colour segment mostly lies on. Invalid samples count as
 * off-plane (the sky beyond a roof halo is often invalid), so the rim of an invalid patch of
 * the same colour (a low-texture hole) is eroded where most of the window is invalid. Only
 * pixels within `radius` of a discontinuity are examined.
 *
 * 2. Sky test (`skyMaxStd` > 0). Halos wider than the vote window survive step 1: inside a
 * band, most same-coloured samples are band pixels on the same extended plane. The sky has
 * no texture of its own, so every depth on it comes from propagation: (a) the texture of each
 * pixel's colour segment (`segmentTextureSampler`: grey std of the same-coloured window
 * samples, so a sky pixel next to a roof measures the sky, not the roof edge) marks
 * untextured pixels; (b) untextured regions connected to the image border(s) facing UP through
 * untextured, similar-coloured 4-neighbours (`borderComponents`) are sky candidates. Without
 * `skyUp` the image is taken as upright (top row). Frames rotated to a session's canonical
 * orientation, or captured under portrait lock, are not upright: with `skyUp` (the up direction
 * in the camera frame, `skyBorders`) the borders within 60° of the projected up direction are
 * used. (c) A candidate region is sky when fewer than `skyMinCoherence` of its
 * pixels carry a coherent depth (valid, with ≥ 3 of 4 neighbours valid and on the pixel's
 * plane): PatchMatch fills the sky with random, locally incoherent depths (or none), while a
 * smooth wall gets one consistent plane. All valid depths of a sky region are rejected, halo
 * bands included. Measured (GT-pose harness, PatchMatch geometric maps, untextured = std < 2):
 * yard sky regions have coherence 0.1–0.5 (all of their area), the object room's
 * top-connected walls 0.5–1 (99.5 % of their area). Coherence is per region AREA, not per valid
 * depth: a sky PatchMatch leaves mostly invalid would otherwise look coherent (its only depths
 * are the halo bands).
 * Limits of the sky test: the thresholds come from synthetic renders (sky std ≈ 1 grey level);
 * denoised iPhone frames put both sky and blank walls below 1, so the coherence test does the
 * separating — untested on device. With simulated phone artefacts (object-real) 19 % of the
 * top-connected wall area fell below coherence 0.5 (its depths were mostly wrong anyway); if real
 * walls keep good but less coherent depths, lower `skyMinCoherence`. A blank wall touching the
 * up border that PatchMatch fills incoherently is removed as sky. Sky not connected to the up
 * border (through foliage), dark or textured (cloudy) skies are untouched.
 *
 * Measured on the GT-pose harness (480 px; fraction of valid pixels rejected by step 1 with
 * the defaults): object 66 % of the > 5 % outliers, 1.2 % of the ≤ 1 % pixels; yard 81 % of
 * the sky estimates, 68 % of the > 5 % outliers, 0.9 % of the ≤ 1 % pixels. The sky test
 * then removes 80–90 % of the sky-coloured points farther than τ that survive fusion on the
 * yard (precision 0.981 → 0.985, completeness −0.03 pt) and changes nothing on the object
 * (docs/research/dense-tuning.md "Sky / low-texture halo removal"). ≈ 40 ms per 480×360 map for step 1 on
 * one x86 core, + 5–30 ms for the sky test (proportional to the top-connected untextured
 * area). Apply it ONCE per map: a second pass sees the removed pixels as invalid and erodes
 * further (`removeDepthHalos` on the final maps, or `FusionOptions.haloFilter`, not both).
 */
import type { CameraIntrinsics, RGBAImage } from '../core/types';
import { rgbAtResolution, scaleCamera } from './dense-util';
import type { DepthMap } from './types';

export interface HaloFilterOptions {
  /** Window radius in depth-map pixels (≥ the halo width: ~PatchMatch window radius + propagation). 0 = off. Default 8. */
  radius: number;
  /** Window sampling step (pixels; 5 × 5 samples with the default radius). Default 4. */
  step: number;
  /** Relative deviation from p's plane that counts as a different surface. Default 0.05. */
  maxRelDepth: number;
  /** Colour distance (mean abs RGB, 0..255) below which a sample has p's colour. Default 10. */
  segmentColor: number;
  /** Reject when more than this fraction of the same-coloured samples is off p's plane. Default 0.5. */
  voteFraction: number;
  /** Minimum number of same-coloured samples for a decision. Default 6. */
  minSegmentSamples: number;
  /**
   * Sky test: grey-level std (0..255) of a pixel's colour segment (`segmentTextureSampler`) below
   * which the pixel counts as untextured. 0 = sky test off. Default 3.
   */
  skyMaxStd: number;
  /**
   * A top-connected untextured region is sky when fewer than this fraction of its pixels carry
   * a coherent depth (valid, ≥ 3 of 4 neighbours valid and on the pixel's plane). Default 0.5.
   */
  skyMinCoherence: number;
  /** Window radius / lattice step / minimum same-coloured samples of the texture measure. Defaults 5 / 2 / 8. */
  skyTextureRadius: number;
  skyTextureStep: number;
  skyTextureMinSamples: number;
  /**
   * Up direction (opposite gravity) in the map's camera frame (+x right, +y down, +z forward),
   * e.g. `worldUpInCamera(pose.R)` for a Z-up world or −gravityCam. It selects the image borders where
   * sky regions are seeded (`skyBorders`). null = upright image (top row). Default null.
   * Per view: pass it to `removeDepthHalos` / `depthHaloMask`, not to `FusionOptions.haloFilter`
   * (one options object for every view).
   */
  skyUp: ArrayLike<number> | null;
}

export const DEFAULT_HALO_FILTER_OPTIONS: Readonly<HaloFilterOptions> = Object.freeze({
  radius: 8,
  step: 4,
  maxRelDepth: 0.05,
  segmentColor: 10,
  voteFraction: 0.5,
  minSegmentSamples: 6,
  skyMaxStd: 3,
  skyMinCoherence: 0.5,
  skyTextureRadius: 5,
  skyTextureStep: 2,
  skyTextureMinSamples: 8,
  skyUp: null,
});

/**
 * Image borders where the sky test seeds its regions, for the up direction `up` in the camera
 * frame (bit 0 top, 1 bottom, 2 left, 3 right; the `borderComponents` bits):
 * - null / undefined / not finite: 1 (upright image, top row);
 * - camera within ~15° of looking straight up (in-plane part of the unit `up` < 0.25, up·z > 0): all borders;
 * - camera within ~15° of looking straight down: 0 (no sky test);
 * - otherwise the borders whose outward direction is within 60° of `up` projected onto the image
 *   plane (one border, or the two of a corner when the camera is rolled by ≈ 30–60°).
 */
export function skyBorders(up: ArrayLike<number> | null | undefined): number {
  if (!up || up.length < 3) return 1;
  const ux = up[0];
  const uy = up[1];
  const uz = up[2];
  const len = Math.hypot(ux, uy, uz);
  if (!(len > 0 && len < Infinity)) return 1;
  const s = Math.hypot(ux, uy) / len;
  if (s < 0.25) return uz > 0 ? 15 : 0;
  const dx = ux / (s * len);
  const dy = uy / (s * len);
  return (-dy > 0.5 ? 1 : 0) | (dy > 0.5 ? 2 : 0) | (-dx > 0.5 ? 4 : 0) | (dx > 0.5 ? 8 : 0);
}

/** World up `worldUp` (default +Z, §2) in the camera frame of a world-to-camera rotation `R` (row-major): R · worldUp. */
export function worldUpInCamera(R: ArrayLike<number>, worldUp: ArrayLike<number> = [0, 0, 1]): [number, number, number] {
  const [a, b, c] = [worldUp[0], worldUp[1], worldUp[2]];
  return [R[0] * a + R[1] * b + R[2] * c, R[3] * a + R[4] * b + R[5] * c, R[6] * a + R[7] * b + R[8] * c];
}

/**
 * Halo mask of one depth map: bit i (Uint32 words, 1 bit per pixel, the layout of fusion's
 * masks) is set for rejected pixels; null when nothing is rejected. `cam` = the view's
 * intrinsics at any resolution (rescaled to the depth-map grid per §2); `image` may have a
 * different resolution (box-filtered to the depth-map grid).
 */
export function depthHaloMask(dm: DepthMap, image: RGBAImage, cam: CameraIntrinsics, options: Partial<HaloFilterOptions> = {}): Uint32Array | null {
  const o: HaloFilterOptions = { ...DEFAULT_HALO_FILTER_OPTIONS, ...options };
  const e = Math.max(0, Math.floor(o.radius));
  const st = Math.max(1, Math.floor(o.step));
  const skyB = o.skyMaxStd > 0 && o.skyMinCoherence > 0 ? skyBorders(o.skyUp) : 0;
  const skyOn = skyB !== 0;
  const W = dm.width;
  const H = dm.height;
  const n = W * H;
  if ((e === 0 && !skyOn) || n === 0) return null;
  const depth = dm.depth;
  const normal = dm.normal;
  const c = scaleCamera(cam, W, H);
  const invF = 1 / c.f;
  const t = o.maxRelDepth;

  // Plane of pixel i on the ray through q: n·r(q) = a + b·x_q + c·y_q, z_i(q) = d_i·nrp/n·r(q).
  // Per pixel: (a, b, c, nrp); nrp = 0 marks "no usable plane" (z_i(q) = d_i).
  const pa = new Float32Array(n);
  const pb = new Float32Array(n);
  const pc = new Float32Array(n);
  const pn = new Float32Array(n);
  let anyValid = false;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const d = depth[i];
      if (!(d > 0 && d < Infinity)) continue;
      anyValid = true;
      const nx = normal[3 * i];
      const ny = normal[3 * i + 1];
      const nz = normal[3 * i + 2];
      if (!(nx * nx + ny * ny + nz * nz > 0.5)) continue;
      const b = nx * invF;
      const cc = ny * invF;
      const a = nz - b * c.cx - cc * c.cy;
      const nrp = a + b * x + cc * y;
      if (!(nrp < 0)) continue;
      pa[i] = a;
      pb[i] = b;
      pc[i] = cc;
      pn[i] = nrp;
    }
  }
  if (!anyValid) return null;
  /** Whether depth dq at (xq, yq) lies on pixel i's plane (within t, relative to dq). */
  const onPlane = (i: number, dq: number, xq: number, yq: number): boolean => {
    const d = depth[i];
    const nrp = pn[i];
    if (nrp < 0) {
      const den = pa[i] + pb[i] * xq + pc[i] * yq;
      // plane usable on q's ray: not edge-on, predicted depth within (0.5, 2)·d
      if (den < -0.05 && nrp < 0.5 * den && nrp > 2 * den) {
        // |dq − d·nrp/den| ≤ t·dq, multiplied by −den > 0
        return Math.abs(dq * den - d * nrp) <= -t * dq * den;
      }
    }
    return Math.abs(dq - d) <= t * dq;
  };
  const rgb = rgbAtResolution(image, W, H);
  const out = new Uint32Array((n + 31) >>> 5);
  let rejected = 0;
  if (e > 0) rejected += haloVote(dm, rgb, o, e, st, onPlane, pa, pb, pc, pn, out);
  if (skyOn) rejected += skyTest(dm, rgb, o, skyB, onPlane, out);
  return rejected > 0 ? out : null;
}

type PlaneTest = (i: number, dq: number, xq: number, yq: number) => boolean;

/** Step 1 (file header): same-colour plane vote near discontinuities; sets bits in `out`, returns the count. */
function haloVote(
  dm: DepthMap,
  rgb: Uint8Array,
  o: HaloFilterOptions,
  e: number,
  st: number,
  onPlane: PlaneTest,
  pa: Float32Array,
  pb: Float32Array,
  pc: Float32Array,
  pn: Float32Array,
  out: Uint32Array,
): number {
  const W = dm.width;
  const H = dm.height;
  const n = W * H;
  const depth = dm.depth;
  const t = o.maxRelDepth;
  // 1. discontinuity pixels: valid with an invalid or off-plane 4-neighbour
  const edge = new Uint8Array(n);
  let anyEdge = false;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const d = depth[i];
      if (!(d > 0 && d < Infinity)) continue;
      for (let k = 0; k < 4; k++) {
        const xx = k === 0 ? x - 1 : k === 1 ? x + 1 : x;
        const yy = k === 2 ? y - 1 : k === 3 ? y + 1 : y;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        const dj = depth[yy * W + xx];
        if (!(dj > 0 && dj < Infinity) || !onPlane(i, dj, xx, yy)) {
          edge[i] = 1;
          anyEdge = true;
          break;
        }
      }
    }
  }
  if (!anyEdge) return 0;

  // 2. pixels within e of a discontinuity (Chebyshev dilation, separable)
  const tmp = new Uint8Array(n);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    let last = -1e9;
    for (let x = 0; x < W; x++) {
      if (edge[row + x]) last = x;
      if (x - last <= e) tmp[row + x] = 1;
    }
    last = 1e9;
    for (let x = W - 1; x >= 0; x--) {
      if (edge[row + x]) last = x;
      if (last - x <= e) tmp[row + x] = 1;
    }
  }
  const near = edge; // reused
  near.fill(0);
  for (let x = 0; x < W; x++) {
    let last = -1e9;
    for (let y = 0; y < H; y++) {
      if (tmp[y * W + x]) last = y;
      if (y - last <= e) near[y * W + x] = 1;
    }
    last = 1e9;
    for (let y = H - 1; y >= 0; y--) {
      if (tmp[y * W + x]) last = y;
      if (last - y <= e) near[y * W + x] = 1;
    }
  }

  // 3. same-colour vote (hot loop: plane test inlined, colours packed)
  const col = new Int32Array(n);
  for (let i = 0; i < n; i++) col[i] = rgb[3 * i] | (rgb[3 * i + 1] << 8) | (rgb[3 * i + 2] << 16);
  const segC = 3 * o.segmentColor;
  const minS = Math.max(1, o.minSegmentSamples);
  const frac = o.voteFraction;
  let rejected = 0;
  for (let y = 0; y < H; y++) {
    // sampling lattice aligned with p: offsets are multiples of the step
    const y0 = y - Math.floor(Math.min(e, y) / st) * st;
    const y1 = Math.min(H - 1, y + e);
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!near[i]) continue;
      const d = depth[i];
      if (!(d > 0 && d < Infinity)) continue;
      const x0 = x - Math.floor(Math.min(e, x) / st) * st;
      const x1 = Math.min(W - 1, x + e);
      const ci = col[i];
      const r = ci & 255;
      const g = (ci >>> 8) & 255;
      const b = ci >>> 16;
      const nrp = pn[i];
      const ia = pa[i];
      const ib = pb[i];
      const ic = pc[i];
      let on = 0;
      let off = 0;
      for (let yy = y0; yy <= y1; yy += st) {
        const row = yy * W;
        const ay = ia + ic * yy;
        for (let xx = x0; xx <= x1; xx += st) {
          const j = row + xx;
          const cj = col[j];
          if (Math.abs((cj & 255) - r) + Math.abs(((cj >>> 8) & 255) - g) + Math.abs((cj >>> 16) - b) >= segC || j === i) continue;
          const dj = depth[j];
          if (!(dj > 0 && dj < Infinity)) {
            off++;
            continue;
          }
          const den = ay + ib * xx;
          let ok: boolean;
          if (nrp < 0 && den < -0.05 && nrp < 0.5 * den && nrp > 2 * den) ok = Math.abs(dj * den - d * nrp) <= -t * dj * den;
          else ok = Math.abs(dj - d) <= t * dj;
          if (ok) on++;
          else off++;
        }
      }
      if (on + off >= minS && off > frac * (on + off)) {
        out[i >>> 5] |= 1 << (i & 31);
        rejected++;
      }
    }
  }
  return rejected;
}

/** Step 2 (file header): rejects every valid depth of the incoherent untextured regions connected to `borders`. */
function skyTest(dm: DepthMap, rgb: Uint8Array, o: HaloFilterOptions, borders: number, onPlane: PlaneTest, out: Uint32Array): number {
  const W = dm.width;
  const H = dm.height;
  const depth = dm.depth;
  const std = segmentTextureSampler(rgb, W, H, o.skyTextureRadius, o.skyTextureStep, o.segmentColor, Math.max(1, o.skyTextureMinSamples));
  const maxStd = o.skyMaxStd;
  // flood fill from the up-facing border(s); each pixel's texture is evaluated at most once, on
  // the way (memo: 0 unknown, 1 untextured, 2 textured)
  const memo = new Uint8Array(W * H);
  const low = (i: number): boolean => {
    if (!memo[i]) memo[i] = std(i) < maxStd ? 1 : 2;
    return memo[i] === 1;
  };
  const { labels, count } = borderComponents(low, rgb, W, H, o.segmentColor, borders);
  if (count === 0) return 0;
  const area = new Float64Array(count + 1);
  const coherent = new Float64Array(count + 1);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const l = labels[i];
      if (!l) continue;
      area[l]++;
      const d = depth[i];
      if (!(d > 0 && d < Infinity)) continue;
      let agree = 0;
      for (let k = 0; k < 4; k++) {
        const xx = k === 0 ? x - 1 : k === 1 ? x + 1 : x;
        const yy = k === 2 ? y - 1 : k === 3 ? y + 1 : y;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        const dj = depth[yy * W + xx];
        if (dj > 0 && dj < Infinity && onPlane(i, dj, xx, yy)) agree++;
      }
      if (agree >= 3) coherent[l]++;
    }
  }
  const sky = new Uint8Array(count + 1);
  let anySky = false;
  for (let l = 1; l <= count; l++) {
    if (coherent[l] < o.skyMinCoherence * area[l]) {
      sky[l] = 1;
      anySky = true;
    }
  }
  if (!anySky) return 0;
  let rejected = 0;
  for (let i = 0; i < W * H; i++) {
    if (!sky[labels[i]]) continue;
    const d = depth[i];
    if (!(d > 0 && d < Infinity) || (out[i >>> 5] >>> (i & 31)) & 1) continue;
    out[i >>> 5] |= 1 << (i & 31);
    rejected++;
  }
  return rejected;
}

/** Copy of `dm` with the pixels of `depthHaloMask` invalidated (depth 0, normal 0, cost 2). */
export function removeDepthHalos(dm: DepthMap, image: RGBAImage, cam: CameraIntrinsics, options: Partial<HaloFilterOptions> = {}): DepthMap {
  const mask = depthHaloMask(dm, image, cam, options);
  const out: DepthMap = { width: dm.width, height: dm.height, depth: dm.depth.slice(), normal: dm.normal.slice(), cost: dm.cost.slice() };
  if (mask) applyDepthHaloMask(out, mask);
  return out;
}

/**
 * Invalidate the pixels of a `depthHaloMask` bit mask IN PLACE (depth 0, normal 0, cost 2) — what
 * `removeDepthHalos` does to its copy, for a mask computed elsewhere (a CPU helper worker).
 */
export function applyDepthHaloMask(dm: DepthMap, mask: Uint32Array): void {
  const n = dm.width * dm.height;
  for (let i = 0; i < n; i++) {
    if (!((mask[i >>> 5] >>> (i & 31)) & 1)) continue;
    dm.depth[i] = 0;
    dm.normal[3 * i] = 0;
    dm.normal[3 * i + 1] = 0;
    dm.normal[3 * i + 2] = 0;
    dm.cost[i] = 2;
  }
}

/**
 * Per-pixel texture of the pixel's colour segment, evaluated on demand: `std(i)` is the
 * grey-level standard deviation (0..255, grey = (R + 2G + B) / 4) of the window samples (radius
 * `radius`, lattice `step` aligned with the pixel, centre included) whose colour is within
 * `segmentColor` of the centre pixel (mean |ΔR|,|ΔG|,|ΔB|). Unlike the bilateral variance
 * PatchMatch tests (`minVariance`), samples of other colours have NO weight, so a sky pixel next
 * to a roof edge measures the sky's texture, not the edge's. Infinity when fewer than
 * `minSamples` samples share the centre's colour (an isolated colour counts as texture).
 * `rgb` = packed RGB at W × H (`rgbAtResolution`).
 */
export function segmentTextureSampler(rgb: Uint8Array, W: number, H: number, radius: number, step: number, segmentColor: number, minSamples: number): (i: number) => number {
  const n = W * H;
  const e = Math.max(1, Math.floor(radius));
  const st = Math.max(1, Math.floor(step));
  // 4 × grey (R + 2G + B) in 2 B/px; colours are compared on `rgb` directly (no extra copies)
  const grey4 = new Uint16Array(n);
  for (let i = 0; i < n; i++) grey4[i] = rgb[3 * i] + 2 * rgb[3 * i + 1] + rgb[3 * i + 2];
  const segC = 3 * segmentColor;
  return (i: number): number => {
    const x = i % W;
    const y = (i - x) / W;
    const y0 = y - Math.floor(Math.min(e, y) / st) * st;
    const y1 = Math.min(H - 1, y + e);
    const x0 = x - Math.floor(Math.min(e, x) / st) * st;
    const x1 = Math.min(W - 1, x + e);
    const r = rgb[3 * i];
    const g = rgb[3 * i + 1];
    const b = rgb[3 * i + 2];
    const g0 = grey4[i];
    let m = 0;
    let s1 = 0;
    let s2 = 0;
    for (let yy = y0; yy <= y1; yy += st) {
      const row = yy * W;
      for (let xx = x0; xx <= x1; xx += st) {
        const j = row + xx;
        const j3 = 3 * j;
        if (Math.abs(rgb[j3] - r) + Math.abs(rgb[j3 + 1] - g) + Math.abs(rgb[j3 + 2] - b) >= segC) continue;
        const d = grey4[j] - g0;
        m++;
        s1 += d;
        s2 += d * d;
      }
    }
    return m < minSamples ? Infinity : 0.25 * Math.sqrt(Math.max(0, s2 / m - (s1 / m) * (s1 / m)));
  };
}

/** `segmentTextureSampler` evaluated at every pixel (Float32Array of W × H). */
export function segmentTextureStd(rgb: Uint8Array, W: number, H: number, radius: number, step: number, segmentColor: number, minSamples: number): Float32Array {
  const f = segmentTextureSampler(rgb, W, H, radius, step, segmentColor, minSamples);
  const out = new Float32Array(W * H);
  for (let i = 0; i < out.length; i++) out[i] = f(i);
  return out;
}

/**
 * Pixels of `low` (1 = low texture, or a predicate evaluated at most once per pixel on the way)
 * connected to the image border rows/columns selected by
 * `borders` (bit 0 top, 1 bottom, 2 left, 3 right) through 4-neighbours of `low` whose colours
 * differ by less than `segmentColor` (mean abs RGB) — e.g. the sky region of an upright image
 * (top border). Returns a 0/1 mask.
 */
export function borderConnectedMask(low: Uint8Array | ((i: number) => boolean), rgb: Uint8Array, W: number, H: number, segmentColor: number, borders = 1): Uint8Array {
  const { labels } = borderComponents(low, rgb, W, H, segmentColor, borders);
  return Uint8Array.from(labels, (l) => (l > 0 ? 1 : 0));
}

/**
 * Connected components of `borderConnectedMask`: `labels[i]` = 1..count for the pixels of
 * component k (one flood fill per unvisited border seed), 0 elsewhere.
 */
export function borderComponents(low: Uint8Array | ((i: number) => boolean), rgb: Uint8Array, W: number, H: number, segmentColor: number, borders = 1): { labels: Uint16Array | Int32Array; count: number } {
  const isLow = typeof low === 'function' ? low : (i: number): boolean => low[i] === 1;
  const n = W * H;
  // at most one component per border seed: 16-bit labels whenever they fit (2 B/px)
  const labels = 2 * (W + H) < 65535 ? new Uint16Array(n) : new Int32Array(n);
  const queue = new Int32Array(n);
  const segC = 3 * segmentColor;
  let count = 0;
  const fill = (s: number): void => {
    if (labels[s] || !isLow(s)) return;
    const lab = ++count;
    let qh = 0;
    let qt = 0;
    labels[s] = lab;
    queue[qt++] = s;
    while (qh < qt) {
      const i = queue[qh++];
      const x = i % W;
      const y = (i - x) / W;
      for (let k = 0; k < 4; k++) {
        const xx = k === 0 ? x - 1 : k === 1 ? x + 1 : x;
        const yy = k === 2 ? y - 1 : k === 3 ? y + 1 : y;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        const j = yy * W + xx;
        if (labels[j]) continue;
        if (Math.abs(rgb[3 * i] - rgb[3 * j]) + Math.abs(rgb[3 * i + 1] - rgb[3 * j + 1]) + Math.abs(rgb[3 * i + 2] - rgb[3 * j + 2]) >= segC) continue;
        if (!isLow(j)) continue;
        labels[j] = lab;
        queue[qt++] = j;
      }
    }
  };
  if (borders & 1) for (let x = 0; x < W; x++) fill(x);
  if (borders & 2) for (let x = 0; x < W; x++) fill((H - 1) * W + x);
  if (borders & 4) for (let y = 0; y < H; y++) fill(y * W);
  if (borders & 8) for (let y = 0; y < H; y++) fill(y * W + W - 1);
  return { labels, count };
}
