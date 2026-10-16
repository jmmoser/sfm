/**
 * Depth-map clean-up before fusion: colour-guided removal of "halos" (foreground
 * fattening) next to depth discontinuities.
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
 * A halo pixel has the colour of one surface and the depth of another. For every valid
 * pixel p within `radius` pixels of a depth discontinuity (an invalid or off-plane
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
 * Measured on the GT-pose harness (480 px, PatchMatch geometric maps; fraction of valid
 * pixels rejected with the defaults): object 66 % of the > 5 % outliers, 1.2 % of the ≤ 1 %
 * pixels; yard 81 % of the sky estimates, 68 % of the > 5 % outliers, 0.9 % of the ≤ 1 %
 * pixels. ≈ 45 ms per 480×360 map on one x86 core (5 × 5 samples per examined pixel; the
 * 9 × 9 lattice of step 2 is 2–3× slower for the same result). Apply it ONCE per map: a second
 * pass sees the removed pixels as invalid and erodes further (`removeDepthHalos` on the final
 * maps, or `FusionOptions.haloFilter`, not both).
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
}

export const DEFAULT_HALO_FILTER_OPTIONS: Readonly<HaloFilterOptions> = Object.freeze({
  radius: 8,
  step: 4,
  maxRelDepth: 0.05,
  segmentColor: 10,
  voteFraction: 0.5,
  minSegmentSamples: 6,
});

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
  const W = dm.width;
  const H = dm.height;
  const n = W * H;
  if (e === 0 || n === 0) return null;
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
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const d = depth[i];
      if (!(d > 0 && d < Infinity)) continue;
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
  if (!anyEdge) return null;

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
  const rgb = rgbAtResolution(image, W, H);
  const col = new Int32Array(n);
  for (let i = 0; i < n; i++) col[i] = rgb[3 * i] | (rgb[3 * i + 1] << 8) | (rgb[3 * i + 2] << 16);
  const out = new Uint32Array((n + 31) >>> 5);
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
  return rejected > 0 ? out : null;
}

/** Copy of `dm` with the pixels of `depthHaloMask` invalidated (depth 0, normal 0, cost 2). */
export function removeDepthHalos(dm: DepthMap, image: RGBAImage, cam: CameraIntrinsics, options: Partial<HaloFilterOptions> = {}): DepthMap {
  const mask = depthHaloMask(dm, image, cam, options);
  const out: DepthMap = { width: dm.width, height: dm.height, depth: dm.depth.slice(), normal: dm.normal.slice(), cost: dm.cost.slice() };
  if (!mask) return out;
  const n = dm.width * dm.height;
  for (let i = 0; i < n; i++) {
    if (!((mask[i >>> 5] >>> (i & 31)) & 1)) continue;
    out.depth[i] = 0;
    out.normal[3 * i] = 0;
    out.normal[3 * i + 1] = 0;
    out.normal[3 * i + 2] = 0;
    out.cost[i] = 2;
  }
  return out;
}
