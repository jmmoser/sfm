/**
 * Per-view depth-map evaluation against GT depth + debug panels for the dense GT harness.
 */
import type { CameraIntrinsics, RGBAImage } from '../../core/types';
import type { DepthMap } from '../../dense/types';

export interface DepthEval {
  /** GT pixels (inside the mask). */
  gtPixels: number;
  /** Fraction of GT pixels with a valid estimate. */
  valid: number;
  /** Among valid: |Δz|/z ≤ 0.5 %, ≤ 1 %, > 5 %. */
  within05: number;
  within1: number;
  outlier5: number;
  /** Among valid: |Δz| ≤ tau (absolute, the dense-metric threshold). */
  withinTau: number;
  /** valid × within1 (fraction of GT pixels with an accurate estimate). */
  completeness1: number;
  medianRel: number;
  p90Rel: number;
  medianNormalDeg: number;
  /** Valid estimates where the GT has no surface (sky / outside), as a fraction of all valid estimates (no mask). */
  validWithoutGt: number;
}

function quant(a: Float32Array, n: number, q: number): number {
  if (n === 0) return NaN;
  const s = a.subarray(0, n).sort();
  return s[Math.min(n - 1, Math.floor(q * n))];
}

const relScratch = { a: new Float32Array(0), b: new Float32Array(0) };

/** Compare `dm` (any map with depth/normal) with GT depth (+ normals) inside an optional mask. */
export function evalDepth(
  dm: Pick<DepthMap, 'depth' | 'normal'>,
  gtDepth: Float32Array,
  gtNormal: Float32Array | null,
  tau: number,
  mask?: Uint8Array | null,
): DepthEval {
  const n = gtDepth.length;
  if (relScratch.a.length < n) {
    relScratch.a = new Float32Array(n);
    relScratch.b = new Float32Array(n);
  }
  const rel = relScratch.a;
  const ang = relScratch.b;
  let gt = 0, valid = 0, w05 = 0, w1 = 0, o5 = 0, wt = 0, nr = 0, na = 0, validAll = 0, validNoGt = 0;
  for (let i = 0; i < n; i++) {
    const z = dm.depth[i];
    const zg = gtDepth[i];
    if (z > 0) {
      validAll++;
      if (!(zg > 0)) validNoGt++;
    }
    if (!(zg > 0)) continue;
    if (mask && !mask[i]) continue;
    gt++;
    if (!(z > 0)) continue;
    valid++;
    const e = Math.abs(z - zg);
    const r = e / zg;
    rel[nr++] = r;
    if (r <= 0.005) w05++;
    if (r <= 0.01) w1++;
    if (r > 0.05) o5++;
    if (e <= tau) wt++;
    if (gtNormal) {
      const gx = gtNormal[3 * i], gy = gtNormal[3 * i + 1], gz = gtNormal[3 * i + 2];
      if (gx || gy || gz) {
        const d = dm.normal[3 * i] * gx + dm.normal[3 * i + 1] * gy + dm.normal[3 * i + 2] * gz;
        ang[na++] = (Math.acos(Math.max(-1, Math.min(1, d))) * 180) / Math.PI;
      }
    }
  }
  return {
    gtPixels: gt,
    valid: gt ? valid / gt : 0,
    within05: valid ? w05 / valid : 0,
    within1: valid ? w1 / valid : 0,
    outlier5: valid ? o5 / valid : 0,
    withinTau: valid ? wt / valid : 0,
    completeness1: gt ? w1 / gt : 0,
    medianRel: quant(rel, nr, 0.5),
    p90Rel: quant(rel, nr, 0.9),
    medianNormalDeg: quant(ang, na, 0.5),
    validWithoutGt: validAll ? validNoGt / validAll : 0,
  };
}

/**
 * "Well-posed surface" mask: GT present, ≥ `edgeRadius` px from depth discontinuities
 * (relative jump > `relJump` between 4-neighbours, or surface / no surface) and viewing
 * angle < `maxAngleDeg` (same definition as `synthetic.surfaceMask`, GT normals may be 0).
 */
export function surfaceMask(cam: CameraIntrinsics, gtDepth: Float32Array, gtNormal: Float32Array | null, edgeRadius = 5, relJump = 0.05, maxAngleDeg = 75): Uint8Array {
  const W = cam.width;
  const H = cam.height;
  const edge = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const z = gtDepth[i];
      if (x + 1 < W) {
        const z2 = gtDepth[i + 1];
        if (z > 0 !== z2 > 0 || Math.abs(z - z2) > relJump * Math.min(z, z2)) edge[i] = edge[i + 1] = 1;
      }
      if (y + 1 < H) {
        const z2 = gtDepth[i + W];
        if (z > 0 !== z2 > 0 || Math.abs(z - z2) > relJump * Math.min(z, z2)) edge[i] = edge[i + W] = 1;
      }
    }
  }
  // distance transform-free dilation: separable max filter
  const tmp = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let m = 0;
      for (let d = -edgeRadius; d <= edgeRadius && !m; d++) {
        const xx = x + d;
        if (xx >= 0 && xx < W) m = edge[y * W + xx];
      }
      tmp[y * W + x] = m;
    }
  }
  const cosMax = Math.cos((maxAngleDeg * Math.PI) / 180);
  const mask = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!(gtDepth[i] > 0)) continue;
      let m = 0;
      for (let d = -edgeRadius; d <= edgeRadius && !m; d++) {
        const yy = y + d;
        if (yy >= 0 && yy < H) m = tmp[yy * W + x];
      }
      if (m) continue;
      if (gtNormal) {
        const nx = gtNormal[3 * i], ny = gtNormal[3 * i + 1], nz = gtNormal[3 * i + 2];
        if (nx || ny || nz) {
          const rx = (x - cam.cx) / cam.f, ry = (y - cam.cy) / cam.f;
          const l = Math.hypot(rx, ry, 1);
          const c = -(nx * rx + ny * ry + nz) / l;
          if (c < cosMax) continue;
        }
      }
      mask[i] = 1;
    }
  }
  return mask;
}

// ------------------------------------------------------------------ visualisation

/** Turbo colormap (Google, polynomial approximation), t in [0, 1] → RGB 0..255. */
export function turbo(t: number, out: number[]): void {
  t = Math.max(0, Math.min(1, t));
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  out[0] = Math.round(255 * Math.max(0, Math.min(1, r)));
  out[1] = Math.round(255 * Math.max(0, Math.min(1, g)));
  out[2] = Math.round(255 * Math.max(0, Math.min(1, b)));
}

/** Depth → turbo (log scale over [zMin, zMax]); 0 = black. */
export function depthToRGBA(depth: Float32Array, W: number, H: number, zMin: number, zMax: number): RGBAImage {
  const data = new Uint8ClampedArray(4 * W * H);
  const l0 = Math.log(zMin);
  const l1 = Math.log(zMax);
  const c = [0, 0, 0];
  for (let i = 0; i < W * H; i++) {
    const z = depth[i];
    data[4 * i + 3] = 255;
    if (!(z > 0)) continue;
    turbo((Math.log(z) - l0) / (l1 - l0), c);
    data[4 * i] = c[0];
    data[4 * i + 1] = c[1];
    data[4 * i + 2] = c[2];
  }
  return { width: W, height: H, data };
}

/**
 * Error map: green ≤ 1 %, yellow ≤ 5 %, red > 5 % relative error; blue = estimate where the
 * GT has no surface; dark grey = GT surface without estimate; black = neither.
 */
export function errorToRGBA(depth: Float32Array, gt: Float32Array, W: number, H: number): RGBAImage {
  const data = new Uint8ClampedArray(4 * W * H);
  for (let i = 0; i < W * H; i++) {
    const z = depth[i];
    const zg = gt[i];
    let r = 0, g = 0, b = 0;
    if (z > 0 && zg > 0) {
      const e = Math.abs(z - zg) / zg;
      if (e <= 0.01) g = 200;
      else if (e <= 0.05) {
        r = 230;
        g = 200;
      } else r = 230;
    } else if (z > 0) b = 230;
    else if (zg > 0) r = g = b = 60;
    data[4 * i] = r;
    data[4 * i + 1] = g;
    data[4 * i + 2] = b;
    data[4 * i + 3] = 255;
  }
  return { width: W, height: H, data };
}

/** Cost (0..2) → grey (0 = white). */
export function costToRGBA(cost: Float32Array, depth: Float32Array, W: number, H: number): RGBAImage {
  const data = new Uint8ClampedArray(4 * W * H);
  for (let i = 0; i < W * H; i++) {
    const v = depth[i] > 0 ? Math.round(255 * (1 - Math.min(1, cost[i]))) : 0;
    data[4 * i] = v;
    data[4 * i + 1] = depth[i] > 0 ? v : 0;
    data[4 * i + 2] = depth[i] > 0 ? v : 90;
    data[4 * i + 3] = 255;
  }
  return { width: W, height: H, data };
}

/** Tile images (same size) into a grid with labels; returns a PNG blob. */
export async function panelPNG(tiles: { img: RGBAImage; label: string }[], cols: number): Promise<Blob> {
  const W = tiles[0].img.width;
  const H = tiles[0].img.height;
  const rows = Math.ceil(tiles.length / cols);
  const pad = 4;
  const canvas = new OffscreenCanvas(cols * (W + pad) + pad, rows * (H + pad + 18) + pad);
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#202020';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.font = '13px sans-serif';
  tiles.forEach((t, k) => {
    const x = pad + (k % cols) * (W + pad);
    const y = pad + Math.floor(k / cols) * (H + pad + 18);
    const px = new Uint8ClampedArray(4 * W * H);
    px.set(t.img.data.subarray(0, 4 * W * H));
    const id = new ImageData(px, W, H);
    ctx.putImageData(id, x, y + 18);
    ctx.fillStyle = '#eee';
    ctx.fillText(t.label, x + 2, y + 13);
  });
  return canvas.convertToBlob({ type: 'image/png' });
}
