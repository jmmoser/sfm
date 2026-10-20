// Image metrics of the novel-view evaluation (e2e/nvs.mjs): PSNR, SSIM, per-view affine colour
// fit, coverage, alignment shift. Pure JS without Node imports: unit-tested in Node
// (src/harness/nvs/nvs-lib.test.ts) and imported by the browser page (src/harness/nvs/main.ts).
//
// Images are RGBA8 (Uint8Array / Uint8ClampedArray), rows top to bottom, w × h.
//  - Ground truth: alpha 255 = valid pixel; anything else (outside the undistorted / rolling-shutter
//    corrected source) is excluded from every metric.
//  - Render: PREMULTIPLIED alpha (what a WebGL canvas with a transparent clear reads back; MSAA
//    edges carry partial alpha). The full-frame image is the render composited over the
//    background colour; a pixel is "covered" when alpha ≥ 128, and its covered-pixel colour is
//    the un-premultiplied one.
//
// PSNR: 10·log10(255² / MSE), MSE over the 3 colour channels of the pixels of a mask, 8-bit sRGB.
// SSIM: Wang et al. 2004 with an 11 × 11 Gaussian window (σ = 1.5), K1 = 0.01, K2 = 0.03,
// L = 255, per RGB channel and averaged over the channels (the 3DGS / skimage convention). Local
// statistics are normalised convolutions over the mask (weights G·m / Σ G·m): the image border and,
// for the covered-pixel score, uncovered pixels do not enter any window; the SSIM map is averaged
// over the mask.

export const SSIM_RADIUS = 5;
export const SSIM_SIGMA = 1.5;
const C1 = (0.01 * 255) ** 2;
const C2 = (0.03 * 255) ** 2;

/** Normalised 1-D Gaussian of 2·radius + 1 taps. */
export function gaussianKernel(radius = SSIM_RADIUS, sigma = SSIM_SIGMA) {
  const k = new Float64Array(2 * radius + 1);
  let s = 0;
  for (let i = -radius; i <= radius; i++) {
    k[i + radius] = Math.exp(-(i * i) / (2 * sigma * sigma));
    s += k[i + radius];
  }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  return k;
}

/**
 * Separable convolution with a symmetric kernel and zero padding (out may not alias src; tmp is
 * scratch of w·h). The 11-tap interior (radius 5, the SSIM window) is unrolled and uses the
 * symmetry; borders and other radii take the generic path.
 */
function blurInto(src, w, h, k, tmp, out) {
  const r = (k.length - 1) >> 1;
  const kc = k[r];
  const ks = new Float64Array(r + 1);
  for (let d = 1; d <= r; d++) ks[d] = k[r + d];
  const fast = r === 5 && w > 2 * r && h > 2 * r;
  const [, k1, k2, k3, k4, k5] = fast ? ks : [0, 0, 0, 0, 0, 0];
  const hEdge = (row, x) => {
    const i = row + x;
    let s = kc * src[i];
    for (let d = 1; d <= r; d++) {
      if (x - d >= 0) s += ks[d] * src[i - d];
      if (x + d < w) s += ks[d] * src[i + d];
    }
    tmp[i] = s;
  };
  for (let y = 0; y < h; y++) {
    const row = y * w;
    if (!fast) {
      for (let x = 0; x < w; x++) hEdge(row, x);
      continue;
    }
    for (let x = 0; x < r; x++) hEdge(row, x);
    for (let i = row + r, e = row + w - r; i < e; i++) {
      tmp[i] =
        kc * src[i] +
        k1 * (src[i - 1] + src[i + 1]) +
        k2 * (src[i - 2] + src[i + 2]) +
        k3 * (src[i - 3] + src[i + 3]) +
        k4 * (src[i - 4] + src[i + 4]) +
        k5 * (src[i - 5] + src[i + 5]);
    }
    for (let x = w - r; x < w; x++) hEdge(row, x);
  }
  // vertical: whole rows (sequential memory access)
  for (let y = 0; y < h; y++) {
    const o = y * w;
    if (fast && y >= r && y + r < h) {
      const w1 = w,
        w2 = 2 * w,
        w3 = 3 * w,
        w4 = 4 * w,
        w5 = 5 * w;
      for (let i = o, e = o + w; i < e; i++) {
        out[i] =
          kc * tmp[i] +
          k1 * (tmp[i - w1] + tmp[i + w1]) +
          k2 * (tmp[i - w2] + tmp[i + w2]) +
          k3 * (tmp[i - w3] + tmp[i + w3]) +
          k4 * (tmp[i - w4] + tmp[i + w4]) +
          k5 * (tmp[i - w5] + tmp[i + w5]);
      }
      continue;
    }
    for (let x = 0; x < w; x++) out[o + x] = kc * tmp[o + x];
    for (let d = 1; d <= r; d++) {
      const kd = ks[d];
      if (y - d >= 0) for (let x = 0, u = (y - d) * w; x < w; x++) out[o + x] += kd * tmp[u + x];
      if (y + d < h) for (let x = 0, u = (y + d) * w; x < w; x++) out[o + x] += kd * tmp[u + x];
    }
  }
  return out;
}

const KERNEL = gaussianKernel();

/** Channel c of an RGBA8 image as Float32, times the mask weights. */
function maskedChannel(img, n, c, m) {
  const o = new Float32Array(n);
  for (let i = 0; i < n; i++) o[i] = m[i] * img[4 * i + c];
  return o;
}

/** Mask weights, their blur and the masked, blurred statistics of the reference image b per channel. */
function referenceStats(b, w, h, mask, tmp) {
  const n = w * h;
  const m = new Float32Array(n);
  for (let i = 0; i < n; i++) m[i] = mask ? (mask[i] ? 1 : 0) : 1;
  const W = blurInto(m, w, h, KERNEL, tmp, new Float32Array(n));
  let count = 0;
  for (let i = 0; i < n; i++) if (m[i] && W[i] > 1e-6) count++;
  // scored pixels and their 1 / Σ G·m
  const idx = new Int32Array(count);
  const iw = new Float64Array(count);
  for (let i = 0, j = 0; i < n; i++)
    if (m[i] && W[i] > 1e-6) {
      idx[j] = i;
      iw[j++] = 1 / W[i];
    }
  const y = [],
    my = [],
    syy = [];
  const buf = new Float32Array(n);
  for (let c = 0; c < 3; c++) {
    const yc = maskedChannel(b, n, c, m);
    y.push(yc);
    my.push(blurInto(yc, w, h, KERNEL, tmp, new Float32Array(n)));
    for (let i = 0; i < n; i++) buf[i] = yc[i] * yc[i];
    syy.push(blurInto(buf, w, h, KERNEL, tmp, new Float32Array(n)));
  }
  return { m, idx, iw, count, y, my, syy };
}

/**
 * Mean SSIM of two RGBA8 images over `mask` (Uint8Array, 1 = scored; null = every pixel), per RGB
 * channel then averaged. Windows use only masked pixels (normalised convolution). `cache` (a Map
 * owned by the caller, for one reference image b) keeps b's statistics per mask object, so scoring
 * several images against the same b and mask blurs b once.
 */
export function ssim(a, b, w, h, mask = null, cache = null) {
  const n = w * h;
  const tmp = new Float32Array(n);
  let ref = cache?.get(mask);
  if (!ref) {
    ref = referenceStats(b, w, h, mask, tmp);
    cache?.set(mask, ref);
  }
  const { m, idx, iw, count } = ref;
  if (count === 0) return NaN;
  const mx = new Float32Array(n),
    sxx = new Float32Array(n),
    sxy = new Float32Array(n),
    buf = new Float32Array(n);
  let total = 0;
  for (let c = 0; c < 3; c++) {
    const x = maskedChannel(a, n, c, m);
    const y = ref.y[c],
      my = ref.my[c],
      syy = ref.syy[c];
    blurInto(x, w, h, KERNEL, tmp, mx);
    for (let i = 0; i < n; i++) buf[i] = x[i] * x[i];
    blurInto(buf, w, h, KERNEL, tmp, sxx);
    for (let i = 0; i < n; i++) buf[i] = x[i] * y[i];
    blurInto(buf, w, h, KERNEL, tmp, sxy);
    let s = 0;
    for (let j = 0; j < count; j++) {
      const i = idx[j],
        q = iw[j];
      const ux = mx[i] * q,
        uy = my[i] * q;
      const vx = Math.max(0, sxx[i] * q - ux * ux),
        vy = Math.max(0, syy[i] * q - uy * uy),
        cv = sxy[i] * q - ux * uy;
      s += ((2 * ux * uy + C1) * (2 * cv + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2));
    }
    total += s / count;
  }
  return total / 3;
}

/** Mean squared error over the RGB channels of the masked pixels ({ mse, n } with n pixels). */
export function mse(a, b, n, mask = null) {
  let se = 0;
  let cnt = 0;
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    const j = 4 * i;
    const d0 = a[j] - b[j],
      d1 = a[j + 1] - b[j + 1],
      d2 = a[j + 2] - b[j + 2];
    se += d0 * d0 + d1 * d1 + d2 * d2;
    cnt++;
  }
  return { mse: cnt ? se / (3 * cnt) : NaN, n: cnt };
}

/** PSNR (dB, peak 255) from an MSE; Infinity for identical images. */
export function psnrFromMse(m) {
  if (!Number.isFinite(m)) return NaN;
  return m <= 0 ? Infinity : 10 * Math.log10((255 * 255) / m);
}

export function psnr(a, b, n, mask = null) {
  return psnrFromMse(mse(a, b, n, mask).mse);
}

/** Solve the symmetric positive (semi)definite 4 × 4 system A x = b (Gaussian elimination, partial pivoting). */
function solve4(A, b) {
  const M = [0, 1, 2, 3].map((r) => [A[4 * r], A[4 * r + 1], A[4 * r + 2], A[4 * r + 3], b[r]]);
  for (let c = 0; c < 4; c++) {
    let p = c;
    for (let r = c + 1; r < 4; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (!(Math.abs(M[p][c]) > 1e-12)) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < 4; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k < 5; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, r) => row[4] / row[r]);
}

/** Identity colour transform (3 × 4 row-major: out_c = Σ_k M[c][k]·in_k + M[c][3]). */
export const AFFINE_IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];

/**
 * Least-squares affine colour transform (3 × 4, row-major, see AFFINE_IDENTITY) mapping `src` RGB
 * to `dst` RGB over the masked pixels (both RGBA8, straight alpha). A tiny ridge towards the
 * identity keeps degenerate inputs (a constant channel) solvable. Identity with < 16 pixels.
 */
export function fitAffineColour(src, dst, n, mask = null) {
  const A = new Float64Array(16);
  const B = [new Float64Array(4), new Float64Array(4), new Float64Array(4)];
  let cnt = 0;
  const p = [0, 0, 0, 1];
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    const j = 4 * i;
    p[0] = src[j];
    p[1] = src[j + 1];
    p[2] = src[j + 2];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) A[4 * r + c] += p[r] * p[c];
    for (let ch = 0; ch < 3; ch++) {
      const y = dst[j + ch];
      for (let r = 0; r < 4; r++) B[ch][r] += p[r] * y;
    }
    cnt++;
  }
  if (cnt < 16) return AFFINE_IDENTITY.slice();
  const lambda = 1e-9 * (A[0] + A[5] + A[10] + 1);
  const Ar = Float64Array.from(A);
  for (let d = 0; d < 3; d++) Ar[5 * d] += lambda;
  const out = [];
  for (let ch = 0; ch < 3; ch++) {
    const b = Array.from(B[ch]);
    b[ch] += lambda; // ridge towards the identity row
    const x = solve4(Ar, b);
    if (!x || x.some((v) => !Number.isFinite(v))) return AFFINE_IDENTITY.slice();
    out.push(...x);
  }
  return out;
}

/** Apply a 3 × 4 colour transform to the RGB of the masked pixels (clamped to 0…255), in place. */
export function applyAffineColour(img, n, M, mask = null) {
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    const j = 4 * i;
    const r = img[j],
      g = img[j + 1],
      b = img[j + 2];
    for (let c = 0; c < 3; c++) {
      const v = M[4 * c] * r + M[4 * c + 1] * g + M[4 * c + 2] * b + M[4 * c + 3];
      img[j + c] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return img;
}

/**
 * Split a premultiplied render into straight colours and coverage:
 * { straight (RGBA8, alpha copied), alpha (0…1 Float32), covered (Uint8 mask, alpha ≥ 0.5) }.
 */
export function unpremultiply(render, n) {
  const straight = new Uint8ClampedArray(4 * n);
  const alpha = new Float32Array(n);
  const covered = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const j = 4 * i;
    const a = render[j + 3];
    alpha[i] = a / 255;
    if (a >= 128) covered[i] = 1;
    if (a > 0) {
      const s = 255 / a;
      straight[j] = render[j] * s;
      straight[j + 1] = render[j + 1] * s;
      straight[j + 2] = render[j + 2] * s;
    }
    straight[j + 3] = a;
  }
  return { straight, alpha, covered };
}

/** Straight colours with coverage alpha composited over `bg` [r, g, b] → opaque RGBA8. */
export function compositeOver(straight, alpha, n, bg = [0, 0, 0]) {
  const out = new Uint8ClampedArray(4 * n);
  for (let i = 0; i < n; i++) {
    const j = 4 * i;
    const a = alpha[i];
    out[j] = a * straight[j] + (1 - a) * bg[0];
    out[j + 1] = a * straight[j + 1] + (1 - a) * bg[1];
    out[j + 2] = a * straight[j + 2] + (1 - a) * bg[2];
    out[j + 3] = 255;
  }
  return out;
}

/** Validity mask of a ground-truth image (alpha 255). */
export function validMask(gt, n) {
  const m = new Uint8Array(n);
  for (let i = 0; i < n; i++) m[i] = gt[4 * i + 3] === 255 ? 1 : 0;
  return m;
}

/**
 * Ground truth prepared for scoring several renders of one view: { data, valid, nValid, ssimCache }
 * (the full-frame SSIM statistics of the GT are computed once).
 */
export function prepareGroundTruth(gt, w, h) {
  const valid = validMask(gt, w * h);
  let nValid = 0;
  for (let i = 0; i < valid.length; i++) nValid += valid[i];
  return { data: gt, valid, nValid, ssimCache: new Map() };
}

function and(a, b) {
  const o = new Uint8Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = a[i] && b[i] ? 1 : 0;
  return o;
}

function count(m) {
  let c = 0;
  for (let i = 0; i < m.length; i++) c += m[i];
  return c;
}

/** Rec. 601 luma of an RGBA8 image, 2 × 2 box-downsampled (⌊w/2⌋ × ⌊h/2⌋), with a mask (all 4 set). */
function lumaHalf(img, w, h, mask) {
  const W = w >> 1,
    H = h >> 1;
  const L = new Float32Array(W * H);
  const M = new Uint8Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let s = 0,
        ok = 1;
      for (let dy = 0; dy < 2; dy++)
        for (let dx = 0; dx < 2; dx++) {
          const i = (2 * y + dy) * w + 2 * x + dx;
          s += 0.299 * img[4 * i] + 0.587 * img[4 * i + 1] + 0.114 * img[4 * i + 2];
          if (mask && !mask[i]) ok = 0;
        }
      L[y * W + x] = s / 4;
      M[y * W + x] = ok;
    }
  return { L, M, W, H };
}

/**
 * Residual image shift between a render and the ground truth (diagnostic of pose / intrinsics /
 * projection errors): the integer shift (on 2× downsampled luma, ±radius) minimising the
 * mean-removed SSD over pixels valid in both, refined by a parabola through the neighbours.
 * Returns { dx, dy } in FULL-resolution pixels (render(x) ≈ gt(x + d)), or null for a flat render
 * (luma variance < 1) or when no shift overlaps min(1000, ¼ of the half-resolution image) pixels.
 */
export function alignmentShift(render, renderMask, gt, gtMask, w, h, radius = 3) {
  const A = lumaHalf(render, w, h, renderMask);
  const B = lumaHalf(gt, w, h, gtMask);
  const { W, H } = A;
  // a flat render has no defined shift
  let n0 = 0,
    s1 = 0,
    s2 = 0;
  for (let i = 0; i < A.L.length; i++)
    if (A.M[i]) {
      n0++;
      s1 += A.L[i];
      s2 += A.L[i] * A.L[i];
    }
  if (!n0 || s2 / n0 - (s1 / n0) ** 2 < 1) return null;
  const minOverlap = Math.min(1000, (W * H) >> 2);
  const S = 2 * radius + 1;
  const cost = new Float64Array(S * S).fill(Infinity);
  for (let sy = -radius; sy <= radius; sy++)
    for (let sx = -radius; sx <= radius; sx++) {
      let n = 0,
        sa = 0,
        sb = 0,
        saa = 0,
        sbb = 0,
        sab = 0;
      for (let y = Math.max(0, -sy); y < Math.min(H, H - sy); y++)
        for (let x = Math.max(0, -sx); x < Math.min(W, W - sx); x++) {
          const i = y * W + x,
            j = (y + sy) * W + x + sx;
          if (!A.M[i] || !B.M[j]) continue;
          const a = A.L[i],
            b = B.L[j];
          n++;
          sa += a;
          sb += b;
          saa += a * a;
          sbb += b * b;
          sab += a * b;
        }
      if (n < minOverlap) continue;
      // Σ ((a − ā) − (b − b̄))² / n
      cost[(sy + radius) * S + sx + radius] = (saa - (sa * sa) / n + sbb - (sb * sb) / n - 2 * (sab - (sa * sb) / n)) / n;
    }
  let best = -1;
  for (let i = 0; i < cost.length; i++) if (Number.isFinite(cost[i]) && (best < 0 || cost[i] < cost[best])) best = i;
  if (best < 0) return null;
  const bx = (best % S) - radius,
    by = Math.floor(best / S) - radius;
  const sub = (cm, c0, cp) => {
    if (!Number.isFinite(cm) || !Number.isFinite(cp)) return 0;
    const den = cm - 2 * c0 + cp;
    return den > 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (cm - cp)) / den)) : 0;
  };
  const at = (x, y) => (Math.abs(x) <= radius && Math.abs(y) <= radius ? cost[(y + radius) * S + x + radius] : Infinity);
  const fx = sub(at(bx - 1, by), at(bx, by), at(bx + 1, by));
  const fy = sub(at(bx, by - 1), at(bx, by), at(bx, by + 1));
  return { dx: 2 * (bx + fx), dy: 2 * (by + fy) };
}

/**
 * Score one rendered view against its ground truth.
 * render: premultiplied RGBA8; gtImage: RGBA8 (alpha 255 = valid) or prepareGroundTruth(gt). Returns
 * { coverage, full: {psnr, ssim}, covered: {psnr, ssim}, cc: {full, covered, affine},
 *   shift: {dx, dy} | null, images?: {full, ccFull} } where
 *  - coverage = covered valid pixels / valid pixels;
 *  - full = render composited over `background` vs GT on every valid pixel;
 *  - covered = un-premultiplied render vs GT on covered valid pixels only;
 *  - cc = the same after the least-squares 3 × 4 affine colour fit render → GT on the covered
 *    pixels (exposure / white-balance compensation), applied to the covered pixels only
 *    (uncovered pixels stay background).
 * opts: { background = [0, 0, 0], ssim = true, shift = true, keepImages = false }.
 */
export function scoreView(render, gtImage, w, h, opts = {}) {
  const n = w * h;
  const bg = opts.background ?? [0, 0, 0];
  const doSsim = opts.ssim !== false;
  const G = gtImage.ssimCache ? gtImage : prepareGroundTruth(gtImage, w, h);
  const { data: gt, valid, nValid, ssimCache } = G;
  const { straight, alpha, covered: cov } = unpremultiply(render, n);
  const covered = and(cov, valid);
  const nCov = count(covered);
  const coverage = nValid ? nCov / nValid : 0;
  const score = (img, mask) => ({
    psnr: psnr(img, gt, n, mask),
    ssim: doSsim ? ssim(img, gt, w, h, mask, ssimCache) : null,
  });
  const full = compositeOver(straight, alpha, n, bg);
  const res = {
    coverage,
    validPixels: nValid,
    full: score(full, valid),
    covered: nCov ? score(straight, covered) : { psnr: NaN, ssim: NaN },
    cc: null,
    shift: null,
  };
  const affine = nCov >= 16 ? fitAffineColour(straight, gt, n, covered) : AFFINE_IDENTITY.slice();
  const ccStraight = applyAffineColour(Uint8ClampedArray.from(straight), n, affine, cov);
  const ccFull = compositeOver(ccStraight, alpha, n, bg);
  res.cc = {
    full: score(ccFull, valid),
    covered: nCov ? score(ccStraight, covered) : { psnr: NaN, ssim: NaN },
    affine: affine.map((v) => +v.toFixed(5)),
  };
  ssimCache.delete(covered); // per-render mask: only the full-frame statistics are reused
  if (opts.shift !== false && nCov > 0) res.shift = alignmentShift(straight, covered, gt, valid, w, h);
  if (opts.keepImages) res.images = { full, ccFull };
  return res;
}

/** Side-by-side panel: GT | render (full frame) | |render − GT| × gain (RGBA8, 3w × h). */
export function triptych(gt, full, w, h, gain = 3) {
  const out = new Uint8ClampedArray(4 * 3 * w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = 4 * (y * w + x);
      const o = 4 * (y * 3 * w + x);
      const ok = gt[i + 3] === 255;
      for (let c = 0; c < 3; c++) {
        out[o + c] = gt[i + c];
        out[o + 4 * w + c] = full[i + c];
        out[o + 8 * w + c] = ok ? gain * Math.abs(full[i + c] - gt[i + c]) : 0;
      }
      out[o + 3] = out[o + 4 * w + 3] = out[o + 8 * w + 3] = 255;
    }
  return out;
}

const finite = (v) => typeof v === 'number' && Number.isFinite(v);

function stats(values) {
  const v = values.filter(finite).sort((a, b) => a - b);
  if (!v.length) return { mean: null, median: null, min: null, max: null, n: 0 };
  const mean = v.reduce((s, x) => s + x, 0) / v.length;
  const med = v.length % 2 ? v[v.length >> 1] : 0.5 * (v[v.length / 2 - 1] + v[v.length / 2]);
  return { mean, median: med, min: v[0], max: v[v.length - 1], n: v.length };
}

/**
 * Aggregate per-view scores of one renderer: mean / median / min / max of every metric over the
 * views (PSNR is averaged in dB per view, the usual NVS convention).
 */
export function aggregateScores(scores) {
  const pick = (f) => stats(scores.map(f));
  return {
    views: scores.length,
    coverage: pick((s) => s.coverage),
    full: { psnr: pick((s) => s.full.psnr), ssim: pick((s) => s.full.ssim) },
    covered: { psnr: pick((s) => s.covered.psnr), ssim: pick((s) => s.covered.ssim) },
    cc: {
      full: { psnr: pick((s) => s.cc?.full.psnr), ssim: pick((s) => s.cc?.full.ssim) },
      covered: { psnr: pick((s) => s.cc?.covered.psnr), ssim: pick((s) => s.cc?.covered.ssim) },
    },
    shiftPx: pick((s) => (s.shift ? Math.hypot(s.shift.dx, s.shift.dy) : NaN)),
  };
}
