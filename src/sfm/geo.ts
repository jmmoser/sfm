/**
 * Geodesy and GPS / compass alignment of a reconstruction:
 *  - WGS84 geodetic ⇄ ECEF ⇄ local ENU (east-north-up) conversions in double precision;
 *  - `alignToGps`: robust weighted similarity (reconstruction → ENU) from GPS fixes, either
 *    4-DOF (scale, yaw, translation) when the reconstruction is already gravity-aligned
 *    (+Z up) or a full Sim3 otherwise, with residual statistics and a scale uncertainty;
 *  - `yawFromCompass`: world yaw from compass headings when GPS cannot provide it.
 *
 * Altitudes: `GeoFix.alt` on iOS is orthometric (MSL), not ellipsoidal. Inside a local ENU
 * frame the geoid undulation is a constant offset, so this only shifts `originLatLonAlt[2]`.
 */
import type { GeoFix, Mat3, Vec3 } from '../core/types';
import { alignSim3Ransac } from '../geometry/align';
import { Rng } from '../math/random';
import { umeyama } from '../math/sim3';
import { median } from '../math/stats';

/** WGS84 semi-major axis (m). */
export const WGS84_A = 6378137.0;
/** WGS84 flattening. */
export const WGS84_F = 1 / 298.257223563;
/** WGS84 semi-minor axis (m). */
export const WGS84_B = WGS84_A * (1 - WGS84_F);
/** WGS84 first eccentricity squared. */
export const WGS84_E2 = WGS84_F * (2 - WGS84_F);

const DEG = Math.PI / 180;

/** Geodetic (latitude°, longitude°, ellipsoidal height m) → ECEF (m). */
export function geodeticToEcef(latDeg: number, lonDeg: number, h: number, out: Vec3 = new Float64Array(3)): Vec3 {
  const lat = latDeg * DEG,
    lon = lonDeg * DEG;
  const sl = Math.sin(lat),
    cl = Math.cos(lat);
  const N = WGS84_A / Math.sqrt(1 - WGS84_E2 * sl * sl);
  out[0] = (N + h) * cl * Math.cos(lon);
  out[1] = (N + h) * cl * Math.sin(lon);
  out[2] = (N * (1 - WGS84_E2) + h) * sl;
  return out;
}

/**
 * ECEF (m) → geodetic [latitude°, longitude°, height m]. Fixed-point iteration
 * φ ← atan2(z + e² N(φ) sin φ, p) (contraction factor ≈ e², converged to < 1e-14 rad in
 * ≤ 8 steps for terrestrial and orbital heights; exact at the poles), height from the
 * pole-safe form h = p cos φ + z sin φ − a √(1 − e² sin² φ).
 */
export function ecefToGeodetic(x: number, y: number, z: number, out: Float64Array = new Float64Array(3)): Float64Array {
  const p = Math.hypot(x, y);
  const lon = Math.atan2(y, x);
  let lat = Math.atan2(z, p * (1 - WGS84_E2));
  for (let it = 0; it < 12; it++) {
    const s = Math.sin(lat);
    const N = WGS84_A / Math.sqrt(1 - WGS84_E2 * s * s);
    const next = Math.atan2(z + WGS84_E2 * N * s, p);
    const d = Math.abs(next - lat);
    lat = next;
    if (d < 1e-15) break;
  }
  const s = Math.sin(lat),
    c = Math.cos(lat);
  out[0] = lat / DEG;
  out[1] = lon / DEG;
  out[2] = p * c + z * s - WGS84_A * Math.sqrt(1 - WGS84_E2 * s * s);
  return out;
}

/**
 * Local east-north-up tangent frame at a WGS84 origin. `R` maps ECEF offsets to ENU
 * (rows = east, north, up unit vectors in ECEF).
 */
export class EnuFrame {
  readonly lat0: number;
  readonly lon0: number;
  readonly alt0: number;
  /** ECEF position of the origin. */
  readonly originEcef: Vec3;
  /** Row-major ECEF → ENU rotation. */
  readonly R: Mat3;
  private tmp = new Float64Array(3);

  constructor(latDeg: number, lonDeg: number, alt = 0) {
    this.lat0 = latDeg;
    this.lon0 = lonDeg;
    this.alt0 = alt;
    this.originEcef = geodeticToEcef(latDeg, lonDeg, alt);
    const sp = Math.sin(latDeg * DEG),
      cp = Math.cos(latDeg * DEG);
    const sl = Math.sin(lonDeg * DEG),
      cl = Math.cos(lonDeg * DEG);
    this.R = Float64Array.of(-sl, cl, 0, -sp * cl, -sp * sl, cp, cp * cl, cp * sl, sp);
  }

  /** ECEF → ENU. */
  fromEcef(ecef: ArrayLike<number>, out: Vec3 = new Float64Array(3)): Vec3 {
    const R = this.R,
      o = this.originEcef;
    const dx = ecef[0] - o[0],
      dy = ecef[1] - o[1],
      dz = ecef[2] - o[2];
    out[0] = R[0] * dx + R[1] * dy + R[2] * dz;
    out[1] = R[3] * dx + R[4] * dy + R[5] * dz;
    out[2] = R[6] * dx + R[7] * dy + R[8] * dz;
    return out;
  }

  /** ENU → ECEF. */
  toEcef(enu: ArrayLike<number>, out: Vec3 = new Float64Array(3)): Vec3 {
    const R = this.R,
      o = this.originEcef;
    const e = enu[0],
      n = enu[1],
      u = enu[2];
    out[0] = o[0] + R[0] * e + R[3] * n + R[6] * u;
    out[1] = o[1] + R[1] * e + R[4] * n + R[7] * u;
    out[2] = o[2] + R[2] * e + R[5] * n + R[8] * u;
    return out;
  }

  /** Geodetic (deg, deg, m) → ENU (m). */
  fromGeodetic(latDeg: number, lonDeg: number, alt: number, out: Vec3 = new Float64Array(3)): Vec3 {
    return this.fromEcef(geodeticToEcef(latDeg, lonDeg, alt, this.tmp), out);
  }

  /** ENU (m) → geodetic [lat°, lon°, alt m]. */
  toGeodetic(e: number, n: number, u: number, out: Float64Array = new Float64Array(3)): Float64Array {
    const t = this.tmp;
    t[0] = e;
    t[1] = n;
    t[2] = u;
    const ecef = this.toEcef(t, new Float64Array(3));
    return ecefToGeodetic(ecef[0], ecef[1], ecef[2], out);
  }
}

/** Geodetic → ENU at the given origin (convenience; build an `EnuFrame` for many points). */
export function geodeticToEnu(
  latDeg: number,
  lonDeg: number,
  alt: number,
  lat0Deg: number,
  lon0Deg: number,
  alt0: number,
  out: Vec3 = new Float64Array(3),
): Vec3 {
  return new EnuFrame(lat0Deg, lon0Deg, alt0).fromGeodetic(latDeg, lonDeg, alt, out);
}

/** ENU → geodetic [lat°, lon°, alt] at the given origin. */
export function enuToGeodetic(
  e: number,
  n: number,
  u: number,
  lat0Deg: number,
  lon0Deg: number,
  alt0: number,
  out: Float64Array = new Float64Array(3),
): Float64Array {
  return new EnuFrame(lat0Deg, lon0Deg, alt0).toGeodetic(e, n, u, out);
}

// ---------------------------------------------------------------------------------------
// GPS alignment
// ---------------------------------------------------------------------------------------

/** Options of `alignToGps`. */
export interface GpsAlignOptions {
  /**
   * Centres are already in a gravity-aligned frame (+Z up): estimate only scale, yaw about
   * +Z and translation (4-DOF). Much better conditioned than a full Sim3 (works for straight
   * tracks; GPS altitude noise does not tilt the model). Default false.
   */
  gravityAligned: boolean;
  /** Minimum number of associated fixes (default 5). */
  minFixes: number;
  /** Required horizontal track extent (`trackExtent`, noise-corrected) in multiples of the median accuracy (default 4). */
  minExtentFactor: number;
  /** Lower bound on the per-fix accuracy used for weighting (m, default 1). */
  minAccuracy: number;
  /** Fixes farther than this (ms) outside the keyframe time span are dropped (default 1000). */
  maxExtrapolationMs: number;
  /** Fixes falling in a keyframe gap longer than this (ms) are dropped (default 5000). */
  maxBracketMs: number;
  /** Robust inlier gate in units of the fix accuracy (default 3). */
  inlierSigma: number;
  /** ENU origin [lat°, lon°, alt]; default: accuracy-weighted mean of the associated fixes. */
  origin: [number, number, number] | null;
  rng: Rng;
}

/** Result of `alignToGps`: ENU ≈ s R X + t for reconstruction points X. */
export interface GpsAlignment {
  s: number;
  R: Mat3;
  t: Vec3;
  /** WGS84 origin [lat°, lon°, alt] of the ENU frame (alt = 0 when no fix has altitude). */
  origin: [number, number, number];
  mode: '4dof' | 'sim3';
  numFixes: number;
  numInliers: number;
  inlierMask: Uint8Array;
  /** Horizontal residuals of the inliers (m). */
  rmsResidual: number;
  medianResidual: number;
  /** 1-σ relative scale uncertainty (inflated for time-correlated GPS errors). */
  scaleRelUncertainty: number;
  /** 1-σ yaw uncertainty (rad). */
  yawStdRad: number;
  /** Horizontal track extent (m, `trackExtent`) and median fix accuracy (m). */
  extent: number;
  medianAccuracy: number;
  /** Lag-1 autocorrelation of the time-ordered residuals (≥ 0 ⇒ inflated uncertainty). */
  residualAutocorrelation: number;
  /** Associated pairs: reconstruction centre at each fix time (3n), ENU position (3n), weights (n, 1/σ²). */
  src: Float64Array;
  dst: Float64Array;
  weights: Float64Array;
  /** Whether each associated fix has an altitude. */
  hasAlt: Uint8Array;
}

/** Horizontal 1-σ used for a fix: its accuracy floored at `minAccuracy` (fixes without a finite accuracy are not used). */
function fixSigma(f: GeoFix, minAccuracy: number): number {
  return Math.max(f.accuracy, minAccuracy);
}

const DEFAULT_GPS_OPTIONS: Omit<GpsAlignOptions, 'rng'> = {
  gravityAligned: false,
  minFixes: 5,
  minExtentFactor: 4,
  minAccuracy: 1,
  maxExtrapolationMs: 1000,
  maxBracketMs: 5000,
  inlierSigma: 3,
  origin: null,
};

/**
 * Horizontal track extent (m) of n ENU points: the length of a uniform segment with the same
 * spread along the principal axis, √(12 · max(0, λ_max − noiseVar)), where λ_max is the largest
 * eigenvalue of the 2-D covariance and `noiseVar` the mean per-axis variance of the GPS noise
 * (subtracting it keeps a stationary but noisy track from looking long).
 */
export function trackExtent(enu: Float64Array, n: number, noiseVar = 0): number {
  if (n < 2) return 0;
  let mx = 0,
    my = 0;
  for (let i = 0; i < n; i++) {
    mx += enu[3 * i];
    my += enu[3 * i + 1];
  }
  mx /= n;
  my /= n;
  let sxx = 0,
    syy = 0,
    sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = enu[3 * i] - mx,
      dy = enu[3 * i + 1] - my;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  sxx /= n - 1;
  syy /= n - 1;
  sxy /= n - 1;
  const tr = 0.5 * (sxx + syy);
  const lmax = tr + Math.sqrt(Math.max(0, 0.25 * (sxx - syy) * (sxx - syy) + sxy * sxy));
  return Math.sqrt(12 * Math.max(0, lmax - noiseVar));
}

/**
 * Robust weighted alignment of camera centres to GPS fixes. For every fix, the camera
 * centre at the fix time is interpolated linearly between the bracketing keyframes; fixes
 * are weighted by 1/max(accuracy, minAccuracy)² (fixes with a non-finite position, time or
 * accuracy are ignored; a non-finite `altitudeAccuracy` falls back to 2 × the horizontal
 * one). Returns null unless ≥ `minFixes` fixes
 * associate and the horizontal track extent exceeds `minExtentFactor` × the median accuracy.
 *
 * 4-DOF mode (`gravityAligned`): horizontal similarity as a complex linear regression
 * EN ≈ a·XY + b (a = s e^{iψ}) — 2-point LO-RANSAC, then Huber IRLS — and the vertical
 * offset from the altitudes (weighted by `altitudeAccuracy`) or, without altitudes, such
 * that the mean camera height is 0. Sim3 mode: `alignSim3Ransac` + weighted Umeyama.
 * The scale uncertainty is the regression standard error scaled by the unit-weight
 * residual variance and inflated by (1+ρ)/(1−ρ) for lag-1 residual autocorrelation ρ
 * (GPS errors are strongly time-correlated).
 *
 * @param centres camera centres 3n (reconstruction frame)
 * @param times keyframe times (ms), same order as `centres`
 */
export function alignToGps(
  centres: ArrayLike<number>,
  times: ArrayLike<number>,
  fixes: readonly GeoFix[],
  options: Partial<GpsAlignOptions> = {},
): GpsAlignment | null {
  const o = { ...DEFAULT_GPS_OPTIONS, ...options };
  const rng = options.rng ?? new Rng(0x6e05);
  const nk = times.length;
  if (nk < 2 || fixes.length < o.minFixes) return null;

  // Keyframe order by time.
  const order = Array.from({ length: nk }, (_, i) => i).sort((a, b) => times[a] - times[b]);
  const kt = new Float64Array(nk);
  const kc = new Float64Array(3 * nk);
  for (let i = 0; i < nk; i++) {
    const j = order[i];
    kt[i] = times[j];
    kc[3 * i] = centres[3 * j];
    kc[3 * i + 1] = centres[3 * j + 1];
    kc[3 * i + 2] = centres[3 * j + 2];
  }

  // Associate fixes (time-sorted).
  // Fixes without a finite position, time or accuracy cannot be weighted: dropped.
  const fsorted = fixes
    .filter((f) => Number.isFinite(f.lat) && Number.isFinite(f.lon) && Number.isFinite(f.t) && Number.isFinite(f.accuracy))
    .sort((a, b) => a.t - b.t);
  const used: GeoFix[] = [];
  const srcList: number[] = [];
  for (const f of fsorted) {
    let cx: number, cy: number, cz: number;
    if (f.t <= kt[0]) {
      if (kt[0] - f.t > o.maxExtrapolationMs) continue;
      cx = kc[0];
      cy = kc[1];
      cz = kc[2];
    } else if (f.t >= kt[nk - 1]) {
      if (f.t - kt[nk - 1] > o.maxExtrapolationMs) continue;
      cx = kc[3 * nk - 3];
      cy = kc[3 * nk - 2];
      cz = kc[3 * nk - 1];
    } else {
      let lo = 0,
        hi = nk - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (kt[mid] <= f.t) lo = mid;
        else hi = mid;
      }
      const dt = kt[hi] - kt[lo];
      if (dt > o.maxBracketMs) continue;
      const a = dt > 0 ? (f.t - kt[lo]) / dt : 0;
      cx = kc[3 * lo] + a * (kc[3 * hi] - kc[3 * lo]);
      cy = kc[3 * lo + 1] + a * (kc[3 * hi + 1] - kc[3 * lo + 1]);
      cz = kc[3 * lo + 2] + a * (kc[3 * hi + 2] - kc[3 * lo + 2]);
    }
    used.push(f);
    srcList.push(cx, cy, cz);
  }
  const n = used.length;
  if (n < o.minFixes) return null;

  // ENU origin.
  let origin: [number, number, number];
  let anyAlt = false;
  if (o.origin) origin = [o.origin[0], o.origin[1], o.origin[2]];
  else {
    let sw = 0,
      slat = 0,
      slon = 0,
      salt = 0,
      swa = 0;
    const lon0 = used[0].lon;
    for (const f of used) {
      const w = 1 / fixSigma(f, o.minAccuracy) ** 2;
      sw += w;
      slat += w * f.lat;
      // unwrap longitude around the first fix (antimeridian-safe)
      let dl = f.lon - lon0;
      if (dl > 180) dl -= 360;
      else if (dl < -180) dl += 360;
      slon += w * dl;
      if (f.alt !== null && Number.isFinite(f.alt)) {
        salt += f.alt;
        swa++;
        anyAlt = true;
      }
    }
    let lonm = lon0 + slon / sw;
    if (lonm > 180) lonm -= 360;
    else if (lonm < -180) lonm += 360;
    origin = [slat / sw, lonm, swa > 0 ? salt / swa : 0];
  }
  if (!anyAlt) anyAlt = used.some((f) => f.alt !== null && Number.isFinite(f.alt));
  const frame = new EnuFrame(origin[0], origin[1], origin[2]);
  const src = Float64Array.from(srcList);
  const dst = new Float64Array(3 * n);
  const weights = new Float64Array(n);
  const sigma = new Float64Array(n);
  const hasAlt = new Uint8Array(n);
  const acc = new Float64Array(n);
  const tmp = new Float64Array(3);
  for (let i = 0; i < n; i++) {
    const f = used[i];
    const alt = f.alt !== null && Number.isFinite(f.alt) ? f.alt : origin[2];
    hasAlt[i] = f.alt !== null && Number.isFinite(f.alt) ? 1 : 0;
    frame.fromGeodetic(f.lat, f.lon, alt, tmp);
    dst[3 * i] = tmp[0];
    dst[3 * i + 1] = tmp[1];
    dst[3 * i + 2] = tmp[2];
    sigma[i] = fixSigma(f, o.minAccuracy);
    weights[i] = 1 / (sigma[i] * sigma[i]);
    acc[i] = sigma[i];
  }
  const medAcc = median(acc);
  let noiseVar = 0;
  for (let i = 0; i < n; i++) noiseVar += acc[i] * acc[i];
  const extent = trackExtent(dst, n, noiseVar / n);
  if (!(extent > o.minExtentFactor * medAcc)) return null;

  const res = o.gravityAligned
    ? align4Dof(src, dst, weights, sigma, hasAlt, used, n, o.inlierSigma, rng)
    : alignSim3(src, dst, weights, sigma, n, o.inlierSigma, medAcc, rng);
  if (!res) return null;

  return {
    ...res,
    origin,
    mode: o.gravityAligned ? '4dof' : 'sim3',
    numFixes: n,
    extent,
    medianAccuracy: medAcc,
    src,
    dst,
    weights,
    hasAlt,
  };
}

interface CoreAlign {
  s: number;
  R: Mat3;
  t: Vec3;
  numInliers: number;
  inlierMask: Uint8Array;
  rmsResidual: number;
  medianResidual: number;
  scaleRelUncertainty: number;
  yawStdRad: number;
  residualAutocorrelation: number;
}

/** Weighted complex regression d ≈ a c + b over the masked set (horizontal). */
function complexRegression(
  src: Float64Array,
  dst: Float64Array,
  w: Float64Array,
  mask: Uint8Array | null,
  n: number,
): { ar: number; ai: number; br: number; bi: number; S: number } | null {
  let sw = 0,
    mcx = 0,
    mcy = 0,
    mdx = 0,
    mdy = 0;
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    const wi = w[i];
    sw += wi;
    mcx += wi * src[3 * i];
    mcy += wi * src[3 * i + 1];
    mdx += wi * dst[3 * i];
    mdy += wi * dst[3 * i + 1];
  }
  if (!(sw > 0)) return null;
  mcx /= sw;
  mcy /= sw;
  mdx /= sw;
  mdy /= sw;
  let S = 0,
    nr = 0,
    ni = 0;
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    const wi = w[i];
    const cx = src[3 * i] - mcx,
      cy = src[3 * i + 1] - mcy;
    const dx = dst[3 * i] - mdx,
      dy = dst[3 * i + 1] - mdy;
    S += wi * (cx * cx + cy * cy);
    // (dx + i dy)(cx − i cy)
    nr += wi * (dx * cx + dy * cy);
    ni += wi * (dy * cx - dx * cy);
  }
  if (!(S > 0)) return null;
  const ar = nr / S,
    ai = ni / S;
  return { ar, ai, br: mdx - (ar * mcx - ai * mcy), bi: mdy - (ar * mcy + ai * mcx), S };
}

function lag1Autocorrelation(rx: Float64Array, ry: Float64Array, mask: Uint8Array, n: number): number {
  let num = 0,
    den = 0,
    prev = -1;
  for (let i = 0; i < n; i++) {
    if (!mask[i]) continue;
    den += rx[i] * rx[i] + ry[i] * ry[i];
    if (prev >= 0) num += rx[i] * rx[prev] + ry[i] * ry[prev];
    prev = i;
  }
  return den > 0 ? num / den : 0;
}

function align4Dof(
  src: Float64Array,
  dst: Float64Array,
  w: Float64Array,
  sigma: Float64Array,
  hasAlt: Uint8Array,
  fixes: readonly GeoFix[],
  n: number,
  gate: number,
  rng: Rng,
): CoreAlign | null {
  // 1) 2-point RANSAC on the horizontal similarity (normalised residuals r/σ, MSAC cost).
  const T2 = gate * gate;
  let best = Infinity;
  let bar = 1,
    bai = 0,
    bbr = 0,
    bbi = 0;
  let found = false;
  const iters = Math.min(500, (n * (n - 1)) / 2);
  const exhaustive = (n * (n - 1)) / 2 <= 500;
  let pi = 0,
    pj = 1;
  for (let it = 0; it < iters; it++) {
    let i: number, j: number;
    if (exhaustive) {
      i = pi;
      j = pj;
      pj++;
      if (pj >= n) {
        pi++;
        pj = pi + 1;
      }
    } else {
      i = rng.int(n);
      j = rng.int(n - 1);
      if (j >= i) j++;
    }
    const cx = src[3 * j] - src[3 * i],
      cy = src[3 * j + 1] - src[3 * i + 1];
    const dx = dst[3 * j] - dst[3 * i],
      dy = dst[3 * j + 1] - dst[3 * i + 1];
    const c2 = cx * cx + cy * cy;
    if (!(c2 > 0)) continue;
    // a = d / c
    const ar = (dx * cx + dy * cy) / c2,
      ai = (dy * cx - dx * cy) / c2;
    if (!(ar * ar + ai * ai > 0)) continue;
    const br = dst[3 * i] - (ar * src[3 * i] - ai * src[3 * i + 1]);
    const bi = dst[3 * i + 1] - (ar * src[3 * i + 1] + ai * src[3 * i]);
    let cost = 0;
    for (let k = 0; k < n; k++) {
      const ex = ar * src[3 * k] - ai * src[3 * k + 1] + br - dst[3 * k];
      const ey = ar * src[3 * k + 1] + ai * src[3 * k] + bi - dst[3 * k + 1];
      const r2 = (ex * ex + ey * ey) / (sigma[k] * sigma[k]);
      cost += r2 < T2 ? r2 : T2;
      if (cost >= best) break;
    }
    if (cost < best) {
      best = cost;
      bar = ar;
      bai = ai;
      bbr = br;
      bbi = bi;
      found = true;
    }
  }
  if (!found) return null;

  // 2) Huber IRLS on all fixes (gate-based inliers for statistics).
  const mask = new Uint8Array(n);
  const rw = new Float64Array(n);
  const rx = new Float64Array(n),
    ry = new Float64Array(n);
  const k = 2;
  let reg: ReturnType<typeof complexRegression> = { ar: bar, ai: bai, br: bbr, bi: bbi, S: 0 };
  for (let it = 0; it < 8; it++) {
    for (let i = 0; i < n; i++) {
      const ex = reg.ar * src[3 * i] - reg.ai * src[3 * i + 1] + reg.br - dst[3 * i];
      const ey = reg.ar * src[3 * i + 1] + reg.ai * src[3 * i] + reg.bi - dst[3 * i + 1];
      const r = Math.hypot(ex, ey) / sigma[i];
      mask[i] = r < gate ? 1 : 0;
      rw[i] = mask[i] ? w[i] * (r <= k ? 1 : k / r) : 0;
    }
    const next = complexRegression(src, dst, rw, null, n);
    if (!next) break;
    const change = Math.hypot(next.ar - reg.ar, next.ai - reg.ai) / Math.hypot(reg.ar, reg.ai);
    reg = next;
    if (change < 1e-10) break;
  }
  let numIn = 0;
  for (let i = 0; i < n; i++) {
    const ex = reg.ar * src[3 * i] - reg.ai * src[3 * i + 1] + reg.br - dst[3 * i];
    const ey = reg.ar * src[3 * i + 1] + reg.ai * src[3 * i] + reg.bi - dst[3 * i + 1];
    rx[i] = ex;
    ry[i] = ey;
    mask[i] = Math.hypot(ex, ey) / sigma[i] < gate ? 1 : 0;
    numIn += mask[i];
  }
  if (numIn < 3) return null;
  const fin = complexRegression(src, dst, w, mask, n);
  if (!fin) return null;
  const s = Math.hypot(fin.ar, fin.ai);
  if (!(s > 0)) return null;
  const psi = Math.atan2(fin.ai, fin.ar);
  const c = Math.cos(psi),
    sn = Math.sin(psi);
  const R = Float64Array.of(c, -sn, 0, sn, c, 0, 0, 0, 1);

  // Residual statistics on inliers.
  let chi2 = 0;
  const res: number[] = [];
  for (let i = 0; i < n; i++) {
    const ex = fin.ar * src[3 * i] - fin.ai * src[3 * i + 1] + fin.br - dst[3 * i];
    const ey = fin.ar * src[3 * i + 1] + fin.ai * src[3 * i] + fin.bi - dst[3 * i + 1];
    rx[i] = ex;
    ry[i] = ey;
    if (mask[i]) {
      chi2 += w[i] * (ex * ex + ey * ey);
      res.push(Math.hypot(ex, ey));
    }
  }
  const dof = Math.max(1, 2 * numIn - 4);
  const s0sq = chi2 / dof; // unit-weight variance per component
  const rho = Math.max(0, lag1Autocorrelation(rx, ry, mask, n));
  const infl = (1 + rho) / Math.max(1e-3, 1 - rho);
  const varA = (Math.max(s0sq, 0.25) * infl) / fin.S;

  // Vertical translation.
  let tz: number;
  let swz = 0,
    sz = 0;
  for (let i = 0; i < n; i++) {
    if (!mask[i] || !hasAlt[i]) continue;
    const f = fixes[i];
    const aa = f.altitudeAccuracy;
    const sa = Math.max(aa !== null && Number.isFinite(aa) && aa > 0 ? aa : 2 * sigma[i], 1);
    const wz = 1 / (sa * sa);
    swz += wz;
    sz += wz * (dst[3 * i + 2] - s * src[3 * i + 2]);
  }
  if (swz > 0) tz = sz / swz;
  else {
    let mz = 0,
      cnt = 0;
    for (let i = 0; i < n; i++) {
      if (!mask[i]) continue;
      mz += src[3 * i + 2];
      cnt++;
    }
    tz = -s * (mz / cnt);
  }
  return {
    s,
    R,
    t: Float64Array.of(fin.br, fin.bi, tz),
    numInliers: numIn,
    inlierMask: mask,
    rmsResidual: Math.sqrt(res.reduce((a, b) => a + b * b, 0) / res.length),
    medianResidual: median(res),
    scaleRelUncertainty: Math.sqrt(varA) / s,
    yawStdRad: Math.sqrt(varA) / s,
    residualAutocorrelation: rho,
  };
}

function alignSim3(
  src: Float64Array,
  dst: Float64Array,
  w: Float64Array,
  sigma: Float64Array,
  n: number,
  gate: number,
  medAcc: number,
  rng: Rng,
): CoreAlign | null {
  const r = alignSim3Ransac(src, dst, n, gate * medAcc, { weights: w, rng, minInliers: 3 });
  if (!r) return null;
  // Weighted Umeyama on the inliers (per-fix gate).
  const mask = new Uint8Array(n);
  let sim = { s: r.s, R: r.R, t: r.t };
  const rx = new Float64Array(n),
    ry = new Float64Array(n),
    rz = new Float64Array(n);
  const mw = new Float64Array(n);
  let numIn = 0;
  for (let it = 0; it < 3; it++) {
    numIn = 0;
    for (let i = 0; i < n; i++) {
      residual3(sim, src, dst, i, rx, ry, rz);
      const e = Math.hypot(rx[i], ry[i], rz[i]) / sigma[i];
      mask[i] = e < gate ? 1 : 0;
      mw[i] = mask[i] ? w[i] : 0;
      numIn += mask[i];
    }
    if (numIn < 3) return null;
    const next = umeyama(src, dst, n, true, mw);
    if (!next) break;
    sim = next;
  }
  let chi2 = 0,
    sw = 0;
  let mcx = 0,
    mcy = 0,
    mcz = 0;
  const res: number[] = [];
  numIn = 0;
  for (let i = 0; i < n; i++) {
    residual3(sim, src, dst, i, rx, ry, rz);
    mask[i] = Math.hypot(rx[i], ry[i], rz[i]) / sigma[i] < gate ? 1 : 0;
    if (!mask[i]) continue;
    numIn++;
    chi2 += w[i] * (rx[i] * rx[i] + ry[i] * ry[i] + rz[i] * rz[i]);
    res.push(Math.hypot(rx[i], ry[i]));
    sw += w[i];
    mcx += w[i] * src[3 * i];
    mcy += w[i] * src[3 * i + 1];
    mcz += w[i] * src[3 * i + 2];
  }
  if (numIn < 3) return null;
  mcx /= sw;
  mcy /= sw;
  mcz /= sw;
  let S = 0;
  for (let i = 0; i < n; i++) {
    if (!mask[i]) continue;
    const dx = src[3 * i] - mcx,
      dy = src[3 * i + 1] - mcy,
      dz = src[3 * i + 2] - mcz;
    S += w[i] * (dx * dx + dy * dy + dz * dz);
  }
  const s0sq = chi2 / Math.max(1, 3 * numIn - 7);
  const rho = Math.max(0, lag1Autocorrelation(rx, ry, mask, n));
  const infl = (1 + rho) / Math.max(1e-3, 1 - rho);
  const varS = (Math.max(s0sq, 0.25) * infl) / (S > 0 ? S : Infinity);
  return {
    s: sim.s,
    R: sim.R,
    t: sim.t,
    numInliers: numIn,
    inlierMask: mask,
    rmsResidual: Math.sqrt(res.reduce((a, b) => a + b * b, 0) / res.length),
    medianResidual: median(res),
    scaleRelUncertainty: Math.sqrt(varS) / sim.s,
    yawStdRad: Math.sqrt(varS) / sim.s,
    residualAutocorrelation: rho,
  };
}

function residual3(
  sim: { s: number; R: Mat3; t: Vec3 },
  src: Float64Array,
  dst: Float64Array,
  i: number,
  rx: Float64Array,
  ry: Float64Array,
  rz: Float64Array,
): void {
  const R = sim.R,
    s = sim.s;
  const x = src[3 * i],
    y = src[3 * i + 1],
    z = src[3 * i + 2];
  rx[i] = s * (R[0] * x + R[1] * y + R[2] * z) + sim.t[0] - dst[3 * i];
  ry[i] = s * (R[3] * x + R[4] * y + R[5] * z) + sim.t[1] - dst[3 * i + 1];
  rz[i] = s * (R[6] * x + R[7] * y + R[8] * z) + sim.t[2] - dst[3 * i + 2];
}

// ---------------------------------------------------------------------------------------
// Compass yaw
// ---------------------------------------------------------------------------------------

/** One compass observation for `yawFromCompass`. */
export interface CompassObservation {
  /** Heading of the reference axis, degrees clockwise from (true) north. */
  headingDeg: number;
  /** 1-σ heading accuracy (degrees); null/undefined/non-finite ⇒ 20°. */
  accuracyDeg?: number | null;
  /** The reference axis (device top or optical axis) in the gravity-aligned (+Z up) frame. */
  direction: ArrayLike<number>;
}

/** Result of `yawFromCompass`. */
export interface CompassYaw {
  /** Counter-clockwise rotation about +Z (rad) that maps the gravity-aligned frame to ENU (+Y north). */
  yaw: number;
  /** 1-σ uncertainty (rad), inflated for time correlation, floored at 1°. */
  stdRad: number;
  numUsed: number;
  numInliers: number;
}

function wrapPi(a: number): number {
  a = (a + Math.PI) % (2 * Math.PI);
  if (a < 0) a += 2 * Math.PI;
  return a - Math.PI;
}

/**
 * World yaw from compass headings. Each observation gives φᵢ = az(dirᵢ) − headingᵢ, where
 * az = atan2(x, y) is the azimuth of the reference axis clockwise from +Y; φ is the
 * counter-clockwise rotation about +Z taking the frame to ENU. Weighted circular mean
 * (weights = horizontal-norm² / accuracy²), observations with a near-vertical axis
 * (horizontal norm < `minHorizontal`) are skipped, outliers beyond `outlierDeg` from the
 * mean are dropped (2 passes). Returns null without usable observations.
 */
export function yawFromCompass(
  obs: readonly CompassObservation[],
  opts: { minHorizontal?: number; outlierDeg?: number } = {},
): CompassYaw | null {
  const minH = opts.minHorizontal ?? 0.5;
  const gate = (opts.outlierDeg ?? 45) * DEG;
  const n = obs.length;
  const phi = new Float64Array(n);
  const w = new Float64Array(n);
  let used = 0;
  for (let i = 0; i < n; i++) {
    const o = obs[i];
    const d = o.direction;
    const dn = Math.hypot(d[0], d[1], d[2]);
    const h = dn > 0 ? Math.hypot(d[0], d[1]) / dn : 0;
    if (!(h >= minH) || !Number.isFinite(o.headingDeg)) continue;
    const ad = o.accuracyDeg;
    const acc = Math.max(ad !== null && ad !== undefined && Number.isFinite(ad) ? ad : 20, 2) * DEG;
    phi[i] = wrapPi(Math.atan2(d[0], d[1]) - o.headingDeg * DEG);
    w[i] = (h * h) / (acc * acc);
    used++;
  }
  if (used === 0) return null;
  const inl = new Uint8Array(n);
  for (let i = 0; i < n; i++) inl[i] = w[i] > 0 ? 1 : 0;
  let mean = 0;
  for (let pass = 0; pass < 3; pass++) {
    let sc = 0,
      ss = 0;
    for (let i = 0; i < n; i++) {
      if (!inl[i]) continue;
      sc += w[i] * Math.cos(phi[i]);
      ss += w[i] * Math.sin(phi[i]);
    }
    if (sc === 0 && ss === 0) return null;
    mean = Math.atan2(ss, sc);
    if (pass === 2) break;
    let any = 0;
    for (let i = 0; i < n; i++) {
      inl[i] = w[i] > 0 && Math.abs(wrapPi(phi[i] - mean)) < gate ? 1 : 0;
      any += inl[i];
    }
    if (any === 0) {
      for (let i = 0; i < n; i++) inl[i] = w[i] > 0 ? 1 : 0;
      break;
    }
  }
  let sw = 0,
    sr = 0,
    cnt = 0,
    num = 0,
    den = 0,
    prev = -1;
  for (let i = 0; i < n; i++) {
    if (!inl[i]) continue;
    const r = wrapPi(phi[i] - mean);
    sw += w[i];
    sr += w[i] * r * r;
    cnt++;
    den += r * r;
    if (prev >= 0) num += r * wrapPi(phi[prev] - mean);
    prev = i;
  }
  const rho = den > 0 ? Math.max(0, num / den) : 0;
  const neff = Math.max(1, (cnt * (1 - rho)) / (1 + rho));
  const sd = Math.sqrt(sr / sw / neff);
  return { yaw: mean, stdRad: Math.max(sd, DEG), numUsed: used, numInliers: cnt };
}
