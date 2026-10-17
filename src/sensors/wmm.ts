/**
 * World Magnetic Model WMM2025 (NOAA NCEI / British Geological Survey; public domain): the
 * main-field and secular-variation model the US DoD, NATO and the mobile platforms use to turn
 * a MAGNETIC compass heading into a TRUE-north heading (true = magnetic + declination).
 *
 * Coefficients: the official `WMM.COF` of WMM2025 (header "2025.0 WMM-2025 11/13/2024",
 * https://www.ncei.noaa.gov/products/world-magnetic-model, `WMM2025COF.zip`, sha256 of the
 * file dfa85978…b075582f), embedded verbatim below; degree and order 12, epoch 2025.0, valid
 * 2025.0–2030.0. `wmm.test.ts` checks the implementation against all 100 rows of the official
 * `WMM2025_TestValues.txt` (D / I to the printed 0.01°, X / Y / Z / H / F to 1e-3 nT, rates).
 *
 * Algorithm (WMM2025 technical report, §1.2; the same steps as NOAA's `GeomagnetismLibrary.c`):
 *  1. geodetic (φ, λ, h above the WGS84 ellipsoid) → geocentric spherical (φ′, λ, r);
 *  2. Gauss coefficients at the date: g(t) = g(t₀) + (t − t₀)·ġ, likewise h;
 *  3. field in spherical coordinates from Schmidt semi-normalised associated Legendre functions
 *     P̆ₙᵐ(sin φ′) and their φ′-derivatives (reference radius 6371.2 km):
 *       X′ = −Σ (a/r)ⁿ⁺² Σ (g cos mλ + h sin mλ) dP̆ₙᵐ/dφ′
 *       Y′ =  Σ (a/r)ⁿ⁺² Σ m (g sin mλ − h cos mλ) P̆ₙᵐ / cos φ′
 *       Z′ = −Σ (n+1)(a/r)ⁿ⁺² Σ (g cos mλ + h sin mλ) P̆ₙᵐ
 *     At the geographic poles (cos φ′ < 1e-10) Y′ uses the limit of P̆ₙ¹ / cos φ′ (only m = 1
 *     survives), so the result stays finite and depends on λ (the meridian that defines "north"
 *     there); the secular variation is summed the same way with (ġ, ḣ);
 *  4. rotation by φ′ − φ back to the geodetic north / east / down frame; D = atan2(Y, X),
 *     I = atan2(Z, H), rates from the rotated secular-variation vector.
 *
 * Accuracy (report): declination RMS error ≈ 0.3–0.5° worldwide (crustal and disturbance fields
 * are not modelled), much larger close to the magnetic poles. Where H < 2000 nT ("blackout
 * zone") a compass is unreliable; H < 6000 nT is the "caution zone" — both are flagged.
 * Heights: the model wants height above the ellipsoid; GeoFix altitudes on iOS are MSL
 * (geoid undulation ≤ ~100 m) — irrelevant here (the field changes ≈ 1e-4° of declination per km).
 * Dates outside the validity period are extrapolated linearly (flagged), clamped to ±5 years.
 *
 * Pure, dependency-free, no DOM (safe in workers and Node).
 */

/** Official WMM2025 coefficient file (WMM.COF), verbatim: n, m, g, h (nT), ġ, ḣ (nT/yr). */
const WMM2025_COF = `
    2025.0            WMM-2025     11/13/2024
  1  0  -29351.8       0.0       12.0        0.0
  1  1   -1410.8    4545.4        9.7      -21.5
  2  0   -2556.6       0.0      -11.6        0.0
  2  1    2951.1   -3133.6       -5.2      -27.7
  2  2    1649.3    -815.1       -8.0      -12.1
  3  0    1361.0       0.0       -1.3        0.0
  3  1   -2404.1     -56.6       -4.2        4.0
  3  2    1243.8     237.5        0.4       -0.3
  3  3     453.6    -549.5      -15.6       -4.1
  4  0     895.0       0.0       -1.6        0.0
  4  1     799.5     278.6       -2.4       -1.1
  4  2      55.7    -133.9       -6.0        4.1
  4  3    -281.1     212.0        5.6        1.6
  4  4      12.1    -375.6       -7.0       -4.4
  5  0    -233.2       0.0        0.6        0.0
  5  1     368.9      45.4        1.4       -0.5
  5  2     187.2     220.2        0.0        2.2
  5  3    -138.7    -122.9        0.6        0.4
  5  4    -142.0      43.0        2.2        1.7
  5  5      20.9     106.1        0.9        1.9
  6  0      64.4       0.0       -0.2        0.0
  6  1      63.8     -18.4       -0.4        0.3
  6  2      76.9      16.8        0.9       -1.6
  6  3    -115.7      48.8        1.2       -0.4
  6  4     -40.9     -59.8       -0.9        0.9
  6  5      14.9      10.9        0.3        0.7
  6  6     -60.7      72.7        0.9        0.9
  7  0      79.5       0.0       -0.0        0.0
  7  1     -77.0     -48.9       -0.1        0.6
  7  2      -8.8     -14.4       -0.1        0.5
  7  3      59.3      -1.0        0.5       -0.8
  7  4      15.8      23.4       -0.1        0.0
  7  5       2.5      -7.4       -0.8       -1.0
  7  6     -11.1     -25.1       -0.8        0.6
  7  7      14.2      -2.3        0.8       -0.2
  8  0      23.2       0.0       -0.1        0.0
  8  1      10.8       7.1        0.2       -0.2
  8  2     -17.5     -12.6        0.0        0.5
  8  3       2.0      11.4        0.5       -0.4
  8  4     -21.7      -9.7       -0.1        0.4
  8  5      16.9      12.7        0.3       -0.5
  8  6      15.0       0.7        0.2       -0.6
  8  7     -16.8      -5.2       -0.0        0.3
  8  8       0.9       3.9        0.2        0.2
  9  0       4.6       0.0       -0.0        0.0
  9  1       7.8     -24.8       -0.1       -0.3
  9  2       3.0      12.2        0.1        0.3
  9  3      -0.2       8.3        0.3       -0.3
  9  4      -2.5      -3.3       -0.3        0.3
  9  5     -13.1      -5.2        0.0        0.2
  9  6       2.4       7.2        0.3       -0.1
  9  7       8.6      -0.6       -0.1       -0.2
  9  8      -8.7       0.8        0.1        0.4
  9  9     -12.9      10.0       -0.1        0.1
 10  0      -1.3       0.0        0.1        0.0
 10  1      -6.4       3.3        0.0        0.0
 10  2       0.2       0.0        0.1       -0.0
 10  3       2.0       2.4        0.1       -0.2
 10  4      -1.0       5.3       -0.0        0.1
 10  5      -0.6      -9.1       -0.3       -0.1
 10  6      -0.9       0.4        0.0        0.1
 10  7       1.5      -4.2       -0.1        0.0
 10  8       0.9      -3.8       -0.1       -0.1
 10  9      -2.7       0.9       -0.0        0.2
 10 10      -3.9      -9.1       -0.0       -0.0
 11  0       2.9       0.0        0.0        0.0
 11  1      -1.5       0.0       -0.0       -0.0
 11  2      -2.5       2.9        0.0        0.1
 11  3       2.4      -0.6        0.0       -0.0
 11  4      -0.6       0.2        0.0        0.1
 11  5      -0.1       0.5       -0.1       -0.0
 11  6      -0.6      -0.3        0.0       -0.0
 11  7      -0.1      -1.2       -0.0        0.1
 11  8       1.1      -1.7       -0.1       -0.0
 11  9      -1.0      -2.9       -0.1        0.0
 11 10      -0.2      -1.8       -0.1        0.0
 11 11       2.6      -2.3       -0.1        0.0
 12  0      -2.0       0.0        0.0        0.0
 12  1      -0.2      -1.3        0.0       -0.0
 12  2       0.3       0.7       -0.0        0.0
 12  3       1.2       1.0       -0.0       -0.1
 12  4      -1.3      -1.4       -0.0        0.1
 12  5       0.6      -0.0       -0.0       -0.0
 12  6       0.6       0.6        0.1       -0.0
 12  7       0.5      -0.1       -0.0       -0.0
 12  8      -0.1       0.8        0.0        0.0
 12  9      -0.4       0.1        0.0       -0.0
 12 10      -0.2      -1.0       -0.1       -0.0
 12 11      -1.3       0.1       -0.0        0.0
 12 12      -0.7       0.2       -0.1       -0.1
999999999999999999999999999999999999999999999999
999999999999999999999999999999999999999999999999
`;

/** A spherical-harmonic main-field model with linear secular variation. */
export interface MagneticModel {
  name: string;
  /** Epoch t₀ (decimal year). */
  epoch: number;
  releaseDate: string;
  /** Validity period [validFrom, validUntil) in decimal years. */
  validFrom: number;
  validUntil: number;
  nMax: number;
  /** Coefficients indexed n(n+1)/2 + m (nT, nT/yr). */
  g: Float64Array;
  h: Float64Array;
  gDot: Float64Array;
  hDot: Float64Array;
}

function parseCof(text: string, validYears: number): MagneticModel {
  const lines = text
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  const head = lines[0].split(/\s+/);
  const epoch = Number(head[0]);
  let nMax = 0;
  const rows: number[][] = [];
  for (let i = 1; i < lines.length; i++) {
    if (/^9{10,}/.test(lines[i])) break;
    const v = lines[i].split(/\s+/).map(Number);
    rows.push(v);
    nMax = Math.max(nMax, v[0]);
  }
  const size = ((nMax + 1) * (nMax + 2)) / 2;
  const g = new Float64Array(size),
    h = new Float64Array(size),
    gDot = new Float64Array(size),
    hDot = new Float64Array(size);
  for (const [n, m, gv, hv, gd, hd] of rows) {
    const k = (n * (n + 1)) / 2 + m;
    g[k] = gv;
    h[k] = hv;
    gDot[k] = gd;
    hDot[k] = hd;
  }
  return { name: head[1], epoch, releaseDate: head[2] ?? '', validFrom: epoch, validUntil: epoch + validYears, nMax, g, h, gDot, hDot };
}

/** WMM2025 (epoch 2025.0, valid until 2030.0). */
export const WMM2025: MagneticModel = parseCof(WMM2025_COF, 5);

/** WGS84 semi-major axis (km), flattening, first eccentricity². */
const A_KM = 6378.137;
const FLAT = 1 / 298.257223563;
const E2 = FLAT * (2 - FLAT);
/** Geomagnetic reference radius (km). */
const RE_KM = 6371.2;
const DEG = Math.PI / 180;
/** Dates are extrapolated at most this many years outside the validity period. */
const MAX_EXTRAPOLATION_YEARS = 5;

/** Magnetic elements at one place and date (geodetic north / east / down frame). */
export interface MagneticElements {
  /** Declination D (deg, EAST positive): true heading = magnetic heading + D. */
  declination: number;
  /** Inclination / dip I (deg, positive down). */
  inclination: number;
  /** North, east, down components (nT). */
  x: number;
  y: number;
  z: number;
  /** Horizontal intensity H and total intensity F (nT). */
  h: number;
  f: number;
  /** Secular variation (per year): dD/dt, dI/dt (deg/yr); dX..dF/dt (nT/yr). */
  declinationRate: number;
  inclinationRate: number;
  xRate: number;
  yRate: number;
  zRate: number;
  hRate: number;
  fRate: number;
  /** Decimal year the model was evaluated at (after clamping). */
  decimalYear: number;
  /** The requested date lies in the model's validity period. */
  inValidityPeriod: boolean;
  /** H < 2000 nT: compass headings unreliable (WMM blackout zone). */
  blackoutZone: boolean;
  /** H < 6000 nT: compass accuracy degraded (WMM caution zone). */
  cautionZone: boolean;
}

/**
 * Decimal year (UTC) of a date (`Date` or epoch milliseconds), e.g. 2025-07-02T12:00Z ≈ 2025.5.
 * Fraction = elapsed time since Jan 1 00:00 UTC / length of that year (leap years included).
 */
export function decimalYearFromDate(date: Date | number): number {
  const ms = typeof date === 'number' ? date : date.getTime();
  if (!Number.isFinite(ms)) return NaN;
  const y = new Date(ms).getUTCFullYear();
  const t0 = Date.UTC(y, 0, 1);
  const t1 = Date.UTC(y + 1, 0, 1);
  return y + (ms - t0) / (t1 - t0);
}

/** Scratch arrays per nMax (evaluation is synchronous ⇒ module-level reuse is safe). */
let scratchN = -1;
let P = new Float64Array(0);
let dP = new Float64Array(0);
let norm = new Float64Array(0);
let rr = new Float64Array(0);
let cosM = new Float64Array(0);
let sinM = new Float64Array(0);
let PS = new Float64Array(0);

function ensureScratch(nMax: number): void {
  if (scratchN === nMax) return;
  const size = ((nMax + 1) * (nMax + 2)) / 2;
  P = new Float64Array(size);
  dP = new Float64Array(size);
  norm = new Float64Array(size);
  rr = new Float64Array(nMax + 1);
  cosM = new Float64Array(nMax + 1);
  sinM = new Float64Array(nMax + 1);
  PS = new Float64Array(nMax + 1);
  // Gauss-normalised → Schmidt semi-normalised ratio: S(n,0) = S(n−1,0)(2n−1)/n,
  // S(n,m) = S(n,m−1) √((n−m+1)(m = 1 ? 2 : 1)/(n+m)).
  norm[0] = 1;
  for (let n = 1; n <= nMax; n++) {
    const k0 = (n * (n + 1)) / 2;
    norm[k0] = (norm[((n - 1) * n) / 2] * (2 * n - 1)) / n;
    for (let m = 1; m <= n; m++) norm[k0 + m] = norm[k0 + m - 1] * Math.sqrt(((n - m + 1) * (m === 1 ? 2 : 1)) / (n + m));
  }
  scratchN = nMax;
}

/**
 * Schmidt semi-normalised P̆ₙᵐ(x) and dP̆ₙᵐ/dφ′ (x = sin φ′, derivative w.r.t. LATITUDE) into
 * the module scratch arrays, via the Gauss-normalised recursions (stable to n = 12 at any φ′).
 */
function legendre(x: number, nMax: number): void {
  const z = Math.sqrt(Math.max(0, (1 - x) * (1 + x))); // cos φ′
  P[0] = 1;
  dP[0] = 0;
  for (let n = 1; n <= nMax; n++) {
    const k0 = (n * (n + 1)) / 2;
    for (let m = 0; m <= n; m++) {
      const k = k0 + m;
      if (n === m) {
        const k1 = ((n - 1) * n) / 2 + m - 1;
        P[k] = z * P[k1];
        dP[k] = z * dP[k1] + x * P[k1];
      } else if (n === 1 && m === 0) {
        P[k] = x * P[0];
        dP[k] = x * dP[0] - z * P[0];
      } else {
        const k1 = ((n - 2) * (n - 1)) / 2 + m;
        const k2 = ((n - 1) * n) / 2 + m;
        if (m > n - 2) {
          P[k] = x * P[k2];
          dP[k] = x * dP[k2] - z * P[k2];
        } else {
          const c = ((n - 1) * (n - 1) - m * m) / ((2 * n - 1) * (2 * n - 3));
          P[k] = x * P[k2] - c * P[k1];
          dP[k] = x * dP[k2] - z * P[k2] - c * dP[k1];
        }
      }
    }
  }
  const size = ((nMax + 1) * (nMax + 2)) / 2;
  for (let k = 0; k < size; k++) {
    P[k] *= norm[k];
    // the recursion differentiates w.r.t. colatitude: dP/dφ′ = −dP/dθ
    dP[k] = -dP[k] * norm[k];
  }
}

/**
 * Field (or secular variation, with ġ / ḣ) in geocentric spherical components
 * (north, east, down along the geocentric meridian), from the scratch Legendre values.
 */
function summation(gc: ArrayLike<number>, hc: ArrayLike<number>, nMax: number, sinPhi: number, cosPhi: number, out: Float64Array): void {
  let bx = 0,
    by = 0,
    bz = 0;
  for (let n = 1; n <= nMax; n++) {
    const k0 = (n * (n + 1)) / 2;
    const r = rr[n];
    for (let m = 0; m <= n; m++) {
      const k = k0 + m;
      const a = gc[k] * cosM[m] + hc[k] * sinM[m];
      bz -= r * a * (n + 1) * P[k];
      by += r * (gc[k] * sinM[m] - hc[k] * cosM[m]) * m * P[k];
      bx -= r * a * dP[k];
    }
  }
  if (Math.abs(cosPhi) > 1e-10) by /= cosPhi;
  else {
    // pole: Y′ = Σ (a/r)ⁿ⁺² (g sin λ − h cos λ) S(n,1) lim P_n¹/cos φ′ (Gauss-normalised recursion)
    by = 0;
    PS[0] = 1;
    for (let n = 1; n <= nMax; n++) {
      const k = (n * (n + 1)) / 2 + 1;
      if (n === 1) PS[n] = PS[n - 1];
      else {
        const c = ((n - 1) * (n - 1) - 1) / ((2 * n - 1) * (2 * n - 3));
        PS[n] = sinPhi * PS[n - 1] - c * PS[n - 2];
      }
      by += rr[n] * (gc[k] * sinM[1] - hc[k] * cosM[1]) * PS[n] * norm[k];
    }
  }
  out[0] = bx;
  out[1] = by;
  out[2] = bz;
}

const NAN_ELEMENTS: MagneticElements = Object.freeze({
  declination: NaN,
  inclination: NaN,
  x: NaN,
  y: NaN,
  z: NaN,
  h: NaN,
  f: NaN,
  declinationRate: NaN,
  inclinationRate: NaN,
  xRate: NaN,
  yRate: NaN,
  zRate: NaN,
  hRate: NaN,
  fRate: NaN,
  decimalYear: NaN,
  inValidityPeriod: false,
  blackoutZone: false,
  cautionZone: false,
});

/**
 * All magnetic elements at a geodetic position and date.
 * @param latDeg geodetic latitude (deg, clamped to [−90, 90])
 * @param lonDeg longitude (deg, east positive; any range)
 * @param altKm height above the WGS84 ellipsoid (km; MSL is fine for declination)
 * @param decimalYear e.g. 2026.5 (default: now); non-finite ⇒ now
 */
export function magneticField(latDeg: number, lonDeg: number, altKm = 0, decimalYear?: number, model: MagneticModel = WMM2025): MagneticElements {
  if (!Number.isFinite(latDeg) || !Number.isFinite(lonDeg)) return { ...NAN_ELEMENTS };
  const alt = Number.isFinite(altKm) ? altKm : 0;
  let year = decimalYear !== undefined && Number.isFinite(decimalYear) ? decimalYear : decimalYearFromDate(Date.now());
  const inValidityPeriod = year >= model.validFrom && year < model.validUntil;
  year = Math.min(model.validUntil + MAX_EXTRAPOLATION_YEARS, Math.max(model.validFrom - MAX_EXTRAPOLATION_YEARS, year));
  const nMax = model.nMax;
  ensureScratch(nMax);

  // 1) geodetic → geocentric spherical
  const lat = Math.max(-90, Math.min(90, latDeg)) * DEG;
  const lon = lonDeg * DEG;
  const sinLat = Math.sin(lat),
    cosLat = Math.abs(latDeg) >= 90 ? 0 : Math.cos(lat);
  const rc = A_KM / Math.sqrt(1 - E2 * sinLat * sinLat);
  const xp = (rc + alt) * cosLat;
  const zp = (rc * (1 - E2) + alt) * sinLat;
  const r = Math.hypot(xp, zp);
  const phiG = Math.asin(Math.max(-1, Math.min(1, zp / r))); // geocentric latitude
  const sinPhi = Math.sin(phiG);
  const cosPhi = Math.abs(latDeg) >= 90 ? 0 : Math.cos(phiG);
  for (let n = 0; n <= nMax; n++) rr[n] = Math.pow(RE_KM / r, n + 2);
  cosM[0] = 1;
  sinM[0] = 0;
  const cl = Math.cos(lon),
    sl = Math.sin(lon);
  for (let m = 1; m <= nMax; m++) {
    // angle addition (as the reference code) keeps cos/sin(mλ) exactly consistent
    cosM[m] = cosM[m - 1] * cl - sinM[m - 1] * sl;
    sinM[m] = cosM[m - 1] * sl + sinM[m - 1] * cl;
  }

  // 2) coefficients at the date
  const dt = year - model.epoch;
  const size = ((nMax + 1) * (nMax + 2)) / 2;
  const g = new Float64Array(size),
    h = new Float64Array(size);
  for (let k = 0; k < size; k++) {
    g[k] = model.g[k] + dt * model.gDot[k];
    h[k] = model.h[k] + dt * model.hDot[k];
  }

  // 3) spherical field and secular variation
  legendre(sinPhi, nMax);
  const B = new Float64Array(3),
    S = new Float64Array(3);
  summation(g, h, nMax, sinPhi, cosPhi, B);
  summation(model.gDot, model.hDot, nMax, sinPhi, cosPhi, S);

  // 4) rotate to the geodetic frame (ψ = φ′ − φ)
  const psi = phiG - lat;
  const cp = Math.cos(psi),
    sp = Math.sin(psi);
  const X = B[0] * cp - B[2] * sp;
  const Y = B[1];
  const Z = B[0] * sp + B[2] * cp;
  const Xd = S[0] * cp - S[2] * sp;
  const Yd = S[1];
  const Zd = S[0] * sp + S[2] * cp;
  const H = Math.hypot(X, Y);
  const F = Math.hypot(H, Z);
  const Hd = H > 0 ? (X * Xd + Y * Yd) / H : 0;
  const Fd = F > 0 ? (X * Xd + Y * Yd + Z * Zd) / F : 0;
  return {
    declination: Math.atan2(Y, X) / DEG,
    inclination: Math.atan2(Z, H) / DEG,
    x: X,
    y: Y,
    z: Z,
    h: H,
    f: F,
    declinationRate: H > 0 ? (X * Yd - Y * Xd) / (H * H) / DEG : 0,
    inclinationRate: F > 0 ? (H * Zd - Z * Hd) / (F * F) / DEG : 0,
    xRate: Xd,
    yRate: Yd,
    zRate: Zd,
    hRate: Hd,
    fRate: Fd,
    decimalYear: year,
    inValidityPeriod,
    blackoutZone: H < 2000,
    cautionZone: H < 6000,
  };
}

/** Magnetic declination (deg, east positive) — true heading = magnetic heading + declination. */
export function declination(latDeg: number, lonDeg: number, altKm = 0, decimalYear?: number): number {
  return magneticField(latDeg, lonDeg, altKm, decimalYear).declination;
}

/** Magnetic inclination / dip (deg, positive down). */
export function inclination(latDeg: number, lonDeg: number, altKm = 0, decimalYear?: number): number {
  return magneticField(latDeg, lonDeg, altKm, decimalYear).inclination;
}

/** Total field intensity F (nT). */
export function intensity(latDeg: number, lonDeg: number, altKm = 0, decimalYear?: number): number {
  return magneticField(latDeg, lonDeg, altKm, decimalYear).f;
}

/** Heading (deg clockwise from north) wrapped to [0, 360). */
function wrap360(d: number): number {
  const r = d % 360;
  return r < 0 ? r + 360 : r === 0 ? 0 : r;
}

/** Magnetic → true heading (deg, [0, 360)) with an east-positive declination. */
export function magneticToTrueHeading(magneticDeg: number, declinationDeg: number): number {
  return wrap360(magneticDeg + declinationDeg);
}

/** True → magnetic heading (deg, [0, 360)) with an east-positive declination. */
export function trueToMagneticHeading(trueDeg: number, declinationDeg: number): number {
  return wrap360(trueDeg - declinationDeg);
}
