/**
 * Homographies x2 ~ H x1 (pixels or normalised coordinates, flat [x0, y0, …]):
 * normalised DLT (Hartley & Zisserman Alg. 4.2), transfer error, and the analytical
 * decomposition H = R + t nᵀ (Ma, Soatto, Košecká & Sastry, "An Invitation to 3-D
 * Vision", 2004, §5.3.3, after Faugeras & Lustman 1988).
 */
import type { Mat3, Vec3 } from '../core/types';
import { Matrix } from '../math/matrix';
import { nullVector, symEig } from '../math/decomp';
import { orthonormalize } from '../math/rotation';
import { hartleyNormalize, nearlyCollinear, smallestEigenvector } from './internal';

const Q8 = new Matrix(8, 9);
const T1 = new Float64Array(9);
const T2 = new Float64Array(9);
let N1 = new Float64Array(64);
let N2 = new Float64Array(64);
const ATA = new Float64Array(81);
const R0 = new Float64Array(9);
const R1 = new Float64Array(9);

function addRows(i: number, a: Float64Array, b: Float64Array, out: Float64Array, useAtA: boolean): void {
  const x = a[2 * i],
    y = a[2 * i + 1],
    u = b[2 * i],
    v = b[2 * i + 1];
  // [x, y, 1, 0, 0, 0, −u x, −u y, −u] and [0, 0, 0, x, y, 1, −v x, −v y, −v]
  R0[0] = x;
  R0[1] = y;
  R0[2] = 1;
  R0[3] = 0;
  R0[4] = 0;
  R0[5] = 0;
  R0[6] = -u * x;
  R0[7] = -u * y;
  R0[8] = -u;
  R1[0] = 0;
  R1[1] = 0;
  R1[2] = 0;
  R1[3] = x;
  R1[4] = y;
  R1[5] = 1;
  R1[6] = -v * x;
  R1[7] = -v * y;
  R1[8] = -v;
  if (useAtA) {
    for (let p = 0; p < 9; p++) {
      const r0 = R0[p],
        r1 = R1[p];
      const o = 9 * p;
      for (let q = p; q < 9; q++) out[o + q] += r0 * R0[q] + r1 * R1[q];
    }
  } else {
    out.set(R0, 18 * i);
    out.set(R1, 18 * i + 9);
  }
}

/**
 * Normalised DLT on correspondences idx[0..n) (all n when idx is null), n ≥ 4.
 * The result is scaled to unit Frobenius norm with the sign that maps the first
 * point to positive w. Null for degenerate input.
 */
export function homographyDLTIdx(
  p1: ArrayLike<number>,
  p2: ArrayLike<number>,
  idx: Uint32Array | null,
  n: number,
): Mat3 | null {
  if (n < 4) return null;
  if (N1.length < 2 * n) {
    N1 = new Float64Array(4 * n);
    N2 = new Float64Array(4 * n);
  }
  if (!hartleyNormalize(p1, n, idx, N1, T1) || !hartleyNormalize(p2, n, idx, N2, T2)) return null;
  const h = new Float64Array(9);
  if (n === 4) {
    for (let i = 0; i < 4; i++) addRows(i, N1, N2, Q8.data, false);
    const v = nullVector(Q8);
    for (let k = 0; k < 9; k++) h[k] = v[k];
  } else {
    ATA.fill(0);
    for (let i = 0; i < n; i++) addRows(i, N1, N2, ATA, true);
    smallestEigenvector(ATA, 9, h);
  }
  if (!Number.isFinite(h[0])) return null;
  // H = T2⁻¹ Ĥ T1, T = [[s,0,a],[0,s,b],[0,0,1]], T⁻¹ = [[1/s,0,−a/s],[0,1/s,−b/s],[0,0,1]]
  const s1 = T1[0],
    a1 = T1[2],
    b1 = T1[5];
  const is2 = 1 / T2[0],
    a2 = T2[2],
    b2 = T2[5];
  // M = Ĥ T1
  const m = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    m[3 * i] = h[3 * i] * s1;
    m[3 * i + 1] = h[3 * i + 1] * s1;
    m[3 * i + 2] = h[3 * i] * a1 + h[3 * i + 1] * b1 + h[3 * i + 2];
  }
  const H = new Float64Array(9);
  for (let j = 0; j < 3; j++) {
    H[j] = is2 * (m[j] - a2 * m[6 + j]);
    H[3 + j] = is2 * (m[3 + j] - b2 * m[6 + j]);
    H[6 + j] = m[6 + j];
  }
  let nrm = 0;
  for (let k = 0; k < 9; k++) nrm += H[k] * H[k];
  if (!(nrm > 0) || !Number.isFinite(nrm)) return null;
  const k0 = idx ? idx[0] : 0;
  const w = H[6] * p1[2 * k0] + H[7] * p1[2 * k0 + 1] + H[8];
  const sc = (w < 0 ? -1 : 1) / Math.sqrt(nrm);
  for (let k = 0; k < 9; k++) H[k] *= sc;
  return H;
}

/** Normalised DLT homography from n ≥ 4 correspondences (p2 ~ H p1). */
export function homographyDLT(p1: ArrayLike<number>, p2: ArrayLike<number>, n: number): Mat3 | null {
  return homographyDLTIdx(p1, p2, null, n);
}

/**
 * One-sided squared transfer error ‖x2 − π(H x1)‖² (in the units of x2, as COLMAP).
 * Infinity if x1 maps to infinity (w = 0). Independent of the sign of H.
 */
export function homographyTransferErrorSq(H: ArrayLike<number>, x1: number, y1: number, x2: number, y2: number): number {
  const w = H[6] * x1 + H[7] * y1 + H[8];
  if (!(w !== 0) || !Number.isFinite(w)) return Infinity;
  const iw = 1 / w;
  const du = (H[0] * x1 + H[1] * y1 + H[2]) * iw - x2;
  const dv = (H[3] * x1 + H[4] * y1 + H[5]) * iw - y2;
  return du * du + dv * dv;
}

/** Transfer errors of all n correspondences into out. */
export function homographyTransferErrorsSq(
  H: ArrayLike<number>,
  p1: ArrayLike<number>,
  p2: ArrayLike<number>,
  n: number,
  out: Float64Array,
): void {
  homographyTransferErrorsSqRange(H, p1, p2, 0, n, out);
}

/** Transfer errors of correspondences [start, end) into out[start..end). */
export function homographyTransferErrorsSqRange(
  H: ArrayLike<number>,
  p1: ArrayLike<number>,
  p2: ArrayLike<number>,
  start: number,
  end: number,
  out: Float64Array,
): void {
  const h0 = H[0],
    h1 = H[1],
    h2 = H[2],
    h3 = H[3],
    h4 = H[4],
    h5 = H[5],
    h6 = H[6],
    h7 = H[7],
    h8 = H[8];
  for (let i = start; i < end; i++) {
    const x = p1[2 * i],
      y = p1[2 * i + 1];
    const w = h6 * x + h7 * y + h8;
    if (w === 0) {
      out[i] = Infinity;
      continue;
    }
    const iw = 1 / w;
    const du = (h0 * x + h1 * y + h2) * iw - p2[2 * i];
    const dv = (h3 * x + h4 * y + h5) * iw - p2[2 * i + 1];
    out[i] = du * du + dv * dv;
  }
}

/** True if any three of the four sample points are (nearly) collinear in either image. */
export function homographySampleDegenerate(
  p1: ArrayLike<number>,
  p2: ArrayLike<number>,
  sample: ArrayLike<number>,
): boolean {
  for (let img = 0; img < 2; img++) {
    const p = img === 0 ? p1 : p2;
    for (let a = 0; a < 4; a++) {
      // the triple without index a
      const i = sample[a === 0 ? 1 : 0],
        j = sample[a <= 1 ? 2 : 1],
        k = sample[a <= 2 ? 3 : 2];
      if (nearlyCollinear(p[2 * i], p[2 * i + 1], p[2 * j], p[2 * j + 1], p[2 * k], p[2 * k + 1], 1e-3)) return true;
    }
  }
  return false;
}

/** One solution of the homography decomposition. */
export interface HomographyDecomposition {
  R: Mat3;
  /** Translation divided by the plane distance d (x2 = R x1 + t·d). Zero for pure rotation. */
  t: Vec3;
  /** Unit plane normal in camera-1 coordinates (plane nᵀX = d, d > 0). */
  n: Vec3;
}

function cross(a: ArrayLike<number>, b: ArrayLike<number>, out: Float64Array): Float64Array {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
  return out;
}

/**
 * Decomposes a homography between NORMALISED coordinates (x2 ~ H x1, as K2⁻¹ H_px K1)
 * into H ∝ R + t nᵀ (X2 = R X1 + T, plane nᵀ X1 = d, t = T/d). The sign of H is
 * fixed with the correspondences x1, x2 (positive depth ratio for the majority) when
 * given. Returns the four solutions {(R1, t1, n1), (R1, −t1, −n1), (R2, t2, n2),
 * (R2, −t2, −n2)}, or a single {R, 0, e_z} when H is (numerically) a rotation
 * (σ1 − σ3 < 1e-9).
 */
export function decomposeHomography(
  Hin: ArrayLike<number>,
  x1?: ArrayLike<number>,
  x2?: ArrayLike<number>,
  numPoints = 0,
): HomographyDecomposition[] {
  const H = Float64Array.from(Hin as ArrayLike<number>);
  // Scale by σ2 (middle singular value) using the eigenvalues of HᵀH.
  const HtH = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) HtH[3 * i + j] = H[i] * H[j] + H[3 + i] * H[3 + j] + H[6 + i] * H[6 + j];
  let { values } = symEig(HtH, 3);
  const s2 = Math.sqrt(Math.max(values[1], 0));
  if (!(s2 > 0) || !Number.isFinite(s2)) return [];
  let sgn = 1;
  if (x1 && x2 && numPoints > 0) {
    let pos = 0;
    for (let i = 0; i < numPoints; i++) {
      const a = x1[2 * i],
        b = x1[2 * i + 1];
      const w0 = H[0] * a + H[1] * b + H[2],
        w1 = H[3] * a + H[4] * b + H[5],
        w2 = H[6] * a + H[7] * b + H[8];
      if (x2[2 * i] * w0 + x2[2 * i + 1] * w1 + w2 > 0) pos++;
      else pos--;
    }
    if (pos < 0) sgn = -1;
  } else if (H[8] < 0) sgn = -1;
  for (let k = 0; k < 9; k++) H[k] *= sgn / s2;
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) HtH[3 * i + j] = H[i] * H[j] + H[3 + i] * H[3 + j] + H[6 + i] * H[6 + j];
  const eig = symEig(HtH, 3);
  values = eig.values; // ascending: σ3², σ2² (= 1), σ1²
  const V = eig.vectors;
  const sig1sq = values[2],
    sig3sq = values[0];
  if (!(sig1sq - sig3sq > 1e-9)) {
    return [{ R: orthonormalize(H), t: new Float64Array(3), n: Float64Array.of(0, 0, 1) }];
  }
  const v1 = Float64Array.of(V[2], V[5], V[8]); // largest
  const v2 = Float64Array.of(V[1], V[4], V[7]);
  const v3 = Float64Array.of(V[0], V[3], V[6]); // smallest
  const a = Math.sqrt(Math.max(0, 1 - sig3sq)),
    b = Math.sqrt(Math.max(0, sig1sq - 1)),
    c = 1 / Math.sqrt(sig1sq - sig3sq);
  const u1 = new Float64Array(3),
    u2 = new Float64Array(3);
  for (let k = 0; k < 3; k++) {
    u1[k] = (a * v1[k] + b * v3[k]) * c;
    u2[k] = (a * v1[k] - b * v3[k]) * c;
  }
  const Hv2 = new Float64Array(3);
  for (let i = 0; i < 3; i++) Hv2[i] = H[3 * i] * v2[0] + H[3 * i + 1] * v2[1] + H[3 * i + 2] * v2[2];
  const out: HomographyDecomposition[] = [];
  const tmp = new Float64Array(3);
  for (const u of [u1, u2]) {
    const Hu = new Float64Array(3);
    for (let i = 0; i < 3; i++) Hu[i] = H[3 * i] * u[0] + H[3 * i + 1] * u[1] + H[3 * i + 2] * u[2];
    const v2u = cross(v2, u, new Float64Array(3));
    const Hv2Hu = cross(Hv2, Hu, tmp);
    // U = [v2, u, v2×u], W = [Hv2, Hu, Hv2×Hu], R = W Uᵀ
    const R = new Float64Array(9);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) R[3 * i + j] = Hv2[i] * v2[j] + Hu[i] * u[j] + Hv2Hu[i] * v2u[j];
    const Ro = orthonormalize(R);
    const nrm = v2u; // N = v2 × u (unit)
    // t = (H − R) N
    const t = new Float64Array(3);
    for (let i = 0; i < 3; i++) {
      let s = 0;
      for (let j = 0; j < 3; j++) s += (H[3 * i + j] - Ro[3 * i + j]) * nrm[j];
      t[i] = s;
    }
    out.push({ R: Ro, t, n: Float64Array.from(nrm) });
    out.push({ R: Float64Array.from(Ro), t: Float64Array.of(-t[0], -t[1], -t[2]), n: Float64Array.of(-nrm[0], -nrm[1], -nrm[2]) });
  }
  return out;
}
