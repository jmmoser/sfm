/**
 * CPU descriptor matching (brute force and geometry-guided) for RootSIFT features.
 *
 * Semantics (docs/ARCHITECTURE.md §4.4):
 *  - Descriptor similarity is the dot product of the unit-L2 RootSIFT vectors
 *    (dequantised /512 and renormalised). It is evaluated EXACTLY as
 *    idot(qa, qb) * invNorm(qa) * invNorm(qb) on the quantised bytes (integer dot,
 *    inverse norms cached per descriptor array in a WeakMap), which is
 *    mathematically identical to Float32 unit vectors and bit-identical between the
 *    plain-JS and the WASM-SIMD brute-force paths.
 *  - distance = sqrt(max(0, 2 - 2 dot)) (Euclidean distance between unit vectors).
 *  - Lowe ratio test on distances: best < ratio * secondBest (no second candidate
 *    = pass), best <= maxDistance.
 *  - Mutual: the matched b feature's best a (over the same candidate structure) must
 *    be the query a feature. No ratio test is applied in the reverse direction.
 *  - Ties: the smaller index wins in both directions.
 *  - Output: flat [i, j, ...] pairs sorted by ascending index in a.
 */

import type { FeatureSet, Mat3, MatchPairs } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';
import { createSimdKernels, simdAvailable } from './simd';
import type { SimdKernels } from './simd';

export interface MatchOptions {
  /** Lowe ratio on distances. Default 0.8. */
  ratio?: number;
  /** Require mutual nearest neighbours. Default true. */
  mutual?: boolean;
  /** Maximum descriptor distance on unit-norm descriptors. Default 0.7. */
  maxDistance?: number;
}

export type MatchGuide =
  /** Epipolar distance in image b: x_b^T F x_a = 0. */
  | { kind: 'F'; F: Mat3; maxErrorPx: number }
  /** Transfer distance in image b: x_b ~ H x_a. */
  | { kind: 'H'; H: Mat3; maxErrorPx: number }
  /**
   * Predicted positions in b, 2 per feature of a. A NaN prediction means
   * "unknown": that feature is matched against all of b (unguided).
   */
  | { kind: 'positions'; predicted: Float32Array; radiusPx: number };

const D = DESCRIPTOR_SIZE;

// ---------------------------------------------------------------------------
// Descriptor normalisation cache
// ---------------------------------------------------------------------------

const unitCache = new WeakMap<Uint8Array, Float32Array>();

/**
 * Unit-L2 Float32 descriptors (dequantised /512 then renormalised) of a FeatureSet,
 * computed once and cached per descriptor array.
 */
export function unitDescriptors(fs: FeatureSet): Float32Array {
  const src = fs.descriptors;
  let out = unitCache.get(src);
  if (out && out.length === fs.count * D) return out;
  const n = fs.count;
  out = new Float32Array(n * D);
  for (let i = 0; i < n; i++) {
    const o = i * D;
    let ss = 0;
    for (let k = 0; k < D; k++) {
      const v = src[o + k];
      ss += v * v;
    }
    const inv = ss > 0 ? 1 / Math.sqrt(ss) : 0;
    for (let k = 0; k < D; k++) out[o + k] = src[o + k] * inv;
  }
  unitCache.set(src, out);
  return out;
}

const invNormCache = new WeakMap<Uint8Array, Float64Array>();

/** 1 / L2 norm of each quantised descriptor (0 for all-zero descriptors), cached. */
export function descriptorInvNorms(fs: FeatureSet): Float64Array {
  const src = fs.descriptors;
  let out = invNormCache.get(src);
  if (out && out.length === fs.count) return out;
  const n = fs.count;
  out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * D;
    let ss = 0;
    for (let k = 0; k < D; k++) ss += src[o + k] * src[o + k];
    out[i] = ss > 0 ? 1 / Math.sqrt(ss) : 0;
  }
  invNormCache.set(src, out);
  return out;
}

/** Exact integer dot product of two quantised descriptors. */
function idot(A: Uint8Array, ao: number, B: Uint8Array, bo: number): number {
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  for (let k = 0; k < D; k += 4) {
    s0 += A[ao + k] * B[bo + k];
    s1 += A[ao + k + 1] * B[bo + k + 1];
    s2 += A[ao + k + 2] * B[bo + k + 2];
    s3 += A[ao + k + 3] * B[bo + k + 3];
  }
  return s0 + s1 + s2 + s3;
}

/** Unit-vector dot product of feature i of a and feature j of b (matcher's similarity). */
export function descriptorDot(a: FeatureSet, i: number, b: FeatureSet, j: number): number {
  return idot(a.descriptors, i * D, b.descriptors, j * D) * descriptorInvNorms(a)[i] * descriptorInvNorms(b)[j];
}

/** Descriptor distance between feature i of a and feature j of b. */
export function descriptorDistance(a: FeatureSet, i: number, b: FeatureSet, j: number): number {
  return Math.sqrt(Math.max(0, 2 - 2 * descriptorDot(a, i, b, j)));
}

interface Resolved {
  ratio: number;
  mutual: boolean;
  maxDistance: number;
}

function resolve(opts?: MatchOptions): Resolved {
  return {
    ratio: opts?.ratio ?? 0.8,
    mutual: opts?.mutual ?? true,
    maxDistance: opts?.maxDistance ?? 0.7,
  };
}

/** Growable pair buffer. */
class PairBuffer {
  data = new Uint32Array(1024);
  n = 0;
  push(i: number, j: number): void {
    if (this.n + 2 > this.data.length) {
      const d = new Uint32Array(this.data.length * 2);
      d.set(this.data);
      this.data = d;
    }
    this.data[this.n++] = i;
    this.data[this.n++] = j;
  }
  result(): Uint32Array {
    return this.data.slice(0, this.n);
  }
}

function distOfDot(d: number): number {
  return Math.sqrt(Math.max(0, 2 - 2 * d));
}

/** Common final filter: ratio test, max distance, mutual check. */
function emit(
  pairs: PairBuffer,
  na: number,
  bestJ: Int32Array,
  bestDot: Float64Array,
  secondDot: Float64Array,
  colBestI: Int32Array,
  o: Resolved,
): MatchPairs {
  for (let i = 0; i < na; i++) {
    const j = bestJ[i];
    if (j < 0) continue;
    const d1 = distOfDot(bestDot[i]);
    if (d1 > o.maxDistance) continue;
    const sd = secondDot[i];
    if (sd > -Infinity) {
      const d2 = distOfDot(sd);
      if (!(d1 < o.ratio * d2)) continue;
    }
    if (o.mutual && colBestI[j] !== i) continue;
    pairs.push(i, j);
  }
  return pairs.result();
}

// ---------------------------------------------------------------------------
// Brute force
// ---------------------------------------------------------------------------

interface BestState {
  bestJ: Int32Array;
  bestDot: Float64Array;
  secondDot: Float64Array;
  colBest: Float64Array;
  colBestI: Int32Array;
}

function newState(na: number, nb: number): BestState {
  return {
    bestJ: new Int32Array(na).fill(-1),
    bestDot: new Float64Array(na).fill(-Infinity),
    secondDot: new Float64Array(na).fill(-Infinity),
    colBest: new Float64Array(nb).fill(-Infinity),
    colBestI: new Int32Array(nb).fill(-1),
  };
}

/** Plain JS: exact integer dots, 4 a-descriptors per pass over b (register reuse). */
function bruteForceJS(A: Uint8Array, B: Uint8Array, na: number, nb: number, invA: Float64Array, invB: Float64Array, st: BestState): void {
  const { bestJ, bestDot, secondDot, colBest, colBestI } = st;
  let i = 0;
  for (; i + 4 <= na; i += 4) {
    const a0 = i * D;
    const a1 = a0 + D;
    const a2 = a1 + D;
    const a3 = a2 + D;
    const ia0 = invA[i], ia1 = invA[i + 1], ia2 = invA[i + 2], ia3 = invA[i + 3];
    let b0 = -Infinity, b1 = -Infinity, b2 = -Infinity, b3 = -Infinity;
    let c0 = -Infinity, c1 = -Infinity, c2 = -Infinity, c3 = -Infinity;
    let j0 = -1, j1 = -1, j2 = -1, j3 = -1;
    for (let j = 0; j < nb; j++) {
      const bo = j * D;
      let d0 = 0, d1 = 0, d2 = 0, d3 = 0;
      for (let k = 0; k < D; k += 2) {
        const x0 = B[bo + k];
        const x1 = B[bo + k + 1];
        d0 += A[a0 + k] * x0 + A[a0 + k + 1] * x1;
        d1 += A[a1 + k] * x0 + A[a1 + k + 1] * x1;
        d2 += A[a2 + k] * x0 + A[a2 + k + 1] * x1;
        d3 += A[a3 + k] * x0 + A[a3 + k + 1] * x1;
      }
      const ib = invB[j];
      const s0 = d0 * ia0 * ib;
      const s1 = d1 * ia1 * ib;
      const s2 = d2 * ia2 * ib;
      const s3 = d3 * ia3 * ib;
      if (s0 > b0) { c0 = b0; b0 = s0; j0 = j; } else if (s0 > c0) c0 = s0;
      if (s1 > b1) { c1 = b1; b1 = s1; j1 = j; } else if (s1 > c1) c1 = s1;
      if (s2 > b2) { c2 = b2; b2 = s2; j2 = j; } else if (s2 > c2) c2 = s2;
      if (s3 > b3) { c3 = b3; b3 = s3; j3 = j; } else if (s3 > c3) c3 = s3;
      let cb = colBest[j];
      let ci = -1;
      if (s0 > cb) { cb = s0; ci = i; }
      if (s1 > cb) { cb = s1; ci = i + 1; }
      if (s2 > cb) { cb = s2; ci = i + 2; }
      if (s3 > cb) { cb = s3; ci = i + 3; }
      if (ci >= 0) {
        colBest[j] = cb;
        colBestI[j] = ci;
      }
    }
    bestJ[i] = j0; bestDot[i] = b0; secondDot[i] = c0;
    bestJ[i + 1] = j1; bestDot[i + 1] = b1; secondDot[i + 1] = c1;
    bestJ[i + 2] = j2; bestDot[i + 2] = b2; secondDot[i + 2] = c2;
    bestJ[i + 3] = j3; bestDot[i + 3] = b3; secondDot[i + 3] = c3;
  }
  for (; i < na; i++) {
    const ao = i * D;
    const ia = invA[i];
    let b0 = -Infinity, c0 = -Infinity, j0 = -1;
    for (let j = 0; j < nb; j++) {
      const s = idot(A, ao, B, j * D) * ia * invB[j];
      if (s > b0) { c0 = b0; b0 = s; j0 = j; } else if (s > c0) c0 = s;
      if (s > colBest[j]) { colBest[j] = s; colBestI[j] = i; }
    }
    bestJ[i] = j0; bestDot[i] = b0; secondDot[i] = c0;
  }
}

let matchKernels: SimdKernels | null | undefined;

/** WASM SIMD: integer dots of 4 a-descriptors against all of b per kernel call. */
function bruteForceSIMD(
  k: SimdKernels,
  A: Uint8Array,
  B: Uint8Array,
  na: number,
  nb: number,
  invA: Float64Array,
  invB: Float64Array,
  st: BestState,
): void {
  const na4 = (na + 3) & ~3;
  const aOff = 0;
  const bOff = na4 * 2 * D;
  const oOff = bOff + nb * 2 * D;
  k.ensure(oOff + nb * 16 + 64);
  const mem = k.memory.buffer;
  const a16 = new Int16Array(mem, aOff, na4 * D);
  a16.set(A.subarray(0, na * D));
  a16.fill(0, na * D);
  new Int16Array(mem, bOff, nb * D).set(B.subarray(0, nb * D));
  const out = new Int32Array(mem, oOff, nb * 4);
  const { bestJ, bestDot, secondDot, colBest, colBestI } = st;
  for (let i = 0; i < na; i += 4) {
    k.dot4(aOff + i * 2 * D, bOff, nb, oOff);
    const q1 = Math.min(4, na - i);
    const ia0 = invA[i];
    const ia1 = q1 > 1 ? invA[i + 1] : 0;
    const ia2 = q1 > 2 ? invA[i + 2] : 0;
    const ia3 = q1 > 3 ? invA[i + 3] : 0;
    let b0 = -Infinity, b1 = -Infinity, b2 = -Infinity, b3 = -Infinity;
    let c0 = -Infinity, c1 = -Infinity, c2 = -Infinity, c3 = -Infinity;
    let j0 = -1, j1 = -1, j2 = -1, j3 = -1;
    for (let j = 0, o = 0; j < nb; j++, o += 4) {
      const ib = invB[j];
      // Padding rows (q >= q1) get -Infinity so they never win anything.
      const s0 = out[o] * ia0 * ib;
      const s1 = q1 > 1 ? out[o + 1] * ia1 * ib : -Infinity;
      const s2 = q1 > 2 ? out[o + 2] * ia2 * ib : -Infinity;
      const s3 = q1 > 3 ? out[o + 3] * ia3 * ib : -Infinity;
      if (s0 > b0) { c0 = b0; b0 = s0; j0 = j; } else if (s0 > c0) c0 = s0;
      if (s1 > b1) { c1 = b1; b1 = s1; j1 = j; } else if (s1 > c1) c1 = s1;
      if (s2 > b2) { c2 = b2; b2 = s2; j2 = j; } else if (s2 > c2) c2 = s2;
      if (s3 > b3) { c3 = b3; b3 = s3; j3 = j; } else if (s3 > c3) c3 = s3;
      let cb = colBest[j];
      let ci = -1;
      if (s0 > cb) { cb = s0; ci = i; }
      if (s1 > cb) { cb = s1; ci = i + 1; }
      if (s2 > cb) { cb = s2; ci = i + 2; }
      if (s3 > cb) { cb = s3; ci = i + 3; }
      if (ci >= 0) {
        colBest[j] = cb;
        colBestI[j] = ci;
      }
    }
    bestJ[i] = j0; bestDot[i] = b0; secondDot[i] = c0;
    if (q1 > 1) { bestJ[i + 1] = j1; bestDot[i + 1] = b1; secondDot[i + 1] = c1; }
    if (q1 > 2) { bestJ[i + 2] = j2; bestDot[i + 2] = b2; secondDot[i + 2] = c2; }
    if (q1 > 3) { bestJ[i + 3] = j3; bestDot[i + 3] = b3; secondDot[i + 3] = c3; }
  }
}

/** True if the brute-force matcher currently uses the WASM SIMD kernels. */
export function matchUsesSIMD(): boolean {
  if (!simdAvailable()) return false;
  if (matchKernels === undefined) matchKernels = createSimdKernels(1 << 22);
  return matchKernels !== null;
}

/**
 * Brute-force mutual nearest-neighbour matching with Lowe's ratio test. Both
 * directions are resolved in a single pass over the Na x Nb dot products
 * (WASM SIMD when available, plain JS otherwise — identical results).
 */
export function matchCPU(a: FeatureSet, b: FeatureSet, opts?: MatchOptions): MatchPairs {
  const o = resolve(opts);
  const na = a.count;
  const nb = b.count;
  if (na === 0 || nb === 0) return new Uint32Array(0);
  const invA = descriptorInvNorms(a);
  const invB = descriptorInvNorms(b);
  const st = newState(na, nb);
  let done = false;
  if (matchUsesSIMD()) {
    try {
      bruteForceSIMD(matchKernels!, a.descriptors, b.descriptors, na, nb, invA, invB, st);
      done = true;
    } catch {
      // WASM memory growth refused: fall back to JS (state is still pristine only
      // if the failure happened before any row was processed — reset to be safe).
      Object.assign(st, newState(na, nb));
    }
  }
  if (!done) bruteForceJS(a.descriptors, b.descriptors, na, nb, invA, invB, st);
  return emit(new PairBuffer(), na, st.bestJ, st.bestDot, st.secondDot, st.colBestI, o);
}

// ---------------------------------------------------------------------------
// Guided matching
// ---------------------------------------------------------------------------

/** Uniform grid over the keypoints of b (counting-sort layout). */
class GridIndex {
  readonly gw: number;
  readonly gh: number;
  readonly cell: number;
  readonly start: Int32Array;
  readonly items: Int32Array;
  readonly x: Float32Array;
  readonly y: Float32Array;

  constructor(fs: FeatureSet, cell: number) {
    const n = fs.count;
    this.cell = Math.max(1, cell);
    // Cover the image frame plus keypoints slightly outside it.
    this.gw = Math.max(1, Math.ceil((fs.width + 1) / this.cell));
    this.gh = Math.max(1, Math.ceil((fs.height + 1) / this.cell));
    this.x = new Float32Array(n);
    this.y = new Float32Array(n);
    const cellOf = new Int32Array(n);
    const counts = new Int32Array(this.gw * this.gh + 1);
    for (let j = 0; j < n; j++) {
      const x = fs.keypoints[4 * j];
      const y = fs.keypoints[4 * j + 1];
      this.x[j] = x;
      this.y[j] = y;
      const c = this.cellIndex(x, y);
      cellOf[j] = c;
      counts[c + 1]++;
    }
    for (let c = 0; c < this.gw * this.gh; c++) counts[c + 1] += counts[c];
    this.start = counts;
    this.items = new Int32Array(n);
    const fillPos = counts.slice(0, this.gw * this.gh);
    for (let j = 0; j < n; j++) this.items[fillPos[cellOf[j]]++] = j;
  }

  cx(x: number): number {
    const c = Math.floor((x + 0.5) / this.cell);
    return c < 0 ? 0 : c >= this.gw ? this.gw - 1 : c;
  }
  cy(y: number): number {
    const c = Math.floor((y + 0.5) / this.cell);
    return c < 0 ? 0 : c >= this.gh ? this.gh - 1 : c;
  }
  cellIndex(x: number, y: number): number {
    return this.cy(y) * this.gw + this.cx(x);
  }
}

/**
 * Guided matching: for every feature of a, only features of b satisfying the
 * geometric constraint are candidates. Ratio test among the candidates; mutual
 * check among the guided candidate sets (b's best a over the a features that had
 * it as a candidate).
 */
export function matchGuidedCPU(a: FeatureSet, b: FeatureSet, guide: MatchGuide, opts?: MatchOptions): MatchPairs {
  const o = resolve(opts);
  const na = a.count;
  const nb = b.count;
  if (na === 0 || nb === 0) return new Uint32Array(0);
  const A = a.descriptors;
  const B = b.descriptors;
  const invA = descriptorInvNorms(a);
  const invB = descriptorInvNorms(b);
  const { bestJ, bestDot, secondDot, colBest, colBestI } = newState(na, nb);

  const consider = (i: number, ao: number, j: number): void => {
    const s = idot(A, ao, B, j * D) * invA[i] * invB[j];
    if (s > bestDot[i]) {
      secondDot[i] = bestDot[i];
      bestDot[i] = s;
      bestJ[i] = j;
    } else if (s > secondDot[i]) secondDot[i] = s;
    if (s > colBest[j]) {
      colBest[j] = s;
      colBestI[j] = i;
    }
  };

  if (guide.kind === 'F') {
    // Walk the epipolar band |l . x_b| <= tol |l_xy| through a grid over b: along
    // columns for flat lines (|l1| >= |l0|), along rows for steep ones.
    const F = guide.F;
    const tol = guide.maxErrorPx;
    // ~4 keypoints of b per cell on average (fewer cells to walk), at least the band width.
    const grid = new GridIndex(b, Math.max(8, 2 * tol, Math.sqrt((4 * b.width * b.height) / nb)));
    const cs = grid.cell;
    const gx = grid.x;
    const gy = grid.y;
    const items = grid.items;
    const start = grid.start;
    for (let i = 0; i < na; i++) {
      const xa = a.keypoints[4 * i];
      const ya = a.keypoints[4 * i + 1];
      const l0 = F[0] * xa + F[1] * ya + F[2];
      const l1 = F[3] * xa + F[4] * ya + F[5];
      const l2 = F[6] * xa + F[7] * ya + F[8];
      const nrm = Math.sqrt(l0 * l0 + l1 * l1);
      if (!(nrm > 1e-15)) continue;
      const lim = tol * nrm;
      const ao = i * D;
      const flat = Math.abs(l1) >= Math.abs(l0);
      const nMajor = flat ? grid.gw : grid.gh;
      // Half-width of the band along the minor axis.
      const half = lim / Math.abs(flat ? l1 : l0);
      for (let c = 0; c < nMajor; c++) {
        // Major-axis extent of this column/row (outer ones extended by a cell).
        const u0 = c === 0 ? -0.5 - cs : c * cs - 0.5;
        const u1 = c === nMajor - 1 ? (c + 1) * cs - 0.5 + cs : (c + 1) * cs - 0.5;
        // Minor coordinate of the line at both ends.
        const v0 = flat ? -(l0 * u0 + l2) / l1 : -(l1 * u0 + l2) / l0;
        const v1 = flat ? -(l0 * u1 + l2) / l1 : -(l1 * u1 + l2) / l0;
        const vlo = Math.min(v0, v1) - half;
        const vhi = Math.max(v0, v1) + half;
        const r0 = flat ? grid.cy(vlo) : grid.cx(vlo);
        const r1 = flat ? grid.cy(vhi) : grid.cx(vhi);
        for (let r = r0; r <= r1; r++) {
          const cell = flat ? r * grid.gw + c : c * grid.gw + r;
          for (let t = start[cell], e = start[cell + 1]; t < e; t++) {
            const j = items[t];
            const d = l0 * gx[j] + l1 * gy[j] + l2;
            if (d > lim || d < -lim) continue;
            consider(i, ao, j);
          }
        }
      }
    }
    return emit(new PairBuffer(), na, bestJ, bestDot, secondDot, colBestI, o);
  }

  // 'positions' and 'H': predicted positions + radius search in a grid over b.
  let radius: number;
  let pred: Float32Array;
  if (guide.kind === 'H') {
    radius = guide.maxErrorPx;
    pred = new Float32Array(2 * na);
    const H = guide.H;
    for (let i = 0; i < na; i++) {
      const x = a.keypoints[4 * i];
      const y = a.keypoints[4 * i + 1];
      const w = H[6] * x + H[7] * y + H[8];
      if (w > 1e-12) {
        pred[2 * i] = (H[0] * x + H[1] * y + H[2]) / w;
        pred[2 * i + 1] = (H[3] * x + H[4] * y + H[5]) / w;
      } else {
        // Maps to/behind infinity: cannot correspond to a finite point of b.
        pred[2 * i] = Infinity;
        pred[2 * i + 1] = Infinity;
      }
    }
  } else {
    radius = guide.radiusPx;
    pred = guide.predicted;
    if (pred.length < 2 * na) throw new Error('matchGuidedCPU: predicted must hold 2 values per feature of a');
  }
  const grid = new GridIndex(b, Math.max(4, radius));
  const r2 = radius * radius;
  const gx = grid.x;
  const gy = grid.y;
  const items = grid.items;
  const start = grid.start;
  for (let i = 0; i < na; i++) {
    const px = pred[2 * i];
    const py = pred[2 * i + 1];
    const ao = i * D;
    if (Number.isNaN(px) || Number.isNaN(py)) {
      for (let j = 0; j < nb; j++) consider(i, ao, j);
      continue;
    }
    if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
    if (px < -radius - 1 || py < -radius - 1 || px > b.width + radius || py > b.height + radius) continue;
    const cx0 = grid.cx(px - radius);
    const cx1 = grid.cx(px + radius);
    const cy0 = grid.cy(py - radius);
    const cy1 = grid.cy(py + radius);
    for (let cy = cy0; cy <= cy1; cy++) {
      const row = cy * grid.gw;
      for (let cx = cx0; cx <= cx1; cx++) {
        const c = row + cx;
        for (let t = start[c], e = start[c + 1]; t < e; t++) {
          const j = items[t];
          const dx = gx[j] - px;
          const dy = gy[j] - py;
          if (dx * dx + dy * dy > r2) continue;
          consider(i, ao, j);
        }
      }
    }
  }
  return emit(new PairBuffer(), na, bestJ, bestDot, secondDot, colBestI, o);
}
