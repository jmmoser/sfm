/**
 * Host-side (CPU) logic of the WebGPU matcher, free of WebGPU calls (unit-tested in
 * Node): descriptor norms + content hash, guide records (same constraint semantics as
 * matchGuidedCPU, incl. the sign-invariant H orientation), spatial (Morton) ordering
 * and tile boxes for tile culling, and the final exact decision (f64 similarities,
 * ratio test, max distance, mutual check — identical to matchCPU's `emit`).
 */

import type { FeatureSet, MatchPairs } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';
import { FeatureInputError } from './gpu-common';
import type { MatchGuide, MatchOptions } from './match';
import { MATCH_INVALID, MATCH_TILE } from './shaders/match-wgsl';

const D = DESCRIPTOR_SIZE;

/**
 * Throws FeatureInputError (a plain Error, not a GPU failure) when a FeatureSet cannot be
 * matched: non-integer / negative count, descriptors (or keypoints, when `positions`)
 * shorter than `count` requires. Without this check a malformed set surfaced as a
 * RangeError inside the GPU path, which was reported as a GPU failure and switched the
 * whole feature pipeline to the CPU permanently.
 */
export function validateFeatureSet(fs: FeatureSet, name: string, positions: boolean): void {
  const n = fs.count;
  if (!(Number.isInteger(n) && n >= 0)) throw new FeatureInputError(`match: FeatureSet ${name} has an invalid count ${n}`);
  if (!(fs.descriptors instanceof Uint8Array) || fs.descriptors.length < n * D) {
    throw new FeatureInputError(`match: FeatureSet ${name} needs ${n * D} descriptor bytes (Uint8Array), has ${fs.descriptors?.length}`);
  }
  if (positions && !(fs.keypoints && fs.keypoints.length >= 4 * n)) {
    throw new FeatureInputError(`match: FeatureSet ${name} needs ${4 * n} keypoint values, has ${fs.keypoints?.length}`);
  }
}

export interface DescriptorStats {
  /** 1 / L2 norm of each quantised descriptor (0 for all-zero), f64 exactly as match.ts. */
  invNorm: Float64Array;
  /** 32-bit content hash of count, descriptors (and keypoint positions when requested). */
  hash: number;
}

/** Norms (as match.ts computeInvNorms) and a content hash of a FeatureSet's descriptors. */
export function descriptorStats(fs: FeatureSet, withPositions = false): DescriptorStats {
  const n = fs.count;
  const src = fs.descriptors;
  const invNorm = new Float64Array(n);
  let h = (0x811c9dc5 ^ n) >>> 0;
  const aligned = src.byteOffset % 4 === 0;
  const words = aligned ? new Uint32Array(src.buffer, src.byteOffset, (n * D) >> 2) : null;
  for (let i = 0; i < n; i++) {
    const o = i * D;
    let ss = 0;
    for (let k = 0; k < D; k++) ss += src[o + k] * src[o + k];
    invNorm[i] = ss > 0 ? 1 / Math.sqrt(ss) : 0;
    if (words) {
      const wo = o >> 2;
      for (let k = 0; k < 32; k++) h = Math.imul(h ^ words[wo + k], 0x01000193);
    } else {
      for (let k = 0; k < D; k++) h = Math.imul(h ^ src[o + k], 0x01000193);
    }
  }
  if (withPositions) {
    const kp = fs.keypoints;
    const kw = new Uint32Array(kp.buffer, kp.byteOffset, 4 * n);
    for (let i = 0; i < 4 * n; i += 4) {
      h = Math.imul(h ^ kw[i], 0x01000193);
      h = Math.imul(h ^ kw[i + 1], 0x01000193);
    }
  }
  return { invNorm, hash: h >>> 0 };
}

/** Descriptors as packed u32 words (4 bytes per word, little endian), copying only if unaligned. */
export function packDescriptorWords(fs: FeatureSet): Uint32Array {
  const src = fs.descriptors;
  const n = fs.count * D;
  if (src.byteOffset % 4 === 0) return new Uint32Array(src.buffer, src.byteOffset, n >> 2);
  const copy = new Uint8Array(n);
  copy.set(src.subarray(0, n));
  return new Uint32Array(copy.buffer);
}

/** Interleaves the low 10 bits of x and y (Morton / Z-order). */
export function morton10(x: number, y: number): number {
  let a = x & 0x3ff;
  let b = y & 0x3ff;
  a = (a | (a << 8)) & 0x00ff00ff;
  a = (a | (a << 4)) & 0x0f0f0f0f;
  a = (a | (a << 2)) & 0x33333333;
  a = (a | (a << 1)) & 0x55555555;
  b = (b | (b << 8)) & 0x00ff00ff;
  b = (b | (b << 4)) & 0x0f0f0f0f;
  b = (b | (b << 2)) & 0x33333333;
  b = (b | (b << 1)) & 0x55555555;
  return (a | (b << 1)) >>> 0;
}

/**
 * Permutation sorting points by (class, Morton code of the position quantised to a
 * 1024 x 1024 grid over [x0, x0 + extent] x [y0, y0 + extent]); stable by index.
 * Class 0 first; points with non-finite coordinates should get a higher class.
 */
export function spatialOrder(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  n: number,
  x0: number,
  y0: number,
  extent: number,
  cls?: Uint8Array,
): Uint32Array {
  const keys = new Float64Array(n);
  const s = 1024 / Math.max(1e-9, extent);
  for (let i = 0; i < n; i++) {
    let m = 0;
    const c = cls ? cls[i] : 0;
    if (c === 0) {
      const qx = Math.min(1023, Math.max(0, Math.floor((xs[i] - x0) * s)));
      const qy = Math.min(1023, Math.max(0, Math.floor((ys[i] - y0) * s)));
      m = morton10(qx, qy);
    }
    keys[i] = (c * 2 ** 20 + m) * n + i;
  }
  keys.sort();
  const out = new Uint32Array(n);
  for (let i = 0; i < n; i++) out[i] = keys[i] % n;
  return out;
}

/** Everything the guided kernels need, built on the CPU. */
export interface GuidePrep {
  mode: 'pos' | 'epi';
  /** vec4 records: [0, na) a-records, [na, na + nb) b positions, then b tile boxes. */
  aux: Float32Array;
  aOff: number;
  bOff: number;
  boxOff: number;
  /** Sorted position -> natural index. */
  aPerm: Uint32Array;
  bPerm: Uint32Array;
  r2: number;
  radius: number;
  tol: number;
}

/**
 * Builds the guide records with the semantics of matchGuidedCPU:
 *  - positions: NaN in a prediction = unconstrained (flag 1); ±Infinity or a
 *    prediction farther than radius + 1 px outside the frame of b = never (flag 2);
 *  - H: predictions as matchGuidedCPU (H oriented by majority vote so that most a
 *    keypoints map with w > 0; others never match);
 *  - F: unit-normalised epipolar line in b in coordinates centred at b's image centre;
 *    a degenerate line (|(l0, l1)| <= 1e-15) never matches.
 */
export function prepareGuide(a: FeatureSet, b: FeatureSet, guide: MatchGuide): GuidePrep {
  const na = a.count;
  const nb = b.count;
  const nbTiles = Math.ceil(nb / MATCH_TILE);
  const aOff = 0;
  const bOff = na;
  const boxOff = na + nb;
  const aux = new Float32Array(4 * (na + nb + nbTiles));
  const cxb = (b.width - 1) / 2;
  const cyb = (b.height - 1) / 2;
  for (let j = 0; j < nb; j++) {
    const x = b.keypoints[4 * j];
    const y = b.keypoints[4 * j + 1];
    const o = 4 * (bOff + j);
    aux[o] = x;
    aux[o + 1] = y;
    aux[o + 2] = x - cxb;
    aux[o + 3] = y - cyb;
  }
  const extentB = Math.max(b.width, b.height, 1);
  const bPerm = spatialOrder(keypointCoord(b, 0), keypointCoord(b, 1), nb, -0.5, -0.5, extentB);
  let mode: 'pos' | 'epi';
  let radius = 0;
  let tol = 0;
  const cls = new Uint8Array(na);
  const ax = new Float64Array(na);
  const ay = new Float64Array(na);
  if (guide.kind === 'F') {
    mode = 'epi';
    tol = guide.maxErrorPx;
    const F = guide.F;
    for (let i = 0; i < na; i++) {
      const xa = a.keypoints[4 * i];
      const ya = a.keypoints[4 * i + 1];
      ax[i] = xa;
      ay[i] = ya;
      const l0 = F[0] * xa + F[1] * ya + F[2];
      const l1 = F[3] * xa + F[4] * ya + F[5];
      const l2 = F[6] * xa + F[7] * ya + F[8];
      const nrm = Math.sqrt(l0 * l0 + l1 * l1);
      const o = 4 * (aOff + i);
      if (!(nrm > 1e-15)) {
        aux[o + 3] = 2;
        cls[i] = 2;
        continue;
      }
      aux[o] = l0 / nrm;
      aux[o + 1] = l1 / nrm;
      aux[o + 2] = (l0 * cxb + l1 * cyb + l2) / nrm;
      aux[o + 3] = 0;
    }
  } else {
    mode = 'pos';
    let pred: Float32Array;
    if (guide.kind === 'H') {
      radius = guide.maxErrorPx;
      pred = predictWithHomography(a, guide.H);
    } else {
      radius = guide.radiusPx;
      pred = guide.predicted;
      if (pred.length < 2 * na) throw new Error('matchGuided: predicted must hold 2 values per feature of a');
    }
    for (let i = 0; i < na; i++) {
      const px = pred[2 * i];
      const py = pred[2 * i + 1];
      const o = 4 * (aOff + i);
      if (Number.isNaN(px) || Number.isNaN(py)) {
        aux[o + 2] = 1;
        cls[i] = 1;
      } else if (
        !Number.isFinite(px) ||
        !Number.isFinite(py) ||
        px < -radius - 1 ||
        py < -radius - 1 ||
        px > b.width + radius ||
        py > b.height + radius
      ) {
        aux[o + 2] = 2;
        cls[i] = 2;
      } else {
        aux[o] = px;
        aux[o + 1] = py;
        aux[o + 2] = 0;
        ax[i] = px;
        ay[i] = py;
      }
    }
  }
  const extentA = Math.max(guide.kind === 'F' ? Math.max(a.width, a.height) : extentB, 1);
  const aPerm = spatialOrder(ax, ay, na, -0.5, -0.5, extentA, cls);
  // Tile boxes of b (sorted order), in the coordinates the predicate uses, +1 px margin.
  const cx = mode === 'pos' ? 0 : cxb;
  const cy = mode === 'pos' ? 0 : cyb;
  for (let t = 0; t < nbTiles; t++) {
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    const e = Math.min(nb, (t + 1) * MATCH_TILE);
    for (let s = t * MATCH_TILE; s < e; s++) {
      const j = bPerm[s];
      const x = b.keypoints[4 * j] - cx;
      const y = b.keypoints[4 * j + 1] - cy;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
    const o = 4 * (boxOff + t);
    aux[o] = x0 - 1;
    aux[o + 1] = y0 - 1;
    aux[o + 2] = x1 + 1;
    aux[o + 3] = y1 + 1;
  }
  return { mode, aux, aOff, bOff, boxOff, aPerm, bPerm, r2: radius * radius, radius, tol };
}

function keypointCoord(fs: FeatureSet, c: 0 | 1): Float32Array {
  const out = new Float32Array(fs.count);
  for (let i = 0; i < fs.count; i++) out[i] = fs.keypoints[4 * i + c];
  return out;
}

/** Predicted positions in b under H (same orientation vote and w handling as matchGuidedCPU). */
export function predictWithHomography(a: FeatureSet, H: ArrayLike<number>): Float32Array {
  const na = a.count;
  const pred = new Float32Array(2 * na);
  let vote = 0;
  for (let i = 0; i < na; i++) {
    const w = H[6] * a.keypoints[4 * i] + H[7] * a.keypoints[4 * i + 1] + H[8];
    if (w > 0) vote++;
    else if (w < 0) vote--;
  }
  const sgn = vote < 0 ? -1 : 1;
  for (let i = 0; i < na; i++) {
    const x = a.keypoints[4 * i];
    const y = a.keypoints[4 * i + 1];
    const w = H[6] * x + H[7] * y + H[8];
    if (sgn * w > 0) {
      pred[2 * i] = (H[0] * x + H[1] * y + H[2]) / w;
      pred[2 * i + 1] = (H[3] * x + H[4] * y + H[5]) / w;
    } else {
      pred[2 * i] = Infinity;
      pred[2 * i + 1] = Infinity;
    }
  }
  return pred;
}

/** Column-chunking of a (rows x cols) problem: number of chunks and column tiles per chunk. */
export function planChunks(nRows: number, nCols: number, targetWorkgroups: number): { nChunks: number; tilesPerChunk: number; nRowTiles: number; nColTiles: number } {
  const nRowTiles = Math.ceil(nRows / MATCH_TILE);
  const nColTiles = Math.ceil(nCols / MATCH_TILE);
  let nChunks = Math.min(nColTiles, Math.max(1, Math.ceil(targetWorkgroups / Math.max(1, nRowTiles))));
  const tilesPerChunk = Math.ceil(nColTiles / nChunks);
  nChunks = Math.ceil(nColTiles / tilesPerChunk);
  return { nChunks, tilesPerChunk, nRowTiles, nColTiles };
}

/**
 * Best / second-best per row from the GPU's per-chunk top-2 lists, with EXACT f64
 * similarities s = idot * invRow[i] * invCol[j] (as match.ts: (idot * invA) * invB),
 * ties -> smaller column index. `rowsAreA` selects the operand order of the product.
 */
export function reduceTop2(
  out: Uint32Array,
  base: number,
  nRows: number,
  nChunks: number,
  invRow: Float64Array,
  invCol: Float64Array,
  rowsAreA: boolean,
  bestJ: Int32Array,
  bestDot: Float64Array,
  secondDot: Float64Array,
): void {
  for (let i = 0; i < nRows; i++) {
    let bj = -1;
    let bs = -Infinity;
    let ss = -Infinity;
    const ir = invRow[i];
    for (let c = 0; c < nChunks; c++) {
      const o = base + 4 * (i * nChunks + c);
      for (let k = 0; k < 4; k += 2) {
        const j = out[o + k];
        if (j === MATCH_INVALID) continue;
        const d = out[o + k + 1];
        const s = rowsAreA ? d * ir * invCol[j] : d * invCol[j] * ir;
        if (s > bs || (s === bs && j < bj)) {
          ss = bs;
          bs = s;
          bj = j;
        } else if (s > ss) ss = s;
      }
    }
    bestJ[i] = bj;
    bestDot[i] = bs;
    secondDot[i] = ss;
  }
}

/** Final filter, identical to match.ts `emit`: max distance, ratio test, mutual check. */
export function emitMatches(
  na: number,
  bestJ: Int32Array,
  bestDot: Float64Array,
  secondDot: Float64Array,
  colBestI: Int32Array | null,
  opts?: MatchOptions,
): MatchPairs {
  const ratio = opts?.ratio ?? 0.8;
  const mutual = opts?.mutual ?? true;
  const maxDistance = opts?.maxDistance ?? 0.7;
  const tmp = new Uint32Array(2 * na);
  let n = 0;
  for (let i = 0; i < na; i++) {
    const j = bestJ[i];
    if (j < 0) continue;
    const d1 = Math.sqrt(Math.max(0, 2 - 2 * bestDot[i]));
    if (d1 > maxDistance) continue;
    const sd = secondDot[i];
    if (sd > -Infinity) {
      const d2 = Math.sqrt(Math.max(0, 2 - 2 * sd));
      if (!(d1 < ratio * d2)) continue;
    }
    if (mutual && (!colBestI || colBestI[j] !== i)) continue;
    tmp[n++] = i;
    tmp[n++] = j;
  }
  return tmp.slice(0, n);
}
