/**
 * Niantic SPZ Gaussian-splat files (.spz), written as version 3: the newest version that is plain
 * gzip, which `CompressionStream('gzip')` produces in Safari 16.4+ and in workers. Version 4 (the
 * reference library's default) compresses each stream with ZSTD, which no browser compresses
 * natively. The reference loader still reads versions 1–3 ("legacy gzip, read-only"), and so do
 * SuperSplat (which writes v3 or v4) and Spark.
 *
 * Layout, checked 2026-10-21 against github.com/nianticlabs/spz (README "File Format";
 * src/cc/load-spz.cc packGaussians / serializePackedGaussians / deserializePackedGaussians;
 * src/cc/splat-utils.h). The file is ONE gzip stream of:
 *   header, 16 B, little-endian: magic u32 0x5053474e ("NGSP"), version u32 = 3, numPoints u32,
 *     shDegree u8, fractionalBits u8, flags u8 (0x1: antialiased), reserved u8 = 0
 *   positions  N × (x, y, z) 24-bit two's-complement little-endian, round(v · 2^fractionalBits)
 *   alphas     N u8      round(sigmoid(opacity logit) · 255)
 *   colours    N × 3 u8  round(f_dc · 0.15 · 255 + 127.5): the SH DC coefficient (colorScale 0.15,
 *                        below SH_C0, keeps headroom outside [0, 1])
 *   scales     N × 3 u8  round((log-scale + 10) · 16)
 *   rotations  N × u32 LE, "smallest three": the unit quaternion in (x, y, z, w) order, negated so
 *                        its largest |component| is positive; bits 30–31 hold that component's
 *                        index, then the other three in ascending index order, 10 bits each (the
 *                        first in bits 20–29): sign bit 9, magnitude round(511 · |q| / √½)
 *   sh         N × dim × 3 u8, round(c · 128) + 128, coefficient-major with the colour inner
 *                        (degree 1: 9 values, quantised to 5 bits like the reference's sh1Bits)
 * Every u8 is clamped to [0, 255].
 *
 * Coordinates: SPZ stores RUB (+x right, +y up, +z back; three.js / OpenGL), and readers convert to
 * their own frame. The model's Z-up world (x east, y north, z up: "RFU" in SPZ's naming) maps by
 * (x, y, z) → (x, z, −y), a −90° rotation about x. This is the same Y-up mapping that the viewer and
 * glb.ts use. Rotations are composed with it, and degree-1 SH coefficients are rotated with it:
 * (f1, f2, f3) → (−f2, f1, f3).
 *
 * fractionalBits follows the model extent: every |coordinate| must stay below 2^(23 − bits). It is
 * capped at 16 (15 µm steps; a 100 m scene gets 16, a 400 m sky shell 14); the reference library
 * always writes 12 (0.24 mm), and readers take the value from the header. Quantisation steps:
 * positions 2^−bits, log-scales 1/16 (6 % in scale), opacity 1/255, rendered colour ≈ 1.9/255,
 * rotations ≤ 0.2°. At SH degree 0 a Gaussian is 20 B before gzip, against 68 B in the PLY.
 *
 * Cross-checked 2026-10-21 with the reference C++ loader (nianticlabs/spz main, built against a
 * ZSTD stub). Loading our files with `UnpackOptions.to = RFU` gave back the Z-up model within the
 * steps above, at SH degree 0 and 1 (300 random Gaussians each): positions 7.6 µm, log-scales
 * 0.031, alpha 0.002, f_dc 0.013, f_rest 0.035 and rotations 0.17°. That load includes the
 * reference's own RUB → RFU SH rotation (docs/research/flythrough-video.md).
 *
 * `decodeSplatSPZ` reads versions 1–3 (v1: f16 positions; v2: (x, y, z) 8-bit quaternions) back
 * into the Z-up world. It is used for round trips and tests; version 4 (ZSTD) is refused.
 */
import { allocSplatModel, shCoeffCount, type ShDegree, type SplatModel } from './types';

export const SPZ_MAGIC = 0x5053474e;
/** Version written by `encodeSplatSPZ`. */
export const SPZ_VERSION = 3;
const HEADER_BYTES = 16;
const FLAG_ANTIALIASED = 0x1;
const COLOR_SCALE = 0.15;
const SQRT1_2 = Math.SQRT1_2;
/** Most fractional bits written (15 µm steps). */
export const SPZ_MAX_FRACTIONAL_BITS = 16;
/** Bits kept of the degree-1 SH coefficients (the reference's default sh1Bits). */
const SH1_BITS = 5;

export interface SpzEncodeOptions {
  /** Trained with the Mip-Splatting opacity compensation (default true): sets flag 0x1. */
  antialiased?: boolean;
  /** Override the fixed-point fraction (default: from the model extent, ≤ 16). */
  fractionalBits?: number;
}

export interface SpzDecoded {
  model: SplatModel;
  antialiased: boolean;
  version: number;
  fractionalBits: number;
}

const toU8 = (x: number): number => Math.min(255, Math.max(0, Math.round(x)));

/** Fixed-point fraction for the model extent: |coordinate| · 2^bits < 2^23, at most 16 bits. */
export function spzFractionalBits(means: ArrayLike<number>, count = Math.floor(means.length / 3)): number {
  let m = 0;
  for (let i = 0; i < 3 * count; i++) {
    const a = Math.abs(means[i]);
    if (a > m && Number.isFinite(a)) m = a;
  }
  if (m === 0) return SPZ_MAX_FRACTIONAL_BITS;
  const bits = Math.floor(Math.log2((2 ** 23 - 1) / m));
  return Math.max(0, Math.min(SPZ_MAX_FRACTIONAL_BITS, bits));
}

/** The quantisation the reference applies to SH coefficients: 8 bits, then buckets of `bucket`. */
function quantizeSH(x: number, bucket: number): number {
  let q = Math.round(x * 128) + 128;
  q = Math.trunc((q + bucket / 2) / bucket) * bucket;
  return Math.min(255, Math.max(0, q));
}

/**
 * The uncompressed SPZ v3 payload of a model in the Z-up world (header + attribute streams):
 * `encodeSplatSPZ` gzips it.
 */
export function packSplatSPZ(m: SplatModel, o: SpzEncodeOptions = {}): Uint8Array {
  const n = m.count;
  if (n <= 0) throw new Error('SPZ: no Gaussians');
  const K = shCoeffCount(m.shDegree);
  const dim = K - 1;
  const bits = o.fractionalBits ?? spzFractionalBits(m.means, n);
  if (!(bits >= 0 && bits <= 23)) throw new Error(`SPZ: fractional bits ${bits} out of range`);
  const size = HEADER_BYTES + n * (9 + 1 + 3 + 3 + 4 + 3 * dim);
  const out = new Uint8Array(size);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, SPZ_MAGIC, true);
  dv.setUint32(4, SPZ_VERSION, true);
  dv.setUint32(8, n, true);
  out[12] = m.shDegree;
  out[13] = bits;
  out[14] = (o.antialiased ?? true) ? FLAG_ANTIALIASED : 0;
  out[15] = 0;
  const posOff = HEADER_BYTES;
  const alphaOff = posOff + 9 * n;
  const colOff = alphaOff + n;
  const scaleOff = colOff + 3 * n;
  const rotOff = scaleOff + 3 * n;
  const shOff = rotOff + 4 * n;
  const scale = 2 ** bits;
  const lim = 2 ** 23 - 1;
  const q = new Float64Array(4);
  const bucket = 1 << (8 - SH1_BITS);
  for (let i = 0; i < n; i++) {
    // Z-up world → RUB: (x, y, z) → (x, z, −y)
    const x = m.means[3 * i];
    const y = m.means[3 * i + 1];
    const z = m.means[3 * i + 2];
    const p = [x, z, -y];
    for (let k = 0; k < 3; k++) {
      const v = Number.isFinite(p[k]) ? Math.max(-lim, Math.min(lim, Math.round(p[k] * scale))) : 0;
      const o3 = posOff + 9 * i + 3 * k;
      out[o3] = v & 0xff;
      out[o3 + 1] = (v >> 8) & 0xff;
      out[o3 + 2] = (v >> 16) & 0xff;
    }
    out[alphaOff + i] = toU8(255 / (1 + Math.exp(-m.opacities[i])));
    const sh = 3 * K * i;
    for (let c = 0; c < 3; c++) out[colOff + 3 * i + c] = toU8(m.sh[sh + c] * (COLOR_SCALE * 255) + 0.5 * 255);
    // log-scales are per local axis: unchanged by the world rotation
    for (let c = 0; c < 3; c++) out[scaleOff + 3 * i + c] = toU8((m.logScales[3 * i + c] + 10) * 16);
    // q' = q_M ⊗ q with q_M = (cos −45°, sin −45°, 0, 0) [w, x, y, z]: the −90° rotation about x
    const w0 = m.quats[4 * i];
    const x0 = m.quats[4 * i + 1];
    const y0 = m.quats[4 * i + 2];
    const z0 = m.quats[4 * i + 3];
    // (a, −a, 0, 0) ⊗ (w, x, y, z) = a (w + x, x − w, y + z, z − y)
    q[3] = w0 + x0; // w  (stored in (x, y, z, w) order)
    q[0] = x0 - w0;
    q[1] = y0 + z0;
    q[2] = z0 - y0;
    out.set(packQuatSmallestThree(q), rotOff + 4 * i);
    if (dim > 0) {
      // degree 1 in RUB: (f1, f2, f3) → (−f2, f1, f3) per colour channel
      for (let c = 0; c < 3; c++) {
        const f1 = m.sh[sh + 3 + c];
        const f2 = m.sh[sh + 6 + c];
        const f3 = m.sh[sh + 9 + c];
        const b = shOff + 9 * i;
        out[b + c] = quantizeSH(-f2, bucket);
        out[b + 3 + c] = quantizeSH(f1, bucket);
        out[b + 6 + c] = quantizeSH(f3, bucket);
      }
    }
  }
  return out;
}

const qBytes = new Uint8Array(4);

/** Smallest-three packing of a quaternion in (x, y, z, w) order (normalised here). */
function packQuatSmallestThree(qxyzw: Float64Array): Uint8Array {
  const len = Math.hypot(qxyzw[0], qxyzw[1], qxyzw[2], qxyzw[3]);
  const q = len > 0 ? [qxyzw[0] / len, qxyzw[1] / len, qxyzw[2] / len, qxyzw[3] / len] : [0, 0, 0, 1];
  let iLargest = 0;
  for (let i = 1; i < 4; i++) if (Math.abs(q[i]) > Math.abs(q[iLargest])) iLargest = i;
  const negate = q[iLargest] < 0 ? 1 : 0;
  let comp = iLargest;
  for (let i = 0; i < 4; i++) {
    if (i === iLargest) continue;
    const negbit = (q[i] < 0 ? 1 : 0) ^ negate;
    const mag = Math.min(511, Math.floor(511 * (Math.abs(q[i]) / SQRT1_2) + 0.5));
    comp = ((comp << 10) | (negbit << 9) | mag) >>> 0;
  }
  qBytes[0] = comp & 0xff;
  qBytes[1] = (comp >>> 8) & 0xff;
  qBytes[2] = (comp >>> 16) & 0xff;
  qBytes[3] = (comp >>> 24) & 0xff;
  return qBytes;
}

/** Unpacks a smallest-three quaternion into (x, y, z, w). */
function unpackQuatSmallestThree(b: Uint8Array, off: number, out: Float64Array): void {
  let comp = (b[off] | (b[off + 1] << 8) | (b[off + 2] << 16) | (b[off + 3] << 24)) >>> 0;
  const iLargest = comp >>> 30;
  let sum = 0;
  for (let i = 3; i >= 0; i--) {
    if (i === iLargest) continue;
    const mag = comp & 511;
    const neg = (comp >>> 9) & 1;
    comp >>>= 10;
    const v = (SQRT1_2 * mag) / 511;
    out[i] = neg ? -v : v;
    sum += v * v;
  }
  out[iLargest] = Math.sqrt(Math.max(0, 1 - sum));
}

/** f16 bits → number (version 1 positions). */
function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

/** Reads an uncompressed SPZ payload of version 1–3 into a Z-up-world model. */
export function unpackSplatSPZ(data: Uint8Array): SpzDecoded {
  if (data.length < HEADER_BYTES) throw new Error('SPZ: truncated header');
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  if (dv.getUint32(0, true) !== SPZ_MAGIC) throw new Error('SPZ: bad magic');
  const version = dv.getUint32(4, true);
  if (version < 1 || version > 3) throw new Error(`SPZ: version ${version} is not supported (1–3)`);
  const n = dv.getUint32(8, true);
  const fileDeg = data[12];
  const bits = data[13];
  const flags = data[14];
  if (fileDeg > 4) throw new Error(`SPZ: SH degree ${fileDeg}`);
  const fileDim = [0, 3, 8, 15, 24][fileDeg];
  const posBytes = version === 1 ? 6 : 9;
  const rotBytes = version >= 3 ? 4 : 3;
  const need = HEADER_BYTES + n * (posBytes + 1 + 3 + 3 + rotBytes + 3 * fileDim);
  if (n === 0 || data.length < need) throw new Error('SPZ: truncated');
  // bands above 1 are dropped (this app renders degree 0 / 1)
  const deg: ShDegree = fileDeg >= 1 ? 1 : 0;
  const K = shCoeffCount(deg);
  const m = allocSplatModel(n, deg);
  const posOff = HEADER_BYTES;
  const alphaOff = posOff + posBytes * n;
  const colOff = alphaOff + n;
  const scaleOff = colOff + 3 * n;
  const rotOff = scaleOff + 3 * n;
  const shOff = rotOff + rotBytes * n;
  const inv = 1 / 2 ** bits;
  const q = new Float64Array(4);
  const p = new Float64Array(3);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      if (version === 1) p[k] = halfToFloat(dv.getUint16(posOff + 6 * i + 2 * k, true));
      else {
        const o3 = posOff + 9 * i + 3 * k;
        let v = data[o3] | (data[o3 + 1] << 8) | (data[o3 + 2] << 16);
        if (v & 0x800000) v |= ~0xffffff;
        p[k] = v * inv;
      }
    }
    // RUB → Z-up world: (x', y', z') → (x', −z', y')
    m.means[3 * i] = p[0];
    m.means[3 * i + 1] = -p[2];
    m.means[3 * i + 2] = p[1];
    const a = Math.min(254.5, Math.max(0.5, data[alphaOff + i])) / 255;
    m.opacities[i] = Math.log(a / (1 - a));
    const sh = 3 * K * i;
    for (let c = 0; c < 3; c++) m.sh[sh + c] = (data[colOff + 3 * i + c] / 255 - 0.5) / COLOR_SCALE;
    for (let c = 0; c < 3; c++) m.logScales[3 * i + c] = data[scaleOff + 3 * i + c] / 16 - 10;
    if (rotBytes === 4) unpackQuatSmallestThree(data, rotOff + 4 * i, q);
    else {
      for (let c = 0; c < 3; c++) q[c] = data[rotOff + 3 * i + c] / 127.5 - 1;
      q[3] = Math.sqrt(Math.max(0, 1 - q[0] * q[0] - q[1] * q[1] - q[2] * q[2]));
    }
    // q = q_M⁻¹ ⊗ q' with q_M⁻¹ = (cos 45°, sin 45°, 0, 0): (b, b, 0, 0) ⊗ (w, x, y, z) = b (w − x, x + w, y − z, z + y)
    const s = SQRT1_2;
    m.quats[4 * i] = s * (q[3] - q[0]);
    m.quats[4 * i + 1] = s * (q[0] + q[3]);
    m.quats[4 * i + 2] = s * (q[1] - q[2]);
    m.quats[4 * i + 3] = s * (q[2] + q[1]);
    if (deg === 1) {
      const b = shOff + 3 * fileDim * i;
      for (let c = 0; c < 3; c++) {
        const f1r = (data[b + c] - 128) / 128;
        const f2r = (data[b + 3 + c] - 128) / 128;
        const f3r = (data[b + 6 + c] - 128) / 128;
        // inverse of (f1, f2, f3) → (−f2, f1, f3)
        m.sh[sh + 3 + c] = f2r;
        m.sh[sh + 6 + c] = -f1r;
        m.sh[sh + 9 + c] = f3r;
      }
    }
  }
  return { model: m, antialiased: (flags & FLAG_ANTIALIASED) !== 0, version, fractionalBits: bits };
}

async function pipeBytes(data: Uint8Array, t: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const stream = new Blob([data as Uint8Array<ArrayBuffer>]).stream().pipeThrough(t);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** SPZ v3 file bytes of a model in the Z-up world (gzip via CompressionStream). */
export async function encodeSplatSPZ(m: SplatModel, o: SpzEncodeOptions = {}): Promise<Uint8Array> {
  if (typeof CompressionStream === 'undefined') throw new Error('The SPZ export needs CompressionStream (Safari 16.4 or later)');
  return pipeBytes(packSplatSPZ(m, o), new CompressionStream('gzip'));
}

/** Reads an SPZ file of version 1–3 (gzip) into a Z-up-world model. */
export async function decodeSplatSPZ(data: Uint8Array): Promise<SpzDecoded> {
  if (data.length >= 4 && new DataView(data.buffer, data.byteOffset, 4).getUint32(0, true) === SPZ_MAGIC)
    throw new Error('SPZ: version 4 (ZSTD streams) is not supported');
  if (data.length < 2 || data[0] !== 0x1f || data[1] !== 0x8b) throw new Error('SPZ: not a gzip stream');
  return unpackSplatSPZ(await pipeBytes(data, new DecompressionStream('gzip')));
}
