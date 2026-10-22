/**
 * SPZ v3 (src/splat/spz.ts): the documented header / stream layout byte by byte, the RUB axis
 * mapping (positions, rotations, degree-1 SH: the rendered colour is unchanged), round trips at
 * SH degree 0 and 1 within the format's quantisation, fractional bits from the extent, legacy
 * v2 reading and the size against the INRIA PLY.
 */
import { describe, expect, it } from 'vitest';
import { quatToMat3 } from '../math/rotation';
import { encodeSplatPLY } from './ply';
import { decodeSplatSPZ, encodeSplatSPZ, packSplatSPZ, spzFractionalBits, unpackSplatSPZ, SPZ_MAGIC } from './spz';
import { randomScene } from './testutil';
import { allocSplatModel, logit, SH_C0, SH_C1, sigmoid, type SplatModel } from './types';

async function gunzip(b: Uint8Array): Promise<Uint8Array> {
  const s = new Blob([b as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

/** World Z-up → RUB, the mapping SPZ files are expected in. */
const toRUB = (v: ArrayLike<number>): number[] => [v[0], v[2], -v[1]];

/** Rotation matrix of quaternion i of a model (row-major). */
const rot = (m: SplatModel, i: number): Float64Array => {
  const q = m.quats.subarray(4 * i, 4 * i + 4);
  const l = Math.hypot(q[0], q[1], q[2], q[3]);
  return quatToMat3([q[0] / l, q[1] / l, q[2] / l, q[3] / l]);
};

/** Angle between two rotations (rad). */
function angleBetween(A: Float64Array, B: Float64Array): number {
  let tr = 0;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) tr += A[3 * r + c] * B[3 * r + c];
  return Math.acos(Math.min(1, Math.max(-1, (tr - 1) / 2)));
}

describe('SPZ v3 writer', () => {
  it('writes the documented 16-byte header and stream layout (one gzip stream)', async () => {
    const m = allocSplatModel(2, 0);
    m.means.set([1.5, -2.25, 0.125, -0.5, 0.75, 3]);
    m.logScales.set([-3, -2, -1, 0, 0.5, -10]);
    m.opacities.set([0, logit(0.9)]);
    m.sh.set([0, 1, -1, 2, -2, 0.5]);
    const file = await encodeSplatSPZ(m, { antialiased: true });
    expect([file[0], file[1]]).toEqual([0x1f, 0x8b]); // gzip, as the reference's legacy loader detects it
    const raw = await gunzip(file);
    expect(raw.length).toBe(16 + 2 * 20); // SH degree 0: 9 + 1 + 3 + 3 + 4 bytes per Gaussian
    const dv = new DataView(raw.buffer);
    expect(new TextDecoder().decode(raw.subarray(0, 4))).toBe('NGSP');
    expect(dv.getUint32(0, true)).toBe(SPZ_MAGIC);
    expect(dv.getUint32(4, true)).toBe(3);
    expect(dv.getUint32(8, true)).toBe(2);
    expect(raw[12]).toBe(0); // SH degree
    const bits = raw[13];
    expect(bits).toBe(spzFractionalBits(m.means));
    expect(raw[14]).toBe(1); // antialiased
    expect(raw[15]).toBe(0);
    // positions: 24-bit LE fixed point of the RUB coordinates, point-major
    const fixed = (o: number): number => {
      let v = raw[o] | (raw[o + 1] << 8) | (raw[o + 2] << 16);
      if (v & 0x800000) v -= 1 << 24;
      return v / 2 ** bits;
    };
    for (let i = 0; i < 2; i++) {
      const exp = toRUB(m.means.subarray(3 * i, 3 * i + 3));
      for (let k = 0; k < 3; k++) expect(fixed(16 + 9 * i + 3 * k)).toBeCloseTo(exp[k], 10);
    }
    // alphas, then colours, then scales
    const a0 = 16 + 18;
    expect([raw[a0], raw[a0 + 1]]).toEqual([Math.round(0.5 * 255), Math.round(0.9 * 255)]);
    const c0 = a0 + 2;
    expect(Array.from(raw.subarray(c0, c0 + 6))).toEqual([128, 166, 89, 204, 51, 147]); // round(f · 38.25 + 127.5)
    const s0 = c0 + 6;
    expect(Array.from(raw.subarray(s0, s0 + 6))).toEqual([112, 128, 144, 160, 168, 0]); // (log σ + 10) · 16
    // rotations: identity in the Z-up world is the −90° rotation about x in RUB: (x, y, z, w) =
    // (−√½, 0, 0, √½); largest |component| (the first of the tie) is x → index 0, made positive:
    // y, z (+0 → sign bit set by the negation) and w (−√½ → sign set, magnitude 511)
    const r0 = s0 + 6;
    const comp = new DataView(raw.buffer).getUint32(r0, true);
    expect(comp >>> 30).toBe(0);
    expect((comp >>> 20) & 1023).toBe(512); // y: sign 1, magnitude 0
    expect((comp >>> 10) & 1023).toBe(512); // z
    expect(comp & 1023).toBe(512 | 511); // w
    expect(raw.length).toBe(r0 + 8);
  });

  it('maps the Z-up world to RUB: positions, rotations and degree-1 SH (same rendered colour)', () => {
    const m = randomScene({ count: 40, seed: 11, shDegree: 1, extent: 2 });
    const d = unpackSplatSPZ(packSplatSPZ(m));
    const raw = packSplatSPZ(m);
    const bits = raw[13];
    // the stored (RUB) position of Gaussian 7 is (x, z, −y)
    let v = raw[16 + 63] | (raw[16 + 64] << 8) | (raw[16 + 65] << 16);
    if (v & 0x800000) v -= 1 << 24;
    expect(v / 2 ** bits).toBeCloseTo(m.means[21], 4);
    v = raw[16 + 66] | (raw[16 + 67] << 8) | (raw[16 + 68] << 16);
    if (v & 0x800000) v -= 1 << 24;
    expect(v / 2 ** bits).toBeCloseTo(m.means[23], 4); // RUB y = world z
    v = raw[16 + 69] | (raw[16 + 70] << 8) | (raw[16 + 71] << 16);
    if (v & 0x800000) v -= 1 << 24;
    expect(v / 2 ** bits).toBeCloseTo(-m.means[22], 4); // RUB z = −world y
    // degree-1 colour for a viewing direction is the same in both frames (within 5-bit SH steps)
    const shColour = (sh: ArrayLike<number>, i: number, c: number, dir: number[]): number => {
      const b = 12 * i;
      return SH_C0 * sh[b + c] + SH_C1 * (-dir[1] * sh[b + 3 + c] + dir[2] * sh[b + 6 + c] - dir[0] * sh[b + 9 + c]) + 0.5;
    };
    for (let i = 0; i < m.count; i++) {
      const dir = [0.3, -0.5, 0.81];
      const l = Math.hypot(dir[0], dir[1], dir[2]);
      const u = dir.map((x) => x / l);
      for (let c = 0; c < 3; c++) {
        // the decoded model is back in the Z-up world: compare in the same frame
        expect(Math.abs(shColour(d.model.sh, i, c, u) - shColour(m.sh, i, c, u))).toBeLessThan(0.025);
      }
      // rotations round-trip through the RUB frame within the 10-bit precision
      expect(angleBetween(rot(m, i), rot(d.model, i))).toBeLessThan((0.25 * Math.PI) / 180);
    }
  });

  for (const deg of [0, 1] as const) {
    it(`round-trips SH degree ${deg} within the format's quantisation`, async () => {
      const m = randomScene({ count: 500, seed: 3 + deg, shDegree: deg, extent: 1.5, logScale: [-6, -1] });
      const d = await decodeSplatSPZ(await encodeSplatSPZ(m, { antialiased: false }));
      expect(d.version).toBe(3);
      expect(d.antialiased).toBe(false);
      expect(d.model.count).toBe(500);
      expect(d.model.shDegree).toBe(deg);
      const step = 2 ** -d.fractionalBits;
      let maxPos = 0;
      let maxAng = 0;
      for (let i = 0; i < m.count; i++) {
        for (let k = 0; k < 3; k++) maxPos = Math.max(maxPos, Math.abs(d.model.means[3 * i + k] - m.means[3 * i + k]));
        for (let k = 0; k < 3; k++) expect(Math.abs(d.model.logScales[3 * i + k] - m.logScales[3 * i + k])).toBeLessThanOrEqual(1 / 32 + 1e-6);
        expect(Math.abs(sigmoid(d.model.opacities[i]) - sigmoid(m.opacities[i]))).toBeLessThanOrEqual(0.5 / 255 + 1e-6);
        maxAng = Math.max(maxAng, angleBetween(rot(m, i), rot(d.model, i)));
        const K = deg === 0 ? 1 : 4;
        for (let c = 0; c < 3; c++) {
          const f = m.sh[3 * K * i + c];
          // f_dc outside ±(127.5 / 38.25) saturates; randomScene stays within ±1.2
          expect(Math.abs(d.model.sh[3 * K * i + c] - f)).toBeLessThanOrEqual(0.5 / 38.25 + 1e-6);
        }
        for (let k = 3; k < 3 * K; k++) expect(Math.abs(d.model.sh[3 * K * i + k] - m.sh[3 * K * i + k])).toBeLessThanOrEqual(4.5 / 128);
      }
      expect(maxPos).toBeLessThanOrEqual(0.5 * step + 1e-9);
      console.log(`[spz] degree ${deg}: ${d.fractionalBits} fractional bits, max position error ${maxPos.toExponential(2)}, max rotation error ${((maxAng * 180) / Math.PI).toFixed(3)}°`);
      expect(maxAng).toBeLessThan((0.25 * Math.PI) / 180);
    });
  }

  it('chooses the fractional bits from the extent (≤ 16, every coordinate within 24 bits)', () => {
    expect(spzFractionalBits([0.1, -0.2, 0.3])).toBe(16);
    expect(spzFractionalBits([100, 0, 0])).toBe(16);
    expect(spzFractionalBits([0, -400, 0])).toBe(14);
    expect(spzFractionalBits([0, 0, 2048])).toBe(11);
    expect(spzFractionalBits([1e7, 0, 0])).toBe(0);
    expect(spzFractionalBits([Infinity, NaN, 1])).toBe(16);
    // a 400 m shell still round-trips its far points
    const m = allocSplatModel(2, 0);
    m.means.set([399.9, -399.9, 12.5, 0.001, 0.002, -0.003]);
    const d = unpackSplatSPZ(packSplatSPZ(m));
    for (let k = 0; k < 6; k++) expect(Math.abs(d.model.means[k] - m.means[k])).toBeLessThanOrEqual(2 ** -15);
  });

  it('reads a legacy version-2 payload (8-bit first-three quaternions)', () => {
    const raw = new Uint8Array(16 + 19);
    const dv = new DataView(raw.buffer);
    dv.setUint32(0, SPZ_MAGIC, true);
    dv.setUint32(4, 2, true);
    dv.setUint32(8, 1, true);
    raw[13] = 12;
    // RUB position (1, 2, −3) at 12 fractional bits
    [4096, 8192, -12288].forEach((v, k) => {
      const u = v & 0xffffff;
      raw.set([u & 0xff, (u >> 8) & 0xff, (u >> 16) & 0xff], 16 + 3 * k);
    });
    raw[25] = 255; // alpha
    raw.set([128, 128, 128], 26);
    raw.set([144, 144, 144], 29);
    // quaternion (x, y, z) = −√½ about x (w = √½): x byte = (−0.7071 + 1) · 127.5
    raw.set([Math.round((1 - Math.SQRT1_2) * 127.5), 128, 128], 32);
    const d = unpackSplatSPZ(raw);
    expect(d.version).toBe(2);
    expect(Array.from(d.model.means)).toEqual([1, 3, 2]); // (x', −z', y')
    expect(d.model.logScales[0]).toBeCloseTo(-1, 6);
    expect(sigmoid(d.model.opacities[0])).toBeGreaterThan(0.99);
    // the RUB −90°-about-x rotation is the identity in the Z-up world
    expect(angleBetween(rot(d.model, 0), Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1))).toBeLessThan(0.02);
  });

  it('refuses version 4 (ZSTD) and non-gzip data', async () => {
    const v4 = new Uint8Array(32);
    new DataView(v4.buffer).setUint32(0, SPZ_MAGIC, true);
    new DataView(v4.buffer).setUint32(4, 4, true);
    await expect(decodeSplatSPZ(v4)).rejects.toThrow(/version 4/);
    await expect(decodeSplatSPZ(new Uint8Array([1, 2, 3, 4]))).rejects.toThrow(/gzip/);
    expect(() => packSplatSPZ(allocSplatModel(0, 0))).toThrow(/no Gaussians/);
  });

  it('is under 30 % of the INRIA PLY for a smooth model (degree 0)', async () => {
    // a surface-like model: Gaussians on a 60 × 60 grid with smooth colours and scales
    const n = 3600;
    const m = allocSplatModel(n, 0);
    for (let i = 0; i < n; i++) {
      const u = (i % 60) / 59;
      const v = Math.floor(i / 60) / 59;
      m.means.set([u - 0.5, v - 0.5, 0.05 * Math.sin(6 * u) * Math.cos(4 * v)], 3 * i);
      m.logScales.set([-4.2, -4.2, -6], 3 * i);
      m.opacities[i] = logit(0.6 + 0.3 * u);
      m.sh.set([(u - 0.5) / SH_C0, (v - 0.5) / SH_C0, 0.2], 3 * i);
    }
    const spz = await encodeSplatSPZ(m);
    const ply = encodeSplatPLY(m);
    console.log(`[spz] ${n} Gaussians: SPZ ${spz.length} B (${(spz.length / n).toFixed(1)} B each), PLY ${ply.length} B → ${((100 * spz.length) / ply.length).toFixed(1)} %`);
    expect(spz.length).toBeLessThan(0.3 * ply.length);
  });
});
