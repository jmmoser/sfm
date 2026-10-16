import { describe, expect, it } from 'vitest';
import type { FeatureSet, PointCloud, TriangleMesh } from '../core/types';
import {
  deserializeDepthMap,
  deserializeFeatureSet,
  deserializePointCloud,
  deserializeTriangleMesh,
  float16BitsToFloat32,
  float32ToFloat16Bits,
  fromJSONSafe,
  packFloat16,
  parseJSONSafe,
  poseFromJSON,
  poseToJSON,
  serializeDepthMap,
  serializeFeatureSet,
  serializePointCloud,
  serializeTriangleMesh,
  serialKind,
  SerialKind,
  stringifyJSONSafe,
  toJSONSafe,
  unpackFloat16,
  type DepthMapLike,
} from './serialize';

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Fast structural equality: typed arrays compared by type + bytes, other fields by Object.is. */
/** Byte-wise equality of two views (no Node Buffer: the main tsconfig is browser-typed). */
function bytesEqual(a: ArrayBufferView, b: ArrayBufferView): boolean {
  if (a.byteLength !== b.byteLength) return false;
  const ua = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  const ub = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) return false;
  return true;
}

function expectSame(a: object, b: object): void {
  const ka = Object.keys(a).filter((k) => (a as Record<string, unknown>)[k] !== undefined).sort();
  const kb = Object.keys(b).filter((k) => (b as Record<string, unknown>)[k] !== undefined).sort();
  expect(kb).toEqual(ka);
  for (const k of ka) {
    const x = (a as Record<string, unknown>)[k];
    const y = (b as Record<string, unknown>)[k];
    if (ArrayBuffer.isView(x)) {
      expect(ArrayBuffer.isView(y) && y.constructor === x.constructor, k).toBe(true);
      const yb = y as ArrayBufferView;
      expect(bytesEqual(x, yb), k).toBe(true);
    } else expect(Object.is(x, y), k).toBe(true);
  }
}

describe('float16', () => {
  it('round-trips every binary16 bit pattern', () => {
    let bad = 0;
    for (let h = 0; h < 65536; h++) {
      const f = float16BitsToFloat32(h);
      const back = float32ToFloat16Bits(f);
      if (Number.isNaN(f)) {
        if ((back & 0x7c00) !== 0x7c00 || (back & 0x3ff) === 0) bad++;
      } else if (back !== h) bad++;
    }
    expect(bad).toBe(0);
  });

  it('decodes known values', () => {
    expect(float16BitsToFloat32(0x3c00)).toBe(1);
    expect(float16BitsToFloat32(0xc000)).toBe(-2);
    expect(float16BitsToFloat32(0x7bff)).toBe(65504);
    expect(float16BitsToFloat32(0x0400)).toBe(2 ** -14);
    expect(float16BitsToFloat32(0x0001)).toBe(2 ** -24);
    expect(float16BitsToFloat32(0x7c00)).toBe(Infinity);
    expect(float16BitsToFloat32(0xfc00)).toBe(-Infinity);
    expect(Object.is(float16BitsToFloat32(0x8000), -0)).toBe(true);
    expect(float16BitsToFloat32(0x3555)).toBeCloseTo(0.33325, 5);
  });

  it('rounds to nearest, ties to even, with overflow and gradual underflow', () => {
    expect(float32ToFloat16Bits(65519)).toBe(0x7bff);
    expect(float32ToFloat16Bits(65520)).toBe(0x7c00); // tie → even → Inf
    expect(float32ToFloat16Bits(1e10)).toBe(0x7c00);
    expect(float32ToFloat16Bits(-1e10)).toBe(0xfc00);
    expect(float32ToFloat16Bits(1 + 2 ** -11)).toBe(0x3c00); // tie → even (down)
    expect(float32ToFloat16Bits(1 + 3 * 2 ** -11)).toBe(0x3c02); // tie → even (up)
    expect(float32ToFloat16Bits(1 + 2 ** -11 + 2 ** -20)).toBe(0x3c01);
    expect(float32ToFloat16Bits(2 ** -25)).toBe(0x0000); // tie → even (0)
    expect(float32ToFloat16Bits(1.5 * 2 ** -25)).toBe(0x0001);
    expect(float32ToFloat16Bits(3 * 2 ** -25)).toBe(0x0002); // tie 1.5 ulp → even (2)
    expect(float32ToFloat16Bits(2 ** -14 - 2 ** -25)).toBe(0x0400); // rounds up into the normals
    expect(float32ToFloat16Bits(1e-10)).toBe(0x0000);
    expect(float32ToFloat16Bits(-1e-10)).toBe(0x8000);
    expect(float32ToFloat16Bits(NaN) & 0x7fff).toBeGreaterThan(0x7c00);
    expect(float32ToFloat16Bits(Infinity)).toBe(0x7c00);
  });

  it('is correctly rounded for random float32 values (vs. exhaustive neighbour search)', () => {
    const r = rng(3);
    const maybeF16 = (Math as unknown as { f16round?: (x: number) => number }).f16round;
    const failures: string[] = [];
    for (let i = 0; i < 200_000 && failures.length < 5; i++) {
      // log-uniform magnitudes across the whole half range incl. subnormals and overflow
      const x = Math.fround((r() < 0.5 ? -1 : 1) * 2 ** (r() * 42 - 26));
      const h = float32ToFloat16Bits(x);
      const y = float16BitsToFloat32(h);
      if (maybeF16 && !Object.is(y, maybeF16(x))) failures.push(`f16round(${x})`);
      if (!Number.isFinite(y)) {
        if (Math.abs(x) < 65520) failures.push(`overflow ${x}`); // IEEE overflow threshold
        continue;
      }
      // y must be at least as close as both neighbours; ties go to the even mantissa.
      const err = Math.abs(x - y);
      for (const nb of [h - 1, h + 1]) {
        if ((nb & 0x7fff) >= 0x7c00 || nb < 0 || nb > 0xffff || ((nb ^ h) & 0x8000) !== 0) continue;
        const e2 = Math.abs(x - float16BitsToFloat32(nb));
        if (err > e2 || (err === e2 && (h & 1) !== 0)) failures.push(`round ${x}`);
      }
      if (Math.abs(x) >= 2 ** -14 && err / Math.abs(x) > 2 ** -11) failures.push(`relerr ${x}`);
    }
    expect(failures).toEqual([]);
  });

  it('bulk pack/unpack keeps unit normals within 0.05°', () => {
    const r = rng(5);
    const n = 10_000;
    const v = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      const x = r() - 0.5, y = r() - 0.5, z = r() - 0.5;
      const l = Math.hypot(x, y, z);
      v.set([x / l, y / l, z / l], 3 * i);
    }
    const back = unpackFloat16(packFloat16(v));
    let maxAngle = 0;
    for (let i = 0; i < n; i++) {
      const a = v.subarray(3 * i, 3 * i + 3), b = back.subarray(3 * i, 3 * i + 3);
      const dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / Math.hypot(b[0], b[1], b[2]);
      maxAngle = Math.max(maxAngle, Math.acos(Math.min(1, dot)));
    }
    expect(maxAngle * (180 / Math.PI)).toBeLessThan(0.05);
  });
});

function featureSet(n: number): FeatureSet {
  const r = rng(11);
  return {
    count: n,
    width: 4032,
    height: 3024,
    keypoints: Float32Array.from({ length: 4 * n }, () => r() * 4000),
    descriptors: Uint8Array.from({ length: 128 * n }, () => Math.floor(r() * 256)),
    scores: Float32Array.from({ length: n }, () => r()),
  };
}

describe('binary serialisers', () => {
  it('FeatureSet round trip is exact; copies by default, views with zeroCopy', () => {
    const fs = featureSet(513);
    const buf = serializeFeatureSet(fs);
    expect(serialKind(buf)).toBe(SerialKind.FeatureSet);
    const back = deserializeFeatureSet(buf);
    expectSame(fs, back);
    expect(back.keypoints.buffer).not.toBe(buf);
    expect(back.keypoints.buffer).not.toBe(back.descriptors.buffer);
    const view = deserializeFeatureSet(buf, { zeroCopy: true });
    expect(view.keypoints.buffer).toBe(buf);
    expectSame(fs, view);
    // Works from an unaligned Uint8Array view too.
    const shifted = new Uint8Array(buf.byteLength + 3);
    shifted.set(new Uint8Array(buf), 3);
    expectSame(fs, deserializeFeatureSet(shifted.subarray(3), { zeroCopy: true }));
  });

  it('serialises only `count` features of oversized arrays', () => {
    const fs = featureSet(10);
    const trimmed = deserializeFeatureSet(serializeFeatureSet({ ...fs, count: 4 }));
    expect(trimmed.count).toBe(4);
    expect(trimmed.keypoints).toEqual(fs.keypoints.subarray(0, 16));
    expect(trimmed.descriptors.length).toBe(512);
  });

  it('PointCloud round trip (all attribute combinations, f32 exact, f16 approx)', () => {
    const r = rng(13);
    const n = 1000;
    const base: PointCloud = {
      count: n,
      positions: Float32Array.from({ length: 3 * n }, () => (r() - 0.5) * 1e3),
      colors: Uint8Array.from({ length: 3 * n }, () => Math.floor(r() * 256)),
    };
    const normals = Float32Array.from({ length: 3 * n }, () => r() * 2 - 1);
    const radii = Float32Array.from({ length: n }, () => r() * 1e-4);
    const confidence = Float32Array.from({ length: n }, () => r());
    for (let mask = 0; mask < 8; mask++) {
      const pc: PointCloud = { ...base };
      if (mask & 1) pc.normals = normals;
      if (mask & 2) pc.radii = radii;
      if (mask & 4) pc.confidence = confidence;
      expectSame(pc, deserializePointCloud(serializePointCloud(pc)));
      const h = deserializePointCloud(serializePointCloud(pc, { float16: true }));
      expectSame({ p: pc.positions, c: pc.colors, r: pc.radii }, { p: h.positions, c: h.colors, r: pc.radii ? h.radii : undefined }); // radii stay f32
      for (const k of ['normals', 'confidence'] as const) {
        const a = pc[k], b = h[k];
        expect(!!b).toBe(!!a);
        if (a && b) {
          let bad = 0;
          for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > Math.abs(a[i]) * 2 ** -11 + 1e-7) bad++;
          expect(bad).toBe(0);
        }
      }
    }
    expect(serializePointCloud({ ...base, normals }, { float16: true }).byteLength).toBeLessThan(serializePointCloud({ ...base, normals }).byteLength);
  });

  it('TriangleMesh round trip with u16 and u32 index storage', () => {
    for (const nv of [100, 70_000]) {
      const r = rng(nv);
      const nt = 500;
      const mesh: TriangleMesh = {
        vertexCount: nv,
        triangleCount: nt,
        positions: Float32Array.from({ length: 3 * nv }, () => r()),
        normals: Float32Array.from({ length: 3 * nv }, () => r()),
        colors: Uint8Array.from({ length: 3 * nv }, () => Math.floor(r() * 256)),
        indices: Uint32Array.from({ length: 3 * nt }, () => Math.floor(r() * nv)),
      };
      mesh.indices[0] = nv - 1;
      const buf = serializeTriangleMesh(mesh);
      expectSame(mesh, deserializeTriangleMesh(buf));
      const small = nv <= 65536;
      // index section: 2 or 4 bytes per index
      expect(buf.byteLength).toBeLessThan(8 + 8 + 5 * 16 + 4 * 6 * nv + 3 * nv + (small ? 2 : 4) * 3 * nt + 40);
      const h = deserializeTriangleMesh(serializeTriangleMesh(mesh, { float16: true }));
      expectSame({ i: mesh.indices }, { i: h.indices });
      expect(h.normals[5]).toBeCloseTo(mesh.normals[5], 3);
    }
  });

  it('DepthMap round trip (f32 exact, f16 for normal + cost)', () => {
    const w = 64, hgt = 48, n = w * hgt;
    const r = rng(17);
    const d: DepthMapLike = {
      width: w,
      height: hgt,
      depth: Float32Array.from({ length: n }, () => (r() < 0.1 ? 0 : 0.2 + r() * 80)),
      normal: Float32Array.from({ length: 3 * n }, () => r() * 2 - 1),
      cost: Float32Array.from({ length: n }, () => r() * 2),
    };
    expectSame(d, deserializeDepthMap(serializeDepthMap(d)));
    const full = serializeDepthMap(d).byteLength;
    const half = serializeDepthMap(d, { float16: true });
    expect(half.byteLength).toBeLessThan(full * 0.65);
    const h = deserializeDepthMap(half);
    expectSame({ d: d.depth }, { d: h.depth });
    let bad = 0;
    for (let i = 0; i < n; i++) if (Math.abs(h.cost[i] - d.cost[i]) > 2 * 2 ** -11) bad++;
    expect(bad).toBe(0);
  });

  it('rejects wrong kinds, bad magic and truncation', () => {
    const buf = serializeFeatureSet(featureSet(3));
    expect(() => deserializePointCloud(buf)).toThrow(/expected PointCloud/);
    expect(() => deserializeFeatureSet(new ArrayBuffer(16))).toThrow(/magic/);
    expect(() => deserializeFeatureSet(buf.slice(0, buf.byteLength - 10))).toThrow(/truncated/);
    expect(serialKind(new ArrayBuffer(4))).toBeNull();
    const future = buf.slice(0);
    new DataView(future).setUint16(6, 99, true);
    expect(() => deserializeFeatureSet(future)).toThrow(/version/);
  });
});

describe('JSON-safe encoding', () => {
  it('round-trips poses and nested typed arrays exactly, including non-finite values', () => {
    const value = {
      pose: { R: Float64Array.of(0.1, 0.2, 0.30000000000000004, 1 / 3, -0.5, 1e-300, 7, 8, 9), t: Float64Array.of(1, 2, 3) },
      list: [Float32Array.of(1.5, NaN), Uint8Array.of(1, 2, 255), Int32Array.of(-5)],
      weird: [NaN, Infinity, -Infinity, 0],
      map: new Map<number, { a: Float64Array }>([[3, { a: Float64Array.of(Infinity) }]]),
      gone: undefined,
      s: 'text',
      n: null,
    };
    const text = stringifyJSONSafe(value);
    expect(() => JSON.parse(text)).not.toThrow();
    const back = parseJSONSafe<typeof value>(text);
    expect(back.pose.R).toBeInstanceOf(Float64Array);
    expect(back.pose.R).toEqual(value.pose.R);
    expect(back.pose.t).toEqual(value.pose.t);
    expect(back.list[0]).toBeInstanceOf(Float32Array);
    expect(Number.isNaN(back.list[0][1])).toBe(true);
    expect(back.list[1]).toEqual(value.list[1]);
    expect(back.list[2]).toEqual(value.list[2]);
    expect(back.weird).toEqual([NaN, Infinity, -Infinity, 0]);
    expect(back.map).toBeInstanceOf(Map);
    expect(back.map.get(3)!.a[0]).toBe(Infinity);
    expect('gone' in back).toBe(false);
    expect(back.s).toBe('text');
    expect(back.n).toBeNull();
    expect(fromJSONSafe(toJSONSafe(42))).toBe(42);
  });

  it('poseToJSON / poseFromJSON', () => {
    const p = { R: Float64Array.of(1, 0, 0, 0, 0, -1, 0, 1, 0), t: Float64Array.of(0.5, -0.25, 1e-17) };
    const j = JSON.parse(JSON.stringify(poseToJSON(p)));
    const back = poseFromJSON(j);
    expect(back.R).toEqual(p.R);
    expect(back.t).toEqual(p.t);
    expect(() => poseFromJSON({ R: [1], t: [] })).toThrow();
  });
});
