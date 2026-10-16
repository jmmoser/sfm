/**
 * Adversarial regression tests for src/io (review pass). Each test names the
 * defect it pins down; all of them failed on the pre-review implementation.
 *
 * Node-only modules are loaded through an untyped dynamic import so the test
 * typechecks under the browser-typed main tsconfig (no @types/node there).
 */
import { describe, expect, it } from 'vitest';
import type { TriangleMesh } from '../core/types';
import { exportColmapText, importColmapText, type ColmapModelInput } from './colmap';
import { sanitizeFilename } from './download';
import { encodeGLB, parseGLB } from './glb';
import { decodePLY } from './ply';
import {
  deserializeTriangleMesh,
  float16BitsToFloat32,
  float32ToFloat16Bits,
  packFloat16,
  fromJSONSafe,
  parseJSONSafe,
  serializeTriangleMesh,
  stringifyJSONSafe,
  toJSONSafe,
} from './serialize';
import { createZip, createZipBlob, normalizeZipPath, readZip, ZipError } from './zip';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const nodeImport = (m: string): Promise<any> => import(/* @vite-ignore */ m);

async function pythonZipInfo(zip: Uint8Array): Promise<{ count: number; bad: string | null; names: string[] }> {
  const { execFileSync } = await nodeImport('node:child_process');
  const { mkdtempSync, rmSync, writeFileSync } = await nodeImport('node:fs');
  const { tmpdir } = await nodeImport('node:os');
  const { join } = await nodeImport('node:path');
  const dir: string = mkdtempSync(join(tmpdir(), 'sfm-zip-adv-'));
  try {
    const p = join(dir, 'a.zip');
    writeFileSync(p, zip);
    const py = [
      'import json, sys, zipfile',
      'with zipfile.ZipFile(sys.argv[1]) as z:',
      '    il = z.infolist()',
      '    print(json.dumps(dict(count=len(il), bad=z.testzip(), names=[i.filename for i in il[:3]])))',
    ].join('\n');
    return JSON.parse(execFileSync('python3', ['-I', '-c', py, p], { encoding: 'utf-8' }));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// ZIP
// ---------------------------------------------------------------------------

describe('zip (adversarial)', () => {
  it('readZip: a valid non-ZIP64 archive with exactly 65535 entries (createZip allows it) is readable', async () => {
    const files = Array.from({ length: 0xffff }, (_, i) => ({ name: `${i}`, data: '', lastModified: 0 }));
    const zip = createZip(files);
    const entries = readZip(zip);
    expect(entries.length).toBe(0xffff);
    expect(entries[0xfffe].name).toBe('65534');
    const py = await pythonZipInfo(zip);
    expect(py.count).toBe(0xffff);
    expect(py.bad).toBeNull();
  });

  it('readZip: still rejects real ZIP64 archives (locator present)', () => {
    const zip = createZip([{ name: 'a', data: 'x' }]);
    // Splice a ZIP64 end-of-central-directory locator in front of the EOCD and mark the count as 0xFFFF.
    const eocd = zip.length - 22;
    const out = new Uint8Array(zip.length + 20);
    out.set(zip.subarray(0, eocd), 0);
    const dv = new DataView(out.buffer);
    dv.setUint32(eocd, 0x07064b50, true);
    out.set(zip.subarray(eocd), eocd + 20);
    dv.setUint16(eocd + 20 + 8, 0xffff, true);
    dv.setUint16(eocd + 20 + 10, 0xffff, true);
    expect(() => readZip(out)).toThrow(/ZIP64/);
  });

  it('readZip: an EOCD signature inside the archive comment does not hijack parsing', () => {
    const files = [{ name: 'a.txt', data: 'hello' }, { name: 'b/c.bin', data: new Uint8Array([1, 2, 3]) }];
    const zip = createZip(files);
    // Hand-build a comment that contains a fake EOCD record (another writer could have produced it).
    const fake = new Uint8Array(40);
    new DataView(fake.buffer).setUint32(4, 0x06054b50, true);
    const out = new Uint8Array(zip.length + fake.length);
    out.set(zip);
    out.set(fake, zip.length);
    new DataView(out.buffer).setUint16(zip.length - 2, fake.length, true);
    const entries = readZip(out);
    expect(entries.map((e) => e.name)).toEqual(['a.txt', 'b/c.bin']);
    expect(Array.from(entries[1].data)).toEqual([1, 2, 3]);
  });

  it('createZip: refuses an archive comment containing the EOCD signature (ambiguous for every reader)', async () => {
    const comment = 'PK\u0005\u0006' + 'x'.repeat(30);
    expect(() => createZip([{ name: 'a', data: 'x' }], { comment })).toThrow(ZipError);
    await expect(createZipBlob([{ name: 'a', data: 'x' }], { comment })).rejects.toThrow(ZipError);
  });

  it('createZip/createZipBlob: non-Uint8 typed arrays are stored as raw bytes, not converted element-wise', async () => {
    const f = Float32Array.of(1.5, -2.25, 1e-3);
    const expected = Array.from(new Uint8Array(f.buffer));
    const zip = createZip([{ name: 'f.bin', data: f as unknown as Uint8Array }]);
    expect(Array.from(readZip(zip)[0].data)).toEqual(expected);
    const view = new DataView(f.buffer, 4, 4);
    const zip2 = createZip([{ name: 'v.bin', data: view as unknown as Uint8Array }]);
    expect(Array.from(readZip(zip2)[0].data)).toEqual(expected.slice(4, 8));
    const entry = { name: 'f.bin', data: f as unknown as Uint8Array, lastModified: 0 };
    const blobBytes = new Uint8Array(await (await createZipBlob([entry])).arrayBuffer());
    expect(blobBytes).toEqual(createZip([entry]));
    expect(Array.from(readZip(blobBytes)[0].data)).toEqual(expected);
  });

  it("normalizeZipPath: '.' segments are removed so 'a/./b' and 'a/b' are detected as duplicates", () => {
    expect(normalizeZipPath('a/./b.txt')).toBe('a/b.txt');
    expect(() => createZip([{ name: 'a/b.txt', data: 'x' }, { name: 'a/./b.txt', data: 'y' }])).toThrow(/Duplicate/);
  });
});

// ---------------------------------------------------------------------------
// PLY
// ---------------------------------------------------------------------------

function plyWithHeader(header: string[], body: Uint8Array): Uint8Array {
  const h = new TextEncoder().encode(header.join('\n') + '\n');
  const out = new Uint8Array(h.length + body.length);
  out.set(h);
  out.set(body, h.length);
  return out;
}

describe('ply (adversarial)', () => {
  it('a header claiming huge element counts is rejected BEFORE allocating (multi-GB allocation = iOS jetsam)', () => {
    const vertexProps = ['property float x', 'property float y', 'property float z', 'property uchar red', 'property uchar green', 'property uchar blue'];
    const cases: [string, string[]][] = [
      ['binary_little_endian', [`element vertex 200000000`, ...vertexProps]],
      ['ascii', [`element vertex 900000000`, ...vertexProps]],
      ['binary_little_endian', ['element vertex 3', ...vertexProps, 'element face 400000000', 'property list uchar int vertex_indices']],
      ['binary_big_endian', ['element vertex 3', ...vertexProps, 'element face 400000000', 'property list uchar int vertex_indices', 'property uchar flags']],
    ];
    for (const [format, els] of cases) {
      const ply = plyWithHeader(['ply', `format ${format} 1.0`, ...els, 'end_header'], new Uint8Array(64).fill(format === 'ascii' ? 0x31 : 0));
      const t0 = performance.now();
      let err: unknown = null;
      try {
        decodePLY(ply);
      } catch (e) {
        err = e;
      }
      expect(err, format).toBeInstanceOf(Error);
      expect(String((err as Error).message), format).toMatch(/^PLY:/);
      expect(performance.now() - t0, format).toBeLessThan(250);
    }
  });

  it('valid files at the exact minimum size still decode (no off-by-one in the size guard)', () => {
    const asc = plyWithHeader(
      ['ply', 'format ascii 1.0', 'element vertex 2', 'property float x', 'property float y', 'property float z', 'end_header'],
      new TextEncoder().encode('1 2 3\n4 5 6'), // no trailing newline: exactly 2·6 − 1 bytes
    );
    expect(Array.from(decodePLY(asc).points!.positions)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("binary 'list uchar int' fast path: an empty polygon as the LAST face does not read past the buffer", () => {
    // vertex: 3 bytes; faces: [0] (empty), [3, 0, 0, 0] (13 bytes), [0] (empty, last byte of the file)
    const body = new Uint8Array(3 + 1 + 13 + 1);
    body.set([1, 2, 3]);
    body[4] = 3; // second face: count 3, indices 0 0 0 (zero bytes already)
    const ply = plyWithHeader(
      ['ply', 'format binary_little_endian 1.0', 'element vertex 1', 'property uchar x', 'property uchar y', 'property uchar z',
        'element face 3', 'property list uchar int vertex_indices', 'end_header'],
      body,
    );
    const m = decodePLY(ply).mesh!;
    expect(m.triangleCount).toBe(1);
    expect(Array.from(m.indices)).toEqual([0, 0, 0]);
    expect(Array.from(m.positions)).toEqual([1, 2, 3]);
  });
});

// ---------------------------------------------------------------------------
// GLB
// ---------------------------------------------------------------------------

describe('glb (adversarial)', () => {
  it('an empty scene omits scene.nodes (glTF schema: nodes has minItems 1)', () => {
    for (const input of [
      {},
      { points: { count: 2, positions: Float32Array.of(NaN, 0, 0, 0, Infinity, 0), colors: new Uint8Array(6) } },
      { mesh: { vertexCount: 0, triangleCount: 0, positions: new Float32Array(0), normals: new Float32Array(0), colors: new Uint8Array(0), indices: new Uint32Array(0) } },
    ]) {
      const g = parseGLB(encodeGLB(input));
      const scenes = g.json.scenes as { nodes?: number[] }[];
      expect(scenes.length).toBe(1);
      if (scenes[0].nodes !== undefined) expect(scenes[0].nodes.length).toBeGreaterThan(0);
      expect(g.json.scene).toBe(0);
    }
  });

  it('the empty scene still loads in three.js GLTFLoader', async () => {
    const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js');
    const gltf = await new GLTFLoader().parseAsync(encodeGLB({}), '');
    expect(gltf.scene.children.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// COLMAP
// ---------------------------------------------------------------------------

function tinyModel(): ColmapModelInput {
  return {
    cameras: [{ id: 1, cam: { width: 640, height: 480, f: 500, cx: 319.5, cy: 239.5, k1: 0, k2: 0 } }],
    images: [
      {
        id: 1,
        cameraId: 1,
        name: 'a.jpg',
        pose: { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(0, 0, 0) },
        points2D: [
          { x: 10, y: 20, point3D: 7 },
          { x: 30, y: 40, point3D: 99 }, // references a point that is not exported
        ],
      },
    ],
    points: [{ id: 7, xyz: [0, 0, 1], color: [NaN, 300, -4], error: 0.5, track: [{ image: 1, point2D: 0 }] }],
  };
}

describe('colmap (adversarial)', () => {
  it('never writes NaN colours (COLMAP parses R G B with stoi)', () => {
    const pts = exportColmapText(tinyModel())[2].data.split('\n').filter((l) => l && !l.startsWith('#'));
    expect(pts[0].split(' ').slice(4, 7)).toEqual(['0', '255', '0']);
  });

  it('POINTS2D entries that reference a point missing from the export are written as -1 (dangling ids crash COLMAP)', () => {
    const imgs = exportColmapText(tinyModel())[1].data.split('\n').filter((l) => !l.startsWith('#'));
    expect(imgs[1].split(' ')).toEqual(['10.5', '20.5', '7', '30.5', '40.5', '-1']);
    const back = importColmapText(exportColmapText(tinyModel()));
    expect(back.images[0].points2D.map((p) => p.point3D)).toEqual([7, -1]);
  });

  it('rejects ids / sizes COLMAP cannot parse (negative, fractional) and broken track references', () => {
    const bad = (mut: (m: ColmapModelInput) => void): (() => unknown) => () => {
      const m = tinyModel();
      mut(m);
      return exportColmapText(m);
    };
    expect(bad((m) => (m.images[0].id = -1))).toThrow(RangeError);
    expect(bad((m) => (m.images[0].cameraId = 1.5))).toThrow(RangeError);
    expect(bad((m) => (m.cameras[0].cam.width = 640.5))).toThrow(RangeError);
    expect(bad((m) => (m.points[0].id = -3))).toThrow(RangeError);
    expect(bad((m) => (m.points[0].track[0].point2D = 5))).toThrow(RangeError);
    expect(bad((m) => (m.points[0].track[0].image = 42))).toThrow(RangeError);
    expect(bad(() => {})).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// serialize
// ---------------------------------------------------------------------------

/** Correctly rounded (RNE) binary16 bits of a double, by exhaustive search over the finite halves. */
function refHalfBits(v: number, table: { bits: number; val: number }[]): number {
  const sign = v < 0 || Object.is(v, -0) ? 0x8000 : 0;
  const a = Math.abs(v);
  if (a >= 65520) return sign | 0x7c00; // ≥ max + ½ ulp → Inf (65520 is a tie with odd 65504 → Inf)
  let lo = 0;
  let hi = table.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (table[mid].val <= a) lo = mid;
    else hi = mid;
  }
  const dl = a - table[lo].val; // exact (Sterbenz / small magnitudes)
  const dh = table[hi].val - a;
  const pick = dl < dh ? lo : dh < dl ? hi : (table[lo].bits & 1) === 0 ? lo : hi;
  return sign | table[pick].bits;
}

describe('float16 (adversarial)', () => {
  it('float32ToFloat16Bits is correctly rounded for DOUBLE inputs too (no double rounding through float32)', () => {
    const table: { bits: number; val: number }[] = [];
    for (let b = 0; b < 0x7c00; b++) table.push({ bits: b, val: float16BitsToFloat32(b) });
    // Midpoints between adjacent halves nudged by far less than a float32 ulp: rounding to
    // float32 first lands exactly on the tie and ties-to-even then picks the wrong side.
    const inputs: number[] = [];
    for (let b = 1; b < 0x7bff; b += 37) {
      const mid = (table[b].val + table[b + 1].val) / 2;
      const eps = mid * 2 ** -40 + 2 ** -60;
      inputs.push(mid + eps, mid - eps, -(mid + eps), mid);
    }
    const packed = packFloat16(Float64Array.from(inputs));
    let wrong = 0;
    for (let i = 0; i < inputs.length; i++) {
      const want = refHalfBits(inputs[i], table);
      if (float32ToFloat16Bits(inputs[i]) !== want || packed[i] !== want) wrong++;
    }
    expect(wrong).toBe(0);
  });
});

describe('serialize (adversarial)', () => {
  it('TriangleMesh: indices ≥ 65536 with ≤ 65536 vertices are not silently truncated to 16 bits', () => {
    const mesh: TriangleMesh = {
      vertexCount: 3,
      triangleCount: 1,
      positions: new Float32Array(9),
      normals: new Float32Array(9),
      colors: new Uint8Array(9),
      indices: Uint32Array.of(0, 1, 70000), // corrupt input must survive a round trip unchanged
    };
    const back = deserializeTriangleMesh(serializeTriangleMesh(mesh));
    expect(Array.from(back.indices)).toEqual([0, 1, 70000]);
  });

  it('JSON-safe encoding preserves -0 (scalars and typed arrays)', () => {
    const v = { a: -0, t: Float64Array.of(-0, 0, 1), arr: [-0] };
    const back = parseJSONSafe<typeof v>(stringifyJSONSafe(v));
    expect(Object.is(back.a, -0)).toBe(true);
    expect(Object.is(back.t[0], -0)).toBe(true);
    expect(Object.is(back.t[1], 0)).toBe(true);
    expect(Object.is(back.arr[0], -0)).toBe(true);
    expect(fromJSONSafe(toJSONSafe(-0))).toBe(-0);
  });

  it('JSON-safe encoding refuses values it would otherwise silently turn into {} (Set, Date, ArrayBuffer, DataView)', () => {
    expect(() => toJSONSafe({ s: new Set([1]) })).toThrow(TypeError);
    expect(() => toJSONSafe({ d: new Date(0) })).toThrow(TypeError);
    expect(() => toJSONSafe({ b: new ArrayBuffer(4) })).toThrow(TypeError);
    expect(() => toJSONSafe({ v: new DataView(new ArrayBuffer(4)) })).toThrow(TypeError);
    expect(toJSONSafe({ m: new Map([[1, 2]]), o: Object.create(null) })).toEqual({ m: { $map: [[1, 2]] }, o: {} });
  });
});

// ---------------------------------------------------------------------------
// download (pure helper)
// ---------------------------------------------------------------------------

describe('sanitizeFilename (adversarial)', () => {
  const loneSurrogate = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

  it('keeps the extension when truncating long names (the share sheet / Files infer the type from it)', () => {
    const s = sanitizeFilename('scan-'.repeat(80) + '.glb');
    expect(s.length).toBeLessThanOrEqual(200);
    expect(s.endsWith('.glb')).toBe(true);
  });

  it('never splits a surrogate pair', () => {
    const s = sanitizeFilename('\u{1F600}'.repeat(150) + '.ply');
    expect(loneSurrogate.test(s)).toBe(false);
    expect(s.endsWith('.ply')).toBe(true);
    expect(s.length).toBeLessThanOrEqual(200);
  });
});
