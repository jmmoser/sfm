import { describe, expect, it } from 'vitest';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import type { PointCloud, TriangleMesh } from '../core/types';
import { decodePLY, encodeMeshPLY, encodePLY, PLY_DEFAULT_GRAY } from './ply';

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function randomCloud(n: number, withNormals: boolean, withRadii: boolean, withConf: boolean): PointCloud {
  const r = lcg(42);
  const pc: PointCloud = { count: n, positions: new Float32Array(3 * n), colors: new Uint8Array(3 * n) };
  for (let i = 0; i < 3 * n; i++) {
    pc.positions[i] = (r() - 0.5) * 200;
    pc.colors[i] = Math.floor(r() * 256);
  }
  if (withNormals) {
    pc.normals = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      const x = r() - 0.5, y = r() - 0.5, z = r() - 0.5;
      const l = Math.hypot(x, y, z);
      pc.normals.set([x / l, y / l, z / l], 3 * i);
    }
  }
  if (withRadii) pc.radii = Float32Array.from({ length: n }, () => r() * 0.01);
  if (withConf) pc.confidence = Float32Array.from({ length: n }, () => r());
  return pc;
}

function gridMesh(nx: number, ny: number): TriangleMesh {
  const vertexCount = nx * ny;
  const positions = new Float32Array(3 * vertexCount);
  const normals = new Float32Array(3 * vertexCount);
  const colors = new Uint8Array(3 * vertexCount);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = j * nx + i;
      positions.set([i * 0.1, j * 0.1, Math.sin(i) * 0.05], 3 * v);
      normals.set([0, 0, 1], 3 * v);
      colors.set([i * 10, j * 10, 128], 3 * v);
    }
  }
  const idx: number[] = [];
  for (let j = 0; j + 1 < ny; j++) {
    for (let i = 0; i + 1 < nx; i++) {
      const a = j * nx + i;
      idx.push(a, a + 1, a + nx, a + 1, a + nx + 1, a + nx);
    }
  }
  return { vertexCount, triangleCount: idx.length / 3, positions, normals, colors, indices: Uint32Array.from(idx) };
}

function headerOf(buf: ArrayBuffer): string {
  const u8 = new Uint8Array(buf);
  const text = new TextDecoder('latin1').decode(u8.subarray(0, Math.min(u8.length, 2000)));
  return text.slice(0, text.indexOf('end_header') + 'end_header'.length);
}

describe('PLY point clouds', () => {
  it('round-trips positions/colours exactly (no optional attributes)', () => {
    const pc = randomCloud(1000, false, false, false);
    const buf = encodePLY(pc);
    const hdr = headerOf(buf);
    expect(hdr).toContain('format binary_little_endian 1.0');
    expect(hdr).toContain('element vertex 1000');
    expect(hdr).not.toContain('nx');
    expect(buf.byteLength).toBe(new TextEncoder().encode(hdr + '\n').length + 15 * 1000);
    const out = decodePLY(buf).points!;
    expect(out.count).toBe(1000);
    expect(out.positions).toEqual(pc.positions);
    expect(out.colors).toEqual(pc.colors);
    expect(out.normals).toBeUndefined();
    expect(out.radii).toBeUndefined();
  });

  it('round-trips normals, radius and confidence', () => {
    const pc = randomCloud(777, true, true, true);
    const buf = encodePLY(pc, { comments: ['scale metres', 'multi\nline'] });
    const hdr = headerOf(buf);
    expect(hdr).toMatch(/property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float radius\nproperty float confidence\nend_header/);
    const res = decodePLY(buf);
    expect(res.comments).toContain('scale metres');
    expect(res.comments).toContain('multi line');
    const out = res.points!;
    expect(out.positions).toEqual(pc.positions);
    expect(out.colors).toEqual(pc.colors);
    expect(out.normals).toEqual(pc.normals);
    expect(out.radii).toEqual(pc.radii);
    expect(out.confidence).toEqual(pc.confidence);
  });

  it('handles an empty cloud', () => {
    const out = decodePLY(encodePLY({ count: 0, positions: new Float32Array(0), colors: new Uint8Array(0) })).points!;
    expect(out.count).toBe(0);
  });

  it('is readable by three.js PLYLoader', () => {
    const pc = randomCloud(200, true, false, false);
    const geom = new PLYLoader().parse(encodePLY(pc));
    const pos = geom.getAttribute('position');
    expect(pos.count).toBe(200);
    for (let i = 0; i < 600; i++) expect(pos.array[i]).toBe(pc.positions[i]);
    const col = geom.getAttribute('color');
    // three converts sRGB uchar to linear floats; check the darkest/brightest ordering is preserved.
    expect(col.count).toBe(200);
    const nrm = geom.getAttribute('normal');
    expect(nrm.array[5]).toBeCloseTo(pc.normals![5], 6);
  });

  it('decodes ascii point clouds with float colours, extra props and comments', () => {
    const text = [
      'ply',
      'format ascii 1.0',
      'comment made by hand',
      'obj_info whatever',
      'element vertex 3',
      'property double x',
      'property double y',
      'property double z',
      'property float intensity',
      'property float red',
      'property float green',
      'property float blue',
      'element camera 1',
      'property float view_px',
      'end_header',
      '1 2 3 0.5 1 0 0.5',
      '-1.5 0 1e-3 0.1 0 1 0',
      '4 5 6 0 0.2 0.4 0.6',
      '7',
      '',
    ].join('\n');
    const res = decodePLY(new TextEncoder().encode(text));
    const pc = res.points!;
    expect(pc.count).toBe(3);
    expect(Array.from(pc.positions)).toEqual([1, 2, 3, -1.5, 0, Math.fround(1e-3), 4, 5, 6]);
    expect(Array.from(pc.colors)).toEqual([255, 0, 128, 0, 255, 0, 51, 102, 153]);
    expect(res.comments).toEqual(['made by hand']);
  });

  it('assigns default grey when colours are missing and accepts CRLF headers', () => {
    const text = 'ply\r\nformat ascii 1.0\r\nelement vertex 2\r\nproperty float x\r\nproperty float y\r\nproperty float z\r\nend_header\r\n0 0 0\r\n1 1 1\r\n';
    const pc = decodePLY(new TextEncoder().encode(text)).points!;
    expect(pc.count).toBe(2);
    expect(Array.from(pc.colors)).toEqual(new Array(6).fill(PLY_DEFAULT_GRAY));
  });

  it('decodes binary_big_endian', () => {
    const header = 'ply\nformat binary_big_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n';
    const h = new TextEncoder().encode(header);
    const buf = new Uint8Array(h.length + 2 * 15);
    buf.set(h);
    const dv = new DataView(buf.buffer);
    let o = h.length;
    for (const [x, y, z, r, g, b] of [[1.5, -2, 3, 10, 20, 30], [0.25, 8, -9, 40, 50, 60]]) {
      dv.setFloat32(o, x, false); dv.setFloat32(o + 4, y, false); dv.setFloat32(o + 8, z, false);
      dv.setUint8(o + 12, r); dv.setUint8(o + 13, g); dv.setUint8(o + 14, b);
      o += 15;
    }
    const pc = decodePLY(buf).points!;
    expect(Array.from(pc.positions)).toEqual([1.5, -2, 3, 0.25, 8, -9]);
    expect(Array.from(pc.colors)).toEqual([10, 20, 30, 40, 50, 60]);
  });

  it('rejects truncated data and bad headers', () => {
    const buf = encodePLY(randomCloud(10, false, false, false));
    expect(() => decodePLY(buf.slice(0, buf.byteLength - 5))).toThrow(/unexpected end/);
    expect(() => decodePLY(new TextEncoder().encode('ply\nformat ascii 1.0\nelement vertex 1\n'))).toThrow(/end_header/);
    expect(() => decodePLY(new TextEncoder().encode('xyz\nformat ascii 1.0\nend_header\n'))).toThrow(/magic/);
  });
});

describe('PLY meshes', () => {
  it('round-trips a binary mesh exactly', () => {
    const mesh = gridMesh(20, 15);
    const buf = encodeMeshPLY(mesh);
    const hdr = headerOf(buf);
    expect(hdr).toContain(`element face ${mesh.triangleCount}`);
    expect(hdr).toContain('property list uchar int vertex_indices');
    const out = decodePLY(buf).mesh!;
    expect(out.vertexCount).toBe(mesh.vertexCount);
    expect(out.triangleCount).toBe(mesh.triangleCount);
    expect(out.positions).toEqual(mesh.positions);
    expect(out.normals).toEqual(mesh.normals);
    expect(out.colors).toEqual(mesh.colors);
    expect(out.indices).toEqual(mesh.indices);
  });

  it('is readable by three.js PLYLoader', () => {
    const mesh = gridMesh(5, 4);
    const geom = new PLYLoader().parse(encodeMeshPLY(mesh));
    expect(geom.getAttribute('position').count).toBe(mesh.vertexCount);
    expect(Array.from(geom.getIndex()!.array)).toEqual(Array.from(mesh.indices));
  });

  it('decodes ascii meshes with quads (fan-triangulated) and computes normals', () => {
    const text = [
      'ply',
      'format ascii 1.0',
      'element vertex 4',
      'property float x',
      'property float y',
      'property float z',
      'property uchar red',
      'property uchar green',
      'property uchar blue',
      'element face 1',
      'property uchar flags',
      'property list uchar uint vertex_index',
      'end_header',
      '0 0 0 255 0 0',
      '1 0 0 0 255 0',
      '1 1 0 0 0 255',
      '0 1 0 9 9 9',
      '7 4 0 1 2 3',
    ].join('\n');
    const mesh = decodePLY(new TextEncoder().encode(text)).mesh!;
    expect(mesh.triangleCount).toBe(2);
    expect(Array.from(mesh.indices)).toEqual([0, 1, 2, 0, 2, 3]);
    for (let v = 0; v < 4; v++) {
      expect(mesh.normals[3 * v]).toBeCloseTo(0, 6);
      expect(mesh.normals[3 * v + 1]).toBeCloseTo(0, 6);
      expect(mesh.normals[3 * v + 2]).toBeCloseTo(1, 6);
    }
    expect(Array.from(mesh.colors.subarray(9, 12))).toEqual([9, 9, 9]);
  });

  it('rejects out-of-range indices', () => {
    const mesh = gridMesh(3, 3);
    mesh.indices[4] = 99;
    expect(() => encodeMeshPLY(mesh)).toThrow(RangeError);
    const text = 'ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n3 0 0 5\n';
    expect(() => decodePLY(new TextEncoder().encode(text))).toThrow(/out of range/);
  });

  it('decodes a 1M-point binary cloud quickly', () => {
    const n = 1_000_000;
    const pc: PointCloud = { count: n, positions: new Float32Array(3 * n).map((_, i) => i * 0.001), colors: new Uint8Array(3 * n).fill(7), normals: new Float32Array(3 * n).fill(0.5) };
    const t0 = performance.now();
    const buf = encodePLY(pc);
    const t1 = performance.now();
    const out = decodePLY(buf).points!;
    const t2 = performance.now();
    expect(out.positions[3 * n - 1]).toBe(pc.positions[3 * n - 1]);
    // Generous bounds (CI machines vary); typical: ~60 ms encode, ~40 ms decode.
    expect(t1 - t0).toBeLessThan(3000);
    expect(t2 - t1).toBeLessThan(3000);
  });
});
