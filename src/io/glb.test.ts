import { describe, expect, it } from 'vitest';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { BufferGeometry, Mesh, Points } from 'three';
import type { PointCloud, TriangleMesh } from '../core/types';
import { encodeGLB, GLB_GENERATOR, parseGLB, type ParsedGLB } from './glb';

const COMPONENT_SIZE: Record<number, number> = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const TYPE_COUNT: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

/** Structural validation in the spirit of gltf-validator for the subset we write. */
function validate(buf: ArrayBuffer): ParsedGLB {
  const u8 = new Uint8Array(buf);
  const dv = new DataView(buf);
  expect(dv.getUint32(0, true)).toBe(0x46546c67);
  expect(dv.getUint32(4, true)).toBe(2);
  expect(dv.getUint32(8, true)).toBe(buf.byteLength);
  expect(buf.byteLength % 4).toBe(0);
  const jsonLen = dv.getUint32(12, true);
  expect(jsonLen % 4).toBe(0);
  expect(dv.getUint32(16, true)).toBe(0x4e4f534a);
  const g = parseGLB(buf);
  expect(g.json.asset.version).toBe('2.0');
  if (g.bin) {
    const binLen = dv.getUint32(20 + jsonLen, true);
    expect(binLen % 4).toBe(0);
    expect(dv.getUint32(24 + jsonLen, true)).toBe(0x004e4942);
    expect((20 + jsonLen + 8) % 4).toBe(0);
    expect(g.json.buffers![0].byteLength).toBeLessThanOrEqual(binLen);
    expect(20 + jsonLen + 8 + binLen).toBe(u8.length);
  }
  for (const bv of g.json.bufferViews ?? []) {
    expect((bv.byteOffset ?? 0) % 4).toBe(0);
    expect((bv.byteOffset ?? 0) + bv.byteLength).toBeLessThanOrEqual(g.json.buffers![bv.buffer].byteLength);
    expect([34962, 34963]).toContain(bv.target);
  }
  for (const acc of g.json.accessors ?? []) {
    const bv = g.json.bufferViews![acc.bufferView!];
    const elt = COMPONENT_SIZE[acc.componentType] * TYPE_COUNT[acc.type];
    expect(acc.count).toBeGreaterThan(0);
    expect((acc.byteOffset ?? 0) + elt * acc.count).toBeLessThanOrEqual(bv.byteLength);
    expect(((bv.byteOffset ?? 0) + (acc.byteOffset ?? 0)) % COMPONENT_SIZE[acc.componentType]).toBe(0);
  }
  for (const m of g.json.meshes ?? []) {
    for (const p of m.primitives) {
      const pos = g.json.accessors![p.attributes.POSITION];
      expect(pos.type).toBe('VEC3');
      expect(pos.componentType).toBe(5126);
      // min/max must exactly match the data.
      const data = accessorFloats(g, p.attributes.POSITION);
      const min = [Infinity, Infinity, Infinity];
      const max = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < pos.count; i++) {
        for (let c = 0; c < 3; c++) {
          min[c] = Math.min(min[c], data[3 * i + c]);
          max[c] = Math.max(max[c], data[3 * i + c]);
        }
      }
      expect(pos.min).toEqual(min);
      expect(pos.max).toEqual(max);
      for (const [k, a] of Object.entries(p.attributes)) {
        expect(g.json.accessors![a].count).toBe(pos.count);
        if (k === 'NORMAL') {
          const n = accessorFloats(g, a);
          for (let i = 0; i < pos.count; i++) expect(Math.hypot(n[3 * i], n[3 * i + 1], n[3 * i + 2])).toBeCloseTo(1, 5);
        }
      }
      if (p.indices !== undefined) {
        const ia = g.json.accessors![p.indices];
        expect(ia.componentType).toBe(5125);
        expect(ia.type).toBe('SCALAR');
        const bv = g.json.bufferViews![ia.bufferView!];
        expect(bv.target).toBe(34963);
        const idx = new Uint32Array(g.bin!.slice((bv.byteOffset ?? 0) + (ia.byteOffset ?? 0)).buffer, 0, ia.count);
        for (let i = 0; i < ia.count; i++) expect(idx[i]).toBeLessThan(pos.count);
      }
    }
  }
  return g;
}

function accessorFloats(g: ParsedGLB, index: number): Float32Array {
  const acc = g.json.accessors![index];
  const bv = g.json.bufferViews![acc.bufferView!];
  const start = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
  const n = acc.count * TYPE_COUNT[acc.type];
  return new Float32Array(g.bin!.slice(start, start + 4 * n).buffer);
}

function accessorRaw(g: ParsedGLB, index: number): Uint8Array {
  const acc = g.json.accessors![index];
  const bv = g.json.bufferViews![acc.bufferView!];
  const start = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
  return g.bin!.slice(start, start + acc.count * COMPONENT_SIZE[acc.componentType] * TYPE_COUNT[acc.type]);
}

function testMesh(): TriangleMesh {
  // Tetrahedron with arbitrary (non-unit) normals and colours.
  const positions = Float32Array.of(0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3);
  const normals = Float32Array.of(-1, -1, -1, 2, 0, 0, 0, 3, 0, 0, 0, 0);
  const colors = Uint8Array.of(255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128);
  const indices = Uint32Array.of(0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3);
  return { vertexCount: 4, triangleCount: 4, positions, normals, colors, indices };
}

function testPoints(n: number): PointCloud {
  const positions = new Float32Array(3 * n);
  const colors = new Uint8Array(3 * n);
  for (let i = 0; i < n; i++) {
    positions.set([Math.cos(i) * 5.1, Math.sin(i) * 3.3, i * 0.37 - 4], 3 * i);
    colors.set([i % 256, (3 * i) % 256, (7 * i) % 256], 3 * i);
  }
  return { count: n, positions, colors };
}

function srgbToLinear(c: number): number {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

describe('encodeGLB', () => {
  it('writes a valid mesh GLB with Y-up conversion, unit normals and linear colours', () => {
    const mesh = testMesh();
    const buf = encodeGLB({ mesh, extras: { metric: true } });
    const g = validate(buf);
    expect(g.json.asset.generator).toBe(GLB_GENERATOR);
    expect((g.json.asset as { extras?: unknown }).extras).toEqual({ metric: true });
    const prim = g.json.meshes![0].primitives[0];
    expect(prim.mode).toBe(4);
    const pos = accessorFloats(g, prim.attributes.POSITION);
    // (x, y, z) -> (x, z, -y)
    expect(Array.from(pos)).toEqual([0, 0, 0, 1, 0, 0, 0, 0, -2, 0, 3, 0]);
    const pa = g.json.accessors![prim.attributes.POSITION];
    expect(pa.min).toEqual([0, 0, -2]);
    expect(pa.max).toEqual([1, 3, 0]);
    const nrm = accessorFloats(g, prim.attributes.NORMAL);
    const s = 1 / Math.sqrt(3);
    expect(nrm[0]).toBeCloseTo(-s, 6);
    expect(nrm[1]).toBeCloseTo(-s, 6);
    expect(nrm[2]).toBeCloseTo(s, 6);
    expect(Array.from(nrm.subarray(9, 12))).toEqual([0, 1, 0]); // zero normal -> +Z -> +Y
    const col = accessorFloats(g, prim.attributes.COLOR_0);
    expect(col[0]).toBeCloseTo(1, 6);
    expect(col[9]).toBeCloseTo(srgbToLinear(128), 6);
    const idx = new Uint32Array(accessorRaw(g, prim.indices!).buffer);
    expect(Array.from(idx)).toEqual(Array.from(mesh.indices));
    expect(g.json.materials![0].doubleSided).toBe(true);
    expect(g.json.extensionsUsed).toBeUndefined();
  });

  it('keeps Z-up data with yUp: false', () => {
    const g = validate(encodeGLB({ mesh: testMesh(), yUp: false }));
    const pos = accessorFloats(g, g.json.meshes![0].primitives[0].attributes.POSITION);
    expect(Array.from(pos)).toEqual([0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3]);
  });

  it('writes points as a POINTS primitive with an unlit material, dropping non-finite points', () => {
    const pc = testPoints(101);
    pc.positions[3 * 50 + 1] = NaN;
    pc.normals = new Float32Array(3 * 101).fill(1);
    const g = validate(encodeGLB({ points: pc }));
    const prim = g.json.meshes![0].primitives[0];
    expect(prim.mode).toBe(0);
    expect(prim.indices).toBeUndefined();
    expect(g.json.accessors![prim.attributes.POSITION].count).toBe(100);
    expect(prim.attributes.NORMAL).toBeDefined();
    expect(g.json.extensionsUsed).toEqual(['KHR_materials_unlit']);
    expect(g.json.materials![prim.material!].extensions).toEqual({ KHR_materials_unlit: {} });
    const pos = accessorFloats(g, prim.attributes.POSITION);
    // point 51 shifted down to slot 50
    expect(pos[3 * 50]).toBe(pc.positions[3 * 51]);
    expect(pos[3 * 50 + 1]).toBe(pc.positions[3 * 51 + 2]);
    expect(pos[3 * 50 + 2]).toBe(-pc.positions[3 * 51 + 1]);
  });

  it('supports unorm8 and unorm16 colours (VEC4, normalized)', () => {
    const pc = testPoints(10);
    for (const [fmt, ct, size] of [['unorm8', 5121, 1], ['unorm16', 5123, 2]] as const) {
      const g = validate(encodeGLB({ points: pc, colorFormat: fmt }));
      const acc = g.json.accessors![g.json.meshes![0].primitives[0].attributes.COLOR_0];
      expect(acc.componentType).toBe(ct);
      expect(acc.type).toBe('VEC4');
      expect(acc.normalized).toBe(true);
      const raw = accessorRaw(g, g.json.meshes![0].primitives[0].attributes.COLOR_0);
      const max = size === 1 ? 255 : 65535;
      const vals = size === 1 ? raw : new Uint16Array(raw.buffer);
      expect(vals[3]).toBe(max);
      expect(vals[4 * 3 + 1] / max).toBeCloseTo(srgbToLinear(pc.colors[3 * 3 + 1]), size === 1 ? 2 : 4);
    }
  });

  it('writes mesh + points as two nodes and an empty input as a valid empty scene', () => {
    const g = validate(encodeGLB({ mesh: testMesh(), points: testPoints(5), unlitMesh: true }));
    expect(g.json.nodes!.length).toBe(2);
    expect(g.json.meshes!.length).toBe(2);
    expect((g.json.scenes as { nodes: number[] }[])[0].nodes).toEqual([0, 1]);
    const empty = encodeGLB({});
    const e = validate(empty);
    expect(e.bin).toBeNull();
    expect(e.json.buffers).toBeUndefined();
  });

  it('rejects out-of-range indices', () => {
    const m = testMesh();
    m.indices[0] = 4;
    expect(() => encodeGLB({ mesh: m })).toThrow(RangeError);
  });

  it('loads in three.js GLTFLoader', async () => {
    const mesh = testMesh();
    const pc = testPoints(64);
    const buf = encodeGLB({ mesh, points: pc, colorFormat: 'unorm8' });
    const gltf = await new GLTFLoader().parseAsync(buf, '');
    const meshes: Mesh[] = [];
    const points: Points[] = [];
    gltf.scene.traverse((o) => {
      if ((o as Mesh).isMesh) meshes.push(o as Mesh);
      if ((o as Points).isPoints) points.push(o as Points);
    });
    expect(meshes.length).toBe(1);
    expect(points.length).toBe(1);
    const mg = meshes[0].geometry as BufferGeometry;
    expect(mg.getAttribute('position').count).toBe(4);
    expect(mg.getIndex()!.count).toBe(12);
    expect(mg.getAttribute('color').normalized).toBe(true);
    const pg = points[0].geometry as BufferGeometry;
    expect(pg.getAttribute('position').count).toBe(64);
    expect(pg.getAttribute('position').getY(3)).toBeCloseTo(pc.positions[3 * 3 + 2], 6);
  });
});
