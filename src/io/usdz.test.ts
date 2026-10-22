/**
 * USDZ writer (src/io/usdz.ts): package layout (STORE, 64-byte aligned data, layer first), the
 * USDA's stage metadata, anchoring and material, mesh arrays and the Y-up / pivot mapping, and
 * the interim AR pivot.
 */
import { describe, expect, it } from 'vitest';
import type { TriangleMesh } from '../core/types';
import { Rng } from '../math/random';
import { buildUSDA, encodeUSDZ, selectArPivot, usdString, USDZ_LAYER_NAME } from './usdz';
import { readZip } from './zip';

/** Numbers of the array attribute `decl` (e.g. 'point3f[] points'), tuples flattened. */
function usdArray(usda: string, decl: string): number[] {
  const start = usda.indexOf(`${decl} = [`);
  if (start < 0) throw new Error(`no ${decl}`);
  const open = start + decl.length + 4;
  const end = usda.indexOf(']', open);
  const body = usda.slice(open, end);
  return body.length ? body.replace(/[()]/g, '').split(',').map(Number) : [];
}

/** A small closed mesh (a tetrahedron) with distinct coordinates, plus per-corner UVs. */
function tetra(): { mesh: TriangleMesh; uvs: Float32Array } {
  const positions = new Float32Array([0.1, 0.2, 0.3, 1.1, 0.2, 0.3, 0.1, 1.7, 0.3, 0.1, 0.2, 2.3]);
  const normals = new Float32Array([0, 0, -1, 1, 0, 0, 0, 1, 0, 0, 0, 2]);
  const mesh: TriangleMesh = {
    vertexCount: 4,
    triangleCount: 4,
    positions,
    normals,
    colors: new Uint8Array(12),
    indices: new Uint32Array([0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2]),
  };
  const uvs = Float32Array.from({ length: 24 }, (_, i) => ((i * 7) % 24) / 23);
  return { mesh, uvs };
}

describe('encodeUSDZ', () => {
  it('packages the layer first, STORE only, every data offset a multiple of 64', () => {
    const { mesh, uvs } = tetra();
    const texture = Uint8Array.from({ length: 1001 }, (_, i) => i & 255);
    const usdz = encodeUSDZ({ mesh, uvs, texture });
    // readZip accepts STORE entries only and returns zero-copy views: their byteOffset is the data offset
    const entries = readZip(usdz);
    expect(entries.map((e) => e.name)).toEqual([USDZ_LAYER_NAME, 'textures/albedo.jpg']);
    for (const e of entries) expect((e.data.byteOffset - usdz.byteOffset) % 64).toBe(0);
    expect(entries[1].data).toEqual(texture);
    const usda = new TextDecoder().decode(entries[0].data);
    expect(usda).toContain('@textures/albedo.jpg@');
    const png = readZip(encodeUSDZ({ mesh, uvs, texture, textureType: 'image/png' }));
    expect(png[1].name).toBe('textures/albedo.png');
  });

  it('writes Y-up metres, the anchoring tokens and a textured UsdPreviewSurface', () => {
    const { mesh, uvs } = tetra();
    const usda = buildUSDA({ mesh, uvs, name: 'My "mug"\nscan' }, 'textures/albedo.jpg');
    expect(usda.startsWith('#usda 1.0\n')).toBe(true);
    expect(usda).toContain('upAxis = "Y"');
    expect(usda).toContain('metersPerUnit = 1\n');
    expect(usda).toContain('defaultPrim = "Root"');
    expect(usda).toContain('bool metricScale = 1');
    expect(usda).toContain('uniform token preliminary:anchoring:type = "plane"');
    expect(usda).toContain('uniform token preliminary:planeAnchoring:alignment = "horizontal"');
    expect(usda).toContain('kind = "component"');
    expect(usda).toContain('string name = "My \\"mug\\"\\nscan"');
    expect(usda).toContain('uniform token info:id = "UsdPreviewSurface"');
    expect(usda).toContain('float inputs:roughness = 1');
    expect(usda).toContain('float inputs:metallic = 0');
    expect(usda).toContain('color3f inputs:diffuseColor.connect = </Root/Materials/Surface/Albedo.outputs:rgb>');
    expect(usda).toContain('uniform token info:id = "UsdUVTexture"');
    expect(usda).toContain('token inputs:sourceColorSpace = "sRGB"');
    expect(usda).toContain('string inputs:varname = "st"');
    expect(usda).toContain('rel material:binding = </Root/Materials/Surface>');
    expect(usda).toContain('interpolation = "faceVarying"');
    expect(usda).toContain('uniform bool doubleSided = 1');
    expect(usda).not.toContain('emissiveColor');
    // balanced braces / brackets / parentheses (a cheap syntax sanity check)
    for (const [o, c] of ['{}', '[]', '()']) expect(usda.split(o).length).toBe(usda.split(c).length);

    const em = buildUSDA({ mesh, uvs, material: 'emissive', metric: false, doubleSided: false }, 't.jpg');
    expect(em).toContain('color3f inputs:diffuseColor = (0, 0, 0)');
    expect(em).toContain('color3f inputs:emissiveColor.connect = </Root/Materials/Surface/Albedo.outputs:rgb>');
    expect(em).toContain('bool metricScale = 0');
    expect(em).toContain('uniform bool doubleSided = 0');
  });

  it('mesh arrays: counts, indices, face-varying st in [0, 1] with st = (u, 1 − v), (x, y, z) → (x, z, −y) about the pivot', () => {
    const { mesh, uvs } = tetra();
    const pivot = [0.5, 0.25, 0.3];
    const usda = buildUSDA({ mesh, uvs, pivot }, 'textures/albedo.jpg');
    const counts = usdArray(usda, 'int[] faceVertexCounts');
    const idx = usdArray(usda, 'int[] faceVertexIndices');
    const pts = usdArray(usda, 'point3f[] points');
    const nrm = usdArray(usda, 'normal3f[] normals');
    const st = usdArray(usda, 'texCoord2f[] primvars:st');
    const extent = usdArray(usda, 'float3[] extent');
    expect(counts.length).toBe(mesh.triangleCount);
    expect(counts.every((c) => c === 3)).toBe(true);
    expect(idx).toEqual([...mesh.indices]);
    expect(idx.every((i) => i >= 0 && i < pts.length / 3)).toBe(true);
    expect(st.length).toBe(2 * 3 * mesh.triangleCount);
    for (let k = 0; k < 3 * mesh.triangleCount; k++) {
      expect(st[2 * k]).toBeGreaterThanOrEqual(0);
      expect(st[2 * k]).toBeLessThanOrEqual(1);
      expect(st[2 * k]).toBeCloseTo(uvs[2 * k], 4);
      expect(st[2 * k + 1]).toBeCloseTo(1 - uvs[2 * k + 1], 4);
    }
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < mesh.vertexCount; i++) {
      const [x, y, z] = mesh.positions.subarray(3 * i, 3 * i + 3);
      const e = [x - pivot[0], z - pivot[2], -(y - pivot[1])];
      for (let a = 0; a < 3; a++) {
        expect(pts[3 * i + a]).toBeCloseTo(e[a], 5);
        lo[a] = Math.min(lo[a], e[a]);
        hi[a] = Math.max(hi[a], e[a]);
      }
      const [nx, ny, nz] = mesh.normals.subarray(3 * i, 3 * i + 3);
      const l = Math.hypot(nx, ny, nz);
      expect(nrm.slice(3 * i, 3 * i + 3).map((v) => +v.toFixed(3))).toEqual([nx / l, nz / l, -ny / l].map((v) => +v.toFixed(3)));
    }
    for (let a = 0; a < 3; a++) {
      expect(extent[a]).toBeCloseTo(lo[a], 6);
      expect(extent[3 + a]).toBeCloseTo(hi[a], 6);
    }
  });

  it('keeps float precision relative to the model size and survives non-finite input', () => {
    const { mesh, uvs } = tetra();
    // a 100 m scene: 1e-4 m resolution is enough; a 2 cm object gets 1e-8 m
    const big = { ...mesh, positions: mesh.positions.map((v) => v * 50) };
    const usdaBig = buildUSDA({ mesh: big, uvs }, 't.jpg');
    const ptsBig = usdArray(usdaBig, 'point3f[] points');
    for (let i = 0; i < 4; i++) {
      const [x, y, z] = big.positions.subarray(3 * i, 3 * i + 3);
      expect(Math.abs(ptsBig[3 * i] - x)).toBeLessThan(1e-4);
      expect(Math.abs(ptsBig[3 * i + 1] - z)).toBeLessThan(1e-4);
      expect(Math.abs(ptsBig[3 * i + 2] + y)).toBeLessThan(1e-4);
    }
    expect(usdaBig).toMatch(/point3f\[\] points = \[\(5\.0000,/);
    const small = { ...mesh, positions: mesh.positions.map((v) => v * 0.01) };
    const ptsSmall = usdArray(buildUSDA({ mesh: small, uvs }, 't.jpg'), 'point3f[] points');
    expect(ptsSmall[0]).toBeCloseTo(small.positions[0], 8);
    const bad = { ...mesh, positions: mesh.positions.slice() };
    bad.positions[3] = NaN;
    const badUv = uvs.slice();
    badUv[0] = Infinity;
    const usda = buildUSDA({ mesh: bad, uvs: badUv }, 't.jpg');
    expect(usda).not.toMatch(/NaN|Infinity/);
    expect(usdArray(usda, 'point3f[] points').slice(3, 6)).toEqual([0, 0, 0]);
  });

  it('rejects inconsistent input', () => {
    const { mesh, uvs } = tetra();
    expect(() => buildUSDA({ mesh, uvs: uvs.subarray(0, 18) }, 't.jpg')).toThrow(/UV/);
    expect(() => buildUSDA({ mesh: { ...mesh, indices: new Uint32Array([0, 1, 9, 0, 1, 2, 0, 1, 2, 0, 1, 2]) }, uvs }, 't.jpg')).toThrow(/out of range/);
    expect(() => buildUSDA({ mesh: { ...mesh, triangleCount: 0 }, uvs }, 't.jpg')).toThrow(/empty/);
  });

  it('usdString escapes quotes, backslashes and newlines', () => {
    expect(usdString('a"b\\c\nd\u0001')).toBe('"a\\"b\\\\c\\nd"');
  });
});

describe('selectArPivot', () => {
  /** An object (z 0.02…0.3 over |x|, |y| < 0.1 around (2, 3)) on a table patch at z = 0 with stray fragments. */
  function objectOnTable(): TriangleMesh {
    const rng = new Rng(4);
    const pts: number[] = [];
    for (let i = 0; i < 4000; i++) pts.push(2 + 0.6 * (rng.next() - 0.5), 3 + 0.6 * (rng.next() - 0.5), 0.002 * rng.gaussian());
    for (let i = 0; i < 3000; i++) pts.push(2 + 0.2 * (rng.next() - 0.5), 3 + 0.2 * (rng.next() - 0.5), 0.02 + 0.28 * rng.next());
    // fragments far away (below the table, to the side): must not move the pivot
    for (let i = 0; i < 40; i++) pts.push(9 + rng.next(), -7, -1.5);
    const n = pts.length / 3;
    return { vertexCount: n, triangleCount: 0, positions: Float32Array.from(pts), normals: new Float32Array(3 * n), colors: new Uint8Array(3 * n), indices: new Uint32Array(0) };
  }

  it('interim: centre of the robust footprint at the 2nd-percentile height under it', () => {
    const p = selectArPivot(objectOnTable());
    expect(p[0]).toBeCloseTo(2, 1);
    expect(p[1]).toBeCloseTo(3, 1);
    expect(Math.abs(p[2])).toBeLessThan(0.01);
  });

  it('uses the object ROI (centre on its support plane) when given one', () => {
    expect([...selectArPivot(objectOnTable(), { centre: [2.01, 2.99, 0.15], planeZ: 0.004 })]).toEqual([2.01, 2.99, 0.004]);
    // an unusable ROI falls back to the interim rule
    expect(selectArPivot(objectOnTable(), { centre: [NaN, 0, 0], planeZ: 0 })[0]).toBeCloseTo(2, 1);
    expect([...selectArPivot({ positions: new Float32Array(0), vertexCount: 0 })]).toEqual([0, 0, 0]);
  });
});
