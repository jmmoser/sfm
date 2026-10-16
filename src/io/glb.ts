/**
 * glTF 2.0 binary (.glb) writer for meshes and point clouds.
 *
 * Layout decisions (compatibility first — three.js / <model-viewer> (which
 * converts to USDZ for iOS AR Quick Look via three's USDZExporter), Blender,
 * Babylon, Sketchfab, Windows 3D Viewer, gltf-validator):
 *  - Non-interleaved buffer views, every view 4-byte aligned, one per attribute.
 *  - POSITION: FLOAT VEC3 with exact accessor min/max (required by the spec).
 *  - NORMAL: FLOAT VEC3, re-normalised to unit length (validator requirement).
 *  - COLOR_0: FLOAT VEC3 by default. glTF defines COLOR_0 as LINEAR RGB, so the
 *    sRGB input colours are linearised with the exact sRGB EOTF. Float is the
 *    representation every loader supports and avoids the dark-tone banding
 *    that 8-bit *linear* storage causes. Smaller alternatives:
 *    'unorm16' (UNSIGNED_SHORT normalized VEC4, 8 B, banding-free) and
 *    'unorm8' (UNSIGNED_BYTE normalized VEC4, 4 B, slight banding in shadows).
 *  - Indices: UNSIGNED_INT SCALAR, mode 4 (TRIANGLES).
 *  - Points: mode 0 (POINTS) primitive with an unlit material
 *    (KHR_materials_unlit, listed in extensionsUsed only — viewers without it
 *    fall back to a rough, non-metallic white PBR material × vertex colour).
 *  - World: our world is Z-up; glTF is Y-up. With `yUp` (default true) vertices
 *    and normals are mapped (x, y, z) → (x, z, −y) (a proper rotation of −90°
 *    about X, so triangle winding is preserved).
 *  - Non-finite positions: such points are dropped; such mesh vertices are
 *    written at the origin (indices cannot be remapped cheaply).
 */
import type { PointCloud, TriangleMesh } from '../core/types';

export type GLBColorFormat = 'float' | 'unorm16' | 'unorm8';

export interface GLBInput {
  mesh?: TriangleMesh;
  points?: PointCloud;
  /** Convert from our Z-up world to glTF's Y-up. Default true. */
  yUp?: boolean;
  /** COLOR_0 storage, see file header. Default 'float'. */
  colorFormat?: GLBColorFormat;
  /** Render the mesh unlit (photo colours already contain shading). Default false. Points are always unlit. */
  unlitMesh?: boolean;
  /** Mesh material double-sided (open scans look broken otherwise). Default true. */
  doubleSided?: boolean;
  /** asset.generator. */
  generator?: string;
  /** Free-form metadata stored in asset.extras (must be JSON-serialisable). */
  extras?: Record<string, unknown>;
  /** Scene / node name prefix. */
  name?: string;
}

export const GLB_GENERATOR = 'sfm on-device photogrammetry (glb.ts)';

const GLB_MAGIC = 0x46546c67; // 'glTF'
const CHUNK_JSON = 0x4e4f534a; // 'JSON'
const CHUNK_BIN = 0x004e4942; // 'BIN\0'

const FLOAT = 5126;
const UNSIGNED_BYTE = 5121;
const UNSIGNED_SHORT = 5123;
const UNSIGNED_INT = 5125;
const ARRAY_BUFFER = 34962;
const ELEMENT_ARRAY_BUFFER = 34963;

let srgbToLinearLut: Float32Array | null = null;
function linearLut(): Float32Array {
  if (srgbToLinearLut) return srgbToLinearLut;
  const t = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    t[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  srgbToLinearLut = t;
  return t;
}

interface ViewPlan {
  byteOffset: number;
  byteLength: number;
  target: number;
  write: (dst: ArrayBuffer, byteOffset: number) => void;
}

/**
 * Pass over positions in output order/frame. With `out` = null computes
 * bounds = [minX, minY, minZ, maxX, maxY, maxZ]; otherwise writes into `out`.
 * The mapping is exact in float32 (permutation + negation; `0 - y` avoids −0,
 * which JSON cannot represent), so bounds equal the stored values exactly.
 */
function transformPositions(
  P: Float32Array, src: Uint32Array | null, count: number, yUp: boolean, zeroInvalid: boolean,
  out: Float32Array | null, bounds: Float64Array | null,
): void {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let k = 0; k < count; k++) {
    const i3 = 3 * (src ? src[k] : k);
    let x = P[i3], y = P[i3 + 1], z = P[i3 + 2];
    if (zeroInvalid && !(x - x === 0 && y - y === 0 && z - z === 0)) x = y = z = 0; // non-finite → origin
    let oy = y, oz = z;
    if (yUp) {
      oy = z;
      oz = 0 - y;
    }
    if (out) {
      out[3 * k] = x;
      out[3 * k + 1] = oy;
      out[3 * k + 2] = oz;
    } else {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (oy < minY) minY = oy;
      if (oy > maxY) maxY = oy;
      if (oz < minZ) minZ = oz;
      if (oz > maxZ) maxZ = oz;
    }
  }
  if (bounds) {
    bounds[0] = minX; bounds[1] = minY; bounds[2] = minZ;
    bounds[3] = maxX; bounds[4] = maxY; bounds[5] = maxZ;
  }
}

/** Encode a mesh and/or point cloud as a .glb ArrayBuffer. */
export function encodeGLB(input: GLBInput): ArrayBuffer {
  const yUp = input.yUp ?? true;
  const colorFormat = input.colorFormat ?? 'float';
  const name = input.name ?? 'reconstruction';
  const lut = linearLut();

  const views: ViewPlan[] = [];
  let binLength = 0;
  const bufferViews: Record<string, unknown>[] = [];
  const accessors: Record<string, unknown>[] = [];
  const meshes: Record<string, unknown>[] = [];
  const nodes: Record<string, unknown>[] = [];
  const materials: Record<string, unknown>[] = [];
  let usesUnlit = false;

  const addView = (byteLength: number, target: number, write: ViewPlan['write']): number => {
    const byteOffset = binLength;
    views.push({ byteOffset, byteLength, target, write });
    bufferViews.push({ buffer: 0, byteOffset, byteLength, target });
    binLength += (byteLength + 3) & ~3;
    return bufferViews.length - 1;
  };
  const addAccessor = (a: Record<string, unknown>): number => {
    accessors.push(a);
    return accessors.length - 1;
  };

  /** Writes positions (with optional index subset), returns accessor index. */
  const addPositions = (P: Float32Array, src: Uint32Array | null, count: number, zeroInvalid: boolean): number => {
    const bounds = new Float64Array(6);
    transformPositions(P, src, count, yUp, zeroInvalid, null, bounds);
    const min = [bounds[0], bounds[1], bounds[2]];
    const max = [bounds[3], bounds[4], bounds[5]];
    const view = addView(12 * count, ARRAY_BUFFER, (dst, off) => {
      transformPositions(P, src, count, yUp, zeroInvalid, new Float32Array(dst, off, 3 * count), null);
    });
    return addAccessor({ bufferView: view, componentType: FLOAT, count, type: 'VEC3', min, max });
  };

  const addNormals = (N: Float32Array, src: Uint32Array | null, count: number): number => {
    const view = addView(12 * count, ARRAY_BUFFER, (dst, off) => {
      const out = new Float32Array(dst, off, 3 * count);
      for (let k = 0; k < count; k++) {
        const i = src ? src[k] : k;
        let x = N[3 * i], y = N[3 * i + 1], z = N[3 * i + 2];
        const l = Math.sqrt(x * x + y * y + z * z);
        if (l > 1e-20 && l < Infinity) {
          x /= l; y /= l; z /= l;
        } else {
          x = 0; y = 0; z = 1;
        }
        if (yUp) {
          out[3 * k] = x; out[3 * k + 1] = z; out[3 * k + 2] = 0 - y;
        } else {
          out[3 * k] = x; out[3 * k + 1] = y; out[3 * k + 2] = z;
        }
      }
    });
    return addAccessor({ bufferView: view, componentType: FLOAT, count, type: 'VEC3' });
  };

  const addColors = (C: Uint8Array, src: Uint32Array | null, count: number): number => {
    if (colorFormat === 'float') {
      const view = addView(12 * count, ARRAY_BUFFER, (dst, off) => {
        const out = new Float32Array(dst, off, 3 * count);
        for (let k = 0; k < count; k++) {
          const i = 3 * (src ? src[k] : k);
          out[3 * k] = lut[C[i]];
          out[3 * k + 1] = lut[C[i + 1]];
          out[3 * k + 2] = lut[C[i + 2]];
        }
      });
      return addAccessor({ bufferView: view, componentType: FLOAT, count, type: 'VEC3' });
    }
    if (colorFormat === 'unorm16') {
      const q = new Uint16Array(256);
      for (let i = 0; i < 256; i++) q[i] = Math.round(lut[i] * 65535);
      const view = addView(8 * count, ARRAY_BUFFER, (dst, off) => {
        const out = new Uint16Array(dst, off, 4 * count);
        for (let k = 0; k < count; k++) {
          const i = 3 * (src ? src[k] : k);
          out[4 * k] = q[C[i]];
          out[4 * k + 1] = q[C[i + 1]];
          out[4 * k + 2] = q[C[i + 2]];
          out[4 * k + 3] = 65535;
        }
      });
      return addAccessor({ bufferView: view, componentType: UNSIGNED_SHORT, normalized: true, count, type: 'VEC4' });
    }
    const q = new Uint8Array(256);
    for (let i = 0; i < 256; i++) q[i] = Math.round(lut[i] * 255);
    const view = addView(4 * count, ARRAY_BUFFER, (dst, off) => {
      const out = new Uint8Array(dst, off, 4 * count);
      for (let k = 0; k < count; k++) {
        const i = 3 * (src ? src[k] : k);
        out[4 * k] = q[C[i]];
        out[4 * k + 1] = q[C[i + 1]];
        out[4 * k + 2] = q[C[i + 2]];
        out[4 * k + 3] = 255;
      }
    });
    return addAccessor({ bufferView: view, componentType: UNSIGNED_BYTE, normalized: true, count, type: 'VEC4' });
  };

  // ---- mesh ---------------------------------------------------------------
  const mesh = input.mesh;
  if (mesh && mesh.vertexCount > 0 && mesh.triangleCount > 0) {
    const nv = mesh.vertexCount;
    const nt = mesh.triangleCount;
    if (mesh.positions.length < 3 * nv || mesh.normals.length < 3 * nv || mesh.colors.length < 3 * nv || mesh.indices.length < 3 * nt) {
      throw new RangeError('GLB: mesh arrays shorter than counts');
    }
    const I = mesh.indices;
    for (let i = 0; i < 3 * nt; i++) if (I[i] >= nv) throw new RangeError(`GLB: index ${I[i]} out of range (${nv} vertices)`);
    const pos = addPositions(mesh.positions, null, nv, true);
    const nrm = addNormals(mesh.normals, null, nv);
    const col = addColors(mesh.colors, null, nv);
    const idxView = addView(12 * nt, ELEMENT_ARRAY_BUFFER, (dst, off) => {
      new Uint32Array(dst, off, 3 * nt).set(I.subarray(0, 3 * nt));
    });
    const idx = addAccessor({ bufferView: idxView, componentType: UNSIGNED_INT, count: 3 * nt, type: 'SCALAR' });
    const mat: Record<string, unknown> = {
      name: `${name}-mesh`,
      pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1 },
      doubleSided: input.doubleSided ?? true,
    };
    if (input.unlitMesh) {
      mat.extensions = { KHR_materials_unlit: {} };
      usesUnlit = true;
    }
    materials.push(mat);
    meshes.push({
      name: `${name}-mesh`,
      primitives: [{ attributes: { POSITION: pos, NORMAL: nrm, COLOR_0: col }, indices: idx, mode: 4, material: materials.length - 1 }],
    });
    nodes.push({ name: `${name}-mesh`, mesh: meshes.length - 1 });
  }

  // ---- points -------------------------------------------------------------
  const pc = input.points;
  if (pc && pc.count > 0) {
    const n = pc.count;
    if (pc.positions.length < 3 * n || pc.colors.length < 3 * n || (pc.normals && pc.normals.length < 3 * n)) {
      throw new RangeError('GLB: point cloud arrays shorter than count');
    }
    const P = pc.positions;
    let valid = 0;
    for (let i = 0; i < n; i++) {
      if (Number.isFinite(P[3 * i]) && Number.isFinite(P[3 * i + 1]) && Number.isFinite(P[3 * i + 2])) valid++;
    }
    let src: Uint32Array | null = null;
    if (valid < n) {
      src = new Uint32Array(valid);
      let k = 0;
      for (let i = 0; i < n; i++) {
        if (Number.isFinite(P[3 * i]) && Number.isFinite(P[3 * i + 1]) && Number.isFinite(P[3 * i + 2])) src[k++] = i;
      }
    }
    if (valid > 0) {
      const attributes: Record<string, number> = {
        POSITION: addPositions(P, src, valid, false),
      };
      if (pc.normals) attributes.NORMAL = addNormals(pc.normals, src, valid);
      attributes.COLOR_0 = addColors(pc.colors, src, valid);
      materials.push({
        name: `${name}-points`,
        pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1 },
        extensions: { KHR_materials_unlit: {} },
      });
      usesUnlit = true;
      meshes.push({ name: `${name}-points`, primitives: [{ attributes, mode: 0, material: materials.length - 1 }] });
      nodes.push({ name: `${name}-points`, mesh: meshes.length - 1 });
    }
  }

  // ---- JSON ---------------------------------------------------------------
  const asset: Record<string, unknown> = { version: '2.0', generator: input.generator ?? GLB_GENERATOR };
  if (input.extras) asset.extras = input.extras;
  const gltf: Record<string, unknown> = {
    asset,
    scene: 0,
    scenes: [{ name, nodes: nodes.map((_, i) => i) }],
  };
  if (nodes.length) gltf.nodes = nodes;
  if (meshes.length) gltf.meshes = meshes;
  if (materials.length) gltf.materials = materials;
  if (accessors.length) gltf.accessors = accessors;
  if (bufferViews.length) gltf.bufferViews = bufferViews;
  if (binLength > 0) gltf.buffers = [{ byteLength: binLength }];
  if (usesUnlit) gltf.extensionsUsed = ['KHR_materials_unlit'];

  const jsonBytes = new TextEncoder().encode(JSON.stringify(gltf));
  const jsonPadded = (jsonBytes.length + 3) & ~3;
  const binChunk = binLength > 0 ? 8 + binLength : 0; // binLength is already a multiple of 4
  const total = 12 + 8 + jsonPadded + binChunk;
  if (total > 0xffffffff) throw new RangeError('GLB: output exceeds 4 GiB');

  const out = new ArrayBuffer(total);
  const dv = new DataView(out);
  const u8 = new Uint8Array(out);
  dv.setUint32(0, GLB_MAGIC, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, jsonPadded, true);
  dv.setUint32(16, CHUNK_JSON, true);
  u8.set(jsonBytes, 20);
  u8.fill(0x20, 20 + jsonBytes.length, 20 + jsonPadded);
  if (binChunk) {
    const binStart = 20 + jsonPadded;
    dv.setUint32(binStart, binLength, true);
    dv.setUint32(binStart + 4, CHUNK_BIN, true);
    const dataStart = binStart + 8; // multiple of 4 → typed views are aligned
    for (const v of views) v.write(out, dataStart + v.byteOffset);
    // Padding bytes are already zero (fresh ArrayBuffer).
  }
  return out;
}

/** Parsed GLB container (for validation, tests and re-import). */
export interface ParsedGLB {
  json: Record<string, unknown> & {
    asset: { version: string; generator?: string };
    accessors?: Array<Record<string, unknown> & { bufferView?: number; componentType: number; count: number; type: string; byteOffset?: number; min?: number[]; max?: number[]; normalized?: boolean }>;
    bufferViews?: Array<{ buffer: number; byteOffset?: number; byteLength: number; byteStride?: number; target?: number }>;
    buffers?: Array<{ byteLength: number; uri?: string }>;
    meshes?: Array<{ name?: string; primitives: Array<{ attributes: Record<string, number>; indices?: number; mode?: number; material?: number }> }>;
    nodes?: Array<Record<string, unknown>>;
    materials?: Array<Record<string, unknown>>;
  };
  /** BIN chunk payload (view into the input), or null. */
  bin: Uint8Array | null;
}

/** Parse and structurally check a GLB container (magic, version, lengths, chunk order). */
export function parseGLB(input: ArrayBuffer | Uint8Array): ParsedGLB {
  const u8 = input instanceof Uint8Array ? input : new Uint8Array(input);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (u8.byteLength < 20) throw new Error('GLB: too short');
  if (dv.getUint32(0, true) !== GLB_MAGIC) throw new Error('GLB: bad magic');
  if (dv.getUint32(4, true) !== 2) throw new Error('GLB: unsupported version');
  const length = dv.getUint32(8, true);
  if (length !== u8.byteLength) throw new Error(`GLB: header length ${length} ≠ ${u8.byteLength}`);
  const jsonLen = dv.getUint32(12, true);
  if (dv.getUint32(16, true) !== CHUNK_JSON) throw new Error('GLB: first chunk is not JSON');
  if (20 + jsonLen > length) throw new Error('GLB: JSON chunk overruns file');
  const json = JSON.parse(new TextDecoder().decode(u8.subarray(20, 20 + jsonLen))) as ParsedGLB['json'];
  let bin: Uint8Array | null = null;
  const binStart = 20 + jsonLen;
  if (binStart + 8 <= length) {
    const binLen = dv.getUint32(binStart, true);
    if (dv.getUint32(binStart + 4, true) !== CHUNK_BIN) throw new Error('GLB: second chunk is not BIN');
    if (binStart + 8 + binLen > length) throw new Error('GLB: BIN chunk overruns file');
    bin = u8.subarray(binStart + 8, binStart + 8 + binLen);
  }
  return { json, bin };
}
