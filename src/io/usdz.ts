/**
 * USDZ writer for AR Quick Look (iOS / iPadOS / macOS): one textured mesh, metric and Y-up.
 *
 *  - Package: an uncompressed ZIP whose entries' data start at multiples of 64 bytes (zip.ts
 *    `align`), the USD layer first (`model.usda`), then `textures/albedo.jpg` (or .png). This is
 *    what the USDZ spec requires (no compression, no encryption, 64-byte alignment).
 *  - Layer: ASCII USDA (a binary crate writer would be far more code; what makes the file big
 *    is the per-corner st, see below).
 *    `upAxis = "Y"`, `metersPerUnit = 1`; our world is Z-up in metres when the scan is metric,
 *    and points / normals are mapped (x, y, z) → (x, z, −y) like glb.ts (a proper rotation, so
 *    the winding — counter-clockwise seen from outside, USD's default rightHanded orientation —
 *    is kept). A non-metric scan is written the same way (arbitrary units read as metres:
 *    customLayerData `metricScale = 0` records it).
 *  - Origin = `pivot` (world frame), e.g. the base centre from `selectArPivot`: AR Quick Look
 *    puts the model's origin on the detected plane.
 *  - Prims: /Root (Xform, kind component, default prim) with the AR anchoring tokens
 *    `preliminary:anchoring:type = "plane"` and `preliminary:planeAnchoring:alignment =
 *    "horizontal"` (Preliminary_AnchoringAPI), /Root/Model (Mesh: points, vertex normals,
 *    faceVertexCounts / Indices, face-varying `primvars:st`, extent, subdivisionScheme "none",
 *    doubleSided) bound to /Root/Materials/Surface: a UsdPreviewSurface (roughness 1,
 *    metallic 0) whose diffuseColor — or, with `material: 'emissive'`, emissiveColor over a
 *    black diffuse (a flat, unlit look; which one reads better in AR Quick Look's lighting is a
 *    device question) — comes from a UsdUVTexture (sRGB, clamp) read through a
 *    UsdPrimvarReader_float2 of `st`. RealityKit / AR Quick Look ignore vertex colours and
 *    displayColor (Apple, developer forums 691674), hence the texture. three.js's USDZExporter
 *    warns that AR Quick Look ignores double-sided materials; `doubleSided` is written anyway
 *    (standard USD, harmless) until the device check (docs/research/ios-platform.md §8).
 *  - Checked with Pixar USD 26.11 (usd-core) UsdValidation, all 30 validators incl.
 *    UsdzPackageValidator (alignment, no compression) and ShaderSdrCompliance: 0 errors for
 *    both materials (2026-10-21); an unaligned repack of the same file fails ByteMisalignment.
 *  - UVs: 6 floats per triangle in the IMAGE convention ((0, 0) = top-left of the texture, v
 *    down — src/dense/colour-atlas.ts); USD's st has its origin at the BOTTOM-left, so st =
 *    (u, 1 − v).
 *  - Numbers: points with 1e-6 of the model extent resolution (float32-level), normals to 1e-3,
 *    st to 1e-4 (≤ 0.1 texel at 2048², inside the atlas' 1-texel gutter). The text is what the
 *    file size is: a 100 k-triangle mesh is ≈ 9.3 MB (5 MB of it the 300 k face-varying st).
 *  - Non-finite positions are written at the origin, non-finite normals as +Y, UVs as 0.
 *
 * Pure (Node-testable): the texture arrives encoded (the engine encodes JPEG with
 * OffscreenCanvas.convertToBlob in its worker).
 */
import type { TriangleMesh } from '../core/types';
import { percentile } from '../math/stats';
import { createZip } from './zip';

export const USDZ_GENERATOR = 'sfm on-device photogrammetry (usdz.ts)';

export interface USDZInput {
  /** Triangle mesh in our world frame (Z-up). Its vertex colours are not written (AR Quick Look ignores them). */
  mesh: TriangleMesh;
  /** Face-varying UVs: 6 per triangle, image convention (see the file header). */
  uvs: Float32Array;
  /** Encoded texture (JPEG or PNG bytes). */
  texture: Uint8Array;
  /** Default 'image/jpeg'. */
  textureType?: 'image/jpeg' | 'image/png';
  /** World point that becomes the USD origin (AR Quick Look sets it on the plane). Default (0, 0, 0). */
  pivot?: ArrayLike<number>;
  /** The scan is in metres (recorded in customLayerData; the file says metres either way). Default true. */
  metric?: boolean;
  /** Model name (assetInfo; free text). */
  name?: string;
  /** 'diffuse' (lit by AR Quick Look, default) or 'emissive' (texture as emission over black diffuse). */
  material?: 'diffuse' | 'emissive';
  /** Default true (open scans would otherwise show holes from behind). */
  doubleSided?: boolean;
  generator?: string;
}

/** USD layer name inside the package (the first entry). */
export const USDZ_LAYER_NAME = 'model.usda';

/** Encode `input` as a .usdz package (Uint8Array, STORE ZIP with 64-byte aligned data). */
export function encodeUSDZ(input: USDZInput): Uint8Array {
  const texName = `textures/albedo.${input.textureType === 'image/png' ? 'png' : 'jpg'}`;
  const usda = buildUSDA(input, texName);
  const when = new Date();
  return createZip(
    [
      { name: USDZ_LAYER_NAME, data: usda, lastModified: when },
      { name: texName, data: input.texture, lastModified: when },
    ],
    { align: 64 },
  );
}

/** The USDA text of `input` referencing the texture at `texturePath` (package-relative). */
export function buildUSDA(input: Omit<USDZInput, 'texture'>, texturePath: string): string {
  const { mesh, uvs } = input;
  const nv = mesh.vertexCount;
  const nt = mesh.triangleCount;
  if (!(nv > 0 && nt > 0)) throw new RangeError('USDZ: empty mesh');
  if (mesh.positions.length < 3 * nv || mesh.normals.length < 3 * nv || mesh.indices.length < 3 * nt) {
    throw new RangeError('USDZ: mesh arrays shorter than counts');
  }
  if (uvs.length < 6 * nt) throw new RangeError(`USDZ: ${uvs.length} UV floats for ${nt} triangles (need 6 per triangle)`);
  const I = mesh.indices;
  for (let i = 0; i < 3 * nt; i++) if (I[i] >= nv) throw new RangeError(`USDZ: index ${I[i]} out of range (${nv} vertices)`);
  const px = input.pivot?.[0] ?? 0;
  const py = input.pivot?.[1] ?? 0;
  const pz = input.pivot?.[2] ?? 0;

  // ---- points in the output frame (Y-up, origin at the pivot), extent, precision
  const P = mesh.positions;
  const out = new Float64Array(3 * nv);
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < nv; i++) {
    let x = P[3 * i] - px;
    let y = P[3 * i + 2] - pz; // up
    let z = py - P[3 * i + 1]; // −y
    if (!(x - x === 0 && y - y === 0 && z - z === 0)) x = y = z = 0;
    out[3 * i] = x;
    out[3 * i + 1] = y;
    out[3 * i + 2] = z;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, Math.abs(minX), Math.abs(maxX), Math.abs(minY), Math.abs(maxY), Math.abs(minZ), Math.abs(maxZ));
  const pd = span > 0 ? Math.min(9, Math.max(0, Math.ceil(-Math.log10(span * 1e-6)))) : 6;
  const f = (v: number): string => v.toFixed(pd);

  const parts: string[] = [];
  const name = input.name ?? 'Scan';
  parts.push(
    '#usda 1.0\n(\n',
    '    customLayerData = {\n',
    `        string creator = ${usdString(input.generator ?? USDZ_GENERATOR)}\n`,
    `        bool metricScale = ${input.metric === false ? 0 : 1}\n`,
    '    }\n',
    '    defaultPrim = "Root"\n',
    '    metersPerUnit = 1\n',
    '    upAxis = "Y"\n',
    ')\n\n',
    'def Xform "Root" (\n',
    '    prepend apiSchemas = ["Preliminary_AnchoringAPI"]\n',
    `    assetInfo = {\n        string name = ${usdString(name)}\n    }\n`,
    '    kind = "component"\n',
    ')\n{\n',
    '    uniform token preliminary:anchoring:type = "plane"\n',
    '    uniform token preliminary:planeAnchoring:alignment = "horizontal"\n\n',
    '    def Mesh "Model" (\n        prepend apiSchemas = ["MaterialBindingAPI"]\n    )\n    {\n',
    `        uniform bool doubleSided = ${input.doubleSided === false ? 0 : 1}\n`,
    `        float3[] extent = [(${f(minX)}, ${f(minY)}, ${f(minZ)}), (${f(maxX)}, ${f(maxY)}, ${f(maxZ)})]\n`,
  );

  // faceVertexCounts: all 3
  parts.push('        int[] faceVertexCounts = [', '3,'.repeat(nt - 1), '3]\n');
  // faceVertexIndices
  {
    const s = new Array<string>(3 * nt);
    for (let i = 0; i < 3 * nt; i++) s[i] = String(I[i]);
    parts.push('        int[] faceVertexIndices = [', s.join(','), ']\n');
  }
  parts.push('        rel material:binding = </Root/Materials/Surface>\n');
  // normals (vertex), mapped like the points
  {
    const N = mesh.normals;
    const s = new Array<string>(nv);
    for (let i = 0; i < nv; i++) {
      let x = N[3 * i];
      let y = N[3 * i + 2];
      let z = -N[3 * i + 1];
      const l = Math.sqrt(x * x + y * y + z * z);
      if (l > 1e-20 && l < Infinity) {
        x /= l;
        y /= l;
        z /= l;
      } else {
        x = 0;
        y = 1;
        z = 0;
      }
      s[i] = `(${x.toFixed(3)},${y.toFixed(3)},${z.toFixed(3)})`;
    }
    parts.push('        normal3f[] normals = [', s.join(','), '] (\n            interpolation = "vertex"\n        )\n');
  }
  // points
  {
    const s = new Array<string>(nv);
    for (let i = 0; i < nv; i++) s[i] = `(${f(out[3 * i])},${f(out[3 * i + 1])},${f(out[3 * i + 2])})`;
    parts.push('        point3f[] points = [', s.join(','), ']\n');
  }
  // st (face-varying, one per face corner in faceVertexIndices order)
  {
    const s = new Array<string>(3 * nt);
    for (let k = 0; k < 3 * nt; k++) {
      let u = uvs[2 * k];
      let v = uvs[2 * k + 1];
      if (!(u - u === 0)) u = 0;
      if (!(v - v === 0)) v = 0;
      s[k] = `(${u.toFixed(4)},${(1 - v).toFixed(4)})`;
    }
    parts.push('        texCoord2f[] primvars:st = [', s.join(','), '] (\n            interpolation = "faceVarying"\n        )\n');
  }
  parts.push('        uniform token subdivisionScheme = "none"\n    }\n\n');

  // ---- material
  const emissive = input.material === 'emissive';
  const M = '/Root/Materials/Surface';
  parts.push(
    '    def Scope "Materials"\n    {\n',
    '        def Material "Surface"\n        {\n',
    `            token outputs:surface.connect = <${M}/PreviewSurface.outputs:surface>\n\n`,
    '            def Shader "PreviewSurface"\n            {\n',
    '                uniform token info:id = "UsdPreviewSurface"\n',
    emissive
      ? `                color3f inputs:diffuseColor = (0, 0, 0)\n                color3f inputs:emissiveColor.connect = <${M}/Albedo.outputs:rgb>\n`
      : `                color3f inputs:diffuseColor.connect = <${M}/Albedo.outputs:rgb>\n`,
    '                float inputs:metallic = 0\n',
    '                float inputs:roughness = 1\n',
    '                token outputs:surface\n',
    '            }\n\n',
    '            def Shader "TexCoords"\n            {\n',
    '                uniform token info:id = "UsdPrimvarReader_float2"\n',
    '                float2 inputs:fallback = (0, 0)\n',
    '                string inputs:varname = "st"\n',
    '                float2 outputs:result\n',
    '            }\n\n',
    '            def Shader "Albedo"\n            {\n',
    '                uniform token info:id = "UsdUVTexture"\n',
    `                asset inputs:file = @${texturePath}@\n`,
    `                float2 inputs:st.connect = <${M}/TexCoords.outputs:result>\n`,
    '                token inputs:sourceColorSpace = "sRGB"\n',
    '                token inputs:wrapS = "clamp"\n',
    '                token inputs:wrapT = "clamp"\n',
    '                float3 outputs:rgb\n',
    '            }\n',
    '        }\n    }\n}\n',
  );
  return parts.join('');
}

/** A USDA string literal (double-quoted, backslash escapes; control characters dropped). */
export function usdString(s: string): string {
  // eslint-disable-next-line no-control-regex
  return `"${s.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n').replace(/[\u0000-\u001f\u007f]/g, '')}"`;
}

// ---------------------------------------------------------------------------------------------
// AR pivot
// ---------------------------------------------------------------------------------------------

/**
 * The parts of an object ROI the AR pivot uses (the `objectRoi` of the mesh result from
 * src/dense/object-roi.ts, once it lands). Pass it only when the ROI is trusted (its confidence
 * is the caller's call).
 */
export interface ArPivotRoi {
  /** Object centre (world frame; x and y are used). */
  centre: ArrayLike<number>;
  /** World z of the support plane the object stands on. */
  planeZ: number;
}

/**
 * >>> AR PIVOT SELECTION — the one place to switch to the object ROI. <<<
 *
 * The world point AR Quick Look puts on the detected floor / table (the USDZ origin):
 *  - with an object ROI: its centre on its support plane, (centre.x, centre.y, planeZ);
 *  - otherwise (interim): the centre of the mesh's robust horizontal footprint (2nd–98th
 *    percentile of x and y, so stray fragments do not move it) at the 2nd-percentile z of the
 *    vertices under that footprint — the base of the object, or the table / ground patch it was
 *    scanned on (an object scan includes that patch until the ROI crops it).
 * Use the undecimated mesh: its vertex density is about uniform (TSDF), so vertex percentiles
 * are area percentiles; after QEM the flat base has few vertices. Assumes a gravity-aligned
 * world (+Z up); without it the result is just a corner-ish point of the model.
 */
export function selectArPivot(mesh: Pick<TriangleMesh, 'positions' | 'vertexCount'>, roi?: ArPivotRoi | null): Float64Array {
  if (roi && Number.isFinite(roi.planeZ) && Number.isFinite(roi.centre[0]) && Number.isFinite(roi.centre[1])) {
    return Float64Array.of(roi.centre[0], roi.centre[1], roi.planeZ);
  }
  const P = mesh.positions;
  const n = mesh.vertexCount;
  const step = Math.max(1, Math.ceil(n / 200_000));
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  for (let i = 0; i < n; i += step) {
    const x = P[3 * i];
    const y = P[3 * i + 1];
    const z = P[3 * i + 2];
    if (!(x - x === 0 && y - y === 0 && z - z === 0)) continue;
    xs.push(x);
    ys.push(y);
    zs.push(z);
  }
  if (!xs.length) return new Float64Array(3);
  const x0 = percentile(xs, 2);
  const x1 = percentile(xs, 98);
  const y0 = percentile(ys, 2);
  const y1 = percentile(ys, 98);
  const under: number[] = [];
  for (let i = 0; i < xs.length; i++) if (xs[i] >= x0 && xs[i] <= x1 && ys[i] >= y0 && ys[i] <= y1) under.push(zs[i]);
  return Float64Array.of(0.5 * (x0 + x1), 0.5 * (y0 + y1), percentile(under.length ? under : zs, 2));
}
