/**
 * Exports (ENGINE.md §2 `exportModel`): PLY points (dense if available, else sparse), PLY
 * mesh, GLB (mesh if available, else points), COLMAP text model + images in a ZIP, Gaussian
 * splats as an INRIA 3DGS PLY (appearance-normalised model of the splat stage), USDZ for AR
 * Quick Look (the mesh simplified to ≤ 100 k triangles with its full-resolution colours baked
 * into one 2048² JPEG atlas — src/dense/colour-atlas.ts —, Y-up metres, origin at the AR pivot:
 * `selectArPivot`, which switches to the object ROI once `objectRoi` is supplied; ≈ 10–11 MB).
 *
 * Privacy: no export carries a location — there is no geo input here, coordinates are local
 * (ENU metres when aligned) and the keyframe JPEGs are canvas re-encodes (no EXIF). Keep it so.
 */
import type { FeatureSet, MetricInfo, PointCloud, RGBAImage, TriangleMesh } from '../core/types';
import { buildColourAtlas } from '../dense/colour-atlas';
import { createColmapZipBlob, type ColmapArchiveImage, type ColmapModelInput } from '../io/colmap';
import { encodeGLB } from '../io/glb';
import { encodeMeshPLY, encodePLY } from '../io/ply';
import { encodeUSDZ, selectArPivot, type ArPivotRoi } from '../io/usdz';
import { encodeSplatPLY } from '../splat/ply';
import type { ExportFormat, SplatModelPayload } from '../workers/protocol';
import { encodeJpegRGBA } from './frame-ingest';
import { isRGBAImage, type EngineStore } from './persistence';
import type { SparseModel } from './types';

export interface ExportSources {
  name: string;
  sparse: PointCloud | null;
  model: SparseModel | null;
  dense: PointCloud | null;
  mesh: TriangleMesh | null;
  store: EngineStore | null;
  /** Features of a keyframe (memory or store). */
  features(id: number): Promise<FeatureSet | null>;
  /** The stored splat model (read on demand), null when none. */
  splat?: () => Promise<SplatModelPayload | null>;
  /** Metadata comments for the files. */
  comments?: string[];
  /** Scale / gravity alignment of the model (USDZ records whether it is metric). */
  metric?: MetricInfo | null;
  /** Object ROI of the mesh (src/dense/object-roi.ts), when trusted: the AR pivot (`selectArPivot`). */
  objectRoi?: ArPivotRoi | null;
  /** Texture encoder of the USDZ atlas (default: JPEG through OffscreenCanvas; tests inject one). */
  encodeTexture?: (img: RGBAImage) => Promise<Blob | null>;
}

/** AR Quick Look budget (Apple: ≈ 100 k polygons and one 2048² texture set per model). */
export const USDZ_MAX_TRIANGLES = 100_000;
export const USDZ_TEXTURE_SIZE = 2048;
/** JPEG quality of the USDZ atlas (≈ 1.5 MB at 2048²). */
export const USDZ_JPEG_QUALITY = 0.9;
/**
 * USDZ material: 'diffuse' (AR Quick Look lights the photo colours, which contain shading
 * already) or 'emissive' (flat, unlit-looking). Which reads better is open until the device
 * check (docs/research/ios-platform.md §8); flip it here for that A/B.
 */
export const USDZ_MATERIAL: 'diffuse' | 'emissive' = 'diffuse';

/** Uncompressed 24-bit BMP (COLMAP / FreeImage read it) for frames stored as raw RGBA. */
export function encodeBMP(img: RGBAImage): Uint8Array {
  const w = img.width,
    h = img.height;
  const row = (3 * w + 3) & ~3;
  const size = 54 + row * h;
  const out = new Uint8Array(size);
  const dv = new DataView(out.buffer);
  out[0] = 0x42;
  out[1] = 0x4d;
  dv.setUint32(2, size, true);
  dv.setUint32(10, 54, true);
  dv.setUint32(14, 40, true);
  dv.setInt32(18, w, true);
  dv.setInt32(22, h, true);
  dv.setUint16(26, 1, true);
  dv.setUint16(28, 24, true);
  dv.setUint32(34, row * h, true);
  dv.setInt32(38, 2835, true);
  dv.setInt32(42, 2835, true);
  for (let y = 0; y < h; y++) {
    const src = (h - 1 - y) * w * 4;
    const dst = 54 + y * row;
    for (let x = 0; x < w; x++) {
      out[dst + 3 * x] = img.data[src + 4 * x + 2];
      out[dst + 3 * x + 1] = img.data[src + 4 * x + 1];
      out[dst + 3 * x + 2] = img.data[src + 4 * x];
    }
  }
  return out;
}

function imageName(id: number, ext: string): string {
  return `kf_${String(id).padStart(5, '0')}.${ext}`;
}

/** COLMAP model of a sparse model (+ keypoints of every registered image as POINTS2D). */
export async function colmapModel(src: ExportSources, ext: (id: number) => string): Promise<ColmapModelInput> {
  const m = src.model;
  if (!m) throw new Error('No sparse model yet (run refine first)');
  // image → feature → point id
  const obs = new Map<number, Map<number, number>>();
  for (let i = 0; i < m.numPoints; i++) {
    for (let k = m.trackOffsets[i]; k < m.trackOffsets[i + 1]; k++) {
      const im = m.trackImages[k];
      let mm = obs.get(im);
      if (!mm) obs.set(im, (mm = new Map()));
      mm.set(m.trackFeatures[k], i);
    }
  }
  const images: ColmapModelInput['images'] = [];
  const exported = new Set<number>();
  const numFeat = new Map<number, number>();
  for (const im of m.images) {
    const fs = await src.features(im.id);
    const pts = obs.get(im.id) ?? new Map<number, number>();
    const points2D: { x: number; y: number; point3D: number }[] = [];
    if (fs) {
      for (let f = 0; f < fs.count; f++) points2D.push({ x: fs.keypoints[4 * f], y: fs.keypoints[4 * f + 1], point3D: pts.get(f) ?? -1 });
      numFeat.set(im.id, fs.count);
    }
    exported.add(im.id);
    images.push({ id: im.id, cameraId: im.cameraId, name: imageName(im.id, ext(im.id)), pose: im.pose, points2D });
  }
  const points: ColmapModelInput['points'] = [];
  for (let i = 0; i < m.numPoints; i++) {
    const track: { image: number; point2D: number }[] = [];
    for (let k = m.trackOffsets[i]; k < m.trackOffsets[i + 1]; k++) {
      const im = m.trackImages[k];
      const f = m.trackFeatures[k];
      if (exported.has(im) && f < (numFeat.get(im) ?? 0)) track.push({ image: im, point2D: f });
    }
    if (track.length < 2) continue;
    points.push({ id: i, xyz: m.xyz.subarray(3 * i, 3 * i + 3), color: m.colors.subarray(3 * i, 3 * i + 3), error: Number.isFinite(m.errors[i]) && m.errors[i] >= 0 ? m.errors[i] : 0, track });
  }
  // point ids referenced by POINTS2D but not exported become −1 (exportColmapText handles it)
  return { cameras: [...m.cameras.entries()].map(([id, cam]) => ({ id, cam })), images, points };
}

/** Build the export Blob. */
export async function buildExport(format: ExportFormat, src: ExportSources): Promise<Blob> {
  const comments = src.comments ?? [`sfm on-device photogrammetry: ${src.name}`];
  // an empty dense cloud (fusion kept no point) falls back to the sparse one
  const dense = src.dense && src.dense.count > 0 ? src.dense : null;
  switch (format) {
    case 'ply-points': {
      const pc = dense ?? src.sparse;
      if (!pc || pc.count === 0) throw new Error('No point cloud yet (run refine first)');
      return new Blob([encodePLY(pc, { comments })], { type: 'application/octet-stream' });
    }
    case 'ply-mesh': {
      if (!src.mesh) throw new Error('No mesh yet (run mesh first)');
      return new Blob([encodeMeshPLY(src.mesh, { comments })], { type: 'application/octet-stream' });
    }
    case 'glb': {
      const points = src.mesh ? undefined : (dense ?? src.sparse ?? undefined);
      if (!src.mesh && (!points || points.count === 0)) throw new Error('Nothing to export yet (run refine first)');
      // unlit (KHR_materials_unlit): the vertex colours are photo colours that already contain the
      // shading; lit, viewers shade them twice (−5 dB held-out PSNR in the app's viewer)
      return new Blob([encodeGLB({ mesh: src.mesh ?? undefined, points, unlitMesh: true, extras: { name: src.name } })], { type: 'model/gltf-binary' });
    }
    case 'colmap-zip': {
      if (!src.model) throw new Error('No sparse model yet (run refine first)');
      // image data first (decides the file extension per image)
      const files = new Map<number, Blob | Uint8Array>();
      if (src.store) {
        for (const im of src.model.images) {
          const s = await src.store.getFrameImage(im.id);
          if (!s) continue;
          files.set(im.id, isRGBAImage(s) ? encodeBMP(s) : s);
        }
      }
      const ext = (id: number): string => (files.get(id) instanceof Uint8Array ? 'bmp' : 'jpg');
      const model = await colmapModel(src, ext);
      const images: ColmapArchiveImage[] = [];
      for (const im of model.images) {
        const data = files.get(im.id);
        if (data) images.push({ name: im.name, data });
      }
      const readme =
        `COLMAP sparse model exported by the on-device photogrammetry app (${src.name}).\n` +
        `Camera model RADIAL; images are the keyframes in the session's canonical orientation.\n` +
        `World frame: +Z up (gravity aligned when the IMU allowed it); units are metres when the scan is metric.\n`;
      return createColmapZipBlob(model, images, { readme });
    }
    case 'ply-splat': {
      const sp = src.splat ? await src.splat() : null;
      if (!sp || sp.model.count === 0) throw new Error('No Gaussian splats yet (run splat first)');
      return new Blob([encodeSplatPLY(sp.model, { antialiased: sp.antialiased, comments }) as Uint8Array<ArrayBuffer>], { type: 'application/octet-stream' });
    }
    case 'usdz': {
      const mesh = src.mesh;
      if (!mesh || mesh.triangleCount === 0) throw new Error('No mesh yet (run mesh first)');
      // the pivot from the undecimated mesh (about uniform vertex density: see selectArPivot)
      const pivot = selectArPivot(mesh, src.objectRoi);
      const atlas = buildColourAtlas(mesh, { maxTriangles: USDZ_MAX_TRIANGLES, size: USDZ_TEXTURE_SIZE });
      const img = await (src.encodeTexture ?? ((t: RGBAImage) => encodeJpegRGBA(t, USDZ_JPEG_QUALITY)))(atlas.texture);
      if (!img) throw new Error('USDZ export needs OffscreenCanvas (JPEG encoding), which this browser lacks');
      const usdz = encodeUSDZ({
        mesh: atlas.mesh,
        uvs: atlas.uvs,
        texture: new Uint8Array(await img.arrayBuffer()),
        textureType: img.type === 'image/png' ? 'image/png' : 'image/jpeg',
        pivot,
        metric: src.metric ? src.metric.metric : undefined,
        name: src.name,
        material: USDZ_MATERIAL,
      });
      return new Blob([usdz as Uint8Array<ArrayBuffer>], { type: 'model/vnd.usdz+zip' });
    }
    default:
      throw new Error(`Unknown export format ${String(format)}`);
  }
}
