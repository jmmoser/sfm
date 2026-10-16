/**
 * COLMAP text model export / import (cameras.txt, images.txt, points3D.txt).
 *
 * Conversions (see docs/ARCHITECTURE.md §2):
 *  - Camera model RADIAL, params `f, cx + 0.5, cy + 0.5, k1, k2` (COLMAP puts the
 *    centre of the top-left pixel at (0.5, 0.5); we put it at (0, 0)).
 *  - Image pose QW QX QY QZ TX TY TZ = world-to-camera rotation (Hamilton,
 *    w ≥ 0) and translation — identical to our `Pose` convention.
 *  - POINTS2D x/y get +0.5. POINT3D_ID −1 marks an unmatched keypoint.
 * Numbers are written with JavaScript's shortest round-trip representation, so
 * export → import is exact up to the ±0.5 shift (≤ 1 ulp).
 *
 * Import additionally accepts SIMPLE_PINHOLE, PINHOLE (fx, fy averaged),
 * SIMPLE_RADIAL and OPENCV (fx, fy averaged; p1, p2 dropped) cameras so
 * external COLMAP datasets can be loaded by the test harness.
 */
import type { CameraIntrinsics, Pose } from '../core/types';
import { createZip, createZipBlob } from './zip';

export interface ColmapModelInput {
  cameras: { id: number; cam: CameraIntrinsics }[];
  images: {
    id: number;
    cameraId: number;
    name: string;
    pose: Pose;
    points2D: { x: number; y: number; point3D: number }[];
  }[];
  points: {
    id: number;
    xyz: ArrayLike<number>;
    color: ArrayLike<number>;
    error: number;
    track: { image: number; point2D: number }[];
  }[];
}

export interface ColmapTextFile {
  name: string;
  data: string;
}

// ---------------------------------------------------------------------------
// Rotation helpers (local to avoid a build-time dependency on src/math)
// ---------------------------------------------------------------------------

/** Row-major rotation matrix → unit quaternion [w, x, y, z] with w ≥ 0 (Shepperd). */
function mat3ToQuatWXYZ(R: ArrayLike<number>): [number, number, number, number] {
  const m00 = R[0], m01 = R[1], m02 = R[2];
  const m10 = R[3], m11 = R[4], m12 = R[5];
  const m20 = R[6], m21 = R[7], m22 = R[8];
  const tr = m00 + m11 + m22;
  let w: number, x: number, y: number, z: number;
  if (tr > 0) {
    const s = 2 * Math.sqrt(tr + 1);
    w = 0.25 * s;
    x = (m21 - m12) / s;
    y = (m02 - m20) / s;
    z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    w = (m21 - m12) / s;
    x = 0.25 * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25 * s;
    z = (m12 + m21) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25 * s;
  }
  const n = Math.hypot(w, x, y, z) * (w < 0 ? -1 : 1);
  return [w / n, x / n, y / n, z / n];
}

/** Unit quaternion [w, x, y, z] → row-major rotation matrix. */
function quatWXYZToMat3(w: number, x: number, y: number, z: number): Float64Array {
  const n = Math.hypot(w, x, y, z);
  w /= n; x /= n; y /= n; z /= n;
  return Float64Array.of(
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  );
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

function num(v: number): string {
  if (!Number.isFinite(v)) throw new RangeError(`COLMAP export: non-finite value ${v}`);
  return String(v);
}

function checkName(name: string): void {
  if (!name || /\s/.test(name)) throw new Error(`COLMAP export: image name must be non-empty without whitespace: '${name}'`);
}

/** Build the three COLMAP text files. Returned in order cameras, images, points3D. */
export function exportColmapText(model: ColmapModelInput): ColmapTextFile[] {
  // cameras.txt
  const cam: string[] = [
    '# Camera list with one line of data per camera:',
    '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]',
    `# Number of cameras: ${model.cameras.length}`,
  ];
  for (const { id, cam: c } of model.cameras) {
    cam.push(`${id} RADIAL ${c.width} ${c.height} ${num(c.f)} ${num(c.cx + 0.5)} ${num(c.cy + 0.5)} ${num(c.k1)} ${num(c.k2)}`);
  }

  // images.txt
  let numObs = 0;
  for (const im of model.images) for (const p of im.points2D) if (p.point3D >= 0) numObs++;
  const meanObs = model.images.length ? numObs / model.images.length : 0;
  const img: string[] = [
    '# Image list with two lines of data per image:',
    '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME',
    '#   POINTS2D[] as (X, Y, POINT3D_ID)',
    `# Number of images: ${model.images.length}, mean observations per image: ${meanObs}`,
  ];
  for (const im of model.images) {
    checkName(im.name);
    const q = mat3ToQuatWXYZ(im.pose.R);
    const t = im.pose.t;
    img.push(`${im.id} ${num(q[0])} ${num(q[1])} ${num(q[2])} ${num(q[3])} ${num(t[0])} ${num(t[1])} ${num(t[2])} ${im.cameraId} ${im.name}`);
    const parts = new Array<string>(im.points2D.length);
    for (let k = 0; k < im.points2D.length; k++) {
      const p = im.points2D[k];
      parts[k] = `${num(p.x + 0.5)} ${num(p.y + 0.5)} ${p.point3D >= 0 ? p.point3D : -1}`;
    }
    img.push(parts.join(' '));
  }

  // points3D.txt
  let trackSum = 0;
  for (const p of model.points) trackSum += p.track.length;
  const meanTrack = model.points.length ? trackSum / model.points.length : 0;
  const pts: string[] = [
    '# 3D point list with one line of data per point:',
    '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)',
    `# Number of points: ${model.points.length}, mean track length: ${meanTrack}`,
  ];
  const c8 = (v: number): number => Math.max(0, Math.min(255, Math.round(v)));
  for (const p of model.points) {
    let line = `${p.id} ${num(p.xyz[0])} ${num(p.xyz[1])} ${num(p.xyz[2])} ${c8(p.color[0])} ${c8(p.color[1])} ${c8(p.color[2])} ${num(p.error)}`;
    for (const t of p.track) line += ` ${t.image} ${t.point2D}`;
    pts.push(line);
  }

  return [
    { name: 'cameras.txt', data: cam.join('\n') + '\n' },
    { name: 'images.txt', data: img.join('\n') + '\n' },
    { name: 'points3D.txt', data: pts.join('\n') + '\n' },
  ];
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

function parseNums(tok: string[], from: number, count: number, ctx: string): number[] {
  const out: number[] = [];
  for (let i = from; i < from + count; i++) {
    const v = Number(tok[i]);
    if (tok[i] === undefined || !Number.isFinite(v)) throw new Error(`COLMAP import: bad number '${tok[i]}' in ${ctx}`);
    out.push(v);
  }
  return out;
}

function cameraFromColmap(model: string, w: number, h: number, p: number[], ctx: string): CameraIntrinsics {
  const need: Record<string, number> = { SIMPLE_PINHOLE: 3, PINHOLE: 4, SIMPLE_RADIAL: 4, RADIAL: 5, OPENCV: 8 };
  if (!(model in need)) throw new Error(`COLMAP import: unsupported camera model ${model}`);
  if (p.length < need[model]) throw new Error(`COLMAP import: ${model} needs ${need[model]} params in ${ctx}`);
  switch (model) {
    case 'SIMPLE_PINHOLE':
      return { width: w, height: h, f: p[0], cx: p[1] - 0.5, cy: p[2] - 0.5, k1: 0, k2: 0 };
    case 'PINHOLE':
      return { width: w, height: h, f: 0.5 * (p[0] + p[1]), cx: p[2] - 0.5, cy: p[3] - 0.5, k1: 0, k2: 0 };
    case 'SIMPLE_RADIAL':
      return { width: w, height: h, f: p[0], cx: p[1] - 0.5, cy: p[2] - 0.5, k1: p[3], k2: 0 };
    case 'RADIAL':
      return { width: w, height: h, f: p[0], cx: p[1] - 0.5, cy: p[2] - 0.5, k1: p[3], k2: p[4] };
    default: // OPENCV: fx fy cx cy k1 k2 p1 p2
      return { width: w, height: h, f: 0.5 * (p[0] + p[1]), cx: p[2] - 0.5, cy: p[3] - 0.5, k1: p[4], k2: p[5] };
  }
}

function lines(text: string): string[] {
  return text.split(/\r?\n/);
}

function isSkippable(line: string): boolean {
  const t = line.trim();
  return t.length === 0 || t[0] === '#';
}

/**
 * Parse COLMAP text files (inverse of `exportColmapText`). Accepts either the
 * array returned by the exporter (matched by file basename) or the three texts.
 */
export function importColmapText(
  files: ColmapTextFile[] | { cameras: string; images: string; points3D: string },
): ColmapModelInput {
  let camerasTxt: string | undefined;
  let imagesTxt: string | undefined;
  let pointsTxt: string | undefined;
  if (Array.isArray(files)) {
    for (const f of files) {
      const base = f.name.split('/').pop();
      if (base === 'cameras.txt') camerasTxt = f.data;
      else if (base === 'images.txt') imagesTxt = f.data;
      else if (base === 'points3D.txt') pointsTxt = f.data;
    }
  } else {
    camerasTxt = files.cameras;
    imagesTxt = files.images;
    pointsTxt = files.points3D;
  }
  if (camerasTxt === undefined || imagesTxt === undefined || pointsTxt === undefined) {
    throw new Error('COLMAP import: need cameras.txt, images.txt and points3D.txt');
  }

  const model: ColmapModelInput = { cameras: [], images: [], points: [] };

  for (const line of lines(camerasTxt)) {
    if (isSkippable(line)) continue;
    const tok = line.trim().split(/\s+/);
    const [id, w, h] = parseNums(tok, 0, 1, 'cameras.txt').concat(parseNums(tok, 2, 2, 'cameras.txt'));
    const params = parseNums(tok, 4, tok.length - 4, 'cameras.txt');
    model.cameras.push({ id, cam: cameraFromColmap(tok[1], w, h, params, `camera ${id}`) });
  }

  // images.txt: an image line is ALWAYS followed by its POINTS2D line (possibly empty).
  const il = lines(imagesTxt);
  for (let i = 0; i < il.length; i++) {
    if (isSkippable(il[i])) continue;
    const tok = il[i].trim().split(/\s+/);
    if (tok.length < 10) throw new Error(`COLMAP import: bad image line '${il[i]}'`);
    const [id, qw, qx, qy, qz, tx, ty, tz, cameraId] = parseNums(tok, 0, 9, 'images.txt');
    const name = tok.slice(9).join(' ');
    const ptsLine = i + 1 < il.length ? il[i + 1].trim() : '';
    i++;
    const points2D: { x: number; y: number; point3D: number }[] = [];
    if (ptsLine.length > 0) {
      const pt = ptsLine.split(/\s+/);
      if (pt.length % 3 !== 0) throw new Error(`COLMAP import: POINTS2D of image ${id} not a multiple of 3`);
      for (let k = 0; k < pt.length; k += 3) {
        const [x, y, p3] = parseNums(pt, k, 3, `POINTS2D of image ${id}`);
        points2D.push({ x: x - 0.5, y: y - 0.5, point3D: p3 < 0 ? -1 : p3 });
      }
    }
    model.images.push({
      id,
      cameraId,
      name,
      pose: { R: quatWXYZToMat3(qw, qx, qy, qz), t: Float64Array.of(tx, ty, tz) },
      points2D,
    });
  }

  for (const line of lines(pointsTxt)) {
    if (isSkippable(line)) continue;
    const tok = line.trim().split(/\s+/);
    if (tok.length < 8 || (tok.length - 8) % 2 !== 0) throw new Error(`COLMAP import: bad point line '${line.slice(0, 80)}'`);
    const v = parseNums(tok, 0, tok.length, 'points3D.txt');
    const track: { image: number; point2D: number }[] = [];
    for (let k = 8; k < v.length; k += 2) track.push({ image: v[k], point2D: v[k + 1] });
    model.points.push({
      id: v[0],
      xyz: Float64Array.of(v[1], v[2], v[3]),
      color: Uint8Array.of(v[4], v[5], v[6]),
      error: v[7],
      track,
    });
  }
  return model;
}

// ---------------------------------------------------------------------------
// Archive
// ---------------------------------------------------------------------------

export interface ColmapArchiveImage {
  /** Must equal the image's `name` in the model. */
  name: string;
  data: Blob | Uint8Array | ArrayBuffer;
}

/**
 * Standard COLMAP project layout in a ZIP:
 *   sparse/0/cameras.txt, sparse/0/images.txt, sparse/0/points3D.txt, images/<name>
 * Blob images are referenced (not copied) — see createZipBlob.
 */
export async function createColmapZipBlob(
  model: ColmapModelInput,
  images: ColmapArchiveImage[] = [],
  opts: { readme?: string } = {},
): Promise<Blob> {
  const files = exportColmapText(model).map((f) => ({ name: `sparse/0/${f.name}`, data: f.data }));
  const entries: { name: string; data: Blob | Uint8Array | ArrayBuffer | string }[] = [...files];
  if (opts.readme) entries.push({ name: 'README.txt', data: opts.readme });
  for (const im of images) entries.push({ name: `images/${im.name}`, data: im.data });
  return createZipBlob(entries);
}

/** Synchronous in-memory variant of createColmapZipBlob (no Blob inputs). */
export function createColmapZip(model: ColmapModelInput, images: { name: string; data: Uint8Array | ArrayBuffer }[] = []): Uint8Array {
  const files = exportColmapText(model).map((f) => ({ name: `sparse/0/${f.name}`, data: f.data as string | Uint8Array | ArrayBuffer }));
  for (const im of images) files.push({ name: `images/${im.name}`, data: im.data });
  return createZip(files);
}
