/**
 * Ground-truth surface samples: area-uniform random points on all scene triangles
 * (world frame) with their face normals, and a per-sample visibility count computed
 * from rendered depth maps (how many frames see the sample unoccluded).
 */
import type { Mesh } from 'three';
import { Vector3 } from 'three';
import type { CameraIntrinsics, Pose } from '../../core/types';
import type { NormalizedWindow } from './conventions';
import type { SynthRng } from './rng';

export interface SurfaceSamples {
  count: number;
  /** xyz per sample (world, metres). */
  positions: Float32Array;
  /** Unit face normal per sample (world). */
  normals: Float32Array;
  /** Total sampled area (m²). */
  totalArea: number;
}

/** World-space triangles of all meshes (9 floats each). */
export function collectTriangles(meshes: readonly Mesh[]): Float32Array {
  let total = 0;
  for (const m of meshes) {
    const g = m.geometry;
    total += g.index ? g.index.count / 3 : g.getAttribute('position').count / 3;
  }
  const out = new Float32Array(total * 9);
  const v = new Vector3();
  let o = 0;
  for (const m of meshes) {
    m.updateWorldMatrix(true, false);
    const g = m.geometry;
    const pos = g.getAttribute('position');
    const idx = g.index;
    const n = idx ? idx.count : pos.count;
    for (let k = 0; k < n; k++) {
      const i = idx ? idx.getX(k) : k;
      v.set(pos.getX(i), pos.getY(i), pos.getZ(i)).applyMatrix4(m.matrixWorld);
      out[o++] = v.x;
      out[o++] = v.y;
      out[o++] = v.z;
    }
  }
  return out;
}

/** Area-uniform samples over the triangles (9 floats per triangle). */
export function sampleTriangles(tris: Float32Array, count: number, rng: SynthRng): SurfaceSamples {
  const nt = tris.length / 9;
  const cdf = new Float64Array(nt);
  let acc = 0;
  for (let t = 0; t < nt; t++) {
    const o = 9 * t;
    const e1x = tris[o + 3] - tris[o],
      e1y = tris[o + 4] - tris[o + 1],
      e1z = tris[o + 5] - tris[o + 2];
    const e2x = tris[o + 6] - tris[o],
      e2y = tris[o + 7] - tris[o + 1],
      e2z = tris[o + 8] - tris[o + 2];
    const cx = e1y * e2z - e1z * e2y,
      cy = e1z * e2x - e1x * e2z,
      cz = e1x * e2y - e1y * e2x;
    acc += 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz);
    cdf[t] = acc;
  }
  const positions = new Float32Array(3 * count);
  const normals = new Float32Array(3 * count);
  for (let s = 0; s < count; s++) {
    const u = rng.next() * acc;
    let lo = 0,
      hi = nt - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (cdf[mid] < u) lo = mid + 1;
      else hi = mid;
    }
    const o = 9 * lo;
    const r1 = Math.sqrt(rng.next()),
      r2 = rng.next();
    const a = 1 - r1,
      b = r1 * (1 - r2),
      c = r1 * r2;
    for (let j = 0; j < 3; j++) positions[3 * s + j] = a * tris[o + j] + b * tris[o + 3 + j] + c * tris[o + 6 + j];
    const e1x = tris[o + 3] - tris[o],
      e1y = tris[o + 4] - tris[o + 1],
      e1z = tris[o + 5] - tris[o + 2];
    const e2x = tris[o + 6] - tris[o],
      e2y = tris[o + 7] - tris[o + 1],
      e2z = tris[o + 8] - tris[o + 2];
    let nx = e1y * e2z - e1z * e2y,
      ny = e1z * e2x - e1x * e2z,
      nz = e1x * e2y - e1y * e2x;
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l;
    ny /= l;
    nz /= l;
    normals[3 * s] = nx;
    normals[3 * s + 1] = ny;
    normals[3 * s + 2] = nz;
  }
  return { count, positions, normals, totalArea: acc };
}

export function sampleSurfaces(meshes: readonly Mesh[], count: number, rng: SynthRng): SurfaceSamples {
  return sampleTriangles(collectTriangles(meshes), count, rng);
}

/** Normals quantised to int8 (round(127·n)). */
export function quantizeNormals(n: Float32Array): Int8Array {
  const q = new Int8Array(n.length);
  for (let i = 0; i < n.length; i++) q[i] = Math.max(-127, Math.min(127, Math.round(n[i] * 127)));
  return q;
}

export interface WindowDepth {
  /** z-depth per pixel, row 0 = TOP of the image, 0 = no surface. */
  data: Float32Array;
  width: number;
  height: number;
  /** Undistorted normalized window covered by the depth map (pixel edges). */
  window: NormalizedWindow;
}

/**
 * Increment `counts[i]` (saturating at 255) for every sample that projects inside the
 * distorted image of `cam` and agrees with the depth map (unoccluded) within
 * max(relTol·z, absTol, 2 depth-pixel footprints) in a 3×3 neighbourhood.
 */
export function accumulateVisibility(
  points: Float32Array,
  counts: Uint8Array,
  pose: Pose,
  cam: CameraIntrinsics,
  depth: WindowDepth,
  relTol = 0.01,
  absTol = 0.005,
): number {
  const R = pose.R,
    t = pose.t;
  const { xmin, xmax, ymin, ymax } = depth.window;
  const Wd = depth.width,
    Hd = depth.height,
    D = depth.data;
  const sx = Wd / (xmax - xmin),
    sy = Hd / (ymax - ymin);
  const pix = Math.max(1 / sx, 1 / sy);
  const umin = -0.5,
    umax = cam.width - 0.5,
    vmin = -0.5,
    vmax = cam.height - 0.5;
  const { f, cx, cy, k1, k2 } = cam;
  const n = points.length / 3;
  let visible = 0;
  for (let i = 0; i < n; i++) {
    const X = points[3 * i],
      Y = points[3 * i + 1],
      Z = points[3 * i + 2];
    const zc = R[6] * X + R[7] * Y + R[8] * Z + t[2];
    if (!(zc > 1e-3)) continue;
    const xn = (R[0] * X + R[1] * Y + R[2] * Z + t[0]) / zc;
    const yn = (R[3] * X + R[4] * Y + R[5] * Z + t[1]) / zc;
    const r2 = xn * xn + yn * yn;
    const dd = 1 + k1 * r2 + k2 * r2 * r2;
    const u = f * dd * xn + cx,
      v = f * dd * yn + cy;
    if (u < umin || u > umax || v < vmin || v > vmax) continue;
    const col = Math.floor((xn - xmin) * sx),
      row = Math.floor((yn - ymin) * sy);
    const tol = Math.max(relTol * zc, absTol, 2 * pix * zc);
    let ok = false;
    for (let dy = -1; dy <= 1 && !ok; dy++) {
      const r = row + dy;
      if (r < 0 || r >= Hd) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const c = col + dx;
        if (c < 0 || c >= Wd) continue;
        const zd = D[r * Wd + c];
        if (zd > 0 && Math.abs(zd - zc) <= tol) {
          ok = true;
          break;
        }
      }
    }
    if (ok) {
      visible++;
      if (counts[i] < 255) counts[i]++;
    }
  }
  return visible;
}
