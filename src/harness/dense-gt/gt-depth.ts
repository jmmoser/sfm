/**
 * Ground-truth depth, visibility and sparse points for the dense GT harness
 * (dense-gt.html), built from the synthetic dataset's area-uniform surface samples
 * (`gt_points.bin` + int8 normals, and the dense `gt_focus_*` set of the object scene).
 *
 * Surfel z-buffer: every GT sample is rendered as a disc of radius
 * `radiusFactor · spacing` in the plane of its normal (ray–plane intersection per pixel
 * centre, disc test, z-min). Area-uniform random samples with disc radius 1.3 × the
 * mean spacing cover a surface with probability 1 − exp(−π·1.3²) ≈ 99.5 %, and inside a
 * disc the depth is exact for planar patches, so the map is a good GT stand-in for every
 * view (the dataset only ships exact depth maps for every 20th frame). Limits: silhouettes
 * are fattened by up to one disc radius (≈ 16 mm for the 12 mm-spaced base set of the
 * object room, ≈ 2 mm for the focus set; 0.15 m in the yard) and curved surfaces deviate by
 * ≈ r²/(2·radius of curvature). `undistortGtDepth` resamples an exact dataset depth map
 * (distorted full-resolution frame) into the undistorted dense frame, so the surfel map can
 * be validated against it.
 *
 * When a focus set exists, base samples within one base spacing (27-cell neighbourhood)
 * of a focus sample are dropped: the coarse base discs of the object would otherwise
 * fatten the object's silhouette by ~16 px at 0.4 m.
 */
import type { CameraIntrinsics, Pose } from '../../core/types';
import { Rng } from '../../math/random';

export interface GtSurfelSet {
  name: string;
  points: Float32Array;
  /** int8 xyz = round(127·n) (world frame); null = discs face the camera. */
  normals: Int8Array | null;
  /** Mean sample spacing (world units). */
  spacing: number;
  /** Number of dataset frames that see each sample (null = unknown). */
  visibility: Uint8Array | null;
  /** Optional keep mask (1 = use); null = all samples. */
  keep: Uint8Array | null;
}

export interface GtDepthMap {
  width: number;
  height: number;
  /** z-depth, 0 = no surface. */
  depth: Float32Array;
  /** Unit normals (camera frame, facing the camera), 0 where no surface. */
  normal: Float32Array;
  /** Index of the set that produced each pixel (255 = none). */
  setId: Uint8Array;
}

export interface SurfelRenderOptions {
  /** Disc radius in units of the set spacing. Default 1.3. */
  radiusFactor?: number;
  /** Near plane (world units). Default 1e-3. */
  near?: number;
}

/** Drop base samples near any focus sample (see file header). Returns the keep mask. */
export function maskNearFocus(base: Float32Array, focus: Float32Array, cell: number): Uint8Array {
  let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  const nf = focus.length / 3;
  for (let i = 0; i < nf; i++) {
    const x = focus[3 * i], y = focus[3 * i + 1], z = focus[3 * i + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  const ox = minX - 2 * cell, oy = minY - 2 * cell, oz = minZ - 2 * cell;
  const nx = Math.ceil((maxX - ox) / cell) + 3;
  const ny = Math.ceil((maxY - oy) / cell) + 3;
  const nz = Math.ceil((maxZ - oz) / cell) + 3;
  const occ = new Uint8Array(nx * ny * nz);
  for (let i = 0; i < nf; i++) {
    const ix = Math.floor((focus[3 * i] - ox) / cell);
    const iy = Math.floor((focus[3 * i + 1] - oy) / cell);
    const iz = Math.floor((focus[3 * i + 2] - oz) / cell);
    occ[(iz * ny + iy) * nx + ix] = 1;
  }
  const nb = base.length / 3;
  const keep = new Uint8Array(nb).fill(1);
  for (let i = 0; i < nb; i++) {
    const ix = Math.floor((base[3 * i] - ox) / cell);
    const iy = Math.floor((base[3 * i + 1] - oy) / cell);
    const iz = Math.floor((base[3 * i + 2] - oz) / cell);
    if (ix < 1 || iy < 1 || iz < 1 || ix >= nx - 1 || iy >= ny - 1 || iz >= nz - 1) continue;
    let hit = false;
    for (let dz = -1; dz <= 1 && !hit; dz++)
      for (let dy = -1; dy <= 1 && !hit; dy++)
        for (let dx = -1; dx <= 1; dx++)
          if (occ[((iz + dz) * ny + iy + dy) * nx + ix + dx]) {
            hit = true;
            break;
          }
    if (hit) keep[i] = 0;
  }
  return keep;
}

/** Render the surfel z-buffer of `sets` for a pinhole camera (k1 = k2 = 0). */
export function renderSurfelDepth(cam: CameraIntrinsics, pose: Pose, sets: GtSurfelSet[], opts: SurfelRenderOptions = {}): GtDepthMap {
  const W = cam.width;
  const H = cam.height;
  const f = cam.f;
  const cx = cam.cx;
  const cy = cam.cy;
  const invF = 1 / f;
  const near = opts.near ?? 1e-3;
  const kR = opts.radiusFactor ?? 1.3;
  const depth = new Float32Array(W * H);
  const normal = new Float32Array(3 * W * H);
  const setId = new Uint8Array(W * H).fill(255);
  const R = pose.R;
  const t = pose.t;
  const R0 = R[0], R1 = R[1], R2 = R[2], R3 = R[3], R4 = R[4], R5 = R[5], R6 = R[6], R7 = R[7], R8 = R[8];
  const t0 = t[0], t1 = t[1], t2 = t[2];
  for (let s = 0; s < sets.length; s++) {
    const S = sets[s];
    const P = S.points;
    const N = S.normals;
    const keep = S.keep;
    const r = kR * S.spacing;
    const r2 = r * r;
    const n = P.length / 3;
    for (let i = 0; i < n; i++) {
      if (keep && !keep[i]) continue;
      const X = P[3 * i], Y = P[3 * i + 1], Z = P[3 * i + 2];
      const zc = R6 * X + R7 * Y + R8 * Z + t2;
      if (zc - r <= near) continue;
      const xc = R0 * X + R1 * Y + R2 * Z + t0;
      const yc = R3 * X + R4 * Y + R5 * Z + t1;
      let nx: number, ny: number, nz: number;
      if (N) {
        const ax = N[3 * i] / 127, ay = N[3 * i + 1] / 127, az = N[3 * i + 2] / 127;
        nx = R0 * ax + R1 * ay + R2 * az;
        ny = R3 * ax + R4 * ay + R5 * az;
        nz = R6 * ax + R7 * ay + R8 * az;
        const l = Math.hypot(nx, ny, nz);
        if (l > 0) {
          nx /= l;
          ny /= l;
          nz /= l;
        } else {
          const lc = Math.hypot(xc, yc, zc);
          nx = -xc / lc;
          ny = -yc / lc;
          nz = -zc / lc;
        }
      } else {
        const lc = Math.hypot(xc, yc, zc);
        nx = -xc / lc;
        ny = -yc / lc;
        nz = -zc / lc;
      }
      const dn = nx * xc + ny * yc + nz * zc;
      // exact image bounding box of the disc: it lies in the axis-aligned box with
      // half-extents r·sqrt(1 − n_a²); x/z is monotone in x and z there (z > near)
      const ex = r * Math.sqrt(Math.max(0, 1 - nx * nx));
      const ey = r * Math.sqrt(Math.max(0, 1 - ny * ny));
      const ez = r * Math.sqrt(Math.max(0, 1 - nz * nz));
      const zlo = zc - ez;
      const zhi = zc + ez;
      if (zlo <= near) continue;
      const xa = xc - ex, xb = xc + ex, ya = yc - ey, yb = yc + ey;
      const uMin = f * Math.min(xa / zlo, xa / zhi) + cx;
      const uMax = f * Math.max(xb / zlo, xb / zhi) + cx;
      const vMin = f * Math.min(ya / zlo, ya / zhi) + cy;
      const vMax = f * Math.max(yb / zlo, yb / zhi) + cy;
      if (uMax < -0.5 || uMin > W - 0.5 || vMax < -0.5 || vMin > H - 0.5) continue;
      const x0 = Math.max(0, Math.ceil(uMin));
      const x1 = Math.min(W - 1, Math.floor(uMax));
      const y0 = Math.max(0, Math.ceil(vMin));
      const y1 = Math.min(H - 1, Math.floor(vMax));
      for (let py = y0; py <= y1; py++) {
        const dy = (py - cy) * invF;
        const row = py * W;
        for (let px = x0; px <= x1; px++) {
          const dx = (px - cx) * invF;
          const den = nx * dx + ny * dy + nz;
          if (den > -1e-6 && den < 1e-6) continue;
          const sz = dn / den;
          if (!(sz > near)) continue;
          const ex = sz * dx - xc;
          const ey = sz * dy - yc;
          const ez = sz - zc;
          if (ex * ex + ey * ey + ez * ez > r2) continue;
          const k = row + px;
          const cur = depth[k];
          if (cur > 0 && cur <= sz) continue;
          depth[k] = sz;
          // facing the camera: n · ray < 0
          const sg = den > 0 ? -1 : 1;
          normal[3 * k] = sg * nx;
          normal[3 * k + 1] = sg * ny;
          normal[3 * k + 2] = sg * nz;
          setId[k] = s;
        }
      }
    }
  }
  return { width: W, height: H, depth, normal, setId };
}

/**
 * Resample an exact dataset depth map (z-depth at the pixel centres of the DISTORTED
 * full-resolution frame, 0 = none) into the undistorted pinhole camera `camU`. Bilinear
 * where the 4 neighbours are valid and within 2 % of each other, nearest otherwise.
 */
export function undistortGtDepth(src: Float32Array, cam0: CameraIntrinsics, camU: CameraIntrinsics): Float32Array {
  const W0 = cam0.width;
  const H0 = cam0.height;
  const W = camU.width;
  const H = camU.height;
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const yn = (y - camU.cy) / camU.f;
    for (let x = 0; x < W; x++) {
      const xn = (x - camU.cx) / camU.f;
      const r2 = xn * xn + yn * yn;
      const d = 1 + cam0.k1 * r2 + cam0.k2 * r2 * r2;
      const u = cam0.f * d * xn + cam0.cx;
      const v = cam0.f * d * yn + cam0.cy;
      if (!(u >= -0.5 && v >= -0.5 && u <= W0 - 0.5 && v <= H0 - 0.5)) continue;
      const ix = Math.floor(u);
      const iy = Math.floor(v);
      const fx = u - ix;
      const fy = v - iy;
      const xa = Math.max(0, ix), xb = Math.min(W0 - 1, ix + 1);
      const ya = Math.max(0, iy), yb = Math.min(H0 - 1, iy + 1);
      const a = src[ya * W0 + xa], b = src[ya * W0 + xb], c = src[yb * W0 + xa], e = src[yb * W0 + xb];
      const mn = Math.min(a, b, c, e);
      const mx = Math.max(a, b, c, e);
      if (mn > 0 && mx <= 1.02 * mn) {
        out[y * W + x] = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + e * fx) * fy;
      } else {
        const nx = Math.min(W0 - 1, Math.max(0, Math.round(u)));
        const ny = Math.min(H0 - 1, Math.max(0, Math.round(v)));
        out[y * W + x] = src[ny * W0 + nx];
      }
    }
  }
  return out;
}

/** Camera-frame normals (facing the camera) from a depth map by central differences. */
export function normalsFromDepth(cam: CameraIntrinsics, depth: Float32Array): Float32Array {
  const W = cam.width;
  const H = cam.height;
  const out = new Float32Array(3 * W * H);
  const P = (x: number, y: number, o: number[]): boolean => {
    const z = depth[y * W + x];
    if (!(z > 0)) return false;
    o[0] = ((x - cam.cx) / cam.f) * z;
    o[1] = ((y - cam.cy) / cam.f) * z;
    o[2] = z;
    return true;
  };
  const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0], d = [0, 0, 0], p = [0, 0, 0];
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      if (!P(x, y, p)) continue;
      if (!P(x - 1, y, a) || !P(x + 1, y, b) || !P(x, y - 1, c) || !P(x, y + 1, d)) continue;
      const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
      const vx = d[0] - c[0], vy = d[1] - c[1], vz = d[2] - c[2];
      let nx = uy * vz - uz * vy;
      let ny = uz * vx - ux * vz;
      let nz = ux * vy - uy * vx;
      const l = Math.hypot(nx, ny, nz);
      if (!(l > 0)) continue;
      nx /= l;
      ny /= l;
      nz /= l;
      if (nx * p[0] + ny * p[1] + nz * p[2] > 0) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
      }
      const k = 3 * (y * W + x);
      out[k] = nx;
      out[k + 1] = ny;
      out[k + 2] = nz;
    }
  }
  return out;
}

/**
 * True when world point X is visible in a view whose GT surfel depth map is `gt`:
 * projects inside the image and is not more than `tol` behind the z-buffer.
 */
export function visibleIn(cam: CameraIntrinsics, pose: Pose, gt: GtDepthMap, X: number, Y: number, Z: number, relTol: number, absTol: number): boolean {
  const R = pose.R;
  const t = pose.t;
  const zc = R[6] * X + R[7] * Y + R[8] * Z + t[2];
  if (!(zc > 0)) return false;
  const u = Math.round((cam.f * (R[0] * X + R[1] * Y + R[2] * Z + t[0])) / zc + cam.cx);
  const v = Math.round((cam.f * (R[3] * X + R[4] * Y + R[5] * Z + t[1])) / zc + cam.cy);
  if (u < 0 || v < 0 || u >= cam.width || v >= cam.height) return false;
  const zb = gt.depth[v * cam.width + u];
  if (!(zb > 0)) return false;
  return zc - zb <= Math.max(relTol * zc, absTol);
}

export interface SparseFromGtResult {
  xyz: Float64Array;
  visibility: number[][];
  /** Candidates drawn, and points kept (seen by ≥ 2 views). */
  drawn: number;
  kept: number;
  meanTrackLength: number;
}

/**
 * Sparse "SfM" points for source-view selection: `count` random GT samples (all sets,
 * honouring keep masks, proportional to set size × `weights`), each kept when it is
 * visible (surfel z-buffer test) in at least 2 of the views.
 */
export function sparsePointsFromGt(
  sets: GtSurfelSet[],
  views: { id: number; cam: CameraIntrinsics; pose: Pose }[],
  gtMaps: GtDepthMap[],
  opts: { count: number; seed?: number; radiusFactor?: number; relTol?: number },
): SparseFromGtResult {
  const rng = new Rng(opts.seed ?? 7);
  const kR = opts.radiusFactor ?? 1.3;
  const relTol = opts.relTol ?? 0.01;
  const pools: { s: number; idx: Int32Array }[] = [];
  let total = 0;
  for (let s = 0; s < sets.length; s++) {
    const S = sets[s];
    const n = S.points.length / 3;
    const idx: number[] = [];
    for (let i = 0; i < n; i++) if (!S.keep || S.keep[i]) idx.push(i);
    pools.push({ s, idx: Int32Array.from(idx) });
    total += idx.length;
  }
  const xyz: number[] = [];
  const visibility: number[][] = [];
  let lenSum = 0;
  for (const { s, idx } of pools) {
    const S = sets[s];
    const m = Math.min(idx.length, Math.round((opts.count * idx.length) / Math.max(1, total)));
    const pick = rng.sample(idx.length, m);
    const absTol = 1.5 * kR * S.spacing;
    for (let k = 0; k < m; k++) {
      const i = idx[pick[k]];
      const X = S.points[3 * i], Y = S.points[3 * i + 1], Z = S.points[3 * i + 2];
      const vis: number[] = [];
      for (let v = 0; v < views.length; v++) {
        if (visibleIn(views[v].cam, views[v].pose, gtMaps[v], X, Y, Z, relTol, absTol)) vis.push(views[v].id);
      }
      if (vis.length >= 2) {
        xyz.push(X, Y, Z);
        visibility.push(vis);
        lenSum += vis.length;
      }
    }
  }
  return {
    xyz: Float64Array.from(xyz),
    visibility,
    drawn: Math.min(opts.count, total),
    kept: visibility.length,
    meanTrackLength: visibility.length ? lenSum / visibility.length : 0,
  };
}

/**
 * Per-sample visibility in the given views (count, saturating at 255) for the
 * completeness set: lets the scorer restrict completeness to what the subset could see.
 */
export function subsetVisibility(
  set: GtSurfelSet,
  views: { cam: CameraIntrinsics; pose: Pose }[],
  gtMaps: GtDepthMap[],
  opts: { radiusFactor?: number; relTol?: number } = {},
): Uint8Array {
  const n = set.points.length / 3;
  const out = new Uint8Array(n);
  const absTol = 1.5 * (opts.radiusFactor ?? 1.3) * set.spacing;
  const relTol = opts.relTol ?? 0.01;
  for (let v = 0; v < views.length; v++) {
    const { cam, pose } = views[v];
    const gt = gtMaps[v];
    for (let i = 0; i < n; i++) {
      if (out[i] === 255) continue;
      if (visibleIn(cam, pose, gt, set.points[3 * i], set.points[3 * i + 1], set.points[3 * i + 2], relTol, absTol)) out[i]++;
    }
  }
  return out;
}
