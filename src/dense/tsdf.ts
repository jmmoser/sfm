/**
 * TSDF meshing (ARCHITECTURE §4.6): sparse voxel-block hashing (8³ blocks), projective
 * TSDF integration of depth maps, marching cubes, small-component removal.
 *
 * Pipeline of `tsdfMesh` / `buildTsdfVolume`:
 *  1. Scene scale: depths sampled from (up to 16) depth maps → median depth z̃ and median
 *     focal length f̃ (depth-map pixels). Auto voxel size = autoVoxelPixels · z̃ / f̃
 *     (≈ 2 pixel footprints at the median depth), clamped to [1e-4, 0.05] · z̃.
 *  2. Allocation pass over all depth maps: every valid pixel (on a stride grid finer than
 *     a third of a block footprint) inserts the blocks along its truncation band
 *     [d − τ, d + τ] on the viewing ray. This runs simultaneously for a ladder of voxel
 *     sizes (×√2 per level); levels whose block count exceeds `maxBlocks` stop early and
 *     the finest level that fits is used (automatic coarsening: the same code works for a
 *     0.5 m object and a 50 m yard within a fixed memory budget). All blocks are known
 *     before integration, so free-space observations of every view also carve blocks that
 *     were allocated by other views (outlier removal).
 *  3. Integration pass: for each depth map, blocks are culled against the frustum and a
 *     16 × 16-pixel tile max-depth map (blocks entirely behind the observed surface are
 *     skipped); each voxel is projected to its nearest pixel and updated with
 *       sdf = n·(X − P)  (point-to-plane, P = pixel's surface point, n its normal; falls back
 *       to the ray distance (d − z)·|r| for missing / grazing normals),
 *       τ_obs = max(τ, adaptiveTruncationPx · d / f)  (truncation never thinner than the
 *       observation's footprint-scaled noise, so far surfaces do not fragment),
 *       tsdf ← running weighted mean of clamp(sdf/τ_obs, ·, 1) for sdf ≥ −τ_obs,
 *       w = (z̃ / d)² · cos θ  (depth-dependent weight, normalised to 1 at the median depth),
 *       colour ← running mean (separate colour weight) for |sdf| < τ_obs.
 *  4. Marching cubes (`marching-cubes.ts`), then components smaller than
 *     `minComponentFraction` of the largest are removed.
 *
 * Memory: 11 B per voxel (Int16 TSDF, Uint16 weight, 3 × Uint16 colour, Uint8 colour
 * weight) = 5.6 KB per block; default maxBlocks 50 000 ≈ 280 MB worst case.
 */
import type { CameraIntrinsics, CancelToken, Pose, ProgressCallback, TriangleMesh } from '../core/types';
import { DepthMapCache, Int3HashMap, Yielder, poseCentre, quantile, rgbAtResolution, scaleCamera } from './dense-util';
import {
  BLOCK_SIZE,
  BLOCK_VOXELS,
  COLOR_SCALE,
  TSDF_SCALE,
  WEIGHT_UNIT,
  marchingCubes,
  removeSmallComponents,
  type SparseVoxelVolume,
} from './marching-cubes';
import type { DenseView, DepthMap, DepthMapStore } from './types';

export interface TsdfOptions {
  /** Voxel edge length (world units). Default: automatic (see file header). */
  voxelSize?: number;
  /** Truncation distance in VOXELS. Default 4. */
  truncation?: number;
  /** Depths beyond this are ignored (world units). Default Infinity. */
  maxDepth?: number;
  /** Block budget (8³ voxels each, ≈ 5.6 KB). Default 50 000. */
  maxBlocks?: number;
  /** Auto voxel size in pixel footprints at the median depth. Default 2. */
  autoVoxelPixels?: number;
  /** τ_obs = max(τ, k · depth / f). Default 2 (0 disables). */
  adaptiveTruncationPx?: number;
  /** Signed distance: point-to-plane (uses depth-map normals) or along the ray. Default 'plane'. */
  sdfMode?: 'plane' | 'ray';
  /** Minimum accumulated weight (nominal observations) of a voxel used by marching cubes. Default 0. */
  minWeight?: number;
  /** Weight cap (nominal observations; turns the mean into a moving average). Default 256. */
  maxWeight?: number;
  /** Remove components with fewer triangles than this fraction of the largest. Default 0.02 (0 disables). */
  minComponentFraction?: number;
  /** Accumulate voxel colours. Default true. */
  color?: boolean;
  /** Depth pixels with a photometric cost above this are ignored. Default 2. */
  maxCost?: number;
  /** Receives statistics when meshing finishes. */
  onStats?: (s: TsdfStats) => void;
}

export interface TsdfStats {
  voxelSize: number;
  /** Base truncation distance (world units). */
  truncation: number;
  /** Automatic voxel size before coarsening (NaN if given). */
  autoVoxelSize: number;
  /** Coarsening levels applied (×√2 each). */
  coarsenLevels: number;
  blocks: number;
  /** True if the block budget was reached (only possible with a user-given voxel size). */
  blockCapHit: boolean;
  medianDepth: number;
  vertices: number;
  triangles: number;
  removedComponents: number;
  allocateMs: number;
  integrateMs: number;
  meshMs: number;
  totalMs: number;
  volumeBytes: number;
}

const DEFAULTS = {
  truncation: 4,
  maxDepth: Infinity,
  maxBlocks: 50_000,
  autoVoxelPixels: 2,
  adaptiveTruncationPx: 2,
  sdfMode: 'plane' as 'plane' | 'ray',
  minWeight: 0,
  maxWeight: 256,
  minComponentFraction: 0.02,
  color: true,
  maxCost: 2,
};

/** Minimum cos(viewing angle) for the point-to-plane distance (else the ray distance is used). */
const MIN_PLANE_COS = 0.25;
/** Normalised depth behind the surface down to which observations keep full weight. */
const BEHIND_FULL = 0.25;
/** Voxel-size ladder for automatic coarsening: up to √2^15 ≈ 181 × the auto voxel. */
const MAX_LEVELS = 16;
/** Levels counted per allocation pass. */
const LEVEL_WINDOW = 4;

// ---------------------------------------------------------------------------
// Volume

/** Sparse TSDF volume of 8³ voxel blocks (see marching-cubes.ts for the voxel/world convention). */
export class TsdfVolume implements SparseVoxelVolume {
  blockCount = 0;
  blockCoords: Int32Array;
  tsdf: Int16Array;
  weight: Uint16Array;
  color: Uint16Array | null;
  colorWeight: Uint8Array | null;
  /** True once an allocation was refused because of `maxBlocks`. */
  capHit = false;
  private capacity: number;
  private readonly map: Int3HashMap;

  constructor(
    readonly voxelSize: number,
    /** Base truncation distance (world units). */
    readonly truncation: number,
    readonly maxBlocks: number,
    withColor = true,
    initialBlocks = 1024,
  ) {
    if (!(voxelSize > 0)) throw new Error('TsdfVolume: voxelSize must be > 0');
    this.capacity = Math.max(16, Math.min(maxBlocks, initialBlocks));
    this.map = new Int3HashMap(this.capacity);
    this.blockCoords = new Int32Array(3 * this.capacity);
    this.tsdf = new Int16Array(BLOCK_VOXELS * this.capacity);
    this.weight = new Uint16Array(BLOCK_VOXELS * this.capacity);
    this.color = withColor ? new Uint16Array(3 * BLOCK_VOXELS * this.capacity) : null;
    this.colorWeight = withColor ? new Uint8Array(BLOCK_VOXELS * this.capacity) : null;
  }

  findBlock(bx: number, by: number, bz: number): number {
    return this.map.get(bx, by, bz);
  }

  /** Index of the block (allocating it if needed); −1 when the budget is exhausted. */
  allocateBlock(bx: number, by: number, bz: number): number {
    const found = this.map.get(bx, by, bz);
    if (found >= 0) return found;
    if (this.blockCount >= this.maxBlocks) {
      this.capHit = true;
      return -1;
    }
    if (this.blockCount >= this.capacity) this.grow();
    const b = this.blockCount++;
    this.map.getOrInsert(bx, by, bz, b);
    this.blockCoords[3 * b] = bx;
    this.blockCoords[3 * b + 1] = by;
    this.blockCoords[3 * b + 2] = bz;
    return b;
  }

  private grow(): void {
    const cap = Math.min(this.maxBlocks, Math.max(this.capacity * 2, 16));
    const bc = new Int32Array(3 * cap);
    bc.set(this.blockCoords);
    this.blockCoords = bc;
    const ts = new Int16Array(BLOCK_VOXELS * cap);
    ts.set(this.tsdf);
    this.tsdf = ts;
    const w = new Uint16Array(BLOCK_VOXELS * cap);
    w.set(this.weight);
    this.weight = w;
    if (this.color) {
      const c = new Uint16Array(3 * BLOCK_VOXELS * cap);
      c.set(this.color);
      this.color = c;
      const cw = new Uint8Array(BLOCK_VOXELS * cap);
      cw.set(this.colorWeight!);
      this.colorWeight = cw;
    }
    this.capacity = cap;
  }

  /** Memory used by the voxel arrays and the hash (bytes). */
  bytes(): number {
    return (
      this.blockCoords.byteLength +
      this.tsdf.byteLength +
      this.weight.byteLength +
      (this.color?.byteLength ?? 0) +
      (this.colorWeight?.byteLength ?? 0) +
      this.map.bytes()
    );
  }

  /**
   * Volume sampled from an analytic signed distance function (positive outside), for
   * tests and debugging: blocks within reach of the surface over [min, max] are allocated
   * and every voxel gets clamp(sdf/τ, −1, 1) with weight 1.
   */
  static fromFunction(
    sdf: (x: number, y: number, z: number) => number,
    min: ArrayLike<number>,
    max: ArrayLike<number>,
    voxelSize: number,
    truncationVoxels = 4,
    color?: (x: number, y: number, z: number) => [number, number, number],
  ): TsdfVolume {
    const tau = truncationVoxels * voxelSize;
    const vol = new TsdfVolume(voxelSize, tau, 1 << 30, !!color);
    const bw = BLOCK_SIZE * voxelSize;
    const lo = [0, 1, 2].map((a) => Math.floor((min[a] / voxelSize + 0.5) / BLOCK_SIZE));
    const hi = [0, 1, 2].map((a) => Math.floor((max[a] / voxelSize + 0.5) / BLOCK_SIZE));
    const reach = tau + bw;
    for (let bz = lo[2]; bz <= hi[2]; bz++) {
      for (let by = lo[1]; by <= hi[1]; by++) {
        for (let bx = lo[0]; bx <= hi[0]; bx++) {
          const cx = (BLOCK_SIZE * bx + 3.5) * voxelSize;
          const cy = (BLOCK_SIZE * by + 3.5) * voxelSize;
          const cz = (BLOCK_SIZE * bz + 3.5) * voxelSize;
          if (Math.abs(sdf(cx, cy, cz)) > reach) continue;
          const b = vol.allocateBlock(bx, by, bz);
          for (let lz = 0; lz < 8; lz++) {
            for (let ly = 0; ly < 8; ly++) {
              for (let lx = 0; lx < 8; lx++) {
                const x = (BLOCK_SIZE * bx + lx) * voxelSize;
                const y = (BLOCK_SIZE * by + ly) * voxelSize;
                const z = (BLOCK_SIZE * bz + lz) * voxelSize;
                const v = Math.max(-1, Math.min(1, sdf(x, y, z) / tau));
                const vi = b * BLOCK_VOXELS + lx + 8 * (ly + 8 * lz);
                vol.tsdf[vi] = Math.round(v * TSDF_SCALE);
                vol.weight[vi] = WEIGHT_UNIT;
                if (color && vol.color) {
                  const c = color(x, y, z);
                  vol.color[3 * vi] = Math.round(c[0] * COLOR_SCALE);
                  vol.color[3 * vi + 1] = Math.round(c[1] * COLOR_SCALE);
                  vol.color[3 * vi + 2] = Math.round(c[2] * COLOR_SCALE);
                  vol.colorWeight![vi] = 16;
                }
              }
            }
          }
        }
      }
    }
    return vol;
  }
}

// ---------------------------------------------------------------------------
// Per-view geometry

interface ViewGeom {
  cam: CameraIntrinsics; // at depth-map resolution
  R: Float64Array;
  t: Float64Array;
  C: Float64Array;
}

function viewGeom(view: DenseView, dm: DepthMap): ViewGeom {
  return { cam: scaleCamera(view.cam, dm.width, dm.height), R: view.pose.R, t: view.pose.t, C: poseCentre(view.pose) };
}

/** Low percentile of valid depths (allocation stride) — sampled. */
function lowDepth(dm: DepthMap, maxDepth: number, maxCost: number): number {
  const n = dm.width * dm.height;
  const step = Math.max(1, Math.floor(n / 4096));
  const s: number[] = [];
  for (let i = 0; i < n; i += step) {
    const d = dm.depth[i];
    if (d > 0 && d <= maxDepth && !(dm.cost && dm.cost[i] > maxCost)) s.push(d);
  }
  return s.length ? quantile(s, 0.02) : NaN;
}

/**
 * Insert the blocks covering the truncation bands of a depth map's pixels into `insert`.
 * Returns false when `insert` reported overflow.
 */
function allocateBands(
  dm: DepthMap,
  g: ViewGeom,
  voxelSize: number,
  tau: number,
  adaptivePx: number,
  maxDepth: number,
  maxCost: number,
  zLow: number,
  insert: (bx: number, by: number, bz: number) => boolean,
): boolean {
  const { cam, R, C } = g;
  const W = dm.width;
  const H = dm.height;
  const bw = BLOCK_SIZE * voxelSize;
  // pixel stride: footprint at the near depth ≤ a third of a block
  const stride = Math.max(1, Math.min(16, Math.floor((bw * cam.f) / (3 * Math.max(zLow, 1e-12)))));
  const invV = 1 / voxelSize;
  let lastX = 0x7fffffff;
  let lastY = 0;
  let lastZ = 0;
  for (let y = 0; y < H; y += stride) {
    const ry = (y - cam.cy) / cam.f;
    for (let x = 0; x < W; x += stride) {
      const i = y * W + x;
      const d = dm.depth[i];
      if (!(d > 0) || d > maxDepth) continue;
      if (dm.cost && dm.cost[i] > maxCost) continue;
      const rx = (x - cam.cx) / cam.f;
      const L = Math.sqrt(rx * rx + ry * ry + 1);
      const tauObs = Math.max(tau, (adaptivePx * d) / cam.f);
      // world ray direction for unit camera z: Rᵀ (rx, ry, 1)
      const dx = R[0] * rx + R[3] * ry + R[6];
      const dy = R[1] * rx + R[4] * ry + R[7];
      const dz = R[2] * rx + R[5] * ry + R[8];
      const z0 = d - tauObs / L;
      const z1 = d + tauObs / L;
      const n = Math.max(2, Math.ceil(((z1 - z0) * L) / (0.5 * bw)) + 1);
      for (let s = 0; s < n; s++) {
        const z = z0 + ((z1 - z0) * s) / (n - 1);
        if (z <= 0) continue;
        const bx = Math.floor(((C[0] + z * dx) * invV + 0.5) / BLOCK_SIZE);
        const by = Math.floor(((C[1] + z * dy) * invV + 0.5) / BLOCK_SIZE);
        const bz = Math.floor(((C[2] + z * dz) * invV + 0.5) / BLOCK_SIZE);
        if (bx === lastX && by === lastY && bz === lastZ) continue;
        lastX = bx;
        lastY = by;
        lastZ = bz;
        if (!insert(bx, by, bz)) return false;
      }
    }
  }
  return true;
}

const TILE = 16;
/** Interleaved per-pixel scratch of `integrateDepthMap` (grow-only, reused across views). */
let pixScratch = new Float32Array(0);

/** Integrate one depth map into the volume (all allocated blocks; see file header). */
export function integrateDepthMap(
  vol: TsdfVolume,
  dm: DepthMap,
  cam: CameraIntrinsics,
  pose: Pose,
  rgb: Uint8Array | null,
  params: { truncation: number; adaptiveTruncationPx: number; sdfMode: 'plane' | 'ray'; refDepth: number; maxDepth: number; maxCost: number; maxWeight: number },
): number {
  const W = dm.width;
  const H = dm.height;
  const f = cam.f;
  const cx = cam.cx;
  const cy = cam.cy;
  const R = pose.R;
  const t = pose.t;
  const vs = vol.voxelSize;
  const tau = params.truncation;
  const kPx = params.adaptiveTruncationPx;
  const planeMode = params.sdfMode === 'plane';
  const zRef2 = params.refDepth * params.refDepth;
  const maxDepth = params.maxDepth;
  const maxCost = params.maxCost;
  const maxW = Math.min(65535, Math.round(params.maxWeight * WEIGHT_UNIT));
  const depth = dm.depth;
  const normal = dm.normal;
  const cost = dm.cost;

  // Interleaved per-pixel record (d, nx, ny, nz) with validity folded into d (0 = unusable):
  // a voxel lookup then touches one cache line instead of three arrays. Plus the tile
  // max-depth map (0 = no valid depth) for block culling.
  const n4 = 4 * W * H;
  if (pixScratch.length < n4) pixScratch = new Float32Array(n4);
  const pix = pixScratch;
  const tw = Math.ceil(W / TILE);
  const th = Math.ceil(H / TILE);
  const tileMax = new Float32Array(tw * th);
  for (let y = 0; y < H; y++) {
    const trow = ((y / TILE) | 0) * tw;
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      let d = depth[i];
      if (!(d > 0) || d > maxDepth || (cost && cost[i] > maxCost)) d = 0;
      pix[4 * i] = d;
      pix[4 * i + 1] = normal[3 * i];
      pix[4 * i + 2] = normal[3 * i + 1];
      pix[4 * i + 3] = normal[3 * i + 2];
      if (d === 0) continue;
      const ti = trow + ((x / TILE) | 0);
      if (d > tileMax[ti]) tileMax[ti] = d;
    }
  }

  const tsdf = vol.tsdf;
  const weight = vol.weight;
  const color = rgb ? vol.color : null;
  const colorW = rgb ? vol.colorWeight : null;
  const blockRadius = 0.5 * Math.sqrt(3) * BLOCK_SIZE * vs;
  // camera-frame increments per voxel step along world x, y, z (columns of R · vs)
  const axx = R[0] * vs;
  const axy = R[3] * vs;
  const axz = R[6] * vs;
  const ayx = R[1] * vs;
  const ayy = R[4] * vs;
  const ayz = R[7] * vs;
  const azx = R[2] * vs;
  const azy = R[5] * vs;
  const azz = R[8] * vs;
  let updated = 0;

  for (let b = 0; b < vol.blockCount; b++) {
    const gx0 = BLOCK_SIZE * vol.blockCoords[3 * b];
    const gy0 = BLOCK_SIZE * vol.blockCoords[3 * b + 1];
    const gz0 = BLOCK_SIZE * vol.blockCoords[3 * b + 2];
    // block centre in camera frame
    const wx = (gx0 + 3.5) * vs;
    const wy = (gy0 + 3.5) * vs;
    const wz = (gz0 + 3.5) * vs;
    const bcz = R[6] * wx + R[7] * wy + R[8] * wz + t[2];
    if (bcz + blockRadius <= 0) continue;
    const zNear = bcz - blockRadius;
    if (zNear > 0) {
      const bcx = R[0] * wx + R[1] * wy + R[2] * wz + t[0];
      const bcy = R[3] * wx + R[4] * wy + R[5] * wz + t[1];
      const u = (f * bcx) / bcz + cx;
      const v = (f * bcy) / bcz + cy;
      const rpx = (f * blockRadius) / zNear;
      const u0 = u - rpx;
      const u1 = u + rpx;
      const v0 = v - rpx;
      const v1 = v + rpx;
      if (u1 < -0.5 || v1 < -0.5 || u0 > W - 0.5 || v0 > H - 0.5) continue;
      const tx0 = Math.max(0, Math.floor((u0 + 0.5) / TILE));
      const tx1 = Math.min(tw - 1, Math.floor((u1 + 0.5) / TILE));
      const ty0 = Math.max(0, Math.floor((v0 + 0.5) / TILE));
      const ty1 = Math.min(th - 1, Math.floor((v1 + 0.5) / TILE));
      let dMax = 0;
      for (let ty = ty0; ty <= ty1; ty++) {
        for (let tx = tx0; tx <= tx1; tx++) {
          const m = tileMax[ty * tw + tx];
          if (m > dMax) dMax = m;
        }
      }
      if (dMax === 0) continue;
      const tauMax = Math.max(tau, (kPx * dMax) / f);
      if (zNear > dMax + tauMax) continue; // entirely behind the observed surface
    }

    // voxel (0,0,0) of the block in camera coordinates
    const ox = gx0 * vs;
    const oy = gy0 * vs;
    const oz = gz0 * vs;
    const c0x = R[0] * ox + R[1] * oy + R[2] * oz + t[0];
    const c0y = R[3] * ox + R[4] * oy + R[5] * oz + t[1];
    const c0z = R[6] * ox + R[7] * oy + R[8] * oz + t[2];
    const vbase = b * BLOCK_VOXELS;
    for (let lz = 0; lz < 8; lz++) {
      for (let ly = 0; ly < 8; ly++) {
        let X = c0x + ly * ayx + lz * azx;
        let Y = c0y + ly * ayy + lz * azy;
        let Z = c0z + ly * ayz + lz * azz;
        const rowBase = vbase + 8 * (ly + 8 * lz);
        for (let lx = 0; lx < 8; lx++, X += axx, Y += axy, Z += axz) {
          if (!(Z > 1e-9)) continue;
          const iz = 1 / Z;
          const u = f * X * iz + cx;
          const v = f * Y * iz + cy;
          if (!(u >= -0.5 && v >= -0.5 && u < W - 0.5 && v < H - 0.5)) continue;
          const px = (u + 0.5) | 0;
          const py = (v + 0.5) | 0;
          const q = py * W + px;
          const q4 = 4 * q;
          const d = pix[q4];
          if (d === 0) continue;
          const tauObs = kPx > 0 ? Math.max(tau, (kPx * d) / f) : tau;
          // only voxels at most τ behind the observed surface ALONG THE RAY are updated
          // (the point-to-plane distance alone would reach τ/cos θ deep on grazing rays,
          // through thin parts and around edges)
          if (Z - d > tauObs) continue;
          const rqx = (px - cx) / f;
          const rqy = (py - cy) / f;
          const L = Math.sqrt(rqx * rqx + rqy * rqy + 1);
          if ((Z - d) * L > tauObs) continue;
          let sdf: number;
          let cosT = 1;
          const nx = pix[q4 + 1];
          const ny = pix[q4 + 2];
          const nz = pix[q4 + 3];
          const nr = nx * rqx + ny * rqy + nz;
          if (planeMode && -nr >= MIN_PLANE_COS * L) {
            sdf = nx * X + ny * Y + nz * Z - d * nr;
            cosT = -nr / L;
          } else {
            sdf = (d - Z) * L;
            if (nx !== 0 || ny !== 0 || nz !== 0) cosT = Math.max(0.1, Math.abs(nr) / L);
          }
          if (sdf < -tauObs) continue;
          const tv = sdf >= tauObs ? 1 : sdf / tauObs;
          let wObs = (zRef2 / (d * d)) * cosT;
          if (wObs > 64) wObs = 64;
          let wq = (wObs * WEIGHT_UNIT + 0.5) | 0;
          if (wq < 1) wq = 1;
          // linear weight fall-off behind the surface (full weight down to −τ/4, 0 at −τ):
          // reduces bulging at edges / thin parts without biasing the zero crossing
          if (tv < -BEHIND_FULL) {
            wq = ((wq * (1 + tv)) / (1 - BEHIND_FULL) + 0.5) | 0;
            if (wq < 1) continue;
          }
          const vi = rowBase + lx;
          const w0 = weight[vi];
          const wn = w0 + wq;
          tsdf[vi] = Math.round((tsdf[vi] * w0 + tv * TSDF_SCALE * wq) * (1 / wn));
          weight[vi] = wn > maxW ? maxW : wn;
          updated++;
          if (color && sdf < tauObs) {
            // colour weight in 1/16 nominal observations, saturating (moving average)
            const cw0 = colorW![vi];
            let cwq = (wObs * 16 + 0.5) | 0;
            if (cwq < 1) cwq = 1;
            const cwn = cw0 + cwq;
            const icw = 1 / cwn;
            const a0 = cw0 * icw;
            const a1 = COLOR_SCALE * cwq * icw;
            const c3 = 3 * vi;
            const p3 = 3 * q;
            // non-negative: (x + 0.5) | 0 rounds
            color[c3] = (color[c3] * a0 + rgb![p3] * a1 + 0.5) | 0;
            color[c3 + 1] = (color[c3 + 1] * a0 + rgb![p3 + 1] * a1 + 0.5) | 0;
            color[c3 + 2] = (color[c3 + 2] * a0 + rgb![p3 + 2] * a1 + 0.5) | 0;
            colorW![vi] = cwn > 255 ? 255 : cwn;
          }
        }
      }
    }
  }
  return updated;
}

// ---------------------------------------------------------------------------
// Driver

/** Build and integrate the TSDF volume (steps 1–3 of the file header). */
export async function buildTsdfVolume(
  views: DenseView[],
  store: DepthMapStore,
  options: TsdfOptions = {},
  progress?: ProgressCallback,
  cancel?: CancelToken,
  stats?: Partial<TsdfStats>,
): Promise<TsdfVolume> {
  const o = { ...DEFAULTS, ...stripUndefined(options) };
  const yielder = new Yielder('tsdfMesh', cancel);
  const cache = new DepthMapCache(store, 2);
  const tA = performance.now();

  // 1. scene scale from a subset of views
  const sampleIdx: number[] = [];
  const nSample = Math.min(views.length, 16);
  for (let k = 0; k < nSample; k++) sampleIdx.push(Math.floor(((k + 0.5) * views.length) / nSample));
  const depthSamples: number[] = [];
  const fs: number[] = [];
  for (const vi of sampleIdx) {
    await yielder.tick();
    const dm = await cache.get(views[vi].id);
    if (!dm) continue;
    const n = dm.width * dm.height;
    const step = Math.max(1, Math.floor(n / 2048));
    let any = false;
    for (let i = 0; i < n; i += step) {
      const d = dm.depth[i];
      if (d > 0 && d <= o.maxDepth && !(dm.cost && dm.cost[i] > o.maxCost)) {
        depthSamples.push(d);
        any = true;
      }
    }
    if (any) fs.push(scaleCamera(views[vi].cam, dm.width, dm.height).f);
  }
  if (depthSamples.length === 0) {
    // maybe the sampled views were empty; scan all
    for (const v of views) {
      const dm = await cache.get(v.id);
      if (!dm) continue;
      for (let i = 0; i < dm.depth.length; i += 7) {
        const d = dm.depth[i];
        if (d > 0 && d <= o.maxDepth) depthSamples.push(d);
      }
      fs.push(scaleCamera(v.cam, dm.width, dm.height).f);
      if (depthSamples.length > 1000) break;
    }
  }
  if (depthSamples.length === 0) {
    // nothing to integrate: empty volume (→ empty mesh)
    const empty = new TsdfVolume(1, 4, 1, false, 16);
    if (stats) Object.assign(stats, { voxelSize: NaN, truncation: NaN, autoVoxelSize: NaN, coarsenLevels: 0, blocks: 0, blockCapHit: false, medianDepth: NaN, allocateMs: 0, integrateMs: 0, volumeBytes: empty.bytes() });
    return empty;
  }
  const zMed = quantile(depthSamples, 0.5);
  const fMed = quantile(fs, 0.5);
  const autoVoxel = Math.min(0.05 * zMed, Math.max(1e-4 * zMed, (o.autoVoxelPixels * zMed) / fMed));
  const userVoxel = options.voxelSize !== undefined && options.voxelSize > 0;
  const voxel0 = userVoxel ? options.voxelSize! : autoVoxel;
  const truncVox = Math.max(1, o.truncation);

  // 2. allocation pass over a ladder of voxel sizes (×√2 per level; each level halves the
  //    number of surface blocks). Levels are counted in windows of LEVEL_WINDOW; if every
  //    level of a window overflows, the next window is counted in a new pass.
  const maxLevels = userVoxel ? 1 : MAX_LEVELS;
  const budget = Math.max(1, Math.floor(o.maxBlocks));
  const cap = userVoxel ? Infinity : budget;
  let level = -1;
  let chosenMap: Int3HashMap | null = null;
  for (let base = 0; base < maxLevels && level < 0; base += LEVEL_WINDOW) {
    const nL = Math.min(LEVEL_WINDOW, maxLevels - base);
    const levelVoxel: number[] = [];
    for (let l = 0; l < nL; l++) levelVoxel.push(voxel0 * Math.pow(Math.SQRT2, base + l));
    const levelMaps = levelVoxel.map(() => new Int3HashMap(1024));
    const levelOverflow = new Uint8Array(nL);
    for (let k = 0; k < views.length; k++) {
      progress?.({ stage: 'tsdf', progress: 0.3 * (k / views.length), message: `allocating ${k + 1}/${views.length}` });
      await yielder.tick();
      const view = views[k];
      const dm = await cache.get(view.id);
      if (!dm) continue;
      const g = viewGeom(view, dm);
      const zLow = lowDepth(dm, o.maxDepth, o.maxCost);
      if (!(zLow > 0)) continue;
      for (let l = 0; l < nL; l++) {
        if (levelOverflow[l]) continue;
        const map = levelMaps[l];
        const ok = allocateBands(dm, g, levelVoxel[l], truncVox * levelVoxel[l], o.adaptiveTruncationPx, o.maxDepth, o.maxCost, zLow, (bx, by, bz) => {
          map.getOrInsert(bx, by, bz, map.size);
          return map.size <= cap;
        });
        if (!ok) {
          levelOverflow[l] = 1;
          levelMaps[l] = new Int3HashMap(16); // release
        }
      }
      if (levelOverflow.every((x) => x === 1)) break;
    }
    for (let l = 0; l < nL; l++) {
      if (!levelOverflow[l]) {
        level = base + l;
        chosenMap = levelMaps[l];
        break;
      }
    }
  }
  const overflowAll = level < 0;
  if (overflowAll) level = maxLevels - 1;
  const voxel = voxel0 * Math.pow(Math.SQRT2, level);
  const tau = truncVox * voxel;
  const vol = new TsdfVolume(voxel, tau, budget, o.color, chosenMap ? Math.max(16, chosenMap.size) : budget);
  if (overflowAll || !chosenMap) {
    // even the coarsest level overflowed: allocate greedily up to the budget during a second pass
    for (const view of views) {
      await yielder.tick();
      const dm = await cache.get(view.id);
      if (!dm) continue;
      const zLow = lowDepth(dm, o.maxDepth, o.maxCost);
      if (!(zLow > 0)) continue;
      allocateBands(dm, viewGeom(view, dm), voxel, tau, o.adaptiveTruncationPx, o.maxDepth, o.maxCost, zLow, (bx, by, bz) => vol.allocateBlock(bx, by, bz) >= 0 || true);
    }
  } else {
    // re-create the blocks of the chosen level (insertion order: views in sequence)
    const keys = collectKeys(chosenMap);
    chosenMap = null;
    for (let i = 0; i < keys.length; i += 3) vol.allocateBlock(keys[i], keys[i + 1], keys[i + 2]);
  }
  const tB = performance.now();

  // 3. integration
  for (let k = 0; k < views.length; k++) {
    progress?.({ stage: 'tsdf', progress: 0.3 + 0.6 * (k / views.length), message: `integrating ${k + 1}/${views.length}` });
    await yielder.tick();
    const view = views[k];
    const dm = await cache.get(view.id);
    if (!dm) continue;
    const g = viewGeom(view, dm);
    const rgb = o.color ? rgbAtResolution(view.image, dm.width, dm.height) : null;
    integrateDepthMap(vol, dm, g.cam, view.pose, rgb, {
      truncation: tau,
      adaptiveTruncationPx: o.adaptiveTruncationPx,
      sdfMode: o.sdfMode,
      refDepth: zMed,
      maxDepth: o.maxDepth,
      maxCost: o.maxCost,
      maxWeight: o.maxWeight,
    });
  }
  const tC = performance.now();
  if (stats) {
    stats.voxelSize = voxel;
    stats.truncation = tau;
    stats.autoVoxelSize = userVoxel ? NaN : autoVoxel;
    stats.coarsenLevels = level;
    stats.blocks = vol.blockCount;
    stats.blockCapHit = vol.capHit;
    stats.medianDepth = zMed;
    stats.allocateMs = tB - tA;
    stats.integrateMs = tC - tB;
    stats.volumeBytes = vol.bytes();
  }
  return vol;
}

function stripUndefined<T extends object>(o: T): Partial<T> {
  const r: Partial<T> = {};
  for (const k of Object.keys(o) as (keyof T)[]) if (o[k] !== undefined) r[k] = o[k];
  return r;
}

/** Keys of an Int3HashMap in insertion order (value = insertion index). */
function collectKeys(map: Int3HashMap): Int32Array {
  const out = new Int32Array(3 * map.size);
  map.forEach((x, y, z, v) => {
    out[3 * v] = x;
    out[3 * v + 1] = y;
    out[3 * v + 2] = z;
  });
  return out;
}

/**
 * TSDF mesh of the depth maps (ARCHITECTURE §4.6): build + integrate the volume, marching
 * cubes, small-component removal. `truncation` is in voxels (default 4), `maxDepth` in
 * world units. Throws `DenseCancelledError` when cancelled.
 */
export async function tsdfMesh(
  views: DenseView[],
  store: DepthMapStore,
  opts: TsdfOptions = {},
  progress?: ProgressCallback,
  cancel?: CancelToken,
): Promise<TriangleMesh> {
  const t0 = performance.now();
  const stats: Partial<TsdfStats> = {};
  const vol = await buildTsdfVolume(views, store, opts, progress, cancel, stats);
  const tM = performance.now();
  progress?.({ stage: 'tsdf', progress: 0.9, message: 'marching cubes' });
  const yielder = new Yielder('tsdfMesh', cancel);
  await yielder.tick(true);
  let mesh = marchingCubes(vol, { minWeight: opts.minWeight ?? DEFAULTS.minWeight, color: opts.color ?? DEFAULTS.color });
  const frac = opts.minComponentFraction ?? DEFAULTS.minComponentFraction;
  let removed = 0;
  if (frac > 0) {
    const r = removeSmallComponents(mesh, frac);
    mesh = r.mesh;
    removed = r.removedComponents;
  }
  progress?.({ stage: 'tsdf', progress: 1, message: `${mesh.triangleCount} triangles` });
  stats.vertices = mesh.vertexCount;
  stats.triangles = mesh.triangleCount;
  stats.removedComponents = removed;
  stats.meshMs = performance.now() - tM;
  stats.totalMs = performance.now() - t0;
  opts.onStats?.(stats as TsdfStats);
  return mesh;
}
