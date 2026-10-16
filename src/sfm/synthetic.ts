/**
 * Deterministic synthetic SfM scenarios for tests (here and in other modules).
 *
 * Scenes (Z-up world, metres):
 *  - 'orbit': a small object (two boxes + a sphere, ~0.3 m) on a textured table,
 *    cameras on a circle of radius ~0.5 m looking at it (close range).
 *  - 'walk': a ~60 m yard (ground plane, houses = boxes, trees = spheres) with a
 *    walking path at eye height looking at structure 5–50 m away (large scale).
 * Points are sampled on the surfaces (with normals); visibility = in front, inside
 * the (distorted) image, surface facing the camera (< 82°) and not occluded by any
 * box / sphere. Keypoints are RADIAL projections + Gaussian pixel noise, plus clutter
 * features without a 3-D point. Descriptors are RootSIFT-like (per-point base vector
 * + per-observation noise, quantised as min(255, round(512 v))). Putative matches
 * between selected pairs contain true matches (some dropped) and a configurable
 * fraction of random outlier matches. Everything is driven by the seeded `Rng`.
 */
import type { CameraIntrinsics, FeatureSet, FrameInfo, MatchPairs, Pose, TwoViewGeometry } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';
import { estimateTwoViewGeometry, type TwoViewOptions } from '../geometry/twoview';
import { Rng } from '../math/random';
import { umeyama } from '../math/sim3';
import { rotationAngleBetween } from '../math/rotation';
import { FeatureDatabase } from './database';
import type { Reconstruction } from './reconstruction';

/**
 * 'orbit': object on a table, cameras on a circle (close range). 'walk': sideways S-walk
 * through a yard with panning (large scale). 'loop': closed walk (ellipse 22 × 12 m)
 * around the yard looking towards its centre with panning — the last images overlap the
 * first (loop closure). 'forward': straight walk towards the houses looking forward
 * (forward motion: the epipole is inside the image).
 */
export type SyntheticSceneKind = 'orbit' | 'walk' | 'loop' | 'forward';

export interface SyntheticSceneOptions {
  kind: SyntheticSceneKind;
  numImages: number;
  seed: number;
  width: number;
  height: number;
  /** Horizontal FOV (deg) across the image width. */
  hfovDeg: number;
  k1: number;
  k2: number;
  /** Gaussian keypoint noise σ (px). */
  noisePx: number;
  /** Fraction of the putative matches of a pair that are wrong (random). */
  outlierRatio: number;
  /** Fraction of true matches that are missing from the putative matches. */
  matchDropRate: number;
  /** Features without a 3-D point per image. */
  clutterPerImage: number;
  /** Number of surface points (default depends on the scene). */
  numPoints: number;
  /** Generate descriptors (default true; false = zero bytes, faster). */
  descriptors: boolean;
  /** Each image is paired with the next `pairWindow` images (if they share ≥ minSharedPoints). */
  pairWindow: number;
  /** Additional covisible pairs per image (most shared points first). */
  extraPairsPerImage: number;
  /** Minimum number of shared visible points for a pair to be matched. */
  minSharedPoints: number;
  /** Offset added to image ids. */
  idOffset: number;
  /** Translation of the whole scene (world), e.g. to separate combined scenes. */
  offset: [number, number, number];
}

export interface SyntheticPair {
  a: number;
  b: number;
  /** Putative matches (a-feature, b-feature), sorted by a-feature. */
  matches: MatchPairs;
  /** 1 per match if it is a true correspondence. */
  isTrue: Uint8Array;
}

export interface SyntheticScene {
  options: SyntheticSceneOptions;
  /** True intrinsics (shared by all images). */
  camera: CameraIntrinsics;
  imageIds: number[];
  /** Ground-truth world-to-camera poses, aligned with imageIds. */
  poses: Pose[];
  /** Surface points (3 per point) and colours. */
  points: Float64Array;
  pointColors: Uint8Array;
  frames: FrameInfo[];
  features: FeatureSet[];
  /** Per-feature sRGB colours (3 bytes per feature). */
  colors: Uint8Array[];
  /** Point index of each feature (−1 = clutter). */
  featurePoint: Int32Array[];
  pairs: SyntheticPair[];
  /** Bounding-box diagonal of camera centres ∪ observed points. */
  extent: number;
}

const DEFAULTS: Omit<SyntheticSceneOptions, 'kind'> = {
  numImages: 60,
  seed: 1,
  width: 1920,
  height: 1440,
  hfovDeg: 69,
  k1: 0,
  k2: 0,
  noisePx: 0.7,
  outlierRatio: 0.25,
  matchDropRate: 0.1,
  clutterPerImage: 150,
  numPoints: 0,
  descriptors: true,
  pairWindow: 6,
  extraPairsPerImage: 4,
  minSharedPoints: 40,
  idOffset: 0,
  offset: [0, 0, 0],
};

// ---------------------------------------------------------------------------- primitives

interface Box {
  kind: 'box';
  min: [number, number, number];
  max: [number, number, number];
  color: [number, number, number];
}
interface Sphere {
  kind: 'sphere';
  c: [number, number, number];
  r: number;
  color: [number, number, number];
}
interface Plane {
  kind: 'plane';
  /** z = height, x ∈ [x0, x1], y ∈ [y0, y1]; normal +z. */
  z: number;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  color: [number, number, number];
}
type Prim = Box | Sphere | Plane;

function primArea(p: Prim): number {
  if (p.kind === 'plane') return (p.x1 - p.x0) * (p.y1 - p.y0);
  if (p.kind === 'sphere') return 4 * Math.PI * p.r * p.r;
  const dx = p.max[0] - p.min[0],
    dy = p.max[1] - p.min[1],
    dz = p.max[2] - p.min[2];
  return 2 * (dx * dy + dy * dz + dx * dz);
}

/** Uniform sample on the surface; writes position and normal. */
function samplePrim(p: Prim, rng: Rng, X: Float64Array, N: Float64Array, o: number): void {
  if (p.kind === 'plane') {
    X[o] = p.x0 + rng.next() * (p.x1 - p.x0);
    X[o + 1] = p.y0 + rng.next() * (p.y1 - p.y0);
    X[o + 2] = p.z;
    N[o] = 0;
    N[o + 1] = 0;
    N[o + 2] = 1;
    return;
  }
  if (p.kind === 'sphere') {
    let x = 0,
      y = 0,
      z = 0,
      n = 0;
    do {
      x = rng.gaussian();
      y = rng.gaussian();
      z = rng.gaussian();
      n = Math.hypot(x, y, z);
    } while (n < 1e-9);
    N[o] = x / n;
    N[o + 1] = y / n;
    N[o + 2] = z / n;
    X[o] = p.c[0] + p.r * N[o];
    X[o + 1] = p.c[1] + p.r * N[o + 1];
    X[o + 2] = p.c[2] + p.r * N[o + 2];
    return;
  }
  const d = [p.max[0] - p.min[0], p.max[1] - p.min[1], p.max[2] - p.min[2]];
  const areas = [d[1] * d[2], d[1] * d[2], d[0] * d[2], d[0] * d[2], d[0] * d[1], d[0] * d[1]];
  let u = rng.next() * (areas[0] + areas[1] + areas[2] + areas[3] + areas[4] + areas[5]);
  let face = 0;
  while (face < 5 && u > areas[face]) u -= areas[face++];
  const axis = face >> 1,
    sign = face & 1 ? 1 : -1;
  for (let k = 0; k < 3; k++) {
    X[o + k] = p.min[k] + rng.next() * d[k];
    N[o + k] = 0;
  }
  X[o + axis] = sign > 0 ? p.max[axis] : p.min[axis];
  N[o + axis] = sign;
}

/** True if the open segment from A to B (excluding the last 1 %) hits the primitive. */
function segmentHits(p: Prim, ax: number, ay: number, az: number, bx: number, by: number, bz: number): boolean {
  const dx = bx - ax,
    dy = by - ay,
    dz = bz - az;
  const tMax = 0.99;
  if (p.kind === 'plane') return false;
  if (p.kind === 'sphere') {
    const ox = ax - p.c[0],
      oy = ay - p.c[1],
      oz = az - p.c[2];
    const a = dx * dx + dy * dy + dz * dz;
    const b = 2 * (ox * dx + oy * dy + oz * dz);
    const c = ox * ox + oy * oy + oz * oz - p.r * p.r;
    const disc = b * b - 4 * a * c;
    if (disc <= 0) return false;
    const s = Math.sqrt(disc);
    const t0 = (-b - s) / (2 * a),
      t1 = (-b + s) / (2 * a);
    return (t0 > 1e-6 && t0 < tMax) || (t1 > 1e-6 && t1 < tMax && t0 < 1e-6 && c < 0);
  }
  let t0 = 1e-6,
    t1 = tMax;
  const o = [ax, ay, az],
    d = [dx, dy, dz];
  for (let k = 0; k < 3; k++) {
    if (Math.abs(d[k]) < 1e-15) {
      if (o[k] < p.min[k] || o[k] > p.max[k]) return false;
      continue;
    }
    let ta = (p.min[k] - o[k]) / d[k],
      tb = (p.max[k] - o[k]) / d[k];
    if (ta > tb) {
      const tmp = ta;
      ta = tb;
      tb = tmp;
    }
    if (ta > t0) t0 = ta;
    if (tb < t1) t1 = tb;
    if (t0 > t1) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------- cameras

/** World-to-camera rotation looking from C along `fwd` with world up +Z and a roll (rad). */
function lookRotation(fx: number, fy: number, fz: number, roll: number): Float64Array {
  let n = Math.hypot(fx, fy, fz);
  fx /= n;
  fy /= n;
  fz /= n;
  // right = f × up(0,0,1)
  let rx = fy,
    ry = -fx,
    rz = 0;
  n = Math.hypot(rx, ry, rz);
  if (n < 1e-9) {
    rx = 1;
    ry = 0;
    n = 1;
  }
  rx /= n;
  ry /= n;
  rz /= n;
  // down = f × right
  let dx = fy * rz - fz * ry,
    dy = fz * rx - fx * rz,
    dz = fx * ry - fy * rx;
  const c = Math.cos(roll),
    s = Math.sin(roll);
  const rx2 = c * rx + s * dx,
    ry2 = c * ry + s * dy,
    rz2 = c * rz + s * dz;
  dx = -s * rx + c * dx;
  dy = -s * ry + c * dy;
  dz = -s * rz + c * dz;
  return Float64Array.of(rx2, ry2, rz2, dx, dy, dz, fx, fy, fz);
}

function poseFrom(R: Float64Array, C: ArrayLike<number>): Pose {
  return {
    R,
    t: Float64Array.of(
      -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
      -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
      -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
    ),
  };
}

function center(p: Pose): [number, number, number] {
  const R = p.R,
    t = p.t;
  return [
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  ];
}

function buildOrbit(o: SyntheticSceneOptions, rng: Rng): { prims: Prim[]; poses: Pose[]; numPoints: number } {
  const prims: Prim[] = [
    { kind: 'plane', z: 0, x0: -0.45, x1: 0.45, y0: -0.45, y1: 0.45, color: [150, 110, 70] },
    { kind: 'box', min: [-0.12, -0.08, 0], max: [0.06, 0.1, 0.14], color: [200, 60, 50] },
    { kind: 'box', min: [0.03, -0.12, 0], max: [0.12, -0.02, 0.22], color: [60, 160, 210] },
    { kind: 'sphere', c: [-0.05, 0.12, 0.06], r: 0.06, color: [230, 200, 60] },
  ];
  const poses: Pose[] = [];
  const n = o.numImages;
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n + 0.02 * rng.gaussian();
    const r = 0.5 * (1 + 0.04 * Math.sin(3 * a) + 0.01 * rng.gaussian());
    const h = 0.25 + 0.05 * Math.sin(2 * a) + 0.005 * rng.gaussian();
    const C = [r * Math.cos(a), r * Math.sin(a), h];
    const target = [0.01 * rng.gaussian(), 0.01 * rng.gaussian(), 0.06];
    const R = lookRotation(target[0] - C[0], target[1] - C[1], target[2] - C[2], 0.05 * rng.gaussian());
    poses.push(poseFrom(R, C));
  }
  return { prims, poses, numPoints: o.numPoints || 5000 };
}

function yardPrims(): Prim[] {
  const prims: Prim[] = [{ kind: 'plane', z: 0, x0: -35, x1: 35, y0: -20, y1: 50, color: [70, 140, 60] }];
  // houses along the far side and the flanks
  const houses: [number, number, number, number, number][] = [
    // x0, y0, w, d, h
    [-30, 28, 12, 9, 7],
    [-12, 32, 10, 8, 6],
    [4, 29, 14, 10, 8],
    [22, 33, 11, 9, 6.5],
    [-34, 5, 8, 14, 6],
    [27, 2, 9, 12, 7],
    [-8, 44, 16, 4, 9],
  ];
  const cols: [number, number, number][] = [
    [190, 170, 150],
    [170, 90, 70],
    [220, 220, 210],
    [120, 120, 140],
    [200, 180, 120],
    [160, 140, 120],
    [140, 100, 90],
  ];
  houses.forEach((h, k) => prims.push({ kind: 'box', min: [h[0], h[1], 0], max: [h[0] + h[2], h[1] + h[3], h[4]], color: cols[k] }));
  // trees and a shed
  prims.push({ kind: 'sphere', c: [-15, 15, 4], r: 2.5, color: [40, 100, 40] });
  prims.push({ kind: 'sphere', c: [12, 18, 4.5], r: 3, color: [50, 110, 45] });
  prims.push({ kind: 'sphere', c: [-3, 22, 3.5], r: 2, color: [45, 95, 50] });
  prims.push({ kind: 'box', min: [8, 8, 0], max: [11, 10.5, 2.5], color: [150, 90, 60] });
  return prims;
}

function buildWalk(o: SyntheticSceneOptions, rng: Rng): { prims: Prim[]; poses: Pose[]; numPoints: number } {
  const prims = yardPrims();
  const poses: Pose[] = [];
  const n = o.numImages;
  for (let i = 0; i < n; i++) {
    const s = i / Math.max(1, n - 1);
    // S-shaped walk of ~30 m
    const C = [-14 + 28 * s + 0.03 * rng.gaussian(), -6 + 4 * Math.sin(Math.PI * s) + 0.03 * rng.gaussian(), 1.6 + 0.03 * Math.sin(9 * s) + 0.01 * rng.gaussian()];
    // look mostly north with a slow pan left/right and slight downward pitch
    const yaw = Math.PI / 2 + 0.45 * Math.sin(2 * Math.PI * s * 1.5) + 0.01 * rng.gaussian();
    const pitch = -0.08 + 0.03 * Math.sin(5 * s) + 0.005 * rng.gaussian();
    const fx = Math.cos(yaw) * Math.cos(pitch),
      fy = Math.sin(yaw) * Math.cos(pitch),
      fz = Math.sin(pitch);
    poses.push(poseFrom(lookRotation(fx, fy, fz, 0.03 * rng.gaussian()), C));
  }
  return { prims, poses, numPoints: o.numPoints || 9000 };
}

/**
 * Closed walk (~110 m) around the yard: ellipse centred at (0, 14) with radii 22 × 12 m
 * (≥ 2 m from every house / tree / the shed), looking towards the centre with a ±20° pan.
 */
function buildLoop(o: SyntheticSceneOptions, rng: Rng): { prims: Prim[]; poses: Pose[]; numPoints: number } {
  const prims = yardPrims();
  // structure on the south side too (seen when walking along the north edge)
  prims.push({ kind: 'box', min: [-28, -10.4, 0], max: [28, -10, 2], color: [120, 90, 70] }); // fence
  prims.push({ kind: 'box', min: [-12, -22, 0], max: [6, -13, 7], color: [210, 200, 180] }); // house
  prims.push({ kind: 'sphere', c: [17, -5, 3], r: 2.2, color: [40, 105, 45] }); // tree
  const poses: Pose[] = [];
  const n = o.numImages;
  for (let i = 0; i < n; i++) {
    const s = i / n; // closed: image n would coincide with image 0
    const th = 2 * Math.PI * s - Math.PI / 2;
    const C = [22 * Math.cos(th) + 0.03 * rng.gaussian(), 14 + 12 * Math.sin(th) + 0.03 * rng.gaussian(), 1.6 + 0.01 * rng.gaussian()];
    const yaw = Math.atan2(14 - C[1], -C[0]) + 0.35 * Math.sin(2 * Math.PI * 4 * s) + 0.01 * rng.gaussian();
    const pitch = -0.1 + 0.005 * rng.gaussian();
    const fx = Math.cos(yaw) * Math.cos(pitch),
      fy = Math.sin(yaw) * Math.cos(pitch),
      fz = Math.sin(pitch);
    poses.push(poseFrom(lookRotation(fx, fy, fz, 0.03 * rng.gaussian()), C));
  }
  return { prims, poses, numPoints: o.numPoints || 9000 };
}

/** Straight 24 m walk north towards the houses, looking forward (small pan / sway). */
function buildForward(o: SyntheticSceneOptions, rng: Rng): { prims: Prim[]; poses: Pose[]; numPoints: number } {
  const prims = yardPrims();
  const poses: Pose[] = [];
  const n = o.numImages;
  for (let i = 0; i < n; i++) {
    const s = i / Math.max(1, n - 1);
    const C = [-1 + 0.15 * Math.sin(6 * Math.PI * s) + 0.02 * rng.gaussian(), -6 + 24 * s + 0.02 * rng.gaussian(), 1.6 + 0.02 * Math.sin(12 * Math.PI * s)];
    const yaw = Math.PI / 2 + 0.05 * Math.sin(2 * Math.PI * s) + 0.005 * rng.gaussian();
    const pitch = -0.05 + 0.005 * rng.gaussian();
    const fx = Math.cos(yaw) * Math.cos(pitch),
      fy = Math.sin(yaw) * Math.cos(pitch),
      fz = Math.sin(pitch);
    poses.push(poseFrom(lookRotation(fx, fy, fz, 0.02 * rng.gaussian()), C));
  }
  return { prims, poses, numPoints: o.numPoints || 9000 };
}

function hashColor(rng: Rng, base: [number, number, number]): [number, number, number] {
  const v = 0.7 + 0.6 * rng.next();
  return [
    Math.max(0, Math.min(255, Math.round(base[0] * v))),
    Math.max(0, Math.min(255, Math.round(base[1] * v))),
    Math.max(0, Math.min(255, Math.round(base[2] * v))),
  ];
}

/** Random RootSIFT-like unit vector (non-negative), 128 floats. */
function randomRootSift(rng: Rng, out: Float64Array): void {
  let s = 0;
  for (let k = 0; k < DESCRIPTOR_SIZE; k++) {
    const g = rng.gaussian();
    const v = rng.next() < 0.3 ? 0 : g * g; // sparse, heavy-tailed histogram
    out[k] = v;
    s += v;
  }
  if (s <= 0) {
    out[0] = 1;
    s = 1;
  }
  // RootSIFT: L1-normalise then sqrt ⇒ unit L2
  for (let k = 0; k < DESCRIPTOR_SIZE; k++) out[k] = Math.sqrt(out[k] / s);
}

function quantizeDescriptor(v: Float64Array, noise: number, rng: Rng, out: Uint8Array, o: number, tmp: Float64Array): void {
  let n2 = 0;
  for (let k = 0; k < DESCRIPTOR_SIZE; k++) {
    const x = Math.max(0, v[k] + noise * rng.gaussian());
    tmp[k] = x;
    n2 += x * x;
  }
  const inv = n2 > 0 ? 1 / Math.sqrt(n2) : 0;
  for (let k = 0; k < DESCRIPTOR_SIZE; k++) out[o + k] = Math.min(255, Math.round(512 * tmp[k] * inv));
}

/**
 * Generates a synthetic scene. Image ids are idOffset + index; frames are in capture
 * order (FrameInfo.t = 100 ms · index).
 */
export function generateScene(opts: Partial<SyntheticSceneOptions> & { kind: SyntheticSceneKind }): SyntheticScene {
  const o: SyntheticSceneOptions = { ...DEFAULTS, ...opts };
  const kindIndex = { orbit: 1, walk: 2, loop: 3, forward: 4 }[o.kind];
  if (kindIndex === undefined) throw new RangeError(`generateScene: unknown kind ${String(o.kind)}`);
  const rng = new Rng(o.seed * 7919 + kindIndex);
  const built =
    o.kind === 'orbit' ? buildOrbit(o, rng) : o.kind === 'walk' ? buildWalk(o, rng) : o.kind === 'loop' ? buildLoop(o, rng) : buildForward(o, rng);
  const { prims } = built;
  // offset the world
  const off = o.offset;
  if (off[0] !== 0 || off[1] !== 0 || off[2] !== 0) {
    for (const p of prims) {
      if (p.kind === 'plane') {
        p.x0 += off[0];
        p.x1 += off[0];
        p.y0 += off[1];
        p.y1 += off[1];
        p.z += off[2];
      } else if (p.kind === 'sphere') for (let k = 0; k < 3; k++) p.c[k] += off[k];
      else
        for (let k = 0; k < 3; k++) {
          p.min[k] += off[k];
          p.max[k] += off[k];
        }
    }
    for (let i = 0; i < built.poses.length; i++) {
      const C = center(built.poses[i]);
      built.poses[i] = poseFrom(built.poses[i].R, [C[0] + off[0], C[1] + off[1], C[2] + off[2]]);
    }
  }
  const camera: CameraIntrinsics = {
    width: o.width,
    height: o.height,
    f: (0.5 * o.width) / Math.tan((0.5 * o.hfovDeg * Math.PI) / 180),
    cx: 0.5 * (o.width - 1),
    cy: 0.5 * (o.height - 1),
    k1: o.k1,
    k2: o.k2,
  };
  // ---- surface points (area-weighted over primitives)
  const NP = built.numPoints;
  const X = new Float64Array(3 * NP),
    Nrm = new Float64Array(3 * NP);
  const primOf = new Int32Array(NP);
  const colorsP = new Uint8Array(3 * NP);
  // planes are large; weight them down so objects get a fair share
  const weights = prims.map((p) => primArea(p) * (p.kind === 'plane' ? (o.kind === 'orbit' ? 0.6 : 0.12) : 1));
  const wsum = weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < NP; i++) {
    let u = rng.next() * wsum,
      k = 0;
    while (k < prims.length - 1 && u > weights[k]) u -= weights[k++];
    primOf[i] = k;
    samplePrim(prims[k], rng, X, Nrm, 3 * i);
    const c = hashColor(rng, prims[k].color);
    colorsP[3 * i] = c[0];
    colorsP[3 * i + 1] = c[1];
    colorsP[3 * i + 2] = c[2];
  }
  // ---- descriptors base per point
  const base = o.descriptors ? new Float64Array(DESCRIPTOR_SIZE * NP) : null;
  if (base) {
    const tmp = new Float64Array(DESCRIPTOR_SIZE);
    for (let i = 0; i < NP; i++) {
      randomRootSift(rng, tmp);
      base.set(tmp, DESCRIPTOR_SIZE * i);
    }
  }
  // ---- visibility and keypoints
  const nImg = built.poses.length;
  const imageIds: number[] = [];
  const frames: FrameInfo[] = [];
  const features: FeatureSet[] = [];
  const colors: Uint8Array[] = [];
  const featurePoint: Int32Array[] = [];
  const visible: Int32Array[] = []; // per image: point → feature index (−1 none)
  const cosMaxView = Math.cos((82 * Math.PI) / 180);
  const occluders = prims.filter((p) => p.kind !== 'plane');
  const dtmp = new Float64Array(DESCRIPTOR_SIZE);
  const vbuf = new Float64Array(DESCRIPTOR_SIZE);
  const bb = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  const observed = new Uint8Array(NP);
  for (let i = 0; i < nImg; i++) {
    const pose = built.poses[i];
    const R = pose.R,
      t = pose.t;
    const C = center(pose);
    for (let k = 0; k < 3; k++) {
      bb[k] = Math.min(bb[k], C[k]);
      bb[k + 3] = Math.max(bb[k + 3], C[k]);
    }
    const uv: number[] = [];
    const pid: number[] = [];
    for (let p = 0; p < NP; p++) {
      const x = X[3 * p],
        y = X[3 * p + 1],
        z = X[3 * p + 2];
      const zc = R[6] * x + R[7] * y + R[8] * z + t[2];
      if (!(zc > 0.05)) continue;
      const xc = (R[0] * x + R[1] * y + R[2] * z + t[0]) / zc;
      const yc = (R[3] * x + R[4] * y + R[5] * z + t[1]) / zc;
      const r2 = xc * xc + yc * yc;
      if (r2 > 2) continue;
      const d = 1 + r2 * (camera.k1 + camera.k2 * r2);
      const u = camera.f * d * xc + camera.cx,
        v = camera.f * d * yc + camera.cy;
      if (u < 2 || v < 2 || u > camera.width - 3 || v > camera.height - 3) continue;
      // facing
      const vx = C[0] - x,
        vy = C[1] - y,
        vz = C[2] - z;
      const dist = Math.hypot(vx, vy, vz);
      if (dist > 90) continue;
      const cosv = (Nrm[3 * p] * vx + Nrm[3 * p + 1] * vy + Nrm[3 * p + 2] * vz) / dist;
      if (cosv < cosMaxView) continue;
      let occ = false;
      for (const q of occluders) {
        if (q === prims[primOf[p]] && q.kind !== 'box') continue; // sphere self-occlusion = facing test
        if (q === prims[primOf[p]]) continue; // convex box: facing test suffices
        if (segmentHits(q, C[0], C[1], C[2], x, y, z)) {
          occ = true;
          break;
        }
      }
      if (occ) continue;
      uv.push(u + o.noisePx * rng.gaussian(), v + o.noisePx * rng.gaussian());
      pid.push(p);
    }
    const nVis = pid.length;
    const nFeat = nVis + o.clutterPerImage;
    // random order of features
    const order = new Uint32Array(nFeat);
    for (let k = 0; k < nFeat; k++) order[k] = k;
    rng.shuffle(order);
    const kp = new Float32Array(4 * nFeat);
    const scores = new Float32Array(nFeat);
    const desc = new Uint8Array(DESCRIPTOR_SIZE * nFeat);
    const col = new Uint8Array(3 * nFeat);
    const fp = new Int32Array(nFeat).fill(-1);
    const vis = new Int32Array(NP).fill(-1);
    for (let k = 0; k < nFeat; k++) {
      const f = order[k];
      let u: number, v: number;
      if (k < nVis) {
        u = uv[2 * k];
        v = uv[2 * k + 1];
        const p = pid[k];
        fp[f] = p;
        vis[p] = f;
        observed[p] = 1;
        col[3 * f] = colorsP[3 * p];
        col[3 * f + 1] = colorsP[3 * p + 1];
        col[3 * f + 2] = colorsP[3 * p + 2];
        if (base) {
          for (let q = 0; q < DESCRIPTOR_SIZE; q++) vbuf[q] = base[DESCRIPTOR_SIZE * p + q];
          quantizeDescriptor(vbuf, 0.03, rng, desc, DESCRIPTOR_SIZE * f, dtmp);
        }
      } else {
        u = rng.next() * (camera.width - 1);
        v = rng.next() * (camera.height - 1);
        col[3 * f] = rng.int(256);
        col[3 * f + 1] = rng.int(256);
        col[3 * f + 2] = rng.int(256);
        if (base) {
          randomRootSift(rng, vbuf);
          quantizeDescriptor(vbuf, 0, rng, desc, DESCRIPTOR_SIZE * f, dtmp);
        }
      }
      kp[4 * f] = u;
      kp[4 * f + 1] = v;
      kp[4 * f + 2] = 1.6 + 6 * rng.next();
      kp[4 * f + 3] = (rng.next() * 2 - 1) * Math.PI;
      scores[f] = 0.01 + rng.next() * 0.05;
    }
    const id = o.idOffset + i;
    imageIds.push(id);
    features.push({ count: nFeat, keypoints: kp, descriptors: desc, scores, width: camera.width, height: camera.height });
    colors.push(col);
    featurePoint.push(fp);
    visible.push(vis);
    frames.push({
      id,
      t: 100 * i,
      width: camera.width,
      height: camera.height,
      sharpness: 100,
      R_cam_device: Float64Array.of(1, 0, 0, 0, -1, 0, 0, 0, -1),
      gravityCam: Float64Array.of(-R[2], -R[5], -R[8]),
      heading: null,
      geo: null,
    });
  }
  for (let p = 0; p < NP; p++) {
    if (!observed[p]) continue;
    for (let k = 0; k < 3; k++) {
      bb[k] = Math.min(bb[k], X[3 * p + k]);
      bb[k + 3] = Math.max(bb[k + 3], X[3 * p + k]);
    }
  }
  const extent = Math.hypot(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]);
  // ---- pairs: sequential window + most covisible extra pairs
  const shared = (a: number, b: number): number => {
    const va = visible[a],
      vb = visible[b];
    let n = 0;
    for (let p = 0; p < NP; p++) if (va[p] >= 0 && vb[p] >= 0) n++;
    return n;
  };
  const pairSet = new Set<number>();
  const pairList: [number, number][] = [];
  const addPair = (a: number, b: number): void => {
    const key = Math.min(a, b) * 100000 + Math.max(a, b);
    if (pairSet.has(key)) return;
    pairSet.add(key);
    pairList.push([Math.min(a, b), Math.max(a, b)]);
  };
  const sharedCache = new Map<number, number>();
  const sharedOf = (a: number, b: number): number => {
    const key = Math.min(a, b) * 100000 + Math.max(a, b);
    let v = sharedCache.get(key);
    if (v === undefined) {
      v = shared(a, b);
      sharedCache.set(key, v);
    }
    return v;
  };
  for (let i = 0; i < nImg; i++) {
    for (let j = i + 1; j <= Math.min(nImg - 1, i + o.pairWindow); j++) if (sharedOf(i, j) >= o.minSharedPoints) addPair(i, j);
    if (o.extraPairsPerImage > 0) {
      const cand: [number, number][] = [];
      for (let j = 0; j < nImg; j++) {
        if (Math.abs(j - i) <= o.pairWindow) continue;
        const s = sharedOf(i, j);
        if (s >= o.minSharedPoints) cand.push([j, s]);
      }
      cand.sort((x, y) => y[1] - x[1]);
      for (let k = 0; k < Math.min(o.extraPairsPerImage, cand.length); k++) addPair(i, cand[k][0]);
    }
  }
  const pairs: SyntheticPair[] = [];
  for (const [a, b] of pairList) {
    const va = visible[a],
      vb = visible[b];
    const fa: number[] = [],
      fb: number[] = [],
      tr: number[] = [];
    const usedA = new Uint8Array(features[a].count),
      usedB = new Uint8Array(features[b].count);
    for (let p = 0; p < NP; p++) {
      if (va[p] < 0 || vb[p] < 0) continue;
      if (rng.next() < o.matchDropRate) continue;
      fa.push(va[p]);
      fb.push(vb[p]);
      tr.push(1);
      usedA[va[p]] = 1;
      usedB[vb[p]] = 1;
    }
    const nTrue = fa.length;
    const nOut = o.outlierRatio > 0 ? Math.round((nTrue * o.outlierRatio) / (1 - o.outlierRatio)) : 0;
    let tries = 0;
    for (let k = 0; k < nOut && tries < 20 * nOut + 100; tries++) {
      const i1 = rng.int(features[a].count),
        i2 = rng.int(features[b].count);
      if (usedA[i1] || usedB[i2]) continue;
      // a random pair that happens to be a true correspondence is not an outlier
      const isTrue = featurePoint[a][i1] >= 0 && featurePoint[a][i1] === featurePoint[b][i2] ? 1 : 0;
      usedA[i1] = 1;
      usedB[i2] = 1;
      fa.push(i1);
      fb.push(i2);
      tr.push(isTrue);
      k++;
    }
    const idx = fa.map((_, k) => k).sort((x, y) => fa[x] - fa[y]);
    const matches = new Uint32Array(2 * idx.length);
    const isTrue = new Uint8Array(idx.length);
    idx.forEach((k, q) => {
      matches[2 * q] = fa[k];
      matches[2 * q + 1] = fb[k];
      isTrue[q] = tr[k];
    });
    pairs.push({ a: o.idOffset + a, b: o.idOffset + b, matches, isTrue });
  }
  return {
    options: o,
    camera,
    imageIds,
    poses: built.poses,
    points: X,
    pointColors: colorsP,
    frames,
    features,
    colors,
    featurePoint,
    pairs,
    extent,
  };
}

/** Merges scenes into one (ids must not collide; no pairs are created between them). */
export function combineScenes(...scenes: SyntheticScene[]): SyntheticScene {
  const s0 = scenes[0];
  const ids = new Set<number>();
  for (const s of scenes)
    for (const id of s.imageIds) {
      if (ids.has(id)) throw new RangeError('combineScenes: image ids collide (use idOffset)');
      ids.add(id);
    }
  const total = scenes.reduce((a, s) => a + s.points.length, 0);
  const points = new Float64Array(total);
  const pointColors = new Uint8Array(total);
  let off = 0;
  const featurePoint: Int32Array[] = [];
  for (const s of scenes) {
    points.set(s.points, off);
    pointColors.set(s.pointColors, off);
    const base = off / 3;
    for (const fp of s.featurePoint) featurePoint.push(fp.map((p) => (p >= 0 ? p + base : -1)));
    off += s.points.length;
  }
  return {
    options: s0.options,
    camera: s0.camera,
    imageIds: scenes.flatMap((s) => s.imageIds),
    poses: scenes.flatMap((s) => s.poses),
    points,
    pointColors,
    frames: scenes.flatMap((s) => s.frames),
    features: scenes.flatMap((s) => s.features),
    colors: scenes.flatMap((s) => s.colors),
    featurePoint,
    pairs: scenes.flatMap((s) => s.pairs),
    extent: Math.max(...scenes.map((s) => s.extent)),
  };
}

export interface BuildDatabaseOptions {
  /**
   * Run `estimateTwoViewGeometry` on the putative matches with `camera` (default
   * true). false = store the TRUE matches as 'calibrated' inliers with GT R, t.
   */
  verify: boolean;
  /** Intrinsics used for verification (default: the true camera). */
  camera: CameraIntrinsics;
  twoView: TwoViewOptions;
  /** Minimum number of inliers to store a pair (default 15). */
  minInliers: number;
  seed: number;
}

export interface BuildDatabaseStats {
  numPairs: number;
  numStored: number;
  /** Fraction of stored inliers that are wrong correspondences. */
  inlierOutlierRate: number;
  timeMs: number;
  configs: Record<string, number>;
}

/** Builds a FeatureDatabase (frames, features, colours, verified pairs) from a scene. */
export function buildSyntheticDatabase(
  scene: SyntheticScene,
  opts: Partial<BuildDatabaseOptions> = {},
): { db: FeatureDatabase; stats: BuildDatabaseStats } {
  const verify = opts.verify ?? true;
  const cam = opts.camera ?? scene.camera;
  const minInliers = opts.minInliers ?? 15;
  const rng = new Rng(opts.seed ?? 99);
  const db = new FeatureDatabase();
  const index = new Map<number, number>();
  scene.imageIds.forEach((id, k) => {
    index.set(id, k);
    db.addFrame(scene.frames[k], scene.features[k], scene.colors[k]);
  });
  const t0 = performance.now();
  let stored = 0,
    wrong = 0,
    totalInl = 0;
  const configs: Record<string, number> = {};
  for (const pr of scene.pairs) {
    const ka = index.get(pr.a)!,
      kb = index.get(pr.b)!;
    let g: TwoViewGeometry;
    if (verify) {
      g = estimateTwoViewGeometry(scene.features[ka].keypoints, scene.features[kb].keypoints, pr.matches, cam, cam, {
        rng,
        ...opts.twoView,
      });
    } else {
      const m: number[] = [];
      for (let k = 0; k < pr.isTrue.length; k++) if (pr.isTrue[k]) m.push(pr.matches[2 * k], pr.matches[2 * k + 1]);
      const Pa = scene.poses[ka],
        Pb = scene.poses[kb];
      // relative pose x_b = R x_a + t
      const R = new Float64Array(9);
      for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++)
          R[3 * r + c] = Pb.R[3 * r] * Pa.R[3 * c] + Pb.R[3 * r + 1] * Pa.R[3 * c + 1] + Pb.R[3 * r + 2] * Pa.R[3 * c + 2];
      const t = Float64Array.of(
        Pb.t[0] - (R[0] * Pa.t[0] + R[1] * Pa.t[1] + R[2] * Pa.t[2]),
        Pb.t[1] - (R[3] * Pa.t[0] + R[4] * Pa.t[1] + R[5] * Pa.t[2]),
        Pb.t[2] - (R[6] * Pa.t[0] + R[7] * Pa.t[1] + R[8] * Pa.t[2]),
      );
      const tn = Math.hypot(t[0], t[1], t[2]) || 1;
      for (let k = 0; k < 3; k++) t[k] /= tn;
      g = { config: 'calibrated', inliers: Uint32Array.from(m), R, t, triAngle: medianTriAngle(scene, ka, kb, m) };
    }
    configs[g.config] = (configs[g.config] ?? 0) + 1;
    if (g.config === 'degenerate' || g.inliers.length / 2 < minInliers) continue;
    db.setTwoView(pr.a, pr.b, g);
    stored++;
    const fpa = scene.featurePoint[ka],
      fpb = scene.featurePoint[kb];
    for (let k = 0; k < g.inliers.length; k += 2) {
      const p = fpa[g.inliers[k]];
      if (p < 0 || p !== fpb[g.inliers[k + 1]]) wrong++;
    }
    totalInl += g.inliers.length / 2;
  }
  return {
    db,
    stats: {
      numPairs: scene.pairs.length,
      numStored: stored,
      inlierOutlierRate: totalInl > 0 ? wrong / totalInl : 0,
      timeMs: performance.now() - t0,
      configs,
    },
  };
}

function medianTriAngle(scene: SyntheticScene, ka: number, kb: number, m: number[]): number {
  const Ca = center(scene.poses[ka]),
    Cb = center(scene.poses[kb]);
  const angles: number[] = [];
  const fp = scene.featurePoint[ka];
  for (let k = 0; k < m.length; k += 2) {
    const p = fp[m[k]];
    if (p < 0) continue;
    const X = scene.points;
    const ax = X[3 * p] - Ca[0],
      ay = X[3 * p + 1] - Ca[1],
      az = X[3 * p + 2] - Ca[2];
    const bx = X[3 * p] - Cb[0],
      by = X[3 * p + 1] - Cb[1],
      bz = X[3 * p + 2] - Cb[2];
    const c = (ax * bx + ay * by + az * bz) / Math.sqrt((ax * ax + ay * ay + az * az) * (bx * bx + by * by + bz * bz));
    angles.push(Math.acos(Math.max(-1, Math.min(1, c))));
  }
  angles.sort((x, y) => x - y);
  return angles.length ? angles[angles.length >> 1] : 0;
}

export interface ReconstructionEvaluation {
  numRegistered: number;
  /** RMS camera-centre error after Sim3 alignment (scene units). */
  ate: number;
  /** ate / scene.extent. */
  ateRelative: number;
  /** Maximum centre error after alignment. */
  maxCenterError: number;
  /** Mean / max rotation error after alignment (deg). */
  meanRotErrorDeg: number;
  maxRotErrorDeg: number;
  /** Relative focal error |f − f_gt| / f_gt (first camera of the first registered image). */
  focalError: number;
  k1: number;
  k2: number;
  /** Sim3 aligning the reconstruction to ground truth (gt ≈ s R x + t). */
  sim: { s: number; R: Float64Array; t: Float64Array } | null;
  /** Fraction of track observations whose feature is a different GT point than the track's majority. */
  wrongObservationRate: number;
}

/** Aligns the reconstruction to ground truth (Sim3 on camera centres) and reports errors. */
export function evaluateReconstruction(rec: Reconstruction, scene: SyntheticScene): ReconstructionEvaluation {
  const index = new Map<number, number>();
  scene.imageIds.forEach((id, k) => index.set(id, k));
  const reg = rec.registeredImageIds().filter((id) => index.has(id));
  const n = reg.length;
  const src = new Float64Array(3 * n),
    dst = new Float64Array(3 * n);
  reg.forEach((id, k) => {
    const c = rec.imageCenter(id);
    const g = center(scene.poses[index.get(id)!]);
    src.set(c, 3 * k);
    dst.set(g, 3 * k);
  });
  const sim = n >= 3 ? umeyama(src, dst, n, true) : null;
  let ate = Infinity,
    maxE = Infinity,
    meanRot = Infinity,
    maxRot = Infinity;
  if (sim) {
    let s2 = 0;
    maxE = 0;
    for (let k = 0; k < n; k++) {
      const x = src[3 * k],
        y = src[3 * k + 1],
        z = src[3 * k + 2];
      const R = sim.R;
      const ex = sim.s * (R[0] * x + R[1] * y + R[2] * z) + sim.t[0] - dst[3 * k];
      const ey = sim.s * (R[3] * x + R[4] * y + R[5] * z) + sim.t[1] - dst[3 * k + 1];
      const ez = sim.s * (R[6] * x + R[7] * y + R[8] * z) + sim.t[2] - dst[3 * k + 2];
      const e2 = ex * ex + ey * ey + ez * ez;
      s2 += e2;
      maxE = Math.max(maxE, Math.sqrt(e2));
    }
    ate = Math.sqrt(s2 / n);
    let sr = 0;
    maxRot = 0;
    const Ra = new Float64Array(9);
    for (const id of reg) {
      const Ri = rec.images.get(id)!.pose.R;
      // aligned rotation: Ri Rsimᵀ
      for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++)
          Ra[3 * r + c] = Ri[3 * r] * sim.R[3 * c] + Ri[3 * r + 1] * sim.R[3 * c + 1] + Ri[3 * r + 2] * sim.R[3 * c + 2];
      const e = (rotationAngleBetween(Ra, scene.poses[index.get(id)!].R) * 180) / Math.PI;
      sr += e;
      maxRot = Math.max(maxRot, e);
    }
    meanRot = sr / n;
  }
  const cam = n > 0 ? rec.cameras.get(rec.images.get(reg[0])!.cameraId)! : null;
  // track purity
  let wrong = 0,
    total = 0;
  for (const p of rec.points.values()) {
    const votes = new Map<number, number>();
    for (const el of p.track) {
      const k = index.get(el.image);
      const gp = k === undefined ? -1 : scene.featurePoint[k][el.feature];
      votes.set(gp, (votes.get(gp) ?? 0) + 1);
    }
    let best = -2,
      bestN = 0;
    for (const [gp, c] of votes)
      if (c > bestN) {
        bestN = c;
        best = gp;
      }
    total += p.track.length;
    wrong += best < 0 ? p.track.length : p.track.length - bestN;
  }
  return {
    numRegistered: n,
    ate,
    ateRelative: ate / scene.extent,
    maxCenterError: maxE,
    meanRotErrorDeg: meanRot,
    maxRotErrorDeg: maxRot,
    focalError: cam ? Math.abs(cam.f - scene.camera.f) / scene.camera.f : Infinity,
    k1: cam ? cam.k1 : NaN,
    k2: cam ? cam.k2 : NaN,
    sim: sim ? { s: sim.s, R: sim.R, t: sim.t } : null,
    wrongObservationRate: total > 0 ? wrong / total : 0,
  };
}
