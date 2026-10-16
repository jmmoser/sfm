/**
 * CPU ray caster for synthetic dense-stereo test data.
 *
 * Renders scenes made of textured quads, (yawed) boxes and spheres with a rich,
 * multi-view-consistent procedural texture (3D fractal value noise evaluated at
 * the surface point, band-limited to the pixel footprint so it does not alias),
 * Lambertian shading, and 2×2 (configurable) supersampling. Returns RGBA images
 * plus ground-truth z-depth and camera-frame normals (facing the camera) for any
 * pinhole camera, so it is reusable by PatchMatch, fusion and TSDF tests.
 *
 * Conventions: world +Z up, world-to-camera poses (x_cam = R X + t), OpenCV
 * camera frame, pixel centres at integer coordinates.
 */
import type { CameraIntrinsics, Pose, RGBAImage } from '../core/types';
import type { DenseView, DepthMap } from './types';

export interface SynthMaterial {
  /** Base (largest) texture wavelength in world units. */
  scale: number;
  /** Number of noise octaves (each halves the wavelength). Default 8. */
  octaves?: number;
  /** Amplitude gain per octave. Default 0.85. */
  gain?: number;
  /** Texture contrast multiplier around mid-grey. Default 2.2. */
  contrast?: number;
  /** Base albedo (0..1). Default (0.6, 0.6, 0.6). */
  color?: readonly [number, number, number];
  /** Noise seed. Default 0. */
  seed?: number;
  /** Untextured (constant albedo) surface. */
  flat?: boolean;
}

export type SynthPrimitive =
  | {
      kind: 'quad';
      center: ArrayLike<number>;
      /** Unit in-plane axes (orthogonal). */
      axisU: ArrayLike<number>;
      axisV: ArrayLike<number>;
      halfU: number;
      halfV: number;
      material: SynthMaterial;
    }
  | {
      kind: 'box';
      center: ArrayLike<number>;
      halfSize: ArrayLike<number>;
      /** Rotation about world +Z (radians). */
      yaw?: number;
      material: SynthMaterial;
    }
  | { kind: 'sphere'; center: ArrayLike<number>; radius: number; material: SynthMaterial };

export interface SynthScene {
  primitives: SynthPrimitive[];
  /** Background (sky) colour 0..1. Default (0.78, 0.84, 0.92). */
  background?: readonly [number, number, number];
  /** Direction TOWARDS the light (world). Default normalize(0.3, -0.4, 0.85). */
  lightDir?: ArrayLike<number>;
  /** Ambient term. Default 0.35. */
  ambient?: number;
}

export interface SynthRenderOptions {
  /** Supersampling factor per axis (s×s samples per pixel). Default 2. */
  supersample?: number;
}

export interface SynthRender {
  image: RGBAImage;
  /** Ground-truth z-depth of the pixel-centre ray, 0 = background. */
  depth: Float32Array;
  /** Ground-truth unit normal (camera frame, facing camera), 3 per pixel. */
  normal: Float32Array;
  /** Index of the primitive hit by the centre ray, −1 = background. */
  primitive: Int16Array;
}

export interface RayHit {
  /** Ray parameter (distance in units of |dir|). */
  t: number;
  /** World-space unit normal (outward, not yet flipped towards the ray). */
  nx: number;
  ny: number;
  nz: number;
  prim: number;
}

// ---------------------------------------------------------------------------
// Camera helpers

/** Pinhole intrinsics with cx = (W−1)/2, cy = (H−1)/2, f = (W/2)/tan(hfov/2). */
export function pinholeIntrinsics(width: number, height: number, hfovRad: number): CameraIntrinsics {
  return {
    width,
    height,
    f: width / 2 / Math.tan(hfovRad / 2),
    cx: (width - 1) / 2,
    cy: (height - 1) / 2,
    k1: 0,
    k2: 0,
  };
}

/** World-to-camera pose of a camera at `eye` looking at `target` (OpenCV frame, y down). */
export function lookAt(eye: ArrayLike<number>, target: ArrayLike<number>, up: ArrayLike<number> = [0, 0, 1]): Pose {
  let zx = target[0] - eye[0];
  let zy = target[1] - eye[1];
  let zz = target[2] - eye[2];
  const zn = Math.hypot(zx, zy, zz);
  zx /= zn;
  zy /= zn;
  zz /= zn;
  // x = z × up (right), y = z × x (down)
  let xx = zy * up[2] - zz * up[1];
  let xy = zz * up[0] - zx * up[2];
  let xz = zx * up[1] - zy * up[0];
  let xn = Math.hypot(xx, xy, xz);
  if (xn < 1e-9) {
    // looking along up: pick any perpendicular
    xx = 1;
    xy = 0;
    xz = 0;
    xn = 1;
  }
  xx /= xn;
  xy /= xn;
  xz /= xn;
  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;
  const R = new Float64Array([xx, xy, xz, yx, yy, yz, zx, zy, zz]);
  const t = new Float64Array([
    -(R[0] * eye[0] + R[1] * eye[1] + R[2] * eye[2]),
    -(R[3] * eye[0] + R[4] * eye[1] + R[5] * eye[2]),
    -(R[6] * eye[0] + R[7] * eye[1] + R[8] * eye[2]),
  ]);
  return { R, t };
}

/** Camera centre C = −Rᵀ t. */
export function cameraCenter(pose: Pose): Float64Array {
  const R = pose.R;
  const t = pose.t;
  return new Float64Array([
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  ]);
}

// ---------------------------------------------------------------------------
// Procedural texture

function hash3(ix: number, iy: number, iz: number, seed: number): number {
  let h = Math.imul(ix, 0x8da6b343) ^ Math.imul(iy, 0xd8163841) ^ Math.imul(iz, 0xcb1ab31f) ^ Math.imul(seed, 0x165667b1);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39);
  h ^= h >>> 15;
  return (h >>> 8) / 16777216;
}

function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

/** 3D value noise in [0, 1). */
export function valueNoise3(x: number, y: number, z: number, seed: number): number {
  const fx = Math.floor(x);
  const fy = Math.floor(y);
  const fz = Math.floor(z);
  const ix = fx | 0;
  const iy = fy | 0;
  const iz = fz | 0;
  const u = fade(x - fx);
  const v = fade(y - fy);
  const w = fade(z - fz);
  const c000 = hash3(ix, iy, iz, seed);
  const c100 = hash3(ix + 1, iy, iz, seed);
  const c010 = hash3(ix, iy + 1, iz, seed);
  const c110 = hash3(ix + 1, iy + 1, iz, seed);
  const c001 = hash3(ix, iy, iz + 1, seed);
  const c101 = hash3(ix + 1, iy, iz + 1, seed);
  const c011 = hash3(ix, iy + 1, iz + 1, seed);
  const c111 = hash3(ix + 1, iy + 1, iz + 1, seed);
  const x00 = c000 + (c100 - c000) * u;
  const x10 = c010 + (c110 - c010) * u;
  const x01 = c001 + (c101 - c001) * u;
  const x11 = c011 + (c111 - c011) * u;
  const y0 = x00 + (x10 - x00) * v;
  const y1 = x01 + (x11 - x01) * v;
  return y0 + (y1 - y0) * w;
}

/**
 * Band-limited fractal luminance texture in [0, 1] at world point (x, y, z).
 * `footprint` = world size of one pixel at the point; octaves with fewer than
 * ~3 px per wavelength are faded to their mean (anti-aliasing).
 */
export function textureLuminance(m: SynthMaterial, x: number, y: number, z: number, footprint: number): number {
  if (m.flat) return 0.5;
  const octaves = m.octaves ?? 8;
  const gain = m.gain ?? 0.85;
  const seed = m.seed ?? 0;
  let amp = 1;
  let sum = 0;
  let norm = 0;
  let lambda = m.scale;
  for (let k = 0; k < octaves; k++) {
    const r = lambda / Math.max(footprint, 1e-12); // pixels per wavelength
    const w = Math.min(1, Math.max(0, (r - 1.5) / 1.5));
    let n = 0.5;
    if (w > 0) {
      const s = 1 / lambda;
      // offset octaves to decorrelate lattices
      n = 0.5 + (valueNoise3(x * s + 17.3 * k, y * s - 5.1 * k, z * s + 3.7 * k, seed + 101 * k) - 0.5) * w;
    }
    sum += amp * n;
    norm += amp;
    amp *= gain;
    lambda *= 0.5;
  }
  const L = sum / norm;
  const c = m.contrast ?? 2.2;
  return Math.min(1, Math.max(0, 0.5 + (L - 0.5) * c));
}

// ---------------------------------------------------------------------------
// Ray casting

interface PreparedPrim {
  kind: 0 | 1 | 2;
  c: Float64Array;
  // quad
  u: Float64Array;
  v: Float64Array;
  n: Float64Array;
  hu: number;
  hv: number;
  // box
  hs: Float64Array;
  cosY: number;
  sinY: number;
  // sphere
  r: number;
  material: SynthMaterial;
}

function prepare(scene: SynthScene): PreparedPrim[] {
  return scene.primitives.map((p) => {
    const base = {
      c: Float64Array.from(p.center),
      u: new Float64Array(3),
      v: new Float64Array(3),
      n: new Float64Array(3),
      hu: 0,
      hv: 0,
      hs: new Float64Array(3),
      cosY: 1,
      sinY: 0,
      r: 0,
      material: p.material,
    };
    if (p.kind === 'quad') {
      const u = Float64Array.from(p.axisU);
      const v = Float64Array.from(p.axisV);
      const n = new Float64Array([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]);
      const nn = Math.hypot(n[0], n[1], n[2]);
      n[0] /= nn;
      n[1] /= nn;
      n[2] /= nn;
      return { ...base, kind: 0 as const, u, v, n, hu: p.halfU, hv: p.halfV };
    }
    if (p.kind === 'box') {
      const yaw = p.yaw ?? 0;
      return { ...base, kind: 1 as const, hs: Float64Array.from(p.halfSize), cosY: Math.cos(yaw), sinY: Math.sin(yaw) };
    }
    return { ...base, kind: 2 as const, r: p.radius };
  });
}

function intersectPrepared(prims: PreparedPrim[], ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, hit: RayHit): boolean {
  let best = Infinity;
  const tMin = 1e-6;
  for (let i = 0; i < prims.length; i++) {
    const p = prims[i];
    if (p.kind === 0) {
      const den = p.n[0] * dx + p.n[1] * dy + p.n[2] * dz;
      if (Math.abs(den) < 1e-12) continue;
      const t = (p.n[0] * (p.c[0] - ox) + p.n[1] * (p.c[1] - oy) + p.n[2] * (p.c[2] - oz)) / den;
      if (t <= tMin || t >= best) continue;
      const qx = ox + t * dx - p.c[0];
      const qy = oy + t * dy - p.c[1];
      const qz = oz + t * dz - p.c[2];
      if (Math.abs(qx * p.u[0] + qy * p.u[1] + qz * p.u[2]) > p.hu) continue;
      if (Math.abs(qx * p.v[0] + qy * p.v[1] + qz * p.v[2]) > p.hv) continue;
      best = t;
      hit.t = t;
      hit.nx = p.n[0];
      hit.ny = p.n[1];
      hit.nz = p.n[2];
      hit.prim = i;
    } else if (p.kind === 1) {
      // to box frame: rotate by −yaw about z
      const rx = ox - p.c[0];
      const ry = oy - p.c[1];
      const lox = p.cosY * rx + p.sinY * ry;
      const loy = -p.sinY * rx + p.cosY * ry;
      const loz = oz - p.c[2];
      const ldx = p.cosY * dx + p.sinY * dy;
      const ldy = -p.sinY * dx + p.cosY * dy;
      const ldz = dz;
      let t0 = -Infinity;
      let t1 = Infinity;
      let axis = -1;
      let sign = 0;
      const lo = [lox, loy, loz];
      const ld = [ldx, ldy, ldz];
      let ok = true;
      for (let a = 0; a < 3; a++) {
        const h = p.hs[a];
        if (Math.abs(ld[a]) < 1e-15) {
          if (lo[a] < -h || lo[a] > h) {
            ok = false;
            break;
          }
          continue;
        }
        let ta = (-h - lo[a]) / ld[a];
        let tb = (h - lo[a]) / ld[a];
        let s = -1;
        if (ta > tb) {
          const tmp = ta;
          ta = tb;
          tb = tmp;
          s = 1;
        }
        if (ta > t0) {
          t0 = ta;
          axis = a;
          sign = s;
        }
        if (tb < t1) t1 = tb;
        if (t0 > t1) {
          ok = false;
          break;
        }
      }
      if (!ok || axis < 0 || t0 <= tMin || t0 >= best) continue;
      best = t0;
      // local normal = sign * e_axis, rotate back by +yaw
      const lnx = axis === 0 ? sign : 0;
      const lny = axis === 1 ? sign : 0;
      const lnz = axis === 2 ? sign : 0;
      hit.t = t0;
      hit.nx = p.cosY * lnx - p.sinY * lny;
      hit.ny = p.sinY * lnx + p.cosY * lny;
      hit.nz = lnz;
      hit.prim = i;
    } else {
      const lx = ox - p.c[0];
      const ly = oy - p.c[1];
      const lz = oz - p.c[2];
      const a = dx * dx + dy * dy + dz * dz;
      const b = lx * dx + ly * dy + lz * dz;
      const c = lx * lx + ly * ly + lz * lz - p.r * p.r;
      const disc = b * b - a * c;
      if (disc < 0) continue;
      const sq = Math.sqrt(disc);
      let t = (-b - sq) / a;
      if (t <= tMin) t = (-b + sq) / a;
      if (t <= tMin || t >= best) continue;
      best = t;
      hit.t = t;
      hit.nx = (lx + t * dx) / p.r;
      hit.ny = (ly + t * dy) / p.r;
      hit.nz = (lz + t * dz) / p.r;
      hit.prim = i;
    }
  }
  return best < Infinity;
}

/** Cast one world-space ray; returns null on miss. Normal is outward (not flipped). */
export function raycast(scene: SynthScene, origin: ArrayLike<number>, dir: ArrayLike<number>): RayHit | null {
  const hit: RayHit = { t: 0, nx: 0, ny: 0, nz: 0, prim: -1 };
  return intersectPrepared(prepare(scene), origin[0], origin[1], origin[2], dir[0], dir[1], dir[2], hit) ? hit : null;
}

/** Render RGBA + GT depth/normals of `scene` seen by the pinhole camera (cam, pose). */
export function renderSynthetic(scene: SynthScene, cam: CameraIntrinsics, pose: Pose, opts: SynthRenderOptions = {}): SynthRender {
  const W = cam.width;
  const H = cam.height;
  const ss = Math.max(1, Math.round(opts.supersample ?? 2));
  const prims = prepare(scene);
  const bg = scene.background ?? [0.78, 0.84, 0.92];
  const ld = scene.lightDir ?? [0.3, -0.4, 0.85];
  const ln = Math.hypot(ld[0], ld[1], ld[2]);
  const lx = ld[0] / ln;
  const ly = ld[1] / ln;
  const lz = ld[2] / ln;
  const ambient = scene.ambient ?? 0.35;
  const R = pose.R;
  const C = cameraCenter(pose);
  const ox = C[0];
  const oy = C[1];
  const oz = C[2];
  const invF = 1 / cam.f;
  const data = new Uint8ClampedArray(W * H * 4);
  const depth = new Float32Array(W * H);
  const normal = new Float32Array(W * H * 3);
  const primitive = new Int16Array(W * H).fill(-1);
  const hit: RayHit = { t: 0, nx: 0, ny: 0, nz: 0, prim: -1 };

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pi = y * W + x;
      let r = 0;
      let g = 0;
      let b = 0;
      for (let sy = 0; sy < ss; sy++) {
        for (let sx = 0; sx < ss; sx++) {
          const u = x + (sx + 0.5) / ss - 0.5;
          const v = y + (sy + 0.5) / ss - 0.5;
          // camera-frame ray with z = 1 => t equals z-depth
          const cxr = (u - cam.cx) * invF;
          const cyr = (v - cam.cy) * invF;
          const dx = R[0] * cxr + R[3] * cyr + R[6];
          const dy = R[1] * cxr + R[4] * cyr + R[7];
          const dz = R[2] * cxr + R[5] * cyr + R[8];
          if (!intersectPrepared(prims, ox, oy, oz, dx, dy, dz, hit)) {
            r += bg[0];
            g += bg[1];
            b += bg[2];
            continue;
          }
          const px = ox + hit.t * dx;
          const py = oy + hit.t * dy;
          const pz = oz + hit.t * dz;
          const dn = Math.hypot(dx, dy, dz);
          let nx = hit.nx;
          let ny = hit.ny;
          let nz = hit.nz;
          const cosv = (nx * dx + ny * dy + nz * dz) / dn;
          if (cosv > 0) {
            nx = -nx;
            ny = -ny;
            nz = -nz;
          }
          const dist = hit.t * dn;
          const footprint = (dist * invF) / Math.max(Math.abs(cosv), 0.2);
          const m = prims[hit.prim].material;
          const L = textureLuminance(m, px, py, pz, footprint);
          const shade = ambient + (1 - ambient) * Math.max(0, nx * lx + ny * ly + nz * lz);
          const col = m.color ?? [0.6, 0.6, 0.6];
          const k = (0.12 + 0.88 * L) * shade;
          r += col[0] * k;
          g += col[1] * k;
          b += col[2] * k;
        }
      }
      const inv = 1 / (ss * ss);
      data[pi * 4] = Math.round(Math.min(1, r * inv) * 255);
      data[pi * 4 + 1] = Math.round(Math.min(1, g * inv) * 255);
      data[pi * 4 + 2] = Math.round(Math.min(1, b * inv) * 255);
      data[pi * 4 + 3] = 255;

      // GT from the pixel-centre ray
      const cxr = (x - cam.cx) * invF;
      const cyr = (y - cam.cy) * invF;
      const dx = R[0] * cxr + R[3] * cyr + R[6];
      const dy = R[1] * cxr + R[4] * cyr + R[7];
      const dz = R[2] * cxr + R[5] * cyr + R[8];
      if (intersectPrepared(prims, ox, oy, oz, dx, dy, dz, hit)) {
        depth[pi] = hit.t;
        primitive[pi] = hit.prim;
        // world normal -> camera frame
        let ncx = R[0] * hit.nx + R[1] * hit.ny + R[2] * hit.nz;
        let ncy = R[3] * hit.nx + R[4] * hit.ny + R[5] * hit.nz;
        let ncz = R[6] * hit.nx + R[7] * hit.ny + R[8] * hit.nz;
        if (ncx * cxr + ncy * cyr + ncz > 0) {
          ncx = -ncx;
          ncy = -ncy;
          ncz = -ncz;
        }
        normal[pi * 3] = ncx;
        normal[pi * 3 + 1] = ncy;
        normal[pi * 3 + 2] = ncz;
      }
    }
  }
  return { image: { width: W, height: H, data }, depth, normal, primitive };
}

// ---------------------------------------------------------------------------
// Ready-made scenes and datasets

/** Close-range scene (~0.2–1 m objects on a 6×6 m textured ground, viewed from ~3 m). */
export function makeNearScene(): SynthScene {
  return {
    primitives: [
      { kind: 'quad', center: [0, 0, 0], axisU: [1, 0, 0], axisV: [0, 1, 0], halfU: 3, halfV: 3, material: { scale: 0.8, octaves: 8, color: [0.62, 0.55, 0.42], seed: 1 } },
      { kind: 'box', center: [0.45, 0.3, 0.35], halfSize: [0.4, 0.3, 0.35], yaw: 0.4, material: { scale: 0.35, octaves: 7, color: [0.8, 0.42, 0.33], seed: 2 } },
      { kind: 'box', center: [-0.75, 0.65, 0.25], halfSize: [0.25, 0.25, 0.25], yaw: -0.3, material: { scale: 0.3, octaves: 7, color: [0.38, 0.55, 0.8], seed: 3 } },
      { kind: 'sphere', center: [-0.5, -0.55, 0.38], radius: 0.38, material: { scale: 0.25, octaves: 7, color: [0.45, 0.75, 0.4], seed: 4 } },
    ],
  };
}

/** Large-scale scene: backyard ground 120×120 m, houses and trees at ~10–60 m. */
export function makeFarScene(): SynthScene {
  const wall = (seed: number, color: [number, number, number]): SynthMaterial => ({ scale: 6, octaves: 9, color, seed });
  return {
    primitives: [
      { kind: 'quad', center: [0, 1, 0], axisU: [1, 0, 0], axisV: [0, 1, 0], halfU: 45, halfV: 61, material: { scale: 8, octaves: 10, color: [0.5, 0.6, 0.35], seed: 11 } },
      { kind: 'box', center: [-12, 24, 4], halfSize: [6, 5, 4], yaw: 0.3, material: wall(12, [0.85, 0.75, 0.6]) },
      { kind: 'box', center: [16, 38, 5], halfSize: [8, 6, 5], yaw: -0.25, material: wall(13, [0.75, 0.6, 0.55]) },
      { kind: 'box', center: [-4, 54, 6], halfSize: [10, 5, 6], yaw: 0.05, material: wall(14, [0.7, 0.72, 0.78]) },
      { kind: 'box', center: [22, 16, 2.5], halfSize: [3, 3, 2.5], yaw: 0.6, material: wall(15, [0.6, 0.5, 0.45]) },
      { kind: 'sphere', center: [7, 21, 4], radius: 3, material: { scale: 3, octaves: 9, color: [0.3, 0.55, 0.25], seed: 16 } },
      { kind: 'sphere', center: [-22, 42, 5], radius: 4.5, material: { scale: 3, octaves: 9, color: [0.28, 0.5, 0.22], seed: 17 } },
    ],
  };
}

export interface SynthDatasetOptions {
  width?: number;
  height?: number;
  hfovRad?: number;
  numSources?: number;
  supersample?: number;
}

export interface SynthDataset {
  scene: SynthScene;
  /** views[0] is the reference; views[1..] the sources. */
  views: DenseView[];
  gtDepth: Float32Array[];
  gtNormal: Float32Array[];
  /** Depth search range covering the GT (with margin). */
  range: { min: number; max: number };
}

/**
 * Reference camera + `numSources` cameras on an ellipse around it (in the
 * reference image plane), all looking at the same target.
 */
export function makeSyntheticDataset(kind: 'near' | 'far', opts: SynthDatasetOptions = {}): SynthDataset {
  const width = opts.width ?? 160;
  const height = opts.height ?? 120;
  const hfov = opts.hfovRad ?? (kind === 'near' ? (60 * Math.PI) / 180 : (65 * Math.PI) / 180);
  const numSources = opts.numSources ?? 6;
  const cam = pinholeIntrinsics(width, height, hfov);
  let scene: SynthScene;
  let eye: number[];
  let target: number[];
  let radU: number;
  let radV: number;
  if (kind === 'near') {
    scene = makeNearScene();
    target = [0, 0, 0.25];
    const az = (-100 * Math.PI) / 180;
    const el = (35 * Math.PI) / 180;
    const dist = 3.2;
    eye = [target[0] + dist * Math.cos(el) * Math.cos(az), target[1] + dist * Math.cos(el) * Math.sin(az), target[2] + dist * Math.sin(el)];
    radU = 0.45;
    radV = 0.35;
  } else {
    scene = makeFarScene();
    eye = [0, 0, 3.5];
    target = [0, 30, 1.5];
    radU = 2.0;
    radV = 0.8;
  }
  const refPose = lookAt(eye, target);
  const poses: Pose[] = [refPose];
  // right = row 0 of R, down = row 1 of R
  for (let i = 0; i < numSources; i++) {
    const a = (2 * Math.PI * i) / numSources + 0.3;
    const du = radU * Math.cos(a);
    const dv = radV * Math.sin(a);
    const e = [0, 1, 2].map((k) => eye[k] + du * refPose.R[k] + dv * refPose.R[3 + k]);
    poses.push(lookAt(e, target));
  }
  return renderDataset(scene, cam, poses, { supersample: opts.supersample });
}

/**
 * Render `poses` (views[i].id = i) with GT depth / normals and a depth range
 * covering all GT depths with margin (×0.8 … ×1.25).
 */
export function renderDataset(scene: SynthScene, cam: CameraIntrinsics, poses: Pose[], opts: SynthRenderOptions = {}): SynthDataset {
  const views: DenseView[] = [];
  const gtDepth: Float32Array[] = [];
  const gtNormal: Float32Array[] = [];
  let dmin = Infinity;
  let dmax = 0;
  for (let i = 0; i < poses.length; i++) {
    const r = renderSynthetic(scene, cam, poses[i], { supersample: opts.supersample ?? 2 });
    views.push({ id: i, cam: { ...cam }, pose: poses[i], image: r.image });
    gtDepth.push(r.depth);
    gtNormal.push(r.normal);
    for (let k = 0; k < r.depth.length; k++) {
      const z = r.depth[k];
      if (z > 0) {
        if (z < dmin) dmin = z;
        if (z > dmax) dmax = z;
      }
    }
  }
  if (!(dmax > 0)) {
    dmin = 1;
    dmax = 1;
  }
  return { scene, views, gtDepth, gtNormal, range: { min: dmin * 0.8, max: dmax * 1.25 } };
}

/**
 * `count` cameras evenly spaced in azimuth on a circle around `target`
 * (horizontal distance `radius`, elevation angle `elevationRad`), all looking at
 * the target — e.g. for fusion / TSDF tests that need all-around coverage.
 */
export function orbitPoses(target: ArrayLike<number>, radius: number, elevationRad: number, count: number, startAzimuthRad = 0): Pose[] {
  const poses: Pose[] = [];
  for (let i = 0; i < count; i++) {
    const az = startAzimuthRad + (2 * Math.PI * i) / count;
    const eye = [
      target[0] + radius * Math.cos(elevationRad) * Math.cos(az),
      target[1] + radius * Math.cos(elevationRad) * Math.sin(az),
      target[2] + radius * Math.sin(elevationRad),
    ];
    poses.push(lookAt(eye, target));
  }
  return poses;
}

/** Wrap GT arrays as a DepthMap (cost 0 where valid, 2 elsewhere). */
export function groundTruthDepthMap(width: number, height: number, depth: Float32Array, normal: Float32Array): DepthMap {
  const cost = new Float32Array(width * height);
  for (let i = 0; i < cost.length; i++) cost[i] = depth[i] > 0 ? 0 : 2;
  return { width, height, depth: depth.slice(), normal: normal.slice(), cost };
}

// ---------------------------------------------------------------------------
// Evaluation

export interface DepthMapStats {
  /** Pixels with GT (and inside the optional mask). */
  gtPixels: number;
  /** Fraction of GT pixels with a valid estimate. */
  validFraction: number;
  /** Among valid estimates, fraction with |z − z_gt| / z_gt ≤ relTol. */
  accurateFraction: number;
  /** validFraction × accurateFraction (completeness at tolerance). */
  completeness: number;
  medianRelError: number;
  meanRelError: number;
  /** Median angle between estimated and GT normals (degrees), over valid pixels. */
  medianNormalErrorDeg: number;
}

function medianOf(a: number[]): number {
  if (a.length === 0) return NaN;
  const s = Float64Array.from(a).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

/** Compare an estimated depth map with ground truth. */
export function evaluateDepthMap(
  dm: DepthMap,
  gtDepth: Float32Array,
  gtNormal: Float32Array | null,
  opts: { relTol?: number; mask?: Uint8Array } = {},
): DepthMapStats {
  const relTol = opts.relTol ?? 0.01;
  let gt = 0;
  let valid = 0;
  let acc = 0;
  let sumRel = 0;
  const rels: number[] = [];
  const angs: number[] = [];
  for (let i = 0; i < gtDepth.length; i++) {
    const zg = gtDepth[i];
    if (!(zg > 0)) continue;
    if (opts.mask && !opts.mask[i]) continue;
    gt++;
    const z = dm.depth[i];
    if (!(z > 0)) continue;
    valid++;
    const rel = Math.abs(z - zg) / zg;
    rels.push(rel);
    sumRel += rel;
    if (rel <= relTol) acc++;
    if (gtNormal) {
      const d = dm.normal[3 * i] * gtNormal[3 * i] + dm.normal[3 * i + 1] * gtNormal[3 * i + 1] + dm.normal[3 * i + 2] * gtNormal[3 * i + 2];
      angs.push((Math.acos(Math.max(-1, Math.min(1, d))) * 180) / Math.PI);
    }
  }
  const validFraction = gt ? valid / gt : 0;
  const accurateFraction = valid ? acc / valid : 0;
  return {
    gtPixels: gt,
    validFraction,
    accurateFraction,
    completeness: validFraction * accurateFraction,
    medianRelError: medianOf(rels),
    meanRelError: valid ? sumRel / valid : NaN,
    medianNormalErrorDeg: medianOf(angs),
  };
}

/**
 * Evaluation mask of "well-posed textured surface" pixels: GT present, at least
 * `edgeRadius` px away from any depth discontinuity (relative jump > `relJump`
 * between 4-neighbours, or surface/background), and viewed at less than
 * `maxAngleDeg` from the surface normal. Window-based stereo cannot be expected
 * to be accurate within a window radius of occlusion boundaries.
 */
export function surfaceMask(
  cam: CameraIntrinsics,
  gtDepth: Float32Array,
  gtNormal: Float32Array,
  opts: { edgeRadius?: number; relJump?: number; maxAngleDeg?: number } = {},
): Uint8Array {
  const W = cam.width;
  const H = cam.height;
  const rad = opts.edgeRadius ?? 5;
  const relJump = opts.relJump ?? 0.05;
  const cosMax = Math.cos(((opts.maxAngleDeg ?? 75) * Math.PI) / 180);
  const edge = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const z = gtDepth[i];
      if (x + 1 < W) {
        const z2 = gtDepth[i + 1];
        if (z > 0 !== z2 > 0 || Math.abs(z - z2) > relJump * Math.min(z, z2)) {
          edge[i] = 1;
          edge[i + 1] = 1;
        }
      }
      if (y + 1 < H) {
        const z2 = gtDepth[i + W];
        if (z > 0 !== z2 > 0 || Math.abs(z - z2) > relJump * Math.min(z, z2)) {
          edge[i] = 1;
          edge[i + W] = 1;
        }
      }
    }
  }
  // distance-to-edge via separable box dilation
  const tmp = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    let last = -1e9;
    for (let x = 0; x < W; x++) {
      if (edge[y * W + x]) last = x;
      if (x - last <= rad) tmp[y * W + x] = 1;
    }
    last = 1e9;
    for (let x = W - 1; x >= 0; x--) {
      if (edge[y * W + x]) last = x;
      if (last - x <= rad) tmp[y * W + x] = 1;
    }
  }
  const near = new Uint8Array(W * H);
  for (let x = 0; x < W; x++) {
    let last = -1e9;
    for (let y = 0; y < H; y++) {
      if (tmp[y * W + x]) last = y;
      if (y - last <= rad) near[y * W + x] = 1;
    }
    last = 1e9;
    for (let y = H - 1; y >= 0; y--) {
      if (tmp[y * W + x]) last = y;
      if (last - y <= rad) near[y * W + x] = 1;
    }
  }
  const mask = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!(gtDepth[i] > 0) || near[i]) continue;
      const rx = (x - cam.cx) / cam.f;
      const ry = (y - cam.cy) / cam.f;
      const c = -(gtNormal[3 * i] * rx + gtNormal[3 * i + 1] * ry + gtNormal[3 * i + 2]) / Math.hypot(rx, ry, 1);
      if (c >= cosMax) mask[i] = 1;
    }
  }
  return mask;
}

/** Human-readable one-line summary of stats (for test logs). */
export function formatDepthStats(s: DepthMapStats, relTol = 0.01): string {
  return (
    `valid ${(100 * s.validFraction).toFixed(1)}% | within ${(100 * relTol).toFixed(1)}%: ${(100 * s.accurateFraction).toFixed(1)}% ` +
    `| complete ${(100 * s.completeness).toFixed(1)}% | medRel ${(100 * s.medianRelError).toFixed(3)}% ` +
    `| meanRel ${(100 * s.meanRelError).toFixed(2)}% | medNormal ${s.medianNormalErrorDeg.toFixed(2)}°`
  );
}
