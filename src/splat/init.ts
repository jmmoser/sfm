/**
 * Initial Gaussians from a coloured point cloud (the fused MVS cloud of src/dense, a sparse
 * SfM cloud, or ground-truth samples in the harness).
 *
 * Each point becomes a Gaussian at the point with SH band 0 from its colour. Its tangent
 * σ is the RMS distance to its 3 nearest neighbours (after subsampling to the budget), at least
 * the point's footprint radius when the cloud has one; with normals it is a flat disc
 * (normal σ = `normalRatio` · tangent σ, rotation taking +z to the normal), otherwise isotropic.
 * Optionally a far "background shell" of large Gaussians on a sphere around the scene, coloured
 * from the training views, gives sky and distant structure (which MVS does not reconstruct)
 * something to fit.
 */
import type { PointCloud } from '../core/types';
import { KdTree3 } from '../dense/kdtree';
import { Rng } from '../math/random';
import { SH_C0, allocSplatModel, logit, shCoeffCount, type ShDegree, type SplatCamera, type SplatModel } from './types';

export interface SplatInitOptions {
  /** Largest number of Gaussians taken from the cloud (random subset above it). */
  maxCount: number;
  shDegree: ShDegree;
  /** Initial opacity (default 0.5). */
  opacity?: number;
  /** Normal σ / tangent σ for clouds with normals (default 0.1). */
  normalRatio?: number;
  /** Scale σ by this factor (default 1). */
  scaleFactor?: number;
  seed?: number;
}

/** Unit quaternion [w, x, y, z] rotating +z onto n. */
export function quatFromZ(nx: number, ny: number, nz: number): [number, number, number, number] {
  const l = Math.hypot(nx, ny, nz);
  if (!(l > 0)) return [1, 0, 0, 0];
  nx /= l;
  ny /= l;
  nz /= l;
  if (nz < -0.999999) return [0, 1, 0, 0];
  const w = 1 + nz;
  const q: [number, number, number, number] = [w, -ny, nx, 0];
  const ql = Math.hypot(q[0], q[1], q[2], q[3]);
  return [q[0] / ql, q[1] / ql, q[2] / ql, q[3] / ql];
}

export function initFromPointCloud(pc: PointCloud, o: SplatInitOptions): SplatModel {
  const rng = new Rng(o.seed ?? 1);
  const n = Math.min(pc.count, o.maxCount);
  const pick = n < pc.count ? Array.from(rng.sample(pc.count, n)).sort((a, b) => a - b) : null;
  const P = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const s = pick ? pick[i] : i;
    P[3 * i] = pc.positions[3 * s];
    P[3 * i + 1] = pc.positions[3 * s + 1];
    P[3 * i + 2] = pc.positions[3 * s + 2];
  }
  const m = allocSplatModel(n, o.shDegree);
  m.means.set(P);
  // RMS distance to the 3 nearest neighbours
  const tree = new KdTree3(P, n);
  const dist = new Float64Array(4);
  const ids = new Int32Array(4);
  const sigma = new Float64Array(n);
  for (let pos = 0; pos < n; pos++) {
    const k = tree.knnAt(pos, 3, dist, ids);
    let s = 0;
    for (let j = 0; j < k; j++) s += dist[j];
    sigma[tree.perm[pos]] = k ? Math.sqrt(s / k) : 0;
  }
  let med = 0;
  {
    const sorted = Float64Array.from(sigma).sort();
    med = sorted[sorted.length >> 1] || 1e-3;
  }
  const opac = logit(o.opacity ?? 0.5);
  const ratio = o.normalRatio ?? 0.1;
  const sf = o.scaleFactor ?? 1;
  const K = shCoeffCount(o.shDegree);
  for (let i = 0; i < n; i++) {
    const s = pick ? pick[i] : i;
    let st = sigma[i] > 0 ? sigma[i] : med;
    if (pc.radii && pc.radii[s] > st) st = pc.radii[s];
    st = Math.max(st * sf, 1e-7);
    const nx = pc.normals ? pc.normals[3 * s] : 0;
    const ny = pc.normals ? pc.normals[3 * s + 1] : 0;
    const nz = pc.normals ? pc.normals[3 * s + 2] : 0;
    const hasN = !!pc.normals && Math.hypot(nx, ny, nz) > 0.5;
    const ls = Math.log(st);
    m.logScales[3 * i] = ls;
    m.logScales[3 * i + 1] = ls;
    m.logScales[3 * i + 2] = hasN ? Math.log(st * ratio) : ls;
    if (hasN) m.quats.set(quatFromZ(nx, ny, nz), 4 * i);
    m.opacities[i] = opac;
    for (let c = 0; c < 3; c++) m.sh[3 * K * i + c] = (pc.colors[3 * s + c] / 255 - 0.5) / SH_C0;
  }
  return m;
}

/** A training image with its pinhole camera, for colouring background samples. */
export interface ColourSource {
  cam: SplatCamera;
  /** RGBA8. */
  data: Uint8Array | Uint8ClampedArray;
}

export interface BackgroundShellOptions {
  count: number;
  /** Shell centre (default: mean of the source camera centres). */
  centre?: [number, number, number];
  radius: number;
  shDegree: ShDegree;
  opacity?: number;
}

/**
 * Gaussians on a sphere (Fibonacci lattice), isotropic with σ ≈ the lattice spacing, coloured
 * from the first source image that sees them (samples that no source sees are dropped).
 */
export function backgroundShell(sources: ColourSource[], o: BackgroundShellOptions): SplatModel {
  const c = o.centre ?? (() => {
    const s = [0, 0, 0];
    for (const v of sources) {
      const R = v.cam.R;
      const t = v.cam.t;
      s[0] -= R[0] * t[0] + R[3] * t[1] + R[6] * t[2];
      s[1] -= R[1] * t[0] + R[4] * t[1] + R[7] * t[2];
      s[2] -= R[2] * t[0] + R[5] * t[1] + R[8] * t[2];
    }
    return s.map((x) => x / Math.max(1, sources.length)) as [number, number, number];
  })();
  const sigma = o.radius * Math.sqrt((4 * Math.PI) / o.count) * 0.6;
  const pts: number[] = [];
  const cols: number[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < o.count; i++) {
    const z = 1 - (2 * (i + 0.5)) / o.count;
    const r = Math.sqrt(1 - z * z);
    const X = c[0] + o.radius * r * Math.cos(golden * i);
    const Y = c[1] + o.radius * r * Math.sin(golden * i);
    const Z = c[2] + o.radius * z;
    for (const s of sources) {
      const R = s.cam.R;
      const t = s.cam.t;
      const pz = R[6] * X + R[7] * Y + R[8] * Z + t[2];
      if (pz <= 0) continue;
      const u = Math.round((s.cam.fx * (R[0] * X + R[1] * Y + R[2] * Z + t[0])) / pz + s.cam.cx);
      const v = Math.round((s.cam.fy * (R[3] * X + R[4] * Y + R[5] * Z + t[1])) / pz + s.cam.cy);
      if (u < 0 || v < 0 || u >= s.cam.width || v >= s.cam.height) continue;
      const o4 = 4 * (v * s.cam.width + u);
      if (s.data[o4 + 3] < 128) continue;
      pts.push(X, Y, Z);
      cols.push(s.data[o4], s.data[o4 + 1], s.data[o4 + 2]);
      break;
    }
  }
  const n = pts.length / 3;
  const m = allocSplatModel(n, o.shDegree);
  const K = shCoeffCount(o.shDegree);
  m.means.set(pts);
  m.logScales.fill(Math.log(sigma));
  m.opacities.fill(logit(o.opacity ?? 0.5));
  for (let i = 0; i < n; i++) for (let ch = 0; ch < 3; ch++) m.sh[3 * K * i + ch] = (cols[3 * i + ch] / 255 - 0.5) / SH_C0;
  return m;
}

/** Concatenation of models with the same SH degree. */
export function concatModels(a: SplatModel, b: SplatModel): SplatModel {
  if (a.shDegree !== b.shDegree) throw new Error('concatModels: SH degrees differ');
  const m = allocSplatModel(a.count + b.count, a.shDegree);
  m.means.set(a.means);
  m.means.set(b.means, a.means.length);
  m.logScales.set(a.logScales);
  m.logScales.set(b.logScales, a.logScales.length);
  m.quats.set(a.quats);
  m.quats.set(b.quats, a.quats.length);
  m.opacities.set(a.opacities);
  m.opacities.set(b.opacities, a.opacities.length);
  m.sh.set(a.sh);
  m.sh.set(b.sh, a.sh.length);
  return m;
}
