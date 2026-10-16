/**
 * Procedural scenes for the MockEngine (UI development / tests): an object on a wooden table
 * (close range) and a backyard with two houses and a tree (large area). Provides surface
 * samples (sparse / dense clouds with colours and normals), a coloured triangle mesh, a camera
 * trajectory (world-to-camera poses, OpenCV camera frame, Z-up world, metres) and projection
 * of visible points into a frame (for the keypoint overlay). Pure TypeScript, deterministic.
 */
import type { CameraIntrinsics, PointCloud, Pose, TriangleMesh } from '../core/types';
import { Rng } from '../math/random';
import { computeVertexNormals } from '../io/ply';

type V3 = [number, number, number];
type RGB = [number, number, number];

export type MockSceneKind = 'object' | 'scene';

interface QuadPrim {
  kind: 'quad';
  o: V3;
  u: V3;
  v: V3;
  color: (p: V3, s: number, t: number, rng: Rng) => RGB;
  /** Mesh subdivisions. */
  nu: number;
  nv: number;
}
interface TriPrim {
  kind: 'tri';
  a: V3;
  b: V3;
  c: V3;
  color: (p: V3, rng: Rng) => RGB;
}
interface RevPrim {
  kind: 'rev';
  c: V3;
  height: number;
  radius: (z: number) => number;
  color: (p: V3, z: number, theta: number, rng: Rng) => RGB;
  rings: number;
  segs: number;
  cap: boolean;
}
interface SpherePrim {
  kind: 'sphere';
  c: V3;
  r: number;
  color: (p: V3, n: V3, rng: Rng) => RGB;
  rings: number;
  segs: number;
}
type Prim = QuadPrim | TriPrim | RevPrim | SpherePrim;

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): number => Math.hypot(a[0], a[1], a[2]);
const normalize = (a: V3): V3 => {
  const n = norm(a) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
};
const clamp255 = (x: number): number => Math.max(0, Math.min(255, Math.round(x)));
const jitter = (c: RGB, rng: Rng, amt: number): RGB => [clamp255(c[0] + (rng.next() - 0.5) * amt), clamp255(c[1] + (rng.next() - 0.5) * amt), clamp255(c[2] + (rng.next() - 0.5) * amt)];
const hash = (i: number): number => {
  let x = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
};

// ---------------------------------------------------------------------------------------------
// Scene definitions
// ---------------------------------------------------------------------------------------------

function quad(o: V3, u: V3, v: V3, color: QuadPrim['color'], nu = 8, nv = 8): QuadPrim {
  return { kind: 'quad', o, u, v, color, nu, nv };
}

/** Axis-aligned box faces with outward normals (bottom face omitted). */
function box(min: V3, max: V3, color: (p: V3, face: number, s: number, t: number, rng: Rng) => RGB, div = 6): QuadPrim[] {
  const [x0, y0, z0] = min;
  const [x1, y1, z1] = max;
  const dx = x1 - x0;
  const dy = y1 - y0;
  const dz = z1 - z0;
  const f = (face: number): QuadPrim['color'] => (p, s, t, rng) => color(p, face, s, t, rng);
  const nu = (len: number): number => Math.max(2, Math.round(div * Math.max(1, len / Math.max(dx, dy, dz) * 2)));
  return [
    quad([x0, y0, z0], [dx, 0, 0], [0, 0, dz], f(0), nu(dx), nu(dz)), // −y face: u=+x, v=+z ⇒ u×v = −y
    quad([x1, y1, z0], [-dx, 0, 0], [0, 0, dz], f(1), nu(dx), nu(dz)), // +y
    quad([x0, y1, z0], [0, -dy, 0], [0, 0, dz], f(2), nu(dy), nu(dz)), // −x
    quad([x1, y0, z0], [0, dy, 0], [0, 0, dz], f(3), nu(dy), nu(dz)), // +x
    quad([x0, y0, z1], [dx, 0, 0], [0, dy, 0], f(4), nu(dx), nu(dy)), // top
  ];
}

function objectScene(): Prim[] {
  const H = 0.3;
  const vaseR = (z: number): number => {
    const t = z / H;
    return 0.055 + 0.05 * Math.sin(Math.PI * Math.min(1, t * 1.15)) + 0.012 * Math.cos(t * 9) - (t > 0.85 ? (t - 0.85) * 0.25 : 0);
  };
  const vaseColor: RevPrim['color'] = (_p, z, theta, rng) => {
    const t = z / H;
    if (t > 0.74 && t < 0.8) return jitter([38, 86, 150], rng, 18);
    const band = Math.sin(z * 95 + Math.sin(theta * 3) * 0.6);
    if (band > 0.82) return jitter([236, 222, 196], rng, 16);
    const dots = Math.sin(theta * 14) * Math.sin(z * 70) > 0.88;
    if (dots) return jitter([60, 40, 30], rng, 10);
    return jitter([196 - 30 * t, 96 - 10 * t, 62], rng, 22);
  };
  const wood: QuadPrim['color'] = (p, _s, _t, rng) => {
    const plank = Math.floor((p[1] + 1) / 0.11);
    const h = hash(plank);
    const grain = Math.sin(p[0] * 70 + h * 20 + Math.sin(p[0] * 9 + plank) * 2) * 9;
    const seam = Math.abs(((p[1] + 1) / 0.11) % 1) < 0.04 ? -45 : 0;
    return jitter([158 + h * 30 + grain + seam, 112 + h * 18 + grain * 0.7 + seam, 76 + h * 10 + grain * 0.4 + seam * 0.8], rng, 10);
  };
  const boxColor = (p: V3, face: number, s: number, t: number, rng: Rng): RGB => {
    if (face === 4) return jitter([232, 232, 236], rng, 10);
    const stripe = Math.abs(t - 0.5) < 0.12;
    const logo = Math.hypot(s - 0.5, t - 0.5) < 0.2 && face % 2 === 0;
    if (logo) return jitter([245, 196, 40], rng, 14);
    return stripe ? jitter([235, 240, 245], rng, 10) : jitter([46 + p[2] * 100, 98, 196], rng, 16);
  };
  return [
    quad([-0.45, -0.45, 0], [0.9, 0, 0], [0, 0.9, 0], wood, 36, 36),
    { kind: 'rev', c: [0, 0, 0], height: H, radius: vaseR, color: vaseColor, rings: 40, segs: 72, cap: true },
    ...box([0.16, 0.08, 0], [0.26, 0.18, 0.1], boxColor, 4),
    { kind: 'sphere', c: [-0.17, 0.12, 0.045], r: 0.045, color: (_p, n, rng) => jitter(n[2] > 0.6 ? [250, 120, 90] : [220, 60 + 40 * n[0], 70], rng, 18), rings: 14, segs: 24 },
  ];
}

function backyardScene(): Prim[] {
  const grass: QuadPrim['color'] = (p, _s, _t, rng) => {
    const patio = p[0] > -4 && p[0] < 4 && p[1] > -2.5 && p[1] < 3.5;
    if (patio) {
      const tx = Math.abs((p[0] + 4) % 0.6);
      const ty = Math.abs((p[1] + 2.5) % 0.6);
      const joint = tx < 0.03 || ty < 0.03;
      const h = hash(Math.floor((p[0] + 4) / 0.6) * 31 + Math.floor((p[1] + 2.5) / 0.6));
      return joint ? jitter([95, 92, 88], rng, 8) : jitter([168 + h * 25, 162 + h * 22, 150 + h * 18], rng, 14);
    }
    const n = Math.sin(p[0] * 1.7) * Math.cos(p[1] * 1.3) * 12 + Math.sin(p[0] * 6.1 + p[1] * 4.3) * 8;
    return jitter([74 + n, 128 + n * 1.2, 58 + n * 0.5], rng, 26);
  };
  const wall = (base: RGB) => (p: V3, face: number, s: number, t: number, rng: Rng): RGB => {
    if (face === 4) return jitter([120, 70, 60], rng, 12);
    // Windows in a regular grid; a door on the front face.
    const wx = (s * 5) % 1;
    const wz = (t * 3) % 1;
    const window = wx > 0.3 && wx < 0.7 && wz > 0.35 && wz < 0.75 && t < 0.95;
    if (face === 0 && s > 0.45 && s < 0.55 && t < 0.4) return jitter([90, 55, 40], rng, 10);
    if (window) return (wx > 0.48 && wx < 0.52) || (wz > 0.53 && wz < 0.57) ? jitter([235, 235, 235], rng, 8) : jitter([40 + p[2] * 3, 55, 72], rng, 12);
    return jitter(base, rng, 14);
  };
  const roof = (c: RGB): TriPrim['color'] => (p, rng) => {
    const shingle = Math.sin(p[2] * 28) > 0.7 ? -22 : 0;
    return jitter([c[0] + shingle, c[1] + shingle, c[2] + shingle], rng, 14);
  };
  const roofQuad = (c: RGB): QuadPrim['color'] => (p, _s, _t, rng) => roof(c)(p, rng);
  const house = (x0: number, y0: number, x1: number, y1: number, hgt: number, ridge: number, base: RGB, roofC: RGB): Prim[] => {
    const walls = box([x0, y0, 0], [x1, y1, hgt], wall(base), 8).filter((_, i) => i !== 4);
    const ym = (y0 + y1) / 2;
    const top = hgt + ridge;
    return [
      ...walls,
      // Roof slopes (normals up/out) and gables.
      quad([x0, y0, hgt], [x1 - x0, 0, 0], [0, ym - y0, ridge], roofQuad(roofC), 10, 6),
      quad([x1, y1, hgt], [x0 - x1, 0, 0], [0, ym - y1, ridge], roofQuad(roofC), 10, 6),
      { kind: 'tri', a: [x0, y1, hgt], b: [x0, y0, hgt], c: [x0, ym, top], color: (_p, rng) => jitter(base, rng, 14) },
      { kind: 'tri', a: [x1, y0, hgt], b: [x1, y1, hgt], c: [x1, ym, top], color: (_p, rng) => jitter(base, rng, 14) },
    ];
  };
  return [
    quad([-14, -14, 0], [28, 0, 0], [0, 28, 0], grass, 56, 56),
    ...house(-10, 5, -3, 11, 4.5, 2.5, [226, 214, 192], [150, 64, 52]),
    ...house(2, 6, 10.5, 12.5, 5.5, 2.2, [184, 194, 206], [74, 74, 80]),
    { kind: 'rev', c: [6.5, -0.5, 0], height: 2.6, radius: () => 0.22, color: (_p, _z, _t, rng) => jitter([96, 66, 44], rng, 16), rings: 4, segs: 16, cap: false },
    {
      kind: 'sphere',
      c: [6.5, -0.5, 4.2],
      r: 2.1,
      color: (p, n, rng) => {
        const k = Math.sin(p[0] * 5) * Math.sin(p[1] * 4.5) * Math.sin(p[2] * 5.5) * 18;
        return jitter([46 + k + n[2] * 12, 104 + k + n[2] * 22, 44 + k * 0.5], rng, 30);
      },
      rings: 18,
      segs: 30,
    },
  ];
}

// ---------------------------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------------------------

function revAreaAndCdf(p: RevPrim, n = 64): { area: number; cdf: Float64Array } {
  const cdf = new Float64Array(n + 1);
  let a = 0;
  for (let i = 0; i < n; i++) {
    const z0 = (p.height * i) / n;
    const z1 = (p.height * (i + 1)) / n;
    const r0 = p.radius(z0);
    const r1 = p.radius(z1);
    a += Math.PI * (r0 + r1) * Math.hypot(r1 - r0, z1 - z0);
    cdf[i + 1] = a;
  }
  if (p.cap) a += Math.PI * p.radius(p.height) ** 2;
  return { area: a, cdf };
}

function primArea(p: Prim): number {
  switch (p.kind) {
    case 'quad':
      return norm(cross(p.u, p.v));
    case 'tri':
      return 0.5 * norm(cross(sub(p.b, p.a), sub(p.c, p.a)));
    case 'rev':
      return revAreaAndCdf(p).area;
    case 'sphere':
      return 4 * Math.PI * p.r * p.r;
  }
}

interface Sample {
  p: V3;
  n: V3;
  c: RGB;
}

function samplePrim(p: Prim, rng: Rng): Sample {
  switch (p.kind) {
    case 'quad': {
      const s = rng.next();
      const t = rng.next();
      const pt: V3 = [p.o[0] + p.u[0] * s + p.v[0] * t, p.o[1] + p.u[1] * s + p.v[1] * t, p.o[2] + p.u[2] * s + p.v[2] * t];
      return { p: pt, n: normalize(cross(p.u, p.v)), c: p.color(pt, s, t, rng) };
    }
    case 'tri': {
      let a = rng.next();
      let b = rng.next();
      if (a + b > 1) {
        a = 1 - a;
        b = 1 - b;
      }
      const e1 = sub(p.b, p.a);
      const e2 = sub(p.c, p.a);
      const pt: V3 = [p.a[0] + e1[0] * a + e2[0] * b, p.a[1] + e1[1] * a + e2[1] * b, p.a[2] + e1[2] * a + e2[2] * b];
      return { p: pt, n: normalize(cross(e1, e2)), c: p.color(pt, rng) };
    }
    case 'rev': {
      const { area, cdf } = revAreaAndCdf(p);
      const lateral = cdf[cdf.length - 1];
      const theta = rng.next() * Math.PI * 2;
      if (p.cap && rng.next() * area > lateral) {
        const R = p.radius(p.height) * Math.sqrt(rng.next());
        const pt: V3 = [p.c[0] + R * Math.cos(theta), p.c[1] + R * Math.sin(theta), p.c[2] + p.height];
        return { p: pt, n: [0, 0, 1], c: p.color(pt, p.height, theta, rng) };
      }
      const target = rng.next() * lateral;
      let i = 0;
      while (i < cdf.length - 2 && cdf[i + 1] < target) i++;
      const f = (target - cdf[i]) / Math.max(1e-12, cdf[i + 1] - cdf[i]);
      const z = (p.height * (i + f)) / (cdf.length - 1);
      const r = p.radius(z);
      const dz = 1e-4;
      const drdz = (p.radius(Math.min(p.height, z + dz)) - p.radius(Math.max(0, z - dz))) / (2 * dz);
      const pt: V3 = [p.c[0] + r * Math.cos(theta), p.c[1] + r * Math.sin(theta), p.c[2] + z];
      return { p: pt, n: normalize([Math.cos(theta), Math.sin(theta), -drdz]), c: p.color(pt, z, theta, rng) };
    }
    case 'sphere': {
      const z = 2 * rng.next() - 1;
      const phi = rng.next() * Math.PI * 2;
      const s = Math.sqrt(1 - z * z);
      const n: V3 = [s * Math.cos(phi), s * Math.sin(phi), z];
      const pt: V3 = [p.c[0] + p.r * n[0], p.c[1] + p.r * n[1], p.c[2] + p.r * n[2]];
      return { p: pt, n, c: p.color(pt, n, rng) };
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------------------------

class MeshBuilder {
  readonly pos: number[] = [];
  readonly col: number[] = [];
  readonly idx: number[] = [];
  v(p: V3, c: RGB): number {
    this.pos.push(p[0], p[1], p[2]);
    this.col.push(c[0], c[1], c[2]);
    return this.pos.length / 3 - 1;
  }
  tri(a: number, b: number, c: number): void {
    this.idx.push(a, b, c);
  }
  /** Grid of (nu+1)×(nv+1) vertices; quads (i,j)…(i+1,j+1) CCW as seen from the u×v side. */
  grid(nu: number, nv: number, at: (s: number, t: number) => { p: V3; c: RGB }, wrapU = false): void {
    const base = this.pos.length / 3;
    const cols = wrapU ? nu : nu + 1;
    for (let j = 0; j <= nv; j++) {
      for (let i = 0; i < cols; i++) {
        const r = at(i / nu, j / nv);
        this.v(r.p, r.c);
      }
    }
    for (let j = 0; j < nv; j++) {
      for (let i = 0; i < nu; i++) {
        const i1 = wrapU ? (i + 1) % nu : i + 1;
        const a = base + j * cols + i;
        const b = base + j * cols + i1;
        const c = base + (j + 1) * cols + i1;
        const d = base + (j + 1) * cols + i;
        this.tri(a, b, c);
        this.tri(a, c, d);
      }
    }
  }
  build(): TriangleMesh {
    const positions = new Float32Array(this.pos);
    const indices = new Uint32Array(this.idx);
    const vertexCount = positions.length / 3;
    return {
      vertexCount,
      triangleCount: indices.length / 3,
      positions,
      normals: computeVertexNormals(positions, indices, vertexCount),
      colors: new Uint8Array(this.col),
      indices,
    };
  }
}

function meshPrim(m: MeshBuilder, p: Prim, rng: Rng): void {
  switch (p.kind) {
    case 'quad':
      m.grid(p.nu, p.nv, (s, t) => {
        const pt: V3 = [p.o[0] + p.u[0] * s + p.v[0] * t, p.o[1] + p.u[1] * s + p.v[1] * t, p.o[2] + p.u[2] * s + p.v[2] * t];
        return { p: pt, c: p.color(pt, s, t, rng) };
      });
      break;
    case 'tri': {
      const a = m.v(p.a, p.color(p.a, rng));
      const b = m.v(p.b, p.color(p.b, rng));
      const c = m.v(p.c, p.color(p.c, rng));
      m.tri(a, b, c);
      break;
    }
    case 'rev': {
      m.grid(
        p.segs,
        p.rings,
        (s, t) => {
          const theta = s * Math.PI * 2;
          const z = t * p.height;
          const r = p.radius(z);
          const pt: V3 = [p.c[0] + r * Math.cos(theta), p.c[1] + r * Math.sin(theta), p.c[2] + z];
          return { p: pt, c: p.color(pt, z, theta, rng) };
        },
        true,
      );
      if (p.cap) {
        const R = p.radius(p.height);
        const top: V3 = [p.c[0], p.c[1], p.c[2] + p.height];
        const ci = m.v(top, p.color(top, p.height, 0, rng));
        const first = m.pos.length / 3;
        for (let i = 0; i < p.segs; i++) {
          const th = (i / p.segs) * Math.PI * 2;
          const pt: V3 = [p.c[0] + R * Math.cos(th), p.c[1] + R * Math.sin(th), p.c[2] + p.height];
          m.v(pt, p.color(pt, p.height, th, rng));
        }
        for (let i = 0; i < p.segs; i++) m.tri(ci, first + i, first + ((i + 1) % p.segs));
      }
      break;
    }
    case 'sphere':
      m.grid(
        p.segs,
        p.rings,
        (s, t) => {
          const theta = s * Math.PI * 2;
          const phi = -Math.PI / 2 + t * Math.PI;
          const n: V3 = [Math.cos(phi) * Math.cos(theta), Math.cos(phi) * Math.sin(theta), Math.sin(phi)];
          const pt: V3 = [p.c[0] + p.r * n[0], p.c[1] + p.r * n[1], p.c[2] + p.r * n[2]];
          return { p: pt, c: p.color(pt, n, rng) };
        },
        true,
      );
      break;
  }
}

// ---------------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------------

export interface MockSamples {
  count: number;
  positions: Float32Array;
  colors: Uint8Array;
  normals: Float32Array;
  /** Trajectory parameter s ∈ [0,1] at which the point becomes visible. */
  reveal: Float32Array;
  /** Index of the primitive each sample lies on. */
  prim: Uint16Array;
}

export class MockScene {
  readonly kind: MockSceneKind;
  private readonly prims: Prim[];
  private readonly areas: number[];
  readonly seed: number;

  constructor(kind: MockSceneKind, seed = 1) {
    this.kind = kind;
    this.seed = seed;
    this.prims = kind === 'object' ? objectScene() : backyardScene();
    this.areas = this.prims.map(primArea);
  }

  /** Typical distance camera → content (m). */
  get scale(): number {
    return this.kind === 'object' ? 0.6 : 10;
  }

  /** Area-weighted surface samples (object surfaces are over-weighted: more texture). */
  sample(count: number, seed = this.seed): MockSamples {
    const rng = new Rng(seed);
    const positions = new Float32Array(count * 3);
    const colors = new Uint8Array(count * 3);
    const normals = new Float32Array(count * 3);
    const reveal = new Float32Array(count);
    const prim = new Uint16Array(count);
    // Weight: ground gets less density than objects (like real feature distributions).
    const w = this.prims.map((_p, i) => this.areas[i] * (i === 0 ? (this.kind === 'object' ? 0.35 : 0.08) : 1));
    const wsum = w.reduce((a, b) => a + b, 0);
    const cdf: number[] = [];
    let acc = 0;
    for (const x of w) cdf.push((acc += x / wsum));
    for (let k = 0; k < count; k++) {
      const r = rng.next();
      let i = 0;
      while (i < cdf.length - 1 && cdf[i] < r) i++;
      const s = samplePrim(this.prims[i], rng);
      positions.set(s.p, 3 * k);
      normals.set(s.n, 3 * k);
      colors.set(s.c, 3 * k);
      reveal[k] = this.revealKey(s.p);
      prim[k] = i;
    }
    return { count, positions, colors, normals, reveal, prim };
  }

  /** Trajectory parameter from which a point counts as observed. */
  revealKey(p: ArrayLike<number>): number {
    if (this.kind === 'object') {
      // Cameras start at azimuth −90° (−y side) and orbit counter-clockwise; a point is
      // observed once the orbit reached its azimuth − 70°.
      const w = (70 * Math.PI) / 180;
      let a = Math.atan2(p[1], p[0]) + Math.PI / 2;
      a = ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
      if (a <= w || a >= 2 * Math.PI - w) return 0; // seen from the start position
      return (a - w) / (2 * Math.PI);
    }
    return Math.max(0, Math.min(1, (p[0] - 7 + 9) / 18));
  }

  /** World-to-camera pose at trajectory parameter s (s = 1: one orbit / end of the path). */
  poseAt(s: number): Pose {
    let C: V3;
    let T: V3;
    if (this.kind === 'object') {
      const phi = -Math.PI / 2 + s * 2 * Math.PI;
      const h = 0.3 + 0.08 * Math.sin(s * 4 * Math.PI);
      C = [0.62 * Math.cos(phi), 0.62 * Math.sin(phi), h];
      T = [0, 0, 0.12];
    } else {
      const x = -9 + 18 * s;
      C = [x, -9 + 1.2 * Math.sin(s * Math.PI), 1.6];
      T = [x * 0.55, 6, 2.2];
    }
    return lookAt(C, T);
  }

  intrinsics(width: number, height: number, hfovLongDeg = 69): CameraIntrinsics {
    const f = Math.max(width, height) / 2 / Math.tan((hfovLongDeg * Math.PI) / 360);
    return { width, height, f, cx: (width - 1) / 2, cy: (height - 1) / 2, k1: -0.02, k2: 0.004 };
  }

  /**
   * Project up to `max` of the given points into a camera; returns normalised coordinates
   * (u, v) ∈ [0,1]² interleaved, for points in front of the camera and inside the frame.
   */
  project(points: Float32Array, count: number, pose: Pose, cam: CameraIntrinsics, max = 300, stride = 1): Float32Array {
    const out: number[] = [];
    const R = pose.R;
    const t = pose.t;
    for (let i = 0; i < count && out.length < 2 * max; i += stride) {
      const X = points[3 * i];
      const Y = points[3 * i + 1];
      const Z = points[3 * i + 2];
      const xc = R[0] * X + R[1] * Y + R[2] * Z + t[0];
      const yc = R[3] * X + R[4] * Y + R[5] * Z + t[1];
      const zc = R[6] * X + R[7] * Y + R[8] * Z + t[2];
      if (zc < 0.05) continue;
      const u = (cam.f * xc) / zc + cam.cx;
      const v = (cam.f * yc) / zc + cam.cy;
      const nu = (u + 0.5) / cam.width;
      const nv = (v + 0.5) / cam.height;
      if (nu < 0 || nu > 1 || nv < 0 || nv > 1) continue;
      out.push(nu, nv);
    }
    return new Float32Array(out);
  }

  /** Dense cloud with normals, radii (splat size) and confidence. */
  dense(count: number): PointCloud {
    const s = this.sample(count, this.seed + 101);
    const radii = new Float32Array(count);
    const confidence = new Float32Array(count);
    const rng = new Rng(this.seed + 7);
    // Splat radius from the local sample spacing of each primitive (gap-free rendering).
    const perPrim = new Float64Array(this.prims.length);
    for (let i = 0; i < count; i++) perPrim[s.prim[i]]++;
    const spacing = this.areas.map((a, i) => Math.sqrt(a / Math.max(1, perPrim[i])));
    for (let i = 0; i < count; i++) {
      radii[i] = spacing[s.prim[i]] * (0.75 + 0.25 * rng.next());
      confidence[i] = 0.45 + 0.55 * rng.next();
    }
    return { count, positions: s.positions, colors: s.colors, normals: s.normals, radii, confidence };
  }

  mesh(): TriangleMesh {
    const m = new MeshBuilder();
    const rng = new Rng(this.seed + 3);
    for (const p of this.prims) meshPrim(m, p, rng);
    return m.build();
  }
}

/** OpenCV world-to-camera pose looking from C at T with world up +Z. */
export function lookAt(C: V3, T: V3): Pose {
  const f = normalize(sub(T, C));
  let r = cross(f, [0, 0, 1]);
  if (norm(r) < 1e-9) r = [1, 0, 0];
  r = normalize(r);
  const d = cross(f, r); // camera +y (down)
  const R = new Float64Array([r[0], r[1], r[2], d[0], d[1], d[2], f[0], f[1], f[2]]);
  const t = new Float64Array([-(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]), -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]), -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2])]);
  return { R, t };
}

/** Camera centre C = −Rᵀt. */
export function poseCenter(p: Pose): [number, number, number] {
  const R = p.R;
  const t = p.t;
  return [-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])];
}
