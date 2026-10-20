/**
 * CPU reference of the Gaussian-splatting forward and backward passes (f64), mirroring the
 * WGSL kernels operation by operation: tests check the GPU against it, and finite
 * differences check its own backward. Small images only (it is O(pixels × splats per tile)).
 *
 * Forward (per Gaussian): p = R μ + t; near cull; Σ = Rq diag(s)² Rqᵀ (s = exp(log-scale),
 * Rq from the normalised quaternion); EWA projection Σ₂ = (J R) Σ (J R)ᵀ with the perspective
 * Jacobian J at the frustum-clamped point (FRUSTUM_GUARD); screen-space low-pass Σ₂ + εI
 * (BLUR_EPS) with the Mip-Splatting opacity compensation √(det Σ₂ / det(Σ₂ + εI)) when
 * `antialiased`; conic = (Σ₂ + εI)⁻¹; opacity o = sigmoid(logit) · comp; screen extent
 * k·√diag with k = √(2 ln(o / alphaMin)) (the ellipse outside which alpha < alphaMin), pixel
 * range → tile rectangle; colour from SH at the view direction, clamped at 0.
 *
 * Raster (per pixel, front to back over its tile's splats sorted by (depth, index)):
 * σ = ½ dᵀ conic d; skip σ < 0; α = min(ALPHA_MAX, o e^{−σ}); skip α < alphaMin; stop before
 * T(1 − α) < T_MIN; C += c α T; the background is added with the final T. `n` = index of the
 * last contributing splat in the tile list + 1 (what the backward pass re-traverses).
 *
 * Backward (front to back, with the final colour): for splat i with w = αT,
 * ∂C/∂c_i = w, ∂C/∂α_i = T c_i − (C − C_{≤i}) / (1 − α_i), then through σ to the 2-D mean and
 * conic, and through the projection to the 3-D parameters.
 */
import {
  ALPHA_MAX,
  ALPHA_MIN,
  BLUR_EPS,
  DEFAULT_NEAR,
  FRUSTUM_GUARD,
  SH_C0,
  SH_C1,
  TILE_SIZE,
  T_MIN,
  cameraCentre,
  shCoeffCount,
  type SplatCamera,
  type SplatModel,
} from './types';

export interface ProjectOptions {
  /** Mip-Splatting opacity compensation of the low-pass filter (default true). */
  antialiased?: boolean;
  /** Alpha threshold (default ALPHA_MIN); finite-difference tests use a tiny value. */
  alphaMin?: number;
}

/** Everything the forward projection computes for one Gaussian (and its backward needs). */
export interface ProjectedGaussian {
  visible: boolean;
  /** Camera-frame position. */
  p: [number, number, number];
  rz: number;
  tx: number;
  ty: number;
  clampX: boolean;
  clampY: boolean;
  /** Normalised quaternion and its rotation (row-major). */
  qn: [number, number, number, number];
  qnorm: number;
  Rq: number[];
  scale: [number, number, number];
  /** M = Rq diag(s) (row-major 3×3) and Σ = M Mᵀ. */
  M: number[];
  Sigma: number[];
  J: number[]; // 2×3
  TM: number[]; // 2×3 = J R
  a: number;
  b: number;
  c: number;
  det0: number;
  det1: number;
  comp: number;
  conic: [number, number, number];
  sig: number;
  opacity: number;
  mean2d: [number, number];
  radius: [number, number];
  /** Tile rectangle, inclusive: x0, y0, x1, y1. */
  rect: [number, number, number, number];
  dir: [number, number, number];
  dist: number;
  colourRaw: [number, number, number];
  colour: [number, number, number];
  depth: number;
}

function quatToRot(w: number, x: number, y: number, z: number): number[] {
  return [
    1 - 2 * (y * y + z * z),
    2 * (x * y - w * z),
    2 * (x * z + w * y),
    2 * (x * y + w * z),
    1 - 2 * (x * x + z * z),
    2 * (y * z - w * x),
    2 * (x * z - w * y),
    2 * (y * z + w * x),
    1 - 2 * (x * x + y * y),
  ];
}

/** Projection of Gaussian `i` (see the module comment). */
export function projectGaussian(model: SplatModel, i: number, cam: SplatCamera, opts: ProjectOptions = {}): ProjectedGaussian {
  const antialiased = opts.antialiased ?? true;
  const alphaMin = opts.alphaMin ?? ALPHA_MIN;
  const R = cam.R;
  const t = cam.t;
  const mx = model.means[3 * i];
  const my = model.means[3 * i + 1];
  const mz = model.means[3 * i + 2];
  const p: [number, number, number] = [
    R[0] * mx + R[1] * my + R[2] * mz + t[0],
    R[3] * mx + R[4] * my + R[5] * mz + t[1],
    R[6] * mx + R[7] * my + R[8] * mz + t[2],
  ];
  const out = { visible: false, p } as ProjectedGaussian;
  if (!(p[2] >= (cam.near ?? DEFAULT_NEAR))) return out;
  const q0 = model.quats[4 * i];
  const q1 = model.quats[4 * i + 1];
  const q2 = model.quats[4 * i + 2];
  const q3 = model.quats[4 * i + 3];
  const qn2 = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
  if (!(qn2 > 1e-24)) return out;
  const qnorm = Math.sqrt(qn2);
  const qn: [number, number, number, number] = [q0 / qnorm, q1 / qnorm, q2 / qnorm, q3 / qnorm];
  const Rq = quatToRot(qn[0], qn[1], qn[2], qn[3]);
  const scale: [number, number, number] = [Math.exp(model.logScales[3 * i]), Math.exp(model.logScales[3 * i + 1]), Math.exp(model.logScales[3 * i + 2])];
  const M = new Array<number>(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) M[3 * r + c] = Rq[3 * r + c] * scale[c];
  const Sigma = new Array<number>(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) Sigma[3 * r + c] = M[3 * r] * M[3 * c] + M[3 * r + 1] * M[3 * c + 1] + M[3 * r + 2] * M[3 * c + 2];
  const rz = 1 / p[2];
  const limXp = (cam.width - cam.cx) / cam.fx + FRUSTUM_GUARD * 0.5 * cam.width / cam.fx;
  const limXn = cam.cx / cam.fx + FRUSTUM_GUARD * 0.5 * cam.width / cam.fx;
  const limYp = (cam.height - cam.cy) / cam.fy + FRUSTUM_GUARD * 0.5 * cam.height / cam.fy;
  const limYn = cam.cy / cam.fy + FRUSTUM_GUARD * 0.5 * cam.height / cam.fy;
  const xr = p[0] * rz;
  const yr = p[1] * rz;
  const clampX = xr < -limXn || xr > limXp;
  const clampY = yr < -limYn || yr > limYp;
  const tx = p[2] * Math.min(limXp, Math.max(-limXn, xr));
  const ty = p[2] * Math.min(limYp, Math.max(-limYn, yr));
  const J = [cam.fx * rz, 0, -cam.fx * tx * rz * rz, 0, cam.fy * rz, -cam.fy * ty * rz * rz];
  const TM = new Array<number>(6);
  for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) TM[3 * r + c] = J[3 * r] * R[c] + J[3 * r + 1] * R[3 + c] + J[3 * r + 2] * R[6 + c];
  // Σ₂ = TM Σ TMᵀ
  const ST0 = [0, 0, 0];
  const ST1 = [0, 0, 0];
  for (let r = 0; r < 3; r++) {
    ST0[r] = Sigma[3 * r] * TM[0] + Sigma[3 * r + 1] * TM[1] + Sigma[3 * r + 2] * TM[2];
    ST1[r] = Sigma[3 * r] * TM[3] + Sigma[3 * r + 1] * TM[4] + Sigma[3 * r + 2] * TM[5];
  }
  const a = TM[0] * ST0[0] + TM[1] * ST0[1] + TM[2] * ST0[2];
  const b = TM[0] * ST1[0] + TM[1] * ST1[1] + TM[2] * ST1[2];
  const c = TM[3] * ST1[0] + TM[4] * ST1[1] + TM[5] * ST1[2];
  const det0 = a * c - b * b;
  const ap = a + BLUR_EPS;
  const cp = c + BLUR_EPS;
  const det1 = ap * cp - b * b;
  Object.assign(out, { rz, tx, ty, clampX, clampY, qn, qnorm, Rq, scale, M, Sigma, J, TM, a, b, c, det0, det1 });
  if (!(det1 > 0)) return out;
  const comp = antialiased ? (det0 > 0 ? Math.sqrt(det0 / det1) : 0) : 1;
  const conic: [number, number, number] = [cp / det1, -b / det1, ap / det1];
  const sig = 1 / (1 + Math.exp(-model.opacities[i]));
  const opacity = sig * comp;
  out.comp = comp;
  out.conic = conic;
  out.sig = sig;
  out.opacity = opacity;
  if (!(opacity >= alphaMin)) return out;
  const k = Math.sqrt(2 * Math.log(opacity / alphaMin));
  const radius: [number, number] = [k * Math.sqrt(ap), k * Math.sqrt(cp)];
  const mean2d: [number, number] = [cam.fx * p[0] * rz + cam.cx, cam.fy * p[1] * rz + cam.cy];
  const x0 = Math.max(0, Math.ceil(mean2d[0] - radius[0]));
  const x1 = Math.min(cam.width - 1, Math.floor(mean2d[0] + radius[0]));
  const y0 = Math.max(0, Math.ceil(mean2d[1] - radius[1]));
  const y1 = Math.min(cam.height - 1, Math.floor(mean2d[1] + radius[1]));
  out.mean2d = mean2d;
  out.radius = radius;
  if (!(x0 <= x1 && y0 <= y1)) return out;
  out.rect = [Math.floor(x0 / TILE_SIZE), Math.floor(y0 / TILE_SIZE), Math.floor(x1 / TILE_SIZE), Math.floor(y1 / TILE_SIZE)];
  // colour
  const C = cameraCentre(cam);
  const dvx = mx - C[0];
  const dvy = my - C[1];
  const dvz = mz - C[2];
  const dist = Math.hypot(dvx, dvy, dvz);
  const dir: [number, number, number] = dist > 0 ? [dvx / dist, dvy / dist, dvz / dist] : [0, 0, 1];
  const K = shCoeffCount(model.shDegree);
  const base = 3 * K * i;
  const sh = model.sh;
  const colourRaw: [number, number, number] = [0, 0, 0];
  for (let ch = 0; ch < 3; ch++) {
    let v = SH_C0 * sh[base + ch];
    if (K > 1) v += SH_C1 * (-dir[1] * sh[base + 3 + ch] + dir[2] * sh[base + 6 + ch] - dir[0] * sh[base + 9 + ch]);
    colourRaw[ch] = v + 0.5;
  }
  out.dir = dir;
  out.dist = dist;
  out.colourRaw = colourRaw;
  out.colour = [Math.max(0, colourRaw[0]), Math.max(0, colourRaw[1]), Math.max(0, colourRaw[2])];
  out.depth = p[2];
  out.visible = true;
  return out;
}

export interface RenderOptions extends ProjectOptions {
  /** Background colour (default black). */
  background?: [number, number, number];
}

export interface CPURender {
  width: number;
  height: number;
  /** RGB per pixel, background included. */
  color: Float64Array;
  /** Final transmittance per pixel. */
  T: Float64Array;
  /** Index of the last contributing splat in the pixel's tile list + 1. */
  n: Int32Array;
  proj: ProjectedGaussian[];
  /** Per tile: Gaussian indices sorted by (depth, index). */
  tileLists: Int32Array[];
  tilesX: number;
  tilesY: number;
}

/** Visible Gaussians binned into tiles, each list sorted by (depth, index). */
export function binTiles(proj: ProjectedGaussian[], width: number, height: number): { lists: Int32Array[]; tilesX: number; tilesY: number } {
  const tilesX = Math.ceil(width / TILE_SIZE);
  const tilesY = Math.ceil(height / TILE_SIZE);
  const order: number[] = [];
  for (let i = 0; i < proj.length; i++) if (proj[i].visible) order.push(i);
  // f32 depth keys like the GPU sort (ties by index: stable LSD radix sort over indices in order)
  const key = new Float32Array(proj.length);
  for (const i of order) key[i] = proj[i].depth;
  order.sort((x, y) => key[x] - key[y] || x - y);
  const tmp: number[][] = Array.from({ length: tilesX * tilesY }, () => []);
  for (const i of order) {
    const [x0, y0, x1, y1] = proj[i].rect;
    for (let ty = y0; ty <= y1; ty++) for (let tx = x0; tx <= x1; tx++) tmp[ty * tilesX + tx].push(i);
  }
  return { lists: tmp.map((l) => Int32Array.from(l)), tilesX, tilesY };
}

/** Forward render (see the module comment). */
export function renderCPU(model: SplatModel, cam: SplatCamera, opts: RenderOptions = {}): CPURender {
  const alphaMin = opts.alphaMin ?? ALPHA_MIN;
  const bg = opts.background ?? [0, 0, 0];
  const W = cam.width;
  const H = cam.height;
  const proj: ProjectedGaussian[] = [];
  for (let i = 0; i < model.count; i++) proj.push(projectGaussian(model, i, cam, opts));
  const { lists, tilesX, tilesY } = binTiles(proj, W, H);
  const color = new Float64Array(3 * W * H);
  const Tout = new Float64Array(W * H);
  const nOut = new Int32Array(W * H);
  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      const list = lists[Math.floor(py / TILE_SIZE) * tilesX + Math.floor(px / TILE_SIZE)];
      let T = 1;
      let r = 0;
      let g = 0;
      let b = 0;
      let last = 0;
      for (let k = 0; k < list.length; k++) {
        const s = proj[list[k]];
        const dx = px - s.mean2d[0];
        const dy = py - s.mean2d[1];
        const sigma = 0.5 * (s.conic[0] * dx * dx + s.conic[2] * dy * dy) + s.conic[1] * dx * dy;
        if (sigma < 0) continue;
        const alpha = Math.min(ALPHA_MAX, s.opacity * Math.exp(-sigma));
        if (alpha < alphaMin) continue;
        const Tn = T * (1 - alpha);
        if (Tn < T_MIN) break;
        const w = alpha * T;
        r += s.colour[0] * w;
        g += s.colour[1] * w;
        b += s.colour[2] * w;
        T = Tn;
        last = k + 1;
      }
      const o = py * W + px;
      color[3 * o] = r + T * bg[0];
      color[3 * o + 1] = g + T * bg[1];
      color[3 * o + 2] = b + T * bg[2];
      Tout[o] = T;
      nOut[o] = last;
    }
  }
  return { width: W, height: H, color, T: Tout, n: nOut, proj, tileLists: lists, tilesX, tilesY };
}

/** Gradients of the projected (2-D) quantities, 9 per Gaussian like the GPU accumulators. */
export interface Grad2D {
  /** [∂u, ∂v, ∂A, ∂B, ∂C, ∂o, ∂r, ∂g, ∂b] per Gaussian (conic = [[A, B], [B, C]]). */
  g: Float64Array;
}

/** Parameter gradients in the SplatModel layout. */
export interface SplatGrads {
  means: Float64Array;
  logScales: Float64Array;
  quats: Float64Array;
  opacities: Float64Array;
  sh: Float64Array;
}

/** Rasteriser backward: ∂L/∂(2-D quantities) given ∂L/∂C per pixel (RGB). */
export function rasterBackwardCPU(fwd: CPURender, dLdC: ArrayLike<number>, opts: ProjectOptions = {}): Grad2D {
  const alphaMin = opts.alphaMin ?? ALPHA_MIN;
  const W = fwd.width;
  const H = fwd.height;
  const g = new Float64Array(9 * fwd.proj.length);
  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      const o = py * W + px;
      const list = fwd.tileLists[Math.floor(py / TILE_SIZE) * fwd.tilesX + Math.floor(px / TILE_SIZE)];
      const gr = dLdC[3 * o];
      const gg = dLdC[3 * o + 1];
      const gb = dLdC[3 * o + 2];
      // r = gᵀ (colour still to come, background included)
      let rem = gr * fwd.color[3 * o] + gg * fwd.color[3 * o + 1] + gb * fwd.color[3 * o + 2];
      let T = 1;
      const n = fwd.n[o];
      for (let k = 0; k < n; k++) {
        const id = list[k];
        const s = fwd.proj[id];
        const dx = px - s.mean2d[0];
        const dy = py - s.mean2d[1];
        const sigma = 0.5 * (s.conic[0] * dx * dx + s.conic[2] * dy * dy) + s.conic[1] * dx * dy;
        if (sigma < 0) continue;
        const G = Math.exp(-sigma);
        const a0 = s.opacity * G;
        if (a0 < alphaMin) continue;
        const alpha = Math.min(ALPHA_MAX, a0);
        const w = alpha * T;
        const gc = gr * s.colour[0] + gg * s.colour[1] + gb * s.colour[2];
        rem -= w * gc;
        const dLda = T * gc - rem / (1 - alpha);
        const b9 = 9 * id;
        g[b9 + 6] += w * gr;
        g[b9 + 7] += w * gg;
        g[b9 + 8] += w * gb;
        if (a0 < ALPHA_MAX) {
          g[b9 + 5] += G * dLda;
          const dLds = -alpha * dLda;
          g[b9 + 2] += dLds * 0.5 * dx * dx;
          g[b9 + 3] += dLds * dx * dy;
          g[b9 + 4] += dLds * 0.5 * dy * dy;
          g[b9] += -dLds * (s.conic[0] * dx + s.conic[1] * dy);
          g[b9 + 1] += -dLds * (s.conic[1] * dx + s.conic[2] * dy);
        }
        T *= 1 - alpha;
      }
    }
  }
  return { g };
}

/**
 * Projection backward of one Gaussian: adds ∂L/∂parameters to `out` given its 2-D gradients
 * `g9` = [∂u, ∂v, ∂A, ∂B, ∂C, ∂o, ∂r, ∂g, ∂b].
 */
export function projectBackwardGaussian(model: SplatModel, i: number, cam: SplatCamera, pg: ProjectedGaussian, g9: ArrayLike<number>, out: SplatGrads, opts: ProjectOptions = {}): void {
  if (!pg.visible) return;
  const antialiased = opts.antialiased ?? true;
  const R = cam.R;
  const [du, dv, dA, dB, dC, dop, dr, dg, db] = Array.from(g9);
  // opacity: o = sigmoid(logit) · comp
  out.opacities[i] += dop * pg.comp * pg.sig * (1 - pg.sig);
  const dcomp = dop * pg.sig;
  // conic Q = (Σ₂ + εI)⁻¹:  ∂L/∂Σ' = −Q G_Q Q with G_Q = [[dA, dB/2], [dB/2, dC]]
  const [A, B, Cc] = pg.conic;
  const gq00 = dA;
  const gq01 = dB / 2;
  const gq11 = dC;
  // Q G_Q
  const t00 = A * gq00 + B * gq01;
  const t01 = A * gq01 + B * gq11;
  const t10 = B * gq00 + Cc * gq01;
  const t11 = B * gq01 + Cc * gq11;
  // (Q G_Q) Q
  const s00 = -(t00 * A + t01 * B);
  const s01 = -(t00 * B + t01 * Cc);
  const s11 = -(t10 * B + t11 * Cc);
  let da = s00;
  let dbb = 2 * s01; // per-parameter gradient of the off-diagonal b
  let dc = s11;
  if (antialiased && pg.comp > 0) {
    // comp² = det0 / det1
    const c2 = pg.comp * pg.comp;
    const f = dcomp / (2 * pg.comp * pg.det1);
    da += f * (pg.c - c2 * (pg.c + BLUR_EPS));
    dc += f * (pg.a - c2 * (pg.a + BLUR_EPS));
    dbb += f * (-2 * pg.b * (1 - c2));
  }
  // matrix gradient of Σ₂
  const G2 = [da, dbb / 2, dbb / 2, dc];
  const TM = pg.TM;
  const S = pg.Sigma;
  // ∂L/∂TM = 2 G₂ TM Σ ; ∂L/∂Σ = TMᵀ G₂ TM
  const TS = new Array<number>(6); // TM Σ (2×3)
  for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) TS[3 * r + c] = TM[3 * r] * S[c] + TM[3 * r + 1] * S[3 + c] + TM[3 * r + 2] * S[6 + c];
  const dTM = new Array<number>(6);
  for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) dTM[3 * r + c] = 2 * (G2[2 * r] * TS[c] + G2[2 * r + 1] * TS[3 + c]);
  const GT = new Array<number>(6); // G₂ TM
  for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) GT[3 * r + c] = G2[2 * r] * TM[c] + G2[2 * r + 1] * TM[3 + c];
  const dSig = new Array<number>(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) dSig[3 * r + c] = TM[r] * GT[c] + TM[3 + r] * GT[3 + c];
  // TM = J R  →  ∂L/∂J = ∂L/∂TM Rᵀ
  const dJ = new Array<number>(6);
  for (let r = 0; r < 2; r++) for (let k = 0; k < 3; k++) dJ[3 * r + k] = dTM[3 * r] * R[3 * k] + dTM[3 * r + 1] * R[3 * k + 1] + dTM[3 * r + 2] * R[3 * k + 2];
  const rz = pg.rz;
  const rz2 = rz * rz;
  const rz3 = rz2 * rz;
  const p = pg.p;
  let dpx = cam.fx * rz * du;
  let dpy = cam.fy * rz * dv;
  let dpz = -cam.fx * p[0] * rz2 * du - cam.fy * p[1] * rz2 * dv;
  dpz += dJ[0] * (-cam.fx * rz2) + dJ[2] * (2 * cam.fx * pg.tx * rz3) + dJ[4] * (-cam.fy * rz2) + dJ[5] * (2 * cam.fy * pg.ty * rz3);
  const dtx = dJ[2] * (-cam.fx * rz2);
  const dty = dJ[5] * (-cam.fy * rz2);
  if (pg.clampX) dpz += dtx * pg.tx * rz;
  else dpx += dtx;
  if (pg.clampY) dpz += dty * pg.ty * rz;
  else dpy += dty;
  // p = R μ + t
  let dmx = R[0] * dpx + R[3] * dpy + R[6] * dpz;
  let dmy = R[1] * dpx + R[4] * dpy + R[7] * dpz;
  let dmz = R[2] * dpx + R[5] * dpy + R[8] * dpz;
  // colour
  const K = shCoeffCount(model.shDegree);
  const base = 3 * K * i;
  const dcol = [pg.colourRaw[0] > 0 ? dr : 0, pg.colourRaw[1] > 0 ? dg : 0, pg.colourRaw[2] > 0 ? db : 0];
  let ddx = 0;
  let ddy = 0;
  let ddz = 0;
  for (let ch = 0; ch < 3; ch++) {
    out.sh[base + ch] += SH_C0 * dcol[ch];
    if (K > 1) {
      const sh = model.sh;
      out.sh[base + 3 + ch] += -SH_C1 * pg.dir[1] * dcol[ch];
      out.sh[base + 6 + ch] += SH_C1 * pg.dir[2] * dcol[ch];
      out.sh[base + 9 + ch] += -SH_C1 * pg.dir[0] * dcol[ch];
      ddx += -SH_C1 * sh[base + 9 + ch] * dcol[ch];
      ddy += -SH_C1 * sh[base + 3 + ch] * dcol[ch];
      ddz += SH_C1 * sh[base + 6 + ch] * dcol[ch];
    }
  }
  if (K > 1 && pg.dist > 0) {
    const dd = pg.dir[0] * ddx + pg.dir[1] * ddy + pg.dir[2] * ddz;
    dmx += (ddx - pg.dir[0] * dd) / pg.dist;
    dmy += (ddy - pg.dir[1] * dd) / pg.dist;
    dmz += (ddz - pg.dir[2] * dd) / pg.dist;
  }
  out.means[3 * i] += dmx;
  out.means[3 * i + 1] += dmy;
  out.means[3 * i + 2] += dmz;
  // Σ = M Mᵀ → ∂L/∂M = 2 ∂L/∂Σ M ; M = Rq diag(s)
  const Mm = pg.M;
  const dM = new Array<number>(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) dM[3 * r + c] = 2 * (dSig[3 * r] * Mm[c] + dSig[3 * r + 1] * Mm[3 + c] + dSig[3 * r + 2] * Mm[6 + c]);
  const Rq = pg.Rq;
  const s = pg.scale;
  const dRq = new Array<number>(9);
  for (let c = 0; c < 3; c++) {
    const ds = dM[c] * Rq[c] + dM[3 + c] * Rq[3 + c] + dM[6 + c] * Rq[6 + c];
    out.logScales[3 * i + c] += ds * s[c];
    for (let r = 0; r < 3; r++) dRq[3 * r + c] = dM[3 * r + c] * s[c];
  }
  const [w, x, y, z] = pg.qn;
  const G = dRq;
  const dw = 2 * (-z * G[1] + y * G[2] + z * G[3] - x * G[5] - y * G[6] + x * G[7]);
  const dx = 2 * (y * G[1] + z * G[2] + y * G[3] - 2 * x * G[4] - w * G[5] + z * G[6] + w * G[7] - 2 * x * G[8]);
  const dy = 2 * (-2 * y * G[0] + x * G[1] + w * G[2] + x * G[3] + z * G[5] - w * G[6] + z * G[7] - 2 * y * G[8]);
  const dz = 2 * (-2 * z * G[0] - w * G[1] + x * G[2] + w * G[3] - 2 * z * G[4] + y * G[5] + x * G[6] + y * G[7]);
  const dot = w * dw + x * dx + y * dy + z * dz;
  out.quats[4 * i] += (dw - w * dot) / pg.qnorm;
  out.quats[4 * i + 1] += (dx - x * dot) / pg.qnorm;
  out.quats[4 * i + 2] += (dy - y * dot) / pg.qnorm;
  out.quats[4 * i + 3] += (dz - z * dot) / pg.qnorm;
}

export function zeroGrads(model: SplatModel): SplatGrads {
  return {
    means: new Float64Array(3 * model.count),
    logScales: new Float64Array(3 * model.count),
    quats: new Float64Array(4 * model.count),
    opacities: new Float64Array(model.count),
    sh: new Float64Array(model.sh.length),
  };
}

/** Full backward: parameter gradients for ∂L/∂C (RGB per pixel) of `renderCPU(model, cam, opts)`. */
export function backwardCPU(model: SplatModel, cam: SplatCamera, dLdC: ArrayLike<number>, opts: RenderOptions = {}, fwd?: CPURender): { grads: SplatGrads; grad2d: Grad2D; fwd: CPURender } {
  const f = fwd ?? renderCPU(model, cam, opts);
  const grad2d = rasterBackwardCPU(f, dLdC, opts);
  const grads = zeroGrads(model);
  for (let i = 0; i < model.count; i++) projectBackwardGaussian(model, i, cam, f.proj[i], grad2d.g.subarray(9 * i, 9 * i + 9), grads, opts);
  return { grads, grad2d, fwd: f };
}

// ------------------------------------------------------------------ loss (L1 + D-SSIM)

/** SSIM window: 11 taps, σ = 1.5, normalised (separable). */
export const SSIM_RADIUS = 5;
export const SSIM_WEIGHTS: Float64Array = (() => {
  const w = new Float64Array(2 * SSIM_RADIUS + 1);
  let s = 0;
  for (let i = -SSIM_RADIUS; i <= SSIM_RADIUS; i++) s += w[i + SSIM_RADIUS] = Math.exp(-(i * i) / (2 * 1.5 * 1.5));
  for (let i = 0; i < w.length; i++) w[i] /= s;
  return w;
})();
export const SSIM_C1 = 0.01 * 0.01;
export const SSIM_C2 = 0.03 * 0.03;

/** Separable SSIM blur with zero padding (its own adjoint). */
function blur(src: Float64Array, W: number, H: number): Float64Array {
  const tmp = new Float64Array(W * H);
  const out = new Float64Array(W * H);
  const R = SSIM_RADIUS;
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let k = -R; k <= R; k++) {
        const xx = x + k;
        if (xx >= 0 && xx < W) s += SSIM_WEIGHTS[k + R] * src[y * W + xx];
      }
      tmp[y * W + x] = s;
    }
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let k = -R; k <= R; k++) {
        const yy = y + k;
        if (yy >= 0 && yy < H) s += SSIM_WEIGHTS[k + R] * tmp[yy * W + x];
      }
      out[y * W + x] = s;
    }
  return out;
}

export interface LossResult {
  loss: number;
  l1: number;
  ssim: number;
  /** ∂loss/∂x (RGB per pixel). */
  grad: Float64Array;
}

/**
 * loss = (1 − λ) · mean|x − y| + λ · (1 − mean SSIM(x, y)) over the pixels with mask ≠ 0 and
 * the 3 channels; SSIM per channel with the 11-tap Gaussian window, zero padding, and pixels
 * outside the mask set to 0 in both images.
 */
export function lossCPU(x: ArrayLike<number>, y: ArrayLike<number>, W: number, H: number, lambda = 0.2, mask?: ArrayLike<number>): LossResult {
  const n = W * H;
  let nv = 0;
  for (let i = 0; i < n; i++) if (!mask || mask[i]) nv++;
  const grad = new Float64Array(3 * n);
  let l1 = 0;
  let ssimSum = 0;
  const inv = 1 / (3 * Math.max(1, nv));
  for (let ch = 0; ch < 3; ch++) {
    const X = new Float64Array(n);
    const Y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      if (mask && !mask[i]) continue;
      X[i] = x[3 * i + ch];
      Y[i] = y[3 * i + ch];
      const d = X[i] - Y[i];
      l1 += Math.abs(d);
      grad[3 * i + ch] += (1 - lambda) * inv * (d > 0 ? 1 : d < 0 ? -1 : 0);
    }
    const mx = blur(X, W, H);
    const my = blur(Y, W, H);
    const xx = blur(X.map((v) => v * v), W, H);
    const yy = blur(Y.map((v) => v * v), W, H);
    const xy = blur(X.map((v, i) => v * Y[i]), W, H);
    const P = new Float64Array(n);
    const Q = new Float64Array(n);
    const Rr = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      if (mask && !mask[i]) continue;
      const sxx = xx[i] - mx[i] * mx[i];
      const syy = yy[i] - my[i] * my[i];
      const sxy = xy[i] - mx[i] * my[i];
      const A1 = 2 * mx[i] * my[i] + SSIM_C1;
      const A2 = 2 * sxy + SSIM_C2;
      const B1 = mx[i] * mx[i] + my[i] * my[i] + SSIM_C1;
      const B2 = sxx + syy + SSIM_C2;
      const S = (A1 * A2) / (B1 * B2);
      ssimSum += S;
      const dmu = (2 * my[i] * A2) / (B1 * B2) - (S * 2 * mx[i]) / B1;
      const dsxx = -S / B2;
      const dsxy = (2 * A1) / (B1 * B2);
      P[i] = dmu - 2 * mx[i] * dsxx - my[i] * dsxy;
      Q[i] = dsxx;
      Rr[i] = dsxy;
    }
    const bP = blur(P, W, H);
    const bQ = blur(Q, W, H);
    const bR = blur(Rr, W, H);
    for (let i = 0; i < n; i++) {
      if (mask && !mask[i]) continue;
      const dS = bP[i] + 2 * X[i] * bQ[i] + Y[i] * bR[i];
      grad[3 * i + ch] += -lambda * inv * dS;
    }
  }
  const l1m = l1 * inv;
  const ssim = ssimSum * inv;
  return { loss: (1 - lambda) * l1m + lambda * (1 - ssim), l1: l1m, ssim, grad };
}

/** Mean SSIM of two RGB images (same definition as the loss; evaluation helper). */
export function ssimCPU(x: ArrayLike<number>, y: ArrayLike<number>, W: number, H: number, mask?: ArrayLike<number>): number {
  return lossCPU(x, y, W, H, 1, mask).ssim;
}
