/**
 * The novel-view renderers draw the app's Viewer at an EXACT pinhole camera: fronto-parallel
 * squares and point splats at known camera-frame positions must land where u = fx·X/Z + cx,
 * v = fy·Y/Z + cy puts them (off-centre principal point, arbitrary pose, both image orientations).
 */
import { afterEach, describe, expect, it } from 'vitest';
import type { PointCloud, TriangleMesh } from '../../core/types';
import { ViewerRenderers, type NvsCamera } from './renderers';

const made: ViewerRenderers[] = [];
afterEach(() => {
  for (const r of made.splice(0)) r.dispose();
});

function rotation(ax: number[], angle: number): number[] {
  const l = Math.hypot(ax[0], ax[1], ax[2]);
  const [x, y, z] = ax.map((v) => v / l);
  const c = Math.cos(angle),
    s = Math.sin(angle),
    C = 1 - c;
  return [c + x * x * C, x * y * C - z * s, x * z * C + y * s, y * x * C + z * s, c + y * y * C, y * z * C - x * s, z * x * C - y * s, z * y * C + x * s, c + z * z * C];
}

/** World point of a camera-frame point: X = Rᵀ (x − t). */
function toWorld(cam: NvsCamera, x: number[]): number[] {
  const R = cam.R,
    d = [x[0] - cam.t[0], x[1] - cam.t[1], x[2] - cam.t[2]];
  return [R[0] * d[0] + R[3] * d[1] + R[6] * d[2], R[1] * d[0] + R[4] * d[1] + R[7] * d[2], R[2] * d[0] + R[5] * d[1] + R[8] * d[2]];
}

/** Camera-frame squares (fronto-parallel, half size h at depth z) as a white mesh in the world. */
function squares(cam: NvsCamera, centres: number[][], half: number): TriangleMesh {
  const positions = new Float32Array(12 * centres.length);
  const indices = new Uint32Array(6 * centres.length);
  centres.forEach((c, k) => {
    const corners = [
      [c[0] - half, c[1] - half, c[2]],
      [c[0] + half, c[1] - half, c[2]],
      [c[0] + half, c[1] + half, c[2]],
      [c[0] - half, c[1] + half, c[2]],
    ];
    corners.forEach((p, j) => positions.set(toWorld(cam, p), 12 * k + 3 * j));
    indices.set([4 * k, 4 * k + 1, 4 * k + 2, 4 * k, 4 * k + 2, 4 * k + 3], 6 * k);
  });
  const n = 4 * centres.length;
  return { vertexCount: n, triangleCount: 2 * centres.length, positions, normals: new Float32Array(3 * n), colors: new Uint8Array(3 * n).fill(255), indices };
}

/** Alpha-weighted centroid and area (px) of the image inside a box. */
function blob(img: Uint8Array, W: number, u0: number, v0: number, r: number): { u: number; v: number; area: number } {
  let s = 0,
    su = 0,
    sv = 0;
  for (let v = Math.max(0, Math.floor(v0 - r)); v <= Math.ceil(v0 + r); v++)
    for (let u = Math.max(0, Math.floor(u0 - r)); u <= Math.ceil(u0 + r); u++) {
      const a = img[4 * (v * W + u) + 3] / 255;
      s += a;
      su += a * u;
      sv += a * v;
    }
  return { u: su / s, v: sv / s, area: s };
}

const project = (cam: NvsCamera, x: number[]) => [cam.fx * (x[0] / x[2]) + cam.cx, cam.fy * (x[1] / x[2]) + cam.cy];

describe('ViewerRenderers pinhole projection', () => {
  for (const [W, H] of [
    [96, 128],
    [160, 120],
  ]) {
    it(`mesh squares and point splats land on the pinhole projection (${W}×${H})`, () => {
      const R = rotation([0.3, -0.5, 0.8], 2.1);
      const cam: NvsCamera = { width: W, height: H, fx: 110, fy: 110, cx: W / 2 - 7.3, cy: H / 2 + 4.6, R, t: [0.2, -0.4, 1.1] };
      // 4 × 4 squares over the image at depths 2–3 (3.7–5.5 px wide)
      const centres: number[][] = [];
      for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) {
        const z = 2 + ((i + 2 * j) % 5) / 4;
        centres.push([((i - 1.5) * 0.22 * W * z) / 110, ((j - 1.5) * 0.22 * H * z) / 110, z]);
      }
      const r = new ViewerRenderers(W, H);
      made.push(r);
      r.setContent(null, squares(cam, centres, 0.05));
      const img = r.render('mesh-unlit', cam);
      // MSAA quantises the edges (≈ ±0.1 px per square, sd 0.059 px per axis): small per square, and
      // no bias on average beyond the rasteriser's vertex snapping (see the mean check below)
      const err: number[][] = [];
      for (const c of centres) {
        const [u, v] = project(cam, c);
        const half = (110 * 0.05) / c[2];
        const b = blob(img, W, u, v, half + 3);
        err.push([b.u - u, b.v - v]);
        expect(Math.hypot(b.u - u, b.v - v)).toBeLessThan(0.2);
        expect(Math.abs(b.area / (2 * half) ** 2 - 1)).toBeLessThan(0.12);
        const i = 4 * (Math.round(v) * W + Math.round(u));
        expect(Array.from(img.subarray(i, i + 4))).toEqual([255, 255, 255, 255]);
      }
      // Mean error: 0.04 px is the quantisation noise of a 16-square mean (sd 0.015 px, ≈ 2.7 σ, at
      // random phases for these five sizes; at 4.0 and 5.5 px wide both edges of a square err alike).
      // On top comes the rasteriser's own offset: vertices snap to 1/2^SUBPIXEL_BITS px and the
      // standard 4× sample locations (odd multiples of 1/8 px) lie on that grid, so an edge within
      // half a step of a sample snaps ONTO it and the top-left fill rule decides; both edges of a
      // square then move the same way, by half a step (2^-(bits+1) px) per axis on average.
      // SwiftShader snaps to 1/16 px ⇒ 1/32 = 0.031 px (towards −u, +v): its frames here equal,
      // alpha for alpha, the exact pinhole projection rasterised that way (mean −0.045, +0.038 px
      // at 96×128 and −0.054, +0.044 at 160×120; the same squares snapped to 1/256 px: +0.022,
      // −0.006 and −0.015, +0.010). 8-bit GPUs ⇒ 0.002 px. No sign is assumed: the fill rule's
      // corner follows the back-end's framebuffer orientation. On SwiftShader the check then lets
      // uniform projection shifts of about −0.03…+0.08 px in u and −0.085…+0.025 px in v through
      // (measured); half-pixel convention errors fail per square (0.36–0.63 px).
      const gl = r.viewer.renderer.getContext();
      const snapBias = 2 ** -((gl.getParameter(gl.SUBPIXEL_BITS) as number) + 1);
      const mean = (k: number) => err.reduce((s, e) => s + e[k], 0) / err.length;
      expect(Math.abs(mean(0))).toBeLessThan(0.04 + snapBias);
      expect(Math.abs(mean(1))).toBeLessThan(0.04 + snapBias);
      // radius-sized point splats (as the app draws the dense cloud): disc of radius r·f/Z px
      const pts: PointCloud = {
        count: centres.length,
        positions: Float32Array.from(centres.flatMap((c) => toWorld(cam, c))),
        colors: new Uint8Array(3 * centres.length).fill(255),
        radii: new Float32Array(centres.length).fill(0.05),
      };
      r.setContent(pts, null);
      const pimg = r.render('points', cam);
      const perr: number[][] = [];
      for (const c of centres) {
        const [u, v] = project(cam, c);
        const rad = (110 * 0.05) / c[2];
        const b = blob(pimg, W, u, v, rad + 3);
        perr.push([b.u - u, b.v - v]);
        expect(Math.hypot(b.u - u, b.v - v)).toBeLessThan(0.35);
        expect(b.area / (Math.PI * rad * rad)).toBeGreaterThan(0.75);
        expect(b.area / (Math.PI * rad * rad)).toBeLessThan(1.3);
      }
      for (const k of [0, 1]) expect(Math.abs(perr.reduce((s, e) => s + e[k], 0) / perr.length)).toBeLessThan(0.1);
    });
  }
});
