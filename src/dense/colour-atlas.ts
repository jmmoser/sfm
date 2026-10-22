/**
 * Colour atlas for AR exports (USDZ: AR Quick Look renders no vertex colours, only textures).
 *
 * `buildColourAtlas(mesh)` simplifies the mesh to the AR budget (≤ 100 k triangles, one 2048²
 * page: the AR Quick Look rule of thumb) and bakes the colours of the UNDECIMATED mesh into a
 * texture, so the 5–6× QEM decimation does not blur them (the simplifier's own vertex colours are
 * averages along each collapsed edge):
 *
 *  - Budget: the simplifier stops above its target when no valid collapse remains (link
 *    condition, folds, boundaries — e.g. a fragment-heavy mesh of many small closed pieces). A
 *    second, relaxed pass then runs on its output (`RELAXED_SIMPLIFY`: faces may turn up to 90°,
 *    weak boundary planes); what that cannot remove stays (`stats.overBudget`: the USDZ exceeds
 *    the soft AR budget but still opens).
 *  - Layout (no general UV unwrap — xatlas measured 20 s for 40 k triangles): triangles are
 *    greedily paired with an edge neighbour of similar area (in mesh order, which follows the
 *    marching-cubes block order, so consecutive cells are spatial neighbours). A pair fills an
 *    L×L right-isosceles square split along its diagonal = the SHARED edge: both triangles map
 *    that edge to the same texels and the surface is continuous across it, so the diagonal needs
 *    no gap. Unpaired triangles (boundaries, odd leftovers) take a square alone.
 *  - Texel grid: square corners sit on texel CENTRES, so a cell holds (L + 1)² texels and
 *    bilinear sampling at full resolution never reads outside its cell; around that, a dilated
 *    `gutter` (default 1 texel, copies of the edge texels) softens bleeding at coarser mip levels
 *    (mips ≥ 1 still mix neighbouring cells: the spatial cell order keeps those colours similar).
 *  - Cell size ∝ √(3-D area) (L² = 2ρ·max area of the pair, clamped to [2, 128]), ρ found by
 *    bisection so the cells fill ~92 % of the page, then shelf-packed (sorted by size, stable);
 *    the original mesh's vertices are about uniformly dense (TSDF), so this spreads the texels
 *    evenly over the original colour samples (≈ 13 page texels per original vertex at 600 k →
 *    100 k triangles in 2048²). Equal-size cells would starve the large flat triangles QEM
 *    leaves on planar (often printed) surfaces.
 *  - Fit: when the cells do not fit the page even at the minimum size (5×5 texels with the
 *    gutter: ≈ 167 k cells in 2048²), the gutter is dropped (3×3), then the minimum side becomes
 *    1 texel (2×2 per cell, ≈ 1 M cells); beyond that it throws. `stats.gutter` / `minSide` tell.
 *  - Fill: each texel's 3-D point on the simplified surface (barycentric in its triangle;
 *    across the diagonal in the pair partner's) → the closest point of the ORIGINAL mesh →
 *    barycentric interpolation of its vertex colours. The closest triangle is found by a walk
 *    from the previous texel's triangle (texels are visited in serpentine order): while the
 *    closest point lies on an edge (a vertex), step to the triangle across that edge (around
 *    that vertex) if it is closer — a local distance minimum, so on thin double-sided parts the
 *    walk stays on the side it started from. Each cell seeds from the one-rings of the 4 nearest
 *    original vertices (k-d tree), and so does a walk that ends > 3 mean edge lengths away.
 *  - Unused page area takes the mean vertex colour (no dark halos in coarse mips).
 *
 * UVs are face-varying (6 floats per triangle: (u, v) of corners 0, 1, 2) in the IMAGE
 * convention: (0, 0) = top-left corner of the texture, u right, v down, in [0, 1] (USD's st is
 * (u, 1 − v): src/io/usdz.ts). The texture is sRGB RGBA, alpha 255.
 *
 * Measured (2026-10-21, Node 22 on a 2.8 GHz Xeon ≈ M4 single-thread for the simplifier): the
 * whole USDZ export of a 571 k-triangle fused-like mesh (this simplification to 100 k, the 2048²
 * fill, USDA, zip; JPEG encode excluded) took 4.1 s, the fill 1–1.6 s (≈ 6 closest-triangle tests
 * per texel); 0.37/255 mean colour error at random surface points vs 2.6/255 for the simplified
 * mesh's own vertex colours (colour-atlas.test.ts). The simplifier is ≈ 4–5 µs per input
 * triangle on fused (noisy) meshes but 60–120 µs on exactly planar synthetic faces (zero-cost
 * collapses). `buildColourAtlas` is synchronous; `buildColourAtlasAsync` (the engine worker's)
 * yields to the event loop about every 40 ms between phases and fill cells and rejects with a
 * CancelledError once its token is set — a worker sees a cancel message only while it yields;
 * the simplifier runs in one piece (≈ 3 s for 600 k triangles). Memory: the 16 MB RGBA page, the
 * original mesh's vertex → triangle lists (≈ 16 B per triangle) and a k-d tree (≈ 16 B per vertex).
 */
import type { CancelToken, RGBAImage, TriangleMesh } from '../core/types';
import { Yielder } from './dense-util';
import { KdTree3 } from './kdtree';
import { simplifyMesh, type SimplifyOptions } from './simplify';

export interface ColourAtlasOptions {
  /** Simplify to at most this many triangles first (AR Quick Look budget). Default 100 000. */
  maxTriangles?: number;
  /** Square page size in texels. Default 2048 (AR Quick Look: one 2048² texture set). */
  size?: number;
  /** Dilated border texels around every cell (copies of its edge texels). Default 1. */
  gutter?: number;
  /** Options of the simplifier. */
  simplify?: SimplifyOptions;
}

export interface ColourAtlasStats {
  inputTriangles: number;
  triangles: number;
  /** More than `maxTriangles` remain: the simplifier found no valid collapse left, even relaxed. */
  overBudget: boolean;
  /** The first simplification pass stopped above the budget and the relaxed one ran. */
  relaxedSimplify: boolean;
  /** Cells holding an edge-sharing triangle pair / a single triangle. */
  pairs: number;
  singles: number;
  /** Gutter used: the option's, or 0 when the cells did not fit the page with it. */
  gutter: number;
  /** Texel side L of the triangle squares: min / mean / max. */
  minSide: number;
  meanSide: number;
  maxSide: number;
  /** Fraction of the page covered by cells (incl. gutters). */
  pageFill: number;
  simplifyMs: number;
  packMs: number;
  fillMs: number;
}

export interface ColourAtlas {
  /**
   * The simplified mesh the atlas belongs to (same frame as the input; its vertex colours are the
   * simplifier's) — the input object itself when it is within `maxTriangles`.
   */
  mesh: TriangleMesh;
  /** Face-varying UVs, 6 per triangle, image convention (see the file header). */
  uvs: Float32Array;
  /** sRGB RGBA page (`size` × `size`). */
  texture: RGBAImage;
  stats: ColourAtlasStats;
}

const MIN_SIDE = 2;
const MAX_SIDE = 128;
/** Target fraction of the page covered by cells (shelf packing wastes a few %). */
const PAGE_FILL = 0.92;
/** Second simplification pass when the first stops above the budget (see the file header). */
const RELAXED_SIMPLIFY: SimplifyOptions = { minNormalCos: 0, boundaryWeight: 1 };
/** Fill texels between two yield points of `atlasSteps` (≈ 6 closest-triangle tests each). */
const YIELD_TEXELS = 1 << 14;

/** Simplify `mesh` to the AR budget and bake its full-resolution vertex colours into one texture page. */
export function buildColourAtlas(mesh: TriangleMesh, opts: ColourAtlasOptions = {}): ColourAtlas {
  const steps = atlasSteps(mesh, opts);
  for (;;) {
    const r = steps.next();
    if (r.done) return r.value;
  }
}

/**
 * `buildColourAtlas` that keeps a worker responsive: yields to the event loop about every 40 ms
 * (between phases and fill cells; the simplifier runs in one piece) and rejects with a
 * CancelledError once `cancel` is set.
 */
export async function buildColourAtlasAsync(mesh: TriangleMesh, opts: ColourAtlasOptions = {}, cancel?: CancelToken): Promise<ColourAtlas> {
  const y = new Yielder('colour atlas', cancel);
  const steps = atlasSteps(mesh, opts);
  for (;;) {
    await y.tick();
    const r = steps.next();
    if (r.done) return r.value;
  }
}

/** The atlas build; each `yield` is a point where the async driver may yield / stop. */
function* atlasSteps(mesh: TriangleMesh, opts: ColourAtlasOptions): Generator<void, ColourAtlas, void> {
  const maxTriangles = opts.maxTriangles ?? 100_000;
  const W = opts.size ?? 2048;
  const g0 = opts.gutter ?? 1;
  if (!(Number.isInteger(W) && W >= 8 && W <= 16384)) throw new RangeError(`colour atlas: invalid page size ${W}`);
  if (!(Number.isInteger(g0) && g0 >= 0 && g0 <= 8)) throw new RangeError(`colour atlas: invalid gutter ${g0}`);
  if (!(mesh.vertexCount > 0 && mesh.triangleCount > 0)) throw new RangeError('colour atlas: empty mesh');
  const t0 = performance.now();
  let dec = mesh;
  let relaxed = false;
  if (mesh.triangleCount > maxTriangles) {
    dec = simplifyMesh(mesh, maxTriangles, opts.simplify);
    if (dec.triangleCount > maxTriangles) {
      // stalled above the budget (no valid collapse left): once more, relaxed
      yield;
      relaxed = true;
      dec = simplifyMesh(dec, maxTriangles, { ...opts.simplify, ...RELAXED_SIMPLIFY });
    }
  }
  const t1 = performance.now();
  yield;

  const T = dec.triangleCount;
  const P = dec.positions;
  const I = dec.indices;
  for (let i = 0; i < 3 * T; i++) if (I[i] >= dec.vertexCount) throw new RangeError(`colour atlas: index ${I[i]} out of range`);

  // ---- areas and edge-neighbour pairing
  const area = new Float64Array(T);
  for (let t = 0; t < T; t++) area[t] = triangleArea(P, I[3 * t], I[3 * t + 1], I[3 * t + 2]);
  const nb = edgeNeighbours(I, T, dec.vertexCount);
  /** Cells: first triangle, partner (−1 = single), first triangle's corner k starting the shared edge (k, k+1). */
  const cellA: number[] = [];
  const cellB: number[] = [];
  const cellK: number[] = [];
  const matched = new Uint8Array(T);
  for (let t = 0; t < T; t++) {
    if (matched[t]) continue;
    matched[t] = 1;
    let best = -1;
    let bestK = 0;
    let bestScore = Infinity;
    for (let k = 0; k < 3; k++) {
      const n = nb[3 * t + k];
      if (n < 0 || matched[n]) continue;
      // similar areas share a square best (both get the larger one's texel density)
      const s = Math.abs(Math.log((area[n] + 1e-30) / (area[t] + 1e-30)));
      if (s < bestScore) {
        bestScore = s;
        best = n;
        bestK = k;
      }
    }
    if (best >= 0) matched[best] = 1;
    cellA.push(t);
    cellB.push(best);
    cellK.push(bestK);
  }
  const C = cellA.length;
  const cellArea = new Float64Array(C);
  for (let c = 0; c < C; c++) cellArea[c] = Math.max(area[cellA[c]], cellB[c] >= 0 ? area[cellB[c]] : 0);
  yield;

  // ---- cell sizes and shelf packing: with the gutter, else the smaller layouts (file header)
  const side = new Int32Array(C);
  const cellX = new Int32Array(C);
  const cellY = new Int32Array(C);
  const layouts: [number, number][] = [
    [g0, MIN_SIDE],
    [0, MIN_SIDE],
    [0, 1],
  ];
  let g = -1;
  for (const [i, [lg, minSide]] of layouts.entries()) {
    if (i === 1 && g0 === 0) continue; // (the first layout already had no gutter)
    if (packCells(cellArea, W, lg, minSide, side, cellX, cellY)) {
      g = lg;
      break;
    }
  }
  if (g < 0) throw new RangeError(`colour atlas: ${T} triangles (${C} cells) do not fit a ${W}² page (simplify further)`);
  const t2 = performance.now();
  yield;

  // ---- UVs (texel coordinates → image-convention UV)
  const uvs = new Float32Array(6 * T);
  const setUV = (t: number, corner: number, x: number, y: number): void => {
    uvs[6 * t + 2 * corner] = x / W;
    uvs[6 * t + 2 * corner + 1] = y / W;
  };
  /** Per cell, the 4 square-corner vertices: j (0,0), k (L,0), k1 (0,L), b's third (L,L) or −1. */
  const cv = new Int32Array(4 * C);
  for (let c = 0; c < C; c++) {
    const a = cellA[c];
    const b = cellB[c];
    const L = side[c];
    const ox = cellX[c] + g + 0.5;
    const oy = cellY[c] + g + 0.5;
    const k = b >= 0 ? cellK[c] : 0;
    const k1 = (k + 1) % 3;
    const j = (k + 2) % 3;
    setUV(a, j, ox, oy);
    setUV(a, k, ox + L, oy);
    setUV(a, k1, ox, oy + L);
    cv[4 * c] = I[3 * a + j];
    cv[4 * c + 1] = I[3 * a + k];
    cv[4 * c + 2] = I[3 * a + k1];
    cv[4 * c + 3] = -1;
    if (b >= 0) {
      const vk = I[3 * a + k];
      const vk1 = I[3 * a + k1];
      for (let m = 0; m < 3; m++) {
        const v = I[3 * b + m];
        if (v === vk) setUV(b, m, ox + L, oy);
        else if (v === vk1) setUV(b, m, ox, oy + L);
        else {
          setUV(b, m, ox + L, oy + L);
          cv[4 * c + 3] = v;
        }
      }
    }
  }

  // ---- fill from the original mesh
  const tex = new Uint8Array(4 * W * W);
  const mean = meanColour(mesh);
  for (let i = 0; i < W * W; i++) {
    tex[4 * i] = mean[0];
    tex[4 * i + 1] = mean[1];
    tex[4 * i + 2] = mean[2];
    tex[4 * i + 3] = 255;
  }
  yield;
  const finder = new ClosestTriangleFinder(mesh);
  yield;
  const rgb = new Float64Array(3);
  let texels = 0;
  for (let c = 0; c < C; c++) {
    const L = side[c];
    fillCell(tex, W, cellX[c] + g, cellY[c] + g, L, P, cv[4 * c], cv[4 * c + 1], cv[4 * c + 2], cv[4 * c + 3], finder, rgb);
    if (g > 0) dilateCell(tex, W, cellX[c], cellY[c], L + 1 + 2 * g, g);
    texels += (L + 1) * (L + 1);
    if (texels >= YIELD_TEXELS) {
      texels = 0;
      yield;
    }
  }
  const t3 = performance.now();

  let sumL = 0;
  let minL = Infinity;
  let maxL = 0;
  let used = 0;
  let pairs = 0;
  for (let c = 0; c < C; c++) {
    sumL += side[c];
    if (side[c] < minL) minL = side[c];
    if (side[c] > maxL) maxL = side[c];
    used += (side[c] + 1 + 2 * g) ** 2;
    if (cellB[c] >= 0) pairs++;
  }
  return {
    mesh: dec,
    uvs,
    texture: { width: W, height: W, data: tex },
    stats: {
      inputTriangles: mesh.triangleCount,
      triangles: T,
      overBudget: T > maxTriangles,
      relaxedSimplify: relaxed,
      pairs,
      singles: C - pairs,
      gutter: g,
      minSide: minL,
      meanSide: sumL / C,
      maxSide: maxL,
      pageFill: used / (W * W),
      simplifyMs: t1 - t0,
      packMs: t2 - t1,
      fillMs: t3 - t2,
    },
  };
}

/**
 * Cell sides for one layout (`g` gutter texels, sides ≥ `minSide`): L ∝ √area at the density ρ
 * whose cells fill PAGE_FILL of the page (bisection), shelf-packed; ρ shrinks until the shelves
 * fit, down to every cell at `minSide`. Writes `side` / `cellX` / `cellY`; false when even that
 * does not fit.
 */
function packCells(cellArea: Float64Array, W: number, g: number, minSide: number, side: Int32Array, cellX: Int32Array, cellY: Int32Array): boolean {
  const C = cellArea.length;
  const maxSide = Math.max(minSide, Math.min(MAX_SIDE, Math.floor(W / 4) - 1 - 2 * g));
  const sideOf = (rho: number, c: number): number => {
    const L = Math.round(Math.sqrt(2 * rho * cellArea[c]));
    return L < minSide ? minSide : L > maxSide ? maxSide : L;
  };
  const budget = PAGE_FILL * W * W;
  const pageUse = (rho: number): number => {
    let s = 0;
    for (let c = 0; c < C; c++) {
      const S = sideOf(rho, c) + 1 + 2 * g;
      s += S * S;
    }
    return s;
  };
  if (pageUse(0) > W * W) return false;
  let totalArea = 0;
  for (let c = 0; c < C; c++) totalArea += cellArea[c];
  let lo = 0;
  let hi = totalArea > 0 ? (16 * W * W) / totalArea : 1;
  while (pageUse(hi) < budget && hi < 1e300 && totalArea > 0) hi *= 4;
  for (let it = 0; it < 48; it++) {
    const mid = lo === 0 ? hi / 1024 : Math.sqrt(lo * hi);
    if (pageUse(mid) <= budget) lo = mid;
    else hi = mid;
    if (hi / Math.max(lo, 1e-300) < 1.002) break;
  }
  let rho = lo;
  for (let attempt = 0; ; attempt++) {
    for (let c = 0; c < C; c++) side[c] = sideOf(rho, c);
    if (shelfPack(side, 1 + 2 * g, W, cellX, cellY)) return true;
    if (rho === 0) return false;
    // shrink; the last try puts every cell at the minimum side
    rho = attempt < 40 ? rho * 0.93 : 0;
  }
}

/**
 * Fill the (L + 1)² texels of one cell, top-left texel (x0, y0), from the original mesh (see the
 * file header: Fill). Square corners: vj (0, 0), vk (L, 0), vk1 (0, L), vb (L, L) (−1: single).
 */
function fillCell(tex: Uint8Array, W: number, x0: number, y0: number, L: number, P: Float32Array, vj: number, vk: number, vk1: number, vb: number, finder: ClosestTriangleFinder, rgb: Float64Array): void {
  const pair = vb >= 0;
  finder.reset();
  for (let y = 0; y <= L; y++) {
    for (let s = 0; s <= L; s++) {
      const x = y & 1 ? L - s : s; // serpentine: consecutive texels are neighbours
      let px: number;
      let py: number;
      let pz: number;
      if (x + y <= L || !pair) {
        let ux = x;
        let uy = y;
        if (x + y > L) {
          // single triangle: the empty half takes the closest hypotenuse point (bilinear
          // sampling along the hypotenuse reads the texels next to it)
          ux = (x - y + L) / 2;
          uy = L - ux;
        }
        const wk = ux / L;
        const wk1 = uy / L;
        const wj = 1 - wk - wk1;
        px = wj * P[3 * vj] + wk * P[3 * vk] + wk1 * P[3 * vk1];
        py = wj * P[3 * vj + 1] + wk * P[3 * vk + 1] + wk1 * P[3 * vk1 + 1];
        pz = wj * P[3 * vj + 2] + wk * P[3 * vk + 2] + wk1 * P[3 * vk1 + 2];
      } else {
        // partner triangle: third vertex at (L, L), shared vertices at (L, 0) and (0, L)
        const wb = (x + y) / L - 1;
        const wk = 1 - y / L;
        const wk1 = 1 - x / L;
        px = wb * P[3 * vb] + wk * P[3 * vk] + wk1 * P[3 * vk1];
        py = wb * P[3 * vb + 1] + wk * P[3 * vk + 1] + wk1 * P[3 * vk1 + 1];
        pz = wb * P[3 * vb + 2] + wk * P[3 * vk + 2] + wk1 * P[3 * vk1 + 2];
      }
      finder.colourAt(px, py, pz, rgb);
      const o = 4 * ((y0 + y) * W + x0 + x);
      tex[o] = rgb[0];
      tex[o + 1] = rgb[1];
      tex[o + 2] = rgb[2];
    }
  }
}

function triangleArea(P: Float32Array, a: number, b: number, c: number): number {
  const ux = P[3 * b] - P[3 * a];
  const uy = P[3 * b + 1] - P[3 * a + 1];
  const uz = P[3 * b + 2] - P[3 * a + 2];
  const vx = P[3 * c] - P[3 * a];
  const vy = P[3 * c + 1] - P[3 * a + 1];
  const vz = P[3 * c + 2] - P[3 * a + 2];
  const x = uy * vz - uz * vy;
  const y = uz * vx - ux * vz;
  const z = ux * vy - uy * vx;
  const s = 0.5 * Math.sqrt(x * x + y * y + z * z);
  return s === s && s < Infinity ? s : 0;
}

/**
 * Per triangle and corner k: the triangle sharing edge (corner k, corner k+1), or −1 at a
 * boundary, a non-manifold edge (≥ 3 triangles) or a degenerate triangle.
 */
function edgeNeighbours(I: Uint32Array, T: number, V: number): Int32Array {
  const nb = new Int32Array(3 * T).fill(-1);
  /** Corner slot (3t + k) → the slot sharing its edge. */
  const partner = new Int32Array(3 * T).fill(-1);
  /** Edge key → its first slot (≥ 0); −2 − first slot once paired; −1 once non-manifold. */
  const seen = new Map<number, number>();
  for (let t = 0; t < T; t++) {
    const a = I[3 * t];
    const b = I[3 * t + 1];
    const c = I[3 * t + 2];
    if (a === b || b === c || a === c) continue;
    for (let k = 0; k < 3; k++) {
      const u = I[3 * t + k];
      const v = I[3 * t + ((k + 1) % 3)];
      const key = u < v ? u * V + v : v * V + u;
      const slot = 3 * t + k;
      const prev = seen.get(key);
      if (prev === undefined) seen.set(key, slot);
      else if (prev >= 0) {
        partner[prev] = slot;
        partner[slot] = prev;
        nb[prev] = t;
        nb[slot] = (prev / 3) | 0;
        seen.set(key, -2 - prev);
      } else if (prev <= -2) {
        const p = -2 - prev;
        nb[p] = -1;
        nb[partner[p]] = -1;
        seen.set(key, -1);
      }
    }
  }
  return nb;
}

/**
 * Shelf packing of squares (side + extra texels) into a W×W page, largest first (stable within
 * a size, keeping the spatial cell order). Writes top-left corners; false when they do not fit.
 */
function shelfPack(side: Int32Array, extra: number, W: number, outX: Int32Array, outY: Int32Array): boolean {
  const C = side.length;
  const order = Array.from({ length: C }, (_, i) => i);
  order.sort((a, b) => side[b] - side[a] || a - b);
  let x = 0;
  let y = 0;
  let shelf = 0;
  for (const c of order) {
    const S = side[c] + extra;
    if (x + S > W) {
      y += shelf;
      x = 0;
      shelf = 0;
    }
    if (y + S > W || S > W) return false;
    outX[c] = x;
    outY[c] = y;
    x += S;
    if (S > shelf) shelf = S;
  }
  return true;
}

/** Copy the edge texels of the cell's inner (S − 2g)² block outwards over its g-texel gutter. */
function dilateCell(tex: Uint8Array, W: number, cx: number, cy: number, S: number, g: number): void {
  const inner0 = g;
  const inner1 = S - 1 - g;
  for (let y = 0; y < S; y++) {
    const sy = y < inner0 ? inner0 : y > inner1 ? inner1 : y;
    for (let x = 0; x < S; x++) {
      if (y >= inner0 && y <= inner1 && x >= inner0 && x <= inner1) {
        x = inner1; // skip the inner block
        continue;
      }
      const sx = x < inner0 ? inner0 : x > inner1 ? inner1 : x;
      const o = 4 * ((cy + y) * W + cx + x);
      const s = 4 * ((cy + sy) * W + cx + sx);
      tex[o] = tex[s];
      tex[o + 1] = tex[s + 1];
      tex[o + 2] = tex[s + 2];
    }
  }
}

function isFinite3(P: Float32Array, v: number): boolean {
  const x = P[3 * v];
  const y = P[3 * v + 1];
  const z = P[3 * v + 2];
  return x - x === 0 && y - y === 0 && z - z === 0;
}

function meanColour(mesh: TriangleMesh): [number, number, number] {
  const n = mesh.vertexCount;
  const step = Math.max(1, Math.floor(n / 65536));
  let r = 0;
  let gg = 0;
  let b = 0;
  let k = 0;
  for (let i = 0; i < n; i += step) {
    r += mesh.colors[3 * i];
    gg += mesh.colors[3 * i + 1];
    b += mesh.colors[3 * i + 2];
    k++;
  }
  return k ? [Math.round(r / k), Math.round(gg / k), Math.round(b / k)] : [128, 128, 128];
}

/**
 * Closest point on the original mesh by a walk across the closest point's edge / vertex (see the
 * file header). `closest` / `colourAt` continue from the triangle of the previous query; `reset`
 * forgets it.
 */
export class ClosestTriangleFinder {
  private readonly P: Float32Array;
  private readonly I: Uint32Array;
  private readonly C: Uint8Array;
  /** Vertex → incident triangles (CSR). */
  private readonly ringStart: Uint32Array;
  private readonly ring: Uint32Array;
  /** k-d tree over the finite vertices that belong to a triangle; `treeVertex` maps back. */
  private readonly tree: KdTree3;
  private readonly treeVertex: Uint32Array;
  /** Squared distance beyond which a walk result is double-checked with a fresh k-d tree seed. */
  private readonly reseedSq: number;
  private cur = -1;
  private readonly bary = new Float64Array(3);
  /** Feature of the last `distTo` closest point: 0 interior, 1–3 corner 0–2, 4–6 edge (0,1) (1,2) (2,0). */
  private region = 0;
  private readonly kd = new Float64Array(8);
  private readonly ki = new Int32Array(8);

  constructor(mesh: TriangleMesh) {
    const V = mesh.vertexCount;
    const T = mesh.triangleCount;
    this.P = mesh.positions;
    this.I = mesh.indices;
    this.C = mesh.colors;
    const deg = new Uint32Array(V + 1);
    for (let i = 0; i < 3 * T; i++) deg[this.I[i] + 1]++;
    let nTree = 0;
    for (let v = 0; v < V; v++) if (deg[v + 1] > 0 && isFinite3(this.P, v)) nTree++;
    const pts = new Float32Array(3 * nTree);
    this.treeVertex = new Uint32Array(nTree);
    for (let v = 0, k = 0; v < V; v++) {
      if (!(deg[v + 1] > 0 && isFinite3(this.P, v))) continue;
      pts[3 * k] = this.P[3 * v];
      pts[3 * k + 1] = this.P[3 * v + 1];
      pts[3 * k + 2] = this.P[3 * v + 2];
      this.treeVertex[k++] = v;
    }
    this.tree = new KdTree3(pts, nTree);
    for (let v = 0; v < V; v++) deg[v + 1] += deg[v];
    this.ringStart = deg;
    const fillPos = deg.slice(0, V);
    this.ring = new Uint32Array(3 * T);
    for (let t = 0; t < T; t++) for (let k = 0; k < 3; k++) this.ring[fillPos[this.I[3 * t + k]]++] = t;
    // mean edge length (sampled): a walk ending further than a few edges away is suspicious
    let s = 0;
    let n = 0;
    const step = Math.max(1, Math.floor(T / 4096));
    for (let t = 0; t < T; t += step) {
      const a = this.I[3 * t];
      const b = this.I[3 * t + 1];
      const d = Math.hypot(this.P[3 * a] - this.P[3 * b], this.P[3 * a + 1] - this.P[3 * b + 1], this.P[3 * a + 2] - this.P[3 * b + 2]);
      if (d === d && d < Infinity) {
        s += d;
        n++;
      }
    }
    this.reseedSq = (3 * (n ? s / n : 0)) ** 2;
  }

  /** Forget the previous query's triangle (the next query seeds from the k-d tree). */
  reset(): void {
    this.cur = -1;
  }

  /** Index of the closest original triangle to (x, y, z) (−1: no usable triangle); `bary` gets its barycentrics. */
  closest(x: number, y: number, z: number): number {
    const prev = this.cur;
    let prevD = Infinity;
    if (prev >= 0) {
      prevD = this.walk(x, y, z, prev);
      if (prevD <= this.reseedSq) return this.cur;
    }
    // (re)seed from the nearest original vertices; keep the better of the two walks
    const walked = this.cur;
    const s = this.seed(x, y, z);
    if (s >= 0 && this.walk(x, y, z, s) <= prevD) return this.cur;
    this.cur = walked;
    if (walked >= 0) this.distTo(x, y, z, walked);
    return this.cur;
  }

  /** sRGB colour of the original mesh at the closest point to (x, y, z), rounded to 0..255. */
  colourAt(x: number, y: number, z: number, out: Float64Array): void {
    const t = this.closest(x, y, z);
    if (t < 0) {
      out[0] = out[1] = out[2] = 128;
      return;
    }
    const C = this.C;
    const a = 3 * this.I[3 * t];
    const b = 3 * this.I[3 * t + 1];
    const c = 3 * this.I[3 * t + 2];
    const u = this.bary[0];
    const v = this.bary[1];
    const w = this.bary[2];
    out[0] = Math.round(u * C[a] + v * C[b] + w * C[c]);
    out[1] = Math.round(u * C[a + 1] + v * C[b + 1] + w * C[c + 1]);
    out[2] = Math.round(u * C[a + 2] + v * C[b + 2] + w * C[c + 2]);
  }

  /** Best triangle in the one-rings of the nearest vertices, or −1. */
  private seed(x: number, y: number, z: number): number {
    const n = this.tree.knn(x, y, z, 4, this.kd, this.ki);
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < n; i++) {
      const v = this.treeVertex[this.ki[i]];
      for (let r = this.ringStart[v]; r < this.ringStart[v + 1]; r++) {
        const t = this.ring[r];
        const d = this.distTo(x, y, z, t);
        if (d < bestD) {
          bestD = d;
          best = t;
        }
      }
    }
    return best;
  }

  /**
   * Walk from triangle `t` to a local distance minimum: while the closest point is on an edge
   * (vertex), try the triangle across that edge (the triangles around that vertex). Sets `cur` /
   * `bary`, returns the squared distance.
   */
  private walk(x: number, y: number, z: number, t: number): number {
    const I = this.I;
    let d = this.distTo(x, y, z, t);
    for (let it = 0; it < 64 && this.region !== 0; it++) {
      const reg = this.region;
      let u: number;
      let w = -1;
      if (reg <= 3) u = I[3 * t + reg - 1];
      else {
        u = I[3 * t + reg - 4];
        w = I[3 * t + ((reg - 3) % 3)];
      }
      let best = t;
      let bestD = d;
      for (let r = this.ringStart[u]; r < this.ringStart[u + 1]; r++) {
        const s = this.ring[r];
        if (s === t) continue;
        if (w >= 0 && I[3 * s] !== w && I[3 * s + 1] !== w && I[3 * s + 2] !== w) continue;
        const ds = this.distTo(x, y, z, s);
        if (ds < bestD) {
          bestD = ds;
          best = s;
        }
      }
      if (best === t) {
        this.distTo(x, y, z, t); // (restore t's barycentrics)
        break;
      }
      t = best;
      d = this.distTo(x, y, z, t);
    }
    this.cur = t;
    return d;
  }

  /**
   * Squared distance from (x, y, z) to triangle t; barycentrics of the closest point in `bary`
   * (Ericson, Real-Time Collision Detection §5.1.5).
   */
  private distTo(px: number, py: number, pz: number, t: number): number {
    const P = this.P;
    const ia = 3 * this.I[3 * t];
    const ib = 3 * this.I[3 * t + 1];
    const ic = 3 * this.I[3 * t + 2];
    const ax = P[ia], ay = P[ia + 1], az = P[ia + 2];
    const abx = P[ib] - ax, aby = P[ib + 1] - ay, abz = P[ib + 2] - az;
    const acx = P[ic] - ax, acy = P[ic + 1] - ay, acz = P[ic + 2] - az;
    const apx = px - ax, apy = py - ay, apz = pz - az;
    const B = this.bary;
    let v: number;
    let w: number;
    const d1 = abx * apx + aby * apy + abz * apz;
    const d2 = acx * apx + acy * apy + acz * apz;
    let region = 0;
    if (d1 <= 0 && d2 <= 0) {
      v = 0;
      w = 0;
      region = 1;
    } else {
      const d3 = d1 - (abx * abx + aby * aby + abz * abz);
      const d4 = d2 - (abx * acx + aby * acy + abz * acz);
      // (bp = ap − ab: d3 = ab·bp, d4 = ac·bp)
      if (d3 >= 0 && d4 <= d3) {
        v = 1;
        w = 0;
        region = 2;
      } else {
        const vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) {
          v = d1 / (d1 - d3);
          w = 0;
          region = 4;
        } else {
          const d5 = d1 - (abx * acx + aby * acy + abz * acz);
          const d6 = d2 - (acx * acx + acy * acy + acz * acz);
          // (cp = ap − ac: d5 = ab·cp, d6 = ac·cp)
          if (d6 >= 0 && d5 <= d6) {
            v = 0;
            w = 1;
            region = 3;
          } else {
            const vb = d5 * d2 - d1 * d6;
            if (vb <= 0 && d2 >= 0 && d6 <= 0) {
              v = 0;
              w = d2 / (d2 - d6);
              region = 6;
            } else {
              const va = d3 * d6 - d5 * d4;
              if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
                w = (d4 - d3) / (d4 - d3 + (d5 - d6));
                v = 1 - w;
                region = 5;
              } else {
                const sum = va + vb + vc;
                if (sum > 0) {
                  v = vb / sum;
                  w = vc / sum;
                } else {
                  v = 0; // degenerate triangle: its first vertex
                  w = 0;
                  region = 1;
                }
              }
            }
          }
        }
      }
    }
    B[0] = 1 - v - w;
    B[1] = v;
    B[2] = w;
    this.region = region;
    const qx = ax + v * abx + w * acx - px;
    const qy = ay + v * aby + w * acy - py;
    const qz = az + v * abz + w * acz - pz;
    const d = qx * qx + qy * qy + qz * qz;
    return d === d ? d : Infinity;
  }
}
