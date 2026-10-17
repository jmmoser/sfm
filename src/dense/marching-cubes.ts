/**
 * Marching cubes over a sparse voxel-block volume (8³ blocks), used by the TSDF mesher.
 *
 * Case tables are GENERATED at load time instead of transcribed: on every cube face the
 * edge crossings are paired by walking the face counter-clockwise (seen from outside the
 * cube) and joining each "entering" crossing (outside → inside corner) with the next
 * "exiting" one. This resolves ambiguous faces by always separating the inside corners,
 * a rule that depends only on the face's four corner signs, so the two cubes sharing a
 * face always agree → the surface is watertight (no cracks, unlike the classic Lorensen
 * table whose complementary cases disagree on ambiguous faces). The face segments link
 * into closed loops which are fan-triangulated. Orientation: "inside" (SDF < 0) on the
 * right of each segment seen from outside ⇒ triangles are counter-clockwise when seen from
 * the positive (free-space) side, i.e. face normals point along +∇SDF (towards the cameras).
 *
 * Vertices live on voxel edges and are welded across block borders through a hash keyed
 * by (owner voxel, axis); normals come from the TSDF gradient (central differences,
 * interpolated along the edge), colours are interpolated from the voxel colours.
 *
 * Voxel (gx, gy, gz) has its centre at world (gx, gy, gz) · voxelSize; block (bx, by, bz)
 * holds voxels 8b … 8b+7 per axis; local voxel index = lx + 8 (ly + 8 lz).
 */
import type { TriangleMesh } from '../core/types';
import { Int3HashMap, growF32, growU32, growU8 } from './dense-util';

export const BLOCK_SIZE = 8;
export const BLOCK_VOXELS = 512;
/** Int16 TSDF value = round(clamp(sdf / τ, −1, 1) · TSDF_SCALE). */
export const TSDF_SCALE = 32767;
/** Uint16 weight units per nominal observation. */
export const WEIGHT_UNIT = 64;
/** Uint16 colour = round(colour8 · COLOR_SCALE) (8.8 fixed point). */
export const COLOR_SCALE = 256;

/** Read-only view of a sparse block volume (implemented by `TsdfVolume`). */
export interface SparseVoxelVolume {
  readonly voxelSize: number;
  readonly blockCount: number;
  /** 3 per block. */
  readonly blockCoords: Int32Array;
  /** BLOCK_VOXELS per block, normalised TSDF × TSDF_SCALE. */
  readonly tsdf: Int16Array;
  /** BLOCK_VOXELS per block, WEIGHT_UNIT per nominal observation; 0 = unobserved. */
  readonly weight: Uint16Array;
  /** 3 · BLOCK_VOXELS per block (RGB × COLOR_SCALE), or null. */
  readonly color: Uint16Array | null;
  /** Block index or −1. */
  findBlock(bx: number, by: number, bz: number): number;
}

// ---------------------------------------------------------------------------
// Table generation

/** Corner c ↔ offset (c & 1, (c >> 1) & 1, (c >> 2) & 1). */
export const MC_EDGE_CORNERS: ReadonlyArray<readonly [number, number]> = (() => {
  const edges: [number, number][] = [];
  for (let a = 0; a < 3; a++) {
    const b = (a + 1) % 3;
    const c = (a + 2) % 3;
    for (let k = 0; k < 4; k++) {
      const c0 = ((k & 1) << b) | ((k >> 1) << c);
      edges.push([c0, c0 | (1 << a)]);
    }
  }
  return edges;
})();

/** Axis of each edge (edge e = axis·4 + k). */
export const MC_EDGE_AXIS: ReadonlyArray<number> = MC_EDGE_CORNERS.map((_, e) => (e / 4) | 0);

interface McTables {
  /**
   * CSR: triangles of case i are triEdges[triStart[i] … triStart[i+1]) as vertex triples;
   * 0–11 = edge vertices, 12 + L = centre vertex of the case's L-th centred loop.
   */
  triStart: Int32Array;
  triEdges: Int8Array;
  /** 12-bit mask of crossing edges per case. */
  edgeMask: Int32Array;
  /** CSR: loops that need a centre vertex (edge lists), per case. */
  centreStart: Int32Array;
  centreLoopStart: Int32Array;
  centreEdges: Int8Array;
}

/** Faces (axis·2 + side) containing edge e. */
function edgeFaces(e: number): [number, number] {
  const [c0] = MC_EDGE_CORNERS[e];
  const a = MC_EDGE_AXIS[e];
  const out: number[] = [];
  for (let d = 0; d < 3; d++) if (d !== a) out.push(2 * d + ((c0 >> d) & 1));
  return [out[0], out[1]];
}

/** True if two cube edges lie on a common face (a vertex pair any neighbouring cube could also join). */
function shareFace(e1: number, e2: number): boolean {
  const f1 = edgeFaces(e1);
  const f2 = edgeFaces(e2);
  return f1[0] === f2[0] || f1[0] === f2[1] || f1[1] === f2[0] || f1[1] === f2[1];
}

/**
 * Triangulate a polygon (loop order kept) so that every diagonal joins two edges that
 * share no cube face — such a diagonal can belong to no other cube, which makes the mesh
 * edge-manifold. Returns null if impossible. Prefers well-spread triangles (balanced splits).
 */
function triangulateLoop(p: number[]): number[] | null {
  const n = p.length;
  if (n === 3) return [p[0], p[1], p[2]];
  // apex k for base (p0, p[n-1]); try central apexes first
  const order: number[] = [];
  for (let k = 1; k <= n - 2; k++) order.push(k);
  order.sort((a, b) => Math.abs(a - (n - 1) / 2) - Math.abs(b - (n - 1) / 2));
  for (const k of order) {
    if (k !== 1 && shareFace(p[0], p[k])) continue;
    if (k !== n - 2 && shareFace(p[k], p[n - 1])) continue;
    const left = k >= 2 ? triangulateLoop(p.slice(0, k + 1)) : [];
    if (left === null) continue;
    const right = k <= n - 3 ? triangulateLoop(p.slice(k)) : [];
    if (right === null) continue;
    return [...left, p[0], p[k], p[n - 1], ...right];
  }
  return null;
}

function buildTables(): McTables {
  const edgeOf = new Int8Array(64).fill(-1);
  MC_EDGE_CORNERS.forEach(([c0, c1], e) => {
    edgeOf[c0 * 8 + c1] = e;
    edgeOf[c1 * 8 + c0] = e;
  });
  // faces: corner cycles counter-clockwise seen from outside
  const faces: number[][] = [];
  for (let a = 0; a < 3; a++) {
    const b = (a + 1) % 3;
    const c = (a + 2) % 3;
    for (let s = 0; s < 2; s++) {
      const quad = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ].map(([ub, uc]) => (s << a) | (ub << b) | (uc << c));
      if (s === 0) quad.reverse();
      faces.push(quad);
    }
  }
  const triStart = new Int32Array(257);
  const tri: number[] = [];
  const edgeMask = new Int32Array(256);
  const centreStart = new Int32Array(257);
  const centreLoopStart: number[] = [];
  const centreEdges: number[] = [];
  for (let cs = 0; cs < 256; cs++) {
    triStart[cs] = tri.length;
    centreStart[cs] = centreLoopStart.length;
    const inside = (c: number): boolean => ((cs >> c) & 1) === 1;
    const next = new Int8Array(12).fill(-1);
    for (const f of faces) {
      const ent: boolean[] = [];
      const ext: boolean[] = [];
      const fe: number[] = [];
      for (let k = 0; k < 4; k++) {
        const ca = f[k];
        const cb = f[(k + 1) % 4];
        fe.push(edgeOf[ca * 8 + cb]);
        ent.push(!inside(ca) && inside(cb));
        ext.push(inside(ca) && !inside(cb));
      }
      for (let k = 0; k < 4; k++) {
        if (!ent[k]) continue;
        for (let m = 1; m < 4; m++) {
          const kk = (k + m) % 4;
          if (ext[kk]) {
            next[fe[k]] = fe[kk];
            break;
          }
        }
      }
    }
    let mask = 0;
    for (let e = 0; e < 12; e++) if (next[e] >= 0) mask |= 1 << e;
    edgeMask[cs] = mask;
    const seen = new Uint8Array(12);
    for (let e = 0; e < 12; e++) {
      if (next[e] < 0 || seen[e]) continue;
      const loop: number[] = [];
      let cur = e;
      while (!seen[cur]) {
        seen[cur] = 1;
        loop.push(cur);
        cur = next[cur];
        if (cur < 0) throw new Error('marching cubes: open loop (table generation bug)');
      }
      const t = triangulateLoop(loop);
      if (t) {
        tri.push(...t);
      } else {
        // centre vertex: fan around it (all spokes are interior to this cube)
        const cv = 12 + (centreLoopStart.length - centreStart[cs]);
        centreLoopStart.push(centreEdges.length);
        centreEdges.push(...loop);
        for (let i = 0; i < loop.length; i++) tri.push(cv, loop[i], loop[(i + 1) % loop.length]);
      }
    }
  }
  triStart[256] = tri.length;
  centreStart[256] = centreLoopStart.length;
  centreLoopStart.push(centreEdges.length);
  return {
    triStart,
    triEdges: Int8Array.from(tri),
    edgeMask,
    centreStart,
    centreLoopStart: Int32Array.from(centreLoopStart),
    centreEdges: Int8Array.from(centreEdges),
  };
}

export const MC_TABLES: Readonly<McTables> = buildTables();

// ---------------------------------------------------------------------------
// Extraction

export interface MarchingCubesOptions {
  /** Minimum voxel weight (nominal observations) for a cube corner to count as observed. Default 0 (any observation). */
  minWeight?: number;
  /** Interpolate voxel colours (default true when the volume has colours). */
  color?: boolean;
  /** Skip edges whose both ends are saturated (|tsdf| ≥ this, normalised). Default 0.999. */
  saturation?: number;
  /** Optional cooperative yield, called every few hundred blocks. */
  onBlock?: (done: number, total: number) => void;
}

const W11 = 11; // window: local −1 … 9 per axis
const W11SQ = W11 * W11;
const WIN = W11 * W11SQ;

/** Extract the zero level set of a sparse TSDF volume as a triangle mesh. */
export function marchingCubes(vol: SparseVoxelVolume, opts: MarchingCubesOptions = {}): TriangleMesh {
  const vs = vol.voxelSize;
  const useColor = (opts.color ?? true) && !!vol.color;
  const minW = Math.max(1, Math.ceil((opts.minWeight ?? 0) * WEIGHT_UNIT));
  const sat = opts.saturation ?? 0.999;
  const { triStart, triEdges, edgeMask, centreStart, centreLoopStart, centreEdges } = MC_TABLES;
  const tsdf = vol.tsdf;
  const weight = vol.weight;
  const colorArr = vol.color;

  const wsrc = new Int32Array(WIN);
  const wv = new Float32Array(WIN);
  const wok = new Uint8Array(WIN);
  const nbBlocks = new Int32Array(27);
  // window coordinate → (block offset, local coordinate)
  const wOff = new Int32Array(W11);
  const wLoc = new Int32Array(W11);
  for (let i = 0; i < W11; i++) {
    const l = i - 1;
    const o = l < 0 ? -1 : l > 7 ? 1 : 0;
    wOff[i] = o + 1;
    wLoc[i] = l - 8 * o;
  }
  const cornerOff = new Int32Array(8);
  for (let c = 0; c < 8; c++) cornerOff[c] = (c & 1) + W11 * ((c >> 1) & 1) + W11SQ * ((c >> 2) & 1);
  const edgeC0 = new Int32Array(12);
  const edgeC1 = new Int32Array(12);
  const edgeAxis = new Int32Array(12);
  for (let e = 0; e < 12; e++) {
    edgeC0[e] = MC_EDGE_CORNERS[e][0];
    edgeC1[e] = MC_EDGE_CORNERS[e][1];
    edgeAxis[e] = MC_EDGE_AXIS[e];
  }
  const axisStride = [1, W11, W11SQ];

  const edgeMap = new Int3HashMap(1 << 16);
  let vcap = 1 << 16;
  let pos = new Float32Array(3 * vcap);
  let nrm = new Float32Array(3 * vcap);
  let col = new Uint8Array(3 * vcap);
  let nVert = 0;
  let idx = new Uint32Array(3 * (1 << 17));
  let nIdx = 0;
  const cubeVert = new Int32Array(24);
  const cv = new Float32Array(8);

  const grad = (w: number, out: Float64Array, o: number): void => {
    for (let a = 0; a < 3; a++) {
      const s = axisStride[a];
      const okP = wok[w + s] === 1;
      const okM = wok[w - s] === 1;
      let g = 0;
      if (okP && okM) g = 0.5 * (wv[w + s] - wv[w - s]);
      else if (okP) g = wv[w + s] - wv[w];
      else if (okM) g = wv[w] - wv[w - s];
      out[o + a] = g;
    }
  };
  const g0 = new Float64Array(3);
  const g1 = new Float64Array(3);

  const nb = vol.blockCount;
  for (let b = 0; b < nb; b++) {
    if (opts.onBlock && (b & 255) === 0) opts.onBlock(b, nb);
    const bx = vol.blockCoords[3 * b];
    const by = vol.blockCoords[3 * b + 1];
    const bz = vol.blockCoords[3 * b + 2];
    for (let dz = 0; dz < 3; dz++) {
      for (let dy = 0; dy < 3; dy++) {
        for (let dx = 0; dx < 3; dx++) {
          nbBlocks[dx + 3 * (dy + 3 * dz)] = dx === 1 && dy === 1 && dz === 1 ? b : vol.findBlock(bx + dx - 1, by + dy - 1, bz + dz - 1);
        }
      }
    }
    // gather the 11³ window
    let hasNeg = false;
    let hasPos = false;
    for (let wz = 0; wz < W11; wz++) {
      const oz = wOff[wz];
      const lz = wLoc[wz];
      for (let wy = 0; wy < W11; wy++) {
        const oy = wOff[wy];
        const ly = wLoc[wy];
        const rowBase = W11 * (wy + W11 * wz);
        for (let wx = 0; wx < W11; wx++) {
          const blk = nbBlocks[wOff[wx] + 3 * (oy + 3 * oz)];
          const wi = rowBase + wx;
          if (blk < 0) {
            wsrc[wi] = -1;
            wok[wi] = 0;
            continue;
          }
          const vi = blk * BLOCK_VOXELS + wLoc[wx] + 8 * (ly + 8 * lz);
          wsrc[wi] = vi;
          if (weight[vi] >= minW) {
            wok[wi] = 1;
            const v = tsdf[vi] / TSDF_SCALE;
            wv[wi] = v;
            // sign census over the cube corners of this block (window 1 … 9)
            if (wx >= 1 && wy >= 1 && wz >= 1 && wx <= 9 && wy <= 9 && wz <= 9) {
              if (v < 0) hasNeg = true;
              else hasPos = true;
            }
          } else {
            wok[wi] = 0;
          }
        }
      }
    }
    if (!hasNeg || !hasPos) continue;

    const gx0 = 8 * bx;
    const gy0 = 8 * by;
    const gz0 = 8 * bz;
    for (let lz = 0; lz < 8; lz++) {
      for (let ly = 0; ly < 8; ly++) {
        for (let lx = 0; lx < 8; lx++) {
          const base = lx + 1 + W11 * (ly + 1 + W11 * (lz + 1));
          let cs = 0;
          let ok = true;
          for (let c = 0; c < 8; c++) {
            const w = base + cornerOff[c];
            if (!wok[w]) {
              ok = false;
              break;
            }
            const v = wv[w];
            cv[c] = v;
            if (v < 0) cs |= 1 << c;
          }
          if (!ok || cs === 0 || cs === 255) continue;
          const em = edgeMask[cs];
          // ghost-surface rejection: a crossing between two saturated values
          let ghost = false;
          for (let e = 0; e < 12; e++) {
            if (!((em >> e) & 1)) continue;
            if (Math.abs(cv[edgeC0[e]]) >= sat && Math.abs(cv[edgeC1[e]]) >= sat) {
              ghost = true;
              break;
            }
          }
          if (ghost) continue;
          for (let e = 0; e < 12; e++) {
            if (!((em >> e) & 1)) continue;
            const c0 = edgeC0[e];
            const a = edgeAxis[e];
            const ogx = gx0 + lx + (c0 & 1);
            const ogy = gy0 + ly + ((c0 >> 1) & 1);
            const ogz = gz0 + lz + ((c0 >> 2) & 1);
            const vid = edgeMap.getOrInsert(ogx, ogy, ogz * 3 + a, nVert);
            if (edgeMap.inserted) {
              // create the vertex
              const va = cv[c0];
              const vb = cv[edgeC1[e]];
              let t = va / (va - vb);
              if (!(t >= 0)) t = 0;
              else if (t > 1) t = 1;
              if (nVert >= vcap) {
                vcap *= 2;
                pos = growF32(pos, 3 * vcap);
                nrm = growF32(nrm, 3 * vcap);
                col = growU8(col, 3 * vcap);
              }
              const p3 = 3 * nVert;
              pos[p3] = (ogx + (a === 0 ? t : 0)) * vs;
              pos[p3 + 1] = (ogy + (a === 1 ? t : 0)) * vs;
              pos[p3 + 2] = (ogz + (a === 2 ? t : 0)) * vs;
              const w0 = base + cornerOff[c0];
              const w1 = base + cornerOff[edgeC1[e]];
              grad(w0, g0, 0);
              grad(w1, g1, 0);
              let nx = g0[0] + t * (g1[0] - g0[0]);
              let ny = g0[1] + t * (g1[1] - g0[1]);
              let nz = g0[2] + t * (g1[2] - g0[2]);
              const nl = Math.hypot(nx, ny, nz);
              if (nl > 0) {
                nx /= nl;
                ny /= nl;
                nz /= nl;
              } else {
                nx = a === 0 ? (vb > va ? 1 : -1) : 0;
                ny = a === 1 ? (vb > va ? 1 : -1) : 0;
                nz = a === 2 ? (vb > va ? 1 : -1) : 0;
              }
              nrm[p3] = nx;
              nrm[p3 + 1] = ny;
              nrm[p3 + 2] = nz;
              if (useColor) {
                const s0 = 3 * wsrc[w0];
                const s1 = 3 * wsrc[w1];
                for (let k = 0; k < 3; k++) {
                  const c = (colorArr![s0 + k] + t * (colorArr![s1 + k] - colorArr![s0 + k])) / COLOR_SCALE;
                  col[p3 + k] = c < 0 ? 0 : c > 255 ? 255 : Math.round(c);
                }
              } else {
                col[p3] = 200;
                col[p3 + 1] = 200;
                col[p3 + 2] = 200;
              }
              nVert++;
            }
            cubeVert[e] = vid;
          }
          // centre vertices of loops that cannot be triangulated with face-safe diagonals
          for (let L = centreStart[cs]; L < centreStart[cs + 1]; L++) {
            if (nVert >= vcap) {
              vcap *= 2;
              pos = growF32(pos, 3 * vcap);
              nrm = growF32(nrm, 3 * vcap);
              col = growU8(col, 3 * vcap);
            }
            const p3 = 3 * nVert;
            let px = 0;
            let py = 0;
            let pz = 0;
            let nx = 0;
            let ny = 0;
            let nz = 0;
            let cr = 0;
            let cg = 0;
            let cb = 0;
            const k0 = centreLoopStart[L];
            const k1 = centreLoopStart[L + 1];
            for (let k = k0; k < k1; k++) {
              const q3 = 3 * cubeVert[centreEdges[k]];
              px += pos[q3];
              py += pos[q3 + 1];
              pz += pos[q3 + 2];
              nx += nrm[q3];
              ny += nrm[q3 + 1];
              nz += nrm[q3 + 2];
              cr += col[q3];
              cg += col[q3 + 1];
              cb += col[q3 + 2];
            }
            const inv = 1 / (k1 - k0);
            pos[p3] = px * inv;
            pos[p3 + 1] = py * inv;
            pos[p3 + 2] = pz * inv;
            const nl = Math.hypot(nx, ny, nz) || 1;
            nrm[p3] = nx / nl;
            nrm[p3 + 1] = ny / nl;
            nrm[p3 + 2] = nz / nl;
            col[p3] = Math.round(cr * inv);
            col[p3 + 1] = Math.round(cg * inv);
            col[p3 + 2] = Math.round(cb * inv);
            cubeVert[12 + L - centreStart[cs]] = nVert++;
          }
          for (let k = triStart[cs]; k < triStart[cs + 1]; k += 3) {
            if (nIdx + 3 > idx.length) idx = growU32(idx, nIdx + 3);
            idx[nIdx] = cubeVert[triEdges[k]];
            idx[nIdx + 1] = cubeVert[triEdges[k + 1]];
            idx[nIdx + 2] = cubeVert[triEdges[k + 2]];
            nIdx += 3;
          }
        }
      }
    }
  }
  return {
    vertexCount: nVert,
    triangleCount: nIdx / 3,
    positions: pos.slice(0, 3 * nVert),
    normals: nrm.slice(0, 3 * nVert),
    colors: col.slice(0, 3 * nVert),
    indices: idx.slice(0, nIdx),
  };
}

// ---------------------------------------------------------------------------
// Mesh utilities

/** Keep only the triangles listed in `keep` (Uint8 per triangle) and drop unreferenced vertices. */
export function compactMesh(mesh: TriangleMesh, keepTriangle: Uint8Array): TriangleMesh {
  const remap = new Int32Array(mesh.vertexCount).fill(-1);
  let nv = 0;
  let nt = 0;
  for (let t = 0; t < mesh.triangleCount; t++) {
    if (!keepTriangle[t]) continue;
    nt++;
    for (let k = 0; k < 3; k++) {
      const v = mesh.indices[3 * t + k];
      if (remap[v] < 0) remap[v] = nv++;
    }
  }
  const positions = new Float32Array(3 * nv);
  const normals = new Float32Array(3 * nv);
  const colors = new Uint8Array(3 * nv);
  for (let v = 0; v < mesh.vertexCount; v++) {
    const r = remap[v];
    if (r < 0) continue;
    for (let k = 0; k < 3; k++) {
      positions[3 * r + k] = mesh.positions[3 * v + k];
      normals[3 * r + k] = mesh.normals[3 * v + k];
      colors[3 * r + k] = mesh.colors[3 * v + k];
    }
  }
  const indices = new Uint32Array(3 * nt);
  let o = 0;
  for (let t = 0; t < mesh.triangleCount; t++) {
    if (!keepTriangle[t]) continue;
    indices[o++] = remap[mesh.indices[3 * t]];
    indices[o++] = remap[mesh.indices[3 * t + 1]];
    indices[o++] = remap[mesh.indices[3 * t + 2]];
  }
  return { vertexCount: nv, triangleCount: nt, positions, normals, colors, indices };
}

/** Connected components (by shared vertices): component id per triangle and triangle counts. */
export function meshComponents(mesh: TriangleMesh): { triComponent: Int32Array; sizes: number[] } {
  const parent = new Int32Array(mesh.vertexCount);
  for (let i = 0; i < parent.length; i++) parent[i] = i;
  const find = (x: number): number => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const union = (a: number, b: number): void => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) {
      if (ra < rb) parent[rb] = ra;
      else parent[ra] = rb;
    }
  };
  const ix = mesh.indices;
  for (let t = 0; t < mesh.triangleCount; t++) {
    union(ix[3 * t], ix[3 * t + 1]);
    union(ix[3 * t], ix[3 * t + 2]);
  }
  const compOf = new Int32Array(mesh.vertexCount).fill(-1);
  const sizes: number[] = [];
  const triComponent = new Int32Array(mesh.triangleCount);
  for (let t = 0; t < mesh.triangleCount; t++) {
    const r = find(ix[3 * t]);
    if (compOf[r] < 0) {
      compOf[r] = sizes.length;
      sizes.push(0);
    }
    const c = compOf[r];
    sizes[c]++;
    triComponent[t] = c;
  }
  return { triComponent, sizes };
}

/**
 * Remove connected components with fewer triangles than `minFraction` × the largest
 * component (and fewer than `minTriangles`, if given, regardless of the fraction).
 * Components whose bounding-box diagonal exceeds `keepDiagonal` (world units) are always
 * kept: in a large scene a tree or a house that is not connected to the main surface can be
 * far below 2 % of it, while noise blobs are small in extent.
 */
export function removeSmallComponents(
  mesh: TriangleMesh,
  minFraction: number,
  minTriangles = 0,
  keepDiagonal = Infinity,
): { mesh: TriangleMesh; removedComponents: number } {
  if (mesh.triangleCount === 0) return { mesh, removedComponents: 0 };
  const { triComponent, sizes } = meshComponents(mesh);
  let largest = 0;
  for (const s of sizes) largest = Math.max(largest, s);
  const thr = Math.max(minFraction * largest, minTriangles);
  const nc = sizes.length;
  const drop = new Uint8Array(nc);
  for (let c = 0; c < nc; c++) drop[c] = sizes[c] < thr ? 1 : 0;
  if (keepDiagonal < Infinity) {
    // bounding boxes of the candidate components
    const bb = new Float64Array(6 * nc);
    for (let c = 0; c < nc; c++) bb.set([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity], 6 * c);
    const P = mesh.positions;
    const ix = mesh.indices;
    for (let t = 0; t < mesh.triangleCount; t++) {
      const c = triComponent[t];
      if (!drop[c]) continue;
      const o = 6 * c;
      for (let k = 0; k < 3; k++) {
        const v = 3 * ix[3 * t + k];
        for (let a = 0; a < 3; a++) {
          const x = P[v + a];
          if (x < bb[o + a]) bb[o + a] = x;
          if (x > bb[o + 3 + a]) bb[o + 3 + a] = x;
        }
      }
    }
    for (let c = 0; c < nc; c++) {
      if (!drop[c]) continue;
      const o = 6 * c;
      if (Math.hypot(bb[o + 3] - bb[o], bb[o + 4] - bb[o + 1], bb[o + 5] - bb[o + 2]) > keepDiagonal) drop[c] = 0;
    }
  }
  let removed = 0;
  for (let c = 0; c < nc; c++) removed += drop[c];
  if (removed === 0) return { mesh, removedComponents: 0 };
  const keep = new Uint8Array(mesh.triangleCount);
  for (let t = 0; t < mesh.triangleCount; t++) keep[t] = drop[triComponent[t]] ? 0 : 1;
  return { mesh: compactMesh(mesh, keep), removedComponents: removed };
}

/** Edge statistics: #unique edges, #boundary edges (1 triangle), #non-manifold edges (> 2), #inconsistently oriented edges. */
export function meshEdgeStats(mesh: TriangleMesh): { edges: number; boundary: number; nonManifold: number; misoriented: number } {
  const map = new Map<number, number>(); // key → count·4 + orientation info
  const V = mesh.vertexCount;
  const dir = new Map<number, number>();
  const ix = mesh.indices;
  for (let t = 0; t < mesh.triangleCount; t++) {
    for (let k = 0; k < 3; k++) {
      const a = ix[3 * t + k];
      const b = ix[3 * t + ((k + 1) % 3)];
      const lo = Math.min(a, b);
      const hi = Math.max(a, b);
      const key = lo * V + hi;
      map.set(key, (map.get(key) ?? 0) + 1);
      dir.set(key, (dir.get(key) ?? 0) + (a < b ? 1 : -1));
    }
  }
  let boundary = 0;
  let nonManifold = 0;
  let misoriented = 0;
  for (const [key, c] of map) {
    if (c === 1) boundary++;
    else if (c > 2) nonManifold++;
    else if (dir.get(key) !== 0) misoriented++;
  }
  return { edges: map.size, boundary, nonManifold, misoriented };
}
