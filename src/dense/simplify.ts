/**
 * Mesh simplification by quadric-error-metric edge collapses (Garland & Heckbert 1997).
 *
 *  - Vertex quadrics = Σ area-weighted face-plane quadrics; boundary edges add a
 *    perpendicular constraint plane (weight `boundaryWeight` · |e|²) so open borders
 *    (e.g. the edge of a scanned ground patch) stay in place.
 *  - Edges sit in a binary min-heap keyed by collapse cost; entries are validated lazily
 *    with per-vertex version counters (an entry is stale once either endpoint changed).
 *  - Collapse target = argmin of the summed quadric (3×3 solve) when well conditioned,
 *    else the best of the two endpoints and the midpoint.
 *  - A collapse is rejected when it would (a) violate the link condition (create a
 *    non-manifold edge / fold), (b) join two boundary vertices through an interior edge,
 *    or (c) flip or degenerate any surrounding face (normal turns by more than
 *    acos(minNormalCos)).
 *  - Colours and normals are interpolated along the collapsed edge at the projection of
 *    the target position.
 */
import type { TriangleMesh } from '../core/types';

export interface SimplifyOptions {
  /** Stop when the cheapest remaining collapse costs more than this (squared distance units). Default Infinity. */
  maxError?: number;
  /** Weight of the boundary constraint planes (× |edge|²). Default 100 (0 = boundaries move freely). */
  boundaryWeight?: number;
  /** Minimum cosine between a surrounding face's normal before and after a collapse. Default 0.2. */
  minNormalCos?: number;
}

/**
 * 4-ary min-heap of collapse candidates. Heap nodes are (cost, slot) pairs in two flat
 * arrays (4 contiguous children per level → few cache misses); the payload (u, v and
 * their versions at push time) lives in a recycled slot pool.
 */
class EdgeHeap {
  private cost = new Float64Array(1024);
  private slot = new Int32Array(1024);
  private pool = new Int32Array(4 * 1024);
  private free = new Int32Array(1024);
  private nFree = 0;
  private nSlots = 0;
  size = 0;

  private allocSlot(): number {
    if (this.nFree > 0) return this.free[--this.nFree];
    if (4 * (this.nSlots + 1) > this.pool.length) {
      const np = new Int32Array(this.pool.length * 2);
      np.set(this.pool);
      this.pool = np;
    }
    return this.nSlots++;
  }

  push(cost: number, u: number, v: number, vu: number, vv: number): void {
    if (this.size >= this.cost.length) {
      const nc = new Float64Array(this.cost.length * 2);
      nc.set(this.cost);
      this.cost = nc;
      const ns = new Int32Array(this.slot.length * 2);
      ns.set(this.slot);
      this.slot = ns;
    }
    const sl = this.allocSlot();
    const P = this.pool;
    P[4 * sl] = u;
    P[4 * sl + 1] = v;
    P[4 * sl + 2] = vu;
    P[4 * sl + 3] = vv;
    const C = this.cost;
    const S = this.slot;
    let i = this.size++;
    while (i > 0) {
      const p = (i - 1) >> 2;
      if (C[p] <= cost) break;
      C[i] = C[p];
      S[i] = S[p];
      i = p;
    }
    C[i] = cost;
    S[i] = sl;
  }

  /** Pops the minimum into `out` [cost, u, v, vu, vv]; false if empty. */
  pop(out: Float64Array): boolean {
    if (this.size === 0) return false;
    const C = this.cost;
    const S = this.slot;
    const P = this.pool;
    const top = S[0];
    out[0] = C[0];
    out[1] = P[4 * top];
    out[2] = P[4 * top + 1];
    out[3] = P[4 * top + 2];
    out[4] = P[4 * top + 3];
    if (this.nFree >= this.free.length) {
      const nf = new Int32Array(this.free.length * 2);
      nf.set(this.free);
      this.free = nf;
    }
    this.free[this.nFree++] = top;
    const n = --this.size;
    if (n === 0) return true;
    const lc = C[n];
    const ls = S[n];
    let i = 0;
    for (;;) {
      const c0 = 4 * i + 1;
      if (c0 >= n) break;
      let best = c0;
      let bc = C[c0];
      const cEnd = c0 + 4 < n ? c0 + 4 : n;
      for (let c = c0 + 1; c < cEnd; c++) {
        if (C[c] < bc) {
          bc = C[c];
          best = c;
        }
      }
      if (bc >= lc) break;
      C[i] = bc;
      S[i] = S[best];
      i = best;
    }
    C[i] = lc;
    S[i] = ls;
    return true;
  }
}

/** Add w · p pᵀ (p = [a, b, c, d]) to the packed symmetric 4×4 quadric at q[o … o+9]. */
function addPlane(q: Float64Array, o: number, a: number, b: number, c: number, d: number, w: number): void {
  q[o] += w * a * a;
  q[o + 1] += w * a * b;
  q[o + 2] += w * a * c;
  q[o + 3] += w * a * d;
  q[o + 4] += w * b * b;
  q[o + 5] += w * b * c;
  q[o + 6] += w * b * d;
  q[o + 7] += w * c * c;
  q[o + 8] += w * c * d;
  q[o + 9] += w * d * d;
}

/**
 * Simplify `mesh` to (at most) `targetTriangles` triangles. Returns a new mesh; the input
 * is not modified. Fewer collapses happen if `maxError` is reached or no valid collapse
 * remains.
 */
export function simplifyMesh(mesh: TriangleMesh, targetTriangles: number, opts: SimplifyOptions = {}): TriangleMesh {
  const V = mesh.vertexCount;
  const T = mesh.triangleCount;
  const maxError = opts.maxError ?? Infinity;
  const boundaryWeight = opts.boundaryWeight ?? 100;
  const minCos = opts.minNormalCos ?? 0.2;
  const pos = Float64Array.from(mesh.positions);
  const nrm = Float64Array.from(mesh.normals);
  const col = Float32Array.from(mesh.colors);
  const tri = Int32Array.from(mesh.indices);
  const triAlive = new Uint8Array(T);
  let alive = 0;
  for (let t = 0; t < T; t++) {
    const a = tri[3 * t];
    const b = tri[3 * t + 1];
    const c = tri[3 * t + 2];
    if (a !== b && b !== c && a !== c) {
      triAlive[t] = 1;
      alive++;
    }
  }
  if (alive <= targetTriangles) return finalize();

  // --- quadrics
  const Q = new Float64Array(10 * V);
  for (let t = 0; t < T; t++) {
    if (!triAlive[t]) continue;
    const a = 3 * tri[3 * t];
    const b = 3 * tri[3 * t + 1];
    const c = 3 * tri[3 * t + 2];
    const e1x = pos[b] - pos[a];
    const e1y = pos[b + 1] - pos[a + 1];
    const e1z = pos[b + 2] - pos[a + 2];
    const e2x = pos[c] - pos[a];
    const e2y = pos[c + 1] - pos[a + 1];
    const e2z = pos[c + 2] - pos[a + 2];
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (!(l > 0)) continue;
    nx /= l;
    ny /= l;
    nz /= l;
    const d = -(nx * pos[a] + ny * pos[a + 1] + nz * pos[a + 2]);
    const w = 0.5 * l;
    addPlane(Q, (10 * a) / 3, nx, ny, nz, d, w);
    addPlane(Q, (10 * b) / 3, nx, ny, nz, d, w);
    addPlane(Q, (10 * c) / 3, nx, ny, nz, d, w);
  }

  // --- vertex → triangle references (CSR in a growable pool)
  const refStart = new Int32Array(V);
  const refCount = new Int32Array(V);
  let refs = new Int32Array(Math.max(64, 6 * T));
  let refsUsed = 0;
  const rebuildRefs = (): void => {
    refCount.fill(0);
    for (let t = 0; t < T; t++) {
      if (!triAlive[t]) continue;
      refCount[tri[3 * t]]++;
      refCount[tri[3 * t + 1]]++;
      refCount[tri[3 * t + 2]]++;
    }
    let s = 0;
    for (let v = 0; v < V; v++) {
      refStart[v] = s;
      s += refCount[v];
    }
    if (refs.length < 2 * s) refs = new Int32Array(Math.max(2 * s, 64));
    refCount.fill(0);
    for (let t = 0; t < T; t++) {
      if (!triAlive[t]) continue;
      for (let k = 0; k < 3; k++) {
        const v = tri[3 * t + k];
        refs[refStart[v] + refCount[v]++] = t;
      }
    }
    refsUsed = s;
  };
  rebuildRefs();

  const vAlive = new Uint8Array(V).fill(1);
  const ver = new Int32Array(V);
  const boundary = new Uint8Array(V);
  const mark = new Int32Array(V);
  const mark2 = new Int32Array(V);
  const cnt = new Int32Array(V);
  let stamp = 0;
  let nbList = new Int32Array(64);

  /** Distinct neighbours of u (alive triangles) into nbList; returns count. Fills cnt[w] = #shared triangles. */
  const neighbours = (u: number): number => {
    stamp++;
    let n = 0;
    const s = refStart[u];
    const e = s + refCount[u];
    for (let r = s; r < e; r++) {
      const t = refs[r];
      if (!triAlive[t]) continue;
      for (let k = 0; k < 3; k++) {
        const w = tri[3 * t + k];
        if (w === u) continue;
        if (mark[w] !== stamp) {
          mark[w] = stamp;
          cnt[w] = 0;
          if (n >= nbList.length) {
            const nl = new Int32Array(nbList.length * 2);
            nl.set(nbList);
            nbList = nl;
          }
          nbList[n++] = w;
        }
        cnt[w]++;
      }
    }
    return n;
  };

  // --- boundary constraint planes
  for (let u = 0; u < V; u++) {
    const n = neighbours(u);
    for (let k = 0; k < n; k++) {
      const v = nbList[k];
      if (cnt[v] !== 1 || v < u) continue;
      // find the triangle holding edge (u, v)
      let tf = -1;
      for (let r = refStart[u]; r < refStart[u] + refCount[u]; r++) {
        const t = refs[r];
        if (!triAlive[t]) continue;
        if (tri[3 * t] === v || tri[3 * t + 1] === v || tri[3 * t + 2] === v) {
          tf = t;
          break;
        }
      }
      if (tf < 0) continue;
      boundary[u] = 1;
      boundary[v] = 1;
      if (boundaryWeight <= 0) continue;
      const a = 3 * tri[3 * tf];
      const b = 3 * tri[3 * tf + 1];
      const c = 3 * tri[3 * tf + 2];
      const e1x = pos[b] - pos[a];
      const e1y = pos[b + 1] - pos[a + 1];
      const e1z = pos[b + 2] - pos[a + 2];
      const e2x = pos[c] - pos[a];
      const e2y = pos[c + 1] - pos[a + 1];
      const e2z = pos[c + 2] - pos[a + 2];
      const fnx = e1y * e2z - e1z * e2y;
      const fny = e1z * e2x - e1x * e2z;
      const fnz = e1x * e2y - e1y * e2x;
      const ex = pos[3 * v] - pos[3 * u];
      const ey = pos[3 * v + 1] - pos[3 * u + 1];
      const ez = pos[3 * v + 2] - pos[3 * u + 2];
      let mx = ey * fnz - ez * fny;
      let my = ez * fnx - ex * fnz;
      let mz = ex * fny - ey * fnx;
      const ml = Math.sqrt(mx * mx + my * my + mz * mz);
      if (!(ml > 0)) continue;
      mx /= ml;
      my /= ml;
      mz /= ml;
      const d = -(mx * pos[3 * u] + my * pos[3 * u + 1] + mz * pos[3 * u + 2]);
      const w = boundaryWeight * (ex * ex + ey * ey + ez * ez);
      addPlane(Q, 10 * u, mx, my, mz, d, w);
      addPlane(Q, 10 * v, mx, my, mz, d, w);
    }
  }

  // --- collapse cost
  const target = new Float64Array(3);
  const q = new Float64Array(10);
  const evalQ = (x: number, y: number, z: number): number =>
    q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x + q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y + q[7] * z * z + 2 * q[8] * z + q[9];
  const collapseCost = (u: number, v: number): number => {
    for (let k = 0; k < 10; k++) q[k] = Q[10 * u + k] + Q[10 * v + k];
    const a11 = q[0];
    const a12 = q[1];
    const a13 = q[2];
    const a22 = q[4];
    const a23 = q[5];
    const a33 = q[7];
    const b1 = q[3];
    const b2 = q[6];
    const b3 = q[8];
    const c11 = a22 * a33 - a23 * a23;
    const c12 = a13 * a23 - a12 * a33;
    const c13 = a12 * a23 - a13 * a22;
    const det = a11 * c11 + a12 * c12 + a13 * c13;
    const tr = a11 + a22 + a33;
    const ux = pos[3 * u];
    const uy = pos[3 * u + 1];
    const uz = pos[3 * u + 2];
    const vx = pos[3 * v];
    const vy = pos[3 * v + 1];
    const vz = pos[3 * v + 2];
    const el2 = (vx - ux) ** 2 + (vy - uy) ** 2 + (vz - uz) ** 2;
    if (Math.abs(det) > 1e-9 * tr * tr * tr && tr > 0) {
      const c22 = a11 * a33 - a13 * a13;
      const c23 = a12 * a13 - a11 * a23;
      const c33 = a11 * a22 - a12 * a12;
      const id = 1 / det;
      const x = -(c11 * b1 + c12 * b2 + c13 * b3) * id;
      const y = -(c12 * b1 + c22 * b2 + c23 * b3) * id;
      const z = -(c13 * b1 + c23 * b2 + c33 * b3) * id;
      const mx = 0.5 * (ux + vx);
      const my = 0.5 * (uy + vy);
      const mz = 0.5 * (uz + vz);
      if (Number.isFinite(x) && (x - mx) ** 2 + (y - my) ** 2 + (z - mz) ** 2 <= 4 * el2) {
        target[0] = x;
        target[1] = y;
        target[2] = z;
        return Math.max(0, evalQ(x, y, z));
      }
    }
    // fallback: best of endpoints and midpoint
    let best = evalQ(ux, uy, uz);
    target[0] = ux;
    target[1] = uy;
    target[2] = uz;
    const cv = evalQ(vx, vy, vz);
    if (cv < best) {
      best = cv;
      target[0] = vx;
      target[1] = vy;
      target[2] = vz;
    }
    const mx = 0.5 * (ux + vx);
    const my = 0.5 * (uy + vy);
    const mz = 0.5 * (uz + vz);
    const cm = evalQ(mx, my, mz);
    if (cm < best) {
      best = cm;
      target[0] = mx;
      target[1] = my;
      target[2] = mz;
    }
    return Math.max(0, best);
  };

  // --- initial heap
  const heap = new EdgeHeap();
  for (let u = 0; u < V; u++) {
    const n = neighbours(u);
    for (let k = 0; k < n; k++) {
      const v = nbList[k];
      if (v < u) continue;
      heap.push(collapseCost(u, v), u, v, 0, 0);
    }
  }

  /** Normal-flip / degeneracy test of the triangles around `a` (not containing `b`) when `a` moves to target. */
  const facesOk = (a: number, b: number): boolean => {
    const x = target[0];
    const y = target[1];
    const z = target[2];
    for (let r = refStart[a]; r < refStart[a] + refCount[a]; r++) {
      const t = refs[r];
      if (!triAlive[t]) continue;
      const i0 = tri[3 * t];
      const i1 = tri[3 * t + 1];
      const i2 = tri[3 * t + 2];
      if (i0 === b || i1 === b || i2 === b) continue;
      // rotate so that `a` is first
      let p = i1;
      let r2 = i2;
      if (i1 === a) {
        p = i2;
        r2 = i0;
      } else if (i2 === a) {
        p = i0;
        r2 = i1;
      }
      const ax = pos[3 * a];
      const ay = pos[3 * a + 1];
      const az = pos[3 * a + 2];
      const px = pos[3 * p];
      const py = pos[3 * p + 1];
      const pz = pos[3 * p + 2];
      const rx = pos[3 * r2];
      const ry = pos[3 * r2 + 1];
      const rz = pos[3 * r2 + 2];
      // old normal
      let e1x = px - ax;
      let e1y = py - ay;
      let e1z = pz - az;
      let e2x = rx - ax;
      let e2y = ry - ay;
      let e2z = rz - az;
      const ox = e1y * e2z - e1z * e2y;
      const oy = e1z * e2x - e1x * e2z;
      const oz = e1x * e2y - e1y * e2x;
      // new normal
      e1x = px - x;
      e1y = py - y;
      e1z = pz - z;
      e2x = rx - x;
      e2y = ry - y;
      e2z = rz - z;
      const nx = e1y * e2z - e1z * e2y;
      const ny = e1z * e2x - e1x * e2z;
      const nz = e1x * e2y - e1y * e2x;
      const lo = Math.sqrt(ox * ox + oy * oy + oz * oz);
      const ln = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (!(ln > 1e-12 * (e1x * e1x + e1y * e1y + e1z * e1z + e2x * e2x + e2y * e2y + e2z * e2z))) return false;
      if (lo > 0 && (ox * nx + oy * ny + oz * nz) / (lo * ln) < minCos) return false;
    }
    return true;
  };

  const canCollapse = (u: number, v: number): boolean => {
    // link condition
    const nu = neighbours(u); // marks u's neighbours with `stamp`, cnt = shared-triangle counts
    void nu;
    const su = stamp;
    if (mark[v] !== su) return false; // edge vanished
    const shared = cnt[v];
    let common = 0;
    for (let r = refStart[v]; r < refStart[v] + refCount[v]; r++) {
      const t = refs[r];
      if (!triAlive[t]) continue;
      for (let k = 0; k < 3; k++) {
        const w = tri[3 * t + k];
        if (w === u || w === v) continue;
        if (mark[w] === su && mark2[w] !== su) {
          mark2[w] = su;
          common++;
        }
      }
    }
    if (common !== shared) return false;
    if (shared >= 2 && boundary[u] && boundary[v]) return false;
    return facesOk(u, v) && facesOk(v, u);
  };

  const popped = new Float64Array(5);
  while (alive > targetTriangles && heap.pop(popped)) {
    const cost = popped[0];
    if (cost > maxError) break;
    const u = popped[1];
    const v = popped[2];
    if (!vAlive[u] || !vAlive[v] || ver[u] !== popped[3] || ver[v] !== popped[4]) continue;
    collapseCost(u, v); // sets target
    if (!canCollapse(u, v)) continue;

    // attribute interpolation parameter
    const ex = pos[3 * v] - pos[3 * u];
    const ey = pos[3 * v + 1] - pos[3 * u + 1];
    const ez = pos[3 * v + 2] - pos[3 * u + 2];
    const el2 = ex * ex + ey * ey + ez * ez;
    let s = el2 > 0 ? ((target[0] - pos[3 * u]) * ex + (target[1] - pos[3 * u + 1]) * ey + (target[2] - pos[3 * u + 2]) * ez) / el2 : 0.5;
    s = s < 0 ? 0 : s > 1 ? 1 : s;
    for (let k = 0; k < 3; k++) {
      col[3 * u + k] += s * (col[3 * v + k] - col[3 * u + k]);
      nrm[3 * u + k] += s * (nrm[3 * v + k] - nrm[3 * u + k]);
    }
    const nl = Math.sqrt(nrm[3 * u] * nrm[3 * u] + nrm[3 * u + 1] * nrm[3 * u + 1] + nrm[3 * u + 2] * nrm[3 * u + 2]);
    if (nl > 0) {
      nrm[3 * u] /= nl;
      nrm[3 * u + 1] /= nl;
      nrm[3 * u + 2] /= nl;
    }
    pos[3 * u] = target[0];
    pos[3 * u + 1] = target[1];
    pos[3 * u + 2] = target[2];
    for (let k = 0; k < 10; k++) Q[10 * u + k] += Q[10 * v + k];
    boundary[u] |= boundary[v];

    // retarget v's triangles
    for (let r = refStart[v]; r < refStart[v] + refCount[v]; r++) {
      const t = refs[r];
      if (!triAlive[t]) continue;
      const b3 = 3 * t;
      if (tri[b3] === u || tri[b3 + 1] === u || tri[b3 + 2] === u) {
        triAlive[t] = 0;
        alive--;
        continue;
      }
      if (tri[b3] === v) tri[b3] = u;
      else if (tri[b3 + 1] === v) tri[b3 + 1] = u;
      else tri[b3 + 2] = u;
    }
    // new reference list of u = alive triangles of u and v
    const need = refCount[u] + refCount[v];
    if (refsUsed + need > refs.length) {
      vAlive[v] = 0;
      rebuildRefs();
      if (refsUsed + need > refs.length) {
        const nr = new Int32Array(Math.max(refs.length * 2, refsUsed + need + 64));
        nr.set(refs.subarray(0, refsUsed));
        refs = nr;
      }
    }
    const start = refsUsed;
    let n = 0;
    for (const src of [u, v]) {
      for (let r = refStart[src]; r < refStart[src] + refCount[src]; r++) {
        const t = refs[r];
        if (triAlive[t]) refs[start + n++] = t;
      }
    }
    // dedupe (a triangle can only be in both lists if it contained u and v — those are dead)
    refStart[u] = start;
    refCount[u] = n;
    refsUsed += n;
    refCount[v] = 0;
    vAlive[v] = 0;
    ver[u]++;
    ver[v]++;

    // re-queue edges around u
    const nn = neighbours(u);
    for (let k = 0; k < nn; k++) {
      const w = nbList[k];
      const c = collapseCost(u, w);
      if (u < w) heap.push(c, u, w, ver[u], ver[w]);
      else heap.push(c, w, u, ver[w], ver[u]);
    }
  }
  return finalize();

  function finalize(): TriangleMesh {
    // keep the original vertex order (stable output when nothing / little changed)
    const remap = new Int32Array(V).fill(-1);
    let nt = 0;
    for (let t = 0; t < T; t++) {
      if (!triAlive[t]) continue;
      nt++;
      remap[tri[3 * t]] = 0;
      remap[tri[3 * t + 1]] = 0;
      remap[tri[3 * t + 2]] = 0;
    }
    let nv = 0;
    for (let v = 0; v < V; v++) if (remap[v] === 0) remap[v] = nv++;
    const positions = new Float32Array(3 * nv);
    const normals = new Float32Array(3 * nv);
    const colors = new Uint8Array(3 * nv);
    for (let v = 0; v < V; v++) {
      const r = remap[v];
      if (r < 0) continue;
      let nx = nrm[3 * v];
      let ny = nrm[3 * v + 1];
      let nz = nrm[3 * v + 2];
      const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (l > 0) {
        nx /= l;
        ny /= l;
        nz /= l;
      }
      positions[3 * r] = pos[3 * v];
      positions[3 * r + 1] = pos[3 * v + 1];
      positions[3 * r + 2] = pos[3 * v + 2];
      normals[3 * r] = nx;
      normals[3 * r + 1] = ny;
      normals[3 * r + 2] = nz;
      for (let k = 0; k < 3; k++) {
        const c = Math.round(col[3 * v + k]);
        colors[3 * r + k] = c < 0 ? 0 : c > 255 ? 255 : c;
      }
    }
    const indices = new Uint32Array(3 * nt);
    let o = 0;
    for (let t = 0; t < T; t++) {
      if (!triAlive[t]) continue;
      indices[o++] = remap[tri[3 * t]];
      indices[o++] = remap[tri[3 * t + 1]];
      indices[o++] = remap[tri[3 * t + 2]];
    }
    return { vertexCount: nv, triangleCount: nt, positions, normals, colors, indices };
  }
}
