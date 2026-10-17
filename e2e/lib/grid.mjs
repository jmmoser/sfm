// Uniform hash grid over 3-D points for nearest-neighbour / radius queries.
// Cells live in an open-addressing hash table on typed arrays (no Map / boxed keys),
// so millions of queries run in a few seconds.

const EMPTY = 0x7fffffff;

function nextPow2(n) {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

export class HashGrid {
  /**
   * @param {Float32Array|Float64Array} points xyz triplets
   * @param {number} cellSize
   */
  constructor(points, cellSize) {
    this.points = points;
    this.cell = cellSize;
    this.inv = 1 / cellSize;
    const n = points.length / 3;
    this.n = n;
    // cell coordinates per point
    const cx = new Int32Array(n),
      cy = new Int32Array(n),
      cz = new Int32Array(n);
    const inv = this.inv;
    for (let i = 0; i < n; i++) {
      cx[i] = Math.floor(points[3 * i] * inv);
      cy[i] = Math.floor(points[3 * i + 1] * inv);
      cz[i] = Math.floor(points[3 * i + 2] * inv);
    }
    const cap = nextPow2(Math.max(16, 2 * n));
    this.mask = cap - 1;
    this.kx = new Int32Array(cap).fill(EMPTY);
    this.ky = new Int32Array(cap);
    this.kz = new Int32Array(cap);
    this.slotCell = new Int32Array(cap); // dense cell id per slot
    const cellOf = new Int32Array(n);
    const counts = [];
    for (let i = 0; i < n; i++) {
      let h = this.hash(cx[i], cy[i], cz[i]);
      for (;;) {
        if (this.kx[h] === EMPTY) {
          this.kx[h] = cx[i];
          this.ky[h] = cy[i];
          this.kz[h] = cz[i];
          this.slotCell[h] = counts.length;
          counts.push(0);
          break;
        }
        if (this.kx[h] === cx[i] && this.ky[h] === cy[i] && this.kz[h] === cz[i]) break;
        h = (h + 1) & this.mask;
      }
      const c = this.slotCell[h];
      counts[c]++;
      cellOf[i] = c;
    }
    const nc = counts.length;
    const start = new Uint32Array(nc + 1);
    for (let c = 0; c < nc; c++) start[c + 1] = start[c] + counts[c];
    const fill = start.slice(0, nc);
    const order = new Uint32Array(n);
    for (let i = 0; i < n; i++) order[fill[cellOf[i]]++] = i;
    this.start = start;
    this.order = order;
  }

  hash(x, y, z) {
    return (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) & this.mask;
  }

  /** Dense cell id of cell (x, y, z) or −1. */
  cellId(x, y, z) {
    let h = this.hash(x, y, z);
    const kx = this.kx;
    for (;;) {
      const k = kx[h];
      if (k === EMPTY) return -1;
      if (k === x && this.ky[h] === y && this.kz[h] === z) return this.slotCell[h];
      h = (h + 1) & this.mask;
    }
  }

  /**
   * Nearest point within maxDist. Returns { index, dist } (index −1, dist Infinity if none).
   */
  nearest(x, y, z, maxDist) {
    const P = this.points,
      start = this.start,
      order = this.order;
    const ix = Math.floor(x * this.inv),
      iy = Math.floor(y * this.inv),
      iz = Math.floor(z * this.inv);
    let best = Infinity,
      bi = -1;
    const K = Math.max(1, Math.ceil(maxDist * this.inv));
    for (let k = 0; k <= K; k++) {
      for (let dz = -k; dz <= k; dz++)
        for (let dy = -k; dy <= k; dy++) {
          const onFace = Math.abs(dz) === k || Math.abs(dy) === k;
          const step = onFace ? 1 : 2 * k;
          for (let dx = -k; dx <= k; dx += step === 0 ? 1 : step) {
            const c = this.cellId(ix + dx, iy + dy, iz + dz);
            if (c < 0) continue;
            for (let q = start[c]; q < start[c + 1]; q++) {
              const i = order[q];
              const ex = P[3 * i] - x,
                ey = P[3 * i + 1] - y,
                ez = P[3 * i + 2] - z;
              const d2 = ex * ex + ey * ey + ez * ez;
              if (d2 < best) {
                best = d2;
                bi = i;
              }
            }
          }
        }
      // Unvisited points are at least k·cell away.
      if (bi >= 0 && Math.sqrt(best) <= k * this.cell) break;
    }
    const d = Math.sqrt(best);
    if (d > maxDist) return { index: -1, dist: Infinity };
    return { index: bi, dist: d };
  }

  /** True if some point lies within r of (x, y, z). */
  hasWithin(x, y, z, r) {
    const P = this.points,
      start = this.start,
      order = this.order;
    const r2 = r * r;
    const K = Math.ceil(r * this.inv);
    const ix = Math.floor(x * this.inv),
      iy = Math.floor(y * this.inv),
      iz = Math.floor(z * this.inv);
    // own cell first (most likely hit)
    for (let pass = 0; pass < 2; pass++)
      for (let dz = -K; dz <= K; dz++)
        for (let dy = -K; dy <= K; dy++)
          for (let dx = -K; dx <= K; dx++) {
            const own = dx === 0 && dy === 0 && dz === 0;
            if ((pass === 0) !== own) continue;
            const c = this.cellId(ix + dx, iy + dy, iz + dz);
            if (c < 0) continue;
            for (let q = start[c]; q < start[c + 1]; q++) {
              const i = order[q];
              const ex = P[3 * i] - x,
                ey = P[3 * i + 1] - y,
                ez = P[3 * i + 2] - z;
              if (ex * ex + ey * ey + ez * ez <= r2) return true;
            }
          }
    return false;
  }
}
