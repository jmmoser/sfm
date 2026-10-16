// Static 3-D k-d tree (bucketed leaves, typed arrays) for exact nearest-neighbour
// queries at any distance in O(log n).

const LEAF = 12;

export class KdTree {
  /** @param {Float32Array|Float64Array} points xyz triplets */
  constructor(points) {
    const n = points.length / 3;
    this.n = n;
    const idx = new Uint32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    const maxNodes = Math.max(1, Math.ceil((4 * n) / LEAF) + 8);
    this.lo = new Uint32Array(maxNodes);
    this.hi = new Uint32Array(maxNodes);
    this.axis = new Int8Array(maxNodes);
    this.split = new Float64Array(maxNodes);
    this.left = new Int32Array(maxNodes);
    this.right = new Int32Array(maxNodes);
    this.count = 0;
    this.points = points;
    this.idx = idx;
    if (n > 0) this.build(0, n);
    // reorder coordinates for locality
    const xs = new Float64Array(n),
      ys = new Float64Array(n),
      zs = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const i = idx[k];
      xs[k] = points[3 * i];
      ys[k] = points[3 * i + 1];
      zs[k] = points[3 * i + 2];
    }
    this.xs = xs;
    this.ys = ys;
    this.zs = zs;
    this.stack = new Int32Array(256);
    this.stackD = new Float64Array(256);
  }

  build(lo, hi) {
    const node = this.count++;
    this.lo[node] = lo;
    this.hi[node] = hi;
    this.left[node] = -1;
    this.right[node] = -1;
    if (hi - lo <= LEAF) return node;
    const P = this.points,
      idx = this.idx;
    let mn0 = Infinity, mn1 = Infinity, mn2 = Infinity, mx0 = -Infinity, mx1 = -Infinity, mx2 = -Infinity;
    for (let k = lo; k < hi; k++) {
      const i = 3 * idx[k];
      const x = P[i], y = P[i + 1], z = P[i + 2];
      if (x < mn0) mn0 = x; if (x > mx0) mx0 = x;
      if (y < mn1) mn1 = y; if (y > mx1) mx1 = y;
      if (z < mn2) mn2 = z; if (z > mx2) mx2 = z;
    }
    const ex = mx0 - mn0, ey = mx1 - mn1, ez = mx2 - mn2;
    const ax = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2;
    const mid = (lo + hi) >> 1;
    this.select(lo, hi - 1, mid, ax);
    this.axis[node] = ax;
    this.split[node] = P[3 * idx[mid] + ax];
    this.left[node] = this.build(lo, mid);
    this.right[node] = this.build(mid, hi);
    return node;
  }

  /** Quickselect: idx[k] gets the element of rank k along axis within [l, r]. */
  select(l, r, k, ax) {
    const P = this.points,
      idx = this.idx;
    while (r > l) {
      const pivot = P[3 * idx[(l + r) >> 1] + ax];
      let i = l,
        j = r;
      while (i <= j) {
        while (P[3 * idx[i] + ax] < pivot) i++;
        while (P[3 * idx[j] + ax] > pivot) j--;
        if (i <= j) {
          const t = idx[i];
          idx[i] = idx[j];
          idx[j] = t;
          i++;
          j--;
        }
      }
      if (k <= j) r = j;
      else if (k >= i) l = i;
      else return;
    }
  }

  /** Nearest point within maxDist → { index (original), dist } or { index: −1, dist: Infinity }. */
  nearest(x, y, z, maxDist = Infinity) {
    if (this.n === 0) return { index: -1, dist: Infinity };
    let best = maxDist * maxDist,
      bk = -1;
    let sp = 0;
    const st = this.stack,
      sd = this.stackD;
    st[sp] = 0;
    sd[sp++] = 0;
    const xs = this.xs, ys = this.ys, zs = this.zs;
    while (sp > 0) {
      sp--;
      if (sd[sp] >= best) continue;
      let node = st[sp];
      // descend to a leaf, pushing far children
      while (this.left[node] >= 0) {
        const ax = this.axis[node];
        const d = (ax === 0 ? x : ax === 1 ? y : z) - this.split[node];
        const near = d < 0 ? this.left[node] : this.right[node];
        const far = d < 0 ? this.right[node] : this.left[node];
        const d2 = d * d;
        if (d2 < best) {
          if (sp >= st.length) this.grow();
          this.stack[sp] = far;
          this.stackD[sp++] = d2;
        }
        node = near;
      }
      const hi = this.hi[node];
      for (let k = this.lo[node]; k < hi; k++) {
        const ex = xs[k] - x, ey = ys[k] - y, ez = zs[k] - z;
        const d2 = ex * ex + ey * ey + ez * ez;
        if (d2 < best) {
          best = d2;
          bk = k;
        }
      }
    }
    if (bk < 0) return { index: -1, dist: Infinity };
    return { index: this.idx[bk], dist: Math.sqrt(best) };
  }

  grow() {
    const s = new Int32Array(this.stack.length * 2);
    s.set(this.stack);
    const d = new Float64Array(this.stackD.length * 2);
    d.set(this.stackD);
    this.stack = s;
    this.stackD = d;
  }
}
