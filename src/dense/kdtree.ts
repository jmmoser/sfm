/**
 * Static 3-D k-d tree over a Float32 point array (implicit, balanced: median splits on
 * the axis of largest extent, leaves of ≤ 8 points). k-nearest-neighbour queries are
 * exact and independent of the point density, so they work for clouds that mix
 * millimetre-spaced close-range points with decimetre-spaced far points (a uniform grid
 * degenerates there).
 *
 * The points are copied in tree order (`tp`), so subtrees and leaves are contiguous in
 * memory; `perm[i]` maps tree position i back to the input index. Querying in tree order
 * (`knnAt`) is the cache-friendly way to process every point.
 */
export class KdTree3 {
  /** Input point index of each tree position. */
  readonly perm: Uint32Array;
  /** Points in tree order: tp[3i … 3i+2] = points[3·perm[i] …]. */
  readonly tp: Float32Array;
  /** Split axis per node, stored at the node's median slot. */
  private readonly axis: Uint8Array;
  private readonly leafSize = 8;
  // query scratch: max-heap of the k best (squared distance, tree position)
  private hd = new Float64Array(16);
  private hi = new Int32Array(16);
  private hn = 0;
  private hk = 0;
  private stack = new Int32Array(256);
  private stackBound = new Float64Array(128);

  constructor(
    points: Float32Array,
    readonly count: number = Math.floor(points.length / 3),
  ) {
    const n = count;
    this.perm = new Uint32Array(n);
    for (let i = 0; i < n; i++) this.perm[i] = i;
    this.tp = points.slice(0, 3 * n);
    this.axis = new Uint8Array(Math.max(1, n));
    const st: number[] = [0, n];
    while (st.length) {
      const hiR = st.pop()!;
      const loR = st.pop()!;
      if (hiR - loR <= this.leafSize) continue;
      let mnx = Infinity;
      let mny = Infinity;
      let mnz = Infinity;
      let mxx = -Infinity;
      let mxy = -Infinity;
      let mxz = -Infinity;
      const T = this.tp;
      for (let i = loR; i < hiR; i++) {
        const x = T[3 * i];
        const y = T[3 * i + 1];
        const z = T[3 * i + 2];
        if (x < mnx) mnx = x;
        if (x > mxx) mxx = x;
        if (y < mny) mny = y;
        if (y > mxy) mxy = y;
        if (z < mnz) mnz = z;
        if (z > mxz) mxz = z;
      }
      const ex = mxx - mnx;
      const ey = mxy - mny;
      const ez = mxz - mnz;
      const ax = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2;
      const mid = (loR + hiR) >> 1;
      this.select(loR, hiR - 1, mid, ax);
      this.axis[mid] = ax;
      st.push(loR, mid, mid + 1, hiR);
    }
  }

  private swap(i: number, j: number): void {
    const T = this.tp;
    const perm = this.perm;
    const t = perm[i];
    perm[i] = perm[j];
    perm[j] = t;
    const a = 3 * i;
    const b = 3 * j;
    let v = T[a];
    T[a] = T[b];
    T[b] = v;
    v = T[a + 1];
    T[a + 1] = T[b + 1];
    T[b + 1] = v;
    v = T[a + 2];
    T[a + 2] = T[b + 2];
    T[b + 2] = v;
  }

  /** Quickselect (Hoare partition, median-of-three) so that position k holds the median along `ax`. */
  private select(lo: number, hi: number, k: number, ax: number): void {
    const T = this.tp;
    while (hi > lo) {
      const m = (lo + hi) >> 1;
      const a = T[3 * lo + ax];
      const b = T[3 * m + ax];
      const c = T[3 * hi + ax];
      const pivot = a < b ? (b < c ? b : a < c ? c : a) : a < c ? a : b < c ? c : b;
      let i = lo;
      let j = hi;
      while (i <= j) {
        while (T[3 * i + ax] < pivot) i++;
        while (T[3 * j + ax] > pivot) j--;
        if (i <= j) {
          if (i !== j) this.swap(i, j);
          i++;
          j--;
        }
      }
      if (k <= j) hi = j;
      else if (k >= i) lo = i;
      else return;
    }
  }

  private heapPush(d: number, idx: number): void {
    const hd = this.hd;
    const hi = this.hi;
    if (this.hn < this.hk) {
      let i = this.hn++;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (hd[p] >= d) break;
        hd[i] = hd[p];
        hi[i] = hi[p];
        i = p;
      }
      hd[i] = d;
      hi[i] = idx;
      return;
    }
    if (d >= hd[0]) return;
    const n = this.hn;
    let i = 0;
    for (;;) {
      let c = 2 * i + 1;
      if (c >= n) break;
      if (c + 1 < n && hd[c + 1] > hd[c]) c++;
      if (hd[c] <= d) break;
      hd[i] = hd[c];
      hi[i] = hi[c];
      i = c;
    }
    hd[i] = d;
    hi[i] = idx;
  }

  /** Core search; results (tree positions) stay in the scratch heap. */
  private search(x: number, y: number, z: number, k: number, excludePos: number): number {
    if (this.hd.length < k) {
      this.hd = new Float64Array(k);
      this.hi = new Int32Array(k);
    }
    this.hn = 0;
    this.hk = k;
    const P = this.tp;
    const axis = this.axis;
    let stack = this.stack;
    let sb = this.stackBound;
    let sp = 0;
    // stack frames: (lo, hi) + lower bound of the squared distance to the region
    stack[0] = 0;
    stack[1] = this.count;
    sb[sp++] = 0;
    while (sp > 0) {
      sp--;
      const loR = stack[2 * sp];
      const hiR = stack[2 * sp + 1];
      if (this.hn >= k && sb[sp] >= this.hd[0]) continue;
      if (hiR - loR <= this.leafSize) {
        for (let i = loR; i < hiR; i++) {
          if (i === excludePos) continue;
          const p = 3 * i;
          const dx = P[p] - x;
          const dy = P[p + 1] - y;
          const dz = P[p + 2] - z;
          this.heapPush(dx * dx + dy * dy + dz * dz, i);
        }
        continue;
      }
      const mid = (loR + hiR) >> 1;
      const ax = axis[mid];
      const p = 3 * mid;
      const diff = (ax === 0 ? x : ax === 1 ? y : z) - P[p + ax];
      if (mid !== excludePos) {
        const dx = P[p] - x;
        const dy = P[p + 1] - y;
        const dz = P[p + 2] - z;
        this.heapPush(dx * dx + dy * dy + dz * dz, mid);
      }
      if (sp + 2 > sb.length) {
        const ns = new Int32Array(stack.length * 2);
        ns.set(stack);
        stack = this.stack = ns;
        const nb = new Float64Array(sb.length * 2);
        nb.set(sb);
        sb = this.stackBound = nb;
      }
      // far side first (popped later), near side last (popped next)
      const d2 = diff * diff;
      if (diff < 0) {
        stack[2 * sp] = mid + 1;
        stack[2 * sp + 1] = hiR;
        sb[sp++] = d2;
        stack[2 * sp] = loR;
        stack[2 * sp + 1] = mid;
        sb[sp++] = 0;
      } else {
        stack[2 * sp] = loR;
        stack[2 * sp + 1] = mid;
        sb[sp++] = d2;
        stack[2 * sp] = mid + 1;
        stack[2 * sp + 1] = hiR;
        sb[sp++] = 0;
      }
    }
    return this.hn;
  }

  /**
   * k nearest neighbours of (x, y, z): writes squared distances and INPUT indices (unsorted)
   * to outD / outI and returns how many were found (≤ k). `exclude` (an input index) is
   * skipped (O(n) lookup of its tree position — use `knnAt` for bulk self-queries).
   */
  knn(x: number, y: number, z: number, k: number, outD: Float64Array, outI: Int32Array, exclude = -1): number {
    let excludePos = -1;
    if (exclude >= 0) {
      const perm = this.perm;
      for (let i = 0; i < this.count; i++) {
        if (perm[i] === exclude) {
          excludePos = i;
          break;
        }
      }
    }
    const n = this.search(x, y, z, k, excludePos);
    for (let i = 0; i < n; i++) {
      outD[i] = this.hd[i];
      outI[i] = this.perm[this.hi[i]];
    }
    return n;
  }

  /** k nearest neighbours of the point at tree position `pos` (itself excluded); indices are TREE positions. */
  knnAt(pos: number, k: number, outD: Float64Array, outPos: Int32Array): number {
    const T = this.tp;
    const n = this.search(T[3 * pos], T[3 * pos + 1], T[3 * pos + 2], k, pos);
    for (let i = 0; i < n; i++) {
      outD[i] = this.hd[i];
      outPos[i] = this.hi[i];
    }
    return n;
  }
}
