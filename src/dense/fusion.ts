/**
 * Consistency-checked depth-map fusion (ARCHITECTURE §4.6), COLMAP StereoFusion style.
 *
 * Reference views are visited in the given order; every valid pixel that has not yet
 * been fused is back-projected and projected into the reference's neighbour views. A
 * neighbour pixel q (nearest to the projection (u, v)) supports the point when
 *  - it is valid and not yet fused,
 *  - its depth agrees: |z_proj − d_q| / d_q < maxDepthError, where d_q is the depth of
 *    q's plane (from its normal) intersected with the ray through (u, v) (removes the
 *    up-to-half-pixel lookup error on slanted surfaces),
 *  - the neighbour's surface point reprojects into the reference within
 *    maxReprojErrorPx of the reference pixel (forward–backward check),
 *  - the normals agree within maxNormalErrorDeg.
 * With ≥ minViews views (reference included) a point is emitted: mean position,
 * normalised mean normal, per-channel median colour, radius ≈ splat radius of the
 * reference pixel footprint, confidence = numViews / (1 + numNeighbours). The reference
 * pixel and every supporting pixel are then marked as fused (bit masks per view), so each
 * surface patch yields one point (no duplicates across references).
 *
 * Depth maps stream through an LRU cache (only the reference and its neighbours must be
 * resident). Optionally the consistency-filtered depth maps (only fused pixels kept) are
 * written to `filteredStore`, e.g. as clean input for `tsdfMesh`.
 */
import type { CancelToken, PointCloud, ProgressCallback } from '../core/types';
import { DepthMapCache, Yielder, growF32, growU8, relativeTransform, scaleCamera } from './dense-util';
import type { DenseView, DepthMap, DepthMapStore } from './types';

export interface FusionOptions {
  /** Minimum number of consistent views per point, reference included. Default 3. */
  minViews: number;
  /** Maximum relative depth difference. Default 0.01. */
  maxDepthError: number;
  /** Maximum forward–backward reprojection error in the reference (pixels of the depth map). Default 2. */
  maxReprojErrorPx: number;
  /** Maximum angle between normals (degrees). Default 10. */
  maxNormalErrorDeg: number;
  /** Pixels with a photometric cost above this are ignored. Default 2 (all valid pixels). */
  maxCost: number;
  /** Reference pixel stride (1 = every pixel). Default 1. */
  step: number;
  /**
   * Use at most this many neighbours per reference (lists are taken best-first, as produced
   * by `neighboursFromSelection`). Bounds memory: the reference and all its neighbours must be
   * resident (20 B/px per depth map). Default 10.
   */
  maxNeighbours: number;
  /** Depth-map LRU cache size (maps). 0 = auto (maxNeighbours + 2). Default 0. */
  cacheSize: number;
  /** Intersect the neighbour pixel's plane with the projection ray for the depth test. Default true. */
  usePlaneDepth: boolean;
  /** Splat radius = radiusScale · depth / f / √max(cos θ, 0.2). Default 0.7. */
  radiusScale: number;
  /** Stop after this many points. Default Infinity. */
  maxPoints: number;
  /** If set, consistency-filtered depth maps (only fused pixels) are written here. */
  filteredStore?: DepthMapStore;
  /** Receives statistics when fusion finishes. */
  onStats?: (s: FusionStats) => void;
}

export const DEFAULT_FUSION_OPTIONS: Readonly<Omit<FusionOptions, 'filteredStore' | 'onStats'>> = Object.freeze({
  minViews: 3,
  maxDepthError: 0.01,
  maxReprojErrorPx: 2,
  maxNormalErrorDeg: 10,
  maxCost: 2,
  step: 1,
  maxNeighbours: 10,
  cacheSize: 0,
  usePlaneDepth: true,
  radiusScale: 0.7,
  maxPoints: Infinity,
});

export interface FusionStats {
  points: number;
  referencesProcessed: number;
  /** Valid, unfused reference pixels examined. */
  candidates: number;
  /** Mean number of views per fused point. */
  meanViews: number;
  cacheHits: number;
  cacheMisses: number;
  timeMs: number;
}

/** Neighbour parameters, flat: R (9), t (3), f, cx, cy, w, h — ref-camera frame → neighbour. */
const NB_STRIDE = 17;

/** Per-reference working set of the fusion kernel (flat arrays: monomorphic, JIT-friendly). */
interface RefContext {
  W: number;
  H: number;
  fR: number;
  cxR: number;
  cyR: number;
  depthR: Float32Array;
  normalR: Float32Array;
  costR: Float32Array | null;
  maskR: Uint32Array;
  imgR: Uint8Array | Uint8ClampedArray;
  imgRW: number;
  imgRH: number;
  Rr: Float64Array;
  tr: Float64Array;
  nNb: number;
  nbP: Float64Array;
  nbDepth: Float32Array[];
  nbNormal: Float32Array[];
  nbCost: (Float32Array | null)[];
  nbMask: Uint32Array[];
  nbImg: (Uint8Array | Uint8ClampedArray)[];
  /** image width, height, depth→image scale x, y per neighbour. */
  nbImgP: Float64Array;
}

/** Growable output buffers. */
class PointSink {
  cap = 1 << 16;
  count = 0;
  viewSum = 0;
  pos = new Float32Array(3 * this.cap);
  nrm = new Float32Array(3 * this.cap);
  col = new Uint8Array(3 * this.cap);
  rad = new Float32Array(this.cap);
  conf = new Float32Array(this.cap);

  reserve(): void {
    if (this.count < this.cap) return;
    this.cap *= 2;
    this.pos = growF32(this.pos, 3 * this.cap);
    this.nrm = growF32(this.nrm, 3 * this.cap);
    this.col = growU8(this.col, 3 * this.cap);
    this.rad = growF32(this.rad, this.cap);
    this.conf = growF32(this.conf, this.cap);
  }
}

interface KernelParams {
  minViews: number;
  maxRel: number;
  maxReprojSq: number;
  cosNormal: number;
  maxCost: number;
  step: number;
  usePlane: boolean;
  radiusScale: number;
  maxPoints: number;
}

/** Scratch shared by kernel calls. */
class KernelScratch {
  sr = new Uint8Array(32);
  sg = new Uint8Array(32);
  sb = new Uint8Array(32);
  accJ = new Int32Array(32);
  accQ = new Int32Array(32);
  ensure(n: number): void {
    if (this.sr.length >= n + 1) return;
    this.sr = new Uint8Array(n + 8);
    this.sg = new Uint8Array(n + 8);
    this.sb = new Uint8Array(n + 8);
    this.accJ = new Int32Array(n + 8);
    this.accQ = new Int32Array(n + 8);
  }
}

function median8(a: Uint8Array, n: number): number {
  for (let i = 1; i < n; i++) {
    const v = a[i];
    let j = i - 1;
    while (j >= 0 && a[j] > v) {
      a[j + 1] = a[j];
      j--;
    }
    a[j + 1] = v;
  }
  return n & 1 ? a[n >> 1] : (a[(n >> 1) - 1] + a[n >> 1] + 1) >> 1;
}

/**
 * Fuse reference rows [y0, y1). Returns the number of candidate (valid, unfused) reference
 * pixels examined. Stops early when `maxPoints` is reached.
 */
function fuseRows(c: RefContext, out: PointSink, sc: KernelScratch, p: KernelParams, y0: number, y1: number): number {
  const W = c.W;
  const fR = c.fR;
  const cxR = c.cxR;
  const cyR = c.cyR;
  const depthR = c.depthR;
  const normalR = c.normalR;
  const costR = c.costR;
  const maskR = c.maskR;
  const nNb = c.nNb;
  const P = c.nbP;
  const IP = c.nbImgP;
  const sr = sc.sr;
  const sg = sc.sg;
  const sb = sc.sb;
  const accJ = sc.accJ;
  const accQ = sc.accQ;
  const maxRel = p.maxRel;
  const maxReprojSq = p.maxReprojSq;
  const cosNormal = p.cosNormal;
  const maxCost = p.maxCost;
  const minViews = p.minViews;
  const usePlane = p.usePlane;
  const step = p.step;
  const rsx = c.imgRW / W;
  const rsy = c.imgRH / c.H;
  const confDen = 1 / (1 + nNb);
  let candidates = 0;

  for (let y = y0; y < y1; y += step) {
    const yr = (y - cyR) / fR;
    for (let x = 0; x < W; x += step) {
      const i = y * W + x;
      const d = depthR[i];
      if (!(d > 0)) continue;
      if ((maskR[i >>> 5] >>> (i & 31)) & 1) continue;
      if (costR !== null && costR[i] > maxCost) continue;
      candidates++;
      const X = d * ((x - cxR) / fR);
      const Y = d * yr;
      const Z = d;
      const rnx = normalR[3 * i];
      const rny = normalR[3 * i + 1];
      const rnz = normalR[3 * i + 2];
      const refNormalOk = rnx !== 0 || rny !== 0 || rnz !== 0;
      let sX = X;
      let sY = Y;
      let sZ = Z;
      let snx = rnx;
      let sny = rny;
      let snz = rnz;
      let nv = 1;
      let nAcc = 0;
      for (let k = 0; k < nNb; k++) {
        const o = k * NB_STRIDE;
        const zj = P[o + 6] * X + P[o + 7] * Y + P[o + 8] * Z + P[o + 11];
        if (!(zj > 0)) continue;
        const rjx = (P[o] * X + P[o + 1] * Y + P[o + 2] * Z + P[o + 9]) / zj;
        const rjy = (P[o + 3] * X + P[o + 4] * Y + P[o + 5] * Z + P[o + 10]) / zj;
        const fj = P[o + 12];
        const cxj = P[o + 13];
        const cyj = P[o + 14];
        const wj = P[o + 15];
        const u = fj * rjx + cxj;
        const v = fj * rjy + cyj;
        if (!(u >= -0.5 && v >= -0.5 && u < wj - 0.5 && v < P[o + 16] - 0.5)) continue;
        const qx = (u + 0.5) | 0;
        const qy = (v + 0.5) | 0;
        const q = qy * wj + qx;
        const mask = c.nbMask[k];
        if ((mask[q >>> 5] >>> (q & 31)) & 1) continue;
        const dj = c.nbDepth[k][q];
        if (!(dj > 0)) continue;
        const cj = c.nbCost[k];
        if (cj !== null && cj[q] > maxCost) continue;
        const nrmj = c.nbNormal[k];
        const njx = nrmj[3 * q];
        const njy = nrmj[3 * q + 1];
        const njz = nrmj[3 * q + 2];
        const nbNormalOk = njx !== 0 || njy !== 0 || njz !== 0;
        let dp = dj;
        if (usePlane && nbNormalOk) {
          const nrq = njx * ((qx - cxj) / fj) + njy * ((qy - cyj) / fj) + njz;
          const nr = njx * rjx + njy * rjy + njz;
          // only when the plane is not seen (nearly) edge-on
          if (nrq < 0 && nr * nr > 0.01 * (rjx * rjx + rjy * rjy + 1) && nr < 0) {
            const zp = (dj * nrq) / nr;
            if (zp > 0.5 * dj && zp < 2 * dj) dp = zp;
          }
        }
        if (Math.abs(zj - dp) > maxRel * dp) continue;
        // neighbour's surface point on the projection ray, back in the reference frame
        const pjx = dp * rjx - P[o + 9];
        const pjy = dp * rjy - P[o + 10];
        const pjz = dp - P[o + 11];
        const bx = P[o] * pjx + P[o + 3] * pjy + P[o + 6] * pjz;
        const by = P[o + 1] * pjx + P[o + 4] * pjy + P[o + 7] * pjz;
        const bz = P[o + 2] * pjx + P[o + 5] * pjy + P[o + 8] * pjz;
        if (!(bz > 0)) continue;
        const eu = (fR * bx) / bz + cxR - x;
        const ev = (fR * by) / bz + cyR - y;
        if (eu * eu + ev * ev > maxReprojSq) continue;
        let mx = 0;
        let my = 0;
        let mz = 0;
        if (nbNormalOk) {
          mx = P[o] * njx + P[o + 3] * njy + P[o + 6] * njz;
          my = P[o + 1] * njx + P[o + 4] * njy + P[o + 7] * njz;
          mz = P[o + 2] * njx + P[o + 5] * njy + P[o + 8] * njz;
          if (refNormalOk && rnx * mx + rny * my + rnz * mz < cosNormal) continue;
        }
        // accepted
        sX += bx;
        sY += by;
        sZ += bz;
        snx += mx;
        sny += my;
        snz += mz;
        const io = 4 * k;
        const iw = IP[io];
        let ix = ((u + 0.5) * IP[io + 2]) | 0;
        let iy = ((v + 0.5) * IP[io + 3]) | 0;
        if (ix >= iw) ix = iw - 1;
        if (iy >= IP[io + 1]) iy = IP[io + 1] - 1;
        const img = c.nbImg[k];
        const j4 = 4 * (iy * iw + ix);
        sr[nv] = img[j4];
        sg[nv] = img[j4 + 1];
        sb[nv] = img[j4 + 2];
        nv++;
        accJ[nAcc] = k;
        accQ[nAcc] = q;
        nAcc++;
      }
      if (nv < minViews) continue;

      // emit; mark the reference pixel and all supporting pixels as fused
      maskR[i >>> 5] |= 1 << (i & 31);
      for (let a = 0; a < nAcc; a++) {
        const m = c.nbMask[accJ[a]];
        const q = accQ[a];
        m[q >>> 5] |= 1 << (q & 31);
      }
      {
        let ix = ((x + 0.5) * rsx) | 0;
        let iy = ((y + 0.5) * rsy) | 0;
        if (ix >= c.imgRW) ix = c.imgRW - 1;
        if (iy >= c.imgRH) iy = c.imgRH - 1;
        const j4 = 4 * (iy * c.imgRW + ix);
        sr[0] = c.imgR[j4];
        sg[0] = c.imgR[j4 + 1];
        sb[0] = c.imgR[j4 + 2];
      }
      const inv = 1 / nv;
      const mxc = sX * inv;
      const myc = sY * inv;
      const mzc = sZ * inv;
      // camera → world: X_w = Rᵀ (X_c − t)
      const Rr = c.Rr;
      const tr = c.tr;
      const ax = mxc - tr[0];
      const ay = myc - tr[1];
      const az = mzc - tr[2];
      out.reserve();
      const n = out.count;
      const p3 = 3 * n;
      const pos = out.pos;
      const nrm = out.nrm;
      const col = out.col;
      pos[p3] = Rr[0] * ax + Rr[3] * ay + Rr[6] * az;
      pos[p3 + 1] = Rr[1] * ax + Rr[4] * ay + Rr[7] * az;
      pos[p3 + 2] = Rr[2] * ax + Rr[5] * ay + Rr[8] * az;
      let nl = Math.sqrt(snx * snx + sny * sny + snz * snz);
      let cosv = 1;
      if (nl > 0) {
        snx /= nl;
        sny /= nl;
        snz /= nl;
        cosv = Math.abs(snx * mxc + sny * myc + snz * mzc) / Math.sqrt(mxc * mxc + myc * myc + mzc * mzc);
      } else {
        // no normals at all: face the reference camera
        nl = Math.sqrt(mxc * mxc + myc * myc + mzc * mzc);
        snx = -mxc / nl;
        sny = -myc / nl;
        snz = -mzc / nl;
      }
      nrm[p3] = Rr[0] * snx + Rr[3] * sny + Rr[6] * snz;
      nrm[p3 + 1] = Rr[1] * snx + Rr[4] * sny + Rr[7] * snz;
      nrm[p3 + 2] = Rr[2] * snx + Rr[5] * sny + Rr[8] * snz;
      col[p3] = median8(sr, nv);
      col[p3 + 1] = median8(sg, nv);
      col[p3 + 2] = median8(sb, nv);
      out.rad[n] = (p.radiusScale * mzc) / fR / Math.sqrt(cosv > 0.2 ? cosv : 0.2);
      const cf = nv * confDen;
      out.conf[n] = cf < 1 ? cf : 1;
      out.count = n + 1;
      out.viewSum += nv;
      if (out.count >= p.maxPoints) return candidates;
    }
  }
  return candidates;
}

/** Fuse depth maps into a coloured point cloud (see file header). */
export async function fuseDepthMaps(
  views: DenseView[],
  store: DepthMapStore,
  neighbours: Map<number, number[]>,
  options: Partial<FusionOptions> = {},
  progress?: ProgressCallback,
  cancel?: CancelToken,
): Promise<PointCloud> {
  const o: FusionOptions = { ...DEFAULT_FUSION_OPTIONS, ...options };
  const t0 = performance.now();
  const yielder = new Yielder('fuseDepthMaps', cancel);
  const viewIndex = new Map<number, number>();
  views.forEach((v, i) => viewIndex.set(v.id, i));
  const maxNbOpt = Math.max(1, Math.floor(o.maxNeighbours));
  let maxNb = 0;
  for (const v of views) maxNb = Math.max(maxNb, Math.min(maxNbOpt, (neighbours.get(v.id) ?? []).length));
  const cache = new DepthMapCache(store, o.cacheSize > 0 ? o.cacheSize : maxNb + 2);
  const masks = new Map<number, Uint32Array>();
  const maskFor = (vi: number, dm: DepthMap): Uint32Array => {
    let m = masks.get(vi);
    if (!m || m.length * 32 < dm.width * dm.height) {
      m = new Uint32Array((dm.width * dm.height + 31) >>> 5);
      masks.set(vi, m);
    }
    return m;
  };
  const params: KernelParams = {
    minViews: Math.max(1, Math.floor(o.minViews)),
    maxRel: o.maxDepthError,
    maxReprojSq: o.maxReprojErrorPx * o.maxReprojErrorPx,
    cosNormal: Math.cos((o.maxNormalErrorDeg * Math.PI) / 180),
    maxCost: o.maxCost,
    step: Math.max(1, Math.floor(o.step)),
    usePlane: o.usePlaneDepth,
    radiusScale: o.radiusScale,
    maxPoints: o.maxPoints,
  };
  const sink = new PointSink();
  const scratch = new KernelScratch();
  let candidates = 0;
  let refsDone = 0;

  for (let ri = 0; ri < views.length && sink.count < params.maxPoints; ri++) {
    const ref = views[ri];
    progress?.({ stage: 'fusion', progress: ri / views.length, message: `view ${ri + 1}/${views.length}` });
    await yielder.tick();
    const nbIds = [...new Set(neighbours.get(ref.id) ?? [])].filter((id) => id !== ref.id && viewIndex.has(id)).slice(0, maxNbOpt);
    if (nbIds.length + 1 < params.minViews) continue;
    if (!(await cache.get(ref.id))) continue;
    const nbMaps = await cache.getMany(nbIds);
    // the reference may have been evicted while loading neighbours: re-fetch (hit or reload)
    const dmRef = await cache.get(ref.id);
    if (!dmRef) continue;
    const nbP: number[] = [];
    const nbImgP: number[] = [];
    const ctxNb = {
      depth: [] as Float32Array[],
      normal: [] as Float32Array[],
      cost: [] as (Float32Array | null)[],
      mask: [] as Uint32Array[],
      img: [] as (Uint8Array | Uint8ClampedArray)[],
    };
    for (let k = 0; k < nbIds.length; k++) {
      const dm = nbMaps[k];
      if (!dm) continue;
      const vi = viewIndex.get(nbIds[k])!;
      const v = views[vi];
      const cam = scaleCamera(v.cam, dm.width, dm.height);
      const { R, t } = relativeTransform(ref.pose, v.pose);
      nbP.push(...R, ...t, cam.f, cam.cx, cam.cy, dm.width, dm.height);
      nbImgP.push(v.image.width, v.image.height, v.image.width / dm.width, v.image.height / dm.height);
      ctxNb.depth.push(dm.depth);
      ctxNb.normal.push(dm.normal);
      ctxNb.cost.push(dm.cost ?? null);
      ctxNb.mask.push(maskFor(vi, dm));
      ctxNb.img.push(v.image.data);
    }
    const nNb = ctxNb.depth.length;
    if (nNb + 1 < params.minViews) continue;
    refsDone++;
    scratch.ensure(nNb);
    const camR = scaleCamera(ref.cam, dmRef.width, dmRef.height);
    const ctx: RefContext = {
      W: dmRef.width,
      H: dmRef.height,
      fR: camR.f,
      cxR: camR.cx,
      cyR: camR.cy,
      depthR: dmRef.depth,
      normalR: dmRef.normal,
      costR: dmRef.cost ?? null,
      maskR: maskFor(ri, dmRef),
      imgR: ref.image.data,
      imgRW: ref.image.width,
      imgRH: ref.image.height,
      Rr: ref.pose.R,
      tr: ref.pose.t,
      nNb,
      nbP: Float64Array.from(nbP),
      nbDepth: ctxNb.depth,
      nbNormal: ctxNb.normal,
      nbCost: ctxNb.cost,
      nbMask: ctxNb.mask,
      nbImg: ctxNb.img,
      nbImgP: Float64Array.from(nbImgP),
    };
    // rows in chunks between cooperative yields
    const H = dmRef.height;
    const rowsPerChunk = Math.max(params.step, Math.ceil(65536 / Math.max(1, dmRef.width)) * params.step);
    for (let y = 0; y < H && sink.count < params.maxPoints; y += rowsPerChunk) {
      candidates += fuseRows(ctx, sink, scratch, params, y, Math.min(H, y + rowsPerChunk));
      if (yielder.due()) await yielder.tick();
    }
  }

  if (o.filteredStore) {
    let k = 0;
    for (const [vi, mask] of masks) {
      progress?.({ stage: 'fusion', progress: 1, message: `writing filtered depth maps ${++k}/${masks.size}` });
      await yielder.tick();
      const dm = await cache.get(views[vi].id);
      if (!dm) continue;
      const n = dm.width * dm.height;
      const depth = new Float32Array(n);
      const normal = new Float32Array(3 * n);
      const cost = new Float32Array(n).fill(2);
      for (let i = 0; i < n; i++) {
        if (!((mask[i >>> 5] >>> (i & 31)) & 1)) continue;
        depth[i] = dm.depth[i];
        normal[3 * i] = dm.normal[3 * i];
        normal[3 * i + 1] = dm.normal[3 * i + 1];
        normal[3 * i + 2] = dm.normal[3 * i + 2];
        cost[i] = dm.cost ? dm.cost[i] : 0;
      }
      await o.filteredStore.put(views[vi].id, { width: dm.width, height: dm.height, depth, normal, cost });
    }
  }
  const count = sink.count;
  progress?.({ stage: 'fusion', progress: 1, message: `${count} points` });
  o.onStats?.({
    points: count,
    referencesProcessed: refsDone,
    candidates,
    meanViews: count ? sink.viewSum / count : 0,
    cacheHits: cache.hits,
    cacheMisses: cache.misses,
    timeMs: performance.now() - t0,
  });
  return {
    count,
    positions: sink.pos.slice(0, 3 * count),
    colors: sink.col.slice(0, 3 * count),
    normals: sink.nrm.slice(0, 3 * count),
    radii: sink.rad.slice(0, count),
    confidence: sink.conf.slice(0, count),
  };
}
