/**
 * CPU reference implementation of PatchMatch multi-view stereo
 * (Gipuma / ACMH / ACMM style). Bit-for-bit the same algorithm as the WGSL in
 * `patchmatch-wgsl.ts` (same PCG random streams, same sampling pattern, same
 * cost), up to floating-point differences. Used for tests and as a
 * low-resolution fallback when WebGPU is unavailable.
 *
 * Per iteration: red pass then black pass. For each pixel of the active colour:
 *   1. ACMH adaptive checkerboard sampling: in each of 8 regions (4 V-shaped near
 *      regions of 7 px, 4 strips of 11 px) take the neighbour hypothesis with the
 *      lowest current cost (planes are propagated, i.e. the depth at this pixel is
 *      re-derived from the neighbour's plane).
 *   2. Per-view bilateral-weighted NCC costs of the 8 candidates + the current
 *      hypothesis (9 × N matrix).
 *   3. ACMH multi-hypothesis view selection: a view is selected when > 2 of the 9
 *      costs are below τ_good(t) = 0.8·exp(−t²/90) and < 3 exceed τ_bad = 1.2; its
 *      weight is the mean of exp(−c²/(2·0.3²)) over the good costs, modulated by a
 *      soft prior from the neighbours' selected views. If no view is selected the
 *      `topK` views with the lowest best cost get weight 1 (documented fallback).
 *   4. Multi-view cost = Σ w_i c_i / Σ w_i (+ λ Σ w_i min(e_i, τ) / Σ w_i in
 *      geometric mode); the best of the 9 is kept.
 *   5. Refinement: up to 6 proposals (perturbed / random depth & normal, depth
 *      perturbation shrinking each iteration, random depths uniform in inverse depth).
 */
import type { DenseView, DepthMap, DepthRange, GeometricInput, PatchMatchOptions } from './types';
import {
  CHECKER_OFFSETS,
  CHECKER_REGION_SIZES,
  COST_MAX,
  buildSourceCamera,
  checkRange,
  clampSources,
  depthPerturbation,
  emptyDepthMap,
  goodCostThreshold,
  isGeometricActive,
  normalPerturbation,
  passSeed,
  patchOffsets,
  pcgHash,
  planesToDepthMap,
  priorToPlanes,
  resolvePatchMatchOptions,
  rgbaToGray8,
} from './patchmatch-common';

/** Timing / diagnostics of the last CPU run (for logging). */
export interface PatchMatchCPUStats {
  timeMs: number;
  iterations: number;
  geometric: boolean;
}

let lastStats: PatchMatchCPUStats = { timeMs: 0, iterations: 0, geometric: false };

export function lastPatchMatchCPUStats(): PatchMatchCPUStats {
  return { ...lastStats };
}

/**
 * Run PatchMatch on the CPU. `range` is the z-depth search range of the
 * reference view. See ARCHITECTURE §4.6 / `PatchMatchOptions`.
 */
export function patchMatchCPU(
  ref: DenseView,
  sources: DenseView[],
  range: DepthRange,
  opts?: Partial<PatchMatchOptions>,
  geometric?: GeometricInput,
): DepthMap {
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  checkRange(range);
  const o = resolvePatchMatchOptions(opts);
  const geo = isGeometricActive(opts, geometric);
  // the prior initialises the hypotheses whenever it is given (geometric pass, or a
  // prior-initialised photometric pass with `geometric: false`, e.g. multi-scale)
  const usePrior = !!geometric;
  const srcs = clampSources(sources, 'patchMatchCPU');
  const W = ref.cam.width;
  const H = ref.cam.height;
  if (ref.image.width !== W || ref.image.height !== H) throw new Error('patchMatchCPU: ref image size does not match ref.cam');
  const N = srcs.length;
  if (N === 0 || W === 0 || H === 0) return emptyDepthMap(W, H);
  if (geo && geometric!.sourceDepths.length < N) throw new Error('patchMatchCPU: geometric.sourceDepths must have one depth map per source');

  // ---- set-up ---------------------------------------------------------------
  const refGray = toFloat(rgbaToGray8(ref.image));
  const refRGB = ref.image.data;
  const srcGray: Float32Array[] = [];
  const srcW = new Int32Array(N);
  const srcH = new Int32Array(N);
  const camA = new Float64Array(9 * N);
  const camB = new Float64Array(3 * N);
  const camM = new Float64Array(9 * N);
  const camV = new Float64Array(3 * N);
  const dScaleX = new Float64Array(N);
  const dScaleY = new Float64Array(N);
  const dW = new Int32Array(N);
  const dH = new Int32Array(N);
  const srcDepth: Float32Array[] = [];
  for (let s = 0; s < N; s++) {
    const v = srcs[s];
    if (v.image.width !== v.cam.width || v.image.height !== v.cam.height) throw new Error('patchMatchCPU: source image size does not match cam');
    srcGray.push(toFloat(rgbaToGray8(v.image)));
    srcW[s] = v.cam.width;
    srcH[s] = v.cam.height;
    const dm = geo ? geometric!.sourceDepths[s] : undefined;
    const sc = buildSourceCamera(ref, v, dm ? { width: dm.width, height: dm.height } : undefined);
    camA.set(sc.A, 9 * s);
    camB.set(sc.b, 3 * s);
    camM.set(sc.M, 9 * s);
    camV.set(sc.v, 3 * s);
    dScaleX[s] = sc.depthScaleX;
    dScaleY[s] = sc.depthScaleY;
    dW[s] = sc.depthWidth;
    dH[s] = sc.depthHeight;
    srcDepth.push(dm ? dm.depth : new Float32Array(0));
  }
  const f = ref.cam.f;
  const cx = ref.cam.cx;
  const cy = ref.cam.cy;
  const invF = 1 / f;
  const zMin = range.min;
  const zMax = range.max;
  const invZMin = 1 / zMin;
  const invZMax = 1 / zMax;
  const lambda = o.geomLambda;
  const tau = o.geomMaxError;
  const minVar = o.minVariance;
  const topK = Math.min(o.topK, N);
  const numProposals = o.numSamplesPerIter;

  // patch sampling pattern
  const offs = patchOffsets(o.patchRadius, o.patchStep);
  const NS = offs.length * offs.length;
  const ox = new Float64Array(NS);
  const oy = new Float64Array(NS);
  const spatialTerm = new Float64Array(NS);
  {
    let k = 0;
    const sCoef = 1 / (2 * o.sigmaSpatial * o.sigmaSpatial);
    for (const dy of offs) {
      for (const dx of offs) {
        ox[k] = dx;
        oy[k] = dy;
        spatialTerm[k] = -Math.sqrt(dx * dx + dy * dy) * sCoef;
        k++;
      }
    }
  }
  // colour distance = mean absolute RGB difference in grey levels (0..255)
  const colorCoef = 1 / (3 * 2 * o.sigmaColor * o.sigmaColor);
  const NP = offs.length;
  const offLo = offs[0];
  const offHi = offs[NP - 1];
  const offStep = o.patchStep;

  // ---- state ----------------------------------------------------------------
  const nPix = W * H;
  const planes = new Float32Array(4 * nPix);
  const cost = new Float32Array(nPix);
  const photo = new Float32Array(nPix);
  const views = new Uint32Array(nPix);
  if (usePrior) priorToPlanes(geometric!.prior, W, H, planes, ref.cam);

  // per-pixel scratch
  const wk = new Float64Array(NS);
  const wrk = new Float64Array(NS);
  let refMean = 0;
  let refVar = 0;
  let invSumW = 0;
  const Hm = new Float64Array(9 * N); // homographies of the current hypothesis
  const C = new Float64Array(9 * N); // per-candidate per-view costs
  const candN = new Float64Array(27);
  const candZ = new Float64Array(9);
  const candOk = new Uint8Array(9);
  const weights = new Float64Array(N);
  const tmpScore = new Float64Array(N);
  let rng = 0;
  const rand = (): number => {
    rng = pcgHash(rng);
    return (rng >>> 8) / 16777216;
  };

  const prepareRef = (x: number, y: number): boolean => {
    const ci = 4 * (y * W + x);
    const cr = refRGB[ci];
    const cg = refRGB[ci + 1];
    const cb = refRGB[ci + 2];
    let sw = 0;
    let swr = 0;
    let swrr = 0;
    for (let k = 0; k < NS; k++) {
      let px = x + ox[k];
      let py = y + oy[k];
      px = px < 0 ? 0 : px > W - 1 ? W - 1 : px;
      py = py < 0 ? 0 : py > H - 1 ? H - 1 : py;
      const pi = py * W + px;
      const r = refGray[pi];
      const dc = Math.abs(refRGB[4 * pi] - cr) + Math.abs(refRGB[4 * pi + 1] - cg) + Math.abs(refRGB[4 * pi + 2] - cb);
      const w = Math.exp(spatialTerm[k] - dc * colorCoef);
      wk[k] = w;
      wrk[k] = w * r;
      sw += w;
      swr += w * r;
      swrr += w * r * r;
    }
    invSumW = 1 / sw;
    refMean = swr * invSumW;
    refVar = swrr * invSumW - refMean * refMean;
    return refVar >= minVar;
  };

  /** Fill Hm for hypothesis (n, z) at pixel; false if plane invalid. */
  const homographies = (x: number, y: number, nx: number, ny: number, nz: number, z: number): boolean => {
    const rx = (x - cx) * invF;
    const ry = (y - cy) * invF;
    const nr = nx * rx + ny * ry + nz;
    const delta = -z * nr;
    if (!(delta > 0)) return false;
    const m0 = nx * invF;
    const m1 = ny * invF;
    const m2 = nz - (nx * cx + ny * cy) * invF;
    const invDelta = 1 / delta;
    for (let s = 0; s < N; s++) {
      for (let r = 0; r < 3; r++) {
        const k = 9 * s + 3 * r;
        const bs = camB[3 * s + r] * invDelta;
        Hm[k] = camA[k] - bs * m0;
        Hm[k + 1] = camA[k + 1] - bs * m1;
        Hm[k + 2] = camA[k + 2] - bs * m2;
      }
    }
    return true;
  };

  const viewCost = (s: number, x: number, y: number): number => {
    const k0 = 9 * s;
    const H0 = Hm[k0];
    const H1 = Hm[k0 + 1];
    const H3 = Hm[k0 + 3];
    const H4 = Hm[k0 + 4];
    const H6 = Hm[k0 + 6];
    const H7 = Hm[k0 + 7];
    const hx = H0 * x + H1 * y + Hm[k0 + 2];
    const hy = H3 * x + H4 * y + Hm[k0 + 5];
    const hz = H6 * x + H7 * y + Hm[k0 + 8];
    const w = srcW[s];
    const h = srcH[s];
    if (!(hz > 1e-9)) return COST_MAX;
    const cu = hx / hz;
    const cv = hy / hz;
    if (!(cu >= 0 && cu <= w - 1 && cv >= 0 && cv <= h - 1)) return COST_MAX;
    const img = srcGray[s];
    const wm1 = w - 1;
    const hm1 = h - 1;
    let sws = 0;
    let swss = 0;
    let swrs = 0;
    // Fast path: if the 4 window corners project strictly inside the image (and in
    // front of the camera) every sample does (projective maps preserve convexity),
    // so no clamping is needed and the window is walked incrementally.
    let inside = true;
    for (let c = 0; c < 4 && inside; c++) {
      const dx = c & 1 ? offHi : offLo;
      const dy = c & 2 ? offHi : offLo;
      const qz = hz + dx * H6 + dy * H7;
      if (!(qz > 1e-9)) {
        inside = false;
        break;
      }
      const u = (hx + dx * H0 + dy * H1) / qz;
      const v = (hy + dx * H3 + dy * H4) / qz;
      inside = u >= 0 && u < wm1 - 1e-6 && v >= 0 && v < hm1 - 1e-6;
    }
    if (inside) {
      const sx = offStep * H0;
      const sy = offStep * H3;
      const sz = offStep * H6;
      let k = 0;
      for (let a = 0; a < NP; a++) {
        const dy = offLo + a * offStep;
        let qx = hx + offLo * H0 + dy * H1;
        let qy = hy + offLo * H3 + dy * H4;
        let qz = hz + offLo * H6 + dy * H7;
        for (let b = 0; b < NP; b++, k++) {
          const iz = 1 / qz;
          const u = qx * iz;
          const v = qy * iz;
          qx += sx;
          qy += sy;
          qz += sz;
          const x0 = u | 0;
          const y0 = v | 0;
          const fx = u - x0;
          const fy = v - y0;
          const r0 = y0 * w + x0;
          const r1 = r0 + w;
          const ia = img[r0];
          const ib = img[r0 + 1];
          const ic = img[r1];
          const id = img[r1 + 1];
          const top = ia + (ib - ia) * fx;
          const val = top + (ic + (id - ic) * fx - top) * fy;
          const wt = wk[k];
          sws += wt * val;
          swss += wt * val * val;
          swrs += wrk[k] * val;
        }
      }
    } else for (let k = 0; k < NS; k++) {
      const dx = ox[k];
      const dy = oy[k];
      const qz = hz + dx * H6 + dy * H7;
      if (!(qz > 1e-9)) return COST_MAX;
      const iz = 1 / qz;
      let u = (hx + dx * H0 + dy * H1) * iz;
      let v = (hy + dx * H3 + dy * H4) * iz;
      u = u < 0 ? 0 : u > wm1 ? wm1 : u;
      v = v < 0 ? 0 : v > hm1 ? hm1 : v;
      const x0 = u | 0;
      const y0 = v | 0;
      const fx = u - x0;
      const fy = v - y0;
      const x1 = x0 < wm1 ? x0 + 1 : x0;
      const y1 = y0 < hm1 ? y0 + 1 : y0;
      const r0 = y0 * w;
      const r1 = y1 * w;
      const a = img[r0 + x0];
      const b = img[r0 + x1];
      const c = img[r1 + x0];
      const d = img[r1 + x1];
      const top = a + (b - a) * fx;
      const bot = c + (d - c) * fx;
      const val = top + (bot - top) * fy;
      const wt = wk[k];
      sws += wt * val;
      swss += wt * val * val;
      swrs += wrk[k] * val;
    }
    const ms = sws * invSumW;
    const vs = swss * invSumW - ms * ms;
    if (vs < minVar) return COST_MAX;
    const cov = swrs * invSumW - refMean * ms;
    const ncc = cov / Math.sqrt(refVar * vs);
    const c = 1 - ncc;
    return c < 0 ? 0 : c > COST_MAX ? COST_MAX : c;
  };

  const geoCost = (s: number, x: number, y: number, z: number): number => {
    const k = 9 * s;
    const qx = z * (camA[k] * x + camA[k + 1] * y + camA[k + 2]) + camB[3 * s];
    const qy = z * (camA[k + 3] * x + camA[k + 4] * y + camA[k + 5]) + camB[3 * s + 1];
    const qz = z * (camA[k + 6] * x + camA[k + 7] * y + camA[k + 8]) + camB[3 * s + 2];
    if (!(qz > 1e-9)) return tau;
    const pu = qx / qz;
    const pv = qy / qz;
    if (!(pu >= 0 && pu <= srcW[s] - 1 && pv >= 0 && pv <= srcH[s] - 1)) return tau;
    const dw = dW[s];
    const dh = dH[s];
    if (dw === 0 || dh === 0) return tau;
    let di = Math.floor((pu + 0.5) * dScaleX[s] - 0.5 + 0.5);
    let dj = Math.floor((pv + 0.5) * dScaleY[s] - 0.5 + 0.5);
    di = di < 0 ? 0 : di > dw - 1 ? dw - 1 : di;
    dj = dj < 0 ? 0 : dj > dh - 1 ? dh - 1 : dj;
    const zs = srcDepth[s][dj * dw + di];
    if (!(zs > 0)) return tau;
    const rx = zs * (camM[k] * pu + camM[k + 1] * pv + camM[k + 2]) + camV[3 * s];
    const ry = zs * (camM[k + 3] * pu + camM[k + 4] * pv + camM[k + 5]) + camV[3 * s + 1];
    const rz = zs * (camM[k + 6] * pu + camM[k + 7] * pv + camM[k + 8]) + camV[3 * s + 2];
    if (!(rz > 1e-9)) return tau;
    const e = Math.hypot(rx / rz - x, ry / rz - y);
    return e < tau ? e : tau;
  };

  const randomInvDepth = (): number => 1 / (invZMax + rand() * (invZMin - invZMax));

  /** Random unit normal facing the camera along ray (rx, ry, 1); writes into candN slot. */
  const randomNormal = (rx: number, ry: number, out: Float64Array, o3: number): void => {
    const zc = 2 * rand() - 1;
    const phi = 2 * Math.PI * rand();
    const rr = Math.sqrt(Math.max(0, 1 - zc * zc));
    let nx = rr * Math.cos(phi);
    let ny = rr * Math.sin(phi);
    let nz = zc;
    if (nx * rx + ny * ry + nz > 0) {
      nx = -nx;
      ny = -ny;
      nz = -nz;
    }
    out[o3] = nx;
    out[o3 + 1] = ny;
    out[o3 + 2] = nz;
  };

  /** Weighted multi-view cost of (n, z) using `weights` (views with w > 0 only). Returns [photo, total] via out. */
  const aggOut = new Float64Array(2);
  const evalWeighted = (x: number, y: number, nx: number, ny: number, nz: number, z: number, sumW: number): boolean => {
    if (!(z >= zMin && z <= zMax)) return false;
    if (!homographies(x, y, nx, ny, nz, z)) return false;
    let p = 0;
    let g = 0;
    for (let s = 0; s < N; s++) {
      const w = weights[s];
      if (w <= 0) continue;
      p += w * viewCost(s, x, y);
      if (geo) g += w * geoCost(s, x, y, z);
    }
    const inv = 1 / sumW;
    aggOut[0] = p * inv;
    aggOut[1] = p * inv + (geo ? lambda * g * inv : 0);
    return true;
  };

  // ---- initialisation ---------------------------------------------------------
  {
    const seed0 = passSeed(o.seed, geo ? 0x1ffff : 0xffff);
    const sel = new Uint8Array(N);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = y * W + x;
        rng = pcgHash((idx ^ seed0) >>> 0);
        const rx = (x - cx) * invF;
        const ry = (y - cy) * invF;
        const pz = planes[4 * idx + 3];
        let ok = usePrior && pz >= zMin && pz <= zMax && planes[4 * idx] * rx + planes[4 * idx + 1] * ry + planes[4 * idx + 2] < 0;
        if (!ok) {
          randomNormal(rx, ry, candN, 0);
          planes[4 * idx] = candN[0];
          planes[4 * idx + 1] = candN[1];
          planes[4 * idx + 2] = candN[2];
          planes[4 * idx + 3] = randomInvDepth();
        }
        if (!prepareRef(x, y)) {
          cost[idx] = COST_MAX;
          photo[idx] = COST_MAX;
          views[idx] = 0;
          continue;
        }
        const nx = planes[4 * idx];
        const ny = planes[4 * idx + 1];
        const nz = planes[4 * idx + 2];
        const z = planes[4 * idx + 3];
        ok = homographies(x, y, nx, ny, nz, z);
        for (let s = 0; s < N; s++) tmpScore[s] = ok ? viewCost(s, x, y) : COST_MAX;
        // top-K mean
        sel.fill(0);
        let sum = 0;
        let g = 0;
        let mask = 0;
        for (let k = 0; k < topK; k++) {
          let bi = -1;
          let bc = Infinity;
          for (let s = 0; s < N; s++) {
            if (!sel[s] && tmpScore[s] < bc) {
              bc = tmpScore[s];
              bi = s;
            }
          }
          sel[bi] = 1;
          mask |= 1 << bi;
          sum += bc;
          if (geo) g += geoCost(bi, x, y, z);
        }
        photo[idx] = sum / topK;
        cost[idx] = sum / topK + (geo ? (lambda * g) / topK : 0);
        views[idx] = mask >>> 0;
      }
    }
  }

  // ---- propagation iterations ------------------------------------------------
  const iterations = geo ? o.geometricIterations : o.iterations;
  const nRegions = CHECKER_REGION_SIZES.length;
  const regionStart = new Int32Array(nRegions);
  for (let r = 1; r < nRegions; r++) regionStart[r] = regionStart[r - 1] + CHECKER_REGION_SIZES[r - 1];

  for (let it = 0; it < iterations; it++) {
    const tauGood = goodCostThreshold(o, it);
    const pert = depthPerturbation(o, it, usePrior);
    const npert = normalPerturbation(o, it, usePrior);
    for (let colour = 0; colour < 2; colour++) {
      const seedP = passSeed(o.seed, (geo ? 0x100 : 0) + 2 * it + colour);
      for (let y = 0; y < H; y++) {
        for (let x = (y + colour) & 1; x < W; x += 2) {
          const idx = y * W + x;
          if (!prepareRef(x, y)) {
            cost[idx] = COST_MAX;
            photo[idx] = COST_MAX;
            views[idx] = 0;
            continue;
          }
          rng = pcgHash((idx ^ seedP) >>> 0);
          const rx = (x - cx) * invF;
          const ry = (y - cy) * invF;
          // slot 0: current hypothesis
          candN[0] = planes[4 * idx];
          candN[1] = planes[4 * idx + 1];
          candN[2] = planes[4 * idx + 2];
          candZ[0] = planes[4 * idx + 3];
          candOk[0] = 1;
          // slots 1..8: best neighbour of each region
          for (let r = 0; r < nRegions; r++) {
            let best = -1;
            let bestC = Infinity;
            const start = regionStart[r];
            const cnt = CHECKER_REGION_SIZES[r];
            for (let k = 0; k < cnt; k++) {
              const qx = x + CHECKER_OFFSETS[2 * (start + k)];
              const qy = y + CHECKER_OFFSETS[2 * (start + k) + 1];
              if (qx < 0 || qy < 0 || qx >= W || qy >= H) continue;
              const qi = qy * W + qx;
              if (cost[qi] < bestC) {
                bestC = cost[qi];
                best = qi;
              }
            }
            const slot = r + 1;
            candOk[slot] = 0;
            if (best < 0) continue;
            const qx = best % W;
            const qy = (best - qx) / W;
            const nx = planes[4 * best];
            const ny = planes[4 * best + 1];
            const nz = planes[4 * best + 2];
            const zq = planes[4 * best + 3];
            const nrq = nx * (qx - cx) * invF + ny * (qy - cy) * invF + nz;
            const nrp = nx * rx + ny * ry + nz;
            if (!(nrp < 0)) continue;
            const zp = (zq * nrq) / nrp;
            if (!(zp >= zMin && zp <= zMax)) continue;
            candN[3 * slot] = nx;
            candN[3 * slot + 1] = ny;
            candN[3 * slot + 2] = nz;
            candZ[slot] = zp;
            candOk[slot] = 1;
          }
          // per-view costs (candidates identical to an earlier one reuse its costs)
          for (let j = 0; j < 9; j++) {
            const base = j * N;
            let dup = -1;
            if (candOk[j]) {
              for (let k = 0; k < j; k++) {
                if (
                  candOk[k] &&
                  candN[3 * k] === candN[3 * j] &&
                  candN[3 * k + 1] === candN[3 * j + 1] &&
                  candN[3 * k + 2] === candN[3 * j + 2] &&
                  Math.abs(candZ[k] - candZ[j]) <= 1e-5 * candZ[k]
                ) {
                  dup = k;
                  break;
                }
              }
            }
            if (dup >= 0) {
              for (let s = 0; s < N; s++) C[base + s] = C[dup * N + s];
            } else if (candOk[j] && homographies(x, y, candN[3 * j], candN[3 * j + 1], candN[3 * j + 2], candZ[j])) {
              for (let s = 0; s < N; s++) C[base + s] = viewCost(s, x, y);
            } else {
              candOk[j] = 0;
              for (let s = 0; s < N; s++) C[base + s] = COST_MAX;
            }
          }
          // view selection (ACMH) with a soft prior from the 4-neighbours' selections
          let nNb = 0;
          const m0 = y > 0 ? views[idx - W] : 0;
          const m1 = y < H - 1 ? views[idx + W] : 0;
          const m2 = x > 0 ? views[idx - 1] : 0;
          const m3 = x < W - 1 ? views[idx + 1] : 0;
          nNb = (y > 0 ? 1 : 0) + (y < H - 1 ? 1 : 0) + (x > 0 ? 1 : 0) + (x < W - 1 ? 1 : 0);
          let sumW = 0;
          for (let s = 0; s < N; s++) {
            let nGood = 0;
            let nBad = 0;
            let conf = 0;
            let minC = COST_MAX;
            for (let j = 0; j < 9; j++) {
              if (!candOk[j]) continue;
              const c = C[j * N + s];
              if (c < minC) minC = c;
              if (c < tauGood) {
                nGood++;
                conf += Math.exp((-c * c) / 0.18);
              } else if (c > o.badCost) {
                nBad++;
              }
            }
            tmpScore[s] = minC;
            let w = 0;
            if (nGood > 2 && nBad < 3) {
              const bit = 1 << s;
              const cnt = ((m0 & bit) !== 0 ? 1 : 0) + ((m1 & bit) !== 0 ? 1 : 0) + ((m2 & bit) !== 0 ? 1 : 0) + ((m3 & bit) !== 0 ? 1 : 0);
              const prior = 0.3 + (0.7 * cnt) / Math.max(1, nNb);
              w = (conf / nGood) * prior;
            }
            weights[s] = w;
            sumW += w;
          }
          if (!(sumW > 0)) {
            // fallback: top-K views by their best cost, equal weights
            sumW = 0;
            for (let k = 0; k < topK; k++) {
              let bi = 0;
              let bc = Infinity;
              for (let s = 0; s < N; s++) {
                if (weights[s] === 0 && tmpScore[s] < bc) {
                  bc = tmpScore[s];
                  bi = s;
                }
              }
              weights[bi] = 1;
              sumW += 1;
            }
          }
          let mask = 0;
          for (let s = 0; s < N; s++) if (weights[s] > 0) mask |= 1 << s;
          // aggregate candidates
          const invSum = 1 / sumW;
          let bestJ = -1;
          let bestTotal = Infinity;
          let bestPhoto = COST_MAX;
          for (let j = 0; j < 9; j++) {
            if (!candOk[j]) continue;
            let p = 0;
            let g = 0;
            for (let s = 0; s < N; s++) {
              const w = weights[s];
              if (w <= 0) continue;
              p += w * C[j * N + s];
              if (geo) g += w * geoCost(s, x, y, candZ[j]);
            }
            p *= invSum;
            const tot = p + (geo ? lambda * g * invSum : 0);
            if (tot < bestTotal) {
              bestTotal = tot;
              bestPhoto = p;
              bestJ = j;
            }
          }
          let bnx: number;
          let bny: number;
          let bnz: number;
          let bz: number;
          if (bestJ < 0) {
            // nothing valid (should not happen: current hypothesis is always kept valid)
            bnx = candN[0];
            bny = candN[1];
            bnz = candN[2];
            bz = candZ[0];
            bestTotal = COST_MAX + (geo ? lambda * tau : 0);
            bestPhoto = COST_MAX;
          } else {
            bnx = candN[3 * bestJ];
            bny = candN[3 * bestJ + 1];
            bnz = candN[3 * bestJ + 2];
            bz = candZ[bestJ];
          }
          // refinement proposals
          if (numProposals > 0) {
            const zPert = Math.min(zMax, Math.max(zMin, bz * (1 + pert * (2 * rand() - 1))));
            let pnx = bnx + npert * (2 * rand() - 1);
            let pny = bny + npert * (2 * rand() - 1);
            let pnz = bnz + npert * (2 * rand() - 1);
            const pl = Math.hypot(pnx, pny, pnz);
            pnx /= pl;
            pny /= pl;
            pnz /= pl;
            if (!(pnx * rx + pny * ry + pnz < 0)) {
              pnx = bnx;
              pny = bny;
              pnz = bnz;
            }
            const zRand = randomInvDepth();
            randomNormal(rx, ry, candN, 0); // scratch slot 0 (no longer needed)
            const qnx = candN[0];
            const qny = candN[1];
            const qnz = candN[2];
            const cnx0 = bnx;
            const cny0 = bny;
            const cnz0 = bnz;
            const cz0 = bz;
            for (let pIdx = 0; pIdx < numProposals; pIdx++) {
              let nx: number;
              let ny: number;
              let nz: number;
              let z: number;
              switch (pIdx) {
                case 0:
                  nx = cnx0; ny = cny0; nz = cnz0; z = zPert;
                  break;
                case 1:
                  nx = pnx; ny = pny; nz = pnz; z = cz0;
                  break;
                case 2:
                  nx = qnx; ny = qny; nz = qnz; z = zRand;
                  break;
                case 3:
                  nx = pnx; ny = pny; nz = pnz; z = zPert;
                  break;
                case 4:
                  nx = cnx0; ny = cny0; nz = cnz0; z = zRand;
                  break;
                default:
                  nx = qnx; ny = qny; nz = qnz; z = cz0;
                  break;
              }
              if (!evalWeighted(x, y, nx, ny, nz, z, sumW)) continue;
              if (aggOut[1] < bestTotal) {
                bestTotal = aggOut[1];
                bestPhoto = aggOut[0];
                bnx = nx;
                bny = ny;
                bnz = nz;
                bz = z;
              }
            }
          }
          planes[4 * idx] = bnx;
          planes[4 * idx + 1] = bny;
          planes[4 * idx + 2] = bnz;
          planes[4 * idx + 3] = bz;
          cost[idx] = bestTotal;
          photo[idx] = bestPhoto;
          views[idx] = mask >>> 0;
        }
      }
    }
  }

  const out = planesToDepthMap(planes, photo, W, H, range, o.maxCost);
  const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  lastStats = { timeMs: t1 - t0, iterations, geometric: geo };
  return out;
}

function toFloat(g: Uint8Array): Float32Array {
  const out = new Float32Array(g.length);
  for (let i = 0; i < g.length; i++) out[i] = g[i] / 255;
  return out;
}
