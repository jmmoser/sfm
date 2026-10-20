/**
 * Gaussian-splat training with ground-truth poses (splat-gt.html).
 *
 * `window.runSplatGT(config)` trains src/splat on a synthetic dataset (e2e/data/<name>/,
 * served by Vite) WITHOUT the SfM engine: exact poses, exact intrinsics (per-frame focal of
 * the -real datasets), JPEGs undistorted to pinhole with the GT k1, k2 (`undistortImage`).
 * Training frames: every `trainStride`-th frame (default 2) or an explicit list; test frames:
 * the others (every `testEvery`-th of them, default 4). The initial Gaussians come from
 * (a) a fused cloud PLY, e.g. a dense-gt run (`e2e/out/dense-gt/<name>.ply`, GT frame),
 * (b) GT surface samples coloured from the frames with GT depth maps, or (c) random points
 * in the dataset ROI. At each `evalAt` iteration the held-out views are rendered and scored:
 * PSNR / SSIM over the valid (undistorted) pixels, raw and after a per-view per-channel
 * affine colour fit (exposure / white-balance drift of the -real datasets).
 *
 * `splatGTSummary(r)` strips the big arrays, `splatGTWrite(prefix)` streams the trained model
 * (INRIA PLY), the result JSON and GT | render | error panels of a few test views to Node
 * through `window.__splatWrite` (e2e/run-splat-gt.mjs).
 */
import type { CameraIntrinsics, PointCloud, RGBAImage } from '../../core/types';
import { getGPU } from '../../gpu/device';
import { undistortImage } from '../../dense/undistort';
import { decodePLY } from '../../io/ply';
import { Rng } from '../../math/random';
import {
  SH_C0,
  SplatTrainer,
  allocSplatModel,
  backgroundShell,
  concatModels,
  encodeSplatPLY,
  initFromPointCloud,
  logit,
  splatCamera,
  type ShDegree,
  type SplatCamera,
  type SplatImage,
  type SplatModel,
  type SplatTrainerOptions,
  type SplatTrainingView,
} from '../../splat';
import { panelPNG } from '../dense-gt/depth-eval';

// ------------------------------------------------------------------ types

export interface SplatGTConfig {
  dataset: string;
  baseUrl?: string;
  /** Long side of the undistorted training / test images (default 800, the 'fast' dense size). */
  maxSize?: number;
  /** Training frames (overrides trainStride / first). */
  trainFrames?: number[];
  trainStride?: number;
  first?: number;
  /** Test frames (default: the frames not used for training, every `testEvery`-th). */
  testFrames?: number[];
  testEvery?: number;
  /** Use the per-frame GT focal length of frames.json when present (default true). */
  perFrameFocal?: boolean;
  init: {
    kind: 'cloud' | 'gt' | 'random';
    /** PLY URL for 'cloud' (e.g. '/e2e/out/dense-gt/object-real-fast.ply'). */
    url?: string;
    /** Gaussians taken from the source (default: maxGaussians / 2). */
    count?: number;
    /**
     * Extra GT surface samples over the whole scene (coloured from the GT-depth frames), a
     * stand-in for the sparse SfM points the engine has: MVS clouds cover only the near field.
     */
    sparse?: number;
    opacity?: number;
    normalRatio?: number;
    scaleFactor?: number;
  };
  /** Far background shell for sky / distant structure. */
  shell?: { count: number; radius: number; opacity?: number };
  maxGaussians?: number;
  iterations?: number;
  /** Iterations at which the test views are scored (default 0, 500, 1000, 3000, 7000 up to `iterations`). */
  evalAt?: number[];
  /** Iterations per `step` call (default 100). */
  chunk?: number;
  /** Trainer options passed through (shDegree, lr, mcmc, colourCorrection, resolution, …). */
  trainer?: Partial<SplatTrainerOptions>;
  /** Number of test views written as panels by splatGTWrite (default 4). */
  panels?: number;
  seed?: number;
}

export interface ViewScore {
  frame: number;
  psnr: number;
  ssim: number;
  psnrCC: number;
  ssimCC: number;
}

export interface EvalPoint {
  iteration: number;
  /** Training wall time so far (ms, evaluation excluded). */
  trainMs: number;
  count: number;
  psnr: number;
  ssim: number;
  psnrCC: number;
  ssimCC: number;
  /** Mean render time per test view (ms, incl. readback). */
  renderMs: number;
  /** GPU ms per training iteration and pass since the previous evaluation point (profile mode). */
  gpuMs: Record<string, number> | null;
  /** Training iterations / s since the previous evaluation point. */
  segmentItPerSec: number;
  views: ViewScore[];
}

export interface SplatGTResult {
  dataset: string;
  config: SplatGTConfig;
  adapter: string;
  size: { width: number; height: number };
  trainFrames: number[];
  testFrames: number[];
  init: { kind: string; count: number; shell: number; ms: number };
  timings: Record<string, number>;
  evals: EvalPoint[];
  train: {
    iterations: number;
    ms: number;
    iterationsPerSecond: number;
    count: number;
    loss: number;
    l1: number;
    ssim: number;
    maxIntersections: number;
    intersectionCapacity: number;
    mcmc: { refinements: number; relocated: number; added: number };
    gpuMsPerIteration: Record<string, number> | null;
    log: { iteration: number; loss: number; count: number; itPerSec: number }[];
  };
  memory: { totalMB: number; breakdownMB: Record<string, number> };
  _model: SplatModel | null;
  _panels: { frame: number; png: Blob }[];
}

type WriteFn = (path: string, data: string, encoding: 'base64' | 'utf8', append: boolean) => Promise<void>;

declare global {
  interface Window {
    runSplatGT: (config: SplatGTConfig) => Promise<SplatGTResult>;
    splatGTSummary: (r?: SplatGTResult) => unknown;
    splatGTWrite: (prefix: string) => Promise<string[]>;
    splatGTLast?: SplatGTResult;
    splatGTReady?: boolean;
    __splatWrite?: WriteFn;
    __splatLog?: (msg: string) => void;
  }
}

interface MetaLite {
  name: string;
  scene: string;
  width: number;
  height: number;
  intrinsics: { f: number; cx: number; cy: number; k1: number; k2: number };
  roi: { min: number[]; max: number[] };
  sceneExtent: number;
  gt: { spacing: number; pointsFile: string; normalsFile: string; focus: { pointsFile: string; normalsFile: string } | null };
}

interface FrameLite {
  index: number;
  file: string;
  pose: { R: number[]; t: number[] };
  f?: number;
  depthFile?: string;
}

// ------------------------------------------------------------------ helpers

const statusEl = typeof document !== 'undefined' ? document.getElementById('status') : null;

function log(msg: string): void {
  if (statusEl) statusEl.textContent = msg;
  if (window.__splatLog) void window.__splatLog(msg);
  else console.log(msg);
}

function join(base: string, rel: string): string {
  return base.endsWith('/') ? base + rel : `${base}/${rel}`;
}

async function fetchOk(url: string): Promise<Response> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  return r;
}

async function decodeJPEG(url: string): Promise<RGBAImage> {
  const blob = await (await fetchOk(url)).blob();
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  const id = ctx.getImageData(0, 0, bmp.width, bmp.height);
  bmp.close();
  return { width: id.width, height: id.height, data: id.data };
}

function camOf(meta: MetaLite, f: FrameLite, perFrameFocal: boolean): CameraIntrinsics {
  return { width: meta.width, height: meta.height, ...meta.intrinsics, f: perFrameFocal && f.f ? f.f : meta.intrinsics.f };
}

async function loadView(base: string, meta: MetaLite, f: FrameLite, maxSize: number, perFrameFocal: boolean): Promise<SplatTrainingView> {
  const img = await decodeJPEG(join(base, f.file));
  const u = undistortImage(img, camOf(meta, f, perFrameFocal), maxSize);
  return { id: f.index, cam: u.cam, pose: { R: Float64Array.from(f.pose.R), t: Float64Array.from(f.pose.t) }, image: u.image };
}

/** Separable 11-tap Gaussian blur (σ 1.5), zero padding — the SSIM window of the loss. */
function blur(src: Float32Array, W: number, H: number, w: Float32Array, tmp: Float32Array, out: Float32Array): void {
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let k = -5; k <= 5; k++) {
        const xx = x + k;
        if (xx >= 0 && xx < W) s += w[k + 5] * src[y * W + xx];
      }
      tmp[y * W + x] = s;
    }
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let k = -5; k <= 5; k++) {
        const yy = y + k;
        if (yy >= 0 && yy < H) s += w[k + 5] * tmp[yy * W + x];
      }
      out[y * W + x] = s;
    }
}

const SSIM_W = (() => {
  const w = new Float32Array(11);
  let s = 0;
  for (let i = -5; i <= 5; i++) s += w[i + 5] = Math.exp(-(i * i) / 4.5);
  return w.map((v) => v / s);
})();

/** PSNR / SSIM of `x` (RGB floats per pixel, stride 4) against `gt` (RGBA8) over gt alpha ≥ 128; optional per-channel affine fit. */
export function scoreImage(x: Float32Array, gt: RGBAImage, affine: boolean): { psnr: number; ssim: number } {
  const W = gt.width;
  const H = gt.height;
  const n = W * H;
  const mask = new Uint8Array(n);
  let nv = 0;
  for (let i = 0; i < n; i++) if (gt.data[4 * i + 3] >= 128) (mask[i] = 1), nv++;
  const X = new Float32Array(n);
  const Y = new Float32Array(n);
  const tmp = new Float32Array(n);
  const b = [new Float32Array(n), new Float32Array(n), new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  const prod = new Float32Array(n);
  let se = 0;
  let ssim = 0;
  for (let c = 0; c < 3; c++) {
    let a = 1;
    let off = 0;
    if (affine) {
      let sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (let i = 0; i < n; i++) {
        if (!mask[i]) continue;
        const xv = x[4 * i + c];
        const yv = gt.data[4 * i + c] / 255;
        sx += xv;
        sy += yv;
        sxx += xv * xv;
        sxy += xv * yv;
      }
      const d = nv * sxx - sx * sx;
      if (d > 1e-12) {
        a = (nv * sxy - sx * sy) / d;
        off = (sy - a * sx) / nv;
      }
    }
    for (let i = 0; i < n; i++) {
      if (!mask[i]) {
        X[i] = 0;
        Y[i] = 0;
        continue;
      }
      X[i] = Math.min(1, Math.max(0, a * x[4 * i + c] + off));
      Y[i] = gt.data[4 * i + c] / 255;
      se += (X[i] - Y[i]) ** 2;
    }
    blur(X, W, H, SSIM_W, tmp, b[0]);
    blur(Y, W, H, SSIM_W, tmp, b[1]);
    for (let i = 0; i < n; i++) prod[i] = X[i] * X[i];
    blur(prod, W, H, SSIM_W, tmp, b[2]);
    for (let i = 0; i < n; i++) prod[i] = Y[i] * Y[i];
    blur(prod, W, H, SSIM_W, tmp, b[3]);
    for (let i = 0; i < n; i++) prod[i] = X[i] * Y[i];
    blur(prod, W, H, SSIM_W, tmp, b[4]);
    const C1 = 1e-4;
    const C2 = 9e-4;
    for (let i = 0; i < n; i++) {
      if (!mask[i]) continue;
      const mx = b[0][i];
      const my = b[1][i];
      const sxx = b[2][i] - mx * mx;
      const syy = b[3][i] - my * my;
      const sxy = b[4][i] - mx * my;
      ssim += ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2));
    }
  }
  const mse = se / (3 * Math.max(1, nv));
  return { psnr: 10 * Math.log10(1 / Math.max(mse, 1e-12)), ssim: ssim / (3 * Math.max(1, nv)) };
}

function viewCamera(v: SplatTrainingView): SplatCamera {
  return splatCamera(v.cam, v.pose);
}

function toRGBA(img: SplatImage): RGBAImage {
  const n = img.width * img.height;
  const d = new Uint8ClampedArray(4 * n);
  for (let i = 0; i < n; i++) {
    d[4 * i] = 255 * img.data[4 * i];
    d[4 * i + 1] = 255 * img.data[4 * i + 1];
    d[4 * i + 2] = 255 * img.data[4 * i + 2];
    d[4 * i + 3] = 255;
  }
  return { width: img.width, height: img.height, data: d };
}

function errorImage(img: SplatImage, gt: RGBAImage): RGBAImage {
  const n = img.width * img.height;
  const d = new Uint8ClampedArray(4 * n);
  for (let i = 0; i < n; i++) {
    let e = 0;
    for (let c = 0; c < 3; c++) e += Math.abs(img.data[4 * i + c] - gt.data[4 * i + c] / 255);
    const v = Math.min(255, (255 * e) / 0.6);
    d[4 * i] = v;
    d[4 * i + 1] = v * 0.6;
    d[4 * i + 2] = 0;
    d[4 * i + 3] = 255;
  }
  return { width: img.width, height: img.height, data: d };
}

// ------------------------------------------------------------------ initialisation

async function initCloud(url: string): Promise<PointCloud> {
  const buf = new Uint8Array(await (await fetchOk(url)).arrayBuffer());
  const r = decodePLY(buf);
  if (!r.points) throw new Error(`${url}: no point cloud`);
  return r.points;
}

/** GT surface samples coloured from the frames that have an exact GT depth map (distorted images). */
async function initGt(base: string, meta: MetaLite, frames: FrameLite[], count: number, rng: Rng, perFrameFocal: boolean): Promise<PointCloud> {
  const fetchF32 = async (u: string): Promise<Float32Array> => new Float32Array(await (await fetchOk(join(base, u))).arrayBuffer());
  const fetchI8 = async (u: string): Promise<Int8Array> => new Int8Array(await (await fetchOk(join(base, u))).arrayBuffer());
  const P0 = await fetchF32(meta.gt.pointsFile);
  const N0 = await fetchI8(meta.gt.normalsFile);
  let P = P0;
  let N = N0;
  if (meta.gt.focus) {
    const P1 = await fetchF32(meta.gt.focus.pointsFile);
    const N1 = await fetchI8(meta.gt.focus.normalsFile);
    P = new Float32Array(P0.length + P1.length);
    P.set(P0);
    P.set(P1, P0.length);
    N = new Int8Array(N0.length + N1.length);
    N.set(N0);
    N.set(N1, N0.length);
  }
  const total = P.length / 3;
  const pick = rng.sample(total, Math.min(total, Math.round(count * 1.6)));
  const depthFrames = frames.filter((f) => f.depthFile);
  const srcs: { cam: CameraIntrinsics; R: number[]; t: number[]; depth: Float32Array; img: RGBAImage }[] = [];
  for (const f of depthFrames) srcs.push({ cam: camOf(meta, f, perFrameFocal), R: f.pose.R, t: f.pose.t, depth: await fetchF32(f.depthFile!), img: await decodeJPEG(join(base, f.file)) });
  const pos: number[] = [];
  const col: number[] = [];
  const nrm: number[] = [];
  for (const i of pick) {
    if (pos.length / 3 >= count) break;
    const X = P[3 * i], Y = P[3 * i + 1], Z = P[3 * i + 2];
    for (const s of srcs) {
      const R = s.R;
      const t = s.t;
      const zc = R[6] * X + R[7] * Y + R[8] * Z + t[2];
      if (zc <= 1e-3) continue;
      const xn = (R[0] * X + R[1] * Y + R[2] * Z + t[0]) / zc;
      const yn = (R[3] * X + R[4] * Y + R[5] * Z + t[1]) / zc;
      const r2 = xn * xn + yn * yn;
      const d = 1 + s.cam.k1 * r2 + s.cam.k2 * r2 * r2;
      const u = Math.round(s.cam.f * d * xn + s.cam.cx);
      const v = Math.round(s.cam.f * d * yn + s.cam.cy);
      if (u < 0 || v < 0 || u >= s.cam.width || v >= s.cam.height) continue;
      const dz = s.depth[v * s.cam.width + u];
      if (!(dz > 0) || Math.abs(dz - zc) > 0.01 * zc + 2 * meta.gt.spacing) continue;
      const o = 4 * (v * s.cam.width + u);
      pos.push(X, Y, Z);
      col.push(s.img.data[o], s.img.data[o + 1], s.img.data[o + 2]);
      nrm.push(N[3 * i] / 127, N[3 * i + 1] / 127, N[3 * i + 2] / 127);
      break;
    }
  }
  return { count: pos.length / 3, positions: Float32Array.from(pos), colors: Uint8Array.from(col), normals: Float32Array.from(nrm) };
}

function subsampleCloud(pc: PointCloud, n: number, rng: Rng): PointCloud {
  if (pc.count <= n) return pc;
  const pick = Array.from(rng.sample(pc.count, n)).sort((a, b) => a - b);
  const take = <T extends Float32Array | Uint8Array>(a: T | undefined, k: number): T | undefined => {
    if (!a) return undefined;
    const out = new (a.constructor as { new (n: number): T })(n * k);
    for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) out[k * i + c] = a[k * pick[i] + c];
    return out;
  };
  return { count: n, positions: take(pc.positions, 3)!, colors: take(pc.colors, 3)!, normals: take(pc.normals, 3), radii: take(pc.radii, 1), confidence: take(pc.confidence, 1) };
}

function mergeClouds(a: PointCloud, b: PointCloud): PointCloud {
  const n = a.count + b.count;
  const cat = <T extends Float32Array | Uint8Array>(x: T | undefined, y: T | undefined, k: number, Ctor: { new (n: number): T }): T => {
    const out = new Ctor(n * k);
    if (x) out.set(x.subarray(0, a.count * k));
    if (y) out.set(y.subarray(0, b.count * k), a.count * k);
    return out;
  };
  return {
    count: n,
    positions: cat(a.positions, b.positions, 3, Float32Array),
    colors: cat(a.colors, b.colors, 3, Uint8Array),
    normals: a.normals || b.normals ? cat(a.normals, b.normals, 3, Float32Array) : undefined,
    radii: a.radii || b.radii ? cat(a.radii, b.radii, 1, Float32Array) : undefined,
  };
}

function initRandom(meta: MetaLite, count: number, shDegree: ShDegree, rng: Rng): SplatModel {
  const m = allocSplatModel(count, shDegree);
  const lo = meta.roi.min;
  const hi = meta.roi.max;
  const vol = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
  const s = Math.log(0.5 * Math.cbrt(vol / count));
  const K = shDegree === 0 ? 1 : 4;
  for (let i = 0; i < count; i++) {
    for (let k = 0; k < 3; k++) {
      m.means[3 * i + k] = rng.uniform(lo[k], hi[k]);
      m.logScales[3 * i + k] = s;
      m.sh[3 * K * i + k] = (rng.next() - 0.5) / SH_C0;
    }
    m.opacities[i] = logit(0.1);
  }
  return m;
}

// ------------------------------------------------------------------ main

async function runSplatGT(config: SplatGTConfig): Promise<SplatGTResult> {
  const T: Record<string, number> = {};
  const base = config.baseUrl ?? `/e2e/data/${config.dataset}/`;
  const meta = (await (await fetchOk(join(base, 'meta.json'))).json()) as MetaLite;
  const frames = (await (await fetchOk(join(base, 'frames.json'))).json()) as FrameLite[];
  const maxSize = config.maxSize ?? 800;
  const perFrameFocal = config.perFrameFocal ?? true;
  const rng = new Rng(config.seed ?? 1);
  const stride = config.trainStride ?? 2;
  const first = config.first ?? 0;
  const trainIdx = config.trainFrames ?? frames.filter((f) => f.index >= first && (f.index - first) % stride === 0).map((f) => f.index);
  const trainSet = new Set(trainIdx);
  const testEvery = config.testEvery ?? 4;
  const testIdx = config.testFrames ?? frames.filter((f) => !trainSet.has(f.index)).map((f) => f.index).filter((_, k) => k % testEvery === 0);
  const byIndex = new Map(frames.map((f) => [f.index, f]));
  const iterations = config.iterations ?? 3000;
  const evalAt = (config.evalAt ?? [0, 500, 1000, 3000, 7000]).filter((x) => x <= iterations);
  if (!evalAt.includes(iterations)) evalAt.push(iterations);
  const maxGaussians = config.maxGaussians ?? 300_000;
  const shDegree = config.trainer?.shDegree ?? 0;

  let t0 = performance.now();
  const train: SplatTrainingView[] = [];
  for (const i of trainIdx) train.push(await loadView(base, meta, byIndex.get(i)!, maxSize, perFrameFocal));
  const test: SplatTrainingView[] = [];
  for (const i of testIdx) test.push(await loadView(base, meta, byIndex.get(i)!, maxSize, perFrameFocal));
  T.loadViews = performance.now() - t0;
  const W = train[0].cam.width;
  const H = train[0].cam.height;
  log(`[splat-gt] ${meta.name}: ${train.length} training / ${test.length} test views at ${W}×${H} (${T.loadViews.toFixed(0)} ms)`);

  // ---- initial model
  t0 = performance.now();
  const initCount = Math.min(maxGaussians, config.init.count ?? Math.round(maxGaussians / 2));
  let model: SplatModel;
  const io = { maxCount: initCount, shDegree, opacity: config.init.opacity, normalRatio: config.init.normalRatio, scaleFactor: config.init.scaleFactor, seed: config.seed };
  if (config.init.kind === 'cloud') {
    if (!config.init.url) throw new Error('init.kind = cloud needs init.url');
    let pc = subsampleCloud(await initCloud(config.init.url), initCount, rng);
    log(`[splat-gt] cloud ${config.init.url}: ${pc.count} points used`);
    if (config.init.sparse) pc = mergeClouds(pc, await initGt(base, meta, frames, config.init.sparse, rng, perFrameFocal));
    model = initFromPointCloud(pc, { ...io, maxCount: pc.count });
  } else if (config.init.kind === 'gt') {
    const pc = await initGt(base, meta, frames, initCount, rng, perFrameFocal);
    model = initFromPointCloud(pc, io);
  } else model = initRandom(meta, initCount, shDegree, rng);
  let shellCount = 0;
  if (config.shell && config.shell.count > 0) {
    const shell = backgroundShell(
      train.map((v) => ({ cam: viewCamera(v), data: v.image.data })),
      { count: config.shell.count, radius: config.shell.radius, shDegree, opacity: config.shell.opacity },
    );
    shellCount = shell.count;
    model = concatModels(model, shell);
  }
  T.init = performance.now() - t0;
  log(`[splat-gt] init ${config.init.kind}: ${model.count} Gaussians (${shellCount} shell) in ${T.init.toFixed(0)} ms`);

  // ---- trainer
  const ctx = await getGPU();
  if (!ctx) throw new Error('WebGPU unavailable');
  t0 = performance.now();
  const trainer = await SplatTrainer.create(ctx, { maxGaussians, iterations, ...(config.trainer ?? {}), shDegree });
  T.compile = performance.now() - t0;
  t0 = performance.now();
  trainer.init(model, train);
  T.upload = performance.now() - t0;
  const mem = trainer.memory();

  const evals: EvalPoint[] = [];
  let segStart = { it: 0, ms: 0 };
  const evaluate = async (trainMs: number): Promise<void> => {
    const sst = trainer.stats();
    const gpuSeg = sst.profiledIterations ? Object.fromEntries(Object.entries(sst.gpuMs).map(([k, v]) => [k, v / sst.profiledIterations])) : null;
    for (const [k, v] of Object.entries(sst.gpuMs)) gpuTotal[k] = (gpuTotal[k] ?? 0) + v;
    gpuIters += sst.profiledIterations;
    trainer.resetProfile();
    const segItPerSec = trainer.iteration > segStart.it ? (1000 * (trainer.iteration - segStart.it)) / Math.max(1, trainMs - segStart.ms) : 0;
    segStart = { it: trainer.iteration, ms: trainMs };
    const views: ViewScore[] = [];
    let rms = 0;
    for (const v of test) {
      const r0 = performance.now();
      const img = await trainer.render(viewCamera(v));
      rms += performance.now() - r0;
      const raw = scoreImage(img.data, v.image, false);
      const cc = scoreImage(img.data, v.image, true);
      views.push({ frame: v.id, psnr: raw.psnr, ssim: raw.ssim, psnrCC: cc.psnr, ssimCC: cc.ssim });
    }
    const mean = (k: keyof ViewScore): number => views.reduce((s, v) => s + v[k], 0) / Math.max(1, views.length);
    const e: EvalPoint = {
      iteration: trainer.iteration,
      trainMs,
      count: trainer.gaussians,
      psnr: mean('psnr'),
      ssim: mean('ssim'),
      psnrCC: mean('psnrCC'),
      ssimCC: mean('ssimCC'),
      renderMs: rms / Math.max(1, test.length),
      gpuMs: gpuSeg,
      segmentItPerSec: segItPerSec,
      views,
    };
    evals.push(e);
    if (gpuSeg) log(`[splat-gt]   GPU ms / iteration since the last evaluation: ${Object.entries(gpuSeg).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(', ')} (total ${Object.values(gpuSeg).reduce((a, b) => a + b, 0).toFixed(2)}); ${segItPerSec.toFixed(1)} it/s`);
    log(`[splat-gt] it ${e.iteration}: PSNR ${e.psnr.toFixed(2)} dB SSIM ${e.ssim.toFixed(4)} | colour-fit PSNR ${e.psnrCC.toFixed(2)} SSIM ${e.ssimCC.toFixed(4)} | ${e.count} Gaussians, train ${(trainMs / 1000).toFixed(1)} s, render ${e.renderMs.toFixed(1)} ms/view`);
  };

  const gpuTotal: Record<string, number> = {};
  let gpuIters = 0;
  const chunk = config.chunk ?? 100;
  let trainMs = 0;
  const trainLog: SplatGTResult['train']['log'] = [];
  let last = { loss: NaN, l1: NaN, ssim: NaN };
  for (const target of evalAt) {
    while (trainer.iteration < target) {
      const n = Math.min(chunk, target - trainer.iteration);
      const r = await trainer.step(n);
      trainMs += r.ms;
      last = r;
      trainLog.push({ iteration: r.iteration, loss: r.loss, count: r.count, itPerSec: r.iterationsPerSecond });
      if (r.iteration % 500 === 0 || r.iteration === iterations) {
        log(`[splat-gt]   it ${r.iteration}: loss ${r.loss.toFixed(4)} (L1 ${r.l1.toFixed(4)}, SSIM ${r.ssim.toFixed(4)}), ${r.count} Gaussians, ${r.iterationsPerSecond.toFixed(1)} it/s${r.overflow ? ', intersection overflow' : ''}`);
      }
    }
    await evaluate(trainMs);
  }
  const st = trainer.stats();
  const gpuMs = gpuIters ? Object.fromEntries(Object.entries(gpuTotal).map(([k, v]) => [k, v / gpuIters])) : null;
  if (gpuMs) log(`[splat-gt] GPU ms / iteration (whole run): ${Object.entries(gpuMs).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(', ')} (total ${Object.values(gpuMs).reduce((a, b) => a + b, 0).toFixed(2)})`);

  // panels: GT | render | |error| for a few test views (final model)
  const panels: SplatGTResult['_panels'] = [];
  const nPanels = Math.min(config.panels ?? 4, test.length);
  for (let k = 0; k < nPanels; k++) {
    const v = test[Math.floor((k * test.length) / nPanels)];
    const img = await trainer.render(viewCamera(v));
    const s = scoreImage(img.data, v.image, false);
    panels.push({ frame: v.id, png: await panelPNG([{ img: v.image, label: `GT ${v.id}` }, { img: toRGBA(img), label: `splat ${s.psnr.toFixed(2)} dB` }, { img: errorImage(img, v.image), label: '|error|' }], 3) });
  }
  const finalModel = await trainer.readModel();
  trainer.destroy();
  const result: SplatGTResult = {
    dataset: meta.name,
    config,
    adapter: `${ctx.info.vendor}/${ctx.info.architecture}${ctx.isSoftware ? ' (software)' : ''}`,
    size: { width: W, height: H },
    trainFrames: trainIdx,
    testFrames: testIdx,
    init: { kind: config.init.kind, count: model.count, shell: shellCount, ms: T.init },
    timings: T,
    evals,
    train: {
      iterations: trainer.iteration,
      ms: trainMs,
      iterationsPerSecond: (1000 * trainer.iteration) / Math.max(1, trainMs),
      count: st.count,
      loss: last.loss,
      l1: last.l1,
      ssim: last.ssim,
      maxIntersections: st.maxIntersections,
      intersectionCapacity: st.intersectionCapacity,
      mcmc: st.mcmc,
      gpuMsPerIteration: gpuMs,
      log: trainLog,
    },
    memory: { totalMB: mem.total / 2 ** 20, breakdownMB: Object.fromEntries(Object.entries(mem.breakdown).map(([k, v]) => [k, v / 2 ** 20])) },
    _model: finalModel,
    _panels: panels,
  };
  window.splatGTLast = result;
  return result;
}

function summary(r: SplatGTResult | undefined = window.splatGTLast): unknown {
  if (!r) return null;
  const { _model, _panels, ...rest } = r;
  void _model;
  void _panels;
  return rest;
}

function blobToBase64(b: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => {
      const s = fr.result as string;
      resolve(s.slice(s.indexOf(',') + 1));
    };
    fr.onerror = () => reject(fr.error);
    fr.readAsDataURL(b);
  });
}

async function writeBinary(path: string, data: Blob | Uint8Array): Promise<void> {
  const w = window.__splatWrite;
  if (!w) throw new Error('window.__splatWrite is not installed (run through e2e/run-splat-gt.mjs)');
  const blob = data instanceof Blob ? data : new Blob([data as BlobPart]);
  const CH = 6 << 20;
  for (let off = 0; off < blob.size; off += CH) await w(path, await blobToBase64(blob.slice(off, Math.min(blob.size, off + CH))), 'base64', off > 0);
}

async function writeOutputs(prefix: string): Promise<string[]> {
  const r = window.splatGTLast;
  if (!r) throw new Error('no result: call runSplatGT first');
  const out: string[] = [];
  if (r._model) {
    await writeBinary(`${prefix}.splat.ply`, encodeSplatPLY(r._model, { comments: [`splat-gt ${r.dataset} iterations ${r.train.iterations}`] }));
    out.push(`${prefix}.splat.ply`);
  }
  for (const p of r._panels) {
    const path = `${prefix}/test-${String(p.frame).padStart(6, '0')}.png`;
    await writeBinary(path, p.png);
    out.push(path);
  }
  return out;
}

window.runSplatGT = runSplatGT;
window.splatGTSummary = summary;
window.splatGTWrite = writeOutputs;
window.splatGTReady = true;
log('ready');
