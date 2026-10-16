/**
 * Dense MVS validation with ground-truth poses (dense-gt.html).
 *
 * `window.runDenseGT(config)` runs the dense pipeline of src/dense on a synthetic dataset
 * (e2e/data/<name>/, served by Vite at /e2e/data/<name>/) with the EXACT camera poses and
 * intrinsics, so every error it measures comes from MVS / fusion, not from SfM:
 *
 *   frames (subset) → JPEG decode (createImageBitmap + OffscreenCanvas) → undistortImage
 *   → GT surfel depth per view (gt-depth.ts) → sparse points from GT samples + z-buffer
 *   visibility → selectSourceViews → PatchMatchGPU photometric (+ optional multi-scale
 *   prior) (+ geometric pass) → MemoryDepthMapStore → fuseDepthMaps (+ filtered maps)
 *   → optional tsdfMesh.
 *
 * Per view it reports PatchMatch timings and the depth accuracy of the photometric,
 * geometric and fusion-filtered maps against the surfel GT depth (all GT pixels and a
 * "well-posed surface" mask), and against the dataset's exact depth maps for the frames
 * that have one (every 20th) — which also validates the surfel GT.
 *
 * The result (stats + fused cloud + optional mesh) is kept in `window.denseGTLast`;
 * `denseGTSummary(r)` strips the big arrays, `denseGTWrite(prefix)` streams PLYs, debug
 * panels and the subset visibility of the completeness set to Node through
 * `window.__denseWrite` (Playwright exposeFunction, see e2e/run-dense-gt.mjs), and
 * `denseGTRender(opts)` renders the fused cloud with src/viewer and writes screenshots.
 */
import type { CameraIntrinsics, PointCloud, Pose, RGBAImage, TriangleMesh } from '../../core/types';
import { getGPU } from '../../gpu/device';
import {
  DENSE_PRESETS,
  MemoryDepthMapStore,
  PatchMatchGPU,
  filterPointCloud,
  fuseDepthMaps,
  neighboursFromSelection,
  selectSourceViews,
  tsdfMesh,
  undistortImage,
  depthHaloMask,
  removeDepthHalos,
  type DenseView,
  type DepthMap,
  type FusionOptions,
  type FusionStats,
  type PointCloudFilterOptions,
  type PatchMatchGPUTimings,
  type PatchMatchOptions,
  type PatchMatchSelfTestResult,
  type TsdfOptions,
  type TsdfStats,
  type ViewSelectionOptions,
  type DenseQuality,
} from '../../dense';
import { encodeMeshPLY, encodePLY } from '../../io/ply';
import { Viewer } from '../../viewer';
import {
  maskNearFocus,
  normalsFromDepth,
  renderSurfelDepth,
  sparsePointsFromGt,
  subsetVisibility,
  undistortGtDepth,
  type GtDepthMap,
  type GtSurfelSet,
} from './gt-depth';
import { costToRGBA, depthToRGBA, errorToRGBA, evalDepth, panelPNG, surfaceMask, type DepthEval } from './depth-eval';

// ------------------------------------------------------------------ types

export interface DenseGTConfig {
  /** Dataset directory name under e2e/data (e.g. 'object', 'yard'). */
  dataset: string;
  /**
   * Start from `DENSE_PRESETS[preset]` (sources, iterations, geometric passes, multi-scale, view
   * selection, fusion incl. object/scene minViews, halo filter, SOR); explicit fields override it.
   * The image size and the number of views stay the harness's (`maxSize`, `maxViews`):
   * SwiftShader is too slow for the preset sizes.
   */
  preset?: DenseQuality;
  /** Base URL of the dataset (default `/e2e/data/<dataset>/`). */
  baseUrl?: string;
  /** Explicit frame indices (overrides first/stride/maxViews). */
  frames?: number[];
  /** First frame (default 0). */
  first?: number;
  /** Frame stride (default floor(numFrames / maxViews)). */
  stride?: number;
  /** Number of views (default 24 for object scenes, 30 otherwise). */
  maxViews?: number;
  /** Long side of the undistorted dense images (default 480). */
  maxSize?: number;
  /** Source views per reference (default 8). */
  numSources?: number;
  /** Extra view-selection options. */
  selection?: Partial<ViewSelectionOptions>;
  /** GT samples drawn as sparse-point candidates (default 30 000). */
  sparseCount?: number;
  /** Photometric PatchMatch options (also the base of the geometric pass). */
  pm?: Partial<PatchMatchOptions>;
  /**
   * ACMM multi-scale recipe: photometric at ½ resolution for all views, then a
   * prior-initialised photometric pass at full resolution (`fullResIterations`, default 2).
   */
  multiScale?: boolean;
  fullResIterations?: number;
  /** Run the geometric-consistency pass (default true). */
  geometric?: boolean;
  /** Number of geometric passes (default 1); pass k uses the maps of pass k − 1 as prior and sources. */
  geometricPasses?: number;
  /** Overrides for the geometric pass. */
  geo?: Partial<PatchMatchOptions>;
  fusion?: Partial<FusionOptions>;
  /**
   * Apply `removeDepthHalos` (default options) to the fusion input maps before fusion, as the
   * engine does (default false). Independent of `fusion.haloFilter`.
   */
  preHalo?: boolean;
  /** `filterPointCloud` options applied to the fused cloud before scoring (default: none). */
  filter?: PointCloudFilterOptions;
  /** TSDF mesh of the fusion-filtered maps (default false). */
  tsdf?: boolean | TsdfOptions;
  /** Per-view depth evaluation (default true). */
  evalDepth?: boolean;
  /** Surfel disc radius in sample spacings (default 1.3). */
  surfelRadius?: number;
  /** Frames to dump debug panels for, or a count (default: up to 3 frames with exact GT depth). */
  debugViews?: number[] | number;
  /** Visibility of the completeness GT set in the used views (default true). */
  subsetVisibility?: boolean;
  /** Run PatchMatchGPU.selfTest() first (default true). */
  selfTest?: boolean;
  /** Keep the depth maps so `denseGTWrite` saves them (`<prefix>/maps/{half,photo,geo}-NNNNNN.f32`). */
  saveMaps?: boolean;
  /**
   * Skip PatchMatch and load maps saved by `saveMaps` from this URL prefix (e.g.
   * '/e2e/out/dense-gt/object/maps/'): fusion / TSDF experiments without re-running MVS.
   */
  loadMaps?: string;
  /**
   * Oracle: skip PatchMatch and fuse the GT surfel depth maps (cost 0) instead — the upper
   * bound of the fused cloud given resolution, view coverage and fusion settings.
   */
  oracle?: boolean;
  /** Depth maps fed to fusion: 'geo' (default when the geometric pass ran) or 'photo'. */
  fusionInput?: 'geo' | 'photo';
  /**
   * Per view, fuse with that view as the ONLY reference and evaluate its filtered map:
   * the coverage of multi-view-consistent pixels before fusion's first-come consumption
   * (default true).
   */
  consistencyCoverage?: boolean;
}

export interface PassStats {
  ms: number;
  timings: PatchMatchGPUTimings | null;
  /** Fraction of pixels with a valid depth. */
  validPx: number;
}

export interface ViewEval {
  all: DepthEval;
  surface: DepthEval;
}

export interface ViewStats {
  frame: number;
  sources: number[];
  scores: number[];
  depthMin: number;
  depthMax: number;
  gtDepthRange: [number, number];
  half?: PassStats;
  photo: PassStats;
  geo?: PassStats;
  fusedPx: number;
  /** vs the surfel GT. */
  eval?: { photo: ViewEval; geo?: ViewEval; fused: ViewEval; half?: ViewEval; consistent?: ViewEval };
  /** vs the dataset's exact depth map (frames with a depth file only). */
  exact?: { surfelVsExact: ViewEval; photo: ViewEval; geo?: ViewEval; fused: ViewEval };
  /** Halo filter on the fusion input (when `fusion.haloFilter` or `preHalo` is set): pixel counts by GT error class. */
  halo?: HaloCounts;
}

/** Valid pixels of a depth map by error class vs the surfel GT: noGt (sky), bad (> 5 %), mid (1–5 %), good (≤ 1 %). */
export interface HaloCounts {
  all: { noGt: number; bad: number; mid: number; good: number };
  rejected: { noGt: number; bad: number; mid: number; good: number };
}

export interface DenseGTResult {
  config: DenseGTConfig;
  dataset: { name: string; scene: string; numFrames: number; width: number; height: number; denseTau: number; sceneExtent: number };
  frames: number[];
  cam: CameraIntrinsics;
  gpu: { description: string; isSoftware: boolean; selfTest: PatchMatchSelfTestResult | null };
  sparse: { drawn: number; kept: number; meanTrackLength: number };
  views: ViewStats[];
  fusion: FusionStats | null;
  tsdf: TsdfStats | null;
  timings: Record<string, number>;
  /** Aggregates over views (pixel-weighted means of the per-view evals). */
  aggregate: Record<string, Record<string, number>>;
  cloud: PointCloud;
  mesh: TriangleMesh | null;
  /** Name of the GT set used for completeness ('focus' | 'base') and its subset visibility. */
  completenessSet: string;
  completenessVisibility: Uint8Array | null;
  /** Sum of the per-view halo counts (when `fusion.haloFilter` or `preHalo` is set). */
  halo: HaloCounts | null;
  /** Kept for debug panels / rendering. */
  _views: DenseView[];
  _debug: { frame: number; png: Blob }[];
  _maps: { half: Map<number, DepthMap> | null; photo: Map<number, DepthMap>; geo: Map<number, DepthMap> | null } | null;
}

interface SynthMetaLite {
  name: string;
  scene: string;
  width: number;
  height: number;
  numFrames: number;
  intrinsics: { f: number; cx: number; cy: number; k1: number; k2: number };
  denseTau: number;
  sceneExtent: number;
  roi: { min: number[]; max: number[] };
  focus: { min: number[]; max: number[] };
  gt: { spacing: number; pointsFile: string; normalsFile: string; visibilityFile: string | null; focus: { spacing: number; pointsFile: string; normalsFile: string; visibilityFile: string | null } | null };
}

interface SynthFrameLite {
  index: number;
  file: string;
  pose: { R: number[]; t: number[] };
  depthFile?: string;
}

type WriteFn = (path: string, data: string, encoding: 'base64' | 'utf8', append: boolean) => Promise<void>;

declare global {
  interface Window {
    runDenseGT: (config: DenseGTConfig) => Promise<DenseGTResult>;
    denseGTSummary: (r?: DenseGTResult) => unknown;
    denseGTWrite: (prefix: string) => Promise<string[]>;
    denseGTRender: (opts: RenderOptions) => Promise<string[]>;
    denseGTLast?: DenseGTResult;
    denseGTReady?: boolean;
    __denseWrite?: WriteFn;
    __denseLog?: (msg: string) => void;
  }
}

// ------------------------------------------------------------------ helpers

const statusEl = typeof document !== 'undefined' ? document.getElementById('status') : null;

function log(msg: string): void {
  if (statusEl) statusEl.textContent = msg;
  if (window.__denseLog) void window.__denseLog(msg);
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

async function fetchTyped<T>(url: string, Ctor: { new (b: ArrayBuffer): T }): Promise<T> {
  return new Ctor(await (await fetchOk(url)).arrayBuffer());
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

function poseOf(f: SynthFrameLite): Pose {
  return { R: Float64Array.from(f.pose.R), t: Float64Array.from(f.pose.t) };
}

function validFraction(dm: DepthMap): number {
  let v = 0;
  for (let i = 0; i < dm.depth.length; i++) if (dm.depth[i] > 0) v++;
  return v / Math.max(1, dm.depth.length);
}

function blobToBase64(b: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => {
      const s = r.result as string;
      resolve(s.slice(s.indexOf(',') + 1));
    };
    r.onerror = () => reject(r.error);
    r.readAsDataURL(b);
  });
}

/** Stream a binary blob to Node in ≤ 6 MB base64 chunks. */
async function writeBinary(path: string, data: Blob | ArrayBuffer | Uint8Array): Promise<void> {
  const w = window.__denseWrite;
  if (!w) throw new Error('window.__denseWrite is not installed (run through e2e/run-dense-gt.mjs)');
  const blob = data instanceof Blob ? data : new Blob([data as BlobPart]);
  const CH = 6 << 20;
  if (blob.size === 0) return w(path, '', 'base64', false);
  for (let off = 0; off < blob.size; off += CH) {
    await w(path, await blobToBase64(blob.slice(off, Math.min(blob.size, off + CH))), 'base64', off > 0);
  }
}

function evalBoth(dm: Pick<DepthMap, 'depth' | 'normal'>, gtDepth: Float32Array, gtNormal: Float32Array | null, tau: number, mask: Uint8Array): ViewEval {
  return { all: evalDepth(dm, gtDepth, gtNormal, tau), surface: evalDepth(dm, gtDepth, gtNormal, tau, mask) };
}

/** Pixel-weighted aggregate of per-view evals. */
function aggregate(views: ViewStats[], pick: (v: ViewStats) => DepthEval | undefined): Record<string, number> {
  let gt = 0, valid = 0, w05 = 0, w1 = 0, o5 = 0, wt = 0;
  const med: number[] = [];
  const nrm: number[] = [];
  for (const v of views) {
    const e = pick(v);
    if (!e || !e.gtPixels) continue;
    const nv = e.valid * e.gtPixels;
    gt += e.gtPixels;
    valid += nv;
    w05 += e.within05 * nv;
    w1 += e.within1 * nv;
    o5 += e.outlier5 * nv;
    wt += e.withinTau * nv;
    if (Number.isFinite(e.medianRel)) med.push(e.medianRel);
    if (Number.isFinite(e.medianNormalDeg)) nrm.push(e.medianNormalDeg);
  }
  med.sort((a, b) => a - b);
  nrm.sort((a, b) => a - b);
  return {
    gtPixels: gt,
    valid: gt ? valid / gt : 0,
    within05: valid ? w05 / valid : 0,
    within1: valid ? w1 / valid : 0,
    outlier5: valid ? o5 / valid : 0,
    withinTau: valid ? wt / valid : 0,
    completeness1: gt ? w1 / gt : 0,
    medianOfMedianRel: med.length ? med[med.length >> 1] : NaN,
    medianNormalDeg: nrm.length ? nrm[nrm.length >> 1] : NaN,
  };
}

// ------------------------------------------------------------------ main pipeline

/** Effective config: `DENSE_PRESETS[config.preset]` under the explicit fields. */
function applyPreset(config: DenseGTConfig, isObject: boolean): DenseGTConfig {
  if (!config.preset) return config;
  const P = DENSE_PRESETS[config.preset];
  if (!P) throw new Error(`unknown preset ${config.preset}`);
  return {
    ...config,
    numSources: config.numSources ?? P.numSources,
    pm: { ...P.patchMatch, iterations: P.iterations, ...(config.pm ?? {}) },
    geo: { geometricIterations: P.geometricIterations, ...(config.geo ?? {}) },
    geometric: config.geometric ?? P.geometricPasses > 0,
    geometricPasses: config.geometricPasses ?? Math.max(1, P.geometricPasses),
    multiScale: config.multiScale ?? P.multiScale,
    fullResIterations: config.fullResIterations ?? P.fullResIterations,
    selection: { ...P.viewSelection, ...(config.selection ?? {}) },
    fusion: {
      ...P.fusion,
      minViews: isObject ? P.fusionMinViewsObject : P.fusionMinViewsScene,
      maxNeighbours: P.fusionMaxNeighbours,
      ...(config.fusion ?? {}),
    },
    preHalo: config.preHalo ?? P.haloFilter,
    filter: config.filter ?? (P.sorK > 0 ? { statistical: { k: P.sorK, std: P.sorStd } } : undefined),
  };
}

async function runDenseGT(configIn: DenseGTConfig): Promise<DenseGTResult> {
  let config = configIn;
  const T: Record<string, number> = {};
  const tAll = performance.now();
  let tStage = tAll;
  const stage = (name: string): void => {
    const now = performance.now();
    T[name] = (T[name] ?? 0) + (now - tStage);
    tStage = now;
  };
  const base = config.baseUrl ?? `/e2e/data/${config.dataset}/`;
  const meta = (await (await fetchOk(join(base, 'meta.json'))).json()) as SynthMetaLite;
  const allFrames = (await (await fetchOk(join(base, 'frames.json'))).json()) as SynthFrameLite[];
  const isObject = meta.scene === 'object';
  config = applyPreset(config, isObject);
  const maxViews = config.maxViews ?? (isObject ? 24 : 30);
  const first = config.first ?? 0;
  const stride = config.stride ?? Math.max(1, Math.floor((allFrames.length - first) / maxViews));
  const frameIdx = config.frames ?? Array.from({ length: Math.min(maxViews, Math.ceil((allFrames.length - first) / stride)) }, (_, k) => first + k * stride);
  const frames = frameIdx.map((i) => {
    const f = allFrames.find((x) => x.index === i);
    if (!f) throw new Error(`frame ${i} not in frames.json`);
    return f;
  });
  const maxSize = config.maxSize ?? 480;
  const tau = meta.denseTau;
  const doEval = config.evalDepth !== false;
  const kR = config.surfelRadius ?? 1.3;
  log(`[dense-gt] ${meta.name}: ${frames.length} views (frames ${frameIdx[0]}..${frameIdx[frameIdx.length - 1]}, stride ${stride}), max size ${maxSize}`);

  // ---- GT samples -----------------------------------------------------------
  const gtBase: GtSurfelSet = {
    name: 'base',
    points: await fetchTyped(join(base, meta.gt.pointsFile), Float32Array),
    normals: await fetchTyped(join(base, meta.gt.normalsFile), Int8Array),
    visibility: meta.gt.visibilityFile ? await fetchTyped(join(base, meta.gt.visibilityFile), Uint8Array) : null,
    spacing: meta.gt.spacing,
    keep: null,
  };
  const sets: GtSurfelSet[] = [gtBase];
  let gtFocus: GtSurfelSet | null = null;
  if (meta.gt.focus) {
    const fm = meta.gt.focus;
    gtFocus = {
      name: 'focus',
      points: await fetchTyped(join(base, fm.pointsFile), Float32Array),
      normals: await fetchTyped(join(base, fm.normalsFile), Int8Array),
      visibility: fm.visibilityFile ? await fetchTyped(join(base, fm.visibilityFile), Uint8Array) : null,
      spacing: fm.spacing,
      keep: null,
    };
    gtBase.keep = maskNearFocus(gtBase.points, gtFocus.points, gtBase.spacing);
    sets.push(gtFocus);
    let kept = 0;
    for (let i = 0; i < gtBase.keep.length; i++) kept += gtBase.keep[i];
    log(`[dense-gt] GT: base ${gtBase.points.length / 3} samples (${kept} kept away from the focus set), focus ${gtFocus.points.length / 3}`);
  }
  stage('loadGt');

  // ---- frames: decode + undistort ---------------------------------------------
  const cam0: CameraIntrinsics = { width: meta.width, height: meta.height, ...meta.intrinsics };
  const views: DenseView[] = [];
  let tDecode = 0;
  let tUnd = 0;
  for (const f of frames) {
    const t0 = performance.now();
    const img = await decodeJPEG(join(base, f.file));
    if (img.width !== meta.width || img.height !== meta.height) throw new Error(`frame ${f.index}: ${img.width}×${img.height}, expected ${meta.width}×${meta.height}`);
    const t1 = performance.now();
    const u = undistortImage(img, cam0, maxSize);
    tDecode += t1 - t0;
    tUnd += performance.now() - t1;
    views.push({ id: f.index, cam: u.cam, pose: poseOf(f), image: u.image });
  }
  T.decode = tDecode;
  T.undistort = tUnd;
  tStage = performance.now();
  const cam = views[0].cam;
  log(`[dense-gt] decoded + undistorted ${views.length} frames → ${cam.width}×${cam.height}, f ${cam.f.toFixed(2)} (decode ${tDecode.toFixed(0)} ms, undistort ${tUnd.toFixed(0)} ms)`);

  // ---- GT depth per view (surfel z-buffer) -------------------------------------
  const gtMaps: GtDepthMap[] = views.map((v) => renderSurfelDepth(v.cam, v.pose, sets, { radiusFactor: kR }));
  stage('gtDepth');

  // ---- sparse points + view selection --------------------------------------------
  const sparse = sparsePointsFromGt(sets, views, gtMaps, { count: config.sparseCount ?? 30_000, radiusFactor: kR });
  stage('sparse');
  const sel = selectSourceViews(views, { xyz: sparse.xyz, visibility: sparse.visibility }, { numSources: config.numSources ?? 8, ...(config.selection ?? {}) });
  stage('selection');
  log(`[dense-gt] sparse points: ${sparse.kept} of ${sparse.drawn} drawn (mean track ${sparse.meanTrackLength.toFixed(1)}); selection done`);

  // ---- PatchMatch ----------------------------------------------------------------
  const ctx = await getGPU();
  if (!ctx) throw new Error('WebGPU unavailable');
  const pmOpts: Partial<PatchMatchOptions> = { ...(config.pm ?? {}) };
  let pm: PatchMatchGPU | null = null;
  let selfTest: PatchMatchSelfTestResult | null = null;
  if (!config.loadMaps && !config.oracle) {
    pm = await PatchMatchGPU.create(ctx, { patchRadius: pmOpts.patchRadius, patchStep: pmOpts.patchStep });
    stage('pmCreate');
    if (config.selfTest !== false) {
      selfTest = await pm.selfTest();
      log(`[dense-gt] PatchMatchGPU selfTest: ${JSON.stringify(selfTest)}`);
      stage('pmSelfTest');
    }
  }
  const vstats: ViewStats[] = views.map((v, k) => {
    const s = sel.get(v.id)!;
    const g = gtMaps[k].depth;
    const zs: number[] = [];
    for (let i = 0; i < g.length; i += 7) if (g[i] > 0) zs.push(g[i]);
    zs.sort((a, b) => a - b);
    return {
      frame: v.id,
      sources: s.sources,
      scores: s.scores.map((x) => Math.round(x * 10) / 10),
      depthMin: s.depthMin,
      depthMax: s.depthMax,
      gtDepthRange: [zs[Math.floor(0.01 * zs.length)] ?? 0, zs[Math.floor(0.99 * zs.length)] ?? 0],
      photo: { ms: 0, timings: null, validPx: 0 },
      fusedPx: 0,
    };
  });

  const runPass = async (
    label: string,
    vs: DenseView[],
    opts: Partial<PatchMatchOptions>,
    geoIn: ((k: number) => { prior: DepthMap; sourceDepths: DepthMap[] }) | null,
    out: Map<number, DepthMap>,
    put: (k: number, p: PassStats) => void,
  ): Promise<void> => {
    const tP = performance.now();
    for (let k = 0; k < vs.length; k++) {
      const v = vs[k];
      const s = sel.get(v.id)!;
      const srcs = s.sources.map((id) => vs.find((x) => x.id === id)!);
      const t0 = performance.now();
      const dm = await pm!.compute(v, srcs, { min: s.depthMin, max: s.depthMax }, opts, geoIn ? geoIn(k) : undefined);
      const ms = performance.now() - t0;
      out.set(v.id, dm);
      const lt = pm!.lastTimings;
      const ps: PassStats = { ms, timings: lt ? { ...lt, iterationsMs: [...lt.iterationsMs] } : null, validPx: validFraction(dm) };
      put(k, ps);
      log(`[dense-gt] ${label} view ${k + 1}/${vs.length} (frame ${v.id}, ${srcs.length} src, z ${s.depthMin.toFixed(2)}–${s.depthMax.toFixed(2)}): ${ms.toFixed(0)} ms, valid ${(100 * ps.validPx).toFixed(1)} %`);
    }
    T[label] = (T[label] ?? 0) + performance.now() - tP;
    tStage = performance.now();
  };

  const photo = new Map<number, DepthMap>();
  let half: Map<number, DepthMap> | null = null;
  let geoMaps: Map<number, DepthMap> | null = null;
  if (config.oracle) {
    views.forEach((v, k) => {
      const g = gtMaps[k];
      const cost = new Float32Array(g.depth.length);
      for (let i = 0; i < cost.length; i++) cost[i] = g.depth[i] > 0 ? 0 : 2;
      const dm: DepthMap = { width: g.width, height: g.height, depth: g.depth, normal: g.normal, cost };
      photo.set(v.id, dm);
      vstats[k].photo = { ms: 0, timings: null, validPx: validFraction(dm) };
    });
    log('[dense-gt] oracle: fusing the GT surfel depth maps');
  } else if (config.loadMaps) {
    const load = async (kind: string): Promise<Map<number, DepthMap> | null> => {
      const m = new Map<number, DepthMap>();
      for (let k = 0; k < views.length; k++) {
        const r = await fetch(join(config.loadMaps!, `${kind}-${String(views[k].id).padStart(6, '0')}.f32`));
        // (the Vite dev server answers missing files with index.html and HTTP 200)
        if (!r.ok || (r.headers.get('content-type') ?? '').includes('html')) return null;
        const buf = await r.arrayBuffer();
        if (buf.byteLength % 4 !== 0) return null;
        const dm = parseMap(buf);
        m.set(views[k].id, dm);
        const ps: PassStats = { ms: 0, timings: null, validPx: validFraction(dm) };
        if (kind === 'photo') vstats[k].photo = ps;
        else if (kind === 'geo') vstats[k].geo = ps;
        else vstats[k].half = ps;
      }
      return m;
    };
    const ph = await load('photo');
    if (!ph) throw new Error(`no photometric maps under ${config.loadMaps}`);
    for (const [id, dm] of ph) photo.set(id, dm);
    half = await load('half');
    if (config.geometric !== false) geoMaps = await load('geo');
    stage('loadMaps');
    log(`[dense-gt] loaded maps from ${config.loadMaps} (half ${!!half}, geo ${!!geoMaps})`);
  } else if (config.multiScale) {
    const hviews: DenseView[] = [];
    for (const v of views) {
      const hs = Math.max(cam.width, cam.height) >> 1;
      // re-undistort from the original frame for an exact half-resolution pinhole view
      const img = await decodeJPEG(join(base, frames.find((f) => f.index === v.id)!.file));
      const u = undistortImage(img, cam0, hs);
      hviews.push({ id: v.id, cam: u.cam, pose: v.pose, image: u.image });
    }
    stage('halfPrep');
    half = new Map();
    await runPass('photoHalf', hviews, pmOpts, null, half, (k, p) => (vstats[k].half = p));
    const fullIt = config.fullResIterations ?? 2;
    await runPass('photo', views, { ...pmOpts, geometric: false, iterations: fullIt }, (k) => ({ prior: half!.get(views[k].id)!, sourceDepths: [] }), photo, (k, p) => (vstats[k].photo = p));
  } else {
    await runPass('photo', views, pmOpts, null, photo, (k, p) => (vstats[k].photo = p));
  }
  if (!config.loadMaps && !config.oracle && config.geometric !== false) {
    const gOpts: Partial<PatchMatchOptions> = { ...pmOpts, ...(config.geo ?? {}), geometric: true };
    // pass k reads the maps of pass k − 1 (prior + sources); the photometric maps for the first
    const passes = Math.max(1, Math.floor(config.geometricPasses ?? 1));
    let prev: Map<number, DepthMap> = photo;
    for (let gp = 0; gp < passes; gp++) {
      const cur = new Map<number, DepthMap>();
      const src = prev;
      await runPass(
        'geo',
        views,
        gOpts,
        (k) => {
          const s = sel.get(views[k].id)!;
          return { prior: src.get(views[k].id)!, sourceDepths: s.sources.map((id) => src.get(id)!) };
        },
        cur,
        (k, p) => {
          if (gp === 0) vstats[k].geo = p;
          else {
            vstats[k].geo!.ms += p.ms;
            vstats[k].geo!.validPx = p.validPx;
          }
        },
      );
      prev = cur;
    }
    geoMaps = prev;
  }
  pm?.destroy();
  const store = new MemoryDepthMapStore();
  const fusionRaw = config.fusionInput === 'photo' || !geoMaps ? photo : geoMaps;
  let fusionSrc = fusionRaw;
  if (config.preHalo) {
    const m = new Map<number, DepthMap>();
    for (const v of views) m.set(v.id, removeDepthHalos(fusionSrc.get(v.id)!, v.image, v.cam));
    fusionSrc = m;
    stage('preHalo');
  }
  for (const [id, dm] of fusionSrc) await store.put(id, dm);

  // ---- fusion ------------------------------------------------------------------------
  let fstats: FusionStats | null = null;
  const filtered = new MemoryDepthMapStore();
  const neighbours = neighboursFromSelection(sel);
  let lastP = 0;
  let cloud = await fuseDepthMaps(
    views,
    store,
    neighbours,
    { ...(config.fusion ?? {}), filteredStore: filtered, onStats: (s) => (fstats = s) },
    (e) => {
      const now = performance.now();
      if (now - lastP > 5000) {
        lastP = now;
        log(`[dense-gt] fusion: ${e.message ?? ''}`);
      }
    },
  );
  stage('fusion');
  log(`[dense-gt] fusion: ${cloud.count} points, ${JSON.stringify(fstats)}`);
  if (config.filter) {
    const n0 = cloud.count;
    cloud = filterPointCloud(cloud, config.filter);
    stage('filter');
    log(`[dense-gt] filterPointCloud ${JSON.stringify(config.filter)}: ${n0} → ${cloud.count} points`);
  }

  // ---- TSDF --------------------------------------------------------------------------
  let mesh: TriangleMesh | null = null;
  let tstats: TsdfStats | null = null;
  if (config.tsdf) {
    const topts: TsdfOptions = typeof config.tsdf === 'object' ? config.tsdf : {};
    mesh = await tsdfMesh(views, filtered, { ...topts, onStats: (s) => (tstats = s) });
    stage('tsdf');
    log(`[dense-gt] TSDF: ${mesh.triangleCount} triangles, voxel ${(tstats as TsdfStats | null)?.voxelSize}`);
  }

  // ---- per-view evaluation ------------------------------------------------------------
  const debugFrames = new Set<number>();
  const withExact = frames.filter((f) => f.depthFile).map((f) => f.index);
  if (Array.isArray(config.debugViews)) config.debugViews.forEach((i) => debugFrames.add(i));
  else {
    const n = config.debugViews ?? 3;
    const pool = withExact.length ? withExact : frameIdx;
    for (let k = 0; k < Math.min(n, pool.length); k++) debugFrames.add(pool[Math.floor(((k + 0.5) * pool.length) / Math.min(n, pool.length))]);
  }
  const debug: { frame: number; png: Blob }[] = [];
  for (let k = 0; k < views.length; k++) {
    const v = views[k];
    const fdm = await filtered.get(v.id);
    vstats[k].fusedPx = fdm ? validFraction(fdm) : 0;
    if (!doEval) continue;
    const gt = gtMaps[k];
    const mask = surfaceMask(v.cam, gt.depth, gt.normal);
    const pdm = photo.get(v.id)!;
    const gdm = geoMaps?.get(v.id);
    const hdm = half?.get(v.id);
    const empty = { depth: new Float32Array(gt.depth.length), normal: new Float32Array(3 * gt.depth.length) };
    vstats[k].eval = {
      photo: evalBoth(pdm, gt.depth, gt.normal, tau, mask),
      geo: gdm ? evalBoth(gdm, gt.depth, gt.normal, tau, mask) : undefined,
      fused: evalBoth(fdm ?? empty, gt.depth, gt.normal, tau, mask),
    };
    const hopt = config.fusion?.haloFilter || (config.preHalo ? {} : null);
    let haloMask: Uint32Array | null = null;
    if (hopt) {
      const src = fusionRaw.get(v.id)!;
      haloMask = depthHaloMask(src, v.image, v.cam, hopt);
      const hc: HaloCounts = { all: { noGt: 0, bad: 0, mid: 0, good: 0 }, rejected: { noGt: 0, bad: 0, mid: 0, good: 0 } };
      for (let i = 0; i < src.depth.length; i++) {
        const z = src.depth[i];
        if (!(z > 0)) continue;
        const zg = gt.depth[i];
        const cls: keyof HaloCounts['all'] = !(zg > 0) ? 'noGt' : Math.abs(z - zg) > 0.05 * zg ? 'bad' : Math.abs(z - zg) > 0.01 * zg ? 'mid' : 'good';
        hc.all[cls]++;
        if (haloMask && (haloMask[i >>> 5] >>> (i & 31)) & 1) hc.rejected[cls]++;
      }
      vstats[k].halo = hc;
    }
    if (config.consistencyCoverage !== false) {
      // this view as the only fusion reference: all its multi-view-consistent pixels
      const only = new Map<number, number[]>([[v.id, neighbours.get(v.id) ?? []]]);
      const fs = new MemoryDepthMapStore();
      await fuseDepthMaps(views, store, only, { ...(config.fusion ?? {}), filteredStore: fs });
      const cdm = await fs.get(v.id);
      vstats[k].eval!.consistent = evalBoth(cdm ?? empty, gt.depth, gt.normal, tau, mask);
    }
    if (hdm) {
      // nearest-upsampled half map vs full-resolution GT
      const up = upsampleNearest(hdm, v.cam.width, v.cam.height);
      vstats[k].eval!.half = evalBoth(up, gt.depth, gt.normal, tau, mask);
    }
    const f = frames[k];
    let exactDepth: Float32Array | null = null;
    if (f.depthFile) {
      const raw = await fetchTyped(join(base, f.depthFile), Float32Array);
      exactDepth = undistortGtDepth(raw, cam0, v.cam);
      const en = normalsFromDepth(v.cam, exactDepth);
      const emask = surfaceMask(v.cam, exactDepth, en);
      vstats[k].exact = {
        surfelVsExact: evalBoth({ depth: gt.depth, normal: gt.normal }, exactDepth, en, tau, emask),
        photo: evalBoth(pdm, exactDepth, en, tau, emask),
        geo: gdm ? evalBoth(gdm, exactDepth, en, tau, emask) : undefined,
        fused: evalBoth(fdm ?? empty, exactDepth, en, tau, emask),
      };
    }
    if (debugFrames.has(v.id)) {
      const W = v.cam.width;
      const H = v.cam.height;
      const zr = vstats[k].gtDepthRange;
      const z0 = Math.max(1e-3, zr[0] * 0.9);
      const z1 = Math.max(z0 * 1.01, zr[1] * 1.1);
      const ref = exactDepth ?? gt.depth;
      const tiles = [
        { img: v.image, label: `frame ${v.id} (undistorted)` },
        { img: depthToRGBA(ref, W, H, z0, z1), label: exactDepth ? 'GT depth (exact)' : 'GT depth (surfel)' },
        { img: depthToRGBA(pdm.depth, W, H, z0, z1), label: config.multiScale ? 'photometric (prior-init)' : 'photometric depth' },
        { img: errorToRGBA(pdm.depth, ref, W, H), label: 'photometric error' },
        { img: costToRGBA(pdm.cost, pdm.depth, W, H), label: 'photometric NCC cost (white = 0)' },
        { img: gdm ? depthToRGBA(gdm.depth, W, H, z0, z1) : depthToRGBA(empty.depth, W, H, 1, 2), label: 'geometric depth' },
        { img: gdm ? errorToRGBA(gdm.depth, ref, W, H) : depthToRGBA(empty.depth, W, H, 1, 2), label: 'geometric error' },
        { img: errorToRGBA((fdm ?? empty).depth, ref, W, H), label: 'fused pixels: error' },
      ];
      if (haloMask) {
        const src = fusionRaw.get(v.id)!;
        const img = errorToRGBA(src.depth, ref, W, H);
        for (let i = 0; i < W * H; i++) {
          if (!((haloMask[i >>> 5] >>> (i & 31)) & 1)) continue;
          img.data[4 * i] = 255;
          img.data[4 * i + 1] = 0;
          img.data[4 * i + 2] = 255;
        }
        tiles.push({ img, label: 'fusion input, magenta = halo-rejected' });
      }
      debug.push({ frame: v.id, png: await panelPNG(tiles, 4) });
    }
  }
  stage('eval');

  // ---- subset visibility of the completeness set ----------------------------------------
  const compSet = gtFocus ?? gtBase;
  let compVis: Uint8Array | null = null;
  if (config.subsetVisibility !== false) {
    const keepSave = compSet.keep;
    compSet.keep = null;
    compVis = subsetVisibility(compSet, views, gtMaps, { radiusFactor: kR });
    compSet.keep = keepSave;
    stage('subsetVisibility');
  }

  const agg: Record<string, Record<string, number>> = {};
  if (doEval) {
    agg.photoAll = aggregate(vstats, (v) => v.eval?.photo.all);
    agg.photoSurface = aggregate(vstats, (v) => v.eval?.photo.surface);
    if (geoMaps) {
      agg.geoAll = aggregate(vstats, (v) => v.eval?.geo?.all);
      agg.geoSurface = aggregate(vstats, (v) => v.eval?.geo?.surface);
    }
    if (half) agg.halfAll = aggregate(vstats, (v) => v.eval?.half?.all);
    agg.fusedAll = aggregate(vstats, (v) => v.eval?.fused.all);
    agg.fusedSurface = aggregate(vstats, (v) => v.eval?.fused.surface);
    agg.consistentAll = aggregate(vstats, (v) => v.eval?.consistent?.all);
    agg.consistentSurface = aggregate(vstats, (v) => v.eval?.consistent?.surface);
    agg.exactSurfelVsExactAll = aggregate(vstats, (v) => v.exact?.surfelVsExact.all);
    agg.exactSurfelVsExactSurface = aggregate(vstats, (v) => v.exact?.surfelVsExact.surface);
    agg.exactPhotoAll = aggregate(vstats, (v) => v.exact?.photo.all);
    agg.exactGeoAll = aggregate(vstats, (v) => v.exact?.geo?.all);
    agg.exactGeoSurface = aggregate(vstats, (v) => v.exact?.geo?.surface);
    agg.exactFusedAll = aggregate(vstats, (v) => v.exact?.fused.all);
  }
  let haloSum: HaloCounts | null = null;
  for (const v of vstats) {
    if (!v.halo) continue;
    haloSum ??= { all: { noGt: 0, bad: 0, mid: 0, good: 0 }, rejected: { noGt: 0, bad: 0, mid: 0, good: 0 } };
    for (const g of ['all', 'rejected'] as const) for (const c of ['noGt', 'bad', 'mid', 'good'] as const) haloSum[g][c] += v.halo[g][c];
  }
  T.total = performance.now() - tAll;

  const result: DenseGTResult = {
    config,
    dataset: { name: meta.name, scene: meta.scene, numFrames: allFrames.length, width: meta.width, height: meta.height, denseTau: tau, sceneExtent: meta.sceneExtent },
    frames: frameIdx,
    cam,
    gpu: { description: `${ctx.info.vendor}/${ctx.info.architecture}/${ctx.info.description}`, isSoftware: ctx.isSoftware, selfTest },
    sparse: { drawn: sparse.drawn, kept: sparse.kept, meanTrackLength: sparse.meanTrackLength },
    views: vstats,
    fusion: fstats,
    tsdf: tstats,
    timings: T,
    aggregate: agg,
    cloud,
    mesh,
    completenessSet: compSet.name,
    completenessVisibility: compVis,
    halo: haloSum,
    _views: views,
    _debug: debug,
    _maps: config.saveMaps ? { half, photo, geo: geoMaps } : null,
  };
  window.denseGTLast = result;
  log(`[dense-gt] done in ${(T.total / 1000).toFixed(1)} s`);
  return result;
}

/** Depth map file: Float32 [W, H, depth (W·H), normal (3·W·H), cost (W·H)]. */
function serializeMap(dm: DepthMap): Float32Array {
  const n = dm.width * dm.height;
  const out = new Float32Array(2 + 5 * n);
  out[0] = dm.width;
  out[1] = dm.height;
  out.set(dm.depth, 2);
  out.set(dm.normal, 2 + n);
  out.set(dm.cost, 2 + 4 * n);
  return out;
}

function parseMap(buf: ArrayBuffer): DepthMap {
  const a = new Float32Array(buf);
  const W = a[0];
  const H = a[1];
  const n = W * H;
  if (a.length !== 2 + 5 * n) throw new Error('bad depth map file');
  return { width: W, height: H, depth: a.slice(2, 2 + n), normal: a.slice(2 + n, 2 + 4 * n), cost: a.slice(2 + 4 * n, 2 + 5 * n) };
}

function upsampleNearest(dm: DepthMap, W: number, H: number): { depth: Float32Array; normal: Float32Array } {
  const depth = new Float32Array(W * H);
  const normal = new Float32Array(3 * W * H);
  const sx = dm.width / W;
  const sy = dm.height / H;
  for (let y = 0; y < H; y++) {
    const yy = Math.min(dm.height - 1, Math.max(0, Math.round((y + 0.5) * sy - 0.5)));
    for (let x = 0; x < W; x++) {
      const xx = Math.min(dm.width - 1, Math.max(0, Math.round((x + 0.5) * sx - 0.5)));
      const s = yy * dm.width + xx;
      const d = y * W + x;
      depth[d] = dm.depth[s];
      normal[3 * d] = dm.normal[3 * s];
      normal[3 * d + 1] = dm.normal[3 * s + 1];
      normal[3 * d + 2] = dm.normal[3 * s + 2];
    }
  }
  return { depth, normal };
}

function summary(r: DenseGTResult | undefined = window.denseGTLast): unknown {
  if (!r) return null;
  const { cloud, mesh, completenessVisibility, _views, _debug, _maps, ...rest } = r;
  return {
    ...rest,
    cloud: { count: cloud.count },
    mesh: mesh ? { vertexCount: mesh.vertexCount, triangleCount: mesh.triangleCount } : null,
    completenessVisibility: completenessVisibility ? { length: completenessVisibility.length } : null,
    debugFrames: _debug.map((d) => d.frame),
  };
}

async function writeOutputs(prefix: string): Promise<string[]> {
  const r = window.denseGTLast;
  if (!r) throw new Error('no result: call runDenseGT first');
  const written: string[] = [];
  const put = async (path: string, data: Blob | ArrayBuffer | Uint8Array): Promise<void> => {
    await writeBinary(path, data);
    written.push(path);
  };
  await put(`${prefix}.ply`, encodePLY(r.cloud, { comments: [`dense-gt ${r.dataset.name} frames ${r.frames.join(',')}`] }));
  if (r.mesh) await put(`${prefix}.mesh.ply`, encodeMeshPLY(r.mesh));
  if (r.completenessVisibility) await put(`${prefix}.compvis.u8`, r.completenessVisibility);
  if (r._maps) {
    const sets: [string, Map<number, DepthMap> | null][] = [['half', r._maps.half], ['photo', r._maps.photo], ['geo', r._maps.geo]];
    for (const [kind, m] of sets) {
      if (!m) continue;
      for (const [id, dm] of m) await put(`${prefix}/maps/${kind}-${String(id).padStart(6, '0')}.f32`, new Uint8Array(serializeMap(dm).buffer));
    }
  }
  for (const d of r._debug) await put(`${prefix}/debug/view-${String(d.frame).padStart(6, '0')}.png`, d.png);
  return written;
}

// ------------------------------------------------------------------ rendering

export interface RenderShot {
  name: string;
  /** 'fit': viewer default framing; 'camera': from a dataset frame's pose; 'orbit': around a world target. */
  kind: 'fit' | 'camera' | 'orbit';
  frame?: number;
  target?: number[];
  azimuthDeg?: number;
  elevationDeg?: number;
  distance?: number;
  fovDeg?: number;
  /** Show the mesh instead of the points. */
  mesh?: boolean;
  cameras?: boolean;
  width?: number;
  height?: number;
}

export interface RenderOptions {
  prefix: string;
  shots: RenderShot[];
}

/** world (Z-up) → three (Y-up): (x, y, z) → (x, z, −y). */
function w2t(p: ArrayLike<number>): [number, number, number] {
  return [p[0], p[2], -p[1]];
}

async function nextFrame(): Promise<void> {
  await new Promise<void>((r) => requestAnimationFrame(() => r()));
}

async function render(opts: RenderOptions): Promise<string[]> {
  const r = window.denseGTLast;
  if (!r) throw new Error('no result: call runDenseGT first');
  const host = document.createElement('div');
  host.style.cssText = 'position:fixed;left:0;top:0;width:960px;height:720px;';
  document.body.appendChild(host);
  const viewer = new Viewer(host, { maxPixelRatio: 1, autoFit: 'off', antialias: false });
  const written: string[] = [];
  try {
    viewer.setPointCloud(r.cloud);
    viewer.setCameras(r._views.map((v) => ({ pose: v.pose, intrinsics: v.cam })), false);
    viewer.setDisplay({ grid: false });
    for (const s of opts.shots) {
      const W = s.width ?? 960;
      const H = s.height ?? 720;
      host.style.width = `${W}px`;
      host.style.height = `${H}px`;
      viewer.resize();
      viewer.setDisplay({ points: !s.mesh, mesh: !!s.mesh, cameras: s.cameras ?? s.kind !== 'camera' });
      if (s.mesh) viewer.setMesh(r.mesh);
      const cam3 = viewer.camera;
      cam3.fov = s.fovDeg ?? 50;
      cam3.up.set(0, 1, 0);
      if (s.kind === 'fit') {
        cam3.updateProjectionMatrix();
        viewer.fitToContent(false);
      } else if (s.kind === 'camera') {
        const v = r._views.find((x) => x.id === s.frame) ?? r._views[0];
        const R = v.pose.R;
        const t = v.pose.t;
        const C = [-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])];
        const fwd = [R[6], R[7], R[8]];
        const up = [-R[3], -R[4], -R[5]];
        const d = Math.max(0.1, r.dataset.sceneExtent * 0.2);
        const tgt = [C[0] + d * fwd[0], C[1] + d * fwd[1], C[2] + d * fwd[2]];
        cam3.fov = (2 * Math.atan(v.cam.height / 2 / v.cam.f) * 180) / Math.PI;
        cam3.position.set(...w2t(C));
        cam3.up.set(...w2t(up));
        viewer.controls.target.set(...w2t(tgt));
        cam3.lookAt(viewer.controls.target);
      } else {
        const tg = s.target ?? [0, 0, 0];
        const az = ((s.azimuthDeg ?? 0) * Math.PI) / 180;
        const el = ((s.elevationDeg ?? 30) * Math.PI) / 180;
        const dist = s.distance ?? 1;
        const C = [tg[0] + dist * Math.cos(el) * Math.cos(az), tg[1] + dist * Math.cos(el) * Math.sin(az), tg[2] + dist * Math.sin(el)];
        cam3.up.set(0, 1, 0);
        cam3.position.set(...w2t(C));
        viewer.controls.target.set(...w2t(tg));
        cam3.lookAt(viewer.controls.target);
      }
      cam3.aspect = W / H;
      cam3.updateProjectionMatrix();
      await nextFrame();
      await nextFrame();
      const blob = await viewer.screenshot();
      const path = `${opts.prefix}/render-${s.name}.png`;
      await writeBinary(path, blob);
      written.push(path);
      if (s.mesh) viewer.setMesh(null);
    }
  } finally {
    viewer.dispose();
    host.remove();
  }
  return written;
}

window.runDenseGT = runDenseGT;
window.denseGTSummary = summary;
window.denseGTWrite = writeOutputs;
window.denseGTRender = render;
window.denseGTReady = true;
if (statusEl) statusEl.textContent = 'ready: call runDenseGT({ dataset: "object" | "yard", ... })';
