/**
 * Novel-view evaluation page (nvs.html), driven by e2e/nvs.mjs.
 *
 *   window.nvsSetup(config)  loads the run's dense.ply / mesh.ply / splats.ply (served by the
 *                            runner under config.runBase) into the renderers;
 *   window.nvsView(view)     one held-out test view: the dataset JPEG (config.dataBase + view.file)
 *                            is decoded, undistorted with the run's intrinsics to ≤ maxSize exactly
 *                            like the engine's dense views (undistortImage, then the rolling-shutter
 *                            correction rsCorrectPinholeImage with the view's ω when config.rs is set,
 *                            as denseViewImage does), every renderer draws the reconstruction at the
 *                            undistorted pinhole camera and the pose, and the images are scored
 *                            (e2e/lib/nvs-metrics.mjs). Optional GT | render | diff PNGs go to Node
 *                            through window.__nvsWrite.
 *   window.nvsTeardown()     frees the renderers.
 */
import type { CameraIntrinsics, PointCloud, RGBAImage, TriangleMesh } from '../../core/types';
import { undistortImage, undistortedSize } from '../../dense/undistort';
import { decodePLY } from '../../io/ply';
import { rsCorrectPinholeImage, type RollingShutterModel } from '../../sfm/rolling-shutter';
import { prepareGroundTruth, scoreCommon, scoreView, triptych } from '../../../e2e/lib/nvs-metrics.mjs';
import { VIEWER_RENDERERS, ViewerRenderers, loadSplatRenderer, type NvsCamera, type NvsRenderer } from './renderers';

export interface NvsPageConfig {
  /** URL prefix of the run directory (result files) and of the dataset directory. */
  runBase: string;
  dataBase: string;
  /** Renderer names (VIEWER_RENDERERS keys and 'splat'). */
  renderers: string[];
  /** Long side of the undistorted test images (never upscaled). */
  maxSize: number;
  /** Rolling shutter of the ground-truth correction (null = none). */
  rs: RollingShutterModel | null;
  /** Background the full-frame image is composited over (sRGB 0…255). */
  background: [number, number, number];
  /** Write GT | render | |diff|·3 PNGs. */
  saveImages: boolean;
  /** Compute SSIM (the slow part). */
  ssim: boolean;
  /** Also score every renderer on the pixels all of them cover (scoreCommon). */
  common?: boolean;
  /** Distorted camera of the first view: sizes the viewer before the first render. */
  camera: CameraIntrinsics;
}

export interface NvsPageView {
  index: number;
  file: string;
  /** Run-frame world-to-camera pose. */
  pose: { R: number[]; t: number[] };
  /** Distorted RADIAL intrinsics of the run for this frame (dataset resolution). */
  camera: CameraIntrinsics;
  /** Camera angular velocity (rad/s, camera frame) for the rolling-shutter correction. */
  omega: number[] | null;
  /** Translational term of that correction (camera-frame velocity / depth, 1/s); null = rotation only. */
  nu?: number[] | null;
}

export interface NvsSetupResult {
  renderers: Record<string, { ok: boolean; reason?: string; loadMs?: number }>;
  points: number;
  triangles: number;
  splatFile: boolean;
  /** WebGL renderer string of the viewer (ANGLE back-end / GPU). */
  gl?: string;
}

type ScoreResult = ReturnType<typeof scoreView>;

export interface NvsViewResult {
  index: number;
  /** Undistorted pinhole camera the view was rendered and scored at. */
  camera: { width: number; height: number; f: number; cx: number; cy: number };
  gtMs: number;
  renderMs: Record<string, number>;
  scoreMs: Record<string, number>;
  scores: Record<string, Omit<ScoreResult, 'images'>>;
  errors: Record<string, string>;
  /** With config.common: fraction of the valid pixels every renderer covers. */
  commonCoverage?: number;
}

type WriteFn = (path: string, data: string, encoding: 'base64' | 'utf8') => Promise<void>;

declare global {
  interface Window {
    nvsSetup: (config: NvsPageConfig) => Promise<NvsSetupResult>;
    nvsView: (view: NvsPageView) => Promise<NvsViewResult>;
    nvsTeardown: () => void;
    nvsReady?: boolean;
    __nvsWrite?: WriteFn;
    __nvsLog?: (msg: string) => void;
  }
}

const statusEl = typeof document !== 'undefined' ? document.getElementById('status') : null;

function log(msg: string): void {
  if (statusEl) statusEl.textContent = msg;
  if (window.__nvsLog) void window.__nvsLog(msg);
  else console.log(msg);
}

async function fetchOptional(url: string): Promise<ArrayBuffer | null> {
  const r = await fetch(url);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  return r.arrayBuffer();
}

async function decodeJPEG(url: string): Promise<RGBAImage> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  const bmp = await createImageBitmap(await r.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  const id = ctx.getImageData(0, 0, bmp.width, bmp.height);
  bmp.close();
  return { width: id.width, height: id.height, data: id.data };
}

async function pngBase64(rgba: Uint8ClampedArray, width: number, height: number): Promise<string> {
  const c = new OffscreenCanvas(width, height);
  c.getContext('2d')!.putImageData(new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, width, height), 0, 0);
  const blob = await c.convertToBlob({ type: 'image/png' });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

let cfg: NvsPageConfig | null = null;
let viewers: ViewerRenderers | null = null;
const renderers = new Map<string, NvsRenderer>();

async function setup(config: NvsPageConfig): Promise<NvsSetupResult> {
  teardown();
  cfg = config;
  const out: NvsSetupResult = { renderers: {}, points: 0, triangles: 0, splatFile: false };
  const wantViewer = config.renderers.filter((r) => r in VIEWER_RENDERERS);
  const need = new Set(wantViewer.map((r) => VIEWER_RENDERERS[r].source));
  let points: PointCloud | null = null;
  let mesh: TriangleMesh | null = null;
  // the app's viewer screen loads both when they exist (clip planes / bounds see both)
  if (need.size) {
    const t0 = performance.now();
    const dense = await fetchOptional(`${config.runBase}dense.ply`);
    points = dense ? (decodePLY(dense).points ?? null) : null;
    const meshBuf = await fetchOptional(`${config.runBase}mesh.ply`);
    mesh = meshBuf ? (decodePLY(meshBuf).mesh ?? null) : null;
    out.points = points?.count ?? 0;
    out.triangles = mesh?.triangleCount ?? 0;
    log(`loaded dense ${out.points} points, mesh ${out.triangles} triangles (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
  }
  const und = undistortedSize(config.camera.width, config.camera.height, config.maxSize);
  if (wantViewer.length) {
    viewers = new ViewerRenderers(und.width, und.height);
    viewers.setContent(points, mesh);
    out.gl = viewers.glRenderer();
    for (const name of wantViewer) {
      const src = VIEWER_RENDERERS[name].source;
      if ((src === 'points' && !points) || (src === 'mesh' && !mesh)) {
        out.renderers[name] = { ok: false, reason: `${src === 'points' ? 'dense.ply' : 'mesh.ply'} missing in the run directory` };
        continue;
      }
      renderers.set(name, viewers.renderer(name));
      out.renderers[name] = { ok: true };
    }
  }
  if (config.renderers.includes('splat')) {
    const t0 = performance.now();
    const ply = await fetchOptional(`${config.runBase}splats.ply`);
    out.splatFile = !!ply;
    if (!ply) out.renderers.splat = { ok: false, reason: 'splats.ply missing in the run directory' };
    else {
      try {
        const r = await loadSplatRenderer(ply, und.width, und.height);
        if ('skipped' in r) out.renderers.splat = { ok: false, reason: r.skipped };
        else {
          renderers.set('splat', r);
          out.renderers.splat = { ok: true, loadMs: Math.round(performance.now() - t0) };
        }
      } catch (e) {
        out.renderers.splat = { ok: false, reason: `splat renderer failed to load: ${(e as Error)?.message ?? e}` };
      }
    }
  }
  return out;
}

async function view(v: NvsPageView): Promise<NvsViewResult> {
  if (!cfg) throw new Error('nvsSetup first');
  const t0 = performance.now();
  const rgba = await decodeJPEG(`${cfg.dataBase}${v.file}`);
  // exactly the engine's dense view (src/engine/dense.ts denseViewImage): undistort, then RS-correct
  const und = undistortImage(rgba, v.camera, cfg.maxSize);
  const gt = cfg.rs && v.omega ? rsCorrectPinholeImage(und.image, und.cam, v.omega, cfg.rs, v.nu ?? null) : und.image;
  const { width: W, height: H } = und.cam;
  const cam: NvsCamera = { width: W, height: H, fx: und.cam.f, fy: und.cam.f, cx: und.cam.cx, cy: und.cam.cy, R: v.pose.R, t: v.pose.t };
  const res: NvsViewResult = {
    index: v.index,
    camera: { width: W, height: H, f: und.cam.f, cx: und.cam.cx, cy: und.cam.cy },
    gtMs: Math.round(performance.now() - t0),
    renderMs: {},
    scoreMs: {},
    scores: {},
    errors: {},
  };
  const gtData = gt.data instanceof Uint8ClampedArray ? gt.data : new Uint8ClampedArray(gt.data.buffer, gt.data.byteOffset, gt.data.length);
  const prepared = prepareGroundTruth(gtData, W, H);
  const images: Record<string, Uint8Array> = {};
  for (const [name, r] of renderers) {
    try {
      const t1 = performance.now();
      const img = await r.render(cam);
      if (img.length !== 4 * W * H) throw new Error(`renderer returned ${img.length} bytes, expected ${4 * W * H}`);
      if (cfg.common) images[name] = img.slice();
      const t2 = performance.now();
      const s = scoreView(img, prepared, W, H, { background: cfg.background, ssim: cfg.ssim, keepImages: cfg.saveImages });
      res.renderMs[name] = Math.round(10 * (t2 - t1)) / 10;
      res.scoreMs[name] = Math.round(performance.now() - t2);
      if (s.images && window.__nvsWrite) {
        const panel = triptych(gtData, s.images.full, W, H);
        await window.__nvsWrite(`view-${String(v.index).padStart(4, '0')}-${name}.png`, await pngBase64(panel, 3 * W, H), 'base64');
      }
      delete s.images;
      res.scores[name] = s;
    } catch (e) {
      res.errors[name] = (e as Error)?.message ?? String(e);
    }
  }
  const scored = Object.keys(images).filter((name) => res.scores[name]);
  if (cfg.common && scored.length >= 2) {
    const c = scoreCommon(Object.fromEntries(scored.map((name) => [name, images[name]])), prepared, W, H, { ssim: cfg.ssim });
    res.commonCoverage = c.coverage;
    for (const name of scored) res.scores[name].common = { coverage: c.coverage, ...c.scores[name] };
  }
  log(`view ${v.index}: ${Object.entries(res.scores).map(([n, s]) => `${n} ${s.full.psnr.toFixed(2)} dB`).join(', ')}`);
  return res;
}

function teardown(): void {
  for (const r of renderers.values()) if (r.name === 'splat') r.dispose?.();
  renderers.clear();
  viewers?.dispose();
  viewers = null;
  cfg = null;
}

window.nvsSetup = setup;
window.nvsView = view;
window.nvsTeardown = teardown;
window.nvsReady = true;
log('ready');
