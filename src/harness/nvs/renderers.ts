/**
 * Renderers of the novel-view evaluation (nvs.html, e2e/nvs.mjs): each draws the reconstruction at
 * an exact pinhole test camera and returns RGBA8 (rows top to bottom, PREMULTIPLIED alpha, alpha 0 =
 * nothing drawn; e2e/lib/nvs-metrics.mjs composites over the background and scores).
 *
 *  - 'points' / 'points-edl': the app's Viewer (src/viewer) drawing dense.ply as the viewer screen
 *    does (splats sized by the per-point radius; Eye-Dome Lighting off / on).
 *  - 'mesh-lit' / 'mesh-unlit': the Viewer drawing mesh.ply (vertex colours with the viewer's
 *    hemisphere + headlight lighting, or unlit — the app default: the vertex colours are photo
 *    colours that already contain the shading).
 *  - 'splat': a 3D Gaussian Splatting model (splats.ply, INRIA 3DGS PLY layout), rendered by the
 *    splat module when it exists — see `loadSplatRenderer` for the contract.
 */
import type { PerspectiveCamera } from 'three';
import type { PointCloud, TriangleMesh } from '../../core/types';
import { Viewer, type DisplayOptions } from '../../viewer';
import { flipRows, pinholePerspective, type NvsCamera } from './pinhole';

export type { NvsCamera } from './pinhole';

export interface NvsRenderer {
  readonly name: string;
  /** RGBA8, width × height of the camera, rows top to bottom, premultiplied alpha. */
  render(cam: NvsCamera): Promise<Uint8Array> | Uint8Array;
  dispose?(): void;
}

export const VIEWER_RENDERERS: Record<string, DisplayOptions & { source: 'points' | 'mesh' }> = {
  points: { source: 'points', points: true, mesh: false, edl: false, cameras: false, grid: false },
  'points-edl': { source: 'points', points: true, mesh: false, edl: true, cameras: false, grid: false },
  'mesh-lit': { source: 'mesh', points: false, mesh: true, unlit: false, cameras: false, grid: false },
  'mesh-unlit': { source: 'mesh', points: false, mesh: true, unlit: true, cameras: false, grid: false },
};

/** Apply a pinhole camera to a three PerspectiveCamera (projection only; the pose is the viewer's). */
export function applyPinhole(camera: PerspectiveCamera, cam: NvsCamera): void {
  const p = pinholePerspective(cam);
  camera.fov = p.fovDeg;
  camera.aspect = p.aspect;
  camera.zoom = 1;
  camera.setViewOffset(cam.width, cam.height, p.offsetX, p.offsetY, cam.width, cam.height);
  camera.updateProjectionMatrix();
}

/**
 * One Viewer (the app's) holding the dense cloud and the mesh, drawn at test cameras with the
 * display options of a renderer. The host element is sized to the camera image in CSS px and the
 * pixel ratio is capped at 1, so the drawing buffer is exactly width × height.
 */
export class ViewerRenderers {
  readonly viewer: Viewer;
  private readonly host: HTMLDivElement;
  private width: number;
  private height: number;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.host = document.createElement('div');
    this.host.style.cssText = `position:fixed;left:0;top:0;width:${width}px;height:${height}px`;
    document.body.appendChild(this.host);
    this.viewer = new Viewer(this.host, { maxPixelRatio: 1, autoFit: 'off', background: 'transparent' });
  }

  setContent(points: PointCloud | null, mesh: TriangleMesh | null): void {
    // as src/app/screens/viewer.ts: the dense cloud with its radii, the mesh with vertex colours
    this.viewer.setPointCloud(points);
    this.viewer.setMesh(mesh);
    this.viewer.setCameras([], false);
  }

  private resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    this.host.style.width = `${width}px`;
    this.host.style.height = `${height}px`;
    this.viewer.resize();
  }

  render(name: string, cam: NvsCamera): Uint8Array {
    const d = VIEWER_RENDERERS[name];
    if (!d) throw new Error(`unknown viewer renderer ${name}`);
    this.resize(cam.width, cam.height);
    const { source: _source, ...display } = d;
    this.viewer.setDisplay(display);
    // pose first (it may set the fov), then the exact projection
    this.viewer.setViewPose({ R: Float64Array.from(cam.R), t: Float64Array.from(cam.t) });
    applyPinhole(this.viewer.camera, cam);
    const px = this.viewer.readPixels();
    if (px.width !== cam.width || px.height !== cam.height) throw new Error(`drawing buffer ${px.width}×${px.height}, expected ${cam.width}×${cam.height} (devicePixelRatio?)`);
    return flipRows(px.data, px.width, px.height);
  }

  /** Unmasked WebGL renderer string (e.g. the ANGLE back-end and GPU), or ''. */
  glRenderer(): string {
    const gl = this.viewer.renderer.getContext();
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : '';
  }

  renderer(name: string): NvsRenderer {
    return { name, render: (cam) => this.render(name, cam) };
  }

  dispose(): void {
    this.viewer.dispose();
    this.host.remove();
  }
}

/**
 * SPLAT RENDERER HOOK. The splat stage (src/splat, built separately) plugs in by providing
 * `src/splat/nvs.ts` with
 *
 *   export async function createNvsRenderer(input: { ply: ArrayBuffer; width: number; height: number }): Promise<NvsRenderer>
 *
 * where `ply` is the run's splats.ply (INRIA 3DGS layout: x y z nx ny nz f_dc_0..2 f_rest_*
 * opacity scale_0..2 rot_0..3, in the run's world frame) and `render(cam)` returns RGBA8 rows top
 * to bottom with premultiplied alpha (accumulated opacity), at cam.width × cam.height (the test
 * camera's pinhole fx, fy, cx, cy and world-to-camera R, t). Without it, an adapter tries the
 * module's public API as drafted (`src/splat/index.ts`: `SplatRenderer.create(gpu, opts)`,
 * `setModel`, `render(cam, { background })` → RGBA float with the transmittance in A, and
 * `decodeSplatPLY`). Neither ⇒ the 'splat' renderer is skipped with a reason.
 * `import.meta.glob` resolves to an empty map when a file does not exist, so this compiles and runs
 * without src/splat.
 */
const SPLAT_MODULES = import.meta.glob(['../../splat/nvs.ts', '../../splat/index.ts']);

type NvsFactory = (input: { ply: ArrayBuffer; width: number; height: number }) => Promise<NvsRenderer>;

interface SplatModuleAPI {
  SplatRenderer?: {
    create(gpu: unknown, opts: { shDegree?: number; antialiased?: boolean; maxWidth: number; maxHeight: number }): Promise<{
      setModel(model: unknown): void;
      render(cam: NvsCamera, o?: { background?: ArrayLike<number> }): Promise<{ width: number; height: number; data: Float32Array }>;
      destroy?(): void;
    }>;
  };
  decodeSplatPLY?: (data: Uint8Array, maxDegree?: number) => { model: { shDegree: number }; antialiased: boolean };
}

/** NvsRenderer over the drafted src/splat API (see the hook comment); null when it does not match. */
async function adaptSplatModule(mod: SplatModuleAPI, ply: ArrayBuffer, width: number, height: number): Promise<NvsRenderer | null> {
  if (!mod.SplatRenderer?.create || !mod.decodeSplatPLY) return null;
  const { getGPU } = await import('../../gpu/device');
  const gpu = await getGPU();
  if (!gpu) throw new Error('no WebGPU device');
  const { model, antialiased } = mod.decodeSplatPLY(new Uint8Array(ply), 3);
  const r = await mod.SplatRenderer.create(gpu, { shDegree: model.shDegree, antialiased, maxWidth: width, maxHeight: height });
  r.setModel(model);
  return {
    name: 'splat',
    async render(cam) {
      // black background ⇒ colour = premultiplied radiance; alpha = 1 − transmittance
      const img = await r.render(cam, { background: [0, 0, 0] });
      const n = img.width * img.height;
      const out = new Uint8Array(4 * n);
      const q = (v: number) => Math.max(0, Math.min(255, Math.round(255 * v)));
      for (let i = 0; i < n; i++) {
        out[4 * i] = q(img.data[4 * i]);
        out[4 * i + 1] = q(img.data[4 * i + 1]);
        out[4 * i + 2] = q(img.data[4 * i + 2]);
        out[4 * i + 3] = q(1 - img.data[4 * i + 3]);
      }
      return out;
    },
    dispose: () => r.destroy?.(),
  };
}

export async function loadSplatRenderer(ply: ArrayBuffer, width: number, height: number): Promise<NvsRenderer | { skipped: string }> {
  const nvs = SPLAT_MODULES['../../splat/nvs.ts'];
  if (nvs) {
    const mod = (await nvs()) as { createNvsRenderer?: NvsFactory };
    if (typeof mod.createNvsRenderer !== 'function') return { skipped: 'src/splat/nvs.ts does not export createNvsRenderer' };
    const r = await mod.createNvsRenderer({ ply, width, height });
    return { name: 'splat', render: (cam) => r.render(cam), dispose: () => r.dispose?.() };
  }
  const index = SPLAT_MODULES['../../splat/index.ts'];
  if (!index) return { skipped: 'src/splat not found (splat stage not built)' };
  const r = await adaptSplatModule((await index()) as SplatModuleAPI, ply, width, height);
  return r ?? { skipped: 'src/splat has neither nvs.ts (createNvsRenderer) nor SplatRenderer + decodeSplatPLY' };
}
