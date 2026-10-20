/**
 * three.js (WebGL2) viewer for reconstructions: dense point clouds as round
 * splats with Eye-Dome Lighting, live sparse points, meshes, camera frustums and
 * trajectory, ground grid, touch navigation and a measure tool.
 *
 * Frames: content is given in the WORLD frame of the reconstruction (Z-up when
 * gravity aligned, docs/ARCHITECTURE.md §2). Everything lives under a "world
 * root" whose rotation maps world up to three's +Y ('z' mode: −90° about X, i.e.
 * (x, y, z) → (x, z, −y); 'auto' mode: estimated up from the camera poses).
 *
 * Rendering is on demand: a frame is drawn only when content, display options,
 * size or the camera change (including OrbitControls damping); no idle loop.
 */
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  DirectionalLight,
  DoubleSide,
  GridHelper,
  Group,
  HemisphereLight,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PerspectiveCamera,
  Points,
  Quaternion,
  Scene,
  Sphere,
  Vector2,
  Vector3,
  WebGLRenderer,
  type LineBasicMaterial,
  type Material,
} from 'three';
import type { PointCloud, Pose, TriangleMesh, Vec3 } from '../core/types';
import { CameraOverlay, type ViewerCamera } from './cameras';
import { createOrbitControls, TapDetector, TargetAnimator, type OrbitControls, type TapInfo } from './controls';
import { EDLPass, ZOOM_SIZE } from './edl';
import {
  boundsOfPoints,
  boundsSphere,
  cameraCenter,
  clampedFullBounds,
  estimateUpVector,
  goldenStridePermutation,
  linearizeDepth,
  niceStep,
  surfaceDepthAt,
  percentileAlong,
  pickNearestProjected,
  raycastTriangles,
  robustBoundsOfSamples,
  subsamplePositions,
  unionBounds,
  type Bounds3,
  type PickSplatModel,
  type RobustBounds,
} from './helpers';
import { MeasureTool, measureDistance, type MeasureCallback } from './measure';
import { PointCloudMaterial, patchMeshMaterial } from './points-material';

export interface ViewerOptions {
  /** CSS colour of the background, or 'transparent'. Default '#101216'. */
  background?: string;
  /** Cap for devicePixelRatio (default 2). */
  maxPixelRatio?: number;
  /** MSAA on the canvas (overlays / direct rendering). Default true. */
  antialias?: boolean;
  /**
   * Dense clouds larger than this draw an evenly spread subset of this many
   * points while the camera moves (full cloud when idle). 0 disables. Default 1.5 M.
   */
  interactivePointBudget?: number;
  /** Picking strategy: GPU depth readback (with CPU fallback) or CPU only. Default 'auto'. */
  pickMode?: 'auto' | 'cpu';
  /**
   * 'first' (default): frame the content when it first appears.
   * 'follow': re-frame (animated) on every content update until the user moves the view.
   * 'off': never automatically.
   */
  autoFit?: 'first' | 'follow' | 'off';
  /** Vertical field of view in degrees (default 50). */
  fovDeg?: number;
}

export interface PointCloudDisplayOptions {
  /** Point size in CSS px when not sized by radius (default 2). */
  pointSizePx?: number;
  /** Size points by their `radii` (world units) when available (default true when radii exist). */
  splats?: boolean;
  /** Fake sphere shading of the round splats (default false). */
  softShading?: boolean;
  /** Hide points whose `confidence` is below this value (needs pc.confidence). */
  minConfidence?: number;
}

export interface DisplayOptions {
  points?: boolean;
  mesh?: boolean;
  cameras?: boolean;
  edl?: boolean;
  grid?: boolean;
  /** Draw the mesh with its vertex colours only (no lighting). */
  unlit?: boolean;
  /** EDL strength (default 1, Potree scale). */
  edlStrength?: number;
  /** EDL neighbour radius in CSS px (default 1.4). */
  edlRadiusPx?: number;
  /** Size of the live sparse points in CSS px (default 3). */
  sparsePointSizePx?: number;
}

export interface MeasureResult {
  distance: number;
  a: Vec3;
  b: Vec3;
}

export interface ViewerStats {
  frames: number;
  edlSupported: boolean;
  /** Picking actually used: 'gpu' (depth readback) or 'cpu'. */
  pickMode: 'gpu' | 'cpu';
  near: number;
  far: number;
  pixelRatio: number;
  /** The last frame drew the interactive subset of a huge dense cloud. */
  lodActive: boolean;
  densePoints: number;
  sparsePoints: number;
  triangles: number;
  cameras: number;
}

/** Screen position in CSS px relative to the container's top-left: {x, y} or [x, y]. */
export type ScreenPoint = { x: number; y: number } | ArrayLike<number>;

interface Layer {
  samples: Float32Array; // world-frame subsample for bounds / grid
  bounds: RobustBounds | null;
}

interface DenseLayer extends Layer {
  source: PointCloud;
  count: number;
  geometry: BufferGeometry;
  points: Points;
  lodGeometry: BufferGeometry | null;
  lodPoints: Points | null;
  lodSizeScale: number;
}

interface MeshLayer extends Layer {
  source: TriangleMesh;
  geometry: BufferGeometry;
  mesh: Mesh;
}

const BOUNDS_SAMPLES = 40000;
const PICK_RADIUS_CSS = 14;
/** Magnification of the sub-pixel pick pass (64² target ⇒ ±8 device px around the tap). */
const PICK_ZOOM = 4;
const Y_UP = new Vector3(0, 1, 0);

const _v = new Vector3();
const _w = new Vector3();
const _v2 = new Vector2();
const _m = new Matrix4();
const _q = new Quaternion();

/** Linear view depth → window depth in [0, 1] (perspective, glDepthRange [0, 1]). */
function viewDepthToWindow(z: number, near: number, far: number): number {
  return (far / (far - near)) * (1 - near / z);
}

function toXY(p: ScreenPoint): { x: number; y: number } {
  const o = p as { x?: unknown; y?: unknown };
  if (typeof o.x === 'number' && typeof o.y === 'number') return { x: o.x, y: o.y };
  const a = p as ArrayLike<number>;
  return { x: a[0], y: a[1] };
}

export class Viewer {
  readonly container: HTMLElement;
  readonly renderer: WebGLRenderer;
  readonly camera: PerspectiveCamera;
  readonly controls: OrbitControls;

  private readonly canvas: HTMLCanvasElement;
  private readonly opts: Required<Omit<ViewerOptions, 'background'>> & { background: string };
  private readonly bgColor = new Color();
  private readonly bgAlpha: number;
  private readonly contentScene = new Scene();
  private readonly overlayScene = new Scene();
  /** World → three transform (content). */
  private readonly contentRoot = new Group();
  /** World → three transform (overlays), kept identical to contentRoot. */
  private readonly overlayRoot = new Group();
  /** Drawn first, never over content (ground grid). */
  private readonly backgroundScene = new Scene();
  private readonly backgroundRoot = new Group();
  private readonly hemiLight = new HemisphereLight(0xffffff, 0x666666, 2.0);
  private readonly dirLight = new DirectionalLight(0xffffff, 1.4);
  private readonly edl: EDLPass;
  private readonly cameraOverlay = new CameraOverlay();
  private readonly measure: MeasureTool;
  private readonly tap: TapDetector;
  private readonly targetAnim: TargetAnimator;
  private readonly resizeObserver: ResizeObserver | null;
  /** Matches the current devicePixelRatio: a change (window moved to another screen) resizes. */
  private dprQuery: MediaQueryList | null = null;
  private readonly maxPointSize: number;
  /** Original inline `position` of the container when the viewer made it relative. */
  private readonly restorePosition: string | null;
  /** fitToContent() requested while the container had no size: run it on the first real resize. */
  private pendingFit = false;

  private dense: DenseLayer | null = null;
  private denseMaterial = new PointCloudMaterial({ sizePx: 2 });
  private densePointSizeCss = 2;
  private sparse: (Layer & { geometry: BufferGeometry; points: Points; capacity: number; count: number }) | null = null;
  private readonly sparseMaterial = new PointCloudMaterial({ sizePx: 3, round: true });
  private meshLayer: MeshLayer | null = null;
  private readonly meshLit = new MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.95,
    metalness: 0,
    side: DoubleSide,
  });
  private readonly meshUnlit = new MeshBasicMaterial({ vertexColors: true, side: DoubleSide });
  private readonly meshLitPlain = new MeshStandardMaterial({ color: 0xc8c8c8, roughness: 0.95, metalness: 0, side: DoubleSide });
  private readonly meshUnlitPlain = new MeshBasicMaterial({ color: 0xc8c8c8, side: DoubleSide });
  /** Shared uniform: mesh output alpha (0 while rendering into the EDL target). */
  private readonly edlWeight = { value: 1 };
  private cams: ViewerCamera[] = [];
  private camCenters: Float64Array[] = [];
  private highlightLast = true;
  private grid: GridHelper | null = null;
  private gridKey = '';

  private display: Required<DisplayOptions> = {
    points: true,
    mesh: true,
    cameras: true,
    edl: true,
    grid: true,
    unlit: false,
    edlStrength: 1,
    edlRadiusPx: 1.4,
    sparsePointSizePx: 3,
  };
  private worldUpMode: 'z' | 'auto' = 'z';
  private upWorld = new Float64Array([0, 0, 1]);
  private metric = false;

  private width = 0; // CSS px
  private height = 0;
  private pixelRatio = 1;
  /** Forces resize() to re-apply the size (first call, context restore). */
  private sizeDirty = true;
  private rafId = 0;
  private disposed = false;
  private contextLost = false;
  private interacting = false;
  private frames = 0;
  private boundsDirty = true;
  private clipSphere: Sphere | null = null;
  private contentBounds: RobustBounds | null = null;
  private autoFitArmed = true;
  /** setViewPose(): the camera is placed from outside, the orbit controls are off. */
  private viewLocked = false;
  private autoFitFirst = true;
  /** The view direction was picked by a fit before any camera was known, and the view has not been moved since. */
  private dirWithoutCams = false;
  private viewAnim: { fromPos: Vector3; toPos: Vector3; fromTarget: Vector3; toTarget: Vector3; t0: number } | null = null;
  private gridActive = false;
  private lastPickMode: 'gpu' | 'cpu' = 'gpu';
  private lastFrameLod = false;
  /** Bumped whenever what the content target would show changes (content, display, size). */
  private contentVersion = 0;
  /** View signature of the last full-quality render into the EDL target (reused by picking). */
  private readonly targetSig = new Float64Array(36).fill(NaN);
  private readonly curSig = new Float64Array(36);
  private pickU = new Float64Array(0);
  private pickV = new Float64Array(0);
  private pickZ = new Float64Array(0);
  private pickD2 = new Float64Array(0);
  private readonly listeners: [EventTarget, string, EventListener][] = [];

  constructor(container: HTMLElement, opts: ViewerOptions = {}) {
    this.container = container;
    this.opts = {
      background: opts.background ?? '#101216',
      maxPixelRatio: opts.maxPixelRatio ?? 2,
      antialias: opts.antialias ?? true,
      interactivePointBudget: opts.interactivePointBudget ?? 1_500_000,
      pickMode: opts.pickMode ?? 'auto',
      autoFit: opts.autoFit ?? 'first',
      fovDeg: opts.fovDeg ?? 50,
    };
    const transparent = this.opts.background === 'transparent';
    this.bgAlpha = transparent ? 0 : 1;
    if (!transparent) this.bgColor.set(this.opts.background);

    const prevPosition = container.style.position;
    const madeRelative = getComputedStyle(container).position === 'static';
    if (madeRelative) container.style.position = 'relative';
    this.restorePosition = madeRelative ? prevPosition : null;
    const canvas = document.createElement('canvas');
    Object.assign(canvas.style, {
      position: 'absolute',
      left: '0',
      top: '0',
      width: '100%',
      height: '100%',
      display: 'block',
      touchAction: 'none',
      outline: 'none',
      userSelect: 'none',
      webkitUserSelect: 'none',
    } satisfies Partial<CSSStyleDeclaration>);
    // iOS: no long-press callout / tap flash on the canvas.
    canvas.style.setProperty('-webkit-touch-callout', 'none');
    canvas.style.setProperty('-webkit-tap-highlight-color', 'transparent');
    container.appendChild(canvas);
    this.canvas = canvas;

    try {
      this.renderer = new WebGLRenderer({
        canvas,
        antialias: this.opts.antialias,
        alpha: transparent,
        powerPreference: 'high-performance',
        preserveDrawingBuffer: false,
        depth: true,
        stencil: false,
      });
    } catch (e) {
      // No WebGL (Lockdown Mode, too many contexts, GPU process crash): leave the
      // container as it was so the app can show a message instead.
      canvas.remove();
      if (madeRelative) container.style.position = prevPosition;
      throw e;
    }
    this.renderer.autoClear = false;
    const gl = this.renderer.getContext();
    const range = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) as Float32Array | number[] | null;
    this.maxPointSize = range && range[1] ? Number(range[1]) : 64;

    this.camera = new PerspectiveCamera(this.opts.fovDeg, 1, 0.01, 1000);
    this.camera.position.set(0, 2, 5);
    this.camera.lookAt(0, 0, 0);

    this.contentScene.add(this.contentRoot, this.hemiLight, this.dirLight, this.dirLight.target);
    this.overlayScene.add(this.overlayRoot);
    this.backgroundScene.add(this.backgroundRoot);
    this.overlayRoot.add(this.cameraOverlay.object);
    this.measure = new MeasureTool(container);
    this.overlayRoot.add(this.measure.object);
    for (const m of [this.meshLit, this.meshUnlit, this.meshLitPlain, this.meshUnlitPlain]) patchMeshMaterial(m, this.edlWeight);
    this.applyRootRotation(this.rootQuaternionFor([0, 0, 1]), false);

    this.edl = new EDLPass(this.renderer);

    this.controls = createOrbitControls(this.camera, canvas);
    this.controls.addEventListener('change', this.onControlsChange);
    this.controls.addEventListener('start', this.onControlsStart);
    this.controls.addEventListener('end', this.onControlsEnd);
    this.targetAnim = new TargetAnimator(this.controls);
    this.tap = new TapDetector(canvas, { onTap: this.onTap, onDoubleTap: this.onDoubleTap });

    this.listen(canvas, 'webglcontextlost', ((e: Event) => {
      e.preventDefault();
      this.contextLost = true;
      this.cancelFrame();
    }) as EventListener);
    this.listen(canvas, 'webglcontextrestored', (() => {
      this.contextLost = false;
      this.edl.onContextRestored();
      this.sizeDirty = true;
      this.resize();
      // Resume animations / damping that were cut off by the loss.
      this.requestRender();
    }) as EventListener);
    // iOS Safari: keep pinch gestures on the canvas from zooming the page
    // (belt and braces with touch-action: none).
    const noGesture = ((e: Event) => e.preventDefault()) as EventListener;
    this.listen(canvas, 'gesturestart', noGesture);
    this.listen(canvas, 'gesturechange', noGesture);

    this.resizeObserver = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => this.resize()) : null;
    this.resizeObserver?.observe(container);
    if (!this.resizeObserver) this.listen(window, 'resize', (() => this.resize()) as EventListener);
    // the container's CSS size need not change with the pixel ratio: no ResizeObserver callback
    this.watchDpr();
    this.resize();
  }

  // ---------------------------------------------------------------- content

  /** Dense point cloud (arrays are used directly as GPU attribute sources, not copied). */
  setPointCloud(pc: PointCloud | null, opts: PointCloudDisplayOptions = {}): void {
    if (this.disposed) return;
    const radii = pc?.radii && pc.radii.length >= pc.count ? pc.radii : undefined;
    const conf = pc?.confidence && pc.confidence.length >= pc.count ? pc.confidence : undefined;
    const useRadius = !!radii && (opts.splats ?? true);
    const useConf = !!conf && (opts.minConfidence ?? 0) > 0;
    if (opts.pointSizePx !== undefined) this.densePointSizeCss = opts.pointSizePx;

    // Same positions / colours ⇒ keep the GPU buffers (option changes, new radii /
    // confidence). In-place edits of the arrays are NOT detected: pass new arrays.
    let same =
      !!pc &&
      !!this.dense &&
      this.dense.source.positions === pc.positions &&
      this.dense.source.colors === pc.colors &&
      this.dense.source.count === pc.count;
    if (same && this.dense) {
      // A new per-point array of another byte size cannot be uploaded into the
      // existing GL buffer (three cannot resize, and a replaced attribute's buffer
      // would leak): rebuild the layer.
      for (const [name, arr] of [
        ['radius', useRadius ? radii : undefined],
        ['confidence', useConf ? conf : undefined],
      ] as const) {
        const cur = this.dense.geometry.getAttribute(name) as BufferAttribute | undefined;
        if (cur && arr && cur.array !== arr && cur.array.byteLength !== arr.byteLength) same = false;
      }
    }
    if (!same) {
      this.disposeDense();
      if (pc && pc.count > 0) this.dense = this.buildDense(pc);
      this.boundsDirty = true;
    }
    if (this.dense && pc) {
      this.dense.source = pc;
      if (useRadius && radii) this.syncDenseAttribute('radius', radii);
      if (useConf && conf) this.syncDenseAttribute('confidence', conf);
    }
    this.denseMaterial.setOptions({
      useRadius,
      useConfidence: useConf,
      minConfidence: opts.minConfidence ?? 0,
      soft: opts.softShading ?? false,
      round: true,
    });
    this.onContentChanged();
  }

  private buildDense(pc: PointCloud): DenseLayer {
    const n = Math.min(pc.count, Math.floor(pc.positions.length / 3), Math.floor(pc.colors.length / 3));
    const geometry = new BufferGeometry();
    const pos = new BufferAttribute(pc.positions, 3);
    const col = new BufferAttribute(pc.colors, 3, true);
    geometry.setAttribute('position', pos);
    geometry.setAttribute('color', col);
    geometry.setDrawRange(0, n);
    const samples = subsamplePositions(pc.positions, n, BOUNDS_SAMPLES);
    const bounds = robustBoundsOfSamples(samples, n);
    if (bounds) {
      const s = boundsSphere(bounds.full);
      geometry.boundingSphere = new Sphere(new Vector3(s.center[0], s.center[1], s.center[2]), s.radius);
    }
    const points = new Points(geometry, this.denseMaterial);
    points.frustumCulled = false;
    points.name = 'dense-points';
    this.contentRoot.add(points);
    let lodGeometry: BufferGeometry | null = null;
    let lodPoints: Points | null = null;
    let lodSizeScale = 1;
    const budget = this.opts.interactivePointBudget;
    if (budget > 0 && n > budget * 1.2) {
      lodGeometry = new BufferGeometry();
      lodGeometry.setAttribute('position', pos);
      lodGeometry.setAttribute('color', col);
      lodGeometry.setIndex(new BufferAttribute(goldenStridePermutation(n, budget), 1));
      lodGeometry.boundingSphere = geometry.boundingSphere;
      lodPoints = new Points(lodGeometry, this.denseMaterial);
      lodPoints.frustumCulled = false;
      lodPoints.visible = false;
      lodPoints.name = 'dense-points-lod';
      this.contentRoot.add(lodPoints);
      lodSizeScale = Math.min(2, Math.sqrt(n / budget));
    }
    return { source: pc, count: n, geometry, points, lodGeometry, lodPoints, lodSizeScale, samples, bounds };
  }

  /** Attach (or replace the values of) a per-point float attribute of the dense layer. */
  private syncDenseAttribute(name: 'radius' | 'confidence', arr: Float32Array): void {
    const d = this.dense!;
    const cur = d.geometry.getAttribute(name) as BufferAttribute | undefined;
    if (cur) {
      if (cur.array === arr) return;
      // Same byte size (guaranteed by setPointCloud): re-upload into the existing buffer.
      cur.array = arr;
      cur.needsUpdate = true;
      return;
    }
    const a = new BufferAttribute(arr, 1);
    d.geometry.setAttribute(name, a);
    d.lodGeometry?.setAttribute(name, a);
  }

  private disposeDense(): void {
    const d = this.dense;
    if (!d) return;
    this.contentRoot.remove(d.points);
    d.geometry.dispose();
    if (d.lodPoints) this.contentRoot.remove(d.lodPoints);
    d.lodGeometry?.dispose();
    this.dense = null;
  }

  /** Live sparse points (copied into reusable GPU buffers; cheap fixed-size round points). */
  setSparsePoints(positions: Float32Array, colors: Uint8Array): void {
    if (this.disposed) return;
    const n = Math.min(Math.floor(positions.length / 3), Math.floor(colors.length / 3));
    if (n === 0) {
      if (this.sparse) {
        this.contentRoot.remove(this.sparse.points);
        this.sparse.geometry.dispose();
        this.sparse = null;
      }
      this.boundsDirty = true;
      this.onContentChanged();
      return;
    }
    let s = this.sparse;
    if (!s || s.capacity < n) {
      if (s) {
        this.contentRoot.remove(s.points);
        s.geometry.dispose();
      }
      const capacity = Math.max(1024, Math.ceil(n * 1.5));
      const geometry = new BufferGeometry();
      geometry.setAttribute('position', new BufferAttribute(new Float32Array(3 * capacity), 3));
      geometry.setAttribute('color', new BufferAttribute(new Uint8Array(3 * capacity), 3, true));
      const points = new Points(geometry, this.sparseMaterial);
      points.frustumCulled = false;
      points.name = 'sparse-points';
      this.contentRoot.add(points);
      s = { geometry, points, capacity, count: 0, samples: new Float32Array(0), bounds: null };
      this.sparse = s;
    }
    const pa = s.geometry.getAttribute('position') as BufferAttribute;
    const ca = s.geometry.getAttribute('color') as BufferAttribute;
    (pa.array as Float32Array).set(positions.subarray(0, 3 * n));
    (ca.array as Uint8Array).set(colors.subarray(0, 3 * n));
    pa.clearUpdateRanges();
    ca.clearUpdateRanges();
    pa.addUpdateRange(0, 3 * n);
    ca.addUpdateRange(0, 3 * n);
    pa.needsUpdate = true;
    ca.needsUpdate = true;
    s.geometry.setDrawRange(0, n);
    s.count = n;
    s.samples = subsamplePositions(positions, n, 20000);
    s.bounds = robustBoundsOfSamples(s.samples, n);
    this.boundsDirty = true;
    this.onContentChanged();
  }

  /** Triangle mesh with sRGB vertex colours (lit with hemisphere + directional light, or unlit). */
  setMesh(mesh: TriangleMesh | null): void {
    if (this.disposed) return;
    if (this.meshLayer) {
      this.contentRoot.remove(this.meshLayer.mesh);
      this.meshLayer.geometry.dispose();
      this.meshLayer = null;
    }
    if (mesh && mesh.vertexCount > 0 && mesh.triangleCount > 0) {
      const g = new BufferGeometry();
      g.setAttribute('position', new BufferAttribute(mesh.positions, 3));
      if (mesh.normals && mesh.normals.length >= 3 * mesh.vertexCount) g.setAttribute('normal', new BufferAttribute(mesh.normals, 3));
      if (mesh.colors && mesh.colors.length >= 3 * mesh.vertexCount) g.setAttribute('color', new BufferAttribute(mesh.colors, 3, true));
      g.setIndex(new BufferAttribute(mesh.indices, 1));
      g.setDrawRange(0, 3 * mesh.triangleCount);
      if (!g.getAttribute('normal')) g.computeVertexNormals();
      const samples = subsamplePositions(mesh.positions, mesh.vertexCount, BOUNDS_SAMPLES);
      const bounds = robustBoundsOfSamples(samples, mesh.vertexCount);
      if (bounds) {
        const s = boundsSphere(bounds.full);
        g.boundingSphere = new Sphere(new Vector3(s.center[0], s.center[1], s.center[2]), s.radius);
      }
      const m = new Mesh(g);
      m.frustumCulled = false;
      m.name = 'mesh';
      m.material = this.meshMaterial(!!g.getAttribute('color'));
      this.contentRoot.add(m);
      this.meshLayer = { source: mesh, geometry: g, mesh: m, samples, bounds };
    }
    this.boundsDirty = true;
    this.onContentChanged();
  }

  /** Camera frustums (≈3 % of the scene radius) + trajectory polyline; last one highlighted. */
  setCameras(cams: ViewerCamera[], highlightLast = true): void {
    if (this.disposed) return;
    this.cams = cams.slice();
    this.highlightLast = highlightLast;
    this.camCenters = this.cams.map((c) => cameraCenter(c.pose));
    if (this.worldUpMode === 'auto') this.updateAutoUp();
    this.boundsDirty = true;
    this.onContentChanged();
  }

  /** 'z': world is gravity aligned (Z up). 'auto': estimate up from the camera poses (and their `upCam`). */
  setWorldUp(up: 'z' | 'auto'): void {
    if (this.disposed) return;
    this.worldUpMode = up;
    if (up === 'z') {
      this.upWorld = new Float64Array([0, 0, 1]);
      this.applyRootRotation(this.rootQuaternionFor(this.upWorld), true);
    } else {
      this.updateAutoUp(true);
    }
    this.boundsDirty = true;
    this.requestRender();
  }

  /** World up currently used (unit, world frame). */
  getWorldUp(): Vec3 {
    return this.upWorld.slice();
  }

  private updateAutoUp(force = false): void {
    const up = estimateUpVector(this.cams.map((c) => ({ R: c.pose.R, upCam: c.upCam }))) ?? new Float64Array([0, 0, 1]);
    const dot = up[0] * this.upWorld[0] + up[1] * this.upWorld[1] + up[2] * this.upWorld[2];
    if (!force && dot > Math.cos((1 * Math.PI) / 180)) return; // < 1° change: keep the view stable
    this.upWorld = Float64Array.from(up);
    this.applyRootRotation(this.rootQuaternionFor(up), true);
  }

  private rootQuaternionFor(up: ArrayLike<number>): Quaternion {
    _v.set(up[0], up[1], up[2]).normalize();
    return new Quaternion().setFromUnitVectors(_v, Y_UP);
  }

  /** Set the world→three rotation; optionally carry the camera along so the view of the content is kept (levelled). */
  private applyRootRotation(q: Quaternion, carryCamera: boolean): void {
    if (carryCamera) {
      _q.copy(this.contentRoot.quaternion).invert().premultiply(q); // delta = q · q_old⁻¹
      this.camera.position.applyQuaternion(_q);
      this.controls?.target.applyQuaternion(_q);
      // locked: the orbit controls do not re-aim the camera — turn it with the content
      if (this.viewLocked) this.camera.quaternion.premultiply(_q);
      else this.controls?.update();
    }
    for (const root of [this.contentRoot, this.overlayRoot, this.backgroundRoot]) {
      root.quaternion.copy(q);
      root.updateMatrixWorld(true);
    }
    this.contentVersion++;
  }

  setDisplay(opts: DisplayOptions): void {
    if (this.disposed) return;
    const d = this.display;
    for (const k of Object.keys(opts) as (keyof DisplayOptions)[]) {
      const v = opts[k];
      if (v !== undefined) (d as Record<string, unknown>)[k] = v;
    }
    if (this.meshLayer) this.meshLayer.mesh.material = this.meshMaterial(!!this.meshLayer.geometry.getAttribute('color'));
    this.contentVersion++;
    this.requestRender();
  }

  private meshMaterial(vertexColors: boolean): Material {
    if (vertexColors) return this.display.unlit ? this.meshUnlit : this.meshLit;
    return this.display.unlit ? this.meshUnlitPlain : this.meshLitPlain;
  }

  getDisplay(): Required<DisplayOptions> {
    return { ...this.display };
  }

  setUnits(metric: boolean): void {
    this.metric = metric;
    this.measure.setUnits(metric);
    this.requestRender();
  }

  // ---------------------------------------------------------------- bounds / framing

  private onContentChanged(): void {
    this.contentVersion++;
    const realContent = !!(this.dense || this.sparse || this.meshLayer);
    const anyContent = realContent || this.cams.length > 0;
    if (!anyContent) {
      this.autoFitArmed = true;
      this.autoFitFirst = true;
      this.dirWithoutCams = false;
    } else if (this.autoFitArmed && this.opts.autoFit !== 'off') {
      const follow = this.opts.autoFit === 'follow';
      this.fitToContent(follow && !this.autoFitFirst);
      // 'first': stay armed until real content (points / mesh) has been framed.
      // 'follow': stay armed until the user moves the view (onControlsStart).
      this.autoFitArmed = follow || !realContent;
    }
    this.requestRender();
  }

  private recomputeBounds(): void {
    if (!this.boundsDirty) return;
    this.boundsDirty = false;
    const layers: Layer[] = [];
    if (this.dense) layers.push(this.dense);
    if (this.sparse) layers.push(this.sparse);
    if (this.meshLayer) layers.push(this.meshLayer);
    // Robust content bounds: union of per-layer robust boxes.
    let robust: Bounds3 | null = null;
    let clip: Bounds3 | null = null;
    let count = 0;
    for (const l of layers) {
      if (!l.bounds) continue;
      robust = unionBounds(robust, l.bounds);
      clip = unionBounds(clip, clampedFullBounds(l.bounds));
      count += l.bounds.count;
    }
    const camBox = boundsOfPoints(this.camCenters);
    if (!robust && camBox) {
      robust = camBox;
      count = this.camCenters.length;
    }
    if (robust) {
      const s = boundsSphere(robust);
      const center = s.center;
      let radius = s.radius;
      if (radius < 1e-6 && camBox) radius = Math.max(radius, boundsSphere(camBox).radius, 1e-3);
      if (radius < 1e-6) radius = 1;
      this.contentBounds = {
        min: robust.min,
        max: robust.max,
        center,
        radius,
        full: clip ?? robust,
        count,
      };
    } else {
      this.contentBounds = null;
    }
    // Frustum size: 3 % of the radius of content ∪ camera centres.
    const sceneBox = unionBounds(robust, camBox);
    const sceneRadius = sceneBox ? Math.max(boundsSphere(sceneBox).radius, 1e-3) : 1;
    this.cameraOverlay.update(this.cams, 0.03 * sceneRadius, this.highlightLast);

    this.updateGrid(layers);

    // Clip sphere (three space) from clamped full bounds ∪ cameras ∪ grid.
    let clipBox = unionBounds(clip, camBox);
    if (camBox) {
      const pad = 0.06 * sceneRadius;
      clipBox = unionBounds(clipBox, { min: camBox.min.map((v) => v - pad), max: camBox.max.map((v) => v + pad) });
    }
    if (this.grid && this.gridActive) {
      const half = (this.grid.userData.size as number) / 2;
      const p = this.grid.position;
      clipBox = unionBounds(clipBox, {
        min: new Float64Array([p.x - half, p.y - half, p.z]),
        max: new Float64Array([p.x + half, p.y + half, p.z]),
      });
    }
    if (clipBox) {
      const s = boundsSphere(clipBox);
      _v.set(s.center[0], s.center[1], s.center[2]);
      this.contentRoot.localToWorld(_v);
      // Degenerate content (all points coincide, no grid / cameras) would give a
      // zero-thickness near/far range that clips everything: keep a minimum radius.
      const minRadius = 0.05 * (this.contentBounds?.radius ?? sceneRadius);
      this.clipSphere = new Sphere(_v.clone(), Math.max(s.radius * 1.02, minRadius));
    } else {
      this.clipSphere = null;
    }
  }

  private updateGrid(layers: Layer[]): void {
    const show = this.worldUpMode === 'z' && layers.some((l) => l.bounds) && !!this.contentBounds;
    this.gridActive = show;
    if (!show) return;
    // Height: 2nd percentile of z over all content samples.
    let total = 0;
    for (const l of layers) total += l.samples.length;
    const all = new Float32Array(total);
    let o = 0;
    for (const l of layers) {
      all.set(l.samples, o);
      o += l.samples.length;
    }
    const h = percentileAlong(all, [0, 0, 1], 2);
    const b = this.contentBounds!;
    const ex = b.max[0] - b.min[0];
    const ey = b.max[1] - b.min[1];
    const extent = Math.max(ex, ey, b.radius * 0.5) * 1.6;
    const step = niceStep(extent / 20);
    const divisions = Math.max(2, 2 * Math.ceil(extent / step / 2));
    const size = divisions * step;
    const key = `${size}|${divisions}`;
    if (!this.grid || this.gridKey !== key) {
      if (this.grid) {
        this.backgroundRoot.remove(this.grid);
        this.grid.geometry.dispose();
        (this.grid.material as LineBasicMaterial).dispose();
      }
      const g = new GridHelper(size, divisions, 0x56606e, 0x2a3039);
      const mat = g.material as LineBasicMaterial;
      mat.transparent = true;
      mat.opacity = 0.55;
      mat.depthWrite = false;
      g.rotation.x = Math.PI / 2; // three-local XZ plane → world XY plane
      g.userData.size = size;
      g.frustumCulled = false;
      g.name = 'grid';
      this.backgroundRoot.add(g);
      this.grid = g;
      this.gridKey = key;
    }
    const cx = Math.round((0.5 * (b.min[0] + b.max[0])) / step) * step;
    const cy = Math.round((0.5 * (b.min[1] + b.max[1])) / step) * step;
    // Drawn before the content (never on top of it), so no z-fighting with the ground.
    this.grid.position.set(cx, cy, Number.isFinite(h) ? h : b.min[2]);
  }

  /** Robust (2–98 %) world bounds of the content, or null. */
  getContentBounds(): RobustBounds | null {
    this.recomputeBounds();
    return this.contentBounds;
  }

  /**
   * Frame the robust bounds of the content (points / mesh; cameras when nothing else). The view
   * direction is kept, except on the first fit (again on the first one with cameras, unless the view
   * was moved since) or with `overview`: then from above, on the side the photos were taken from.
   */
  fitToContent(animate = false, overview = false): void {
    // locked to a camera pose (setViewPose): framing is the caller's, after unlocking
    if (this.disposed || this.viewLocked) return;
    if (this.width === 0 || this.height === 0) {
      // Hidden / not laid out yet: the aspect ratio is unknown. Frame on the first real resize.
      this.pendingFit = true;
      return;
    }
    this.pendingFit = false;
    this.recomputeBounds();
    const b = this.contentBounds;
    if (!b) return;
    const c3 = new Vector3(b.center[0], b.center[1], b.center[2]);
    this.contentRoot.localToWorld(c3);
    const r = b.radius;
    const fovV = (this.camera.fov * Math.PI) / 180;
    const aspect = this.width > 0 && this.height > 0 ? this.width / this.height : this.camera.aspect;
    const fovH = 2 * Math.atan(Math.tan(fovV / 2) * aspect);
    const fmin = Math.min(fovV, fovH);
    const dist = (r / Math.sin(fmin / 2)) * 1.05;

    // Direction: keep the current one unless this is the first fit, or the first with
    // cameras (content usually arrives before them, and so before their 'auto' up) while
    // the view was not moved; prefer the side the photos were taken from when the cameras
    // are off-centre.
    const dir = new Vector3();
    const redoDir = this.dirWithoutCams && this.camCenters.length > 0;
    if (this.frames > 0 && !this.autoFitFirst && !overview && !redoDir) {
      dir.copy(this.camera.position).sub(this.controls.target);
    }
    if (dir.lengthSq() < 1e-20) {
      dir.set(0, 0.55, 1);
      if (this.camCenters.length > 0) {
        const m = new Vector3();
        for (const c of this.camCenters) m.add(_w.set(c[0], c[1], c[2]));
        m.multiplyScalar(1 / this.camCenters.length);
        this.contentRoot.localToWorld(m);
        m.sub(c3);
        m.y = 0;
        if (m.length() > 0.3 * r) dir.copy(m.normalize()).multiplyScalar(Math.cos(0.5)).setY(Math.sin(0.5));
      }
      this.dirWithoutCams = this.camCenters.length === 0;
    }
    this.autoFitFirst = false;
    dir.normalize();
    const pos = c3.clone().addScaledVector(dir, dist);
    this.controls.minDistance = r * 1e-4;
    this.controls.maxDistance = r * 200;
    this.targetAnim.cancel();
    if (animate) {
      this.viewAnim = { fromPos: this.camera.position.clone(), toPos: pos, fromTarget: this.controls.target.clone(), toTarget: c3, t0: performance.now() };
    } else {
      this.viewAnim = null;
      this.camera.position.copy(pos);
      this.controls.target.copy(c3);
      this.controls.update();
    }
    if (this.opts.autoFit === 'follow') this.autoFitArmed = true;
    this.requestRender();
  }

  private stepViewAnim(now: number): boolean {
    const a = this.viewAnim;
    if (!a) return false;
    const u = Math.min(1, Math.max(0, (now - a.t0) / 450));
    const e = 1 - Math.pow(1 - u, 3);
    this.camera.position.lerpVectors(a.fromPos, a.toPos, e);
    this.controls.target.lerpVectors(a.fromTarget, a.toTarget, e);
    if (u >= 1) this.viewAnim = null;
    return this.viewAnim !== null;
  }

  private updateClipPlanes(): void {
    const s = this.clipSphere;
    let near = 0.01;
    let far = 1000;
    if (s) {
      const d = this.camera.position.distanceTo(s.center);
      far = d + s.radius * 1.05;
      const targetDist = Math.max(1e-9, this.camera.position.distanceTo(this.controls.target));
      near = Math.max(d - s.radius * 1.05, Math.min(far / 2000, 0.1 * targetDist));
      near = Math.max(near, far * 1e-6, 1e-9);
    }
    if (Math.abs(near - this.camera.near) > 1e-6 * near || Math.abs(far - this.camera.far) > 1e-6 * far) {
      this.camera.near = near;
      this.camera.far = far;
      this.camera.updateProjectionMatrix();
    }
  }

  // ---------------------------------------------------------------- render loop

  /** Schedule a frame (coalesced; no-op if one is pending). */
  requestRender(): void {
    if (this.disposed || this.contextLost || this.rafId) return;
    this.rafId = requestAnimationFrame(this.frame);
  }

  private cancelFrame(): void {
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = 0;
  }

  private readonly frame = (now: number): void => {
    this.rafId = 0;
    if (this.disposed || this.contextLost) return;
    const anim = this.targetAnim.step(now) || this.stepViewAnim(now);
    const moved = this.viewLocked ? false : this.controls.update(); // applies damping; emits 'change' → requestRender while moving
    this.renderFrame(anim || moved || this.interacting);
    if (anim) this.requestRender();
  };

  /** Synchronous full-quality render (e.g. before reading pixels). */
  renderNow(): void {
    if (this.disposed || this.contextLost) return;
    this.renderFrame(false);
  }

  private renderFrame(lowQuality: boolean): void {
    if (this.width === 0 || this.height === 0) return;
    this.recomputeBounds();
    this.updateClipPlanes();
    this.camera.updateMatrixWorld();
    const r = this.renderer;
    const pr = this.pixelRatio;
    const bufH = this.height * pr;
    const projScale = bufH / (2 * Math.tan((this.camera.fov * Math.PI) / 360));
    const maxSize = Math.min(64 * pr, this.maxPointSize);

    const dense = this.dense;
    if (dense) {
      const lod = lowQuality && !!dense.lodPoints;
      dense.points.visible = this.display.points && !lod;
      if (dense.lodPoints) dense.lodPoints.visible = this.display.points && lod;
      this.denseMaterial.setSizes(this.densePointSizeCss * pr, 1, maxSize);
      this.denseMaterial.updateView(projScale, lod ? dense.lodSizeScale : 1);
      this.lastFrameLod = lod;
    }
    if (this.sparse) {
      this.sparse.points.visible = this.display.points;
      this.sparseMaterial.setSizes(this.display.sparsePointSizePx * pr, 1, maxSize);
      this.sparseMaterial.updateView(projScale, 1);
    }
    if (this.meshLayer) this.meshLayer.mesh.visible = this.display.mesh;
    this.cameraOverlay.object.visible = this.display.cameras;
    if (this.grid) this.grid.visible = this.gridActive && this.display.grid;

    // Headlight from above-left of the viewer.
    const ld = this.camera.position.distanceTo(this.controls.target) + 1e-9;
    this.dirLight.position.copy(this.camera.position).add(_v.set(-0.4, 0.8, 0.4).applyQuaternion(this.camera.quaternion).multiplyScalar(ld));
    this.dirLight.target.position.copy(this.controls.target);
    this.dirLight.target.updateMatrixWorld();

    this.measure.update(this.camera, this.overlayRoot, this.width, this.height);

    const pointsVisible = this.display.points && (!!dense || !!this.sparse);
    const hasContent = pointsVisible || (this.display.mesh && !!this.meshLayer);
    // EDL shades points only: a mesh alone is drawn directly (keeps canvas MSAA).
    const useEdl = this.display.edl && this.edl.supported && pointsVisible;
    if (useEdl) {
      this.renderContentTarget();
      if (!lowQuality || !dense?.lodPoints) this.targetSig.set(this.viewSignature());
      else this.targetSig[0] = NaN;
    }
    r.setRenderTarget(null);
    r.setClearColor(this.bgColor, this.bgAlpha);
    r.clear(true, true, false);
    if (this.grid?.visible) r.render(this.backgroundScene, this.camera);
    if (useEdl) {
      this.edl.composite(r, this.camera.near, this.camera.far, {
        enabled: true,
        strength: this.display.edlStrength,
        radiusPx: this.display.edlRadiusPx * pr,
      });
    } else if (hasContent) {
      r.render(this.contentScene, this.camera);
    }
    r.render(this.overlayScene, this.camera);
    this.frames++;
  }

  /** Camera matrices + content version + size: equal signatures ⇒ identical content target. */
  private viewSignature(): Float64Array {
    const sig = this.curSig;
    sig.set(this.camera.matrixWorld.elements, 0);
    sig.set(this.camera.projectionMatrix.elements, 16);
    sig[32] = this.contentVersion;
    sig[33] = this.width;
    sig[34] = this.height;
    sig[35] = this.pixelRatio;
    return sig;
  }

  private renderContentTarget(): void {
    this.edlWeight.value = 0;
    this.edl.renderContent(this.renderer, this.contentScene, this.camera);
    this.edlWeight.value = 1;
  }

  // ---------------------------------------------------------------- size / lifecycle

  resize(): void {
    if (this.disposed) return;
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    const pr = Math.min(typeof devicePixelRatio === 'number' && devicePixelRatio > 0 ? devicePixelRatio : 1, this.opts.maxPixelRatio);
    if (w === this.width && h === this.height && pr === this.pixelRatio && !this.sizeDirty) return;
    this.sizeDirty = false;
    this.width = w;
    this.height = h;
    if (w === 0 || h === 0) return;
    this.pixelRatio = pr;
    this.renderer.setPixelRatio(this.pixelRatio);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const db = this.renderer.getDrawingBufferSize(_v2);
    this.edl.setSize(db.x, db.y);
    this.contentVersion++;
    if (this.pendingFit) this.fitToContent(false);
    // Resizing clears the canvas: redraw synchronously (no blank frame). A pending
    // frame (e.g. damping) stays scheduled.
    if (!this.contextLost) this.renderFrame(false);
  }

  private readonly onDprChange = (): void => {
    this.watchDpr();
    this.resize();
  };

  /** (Re-)arm the media query for the current devicePixelRatio. */
  private watchDpr(): void {
    this.dprQuery?.removeEventListener?.('change', this.onDprChange);
    this.dprQuery = null;
    if (this.disposed || typeof matchMedia !== 'function' || typeof devicePixelRatio !== 'number' || !(devicePixelRatio > 0)) return;
    this.dprQuery = matchMedia(`(resolution: ${devicePixelRatio}dppx)`);
    this.dprQuery.addEventListener?.('change', this.onDprChange);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.cancelFrame();
    this.resizeObserver?.disconnect();
    this.dprQuery?.removeEventListener?.('change', this.onDprChange);
    this.dprQuery = null;
    for (const [t, type, fn] of this.listeners) t.removeEventListener(type, fn);
    this.listeners.length = 0;
    this.tap.dispose();
    this.controls.removeEventListener('change', this.onControlsChange);
    this.controls.removeEventListener('start', this.onControlsStart);
    this.controls.removeEventListener('end', this.onControlsEnd);
    this.controls.dispose();
    this.disposeDense();
    if (this.sparse) this.sparse.geometry.dispose();
    this.sparse = null;
    if (this.meshLayer) this.meshLayer.geometry.dispose();
    this.meshLayer = null;
    if (this.grid) {
      this.grid.geometry.dispose();
      (this.grid.material as LineBasicMaterial).dispose();
    }
    this.cameraOverlay.dispose();
    this.measure.dispose();
    this.denseMaterial.dispose();
    this.sparseMaterial.dispose();
    this.meshLit.dispose();
    this.meshUnlit.dispose();
    this.meshLitPlain.dispose();
    this.meshUnlitPlain.dispose();
    this.edl.dispose();
    this.renderer.dispose();
    try {
      this.renderer.forceContextLoss();
    } catch {
      /* ignore */
    }
    this.canvas.remove();
    if (this.restorePosition !== null) this.container.style.position = this.restorePosition;
  }

  /** PNG of the current view (full quality; includes the measurement label). */
  screenshot(): Promise<Blob> {
    if (this.disposed) return Promise.reject(new Error('Viewer disposed'));
    // A lost context has no drawing buffer: toBlob would yield a blank image.
    if (this.contextLost || this.renderer.getContext().isContextLost()) {
      return Promise.reject(new Error('WebGL context lost'));
    }
    this.renderFrame(false);
    let src: HTMLCanvasElement = this.canvas;
    const label = this.measure.labelForScreenshot();
    if (label) {
      const c2 = document.createElement('canvas');
      c2.width = this.canvas.width;
      c2.height = this.canvas.height;
      const ctx = c2.getContext('2d');
      if (ctx) {
        ctx.drawImage(this.canvas, 0, 0);
        const s = c2.width / Math.max(1, this.width);
        ctx.font = `600 ${13 * s}px -apple-system, system-ui, sans-serif`;
        const tw = ctx.measureText(label.text).width;
        const padX = 8 * s;
        const hgt = 22 * s;
        const x = label.x * s - tw / 2 - padX;
        const y = label.y * s - hgt * 1.4;
        ctx.fillStyle = 'rgba(12,14,18,0.78)';
        ctx.fillRect(x, y, tw + 2 * padX, hgt);
        ctx.fillStyle = '#fff';
        ctx.textBaseline = 'middle';
        ctx.fillText(label.text, x + padX, y + hgt / 2);
        src = c2;
      }
    }
    return new Promise((resolve, reject) =>
      src.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/png'),
    );
  }

  /**
   * Render now and read back the canvas pixels (RGBA8, rows bottom-to-top as in
   * GL). For tests / thumbnails.
   */
  readPixels(): { width: number; height: number; data: Uint8Array } {
    this.renderNow();
    const gl = this.renderer.getContext();
    const w = gl.drawingBufferWidth;
    const h = gl.drawingBufferHeight;
    const data = new Uint8Array(4 * w * h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, data);
    return { width: w, height: h, data };
  }

  getStats(): ViewerStats {
    return {
      frames: this.frames,
      edlSupported: this.edl.supported,
      pickMode: this.lastPickMode,
      near: this.camera.near,
      far: this.camera.far,
      pixelRatio: this.pixelRatio,
      lodActive: this.lastFrameLod,
      densePoints: this.dense?.count ?? 0,
      sparsePoints: this.sparse?.count ?? 0,
      triangles: this.meshLayer?.source.triangleCount ?? 0,
      cameras: this.cams.length,
    };
  }

  /** True while a frame is scheduled. */
  get framePending(): boolean {
    return this.rafId !== 0;
  }

  // ---------------------------------------------------------------- picking / measuring

  /** World point → CSS px in the container (null if behind the camera). */
  worldToScreen(p: ArrayLike<number>): { x: number; y: number } | null {
    this.recomputeBounds();
    this.updateClipPlanes();
    this.camera.updateMatrixWorld();
    _v.set(p[0], p[1], p[2]);
    this.contentRoot.localToWorld(_v);
    _w.copy(_v).applyMatrix4(this.camera.matrixWorldInverse);
    if (_w.z >= 0) return null;
    _v.project(this.camera);
    return { x: (_v.x * 0.5 + 0.5) * this.width, y: (0.5 - _v.y * 0.5) * this.height };
  }

  /** World point under a screen position (CSS px relative to the container), or null. */
  pickPoint(p: ScreenPoint): Vec3 | null {
    return this.pickPoints([p])[0];
  }

  /**
   * Picks several screen points. GPU path (default):
   *  1. render the content depth (normal splat sizes) and read a ±14 CSS px
   *     window around each tap: the nearest covered pixel gives the visible
   *     ("front") surface depth z_A — occlusion-correct for points and meshes;
   *  2. render a ×4 magnified 64² window around the tap with every point as a
   *     1 px splat: each covered sub-pixel holds the depth of a point whose
   *     centre projects into it, or of the mesh, at ¼ px precision. The hit is placed on the tap ray at the depth of a planar fit
   *     (1/z affine in screen space) to such pixels near the tap that belong to
   *     the front surface (z ≤ 1.05 · pass-A depth at the same pixel),
   *     which removes the bias of screen-aligned splats (flat at their centre's
   *     depth) on oblique surfaces.
   * Cost is independent of the number of points (1 + 1 per tap renders and
   * tiny readbacks, a few ms). CPU fallback: see pickNearestProjected().
   */
  pickPoints(ps: ScreenPoint[]): (Vec3 | null)[] {
    if (this.disposed || this.contextLost || this.width === 0 || this.height === 0) return ps.map(() => null);
    this.recomputeBounds();
    this.updateClipPlanes();
    this.camera.updateMatrixWorld();
    const useGpu = this.opts.pickMode !== 'cpu' && this.edl.supported && this.edl.readbackReliable(this.renderer);
    this.lastPickMode = useGpu ? 'gpu' : 'cpu';
    if (!useGpu) return ps.map((p) => this.pickCPU(toXY(p)));

    // Full-quality content for the current camera.
    const dense = this.dense;
    const pr = this.pixelRatio;
    const projScale = (this.height * pr) / (2 * Math.tan((this.camera.fov * Math.PI) / 360));
    const maxSize = Math.min(64 * pr, this.maxPointSize);
    if (dense) {
      dense.points.visible = this.display.points;
      if (dense.lodPoints) dense.lodPoints.visible = false;
      this.denseMaterial.setSizes(this.densePointSizeCss * pr, 1, maxSize);
      this.denseMaterial.updateView(projScale, 1);
    }
    if (this.sparse) {
      this.sparse.points.visible = this.display.points;
      this.sparseMaterial.setSizes(this.display.sparsePointSizePx * pr, 1, maxSize);
    }
    if (this.meshLayer) this.meshLayer.mesh.visible = this.display.mesh;

    const db = this.renderer.getDrawingBufferSize(_v2);
    const W = db.x;
    const H = db.y;
    const sx = W / this.width;
    const sy = H / this.height;
    const rPx = Math.round(PICK_RADIUS_CSS * pr);
    const taps = ps.map((p0) => {
      const p = toXY(p0);
      return { gx: p.x * sx, gy: H - p.y * sy }; // GL convention (origin bottom-left)
    });

    // Reuse the last displayed frame's content target when nothing changed since
    // (common case: tapping a still view) — saves one full render.
    const sig = this.viewSignature();
    let reuse = this.edl.hasTarget;
    for (let i = 0; i < 36 && reuse; i++) reuse = sig[i] === this.targetSig[i];
    if (!reuse) {
      this.renderContentTarget();
      this.targetSig.set(this.viewSignature());
    }
    const front = taps.map((t) => this.edl.readDepthWindow(this.renderer, t.gx, t.gy, rPx));
    // Pass B: magnified (×PICK_ZOOM) render of ±8 device px around each tap with
    // every point as a 1 px splat → sub-pixel depths (points and mesh).
    const exact: (Float32Array | null)[] = taps.map(() => null);
    const offsets = taps.map((t) => ({
      x: Math.round(t.gx * PICK_ZOOM - ZOOM_SIZE / 2),
      y: Math.round((H - t.gy) * PICK_ZOOM - ZOOM_SIZE / 2), // top-left origin
    }));
    {
      const mats = [this.denseMaterial, this.sparseMaterial];
      const saved = mats.map((m) => [m.uniforms.uMinSize.value as number, m.uniforms.uMaxSize.value as number]);
      for (const m of mats) {
        m.uniforms.uMinSize.value = 1;
        m.uniforms.uMaxSize.value = 1;
      }
      this.edlWeight.value = 0;
      for (let k = 0; k < taps.length; k++) {
        exact[k] = this.edl.readZoomedDepth(
          this.renderer,
          this.contentScene,
          this.camera,
          W * PICK_ZOOM,
          H * PICK_ZOOM,
          offsets[k].x,
          offsets[k].y,
        );
      }
      this.edlWeight.value = 1;
      mats.forEach((m, k) => {
        m.uniforms.uMinSize.value = saved[k][0];
        m.uniforms.uMaxSize.value = saved[k][1];
      });
    }

    const near = this.camera.near;
    const far = this.camera.far;
    const out: (Vec3 | null)[] = [];
    for (let k = 0; k < taps.length; k++) {
      const { gx, gy } = taps[k];
      const win = front[k];
      if (!win) {
        out.push(null);
        continue;
      }
      const rr = (win.size - 1) / 2;
      const cx = gx - win.x0 - 0.5; // tap in window pixel-centre coordinates
      const cy = gy - win.y0 - 0.5;
      const valid = (d: number) => d < 0.9999999 && d > 0;
      // 1. front surface depth: the tapped pixel if covered, else the nearest covered pixel.
      const ti = Math.floor(gx) - win.x0;
      const tj = Math.floor(gy) - win.y0;
      const tapDepth = win.depths[tj * win.size + ti];
      const tapCovered = valid(tapDepth);
      let best = tapCovered ? tj * win.size + ti : -1;
      if (!tapCovered) {
        let bestD2 = Infinity;
        let bestD = 1;
        for (let j = 0; j < win.size; j++) {
          for (let i = 0; i < win.size; i++) {
            const d = win.depths[j * win.size + i];
            if (!valid(d)) continue;
            const d2 = (i - cx) * (i - cx) + (j - cy) * (j - cy);
            if (d2 > rr * rr) continue;
            if (d2 < bestD2 - 1e-9 || (Math.abs(d2 - bestD2) <= 1e-9 && d < bestD)) {
              best = j * win.size + i;
              bestD2 = d2;
              bestD = d;
            }
          }
        }
      }
      if (best < 0) {
        out.push(null);
        continue;
      }
      const bestDepth = win.depths[best];
      // Default: tap ray at the front depth, or the nearest covered pixel (tap in a gap).
      let px = tapCovered ? gx : win.x0 + (best % win.size) + 0.5;
      let py = tapCovered ? gy : win.y0 + Math.floor(best / win.size) + 0.5;
      let depth = bestDepth;
      // 2. refine with the sub-pixel point depths of the front surface around the tap.
      const ex = exact[k];
      if (ex) {
        const ns = ZOOM_SIZE * ZOOM_SIZE;
        if (this.pickU.length < ns) {
          this.pickU = new Float64Array(ns);
          this.pickV = new Float64Array(ns);
          this.pickZ = new Float64Array(ns);
          this.pickD2 = new Float64Array(ns);
        }
        const U = this.pickU;
        const V = this.pickV;
        const Z = this.pickZ;
        const D2 = this.pickD2;
        const { x: offX, y: offY } = offsets[k];
        let m = 0;
        let nbD2 = Infinity;
        let nbX = 0;
        let nbY = 0;
        let nbDepth = 1;
        for (let j = 0; j < ZOOM_SIZE; j++) {
          for (let i = 0; i < ZOOM_SIZE; i++) {
            const d = ex[j * ZOOM_SIZE + i];
            if (!valid(d)) continue;
            // Sample position in device px, GL convention (origin bottom-left).
            const sxp = (offX + i + 0.5) / PICK_ZOOM;
            const syp = H - (offY + (ZOOM_SIZE - 1 - j) + 0.5) / PICK_ZOOM;
            const ia = Math.floor(sxp) - win.x0;
            const ja = Math.floor(syp) - win.y0;
            if (ia < 0 || ja < 0 || ia >= win.size || ja >= win.size) continue;
            const dA = win.depths[ja * win.size + ia];
            if (!valid(dA)) continue;
            // Front surface only: the point must not lie (much) behind the
            // front-most splat drawn at that pixel in pass A (5 % covers the
            // splat bias on oblique surfaces; anything further is occluded).
            const z = linearizeDepth(d, near, far);
            if (z > linearizeDepth(dA, near, far) * 1.05) continue;
            const du = sxp - gx;
            const dv = syp - gy;
            const d2 = du * du + dv * dv;
            U[m] = du;
            V[m] = dv;
            Z[m] = z;
            D2[m] = d2;
            m++;
            if (d2 < nbD2) {
              nbD2 = d2;
              nbX = sxp;
              nbY = syp;
              nbDepth = d;
            }
          }
        }
        // Nearest samples (all within ~2 CSS px, at least 16 when available).
        const rSmall2 = (2 * pr + 1) * (2 * pr + 1);
        let inSmall = 0;
        for (let q = 0; q < m; q++) if (D2[q] <= rSmall2) inSmall++;
        const want = Math.min(m, Math.max(16, inSmall));
        const swap = (a: number, b: number) => {
          let t = U[a];
          U[a] = U[b];
          U[b] = t;
          t = V[a];
          V[a] = V[b];
          V[b] = t;
          t = Z[a];
          Z[a] = Z[b];
          Z[b] = t;
          t = D2[a];
          D2[a] = D2[b];
          D2[b] = t;
        };
        if (want < m && want === inSmall) {
          // All samples within the small radius: O(m) partition (up to ~1–4 k samples per tap).
          for (let q = 0, w = 0; q < m; q++) if (D2[q] <= rSmall2) swap(w++, q);
        } else if (want < m) {
          // < 16 within the radius: the 16 nearest (partial selection, 16·m).
          for (let q = 0; q < want; q++) {
            let bi = q;
            for (let r = q + 1; r < m; r++) if (D2[r] < D2[bi]) bi = r;
            if (bi !== q) swap(q, bi);
          }
        }
        const zTap = want > 0 && (tapCovered || inSmall > 0) ? surfaceDepthAt(U, V, Z, want) : NaN;
        if (Number.isFinite(zTap) && zTap > near && zTap < far) {
          // Surface around the tap: hit on the tap ray at the fitted depth.
          px = gx;
          py = gy;
          depth = viewDepthToWindow(zTap, near, far);
        } else if (m > 0 && !tapCovered) {
          // Tap in a gap: snap to the nearest point of the front surface.
          px = nbX;
          py = nbY;
          depth = nbDepth;
        }
      }
      _v.set((px / W) * 2 - 1, (py / H) * 2 - 1, 2 * depth - 1);
      _v.unproject(this.camera);
      this.contentRoot.worldToLocal(_v);
      out.push(new Float64Array([_v.x, _v.y, _v.z]));
    }
    return out;
  }

  private pickCPU(p: { x: number; y: number }): Vec3 | null {
    _m.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse).multiply(this.contentRoot.matrixWorld);
    const e = _m.elements;
    let best: Vec3 | null = null;
    let bestW = Infinity;
    let bestCovered = false;
    const info = { covered: false, w: 0 };
    const projScale = this.height / (2 * Math.tan((this.camera.fov * Math.PI) / 360));
    const consider = (pos: ArrayLike<number>, count: number, splat: PickSplatModel) => {
      const i = pickNearestProjected(pos, count, e, this.width, this.height, p.x, p.y, PICK_RADIUS_CSS, splat, info);
      if (i < 0) return;
      // A covering splat beats a merely nearby point; then the front-most wins.
      if (bestCovered && !info.covered) return;
      if (info.covered === bestCovered && info.w >= bestW) return;
      bestW = info.w;
      bestCovered = info.covered;
      best = new Float64Array([pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]);
    };
    const pr = this.pixelRatio;
    if (this.display.points && this.dense) {
      // Use exactly the attributes the GPU draws with (radius / confidence filter).
      const g = this.dense.geometry;
      const radiusAttr = this.denseMaterial.usesRadius ? (g.getAttribute('radius') as BufferAttribute | undefined) : undefined;
      const confAttr = this.denseMaterial.usesConfidence ? (g.getAttribute('confidence') as BufferAttribute | undefined) : undefined;
      consider(this.dense.source.positions, this.dense.count, {
        sizePx: this.densePointSizeCss,
        radii: radiusAttr ? (radiusAttr.array as Float32Array) : undefined,
        projScale,
        minPx: 1 / pr,
        maxPx: Math.min(64 * pr, this.maxPointSize) / pr,
        confidence: confAttr ? (confAttr.array as Float32Array) : undefined,
        minConfidence: this.denseMaterial.minConfidence,
      });
    }
    if (this.display.points && this.sparse) {
      consider((this.sparse.geometry.getAttribute('position') as BufferAttribute).array as Float32Array, this.sparse.count, {
        sizePx: this.display.sparsePointSizePx,
      });
    }
    if (this.display.mesh && this.meshLayer) {
      const src = this.meshLayer.source;
      // Vertices as 1 px points: snapping when the tap is just off the mesh.
      consider(src.positions, src.vertexCount, { sizePx: 1 });
      // Exact surface under the tap (also inside large triangles): ray in the WORLD frame.
      _v.copy(this.camera.position);
      this.contentRoot.worldToLocal(_v);
      _w.set((p.x / this.width) * 2 - 1, 1 - (p.y / this.height) * 2, 0.5).unproject(this.camera);
      this.contentRoot.worldToLocal(_w);
      _w.sub(_v).normalize();
      const t = raycastTriangles(src.positions, src.indices, src.triangleCount, _v.x, _v.y, _v.z, _w.x, _w.y, _w.z);
      if (Number.isFinite(t)) {
        const hx = _v.x + t * _w.x;
        const hy = _v.y + t * _w.y;
        const hz = _v.z + t * _w.z;
        const w = e[3] * hx + e[7] * hy + e[11] * hz + e[15]; // view depth (clip w)
        if (w > 0 && (!bestCovered || w < bestW)) {
          bestW = w;
          bestCovered = true;
          best = new Float64Array([hx, hy, hz]);
        }
      }
    }
    if (!best) return null;
    if (!bestCovered) return best; // tap in a gap: snap to the nearest point
    // Like the depth buffer: the hit is on the tap ray at the covering splat's depth.
    _v.set((p.x / this.width) * 2 - 1, 1 - (p.y / this.height) * 2, 0.5).unproject(this.camera);
    _v.sub(this.camera.position).normalize();
    this.camera.getWorldDirection(_w);
    const t = bestW / Math.max(1e-12, _v.dot(_w));
    _v.multiplyScalar(t).add(this.camera.position);
    this.contentRoot.worldToLocal(_v);
    return new Float64Array([_v.x, _v.y, _v.z]);
  }

  /** Programmatic measurement between two screen points (CSS px); null if either misses. */
  measureBetweenScreenPoints(p1: ScreenPoint, p2: ScreenPoint): MeasureResult | null {
    const [a, b] = this.pickPoints([p1, p2]);
    if (!a || !b) return null;
    return { distance: measureDistance(a, b), a, b };
  }

  /** Enter measure mode: each tap picks a point; every second tap reports a distance. */
  startMeasure(onResult: MeasureCallback): void {
    this.measure.start(onResult);
    this.measure.setUnits(this.metric);
    this.requestRender();
  }

  stopMeasure(): void {
    this.measure.stop();
    this.requestRender();
  }

  /** Measure-mode tap at a screen position (what a user tap does). Returns the picked point. */
  measureTap(p: ScreenPoint): Vec3 | null {
    const w = this.pickPoint(p);
    if (this.measure.addPoint(w)) this.requestRender();
    return w;
  }

  // ---------------------------------------------------------------- events

  private listen(t: EventTarget, type: string, fn: EventListener): void {
    t.addEventListener(type, fn);
    this.listeners.push([t, type, fn]);
  }

  private readonly onControlsChange = (): void => {
    this.requestRender();
  };

  private readonly onControlsStart = (): void => {
    this.interacting = true;
    this.viewAnim = null;
    this.dirWithoutCams = false;
    if (this.opts.autoFit === 'follow') this.autoFitArmed = false;
    this.targetAnim.cancel();
    this.requestRender();
  };

  private readonly onControlsEnd = (): void => {
    this.interacting = false;
    this.requestRender();
  };

  private readonly onTap = (t: TapInfo): void => {
    if (!this.measure.active) return;
    this.measureTap(t);
  };

  private readonly onDoubleTap = (t: TapInfo): void => {
    if (this.measure.active || this.viewLocked) return;
    const p = this.pickPoint(t);
    if (!p) return;
    _v.set(p[0], p[1], p[2]);
    this.contentRoot.localToWorld(_v);
    this.focusOn(_v.clone(), true);
  };

  /**
   * Lock the view to a WORLD camera pose (world-to-camera, x right / y down / z forward on the
   * screen; e.g. following the capturing phone), with an optional vertical field of view. The orbit
   * controls are off while locked; null hands the view back to them (looking where it looked).
   */
  setViewPose(pose: Pose | null, fovDeg?: number): void {
    if (this.disposed) return;
    if (!pose) {
      if (!this.viewLocked) return;
      this.viewLocked = false;
      this.controls.enabled = true;
      if (this.camera.fov !== this.opts.fovDeg) {
        this.camera.fov = this.opts.fovDeg;
        this.camera.updateProjectionMatrix();
      }
      this.controls.update();
      this.requestRender();
      return;
    }
    this.viewLocked = true;
    this.controls.enabled = false;
    this.viewAnim = null;
    this.dirWithoutCams = false;
    this.targetAnim.cancel();
    const R = pose.R;
    const c = cameraCenter(pose);
    this.camera.position.set(c[0], c[1], c[2]);
    this.contentRoot.localToWorld(this.camera.position);
    // three's camera axes in the world: right = row 0 of R, up = −row 1, backward = −row 2.
    _m.set(R[0], -R[3], -R[6], 0, R[1], -R[4], -R[7], 0, R[2], -R[5], -R[8], 0, 0, 0, 0, 1);
    this.camera.quaternion.setFromRotationMatrix(_m).premultiply(this.contentRoot.quaternion);
    // Orbit target ahead of the camera: clip planes, headlight, and the view after unlocking.
    this.recomputeBounds();
    const ahead = Math.max(1e-6, 0.5 * (this.contentBounds?.radius ?? 1));
    this.controls.target.copy(this.camera.position).addScaledVector(_v.set(0, 0, -1).applyQuaternion(this.camera.quaternion), ahead);
    if (fovDeg !== undefined && Number.isFinite(fovDeg) && fovDeg > 0 && Math.abs(fovDeg - this.camera.fov) > 1e-3) {
      this.camera.fov = fovDeg;
      this.camera.updateProjectionMatrix();
    }
    this.requestRender();
  }

  /** Set the orbit target (three-space) — animated by default. */
  private focusOn(target: Vector3, animate: boolean): void {
    this.viewAnim = null;
    if (animate) this.targetAnim.start(target, performance.now());
    else {
      this.controls.target.copy(target);
      this.controls.update();
    }
    this.requestRender();
  }

  /** Set the orbit target to a WORLD point (what a double tap does). */
  setOrbitTarget(p: ArrayLike<number>, animate = false): void {
    _v.set(p[0], p[1], p[2]);
    this.contentRoot.localToWorld(_v);
    this.focusOn(_v.clone(), animate);
  }

  /** Current orbit target in the WORLD frame. */
  getOrbitTarget(): Vec3 {
    _v.copy(this.controls.target);
    this.contentRoot.worldToLocal(_v);
    return new Float64Array([_v.x, _v.y, _v.z]);
  }
}
