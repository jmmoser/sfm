/**
 * Types of the dataset-driven engine harness (harness.html, `window.runPipeline(config)`).
 * Everything in {@link HarnessResult} is plain JSON (typed arrays are converted), so it survives
 * Playwright's `page.evaluate` (or the bridge's POST of a phone run) and can be written to disk as is.
 */
import type { MetricInfo } from '../../core/types';
import type { DebugKeyframe, DebugPair } from './debug-dump';
import type {
  CaptureMode,
  DenseOptions,
  ExportFormat,
  Guidance,
  InitResult,
  KeyframeDecision,
  MeshOptions,
  QualityPreset,
  SplatOptions,
  TrackingState,
} from '../../workers/protocol';

export type HarnessEngineKind = 'worker' | 'inline' | 'mock';

export interface HarnessTimeouts {
  /** EngineClient init timeout (default 600 s: pipeline compilation is slow on SwiftShader). */
  initMs?: number;
  /** EngineClient per-frame timeout (default 300 s; the app uses 30 s). */
  frameMs?: number;
  /** EngineClient short calls (default 120 s). */
  callMs?: number;
  /** EngineClient inactivity timeout of long operations (default 30 min). */
  inactivityMs?: number;
}

export interface HarnessOutput {
  /** Path prefix (relative to the runner's output directory) of the streamed files, e.g. 'object/'. */
  prefix: string;
  /** Stream `sparse.ply` (default true). */
  sparse?: boolean;
  /** Stream `dense.ply` (default true). */
  dense?: boolean;
  /** Stream `mesh.ply` (default true). */
  mesh?: boolean;
  /** Stream `splats.ply` (exportModel('ply-splat'), default true). */
  splat?: boolean;
}

export interface HarnessConfig {
  /** Dataset directory under e2e/data (e.g. 'object', 'yard', 'synth40-object'). */
  dataset: string;
  /** Dataset base URL (default `/e2e/data/<dataset>/`). */
  baseUrl?: string;
  /** 'worker' (EngineClient over src/workers/engine.worker.ts, default), 'inline' (src/engine on the page) or 'mock'. */
  engine?: HarnessEngineKind;
  /** Default: 'object' for object datasets, 'scene' otherwise. */
  mode?: CaptureMode;
  /** Default 'fast'. */
  quality?: QualityPreset;
  /** First dataset frame (position in time order, default 0). */
  startFrame?: number;
  /** Replay every Nth frame (default 1). */
  frameStride?: number;
  /** Maximum number of frames delivered to the engine (default: all). */
  maxFrames?: number;
  /**
   * Simulated FramePump rate (Hz): frames closer than 1000/fps ms (minus half a dataset frame
   * interval of tolerance) to the previously delivered one are dropped. Applied after the stride.
   */
  fps?: number;
  /** Pass IMU samples (default true). */
  useImu?: boolean;
  /** Pass GPS fixes when the dataset has them (default true). */
  useGps?: boolean;
  /** Pass compass headings (default true). */
  useHeading?: boolean;
  /**
   * North reference of the replayed compass headings (feed.ts `HeadingNorthMode`): 'magnetic'
   * (WMM2025 at the dataset location; the engine must apply the declination), 'app' (magnetic
   * until the first GPS fix, then true north like the app) or 'true'. Default: 'magnetic' when
   * the dataset has a location and GPS is fed, else 'true'.
   */
  headingNorth?: 'magnetic' | 'app' | 'true';
  /** Capture date (epoch ms) for the magnetic declination of the replayed headings (default: the run's start). */
  captureDateMs?: number;
  /** Pass `gravityDevice` (FramePump metadata, from the IMU; default = useImu). */
  useGravity?: boolean;
  /**
   * EngineOptions.hfovPriorDeg (FOV across the long side). Default 67.3 (what CameraCapture
   * reports for an iPhone main camera; the synthetic datasets use 65°). 'gt' = exact value from
   * the dataset intrinsics, 'none' = no prior.
   */
  hfovPriorDeg?: number | 'gt' | 'none';
  /** Run densify (default false). */
  densify?: boolean | DenseOptions;
  /** Run mesh (default false; needs densify). */
  mesh?: boolean | MeshOptions;
  /** Run the Gaussian-splatting stage after mesh / densify / refine (default false). */
  splat?: boolean | SplatOptions;
  /** Export formats (sizes only): true = every applicable format (default false). */
  exports?: boolean | ExportFormat[];
  /** EngineOptions.persist (default false). */
  persist?: boolean;
  /** EngineOptions.preferGPU (default true). */
  preferGPU?: boolean;
  /**
   * Engine tuning (src/engine EngineConfig, e.g. `{ mapping: 'inline' }`): sent as the
   * `engineConfig` extension field of the init options (worker) / second constructor argument
   * (inline). Default: none (engine defaults).
   */
  engineConfig?: Record<string, unknown> | null;
  /** MockEngine time multiplier (default 4). */
  mockSpeed?: number;
  timeouts?: HarnessTimeouts;
  /** Stream PLY files to the runner through `window.__e2eWrite` (Playwright or the fetch bridge, e2e/run-e2e.mjs). */
  output?: HarnessOutput;
  /** Embed the dense cloud as base64 in the result when it is not streamed (default true). */
  embedDensePoints?: boolean;
  /** Abort the capture after this many consecutive frame errors (default 3). */
  maxFrameErrors?: number;
  /** Log every frame to the runner (default true; false = every 10th). */
  logFrames?: boolean;
  /**
   * Diagnostics: keep the camera poses of every LiveUpdate that carries the live model in
   * `result.liveTrace` (≈ 1–2 MB for 80 keyframes). Default false.
   */
  liveTrace?: boolean;
  /**
   * Diagnostics, `engine: 'inline'` only (debug-dump.ts reads engine internals): `result.debug`
   * gets every keyframe's live registration state + pose after each keyframe frame (use
   * `engineConfig.mapping: 'inline'` so the frame's keyframe is already mapped), and the verified
   * pairs + live model after finishCapture and after refine. Default false.
   */
  debugDump?: boolean;
  /**
   * Experiments (needs `persist: true`): after mesh, write the engine's session store (keyframe
   * JPEGs + records, values, binaries incl. the filtered `depth/` maps) to `<output prefix>session/`
   * through `window.__e2eWrite`, so the splat stage can be re-run on the same inputs
   * (src/harness/engine/replay.ts, e2e/splat-replay.mjs). Default false.
   */
  dumpSession?: boolean;
}

/** Config with every default filled in. */
export interface ResolvedHarnessConfig extends Required<Omit<HarnessConfig, 'output' | 'fps' | 'maxFrames' | 'timeouts' | 'hfovPriorDeg' | 'engineConfig'>> {
  engineConfig: Record<string, unknown> | null;
  output: HarnessOutput | null;
  fps: number | null;
  maxFrames: number | null;
  timeouts: Required<HarnessTimeouts>;
  hfovPriorDeg: number | 'gt' | 'none';
  /** The prior actually passed to the engine. */
  hfovPriorDegUsed: number | null;
}

export interface PoseJSON {
  R: number[];
  t: number[];
}

export interface IntrinsicsJSON {
  f: number;
  cx: number;
  cy: number;
  k1: number;
  k2: number;
  width: number;
  height: number;
}

export interface FrameRecord {
  /** Delivered-frame counter (FrameInput.index). */
  i: number;
  /** Dataset frame index (frames.json `index`). */
  index: number;
  /** Capture time (ms). */
  t: number;
  /** LiveUpdate.frameIndex. */
  frameIndex: number | null;
  decision: KeyframeDecision | null;
  guidance: Guidance | null;
  tracking: TrackingState | null;
  numKeyframes: number;
  numRegistered: number;
  numPoints: number;
  /** Overlay keypoints. */
  overlay: number;
  /** Sparse points in the update (when the model changed). */
  sparse: number | null;
  /** Cameras in the update (when the model changed). */
  cameras: number | null;
  /** IMU samples passed with the frame. */
  imu: number;
  /** Engine timings (LiveUpdate.timings). */
  timings: Record<string, number>;
  /** Wall time of processFrame (ms). */
  wallMs: number;
  /** Fetch + decode of the JPEG (ms; overlapped with the previous frame). */
  decodeMs: number;
  error?: string;
}

export interface ProgressRecord {
  step: string;
  stage: string;
  /** UI stage (src/app/stages.ts) or null when unmapped. */
  uiStage: string | null;
  progress: number;
  message?: string;
  /** ms since the start of the run. */
  ms: number;
}

export interface StageTiming {
  step: string;
  stage: string;
  uiStage: string | null;
  /** First / last progress event (ms since the start). */
  firstMs: number;
  lastMs: number;
  /** Until the next stage of the step started (or the step ended). */
  durationMs: number;
  events: number;
}

export interface NumStats {
  n: number;
  mean: number | null;
  median: number | null;
  p90: number | null;
  max: number | null;
}

export interface MappedImage {
  /** Dataset frame index. */
  index: number;
  /** SparseCamera.id. */
  id: number;
  pose: PoseJSON;
  f: number;
  /** How the camera was mapped to a dataset frame. */
  via: 'frameIndex' | 't' | 'keyframe-ordinal' | 'id-as-frame';
}

export interface SparseSummary {
  intrinsics: IntrinsicsJSON | null;
  /** Set when the engine's intrinsics were rescaled / are in another orientation. */
  intrinsicsNote: string | null;
  images: MappedImage[];
  numCameras: number;
  /** Camera ids that could not be mapped to a dataset frame. */
  unmapped: number[];
  /** Camera ids mapped to an already used frame (dropped). */
  duplicates: number[];
  mapping: string;
  metric: MetricInfo;
  stats: Record<string, number | string>;
  pointsCount: number;
}

export interface LiveSummary {
  delivered: number;
  keyframes: number;
  registered: number;
  points: number;
  /** First update with tracking === 'tracking'. */
  timeToInit: { delivered: number; index: number; captureMs: number; wallMs: number; keyframes: number } | null;
  trackingLostFrames: number;
  /** `live` events not returned as a frame response (background mapping). */
  unsolicitedUpdates: number;
  /** Dataset indices of the frames the engine chose as keyframes. */
  keyframeIndices: number[];
  frameMs: NumStats;
  keyframeMs: NumStats;
  /** Last live cameras (LiveUpdate.cameras of the newest update carrying them). */
  cameras: { pose: PoseJSON; f: number }[];
  /**
   * Last live cameras mapped to dataset frames by keyframe order (only when every keyframe of
   * that update is registered and each one was announced by a 'keyframe' frame response;
   * integrator must verify the ordering assumption).
   */
  images: { index: number; pose: PoseJSON }[] | null;
}

export interface CloudSummary {
  count: number;
  hasNormals: boolean;
  hasConfidence: boolean;
  bboxMin: number[] | null;
  bboxMax: number[] | null;
  /** Streamed file (relative to the run's output directory). */
  file: string | null;
}

export interface HarnessError {
  step: string;
  name: string;
  message: string;
  stack?: string;
}

export interface LogRecord {
  ms: number;
  level: string;
  message: string;
}

/** What the run ran on (./device.ts); queried while the dataset loads. */
export interface HarnessDevice {
  userAgent: string;
  /** `navigator.platform` (deprecated, but 'iPhone' / 'MacIntel' / 'Linux x86_64' is useful here). */
  platform: string | null;
  /** Every iPhone reports 4 (docs/research/ios-platform.md). */
  hardwareConcurrency: number | null;
  devicePixelRatio: number;
  /** CSS px. */
  screen: [number, number] | null;
  secureContext: boolean;
  /** The page's WebGPU adapter (requested only, no device). */
  adapter: {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
    features: string[];
    /** maxBufferSize (≈ RAM class on Apple GPUs), maxStorageBufferBindingSize, … */
    limits: Record<string, number>;
  } | null;
  /** Why `adapter` is null. */
  adapterError?: string;
}

export interface HarnessResult {
  ok: boolean;
  aborted: boolean;
  config: ResolvedHarnessConfig;
  dataset: {
    name: string;
    scene: string;
    numFrames: number;
    width: number;
    height: number;
    fps: number;
    screenAngle: number;
    sceneExtent: number;
    hasGeo: boolean;
    hasHeading: boolean;
    imuSamples: number;
  };
  engine: { kind: HarnessEngineKind; init: InitResult | null; respawns: number; statusChanges: { ms: number; state: string; reason?: string }[] };
  device: HarnessDevice | null;
  /** Harness steps (ms): load, init, capture, finishCapture, refine, densify, mesh, splat, export, total. */
  timings: Record<string, number>;
  stageTimings: StageTiming[];
  frames: FrameRecord[];
  decisions: Record<string, number>;
  guidance: Record<string, number>;
  tracking: Record<string, number>;
  live: LiveSummary;
  progress: ProgressRecord[];
  sparse: SparseSummary | null;
  dense: CloudSummary | null;
  /** Dense cloud as base64 (Float32 LE xyz + Uint8 rgb) when not streamed. */
  densePoints?: { count: number; xyz: string; rgb: string };
  mesh: { vertexCount: number; triangleCount: number; file: string | null; bboxMin: number[] | null; bboxMax: number[] | null } | null;
  /** The splat stage's result (SplatResult) and the streamed `splats.ply`. */
  splat?: {
    count: number;
    iterations: number;
    ms: number;
    shDegree: number;
    bytes: number;
    antialiased: boolean;
    stats: Record<string, number | string>;
    /** `detail` of the last 'splat_training' progress event (SplatTrainingDetail). */
    lastDetail: Record<string, number> | null;
    file: string | null;
  } | null;
  exports: Record<string, { bytes: number; ms: number; type: string } | { error: string }>;
  /** Files streamed through `__e2eWrite` (relative to the runner's output directory). */
  files: string[];
  logs: LogRecord[];
  errors: HarnessError[];
  /** JS heap of the page (Chromium `performance.memory`, bytes) after each step. Worker memory not included. */
  memory: Record<string, number>;
  /**
   * With `liveTrace`: one entry per LiveUpdate with cameras (frame responses and background
   * model updates). `poses` are the registered keyframes' poses sorted by keyframe id; when
   * `registered === keyframes` the k-th pose is the k-th keyframe (`live.keyframeIndices[k]`).
   */
  liveTrace?: { ms: number; frameIndex: number; keyframes: number; registered: number; points: number; poses: PoseJSON[] }[];
  /** With `debugDump` (inline engine). */
  debug?: {
    keyframeTrace: { frameIndex: number; keyframes: DebugKeyframe[] }[];
    afterCapture?: { keyframes: DebugKeyframe[] | null; pairs: DebugPair[] | null };
    afterRefine?: { keyframes: DebugKeyframe[] | null; pairs: DebugPair[] | null };
  };
}

