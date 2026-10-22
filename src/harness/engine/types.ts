/**
 * Types of the dataset-driven engine harness (harness.html, `window.runPipeline(config)`).
 * Everything in {@link HarnessResult} is plain JSON (typed arrays are converted), so it survives
 * Playwright's `page.evaluate` and can be written to disk as is.
 */
import type { MetricInfo } from '../../core/types';
import type { CoveragePrompt } from '../../app/coverage';
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
  /** Default: 'object' for object datasets, 'scene' otherwise; a recording's recorded mode. */
  mode?: CaptureMode;
  /** Default 'fast'; a recording's recorded preset. */
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
  /** Capture date (epoch ms) for the magnetic declination of the replayed headings (default: the run's start; a recording's own date). */
  captureDateMs?: number;
  /**
   * Raw capture recordings (frames.jsonl, src/io/recording.ts): replay the recorded FrameInputs
   * verbatim (default true when the dataset is a recording). false = synthesize them from
   * frames.json / imu.json / geo.json / heading.json like a synthetic dataset.
   */
  verbatim?: boolean;
  /**
   * Verbatim replay: 'recorded' (default) = the recorded JPEG as is (≤ 1280 px); 'original' =
   * resized back to the size the engine got on the device (a VideoFrame's display size fitted to
   * its `maxSide`, else the bitmap size).
   */
  replaySize?: 'recorded' | 'original';
  /** Pass `gravityDevice` (FramePump metadata, from the IMU; default = useImu). */
  useGravity?: boolean;
  /**
   * EngineOptions.hfovPriorDeg (FOV across the long side). Default 67.3 (what CameraCapture
   * reports for an iPhone main camera; the synthetic datasets use 65°), for a recording the
   * recorded prior. 'gt' = exact value from the dataset intrinsics, 'none' = no prior.
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
  /** Stream PLY files to Node through `window.__e2eWrite` (installed by e2e/run-e2e.mjs). */
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
  /**
   * Continue the capture after a page reload (EngineOptions.extend, docs/ENGINE.md "Extend"): at
   * this fraction (0 < f < 1) of the planned frames the harness finishes and refines the first
   * segment, re-inits the engine with `{ sessionId, resume: true, extend: true }` (implies
   * `persist`), skips `extendSkipS` seconds of frames and feeds the rest with every time (frame,
   * IMU, GPS) rebased to start at 500 ms, like a reloaded page's performance.now(); then
   * finishCapture, refine and the rest as usual (`result.extend`). Default null (off).
   */
  extendAt?: number | null;
  /** Seconds of planned frames skipped after the `extendAt` split (default 0). */
  extendSkipS?: number;
}

/** Config with every default filled in. */
export interface ResolvedHarnessConfig extends Required<Omit<HarnessConfig, 'output' | 'fps' | 'maxFrames' | 'timeouts' | 'hfovPriorDeg' | 'engineConfig' | 'extendAt'>> {
  extendAt: number | null;
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
  /** FrameInput.index sent: the delivered-frame counter, or in a verbatim replay the recorded index. */
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
  /** Coverage of those cameras (what the capture screen's object-mode ring shows); null without cameras. */
  coverage: LiveCoverage | null;
  /**
   * Last live cameras mapped to dataset frames by keyframe order (only when every keyframe of
   * that update is registered and each one was announced by a 'keyframe' frame response;
   * integrator must verify the ordering assumption).
   */
  images: { index: number; pose: PoseJSON }[] | null;
}

/**
 * Object coverage of a set of live cameras (src/app/coverage.ts `computeCoverage`; elevation
 * bands only when gravity is fed — the app's rule for a running motion sensor). Recorded for
 * every capture: a scene capture usually stays `stable: false` (the cameras do not converge).
 */
export interface LiveCoverage {
  stable: boolean;
  /** Elevation bands in use (gravity fed). */
  elevation: boolean;
  /** Covered sectors of the level (low + mid) and high tiers / 24 (azimuth only: covered sectors / 12). */
  completeness: number;
  /** Covered cells (≥ 2 keyframes) of the 12 sectors per band: low −10–10°, mid 10–40°, high > 40°. */
  covered: [number, number, number];
  /** Covered sectors at any height. */
  coveredSectors: number;
  prompt: CoveragePrompt;
  /** Keyframes per cell, [band][sector] (sector 0 = the first keyframe, counter-clockwise from above). */
  counts: number[][];
  /** Cameras the coverage was computed from. */
  cameras: number;
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

/** A raw capture recording's replay (`HarnessResult.recording`). */
export interface RecordingSummary {
  /** The recorded FrameInputs were replayed verbatim. */
  verbatim: boolean;
  /**
   * frames.jsonl rows replayed as frames (the engine got them, with pixels) / not replayed
   * (unparseable, no pixels — their IMU goes to the next frame —, not sent) / of those, rows the
   * device engine never got (left out, IMU included).
   */
  rows: number;
  skipped: number;
  notSent: number;
  /** What the engine answered on the device (live.jsonl; null without it). */
  device: {
    frames: number;
    decisions: Record<string, number>;
    /** Counts of the device's last answer. */
    keyframes: number | null;
    registered: number | null;
    /** processFrame wall time / recorder back-pressure wait on the device (ms). */
    frameMs: NumStats;
    waitMs: NumStats;
    errors: number;
  } | null;
  /** Per-frame decisions of the replay vs the device's (same recorded frame). */
  agreement: { compared: number; same: number; rate: number | null; keyframesDevice: number; keyframesReplay: number; confusion: Record<string, number> } | null;
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
    /** A raw capture recording (frames.jsonl). */
    recording?: boolean;
  };
  /** Raw capture recordings: verbatim replay and the device comparison. */
  recording?: RecordingSummary;
  engine: { kind: HarnessEngineKind; init: InitResult | null; respawns: number; statusChanges: { ms: number; state: string; reason?: string }[] };
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
  /** With `extendAt`: the split, the first segment's refine and the second segment's keyframes. */
  extend?: ExtendSummary | null;
  /**
   * With `liveTrace`: one entry per LiveUpdate with cameras (frame responses and background
   * model updates). `poses` are the registered keyframes' poses sorted by keyframe id; when
   * `registered === keyframes` the k-th pose is the k-th keyframe (`live.keyframeIndices[k]`).
   */
  liveTrace?: {
    ms: number;
    frameIndex: number;
    keyframes: number;
    registered: number;
    points: number;
    poses: PoseJSON[];
    /** The coverage ring of this update (LiveCoverage without the counts). */
    coverage: Omit<LiveCoverage, 'counts'>;
  }[];
  /** With `debugDump` (inline engine). */
  debug?: {
    keyframeTrace: { frameIndex: number; keyframes: DebugKeyframe[] }[];
    afterCapture?: { keyframes: DebugKeyframe[] | null; pairs: DebugPair[] | null };
    afterRefine?: { keyframes: DebugKeyframe[] | null; pairs: DebugPair[] | null };
  };
}

/** HarnessResult.extend (`extendAt`). */
export interface ExtendSummary {
  /** Frames delivered before the "reload" and the dataset frame index the second segment starts at. */
  firstFrames: number;
  resumeIndex: number;
  /** Planned frames skipped after the split (`extendSkipS`). */
  skippedFrames: number;
  /** Subtracted from the dataset times of the second segment (ms). */
  rebaseMs: number;
  /** The first segment's finishCapture + refine (live counts after finishCapture). */
  first: { keyframes: number; registeredLive: number; registeredImages: number | null; refineMs: number; error: string | null };
  /** The re-init with resume + extend. */
  init: InitResult | null;
  resumedKeyframes: number | null;
  /** Keyframes of the second segment: delivered, registered in the live model, in the final model. */
  secondKeyframes: number | null;
  secondRegisteredLive: number | null;
  secondRegisteredFinal: number | null;
}
