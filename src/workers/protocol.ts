/**
 * Typed RPC protocol between the UI thread and the engine worker (docs/ENGINE.md §2–3).
 *
 *   main → worker   EngineRequest   { id, method, params }
 *   worker → main   EngineResponse  { id, ok: true, result } | { id, ok: false, error }
 *   worker → main   EngineEvent     { type: 'progress' | 'live' | 'log' | 'error', … }
 *
 * Responses are told apart from events by the `type` field (events) vs `id` + `ok` (responses):
 * use {@link isEngineEvent} / {@link isEngineResponse}.
 *
 * Large typed arrays and ImageBitmaps are TRANSFERRED, never copied: build the transfer list
 * with {@link collectTransferables}. Small arrays (poses, intrinsics) are structured-cloned —
 * `Float64Array`s and plain objects survive the clone unchanged. Only transfer buffers the
 * sender no longer needs (a transferred buffer is detached on the sending side).
 *
 * Extensions over ENGINE.md (superset, nothing removed): the `visibility` request (workers get
 * no `visibilitychange`; the engine must not submit GPU work while the page is hidden), the
 * `close` request (let go of a session before it is deleted),
 * `EngineOptions.resume` (re-open a persisted session after a worker respawn), optional
 * FramePump metadata on `FrameInput`, `SessionSummary` carries the stored models, the Gaussian
 * splatting stage (`splat`, `splatModel`, export 'ply-splat', `ProgressEvent.detail`, the
 * 'splat_preview' snapshots in `ProgressEvent.image`, the `finishEarly` request).
 */
import type {
  CameraIntrinsics,
  GeoFix,
  ImuSample,
  Mat3,
  MetricInfo,
  PointCloud,
  Pose,
  ProgressEvent,
  RGBAImage,
  TriangleMesh,
  Vec3,
} from '../core/types';
import type { SessionStatus } from '../io/storage';
import type { SplatModel } from '../splat/types';

export const ENGINE_PROTOCOL_VERSION = 1;

export type CaptureMode = 'object' | 'scene' | 'auto';
export type QualityPreset = 'fast' | 'balanced' | 'max';
export type ScreenAngle = 0 | 90 | 180 | 270;

// ---------------------------------------------------------------------------------------------
// Engine API types (ENGINE.md §2)
// ---------------------------------------------------------------------------------------------

export interface EngineOptions {
  mode: CaptureMode;
  quality: QualityPreset;
  /** Persist frames/models in IndexedDB under this session. */
  sessionId?: string;
  /** Default true in the app, false in tests. */
  persist?: boolean;
  /** Default true. */
  preferGPU?: boolean;
  /** From CameraCapture (lens + aspect): FOV across the LONG image side, degrees. */
  hfovPriorDeg?: number;
  /**
   * Extension: re-open the persisted state of `sessionId` instead of starting empty (sent by
   * EngineClient when it re-initialises a respawned worker).
   */
  resume?: boolean;
}

export interface InitResult {
  gpu: boolean;
  adapter?: string;
  /** Why the engine runs on the CPU although a GPU was preferred. */
  fallbackReason?: string;
}

export interface HeadingSample {
  /** Degrees clockwise from north (see `north`). */
  heading: number;
  /** Degrees. */
  accuracy: number;
  /**
   * Extension: north reference of `heading`. 'magnetic' (platform compass, declination unknown):
   * the engine applies the WMM declination at the capture location in the world alignment.
   * Absent = 'true' (the documented FrameInfo.heading contract).
   */
  north?: 'true' | 'magnetic';
  /** Extension: declination (deg, east positive) already included in `heading` (informational). */
  declinationDeg?: number | null;
}

export interface AttitudeSample {
  t: number;
  /** Device → world rotation (row-major 3×3). ENU when `absolute`. */
  R_world_device: Mat3;
  absolute: boolean;
  yawAccuracyDeg: number | null;
}

export interface FrameInput {
  /**
   * Full resolution, any UI orientation. Transferred to the worker (owned by the engine). A
   * VideoFrame (the "VideoFrame grab" setting: `new VideoFrame(<video>)`, no copy on the main
   * thread) is converted to an ImageBitmap by the engine — downscaled to `maxSide` — and closed.
   */
  bitmap: ImageBitmap | RGBAImage | VideoFrame;
  /** Capture time, ms (performance.now() domain). */
  t: number;
  /** θ of ARCHITECTURE §2 when the frame was captured. */
  screenAngle: ScreenAngle;
  /**
   * performance.now() time θ took the value `screenAngle` (FramePump: when the UI rotated; −∞: no
   * rotation seen). The engine's settle window after a rotation starts there, else at the first
   * frame with the new θ.
   */
  screenAngleSince?: number;
  /** IMU samples since the previous frame. */
  imu: ImuSample[];
  geo?: GeoFix | null;
  heading?: HeadingSample | null;
  // ---- optional FramePump metadata (extension) ----
  /** 0-based index of delivered frames. */
  index?: number;
  /** Camera capture segment: changes after an interruption (relocalise / new segment). */
  segment?: number;
  /** Unit gravity direction (DOWN) in the device frame at t. */
  gravityDevice?: Vec3 | null;
  attitude?: AttitudeSample | null;
  /** MotionSensor calibration version the IMU samples were normalised with (-1: no motion). */
  imuCalibrationVersion?: number;
  timeSource?: 'captureTime' | 'presentationTime' | 'callback';
  /**
   * The pixels are a raw sensor-orientation buffer (a VideoFrame of the <video> on iOS): upright
   * for θ = 90 in every UI rotation (src/engine/frame-ingest.ts). Per frame: never persisted.
   */
  rawBuffer?: boolean;
  /** VideoFrame only: downscale so max(W, H) ≤ maxSide when converting (null/absent: native). */
  maxSide?: number | null;
}

/** A FrameInput after the engine converted a VideoFrame (what the live session works on). */
export type DecodedFrameInput = FrameInput & { bitmap: ImageBitmap | RGBAImage };

export type KeyframeDecision = 'keyframe' | 'skip' | 'reject_blur' | 'reject_motion';
export type Guidance = 'good' | 'move_slower' | 'more_texture' | 'translate_more' | 'tracking_lost' | 'initializing';
export type TrackingState = 'initializing' | 'tracking' | 'lost';

export interface LiveCamera {
  pose: Pose;
  intrinsics: CameraIntrinsics;
  /** Capture time of the keyframe on the IMU clock (camera time + estimated camera–IMU offset), ms. */
  t?: number;
  /**
   * Physical up in the camera frame (unit): −gravity when known, else up for a phone held in the
   * frame's UI rotation. The images are upright for the session's θ₀, not for how the phone was
   * held (raw sensor buffers: θ₀ = 90 in a portrait UI), so image up (−y) need not be up.
   */
  upCam?: [number, number, number];
}

export interface LiveUpdate {
  frameIndex: number;
  decision: KeyframeDecision;
  guidance: Guidance;
  tracking: TrackingState;
  numKeyframes: number;
  numRegistered: number;
  numPoints: number;
  /**
   * Keypoints of the processed frame in [0,1] normalised image coordinates, interleaved
   * (u0, v0, u1, v1, …), decimated. Relative to the frame AS SUBMITTED (its bitmap / UI
   * orientation `screenAngle`), even when the engine rotates frames to a canonical
   * orientation internally.
   */
  overlay?: Float32Array;
  /** Present when the sparse model changed (decimated to ≤ 50k points). */
  sparse?: { positions: Float32Array; colors: Uint8Array };
  cameras?: LiveCamera[];
  /**
   * Pose of THIS frame in the live model (tracked against a registered keyframe), and its capture
   * time on the IMU clock (camera time + estimated camera–IMU offset), ms. Absent when not tracked.
   */
  framePose?: { pose: Pose; t: number };
  /** With `cameras` / `framePose`: R_cam_device of the camera frame the poses use (v_cam = R · v_device). */
  R_cam_device?: Mat3;
  /** With `cameras`: the engine's gyro bias estimate (rad/s, device frame, subtracted from ω); null = none yet. */
  gyroBias?: [number, number, number] | null;
  timings: Record<string, number>;
  /** Per-frame diagnostics for the capture debug overlay. */
  debug?: LiveDebug;
}

/** Diagnostics of one processed frame (capture debug overlay). */
export interface LiveDebug {
  /** Feature backend in use. */
  backend: 'webgpu' | 'cpu';
  /** Why the CPU backend is used (null on WebGPU). */
  fallbackReason: string | null;
  /** Input the features of this frame were extracted from (null = not extracted). */
  featureInput: 'bitmap' | 'gray' | null;
  /** Frame size as submitted / canonical (0 = dropped before preparation). */
  srcWidth: number;
  srcHeight: number;
  width: number;
  height: number;
  /** Features of this frame (null = not extracted) and the keyframe minimum. */
  numFeatures: number | null;
  minFeatures: number;
  /** Sharpness of this frame and the running reference of the blur gate (NaN = none yet). */
  sharpness: number;
  sharpnessRef: number;
  /** Gyro angular speed (rad/s, NaN = unknown). */
  angularSpeed: number;
  /** Mean / standard deviation of the frame's luma (0–255); ≈ 0 / 0 = blank frame. */
  lumaMean: number;
  lumaStd: number;
  /** Inliers to the last keyframe (NaN = not tracked). */
  inliers: number;
  /** Why the frame got its decision. */
  reason: string;
}

export interface SparseCamera {
  id: number;
  pose: Pose;
  intrinsics: CameraIntrinsics;
  /** Physical up in the camera frame (unit), as LiveCamera.upCam; absent = unknown. */
  upCam?: [number, number, number];
}

export interface SparseResult {
  points: PointCloud;
  cameras: SparseCamera[];
  metric: MetricInfo;
  /**
   * Free-form statistics for the info sheet. Well-known keys (all optional): `images`,
   * `registeredImages`, `points`, `meanReprojErrorPx`, `meanTrackLength`, `focalPx`, `k1`, `k2`,
   * and timings `<stage>Ms` (e.g. `matchingMs`, `reconstructionMs`, `alignmentMs`).
   */
  stats: Record<string, number | string>;
}

export interface DenseOptions {
  /** Default: the session preset. */
  quality?: QualityPreset;
  /** Long side of the undistorted images used for stereo (overrides the preset). */
  maxImageSize?: number;
  /** Limit the number of reference views. */
  maxViews?: number;
}

export interface MeshOptions {
  quality?: QualityPreset;
  /** TSDF voxel size in world units (default: automatic). */
  voxelSize?: number;
  /** Simplify to about this many triangles (default: no simplification). */
  targetTriangles?: number;
}

/**
 * 'ply-splat': the trained Gaussian splats as an INRIA 3DGS PLY (x y z, f_dc / f_rest, opacity
 * logit, log-scales, rot w x y z; appearance-normalised; world frame of the model) — opens in
 * SuperSplat, Brush, Spark and most splat viewers.
 */
export type ExportFormat = 'ply-points' | 'ply-mesh' | 'glb' | 'colmap-zip' | 'ply-splat';
export const EXPORT_FORMATS: readonly ExportFormat[] = ['ply-points', 'ply-mesh', 'glb', 'colmap-zip', 'ply-splat'];

// ---------------------------------------------------------------------------------------------
// Gaussian splatting stage (extension; docs/ENGINE.md "Splat")
// ---------------------------------------------------------------------------------------------

export interface SplatOptions {
  /** Default: the session preset (iterations, Gaussian cap, image size, views: docs/ENGINE.md). */
  quality?: QualityPreset;
  /** Training iterations (overrides the preset). */
  iterations?: number;
  /** Hard cap on the number of Gaussians (overrides the preset; the memory plan may lower it). */
  maxGaussians?: number;
  /** Long side of the training images (px, ≤ the dense size; the memory plan may lower it). */
  maxImageSize?: number;
  /** Most training views (registered keyframes, evenly spread). */
  maxViews?: number;
  /** Far background shell (sky, distant structure). Default: on for scene captures. */
  backgroundShell?: boolean;
  /** Ignore a resumable checkpoint of the same run and train from the start. */
  restart?: boolean;
  /**
   * Post 'splat_preview' events while training: snapshots of the model from one training view
   * (default off: the harness and tests opt in like the app).
   */
  preview?: boolean;
}

/** Progress stage names of `splat` (ProgressEvent.stage). */
export const SPLAT_STAGES = {
  /** Views, memory plan, initial Gaussians (dense + sparse + shell), pipeline compilation, uploads. */
  prepare: 'splat_prepare',
  /** Training; `ProgressEvent.detail` is a `SplatTrainingDetail`. */
  training: 'splat_training',
  /**
   * With `SplatOptions.preview`: a snapshot of the model being trained (`ProgressEvent.image`,
   * RGBA8 ≤ 400 px; `detail`: `SplatPreviewDetail`), once before the training and then every ≥ 2 s
   * of it. Interleaved with the other stages and never throttled.
   */
  preview: 'splat_preview',
  /** Appearance normalisation, read-back, persisting the model. */
  finishing: 'splat_finishing',
} as const;

/** `ProgressEvent.detail` of the 'splat_training' stage. */
export interface SplatTrainingDetail {
  [key: string]: number;
  iteration: number;
  iterations: number;
  /** Training loss of the last iteration ((1 − λ)·L1 + λ·D-SSIM, λ = 0.2). */
  loss: number;
  /** Iterations per second of the last chunk. */
  itPerSec: number;
  gaussians: number;
  /**
   * First iteration at which `finishEarly` ends the training (the end of the coarse-to-fine
   * resolution schedule: 30 % of the iterations by default); a request before it takes effect there.
   */
  finishFrom: number;
}

/** `ProgressEvent.detail` of the 'splat_preview' stage. */
export interface SplatPreviewDetail {
  [key: string]: number;
  /** Iterations trained when the snapshot was rendered (0: the initial Gaussians). */
  iteration: number;
  iterations: number;
  /** Keyframe id of the training view it is rendered from (the one nearest the middle photo). */
  viewId: number;
  /**
   * PSNR (dB) against that view's training photo at the snapshot size, raw colours (no per-view
   * colour correction or appearance normalisation). A training view: it flatters the model.
   */
  psnr: number;
}

/** What `splat` resolves with (the model itself: `splatModel`). */
export interface SplatResult {
  /** Gaussians in the trained model. */
  count: number;
  /** Iterations trained (including those of a resumed checkpoint). */
  iterations: number;
  /** Wall time of this call (ms). */
  ms: number;
  shDegree: 0 | 1;
  /** Bytes of the packed model (`SplatModelPayload` arrays). */
  bytes: number;
  /** Trained with the Mip-Splatting opacity compensation (a renderer must apply it). */
  antialiased: boolean;
  /**
   * Well-known keys: splatViews, splatWidth / splatHeight, splatMaxGaussians, splatInitGaussians,
   * splatShell, splatItPerSec, splatLoss, splatResumedFrom, splatMemPeakMB, splatGpuMB,
   * splatDegraded, splatTrainMs, splatPrepareMs, splatStoredBytes, splatIterations (the
   * schedule), splatPreviews and splatPreviewFinalPsnr (with `preview`: the trained model from the
   * preview view, the PSNR the previews converge to). Ended by `finishEarly`: splatFinishedEarly = 1
   * and, when the checkpoint at that iteration was kept for a later `splat` to continue from,
   * splatContinueFrom (that iteration) and splatContinueKey (its run): loadSession then reports
   * splatResumableIteration and, while that checkpoint is still the kept one, splatContinuesModel = 1.
   */
  stats: Record<string, number | string>;
}

/**
 * The trained model for a viewer (`splatModel`; the arrays are transferred). World frame of the
 * sparse model (+Z up, metres when metric, ARCHITECTURE §2). Per Gaussian i (src/splat/types.ts):
 * mean `means[3i..]`; scale σ = exp(logScales[3i..]) along its local axes; rotation = the
 * quaternion [w, x, y, z] `quats[4i..]` (unnormalised; local → world, Σ = R S² Rᵀ); opacity =
 * sigmoid(opacities[i]); sRGB colour = max(0, SH_C0·f0 + 0.5) for SH degree 0 (`sh[3i..]`; degree 1
 * adds SH_C1·(−y f1 + z f2 − x f3) with (x, y, z) the unit camera → mean direction). `antialiased`:
 * multiply the opacity by √(det Σ₂ / det(Σ₂ + 0.3 I)) after adding 0.3 px² to the projected 2-D
 * covariance Σ₂ (Mip-Splatting), as the trainer did. Colours are already appearance-normalised.
 */
export interface SplatModelPayload {
  model: SplatModel;
  antialiased: boolean;
  /** Training background (sRGB); viewers usually draw their own. */
  background: [number, number, number];
  iterations: number;
}

export interface SessionSummary {
  id: string;
  name: string;
  mode: CaptureMode;
  quality: QualityPreset | null;
  status: SessionStatus;
  numFrames: number;
  /** Epoch ms. */
  createdAt: number;
  updatedAt: number;
  /** Stored models (null when not built yet). */
  sparse: SparseResult | null;
  dense: PointCloud | null;
  mesh: TriangleMesh | null;
  /** Extension: the stored splat model's info (fetch the model with `splatModel`). */
  splat?: SplatResult | null;
  stats: Record<string, number | string>;
}

// ---------------------------------------------------------------------------------------------
// RPC envelopes (ENGINE.md §3)
// ---------------------------------------------------------------------------------------------

export interface EngineMethodMap {
  init: { params: EngineOptions; result: InitResult };
  frame: { params: FrameInput; result: LiveUpdate };
  finishCapture: { params: undefined; result: void };
  refine: { params: undefined; result: SparseResult };
  densify: { params: DenseOptions | undefined; result: PointCloud };
  mesh: { params: MeshOptions | undefined; result: TriangleMesh };
  /** Extension: train Gaussian splats (after densify, or after refine without a dense cloud). */
  splat: { params: SplatOptions | undefined; result: SplatResult };
  /** Extension: the stored splat model (packed, transferred), null when none was trained. */
  splatModel: { params: undefined; result: SplatModelPayload | null };
  export: { params: { format: ExportFormat }; result: Blob };
  /** May be called without `init`: the engine is created from the stored session. */
  loadSession: { params: { id: string; options?: Partial<EngineOptions> }; result: SessionSummary };
  /** Cancel the request `target` (default: the running long operation). Resolves at once. */
  cancel: { params: { target?: number } | undefined; result: void };
  /**
   * Extension: end the request `target` (default: the running long operation) early and keep its
   * result — the splat training stops at its current iteration (from `SplatTrainingDetail.finishFrom`
   * on) and finishes like a complete run. Others ignore it. Resolves at once.
   */
  finishEarly: { params: { target?: number } | undefined; result: void };
  /** Extension: page visibility (pause GPU submissions while hidden). */
  visibility: { params: { hidden: boolean }; result: void };
  /**
   * Extension: let go of session `sessionId` before it is deleted — its engine is disposed (no
   * more writes to it), the compiled feature pipeline is kept for the next init. Result: whether
   * the engine held that session.
   */
  close: { params: { sessionId: string }; result: boolean };
  dispose: { params: undefined; result: void };
}

export type EngineMethod = keyof EngineMethodMap;
export type EngineParams<M extends EngineMethod> = EngineMethodMap[M]['params'];
export type EngineResult<M extends EngineMethod> = EngineMethodMap[M]['result'];

export const ENGINE_METHODS: readonly EngineMethod[] = [
  'init',
  'frame',
  'finishCapture',
  'refine',
  'densify',
  'mesh',
  'splat',
  'splatModel',
  'export',
  'loadSession',
  'cancel',
  'finishEarly',
  'visibility',
  'close',
  'dispose',
];

/** Long-running methods that report `progress` events and can be cancelled. */
export const LONG_RUNNING_METHODS: readonly EngineMethod[] = ['refine', 'densify', 'mesh', 'splat', 'export', 'loadSession', 'finishCapture'];

export interface EngineRequest<M extends EngineMethod = EngineMethod> {
  id: number;
  method: M;
  params: EngineParams<M>;
}

/** Well-known error names (Error.name survives the round trip). */
export const ENGINE_ERRORS = {
  cancelled: 'CancelledError',
  notImplemented: 'NotImplementedError',
  crashed: 'EngineCrashedError',
  timeout: 'EngineTimeoutError',
  busy: 'EngineBusyError',
  notInitialized: 'EngineNotInitializedError',
  gpuLost: 'GPUDeviceLostError',
  /** A write to the session store failed for a reason other than quota (code 'storage'). */
  storageWrite: 'StorageWriteError',
} as const;

export interface SerializedError {
  name: string;
  message: string;
  stack?: string;
  /** Optional machine-readable code (e.g. 'quota', 'storage', 'gpu-lost'). */
  code?: string;
}

export interface EngineSuccess<M extends EngineMethod = EngineMethod> {
  id: number;
  ok: true;
  result: EngineResult<M>;
}

export interface EngineFailure {
  id: number;
  ok: false;
  error: SerializedError;
}

export type EngineResponse<M extends EngineMethod = EngineMethod> = EngineSuccess<M> | EngineFailure;

/** Progress of a long operation (`requestId` = the request it belongs to, if any). */
export interface ProgressMessage extends ProgressEvent {
  type: 'progress';
  requestId?: number;
}

/** Unsolicited live update (e.g. background mapping finished); frame responses are NOT repeated as events. */
export interface LiveMessage {
  type: 'live';
  update: LiveUpdate;
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogMessage {
  type: 'log';
  level: LogLevel;
  message: string;
  data?: unknown;
}

export type EngineErrorReason = 'gpu-lost' | 'out-of-memory' | 'internal' | 'storage';

/**
 * Error not tied to a request. `fatal: true` means the worker cannot continue (e.g. the GPU
 * device was lost and `requestAdapter()` returns null in this realm — ios-webgpu.md §5): the
 * client terminates and respawns it.
 */
export interface ErrorMessage {
  type: 'error';
  error: SerializedError;
  fatal: boolean;
  reason?: EngineErrorReason;
}

export type EngineEvent = ProgressMessage | LiveMessage | LogMessage | ErrorMessage;
export type EngineEventType = EngineEvent['type'];
export type WorkerToMainMessage = EngineResponse | EngineEvent;
export type MainToWorkerMessage = EngineRequest;

const EVENT_TYPES = new Set<string>(['progress', 'live', 'log', 'error']);

export function isEngineEvent(m: unknown): m is EngineEvent {
  return !!m && typeof m === 'object' && typeof (m as { type?: unknown }).type === 'string' && EVENT_TYPES.has((m as { type: string }).type);
}

export function isEngineResponse(m: unknown): m is EngineResponse {
  if (!m || typeof m !== 'object') return false;
  const r = m as { id?: unknown; ok?: unknown; type?: unknown };
  return typeof r.id === 'number' && typeof r.ok === 'boolean' && r.type === undefined;
}

export function isEngineRequest(m: unknown): m is EngineRequest {
  if (!m || typeof m !== 'object') return false;
  const r = m as { id?: unknown; method?: unknown };
  return typeof r.id === 'number' && typeof r.method === 'string' && (ENGINE_METHODS as readonly string[]).includes(r.method);
}

/** Error → plain object (Error instances lose their name/stack in some structured clones). */
export function serializeError(e: unknown): SerializedError {
  if (e && typeof e === 'object') {
    const o = e as { name?: unknown; message?: unknown; stack?: unknown; code?: unknown };
    const out: SerializedError = {
      name: typeof o.name === 'string' && o.name ? o.name : 'Error',
      message: typeof o.message === 'string' ? o.message : String(e),
    };
    if (typeof o.stack === 'string') out.stack = o.stack;
    if (typeof o.code === 'string') out.code = o.code;
    return out;
  }
  return { name: 'Error', message: String(e) };
}

/** Plain object → Error with the original name (and `code`, when present). */
export function deserializeError(s: SerializedError | undefined | null): Error & { code?: string } {
  const err = new Error(s?.message ?? 'Unknown engine error') as Error & { code?: string };
  err.name = s?.name || 'Error';
  if (s?.stack) err.stack = s.stack;
  if (s?.code) err.code = s.code;
  return err;
}

/** An Error with a given name (e.g. ENGINE_ERRORS.cancelled). */
export function namedError(name: string, message: string, code?: string): Error & { code?: string } {
  const err = new Error(message) as Error & { code?: string };
  err.name = name;
  if (code) err.code = code;
  return err;
}

export function isCancelledError(e: unknown): boolean {
  return !!e && typeof e === 'object' && ((e as { name?: string }).name === ENGINE_ERRORS.cancelled || (e as { name?: string }).name === 'AbortError');
}

// ---------------------------------------------------------------------------------------------
// Transfer lists
// ---------------------------------------------------------------------------------------------

export interface CollectTransferablesOptions {
  /**
   * ArrayBuffers smaller than this are cloned instead of transferred (default 1024 bytes):
   * poses, intrinsics and other tiny arrays stay usable on the sending side.
   */
  minByteLength?: number;
  /** Maximum object nesting that is searched (default 8). */
  maxDepth?: number;
  /**
   * Views that cover only PART of their ArrayBuffer (`pool.subarray(0, n)`, several arrays in
   * one allocation) are NOT transferred by default: transferring detaches the whole buffer, so a
   * sender that keeps using its pool (a mapper's growing point buffer, a reused scratch arena)
   * would silently see length-0 arrays afterwards. They are structured-cloned instead (which
   * copies the whole underlying buffer — `slice()` such views before sending to copy only the
   * view). `true` transfers them anyway when the sender gives the whole buffer up.
   */
  transferPartialViews?: boolean;
}

type Ctor = abstract new (...args: never[]) => unknown;

function ctor(name: string): Ctor | null {
  const c = (globalThis as unknown as Record<string, unknown>)[name];
  return typeof c === 'function' ? (c as Ctor) : null;
}

/**
 * Walk `value` (arrays, plain objects, Maps, Sets) and return the transfer list for
 * `postMessage`: the ArrayBuffers behind typed arrays / DataViews (each buffer once, ≥
 * `minByteLength`, only views spanning their whole buffer unless `transferPartialViews`,
 * never SharedArrayBuffers or detached buffers) and ImageBitmaps,
 * OffscreenCanvases, VideoFrames and MessagePorts. Cycles are handled.
 */
export function collectTransferables(value: unknown, opts: CollectTransferablesOptions = {}): Transferable[] {
  const minBytes = opts.minByteLength ?? 1024;
  const maxDepth = opts.maxDepth ?? 8;
  const partial = opts.transferPartialViews ?? false;
  const out: Transferable[] = [];
  const seen = new Set<unknown>();
  const transferableCtors = ['ImageBitmap', 'OffscreenCanvas', 'VideoFrame', 'MessagePort'].map(ctor).filter((c): c is Ctor => !!c);
  const SAB = ctor('SharedArrayBuffer');
  const BlobCtor = ctor('Blob');

  const addBuffer = (buf: ArrayBufferLike): void => {
    if (SAB && buf instanceof SAB) return;
    if (!(buf instanceof ArrayBuffer)) return;
    if (buf.byteLength === 0 || buf.byteLength < minBytes) return; // detached buffers have byteLength 0
    if (seen.has(buf)) return;
    seen.add(buf);
    out.push(buf);
  };

  const visit = (v: unknown, depth: number): void => {
    if (v === null || typeof v !== 'object') return;
    if (ArrayBuffer.isView(v)) {
      if (!partial && (v.byteOffset !== 0 || v.byteLength !== v.buffer.byteLength)) return;
      addBuffer(v.buffer);
      return;
    }
    if (v instanceof ArrayBuffer) {
      addBuffer(v);
      return;
    }
    for (const C of transferableCtors) {
      if (v instanceof C) {
        if (!seen.has(v)) {
          seen.add(v);
          out.push(v as Transferable);
        }
        return;
      }
    }
    if (BlobCtor && v instanceof BlobCtor) return; // cloned by reference
    if (depth >= maxDepth || seen.has(v)) return;
    seen.add(v);
    if (Array.isArray(v)) {
      for (const x of v) visit(x, depth + 1);
      return;
    }
    if (v instanceof Map) {
      for (const x of v.values()) visit(x, depth + 1);
      return;
    }
    if (v instanceof Set) {
      for (const x of v) visit(x, depth + 1);
      return;
    }
    if (Object.prototype.toString.call(v) !== '[object Object]') return; // Date, RegExp, Error, …
    for (const k of Object.keys(v)) visit((v as Record<string, unknown>)[k], depth + 1);
  };

  visit(value, 0);
  return out;
}
