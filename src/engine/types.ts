/**
 * Engine-internal types: tuning knobs (`EngineConfig`, harness / tests only), the extended
 * sparse camera (`SparseCameraEx`: capture time + source frame index), event hooks and the
 * keyframe record kept by the engine.
 *
 * The public API types (EngineOptions, FrameInput, LiveUpdate, SparseResult, …) live in
 * `src/workers/protocol.ts` — they are re-exported from `./index` for convenience.
 */
import type { CameraIntrinsics, FrameInfo, Pose } from '../core/types';
import type { FeaturePipeline } from '../features/extractor';
import type { SiftOptions } from '../features/sift-common';
import type { KeyframeSelectorOptions } from '../sfm/keyframes';
import type { LiveMapperOptions } from '../sfm/live';
import type { MapperOptions } from '../sfm/mapper';
import type { LiveUpdate, LogLevel, SparseCamera, SparseResult } from '../workers/protocol';
import type { EngineStore } from './persistence';
import type { DensePresetConfig } from './presets';

/**
 * `SparseCamera` plus the capture time and the source frame index (FrameInput.index, or the
 * engine's own 0-based frame counter when the caller did not set `index`). Structured clones
 * keep the extra fields, so they also arrive through the worker / EngineClient.
 */
export type SparseCameraEx = SparseCamera & {
  /** Capture time of the keyframe (ms, FrameInput.t). */
  t?: number;
  /** FrameInput.index of the frame that became this keyframe. */
  frameIndex?: number;
};

/** SparseResult whose cameras carry `t` / `frameIndex` (what the engine returns). */
export type SparseResultEx = Omit<SparseResult, 'cameras'> & { cameras: SparseCameraEx[] };

/** Per-call progress phase names used by the engine (match src/app/stages.ts STAGES ids). */
export type EngineStage = 'matching' | 'reconstruction' | 'alignment' | 'depth_maps' | 'fusion' | 'meshing';

/**
 * Optional engine tuning (second constructor argument; the worker accepts it as the
 * `engineConfig` extension field of the `init` / `loadSession` options). Every field is optional:
 * the defaults come from the quality preset and mode.
 */
export interface EngineConfig {
  /** Persistence backend (default: SessionStore when `persist` + `sessionId`, else memory). */
  store?: EngineStore;
  /** Use this feature pipeline instead of creating one (tests). */
  pipeline?: FeaturePipeline;
  /**
   * Feature backend: 'auto' (default: WebGPU SIFT / matcher when available, else CPU) or 'cpu'
   * (CPU SIFT / matching even when WebGPU exists; PatchMatch still uses the GPU). The GPU and CPU
   * SIFT produce the same features (ARCHITECTURE §6 features GPU notes); 'cpu' is for test runs on
   * software adapters (SwiftShader), where GPU SIFT is ~20× slower than the CPU code.
   */
  featureBackend?: 'auto' | 'cpu';
  /** SIFT option overrides (applied over the preset's maxImageSize / maxFeatures). */
  sift?: Partial<SiftOptions>;
  /**
   * 'background' (default): a keyframe's pair matching + mapping runs in a serial queue after
   * the frame response; the model update arrives as a `live` event. 'inline': processFrame
   * resolves after the keyframe was mapped (deterministic timing for tests / harness).
   */
  mapping?: 'background' | 'inline';
  /** KeyframeSelector overrides. */
  keyframe?: Partial<KeyframeSelectorOptions>;
  /** LiveMapper overrides. */
  live?: Partial<LiveMapperOptions>;
  /** Offline mapper (refine) overrides. */
  mapper?: Partial<MapperOptions>;
  /** Pairs matched per new keyframe during capture. */
  livePairs?: { sequential?: number; retrieval?: number; spatial?: number };
  /**
   * Refine the principal point in a final global BA of refine: true / false, or undefined =
   * when at least 20 images are registered (ENGINE_DEFAULTS.refinePrincipalPointMinImages).
   */
  refinePrincipalPoint?: boolean;
  /** Candidate pairs of refine (defaults 10 / 15 / 10). */
  refinePairs?: { sequential?: number; retrieval?: number; spatial?: number };
  /** Vocabulary: train after this many keyframes (default 25) with this many words (default 256 live, 512 refine). */
  vocabulary?: { trainAfter?: number; words?: number; refineWords?: number };
  /** Dense preset overrides. */
  dense?: Partial<DensePresetConfig>;
  /** Approximate memory budget of the tab in MB (default 1200, iPhone). */
  memoryBudgetMB?: number;
  /** Guided-matching search radius as a fraction of the image long side (default 0.08). */
  guidedRadiusFrac?: number;
  /** Max overlay keypoints per LiveUpdate (default 300). */
  overlayMaxPoints?: number;
  /** Frames within this many ms of a UI rotation are dropped (default 500). */
  settleMs?: number;
  /** Max points of LiveUpdate.sparse (default 50 000). */
  liveMaxPoints?: number;
  /** Run the R_cam_device hand-eye check (default true). */
  verifyCameraFromDevice?: boolean;
  /** RANSAC iterations of the per-frame verification against the last keyframe (default 400). */
  trackingMaxIterations?: number;
  /**
   * Back-pressure of 'background' mapping: a keyframe frame's response waits until at most this
   * many keyframes are queued or being mapped (default 2; -1 = never wait). Without it the live
   * model falls arbitrarily far behind when mapping is slower than the keyframe rate (CPU tier,
   * fast motion): the user then sees a stale model and stale guidance. Waiting slows the frame
   * response, so the FramePump simply delivers fewer frames.
   */
  maxMappingBacklog?: number;
}

/** What the engine knows about one keyframe (persisted with its JPEG as the frame `info`). */
export interface KeyframeRecord {
  id: number;
  /** FrameInput.index (or the engine's frame counter). */
  frameIndex: number;
  t: number;
  /** Canonical full-resolution size. */
  width: number;
  height: number;
  /** FrameInfo (R_cam_device / gravityCam follow the verified camera←device mapping). */
  info: FrameInfo;
  /** Gravity DOWN in the DEVICE frame (to recompute gravityCam after a mapping change). */
  gravityDevice: [number, number, number] | null;
  headingAccuracy: number | null;
  segment: number;
  /** UI angle θ the frame was captured at and the quarter turns applied to make it canonical. */
  screenAngle: number;
  quarterTurns: number;
}

/** Event hooks of PhotogrammetryEngine (the worker forwards them as protocol events). */
export interface EngineEvents {
  live: (u: LiveUpdate) => void;
  log: (level: LogLevel, message: string, data?: unknown) => void;
  /** Non-request errors; `fatal` ⇒ the engine cannot continue (e.g. 'gpu-lost'). */
  error: (err: Error, fatal: boolean, reason?: 'gpu-lost' | 'out-of-memory' | 'internal' | 'storage') => void;
}

/** Final sparse model in plain arrays (independent of Reconstruction; persisted, used by dense / export). */
export interface SparseModel {
  cameras: Map<number, CameraIntrinsics>;
  /** Registered images in capture order. */
  images: { id: number; cameraId: number; pose: Pose }[];
  numPoints: number;
  /** 3 per point (world frame, float64). */
  xyz: Float64Array;
  /** 3 per point, sRGB. */
  colors: Uint8Array;
  /** Mean reprojection error per point (px). */
  errors: Float32Array;
  /** CSR tracks: elements trackOffsets[i] … trackOffsets[i+1]−1 of trackImages / trackFeatures. */
  trackOffsets: Uint32Array;
  trackImages: Uint32Array;
  trackFeatures: Uint32Array;
}
