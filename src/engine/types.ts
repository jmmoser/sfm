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
import type { GyroToleranceOptions } from '../sfm/gyro';
import type { KeyframeSelectorOptions } from '../sfm/keyframes';
import type { RollingShutterDevice, RollingShutterEstimateOptions } from '../sfm/rolling-shutter';
import type { LiveMapperOptions } from '../sfm/live';
import type { MapperOptions } from '../sfm/mapper';
import type { SplatTrainer, SplatTrainerOptions } from '../splat/trainer';
import type { LiveUpdate, LogLevel, SparseCamera, SparseResult } from '../workers/protocol';
import type { EngineStore } from './persistence';
import type { DensePresetConfig, SplatPresetConfig } from './presets';

/**
 * `SparseCamera` plus the capture time and the source frame index (FrameInput.index, or the
 * engine's own 0-based frame counter when the caller did not set `index`). Structured clones
 * keep the extra fields, so they also arrive through the worker / EngineClient.
 */
export type SparseCameraEx = SparseCamera & {
  /** Capture time of the keyframe (ms, FrameInput.t on the session time base: + its segment's shift). */
  t?: number;
  /** FrameInput.index of the frame that became this keyframe (+ its segment's index offset). */
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
   * SIFT produce the same features (docs/research/features-gpu.md); 'cpu' is for test runs on
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
  /**
   * Splat stage overrides: preset fields (iterations, maxGaussians, maxImageSize, maxViews,
   * denseInitFraction, shellCount, shDegree, checkpointEvery, chunk) and `trainer` options passed
   * to SplatTrainer (learning rates, MCMC, colour correction, seed, inFlight, …).
   */
  splat?: Partial<SplatPresetConfig> & { trainer?: Partial<SplatTrainerOptions> };
  /** TEST-ONLY (inline engine; functions do not cross the worker boundary): called with every trainer the splat stage creates. */
  splatDebug?: { onTrainer?(trainer: SplatTrainer): void };
  /**
   * Diagnostics: time the PatchMatch compute passes with GPU timestamp queries (when the adapter
   * has 'timestamp-query'); the dense timing log then reports GPU time. Default false.
   */
  denseGpuTimestamps?: boolean;
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
   * Gyro consistency gates (default true, active once R_cam_device is verified): verified pairs,
   * registrations and initial pairs whose rotation disagrees with the integrated gyro are rejected
   * (live and refine), and refine prefers the model without gyro breaks (src/sfm/gyro.ts).
   */
  gyroGate?: boolean;
  /** Tolerance of the gyro gates (defaults: 5° + 0.3°/s × |Δt| + 5 % of the rotation). */
  gyroTolerance?: Partial<GyroToleranceOptions>;
  /**
   * Gyro-aided registration (active once R_cam_device is verified; independent of `gyroGate`): an
   * image P3P cannot register (fewer than 30 inliers, or a pose the gyro rejects) is registered with
   * its rotation predicted by the gyro from the registered keyframe nearest in time, from ≥ 15
   * inliers and a baseline that agrees with its best verified pair — refused without such a pair
   * (`LiveSession.predictRotation`, `MapperOptions.predictRotation`).
   *  - 'offline' (default): only in refine's offline SfM, after P3P failed on every candidate. It
   *    joins what the phone-rate live map holds but P3P alone splits into fragments (e2e yard-real
   *    at stride 3 from frame 1: 1 model of 48 keyframes instead of 3, the largest 32), so that an
   *    open-loop live model can be replaced; the model choice counts such registrations only for a
   *    model without an unclosed loop (`unclosedLoop`).
   *  - 'all': also live and in refine's retry of the pending keyframes, after LiveMapper
   *    re-triangulated the registered neighbours (`live.retriangulateNeighbours`) and P3P still
   *    failed; a live model with such keyframes is checked by an offline SfM WITHOUT the gyro path
   *    (`liveModelHealth`). Opt-in: it registered no keyframe live in any e2e run (re-triangulation
   *    gives P3P its points), and nothing checks it against translational aliasing (false matches on
   *    structure repeating along the walking direction shift the centre along the baseline).
   *  - 'off': never.
   * Measurements: docs/research/real-capture-robustness.md "Phone-rate capture".
   */
  gyroAidedRegistration?: 'off' | 'offline' | 'all';
  /**
   * Shift the frames of a restored / extended capture past the stored capture times (default true:
   * `LiveSession` time segments, EngineStateRecord.timeSegments). false only in tests that prove
   * the hazard (a reloaded page's performance.now() interleaves new IMU samples with stored ones).
   */
  extendTimeShift?: boolean;
  /**
   * Estimate the gyro bias / camera–IMU offset from the live keyframes every 10 registrations
   * during capture (default false; refine always calibrates once before the rolling-shutter estimation).
   * Off because it changes the keyframe selection and measured equal or worse (yard-real ATE 7.0e-4
   * vs 3.3e-4; docs/research/real-capture-robustness.md).
   */
  liveGyroCalibration?: boolean;
  /**
   * Rolling shutter estimated in refine (default true, needs the gyro): readout axis / direction /
   * time from the reconstruction, keypoints corrected when it lowers the reprojection cost
   * significantly (src/sfm/rolling-shutter.ts, src/engine/refine-camera.ts).
   */
  estimateRollingShutter?: boolean;
  /** Options of the rolling-shutter estimation (search range, acceptance threshold, …). */
  rollingShutterOptions?: Partial<RollingShutterEstimateOptions> & {
    minImages?: number;
    maxImages?: number;
    minOmega?: number;
    maxObsPerImage?: number;
    refineFocal?: boolean;
    refinePrincipalPoint?: boolean;
    /**
     * Translational term of the correction (default true): camera velocity from the neighbouring
     * keyframes × inverse depth of each keypoint's 3-D point (needed on close-range orbits).
     */
    translation?: boolean;
    /** Estimation passes, each on the model re-adjusted with the previous estimate (default 2). */
    passes?: number;
  };
  /**
   * Readout of this device, known in advance (e.g. estimated by an earlier capture: stats
   * rsReadoutMs / rsSweepDevice): every keyframe's keypoints are corrected live and the motion gate
   * allows faster rotation (`rsMaxAngularSpeed`).
   */
  rollingShutter?: RollingShutterDevice | null;
  /** Motion gate (rad/s) while a known rolling shutter is corrected live (default 2.0). */
  rsMaxAngularSpeed?: number;
  /**
   * Model choice of refine (after gyro breaks and registered images): the live or offline model whose
   * share of explained verified correspondences is higher by more than this wins (default 0.015);
   * otherwise the lower mean reprojection error.
   */
  modelChoiceExplainedGap?: number;
  /**
   * The from-scratch offline SfM of refine, the independent check of the live model: 'auto'
   * (default) runs it only when the live model shows a symptom it has to be checked for (keyframes
   * not registered, no verified gyro, keyframes registered with the gyro-predicted rotation, gyro
   * breaks or outliers, few images, a low explained ratio, unmerged loop-closure pairs:
   * `liveModelHealth` in refine.ts); 'always' runs it every time.
   */
  refineOfflineSfM?: 'auto' | 'always';
  /**
   * CPU helper workers (nested in the engine worker) that verify refine's candidate pairs and
   * quantise its vocabulary in parallel, and in densify decode the next views and halo-filter the
   * final depth maps while the GPU runs (default: hardware threads − 2, at most 3; 0 = everything
   * on the engine thread). Same results either way.
   */
  cpuWorkers?: number;
  /**
   * CPU/GPU overlap of densify's depth-map stage (src/engine/dense.ts; default true): keyframe
   * decoding ahead and the final maps' halo filter on the CPU helper workers, and map writes in
   * flight, while the GPU computes the next views — within the memory plan's headroom. false =
   * serial (every CPU step on the engine thread between GPU passes). Bit-identical results.
   */
  denseOverlap?: boolean;
  /** Per-image focal lengths tied to the shared focal in the final refinement (default true; kept only if it helps). */
  perImageFocal?: boolean;
  /** Prior σ of the per-image focal (relative, default 0.005) and the required reprojection improvement (default 0.005). */
  perImageFocalSigma?: number;
  perImageFocalMinImprovement?: number;
  /** Minimum lag-1 autocorrelation (time order) of the per-image focal lengths (default 0.6: focus breathing is smooth). */
  perImageFocalMinAutocorrelation?: number;
  /** Maximum alternations of the shared-camera re-refinement without the breathing and the per-image focal BA (default 6; stops early once the principal point moves < 0.25 px). */
  perImageFocalRounds?: number;
  /**
   * Weak prior on the shared principal point during those rounds (σ in px; default 0 = off). Anchor
   * `perImageFocalPPAnchor`: 'input' = the shared principal point before the per-image focal
   * refinement (default), 'centre' = the image centre. Measured on object-real / abl-object-breath
   * (e2e/README "Principal-point prior"): it does not make the alternation converge (the per-round
   * contraction is unchanged), it only shifts where the round cap stops it — hence off.
   */
  perImageFocalPPSigmaPx?: number;
  perImageFocalPPAnchor?: 'input' | 'centre';
  /**
   * Back-pressure of 'background' mapping: a keyframe frame's response waits until at most this
   * many keyframes are queued or being mapped (default 2; -1 = never wait). Without it the live
   * model falls arbitrarily far behind when mapping is slower than the keyframe rate (CPU tier,
   * fast motion): the user then sees a stale model and stale guidance. Waiting slows the frame
   * response, so the FramePump simply delivers fewer frames.
   */
  maxMappingBacklog?: number;
  /**
   * Store back-pressure: a keyframe frame's response waits while more than this many MB of
   * keyframe writes (full-size canvas / JPEG, features, colours) are not yet in the store
   * (default: 3 % of the memory budget, at least 24 MB). Protects against a slow or stalled
   * IndexedDB piling up ≈ 12 MB per 1920×1440 keyframe.
   */
  maxPendingWriteMB?: number;
  /**
   * No keyframes are added once the estimated capture state exceeds this fraction of the memory
   * budget minus the 250 MB page reserve and 64 MB of feature GPU caches (default 0.45; refine
   * peaks at ≈ 1.9× the capture state). The engine emits a non-fatal `error` event with reason
   * 'out-of-memory' once.
   */
  captureMemoryFraction?: number;
}

/** What the engine knows about one keyframe (persisted with its JPEG as the frame `info`). */
export interface KeyframeRecord {
  id: number;
  /** FrameInput.index (or the engine's frame counter), + an offset in a continued capture's segment. */
  frameIndex: number;
  /** FrameInput.t on the session time base (+ the shift of its capture segment, EngineStateRecord.timeSegments). */
  t: number;
  /** Canonical full-resolution size. */
  width: number;
  height: number;
  /** FrameInfo (R_cam_device / gravityCam follow the verified camera←device mapping). */
  info: FrameInfo;
  /** Gravity DOWN in the DEVICE frame (to recompute gravityCam after a mapping change). */
  gravityDevice: [number, number, number] | null;
  headingAccuracy: number | null;
  /** North reference of info.heading ('magnetic' ⇒ declination applied in the world alignment; absent = 'true'). */
  headingNorth?: 'true' | 'magnetic';
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
  /**
   * Rolling shutter the keypoints were corrected for (refine), with each registered image's camera
   * angular velocity: the dense stage removes it from the images too (its poses are those of the
   * frame stamp, the pixels are not). Persisted in `sparse/meta`; absent = none.
   */
  rollingShutter?: SparseRollingShutter | null;
}

/** Rolling-shutter data of a sparse model (plain JSON). */
export interface SparseRollingShutter {
  axis: 'rows' | 'cols';
  /** Signed readout (ms) of the canonical image (src/sfm/rolling-shutter.ts). */
  readoutMs: number;
  /** Camera angular velocity (rad/s, camera frame) per image id at its frame time. */
  omega: Record<number, [number, number, number]>;
  /**
   * Translational term per image id: camera-frame velocity of the camera centre divided by the
   * image's median scene depth (1/s); absent = rotation only.
   */
  nu?: Record<number, [number, number, number]>;
}
