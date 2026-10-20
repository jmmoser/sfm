# Engine, worker protocol and app integration

Companion to `ARCHITECTURE.md`. Describes the orchestration layer that ties the
modules together, the worker message protocol and the UI flow.

## 1. Threads

```
main thread                                   engine worker (module worker)
───────────                                   ─────────────────────────────
CameraCapture ─ requestVideoFrameCallback
FramePump (≤10 Hz, back-pressure) ──frame──▶  PhotogrammetryEngine.processFrame
MotionSensor / Compass / GeoSensor              ├─ FeaturePipeline (WebGPU SIFT + matcher)
                                                ├─ KeyframeSelector
Viewer (three.js) ◀──live update───────────     ├─ FeatureDatabase / CorrespondenceGraph
UI screens ◀──progress / results──────────      ├─ LiveMapper (incremental SfM, local BA)
                                                ├─ SessionStore (IndexedDB: JPEGs, models)
                                                └─ refine / densify / mesh / splat / export
```

All heavy work runs in the worker: the UI thread only shows video, the viewer and
the UI. WebGPU is used from the worker. The engine class itself is
thread-agnostic so the test harness can run it on the main thread.

## 2. `PhotogrammetryEngine` (`src/engine/engine.ts`)

```ts
interface EngineOptions {
  mode: 'object' | 'scene' | 'auto';
  quality: 'fast' | 'balanced' | 'max';
  sessionId?: string;          // persist frames/models in IndexedDB under this session
  persist?: boolean;           // default true in the app, false in tests
  preferGPU?: boolean;         // default true
  hfovPriorDeg?: number;       // from CameraCapture (lens + aspect)
  resume?: boolean;            // re-open sessionId's persisted state (EngineClient after a worker respawn)
}

interface FrameInput {
  bitmap: ImageBitmap | RGBAImage;  // full resolution, any UI orientation
  t: number;                        // capture time, ms (performance.now domain)
  screenAngle: 0 | 90 | 180 | 270;  // θ of ARCHITECTURE §2 when the frame was captured
  imu: ImuSample[];                 // samples since the previous frame
  geo?: GeoFix | null;
  heading?: { heading: number; accuracy: number; north?: 'true' | 'magnetic' } | null;
  // + optional FramePump metadata: index, segment, gravityDevice, attitude,
  //   imuCalibrationVersion, timeSource (src/workers/protocol.ts)
}

interface LiveUpdate {
  frameIndex: number;
  decision: 'keyframe' | 'skip' | 'reject_blur' | 'reject_motion';
  guidance: 'good' | 'move_slower' | 'more_texture' | 'translate_more' | 'tracking_lost' | 'initializing';
  tracking: 'initializing' | 'tracking' | 'lost';
  numKeyframes: number;
  numRegistered: number;
  numPoints: number;
  /** Keypoints of the processed frame in [0,1] normalised image coords (for an overlay), decimated. */
  overlay?: Float32Array;
  /** Present when the sparse model changed (decimated to ≤ 50k points). */
  sparse?: { positions: Float32Array; colors: Uint8Array };
  cameras?: { pose: Pose; intrinsics: CameraIntrinsics }[];
  timings: Record<string, number>;
}

class PhotogrammetryEngine {
  constructor(opts: EngineOptions, config?: EngineConfig)   // EngineConfig: tuning knobs (harness / tests)
  init(): Promise<{ gpu: boolean; adapter?: string; fallbackReason?: string }>
  processFrame(f: FrameInput): Promise<LiveUpdate>
  finishCapture(progress?, cancel?): Promise<void>
  refine(progress?, cancel?): Promise<SparseResult>
  densify(opts?, progress?, cancel?): Promise<PointCloud>
  mesh(opts?, progress?, cancel?): Promise<TriangleMesh>
  splat(opts?: SplatOptions, progress?, cancel?): Promise<SplatResult>   // Gaussian splats (§2 "Splat")
  splatModel(): Promise<SplatModelPayload | null>                        // the stored splat model
  exportModel(format: 'ply-points' | 'ply-mesh' | 'glb' | 'colmap-zip' | 'ply-splat'): Promise<Blob>
  loadSession(id: string, options?: Partial<EngineOptions>): Promise<SessionSummary>
  dispose(): void
  // worker / harness hooks: setHidden, flush, recoverGPU, memoryReport, releaseResultMemory
}
```

Quality presets (effective values: `QUALITY_PRESETS` of `src/engine/presets.ts`, dense part from the
tuned `DENSE_PRESETS` of `src/dense/presets.ts`):

| preset | feature max side | max features | dense max side | views | sources | PM iterations (½ res → full res) | geometric passes × iterations |
|---|---|---|---|---|---|---|---|
| fast | 1024 | 2048 | 800 | 40 | 5 | 4 → 2 | 1 × 2 |
| balanced | 1600 | 4096 | 1280 | 80 | 8 | 6 → 2 | 1 × 3 |
| max | 2048 | 8192 | 1600 | 150 | 8 | 6 → 2 | 1 × 3 |

Splat stage (`SPLAT_PRESETS`, `EngineConfig.splat` / `SplatOptions` override; the memory plan may lower views,
Gaussians and image size):

| preset | iterations | max Gaussians | training long side | max views | dense seed | sky shell (scene) | checkpoint every |
|---|---|---|---|---|---|---|---|
| fast | 3 000 | 200 000 | 800 (= dense) | 120 | ½ of the cap | 20 000 | 500 it |
| balanced | 5 000 | 300 000 | 1024 (dense 1280) | 160 | ½ of the cap | 20 000 | 500 it |
| max | 7 000 | 500 000 | 1280 (dense 1600) | 200 | ½ of the cap | 30 000 | 500 it |

Mode affects keyframe thresholds, TSDF voxel size and depth-range margins; `auto`
picks object vs scene after initialisation from the scene depth / baseline.

### Frame flow (live)
1. Rotate to the session's canonical orientation (first frame's θ) when needed;
   `R_cam_device` for the session follows from θ₀.
2. Small RGBA copy (≤ 640 px) → sharpness; IMU angular speed + gyro rotation since the
   last keyframe → blur/motion gate (`KeyframeSelector.gate`; relaxed when the hard
   rotation limit forces a keyframe through a fast pan).
3. Features at the preset's working resolution (GPU path takes the ImageBitmap
   directly). Per-feature colours sampled from the small copy.
4. Match to the last keyframe, guided by the gyro rotation homography
   `H = K R_c K⁻¹` with `R_c = R_cam_device · R_{d(t)←d(t_kf)} · R_cam_deviceᵀ`;
   verify with two-view geometry; compute rotation-compensated parallax.
5. `KeyframeSelector` decision (soft triggers wait briefly for a sharper frame and keep
   the best frame of that window — best-of-window — which may become the keyframe on a
   later frame). On keyframe: JPEG to IndexedDB (async), add to DB, match against the
   previous 6 keyframes + retrieval candidates (+ spatial neighbours of the gyro-propagated
   pose of the last registered keyframe), verify, reject pairs whose rotation disagrees
   with the gyro, update the graph, `LiveMapper.addKeyframe` (registrations checked
   against the gyro too).

### Refine
Train/extend the vocabulary → candidate pairs (sequential 10, retrieval 15,
spatial 10) → match + verify (+ gyro gate) → refine the live model → gyro calibration
(bias, camera–IMU offset) → rolling-shutter estimation on the live model (gyro rotation +
translation from the neighbouring keyframes × point depth; 2 passes) and keypoint
correction → only when the live model is unhealthy (`liveModelHealth`: unregistered keyframes,
no verified gyro, gyro breaks / outliers, few explained correspondences, an unclosed loop),
`runIncrementalSfM` (gyro-checked registrations) and the better of live
vs offline (gyro breaks, then registered images, then explained correspondences, then
reprojection error; isolated gyro outliers deregistered) → final global BA (focal + principal point + k1 + k2) →
per-image focal lengths (focus breathing; kept only if they help) → world alignment
(`sfm/world.ts`) → persist. Pair verification, vocabulary quantisation and the rolling-shutter
hypotheses run on CPU helper workers (`EngineConfig.cpuWorkers`). Details: ARCHITECTURE §6
"Real-capture robustness" and "Refine performance".

### Dense / mesh
Views stream from IndexedDB through an LRU image cache; depth maps go through a
`DepthMapStore` with an LRU memory cache backed by IndexedDB to bound memory on
phones: the memory planner (`src/engine/memory.ts`) budgets 1.2 GB (600 / 850 MB on 4 / 6 GB
iPhones); iOS kills a tab at ~1.5 GB (soft) / 2 GB (hard), docs/research/ios-platform.md.
PatchMatch passes (multi-scale): photometric at ½ resolution for every view, a full-resolution
photometric pass initialised from it, then the geometric pass(es); full-resolution passes
alternate direction so the image LRU (sized up to every view when the plan allows) and
PatchMatchGPU's upload caches stay warm. A fused cloud above the preset's `maxPoints` is merged
with a footprint-adaptive LOD (cells ∝ each point's radius). Details: ARCHITECTURE §6 "Dense on a
real GPU".

### Splat
Gaussian splatting (`src/engine/splat.ts` over `src/splat`, ARCHITECTURE §6 "Splat stage in the engine"), after
densify — or after refine when there is no dense cloud. One long operation with three progress stages:

1. `splat_prepare`: training views = the registered keyframes with a stored photo, evenly spread to `maxViews`,
   decoded and built exactly like the dense views (`loadDenseViewImage` → `denseViewImage`: undistorted pinhole,
   rolling-shutter corrected) at the training size (≤ the dense size). Memory plan (`fitSplatPlan` in
   `src/engine/memory.ts`: the trainer's own `SplatTrainer.memoryEstimate` + initialisation / view-decode /
   checkpoint transients; views, then Gaussians, then image size lowered until it fits; logged as
   `splat memory plan (…)`). Before it, PatchMatchGPU, the feature caches, the capture state and (persistent
   sessions) the dense colour views are released. Initial Gaussians: the fused dense cloud reduced by the footprint
   LOD (`footprintDownsample`) to `denseInitFraction` × the cap, plus every sparse SfM point
   (`initFromPointCloud`: 3-NN scales, normal-oriented discs), plus for scene captures (or
   `SplatOptions.backgroundShell`) a far shell coloured from 160 px view thumbnails (`backgroundShell`, radius
   max(20 × camera extent, 3 × the 98th-percentile sparse-point distance)). Pipelines compiled
   (`SplatTrainer.create`), views uploaded one at a time (`setViewImage`). Training views that are dense views
   get an inverse-depth target from the dense stage's fusion-filtered depth map (½ size, 8 B per pixel,
   `setViewDepth`): the preset's trainer default `geometry` supervises the rendered depth during the first 30 % of
   the iterations (ARCHITECTURE §6 "Geometry supervision … off-trajectory views"; off without depth maps or with
   `EngineConfig.splat.trainer.geometry: false`).
2. `splat_training`: `trainer.step(chunk)` (25 iterations) under the engine's GPU lock, waiting while the page is
   hidden; `ProgressEvent.detail` = `SplatTrainingDetail` {iteration, iterations, loss, itPerSec, gaussians}.
   Cancel is checked between chunks. Every `checkpointEvery` iterations a checkpoint — f32 parameters, bf16 Adam
   moments, colour-correction state, plan — goes to the session store (persistent) or the scratch store
   (`splat/ckpt<0|1>/…` + `splat/progress`, two slots so a kill mid-write keeps the previous one); a cancel ≥ 100
   iterations after the last checkpoint writes one more. Dropped GPU work (`SplatWorkDroppedError`: iteration /
   relocation stamps missing) rolls the trainer back to the last checkpoint in place; a device loss rebuilds the
   trainer on the new device from it (twice; then `GPUDeviceLostError`, code 'gpu-lost').
3. `splat_finishing`: appearance normalisation (the mean training-view colour matrix folded into the colours),
   read-back, persisted compactly (`splat/model`: f32 means, f16 rest ≈ 34 B per Gaussian + `splat/meta`);
   checkpoints cleared.

A later `splat()` of the same run (same model, dense cloud, preset, options) resumes from the checkpoint with the
stored plan; `loadSession` reports `stats.splatResumableIteration` / `splatResumableQuality`. `refine` deletes the
splats and checkpoints (new poses); densify / mesh leave them. Without WebGPU (`preferGPU: false`, CPU tier) `splat`
rejects with `NotSupportedError` (code 'no-webgpu').

Measured (object-real, fast, worker engine, M4 / Metal): 60 s for 3 000 iterations / 200 k Gaussians on 90 views at
600×800 (prepare 12.6 s, 64.6 it/s), memory plan peak ≈ 811 MB; held-out novel views 31.2 dB full-frame PSNR vs 8.7 dB
for the lit mesh (ARCHITECTURE §6 "Splat stage in the engine", e2e/README.md).

## 3. Worker protocol (`src/workers/protocol.ts`)

Request/response RPC with ids, plus unsolicited events:

* requests: `init`, `frame`, `finishCapture`, `refine`, `densify`, `mesh`, `splat`, `splatModel`,
  `export`, `loadSession`, `cancel`, `visibility` (page hidden / visible: pause GPU work), `dispose`
* events: `progress` (ProgressEvent, optional numeric `detail`), `live` (LiveUpdate), `log`, `error`
* large typed arrays and ImageBitmaps are transferred, never copied.

### Splat protocol (for the viewer / app UI)

| call | params → result | notes |
|---|---|---|
| `splat` (long: progress, cancel, inactivity timeout) | `SplatOptions?` {quality, iterations, maxGaussians, maxImageSize, maxViews, backgroundShell, restart} → `SplatResult` {count, iterations, ms, shDegree, bytes, antialiased, stats} | stages `SPLAT_STAGES` = `splat_prepare` → `splat_training` → `splat_finishing`; training events carry `detail: SplatTrainingDetail`. `stats`: splatViews, splatWidth / splatHeight, splatMaxGaussians, splatInitGaussians (Dense / Sparse / Shell), splatItPerSec, splatLoss, splatResumedFrom, splatMemPeakMB, splatGpuMB, splatDegraded, splatPrepareMs / TrainMs / FinishMs, splatCheckpoints, splatStoredBytes. Cancel → `CancelledError`, the next call resumes (`restart: true` starts over). |
| `splatModel` (short) | — → `SplatModelPayload \| null` | {model: SplatModel, antialiased, background, iterations}; arrays freshly decoded and TRANSFERRED (≈ 56 B per Gaussian at SH 0). |
| `export` | `{format: 'ply-splat'}` → Blob | INRIA 3DGS PLY (`encodeSplatPLY`; SuperSplat / Brush / Spark), appearance-normalised, comment `antialiased`. |
| `loadSession` | → `SessionSummary.splat` (SplatResult of the stored model, or null) | `stats.splatResumableIteration` when an interrupted run can resume. |

`SplatModel` conventions (world frame of the sparse model: +Z up, metres when metric): per Gaussian i, mean
`means[3i..3i+2]`; σ = exp(`logScales[3i..]`) along its local axes; rotation = quaternion [w, x, y, z]
`quats[4i..]` (unnormalised, local → world: Σ = R diag(σ²) Rᵀ); opacity = sigmoid(`opacities[i]`); sRGB colour =
max(0, SH_C0 · `sh[3i..3i+2]` + 0.5) (SH degree 1 adds SH_C1 · (−y f1 + z f2 − x f3), (x, y, z) the unit camera →
mean direction; `sh` holds K = 1 or 4 RGB triplets per Gaussian). `antialiased` (always true today): add 0.3 px²
to the projected 2-D covariance and multiply the opacity by √(det Σ₂ / det(Σ₂ + 0.3 I)) (Mip-Splatting), like the
trainer. Colours are sRGB values (no transfer function on output); render premultiplied, front to back. The
MockEngine implements the same calls (a fast fake run whose model is one Gaussian per dense point).

## 4. UI flow (`src/app`)

Home (sessions, new scan, diagnostics) → Setup (mode, quality) → Permissions
(one tap: motion → camera → location) → Capture (video, guidance, live sparse
view, counters, finish) → Processing (stages + progress + cancel) → Viewer
(splats / points / mesh layer pill, cameras, EDL, measure, export, run dense / mesh / photoreal). The splat
stage runs from the viewer itself ("Photoreal model" in the Build sheet → `splat` with a progress card fed by
`SplatTrainingDetail`, Cancel, Resume after an interruption), the model is fetched with `splatModel` into the
viewer's WebGPU splat layer and shown by default, and the export sheet offers 'ply-splat' (ARCHITECTURE §6
"Splat layer in the viewer").
