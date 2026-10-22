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

With the developer option "Record raw capture" a third worker records what the engine gets
(§5): `src/workers/recorder.worker.ts` (JPEG encode + OPFS / IndexedDB writes), fed by a FrameTap
on the main thread.

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
  extend?: boolean;            // with resume: continue the stored capture, also a finished one ("Add more views")
}

interface FrameInput {
  bitmap: ImageBitmap | RGBAImage | VideoFrame;  // full resolution, any UI orientation; a VideoFrame
                                    //   (default on iOS) is converted + downscaled in the worker, then closed
  t: number;                        // capture time, ms (performance.now domain)
  screenAngle: 0 | 90 | 180 | 270;  // θ of ARCHITECTURE §2 when the frame was captured
  screenAngleSince?: number;        // performance.now() time θ took that value (FramePump); the 500 ms
                                    //   settle window starts there, else at the first frame with the new θ
  imu: ImuSample[];                 // samples since the previous frame
  geo?: GeoFix | null;
  heading?: { heading: number; accuracy: number; north?: 'true' | 'magnetic' } | null;
  // + optional FramePump metadata: index, segment, gravityDevice, attitude,
  //   imuCalibrationVersion, timeSource (src/workers/protocol.ts)
}

interface LiveUpdate {
  frameIndex: number;
  decision: 'keyframe' | 'skip' | 'reject_blur' | 'reject_motion';
  guidance: 'good' | 'move_slower' | 'more_texture' | 'translate_more' | 'tracking_lost' | 'initializing'
    | 'too_dark';                   // median luma of the last 5 frames < 40, or < 70 with 3 blur rejects
  tracking: 'initializing' | 'tracking' | 'lost';
  /** Mode in effect: the session's mode, or what 'auto' resolved to ('auto' until then). */
  resolvedMode?: 'object' | 'scene' | 'auto';
  numKeyframes: number;
  numRegistered: number;
  numPoints: number;
  /** Keypoints of the processed frame in [0,1] normalised image coords (for an overlay), decimated. */
  overlay?: Float32Array;
  /** Present when the sparse model changed (decimated to ≤ 50k points). */
  sparse?: { positions: Float32Array; colors: Uint8Array };
  /** upCam: physical up in the camera frame (−gravityCam, else R_cam_device · up for the keyframe's UI
   *  rotation); the final cameras (SparseResult of refine / loadSession) carry the same upCam. */
  cameras?: { pose: Pose; intrinsics: CameraIntrinsics; t?: number; upCam?: [number, number, number] }[];
  timings: Record<string, number>;
}

class PhotogrammetryEngine {
  constructor(opts: EngineOptions, config?: EngineConfig)   // EngineConfig: tuning knobs (harness / tests)
  init(): Promise<InitResult>   // { gpu, adapter?, fallbackReason?, resumedKeyframes?, captureMemory?: { usedMB, capMB }, storedModel? }
  processFrame(f: FrameInput): Promise<LiveUpdate>
  finishCapture(progress?, cancel?): Promise<void>
  refine(progress?, cancel?): Promise<SparseResult>
  densify(opts?, progress?, cancel?): Promise<PointCloud>
  mesh(opts?, progress?, cancel?): Promise<MeshResult>   // TriangleMesh + objectRoi (object box or null) + stats
  splat(opts?: SplatOptions, progress?, cancel?): Promise<SplatResult>   // Gaussian splats (§2 "Splat")
  splatModel(): Promise<SplatModelPayload | null>                        // the stored splat model
  exportModel(format: 'ply-points' | 'ply-mesh' | 'glb' | 'colmap-zip' | 'ply-splat' | 'usdz' | 'spz', cancel?): Promise<Blob>
  loadSession(id: string, options?: Partial<EngineOptions>): Promise<SessionSummary>
  dispose(): void
  // worker / harness hooks: setHidden, flush, recoverGPU, memoryReport, releaseResultMemory,
  //   releasePipeline / adoptPipeline
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
picks object vs scene after initialisation from the scene depth / baseline (`classifyScene`: the
optical-axis convergence of ≥ 8 registered keyframes) and reports it as `LiveUpdate.resolvedMode`.

### Frame flow (live)
1. Rotate to the session's canonical orientation (first frame's θ) when needed;
   `R_cam_device` for the session follows from θ₀.
2. Small RGBA copy (≤ 640 px) → sharpness and luma → `LowLightMonitor` ('too_dark', which
   overrides 'tracking_lost'); IMU angular speed + gyro rotation since the last keyframe →
   blur/motion gate (`KeyframeSelector.gate`; relaxed when the hard rotation limit forces a
   keyframe through a fast pan).
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

### Extend (continue a stored capture)
`init` with `{ sessionId, persist: true, resume: true, extend: true }` restores the capture (keyframes,
features, pairs, vocabulary, IMU, GPS, θ₀ / R_cam_device, time offset, gyro bias; the live model is
rebuilt in the mapping queue) and reopens it (`LiveSession.reopenCapture`), also after finishCapture /
refine: tracking restarts, so the first sharp frame becomes a keyframe that relocalises through the
mapping queue's sequential, retrieval and spatial candidates. A missing or unreadable stored state, or
a failed restore, rejects the init before anything is written (a scan is never continued from
scratch). The stored mode and quality are kept; only the sparse model is loaded (the dense cloud /
mesh stay in the store, outside the capture's memory cap, which applies from the first frame:
`InitResult.captureMemory`) and previewed (`InitResult.storedModel`, decimated, no camera times).
The live model rebuild is one mapping job (≈ 190 ms per keyframe on SwiftShader / CPU features — a
minute for a large scan on a phone is plausible): frames are not held back by it — the back-pressure
counts keyframe jobs only and never waits on it, at most `maxMappingBacklog` new keyframes queue
behind it (later keyframe decisions are skipped, `timings.rebuildLimited`), and every LiveUpdate
reports its progress (`modelRebuild: { done, total }`). Times: performance.now() restarts with
every page load, so the engine keeps a capture on its SESSION time base — FrameInput time + the capture
segment's shift (`EngineStateRecord.timeSegments`). The first frame of a continued capture starts a
segment ≥ 10 s (`EXTEND_TIME_GAP_MS`) after everything stored (frame, IMU, GPS and attitude times shifted
alike; lower frame indices offset): the gyro never integrates across the gap, the inertial scale splits
its segments there and the gravity skips the gap (`estimateGravity`). The segment is decided by the
frame's own streams (frame, IMU, attitude), not its GPS fix, and a fix at or before the stored capture's
end is dropped (the FramePump re-attaches the nearest fix for up to 30 s: after a same-page respawn often
a stored one). LiveUpdate times stay on the page's base; model updates sent before the first frame
decided the segment carry no keyframe times. The first new keyframe marks the stored
results stale (`resultsStale`: `loadSession` then opens the scan with its sparse model only, status
'capturing' / 'captured') until the next `refine`, which covers every keyframe and invalidates the dense
cloud, mesh and splats as always (a cancelled or failed refine keeps the previous model). No multi-map
merging: refine keeps the largest offline model and reports the keyframes outside it (`unconnectedImages`,
`offlineModels`, `offlineDroppedImages`). Measurements: research/extend-scan.md.

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
hypotheses run on CPU helper workers (`EngineConfig.cpuWorkers`). Details: research/real-capture-robustness.md
and research/refine-performance.md.

### Dense / mesh
Views stream from IndexedDB through an LRU image cache; depth maps go through a
`DepthMapStore` with an LRU memory cache backed by IndexedDB to bound memory on
phones: the memory planner (`src/engine/memory.ts`) budgets 1.2 GB (600 / 850 MB on 4 / 6 GB
iPhones); iOS kills a tab at ~1.5 GB (soft) / 2 GB (hard), docs/research/ios-platform.md.
PatchMatch passes (multi-scale): photometric at ½ resolution for every view, a full-resolution
photometric pass initialised from it, then the geometric pass(es); full-resolution passes
alternate direction so the image LRU (sized up to every view when the plan allows) and
PatchMatchGPU's upload caches stay warm. A fused cloud above the preset's `maxPoints` is merged
with a footprint-adaptive LOD (cells ∝ each point's radius). Details: research/dense-tuning.md "Dense on a
real GPU".

Mesh: TSDF over the fusion-filtered depth maps (2 × median sparse depth cut in object mode). In object mode
with a gravity-aligned model (`MeshOptions.objectRoi` not false) the TSDF is bounded to the automatic object
box (`proposeObjectRoi`, src/dense/object-roi.ts: optical-axis convergence centre, support plane from a
z-histogram of the sparse points in an annulus around the orbit axis, object = points above the plane within
0.6 × the camera ring, without weak and isolated points), so the table and room stay out of the mesh, the block
budget goes to the object and the auto voxel follows the object's depth. No box (not gravity-aligned, no orbit,
no plane, hand-held, fewer than 100 object points or confidence below 0.3): the whole scene within the depth cut,
as before. The dense cloud and splats are never cropped. The app has no "mesh the full scene" action yet
(`MeshOptions.objectRoi: false` is used by the e2e `--no-object-roi` baseline only). The box is returned as
`MeshResult.objectRoi`, persisted in `mesh/meta` (cleared with the mesh and by refine: it is in the refined
world frame), reported as `SessionSummary.objectRoi` and passed to exports as `ExportSources.objectRoi`, for
exports that will stand the object on the ground with its bottom centre `objectRoiPivot` at the origin (USDZ /
GLB for AR). No export reads it yet: GLB and PLY are written in world coordinates. Stats: `objectRoi` (the box,
`none: <reason>` or `off`), `objectRoiConfidence`, `voxelSize`, `tsdfAutoVoxelSize`, `tsdfCoarsenLevels`,
`tsdfBlocks`, `tsdfMaxBlocks`.

### Splat
Gaussian splatting (`src/engine/splat.ts` over `src/splat`, research/splat-tuning.md "Engine splat stage"), after
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
   the iterations (src/splat/trainer.ts, research/splat-tuning.md "Geometry supervision, shape regularisers, colour correction"; off without depth maps or with
   `EngineConfig.splat.trainer.geometry: false`).
2. `splat_training`: `trainer.step(chunk)` (25 iterations) under the engine's GPU lock, waiting while the page is
   hidden; `ProgressEvent.detail` = `SplatTrainingDetail` {iteration, iterations, loss, itPerSec, gaussians}.
   Cancel is checked between chunks. Every `checkpointEvery` iterations a checkpoint — f32 parameters, bf16 Adam
   moments, colour-correction state, plan — goes to the session store (persistent) or the scratch store
   (`splat/ckpt<0|1>/…` + `splat/progress`, two slots so a kill mid-write keeps the previous one); a cancel ≥ 100
   iterations after the last checkpoint writes one more. Dropped GPU work (`SplatWorkDroppedError`: iteration /
   relocation stamps missing) rolls the trainer back to the last checkpoint in place; a dropped readback (its own
   stamp missing) is read again once the page is visible, so a checkpoint or the saved model never holds stale
   staging data; a device loss — in a step, a checkpoint or the finishing readbacks — rebuilds the trainer on the
   new device from it (from the initial Gaussians before the first one) and trains on from there (twice; then
   `GPUDeviceLostError`, code 'gpu-lost'). Finish early (`CancelToken.finishEarly`, the `finishEarly` request,
   the app's "Use it now"): from `SplatTrainingDetail.finishFrom` on — the end of the coarse-to-fine resolution
   schedule, 30 % of the iterations; an earlier request waits for it — the loop stops between chunks, writes a
   checkpoint at that iteration and goes on to 3. like a complete run (stats `splatFinishedEarly`).
   `splat_preview` (with `SplatOptions.preview`): `trainer.render` of the training view nearest the middle
   registered image (the viewer's default view), box-downscaled by an integer factor to ≤ 400 px, as RGBA8 in
   `ProgressEvent.image` with `detail: SplatPreviewDetail` {iteration, iterations, viewId, psnr (against that
   photo, raw colours)}; once right after that view's upload (it goes first: the initial Gaussians, seconds
   before the first iteration) and then between chunks every ≥ 2 s. Forward only under the GPU lock (no new
   pipeline; it waits while the page is hidden), one ≤ 1.9 MB readback stamped like the others, in the forward's
   command buffer: work iOS drops as the page hides (`SplatWorkDroppedError`) or any other failure only skips
   that snapshot, never posts stale bytes. When the training ends the same view is rendered once more (not
   posted): `stats.splatPreviewFinalPsnr`, the PSNR the previews converge to.
3. `splat_finishing`: appearance normalisation (the mean training-view colour matrix folded into the colours),
   read-back, persisted compactly (`splat/model`: f32 means, f16 rest ≈ 34 B per Gaussian + `splat/meta`);
   checkpoints cleared — except after a finish early whose checkpoint at the model's iteration is stored
   (`stats.splatContinueFrom`, its run key `splatContinueKey`).

A later `splat()` of the same run (same model, dense cloud, preset, options) resumes from the checkpoint with the
stored plan — after a cancel / kill, or to continue a model ended early; `loadSession` reports
`stats.splatResumableIteration` / `splatResumableQuality`, and `splatContinuesModel = 1` while that checkpoint is
the one kept for the stored model ended early (same run key and iteration), not one of another, interrupted run.
`refine` deletes the splats and checkpoints (new poses); a completed densify deletes the checkpoints (the dense
cloud is part of the run key: they could not resume) and keeps the model; mesh leaves both. Without WebGPU
(`preferGPU: false`, CPU tier) `splat` rejects with `NotSupportedError` (code 'no-webgpu').

Measured (object-real, fast, worker engine, M4 / Metal): 60 s for 3 000 iterations / 200 k Gaussians on 90 views at
600×800 (prepare 12.6 s, 64.6 it/s), memory plan peak ≈ 811 MB; held-out novel views 31.2 dB full-frame PSNR vs 8.7 dB
for the lit mesh (research/splat-tuning.md "Engine splat stage", e2e/README.md).

## 3. Worker protocol (`src/workers/protocol.ts`)

Request/response RPC with ids, plus unsolicited events:

* requests: `init`, `frame`, `finishCapture`, `refine`, `densify`, `mesh`, `splat`, `splatModel`,
  `export`, `loadSession`, `cancel`, `finishEarly` (end a long operation early and keep its result: only
  the splat training reacts), `visibility` (page hidden / visible: pause GPU work), `close`
  (let go of a session before it is deleted: its engine is disposed and the CPU SIFT scratch freed,
  the compiled feature pipeline kept for the next init; answers whether the engine held it), `dispose`
* events: `progress` (ProgressEvent, optional numeric `detail`, optional RGBA8 `image` — never throttled,
  its buffer transferred), `live` (LiveUpdate), `log`, `error`.
  With background mapping (the default) the model (`sparse`, `cameras`) comes only in `live` model
  updates sent when a keyframe is mapped: no `overlay` / `framePose`, the `frameIndex` of the frame
  in flight and the previous frame's `guidance` — per-frame consumers use the `frame` responses.
* large typed arrays and ImageBitmaps are transferred, never copied.
* App side, not a worker message: `EngineLike.setFrameTap(tap)` (EngineClient and MockEngine,
  `src/app/frame-tap.ts`) lets an observer see every FrameInput just before it is sent — after
  the busy check, while the pixels are still the caller's — and the answer afterwards, with
  `sent: false` when the client rejected it before posting it to the worker (disposed, dead, no
  worker); the tap may hold back `processFrame`'s resolution (back-pressure), never the engine.

### Splat protocol (for the viewer / app UI)

| call | params → result | notes |
|---|---|---|
| `splat` (long: progress, cancel, inactivity timeout) | `SplatOptions?` {quality, iterations, maxGaussians, maxImageSize, maxViews, backgroundShell, restart, preview} → `SplatResult` {count, iterations, ms, shDegree, bytes, antialiased, stats} | stages `SPLAT_STAGES` = `splat_prepare` → `splat_training` → `splat_finishing`; training events carry `detail: SplatTrainingDetail` {iteration, iterations, loss, itPerSec, gaussians, finishFrom}; with `preview: true` `splat_preview` events (interleaved) carry `image` (RGBA8 ≤ 400 px) + `detail: SplatPreviewDetail`. `stats`: splatViews, splatWidth / splatHeight, splatMaxGaussians, splatInitGaussians (Dense / Sparse / Shell), splatItPerSec, splatLoss, splatResumedFrom, splatMemPeakMB, splatGpuMB, splatDegraded, splatPrepareMs / TrainMs / FinishMs, splatCheckpoints, splatStoredBytes, splatIterations (schedule), splatPreviews, splatPreviewFinalPsnr, splatFinishedEarly, splatContinueFrom, splatContinueKey. Cancel → `CancelledError`, the next call resumes (`restart: true` starts over). |
| `finishEarly` (short, fire and forget) | `{target?}` → — | the running `splat` stops at its current iteration (from `finishFrom`), resolves with that model (`iterations` < the schedule) and keeps its checkpoint: the next `splat` of the same settings continues from it. |
| `splatModel` (short) | — → `SplatModelPayload \| null` | {model: SplatModel, antialiased, background, iterations}; arrays freshly decoded and TRANSFERRED (≈ 56 B per Gaussian at SH 0). |
| `export` | `{format: 'ply-splat'}` → Blob | INRIA 3DGS PLY (`encodeSplatPLY`; SuperSplat / Brush / Spark), appearance-normalised, comment `antialiased`. |
| `export` | `{format: 'spz'}` → Blob | Niantic SPZ v3 (`encodeSplatSPZ`: one gzip stream from `CompressionStream`, RUB axes, flag 0x1 antialiased; ≈ 20 B per Gaussian before gzip at SH 0; SuperSplat / Spark / the reference spz library). |
| `loadSession` | → `SessionSummary.splat` (SplatResult of the stored model, or null) | `stats.splatResumableIteration` (+ `splatResumableQuality`) when a checkpoint can resume; `splatContinuesModel = 1` when it is the stored model's, ended early (it can continue). |

`SplatModel` conventions (world frame of the sparse model: +Z up, metres when metric): per Gaussian i, mean
`means[3i..3i+2]`; σ = exp(`logScales[3i..]`) along its local axes; rotation = quaternion [w, x, y, z]
`quats[4i..]` (unnormalised, local → world: Σ = R diag(σ²) Rᵀ); opacity = sigmoid(`opacities[i]`); sRGB colour =
max(0, SH_C0 · `sh[3i..3i+2]` + 0.5) (SH degree 1 adds SH_C1 · (−y f1 + z f2 − x f3), (x, y, z) the unit camera →
mean direction; `sh` holds K = 1 or 4 RGB triplets per Gaussian). `antialiased` (always true today): add 0.3 px²
to the projected 2-D covariance and multiply the opacity by √(det Σ₂ / det(Σ₂ + 0.3 I)) (Mip-Splatting), like the
trainer. Colours are sRGB values (no transfer function on output); render premultiplied, front to back. The
MockEngine implements the same calls (a fast fake run whose model is one Gaussian per dense point; synthetic
previews: the cloud projected into the middle camera, coarse to fine; `finishEarly` from 30 %).

## 4. UI flow (`src/app`)

Home (sessions, new scan, diagnostics, the "run interrupted" card below) → Setup (mode, quality, options; developer:
debug overlay, VideoFrame grab, record raw capture — §5) → Permissions
(one tap: motion → camera → location) → Capture (video, guidance, coverage ring, live sparse
view, counters, finish) → Processing (stages + progress + cancel) → Viewer
(splats / points / mesh layer pill, cameras, EDL, measure, export, run dense / mesh / photoreal).
"Add more views" (viewer Build sheet, the viewer's "N photos could not be connected" card, Home's
session actions; "Continue scanning" for an interrupted capture) confirms that processing runs again,
which stages are rebuilt and that a photoreal model is removed (built again from Build), then opens
Capture on the stored session (`EXTEND_CAPTURE`: init with `resume` + `extend`, the stored mode /
quality, the stored sparse model — `InitResult.storedModel` — shown until the live model is rebuilt,
"Loading the scan's model n / N" while `modelRebuild` is reported); the first new keyframe sets the
session status back to 'capturing'. Finish re-runs refine and the stages the scan had (densify, mesh;
densify also with auto-dense); stopping waits for the frame in flight and without new views returns
to the stored model unchanged. A scan whose record missed new views (killed before the status change, a
frame answered after Stop) loads as a capture to process (`loadSession` status 'capturing' /
'captured'): App.ensureSessionLoaded sets the record back and the viewer hands it to processing. The splat
stage runs from the viewer itself ("Photoreal model" in the Build sheet → `splat` with a progress card fed by
`SplatTrainingDetail` and the latest `splat_preview` snapshot (tap: larger), "Use it now" from `finishFrom` on
(`finishEarly`: the model is stored, the Build sheet offers Continue — Rebuild once a new dense cloud or another
run dropped its checkpoint), Cancel, Resume after an interruption),
the model is fetched with `splatModel` into the
viewer's WebGPU splat layer and shown by default, and the export sheet offers 'ply-splat' and 'spz' (src/viewer/splat-layer.ts,
research/viewer-rendering.md). The export sheet's 'usdz' (AR model: the mesh simplified to ≤ 100 k triangles with its
colours baked into a 2048² JPEG atlas, built in the engine worker: seconds on a phone, so it yields between steps and
honours cancel — leaving the viewer cancels an export still being prepared) adds "View in AR" once prepared, in an
iPhone / iPad Safari tab only (not in the Home Screen app until the device check, ios-platform.md §8 item 11): a hidden
`<a rel="ar">` on the Blob's URL opens AR Quick Look, and a hint toast follows when the page neither hides nor loses
focus within 2 s (src/app/ar-quicklook.ts). With a splat model and an H.264 encoder (`VideoEncoder.isConfigSupported`) the export
sheet also offers "Fly-through video (MP4)". This is no engine call: the page renders it from the viewer's splat
layer (`Viewer.openSplatFrames`, one command buffer per frame). The camera path is src/viewer/flythrough.ts (an orbit
or a walk as the user chose the capture mode, else as the engine resolved it: `refine` reports it as
`stats.captureMode`), the encoder WebCodecs H.264 at 30 fps and 6 Mbit/s, and the muxer src/io/mp4.ts
(src/app/video-export.ts). The row shows progress (updated in place, the sheet is not rebuilt per frame) and Cancel
and then the two-tap Share / Save. The export stops when the page is hidden, when a photoreal
build starts, or when the viewer is left. Diagnostics has "Test video encoding" (codec, frame path, encode speed).

Capture guidance (`src/app/screens/capture.ts`): the banner shows the engine's per-frame `guidance` (from the
`processFrame` results, not the model updates) through a `GuidanceGate` (`src/app/capture-policy.ts`): priority
too_dark > tracking_lost > move_slower > more_texture > translate_more > initializing > good, a new code after 2
consecutive frames (too_dark / tracking_lost at once), a shown code kept ≥ 1.5 s unless a higher-priority one
replaces it, and a shown code no frame reported for 3 frames replaced by the most frequent of them (alternating
codes never confirm); screen readers hear only severity changes. 'too_dark' points at the torch (which pulses)
when the camera has one. In object mode — or auto once `resolvedMode` is 'object' — a coverage ring
(`src/app/coverage.ts`, `components/coverage-ring.ts`) shows, per live model update, which of 12 azimuth sectors
× 3 elevation bands (−10–10°, 10–40°, > 40°) around the cameras' convergence point have ≥ 2 registered
keyframes, the phone's cell from `framePose` (none while tracking is lost), and one prompt (other side / gaps /
raise the phone and look down / lower the phone / done; low + mid count as one level loop, so an object at eye
level is circled in the low band); azimuth only without the motion sensor. After 5 keyframes the Finish hint
shows the readiness ("Coverage 62 % · top not captured", not a live region); it never blocks Finish.
### Interrupted runs (run journal, `src/app/run-journal.ts`)

iOS kills a tab that exceeds its memory limit and evicts one in the background without any event, so the
app keeps a crash-safe marker of the long run in progress in localStorage `sfm.run.v1` = {v, sessionId,
name, step: refine | densify | mesh | splat, stage, quality, visible, startedAt, lastBeat, kills{step: n},
interrupted?}:

* written when a step of a stored scan starts (Processing's refine / densify / mesh; the viewer's photoreal
  build), on each stage change (at most every 5 s otherwise) and on visibilitychange while the run is open;
  the page's own unload (`pagehide`, not persisted) is not recorded as going to the background;
* closed on success (that step's kill count resets), on cancel, on any error the engine reported, and when
  the screen is left (kill counts kept). A kill runs nothing, so the open entry survives.

At App start an open entry becomes an interruption: **killed** when the page was visible at the last write
(likely memory or a crash — a closed tab looks the same, so the copy says "stopped"), **evicted** when it
was hidden; `kills[step]` is incremented. Home shows a "run interrupted" card (testId `run-interrupted`)
naming the scan and the step until it is resumed or dismissed (or the next run starts). Home cannot see
whether depth maps or a splat checkpoint were stored, so the card says Resume "continues from what was saved
so far, or starts again". Resume opens the scan through the normal OPEN_SESSION route: Processing for a
captured scan (refine), otherwise the viewer, whose dense / photoreal resume card continues from the stored
depth maps / splat checkpoint (`loadSession` stats). Opening the scan hands the interruption to that screen
(`RunJournal.open` / `takeOpened`, in memory): when no resume card covers it (nothing was stored yet, a
dense rebuild, a mesh), the viewer shows a toast whose action opens the Build sheet.

While `kills[step] ≥ 1` for the scan, its next densify / mesh / photoreal build runs in **light mode**: the
3D view shows at most 1.5 M points of the dense cloud (the existing `decimate`), no mesh, and the viewer's
prepared export files are dropped (an export in flight is discarded when it lands; Export makes none until
the build stops); the full view returns when the build stops. The card mentions the lighter view only for a
scan with a dense cloud. The engine's memory plan is unchanged, so splat checkpoints stay valid (lowering
`memoryBudgetMB` after a second kill is a later step). The MockEngine mirrors the engine's stored progress
in the session record's stats, so e2e/ui/states.mjs kills runs with a page reload (`killed-dense`,
`killed-splat`, `evicted-refine`, `killed-splat-early`).

## 5. Raw capture recorder (developer option)

Setup › Options › "Record raw capture" (off by default; `CaptureConfig.recordRaw`) records a
capture for replay on a computer (format: `src/io/recording.ts`; replay and scoring: e2e/README.md
"Raw capture recordings").

* Capture → `App.startRecording` after the engine's init, before the FramePump starts; stopped
  with the sensors (finish, cancel, leaving the screen; a start still opening then is stopped once
  open, never installed). It installs `CaptureRecorder` (`src/app/recorder.ts`) as the engine's
  FrameTap: per frame sent, a VideoFrame `clone()`, an ImageBitmap blitted at ≤ 1280 px, or an
  RGBAImage copy goes with the FrameInput metadata to the recorder worker, which writes a
  ≤ 1280 px JPEG and a `frames.jsonl` line (every FrameInput field verbatim) to OPFS
  (`sfm-recordings/<id>/`, sync access handles, JPEGs in 64 MB packs) or to the IndexedDB
  database `sfm-recordings`; the engine's answer goes to `live.jsonl` (`notSent` when the client
  never posted the frame). The raw DeviceMotion / DeviceOrientation events (the recorder's own
  window listeners, removed before the capture's sensors stop) and every GeoSensor fix go to
  `sensors.jsonl` in 500 ms batches.
* Bounds: when more than 3 frames are in the recorder worker, the frame just sent resolves its
  processFrame only when the worker caught up (≤ 4 in it; the FramePump skips frames as when the
  engine is busy, so the recording still equals the engine input); 1 GB of free origin quota to
  start, checked every 2 s; low storage, a write error or a worker stuck for 8 s of visible time
  stop the recording (a toast), never the scan.
* Diagnostics › "Raw capture recordings" (`src/app/screens/recordings.ts`, shown whether or not
  the diagnostics probes succeed): each recording with its frames, duration and size; Export = ZIP
  parts of ≤ 200 MB (each: Prepare, then Share / Save through `saveFile` in the tap), GPS left out
  unless "Include location"; Delete.
