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
                                                └─ refine / densify / mesh / export
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
  exportModel(format: 'ply-points' | 'ply-mesh' | 'glb' | 'colmap-zip'): Promise<Blob>
  loadSession(id: string, options?: Partial<EngineOptions>): Promise<SessionSummary>
  dispose(): void
  // worker / harness hooks: setHidden, flush, recoverGPU, memoryReport, releaseResultMemory
}
```

Quality presets (effective values: `QUALITY_PRESETS` of `src/engine/presets.ts`, dense part from the
tuned `DENSE_PRESETS` of `src/dense/presets.ts`):

| preset | feature max side | max features | dense max side | views | sources | PM iterations | geometric passes × iterations |
|---|---|---|---|---|---|---|---|
| fast | 1024 | 2048 | 800 | 40 | 5 | 4 | 1 × 2 |
| balanced | 1600 | 4096 | 1280 | 80 | 8 | 6 | 1 × 3 |
| max | 2048 | 8192 | 1600 | 150 | 8 | 6 | 1 × 3 |

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
correction → `runIncrementalSfM` (gyro-checked registrations) → keep the better of live
vs offline (gyro breaks, then registered images, then explained correspondences, then
reprojection error; isolated gyro outliers deregistered) → final global BA (focal + principal point + k1 + k2) →
per-image focal lengths (focus breathing; kept only if they help) → world alignment
(`sfm/world.ts`) → persist. Details: ARCHITECTURE §6 "Real-capture robustness".

### Dense / mesh
Views stream from IndexedDB through an LRU image cache; depth maps go through a
`DepthMapStore` with an LRU memory cache backed by IndexedDB to bound memory on
phones: the memory planner (`src/engine/memory.ts`) budgets 1.2 GB (600 / 850 MB on 4 / 6 GB
iPhones); iOS kills a tab at ~1.5 GB (soft) / 2 GB (hard), docs/research/ios-platform.md.

## 3. Worker protocol (`src/workers/protocol.ts`)

Request/response RPC with ids, plus unsolicited events:

* requests: `init`, `frame`, `finishCapture`, `refine`, `densify`, `mesh`,
  `export`, `loadSession`, `cancel`, `visibility` (page hidden / visible: pause GPU work), `dispose`
* events: `progress` (ProgressEvent), `live` (LiveUpdate), `log`, `error`
* large typed arrays and ImageBitmaps are transferred, never copied.

## 4. UI flow (`src/app`)

Home (sessions, new scan, diagnostics) → Setup (mode, quality) → Permissions
(one tap: motion → camera → location) → Capture (video, guidance, live sparse
view, counters, finish) → Processing (stages + progress + cancel) → Viewer
(points / mesh / cameras, EDL, measure, export, run dense / mesh).
