# Architecture & module contracts

How the photogrammetry app is structured: goals, conventions, platform rules, module
contracts and the pipeline. Keep it short and stable. Update it in the same change when you
alter a convention, a public contract or the pipeline (when deviating from a signature, keep
the documented one working: add, don't break; list legitimate deviations in §4.10).

How each module works, why its defaults are what they are and its known limits live in the
source file headers. Measurements behind the defaults are dated reports in `docs/research/`.

## 1. Goals

* Runs entirely on-device in mobile Safari (iOS 26+, WebGPU) — no server. Works
  (slower) without WebGPU via CPU fallbacks.
* Continuous capture: the user just moves the phone; the app picks sharp,
  informative keyframes automatically and reconstructs live (sparse) so the user
  sees coverage while capturing.
* Close range (objects, 0.2–2 m) **and** large scenes (backyard + surrounding
  houses, 5–100 m). Nothing assumes a fixed depth range.
* Highest accuracy: COLMAP-style incremental SfM with robust bundle adjustment and
  self-calibration (focal + radial distortion), RootSIFT features on the GPU,
  PatchMatch multi-view stereo with geometric consistency, consistency-checked
  depth-map fusion, TSDF meshing.
* Sensors: gyroscope (rotation priors, blur rejection, guided matching, time-offset
  calibration), accelerometer (gravity alignment, metric scale), compass (north),
  GPS (metric scale and geo-referencing at large scale).
* Exports: PLY (points / mesh), GLB, COLMAP text model (+ images) in a ZIP.

## 2. Conventions (MUST follow)

| Item | Convention |
|---|---|
| Matrices | Row-major `Float64Array`. `Mat3` = 9, `Mat34` = 12, `Mat4` = 16. Element (r,c) of a 3x3 is `m[3*r + c]`. |
| Vectors | `Float64Array` (`Vec3` = 3). Hot loops may use scalars/flat arrays. |
| Camera frame | OpenCV: +x right, +y down, +z forward (optical axis). |
| Pose | World-to-camera: `x_cam = R * X_world + t`; camera centre `C = -Rᵀ t`. |
| Relative pose (two-view) | `x2 = R * x1 + t` (camera-1 frame to camera-2 frame), `|t| = 1`. |
| Pixel coordinates | Centre of top-left pixel = (0, 0). Image covers [-0.5, W-0.5] × [-0.5, H-0.5]. COLMAP export adds +0.5. |
| Resampling | Downscale by s = W_small / W_full: `x_small = (x_full + 0.5) * s - 0.5`, inverse `x_full = (x_small + 0.5) / s - 0.5`. Intrinsics: `f' = f*s`, `c' = (c + 0.5)*s - 0.5`. |
| Keypoints | Always in the FULL-resolution pixel frame of their keyframe. |
| Camera model | `RADIAL`: normalized `x=X/Z, y=Y/Z`, `r²=x²+y²`, `d = 1 + k1 r² + k2 r⁴`, `u = f d x + cx`, `v = f d y + cy`. Square pixels, one `f`. |
| Quaternion | `[w, x, y, z]`, Hamilton, unit. |
| Rotation update in optimisers | Left-multiplicative: `R ← exp([δω]×) R`, `t ← t + δt`. |
| Angles | Radians unless the identifier ends in `Deg`. |
| Time | Milliseconds, `performance.now()` domain. A stored capture is on its session time base: FrameInput time + the shift of its capture segment (`EngineStateRecord.timeSegments`), so a capture continued after a page reload stays in time order (src/engine/live.ts). |
| Device frame (W3C DeviceMotion) | +x right edge, +y top edge, +z out of screen (portrait natural orientation). |
| Camera ← device rotation | Rear camera, processed image upright for the UI rotated by screen angle θ (device turned θ counter-clockwise from natural portrait as seen by the user; θ ∈ {0, 90, 180, 270}): `R_cam_device(θ) = diag(1, −1, −1) · Rz(θ)` where `Rz(θ) = [[cosθ, −sinθ, 0], [sinθ, cosθ, 0], [0, 0, 1]]`. θ=0: `diag(1,−1,−1)`; θ=90: rows `(0,−1,0), (−1,0,0), (0,0,−1)`. `v_cam = R_cam_device · v_device`. (iOS mapping of `screen.orientation.angle` / `window.orientation` to θ is verified in `src/sensors/frame-orientation.ts`.) |
| Device attitude | `R_wd` = device-to-world rotation; camera pose rotation `R = R_cam_device · R_wdᵀ`. Gyro `ω_dev`: `dR_wd/dt = R_wd [ω_dev]×`. `accel_dev = R_wdᵀ a_world`, `accelG_dev = R_wdᵀ (a_world − g_world)`, `g_world = (0, 0, −9.81)` in the Z-up world. |
| Gravity | `ImuSample.accelG` uses the W3C spec sign: device lying flat face-up reports ≈ (0, 0, +9.81) (specific force). "Gravity direction" vectors in this code base always point DOWN. |
| World frame after alignment | +Z up (opposite gravity). If north is known: +Y = true north, +X = east (local ENU). Units = metres when `MetricInfo.metric`. `northAligned` means TRUE north only (compass headings are magnetic until the WMM declination is applied; a magnetic-only yaw leaves +Y ≈ magnetic north with `northAligned: false` — `src/sfm/world.ts`, docs/research/true-north-heading.md). |
| Viewer | three.js is Y-up; the viewer converts from the Z-up world (`(x, y, z) → (x, z, -y)`). |
| Random numbers | Always through the seeded `Rng` from `src/math/random.ts` (deterministic tests). |
| Descriptors | RootSIFT, 128 bytes, `min(255, round(512 * v))` of the unit-L2 RootSIFT vector. |

Shared types live in `src/core/types.ts` (do not duplicate them).

### 2.1 WebGPU rules for iOS Safari (from docs/research/ios-webgpu.md)

* At most ONE heavy command buffer in flight: submit, then `await device.queue.onSubmittedWorkDone()`
  (`BudgetEncoder` in `src/features/gpu-common.ts`, the band loop of `PatchMatchGPU`) before the next. Size command buffers to ~2–30 ms of GPU
  work (tile large images / batch pairs); never ~100 ms+.
* Create all pipelines up front (pipeline compilation blocks the queue on Safari 26/27) — e.g. in a
  `create()`/warm-up step, not lazily in the middle of the capture loop.
* No `vec3` in storage/uniform buffers (use vec4 / scalars); no f32 atomics (use fixed-point i32 or
  `atomicMin` on u32 bits); no subgroups (not shipped in Safari); `shader-f16` is available but
  optional; `float32-filterable` only on A17 Pro+ (do not rely on it — filter r8unorm/r16float).
* Budget 256 MiB per buffer/binding (Safari 26.0–26.1); read `device.limits` for more.
* Workgroup memory ≤ 32 KiB on iPhone but keep ≤ 16 KiB for portability; 64–256 invocations.
* `getMappedRange` is a copying cross-process call: read back once per map, small results only.
* Treat `onSubmittedWorkDone` as "finished", not "succeeded"; GPU work submitted while the page is
  hidden may be silently dropped — the engine pauses GPU work when `document.hidden`.
* The whole tab (JS heap + GPU + workers) should stay below ~1.2 GB on iPhone or iOS kills it
  without warning: stream large data (images, depth maps) through storage with LRU caches.

### 2.2 iOS capture facts (from docs/research/ios-camera.md, ios-sensors.md)

* `facingMode: 'environment'` selects a lens-switching virtual camera — select the physical
  "Back Camera" by `deviceId` (labels are localised and only available after permission).
* Settable: zoom (keep 1), torch, whiteBalanceMode, frameRate. NOT settable: focus, exposure.
* Default 1920×1440 (4:3, full FOV); max 4032×3024. Main-camera HFOV prior ≈ 67–72° (±10%).
* DeviceMotion on iOS: ~60 Hz fixed, `rotationRate` deg/s (alpha=x, beta=y, gamma=z axes),
  `interval` in seconds, and BOTH `acceleration` and `accelerationIncludingGravity` have the
  OPPOSITE sign of the W3C spec (face-up ⇒ z ≈ −9.81). `src/sensors` normalises to the spec sign
  with a runtime self-check; everything downstream sees spec-sign `ImuSample`s.
* θ for `R_cam_device` = `(window.orientation + 360) % 360` on iPhone (90 = top edge pointing left).
  Drop frames for ~500 ms after an orientation change.

### 2.3 Browser compatibility

* Build target es2022 (no lowering needed for Safari 16.4+). Newer built-ins and APIs
  (`Float16Array`, `requestIdleCallback`, `scheduler.yield`, `navigator.userAgentData`,
  `deviceMemory`, `VideoFrame`, `MediaStreamTrackProcessor`) are feature-detected with fallbacks.
* WGSL kernels never require an optional feature: no `enable f16`, subgroups, storage textures,
  `vec3` in buffers; `r32float` is only `textureLoad`ed (details: `src/gpu/device.ts`).
* CSS: every `backdrop-filter` / `user-select` has its `-webkit-` twin, `100dvh` has a `100vh`
  fallback, safe-area insets with `viewport-fit=cover`. No `:has()`, `color-mix()`, container
  queries, `@layer`, nesting or `<dialog>` (modals are `role=dialog` + `inert`).

## 3. Directory layout

```
src/
  core/types.ts          shared types (read-only for module authors)
  math/                  linear algebra, rotations, decompositions, RNG, stats, sim3
  geometry/              camera model, pose utils, minimal solvers, RANSAC, two-view
  ba/                    bundle adjustment (LM + Schur complement)
  features/              image utils, SIFT (CPU reference + WebGPU), matching (CPU + WebGPU)
  gpu/                   WebGPU device + helpers (device.ts is shared)
  sfm/                   database, correspondence graph, reconstruction, incremental mapper,
                         pair selection, retrieval, keyframe selection, inertial & GPS alignment
  dense/                 undistortion, view selection, PatchMatch MVS (WebGPU + CPU),
                         depth-map fusion, TSDF + marching cubes, point-cloud filters
  splat/                 3D Gaussian Splatting: WebGPU trainer + renderer, CPU reference,
                         splat PLY, point-cloud initialisation, MCMC densification
  sensors/               camera capture, motion, compass, geolocation, wake lock, frame orientation
  io/                    PLY / GLB / COLMAP export, ZIP writer, IndexedDB session storage, share/download,
                         raw capture recordings (format, OPFS / IndexedDB store, dataset export)
  viewer/                three.js viewer (points as splats + EDL, mesh, cameras, measure tool)
  engine/                PhotogrammetryEngine (pipeline orchestration, runs in a worker)
  workers/               engine.worker.ts + typed message protocol, recorder.worker.ts (raw capture)
  app/                   mobile UI (vanilla TS + CSS), screens, state machine
  harness/               test harness pages: synthetic dataset generator, engine e2e, dense GT,
                         splat GT, novel-view evaluation
  main.ts                app entry
e2e/                     synthetic dataset generator, end-to-end runner, metrics, novel-view scorer
docs/                    this file, ENGINE.md, research and measurement reports
```

Tests: `*.test.ts` next to the code (Node, `npx vitest run --project unit <path>`),
`*.gpu.test.ts` for WebGPU/DOM code (headless Chromium + SwiftShader,
`npx vitest run --project gpu <path>`). SwiftShader is slow: keep GPU test images
small (≤ 640 px).

## 4. Module contracts

### 4.1 `src/math`

Files: `vec.ts`, `mat3.ts`, `matrix.ts`, `decomp.ts`, `poly.ts`, `rotation.ts`,
`random.ts`, `stats.ts`, `sim3.ts`, `index.ts` (re-exports everything).

```ts
// vec.ts
vec3(x?: number, y?: number, z?: number): Vec3
vec3Add(a, b, out?): Vec3; vec3Sub(a, b, out?): Vec3; vec3Scale(a, s, out?): Vec3
vec3Dot(a, b): number; vec3Cross(a, b, out?): Vec3; vec3Norm(a): number
vec3Normalize(a, out?): Vec3; vec3Dist(a, b): number; vec3Copy(a, out?): Vec3
// (a, b are ArrayLike<number>; outputs are Float64Array)

// mat3.ts
mat3Identity(out?): Mat3; mat3Copy(a, out?); mat3Mul(a, b, out?); mat3MulVec(a, v, out?)
mat3TransposeMulVec(a, v, out?) /* aᵀ v */; mat3MulTranspose(a, b, out?) /* a bᵀ */
mat3TransposeMul(a, b, out?) /* aᵀ b */; mat3Transpose(a, out?); mat3Det(a): number
mat3Inv(a, out?): Mat3 | null; mat3Skew(v, out?): Mat3; mat3Scale(a, s, out?); mat3Add(a, b, out?)
mat3FrobeniusNorm(a): number

// matrix.ts — general dense row-major matrix
class Matrix { rows: number; cols: number; data: Float64Array;
  constructor(rows, cols, data?); static identity(n); static fromRows(rows: number[][]);
  get(i, j); set(i, j, v); clone(); transpose(): Matrix; mul(b: Matrix): Matrix; mulVec(x): Float64Array }

// decomp.ts
cholesky(A: Float64Array, n: number): boolean          // in-place, lower triangle = L; false if not SPD
choleskySolve(L: Float64Array, n: number, b: Float64Array): void   // in-place solve L Lᵀ x = b
solveSPD(A: Float64Array, n: number, b: Float64Array): Float64Array | null  // non-destructive
luSolve(A: Float64Array, n: number, b: Float64Array): Float64Array | null   // partial pivoting, non-destructive
qr(A: Matrix): { Q: Matrix; R: Matrix }                // Householder, thin (Q m×k, R k×n, k=min(m,n))
svd(A: Matrix): { U: Matrix; S: Float64Array; V: Matrix }  // thin, A = U diag(S) Vᵀ, S descending, any m×n
svd3(A: Mat3): { U: Mat3; S: Vec3; V: Mat3 }           // fast 3×3 path, A = U diag(S) Vᵀ
nullVector(A: Matrix): Float64Array                    // unit right-singular vector of smallest σ (works for m < n too)
symEig(A: Float64Array, n: number): { values: Float64Array; vectors: Float64Array } // ascending; column k of row-major n×n `vectors` is eigvec k

// poly.ts
polyEval(c: ArrayLike<number>, x): number              // c in DESCENDING powers
polyRealRoots(c: ArrayLike<number>, lo?: number, hi?: number): number[]  // all real roots (Sturm + bisection + Newton polish), any degree ≤ 20
solveQuadratic(a, b, c): number[]; solveCubic(a, b, c, d): number[]; solveQuartic(a, b, c, d, e): number[]  // real roots

// rotation.ts
quatToMat3(q, out?): Mat3; mat3ToQuat(R, out?): Quat   // robust (Shepperd), w ≥ 0
quatMul(a, b, out?); quatConj(q, out?); quatNormalize(q, out?); quatSlerp(a, b, t, out?)
rotvecToMat3(w, out?): Mat3       // exp map, Taylor near 0
mat3ToRotvec(R, out?): Vec3       // log map, stable near 0 and π
axisAngleToMat3(axis, angle, out?): Mat3
rotationAngle(R): number; rotationAngleBetween(Ra, Rb): number
orthonormalize(R, out?): Mat3     // nearest rotation (SVD), det = +1
rotationBetweenVectors(a, b, out?): Mat3   // minimal rotation mapping unit a → unit b (handles a ≈ -b)

// random.ts
class Rng { constructor(seed?: number); next(): number /*[0,1)*/; int(n): number /*[0,n)*/;
  gaussian(): number; sample(n: number, k: number, out?: Uint32Array): Uint32Array /*k distinct*/;
  shuffle<T>(a: T[] | Uint32Array): void }

// stats.ts
median(v: ArrayLike<number>): number; percentile(v, p /*0..100*/): number   // non-destructive
mean(v): number; mad(v): number; quickselect(a: Float64Array | number[], k): number // in-place

// sim3.ts
umeyama(src: Float64Array, dst: Float64Array, n: number, withScale: boolean, weights?: Float64Array)
  : { s: number; R: Mat3; t: Vec3 } | null               // dst ≈ s R src + t (3n flat arrays)
applySim3(sim, p: Float64Array /*3n*/, out?): Float64Array
```

### 4.2 `src/geometry`  (depends on math)

Files: `camera.ts`, `pose.ts`, `ransac.ts`, `essential.ts`, `fundamental.ts`,
`homography.ts`, `pnp.ts`, `triangulation.ts`, `twoview.ts`, `align.ts`, `convergence.ts`, `index.ts`.

```ts
// camera.ts
createIntrinsics(width, height, hfovRad, k1 = 0, k2 = 0): CameraIntrinsics  // cx=(W-1)/2, cy=(H-1)/2, f=(W/2)/tan(hfov/2)
scaleIntrinsics(cam, newWidth, newHeight): CameraIntrinsics                 // per §2 resampling rule
distortNormalized(cam, x, y, out: Float64Array /*2*/): void
undistortNormalized(cam, xd, yd, out): void           // Newton; exact inverse of distortNormalized within 1e-10 for |k| < 0.5
projectCam(cam, X: ArrayLike<number>, out): boolean    // camera-frame point → pixel; false if Z ≤ 1e-12
projectWorld(cam, pose, X, out): boolean
pixelToNormalized(cam, u, v, out): void                // undistorted normalized image coords
pixelToBearing(cam, u, v, out /*3*/): void             // unit vector, camera frame
keypointsToNormalized(cam, kp: Float32Array /*stride 4*/, out?: Float64Array /*2n*/): Float64Array

// pose.ts
poseIdentity(): Pose; poseCopy(p): Pose; poseCenter(p, out?): Vec3; poseFromCenter(R, C): Pose
poseCompose(a, b): Pose   // a∘b : x ↦ a(b(x))
poseInverse(p): Pose; poseRelative(p1, p2): Pose   // returns {R: R2 R1ᵀ, t: t2 − R t1}
poseToMat34(p, out?): Mat34; poseTransformPoint(p, X, out?): Vec3

// ransac.ts — generic LO-MSAC
interface RansacEstimator<M> {
  sampleSize: number; numData: number;
  fit(sample: Uint32Array, models: M[]): number;            // fills models, returns count
  residualsSq(model: M, out: Float64Array): void;           // squared residual per datum
  refit?(inliers: Uint32Array, count: number): M | null;    // least squares on inliers (local optimisation)
  isDegenerate?(sample: Uint32Array): boolean;
}
interface RansacOptions { thresholdSq: number; confidence?: number /*0.9999*/; maxIterations?: number /*10000*/;
  minIterations?: number; rng?: Rng; localOptimization?: boolean /*true*/; }
interface RansacResult<M> { model: M | null; inlierMask: Uint8Array; numInliers: number; iterations: number; score: number }
ransac<M>(est: RansacEstimator<M>, opts: RansacOptions): RansacResult<M>

// essential.ts (all inputs are undistorted NORMALIZED coords, flat [x0,y0,x1,y1,...])
essential5pt(x1, x2): Mat3[]                // minimal 5-point (Nistér/Stewénius), ≤ 10 solutions
essentialLinear(x1, x2, n): Mat3 | null     // ≥ 8 points, projected onto the essential manifold
decomposeEssential(E): { R1: Mat3; R2: Mat3; t: Vec3 }
recoverPose(E, x1, x2, n, mask?: Uint8Array): { R: Mat3; t: Vec3; numInFront: number; inFrontMask: Uint8Array }
sampsonDistanceSq(M: Mat3, x1, y1, x2, y2): number   // works for E (normalized) or F (pixels)

// fundamental.ts (pixels)
fundamental7pt(p1, p2): Mat3[]; fundamental8pt(p1, p2, n): Mat3 | null  // Hartley-normalised, rank-2

// homography.ts (pixels or normalized)
homographyDLT(p1, p2, n): Mat3 | null      // n ≥ 4, normalised DLT, p2 ~ H p1
homographyTransferErrorSq(H, x1, y1, x2, y2): number

// pnp.ts
p3p(X: Float64Array /*9*/, bearings: Float64Array /*9 unit vectors, camera frame*/): Pose[]   // Lambda Twist
epnp(X: Float64Array, xn: Float64Array /*normalized 2n*/, n): Pose | null
refinePose(pose, X, xn, n, opts?: { iterations?: number; huberNormalized?: number }): Pose   // LM, robust
solvePnPRansac(X: Float64Array /*3n*/, uv: Float64Array /*2n pixels*/, n, cam, opts?: {
  maxErrorPx?: number /*8*/; confidence?: number; maxIterations?: number; minInliers?: number; rng?: Rng })
  : { pose: Pose; inlierMask: Uint8Array; numInliers: number } | null

// triangulation.ts
triangulateDLT(poses: Pose[], xn: Float64Array /*2n normalized*/, n, out?): Vec3 | null
triangulateTwoView(p1: Pose, p2: Pose, x1: ArrayLike<number>, x2: ArrayLike<number>, out?): Vec3 | null
refinePoint(poses: Pose[], cams: CameraIntrinsics[], uv: Float64Array /*2n px*/, n, X: Vec3, iterations?): Vec3
triangulationAngle(C1: Vec3, C2: Vec3, X: Vec3): number
pointDepth(pose: Pose, X: ArrayLike<number>): number   // z in camera frame

// twoview.ts
estimateTwoViewGeometry(kp1: Float32Array, kp2: Float32Array /*stride 4, px*/, matches: MatchPairs,
  cam1: CameraIntrinsics, cam2: CameraIntrinsics, opts?: {
    maxErrorPx?: number /*4*/; minNumInliers?: number /*15*/; minInlierRatio?: number /*0.25*/;
    confidence?: number; maxIterations?: number; rng?: Rng; rotationPrior?: Mat3 /*R_2←1 from gyro*/ })
  : TwoViewGeometry
// COLMAP-style model selection between E, F, H; computes R, t (|t|=1) and triAngle for
// calibrated/planar configs.

// align.ts
alignSim3Ransac(src: Float64Array, dst: Float64Array, n, maxError: number, opts?: { weights?; rng?; minInliers? })
  : { s: number; R: Mat3; t: Vec3; inlierMask: Uint8Array; numInliers: number } | null

// convergence.ts — where the cameras look (object centre for classification, coverage, crop, fly-through)
opticalAxisConvergence(poses: Pose[]): { point: Vec3 | null; spreadRad: number; inFront: number;
  distMedian: number; offAxisMedian: number; converging: boolean }   // least-squares point closest to all optical axes
```

### 4.3 `src/ba`  (depends on math)

Files: `problem.ts`, `solver.ts`, `loss.ts`, `index.ts`.

```ts
interface BAProblem {
  cameras: CameraIntrinsics[];                    // refined in place
  cameraRefine: { focal: boolean; principalPoint: boolean; distortion: boolean }[];
  poses: Pose[];                                  // refined in place (world-to-camera)
  imageCamera: Uint32Array;                       // camera index per pose
  poseFixed: Uint8Array;                          // 1 ⇒ pose constant
  translationFixedMask?: Uint8Array;              // per pose, bit0=x bit1=y bit2=z of t held constant (gauge)
  points: Float64Array;                           // 3 per point, refined in place
  pointFixed?: Uint8Array;
  obsPose: Uint32Array; obsPoint: Uint32Array;    // observation → pose index / point index
  obsUV: Float64Array;                            // 2 per observation, pixels
  obsWeight?: Float32Array;                       // optional per-observation weight (1/σ)
  centerPriors?: { pose: number; center: Vec3; sigma: number }[];          // e.g. GPS, same world frame
  relRotationPriors?: { i: number; j: number; R_ji: Mat3; sigma: number }[]; // R_j ≈ R_ji R_i (gyro)
  focalPriors?: { camera: number; f: number; sigma: number }[];            // (f − f₀)/σ, only when f is refined
  principalPointPriors?: { camera: number; cx: number; cy: number; sigma: number }[]; // (c − c₀)/σ, only when c is refined
}
interface BAOptions {
  loss?: 'trivial' | 'huber' | 'cauchy' | 'soft_l1';   // default 'cauchy'
  lossScalePx?: number;                                // default 1.0 px
  maxIterations?: number;                              // default 50
  functionTolerance?: number; gradientTolerance?: number; parameterTolerance?: number;
  initialLambda?: number;
  linearSolver?: 'auto' | 'dense_schur' | 'pcg_schur'; // auto: dense if reduced size ≤ 2400
  pcgMaxIterations?: number; pcgTolerance?: number;
  timeLimitMs?: number; cancel?: CancelToken;
  onIteration?: (iter: number, cost: number) => void;
}
interface BAResult { success: boolean; termination: string; iterations: number;
  initialCost: number; finalCost: number; initialRmsPx: number; finalRmsPx: number; timeMs: number }
bundleAdjust(problem: BAProblem, options?: BAOptions): BAResult
reprojectionErrors(problem: BAProblem, out?: Float64Array): Float64Array   // per observation, px
```

Levenberg–Marquardt with analytic Jacobians, Schur complement on points, shared
intrinsics (one param block per camera connected to all its poses), dense
Cholesky of the reduced camera system when small, block-Jacobi PCG otherwise.

### 4.4 `src/features`

Files: `image.ts`, `sift-cpu.ts`, `sift-gpu.ts` (+ `shaders/*.wgsl` or WGSL in `.ts`),
`match.ts`, `match-gpu.ts`, `distribute.ts`, `extractor.ts`, `index.ts`.

```ts
// image.ts
rgbaToGray(img: RGBAImage): GrayImage                         // Float32 0..1, Rec.709 luma on sRGB values
resizeGray(img: GrayImage, w: number, h: number): GrayImage   // area-averaging when shrinking, bilinear when enlarging
gaussianBlur(img: GrayImage, sigma: number): GrayImage
sharpness(img: GrayImage): number                              // variance of Laplacian (on ≤ 640 px copy)
rotateRGBA90(img: RGBAImage, quarterTurns: number): RGBAImage

// sift-cpu.ts / sift-gpu.ts
interface SiftOptions {
  maxFeatures: number;          // default 4096
  maxImageSize: number;         // default 1600 — input downscaled so long side ≤ this
  firstOctave: number;          // -1 = upsample ×2 first, default 0 (auto -1 when long side < 800)
  scalesPerOctave: number;      // 3
  sigma: number;                // 1.6
  peakThreshold: number;        // 0.02 / scalesPerOctave on 0..1 images
  edgeThreshold: number;        // 10
  maxOrientations: number;      // 2
  distribute: boolean;          // grid-based spatial distribution of kept features (default true)
}
extractSiftCPU(img: GrayImage, opts?: Partial<SiftOptions>, fullWidth?: number, fullHeight?: number): FeatureSet
class SiftGPU { static create(ctx: GPUContext, opts?: Partial<SiftOptions>): Promise<SiftGPU>;
  extract(img: GrayImage, fullWidth?: number, fullHeight?: number): Promise<FeatureSet>; destroy(): void }

// match.ts
interface MatchOptions { ratio?: number /*0.8*/; mutual?: boolean /*true*/; maxDistance?: number /*0.7 on unit-norm descriptors*/ }
type MatchGuide =
  | { kind: 'F'; F: Mat3; maxErrorPx: number }               // epipolar distance in image b
  | { kind: 'H'; H: Mat3; maxErrorPx: number }               // x_b ~ H x_a
  | { kind: 'positions'; predicted: Float32Array; radiusPx: number }  // 2 per feature of a (NaN = unknown)
matchCPU(a: FeatureSet, b: FeatureSet, opts?: MatchOptions): MatchPairs
matchGuidedCPU(a: FeatureSet, b: FeatureSet, guide: MatchGuide, opts?: MatchOptions): MatchPairs

// match-gpu.ts
class MatcherGPU { static create(ctx: GPUContext): Promise<MatcherGPU>;
  match(a: FeatureSet, b: FeatureSet, opts?: MatchOptions): Promise<MatchPairs>;
  matchGuided(a, b, guide: MatchGuide, opts?): Promise<MatchPairs>; destroy(): void }

// extractor.ts — picks GPU when available, CPU otherwise
interface FeaturePipeline { readonly gpu: boolean;
  extract(img: GrayImage, fullWidth: number, fullHeight: number): Promise<FeatureSet>;
  match(a: FeatureSet, b: FeatureSet, opts?: MatchOptions): Promise<MatchPairs>;
  matchGuided(a, b, guide: MatchGuide, opts?: MatchOptions): Promise<MatchPairs>; destroy(): void }
createFeaturePipeline(opts?: { preferGPU?: boolean; sift?: Partial<SiftOptions> }): Promise<FeaturePipeline>
```

Matching semantics: descriptor distance = Euclidean distance between unit RootSIFT
vectors (dequantised /512). Lowe ratio test on best/second-best. Mutual = also
best in the reverse direction. Output pairs sorted by index in `a`.

### 4.5 `src/sfm`  (depends on math, geometry, ba, core)

```ts
// database.ts
pairKey(a: number, b: number): number            // order-independent integer key (ids < 2^20)
class FeatureDatabase {
  frames: Map<number, FrameInfo>; features: Map<number, FeatureSet>;
  addFrame(info: FrameInfo, features: FeatureSet): void;
  setTwoView(a: number, b: number, geom: TwoViewGeometry): void   // stored canonically (min id first)
  getTwoView(a: number, b: number): TwoViewGeometry | undefined    // inliers oriented (a-feature, b-feature)
  hasPair(a, b): boolean; pairs(): IterableIterator<[number, number, TwoViewGeometry]>
  frameIds(): number[]  // ascending
}
// correspondence-graph.ts — supports incremental growth
class CorrespondenceGraph {
  addImage(id: number, numFeatures: number): void
  addTwoView(a: number, b: number, inliers: MatchPairs): void
  correspondences(image: number, feature: number): ReadonlyArray<{ image: number; feature: number }>  // empty array if none
  numCorrespondences(image: number): number
  numCorrespondencesBetween(a: number, b: number): number
  imagePairs(): IterableIterator<[number, number, number /*count*/]>
}
// reconstruction.ts
interface RecImage { id: number; cameraId: number; pose: Pose; registered: boolean;
  point3D: Int32Array /* per feature, -1 = none */; numPoints3D: number }
interface Track { image: number; feature: number }
interface RecPoint { id: number; xyz: Vec3; color: [number, number, number]; error: number; track: Track[] }
class Reconstruction {
  cameras: Map<number, CameraIntrinsics>; images: Map<number, RecImage>; points: Map<number, RecPoint>;
  metric: MetricInfo;
  addCamera(cam): number; addImage(id, cameraId, numFeatures): RecImage;
  registerImage(id, pose): void; deregisterImage(id): void;
  addPoint(xyz, track: Track[], color?): number; addObservation(pointId, t: Track): void;
  deleteObservation(image, feature): void; deletePoint(id): void; mergePoints(a, b): number;
  registeredImageIds(): number[];
  transform(s: number, R: Mat3, t: Vec3): void          // X' = s R X + t applied to points and poses
  meanReprojectionError(): number; meanTrackLength(): number;
  clone(): Reconstruction
}
// mapper.ts
interface MapperOptions { … see file … }
class IncrementalMapper { constructor(db, graph, rec, opts?: Partial<MapperOptions>) … }
runIncrementalSfM(db: FeatureDatabase, opts?: Partial<MapperOptions>, progress?: ProgressCallback,
  cancel?: CancelToken): Promise<Reconstruction[]>   // largest model first
// keyframes.ts, pairs.ts, vocab.ts, inertial.ts, geo.ts, live.ts — see §5.
```

### 4.6 `src/dense`  (depends on math, geometry, core, gpu)

```ts
interface DenseView { id: number; cam: CameraIntrinsics /* pinhole, k1=k2=0 */; pose: Pose;
  image: RGBAImage /* undistorted, at cam.width × cam.height */ }
interface DepthMap { width: number; height: number; depth: Float32Array /* 0 = invalid, z-depth */;
  normal: Float32Array /* 3 per px, camera frame, facing the camera */; cost: Float32Array /* NCC cost 0..2 */ }
interface DepthMapStore { get(id: number): Promise<DepthMap | null>; put(id: number, d: DepthMap): Promise<void> }
class MemoryDepthMapStore implements DepthMapStore
undistortImage(src: RGBAImage, cam: CameraIntrinsics, maxSize: number): { image: RGBAImage; cam: CameraIntrinsics }
selectSourceViews(views: { id; pose; cam }[], sparsePoints: { xyz: Float64Array /*3n*/; visibility: number[][] /* view ids per point */ },
  opts?): Map<number, { sources: number[]; depthMin: number; depthMax: number }>
interface PatchMatchOptions { iterations: number /*6*/; patchRadius: number /*5*/; patchStep: number /*2*/;
  numSamplesPerIter?: number; geometric: boolean; ... }
class PatchMatchGPU { static create(ctx: GPUContext, opts?): Promise<PatchMatchGPU>;
  compute(ref: DenseView, sources: DenseView[], range: { min: number; max: number },
          opts?: Partial<PatchMatchOptions>, geometric?: { prior: DepthMap; sourceDepths: DepthMap[] }): Promise<DepthMap>;
  destroy(): void }
patchMatchCPU(ref, sources, range, opts?, geometric?): DepthMap    // slow reference / fallback
fuseDepthMaps(views: DenseView[], store: DepthMapStore, neighbours: Map<number, number[]>, opts?, progress?, cancel?)
  : Promise<PointCloud>
tsdfMesh(views: DenseView[], store: DepthMapStore, opts?: { voxelSize?: number; truncation?: number; maxDepth?: number;
  bounds?: { min: ArrayLike<number>; max: ArrayLike<number> } /* world AABB: no vertex outside */ }, progress?, cancel?): Promise<TriangleMesh>
filterPointCloud(pc: PointCloud, opts?: { voxelSize?: number; statistical?: { k: number; std: number } }): PointCloud
// object-roi.ts — object box of an orbit capture (gravity-aligned world): orbit centre + support plane
proposeObjectRoi(model: { images: { pose: Pose }[]; xyz: ArrayLike<number>; numPoints: number; trackOffsets?: ArrayLike<number> },
  opts: { gravityAligned: boolean; minClearance?: number | { perDistance: number }; minConfidence?: number /* 0.3 */ })
  : { roi: ObjectRoi | null; reason: string; … }   // ObjectRoi {min, max, planeZ, centre, confidence} (core/types.ts)
objectRoiPivot(roi: ObjectRoi): [x, y, planeZ]      // bottom centre: the origin for exports that stand the object on the ground
```

### 4.7 `src/sensors`

```ts
// camera.ts
interface CameraInfo { deviceId: string; label: string; lens: 'wide' | 'ultrawide' | 'tele' | 'virtual' | 'front' | 'unknown' }
interface CameraStartResult { width: number; height: number; frameRate: number; label: string; deviceId: string;
  hfovPriorDeg: number; capabilities: Record<string, unknown>; settings: Record<string, unknown> }
class CameraCapture {
  static listCameras(): Promise<CameraInfo[]>
  start(video: HTMLVideoElement, opts: { quality: 'fast' | 'balanced' | 'max'; deviceId?: string }): Promise<CameraStartResult>
  stop(): void
  onFrame(cb: (t: number, meta: { mediaTime: number; captureTime?: number; width: number; height: number }) => void): () => void
  grabFrame(): Promise<ImageBitmap>              // full-resolution current frame
  setTorch(on: boolean): Promise<boolean>; setZoom(z: number): Promise<boolean>
}
// motion.ts
class MotionSensor {
  static isSupported(): boolean
  static requestPermission(): Promise<'granted' | 'denied' | 'unsupported'>   // call from a user gesture
  start(): void; stop(): void
  readonly rateHz: number
  samplesBetween(t0: number, t1: number): ImuSample[]
  drain(): ImuSample[]                       // samples since last drain (for streaming to the worker)
  angularSpeed(t: number): number            // rad/s near time t
}
integrateGyro(samples: ImuSample[], t0: number, t1: number): Mat3   // R_{d(t1)←d(t0)} device-frame rotation
// orientation.ts
class CompassSensor { start(): void; stop(): void; headingAt(t: number): { heading: number; accuracy: number; north?: 'true' | 'magnetic' } | null
  setLocation(latDeg, lonDeg, altM?, date?): MagneticElements | null   // WMM declination ⇒ TRUE north from now on
  setLocationFromFix(fix, date?): boolean; setDeclination(deg | null): void; readonly northReference: 'true' | 'magnetic' }
// wmm.ts — World Magnetic Model WMM2025 (pure; official coefficients; valid 2025.0–2030.0)
magneticField(latDeg, lonDeg, altKm?, decimalYear?): MagneticElements   // D, I, X/Y/Z/H/F + rates, blackout flags
declination(latDeg, lonDeg, altKm?, decimalYear?): number               // deg, east positive: true = magnetic + D
decimalYearFromDate(d: Date | number): number
// geolocation.ts
class GeoSensor { start(): void; stop(): void; readonly fixes: GeoFix[]; nearest(t: number): GeoFix | null }
// frame-orientation.ts
screenAngle(): 0 | 90 | 180 | 270
cameraFromDevice(screenAngleDeg: number): Mat3   // R_cam_device for the rear camera given the UI rotation of the video frames
// wakelock.ts
class WakeLock { acquire(): Promise<void>; release(): Promise<void> }
```

### 4.8 `src/io`

```ts
encodePLY(pc: PointCloud): ArrayBuffer;  encodeMeshPLY(mesh: TriangleMesh): ArrayBuffer   // binary_little_endian
decodePLY(buf: ArrayBuffer): { points?: PointCloud; mesh?: TriangleMesh }
encodeGLB(input: { mesh?: TriangleMesh; points?: PointCloud; yUp?: boolean }): ArrayBuffer
interface ColmapModelInput { cameras: { id: number; cam: CameraIntrinsics }[];
  images: { id: number; cameraId: number; name: string; pose: Pose; points2D: { x: number; y: number; point3D: number }[] }[];
  points: { id: number; xyz: ArrayLike<number>; color: ArrayLike<number>; error: number; track: { image: number; point2D: number }[] }[] }
exportColmapText(model: ColmapModelInput): { name: string; data: string }[]   // cameras.txt, images.txt, points3D.txt
createZip(files: { name: string; data: Uint8Array | string }[]): Uint8Array   // STORE, CRC32, UTF-8
class SessionStore { static open(): Promise<SessionStore>; … IndexedDB sessions, frame JPEG blobs, JSON & binary blobs }
saveFile(data: Blob, filename: string): Promise<'shared' | 'downloaded'>    // Web Share (files) or <a download>
// recording.ts — raw capture recordings (developer option; docs/ENGINE.md §5, e2e/README.md)
interface RecordedFrame { seq: number; sentAt: number; input: Omit<FrameInput, 'bitmap'> /* verbatim */;
  image: { file: string; width: number; height: number; srcWidth: number; srcHeight: number;
           kind: 'VideoFrame' | 'ImageBitmap' | 'RGBAImage'; bytes: number } | null /* engine got it, pixels not recorded */;
  error?: string; notSent?: boolean /* the engine never got it */ }
encodeFrameRow(r: RecordedFrame): string; decodeFrameRow(line: string): RecordedFrame   // frames.jsonl, exact (JSON-safe)
type RecordedSensorRow = { k: 'motion' | 'orientation' | 'geo'; t: number; … }          // sensors.jsonl, raw readings
buildDatasetFiles(header, framesJsonl, liveJsonl, { stripLocation?, sensorsText? }): { meta, texts, images }   // e2e dataset, no GT
planZipParts(entries: { name; size }[], maxBytes = 200e6): { part; names; bytes }[]
// recording-store.ts
class RecordingStore { static open(); list(): Promise<StoredRecording[]>; reader(info); delete(info) }
prepareRecordingExport(store, info, { stripLocation? = true, maxPartBytes? }): Promise<{ meta; parts: { fileName; bytes; build(): Promise<Blob> }[] }>
```

### 4.9 `src/viewer`

```ts
class Viewer {
  constructor(container: HTMLElement, opts?: { background?: string })
  setPointCloud(pc: PointCloud | null, opts?: { pointSizePx?: number; splats?: boolean }): void
  setSparsePoints(positions: Float32Array, colors: Uint8Array): void
  setMesh(mesh: TriangleMesh | null): void
  setCameras(cams: { pose: Pose; intrinsics: CameraIntrinsics }[], highlightLast?: boolean): void
  setWorldUp(up: 'z' | 'auto'): void
  setDisplay(opts: { points?: boolean; mesh?: boolean; cameras?: boolean; edl?: boolean; grid?: boolean }): void
  fitToContent(): void
  startMeasure(onResult: (distance: number, a: Vec3, b: Vec3) => void): void; stopMeasure(): void
  setUnits(metric: boolean): void
  resize(): void; dispose(): void; screenshot(): Promise<Blob>
}
```

### 4.10 Contract deviations

Where the code legitimately differs from the contracts above. Optional parameters, extra fields
and extra methods that keep the documented call working are not listed.

* **File lists.** There are more files than listed. `src/ba` also has `camera-model.ts`, `linear.ts`
  and `priors.ts`, plus `synthetic.ts` for tests and the benchmark. `src/features` also has
  `sift-common.ts`, `gpu-common.ts`, `sift-gpu-host.ts`, `match-gpu-host.ts`, `simd.ts` (WASM
  SIMD kernels) and `testdata.ts`. `src/geometry` also has `eigen.ts` and `internal.ts`. WGSL
  lives in `.ts` modules (`features/shaders/*.ts`, `dense/patchmatch-wgsl.ts`), never in `.wgsl`
  files.
* **`src/features/index.ts`** re-exports only the CPU modules. `sift-gpu.ts` and `match-gpu.ts`
  read `GPUBufferUsage` at module load, which throws where WebGPU does not exist, so import them
  and `extractor.ts` directly. The engine loads `extractor.ts` lazily.
* **`saveFile`** can also resolve `'cancelled'`, when the user dismisses the share sheet.
  Callers must not report that as a saved file.
* **`tsdfMesh` `truncation`** is in voxels (default 4), not in world units like `voxelSize`.
* **`BAOptions.linearSolver: 'auto'`** also picks the dense Schur solver above 2400 reduced
  parameters when the Cholesky envelope is small, as it is for sequential captures.

## 5. Pipeline

**Live (during capture)** — `engine/` in a worker:
1. Main thread: `requestVideoFrameCallback` → at ≤ 10 Hz, if the worker is idle, grab the
   `<video>` frame (`new VideoFrame(video)` by default on iOS WebKit, else
   `createImageBitmap(video)`; `src/sensors/frame-pump.ts`) + IMU samples since last frame →
   worker (transfer).
2. Worker: rotate to the session's canonical orientation, compute a ≤ 640 px gray
   copy → sharpness + gyro angular speed → reject blurry frames.
3. Extract RootSIFT on GPU at the working resolution, match against the last
   keyframe (gyro-predicted guided matching), measure overlap and
   rotation-compensated parallax → keyframe decision (`sfm/keyframes.ts`).
4. Keyframe: store JPEG (IndexedDB), add to `FeatureDatabase`, match against the
   last N keyframes + retrieval candidates, verify (two-view geometry), update the
   correspondence graph, then `LiveMapper` initialises / registers (PnP) /
   triangulates / local BA. Emits live state: poses, sparse points, tracking
   quality, user guidance.

**Refine (after capture)**: extra pairs (sequential window, vocabulary-tree
retrieval for loop closure, spatial neighbours) → verify (pairs and vocabulary on CPU helper
workers) → refined live model → only when it is unhealthy, a full incremental SfM from scratch
on the enriched graph (keep the better of live/offline model) → global
BA with self-calibration → retriangulate → final BA → inertial alignment (time
offset, gravity, metric scale), GPS Sim3 alignment, compass yaw → world frame per §2
(timings: docs/research/refine-performance.md).

**Dense**: undistort keyframes (≤ 1600 px by preset) → source-view selection →
PatchMatch at ½ resolution (multi-scale) → PatchMatch (photometric) for all views →
PatchMatch (geometric consistency) → halo / sky filter → fusion → filtered coloured
point cloud → optional TSDF mesh (object mode with a gravity-aligned model: bounded to the automatic
object box from the sparse model — orbit centre + support plane, `dense/object-roi.ts` — which the mesh
result and `mesh/meta` carry for later ground-placed exports; the dense cloud and splats are not cropped).

**Splat**: dense (else sparse) cloud → 3D Gaussians → WebGPU training against all keyframes
(MCMC densification, depth supervision from the dense maps, per-view colour correction).
