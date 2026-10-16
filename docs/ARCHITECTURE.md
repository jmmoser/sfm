# Architecture & module contracts

This document is the single source of truth for how the photogrammetry app is
structured. Every module is implemented against the contracts below. If you need
to deviate from a signature, keep the documented one working (add, don't break)
and note the change in your module's README section at the bottom of this file.

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
| Time | Milliseconds, `performance.now()` domain. |
| Device frame (W3C DeviceMotion) | +x right edge, +y top edge, +z out of screen (portrait natural orientation). |
| Camera ← device rotation | Rear camera, processed image upright for the UI rotated by screen angle θ (device turned θ counter-clockwise from natural portrait as seen by the user; θ ∈ {0, 90, 180, 270}): `R_cam_device(θ) = diag(1, −1, −1) · Rz(θ)` where `Rz(θ) = [[cosθ, −sinθ, 0], [sinθ, cosθ, 0], [0, 0, 1]]`. θ=0: `diag(1,−1,−1)`; θ=90: rows `(0,−1,0), (−1,0,0), (0,0,−1)`. `v_cam = R_cam_device · v_device`. (iOS mapping of `screen.orientation.angle` / `window.orientation` to θ is verified in `src/sensors/frame-orientation.ts`.) |
| Device attitude | `R_wd` = device-to-world rotation; camera pose rotation `R = R_cam_device · R_wdᵀ`. Gyro `ω_dev`: `dR_wd/dt = R_wd [ω_dev]×`. `accel_dev = R_wdᵀ a_world`, `accelG_dev = R_wdᵀ (a_world − g_world)`, `g_world = (0, 0, −9.81)` in the Z-up world. |
| Gravity | `ImuSample.accelG` uses the W3C spec sign: device lying flat face-up reports ≈ (0, 0, +9.81) (specific force). "Gravity direction" vectors in this code base always point DOWN. |
| World frame after alignment | +Z up (opposite gravity). If north is known: +Y = true north, +X = east (local ENU). Units = metres when `MetricInfo.metric`. |
| Viewer | three.js is Y-up; the viewer converts from the Z-up world (`(x, y, z) → (x, z, -y)`). |
| Random numbers | Always through the seeded `Rng` from `src/math/random.ts` (deterministic tests). |
| Descriptors | RootSIFT, 128 bytes, `min(255, round(512 * v))` of the unit-L2 RootSIFT vector. |

Shared types live in `src/core/types.ts` (do not duplicate them).

### 2.1 WebGPU rules for iOS Safari (from docs/research/ios-webgpu.md)

* At most ONE heavy command buffer in flight: submit, then `await device.queue.onSubmittedWorkDone()`
  (`submitAndWait` in `src/gpu/device.ts`) before the next. Size command buffers to ~2–30 ms of GPU
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
  sensors/               camera capture, motion, compass, geolocation, wake lock, frame orientation
  io/                    PLY / GLB / COLMAP export, ZIP writer, IndexedDB session storage, share/download
  viewer/                three.js viewer (points as splats + EDL, mesh, cameras, measure tool)
  engine/                PhotogrammetryEngine (pipeline orchestration, runs in a worker)
  workers/               engine.worker.ts + typed message protocol
  app/                   mobile UI (vanilla TS + CSS), screens, state machine
  harness/               test harness page logic (feeds synthetic datasets to the engine)
  main.ts                app entry
e2e/                     synthetic dataset generator, end-to-end runner, metrics
docs/                    this file, research notes
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
`homography.ts`, `pnp.ts`, `triangulation.ts`, `twoview.ts`, `align.ts`, `index.ts`.

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

### 4.6 `src/dense`  (depends on math, core, gpu)

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
tsdfMesh(views: DenseView[], store: DepthMapStore, opts?: { voxelSize?: number; truncation?: number; maxDepth?: number },
  progress?, cancel?): Promise<TriangleMesh>
filterPointCloud(pc: PointCloud, opts?: { voxelSize?: number; statistical?: { k: number; std: number } }): PointCloud
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
class CompassSensor { start(): void; stop(): void; headingAt(t: number): { heading: number; accuracy: number } | null }
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

## 5. Pipeline

**Live (during capture)** — `engine/` in a worker:
1. Main thread: `requestVideoFrameCallback` → at ≤ 10 Hz, if the worker is idle,
   `createImageBitmap(video)` + IMU samples since last frame → worker (transfer).
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
retrieval for loop closure, spatial neighbours) → verify → full incremental SfM
from scratch on the enriched graph (keep the better of live/offline model) → global
BA with self-calibration → retriangulate → final BA → inertial alignment (time
offset, gravity, metric scale), GPS Sim3 alignment, compass yaw → world frame per §2.

**Dense**: undistort keyframes (≤ 1600 px by preset) → source-view selection →
PatchMatch (photometric) for all views → PatchMatch (geometric consistency) →
fusion → filtered coloured point cloud → optional TSDF mesh.

## 6. Module notes
(Authors append short notes here: design decisions, deviations, known limits.)

### `src/io` (export, persistence, sharing)

* **API superset of §4.8** (all documented signatures kept): `decodePLY` also accepts
  `Uint8Array` and returns header `comments`; `encodeGLB` adds `colorFormat`,
  `unlitMesh`, `doubleSided`, `generator`, `extras`; `createZip(files, opts?)` adds
  `lastModified`/`comment`; `saveFile` returns `'shared' | 'downloaded' | 'cancelled'`
  (`'cancelled'` = user dismissed the iOS share sheet, AbortError). Extras:
  `createZipBlob` (Blob parts — frame Blobs from IndexedDB are streamed once for the
  CRC and never copied into one buffer; use this for exports with images on iOS),
  `readZip`, `crc32`, `parseGLB`, `importColmapText`, `createColmapZip[Blob]`
  (`sparse/0/*.txt` + `images/<name>`), binary serialisers and `SessionStore` extras
  (`deleteFrame`, `listBlobKeys`, `listJSONKeys`, `getArrayBuffer`, `sessionBytes`).
* **PLY**: binary_little_endian writer (`x y z [nx ny nz] red green blue [radius]
  [confidence]`; meshes `list uchar int vertex_indices`). Reader: ascii / binary LE / BE,
  any scalar types, fan-triangulates polygons, computes normals for meshes without them,
  missing colour → grey 200.
* **GLB**: COLOR_0 = FLOAT VEC3 **linear** (sRGB → linear via the exact EOTF; glTF
  defines vertex colours as linear) by default — universally supported, no 8-bit-linear
  banding; `'unorm16'`/`'unorm8'` (normalized VEC4) for smaller files. Y-up conversion
  `(x,y,z) → (x,z,−y)` on by default (glTF is Y-up). Points: mode 0 + KHR_materials_unlit
  (in `extensionsUsed` only). Mesh material double-sided, rough, non-metallic.
* **COLMAP**: RADIAL, `cx+0.5, cy+0.5`, POINTS2D `+0.5`; image names must not contain
  whitespace (COLMAP's text reader splits on spaces). Ids are written as given (callers
  may use 0-based ids; COLMAP accepts them).
* **serialize**: container `'SFMB'` + kind + version; arrays carry a dtype so precision
  can change without a version bump. Decoders copy by default (each array gets its own
  ArrayBuffer — safe to transfer individually); `{ zeroCopy: true }` returns views.
  `float16: true` stores normals/confidence/cost as binary16 (manual, correctly rounded).
  `DepthMapLike` mirrors `dense/DepthMap` structurally.
* **SessionStore**: DB `sfm` v1, stores `sessions`, `frames` (info), `frameData` (JPEG),
  `blobs`, `json`; compound keys `[sessionId, key]`, cascade delete = one transaction with
  range deletes. Promises resolve on transaction *commit*. `putFrame` requires an existing
  session and keeps `numFrames`/`updatedAt` in the same transaction. QuotaExceededError →
  `StorageQuotaError` (transaction aborted, nothing partially written). Timestamps are
  `Date.now()` (persisted across loads), not `performance.now()`. Works in workers
  (`requestPersistence()` returns false there — call it from the page after a gesture).
* **saveFile**: Web Share needs transient user activation on iOS — prepare the Blob first,
  then call `saveFile` directly in the tap handler; an expired activation (NotAllowedError)
  falls back to `<a download>`.

### `src/math` (linear algebra, polynomials, rotations, RNG, stats, Sim3)

* **API superset of §4.1** (all documented signatures kept). Extras: `vec3AddScaled`,
  `vec3NormSq`, `vec3DistSq`, `vec3Set`, `vec3Lerp`, `vec3Angle`; `mat3FromValues`,
  `mat3FromColumns`, `mat3GetColumn/Row`, `mat3Sub`, `mat3Trace`, `mat3Outer`;
  `Matrix.zeros`, `transposeMul`, `transposeMulVec`, `frobeniusNorm`, `getRow/Column`;
  `nullSpace(A, dim)` (n x dim orthonormal basis, column 0 = smallest σ — use it for
  5-point / 7-point null spaces), `solveLeastSquares(A, b)` (Householder QR);
  `polyDerivative`, `polyMul`, `polyFromRoots`, `polyEvalErrorBound`,
  `polyRealRootsSturm`; `quatRotateVec`; `Rng.nextU32/nextDouble/uniform/seed`;
  `variance`, `rms`; `Sim3` type, `sim3Inverse`, `sim3Compose`.
* `new Matrix(r, c, f64)` WRAPS a Float64Array (no copy); other array-likes are copied.
* **SVD**: one-sided Jacobi (Hestenes) with Householder-QR preconditioning when m > n;
  V = accumulated rotations (orthonormal to ~1e-15 even for rank-deficient input).
  Singular values ≤ ε‖A‖_F are returned as exactly 0 and their U columns completed.
  Non-finite input → NaN outputs (no throw). `nullVector`/`nullSpace` for m < n with
  dim ≤ n − m use the trailing columns of the full Q of Aᵀ = QR (exact null space,
  ~2 µs for 8x9). `svd3`: U, V orthogonal but det may be −1 (use `orthonormalize`).
  Timings (Node/V8, x86): nullVector 8x9 ≈ 2 µs, svd 9x9 ≈ 20–25 µs, svd 100x12 ≈ 130 µs,
  svd3 ≈ 2.5 µs, symEig 12x12 ≈ 35 µs. Scratch buffers are module-level (sync code only).
* **cholesky** rejects pivots ≤ 1e-15·(original diagonal) (semi-definite / gauge-
  deficient systems return false instead of garbage); zeroes the strict upper triangle.
  **luSolve** returns null if a pivot ≤ 4nε·max|original row|. **mat3Inv** returns null
  if |det| ≤ 1e-14·∏‖rows‖ (scale-safe).
* **Polynomials — deviation from the "Sturm" note in §4.1**: `polyRealRoots` isolates
  roots by the critical points (real roots of p′, recursively; p is monotone between
  them) and refines each bracket with safeguarded Newton (rtsafe) after exact
  power-of-two scaling of x into [−1, 1]. In stress tests (40k random polynomials of
  degree 3–20, coefficients spanning 1e±4) it had 0 missed / 0 spurious roots, while
  the floating-point Sturm variant (still available as `polyRealRootsSturm`) had 1+1,
  and Cardano/Ferrari closed forms failed on ~2 % of badly scaled cubics/quartics.
  `solveCubic`/`solveQuartic` therefore delegate to the same robust routine
  (≈1.4 / 2 µs). Roots are DISTINCT, ascending; multiple roots / clusters that are
  indistinguishable at the rounding-noise level are reported once.
* **Rotations**: `mat3ToRotvec` uses atan2(|vee(R−Rᵀ)|/2, (trR−1)/2) and the symmetric
  part near π (θ > 2π/3); `rotationBetweenVectors` is O(ε) accurate also for nearly
  antiparallel inputs. `quatSlerp` takes the shortest arc.
* **Rng**: sfc32 seeded via splitmix32; default seed fixed (0x2545F491). `sample` returns
  a uniformly random k-subset; the ORDER is not uniformly random for small k (Floyd).
* **stats**: inputs must be NaN-free. `median`/`percentile`/`mad` copy into a reused
  scratch buffer (temporary copy above 2¹⁸ elements); `mad` is unscaled.
* **umeyama**: weights < 0 / NaN count as 0; null if < 3 positive-weight points, zero
  spread in src or dst, or cross-covariance rank < 2 (σ₂ ≤ 1e-12·√(var_src·var_dst),
  i.e. collinear to ~1e-6 aspect ratio). Two-pass centring (accurate at ENU offsets).

### `src/features` — CPU part (image utils, SIFT, distribution, matching)

* **API superset of §4.4** (documented signatures kept). Extras: `grayToFloat32`,
  `resizeFloat32`, `blurFloat32`, `gaussianHalfKernel` (image.ts); `DEFAULT_SIFT_OPTIONS`,
  `resolveSiftOptions`, `planSift`, `octaveToFull`, `finalizeRootSift` (sift-common.ts —
  shared with sift-gpu so both produce identical frames/layouts); `releaseSiftCPUMemory`,
  `lastSiftCPUTimings` (sift-cpu.ts); `selectFeatureIndices`, `selectFeatures`,
  `subsetFeatureSet`, `sortByScoreDesc`, `gridDims` (distribute.ts — use `selectFeatures`
  after GPU extraction); `descriptorInvNorms`, `descriptorDot`, `descriptorDistance`,
  `unitDescriptors`, `matchUsesSIMD` (match.ts); `simdAvailable` (simd.ts).
* **image.ts**: every function accepts Uint8 (0..255) or Float32 (0..1) gray and returns
  **Float32 0..1**. `sharpness` = Laplacian variance on the **0..255 scale** (OpenCV
  convention) of a ≤ 640 px area-downscaled copy. `rotateRGBA90(img, k)` rotates
  **clockwise** (as displayed) for k > 0: `dst(x', y') = src(y', H−1−x')`.
* **SIFT conventions** (header of sift-common.ts is normative for sift-gpu): octave o
  pixel grid `x_proc = x_oct · 2^o` (octave −1 = upsample with up(2i)=in(i),
  up(2i+1)=mean(in(i), in(i+1)); next octave = decimation of level S); Gaussian level i
  has σ·2^(i/S); a keypoint at continuous DoG level s has scale σ·2^(s/S) and uses
  Gaussian level clamp(round(s),1,S) for orientation/descriptor. Reported `scale` follows
  the Lowe/VLFeat convention (≈ 0.89 × σ of an ideal Gaussian blob). Descriptor layout
  `(by·4 + bx)·8 + bo`, bins along x' = (cos θ, sin θ), y' = (−sin θ, cos θ), orientation
  bin bo centred at bo·45° of (gradient angle − θ). `firstOctave`: −1/0 as documented;
  positive values are folded into an extra area downscale. **Extension**
  `upsampleBelow` (default 800): with firstOctave 0, processed images whose long side is
  below it use octave −1 (set 0 to disable).
* **Selection**: `maxFeatures` counts *oriented* features. Distribution grid ≈
  maxFeatures/16 cells (aspect-matched), 50 % of the budget as per-cell quotas of the
  strongest responses, the rest filled globally by response; applied to keypoints before
  orientation (descriptors are only computed for survivors) and to oriented features at the
  end. Output sorted by descending score (|DoG| at the refined extremum).
* **Matching**: similarity = exact integer dot of the quantised bytes × cached inverse
  norms (mathematically = unit-vector dot; bit-identical between JS and WASM paths).
  Mutual check has no reverse ratio test; ties → smaller index. Guides: `F` uses
  x_bᵀ F x_a = 0 (band walked through a grid over b); `H` predictions with w ≤ 0 never
  match; `positions` NaN = **unconstrained** (searched against all of b).
* **WASM SIMD** (simd.ts, hand-assembled, 3.4 KB — keep < 4 KB so Chrome allows sync
  compile on a main thread): separable blur (f32x4), gradient magnitude/atan2 (f32x4),
  descriptor accumulation (scalar), brute-force dots (i32x4.dot_i16x8_s). Falls back to
  plain JS when unavailable or when `globalThis.SFM_DISABLE_SIMD = true`. SIFT planes live
  in a reusable arena (WASM memory): ≈ 110 MB for a 1600×1200 (or < 800 px upsampled)
  image — call `releaseSiftCPUMemory()` when done; use it from a worker.
* **Performance** (Node 22, x86, single thread): extraction 1600×1200 → 4096 features
  ≈ 420 ms (JS-only ≈ 1.4 s); brute-force matching 4000×4000 ≈ 180 ms (JS-only ≈ 2.4 s);
  guided positions (r = 20 px) ≈ 5–15 ms, guided F (3 px) ≈ 30–45 ms.
* **Accuracy** (tests): Gaussian-blob localisation bias < 0.03 px in the full-res frame
  for all processing paths (internal/external downscale, octave −1/0); precision ≥ 0.957
  and repeatability 0.55–0.99 under rotation 30/90°, scale 0.6/1.5, tilt 40°, photometric
  change and noise. `testdata.ts` (browser-safe) provides procedural textures, homography
  warps with anti-aliasing, photometric changes and evaluation helpers for the GPU tests.
